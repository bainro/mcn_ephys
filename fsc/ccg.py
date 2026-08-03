#!/usr/bin/env python3
"""
ccg.py

Clean compact CCG module for FSC.

This module contains only the new compact Numba CCG path. It intentionally does
not import the old Cython CCGFast path, so downstream FSC code can depend on the
new implementation without pulling in the legacy CCG implementation.

Direction convention
--------------------
For compact pair p = (i, j), where i <= j:

    pair_ccg[p, positive_lag_bins] describes i -> j timing.
    pair_ccg[p, negative_lag_bins] describes j -> i timing.

Autocorrelograms keep both positive and negative lag counts.
"""

from __future__ import annotations

import os
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads


def make_pair_metadata(n_units: int):
    """
    Build metadata for compact upper-triangle CCG storage.
    """
    n_pairs = n_units * (n_units + 1) // 2

    pair_first = np.empty(n_pairs, dtype=np.int32)
    pair_second = np.empty(n_pairs, dtype=np.int32)
    pair_lookup = np.empty((n_units, n_units), dtype=np.int32)
    pair_sign = np.empty((n_units, n_units), dtype=np.int8)

    pair_index = 0
    for i in range(n_units):
        for j in range(i, n_units):
            pair_first[pair_index] = i
            pair_second[pair_index] = j

            pair_lookup[i, j] = pair_index
            pair_lookup[j, i] = pair_index

            pair_sign[i, j] = 1
            pair_sign[j, i] = -1
            pair_sign[i, i] = 1

            pair_index += 1

    return pair_first, pair_second, pair_lookup, pair_sign


@njit(parallel=True, cache=True)
def _compute_compact_ccg_blocks(
    spike_times_s,
    unit_ids,
    end_index,
    pair_lookup,
    pair_sign,
    n_units,
    bin_size,
    half_bins,
    max_dt,
    n_blocks,
):
    """
    Compute compact all-pairs CCGs using one private output block per worker chunk.
    """
    n_spikes = spike_times_s.shape[0]
    n_bins = 2 * half_bins + 1
    n_pairs = n_units * (n_units + 1) // 2
    inv_bin_size = 1.0 / bin_size

    blocks = np.zeros((n_blocks, n_pairs, n_bins), dtype=np.uint32)

    for block in prange(n_blocks):
        start_spike = (n_spikes * block) // n_blocks
        stop_spike = (n_spikes * (block + 1)) // n_blocks
        local_counts = blocks[block]

        for spike_index in range(start_spike, stop_spike):
            t0 = spike_times_s[spike_index]
            unit0 = unit_ids[spike_index]

            for next_index in range(spike_index + 1, end_index[spike_index]):
                dt = spike_times_s[next_index] - t0
                if dt > max_dt:
                    break

                unit1 = unit_ids[next_index]
                pair_index = pair_lookup[unit0, unit1]
                dt_bins = dt * inv_bin_size

                if unit0 != unit1:
                    signed_dt_bins = pair_sign[unit0, unit1] * dt_bins
                    bin_index = half_bins + int(np.floor(0.5 + signed_dt_bins))

                    if 0 <= bin_index < n_bins:
                        local_counts[pair_index, bin_index] += 1
                else:
                    positive_bin = half_bins + int(np.floor(0.5 + dt_bins))
                    negative_bin = half_bins + int(np.floor(0.5 - dt_bins))

                    if 0 <= positive_bin < n_bins:
                        local_counts[pair_index, positive_bin] += 1
                    if 0 <= negative_bin < n_bins:
                        local_counts[pair_index, negative_bin] += 1

    return blocks


def dense_unit_ids_from_cluster_ids(cluster_ids):
    """
    Convert arbitrary cluster IDs to dense 0-indexed unit IDs.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    dense_cluster_ids, unit_ids = np.unique(cluster_ids, return_inverse=True)
    return unit_ids.astype(np.int32, copy=False), dense_cluster_ids


def load_kilosort_spikes(data_dir, sample_rate, keep_clusters=None):
    """
    Load Kilosort spike_times.npy and spike_clusters.npy.
    """
    spike_times = np.load(os.path.join(data_dir, "spike_times.npy")).reshape(-1)
    spike_clusters = np.load(os.path.join(data_dir, "spike_clusters.npy")).reshape(-1)

    spike_times = spike_times.astype(np.int64, copy=False)
    spike_clusters = spike_clusters.astype(np.int64, copy=False)

    if keep_clusters is not None:
        keep_clusters = np.asarray(keep_clusters, dtype=np.int64)
        keep_mask = np.isin(spike_clusters, keep_clusters)
        spike_times = spike_times[keep_mask]
        spike_clusters = spike_clusters[keep_mask]

    cluster_ids, unit_ids = np.unique(spike_clusters, return_inverse=True)
    unit_ids = unit_ids.astype(np.int32, copy=False)

    sort_order = np.argsort(spike_times, kind="mergesort")
    spike_times = spike_times[sort_order]
    unit_ids = unit_ids[sort_order]

    spike_times_s = spike_times.astype(np.float64) / float(sample_rate)
    return spike_times_s, unit_ids, cluster_ids


def compute_compact_ccg_from_arrays(
    spike_times_s,
    unit_ids,
    n_units=None,
    bin_size=0.0004,
    duration=0.05,
    n_threads=8,
    n_blocks=None,
):
    """
    Compute compact CCGs from spike times in seconds and dense 0-indexed unit IDs.
    """
    set_num_threads(int(n_threads))
    if n_blocks is None:
        n_blocks = max(1, get_num_threads() * 2)

    spike_times_s = np.asarray(spike_times_s, dtype=np.float64).reshape(-1)
    unit_ids = np.asarray(unit_ids, dtype=np.int32).reshape(-1)

    if spike_times_s.shape[0] != unit_ids.shape[0]:
        raise ValueError("spike_times_s and unit_ids must have the same length.")

    if n_units is None:
        n_units = int(np.max(unit_ids)) + 1 if unit_ids.size else 0
    n_units = int(n_units)

    if unit_ids.size and (np.min(unit_ids) < 0 or np.max(unit_ids) >= n_units):
        raise ValueError("unit_ids must be dense 0-indexed IDs in [0, n_units).")

    sort_order = np.argsort(spike_times_s, kind="mergesort")
    spike_times_s = spike_times_s[sort_order]
    unit_ids = unit_ids[sort_order].astype(np.int32, copy=False)

    half_bins = int(np.ceil(duration / bin_size / 2.0))
    n_bins = 2 * half_bins + 1
    max_dt = bin_size * (half_bins + 0.5)

    end_index = np.searchsorted(spike_times_s, spike_times_s + max_dt, side="right")
    end_index = end_index.astype(np.int64, copy=False)

    pair_first, pair_second, pair_lookup, pair_sign = make_pair_metadata(n_units)
    lags = np.arange(-half_bins, half_bins + 1, dtype=np.float64) * float(bin_size)

    start_time = time.perf_counter()

    partial_counts = _compute_compact_ccg_blocks(
        spike_times_s,
        unit_ids,
        end_index,
        pair_lookup,
        pair_sign,
        n_units,
        float(bin_size),
        int(half_bins),
        float(max_dt),
        int(n_blocks),
    )

    pair_ccg = partial_counts.sum(axis=0, dtype=np.uint64)
    elapsed = time.perf_counter() - start_time

    return {
        "pair_ccg": pair_ccg,
        "lags": lags,
        "pair_first": pair_first,
        "pair_second": pair_second,
        "elapsed": elapsed,
        "n_threads": int(n_threads),
        "n_blocks": int(n_blocks),
        "bin_size": float(bin_size),
        "duration": float(duration),
    }


def compute_compact_ccg(
    data_dir,
    sample_rate=30000.0,
    bin_size=0.0004,
    duration=0.05,
    n_threads=8,
    n_blocks=None,
    keep_clusters=None,
    make_full_validation=False,
):
    """
    Compute compact CCGs from a Kilosort folder.
    """
    spike_times_s, unit_ids, cluster_ids = load_kilosort_spikes(
        data_dir=data_dir,
        sample_rate=sample_rate,
        keep_clusters=keep_clusters,
    )

    result = compute_compact_ccg_from_arrays(
        spike_times_s=spike_times_s,
        unit_ids=unit_ids,
        n_units=int(cluster_ids.size),
        bin_size=bin_size,
        duration=duration,
        n_threads=n_threads,
        n_blocks=n_blocks,
    )

    result["cluster_ids"] = cluster_ids
    result["sample_rate"] = float(sample_rate)

    if make_full_validation:
        result["full_cch_validation"] = compact_pairs_to_full_cch(
            result["pair_ccg"],
            result["pair_first"],
            result["pair_second"],
            int(cluster_ids.size),
        )
    else:
        result["full_cch_validation"] = None

    return result


def compact_pairs_to_full_cch(pair_ccg, pair_first, pair_second, n_units):
    """
    Reconstruct a full directed CCG tensor from compact pair storage.
    """
    n_pairs, n_bins = pair_ccg.shape
    cch = np.zeros((n_bins, n_units, n_units), dtype=pair_ccg.dtype)

    for pair_index in range(n_pairs):
        unit_i = int(pair_first[pair_index])
        unit_j = int(pair_second[pair_index])
        ccg = pair_ccg[pair_index]

        cch[:, unit_i, unit_j] = ccg
        if unit_i != unit_j:
            cch[:, unit_j, unit_i] = ccg[::-1]

    return cch
