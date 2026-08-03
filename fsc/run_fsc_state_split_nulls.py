#!/usr/bin/env python3
"""
run_fsc_state_split_nulls.py

State-split FSC sniper analysis.

Hypothesis:
  FSC estimated over mixed behavior may blur state-specific functional coupling.
  If the true split is 8 hr running vs 8 hr SWS, then running two FSCs
  separately and taking the elementwise max weight should produce a larger
  score than random duration-matched 8 hr / 8 hr splits.

This script:
  1. Loads a merged KS folder with UnitMetrics.csv good == TRUE.
  2. Uses the same compact CCG/FSC core as the current pipeline.
  3. Runs FSC on:
       - true split: 0-8 hr running, 8-16 hr SWS
       - N random splits: 96 ten-minute blocks assigned 48/48 to A/B
  4. For each split, computes:
       W = max(abs(strengthexc), abs(strengthinh))
       R = max(abs(ratioexc), abs(ratioinh))
       W_max = max(W_A, W_B) elementwise
       delta_W = abs(W_A - W_B)
  5. Saves summary CSV, union-detected per-pair CSV, optional compressed matrices,
     and exploratory plots.

Place this file in the same folder as ccg.py/fsc.py, or edit CCG_MODULE_PATH and
FSC_MODULE_PATH below. It can also load files named ccg(4).py and fsc(1).py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


# -----------------------------
# EDIT THESE SETTINGS
# -----------------------------

KS_DIR = Path(r"G:\\merged_16hr_SWS_run_CONNECTS")
OUT_DIR = Path(r"G:\\merged_16hr_SWS_run_CONNECTS")
ANIMAL_ID = "ms17"

SAMPLE_RATE = 30000.0
REFRACTORY_S = 0.002
THREADS = 8
BLOCKS_FOR_CCG = 24

TOTAL_DURATION_H = 16.0
STATE_BOUNDARY_H = 8.0
BLOCK_MIN = 10.0
N_RANDOM_SPLITS = 1000
RANDOM_SEED = 0

RUNNING_LABEL = "running"
SWS_LABEL = "SWS"

# Save compressed float32 weight/ratio matrices for every split so scores can be
# recomputed without rerunning FSC. Turn off if the unit count is huge.
SAVE_WEIGHT_MATRICES_NPZ = True

# Dynamic import paths. Leave None to auto-detect nearby ccg.py/fsc.py or
# ccg(4).py/fsc(1).py.
CCG_MODULE_PATH = None
FSC_MODULE_PATH = None


# -----------------------------
# Dynamic imports
# -----------------------------
def _import_module_from_path(name: str, path: Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name} module path: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _find_module_file(name: str, explicit_path, candidates):
    if explicit_path is not None:
        return Path(explicit_path)

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    search_dirs = [here, cwd, Path("/mnt/data")]
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Could not auto-find {name}. Tried candidates {candidates} in {search_dirs}. "
        f"Edit {name.upper()}_MODULE_PATH at the top of this script."
    )


def load_fsc_modules():
    ccg_path = _find_module_file("ccg", CCG_MODULE_PATH, ["ccg.py", "ccg(4).py"])
    ccg_mod = _import_module_from_path("ccg", ccg_path)

    fsc_path = _find_module_file("fsc", FSC_MODULE_PATH, ["fsc.py", "fsc(1).py"])
    fsc_mod = _import_module_from_path("fsc", fsc_path)

    print(f"Using ccg module: {ccg_path}")
    print(f"Using fsc module: {fsc_path}")
    return ccg_mod, fsc_mod


# -----------------------------
# Loading
# -----------------------------
def _good_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    if np.issubdtype(series.dtype, np.number):
        return (series.to_numpy(dtype=float) > 0)
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin({"true", "t", "1", "yes", "y", "good"}).to_numpy(dtype=bool)


def load_good_cluster_ids(ks_dir: Path):
    unit_path = ks_dir / "UnitMetrics.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing UnitMetrics.csv in {ks_dir}")

    df = pd.read_csv(unit_path)
    if "cluster_id" not in df.columns or "good" not in df.columns:
        raise ValueError(f"UnitMetrics.csv must have cluster_id and good columns. Columns: {list(df.columns)}")

    keep = _good_mask(df["good"])
    good_ids = np.unique(df.loc[keep, "cluster_id"].astype(np.int64).to_numpy())
    if good_ids.size == 0:
        raise ValueError(f"UnitMetrics.csv has zero good units in {ks_dir}")

    return good_ids, df.loc[keep].copy()


def load_kilosort_spikes(ks_dir: Path, sample_rate: float, refractory_s: float):
    spike_times = np.load(ks_dir / "spike_times.npy").reshape(-1).astype(np.int64, copy=False)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").reshape(-1).astype(np.int64, copy=False)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times.npy and spike_clusters.npy have different lengths.")

    good_ids, unit_table = load_good_cluster_ids(ks_dir)
    keep_good = np.isin(spike_clusters, good_ids)
    spike_times = spike_times[keep_good]
    spike_clusters = spike_clusters[keep_good]

    if spike_times.size > 1 and np.any(spike_times[1:] < spike_times[:-1]):
        order = np.argsort(spike_times, kind="mergesort")
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]

    cluster_ids, dense_unit_ids = np.unique(spike_clusters, return_inverse=True)
    dense_unit_ids = dense_unit_ids.astype(np.int32, copy=False)

    # Refractory/near-duplicate filtering matching the main pipeline logic.
    keep = np.zeros(spike_times.shape[0], dtype=bool)
    prev_by_unit = np.full(cluster_ids.size, -1, dtype=np.int64)
    for i, (t, u) in enumerate(zip(spike_times, dense_unit_ids)):
        prev = prev_by_unit[int(u)]
        if prev >= 0:
            dt_s = (float(t) - float(prev)) / float(sample_rate)
            if dt_s > float(refractory_s):
                keep[i] = True
        prev_by_unit[int(u)] = int(t)

    spike_times = spike_times[keep]
    dense_unit_ids = dense_unit_ids[keep].astype(np.int32, copy=False)
    spike_times_s = spike_times.astype(np.float64) / float(sample_rate)

    # Unit metadata aligned to dense order.
    dense_meta = pd.DataFrame({"dense_id": np.arange(cluster_ids.size, dtype=int), "cluster_id": cluster_ids.astype(int)})
    dense_meta = dense_meta.merge(unit_table, on="cluster_id", how="left", suffixes=("", "_unitmetrics"))

    return spike_times_s, dense_unit_ids, cluster_ids.astype(np.int64), dense_meta


def mask_for_block_set(spike_times_s, selected_blocks, block_s, analysis_start_s, total_s):
    t_rel = spike_times_s - float(analysis_start_s)
    in_analysis = (t_rel >= 0.0) & (t_rel < float(total_s))
    block_idx = np.floor(t_rel / float(block_s)).astype(np.int64)
    return in_analysis & np.isin(block_idx, np.asarray(selected_blocks, dtype=np.int64))


def make_true_and_random_splits(total_h, boundary_h, block_min, n_random, seed):
    total_s = float(total_h) * 3600.0
    boundary_s = float(boundary_h) * 3600.0
    block_s = float(block_min) * 60.0

    n_blocks = int(np.floor(total_s / block_s))
    if n_blocks <= 1:
        raise ValueError("Not enough blocks. Check TOTAL_DURATION_H and BLOCK_MIN.")

    blocks = np.arange(n_blocks, dtype=np.int64)
    labels = np.where((blocks + 0.5) * block_s < boundary_s, RUNNING_LABEL, SWS_LABEL)
    running_blocks = blocks[labels == RUNNING_LABEL]
    sws_blocks = blocks[labels == SWS_LABEL]

    if running_blocks.size != sws_blocks.size:
        print(
            f"WARNING: running and SWS block counts differ: "
            f"{running_blocks.size} vs {sws_blocks.size}. Random splits will be half/half by block count."
        )

    half_n = n_blocks // 2
    if n_blocks % 2 != 0:
        raise ValueError(f"Total number of blocks must be even for duration-matched A/B splits. Got {n_blocks}.")

    split_specs = [
        {
            "split_id": "true_running_vs_sws",
            "split_type": "true_state",
            "blocks_A": running_blocks.astype(int).tolist(),
            "blocks_B": sws_blocks.astype(int).tolist(),
            "label_A": RUNNING_LABEL,
            "label_B": SWS_LABEL,
        }
    ]

    rng = np.random.default_rng(int(seed))
    seen = {tuple(sorted(running_blocks.astype(int).tolist()))}
    for i in range(int(n_random)):
        # Avoid accidentally reproducing the true split exactly.
        for _attempt in range(10000):
            a = np.sort(rng.choice(blocks, size=half_n, replace=False)).astype(int)
            key = tuple(a.tolist())
            if key not in seen:
                seen.add(key)
                break
        b = np.setdiff1d(blocks, a, assume_unique=False).astype(int)
        split_specs.append(
            {
                "split_id": f"random_{i:03d}",
                "split_type": "random",
                "blocks_A": a.tolist(),
                "blocks_B": b.tolist(),
                "label_A": "random_A",
                "label_B": "random_B",
            }
        )

    block_table = pd.DataFrame(
        {
            "block_index": blocks.astype(int),
            "start_s": blocks.astype(float) * block_s,
            "end_s": (blocks.astype(float) + 1.0) * block_s,
            "state_label": labels,
        }
    )
    return split_specs, block_table, block_s, total_s


def split_state_counts(blocks_A, blocks_B, block_table):
    state_by_block = dict(zip(block_table["block_index"].astype(int), block_table["state_label"].astype(str)))

    def counts(blocks):
        vals = [state_by_block[int(b)] for b in blocks]
        n_run = int(sum(v == RUNNING_LABEL for v in vals))
        n_sws = int(sum(v == SWS_LABEL for v in vals))
        return n_run, n_sws

    run_A, sws_A = counts(blocks_A)
    run_B, sws_B = counts(blocks_B)
    frac_run_A = run_A / max(1, len(blocks_A))
    frac_run_B = run_B / max(1, len(blocks_B))
    sep = abs(frac_run_A - frac_run_B)

    ratio_A = (sws_A + 0.5) / (run_A + 0.5)
    ratio_B = (sws_B + 0.5) / (run_B + 0.5)
    return {
        "n_blocks_A": int(len(blocks_A)),
        "n_blocks_B": int(len(blocks_B)),
        "n_running_blocks_A": run_A,
        "n_sws_blocks_A": sws_A,
        "n_running_blocks_B": run_B,
        "n_sws_blocks_B": sws_B,
        "frac_running_A": float(frac_run_A),
        "frac_running_B": float(frac_run_B),
        "state_separation": float(sep),
        "sws_to_running_ratio_A_pseudocount": float(ratio_A),
        "sws_to_running_ratio_B_pseudocount": float(ratio_B),
        "max_sws_to_running_ratio_pseudocount": float(max(ratio_A, ratio_B)),
    }


# -----------------------------
# FSC core from arrays
# -----------------------------
def indices_from_mask(mask):
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def run_fsc_arrays(ccg, fsc, spike_times_s, unit_ids, n_cells, n_threads, n_blocks):
    n_spikes_by_cell = np.bincount(unit_ids.astype(np.int64), minlength=int(n_cells)).astype(np.int64)

    ccg_result = ccg.compute_compact_ccg_from_arrays(
        spike_times_s=spike_times_s,
        unit_ids=unit_ids,
        n_units=int(n_cells),
        bin_size=fsc.BIN_DUR,
        duration=fsc.WIN_DUR,
        n_threads=int(n_threads),
        n_blocks=int(n_blocks),
    )

    pair_ccg = ccg_result["pair_ccg"]
    lags = ccg_result["lags"]
    pair_first = ccg_result["pair_first"]
    pair_second = ccg_result["pair_second"]
    n_pairs, n_bins = pair_ccg.shape

    post_mask = (lags >= fsc.MIN_WIN_MONOSYN) & (lags <= fsc.WIN_MONOSYN + fsc.BIN_DUR / 2)
    pre_mask = (lags <= -fsc.MIN_WIN_MONOSYN) & (lags >= -fsc.WIN_MONOSYN - fsc.BIN_DUR / 2)
    zero_mask = (lags >= fsc.ZERO_BINS_EXC[0] - fsc.BIN_DUR / 2) & (lags <= fsc.ZERO_BINS_EXC[1] + fsc.BIN_DUR / 2)
    zero2_mask = (lags >= fsc.ZERO_BINS_STRICT[0] - fsc.BIN_DUR / 2) & (lags <= fsc.ZERO_BINS_STRICT[1] + fsc.BIN_DUR / 2)

    post_idx = fsc._indices_from_mask(post_mask) if hasattr(fsc, "_indices_from_mask") else indices_from_mask(post_mask)
    pre_idx = fsc._indices_from_mask(pre_mask) if hasattr(fsc, "_indices_from_mask") else indices_from_mask(pre_mask)
    zero_idx = fsc._indices_from_mask(zero_mask) if hasattr(fsc, "_indices_from_mask") else indices_from_mask(zero_mask)
    zero2_idx = fsc._indices_from_mask(zero2_mask) if hasattr(fsc, "_indices_from_mask") else indices_from_mask(zero2_mask)

    non_diagonal = pair_first != pair_second
    enough_counts = np.nanmedian(pair_ccg[:, lags > -0.01], axis=1) > fsc.SPIKE_COUNT_CCG_THRESHOLD
    zero_lag_is_global_max = np.nanmax(pair_ccg[:, zero_mask], axis=1) == np.nanmax(pair_ccg, axis=1)
    valid_pair_mask = non_diagonal & enough_counts & (~zero_lag_is_global_max)
    valid_pair_indices = np.flatnonzero(valid_pair_mask).astype(np.int64)

    pcausal_inh = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_strength_inh = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_ratio_inh = np.zeros((n_cells, n_cells), dtype=np.float64)
    pcausal_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_strength_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_ratio_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
    sig_con_exc = []
    sig_con_inh = []

    if valid_pair_indices.size:
        selected_pair_ccg = pair_ccg[valid_pair_indices, :].astype(np.float64, copy=False)
        pair_first_valid = pair_first[valid_pair_indices].astype(np.int32, copy=False)
        pair_second_valid = pair_second[valid_pair_indices].astype(np.int32, copy=False)

        pvals_by_pair, pred_by_pair, qvals_by_pair = fsc.fast_cch_conv_median_numba(selected_pair_ccg, fsc.W)

        n_bonferroni = np.ceil(fsc.WIN_MONOSYN / fsc.BIN_DUR) * 2
        hi_by_pair, lo_by_pair = fsc.cached_poisson_bounds(pred_by_pair, fsc.ALPHA, n_bonferroni)

        (
            pcausal_exc,
            syn_strength_exc,
            syn_ratio_exc,
            post_exc_flags,
            pre_exc_flags,
            post_inh_flags,
            pre_inh_flags,
            post_inh_strength,
            pre_inh_strength,
            post_inh_ratio,
            pre_inh_ratio,
            min_post_ccg,
            min_pre_ccg,
        ) = fsc._compute_checks_and_exc_outputs_numba(
            selected_pair_ccg,
            pred_by_pair,
            pvals_by_pair,
            qvals_by_pair,
            hi_by_pair,
            lo_by_pair,
            pair_first_valid,
            pair_second_valid,
            n_spikes_by_cell,
            post_idx,
            pre_idx,
            zero_idx,
            zero2_idx,
            int(n_cells),
            fsc.BIN_DUR,
            fsc.ALPHA,
            fsc.ALPHA2,
            fsc.SIG_WIDTH_EXC[0],
            fsc.SIG_WIDTH_EXC[-1],
            fsc.SIG_WIDTH_INH[0],
            fsc.SIG_WIDTH_INH[-1],
            fsc.PEAK_SD,
            fsc.DIP_SD,
            fsc.WIDTH_SD,
        )

        for local_k in range(valid_pair_indices.size):
            refcid = int(pair_first_valid[local_k])
            targetcid = int(pair_second_valid[local_k])

            if post_exc_flags[local_k]:
                sig_con_exc.append([refcid, targetcid])
            if pre_exc_flags[local_k]:
                sig_con_exc.append([targetcid, refcid])

            if post_inh_flags[local_k]:
                p_causal = poisson.cdf(min_post_ccg[local_k], min_pre_ccg[local_k])
                if p_causal < fsc.ALPHA and post_inh_strength[local_k] > syn_strength_exc[refcid, targetcid]:
                    syn_strength_inh[refcid, targetcid] = post_inh_strength[local_k]
                    syn_ratio_inh[refcid, targetcid] = post_inh_ratio[local_k]
                    sig_con_inh.append([refcid, targetcid])
                    pcausal_inh[refcid, targetcid] = -1.0

            if pre_inh_flags[local_k]:
                p_causal_reverse = poisson.cdf(min_pre_ccg[local_k], min_post_ccg[local_k])
                if p_causal_reverse < fsc.ALPHA and pre_inh_strength[local_k] > syn_strength_exc[targetcid, refcid]:
                    syn_strength_inh[targetcid, refcid] = pre_inh_strength[local_k]
                    syn_ratio_inh[targetcid, refcid] = pre_inh_ratio[local_k]
                    sig_con_inh.append([targetcid, refcid])
                    pcausal_inh[targetcid, refcid] = -1.0

    return {
        "Pexc": pcausal_exc,
        "Pinh": pcausal_inh,
        "strengthexc": syn_strength_exc,
        "strengthinh": syn_strength_inh,
        "ratioexc": syn_ratio_exc,
        "ratioinh": syn_ratio_inh,
        "sig_exc_con": np.asarray(sig_con_exc, dtype=int).reshape(-1, 2),
        "sig_inh_con": np.asarray(sig_con_inh, dtype=int).reshape(-1, 2),
        "n_spikes_by_cell": n_spikes_by_cell,
        "n_pairs_checked": int(valid_pair_indices.size),
        "ccg_elapsed": float(ccg_result["elapsed"]),
    }


def combined_weight_and_ratio(result):
    W = np.maximum(np.abs(result["strengthexc"]), np.abs(result["strengthinh"]))
    R = np.maximum(np.abs(result["ratioexc"]), np.abs(result["ratioinh"]))
    return W.astype(np.float64, copy=False), R.astype(np.float64, copy=False)


def edge_type_matrix(result):
    exc = np.abs(result["strengthexc"]) > 0
    inh = np.abs(result["strengthinh"]) > 0
    out = np.full(exc.shape, "", dtype=object)
    out[exc] = "exc"
    out[inh] = "inh"
    out[exc & inh] = "both"
    return out


# -----------------------------
# Scoring/logging
# -----------------------------
def score_split(split_spec, result_A, result_B, cluster_ids, block_table):
    W_A, R_A = combined_weight_and_ratio(result_A)
    W_B, R_B = combined_weight_and_ratio(result_B)

    n_cells = W_A.shape[0]
    offdiag = ~np.eye(n_cells, dtype=bool)

    detected_A = (W_A > 0) & offdiag
    detected_B = (W_B > 0) & offdiag
    union = detected_A | detected_B

    W_max = np.maximum(W_A, W_B)
    R_max = np.maximum(R_A, R_B)
    delta_W = np.abs(W_A - W_B)
    delta_R = np.abs(R_A - R_B)

    n_possible = int(np.sum(offdiag))
    n_union = int(np.sum(union))

    vals_max_union = W_max[union]
    vals_delta_union = delta_W[union]
    vals_Rmax_union = R_max[union]
    vals_Rdelta_union = delta_R[union]

    def top_mean(vals, frac=0.01):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return 0.0
        k = max(1, int(np.ceil(vals.size * frac)))
        return float(np.mean(np.sort(vals)[-k:]))

    state_counts = split_state_counts(split_spec["blocks_A"], split_spec["blocks_B"], block_table)

    summary = {
        "split_id": split_spec["split_id"],
        "split_type": split_spec["split_type"],
        "label_A": split_spec["label_A"],
        "label_B": split_spec["label_B"],
        **state_counts,
        "n_cells": int(n_cells),
        "n_possible_directed_pairs": n_possible,
        "n_edges_A": int(np.sum(detected_A)),
        "n_edges_B": int(np.sum(detected_B)),
        "n_union_edges": n_union,
        "n_A_only_edges": int(np.sum(detected_A & (~detected_B))),
        "n_B_only_edges": int(np.sum(detected_B & (~detected_A))),
        "state_specific_edge_fraction": float((np.sum(detected_A ^ detected_B) / n_union) if n_union else 0.0),
        "sum_weight_A_all_pairs": float(np.sum(W_A[offdiag])),
        "sum_weight_B_all_pairs": float(np.sum(W_B[offdiag])),
        "max_weight_score_all_pairs": float(np.sum(W_max[offdiag])),
        "max_weight_score_per_possible_pair": float(np.sum(W_max[offdiag]) / n_possible),
        "max_weight_score_union_edges": float(np.sum(vals_max_union)) if n_union else 0.0,
        "max_weight_score_per_union_edge": float(np.mean(vals_max_union)) if n_union else 0.0,
        "delta_weight_score_all_pairs": float(np.sum(delta_W[offdiag])),
        "delta_weight_score_per_possible_pair": float(np.sum(delta_W[offdiag]) / n_possible),
        "delta_weight_score_union_edges": float(np.sum(vals_delta_union)) if n_union else 0.0,
        "delta_weight_score_per_union_edge": float(np.mean(vals_delta_union)) if n_union else 0.0,
        "top1pct_max_weight_union_mean": top_mean(vals_max_union, 0.01),
        "top1pct_delta_weight_union_mean": top_mean(vals_delta_union, 0.01),
        "max_ratio_score_union_edges": float(np.sum(vals_Rmax_union)) if n_union else 0.0,
        "delta_ratio_score_union_edges": float(np.sum(vals_Rdelta_union)) if n_union else 0.0,
        "top1pct_max_ratio_union_mean": top_mean(vals_Rmax_union, 0.01),
        "top1pct_delta_ratio_union_mean": top_mean(vals_Rdelta_union, 0.01),
        "n_pairs_checked_A": int(result_A["n_pairs_checked"]),
        "n_pairs_checked_B": int(result_B["n_pairs_checked"]),
        "ccg_elapsed_A_s": float(result_A["ccg_elapsed"]),
        "ccg_elapsed_B_s": float(result_B["ccg_elapsed"]),
        "n_spikes_A": int(np.sum(result_A["n_spikes_by_cell"])),
        "n_spikes_B": int(np.sum(result_B["n_spikes_by_cell"])),
    }

    # Union-detected pair rows only; all zero pairs are implicit from n_possible.
    pair_rows = []
    etype_A = edge_type_matrix(result_A)
    etype_B = edge_type_matrix(result_B)
    pre_ids, post_ids = np.where(union)
    for pre, post in zip(pre_ids, post_ids):
        pair_rows.append(
            {
                "split_id": split_spec["split_id"],
                "split_type": split_spec["split_type"],
                "pre_dense_id": int(pre),
                "post_dense_id": int(post),
                "pre_cluster_id": int(cluster_ids[pre]),
                "post_cluster_id": int(cluster_ids[post]),
                "weight_A": float(W_A[pre, post]),
                "weight_B": float(W_B[pre, post]),
                "max_weight": float(W_max[pre, post]),
                "abs_delta_weight": float(delta_W[pre, post]),
                "ratio_A": float(R_A[pre, post]),
                "ratio_B": float(R_B[pre, post]),
                "max_ratio": float(R_max[pre, post]),
                "abs_delta_ratio": float(delta_R[pre, post]),
                "detected_A": bool(detected_A[pre, post]),
                "detected_B": bool(detected_B[pre, post]),
                "edge_type_A": str(etype_A[pre, post]),
                "edge_type_B": str(etype_B[pre, post]),
            }
        )

    return summary, pair_rows, {
        "W_A": W_A.astype(np.float32),
        "W_B": W_B.astype(np.float32),
        "R_A": R_A.astype(np.float32),
        "R_B": R_B.astype(np.float32),
    }


def make_plots(summary_df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    true = summary_df[summary_df["split_type"] == "true_state"]
    rand = summary_df[summary_df["split_type"] == "random"]
    if true.empty or rand.empty:
        print("Skipping plots because true or random splits are missing.")
        return []

    true_row = true.iloc[0]
    saved = []

    plot_specs = [
        ("max_weight_score_per_possible_pair", "Max-weight score per possible pair", "max_weight_score_hist"),
        ("delta_weight_score_per_possible_pair", "Delta-weight score per possible pair", "delta_weight_score_hist"),
    ]

    for col, xlabel, stem in plot_specs:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.hist(rand[col].to_numpy(dtype=float), bins=20, alpha=0.8)
        ax.axvline(float(true_row[col]), linewidth=2.5, linestyle="--", label="true running vs SWS")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Random split count")
        ax.set_title(f"True state split vs random splits\n{col}")
        ax.legend()
        fig.tight_layout()
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        saved.extend([png, pdf])

    for col, ylabel, stem in [
        ("max_weight_score_per_possible_pair", "Max-weight score per possible pair", "state_separation_vs_max_weight"),
        ("delta_weight_score_per_possible_pair", "Delta-weight score per possible pair", "state_separation_vs_delta_weight"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(rand["state_separation"], rand[col], s=40, alpha=0.75, label="random splits")
        ax.scatter(true_row["state_separation"], true_row[col], s=110, marker="*", label="true running vs SWS")
        ax.set_xlabel("State separation")
        ax.set_ylabel(ylabel)
        ax.set_title("State purity vs FSC score")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        saved.extend([png, pdf])

    return saved


# -----------------------------
# Main
# -----------------------------
def run_analysis(
    ks_dir: Path,
    out_dir: Path,
    animal_id: str,
    sample_rate: float,
    threads: int,
    blocks_for_ccg: int,
    total_duration_h: float,
    state_boundary_h: float,
    block_min: float,
    n_random_splits: int,
    seed: int,
    save_matrices: bool,
):
    start_time = time.time()

    ccg, fsc = load_fsc_modules()

    ks_dir = Path(ks_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading KS directory: {ks_dir}")
    spike_times_s, unit_ids, cluster_ids, dense_meta = load_kilosort_spikes(
        ks_dir=ks_dir,
        sample_rate=sample_rate,
        refractory_s=REFRACTORY_S,
    )
    n_cells = int(cluster_ids.size)
    print(f"  n_cells: {n_cells}")
    print(f"  total good-unit spikes after close-spike filter: {spike_times_s.size:,}")

    split_specs, block_table, block_s, total_s = make_true_and_random_splits(
        total_h=total_duration_h,
        boundary_h=state_boundary_h,
        block_min=block_min,
        n_random=n_random_splits,
        seed=seed,
    )

    block_table.to_csv(out_dir / f"{animal_id}_state_split_blocks.csv", index=False)
    dense_meta.to_csv(out_dir / f"{animal_id}_state_split_unit_metadata.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "ks_dir": str(ks_dir),
        "out_dir": str(out_dir),
        "animal_id": animal_id,
        "sample_rate": float(sample_rate),
        "refractory_s": float(REFRACTORY_S),
        "threads": int(threads),
        "blocks_for_ccg": int(blocks_for_ccg),
        "total_duration_h": float(total_duration_h),
        "state_boundary_h": float(state_boundary_h),
        "block_min": float(block_min),
        "n_random_splits": int(n_random_splits),
        "random_seed": int(seed),
        "primary_weight": "max(abs(strengthexc), abs(strengthinh))",
        "primary_score": "sum(max(W_A, W_B)) over off-diagonal directed pairs",
        "save_weight_matrices_npz": bool(save_matrices),
    }
    with open(out_dir / f"{animal_id}_state_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary_rows = []
    pair_rows_all = []

    # Optional matrix storage.
    matrix_store = {
        "cluster_ids": cluster_ids.astype(np.int64),
        "split_ids": np.asarray([s["split_id"] for s in split_specs], dtype=object),
    } if save_matrices else None
    if save_matrices:
        W_A_all = []
        W_B_all = []
        R_A_all = []
        R_B_all = []

    for idx, split in enumerate(split_specs):
        print(f"\n[{idx+1}/{len(split_specs)}] Split: {split['split_id']} ({split['split_type']})")

        mask_A = mask_for_block_set(spike_times_s, split["blocks_A"], block_s, 0.0, total_s)
        mask_B = mask_for_block_set(spike_times_s, split["blocks_B"], block_s, 0.0, total_s)

        print(f"  spikes A: {int(np.sum(mask_A)):,}")
        print(f"  spikes B: {int(np.sum(mask_B)):,}")

        result_A = run_fsc_arrays(
            ccg=ccg,
            fsc=fsc,
            spike_times_s=spike_times_s[mask_A],
            unit_ids=unit_ids[mask_A],
            n_cells=n_cells,
            n_threads=threads,
            n_blocks=blocks_for_ccg,
        )
        result_B = run_fsc_arrays(
            ccg=ccg,
            fsc=fsc,
            spike_times_s=spike_times_s[mask_B],
            unit_ids=unit_ids[mask_B],
            n_cells=n_cells,
            n_threads=threads,
            n_blocks=blocks_for_ccg,
        )

        summary, pair_rows, matrices = score_split(split, result_A, result_B, cluster_ids, block_table)
        summary_rows.append(summary)
        pair_rows_all.extend(pair_rows)

        # Incremental writes so partial progress is preserved if a long run is interrupted.
        pd.DataFrame(summary_rows).to_csv(out_dir / f"{animal_id}_state_split_summary.csv", index=False)
        if pair_rows_all:
            pd.DataFrame(pair_rows_all).to_csv(out_dir / f"{animal_id}_state_split_pair_weights_detected_union.csv", index=False)

        if save_matrices:
            W_A_all.append(matrices["W_A"])
            W_B_all.append(matrices["W_B"])
            R_A_all.append(matrices["R_A"])
            R_B_all.append(matrices["R_B"])

        print(
            f"  max_weight_score_per_pair: {summary['max_weight_score_per_possible_pair']:.6g}; "
            f"delta_weight_score_per_pair: {summary['delta_weight_score_per_possible_pair']:.6g}; "
            f"union edges: {summary['n_union_edges']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows_all)

    summary_path = out_dir / f"{animal_id}_state_split_summary.csv"
    pair_path = out_dir / f"{animal_id}_state_split_pair_weights_detected_union.csv"
    summary_df.to_csv(summary_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    if save_matrices:
        npz_path = out_dir / f"{animal_id}_state_split_weight_matrices_float32.npz"
        np.savez_compressed(
            npz_path,
            cluster_ids=cluster_ids.astype(np.int64),
            split_ids=np.asarray([s["split_id"] for s in split_specs], dtype=object),
            W_A=np.stack(W_A_all, axis=0).astype(np.float32),
            W_B=np.stack(W_B_all, axis=0).astype(np.float32),
            R_A=np.stack(R_A_all, axis=0).astype(np.float32),
            R_B=np.stack(R_B_all, axis=0).astype(np.float32),
        )
        print(f"\nSaved matrix NPZ: {npz_path}")

    saved_plots = make_plots(summary_df, out_dir)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {pair_path}")
    print(f"  {out_dir / f'{animal_id}_state_split_blocks.csv'}")
    print(f"  {out_dir / f'{animal_id}_state_split_unit_metadata.csv'}")
    print(f"  {out_dir / f'{animal_id}_state_split_manifest.json'}")
    for p in saved_plots:
        print(f"  {p}")

    true = summary_df[summary_df["split_type"] == "true_state"].iloc[0]
    rand = summary_df[summary_df["split_type"] == "random"]
    if not rand.empty:
        for col in ["max_weight_score_per_possible_pair", "delta_weight_score_per_possible_pair"]:
            rank = int(np.sum(rand[col].to_numpy(float) >= float(true[col])) + 1)
            print(
                f"\nTrue split {col}: {float(true[col]):.6g}; "
                f"rank {rank}/{len(rand)+1} where 1 is largest/outlier-high."
            )

    print(f"\nElapsed: {(time.time() - start_time) / 60.0:.2f} min")


def parse_args():
    p = argparse.ArgumentParser(description="Run FSC true state split vs random 10-min-block null splits.")
    p.add_argument("--ks-dir", type=Path, default=KS_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--animal-id", type=str, default=ANIMAL_ID)
    p.add_argument("--sample-rate", type=float, default=SAMPLE_RATE)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--blocks-for-ccg", type=int, default=BLOCKS_FOR_CCG)
    p.add_argument("--total-duration-h", type=float, default=TOTAL_DURATION_H)
    p.add_argument("--state-boundary-h", type=float, default=STATE_BOUNDARY_H)
    p.add_argument("--block-min", type=float, default=BLOCK_MIN)
    p.add_argument("--n-random-splits", type=int, default=N_RANDOM_SPLITS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--no-save-matrices", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_analysis(
        ks_dir=args.ks_dir,
        out_dir=args.out_dir,
        animal_id=args.animal_id,
        sample_rate=args.sample_rate,
        threads=args.threads,
        blocks_for_ccg=args.blocks_for_ccg,
        total_duration_h=args.total_duration_h,
        state_boundary_h=args.state_boundary_h,
        block_min=args.block_min,
        n_random_splits=args.n_random_splits,
        seed=args.seed,
        save_matrices=(not args.no_save_matrices),
    )


if __name__ == "__main__":
    main()
