#!/usr/bin/env python3
"""
run_fast_fsc.py

Final compact/fast FSC runner for the fake FSC harness and similarly structured
object-array spike-time inputs.

This stripped version intentionally removes:
    - old full cch/pred/pval/bounds compatibility reconstruction
    - profiling fields
    - explicit Numba warmup calls
    - imports from old ccgutils/reference scripts

For timing, run once to compile Numba functions, then time a second run.

Output contains the scientific/regression fields:
    Pexc, Pinh
    sig_exc_con, sig_inh_con
    strengthexc, strengthinh
    ratioexc, ratioinh
    pair_ccg, pair_pred, pair_pval, pair_bounds
    pair_first, pair_second, valid_pair_indices
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import poisson

import ccg


# FSC constants matching the old fake reference runner.
BIN_DUR = 0.0004
WIN_DUR = 0.05
ZERO_BINS_EXC = [-0.0006, 0.0006]
ZERO_BINS_INH = [-0.00075, 0.00075]
ZERO_BINS_STRICT = [-0.0002, 0.0002]
WIN_MONOSYN = 0.0045
MIN_WIN_MONOSYN = 0.0008
SIG_WIDTH_EXC = [0.0004, 0.004]
SIG_WIDTH_INH = [0.0004, 0.004]
ALPHA = 0.001
ALPHA2 = 0.01
W = int(0.01 / BIN_DUR)
PEAK_SD = 3.0
DIP_SD = 2.0
WIDTH_SD = 1.0
SPIKE_COUNT_CCG_THRESHOLD = 15.0


def remove_close_spikes_legacy_style(spikes: np.ndarray, refractory_s: float = 0.002) -> np.ndarray:
    spikes = np.asarray(spikes, dtype=np.float64)
    if spikes.size <= 1:
        return spikes
    return spikes[np.where(np.diff(spikes) > refractory_s)[0] + 1]


def load_fake_fsc_inputs(root: Path, animal: str):
    metrics_dir = root / "fake_metrics"
    metrics_csv = root / "fake_pooled_metrics.csv"

    df = pd.read_csv(metrics_csv)
    df["cellType"] = df["cellType"].replace("Wide Interneuron", "Pyramidal Cell")

    dfanimal = df[df.aname == animal].copy()
    dfanimal = dfanimal[
        [
            "aname",
            "cluster_id",
            "ch",
            "depth",
            "cellType",
            "amp",
            "region",
            "isGood_awk",
            "isGood_slp",
            "layer",
            "wftroughToPeak",
        ]
    ]

    spktimes_obj = np.load(metrics_dir / f"{animal}-spiketimes-RSC.npy", allow_pickle=True)

    keep_idx = np.where(
        (dfanimal.cellType != "Positive")
        & (dfanimal.cellType != "Biphasic")
        & (dfanimal.region != "DG")
    )[0]
    spktimes_obj = spktimes_obj[keep_idx]
    dfanimal = dfanimal.iloc[keep_idx].reset_index(drop=True)

    order = {"DG": 1, "dCA1": 2, "RSCag": 3, "RSCg": 4}
    layer_order = {
        ("DG", np.nan): 1,
        ("dCA1", np.nan): 2,
        ("RSCag", "deep"): 3,
        ("RSCg", "deep"): 4,
        ("RSCag", "sup"): 5,
        ("RSCg", "sup"): 6,
    }

    dfanimal["region_sort"] = dfanimal["region"].map(order)
    dfanimal["layer_sort"] = dfanimal.apply(
        lambda x: layer_order.get((x["region"], x["layer"])), axis=1
    )

    dfanimal.sort_values(by=["region_sort", "layer_sort", "depth"], inplace=True)
    sorted_index = dfanimal.index.to_numpy()

    spktimes_obj = spktimes_obj[sorted_index]
    dfanimal = dfanimal.reset_index(drop=True)

    all_spike_times = []
    all_unit_ids = []
    cleaned_spike_counts = []

    for dense_cell_id, spikes in enumerate(spktimes_obj):
        spikes = remove_close_spikes_legacy_style(spikes, refractory_s=0.002)
        cleaned_spike_counts.append(spikes.size)

        all_spike_times.extend(spikes)
        all_unit_ids.extend([dense_cell_id] * spikes.size)

    spike_times_s = np.asarray(all_spike_times, dtype=np.float64)
    unit_ids = np.asarray(all_unit_ids, dtype=np.int32)
    n_spikes_by_cell = np.asarray(cleaned_spike_counts, dtype=np.int64)
    n_cells = int(n_spikes_by_cell.size)

    return spike_times_s, unit_ids, n_spikes_by_cell, n_cells, dfanimal


def build_reflect_median_indices(n_bins: int, W: int, hf: float = 1.0) -> np.ndarray:
    W = (W // 2) * 2 + 1
    hw = W // 2

    base = np.arange(n_bins, dtype=np.int64)
    padded = np.pad(base, (hw, hw), mode="reflect")

    rows = []
    for r in range(n_bins):
        idx = padded[r : r + W]
        if hf > 0.5:
            idx = np.concatenate([idx[:hw], idx[hw + 1 :]])
        rows.append(idx.astype(np.int64, copy=False))

    return np.vstack(rows)


@njit(cache=True)
def _median_small(values):
    n = values.shape[0]

    for i in range(1, n):
        x = values[i]
        j = i - 1
        while j >= 0 and values[j] > x:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = x

    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


@njit(parallel=True, cache=True)
def _median_predictor_numba(cch_pair_by_bin, median_indices):
    n_pairs, n_bins = cch_pair_by_bin.shape
    window_len = median_indices.shape[1]
    pred = np.empty((n_pairs, n_bins), dtype=np.float64)

    for pair_idx in prange(n_pairs):
        scratch = np.empty(window_len, dtype=np.float64)

        for bin_idx in range(n_bins):
            for k in range(window_len):
                scratch[k] = cch_pair_by_bin[pair_idx, median_indices[bin_idx, k]]

            pred[pair_idx, bin_idx] = _median_small(scratch)

    return pred


def fast_cch_conv_median_numba(pair_ccg_selected: np.ndarray, W: int):
    pair_ccg_selected = np.asarray(pair_ccg_selected, dtype=np.float64)
    n_selected, n_bins = pair_ccg_selected.shape

    median_indices = build_reflect_median_indices(n_bins, W, hf=1.0)
    pred = _median_predictor_numba(pair_ccg_selected, median_indices)

    cch_int = np.round(pair_ccg_selected).astype(np.int64)
    pvals = 1.0 - poisson.cdf(cch_int - 1, pred) - poisson.pmf(cch_int, pred) * 0.5
    qvals = 1.0 - pvals

    return pvals, pred, qvals


def cached_poisson_bounds(pred_by_pair: np.ndarray, alpha: float, n_bonferroni: float):
    pred_by_pair = np.asarray(pred_by_pair, dtype=np.float64)
    flat_pred = pred_by_pair.reshape(-1)

    unique_pred, inverse = np.unique(flat_pred, return_inverse=True)

    q_hi = 1.0 - alpha / n_bonferroni
    q_lo = alpha / n_bonferroni

    hi_unique = poisson.ppf(q_hi, unique_pred)
    lo_unique = poisson.ppf(q_lo, unique_pred)

    hi = hi_unique[inverse].reshape(pred_by_pair.shape)
    lo = lo_unique[inverse].reshape(pred_by_pair.shape)

    return hi, lo


@njit(cache=True)
def _max_over_indices(x, idx):
    m = x[idx[0]]
    for k in range(1, idx.shape[0]):
        v = x[idx[k]]
        if v > m:
            m = v
    return m


@njit(cache=True)
def _min_over_indices(x, idx):
    m = x[idx[0]]
    for k in range(1, idx.shape[0]):
        v = x[idx[k]]
        if v < m:
            m = v
    return m


@njit(cache=True)
def _mean_over_indices(x, idx):
    s = 0.0
    for k in range(idx.shape[0]):
        s += x[idx[k]]
    return s / idx.shape[0]


@njit(cache=True)
def _std_over_indices(x, idx):
    mu = _mean_over_indices(x, idx)
    ss = 0.0
    for k in range(idx.shape[0]):
        d = x[idx[k]] - mu
        ss += d * d
    return np.sqrt(ss / idx.shape[0])


@njit(cache=True)
def _std_all(x):
    n = x.shape[0]
    mu = 0.0
    for i in range(n):
        mu += x[i]
    mu /= n

    ss = 0.0
    for i in range(n):
        d = x[i] - mu
        ss += d * d

    return np.sqrt(ss / n)


@njit(cache=True)
def _any_sig_gt_at_indices(ccg, bound, idx):
    for k in range(idx.shape[0]):
        i = idx[k]
        if ccg[i] > bound[i]:
            return True
    return False


@njit(cache=True)
def _any_sig_lt_at_indices(ccg, bound, idx):
    for k in range(idx.shape[0]):
        i = idx[k]
        if ccg[i] < bound[i]:
            return True
    return False


@njit(cache=True)
def _any_p_lt_at_indices(p, idx, alpha):
    for k in range(idx.shape[0]):
        if p[idx[k]] < alpha:
            return True
    return False


@njit(cache=True)
def _all_not_sig_gt_at_indices(ccg, bound, idx):
    for k in range(idx.shape[0]):
        i = idx[k]
        if ccg[i] > bound[i]:
            return False
    return True


@njit(cache=True)
def _any_not_sig_lt_at_indices(ccg, bound, idx):
    for k in range(idx.shape[0]):
        i = idx[k]
        if not (ccg[i] < bound[i]):
            return True
    return False


@njit(cache=True)
def _has_width_exc(pvals, baseline, idx, alpha2, peak_ht, std_baseline, width_sd, bin_dur, width_min, width_max):
    run_len = 0

    for k in range(idx.shape[0]):
        i = idx[k]
        flag = (pvals[i] < alpha2) and (
            (baseline[i] > 0.5 * peak_ht) or (baseline[i] > width_sd * std_baseline)
        )

        if flag:
            run_len += 1
        else:
            if run_len > 0:
                width = run_len * bin_dur
                if width_min <= width <= width_max:
                    return True
            run_len = 0

    if run_len > 0:
        width = run_len * bin_dur
        if width_min <= width <= width_max:
            return True

    return False


@njit(cache=True)
def _has_width_inh(qvals, baseline, idx, alpha2, dip_ht, std_baseline, width_sd, bin_dur, width_min, width_max):
    run_len = 0

    for k in range(idx.shape[0]):
        i = idx[k]
        flag = (qvals[i] < alpha2) and (
            (baseline[i] < 0.5 * dip_ht) or (baseline[i] < -width_sd * std_baseline)
        )

        if flag:
            run_len += 1
        else:
            if run_len > 0:
                width = run_len * bin_dur
                if width_min <= width <= width_max:
                    return True
            run_len = 0

    if run_len > 0:
        width = run_len * bin_dur
        if width_min <= width <= width_max:
            return True

    return False


@njit(parallel=True, cache=True)
def _compute_checks_and_exc_outputs_numba(
    selected_ccg,
    pred,
    pvals,
    qvals,
    hi_bound,
    lo_bound,
    pair_first_valid,
    pair_second_valid,
    n_spikes_by_cell,
    post_idx,
    pre_idx,
    zero_idx,
    zero2_idx,
    n_cells,
    bin_dur,
    alpha,
    alpha2,
    exc_width_min,
    exc_width_max,
    inh_width_min,
    inh_width_max,
    peak_sd,
    dip_sd,
    width_sd,
):
    n_valid, n_bins = selected_ccg.shape

    pcausal_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_strength_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_ratio_exc = np.zeros((n_cells, n_cells), dtype=np.float64)

    post_exc_flags = np.zeros(n_valid, dtype=np.bool_)
    pre_exc_flags = np.zeros(n_valid, dtype=np.bool_)
    post_inh_flags = np.zeros(n_valid, dtype=np.bool_)
    pre_inh_flags = np.zeros(n_valid, dtype=np.bool_)

    post_inh_strength = np.zeros(n_valid, dtype=np.float64)
    pre_inh_strength = np.zeros(n_valid, dtype=np.float64)
    post_inh_ratio = np.zeros(n_valid, dtype=np.float64)
    pre_inh_ratio = np.zeros(n_valid, dtype=np.float64)

    min_post_ccg = np.zeros(n_valid, dtype=np.float64)
    min_pre_ccg = np.zeros(n_valid, dtype=np.float64)

    for k in prange(n_valid):
        refcid = pair_first_valid[k]
        targetcid = pair_second_valid[k]

        baseline = np.empty(n_bins, dtype=np.float64)
        for b in range(n_bins):
            baseline[b] = selected_ccg[k, b] - pred[k, b]

        std_baseline = _std_all(baseline)

        peak_post_any = _any_sig_gt_at_indices(selected_ccg[k], hi_bound[k], post_idx)
        peak_pre_any = _any_sig_gt_at_indices(selected_ccg[k], hi_bound[k], pre_idx)

        peak_pval_check_post = _any_p_lt_at_indices(pvals[k], post_idx, alpha)
        peak_pval_check_pre = _any_p_lt_at_indices(pvals[k], pre_idx, alpha)

        peak_ht_post = _max_over_indices(baseline, post_idx)
        peak_ht_pre = _max_over_indices(baseline, pre_idx)

        peakstd_check_post = peak_ht_post > peak_sd * std_baseline
        peakstd_check_pre = peak_ht_pre > peak_sd * std_baseline

        width_check_post = _has_width_exc(
            pvals[k],
            baseline,
            post_idx,
            alpha2,
            peak_ht_post,
            std_baseline,
            width_sd,
            bin_dur,
            exc_width_min,
            exc_width_max,
        )

        width_check_pre = _has_width_exc(
            pvals[k],
            baseline,
            pre_idx,
            alpha2,
            peak_ht_pre,
            std_baseline,
            width_sd,
            bin_dur,
            exc_width_min,
            exc_width_max,
        )

        zero_peak = _max_over_indices(selected_ccg[k], zero_idx)
        max_post_ccg = _max_over_indices(selected_ccg[k], post_idx)
        max_pre_ccg = _max_over_indices(selected_ccg[k], pre_idx)

        no_overlap_exc = _all_not_sig_gt_at_indices(selected_ccg[k], hi_bound[k], zero2_idx) and (
            (zero_peak < max_post_ccg) or (zero_peak < max_pre_ccg)
        )

        post_exc = peak_post_any and peak_pval_check_post and width_check_post and no_overlap_exc and peakstd_check_post
        pre_exc = peak_pre_any and peak_pval_check_pre and width_check_pre and no_overlap_exc and peakstd_check_pre

        post_exc_flags[k] = post_exc
        pre_exc_flags[k] = pre_exc

        if post_exc:
            positive_sum = 0.0
            for ii in range(post_idx.shape[0]):
                v = baseline[post_idx[ii]]
                if v > 0.0:
                    positive_sum += v

            syn_strength_exc[refcid, targetcid] = positive_sum / n_spikes_by_cell[refcid]

            pred_std = _std_over_indices(pred[k], post_idx)
            if pred_std != 0.0:
                syn_ratio_exc[refcid, targetcid] = (
                    max_post_ccg - _mean_over_indices(pred[k], post_idx)
                ) / pred_std
            else:
                syn_ratio_exc[refcid, targetcid] = np.nan

            pcausal_exc[refcid, targetcid] = 1.0

        if pre_exc:
            positive_sum = 0.0
            for ii in range(pre_idx.shape[0]):
                v = baseline[pre_idx[ii]]
                if v > 0.0:
                    positive_sum += v

            syn_strength_exc[targetcid, refcid] = positive_sum / n_spikes_by_cell[targetcid]

            pred_std = _std_over_indices(pred[k], pre_idx)
            if pred_std != 0.0:
                syn_ratio_exc[targetcid, refcid] = (
                    max_pre_ccg - _mean_over_indices(pred[k], pre_idx)
                ) / pred_std
            else:
                syn_ratio_exc[targetcid, refcid] = np.nan

            pcausal_exc[targetcid, refcid] = 1.0

        dip_post_any = _any_sig_lt_at_indices(selected_ccg[k], lo_bound[k], post_idx)
        dip_pre_any = _any_sig_lt_at_indices(selected_ccg[k], lo_bound[k], pre_idx)

        dip_pval_check_post = _any_p_lt_at_indices(qvals[k], post_idx, alpha)
        dip_pval_check_pre = _any_p_lt_at_indices(qvals[k], pre_idx, alpha)

        dip_ht_post = _min_over_indices(baseline, post_idx)
        dip_ht_pre = _min_over_indices(baseline, pre_idx)

        dipstd_check_post = dip_ht_post < dip_sd * std_baseline
        dipstd_check_pre = dip_ht_pre < dip_sd * std_baseline

        width_check_post_inh = _has_width_inh(
            qvals[k],
            baseline,
            post_idx,
            alpha2,
            dip_ht_post,
            std_baseline,
            width_sd,
            bin_dur,
            inh_width_min,
            inh_width_max,
        )

        width_check_pre_inh = _has_width_inh(
            qvals[k],
            baseline,
            pre_idx,
            alpha2,
            dip_ht_pre,
            std_baseline,
            width_sd,
            bin_dur,
            inh_width_min,
            inh_width_max,
        )

        zero_dip = _min_over_indices(selected_ccg[k], zero_idx)
        min_post = _min_over_indices(selected_ccg[k], post_idx)
        min_pre = _min_over_indices(selected_ccg[k], pre_idx)

        min_post_ccg[k] = min_post
        min_pre_ccg[k] = min_pre

        min_post_pre = min_post
        if min_pre < min_post_pre:
            min_post_pre = min_pre

        no_overlap_inh = _any_not_sig_lt_at_indices(selected_ccg[k], lo_bound[k], zero2_idx) and not (
            zero_dip < min_post_pre
        )

        post_inh = dip_post_any and dip_pval_check_post and no_overlap_inh and dipstd_check_post and width_check_post_inh
        pre_inh = dip_pre_any and dip_pval_check_pre and no_overlap_inh and dipstd_check_pre and width_check_pre_inh

        post_inh_flags[k] = post_inh
        pre_inh_flags[k] = pre_inh

        if post_inh:
            neg_sum = 0.0
            for ii in range(post_idx.shape[0]):
                v = baseline[post_idx[ii]]
                if v < 0.0:
                    neg_sum += v

            post_inh_strength[k] = abs(neg_sum) / n_spikes_by_cell[refcid]

            pred_std = _std_over_indices(pred[k], post_idx)
            if pred_std != 0.0:
                post_inh_ratio[k] = abs(
                    (min_post - _mean_over_indices(pred[k], post_idx)) / pred_std
                )
            else:
                post_inh_ratio[k] = np.nan

        if pre_inh:
            neg_sum = 0.0
            for ii in range(pre_idx.shape[0]):
                v = baseline[pre_idx[ii]]
                if v < 0.0:
                    neg_sum += v

            pre_inh_strength[k] = abs(neg_sum) / n_spikes_by_cell[targetcid]

            pred_std = _std_over_indices(pred[k], pre_idx)
            if pred_std != 0.0:
                pre_inh_ratio[k] = abs(
                    (min_pre - _mean_over_indices(pred[k], pre_idx)) / pred_std
                )
            else:
                pre_inh_ratio[k] = np.nan

    return (
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
    )


def _indices_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def run_fast_fsc(root: Path, animal: str, n_threads: int, n_blocks: int | None):
    output_dir = root / "fake_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    spike_times_s, unit_ids, n_spikes_by_cell, n_cells, dfanimal = load_fake_fsc_inputs(root, animal)

    print(f"Processing {animal}")
    print(f"  n_cells: {n_cells}")
    print(f"  total spikes after legacy close-spike filter: {spike_times_s.size}")

    ccg_result = ccg.compute_compact_ccg_from_arrays(
        spike_times_s=spike_times_s,
        unit_ids=unit_ids,
        n_units=n_cells,
        bin_size=BIN_DUR,
        duration=WIN_DUR,
        n_threads=n_threads,
        n_blocks=n_blocks,
    )

    pair_ccg = ccg_result["pair_ccg"]
    lags = ccg_result["lags"]
    pair_first = ccg_result["pair_first"]
    pair_second = ccg_result["pair_second"]

    n_pairs, n_bins = pair_ccg.shape

    post_mask = (lags >= MIN_WIN_MONOSYN) & (lags <= WIN_MONOSYN + BIN_DUR / 2)
    pre_mask = (lags <= -MIN_WIN_MONOSYN) & (lags >= -WIN_MONOSYN - BIN_DUR / 2)
    zero_mask = (lags >= ZERO_BINS_EXC[0] - BIN_DUR / 2) & (lags <= ZERO_BINS_EXC[1] + BIN_DUR / 2)
    zero2_mask = (lags >= ZERO_BINS_STRICT[0] - BIN_DUR / 2) & (lags <= ZERO_BINS_STRICT[1] + BIN_DUR / 2)

    post_idx = _indices_from_mask(post_mask)
    pre_idx = _indices_from_mask(pre_mask)
    zero_idx = _indices_from_mask(zero_mask)
    zero2_idx = _indices_from_mask(zero2_mask)

    non_diagonal = pair_first != pair_second
    enough_counts = np.nanmedian(pair_ccg[:, lags > -0.01], axis=1) > SPIKE_COUNT_CCG_THRESHOLD
    zero_lag_is_global_max = (
        np.nanmax(pair_ccg[:, zero_mask], axis=1) == np.nanmax(pair_ccg, axis=1)
    )

    valid_pair_mask = non_diagonal & enough_counts & (~zero_lag_is_global_max)
    valid_pair_indices = np.flatnonzero(valid_pair_mask).astype(np.int64)

    print(f"  pair_ccg shape: {pair_ccg.shape}")
    print(f"  pairs passing prefilter: {valid_pair_indices.size} / {n_pairs - n_cells}")

    pair_pred = np.full((n_pairs, n_bins), np.nan, dtype=np.float64)
    pair_pval = np.full((n_pairs, n_bins), np.nan, dtype=np.float64)
    pair_bounds = np.full((n_pairs, n_bins, 2), np.nan, dtype=np.float64)

    pcausal_inh = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_strength_inh = np.zeros((n_cells, n_cells), dtype=np.float64)
    syn_ratio_inh = np.zeros((n_cells, n_cells), dtype=np.float64)

    sig_con_exc = []
    sig_con_inh = []

    if valid_pair_indices.size:
        selected_pair_ccg = pair_ccg[valid_pair_indices, :].astype(np.float64, copy=False)
        pair_first_valid = pair_first[valid_pair_indices].astype(np.int32, copy=False)
        pair_second_valid = pair_second[valid_pair_indices].astype(np.int32, copy=False)

        pvals_by_pair, pred_by_pair, qvals_by_pair = fast_cch_conv_median_numba(selected_pair_ccg, W)

        pair_pval[valid_pair_indices, :] = pvals_by_pair
        pair_pred[valid_pair_indices, :] = pred_by_pair

        n_bonferroni = np.ceil(WIN_MONOSYN / BIN_DUR) * 2
        hi_by_pair, lo_by_pair = cached_poisson_bounds(pred_by_pair, ALPHA, n_bonferroni)

        pair_bounds[valid_pair_indices, :, 0] = hi_by_pair
        pair_bounds[valid_pair_indices, :, 1] = lo_by_pair

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
        ) = _compute_checks_and_exc_outputs_numba(
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
            n_cells,
            BIN_DUR,
            ALPHA,
            ALPHA2,
            SIG_WIDTH_EXC[0],
            SIG_WIDTH_EXC[-1],
            SIG_WIDTH_INH[0],
            SIG_WIDTH_INH[-1],
            PEAK_SD,
            DIP_SD,
            WIDTH_SD,
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
                if p_causal < ALPHA and post_inh_strength[local_k] > syn_strength_exc[refcid, targetcid]:
                    syn_strength_inh[refcid, targetcid] = post_inh_strength[local_k]
                    syn_ratio_inh[refcid, targetcid] = post_inh_ratio[local_k]
                    sig_con_inh.append([refcid, targetcid])
                    pcausal_inh[refcid, targetcid] = -1.0

            if pre_inh_flags[local_k]:
                p_causal_reverse = poisson.cdf(min_pre_ccg[local_k], min_post_ccg[local_k])
                if p_causal_reverse < ALPHA and pre_inh_strength[local_k] > syn_strength_exc[targetcid, refcid]:
                    syn_strength_inh[targetcid, refcid] = pre_inh_strength[local_k]
                    syn_ratio_inh[targetcid, refcid] = pre_inh_ratio[local_k]
                    sig_con_inh.append([targetcid, refcid])
                    pcausal_inh[targetcid, refcid] = -1.0

    else:
        pcausal_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
        syn_strength_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
        syn_ratio_exc = np.zeros((n_cells, n_cells), dtype=np.float64)

    sig_con_exc = np.asarray(sig_con_exc, dtype=int).reshape(-1, 2)
    sig_con_inh = np.asarray(sig_con_inh, dtype=int).reshape(-1, 2)

    out_path = output_dir / f"{animal}-fast-fsc.npz"
    np.savez(
        out_path,
        Pexc=pcausal_exc,
        Pinh=pcausal_inh,
        time=lags,
        sig_exc_con=sig_con_exc,
        sig_inh_con=sig_con_inh,
        strengthexc=syn_strength_exc,
        strengthinh=syn_strength_inh,
        ratioexc=syn_ratio_exc,
        ratioinh=syn_ratio_inh,
        n_spikes_by_cell=n_spikes_by_cell,
        df=dfanimal.to_dict("list"),
        n_pairs_checked=int(valid_pair_indices.size),
        pair_ccg=pair_ccg,
        pair_pred=pair_pred,
        pair_pval=pair_pval,
        pair_bounds=pair_bounds,
        pair_first=pair_first,
        pair_second=pair_second,
        valid_pair_indices=valid_pair_indices,
        ccg_elapsed=float(ccg_result["elapsed"]),
        n_threads=n_threads,
        n_blocks=-1 if n_blocks is None else n_blocks,
    )

    print(f"  excitatory edges: {sig_con_exc.tolist()}")
    print(f"  inhibitory edges: {sig_con_inh.tolist()}")
    print(f"  saved: {out_path}")

    return out_path



def _module_entry_message():
    print("This is the FSC core module. Run `python main.py` for the interactive/CLI entry point.")


if __name__ == "__main__":
    _module_entry_message()
