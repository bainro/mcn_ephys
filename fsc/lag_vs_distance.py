#!/usr/bin/env python3
"""
analyze_fsc_bound_exceedance_vs_distance.py

Standalone exploratory analyzer for existing fast-FSC outputs.

Given one or more output directories containing:
  <animal_id>-fast-fsc.npz
  <animal_id>_directed_pair_distance_table.csv

it computes, for each detected directed edge, the average lag/bin where the CCG
exceeds its Poisson bound. Positive lag is oriented as source -> target.

Outputs:
  aggregate_bound_exceedance_edges.csv
  aggregate_bound_exceedance_by_side.csv
  aggregate_bound_exceedance_center_abs_ms_vs_distance.png
  aggregate_bound_exceedance_signed_center_ms_by_side_vs_distance.png
  aggregate_bound_exceedance_binned_mean_center_abs_ms.csv
  aggregate_bound_exceedance_binned_mean_center_abs_ms.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def get_pair_index_and_direction(source: int, target: int, pair_first, pair_second):
    lo = min(source, target)
    hi = max(source, target)
    hits = np.where((pair_first == lo) & (pair_second == hi))[0]
    if hits.size != 1:
        return None, 1
    pair_idx = int(hits[0])
    direction = 1 if source == lo else -1
    return pair_idx, direction


def mean_lag_summary(lags_s, mask):
    lags_s = np.asarray(lags_s, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    pos = mask & (lags_s > 0)
    neg = mask & (lags_s < 0)

    def mean_or_nan(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.mean(vals)) if vals.size else np.nan

    if np.any(pos) and np.any(neg):
        side_pattern = "both"
    elif np.any(pos):
        side_pattern = "positive"
    elif np.any(neg):
        side_pattern = "negative"
    else:
        side_pattern = "none"

    return {
        "n_bins_exceeded_total": int(np.sum(mask)),
        "n_bins_exceeded_positive": int(np.sum(pos)),
        "n_bins_exceeded_negative": int(np.sum(neg)),
        "exceedance_side_pattern": side_pattern,
        "center_signed_all_ms": mean_or_nan(lags_s[mask] * 1000.0),
        "center_abs_all_ms": mean_or_nan(np.abs(lags_s[mask]) * 1000.0),
        "center_signed_positive_ms": mean_or_nan(lags_s[pos] * 1000.0),
        "center_abs_positive_ms": mean_or_nan(np.abs(lags_s[pos]) * 1000.0),
        "center_signed_negative_ms": mean_or_nan(lags_s[neg] * 1000.0),
        "center_abs_negative_ms": mean_or_nan(np.abs(lags_s[neg]) * 1000.0),
    }


def edge_distance_lookup(path: Path | None):
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if not {"pre_dense_id", "post_dense_id"}.issubset(df.columns):
        return {}
    return {
        (int(r["pre_dense_id"]), int(r["post_dense_id"])): {c: r[c] for c in df.columns}
        for _, r in df.iterrows()
    }


def analyze_one(npz_path: Path, distance_csv: Path | None, animal_id: str | None, min_ms: float, max_ms: float):
    z = np.load(npz_path, allow_pickle=True)
    if animal_id is None:
        animal_id = npz_path.name.replace("-fast-fsc.npz", "")

    lags = z["time"].astype(float)
    pair_ccg = z["pair_ccg"]
    pair_bounds = z["pair_bounds"]
    pair_first = z["pair_first"].astype(int)
    pair_second = z["pair_second"].astype(int)
    sig_exc = z["sig_exc_con"].reshape(-1, 2) if z["sig_exc_con"].size else np.empty((0, 2), dtype=int)
    sig_inh = z["sig_inh_con"].reshape(-1, 2) if z["sig_inh_con"].size else np.empty((0, 2), dtype=int)

    min_s = min_ms / 1000.0
    max_s = max_ms / 1000.0
    bin_dur = float(z["bin_dur"]) if "bin_dur" in z else 0.0004

    distance_lookup = edge_distance_lookup(distance_csv)

    rows = []
    side_rows = []

    def add(source, target, edge_type):
        source = int(source)
        target = int(target)
        pair_idx, direction = get_pair_index_and_direction(source, target, pair_first, pair_second)
        if pair_idx is None:
            return

        if direction == 1:
            oriented_lags = lags
            ccg_counts = pair_ccg[pair_idx].astype(float)
            bounds = pair_bounds[pair_idx].astype(float)
        else:
            oriented_lags = -lags[::-1]
            ccg_counts = pair_ccg[pair_idx][::-1].astype(float)
            bounds = pair_bounds[pair_idx][::-1].astype(float)

        search = (
            np.isfinite(oriented_lags)
            & np.isfinite(ccg_counts)
            & np.isfinite(bounds[:, 0])
            & np.isfinite(bounds[:, 1])
            & (np.abs(oriented_lags) >= min_s - bin_dur / 2.0)
            & (np.abs(oriented_lags) <= max_s + bin_dur / 2.0)
        )
        if edge_type == "exc":
            exceed = search & (ccg_counts > bounds[:, 0])
            bound_name = "upper"
        else:
            exceed = search & (ccg_counts < bounds[:, 1])
            bound_name = "lower"

        out = {
            "animal_id": animal_id,
            "source_dense_id": source,
            "target_dense_id": target,
            "edge_type": edge_type,
            "bound_exceeded": bound_name,
            "pair_index": pair_idx,
            "direction_was_reversed_from_compact_pair": bool(direction == -1),
            "min_abs_lag_ms": min_ms,
            "max_abs_lag_ms": max_ms,
        }
        out.update(mean_lag_summary(oriented_lags, exceed))

        dist_row = distance_lookup.get((source, target), {})
        for col in ["distance_um", "comparison", "directed_comparison", "comparison_category", "pre_region", "post_region", "pre_cluster_id", "post_cluster_id"]:
            out[col] = dist_row.get(col, np.nan)
        rows.append(out)

        for side_name, side_mask in [("positive", exceed & (oriented_lags > 0)), ("negative", exceed & (oriented_lags < 0))]:
            if not np.any(side_mask):
                continue
            side_out = {
                "animal_id": animal_id,
                "source_dense_id": source,
                "target_dense_id": target,
                "edge_type": edge_type,
                "bound_exceeded": bound_name,
                "side": side_name,
                "pair_index": pair_idx,
                "n_bins_exceeded": int(np.sum(side_mask)),
                "center_signed_ms": float(np.mean(oriented_lags[side_mask] * 1000.0)),
                "center_abs_ms": float(np.mean(np.abs(oriented_lags[side_mask]) * 1000.0)),
                "min_abs_lag_ms": min_ms,
                "max_abs_lag_ms": max_ms,
            }
            for col in ["distance_um", "comparison", "directed_comparison", "comparison_category", "pre_region", "post_region", "pre_cluster_id", "post_cluster_id"]:
                side_out[col] = dist_row.get(col, np.nan)
            side_rows.append(side_out)

    for s, t in sig_exc:
        add(s, t, "exc")
    for s, t in sig_inh:
        add(s, t, "inh")

    return pd.DataFrame(rows), pd.DataFrame(side_rows)


def distance_bin_edges(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0, 1.0])
    template = np.asarray([0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
                           1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 5000], dtype=float)
    max_v = float(np.nanmax(values))
    if max_v <= template[-1]:
        return template[template <= max_v + 500.0]
    extra = np.arange(5500.0, np.ceil(max_v / 500.0) * 500.0 + 1000.0, 500.0)
    return np.concatenate([template, extra])


def make_plots(edge_df, side_df, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}

    work = edge_df[np.isfinite(edge_df.get("distance_um", np.nan)) & np.isfinite(edge_df.get("center_abs_all_ms", np.nan))].copy()
    if not work.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(work["distance_um"], work["center_abs_all_ms"], s=22, alpha=0.75)
        ax.set_xlabel("Distance between unit peak channels (um)")
        ax.set_ylabel("Mean absolute bound-exceedance time (ms)")
        ax.set_title("Bound-exceedance timing vs distance")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = out_dir / "aggregate_bound_exceedance_center_abs_ms_vs_distance.png"
        fig.savefig(p, dpi=250, bbox_inches="tight")
        plt.close(fig)
        out["edge_scatter_png"] = p

        edges = distance_bin_edges(work["distance_um"])
        work["distance_bin"] = pd.cut(work["distance_um"], bins=edges, right=False, include_lowest=True)
        rows = []
        for interval, g in work.groupby("distance_bin", observed=True):
            if pd.isna(interval) or g.empty:
                continue
            rows.append({
                "distance_bin_low_um": float(interval.left),
                "distance_bin_high_um": float(interval.right),
                "distance_bin_center_um": float((interval.left + interval.right)/2.0),
                "n_edges": int(len(g)),
                "mean_center_abs_all_ms": float(g["center_abs_all_ms"].mean()),
                "mean_n_bins_exceeded_total": float(g["n_bins_exceeded_total"].mean()),
            })
        binned = pd.DataFrame(rows)
        if not binned.empty:
            binned_path = out_dir / "aggregate_bound_exceedance_binned_mean_center_abs_ms.csv"
            binned.to_csv(binned_path, index=False)
            out["binned_csv"] = binned_path

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.plot(binned["distance_bin_center_um"], binned["mean_center_abs_all_ms"], marker="o", linewidth=2.0)
            ax.set_xlabel("Distance bin center (um)")
            ax.set_ylabel("Mean absolute bound-exceedance time (ms)")
            ax.set_title("Mean bound-exceedance timing by distance")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            p = out_dir / "aggregate_bound_exceedance_binned_mean_center_abs_ms.png"
            fig.savefig(p, dpi=250, bbox_inches="tight")
            plt.close(fig)
            out["binned_plot_png"] = p

    side = side_df[np.isfinite(side_df.get("distance_um", np.nan)) & np.isfinite(side_df.get("center_signed_ms", np.nan))].copy()
    if not side.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for side_name, g in side.groupby("side", sort=False):
            ax.scatter(g["distance_um"], g["center_signed_ms"], s=22, alpha=0.75, label=str(side_name))
        ax.axhline(0, linewidth=1, linestyle=":")
        ax.set_xlabel("Distance between unit peak channels (um)")
        ax.set_ylabel("Signed bound-exceedance center (ms)")
        ax.set_title("Bound-exceedance side vs distance")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = out_dir / "aggregate_bound_exceedance_signed_center_ms_by_side_vs_distance.png"
        fig.savefig(p, dpi=250, bbox_inches="tight")
        plt.close(fig)
        out["side_scatter_png"] = p

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Output directories or -fast-fsc.npz files.")
    ap.add_argument("--out-dir", default=None, help="Where to save aggregate outputs. Default: first directory / aggregate_bound_exceedance_timing.")
    ap.add_argument("--min-ms", type=float, default=0.8)
    ap.add_argument("--max-ms", type=float, default=4.5)
    args = ap.parse_args()

    edge_rows = []
    side_rows = []
    for ptxt in args.paths:
        p = Path(ptxt)
        if p.is_dir():
            npzs = sorted(p.glob("*-fast-fsc.npz"))
        else:
            npzs = [p]

        for npz_path in npzs:
            animal_id = npz_path.name.replace("-fast-fsc.npz", "")
            distance_csv = npz_path.parent / f"{animal_id}_directed_pair_distance_table.csv"
            edge_df, side_df = analyze_one(npz_path, distance_csv, animal_id, args.min_ms, args.max_ms)
            edge_rows.append(edge_df)
            side_rows.append(side_df)

    if not edge_rows:
        raise RuntimeError("No edge rows generated.")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.paths[0]).parent / "aggregate_bound_exceedance_timing"
    if Path(args.paths[0]).is_dir() and args.out_dir is None:
        out_dir = Path(args.paths[0]) / "aggregate_bound_exceedance_timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_all = pd.concat(edge_rows, ignore_index=True)
    side_all = pd.concat(side_rows, ignore_index=True) if side_rows else pd.DataFrame()

    edge_path = out_dir / "aggregate_bound_exceedance_edges.csv"
    side_path = out_dir / "aggregate_bound_exceedance_by_side.csv"
    edge_all.to_csv(edge_path, index=False)
    side_all.to_csv(side_path, index=False)

    print("Saved:", edge_path)
    print("Saved:", side_path)
    for _, out_path in make_plots(edge_all, side_all, out_dir).items():
        print("Saved:", out_path)


if __name__ == "__main__":
    main()