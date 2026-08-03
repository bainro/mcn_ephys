#!/usr/bin/env python3
"""
main.py

User-facing entry point for fast FSC analysis from Kilosort outputs.

Expected Kilosort files:
    spike_times.npy
    spike_clusters.npy
    cluster_info.tsv

Important:
    - spike_times.npy and spike_clusters.npy may be shaped (N,) or (N,1).
    - They are flattened safely.
    - cluster_info.tsv is used automatically, and only clusters labeled "good" are analyzed.
    - If cluster_info.tsv is missing, interactive mode asks for confirmation before using all clusters.
      CLI mode requires --allow-all-clusters if cluster_info.tsv is missing.

Outputs:
    <animal_id>-fast-fsc.npz
    optional <animal_id>_ccg_plots/ directory with significant CCG plots plus null examples
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import poisson

import ccg
import fsc


DEFAULT_SAMPLE_RATE = 30000.0
DEFAULT_THREADS = 8
DEFAULT_BLOCKS = 24
DEFAULT_REFRACTORY_S = 0.002


# -----------------------------
# Basic user interaction
# -----------------------------
def yes_no(prompt: str, default: bool | None = None) -> bool:
    suffix = " (y/n) " if default is None else (" (Y/n) " if default else " (y/N) ")
    while True:
        ans = input(prompt + suffix).strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def parse_comma_dirs(text: str) -> list[Path]:
    dirs = [Path(x.strip()).expanduser() for x in text.replace(", ", ",").split(",") if x.strip()]
    seen = set()
    out = []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def choose_ks_dirs() -> list[Path]:
    dirs = []
    try:
        import tkinter as tk
        from tkinter import filedialog

        gui_root = tk.Tk()
        if os.name == "nt":
            gui_root.attributes("-topmost", True, "-alpha", 0)
        else:
            gui_root.withdraw()

        print("\nA GUI/dialog box should appear. Press cancel or ESC when done selecting KS directories.")
        while True:
            d = filedialog.askdirectory(mustexist=True, title="Choose Kilosort output directory/directories")
            if os.name == "nt":
                gui_root.attributes("-topmost", True, "-alpha", 0)
            if d == () or d == "":
                break
            dirs.append(Path(d).resolve())

        gui_root.destroy()

    except Exception:
        msg = "\nWhich Kilosort directories do you want to process? You may specify multiple, separated by commas.\n"
        msg += "Each directory should contain spike_times.npy, spike_clusters.npy, and ideally cluster_info.tsv.\n\n"
        msg += "Example:\n  /data/mouse1_day1/ks, /data/mouse2_day6/ks\n\n"
        dirs = parse_comma_dirs(input(msg))

    if not dirs:
        raise RuntimeError("No input directories specified.")

    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"Kilosort directory not found: {d}")
        if not (d / "spike_times.npy").exists():
            raise FileNotFoundError(f"Missing spike_times.npy in {d}")
        if not (d / "spike_clusters.npy").exists():
            raise FileNotFoundError(f"Missing spike_clusters.npy in {d}")

    return dirs


def choose_output_root() -> Path | None:
    same_dir = yes_no("\nSave outputs to the same input directory/directories?", default=True)
    if same_dir:
        return None

    try:
        import tkinter as tk
        from tkinter import filedialog

        gui_root = tk.Tk()
        if os.name == "nt":
            gui_root.attributes("-topmost", True, "-alpha", 0)
        else:
            gui_root.withdraw()

        print("\nA GUI/dialog box should appear for the output root.")
        d = filedialog.askdirectory(title="Select root directory to save FSC outputs")
        if os.name == "nt":
            gui_root.attributes("-topmost", True, "-alpha", 0)
        gui_root.destroy()

        if d == () or d == "":
            raise RuntimeError("No output directory selected.")
        out_root = Path(d).resolve()

    except Exception:
        out_root = Path(input("\nWhere would you like to save the outputs?\n\n").strip()).expanduser().resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def output_dir_for_ks_dir(ks_dir: Path, output_root: Path | None, animal_id: str | None = None) -> Path:
    if output_root is None:
        return ks_dir.resolve()
    out_dir = output_root / (animal_id if animal_id else ks_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()


def ask_animal_ids(ks_dirs: list[Path]) -> list[str]:
    ids = []
    for d in ks_dirs:
        default_name = d.name
        ans = input(f"\nAnimal/session ID for:\n  {d}\nPress enter for '{default_name}'.\n\n").strip()
        ids.append(ans if ans else default_name)
    return ids


# -----------------------------
# UnitMetrics.csv / cluster_info.tsv handling
# -----------------------------
def _good_bool_from_exact_column(df: pd.DataFrame, path_label: str):
    col_map = {c.lower(): c for c in df.columns}

    if "good" in col_map:
        good_col = col_map["good"]
    elif "isgood" in col_map:
        good_col = col_map["isgood"]
    else:
        raise ValueError(
            f"{path_label} exists, but expected a good-unit column named "
            f"'good' or 'isGood'. Columns are: {list(df.columns)}"
        )

    s = df[good_col]

    if s.dtype == bool:
        return s.to_numpy(dtype=bool), good_col

    if np.issubdtype(s.dtype, np.number):
        return (s.to_numpy(dtype=float) > 0), good_col

    vals = s.astype(str).str.strip().str.lower()
    return vals.isin({"true", "1", "yes", "good"}).to_numpy(dtype=bool), good_col


def load_good_cluster_ids(
    ks_dir: Path,
    require_file: bool,
    interactive: bool,
    confirm_cluster_info_fallback: bool = False,
):
    unitmetrics_path = ks_dir / "UnitMetrics.csv"
    if unitmetrics_path.exists():
        df = pd.read_csv(unitmetrics_path)

        if "cluster_id" not in df.columns:
            raise ValueError(
                f"UnitMetrics.csv exists, but expected column 'cluster_id'. "
                f"Columns are: {list(df.columns)}"
            )

        good_mask, good_col = _good_bool_from_exact_column(df, "UnitMetrics.csv")
        good_ids = np.unique(df.loc[good_mask, "cluster_id"].astype(np.int64).to_numpy())

        if good_ids.size == 0:
            raise ValueError(f"UnitMetrics.csv in {ks_dir} contains zero good clusters.")

        print(f"  using {good_ids.size} good clusters from UnitMetrics.csv column '{good_col}'")
        return good_ids, df

    cluster_info_path = ks_dir / "cluster_info.tsv"
    if cluster_info_path.exists():
        df = pd.read_csv(cluster_info_path, sep="\t")

        if "cluster_id" in df.columns:
            cluster_col = "cluster_id"
        elif "id" in df.columns:
            cluster_col = "id"
        else:
            raise ValueError(
                f"cluster_info.tsv exists, but expected 'cluster_id' or 'id'. "
                f"Columns are: {list(df.columns)}"
            )

        if "group" not in df.columns:
            raise ValueError(
                f"cluster_info.tsv exists, but expected column 'group'. "
                f"Columns are: {list(df.columns)}"
            )

        good_mask = df["group"].astype(str).str.strip().eq("good")
        good_ids = np.unique(df.loc[good_mask, cluster_col].astype(np.int64).to_numpy())

        if good_ids.size == 0:
            raise ValueError(f"cluster_info.tsv in {ks_dir} contains zero clusters where group == 'good'.")

        msg = (
            f"  WARNING: UnitMetrics.csv not found in {ks_dir}; found {good_ids.size} "
            f"clusters where cluster_info.tsv group == 'good'"
        )
        if interactive and confirm_cluster_info_fallback:
            ok = yes_no(msg + "\nProceed using cluster_info.tsv instead?", default=True)
            if not ok:
                raise RuntimeError("Stopped because UnitMetrics.csv was missing.")
        else:
            print(msg)

        return good_ids, df

    msg = f"No UnitMetrics.csv or cluster_info.tsv found in:\n  {ks_dir}\n"
    if require_file:
        raise FileNotFoundError(
            msg + "Refusing to use all clusters. Add UnitMetrics.csv/cluster_info.tsv or pass --allow-all-clusters."
        )
    if interactive:
        ok = yes_no(msg + "Continue using ALL clusters for this recording?", default=False)
        if not ok:
            raise RuntimeError("Stopped because no good-unit table was found.")
    else:
        print(msg + "Using all clusters because --allow-all-clusters was provided.")

    return None, None


def get_cluster_and_channel_columns(unit_table: pd.DataFrame):
    if "cluster_id" in unit_table.columns:
        cluster_col = "cluster_id"
    elif "id" in unit_table.columns:
        cluster_col = "id"
    else:
        raise ValueError(
            f"Expected cluster ID column 'cluster_id' or 'id'. "
            f"Columns are: {list(unit_table.columns)}"
        )

    if "ch" not in unit_table.columns:
        raise ValueError(
            f"Expected peak-channel column 'ch'. "
            f"Columns are: {list(unit_table.columns)}"
        )

    return cluster_col, "ch"


def label_from_cluster_info_row(row) -> str:
    # Prefer a single explicit combined label if future tools add one.
    for c in ("fsc_label", "region_label", "roi_layer", "regionLayer", "brain_region"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()

    region = ""
    layer = ""

    for c in ("region", "ROI", "roi", "area"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            region = str(row[c]).strip()
            break

    for c in ("layer", "Layer"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            layer = str(row[c]).strip()
            break

    if region and layer:
        return f"{region}_{layer}"
    if region:
        return region
    return ""


# -----------------------------
# Fast KS loader
# -----------------------------
@njit(cache=True)
def _keep_refractory_sorted_by_time(spike_samples, dense_unit_ids, n_units, sample_rate, refractory_s):
    n = spike_samples.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    prev_by_unit = np.full(n_units, -1, dtype=np.int64)
    counts = np.zeros(n_units, dtype=np.int64)

    for i in range(n):
        u = dense_unit_ids[i]
        t = spike_samples[i]
        prev = prev_by_unit[u]

        if prev >= 0:
            dt_s = (float(t) / sample_rate) - (float(prev) / sample_rate)
            if dt_s > refractory_s:
                keep[i] = True
                counts[u] += 1

        prev_by_unit[u] = t

    return keep, counts

def load_kilosort_spikes_for_fsc(
    ks_dir: Path,
    sample_rate: float,
    good_cluster_ids: np.ndarray | None,
    refractory_s: float = DEFAULT_REFRACTORY_S,
):
    spike_times = np.load(ks_dir / "spike_times.npy").reshape(-1).astype(np.int64, copy=False)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").reshape(-1).astype(np.int64, copy=False)

    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError(
            f"spike_times.npy and spike_clusters.npy have different lengths in {ks_dir}: "
            f"{spike_times.shape[0]} vs {spike_clusters.shape[0]}"
        )

    if good_cluster_ids is not None:
        mask = np.isin(spike_clusters, good_cluster_ids)
        spike_times = spike_times[mask]
        spike_clusters = spike_clusters[mask]

    # Kilosort outputs are normally sorted by spike time, but this makes the code safe.
    if spike_times.size > 1 and np.any(spike_times[1:] < spike_times[:-1]):
        order = np.argsort(spike_times, kind="mergesort")
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]

    cluster_ids, dense_unit_ids = np.unique(spike_clusters, return_inverse=True)
    dense_unit_ids = dense_unit_ids.astype(np.int32, copy=False)

    keep, n_spikes_by_cell = _keep_refractory_sorted_by_time(
        spike_times.astype(np.int64, copy=False),
        dense_unit_ids,
        int(cluster_ids.size),
        float(sample_rate),
        float(refractory_s),
    )

    spike_times = spike_times[keep]
    dense_unit_ids = dense_unit_ids[keep].astype(np.int32, copy=False)

    spike_times_s = spike_times.astype(np.float64) / float(sample_rate)

    return spike_times_s, dense_unit_ids, n_spikes_by_cell, cluster_ids


# -----------------------------
# Plotting
# -----------------------------
def get_pair_index_and_direction(source: int, target: int, pair_first, pair_second):
    lo = min(source, target)
    hi = max(source, target)
    hits = np.where((pair_first == lo) & (pair_second == hi))[0]
    if hits.size != 1:
        return None, 1
    pair_idx = int(hits[0])
    direction = 1 if source == lo else -1
    return pair_idx, direction


def format_unit(dense_id: int, cluster_ids, label_by_dense: dict[int, str] | None):
    cluster_id = int(cluster_ids[dense_id]) if cluster_ids is not None and dense_id < len(cluster_ids) else dense_id
    label = ""
    if label_by_dense:
        label = label_by_dense.get(int(dense_id), "")
    if label:
        return f"dense {dense_id} / cluster {cluster_id} / {label}"
    return f"dense {dense_id} / cluster {cluster_id}"


def save_ccg_plot(
    out_path: Path,
    lags,
    ccg_counts,
    pred,
    bounds,
    source: int,
    target: int,
    edge_type: str,
    cluster_ids,
    label_by_dense: dict[int, str] | None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_txt = format_unit(source, cluster_ids, label_by_dense)
    target_txt = format_unit(target, cluster_ids, label_by_dense)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lags * 1000.0, ccg_counts, label="CCG")
    ax.plot(lags * 1000.0, pred, label="Median predictor")
    ax.plot(lags * 1000.0, bounds[:, 0], linestyle="--", label="Upper bound")
    ax.plot(lags * 1000.0, bounds[:, 1], linestyle="--", label="Lower bound")
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axvspan(fsc.MIN_WIN_MONOSYN * 1000, fsc.WIN_MONOSYN * 1000, alpha=0.12)
    ax.set_xlabel("Lag for source → target (ms)")
    ax.set_ylabel("Spike count")
    ax.set_title(f"{edge_type}: {source_txt} → {target_txt}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_ccg_plots(
    output_npz_path: Path,
    plot_dir: Path,
    n_null_plots: int = 20,
    seed: int = 0,
    label_by_dense: dict[int, str] | None = None,
):
    z = np.load(output_npz_path, allow_pickle=True)

    lags = z["time"]
    pair_ccg = z["pair_ccg"]
    pair_pred = z["pair_pred"]
    pair_bounds = z["pair_bounds"]
    pair_first = z["pair_first"]
    pair_second = z["pair_second"]
    valid_pair_indices = z["valid_pair_indices"]
    cluster_ids = z["cluster_ids"] if "cluster_ids" in z else None

    sig_exc = z["sig_exc_con"].reshape(-1, 2) if z["sig_exc_con"].size else np.empty((0, 2), dtype=int)
    sig_inh = z["sig_inh_con"].reshape(-1, 2) if z["sig_inh_con"].size else np.empty((0, 2), dtype=int)

    plotted_directed = set()

    def plot_directed(source, target, edge_type, index):
        pair_idx, direction = get_pair_index_and_direction(source, target, pair_first, pair_second)
        if pair_idx is None:
            return False
        if not np.all(np.isfinite(pair_pred[pair_idx])):
            return False

        if direction == 1:
            x = lags
            ccg_counts = pair_ccg[pair_idx]
            pred = pair_pred[pair_idx]
            bounds = pair_bounds[pair_idx]
        else:
            x = -lags[::-1]
            ccg_counts = pair_ccg[pair_idx][::-1]
            pred = pair_pred[pair_idx][::-1]
            bounds = pair_bounds[pair_idx][::-1]

        safe_type = edge_type.lower().replace(" ", "_")
        fname = f"{safe_type}_{index:03d}_dense{source}_to_dense{target}.png"
        save_ccg_plot(
            plot_dir / fname,
            x,
            ccg_counts,
            pred,
            bounds,
            int(source),
            int(target),
            edge_type,
            cluster_ids,
            label_by_dense,
        )
        plotted_directed.add((int(source), int(target)))
        return True

    n = 0
    for source, target in sig_exc:
        if plot_directed(int(source), int(target), "Excitatory peak", n):
            n += 1

    n = 0
    for source, target in sig_inh:
        if plot_directed(int(source), int(target), "Inhibitory trough", n):
            n += 1

    # Null examples: valid checked pairs with finite pred, excluding significant directed pairs.
    rng = np.random.default_rng(seed)
    valid = np.asarray(valid_pair_indices, dtype=int)
    if valid.size and n_null_plots > 0:
        shuffled = valid.copy()
        rng.shuffle(shuffled)
        count = 0
        for pair_idx in shuffled:
            a = int(pair_first[pair_idx])
            b = int(pair_second[pair_idx])
            if a == b:
                continue
            if (a, b) in plotted_directed or (b, a) in plotted_directed:
                continue
            if not np.all(np.isfinite(pair_pred[pair_idx])):
                continue
            if plot_directed(a, b, "Null example", count):
                count += 1
            if count >= n_null_plots:
                break

    print(f"  saved CCG plots to: {plot_dir}")


def assign_shanks_from_ks_channel_map(
    ks_dir: Path,
    out_dir: Path,
    animal_id: str,
    cluster_ids: np.ndarray,
    unit_table: pd.DataFrame | None,
    shank_split_x: float | None = None,
    interactive: bool = True,
):
    if unit_table is None:
        print("  no UnitMetrics.csv/cluster_info.tsv table available; skipping shank plots")
        return None

    channel_positions_path = ks_dir / "channel_positions.npy"
    if not channel_positions_path.exists():
        print("  channel_positions.npy not found; skipping shank plots")
        return None

    if shank_split_x is None and not interactive:
        print("  no --shank-split-x provided; skipping shank plots")
        return None

    channel_positions = np.load(channel_positions_path)
    n_channels = channel_positions.shape[0]

    channel_map_path = ks_dir / "channel_map.npy"
    if channel_map_path.exists():
        channel_map = np.load(channel_map_path).reshape(-1).astype(int)
    else:
        channel_map = np.arange(n_channels, dtype=int)

    cluster_col, ch_col = get_cluster_and_channel_columns(unit_table)
    channel_id_to_row = {int(ch): i for i, ch in enumerate(channel_map)}

    rows = []
    table_cluster_ids = unit_table[cluster_col].astype(int).to_numpy()

    for cid in cluster_ids:
        hits = np.where(table_cluster_ids == int(cid))[0]
        if hits.size == 0:
            continue

        row = unit_table.iloc[int(hits[0])]
        ch = int(row[ch_col])

        if ch in channel_id_to_row:
            ch_row = channel_id_to_row[ch]
        elif 0 <= ch < n_channels:
            ch_row = ch
        else:
            raise ValueError(f"Could not map cluster {cid} channel {ch} to channel_positions.npy")

        x, y = channel_positions[ch_row]
        rows.append(
            {
                "cluster_id": int(cid),
                "peak_channel": ch,
                "channel_row": int(ch_row),
                "x_um": float(x),
                "y_um": float(y),
            }
        )

    meta = pd.DataFrame(rows)
    if meta.empty:
        print("  no cluster/channel matches found; skipping shank plots")
        return None

    import matplotlib
    if interactive and shank_split_x is None:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.scatter(channel_positions[:, 0], channel_positions[:, 1], s=18, marker="s", alpha=0.35)
    ax.scatter(meta["x_um"], meta["y_um"], s=24, marker="s")
    ax.set_xlabel("x position (um)")
    ax.set_ylabel("y position (um)")
    ax.set_title("KS/Phy channel map: choose x split for shank 1 vs shank 2")
    ax.axis("equal")

    if shank_split_x is not None:
        ax.axvline(float(shank_split_x), linestyle="--", linewidth=1)

    preview_path = out_dir / f"{animal_id}_ks_channel_map_for_shank_split.png"
    fig.savefig(preview_path, dpi=200, bbox_inches="tight")
    print(f"  saved channel map preview: {preview_path}")

    if shank_split_x is None:
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception:
            pass
        shank_split_x = float(input("Enter x split between shank 1 and shank 2 in microns: ").strip())

    meta["shank"] = np.where(meta["x_um"] <= float(shank_split_x), "shank1", "shank2")

    saved_path = out_dir / f"{animal_id}_fsc_unit_shank_labels.csv"
    meta.to_csv(saved_path, index=False)
    print(f"  saved unit shank labels: {saved_path}")

    try:
        plt.close(fig)
    except Exception:
        pass

    return meta

def _edges_to_dense_indices(edges, cluster_ids):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if edges.size == 0:
        return edges

    n = len(cluster_ids)
    if np.all((edges >= 0) & (edges < n)):
        return edges

    cid_to_dense = {int(cid): i for i, cid in enumerate(cluster_ids)}
    dense = []
    for pre, post in edges:
        if int(pre) in cid_to_dense and int(post) in cid_to_dense:
            dense.append([cid_to_dense[int(pre)], cid_to_dense[int(post)]])
    return np.asarray(dense, dtype=int).reshape(-1, 2)


def plot_directed_shank_connectivity_matrix(
    out_dir: Path,
    animal_id: str,
    cluster_ids: np.ndarray,
    unit_meta: pd.DataFrame | None,
    sig_exc_con,
    sig_inh_con=None,
):
    if unit_meta is None:
        return None

    exc_edges = _edges_to_dense_indices(sig_exc_con, cluster_ids)
    inh_edges = (
        _edges_to_dense_indices(sig_inh_con, cluster_ids)
        if sig_inh_con is not None
        else np.empty((0, 2), dtype=int)
    )

    meta = unit_meta.copy()
    cluster_to_dense = {int(cid): i for i, cid in enumerate(cluster_ids)}
    meta["dense_id"] = meta["cluster_id"].astype(int).map(cluster_to_dense)
    meta = meta.dropna(subset=["dense_id"]).copy()
    meta["dense_id"] = meta["dense_id"].astype(int)

    meta = meta.sort_values(["shank", "y_um", "x_um", "cluster_id"]).reset_index(drop=True)

    dense_to_order = {int(d): i for i, d in enumerate(meta["dense_id"].to_numpy())}
    dense_to_xy = dict(zip(meta["dense_id"].astype(int), zip(meta["x_um"], meta["y_um"])))

    all_edges = []
    for pre, post in exc_edges:
        if int(pre) in dense_to_order and int(post) in dense_to_order and int(pre) != int(post):
            all_edges.append((int(pre), int(post), "exc"))
    for pre, post in inh_edges:
        if int(pre) in dense_to_order and int(post) in dense_to_order and int(pre) != int(post):
            all_edges.append((int(pre), int(post), "inh"))

    xs, ys, ds = [], [], []
    for pre, post, _sign in all_edges:
        xs.append(dense_to_order[pre])
        ys.append(dense_to_order[post])

        x1, y1 = dense_to_xy[pre]
        x2, y2 = dense_to_xy[post]
        ds.append(float(np.hypot(x1 - x2, y1 - y2)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    if xs:
        sc = ax.scatter(xs, ys, c=ds, marker="s", s=36)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Distance between peak channels (um)")

    ax.set_xlabel("Presynaptic cells")
    ax.set_ylabel("Postsynaptic cells")
    ax.set_title(f"{animal_id}: directed putative monosynaptic connectivity")

    n = len(meta)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    shank_vals = meta["shank"].to_numpy()
    boundaries = np.where(shank_vals[:-1] != shank_vals[1:])[0] + 0.5
    for b in boundaries:
        ax.axvline(b, color="k", linewidth=1)
        ax.axhline(b, color="k", linewidth=1)

    ax.set_aspect("equal")
    fig.tight_layout()

    out_png = out_dir / f"{animal_id}_directed_shank_connectivity_matrix.png"
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    shanks = list(dict.fromkeys(meta["shank"].to_list()))
    edge_set = {(pre, post) for pre, post, _sign in all_edges}

    for i, s_a in enumerate(shanks):
        ids_a = meta.loc[meta["shank"] == s_a, "dense_id"].astype(int).to_numpy()
        ids_a_set = set(ids_a.tolist())

        for s_b in shanks[i:]:
            ids_b = meta.loc[meta["shank"] == s_b, "dense_id"].astype(int).to_numpy()
            ids_b_set = set(ids_b.tolist())

            if s_a == s_b:
                possible = len(ids_a) * (len(ids_a) - 1)
                detected = sum((pre in ids_a_set) and (post in ids_a_set) and (pre != post) for pre, post in edge_set)
                label = s_a
            else:
                possible = 2 * len(ids_a) * len(ids_b)
                detected = sum(
                    ((pre in ids_a_set) and (post in ids_b_set)) or
                    ((pre in ids_b_set) and (post in ids_a_set))
                    for pre, post in edge_set
                )
                label = f"{s_a}<->{s_b}"

            pct = 100.0 * detected / possible if possible else np.nan
            summary_rows.append(
                {
                    "comparison": label,
                    "n_prepost_possible_directed": int(possible),
                    "n_detected_connections": int(detected),
                    "percent_possible_connections": pct,
                }
            )

    summary = pd.DataFrame(summary_rows)

    out_csv = out_dir / f"{animal_id}_shank_connection_percentages.csv"
    summary.to_csv(out_csv, index=False)

    print(f"  saved directed shank matrix: {out_png}")
    print(f"  saved shank connection percentages: {out_csv}")

    return summary


def plot_cross_shank_geometry_connections(
    ks_dir: Path,
    out_dir: Path,
    animal_id: str,
    cluster_ids: np.ndarray,
    unit_meta: pd.DataFrame | None,
    sig_exc_con,
    sig_inh_con=None,
):
    if unit_meta is None:
        return None

    channel_positions_path = ks_dir / "channel_positions.npy"
    if not channel_positions_path.exists():
        return None

    channel_positions = np.load(channel_positions_path)

    exc_edges = _edges_to_dense_indices(sig_exc_con, cluster_ids)
    inh_edges = (
        _edges_to_dense_indices(sig_inh_con, cluster_ids)
        if sig_inh_con is not None
        else np.empty((0, 2), dtype=int)
    )

    meta = unit_meta.copy()
    cluster_to_dense = {int(cid): i for i, cid in enumerate(cluster_ids)}
    meta["dense_id"] = meta["cluster_id"].astype(int).map(cluster_to_dense)
    meta = meta.dropna(subset=["dense_id"]).copy()
    if meta.empty:
        return None

    meta["dense_id"] = meta["dense_id"].astype(int)
    dense_to_row = {int(r["dense_id"]): r for _, r in meta.iterrows()}

    all_edges = []
    for pre, post in exc_edges:
        all_edges.append((int(pre), int(post), "exc"))
    for pre, post in inh_edges:
        all_edges.append((int(pre), int(post), "inh"))

    cross_edges = []
    for pre, post, sign in all_edges:
        if pre not in dense_to_row or post not in dense_to_row or pre == post:
            continue
        pre_row = dense_to_row[pre]
        post_row = dense_to_row[post]
        if str(pre_row["shank"]) != str(post_row["shank"]):
            cross_edges.append((pre, post, sign))

    shank_counts = meta.groupby("shank")["dense_id"].count().to_dict()
    total_units = int(sum(shank_counts.values()))
    possible_cross_directed = int(sum(n * (total_units - n) for n in shank_counts.values()))
    percent = 100.0 * len(cross_edges) / possible_cross_directed if possible_cross_directed else np.nan

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(6, 9))

    ax.scatter(
        channel_positions[:, 0],
        channel_positions[:, 1],
        s=18,
        marker="s",
        facecolors="none",
        edgecolors="0.65",
        linewidths=0.5,
    )

    ax.scatter(
        meta["x_um"],
        meta["y_um"],
        s=22,
        marker="s",
        linewidths=0.6,
    )

    for pre, post, _sign in cross_edges:
        pre_row = dense_to_row[pre]
        post_row = dense_to_row[post]
        x1, y1 = float(pre_row["x_um"]), float(pre_row["y_um"])
        x2, y2 = float(post_row["x_um"]), float(post_row["y_um"])

        ax.plot([x1, x2], [y1, y2], linewidth=0.7, alpha=0.45)

        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        dx = (x2 - x1) * 0.10
        dy = (y2 - y1) * 0.10

        arrow = FancyArrowPatch(
            (mx - dx, my - dy),
            (mx + dx, my + dy),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.7,
            alpha=0.65,
        )
        ax.add_patch(arrow)

    pct_txt = "nan" if np.isnan(percent) else f"{percent:.2f}"
    ax.set_title(
        f"{animal_id}: cross-shank directed connections\n"
        f"{len(cross_edges)} / {possible_cross_directed} possible = {pct_txt}%"
    )
    ax.set_xlabel("x position (um)")
    ax.set_ylabel("y position (um)")
    ax.axis("equal")
    fig.tight_layout()

    out_png = out_dir / f"{animal_id}_cross_shank_directed_connections.png"
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved cross-shank geometry plot: {out_png}")
    return out_png


# -----------------------------
# FSC runner
# -----------------------------
def build_label_by_dense(cluster_ids, cluster_info_df):
    if cluster_info_df is None:
        return {}

    label_by_cluster = {}
    for _, row in cluster_info_df.iterrows():
        if "cluster_id" not in row:
            continue
        label = label_from_cluster_info_row(row)
        if label:
            label_by_cluster[int(row["cluster_id"])] = label

    return {
        dense_id: label_by_cluster.get(int(cluster_id), "")
        for dense_id, cluster_id in enumerate(cluster_ids)
    }


def run_one_recording(
    ks_dir: Path,
    out_dir: Path,
    animal_id: str,
    sample_rate: float,
    threads: int,
    blocks: int,
    save_plots: bool,
    n_null_plots: int,
    allow_all_clusters_without_cluster_info: bool,
    interactive: bool,
    good_ids: np.ndarray | None = None,
    unit_table: pd.DataFrame | None = None,
    unit_shank_meta: pd.DataFrame | None = None,
    shank_split_x: float | None = None,
):
    start = time.time()

    print(f"\nProcessing: {ks_dir}")
    print(f"  animal/session ID: {animal_id}")

    if good_ids is None and unit_table is None:
        good_ids, unit_table = load_good_cluster_ids(
            ks_dir,
            require_file=not allow_all_clusters_without_cluster_info,
            interactive=interactive,
            confirm_cluster_info_fallback=interactive,
        )

    spike_times_s, unit_ids, n_spikes_by_cell, cluster_ids = load_kilosort_spikes_for_fsc(
        ks_dir=ks_dir,
        sample_rate=sample_rate,
        good_cluster_ids=good_ids,
        refractory_s=DEFAULT_REFRACTORY_S,
    )

    n_cells = int(cluster_ids.size)

    print(f"  n_cells: {n_cells}")
    print(f"  total spikes after close-spike filter: {spike_times_s.size}")

    if unit_shank_meta is None:
        unit_shank_meta = assign_shanks_from_ks_channel_map(
            ks_dir=ks_dir,
            out_dir=out_dir,
            animal_id=animal_id,
            cluster_ids=cluster_ids,
            unit_table=unit_table,
            shank_split_x=shank_split_x,
            interactive=interactive,
        )

    ccg_result = ccg.compute_compact_ccg_from_arrays(
        spike_times_s=spike_times_s,
        unit_ids=unit_ids,
        n_units=n_cells,
        bin_size=fsc.BIN_DUR,
        duration=fsc.WIN_DUR,
        n_threads=threads,
        n_blocks=blocks,
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

    post_idx = fsc._indices_from_mask(post_mask)
    pre_idx = fsc._indices_from_mask(pre_mask)
    zero_idx = fsc._indices_from_mask(zero_mask)
    zero2_idx = fsc._indices_from_mask(zero2_mask)

    non_diagonal = pair_first != pair_second
    enough_counts = np.nanmedian(pair_ccg[:, lags > -0.01], axis=1) > fsc.SPIKE_COUNT_CCG_THRESHOLD
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

        pvals_by_pair, pred_by_pair, qvals_by_pair = fsc.fast_cch_conv_median_numba(selected_pair_ccg, fsc.W)

        pair_pval[valid_pair_indices, :] = pvals_by_pair
        pair_pred[valid_pair_indices, :] = pred_by_pair

        n_bonferroni = np.ceil(fsc.WIN_MONOSYN / fsc.BIN_DUR) * 2
        hi_by_pair, lo_by_pair = fsc.cached_poisson_bounds(pred_by_pair, fsc.ALPHA, n_bonferroni)

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
            n_cells,
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
    else:
        pcausal_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
        syn_strength_exc = np.zeros((n_cells, n_cells), dtype=np.float64)
        syn_ratio_exc = np.zeros((n_cells, n_cells), dtype=np.float64)

    sig_con_exc = np.asarray(sig_con_exc, dtype=int).reshape(-1, 2)
    sig_con_inh = np.asarray(sig_con_inh, dtype=int).reshape(-1, 2)

    label_by_dense = build_label_by_dense(cluster_ids, unit_table)

    metadata = pd.DataFrame(
        {
            "dense_unit_id": np.arange(n_cells, dtype=np.int32),
            "cluster_id": cluster_ids.astype(np.int64),
            "n_spikes_after_close_filter": n_spikes_by_cell,
            "label": [label_by_dense.get(i, "") for i in range(n_cells)],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{animal_id}-fast-fsc.npz"

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
        cluster_ids=cluster_ids,
        unit_metadata=metadata.to_dict("list"),
        n_pairs_checked=int(valid_pair_indices.size),
        pair_ccg=pair_ccg,
        pair_pred=pair_pred,
        pair_pval=pair_pval,
        pair_bounds=pair_bounds,
        pair_first=pair_first,
        pair_second=pair_second,
        valid_pair_indices=valid_pair_indices,
        ccg_elapsed=float(ccg_result["elapsed"]),
        sample_rate=float(sample_rate),
        bin_dur=fsc.BIN_DUR,
        win_dur=fsc.WIN_DUR,
        n_threads=int(threads),
        n_blocks=int(blocks),
    )

    metadata.to_csv(out_dir / f"{animal_id}-fast-fsc-unit-metadata.csv", index=False)

    plot_directed_shank_connectivity_matrix(
        out_dir=out_dir,
        animal_id=animal_id,
        cluster_ids=cluster_ids,
        unit_meta=unit_shank_meta,
        sig_exc_con=sig_con_exc,
        sig_inh_con=sig_con_inh,
    )

    plot_cross_shank_geometry_connections(
        ks_dir=ks_dir,
        out_dir=out_dir,
        animal_id=animal_id,
        cluster_ids=cluster_ids,
        unit_meta=unit_shank_meta,
        sig_exc_con=sig_con_exc,
        sig_inh_con=sig_con_inh,
    )

    print(f"  excitatory edges: {sig_con_exc.tolist()}")
    print(f"  inhibitory edges: {sig_con_inh.tolist()}")
    print(f"  saved: {out_path}")

    if save_plots:
        plot_dir = out_dir / f"{animal_id}_ccg_plots"
        save_ccg_plots(
            out_path,
            plot_dir=plot_dir,
            n_null_plots=n_null_plots,
            label_by_dense=label_by_dense,
        )

    print(f"  elapsed: {(time.time() - start) / 60.0:.2f} min")
    return out_path


def _normalize_shank_splits(shank_split_x, n: int):
    if shank_split_x is None:
        return [None] * n

    if isinstance(shank_split_x, (int, float)):
        return [float(shank_split_x)] * n

    vals = list(shank_split_x)
    if len(vals) == 1 and n > 1:
        return [float(vals[0])] * n
    if len(vals) != n:
        raise ValueError("shank_split_x must be None, one value, or one value per KS directory.")
    return [None if v is None else float(v) for v in vals]


def validate_ks_dir_basic(ks_dir: Path):
    if not ks_dir.exists():
        raise FileNotFoundError(f"Kilosort directory not found: {ks_dir}")
    if not (ks_dir / "spike_times.npy").exists():
        raise FileNotFoundError(f"Missing spike_times.npy in {ks_dir}")
    if not (ks_dir / "spike_clusters.npy").exists():
        raise FileNotFoundError(f"Missing spike_clusters.npy in {ks_dir}")


def prepare_recording_inputs(
    ks_dir: Path,
    out_dir: Path,
    animal_id: str,
    allow_all_clusters_without_cluster_info: bool,
    interactive: bool,
    shank_split_x: float | None,
):
    validate_ks_dir_basic(ks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    good_ids, unit_table = load_good_cluster_ids(
        ks_dir,
        require_file=not allow_all_clusters_without_cluster_info,
        interactive=interactive,
        confirm_cluster_info_fallback=interactive,
    )

    unit_shank_meta = None
    if unit_table is not None:
        unit_shank_meta = assign_shanks_from_ks_channel_map(
            ks_dir=ks_dir,
            out_dir=out_dir,
            animal_id=animal_id,
            cluster_ids=good_ids,
            unit_table=unit_table,
            shank_split_x=shank_split_x,
            interactive=interactive,
        )

    return {
        "ks_dir": ks_dir,
        "out_dir": out_dir,
        "animal_id": animal_id,
        "good_ids": good_ids,
        "unit_table": unit_table,
        "unit_shank_meta": unit_shank_meta,
        "shank_split_x": shank_split_x,
    }


def run_batch(
    ks_dirs,
    animal_ids=None,
    out_root: str | Path | None = None,
    shank_split_x=None,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    threads: int = DEFAULT_THREADS,
    blocks: int = DEFAULT_BLOCKS,
    save_plots: bool = False,
    n_null_plots: int = 20,
    allow_all_clusters_without_cluster_info: bool = False,
    interactive: bool = False,
):
    ks_dirs = [Path(x).expanduser().resolve() for x in ks_dirs]

    if animal_ids is None:
        animal_ids = [d.name for d in ks_dirs]
    elif len(animal_ids) != len(ks_dirs):
        raise ValueError("animal_ids must be None or have one entry per KS directory.")

    out_root_path = None if out_root is None else Path(out_root).expanduser().resolve()
    if out_root_path is not None:
        out_root_path.mkdir(parents=True, exist_ok=True)

    shank_splits = _normalize_shank_splits(shank_split_x, len(ks_dirs))

    plans = []
    for ks_dir, animal_id, split_x in zip(ks_dirs, animal_ids, shank_splits):
        out_dir = output_dir_for_ks_dir(ks_dir, out_root_path, animal_id)
        plans.append(
            prepare_recording_inputs(
                ks_dir=ks_dir,
                out_dir=out_dir,
                animal_id=animal_id,
                allow_all_clusters_without_cluster_info=allow_all_clusters_without_cluster_info,
                interactive=interactive,
                shank_split_x=split_x,
            )
        )

    outputs = []
    for plan in plans:
        outputs.append(
            run_one_recording(
                ks_dir=plan["ks_dir"],
                out_dir=plan["out_dir"],
                animal_id=plan["animal_id"],
                sample_rate=sample_rate,
                threads=threads,
                blocks=blocks,
                save_plots=save_plots,
                n_null_plots=n_null_plots,
                allow_all_clusters_without_cluster_info=allow_all_clusters_without_cluster_info,
                interactive=interactive,
                good_ids=plan["good_ids"],
                unit_table=plan["unit_table"],
                unit_shank_meta=plan["unit_shank_meta"],
                shank_split_x=plan["shank_split_x"],
            )
        )

    return outputs


# -----------------------------
# Entry points
# -----------------------------
def interactive_main():
    ks_dirs = choose_ks_dirs()
    out_root = choose_output_root()
    animal_ids = ask_animal_ids(ks_dirs)

    sample_rate_txt = input("\nSample rate in Hz? Press enter for 30000.\n\n").strip()
    sample_rate = DEFAULT_SAMPLE_RATE if sample_rate_txt == "" else float(sample_rate_txt)

    save_plots = yes_no("\nSave CCG plots for significant pairs plus null examples?", default=True)
    n_null_plots = 20
    if save_plots:
        n_txt = input("\nHow many null/non-significant example pairs should be plotted? Press enter for 20.\n\n").strip()
        n_null_plots = 20 if n_txt == "" else int(n_txt)

    print("\nChecking files and collecting shank split inputs before FSC computation.")
    plans = []
    for ks_dir, animal_id in zip(ks_dirs, animal_ids):
        out_dir = output_dir_for_ks_dir(ks_dir, out_root, animal_id)
        plans.append(
            prepare_recording_inputs(
                ks_dir=ks_dir,
                out_dir=out_dir,
                animal_id=animal_id,
                allow_all_clusters_without_cluster_info=True,
                interactive=True,
                shank_split_x=None,
            )
        )

    print("\nInputs collected. Starting FSC processing.")
    start_all = time.time()

    for plan in plans:
        run_one_recording(
            ks_dir=plan["ks_dir"],
            out_dir=plan["out_dir"],
            animal_id=plan["animal_id"],
            sample_rate=sample_rate,
            threads=DEFAULT_THREADS,
            blocks=DEFAULT_BLOCKS,
            save_plots=save_plots,
            n_null_plots=n_null_plots,
            allow_all_clusters_without_cluster_info=True,
            interactive=True,
            good_ids=plan["good_ids"],
            unit_table=plan["unit_table"],
            unit_shank_meta=plan["unit_shank_meta"],
            shank_split_x=plan["shank_split_x"],
        )

    print(f"\nFinished all directories in {(time.time() - start_all) / 60.0:.2f} min")

def cli_main(args):
    ks_dirs = [Path(x).expanduser().resolve() for x in args.ks_dir]
    if args.animal_id:
        if len(args.animal_id) != len(ks_dirs):
            raise ValueError("--animal-id must be provided once per --ks-dir")
        animal_ids = args.animal_id
    else:
        animal_ids = [d.name for d in ks_dirs]

    out_root = None if args.same_output_dir or args.out_root is None else Path(args.out_root).expanduser().resolve()
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    shank_splits = _normalize_shank_splits(args.shank_split_x, len(ks_dirs))

    plans = []
    for ks_dir, animal_id, split_x in zip(ks_dirs, animal_ids, shank_splits):
        out_dir = output_dir_for_ks_dir(ks_dir, out_root, animal_id)
        plans.append(
            prepare_recording_inputs(
                ks_dir=ks_dir,
                out_dir=out_dir,
                animal_id=animal_id,
                allow_all_clusters_without_cluster_info=args.allow_all_clusters,
                interactive=False,
                shank_split_x=split_x,
            )
        )

    for plan in plans:
        run_one_recording(
            ks_dir=plan["ks_dir"],
            out_dir=plan["out_dir"],
            animal_id=plan["animal_id"],
            sample_rate=args.sample_rate,
            threads=args.threads,
            blocks=args.blocks,
            save_plots=args.save_plots,
            n_null_plots=args.n_null_plots,
            allow_all_clusters_without_cluster_info=args.allow_all_clusters,
            interactive=False,
            good_ids=plan["good_ids"],
            unit_table=plan["unit_table"],
            unit_shank_meta=plan["unit_shank_meta"],
            shank_split_x=plan["shank_split_x"],
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks-dir", action="append", help="Kilosort output directory. Can be repeated.")
    parser.add_argument("--animal-id", action="append", help="Animal/session ID. Provide once per --ks-dir.")
    parser.add_argument("--out-root", default=None, help="Optional global output root. If set, outputs save to <out-root>/<animal-id>/. Default saves beside each KS directory.")
    parser.add_argument("--same-output-dir", action="store_true", help="Force saving output beside each input KS directory, even if --out-root is set.")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--allow-all-clusters", action="store_true", help="Allow all clusters if cluster_info.tsv is missing.")
    parser.add_argument("--save-plots", action="store_true", help="Save CCG plots for significant pairs and null examples.")
    parser.add_argument("--n-null-plots", type=int, default=20)
    parser.add_argument("--shank-split-x", action="append", type=float, help="x-coordinate split in microns for shank1/shank2. Can be repeated once per --ks-dir, or one value can be reused for all.")

    args = parser.parse_args()

    if args.ks_dir:
        cli_main(args)
    else:
        interactive_main()


if __name__ == "__main__":
    main()
