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


def _parse_split_values(text: str) -> list[float]:
    vals = text.replace(",", " ").split()
    return [float(v) for v in vals]


def _default_region_names(n_regions: int) -> list[str]:
    return [f"shank{i + 1}" for i in range(n_regions)]


def _validate_region_names(region_names, n_regions: int):
    if region_names is None:
        return _default_region_names(n_regions)

    names = [str(x).strip() for x in region_names if str(x).strip()]
    if len(names) != n_regions:
        raise ValueError(
            f"Expected {n_regions} region names for {n_regions - 1} split values, "
            f"but got {len(names)}: {names}"
        )
    return names


def load_full_channel_geometry(ks_dir: Path) -> pd.DataFrame | None:
    """Load full physical channel geometry written by the merge script, if present.

    This file is preferred for background probe-site squares because KS output
    channel_positions.npy can be pruned/reindexed after dead-channel removal.
    """
    path = Path(ks_dir) / "full_channel_geometry.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"x_um", "y_um"}
    if not required.issubset(df.columns):
        print(f"  WARNING: {path} exists but lacks x_um/y_um; ignoring full channel geometry")
        return None

    df = df.dropna(subset=["x_um", "y_um"]).copy()
    if df.empty:
        print(f"  WARNING: {path} has no valid x_um/y_um rows; ignoring full channel geometry")
        return None

    df["x_um"] = df["x_um"].astype(float)
    df["y_um"] = df["y_um"].astype(float)
    return df


def assign_shanks_from_ks_channel_map(
    ks_dir: Path,
    out_dir: Path,
    animal_id: str,
    cluster_ids: np.ndarray,
    unit_table: pd.DataFrame | None,
    shank_split_x=None,
    shank_region_names=None,
    interactive: bool = True,
):
    if unit_table is None:
        print("  no UnitMetrics.csv/cluster_info.tsv table available; skipping shank/region plots")
        return None

    if shank_split_x is None and not interactive:
        print("  no --shank-split-x provided; skipping shank/region plots")
        return None

    if shank_split_x is None:
        split_values = None
    elif isinstance(shank_split_x, (int, float)):
        split_values = [float(shank_split_x)]
    else:
        split_values = sorted(float(x) for x in shank_split_x)

    if split_values is not None:
        region_names = _validate_region_names(shank_region_names, len(split_values) + 1)
    else:
        region_names = None

    channel_positions_path = ks_dir / "channel_positions.npy"
    channel_positions = None
    if channel_positions_path.exists():
        channel_positions = np.load(channel_positions_path)

    full_channel_geometry = load_full_channel_geometry(ks_dir)

    # Preferred path for merged Winny-style probe analyses: UnitMetrics.csv carries
    # physical x/y from chmapfull.mat via unitdata.channelnum. This avoids KS/Phy
    # dead-channel reindexing artifacts in distance calculations.
    has_unit_xy = {"x_um", "y_um"}.issubset(set(unit_table.columns))

    cluster_col = None
    ch_col = None
    if has_unit_xy:
        if "cluster_id" in unit_table.columns:
            cluster_col = "cluster_id"
        elif "id" in unit_table.columns:
            cluster_col = "id"
        else:
            raise ValueError(
                f"Expected cluster ID column 'cluster_id' or 'id'. Columns are: {list(unit_table.columns)}"
            )
        if "ch" in unit_table.columns:
            ch_col = "ch"
    else:
        if channel_positions is None:
            print("  channel_positions.npy not found and UnitMetrics.csv has no x_um/y_um; skipping shank/region plots")
            return None
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
        ch = int(row[ch_col]) if ch_col is not None and pd.notna(row[ch_col]) else -1

        if has_unit_xy:
            x = float(row["x_um"])
            y = float(row["y_um"])
            ch_row = -1
        else:
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
                "coordinate_source": "UnitMetrics.csv x_um/y_um" if has_unit_xy else "channel_positions.npy",
            }
        )

    meta = pd.DataFrame(rows)
    if meta.empty:
        print("  no cluster/channel matches found; skipping shank/region plots")
        return None

    import matplotlib
    if interactive and split_values is None:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if full_channel_geometry is not None:
        bg_x = full_channel_geometry["x_um"].to_numpy(dtype=float)
        bg_y = full_channel_geometry["y_um"].to_numpy(dtype=float)
        bg_source = "full_channel_geometry.csv"
    elif channel_positions is not None:
        bg_x = channel_positions[:, 0]
        bg_y = channel_positions[:, 1]
        bg_source = "channel_positions.npy"
    else:
        bg_x = meta["x_um"].to_numpy(dtype=float)
        bg_y = meta["y_um"].to_numpy(dtype=float)
        bg_source = "unit coordinates only"

    plot_x = bg_x
    plot_y = bg_y
    x_span = float(np.ptp(plot_x)) if len(plot_x) else 1.0
    y_span = float(np.ptp(plot_y)) if len(plot_y) else 1.0
    fig_w = max(5.0, min(28.0, 5.0 * max(1.0, x_span / max(y_span, 1.0))))
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    ax.scatter(bg_x, bg_y, s=18, marker="s", alpha=0.35)
    ax.scatter(meta["x_um"], meta["y_um"], s=24, marker="s")
    ax.set_xlabel("x position (um)")
    ax.set_ylabel("y position (um)")
    ax.set_title(f"Probe geometry: choose x split(s) for shank/region labels\nbackground sites: {bg_source}")
    ax.axis("equal")

    if split_values is not None:
        for sx in split_values:
            ax.axvline(float(sx), linestyle="--", linewidth=1)

    preview_path = out_dir / f"{animal_id}_ks_channel_map_for_shank_split.png"
    fig.savefig(preview_path, dpi=200, bbox_inches="tight")
    print(f"  saved channel map preview: {preview_path}")

    if split_values is None:
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception:
            pass

        txt = input(
            "Enter one or more x split values in microns. "
            "Example for 4 shanks: 100, 1000, 2100\n\n"
        ).strip()
        split_values = sorted(_parse_split_values(txt))
        if not split_values:
            raise ValueError("At least one x split value is required for shank/region plotting.")

        default_names = _default_region_names(len(split_values) + 1)
        names_txt = input(
            "Optional region names from left to right, separated by commas/spaces.\n"
            f"Press enter for default: {', '.join(default_names)}\n\n"
        ).strip()
        if names_txt:
            region_names = _validate_region_names(names_txt.replace(",", " ").split(), len(split_values) + 1)
        else:
            region_names = default_names

    for sx in split_values:
        ax.axvline(float(sx), linestyle="--", linewidth=1)
    fig.savefig(preview_path, dpi=200, bbox_inches="tight")

    region_idx = np.searchsorted(np.asarray(split_values, dtype=float), meta["x_um"].to_numpy(dtype=float), side="right")
    meta["region_index"] = region_idx.astype(int)
    meta["shank"] = [region_names[int(i)] for i in region_idx]
    meta["region_label"] = meta["shank"]

    saved_path = out_dir / f"{animal_id}_fsc_unit_shank_labels.csv"
    meta.to_csv(saved_path, index=False)
    print(f"  saved unit shank/region labels: {saved_path}")

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

    sort_cols = ["shank", "y_um", "x_um", "cluster_id"]
    if "region_index" in meta.columns:
        sort_cols = ["region_index", "y_um", "x_um", "cluster_id"]
    meta = meta.sort_values(sort_cols).reset_index(drop=True)

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

    tick_pos = []
    tick_labels = []
    for label in list(dict.fromkeys(shank_vals.tolist())):
        inds = np.where(shank_vals == label)[0]
        if inds.size:
            tick_pos.append(float((inds[0] + inds[-1]) / 2.0))
            tick_labels.append(str(label))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=7)

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
    channel_positions = np.load(channel_positions_path) if channel_positions_path.exists() else None
    full_channel_geometry = load_full_channel_geometry(ks_dir)

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

    if full_channel_geometry is not None:
        bg_x = full_channel_geometry["x_um"].to_numpy(dtype=float)
        bg_y = full_channel_geometry["y_um"].to_numpy(dtype=float)
        bg_source = "full_channel_geometry.csv"
    elif channel_positions is not None:
        bg_x = channel_positions[:, 0]
        bg_y = channel_positions[:, 1]
        bg_source = "channel_positions.npy"
    else:
        bg_x = meta["x_um"].to_numpy(dtype=float)
        bg_y = meta["y_um"].to_numpy(dtype=float)
        bg_source = "unit coordinates only"

    plot_x = bg_x
    plot_y = bg_y
    x_span = float(np.ptp(plot_x)) if len(plot_x) else 1.0
    y_span = float(np.ptp(plot_y)) if len(plot_y) else 1.0
    fig_w = max(6.0, min(28.0, 7.0 * max(1.0, x_span / max(y_span, 1.0))))
    fig, ax = plt.subplots(figsize=(fig_w, 9))

    ax.scatter(
        bg_x,
        bg_y,
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
        f"{animal_id}: cross-region directed connections\n"
        f"{len(cross_edges)} / {possible_cross_directed} possible = {pct_txt}%\n"
        f"background sites: {bg_source}"
    )
    ax.set_xlabel("x position (um)")
    ax.set_ylabel("y position (um)")
    ax.axis("equal")
    fig.tight_layout()

    out_png = out_dir / f"{animal_id}_cross_region_directed_connections.png"
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved cross-region geometry plot: {out_png}")
    return out_png


def save_directed_pair_distance_table(
    out_dir: Path,
    animal_id: str,
    cluster_ids: np.ndarray,
    unit_meta: pd.DataFrame | None,
    sig_exc_con,
    sig_inh_con=None,
):
    """
    Save one row per possible directed unit pair with physical distance and detection status.

    This is the source table for distance-binned FSC probability plots. It uses the
    unit x_um/y_um coordinates already assigned from UnitMetrics.csv/chmapfull when
    available, so it is independent of KS dead-channel reindexing.
    """
    if unit_meta is None:
        return None

    meta = unit_meta.copy()
    needed = {"cluster_id", "x_um", "y_um", "shank"}
    if not needed.issubset(set(meta.columns)):
        print("  pair-distance table: unit metadata missing cluster_id/x_um/y_um/shank; skipping")
        return None

    cluster_to_dense = {int(cid): i for i, cid in enumerate(cluster_ids)}
    meta["dense_id"] = meta["cluster_id"].astype(int).map(cluster_to_dense)
    meta = meta.dropna(subset=["dense_id", "x_um", "y_um", "shank"]).copy()
    if meta.empty:
        print("  pair-distance table: no valid units after metadata/dense-id matching; skipping")
        return None

    meta["dense_id"] = meta["dense_id"].astype(int)
    sort_cols = ["shank", "y_um", "x_um", "cluster_id"]
    if "region_index" in meta.columns:
        sort_cols = ["region_index", "y_um", "x_um", "cluster_id"]
    meta = meta.sort_values(sort_cols).reset_index(drop=True)

    dense_ids = meta["dense_id"].astype(int).to_numpy()
    cluster_ids_ordered = meta["cluster_id"].astype(int).to_numpy()
    x = meta["x_um"].astype(float).to_numpy()
    y = meta["y_um"].astype(float).to_numpy()
    shank = meta["shank"].astype(str).to_numpy()

    n = len(meta)
    if n < 2:
        print("  pair-distance table: fewer than 2 units; skipping")
        return None

    pre_idx = np.repeat(np.arange(n, dtype=np.int32), n)
    post_idx = np.tile(np.arange(n, dtype=np.int32), n)
    keep = pre_idx != post_idx
    pre_idx = pre_idx[keep]
    post_idx = post_idx[keep]

    dx = x[pre_idx] - x[post_idx]
    dy = y[pre_idx] - y[post_idx]
    dist = np.sqrt(dx * dx + dy * dy)

    exc_edges = _edges_to_dense_indices(sig_exc_con, cluster_ids)
    inh_edges = (
        _edges_to_dense_indices(sig_inh_con, cluster_ids)
        if sig_inh_con is not None
        else np.empty((0, 2), dtype=int)
    )
    exc_set = {(int(a), int(b)) for a, b in exc_edges if int(a) != int(b)}
    inh_set = {(int(a), int(b)) for a, b in inh_edges if int(a) != int(b)}

    pre_dense = dense_ids[pre_idx]
    post_dense = dense_ids[post_idx]
    is_exc = np.fromiter(((int(a), int(b)) in exc_set for a, b in zip(pre_dense, post_dense)), dtype=bool, count=len(pre_dense))
    is_inh = np.fromiter(((int(a), int(b)) in inh_set for a, b in zip(pre_dense, post_dense)), dtype=bool, count=len(pre_dense))
    is_any = is_exc | is_inh

    pre_region = shank[pre_idx]
    post_region = shank[post_idx]

    categories = []
    unordered_comparisons = []
    directed_comparisons = []
    same_region = []
    same_probe = []
    for a, b in zip(pre_region, post_region):
        a = str(a)
        b = str(b)
        directed_comparisons.append(f"{a}->{b}")
        if a <= b:
            unordered_comparisons.append(a if a == b else f"{a}<->{b}")
        else:
            unordered_comparisons.append(b if a == b else f"{b}<->{a}")

        if a == b:
            categories.append("within_region")
            same_region.append(True)
            same_probe.append(True)
        else:
            pa = _region_probe_prefix(a)
            pb = _region_probe_prefix(b)
            same_region.append(False)
            if pa and pb and pa == pb:
                categories.append("same_probe_between_regions")
                same_probe.append(True)
            elif pa and pb and pa != pb:
                categories.append("different_probe_between_regions")
                same_probe.append(False)
            else:
                categories.append("between_regions")
                same_probe.append(False)

    out = pd.DataFrame(
        {
            "animal_id": animal_id,
            "pre_dense_id": pre_dense.astype(int),
            "post_dense_id": post_dense.astype(int),
            "pre_cluster_id": cluster_ids_ordered[pre_idx].astype(int),
            "post_cluster_id": cluster_ids_ordered[post_idx].astype(int),
            "pre_region": pre_region,
            "post_region": post_region,
            "comparison": unordered_comparisons,
            "directed_comparison": directed_comparisons,
            "comparison_category": categories,
            "same_region": same_region,
            "same_probe_prefix": same_probe,
            "pre_x_um": x[pre_idx],
            "pre_y_um": y[pre_idx],
            "post_x_um": x[post_idx],
            "post_y_um": y[post_idx],
            "distance_um": dist,
            "is_detected_exc": is_exc,
            "is_detected_inh": is_inh,
            "is_detected_any": is_any,
        }
    )

    out_path = Path(out_dir) / f"{animal_id}_directed_pair_distance_table.csv"
    out.to_csv(out_path, index=False)
    print(f"  saved directed pair distance table: {out_path}")
    return out_path


def _wilson_ci(k, n, z=1.959963984540054):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    lo = np.full_like(k, np.nan, dtype=float)
    hi = np.full_like(k, np.nan, dtype=float)
    valid = n > 0
    if not np.any(valid):
        return lo, hi
    phat = k[valid] / n[valid]
    denom = 1.0 + (z * z) / n[valid]
    center = (phat + (z * z) / (2.0 * n[valid])) / denom
    half = (z / denom) * np.sqrt((phat * (1.0 - phat) / n[valid]) + (z * z) / (4.0 * n[valid] * n[valid]))
    lo[valid] = np.maximum(0.0, center - half)
    hi[valid] = np.minimum(1.0, center + half)
    return lo, hi


def _distance_bin_edges_for_values(values_um, tail_only=False):
    values = np.asarray(values_um, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0, 1.0])

    max_v = float(np.nanmax(values))
    if tail_only:
        min_v = max(0.0, float(np.nanmin(values)))
        start = np.floor(min_v / 100.0) * 100.0
        stop = np.ceil(max_v / 100.0) * 100.0 + 100.0
        return np.arange(start, stop + 0.5, 100.0)

    template = np.asarray(
        [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
         1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 5000],
        dtype=float,
    )
    if max_v <= template[-1]:
        return template[template <= max_v + 500.0]

    extra = np.arange(5500.0, np.ceil(max_v / 500.0) * 500.0 + 1000.0, 500.0)
    return np.concatenate([template, extra])


def _coerce_bool_series(series):
    if series.dtype == bool:
        return series.astype(bool)
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _bin_distance_probability(df: pd.DataFrame, bin_edges, min_possible_per_bin: int = 1):
    if df.empty:
        return pd.DataFrame()

    bin_edges = np.asarray(bin_edges, dtype=float)
    work = df.copy()
    work = work[np.isfinite(work["distance_um"].astype(float))].copy()
    if work.empty:
        return pd.DataFrame()

    work["distance_bin"] = pd.cut(
        work["distance_um"].astype(float),
        bins=bin_edges,
        right=False,
        include_lowest=True,
    )

    rows = []
    for interval, g in work.groupby("distance_bin", observed=True):
        if pd.isna(interval):
            continue
        possible = int(len(g))
        if possible < int(min_possible_per_bin):
            continue
        detected = int(_coerce_bool_series(g["is_detected_any"]).sum())
        lo, hi = _wilson_ci(np.asarray([detected]), np.asarray([possible]))
        rows.append(
            {
                "distance_bin_low_um": float(interval.left),
                "distance_bin_high_um": float(interval.right),
                "distance_bin_center_um": float((interval.left + interval.right) / 2.0),
                "n_possible_directed": possible,
                "n_detected": detected,
                "connection_probability": float(detected / possible) if possible else np.nan,
                "connection_probability_percent": float(100.0 * detected / possible) if possible else np.nan,
                "wilson_ci_low_percent": float(100.0 * lo[0]),
                "wilson_ci_high_percent": float(100.0 * hi[0]),
                "n_animals_contributing": int(g["animal_id"].nunique()) if "animal_id" in g.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_aggregate_distance_probability_plots(plans, aggregate_out_dir: Path, prefix: str = "aggregate_distance"):
    """
    Pool directed pair distance tables across animals and plot connection probability by distance.

    Produces one figure with two panels:
      A) all possible directed pairs across the full distance range
      B) the far-tail subset: different-probe/between-region pairs only
    """
    rows = []
    for plan in plans:
        animal_id = plan["animal_id"]
        out_dir = Path(plan["out_dir"])
        path = out_dir / f"{animal_id}_directed_pair_distance_table.csv"
        if not path.exists():
            print(f"  aggregate distance: missing {path}; skipping")
            continue
        df = pd.read_csv(path)
        needed = {"distance_um", "is_detected_any", "comparison_category"}
        if not needed.issubset(set(df.columns)):
            print(f"  aggregate distance: {path} missing needed columns; skipping")
            continue
        rows.append(df)

    if not rows:
        print("  aggregate distance: no directed pair distance tables found")
        return None

    all_pairs = pd.concat(rows, ignore_index=True)
    all_pairs["is_detected_any"] = _coerce_bool_series(all_pairs["is_detected_any"])

    aggregate_out_dir = Path(aggregate_out_dir)
    aggregate_out_dir.mkdir(parents=True, exist_ok=True)

    all_rows_path = aggregate_out_dir / f"{prefix}_all_directed_pair_rows.csv"
    all_pairs.to_csv(all_rows_path, index=False)

    all_edges = _distance_bin_edges_for_values(all_pairs["distance_um"].to_numpy(dtype=float), tail_only=False)
    all_binned = _bin_distance_probability(all_pairs, all_edges, min_possible_per_bin=1)
    all_binned_path = aggregate_out_dir / f"{prefix}_binned_probability_all_pairs.csv"
    all_binned.to_csv(all_binned_path, index=False)

    tail_pairs = all_pairs.loc[all_pairs["comparison_category"].astype(str) == "different_probe_between_regions"].copy()
    tail_binned = pd.DataFrame()
    tail_binned_path = None
    if not tail_pairs.empty:
        tail_edges = _distance_bin_edges_for_values(tail_pairs["distance_um"].to_numpy(dtype=float), tail_only=True)
        tail_binned = _bin_distance_probability(tail_pairs, tail_edges, min_possible_per_bin=1)
        tail_binned_path = aggregate_out_dir / f"{prefix}_binned_probability_different_probe_tail.csv"
        tail_binned.to_csv(tail_binned_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    if not all_binned.empty:
        y = all_binned["connection_probability_percent"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - all_binned["wilson_ci_low_percent"].to_numpy(dtype=float),
            all_binned["wilson_ci_high_percent"].to_numpy(dtype=float) - y,
        ])
        ax.errorbar(
            all_binned["distance_bin_center_um"].to_numpy(dtype=float),
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=2,
        )
    ax.set_xlabel("Distance between unit peak channels (um)")
    ax.set_ylabel("Detected directed pairs (% of possible)")
    ax.set_title("All directed pair opportunities")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    if not tail_binned.empty:
        y = tail_binned["connection_probability_percent"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - tail_binned["wilson_ci_low_percent"].to_numpy(dtype=float),
            tail_binned["wilson_ci_high_percent"].to_numpy(dtype=float) - y,
        ])
        ax.errorbar(
            tail_binned["distance_bin_center_um"].to_numpy(dtype=float),
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=2,
        )
    ax.set_xlabel("Distance between unit peak channels (um)")
    ax.set_ylabel("Detected directed pairs (% of possible)")
    ax.set_title("Far tail: different-probe pairs only")
    ax.grid(True, alpha=0.25)

    fig.suptitle("FSC connection probability falls with physical/anatomical separation", fontsize=13)
    fig.tight_layout()
    plot_path = aggregate_out_dir / f"{prefix}_probability_by_distance.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log_plot_path = None
    if not all_binned.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        for ax, binned, title in [
            (axes[0], all_binned, "All directed pair opportunities"),
            (axes[1], tail_binned, "Far tail: different-probe pairs only"),
        ]:
            if binned is not None and not binned.empty:
                y = binned["connection_probability_percent"].to_numpy(dtype=float)
                positive = y > 0
                if np.any(positive):
                    ax.plot(
                        binned.loc[positive, "distance_bin_center_um"].to_numpy(dtype=float),
                        y[positive],
                        marker="o",
                        linewidth=1.4,
                    )
                    ax.set_yscale("log")
            ax.set_xlabel("Distance between unit peak channels (um)")
            ax.set_ylabel("Detected directed pairs (% of possible, log scale)")
            ax.set_title(title)
            ax.grid(True, alpha=0.25, which="both")
        fig.suptitle("FSC connection probability by distance, log y-axis", fontsize=13)
        fig.tight_layout()
        log_plot_path = aggregate_out_dir / f"{prefix}_probability_by_distance_logy.png"
        fig.savefig(log_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved aggregate directed pair rows: {all_rows_path}")
    print(f"  saved aggregate distance bins: {all_binned_path}")
    if tail_binned_path is not None:
        print(f"  saved aggregate different-probe tail bins: {tail_binned_path}")
    print(f"  saved aggregate distance plot: {plot_path}")
    if log_plot_path is not None:
        print(f"  saved aggregate log-y distance plot: {log_plot_path}")

    return {
        "all_pair_rows_csv": all_rows_path,
        "all_binned_csv": all_binned_path,
        "tail_binned_csv": tail_binned_path,
        "distance_plot_png": plot_path,
        "distance_log_plot_png": log_plot_path,
    }


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
    shank_split_x=None,
    shank_region_names=None,
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
            shank_region_names=shank_region_names,
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

    save_directed_pair_distance_table(
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


def _is_scalar_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _normalize_shank_splits(shank_split_x, n: int):
    if shank_split_x is None:
        return [None] * n

    if _is_scalar_number(shank_split_x):
        return [[float(shank_split_x)] for _ in range(n)]

    vals = list(shank_split_x)
    if not vals:
        return [None] * n

    # argparse with action="append", nargs="+" gives list-of-lists.
    if all(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
        if len(vals) == 1:
            one = [float(x) for x in vals[0]]
            return [one for _ in range(n)]
        if len(vals) != n:
            raise ValueError("Provide one --shank-split-x group per KS directory, or one group to reuse for all.")
        return [[float(x) for x in group] for group in vals]

    # Flat list: for one recording, it is a multi-split list. For multiple recordings,
    # len(vals)==n means one split per recording; otherwise reuse the multi-split list for all.
    if all(_is_scalar_number(v) for v in vals):
        if n == 1:
            return [[float(x) for x in vals]]
        if len(vals) == n:
            return [[float(x)] for x in vals]
        one = [float(x) for x in vals]
        return [one for _ in range(n)]

    raise ValueError("Could not interpret shank_split_x. Use floats or lists of floats.")


def _normalize_region_names(shank_region_names, split_groups, n: int):
    if shank_region_names is None:
        return [None] * n

    if isinstance(shank_region_names, str):
        names = shank_region_names.replace(",", " ").split()
        return [names for _ in range(n)]

    vals = list(shank_region_names)
    if not vals:
        return [None] * n

    # argparse with action="append", nargs="+" gives list-of-lists.
    if all(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
        if len(vals) == 1:
            one = [str(x) for x in vals[0]]
            return [one for _ in range(n)]
        if len(vals) != n:
            raise ValueError("Provide one --shank-region-names group per KS directory, or one group to reuse for all.")
        return [[str(x) for x in group] for group in vals]

    if all(isinstance(v, str) for v in vals):
        return [[str(x) for x in vals] for _ in range(n)]

    raise ValueError("Could not interpret shank_region_names. Use strings or lists of strings.")


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
    shank_split_x=None,
    shank_region_names=None,
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
            shank_region_names=shank_region_names,
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
        "shank_region_names": shank_region_names,
    }


def run_batch(
    ks_dirs,
    animal_ids=None,
    out_root: str | Path | None = None,
    shank_split_x=None,
    shank_region_names=None,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    threads: int = DEFAULT_THREADS,
    blocks: int = DEFAULT_BLOCKS,
    save_plots: bool = False,
    n_null_plots: int = 20,
    allow_all_clusters_without_cluster_info: bool = False,
    interactive: bool = False,
    aggregate_shank_stats: bool = False,
    aggregate_distance_plots: bool = False,
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
    region_names = _normalize_region_names(shank_region_names, shank_splits, len(ks_dirs))

    plans = []
    for ks_dir, animal_id, split_x, region_names_i in zip(ks_dirs, animal_ids, shank_splits, region_names):
        out_dir = output_dir_for_ks_dir(ks_dir, out_root_path, animal_id)
        plans.append(
            prepare_recording_inputs(
                ks_dir=ks_dir,
                out_dir=out_dir,
                animal_id=animal_id,
                allow_all_clusters_without_cluster_info=allow_all_clusters_without_cluster_info,
                interactive=interactive,
                shank_split_x=split_x,
                shank_region_names=region_names_i,
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
                shank_region_names=plan["shank_region_names"],
            )
        )

    if aggregate_shank_stats and len(plans) > 1:
        aggregate_out_dir = out_root_path if out_root_path is not None else Path(plans[0]["out_dir"]) / "aggregate_shank_stats"
        save_aggregate_shank_stats(plans, aggregate_out_dir=aggregate_out_dir)

    if aggregate_distance_plots:
        aggregate_out_dir = out_root_path if out_root_path is not None else Path(plans[0]["out_dir"]) / "aggregate_distance_probability"
        save_aggregate_distance_probability_plots(plans, aggregate_out_dir=aggregate_out_dir)

    return outputs


def _comparison_regions(comparison: str):
    comparison = str(comparison)
    if "<->" in comparison:
        a, b = comparison.split("<->", 1)
        return a.strip(), b.strip()
    return comparison.strip(), comparison.strip()


def _region_probe_prefix(region_name: str) -> str:
    name = str(region_name).strip()
    for sep in ("_", "-", ":", "|"):
        if sep in name:
            return name.split(sep, 1)[0].strip()
    return ""


def classify_shank_comparison(comparison: str) -> str:
    a, b = _comparison_regions(comparison)
    if a == b:
        return "within_region"

    pa = _region_probe_prefix(a)
    pb = _region_probe_prefix(b)
    if pa and pb:
        if pa == pb:
            return "same_probe_between_regions"
        return "different_probe_between_regions"

    return "between_regions"


def _bonferroni_p(p: float, n_tests: int) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(min(1.0, p * max(1, n_tests)))


def save_aggregate_shank_stats(plans, aggregate_out_dir: Path, prefix: str = "aggregate_shank"):
    """
    Aggregate per-animal shank/region connection percentages across multiple recordings.

    Statistics are performed on one aggregated category rate per animal, not on every
    possible neuron-pair opportunity. This avoids treating thousands of CCG pair tests
    from the same mouse as independent biological replicates.
    """
    rows = []
    for plan in plans:
        animal_id = plan["animal_id"]
        out_dir = Path(plan["out_dir"])
        csv_path = out_dir / f"{animal_id}_shank_connection_percentages.csv"
        if not csv_path.exists():
            print(f"  aggregate stats: missing {csv_path}; skipping")
            continue

        df = pd.read_csv(csv_path)
        needed = {"comparison", "n_prepost_possible_directed", "n_detected_connections"}
        if not needed.issubset(set(df.columns)):
            print(f"  aggregate stats: {csv_path} missing needed columns; skipping")
            continue

        df = df.copy()
        df["animal_id"] = animal_id
        df["comparison_category"] = df["comparison"].map(classify_shank_comparison)
        rows.append(df)

    if not rows:
        print("  aggregate stats: no per-animal shank percentage CSVs found")
        return None

    all_rows = pd.concat(rows, ignore_index=True)
    aggregate_out_dir = Path(aggregate_out_dir)
    aggregate_out_dir.mkdir(parents=True, exist_ok=True)

    all_rows_path = aggregate_out_dir / f"{prefix}_all_comparison_rows.csv"
    all_rows.to_csv(all_rows_path, index=False)

    category_rows = []
    for (animal_id, category), g in all_rows.groupby(["animal_id", "comparison_category"], sort=False):
        possible = int(g["n_prepost_possible_directed"].sum())
        detected = int(g["n_detected_connections"].sum())
        pct = 100.0 * detected / possible if possible else np.nan
        category_rows.append(
            {
                "animal_id": animal_id,
                "comparison_category": category,
                "n_prepost_possible_directed": possible,
                "n_detected_connections": detected,
                "percent_possible_connections": pct,
                "n_component_comparisons": int(len(g)),
            }
        )

    category_df = pd.DataFrame(category_rows)
    category_path = aggregate_out_dir / f"{prefix}_category_rates_by_animal.csv"
    category_df.to_csv(category_path, index=False)

    stat_rows = []
    categories = list(dict.fromkeys(category_df["comparison_category"].dropna().tolist()))
    n_tests = len(categories) * (len(categories) - 1) // 2
    for i, cat_a in enumerate(categories):
        for cat_b in categories[i + 1:]:
            a = category_df.loc[category_df["comparison_category"] == cat_a, ["animal_id", "percent_possible_connections"]]
            b = category_df.loc[category_df["comparison_category"] == cat_b, ["animal_id", "percent_possible_connections"]]
            merged = a.merge(b, on="animal_id", suffixes=("_a", "_b"))
            n = int(len(merged))

            p = np.nan
            stat = np.nan
            mean_a = float(merged["percent_possible_connections_a"].mean()) if n else np.nan
            mean_b = float(merged["percent_possible_connections_b"].mean()) if n else np.nan
            median_a = float(merged["percent_possible_connections_a"].median()) if n else np.nan
            median_b = float(merged["percent_possible_connections_b"].median()) if n else np.nan
            mean_diff = float((merged["percent_possible_connections_a"] - merged["percent_possible_connections_b"]).mean()) if n else np.nan

            if n >= 2:
                try:
                    from scipy.stats import wilcoxon
                    stat, p = wilcoxon(
                        merged["percent_possible_connections_a"],
                        merged["percent_possible_connections_b"],
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                    stat = float(stat)
                    p = float(p)
                except Exception:
                    stat = np.nan
                    p = np.nan

            stat_rows.append(
                {
                    "category_a": cat_a,
                    "category_b": cat_b,
                    "paired_n_animals": n,
                    "test": "paired Wilcoxon signed-rank on per-animal category percentages",
                    "statistic": stat,
                    "p_value": p,
                    "p_value_bonferroni": _bonferroni_p(p, n_tests),
                    "mean_percent_a": mean_a,
                    "mean_percent_b": mean_b,
                    "median_percent_a": median_a,
                    "median_percent_b": median_b,
                    "mean_percent_a_minus_b": mean_diff,
                }
            )

    stats_df = pd.DataFrame(stat_rows)
    stats_path = aggregate_out_dir / f"{prefix}_pairwise_category_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_categories = [
        c for c in categories
        if np.any(np.isfinite(category_df.loc[category_df["comparison_category"] == c, "percent_possible_connections"]))
    ]
    data = [
        category_df.loc[category_df["comparison_category"] == c, "percent_possible_connections"].dropna().to_numpy()
        for c in plot_categories
    ]

    plot_path = None
    if data:
        fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(plot_categories)), 5))
        ax.boxplot(data, labels=plot_categories, showfliers=False)

        rng = np.random.default_rng(0)
        for x, vals in enumerate(data, start=1):
            if len(vals):
                jitter = rng.normal(0, 0.035, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=24, alpha=0.75)

        ax.set_ylabel("Detected directed connections (% of possible)")
        ax.set_title("Aggregate FSC connection rates by shank/region comparison class")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()

        plot_path = aggregate_out_dir / f"{prefix}_category_boxplot.png"
        fig.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved aggregate comparison rows: {all_rows_path}")
    print(f"  saved aggregate category rates: {category_path}")
    print(f"  saved aggregate pairwise stats: {stats_path}")
    if plot_path is not None:
        print(f"  saved aggregate boxplot: {plot_path}")

    return {
        "all_rows_csv": all_rows_path,
        "category_rates_csv": category_path,
        "pairwise_stats_csv": stats_path,
        "boxplot_png": plot_path,
    }


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
                shank_region_names=None,
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
            shank_region_names=plan["shank_region_names"],
        )

    print(f"\nFinished all directories in {(time.time() - start_all) / 60.0:.2f} min")


    if len(plans) >= 1 and yes_no("\nSave aggregate probability-vs-distance plots from directed pair tables?", default=True):
        aggregate_out_dir = out_root if out_root is not None else Path(plans[0]["out_dir"]) / "aggregate_distance_probability"
        save_aggregate_distance_probability_plots(plans, aggregate_out_dir=aggregate_out_dir)

    if len(plans) > 1 and yes_no("\nSave aggregate shank/region category boxplot and stats?", default=False):
        aggregate_out_dir = out_root if out_root is not None else Path(plans[0]["out_dir"]) / "aggregate_shank_stats"
        save_aggregate_shank_stats(plans, aggregate_out_dir=aggregate_out_dir)

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
    region_names = _normalize_region_names(args.shank_region_names, shank_splits, len(ks_dirs))

    plans = []
    for ks_dir, animal_id, split_x, region_names_i in zip(ks_dirs, animal_ids, shank_splits, region_names):
        out_dir = output_dir_for_ks_dir(ks_dir, out_root, animal_id)
        plans.append(
            prepare_recording_inputs(
                ks_dir=ks_dir,
                out_dir=out_dir,
                animal_id=animal_id,
                allow_all_clusters_without_cluster_info=args.allow_all_clusters,
                interactive=False,
                shank_split_x=split_x,
                shank_region_names=region_names_i,
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
            shank_region_names=plan["shank_region_names"],
        )

    if args.aggregate_shank_stats and len(plans) > 1:
        aggregate_out_dir = out_root if out_root is not None else Path(plans[0]["out_dir"]) / "aggregate_shank_stats"
        save_aggregate_shank_stats(plans, aggregate_out_dir=aggregate_out_dir)

    if args.aggregate_distance_plots:
        aggregate_out_dir = out_root if out_root is not None else Path(plans[0]["out_dir"]) / "aggregate_distance_probability"
        save_aggregate_distance_probability_plots(plans, aggregate_out_dir=aggregate_out_dir)

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
    parser.add_argument("--shank-split-x", action="append", nargs="+", type=float, help="x-coordinate split(s) in microns. Example: --shank-split-x 100 1000 2100 for 4 regions. Repeat once per --ks-dir if needed.")
    parser.add_argument("--shank-region-names", action="append", nargs="+", help="Optional left-to-right region names. Example: --shank-region-names probe1_s1 probe1_s2 probe2_s1 probe2_s2. Repeat once per --ks-dir if needed.")
    parser.add_argument("--aggregate-shank-stats", action="store_true", help="After multi-directory runs, save an aggregate boxplot and pairwise stats CSV for shank/region comparison classes.")
    parser.add_argument("--aggregate-distance-plots", action="store_true", help="After runs, pool directed pair distance tables and save probability-by-distance plots, including a different-probe far-tail zoom.")

    args = parser.parse_args()

    if args.ks_dir:
        cli_main(args)
    else:
        interactive_main()


if __name__ == "__main__":
    main()
