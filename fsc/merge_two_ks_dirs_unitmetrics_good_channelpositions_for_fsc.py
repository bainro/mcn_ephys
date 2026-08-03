#!/usr/bin/env python3
"""
merge_two_ks_dirs_unitmetrics_good_channelpositions_for_fsc.py

Deadline-friendly merge script for two simultaneously recorded Kilosort output
folders where each folder already contains two probes/channels from one KS sort.

Use case:
  - Two KS directories recorded on the same sample clock/time origin.
  - Each KS directory has spike_times.npy, spike_clusters.npy, channel_positions.npy,
    and UnitMetrics.csv.
  - UnitMetrics.csv has columns: cluster_id, ch, good.
  - No XML/chmapfull.mat needed.
  - Coordinates are taken from KS channel_positions.npy via UnitMetrics.csv ch.
  - Cluster IDs in KS_DIR_2 are offset to avoid collisions.
  - Channel IDs in KS_DIR_2 are offset to avoid channel_map collisions.
  - Fake x/y offsets may be used; distance is not the target analysis here.

This script writes a minimal merged KS-like folder:
  spike_times.npy
  spike_clusters.npy
  channel_map.npy
  channel_positions.npy
  UnitMetrics.csv
  cluster_info.tsv
  full_channel_geometry.csv
  merge_manifest.json

It does not modify either input KS directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# EDIT THESE SETTINGS
# -----------------------------

KS_DIR_1 = Path(r"D:\\ms17\\8_hrs_ms17_CONNECTS_LHS")
KS_DIR_2 = Path(r"D:\\ms17\\8_hrs_ms17_CONNECTS_RHS")
MERGED_KS_DIR = Path(r"G:\\merged_16hr_SWS_run_CONNECTS")

SOURCE1_NAME = "MOs_RSC_LHS"
SOURCE2_NAME = "MOs_RSC_RHS"

# Since distance is not the target for the state-split analysis, these can be
# arbitrary large offsets to keep the two KS geometries visually separated.
KS2_X_OFFSET_UM = 1000.0
KS2_Y_OFFSET_UM = 1000.0

CLUSTER_OFFSET = 1_000_000
CHANNEL_OFFSET = 1_000_000

OVERWRITE_OUTPUT_DIR = False

# UnitMetrics column names.
CLUSTER_COL = "cluster_id"
CHANNEL_COL = "ch"
GOOD_COL = "good"

# Because you did not deactivate channels in KS, this defaults to treating
# UnitMetrics.csv ch as the row index into channel_positions.npy. Other options:
#   "channel_map"  -> map UnitMetrics ch through channel_map.npy first
#   "auto"         -> try row_index first, then channel_map
CHANNEL_TO_POSITION_MODE = "row_index"


# -----------------------------
# Helpers
# -----------------------------
def _resolve_existing(pathlike, label: str) -> Path:
    p = Path(pathlike).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _good_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)

    if np.issubdtype(series.dtype, np.number):
        return (series.to_numpy(dtype=float) > 0)

    vals = series.astype(str).str.strip().str.lower()
    return vals.isin({"true", "t", "1", "yes", "y", "good"}).to_numpy(dtype=bool)


def _load_ks_arrays(ks_dir: Path):
    required = ["spike_times.npy", "spike_clusters.npy", "channel_positions.npy"]
    for name in required:
        if not (ks_dir / name).exists():
            raise FileNotFoundError(f"Missing {name} in {ks_dir}")

    spike_times = np.load(ks_dir / "spike_times.npy").reshape(-1)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").reshape(-1)
    channel_positions = np.asarray(np.load(ks_dir / "channel_positions.npy"), dtype=float)

    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError(
            f"{ks_dir}: spike_times and spike_clusters have different lengths: "
            f"{spike_times.shape[0]} vs {spike_clusters.shape[0]}"
        )

    if (ks_dir / "channel_map.npy").exists():
        channel_map = np.load(ks_dir / "channel_map.npy").reshape(-1).astype(int)
    else:
        channel_map = np.arange(channel_positions.shape[0], dtype=int)

    if channel_positions.shape[0] != channel_map.shape[0]:
        raise ValueError(
            f"{ks_dir}: channel_positions rows ({channel_positions.shape[0]}) != "
            f"channel_map length ({channel_map.shape[0]})"
        )

    return spike_times, spike_clusters, channel_map, channel_positions


def _load_unitmetrics(ks_dir: Path, source_name: str, x_offset: float, y_offset: float, cluster_offset: int, channel_offset: int):
    unit_path = ks_dir / "UnitMetrics.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing UnitMetrics.csv in {ks_dir}")

    df = pd.read_csv(unit_path)
    missing = [c for c in (CLUSTER_COL, CHANNEL_COL, GOOD_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{unit_path} missing columns {missing}. Columns: {list(df.columns)}")

    good = _good_mask(df[GOOD_COL])
    df = df.loc[good].copy()
    if df.empty:
        raise ValueError(f"{unit_path}: zero rows with {GOOD_COL}=TRUE")

    channel_map, channel_positions = None, None
    _, _, channel_map, channel_positions = _load_ks_arrays(ks_dir)

    # Build both common lookup conventions:
    #   row_index   : ch is already the row into channel_positions.npy
    #   channel_map : ch should be looked up in channel_map.npy
    #   auto        : try row_index first, then channel_map
    map_value_to_row = {int(ch): i for i, ch in enumerate(channel_map.astype(int).tolist())}

    def ch_to_position_row(ch_raw: int) -> int:
        mode = str(CHANNEL_TO_POSITION_MODE).strip().lower()
        ch_raw = int(ch_raw)

        if mode in {"row", "row_index", "index"}:
            if 0 <= ch_raw < channel_positions.shape[0]:
                return ch_raw

        elif mode in {"channel_map", "map"}:
            if ch_raw in map_value_to_row:
                return int(map_value_to_row[ch_raw])

        elif mode == "auto":
            if 0 <= ch_raw < channel_positions.shape[0]:
                return ch_raw
            if ch_raw in map_value_to_row:
                return int(map_value_to_row[ch_raw])
        else:
            raise ValueError(
                "CHANNEL_TO_POSITION_MODE must be 'row_index', 'channel_map', or 'auto'. "
                f"Got {CHANNEL_TO_POSITION_MODE!r}"
            )

        raise ValueError(
            f"{unit_path}: could not map ch={ch_raw} to channel_positions.npy using "
            f"CHANNEL_TO_POSITION_MODE={CHANNEL_TO_POSITION_MODE!r}. "
            f"channel_positions has {channel_positions.shape[0]} rows; channel_map min/max = "
            f"{int(np.min(channel_map))}/{int(np.max(channel_map))}."
        )

    x_vals = []
    y_vals = []
    channel_row_vals = []
    for ch_raw in df[CHANNEL_COL].astype(int).to_numpy():
        row = ch_to_position_row(int(ch_raw))
        channel_row_vals.append(row)
        x_vals.append(float(channel_positions[row, 0]) + float(x_offset))
        y_vals.append(float(channel_positions[row, 1]) + float(y_offset))

    out = df.copy()
    out["original_cluster_id"] = out[CLUSTER_COL].astype(int)
    out["original_ch"] = out[CHANNEL_COL].astype(int)
    out["original_channel_row"] = np.asarray(channel_row_vals, dtype=int)
    out["source_ks_dir"] = str(ks_dir)
    out["source_name"] = source_name
    out["source_probe"] = source_name
    out["fsc_label"] = source_name
    out["region"] = source_name

    out[CLUSTER_COL] = out[CLUSTER_COL].astype(int) + int(cluster_offset)
    out[CHANNEL_COL] = out[CHANNEL_COL].astype(int) + int(channel_offset)
    out["cluster_id"] = out[CLUSTER_COL].astype(int)
    out["ch"] = out[CHANNEL_COL].astype(int)
    out["good"] = True
    out["isGood"] = True
    out["group"] = "good"
    out["x_um"] = np.asarray(x_vals, dtype=float)
    out["y_um"] = np.asarray(y_vals, dtype=float)
    out["coordinate_source"] = "channel_positions.npy via UnitMetrics.csv ch"

    print(f"  {source_name}: kept {len(out)} good units from {unit_path}")
    return out


def _full_channel_geometry(ks_dir: Path, source_name: str, x_offset: float, y_offset: float, channel_offset: int):
    _, _, channel_map, channel_positions = _load_ks_arrays(ks_dir)
    rows = []
    for row, ch in enumerate(channel_map.astype(int).tolist()):
        rows.append(
            {
                "source_name": source_name,
                "source_ks_dir": str(ks_dir),
                "channel_row": int(row),
                "original_ch": int(ch),
                "ch": int(ch) + int(channel_offset),
                "x_um": float(channel_positions[row, 0]) + float(x_offset),
                "y_um": float(channel_positions[row, 1]) + float(y_offset),
            }
        )
    return pd.DataFrame(rows)


def _dtype_can_hold(dtype, values, label):
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        return
    values = np.asarray(values)
    if values.size == 0:
        return
    info = np.iinfo(dtype)
    vmin = int(np.min(values))
    vmax = int(np.max(values))
    if vmin < info.min or vmax > info.max:
        raise OverflowError(f"{label} [{vmin}, {vmax}] does not fit dtype {dtype}")


def main():
    ks1 = _resolve_existing(KS_DIR_1, "KS_DIR_1")
    ks2 = _resolve_existing(KS_DIR_2, "KS_DIR_2")
    out_dir = Path(MERGED_KS_DIR).expanduser().resolve()

    if out_dir.exists():
        if not OVERWRITE_OUTPUT_DIR:
            raise FileExistsError(f"Output exists and OVERWRITE_OUTPUT_DIR=False: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading KS arrays...")
    st1, sc1, cmap1, cpos1 = _load_ks_arrays(ks1)
    st2, sc2, cmap2, cpos2 = _load_ks_arrays(ks2)

    print("Loading UnitMetrics good units...")
    um1 = _load_unitmetrics(
        ks1, SOURCE1_NAME,
        x_offset=0.0,
        y_offset=0.0,
        cluster_offset=0,
        channel_offset=0,
    )
    um2 = _load_unitmetrics(
        ks2, SOURCE2_NAME,
        x_offset=KS2_X_OFFSET_UM,
        y_offset=KS2_Y_OFFSET_UM,
        cluster_offset=CLUSTER_OFFSET,
        channel_offset=CHANNEL_OFFSET,
    )

    good1 = um1["original_cluster_id"].astype(int).to_numpy()
    good2 = um2["original_cluster_id"].astype(int).to_numpy()

    mask1 = np.isin(sc1.astype(np.int64), good1.astype(np.int64))
    mask2 = np.isin(sc2.astype(np.int64), good2.astype(np.int64))

    sc2_shifted = sc2[mask2].astype(np.int64) + int(CLUSTER_OFFSET)
    _dtype_can_hold(sc1.dtype, sc2_shifted, "shifted KS2 cluster IDs")
    sc2_shifted = sc2_shifted.astype(sc1.dtype, copy=False)

    merged_spike_times = np.concatenate([st1[mask1], st2[mask2]])
    merged_spike_clusters = np.concatenate([sc1[mask1], sc2_shifted])

    # Sort by time for safety. These are simultaneous recordings with the same clock.
    order = np.argsort(merged_spike_times, kind="mergesort")
    merged_spike_times = merged_spike_times[order].astype(st1.dtype, copy=False)
    merged_spike_clusters = merged_spike_clusters[order].astype(sc1.dtype, copy=False)

    # Channel metadata. Main.py/state script will prefer UnitMetrics x_um/y_um for units.
    cmap2_shifted = cmap2.astype(np.int64) + int(CHANNEL_OFFSET)
    _dtype_can_hold(cmap1.dtype, cmap2_shifted, "shifted KS2 channel map")
    merged_channel_map = np.concatenate([cmap1, cmap2_shifted.astype(cmap1.dtype, copy=False)])

    cpos2_shifted = cpos2.astype(float).copy()
    cpos2_shifted[:, 0] += float(KS2_X_OFFSET_UM)
    cpos2_shifted[:, 1] += float(KS2_Y_OFFSET_UM)
    merged_channel_positions = np.vstack([cpos1.astype(float), cpos2_shifted])

    unitmetrics = pd.concat([um1, um2], ignore_index=True)
    unitmetrics = unitmetrics.sort_values("cluster_id").reset_index(drop=True)

    cluster_info = pd.DataFrame(
        {
            "cluster_id": unitmetrics["cluster_id"].astype(int),
            "id": unitmetrics["cluster_id"].astype(int),
            "ch": unitmetrics["ch"].astype(int),
            "group": "good",
            "source_name": unitmetrics["source_name"].astype(str),
        }
    )

    full_geom = pd.concat(
        [
            _full_channel_geometry(ks1, SOURCE1_NAME, 0.0, 0.0, 0),
            _full_channel_geometry(ks2, SOURCE2_NAME, KS2_X_OFFSET_UM, KS2_Y_OFFSET_UM, CHANNEL_OFFSET),
        ],
        ignore_index=True,
    )

    np.save(out_dir / "spike_times.npy", merged_spike_times)
    np.save(out_dir / "spike_clusters.npy", merged_spike_clusters)
    np.save(out_dir / "channel_map.npy", merged_channel_map)
    np.save(out_dir / "channel_positions.npy", merged_channel_positions)
    unitmetrics.to_csv(out_dir / "UnitMetrics.csv", index=False)
    cluster_info.to_csv(out_dir / "cluster_info.tsv", sep="\t", index=False)
    full_geom.to_csv(out_dir / "full_channel_geometry.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "ks_dir_1": str(ks1),
        "ks_dir_2": str(ks2),
        "merged_ks_dir": str(out_dir),
        "source1_name": SOURCE1_NAME,
        "source2_name": SOURCE2_NAME,
        "cluster_offset": int(CLUSTER_OFFSET),
        "channel_offset": int(CHANNEL_OFFSET),
        "ks2_x_offset_um": float(KS2_X_OFFSET_UM),
        "ks2_y_offset_um": float(KS2_Y_OFFSET_UM),
        "n_good_units_1": int(len(um1)),
        "n_good_units_2": int(len(um2)),
        "n_good_units_total": int(len(unitmetrics)),
        "n_spikes_1_selected": int(np.sum(mask1)),
        "n_spikes_2_selected": int(np.sum(mask2)),
        "n_spikes_total": int(merged_spike_times.size),
        "coordinate_source": "channel_positions.npy via UnitMetrics.csv ch",
    }
    with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nMerged KS directory created:")
    print(f"  {out_dir}")
    print("\nSelected units:")
    print(f"  {SOURCE1_NAME}: {len(um1)}")
    print(f"  {SOURCE2_NAME}: {len(um2)}")
    print(f"  total: {len(unitmetrics)}")
    print("\nSelected spikes:")
    print(f"  {SOURCE1_NAME}: {int(np.sum(mask1)):,}")
    print(f"  {SOURCE2_NAME}: {int(np.sum(mask2)):,}")
    print(f"  total: {merged_spike_times.size:,}")
    print("\nWrote:")
    print("  spike_times.npy")
    print("  spike_clusters.npy")
    print("  channel_map.npy")
    print("  channel_positions.npy")
    print("  UnitMetrics.csv")
    print("  cluster_info.tsv")
    print("  full_channel_geometry.csv")
    print("  merge_manifest.json")


if __name__ == "__main__":
    main()
