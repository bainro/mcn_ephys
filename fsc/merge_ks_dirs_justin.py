#!/usr/bin/env python3
"""
merge_two_ks_dirs_goodunitinfo_chmapfull_for_fsc.py

Merge two simultaneously recorded, separately sorted Kilosort output folders into
one minimal KS-like folder for FSC analysis, using Justin/VR-style
GoodUnitInfo.csv files instead of Winny-style unitdata .mat files.

ASSUMPTIONS / BUG-RISK NOTES
----------------------------
1. The two KS folders come from recordings with the same Intan sample clock and
   the same sample-time origin. This script does NOT apply any time shift.
2. Each probe/region has its own KS directory and its own GoodUnitInfo.csv.
3. Every row in each GoodUnitInfo.csv is assumed to be a good unit. The script
   filters only by the configured Region substring(s), because the files can
   contain units from regions you want to exclude, e.g. CA1.
4. Expected GoodUnitInfo.csv columns:
      ClusterID, Channel, Shank, Region
   Optional/debug columns preserved when present:
      XPosition, Depth, CellType, num_spikes
5. Physical coordinates are NOT taken from GoodUnitInfo XPosition/Depth. They are
   taken from CHMAPFULL_PATH by mapping GoodUnitInfo.Channel -> chmapfull channel.
6. Probe 2 receives an x and y offset relative to probe 1. Positive x means
   probe 2 is shifted to the RIGHT of probe 1 in plot coordinates. Positive y
   means probe 2 is shifted UP relative to probe 1 in plot coordinates.
7. Probe 2 cluster IDs are shifted by CLUSTER_OFFSET to prevent collisions.
8. Probe 2 channel IDs are shifted by CHANNEL_OFFSET to prevent channel-map
   collisions. UnitMetrics.csv `ch` is included for compatibility, but downstream
   FSC plotting should prefer UnitMetrics.csv x_um/y_um and full_channel_geometry.csv.
9. The script writes full_channel_geometry.csv containing every physical channel
   from chmapfull for both probes, including probe 2 x/y offsets. Downstream
   plotting should use this for background probe-site squares.
10. This script only writes to MERGED_KS_DIR. It does not modify KS_DIR_1,
    KS_DIR_2, GOOD_UNIT_CSV_1, GOOD_UNIT_CSV_2, or CHMAPFULL_PATH. If
    OVERWRITE_OUTPUT_DIR=True and MERGED_KS_DIR already exists, MERGED_KS_DIR
    will be deleted and recreated.
11. The merged spike_times.npy dtype is preserved from the original KS files.
    The merged spike_clusters.npy dtype is preserved from the original KS files,
    so int32 clusters stay int32 and the cluster file stays ~half the size of
    uint64 spike_times.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# -----------------------------
# EDIT THESE PATHS / SETTINGS
# -----------------------------

### more parameters in main()!

PROBE1_NAME = "VC"
PROBE2_NAME = "PPC"

# Full probe map used by the earlier Winny/QC workflow.
# Expected variable: chmapfull, with columns [channel_number, x_um, y_um]
# after transposing if stored as 3 x N in MATLAB v7.3/HDF5.
CHMAPFULL_PATH = Path(r'Y:\\Winny\\Data\\chmapfull.mat')

# Region substring filters. All rows in GoodUnitInfo.csv are assumed good, but
# these filters exclude off-target regions. Empty tuple/list/None means keep all.
REGION_CONTAINS_1 = ("VC",)
REGION_CONTAINS_2 = ("PPC",)
CASE_SENSITIVE_REGION = False

# If None, the script asks interactively every time. Positive x = probe2 right of
# probe1; positive y = probe2 up relative to probe1 in the plot coordinate system.
PROBE2_X_OFFSET_UM = -2400 # um
PROBE2_Y_OFFSET_UM = -900 # um

CLUSTER_OFFSET = 1_000_000
CHANNEL_OFFSET = 1_000_000

OVERWRITE_OUTPUT_DIR = False
ERROR_ON_MISSING_SELECTED_CLUSTERS = True

# Skip animals unless each region/probe has at least this many selected units
# after the Region substring filter. Set either value to 0/None to disable.
MIN_UNITS_PROBE1 = 40  # e.g., VC
MIN_UNITS_PROBE2 = 40  # e.g., PPC


# -----------------------------
# General helpers
# -----------------------------
def _region_matches(region: str, substrings, case_sensitive: bool) -> bool:
    if substrings is None or len(substrings) == 0:
        return True
    hay = str(region) if case_sensitive else str(region).lower()
    for sub in substrings:
        needle = str(sub) if case_sensitive else str(sub).lower()
        if needle in hay:
            return True
    return False


def _dtype_can_hold_values(dtype, values, label):
    dtype = np.dtype(dtype)
    values = np.asarray(values)
    if values.size == 0:
        return
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        vmin = int(np.min(values))
        vmax = int(np.max(values))
        if vmin < info.min or vmax > info.max:
            raise OverflowError(
                f"{label} values [{vmin}, {vmax}] do not fit into {dtype}. "
                f"Use a smaller offset or a wider dtype."
            )


def _resolve_path(pathlike, label: str) -> Path:
    p = Path(pathlike)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


# -----------------------------
# KS loading
# -----------------------------
def load_ks_arrays(ks_dir: Path):
    ks_dir = _resolve_path(ks_dir, "KS directory")
    needed = [
        "spike_times.npy",
        "spike_clusters.npy",
        "channel_map.npy",
        "channel_positions.npy",
    ]
    for name in needed:
        if not (ks_dir / name).exists():
            raise FileNotFoundError(f"Missing {name} in {ks_dir}")

    spike_times = np.load(ks_dir / "spike_times.npy").reshape(-1)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").reshape(-1)
    channel_map = np.load(ks_dir / "channel_map.npy").reshape(-1)
    channel_positions = np.load(ks_dir / "channel_positions.npy")

    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError(
            f"{ks_dir}: spike_times and spike_clusters have different lengths: "
            f"{spike_times.shape[0]} vs {spike_clusters.shape[0]}"
        )

    if channel_positions.shape[0] != channel_map.shape[0]:
        raise ValueError(
            f"{ks_dir}: channel_positions rows != channel_map length: "
            f"{channel_positions.shape[0]} vs {channel_map.shape[0]}"
        )

    return spike_times, spike_clusters, channel_map, channel_positions


# -----------------------------
# GoodUnitInfo.csv loading
# -----------------------------
def load_goodunitinfo_csv(csv_path: Path, region_contains, probe_name: str) -> pd.DataFrame:
    csv_path = _resolve_path(csv_path, f"{probe_name} GoodUnitInfo.csv")
    df = pd.read_csv(csv_path)

    required = {"ClusterID", "Channel", "Shank", "Region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns {sorted(missing)}. Columns: {list(df.columns)}")

    before = len(df)
    keep = df["Region"].astype(str).apply(
        lambda r: _region_matches(r, region_contains, CASE_SENSITIVE_REGION)
    )
    df = df.loc[keep].copy()

    if df.empty:
        available = sorted(pd.read_csv(csv_path)["Region"].astype(str).unique().tolist())
        raise ValueError(
            f"{probe_name}: zero rows passed Region filter {region_contains} in {csv_path}. "
            f"Available Region values: {available}"
        )

    # Normalize to the UnitMetrics-like schema used by main.py.
    out = pd.DataFrame(
        {
            "original_cluster_id": df["ClusterID"].astype(int),
            "channelnum_original": df["Channel"].astype(int),
            "shanknum": df["Shank"].astype(int),
            "regname": df["Region"].astype(str),
            "region": df["Region"].astype(str),
            "fsc_label": df["Region"].astype(str),
            "source_probe": probe_name,
            "good": True,
            "isGood": True,
        }
    )

    # Preserve optional/debug columns without trusting them for coordinates.
    optional_map = {
        "XPosition": "csv_x_position",
        "Depth": "csv_depth",
        "CellType": "cell_type",
        "num_spikes": "csv_num_spikes",
    }
    for src, dst in optional_map.items():
        if src in df.columns:
            out[dst] = df[src].to_numpy()

    print(
        f"  {probe_name}: kept {len(out)} / {before} rows from {csv_path.name} "
        f"using Region contains {region_contains}"
    )
    return out


def count_goodunitinfo_units(csv_path: Path, region_contains, probe_name: str) -> int:
    """Count selected units after Region filtering without building the full merge."""
    csv_path = _resolve_path(csv_path, f"{probe_name} GoodUnitInfo.csv")
    df = pd.read_csv(csv_path)

    required = {"ClusterID", "Channel", "Shank", "Region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns {sorted(missing)}. Columns: {list(df.columns)}")

    keep = df["Region"].astype(str).apply(
        lambda r: _region_matches(r, region_contains, CASE_SENSITIVE_REGION)
    )
    return int(keep.sum())


# -----------------------------
# chmapfull loading/coordinate mapping
# -----------------------------
def load_chmapfull(chmap_path: Path):
    """Load Winny-style chmapfull.mat and return channel -> (x_um, y_um)."""
    chmap_path = _resolve_path(chmap_path, "CHMAPFULL_PATH")

    try:
        mat = loadmat(chmap_path, squeeze_me=True, struct_as_record=False)
        if "chmapfull" not in mat:
            raise KeyError("chmapfull")
        arr = np.asarray(mat["chmapfull"])
    except NotImplementedError:
        import h5py
        with h5py.File(chmap_path, "r") as f:
            if "chmapfull" not in f:
                raise KeyError(f"{chmap_path} does not contain dataset 'chmapfull'. Keys: {list(f.keys())}")
            arr = np.asarray(f["chmapfull"])

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"chmapfull should be a 2D array, got shape {arr.shape}")

    # Earlier workflow does: chmapfull = np.array(f['chmapfull']).T
    # Accept either 3 x N or N x 3.
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    elif arr.shape[1] != 3:
        raise ValueError(f"chmapfull should have 3 columns [channel, x, y], got shape {arr.shape}")

    chnum = arr[:, 0].astype(int)
    xcoord = arr[:, 1].astype(float)
    ycoord = arr[:, 2].astype(float)

    if len(np.unique(chnum)) != len(chnum):
        raise ValueError("chmapfull contains duplicate channel numbers.")

    ch_to_xy = {int(ch): (float(x), float(y)) for ch, x, y in zip(chnum, xcoord, ycoord)}
    return ch_to_xy, arr


def attach_chmapfull_coordinates(unit_df: pd.DataFrame, ch_to_xy: dict[int, tuple[float, float]], probe_name: str):
    """Attach physical x/y coordinates using GoodUnitInfo.Channel and chmapfull."""
    out = unit_df.copy()
    xs, ys, mapped_channels = [], [], []
    missing = []

    for ch in out["channelnum_original"].astype(int).to_numpy():
        ch_key = int(ch)
        if ch_key in ch_to_xy:
            mapped_ch = ch_key
        elif (ch_key - 1) in ch_to_xy:
            # Defensive fallback in case one file is 1-based and chmapfull is 0-based.
            mapped_ch = ch_key - 1
        elif (ch_key + 1) in ch_to_xy:
            # Defensive fallback in case one file is 0-based and chmapfull is 1-based.
            mapped_ch = ch_key + 1
        else:
            missing.append(ch_key)
            continue

        x, y = ch_to_xy[mapped_ch]
        xs.append(float(x))
        ys.append(float(y))
        mapped_channels.append(int(mapped_ch))

    if missing:
        raise ValueError(
            f"{probe_name}: {len(missing)} GoodUnitInfo.Channel values could not be found in chmapfull. "
            f"First few missing: {missing[:10]}"
        )

    out["chmapfull_channel"] = mapped_channels
    out["x_um_unshifted"] = xs
    out["y_um_unshifted"] = ys
    return out


def build_full_channel_geometry(
    chmapfull_arr: np.ndarray,
    probe1_name: str,
    probe2_name: str,
    probe2_x_shift_um: float,
    probe2_y_shift_um: float,
):
    """Return all physical channel sites for both probes using full chmapfull geometry."""
    arr = np.asarray(chmapfull_arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected chmapfull_arr as N x 3+, got shape {arr.shape}")

    ch = arr[:, 0].astype(int)
    x = arr[:, 1].astype(float)
    y = arr[:, 2].astype(float)

    geom1 = pd.DataFrame(
        {
            "source_probe": probe1_name,
            "probe_index": 1,
            "chmapfull_channel": ch,
            "x_um_unshifted": x,
            "y_um_unshifted": y,
            "probe_x_shift_um": 0.0,
            "probe_y_shift_um": 0.0,
            "x_um": x,
            "y_um": y,
        }
    )
    geom2 = pd.DataFrame(
        {
            "source_probe": probe2_name,
            "probe_index": 2,
            "chmapfull_channel": ch,
            "x_um_unshifted": x,
            "y_um_unshifted": y,
            "probe_x_shift_um": float(probe2_x_shift_um),
            "probe_y_shift_um": float(probe2_y_shift_um),
            "x_um": x + float(probe2_x_shift_um),
            "y_um": y + float(probe2_y_shift_um),
        }
    )
    return pd.concat([geom1, geom2], ignore_index=True)


def rebuild_channel_positions_from_chmapfull(
    channel_map: np.ndarray,
    ch_to_xy: dict[int, tuple[float, float]],
    x_shift: float = 0.0,
    y_shift: float = 0.0,
):
    """Create channel_positions rows matching channel_map values when possible."""
    coords = []
    missing = []
    for ch in channel_map.astype(int):
        ch_key = int(ch)
        if ch_key in ch_to_xy:
            x, y = ch_to_xy[ch_key]
        elif ch_key - 1 in ch_to_xy:
            x, y = ch_to_xy[ch_key - 1]
        elif ch_key + 1 in ch_to_xy:
            x, y = ch_to_xy[ch_key + 1]
        else:
            missing.append(ch_key)
            coords.append((np.nan, np.nan))
            continue
        coords.append((float(x) + float(x_shift), float(y) + float(y_shift)))

    arr = np.asarray(coords, dtype=float)
    return arr, missing


# -----------------------------
# Merge/unitmetrics construction
# -----------------------------
def count_spikes_by_cluster(spike_clusters: np.ndarray, cluster_ids: np.ndarray) -> dict[int, int]:
    return {int(cid): int(np.sum(spike_clusters == int(cid))) for cid in cluster_ids}


def check_selected_clusters_present(units: pd.DataFrame, spike_clusters: np.ndarray, probe_name: str):
    selected = set(units["original_cluster_id"].astype(int).tolist())
    present = set(np.unique(spike_clusters).astype(int).tolist())
    missing = sorted(selected - present)
    if missing:
        msg = (
            f"{probe_name}: {len(missing)} selected GoodUnitInfo ClusterIDs were not found in spike_clusters.npy. "
            f"First few missing: {missing[:10]}"
        )
        if ERROR_ON_MISSING_SELECTED_CLUSTERS:
            raise ValueError(msg)
        print("WARNING:", msg)


def build_unitmetrics(units1: pd.DataFrame, units2: pd.DataFrame, sc1: np.ndarray, sc2: np.ndarray):
    units1 = units1.copy()
    units2 = units2.copy()

    units1["cluster_id"] = units1["original_cluster_id"].astype(int)
    units2["cluster_id"] = units2["original_cluster_id"].astype(int) + int(CLUSTER_OFFSET)

    # `ch` is kept for compatibility/debugging. Main.py should prefer x_um/y_um.
    units1["ch"] = units1["channelnum_original"].astype(int)
    units2["ch"] = units2["channelnum_original"].astype(int) + int(CHANNEL_OFFSET)

    units1["x_um"] = units1["x_um_unshifted"].astype(float)
    units1["y_um"] = units1["y_um_unshifted"].astype(float)
    units2["x_um"] = units2["x_um_unshifted"].astype(float) + units2["probe_x_shift_um"].astype(float)
    units2["y_um"] = units2["y_um_unshifted"].astype(float) + units2["probe_y_shift_um"].astype(float)

    units1["coordinate_source"] = "chmapfull.mat via GoodUnitInfo.Channel"
    units2["coordinate_source"] = "chmapfull.mat via GoodUnitInfo.Channel + probe2 x/y shift"

    counts1 = count_spikes_by_cluster(sc1, units1["original_cluster_id"].astype(int).to_numpy())
    counts2 = count_spikes_by_cluster(sc2, units2["original_cluster_id"].astype(int).to_numpy())
    units1["num_spikes"] = [counts1[int(cid)] for cid in units1["original_cluster_id"]]
    units2["num_spikes"] = [counts2[int(cid)] for cid in units2["original_cluster_id"]]

    keep_cols = [
        "cluster_id",
        "good",
        "isGood",
        "ch",
        "x_um",
        "y_um",
        "x_um_unshifted",
        "y_um_unshifted",
        "probe_x_shift_um",
        "probe_y_shift_um",
        "num_spikes",
        "source_probe",
        "original_cluster_id",
        "regname",
        "region",
        "fsc_label",
        "shanknum",
        "channelnum_original",
        "chmapfull_channel",
        "coordinate_source",
    ]
    for opt in ["csv_x_position", "csv_depth", "cell_type", "csv_num_spikes"]:
        if opt in units1.columns or opt in units2.columns:
            if opt not in units1.columns:
                units1[opt] = np.nan
            if opt not in units2.columns:
                units2[opt] = np.nan
            keep_cols.append(opt)

    return pd.concat([units1[keep_cols], units2[keep_cols]], ignore_index=True)


def ask_float_if_none(value, prompt: str) -> float:
    if value is not None:
        return float(value)
    return float(input(prompt).strip())


def main():
    '''
    animals = ["38"] # edge case, just run another time with codeblock below instead
    for aname in animals:
        # Probe #1: VC
        KS_DIR_1 = Path(f"Z:\\Justin\\VR mice\\VR{aname}-noBLCK3\\VC")
        # Probe #2: PPC
        KS_DIR_2 = Path(f"Z:\\Justin\\VR mice\\VR{aname}-noBLCK3\\PPC")
        GOOD_UNIT_CSV_1 = Path(f"Z:\\Justin\\VR mice\\VR{aname}-noBLCK3\\VC\\UnitMetrics\\VR{aname}_GoodUnitInfo.csv")
        GOOD_UNIT_CSV_2 = Path(f"Z:\\Justin\\VR mice\\VR{aname}-noBLCK3\\PPC\\UnitMetrics\\VR{aname}_GoodUnitInfo.csv")
        MERGED_KS_DIR = Path(f"E:\\rob\\justin\\merged_VR{aname}-noBLCK3_VC_PPC")
    '''
    #'''
    animals = ["29", "32", "33", "34", "37", "39", "42", "43", "44", "45", "472", "48"]
    for aname in animals:
        # Probe #1: VC
        KS_DIR_1 = Path(f"Z:\\Justin\\VR mice\\VR{aname}\\VC")
        # Probe #2: PPC
        KS_DIR_2 = Path(f"Z:\\Justin\\VR mice\\VR{aname}\\PPC")
        GOOD_UNIT_CSV_1 = Path(f"Z:\\Justin\\VR mice\\VR{aname}\\VC\\UnitMetrics\\VR{aname}_GoodUnitInfo.csv")
        GOOD_UNIT_CSV_2 = Path(f"Z:\\Justin\\VR mice\\VR{aname}\\PPC\\UnitMetrics\\VR{aname}_GoodUnitInfo.csv")
        MERGED_KS_DIR = Path(f"E:\\rob\\justin\\merged_VR{aname}_VC_PPC")
    #'''
        ks1 = Path(KS_DIR_1)
        ks2 = Path(KS_DIR_2)
        csv1 = Path(GOOD_UNIT_CSV_1)
        csv2 = Path(GOOD_UNIT_CSV_2)
        chmap_path = Path(CHMAPFULL_PATH)
        out_dir = Path(MERGED_KS_DIR)

        # Preflight region/unit-count threshold before creating or overwriting output.
        # This filters animals by selected GoodUnitInfo rows only; all rows in each
        # GoodUnitInfo.csv are assumed good, and Region substring filtering is applied.
        n_units_probe1_preflight = count_goodunitinfo_units(csv1, REGION_CONTAINS_1, PROBE1_NAME)
        n_units_probe2_preflight = count_goodunitinfo_units(csv2, REGION_CONTAINS_2, PROBE2_NAME)

        min_probe1 = 0 if MIN_UNITS_PROBE1 is None else int(MIN_UNITS_PROBE1)
        min_probe2 = 0 if MIN_UNITS_PROBE2 is None else int(MIN_UNITS_PROBE2)

        if n_units_probe1_preflight < min_probe1 or n_units_probe2_preflight < min_probe2:
            print(
                f"\nSkipping VR{aname}: "
                f"{PROBE1_NAME} has {n_units_probe1_preflight} selected units "
                f"(minimum {min_probe1}); "
                f"{PROBE2_NAME} has {n_units_probe2_preflight} selected units "
                f"(minimum {min_probe2})."
            )
            continue

        print(
            f"\nVR{aname} passes unit threshold: "
            f"{PROBE1_NAME}={n_units_probe1_preflight} >= {min_probe1}, "
            f"{PROBE2_NAME}={n_units_probe2_preflight} >= {min_probe2}"
        )

        if out_dir.exists():
            if not OVERWRITE_OUTPUT_DIR:
                raise FileExistsError(
                    f"Output directory already exists: {out_dir}\n"
                    f"Set OVERWRITE_OUTPUT_DIR=True only if you are okay deleting/recreating that folder."
                )
            shutil.rmtree(out_dir)

        out_dir.mkdir(parents=True, exist_ok=False)

        print("\nPositive offset convention:")
        print("  +x shifts probe 2 to the RIGHT of probe 1 in plot coordinates.")
        print("  +y shifts probe 2 UP relative to probe 1 in plot coordinates.\n")

        probe2_x_offset = ask_float_if_none(
            PROBE2_X_OFFSET_UM,
            "Enter probe2 x offset relative to probe1 in microns (+ = right):\n\n",
        )
        probe2_y_offset = ask_float_if_none(
            PROBE2_Y_OFFSET_UM,
            "Enter probe2 y offset relative to probe1 in microns (+ = up):\n\n",
        )

        st1, sc1, cmap1, cpos1 = load_ks_arrays(ks1)
        st2, sc2, cmap2, cpos2 = load_ks_arrays(ks2)

        if st1.dtype != st2.dtype:
            raise TypeError(f"spike_times dtypes differ: {st1.dtype} vs {st2.dtype}")
        if sc1.dtype != sc2.dtype:
            raise TypeError(f"spike_clusters dtypes differ: {sc1.dtype} vs {sc2.dtype}")
        if cmap1.dtype != cmap2.dtype:
            raise TypeError(f"channel_map dtypes differ: {cmap1.dtype} vs {cmap2.dtype}")
        if cpos1.dtype != cpos2.dtype:
            raise TypeError(f"channel_positions dtypes differ: {cpos1.dtype} vs {cpos2.dtype}")

        ch_to_xy, chmapfull_arr = load_chmapfull(chmap_path)

        units1 = load_goodunitinfo_csv(csv1, REGION_CONTAINS_1, PROBE1_NAME)
        units2 = load_goodunitinfo_csv(csv2, REGION_CONTAINS_2, PROBE2_NAME)

        check_selected_clusters_present(units1, sc1, PROBE1_NAME)
        check_selected_clusters_present(units2, sc2, PROBE2_NAME)

        units1 = attach_chmapfull_coordinates(units1, ch_to_xy, PROBE1_NAME)
        units2 = attach_chmapfull_coordinates(units2, ch_to_xy, PROBE2_NAME)
        units1["probe_x_shift_um"] = 0.0
        units1["probe_y_shift_um"] = 0.0
        units2["probe_x_shift_um"] = float(probe2_x_offset)
        units2["probe_y_shift_um"] = float(probe2_y_offset)

        cids1 = units1["original_cluster_id"].astype(sc1.dtype, copy=False).to_numpy()
        cids2 = units2["original_cluster_id"].astype(sc2.dtype, copy=False).to_numpy()

        shifted_cids2 = units2["original_cluster_id"].astype(np.int64).to_numpy() + int(CLUSTER_OFFSET)
        shifted_cmap2 = cmap2.astype(np.int64) + int(CHANNEL_OFFSET)

        _dtype_can_hold_values(sc1.dtype, shifted_cids2, "shifted probe 2 cluster IDs")
        _dtype_can_hold_values(cmap1.dtype, shifted_cmap2, "shifted probe 2 channel IDs")

        mask1 = np.isin(sc1, cids1)
        mask2 = np.isin(sc2, cids2)

        merged_spike_times = np.concatenate([st1[mask1], st2[mask2]])
        merged_spike_clusters = np.concatenate(
            [
                sc1[mask1].astype(np.int64, copy=False),
                sc2[mask2].astype(np.int64, copy=False) + int(CLUSTER_OFFSET),
            ]
        )

        if merged_spike_times.size > 1:
            order = np.argsort(merged_spike_times, kind="mergesort")
            merged_spike_times = merged_spike_times[order]
            merged_spike_clusters = merged_spike_clusters[order]

        _dtype_can_hold_values(sc1.dtype, merged_spike_clusters, "merged spike cluster IDs")

        # Full physical channel geometry for clean plotting/background sites.
        full_channel_geometry = build_full_channel_geometry(
            chmapfull_arr=chmapfull_arr,
            probe1_name=PROBE1_NAME,
            probe2_name=PROBE2_NAME,
            probe2_x_shift_um=float(probe2_x_offset),
            probe2_y_shift_um=float(probe2_y_offset),
        )

        # channel_positions.npy is kept for KS-like compatibility. Main.py should prefer
        # UnitMetrics.csv x/y and full_channel_geometry.csv when present.
        cpos1_from_chmap, missing_cpos1 = rebuild_channel_positions_from_chmapfull(
            cmap1, ch_to_xy, x_shift=0.0, y_shift=0.0
        )
        cpos2_from_chmap, missing_cpos2 = rebuild_channel_positions_from_chmapfull(
            cmap2, ch_to_xy, x_shift=probe2_x_offset, y_shift=probe2_y_offset
        )
        if missing_cpos1 or missing_cpos2:
            print("WARNING: Could not map some channel_map entries to chmapfull; falling back to KS channel_positions for those rows.")
            if missing_cpos1:
                bad = np.isnan(cpos1_from_chmap[:, 0])
                cpos1_from_chmap[bad, :] = cpos1[bad, :]
            if missing_cpos2:
                bad = np.isnan(cpos2_from_chmap[:, 0])
                cpos2_fallback = cpos2.copy()
                cpos2_fallback[:, 0] = cpos2_fallback[:, 0] + float(probe2_x_offset)
                cpos2_fallback[:, 1] = cpos2_fallback[:, 1] + float(probe2_y_offset)
                cpos2_from_chmap[bad, :] = cpos2_fallback[bad, :]

        merged_channel_positions = np.vstack([cpos1_from_chmap, cpos2_from_chmap]).astype(cpos1.dtype, copy=False)
        merged_channel_map = np.concatenate(
            [
                cmap1.astype(np.int64, copy=False),
                cmap2.astype(np.int64, copy=False) + int(CHANNEL_OFFSET),
            ]
        )

        _dtype_can_hold_values(cmap1.dtype, merged_channel_map, "merged channel_map values")
        merged_channel_map = merged_channel_map.astype(cmap1.dtype, copy=False)

        unitmetrics = build_unitmetrics(units1=units1, units2=units2, sc1=sc1, sc2=sc2)

        cluster_info = pd.DataFrame(
            {
                "cluster_id": unitmetrics["cluster_id"].astype(int),
                "group": "good",
                "ch": unitmetrics["ch"].astype(int),
                "n_spikes": unitmetrics["num_spikes"].astype(int),
                "source_probe": unitmetrics["source_probe"],
                "regname": unitmetrics["regname"],
                "region": unitmetrics["region"],
                "shanknum": unitmetrics["shanknum"].astype(int),
            }
        )

        np.save(out_dir / "spike_times.npy", merged_spike_times.astype(st1.dtype, copy=False))
        np.save(out_dir / "spike_clusters.npy", merged_spike_clusters.astype(sc1.dtype, copy=False))
        np.save(out_dir / "channel_map.npy", merged_channel_map)
        np.save(out_dir / "channel_positions.npy", merged_channel_positions)

        unitmetrics.to_csv(out_dir / "UnitMetrics.csv", index=False)
        cluster_info.to_csv(out_dir / "cluster_info.tsv", sep="\t", index=False)
        full_channel_geometry.to_csv(out_dir / "full_channel_geometry.csv", index=False)

        manifest = {
            "ks_dir_1": str(ks1),
            "ks_dir_2": str(ks2),
            "good_unit_csv_1": str(csv1),
            "good_unit_csv_2": str(csv2),
            "chmapfull_path": str(chmap_path),
            "merged_ks_dir": str(out_dir),
            "probe1_name": PROBE1_NAME,
            "probe2_name": PROBE2_NAME,
            "region_contains_1": list(REGION_CONTAINS_1) if REGION_CONTAINS_1 is not None else [],
            "region_contains_2": list(REGION_CONTAINS_2) if REGION_CONTAINS_2 is not None else [],
            "case_sensitive_region": bool(CASE_SENSITIVE_REGION),
            "cluster_offset": int(CLUSTER_OFFSET),
            "channel_offset": int(CHANNEL_OFFSET),
            "probe2_x_offset_um": float(probe2_x_offset),
            "probe2_y_offset_um": float(probe2_y_offset),
            "positive_offset_convention": "+x = right, +y = up for probe 2 relative to probe 1",
            "n_units_probe1": int(len(units1)),
            "n_units_probe2": int(len(units2)),
            "n_units_total": int(len(unitmetrics)),
            "n_spikes_probe1_selected": int(np.sum(mask1)),
            "n_spikes_probe2_selected": int(np.sum(mask2)),
            "n_spikes_total": int(merged_spike_times.size),
            "spike_times_dtype": str(st1.dtype),
            "spike_clusters_dtype": str(sc1.dtype),
            "channel_map_dtype": str(cmap1.dtype),
            "channel_positions_dtype": str(cpos1.dtype),
            "unitmetrics_good_source": "all rows in GoodUnitInfo.csv are assumed good; region filter applied",
            "unitmetrics_coordinate_source": "chmapfull.mat via GoodUnitInfo.Channel; CSV XPosition/Depth not used for coordinates",
        }

        with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\nMerged KS directory created:")
        print(f"  {out_dir}")
        print("\nSelected units:")
        print(f"  {PROBE1_NAME}: {len(units1)}")
        print(f"  {PROBE2_NAME}: {len(units2)}")
        print(f"  total: {len(unitmetrics)}")
        print("\nProbe 2 offsets:")
        print(f"  x offset: {probe2_x_offset:.3f} µm  (+ = right)")
        print(f"  y offset: {probe2_y_offset:.3f} µm  (+ = up)")
        print("\nSelected spikes:")
        print(f"  {PROBE1_NAME}: {int(np.sum(mask1)):,}")
        print(f"  {PROBE2_NAME}: {int(np.sum(mask2)):,}")
        print(f"  total: {merged_spike_times.size:,}")
        print("\nOutput dtypes:")
        print(f"  spike_times.npy: {st1.dtype}")
        print(f"  spike_clusters.npy: {sc1.dtype}")
        print(f"  channel_map.npy: {cmap1.dtype}")
        print(f"  channel_positions.npy: {cpos1.dtype}")
        print("\nWrote:")
        print("  spike_times.npy")
        print("  spike_clusters.npy")
        print("  channel_map.npy")
        print("  channel_positions.npy")
        print("  UnitMetrics.csv  (includes x_um/y_um from chmapfull.mat)")
        print("  full_channel_geometry.csv  (all chmapfull sites for both probes, with x/y offsets)")
        print("  cluster_info.tsv")
        print("  merge_manifest.json")


if __name__ == "__main__":
    main()
