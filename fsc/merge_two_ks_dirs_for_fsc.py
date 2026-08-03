#!/usr/bin/env python3
"""
merge_two_ks_dirs_with_unitdata_for_fsc.py

Standalone utility for merging two simultaneously recorded, separately sorted
Kilosort output folders into one minimal KS-like folder for FSC analysis.

ASSUMPTIONS / BUG-RISK NOTES
----------------------------
1. The two KS folders come from recordings with the same Intan sample clock and
   the same sample-time origin. This script does NOT apply any time shift.
2. The MATLAB unitdata .mat files contain one struct array named `unitdata`.
3. The unitdata fields are assumed to include:
      clusterID
      regname
      shanknum
      channelnum
   and a good-unit field matching `isGood` or `good`, case-insensitive.
4. Region filtering uses substring matching on `regname`, similar to MATLAB:
      contains({unitdata.regname}, "SS")
   By default this is case-sensitive, matching MATLAB's default behavior.
5. Spike filtering is done from the raw KS `spike_times.npy` / `spike_clusters.npy`
   using selected good cluster IDs from the .mat files. The unitdata `spikets`
   field is NOT used to rebuild spikes.
6. Probe 2 cluster IDs are shifted by CLUSTER_OFFSET to prevent collisions.
7. Probe 2 channel IDs are shifted by CHANNEL_OFFSET to prevent channel-map
   collisions. UnitMetrics.csv `ch` values and channel_map.npy are shifted
   together.
8. UnitMetrics.csv `ch` values are taken from each KS directory's cluster_info.tsv,
   not from unitdata.channelnum. This matters because KS/Phy can reindex kept
   channels after dead-channel removal.
9. UnitMetrics.csv physical coordinates (`x_um`, `y_um`) are taken from
   CHMAPFULL_PATH using unitdata.channelnum. This is intentionally independent
   of KS/Phy channel reindexing and should be the better distance/sorting source.
10. Probe 2 x coordinates are shifted by aligning the FULL chmapfull probe
    geometry, not the median of selected good units. This prevents animal-to-animal
    visual spacing changes caused by uneven unit yield. If both probes use the same
    chmapfull coordinate system, entering 2000 simply makes probe 2 x = probe 1 x + 2000.
11. This script writes full_channel_geometry.csv containing every physical channel
    from chmapfull for both probes. Downstream plotting should use this for background
    probe-site squares, so disabled/removed KS channels do not appear as missing geometry.
12. This script also rebuilds channel_positions.npy from CHMAPFULL_PATH when it can
   map merged channel_map values back to full-probe channel numbers. The downstream
   FSC script should still prefer UnitMetrics.csv x/y columns for unit positions.
13. This script only writes to MERGED_KS_DIR. It does not modify KS_DIR_1,
    KS_DIR_2, MAT_PATH_1, MAT_PATH_2, or CHMAPFULL_PATH. If OVERWRITE_OUTPUT_DIR=True
    and MERGED_KS_DIR already exists, MERGED_KS_DIR will be deleted and recreated.
14. The merged spike_times.npy dtype is preserved from the original KS files.
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
n = 31
KS_DIR_1 = Path(f"Y:\Winny\FinishedPhy\TR{n}_S1_spikesorted_DONE")
KS_DIR_2 = Path(f"Y:\Winny\FinishedPhy\TR{n}_HPC_spikesorted_DONE")

MAT_PATH_1 = Path(f"Y:\Winny\FinishedPhy\TR{n}_S1_spikesorted_DONE\Analysis\TR{n}_S1_unitdata_CQ.mat")
MAT_PATH_2 = Path(f"Y:\Winny\FinishedPhy\TR{n}_HPC_spikesorted_DONE\Analysis\TR{n}_PPC_HPC_unitdata_CQ.mat")

MERGED_KS_DIR = Path(f"E:\\rob2\\merged_TR{n}_S1_V1")

# Full probe map used by Winny's plotting/QC workflow.
# Expected variable: chmapfull, with columns [channel_number, x_um, y_um]
# after transposing if stored as 3 x N in MATLAB v7.3/HDF5.
CHMAPFULL_PATH = Path(r'Y:\Winny\Data\chmapfull.mat')

PROBE1_NAME = "S1"
PROBE2_NAME = "V1"

# Examples:
#   S1 / SS: ("SS",)
#   VISa:    ("VISa",)
#   HPC:     ("CA1", "SUB") or whatever substring is appropriate for that file.
REGION_CONTAINS_1 = ("SS",)
REGION_CONTAINS_2 = ("VISa",)

CASE_SENSITIVE_REGION = False

# If None, script asks interactively.
PROBE2_CENTER_X_MINUS_PROBE1_CENTER_X_UM = 1800

CLUSTER_OFFSET = 1_000_000
CHANNEL_OFFSET = 1_000_000

OVERWRITE_OUTPUT_DIR = False


# -----------------------------
# MATLAB struct helpers
# -----------------------------
def _as_list(x):
    if isinstance(x, np.ndarray):
        return list(x.reshape(-1))
    return [x]


def _field_names(obj):
    if hasattr(obj, "_fieldnames"):
        return list(obj._fieldnames)
    if isinstance(obj, np.void) and obj.dtype.names:
        return list(obj.dtype.names)
    if isinstance(obj, dict):
        return list(obj.keys())
    return []


def _get_field(obj, field_name):
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if isinstance(obj, np.void) and obj.dtype.names and field_name in obj.dtype.names:
        return obj[field_name]
    if isinstance(obj, dict) and field_name in obj:
        return obj[field_name]
    raise AttributeError(f"Field {field_name!r} not found in MATLAB struct.")


def _find_field(obj, candidates):
    fields = _field_names(obj)
    lower_to_original = {f.lower(): f for f in fields}
    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    raise KeyError(f"Could not find any of {candidates}. Available fields: {fields}")


def _scalar(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return x


def _string(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S"}:
        if arr.shape == ():
            return str(arr.item())
        return "".join(str(v) for v in arr.reshape(-1))
    if arr.dtype == object and arr.size == 1:
        return _string(arr.reshape(-1)[0])
    return str(x)


def _region_matches(regname: str, substrings, case_sensitive: bool):
    if not substrings:
        return True

    hay = regname if case_sensitive else regname.lower()
    for sub in substrings:
        needle = str(sub) if case_sensitive else str(sub).lower()
        if needle in hay:
            return True
    return False


def _load_selected_units_from_mat_v7(mat_path: Path, region_contains, probe_name: str):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "unitdata" not in mat:
        raise KeyError(f"{mat_path} does not contain a variable named 'unitdata'.")

    units = _as_list(mat["unitdata"])
    if not units:
        raise ValueError(f"No units found in {mat_path}")

    first = units[0]
    isgood_field = _find_field(first, ("isGood", "good"))
    cluster_field = _find_field(first, ("clusterID",))
    reg_field = _find_field(first, ("regname",))
    shank_field = _find_field(first, ("shanknum",))
    channel_field = _find_field(first, ("channelnum",))

    rows = []
    for u in units:
        is_good = int(_scalar(_get_field(u, isgood_field))) == 1
        regname = _string(_get_field(u, reg_field))

        if not is_good:
            continue
        if not _region_matches(regname, region_contains, CASE_SENSITIVE_REGION):
            continue

        rows.append(
            {
                "original_cluster_id": int(_scalar(_get_field(u, cluster_field))),
                "regname": regname,
                "region": regname,
                "fsc_label": regname,
                "shanknum": int(_scalar(_get_field(u, shank_field))),
                "channelnum_original": int(_scalar(_get_field(u, channel_field))),
                "source_probe": probe_name,
                "good": True,
                "isGood": True,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No units passed filters for {mat_path}. "
            f"region_contains={region_contains}, good field={isgood_field}"
        )

    return df


def _h5_read_dataset_value(f, obj):
    """Read a MATLAB v7.3/HDF5 dataset or referenced object into a Python scalar/string when possible."""
    import h5py

    if isinstance(obj, h5py.Reference):
        if not obj:
            return None
        obj = f[obj]

    if isinstance(obj, h5py.Group):
        # Rare for scalar struct fields. Keep it inspectable rather than failing cryptically.
        return {k: _h5_read_dataset_value(f, obj[k]) for k in obj.keys()}

    arr = np.asarray(obj[()] if hasattr(obj, "__getitem__") else obj)

    matlab_class = obj.attrs.get("MATLAB_class", b"") if hasattr(obj, "attrs") else b""
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode(errors="replace")

    if matlab_class == "char" or arr.dtype.kind in {"U", "S"}:
        return _h5_decode_char_array(arr)

    # MATLAB v7.3 chars usually arrive as uint16 code units.
    if arr.dtype == np.uint16 and arr.size > 1:
        return _h5_decode_char_array(arr)

    if arr.dtype == object:
        flat = arr.reshape(-1, order="F")
        if flat.size == 1:
            return _h5_read_dataset_value(f, flat[0])
        return [_h5_read_dataset_value(f, x) for x in flat]

    if arr.size == 1:
        return arr.reshape(-1)[0].item()

    return arr


def _h5_decode_char_array(arr):
    arr = np.asarray(arr)

    if arr.dtype.kind == "S":
        return b"".join(arr.reshape(-1, order="F")).decode(errors="replace")
    if arr.dtype.kind == "U":
        return "".join(arr.reshape(-1, order="F").astype(str))

    # MATLAB stores chars as uint16 code units in v7.3 files.
    vals = arr.reshape(-1, order="F")
    chars = []
    for v in vals:
        iv = int(v)
        if iv != 0:
            chars.append(chr(iv))
    return "".join(chars)


def _h5_flatten_field_dataset(ds):
    arr = np.asarray(ds)
    return arr.reshape(-1, order="F")


def _h5_find_field(field_names, candidates):
    lower_to_original = {str(f).lower(): str(f) for f in field_names}
    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    raise KeyError(f"Could not find any of {candidates}. Available fields: {list(field_names)}")


def _load_selected_units_from_mat_v73(mat_path: Path, region_contains, probe_name: str):
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "This .mat file is MATLAB v7.3/HDF5. Install h5py in this environment, "
            "or resave the MATLAB file with save(..., '-v7')."
        ) from e

    with h5py.File(mat_path, "r") as f:
        if "unitdata" not in f:
            raise KeyError(f"{mat_path} does not contain a variable named 'unitdata'.")

        unitdata = f["unitdata"]

        if isinstance(unitdata, h5py.Group):
            field_names = list(unitdata.keys())
            isgood_field = _h5_find_field(field_names, ("isGood", "good"))
            cluster_field = _h5_find_field(field_names, ("clusterID",))
            reg_field = _h5_find_field(field_names, ("regname",))
            shank_field = _h5_find_field(field_names, ("shanknum",))
            channel_field = _h5_find_field(field_names, ("channelnum",))

            fields = {
                "isgood": _h5_flatten_field_dataset(unitdata[isgood_field]),
                "cluster": _h5_flatten_field_dataset(unitdata[cluster_field]),
                "reg": _h5_flatten_field_dataset(unitdata[reg_field]),
                "shank": _h5_flatten_field_dataset(unitdata[shank_field]),
                "channel": _h5_flatten_field_dataset(unitdata[channel_field]),
            }
            n = min(len(v) for v in fields.values())

            rows = []
            for i in range(n):
                is_good = int(_scalar(_h5_read_dataset_value(f, fields["isgood"][i]))) == 1
                regname = _string(_h5_read_dataset_value(f, fields["reg"][i]))

                if not is_good:
                    continue
                if not _region_matches(regname, region_contains, CASE_SENSITIVE_REGION):
                    continue

                rows.append(
                    {
                        "original_cluster_id": int(_scalar(_h5_read_dataset_value(f, fields["cluster"][i]))),
                        "regname": regname,
                        "region": regname,
                        "fsc_label": regname,
                        "shanknum": int(_scalar(_h5_read_dataset_value(f, fields["shank"][i]))),
                        "channelnum_original": int(_scalar(_h5_read_dataset_value(f, fields["channel"][i]))),
                        "source_probe": probe_name,
                        "good": True,
                        "isGood": True,
                    }
                )

        else:
            # Less common layout: unitdata is an array of object references to per-unit structs.
            refs = _h5_flatten_field_dataset(unitdata)
            first_group = f[refs[0]]
            field_names = list(first_group.keys())
            isgood_field = _h5_find_field(field_names, ("isGood", "good"))
            cluster_field = _h5_find_field(field_names, ("clusterID",))
            reg_field = _h5_find_field(field_names, ("regname",))
            shank_field = _h5_find_field(field_names, ("shanknum",))
            channel_field = _h5_find_field(field_names, ("channelnum",))

            rows = []
            for ref in refs:
                g = f[ref]
                is_good = int(_scalar(_h5_read_dataset_value(f, g[isgood_field]))) == 1
                regname = _string(_h5_read_dataset_value(f, g[reg_field]))

                if not is_good:
                    continue
                if not _region_matches(regname, region_contains, CASE_SENSITIVE_REGION):
                    continue

                rows.append(
                    {
                        "original_cluster_id": int(_scalar(_h5_read_dataset_value(f, g[cluster_field]))),
                        "regname": regname,
                        "region": regname,
                        "fsc_label": regname,
                        "shanknum": int(_scalar(_h5_read_dataset_value(f, g[shank_field]))),
                        "channelnum_original": int(_scalar(_h5_read_dataset_value(f, g[channel_field]))),
                        "source_probe": probe_name,
                        "good": True,
                        "isGood": True,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No units passed filters for {mat_path}. "
            f"region_contains={region_contains}. If this seems wrong, inspect regname strings in the .mat file."
        )
    return df


def load_selected_units_from_mat(mat_path: Path, region_contains, probe_name: str):
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    try:
        return _load_selected_units_from_mat_v7(mat_path, region_contains, probe_name)
    except NotImplementedError:
        print(f"  {mat_path.name} is MATLAB v7.3/HDF5; reading with h5py fallback")
        return _load_selected_units_from_mat_v73(mat_path, region_contains, probe_name)



# -----------------------------
# KS loading / merge helpers
# -----------------------------
def load_ks_arrays(ks_dir: Path):
    ks_dir = Path(ks_dir)
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


def _dtype_can_hold_values(dtype, values, label):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        vmin = int(np.min(values)) if len(values) else 0
        vmax = int(np.max(values)) if len(values) else 0
        if vmin < info.min or vmax > info.max:
            raise OverflowError(
                f"{label} values [{vmin}, {vmax}] do not fit into {dtype}. "
                f"Use a smaller offset or a wider dtype."
            )



def load_chmapfull(chmap_path: Path):
    """Load Winny-style chmapfull.mat and return channel -> (x_um, y_um)."""
    chmap_path = Path(chmap_path)
    if not chmap_path.exists():
        raise FileNotFoundError(f"CHMAPFULL_PATH not found: {chmap_path}")

    arr = None
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

    # Winny's script does: chmapfull = np.array(f['chmapfull']).T
    # We accept either 3 x N or N x 3.
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
    """Attach physical x/y coordinates using unitdata.channelnum and chmapfull."""
    out = unit_df.copy()
    xs = []
    ys = []
    mapped_channels = []
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
        xs.append(x)
        ys.append(y)
        mapped_channels.append(mapped_ch)

    if missing:
        raise ValueError(
            f"{probe_name}: {len(missing)} unitdata.channelnum values could not be found in chmapfull. "
            f"First few missing: {missing[:10]}"
        )

    out["chmapfull_channel"] = mapped_channels
    out["x_um_unshifted"] = xs
    out["y_um"] = ys
    return out


def rebuild_channel_positions_from_chmapfull(channel_map: np.ndarray, ch_to_xy: dict[int, tuple[float, float]], x_shift: float = 0.0):
    """Create channel_positions rows matching channel_map values when possible."""
    coords = []
    missing = []
    for ch in channel_map.astype(int):
        if int(ch) in ch_to_xy:
            x, y = ch_to_xy[int(ch)]
        elif int(ch) - 1 in ch_to_xy:
            x, y = ch_to_xy[int(ch) - 1]
        elif int(ch) + 1 in ch_to_xy:
            x, y = ch_to_xy[int(ch) + 1]
        else:
            missing.append(int(ch))
            coords.append((np.nan, np.nan))
            continue
        coords.append((float(x) + float(x_shift), float(y)))

    arr = np.asarray(coords, dtype=float)
    return arr, missing


def build_full_channel_geometry(chmapfull_arr: np.ndarray, probe1_name: str, probe2_name: str, probe2_x_shift_um: float):
    """Return all physical channel sites for both probes using full chmapfull geometry.

    This is intentionally independent of KS channel_map/channel_positions because
    KS may omit bad/dead channels and reindex the kept channels. The output is for
    visualization/background geometry and distance sanity checks.
    """
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
            "y_um": y,
            "probe_x_shift_um": 0.0,
            "x_um": x,
        }
    )
    geom2 = pd.DataFrame(
        {
            "source_probe": probe2_name,
            "probe_index": 2,
            "chmapfull_channel": ch,
            "x_um_unshifted": x,
            "y_um": y,
            "probe_x_shift_um": float(probe2_x_shift_um),
            "x_um": x + float(probe2_x_shift_um),
        }
    )
    return pd.concat([geom1, geom2], ignore_index=True)


def load_cluster_info_channels(ks_dir: Path):
    """Load cluster_info.tsv and return cluster_id -> KS/Phy peak channel (`ch`)."""
    path = Path(ks_dir) / "cluster_info.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cluster_info.tsv in {ks_dir}. Open/save in Phy first, or provide a KS dir with cluster_info.tsv."
        )

    df = pd.read_csv(path, sep="\t")

    if "cluster_id" in df.columns:
        cluster_col = "cluster_id"
    elif "id" in df.columns:
        cluster_col = "id"
    else:
        raise ValueError(f"{path} needs a cluster_id or id column. Columns: {list(df.columns)}")

    if "ch" not in df.columns:
        raise ValueError(f"{path} needs a ch column. Columns: {list(df.columns)}")

    out = df[[cluster_col, "ch"]].copy()
    out.columns = ["original_cluster_id", "ch_from_cluster_info"]
    out["original_cluster_id"] = out["original_cluster_id"].astype(int)
    out["ch_from_cluster_info"] = out["ch_from_cluster_info"].astype(int)
    return out


def attach_cluster_info_channels(unit_df: pd.DataFrame, cluster_info_channels: pd.DataFrame, probe_name: str):
    """Attach KS/Phy-compatible peak-channel values using cluster_info.tsv.

    Do NOT use unitdata.channelnum for FSC `ch`; it can refer to original probe
    channels and fail after KS dead-channel reindexing.
    """
    out = unit_df.merge(cluster_info_channels, on="original_cluster_id", how="left", validate="one_to_one")

    missing = out.loc[out["ch_from_cluster_info"].isna(), "original_cluster_id"].astype(int).to_list()
    if missing:
        raise ValueError(
            f"{probe_name}: {len(missing)} selected units from unitdata were not found in cluster_info.tsv. "
            f"First few missing cluster IDs: {missing[:10]}"
        )

    out["ch_for_ks"] = out["ch_from_cluster_info"].astype(int)
    out["channelnum_interpretation"] = "cluster_info_ch_after_KS_reindexing"
    return out


def count_spikes_by_cluster(spike_clusters: np.ndarray, cluster_ids: np.ndarray):
    counts = {}
    for cid in cluster_ids:
        counts[int(cid)] = int(np.sum(spike_clusters == int(cid)))
    return counts


def build_unitmetrics(units1, units2, sc1, sc2):
    units1 = units1.copy()
    units2 = units2.copy()

    units1["cluster_id"] = units1["original_cluster_id"].astype(int)
    units2["cluster_id"] = units2["original_cluster_id"].astype(int) + int(CLUSTER_OFFSET)

    units1["ch"] = units1["ch_for_ks"].astype(int)
    units2["ch"] = units2["ch_for_ks"].astype(int) + int(CHANNEL_OFFSET)

    units1["x_um"] = units1["x_um_unshifted"].astype(float) + units1.get("probe_x_shift_um", 0.0)
    units2["x_um"] = units2["x_um_unshifted"].astype(float) + units2.get("probe_x_shift_um", 0.0)
    units1["coordinate_source"] = "chmapfull.mat via unitdata.channelnum"
    units2["coordinate_source"] = "chmapfull.mat via unitdata.channelnum + probe2 x shift"

    counts1 = count_spikes_by_cluster(sc1, units1["original_cluster_id"].astype(int).to_numpy())
    counts2 = count_spikes_by_cluster(sc2, units2["original_cluster_id"].astype(int).to_numpy())

    units1["num_spikes"] = [counts1[int(cid)] for cid in units1["original_cluster_id"]]
    units2["num_spikes"] = [counts2[int(cid)] for cid in units2["original_cluster_id"]]

    # Minimal columns used downstream by FSC main.py:
    #   cluster_id, good/isGood, ch
    # Extra columns help labels/debugging.
    keep_cols = [
        "cluster_id",
        "good",
        "isGood",
        "ch",
        "x_um",
        "y_um",
        "probe_x_shift_um",
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
        "channelnum_interpretation",
    ]

    metrics = pd.concat([units1[keep_cols], units2[keep_cols]], ignore_index=True)
    return metrics


def main():
    ks1 = Path(KS_DIR_1)
    ks2 = Path(KS_DIR_2)
    mat1 = Path(MAT_PATH_1)
    mat2 = Path(MAT_PATH_2)
    chmap_path = Path(CHMAPFULL_PATH)
    out_dir = Path(MERGED_KS_DIR)

    if out_dir.exists():
        if not OVERWRITE_OUTPUT_DIR:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                f"Set OVERWRITE_OUTPUT_DIR=True only if you are okay deleting/recreating that folder."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=False)

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

    ci_ch1 = load_cluster_info_channels(ks1)
    ci_ch2 = load_cluster_info_channels(ks2)
    ch_to_xy, chmapfull_arr = load_chmapfull(chmap_path)

    units1 = load_selected_units_from_mat(mat1, REGION_CONTAINS_1, PROBE1_NAME)
    units2 = load_selected_units_from_mat(mat2, REGION_CONTAINS_2, PROBE2_NAME)

    units1 = attach_cluster_info_channels(units1, ci_ch1, PROBE1_NAME)
    units2 = attach_cluster_info_channels(units2, ci_ch2, PROBE2_NAME)
    units1 = attach_chmapfull_coordinates(units1, ch_to_xy, PROBE1_NAME)
    units2 = attach_chmapfull_coordinates(units2, ch_to_xy, PROBE2_NAME)

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

    desired_sep = PROBE2_CENTER_X_MINUS_PROBE1_CENTER_X_UM
    if desired_sep is None:
        desired_sep = float(
            input(
                "\nEnter desired probe2 center x minus probe1 center x in microns "
                "(e.g. 2000 for 2 mm):\n\n"
            ).strip()
        )
    else:
        desired_sep = float(desired_sep)

    # IMPORTANT: align the full probe geometry, not the selected good-unit clouds.
    # Since both KS dirs use the same chmapfull probe definition, the two full-probe
    # centers are identical before shifting. This means x_shift2 is normally exactly
    # the requested physical probe separation. Keeping this written as center math
    # makes the assumption explicit and easy to extend if future probes use different maps.
    full_probe_center_x = float(np.median(chmapfull_arr[:, 1].astype(float)))
    center1 = full_probe_center_x
    center2 = full_probe_center_x
    x_shift2 = desired_sep - (center2 - center1)
    units1["probe_x_shift_um"] = 0.0
    units2["probe_x_shift_um"] = float(x_shift2)

    full_channel_geometry = build_full_channel_geometry(
        chmapfull_arr=chmapfull_arr,
        probe1_name=PROBE1_NAME,
        probe2_name=PROBE2_NAME,
        probe2_x_shift_um=float(x_shift2),
    )

    # Rebuild channel positions from chmapfull where possible. This is primarily
    # for preview/background plotting; downstream unit distances should use
    # UnitMetrics.csv x_um/y_um.
    cpos1_from_chmap, missing_cpos1 = rebuild_channel_positions_from_chmapfull(cmap1, ch_to_xy, x_shift=0.0)
    cpos2_from_chmap, missing_cpos2 = rebuild_channel_positions_from_chmapfull(cmap2, ch_to_xy, x_shift=x_shift2)
    if missing_cpos1 or missing_cpos2:
        print("WARNING: Could not map some channel_map entries to chmapfull; falling back to KS channel_positions for those rows.")
        if missing_cpos1:
            bad = np.isnan(cpos1_from_chmap[:, 0])
            cpos1_from_chmap[bad, :] = cpos1[bad, :]
        if missing_cpos2:
            bad = np.isnan(cpos2_from_chmap[:, 0])
            cpos2_shifted_fallback = cpos2.copy()
            cpos2_shifted_fallback[:, 0] = cpos2_shifted_fallback[:, 0] + x_shift2
            cpos2_from_chmap[bad, :] = cpos2_shifted_fallback[bad, :]

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

    # Make a minimal cluster_info.tsv as a fallback/debug file.
    cluster_info = pd.DataFrame(
        {
            "cluster_id": unitmetrics["cluster_id"].astype(int),
            "group": "good",
            "ch": unitmetrics["ch"].astype(int),
            "n_spikes": unitmetrics["num_spikes"].astype(int),
            "source_probe": unitmetrics["source_probe"],
            "regname": unitmetrics["regname"],
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
        "mat_path_1": str(mat1),
        "mat_path_2": str(mat2),
        "chmapfull_path": str(chmap_path),
        "merged_ks_dir": str(out_dir),
        "probe1_name": PROBE1_NAME,
        "probe2_name": PROBE2_NAME,
        "region_contains_1": list(REGION_CONTAINS_1),
        "region_contains_2": list(REGION_CONTAINS_2),
        "cluster_offset": int(CLUSTER_OFFSET),
        "channel_offset": int(CHANNEL_OFFSET),
        "probe2_center_x_minus_probe1_center_x_um": float(desired_sep),
        "probe2_x_shift_applied_um": float(x_shift2),
        "probe_shift_source": "full chmapfull geometry center, not selected unit median",
        "full_probe_center_x_um": float(full_probe_center_x),
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
        "unitmetrics_ch_source": "cluster_info.tsv ch",
        "unitmetrics_coordinate_source": "chmapfull.mat via unitdata.channelnum",
    }

    with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nMerged KS directory created:")
    print(f"  {out_dir}")
    print("\nSelected units:")
    print(f"  {PROBE1_NAME}: {len(units1)}")
    print(f"  {PROBE2_NAME}: {len(units2)}")
    print(f"  total: {len(unitmetrics)}")
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
    print("  full_channel_geometry.csv  (all chmapfull sites for both probes)")
    print("  cluster_info.tsv")
    print("  merge_manifest.json")


if __name__ == "__main__":
    main()
