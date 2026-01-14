import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================================================
# ---------------------- CONFIG ------------------------------
# ============================================================

RECORDINGS = {
    "wk1": Path(r"D:\mouse_named_rat\swap_hs_ball_headfixed_12_08_2025_day6_postop_251208_154338"),
    "wk2": Path(r'D:\mouse_named_rat\mouse_named_rat_128ch_commutator_12_16_25_251216_205649'),
    "wk3": Path(r'D:\mouse_named_rat\3wks_post_headfixed_ball_12_23_25_251223_134736_251223_135002\3wks_post_headfixed_ball_12_23_25_251223_134736_251223_135002'),
    "wk4": Path(r'F:\mouse_named_rat_data\mouse_named_rat_wk4_first_5_hrs'),
}

DEFAULT_FS = 20000
XML_PATH = Path(r"d:\flex_probe_layout.xml")

print("SCALE_UV is not automated! You must hand-tune!")
SCALE_UV = 500.0   # µV (PTP) # NOT AUTOMATED!
SCALE_MS = 1.0     # ms

N_ROWS = 20
N_COLS = 5
THRESHOLD_AU = 0.095

CELL_W = 1.0
CELL_H = 1.0
WEEK_GAP = 0.9
OUTER_PAD = 0.6

ROW_SHADE = (0.93, 0.93, 0.93)

PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#3B5B92",
]

# ============================================================
# ---------------- SAMPLING RATE SAFETY -----------------------
# ============================================================

def confirm_sampling_rate(default_fs, label):
    resp = input(f"[{label}] Sampling rate in Hz? [default {default_fs}]: ").strip()
    return float(resp) if resp else default_fs

# ============================================================
# ------------- INTAN XML PARSING -----------------------------
# ============================================================

def load_intan_sites(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return pd.DataFrame([
        {"ch": int(s.attrib["channelNumber"]),
         "x": float(s.attrib["x"]),
         "y": float(s.attrib["y"])}
        for s in root.iter("ElectrodeSite")
    ])

def select_probe1_upper_left_leg(df, n=20):
    leg = df[df["x"] < 0].sort_values("y").iloc[-n:]
    return leg["ch"].tolist()

def build_geometric_neighbors(df, max_dist=80):
    neighbors = {}
    for _, row in df.iterrows():
        d = np.sqrt((df["x"]-row["x"])**2 + (df["y"]-row["y"])**2)
        d[df["ch"] == row["ch"]] = np.inf
        j = np.argmin(d)
        if d.iloc[j] <= max_dist:
            neighbors[int(row["ch"])] = int(df.iloc[j]["ch"])
    return neighbors

# ============================================================
# ----------------- KILOSORT HELPERS --------------------------
# ============================================================

def load_ks_data(ks_dir):
    return {
        "spike_clusters": np.load(ks_dir / "spike_clusters.npy"),
        "spike_templates": np.load(ks_dir / "spike_templates.npy"),
        "templates": np.load(ks_dir / "templates.npy"),
        "metrics": pd.read_csv(ks_dir / "UnitMetrics.csv"),
    }

def cluster_to_template(spike_clusters, spike_templates, good_clusters):
    return {
        c: np.unique(
            spike_templates[spike_clusters == c],
            return_counts=True
        )[0][0]
        for c in good_clusters
    }

# ============================================================
# ------------- BUILD PER-WEEK GRID ---------------------------
# ============================================================

def build_week_grid(ks_dir, row_channels, geom_neighbors):
    ks = load_ks_data(ks_dir)
    good = ks["metrics"].loc[ks["metrics"]["good"], "cluster_id"].values
    cl2t = cluster_to_template(
        ks["spike_clusters"], ks["spike_templates"], good
    )

    grid = [[None]*N_COLS for _ in range(N_ROWS)]
    gid = 0

    for c in good:
        tmpl = ks["templates"][cl2t[c]]
        ptp = tmpl.ptp(axis=0)
        top = np.argmax(ptp)
        if top not in row_channels:
            continue

        r = row_channels.index(top)
        rows = [r]
        chs = [top]

        if top in geom_neighbors:
            nb = geom_neighbors[top]
            if ptp[nb] >= THRESHOLD_AU and nb in row_channels:
                rows.append(row_channels.index(nb))
                chs.append(nb)

        rows, chs = zip(*sorted(zip(rows, chs)))

        for col in range(N_COLS):
            if all(grid[rr][col] is None for rr in rows):
                for rr, ch in zip(rows, chs):
                    grid[rr][col] = (tmpl[:, ch], c, gid)
                gid += 1
                break

    return grid

# ============================================================
# ------------------ FIGURE DRAW ------------------------------
# ============================================================

def draw_figure(grids, fs):
    max_au = max(
        np.max(np.abs(wf))
        for g in grids.values()
        for row in g
        for cell in row if cell
        for wf,_,_ in [cell]
    )

    total_w = OUTER_PAD*2 + len(grids)*(N_COLS*CELL_W + WEEK_GAP)
    total_h = OUTER_PAD*2 + N_ROWS*CELL_H

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    for wi,(wk,grid) in enumerate(grids.items()):
        x0 = OUTER_PAD + wi*(N_COLS*CELL_W + WEEK_GAP)
        groups = defaultdict(list)

        for r in range(N_ROWS):
            y = OUTER_PAD + r*CELL_H
            if r % 2:
                ax.add_patch(plt.Rectangle(
                    (x0, y), N_COLS*CELL_W, CELL_H,
                    color=ROW_SHADE, zorder=0
                ))

            for c in range(N_COLS):
                cell = grid[r][c]
                if not cell:
                    continue

                ### CHANGED: crop to inner 40% ###
                wf_full, cl, gid = cell
                n = len(wf_full)
                i0 = int(0.3 * n)
                i1 = int(0.7 * n)
                wf = wf_full[i0:i1]

                groups[gid].append((r, c))

                xs = np.linspace(
                    x0 + c*CELL_W + 0.2,
                    x0 + (c+1)*CELL_W - 0.2,
                    len(wf)
                )
                ys = y + CELL_H/2 + (wf/max_au)*(CELL_H*0.35)
                ax.plot(xs, ys, color=PALETTE[cl % len(PALETTE)], lw=1.5)

        for gid, cells in groups.items():
            rs = [r for r,_ in cells]
            c = cells[0][1]
            y0 = OUTER_PAD + min(rs)*CELL_H
            h = (max(rs) - min(rs) + 1) * CELL_H
            ax.add_patch(plt.Rectangle(
                (x0 + c*CELL_W + 0.1, y0 + 0.15),
                CELL_W - 0.2, h - 0.3,
                fill=False, ls="--", lw=1.2
            ))

        ax.text(
            x0 + N_COLS*CELL_W/2,
            OUTER_PAD + N_ROWS*CELL_H + 0.2,
            wk, ha="center", va="bottom", weight="bold"
        )

    # ---------------- SCALE BAR ----------------
    bar_x = OUTER_PAD - 0.35
    bar_y = OUTER_PAD - 0.25

    # vertical scale bar = largest PTP waveform shown
    uv_h = 2 * (CELL_H * 0.35)
    ax.plot([bar_x, bar_x], [bar_y, bar_y + uv_h], color="black", lw=2)
    ax.text(
        bar_x - 0.05,
        bar_y + uv_h / 2,
        f"{int(SCALE_UV)} µV",
        ha="right",
        va="center",
        fontsize=10
    )

    ### CHANGED: scale bar uses cropped waveform length ###
    tmpl_len = next(
        int(0.4 * len(cell[0])) for g in grids.values()
        for row in g for cell in row if cell
    )

    ms_per_sample = 1000 / fs
    samples_for_ms = int(SCALE_MS / ms_per_sample)
    frac = samples_for_ms / tmpl_len
    ms_w = frac * (CELL_W - 0.4)

    ax.plot([bar_x, bar_x + ms_w], [bar_y, bar_y], color="black", lw=2)
    ax.text(bar_x + ms_w/2, bar_y - 0.08,
            f"{SCALE_MS} ms", ha="center", va="top")

    plt.show()

# ============================================================
# ------------------------- RUN -------------------------------
# ============================================================

sites = load_intan_sites(XML_PATH)
ROW_CHANNELS = select_probe1_upper_left_leg(sites, N_ROWS)
GEOM_NEIGHBORS = build_geometric_neighbors(sites)

grids = {}
for wk, ks in RECORDINGS.items():
    fs = confirm_sampling_rate(DEFAULT_FS, wk)
    grids[wk] = build_week_grid(ks, ROW_CHANNELS, GEOM_NEIGHBORS)

draw_figure(grids, fs)
