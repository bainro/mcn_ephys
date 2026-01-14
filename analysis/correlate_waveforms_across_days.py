# ==============================
# Cross-day Kilosort template matching
# Day 6 vs Day 14
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------
# Load Kilosort output
# ------------------------------
def load_ks(folder):
    templates = np.load(f"{folder}/templates.npy")
    templates_ind = np.load(f"{folder}/templates_ind.npy")
    info = pd.read_csv(f"{folder}/cluster_info.tsv", sep="\t")
    groups = pd.read_csv(f"{folder}/cluster_group.tsv", sep="\t")

    good_ids = groups[groups.group == "good"].cluster_id.values
    info = info.set_index("cluster_id")

    return templates, templates_ind, info, good_ids


# ------------------------------
# Extract top-N channels by amplitude
# ------------------------------
def extract_top_channels(template, inds, n_ch=6):
    """
    template: (time, local_channels)
    inds: (local_channels,)
    """
    amp = np.max(np.abs(template), axis=0)
    top_idx = np.argsort(amp)[-n_ch:]
    return template[:, top_idx], inds[top_idx]


# ------------------------------
# Normalize waveform (shape only)
# ------------------------------
def normalize(w):
    return w / np.linalg.norm(w)


# ------------------------------
# Correlate two templates on overlapping channels
# ------------------------------
def template_corr(tA, indA, tB, indB):
    common = np.intersect1d(indA, indB)
    if len(common) < 2:
        return np.nan

    idxA = [np.where(indA == c)[0][0] for c in common]
    idxB = [np.where(indB == c)[0][0] for c in common]

    wA = normalize(tA[:, idxA].flatten())
    wB = normalize(tB[:, idxB].flatten())

    return pearsonr(wA, wB)[0]


# ------------------------------
# Load both days
# ------------------------------
day6 = r'C:\Users\Priyansha\Desktop\mouse_named_rat\swap_hs_ball_headfixed_12_08_2025_day6_postop_251208_154338'
day14 = r'C:\Users\Priyansha\Desktop\mouse_named_rat\mouse_named_rat_128ch_commutator_12_16_25_251216_205649'
day21 = r'D:\3wks_post_headfixed_ball_12_23_25_251223_134736_251223_135002\3wks_post_headfixed_ball_12_23_25_251223_134736_251223_135002'
templates_A, inds_A, info_A, good_A = load_ks(day6)
templates_B, inds_B, info_B, good_B = load_ks(day14)

def filter_valid_clusters(cluster_ids, templates):
    return np.array([k for k in cluster_ids if k < templates.shape[0]])

good_A = filter_valid_clusters(good_A, templates_A)
good_B = filter_valid_clusters(good_B, templates_B)

# ------------------------------
# Matching parameters
# ------------------------------
CHANNEL_TOL = 2   # ± channels for candidate matching
TOP_CH = 5        # number of channels per template


# ------------------------------
# Match templates across days
# ------------------------------
matches = []

for kA in good_A:
    chA = info_A.loc[kA, "ch"]

    candidates = [
        kB for kB in good_B
        if abs(info_B.loc[kB, "ch"] - chA) <= CHANNEL_TOL
    ]

    if not candidates:
        continue

    tA, indA = extract_top_channels(templates_A[kA], inds_A[kA], TOP_CH)

    best_corr = -np.inf
    best_kB = None

    for kB in candidates:
        tB, indB = extract_top_channels(templates_B[kB], inds_B[kB], TOP_CH)
        c = template_corr(tA, indA, tB, indB)
        if np.isnan(c):
            continue
        if c > best_corr:
            best_corr = c
            best_kB = kB

    matches.append((kA, best_kB, best_corr))


# ------------------------------
# Results dataframe
# ------------------------------
df = pd.DataFrame(matches, columns=["day6_cluster", "day14_cluster", "corr"])

c_hit = 0.3
c_near = c_hit - 0.2
hits = df[df["corr"] > c_hit]
near = df[(df["corr"] > c_near) & (df["corr"] <= c_hit)]
miss = df[df["corr"] <= c_near]

print(f"Hits (>{c_hit}): {len(hits)}")
print(f"Near hits ({c_near}–{c_hit}): {len(near)}")
print(f"Misses (<={c_near}): {len(miss)}")


# ------------------------------
# Visualization helper
# ------------------------------
def plot_match(kA, kB, n_ch=6):
    kA = int(kA)
    kB = int(kB)

    tA, indA = extract_top_channels(templates_A[kA], inds_A[kA], n_ch)
    tB, indB = extract_top_channels(templates_B[kB], inds_B[kB], n_ch)

    common = np.intersect1d(indA, indB)
    if len(common) == 0:
        print("No overlapping channels")
        return

    n = len(common)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(6, 1.8 * n),
        sharex=True
    )

    if n == 1:
        axes = [axes]

    for ax, ch in zip(axes, common):
        iA = np.where(indA == ch)[0][0]
        iB = np.where(indB == ch)[0][0]

        ax.plot(tA[:, iA], color="blue", alpha=0.8, label="Day 6")
        ax.plot(tB[:, iB], color="red", alpha=0.8, label="Day 14")

        ax.set_ylabel(f"Ch {ch}")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (samples)")

    fig.suptitle(f"Cluster match: day6 {kA} vs day14 {kB}", y=0.98)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Example visual inspection
# ------------------------------
for _, row in hits.sample(min(100, len(hits))).iterrows():
    plot_match(row.day6_cluster, row.day14_cluster)
