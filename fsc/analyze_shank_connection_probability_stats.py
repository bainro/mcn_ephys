#!/usr/bin/env python3
"""
analyze_shank_connection_probability_stats.py

Standalone statistics/plotting script for FSC shank/region connection probability CSVs.

Give this script a list of per-animal CSVs like:
    <animal_id>_shank_connection_percentages.csv

Expected input columns, from fsc/main.py:
    comparison
    n_prepost_possible_directed
    n_detected_connections

Optional input column:
    percent_possible_connections

Outputs:
    aggregate_all_input_rows.csv
    aggregate_category_rates_by_animal.csv
    aggregate_category_pooled_weighted_summary.csv
    aggregate_pairwise_category_stats.csv
    aggregate_category_boxplot_animal_balanced.png
    aggregate_category_pooled_weighted_barplot.png

Statistical philosophy:
-----------------------
This script intentionally reports BOTH:

1. Animal-balanced rates/tests:
   One category rate per animal is computed as:
       detected category connections in animal / possible category connections in animal.
   Pairwise Wilcoxon signed-rank tests are run across paired animals. This treats animal
   as the biological replicate and avoids pseudoreplication.

2. Opportunity-weighted / pooled rates/tests:
   For each category, detected and possible directed connections are summed across animals:
       pooled rate = sum(detected) / sum(possible).
   Pairwise z-tests and Cochran-Mantel-Haenszel tests use the count denominators. These
   estimate the expected connection probability per possible directed pair, but they are
   more statistically powerful and can be anti-conservative if neuron-pair opportunities
   are treated as fully independent.

Recommended interpretation:
---------------------------
Use the pooled weighted rate as the intuitive expectation / effect-size estimate, and use
animal-level summaries/tests to show the effect is reproducible across animals. The CMH
stratified count test is a useful middle ground because it weights by opportunities while
stratifying by animal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =========================
# USER CONFIG
# =========================

CSV_PATHS = [
    r"E:\\rob\\merged_TR8_S1_V1\\merged_TR8_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR14_S1_V1\\merged_TR14_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR15_S1_V1\\merged_TR15_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR17_S1_V1\\merged_TR17_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR18_S1_V1\\merged_TR18_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR19_S1_V1\\merged_TR19_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR21_S1_V1\\merged_TR21_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR28_S1_V1\\merged_TR28_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR30_S1_V1\\merged_TR30_S1_V1_shank_connection_percentages.csv",
    r"E:\\rob\\merged_TR31_S1_V1\\merged_TR31_S1_V1_shank_connection_percentages.csv"
]

OUTPUT_DIR = Path(r"E:\\rob\\aggregate_shank_stats")
OUTPUT_PREFIX = "aggregate"

# If True, use the filename stem as animal_id when there is no animal_id column.
# The script strips common suffixes like "_shank_connection_percentages".
INFER_ANIMAL_ID_FROM_FILENAME = True

# Optional manual mapping from region/shank labels to broader probe/region groups.
# This is useful if names do not share clean prefixes.
# Example:
# REGION_TO_PROBE = {
#     "S1 Sh1": "S1",
#     "S1 Sh2": "S1",
#     "V1 Sh1": "V1",
#     "V1 Sh2": "V1",
# }
REGION_TO_PROBE = {}

# If REGION_TO_PROBE is empty or missing a label, infer probe/region group from the
# label prefix before one of these separators. This makes labels like S1_Sh1, S1-Sh1,
# S1 Sh1, probe1_s1, probe1_s2 work automatically.
PREFIX_SEPARATORS = ("_", "-", ":", "|", " ")

# If True, also save per-original-comparison pooled summaries, not just category summaries.
SAVE_COMPARISON_LEVEL_SUMMARY = True

# Plot settings. Uses matplotlib defaults; no seaborn/styles/colors.
SAVE_PLOTS = True
FIG_DPI = 250


# =========================
# Label/category utilities
# =========================

def infer_animal_id_from_path(path: Path) -> str:
    stem = path.stem
    suffixes = [
        "_shank_connection_percentages",
        "_connection_percentages",
        "-shank_connection_percentages",
        "-connection_percentages",
    ]
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def split_comparison(comparison: str) -> tuple[str, str]:
    comparison = str(comparison).strip()
    if "<->" in comparison:
        a, b = comparison.split("<->", 1)
        return a.strip(), b.strip()
    return comparison, comparison


def label_to_probe(label: str) -> str:
    label = str(label).strip()
    if label in REGION_TO_PROBE:
        return str(REGION_TO_PROBE[label])

    for sep in PREFIX_SEPARATORS:
        if sep in label:
            prefix = label.split(sep, 1)[0].strip()
            if prefix:
                return prefix

    # Fallback: remove common shank suffix patterns, e.g. S1Sh1 -> S1, probe1shank2 -> probe1.
    m = re.match(r"^(.*?)(?:shank|sh|s)\s*\d+$", label, flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip(" _-:|")

    return ""


def classify_comparison(comparison: str) -> str:
    a, b = split_comparison(comparison)
    if a == b:
        return "within_region"

    pa = label_to_probe(a)
    pb = label_to_probe(b)
    if pa and pb:
        if pa == pb:
            return "same_probe_between_regions"
        return "different_probe_between_regions"

    return "between_regions"


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([str(v) for v in values]))


# =========================
# Stats helpers
# =========================

def wilson_ci(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns proportions, not percent."""
    if n <= 0:
        return (np.nan, np.nan)
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test. Returns z, p."""
    if n1 <= 0 or n2 <= 0:
        return np.nan, np.nan
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (x1 / n1 - x2 / n2) / se
    try:
        from scipy.stats import norm
        p = 2.0 * norm.sf(abs(z))
    except Exception:
        # Normal survival fallback via erfc.
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)


def cochran_mantel_haenszel_test(paired_counts: pd.DataFrame, cat_a: str, cat_b: str) -> tuple[float, float, float]:
    """
    CMH test for association in 2x2 tables stratified by animal.

    Each animal supplies:
        category A: detected_a, non_detected_a
        category B: detected_b, non_detected_b

    Returns chi2, p, common_odds_ratio_mh.
    """
    rows = []
    for animal_id, g in paired_counts.groupby("animal_id", sort=False):
        ga = g[g["comparison_category"] == cat_a]
        gb = g[g["comparison_category"] == cat_b]
        if ga.empty or gb.empty:
            continue
        a = int(ga["n_detected_connections"].iloc[0])
        n_a = int(ga["n_prepost_possible_directed"].iloc[0])
        c = int(gb["n_detected_connections"].iloc[0])
        n_b = int(gb["n_prepost_possible_directed"].iloc[0])
        b = n_a - a
        d = n_b - c
        if n_a <= 0 or n_b <= 0:
            continue
        rows.append((a, b, c, d))

    if not rows:
        return np.nan, np.nan, np.nan

    num = 0.0
    den = 0.0
    or_num = 0.0
    or_den = 0.0

    for a, b, c, d in rows:
        n1 = a + b
        n2 = c + d
        m1 = a + c
        m2 = b + d
        n = n1 + n2
        if n <= 1:
            continue
        expected_a = n1 * m1 / n
        var_a = n1 * n2 * m1 * m2 / (n * n * (n - 1))
        num += a - expected_a
        den += var_a

        # Mantel-Haenszel common OR components.
        or_num += a * d / n
        or_den += b * c / n

    if den <= 0:
        chi2_stat = np.nan
        p = np.nan
    else:
        chi2_stat = (num * num) / den
        try:
            from scipy.stats import chi2
            p = chi2.sf(chi2_stat, df=1)
        except Exception:
            # chi-square df=1 survival equals erfc(sqrt(x/2))
            p = math.erfc(math.sqrt(chi2_stat / 2.0))

    common_or = (or_num / or_den) if or_den > 0 else np.nan
    return float(chi2_stat), float(p), float(common_or)


def bonferroni(p: float, n_tests: int) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(min(1.0, p * max(1, n_tests)))


# =========================
# Load/aggregate
# =========================

def load_all_rows(csv_paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        needed = {"comparison", "n_prepost_possible_directed", "n_detected_connections"}
        missing = sorted(needed - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df = df.copy()
        if "animal_id" not in df.columns:
            if not INFER_ANIMAL_ID_FROM_FILENAME:
                raise ValueError(f"{path} has no animal_id column and filename inference is disabled.")
            df["animal_id"] = infer_animal_id_from_path(path)

        df["source_csv"] = str(path)
        df["comparison"] = df["comparison"].astype(str)
        df["comparison_a"] = df["comparison"].map(lambda x: split_comparison(x)[0])
        df["comparison_b"] = df["comparison"].map(lambda x: split_comparison(x)[1])
        df["probe_group_a"] = df["comparison_a"].map(label_to_probe)
        df["probe_group_b"] = df["comparison_b"].map(label_to_probe)
        df["comparison_category"] = df["comparison"].map(classify_comparison)
        df["n_prepost_possible_directed"] = df["n_prepost_possible_directed"].astype(int)
        df["n_detected_connections"] = df["n_detected_connections"].astype(int)
        df["percent_possible_connections"] = np.where(
            df["n_prepost_possible_directed"] > 0,
            100.0 * df["n_detected_connections"] / df["n_prepost_possible_directed"],
            np.nan,
        )
        rows.append(df)

    if not rows:
        raise ValueError("CSV_PATHS is empty. Add at least one per-animal shank_connection_percentages.csv path.")

    return pd.concat(rows, ignore_index=True)


def category_rates_by_animal(all_rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (animal_id, category), g in all_rows.groupby(["animal_id", "comparison_category"], sort=False):
        possible = int(g["n_prepost_possible_directed"].sum())
        detected = int(g["n_detected_connections"].sum())
        pct = 100.0 * detected / possible if possible else np.nan
        lo, hi = wilson_ci(detected, possible)
        out_rows.append(
            {
                "animal_id": animal_id,
                "comparison_category": category,
                "n_prepost_possible_directed": possible,
                "n_detected_connections": detected,
                "percent_possible_connections": pct,
                "wilson_ci_low_percent": 100.0 * lo if np.isfinite(lo) else np.nan,
                "wilson_ci_high_percent": 100.0 * hi if np.isfinite(hi) else np.nan,
                "n_component_comparisons": int(len(g)),
                "component_comparisons": ";".join(g["comparison"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(out_rows)


def pooled_weighted_summary(category_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    categories = ordered_unique(category_df["comparison_category"].dropna())
    for category in categories:
        g = category_df[category_df["comparison_category"] == category]
        possible = int(g["n_prepost_possible_directed"].sum())
        detected = int(g["n_detected_connections"].sum())
        pooled_pct = 100.0 * detected / possible if possible else np.nan
        lo, hi = wilson_ci(detected, possible)
        animal_rates = g["percent_possible_connections"].dropna().to_numpy(dtype=float)
        out_rows.append(
            {
                "comparison_category": category,
                "total_detected_connections": detected,
                "total_possible_directed": possible,
                "pooled_weighted_percent": pooled_pct,
                "pooled_wilson_ci_low_percent": 100.0 * lo if np.isfinite(lo) else np.nan,
                "pooled_wilson_ci_high_percent": 100.0 * hi if np.isfinite(hi) else np.nan,
                "n_animals": int(g["animal_id"].nunique()),
                "animal_balanced_mean_percent": float(np.mean(animal_rates)) if animal_rates.size else np.nan,
                "animal_balanced_median_percent": float(np.median(animal_rates)) if animal_rates.size else np.nan,
                "animal_balanced_sd_percent": float(np.std(animal_rates, ddof=1)) if animal_rates.size >= 2 else np.nan,
                "animal_balanced_sem_percent": float(np.std(animal_rates, ddof=1) / math.sqrt(animal_rates.size)) if animal_rates.size >= 2 else np.nan,
                "min_animal_percent": float(np.min(animal_rates)) if animal_rates.size else np.nan,
                "max_animal_percent": float(np.max(animal_rates)) if animal_rates.size else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def comparison_level_pooled_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for comparison, g in all_rows.groupby("comparison", sort=False):
        possible = int(g["n_prepost_possible_directed"].sum())
        detected = int(g["n_detected_connections"].sum())
        lo, hi = wilson_ci(detected, possible)
        out_rows.append(
            {
                "comparison": comparison,
                "comparison_category": classify_comparison(comparison),
                "total_detected_connections": detected,
                "total_possible_directed": possible,
                "pooled_weighted_percent": 100.0 * detected / possible if possible else np.nan,
                "pooled_wilson_ci_low_percent": 100.0 * lo if np.isfinite(lo) else np.nan,
                "pooled_wilson_ci_high_percent": 100.0 * hi if np.isfinite(hi) else np.nan,
                "n_animals": int(g["animal_id"].nunique()),
            }
        )
    return pd.DataFrame(out_rows)


def pairwise_category_stats(category_df: pd.DataFrame) -> pd.DataFrame:
    categories = ordered_unique(category_df["comparison_category"].dropna())
    n_tests = len(categories) * (len(categories) - 1) // 2
    rows = []

    for i, cat_a in enumerate(categories):
        for cat_b in categories[i + 1:]:
            a = category_df[category_df["comparison_category"] == cat_a][
                ["animal_id", "percent_possible_connections", "n_detected_connections", "n_prepost_possible_directed"]
            ].rename(
                columns={
                    "percent_possible_connections": "percent_a",
                    "n_detected_connections": "detected_a",
                    "n_prepost_possible_directed": "possible_a",
                }
            )
            b = category_df[category_df["comparison_category"] == cat_b][
                ["animal_id", "percent_possible_connections", "n_detected_connections", "n_prepost_possible_directed"]
            ].rename(
                columns={
                    "percent_possible_connections": "percent_b",
                    "n_detected_connections": "detected_b",
                    "n_prepost_possible_directed": "possible_b",
                }
            )
            paired = a.merge(b, on="animal_id", how="inner")
            n_paired = int(len(paired))

            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
            if n_paired >= 2:
                try:
                    from scipy.stats import wilcoxon
                    wilcoxon_stat, wilcoxon_p = wilcoxon(
                        paired["percent_a"],
                        paired["percent_b"],
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                    wilcoxon_stat = float(wilcoxon_stat)
                    wilcoxon_p = float(wilcoxon_p)
                except Exception:
                    pass

            det_a_total = int(paired["detected_a"].sum()) if n_paired else 0
            poss_a_total = int(paired["possible_a"].sum()) if n_paired else 0
            det_b_total = int(paired["detected_b"].sum()) if n_paired else 0
            poss_b_total = int(paired["possible_b"].sum()) if n_paired else 0
            pooled_pct_a = 100.0 * det_a_total / poss_a_total if poss_a_total else np.nan
            pooled_pct_b = 100.0 * det_b_total / poss_b_total if poss_b_total else np.nan

            z_stat, z_p = two_proportion_z_test(det_a_total, poss_a_total, det_b_total, poss_b_total)
            cmh_chi2, cmh_p, cmh_or = cochran_mantel_haenszel_test(category_df, cat_a, cat_b)

            rows.append(
                {
                    "category_a": cat_a,
                    "category_b": cat_b,
                    "paired_n_animals": n_paired,
                    "animal_balanced_mean_percent_a": float(paired["percent_a"].mean()) if n_paired else np.nan,
                    "animal_balanced_mean_percent_b": float(paired["percent_b"].mean()) if n_paired else np.nan,
                    "animal_balanced_mean_percent_a_minus_b": float((paired["percent_a"] - paired["percent_b"]).mean()) if n_paired else np.nan,
                    "animal_balanced_median_percent_a": float(paired["percent_a"].median()) if n_paired else np.nan,
                    "animal_balanced_median_percent_b": float(paired["percent_b"].median()) if n_paired else np.nan,
                    "wilcoxon_test": "paired Wilcoxon signed-rank on per-animal category percentages",
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_value": wilcoxon_p,
                    "wilcoxon_p_value_bonferroni": bonferroni(wilcoxon_p, n_tests),
                    "pooled_detected_a": det_a_total,
                    "pooled_possible_a": poss_a_total,
                    "pooled_weighted_percent_a": pooled_pct_a,
                    "pooled_detected_b": det_b_total,
                    "pooled_possible_b": poss_b_total,
                    "pooled_weighted_percent_b": pooled_pct_b,
                    "pooled_weighted_percent_a_minus_b": pooled_pct_a - pooled_pct_b if np.isfinite(pooled_pct_a) and np.isfinite(pooled_pct_b) else np.nan,
                    "two_proportion_z_test_note": "count-weighted pooled test; can be anti-conservative if neuron-pair opportunities are not independent",
                    "z_statistic": z_stat,
                    "z_p_value": z_p,
                    "z_p_value_bonferroni": bonferroni(z_p, n_tests),
                    "cmh_test_note": "Cochran-Mantel-Haenszel count test stratified by animal; still assumes independent opportunities within strata",
                    "cmh_chi2": cmh_chi2,
                    "cmh_common_odds_ratio": cmh_or,
                    "cmh_p_value": cmh_p,
                    "cmh_p_value_bonferroni": bonferroni(cmh_p, n_tests),
                }
            )

    return pd.DataFrame(rows)


# =========================
# Plotting
# =========================

def save_plots(category_df: pd.DataFrame, pooled_df: pd.DataFrame, out_dir: Path, prefix: str):
    if not SAVE_PLOTS:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ordered_unique(category_df["comparison_category"].dropna())
    data = [
        category_df.loc[category_df["comparison_category"] == c, "percent_possible_connections"].dropna().to_numpy(dtype=float)
        for c in categories
    ]

    if categories:
        fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(categories)), 5.5))
        ax.boxplot(data, labels=categories, showfliers=False)
        for idx, vals in enumerate(data, start=1):
            if len(vals):
                x = np.full(len(vals), idx, dtype=float)
                offsets = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(x + offsets, vals, s=30, alpha=0.8)
        ax.set_ylabel("Connection probability (% of possible directed pairs)")
        ax.set_title("Animal-balanced FSC connection rates by comparison category")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_category_boxplot_animal_balanced.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    if not pooled_df.empty:
        pooled_plot = pooled_df.copy()
        xlabels = pooled_plot["comparison_category"].astype(str).tolist()
        y = pooled_plot["pooled_weighted_percent"].to_numpy(dtype=float)
        yerr_low = y - pooled_plot["pooled_wilson_ci_low_percent"].to_numpy(dtype=float)
        yerr_high = pooled_plot["pooled_wilson_ci_high_percent"].to_numpy(dtype=float) - y
        yerr = np.vstack([yerr_low, yerr_high])

        fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(xlabels)), 5.5))
        ax.bar(np.arange(len(xlabels)), y)
        ax.errorbar(np.arange(len(xlabels)), y, yerr=yerr, fmt="none", capsize=4)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=30, ha="right")
        ax.set_ylabel("Pooled weighted connection probability (%)")
        ax.set_title("Opportunity-weighted pooled FSC connection rates")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_category_pooled_weighted_barplot.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


# =========================
# Main
# =========================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_rows(CSV_PATHS)
    category_df = category_rates_by_animal(all_rows)
    pooled_df = pooled_weighted_summary(category_df)
    stats_df = pairwise_category_stats(category_df)

    all_rows_path = out_dir / f"{OUTPUT_PREFIX}_all_input_rows.csv"
    category_path = out_dir / f"{OUTPUT_PREFIX}_category_rates_by_animal.csv"
    pooled_path = out_dir / f"{OUTPUT_PREFIX}_category_pooled_weighted_summary.csv"
    stats_path = out_dir / f"{OUTPUT_PREFIX}_pairwise_category_stats.csv"

    all_rows.to_csv(all_rows_path, index=False)
    category_df.to_csv(category_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    if SAVE_COMPARISON_LEVEL_SUMMARY:
        comp_df = comparison_level_pooled_summary(all_rows)
        comp_path = out_dir / f"{OUTPUT_PREFIX}_comparison_level_pooled_summary.csv"
        comp_df.to_csv(comp_path, index=False)
    else:
        comp_path = None

    save_plots(category_df, pooled_df, out_dir, OUTPUT_PREFIX)

    print("\nDone. Wrote:")
    print(f"  {all_rows_path}")
    print(f"  {category_path}")
    print(f"  {pooled_path}")
    print(f"  {stats_path}")
    if comp_path is not None:
        print(f"  {comp_path}")
    if SAVE_PLOTS:
        print(f"  {out_dir / f'{OUTPUT_PREFIX}_category_boxplot_animal_balanced.png'}")
        print(f"  {out_dir / f'{OUTPUT_PREFIX}_category_pooled_weighted_barplot.png'}")

    print("\nPooled weighted summary:")
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(pooled_df)

    if not stats_df.empty:
        print("\nPairwise category stats:")
        show_cols = [
            "category_a", "category_b", "paired_n_animals",
            "pooled_weighted_percent_a", "pooled_weighted_percent_b", "pooled_weighted_percent_a_minus_b",
            "wilcoxon_p_value_bonferroni", "cmh_p_value_bonferroni", "z_p_value_bonferroni",
        ]
        with pd.option_context("display.max_columns", 20, "display.width", 180):
            print(stats_df[show_cols])


if __name__ == "__main__":
    main()
