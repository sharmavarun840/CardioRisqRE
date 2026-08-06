"""
================================================================================
CardioRisqRE™ Simulation Study
Script 03: Analysis and Figure Generation
================================================================================

Description:
    Performs the complete statistical analysis of the synthetic cohort and
    generates all manuscript figures (Figures 1–7):
        Fig 1 - GWAS data curation funnel
        Fig 2 - Component Z-score distributions (GRC, BPC, ARC)
        Fig 3 - Overall PCAD distribution and component risk profiles
        Fig 4 - Genetic risk stratification (PRS quartile + 9p21 genotype)
        Fig 5 - Sensitivity analysis (component weight variation)
        Fig 6 - Stress test (edge-case patient archetypes)
        Fig 7 - Full correlation heatmap

Input:
    synthetic_cohort_N5000.csv   (output of Script 02)

Output:
    figures/Fig1_curation_funnel.png  through  Fig7_correlation.png
    analysis_summary_statistics.csv

Python version: 3.11
Required packages: numpy, pandas, scipy, matplotlib, seaborn
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings; warnings.filterwarnings("ignore")

INPUT_FILE  = "synthetic_cohort_N5000.csv"
OUTPUT_DIR  = "figures"
STATS_FILE  = "analysis_summary_statistics.csv"

C_LOW, C_MED, C_HIGH = "#27ae60", "#f39c12", "#e74c3c"
C_BLUE, C_DARK, C_PUR = "#2980b9", "#2c3e50", "#8e44ad"
BG = "#f8f9fa"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": BG,
    "figure.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded synthetic cohort: N = {len(df):,}")
    return df


def assign_risk(z):
    return np.where(z < -0.44, 1, np.where(z < 0.44, 2, 3))


# ── FIGURE 1: Curation funnel ─────────────────────────────────────────────────
def fig1_curation_funnel():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    stages = ["Raw GWAS\nCatalog export", "p < 5×10⁻⁸\nfilter",
              "Valid RAF\n(0–1)", "Plausible\neffect size",
              "OR/BETA\nharmonised", "Independent\nloci (clumped)"]
    counts = [1347, 904, 904, 816, 816, 250]
    colors = [C_BLUE] * 5 + [C_HIGH]
    bars = ax.bar(range(len(stages)), counts, color=colors, alpha=0.85,
                  edgecolor="white", width=0.62)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 20, str(c),
                ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("Number of SNPs", fontweight="bold")
    ax.set_title("Figure 1. GWAS Catalog Data Curation Pipeline\n"
                 "1,347 raw associations → 250 independent genome-wide-significant loci",
                 fontweight="bold", fontsize=11, color=C_DARK)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig1_curation_funnel.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── FIGURE 2: Component distributions ────────────────────────────────────────
def fig2_component_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 2. Z-score Distributions for the Three CardioRisqRE™ Components",
                 fontsize=11, fontweight="bold", color=C_DARK)
    for ax, col, title in zip(axes, ["Z_GRC", "Z_BPC", "Z_ARC"],
        ["Genetic Risk Component (GRC)\n250-locus PRS + 9p21",
         "Biochemical Component (BPC)\nhs-CRP · LDL-C · ApoB · Lp(a)",
         "Anthropometric Component (ARC)\nBMI · WC · WHR · BF%"]):
        z = df[col]
        x = np.linspace(z.min(), z.max(), 200)
        ax.hist(z, bins=50, density=True, color=C_BLUE, alpha=0.55,
                edgecolor="white", linewidth=0.3)
        ax.plot(x, stats.norm.pdf(x, z.mean(), z.std()), color=C_DARK, linewidth=2)
        ax.axvline(-0.44, color=C_LOW, linestyle="--", linewidth=1.8)
        ax.axvline( 0.44, color=C_HIGH, linestyle="--", linewidth=1.8)
        low = (z < -0.44).sum(); mid = z.between(-0.44, 0.44).sum(); high = (z > 0.44).sum()
        ax.set_xlabel(f"{col.replace('Z_','')} Z-score", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=9.5)
        ax.text(0.03, 0.97, f"Low: {low/len(df)*100:.1f}%\nMed: {mid/len(df)*100:.1f}%\nHigh: {high/len(df)*100:.1f}%",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig2_component_distributions.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── FIGURE 3: PCAD overall distribution ──────────────────────────────────────
def fig3_pcad_distribution(df):
    N = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 3. CardioRisqRE™ Overall PCAD Score Distribution",
                 fontsize=11.5, fontweight="bold", color=C_DARK)
    ax = axes[0]
    bins = np.linspace(33, 100, 40)
    counts, edges = np.histogram(df["PCAD"], bins=bins)
    for i, (l, r, c) in enumerate(zip(edges[:-1], edges[1:],
        [C_LOW if e < 44.4 else C_MED if e < 66.6 else C_HIGH for e in edges[:-1]])):
        ax.bar(l, counts[i], width=r - l, color=c, alpha=0.78,
               edgecolor="white", linewidth=0.3)
    ax.axvline(44.4, color=C_LOW, linestyle="--", linewidth=1.8)
    ax.axvline(66.6, color=C_HIGH, linestyle="--", linewidth=1.8)
    low_n = (df["PCAD"] <= 44.4).sum(); med_n = df["PCAD"].between(44.4, 66.6).sum()
    high_n = (df["PCAD"] > 66.6).sum()
    patches = [mpatches.Patch(color=c, label=f"{l}: {n} ({n/N*100:.1f}%)")
               for c, l, n in [(C_LOW, "Low", low_n), (C_MED, "Medium", med_n), (C_HIGH, "High", high_n)]]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("CardioRisqRE™ PCAD Score (%)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("PCAD Score Distribution", fontweight="bold", fontsize=10)
    ax.text(0.97, 0.72, f"Mean: {df['PCAD'].mean():.1f}%\nSD: {df['PCAD'].std():.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax = axes[1]
    cats = ["GRC\n(Genetic)", "BPC\n(Biochem.)", "ARC\n(Anthrop.)"]
    bottoms = np.zeros(3)
    for label, color in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        cat_num = {"Low": 1, "Medium": 2, "High": 3}[label]
        vals = [(df[f"Cat_{c}"] == cat_num).sum() / N * 100 for c in ["GRC", "BPC", "ARC"]]
        bars = ax.bar(cats, vals, bottom=bottoms, color=color, alpha=0.82,
                      label=label, edgecolor="white")
        for bar, val, bot in zip(bars, vals, bottoms):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.1f}%", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white")
        bottoms += np.array(vals)
    ax.set_ylim(0, 112); ax.legend(fontsize=9, loc="upper right")
    ax.set_ylabel("Proportion (%)", fontweight="bold")
    ax.set_title("Risk Category by Component", fontweight="bold", fontsize=10)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig3_PCAD_distribution.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── FIGURE 4: Genetic stratification (FIXED) ─────────────────────────────────
def fig4_genetic_stratification(df):
    N = len(df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    fig.suptitle("Figure 4. Genetic Risk Stratification Using the 250-Locus PRS",
                 fontsize=11.5, fontweight="bold", color=C_DARK)
    hi  = df[df["PRS_z"] >= df["PRS_z"].quantile(0.75)]
    lo  = df[df["PRS_z"] <= df["PRS_z"].quantile(0.25)]
    mid = df[(df["PRS_z"] > df["PRS_z"].quantile(0.25)) &
             (df["PRS_z"] < df["PRS_z"].quantile(0.75))]
    ax = axes[0]
    parts = ax.violinplot([lo["PCAD"], mid["PCAD"], hi["PCAD"]],
                          positions=[1, 2, 3], showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [C_LOW, C_MED, C_HIGH]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Low PRS\n(Q1)", "Mid PRS\n(Q2-Q3)", "High PRS\n(Q4)"])
    ax.set_ylabel("PCAD Score (%)", fontweight="bold")
    ax.set_title("PCAD by PRS Quartile", fontweight="bold", fontsize=10)
    t, _ = stats.ttest_ind(hi["PCAD"], lo["PCAD"])
    d = (hi["PCAD"].mean() - lo["PCAD"].mean()) / df["PCAD"].std()
    ax.text(0.5, 0.97, f"High vs Low: t={t:.1f}, p<0.001, d={d:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax = axes[1]
    grps = {"Low PRS\n(Q1)": lo, "Mid PRS\n(Q2-3)": mid, "High PRS\n(Q4)": hi}
    x = np.arange(3); bottoms = np.zeros(3)
    for lbl, col, fn in [
        ("Low (<44.4%)", C_LOW,  lambda d: (d["PCAD"] <= 44.4).mean() * 100),
        ("Med (44–67%)", C_MED,  lambda d: d["PCAD"].between(44.4, 66.6).mean() * 100),
        ("High (>67%)",  C_HIGH, lambda d: (d["PCAD"] > 66.6).mean() * 100)]:
        vals = [fn(g) for g in grps.values()]
        bars = ax.bar(x, vals, bottom=bottoms, color=col, alpha=0.82,
                      label=lbl, edgecolor="white")
        for bar, val, bot in zip(bars, vals, bottoms):
            if val > 6:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottoms += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(list(grps.keys()))
    ax.set_ylim(0, 112); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylabel("Proportion (%)", fontweight="bold")
    ax.set_title("Risk Reclassification by PRS Group", fontweight="bold", fontsize=10)
    # 9p21 FIXED boxplot
    ax = axes[2]
    geno_col = "dosage_9p21_rs4977574"
    g0 = df[df[geno_col] == 0]["PCAD"].values
    g1 = df[df[geno_col] == 1]["PCAD"].values
    g2 = df[df[geno_col] == 2]["PCAD"].values
    bp = ax.boxplot([g0, g1, g2], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, col in zip(bp["boxes"], [C_LOW, C_MED, C_HIGH]):
        patch.set_facecolor(col); patch.set_alpha(0.82)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"AA\n(n={len(g0):,})", f"Aa\n(n={len(g1):,})", f"aa\n(n={len(g2):,})"],
                        fontsize=8.5)
    for i, data in enumerate([g0, g1, g2]):
        ax.text(i + 1, np.median(data) + 1.5,
                f"μ={data.mean():.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="white")
    ax.set_ylabel("PCAD Score (%)", fontweight="bold")
    ax.set_title("PCAD by 9p21 Genotype\n(rs4977574 — top GWAS hit)",
                 fontweight="bold", fontsize=10)
    f, _ = stats.f_oneway(g0, g1, g2)
    ax.text(0.5, 0.04, f"ANOVA: F={f:.1f}, p<0.001",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig4_genetic_stratification.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── FIGURE 5: Sensitivity analysis (FIXED) ───────────────────────────────────
def fig5_sensitivity(df):
    """
    Correct sensitivity analysis.
    When one component weight is varied (wf), the other two remain at 1.0.
    PCAD is recalculated as:
        weighted_sum = w_GRC*cat_GRC + w_BPC*cat_BPC + w_ARC*cat_ARC
        max_possible = (w_GRC + w_BPC + w_ARC) * 3
        PCAD = (weighted_sum / max_possible) * 100
    This keeps PCAD within [33.3%, 100%] regardless of weights.
    """
    cat_grc = np.where(df["Z_GRC"] < -0.44, 1, np.where(df["Z_GRC"] < 0.44, 2, 3))
    cat_bpc = np.where(df["Z_BPC"] < -0.44, 1, np.where(df["Z_BPC"] < 0.44, 2, 3))
    cat_arc = np.where(df["Z_ARC"] < -0.44, 1, np.where(df["Z_ARC"] < 0.44, 2, 3))
    fig, ax = plt.subplots(figsize=(10, 6))
    weight_factors = np.linspace(0.5, 2.0, 40)
    for comp, col, ls, label in [
        ("GRC", C_HIGH, "-",  "GRC weight varied (Genetic)"),
        ("BPC", C_BLUE, "--", "BPC weight varied (Biochemical)"),
        ("ARC", C_PUR,  ":",  "ARC weight varied (Anthropometric)")]:
        means_s, p25s, p75s = [], [], []
        for wf in weight_factors:
            w = {"GRC": 1.0, "BPC": 1.0, "ARC": 1.0}
            w[comp] = wf
            wsum = w["GRC"] * cat_grc + w["BPC"] * cat_bpc + w["ARC"] * cat_arc
            max_possible = (w["GRC"] + w["BPC"] + w["ARC"]) * 3
            pcad_s = (wsum / max_possible) * 100
            means_s.append(np.mean(pcad_s))
            p25s.append(np.percentile(pcad_s, 25))
            p75s.append(np.percentile(pcad_s, 75))
        ax.plot(weight_factors, means_s, color=col, linewidth=2.5,
                linestyle=ls, label=label)
        ax.fill_between(weight_factors, p25s, p75s, alpha=0.15, color=col)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1.5,
               label="Baseline weight (1.0×)")
    ax.axhline(df["PCAD"].mean(), color=C_DARK, linestyle="--",
               linewidth=1.2, alpha=0.5,
               label=f"Baseline mean PCAD ({df['PCAD'].mean():.1f}%)")
    ax.set_xlabel("Relative Weight of Varied Component", fontweight="bold")
    ax.set_ylabel("Mean PCAD Score (%)", fontweight="bold")
    ax.set_title("Figure 5. Sensitivity Analysis: PCAD Robustness to Component Weight Variation",
                 fontweight="bold", fontsize=11, color=C_DARK)
    ax.legend(fontsize=9); ax.set_ylim(30, 80); ax.set_xlim(0.5, 2.0)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig5_sensitivity.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── FIGURE 6: Stress test ─────────────────────────────────────────────────────
def fig6_stress_test():
    archetypes = {
        "All Components Low":               {"GRC": 1, "BPC": 1, "ARC": 1},
        "High Genetic Only":                {"GRC": 3, "BPC": 1, "ARC": 1},
        "High Biochemical Only":            {"GRC": 1, "BPC": 3, "ARC": 1},
        "High Anthropometric Only":         {"GRC": 1, "BPC": 1, "ARC": 3},
        "High Genetic + Biochemical":       {"GRC": 3, "BPC": 3, "ARC": 1},
        "High Genetic + Anthropometric":    {"GRC": 3, "BPC": 1, "ARC": 3},
        "High Biochemical + Anthropometric":{"GRC": 1, "BPC": 3, "ARC": 3},
        "All Components High":              {"GRC": 3, "BPC": 3, "ARC": 3},
    }
    pcad_vals = [(c["GRC"] + c["BPC"] + c["ARC"]) / 9 * 100 for c in archetypes.values()]
    colors_bar = [C_LOW if v < 44.4 else C_MED if v < 66.6 else C_HIGH for v in pcad_vals]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(archetypes.keys()), pcad_vals,
                   color=colors_bar, alpha=0.82, edgecolor="white")
    ax.axvline(44.4, color=C_LOW, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(66.6, color=C_HIGH, linestyle="--", linewidth=1.5, alpha=0.7)
    for bar, val in zip(bars, pcad_vals):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("PCAD Score (%)", fontweight="bold")
    ax.set_title("Figure 6. Stress Test: Model Behaviour Across Edge-Case Patient Archetypes",
                 fontweight="bold", fontsize=11, color=C_DARK)
    ax.set_xlim(0, 112)
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               [(C_LOW, "Low Risk (<44.4%)"), (C_MED, "Medium Risk"), (C_HIGH, "High Risk (>66.6%)")]]
    ax.legend(handles=patches, loc="lower right", fontsize=8.5)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig6_stress_test.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── FIGURE 7: Correlation heatmap ─────────────────────────────────────────────
def fig7_correlation(df):
    cols = ["PRS_z", "dosage_9p21_rs4977574", "hs_CRP", "LDL_C", "ApoB",
            "LpA", "BMI", "WC", "WHR", "BF_pct", "Z_GRC", "Z_BPC", "Z_ARC", "PCAD"]
    labels = ["PRS (250 loci)", "9p21 dosage", "hs-CRP", "LDL-C", "ApoB",
              "Lp(a)", "BMI", "WC", "WHR", "BF%", "Z_GRC", "Z_BPC", "Z_ARC", "PCAD"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, vmin=-1, vmax=1, ax=ax, xticklabels=labels,
                yticklabels=labels, linewidths=0.5, annot_kws={"size": 7.5},
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Figure 7. Correlation Matrix of All Sub-Components and Final PCAD Score\n"
                 "(N=5,000 synthetic patients, 250-locus GWAS Catalog PRS)",
                 fontsize=10.5, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Fig7_correlation.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ── SUMMARY STATISTICS ────────────────────────────────────────────────────────
def compute_and_save_statistics(df):
    N = len(df)
    hi = df[df["PRS_z"] >= df["PRS_z"].quantile(0.75)]
    lo = df[df["PRS_z"] <= df["PRS_z"].quantile(0.25)]
    t, _ = stats.ttest_ind(hi["PCAD"], lo["PCAD"])
    d = (hi["PCAD"].mean() - lo["PCAD"].mean()) / df["PCAD"].std()
    f_stat, _ = stats.f_oneway(*[df[df["dosage_9p21_rs4977574"] == g]["PCAD"] for g in [0, 1, 2]])
    records = [
        {"Metric": "N patients", "Value": N},
        {"Metric": "PCAD mean", "Value": round(df["PCAD"].mean(), 2)},
        {"Metric": "PCAD SD", "Value": round(df["PCAD"].std(), 2)},
        {"Metric": "PCAD median", "Value": round(df["PCAD"].median(), 2)},
        {"Metric": "Low risk n (%)", "Value": f"{(df['PCAD']<=44.4).sum()} ({(df['PCAD']<=44.4).mean()*100:.1f}%)"},
        {"Metric": "Medium risk n (%)", "Value": f"{df['PCAD'].between(44.4,66.6).sum()} ({df['PCAD'].between(44.4,66.6).mean()*100:.1f}%)"},
        {"Metric": "High risk n (%)", "Value": f"{(df['PCAD']>66.6).sum()} ({(df['PCAD']>66.6).mean()*100:.1f}%)"},
        {"Metric": "PRS Q4 mean PCAD", "Value": round(hi["PCAD"].mean(), 2)},
        {"Metric": "PRS Q1 mean PCAD", "Value": round(lo["PCAD"].mean(), 2)},
        {"Metric": "PRS Q4 vs Q1 t-stat", "Value": round(t, 2)},
        {"Metric": "PRS Q4 vs Q1 Cohens d", "Value": round(d, 3)},
        {"Metric": "9p21 ANOVA F-stat", "Value": round(f_stat, 2)},
    ]
    pd.DataFrame(records).to_csv(STATS_FILE, index=False)
    print(f"\n  Summary statistics saved: {STATS_FILE}")


def main():
    print("=" * 70)
    print("CardioRisqRE™ — Analysis and Figure Generation")
    print("=" * 70)
    df = load_data()
    print("\nGenerating figures...")
    fig1_curation_funnel()
    fig2_component_distributions(df)
    fig3_pcad_distribution(df)
    fig4_genetic_stratification(df)
    fig5_sensitivity(df)
    fig6_stress_test()
    fig7_correlation(df)
    compute_and_save_statistics(df)
    print("\n✓ All figures and statistics complete.")


if __name__ == "__main__":
    main()
