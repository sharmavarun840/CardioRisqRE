"""
================================================================================
CardioRisqRE™ Simulation Study
Script 01: GWAS Catalog Data Curation Pipeline
================================================================================

Description:
    Reads the raw NHGRI-EBI GWAS Catalog export for MONDO_0005010
    (coronary artery disease including child traits) and applies a
    six-step quality-control pipeline to produce a final curated panel
    of independent, genome-wide-significant CAD risk loci suitable for
    polygenic risk score construction.

Input:
    gwas-association-downloaded_2026-06-30-MONDO_0005010-withChildTraits.tsv
    (downloaded from https://www.ebi.ac.uk/gwas on 30 June 2026)

Output:
    curated_250_independent_loci.csv

Citation for methodology:
    - GWAS Catalog: Sollis E et al. Nucleic Acids Res. 2023;51(D1):D977-D985
    - GW significance threshold: Panagiotou OA & Ioannidis JPA.
      Int J Epidemiol. 2012;41(1):273-286
    - OR/BETA harmonisation: Lambert SA et al. Nat Genet. 2021;53(4):420-425
      (PGS Catalog harmonisation standard)
    - LD clumping protocol: Choi SW, Mak TS-H, O'Reilly PF.
      Nat Protoc. 2020;15(9):2759-2772

Python version: 3.11
Required packages: pandas, numpy
================================================================================
"""

import pandas as pd
import numpy as np

# ── INPUT FILE ────────────────────────────────────────────────────────────────
INPUT_FILE = "gwas-association-downloaded_2026-06-30-MONDO_0005010-withChildTraits.tsv"
OUTPUT_FILE = "curated_250_independent_loci.csv"

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
GW_SIGNIFICANCE_THRESHOLD = 5e-8   # Panagiotou & Ioannidis 2012
OR_LOWER_BOUND = 0.3                # minimum plausible OR for CAD variants
OR_UPPER_BOUND = 3.5                # maximum plausible OR for common CAD variants
COMMON_VARIANT_RAF_THRESHOLD = 0.05 # RAF > 5% = common variant
IMPLAUSIBLE_OR_COMMON = 3.0         # common variants should not have OR > 3.0

# Studies identified as reporting beta coefficients (log-odds scale)
# rather than odds ratios, based on systematic inspection of value distributions.
# These are converted to OR via OR = exp(beta).
# Identification method: if >70% of a study's values fall below |0.5|,
# the study is classified as reporting betas rather than ORs.
BETA_SCALE_STUDIES = [
    "van der Harst P",   # Circ Res 2018 - large biobank regression
    "Verma A",           # Science 2024 - biobank
    "Tcheandjieu C",     # Nat Med 2022 - biobank
    "Zhao C",            # biobank study
    "Kulminski AM",      # biobank study
]


def load_raw_data(filepath):
    """Load raw GWAS Catalog TSV export."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"[Step 0] Raw GWAS Catalog export: {len(df)} associations")
    return df


def convert_types(df):
    """Ensure numeric columns are correctly typed."""
    df["P-VALUE"] = pd.to_numeric(df["P-VALUE"], errors="coerce")
    df["RISK ALLELE FREQUENCY"] = pd.to_numeric(df["RISK ALLELE FREQUENCY"], errors="coerce")
    df["OR or BETA"] = pd.to_numeric(df["OR or BETA"], errors="coerce")
    return df


def step1_gwas_significance(df):
    """Step 1: Retain only genome-wide significant associations (p < 5e-8)."""
    before = len(df)
    df = df[df["P-VALUE"] < GW_SIGNIFICANCE_THRESHOLD].copy()
    print(f"[Step 1] After GW significance filter (p<5e-8): {len(df)} retained, {before - len(df)} removed")
    return df


def step2_valid_raf(df):
    """Step 2: Require valid risk allele frequency (0 < RAF < 1)."""
    before = len(df)
    df = df[(df["RISK ALLELE FREQUENCY"] > 0) & (df["RISK ALLELE FREQUENCY"] < 1)].copy()
    print(f"[Step 2] After valid RAF filter: {len(df)} retained, {before - len(df)} removed")
    return df


def step3_implausible_effects(df):
    """Step 3: Remove biologically implausible effect sizes.

    Common variants (RAF > 5%) with claimed OR > 3.0 are essentially
    unobserved in the published CAD genetics literature and likely
    represent catalogue transcription or unit errors.
    Excludes known beta-scale studies from this filter (handled in Step 4).
    """
    before = len(df)
    is_beta_study = df["FIRST AUTHOR"].isin(BETA_SCALE_STUDIES)
    is_common = df["RISK ALLELE FREQUENCY"] > COMMON_VARIANT_RAF_THRESHOLD
    is_implausible = df["OR or BETA"] > IMPLAUSIBLE_OR_COMMON
    mask_remove = is_common & is_implausible & ~is_beta_study
    df = df[~mask_remove].copy()
    print(f"[Step 3] After implausible effect-size exclusion: {len(df)} retained, {before - len(df)} removed")
    return df


def step4_harmonise_scale(df):
    """Step 4: Harmonise OR/BETA scale.

    Convert beta-scale entries to odds ratios via OR = exp(beta),
    following the PGS Catalog harmonisation convention (Lambert et al. 2021).
    No SNPs are removed at this step.
    """
    df["scale"] = "OR"
    df.loc[df["FIRST AUTHOR"].isin(BETA_SCALE_STUDIES), "scale"] = "BETA"
    df["OR_final"] = df["OR or BETA"].copy()
    df.loc[df["scale"] == "BETA", "OR_final"] = np.exp(df.loc[df["scale"] == "BETA", "OR or BETA"])
    beta_count = (df["scale"] == "BETA").sum()
    print(f"[Step 4] Scale harmonisation: {beta_count} beta-scale entries converted via OR=exp(beta); 0 SNPs removed")
    return df


def step5_sanity_bound(df):
    """Step 5: Apply sanity bounds on the harmonised OR (0.3 to 3.5)."""
    before = len(df)
    df = df[(df["OR_final"] >= OR_LOWER_BOUND) & (df["OR_final"] <= OR_UPPER_BOUND)].copy()
    print(f"[Step 5] After OR sanity bound ({OR_LOWER_BOUND}-{OR_UPPER_BOUND}): {len(df)} retained, {before - len(df)} removed")
    return df


def step6_ld_clumping(df):
    """Step 6: Region-based LD clumping.

    Retains only the most statistically significant SNP (lowest p-value)
    per cytogenetic REGION, yielding independent loci.
    Method: Choi, Mak & O'Reilly, Nat Protoc 2020.
    """
    before = len(df)
    df_sorted = df.sort_values("P-VALUE")
    df_clumped = df_sorted.drop_duplicates(subset="REGION", keep="first").copy()
    df_clumped = df_clumped.reset_index(drop=True)
    print(f"[Step 6] After region-based clumping: {len(df_clumped)} independent loci retained, {before - len(df_clumped)} removed")
    return df_clumped


def main():
    print("=" * 70)
    print("CardioRisqRE™ — GWAS Catalog Curation Pipeline")
    print("=" * 70)

    df = load_raw_data(INPUT_FILE)
    df = convert_types(df)
    df = step1_gwas_significance(df)
    df = step2_valid_raf(df)
    df = step3_implausible_effects(df)
    df = step4_harmonise_scale(df)
    df = step5_sanity_bound(df)
    df = step6_ld_clumping(df)

    print(f"\n{'=' * 70}")
    print(f"FINAL CURATED PANEL: {len(df)} independent genome-wide-significant CAD loci")
    print(f"OR range: {df['OR_final'].min():.3f} – {df['OR_final'].max():.3f}")
    print(f"RAF range: {df['RISK ALLELE FREQUENCY'].min():.3f} – {df['RISK ALLELE FREQUENCY'].max():.3f}")
    print(f"{'=' * 70}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
