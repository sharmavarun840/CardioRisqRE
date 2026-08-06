"""
================================================================================
CardioRisqRE™ Simulation Study
Script 02: Synthetic Cohort Generation
================================================================================

Description:
    Generates a synthetic cohort of N = 5,000 virtual patients by simulating
    all three major CardioRisqRE™ components:

    1. GRC — Genetic Risk Component
       250-locus composite PRS + 9p21 standalone sub-component
       Genotype simulation: PLINK --simulate protocol (Purcell et al. 2007)

    2. BPC — Biochemical/Pathological Investigation Component
       hs-CRP, LDL-C, ApoB, Lp(a)
       Multivariate normal sampling with published correlation structure

    3. ARC — Anthropometric Test Results Component
       BMI, waist circumference, WHR, body fat %
       Multivariate normal sampling with published correlation structure

    Final PCAD score computed as per CardioRisqRE™ manuscript methodology.

Input:
    curated_250_independent_loci.csv   (output of Script 01)

Output:
    synthetic_cohort_N5000.csv         (full synthetic dataset)

Citations for synthetic data generation methodology:
    - PLINK --simulate protocol (primary):
      Purcell S et al. Am J Hum Genet. 2007;81(3):559-575
    - PLINK v2 (updated implementation):
      Chang CC et al. GigaScience. 2015;4:7
    - Additive PRS construction:
      Inouye M et al. J Am Coll Cardiol. 2018;72(16):1883-1893
    - LDL-C/ApoB correlation structure:
      Contois JH et al. Clin Chem. 2009;55(3):407-419
    - Lp(a) log-normal distribution:
      Kronenberg F et al. Eur Heart J. 2022;43(39):3925-3946
    - Anthropometric correlation structure:
      Romero-Corral A et al. Int J Obes. 2008;32(6):959-966
    - Anthropometric reference ranges:
      WHO Expert Consultation. Geneva: WHO; 2008

Python version: 3.11
Required packages: numpy, pandas, scipy
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42          # fixed seed for full reproducibility
N_PATIENTS = 5000         # synthetic cohort size
LOCUS_FILE = "curated_250_independent_loci.csv"
OUTPUT_FILE = "synthetic_cohort_N5000.csv"

# 9p21 top SNP (most significant in curated panel)
SNP_9P21 = "rs4977574"

# GRC sub-component weights (from CardioRisqRE™ manuscript Table 5)
W_PRS  = 0.8
W_9P21 = 0.2

# BPC sub-component weights (from CardioRisqRE™ manuscript Table 5)
# hs-CRP, LDL-C, ApoB, Lp(a)
BPC_WEIGHTS = np.array([0.20, 0.30, 0.30, 0.20])

# ARC sub-component weights (from CardioRisqRE™ manuscript Table 5)
# BMI, WC, WHR, BF%
ARC_WEIGHTS = np.array([0.10, 0.30, 0.30, 0.30])

# Risk category Z-score boundaries (tertile-equivalent)
Z_LOW_BOUNDARY  = -0.44   # below this = Low risk  (~bottom 33%)
Z_HIGH_BOUNDARY = +0.44   # above this = High risk (~top 33%)


# ── SECTION 1: GENETIC RISK COMPONENT (GRC) ──────────────────────────────────

def simulate_genotypes_plink_protocol(locus_df, n_patients, rng):
    """
    Simulate allele dosages for all loci using the PLINK --simulate protocol.

    For each locus, dosage (0, 1, or 2 risk allele copies) is sampled
    independently per patient under Hardy-Weinberg equilibrium (HWE):
        P(dosage=0) = (1 - RAF)^2
        P(dosage=1) = 2 * RAF * (1 - RAF)
        P(dosage=2) = RAF^2

    This implements the PLINK --simulate algorithm in Python.
    Reference: Purcell et al. Am J Hum Genet. 2007;81(3):559-575.
               Chang et al. GigaScience. 2015;4:7.

    Parameters
    ----------
    locus_df : pd.DataFrame
        Curated locus panel with columns RISK ALLELE FREQUENCY and OR_final
    n_patients : int
        Number of synthetic patients
    rng : np.random.Generator
        Seeded random number generator for reproducibility

    Returns
    -------
    dosage_matrix : np.ndarray, shape (n_patients, n_loci)
        Integer allele dosages (0, 1, or 2) for each patient × locus
    """
    n_loci = len(locus_df)
    dosage_matrix = np.zeros((n_patients, n_loci), dtype=np.int8)

    for i, raf in enumerate(locus_df["RISK ALLELE FREQUENCY"].values):
        # HWE probabilities
        p0 = (1 - raf) ** 2   # homozygous reference
        p1 = 2 * raf * (1 - raf)  # heterozygous
        p2 = raf ** 2          # homozygous risk

        dosage_matrix[:, i] = rng.choice([0, 1, 2], size=n_patients, p=[p0, p1, p2])

    return dosage_matrix


def compute_prs(dosage_matrix, betas):
    """
    Compute raw PRS as the weighted sum of risk allele dosages.

    PRS = Σ (beta_i × dosage_i)   for all i loci

    where beta_i = ln(OR_i), the natural log of the harmonised odds ratio.
    This is the standard additive log-odds PRS construction method.
    Reference: Inouye et al. J Am Coll Cardiol. 2018;72(16):1883-1893.

    Parameters
    ----------
    dosage_matrix : np.ndarray, shape (n_patients, n_loci)
    betas : np.ndarray, shape (n_loci,)
        ln(OR) for each locus

    Returns
    -------
    prs_raw : np.ndarray, shape (n_patients,)
    """
    return dosage_matrix @ betas


def standardise_to_zscore(values):
    """Standardise values to Z-score (mean=0, SD=1)."""
    return (values - values.mean()) / values.std()


def zscore_to_percentile(z):
    """Convert Z-scores to percentiles via standard normal CDF."""
    return stats.norm.cdf(z) * 100


def compute_grc(dosage_matrix, locus_df, rng):
    """
    Compute the Genetic Risk Component (GRC) Z-score.

    GRC combines:
    - 250-locus composite PRS (weight 0.8)
    - 9p21 locus standalone sub-component (weight 0.2)

    Returns
    -------
    z_grc : np.ndarray
    z_prs : np.ndarray
    prs_raw : np.ndarray
    dosage_9p21 : np.ndarray
    """
    betas = np.log(locus_df["OR_final"].values)
    prs_raw = compute_prs(dosage_matrix, betas)
    z_prs = standardise_to_zscore(prs_raw)

    # Extract 9p21 dosage as standalone sub-component
    idx_9p21 = locus_df[locus_df["STRONGEST SNP"] == SNP_9P21].index
    if len(idx_9p21) == 0:
        raise ValueError(f"SNP {SNP_9P21} not found in curated locus panel.")
    # Get positional index in the dataframe
    pos_9p21 = locus_df.index.get_loc(idx_9p21[0])
    dosage_9p21 = dosage_matrix[:, pos_9p21].astype(float)

    # Standardise 9p21 dosage (reference: heterozygote mean=1, SD=0.5)
    z_9p21 = (dosage_9p21 - 1.0) / 0.5

    # Weighted GRC composite (weights sum to 1.0)
    z_grc = W_PRS * z_prs + W_9P21 * z_9p21

    return z_grc, z_prs, prs_raw, dosage_9p21


# ── SECTION 2: BIOCHEMICAL COMPONENT (BPC) ───────────────────────────────────

def simulate_bpc(n_patients, rng):
    """
    Simulate Biochemical/Pathological Investigation Component sub-components.

    Variables: hs-CRP (mg/L), LDL-C (mg/dL), ApoB (mg/dL), Lp(a) (mg/dL)

    hs-CRP, LDL-C, ApoB: multivariate normal with published correlation structure.
    Lp(a): log-normal (right-skewed; Kronenberg et al. 2022).

    Returns
    -------
    bpc_raw : np.ndarray, shape (n_patients, 4)
    z_bpc   : np.ndarray, shape (n_patients,)
    """
    # Published reference means and SDs (from CardioRisqRE manuscript Table 5)
    means = np.array([1.0,   100.0, 90.0,  20.0])   # hs-CRP, LDL-C, ApoB, Lp(a)
    sds   = np.array([0.5,    25.0, 20.0,  15.0])

    # Inter-marker correlation matrix
    # LDL-C ↔ ApoB: r≈0.85 (Contois et al. 2009)
    # hs-CRP largely independent of lipids (Ridker et al. 2003)
    # Lp(a) weakly correlated with LDL-C/ApoB
    corr = np.array([
        [1.00, 0.10, 0.10, 0.05],
        [0.10, 1.00, 0.85, 0.20],
        [0.10, 0.85, 1.00, 0.20],
        [0.05, 0.20, 0.20, 1.00]
    ])
    cov = np.outer(sds, sds) * corr

    # Sample hs-CRP, LDL-C, ApoB jointly from multivariate normal
    bpc_raw = rng.multivariate_normal(means, cov, size=n_patients)

    # Replace Lp(a) column with log-normal sample (Kronenberg et al. 2022)
    bpc_raw[:, 3] = rng.lognormal(mean=np.log(20), sigma=0.8, size=n_patients)

    # Compute weighted composite Z-score
    z_markers = (bpc_raw - means) / sds
    z_bpc = (z_markers @ BPC_WEIGHTS) / BPC_WEIGHTS.sum()

    return bpc_raw, z_bpc


# ── SECTION 3: ANTHROPOMETRIC COMPONENT (ARC) ────────────────────────────────

def simulate_arc(n_patients, rng):
    """
    Simulate Anthropometric Test Results Component sub-components.

    Variables: BMI (kg/m²), WC (cm), WHR, BF%

    Multivariate normal sampling with published correlation structure.
    Reference ranges: WHO Expert Consultation (2008).
    Correlations: Romero-Corral et al. Int J Obes. 2008.

    Returns
    -------
    arc_raw : np.ndarray, shape (n_patients, 4)
    z_arc   : np.ndarray, shape (n_patients,)
    """
    means = np.array([24.0, 90.0, 0.85, 20.0])   # BMI, WC, WHR, BF%
    sds   = np.array([ 3.0,  5.0, 0.04,  4.0])

    # Inter-marker correlation matrix
    # WC ↔ WHR: r≈0.80 (both measure central adiposity)
    # BMI ↔ BF%: r≈0.75 (Romero-Corral et al. 2008)
    corr = np.array([
        [1.00, 0.72, 0.65, 0.75],
        [0.72, 1.00, 0.80, 0.68],
        [0.65, 0.80, 1.00, 0.60],
        [0.75, 0.68, 0.60, 1.00]
    ])
    cov = np.outer(sds, sds) * corr

    arc_raw = rng.multivariate_normal(means, cov, size=n_patients)

    z_markers = (arc_raw - means) / sds
    z_arc = (z_markers @ ARC_WEIGHTS) / ARC_WEIGHTS.sum()

    return arc_raw, z_arc


# ── SECTION 4: SCORE CALCULATION ─────────────────────────────────────────────

def assign_risk_category(z):
    """
    Assign categorical risk level based on Z-score thresholds.

    1 = Low risk  (Z ≤ -0.44, approximately bottom tertile)
    2 = Medium risk (-0.44 < Z ≤ +0.44)
    3 = High risk (Z > +0.44, approximately top tertile)
    """
    return np.where(z < Z_LOW_BOUNDARY, 1,
           np.where(z < Z_HIGH_BOUNDARY, 2, 3))


def compute_pcad(cat_grc, cat_bpc, cat_arc):
    """
    Compute final CardioRisqRE™ PCAD score.

    PCAD = [(Cat_GRC + Cat_BPC + Cat_ARC) / 9] × 100%

    Range: 33.33% (all Low, 1+1+1) to 100% (all High, 3+3+3)
    """
    return ((cat_grc + cat_bpc + cat_arc) / 9.0) * 100


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CardioRisqRE™ — Synthetic Cohort Generation")
    print(f"N = {N_PATIENTS} patients | Random seed = {RANDOM_SEED}")
    print("=" * 70)

    # Initialise seeded random number generator (full reproducibility)
    rng = np.random.default_rng(RANDOM_SEED)

    # Load curated locus panel
    locus_df = pd.read_csv(LOCUS_FILE)
    locus_df = locus_df.reset_index(drop=True)
    print(f"\nLoaded curated locus panel: {len(locus_df)} independent CAD loci")

    # ── GRC ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Simulating Genetic Risk Component (GRC)...")
    print("      Protocol: PLINK --simulate (Purcell et al. 2007; Chang et al. 2015)")
    dosage_matrix = simulate_genotypes_plink_protocol(locus_df, N_PATIENTS, rng)
    z_grc, z_prs, prs_raw, dosage_9p21 = compute_grc(dosage_matrix, locus_df, rng)
    pct_grc = zscore_to_percentile(z_grc)
    cat_grc = assign_risk_category(z_grc)
    print(f"      PRS distribution: mean={prs_raw.mean():.4f}, SD={prs_raw.std():.4f}")
    print(f"      GRC categories: Low={( cat_grc==1).sum()} Med={(cat_grc==2).sum()} High={(cat_grc==3).sum()}")

    # ── BPC ──────────────────────────────────────────────────────────────────
    print("\n[2/3] Simulating Biochemical Component (BPC)...")
    bpc_raw, z_bpc = simulate_bpc(N_PATIENTS, rng)
    pct_bpc = zscore_to_percentile(z_bpc)
    cat_bpc = assign_risk_category(z_bpc)
    print(f"      BPC categories: Low={(cat_bpc==1).sum()} Med={(cat_bpc==2).sum()} High={(cat_bpc==3).sum()}")

    # ── ARC ──────────────────────────────────────────────────────────────────
    print("\n[3/3] Simulating Anthropometric Component (ARC)...")
    arc_raw, z_arc = simulate_arc(N_PATIENTS, rng)
    pct_arc = zscore_to_percentile(z_arc)
    cat_arc = assign_risk_category(z_arc)
    print(f"      ARC categories: Low={(cat_arc==1).sum()} Med={(cat_arc==2).sum()} High={(cat_arc==3).sum()}")

    # ── PCAD ─────────────────────────────────────────────────────────────────
    pcad = compute_pcad(cat_grc, cat_bpc, cat_arc)

    # ── COMPILE OUTPUT DATAFRAME ──────────────────────────────────────────────
    meta = pd.DataFrame({
        "PRS_raw":        prs_raw,
        "PRS_z":          z_prs,
        "dosage_9p21_rs4977574": dosage_9p21,
        "hs_CRP":         bpc_raw[:, 0],
        "LDL_C":          bpc_raw[:, 1],
        "ApoB":           bpc_raw[:, 2],
        "LpA":            bpc_raw[:, 3],
        "BMI":            arc_raw[:, 0],
        "WC":             arc_raw[:, 1],
        "WHR":            arc_raw[:, 2],
        "BF_pct":         arc_raw[:, 3],
        "Z_GRC":  z_grc,  "Pct_GRC": pct_grc,  "Cat_GRC": cat_grc,
        "Z_BPC":  z_bpc,  "Pct_BPC": pct_bpc,  "Cat_BPC": cat_bpc,
        "Z_ARC":  z_arc,  "Pct_ARC": pct_arc,  "Cat_ARC": cat_arc,
        "PCAD":   pcad,
    })

    # Add all 250 SNP dosage columns
    snp_cols = pd.DataFrame(
        dosage_matrix.astype(int),
        columns=[
            f"{row['STRONGEST SNP']}_{str(row['MAPPED_GENE']).split('-')[0][:10]}"
            for _, row in locus_df.iterrows()
        ]
    )
    full_df = pd.concat([meta, snp_cols], axis=1)
    full_df.to_csv(OUTPUT_FILE, index=False)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    low  = (pcad <= 44.4).sum()
    med  = ((pcad > 44.4) & (pcad <= 66.6)).sum()
    high = (pcad > 66.6).sum()
    print(f"\n{'=' * 70}")
    print("SYNTHETIC COHORT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  N patients         : {N_PATIENTS:,}")
    print(f"  SNPs in PRS        : {len(locus_df)}")
    print(f"  PCAD mean ± SD     : {pcad.mean():.2f} ± {pcad.std():.2f}%")
    print(f"  PCAD median (IQR)  : {np.median(pcad):.2f}% ({np.percentile(pcad,25):.2f}–{np.percentile(pcad,75):.2f}%)")
    print(f"  Low risk  (<44.4%) : {low:,} ({low/N_PATIENTS*100:.1f}%)")
    print(f"  Med risk (44–67%)  : {med:,} ({med/N_PATIENTS*100:.1f}%)")
    print(f"  High risk (>66.6%) : {high:,} ({high/N_PATIENTS*100:.1f}%)")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
