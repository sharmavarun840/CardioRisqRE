# CardioRisqRE™ Simulation Study

## Overview

This repository contains the Python-based computational workflow used for the **CardioRisqRE™ simulation study**. The pipeline was developed to curate coronary artery disease associated variants from the NHGRI-EBI GWAS Catalog, generate a reproducible synthetic patient cohort, evaluate the behaviour of the CardioRisqRE™ risk scoring framework, perform statistical and sensitivity analyses, generate manuscript figures, and assess synthetic data quality.

The complete workflow consists of four sequential Python scripts.

## Workflow

```text
NHGRI-EBI GWAS Catalog
        |
        v
01_gwas_curation.py
        |
        v
Curated independent CAD loci
        |
        v
02_synthetic_data_generation.py
        |
        v
Synthetic cohort, N = 5,000
        |
        +----------------------+
        |                      |
        v                      v
03_analysis_and_figures.py    04_synthro_quality_evaluation.py
        |                      |
        v                      v
Statistical analysis          Synthetic data quality assessment
and manuscript figures       using the SynthRO framework
```

## 1. GWAS Catalog Data Curation

**Script:** `01_gwas_curation.py`

This script processes coronary artery disease associations obtained from the NHGRI-EBI GWAS Catalog and generates the curated variant panel used for polygenic risk score construction.

The curation pipeline includes:

1. Selection of genome-wide significant associations at `p < 5 × 10^-8`
2. Removal of records without valid risk allele frequencies
3. Exclusion of implausible effect sizes
4. Harmonisation of odds ratio and beta effect estimates
5. Application of sanity bounds to harmonised odds ratios
6. Region-based selection of independent loci

Beta coefficients from identified beta-scale studies are transformed to odds ratios using:

```text
OR = exp(beta)
```

The resulting curated dataset contains independent genome-wide significant CAD-associated loci used in the subsequent simulation.

**Input**

```text
gwas-association-downloaded_2026-06-30-MONDO_0005010-withChildTraits.tsv
```

**Output**

```text
curated_250_independent_loci.csv
```

## 2. Synthetic Cohort Generation

**Script:** `02_synthetic_data_generation.py`

This script generates a synthetic cohort of **5,000 virtual individuals** representing the three major components of the CardioRisqRE™ framework.

A fixed random seed of `42` is used to ensure reproducibility.

### Genetic Risk Component, GRC

Genotypes are simulated for the curated CAD loci under Hardy-Weinberg equilibrium.

For a risk allele frequency `p`, genotype dosages of 0, 1, and 2 risk alleles are generated according to:

```text
P(0) = (1-p)^2
P(1) = 2p(1-p)
P(2) = p^2
```

The polygenic risk score is calculated using an additive weighted model:

```text
PRS = Σ beta_i × dosage_i
```

where:

```text
beta_i = ln(OR_i)
```

The GRC combines the multi-locus PRS with the 9p21 CAD risk locus represented by `rs4977574`.

The implemented component weights are:

```text
PRS   = 0.80
9p21  = 0.20
```

### Biochemical/Pathological Investigation Component, BPC

The biochemical component includes:

```text
hs-CRP
LDL-C
ApoB
Lp(a)
```

Correlated biochemical variables are generated using multivariate sampling based on the correlation structure implemented in the simulation model. Lp(a) is generated using a log-normal distribution.

The BPC weights are:

```text
hs-CRP = 0.20
LDL-C  = 0.30
ApoB   = 0.30
Lp(a)  = 0.20
```

### Anthropometric Test Results Component, ARC

The anthropometric component includes:

```text
BMI
Waist circumference
Waist-to-hip ratio
Body fat percentage
```

These variables are generated using multivariate normal sampling with the correlation structure specified in the simulation framework.

The ARC weights are:

```text
BMI                     = 0.10
Waist circumference     = 0.30
Waist-to-hip ratio      = 0.30
Body fat percentage     = 0.30
```

### Risk Classification

Component Z-scores are classified into three risk categories:

```text
Low risk:       Z < -0.44
Medium risk:   -0.44 ≤ Z < 0.44
High risk:      Z ≥ 0.44
```

The final CardioRisqRE™ PCAD score is calculated as:

```text
PCAD = [(Cat_GRC + Cat_BPC + Cat_ARC) / 9] × 100
```

The theoretical PCAD score therefore ranges from approximately **33.33% to 100%**.

**Input**

```text
curated_250_independent_loci.csv
```

**Output**

```text
synthetic_cohort_N5000.csv
```

## 3. Statistical Analysis and Figure Generation

**Script:** `03_analysis_and_figures.py`

This script performs statistical evaluation of the synthetic cohort and generates the principal figures used to examine the behaviour of the CardioRisqRE™ scoring framework.

The analysis includes:

* Distribution of GRC, BPC, and ARC Z-scores
* Overall PCAD score distribution
* Component-specific risk classification
* PRS-based genetic risk stratification
* Comparison of low and high PRS groups
* Evaluation of the 9p21 genotype contribution
* Component-weight sensitivity analysis
* Edge-case stress testing
* Correlation analysis across model variables

### Generated Figures

```text
Fig1_curation_funnel.png
Fig2_component_distributions.png
Fig3_PCAD_distribution.png
Fig4_genetic_stratification.png
Fig5_sensitivity.png
Fig6_stress_test.png
Fig7_correlation.png
```

The script also generates:

```text
analysis_summary_statistics.csv
```

All figures are written to the:

```text
figures/
```

directory.

## 4. Synthetic Data Quality Evaluation

**Script:** `04_synthro_quality_evaluation.py`

Synthetic data quality is evaluated using three resemblance modules implemented according to the SynthRO framework.

An independent reference cohort of **5,000 individuals** is generated using a separate random seed of `999` and an independent implementation of the simulation procedure.

### Univariate Resemblance Analysis, URA

URA evaluates individual variable distributions using:

```text
Kolmogorov-Smirnov test
Wasserstein distance
```

The Wasserstein distance is normalised relative to the standard deviation of the corresponding reference variable.

### Multivariate Relationship Analysis, MRA

MRA evaluates preservation of relationships between variables by comparing synthetic and reference correlation matrices.

Metrics include:

```text
Frobenius norm
Maximum absolute correlation difference
Mean absolute correlation difference
```

### Data Labeling Analysis, DLA

A Random Forest classifier is trained to distinguish synthetic observations from independently generated reference observations.

The area under the receiver operating characteristic curve, AUC, is used as the discrimination metric.

An AUC approaching `0.50` indicates that the two datasets are difficult for the classifier to distinguish.

### Outputs

```text
synthro_URA_results.csv
synthro_MRA_corrdiff.csv
synthro_quality_summary.csv
```

Additional figures are generated as:

```text
figures/Fig8_URA.png
figures/Fig9_MRA_heatmap.png
figures/Fig10_DLA.png
```

## Repository Structure

```text
CardioRisqRE/
|
|-- 01_gwas_curation.py
|-- 02_synthetic_data_generation.py
|-- 03_analysis_and_figures.py
|-- 04_synthro_quality_evaluation.py
|
|-- gwas-association-downloaded_2026-06-30-MONDO_0005010-withChildTraits.tsv
|-- curated_250_independent_loci.csv
|-- synthetic_cohort_N5000.csv
|
|-- analysis_summary_statistics.csv
|-- synthro_URA_results.csv
|-- synthro_MRA_corrdiff.csv
|-- synthro_quality_summary.csv
|
`-- figures/
    |-- Fig1_curation_funnel.png
    |-- Fig2_component_distributions.png
    |-- Fig3_PCAD_distribution.png
    |-- Fig4_genetic_stratification.png
    |-- Fig5_sensitivity.png
    |-- Fig6_stress_test.png
    |-- Fig7_correlation.png
    |-- Fig8_URA.png
    |-- Fig9_MRA_heatmap.png
    `-- Fig10_DLA.png
```

## Requirements

The workflow was developed using **Python 3.11**.

Required Python packages include:

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
```

Packages can be installed using:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

## Running the Pipeline

The scripts should be executed sequentially because outputs from earlier stages are used as inputs for subsequent analyses.

```bash
python 01_gwas_curation.py
python 02_synthetic_data_generation.py
python 03_analysis_and_figures.py
python 04_synthro_quality_evaluation.py
```

The expected workflow is:

```text
GWAS Catalog data
    ↓
Variant curation
    ↓
Synthetic cohort generation
    ↓
CardioRisqRE™ statistical analysis
    ↓
Sensitivity and stress testing
    ↓
Synthetic data quality evaluation
```

## Reproducibility

Random number generation is explicitly seeded to support reproducibility.

```text
Synthetic cohort seed:   42
Reference cohort seed:    999
```

Running the scripts with the same input files, package environment, and random seeds should reproduce the simulation framework and associated analyses.

## Data Source

CAD-associated variants were obtained from the **NHGRI-EBI GWAS Catalog** using the coronary artery disease ontology entry `MONDO_0005010`, including child traits.

The GWAS Catalog dataset used by the curation pipeline was downloaded on **30 June 2026**.

## Methodological References

The computational workflow draws on the following methodological resources:

1. Sollis E, et al. The NHGRI-EBI GWAS Catalog: knowledgebase and deposition resource. *Nucleic Acids Research*. 2023;51(D1):D977-D985.

2. Purcell S, et al. PLINK: a tool set for whole-genome association and population-based linkage analyses. *American Journal of Human Genetics*. 2007;81(3):559-575.

3. Chang CC, et al. Second-generation PLINK: rising to the challenge of larger and richer datasets. *GigaScience*. 2015;4:7.

4. Choi SW, Mak TS-H, O'Reilly PF. Tutorial: a guide to performing polygenic risk score analyses. *Nature Protocols*. 2020;15(9):2759-2772.

5. Lambert SA, et al. The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation. *Nature Genetics*. 2021;53(4):420-425.

6. Inouye M, et al. Genomic risk prediction of coronary artery disease in 480,000 adults. *Journal of the American College of Cardiology*. 2018;72(16):1883-1893.

7. Santangelo G, Nicora G, Bellazzi R, Dagliati A. How good is your synthetic data? SynthRO, a dashboard to evaluate and benchmark synthetic tabular data. *BMC Medical Informatics and Decision Making*. 2025;25:89.

8. Raab GM, Nowok B, Dibben C. Assessing, visualizing and improving the utility of synthetic data. *arXiv*. 2021;2109.12717.

## Important Note

The generated cohort is **synthetic** and was created for methodological evaluation of the CardioRisqRE™ scoring framework. The simulated individuals do not represent real patients or individual-level clinical records.

The simulation results should therefore be interpreted as an evaluation of the computational behaviour, internal consistency, sensitivity, and risk-stratification characteristics of the proposed framework rather than as prospective clinical validation.

## Citation

If this code is used in academic work, please cite the associated CardioRisqRE™ manuscript and the methodological references relevant to the specific analysis performed.

## License

Copyright © 2026. All rights reserved.

CardioRisqRE™ is a trademark of its respective owner.
