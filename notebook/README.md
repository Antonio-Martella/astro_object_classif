# Exploratory Data Analysis & Prototyping Layer (`notebook/`)

This directory houses the scientific exploratory data analyses, statistical proofs, astrophysical domain investigations, and feature engineering prototypes for the **Astro Object Classification** system. 

Serving as the research foundation for the production pipelines in `src/`, this directory bridges observational astrophysics with enterprise MLOps: every architectural decision in the production codebase (data cleaning, validation splitting, feature engineering, and drift monitoring) originates from empirical discoveries documented here.

---

## Table of Contents

1. [Architecture Overview & MLOps Role](#architecture-overview--mlops-role)
2. [Directory Structure](#directory-structure)
3. [Exploratory Analysis Walkthrough (`EDA.ipynb`)](#exploratory-analysis-walkthrough-edaipynb)
   - [Section 1: Project Scope & Executive Summary](#section-1-project-scope--executive-summary)
   - [Section 2: Initial Data Hygiene & Target Variable Exploration](#section-2-initial-data-hygiene--target-variable-exploration)
   - [Section 3: Instrumental Metadata & Leakage Prevention](#section-3-instrumental-metadata--leakage-prevention)
   - [Section 4: Photometric Flux Analysis & Anomaly Detection](#section-4-photometric-flux-analysis--anomaly-detection)
   - [Section 5: Kinematics & Redshift Analysis](#section-5-kinematics--redshift-analysis)
   - [Section 6: Astrophysical Feature Engineering (Color Indices & Color-Color Space)](#section-6-astrophysical-feature-engineering-color-indices--color-color-space)
   - [Section 7: Final MLOps Architectural Blueprint & Production Strategy](#section-7-final-mlops-architectural-blueprint--production-strategy)
4. [The MLOps Bridge: From Notebook Discoveries to Production Code](#the-mlops-bridge-from-notebook-discoveries-to-production-code)
5. [Execution Guide & Environment Setup](#execution-guide--environment-setup)
6. [Governance, Reproducibility & Git Hygiene](#governance-reproducibility--git-hygiene)

---

## Architecture Overview & MLOps Role

In production machine learning systems, notebooks serve exclusively as **exploratory, investigative, and pedagogical workspaces**. Long-term maintenance issues arise when production logic is developed haphazardly in notebooks and copied manually into backend services.

To ensure strict separation of concerns, the relationship between `notebook/` and `src/` is governed by three operational principles:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NOTEBOOK LAYER (notebook/)                      │
│   • Hypothesis formulation and statistical significance testing        │
│   • Sensor anomaly detection (-9999 sentinels)                         │
│   • Astrophysical color-index feature engineering prototyping          │
│   • Publication-grade diagnostic plots (reports/figures/)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Insights Formalized into
                                    │ Deterministic Software Components
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION LAYER (src/)                         │
│   • src/data/        -> GroupShuffleSplit holdout & cleaning pipelines │
│   • src/features/    -> AstroFeatureEngineer scikit-learn transformer  │
│   • src/training/    -> Stratified & GroupKFold cross-validation       │
│   • src/monitoring/  -> Automated KS-test data & target drift triggers │
└────────────────────────────────────────────────────────────────────────┘
```

1. **One-Way Direction of Logic:** Discoveries in `notebook/EDA.ipynb` dictate what is written into `src/`. Production code is **never** imported or executed backwards into ad-hoc untracked notebooks.
2. **Centralized Configuration:** The notebook imports paths directly from `configs.paths.DataPathConfig`. File locations, raw inputs, and output figure directories are never hardcoded as relative strings.
3. **Reproducibility & Execution Hygiene:** Every notebook must execute end-to-end sequentially without errors (`Restart & Run All`), producing identical statistical outputs across development environments.

---

## Directory Structure

```
notebook/
├── README.md               # Architecture documentation, research synthesis & execution guide
└── EDA.ipynb               # 7-section exploratory data analysis
```

---

## Exploratory Analysis Walkthrough (`EDA.ipynb`)

`EDA.ipynb` is structured into seven logical sections, progressing from initial data integrity checks to statistical leakage screening, physical domain modeling, and production deployment directives.

### Section 1: Project Scope & Executive Summary
* **Scope:** 100,000 celestial observations from the Sloan Digital Sky Survey (SDSS DR17).
* **CLI Ingestion Guard:** Features an integrated operational alert reminding engineers to generate the raw dataset via `python main.py --mode make-datasets` if `data/raw/Star_Classification.csv` is missing.
* **Environment Setup:** Configures Matplotlib/Seaborn graphic standards, resolves project root via `sys.path`, and loads directory paths from `DataPathConfig`.

### Section 2: Initial Data Hygiene & Target Variable Exploration
* **Tabular Audit:** Confirms zero full-row duplicates and zero explicit `NaN` values across the 100,000 rows and 18 attributes.
* **Target Distribution:** Quantifies the three-class astronomical distribution:
  * `GALAXY`: 59,445 samples (**59.45%**) — Majority Class
  * `STAR`: 21,592 samples (**21.59%**) — Intermediate Class
  * `QSO` (Quasar): 18,963 samples (**18.96%**) — Minority Class
* **MLOps Directive:** Demonstrates the **Accuracy Fallacy** (a naive majority classifier scores 59.4% accuracy with zero skill on Quasars and Stars). Formally establishes **Macro F1-Score**, **Balanced Accuracy**, and **Matthews Correlation Coefficient (MCC)** as mandatory evaluation metrics.
* **Diagnostic Figure:** Automatically saves a publication-grade class distribution plot to `reports/figures/target_distr.png`.

### Section 3: Instrumental Metadata & Leakage Prevention
Systematically examines all non-astrophysical metadata across four sub-modules:

* **3.1 Observation Identifiers (`obj_ID`, `spec_obj_ID`):**
  * Evaluates 78,053 unique `obj_ID` sources across 100,000 observations (15,359 re-observed objects).
  * **Zero Cross-Field Leakage Proof:** Mathematically proves that `max_fields_per_obj == 1`. Because the SDSS 64-bit `obj_ID` bit-encodes the `field_ID`, duplicate re-observations of an astronomical source **never cross different sky fields**.
  * **Label Discordance Discovery:** Identifies that ~37.5% of re-observed objects have conflicting class labels across exposures due to observation-level signal-to-noise variations and physical AGN variability. Confirms that each row is an individual spectroscopic event (`spec_obj_ID`), not a de-duplicated catalog entry.
  * *Verdict:* Drop both identifiers from the feature matrix $X$.
* **3.2 Telescope Hardware & Survey Parameters (`rerun_ID`, `run_ID`, `cam_col`, `plate`, `fiber_ID`):**
  * `rerun_ID`: Zero-variance constant (`301` across all 100k rows) $\to$ Drop.
  * `run_ID`: 430 discrete scanning strips. Chi-Square contingency test against `class` rejects independence ($p \ll 0.05$) due to observational scheduling bias (pointing at stellar fields vs extragalactic survey areas on specific nights). Retaining it would lead to survey calendar memorization $\to$ Drop.
  * `cam_col`: Uniform distribution across 6 CCD columns on the telescope focal plane. Instrument geometry without astrophysical signal $\to$ Drop.
  * `plate` & `fiber_ID`: Mechanical plug-plate (6,284 IDs) and fiber routing (1,000 IDs) hardware. Massive high-cardinality noise $\to$ Drop.
* **3.3 Spatial & Temporal Coordinates (`alpha`, `delta`, `MJD`):**
  * `MJD` (Modified Julian Date): Multi-modal distribution mapping historical survey generations (SDSS-I/II, BOSS, eBOSS). Class densities vary by campaign era $\to$ Drop to prevent temporal shortcut learning.
  * `alpha` (Right Ascension) & `delta` (Declination): Equatorial coordinates show geometric scanning stripes and distinct spatial clustering by celestial class. Retaining coordinates causes the model to memorize local sky patches rather than universal physics $\to$ Drop.
* **3.4 Spatial Grouping Strategy & Validation Anchor (`field_ID`):**
  * Audits 856 discrete sky fields (averaging ~117 observations per tile).
  * Chi-Square test of independence strongly rejects spatial independence ($p \ll 0.05$).
  * *Architectural Validation Blueprint:* Formally establishes **`GroupShuffleSplit`** (for holdout splitting) and **`GroupKFold`** (for cross-validation) grouped strictly on `field_ID`, guaranteeing zero identity leakage and zero spatial leakage.

### Section 4: Photometric Flux Analysis & Anomaly Detection
* **Physical System:** Evaluates the 5 SDSS optical bands ($u, g, r, i, z$) on the asinh magnitude scale.
* **Sentinel Value Discovery:** Uncovers that exactly one observation (row index 79543, labeled `STAR`) contains the non-physical sentinel `-9999` across filters `u`, `g`, and `z` ($0.001\%$ of dataset). Sanitizes occurrences by replacing with `np.nan`.
* **Statistical Metrics:** Consolidates central tendency, spread, skewness ($0.1 - 0.7$), and kurtosis across all 5 bands in a unified summary DataFrame.
* **Class-Conditional Densities:** Compares 5-panel KDE distributions, showing narrow QSO distributions, bimodal galaxy distributions, and bright stellar tails.
* **Multicollinearity Discovery:** Pearson correlation heatmap reveals extreme collinearity among adjacent filters ($r > 0.85$ between $u-g$; $r > 0.95$ across $r, i, z$), demonstrating that raw flux magnitudes provide redundant luminosity information while obscuring spectral curvature.

### Section 5: Kinematics & Redshift Analysis
* **Physical Formulation:** Redshift ($z$) measures spectral line displacement from cosmological expansion and Doppler kinematics.
* **The Doppler Blueshift Discovery:** Explains that 13,724 observations ($13.7\%$) have slightly negative redshifts (down to $z \approx -0.01$). Crucially, **$99.7\%$ of negative values belong exclusively to Galactic stars**, reflecting true physical Doppler blueshift (stars moving toward the Solar System within the Milky Way).
* **Cosmological Regimes:** 
  * `STAR`: Strictly local kinematics centered at $z \approx 0.0000$ (median $0.0000$, $\sigma \approx 0.0005$).
  * `GALAXY`: Bimodal peaks at $z \approx 0.08$ and $z \approx 0.55$, tapering off by $z \approx 0.8$.
  * `QSO`: Cosmological active galactic nuclei centered at $z \approx 1.4$, extending past $z > 7.0$.
* **Dominant Feature Diagnostic:** Univariate Quasar ROC-AUC reaches **$0.98$**, and baseline Random Forest attributes **$> 65\%$** of total feature importance to `redshift`. Warns against the "Dominant Feature Syndrome" where models degenerate into redshift thresholders.

### Section 6: Astrophysical Feature Engineering (Color Indices & Color-Color Space)
* **Astrophysical Principle:** Taking magnitude differences:
  $$(m_1 - m_2) = -2.5 \log_{10}\left(\frac{F_1}{F_2}\right)$$
  Cancels out distance and absolute luminosity, isolating pure flux ratios (spectral slope, effective temperature, and chemical absorption features).
* **Engineered Features:** Generates four adjacent color indices:
  * `u - g`: Ultraviolet excess (Quasars and hot stars).
  * `g - r`: Optical continuum slope (galaxy morphology).
  * `r - i`: Red-to-NIR slope (4000 Å break).
  * `i - z`: Near-infrared continuum (cool stellar atmospheres).
* **Collinearity Relief:** Pairwise correlations drop from $> 0.95$ down to $|r| \le 0.45$.
* **Color-Color Diagrams:** Dual-panel scatter plots ($(u-g)$ vs $(g-r)$ and $(g-r)$ vs $(r-i)$) visually validate physical segregation into the stellar locus, galaxy nebular cloud, and Quasar UV-excess domain.
* **Optimized Feature Set:** A Random Forest trained on `['r', 'u-g', 'g-r', 'r-i', 'i-z', 'redshift']` confirms that color indices capture over **$35\%$** of model importance, while standalone luminosity (`r`) accounts for $< 5\%$.

### Section 7: Final MLOps Architectural Blueprint & Production Strategy
* **Consolidated Decision Matrix:** Tabulates all 18 variables with their category, action in $X$, technical justification, and production drift monitoring protocol.
* **Pipeline Verification:** Executes an end-to-end verification block confirming the shape, column composition, and absence of parasite features in the production-ready feature matrix:
  ```python
  final_features = ["r", "u-g", "g-r", "r-i", "i-z", "redshift"]
  ```

---

## The MLOps Bridge: From Notebook Discoveries to Production Code

Every key architectural design in `src/` directly implements a scientific conclusion from `EDA.ipynb`:

| Discovery in `EDA.ipynb` | Physical / Statistical Rationale | Implementation in `src/` |
| :--- | :--- | :--- |
| **`max_fields_per_obj == 1` & Field $\chi^2$ Significance** | Re-observations never cross fields; classes cluster spatially. Random split causes severe leakage. | `src/data/make_dataset.py` uses `GroupShuffleSplit` on `field_ID` for holdout splitting. |
| **Extreme Multicollinearity ($r > 0.95$) in Raw Fluxes** | Raw magnitudes measure distance and size rather than intrinsic physical type. | Raw bands `u, g, i, z` are pruned from the training matrix after feature extraction. |
| **Color Indices Capture $> 35\%$ Importance** | Magnitude differences isolate spectral slope and effective temperature. | `src/features/build_features.py` defines `AstroFeatureEngineer(BaseEstimator, TransformerMixin)` to generate `['u-g', 'g-r', 'r-i', 'i-z']`. |
| **Single Sentinel Anomaly at Row 79543** | Pipeline extraction failures encode as `-9999`. Distorts variances and tree splits. | `src/data/clean_data.py` replaces sentinels with `np.nan` and applies median imputation. |
| **Redshift Dominance & Campaign Shifts** | `redshift` accounts for $55\%$ importance; historical campaigns shift class distributions. | `src/monitoring/` deploys automated Kolmogorov-Smirnov test alerts for `redshift` drift and $\chi^2$ tests for `class` drift. |

---

## Execution Guide & Environment Setup

### 1. Prerequisites
Ensure the virtual environment is active and dependencies are installed:
```bash
# From repository root
source venv/bin/activate
pip install -r requirements.txt
```

Verify that the raw dataset exists in `data/raw/`:
```bash
python main.py --mode make-datasets
```

### 2. Launching the Notebook
Launch Jupyter Lab or VS Code Jupyter:
```bash
jupyter lab notebook/EDA.ipynb
```
Select the `./venv/bin/python` kernel.

### 3. Execution Verification
To execute the entire notebook headlessly from the CLI and verify complete execution integrity:
```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebook/EDA.ipynb
```

---

## Governance, Reproducibility & Git Hygiene

To prevent notebook decay and maintain auditability across the team:

1. **Clear Execution State Before Committing (or Commit Fully Executed):**
   * If committing executed outputs, ensure all 78 cells have run sequentially from cell 0 to cell 77 with clean execution counts.
   * Never commit notebooks stopped midway with tracebacks or dangling partial execution states.
2. **Zero Scratchpad Cells:**
   * Ad-hoc code testing, variable inspections, or dictionary experiments must occur in local scratch files (or `sys.path` scratchpads) and never be committed to `EDA.ipynb`.
3. **No Heavy In-Memory Arrays in Version Control:**
   * Do not serialize massive binary objects or multi-gigabyte DataFrames into the notebook JSON. Save figures to `reports/figures/` via `plt.savefig()` instead.
4. **Strict Typing & Module Imports:**
   * All production transformations tested in the notebook must match the class signatures and typing standards implemented in `src/features/` and `src/utils/validate_type.py`.
