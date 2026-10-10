# Data Architecture & Storage Layer (`data/`)

This directory implements the multi-tier **Data Lakehouse & Storage Architecture** for the **Astro Object Classification** system. Built according to modern MLOps principles, it enforces deterministic data lineage, strict temporal segregation between training and production simulation, schema validation, and cryptographic tracking across all stages.

---

## Table of Contents

1. [Architecture Overview & Data Flow](#architecture-overview--data-flow)
2. [Directory Structure](#directory-structure)
3. [Tiers & Lifecycle Stages](#tiers--lifecycle-stages)
   - [3.1 Raw Storage Layer (`data/raw/`)](#31-raw-storage-layer-dataraw)
   - [3.2 Interim Partitioning Layer (`data/interim/`)](#32-interim-partitioning-layer-datainterim)
   - [3.3 Processed Training Layer (`data/processed/`)](#33-processed-training-layer-dataprocessed)
   - [3.4 Production Simulation Layer (`data/production/`)](#34-production-simulation-layer-dataproduction)
4. [Metadata & Cryptographic Lineage](#metadata--cryptographic-lineage)
5. [Programmatic Path Resolution (`DataPathConfig`)](#programmatic-path-resolution-datapathconfig)
6. [Data Pipeline Execution Guide (How to Populate Data)](#data-pipeline-execution-guide-how-to-populate-data)
7. [Repository State (Git) vs Runtime State](#repository-state-git-vs-runtime-state)
8. [Astronomical Domain Context](#astronomical-domain-context)
9. [Governance & Git Hygiene](#governance--git-hygiene)

---

## Architecture Overview & Data Flow

The data subsystem follows a staged progressive refinement pattern, separating immutable source data, partitioned datasets, feature-engineered training data, and temporal streaming simulation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. RAW STORAGE LAYER                            │
│               Kaggle SDSS DR17 Photometric Ingestion                   │
│                     data/raw/Star_Classification.csv                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ run_split() driven by configs/data.yaml
                                    │ (split_dataset_holdout.holdout_split: 0.2)
                                    ▼
         ┌──────────────────────────┴────────────────────────────┐
         │ 80% Training Base (1 - holdout_split)                 │ 20% Holdout Simulation (holdout_split)
         ▼                                                       ▼
┌───────────────────────────────────┐           ┌───────────────────────────────────┐
│     2. INTERIM LAYER              │           │   4. PRODUCTION SIMULATION LAYER  │
│  data/interim/                    │           │  data/production/                 │
│  training_dataset.csv             │           │  holdout_dataset.csv              │
└────────────────┬──────────────────┘           └─────────────────┬─────────────────┘
                 │ run_cleaning() via configs/preprocessing.yaml  │ daily batch slicing via
                 │ (anomaly_values, new_features)                 │ daily_split_batch_size: 1000
                 ▼                                                ▼
┌───────────────────────────────────┐            ┌───────────────────────────────────┐
│     3. PROCESSED LAYER            │            │  • daily_batch/ (day_1..day_N)    │
│  data/processed/                  │            │  • archived_batches/ (post-retrain│
│  processed_data.csv               │            │  • predictions/ (inference & perf)│
│  (Cleaned + Astro Color Features) │            └───────────────────────────────────┘
└───────────────────────────────────┘
```

### Core Design Principles

- **Declarative Configuration Authority:** Partitioning ratios (e.g., the 20% holdout vs 80% training split), batch sizes, sentinel thresholds, and feature derivations are **never hardcoded in Python execution scripts**. All responsibilities are governed by declarative YAML configuration files (`configs/data.yaml`, `configs/preprocessing.yaml`) and loaded via immutable dataclasses (`SplitHoldoutConfig`, `CleanPreprocessingConfig`).
- **Immutability (WORM - Write Once, Read Many):** Raw data ingested from external providers is immutable and never modified in place.
- **Leakage Prevention:** Segregation between the offline training corpus and the production simulation holdout occurs **before** any feature engineering or data cleaning.
- **Chronological Validity:** Partitioning respects the temporal ordering of astronomical observations via Modified Julian Date (`MJD`), simulating real-world survey streaming.
- **Cryptographic Auditability:** Every processing step computes MD5 checksums, row counters, and schema validations preserved in accompanying `.json` metadata sidecars.

---

## Directory Structure

```bash
data/
├── README.md                                 # Subsystem documentation & architecture
│
├── raw/                                      # Tier 1: Ingested source datasets
│   ├── Star_Classification.csv               # Immutable raw SDSS DR17 catalog (100k rows)
│   └── Star_Classification_metadata.json     # Ingestion audit log (MD5, schema, row count)
│
├── interim/                                  # Tier 2: Partitioned intermediate data
│   ├── training_dataset.csv                  # Base training split (80% / 80,000 observations)
│   └── interim_dataset_metadata.json         # Split parameters, ratio, and verification hashes
│
├── processed/                                # Tier 3: Feature-engineered model inputs
│   ├── processed_data.csv                    # Cleaned features + derived astronomical colors
│   └── processed_data_metadata.json          # Transformation lineage & feature catalog
│
└── production/                               # Tier 4: Streaming & monitoring simulation
    ├── holdout_dataset.csv                   # Production simulation holdout (20% / 20k rows)
    ├── holdout_dataset_metadata.json         # Holdout partition provenance metadata
    │
    ├── daily_batch/                          # Active chronological streaming batches
    │   ├── daily_batch_metadata.json         # Current batch queue state & temporal cursor
    │   ├── day_1.csv ... day_N.csv           # Chronological streaming batches (e.g. 1,000 rows/day)
    │   └── .gitkeep
    │
    ├── archived_batches/                     # Post-retraining archive store
    │   ├── archived_batches_metadata.json    # Retraining consumption log & archived index
    │   ├── day_*.csv                         # Batches migrated here dynamically after retraining triggers
    │   └── .gitkeep
    │
    └── predictions/                          # Production simulation inference sink
        ├── holdout_pred.csv                  # Full holdout model predictions
        └── holdout_pred_metrics.json         # Global holdout performance evaluation
```

---

## Tiers & Lifecycle Stages

### 3.1 Raw Storage Layer (`data/raw/`)

- **Source:** Sloan Digital Sky Survey Data Release 17 (SDSS DR17), ingested via Kaggle API (`fedesoriano/stellar-classification-dataset-sdss17`).
- **Volume:** 100,000 observations, 18 features (spectral bands, telescope metadata, coordinates, target).
- **Target:** Multi-class classification label (`class`: `GALAXY`, `STAR`, `QSO`).
- **Access Rule:** Pure read-only for all training and pipeline components.

**Metadata Contract (`Star_Classification_metadata.json`):**
```json
{
  "source": "fedesoriano/stellar-classification-dataset-sdss17",
  "filename": "Star_Classification.csv",
  "timestamp": "2026-08-15T10:30:00Z",
  "row_count": 100000,
  "column_count": 18,
  "md5_checksum": "a8f3b...e41d",
  "status": "VALIDATED"
}
```

---

### 3.2 Interim Partitioning Layer (`data/interim/`)

- **Component:** `src.data.holdout_split_data.SplitProductionSimulation`
- **Configuration Source of Truth:** Defined declaratively in [`configs/data.yaml`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/data.yaml) under the `split_dataset_holdout` section and deserialized into the strongly typed [`SplitHoldoutConfig`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/pipeline_config.py).
- **Governing YAML Parameters:**
  - `holdout_split: 0.2`: Explicitly allocates **20%** of the ingested data to production simulation (`holdout_dataset.csv`), leaving the remaining **80%** ($1 - \text{holdout\_split}$) for model training (`training_dataset.csv`).
  - `ref_col: "MJD"`: Designates the chronological sorting anchor (*Modified Julian Date*) to preserve temporal causality and prevent lookahead data leakage.
  - `daily_split_batch_size: 1000`: Sets the slice granularity for subsequent daily streaming batches.
- **Architectural Responsibility:**
  - Python execution scripts contain **zero hardcoded ratios or magic numbers**.
  - Any operational change to partition sizes or temporal sorting columns is enacted solely by modulating `configs/data.yaml` without modifying business logic code.
- **Mechanism:**
  - Loads `Star_Classification.csv` from `data/raw/`.
  - Sorts observations chronologically by `ref_col` (`MJD`).
  - Emits `training_dataset.csv` (the earliest 80%, ~80k rows) to `data/interim/`.
  - Emits `holdout_dataset.csv` (the subsequent 20%, ~20k rows) to `data/production/`.
- **Purpose:** Prevents temporal lookahead bias and guarantees an untouched simulation corpus for drift and degradation detection.

---

### 3.3 Processed Training Layer (`data/processed/`)

- **Component:** `src.data.preprocessing.ProcessedDataSaver` & `build_stateless_cleaning_pipeline`
- **Configuration Source of Truth:** Defined declaratively in [`configs/preprocessing.yaml`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/preprocessing.yaml) under `cleaning_preprocessing` and deserialized into [`CleanPreprocessingConfig`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/pipeline_config.py).
- **Governing YAML Parameters:**
  - `anomaly_values: [-9999]`: Explicitly declares erroneous photometric filter sentinels to be sanitized.
  - `new_features: [...]`: Declares the derived astronomical color index equations to compute ($\Delta(u-g), \Delta(g-r), \Delta(r-i), \Delta(i-z)$).
  - `columns_to_hold: ["field_ID", "class"]`: Preserves spatial cross-validation grouping identifiers and target labels without scaling or alteration.
- **Transformations Applied:**
  1. **Anomaly Sentinel Replacement:** Extreme erroneous photometric values replaced or sanitized.
  2. **Astronomical Color Derivation:** Computes relative photometric color indices to capture stellar temperatures and galactic spectral energy distributions.
  3. **Spatial Group Preservation:** Retains `field_ID` for **Stratified Group K-Fold (`SGKF`)** cross-validation, guaranteeing telescope field disjointness across CV folds.
  4. **Target Isolation:** Preserves target label `class` for model training pipelines.

---

### 3.4 Production Simulation Layer (`data/production/`)

This layer emulates an operational telescope survey environment where new data arrives over time, with all operational parameters governed by configuration:

#### Active Streaming Queue (`daily_batch/`)
- Slices the production holdout into sequential batches ordered by `MJD`.
- **Governing Parameter:** `daily_split_batch_size: 1000` from [`configs/data.yaml`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/data.yaml).
- Used by `src.predict.batch_predict` and `src.monitoring` to simulate daily inference workloads.
- Allows testing data drift detectors (Kolmogorov-Smirnov test, Chi-Square class balance shift) under realistic production pacing.

#### Archival & Retraining Store (`archived_batches/`)
- Managed by `src.data.retraining_data_collector` and `src.monitoring.retraining_policy`.
- **Governing Parameters:** Thresholds, drop tolerances, and cooldown windows (`cooldown_batches: 5`) defined in [`configs/monitoring.yaml`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/monitoring.yaml).
- When consecutive performance drops or critical data drift triggers a model retraining event:
  1. The batches that triggered the event are consumed.
  2. Batches are moved from `daily_batch/` to `archived_batches/`.
  3. The retrained model is evaluated against historical benchmarks, respecting the cooldown period (`cooldown_batches`).

#### Predictions & Performance Sink (`predictions/`)
- Stores complete inference outputs (`holdout_pred.csv`) and calculated metrics (`holdout_pred_metrics.json`) for auditability, regression analysis, and MLflow logging.

---

## Metadata & Cryptographic Lineage

To satisfy enterprise MLOps compliance and governance requirements, every artifact in `data/` is accompanied by structured metadata sidecars.

### Tracking Integration with MLflow

During execution of `make_datasets.py`, data provenance is logged directly into the centralized MLflow experiment (`Astro_Object_Classification`):

```python
mlflow.set_tags({
    "raw_dataset_md5": file_md5(path_config.raw_data_path),
    "training_dataset_md5": file_md5(path_config.split_training_path),
    "processed_dataset_md5": file_md5(path_config.processed_data),
    "holdout_dataset_md5": file_md5(path_config.split_production_path),
})
```

If an upstream CSV file is modified, its MD5 signature changes, allowing downstream validation pipelines to flag hash discrepancies immediately.

---

## Programmatic Path Resolution (`DataPathConfig`)

Paths inside `data/` must **never be hardcoded** as string literals in application code. All operations bind through `DataPathConfig` from `configs.paths`:

```python
from configs.paths import DataPathConfig
import pandas as pd

paths = DataPathConfig()

# Reading raw data
df_raw = pd.read_csv(paths.raw_data_path)

# Reading intermediate partitioned training split
df_train = pd.read_csv(paths.split_training_path)

# Reading preprocessed, feature-engineered training data
df_processed = pd.read_csv(paths.processed_data)

# Accessing streaming daily batch directory
batch_dir = paths.split_daily_batch_path
```

### Complete Attribute Matrix

| Category | `DataPathConfig` Attribute | Target Path |
|---|---|---|
| **Raw** | `raw_data_dir` | `data/raw` |
| **Raw** | `raw_data_path` | `data/raw/Star_Classification.csv` |
| **Raw** | `raw_data_metadata` | `data/raw/Star_Classification_metadata.json` |
| **Interim** | `interim_dir` | `data/interim` |
| **Interim** | `split_training_path` | `data/interim/training_dataset.csv` |
| **Interim** | `split_interim_metadata_path` | `data/interim/interim_dataset_metadata.json` |
| **Processed** | `processed_dir` | `data/processed` |
| **Processed** | `processed_data` | `data/processed/processed_data.csv` |
| **Processed** | `processed_data_metadata` | `data/processed/processed_data_metadata.json` |
| **Production** | `production_dir` | `data/production` |
| **Production** | `split_production_path` | `data/production/holdout_dataset.csv` |
| **Production** | `split_production_metadata_path` | `data/production/holdout_dataset_metadata.json` |
| **Production** | `split_daily_batch_path` | `data/production/daily_batch` |
| **Production** | `split_daily_batch_metadata_path` | `data/production/daily_batch/daily_batch_metadata.json` |
| **Production** | `split_archived_batch_path` | `data/production/archived_batches` |
| **Production** | `split_archived_batch_metadata_path`| `data/production/archived_batches/archived_batches_metadata.json` |
| **Production** | `holdout_dataset_full_predicted` | `data/production/predictions/holdout_pred.csv` |
| **Production** | `holdout_dataset_full_metrics` | `data/production/predictions/holdout_pred_metrics.json` |

---

## Data Pipeline Execution Guide (How to Populate Data)

When cloning this repository fresh from GitHub, the `data/` subdirectories exist as skeletal structures preserved via `.gitkeep` files, without storing heavy dataset files. To bootstrap and populate the entire data lakehouse from scratch, follow these instructions:

### Prerequisites: Kaggle API Credentials

The raw SDSS dataset is ingested automatically using the Kaggle API. Ensure you have your Kaggle credentials configured locally:

```bash
# Option 1: Kaggle API Token file
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Option 2: Environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### Automated End-to-End Pipeline Execution

To trigger the end-to-end data creation process, execute the pipeline orchestrator from the project root:

```bash
python main.py --mode make-datasets
```

*(Alternatively, run the dataset module directly: `python -m src.data.make_datasets`)*

### What Happens Behind the Scenes:

1. **Automated Ingestion (`src.data.ingestion`):** Downloads `Star_Classification.csv` from Kaggle into `data/raw/` and records `Star_Classification_metadata.json` with MD5 checksums.
2. **Chronological Holdout Partitioning (`src.data.holdout_split_data`):** Reads [`configs/data.yaml`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/data.yaml) (`holdout_split: 0.2`, `ref_col: "MJD"`), sorts chronologically, and emits:
   - `data/interim/training_dataset.csv` (earliest 80%)
   - `data/production/holdout_dataset.csv` (latest 20%)
3. **Stateless Cleaning & Feature Engineering (`src.data.preprocessing`):** Reads [`configs/preprocessing.yaml`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/preprocessing.yaml), replaces `-9999` sentinels, derives photometric color indices ($\Delta(u-g), \Delta(g-r), \Delta(r-i), \Delta(i-z)$), serializes `models/cleaner pipeline/cleaner_pipeline.pkl`, and writes `data/processed/processed_data.csv`.
4. **Production Slicing (`src.data.holdout_split_data`):** Slices the 20% holdout into sequential batches of 1,000 observations each, populating `data/production/daily_batch/day_1.csv` ... `day_N.csv`.
5. **Cryptographic Lineage & MLflow Tracking:** Logs all dataset hashes, byte sizes, and paths into MLflow Tracking under the experiment `Astro_Object_Classification` (run: `Cleaner_pipeline`).

---

## Repository State (Git) vs Runtime State

In accordance with MLOps best practices, tabular data files and transient model artifacts are decoupled from Git version control.

| Resource / Path | State in Git Repository | State at Runtime (After `make-datasets`) | Rationale |
|---|---|---|---|
| `data/raw/` | Empty directory with `.gitkeep` | `Star_Classification.csv` + metadata | Source data fetched on-demand via Kaggle API |
| `data/interim/` | Empty directory with `.gitkeep` | `training_dataset.csv` + metadata | Intermediate split generated deterministically |
| `data/processed/` | Empty directory with `.gitkeep` | `processed_data.csv` + metadata | Cleaned training dataset with engineered features |
| `data/production/daily_batch/` | Empty directory with `.gitkeep` | `day_1.csv` ... `day_N.csv` | Chronological streaming simulation batches |
| `data/production/archived_batches/` | Empty directory with `.gitkeep` | Batches retired after retraining triggers | Dynamic state managed by monitoring policy |
| `configs/*.yaml` | **Tracked in Git** | **Tracked in Git** | Declarative configuration is the Single Source of Truth |
| `configs/paths.py` | **Tracked in Git** | **Tracked in Git** | Immutable path resolution engine |

> [!TIP]
> This separation ensures that the Git repository remains lightweight (clonable in seconds), while any developer or CI/CD runner can deterministically reproduce the exact same data artifacts by executing `python main.py --mode make-datasets`.

---

## Astronomical Domain Context

The dataset originates from the **Sloan Digital Sky Survey (SDSS)**. Key domain-specific features and their roles across the data layer:

| Column | Description | Pipeline Role |
|---|---|---|
| `obj_ID` | Unique SDSS photometric identifier | Identification only (dropped during modeling) |
| `alpha`, `delta` | Celestial coordinates (Right Ascension $\alpha$, Declination $\delta$) | Spatial positioning |
| `u`, `g`, `r`, `i`, `z` | Ultraviolet, Green, Red, Infrared, Near-Infrared photometric filter bands | Core spectral features; used to compute astronomical color differences |
| `redshift` | Spectroscopic cosmological redshift ($z$) | High-importance discriminator between stars ($z \approx 0$), galaxies ($z < 1$), and distant quasars ($z > 1$) |
| `field_ID` | Telescope scan field identifier | **Grouping variable** for Stratified Group K-Fold cross-validation to prevent spatial observation leakage |
| `MJD` | Modified Julian Date of observation | **Temporal sorting variable** used for production holdout simulation split |
| `class` | Target label (`GALAXY`, `STAR`, `QSO`) | Classification target (supervised learning) |

---

## Governance & Git Hygiene

- **Version Control Policy:** Large tabular artifacts (`*.csv`), binary serializations, and streaming batches should be tracked via [DVC (Data Version Control)](https://dvc.org/) or external object storage (e.g., S3/GCS) in high-scale environments.
- **Repository Cleanliness:** Local runtime files, generated predictions, and transient batches in `daily_batch/` are excluded by `.gitignore`, while folder structures are preserved in Git using `.gitkeep`.
- **Integrity Validation:** Before starting model training or HPO sweeps, the test suite (`tests/unit/configs/test_paths.py` and `tests/unit/data/`) validates that all critical directories, suffixes, and metadata schemas exist and conform to domain invariants.
