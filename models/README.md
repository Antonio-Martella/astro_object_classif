# Model Artifacts & Pipeline Registry (`models/`)

This directory serves as the local **Model Registry & Pipeline Storage Layer** for the **Astro Object Classification** system. It stores serialized, end-to-end Machine Learning pipelines, preprocessing transformers, and label encoders generated across the training, hyperparameter optimization, and data preparation lifecycles.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Model Subsystems Breakdown](#model-subsystems-breakdown)
   - [3.1 Champion Model Pipeline (`models/best_model/`)](#31-champion-model-pipeline-modelsbest_model)
   - [3.2 Baseline & Custom Model Pipeline (`models/custom_model/`)](#32-baseline--custom-model-pipeline-modelscustom_model)
   - [3.3 Data Cleaning & Transformation Pipeline (`models/cleaner pipeline/`)](#33-data-cleaning--transformation-pipeline-modelscleaner-pipeline)
   - [3.4 Target Label Encoder (`models/target_label_encoder/`)](#34-target-label-encoder-modelstarget_label_encoder)
4. [Programmatic Path Resolution (`DataPathConfig`)](#programmatic-path-resolution-datapathconfig)
5. [Serialization & Deserialization Protocol](#serialization--deserialization-protocol)
6. [Generation & Consumption Lifecycle Matrix](#generation--consumption-lifecycle-matrix)
7. [MLOps Best Practices & Git Hygiene](#mlops-best-practices--git-hygiene)

---

## Architecture Overview

In modern MLOps architectures, models are rarely serialized as standalone estimators. Instead, they are packaged as **composite pipelines** that bundle feature scalers, resampling strategies, and the estimator together into a single atomic artifact. This prevents **training-serving skew** and guarantees that transformations applied during training are identically replicated at inference time.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                    PIPELINE GENERATION                           │
│  src/data/preprocessing.py │ src/optimization/search.py │ src/training/train.py  │
└──────────────┬──────────────────────────────┬───────────────────────────┬────────┘
               │                              │                           │
               ▼                              ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────┐ ┌───────────────────────────┐
│   models/cleaner_pipeline/   │ │    models/best_model/    │ │ models/custom_model/      │
│   • cleaner_pipeline.pkl     │ │ • best_model_pipeline.pkl│ │• custom_model_pipeline.pkl│
│  (Data Cleaning & Anomalies) │ │ (Optuna Champion Model)  │ └───────────────────────────┘
└──────────────────────────────┘ └────────────┬─────────────┘ 
                                              │               
                                              ▼
                               ┌─────────────────────────────┐
                               │     INFERENCE & SERVING     │
                               │   src/predict/predictor.py  │
                               │   src/monitoring/           │
                               │   MLflow Model Registry     │
                               └─────────────────────────────┘
```

* **Atomic Serving:** Inference modules load a single `.pkl` pipeline that handles scaling, feature alignment, and prediction in a single call.
* **Separation of Concerns:** Heavy model binaries reside here in `models/`, completely separated from analytical audit reports (`reports/`) and configuration schemas (`configs/`).
* **Dual-Track Modeling:** The architecture maintains separate tracks for automated optimization (`best_model`) and interactive experimentation (`custom_model`).

---

## Directory Structure

```bash
models/
├── README.md                                  # Subsystem documentation and operational guide
├── best_model/                                # Champion model artifacts (Optuna HPO outcome)
│   ├── .gitkeep                               # Structure tracking in version control
│   └── best_model_pipeline.pkl                # Fully fitted champion pipeline (Scaler + Model)
├── custom_model/                              # Baseline / interactive training artifacts
│   ├── .gitkeep                               # Structure tracking in version control
│   └── custom_model_pipeline.pkl              # Fitted experimental pipeline
├── cleaner_pipeline/                          # Data cleaning & anomaly treatment pipeline
│   ├── .gitkeep                               # Structure tracking in version control
│   └── cleaner_pipeline.pkl                   # Fitted cleaning transformer
└── target_label_encoder/                      # Astronomical class encoder
    ├── .gitkeep                               # Structure tracking in version control
    └── target_label_encoder.pkl               # Fitted LabelEncoder ('GALAXY', 'QSO', 'STAR')
```

---

## Model Subsystems Breakdown

### 3.1 Champion Model Pipeline (`models/best_model/`)

Contains the **Champion Model** (`best_model_pipeline.pkl`), selected after completing the multi-trial Optuna Hyperparameter Optimization study runs via `python main.py --mode training`:
* **Artifact:** `best_model_pipeline.pkl` (~73 MB)
* **Pipeline Structure:** An `imblearn.pipeline.Pipeline` or `sklearn.pipeline.Pipeline` consisting of:
  1. **Resampling Step:** Optional balancing strategy (e.g., RandomUnderSampler or SMOTE).
  2. **Scaler Step:** RobustScaler or StandardScaler tuned to photometric features.
  3. **Classifier:** Optimized classifier (e.g., `RandomForestClassifier` with optimal depth, estimators, and criterion).
* **Serving Role:** Used by [`src/predict/predictor.py`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/src/predict/predictor.py) as the default production model for batch and real-time prediction, as well as the baseline for drift monitoring.
* **Companion Reports:** Paired 1:1 with [`reports/best_model/`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/reports/best_model) (`best_model_metrics.json`, `best_model_hyperparameters.json`, confusion matrix, and feature importances).

---

### 3.2 Baseline & Custom Model Pipeline (`models/custom_model/`)

Stores models produced during manual, interactive training runs via `python main.py --mode training`:

* **Artifact:** `custom_model_pipeline.pkl` (~22 MB)
* **Purpose:** Serves as a reference benchmark (Baseline) to evaluate whether Optuna optimization delivered meaningful gains over default or hand-crafted parameters.
* **Companion Reports:** Paired 1:1 with [`reports/custom_model/`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/reports/custom_model) (`custom_model_metrics.json` and `custom_model_hyperparameters.json`).

---

### 3.3 Data Cleaning & Transformation Pipeline (`models/cleaner_pipeline/`)

Stores the fitted data preprocessing and anomaly cleaning pipeline:

* **Artifact:** `cleaner_pipeline.pkl` (~567 B)
* **Fitted On:** Strictly on the training partition (`data/interim/training_dataset.csv`) to prevent **data leakage**.
* **Role:** Applies deterministic data sanitation:
  * Eliminates non-astronomical anomaly values (e.g., error codes such as `-9999.0` in SDSS photometric bands).
  * Feature selection and index alignment.
* **Consumption:** Ingested during dataset construction (`src/data/make_datasets.py`) and by raw batch ingestion pipelines.

---

### 3.4 Target Label Encoder (`models/target_label_encoder/`)

Stores the serialized target encoder mapping categorical astronomical classes to integer indices:

* **Artifact:** `target_label_encoder.pkl` (~494 B)
* **Class Mapping:**
  * `0` $\rightarrow$ `GALAXY`
  * `1` $\rightarrow$ `QSO` (Quasar)
  * `2` $\rightarrow$ `STAR`
* **Role:**
  * **Training Phase:** Encodes string target labels into numeric format for loss function computation.
  * **Inference Phase:** Inversely transforms model predictions ($\hat{y} \in \{0, 1, 2\}$) back into human-interpretable astronomical class names for client APIs and monitoring dashboards.

---

## Programmatic Path Resolution (`DataPathConfig`)

All artifact paths within `models/` are strongly typed and resolved relative to `PROJECT_ROOT` via [`configs.paths.DataPathConfig`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/paths.py):

```python
from configs.paths import DataPathConfig

paths = DataPathConfig()

# Champion Pipeline
print(paths.best_model_pipeline)      # .../models/best_model/best_model_pipeline.pkl

# Custom / Baseline Pipeline
print(paths.custom_model_pipeline)    # .../models/custom_model/custom_model_pipeline.pkl

# Preprocessing & Cleaner Pipeline
print(paths.cleaner_pipeline)         # .../models/cleaner pipeline/cleaner_pipeline.pkl

# Target Label Encoder
print(paths.target_le)                # .../models/target_label_encoder/target_label_encoder.pkl
```

---

## Serialization & Deserialization Protocol

All artifacts are persisted using Python's [`joblib`](https://joblib.readthedocs.io/) library, which provides efficient caching and compression for NumPy arrays and scikit-learn models:

```python
import joblib
import pandas as pd
from sklearn import set_config
from configs.paths import DataPathConfig

set_config(transform_output="pandas")

paths = DataPathConfig()

# 1. Loading the Champion model and the Label Encoder
pipeline = joblib.load(paths.best_model_pipeline)
label_encoder = joblib.load(paths.target_le)

# 2. Real photometric sample with the 6 features required by the model
# (SDSS observation: r-band magnitude, redshift, and color indices)
sample_observation = pd.DataFrame([{
    "r": 17.55,
    "redshift": 0.045,
    "u-g": 1.25,
    "g-r": 0.85,
    "r-i": 0.40,
    "i-z": 0.30,
}])

# 3. Prediction and decoding of the astronomical label
pred_encoded = pipeline.predict(sample_observation)
pred_class = label_encoder.inverse_transform(pred_encoded)

print(f"Predicted astronomical class: {pred_class[0]}")
# Expected output: Predicted astronomical class: GALAXY
```

---

## Generation & Consumption Lifecycle Matrix

| Artifact | Producer Script / Stage | Trigger Command | Primary Consumer |
| :--- | :--- | :--- | :--- |
| **`best_model_pipeline.pkl`** | `src/optimization/search.py` | `python main.py --mode optimize` | `src/predict/predictor.py`, MLflow Registry |
| **`custom_model_pipeline.pkl`** | `src/training/train.py` | `python main.py --mode training` | Comparative evaluation, MLflow Staging |
| **`cleaner_pipeline.pkl`** | `src/data/preprocessing.py` | `python main.py --mode make_dataset` | `src/data/make_datasets.py`, Data Pipelines |
| **`target_label_encoder.pkl`** | `src/training/data.py` | Training / Optimization runs | `src/predict/predictor.py`, `src/monitoring/` |

---

## MLOps Best Practices & Git Hygiene

1. **Exclusion of Binary Files from Git:**
   * Model binaries (`.pkl`) can reach tens of megabytes (e.g., 73 MB for random forest ensembles).
   * Storing large binary weights directly in Git causes repository bloat and exceeds GitHub recommendations.
   * All serialized weights are strictly ignored in [`.gitignore`](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/.gitignore):
     ```gitignore
     # models/
     models/**/*.pkl
     ```
2. **Directory Preservation via `.gitkeep`:**
   * Every subfolder (`best_model/`, `custom_model/`, `cleaner_pipeline/`, `target_label_encoder/`) contains a tracked `.gitkeep` file.
   * This guarantees that cloned repositories and CI/CD environments possess the complete directory hierarchy before pipeline execution.
3. **Model Versioning & Centralized Tracking (MLflow):**
   * Production versioning is handled outside Git via **MLflow Model Registry** (`mlflow.db`).
   * Models registered to MLflow are tracked with complete metadata, training metrics, parameters, and transition tags (`Staging` / `Production`).
