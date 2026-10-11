# Analytics, Model Evaluation & Monitoring Reports (`reports/`)

This directory houses the analytical artifacts, diagnostic figures, model evaluation metrics, and drift monitoring reports for the **Astro Object Classification** system. Built according to MLOps auditability standards, it provides a transparent, versioned record of model performance, comparative baselines, exploratory analyses, and automated retraining triggers.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Subsystems Breakdown](#subsystems-breakdown)
   - [3.1 Best Model Evaluation (`reports/best_model/`)](#31-best-model-evaluation-reportsbest_model)
   - [3.2 Baseline & Custom Model Evaluation (`reports/custom_model/`)](#32-baseline--custom-model-evaluation-reportscustom_model)
   - [3.3 Exploratory Figures (`reports/figures/`)](#33-exploratory-figures-reportsfigures)
   - [3.4 Drift & Retraining Reports (`reports/monitoring/`)](#34-drift--retraining-reports-reportsmonitoring)
4. [Programmatic Path Resolution (`DataPathConfig`)](#programmatic-path-resolution-datapathconfig)
5. [Generation & Execution Lifecycle](#generation--execution-lifecycle)
6. [Comparative Evaluation: Champion vs Baseline](#comparative-evaluation-champion-vs-baseline)
7. [MLOps Best Practices & Git Hygiene](#mlops-best-practices--git-hygiene)

---

## Architecture Overview

In an enterprise MLOps workflow, model weights (`models/`) are separated from analytical outputs (`reports/`). While model binaries are transient and frequently excluded from version control due to size, **evaluation reports and diagnostic plots serve as lightweight, permanent audit trails**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTION PIPELINES                              │
│         src/optimization/      │   src/training/   │    src/monitoring/     │
└─────────────────┬──────────────┴─────────┬─────────┴───────────┬────────────┘
                  │                        │                     │
                  ▼                        ▼                     ▼
┌──────────────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
│     reports/best_model/      │ │reports/custom_model│ │ reports/monitoring/ │
│  • Hyperparameters (JSON)    │ │• Hyperparams (JSON)│ │ • Drift Report(JSON)│
│  • Metrics (JSON)            │ │• Metrics (JSON)    │ │ • Retraining Policy │
│  • Diagnostic Figures (PNG)  │ │                    │ │                     │
└──────────────────────────────┘ └────────────────────┘ └─────────────────────┘
```

* **Auditability & Traceability:** Every hyperparameter optimization (HPO) campaign and training run produces machine-readable JSON metrics and human-readable diagnostic plots.
* **Deterministic Reproducibility:** Every metric is computed against validated test/holdout splits using standardized metric calculations (`src/utils/metrics.py`).
* **Automated Decision Feedback:** Continuous monitoring records data drift and performance drops into a persistent JSON report, enabling automated or human-in-the-loop retraining decisions.

---

## Directory Structure

```bash
reports/
├── README.md                                 # Subsystem documentation and schema reference
├── best_model/                               # Champion model artifacts (Optuna HPO outcome)
│   ├── .gitkeep                              # Structure tracking
│   ├── best_model_hyperparameters.json       # Optimal hyperparameters dictionary
│   ├── best_model_metrics.json               # Comprehensive multi-class test metrics
│   ├── confusion_matrix.png                  # Multi-class confusion matrix
│   ├── corr_matrix.png                       # Feature correlation heatmap
│   ├── feature_importance.png                # Top feature importances ranking
│   ├── target_test_distribution.png          # Target class frequencies in test split
│   └── target_train_distribution.png         # Target class frequencies in train split
├── custom_model/                             # Baseline / manual configuration benchmark
│   ├── .gitkeep                              # Structure tracking
│   ├── custom_model_hyperparameters.json     # Hyperparameters applied to the custom model
│   └── custom_model_metrics.json             # Baseline model test metrics
├── figures/                                  # Exploratory Data Analysis (EDA) visualizations
│   ├── .gitkeep                              # Structure tracking
│   ├── camcol_distr.png                      # SDSS camera column feature distribution
│   └── target_distr.png                      # Raw dataset target distribution (GALAXY, QSO, STAR)
└── monitoring/                               # Production drift & performance monitoring
    ├── .gitkeep                              # Structure tracking
    └── retraining_decision_report.json       # Batch-by-batch drift & retraining decisions
```

---

## Subsystems Breakdown

### 3.1 Best Model Evaluation (`reports/best_model/`)

Generated automatically by `src/optimization/search.py` after completing the Optuna Hyperparameter Optimization study and refitting the winning pipeline.

#### A. Hyperparameters Record (`best_model_hyperparameters.json`)
Stores the winning configuration discovered during HPO:
```json
{
    "n_estimators": 351,
    "max_depth": 28,
    "min_samples_split": 8,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "criterion": "entropy",
    "bootstrap": false,
    "class_weight": "balanced",
    "verbose": 0,
    "n_jobs": -1,
    "random_state": 42
}
```

#### B. Evaluation Metrics (`best_model_metrics.json`)
Stores comprehensive global and per-class metrics across all three astronomical classes (`GALAXY`, `QSO`, `STAR`), along with Matthews Correlation Coefficient (MCC):
```json
{
    "accuracy": 0.9848,
    "balance_accuracy": 0.9767,
    "f1_weighted": 0.9848,
    "f1_macro": 0.9784,
    "precision_weighted": 0.9848,
    "precision_macro": 0.9801,
    "recall_weighted": 0.9848,
    "recall_macro": 0.9767,
    "f1_GALAXY": 0.9880,
    "precision_GALAXY": 0.9869,
    "recall_GALAXY": 0.9892,
    "f1_QSO": 0.9496,
    "precision_QSO": 0.9580,
    "recall_QSO": 0.9413,
    "f1_STAR": 0.9975,
    "precision_STAR": 0.9954,
    "recall_STAR": 0.9996,
    "mcc": 0.9712,
    "model_name": "random_forest"
}
```

#### C. Diagnostic Figures Gallery
* **`confusion_matrix.png`:** Normalized matrix showing classification accuracy and confusion rates between Galaxies, Quasars (QSO), and Stars.
* **`feature_importance.png`:** Bar chart ranking the predictive contribution of engineered photometric features (such as `redshift`, extinction color indices $u-g$, $g-r$, $r-i$, $i-z$).
* **`corr_matrix.png`:** Heatmap visualizing pairwise Pearson correlations among input features to monitor multicollinearity.
* **`target_train_distribution.png` & `target_test_distribution.png`:** Class balance verification plots ensuring representative stratification across folds and splits.

---

### 3.2 Baseline & Custom Model Evaluation (`reports/custom_model/`)

Generated by `src/training/train.py`. Stores the hyperparameters and evaluation metrics of a manually configured baseline or alternative architecture (`custom_model_hyperparameters.json` and `custom_model_metrics.json`).

#### A. Hyperparameters Record (`custom_model_hyperparameters.json`)
Stores the user-selected or default configuration used to train the custom model:
```json
{
    "criterion": "gini",
    "model_name": "random_forest"
}
```
* Enforces complete configuration transparency, pairing the serialized model with its exact hyperparameters.

#### B. Evaluation Metrics (`custom_model_metrics.json`)
Stores comprehensive test metrics evaluated on the test split:
```json
{
    "accuracy": 0.9828,
    "balance_accuracy": 0.9743,
    "f1_weighted": 0.9827,
    "f1_macro": 0.9777,
    "precision_weighted": 0.9827,
    "precision_macro": 0.9813,
    "recall_weighted": 0.9828,
    "recall_macro": 0.9743,
    "f1_GALAXY": 0.9861,
    "precision_GALAXY": 0.9826,
    "recall_GALAXY": 0.9897,
    "f1_QSO": 0.9493,
    "precision_QSO": 0.9656,
    "recall_QSO": 0.9335,
    "f1_STAR": 0.9976,
    "precision_STAR": 0.9956,
    "recall_STAR": 0.9996,
    "mcc": 0.9683
}
```

* Provides a **direct benchmark** to quantify the marginal value added by Optuna HPO.
* Shares an identical JSON schema with `best_model_metrics.json` for automated differential comparisons.

---

### 3.3 Exploratory Figures (`reports/figures/`)

Contains static visualizations produced during initial Exploratory Data Analysis (`notebook/EDA.ipynb`):

* **`target_distr.png`:** High-level class balance of the astronomical objects across the Sloan Digital Sky Survey dataset (GALAXY ~59%, STAR ~21%, QSO ~19%).
* **`camcol_distr.png`:** Distribution across SDSS scan strips / camera columns (1 through 6) to detect telescope sensor bias.

---

### 3.4 Drift & Retraining Reports (`reports/monitoring/`)

Generated continuously or incrementally by `src/monitoring/monitoring_pipeline.py` through `AstroRetrainingPolicy` (`src/monitoring/retraining_policy.py`).

#### Retraining Decision Report (`retraining_decision_report.json`)
Tracks production batch evaluations sequentially (`day_1.csv` through `day_N.csv`). Each entry records a multi-dimensional assessment across **performance degradation**, **statistical feature/prediction drift**, **retraining cooldown constraints**, and the final **orchestration decision**.

##### Schema Specification (Real Production Output):

```json
{
    "day_1.csv": {
        "performance": {
            "triggered": false,
            "consecutive_critical_count": 1,
            "is_catastrophic": false,
            "reason": "🟡 f1_weighted critical: 1/3 (drop of 4.74%)"
        },
        "drift": {
            "triggered": true,
            "drift_ratio": 0.6667,
            "drifted_features_count": 4,
            "total_features_count": 6,
            "drifted_features": [
                "r",
                "redshift",
                "u-g",
                "i-z"
            ],
            "prediction_drift_detected": false,
            "target_drift_detected": false,
            "reason": "🔴 Feature drift detected: 4/6 features drifted (66.67% >= 50.00%) -> TRIGGERED"
        },
        "cooldown": {
            "in_cooldown": false,
            "batches_since_last_retraining": null,
            "remaining_cooldown_batches": 0,
            "last_retraining_batch": null,
            "reason": "🟢 Cooldown inactive"
        },
        "decision": {
            "retraining": false,
            "action": "NO_ACTION",
            "primary_trigger": null,
            "reason": "🟢 Performance healthy, but feature drift detected (monitoring)."
        }
    },
    "day_6.csv": {
        "performance": {
            "triggered": true,
            "consecutive_critical_count": 1,
            "is_catastrophic": true,
            "reason": "🔴🔴🔴 Catastrophic drop f1_weighted (12.02%) -> TRIGGERED"
        },
        "drift": {
            "triggered": true,
            "drift_ratio": 0.6667,
            "drifted_features_count": 4,
            "total_features_count": 6,
            "drifted_features": [
                "r",
                "redshift",
                "u-g",
                "g-r"
            ],
            "prediction_drift_detected": false,
            "target_drift_detected": false,
            "reason": "🔴 Feature drift detected: 4/6 features drifted (66.67% >= 50.00%) -> TRIGGERED"
        },
        "cooldown": {
            "in_cooldown": false,
            "batches_since_last_retraining": null,
            "remaining_cooldown_batches": 0,
            "last_retraining_batch": null,
            "reason": "🟢 Cooldown inactive"
        },
        "decision": {
            "retraining": true,
            "action": "RETRAINING",
            "primary_trigger": "PERFORMANCE_CATASTROPHIC",
            "reason": "🔴 Catastrophic drop in performance, retraining required."
        }
    }
}
```

##### Field Breakdown & Decision Logic:

* **`performance`:**
  * `triggered` (*bool*): True when performance degradation criteria are met.
  * `consecutive_critical_count` (*int*): Accumulator tracking consecutive batches with metric degradation below threshold (triggers after configured tolerance count).
  * `is_catastrophic` (*bool*): Immediate trigger flag if a single batch experiences severe degradation (e.g. $>12\%$ relative drop in weighted F1).
  * `reason` (*str*): Diagnostic alert with status indicators (`🟢`, `🟡`, `🔴`).
* **`drift`:**
  * `triggered` (*bool*): True when the proportion of drifted features exceeds the tolerance ratio (`drift_ratio >= 0.50`).
  * `drift_ratio` (*float*): Fraction of input features exhibiting statistical distribution drift (`drifted_features_count / total_features_count`).
  * `drifted_features_count` (*int*) & `total_features_count` (*int*): Count of drifted vs total monitored features.
  * `drifted_features` (*list[str]*): Specific input columns exhibiting statistical distribution shifts (Kolmogorov-Smirnov test).
  * `prediction_drift_detected` (*bool*): Statistical distribution shift in model predictions $\hat{y}$.
  * `target_drift_detected` (*bool*): Distribution shift in ground-truth labels $y$ (when labels are provided).
  * `reason` (*str*): Diagnostic summary of feature drift.
* **`cooldown`:**
  * `in_cooldown` (*bool*): Prevents repeated cascading retraining loops immediately after a retraining event.
  * `batches_since_last_retraining` (*int | null*): Number of production batches evaluated since the last retraining.
  * `remaining_cooldown_batches` (*int*): Number of subsequent batches that must pass before retraining is re-enabled.
  * `last_retraining_batch` (*str | null*): Identifier of the batch that triggered the previous retraining.
* **`decision`:**
  * `retraining` (*bool*): Final boolean execution signal consumed by the orchestrator/scheduler.
  * `action` (*str*): Action code (`NO_ACTION`, `RETRAINING`).
  * `primary_trigger` (*str | null*): Specific trigger root cause (e.g. `PERFORMANCE_CATASTROPHIC`, `PERFORMANCE_CRITICAL`, `DATA_DRIFT`).
  * `reason` (*str*): Operational summary justifying the action.

---

## Programmatic Path Resolution (`DataPathConfig`)

All paths within `reports/` are strongly typed and resolved relative to `PROJECT_ROOT` via `configs.paths.DataPathConfig`:

```python
from configs.paths import DataPathConfig

paths = DataPathConfig()

# Best Model Paths
print(paths.best_model)                       # /path/to/project/reports/best_model
print(paths.best_model_metrics)               # /path/to/project/reports/best_model/best_model_metrics.json
print(paths.best_model_hyperparameters)       # /path/to/project/reports/best_model/best_model_hyperparameters.json

# Custom Model Paths
print(paths.custom_model_dir)                 # /path/to/project/reports/custom_model
print(paths.custom_model_metrics)             # /path/to/project/reports/custom_model/custom_model_metrics.json
print(paths.custom_model_hyperparameters)     # /path/to/project/reports/custom_model/custom_model_hyperparameters.json

# Monitoring Reports
print(paths.monitoring_reports_dir)           # /path/to/project/reports/monitoring
print(paths.retraining_decision_report)       # /path/to/project/reports/monitoring/retraining_decision_report.json
```

---

## Generation & Execution Lifecycle

The following matrix describes how each report artifact is produced:

| Report Artifact | Source Script / Pipeline | Trigger CLI Command | Output Target |
| :--- | :--- | :--- | :--- |
| **`best_model_hyperparameters.json`** | `src/optimization/search.py` | `python main.py --mode optimize` | `reports/best_model/` |
| **`best_model_metrics.json`** | `src/optimization/search.py` | `python main.py --mode optimize` | `reports/best_model/` |
| **Diagnostic PNGs** (5 figures) | `src/optimization/search.py` | `python main.py --mode optimize` | `reports/best_model/` |
| **`custom_model_hyperparameters.json`** | `src/training/train.py` | `python main.py --mode train` | `reports/custom_model/` |
| **`custom_model_metrics.json`** | `src/training/train.py` | `python main.py --mode train` | `reports/custom_model/` |
| **`retraining_decision_report.json`** | `src/monitoring/monitoring_pipeline.py` | `python main.py --mode monitor` | `reports/monitoring/` |
| **EDA Plots** (`target_distr.png`, etc.) | `notebook/EDA.ipynb` | Manual execution | `reports/figures/` |

---

## Comparative Evaluation: Champion vs Baseline

By standardizing metric schemas across `best_model` and `custom_model`, the system enables automated differential performance audits:

| Metric | Champion (`best_model`) | Baseline (`custom_model`) | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.9848** | 0.9828 | +0.20% | Superior |
| **Balanced Accuracy** | **0.9767** | 0.9743 | +0.24% | Superior |
| **Macro F1-Score** | **0.9784** | 0.9777 | +0.07% | Superior |
| **MCC (Matthews Corr)** | **0.9712** | 0.9683 | +0.29% | Superior |
| **GALAXY F1-Score** | **0.9880** | 0.9861 | +0.19% | Superior |
| **QSO F1-Score (Minority)** | **0.9496** | 0.9493 | +0.03% | Superior |
| **STAR F1-Score** | 0.9975 | **0.9976** | -0.01% | Comparable |

> [!NOTE]
> The hyperparameter-optimized Random Forest model (`best_model`) consistently outperforms the baseline across almost all primary dimensions, yielding higher Matthews Correlation Coefficient (0.9712 vs 0.9683) and superior minority class separation.

---

## MLOps Best Practices & Git Hygiene

1. **Version Control Auditability:** Unlike heavy binary model weights (`models/**/*.pkl`), all files in `reports/` are lightweight (<250 KB total) and **intentionally tracked in Git**. This guarantees full historical visibility of model performance across git commits and pull requests.
2. **Directory Anchoring (`.gitkeep`):** Every subfolder contains a `.gitkeep` file so that fresh clones or CI/CD runner workspaces maintain the directory hierarchy before pipelines run.
3. **Immutability of Evaluation:** Reports should never be edited manually; they are generated deterministically by pipeline code using `DataPathConfig`.
