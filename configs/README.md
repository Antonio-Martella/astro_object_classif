# Configuration Management Architecture (`configs/`)

This directory provides the centralized, enterprise-grade configuration subsystem for the **Astro Object Classification** pipeline. Built upon MLOps best practices, it enforces a strict **Single Source of Truth (SSOT)**, complete separation between configuration and runtime execution, deterministic reproducibility, and strong type safety.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Domain Configuration Files (YAML)](#domain-configuration-files-yaml)
4. [Python Typed Schemas & Loaders](#python-typed-schemas--loaders)
5. [Path Resolution Engine (`paths.py`)](#path-resolution-engine-pathspy)
6. [Hyperparameter Optimization (`search_space_models/`)](#hyperparameter-optimization-search_space_models)
7. [Usage Examples](#usage-examples)
8. [Extension Guide: Adding New Models](#extension-guide-adding-new-models)
9. [Design Principles & MLOps Standards](#design-principles--mlops-standards)

---

## Architecture Overview

The configuration layer is organized into four distinct tiers:

```
┌──────────────────────────────────────────────────────────────────┐
│                   1. Declarative Domain YAMLs                    │
│   (data.yaml, preprocessing.yaml, monitoring.yaml, seeds.yaml)   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ parses via PyYAML
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│               2. Path Engine & Typed Data Schemas                │
│    (paths.py: DataPathConfig | pipeline_config.py: @dataclass)   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ unpacks & injects dependencies
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                 3. Typed Loaders & Accessors                     │
│  (pipeline_loader.py: pure loaders | optuna_config.py: HPO space)│
└────────────────────────────────┬─────────────────────────────────┘
                                 │ exports via __init__.py
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                4. Application & Execution Layer                  │
│       (src/data/, src/training/, src/optimization/, main.py)     │
└──────────────────────────────────────────────────────────────────┘
```

* **Zero Hardcoded Values:** Pipeline parameters, threshold values, column names, random seeds, and model hyperparameters reside strictly in YAML files.
* **Type Safety & Immutability:** All configurations are validated and deserialized into `@dataclass(frozen=True)` containers, preventing runtime side effects and mutations.
* **Reproducibility by Design:** Dedicated pseudo-random seeds are decoupled into a central seed registry (`random_seed.yaml`) and injected into downstream tasks.

---

## Directory Structure

```bash
configs/
├── README.md                    # Subsystem documentation and usage guide
├── __init__.py                  # Public API re-exporting schemas, loaders, and paths
├── data.yaml                    # Dataset acquisition and train/test partition specs
├── preprocessing.yaml           # Data cleaning, outlier filtering, and scaling targets
├── monitoring.yaml              # Drift detection, performance alerts, and retraining policy
├── random_seed.yaml             # Centralized pseudo-random seed definitions
├── optuna.yaml                  # Global HPO search space, sampler, and trial limits
├── paths.py                     # Central filesystem path resolver (DataPathConfig)
├── pipeline_config.py           # Strongly-typed immutable dataclass schema definitions
├── pipeline_loader.py           # Pure loader functions parsing YAMLs into dataclasses
├── optuna_config.py             # OptunaConfigLoader managing per-model search spaces
└── search_space_models/         # Standardized hyperparameter spaces per algorithm
    ├── defined_models.yaml      # Master catalog of supported model architectures
    ├── catboost.yaml            # CatBoost search space and fixed parameters
    ├── dense_nn.yaml            # Deep Neural Network (MLP) search space and callbacks
    ├── extra_trees.yaml         # Extra Trees classifier search space
    ├── lightgbm.yaml            # LightGBM classifier search space
    ├── logreg.yaml              # Logistic Regression (SAGA / ElasticNet) search space
    ├── random_forest.yaml       # Random Forest classifier search space
    ├── sgd.yaml                 # Stochastic Gradient Descent classifier search space
    ├── svc.yaml                 # Support Vector Classifier search space
    ├── xgboost.yaml             # XGBoost classifier search space
    └── README.md                # Documentation and usage guide for search space model
```

---

## Domain Configuration Files (YAML)

### 1. `data.yaml`
Defines dataset ingress, schema expectations, and data splitting ratios:
* **`kaggle`**: Remote dataset slug, filename, expected schema columns, and dataset row count.
* **`split_dataset_holdout`**: Parameters for generating the unobserved streaming simulation holdout set (`holdout_split: 0.2`, `daily_split_batch_size: 1000`).
* **`split_dataset_training`**: Stratified train/validation partitioning ratio (`train_split: 0.8`), grouping column (`col_group: plate`), and target variable (`target_column: class`).

### 2. `preprocessing.yaml`
Specifies data transformation rules and feature engineering boundaries:
* **`cleaning_preprocessing`**: Sentinel outlier values (`-9999.0` for missing photometric measurements), photometric error bounds, and engineered features (color indices: $u-g, g-r, r-i, i-z$).
* **`preprocessing`**: Designated continuous numerical features targeted for scaling (`columns_to_scale`).

### 3. `monitoring.yaml`
Defines operational production monitoring constraints:
* **`drift_monitoring`**: Kolmogorov-Smirnov statistical significance threshold (`ks_threshold: 0.30`), class distribution divergence (`max_shift_class: 0.30`), and feature drift alert ratio (`max_drift_ratio: 0.50`).
* **`performance_monitoring`**: Weighted F1 evaluation metric, absolute drop thresholds, and sectoral class decay limits.
* **`retraining_policy`**: Rules triggering automated model retraining (consecutive alert triggers, catastrophic drops, and cooldown batch counts).

### 4. `random_seed.yaml`
Centralized seed repository guaranteeing end-to-end determinism:
```yaml
global_seed:
  random_seed_train_split: 42
  random_seed_resempling: 42
  random_seed_sgkf: 42
  random_seed_models: 42
  random_seed_optuna: 42
```

### 5. `optuna.yaml`
Global configuration for the Hyperparameter Optimization (HPO) engine:
* Optimization direction (`maximize`), target metric (`f1_weighted`), number of trials (`500`), timeout limits, and Stratified Group K-Fold split counts (`3`).
* Candidate model whitelist and shared strategy search spaces (resampling strategies: `smote`, `undersampling`, `class_weight`; scalers: `standard`, `robust`, `minmax`).

---

## Python Typed Schemas & Loaders

### Schemas (`configs/pipeline_config.py`)
All YAML dictionaries are converted into typed, frozen dataclasses:
```python
@dataclass(frozen=True)
class SplitTrainingConfig:
    """Configuration for ML model training and validation splitting."""
    random_seed: int
    col_group: str
    train_split: float
    target_column: str
```

### Loaders (`configs/pipeline_loader.py`)
Provides dedicated loader functions that deserialize YAML content into dataclasses with dependency injection:
* `load_kaggle_config() -> KaggleConfig`
* `load_split_holdout_config() -> SplitHoldoutConfig`
* `load_split_training_config() -> SplitTrainingConfig` *(injects `random_seed_train_split`)*
* `load_cleaning_preprocessing_config() -> CleanPreprocessingConfig`
* `load_preprocessing_config() -> PreprocessingConfig`
* `load_drift_detection_config() -> DriftDetectionConfig`
* `load_performances_monitoring_config() -> PerformanceMonitoringConfig`
* `load_retraining_policy_config() -> RetrainingPolicyConfig`
* `load_random_seed_config() -> RandomSeedConfig`

---

## Path Resolution Engine (`paths.py`)

The `DataPathConfig` class dynamically resolves all paths relative to the repository root directory (`PROJECT_DIR`). This prevents broken relative paths when scripts are invoked from different working directories or Docker containers:

```python
from configs.paths import DataPathConfig

paths = DataPathConfig()

# Access key directories
print(paths.DATA_DIR)        # /path/to/project/data
print(paths.MODELS_DIR)      # /path/to/project/models
print(paths.REPORTS_DIR)     # /path/to/project/reports

# Access pipeline file paths
print(paths.raw_data)        # .../data/raw/star_classification.csv
print(paths.best_model_file) # .../models/best_model.joblib
```

---

## Hyperparameter Optimization (`search_space_models/`)

Each candidate model has an isolated YAML specification adhering to an identical enterprise standard:
1. **Header Banner**: Uniform ASCII banner indicating the model architecture.
2. **Two-Line Technical Comments**: Every hyperparameter features:
   * Line 1: Clear technical description of what the hyperparameter controls.
   * Line 2: The direct trade-off (bias/variance, runtime impact, regularization effect).
3. **Fixed Execution Section**: Execution parameters (`n_jobs: -1`, `verbose: 0`, objective functions, class balancing).

### Model Architecture Catalog (`defined_models.yaml`)
```yaml
names_defined_models:
  # Ensemble & Tree-based Classifiers
  - "random_forest"
  - "extra_trees"
  - "xgboost"
  - "lightgbm"
  - "catboost"

  # Deep Neural Network Architecture
  - "dense_nn"

  # Linear & Kernel Classifiers
  - "svc"
  - "sgd"
  - "logreg"
```

---

## Usage Examples & Verification

All public classes, dataclasses, and loader functions are exported directly at the package root (`configs`):

### Example 1: Loading Pipeline Configurations
```python
from configs import (
    load_split_training_config,
    load_preprocessing_config,
    load_drift_detection_config,
)

# Load immutable, strongly-typed configuration objects
split_cfg = load_split_training_config()
print(f"Target column: {split_cfg.target_column}")
print(f"Train split ratio: {split_cfg.train_split}")
print(f"Injected seed: {split_cfg.random_seed}")

prep_cfg = load_preprocessing_config()
print(f"Columns to scale: {prep_cfg.columns_to_scale}")

drift_cfg = load_drift_detection_config()
print(f"KS threshold: {drift_cfg.ks_threshold}")
print(f"Max class shift: {drift_cfg.max_shift_class}")
```

#### Expected Output:
```text
Target column: class
Train split ratio: 0.85
Injected seed: 42
Columns to scale: ['r', 'redshift', 'u-g', 'g-r', 'r-i', 'i-z']
KS threshold: 0.3
Max class shift: 0.3
```

---

### Example 2: Resolving Project Paths
```python
from configs import DataPathConfig, PROJECT_ROOT

paths = DataPathConfig()

# Display paths relative to the project root
print(f"Project Root: {PROJECT_ROOT}")
print(f"Raw data file: {paths.raw_data_path.relative_to(PROJECT_ROOT)}")
print(f"Processed training split: {paths.processed_data.relative_to(PROJECT_ROOT)}")
print(f"Best model artifact: {paths.pipeline_best_model.relative_to(PROJECT_ROOT)}")
print(f"Data config file: {paths.data_config.relative_to(PROJECT_ROOT)}")
```

#### Expected Output:
```text
Project Root: /astro_object_classif
Raw data file: data/raw/Star_Classification.csv
Processed training split: data/interim/training_dataset.csv
Best model artifact: models/best_model/pipeline_best_model.pkl
Data config file: configs/data.yaml
```

---

### Example 3: Loading an Optuna Model Search Space
```python
from configs import OptunaConfigLoader

loader = OptunaConfigLoader()

# Retrieve complete Optuna search space for Extra Trees
extra_trees_space = loader.get_search_space("extra_trees")

print(f"Total parameter count: {len(extra_trees_space)}")
print(f"Estimators parameter: {extra_trees_space['extra_trees_n_estimators']}")
print(f"Injected random seed: {extra_trees_space['random_state']}")
print(f"Resampling strategy choices: {extra_trees_space['resampling_strategy']['choices']}")
print(f"Scaler strategy choices: {extra_trees_space['scaler_strategy']['choices']}")
```

#### Expected Output:
```text
Total parameter count: 13
Estimators parameter: {'type': 'int', 'low': 50, 'high': 500}
Injected random seed: {'type': 'fixed', 'value': 42}
Resampling strategy choices: ['class_weight', 'smote', 'undersampling']
Scaler strategy choices: ['standard', 'robust', 'minmax']
```

---

### Quick Verification Script (Smoke Test)
You can verify the end-to-end integrity of all configuration YAMLs, paths, and loaders by running this snippet from the repository root:

```python
from configs import (
    DataPathConfig,
    OptunaConfigLoader,
    load_kaggle_config,
    load_split_holdout_config,
    load_split_training_config,
    load_cleaning_preprocessing_config,
    load_preprocessing_config,
    load_drift_detection_config,
    load_performances_monitoring_config,
    load_retraining_policy_config,
    load_random_seed_config,
)

# 1. Path Integrity Verification
paths = DataPathConfig()
assert paths.data_config.exists(), "data.yaml missing"
assert paths.preprocessing_config.exists(), "preprocessing.yaml missing"
assert paths.monitoring_config.exists(), "monitoring.yaml missing"
assert paths.randomseed_config.exists(), "random_seed.yaml missing"
assert paths.optuna_config.exists(), "optuna.yaml missing"

# 2. Schema Deserialization
kaggle = load_kaggle_config()
holdout = load_split_holdout_config()
split = load_split_training_config()
clean = load_cleaning_preprocessing_config()
prep = load_preprocessing_config()
drift = load_drift_detection_config()
perf = load_performances_monitoring_config()
policy = load_retraining_policy_config()
seed = load_random_seed_config()

# 3. Optuna Search Space Loader
loader = OptunaConfigLoader()
rf_space = loader.get_search_space("random_forest")
nn_space = loader.get_search_space("dense_nn")

print("=" * 60)
print("CONFIGURATION SUBSYSTEM VERIFICATION PASSED")
print("=" * 60)
print(f"Dataset Target: {split.target_column} (Train ratio: {split.train_split})")
print(f"Injected Random Seed: {split.random_seed}")
print(f"Features to Scale: {prep.columns_to_scale}")
print(f"Drift KS Threshold: {drift.ks_threshold} | Max Shift: {drift.max_shift_class}")
print(f"Retraining Metric: {policy.metric_to_monitor}")
print(f"Random Forest Optuna Search Space Parameters: {len(rf_space)}")
print(f"Dense NN Optuna Search Space Parameters: {len(nn_space)}")
print("=" * 60)
```

#### Expected Output:
```text
============================================================
CONFIGURATION SUBSYSTEM VERIFICATION PASSED
============================================================
Dataset Target: class (Train ratio: 0.85)
Injected Random Seed: 42
Features to Scale: ['r', 'redshift', 'u-g', 'g-r', 'r-i', 'i-z']
Drift KS Threshold: 0.3 | Max Shift: 0.3
Retraining Metric: f1_weighted
Random Forest Optuna Search Space Parameters: 13
Dense NN Optuna Search Space Parameters: 26
============================================================
```

---

## Extension Guide: Adding New Models

To register a new algorithm into the hyperparameter tuning engine (e.g., `adaboost`):

1. **Create the Search Space YAML**:
   Add `configs/search_space_models/adaboost.yaml` following the standard banner and 2-line comment structure.
2. **Register in Path Resolver**:
   In `configs/paths.py`, add `adaboost_search_space: Path = search_space_models_config_dir / "adaboost.yaml"`.
3. **Register in Master Catalog**:
   Add `"adaboost"` to `names_defined_models` in `configs/search_space_models/defined_models.yaml`.
4. **Register in Optuna Candidate Models**:
   Add `"adaboost"` to `search_space.candidate_models` in `configs/optuna.yaml`.
5. **Implement Model Wrapper**:
   Implement `AdaBoostModel` in `src/models/` and register it in `ModelFactory` (`src/models/model_factory.py`).

---

## Design Principles & MLOps Standards

* **Immutability (`frozen=True`)**: Prevents unintentional in-memory modification of configuration settings across multi-threaded or distributed workers.
* **Separation of Concerns**: Data ingestion logic does not need to know where files reside; it simply receives a `DataPathConfig` or domain dataclass.
* **Safe Deserialization (`yaml.safe_load`)**: Protects against arbitrary code execution vulnerabilities during YAML parsing.
* **Zero Magic Numbers**: Every algorithmic threshold, patience epoch, learning rate boundary, and split ratio is documented and exposed for version control.
