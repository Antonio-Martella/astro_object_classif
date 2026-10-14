from dataclasses import dataclass


# =============================================================================
# 1. DATA PIPELINE SCHEMAS (data.yaml)
# =============================================================================
@dataclass(frozen=True)
class KaggleConfig:
    """Configuration for Kaggle raw dataset acquisition and validation."""
    dataset_path: str
    file_name: str
    dataset_columns: list[str]
    dataset_length: int


@dataclass(frozen=True)
class SplitHoldoutConfig:
    """Configuration for production simulation holdout splitting."""
    daily_split_batch_size: int
    holdout_split: float
    ref_col: str


@dataclass(frozen=True)
class SplitTrainingConfig:
    """Configuration for ML model training and validation splitting."""
    col_group: str
    train_split: float
    target_column: str
    random_seed: int


# =============================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING SCHEMAS (preprocessing.yaml)
# =============================================================================
@dataclass(frozen=True)
class CleanPreprocessingConfig:
    """Configuration for data cleaning, sentinel removal, and color indices creation."""
    anomaly_values: list[int | float]
    new_features: list[list[str]]
    columns_to_hold: list[str]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for continuous feature normalization and scaling."""
    columns_to_scale: list[str]


# =============================================================================
# 3. MONITORING & RETRAINING SCHEMAS (monitoring.yaml)
# =============================================================================
@dataclass(frozen=True)
class DriftDetectionConfig:
    """Configuration for feature and class distribution drift detection."""
    ks_threshold: float
    max_shift_class: float
    max_drift_ratio: float


@dataclass(frozen=True)
class PerformanceMonitoringConfig:
    """Configuration for batch performance tracking and alert degradation limits."""
    metric_to_monitor: str
    min_global_threshold: float
    min_sectorial_threshold: float
    max_global_relative_drop: float
    max_sectorial_relative_drop: float


@dataclass(frozen=True)
class RetrainingPolicyConfig:
    """Configuration for MLOps automated model retraining decision logic."""
    metric_to_monitor: str
    consecutive_critical_threshold: int
    catastrophic_drop_threshold: float
    max_drift_ratio: float
    cooldown_batches: int


# =============================================================================
# 4. RANDOM SEEDS SCHEMA (random_seed.yaml)
# =============================================================================
@dataclass(frozen=True)
class RandomSeedConfig:
    """Configuration for global random seeds across the pipeline."""
    random_seed_train_split: int
    random_seed_resempling: int
    random_seed_sgkf: int
    random_seed_models: int
    random_seed_optuna: int