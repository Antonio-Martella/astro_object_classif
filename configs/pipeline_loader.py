from pathlib import Path
from typing import Any
import yaml

from configs.paths import DataPathConfig
from configs.pipeline_config import (
    CleanPreprocessingConfig,
    DriftDetectionConfig,
    KaggleConfig,
    PerformanceMonitoringConfig,
    PreprocessingConfig,
    RandomSeedConfig,
    RetrainingPolicyConfig,
    SplitHoldoutConfig,
    SplitTrainingConfig,
)
from src.utils.validate_type import validate_type

# Central path resolver for all configuration file locations
path_config = DataPathConfig()


def _read_yaml(file_yaml_path: Path) -> dict[str, Any]:
    """
    Safely read and parse a YAML file into a dictionary.

    Args:
        file_yaml_path (Path): Absolute filesystem path to the target YAML file.

    Returns:
        dict[str, Any]: Parsed configuration key-value dictionary.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        ValueError: If the file extension is not '.yaml'.
    """
    validate_type(file_yaml_path=(file_yaml_path, Path))

    if not file_yaml_path.exists():
        raise FileNotFoundError("The yaml file not found")

    if file_yaml_path.suffix != '.yaml':
        raise ValueError(f"Invalid extension: the file must be a yaml (received: '{file_yaml_path.suffix }')")

    with open(file_yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# 1. DATA LOADERS (data.yaml & random_seed.yaml)
# =============================================================================
def load_kaggle_config(path_config: DataPathConfig | None = None) -> KaggleConfig:
    """
    Load raw dataset acquisition and validation parameters.

    Reads the 'kaggle' section from data.yaml to configure remote dataset
    downloading, expected schema columns, and integrity checks.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        KaggleConfig: Immutable dataclass with Kaggle dataset metadata.
    
    Raises:
        FileNotFoundError: If the data.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.data_config.exists():
        raise FileNotFoundError(f"Configuration data file not found at path: {path_config.data_config}")
    
    raw = _read_yaml(path_config.data_config)
    return KaggleConfig(**raw["kaggle"])


def load_split_holdout_config(path_config: DataPathConfig | None = None) -> SplitHoldoutConfig:
    """
    Load configuration for simulation holdout dataset splitting.

    Reads the 'split_dataset_holdout' section from data.yaml to isolate
    production simulation streaming batches from model training data.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        SplitHoldoutConfig: Parameters for temporal/holdout partitioning.
    
    Raises:
        FileNotFoundError: If the data.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.data_config.exists():
        raise FileNotFoundError(f"Configuration data file not found at path: {path_config.data_config}")
    
    raw = _read_yaml(path_config.data_config)
    return SplitHoldoutConfig(**raw["split_dataset_holdout"])


def load_split_training_config(path_config: DataPathConfig | None = None) -> SplitTrainingConfig:
    """
    Load ML training and validation split configuration.

    Reads 'split_dataset_training' from data.yaml and dynamically injects
    the dedicated pseudo-random seed from random_seed.yaml to enforce
    end-to-end reproducibility.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        SplitTrainingConfig: Training partition ratios, grouping column, and seed.

    Raises:
        FileNotFoundError: If either data.yaml or random_seed.yaml does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.data_config.exists():
        raise FileNotFoundError(f"Configuration data file not found at path: {path_config.data_config}")

    if not path_config.randomseed_config.exists():
        raise FileNotFoundError(f"Random seed configuration file not found at path: {path_config.randomseed_config}")

    raw_data = _read_yaml(path_config.data_config)
    raw_seed = _read_yaml(path_config.randomseed_config)

    return SplitTrainingConfig(
        **raw_data["split_dataset_training"],
        random_seed=raw_seed["global_seed"]["random_seed_train_split"],
    )


# =============================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING LOADERS (preprocessing.yaml)
# =============================================================================
def load_cleaning_preprocessing_config(path_config: DataPathConfig | None = None) -> CleanPreprocessingConfig:
    """Load data cleaning, sentinel value pruning, and color indices rules.

    Reads 'cleaning_preprocessing' from preprocessing.yaml to define photometric
    error bounds, outlier sentinels (-9999), and feature engineering combinations.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        CleanPreprocessingConfig: Cleaning thresholds and feature derivation rules.
    
    Raises:
        FileNotFoundError: If the preprocessing.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.preprocessing_config.exists():
        raise FileNotFoundError(f"Preprocessing configuration file not found at path: {path_config.preprocessing_config}")

    raw = _read_yaml(path_config.preprocessing_config)
    return CleanPreprocessingConfig(**raw["cleaning_preprocessing"])


def load_preprocessing_config(path_config: DataPathConfig | None = None) -> PreprocessingConfig:
    """Load feature scaling and normalization configurations.

    Reads the 'preprocessing' section from preprocessing.yaml to designate
    which continuous numerical columns are subject to scaling transformations.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        PreprocessingConfig: List of columns targeted for feature scaling.
    
    Raises:
        FileNotFoundError: If the preprocessing.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.preprocessing_config.exists():
        raise FileNotFoundError(f"Preprocessing configuration file not found at path: {path_config.preprocessing_config}")

    raw = _read_yaml(path_config.preprocessing_config)
    return PreprocessingConfig(**raw["preprocessing"])


# =============================================================================
# 3. MONITORING & RETRAINING LOADERS (monitoring.yaml)
# =============================================================================
def load_drift_detection_config(path_config: DataPathConfig | None = None) -> DriftDetectionConfig:
    """Load statistical data and class distribution drift parameters.

    Reads 'drift_monitoring' from monitoring.yaml to configure Kolmogorov-Smirnov
    test significance thresholds and class frequency shift tolerances.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        DriftDetectionConfig: Statistical drift testing limits and thresholds.

    Raises:
        FileNotFoundError: If the monitoring.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.monitoring_config.exists():
        raise FileNotFoundError(f"Monitoring configuration file not found at path: {path_config.monitoring_config}")
    
    raw = _read_yaml(path_config.monitoring_config)
    return DriftDetectionConfig(**raw["drift_monitoring"])


def load_performances_monitoring_config(path_config: DataPathConfig | None = None) -> PerformanceMonitoringConfig:
    """Load batch performance tracking and metric degradation limits.

    Reads 'performance_monitoring' from monitoring.yaml to set alert limits
    for macro F1-score drops across overall and per-class streaming windows.
    
    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        PerformanceMonitoringConfig: Performance evaluation alert thresholds.

    Raises:
        FileNotFoundError: If the monitoring.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.monitoring_config.exists():
        raise FileNotFoundError(f"Monitoring configuration file not found at path: {path_config.monitoring_config}")

    raw = _read_yaml(path_config.monitoring_config)
    return PerformanceMonitoringConfig(**raw["performance_monitoring"])


def load_retraining_policy_config(path_config: DataPathConfig | None = None) -> RetrainingPolicyConfig:
    """Load automated pipeline retraining triggers and cooldown policies.

    Reads 'retraining_policy' from monitoring.yaml to define criteria for
    triggering retraining jobs upon consecutive alerts or catastrophic drift.

    Returns:
        RetrainingPolicyConfig: Rules governing automated retraining decisions.

    Raises:
        FileNotFoundError: If the monitoring.yaml file does not exist at the specified path.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.monitoring_config.exists():
        raise FileNotFoundError(f"Monitoring configuration file not found at path: {path_config.monitoring_config}")
    
    raw = _read_yaml(path_config.monitoring_config)
    return RetrainingPolicyConfig(**raw["retraining_policy"])


# =============================================================================
# 4. RANDOM SEED LOADER (random_seed.yaml)
# =============================================================================
def load_random_seed_config(path_config: DataPathConfig | None = None) -> RandomSeedConfig:
    """Load system-wide pseudo-random seeds.

    Reads 'global_seed' from random_seed.yaml to ensure deterministic execution
    across cross-validation splits, sampling, model initializations, and Optuna.

    Args:
        path_config (DataPathConfig | None): Optional custom path configuration.

    Returns:
        RandomSeedConfig: Dataclass containing all subsystem random seeds.
    """
    path_config = DataPathConfig() if path_config is None else path_config

    if not path_config.randomseed_config.exists():
        raise FileNotFoundError(f"Random seed configuration file not found at path: {path_config.randomseed_config}")

    raw = _read_yaml(path_config.randomseed_config)
    return RandomSeedConfig(**raw["global_seed"])