import pytest
from pathlib import Path
from unittest.mock import MagicMock

from configs.pipeline_loader import (
    _read_yaml,
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
from configs.pipeline_config import (
    KaggleConfig, 
    SplitHoldoutConfig, 
    SplitTrainingConfig,
    CleanPreprocessingConfig,
    PreprocessingConfig,
    DriftDetectionConfig,
    PerformanceMonitoringConfig,
    RetrainingPolicyConfig,
    RandomSeedConfig,
)
from configs.paths import DataPathConfig

# =============================================================================
# 1. TEST HELPER FUNCTION (_read_yaml)
# =============================================================================
class TestReadYaml:
    """Unit tests for the internal YAML file reader helper."""
    def test_file_not_exist(self, tmp_path):
        """Verify that attempting to read a non-existent file raises FileNotFoundError."""
        path_file = tmp_path / "example.yaml"

        with pytest.raises(FileNotFoundError, match="The yaml file not found"):
            _read_yaml(Path(path_file))

    def test_invalid_file_extension(self, tmp_path):
        """Verify that passing a file with a non-.yaml extension raises ValueError."""
        path_file = tmp_path / "example.csv"
        path_file.touch()

        with pytest.raises(ValueError, match="Invalid extension"):
            _read_yaml(Path(path_file))

    def test_read_valid_yaml(self, tmp_path):
        """Verify that a valid YAML file is successfully read and parsed into a dictionary."""
        path_file = tmp_path / "valid_config.yaml"
        path_file.write_text("learning_rate: 0.01\nbatch_size: 32\n", encoding="utf-8")

        result = _read_yaml(Path(path_file))

        assert isinstance(result, dict)
        assert result == {"learning_rate": 0.01, "batch_size": 32}

# =============================================================================
# 2. TEST DATA LOADERS (data.yaml)
# =============================================================================
class TestLoadKaggleConfig:
    """Unit tests for the Kaggle dataset configuration loader."""
    def test_path_config_not_exist(self, tmp_path):
        """Verify that a missing data.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.data_config = tmp_path / "data.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration data file not found"):
            load_kaggle_config(data_path)

    def test_correct_load_kaggle_config(self):
        """Verify that load_kaggle_config loads valid and populated Kaggle metadata."""
        load_config = load_kaggle_config()

        assert isinstance(load_config, KaggleConfig)
        assert len(load_config.dataset_path) > 0
        assert load_config.file_name.endswith(".csv")
        assert len(load_config.dataset_columns) > 0
        assert load_config.dataset_length > 0

class TestLoadSplitHoldoutConfig:
    """Unit tests for the simulation holdout partition configuration loader."""
    def test_path_config_not_exist(self, tmp_path):
        """Verify that a missing data.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.data_config = tmp_path / "data.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration data file not found"):
            load_split_holdout_config(data_path)

    def test_correct_load_split_holdout_config(self):
        """Verify that load_split_holdout_config parses batch size, ratio, and reference column."""
        load_config = load_split_holdout_config()

        assert isinstance(load_config, SplitHoldoutConfig)
        assert isinstance(load_config.daily_split_batch_size, int)
        assert isinstance(load_config.holdout_split, float)
        assert isinstance(load_config.ref_col, str)
        assert load_config.daily_split_batch_size > 0
        assert 0 < load_config.holdout_split < 1
        assert len(load_config.ref_col) > 0


class TestLoadSplitTrainingConfig:
    """Unit tests for the ML training split loader with random seed injection."""
    def test_data_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing data.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.data_config = tmp_path / "data.yaml"
        data_path.randomseed_config = tmp_path / "random_seed.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration data file not found"):
            load_split_training_config(data_path)

    def test_random_seed_config_not_exist_raises_error(self, tmp_path):
        """Verify that an existing data.yaml with a missing random_seed.yaml raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.data_config = tmp_path / "data.yaml"
        data_path.data_config.touch()
        data_path.randomseed_config = tmp_path / "random_seed.yaml"

        with pytest.raises(FileNotFoundError, match="Random seed configuration file not found"):
            load_split_training_config(data_path)

    def test_correct_load_split_training_config(self):
        """Verify that load_split_training_config returns a valid config with injected random seed."""
        load_config = load_split_training_config()

        assert isinstance(load_config, SplitTrainingConfig)
        assert isinstance(load_config.col_group, str)
        assert isinstance(load_config.train_split, float)
        assert isinstance(load_config.target_column, str)
        assert isinstance(load_config.random_seed, int)
        assert len(load_config.col_group) > 0
        assert 0 < load_config.train_split < 1
        assert len(load_config.target_column) > 0
        assert load_config.random_seed >= 0

# =============================================================================
# 3. TEST PREPROCESSING LOADERS (preprocessing.yaml)
# =============================================================================
class TestLoadCleaningPreprocessingConfig:
    """Unit tests for data cleaning, sentinel handling, and feature derivation loader."""
    def test_preprocessing_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing preprocessing.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.preprocessing_config = tmp_path / "preprocessing.yaml"

        with pytest.raises(FileNotFoundError, match="Preprocessing configuration file not found"):
            load_cleaning_preprocessing_config(data_path)

    def test_correct_load_cleaning_preprocessing_config(self):
        """Verify that cleaning rules, engineered features, and anomaly sentinels are correctly loaded."""
        clean_prep = load_cleaning_preprocessing_config()

        assert isinstance(clean_prep, CleanPreprocessingConfig)
        assert len(clean_prep.anomaly_values) > 0
        assert len(clean_prep.new_features) > 0
        assert len(clean_prep.columns_to_hold) > 0
        assert -9999 in clean_prep.anomaly_values or -9999.0 in clean_prep.anomaly_values

class TestLoadPreprocessingConfig:
    """Unit tests for continuous feature scaling configuration loader."""
    def test_preprocessing_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing preprocessing.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.preprocessing_config = tmp_path / "preprocessing.yaml"

        with pytest.raises(FileNotFoundError, match="Preprocessing configuration file not found"):
            load_preprocessing_config(data_path)

    def test_correct_load_preprocessing_config(self):
        """Verify that columns_to_scale is successfully populated from preprocessing.yaml."""
        preprocessing = load_preprocessing_config()

        assert isinstance(preprocessing, PreprocessingConfig)
        assert len(preprocessing.columns_to_scale) > 0

# =============================================================================
# 4. TEST MONITORING & RETRAINING LOADERS (monitoring.yaml)
# =============================================================================
class TestLoadDriftDetectionConfig:
    """Unit tests for statistical data drift detection configuration loader."""
    def test_monitoring_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing monitoring.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.monitoring_config = tmp_path / "monitoring.yaml"

        with pytest.raises(FileNotFoundError, match="Monitoring configuration file not found"):
            load_drift_detection_config(data_path)

    def test_correct_load_drift_detection_config(self):
        """Verify that Kolmogorov-Smirnov and class shift drift thresholds fall in (0, 1)."""
        drift_detect_config = load_drift_detection_config()

        assert isinstance(drift_detect_config, DriftDetectionConfig)
        assert isinstance(drift_detect_config.ks_threshold, float)
        assert isinstance(drift_detect_config.max_shift_class, float)
        assert isinstance(drift_detect_config.max_drift_ratio, float)
        assert 0 < drift_detect_config.ks_threshold < 1
        assert 0 < drift_detect_config.max_shift_class < 1
        assert 0 < drift_detect_config.max_drift_ratio < 1

class TestLoadPerformancesMonitoringConfig:
    """Unit tests for batch performance monitoring and degradation alert loader."""
    def test_monitoring_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing monitoring.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.monitoring_config = tmp_path / "monitoring.yaml"

        with pytest.raises(FileNotFoundError, match="Monitoring configuration file not found"):
            load_performances_monitoring_config(data_path)

    def test_correct_load_performances_monitoring_config(self):
        """Verify that performance thresholds, drop limits, and target metrics are valid."""
        perf_monitoring_config = load_performances_monitoring_config()

        assert isinstance(perf_monitoring_config, PerformanceMonitoringConfig)
        assert isinstance(perf_monitoring_config.metric_to_monitor, str)
        assert isinstance(perf_monitoring_config.min_global_threshold, float)
        assert isinstance(perf_monitoring_config.min_sectorial_threshold, float)
        assert isinstance(perf_monitoring_config.max_global_relative_drop, float)
        assert isinstance(perf_monitoring_config.max_sectorial_relative_drop, float)
        assert len(perf_monitoring_config.metric_to_monitor) > 0
        assert 0 < perf_monitoring_config.min_global_threshold < 1
        assert 0 < perf_monitoring_config.min_sectorial_threshold < 1
        assert 0 < perf_monitoring_config.max_global_relative_drop < 1
        assert 0 < perf_monitoring_config.max_sectorial_relative_drop < 1

class TestLoadRetrainingPolicyConfig:
    """Unit tests for automated model retraining decision policy loader."""
    def test_monitoring_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing monitoring.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.monitoring_config = tmp_path / "monitoring.yaml"

        with pytest.raises(FileNotFoundError, match="Monitoring configuration file not found"):
            load_retraining_policy_config(data_path)

    def test_correct_load_retraining_policy_config(self):
        """Verify that retraining triggers, cooldown batches, and drop tolerances are valid."""
        retraining_config = load_retraining_policy_config()

        assert isinstance(retraining_config, RetrainingPolicyConfig)
        assert isinstance(retraining_config.metric_to_monitor, str)
        assert isinstance(retraining_config.consecutive_critical_threshold, int)
        assert isinstance(retraining_config.catastrophic_drop_threshold, float)
        assert isinstance(retraining_config.max_drift_ratio, float)
        assert isinstance(retraining_config.cooldown_batches, int)
        assert len(retraining_config.metric_to_monitor) > 0
        assert retraining_config.consecutive_critical_threshold > 0
        assert 0 < retraining_config.catastrophic_drop_threshold < 1
        assert 0 < retraining_config.max_drift_ratio < 1
        assert retraining_config.cooldown_batches > 0 

# =============================================================================
# 5. TEST RANDOM SEED LOADER (random_seed.yaml)
# =============================================================================
class TestLoadRandomSeedConfig:
    """Unit tests for global pseudo-random seed configuration loader."""
    def test_load_random_seed_config_not_exist_raises_error(self, tmp_path):
        """Verify that a missing random_seed.yaml file raises FileNotFoundError."""
        data_path = MagicMock(spec=DataPathConfig)
        data_path.randomseed_config = tmp_path / "random_seed.yaml"

        with pytest.raises(FileNotFoundError, match="Random seed configuration file not found"):
            load_random_seed_config(data_path)

    def test_correct_load_random_seed_config(self):
        """Verify that all global random seeds are non-negative integer values."""
        random_seed_config = load_random_seed_config()

        assert isinstance(random_seed_config, RandomSeedConfig)
        assert isinstance(random_seed_config.random_seed_train_split, int)
        assert isinstance(random_seed_config.random_seed_resempling, int)
        assert isinstance(random_seed_config.random_seed_sgkf, int)
        assert isinstance(random_seed_config.random_seed_models, int)
        assert isinstance(random_seed_config.random_seed_optuna, int)
        assert random_seed_config.random_seed_train_split >= 0
        assert random_seed_config.random_seed_resempling >= 0
        assert random_seed_config.random_seed_sgkf >= 0
        assert random_seed_config.random_seed_models >= 0
        assert random_seed_config.random_seed_optuna >= 0