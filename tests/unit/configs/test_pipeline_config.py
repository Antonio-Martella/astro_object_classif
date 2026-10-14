from dataclasses import FrozenInstanceError
import pytest
import yaml

from configs.pipeline_loader import (
    load_kaggle_config,
    load_cleaning_preprocessing_config,
    load_preprocessing_config,
    load_split_holdout_config,
    load_split_training_config,
    load_performances_monitoring_config,
    load_drift_detection_config,
    load_retraining_policy_config,
    load_random_seed_config,
)
from configs.paths import DataPathConfig

path_config = DataPathConfig()
with open(path_config.data_config, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)
with open(path_config.preprocessing_config, 'r', encoding='utf-8') as f:
    prep_config = yaml.safe_load(f)
with open(path_config.monitoring_config, 'r', encoding='utf-8') as f:
    monitoring_config = yaml.safe_load(f)
with open(path_config.randomseed_config, 'r', encoding='utf-8') as f:
    random_seed_config = yaml.safe_load(f)

class TestPipelineConfigImmutability:
    """Suite of unit tests verifying the frozen state and runtime immutability of pipeline configurations."""
    @pytest.mark.parametrize("config_instance, field_to_mutate, dummy_value", [
        (load_kaggle_config(), "dataset_columns", ["col_1", "col_2"]),
        (load_cleaning_preprocessing_config(), "new_features", "Unknown"),
        (load_preprocessing_config(), "columns_to_scale", ["Unknown"]),
        (load_split_holdout_config(), "holdout_split", 0.50),
        (load_split_training_config(), "target_column", "Unknown"),
        (load_performances_monitoring_config(), "max_global_relative_drop", 0.99),
        (load_drift_detection_config(), "ks_threshold", 0.99),
        (load_retraining_policy_config(), "metric_to_monitor", "Unknown"),
        (load_random_seed_config(), "random_seed_models", 0),
    ])
    def test_each_config_is_strictly_immutable(self, config_instance, field_to_mutate, dummy_value):
        """
        Verify individually for each dataclass that field re-assignment raises FrozenInstanceError.
        Args:
            config_instance: Instantiated configuration dataclass loaded from pipeline_loader.
            field_to_mutate (str): Name of the attribute targeted for modification.
            dummy_value: Arbitrary value used to attempt overwriting the frozen attribute.
        """
        with pytest.raises(FrozenInstanceError):
            setattr(config_instance, field_to_mutate, dummy_value)

    @pytest.mark.parametrize("config_instance, field, value, type_value", [
        (load_kaggle_config(), "dataset_path", data_config["kaggle"]["dataset_path"], str),
        (load_cleaning_preprocessing_config(), \
            "new_features", prep_config["cleaning_preprocessing"]["new_features"], list),
        (load_preprocessing_config(), "columns_to_scale", prep_config["preprocessing"]["columns_to_scale"], list),
        (load_split_holdout_config(), "ref_col", data_config["split_dataset_holdout"]["ref_col"], str),
        (load_split_training_config(), "col_group", data_config["split_dataset_training"]["col_group"], str),
        (load_performances_monitoring_config(), \
            "metric_to_monitor", monitoring_config["performance_monitoring"]["metric_to_monitor"], str),
        (load_drift_detection_config(), \
            "max_drift_ratio", monitoring_config["drift_monitoring"]["max_drift_ratio"], float),
        (load_retraining_policy_config(), \
            "metric_to_monitor", monitoring_config["retraining_policy"]["metric_to_monitor"], str),
        (load_random_seed_config(), \
            "random_seed_optuna", random_seed_config["global_seed"]["random_seed_optuna"], int),
    ])
    def test_each_config_is_correct(self, config_instance, field, value, type_value):
            assert getattr(config_instance, field) ==  value
            assert isinstance(value, type_value)