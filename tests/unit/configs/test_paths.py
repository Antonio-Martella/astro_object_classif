from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from configs.paths import (
    CONFIGS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    DataPathConfig,
)


class TestDataPathConfig:
    """
    Unit test suite to verify the integrity and behavior of DataPathConfig. 

    Ensures configuration immutability, the existence and validity of
    main paths and subfolders, as well as the accuracy of log files,
    configuration files, and project artifacts.
    """

    @pytest.fixture
    def paths_configs(self):
        """Initializes a default instance of Data Path Config for testing."""
        return DataPathConfig()

    def test_paths_config_is_immutable(self, paths_configs):
        """Verify that DataPathConfig is immutable and raises a FrozenInstanceError upon modification."""
        with pytest.raises(FrozenInstanceError):
            paths_configs.data_config = Path("/tmp/fake_data.yaml")

    @pytest.mark.parametrize(
        "paths",
        [
            PROJECT_ROOT,
            CONFIGS_DIR,
            MODELS_DIR,
            REPORTS_DIR,
            LOGS_DIR,
        ],
    )
    def test_default_principal_directory(self, paths):
        """Verifica che le directory radice esistano, siano istanze di Path e interne a PROJECT_ROOT."""
        assert paths.exists()
        assert isinstance(paths, Path)
        assert paths.is_relative_to(PROJECT_ROOT)

    @pytest.mark.parametrize(
        "paths",
        [
            "raw_data_dir",
            "interim_dir",
            "production_dir",
            "split_daily_batch_path",
            "split_archived_batch_path",
            "processed_dir",
            "optuna_config",
            "data_config",
            "preprocessing_config",
            "monitoring_config",
            "randomseed_config",
            "search_space_models_config_dir",
            "names_defined_models",
            "random_forest_search_space",
            "extra_trees_search_space",
            "xgboost_search_space",
            "lightgbm_search_space",
            "catboost_search_space",
            "dense_nn_search_space",
            "logreg_search_space",
            "sgd_search_space",
            "svc_search_space",
            "best_model",
            "optimization_dir_log",
            "custom_model_dir",
            "monitoring_reports_dir",
        ],
    )
    def test_default_subdirectory_and_file(self, paths, paths_configs):
        """
        Verify the physical existence, extension (.yaml if applicable), and nesting within PROJECT_ROOT.
        """
        p = getattr(paths_configs, paths)

        assert p.exists()
        assert isinstance(p, Path)
        assert p.suffix == ".yaml" if len(p.suffix) != 0 else True
        assert p.is_relative_to(PROJECT_ROOT)

    def test_existence_of_requirements_file(self, paths_configs):
        """
        Verify that the requirements file path is valid, has a .txt extension, and is located within the project.
        """
        requirements_file = getattr(paths_configs, "requirements_file")

        assert requirements_file
        assert isinstance(requirements_file, Path)
        assert requirements_file.suffix == ".txt"
        assert requirements_file.is_relative_to(PROJECT_ROOT)

    @pytest.mark.parametrize(
        "log_attribute",
        [
            "main_log",
            "make_dataset_log",
            "training_log",
            "inference_log",
            "monitoring_log",
            "api_log",
        ],
    )
    def test_logs_directory(self, log_attribute, paths_configs):
        """
        Verify that the log paths have the .log extension and that the parent directory exists.
        """
        log = getattr(paths_configs, log_attribute)

        assert log.parent.exists()
        assert isinstance(log, Path)
        assert log.suffix == ".log"
        assert log.is_relative_to(PROJECT_ROOT)

    @pytest.mark.parametrize(
        "paths",
        [
            "raw_data_path",
            "raw_data_metadata",
            "split_training_path",
            "split_interim_metadata_path",
            "split_production_path",
            "split_production_metadata_path",
            "split_daily_batches_metadata_path",
            "split_archived_batches_metadata_path",
            "holdout_dataset_predicted",
            "holdout_dataset_metrics",
            "processed_data",
            "processed_data_metadata",
            "cleaner_pipeline",
            "target_le",
            "pipeline_best_model",
            "custom_model_pipeline",
            "mlflow_db_path",
            "best_model_metrics",
            "best_model_hyperparameters",
            "custom_model_metrics",
            "retraining_decision_report",
        ],
    )
    def test_presence_path_in_data_configs(self, paths, paths_configs):
        """
        Verifies the presence of dataset/artifact attributes 
        and the validity of the extensions (.csv, .json, .db, .pkl)
        """
        assert hasattr(paths_configs, paths)
        assert isinstance(getattr(paths_configs, paths), Path)
        assert getattr(paths_configs, paths).suffix in [".csv", ".json", ".db", ".pkl"]