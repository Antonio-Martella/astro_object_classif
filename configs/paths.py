from dataclasses import dataclass
from pathlib import Path

# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base subdirectories
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class DataPathConfig:
    """
    Centralized path configuration for the Astro Object Classification project.
    Provides consistent access to datasets, configs, model artifacts, reports, and logs.
    """

    # =========================================================================
    # 1. DATA STORAGE LAYER (data/)
    # =========================================================================
    # --- 1.1 Raw Data (Original SDSS dataset from Kaggle) ---
    raw_data_dir: Path = DATA_DIR / "raw"
    raw_data_path: Path = raw_data_dir / "Star_Classification.csv"
    raw_data_metadata: Path = raw_data_dir / "Star_Classification_metadata.json"

    # --- 1.2 Interim Data (Training dataset & simulation holdout) ---
    interim_dir: Path = DATA_DIR / "interim"
    split_training_path: Path = interim_dir / "training_dataset.csv"
    split_interim_metadata_path: Path = interim_dir / "interim_dataset_metadata.json"

    # --- 1.3 Production Data (Daily batches & holdout simulation) ---
    production_dir: Path = DATA_DIR / "production"
    split_production_path: Path = production_dir / "holdout_dataset.csv"
    split_production_metadata_path: Path = production_dir / "holdout_dataset_metadata.json"

    # Daily batches for monitoring & streaming inference
    split_daily_batches_path: Path = production_dir / "daily_batches"
    split_daily_batches_metadata_path: Path = (
        split_daily_batches_path / "daily_batches_metadata.json"
    )

    # Archived daily batches after retraining triggers
    split_archived_batch_path: Path = production_dir / "archived_batches"
    split_archived_batch_metadata_path: Path = (
        split_archived_batch_path / "archived_batches_metadata.json"
    )

    # Holdout batch full predictions & evaluation metrics
    holdout_dataset_predicted: Path = (
        production_dir / "predictions" / "holdout_pred.csv"
    )
    holdout_dataset_metrics: Path = (
        production_dir / "predictions" / "holdout_pred_metrics.json"
    )

    # --- 1.4 Processed Data (Cleaned and preprocessed for training) ---
    processed_dir: Path = DATA_DIR / "processed"
    processed_data: Path = processed_dir / "processed_data.csv"
    processed_data_metadata: Path = (
        processed_dir / "processed_data_metadata.json"
    )

    # =========================================================================
    # 2. CONFIGURATION LAYER (configs/)
    # =========================================================================
    optuna_config: Path = CONFIGS_DIR / "optuna.yaml"
    data_config: Path = CONFIGS_DIR / "data.yaml"
    preprocessing_config: Path = CONFIGS_DIR / "preprocessing.yaml"
    monitoring_config: Path = CONFIGS_DIR / "monitoring.yaml"
    randomseed_config: Path = CONFIGS_DIR / "random_seed.yaml"

    # Model definitions and hyperparameter search spaces
    search_space_models_config_dir: Path = CONFIGS_DIR / "search_space_models"
    names_defined_models: Path = search_space_models_config_dir / "defined_models.yaml"
    random_forest_search_space: Path = search_space_models_config_dir / "random_forest.yaml"
    extra_trees_search_space: Path = search_space_models_config_dir / "extra_trees.yaml"
    xgboost_search_space: Path = search_space_models_config_dir / "xgboost.yaml"
    lightgbm_search_space: Path = search_space_models_config_dir / "lightgbm.yaml"
    catboost_search_space: Path = search_space_models_config_dir / "catboost.yaml"
    dense_nn_search_space: Path = search_space_models_config_dir / "dense_nn.yaml"
    logreg_search_space: Path = search_space_models_config_dir / "logreg.yaml"
    sgd_search_space: Path = search_space_models_config_dir / "sgd.yaml"
    svc_search_space: Path = search_space_models_config_dir / "svc.yaml"

    # =========================================================================
    # 3. SERIALIZED MODELS & ARTIFACTS (models/)
    # =========================================================================
    cleaner_pipeline: Path = (
        MODELS_DIR / "cleaner pipeline" / "cleaner_pipeline.pkl"
    )
    target_le: Path = (
        MODELS_DIR / "target_label_encoder" / "target_encoder.pkl"
    )
    pipeline_best_model: Path = (
        MODELS_DIR / "best_model" / "pipeline_best_model.pkl"
    )
    custom_model_pipeline: Path = (
        MODELS_DIR / "custom_model" / "custom_model_pipeline.pkl"
    )

    # =========================================================================
    # 4. TRACKING, REPORTS & EVALUATION (reports/ & mlflow.db)
    # =========================================================================
    mlflow_db_path: Path = PROJECT_ROOT / "mlflow.db"

    # Best model evaluation & figures
    best_model: Path = REPORTS_DIR / "best_model"
    best_model_metrics: Path = best_model / "best_model_metrics.json"
    best_model_hyperparameters: Path = (
        best_model / "best_model_iperparameters.json"
    )

    # Custom model metrics
    custom_model_dir: Path = REPORTS_DIR / "custom_model"
    custom_model_metrics: Path = (
        custom_model_dir / "custom_model_metric.json"
    )

    # Monitoring reports (data drift, model performance, retraining triggers)
    monitoring_reports_dir: Path = REPORTS_DIR / "monitoring"
    retraining_decision_report: Path = (
        monitoring_reports_dir / "retraining_decision_report.json"
    )

    # =========================================================================
    # 5. LOGGING LAYER (logs/)
    # =========================================================================
    main_log: Path = LOGS_DIR / "main" / "main.log"
    make_dataset_log: Path = LOGS_DIR / "make_dataset" / "make_dataset.log"
    training_log: Path = LOGS_DIR / "training" / "training.log"
    optimization_dir_log: Path = LOGS_DIR / "optimization"
    inference_log: Path = LOGS_DIR / "inference" / "inference.log"
    monitoring_log: Path = LOGS_DIR / "monitoring" / "monitoring.log"
    api_log: Path = LOGS_DIR / "api" / "api.log"

    # =========================================================================
    # 6. SYSTEM & DEPENDENCIES
    # =========================================================================
    requirements_file: Path = PROJECT_ROOT / "requirements.txt"