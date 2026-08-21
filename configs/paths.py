from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DataPathConfig:
    # --- Dataset Originale ---
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "Star_Classification.csv"
    raw_data_columns: Path = PROJECT_ROOT / "data" / "raw" / "Star_Classification_columns.json"

    # --- Dataset Splittati ---
    # Path dataset per training modelli
    split_training_path: Path = PROJECT_ROOT / "data" / "interim" / "training_dataset.csv"
    # Path dei metadati per il dataset di simulazione
    split_metadata_path: Path = PROJECT_ROOT / "data" / "interim" / "metadata.json"
    # Path dataset di simulazione di produzione
    split_production_path: Path = PROJECT_ROOT / "data" / "production" / "holdout_dataset.csv"
    # Path subdirectory di salvataggio batch giornalieri
    split_production_batch_path: Path = PROJECT_ROOT / "data" / "production" / "daily_batch"

    # Path dei metadati per il dataset di simulazione
    holdout_dataset_full_predicted: Path = (
        PROJECT_ROOT / "data" / "interim" / "holdout_full_predict" / "holdout_pred_full.csv"
    )
    holdout_dataset_full_metrics: Path = (
        PROJECT_ROOT / "data" / "interim" / "holdout_full_predict" / "full_metrics.json"
    )
    holdout_dataset_batch_predicted: Path = (
        PROJECT_ROOT / "data" / "interim" / "holdout_batch_predict" / "holdout_pred_batch.csv"
    )
    holdout_dataset_batch_metrics: Path = (
        PROJECT_ROOT / "data" / "interim" / "holdout_batch_predict" / "batch_metrics.json"
    )

    # --- Dataset Processato ---
    processed_data_path: Path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

    # --- Pipeline preprocess ---
    cleaner_pipeline: Path = PROJECT_ROOT / "models" / "cleaner pipeline" / "cleaner_pipeline.pkl"

    # --- OPTUNA config ---
    optuna_config: Path = PROJECT_ROOT / "configs" / "optuna.yaml"

    # ---
    params_config: Path = PROJECT_ROOT / "configs" / "params.yaml"

    # --- Configurazione generale per i random seed ---
    randomseed_config: Path = PROJECT_ROOT / "configs" / "random_seed.yaml"

    # --- Path Model's Space configs ---
    random_forest_search_space: Path = PROJECT_ROOT / "configs" / "models" / "random_forest.yaml"
    xgboost_search_space: Path = PROJECT_ROOT / "configs" / "models" / "xgboost.yaml"
    lightgbm_search_space: Path = PROJECT_ROOT / "configs" / "models" / "lightgbm.yaml"
    catboost_search_space: Path = PROJECT_ROOT / "configs" / "models" / "catboost.yaml"
    dense_nn_search_space: Path = PROJECT_ROOT / "configs" / "models" / "dense_nn.yaml"
    logreg_search_space: Path = PROJECT_ROOT / "configs" / "models" / "logreg.yaml"
    sgd_search_space: Path = PROJECT_ROOT / "configs" / "models" / "sgd.yaml"
    svc_search_space: Path = PROJECT_ROOT / "configs" / "models" / "svc.yaml"

    # --- Save Label Encoding Target ---
    target_le: Path = PROJECT_ROOT / "models" / "target_label_encoder" / "target_encoder.pkl"

    # --- Save Best Model (retrained) ---
    pipeline_best_model: Path = PROJECT_ROOT / "models" / "best_model" / "pipeline_best_model.pkl"

    # --- MLflow Database ---
    mlflow_db_path: Path = PROJECT_ROOT / "mlflow.db"

    # --- Salvataggio delle info del best model in locale ---
    best_model_metrics: Path = PROJECT_ROOT / "reports" / "best_model" / "best_model_metrics.json"
    best_model_hyperparameters: Path = PROJECT_ROOT / "reports" / "best_model" / "best_model_iperparameters.json"

    # --- Salvataggio immagini esplicative best model ---
    best_model: Path = PROJECT_ROOT / "reports" / "best_model"

    # --- logs file ---
    opt_logs_dir: Path = PROJECT_ROOT / "logs"
    opt_logs_make_dataset = opt_logs_dir / "run_clean_dataset_creation.log"

    # --- requirements ---
    requirements_file: Path = PROJECT_ROOT / "requirements.txt"

    # --- Monitoring & Drift Reports ---
    monitoring_reports_dir: Path = PROJECT_ROOT / "reports" / "monitoring"
    drift_report_full: Path = PROJECT_ROOT / "reports" / "monitoring" / "drift_report_full.json"
    drift_report_batches: Path = PROJECT_ROOT / "reports" / "monitoring" / "drift_report_batches.json"
