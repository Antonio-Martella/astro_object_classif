import logging
import os
from pathlib import Path
from typing import get_type_hints

import joblib
import mlflow
import pandas as pd
from imblearn.pipeline import Pipeline as Imbpipeline

from configs.paths import PROJECT_ROOT, DataPathConfig
from configs.schemas import CleanPreprocessingConfig, KaggleConfig, SplitHoldoutConfig
from configs.schemas_loader import (
    load_cleaning_preprocessing_config,
    load_kaggle_config,
    load_split_holdout_config,
)
from src.data.holdout_split_data import SplitProductionSimulation

# Internal moduls
from src.data.ingestion import KaggleDownloader
from src.data.preprocessing import ProcessedDataSaver, build_stateless_cleaning_pipeline
from src.optimization.search import file_md5
from src.utils.logger import restore_logging_after_mlflow, setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_ingestion(kaggle_config: KaggleConfig, path_config: DataPathConfig) -> None:
    for name, expected_type in get_type_hints(run_ingestion).items():
        if name == "return":
            continue
        validate_type(**{name: (locals()[name], expected_type)})

    downloader = KaggleDownloader(kaggle_config, path_config)
    df = downloader.download()
    downloader.save(df)


def run_split(holdout_split_config: SplitHoldoutConfig, path_config: DataPathConfig) -> pd.DataFrame:
    for name, expected_type in get_type_hints(run_split).items():
        if name == "return":
            continue
        validate_type(name=(locals()[name], expected_type))

    return SplitProductionSimulation(path_config, holdout_split_config).execute()


def run_cleaning(
    df: pd.DataFrame,
    path_config: DataPathConfig,
    clean_preprocessing_config: CleanPreprocessingConfig,
) -> Imbpipeline:
    for name, expected_type in get_type_hints(run_cleaning).items():
        if name == "return":
            continue
        validate_type(name=(locals()[name], expected_type))

    cleaner_pipeline = build_stateless_cleaning_pipeline(clean_preprocessing_config)

    df_clean = cleaner_pipeline.fit_transform(df)
    ProcessedDataSaver(path_config).save(df_clean)

    os.makedirs(Path(path_config.cleaner_pipeline).parent, exist_ok=True)
    joblib.dump(cleaner_pipeline, path_config.cleaner_pipeline)

    return cleaner_pipeline


def run_mlflow_tracking(pipeline: Imbpipeline, path_config: DataPathConfig) -> None:
    for name, expected_type in get_type_hints(run_mlflow_tracking).items():
        if name == "return":
            continue
        validate_type(name=(locals()[name], expected_type))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{path_config.mlflow_db_path}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Astro_Object_Classification")

    with mlflow.start_run(run_name="Cleaner_pipeline"):
        mlflow.set_tags(
            {
                "raw_dataset_path": path_config.raw_data_path.relative_to(PROJECT_ROOT),
                "raw_dataset_md5": file_md5(path_config.raw_data_path),
                "raw_dataset_size_bytes": os.path.getsize(path_config.raw_data_path),
            }
        )
        mlflow.set_tags(
            {
                "holdout_dataset_path": path_config.split_production_path.relative_to(PROJECT_ROOT),
                "holdout_dataset_md5": file_md5(path_config.split_production_path),
                "holdout_dataset_size_bytes": os.path.getsize(path_config.split_production_path),
            }
        )
        mlflow.set_tags(
            {
                "training_dataset_path": path_config.split_training_path.relative_to(PROJECT_ROOT),
                "training_dataset_md5": file_md5(path_config.split_training_path),
                "training_dataset_size_bytes": os.path.getsize(path_config.split_training_path),
            }
        )
        mlflow.set_tags(
            {
                "processed_dataset_path": path_config.processed_data_path.relative_to(PROJECT_ROOT),
                "processed_dataset_md5": file_md5(path_config.processed_data_path),
                "processed_dataset_size_bytes": os.path.getsize(path_config.processed_data_path),
            }
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="cleaner_pipeline",
            registered_model_name="Cleaner_Astro_Pipeline",
        )

        restore_logging_after_mlflow()
        logger.info("Process finished!")
        mlflow.log_artifact(local_path=str(path_config.opt_logs_make_dataset), artifact_path="logs")


def main():
    path_config = DataPathConfig()
    setup_logger(run_log_file_path=path_config.opt_logs_make_dataset)

    # ###################################
    # #### 0. Load the Configuration ####
    # ###################################
    logger.info("Configuration Load...")
    try:
        kaggle_config = load_kaggle_config()
        holdout_split_config = load_split_holdout_config()
        preprocessing_config = load_cleaning_preprocessing_config()
    except Exception as e:
        logger.exception("Error occurred while loading the configuration.")
        raise RuntimeError(f"Phase 0 failed: {e}")

    # #############################################
    # #### 1. Download the dataset from Kaggle ####
    # #############################################
    logger.info("Fase 1: Download dati da Kaggle...")
    try:
        run_ingestion(kaggle_config, path_config)
    except Exception as e:
        logger.exception("Error occurred while downloading the dataset from Kaggle.")
        raise RuntimeError(f"Phase 1 failed: {e}") from e

    # ########################################################################
    # #### 2. Splitting raw dataset into Training and Production datasets ####
    # ########################################################################
    logger.info("Fase 2: The raw dataset is split into a Training dataset and a Production datase...")
    try:
        df_training = run_split(holdout_split_config, path_config)
    except Exception as e:
        logger.exception("Error occurred during the dataset splitting (Train/Production).")
        raise RuntimeError(f"Phase 2 failed: {e}") from e

    # ######################################
    # #### 3. CLEANING and SAVE DATASET ####
    # ######################################
    logger.info("Fase 3: Cleaning Training dataset...")
    try:
        cleaner_pipeline = run_cleaning(df_training, path_config, preprocessing_config)
    except Exception as e:
        logger.exception("Error occurred while cleaning the dataset.")
        raise RuntimeError(f"Phase 3 failed: {e}") from e

    # ############################
    # #### 4. MLflow Tracking ####
    # ############################
    logger.info("Fase 4: MLflow Tracking...")
    try:
        run_mlflow_tracking(cleaner_pipeline, path_config)
    except Exception as e:
        logger.exception("Error occurred during MLflow tracking.")
        raise RuntimeError(f"Phase 4 failed: {e}") from e


if __name__ == "__main__":
    main()
