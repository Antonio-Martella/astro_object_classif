import json
import logging
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn import set_config

from configs.paths import DataPathConfig
from configs.pipeline_loader import load_cleaning_preprocessing_config, load_kaggle_config
from src.data.preprocessing import build_stateless_cleaning_pipeline
from src.utils.logger import restore_logging_after_mlflow
from src.utils.validate_type import validate_type

set_config(transform_output="pandas")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class AstroPredict:
    def __init__(
        self, 
        model_name: str | None = None, 
        model_version: str | None = "latest", 
        data_path: DataPathConfig | None = None,
        verbose: bool = False
    ):
        validate_type(
            model_name=(model_name, (str, type(None))),
            model_version=(model_version, str),
            data_path=(data_path, (DataPathConfig, type(None))),
            verbose=(verbose, bool),
        )

        self.verbose = verbose
        if not self.verbose:
            logger.setLevel(logging.WARNING)

        self.model_name = "Classification_Astro_Model" if model_name is None else model_name

        self.data_path = DataPathConfig() if data_path is None else data_path

        mlflow.set_tracking_uri(f"sqlite:///{self.data_path.mlflow_db_path}")
        self.client = MlflowClient()
        self.model_uri = f"models:/{self.model_name}/{model_version}"
        restore_logging_after_mlflow()

        try:
            logger.info(f"Loading model '{self.model_name}' (version: {model_version}) from MLflow...")
            self.model_pipeline = mlflow.sklearn.load_model(self.model_uri)
            self.model_pipeline.set_output(transform="pandas")

            model_version_details = self.client.get_latest_versions(self.model_name)
            encoder_local_path = mlflow.artifacts.download_artifacts(
                run_id=model_version_details[0].run_id, artifact_path="encoder/label_encoder.json"
            )
            with open(encoder_local_path, "r", encoding="utf-8") as file:
                self.label_encoder = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts from MLflow. Error: {e}") from e

        self.kaggle_config = load_kaggle_config()
        self.clean_config = load_cleaning_preprocessing_config()
        self.cleaning_pipeline = build_stateless_cleaning_pipeline(self.clean_config, verbose)

        with open(self.data_path.processed_data_metadata, "r", encoding="utf-8") as file:
            self.columns_processed = json.load(file)["columns"]

        with open(self.data_path.raw_data_metadata, "r", encoding="utf-8") as file:
            self.columns_original = json.load(file)["columns"]

    def _validate_and_sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_type(X=(X, pd.DataFrame))

        if "class" in X.columns:
            X = X.drop(["class"], axis=1)
        if "field_ID" in X.columns:
            X = X.drop(["field_ID"], axis=1)

        if not all((col in self.columns_processed or col in self.columns_original) for col in X.columns):
            raise ValueError(
                "Attention: The features of the provided datapoints do not match those of the original dataset"
            )

        return X

    def _decode_predictions(self, raw_preds: np.ndarray) -> list[str]:
        validate_type(raw_preds=(raw_preds, np.ndarray))

        return [self.label_encoder["classes"][prediction] for prediction in raw_preds]

    def predict(self, X: pd.DataFrame, cleaned_dataset: bool = False):
        validate_type(X=(X, pd.DataFrame))

        X_clean = self._validate_and_sanitize(X)
        if not cleaned_dataset:
            X_trans = self.cleaning_pipeline.transform(X_clean)
        else:
            X_trans = X_clean.copy()

        preds = self.model_pipeline.predict(X_trans)
        predict_proba = self.model_pipeline.predict_proba(X_trans)
        decoded_preds = self._decode_predictions(preds)

        result_df = X.copy()
        result_df["prediction"] = decoded_preds
        result_df["proba_prediction"] = [max(single) for single in predict_proba]
        result_df["prediction_timestamp"] = datetime.now().isoformat()

        return result_df

    def metrics_best_model(self) -> dict[str, float]:
        model_versions = self.client.get_latest_versions(self.model_name)

        if not model_versions:
            raise ValueError(
                f"No versions found for registered model '{self.model_name}'."
            )

        model_version = model_versions[0]
        run = self.client.get_run(model_version.run_id)
        metrics = run.data.metrics

        logger.info(f"Metrics loaded successfully from the best model in mlflow.")

        return metrics
