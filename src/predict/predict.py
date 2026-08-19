import json
import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn import set_config

from configs.schemas_loader import load_cleaning_preprocessing_config, load_kaggle_config
from src.data.preprocessing import build_stateless_cleaning_pipeline
from src.utils.logger import restore_logging_after_mlflow
from src.utils.validate_type import validate_type

set_config(transform_output="pandas")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class AstroPredict:
    def __init__(self, model_name: str | None = None, model_version: str = "latest"):
        validate_type(model_name=(model_name, (str, type(None))), model_version=(model_version, str))

        client = MlflowClient()
        self.model_uri = f"models:/{model_name}/{model_version}"
        restore_logging_after_mlflow()

        try:
            logger.info(f"Loading model '{model_name}' (version: {model_version}) from MLflow...")
            self.model_pipeline = mlflow.sklearn.load_model(self.model_uri)

            model_version_details = client.get_latest_versions(model_name)
            encoder_local_path = mlflow.artifacts.download_artifacts(
                run_id=model_version_details[0].run_id, artifact_path="encoder/label_encoder.json"
            )
            with open(encoder_local_path, "r", encoding="utf-8") as file:
                self.label_encoder = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts from MLflow. Error: {e}") from e

        self.kaggle_config = load_kaggle_config()
        self.clean_config = load_cleaning_preprocessing_config()
        self.cleaning_pipeline = build_stateless_cleaning_pipeline(self.clean_config)

        self.expected_features = [col for col in self.kaggle_config.dataset_columns if col not in ["class", "field_ID"]]

    def _validate_and_sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_type(X=(X, pd.DataFrame))

        if "class" in X.columns:
            X = X.drop(["class"], axis=1)
        if "field_ID" in X.columns:
            X = X.drop(["field_ID"], axis=1)

        if sorted(list(X.columns)) != sorted(self.expected_features):
            raise ValueError(
                "Attention: The features of the provided datapoints do not match those of the original dataset"
            )

        return X

    def _decode_predictions(self, raw_preds: np.ndarray) -> list[str]:
        validate_type(raw_preds=(raw_preds, np.ndarray))

        return [self.label_encoder["classes"][prediction] for prediction in raw_preds]

    def predict(self, X: pd.DataFrame):
        validate_type(X=(X, pd.DataFrame))

        X_clean = self._validate_and_sanitize(X)
        X_trans = self.cleaning_pipeline.transform(X_clean)

        preds = self.model_pipeline.predict(X_trans)
        decoded_preds = self._decode_predictions(preds)

        result_df = X.copy()
        result_df["prediction"] = decoded_preds

        return result_df
