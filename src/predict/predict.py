import logging
import random
import time

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn import set_config

from configs.paths import DataPathConfig
from src.utils.logger import setup_logger

set_config(transform_output="pandas")


setup_logger()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class AstroPredict:
    def __init__(
        self,
        model_name: str | None = None,
        model_version: str = "latest",
        cleaner_name: str | None = None,
        cleaner_version: str = "latest",
    ):
        client = MlflowClient()

        setup_logger()
        logger = logging.getLogger(__name__)

        self.model_uri = f"models:/{model_name}/{model_version}"
        self.cleaner_uri = f"models:/{cleaner_name}/{cleaner_version}"

        logger.info("Caricamento del modello previsionale da MLflow.")
        self.model_pipeline = mlflow.sklearn.load_model(self.model_uri)

        logger.info("Caricamento della pipeline cleaner preprocessor da MLflow.")
        self.cleaner_pipeline = mlflow.sklearn.load_model(self.cleaner_uri)

        logger.info("Caricamento del label encored del target da MLflow.")
        model_version_details = client.get_latest_versions(model_name)
        encoder_local_path = mlflow.artifacts.download_artifacts(
            run_id=model_version_details[0]._run_id, artifact_path="encoder/target_encoder.pkl"
        )

        self.label_encoder = joblib.load(encoder_local_path)

    def predict(self, X: pd.DataFrame):
        X_copy = X.copy()

        if "field_ID" in X.columns:
            X_copy = X_copy.drop(["field_ID"], axis=1)
        if "class" in X.columns:
            X_copy = X_copy.drop(["class"], axis=1)

        X_cleaned = self.cleaner_pipeline.transform(X_copy)
        prediction = self.model_pipeline.predict(X_cleaned)
        text_prediction = self.label_encoder.inverse_transform(prediction)
        X_copy["prediction"] = text_prediction

        return X_copy


if __name__ == "__main__":
    data_path = DataPathConfig()
    mlflow.set_tracking_uri(f"sqlite:///{data_path.mlflow_db_path}")

    MODEL_REGISTRY_NAME = "Classification_Astro_Model"
    CLEANER_REFISTRY_NAME = "Cleaner_Astro_Pipeline"

    logger.info("START")

    predictor = AstroPredict(
        model_name=MODEL_REGISTRY_NAME,
        model_version="latest",
        cleaner_name=CLEANER_REFISTRY_NAME,
        cleaner_version="latest",
    )

    setup_logger()

    logger = logging.getLogger(__name__)

    logger.info("MODELLO CARICATO")

    df = pd.read_csv(data_path.split_production_path)

    logger.info("DATASET CARICATO")

    logger.info("=" * 50)
    logger.info("INZIO SIMULAZIONE DI PRODUZIONE")
    logger.info("=" * 50)

    while True:
        time_lat = random.uniform(1.0, 2.0)
        time.sleep(time_lat)

        record_num = random.randint(0, len(df))
        record = df.iloc[record_num : record_num + 1]

        # prendo il tempo inziale
        start_time = time.perf_counter()

        prediction = predictor.predict(record)["prediction"]

        # prendo il tempo finale
        end_time = time.perf_counter()

        logger.info(
            f"\033[32m(true latency {round(end_time-start_time, 4)}s) \033[0m"
            f"OSSERVAZIONE {record_num}: predizione --> "
            f"\x1b[31m{prediction.iloc[0]} (true {df['class'].iloc[record_num]})\x1b[0m"
        )
