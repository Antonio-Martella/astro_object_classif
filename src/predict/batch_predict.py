import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import classification_report

from configs.paths import DataPathConfig
from src.predict.predict import AstroPredict
from src.utils.logger import restore_logging_after_mlflow, setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def run_batch_prediction(
    input_csv_path: str | Path,
    output_csv_path: str | Path | None = None,
    model_name: str = "Classification_Astro_Model",
    model_version: str = "latest",
    data_path: DataPathConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    logger.info("Let's start predicting the provided batch file...")
    validate_type(
        input_csv_path=(input_csv_path, (str, Path)),
        output_csv_path=(output_csv_path, (str, Path, type(None))),
        model_name=(model_name, str),
        model_version=(model_version, str),
        data_path=(data_path, (DataPathConfig, type(None))),
    )

    if data_path is None:
        data_path = DataPathConfig()

    mlflow.set_tracking_uri(f"sqlite:///{data_path.mlflow_db_path}")

    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found in: '{input_path}'")

    logger.info(f"Loading dataset for Batch Prediction from: {input_path}")
    df = pd.read_csv(input_csv_path)
    logger.info(f"Dataset successfully loaded ({len(df)} observations).")

    y_true = None
    if "class" in df.columns:
        y_true = df["class"].copy()

    try:
        predictor = AstroPredict(
            model_name=model_name,
            model_version=model_version,
        )
        logger.info("Predictor instantiation successful.")
    except Exception as e:
        raise RuntimeError(f"Predictor instantiation failed. Error: {e}") from e

    try:
        restore_logging_after_mlflow()
        df_predicted = predictor.predict(df)
        df_predicted["prediction_timestamp"] = datetime.now().isoformat()
        logger.info("Batch file prediction successful.")
    except Exception as e:
        raise RuntimeError(f"Unable to perform prediction on batch file datapoints. Error: {e}") from e

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_predicted.to_csv(out_path, index=False)
        logger.info(f"Predizioni salvate con successo in: '{out_path}'")

    if y_true is not None:
        logger.info("Report di Classificazione sul Batch:")
        print(classification_report(y_true, df_predicted["prediction"]))
        return df_predicted, classification_report(y_true, df_predicted["prediction"], output_dict=True)

    return df_predicted, None


if __name__ == "__main__":
    setup_logger()

    data_path = DataPathConfig()

    _, metrics = run_batch_prediction(
        input_csv_path=data_path.split_production_path,
        output_csv_path=data_path.holdout_dataset_predicted,
    )
