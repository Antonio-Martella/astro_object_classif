import logging
import random
import time
from pathlib import Path

import mlflow
import pandas as pd

from configs.paths import DataPathConfig
from src.predict.predict import AstroPredict
from src.utils.logger import restore_logging_after_mlflow, setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def predict_single_instance(record: pd.DataFrame, predictor: AstroPredict) -> dict:
    """
    It performs inference on a single astronomical record and measures the exact latency.

    Args:
        record (pd.DataFrame): DataFrame containing the single astronomical record
            to be used for the prediction.
        predictor (AstroPredict): A predictor instance containing the trained model
            and the pipeline needed to perform inference.

    Returns:
            dict: Contains 'prediction', 'latency_seconds' and 'true_class' (if present).
    """
    validate_type(record=(record, pd.DataFrame), predictor=(predictor, AstroPredict))
    if len(record) != 1:
        raise ValueError(f"predict_single_instance expects exactly 1 row, got {len(record)}.")

    true_class = record["class"].iloc[0] if "class" in record.columns else None

    start_time = time.perf_counter()
    df_pred = predictor.predict(record)
    end_time = time.perf_counter()

    latency = end_time - start_time
    predicted_class = df_pred["prediction"].iloc[0]

    return {
        "prediction": predicted_class,
        "latency_seconds": latency,
        "true_class": true_class,
    }


def run_simulation_streaming(
    input_csv_path: str | Path,
    model_name: str = "Classification_Astro_Model",
    model_version: str = "latest",
    max_records: int | None = None,
    min_delay: float = 1.0,
    max_delay: float = 2.0,
    data_path: DataPathConfig | None = None,
) -> None:
    """
    Simulates a real-time astronomical telemetry stream.

    Args:
        input_csv_path (str | Path): Path to the CSV file
            containing the astronomical data to be used for the simulation.
        model_name (str): Name of the model registered in MLflow to use
            for inference.
        model_version (str): Version of the model registered in MLflow to
            use. The value "latest" indicates the use of the most recent
            available version.
        max_records (int | None): Maximum number of observations to simulate.
            If set to None, all available observations are processed.
        min_delay (float): Minimum delay, in seconds, between two consecutive
            observations.
        max_delay (float): Maximum delay, in seconds, between two consecutive
            observations.
        data_path (DataPathConfig | None): Configuration containing the paths
            of files and resources used during inference. If None,
            the default configuration is used.

    Returns:
        None: The function runs the simulation and returns no value.
    """
    validate_type(
        input_csv_path=(input_csv_path, (str, Path)),
        model_name=(model_name, str),
        model_version=(model_version, str),
        max_records=(max_records, (int, type(None))),
        min_delay=(min_delay, float),
        max_delay=(max_delay, float),
        data_path=(data_path, (DataPathConfig, type(None))),
    )

    if data_path is None:
        data_path = DataPathConfig()

    mlflow.set_tracking_uri(f"sqlite:///{data_path.mlflow_db_path}")

    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found in: '{input_path}'")

    logger.info(f"Loading dataset for Streaming Prediction: {input_path}")
    df = pd.read_csv(input_csv_path)
    logger.info(f"Dataset successfully loaded ({len(df)} observations).")

    try:
        predictor = AstroPredict(
            model_name=model_name,
            model_version=model_version,
        )
        restore_logging_after_mlflow()
        logger.info("Predictor instantiation successful.")
    except Exception as e:
        raise RuntimeError(f"Predictor instantiation failed. Error: {e}") from e

    logger.info("=" * 60)
    logger.info("STARTING REAL-TIME STREAMING SIMULATION")
    logger.info("=" * 60)

    count = 0
    while True:
        if max_records is not None and count >= max_records:
            logger.info(f"The limit of {max_records} simulated observations has been reached. Stop.")
            break

        time.sleep(random.uniform(min_delay, max_delay))

        record_num = random.randint(0, len(df) - 1)
        record = df.iloc[record_num : record_num + 1]

        try:
            res = predict_single_instance(record=record, predictor=predictor)
            count += 1
            true_info = f" (true: {res['true_class']})" if res["true_class"] else ""
            logger.info(
                f"\033[32m(latency {res['latency_seconds']:.4f}s)\033[0m "
                f"OBS #{record_num}: prediction --> \x1b[31m{res['prediction']}{true_info}\x1b[0m"
            )
        except Exception as e:
            logger.error(f"Errore durante l'elaborazione del record #{record_num}: {e}")


if __name__ == "__main__":
    setup_logger()
    data_path = DataPathConfig()
    run_simulation_streaming(input_csv_path=data_path.split_production_path)
