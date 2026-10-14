import logging
import time
from pathlib import Path

import mlflow
import pandas as pd

from configs.paths import DataPathConfig
from src.predict.predictor import AstroPredict
from src.utils.logger import restore_logging_after_mlflow, setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)

def _predict_single_instance(record: pd.DataFrame, predictor: AstroPredict) -> dict:
    """
    Performs inference on a single astronomical record and measures exact latency.
    """
    validate_type(record=(record, pd.DataFrame), predictor=(predictor, AstroPredict))
    if len(record) != 1:
        raise ValueError(f"_predict_single_instance expects exactly 1 row, got {len(record)}.")
    start_time = time.perf_counter()
    df_pred = predictor.predict(record)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    return {
        "prediction": str(df_pred["prediction"].iloc[0]),
        "proba_prediction": float(df_pred["proba_prediction"].iloc[0]),
        "latency_ms": latency_ms,
        "timestamp": str(df_pred["prediction_timestamp"].iloc[0]),
    }


def run_single_prediction(
    predictor: AstroPredict | None = None,
    model_name: str = "Classification_Astro_Model",
    model_version: str = "latest",
    data_path: DataPathConfig | None = None,
) -> dict | None:
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
        predictor=(predictor, (AstroPredict, type(None))),
        model_name=(model_name, str),
        model_version=(model_version, str),
        data_path=(data_path, (DataPathConfig, type(None))),
    )

    data_path = DataPathConfig() if data_path is None else data_path

    if predictor is None:
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

    mlflow.set_tracking_uri(f"sqlite:///{data_path.mlflow_db_path}")

    while True:
        raw_values = input(
            "\nEnter the 6 values ['u', 'g', 'r', 'i', 'z', 'redshift'] separated by comma\n"
            "(e.g. 19.47, 17.04, 15.95, 15.50, 15.23, 0.635): "
        )
        try:
            record_values = [float(x.strip()) for x in raw_values.split(",")]
        except ValueError:
            logger.warning("Invalid input: all 6 values must be numbers (floats). Please try again.")
            continue

        if len(record_values) != 6:
            logger.warning(
                f"Invalid input: expected 6 values ('u', 'g', 'r', 'i', 'z', 'redshift'), got {len(record_values)}."
            )
            continue

        data = {
            "u": record_values[0],
            "g": record_values[1],
            "r": record_values[2],
            "i": record_values[3],
            "z": record_values[4],
            "redshift": record_values[5],
        }
        record = pd.DataFrame(data, index=[0])

        prediction = _predict_single_instance(record, predictor)

        prob = prediction["proba_prediction"]
        filled = int(round(prob * 40))
        bar = "█" * filled + "░" * (40 - filled)

        # Indici di colore astrofisici (uguali a quelli generati dalla pipeline)
        u_g = data["u"] - data["g"]
        g_r = data["g"] - data["r"]
        r_i = data["r"] - data["i"]
        i_z = data["i"] - data["z"]

        line_bands = f"Bands       : u={data['u']:.2f} | g={data['g']:.2f} | r={data['r']:.2f} | i={data['i']:.2f} | z={data['z']:.2f}"
        line_colors = f"Color Index : u-g={u_g:.2f} | g-r={g_r:.2f} | r-i={r_i:.2f} | i-z={i_z:.2f}"
        line_red = f"Redshift    : {data['redshift']:.5f}"
        line_pred = f"Prediction  : {prediction['prediction']}"
        line_conf = f"Confidence  : [{bar}] {prob * 100:.2f}%"
        line_lat = f"Latency     : {prediction['latency_ms']:.2f} ms (Model: {model_name} [{model_version}])"
        line_time = f"Timestamp   : {prediction['timestamp']}"

        logger.info("╭" + "─" * 70 + "╮")
        logger.info("│" + "ASTRONOMICAL OBJECT CLASSIFICATION REPORT".center(70) + "│")
        logger.info("├" + "─" * 70 + "┤")
        logger.info(f"│  {line_bands:<67} │")
        logger.info(f"│  {line_colors:<67} │")
        logger.info(f"│  {line_red:<67} │")
        logger.info("├" + "─" * 70 + "┤")
        logger.info(f"│  {line_pred:<67} │")
        logger.info(f"│  {line_conf:<67} │")
        logger.info(f"│  {line_lat:<67} │")
        logger.info(f"│  {line_time:<67} │")
        logger.info("╰" + "─" * 70 + "╯")

        while True:
            continue_evaluation = input("\nDo you wish to evaluate other records? [y/n]: ").strip().lower()
            if continue_evaluation in ("y", "yes"):
                logger.info("User requested another single-record evaluation.")
                break
            elif continue_evaluation in ("n", "no"):
                logger.info("User ended single-record evaluation session.")
                return
            else:
                logger.warning("Invalid response; please answer with 'y' or 'n'.")
