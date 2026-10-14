import logging
import os
import json
from typing import Any
from pathlib import Path

from configs.paths import DataPathConfig
from src.predict.predictor import AstroPredict
from src.monitoring.retraining_policy import AstroRetrainingPolicy
from src.monitoring.performance_tracker import AstroPerformanceTracker
from src.monitoring.drift_detector import AstroDriftDetector
from src.utils.logger import setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def monitoring_pipeline(
        batch_path: str | Path, 
        past_report_path: str | Path,
        path_config: DataPathConfig | None = None,
        predictor: AstroPredict | None = None,
        astro_datadrift: AstroDriftDetector | None = None,
        astro_performance: AstroPerformanceTracker | None = None,
        verbose: int = 2,
    ):
    """
    Monitoring pipeline that evaluates the performance and drift of a model on a given batch of data.

    Args:
        batch_path (str | Path): Path to the batch of data to be evaluated.
        past_report_path (str | Path): Path to the past report file.
        path_config (DataPathConfig | None): Configuration for data paths. If None, a
            default configuration will be used.
        predictor (AstroPredict | None): The predictor object. If None, a default predictor will be used.
        astro_datadrift (AstroDriftDetector | None): The drift detector object. If None, 
            a default drift detector will be used.
        astro_performance (AstroPerformanceTracker | None): The performance tracker object. If None, 
            a default performance tracker will be used.
        verbose (int): Verbosity level for logging. Default is 0: 
            - 0 for logging all messages
            - 1 for minimal logging
            - 2 for no logging
    """

    validate_type(
        batch_path=(batch_path, (str, Path)),
        past_report_path=(past_report_path, (str, Path)),
        path_config=(path_config, (DataPathConfig, type(None))),
        predictor=(predictor, (AstroPredict, type(None))),
        astro_datadrift=(astro_datadrift, (AstroDriftDetector, type(None))),
        astro_performance=(astro_performance, (AstroPerformanceTracker, type(None))),
        verbose=(verbose, int),
    )

    if verbose == 1:
        logger.setLevel(logging.WARNING)
    elif verbose == 2:
        logger.setLevel(logging.ERROR)

    # paths configuration
    path_config = DataPathConfig() if path_config is None else path_config

    # Load the past daily report if it exists, otherwise initialize an empty report
    if (
        Path(past_report_path).exists() and 
        Path(past_report_path).stat().st_size > 0
    ):
        with open(past_report_path, "r", encoding="utf-8") as file:
            past_report = json.load(file)
    else:
        Path(past_report_path).parent.mkdir(parents=True, exist_ok=True)
        past_report: dict[str, dict[str, dict[str, Any]]] = {}

    # definition of the predictor
    astro_predictor = AstroPredict() if predictor is None else predictor

    # Initialize the retraining policy
    retraining_policy = AstroRetrainingPolicy(verbose=verbose)

    # Extract the batch file name
    name_file = Path(batch_path).name

    # DRIFT
    astro_datadrift = \
        AstroDriftDetector(predictor=astro_predictor, train_dataset_path=path_config.processed_data) \
            if astro_datadrift is None else astro_datadrift
    daily_drift = astro_datadrift.detect_data_drift(dataset_path=batch_path)

    # PERFORMANCE
    astro_performance = \
        AstroPerformanceTracker(predictor=astro_predictor) if astro_performance is None else astro_performance
    daily_performance = astro_performance.evaluate_single_batches(path_dataset=batch_path)

    updated_past_report = retraining_policy.evaluate(
        name_file=name_file,
        daily_drift=daily_drift,
        daily_performance=daily_performance,
        past_report=past_report,
    )

    with open(past_report_path, "w", encoding="utf-8") as file:
        json.dump(updated_past_report, file, indent=4, ensure_ascii=False)

    logger.info(
        f"{Path(batch_path).name} --> [Perf: {updated_past_report[name_file]['performance']['reason']}] | "
        f"[Drift: {updated_past_report[name_file]['drift']['reason']}] | "
        f"[Decision: {updated_past_report[name_file]['decision']['action']}]"
    )

    return updated_past_report


def run_monitoring(batch_file: list[str] | None = None) -> tuple[bool, str]:
    """
    Run the monitoring pipeline for a list of batch files or for all batch files if none are specified.

    Args:
        batch_file (list[str] | None): List of batch file names to be processed. If None, all batch files will be processed.

    Returns:
        bool: Decision on retraining according to the given monitoring conditions.
        str: filename that triggered the retrigger; empty string if the trigger does not fire.
    """
    validate_type(batch_file=(batch_file, (list, type(None))))

    path_config = DataPathConfig()

    # Instantiate the predictor.
    astro_predictor = AstroPredict()

    # Instantiate the drift detector and performance tracker.
    astro_datadrift = AstroDriftDetector(
        predictor=astro_predictor, 
        train_dataset_path=path_config.processed_data,
    )

    # Instantiate the performance tracker.
    astro_performance = AstroPerformanceTracker(
        predictor=astro_predictor, 
    )

    # If batch_file is provided, use it; otherwise, load the list of batch files from the metadata.
    if batch_file is not None:
        files = batch_file
    else:
        with open(path_config.split_daily_batches_metadata_path, "r", encoding="utf-8") as file:
            files = json.load(file)["path_batch_files"]

    # Run the monitoring pipeline for each batch file.
    for file in files:
        report = monitoring_pipeline(
            batch_path=path_config.split_daily_batches_path / file,
            past_report_path=path_config.retraining_decision_report,
            path_config=path_config,
            predictor=astro_predictor,
            astro_datadrift=astro_datadrift,
            astro_performance=astro_performance,
            verbose=0
        )

        if report[file]["decision"]["retraining"]:
            break
    
    if report[file]["decision"]["retraining"]:
        return True, file
    else:
        return False, ""
