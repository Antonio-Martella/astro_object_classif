import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import classification_report

from configs.paths import DataPathConfig
from src.predict.predictor import AstroPredict
from src.utils.logger import restore_logging_after_mlflow, setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def batch_dataset_prediction(
    df: pd.DataFrame,
    batch_size: int,
    predictor: AstroPredict,
    output_csv_path: str | Path | None,
    metrics_batch_path: str | Path | None,
    save_individual_batches: bool = True,
) -> tuple[pd.DataFrame, dict | None]:
    validate_type(
        df=(df, pd.DataFrame),
        batch_size=(batch_size, int),
        predictor=(predictor, AstroPredict),
        output_csv_path=(output_csv_path, (str, Path, type(None))),
        metrics_batch_path=(metrics_batch_path, (str, Path, type(None))),
        save_individual_batches=(save_individual_batches, bool),
    )

    y_true = None
    if "class" in df.columns:
        y_true = df["class"].copy()

    out_path = Path(output_csv_path) if output_csv_path is not None else None
    metrics_path = Path(metrics_batch_path) if metrics_batch_path is not None else None

    metrics_batch = {}
    batch_id = 0
    list_predicted = []

    while len(df) > batch_id * batch_size:
        df_batch = df.iloc[batch_id * batch_size : (batch_id + 1) * batch_size]

        try:
            df_batch_predict = predictor.predict(df_batch)
            df_batch_predict["batch_id"] = f"batch_{batch_id}"
            list_predicted.append(df_batch_predict)
        except Exception as e:
            raise RuntimeError(f"Unable to perform prediction on batch file datapoints. Error: {e}") from e

        if output_csv_path is not None and save_individual_batches:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path_batch = out_path.with_stem(f"{out_path.stem}_{batch_id}")
            df_batch_predict.to_csv(out_path_batch, index=False)

        if y_true is not None:
            y_true_batch = y_true.iloc[batch_id * batch_size : (batch_id + 1) * batch_size]
            metrics_batch[f"batch_id_{batch_id}"] = classification_report(
                y_true_batch,
                df_batch_predict["prediction"],
                digits=4,
                output_dict=True,
            )

        batch_id += 1

    logger.info("Successful batch predictions.")
    if output_csv_path is not None:
        logger.info(f"Predictions successfully saved in: '{out_path.parent}'")

    df_all_predicted = pd.concat(list_predicted, ignore_index=True)

    if y_true is not None and metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as file:
            json.dump(metrics_batch, file, indent=4)
        logger.info(f"Batch metrics report successfully generated in: '{metrics_path.parent}'")
        return df_all_predicted, metrics_batch

    return df_all_predicted, None


def full_dataset_predict(
    df: pd.DataFrame,
    predictor: AstroPredict,
    output_csv_path: str | Path | None,
    metrics_path: str | Path | None,
) -> tuple[pd.DataFrame, dict | None]:
    validate_type(
        df=(df, pd.DataFrame),
        predictor=(predictor, AstroPredict),
        output_csv_path=(output_csv_path, (str, Path, type(None))),
        metrics_path=(metrics_path, (str, Path, type(None))),
    )
    print(output_csv_path)
    y_true = None
    if "class" in df.columns:
        y_true = df["class"].copy()

    try:
        df_predicted = predictor.predict(df)
        logger.info("Batch file prediction successful.")
    except Exception as e:
        raise RuntimeError(f"Unable to perform prediction on batch file datapoints. Error: {e}") from e

    out_path = Path(output_csv_path) if output_csv_path is not None else None
    metrics_path = Path(metrics_path) if metrics_path is not None else None

    if output_csv_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_predicted.to_csv(out_path, index=False)
        logger.info(f"Predictions successfully saved in: '{out_path}'")

    if y_true is not None and metrics_path is not None:
        metrics = classification_report(y_true, df_predicted["prediction"], digits=4, output_dict=True)
        logger.info("Report di Classificazione sul Batch:")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)
        return df_predicted, metrics

    return df_predicted, None



def _single_batch_prediction(
        predictor: AstroPredict,
        input_csv_path: Path,
        input_dir_path: Path,
    ) -> None:
    """
    """
    logger.info("Let's start predicting the provided batch file...")
    logger.info(f"Loading dataset for Batch Prediction from: {input_csv_path}")

    # Define the path where the prediction file will be saved, if not already provided.
    output_csv_path = input_dir_path / "prediction" / \
        f"prediction_{str(input_csv_path.relative_to(input_dir_path))}" 
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Define the path where the dataset metrics will be saved,
    # provided the dataset contains the 'target' column.
    output_metrics_path = Path(output_csv_path).parent / \
        f"metrics_{str(Path(input_csv_path).relative_to(input_dir_path)).removesuffix('.csv')}.json" \

    # Load the dataset
    df = pd.read_csv(input_csv_path)
    logger.info(f"Dataset successfully loaded ({len(df)} observations).")

    # Default target to None; if the target is present in the dataset, copy it.
    y_true = None
    if "class" in df.columns:
        y_true = df["class"].copy()

    # Starting the prediction
    try:
        df_predicted = predictor.predict(df)
        logger.info("Batch file prediction successful.")
    except Exception as e:
        raise RuntimeError(f"Unable to perform prediction on batch file datapoints. Error: {e}") from e

    # Save the dataset with the predictions and print the path
    df_predicted.to_csv(output_csv_path, index=False)
    logger.info(f"Predictions successfully saved in: '{output_csv_path}'")

    # If y_true are present, metric evaluation for the file is performed, and finally, we save.
    if y_true is not None:
        metrics = classification_report(y_true, df_predicted["prediction"], digits=4, output_dict=True)
        with open(output_metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Report on saved metrics in: {output_metrics_path}")


def _dir_batch_prediction(
        predictor: AstroPredict,
        input_dir_path: Path,
    ):

    csv_files = [csv_file for csv_file in os.listdir(input_dir_path) if '.csv' in csv_file]

    for csv_file in csv_files:
        dataset_path = input_dir_path / f"{csv_file}"
        _single_batch_prediction(
            predictor=predictor,
            input_csv_path=dataset_path,
            input_dir_path = input_dir_path,
        )

def run_prediction(
    input_path: str | Path | list,
    output_path: str | Path | None = None,
    output_metrics_path: str | Path | None = None,
    model_name: str = "Classification_Astro_Model",
    model_version: str = "latest",
    config_path: DataPathConfig | None = None,
    save_individual_batches: bool = False,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    validate_type(
        input_path=(input_path, (str, Path)),
        output_path=(output_path, (str, Path, type(None))),
        output_metrics_path=(output_metrics_path, (str, Path, type(None))),
        model_name=(model_name, str),
        model_version=(model_version, str),
        config_path=(config_path, (DataPathConfig, type(None))),
        save_individual_batches=(save_individual_batches, bool),
    )

    # Path configuration
    config_path = DataPathConfig() if config_path is None else config_path

    # 
    mlflow.set_tracking_uri(f"sqlite:///{config_path.mlflow_db_path}")
    restore_logging_after_mlflow()

    # Existence of the input path
    if not Path(input_path).exists():
        raise FileNotFoundError(f"File or directory does not exist in: {Path(input_path)}")

    # Load the predictor
    try:
        predictor = AstroPredict(
            model_name=model_name,
            model_version=model_version,
        )
        restore_logging_after_mlflow()
        logger.info("Predictor instantiation successful.")
    except Exception as e:
        raise RuntimeError(f"Predictor instantiation failed. Error: {e}") from e
    
    # The path of the provided dataset directory is extracted.
    if Path(input_path).is_dir():
        input_dir_path = Path(input_path)
        output_path_dir = input_dir_path / "prediction" 

        _dir_batch_prediction(
            predictor,
            input_dir_path,
        )


    elif Path(input_path).is_file():
        # Verifico che effettivamente il file passato sia in formatao '.csv'
        if not Path(input_path).suffix.lower() == '.csv':
            raise ValueError(f"The provided dataset file must be in '.csv' format.")

        # Memorizzo il path della directory di input
        input_dir_path = Path(input_path).parent

        try:
            _single_batch_prediction(
                predictor,
                Path(input_path), 
                input_dir_path,
            )
            logger.info("Batch prediction completed.")
        except Exception as e:
            raise ValueError(f"Error during single-batch prediction: {e}") from e
    else:
        raise TypeError(f"The provided path is neither a file nor a directory: {Path(input_path)}")


if __name__ == "__main__":
    setup_logger()

    data_path = DataPathConfig()

    df_predicted, metrics_full = run_prediction(
        input_csv_path=data_path.split_production_path,
        output_csv_path=data_path.holdout_dataset_predicted,
        output_metrics_path=data_path.holdout_dataset_metrics,
        batch_size=None,
    )
    print(df_predicted.info())
