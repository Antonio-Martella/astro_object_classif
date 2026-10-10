from typing import Any
import json
from pathlib import Path
import joblib
import logging
import mlflow

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder

from configs.paths import DataPathConfig
from configs.pipeline_config import PreprocessingConfig
from configs.pipeline_loader import load_preprocessing_config
from src.training.pipeline import build_training_pipeline
from src.training.data import load_split_and_encode_dataset
from src.utils.metrics import evaluate_classification_metrics
from src.utils.validate_type import validate_type
from src.utils.logger import restore_logging_after_mlflow

logger = logging.getLogger(__name__)

def _log_model_to_mlflow(
    pipeline: ImbPipeline,
    model_name: str,
    metrics: dict[str, Any],
    custom_params: dict[str, Any] | None,
    label_encoder: LabelEncoder,
    config_path: DataPathConfig,
) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{config_path.mlflow_db_path}")
    mlflow.set_experiment("Astro_Object_Classification")

    with mlflow.start_run(run_name=f"Manual_Training_{model_name.upper()}"):
        mlflow.log_param("model_type", model_name)
        if custom_params:
            mlflow.log_params(custom_params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model_pipeline",
            registered_model_name="Custom_Classification_Astro_Model",
        )

        mlflow.log_dict(
            {"classes": label_encoder.classes_.tolist()},
            artifact_file="encoder/label_encoder.json",
        )

    restore_logging_after_mlflow()
    logger.info("Model successfully registered to MLflow as 'Classification_Astro_Model'!")

def run_training(
        model_name: str,
        custom_params: dict[str, Any] | None = None,
        config_path: DataPathConfig | None = None
    ):

    config_path = DataPathConfig() if config_path is None else config_path

    X_train, X_test, y_train_encoded, y_test_encoded, _, _, label_encoder = load_split_and_encode_dataset(save_encoder=True)
    logger.info("Dataset split and encoding completed successfully.")

    if custom_params is None:
            logger.info(
                f"Starting the training phase for model '{model_name}' " \
                "with DEFAULT hyperparameters."
            )
    else:
        logger.info(
            f"Starting the training phase for model '{model_name}' with CUSTOM hyperparameters:"
        )
        print(custom_params)
        for k, v in custom_params.items():
            logger.info(f"   {k} = {v}")

    pipeline, metrics = fit_and_evaluate_model(
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train_encoded=y_train_encoded,
        y_test_encoded=y_test_encoded,
        label_encoder=label_encoder,
    )
    logger.info("Model fitting and evaluation completed successfully.")
    logger.info("Model metrics:")
    for k, v in metrics.items():
        logger.info(f"   {k} = {round(v, 4)}")

    # Save the model and metrics locallys
    Path(config_path.custom_model_pipeline).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config_path.custom_model_pipeline)

    Path(config_path.custom_model_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path.custom_model_metrics, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=4)

    logger.info(
        "Saving metrics and the model locally to: \n" \
        f"  - pipeline model: {config_path.custom_model_pipeline}\n" \
        f"  - metrics report: {config_path.custom_model_metrics}"
    )

    while True:
        answer = input("Do you want to register this model in MLflow as the active model for inference? [y/n]: ")
        if answer == 'n':
            logger.info("The model was saved locally, but it was not saved to MLflow.")
            break
        elif answer == 'y':
            _log_model_to_mlflow(
                pipeline=pipeline,
                model_name=model_name,
                metrics=metrics,
                custom_params=custom_params,
                label_encoder=label_encoder,
                config_path=config_path,
            )
            break
        else:
            logger.warning("You must answer [y/n] to the question above.")
            continue


def fit_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_test_encoded: pd.Series,
    preprocess_config: PreprocessingConfig | None = None,
    custom_params: dict[str, Any] | None = None,
    resampling_strategy: str | None = None,
    scaler_strategy: str = "standard",
    label_encoder: list[str] | LabelEncoder | dict | None = None,
) -> tuple[ImbPipeline, dict[str, Any]]:
    """
    Fits a training pipeline on the provided data and evaluates the trained model on the test set.

    Args:
        model_name (str): Name of the model to use (e.g., "random_forest", "xgboost", etc).
        X_train (pd.Dataframe): Training feature matrix used to fit the pipeline.
        X_test (pd.Dataframe): Test feature matrix used to evaluate the model.
        y_train_encoded (pd.Series) : Target labels for the training set.
        y_test_encoded (pd.Series) : Target labels for the test set.
        preprocess_config (PreprocessingConfig, None) : Optional preprocessing configuration. If None, the default config is loaded.
        custom_params (dict[str, Any], None) : Optional custom hyperparameters for the selected model.
        resampling_strategy (str, None): Optional resampling strategy to apply before model training
            (e.g., "smote", "undersampling").
        scaler_strategy (str) : Strategy for feature scaling (e.g., "standard", "robust").
        label_encoder (list[str], LabelEncoder, dict, None) : Label encoder for the target.

    Returns:
        tuple[ImbPipeline, dict[str, Any]] : A trained pipeline and a dictionary of evaluation metrics computed on the test set.

    Raises:
        ValueError : If the input dimensions do not match, or if the preprocessing,
            resampling, or model initialization fails.

    Example:
        >>> pipeline, metrics = fit_and_evaluate_model(
        ...     model_name="random_forest",
        ...     X_train=X_train,
        ...     X_test=X_test,
        ...     y_train_encoded=y_train,
        ...     y_test_encoded=y_test,
        ...     preprocess_config=None,
        ...     custom_params={"n_estimators": 100},
        ...     resampling_strategy="smote",
        ...     scaler_strategy="standard",
        ... )
        >>> pipeline.fit(X_train, y_train)
    """
    validate_type(
        model_name=(model_name, str),
        X_train=(X_train, pd.DataFrame),
        X_test=(X_test, pd.DataFrame),
        y_train_encoded=(y_train_encoded, pd.Series),
        y_test_encoded=(y_test_encoded, pd.Series),
        preprocess_config=(preprocess_config, (PreprocessingConfig, type(None))),
        custom_params=(custom_params, (dict, type(None))),
        resampling_strategy=(resampling_strategy, (str, type(None))),
        scaler_strategy=(scaler_strategy, str),
        label_encoder=(label_encoder, (list, LabelEncoder, dict, type(None))),
    )

    if len(X_train) != len(y_train_encoded):
        raise ValueError("The number of training samples in X_train and y_train_encoded must be the same.")

    if len(X_test) != len(y_test_encoded):
        raise ValueError("The number of training samples in X_test and y_test_encoded must be the same.")

    if X_train.empty or X_test.empty or y_train_encoded.empty or y_test_encoded.empty:
        raise ValueError("Training and test datasets cannot be empty.")

    if preprocess_config is None:
        preprocess_config = load_preprocessing_config()

    try:
        pipeline = build_training_pipeline(
            preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params
        )
        pipeline.fit(X_train, y_train_encoded)
    except Exception as e:
        raise ValueError(f"Attention: Problem during PIPELINE definition in training pipeline. Error {e}.") from e

    try:
        metrics = evaluate_classification_metrics(X=X_test, y_true=y_test_encoded, model=pipeline, label_encoder=label_encoder)
    except Exception as e:
        raise ValueError(f"Attention: Problem during EVALUATION of the trained model. Error {e}.") from e

    return pipeline, metrics
