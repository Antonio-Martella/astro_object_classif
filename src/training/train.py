from typing import Any

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from configs.schemas import PreprocessingConfig
from configs.schemas_loader import load_preprocessing_config
from src.training.pipeline import build_training_pipeline
from src.utils.metrics import evaluate_classification_metrics
from src.utils.validate_type import validate_type


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
) -> tuple[ImbPipeline, dict[str, Any]]:
    """
    Fits a training pipeline on the provided data and evaluates the trained model on the test set.

    Args:
        model_name : Name of the model to use (e.g., "random_forest", "xgboost", etc).
        X_train : Training feature matrix used to fit the pipeline.
        X_test : Test feature matrix used to evaluate the model.
        y_train_encoded : Target labels for the training set.
        y_test_encoded : Target labels for the test set.
        preprocess_config : Optional preprocessing configuration. If None, the default config is loaded.
        custom_params : Optional custom hyperparameters for the selected model.
        resampling_strategy : Optional resampling strategy to apply before model training
            (e.g., "smote", "undersampling").
        scaler_strategy : Strategy for feature scaling (e.g., "standard", "robust").

    Returns:
        tuple[ImbPipeline, dict[str, Any]] : A trained pipeline and a dictionary of evaluation metrics computed on the
        test set.

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
        metrics = evaluate_classification_metrics(pipeline, X_test, y_test_encoded)
    except Exception as e:
        raise ValueError(f"Attention: Problem during EVALUATION of the trained model. Error {e}.") from e

    return pipeline, metrics
