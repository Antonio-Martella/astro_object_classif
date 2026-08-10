from typing import Any

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from configs.schemas import PreprocessingConfig
from configs.schemas_loader import load_preprocessing_config
from src.training.pipeline import build_training_pipeline
from src.utils.metrics import evaluate_classification_metrics


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
    This function trains a model using the specified training data and evaluates it on the test data.
    It returns the trained pipeline and a dictionary of evaluation metrics.
    """
    if preprocess_config is None:
        preprocess_config = load_preprocessing_config()

    pipeline = build_training_pipeline(
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params
    )
    pipeline.fit(X_train, y_train_encoded)

    return pipeline, evaluate_classification_metrics(pipeline, X_test, y_test_encoded)
