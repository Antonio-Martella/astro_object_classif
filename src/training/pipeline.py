import logging
import os
from typing import Any

from imblearn.pipeline import Pipeline as ImbPipeline

from configs.schemas import PreprocessingConfig
from configs.schemas_loader import load_preprocessing_config
from src.data.preprocessing import build_stateful_ml_pipeline
from src.data.resampling import ResamplerFactory
from src.models.model_factory import ModelFactory
from src.utils.validate_type import validate_type

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


logger = logging.getLogger(__name__)


def build_training_pipeline(
    preprocess_config: PreprocessingConfig | None,
    model_name: str,
    scaler_strategy: str,
    resampling_strategy: str | None,
    custom_params: dict[str, Any] | None,
) -> ImbPipeline:
    """
    Builds a complete training pipeline with preprocessing, optional resampling, and model training.

    Args:
        - preprocess_config : Configuration for preprocessing steps. If None, loads default config.
        - model_name : Name of the model to use (e.g., "random_forest", "xgboost").
        - scaler_strategy : Strategy for feature scaling (e.g., "standard", "robust").
        - resampling_strategy : Optional resampling strategy (e.g., "smote", "undersampling").
        - custom_params : Optional custom hyperparameters for the model.

    Returns:
        - ImbPipeline : A pipeline ready for training with fit() and predict().

    Raises:
        - ValueError : If any component (preprocessor, resampler, model) fails to initialize.

    Example:
        >>> pipeline = build_training_pipeline(
        ...     preprocess_config=None,
        ...     model_name="random_forest",
        ...     scaler_strategy="standard",
        ...     resampling_strategy="smote",
        ...     custom_params={"n_estimators": 100}
        ... )
        >>> pipeline.fit(X_train, y_train)
    """
    validate_type(
        preprocess_config=(preprocess_config, (PreprocessingConfig, type(None))),
        model_name=(model_name, str),
        scaler_strategy=(scaler_strategy, str),
        resampling_strategy=(resampling_strategy, (str, type(None))),
        custom_params=(custom_params, (dict, type(None))),
    )

    if preprocess_config is None:
        preprocess_config = load_preprocessing_config()

    try:
        preprocessor = build_stateful_ml_pipeline(preprocess_config, scaler_strategy)
        steps = preprocessor.steps.copy()
    except Exception as e:
        raise ValueError(f"Attention: Problem during PREPROCESSOR definition in training pipeline. Error {e}.") from e

    try:
        if resampling_strategy is not None:
            resampler = ResamplerFactory.get_resampler(strategy_name=resampling_strategy)
            if resampler is not None:
                steps.append(("resampler", resampler))
    except Exception as e:
        raise ValueError(f"Attention: Problem during RESAMPLER definition in training pipeline. Error {e}.") from e

    try:
        if custom_params is not None:
            model = ModelFactory.get_model(model_name, **custom_params)
        else:
            model = ModelFactory.get_model(model_name)
        steps.append(("model", model))
    except Exception as e:
        raise ValueError(f"Attention: Problem during MODEL definition in training pipeline. Error {e}.") from e

    logger.info("The training pipeline successfully prepared!")

    return ImbPipeline(steps=steps)
