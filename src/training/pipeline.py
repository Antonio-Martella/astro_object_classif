import os
from typing import Any

from imblearn.pipeline import Pipeline as ImbPipeline

from configs.schemas import PreprocessingConfig
from src.data.preprocessing import build_stateful_ml_pipeline
from src.data.resampling import ResamplerFactory
from src.models.model_factory import ModelFactory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_training_pipeline(
    preprocess_config: PreprocessingConfig,
    model_name: str,
    scaler_strategy: str,
    resampling_strategy: str | None,
    custom_params: dict[str, Any] | None,
) -> ImbPipeline:
    """
    This function builds a complete training pipeline that includes preprocessing, optional resampling,
    and model training. It returns an ImbPipeline object that can be used for training and evaluation.
    """
    preprocessor = build_stateful_ml_pipeline(preprocess_config, scaler_strategy)
    steps = preprocessor.steps.copy()

    if resampling_strategy is not None:
        resampler = ResamplerFactory.get_resampler(strategy_name=resampling_strategy)
        if resampler is not None:
            steps.append(("resampler", resampler))

    if custom_params is not None:
        model = ModelFactory.get_model(model_name, **custom_params)
    else:
        model = ModelFactory.get_model(model_name)
    steps.append(("model", model))

    return ImbPipeline(steps=steps)
