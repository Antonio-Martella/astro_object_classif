import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder

from configs.paths import DataPathConfig
from configs.schemas import PreprocessingConfig
from configs.schemas_loader import load_preprocessing_config, load_split_training_config
from src.data.data_loader import load_and_split_data
from src.data.preprocessing import build_stateful_ml_pipeline
from src.data.resampling import ResamplerFactory
from src.models.model_factory import ModelFactory
from src.utils.metrics import evaluate_classification_metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_training_data(datapath_config=None, split_config=None, preprocess_config=None):
    datapath_config = datapath_config or DataPathConfig()
    split_config = split_config or load_split_training_config()
    preprocess_config = preprocess_config or load_preprocessing_config()

    X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split_data(datapath_config, split_config)

    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
    y_test_encoded = pd.Series(le.transform(y_test), index=y_test.index)

    os.makedirs(Path(datapath_config.target_le).parent, exist_ok=True)
    joblib.dump(le, datapath_config.target_le)

    return X_train, X_test, y_train_encoded, y_test_encoded, groups_train, groups_test, le


def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_test_encoded: pd.Series,
    preprocess_config: PreprocessingConfig,
    custom_params: dict[str, Any] | None = None,
    resampling_strategy: str | None = None,
    scaler_strategy: str = "standard",
) -> tuple[ImbPipeline, dict[str, Any]]:
    preprocess_config = preprocess_config or load_preprocessing_config()

    resempler = None
    if resampling_strategy is not None:
        resempler = ResamplerFactory.get_resempler(strategy_name=resampling_strategy)

    if custom_params is not None:
        model = ModelFactory.get_model(model_name, **custom_params)
    else:
        model = ModelFactory.get_model(model_name)

    # Definisco gli step della pipeline da applicare ai dati:
    # - preprocessor
    # - resempling sui dati di test
    # - applicazione del modello di ML
    preprocessor = build_stateful_ml_pipeline(preprocess_config, scaler_strategy)
    steps = preprocessor.steps.copy()
    # Appendo alla pipeline del preprocessor il metodo di resampling scelto e il modello di ML indicato
    if resempler is not None:
        steps.append(("resempler", resempler))
    steps.append(("model", model))

    # Definisco la pipeline con gli steps scritti sopra
    pipeline = ImbPipeline(steps=steps)

    # fit della pipeline
    pipeline.fit(X_train, y_train_encoded)

    return pipeline, evaluate_classification_metrics(pipeline, X_test, y_test_encoded)
