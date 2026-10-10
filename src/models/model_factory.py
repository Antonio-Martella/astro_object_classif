import warnings
from typing import Any, cast

from src.models import (
    CatBoostModel,
    DenseNNModel,
    LightGBMModel,
    LogRegModel,
    RandomForestModel,
    ExtraTreesModel,
    SGDModel,
    SVCModel,
    XGBoostModel,
)
from src.models.base_model import BaseModel

warnings.filterwarnings("ignore", category=FutureWarning)


class ModelFactory:
    _registry: dict[str, type[BaseModel]] = {
        "random_forest": RandomForestModel,
        "extra_trees": ExtraTreesModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "logreg": LogRegModel,
        "sgd": SGDModel,
        "svc": SVCModel,
        "dense_nn": DenseNNModel,
    }

    @classmethod
    def get_model(cls, model_name: str, **kwargs: Any) -> BaseModel:
        if model_name not in cls._registry:
            valid_models = list(cls._registry.keys())
            raise ValueError(f"Model '{model_name}' is not supported. Valid models are:'{valid_models}'")

        model_class = cls._registry[model_name]

        if kwargs:
            return cast(BaseModel, model_class(**kwargs))
        return cast(BaseModel, model_class())
