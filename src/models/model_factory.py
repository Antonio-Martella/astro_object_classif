import warnings
from typing import Any, cast
import inspect

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
    def get_valid_params(cls, model_name: str) -> set[str]:
        """Restituisce il set dei parametri supportati dal modello."""
        if model_name not in cls._registry:
            raise ValueError(f"Model '{model_name}' is not supported.")
        # Per la rete neurale elenchiamo i parametri gestiti dal suo costruttore
        if model_name == "dense_nn":
            return {
                "optimizer", "num_layer", "num_perceptron", "drop_out",
                "batch_norm", "learning_rate", "hidden_act_func", "act_func",
                "class_weights", "verbose", "random_seed", "batch_size",
                "metric_monitor_callback", "factor_reducer_callback",
                "patience_lrreducer", "patience_earlystop", "verbose_cb",
            }
        
        model_class = cls._registry[model_name]
        
        temp_instance = model_class()
        underlying = getattr(temp_instance, "model", None)
        
        if underlying is not None and hasattr(underlying, "get_params"):
            return set(underlying.get_params().keys())
        
        sig = inspect.signature(model_class.__init__)
        return {p for p in sig.parameters.keys() if p not in ("self", "kwargs")}

    @classmethod
    def get_model(cls, model_name: str, **kwargs: Any) -> BaseModel:
        if model_name not in cls._registry:
            valid_models = list(cls._registry.keys())
            raise ValueError(
                f"Model '{model_name}' is not supported. Valid models are: {valid_models}"
            )
        
        if kwargs:
            valid_params = cls.get_valid_params(model_name)
            invalid_keys = set(kwargs.keys()) - valid_params
            if invalid_keys:
                raise ValueError(
                    f"Invalid hyperparameter(s) for model '{model_name}': {sorted(invalid_keys)}.\n"
                    f"Valid hyperparameters are: {sorted(valid_params)}"
                )
            model_class = cls._registry[model_name]
            return cast(BaseModel, model_class(**kwargs))
        return cast(BaseModel, cls._registry[model_name]())
