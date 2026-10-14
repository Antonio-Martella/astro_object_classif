import pytest
from configs.optuna_config import OptunaConfigLoader


class TestOptunaConfigLoader:
    """Suite of unit tests for the OptunaConfigLoader class and HPO search space loading."""
    @pytest.fixture
    def optuna_config_loader(self):
        """Initializes a default instance of Optuna Config Loader in test."""
        return OptunaConfigLoader()

    def test_default_initialization(self, optuna_config_loader):
        """Verify that OptunaConfigLoader initializes correctly using default project paths and seeds."""
        assert optuna_config_loader.path_config is not None
        assert optuna_config_loader.random_seed_config is not None
        assert "search_space" in optuna_config_loader.optuna_config

    @pytest.mark.parametrize("model_name", [
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
        "dense_nn",
        "svc",
        "sgd",
        "logreg",
    ])
    def test_all_registered_models_search_spaces_are_valid(self, model_name, optuna_config_loader):
        """
        Verify that all 9 registered models load a valid search space dictionary.

        Ensures that each model's search space is properly enriched with the global
        model random seed, resampling strategies, and feature scaling options.

        Args:
            model_name (str): Identifier of the candidate model architecture.
        """

        space = optuna_config_loader.get_search_space(model_name)

        assert isinstance(space, dict)
        assert len(space) > 0
        assert "random_state" in space
        assert space["random_state"]["value"] == optuna_config_loader.random_seed_config.random_seed_models
        assert "resampling_strategy" in space
        assert "scaler_strategy" in space

    def test_unknown_model_raises_value_error(self, optuna_config_loader):
        """Verify that requesting an unregistered or invalid model name raises a ValueError."""
    
        with pytest.raises(ValueError, match="No search space file defined for the model"):
            optuna_config_loader.get_search_space("unknown_invented_model")