import yaml

from configs.paths import DataPathConfig
from configs.pipeline_config import RandomSeedConfig
from configs.pipeline_loader import load_random_seed_config
from src.utils.validate_type import validate_type


class OptunaConfigLoader:
    """
    This class is responsible for loading the Optuna configuration from a YAML file. 
    It provides methods to access the configuration parameters and validate their types.
    """
    def __init__(
            self,
            path_config: DataPathConfig | None = None,
            random_seed_config: RandomSeedConfig | None = None,
        ) -> None:
        """
        Initialize the OptunaConfigLoader.

        Args:
            path_config (DataPathConfig | None): The data path configuration.
            random_seed_config (RandomSeedConfig | None): The random seed configuration.
        """
        validate_type(
            path_config=(path_config, (DataPathConfig, type(None))),
            random_seed_config=(random_seed_config, (RandomSeedConfig, type(None))),
        )
        
        self.path_config = DataPathConfig() if path_config is None else path_config
        self.random_seed_config = load_random_seed_config() if random_seed_config is None else random_seed_config

        # Load the Optuna configuration from the specified YAML file.
        with open(self.path_config.optuna_config, "r") as f:
            self.optuna_config = yaml.safe_load(f)


    def get_search_space(self, model_name: str) -> dict:
        """
        Get the search space for a specific model.

        Args:
            model_name (str): The name of the model.

        Returns:
            dict: The search space configuration.
        """
        validate_type(model_name=(model_name, str))

        # Check if the search space file for the specified model exists in the path configuration.
        if not hasattr(self.path_config, f"{model_name}_search_space"):
            raise ValueError(f"No search space file defined for the model '{model_name}'. ")

        # Load the search space configuration for the specified model from the corresponding YAML file.
        with open(getattr(self.path_config, f"{model_name}_search_space"), "r") as f:
            model_config = yaml.safe_load(f)

        # Process the search space configuration.
        search_space = model_config.get("search_space", {})

        # Add additional parameters to the search space from the Optuna configuration and random seed configuration.
        search_space["random_state"] = {"type": "fixed", "value": self.random_seed_config.random_seed_models}
        search_space["resampling_strategy"] = self.optuna_config["search_space"]["resampling_strategy"]
        search_space["scaler_strategy"] = self.optuna_config["search_space"]["scaler_strategy"]

        return search_space