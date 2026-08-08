import yaml

from configs.paths import DataPathConfig
from configs.random_seed_loader import load_random_seed_config

path_config = DataPathConfig()
random_config = load_random_seed_config()


class TuningConfigLoader:
    def __init__(self):
        with open(path_config.optuna_config, "r") as f:
            self.optuna_config = yaml.safe_load(f)["experiment"]

    def get_search_space(self, model_name: str) -> dict:
        if not hasattr(path_config, f"{model_name}_search_space"):
            raise ValueError(f"Nessun file di search space definito per il modello '{model_name}_search_space'. ")

        with open(getattr(path_config, f"{model_name}_search_space"), "r") as f:
            model_config = yaml.safe_load(f)

        search_space = model_config.get("search_space", {})

        search_space["random_state"] = {"type": "fixed", "value": random_config.random_seed_models}
        search_space["resampling_strategy"] = self.optuna_config["resampling_strategy"]
        search_space["scaler_strategy"] = self.optuna_config["scaler_strategy"]

        return search_space


if __name__ == "__main__":
    tuning_config = TuningConfigLoader()
    tuning_config.get_search_space("lightgbm")
    print(tuning_config.optuna_config)
