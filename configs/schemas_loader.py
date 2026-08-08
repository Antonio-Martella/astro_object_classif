import yaml

from configs.paths import DataPathConfig
from configs.schemas import (
    CleanPreprocessingConfig,
    KaggleConfig,
    PreprocessingConfig,
    SplitHoldoutConfig,
    SplitTrainingConfig,
)

paths_configs = DataPathConfig()

params_path_configs = paths_configs.params_config
randomseed_path_configs = paths_configs.randomseed_config


with open(params_path_configs, "r") as f:
    _raw_config = yaml.safe_load(f)

with open(randomseed_path_configs, "r") as f:
    _raw_random_seed_config = yaml.safe_load(f)


def load_kaggle_config():
    return KaggleConfig(**_raw_config["kaggle"])


def load_split_holdout_config():
    return SplitHoldoutConfig(**_raw_config["split_dataset_holdout"])


def load_split_training_config():
    return SplitTrainingConfig(
        **_raw_config["split_dataset_training"],
        random_seed=_raw_random_seed_config["global_seed"]["random_seed_train_split"],
    )


def load_cleaning_preprocessing_config():
    return CleanPreprocessingConfig(**_raw_config["cleaning_preprocessing"])


def load_preprocessing_config():
    return PreprocessingConfig(**_raw_config["preprocessing"])


if __name__ == "__main__":
    config = load_preprocessing_config()
    print(config)
