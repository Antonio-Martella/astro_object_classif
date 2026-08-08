from dataclasses import dataclass


@dataclass
class RandomSeedConfig:
    random_seed_holdout: int
    random_seed_models: int
    random_seed_train_split: int
    random_seed_resempling: int
    random_seed_sgkf: int
    random_seed_training: int
    random_seed_optuna: int
    random_seed_trainsplit: int
