from dataclasses import dataclass


@dataclass
class KaggleConfig:
    dataset_path: str
    file_name: str
    dataset_columns: list
    dataset_length: int


@dataclass
class SplitHoldoutConfig:
    daily_split_batch_size: int
    holdout_split: float
    ref_col: str


@dataclass
class SplitTrainingConfig:
    random_seed: int
    col_group: str
    train_split: float
    target_column: str


@dataclass
class CleanPreprocessingConfig:
    anomaly_values: list
    new_features: list
    columns_to_drop: list


@dataclass
class PreprocessingConfig:
    columns_to_scale: list
