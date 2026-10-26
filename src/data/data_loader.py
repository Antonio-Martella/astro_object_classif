import logging
from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from configs.paths import DataPathConfig
from configs.pipeline_config import SplitTrainingConfig
from configs.pipeline_loader import load_split_training_config
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def _load_dataset(data_path: DataPathConfig | None = None) -> pd.DataFrame:
    """
    Load the dataset from a CSV file located at the specified path.

    Args:
        data_path (DataPathConfig | None): Configuration object containing the path to the processed data CSV file

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    validate_type(data_path=(data_path, (DataPathConfig, type(None))))

    data_path = DataPathConfig() if data_path is None else data_path

    if not Path(data_path.processed_data).exists():
        raise FileNotFoundError(f"The file {data_path.processed_data} does not exist!")
    else:
        df = pd.read_csv(data_path.processed_data)

    if df.empty:
        raise ValueError("The loaded dataset is empty!")

    validate_type(df=(df, pd.DataFrame))

    return df


def _define_features_target(
    df: pd.DataFrame, split_config: SplitTrainingConfig | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Define the features and target variables from the dataset.

    Args:
        df (pd.DataFrame): The dataset.
        split_config (SplitTrainingConfig | None): Configuration object containing the column names for the target and group variables.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target Series.
    """
    validate_type(df=(df, pd.DataFrame), split_config=(split_config, (SplitTrainingConfig, type(None))))

    split_config = load_split_training_config() if split_config is None else split_config

    if df.empty:
        raise ValueError("Dataset is empty!")

    if not all([feature in df.columns for feature in [split_config.target_column, split_config.col_group]]):
        raise ValueError("Attention 'target' or 'col_group' column is not present in the provided dataset!")

    X, y = (
        df.drop(columns=[split_config.target_column, split_config.col_group]),
        df[split_config.target_column],
    )

    if X.empty:
        raise ValueError("The features dataset is empty!")

    if y.empty or y.isnull().all():
        raise ValueError("The target series is empty or contains only missing values!")

    validate_type(X=(X, pd.DataFrame), y=(y, pd.Series))

    return X, y


def _split_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    group_labels: pd.Series,
    split_config: SplitTrainingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets while preserving the group structure defined by the reference column.

    Args:
        X (pd.DataFrame): The features dataset.
        y (pd.Series): The target series.
        group_labels (pd.Series): The reference column used to define groups for splitting.
        split_config (SplitTrainingConfig | None): Configuration object containing the split parameters.

    Returns:
        tuple: A tuple containing the indices for the training and testing sets.
    """
    validate_type(
        X=(X, pd.DataFrame),
        y=(y, pd.Series),
        group_labels=(group_labels, pd.Series),
        split_config=(split_config, (SplitTrainingConfig, type(None))),
    )

    split_config = load_split_training_config() if split_config is None else split_config

    if X.empty:
        raise ValueError("The features DataFrame is empty!")

    if y.empty:
        raise ValueError("The target Series is empty!")

    if group_labels.empty:
        raise ValueError("The group label Series is empty!")

    if not 0.0 < split_config.train_split < 1.0:
        raise ValueError(f"The fraction of train/test split is {split_config.train_split}, must be between (0.0, 1.0)")

    # let's create a GroupShuffleSplit object to split the dataset into training and testing sets
    # while preserving the group structure defined by the reference column (split_config.col_group).
    gss = GroupShuffleSplit(
        n_splits=1,  # Numeber of re-shuffling & splitting iterations
        train_size=split_config.train_split,  # Proportion of the dataset to include in the train split
        random_state=split_config.random_seed,  # Random seed for reproducibility
    )

    # Generate indices to split the data into training and testing sets.
    split = gss.split(X=X, y=y, groups=group_labels)

    # Use next() to directly extract the only tuple of indices (Train/Test)
    train_idx, test_idx = next(split)

    return train_idx, test_idx


def load_and_split_data(
    data_path: DataPathConfig | None = None,
    split_config: SplitTrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load the dataset from a CSV file and split it into training and testing sets while preserving the group structure.

    Args:
        data_path (DataPathConfig | None): Configuration object containing the path to the processed data CSV file
        split_config (SplitTrainingConfig | None): Configuration object containing the column names for the target and group
            variables, as well as the split parameters.

    Returns:
        tuple: A tuple containing the training and testing features DataFrames, the training and testing target
            Series, and the training and testing group labels Series.
    """
    validate_type(
        data_path=(data_path, (DataPathConfig, type(None))),
        split_config=(split_config, (SplitTrainingConfig, type(None))),
    )

    data_path = DataPathConfig() if data_path is None else data_path
    split_config = load_split_training_config() if split_config is None else split_config

    ########################################
    #### 1. Load the dataset from local ####
    ########################################
    try:
        df = _load_dataset(data_path)
    except Exception as e:
        logger.exception("Error loading dataset.")
        raise RuntimeError(f"Dataset loading failed: {e}") from e

    ##############################################
    #### 2. Definition of features and target ####
    ##############################################
    if split_config.target_column not in df.columns:
        raise ValueError("The target column does not exist in the loaded dataframe.")
    if split_config.col_group not in df.columns:
        raise ValueError("The group column does not exist in the loaded dataframe.")

    try:
        X, y = _define_features_target(df, split_config)
    except Exception as e:
        logger.exception("Error defining features and target.")
        raise RuntimeError(f"Feature/target definition failed: {e}") from e

    if len(X) != len(y):
        raise ValueError("The length of features dataset and target series do not match!")

    #######################################################################
    #### 3. Dataset split by reference column (split_config.col_group) ####
    #######################################################################

    # Define the reference column (default is 'field_ID' in configs/params.yaml) on which to split the dataset
    group_labels = df[split_config.col_group]

    if group_labels.nunique() < 2:
        raise ValueError(f"Not enough unique groups to perform a split (found {group_labels.nunique()}).")

    try:
        train_idx, test_idx = _split_by_group(X, y, group_labels, split_config)
    except Exception as e:
        logger.exception("Error while splitting training and testing datasets")
        raise RuntimeError(f"Dataset split failed: {e}") from e

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        group_labels.iloc[train_idx],
        group_labels.iloc[test_idx],
    )
