import logging

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from configs.paths import DataPathConfig
from configs.schemas import SplitTrainingConfig
from configs.schemas_loader import load_split_training_config
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def _load_dataset(data_path: DataPathConfig) -> pd.DataFrame:
    validate_type(data_path=(data_path, DataPathConfig))
    return pd.read_csv(data_path.processed_data_path)


def _define_features_target(df: pd.DataFrame, split_config: SplitTrainingConfig) -> tuple[pd.DataFrame, pd.Series]:
    validate_type(df=(df, pd.DataFrame), split_config=(split_config, SplitTrainingConfig))

    X, y = (
        df.drop(columns=[split_config.target_column, split_config.col_group]),
        df[split_config.target_column],
    )

    if X.empty:
        raise ValueError("The features dataset is empty!")

    if y.empty:
        raise ValueError("The target series is empty!")

    return X, y


def _split_by_group(X: pd.DataFrame, y: pd.Series, group_labels: pd.Series, split_config: SplitTrainingConfig):
    validate_type(
        X=(X, pd.DataFrame),
        y=(y, pd.Series),
        group_labels=(group_labels, pd.Series),
        split_config=(split_config, SplitTrainingConfig),
    )

    # let's create a GroupShuffleSplit object to split the dataset into training and testing sets
    # while preserving the group structure defined by the reference column (split_config.col_group).
    gss = GroupShuffleSplit(
        n_splits=1,  # Numeber of re-shuffling & splitting iterations
        train_size=split_config.train_split,  # Proportion of the dataset to include in the train split
        random_state=split_config.random_seed,
    )

    # Generate indices to split the data into training and testing sets.
    split = gss.split(X=X, y=y, groups=group_labels)

    # Use next() to directly extract the only tuple of indices (Train/Test)
    train_idx, test_idx = next(split)

    return train_idx, test_idx


def load_and_split_data(
    data_path: DataPathConfig, split_config: SplitTrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    validate_type(data_path=(data_path, DataPathConfig), split_config=(split_config, SplitTrainingConfig))

    ########################################
    #### 1. Load the dataset from local ####
    ########################################
    try:
        df = _load_dataset(data_path)
    except Exception as e:
        logger.exception("Error loading dataset.")
        raise RuntimeError(f"Dataset loading failed: {e}") from e

    if df.empty:
        raise ValueError("The dataset loaded is empty!")

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
        raise ValueError("Not enough unique groups to perform a split (found {group_labels.nunique()}).")

    try:
        train_idx, test_idx = _split_by_group(X, y, group_labels, split_config)
    except Exception as e:
        logger.exception("Error while splitting training and testing datasets")
        raise RuntimeError(f"Dataset split failed: {e}") from e

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("The split produced an empty train or test set!")

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        group_labels.iloc[train_idx],
        group_labels.iloc[test_idx],
    )


if __name__ == "__main__":
    path_config = DataPathConfig()
    split_config = load_split_training_config()

    X_train, X_test, y_train, y_test, groups_train, group_test = load_and_split_data(path_config, split_config)
