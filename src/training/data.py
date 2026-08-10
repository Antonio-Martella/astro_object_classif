import pandas as pd
from sklearn.preprocessing import LabelEncoder

from configs.paths import DataPathConfig
from configs.schemas import SplitTrainingConfig
from configs.schemas_loader import load_split_training_config
from src.data.data_loader import load_and_split_data
from src.training.target_encoding import encode_targets_and_save


def load_split_and_encode_dataset(
    datapath_config: DataPathConfig | None = None,
    split_config: SplitTrainingConfig | None = None,
    save_encoder: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, LabelEncoder]:
    """
    This function loads the dataset, splits it into training and testing sets, and encodes the target labels.
    It returns the training and testing features, encoded target labels, and the fitted LabelEncoder.
    """
    if datapath_config is None:
        datapath_config = DataPathConfig()

    if split_config is None:
        split_config = load_split_training_config()

    X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split_data(datapath_config, split_config)
    y_train_encoded, y_test_encoded, le = encode_targets_and_save(
        y_train, y_test, save_encoder, datapath_config.target_le
    )

    return X_train, X_test, y_train_encoded, y_test_encoded, groups_train, groups_test, le
