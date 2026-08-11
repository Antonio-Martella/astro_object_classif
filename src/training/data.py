import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from configs.paths import DataPathConfig
from configs.schemas import SplitTrainingConfig
from configs.schemas_loader import load_split_training_config
from src.data.data_loader import load_and_split_data
from src.training.target_encoding import encode_targets_and_save
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def load_split_and_encode_dataset(
    datapath_config: DataPathConfig | None = None,
    split_config: SplitTrainingConfig | None = None,
    save_encoder: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, LabelEncoder]:
    """
    This function loads the dataset, splits it into training and testing sets, and encodes the target labels.
    It returns the training and testing features, encoded target labels, and the fitted LabelEncoder.
    """
    validate_type(
        datapath_config=(datapath_config, (DataPathConfig, type(None))),
        split_config=(split_config, (SplitTrainingConfig, type(None))),
        save_encoder=(save_encoder, bool),
    )

    if datapath_config is None:
        datapath_config = DataPathConfig()

    if split_config is None:
        split_config = load_split_training_config()

    try:
        X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split_data(datapath_config, split_config)
        y_train_encoded, y_test_encoded, le = encode_targets_and_save(
            y_train, y_test, save_encoder, datapath_config.target_le
        )
    except Exception as e:
        logger.exception("Failed to load, split, or encode dataset.")
        raise RuntimeError(f"Data pipeline orchestration failed: {e}") from e

    logger.info(
        f"Data pipeline completed successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}. "
        f"Encoded target classes: {list(le.classes_)}"
    )

    return X_train, X_test, y_train_encoded, y_test_encoded, groups_train, groups_test, le
