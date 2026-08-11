import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.validate_type import validate_type


def encode_targets_and_save(
    y_train: pd.Series, y_test: pd.Series, save: bool = False, path_encoder_save: Path | None = None
) -> tuple[pd.Series, pd.Series, LabelEncoder]:
    """
    This function encodes the target labels using a LabelEncoder and optionally saves the encoder to a specified path.
    Args:
        y_train (pd.Series): The training target labels.
        y_test (pd.Series): The testing target labels.
        save (bool): Whether to save the encoder to a file. Default is False.
        path_encoder_save (Path | None): The path where the encoder should be saved. Required if save is True.
    Returns:
        tuple[pd.Series, pd.Series, LabelEncoder]: A tuple containing the encoded training labels,
                                                   encoded testing labels, and the fitted LabelEncoder.
    """
    validate_type(
        y_train=(y_train, pd.Series),
        y_test=(y_test, pd.Series),
        save=(save, bool),
        path_encoder_save=(path_encoder_save, (Path, type(None))),
    )

    if not all(label in y_train.unique() for label in y_test.unique()):
        raise ValueError("Attention: Some labels in the test set are not present in the training set!")

    label_encoder = LabelEncoder()
    y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
    y_test_encoded = pd.Series(label_encoder.transform(y_test), index=y_test.index)

    if save:
        if path_encoder_save is None:
            raise ValueError(
                "Attention: it is not possible to save the target encoder locally. "
                "Please check that the correct save path has been specified."
            )

        save_target_encoder(label_encoder, path_encoder_save)

    return y_train_encoded, y_test_encoded, label_encoder


def save_target_encoder(label_encoder: LabelEncoder, path: Path) -> None:
    """
    Saves the fitted LabelEncoder to a specified path using joblib.
    Args:
        label_encoder (LabelEncoder): The fitted LabelEncoder to be saved.
        path (Path): The path where the encoder should be saved.
    """
    validate_type(
        label_encoder=(label_encoder, LabelEncoder),
        path=(path, Path),
    )
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as file:
        joblib.dump(label_encoder, file)
