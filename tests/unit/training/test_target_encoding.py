import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.training.target_encoding import encode_targets_and_save, save_target_encoder


@pytest.fixture
def dummy_target_dataset():
    y_train = pd.Series(["GALAXY", "STAR", "QSO", "QSO", "GALAXY", "STAR"], name="target")
    y_test = pd.Series(["QSO", "QSO", "STAR", "STAR"], name="target")
    return y_train, y_test


class TestEncodeTargetsAndSave:
    @pytest.mark.parametrize(
        "field_to_break, bad_value",
        [
            ("y_train", "hello"),
            ("y_test", [1, 2, 3]),
            ("save", "not_a_bool"),
            ("path_encoder_save", 42),
        ],
    )
    def test_target_ecoding_typeerror(self, tmp_path, dummy_target_dataset, field_to_break, bad_value):
        y_train, y_test = dummy_target_dataset

        save = True

        path_encoder_save = tmp_path / "encoder.pkl"

        kwargs = {"y_train": y_train, "y_test": y_test, "save": save, "path_encoder_save": path_encoder_save}
        kwargs[field_to_break] = bad_value

        with pytest.raises(TypeError):
            encode_targets_and_save(**kwargs)

    def test_encodes_train_and_test_preserving_indices_and_mapping(self, dummy_target_dataset):
        y_train, y_test = dummy_target_dataset

        y_train_encoded, y_test_encoded, label_encoder = encode_targets_and_save(y_train, y_test)

        y_train_decoded = pd.Series(
            label_encoder.inverse_transform(y_train_encoded), index=y_train.index, name=y_train.name
        )
        y_test_decoded = pd.Series(
            label_encoder.inverse_transform(y_test_encoded), index=y_test.index, name=y_test.name
        )

        assert len(label_encoder.classes_) == len(y_train.unique())
        pd.testing.assert_series_equal(y_train, y_train_decoded)
        pd.testing.assert_series_equal(y_test, y_test_decoded)
        pd.testing.assert_index_equal(y_train_encoded.index, y_train.index)
        pd.testing.assert_index_equal(y_test_encoded.index, y_test.index)

    def test_encode_targets_raises_value_error_when_path_is_none(self, dummy_target_dataset):
        y_train, y_test = dummy_target_dataset

        with pytest.raises(ValueError, match="it is not possible to save the target encoder locally"):
            encode_targets_and_save(y_train, y_test, save=True, path_encoder_save=None)

    def test_saves_encoder_when_save_is_true(self, tmp_path, dummy_target_dataset):
        path_encoder_save = tmp_path / "target_encoder.pkl"
        y_train, y_test = dummy_target_dataset

        y_train_encoded, y_test_encoded, label_encoder = encode_targets_and_save(
            y_train, y_test, save=True, path_encoder_save=path_encoder_save
        )

        with open(path_encoder_save, "rb") as file:
            label_encoder_load = joblib.load(file)

        pd.testing.assert_series_equal(pd.Series(label_encoder_load.transform(y_train)), y_train_encoded)
        pd.testing.assert_series_equal(pd.Series(label_encoder_load.transform(y_test)), y_test_encoded)

        pd.testing.assert_series_equal(
            pd.Series(label_encoder_load.inverse_transform(y_train_encoded)),
            pd.Series(label_encoder.inverse_transform(y_train_encoded)),
        )
        pd.testing.assert_series_equal(
            pd.Series(label_encoder_load.inverse_transform(y_test_encoded)),
            pd.Series(label_encoder.inverse_transform(y_test_encoded)),
        )

    def test_raises_error_on_unseen_labels_in_test(self, dummy_target_dataset):
        y_train, y_test = dummy_target_dataset
        y_test[len(y_test)] = "SUPERNOVA"

        with pytest.raises(
            ValueError, match="Attention: Some labels in the test set are not present in the training set"
        ):
            encode_targets_and_save(y_train, y_test)


class TestSaveTargetEncoder:
    @pytest.mark.parametrize("field_to_break, bad_value", [("encoder", [1, 2, 3]), ("path", 1234)])
    def test_save_ecoder_typeerror(self, tmp_path, field_to_break, bad_value):
        path = tmp_path / "encoder.pkl"
        encoder = LabelEncoder()
        kwargs = {"encoder": encoder, "path": path}
        kwargs[field_to_break] = bad_value

        with pytest.raises(TypeError):
            save_target_encoder(**kwargs)

    def test_saves_encoder_and_creates_nested_directories(self, tmp_path):
        nested_path = tmp_path / "nested" / "dir" / "encoder.pkl"

        encoder = LabelEncoder()
        encoder.fit(["A", "B"])

        save_target_encoder(encoder, nested_path)

        assert nested_path.exists()
