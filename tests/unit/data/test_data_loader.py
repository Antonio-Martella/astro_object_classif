from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import pytest

from configs.paths import DataPathConfig
from configs.pipeline_config import SplitTrainingConfig
from src.data.data_loader import (
    _define_features_target,
    _load_dataset,
    _split_by_group,
    load_and_split_data,
)


@pytest.fixture
def dummy_dataset():
    """
    Dummy dataset fixture for testing.
    """
    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_2": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "class": [0, 1, 2, 1, 2, 1, 0, 0, 2, 1],
            "field_ID": [152, 153, 153, 151, 150, 149, 153, 152, 160, 149],
        }
    )


@pytest.fixture
def split_train_config():
    return SplitTrainingConfig(col_group="field_ID", train_split=0.6, target_column="class", random_seed=42)


class TestLoadAndSplitDataLoading:
    def test_invalid_data_path_raises_typeerror(self):
        """
        This test ensures that _load_dataset raises a TypeError when the data path is of an invalid type.
        """
        with pytest.raises(TypeError):
            _load_dataset("unknown_path")

    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        """
        This test ensures that _load_dataset raises an error when the specified CSV file does not exist.
        """
        data_path = DataPathConfig(processed_data=tmp_path / "unknown_file.csv")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            _load_dataset(data_path)

    def test_empty_dataset_raises_valueerror(self, tmp_path):
        """
        This test ensures that _load_dataset raises a ValueError when the loaded dataset is empty.
        """
        path = tmp_path / "processed_data.csv"
        data_path = DataPathConfig(processed_data=path)

        path.touch()

        pd.DataFrame({}).to_csv(data_path.processed_data)

        with pytest.raises(ValueError, match="The loaded dataset is empty"):
            _load_dataset(data_path)

    def test_load_dataset_without_argument(self, tmp_path, monkeypatch, dummy_dataset):
        """
        This test ensures that _load_dataset correctly loads a DataFrame from a CSV file
        when no argument is provided, using the default DataPathConfig.
        """
        csv_path = tmp_path / "processed_data.csv"
        dummy_dataset.to_csv(csv_path, index=False)

        monkeypatch.setattr(
            DataPathConfig,
            "__init__",
            lambda self, *args, **kwargs: object.__setattr__(self, "processed_data", csv_path),
        )
        result = _load_dataset()

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, dummy_dataset)

    def test_load_dataset_success(self, tmp_path, dummy_dataset):
        """
        This test ensures that _load_dataset correctly loads a DataFrame from a CSV file.
        """
        data_path = DataPathConfig(processed_data=tmp_path / "processed_data.csv")

        dummy_dataset.to_csv(data_path.processed_data, index=False)
        expected_df = pd.read_csv(data_path.processed_data)

        result = _load_dataset(data_path)

        pd.testing.assert_frame_equal(result, expected_df)


class TestDefineFeaturesTarget:
    @pytest.mark.parametrize(
        "input, bad_value",
        [
            ("df", pd.Series([1, 2, 3], name="feature_1")),
            ("split_config", DataPathConfig()),
        ],
    )
    def test_invalid_input_raises_typeerror(self, dummy_dataset, split_train_config, input, bad_value):
        """
        Test that _define_features_target raises a TypeError when provided with invalid input types.
        """
        kwargs = {
            "df": dummy_dataset,
            "split_config": split_train_config,
        }

        kwargs[input] = bad_value

        with pytest.raises(TypeError):
            _define_features_target(**kwargs)

    def test_empty_dataset_raises_valueerror(self, split_train_config):
        """
        Test that _define_features_target raises a ValueError when provided with an empty dataset.
        """
        df_empty = pd.DataFrame()
        with pytest.raises(ValueError, match="Dataset is empty"):
            _define_features_target(df_empty, split_train_config)

    @pytest.mark.parametrize(
        "col_name, bad_value",
        [
            ("target_column", "wrong_name"),
            ("col_group", "wrong_name"),
        ],
    )
    def test_wrong_column_name_passed_raises_valueerror(self, dummy_dataset, split_train_config, col_name, bad_value):
        """
        Test that _define_features_target raises a ValueError when provided with a column name
        that is not present in the dataset.
        """
        bad_config = replace(split_train_config, **{col_name: bad_value})
        with pytest.raises(ValueError, match="column is not present in the provided dataset"):
            _define_features_target(dummy_dataset, bad_config)

    def test_empty_features_dataset_raises_valueerror(self, dummy_dataset, split_train_config):
        """
        Test that _define_features_target raises a ValueError when the features dataset is empty
        after dropping the target and group columns.
        """
        dummy_dataset = dummy_dataset.drop(columns=["feature_1", "feature_2"])

        with pytest.raises(ValueError, match="features dataset"):
            _define_features_target(dummy_dataset, split_train_config)

    def test_target_with_only_missing_values_raises_valueerror(self, dummy_dataset, split_train_config):
        """
        Test that _define_features_target raises a ValueError when the target series is empty
        or contains only missing values.
        """
        df_with_nan = dummy_dataset.copy()
        df_with_nan["class"] = None

        with pytest.raises(ValueError, match="The target series is empty"):
            _define_features_target(df_with_nan, split_train_config)

    def test_define_features_target_success_without_splitconfig(self, monkeypatch, dummy_dataset):
        """
        Test that _define_features_target successfully defines features and target when no split_config is provided.
        """

        def mock_init(self, *args, **kwargs):
            object.__setattr__(self, "target_column", "class")
            object.__setattr__(self, "col_group", "field_ID")

        monkeypatch.setattr(
            SplitTrainingConfig,
            "__init__",
            mock_init,
        )

        X, y = _define_features_target(dummy_dataset)

        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        assert list(X.columns) == ["feature_1", "feature_2"] and y.name == "class"
        assert len(y) == len(dummy_dataset) and len(y) == len(dummy_dataset)
        pd.testing.assert_frame_equal(X, dummy_dataset[["feature_1", "feature_2"]])
        pd.testing.assert_series_equal(y, dummy_dataset["class"])
        assert X.index.equals(y.index)

    def test_define_features_target_success(self, dummy_dataset, split_train_config):
        """
        Test that _define_features_target successfully defines features and target when a valid split_config is
        provided.
        """
        X, y = _define_features_target(dummy_dataset, split_train_config)

        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        assert list(X.columns) == ["feature_1", "feature_2"] and y.name == "class"
        assert len(X) == len(dummy_dataset) and len(y) == len(dummy_dataset)
        pd.testing.assert_frame_equal(X, dummy_dataset[["feature_1", "feature_2"]])
        pd.testing.assert_series_equal(y, dummy_dataset["class"])
        assert X.index.equals(y.index)


class TestSplitByGroup:
    @pytest.mark.parametrize(
        "input, bad_value",
        [("X", pd.Series([1, 2])), ("y", 123), ("group_labels", "wrong_type"), ("split_config", DataPathConfig())],
    )
    def test_invalid_input_raises_typeerror(self, input, bad_value, dummy_dataset, split_train_config):
        """
        Test that the _split_by_group function raises a TypeError when any of the input objects
        (X, y, group_labels) have an invalid type.
        """
        kwargs = {
            "X": dummy_dataset[["feature_1", "feature_2"]],
            "y": dummy_dataset["class"],
            "group_labels": dummy_dataset["field_ID"],
            "split_config": split_train_config,
        }
        kwargs[input] = bad_value

        with pytest.raises(TypeError):
            _split_by_group(**kwargs)

    @pytest.mark.parametrize(
        "input, bad_value",
        [
            ("X", pd.DataFrame({"feature_1": [], "feature_2": []})),
            ("y", pd.Series(data=[])),
            ("group_labels", pd.Series(data=[])),
        ],
    )
    def test_empty_input_object_raises_valueerror(self, input, bad_value, dummy_dataset, split_train_config):
        """
        Test that the _split_by_group function raises a ValueError when any of the input objects
        (X, y, group_labels) are empty.
        """
        kwargs = {
            "X": dummy_dataset[["feature_1", "feature_2"]],
            "y": dummy_dataset["class"],
            "group_labels": dummy_dataset["field_ID"],
            "split_config": split_train_config,
        }
        kwargs[input] = bad_value

        with pytest.raises(ValueError, match="is empty"):
            _split_by_group(**kwargs)

    @pytest.mark.parametrize("bad_value", [0.0, 1.0, 1.1, -0.1])
    def test_train_test_fraction_raises_valueerror(self, dummy_dataset, bad_value, split_train_config):
        """
        Test that the _split_by_group function raises a ValueError when the train/test split fraction
        is not between 0.0 and 1.0.
        """
        X = dummy_dataset[["feature_1", "feature_2"]]
        y = dummy_dataset["class"]
        group_labels = dummy_dataset["field_ID"]
        split_train_config_err = replace(split_train_config, train_split=bad_value)

        with pytest.raises(ValueError, match="The fraction of train/test split"):
            _split_by_group(X, y, group_labels, split_train_config_err)

    def test_split_determinism_and_seed_control(self, dummy_dataset, split_train_config):
        """
        Test that the split is deterministic and controlled by the random seed.
        """
        X = dummy_dataset[["feature_1", "feature_2"]]
        y = dummy_dataset["class"]
        group_labels = dummy_dataset["field_ID"]

        train_idx_1, test_idx_1 = _split_by_group(X, y, group_labels, split_train_config)
        train_idx_2, test_idx_2 = _split_by_group(X, y, group_labels, split_train_config)

        assert list(train_idx_1) == list(train_idx_2) and list(test_idx_1) == list(test_idx_2)

        config_2 = replace(split_train_config, random_seed=0)
        train_idx_2, test_idx_2 = _split_by_group(X, y, group_labels, config_2)

        assert set(train_idx_1) != set(train_idx_2) and set(test_idx_1) != set(test_idx_2)

    def test_success_split_by_group_without_splitconfig(self, monkeypatch, dummy_dataset):
        """
        Test that the _split_by_group function correctly splits the dataset into training and testing
        sets based on the provided SplitTrainingConfig, even when the split_config is not provided.
        """

        def mock_init(self, *args, **kwargs):
            object.__setattr__(self, "train_split", 0.6)
            object.__setattr__(self, "random_seed", 42)

        monkeypatch.setattr(
            SplitTrainingConfig,
            "__init__",
            mock_init,
        )

        X = dummy_dataset[["feature_1", "feature_2"]]
        y = dummy_dataset["class"]
        group_labels = dummy_dataset["field_ID"]

        train_idx, test_idx = _split_by_group(X, y, group_labels)

        assert len(train_idx) > len(test_idx)
        assert len(train_idx) + len(test_idx) == len(dummy_dataset)
        assert len(X.iloc[train_idx]) == len(y.iloc[train_idx]) and len(X.iloc[test_idx]) == len(y.iloc[test_idx])
        assert set(group_labels.iloc[train_idx]).isdisjoint(set(group_labels.iloc[test_idx]))
        assert set(train_idx).isdisjoint(set(test_idx))

    def test_success_split_by_group(self, dummy_dataset, split_train_config):
        """
        Test that the _split_by_group function correctly splits the dataset into training and testing
        sets based on the provided SplitTrainingConfig.
        """
        X = dummy_dataset[["feature_1", "feature_2"]]
        y = dummy_dataset["class"]
        group_labels = dummy_dataset["field_ID"]
        split_config = split_train_config

        train_idx, test_idx = _split_by_group(X, y, group_labels, split_config)

        assert len(train_idx) > len(test_idx)
        assert len(train_idx) + len(test_idx) == len(dummy_dataset)
        assert len(X.iloc[train_idx]) == len(y.iloc[train_idx]) and len(X.iloc[test_idx]) == len(y.iloc[test_idx])
        assert set(group_labels.iloc[train_idx]).isdisjoint(set(group_labels.iloc[test_idx]))
        assert set(train_idx).isdisjoint(set(test_idx))


class TestLoadAndSplitData:
    @pytest.mark.parametrize("input, bad_value", [("data_path", 123), ("split_config", "invalid")])
    def test_invalid_input_type_raises_typeerror(self, input, bad_value, split_train_config):
        """
        Test that load_and_split_data raises a TypeError when provided with invalid input types.
        """
        kwargs = {
            "data_path": DataPathConfig(),
            "split_config": split_train_config,
        }

        kwargs[input] = bad_value

        with pytest.raises(TypeError):
            load_and_split_data(**kwargs)

    @patch("src.data.data_loader._load_dataset")
    def test_load_dataset_failure_raises_runtimeerror(self, mock_load_dataset, split_train_config):
        """ """
        mock_load_dataset.side_effect = FileNotFoundError("File not found on disk")

        with pytest.raises(RuntimeError, match="Dataset loading failed: File not found on disk"):
            load_and_split_data(DataPathConfig(), split_train_config)

    @patch("src.data.data_loader._load_dataset")
    def test_missing_target_column_raises_valueerror(self, mock_load_dataset, dummy_dataset, split_train_config):
        """
        This test checks that if the target column specified in the SplitTrainingConfig does not exist in the
        loaded dataset, the load_and_split_data function should raise a ValueError with the appropriate message.
        """
        mock_load_dataset.return_value = dummy_dataset.drop(columns=["class"])

        with pytest.raises(ValueError, match="The target column does not exist in the loaded dataframe"):
            load_and_split_data(DataPathConfig(), split_train_config)

    @patch("src.data.data_loader._load_dataset")
    def test_missing_group_column_raises_valueerror(self, mock_load_dataset, dummy_dataset, split_train_config):
        """
        This test checks that if the group column specified in the SplitTrainingConfig does not exist in the
        loaded dataset, the load_and_split_data function should raise a ValueError with the appropriate message.
        """
        mock_load_dataset.return_value = dummy_dataset.drop(columns=["field_ID"])

        with pytest.raises(ValueError, match="The group column does not exist in the loaded dataframe"):
            load_and_split_data(DataPathConfig(), split_train_config)

    @patch("src.data.data_loader._define_features_target")
    @patch("src.data.data_loader._load_dataset")
    def test_define_features_target_failure_raises_runtimeerror(
        self, mock_load_dataset, mock_define_features_target, split_train_config, dummy_dataset
    ):
        """
        This test checks that if the _define_features_target function raises an exception, the load_and_split_data
        function should catch it and raise a RuntimeError with the appropriate message.
        """
        mock_load_dataset.return_value = dummy_dataset
        mock_define_features_target.side_effect = ValueError(
            "The splitting of the feature dataset and the target series did not take place."
        )

        with pytest.raises(
            RuntimeError,
            match="Feature/target definition failed: The splitting of the feature dataset and the "
            "target series did not take place",
        ):
            load_and_split_data(DataPathConfig(), split_train_config)

    @patch("src.data.data_loader._define_features_target")
    @patch("src.data.data_loader._load_dataset")
    def test_mismatched_features_target_length_raises_valueerror(
        self, mock_load_dataset, mock_define_features_target, split_train_config, dummy_dataset
    ):
        """
            This test checks that if the lengths of the features dataset and target series do not match, the
        load_and_split_data function should raise a ValueError with the appropriate message.
        It mocks the dataset loading and feature/target definition to simulate a scenario where the lengths are mismatched.
        """
        mock_load_dataset.return_value = dummy_dataset
        mock_define_features_target.return_value = (
            dummy_dataset[["feature_1", "feature_2"]].drop(index=0),
            dummy_dataset["class"],
        )

        with pytest.raises(ValueError, match="The length of features dataset and target series do not match"):
            load_and_split_data(DataPathConfig(), split_train_config)

    @patch("src.data.data_loader._define_features_target")
    @patch("src.data.data_loader._load_dataset")
    def test_insufficient_unique_groups_raises_valueerror(
        self, mock_load_dataset, mock_define_features_target, split_train_config
    ):
        """
        This test checks that if the dataset has insufficient unique groups for splitting, the load_and_split_data
        function should raise a ValueError with the appropriate message. It mocks the dataset loading and
        feature/target definition to simulate a scenario where all samples belong to the same group.
        """
        internal_dummy_dataset = pd.DataFrame({"feature_1": [1.1, 1.2, 1.3], "class": [0, 1, 2], "field_ID": [0, 0, 0]})
        mock_load_dataset.return_value = internal_dummy_dataset
        mock_define_features_target.return_value = internal_dummy_dataset["feature_1"], internal_dummy_dataset["class"]

        with pytest.raises(ValueError, match="Not enough unique groups to perform a split"):
            load_and_split_data(DataPathConfig(), split_train_config)

    @patch("src.data.data_loader._split_by_group")
    @patch("src.data.data_loader._define_features_target")
    @patch("src.data.data_loader._load_dataset")
    def test_split_by_group_failure_raises_runtimeerror(
        self, mock_load_dataset, mock_define_features_target, mock_split_by_group, split_train_config, dummy_dataset
    ):
        """
        This test checks that if the _split_by_group function raises an exception, the load_and_split_data function
        should catch it and raise a RuntimeError with the appropriate message.
        """
        mock_load_dataset.return_value = dummy_dataset
        mock_define_features_target.return_value = dummy_dataset[["feature_1", "feature_2"]], dummy_dataset["class"]
        mock_split_by_group.side_effect = ValueError("It was not possible to split the dataset")

        with pytest.raises(RuntimeError, match="Dataset split failed: It was not possible to split the dataset"):
            load_and_split_data(DataPathConfig(), split_train_config)

    def test_success_load_and_split_data_without_configs(self, tmp_path, monkeypatch, dummy_dataset):
        """
        Test the successful loading and splitting of data using a dummy dataset without providing explicit
        configurations.
        """
        processed_dataset_path = tmp_path / "processed_data.csv"
        dummy_dataset.to_csv(processed_dataset_path, index=False)

        def mock_split_init(self, *args, **kwargs):
            object.__setattr__(self, "col_group", "field_ID")
            object.__setattr__(self, "train_split", 0.6)
            object.__setattr__(self, "target_column", "class")
            object.__setattr__(self, "random_seed", 42)

        monkeypatch.setattr(
            SplitTrainingConfig,
            "__init__",
            mock_split_init,
        )

        def mock_data_path_init(self, *args, **kwargs):
            object.__setattr__(self, "processed_data", processed_dataset_path)

        monkeypatch.setattr(
            DataPathConfig,
            "__init__",
            mock_data_path_init,
        )

        X_train, X_test, y_train, y_test, train_group_label, test_group_label = load_and_split_data()

        assert set(train_group_label).isdisjoint(set(test_group_label))
        assert len(X_train) == len(y_train) == len(train_group_label)
        assert len(X_test) == len(y_test) == len(test_group_label)
        assert not X_train.empty and not X_test.empty
        assert len(X_train) + len(X_test) == len(dummy_dataset)
        assert "class" not in X_train.columns
        assert "field_ID" not in X_train.columns
        assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)

    def test_success_load_and_split_data(self, tmp_path, split_train_config, dummy_dataset):
        """
        Test the successful loading and splitting of data using a dummy dataset.
        This test creates a temporary CSV file with the dummy dataset, loads it using the load_and_split_data function,
        and verifies that the resulting training and testing datasets are correctly split and contain the expected
        columns and indices.
        """
        processed_dataset_path = tmp_path / "processed_data.csv"
        dummy_dataset.to_csv(processed_dataset_path, index=False)

        data_path = DataPathConfig(processed_data=processed_dataset_path)

        X_train, X_test, y_train, y_test, train_group_label, test_group_label = load_and_split_data(
            data_path, split_train_config
        )

        assert set(train_group_label).isdisjoint(set(test_group_label))
        assert len(X_train) == len(y_train) == len(train_group_label)
        assert len(X_test) == len(y_test) == len(test_group_label)
        assert not X_train.empty and not X_test.empty
        assert len(X_train) + len(X_test) == len(dummy_dataset)
        assert "class" not in X_train.columns
        assert "field_ID" not in X_train.columns
        assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)
