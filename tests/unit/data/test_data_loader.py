from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from configs.paths import DataPathConfig
from configs.schemas import SplitTrainingConfig
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
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
            "class": [0, 1, 2],
            "field_ID": [152, 153, 153],
        }
    )


class TestLoadAndSplitDataLoading:
    def test_loads_dataset_from_csv(self, tmp_path):
        """
        This test ensures that _load_dataset correctly loads a DataFrame from a CSV file.
        """
        csv_path = tmp_path / "processed_data.csv"
        expected_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected_df.to_csv(csv_path, index=False)

        data_path = MagicMock(spec=DataPathConfig)
        data_path.processed_data_path = csv_path

        result = _load_dataset(data_path)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_invalid_data_path_raises_typeerror(self):
        """
        This test ensures that _load_dataset raises a TypeError when the data path is of an invalid type.
        """
        with pytest.raises(TypeError):
            _load_dataset("unknown_type")

    def test_missing_file_raises_error(self, tmp_path):
        """
        This test ensures that _load_dataset raises an error when the specified CSV file does not exist.
        """

        data_path = MagicMock(spec=DataPathConfig)
        data_path.processed_data_path = tmp_path / "unknown_file.csv"

        with pytest.raises(Exception):
            _load_dataset(data_path)


class TestDefineFeaturesTarget:
    def test_invalid_dataset_raises_typeerror(self):
        """
        This test ensures that _define_features_target raises a TypeError
        when the dataset is of an invalid type.
        """
        df_err = pd.Series([1, 2, 3], name="feature_1")
        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(TypeError):
            _define_features_target(df_err, split_config)

    def test_invalid_split_config_raises_typeerror(self):
        """
        This test ensures that _define_features_target raises a TypeError
        when the split configuration is of an invalid type.
        """
        df_err = pd.DataFrame({"a": [1, 2, 3]})
        split_config = "unknown_type"

        with pytest.raises(TypeError):
            _define_features_target(df_err, split_config)

    def test_separates_features_and_target(self):
        """
        This test ensures that _define_features_target correctly separates
        features and target based on the provided SplitTrainingConfig.
        """
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3, 4],
                "feature_2": [5, 6, 7, 8],
                "class": [0, 0, 1, 0],
                "field_ID": [1, 2, 2, 3],
            }
        )

        X, y = _define_features_target(df, split_config)

        X_expected = pd.DataFrame(
            {
                "feature_1": [1, 2, 3, 4],
                "feature_2": [5, 6, 7, 8],
            }
        )

        y_expected = pd.Series([0, 0, 1, 0], name="class")

        pd.testing.assert_frame_equal(X, X_expected)
        pd.testing.assert_series_equal(y, y_expected)

    def test_missing_target_column_raises_error(self):
        """
        This test ensures that _define_features_target raises a KeyError when the target column is missing.
        """
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        df = pd.DataFrame({"feature_1": [1, 2, 3, 4], "feature_2": [5, 6, 7, 8], "field_ID": [1, 2, 2, 3]})

        with pytest.raises(KeyError, match="class"):
            _define_features_target(df, split_config)

    def test_missing_group_column_raises_error(self):
        """
        This test ensures that _define_features_target raises a KeyError when the group column is missing.
        """
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        df = pd.DataFrame({"feature_1": [1, 2, 3, 4], "feature_2": [5, 6, 7, 8], "class": [1, 2, 2, 3]})

        with pytest.raises(KeyError, match="field_ID"):
            _define_features_target(df, split_config)

    def test_no_remaining_features_raises_valueerror(self):
        """
        This test ensures that _define_features_target raises a ValueError when there are no remaining features.
        """
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        df = pd.DataFrame({"class": [0, 0, 1, 0], "field_ID": [1, 2, 2, 3]})

        with pytest.raises(ValueError, match="The features dataset is empty!"):
            _define_features_target(df, split_config)


class TestSplitByGroup:
    def test_invalid_features_dataset_raises_typeerror(self):
        """
        This test ensures that _split_by_group raises a TypeError when the features dataset is of an invalid type.
        """
        X = "unknow_type"
        y = pd.Series([0, 0, 1], name="class")
        group_labels = pd.Series([152, 153, 153], name="class")

        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(TypeError):
            _split_by_group(X, y, group_labels, split_config)

    def test_invalid_target_raises_typeerror(self):
        """
        This test ensures that _split_by_group raises a TypeError when the target is of an invalid type.
        """
        X = pd.DataFrame(
            {
                "features_1": [1, 2, 3],
                "features_2": [4, 5, 6],
            }
        )
        y = "unknow_type"
        group_labels = pd.Series([152, 153, 153], name="class")

        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(TypeError):
            _split_by_group(X, y, group_labels, split_config)

    def test_invalid_group_labels_raises_typeerror(self):
        """
        This test ensures that _split_by_group raises a TypeError when the group labels are of an invalid type.
        """
        X = pd.DataFrame(
            {
                "features_1": [1, 2, 3],
                "features_2": [4, 5, 6],
            }
        )
        y = pd.Series([0, 0, 1], name="class")
        group_labels = "unknow_type"

        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(TypeError):
            _split_by_group(X, y, group_labels, split_config)

    def test_split_produces_valid_indices(self):
        """
        This test ensures that _split_by_group produces valid train and test indices.
        """
        X = pd.DataFrame({"feature_1": range(10)})
        y = pd.Series(["A"] * 5 + ["B"] * 5)
        group_labels = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], name="field_ID")

        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"
        split_config.train_split = 0.6
        split_config.random_seed = 42

        train_idx, test_idx = _split_by_group(X, y, group_labels, split_config)

        assert len(X.iloc[train_idx]) == 6 and len(X.iloc[test_idx]) == 4
        assert len(X.iloc[train_idx]) == len(y.iloc[train_idx]) and len(X.iloc[test_idx]) == len(y.iloc[test_idx])

    def test_groups_do_not_overlap_between_train_and_test(self):
        """
        This test ensures that groups do not overlap between the training and test sets.
        """
        X = pd.DataFrame({"feature_1": range(10)})
        y = pd.Series(["A"] * 5 + ["B"] * 5)
        group_labels = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], name="field_ID")

        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"
        split_config.train_split = 0.6
        split_config.random_seed = 42

        train_idx, test_idx = _split_by_group(X, y, group_labels, split_config)

        train_group = set(group_labels.iloc[train_idx])
        test_group = set(group_labels.iloc[test_idx])

        assert train_group.isdisjoint(test_group)


class TestLoadAndSplitDataIntegration:
    def test_invalid_data_path_raises_typeerror(self):
        """
        This test ensures that load_and_split_data raises a TypeError when the data path is of an invalid type.
        """
        data_path = "unknown_type"
        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(TypeError):
            load_and_split_data(data_path, split_config)

    def test_invalid_split_config_raises_typeerror(self):
        """
        This test ensures that load_and_split_data raises a TypeError when the split config is of an invalid type.
        """
        data_path = MagicMock(spec=DataPathConfig)
        split_config = "unknown_type"

        with pytest.raises(TypeError):
            load_and_split_data(data_path, split_config)

    def test_missing_file_raises_error(self, tmp_path):
        """
        This test ensures that load_and_split_data raises an error when the specified CSV file does not exist.
        """
        data_path = MagicMock(spec=DataPathConfig)
        data_path.processed_data_path = tmp_path / "data" / "processed" / "processed_data.csv"

        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(RuntimeError, match="Dataset loading failed"):
            load_and_split_data(data_path, split_config)

    def test_empty_dataset_raises_valueerror(self, tmp_path):
        """
        This test ensures that load_and_split_data raises a ValueError when the loaded dataset is empty.
        """
        processed_data_path = tmp_path / "processed_data.csv"

        df = pd.DataFrame(columns=["target", "field_ID", "f1"])
        df.to_csv(processed_data_path, index=False)

        data_path = MagicMock(spec=DataPathConfig)
        data_path.processed_data_path = processed_data_path

        split_config = MagicMock(spec=SplitTrainingConfig)

        with pytest.raises(ValueError, match="The dataset loaded is empty!"):
            load_and_split_data(data_path, split_config)

    @patch("src.data.data_loader._load_dataset")
    def test_missing_target_column_raises_valueerror(self, mock_load_dataset):
        """
        This test ensures that load_and_split_data raises a ValueError
        when the target column does not exist in the loaded dataframe.
        """
        df_err = pd.DataFrame({"feature_1": [1, 2, 3], "ERROR": [0, 0, 1], "field_ID": [153, 152, 153]})

        mock_load_dataset.return_value = df_err

        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        with pytest.raises(ValueError, match="The target column does not exist in the loaded dataframe."):
            load_and_split_data(data_path, split_config)

    @patch("src.data.data_loader._load_dataset")
    def test_missing_group_column_raises_valueerror(self, mock_load_dataset):
        """
        This test ensures that load_and_split_data raises a ValueError
        when the group column does not exist in the loaded dataframe.
        """
        df_err = pd.DataFrame({"feature_1": [1, 2, 3], "class": [0, 0, 1], "ERROR": [153, 152, 153]})

        mock_load_dataset.return_value = df_err

        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        with pytest.raises(ValueError, match="The group column does not exist in the loaded dataframe."):
            load_and_split_data(data_path, split_config)

    @patch("src.data.data_loader._load_dataset")
    @patch("src.data.data_loader._define_features_target")
    def test_define_features_target_failure_raises_runtimeerror(
        self, mock_define_features_target, mock_load_dataset, dummy_dataset
    ):
        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        mock_load_dataset.return_value = dummy_dataset
        mock_define_features_target.side_effect = RuntimeError("Mock failure")

        with pytest.raises(RuntimeError, match="Feature/target definition failed"):
            load_and_split_data(data_path, split_config)

        mock_load_dataset.assert_called_once_with(data_path)
        mock_define_features_target.assert_called_once()

    @patch("src.data.data_loader._load_dataset")
    @patch("src.data.data_loader._define_features_target")
    def test_mismatched_features_target_length_raises_valueerror(
        self, mock_define_features_target, mock_load_dataset, dummy_dataset
    ):
        df_fake = dummy_dataset

        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        mock_load_dataset.return_value = df_fake

        X = df_fake.drop(columns=[split_config.target_column, split_config.col_group])
        y_err = pd.Series([0, 1])

        mock_define_features_target.return_value = (X, y_err)

        with pytest.raises(ValueError, match="The length of features dataset and target series do not match!"):
            load_and_split_data(data_path, split_config)

    @patch("src.data.data_loader._load_dataset")
    @patch("src.data.data_loader._define_features_target")
    def test_insufficient_unique_groups_raises_valueerror(self, mock_define_features_target, mock_load_dataset):
        df_err = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
                "class": [0, 1, 1],
                "field_ID": [153, 153, 153],  # <-- only one group
            }
        )

        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        X = df_err.drop(columns=[split_config.target_column, split_config.col_group])
        y = df_err[split_config.col_group]

        mock_load_dataset.return_value = df_err
        mock_define_features_target.return_value = (X, y)

        with pytest.raises(ValueError, match="Not enough unique groups to perform a split"):
            load_and_split_data(data_path, split_config)

    @patch("src.data.data_loader._load_dataset")
    @patch("src.data.data_loader._define_features_target")
    @patch("src.data.data_loader._split_by_group")
    def test_split_by_group_failure_raises_runtimeerror(
        self, mock_split_by_group, mock_define_features_target, mock_load_dataset, dummy_dataset
    ):
        df_fake = dummy_dataset

        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        X = df_fake.drop(columns=[split_config.target_column, split_config.col_group])
        y = df_fake[split_config.col_group]

        mock_load_dataset.return_value = df_fake
        mock_define_features_target.return_value = (X, y)
        mock_split_by_group.side_effect = RuntimeError("Mock failure")

        with pytest.raises(RuntimeError, match="Mock failure"):
            load_and_split_data(data_path, split_config)

    @patch("src.data.data_loader._load_dataset")
    @patch("src.data.data_loader._define_features_target")
    @patch("src.data.data_loader._split_by_group")
    def test_empty_train_or_test_split_raises_valueerror(
        self, mock_split_by_group, mock_define_features_target, mock_load_dataset, dummy_dataset
    ):
        df_fake = dummy_dataset

        data_path = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"

        X = df_fake.drop(columns=[split_config.target_column, split_config.col_group])
        y = df_fake[split_config.col_group]

        mock_load_dataset.return_value = df_fake
        mock_define_features_target.return_value = (X, y)
        train_idx, test_idx = [0, 1], []
        mock_split_by_group.return_value = train_idx, test_idx

        with pytest.raises(ValueError, match="The split produced an empty train or test set!"):
            load_and_split_data(data_path, split_config)

    def test_valid_load_and_split_data(self, tmp_path, dummy_dataset):
        processed_data_path = tmp_path / "processed_dataset.csv"

        df = dummy_dataset
        df.to_csv(processed_data_path, index=False)

        data_path = MagicMock(spec=DataPathConfig)
        data_path.processed_data_path = processed_data_path

        split_config = MagicMock(spec=SplitTrainingConfig)
        split_config.target_column = "class"
        split_config.col_group = "field_ID"
        split_config.train_split = 0.6
        split_config.random_seed = 42

        X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split_data(data_path, split_config)

        assert not X_train.empty
        assert not X_test.empty
        assert not y_train.empty
        assert not y_test.empty

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        assert len(X_train) == len(groups_train)
        assert len(X_test) == len(groups_test)

        assert len(X_train) + len(X_test) == len(dummy_dataset)

        assert "class" not in X_train.columns
        assert "field_ID" not in X_train.columns

        assert set(groups_train).isdisjoint(set(groups_test))
