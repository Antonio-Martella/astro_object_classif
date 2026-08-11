from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from configs.paths import DataPathConfig
from configs.schemas import SplitTrainingConfig
from src.training.data import load_split_and_encode_dataset


class TestLoadSplitAndEncodeDataset:
    def test_load_split_and_encode_typeerror(self):
        datapath_config = MagicMock(spec=DataPathConfig)
        split_config = "unknown_type"

        with pytest.raises(TypeError, match="Parameter 'split_config'"):
            load_split_and_encode_dataset(datapath_config, split_config)

    @patch("src.training.data.encode_targets_and_save")
    @patch("src.training.data.load_and_split_data")
    def test_uses_default_configs_when_none_provided(self, mock_load_and_split_data, mock_encode_targets_and_save):
        mock_load_and_split_data.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        mock_encode_targets_and_save.return_value = (MagicMock(), MagicMock(), MagicMock())

        load_split_and_encode_dataset(datapath_config=None, split_config=None)

        assert mock_load_and_split_data.called
        assert mock_encode_targets_and_save.called

        args, _ = mock_load_and_split_data.call_args
        assert isinstance(args[0], DataPathConfig)

    @patch("src.training.data.encode_targets_and_save")
    @patch("src.training.data.load_and_split_data")
    def test_try_except_error(self, mock_load_and_split_data, tmp_path):
        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.target_le = tmp_path / "target_label_encoder.pkl"
        split_config = MagicMock(spec=SplitTrainingConfig)

        y_train_mock = MagicMock(spec=pd.Series)
        y_test_mock = MagicMock(spec=pd.Series)
        le_mock = MagicMock(spec=LabelEncoder)
        le_mock.classes_ = ["GALAXY", "QSO", "STAR"]

        mock_load_and_split_data.return_value = (
            MagicMock(spec=pd.DataFrame),
            MagicMock(spec=pd.DataFrame),
            y_train_mock,
            y_test_mock,
            MagicMock(spec=pd.Series),
            MagicMock(spec=pd.Series),
        )

        mock_load_and_split_data.side_effect = ValueError

        with pytest.raises(RuntimeError, match="Data pipeline orchestration failed"):
            load_split_and_encode_dataset(datapath_config=datapath_config, split_config=split_config, save_encoder=True)

    @patch("src.training.data.encode_targets_and_save")
    @patch("src.training.data.load_and_split_data")
    def test_orchestrates_load_split_and_encode_correctly(
        self, mock_load_and_split_data, mock_encode_targets_and_save, tmp_path
    ):
        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.target_le = tmp_path / "target_label_encoder.pkl"
        split_config = MagicMock(spec=SplitTrainingConfig)

        y_train_mock = MagicMock(spec=pd.Series)
        y_test_mock = MagicMock(spec=pd.Series)
        le_mock = MagicMock(spec=LabelEncoder)
        le_mock.classes_ = ["GALAXY", "QSO", "STAR"]

        mock_load_and_split_data.return_value = (
            MagicMock(spec=pd.DataFrame),
            MagicMock(spec=pd.DataFrame),
            y_train_mock,
            y_test_mock,
            MagicMock(spec=pd.Series),
            MagicMock(spec=pd.Series),
        )

        mock_encode_targets_and_save.return_value = (MagicMock(spec=pd.Series), MagicMock(spec=pd.Series), le_mock)

        X_train, X_test, y_train_encoded, y_test_encoded, groups_train, groups_test, le = load_split_and_encode_dataset(
            datapath_config=datapath_config, split_config=split_config, save_encoder=True
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train_encoded, pd.Series)
        assert isinstance(y_test_encoded, pd.Series)
        assert isinstance(groups_train, pd.Series)
        assert isinstance(groups_test, pd.Series)
        assert isinstance(le, LabelEncoder)

        mock_load_and_split_data.assert_called_once_with(datapath_config, split_config)
        mock_encode_targets_and_save.assert_called_once_with(y_train_mock, y_test_mock, True, datapath_config.target_le)
