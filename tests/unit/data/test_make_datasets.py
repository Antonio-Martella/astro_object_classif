from unittest.mock import MagicMock, patch

import pytest

from configs.paths import DataPathConfig
from configs.schemas import KaggleConfig
from src.data.make_datasets import run_ingestion


@pytest.fixture
def mock_config_path(tmp_path):
    config = MagicMock(spec=DataPathConfig)

    config.raw_data_path = tmp_path / "data" / "raw" / "Star_Classification.csv"
    config.split_training_path = tmp_path / "data" / "interim" / "training_dataset.csv"
    config.split_production_path = tmp_path / "data" / "interim" / "holdout_dataset.csv"
    config.processed_data_path = tmp_path / "data" / "processed" / "processed_data.csv"

    config.opt_logs_make_dataset = tmp_path / "logs" / "run_clean_dataset_creation.log"

    return config


@patch("src.data.make_datasets.KaggleDownloader")
def test_run_ingestion_calls_download_and_save(mock_downloader_cls, mock_config_path):
    mock_downloader_instance = mock_downloader_cls.return_value
    mock_df = MagicMock()
    mock_downloader_instance.download.return_value = mock_df

    kaggle_config = MagicMock(spec=KaggleConfig)
    run_ingestion(kaggle_config, mock_config_path)

    mock_downloader_cls.assert_called_once_with(kaggle_config, mock_config_path)
    mock_downloader_instance.download.assert_called_once()
    mock_downloader_instance.save.assert_called_once_with(mock_df)


@patch("src.data.make_datasets.KaggleDownloader")
def test_run_ingestion_propagates_download_error(mock_downloader_cls, mock_config_path):
    mock_downloader_cls.return_value.download.side_effect = ConnectionError("network down")

    kaggle_config = MagicMock(spec=KaggleConfig)
    with pytest.raises(ConnectionError):
        run_ingestion(kaggle_config, mock_config_path)
