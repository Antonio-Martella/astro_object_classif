from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kagglehub import KaggleDatasetAdapter

from configs.paths import DataPathConfig
from configs.schemas import KaggleConfig
from src.data.ingestion import KaggleDownloader


@pytest.fixture
def dummy_dataset():
    """
    Dummy dataset fixture for testing.
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    dataset_columns = list(df.keys())
    dataset_length = len(df)
    dataset_path = "some/dataset"
    file_name = "data.csv"

    return df, dataset_columns, dataset_length, dataset_path, file_name


class TestKaggleDownloaderInit:
    def test_valid_configs_creates_instance(self):
        """
        This test ensures that the KaggleDownloader can be instantiated with valid configurations.
        """
        kaggle_config = MagicMock(spec=KaggleConfig)
        datapath_config = MagicMock(spec=DataPathConfig)

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        assert downloader.kaggle_config is kaggle_config
        assert downloader.datapath_config is datapath_config

    def test_invalid_kaggle_config_raises_typeerror(self):
        """
        This test ensures that an invalid KaggleConfig raises a TypeError.
        """
        invalid_kaggle_config = "not_a_kaggle_config"
        datapath_config = MagicMock(spec=DataPathConfig)

        with pytest.raises(TypeError):
            KaggleDownloader(invalid_kaggle_config, datapath_config)

    def test_invalid_datapath_config_raises_typeerror(self):
        """
        This test ensures that an invalid DataPathConfig raises a TypeError.
        """
        kaggle_config = MagicMock(spec=KaggleConfig)
        invalid_datapath_config = "not_a_datapath_config"

        with pytest.raises(TypeError):
            KaggleDownloader(kaggle_config, invalid_datapath_config)


class TestKaggleDownloaderDownload:
    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_calls_dataset_load_with_correct_params(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that the download method calls the dataset load with the correct parameters.
        """
        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        (
            fake_df,
            kaggle_config.dataset_columns,
            kaggle_config.dataset_length,
            kaggle_config.dataset_path,
            kaggle_config.file_name,
        ) = dummy_dataset

        mock_dataset_load.return_value = fake_df

        downloader = KaggleDownloader(kaggle_config, datapath_config)
        downloader.download()

        mock_dataset_load.assert_called_once_with(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle="some/dataset",
            path="data.csv",
        )

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_returns_dataframe_from_dataset_load(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that the download method returns the DataFrame returned by the dataset load.
        """
        fake_df, columns, length, dataset_path, file_name = dummy_dataset

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_columns = columns
        kaggle_config.dataset_length = length
        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.return_value = fake_df

        downloader = KaggleDownloader(kaggle_config, datapath_config)
        result = downloader.download()

        assert result is fake_df

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_propagates_exception_from_dataset_load(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that the download method propagates exceptions from the dataset load.
        """
        fake_df, columns, length, dataset_path, file_name = dummy_dataset

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_columns = columns
        kaggle_config.dataset_length = length
        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.side_effect = ValueError("Kaggle dataset not found")

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(ValueError, match="Kaggle dataset not found"):
            downloader.download()

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_download_raises_valueerror_on_wrong_columns(self, mock_dataset_load):
        """
        This test ensures that the download method raises a ValueError when the dataset columns are incorrect.
        """
        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        kaggle_config.dataset_path = "some/dataset"
        kaggle_config.file_name = "data.csv"
        kaggle_config.dataset_columns = ["a", "b"]
        kaggle_config.dataset_length = 3

        mock_dataset_load.return_value = df

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(ValueError, match="columns"):
            downloader.download()

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_download_raises_valueerror_on_wrong_length(self, mock_dataset_load):
        """
        This test ensures that the download method raises a ValueError when the dataset length is incorrect.
        """
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        columns = ["a", "b", "c"]
        length = len(df) - 1
        dataset_path = "some/dataset"
        file_name = "data.csv"

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_columns = columns
        kaggle_config.dataset_length = length
        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.return_value = df

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(ValueError, match="length"):
            downloader.download()


class TestKaggleDownloaderSave:
    def test_creates_missing_directory(self, tmp_path):
        """
        This test ensures that the save method creates the missing directory if it does not exist.
        """
        raw_data_path = tmp_path / "data" / "raw" / "Star_Classification.csv"

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path

        kaggle_config = MagicMock(spec=KaggleConfig)

        df = pd.DataFrame({"a": [1, 2, 3]})

        assert not raw_data_path.parent.exists()

        downloader = KaggleDownloader(kaggle_config, datapath_config)
        downloader.save(df)

        assert raw_data_path.parent.exists()

    def test_does_not_fail_if_directory_exists(self, tmp_path):
        """
        This test ensures that the save method does not fail if the directory already exists.
        """
        raw_data_path = tmp_path / "data" / "raw" / "Star_Classification.csv"

        raw_data_path.parent.mkdir(parents=True)

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path

        kaggle_config = MagicMock(spec=KaggleConfig)

        df = pd.DataFrame({"a": [1, 2, 3]})

        downloader = KaggleDownloader(kaggle_config, datapath_config)
        downloader.save(df)

        assert raw_data_path.exists()

    def test_writes_correct_csv_content(self, tmp_path):
        """
        This test ensures that the correct CSV content is written to the file.
        """
        raw_data_path = tmp_path / "data" / "raw" / "Star_Classification.csv"

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path

        kaggle_config = MagicMock(spec=KaggleConfig)

        df_test = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        downloader = KaggleDownloader(kaggle_config, datapath_config)
        downloader.save(df_test)

        df_copy = pd.read_csv(raw_data_path)

        pd.testing.assert_frame_equal(df_test, df_copy)

    def test_invalid_dataframe_type_raises_typeerror_and_does_not_write_file(self, tmp_path):
        """
        This test ensures that an invalid dataframe type raises a TypeError and does not write the file.
        """
        raw_data_path = tmp_path / "data" / "raw" / "Star_Classification.csv"

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path

        kaggle_config = MagicMock(spec=KaggleConfig)

        invalid_series = pd.Series([1, 2, 3])

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(TypeError):
            downloader.save(invalid_series)

        assert not raw_data_path.exists()


class TestKaggleDownloaderRetry:
    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_gives_up_after_max_retries_on_connection_error(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that the downloader gives up after the maximum number of retries on a connection error.
        """

        fake_df, columns, length, dataset_path, file_name = dummy_dataset

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.side_effect = ConnectionError("network down")

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(ConnectionError, match="network down"):
            downloader._download_with_retry()

        assert mock_dataset_load.call_count == 3

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_gives_up_after_max_retries_on_timeout_error(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that the downloader gives up after a timeout error.
        """

        fake_df, columns, length, dataset_path, file_name = dummy_dataset

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.side_effect = TimeoutError()

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(TimeoutError):
            downloader._download_with_retry()

        assert mock_dataset_load.call_count == 3

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_non_retriable_exception_fails_immediately_without_retry(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that if a non-retriable exception is raised,
        the retry mechanism does not attempt to retry and the exception is raised immediately.
        """
        fake_df, columns, length, dataset_path, file_name = dummy_dataset

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.side_effect = ValueError()

        downloader = KaggleDownloader(kaggle_config, datapath_config)

        with pytest.raises(ValueError):
            downloader._download_with_retry()

        assert mock_dataset_load.call_count == 1

    @patch("src.data.ingestion.kagglehub.dataset_load")
    def test_retries_on_connection_error_then_succeeds(self, mock_dataset_load, dummy_dataset):
        """
        This test ensures that the downloader retries on connection errors and eventually succeeds.
        """
        fake_df, columns, length, dataset_path, file_name = dummy_dataset

        datapath_config = MagicMock(spec=DataPathConfig)
        kaggle_config = MagicMock(spec=KaggleConfig)

        kaggle_config.dataset_columns = columns
        kaggle_config.dataset_length = length
        kaggle_config.dataset_path = dataset_path
        kaggle_config.file_name = file_name

        mock_dataset_load.side_effect = [ConnectionError(), ConnectionError(), fake_df]

        downloader = KaggleDownloader(kaggle_config, datapath_config)
        result = downloader._download_with_retry()

        assert mock_dataset_load.call_count == 3
        assert result is fake_df
