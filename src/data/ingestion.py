import logging

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from configs import KaggleConfig, load_kaggle_config

# Import local modules
from configs.paths import PROJECT_ROOT, DataPathConfig
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


class KaggleDownloader:
    def __init__(self, kaggle_config: KaggleConfig, datapath_config: DataPathConfig):
        self.kaggle_config = kaggle_config
        self.datapath_config = datapath_config

        validate_type(
            kaggle_config=(kaggle_config, KaggleConfig),
            datapath_config=(datapath_config, DataPathConfig),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _download_with_retry(self) -> pd.DataFrame:
        return kagglehub.dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=self.kaggle_config.dataset_path,
            path=self.kaggle_config.file_name,
        )

    def download(self) -> pd.DataFrame:
        try:
            df = self._download_with_retry()
            validate_type(dataset=(df, pd.DataFrame))
            self._validation_dataset(df)
            logger.info("Dataset successfully downloaded from Kaggle!")
            return df
        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            raise

    def _validation_dataset(self, df: pd.DataFrame) -> None:
        if sorted(df.columns) != sorted(self.kaggle_config.dataset_columns):
            raise ValueError(
                "The columns in the downloaded dataset do not match those expected (in configs/params.yaml)."
            )

        if len(df) != self.kaggle_config.dataset_length:
            raise ValueError("The length of the dataset is not as expected!")

    def save(self, df: pd.DataFrame) -> None:
        validate_type(df=(df, pd.DataFrame))

        self.datapath_config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.datapath_config.raw_data_path, index=False)

        try:
            display_path = self.datapath_config.raw_data_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = self.datapath_config.raw_data_path

        logger.info(f"Dataset saved successfully to: {display_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    k_config = load_kaggle_config()
    d_config = DataPathConfig()

    downloader = KaggleDownloader(kaggle_config=k_config, datapath_config=d_config)
    df = downloader.download()
    downloader.save(df)
