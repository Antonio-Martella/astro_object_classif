import json
import logging

import mlflow
import pandas as pd

from configs import SplitHoldoutConfig, load_split_holdout_config

# Import local modules
from configs.paths import PROJECT_ROOT, DataPathConfig
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


class TimeBasedSplitter:
    """
    A class that implements dataset splitting based on a reference time column (default ref_col: "MJD")
    necessary to simulate a production dataset (holdout) and a training dataset.
    The split method returns the two dataframes and a dictionary with the split metadata.
    """

    def __init__(self, splitdata_config: SplitHoldoutConfig):
        validate_type(splitdata_config=(splitdata_config, SplitHoldoutConfig))
        self.splitdata_config = splitdata_config

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        validate_type(df=(df, pd.DataFrame))
        metadata: dict[str, int | float | str] = {}

        if self.splitdata_config.ref_col not in df.columns:
            raise ValueError(f"Column '{self.splitdata_config.ref_col}' not found in the dataset!")

        ref_col = self.splitdata_config.ref_col

        if not (0 < self.splitdata_config.holdout_split < 1):
            raise ValueError("The holdout fraction must be between (0,1)")

        metadata["split_prod_ratio"] = self.splitdata_config.holdout_split

        df_sorted = df.sort_values(by=ref_col, ascending=True)

        n_holdout = int(self.splitdata_config.holdout_split * len(df_sorted))

        if n_holdout < 1:
            raise ValueError("Production dataset length is zero!")

        df_train = df_sorted.iloc[:-n_holdout]
        df_prod = df_sorted.iloc[-n_holdout:]

        metadata["n_train"] = len(df_sorted) - n_holdout
        metadata["n_prod"] = n_holdout

        metadata[f"train_{ref_col.lower()}_min"] = int(df_train[ref_col].min())
        metadata[f"train_{ref_col.lower()}_max"] = int(df_train[ref_col].max())

        metadata[f"prod_{ref_col.lower()}_min"] = int(df_prod[ref_col].min())
        metadata[f"prod_{ref_col.lower()}_max"] = int(df_prod[ref_col].max())

        metadata["split_method"] = f"time_based_{ref_col}"

        logger.info(f"Training dataset shape: {df_train.shape}")
        logger.info(f"Production dataset shape: {df_prod.shape}")

        return df_train, df_prod, metadata


class SplitProductionSimulation:
    """
    A class that manages the entire process of splitting the dataset into training and production,
    logging the split metadata to MLflow and saving the two datasets in CSV format.
    """

    def __init__(self, datapath_config: DataPathConfig, split_config: SplitHoldoutConfig):
        validate_type(
            datapath_config=(datapath_config, DataPathConfig),
            split_config=(split_config, SplitHoldoutConfig),
        )
        self.datapath_config = datapath_config
        self.splitter = TimeBasedSplitter(split_config)

    def execute(self) -> pd.DataFrame:
        df = self._load_dataset()
        df_train, df_prod, metadata = self.splitter.split(df)

        try:
            metadata["raw_path"] = f"/{self.datapath_config.raw_data_path.relative_to(PROJECT_ROOT)}"
            metadata["training_path"] = f"/{self.datapath_config.split_training_path.relative_to(PROJECT_ROOT)}"
            metadata["production_path"] = f"/{self.datapath_config.split_production_path.relative_to(PROJECT_ROOT)}"
        except ValueError:
            metadata["raw_path"] = f"/{self.datapath_config.raw_data_path}"
            metadata["training_path"] = f"/{self.datapath_config.split_training_path}"
            metadata["production_path"] = f"/{self.datapath_config.split_production_path}"

        self._save(df_train, df_prod, metadata)

        return df_train

    def _load_dataset(self) -> pd.DataFrame:
        load_path = self.datapath_config.raw_data_path

        if not load_path.exists():
            raise FileNotFoundError("Raw dataset file not found on disk.")

        if not load_path.is_file():
            try:
                display_path = load_path.relative_to(PROJECT_ROOT)
            except ValueError:
                display_path = load_path

            raise ValueError(f"Expected a file, but received: {display_path}")

        return pd.read_csv(load_path)

    def _save(self, df_train: pd.DataFrame, df_prod: pd.DataFrame, metadata: dict) -> None:
        validate_type(
            df_train=(df_train, pd.DataFrame),
            df_prod=(df_prod, pd.DataFrame),
            metadata=(metadata, dict),
        )

        train_path = self.datapath_config.split_training_path
        prod_path = self.datapath_config.split_production_path
        metadata_path = self.datapath_config.split_metadata_path

        train_path.parent.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(train_path, index=False)
        df_prod.to_csv(prod_path, index=False)

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

        try:
            display_train_path = train_path.relative_to(PROJECT_ROOT)
            display_prod_path = prod_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_train_path = train_path
            display_prod_path = prod_path

        logger.info(f"Training dataset saved in: {display_train_path}")
        logger.info(f"Production dataset saved in: {display_prod_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    d_config = DataPathConfig()
    s_config = load_split_holdout_config()

    db_path = d_config.mlflow_db_path  # Path.cwd() / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Astro_Object_Classification_DataPrep")

    SplitProductionSimulation(d_config, s_config).execute()
