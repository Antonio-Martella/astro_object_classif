import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from configs.paths import DataPathConfig
from configs.schemas import SplitHoldoutConfig
from src.data.holdout_split_data import SplitProductionSimulation, TimeBasedSplitter


@pytest.fixture
def dummy_dataset():
    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [5, 6, 7, 8],
            "class": [0, 1, 1, 9],
            "MJD": [10.1, 10.3, 10.2, 10.0],
        }
    )


@pytest.fixture
def mock_config():
    config = MagicMock(spec=SplitHoldoutConfig)
    config.ref_col = "MJD"
    config.holdout_split = 0.5
    return config


class TestTimeBasedSplitterInit:
    def test_valid_configs_creates_instance(self):
        splitdata_config = MagicMock(spec=SplitHoldoutConfig)
        splitter = TimeBasedSplitter(splitdata_config)

        assert splitter.splitdata_config is splitdata_config

    def test_invalid_split_config_raises_typeerror(self):
        with pytest.raises(TypeError):
            TimeBasedSplitter("unknow_type")


class TestTimeBasedSplitterSplit:
    def test_missing_ref_col_raises_valueerror(self, dummy_dataset):
        df = dummy_dataset
        splitdata_config = MagicMock(spec=SplitHoldoutConfig)
        splitdata_config.ref_col = "unknow"
        splitter = TimeBasedSplitter(splitdata_config)

        with pytest.raises(ValueError, match=f"Column '{splitdata_config.ref_col}' not found in the dataset!"):
            splitter.split(df)

    def test_holdout_split_zero_or_below_raises_valueerror(self, dummy_dataset):
        df = dummy_dataset
        splitdata_config = MagicMock(spec=SplitHoldoutConfig)
        splitdata_config.ref_col = "MJD"
        splitdata_config.holdout_split = 0.0
        splitter = TimeBasedSplitter(splitdata_config)

        with pytest.raises(ValueError, match="The holdout fraction"):
            splitter.split(df)

    def test_holdout_split_above_one_raises_valueerror(self, dummy_dataset):
        df = dummy_dataset
        splitdata_config = MagicMock(spec=SplitHoldoutConfig)
        splitdata_config.ref_col = "MJD"
        splitdata_config.holdout_split = 1.1
        splitter = TimeBasedSplitter(splitdata_config)

        with pytest.raises(ValueError, match="The holdout fraction"):
            splitter.split(df)

    def test_zero_length_holdout_raises_valueerror(self, dummy_dataset):
        df = dummy_dataset
        splitdata_config = MagicMock(spec=SplitHoldoutConfig)
        splitdata_config.ref_col = "MJD"
        splitdata_config.holdout_split = 0.0001
        splitter = TimeBasedSplitter(splitdata_config)

        with pytest.raises(ValueError, match="Production dataset length is zero!"):
            splitter.split(df)

    def test_split_is_time_ordered(self, dummy_dataset, mock_config):
        df = dummy_dataset
        splitdata_config = mock_config
        splitter = TimeBasedSplitter(splitdata_config)
        df_train, df_holdout, _ = splitter.split(df)

        assert df_train["MJD"].is_monotonic_increasing
        assert df_holdout["MJD"].is_monotonic_increasing
        assert df_train["MJD"].iloc[-1] <= df_holdout["MJD"].iloc[0]

    def test_metadata_contains_expected_keys(self, dummy_dataset, mock_config):
        df = dummy_dataset
        splitdata_config = mock_config
        splitter = TimeBasedSplitter(splitdata_config)
        _, _, metadata = splitter.split(df)

        expected_keys = {
            "split_prod_ratio",
            "n_train",
            "n_prod",
            "train_mjd_min",
            "train_mjd_max",
            "prod_mjd_min",
            "prod_mjd_max",
        }

        assert expected_keys.issubset(metadata.keys())

    def test_metadata_min_max_values_are_correct(self, dummy_dataset, mock_config):
        df = dummy_dataset
        splitdata_config = mock_config
        splitter = TimeBasedSplitter(splitdata_config)
        df_train, df_holdout, metadata = splitter.split(df)

        assert metadata["train_mjd_max"] <= metadata["prod_mjd_min"]
        assert metadata["n_train"] == len(df_train)
        assert metadata["n_prod"] == len(df_holdout)

    def test_train_and_prod_do_not_overlap(self, dummy_dataset, mock_config):
        df = dummy_dataset
        splitdata_config = mock_config
        splitter = TimeBasedSplitter(splitdata_config)
        df_train, df_holdout, _ = splitter.split(df)

        assert set(df_train["MJD"]).isdisjoint(set(df_holdout["MJD"]))


class TestSplitProductionSimulationInit:
    def test_valid_sim_configs_creates_instance(self):
        datapath_config = MagicMock(spec=DataPathConfig)
        splitdata_config = MagicMock(spec=SplitHoldoutConfig)

        split = SplitProductionSimulation(datapath_config, splitdata_config)

        assert split.datapath_config is datapath_config
        assert isinstance(split.splitter, TimeBasedSplitter)
        assert split.splitter.splitdata_config is splitdata_config

    def test_invalid_sim_datapath_config_raises_typeerror(self):
        datapath_config = "unknown_type"
        split_config = MagicMock(spec=SplitHoldoutConfig)

        with pytest.raises(TypeError):
            SplitProductionSimulation(datapath_config, split_config)

    def test_invalid_sim_split_config_raises_typeerror(self):
        datapath_config = MagicMock(spec=DataPathConfig)
        split_config = "unknown_type"

        with pytest.raises(TypeError):
            SplitProductionSimulation(datapath_config, split_config)


class TestSplitProductionSimulationLoadDataset:
    def test_missing_raw_dataset_raises_filenotfounderror(self, tmp_path):
        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = tmp_path / "raw_dataset.csv"
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(FileNotFoundError, match="Raw dataset file not found on disk."):
            sim._load_dataset()

    def test_raw_data_path_is_directory_raises_valueerror(self, tmp_path):
        dir_path = tmp_path / "raw_dir"
        dir_path.mkdir()

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = dir_path
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(ValueError, match="Expected a file, but received"):
            sim._load_dataset()

    def test_loads_dataset_from_csv(self, tmp_path, dummy_dataset):
        df = dummy_dataset

        raw_dataset = tmp_path / "raw_dataset.csv"
        df.to_csv(raw_dataset, index=False)

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_dataset
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        df_load = sim._load_dataset()

        pd.testing.assert_frame_equal(df_load, df)


class TestSplitProductionSimulationSave:
    def test_invalid_save_train_dataframe_raises_typeerror(self):
        df_train = pd.Series([1, 2, 3], name="feature_1")
        df_prod = pd.DataFrame({"feature_1": [1, 2, 3]})
        metadata = dict({"a": 1})

        datapath_config = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(TypeError, match="Series"):
            sim._save(df_train, df_prod, metadata)

    def test_invalid_save_prod_dataframe_raises_typeerror(self):
        df_train = pd.DataFrame({"feature_1": [1, 2, 3]})
        df_prod = pd.Series([1, 2, 3], name="feature_1")
        metadata = dict({"a": 1})

        datapath_config = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(TypeError, match="Series"):
            sim._save(df_train, df_prod, metadata)

    def test_invalid_save_metadata_dict_raises_typeerror(self):
        df_train = pd.DataFrame({"feature_1": [1, 2, 3]})
        df_prod = pd.DataFrame({"feature_1": [1, 2, 3]})
        metadata = list({"a": 1})

        datapath_config = MagicMock(spec=DataPathConfig)
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(TypeError, match="dict"):
            sim._save(df_train, df_prod, metadata)

    def test_save_persists_all_outputs_correctly(self, tmp_path):
        train_path = tmp_path / "training_dataset.csv"
        prod_path = tmp_path / "holdout_dataset.csv"
        metadata_path = tmp_path / "metadata.json"

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.split_training_path = train_path
        datapath_config.split_production_path = prod_path
        datapath_config.split_metadata_path = metadata_path
        split_config = MagicMock(spec=SplitHoldoutConfig)

        df_train = pd.DataFrame({"feature_1": [1, 2, 3]})
        df_prod = pd.DataFrame({"feature_1": [1, 2, 3]})
        metadata = dict({"a": 1})

        sim = SplitProductionSimulation(datapath_config, split_config)

        sim._save(df_train, df_prod, metadata)

        df_train_read = pd.read_csv(train_path)
        df_prod_read = pd.read_csv(prod_path)
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata_read = json.load(file)

        pd.testing.assert_frame_equal(df_train_read, df_train)
        pd.testing.assert_frame_equal(df_prod_read, df_prod)
        assert metadata_read == metadata

    def test_save_creates_parent_directory_if_not_exists(self, tmp_path):
        save_dir = tmp_path / "interim"

        train_path = save_dir / "training_dataset.csv"
        prod_path = save_dir / "holdout_dataset.csv"
        metadata_path = save_dir / "metadata.json"

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.split_training_path = train_path
        datapath_config.split_production_path = prod_path
        datapath_config.split_metadata_path = metadata_path

        split_config = MagicMock(spec=SplitHoldoutConfig)

        df_train = pd.DataFrame({"feature_1": [1, 2, 3]})
        df_prod = pd.DataFrame({"feature_1": [1, 2, 3]})
        metadata = dict({"a": 1})

        sim = SplitProductionSimulation(datapath_config, split_config)

        assert not save_dir.exists()

        sim._save(df_train, df_prod, metadata)

        assert save_dir.exists()
        assert save_dir.is_dir()


class TestSplitProductionSimulationExecute:
    def test_execute_returns_training_dataframe(self, tmp_path, dummy_dataset):
        raw_data_path = tmp_path / "raw_dataset.csv"
        train_path = tmp_path / "training_dataset.csv"
        prod_path = tmp_path / "holdout_dataset.csv"
        metadata_path = tmp_path / "metadata.json"

        df = dummy_dataset
        df.to_csv(raw_data_path, index=False)

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path
        datapath_config.split_training_path = train_path
        datapath_config.split_production_path = prod_path
        datapath_config.split_metadata_path = metadata_path

        split_config = MagicMock(spec=SplitHoldoutConfig)
        split_config.ref_col = "MJD"
        split_config.holdout_split = 0.5

        sim = SplitProductionSimulation(datapath_config, split_config)

        df_train = sim.execute()

        assert set(df_train.index).issubset(set(df.index))
        assert isinstance(df_train, pd.DataFrame)
        assert len(df_train) == len(df) - int(len(df) * split_config.holdout_split)

        assert train_path.exists()
        assert prod_path.exists()
        assert metadata_path.exists()

        saved_train = pd.read_csv(train_path)

        pd.testing.assert_frame_equal(saved_train.reset_index(drop=True), df_train.reset_index(drop=True))

    def test_execute_propagates_load_dataset_error(self, tmp_path):
        raw_data_path = tmp_path / "raw_dataset.csv"

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path
        split_config = MagicMock(spec=SplitHoldoutConfig)

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(FileNotFoundError, match="Raw dataset file not found on disk"):
            sim.execute()

    def test_execute_propagates_split_error(self, tmp_path, dummy_dataset):
        raw_data_path = tmp_path / "raw_dataset.csv"
        train_path = tmp_path / "training_dataset.csv"
        prod_path = tmp_path / "holdout_dataset.csv"
        metadata_path = tmp_path / "metadata.json"

        df = dummy_dataset
        df.to_csv(raw_data_path, index=False)

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path
        datapath_config.split_training_path = train_path
        datapath_config.split_production_path = prod_path
        datapath_config.split_metadata_path = metadata_path

        split_config = MagicMock(spec=SplitHoldoutConfig)
        split_config.ref_col = "MJD"
        split_config.holdout_split = 0.0

        sim = SplitProductionSimulation(datapath_config, split_config)

        with pytest.raises(ValueError, match="The holdout fraction must be between"):
            sim.execute()

    def test_execute_metadata_includes_path_keys(self, tmp_path, dummy_dataset):
        raw_data_path = tmp_path / "raw_dataset.csv"
        train_path = tmp_path / "training_dataset.csv"
        prod_path = tmp_path / "holdout_dataset.csv"
        metadata_path = tmp_path / "metadata.json"

        df = dummy_dataset
        df.to_csv(raw_data_path, index=False)

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path
        datapath_config.split_training_path = train_path
        datapath_config.split_production_path = prod_path
        datapath_config.split_metadata_path = metadata_path

        split_config = MagicMock(spec=SplitHoldoutConfig)
        split_config.ref_col = "MJD"
        split_config.holdout_split = 0.5

        sim = SplitProductionSimulation(datapath_config, split_config)

        _ = sim.execute()

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        assert "raw_path" in metadata
        assert "training_path" in metadata
        assert "production_path" in metadata

    @patch.object(SplitProductionSimulation, "_save")
    def test_execute_calls_save_with_correct_arguments(self, mock_save, tmp_path, dummy_dataset):
        raw_data_path = tmp_path / "raw_dataset.csv"
        train_path = tmp_path / "training_dataset.csv"
        prod_path = tmp_path / "holdout_dataset.csv"
        metadata_path = tmp_path / "metadata.json"

        df = dummy_dataset
        df.to_csv(raw_data_path, index=False)

        datapath_config = MagicMock(spec=DataPathConfig)
        datapath_config.raw_data_path = raw_data_path
        datapath_config.split_training_path = train_path
        datapath_config.split_production_path = prod_path
        datapath_config.split_metadata_path = metadata_path

        split_config = MagicMock(spec=SplitHoldoutConfig)
        split_config.ref_col = "MJD"
        split_config.holdout_split = 0.5

        sim = SplitProductionSimulation(datapath_config, split_config)

        _ = sim.execute()

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        _, _, metadata_passed = call_args[0]

        assert "raw_path" in metadata_passed
        assert "training_path" in metadata_passed
        assert "production_path" in metadata_passed
