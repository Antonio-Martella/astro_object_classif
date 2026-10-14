import json
import logging
import os
import pandas as pd
import shutil
from pathlib import Path

from configs.paths import DataPathConfig
from src.data.make_datasets import run_cleaning
from src.utils.validate_type import validate_type
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

def _get_batches_to_consolidate(
        trigger_batch_name: str, 
        path_config: DataPathConfig,
    ) -> list:  
    """
    Retrieves the list of files to add to the training dataset.
    
    Args:
        trigger_batch_name (str) : Name of the batch file that triggered the retraining.
        path_config (DataPathConfig) : Configuration for data paths.
    
    Return: 
        list : List of all batch filenames to be included in the training dataset (from the first 
            one up to the trigger file).

    Raises:
        FileNotFoundError : If the trigger file is not found in the production directory.

    """
    files_name = [f for f in os.listdir(path_config.split_daily_batches_path) if f.endswith(".csv")] 
    files_name.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Check if the trigger batch file exists in the list of batch files
    if trigger_batch_name not in files_name:
        raise FileNotFoundError(f"Batch file {trigger_batch_name} not found. It might have already been archived.")

    # Find the index of the trigger batch file and create a sublist of all batch files up to and including the 
    # trigger batch
    idx = files_name.index(trigger_batch_name)
    batchs_dataset_file = files_name[0:idx+1]

    logger.info(f"Batch files to be consolidated into the training dataset: {batchs_dataset_file}")

    return batchs_dataset_file


def _merge_and_save_training(
        batchs_dataset_file: list,
        metadata_archived_batch: dict,
        metadata_interim_training: dict,
        path_config: DataPathConfig,
    ) -> tuple[dict, dict]:
    """
    Merges the selected batch files with the existing training dataset and updates the metadata accordingly.
    
    Args:
        batchs_dataset_file (list) : List of batch filenames to be merged with the training dataset.
        metadata_archived_batch (dict) : Metadata for the archived batch files.
        metadata_interim_training (dict) : Metadata for the interim training dataset.
        path_config (DataPathConfig) : Configuration for data paths.
        
    Returns:
        tuple : Updated metadata for archived batch files and interim training dataset.
    """
    validate_type(
        batchs_dataset_file=(batchs_dataset_file, list),
        metadata_archived_batch=(metadata_archived_batch, dict),
        metadata_interim_training=(metadata_interim_training, dict),
        path_config=(path_config, DataPathConfig)
    )

    # Load the existing training dataset
    df_training = pd.read_csv(path_config.split_training_path)

    # Concatenate the training dataset with the selected batch files
    for i, file in enumerate(batchs_dataset_file):
        batch = pd.read_csv(path_config.split_daily_batches_path / file)
        if "class" not in batch.columns:
            raise ValueError(f"The batch file {file} does not contain the required 'class' column.")
        if i == 0:
            df_training_upload = pd.concat([df_training, batch], ignore_index=True)
        else:
            df_training_upload = pd.concat([df_training_upload, batch], ignore_index=True)

        metadata_archived_batch["number_file"] += 1
        metadata_archived_batch["number_records"] += len(batch)

    metadata_archived_batch["archived_batch_files"].extend(batchs_dataset_file)

    # Training metadata upgrade and save
    metadata_interim_training["current_n_train"] = len(df_training_upload)
    if "MJD" in df_training_upload.columns:
        metadata_interim_training["current_train_mjd_max"] = int(df_training_upload["MJD"].max())

    # Save the new dataset
    df_training_upload.to_csv(path_config.split_training_path, index=False)

    _make_processed_dataset(df_training_upload, path_config)

    logger.info(
        f"Training dataset updated with {len(batchs_dataset_file)} batch files. "
        f"New training dataset size: {len(df_training_upload)} records."
    )

    return metadata_archived_batch, metadata_interim_training


def _make_processed_dataset(df_training: pd.DataFrame, path_config: DataPathConfig) -> None:
    """
    Processes the training dataset and saves the processed version along with its metadata.
    
    Args:
        df_training (pd.DataFrame) : The training dataset to be processed.
        path_config (DataPathConfig) : Configuration for data paths.
    """
    # Process the training dataset and save the processed version along with its metadata
    _ = run_cleaning(df_training, path_config)

    logger.info(
        f"Processed dataset saved at {path_config.processed_data} with metadata at "
        f"{path_config.processed_data_metadata}."
    )


def _move_batches_to_archive(batchs_dataset_file: list[str], path_config: DataPathConfig) -> None:
    """
    Moves the specified batch files from the daily batch directory to the archived batch directory.
    
    Args:
        batchs_dataset_file (list[str]) : List of batch filenames to be moved to the archived directory.
        path_config (DataPathConfig) : Configuration for data paths.
    
    Raises:
        FileNotFoundError : If any of the specified batch files are not found in the daily batch directory.
    """
    # Move the specified batch files from the daily batch directory to the archived batch directory
    for file in batchs_dataset_file:
        source = Path(path_config.split_daily_batches_path / file)
        destination = Path(path_config.split_archived_batches_path / file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)

    logger.info(
        f"Moved {len(batchs_dataset_file)} batch files to the archived directory: "
        f"{path_config.split_archived_batches_path}"
    )


def _load_all_metadata(path_config: DataPathConfig) -> tuple[dict, dict, dict]:
    """
    Loads the metadata for archived batch files, daily batch files, and interim training dataset.
    
    Args:
        path_config (DataPathConfig) : Configuration for data paths.
    
    Returns:
        tuple[dict, dict, dict] : A tuple containing the metadata for archived batches, daily batches, and interim training.

    Raises:
        FileNotFoundError : If any of the metadata files are not found.
    """
    # Load the metadata for archived batch files, daily batch files, and interim training dataset
    try: 
        with open(path_config.split_archived_batches_metadata_path, "r", encoding="utf-8") as file:
            metadata_archived_batch = json.load(file)
    except FileNotFoundError:
        metadata_archived_batch = {
            "number_file": 0,
            "number_records": 0,
            "archived_batch_files": []
        }

    # Load the metadata for daily batch files
    try:
        with open(path_config.split_daily_batches_metadata_path, "r", encoding="utf-8") as file:
            metadata_daily_batch = json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find daily_batch metadata in {path_config.split_daily_batches_metadata_path}"
        ) from e
    
    # Training dataset upload
    try: 
        with open(path_config.split_interim_metadata_path, "r", encoding="utf-8") as file:
            metadata_interim_training = json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find interim training metadata in {path_config.split_interim_metadata_path}"
        ) from e

    logger.info("Metadata for archived batches, daily batches, and interim training dataset loaded successfully.")

    return metadata_archived_batch, metadata_daily_batch, metadata_interim_training, 


def _update_all_metadata(
        path_config: DataPathConfig,
        metadata_archived_batch: dict, 
        metadata_daily_batch: dict, 
        metadata_interim_training: dict,
        batchs_dataset_file: list,
    ) -> None:
    """
    Updates and saves the metadata for archived batch files, daily batch files, and interim training dataset.
    
    Args:
        path_config (DataPathConfig) : Configuration for data paths.
        metadata_archived_batch (dict) : Updated metadata for archived batch files.
        metadata_daily_batch (dict) : Updated metadata for daily batch files.
        metadata_interim_training (dict) : Updated metadata for interim training dataset.
        batchs_dataset_file (list) : List of batch filenames that were consolidated into the training dataset.
        
    Raises:
            FileNotFoundError: If any of the metadata files are not found.
    """
    # Archived metadata save
    with open(path_config.split_archived_batches_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_archived_batch, file, indent=4)

    # Daily_batch metadata upgrade and save
    metadata_daily_batch["number_file"] -= len(batchs_dataset_file)
    metadata_daily_batch["path_batch_files"] = \
        [x for x in metadata_daily_batch["path_batch_files"] if x not in batchs_dataset_file]

    # Daily_batch metadata save
    with open(path_config.split_daily_batches_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_daily_batch, file, indent=4)

    # Training metadata save
    with open(path_config.split_interim_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_interim_training, file, indent=4)

    logger.info(f"Metadata updated and saved for archived batches, daily batches, and interim training dataset.")


def consolidate_batches_for_retraining(trigger_batch_name: str, path_config: DataPathConfig | None = None) -> None:
    """
    Consolidates the specified batch files into the training dataset and updates the relevant metadata.
    
    Args:
        trigger_batch_name (str) : Name of the batch file that triggered the retraining.
        path_config (DataPathConfig | None) : Configuration for data paths. If None, a default DataPathConfig 
            instance will be created.
            
    Raises:
        FileNotFoundError : If the trigger batch file is not found in the daily batch directory.
        RuntimeError : If any error occurs during the consolidation process.
    """
    validate_type(
        trigger_batch_name=(trigger_batch_name, str),
        path_config=(path_config, (DataPathConfig, type(None))),
    )

    path_config=DataPathConfig() if path_config is None else path_config

    # Load all metadata
    metadata_archived_batch, metadata_daily_batch, metadata_interim_training = _load_all_metadata(path_config)

    # Get the list of batch files to consolidate based on the trigger batch name
    batchs_dataset_file=_get_batches_to_consolidate(trigger_batch_name, path_config)

    # Merge the selected batch files with the existing training dataset and update the metadata
    metadata_archived_batch, metadata_interim_training = _merge_and_save_training(
        batchs_dataset_file=batchs_dataset_file,
        metadata_archived_batch=metadata_archived_batch,
        metadata_interim_training=metadata_interim_training,
        path_config=path_config,
    )

    # Move the selected batch files from the daily batch directory to the archived batch directory
    _move_batches_to_archive(batchs_dataset_file, path_config)

    # Update and save all metadata files
    _update_all_metadata(
        path_config,
        metadata_archived_batch,
        metadata_daily_batch,
        metadata_interim_training,
        batchs_dataset_file
    )


if __name__ == "__main__":
    setup_logger()
    consolidate_batches_for_retraining("day_5.csv")