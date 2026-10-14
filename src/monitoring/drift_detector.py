import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from configs.paths import PROJECT_ROOT, DataPathConfig
from configs.pipeline_config import CleanPreprocessingConfig, DriftDetectionConfig
from configs.pipeline_loader import load_cleaning_preprocessing_config, load_split_training_config, load_drift_detection_config
from src.data.preprocessing import build_stateless_cleaning_pipeline
from src.predict.predictor import AstroPredict
from src.utils.logger import restore_logging_after_mlflow, setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


class AstroDriftDetector:
    def __init__(
        self,
        predictor: AstroPredict,
        train_dataset_path: str | Path,
        path_config: DataPathConfig | None = None,
        clean_config: CleanPreprocessingConfig | None = None,
        drift_config: DriftDetectionConfig | None = None,
        verbose: int = 2,
    ):
        """
        Class for detecting data drift in production datasets compared to a training dataset.

        Args:
            predictor (AstroPredict): The predictor object used for making predictions.
            train_dataset_path (str | Path): Path to the training dataset.
            path_config (DataPathConfig | None): Configuration for data paths. If None, a default configuration will be used.
            clean_config (CleanPreprocessingConfig | None): Configuration for data cleaning and preprocessing. If None, a default configuration will be used.
            drift_config (DriftDetectionConfig | None): Configuration for drift detection thresholds. If None, a default configuration will be used.
            verbose (int): Verbosity level for logging. Default is 2:
                - 0 for logging all messages
                - 1 for minimal logging
                - 2 for no logging
        
        Raises:
            ValueError: If the training dataset does not exist or is empty.
            ValueError: If the training dataset features do not match the expected schema.
            ValueError: If the drift detection thresholds are not in the (0, 1) range.  
        """
        validate_type(
            predictor=(predictor, AstroPredict),
            train_dataset_path=(train_dataset_path, (str, Path)),
            path_config=(path_config, (DataPathConfig, type(None))),
            clean_config=(clean_config, (CleanPreprocessingConfig, type(None))),
            drift_config=(drift_config, (DriftDetectionConfig, type(None))),
            verbose=(verbose, int),
        )

        self.verbose = verbose
        if self.verbose == 1:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 2:
            logger.setLevel(logging.ERROR)
        
        # Best model versionaned in MLFlow
        self.predictor = predictor

        # Configuration
        self.path_config = DataPathConfig() if path_config is None else path_config
        self.clean_config = load_cleaning_preprocessing_config() if clean_config is None else clean_config
        self.drift_config = load_drift_detection_config() if drift_config is None else drift_config

        # Drift detection thresholds
        self.ks_threshold = self.drift_config.ks_threshold
        self.max_shift_class = self.drift_config.max_shift_class
        self.max_drift_ratio = self.drift_config.max_drift_ratio

        # Validate threshold for Kolmogorov-Smirnov test
        if self.ks_threshold >= 1 or self.ks_threshold <= 0:
            raise ValueError("The ks_threshold must be in (0, 1) range!")

        # Validate thresholds for class shift and drift ratio
        if self.max_shift_class >= 1 or self.max_shift_class <= 0:
            raise ValueError("The max_shift_class must be in (0, 1) range!")

        # Validate thresholds for class shift and drift ratio
        if self.max_drift_ratio >= 1 or self.max_drift_ratio <= 0:
            raise ValueError("The max_drift_ratio must be in (0, 1) range!")


        # Path training dataset
        self.train_dataset_path = Path(train_dataset_path)

        # Load split configuration
        self.split_config = load_split_training_config()

        # Load metadata for raw and processed datasets to validate the schema
        with open(self.path_config.raw_data_metadata, "r", encoding="utf-8") as file:
            self.metadata_raw_data = json.load(file)
        with open(self.path_config.processed_data_metadata, "r", encoding="utf-8") as file:
            self.metadata_processed_data = json.load(file)

        # Remove the target column from the metadata for raw and processed datasets to validate 
        # the schema without the target variable
        self.metadata_raw_data["columns"].remove(self.split_config.target_column)
        self.metadata_processed_data["columns"].remove(self.split_config.target_column)

        # Load training dataset
        self.df_train = self._load_baseline_processed()

        # Predict on the training dataset to get predicted classes for drift detection
        self.y_train_pred = self.predictor.predict(X=self.df_train, cleaned_dataset=True)["prediction"]

        # Calculate class proportions for the training dataset
        self.train_class_proportions = (
            self.df_train[self.split_config.target_column].value_counts(normalize=True).sort_index()
        )

        # Calculate predicted class proportions for the training dataset
        self.train_pred_proportions = self.y_train_pred.value_counts(normalize=True).sort_index()

    def _clean_and_validate_dataset(self, df: pd.DataFrame, original_path: str | Path) -> pd.DataFrame:
        """
        Cleans and validates the input dataset against the expected schema.
        
        Args:
            df (pd.DataFrame): The input dataset to be cleaned and validated.
            original_path (str | Path): The original path of the dataset, used for logging purposes.
            
        Returns:
            pd.DataFrame: The cleaned and validated dataset.
            
        Raises:
            ValueError: If the features of the dataset do not match the expected schema."""
        restore_logging_after_mlflow()
        validate_type(
            df=(df, pd.DataFrame),
            original_path=(original_path, (str, Path)),
        )

        # Check if the target column is present in the dataset and drop it for schema validation
        if self.split_config.target_column in df.columns.to_list():
            df_copy = df.drop(self.split_config.target_column, axis=1)
        # If the target column is not present, create a copy of the dataset for schema validation
        else:
            df_copy = df.copy()

        # Validate the dataset schema against the expected schema for raw and processed datasets
        if df_copy.columns.to_list() == self.metadata_raw_data["columns"]:
            clean_pipeline = build_stateless_cleaning_pipeline(self.clean_config, verbose=False)
            df_cleaned = clean_pipeline.transform(df)
            logger.info(f"The dataset in '{original_path}' has been cleaned.")
        # If the dataset already matches the expected schema for processed data, no cleaning is needed
        elif df_copy.columns.to_list() == self.metadata_processed_data["columns"]:
            df_cleaned = df.copy()
            logger.info(f"The dataset in '{original_path}' is already cleaned.")
        else:
            raise ValueError("Features of training dataset do not match expected schema.")

        # If the group column is present in the cleaned dataset, drop it as it's not needed for drift detection
        if self.split_config.col_group in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(self.split_config.col_group, axis=1)

        return df_cleaned

    def _load_baseline_processed(self) -> pd.DataFrame:
        """
        Loads the baseline processed training dataset from the specified path.

        Returns:
            pd.DataFrame: The loaded and cleaned training dataset.

        Raises:
            ValueError: If the training dataset does not exist or is empty.
        """
        restore_logging_after_mlflow()

        # Validate if the training dataset exists in the specified path. If not, raise a ValueError.
        if not Path(self.train_dataset_path).exists():
            raise ValueError(f"Training dataset does not exist in the path: {self.train_dataset_path}")

        # Load the training dataset using pandas and log the action.
        train_dataset = pd.read_csv(self.train_dataset_path)

        logger.info(f"Loaded train dataset from: {self.train_dataset_path.relative_to(PROJECT_ROOT)}")

        # Validate if the training dataset is empty. If it is, raise a ValueError.
        if len(train_dataset) == 0:
            raise ValueError("Training dataset empty.")

        return self._clean_and_validate_dataset(train_dataset, self.train_dataset_path.relative_to(PROJECT_ROOT))

    def _load_dataset(
        self,
        dataset_path: str | Path | None = None,
        dir_path: str | Path | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Loads a dataset from a specified file path or directory path. If a file path is provided, 
        it loads a single dataset. If a directory path is provided, it loads multiple datasets from 
        the specified directory.
        
        Args:
            dataset_path (str | Path | None): The path to the dataset file. 
                If None, the method will attempt to load datasets from the specified directory path.
            dir_path (str | Path | None): The path to the directory containing multiple dataset files. 
                If None, the method will attempt to load a single dataset from the specified file path.

        Returns:
            pd.DataFrame | dict[str, pd.DataFrame]: If a file path is provided, it returns a single DataFrame 
                containing the loaded dataset. If a directory path is provided, it returns a dictionary where the 
                keys are the names of the dataset files and the values are the corresponding DataFrames containing 
                the loaded datasets.
        
        Raises:
            ValueError: If neither a dataset file path nor a directory path is provided.
        """
        validate_type(
            dataset_path=(dataset_path, (str, Path, type(None))),
            dir_path=(dir_path, (str, Path, type(None))),
        )

        # Check if both dataset_path and dir_path are None. 
        # If so, raise a ValueError indicating that no production file path is defined.
        if dataset_path is None and dir_path is None:
            raise ValueError("No production file path defined")

        ##################################
        ##### Single file production #####
        ##################################
        # If a dataset file path is provided, load the dataset from that file.
        if dataset_path is not None:
            logger.info(f"Load FULL production dataset from: {Path(dataset_path).relative_to(PROJECT_ROOT)}")
            prod_dataset = pd.read_csv(dataset_path)
            if len(prod_dataset) == 0:
                raise ValueError("Production dataset is empty!")
            return prod_dataset

        ##################################
        ##### Batch files production #####
        ##################################

        # Load metadata for batch files to get the list of batch file names.
        with open(self.path_config.split_production_batch_metadata_path, "r", encoding="utf-8") as file:
            metadata_dir_batchs = json.load(file)

        # If a directory path is provided, load all batch datasets from that directory.
        if dir_path is not None:
            batchs_name = metadata_dir_batchs["path_batch_files"]
            batchs_dataset = {}
            logger.info(f"Load BATCHS production datasets from: {Path(dir_path).relative_to(PROJECT_ROOT)}")
            # Load each batch dataset from the specified directory and store it in a dictionary.
            for batch_name in batchs_name:
                batchs_dataset[batch_name] = pd.read_csv(dir_path / batch_name)
                if len(batchs_dataset[batch_name]) == 0:
                    raise ValueError(f"Production batch dataset is empty: {batch_name}")
            return batchs_dataset

    def _detect_features_drift(self, prod_dataset: pd.DataFrame) -> dict:
        """
        Detects drift in the features of the production dataset compared to the training dataset using the 
        Kolmogorov-Smirnov test for continuous features. It calculates the drift ratio and identifies drifted 
        features based on the specified thresholds.
        
        Args:
            prod_dataset (pd.DataFrame): The production dataset to be analyzed for feature drift. 

        Returns:
            dict: A dictionary containing the results of the feature drift analysis, including:
                - "is_dataset_drifted" (bool): Indicates whether the dataset is drifted based on the drift ratio and threshold.
                - "drift_ratio" (float): The ratio of drifted features to total features.
                - "drifted_features_count" (int): The number of features that are detected as drifted.
                - "total_features_count" (int): The total number of features analyzed for drift.
                - "drifted_features" (list[str]): A list of feature names that are detected as drifted.
                - "feature_details" (dict): A dictionary containing detailed information for each feature, 
                    including the Kolmogorov-Smirnov statistic and whether drift was detected for that feature.
        
        Raises:
            ValueError: If the input production dataset is not a pandas DataFrame. 
        """
        validate_type(prod_dataset=(prod_dataset, pd.DataFrame))

        # Initialize dictionaries to store feature details and drifted features.
        feature_details: dict[str, Any] = {}
        drifted_features: list[str] = []

        # Iterate through each feature in the production dataset and perform the 
        # Kolmogorov-Smirnov test against the training dataset.
        for col in prod_dataset.columns:
            # Skip the target column as we are only interested in feature drift.
            if col != self.split_config.target_column:
                # Extract the non-null values of the feature from both the training and production datasets.
                s_train = self.df_train[col].dropna()
                s_prod = prod_dataset[col].dropna()

                # Perform the Kolmogorov-Smirnov test to compare the distributions of the feature in the training 
                # and production datasets.
                ks_result = ks_2samp(s_train, s_prod)
                is_drift = bool(ks_result.statistic > self.ks_threshold)

                # If drift is detected for the feature, add it to the list of drifted features.
                if is_drift:
                    drifted_features.append(col)

                # Store the results of the Kolmogorov-Smirnov test and drift detection for the feature in the 
                # feature_details dictionary.
                feature_details[col] = {
                    "ks_statistic": round(float(ks_result.statistic), 4),
                    "drift_detected": is_drift,
                }

        # Calculate the drift ratio as the number of drifted features divided by the total number of features.
        drift_ratio = len(drifted_features) / (len(feature_details))
        is_dataset_drifted = bool(drift_ratio > self.max_drift_ratio)

        logger.info("Features variables drift analysis performed.")

        return {
            "is_dataset_drifted": is_dataset_drifted,
            "drift_ratio": round(drift_ratio, 4),
            "drifted_features_count": len(drifted_features),
            "total_features_count": len(feature_details),
            "drifted_features": drifted_features,
            "feature_details": feature_details,
        }

    def _detect_target_drift(self, prod_class: pd.Series) -> dict:
        """
        Detects drift in the target variable of the production dataset compared to the training dataset by
        calculating the maximum class shift. It compares the observed class proportions in the production dataset
        with the expected class proportions from the training dataset and determines if drift is detected based on
        the specified threshold for maximum class shift.
        
        Args:
            prod_class (pd.Series): The target variable from the production dataset to be analyzed for drift
        
        Returns:
            dict: A dictionary containing the results of the target variable drift analysis, including: 
                - "max_class_shift" (float): The maximum absolute difference between the observed and expected class proportions.
                - "target_drift_detected" (bool): Indicates whether drift is detected based on the maximum class shift and threshold.
                - "observed_proportions" (dict): A dictionary containing the observed class proportions in the production dataset.
                - "expected_proportions" (dict): A dictionary containing the expected class proportions from the training dataset.
        
        Raises:
            ValueError: If the input production target variable is not a pandas Series.
        """
        validate_type(prod_class=(prod_class, pd.Series))

        # Calculate the observed class proportions in the production dataset and reindex it to match the training class proportions.
        prod_proportions = prod_class.value_counts(normalize=True).sort_index()
        prod_proportions = prod_proportions.reindex(self.train_class_proportions.index, fill_value=0)

        # Calculate the maximum class shift as the maximum absolute difference between the observed and expected class proportions.
        dict_train_class_proportions = self.train_class_proportions.to_dict()
        max_class_shift = max(
            abs(value - dict_train_class_proportions[key]) 
            for key, value in prod_proportions.items()
        )

        logger.info("Target variable drift analysis performed.")

        return {
            "max_class_shift": round(float(max_class_shift), 4),
            "target_drift_detected": bool(max_class_shift > self.max_shift_class),
            "observed_proportions": prod_class.value_counts(normalize=True).round(4).to_dict(),
            "expected_proportions": self.train_class_proportions.round(4).to_dict(),
        }

    def _detect_prediction_drift(self, df_prod_cleaned: pd.DataFrame) -> dict:
        """
        Detects drift in the predictions of the production dataset compared to the training dataset by
        calculating the maximum class shift in the predicted classes. It compares the observed predicted class
        proportions in the production dataset with the expected predicted class proportions from the training dataset
        and determines if drift is detected based on the specified threshold for maximum class shift.
        
        Args:
            df_prod_cleaned (pd.DataFrame): The cleaned production dataset to be analyzed for prediction
                drift. It should contain the same features as the training dataset, excluding the target variable.
        
        Returns:
            dict: A dictionary containing the results of the prediction drift analysis, including:
                - "max_class_shift" (float): The maximum absolute difference between 
                    the observed and expected predicted class proportions.
                - "prediction_drift_detected" (bool): Indicates whether drift is detected based 
                on the maximum class shift and threshold.
                - "observed_proportions" (dict): A dictionary containing the observed predicted 
                class proportions in the production dataset.
                - "expected_proportions" (dict): A dictionary containing the expected predicted 
                    class proportions from the training dataset.
        """
        validate_type(df_prod_cleaned=(df_prod_cleaned, pd.DataFrame))

        # Make predictions on the cleaned production dataset using the predictor and calculate the 
        # observed predicted class proportions.
        y_prod_pred = self.predictor.predict(X=df_prod_cleaned, cleaned_dataset=True)["prediction"]
        prod_proportions = y_prod_pred.value_counts(normalize=True).sort_index()
        prod_proportions = prod_proportions.reindex(self.train_pred_proportions.index, fill_value=0).to_dict()

        # Calculate the maximum class shift as the maximum absolute difference between the observed and expected 
        # predicted class proportions.
        dict_train_pred_proportions = self.train_pred_proportions.to_dict()

        # Calculate the maximum class shift as the maximum absolute difference between 
        # the observed and expected predicted class proportions.
        max_class_shift = max(
            abs(value - dict_train_pred_proportions[key]) 
            for key, value in prod_proportions.items()
        )

        logger.info("Drift analysis on class predictions performed.")

        return {
            "max_class_shift": round(float(max_class_shift), 4),
            "prediction_drift_detected": bool(max_class_shift > self.max_shift_class),
            "observed_proportions": y_prod_pred.value_counts(normalize=True).round(4).to_dict(),
            "expected_proportions": self.y_train_pred.value_counts(normalize=True).sort_index().round(4).to_dict(),
        }

    def detect_data_drift(self, dataset_path: str | Path, output_report_path: str | Path | None = None) -> dict:
        """
        Detects data drift for a single production dataset compared to the training dataset. It performs feature drift,
        target drift, and prediction drift analysis, and generates a report containing the results of the drift detection.
        
        Args:
            dataset_path (str | Path): The path to the production dataset file to be analyzed for
                data drift.
            output_report_path (str | Path | None): The path to save the drift detection report. 
                If None, the report will not be saved to a file.
        
        Returns:
            dict: A dictionary containing the results of the data drift detection, including:
                - "features_drift": A dictionary containing the results of the feature drift analysis."
                - "prediction_drift": A dictionary containing the results of the prediction drift analysis.
                - "true_target_drift": A dictionary containing the results of the target variable drift analysis    
        """
        validate_type(
            dataset_path=(dataset_path, (str, Path)),
            output_report_path=(output_report_path, (str, Path, type(None))),
        )

        logger.info(f"Starting data drift detection for file: '{Path(dataset_path).relative_to(PROJECT_ROOT)}'")

        # Check if the production dataset file exists. If not, raise a ValueError.
        if not Path(dataset_path).exists():
            raise ValueError(f"File not found: {dataset_path}")

        # Check if the production dataset file is empty. If it is, raise a ValueError.
        batch_path_dir = Path(dataset_path).parent
        name_file = Path(dataset_path).relative_to(batch_path_dir)
        report_data_drift = {}

        # Load the production dataset and clean it using the cleaning and validation method.
        df_prod = self._load_dataset(dataset_path=dataset_path)
        clean_dataset = self._clean_and_validate_dataset(df_prod, Path(dataset_path).relative_to(PROJECT_ROOT))
        report_data_drift[f"{name_file}"] = self.reports_data_drift(clean_dataset)

        # Save the drift detection report to the specified output path if provided.
        if output_report_path is not None:
            Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as file:
                json.dump(report_data_drift, file, indent=4)
            logger.info(f"The full drift report has been saved in: {Path(output_report_path).relative_to(PROJECT_ROOT)}")

        return report_data_drift

    def detect_batchs_data_drift(
            self, 
            dir_path: str | Path,
            output_report_path: str | Path | None = None
        ) -> dict:
        """
        Detects data drift for multiple production datasets (batches) in a specified directory compared to
        the training dataset. It performs feature drift, target drift, and prediction drift analysis for each batch,
        and generates a report containing the results of the drift detection for all batches.
        
        Args:
            dir_path (str | Path): The path to the directory containing the production dataset 
                files (batches) to be analyzed for data drift.
            output_report_path (str | Path | None): The path to save the drift detection report for all batches. 
                If None, the report will not be saved to a file.

        Returns:
            dict: A dictionary containing the results of the data drift detection for all batches, where each
                key is the name of a batch file and the corresponding value is a dictionary containing the results
                of the drift detection for that batch, including:
                - "features_drift": A dictionary containing the results of the feature drift analysis for the batch.
                - "prediction_drift": A dictionary containing the results of the prediction drift analysis for the batch.
                - "true_target_drift": A dictionary containing the results of the target variable drift analysis for the batch.
        """
        validate_type(
            dir_path=(dir_path, (str, Path)),
            output_report_path=(output_report_path, (str, Path, type(None))),
        )

        logger.info(
            f"Starting batch data drift detection for files in the directory:"
            f" '{Path(dir_path).relative_to(PROJECT_ROOT)}'"
        )

        # Check if the specified directory exists. If not, raise a ValueError.
        if not Path(dir_path).exists():
            raise ValueError(f"Attention, does not exist the directory of batch production in: {dir_path}")
        if len(os.listdir(dir_path)) == 0:
            raise ValueError(f"Attention, the batch directory is empty: {dir_path}")

        # Load all production datasets (batches) from the specified directory and store them in a dictionary.
        dict_prod = self._load_dataset(dir_path=dir_path)
        daily_report_data_drift: dict[str, dict[str, Any]] = {}

        # Iterate through each batch dataset in the dictionary and perform drift detection for each batch.
        for batch_name, batch_dataset in dict_prod.items():
            restore_logging_after_mlflow()
            # Clean and validate the batch dataset using the cleaning and validation method.
            clean_batch_dataset = self._clean_and_validate_dataset(
                batch_dataset, (Path(dir_path) / batch_name).relative_to(PROJECT_ROOT)
            )
            daily_report_data_drift[f"{batch_name}"] = self.reports_data_drift(clean_batch_dataset)

        # Save the drift detection report for all batches to the specified output path if provided.
        if output_report_path is not None:
            Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as file:
                json.dump(daily_report_data_drift, file, indent=4)
            logger.info(f"The full drift report has been saved in: {Path(output_report_path).relative_to(PROJECT_ROOT)}")

        return daily_report_data_drift

    def reports_data_drift(self, prod_dataset: pd.DataFrame) -> dict:
        """
        Generates a comprehensive report on data drift for a given production dataset. It performs feature drift,
        target drift, and prediction drift analysis, and compiles the results into a structured report.
        
        Args:
            prod_dataset (pd.DataFrame): The production dataset to be analyzed for data drift.
        
        Returns:
            dict: A dictionary containing the results of the data drift analysis, including:
                - "features_drift": A dictionary containing the results of the feature drift analysis."
                - "prediction_drift": A dictionary containing the results of the prediction drift analysis.
                - "true_target_drift": A dictionary containing the results of the target variable drift analysis
        """
        validate_type(prod_dataset=(prod_dataset, pd.DataFrame))

        # Perform feature drift detection on the production dataset and store the results in a dictionary.
        features_drift_info = self._detect_features_drift(prod_dataset)
        prediction_drift_info = self._detect_prediction_drift(prod_dataset)

        # Perform target drift detection only if the target column is present in the production dataset.
        target_drift_info = None
        if self.split_config.target_column in prod_dataset.columns:
            target_drift_info = self._detect_target_drift(prod_dataset[self.split_config.target_column])

        report = {
            "features_drift": features_drift_info,
            "prediction_drift": prediction_drift_info,
            "true_target_drift": target_drift_info,
        }

        return report
