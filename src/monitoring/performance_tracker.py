import pandas as pd
import os
import json
from pathlib import Path
import logging

from configs.paths import DataPathConfig, PROJECT_ROOT
from configs.pipeline_loader import load_performances_monitoring_config
from configs.pipeline_config import PerformanceMonitoringConfig
from src.predict.predictor import AstroPredict
from src.utils.validate_type import validate_type
from src.utils.metrics import evaluate_classification_metrics
from src.utils.logger import setup_logger, restore_logging_after_mlflow

logger = logging.getLogger(__name__)


class AstroPerformanceTracker:
    def __init__(
            self, 
            predictor: AstroPredict,
            performance_monitoring_config: PerformanceMonitoringConfig | None = None,
            verbose: int = 2,
        ) -> None:
        """
        Class for tracking the performance of a model on production datasets and generating alerts 
        based on predefined thresholds.

        Args:
            predictor (AstroPredict): The predictor object used for making predictions.
            performance_monitoring_config (PerformanceMonitoringConfig | None): Configuration for performance monitoring.
                If None, a default configuration will be loaded.
            verbose (int): Verbosity level for logging. Default is 2:
                - 0 for logging all messages
                - 1 for minimal logging
                - 2 for no logging

        Raises:
            ValueError: If the number of unique classes in y_true does not match the number 
            of classes expected by the label encoder.
        """
        validate_type(
            predictor=(predictor, AstroPredict),
            performance_monitoring_config=(performance_monitoring_config, (PerformanceMonitoringConfig, type(None))),
            verbose=(verbose, int),
        )

        # Load the predictor
        self.predictor = predictor

        # Encoder definition
        self.label_encoder = self.predictor.label_encoder

        self.performance_monitoring_config = load_performances_monitoring_config() \
            if performance_monitoring_config is None else performance_monitoring_config
        restore_logging_after_mlflow() 

        # Load the metrics of the best model
        self.best_model_metrics = self.predictor.metrics_best_model()

        # Metric to monitor
        self.reference_metric = self.performance_monitoring_config.metric_to_monitor

        # Thresholds and relative drops for performance monitoring
        self.min_global_threshold = self.performance_monitoring_config.min_global_threshold
        self.min_sectorial_threshold = self.performance_monitoring_config.min_sectorial_threshold
        self.max_global_relative_drop = self.performance_monitoring_config.max_global_relative_drop
        self.max_sectorial_relative_drop = self.performance_monitoring_config.max_sectorial_relative_drop

        # Verbosity level for logging
        self.verbose = verbose
        if self.verbose == 1:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 2:
            logger.setLevel(logging.ERROR)

    def _dataset_scores_evaluation(self, X: pd.DataFrame, y_true: pd.Series) -> dict:
        """
        Evaluates the performance of the model on a production dataset.

        Args:
            X (pd.DataFrame): The input features for the production dataset.
            y_true (pd.Series): The true labels for the production dataset.

        Returns:
            dict: A dictionary containing the evaluated metrics.

        Raises:
            ValueError: If y_true is empty or if the number of unique classes in y_true
                        does not match the number of classes expected by the label encoder. 
            
        """
        validate_type(
            X=(X, pd.DataFrame),
            y_true=(y_true, pd.Series)
        )
        # Check if y_true is empty
        if y_true.empty:
            raise ValueError(
            "The vector with y_true of the production dataset " \
            "to be evaluated has not been passed.")

        logger.info(
            "The dataset has the true labels 'y_true = class', it is possible to evaluate the performances."
        )

        # Check if the number of unique classes in y_true matches the number of classes expected by the label encoder.
        if not all([x in list(self.label_encoder["classes"]) for x in list(y_true.unique())]):
            raise ValueError(
                "The number of unique classes in y_true does not match " \
                "the number of classes expected by the label encoder."
            )

        # Make the prediction on the production dataset 
        pred = self.predictor.predict(X)
        y_pred = pred["prediction"]

        # Calculates metrics on the production dataset
        batch_scores = evaluate_classification_metrics(X=X, y_true=y_true, y_pred=y_pred, label_encoder=self.label_encoder)

        logger.info("Calculated metrics")
        
        return batch_scores


    def _delta_analysis(self, prod_scores: dict) -> dict:
        """
        Compares the metrics of the production dataset with those of the best model and calculates the delta scores.
        
        Args:
            prod_scores (dict): A dictionary containing the metrics of the production dataset.
            
        Returns:
            dict: A dictionary containing the delta scores for each metric, including the production value,
                  the baseline value, the absolute delta, and the relative drop percentage.
                  
        Raises:
            ValueError: If the keys in the metric dictionaries of the best model and the production dataset do 
            not match.
        """
        validate_type(prod_scores=(prod_scores, dict))

        # Ensure that the production scores have the same keys as the best model metrics.
        if prod_scores.keys() != self.best_model_metrics.keys():
            raise ValueError(
                "The keys in the metric dictionaries of the best model and the production dataset do not match."
            )

        # Calculate the delta scores for each metric.
        delta_scores = {}
        for metric in prod_scores.keys():
            delta_scores[f"{metric}"] = {}
            delta_scores[f"{metric}"]["prod_value"] = prod_scores[f"{metric}"]
            delta_scores[f"{metric}"]["baseline_value"] = self.best_model_metrics[f"{metric}"]
            delta_scores[f"{metric}"]["absolute_delta"] = round(prod_scores[f"{metric}"] - self.best_model_metrics[f"{metric}"], 4)
            delta_scores[f"{metric}"]["relative_drop_pct"] = round((1 - prod_scores[f"{metric}"]/self.best_model_metrics[f"{metric}"]), 4)

        logger.info("Delta score between the reference metrics and those of the dataset, calculated.")

        return delta_scores

    def _threshold_and_alarm_engine(self, delta_scores: dict) -> dict:
        """
        Evaluates the delta scores against predefined thresholds and generates an alert report.
        
        Args:
            delta_scores (dict): A dictionary containing the delta scores for each metric.
            
        Returns:
            dict: A dictionary containing the alert report with severity and reasons.

        Raises:
            ValueError: If the reference metric is not present in the delta scores.
        """
        validate_type(delta_scores=(delta_scores, dict))

        # Initialize the report dictionary
        report = {
            "is_degraded": False,
            "severity": "OK",
            "reasons": []
        } 

        # Check if the reference metric is present in the delta scores.
        if self.reference_metric in delta_scores.keys():
            # Extract the delta scores for the reference metric.
            scores = delta_scores[f"{self.reference_metric}"] 
            # Check if the production value for the reference metric is below the minimum global threshold.
            if scores["prod_value"] <= self.min_global_threshold:
                report["is_degraded"] = True
                report["severity"] = "CRITICAL"
                report["reasons"].append(f"{self.reference_metric} current value {scores['prod_value']} is below safety floor {self.min_global_threshold}")
            # Check if the relative drop for the reference metric exceeds the maximum global relative drop.
            if scores["relative_drop_pct"] >= self.max_global_relative_drop:
                report["is_degraded"] = True
                report["severity"] = "CRITICAL"
                report["reasons"].append(f"{self.reference_metric} dropped by {scores['relative_drop_pct']}% (threshold: {self.max_global_relative_drop}%)")

            if report["severity"] == "OK":
                # Check sectorial thresholds for each label.
                list_metrics = ["f1", "precision", "recall"]
                for metric in list_metrics:
                    if metric in self.reference_metric:
                        for label in self.label_encoder["classes"]:
                            scores = delta_scores[f"{metric}_{label}"] 
                            if scores["prod_value"] <= self.min_sectorial_threshold:
                                report["severity"] = "WARNING"
                                report["reasons"].append(f"{metric}_{label} current value {scores['prod_value']} is below safety floor {self.min_sectorial_threshold}")
                            if scores["relative_drop_pct"] >= self.max_sectorial_relative_drop:
                                report["severity"] = "WARNING"
                                report["reasons"].append(f"{metric}_{label} dropped by {scores['relative_drop_pct']}% (threshold: {self.max_sectorial_relative_drop}%)")
                # Check sectorial thresholds for accuracy.
                if "accuracy" in self.reference_metric:
                    scores = delta_scores["accuracy"]
                    if scores["prod_value"] <= self.min_sectorial_threshold:
                        report["severity"] = "WARNING"
                        report["reasons"].append(f"accuracy current value {scores['prod_value']} is below safety floor {self.min_sectorial_threshold}")
                    if scores["relative_drop_pct"] >= self.max_sectorial_relative_drop:
                        report["severity"] = "WARNING"
                        report["reasons"].append(f"accuracy dropped by {scores['relative_drop_pct']}% (threshold: {self.max_sectorial_relative_drop}%)")
        else:
            raise ValueError(f"The selected metric '{self.reference_metric}' is not present in the delta scores.")

        logger.info("Generate performance degradation alert reports.")

        return report

    def evaluate_all_batches(self, dir_prod_path: str | Path, output_report_path: str | Path | None = None) -> dict:
        """
        Evaluates the performance of the model on all production datasets in a specified directory and generates a report.
        
        Args:
            dir_prod_path (str | Path): The path to the directory containing the production datasets.
            output_report_path (str | Path | None): The path to save the output report.
                If None, the report will not be saved to a file.    
                
        Returns:
            dict: A dictionary containing the evaluation results for each production dataset.
        
        Raises:
            FileNotFoundError: If no CSV or metadata files are found in the specified directory.
        """
        validate_type(
            dir_prod_path=(dir_prod_path, (str, Path)),
            output_report_path=(output_report_path, (str, Path, type(None)))
        )

        # Search for the metadata file in the production directory
        metadata_file = [f for f in os.listdir(dir_prod_path) if f.startswith("metadata") and f.endswith(".json")]

        # Check if the metadata file exists and read the batch files accordingly
        if metadata_file:
            # Construct the full path to the metadata file
            metadata_path = Path(dir_prod_path) / metadata_file[0]
            with open(metadata_path, "r", encoding="utf-8") as f:
                batch_files = json.load(f)["path_batch_files"]
        # If the metadata file does not exist, search for CSV files in the production directory
        else:
            batch_files = sorted([f for f in os.listdir(dir_prod_path) if f.endswith(".csv")])
            if not batch_files:
                raise FileNotFoundError(
                    f"No CSV and metadata files found in: "
                    f"{Path(dir_prod_path).relative_to(PROJECT_ROOT)}"
                )

        # Initialize the report dictionary to store evaluation results for each batch file.
        report = {}
        for batch_file in batch_files:
            single_res = self.evaluate_single_batches(Path(dir_prod_path) / batch_file)
            report[batch_file] = single_res[batch_file]

        # Save the full report to the specified output path if provided.
        if output_report_path is not None:
            with open(output_report_path, 'w', encoding='utf-8') as file:
                json.dump(report, file, indent=4)
            logger.info(
                f"The full batchs report has been successfully saved to: "
                f"{Path(output_report_path).relative_to(PROJECT_ROOT)}"
            )

        return report

    def evaluate_single_batches(self, path_dataset: str | Path, output_report_path: str | Path | None = None) -> dict:
        """
        
        Evaluates the performance of the model on a single production dataset and generates a report.
        
        Args:
            path_dataset (str | Path): The path to the production dataset.
            output_report_path (str | Path | None): The path to save the output report.
                If None, the report will not be saved to a file.

        Returns:
            dict: A dictionary containing the evaluation results for the production dataset.
        
        Raises:
            ValueError: If the production dataset does not contain the 'class' column or if the number of unique 
                classes in y_true does not match the number of classes expected by the label encoder.
        """
        validate_type(
            path_dataset=(path_dataset, (str, Path)),
            output_report_path=(output_report_path, (str, Path, type(None)))
        )

        # Read the dataset from the specified path.
        df = pd.read_csv(path_dataset)
        # Extract the file name from the dataset path.
        file_name = os.path.basename(path_dataset)

        # Check if the 'class' column is present in the dataset. If not, return an empty report.
        if "class" not in df.columns:
            return {}

        # Separate the features and the target variable.
        y_true = df["class"]
        X = df.drop(columns=["class"])

        # Initialize the report and delta_scores dictionaries to store evaluation results.
        report = {}
        delta_scores = {}

        # Evaluate the dataset, perform delta analysis and update the report accordingly.
        batch_scores = self._dataset_scores_evaluation(X, y_true)
        delta_scores[f"{file_name}"] = self._delta_analysis(batch_scores)
        report[f"{file_name}"] = {
            "alarm": self._threshold_and_alarm_engine(delta_scores[f"{file_name}"]),
            "metrics": batch_scores,
            "delta_vs_baseline": delta_scores[f"{file_name}"],
        }

        # Save the report to the specified output path if provided.
        if output_report_path is not None:
            with open(output_report_path, 'w', encoding='utf-8') as file:
                json.dump(report, file, indent=4)
            logger.info(
                f"The report has been successfully saved to: "
                f"{Path(output_report_path).relative_to(PROJECT_ROOT)}"
            )

        return report