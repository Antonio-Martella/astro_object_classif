import hashlib
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.models.signature import infer_signature

from configs.paths import DataPathConfig
from configs.schemas_loader import load_preprocessing_config
from configs.tuning_config import TuningConfigLoader
from src.optimization.core import optimize_model
from src.training.pipeline import load_training_data, train_and_evaluate_model
from src.utils.figures import (
    confusion_matrix_imag,
    corr_matrix,
    log_feature_importance,
    plot_target_distribution,
)
from src.utils.logger import setup_logger

# Import local modules
from src.utils.validate_type import validate_type

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def _optimize_step(
    tuning_config: TuningConfigLoader,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    path_config: DataPathConfig,
    trials_log_file: Path,
    save: bool = True,
) -> tuple[str, str, str, dict]:
    """
    Optimize your model using Optuna and return the best model, resampling and scaling strategies, and hyperparameters.

    Returns:
        tuple[str, str, str, dict]: best_model_name, best_resampling_strategy, best_scaler_strategy,
                                    clean_best_hyperparams
    """

    # Validation of input elements
    validate_type(
        tuning_config=(tuning_config, TuningConfigLoader),
        X=(X, pd.DataFrame),
        y=(y, pd.Series),
        groups=(groups, pd.Series),
        path_config=(path_config, DataPathConfig),
        trials_log_file=(trials_log_file, Path),
        save=(save, bool),
    )

    if len(X) != len(y):
        raise ValueError("Incompatibility between number of data points X and y!")
    if len(groups) != len(y):
        raise ValueError("Incompatibility between number of data points y and groups!")
    # ------------------------------

    # Start the search for the best model and best hyperparameters
    with mlflow.start_run(run_name="Optuna_Search"):
        best_hyperparameters = optimize_model(
            tuning_config=tuning_config, X=X, y=y, groups=groups, trials_log_file=trials_log_file
        )

    best_resampling_strategy = best_hyperparameters.pop("resampling_strategy")
    best_scaler_strategy = best_hyperparameters.pop("scaler_strategy")
    best_model_name = best_hyperparameters["model_name"]

    # Clean the best hyperparameters by removing any prefixes related to the model name
    clean_best_hyperparams = {}
    prefix = f"{best_model_name}_"

    for key, value in best_hyperparameters.items():
        if key.startswith(prefix):
            clean_key = key.replace(prefix, "", 1)
            clean_best_hyperparams[clean_key] = value
        else:
            clean_best_hyperparams[key] = value

    clean_best_hyperparams.pop("model_name")

    # Save the best hyperparameters locally
    if save:
        os.makedirs(Path(path_config.best_model_hyperparameters).parent, exist_ok=True)
        with open(path_config.best_model_hyperparameters, "w") as f:
            json.dump(clean_best_hyperparams, f, indent=4, ensure_ascii=False)

    # Print the best hyperparameters and the best resampling and scaler strategies
    logger.info(
        "3. Best Model %s with the following hyperparameters and strategies:",
        best_model_name.upper(),
    )
    logger.info("   Hyperparameters:")
    for k, v in clean_best_hyperparams.items():
        logger.info("       %s = %s", k, v)
    logger.info("   Strategies:")
    logger.info("       Resampling = %s", best_resampling_strategy)
    logger.info("       Scaler = %s", best_scaler_strategy)

    return best_model_name, best_resampling_strategy, best_scaler_strategy, clean_best_hyperparams


def _train_final_model_step(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_test_encoded: pd.Series,
    hyperparameters: dict,
    resampling_strategy: str,
    scaler_strategy: str,
    path_config: DataPathConfig,
    save: bool = True,
) -> tuple[ImbPipeline, dict]:
    """
    Train the final model with the best hyperparameters and strategies,
    and return the trained pipeline and the test scores.

    Returns:
        tuple[ImbPipeline, dict]: trained_pipeline, test_scores
    """

    # Validation of input elements
    validate_type(
        model_name=(model_name, str),
        X_train=(X_train, pd.DataFrame),
        X_test=(X_test, pd.DataFrame),
        y_train_encoded=(y_train_encoded, pd.Series),
        y_test_encoded=(y_test_encoded, pd.Series),
        hyperparameters=(hyperparameters, dict),
        resampling_strategy=(resampling_strategy, str),
        scaler_strategy=(scaler_strategy, str),
        path_config=(path_config, DataPathConfig),
        save=(save, bool),
    )

    if len(X_train) != len(y_train_encoded):
        raise ValueError("Incompatibility between number of training data points X_train and y_train_encoded!")
    if len(X_test) != len(y_test_encoded):
        raise ValueError("Incompatibility between number of test data points X_test and y_test_encoded!")
    # ------------------------------

    # Train the final model with the best hyperparameters and strategies
    pipeline, test_scores = train_and_evaluate_model(
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train_encoded=y_train_encoded,
        y_test_encoded=y_test_encoded,
        preprocess_config=load_preprocessing_config(),
        custom_params=hyperparameters,
        resampling_strategy=resampling_strategy,
        scaler_strategy=scaler_strategy,
    )

    test_scores_copy = test_scores.copy()
    test_scores_copy["model_name"] = model_name

    # Save the best model and its metrics locally if specified
    if save:
        os.makedirs(Path(path_config.best_model_metrics).parent, exist_ok=True)
        with open(path_config.best_model_metrics, "w") as f:
            json.dump(test_scores_copy, f, indent=4, ensure_ascii=False)

        # Salvataggio Locale Modello
        os.makedirs(Path(path_config.pipeline_best_model).parent, exist_ok=True)
        joblib.dump(pipeline, path_config.pipeline_best_model)

    logger.info("   Scores for the best model %s with the best hyperparameters:", model_name.upper())
    for k, v in test_scores.items():
        logger.info("       %s = %s", k, round(v, 4))

    return pipeline, test_scores


def _generate_reports_step(
    model_pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    path_config: DataPathConfig,
) -> None:
    """
    Generate analytical reports and figures for the best model, including confusion matrix,
    target distribution, feature importance, and correlation matrix.
    """

    # Validation of input elements
    validate_type(
        model_pipeline=(model_pipeline, ImbPipeline),
        X_train=(X_train, pd.DataFrame),
        X_test=(X_test, pd.DataFrame),
        y_train=(y_train, pd.Series),
        y_test=(y_test, pd.Series),
        path_config=(path_config, DataPathConfig),
    )

    if len(X_train) != len(y_train):
        raise ValueError("Attention inconsistency between X_train and y_train!")
    elif len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Attention length of X_train or y_train is zero!")
    if len(X_test) != len(y_test):
        raise ValueError("Attention inconsistency between X_test and y_test!")
    elif len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Attention length of X_test or y_test is zero!")
    # -----------------------------

    # Generate analytical reports and figures for the best model
    os.makedirs(path_config.best_model, exist_ok=True)
    test_predictions = model_pipeline.predict(X_test)

    confusion_matrix_imag(
        y_true=y_test,
        y_pred=pd.Series(test_predictions),
        save_path=path_config.best_model / "confusion_matrix.png",
    )
    plot_target_distribution(
        y=y_train,
        save_path=path_config.best_model / "target_train_distribution.png",
        dataset_name="train",
    )
    plot_target_distribution(
        y=y_test,
        save_path=path_config.best_model / "target_test_distribution.png",
        dataset_name="test",
    )
    log_feature_importance(
        pipeline=model_pipeline,
        X=X_test,
        y=y_test,
        feature_names=X_train.columns,
        save_path=path_config.best_model / "feature_importance.png",
    )
    corr_matrix(X=X_train, save_path=path_config.best_model / "corr_matrix.png")


def file_md5(path: Path, chunk_size: int = 8192) -> str:
    """
    Compute the MD5 hash of a file.
    """
    validate_type(path=(path, Path), chunk_size=(chunk_size, int))
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def _log_experiment_result(
    model_pipeline: ImbPipeline,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    best_hyperparams: dict,
    resampling_strategy: str,
    scaler_strategy: str,
    test_scores: dict,
    experiment_metadata: dict,
    path_config: DataPathConfig,
    run_log_file: Path,
    trials_log_file: Path,
) -> None:
    """
    Log the results of the best model experiment to MLflow, including parameters,
    metrics, artifacts, and the trained model pipeline.
    """

    # Validation of input elements
    validate_type(
        model_pipeline=(model_pipeline, ImbPipeline),
        model_name=(model_name, str),
        X_train=(X_train, pd.DataFrame),
        X_test=(X_test, pd.DataFrame),
        best_hyperparams=(best_hyperparams, dict),
        resampling_strategy=(resampling_strategy, str),
        scaler_strategy=(scaler_strategy, str),
        test_scores=(test_scores, dict),
        experiment_metadata=(experiment_metadata, dict),
        path_config=(path_config, DataPathConfig),
        run_log_file=(run_log_file, Path),
        trials_log_file=(trials_log_file, Path),
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Attention length of X_train or X_test is zero!")
    # -----------------------------

    try:
        path_dataset = path_config.processed_data_path.relative_to(Path(__file__).parent.parent.parent)
    except ValueError:
        path_dataset = path_config.processed_data_path

    # Log the results of the best model experiment to MLflow
    with mlflow.start_run(run_name=f"Optuna_Best_Model: {model_name.upper()}"):
        # Log the experiment metadata and dataset information
        mlflow.set_tags(experiment_metadata)
        mlflow.set_tags(
            {
                "dataset_path": path_dataset,
                "dataset_md5": file_md5(path_dataset),
                "dataset_size_bytes": os.path.getsize(path_dataset),
                "dataset_row_count": len(X_train) + len(X_test),
            }
        )
        # Log the parameters, metrics, and artifacts of the best model experiment
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("strategy_resempling", resampling_strategy)
        mlflow.log_param("strategy_scaler", scaler_strategy)
        mlflow.log_params(best_hyperparams)
        mlflow.log_metrics(test_scores)

        # Log the generated figures and configuration files as artifacts in MLflow
        for fig_name in [
            "confusion_matrix.png",
            "target_train_distribution.png",
            "target_test_distribution.png",
            "feature_importance.png",
        ]:
            fig_path = path_config.best_model / fig_name
            mlflow.log_artifact(local_path=str(fig_path), artifact_path="figures")

        mlflow.log_artifact(local_path=str(path_config.optuna_config), artifact_path="configs")
        mlflow.log_artifact(local_path=str(path_config.params_config), artifact_path="configs")
        mlflow.log_artifacts(local_dir=str(path_config.random_forest_search_space.parent), artifact_path="search_space")
        mlflow.log_artifact(local_path=str(path_config.target_le), artifact_path="encoder")
        mlflow.log_artifact(local_path=str(path_config.requirements_file), artifact_path="environment")

        # Log the run and trials log files as artifacts in MLflow if they exist
        if run_log_file and run_log_file.exists():
            mlflow.log_artifact(local_path=str(run_log_file), artifact_path="logs")
        if trials_log_file and trials_log_file.exists():
            mlflow.log_artifact(local_path=str(trials_log_file), artifact_path="logs")

        # Log the trained model pipeline to MLflow with input signature and example
        signature = infer_signature(X_test, model_pipeline.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="best_model_pipeline",
            signature=signature,
            registered_model_name="Classification_Astro_Model",
            input_example=X_train.iloc[:3],
        )


def run_optimization_pipeline() -> None:
    """
    Run the entire optimization pipeline, including data loading, model optimization,
    final training, report generation, and logging to MLflow.
    """
    # Load configurations and set up MLflow tracking
    path_config = DataPathConfig()
    tuning_config = TuningConfigLoader()

    # Set the MLflow tracking URI and experiment name
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{path_config.mlflow_db_path}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(tuning_config.optuna_config["name_experiment"])

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_file = path_config.opt_logs_dir / f"run_{run_timestamp}.log"
    trials_log_file = path_config.opt_logs_dir / f"trials_{run_timestamp}.log"

    setup_logger(run_log_file_path=run_log_file)

    logger.info("=" * 60)
    logger.info("Starting Optimization Pipeline")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # 1. LOADING DATA AND SPATIAL SPLIT
    # ---------------------------------------------------------
    logger.info("1. Loading data and performing spatial split...")
    try:
        X_train, X_test, y_train_encoded, y_test_encoded, groups_train, _, _ = load_training_data()
    except Exception as e:
        logger.error("Error during data split: %s", e, exc_info=True)
        raise

    logger.info("2. Starting Optimization with Optuna...")
    # ---------------------------------------------------------
    # 2. OPTUNA OPTIMIZATION
    # ---------------------------------------------------------
    try:
        best_model_name, best_resampling_strategy, best_scaler_strategy, clean_best_hyperparams = _optimize_step(
            tuning_config, X_train, y_train_encoded, groups_train, path_config, trials_log_file
        )
    except Exception as e:
        logger.error("Error during the search for the best model and hyperparameters: %s", e, exc_info=True)
        raise

    # ---------------------------------------------------------
    # 4. TRAINING BEST MODEL
    # ---------------------------------------------------------
    logger.info("4. STARTING TRAINING %s WITH THE BEST HYPERPARAMETERS", best_model_name.upper())

    print(X_train.isna().sum())
    print(X_train.isnull().values.any())

    try:
        pipeline_best_model, score_best_model = _train_final_model_step(
            model_name=best_model_name,
            X_train=X_train,
            X_test=X_test,
            y_train_encoded=y_train_encoded,
            y_test_encoded=y_test_encoded,
            hyperparameters=clean_best_hyperparams,
            resampling_strategy=best_resampling_strategy,
            scaler_strategy=best_scaler_strategy,
            path_config=path_config,
        )
    except Exception as e:
        logger.error("Error during the training of the best model: %s", e, exc_info=True)
        raise

    # ---------------------------------------------------------
    # 5. REPORTING
    # ---------------------------------------------------------
    logger.info("5. Generating Analytical Reports")
    try:
        _generate_reports_step(pipeline_best_model, X_train, X_test, y_train_encoded, y_test_encoded, path_config)
    except Exception as e:
        logger.error("Error during the generation of reports and figures: %s", e, exc_info=True)

    # ---------------------------------------------------------
    # 6. LOGGING TO MLFLOW
    # ---------------------------------------------------------
    logger.info("6. Saving results to MLflow")

    try:
        experiment_metadata = {
            "cv_folds": str(tuning_config.optuna_config["n_folds_cv"]),
            "optimized_metric": tuning_config.optuna_config["metric_to_optimize"],
            "optuna_trials": str(tuning_config.optuna_config["n_trials"]),
            "models_set": str(tuning_config.optuna_config["model_name"]),
        }

        _log_experiment_result(
            model_pipeline=pipeline_best_model,
            model_name=best_model_name,
            X_train=X_train,
            X_test=X_test,
            best_hyperparams=clean_best_hyperparams,
            resampling_strategy=best_resampling_strategy,
            scaler_strategy=best_scaler_strategy,
            test_scores=score_best_model,
            experiment_metadata=experiment_metadata,
            path_config=path_config,
            run_log_file=run_log_file,
            trials_log_file=trials_log_file,
        )
    except Exception as e:
        logger.error("Error during saving to MLflow: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    run_optimization_pipeline()
