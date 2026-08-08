from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier

from configs.paths import DataPathConfig
from configs.tuning_config import TuningConfigLoader
from src.optimization.search import (
    _generate_reports_step,
    _log_experiment_result,
    _optimize_step,
    _train_final_model_step,
    file_md5,
    run_optimization_pipeline,
)


@pytest.fixture
def dummy_data():
    """
    Genera dataset sintetico per i test.
    """
    X = pd.DataFrame(np.random.rand(20, 6), columns=["f1", "f2", "f3", "f4", "f5", "f6"])
    y = pd.Series(np.random.choice([0, 1, 2], size=20))
    groups = pd.Series(np.repeat([0, 1, 2, 3], 5))

    return X, y, groups


@pytest.fixture
def dummy_pipeline():
    """
    Crea un pipeline fittizia imblearn.
    """
    return ImbPipeline([("classifier", DummyClassifier(strategy="most_frequent"))])


@pytest.fixture
def mock_path_config(tmp_path):
    """
    Mock della tua DataPathConfig reale basato sulla cartella temporanea di pytest (tmp_path).
    In questo modo non viene toccato il tuo file system reale.
    """
    config = MagicMock(spec=DataPathConfig)

    # Mappatura esatta sui path della tua classe DataPathConfig
    config.raw_data_path = tmp_path / "data" / "raw" / "Star_Classification.csv"
    config.split_training_path = tmp_path / "data" / "interim" / "training_dataset.csv"
    config.split_production_path = tmp_path / "data" / "interim" / "holdout_dataset.csv"
    config.processed_data_path = tmp_path / "data" / "processed" / "processed_data.csv"

    config.cleaner_pipeline = tmp_path / "models" / "cleaner_pipeline.pkl"
    config.target_le = tmp_path / "models" / "target_encoder.pkl"
    config.pipeline_best_model = tmp_path / "models" / "best_model" / "pipeline_best_model.pkl"

    config.optuna_config = tmp_path / "configs" / "optuna.yaml"
    config.params_config = tmp_path / "configs" / "params.yaml"
    config.randomseed_config = tmp_path / "configs" / "random_seed.yaml"

    # Search spaces
    config.random_forest_search_space = tmp_path / "configs" / "models" / "random_forest.yaml"

    # Reports & Best Model Outputs
    config.best_model = tmp_path / "reports" / "best_model"
    config.best_model_metrics = config.best_model / "best_model_metrics.json"
    config.best_model_hyperparameters = config.best_model / "best_model_iperparameters.json"

    # MLflow & Logs
    config.mlflow_db_path = tmp_path / "mlflow.db"
    config.opt_logs_dir = tmp_path / "logs"

    # CREAZIONE FILE E CARTELLE FITTIZIE (per evitare FileNotFoundError durante le letture)
    config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    config.processed_data_path.write_text("f1,f2,y\n1,2,0\n")

    config.optuna_config.parent.mkdir(parents=True, exist_ok=True)
    config.optuna_config.write_text("name: test")
    config.params_config.write_text("params: test")

    config.random_forest_search_space.parent.mkdir(parents=True, exist_ok=True)
    config.random_forest_search_space.write_text("space: test")

    config.target_le.parent.mkdir(parents=True, exist_ok=True)
    config.target_le.write_text("encoder")

    config.best_model.mkdir(parents=True, exist_ok=True)
    config.opt_logs_dir.mkdir(parents=True, exist_ok=True)

    return config


@pytest.fixture
def mock_tuning_config():
    """
    Mock della vera classe TuningConfigLoader che riflette sia
    self.optuna_config sia il metodo get_search_space().
    """
    config = MagicMock(spec=TuningConfigLoader)

    config.optuna_config = {
        "name_experiment": "Astro_Object_Classification",
        "model_name": ["random_forest", "xgboost"],
        "n_trials": 5,
        "n_startup_trials": 2,
        "metric_to_optimize": "f1",
        "multivariate": True,
        "direction": "maximize",
        "n_folds_cv": 3,
        "resampling_strategy": {
            "type": "categorical",
            "choices": ["class_weight", "smote", "undersampling"],
        },
        "scaler_strategy": {"type": "categorical", "choices": ["standard", "robust", "minmax"]},
    }

    def mock_get_search_space(model_name: str) -> dict:
        return {
            f"{model_name}_n_estimators": {"type": "int", "low": 10, "high": 100},
            f"{model_name}_max_depth": {"type": "int", "low": 2, "high": 10},
            "random_state": {"type": "fixed", "value": 42},
            "resampling_strategy": config.optuna_config["resampling_strategy"],
            "scaler": config.optuna_config["scaler"],
        }

    config.get_search_space.side_effect = mock_get_search_space

    return config


def test_file_md5(tmp_path):
    """Verifica il calcolo dell'hash MD5."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    expected_md5 = "5eb63bbbe01eeed093cb22bb8f5acdc3"
    assert file_md5(test_file) == expected_md5


@patch("src.optimization.search.optimize_model")
def test_optimize_step(mock_opt_model, dummy_data, mock_path_config, mock_tuning_config):
    """Testa lo step di ottimizzazione Optuna."""
    X, y, groups = dummy_data

    mock_opt_model.return_value = {
        "model_name": "random_forest",
        "random_forest_n_estimators": 100,
        "random_forest_max_depth": 5,
        "resampling_strategy": "smote",
        "scaler_strategy": "standard",
    }

    trials_log_file = mock_path_config.opt_logs_dir / "trials.log"

    model_name, resamp, scaler, clean_params = _optimize_step(
        tuning_config=mock_tuning_config,
        X=X,
        y=y,
        groups=groups,
        path_config=mock_path_config,
        trials_log_file=trials_log_file,
        save=True,
    )

    assert model_name == "random_forest"
    assert resamp == "smote"
    assert scaler == "standard"
    assert "n_estimators" in clean_params
    assert "random_forest_n_estimators" not in clean_params
    assert mock_path_config.best_model_hyperparameters.exists()


@patch("src.optimization.search.train_and_evaluate_model")
def test_train_final_model_step(mock_train_eval, dummy_data, mock_path_config, dummy_pipeline):
    """Testa lo step di training finale."""
    X, y, _ = dummy_data
    mock_train_eval.return_value = (dummy_pipeline, {"f1": 0.95, "accuracy": 0.96})

    pipeline, scores = _train_final_model_step(
        model_name="random_forest",
        X_train=X,
        X_test=X,
        y_train_encoded=y,
        y_test_encoded=y,
        hyperparameters={"n_estimators": 100},
        resampling_strategy="smote",
        scaler_strategy="standard",
        path_config=mock_path_config,
        save=True,
    )

    assert scores["f1"] == 0.95
    assert mock_path_config.best_model_metrics.exists()
    assert mock_path_config.pipeline_best_model.exists()


@patch("src.optimization.search.confusion_matrix_imag")
@patch("src.optimization.search.plot_target_distribution")
@patch("src.optimization.search.log_feature_importance")
@patch("src.optimization.search.corr_matrix")
def test_generate_reports_step(mock_corr, mock_fi, mock_target, mock_cm, dummy_data, mock_path_config, dummy_pipeline):
    """Testa la generazione dei grafici."""
    X, y, _ = dummy_data
    dummy_pipeline.predict = MagicMock(return_value=y.values)

    _generate_reports_step(
        model_pipeline=dummy_pipeline,
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        path_config=mock_path_config,
    )

    assert mock_cm.called
    assert mock_target.call_count == 2
    assert mock_fi.called
    assert mock_corr.called


@patch("src.optimization.search.mlflow")
def test_log_experiment_result(mock_mlflow, dummy_data, mock_path_config, dummy_pipeline):
    """Testa la registrazione dei risultati su MLflow."""
    X, y, _ = dummy_data
    dummy_pipeline.predict = MagicMock(return_value=y.values)

    run_log = mock_path_config.opt_logs_dir / "run.log"
    trials_log = mock_path_config.opt_logs_dir / "trials.log"
    run_log.write_text("run log content")
    trials_log.write_text("trials log content")

    _log_experiment_result(
        model_pipeline=dummy_pipeline,
        model_name="random_forest",
        X_train=X,
        X_test=X,
        best_hyperparams={"n_estimators": 100},
        resampling_strategy="smote",
        scaler_strategy="standard",
        test_scores={"f1": 0.95},
        experiment_metadata={"cv": "3"},
        path_config=mock_path_config,
        run_log_file=run_log,
        trials_log_file=trials_log,
    )

    assert mock_mlflow.start_run.called
    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metrics.called
    assert mock_mlflow.log_artifact.called
    assert mock_mlflow.sklearn.log_model.called


@patch("src.optimization.search.load_training_data")
@patch("src.optimization.search._optimize_step")
@patch("src.optimization.search._train_final_model_step")
@patch("src.optimization.search._generate_reports_step")
@patch("src.optimization.search._log_experiment_result")
def test_run_optimization_pipeline_flow(
    mock_log_exp, mock_reports, mock_train, mock_opt, mock_load, dummy_data, dummy_pipeline
):
    """Testa l'intero flusso di esecuzione (run_optimization_pipeline)."""
    X, y, groups = dummy_data

    mock_load.return_value = (X, X, y, y, groups, None, None)
    mock_opt.return_value = ("random_forest", "smote", "standard", {"n_estimators": 100})
    mock_train.return_value = (dummy_pipeline, {"f1": 0.95})

    run_optimization_pipeline()

    # Verifica che tutti gli step vengano chiamati nell'ordine corretto
    assert mock_load.called
    assert mock_opt.called
    assert mock_train.called
    assert mock_reports.called
    assert mock_log_exp.called
