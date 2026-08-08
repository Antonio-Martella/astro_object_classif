import logging
import os
from pathlib import Path
from typing import Callable

import mlflow
import optuna
import pandas as pd
from optuna.trial import FrozenTrial

# Local imports
from configs.random_seed_loader import load_random_seed_config
from configs.tuning_config import TuningConfigLoader
from src.training.cv import run_cross_validation
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def apply_optuna_suggestions(trial, search_space: dict) -> dict:
    """
    Dynamically selects hyperparameters based on the search space defined in the
    YAML configuration files (configs/models/). The hyperparameters belong to four families:
    - int: Optuna samples integers from a defined range (specified by 'low' and 'high').
    - float: Optuna samples floats from a defined range, supporting logarithmic sampling ('log': True).
    - categorical: Optuna samples from a fixed list of choices ('choices').
    - fixed: Assigns a fixed value to the hyperparameter for the entire duration of the trials.

    Args:
        trial (optuna.trial.Trial): The current trial object provided by Optuna.
        search_space (dict): A dictionary containing the search rules loaded from the YAML.
    Returns:
        dict: A dictionary of hyperparameters dynamically selected for the current trial.
    Raises:
        ValueError: If an unsupported 'type' is detected in the search_space dictionary.
    """

    # Validation of input elements
    validate_type(trial=(trial, optuna.trial.Trial), search_space=(search_space, dict))

    if search_space is None:
        raise ValueError("Attention hyperparameter search space not provided!")
    # -----------------------------------------

    params = {}

    # Iterate through the search space and apply Optuna suggestions based on the defined types
    for param_name, config in search_space.items():
        p_type = config["type"]

        if p_type == "int":
            params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
        elif p_type == "float":
            params[param_name] = trial.suggest_float(
                param_name, config["low"], config["high"], log=config.get("log", False)
            )
        elif p_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config["choices"])
        elif p_type == "fixed":
            params[param_name] = config["value"]
        else:
            raise ValueError(f"Attention, 'type = {p_type}' defined is not valid!")

    return params


def make_mlflow_callback(
    tuning_config: TuningConfigLoader,
    trials_log_file: Path,
) -> Callable[[optuna.Study, FrozenTrial], None]:
    """
    Create a callback for Optuna that logs the results of each trial to MLflow and to a separate log file.
    """
    validate_type(tuning_config=(tuning_config, TuningConfigLoader), trials_log_file=(trials_log_file, Path))

    metric_name = tuning_config.optuna_config["metric_to_optimize"]

    os.makedirs(trials_log_file.parent, exist_ok=True)
    trial_logger = logging.getLogger("optuna_trials")
    trial_logger.setLevel(logging.INFO)
    trial_logger.propagate = False

    if trial_logger.hasHandlers():
        for h in trial_logger.handlers[:]:
            h.close()
            trial_logger.removeHandler(h)

    file_handler = logging.FileHandler(trials_log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    trial_logger.addHandler(file_handler)

    def mlflow_callback(study: optuna.Study, trial: FrozenTrial) -> None:
        trial_value = getattr(trial, "value", None)
        trial_state = getattr(trial, "state", None)
        state_name = getattr(trial_state, "name", "UNKNOWN")

        log_line = (
            f"Trial {trial.number:03d} | State: {state_name} | "
            f"Value ({metric_name}): {trial_value} | Params: {trial.params}"
        )
        trial_logger.info(log_line)

        is_nested = mlflow.active_run() is not None

        with mlflow.start_run(nested=is_nested, run_name=f"trial_{trial.number}"):
            mlflow.log_params(trial.params)
            if trial_value is not None:
                mlflow.log_metric(metric_name, trial_value)

    return mlflow_callback


def optimize_model(
    tuning_config: TuningConfigLoader,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    trials_log_file: Path,
) -> dict:
    """
    Orchestrates the Bayesian optimization of hyperparameters for a specific model using Optuna.

    The function retrieves the search space defined in the configuration and launches an Optuna study.
    For each trial, it generates a combination of hyperparameters, trains the model using nested cross-validation,
    and evaluates its performance. The process is parallelized to maximize efficiency.

    Args:
        - tuning_config (TuningConfigLoader): Configuration for model optimization,
          including search space and optimization settings.
        - X (pd.DataFrame): The feature dataset used for training and validation.
        - y (pd.Series): The target variable corresponding to the feature dataset.
        - groups (pd.Series): A series indicating the group for each sample, used for group-based cross-validation.
        - trials_log_file (Path): Path to the log file where trial details will be recorded

    Returns:
        - dict: A dictionary containing the best combination of hyperparameters found at the end of all trials.
    """

    # Validation of input elements
    validate_type(
        tuning_config=(tuning_config, TuningConfigLoader),
        X=(X, pd.DataFrame),
        y=(y, pd.Series),
        groups=(groups, pd.Series),
        trials_log_file=(trials_log_file, Path),
    )

    if tuning_config is None:
        raise TypeError("Attention configuration for optimization not provided or not valid!")

    if X.empty or y.empty:
        raise ValueError("Dataset provided for optimization is empty!")

    if len(X) != len(y):
        raise ValueError("Length of dataset X is different from that of target y!")

    if groups is None or groups.empty:
        raise ValueError("Group provided is empty or missing!")
    # -----------------------------------------

    # Define the objective function for Optuna, which will be called for each trial.
    # This function generates a set of hyperparameters, trains the model using cross-validation,
    # and returns the performance metric to be optimized.
    def objective(trial):
        # Generate a set of hyperparameters for the current trial based on the search space defined
        # in the configuration.
        current_model = trial.suggest_categorical("model_name", tuning_config.optuna_config["model_name"])
        search_space = tuning_config.get_search_space(model_name=current_model)
        raw_trial_params = apply_optuna_suggestions(trial, search_space)
        clean_params = {}
        prefix = f"{current_model}_"

        for key, value in raw_trial_params.items():
            if key.startswith(prefix):
                clean_key = key.replace(prefix, "", 1)
                clean_params[clean_key] = value
            else:
                clean_params[key] = value

        # Extract the resampling and scaler strategies from the hyperparameters, if they exist,
        # otherwise use default values.
        current_strategy = clean_params.pop("resampling_strategy", "none")
        current_scaler = clean_params.pop("scaler_strategy", "standard")

        # Run cross-validation with the current set of hyperparameters and strategies,
        # and retrieve the performance metric to be optimized.
        cv_results = run_cross_validation(
            model_name=current_model,
            X=X,
            y=y,
            groups=groups,
            n_splits=tuning_config.optuna_config["n_folds_cv"],
            resampling_strategy=current_strategy,
            scaler_strategy=current_scaler,
            custom_params=clean_params,
        )

        return cv_results[tuning_config.optuna_config["metric_to_optimize"]]

    # Set up the Optuna sampler with the specified number of startup trials, multivariate option,
    # and random seed for reproducibility.
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=tuning_config.optuna_config["n_startup_trials"],
        multivariate=tuning_config.optuna_config["multivariate"],
        warn_independent_sampling=False,
        seed=load_random_seed_config().random_seed_optuna,
    )

    logger.info(f"Number of trials {tuning_config.optuna_config['n_trials']}")
    # Set up the Optuna study with the specified optimization direction, study name, and sampler.
    study = optuna.create_study(
        direction=tuning_config.optuna_config["direction"], study_name="opt_astro", sampler=sampler
    )

    # Run the optimization process, which will execute the objective function for the specified number of trials.
    study.optimize(
        objective,
        n_trials=tuning_config.optuna_config["n_trials"],
        show_progress_bar=True,
        callbacks=[make_mlflow_callback(tuning_config, trials_log_file)],
    )

    # Capture the best hyperparameters found at the end of the study, which will then be injected into the final
    # training
    final_best_params = study.best_params.copy()

    # Retrieve the complete search space for the optimized model, so that FIXED parameters can be re-injected
    search_space = tuning_config.get_search_space(model_name=final_best_params["model_name"])

    # Re-inject all FIXED parameters from our search_space
    for param_name, config in search_space.items():
        if config["type"] == "fixed":
            val = config["value"]
            final_best_params[param_name] = None if str(val).lower() in ["none", "null"] else val

    return final_best_params
