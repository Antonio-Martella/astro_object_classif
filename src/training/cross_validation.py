from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from configs.random_seed_loader import load_random_seed_config
from configs.schemas import PreprocessingConfig
from configs.schemas_loader import load_preprocessing_config
from src.training.pipeline import build_training_pipeline
from src.utils.metrics import evaluate_classification_metrics
from src.utils.validate_type import validate_type


def run_cross_validation(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    resampling_strategy: str = "class_weight",
    scaler_strategy: str = "standard",
    custom_params: dict[str, Any] | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict[str, float]:
    """
    Performs robust cross-validation (Stratified Group K-Fold) to evaluate a model's performance.

    For each fold, the function dynamically builds an entire MLOps pipeline (ImbPipeline)
    which includes:
    * Data preprocessing.
    * Possible class rebalancing ('class_weight', 'smote', and 'undersampling')
    configurable in the configs/... file.
    * Model training.
    Validation is structured to prevent data leakage by ensuring that records
    belonging to the same group (e.g., same 'field_ID') are not split between the training and validation sets.

    **Args**:
    * *model_name* (str): The textual identifier of the model to be instantiated via ModelFactory
                          (e.g., 'xgboost', 'random_forest', 'dense_nn', etc.).
    * *X* (pd.DataFrame): The dataset containing the training features.
    * *y* (pd.Series): The target variable (labels).
    * *groups* (pd.Series): The feature used to group the data and avoid spatial/temporal leakage.
    * *n_splits* (int, optional): The number of folds into which to split the dataset. Defaults to 5.
    * *resampling_strategy* (str): Specifies the resemplifying strategy to apply to the training dataset.
                                   The default is 'class_weight', which implies 'balanced' for all models.
    * *scaler_strategy* (str): Specifies the scaling strategy to apply to dataset.
    * *custom_params* (dict, optional): Dictionary of custom hyperparameters (e.g., injected by Optuna).
                                        If None, the model will use the default parameters from the config file.
    * *preprocessing_config* (PreprocessingConfig, None): Preprocess configuration, necessary for the split train/test.

    **Returns**:
    * *dict*: A dictionary containing the arithmetic mean of the evaluation metrics
    (e.g., log_loss, accuracy, f1, precision, recall, roc-auc) computed over all folds.
    """
    validate_type(
        model_name=(model_name, str),
        X=(X, pd.DataFrame),
        y=(y, pd.Series),
        groups=(groups, pd.Series),
        n_splits=(n_splits, int),
        resampling_strategy=(resampling_strategy, str),
        scaler_strategy=(scaler_strategy, str),
        custom_params=(custom_params, (dict, type(None))),
        preprocessing_config=(preprocessing_config, (PreprocessingConfig, type(None))),
    )

    if X.empty or y.empty:
        raise ValueError("Please note that the datasets passed to cross-validation are empty!")

    if len(X) != len(y):
        raise ValueError("Please note that the lengths of the datasets (X and y) in the cross validation do not match!")

    if y.nunique() < 2:
        raise ValueError("y must contain at least 2 unique classes for stratified cross-validation.")

    if groups.empty:
        raise ValueError("Group passed to empty cross-validation!")

    if len(groups) != len(X):
        raise ValueError("The length of 'groups' series does not match the dataset X length!")

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    if groups.nunique() < n_splits:
        raise ValueError(f"Number of unique groups ({groups.nunique()}) must be >= n_splits ({n_splits}).")

    if preprocessing_config is None:
        preprocessing_config = load_preprocessing_config()

    try:
        random_seed_config = load_random_seed_config()
    except Exception as e:
        raise ValueError(f"Attention: Could not load random_seeds to handle reproducibility. Error {e}.") from e

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed_config.random_seed_sgkf)
    except Exception as e:
        raise RuntimeError(f"Attention: Stratified group k fold failed. Error {e}.") from e

    all_folds_metrics = []

    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        try:
            pipeline = build_training_pipeline(
                preprocessing_config, model_name, scaler_strategy, resampling_strategy, custom_params
            )
            pipeline.fit(X_fold_train, y_fold_train)
        except Exception as e:
            raise RuntimeError(f"Attention: the cross-validation PIPELINE failed on fold. Error {e}.") from e

        try:
            fold_scores = evaluate_classification_metrics(pipeline, X_fold_val, y_fold_val)
        except Exception as e:
            raise RuntimeError(f"Attention: the cross-validation EVALUATION failed on fold. Error {e}.") from e

        all_folds_metrics.append(fold_scores)

    df_metrics = pd.DataFrame(all_folds_metrics)

    return df_metrics.mean().to_dict()
