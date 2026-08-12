import logging

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from configs.paths import PROJECT_ROOT, DataPathConfig
from configs.schemas import CleanPreprocessingConfig, PreprocessingConfig
from configs.schemas_loader import load_cleaning_preprocessing_config
from src.features.build_features import AstroFeatureEngineer

logger = logging.getLogger(__name__)


class FeaturesDrop(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for dropping non-informative or unwanted features
    from the input dataset.

    This transformer removes columns specified in the configs/params.yaml file (preprocessing)
    and validates their existence before applying the transformation.

    Parameters
    ----------
    - **features_config** : **PreprocessingConfig**
        Configuration object containing the list of columns to drop through
        the attribute `columns_to_drop`.

    Methods
    -------
    - **fit(X, y=None)**
        No-op fit method for sklearn compatibility.

    - **transform(X, y=None)**
        Returns a transformed copy of the dataframe with specified columns removed.

    Raises
    ------
    **ValueError**
        If one or more columns defined in `columns_to_drop`
        are not present in the input dataframe.
    """

    def __init__(self, clean_prepr_config: CleanPreprocessingConfig):
        self.clean_prepr_config = clean_prepr_config

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_copy = X.copy()

        if not all(x in X_copy.columns for x in self.clean_prepr_config.columns_to_drop):
            raise ValueError(
                "Please note that not all columns defined in 'columns_to_drop' (configs/params.yaml)",
                "are present in the dataset!",
            )

        X_copy = X_copy.drop(labels=self.clean_prepr_config.columns_to_drop, axis=1)

        logger.info("Dataset features dropped successfully!")

        return X_copy


class AnomalyHandler(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for handling anomalous values in datasets.

    This transformer replaces predefined anomaly or sentinel values
    (commonly used in scientific datasets to encode missing or invalid measurements)
    with NaN values, enabling downstream compatibility with sklearn estimators
    and imputation strategies.

    Typical use case:
        Astronomical datasets often encode missing or corrupted measurements
        using placeholder values such as -9999. This transformer standardizes
        such values into NaN.

    Parameters
    ----------
    anomaly_value : int, default=-9999
        Sentinel value representing invalid or missing observations.

    Methods
    -------
    fit(X, y=None)
        No-op method for sklearn compatibility.
        Returns self without learning any parameters.

    transform(X, y=None)
        Returns a copy of X where all occurrences of `anomaly_value`
        are replaced with NaN.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with anomaly values replaced by NaN.
    """

    def __init__(self, clean_prepr_config: CleanPreprocessingConfig):
        self.clean_prepr_config = clean_prepr_config

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_copy = X.copy()

        for value in self.clean_prepr_config.anomaly_values:
            X_copy = X_copy.replace(value, np.nan)
            logger.info(f"Replaced anomaly value {value} with NaN.")

        return X_copy


class ProcessedDataSaver:
    """
    Utility class responsible for persisting processed machine learning datasets.

    This class combines transformed feature matrices and target labels into a
    single dataframe and saves the resulting processed dataset to disk.

    The saver is designed for ML pipelines where preprocessing steps are
    executed independently from persistence logic, promoting modularity and
    reproducibility.


    Parameters
    ----------
    * **path_config** : **DataPathConfig**
        Configuration object containing output paths for processed datasets.

    Methods
    -------
    * **save(X, y)**
        Saves the processed feature matrix and target labels into a single CSV file.

    Compatible with
    ---------------
    - sklearn preprocessing pipelines
    - MLflow tracking workflows
    - reproducible ML data pipelines
    """

    def __init__(self, path_config: DataPathConfig):
        self.path_config = path_config

    def save(self, df: pd.DataFrame) -> None:
        df_processed = df.copy()

        self.path_config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)

        df_processed.to_csv(self.path_config.processed_data_path, index=False)

        logger.info(
            f"Processed dataset successfully saved to: {self.path_config.processed_data_path.relative_to(PROJECT_ROOT)}"
        )
        logger.info(f"Final shape of the saved dataset: {df_processed.shape}")


def build_stateless_cleaning_pipeline(clean_prepr_config: CleanPreprocessingConfig) -> Pipeline:
    """
    Builds a full sklearn preprocessing pipeline for astronomical classification tasks.

    The pipeline is designed to prepare raw SDSS-like photometric data for
    machine learning models by applying a sequence of deterministic transformations:

    Pipeline stages
    ---------------
    1. Anomaly handling
        Replaces sentinel anomaly values (e.g. -9999) with NaN.

    2. Feature engineering
        Constructs astrophysical color indices:
            - u-g,
            - g-r,
            - r-i,
            - i-z

    3. Feature selection
        Removes non-informative or redundant features defined in config.

    Parameters
    ----------
    prep_config : PreprocessingConfig
        Configuration object controlling:
            - whether scaling is applied
            - columns selected for scaling
            - features to drop

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully composed preprocessing pipeline ready for fit/transform usage.
    """

    set_config(transform_output="pandas")

    pipeline = ImbPipeline(
        steps=[
            ("anomaly_handler", AnomalyHandler(clean_prepr_config=clean_prepr_config)),
            ("features_engineer", AstroFeatureEngineer(clean_prepr_config=clean_prepr_config)),
            ("dropper", FeaturesDrop(clean_prepr_config=clean_prepr_config)),
        ]
    )

    return pipeline


class ScalarFactory:
    _registry = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}

    @classmethod
    def get_scaler(cls, scaler_name: str):
        if scaler_name.lower() not in cls._registry.keys():
            raise ValueError(f"Scaler '{scaler_name}' non supportato! " f"Scegli tra: {list(cls._registry.keys())}")

        return cls._registry[scaler_name]


def build_stateful_ml_pipeline(prep_config: PreprocessingConfig, scaler_strategy: str) -> Pipeline:
    """
    Builds a full sklearn preprocessing pipeline for astronomical classification tasks.

    The pipeline is designed to prepare raw SDSS-like photometric data for
    machine learning models by applying a sequence of deterministic transformations:

    Pipeline stages
    ---------------

    1. Fill missing value imputation
        Fills NaN values using median strategy.

    2. Feature scaling (optional)
        Applies StandardScaler to selected numerical columns if enabled in config.

    Parameters
    ----------
    prep_config : PreprocessingConfig
        Configuration object controlling:
            - whether scaling is applied
            - columns selected for scaling
            - features to drop

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully composed preprocessing pipeline ready for fit/transform usage.
    """

    set_config(transform_output="pandas")

    # Normalizzo se in configs/params.yaml è passato come true
    scaler_step = ScalarFactory.get_scaler(scaler_name=scaler_strategy)

    # Costruisco la pipeline del preprocessing
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                ColumnTransformer(
                    transformers=[("num", scaler_step, prep_config.columns_to_scale)],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )

    return pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    path_config = DataPathConfig()

    db_path = path_config.mlflow_db_path
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Astro_Object_Classification")

    # carica i dati
    df = pd.read_csv(path_config.split_training_path)

    # creo il dataset delle features
    X = df.drop(columns=["class"])
    # creo il dataset del traget
    y = df["class"]

    # inizializzo le classi di configurazione
    prep_config = load_cleaning_preprocessing_config()

    # Creo la pipeline
    cleaning_pipeline = build_stateless_cleaning_pipeline(prep_config)

    # Transformo il dataset X
    with mlflow.start_run(run_name="pipeline_preprocessing"):
        # Fittiamo e trasformiamo i dati
        X_processed = cleaning_pipeline.fit_transform(X)

        # Tracciamo tutti i parametri della pipeline su mlflow
        mlflow.log_params(
            {
                "dropped_columns": prep_config.columns_to_drop,
                "remaining_columns": [X_processed.columns.tolist(), "class"],
            }
        )

        # Salvo la pipeline, necessaria per il test, in mlflow
        mlflow.sklearn.log_model(sk_model=cleaning_pipeline, name="cleaning_pipeline")

        # Salviamo anche una copia in locale
        models_path = path_config.cleaner_pipeline
        models_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cleaning_pipeline, models_path)

        logger.info(f"Copia locale della pipeline salvata in {models_path}.")

    saver = ProcessedDataSaver(path_config=path_config)
    saver.save(pd.concat([X_processed, y], axis=1))
