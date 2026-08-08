import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from configs import CleanPreprocessingConfig

logger = logging.getLogger(__name__)


class AstroFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for astronomical feature engineering.

    This transformer generates astrophysical color indices from SDSS
    photometric magnitudes ['u-g', 'g-r', 'r-i', 'i-z'].

    Color indices are physically meaningful features representing flux
    differences between adjacent photometric bands. These quantities are
    widely used in astronomy for discriminating celestial objects such as:

        - GALAXY
        - STAR
        - QSO (Quasar)

    The transformer requires the following input columns ['u', 'g', 'r', 'i', 'z'].

    Notes
    -----
    This transformer must be executed BEFORE any feature selection or
    column dropping stage removing the required photometric bands.

    Compatible with:
        - sklearn Pipeline
        - ColumnTransformer
        - GridSearchCV

    Methods
    -------
    - **fit(X, y=None)**
        No-op fit method required for sklearn compatibility.
        Returns self without modifying the transformer state.

    - **transform(X, y=None)**
        Returns a transformed copy of the input dataframe with the
        following engineered color-index features added ['u-g', 'g-r', 'r-i', 'i-z']

    Parameters
    ----------
    **None**

    Returns
    -------
    **pd.DataFrame**
        Input dataframe enriched with four additional color-index features.
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

        required_cols = list(set([col for pair in self.clean_prepr_config.new_features for col in pair]))
        required_cols.append(self.clean_prepr_config.new_features[-1][-1])

        if not all(x in X_copy.columns for x in required_cols):
            raise ValueError(
                f"Missing columns! Make sure you're running AstroFeatureEngineer."
                f"BEFORE FeaturesDrop in the Pipeline. Required columns: {self.clean_prepr_config.new_features}"
            )

        for _, col in enumerate(self.clean_prepr_config.new_features):
            X_copy[f"{col[0]}-{col[1]}"] = X_copy[col[0]] - X_copy[col[1]]

        logger.info(
            "Feature engineering completed successfully. New color-index features ['u-g', 'g-r', 'r-i', 'i-z'] "
            "have been added to the dataset."
        )
        return X_copy
