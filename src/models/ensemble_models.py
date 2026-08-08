import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# internal moduls
from src.models import BaseModel


class RandomForestModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X=X, y=y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
