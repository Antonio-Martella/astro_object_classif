from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier

from src.models import BaseModel


class LogRegModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SGDModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = SGDClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
