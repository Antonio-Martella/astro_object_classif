from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC

from src.models import BaseModel


class SVCModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LinearSVCModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = LinearSVC(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.decision_function(X)
