from typing import Callable, Literal, Union, Optional

from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator

Method = Literal['statistical parity', 'equal opportunity']


def statistical_parity(y_true, protected_attr):
    pass


def equal_opportunity(y_true, protected_attr):
    pass


class ReweightingClassifier(BaseEstimator, ClassifierMixin):

    method_mapping = {
        'statistical parity': statistical_parity,
        'equal opportunity': equal_opportunity
    }

    def __init__(
            self,
            estimator,
            method: Union[Callable, Method] = 'statistical parity',
            process_attr: Optional[Callable] = None
    ):
        self.estimator = estimator
        if isinstance(method, str):
            method = self.method_mapping[method]
        self.method = method
        self.process_attr = process_attr

    def fit(self, X, y, protected_attr: Optional[ArrayLike] = None, **fit_params):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

