from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from scipy.spatial import ConvexHull

class ROCCHCalibratedClassifierCV(CalibratedClassifierCV):
    def __init__(
            self,
            estimator=None,
            cv=None,
            n_jobs=None,
            ensemble=True
    ):
        super().__init__(estimator=estimator, cv=cv, n_jobs=n_jobs, ensemble=ensemble)
        self.method = 'roc_convex_hull'

    def fit(self, X, y, **params):
        super().fit(X, y, **params)
        return self
