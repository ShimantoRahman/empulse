from sklearn.base import OneToOneFeatureMixin, BaseEstimator


class FairSampler(OneToOneFeatureMixin, BaseEstimator):

    _estimator_type = "sampler"

    def __init__(self, *, method='statistical parity', protected_attr=None):
        self.protected_attr = protected_attr
        self.method = method

    def fit_resample(self, X, y, protected_attr=None):
        pass
