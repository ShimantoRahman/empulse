import numpy as np
import pandas as pd
from sklearn.base import OneToOneFeatureMixin, BaseEstimator
from sklearn.utils import _safe_indexing


# TODO: look into imbalanced learn baseclasses
class FairRelabler(OneToOneFeatureMixin, BaseEstimator):

    _estimator_type = "sampler"

    def __init__(self, estimator, *, method='statistical parity', process_attr=None):
        self.estimator = estimator
        self.process_attr = process_attr
        self.method = method
        self.promotion_candidates = []
        self.demotion_candidates = []

    def fit_resample(self, X, y, protected_attr=None):
        if self.process_attr is not None:
            protected_attr = self.process_attr(protected_attr)

        self.estimator.fit(X, y)
        probas = self.estimator.predict_proba(X)

        # determine number of promotor and demotor pairs to achieve demographic parity
        protected_indices = np.where(protected_attr == 0)[0]
        unprotected_indices = np.where(protected_attr == 1)[0]
        n_protected = len(protected_indices)
        n_unprotected = len(unprotected_indices)
        n = n_protected + n_unprotected

        pos_decision_ratio_protected_indices = np.sum(_safe_indexing(y, protected_indices)) / n_protected
        pos_decision_ratio_unprotected_indices = np.sum(_safe_indexing(y, unprotected_indices)) / n_unprotected

        disc = pos_decision_ratio_unprotected_indices - pos_decision_ratio_protected_indices

        n_pairs = abs(round((disc * n_protected * n_unprotected) / n))

        probas_unprotected = pd.DataFrame(probas[unprotected_indices])
        probas_unprotected.index = unprotected_indices

        probas_protected = pd.DataFrame(probas[protected_indices])
        probas_protected.index = protected_indices

        if n_pairs > 0:
            self.demotion_candidates = get_doubtful_positive_cases(
                probas_unprotected,
                _safe_indexing(y, unprotected_indices),
                n_pairs
            ).index
            self.promotion_candidates = get_doubtful_negative_cases(
                probas_protected,
                _safe_indexing(y, protected_indices),
                n_pairs
            ).index

        # relabel the data
        relabled_y = y.copy()
        if isinstance(relabled_y, pd.Series):
            relabled_y.iloc[self.demotion_candidates] = 0
            relabled_y.iloc[self.promotion_candidates] = 1
        else:
            relabled_y[self.demotion_candidates] = 0
            relabled_y[self.promotion_candidates] = 1


# these are the cases of the unprotected indices that need to change
def get_doubtful_positive_cases(probability_labels, class_labels, n_doubtful_cases):
    """ Returns the n_doubtful_cases instances with the highest probability of being positive class label """
    indices_with_positive_class_label = np.where(class_labels == 1)[0]
    probability_labels_of_positive_class_labels = probability_labels.iloc[indices_with_positive_class_label]
    sorted_probability_labels = order_instances(probability_labels_of_positive_class_labels)
    demotion_candidates = sorted_probability_labels.iloc[0: n_doubtful_cases]
    return demotion_candidates


# these are the cases of the protected indices that need to change
def get_doubtful_negative_cases(probability_labels, class_labels, n_doubtful_cases):
    """ Returns the n_doubtful_cases instances with the lowest probability of being negative class label """
    indices_with_negative_class_label = np.where(class_labels == 0)[0]
    probability_labels_of_negative_class_labels = probability_labels.iloc[indices_with_negative_class_label]
    sorted_probability_labels = order_instances(probability_labels_of_negative_class_labels)
    promotion_candidates = sorted_probability_labels.iloc[-n_doubtful_cases:]
    return promotion_candidates


def order_instances(probability_labels):
    sort_by_probability = probability_labels.sort_values(1)
    return sort_by_probability
