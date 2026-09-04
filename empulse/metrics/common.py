import numpy as np

from .._types import FloatArrayLike, FloatNDArray


def classification_threshold(y_true: FloatArrayLike, y_score: FloatArrayLike, customer_threshold: float) -> float:
    """
    Return classification threshold for given customer threshold.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    customer_threshold : float
        Customer threshold determined by value-driven metric.

    Returns
    -------
    threshold : float
        Classification threshold for given customer threshold.

    Examples
    --------
    >>> from empulse.metrics import classification_threshold
    >>> from empulse.metrics import empc_score
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> threshold = empc_score.optimal_rate(y_true, y_score)
    >>> classification_threshold(y_true, y_score, threshold)
    0.2
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    confusion_matrix, sorted_indices, duplicated_prediction_indices = _compute_confusion_matrix(y_true, y_score)
    classification_thresholds = np.pad(y_score[sorted_indices], pad_width=(1, 0), constant_values=1)
    classification_thresholds = np.delete(classification_thresholds, duplicated_prediction_indices)  # type: ignore[arg-type]
    customer_thresholds = np.sum(confusion_matrix, axis=0) / y_score.shape[0]
    return float(classification_thresholds[np.argmin(np.abs(customer_thresholds - customer_threshold))])


def _compute_confusion_matrix(
    y_true: FloatNDArray, y_pred: FloatNDArray
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    # sort true labels and predictions by highest to the lowest predicted score
    sorted_indices = y_pred.argsort()[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_predictions = y_pred[sorted_indices]

    # calculate the TP & FP at each new lead targeted
    true_positives = np.pad(np.cumsum(sorted_labels), pad_width=(1, 0))
    false_positives = np.pad(np.cumsum(sorted_labels == 0), pad_width=(1, 0))

    # merge consecutive equal prediction values
    duplicated_prediction_indices = np.where(np.diff(sorted_predictions) == 0)[0] + 1
    true_positives = np.delete(true_positives, duplicated_prediction_indices)
    false_positives = np.delete(false_positives, duplicated_prediction_indices)

    return np.array([true_positives, false_positives]), sorted_indices, duplicated_prediction_indices
