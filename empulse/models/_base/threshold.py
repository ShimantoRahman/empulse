from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray
from ...metrics import BaseMetric

THRESHOLD_CONSTRAINT = [Interval(Real, 0, 1, closed='both'), StrOptions({'auto'})]


class LearnsThreshold:
    """
    Mixin for a cost-sensitive classifier whose :meth:`predict` applies a threshold learned during fit.

    The classifier stores its ``threshold`` parameter and lists :data:`THRESHOLD_CONSTRAINT` among its
    parameter constraints. With ``threshold='auto'``, ``threshold_`` is the score above which
    predicting the positive class costs the training samples least.
    """

    threshold: float | Literal['auto']
    classes_: NDArray[Any]

    if TYPE_CHECKING:

        def predict_proba(self, X: FloatArrayLike) -> FloatNDArray: ...

    def _finish_fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, loss_params: dict[str, Any]) -> None:
        if self.threshold == 'auto':
            y_score = self.predict_proba(X)[:, 1]
            self.threshold_ = loss.model.decision_threshold(y, y_score, **loss_params)
        else:
            self.threshold_ = float(self.threshold)

    def predict(self, X: FloatArrayLike) -> NDArray[Any]:
        """
        Predict class labels for samples in X.

        A sample is predicted positive when its predicted probability of the positive class is above
        ``threshold_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels for each sample.
        """
        check_is_fitted(self)
        y_score = self.predict_proba(X)[:, 1]
        y_pred: NDArray[Any] = self.classes_[(y_score > self.threshold_).astype(np.intp)]
        return y_pred
