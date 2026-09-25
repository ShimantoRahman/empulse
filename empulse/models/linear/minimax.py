from scipy.optimize import OptimizeResult

from ..._types import FloatNDArray
from .._base import BaseMinimaxProbabilityMachine


class ProfMPMClassifier(BaseMinimaxProbabilityMachine):
    """
    Profit-driven minimax probability machine classifier with a shared worst-case accuracy bound.

    Learns a linear decision boundary that maximizes the worst-case (distribution-free)
    expected profit, using only the empirical means and covariances of each class
    through the multivariate Chebyshev-Cantelli inequality.

    Unlike :class:`~empulse.models.ProfMEMPMClassifier`, which allows the worst-case class
    accuracies to differ between classes, this model constrains both classes to share the
    same worst-case accuracy bound, determined by the tighter (minimum) of the two class bounds.

    Setting ``lambda_reg=0`` (default) reproduces the original Profit Maximizing Minimax
    Probability Machine (ProfMPM): the weight vector is unconstrained and no additional
    regularization is applied.

    Setting ``lambda_reg>0`` switches to the Lp-regularized variant (Lp-ProfMPM):
    an L1 or L2 penalty (controlled by ``penalty``) on the weight vector is added
    to the objective, and each class mean must lie at least one unit inside its own
    half-space, so the penalty trades worst-case accuracy for smaller weights.

    Read more in the :ref:`User Guide <profmpm>`.

    Parameters
    ----------
    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    loss : :class:`~empulse.metrics.BaseMetric` or None, default=None
        Only :class:`~empulse.metrics.BaseMetric` instances built with the
        :class:`~empulse.metrics.MaxProfit` strategy are supported, since this model requires
        the costs and benefits to be reducible to four scalar values.

        .. note::
            If the costs or benefits contain stochastic variables, they are replaced by
            their mean/expectation before fitting.

        If :class:`~empulse.metrics.BaseMetric`, metric parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.ProfMPMClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    penalty : 'l1' or 'l2', default='l2'
        Norm used in the regularization term. Only used when ``lambda_reg > 0``.

    lambda_reg : float, default=0.0
        Regularization strength of the ``penalty`` term. Must be non-negative.

        If ``0.0``, no regularization is applied, reproducing the original
        (non-regularized) ProfMPM formulation.

        If greater than ``0.0``, ``w`` is regularized, reproducing the Lp-ProfMPM formulation.

    ridge_penalty : float, default=1e-6
        Small positive value added to the diagonal of the empirical covariance matrices
        to keep them positive definite (Tikhonov/ridge stabilization).
        This is independent of ``lambda_reg`` and is always applied.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    coef_ : numpy.ndarray, shape=(n_features,)
        Coefficients of the linear decision boundary.

    intercept_ : float
        Intercept of the linear decision boundary.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    References
    ----------
    .. [1] Maldonado, S., López, J., & Vairetti, C. (2020).
       Profit-based churn prediction based on minimax probability machines.
       European Journal of Operational Research, 284(1), 273-284.

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfMPMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfMPMClassifier()
        model.fit(X, y, tp_cost=-200, fp_cost=10)
    """

    def _worst_case_accuracies(self, k_1: float, k_0: float) -> tuple[float, float]:
        # ProfMPM forces alpha_1 = alpha_0 = alpha.
        # The maximum valid kappa for a shared alpha is constrained by the tighter of the two class bounds.
        k_min = min(k_1, k_0)
        alpha = (k_min**2 / (1 + k_min**2)) if k_min > 0 else 0.0
        return alpha, alpha

    def _fit_minimax(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        if self.lambda_reg == 0.0:
            return self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)
        return self._solve_regularized_mpm(mu_1, mu_0, sigma_1, sigma_0, c1, c0)


class ProfMEMPMClassifier(BaseMinimaxProbabilityMachine):
    r"""
    Profit-driven minimax probability machine classifier.

    Learns a linear decision boundary that maximizes the worst-case (distribution-free)
    expected profit, using only the empirical means and covariances of each class
    through the multivariate Chebyshev-Cantelli inequality.

    Setting ``lambda_reg=0`` (default) reproduces the original Profit Maximizing Minimax
    Probability Machine (MEMPM): the weight vector is scaled by the canonical constraint
    :math:`w^T(\mu_1 - \mu_0) = 1` and no additional regularization is applied.

    Setting ``lambda_reg>0`` switches to the Lp-regularized variant (Lp-ProfMEMPM):
    the canonical scale constraint is replaced by requiring each class mean to lie at least
    one unit inside its own half-space, and an L1 or L2 penalty (controlled by ``penalty``)
    on the weight vector is added to the objective, trading worst-case accuracy for smaller
    weights.

    Read more in the :ref:`User Guide <profmempm>`.

    .. seealso::

        :class:`~empulse.models.ProfMPMClassifier` : The unregularized minimax probability
        machine this class extends.

    Parameters
    ----------
    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    loss : :class:`~empulse.metrics.BaseMetric` or None, default=None
        Only :class:`~empulse.metrics.BaseMetric` instances built with the
        :class:`~empulse.metrics.MaxProfit` strategy are supported, since this model requires
        the costs and benefits to be reducible to four scalar values.

        .. note::
            If the costs or benefits contain stochastic variables, they are replaced by
            their mean/expectation before fitting.

        If :class:`~empulse.metrics.BaseMetric`, metric parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.ProfMEMPMClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    penalty : 'l1' or 'l2', default='l2'
        Norm used in the regularization term. Only used when ``lambda_reg > 0``.

    lambda_reg : float, default=0.0
        Regularization strength of the ``penalty`` term. Must be non-negative.

        If ``0.0``, no regularization is applied and ``w`` is instead scaled by the
        canonical constraint :math:`w^T(\mu_1 - \mu_0) = 1`, reproducing the original
        (non-regularized) MEMPM formulation.

        If greater than ``0.0``, the canonical scale constraint is dropped and ``w`` is
        regularized instead, reproducing the Lp-ProfMEMPM formulation.

    ridge_penalty : float, default=1e-6
        Small positive value added to the diagonal of the empirical covariance matrices
        to keep them positive definite (Tikhonov/ridge stabilization).
        This is independent of ``lambda_reg`` and is always applied.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    coef_ : numpy.ndarray, shape=(n_features,)
        Coefficients of the linear decision boundary.

    intercept_ : float
        Intercept of the linear decision boundary.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    References
    ----------
    .. [1] Maldonado, S., López, J., & Vairetti, C. (2020).
       Profit-based churn prediction based on minimax probability machines.
       European Journal of Operational Research, 284(1), 273-284.

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfMEMPMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfMEMPMClassifier()
        model.fit(X, y, tp_cost=-200, fp_cost=10)
    """

    def _worst_case_accuracies(self, k_1: float, k_0: float) -> tuple[float, float]:
        alpha_1 = (k_1**2 / (1 + k_1**2)) if k_1 > 0 else 0.0
        alpha_0 = (k_0**2 / (1 + k_0**2)) if k_0 > 0 else 0.0
        return alpha_1, alpha_0

    def _fit_minimax(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        if self.lambda_reg == 0.0:
            return self._solve_unregularized_mempm(mu_1, mu_0, sigma_1, sigma_0, c1, c0)
        return self._solve_regularized_mempm(mu_1, mu_0, sigma_1, sigma_0, c1, c0)
