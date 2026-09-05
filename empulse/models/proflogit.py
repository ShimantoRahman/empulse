from typing import ClassVar

from empulse.optimizers import GeneticAlgorithmOptimizer, Optimizer

from ..metrics import MaxProfit
from ._base import BaseLogitClassifier
from .csclassifier import MetricStrategyFactory


class ProfLogitClassifier(BaseLogitClassifier):
    """
    Profit-driven logistic regression classifier.

    Maximizing empirical cost-sensitive/value-driven metric
    by optimizing the regression coefficients of the logistic model through a Real-coded Genetic Algorithm (RGA).

    Read more in the :ref:`User Guide <proflogit>`.

    Parameters
    ----------
    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    loss : :class:`empulse.metrics.BaseMetric` or None, default=None
        Loss function to optimize.

        If :class:`~empulse.metrics.BaseMetric`, metric parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.ProfLogitClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive ``float``.
        Like in support vector machines, smaller values specify stronger regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    soft_threshold : bool, default=True
        If ``True``, apply soft-thresholding to the regression coefficients.

    l1_ratio : float, default=1.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is a L2 penalty.
        For ``l1_ratio = 1`` it is a L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    optimizer : :class:`empulse.optimizers.Optimizer`, optional
        Optimization algorithm. See :ref:`proflogit` for more information.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    coef_ : numpy.ndarray
        Coefficients of the logit model.

    intercept_ : float
        Intercept of the logit model.
        Only available when ``fit_intercept=True``.

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfLogitClassifier
        from empulse.optimizers import GeneticAlgorithmOptimizer
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfLogitClassifier(
            C=0.1, l1_ratio=0.5, optimizer=GeneticAlgorithmOptimizer(max_iter=10)
        )
        model.fit(X, y, tp_cost=-200, fp_cost=10)

    References
    ----------
    .. [1] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2017). Profit Maximizing Logistic Model for
        Customer Churn Prediction Using Genetic Algorithms.
        Swarm and Evolutionary Computation.
    .. [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2015). Profit Maximizing Logistic Regression Modeling for
        Customer Churn Prediction. IEEE International Conference on
        Data Science and Advanced Analytics (DSAA) (pp. 1–10). Paris, France.
    """

    _default_metric_strategy: ClassVar[MetricStrategyFactory] = MaxProfit
    _default_optimizer: ClassVar[type[Optimizer]] = GeneticAlgorithmOptimizer
