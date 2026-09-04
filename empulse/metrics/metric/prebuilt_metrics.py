import sympy
import sympy.stats

from .cost_matrix import CostMatrix
from .metric import Metric
from .mixture_metric import MixtureComponent, MixtureMetric
from .strategies import AUEPC, Cost, EmpiricalMaxProfit, LogCost, MaxProfit, MetricStrategy, Savings


def make_generic_cost_matrix() -> CostMatrix:
    """Create a generic cost matrix."""
    return (
        CostMatrix()
        .add_tp_cost('tp_cost')
        .add_tn_cost('tn_cost')
        .add_fp_cost('fp_cost')
        .add_fn_cost('fn_cost')
        .set_default(tp_cost=0.0, tn_cost=0.0, fp_cost=0.0, fn_cost=0.0)
    )


def make_generic_cost_metric() -> Metric:
    """Create a generic cost metric."""
    return make_generic_metric(Cost())


def make_generic_log_cost_metric() -> Metric:
    """Create a generic log-cost metric."""
    return make_generic_metric(LogCost())


def make_generic_savings_metric() -> Metric:
    """Create a generic savings metric."""
    return make_generic_metric(Savings())


def make_generic_max_profit_metric() -> Metric:
    """Create a generic maximum profit metric."""
    return make_generic_metric(MaxProfit())


def make_generic_metric(strategy: MetricStrategy) -> Metric:
    """Create a generic metric with the given strategy."""
    return Metric(cost_matrix=make_generic_cost_matrix(), strategy=strategy)


# --- Customer churn ---------------------------------------------------------------------------


def _churn_cost_matrix_incentive_cost(gamma: sympy.Expr) -> CostMatrix:
    """Cost matrix parameterized by an absolute incentive cost (used by mpc_score/empc_score)."""
    clv, d, f = sympy.symbols('clv d f')
    return (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d - f))  # when churner accepts the incentive offer
        .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept the incentive offer
        .add_fp_cost(d + f)  # when an offer is sent to a non-churner
        .alias({'incentive_cost': 'd', 'contact_cost': 'f'})
    )


def _churn_cost_matrix_incentive_fraction(gamma: sympy.Expr) -> CostMatrix:
    """Cost matrix parameterized by a relative incentive fraction.

    Used by expected_cost_loss_churn/empb_score/auepc_score.
    """
    clv, delta, f = sympy.symbols('clv delta f')
    return (
        CostMatrix()
        .add_tp_benefit(gamma * ((1 - delta) * clv - f))  # when churner accepts the incentive offer
        .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept the incentive offer
        .add_fp_cost(delta * clv + f)  # when an offer is sent to a non-churner
        .alias({'incentive_fraction': 'delta', 'contact_cost': 'f'})
    )


def make_churn_max_profit_metric(*, stochastic: bool) -> Metric:
    """Create the churn MaxProfit metric (mpc_score if deterministic, empc_score if stochastic)."""
    if stochastic:
        alpha, beta = sympy.symbols('alpha beta')
        gamma = sympy.stats.Beta('gamma', alpha, beta)
        cost_matrix = _churn_cost_matrix_incentive_cost(gamma).set_default(
            alpha=6, beta=14, incentive_cost=10, contact_cost=1, clv=200
        )
    else:
        gamma = sympy.Symbol('gamma')
        cost_matrix = (
            _churn_cost_matrix_incentive_cost(gamma)
            .alias({'accept_rate': 'gamma'})
            .set_default(accept_rate=0.3, incentive_cost=10, contact_cost=1, clv=200)
        )
    return Metric(cost_matrix, MaxProfit())


def make_churn_cost_metric() -> Metric:
    """Create the churn expected cost metric (expected_cost_loss_churn)."""
    gamma = sympy.Symbol('gamma')
    cost_matrix = (
        _churn_cost_matrix_incentive_fraction(gamma)
        .alias({'accept_rate': 'gamma'})
        .set_default(accept_rate=0.3, incentive_fraction=0.05, contact_cost=1, clv=200)
    )
    return Metric(cost_matrix, Cost())


def make_churn_empirical_max_profit_metric() -> Metric:
    """Create the churn EmpiricalMaxProfit metric (empb_score)."""
    alpha, beta = sympy.symbols('alpha beta')
    gamma = sympy.stats.Beta('gamma', alpha, beta)
    cost_matrix = _churn_cost_matrix_incentive_fraction(gamma).set_default(
        alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15
    )
    return Metric(cost_matrix, EmpiricalMaxProfit())


def make_churn_auepc_metric() -> Metric:
    """Create the churn AUEPC metric (auepc_score)."""
    alpha, beta = sympy.symbols('alpha beta')
    gamma = sympy.stats.Beta('gamma', alpha, beta)
    cost_matrix = _churn_cost_matrix_incentive_fraction(gamma).set_default(
        alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15
    )
    return Metric(cost_matrix, AUEPC(normalize=True))


# --- Customer acquisition ----------------------------------------------------------------------


def _acquisition_cost_matrix(contribution: sympy.Expr) -> CostMatrix:
    contact_cost, sales_cost, direct_selling, commission = sympy.symbols(
        'contact_cost sales_cost direct_selling commission'
    )
    return (
        CostMatrix()
        .add_tp_benefit(
            direct_selling * (contribution - sales_cost - contact_cost)
            + (1 - direct_selling) * ((1 - commission) * contribution - contact_cost)
        )
        .add_fp_cost(contact_cost)
    )


def make_acquisition_max_profit_metric(*, stochastic: bool) -> Metric:
    """Create the acquisition MaxProfit metric (mpa_score if deterministic, empa_score if stochastic).

    Notes
    -----
    :class:`~empulse.metrics.MaxProfit`'s stochastic-integration engine requires a random
    variable's distribution parameters to be plain symbols, not derived expressions (e.g.
    ``1 / beta`` is not supported as a distribution argument). ``sympy.stats.Gamma`` is
    parameterized by shape and *scale*, whereas the native ``empa_score`` parameterizes its
    Gamma-distributed contribution by shape ``alpha`` and *rate* ``beta`` (mean = alpha / beta).
    To work within that constraint, ``beta`` here is passed directly as the scale (mean = alpha
    * beta) with its default inverted (``1 / 0.0015``) so that the default call matches the
    native function's default output exactly.
    """
    if stochastic:
        alpha, beta = sympy.symbols('alpha beta')  # beta is the Gamma *scale* here, see Notes above
        contribution = sympy.stats.Gamma('R', alpha, beta)
        cost_matrix = _acquisition_cost_matrix(contribution).set_default(
            alpha=12, beta=1 / 0.0015, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1
        )
    else:
        contribution = sympy.Symbol('R')
        cost_matrix = (
            _acquisition_cost_matrix(contribution)
            .alias({'contribution': 'R'})
            .set_default(contribution=8_000, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)
        )
    return Metric(cost_matrix, MaxProfit())


def make_acquisition_cost_metric() -> Metric:
    """Create the acquisition expected cost metric (expected_cost_loss_acquisition)."""
    contribution = sympy.Symbol('R')
    cost_matrix = (
        _acquisition_cost_matrix(contribution)
        .alias({'contribution': 'R'})
        .set_default(contribution=7_000, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)
    )
    return Metric(cost_matrix, Cost())


# --- Credit scoring -----------------------------------------------------------------------------


def make_credit_scoring_max_profit_metric() -> Metric:
    """Create the credit scoring MaxProfit metric (mpcs_score)."""
    lam, roi = sympy.symbols('lam roi')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(lam)
        .add_fp_cost(roi)
        .alias({'loan_lost_rate': 'lam'})
        .set_default(loan_lost_rate=0.275, roi=0.2644)
    )
    return Metric(cost_matrix, MaxProfit())


def make_credit_scoring_empirical_max_profit_metric() -> MixtureMetric:
    """
    Create the credit scoring EMP metric (empcs_score).

    The fraction of a defaulted loan that is lost (``lambda``) is 0 with probability
    ``success_rate`` (full recovery), 1 with probability ``default_rate`` (full loss), and
    otherwise follows a Uniform(0, 1) distribution. :mod:`sympy.stats` cannot represent this
    "spike + spike + continuous" mixture as a single random variable, so it is built as a
    :class:`~empulse.metrics.MixtureMetric` of three :class:`~empulse.metrics.Metric` components,
    exploiting linearity of expectation.
    """
    lam, roi = sympy.symbols('lam roi')
    credit_matrix = CostMatrix().add_tp_benefit(lam).add_fp_cost(roi)
    metric_det = Metric(credit_matrix, MaxProfit())

    lam_rv = sympy.stats.Uniform('lam', 0, 1)
    credit_matrix_stoch = CostMatrix().add_tp_benefit(lam_rv).add_fp_cost(roi)
    metric_stoch = Metric(credit_matrix_stoch, MaxProfit())

    return MixtureMetric(
        [
            MixtureComponent('success_rate', metric_det, {'lam': 0.0}),
            MixtureComponent('default_rate', metric_det, {'lam': 1.0}),
            MixtureComponent(lambda p: 1 - p['success_rate'] - p['default_rate'], metric_stoch, {}),
        ],
        defaults={'success_rate': 0.55, 'default_rate': 0.1, 'roi': 0.2644},
    )
