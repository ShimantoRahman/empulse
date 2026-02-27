from .cost_matrix import CostMatrix
from .metric import Metric, MetricStrategy
from .strategies import Cost, MaxProfit, Savings


def make_generic_cost_matrix() -> CostMatrix:
    """Create a generic cost matrix."""
    return CostMatrix().add_tp_cost('tp_cost').add_tn_cost('tn_cost').add_fp_cost('fp_cost').add_fn_cost('fn_cost')


def make_generic_cost_metric() -> Metric:
    """Create a generic cost metric."""
    return make_generic_metric(Cost())


def make_generic_savings_metric() -> Metric:
    """Create a generic savings metric."""
    return make_generic_metric(Savings())


def make_generic_max_profit_metric() -> Metric:
    """Create a generic maximum profit metric."""
    return make_generic_metric(MaxProfit())


def make_generic_metric(strategy: MetricStrategy) -> Metric:
    """Create a generic metric with the given strategy."""
    return Metric(cost_matrix=make_generic_cost_matrix(), strategy=strategy)
