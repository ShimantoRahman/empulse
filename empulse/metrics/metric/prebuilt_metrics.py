from .cost_matrix import CostMatrix
from .cost_metric import Cost
from .metric import Metric
from .metric_strategies import MaxProfit
from .savings_metric import Savings


def make_generic_cost_matrix() -> CostMatrix:
    """Create a generic cost matrix."""
    return CostMatrix().add_tp_cost('tp_cost').add_tn_cost('tn_cost').add_fp_cost('fp_cost').add_fn_cost('fn_cost')


def make_generic_cost_metric() -> Metric:
    """Create a generic cost metric."""
    return Metric(cost_matrix=make_generic_cost_matrix(), strategy=Cost())


def make_generic_savings_metric() -> Metric:
    """Create a generic savings metric."""
    return Metric(cost_matrix=make_generic_cost_matrix(), strategy=Savings())


def make_generic_max_profit_metric() -> Metric:
    """Create a generic maximum profit metric."""
    return Metric(cost_matrix=make_generic_cost_matrix(), strategy=MaxProfit())
