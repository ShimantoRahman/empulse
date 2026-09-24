"""A metric caches the names of its parameters, until its cost matrix changes."""

import pickle

import sympy

from empulse.metrics import Cost, CostMatrix, Metric

CLV, D = sympy.symbols('clv d')


def _metric():
    cost_matrix = CostMatrix().add_tp_benefit(CLV / 4).add_fp_cost(D).add_fn_cost(CLV / 10)
    return Metric(cost_matrix.set_default(clv=100, d=5), Cost())


def test_parameter_names_are_cached_until_the_cost_matrix_changes():
    metric = _metric()
    assert metric._all_parameters == {'clv', 'd'}
    # The returned set is a copy, so changing it cannot corrupt the cache.
    metric._all_parameters.add('typo')
    assert metric._all_parameters == {'clv', 'd'}

    metric.cost_matrix.alias('discount', 'd')
    assert metric._all_parameters == {'clv', 'd', 'discount'}
    metric.cost_matrix.add_fp_cost(sympy.Symbol('f'))
    assert metric._all_parameters == {'clv', 'd', 'discount', 'f'}


def test_parameter_name_cache_survives_pickling():
    metric = _metric()
    names = metric._all_parameters
    restored = pickle.loads(pickle.dumps(metric))
    assert restored._all_parameters == names
    # A metric pickled before the cache existed has no cache attribute at all.
    del restored._all_parameters_cache
    assert restored._all_parameters == names
