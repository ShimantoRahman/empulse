"""
The parameters a :class:`~empulse.metrics.Metric` asks for: which are missing, which are aliases of
one another, which are outlier-sensitive, and how their names are cached.
"""

import pickle

import numpy as np
import pytest
import sympy

from empulse.metrics import (
    Cost,
    CostMatrix,
    Metric,
)

Y_TRUE = np.array([1, 0, 1, 0, 1])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.2, 0.7])


# --- Missing and outlier-sensitive parameters --------------------------------------------------------


class TestMissingParameters:
    """
    Tests for Metric._missing_parameters, used by CSThresholdClassifier/CSRateClassifier.

    Regression coverage for a bug where comparing `_all_parameters` (which lists both a symbol's
    raw name and its alias) for equality against the caller-supplied keys could never succeed for
    an aliased metric, since a caller only ever supplies one spelling per parameter.
    """

    @staticmethod
    def _aliased_metric():
        clv, d = sympy.symbols('clv d')
        return Metric(CostMatrix().add_fn_cost(clv).add_fp_cost(d).alias({'incentive_cost': 'd'}), Cost())

    def test_nothing_supplied(self):
        metric = self._aliased_metric()
        assert metric._missing_parameters([]) == {'clv', 'd'}

    def test_supplied_by_symbol_name(self):
        metric = self._aliased_metric()
        assert metric._missing_parameters(['clv', 'd']) == set()

    def test_supplied_by_alias(self):
        """A parameter is satisfied by its alias just as well as by its raw symbol name."""
        metric = self._aliased_metric()
        assert metric._missing_parameters(['clv', 'incentive_cost']) == set()

    def test_partial_supply_still_reports_missing(self):
        metric = self._aliased_metric()
        assert metric._missing_parameters(['clv']) == {'d'}

    def test_unrelated_extra_keys_are_ignored(self):
        """Extra keys the metric doesn't recognize (e.g. sample_weight) don't affect the result."""
        metric = self._aliased_metric()
        assert metric._missing_parameters(['clv', 'incentive_cost', 'sample_weight']) == set()

    def test_default_parameters_are_never_missing(self):
        clv, d = sympy.symbols('clv d')
        metric = Metric(CostMatrix().add_fn_cost(clv).add_fp_cost(d).set_default(clv=100.0), Cost())
        assert metric._missing_parameters([]) == {'d'}
        assert metric._missing_parameters(['d']) == set()


class TestOutlierSensitiveParameters:
    """Tests for Metric._outlier_sensitive_parameters, used by RobustCSClassifier."""

    def test_no_outlier_sensitive_parameters(self):
        metric = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost('b'), Cost())
        assert metric._outlier_sensitive_parameters() == {}

    def test_positive_class_parameter(self):
        """A symbol used only in tp_cost/fn_cost is classified 'positive'."""
        a = sympy.Symbol('a')
        metric = Metric(CostMatrix().add_tp_benefit(a).add_fp_cost('b').mark_outlier_sensitive(a), Cost())
        assert metric._outlier_sensitive_parameters() == {'a': 'positive'}

    def test_negative_class_parameter(self):
        """A symbol used only in tn_cost/fp_cost is classified 'negative'."""
        b = sympy.Symbol('b')
        metric = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost(b).mark_outlier_sensitive(b), Cost())
        assert metric._outlier_sensitive_parameters() == {'b': 'negative'}

    def test_both_classes_parameter(self):
        """A symbol used on both sides is classified 'both'."""
        a = sympy.Symbol('a')
        metric = Metric(
            CostMatrix().add_tp_benefit(a).add_fp_cost(a).mark_outlier_sensitive(a),
            Cost(),
        )
        assert metric._outlier_sensitive_parameters() == {'a': 'both'}

    def test_returned_under_alias_not_raw_symbol_name(self):
        """The key is the caller-facing spelling (the alias), not the raw sympy symbol name."""
        d = sympy.Symbol('d')
        metric = Metric(
            CostMatrix().add_fp_cost(d).alias({'incentive_cost': 'd'}).mark_outlier_sensitive(d),
            Cost(),
        )
        assert metric._outlier_sensitive_parameters() == {'incentive_cost': 'negative'}

    def test_unaliased_symbol_uses_raw_name(self):
        clv = sympy.Symbol('clv')
        metric = Metric(CostMatrix().add_tp_benefit(clv).mark_outlier_sensitive(clv), Cost())
        assert metric._outlier_sensitive_parameters() == {'clv': 'positive'}

    def test_multiple_outlier_sensitive_parameters(self):
        clv, d = sympy.symbols('clv d')
        metric = Metric(
            CostMatrix().add_tp_benefit(clv).add_fp_cost(d).mark_outlier_sensitive(clv).mark_outlier_sensitive(d),
            Cost(),
        )
        assert metric._outlier_sensitive_parameters() == {'clv': 'positive', 'd': 'negative'}


# --- Parameter names and aliases ---------------------------------------------------------------------


class TestParameterNames:
    """`parameter_names` is the public alias for `_all_symbols`."""

    def test_matches_all_symbols(self):
        metric = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost('b'), Cost())
        assert metric.parameter_names == metric._all_symbols == {'a', 'b'}


@pytest.mark.parametrize('kwarg_order', ['symbol_first', 'alias_first'])
def test_alias_and_underlying_symbol_together_raises(kwarg_order):
    """
    Passing both a symbol and one of its aliases must raise, regardless of which one the caller
    happened to type first - not silently let whichever kwarg is seen last win, which would make
    the result depend on kwarg order.
    """
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())
    kwargs = (
        {'a': 1.0, 'my_cost': 999.0, 'b': 1.0}
        if kwarg_order == 'symbol_first'
        else {
            'my_cost': 999.0,
            'a': 1.0,
            'b': 1.0,
        }
    )

    with pytest.raises(ValueError, match="conflicting values for symbol 'a'"):
        metric(Y_TRUE, Y_SCORE, **kwargs)


def test_two_aliases_for_the_same_symbol_together_raises():
    """Passing two different aliases that both target the same symbol must raise too."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a', 'other_name': 'a'})
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match="conflicting values for symbol 'a'"):
        metric(Y_TRUE, Y_SCORE, my_cost=1.0, other_name=2.0, b=1.0)


def test_alias_or_symbol_passed_alone_is_unaffected():
    """Sanity check: passing only the symbol, or only the alias, must still work as before."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())

    assert metric(Y_TRUE, Y_SCORE, a=1.0, b=1.0) == pytest.approx(0.18)
    assert metric(Y_TRUE, Y_SCORE, my_cost=1.0, b=1.0) == pytest.approx(0.18)


# --- The parameter-name cache ------------------------------------------------------------------------


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
