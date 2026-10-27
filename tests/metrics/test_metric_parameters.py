"""
The parameters a :class:`~empulse.metrics.Metric` asks for: which are missing, which are aliases of
one another, which are outlier-sensitive, and how their names are cached.
"""

import pickle
import warnings

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import (
    Cost,
    CostMatrix,
    MaxProfit,
    Metric,
)

Y_TRUE = np.array([1, 0, 1, 0, 1])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.2, 0.7])


# --- Missing and outlier-sensitive parameters --------------------------------------------------------


class TestMissingParameters:
    """
    Tests for ``ModelInterface.missing_parameters``, used by CSThresholdClassifier/CSRateClassifier.

    A caller supplies one spelling per parameter (a symbol's raw name or its alias), so either must
    count as covered. Comparing the caller's keys for equality against `_all_parameters`, which lists
    both spellings, could never succeed for an aliased metric.
    """

    @staticmethod
    def _aliased_metric():
        clv, d = sympy.symbols('clv d')
        return Metric(CostMatrix().add_fn_cost(clv).add_fp_cost(d).alias({'incentive_cost': 'd'}), Cost())

    def test_nothing_supplied(self):
        metric = self._aliased_metric()
        assert metric.model.missing_parameters([]) == {'clv', 'd'}

    def test_supplied_by_symbol_name(self):
        metric = self._aliased_metric()
        assert metric.model.missing_parameters(['clv', 'd']) == set()

    def test_supplied_by_alias(self):
        """A parameter is satisfied by its alias just as well as by its raw symbol name."""
        metric = self._aliased_metric()
        assert metric.model.missing_parameters(['clv', 'incentive_cost']) == set()

    def test_partial_supply_still_reports_missing(self):
        metric = self._aliased_metric()
        assert metric.model.missing_parameters(['clv']) == {'d'}

    def test_unrelated_extra_keys_are_ignored(self):
        """Extra keys the metric doesn't recognize (e.g. sample_weight) don't affect the result."""
        metric = self._aliased_metric()
        assert metric.model.missing_parameters(['clv', 'incentive_cost', 'sample_weight']) == set()

    def test_default_parameters_are_never_missing(self):
        clv, d = sympy.symbols('clv d')
        metric = Metric(CostMatrix().add_fn_cost(clv).add_fp_cost(d).set_default(clv=100.0), Cost())
        assert metric.model.missing_parameters([]) == {'d'}
        assert metric.model.missing_parameters(['d']) == set()


class TestOutlierSensitiveParameters:
    """Tests for ``ModelInterface.outlier_sensitive_parameters``, used by RobustCSClassifier."""

    def test_no_outlier_sensitive_parameters(self):
        metric = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost('b'), Cost())
        assert metric.model.outlier_sensitive_parameters() == {}

    def test_positive_class_parameter(self):
        """A symbol used only in tp_cost/fn_cost is classified 'positive'."""
        a = sympy.Symbol('a')
        metric = Metric(CostMatrix().add_tp_benefit(a).add_fp_cost('b').mark_outlier_sensitive(a), Cost())
        assert metric.model.outlier_sensitive_parameters() == {'a': 'positive'}

    def test_negative_class_parameter(self):
        """A symbol used only in tn_cost/fp_cost is classified 'negative'."""
        b = sympy.Symbol('b')
        metric = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost(b).mark_outlier_sensitive(b), Cost())
        assert metric.model.outlier_sensitive_parameters() == {'b': 'negative'}

    def test_both_classes_parameter(self):
        """A symbol used on both sides is classified 'both'."""
        a = sympy.Symbol('a')
        metric = Metric(
            CostMatrix().add_tp_benefit(a).add_fp_cost(a).mark_outlier_sensitive(a),
            Cost(),
        )
        assert metric.model.outlier_sensitive_parameters() == {'a': 'both'}

    def test_returned_under_its_symbol_name_with_every_spelling_available(self):
        """The key is the symbol name; `_parameter_spellings` gives every keyword it can be passed as."""
        d = sympy.Symbol('d')
        metric = Metric(
            CostMatrix().add_fp_cost(d).alias({'incentive_cost': 'd'}).mark_outlier_sensitive(d),
            Cost(),
        )
        assert metric.model.outlier_sensitive_parameters() == {'d': 'negative'}
        assert metric.model.parameter_spellings('d') == {'d', 'incentive_cost'}

    def test_a_symbol_with_two_aliases(self):
        """Inverting the alias map to find "the" alias raised for a symbol with two of them."""
        metric = Metric(
            CostMatrix()
            .add_fn_cost('clv')
            .add_fp_cost(1)
            .alias({'value': 'clv', 'customer_value': 'clv'})
            .mark_outlier_sensitive('value'),
            Cost(),
        )
        assert metric.model.outlier_sensitive_parameters() == {'clv': 'positive'}

    def test_a_distribution_parameter_is_classified_by_the_costs_it_enters(self):
        alpha = sympy.Symbol('alpha')
        gamma = sympy.stats.Beta('gamma', alpha, 2)
        metric = Metric(
            CostMatrix().add_tp_benefit(gamma * sympy.Symbol('clv')).add_fp_cost(1).mark_outlier_sensitive(alpha),
            MaxProfit(),
        )
        assert metric.model.outlier_sensitive_parameters() == {'alpha': 'positive'}

    def test_marked_by_alias(self):
        """Marking a symbol by its alias classifies the symbol the alias stands for."""
        d = sympy.Symbol('d')
        metric = Metric(
            CostMatrix()
            .add_fn_cost(d)
            .add_fp_cost('f')
            .alias('incentive_cost', d)
            .mark_outlier_sensitive('incentive_cost'),
            Cost(),
        )
        assert metric.model.outlier_sensitive_parameters() == {'d': 'positive'}

    def test_marked_by_name_of_a_symbol_with_assumptions(self):
        """``mark_outlier_sensitive('d')`` creates an assumption-free symbol, which must still match ``d``."""
        d = sympy.Symbol('d', positive=True)
        metric = Metric(CostMatrix().add_fn_cost(d).add_fp_cost('f').mark_outlier_sensitive('d'), Cost())
        assert metric.model.outlier_sensitive_parameters() == {'d': 'positive'}

    def test_unaliased_symbol_uses_raw_name(self):
        clv = sympy.Symbol('clv')
        metric = Metric(CostMatrix().add_tp_benefit(clv).mark_outlier_sensitive(clv), Cost())
        assert metric.model.outlier_sensitive_parameters() == {'clv': 'positive'}

    def test_multiple_outlier_sensitive_parameters(self):
        clv, d = sympy.symbols('clv d')
        metric = Metric(
            CostMatrix().add_tp_benefit(clv).add_fp_cost(d).mark_outlier_sensitive(clv).mark_outlier_sensitive(d),
            Cost(),
        )
        assert metric.model.outlier_sensitive_parameters() == {'clv': 'positive', 'd': 'negative'}


# --- Parameter names and aliases ---------------------------------------------------------------------


class TestParameterNames:
    """`parameter_names` lists what a caller can pass: every spelling, but no random variable."""

    def test_deterministic_symbols_and_their_aliases(self):
        metric = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost('b').alias('rate', 'a'), Cost())
        assert metric.parameter_names == {'a', 'b', 'rate'}

    def test_a_random_variable_is_replaced_by_its_distribution_parameters(self):
        """Passing every listed name must not warn that one is unknown."""
        alpha, beta = sympy.symbols('alpha beta')
        gamma = sympy.stats.Beta('gamma', alpha, beta)
        cost_matrix = (
            CostMatrix().add_tp_benefit(gamma * sympy.Symbol('clv')).add_fp_cost(1).alias({'accept_rate': 'gamma'})
        )
        metric = Metric(cost_matrix, MaxProfit())

        assert metric.parameter_names == {'alpha', 'beta', 'clv'}
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            metric(Y_TRUE, Y_SCORE, alpha=2.0, beta=3.0, clv=10.0)


class TestParameterSpellings:
    """`_parameter_spellings` lists every keyword one parameter is accepted under."""

    @staticmethod
    def _aliased_metric():
        cost_matrix = CostMatrix().add_fn_cost('d').add_fp_cost('f').alias({'amount': 'd', 'incentive': 'd'})
        return Metric(cost_matrix, Cost())

    @pytest.mark.parametrize('name', ['d', 'amount', 'incentive'])
    def test_any_spelling_gives_all_of_them(self, name):
        assert self._aliased_metric().model.parameter_spellings(name) == {'d', 'amount', 'incentive'}

    def test_an_unaliased_symbol_has_one_spelling(self):
        assert self._aliased_metric().model.parameter_spellings('f') == {'f'}


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


def test_parameter_names_are_a_copy():
    metric = _metric()
    assert metric.parameter_names == {'clv', 'd'}
    metric.parameter_names.add('typo')
    assert metric.parameter_names == {'clv', 'd'}


def test_changing_the_cost_matrix_copy_does_not_change_the_metric():
    metric = _metric()
    expected = metric(Y_TRUE, Y_SCORE)
    metric.cost_matrix.alias('discount', 'd').add_fp_cost(sympy.Symbol('f'))
    assert metric.parameter_names == {'clv', 'd'}
    assert metric(Y_TRUE, Y_SCORE) == expected


def test_parameter_names_survive_pickling():
    metric = _metric()
    names = metric.parameter_names
    restored = pickle.loads(pickle.dumps(metric))
    assert restored.parameter_names == names
