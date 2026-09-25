"""
Cost handling defined once, by ``CostSensitiveClassifier``, for every model.

* the all-costs-zero warning and its ``fp_cost = fn_cost = 1`` fallback -- additionally suppressed
  suite-wide by ``filterwarnings`` in ``pyproject.toml``, so it needs an explicit ``pytest.warns``;
* binary-only enforcement and the recoding of ``classes_`` to 0/1;
* the ``Parameter.UNCHANGED`` contract: a cost passed to ``fit`` overrides the constructor's;
* cost shape normalisation, and cost-matrix symbols that share a name with a ``fit`` argument.
"""

import numpy as np
import pytest
from xgboost import XGBClassifier

from empulse.metrics import Cost, CostMatrix, Metric
from empulse.models import (
    CSBoostClassifier,
    CSLogitClassifier,
    CSTreeClassifier,
    RobustCSClassifier,
)
from empulse.optimizers import LBFGSBOptimizer


# Estimators that take the four plain cost arguments both at construction and at fit time.
def _cslogit(**kw):
    return CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5), **kw)


def _cstree(**kw):
    return CSTreeClassifier(max_depth=2, random_state=42, **kw)


def _csboost(**kw):
    return CSBoostClassifier(XGBClassifier(n_estimators=5, max_depth=2), **kw)


COST_TAKING_ESTIMATORS = [
    pytest.param(_cslogit, id='CSLogitClassifier'),
    pytest.param(_cstree, id='CSTreeClassifier'),
    pytest.param(_csboost, id='CSBoostClassifier'),
]


@pytest.fixture(scope='module')
def cost_data(make_data):
    return make_data(n_samples=80, n_features=5)


@pytest.fixture(scope='module')
def small_data(make_data):
    return make_data(n_samples=20, n_features=4, random_state=0)


# --- Cost shapes and names ---------------------------------------------------------------------


class TestNormalizeCostShapes:
    """
    `Metric._prepare_parameters` accepts a length-1 array as shorthand for "one value applied to
    every sample", but `_normalize_cost_shapes` used to reject anything whose size wasn't exactly
    `n_samples`, so a plain length-1 cost array was rejected by the model even though the metric
    itself would have accepted it.
    """

    def test_length_one_array_cost_is_accepted(self, small_data):
        X, y = small_data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.array([2.0]), fn_cost=5.0)

    def test_full_length_array_cost_is_accepted(self, small_data):
        X, y = small_data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.full(y.shape[0], 2.0), fn_cost=5.0)

    def test_error_message_names_expected_lengths(self, small_data):
        X, y = small_data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        with pytest.raises(ValueError, match=f'expected length {y.shape[0]}'):
            model.fit(X, y, fp_cost=np.array([1.0, 2.0]), fn_cost=5.0)


class TestCostSymbolsCollidingWithFitParameters:
    """
    A cost matrix may name one of its symbols ``tp_cost``/``tn_cost``/``fp_cost``/``fn_cost`` --
    several of the bundled datasets do (e.g. ``fetch_give_me_some_credit`` supplies ``'cl'`` and
    ``'fp_cost'``). Those names collide with ``fit``'s own keyword parameters, so the value used to
    bind to the parameter and be dropped instead of reaching the metric, and training failed with
    ``TypeError: _lambdifygenerated() missing 1 required positional argument: 'fp_cost'``.
    """

    @staticmethod
    def _metric():
        return Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('cl'), Cost())

    def test_colliding_symbol_is_routed_to_the_loss(self, small_data):
        X, y = small_data
        model = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.full(y.shape[0], 2.0), cl=5.0)

    def test_colliding_symbol_influences_training(self, small_data):
        """Routing must actually reach the objective, not merely avoid the TypeError."""
        X, y = small_data
        cheap = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=20))
        cheap.fit(X, y, fp_cost=np.full(y.shape[0], 1.0), cl=5.0)
        pricey = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=20))
        pricey.fit(X, y, fp_cost=np.full(y.shape[0], 100.0), cl=5.0)

        assert not np.allclose(cheap.predict_proba(X)[:, 1], pricey.predict_proba(X)[:, 1])

    def test_all_four_names_can_be_symbols(self, small_data):
        X, y = small_data
        cost_matrix = (
            CostMatrix()
            .add_tp_benefit('tp_cost')
            .add_tn_benefit('tn_cost')
            .add_fp_cost('fp_cost')
            .add_fn_cost('fn_cost')
        )
        model = CSLogitClassifier(loss=Metric(cost_matrix, Cost()), optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, tp_cost=1.0, tn_cost=0.0, fp_cost=2.0, fn_cost=3.0)

    def test_cost_not_used_by_the_metric_warns(self, small_data):
        X, y = small_data
        model = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=5))
        with pytest.warns(UserWarning, match='tn_cost passed to CSLogitClassifier.fit'):
            model.fit(X, y, fp_cost=2.0, cl=5.0, tn_cost=7.0)

    def test_no_warning_when_costs_are_not_passed(self, small_data, recwarn):
        X, y = small_data
        model = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=2.0, cl=5.0)

        assert not [w for w in recwarn if 'ignored because a `loss` metric is set' in str(w.message)]

    def test_init_time_costs_do_not_leak_into_the_metric(self, small_data):
        """
        ``__init__`` costs describe plain costs and default to 0.0, so forwarding them would
        silently zero out a cost matrix symbol of the same name.
        """
        X, y = small_data
        default = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=20))
        default.fit(X, y, fp_cost=np.full(y.shape[0], 3.0), cl=5.0)
        explicit_zero_init = CSLogitClassifier(loss=self._metric(), fp_cost=0.0, optimizer=LBFGSBOptimizer(max_iter=20))
        explicit_zero_init.fit(X, y, fp_cost=np.full(y.shape[0], 3.0), cl=5.0)

        np.testing.assert_allclose(default.predict_proba(X)[:, 1], explicit_zero_init.predict_proba(X)[:, 1])


# --- The all-costs-zero fallback ---------------------------------------------------------------


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_all_zero_costs_warns(make_estimator, cost_data):
    X, y = cost_data
    with pytest.warns(UserWarning, match='All costs are zero'):
        make_estimator().fit(X, y)


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_all_zero_costs_falls_back_to_unit_costs(make_estimator, cost_data):
    """The fallback must be ``fp_cost = fn_cost = 1``, not merely "something non-zero"."""
    X, y = cost_data
    with pytest.warns(UserWarning, match='All costs are zero'):
        fallback = make_estimator().fit(X, y)
    explicit = make_estimator(fp_cost=1, fn_cost=1).fit(X, y)
    assert np.array_equal(fallback.predict(X), explicit.predict(X))


# --- Binary-only enforcement and `classes_` recoding -------------------------------------------


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_multiclass_target_is_rejected(make_estimator, cost_data):
    X, y = cost_data
    y_multiclass = np.arange(len(y)) % 3
    with pytest.raises(ValueError, match='Only binary classification is supported'):
        make_estimator(fp_cost=1, fn_cost=1).fit(X, y_multiclass)


LABEL_ROUND_TRIP_ESTIMATORS = [
    pytest.param(_cslogit, id='CSLogitClassifier'),
    pytest.param(_csboost, id='CSBoostClassifier'),
    pytest.param(_cstree, id='CSTreeClassifier'),
]


@pytest.mark.parametrize('make_estimator', LABEL_ROUND_TRIP_ESTIMATORS)
def test_non_zero_one_labels_are_recoded_and_restored(make_estimator, cost_data):
    """
    Costs are defined against the positive class, so ``y`` is recoded to 0/1 internally.

    ``classes_`` must still report the caller's own labels, and ``predict`` must return them.
    """
    X, y = cost_data
    labels = np.where(y == 1, 'yes', 'no')
    model = make_estimator(fp_cost=1, fn_cost=1).fit(X, labels)
    assert list(model.classes_) == ['no', 'yes']
    assert set(np.unique(model.predict(X))) <= {'no', 'yes'}


# --- `Parameter.UNCHANGED` ----------------------------------------------------------------------


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_constructor_costs_are_kept_when_fit_omits_them(make_estimator, cost_data):
    X, y = cost_data
    from_init = make_estimator(fp_cost=1.0, fn_cost=10.0).fit(X, y)
    from_fit = make_estimator().fit(X, y, fp_cost=1.0, fn_cost=10.0)
    assert np.allclose(from_init.predict_proba(X), from_fit.predict_proba(X))


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_fit_costs_override_constructor_costs(make_estimator, cost_data):
    """
    The half of the ``Parameter.UNCHANGED`` contract that had no test.

    A cost passed to ``fit`` must win over the one given to ``__init__`` -- not be merged with it,
    and not be ignored.
    """
    X, y = cost_data
    overridden = make_estimator(fp_cost=1.0, fn_cost=1.0).fit(X, y, fp_cost=1.0, fn_cost=50.0)
    as_if_direct = make_estimator().fit(X, y, fp_cost=1.0, fn_cost=50.0)
    assert np.allclose(overridden.predict_proba(X), as_if_direct.predict_proba(X))

    # Compared on probabilities rather than labels: a boosted model can saturate to all-positive
    # under both cost settings, which would make a `predict` comparison silently vacuous.
    kept = make_estimator(fp_cost=1.0, fn_cost=1.0).fit(X, y)
    assert not np.allclose(overridden.predict_proba(X), kept.predict_proba(X)), (
        'the fit-time fn_cost did not override the constructor value'
    )


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_only_the_named_cost_is_overridden_at_fit_time(make_estimator, cost_data):
    """Passing one cost to ``fit`` must leave the other constructor costs untouched."""
    X, y = cost_data
    partial = make_estimator(fp_cost=3.0, fn_cost=1.0).fit(X, y, fn_cost=50.0)
    explicit = make_estimator().fit(X, y, fp_cost=3.0, fn_cost=50.0)
    assert np.allclose(partial.predict_proba(X), explicit.predict_proba(X))


# --- Cost shape validation ----------------------------------------------------------------------


@pytest.mark.parametrize('make_estimator', COST_TAKING_ESTIMATORS)
def test_per_sample_cost_of_the_wrong_length_is_rejected(make_estimator, cost_data):
    X, y = cost_data
    with pytest.raises(ValueError):
        make_estimator().fit(X, y, fp_cost=1.0, fn_cost=np.ones(len(y) - 1))


# --- RobustCSClassifier forwards costs to the wrapped estimator ---------------------------------


def test_robust_cs_all_zero_costs_warns(cost_data):
    X, y = cost_data
    model = RobustCSClassifier(estimator=CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5)))
    with pytest.warns(UserWarning, match='All costs are zero'):
        model.fit(X, y)
