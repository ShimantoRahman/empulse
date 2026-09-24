"""
Cross-cutting cost behaviours defined by ``CostSensitiveClassifier``.

* the all-costs-zero warning and its ``fp_cost = fn_cost = 1`` fallback -- additionally suppressed
  suite-wide by ``filterwarnings`` in ``pyproject.toml``, so it needs an explicit ``pytest.warns``;
* binary-only enforcement and the recoding of ``classes_`` to 0/1;
* the ``Parameter.UNCHANGED`` contract, whose "fit-time cost overrides the constructor cost" half
  had no coverage at all;
* ``CSBoostClassifier`` honouring ``sample_weight`` on the CatBoost backend.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

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
    return CSBoostClassifier(**kw)


COST_TAKING_ESTIMATORS = [
    pytest.param(_cslogit, id='CSLogitClassifier'),
    pytest.param(_cstree, id='CSTreeClassifier'),
    pytest.param(_csboost, id='CSBoostClassifier'),
]


@pytest.fixture(scope='module')
def cost_data(request):
    from sklearn.datasets import make_classification

    return make_classification(n_samples=80, n_features=5, random_state=42)


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
    pytest.param(
        _cstree,
        id='CSTreeClassifier',
        marks=pytest.mark.xfail(
            reason=(
                'CSTreeClassifier.predict returns the internal 0/1 recoding instead of the original '
                'labels, even though its own classes_ reports them correctly. CSLogit, CSBoost, '
                'CSForest and ProfTree all round-trip the labels; only CSTree does not, which breaks '
                "scikit-learn's contract that predict returns values drawn from classes_."
            )
        ),
    ),
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


# --- CatBoost's sample_weight support ----------------------------------------------------------


def test_catboost_backend_uses_sample_weight(cost_data):
    """
    CatBoost trains on sample weights ``|gradient constant|``; a user's ``sample_weight`` multiplies them.

    Before, the backend used ``sample_weight`` to carry row indices and so rejected a user's own.
    """
    catboost = pytest.importorskip('catboost')
    X, y = cost_data

    def fit(**weights):
        model = CSBoostClassifier(
            catboost.CatBoostClassifier(n_estimators=5, depth=2, verbose=False, random_seed=0), fp_cost=1, fn_cost=1
        )
        return model.fit(X, y, **weights).predict_proba(X)

    weight = np.where(y == 1, 10.0, 1.0)
    unweighted, weighted = fit(), fit(sample_weight=weight)
    assert not np.allclose(unweighted, weighted)
    # Up-weighting the positives must raise their predicted probability.
    assert weighted[y == 1, 1].mean() > unweighted[y == 1, 1].mean()


# --- RobustCSClassifier forwards costs to the wrapped estimator ---------------------------------


def test_robust_cs_all_zero_costs_warns(cost_data):
    X, y = cost_data
    model = RobustCSClassifier(estimator=CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5)))
    with pytest.warns(UserWarning, match='All costs are zero'):
        model.fit(X, y)


def test_cs_threshold_multiclass_target_is_rejected(cost_data):
    from empulse.models import CSThresholdClassifier

    X, y = cost_data
    model = CSThresholdClassifier(estimator=LogisticRegression(max_iter=10), fp_cost=1, fn_cost=1, random_state=42)
    with pytest.raises(ValueError, match='Only binary classification is supported'):
        model.fit(X, np.arange(len(y)) % 3)
