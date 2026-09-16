"""
Regression tests for per-instance metadata routing (``empulse._common._cost_routing``).

sklearn's ``RequestMethod`` descriptor fixes its accepted keys at class-definition time.
Empulse's accepted keys depend on the ``loss`` instance each estimator was constructed with, so
the package used to install a fresh ``RequestMethod`` on ``self.__class__`` from ``__init__`` --
which meant constructing a *second* instance of the same class with a different loss silently
overwrote the routing keys of every earlier instance. The tests below reproduce that scenario and
would fail against the old class-mutating implementation.
"""

import pytest
import sklearn.datasets
import sklearn.linear_model
import sklearn.pipeline
from sklearn import config_context
from sklearn.model_selection import cross_val_score

from empulse.metrics import Cost, CostMatrix, Metric
from empulse.models import (
    B2BoostClassifier,
    CSBoostClassifier,
    CSThresholdClassifier,
    RobustCSClassifier,
)
from empulse.samplers import CostSensitiveSampler


@pytest.fixture(autouse=True)
def _enable_metadata_routing():
    with config_context(enable_metadata_routing=True):
        yield


@pytest.fixture
def loss_a():
    cost_matrix = CostMatrix().add_fp_cost('clv').add_fn_cost('other').set_default(other=1.0)
    return Metric(cost_matrix, Cost())


@pytest.fixture
def loss_b():
    cost_matrix = CostMatrix().add_fp_cost('roi').add_fn_cost('another').set_default(another=1.0)
    return Metric(cost_matrix, Cost())


class TestRoutingIsPerInstance:
    """Constructing a second instance with a different loss must not disturb the first."""

    def test_two_instances_keep_independent_routing_keys(self, loss_a, loss_b):
        a = CSBoostClassifier(loss=loss_a)
        b = CSBoostClassifier(loss=loss_b)

        # Order matters for the regression this guards: `b` is constructed *after* `a`, which is
        # exactly what broke the class-mutating implementation.
        a.set_fit_request(clv=True)
        b.set_fit_request(roi=True)

        assert a.get_metadata_routing().fit.requests['clv'] is True
        assert b.get_metadata_routing().fit.requests['roi'] is True

    def test_instance_rejects_the_other_instances_symbols(self, loss_a, loss_b):
        a = CSBoostClassifier(loss=loss_a)
        CSBoostClassifier(loss=loss_b)  # constructing this must not affect `a` below

        with pytest.raises(TypeError, match='roi'):
            a.set_fit_request(roi=True)

    def test_constructing_a_second_instance_does_not_break_the_first(self, loss_a, loss_b):
        a = CSBoostClassifier(loss=loss_a)
        a.set_fit_request(clv=True)

        CSBoostClassifier(loss=loss_b)  # must not un-teach `a` about `clv`

        # Re-accessing the descriptor after `b` exists must still accept `clv` on `a`.
        a.set_fit_request(clv=True)
        assert a.get_metadata_routing().fit.requests['clv'] is True

    def test_routing_tracks_a_loss_swapped_in_via_set_params(self, loss_a, loss_b):
        a = CSBoostClassifier(loss=loss_a)
        a.set_params(loss=loss_b)

        a.set_fit_request(roi=True)
        with pytest.raises(TypeError, match='clv'):
            a.set_fit_request(clv=True)

    def test_business_parameter_reaches_fit_through_a_pipeline(self, loss_a):
        X, y = sklearn.datasets.make_classification(n_samples=20, n_features=4, random_state=0)
        model = CSBoostClassifier(loss=loss_a).set_fit_request(clv=True)
        pipeline = sklearn.pipeline.Pipeline([('model', model)])
        cross_val_score(pipeline, X, y, cv=2, params={'clv': 5.0})


class TestRoutingPropagatesThroughSubclassing:
    """sklearn's own ``__init_subclass__`` regenerates a plain descriptor on every subclass;
    the loss-aware replacement must survive that on every level of the hierarchy."""

    def test_grandchild_class_keeps_the_loss_aware_descriptor(self):
        import inspect

        from empulse._common._cost_routing import _LossAwareRequestMethod

        for cls in (CSBoostClassifier, B2BoostClassifier):
            descriptor = inspect.getattr_static(cls, 'set_fit_request')
            assert isinstance(descriptor, _LossAwareRequestMethod), cls

    def test_b2boost_routes_its_fixed_loss(self):
        model = B2BoostClassifier()
        model.set_fit_request(clv=True)
        assert model.get_metadata_routing().fit.requests['clv'] is True


class TestRobustCSClassifierDelegatesRouting:
    """RobustCSClassifier's loss is whatever its wrapped estimator's loss is."""

    def test_routes_the_wrapped_estimators_loss(self, loss_a):
        robust = RobustCSClassifier(estimator=CSBoostClassifier(loss=loss_a))
        robust.set_fit_request(clv=True)
        assert robust.get_metadata_routing().fit.requests['clv'] is True


class TestCSDecisionRuleClassifierRoutesFitAndPredict:
    """CSThresholdClassifier/CSRateClassifier route both `fit` and `predict`."""

    def test_routes_both_methods(self, loss_a):
        model = CSThresholdClassifier(estimator=sklearn.linear_model.LogisticRegression(), loss=loss_a)
        model.set_fit_request(clv=True)
        model.set_predict_request(clv=True)
        assert model.get_metadata_routing()._self_request.fit.requests['clv'] is True
        assert model.get_metadata_routing()._self_request.predict.requests['clv'] is True


class TestCostSensitiveSamplerRouting:
    """CostSensitiveSampler routes `fit_resample`, not `fit`."""

    def test_two_samplers_keep_independent_routing_keys(self, loss_a, loss_b):
        a = CostSensitiveSampler(loss=loss_a)
        b = CostSensitiveSampler(loss=loss_b)

        a.set_fit_resample_request(clv=True)
        b.set_fit_resample_request(roi=True)

        assert a.get_metadata_routing().fit_resample.requests['clv'] is True
        with pytest.raises(TypeError, match='roi'):
            a.set_fit_resample_request(roi=True)

    def test_no_loss_still_routes_plain_costs(self):
        sampler = CostSensitiveSampler()
        sampler.set_fit_resample_request(fp_cost=True, fn_cost=True)
        assert sampler.get_metadata_routing().fit_resample.requests['fp_cost'] is True
