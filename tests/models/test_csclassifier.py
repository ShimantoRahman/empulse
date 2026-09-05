import numpy as np
import pytest
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, Metric
from empulse.models import CSLogitClassifier
from empulse.optimizers import LBFGSBOptimizer


@pytest.fixture
def data():
    return make_classification(n_samples=20, n_features=4, random_state=0)


class TestNormalizeCostShapes:
    """
    `Metric._prepare_parameters` accepts a length-1 array as shorthand for "one value applied to
    every sample", but `_normalize_cost_shapes` used to reject anything whose size wasn't exactly
    `n_samples`, so a plain length-1 cost array was rejected by the model even though the metric
    itself would have accepted it.
    """

    def test_length_one_array_cost_is_accepted(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.array([2.0]), fn_cost=5.0)

    def test_full_length_array_cost_is_accepted(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.full(y.shape[0], 2.0), fn_cost=5.0)

    def test_wrong_length_array_cost_raises(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        with pytest.raises(ValueError, match='fp_cost'):
            model.fit(X, y, fp_cost=np.array([1.0, 2.0, 3.0]), fn_cost=5.0)

    def test_error_message_names_expected_lengths(self, data):
        X, y = data
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

    def test_colliding_symbol_is_routed_to_the_loss(self, data):
        X, y = data
        model = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.full(y.shape[0], 2.0), cl=5.0)

    def test_colliding_symbol_influences_training(self, data):
        """Routing must actually reach the objective, not merely avoid the TypeError."""
        X, y = data
        cheap = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=20))
        cheap.fit(X, y, fp_cost=np.full(y.shape[0], 1.0), cl=5.0)
        pricey = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=20))
        pricey.fit(X, y, fp_cost=np.full(y.shape[0], 100.0), cl=5.0)

        assert not np.allclose(cheap.predict_proba(X)[:, 1], pricey.predict_proba(X)[:, 1])

    def test_all_four_names_can_be_symbols(self, data):
        X, y = data
        cost_matrix = (
            CostMatrix()
            .add_tp_benefit('tp_cost')
            .add_tn_benefit('tn_cost')
            .add_fp_cost('fp_cost')
            .add_fn_cost('fn_cost')
        )
        model = CSLogitClassifier(loss=Metric(cost_matrix, Cost()), optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, tp_cost=1.0, tn_cost=0.0, fp_cost=2.0, fn_cost=3.0)

    def test_cost_not_used_by_the_metric_warns(self, data):
        X, y = data
        model = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=5))
        with pytest.warns(UserWarning, match='tn_cost passed to CSLogitClassifier.fit'):
            model.fit(X, y, fp_cost=2.0, cl=5.0, tn_cost=7.0)

    def test_no_warning_when_costs_are_not_passed(self, data, recwarn):
        X, y = data
        model = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=2.0, cl=5.0)

        assert not [w for w in recwarn if 'ignored because a `loss` metric is set' in str(w.message)]

    def test_init_time_costs_do_not_leak_into_the_metric(self, data):
        """
        ``__init__`` costs describe plain costs and default to 0.0, so forwarding them would
        silently zero out a cost matrix symbol of the same name.
        """
        X, y = data
        default = CSLogitClassifier(loss=self._metric(), optimizer=LBFGSBOptimizer(max_iter=20))
        default.fit(X, y, fp_cost=np.full(y.shape[0], 3.0), cl=5.0)
        explicit_zero_init = CSLogitClassifier(loss=self._metric(), fp_cost=0.0, optimizer=LBFGSBOptimizer(max_iter=20))
        explicit_zero_init.fit(X, y, fp_cost=np.full(y.shape[0], 3.0), cl=5.0)

        np.testing.assert_allclose(default.predict_proba(X)[:, 1], explicit_zero_init.predict_proba(X)[:, 1])
