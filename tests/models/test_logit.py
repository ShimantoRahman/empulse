"""Tests for the two logistic models, CSLogitClassifier and ProfLogitClassifier."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from sklearn.base import clone
from sklearn.utils.validation import NotFittedError, check_is_fitted

from empulse.metrics import CostMatrix, LogCost, MaxProfit, Metric, Savings
from empulse.models import CSLogitClassifier, ProfLogitClassifier
from empulse.optimizers import GeneticAlgorithmOptimizer, LBFGSBOptimizer


@pytest.fixture(scope='module')
def proflogit(X, y):
    clf = ProfLogitClassifier(
        tp_cost=-1, fp_cost=1, optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42)
    )
    clf.fit(X, y)
    return clf


@pytest.mark.parametrize(
    ('classifier', 'default'),
    [(CSLogitClassifier, LBFGSBOptimizer), (ProfLogitClassifier, GeneticAlgorithmOptimizer)],
    ids=['CSLogitClassifier', 'ProfLogitClassifier'],
)
class TestDefaultOptimizer:
    """`_resolve_optimizer` is implemented once on `BaseLogitClassifier`, driven by the
    `_default_optimizer` ClassVar each subclass sets.

    These check the resolved optimizer rather than fitting with it: the default genetic algorithm
    runs at least 250 generations, and `_fit_estimator` calls `_resolve_optimizer` for every fit.
    """

    def test_default_optimizer(self, classifier, default):
        assert classifier._default_optimizer is default

    def test_none_optimizer_falls_back_to_default(self, classifier, default):
        assert isinstance(classifier(optimizer=None)._resolve_optimizer(), default)

    def test_explicit_optimizer_overrides_default(self, classifier, default):
        optimizer = default(max_iter=3)
        assert classifier(optimizer=optimizer)._resolve_optimizer() is optimizer


class TestCSLogit:
    def test_works_with_different_loss(self, X, y):
        clf = CSLogitClassifier(loss=Metric(CostMatrix().add_fp_cost('fp').add_fn_cost('fn'), Savings()))
        clf.fit(X, y, fp=10, fn=1)
        assert clf.result_.x.shape == (3,)
        assert isinstance(clf.result_, OptimizeResult)
        assert clf.result_.success is True

    @pytest.mark.parametrize('l1_ratio', [1.0, 0.5])
    def test_fits_a_max_profit_loss_with_an_l1_penalty(self, X, y, l1_ratio):
        """L-BFGS-B minimizes an L1 penalty through a reformulation that needs the unpenalized MaxProfit."""
        loss = Metric(CostMatrix().add_tp_cost('tp').add_fp_cost('fp').add_fn_cost('fn'), MaxProfit())
        clf = CSLogitClassifier(loss=loss, l1_ratio=l1_ratio).fit(X, y, tp=-1.0, fp=1.0, fn=4.0)
        objective = loss._logit_objective(
            features=np.column_stack([np.ones(len(y)), X]),
            y_true=y,
            C=1.0,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            tp=-1.0,
            fp=1.0,
            fn=4.0,
        )
        weights = clf.result_.x
        value, gradient = objective.logit_loss_gradient(weights)
        data_value, data_gradient = objective.data_loss_gradient(weights)

        assert data_value + objective.penalty.value(weights) == pytest.approx(value)
        np.testing.assert_allclose(data_gradient + objective.penalty.gradient(weights), gradient)

    def test_explicit_optimizer_is_used_for_fitting(self, X, y):
        clf = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=3))
        clf.fit(X, y, fp_cost=1.0, fn_cost=1.0)
        assert clf.n_iter_ <= 3

    @pytest.mark.parametrize('fit_intercept', [True, False])
    def test_fits_read_only_data(self, X, y, fit_intercept):
        """Such as the memory-mapped arrays joblib hands to parallel workers, e.g. in GridSearchCV."""
        expected = CSLogitClassifier(fit_intercept=fit_intercept).fit(X, y, fp_cost=1.0, fn_cost=5.0).result_.x
        X = np.array(X, dtype=np.float64)
        X.flags.writeable = False
        clf = CSLogitClassifier(fit_intercept=fit_intercept).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        np.testing.assert_array_equal(clf.result_.x, expected)


class _RecordingLBFGSB(LBFGSBOptimizer):
    """Records the number of threads the objective it is given may use."""

    def __call__(self, objective, X, **kwargs):
        self.n_threads_ = objective.n_threads
        return super().__call__(objective, X, **kwargs)


class _RecordingGeneticAlgorithm(GeneticAlgorithmOptimizer):
    """Records the number of threads the objective it is given may use."""

    def __call__(self, objective, X, **kwargs):
        self.n_threads_ = objective.n_threads
        return super().__call__(objective, X, **kwargs)


class TestNJobs:
    @pytest.mark.parametrize('loss', [None, 'log_cost'], ids=['cost', 'log_cost'])
    def test_fitted_model_does_not_depend_on_n_jobs(self, make_data, loss):
        # Large enough that the loss is split over several chunks of rows.
        X, y = make_data(n_samples=3000, n_features=10)
        fn_cost = np.random.default_rng(0).uniform(1, 20, size=y.size)
        if loss == 'log_cost':
            loss = Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost'), LogCost())
        models = [
            CSLogitClassifier(loss=loss, n_jobs=n_jobs).fit(X, y, fp_cost=5.0, fn_cost=fn_cost) for n_jobs in (1, 3, -1)
        ]
        for model in models[1:]:
            np.testing.assert_array_equal(model.coef_, models[0].coef_)
            assert model.intercept_ == models[0].intercept_

    @pytest.mark.parametrize(('n_jobs', 'expected'), [(1, 1), (None, 1), (3, 3)])
    def test_objective_gets_n_jobs_threads(self, X, y, n_jobs, expected):
        optimizer = _RecordingLBFGSB()
        CSLogitClassifier(n_jobs=n_jobs, optimizer=optimizer).fit(X, y, fp_cost=1.0, fn_cost=1.0)
        assert optimizer.n_threads_ == expected

    def test_optimizer_with_threads_of_its_own_gets_one_thread_per_evaluation(self, X, y):
        optimizer = _RecordingGeneticAlgorithm(max_iter=2, population_size=10, random_state=0, n_jobs=2)
        CSLogitClassifier(n_jobs=4, optimizer=optimizer).fit(X, y, fp_cost=1.0, fn_cost=1.0)
        assert optimizer.n_threads_ == 1

    def test_rejects_zero_n_jobs(self, X, y):
        with pytest.raises(ValueError, match='n_jobs'):
            CSLogitClassifier(n_jobs=0).fit(X, y, fp_cost=1.0, fn_cost=1.0)


class TestProfLogit:
    def test_stores_its_parameters(self):
        clf = ProfLogitClassifier(tp_cost=-1, fp_cost=1, C=0.5, fit_intercept=False, l1_ratio=0.5)
        assert clf.C == 0.5
        assert clf.fit_intercept is False
        assert clf.l1_ratio == 0.5

    def test_fit(self, proflogit):
        assert isinstance(proflogit.result_, OptimizeResult)

    def test_fit_no_intercept(self, X, y):
        clf = ProfLogitClassifier(
            tp_cost=-1, fp_cost=1, fit_intercept=False, optimizer=GeneticAlgorithmOptimizer(max_iter=2, random_state=42)
        )
        clf.fit(X, y)
        try:
            check_is_fitted(clf)
        except NotFittedError:
            pytest.fail('ProfLogitClassifier is not fitted')
        assert isinstance(clf.result_, OptimizeResult)

    def test_one_variable(self, y):
        X = np.arange(10).reshape(10, 1)
        clf = ProfLogitClassifier(
            tp_cost=-1,
            fp_cost=1,
            fit_intercept=False,
            optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42),
        )
        clf.fit(X, y)
        assert clf.result_.x.shape == (1,)
        assert isinstance(clf.result_, OptimizeResult)
        assert clf.result_.message == 'Maximum number of iterations reached.'


@pytest.mark.parametrize(
    'classifier',
    [
        CSLogitClassifier(),
        ProfLogitClassifier(optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42)),
    ],
    ids=['CSLogitClassifier', 'ProfLogitClassifier'],
)
class TestIntercept:
    """The intercept is a column the model adds, whatever the values of the first feature."""

    def test_prediction_does_not_depend_on_the_other_rows(self, classifier, seeded_rng):
        X = seeded_rng.normal(size=(100, 3))
        X[:, 0] = seeded_rng.integers(0, 2, 100)
        y = (X[:, 1] > 0).astype(int)
        model = clone(classifier).fit(X, y, fp_cost=1.0, fn_cost=1.0)

        first_feature_one = X[:, 0] == 1
        expected = model.predict_proba(X)[first_feature_one]
        np.testing.assert_allclose(model.predict_proba(X[first_feature_one]), expected)
        np.testing.assert_allclose(model.predict_proba(X[first_feature_one][:1]), expected[:1])

    def test_a_constant_first_feature_is_not_taken_for_the_intercept(self, classifier, seeded_rng):
        X = seeded_rng.normal(size=(100, 3))
        X[:, 0] = 1.0
        y = (X[:, 1] > 0).astype(int)
        model = clone(classifier).fit(X, y, fp_cost=1.0, fn_cost=1.0)

        assert model.coef_.shape == (3,)
