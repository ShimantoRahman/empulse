import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression

from empulse._common._sklearn_compat import get_tags
from empulse.metrics import Cost, CostMatrix, Metric, MixtureComponent, MixtureMetric
from empulse.models import CSLogitClassifier, CSThresholdClassifier, CSTreeClassifier, RobustCSClassifier

ARRAY_COST_CASES = [
    ('fn_cost',),
    ('tp_cost', 'fp_cost'),
    ('tn_cost', 'fn_cost'),
    ('tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'),
]
COST_NAMES = ('tp_cost', 'tn_cost', 'fn_cost', 'fp_cost')


@pytest.mark.parametrize('array_costs', ARRAY_COST_CASES, ids='+'.join)
def test_fit(classification_data, seeded_rng, array_costs):
    X, y = classification_data
    costs = {name: (seeded_rng.random(100) if name in array_costs else 0.0) for name in COST_NAMES}
    tp_cost, tn_cost, fn_cost, fp_cost = (costs[name] for name in COST_NAMES)
    clf = RobustCSClassifier(CSLogitClassifier(), HuberRegressor(), detect_outliers_for='all')
    clf.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert hasattr(clf, 'estimator_')
    assert hasattr(clf, 'outlier_estimators_')
    for cost_name, original_cost in zip(
        ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=False
    ):
        if isinstance(original_cost, np.ndarray):
            assert not np.array_equal(clf.costs_[cost_name], original_cost)
        else:
            assert clf.costs_[cost_name] == original_cost


@pytest.mark.parametrize('detect_outliers_for', ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', ['tp_cost', 'fn_cost']])
def test_detect_outliers_for(classification_data, seeded_rng, detect_outliers_for):
    X, y = classification_data
    tp_cost, tn_cost, fn_cost, fp_cost = (seeded_rng.random(100) for _ in range(4))
    clf = RobustCSClassifier(CSLogitClassifier(), HuberRegressor(), detect_outliers_for=detect_outliers_for)
    clf.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert hasattr(clf, 'estimator_')
    assert hasattr(clf, 'outlier_estimators_')
    for cost_name, original_cost in zip(
        ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=False
    ):
        if isinstance(original_cost, np.ndarray) and cost_name in detect_outliers_for:
            assert not np.array_equal(clf.costs_[cost_name], original_cost)
        elif isinstance(original_cost, np.ndarray) and cost_name not in detect_outliers_for:
            assert np.array_equal(clf.costs_[cost_name], original_cost)


def test_robustcs_metric_loss(classification_data):
    X, y = classification_data

    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .alias('accept_rate', gamma)
        .alias('incentive_cost', d)
        .alias('contact_cost', f)
        .mark_outlier_sensitive(clv)
        .mark_outlier_sensitive(d)
    )
    cost_loss = Metric(cost_matrix, Cost())
    rng = np.random.default_rng(42)
    clv_val = rng.uniform(100, 200, size=X.shape[0])
    d_val = rng.uniform(0, 100, size=X.shape[0])
    model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
    model.fit(X, y, clv=clv_val, accept_rate=0.3, incentive_cost=d_val, contact_cost=1)
    assert hasattr(model, 'estimator_')
    assert hasattr(model, 'outlier_estimators_')
    assert not np.array_equal(model.costs_['clv'], clv_val)
    assert not np.array_equal(model.costs_['incentive_cost'], d_val)


def test_robustcs_mixture_metric_loss(classification_data):
    """
    A ``MixtureMetric`` loss is imputed, not rejected.

    ``loss.model.outlier_sensitive_parameters()`` answers which parameters are outlier-sensitive
    uniformly for a ``Metric`` and a ``MixtureMetric`` (here, trivially, one component), so
    ``RobustCSClassifier`` does not need to walk a single ``CostMatrix``'s sympy expressions.
    """
    X, y = classification_data

    clv = sympy.symbols('clv')
    cost_matrix = CostMatrix().add_tp_benefit(clv).add_fp_cost('b').mark_outlier_sensitive(clv)
    mixture = MixtureMetric([MixtureComponent(1.0, Metric(cost_matrix, Cost()), {})])

    rng = np.random.default_rng(42)
    clv_val = rng.uniform(100, 200, size=X.shape[0])
    model = RobustCSClassifier(CSLogitClassifier(loss=mixture))
    model.fit(X, y, clv=clv_val, b=1.0)
    assert hasattr(model, 'estimator_')
    assert hasattr(model, 'outlier_estimators_')
    assert not np.array_equal(model.costs_['clv'], clv_val)


@pytest.mark.parametrize(
    'labels',
    [np.array([-1, 1]), np.array(['no', 'yes']), np.array([2, 5])],
    ids=['minus_one_one', 'strings', 'two_five'],
)
def test_outlier_detection_does_not_depend_on_label_encoding(labels):
    """Outlier detection does not depend on the label encoding.

    Selecting the classes with ``y > 0`` and ``y == 0`` would match no sample on ``y == 0`` for labels
    other than 0/1, so the negative-class costs (``fp_cost``, ``tn_cost``) would never be cleaned;
    with labels such as 2/5 every sample would count as positive.
    """
    X, y = make_classification(n_samples=300, random_state=0)
    fp_cost = np.random.default_rng(0).random(300)
    fp_cost[:5] = 1000.0  # outliers
    fn_cost = np.random.default_rng(1).random(300)
    fn_cost[5:10] = 1000.0

    reference = RobustCSClassifier(CSLogitClassifier()).fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    model = RobustCSClassifier(CSLogitClassifier()).fit(X, labels[y], fn_cost=fn_cost, fp_cost=fp_cost)

    for cost_name in ('fp_cost', 'fn_cost'):
        assert model.outlier_estimators_[cost_name] is not None
        np.testing.assert_allclose(model.costs_[cost_name], reference.costs_[cost_name])
    # fp_cost only enters the negatives' costs, so only their outliers are imputed.
    assert model.costs_['fp_cost'][y == 0].max() < 1000.0


@pytest.mark.parametrize(
    'labels',
    [np.array([-1, 1]), np.array(['no', 'yes']), np.array([2, 5])],
    ids=['minus_one_one', 'strings', 'two_five'],
)
def test_metric_outlier_detection_does_not_depend_on_label_encoding(labels):
    """The metric-loss path selected each parameter's class the same way, with ``y > 0``/``y == 0``."""
    X, y = make_classification(n_samples=300, random_state=0)
    fn, fp = sympy.symbols('fn fp')
    # `fp` only appears in the false-positive cost, so its outliers are detected on the negatives.
    loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp).mark_outlier_sensitive(fp), Cost())
    fp_cost = np.random.default_rng(0).random(300)
    fp_cost[:5] = 1000.0

    reference = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, y, fn=1.0, fp=fp_cost)
    model = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, labels[y], fn=1.0, fp=fp_cost)

    assert 'fp' in model.outlier_estimators_
    np.testing.assert_allclose(model.costs_['fp'], reference.costs_['fp'])
    assert model.costs_['fp'][y == 0].max() < 1000.0


@pytest.mark.parametrize('spelling', ['d', 'amount', 'value'], ids=['symbol_name', 'alias', 'second_alias'])
def test_an_outlier_sensitive_parameter_is_imputed_under_either_spelling(spelling):
    """The metric accepts a parameter as its symbol name or any alias, so outlier detection must too."""
    X, y = make_classification(n_samples=300, random_state=0)
    cost_matrix = (
        CostMatrix().add_fp_cost('d').add_fn_cost('f').alias({'amount': 'd', 'value': 'd'}).mark_outlier_sensitive('d')
    )
    loss = Metric(cost_matrix, Cost())
    amount = np.random.default_rng(0).random(300)
    amount[:5] = 1000.0

    model = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, y, f=1.0, **{spelling: amount})

    assert spelling in model.outlier_estimators_
    assert model.costs_[spelling][y == 0].max() < 1000.0


def test_metric_symbols_named_like_the_plain_costs_reach_the_loss(classification_data, seeded_rng):
    """Several bundled datasets name a cost-matrix symbol ``fp_cost``, which ``fit`` also takes as a plain cost."""
    X, y = classification_data
    cost_matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('cl * lgd').set_default(lgd=0.75)
    fp_cost = seeded_rng.uniform(1, 2, size=X.shape[0])
    cl = seeded_rng.uniform(10, 100, size=X.shape[0])

    model = RobustCSClassifier(CSLogitClassifier(loss=Metric(cost_matrix, Cost()))).fit(X, y, fp_cost=fp_cost, cl=cl)
    reference = CSLogitClassifier(loss=Metric(cost_matrix, Cost())).fit(X, y, fp_cost=fp_cost, cl=cl)

    np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X))


def test_costs_set_on_the_wrapped_estimator_are_kept():
    """Costs left at their default here must not override the wrapped estimator's own."""
    X, y = make_classification(n_samples=200, random_state=0)
    fn_cost = np.random.default_rng(0).uniform(1, 10, 200)

    robust = RobustCSClassifier(CSLogitClassifier(fp_cost=5.0)).fit(X, y, fn_cost=fn_cost)

    assert robust.costs_['fp_cost'] == 5.0
    imputed_fn_cost = robust.costs_['fn_cost']
    expected = CSLogitClassifier(fp_cost=5.0).fit(X, y, fn_cost=imputed_fn_cost)
    np.testing.assert_allclose(robust.estimator_.coef_, expected.coef_)


def test_feature_names_reach_the_wrapped_estimator():
    pd = pytest.importorskip('pandas')
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    frame = pd.DataFrame(X, columns=list('abcd'))
    fn_cost = np.random.default_rng(0).uniform(1, 10, 200)

    robust = RobustCSClassifier(CSLogitClassifier()).fit(frame, y, fp_cost=1.0, fn_cost=fn_cost)

    assert list(robust.feature_names_in_) == list('abcd')
    assert list(robust.estimator_.feature_names_in_) == list('abcd')


def test_outliers_are_detected_on_the_wrapped_estimators_positive_class():
    """A wrapped decision rule's ``pos_label`` decides which rows describe the positive class."""
    X, y = make_classification(n_samples=300, random_state=0)
    fn_cost = np.random.default_rng(1).random(300)
    fn_cost[5:10] = 1000.0

    reference = RobustCSClassifier(CSThresholdClassifier(LogisticRegression())).fit(X, y, fn_cost=fn_cost, fp_cost=1)
    flipped = RobustCSClassifier(CSThresholdClassifier(LogisticRegression(), pos_label=0))
    flipped.fit(X, 1 - y, fn_cost=fn_cost, fp_cost=1)

    np.testing.assert_allclose(flipped.costs_['fn_cost'], reference.costs_['fn_cost'])


def test_a_list_valued_outlier_sensitive_parameter_is_imputed():
    X, y = make_classification(n_samples=300, random_state=0)
    loss = Metric(CostMatrix().add_fp_cost('d').add_fn_cost('f').mark_outlier_sensitive('d'), Cost())
    amount = np.random.default_rng(0).random(300)
    amount[:5] = 1000.0

    model = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, y, f=1.0, d=amount.tolist())

    assert model.costs_['d'][y == 0].max() < 1000.0


def test_outliers_of_a_one_class_cost_are_measured_among_that_class_only():
    """
    A cost of the positive rows must not be judged against the negatives' values of the same parameter.

    The negatives' values never enter the cost, and the outlier model never sees them, so their large
    residuals widened the spread and hid the positives' outliers, and the negatives got "imputed".
    """
    X, y = make_classification(n_samples=600, n_features=6, weights=[0.8], random_state=0)
    rng = np.random.default_rng(0)
    clv = np.where(y == 1, 100 + 10 * X[:, 0], rng.uniform(0, 2000, len(y)))
    planted = np.flatnonzero(y == 1)[:5]
    clv[planted] = 400.0
    cost_matrix = CostMatrix().add_fn_cost('clv').add_fp_cost(5).mark_outlier_sensitive('clv')

    model = RobustCSClassifier(CSLogitClassifier(loss=Metric(cost_matrix, Cost()))).fit(X, y, clv=clv)

    changed = np.flatnonzero(~np.isclose(model.costs_['clv'], clv))
    assert set(planted) <= set(changed)
    assert not (y[changed] == 0).any()


def test_outliers_of_a_one_class_plain_cost_are_measured_among_that_class_only():
    X, y = make_classification(n_samples=600, n_features=6, weights=[0.8], random_state=0)
    rng = np.random.default_rng(0)
    fn_cost = np.where(y == 1, 100 + 10 * X[:, 0], rng.uniform(0, 2000, len(y)))
    planted = np.flatnonzero(y == 1)[:5]
    fn_cost[planted] = 400.0

    model = RobustCSClassifier(CSLogitClassifier(), detect_outliers_for='fn_cost').fit(
        X, y, fn_cost=fn_cost, fp_cost=5.0
    )

    changed = np.flatnonzero(~np.isclose(model.costs_['fn_cost'], fn_cost))
    assert set(planted) <= set(changed)
    assert not (y[changed] == 0).any()


def test_outlier_threshold_counts_standard_deviations_of_the_residuals(seeded_rng):
    """With normal residuals, a threshold of 2.5 flags the about 1.2% of costs beyond 2.5 standard deviations."""
    n_samples = 20_000
    X = seeded_rng.normal(size=(n_samples, 2))
    y = np.tile([0, 1], n_samples // 2)
    fn_cost = 100 + 10 * X[:, 0] + seeded_rng.normal(0, 5, n_samples)

    model = RobustCSClassifier(CSLogitClassifier(), detect_outliers_for='fn_cost').fit(
        X, y, fn_cost=fn_cost, fp_cost=5.0
    )

    flagged = np.mean(~np.isclose(model.costs_['fn_cost'], fn_cost)[y == 1])
    assert flagged == pytest.approx(0.0124, abs=0.004)


def test_residuals_are_centered_before_they_are_compared_with_the_threshold():
    class Zero:
        def predict(self, X):
            return np.zeros(len(X))

    # Residuals of 50 plus or minus 1: none is an outlier once centered, however far from zero they are.
    target = 50.0 + np.tile([-1.0, 1.0], 10)
    imputed = RobustCSClassifier(CSLogitClassifier())._impute_outliers(Zero(), np.zeros((20, 1)), target)
    np.testing.assert_array_equal(imputed, target)


@pytest.mark.parametrize(
    ('outlier_estimator', 'allow_nan'),
    [(None, False), (HuberRegressor(), False), (HistGradientBoostingRegressor(), True)],
    ids=['default', 'huber', 'hist_gradient_boosting'],
)
def test_allow_nan_needs_the_outlier_estimator_to_take_missing_values(outlier_estimator, allow_nan):
    """The outlier estimator is fit on X too, so NaN in X is only fine if it takes them as well."""
    model = RobustCSClassifier(CSTreeClassifier(), outlier_estimator=outlier_estimator)
    assert get_tags(model).input_tags.allow_nan is allow_nan
    assert (
        get_tags(RobustCSClassifier(CSLogitClassifier(), outlier_estimator=outlier_estimator)).input_tags.allow_nan
        is False
    )


@pytest.mark.parametrize('routing', [False, True], ids=['no-routing', 'routing'])
def test_wraps_a_pipeline_ending_in_a_cost_sensitive_classifier(routing):
    """The loss is found on the pipeline's last step, and the costs reach it as the pipeline takes them."""
    import sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(int)
    clv = np.exp(X[:, 1]) * 100
    clv[:5] = 1e5
    cost_matrix = CostMatrix().add_fp_cost('d').add_fn_cost('clv').set_default(d=10).mark_outlier_sensitive('clv')
    with sklearn.config_context(enable_metadata_routing=routing):
        classifier = CSLogitClassifier(loss=Metric(cost_matrix, Cost()))
        if routing:
            classifier.set_fit_request(clv=True, d=True)
        direct = RobustCSClassifier(CSLogitClassifier(loss=Metric(cost_matrix, Cost()))).fit(X, y, clv=clv)
        piped = RobustCSClassifier(make_pipeline(StandardScaler(), classifier)).fit(X, y, clv=clv)

    np.testing.assert_allclose(piped.costs_['clv'], direct.costs_['clv'])
    # clv is the cost of a false negative, so only the outliers among the positive rows are imputed.
    planted = np.arange(5)
    assert np.all((piped.costs_['clv'][planted] < 1e5) == (y[planted] == 1))
    assert piped.n_features_in_ == 3
    np.testing.assert_array_equal(piped.classes_, [0, 1])
