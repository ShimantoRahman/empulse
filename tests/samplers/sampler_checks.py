import sys
import traceback
from collections import Counter
from functools import partial

import numpy as np
from imblearn.utils._test_common.instance_generator import _get_check_estimator_ids
# from imblearn.utils.estimator_checks import (
#     _maybe_mark,
# check_sampler_get_feature_names_out,
# check_sampler_get_feature_names_out_pandas,
# check_samplers_2d_target,
# check_samplers_fit,
# check_samplers_fit_resample,
# check_samplers_list,
# check_samplers_multiclass_ova,
# check_samplers_nan,
# check_samplers_one_label,
# check_samplers_pandas,
# check_samplers_pandas_sparse,
# check_samplers_preserve_dtype,
# check_samplers_sample_indices,
# check_samplers_sampling_strategy_fit_resample,
# check_samplers_sparse,
# check_samplers_string,
# check_target_type,
# parametrize_with_checks
# )
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import get_tags
from sklearn.utils._testing import SkipTest, assert_allclose, raises, set_random_state
from sklearn.utils.estimator_checks import _enforce_estimator_tags_X

from empulse.samplers import BiasRelabler, CostSensitiveSampler


def parametrize_with_checks_samplers(estimators, fit_params, *, legacy=True, expected_failed_checks=None):
    """Pytest specific decorator for parametrizing estimator checks.

    Checks are categorised into the following groups:

    - API checks: a set of checks to ensure API compatibility with scikit-learn.
      Refer to https://scikit-learn.org/dev/developers/develop.html a requirement of
      scikit-learn estimators.
    - legacy: a set of checks which gradually will be grouped into other categories.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

        .. versionchanged:: 0.24
           Passing a class was deprecated in version 0.23, and support for
           classes was removed in 0.24. Pass an instance instead.

        .. versionadded:: 0.24


    legacy : bool, default=True
        Whether to include legacy checks. Over time we remove checks from this category
        and move them into their specific category.

        .. versionadded:: 1.6

    expected_failed_checks : callable, default=None
        A callable that takes an estimator as input and returns a dictionary of the
        form::

            {
                "check_name": "my reason",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails. These tests will be marked as xfail if the check fails.


        .. versionadded:: 1.6

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to scikit-learn conventions.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)

    """
    import pytest

    if any(isinstance(est, type) for est in estimators):
        msg = (
            'Passing a class was deprecated in version 0.23 '
            "and isn't supported anymore from 0.24."
            'Please pass an instance instead.'
        )
        raise TypeError(msg)

    def _checks_generator(estimators, fit_params, expected_failed_checks):
        for estimator, fit_param in zip(estimators, fit_params):
            args = {'estimator': estimator, 'fit_params': fit_param, 'mark': 'xfail'}
            if callable(expected_failed_checks):
                args['expected_failed_checks'] = expected_failed_checks(estimator)
            yield from estimator_checks_generator(**args)

    return pytest.mark.parametrize(
        'estimator, check',
        _checks_generator(estimators, fit_params, expected_failed_checks),
        ids=_get_check_estimator_ids,
    )


def estimator_checks_generator(estimator, *, fit_params, expected_failed_checks=None, mark=None):
    """Iteratively yield all check callables for an estimator.

    .. versionadded:: 1.6

    Parameters
    ----------
    estimator : estimator object
        Estimator instance for which to generate checks.
    legacy : bool, default=True
        Whether to include legacy checks. Over time we remove checks from this category
        and move them into their specific category.
    expected_failed_checks : dict[str, str], default=None
        Dictionary of the form {check_name: reason} for checks that are expected to
        fail.
    mark : {"xfail", "skip"} or None, default=None
        Whether to mark the checks that are expected to fail as
        xfail(`pytest.mark.xfail`) or skip. Marking a test as "skip" is done via
        wrapping the check in a function that raises a
        :class:`~sklearn.exceptions.SkipTest` exception.

    Returns
    -------
    estimator_checks_generator : generator
        Generator that yields (estimator, check) tuples.
    """
    if mark == 'xfail':
        import pytest
    else:
        pytest = None  # noqa F841

    name = type(estimator).__name__
    for check in _yield_sampler_checks(estimator):
        check_with_name = partial(check, name, fit_params)
        # yield _maybe_mark(
        #     estimator,
        #     check_with_name,
        #     expected_failed_checks=expected_failed_checks,
        #     mark=mark,
        #     pytest=pytest,
        # )
        yield estimator, check_with_name


def _yield_sampler_checks(sampler):
    tags = get_tags(sampler)
    accept_dataframe = tags.input_tags.dataframe
    accept_string = tags.input_tags.string
    allow_nan = tags.input_tags.allow_nan

    yield check_target_type
    yield check_samplers_one_label
    yield check_samplers_fit
    yield check_samplers_fit_resample

    if accept_dataframe:
        yield check_samplers_pandas
    if accept_string:
        yield check_samplers_string
    if allow_nan:
        yield check_samplers_nan
    yield check_samplers_list
    yield check_samplers_preserve_dtype
    # we don't filter samplers based on their tag here because we want to make
    # sure that the fitted attribute does not exist if the tag is not
    # stipulated
    yield check_samplers_sample_indices
    yield check_samplers_2d_target
    yield check_sampler_get_feature_names_out
    yield check_sampler_get_feature_names_out_pandas


def sample_dataset_generator():
    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        n_informative=4,
        weights=[0.3, 0.7],
        random_state=0,
    )
    return X, y


def check_samplers_fit(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    sampler.fit_resample(X, y, **fit_params)
    assert hasattr(sampler, 'sampling_strategy_'), 'No fitted attribute sampling_strategy_'


def check_samplers_fit_resample(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    target_stats = Counter(y)
    X_res, y_res = sampler.fit_resample(X, y, **fit_params)
    if isinstance(sampler, CostSensitiveSampler) and sampler.method == 'oversampling':
        n_samples = max(target_stats.values())
        assert all(value >= n_samples for value in Counter(y_res).values())
    elif isinstance(sampler, CostSensitiveSampler) and sampler.method == 'rejection sampling':
        n_samples = min(target_stats.values())
        assert all(value <= n_samples for value in Counter(y_res).values())
    elif isinstance(sampler, BiasRelabler):
        assert np.sum(y_res) == np.sum(y)  # relabeling should be symmetric


def check_samplers_pandas(name, fit_params, sampler_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest('pandas is not installed: not checking column name consistency for pandas')
    sampler = clone(sampler_orig)
    # Check that the samplers handle pandas dataframe and pandas series
    X, y = sample_dataset_generator()
    X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y_df = pd.DataFrame(y)
    y_s = pd.Series(y, name='class')

    X_res_df, y_res_s = sampler.fit_resample(X_df, y_s, **fit_params)
    X_res_df, y_res_df = sampler.fit_resample(X_df, y_df, **fit_params)
    X_res, y_res = sampler.fit_resample(X, y, **fit_params)

    # check that we return the same type for dataframes or series types
    assert isinstance(X_res_df, pd.DataFrame)
    assert isinstance(y_res_df, pd.DataFrame)
    assert isinstance(y_res_s, pd.Series)

    assert X_df.columns.tolist() == X_res_df.columns.tolist()
    assert y_df.columns.tolist() == y_res_df.columns.tolist()
    assert y_s.name == y_res_s.name

    assert_allclose(X_res_df.to_numpy(), X_res)
    assert_allclose(y_res_df.to_numpy().ravel(), y_res)
    assert_allclose(y_res_s.to_numpy(), y_res)


def check_samplers_list(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    # Check that the can samplers handle simple lists
    X, y = sample_dataset_generator()
    X_list = X.tolist()
    y_list = y.tolist()

    X_res, y_res = sampler.fit_resample(X, y, **fit_params)
    X_res_list, y_res_list = sampler.fit_resample(X_list, y_list, **fit_params)

    assert isinstance(X_res_list, list)
    assert isinstance(y_res_list, list)

    assert_allclose(X_res, X_res_list)
    assert_allclose(y_res, y_res_list)


def check_samplers_preserve_dtype(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_res, y_res = sampler.fit_resample(X, y, **fit_params)
    assert X.dtype == X_res.dtype, 'X dtype is not preserved'
    assert y.dtype == y_res.dtype, 'y dtype is not preserved'


def check_samplers_sample_indices(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    sampler.fit_resample(X, y, **fit_params)
    tags = get_tags(sampler)
    if tags.sampler_tags.sample_indices:
        assert hasattr(sampler, 'sample_indices_') is tags.sampler_tags.sample_indices
    else:
        assert not hasattr(sampler, 'sample_indices_')


def check_samplers_2d_target(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()

    y = y.reshape(-1, 1)  # Make the target 2d
    sampler.fit_resample(X, y, **fit_params)


def check_sampler_get_feature_names_out(name, fit_params, sampler_orig):
    tags = get_tags(sampler_orig)

    two_d_array = tags.input_tags.two_d_array
    no_validation = tags.no_validation

    if not two_d_array or no_validation:
        return

    X, y = make_blobs(
        n_samples=1000,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)

    sampler = clone(sampler_orig)
    X = _enforce_estimator_tags_X(sampler, X)

    n_features = X.shape[1]
    set_random_state(sampler)

    y_ = y
    X_res, y_res = sampler.fit_resample(X, y=y_, **fit_params)
    input_features = [f'feature{i}' for i in range(n_features)]

    # input_features names is not the same length as n_features_in_
    with raises(ValueError, match='input_features should have length equal'):
        sampler.get_feature_names_out(input_features[::2])

    feature_names_out = sampler.get_feature_names_out(input_features)
    assert feature_names_out is not None
    assert isinstance(feature_names_out, np.ndarray)
    assert feature_names_out.dtype == object
    assert all(isinstance(name, str) for name in feature_names_out)

    n_features_out = X_res.shape[1]

    assert len(feature_names_out) == n_features_out, (
        f'Expected {n_features_out} feature names, got {len(feature_names_out)}'
    )


def check_sampler_get_feature_names_out_pandas(name, fit_params, sampler_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest('pandas is not installed: not checking column name consistency for pandas')

    tags = get_tags(sampler_orig)
    two_d_array = tags.input_tags.two_d_array
    no_validation = tags.no_validation

    if not two_d_array or no_validation:
        return

    X, y = make_blobs(
        n_samples=1000,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)

    sampler = clone(sampler_orig)
    X = _enforce_estimator_tags_X(sampler, X)

    n_features = X.shape[1]
    set_random_state(sampler)

    y_ = y
    feature_names_in = [f'col{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names_in)
    X_res, y_res = sampler.fit_resample(df, y=y_, **fit_params)

    # error is raised when `input_features` do not match feature_names_in
    invalid_feature_names = [f'bad{i}' for i in range(n_features)]
    with raises(ValueError, match='input_features is not equal to feature_names_in_'):
        sampler.get_feature_names_out(invalid_feature_names)

    feature_names_out_default = sampler.get_feature_names_out()
    feature_names_in_explicit_names = sampler.get_feature_names_out(feature_names_in)
    assert_array_equal(feature_names_out_default, feature_names_in_explicit_names)

    n_features_out = X_res.shape[1]

    assert len(feature_names_out_default) == n_features_out, (
        f'Expected {n_features_out} feature names, got {len(feature_names_out_default)}'
    )


def check_samplers_nan(name, fit_params, sampler_orig):
    rng = np.random.RandomState(0)
    sampler = clone(sampler_orig)
    categories = np.array([0, 1, np.nan], dtype=np.float64)
    n_samples = 100
    X = rng.randint(low=0, high=3, size=n_samples).reshape(-1, 1)
    X = categories[X]
    y = rng.permutation([0] * 40 + [1] * 60)

    X_res, y_res = sampler.fit_resample(X, y, **fit_params)
    assert X_res.dtype == np.float64
    assert X_res.shape[0] == y_res.shape[0]
    assert np.any(np.isnan(X_res.ravel()))


def check_samplers_one_label(name, fit_params, sampler_orig):
    sampler = clone(sampler_orig)
    error_string_fit = "Sampler can't balance when only one class is present."
    X = np.random.random((20, 2))
    y = np.zeros(20)
    try:
        sampler.fit_resample(X, y, **fit_params)
    except ValueError as e:
        if 'class' not in repr(e):
            print(error_string_fit, sampler.__class__.__name__, e)
            traceback.print_exc(file=sys.stdout)
            raise e
        else:
            return
    except Exception as exc:
        print(error_string_fit, traceback, exc)
        traceback.print_exc(file=sys.stdout)
        raise exc
    raise AssertionError(error_string_fit)


def check_samplers_string(name, fit_params, sampler_orig):
    rng = np.random.RandomState(0)
    sampler = clone(sampler_orig)
    categories = np.array(['A', 'B', 'C'], dtype=object)
    n_samples = 30
    X = rng.randint(low=0, high=3, size=n_samples).reshape(-1, 1)
    X = categories[X]
    y = rng.permutation([0] * 10 + [1] * 20)

    X_res, y_res = sampler.fit_resample(X, y, **fit_params)
    assert X_res.dtype == object
    assert X_res.shape[0] == y_res.shape[0]
    assert_array_equal(np.unique(X_res.ravel()), categories)


def check_target_type(name, fit_params, estimator_orig):
    estimator = clone(estimator_orig)
    # should raise warning if the target is continuous (we cannot raise error)
    X = np.random.random((20, 2))
    y = np.linspace(0, 1, 20)
    msg = 'Unknown label type:'
    with raises(ValueError, err_msg=msg):
        estimator.fit_resample(X, y, **fit_params)
    # if the target is multilabel then we should raise an error
    rng = np.random.RandomState(42)
    y = rng.randint(2, size=(20, 3))
    msg = 'Multilabel and multioutput targets are not supported.'
    with raises(ValueError, err_msg=msg):
        estimator.fit_resample(X, y, **fit_params)
