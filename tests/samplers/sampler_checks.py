from functools import partial

from imblearn.utils._test_common.instance_generator import _get_check_estimator_ids
from imblearn.utils.estimator_checks import (
    _maybe_mark, check_sampler_get_feature_names_out,
    check_sampler_get_feature_names_out_pandas,
    # check_samplers_2d_target,
    check_samplers_fit,
    # check_samplers_fit_resample,
    # check_samplers_list,
    # check_samplers_multiclass_ova,
    check_samplers_nan,
    check_samplers_one_label,
    # check_samplers_pandas,
    # check_samplers_pandas_sparse,
    # check_samplers_preserve_dtype,
    # check_samplers_sample_indices,
    # check_samplers_sampling_strategy_fit_resample,
    # check_samplers_sparse,
    check_samplers_string,
    check_target_type,
    parametrize_with_checks
)
from sklearn.utils import get_tags
from sklearn.base import clone
from sklearn.datasets import make_classification
import numpy as np
from collections import Counter
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_allclose

from empulse.samplers import BiasResampler, BiasRelabler, CostSensitiveSampler

ESTIMATORS = (
    BiasResampler(),
    BiasRelabler(estimator=LogisticRegression()),
    CostSensitiveSampler(),
)





def parametrize_with_checks(estimators, *, legacy=True, expected_failed_checks=None):
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
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)

    def _checks_generator(estimators, legacy, expected_failed_checks):
        for estimator in estimators:
            args = {"estimator": estimator, "legacy": legacy, "mark": "xfail"}
            if callable(expected_failed_checks):
                args["expected_failed_checks"] = expected_failed_checks(estimator)
            yield from estimator_checks_generator(**args)

    return pytest.mark.parametrize(
        "estimator, check",
        _checks_generator(estimators, legacy, expected_failed_checks),
        ids=_get_check_estimator_ids,
    )

@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)


def estimator_checks_generator(
        estimator, *, legacy=True, expected_failed_checks=None, mark=None
):
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
    if mark == "xfail":
        import pytest
    else:
        pytest = None  # type: ignore

    name = type(estimator).__name__
    for check in _yield_sampler_checks(estimator):
        check_with_name = partial(check, name)
        yield _maybe_mark(
            estimator,
            check_with_name,
            expected_failed_checks=expected_failed_checks,
            mark=mark,
            pytest=pytest,
        )


def _yield_sampler_checks(sampler):
    tags = get_tags(sampler)
    accept_sparse = tags.input_tags.sparse
    accept_dataframe = tags.input_tags.dataframe
    accept_string = tags.input_tags.string
    allow_nan = tags.input_tags.allow_nan

    yield check_target_type
    yield check_samplers_one_label
    yield check_samplers_fit
    yield check_samplers_fit_resample
    # yield check_samplers_sampling_strategy_fit_resample
    # if accept_sparse:
    #     yield check_samplers_sparse
    if accept_dataframe:
        yield check_samplers_pandas
        # yield check_samplers_pandas_sparse
    if accept_string:
        yield check_samplers_string
    if allow_nan:
        yield check_samplers_nan
    yield check_samplers_list
    # yield check_samplers_multiclass_ova
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


def check_samplers_fit_resample(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    target_stats = Counter(y)
    X_res, y_res = sampler.fit_resample(X, y)
    # if isinstance(sampler, BaseOverSampler):
    #     n_samples = max(target_stats.values())
    #     assert all(value >= n_samples for value in Counter(y_res).values())
    # elif isinstance(sampler, BaseUnderSampler):
    #     n_samples = min(target_stats.values())
    #     assert all(value == n_samples for value in Counter(y_res).values())


def check_samplers_pandas(name, sampler_orig):
    try:
        import pandas as pd
    except ImportError:
        raise pytest.SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )
    sampler = clone(sampler_orig)
    # Check that the samplers handle pandas dataframe and pandas series
    X, y = sample_dataset_generator()
    X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y_df = pd.DataFrame(y)
    y_s = pd.Series(y, name="class")

    X_res_df, y_res_s = sampler.fit_resample(X_df, y_s)
    X_res_df, y_res_df = sampler.fit_resample(X_df, y_df)
    X_res, y_res = sampler.fit_resample(X, y)

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


def check_samplers_list(name, sampler_orig):
    sampler = clone(sampler_orig)
    # Check that the can samplers handle simple lists
    X, y = sample_dataset_generator()
    X_list = X.tolist()
    y_list = y.tolist()

    X_res, y_res = sampler.fit_resample(X, y)
    X_res_list, y_res_list = sampler.fit_resample(X_list, y_list)

    assert isinstance(X_res_list, list)
    assert isinstance(y_res_list, list)

    assert_allclose(X_res, X_res_list)
    assert_allclose(y_res, y_res_list)


def check_samplers_preserve_dtype(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_res, y_res = sampler.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"


def check_samplers_sample_indices(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    sampler.fit_resample(X, y)
    tags = get_tags(sampler)
    if tags.sampler_tags.sample_indices:
        assert hasattr(sampler, "sample_indices_") is tags.sampler_tags.sample_indices
    else:
        assert not hasattr(sampler, "sample_indices_")


def check_samplers_2d_target(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()

    y = y.reshape(-1, 1)  # Make the target 2d
    sampler.fit_resample(X, y)
