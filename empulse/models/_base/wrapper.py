"""Scaffolding shared by the binary classifiers and by the meta-estimators that wrap another estimator."""

from typing import Any

from numpy.typing import NDArray
from sklearn import get_config
from sklearn.base import MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ..._common._sklearn_compat import Tags, get_tags, type_of_target


class BinaryClassifierTags:
    """Mixin for a classifier of two classes only, which scikit-learn's checks should not hold to a good score."""

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags: Tags = super().__sklearn_tags__()  # type: ignore[misc]
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags


def check_binary_target(y: Any) -> None:
    """Raise a ``ValueError`` unless *y* holds the labels of a binary classification."""
    y_type = type_of_target(y, input_name='y', raise_unknown=True)
    if y_type != 'binary':
        raise ValueError(
            f'Unknown label type: Only binary classification is supported. The type of the target is {y_type}.'
        )


class PassesXThrough(MetaEstimatorMixin):
    """
    Mixin for a meta-estimator that gives ``X`` to its ``estimator`` as it receives it.

    The wrapped estimator validates ``X``: it may be a pipeline that encodes strings, handles missing
    values or needs the feature names. So it decides what ``X`` may contain, and it knows the features
    once fitted as ``estimator_``.
    """

    estimator: Any
    estimator_: Any

    def __sklearn_tags__(self) -> Tags:
        tags: Tags = super().__sklearn_tags__()  # type: ignore[misc]
        estimator_tags = get_tags(self.estimator)
        tags.input_tags.sparse = estimator_tags.input_tags.sparse
        tags.input_tags.allow_nan = estimator_tags.input_tags.allow_nan
        return tags

    def _copy_feature_attributes(self) -> None:
        """Take ``n_features_in_`` and ``feature_names_in_`` from ``estimator_``, dropping a previous fit's."""
        for attribute in ('n_features_in_', 'feature_names_in_'):
            if hasattr(self.estimator_, attribute):
                setattr(self, attribute, getattr(self.estimator_, attribute))
            elif attribute in vars(self):
                delattr(self, attribute)


class ClassesFromEstimator:
    """Mixin for a meta-estimator whose ``classes_`` are its fitted ``estimator_``'s, or a prefit ``estimator``'s."""

    estimator: Any

    @property
    def classes_(self) -> NDArray[Any]:
        """The class labels, taken from the wrapped estimator."""
        if (estimator := getattr(self, 'estimator_', None)) is not None:
            classes: NDArray[Any] = estimator.classes_
            return classes
        try:
            check_is_fitted(self.estimator)
        except NotFittedError as error:
            raise AttributeError('The underlying estimator is not fitted yet.') from error
        prefit_classes: NDArray[Any] = self.estimator.classes_
        return prefit_classes

    @classes_.setter
    def classes_(self, value: NDArray[Any]) -> None:
        if (estimator := getattr(self, 'estimator_', None)) is None:
            raise AttributeError('The underlying estimator is not fitted yet.')
        estimator.classes_ = value


def final_estimator(estimator: Any) -> Any:
    """Return *estimator*, or for a pipeline, the estimator of its last step, through nested pipelines."""
    while isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    return estimator


def fit_parameter_name(estimator: Any, name: str) -> str:
    """
    Return the name under which *estimator*'s ``fit`` passes the parameter *name* on to its final estimator.

    Without metadata routing a pipeline takes it as its last step's ``<step>__<name>``; with it, as
    ``name``, routed to the steps that request it.
    """
    if isinstance(estimator, Pipeline) and not get_config()['enable_metadata_routing']:
        step_name, step = estimator.steps[-1]
        return f'{step_name}__{fit_parameter_name(step, name)}'
    return name
