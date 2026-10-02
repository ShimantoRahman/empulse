"""Helpers shared by the model and sampler conformance suites."""

import inspect
from collections.abc import Iterator
from typing import Any


class InvalidParameter:
    """A value no estimator parameter can legitimately take."""


def iter_invalid_params(estimator_class: type) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Yield ``(parameter_name, kwargs)`` with exactly one constructor parameter made invalid.

    Yielding one parameter at a time makes the test id name the offending parameter, so a failure says
    which parameter stopped validating.
    """
    for name in inspect.signature(estimator_class.__init__).parameters:
        if name == 'self':
            continue
        yield name, {name: InvalidParameter()}
