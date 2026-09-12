"""
Helpers shared by the model and sampler conformance suites.

``InvalidParameter`` + ``generate_invalid_params`` + ``test_invalid_params`` previously existed
twice, in ``tests/models/test_models.py`` and ``tests/samplers/test_samplers.py``, differing only in
whether the final call was ``fit`` or ``fit_resample``.
"""

import inspect
from collections.abc import Iterator
from typing import Any


class InvalidParameter:
    """A value no estimator parameter can legitimately take."""


def iter_invalid_params(estimator_class: type) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Yield ``(parameter_name, kwargs)`` with exactly one constructor parameter made invalid.

    Yielding one parameter at a time -- rather than looping inside a single test, as both copies of
    this used to -- means the test id names the offending parameter, so a failure says which one
    stopped validating instead of only that some parameter did.
    """
    for name in inspect.signature(estimator_class.__init__).parameters:
        if name == 'self':
            continue
        yield name, {name: InvalidParameter()}
