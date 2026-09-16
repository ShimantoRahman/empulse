"""
Per-instance metadata routing for a ``loss``'s cost-matrix parameters.

:class:`~empulse.models.CostSensitiveClassifier` and :class:`~empulse.samplers.CostSensitiveSampler`
both accept a ``loss`` whose cost-matrix parameter names (e.g. ``clv``, ``incentive_cost``) should be
routable through ``set_{method}_request``, exactly like a builtin ``sample_weight`` -- that is what
lets :class:`~sklearn.pipeline.Pipeline` and :class:`~sklearn.model_selection.GridSearchCV` carry a
business parameter like ``clv`` through to ``fit``.

sklearn's ``RequestMethod`` descriptor (``sklearn.utils._metadata_requests``) fixes its accepted
keys once, at class-definition time, and ``_MetadataRequester.__init_subclass__`` regenerates it on
every subclass from a fixed set of sources (the method's own signature, or ``__metadata_request__*``
class attributes) -- it has no notion of a key set that depends on which *instance* you access it
through. Empulse's accepted keys depend on the ``loss`` object passed to ``__init__``, which is
per-instance, so mutating the class-level descriptor (as this package used to do, in each of
``CostSensitiveClassifier``, ``CSDecisionRuleClassifier`` and ``CostSensitiveSampler``) means the
last instance constructed silently overwrites every earlier instance's accepted keys, illustrated
here without the ``>>>`` doctest prompt so this example is not itself collected as a doctest::

    a = CSBoostClassifier(loss=empc_score)  # class keys become empc_score's symbols
    b = CSBoostClassifier(
        loss=mpcs_score
    )  # class keys become mpcs_score's symbols -- A's are gone
    a.set_fit_request(clv=True)
    # TypeError: Unexpected args: {'clv'} in fit. Accepted arguments are: {...mpcs_score's names...}

:class:`RoutesLossParameters` fixes this by making both halves of the mechanism per-instance:

- :class:`_LossAwareRequestMethod` recomputes its accepted keys from the *instance* being accessed,
  instead of serving the keys frozen in at class-definition time.
- :meth:`RoutesLossParameters.__init_subclass__` swaps a :class:`_LossAwareRequestMethod` in for
  the plain ``RequestMethod`` sklearn generates on every subclass -- it has to run after
  ``super().__init_subclass__()``, since that is what generates the descriptor being replaced, and
  it has to run on every subclass, since sklearn's own hook regenerates its descriptor there too.
- :meth:`RoutesLossParameters._get_metadata_request` registers each loss parameter name on the
  instance's own request objects as known-but-not-yet-requested, so it appears in
  ``set_{method}_request``'s accepted keys and in ``get_metadata_routing()`` without the user
  having to request it first, and without disturbing a request the user already made explicitly.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.utils._metadata_requests import MetadataRequest, RequestMethod

if TYPE_CHECKING:
    from ..metrics import BaseMetric


class _LossAwareRequestMethod(RequestMethod):
    """
    A ``set_{method}_request`` descriptor whose accepted keys come from the instance's loss.

    A plain :class:`~sklearn.utils._metadata_requests.RequestMethod` is built once with a fixed
    ``keys`` list and serves it to every instance of the owning class. This subclass instead
    recomputes ``keys`` from the accessing instance on every attribute access, by asking it for
    :meth:`RoutesLossParameters._loss_parameter_names`, and hands the result to a fresh, ordinary
    ``RequestMethod`` bound to that instance. No state is ever written back onto the class.
    """

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return super().__get__(instance, owner)
        keys = sorted(set(self.keys) | instance._loss_parameter_names())
        bound: Any = RequestMethod(self.name, keys, self.validate_keys).__get__(instance, owner)
        return bound


class RoutesLossParameters:
    """
    Mixin making a ``loss``'s cost-matrix parameter names routable, per instance.

    Mix this in ahead of ``BaseEstimator`` (directly, or via ``imblearn``'s ``BaseSampler``, which
    itself includes it in its MRO). A consumer:

    - sets :attr:`_routed_methods` to the metadata-routed method name(s) that should accept the
      loss's parameter names, e.g. ``('fit',)`` or ``('fit', 'predict')``;
    - overrides :meth:`_get_metric_loss` to return the :class:`~empulse.metrics.BaseMetric`
      currently in effect (or ``None``), if it is not simply ``self.loss``.

    Nothing here mutates a class attribute after class-definition time; every accepted key set is
    derived fresh from the accessing instance.
    """

    #: The metadata-routed method name(s) whose request should include the loss's parameters.
    _routed_methods: ClassVar[tuple[str, ...]] = ('fit',)

    def _get_metric_loss(self) -> BaseMetric | None:
        """Return the loss currently in effect, or ``None``. Overridden where ``self.loss`` isn't it."""
        loss: BaseMetric | None = getattr(self, 'loss', None)
        return loss

    def _loss_parameter_names(self) -> set[str]:
        """Return the cost-matrix parameter names of :meth:`_get_metric_loss`, or an empty set."""
        loss = self._get_metric_loss()
        return set(loss._all_symbols) if loss is not None else set()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # sklearn's own __init_subclass__ (de)generates a plain RequestMethod for every
        # set_{method}_request on every subclass definition; let it run first, then replace
        # what it just installed. Running this only once (e.g. at import time on the base class)
        # would not survive a grandchild class, whose own __init_subclass__ pass overwrites it again.
        super().__init_subclass__(**kwargs)
        for method in cls._routed_methods:
            name = f'set_{method}_request'
            descriptor = inspect.getattr_static(cls, name, None)
            if isinstance(descriptor, RequestMethod) and not isinstance(descriptor, _LossAwareRequestMethod):
                setattr(cls, name, _LossAwareRequestMethod(descriptor.name, descriptor.keys, descriptor.validate_keys))

    def _get_metadata_request(self) -> MetadataRequest:
        requests: MetadataRequest = super()._get_metadata_request()  # type: ignore[misc]
        names = self._loss_parameter_names()
        if names:
            for method in self._routed_methods:
                method_request = getattr(requests, method)
                for name in names:
                    if name not in method_request.requests:
                        method_request.add_request(param=name, alias=None)
        return requests
