"""
Per-instance metadata routing and cost resolution shared by every cost-sensitive estimator.

:class:`~empulse.models.CostSensitiveClassifier`, :class:`~empulse.models.RobustCSClassifier` and
:class:`~empulse.samplers.CostSensitiveSampler` all accept plain ``tp_cost``/``tn_cost``/``fn_cost``/
``fp_cost`` (or a subset -- the sampler only uses two) and, except for ``RobustCSClassifier``, a
``loss`` whose cost-matrix parameter names (e.g. ``clv``, ``incentive_cost``) should be routable
through ``set_{method}_request``, exactly like a builtin ``sample_weight`` -- that is what lets
:class:`~sklearn.pipeline.Pipeline` and :class:`~sklearn.model_selection.GridSearchCV` carry a
business parameter like ``clv`` through to ``fit``. :class:`RoutesLossParameters` is the shared home
for both concerns, so a fix to either lands once instead of drifting across three copies.

**Metadata routing.** sklearn's ``RequestMethod`` descriptor (``sklearn.utils._metadata_requests``)
fixes its accepted keys once, at class-definition time, and ``_MetadataRequester.__init_subclass__``
regenerates it on every subclass from a fixed set of sources (the method's own signature, or
``__metadata_request__*`` class attributes) -- it has no notion of a key set that depends on which
*instance* you access it through. Empulse's accepted keys depend on the ``loss`` object passed to
``__init__``, which is per-instance, so mutating the class-level descriptor (as this package used to
do, in each of ``CostSensitiveClassifier``, ``CSDecisionRuleClassifier`` and ``CostSensitiveSampler``)
means the last instance constructed silently overwrites every earlier instance's accepted keys,
illustrated here without the ``>>>`` doctest prompt so this example is not itself collected as a
doctest::

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

**Cost resolution.** :meth:`~RoutesLossParameters._check_costs` resolves ``Parameter.UNCHANGED`` costs
to the constructor's, warns and substitutes ``fp_cost=fn_cost=1`` when every cost is zero, and
converts array-like costs to ``numpy`` arrays. :meth:`~RoutesLossParameters._route_costs_to_loss`
forwards an explicitly passed cost argument to a ``loss`` metric when the metric's cost matrix
happens to use that name as a symbol (several bundled datasets name a symbol ``fp_cost``), and warns
rather than silently dropping it otherwise. Both read the cost names to resolve from
:attr:`_cost_names`, so :class:`~empulse.samplers.CostSensitiveSampler` (``('fp_cost', 'fn_cost')``)
and :class:`~empulse.models.CostSensitiveClassifier` (all four) share one implementation instead of
three that can drift apart. :meth:`~RoutesLossParameters._take_fit_local_loss` takes a deep copy of
``self.loss`` for the current fit: a module-level prebuilt metric such as ``empc_score`` is shared by
every caller in the process and memoizes per-call state on itself (the boosting objective, the Monte
Carlo sample grid, the RNG), so two concurrent fits through one prebuilt metric would read each
other's cache without this. :meth:`~RoutesLossParameters._get_metric_loss` prefers that per-fit copy
once one has been taken, so nothing below that point mutates a metric object the caller still holds a
reference to.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from sklearn.utils._metadata_requests import MetadataRequest, RequestMethod

from ._parameter import Parameter

if TYPE_CHECKING:
    from .._types import FloatArrayLike, FloatNDArray
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

    Also the shared home for resolving the plain ``tp_cost``/``tn_cost``/``fn_cost``/``fp_cost``
    arguments every consumer accepts alongside ``loss`` -- see :meth:`_check_costs` and
    :meth:`_route_costs_to_loss`.

    Mix this in ahead of ``BaseEstimator`` (directly, or via ``imblearn``'s ``BaseSampler``, which
    itself includes it in its MRO). A consumer:

    - sets :attr:`_routed_methods` to the metadata-routed method name(s) that should accept the
      loss's parameter names, e.g. ``('fit',)`` or ``('fit', 'predict')``;
    - sets :attr:`_cost_names` to the plain cost arguments it accepts, if not all four;
    - overrides :meth:`_get_metric_loss` to return the :class:`~empulse.metrics.BaseMetric`
      currently in effect (or ``None``), if it is not ``self._loss`` / ``self.loss``.

    Nothing here mutates a class attribute after class-definition time; every accepted key set is
    derived fresh from the accessing instance.
    """

    #: The metadata-routed method name(s) whose request should include the loss's parameters.
    _routed_methods: ClassVar[tuple[str, ...]] = ('fit',)

    #: The plain cost arguments this estimator accepts, resolved by :meth:`_check_costs` and
    #: :meth:`_route_costs_to_loss`. A subset for a consumer that does not use all four, e.g.
    #: ``('fp_cost', 'fn_cost')`` for :class:`~empulse.samplers.CostSensitiveSampler`.
    _cost_names: ClassVar[tuple[str, ...]] = ('tp_cost', 'tn_cost', 'fn_cost', 'fp_cost')

    #: Whether :meth:`_check_costs` should substitute ``fp_cost=fn_cost=1`` (with a warning) when
    #: every cost resolves to zero. ``False`` where a metric loss is the normal case and an all-zero
    #: plain cost is not a user mistake worth warning about (e.g. ``CSDecisionRuleClassifier``).
    _set_default_costs: ClassVar[bool] = True

    def _get_metric_loss(self) -> BaseMetric | None:
        """
        Return the loss currently in effect, or ``None``.

        During and after a fit that called :meth:`_take_fit_local_loss`, this is that per-fit copy,
        so nothing below this point mutates a metric object the caller still holds a reference to.
        Before the first such fit, it is the constructor argument itself. Overridden where the loss
        in effect is not ``self.loss``, e.g. :class:`~empulse.models.RobustCSClassifier` delegates
        to the estimator it wraps.
        """
        fit_local_loss: BaseMetric | None = getattr(self, '_loss', None)
        if fit_local_loss is not None:
            return fit_local_loss
        loss: BaseMetric | None = getattr(self, 'loss', None)
        return loss

    def _take_fit_local_loss(self) -> BaseMetric | None:
        """Deep-copy ``self.loss`` for the current fit and return it; see :meth:`_get_metric_loss`."""
        loss_attr = getattr(self, 'loss', None)
        fit_local_loss: BaseMetric | None = copy.deepcopy(loss_attr) if loss_attr is not None else None
        self._loss = fit_local_loss
        return fit_local_loss

    def _check_costs(
        self, *, caller: str = 'fit', **costs: FloatArrayLike | float | Parameter
    ) -> dict[str, FloatNDArray | float]:
        """
        Resolve this estimator's plain costs (named in :attr:`_cost_names`) and return them.

        Also convert them to numpy arrays if they are array-like.
        Overwrite costs set in constructor if they are set in the fit/predict method.

        Parameters
        ----------
        caller : str, default='fit'
            Name of the calling method, used in the all-zero-costs warning message.
        **costs : float or array-like or Parameter
            The cost arguments as passed by the caller, keyed by name. ``Parameter.UNCHANGED``
            (the default when a name is absent) means "not passed"; the constructor's value is used.

        Returns
        -------
        resolved : dict[str, float or FloatNDArray]
            One entry per name in :attr:`_cost_names`.
        """
        resolved: dict[str, FloatArrayLike | float | Parameter] = {}
        for name in self._cost_names:
            value = costs.get(name, Parameter.UNCHANGED)
            if value is Parameter.UNCHANGED:
                value = getattr(self, name)
            resolved[name] = value

        if self._set_default_costs and _all_costs_zero(resolved):
            warnings.warn(
                'All costs are zero. Setting fp_cost=1 and fn_cost=1. '
                f'To avoid this warning, set costs explicitly in the {self.__class__.__name__}.{caller}() method.',
                UserWarning,
                stacklevel=2,
            )
            if 'fp_cost' in resolved:
                resolved['fp_cost'] = 1
            if 'fn_cost' in resolved:
                resolved['fn_cost'] = 1

        checked: dict[str, FloatNDArray | float] = {
            name: value if isinstance(value, Real) else np.asarray(value) for name, value in resolved.items()
        }
        return checked

    def _route_costs_to_loss(
        self,
        loss: BaseMetric,
        params: dict[str, Any],
        *,
        caller: str = 'fit',
        **costs: FloatArrayLike | float | Parameter,
    ) -> dict[str, Any]:
        """
        Route explicitly passed cost arguments through to a :class:`~empulse.metrics.BaseMetric` loss.

        A cost matrix may legitimately name one of its symbols (or aliases) ``tp_cost``, ``tn_cost``,
        ``fn_cost`` or ``fp_cost`` -- several of the bundled datasets do. Those names collide with the
        caller's own keyword parameters, so the value binds there instead of landing in ``params`` and
        would otherwise never reach the metric. Any such value that names a symbol of ``loss`` is
        forwarded here; the rest are dropped with a warning rather than silently.

        Only values passed explicitly by the caller are forwarded. The ``__init__``-time cost attributes
        are deliberately *not* consulted: they describe plain costs and default to ``0.0``, so falling
        back to them would silently override a cost matrix default with zero.

        Parameters
        ----------
        loss : BaseMetric
            The metric loss the costs should be routed to.
        params : dict[str, Any]
            Loss parameters collected so far. Not mutated; an updated copy is returned.
        caller : str, default='fit'
            Name of the calling method, used in the warning message.
        **costs : float or array-like or Parameter
            The cost arguments as passed by the caller, keyed by name. ``Parameter.UNCHANGED`` means
            "not passed".

        Returns
        -------
        params : dict[str, Any]
            The loss parameters, extended with any cost argument that names a symbol of ``loss``.
        """
        params = dict(params)
        symbols = loss._all_symbols
        ignored = []
        for name, value in costs.items():
            if value is Parameter.UNCHANGED:
                continue
            if name in symbols:
                params[name] = value
            else:
                ignored.append(name)

        if ignored:
            warnings.warn(
                f'{", ".join(ignored)} passed to {self.__class__.__name__}.{caller}() '
                f'{"is" if len(ignored) == 1 else "are"} ignored because a `loss` metric is set '
                f'and the metric does not use {"that name" if len(ignored) == 1 else "those names"}. '
                f'Pass the parameters its cost matrix expects instead: {sorted(symbols)}.',
                UserWarning,
                stacklevel=3,
            )
        return params

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


def _all_float(*values: Any) -> bool:
    return all(isinstance(value, Real) and not isinstance(value, Parameter) for value in values)


def _all_costs_zero(costs: dict[str, Any]) -> bool:
    values = costs.values()
    return _all_float(*values) and sum(abs(value) for value in values) == 0.0
