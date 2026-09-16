"""
What a :class:`~empulse.metrics.MetricStrategy` supports, so callers can ask instead of guess.

:class:`MetricStrategy` declares eleven members (see its docstring), of which only three --
:meth:`~empulse.metrics.MetricStrategy.build`, :meth:`~empulse.metrics.MetricStrategy.score` and
:meth:`~empulse.metrics.MetricStrategy.to_latex` -- are ``@abstractmethod``. The rest
(:meth:`~empulse.metrics.MetricStrategy.optimal_threshold`,
:meth:`~empulse.metrics.MetricStrategy.optimal_rate`,
:meth:`~empulse.metrics.MetricStrategy.logit_objective`,
:meth:`~empulse.metrics.MetricStrategy.gradient_boost_objective` and
:meth:`~empulse.metrics.MetricStrategy.prepare_boost_objective`) are concrete methods on the base
class whose body is ``raise NotImplementedError``, so a strategy opts in simply by overriding one.
That leaves no way for a caller to *ask* what a given strategy (or a given instance of one, since
:class:`~empulse.metrics.MaxProfit`'s support depends on the cost matrix it was built from) can do
without either calling the method and catching ``NotImplementedError``, or checking
``isinstance(strategy, SomeConcreteStrategy)`` -- which is brittle against
:class:`~empulse.metrics.MixtureMetric` and against any third-party strategy that overrides the
same methods without being a subclass of the built-in ones.

:class:`Capability` names each such question as one member of a :class:`frozenset`, so that
:attr:`~empulse.metrics.BaseMetric.capabilities` (or, for a strategy directly,
:attr:`~empulse.metrics.MetricStrategy.capabilities`) answers all of them at once, and
:class:`~empulse.metrics.MixtureMetric` can combine its components' capabilities by simple set
intersection -- which is also why every member below is phrased as *supports X*, never as
*requires X*: a "requires" member would need union across components, not intersection.
"""

from enum import Enum


class Capability(Enum):
    """
    A capability a :class:`~empulse.metrics.MetricStrategy` may or may not support.

    A plain :class:`~enum.Enum`, not a :class:`~enum.StrEnum`: nothing here relies on a member
    behaving like a ``str``, and the reference docs generate one page per class with
    ``:inherited-members:``, which would otherwise enumerate (and try to numpydoc-validate) the
    entire inherited ``str`` method surface -- ``.zfill()``, ``.translate()``, and so on.

    Attributes
    ----------
    OPTIMAL_THRESHOLD : Capability
        :meth:`~empulse.metrics.MetricStrategy.optimal_threshold` is implemented.
    OPTIMAL_RATE : Capability
        :meth:`~empulse.metrics.MetricStrategy.optimal_rate` is implemented.
    LOGIT_OBJECTIVE : Capability
        :meth:`~empulse.metrics.MetricStrategy.logit_objective` is implemented.
    BOOST_OBJECTIVE : Capability
        :meth:`~empulse.metrics.MetricStrategy.gradient_boost_objective` is implemented: the
        gradient and hessian can be recomputed directly from the metric on every boosting round.
    PRECOMPUTED_BOOST_OBJECTIVE : Capability
        :meth:`~empulse.metrics.MetricStrategy.prepare_boost_objective` is implemented: the
        gradient's constant term can be computed once, before the first boosting round, rather
        than recomputed every round. Mutually exclusive with ``BOOST_OBJECTIVE`` in practice,
        since a strategy that needs ``BOOST_OBJECTIVE``'s dynamic recomputation (its per-sample
        loss is non-linear in the predicted probability, or it needs the current round's scores
        to locate a threshold) cannot also offer a round-independent constant term.
    COST_ONLY_DECISION : Capability
        :meth:`~empulse.metrics.MetricStrategy.optimal_threshold` and
        :meth:`~empulse.metrics.MetricStrategy.optimal_rate` depend only on the cost values, not
        on the actual ``y_true``/``y_score`` -- they can be called with empty or dummy arrays,
        which is what lets a decision be recomputed at predict time from cost parameters alone
        rather than only from what was learned during ``fit``. Absent for a ranking-based
        strategy (e.g. :class:`~empulse.metrics.MaxProfit`), whose threshold is a genuine
        function of the observed scores.
    CLASS_COSTS : Capability
        The metric is reducible to four class-level scalars (``tp_benefit``, ``tn_benefit``,
        ``fp_cost``, ``fn_cost``), which is what a model needs to feed a cost-sensitive criterion
        (e.g. a decision tree's split criterion) that only accepts scalar, class-dependent costs.
    """

    OPTIMAL_THRESHOLD = 'optimal_threshold'
    OPTIMAL_RATE = 'optimal_rate'
    LOGIT_OBJECTIVE = 'logit_objective'
    BOOST_OBJECTIVE = 'boost_objective'
    PRECOMPUTED_BOOST_OBJECTIVE = 'precomputed_boost_objective'
    COST_ONLY_DECISION = 'cost_only_decision'
    CLASS_COSTS = 'class_costs'
