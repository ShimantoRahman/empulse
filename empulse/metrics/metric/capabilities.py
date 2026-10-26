"""
What a :class:`~empulse.metrics.MetricStrategy` supports, so callers can ask instead of guess.

:class:`MetricStrategy` has three ``@abstractmethod`` members --
:meth:`~empulse.metrics.MetricStrategy.build`, :meth:`~empulse.metrics.MetricStrategy.score` and
:meth:`~empulse.metrics.MetricStrategy.to_latex`. Its optional methods
(:meth:`~empulse.metrics.MetricStrategy.optimal_threshold`,
:meth:`~empulse.metrics.MetricStrategy.optimal_rate`,
:meth:`~empulse.metrics.MetricStrategy.logit_objective`,
:meth:`~empulse.metrics.MetricStrategy.gradient_boost_objective` and
:meth:`~empulse.metrics.MetricStrategy.prepare_boost_objective`) are concrete methods on the base
class whose body is ``raise NotImplementedError``, so a strategy opts in simply by overriding one.
:meth:`~empulse.metrics.MetricStrategy.logit_value_objective` is optional too, but its base body
returns :meth:`~empulse.metrics.MetricStrategy.logit_objective`, so it works wherever that does.
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
        :meth:`~empulse.metrics.MetricStrategy.logit_objective` is implemented: a logistic regression
        can be trained on the metric by an optimizer that follows its gradient.
    LOGIT_VALUE_OBJECTIVE : Capability
        :meth:`~empulse.metrics.MetricStrategy.logit_value_objective` is implemented: a logistic
        regression can be trained on the metric by an optimizer that only compares its values, such as
        :class:`~empulse.optimizers.GeneticAlgorithmOptimizer`. Implied by ``LOGIT_OBJECTIVE``, whose
        objective has a value too.
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
        :meth:`~empulse.metrics.MetricStrategy.optimal_threshold` depends only on the cost values,
        not on ``y_true`` or ``y_score``, so it can be called with empty arrays, and
        :meth:`~empulse.metrics.MetricStrategy.optimal_rate` is the fraction of ``y_score`` at or
        above that threshold, so it needs the scores but not the labels. This is what lets a
        decision be recomputed at predict time from cost parameters alone, rather than only from
        what was learned during ``fit``. Absent for a ranking-based strategy (e.g.
        :class:`~empulse.metrics.MaxProfit`), whose threshold is a genuine function of the observed
        scores.
    CLASS_COSTS : Capability
        The metric is reducible to four class-level scalars (``tp_benefit``, ``tn_benefit``,
        ``fp_cost``, ``fn_cost``), which is what a model needs to feed a cost-sensitive criterion
        (e.g. a decision tree's split criterion) that only accepts scalar, class-dependent costs.
    EXPECTED_COST : Capability
        For given labels, the score is the expected cost of the metric's decision costs, averaged
        over the samples with ``y_score`` read as a calibrated probability, or an increasing or
        decreasing affine map of it. The decision costs are those of the cost matrix; a
        :class:`~empulse.metrics.MixtureMetric` weighs each component's by how much its score
        changes per unit of cost, which for :class:`~empulse.metrics.Savings` depends on the labels.
        A model that minimizes the expected cost directly from those (instance-dependent) costs, such
        as a tree split on cost impurity, then optimizes the metric itself. Present for
        :class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.Profit` and
        :class:`~empulse.metrics.Savings`, and for mixtures of them.
    RANKING : Capability
        The score depends on ``y_score`` only through how it ranks the samples, ties included: any
        strictly increasing transformation of the scores gives the same score. A model can then
        score its raw decision values (e.g. logits) without converting them to probabilities,
        which is both cheaper and keeps apart large values that rounding would make equal
        probabilities. Present for the ranking-based strategies (e.g.
        :class:`~empulse.metrics.MaxProfit`), absent for those that read ``y_score`` as a
        calibrated probability (e.g. :class:`~empulse.metrics.Cost`).
    """

    OPTIMAL_THRESHOLD = 'optimal_threshold'
    OPTIMAL_RATE = 'optimal_rate'
    LOGIT_OBJECTIVE = 'logit_objective'
    LOGIT_VALUE_OBJECTIVE = 'logit_value_objective'
    BOOST_OBJECTIVE = 'boost_objective'
    PRECOMPUTED_BOOST_OBJECTIVE = 'precomputed_boost_objective'
    COST_ONLY_DECISION = 'cost_only_decision'
    CLASS_COSTS = 'class_costs'
    EXPECTED_COST = 'expected_cost'
    RANKING = 'ranking'
