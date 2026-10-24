import ast
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple, Self

import numpy as np
import sympy
import sympy.stats
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction

from ._compile import _safe_lambdify, _safe_run_lambda
from ._symbolic import _latex, _sympify_term


def _tightest(new: float | None, existing: float | None, tighter: Callable[[float, float], float]) -> float | None:
    """Combine two optional bounds, keeping whichever is more restrictive."""
    if new is None:
        return existing
    if existing is None:
        return new
    return tighter(new, existing)


@dataclass(frozen=True)
class ParameterBounds:
    """An inclusive numeric range a cost-matrix parameter must lie within."""

    lower: float | None = None
    upper: float | None = None


def _check_bounds(target: Any, lower: float | None, upper: float | None) -> ParameterBounds:
    """Return the bounds *lower* and *upper* on *target*, raising if they do not describe an interval."""
    if lower is None and upper is None:
        raise ValueError(f"Constraining '{target}' requires a lower bound, an upper bound, or both.")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"Lower bound {lower} is greater than upper bound {upper} for '{target}'.")
    return ParameterBounds(lower=lower, upper=upper)


def _add_bounds(bounds: dict[str, ParameterBounds], name: str, new: ParameterBounds) -> None:
    """Add *new* to the bounds of *name*, keeping the tighter of each bound if it already has some."""
    existing = bounds.get(name)
    if existing is not None:
        new = ParameterBounds(
            lower=_tightest(new.lower, existing.lower, max), upper=_tightest(new.upper, existing.upper, min)
        )
    bounds[name] = new


@dataclass(frozen=True)
class ParameterPredicate:
    """An arbitrary condition over several cost-matrix parameters at once."""

    predicate: Callable[[Mapping[str, Any]], bool]
    message: str


#: Why a condition written as an expression takes no bounds, shared by every ``constrain``.
_CONDITION_TAKES_NO_BOUNDS = 'lower and upper bound a single parameter; a condition such as "clv > d" takes neither.'


class _ExpressionConstraint:
    """
    A condition on the parameters written as a sympy relation, such as ``clv > d``.

    Unlike a lambda, it can be pickled: it is stored as its expression and compiled again when loaded.
    A condition on a parameter that was not passed, such as a variable that this metric draws rather
    than takes, is not checked, as a bound on it is not.
    """

    def __init__(self, expression: sympy.Basic) -> None:
        self.expression = expression
        self.names = frozenset(str(symbol) for symbol in expression.free_symbols)

    @cached_property
    def _function(self) -> Callable[..., Any]:
        return _safe_lambdify(self.expression)  # type: ignore[arg-type]

    def __call__(self, parameters: Mapping[str, Any]) -> Any:
        if not all(name in parameters for name in self.names):
            return True
        # Broadcast first: a condition joined with And reduces its parts together, which fails when one
        # part is per sample and another is not.
        names = sorted(self.names)
        try:
            values = np.broadcast_arrays(*(np.asarray(parameters[name]) for name in names))
        except ValueError:
            return True  # arrays of different lengths, which the metric reports with their names
        return _safe_run_lambda(self._function, self.expression, **dict(zip(names, values, strict=True)))

    def renamed(self, rename: Callable[[str], str]) -> '_ExpressionConstraint':
        """Return the condition with each parameter renamed, e.g. from an alias to its symbol."""
        return _ExpressionConstraint(
            self.expression.xreplace({
                symbol: sympy.Symbol(rename(str(symbol))) for symbol in self.expression.free_symbols
            })
        )


def _constraint_expression(target: Any) -> sympy.Basic | None:
    """
    Return *target* as a condition on the parameters, such as ``clv > d``, or ``None`` if it is not one.

    A string is a condition when it is a comparison; any other string names the parameter to bound.

    Raises
    ------
    ValueError
        If *target* is a string that is not one comparison by ``<``, ``<=``, ``>`` or ``>=`` but
        contains one, or a condition that does not depend on any parameter.
    """
    if isinstance(target, str):
        try:
            tree = ast.parse(target, mode='eval')
        except SyntaxError:
            return None
        if not isinstance(tree.body, ast.Compare):
            if any(isinstance(node, ast.Compare) for node in ast.walk(tree)):
                raise ValueError(
                    f'Cannot read {target!r} as a constraint: a string is a single comparison. Join several '
                    'with sympy.And or sympy.Or, or pass each to its own constrain call.'
                )
            return None
        if len(tree.body.ops) > 1:
            raise ValueError(
                f'Cannot read {target!r} as a constraint: write a chained comparison as one constraint per '
                'comparison, or bound a single parameter with constrain(name, lower, upper).'
            )
        if any(isinstance(op, ast.Eq | ast.NotEq) for op in tree.body.ops):
            raise ValueError(
                f'Cannot read {target!r} as a constraint: a string compares with <, <=, > or >=. Pass '
                'sympy.Eq or sympy.Ne for equality.'
            )
        target = _sympify_term(target)
    if target is sympy.true or target is sympy.false:  # sympy evaluates a condition such as 1 > 0 to True
        raise ValueError(f'The constraint does not depend on any parameter: sympy evaluates it to {target}.')
    if isinstance(target, Relational | BooleanFunction):
        return target
    return None


class CostMatrix:
    """
    Class to create a custom value/cost-sensitive cost matrix.

    You add the costs and benefits that make up the cost matrix for each case
    (true positive, true negative, false positive, false negative).
    The costs and benefits are specified using sympy symbols or expressions.
    Stochastic variables are supported and can be specified using sympy.stats random variables.
    Stochastic variables are assumed to be independent of each other.

    Read more in the :ref:`User Guide <cost_matrix>`.

    Attributes
    ----------
    tp_benefit : sympy.Expr
        The benefit of a true positive.
        See :meth:`~empulse.metrics.CostMatrix.add_tp_benefit` for more details.

    tn_benefit : sympy.Expr
        The benefit of a true negative.
        See :meth:`~empulse.metrics.CostMatrix.add_tn_benefit` for more details.

    fp_benefit : sympy.Expr
        The benefit of a false positive.
        See :meth:`~empulse.metrics.CostMatrix.add_fp_benefit` for more details.

    fn_benefit : sympy.Expr
        The benefit of a false negative.
        See :meth:`~empulse.metrics.CostMatrix.add_fn_benefit` for more details.

    tp_cost : sympy.Expr
        The cost of a true positive.
        See :meth:`~empulse.metrics.CostMatrix.add_tp_cost` for more details.

    tn_cost : sympy.Expr
        The cost of a true negative.
        See :meth:`~empulse.metrics.CostMatrix.add_tn_cost` for more details.

    fp_cost : sympy.Expr
        The cost of a false positive.
        See :meth:`~empulse.metrics.CostMatrix.add_fp_cost` for more details.

    fn_cost : sympy.Expr
        The cost of a false negative.
        See :meth:`~empulse.metrics.CostMatrix.add_fn_cost` for more details.

    Examples
    --------
    Reimplementing the :func:`~empulse.metrics.empc_score` cost matrix.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import CostMatrix

        clv, d, f, alpha, beta = sp.symbols(
            'clv d f alpha beta'
        )  # define deterministic variables
        gamma = sp.stats.Beta('gamma', alpha, beta)  # define gamma to follow a Beta distribution

        cost_matrix = (
            CostMatrix()
            .add_tp_benefit(gamma * (clv - d - f))  # when churner accepts offer
            .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
            .add_fp_cost(d + f)  # when you send an offer to a non-churner
            .alias({'incentive_cost': 'd', 'contact_cost': 'f'})
        )
    """

    def __init__(self) -> None:
        self._tp_benefit: sympy.Expr = sympy.core.numbers.Zero()
        self._tn_benefit: sympy.Expr = sympy.core.numbers.Zero()
        self._fp_cost: sympy.Expr = sympy.core.numbers.Zero()
        self._fn_cost: sympy.Expr = sympy.core.numbers.Zero()
        self._aliases: MutableMapping[str, str | sympy.Symbol] = {}
        self._defaults: dict[str, Any] = {}
        self._outlier_sensitive_symbols: set[sympy.Symbol] = set()
        self._bounds: dict[str, ParameterBounds] = {}
        self._predicates: list[ParameterPredicate] = []

    @property
    def tp_benefit(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return self._tp_benefit

    @property
    def tn_benefit(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return self._tn_benefit

    @property
    def fp_benefit(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return -self._fp_cost

    @property
    def fn_benefit(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return -self._fn_cost

    @property
    def tp_cost(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return -self._tp_benefit

    @property
    def tn_cost(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return -self._tn_benefit

    @property
    def fp_cost(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return self._fp_cost

    @property
    def fn_cost(self) -> sympy.Expr:  # ruff: ignore[undocumented-public-method]
        return self._fn_cost

    def add_tp_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a true positive.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the benefit of classifying a true positive.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._tp_benefit += term
        return self

    def add_tn_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a true negative.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the benefit of classifying a true negative.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._tn_benefit += term
        return self

    def add_fp_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a false positive.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the benefit of classifying a false positive.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._fp_cost -= term
        return self

    def add_fn_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a false negative.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the benefit of classifying a false negative.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._fn_cost -= term
        return self

    def add_tp_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a true positive.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the cost of classifying a true positive.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._tp_benefit -= term
        return self

    def add_tn_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a true negative.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the cost of classifying a true negative.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._tn_benefit -= term
        return self

    def add_fp_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a false positive.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the cost of classifying a false positive.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._fp_cost += term
        return self

    def add_fn_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a false negative.

        Parameters
        ----------
        term : sympy.Expr | str
            The term to add to the cost of classifying a false negative.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.
        """
        term = _sympify_term(term)
        self._fn_cost += term
        return self

    def alias(self, alias: str | MutableMapping[str, sympy.Symbol | str], symbol: sympy.Symbol | None = None) -> Self:
        """
        Add an alias for a symbol.

        Parameters
        ----------
        alias : str | MutableMapping[str, sympy.Symbol | str]
            The alias to add. If a MutableMapping (e.g., dictionary) is passed,
            the keys are the aliases and the values are the symbols.
        symbol : sympy.Symbol, optional
            The symbol to alias to. Required unless ``alias`` is a mapping.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.

        Raises
        ------
        TypeError
            If a mapping value is not a ``str`` or :class:`sympy.Symbol`.
        ValueError
            If neither a mapping nor both an alias and a symbol are given.

        Notes
        -----
        A symbol may have several aliases. An alias may not be named after another symbol of the cost
        matrix, which could then not be passed under its own name; building a
        :class:`~empulse.metrics.Metric` from such a cost matrix raises a :class:`ValueError`.

        Examples
        --------
        .. code-block:: python

            import sympy as sp
            from empulse.metrics import CostMatrix, Metric, Cost

            clv, delta, f, gamma = sp.symbols('clv delta f gamma')
            cost_matrix = (
                CostMatrix()
                .add_tp_benefit(gamma * (clv - delta * clv - f))  # when churner accepts offer
                .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
                .add_fp_cost(delta * clv + f)  # when you send an offer to a non-churner
                .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
            )
            cost_loss = Metric(cost_matrix, Cost())

            y_true = [1, 0, 1, 0, 1]
            y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
            cost_loss(
                y_true, y_proba, clv=100, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
            )
        """
        if isinstance(alias, MutableMapping):
            for key, value in alias.items():
                if not isinstance(value, str | sympy.Symbol):
                    raise TypeError(
                        f'Alias values must be str or sympy.Symbol, got {type(value).__name__!r} for key {key!r}.'
                    )
            self._aliases.update(alias)
        elif symbol is not None:
            self._aliases[alias] = str(symbol)
        else:
            raise ValueError('Either a dictionary or both an alias and a symbol should be provided')
        return self

    def set_default(self, **defaults: float) -> Self:
        """
        Set default values for symbols or their aliases.

        Parameters
        ----------
        **defaults : float
            Default values for symbols or their aliases.
            These default values will be used if not provided in __call__.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.

        Notes
        -----
        A default may name a symbol or an alias, whether that alias is registered before or after
        this call. When several calls set the same parameter, under either spelling, the last wins.

        A stochastic variable's values are drawn rather than passed, so it takes no default: set
        defaults for the parameters of its distribution instead. Building a
        :class:`~empulse.metrics.Metric` with a default on a stochastic variable, or on a name that is
        no symbol or alias, raises a :class:`ValueError`.

        Examples
        --------
        .. code-block:: python

            import sympy as sp
            from empulse.metrics import CostMatrix, Metric, Cost

            clv, delta, f, gamma = sp.symbols('clv delta f gamma')
            cost_matrix = (
                CostMatrix()
                .add_tp_benefit(gamma * (clv - delta * clv - f))  # when churner accepts offer
                .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
                .add_fp_cost(delta * clv + f)  # when you send an offer to a non-churner
                .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
                .set_default(incentive_fraction=0.05, contact_cost=1, accept_rate=0.3)
            )
            cost_loss = Metric(cost_matrix, Cost())

            y_true = [1, 0, 1, 0, 1]
            y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
            cost_loss(y_true, y_proba, clv=100, incentive_fraction=0.1)
        """
        for key, value in defaults.items():
            # Moved to the end, so that resolving aliases later keeps the last value set.
            self._defaults.pop(key, None)
            self._defaults[key] = value
        return self

    def mark_outlier_sensitive(self, symbol: str | sympy.Symbol) -> Self:
        """
        Mark a symbol as outlier-sensitive.

        This is used to indicate that the symbol is sensitive to outliers.
        When the metric is used as a loss function or criterion for training a model,
        :class:`~empulse.models.RobustCSClassifier` will impute outliers for this symbol's value.
        This is ignored when not using a :class:`~empulse.models.RobustCSClassifier` model.

        Parameters
        ----------
        symbol : str | sympy.Symbol
            The symbol, or its alias, to mark as outlier-sensitive.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.

        Raises
        ------
        TypeError
            If ``symbol`` is not a ``str`` or :class:`sympy.Symbol`.

        Notes
        -----
        Building a :class:`~empulse.metrics.Metric` raises a :class:`ValueError` if the marked name is
        no symbol or alias of the cost matrix, or is a stochastic variable, whose values are drawn
        rather than passed.

        Examples
        --------
        .. code-block:: python

            import numpy as np
            import sympy as sp
            from empulse.metrics import CostMatrix, Metric, Cost
            from empulse.models import CSLogitClassifier, RobustCSClassifier
            from sklearn.datasets import make_classification

            X, y = make_classification()
            a, b = sp.symbols('a b')
            cost_matrix = CostMatrix().add_fp_cost(a).add_fn_cost(b).mark_outlier_sensitive(a)
            cost_loss = Metric(cost_matrix, Cost())

            model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
            model.fit(X, y, a=np.random.rand(y.size), b=5)
        """
        if isinstance(symbol, str):
            symbol = sympy.Symbol(symbol)
        if not isinstance(symbol, sympy.Symbol):
            raise TypeError('The symbol must be a sympy.Symbol or a string that can be converted to a sympy.Symbol')
        self._outlier_sensitive_symbols.add(symbol)
        return self

    def constrain(
        self,
        target: str | sympy.Basic | Callable[[Mapping[str, Any]], bool],
        lower: float | None = None,
        upper: float | None = None,
        *,
        message: str | None = None,
    ) -> Self:
        """
        Restrict the values a parameter is allowed to take.

        Constraints are checked when the metric is called, and when a model fitting on this metric
        first receives its parameters. A violation raises a :class:`ValueError`.

        Three forms are supported. Passing a symbol (or alias) with `lower` and/or `upper` bounds the
        values of that one parameter. Passing a condition, such as ``'clv > incentive_cost'``,
        relates several parameters at once. Passing a callable expresses any other condition.

        Parameters
        ----------
        target : str, sympy.Symbol, sympy relation or callable
            What to constrain:

            - The name of a symbol or alias, or the :class:`sympy.Symbol`, to bound with `lower` and
              `upper`.
            - A condition the parameters must satisfy: a comparison such as ``'clv > d'``, written as a
              string or as a sympy relation, or several joined with :func:`sympy.And`,
              :func:`sympy.Or` or :func:`sympy.Not`. It may name symbols and aliases.
            - A callable taking the mapping of resolved parameter values and returning whether they
              are acceptable.

        lower : float, optional
            Smallest allowed value, inclusive. Only used when `target` names a symbol or alias.

        upper : float, optional
            Largest allowed value, inclusive. Only used when `target` names a symbol or alias.

        message : str, optional
            Explanation to report when a condition or callable `target` rejects the parameters.
            Required when `target` is a callable; a condition reports itself by default.

        Returns
        -------
        self : CostMatrix
            The cost matrix with the constraint added.

        Raises
        ------
        TypeError
            If `target` is neither a str, a :class:`sympy.Symbol`, nor a callable.

        ValueError
            If `target` is a symbol and neither `lower` nor `upper` is given,
            if `lower` is greater than `upper`,
            if `target` is a condition and `lower` or `upper` is given, does not depend on any
            parameter, or is a string other than one comparison by ``<``, ``<=``, ``>`` or ``>=``
            (such as ``'0 < a < 1'`` or ``'a > b and b > 0'``),
            or if `target` is a callable and `message` is not given.

        Notes
        -----
        A parameter may be bounded by its symbol or by an alias, whether that alias is registered
        before or after this call. Bounding the same parameter twice keeps the tighter of each bound.

        A condition holds for an instance-dependent parameter only if it holds for every sample. A
        condition on a name that is no symbol or alias raises when the metric is built; one on a
        stochastic variable, whose values are drawn rather than passed, is not checked.

        A callable receives the parameters keyed by **symbol name**, with aliases already resolved
        and defaults already applied. An instance-dependent parameter arrives as an array, so the
        callable may return one boolean per sample; the parameters are accepted only if all are true.

        Prefer a condition to a callable where one will do. A lambda, or a function defined inside
        another function, cannot be pickled, so neither can a metric built on it, nor a model fitted
        with that metric, for instance with :func:`joblib.dump`. A condition pickles like the rest of
        the cost matrix, and so does a callable defined at the top level of a module.

        Distribution parameters are validated automatically and do not need a constraint: the
        shape of a ``sympy.stats`` random variable is checked by the distribution itself, so
        ``alpha=-1`` on a Beta-distributed term is rejected without any declaration here.

        Examples
        --------
        Bound a probability to the unit interval:

        .. code-block:: python

            import sympy as sp
            from empulse.metrics import CostMatrix, Metric, MaxProfit

            clv, d, f, gamma = sp.symbols('clv d f gamma')
            cost_matrix = (
                CostMatrix()
                .add_tp_benefit(gamma * (clv - d - f))
                .add_fp_cost(d + f)
                .alias('accept_rate', gamma)
                .constrain('accept_rate', 0, 1)
            )
            metric = Metric(cost_matrix, MaxProfit())

        Express a condition spanning several parameters:

        .. code-block:: python

            import sympy as sp
            from empulse.metrics import CostMatrix, Metric, MaxProfit

            clv, d, f, gamma = sp.symbols('clv d f gamma')
            cost_matrix = (
                CostMatrix()
                .add_tp_benefit(gamma * (clv - d - f))
                .add_fp_cost(d + f)
                .alias({'incentive_cost': 'd'})
                .constrain('clv > incentive_cost', message='clv must exceed the incentive cost')
            )
            metric = Metric(cost_matrix, MaxProfit())
        """
        expression = _constraint_expression(target)
        if expression is not None:
            if lower is not None or upper is not None:
                raise ValueError(_CONDITION_TAKES_NO_BOUNDS)
            self._predicates.append(
                ParameterPredicate(predicate=_ExpressionConstraint(expression), message=message or str(expression))
            )
            return self

        if callable(target) and not isinstance(target, sympy.Symbol):
            if message is None:
                raise ValueError('A message is required when constraining with a callable.')
            self._predicates.append(ParameterPredicate(predicate=target, message=message))
            return self

        if isinstance(target, str):
            target = sympy.Symbol(target)
        # A sympy.stats random variable is a RandomSymbol, not a Symbol. Accept it so one builder can
        # constrain a symbol that is stochastic in one metric and deterministic in another; the
        # constraint has no effect on a variable that is drawn instead of supplied.
        if not isinstance(target, sympy.Symbol | sympy.stats.rv.RandomSymbol):
            raise TypeError(
                'The target must be a sympy.Symbol, a string that can be converted to one, a condition such '
                f"as 'clv > d', or a callable, got {type(target).__name__!r} instead."
            )
        _add_bounds(self._bounds, str(target), _check_bounds(target, lower, upper))
        return self

    def _resolve_aliases(self) -> None:
        """Key every default, bound, condition and outlier marker by the symbol it is for, not the alias it may name."""
        defaults, bounds = self._defaults, self._bounds
        self._defaults, self._bounds = {}, {}
        for name, value in defaults.items():
            self._defaults[self._symbol_name(name)] = value
        for name, bound in bounds.items():
            _add_bounds(self._bounds, self._symbol_name(name), bound)
        self._predicates = [
            ParameterPredicate(predicate=constraint.predicate.renamed(self._symbol_name), message=constraint.message)
            if isinstance(constraint.predicate, _ExpressionConstraint)
            else constraint
            for constraint in self._predicates
        ]
        self._outlier_sensitive_symbols = {
            sympy.Symbol(self._symbol_name(str(symbol))) for symbol in self._outlier_sensitive_symbols
        }

    def _constraint_names(self) -> set[str]:
        """Return the parameter names the conditions written as expressions read."""
        return set().union(
            *(
                constraint.predicate.names
                for constraint in self._predicates
                if isinstance(constraint.predicate, _ExpressionConstraint)
            )
        )

    def _symbol_name(self, name: str) -> str:
        """Return the symbol name a caller's keyword *name* stands for: the alias's target, or *name* itself."""
        return str(self._aliases.get(name, name))

    def _spellings(self, name: str) -> set[str]:
        """Return every keyword the parameter *name* (a symbol name or alias) can be passed as."""
        symbol_name = self._symbol_name(name)
        return {symbol_name} | {alias for alias, target in self._aliases.items() if str(target) == symbol_name}

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'tp_cost={self.tp_cost}, tn_cost={self.tn_cost}, '
            f'fp_cost={self.fp_cost}, fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self) -> str:
        """Render the cost matrix as a LaTeX array, for Jupyter and other rich-display frontends."""
        tn, fn = _latex(self.tn_cost), _latex(self.fn_cost)
        fp, tp = _latex(self.fp_cost), _latex(self.tp_cost)
        return (
            r'$\displaystyle \begin{array}{c|cc}'
            r' & y=0 & y=1 \\ \hline '
            rf'\hat y=0 & {tn} & {fn} \\ '
            rf'\hat y=1 & {fp} & {tp} '
            r'\end{array}$'
        )


# Names the metric machinery injects into the lambdified expressions (labels, scores, rates, class
# priors) or uses when rendering LaTeX. A user symbol or alias with one of these names would fuse
# with the internal one or raise a confusing TypeError. 'n_samples' and 'validate' are also
# keyword-only parameters of Metric._prepare_parameters() and the public scoring methods, which
# would capture a symbol of that name before it reached **kwargs.
# See _check_reserved_symbol_names().
RESERVED_SYMBOL_NAMES = frozenset({'y', 's', 'F_0', 'F_1', 'pi_0', 'pi_1', 'N', 'i', 'n_samples', 'validate'})


def _check_reserved_symbol_names(*expressions: sympy.Expr, alias_names: Iterable[str] = ()) -> None:
    """
    Raise if a cost matrix uses a symbol name (or alias) reserved for internal use.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) to check for reserved free symbol names.
    alias_names : Iterable[str], default=()
        The alias names registered on the cost matrix, checked alongside the expressions' own
        free symbol names.

    Raises
    ------
    ValueError
        If any free symbol name or alias collides with a name in :data:`RESERVED_SYMBOL_NAMES`.
    """
    symbol_names = {str(symbol) for symbol in _free_symbols(*expressions)}
    reserved_in_use = (symbol_names | set(alias_names)) & RESERVED_SYMBOL_NAMES
    if reserved_in_use:
        raise ValueError(
            f'The cost matrix uses symbol name(s) or alias(es) {sorted(reserved_in_use)} that are reserved '
            f'for internal use by Metric and its strategies. Reserved names: {sorted(RESERVED_SYMBOL_NAMES)}. '
            'Please rename the corresponding symbol(s) or alias(es).'
        )


def _declared_assumptions(symbol: sympy.Symbol) -> dict[str, Any]:
    """Return the assumptions a symbol was declared with, minus the implicit ``commutative``."""
    declared = dict(symbol._assumptions.generator)
    declared.pop('commutative', None)
    return declared


def _describe_symbol(symbol: sympy.Symbol) -> str:
    """Spell a symbol the way the caller would have constructed it, for use in error messages."""
    declared = _declared_assumptions(symbol)
    if not declared:
        return f'sympy.Symbol({str(symbol)!r})'
    keywords = ', '.join(f'{key}={value!r}' for key, value in sorted(declared.items()))
    return f'sympy.Symbol({str(symbol)!r}, {keywords})'


def _check_duplicate_symbol_names(*expressions: sympy.Expr) -> None:
    """
    Raise if two distinct symbols in the cost matrix share a name.

    ``sympy.Symbol('clv')`` and ``sympy.Symbol('clv', positive=True)`` are different objects that
    are not equal and do not cancel, yet both print as ``clv``. Mixing them is easy, because a term
    given as a string always produces assumption-free symbols. Left alone, the pair survives all
    the way into ``sympy.lambdify``, which emits a function with two parameters of the same name
    and fails with ``SyntaxError: duplicate argument 'clv' in function definition`` pointing at
    generated source that the caller has no way to connect back to their cost matrix.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions to check.

    Raises
    ------
    ValueError
        If any two distinct symbols share a name.
    """
    by_name: dict[str, set[sympy.Symbol]] = {}
    for symbol in _free_symbols(*expressions):
        by_name.setdefault(str(symbol), set()).add(symbol)

    duplicates = {name: symbols for name, symbols in by_name.items() if len(symbols) > 1}
    if not duplicates:
        return

    details = [
        ' and '.join(
            _describe_symbol(symbol)
            for symbol in sorted(symbols, key=lambda symbol: str(_declared_assumptions(symbol)))
        )
        for _, symbols in sorted(duplicates.items())
    ]
    raise ValueError(
        'The cost matrix contains distinct symbols that share a name, which sympy treats as '
        f'different variables: {"; ".join(details)}. This usually happens when the same name is '
        'introduced both as a string term (which creates a symbol with no assumptions) and as an '
        'explicitly constructed sympy.Symbol carrying assumptions. Use one spelling throughout.'
    )


class SymbolNames(NamedTuple):
    """The names of the symbols a cost matrix's expressions are built from, by kind."""

    #: Symbols that are not random variables, such as ``clv``.
    deterministic: frozenset[str]
    #: The random variables themselves, e.g. ``gamma`` for ``sympy.stats.Beta('gamma', alpha, beta)``.
    random: frozenset[str]
    #: The symbols in the random variables' distribution parameters, e.g. ``alpha`` and ``beta`` above.
    distribution: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        """Every name, of whatever kind."""
        return self.deterministic | self.random | self.distribution

    @property
    def supplied(self) -> frozenset[str]:
        """The names a caller gives values for: every name but the random variables'."""
        return (self.deterministic | self.distribution) - self.random


def _distribution_symbols(random_symbols: Iterable[sympy.Expr]) -> set[sympy.Expr]:
    """Return the free symbols of the parameters of *random_symbols*' distributions, e.g. ``alpha``."""
    distribution_symbols: set[sympy.Expr] = set()
    for symbol in random_symbols:
        distribution = getattr(symbol.pspace, 'distribution', None)
        for argument in getattr(distribution, 'args', ()):
            distribution_symbols |= sympy.sympify(argument).free_symbols
    return distribution_symbols


def _free_symbols(*expressions: sympy.Expr) -> set[sympy.Expr]:
    """
    Return the free symbols of *expressions*, including those of their random variables' distributions.

    ``expression.free_symbols`` stops at a random variable, so ``alpha`` in ``Beta('gamma', alpha, 2)``
    is not among them, although a caller passes it like any other parameter.
    """
    free_symbols: set[sympy.Expr] = set().union(*(expression.free_symbols for expression in expressions))
    random_symbols = {symbol for symbol in free_symbols if sympy.stats.rv.is_random(symbol)}
    return free_symbols | _distribution_symbols(random_symbols)


def _symbol_names(*expressions: sympy.Expr) -> SymbolNames:
    """
    Collect the names of the symbols in the cost-matrix *expressions*, by kind.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) to collect symbol names from.

    Returns
    -------
    SymbolNames
        The names found.
    """
    free_symbols: set[sympy.Expr] = set().union(*(expression.free_symbols for expression in expressions))
    random_symbols = {symbol for symbol in free_symbols if sympy.stats.rv.is_random(symbol)}
    distribution_symbols = _distribution_symbols(random_symbols)
    return SymbolNames(
        deterministic=frozenset(str(symbol) for symbol in free_symbols - random_symbols),
        random=frozenset(str(symbol) for symbol in random_symbols),
        distribution=frozenset(str(symbol) for symbol in distribution_symbols),
    )


def _check_parameter_names(
    *expressions: sympy.Expr,
    aliases: Mapping[str, str | sympy.Symbol],
    default_names: Iterable[str],
    bound_names: Iterable[str] = (),
    outlier_names: Iterable[str] = (),
    constraint_names: Iterable[str] = (),
) -> None:
    """
    Raise if an alias, default, bound, condition or outlier marker of the cost matrix cannot mean what it says.

    ``CostMatrix`` is built incrementally, so ``alias()``, ``set_default()``, ``constrain()`` and
    ``mark_outlier_sensitive()`` cannot check their arguments against the cost matrix's symbols when
    they are called. ``Metric`` can, once the cost matrix is complete, and does so here, so that a typo
    is one clear error at construction instead of a value that is silently never used. Rejected are:

    - an alias whose target, or a default, bound, condition or outlier marker whose parameter, is no symbol;
    - an alias named after another symbol, which would make that symbol impossible to pass;
    - a default or outlier marker on a random variable, whose values are drawn rather than passed.

    A bound or condition on a random variable is allowed, so that one cost matrix builder can serve a
    metric in which the variable is stochastic and one in which it is not.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) whose symbols are considered valid alias targets / parameter names.
    aliases : Mapping[str, str]
        The cost matrix's registered aliases, mapping alias name to target symbol name.
    default_names : Iterable[str]
        The parameter names of the cost matrix's defaults, aliases already resolved.
    bound_names : Iterable[str], default=()
        The parameter names of the cost matrix's bounds, aliases already resolved.
    outlier_names : Iterable[str], default=()
        The parameter names marked outlier-sensitive, aliases already resolved.
    constraint_names : Iterable[str], default=()
        The parameter names the conditions passed to ``constrain`` read, aliases already resolved.

    Raises
    ------
    ValueError
        If any of the above is found.
    """
    names = _symbol_names(*expressions)
    known_names = names.all

    unknown_alias_targets = {
        f'{alias!r} -> {str(target)!r}' for alias, target in aliases.items() if str(target) not in known_names
    }
    shadowing_aliases = {
        f'{alias!r} -> {str(target)!r}'
        for alias, target in aliases.items()
        if alias in known_names and alias != str(target)
    }
    default_names, outlier_names = set(default_names), set(outlier_names)
    unknown_defaults = default_names - known_names
    unknown_bounds = set(bound_names) - known_names
    unknown_constraints = set(constraint_names) - known_names
    unknown_outliers = outlier_names - known_names

    messages = []
    if unknown_alias_targets:
        messages.append(f'alias(es) {sorted(unknown_alias_targets)} whose target is not a cost matrix symbol')
    if unknown_defaults:
        messages.append(f'default(s) for {sorted(unknown_defaults)}, which is not a cost matrix symbol or alias')
    if unknown_bounds:
        messages.append(f'bound(s) on {sorted(unknown_bounds)}, which is not a cost matrix symbol or alias')
    if unknown_constraints:
        messages.append(f'condition(s) on {sorted(unknown_constraints)}, which is not a cost matrix symbol or alias')
    if unknown_outliers:
        messages.append(
            f'outlier-sensitive marker(s) on {sorted(unknown_outliers)}, which is not a cost matrix symbol or alias'
        )
    if shadowing_aliases:
        messages.append(
            f'alias(es) {sorted(shadowing_aliases)} named after another cost matrix symbol, which could then '
            'not be passed under its own name'
        )
    if random_defaults := default_names & names.random:
        messages.append(
            f'default(s) for the random variable(s) {sorted(random_defaults)}, whose values are drawn rather '
            'than passed: set defaults for the parameters of their distributions instead'
        )
    if random_outliers := outlier_names & names.random:
        messages.append(
            f'outlier-sensitive marker(s) on the random variable(s) {sorted(random_outliers)}, whose values are '
            'drawn rather than passed'
        )
    if messages:
        raise ValueError(
            f'The cost matrix has {"; and ".join(messages)}. Known symbol names: {sorted(known_names)}. '
            'A name that matches no symbol is usually a typo in a symbol or alias name.'
        )
