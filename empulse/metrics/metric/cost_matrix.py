from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Self

import sympy
import sympy.stats

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


@dataclass(frozen=True)
class ParameterPredicate:
    """An arbitrary condition over several cost-matrix parameters at once."""

    predicate: Callable[[Mapping[str, Any]], bool]
    message: str


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
        If you want to set a default using an alias name, you must call
        :meth:`alias` **before** calling :meth:`set_default`.  Defaults passed
        via alias names are immediately resolved to their underlying symbol names
        during this call; any alias registered afterwards will *not* retroactively
        match previously stored defaults.

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
        # Convert aliases to symbol names before storing defaults
        converted_defaults = {}
        for key, value in defaults.items():
            if key in self._aliases:
                symbol_name = str(self._aliases[key])
                converted_defaults[symbol_name] = value
            else:
                converted_defaults[key] = value

        self._defaults.update(converted_defaults)

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
            The symbol to mark as outlier-sensitive.

        Returns
        -------
        CostMatrix
            The cost matrix, to allow method chaining.

        Raises
        ------
        TypeError
            If ``symbol`` is not a ``str`` or :class:`sympy.Symbol`.

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
        target: str | sympy.Symbol | Callable[[Mapping[str, Any]], bool],
        lower: float | None = None,
        upper: float | None = None,
        *,
        message: str | None = None,
    ) -> Self:
        """
        Restrict the values a parameter is allowed to take.

        Constraints are checked when the metric is called, and when a model fitting on this metric
        first receives its parameters. A violation raises a :class:`ValueError`.

        Two forms are supported. Passing a symbol (or alias) with `lower` and/or `upper` bounds the
        values of that one parameter. Passing a callable expresses a condition over several
        parameters at once.

        Parameters
        ----------
        target : str, sympy.Symbol or callable
            The symbol or alias to bound, or a callable taking the mapping of resolved parameter
            values and returning whether they are acceptable.

        lower : float, optional
            Smallest allowed value, inclusive. Only used when `target` is a symbol or alias.

        upper : float, optional
            Largest allowed value, inclusive. Only used when `target` is a symbol or alias.

        message : str, optional
            Explanation to report when a callable `target` rejects the parameters.
            Required when `target` is a callable.

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
            or if `target` is a callable and `message` is not given.

        Notes
        -----
        Bounds are stored against the resolved symbol, so call :meth:`alias` *before*
        :meth:`constrain` if you want to constrain a parameter by its alias.

        A callable receives the parameters keyed by **symbol name**, with aliases already resolved
        and defaults already applied.

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
                .constrain(
                    lambda params: params['clv'] > params['d'],
                    message='clv must exceed the incentive cost',
                )
            )
            metric = Metric(cost_matrix, MaxProfit())
        """
        if callable(target) and not isinstance(target, sympy.Symbol):
            if message is None:
                raise ValueError('A message is required when constraining with a callable.')
            self._predicates.append(ParameterPredicate(predicate=target, message=message))
            return self

        if isinstance(target, str):
            target = sympy.Symbol(target)
        # A sympy.stats random variable is a RandomSymbol, not a Symbol, but it names a parameter
        # just the same. Accepting it lets one builder constrain a symbol that is stochastic in one
        # metric and deterministic in its sibling; the constraint is simply inert in the stochastic
        # case, where the variable is drawn rather than supplied by the caller.
        if not isinstance(target, sympy.Symbol | sympy.stats.rv.RandomSymbol):
            raise TypeError(
                'The target must be a sympy.Symbol, a string that can be converted to one, or a callable, '
                f'got {type(target).__name__!r} instead.'
            )
        if lower is None and upper is None:
            raise ValueError(f"Constraining '{target}' requires a lower bound, an upper bound, or both.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound {lower} is greater than upper bound {upper} for '{target}'.")

        name = str(self._aliases.get(str(target), target))
        existing = self._bounds.get(name)
        if existing is not None:  # tighten rather than replace, so two calls compose
            lower = _tightest(lower, existing.lower, max)
            upper = _tightest(upper, existing.upper, min)
        self._bounds[name] = ParameterBounds(lower=lower, upper=upper)
        return self

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


# Symbol names used internally by Metric and its strategies to inject data (labels, scores,
# true/false positive rates, class priors) into the lambdified cost-matrix expressions, or used
# by to_latex() rendering, or as a keyword-only parameter of Metric._prepare_parameters(). A
# user-defined symbol or alias sharing one of these names would either be silently fused with the
# internal one, raise a confusing internal TypeError, or (for 'n_samples') actually be captured by
# _prepare_parameters()'s own n_samples parameter instead of reaching its **kwargs - see
# Metric.__init__ and _check_reserved_symbol_names(). 'validate' is reserved for the same reason
# as 'n_samples': it is a keyword-only parameter of _prepare_parameters() and of the public
# scoring methods, so a symbol of that name would be captured by it rather than reach **kwargs.
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
    symbol_names = {str(symbol) for expression in expressions for symbol in expression.free_symbols}
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
    for expression in expressions:
        for symbol in expression.free_symbols:
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


def _collect_known_symbol_names(*expressions: sympy.Expr) -> set[str]:
    """
    Collect the names of every symbol a cost matrix's expressions can be evaluated with.

    This includes each expression's own free symbols (deterministic symbols and, for a
    stochastic expression, the random variable's own symbol, e.g. ``gamma`` for
    ``sympy.stats.Beta('gamma', alpha, beta)``) as well as any random variable's distribution
    parameters (e.g. ``alpha``/``beta`` above), since those are what a caller actually supplies
    a value for at call time.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) to collect symbol names from.

    Returns
    -------
    set of str
        The names of every symbol found.
    """
    free_symbols: set[sympy.Expr] = set()
    for expression in expressions:
        free_symbols |= expression.free_symbols

    stochastic_params: set[sympy.Expr] = set()
    for expression in expressions:
        for atom in expression.atoms(sympy.stats.rv.RandomSymbol):
            pspace = atom.pspace
            if hasattr(pspace, 'distribution') and hasattr(pspace.distribution, 'args'):
                for arg in pspace.distribution.args:
                    stochastic_params.update(arg.free_symbols)

    return {str(symbol) for symbol in free_symbols | stochastic_params}


def _check_known_alias_and_default_targets(
    *expressions: sympy.Expr, aliases: Mapping[str, str | sympy.Symbol], default_names: Iterable[str]
) -> None:
    """
    Raise if an alias's target, or a default's parameter name, matches no symbol in the cost matrix.

    ``CostMatrix`` is built incrementally, so ``alias()`` and ``set_default()`` cannot validate
    their arguments against the cost matrix's symbols at the time they are called. ``Metric``
    can, once the cost matrix is complete, and does so here - turning a typo'd alias target, or a
    ``set_default()`` call for an alias that had not been registered yet (see the ordering note
    in :meth:`~empulse.metrics.CostMatrix.set_default`), into one clear error at construction
    time instead of a confusing "did not receive it" error only once the metric is called.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) whose symbols are considered valid alias targets / default names.
    aliases : Mapping[str, str]
        The cost matrix's registered aliases, mapping alias name to target symbol name.
    default_names : Iterable[str]
        The cost matrix's registered default parameter names (already resolved to symbol names,
        for aliases registered before the corresponding ``set_default()`` call).

    Raises
    ------
    ValueError
        If any alias target or default name does not match a known symbol name.
    """
    known_names = _collect_known_symbol_names(*expressions)

    unknown_alias_targets = {
        f'{alias!r} -> {str(target)!r}' for alias, target in aliases.items() if str(target) not in known_names
    }
    unknown_defaults = {name for name in default_names if name not in known_names}

    if unknown_alias_targets or unknown_defaults:
        messages = []
        if unknown_alias_targets:
            messages.append(f'alias(es) {sorted(unknown_alias_targets)} whose target is not a cost matrix symbol')
        if unknown_defaults:
            messages.append(f'default(s) for {sorted(unknown_defaults)}, which is not a cost matrix symbol or alias')
        raise ValueError(
            f'The cost matrix has {" and ".join(messages)}. Known symbol names: {sorted(known_names)}. '
            'This is usually a typo, or set_default() was called with an alias name before that alias '
            'was registered with alias() - aliases must be registered before set_default() can resolve '
            'them (see the CostMatrix.set_default() docstring).'
        )
