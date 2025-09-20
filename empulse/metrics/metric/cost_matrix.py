import sys
from collections.abc import MutableMapping
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import sympy


class CostMatrix:
    """
    Class to create a custom value/cost-sensitive cost matrix.

    You add the costs and benefits that make up the cost matrix for each case
    (true positive, true negative, false positive, false negative).
    The costs and benefits are specified using sympy symbols or expressions.
    Stochastic variables are supported and can be specified using sympy.stats random variables.
    Stochastic variables are assumed to be independent of each other.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

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

    @property
    def tp_benefit(self) -> sympy.Expr:  # noqa: D102
        return self._tp_benefit

    @property
    def tn_benefit(self) -> sympy.Expr:  # noqa: D102
        return self._tn_benefit

    @property
    def fp_benefit(self) -> sympy.Expr:  # noqa: D102
        return -self._fp_cost

    @property
    def fn_benefit(self) -> sympy.Expr:  # noqa: D102
        return -self._fn_cost

    @property
    def tp_cost(self) -> sympy.Expr:  # noqa: D102
        return -self._tp_benefit

    @property
    def tn_cost(self) -> sympy.Expr:  # noqa: D102
        return -self._tn_benefit

    @property
    def fp_cost(self) -> sympy.Expr:  # noqa: D102
        return self._fp_cost

    @property
    def fn_cost(self) -> sympy.Expr:  # noqa: D102
        return self._fn_cost

    def add_tp_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a true positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a true positive.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tp_benefit += term
        return self

    def add_tn_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a true negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a true negative.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tn_benefit += term
        return self

    def add_fp_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a false positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a false positive.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fp_cost -= term
        return self

    def add_fn_benefit(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the benefit of classifying a false negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a false negative.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fn_cost -= term
        return self

    def add_tp_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a true positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a true positive.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tp_benefit -= term
        return self

    def add_tn_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a true negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a true negative.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tn_benefit -= term
        return self

    def add_fp_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a false positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a false positive.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fp_cost += term
        return self

    def add_fn_cost(self, term: sympy.Expr | str) -> Self:
        """
        Add a term to the cost of classifying a false negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a false negative.

        Returns
        -------
        CostMatrix
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fn_cost += term
        return self

    def alias(self, alias: str | MutableMapping[str, sympy.Symbol | str], symbol: sympy.Symbol | None = None) -> Self:
        """
        Add an alias for a symbol.

        Parameters
        ----------
        alias: str | MutableMapping[str, sympy.Symbol | str]
            The alias to add. If a MutableMapping (.e.g, dictionary) is passed,
            the keys are the aliases and the values are the symbols.
        symbol: sympy.Symbol, optional
            The symbol to alias to.

        Returns
        -------
        CostMatrix

        Examples
        --------

        .. code-block:: python

            import sympy as sp
            from empulse.metrics import Metric, Cost

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
        defaults: float
            Default values for symbols or their aliases.
            These default values will be used if not provided in __call__.

        Returns
        -------
        CostMatrix

        Examples
        --------

        .. code-block:: python

            import sympy as sp
            from empulse.metrics import Metric, Cost

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
        symbol: str | sympy.Symbol
            The symbol to mark as outlier-sensitive.

        Returns
        -------
        CostMatrix

        Examples
        --------
        .. code-block:: python

            import numpy as np
            import sympy as sp
            from empulse.metrics import Metric, Cost
            from empulse.models import CSLogitClassifier, RobustCSClassifier
            from sklearn.datasets import make_classification

            X, y = make_classification()
            a, b = sp.symbols('a b')
            cost_matrix = CostMatrix().add_fp_cost(a).add_fn_cost(b).mark_outlier_sensitive(a)
            cost_loss = Metric(cost_matrix, Cost())
            fn_cost = np.random.rand(y.size)

            model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
            model.fit(X, y, a=np.random.rand(y.size), b=5)
        """
        if isinstance(symbol, str):
            symbol = sympy.sympify(symbol)
        if not isinstance(symbol, sympy.Symbol):
            raise ValueError('The symbol must be a sympy.Symbol or a string that can be converted to a sympy.Symbol')
        self._outlier_sensitive_symbols.add(symbol)
        return self

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'tp_cost={self.tp_cost}, tn_cost={self.tn_cost}, '
            f'fp_cost={self.fp_cost}, fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self) -> str:
        return (  # type: ignore[no-any-return]
            r"""
        \begin{array}{c|cc}
          & y=0 & y=1 \\
        \hline
        \hat y=0 & \text{"""
            + self.tn_cost._repr_latex_()
            + r"""} & \text{"""
            + self.fn_cost._repr_latex_()
            + r"""} \\
        \hat y=1 & \text{"""
            + self.fp_cost._repr_latex_()
            + r"""} & \text{"""
            + self.tp_cost._repr_latex_()
            + r"""} \\
        \end{array}
        """
        )
