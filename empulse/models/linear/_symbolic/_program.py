# Adapted from gplearn 0.4.3 (https://github.com/trevorstephens/gplearn), BSD-3-Clause,
# Copyright (c) 2015-2026 Trevor Stephens. See the license text in this package's ``__init__``.
"""Expressions as flattened prefix trees, and the genetic operators that act on them."""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

from ...._types import FloatNDArray
from ._functions import Function

# A node is a function, a feature (an ``int`` index) or a constant (a ``float``).
Node: TypeAlias = Function | int | float

# Following Koza (1992), crossover points are functions 90% of the time and leaves 10% of the time.
_FUNCTION_WEIGHT = 0.9
_LEAF_WEIGHT = 0.1


class Program:
    """
    An expression in prefix order, for example ``add(X0, mul(X1, 0.5))`` is ``[add, 0, mul, 1, 0.5]``.

    Parameters
    ----------
    nodes : list of Function, int or float
        The expression in prefix order. Functions are operators, ``int`` nodes the index of a feature
        and ``float`` nodes constants.

    feature_names : list of str or None, default=None
        Names printed in place of ``X0``, ``X1``, ... .
    """

    __slots__ = ('feature_names', 'nodes')

    def __init__(self, nodes: list[Node], feature_names: Sequence[str] | None = None) -> None:
        self.nodes = nodes
        self.feature_names = feature_names

    @property
    def length_(self) -> int:
        """Number of operators, features and constants in the expression."""
        return len(self.nodes)

    @property
    def depth_(self) -> int:
        """Depth of the expression tree; a lone feature or constant has depth 0."""
        pending = [0]
        depth = 1
        for node in self.nodes:
            if isinstance(node, Function):
                pending.append(node.arity)
                depth = max(len(pending), depth)
            else:
                pending[-1] -= 1
                while pending[-1] == 0:
                    pending.pop()
                    pending[-1] -= 1
        return depth - 1

    @property
    def key(self) -> tuple[object, ...]:
        """
        Hashable identity of the expression.

        A constant is wrapped so that it cannot collide with the feature of the same value.
        """
        return tuple(
            node.name if isinstance(node, Function) else node if isinstance(node, int) else ('c', node)
            for node in self.nodes
        )

    def is_valid(self) -> bool:
        """Check that the nodes form exactly one complete expression."""
        pending = [0]
        for node in self.nodes:
            if isinstance(node, Function):
                pending.append(node.arity)
            else:
                pending[-1] -= 1
                while pending[-1] == 0:
                    pending.pop()
                    pending[-1] -= 1
        return pending == [-1]

    def constants(self) -> list[float]:
        """Return the constants of the expression, in prefix order."""
        return [node for node in self.nodes if isinstance(node, float)]

    def with_constants(self, values: Sequence[float] | FloatNDArray) -> 'Program':
        """Return a copy of the expression with its constants replaced by *values*, in prefix order."""
        replacement = iter(values)
        nodes: list[Node] = [float(next(replacement)) if isinstance(node, float) else node for node in self.nodes]
        return Program(nodes, self.feature_names)

    def execute(self, X: FloatNDArray) -> FloatNDArray:
        """
        Evaluate the expression on every row of *X*.

        Operators that would overflow, divide by zero or take the logarithm of a non-positive number
        either have a protected definition or produce a non-finite value, never a warning.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_features)
            Features.

        Returns
        -------
        y : 1D numpy.ndarray, shape=(n_samples,)
            The value of the expression for every row.
        """
        n_samples = X.shape[0]
        first = self.nodes[0]
        if isinstance(first, float):
            return np.full(n_samples, first)
        if isinstance(first, int):
            return X[:, first]  # type: ignore[no-any-return]

        # Each entry pairs a function with the arguments it has received so far. A function is applied as
        # soon as it has all of them; constants stay scalars and broadcast against the feature columns.
        stack: list[tuple[Function, list[Any]]] = []
        with np.errstate(all='ignore'):
            for node in self.nodes:
                if isinstance(node, Function):
                    stack.append((node, []))
                else:
                    stack[-1][1].append(X[:, node] if isinstance(node, int) else node)
                while len(stack[-1][1]) == stack[-1][0].arity:
                    function, arguments = stack.pop()
                    result = function(*arguments)
                    if not stack:
                        result = np.asarray(result, dtype=np.float64)
                        return result if result.ndim else np.full(n_samples, result)  # type: ignore[no-any-return]
                    stack[-1][1].append(result)
        raise AssertionError('unreachable: the expression is incomplete')  # pragma: no cover

    def __str__(self) -> str:
        """Return the expression with nested function-call syntax, like ``add(X0, mul(X1, 0.500))``."""
        pending = [0]
        parts: list[str] = []
        last = len(self.nodes) - 1
        for i, node in enumerate(self.nodes):
            if isinstance(node, Function):
                pending.append(node.arity)
                parts.append(node.name + '(')
            else:
                if isinstance(node, int):
                    parts.append(f'X{node}' if self.feature_names is None else self.feature_names[node])
                else:
                    parts.append(f'{node:.3f}')
                pending[-1] -= 1
                while pending[-1] == 0:
                    pending.pop()
                    pending[-1] -= 1
                    parts.append(')')
                if i != last:
                    parts.append(', ')
        return ''.join(parts)

    def __repr__(self) -> str:
        return f'Program({self})'


class ProgramSpace:
    """
    The search space of expressions, and the operators that create and vary them.

    Every operator draws all of its randomness from the ``RandomState`` it is given, in the order
    of the gplearn operators it is adapted from.

    Parameters
    ----------
    function_set : sequence of Function
        The operators an expression may use.

    n_features : int
        Number of features an expression may refer to.

    const_range : tuple of two floats or None
        Range constants are drawn from. ``None`` creates expressions without constants.

    init_depth : tuple of two ints
        Range the maximum depth of a new expression is drawn from.

    init_method : {'grow', 'full', 'half_and_half'}
        How new expressions are grown.

    point_replace_rate : float
        Probability that a node is replaced by a point mutation.

    max_length : int or None
        The most nodes an expression created by :meth:`build` may have. ``None`` for no limit.

    constant_rate : float or None, default=None
        Probability that a new leaf is a constant rather than a feature. ``None`` treats a constant
        as one more choice next to the features, as gplearn does, so it has probability ``1 / (n_features + 1)``.
    """

    def __init__(
        self,
        function_set: Sequence[Function],
        n_features: int,
        const_range: tuple[float, float] | None,
        init_depth: tuple[int, int],
        init_method: str,
        point_replace_rate: float,
        max_length: int | None,
        constant_rate: float | None = None,
    ) -> None:
        self.function_set = tuple(function_set)
        self.n_features = n_features
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.point_replace_rate = point_replace_rate
        self.max_length = max_length
        self.constant_rate = constant_rate
        self.arities: dict[int, list[Function]] = {}
        for function in self.function_set:
            self.arities.setdefault(function.arity, []).append(function)

    @property
    def min_length(self) -> int:
        """Length of the shortest expression :meth:`build` can create."""
        return 1 + min(function.arity for function in self.function_set)

    def _random_terminal(self, random_state: np.random.RandomState) -> int | float:
        if self.const_range is None:
            return int(random_state.randint(self.n_features))
        if self.constant_rate is not None:
            if random_state.uniform() < self.constant_rate:
                return float(random_state.uniform(*self.const_range))
            return int(random_state.randint(self.n_features))
        terminal = random_state.randint(self.n_features + 1)
        if terminal == self.n_features:
            return float(random_state.uniform(*self.const_range))
        return int(terminal)

    def _random_function(self, random_state: np.random.RandomState, shortest_length: int) -> Function | None:
        """
        Draw a function that keeps the expression within ``max_length``, or ``None`` if there is none.

        *shortest_length* is the length of the expression once it is completed with only leaves.
        Adding a function raises that by its arity.
        """
        if self.max_length is None:
            return self.function_set[random_state.randint(len(self.function_set))]
        fitting = [f for f in self.function_set if shortest_length + f.arity <= self.max_length]
        if not fitting:
            return None
        return fitting[random_state.randint(len(fitting))]

    def build(self, random_state: np.random.RandomState) -> list[Node]:
        """Grow a random expression, starting from a function to avoid degenerate expressions."""
        if self.init_method == 'half_and_half':
            method = 'full' if random_state.randint(2) else 'grow'
        else:
            method = self.init_method
        max_depth = random_state.randint(self.init_depth[0], self.init_depth[1] + 1)

        # With ``max_length`` set, a function is only added when the open slots still fit after it.
        function = self._random_function(random_state, shortest_length=1)
        assert function is not None, 'max_length is smaller than the smallest expression'
        program: list[Node] = [function]
        pending = [function.arity]
        pending_total = function.arity

        while pending:
            depth = len(pending)
            choice = random_state.randint(self.n_features + len(self.function_set))
            function = None
            if depth < max_depth and (method == 'full' or choice <= len(self.function_set)):
                function = self._random_function(random_state, shortest_length=len(program) + pending_total)
            if function is not None:
                program.append(function)
                pending.append(function.arity)
                pending_total += function.arity - 1
            else:
                program.append(self._random_terminal(random_state))
                pending[-1] -= 1
                pending_total -= 1
                while pending[-1] == 0:
                    pending.pop()
                    if not pending:
                        return program
                    pending[-1] -= 1
        raise AssertionError('unreachable: the expression was never completed')  # pragma: no cover

    @staticmethod
    def get_subtree(random_state: np.random.RandomState, program: Sequence[Node]) -> tuple[int, int]:
        """Pick a random subtree and return the ``start`` and ``end`` of its nodes."""
        probabilities = np.array([_FUNCTION_WEIGHT if isinstance(node, Function) else _LEAF_WEIGHT for node in program])
        probabilities = np.cumsum(probabilities / probabilities.sum())
        start = int(np.searchsorted(probabilities, random_state.uniform()))

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, Function):
                stack += node.arity
            end += 1
        return start, end

    def crossover(self, program: list[Node], donor: list[Node], random_state: np.random.RandomState) -> list[Node]:
        """Replace a random subtree of *program* by a random subtree of *donor*."""
        start, end = self.get_subtree(random_state, program)
        donor_start, donor_end = self.get_subtree(random_state, donor)
        return program[:start] + donor[donor_start:donor_end] + program[end:]

    def subtree_mutation(self, program: list[Node], random_state: np.random.RandomState) -> list[Node]:
        """Crossover with a new random expression ('headless chicken' mutation)."""
        return self.crossover(program, self.build(random_state), random_state)

    def hoist_mutation(self, program: list[Node], random_state: np.random.RandomState) -> list[Node]:
        """Replace a random subtree by one of its own subtrees, which shrinks the expression."""
        start, end = self.get_subtree(random_state, program)
        subtree = program[start:end]
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        return program[:start] + subtree[sub_start:sub_end] + program[end:]

    def point_mutation(self, program: list[Node], random_state: np.random.RandomState) -> list[Node]:
        """Replace random nodes by another node of the same arity, which keeps the shape of the expression."""
        mutated = list(program)
        for position in np.where(random_state.uniform(size=len(mutated)) < self.point_replace_rate)[0]:
            node = mutated[position]
            if isinstance(node, Function):
                candidates = self.arities[node.arity]
                mutated[position] = candidates[random_state.randint(len(candidates))]
            else:
                mutated[position] = self._random_terminal(random_state)
        return mutated
