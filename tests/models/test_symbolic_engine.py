"""
The symbolic regression engine behind ``ProfSRClassifier``, tested without a classifier around it.

Most of it is adapted from gplearn and must keep behaving like it (the reference values below were produced by
gplearn 0.4.3); the cache, the length limit, the Pareto front and the constant tuning are additions.
"""

import itertools
import pickle
from collections import Counter

import numpy as np
import pytest

from empulse.models.linear._symbolic import (
    FUNCTION_NAMES,
    Program,
    ProgramSpace,
    SearchSettings,
    evolve,
    get_function,
)
from empulse.models.linear._symbolic._evolve import _tune_program

FOUR = ('add', 'sub', 'mul', 'div')

REFERENCE = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'div': lambda a, b: np.where(np.abs(b) > 0.001, a / np.where(b == 0, 1, b), 1.0),
    'sqrt': lambda a: np.sqrt(np.abs(a)),
    'log': lambda a: np.where(np.abs(a) > 0.001, np.log(np.abs(np.where(a == 0, 1, a))), 0.0),
    'abs': np.abs,
    'neg': np.negative,
    'inv': lambda a: np.where(np.abs(a) > 0.001, 1.0 / np.where(a == 0, 1, a), 0.0),
    'max': np.maximum,
    'min': np.minimum,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': lambda a: np.exp(np.minimum(a, 100.0)),
    'sig': lambda a: 1 / (1 + np.exp(-a)),
}


def make_space(
    function_names=FOUR, *, n_features=3, max_length=None, init_method='half_and_half', const_range=(-1.0, 1.0)
):
    return ProgramSpace(
        function_set=[get_function(name) for name in function_names],
        n_features=n_features,
        const_range=const_range,
        init_depth=(2, 6),
        init_method=init_method,
        point_replace_rate=0.05,
        max_length=max_length,
    )


def make_settings(**overrides):
    settings = {
        'max_iter': 6,
        'population_size': 60,
        'tournament_size': 20,
        'crossover_rate': 0.9,
        'mutate_subtree_rate': 0.01,
        'hoist_rate': 0.01,
        'mutate_point_rate': 0.01,
        'parsimony_coefficient': 0.001,
        'max_samples': 1.0,
        'n_tuned_programs': 0,
        'tuning_interval': 5,
        'tuning_max_iter': 50,
        'patience': None,
        'tolerance': 1e-4,
        'max_time': None,
        'n_jobs': 1,
    }
    return SearchSettings(**{**settings, **overrides})


@pytest.fixture(scope='module')
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    return np.asfortranarray(X), X[:, 0] * X[:, 1] + X[:, 2]


def mse_fitness(y):
    return lambda y_pred, rows: float(np.mean((y_pred - (y if rows is None else y[rows])) ** 2))


class TestFunctions:
    @pytest.mark.parametrize('name', FUNCTION_NAMES)
    def test_matches_the_numpy_definition(self, name):
        function = get_function(name)
        values = np.array([-2.0, -0.0005, 0.0, 0.0005, 0.5, 3.0])
        arguments = [values, values[::-1]][: function.arity]

        np.testing.assert_allclose(function(*arguments), REFERENCE[name](*arguments), equal_nan=True)

    @pytest.mark.parametrize('name', FUNCTION_NAMES)
    def test_functions_stay_finite_on_awkward_inputs(self, name):
        function = get_function(name)
        edge = np.array([0.0, -0.0, 1e-300, -1.0, 1e6])
        with np.errstate(all='ignore'):
            result = function(*[edge] * function.arity)
        assert np.isfinite(result).all()

    def test_exponential_does_not_overflow(self):
        assert np.isfinite(get_function('exp')(np.array([1e6, 800.0]))).all()

    def test_unknown_function_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown function'):
            get_function('cube')

    def test_functions_pickle_as_the_singleton(self):
        function = get_function('mul')
        assert pickle.loads(pickle.dumps(function)) is function


class TestProgram:
    add, mul = get_function('add'), get_function('mul')

    def program(self, *nodes):
        return Program(list(nodes))

    def test_executes_features_and_constants(self):
        X = np.arange(12.0).reshape(4, 3)
        program = self.program(self.add, 0, self.mul, 1, 2.0)

        np.testing.assert_array_equal(program.execute(X), X[:, 0] + X[:, 1] * 2.0)

    @pytest.mark.parametrize(('nodes', 'expected'), [([2.5], [2.5] * 4), ([1], [1.0, 4.0, 7.0, 10.0])])
    def test_executes_single_node_expressions(self, nodes, expected):
        X = np.arange(12.0).reshape(4, 3)

        np.testing.assert_array_equal(Program(nodes).execute(X), expected)

    def test_expression_of_constants_has_one_value_per_row(self):
        X = np.arange(12.0).reshape(4, 3)

        np.testing.assert_array_equal(self.program(self.add, 1.5, 2.5).execute(X), [4.0] * 4)

    def test_never_warns_about_overflow_or_invalid_values(self, recwarn):
        X = np.full((3, 2), 1e200)
        result = self.program(self.mul, self.mul, 0, 1, 0).execute(X)

        assert np.isinf(result).all()
        assert not recwarn.list

    def test_string_uses_function_call_syntax(self):
        program = self.program(self.add, 0, self.mul, 1, 0.5)

        assert str(program) == 'add(X0, mul(X1, 0.500))'
        program.feature_names = ['recency', 'tenure']
        assert str(program) == 'add(recency, mul(tenure, 0.500))'

    def test_length_and_depth(self):
        program = self.program(self.add, 0, self.mul, 1, 0.5)

        assert program.length_ == 5
        assert program.depth_ == 2
        assert self.program(0).depth_ == 0

    def test_key_does_not_confuse_feature_one_with_constant_one(self):
        feature, constant = self.program(self.add, 1, 0), self.program(self.add, 1.0, 0)

        assert feature.key != constant.key
        assert feature.key == self.program(self.add, 1, 0).key

    def test_validity(self):
        assert self.program(self.add, 0, 1).is_valid()
        assert not self.program(self.add, 0).is_valid()
        assert not self.program(self.add, 0, 1, 2).is_valid()

    def test_constants_round_trip(self):
        program = self.program(self.add, 0.25, self.mul, 1, 0.5)
        replaced = program.with_constants([3.0, 4.0])

        assert program.constants() == [0.25, 0.5]
        assert replaced.constants() == [3.0, 4.0]
        assert [node for node in replaced.nodes if not isinstance(node, float)] == [self.add, self.mul, 1]
        assert all(type(constant) is float for constant in program.with_constants(np.array([1, 2])).constants())

    def test_pickles(self):
        program = self.program(self.add, 0, self.mul, 1, 0.5)

        assert pickle.loads(pickle.dumps(program)).key == program.key


@pytest.mark.parametrize('init_method', ['grow', 'full', 'half_and_half'])
@pytest.mark.parametrize('max_length', [None, 9, 3], ids=['unbounded', 'cap9', 'cap3'])
class TestOperators:
    def population(self, init_method, max_length, n=150):
        space = make_space(('add', 'mul', 'sqrt', 'sig'), max_length=max_length, init_method=init_method)
        return space, [Program(space.build(np.random.RandomState(seed))) for seed in range(n)]

    def test_new_expressions_are_valid_and_within_limits(self, init_method, max_length):
        _, programs = self.population(init_method, max_length)

        assert all(program.is_valid() for program in programs)
        assert all(program.depth_ <= 6 for program in programs)
        if max_length is not None:
            assert all(program.length_ <= max_length for program in programs)

    def test_variation_keeps_expressions_valid(self, init_method, max_length):
        space, programs = self.population(init_method, max_length)
        for seed, (parent, donor) in enumerate(itertools.pairwise(programs)):
            rs = np.random.RandomState(seed)
            for child in (
                space.crossover(parent.nodes, donor.nodes, rs),
                space.subtree_mutation(parent.nodes, rs),
                space.hoist_mutation(parent.nodes, rs),
                space.point_mutation(parent.nodes, rs),
            ):
                assert Program(child).is_valid()

    def test_hoisting_never_grows_and_point_mutation_keeps_the_shape(self, init_method, max_length):
        space, programs = self.population(init_method, max_length)
        for seed, parent in enumerate(programs):
            rs = np.random.RandomState(seed)
            assert len(space.hoist_mutation(parent.nodes, rs)) <= parent.length_
            mutated = space.point_mutation(parent.nodes, rs)
            assert [getattr(node, 'arity', None) for node in mutated] == [
                getattr(n, 'arity', None) for n in parent.nodes
            ]


@pytest.mark.parametrize(('rate', 'expected'), [(0.0, (0.0, 0.0)), (1.0, (1.0, 1.0)), (0.5, (0.4, 0.6))])
def test_constant_rate_sets_the_share_of_constant_leaves(rate, expected):
    space = ProgramSpace(
        [get_function(name) for name in FOUR], 3, (-1.0, 1.0), (2, 6), 'half_and_half', 0.05, None, constant_rate=rate
    )
    leaves = [
        node for seed in range(200) for node in space.build(np.random.RandomState(seed)) if not hasattr(node, 'arity')
    ]
    share = sum(isinstance(node, float) for node in leaves) / len(leaves)

    assert expected[0] <= share <= expected[1]


def test_without_a_constant_rate_a_constant_is_one_more_choice_next_to_the_features():
    space = make_space(n_features=3)
    leaves = [
        node for seed in range(300) for node in space.build(np.random.RandomState(seed)) if not hasattr(node, 'arity')
    ]

    assert sum(isinstance(node, float) for node in leaves) / len(leaves) == pytest.approx(0.25, abs=0.05)


def test_the_smallest_expression_has_a_function_over_leaves():
    assert make_space(('add', 'sig')).min_length == 2
    assert make_space(('add', 'mul')).min_length == 3


class TestEvolve:
    @pytest.mark.parametrize(
        ('names', 'seed', 'best_fitness', 'average_length'),
        [
            (
                FOUR,
                3,
                [1.626099683907504, *[0.8415613087883668] * 5],
                [40.066667, 24.366667, 4.733333, 1.033333, 1.3, 1.333333],
            ),
            (
                tuple(n for n in FUNCTION_NAMES if n not in {'exp', 'sig'}),
                7,
                [1.609333651487194, 1.3785736719590915, 0.8993017889166123, *[0.8415613087883668] * 3],
                [14.0, 9.833333, 11.95, 10.083333, 6.416667, 1.233333],
            ),
        ],
        ids=['four_functions', 'all_gplearn_functions'],
    )
    def test_reproduces_gplearn(self, regression_data, names, seed, best_fitness, average_length):
        X, y = regression_data
        result = evolve(X, mse_fitness(y), make_space(names), make_settings(), seed)

        np.testing.assert_allclose(result.run_details['best_fitness'], best_fitness, rtol=1e-12)
        np.testing.assert_allclose(result.run_details['average_length'], average_length, atol=1e-6)
        assert str(result.last_best) == 'X2'

    def test_same_seed_same_search(self, regression_data):
        X, y = regression_data
        runs = [evolve(X, mse_fitness(y), make_space(), make_settings(), 5) for _ in range(2)]

        assert runs[0].run_details['best_fitness'] == runs[1].run_details['best_fitness']
        assert [str(p.program) for p in runs[0].pareto_front] == [str(p.program) for p in runs[1].pareto_front]

    def test_accepts_a_random_state_instance(self, regression_data):
        X, y = regression_data
        result = evolve(X, mse_fitness(y), make_space(), make_settings(max_iter=2), np.random.RandomState(0))

        assert len(result.run_details['generation']) == 2

    def test_every_distinct_expression_is_scored_once(self, regression_data, monkeypatch):
        X, y = regression_data
        executed = []
        original = Program.execute

        def recording(self, X):
            executed.append(self.key)
            return original(self, X)

        monkeypatch.setattr(Program, 'execute', recording)
        result = evolve(X, mse_fitness(y), make_space(), make_settings(max_iter=8, population_size=100), 0)

        assert Counter(executed).most_common(1)[0][1] == 1
        assert len(executed) == sum(result.run_details['n_evaluated'])
        # An evolved population repeats expressions a lot, which is what the cache saves.
        assert len(executed) < 8 * 100

    def test_a_loss_that_cannot_score_an_expression_leaves_it_out(self, regression_data):
        X, y = regression_data

        def fitness(y_pred, rows):
            return np.inf if y_pred.std() < 1e-12 else float(np.mean((y_pred - y) ** 2))

        result = evolve(X, fitness, make_space(), make_settings(), 0)

        assert all(np.isfinite(point.loss) for point in result.pareto_front)
        assert all(point.program.execute(X).std() >= 1e-12 for point in result.pareto_front)

    def test_front_with_nothing_scorable_falls_back_to_the_last_best(self, regression_data):
        X, _ = regression_data
        result = evolve(X, lambda y_pred, rows: np.inf, make_space(), make_settings(max_iter=2), 0)

        assert result.pareto_front == []
        assert result.program is result.last_best

    def test_length_limit_holds_in_every_generation(self, regression_data):
        X, y = regression_data
        result = evolve(X, mse_fitness(y), make_space(max_length=6), make_settings(max_iter=10, population_size=80), 0)

        assert max(result.run_details['average_length']) <= 6
        assert all(point.length <= 6 for point in result.pareto_front)

    def test_batches_are_random_subsets_of_the_requested_size(self, regression_data):
        X, y = regression_data
        seen = []

        def fitness(y_pred, rows):
            seen.append(None if rows is None else rows)
            return float(np.mean((y_pred - (y if rows is None else y[rows])) ** 2))

        evolve(X, fitness, make_space(), make_settings(max_iter=2, max_samples=0.25), 0)

        batches = [rows for rows in seen if rows is not None and rows.size == 50]
        assert batches, 'no program was scored on a batch'
        assert all(np.array_equal(rows, np.unique(rows)) and rows.min() >= 0 and rows.max() < 200 for rows in batches)
        # The out-of-batch rows of the best program, and the final scoring of the front on every row.
        assert any(rows is not None and rows.size == 150 for rows in seen)
        assert seen[-1] is None

    def test_a_batch_covering_every_row_is_not_a_batch(self, regression_data):
        X, y = regression_data
        result = evolve(X, mse_fitness(y), make_space(), make_settings(max_iter=2, max_samples=0.9999), 0)

        assert all(np.isnan(result.run_details['best_oob_fitness']))

    def test_pareto_front_lists_every_improvement_over_shorter_expressions(self, regression_data):
        X, y = regression_data
        result = evolve(X, mse_fitness(y), make_space(), make_settings(max_iter=10, population_size=150), 1)
        front = result.pareto_front

        assert [p.length for p in front] == sorted({p.length for p in front})
        assert all(b.loss < a.loss for a, b in itertools.pairwise(front))
        assert result.program in [p.program for p in front]

    def test_patience_counts_generations_without_improvement(self, regression_data):
        X, y = regression_data
        result = evolve(X, mse_fitness(y), make_space(), make_settings(max_iter=30, patience=3, tolerance=1e-4), 3)

        # This search stops improving after its second generation.
        assert len(result.run_details['generation']) == 5


class TestTuning:
    def test_tuning_finds_the_constant_and_keeps_the_structure(self, regression_data):
        X, _ = regression_data
        mul = get_function('mul')
        program = Program([mul, 0, 0.5])
        target = 3.0 * X[:, 0]

        tuned, loss = _tune_program(program, X, mse_fitness(target), None, max_iter=200)

        assert loss < 1e-6
        assert tuned.constants()[0] == pytest.approx(3.0, abs=1e-2)
        assert [n for n in tuned.nodes if not isinstance(n, float)] == [mul, 0]
        assert program.constants() == [0.5]

    def test_tuning_never_makes_a_program_worse(self, regression_data):
        X, _ = regression_data
        mul = get_function('mul')
        program = Program([mul, 0, 1.0])
        # The starting constant is already the best there is, so nothing beats it.
        tuned, loss = _tune_program(program, X, mse_fitness(X[:, 0]), None, max_iter=20)

        assert tuned is program
        assert loss == 0.0

    def test_tuning_gives_up_on_an_expression_that_cannot_be_scored(self, regression_data):
        X, _ = regression_data
        program = Program([get_function('mul'), 0, 1.0])

        tuned, loss = _tune_program(program, X, lambda y_pred, rows: np.inf, None, max_iter=20)

        assert tuned is program
        assert loss == np.inf

    def test_evolution_tunes_the_constants_of_its_best_expressions(self, regression_data):
        X, _ = regression_data
        target = 3.0 * X[:, 0] + 0.7

        def settings(**kwargs):
            return make_settings(max_iter=10, population_size=100, parsimony_coefficient=0.0, **kwargs)

        plain = evolve(X, mse_fitness(target), make_space(('add', 'mul')), settings(), 2)
        tuned = evolve(
            X, mse_fitness(target), make_space(('add', 'mul')), settings(n_tuned_programs=10, tuning_interval=2), 2
        )

        assert min(tuned.run_details['best_fitness']) < min(plain.run_details['best_fitness'])
        assert tuned.pareto_front[-1].loss == min(tuned.run_details['best_fitness'])

    def test_tuning_is_not_run_when_turned_off(self, regression_data):
        X, y = regression_data
        calls = []

        def fitness(y_pred, rows):
            calls.append(1)
            return float(np.mean((y_pred - y) ** 2))

        result = evolve(X, fitness, make_space(), make_settings(n_tuned_programs=0, tuning_interval=1), 0)

        assert len(calls) == sum(result.run_details['n_evaluated'])
