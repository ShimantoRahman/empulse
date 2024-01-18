import unittest
import random
from itertools import islice
from typing import Callable, Generator, Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from empulse.metrics import empc, mpc


class Metric(Protocol):
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs: Any) -> tuple[float, float]:
        ...


# wrap with empty class to avoid unittest discovery
class BaseTestMetric:
    class TestMetric(unittest.TestCase):
        path_predictions = "data/predictions.dat"
        parameters = {}
        expected_values = {}

        @property
        def metric(self) -> Metric:
            return ...

        def assertAlmostEqualMetric(self, generated: tuple[float, float], expected: tuple[float, float]):
            """Assert that the generated value and threshold is equal to the expected value and threshold."""
            self.assertAlmostEqual(generated[0], expected[0])
            self.assertAlmostEqual(generated[1], expected[1])

        def test_perfect_prediction(self):
            self.assertAlmostEqualMetric(self.metric([0, 1], [0, 1]), self.expected_values["perfect_prediction"])
            self.assertAlmostEqualMetric(self.metric([1, 0], [1, 0]), self.expected_values["perfect_prediction"])

        def test_incorrect_prediction(self):
            self.assertAlmostEqualMetric(self.metric([0, 1], [1, 0]), self.expected_values["incorrect_prediction"])

        def test_half_correct_prediction(self):
            self.assertAlmostEqualMetric(self.metric([0, 1], [1, 1]), self.expected_values["half_correct_prediction"])
            self.assertAlmostEqualMetric(self.metric([1, 0], [1, 1]), self.expected_values["half_correct_prediction"])
            self.assertAlmostEqualMetric(self.metric([0, 1], [0, 0]), self.expected_values["half_correct_prediction"])
            self.assertAlmostEqualMetric(self.metric([1, 0], [0, 0]), self.expected_values["half_correct_prediction"])
            self.assertAlmostEqualMetric(
                self.metric([0, 1], [0.5, 0.5]),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0], [0.5, 0.5]),
                self.expected_values["half_correct_prediction"]
            )

        def test_arraylikes(self):
            import pandas as pd
            self.assertAlmostEqualMetric(self.metric((0, 1), (0, 1)), self.expected_values["perfect_prediction"])
            self.assertAlmostEqualMetric(
                self.metric(pd.Series([0, 1]), pd.Series([0, 1])),
                self.expected_values["perfect_prediction"]
            )

        def test_all_y_true_same_value(self):
            with self.assertRaises(ValueError):
                self.metric([0, 0], [0.25, 0.75])
            with self.assertRaises(ValueError):
                self.metric([1, 1], [0.25, 0.75])

        def test_missing_y_true(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, np.nan], [0.25, 0.75, 0.5])
            with self.assertRaises(TypeError):  # type error because array will be of type object
                self.metric([1, 0, None], [0.25, 0.75, 0.5])

        def test_missing_y_score(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, 1], [0.25, 0.75, np.nan])
            with self.assertRaises(TypeError):  # type error because array will be of type object
                self.metric([1, 0, 1], [0.25, 0.75, None])

        def test_infinite_y_true(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, np.inf], [0.25, 0.75, 0.5])

        def test_infinite_y_score(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, 1], [0.25, 0.75, np.inf])

        def test_non_numeric_y_true(self):
            with self.assertRaises(TypeError):
                self.metric(["a", "b"], [0.25, 0.75])

        def test_non_numeric_y_score(self):
            with self.assertRaises(TypeError):
                self.metric([0, 1], ["a", "b"])

        def test_unequal_array_lengths(self):
            with self.assertRaises(ValueError):
                self.metric([0, 0, 1], [0.25, 0.75])
            with self.assertRaises(ValueError):
                self.metric([0, 1], [0.25, 0.5, 0.75])

        def test_different_parameters(self):
            data = np.loadtxt(self.path_predictions)
            y_true = data[:, 0]
            y_score = data[:, 1]

            for params, expected in zip(self.parameters, self.expected_values["different_parameters"]):
                with self.subTest(params):
                    self.assertAlmostEqualMetric(self.metric(y_true, y_score, **params), expected)


# wrap with empty class to avoid unittest discovery
class BaseTestRelationMetrics:
    class TestRelationshipMetrics(unittest.TestCase):

        @property
        def stochastic_metric(self) -> Metric:
            return ...

        @property
        def deterministic_metric(self) -> Metric:
            return ...

        def generate_data(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
            """Generates data for the metric."""
            while True:
                N_SAMPLES = 1000
                positive_class_prior = random.random()
                n_positive_class = int(positive_class_prior * N_SAMPLES)
                if n_positive_class == 0:
                    n_positive_class = 1
                n_negative_class = N_SAMPLES - n_positive_class
                random_param = lambda x: random.random() * x
                positive_class_predictions = np.random.beta(random_param(20), random_param(20), n_positive_class)
                negative_class_predictions = np.random.beta(random_param(20), random_param(20), n_negative_class)
                predictions = np.concatenate([positive_class_predictions, negative_class_predictions])
                labels = np.concatenate([
                    np.ones(n_positive_class, dtype=np.int8),
                    np.zeros(n_negative_class, dtype=np.int8)
                ])
                yield labels, predictions

        def generate_parameters(self) -> Generator[dict, None, None]:
            """Generates parameters for the metric."""
            ...

        def to_deterministic_params(self, params: dict) -> dict:
            """Converts the parameters to deterministic values."""
            ...

        def test_deterministic_metric_lower_bound(self):
            """Test whether the deterministic metric is lower or equal than the stochastic metric."""
            for params, (y_true, y_pred) in zip(islice(self.generate_parameters(), 1000),
                                                islice(self.generate_data(), 1000)):
                stochastic_score, _ = self.stochastic_metric(y_true, y_pred, **params)
                params = self.to_deterministic_params(params)
                deterministic_score, _ = self.deterministic_metric(y_true, y_pred, **params)
                with self.subTest(params):
                    self.assertTrue(deterministic_score <= stochastic_score + 0.000001)


class TestEMPC(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"alpha": 2},
        {"beta": 5},
        {"clv": 50},
        {"incentive_cost": 100},
        {"contact_cost": 15},
        {
            "alpha": 10,
            "beta": 7,
            "clv": 300,
            "incentive_cost": 50,
            "contact_cost": 25,
        },
    ]
    expected_values = {
        "perfect_prediction": (28.0000000000391, 0.499999999728095),
        "incorrect_prediction": (22.5007912244511, 0.9991601018889205),
        "half_correct_prediction": (22.5007912244511, 0.9991601018889205),
        "different_parameters": [
            (14.9946647690104, 0.5807038600143406),
            (5.04518077557418, 0.3984799748965802),
            (30.2276738879406, 0.6595939761685953),
            (1.62729137977619, 0.3397172176773871),
            (1.95497116583216, 0.11706222855204537),
            (8.47435415117762, 0.40263339050268493),
            (23.8505727106226, 0.4360480478313282),
        ],
    }

    @property
    def metric(self) -> Metric:
        return empc

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], alpha=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], beta=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=1)  # clv < incentive_cost
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], incentive_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)


class TestMPC(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"accept_rate": 0.2},
        {"clv": 50},
        {"incentive_cost": 100},
        {"contact_cost": 15},
        {
            "accept_rate": 0.9,
            "clv": 300,
            "incentive_cost": 50,
            "contact_cost": 25,
        },
    ]
    expected_values = {
        "perfect_prediction": (28.0, 0.5),
        "incorrect_prediction": (22.5, 1.0),
        "half_correct_prediction": (22.5, 1.0),
        "different_parameters": [
            (14.814479638009047, 0.665158371040724),
            (9.02262443438914, 0.4434389140271493),
            (1.493212669683258, 0.4434389140271493),
            (1.9095022624434383, 0.12669683257918551),
            (8.316742081447961, 0.4434389140271493),
            (46.380090497737555, 0.4434389140271493),
            (23.8505727106226, 0.4360480478313282),
        ],
    }

    @property
    def metric(self) -> Metric:
        return mpc


class TestRelationChurnMetrics(BaseTestRelationMetrics.TestRelationshipMetrics):

    @property
    def stochastic_metric(self) -> Metric:
        return empc

    @property
    def deterministic_metric(self) -> Metric:
        return mpc

    def generate_parameters(self) -> Generator[dict, None, None]:
        """Generates parameters for the metric."""
        while True:
            incentive_cost = random.uniform(0, 500)
            contact_cost = random.uniform(0, 100)
            clv = random.uniform(0, 1000)
            if clv < contact_cost + incentive_cost:
                clv = contact_cost + incentive_cost + 1
            yield {
                "alpha": random.uniform(0.00001, 100),
                "beta": random.uniform(0.00001, 100),
                "clv": clv,
                "incentive_cost": incentive_cost,
                "contact_cost": contact_cost,
            }

    def to_deterministic_params(self, params: dict) -> dict:
        """Converts the parameters to deterministic values."""
        deterministic_params = params.copy()
        deterministic_params["accept_rate"] = params['alpha'] / (params['alpha'] + params['beta'])
        del deterministic_params["alpha"]
        del deterministic_params["beta"]
        return deterministic_params


class TestEMPA(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"alpha": 10},
        {"beta": 0.02},
        {"contact_cost": 100},
        {"sales_cost": 1000},
        {"direct_selling": 0.5},
        {
            "direct_selling": 0.0,
            "commission": 0.5
        },
        {
            "alpha": 10,
            "beta": 0.0012,
            "contact_cost": 80,
            "sales_cost": 400,
            "direct_selling": 0.3,
            "commission": 0.2,
        },
    ]
    expected_values = {
        "perfect_prediction": (28.0000000000391, 0.499999999728095),
        "incorrect_prediction": (22.5007912244511, 0.9991601018889205),
        "half_correct_prediction": (22.5007912244511, 0.9991601018889205),
        "different_parameters": [
            (14.9946647690104, 0.5807038600143406),
            (5.04518077557418, 0.3984799748965802),
            (30.2276738879406, 0.6595939761685953),
            (1.62729137977619, 0.3397172176773871),
            (1.95497116583216, 0.11706222855204537),
            (8.47435415117762, 0.40263339050268493),
            (23.8505727106226, 0.4360480478313282),
        ],
    }

    @property
    def metric(self) -> Callable:
        return empc

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], alpha=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], beta=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=1)  # clv < incentive_cost
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], incentive_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)


if __name__ == "__main__":
    unittest.main()
