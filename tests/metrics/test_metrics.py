import unittest
import random
from itertools import islice
from typing import Generator

import numpy as np


# wrap with empty class to avoid unittest discovery
class BaseTestMetric:
    class TestMetric(unittest.TestCase):
        path_predictions = "tests/data/predictions.dat"
        parameters = {}
        expected_values = {}

        @property
        def metric(self):
            return ...

        def assertAlmostEqualMetric(self, generated: tuple[float, float], expected: tuple[float, float]):
            """Assert that the generated value and threshold is equal to the expected value and threshold."""
            self.assertAlmostEqual(generated[0], expected[0])
            self.assertAlmostEqual(generated[1], expected[1])

        def test_perfect_prediction(self):
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [0, 1] * 10),
                self.expected_values["perfect_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [1, 0] * 10),
                self.expected_values["perfect_prediction"]
            )

        def test_incorrect_prediction(self):
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [1, 0] * 10),
                self.expected_values["incorrect_prediction"]
            )

        def test_half_correct_prediction(self):
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [1, 1] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [1, 1] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [0, 0] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [0, 0] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [0.5, 0.5] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [0.5, 0.5] * 10),
                self.expected_values["half_correct_prediction"]
            )

        def test_arraylikes(self):
            import pandas as pd
            self.assertAlmostEqualMetric(
                self.metric((0, 1) * 10, (0, 1) * 10),
                self.expected_values["perfect_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric(pd.Series([0, 1] * 10), pd.Series([0, 1] * 10)),
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
        def stochastic_metric(self):
            return ...

        @property
        def deterministic_metric(self):
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


if __name__ == "__main__":
    unittest.main()
