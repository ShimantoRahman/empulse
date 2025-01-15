from empulse.metrics import lift_score

from .test_metrics import BaseTestMetric


class TestLift(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"fraction": 0.2},
        {"fraction": 1},
        {"fraction": 0.5},
    ]
    expected_values = {
        "perfect_prediction": 2.0,
        "incorrect_prediction": 0.0,
        "half_correct_prediction": 1.0,
        "different_parameters": [
            2.714987714987715,
            2.307739557739558,
            1.0,
            1.764742014742015,
        ],
    }

    def assertAlmostEqualMetric(self, generated: float, expected: float):
        self.assertAlmostEqual(generated, expected)

    @property
    def metric(self):
        return lift_score

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], fraction=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], fraction=2)
