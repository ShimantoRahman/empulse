import unittest
import numpy as np
from empulse.metrics import empa, mpa


class TestEMPA(unittest.TestCase):
    # Class variable
    y_true = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    y_pred = [
        0.6125478,
        0.3642710,
        0.4321361,
        0.1402911,
        0.3848959,
        0.2444155,
        0.9706413,
        0.8901728,
        0.7817814,
        0.8687518,
    ]
    true_score = 4455.000000004892
    true_threshold = 0.8999999998296313
    params = {
        "alpha": 12,
        "beta": 0.0015,
        "contact_cost": 50,
        "sales_cost": 500,
        "direct_selling": 1,
        "commission": 0.1,
    }
    path_predictions = "data/predictions.dat"

    def test_defaults(self):
        score, threshold = empa(y_true=self.y_true, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_y_true_labels_float(self):
        y = [float(l) for l in self.y_true]
        score, threshold = empa(y_true=y, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_perfect_prediction(self):
        y_true = [0, 1]
        y_score = [0, 1]
        score, threshold = empa(y_true, y_score)
        self.assertAlmostEqual(score, 3725.0000000021705)
        self.assertAlmostEqual(threshold, 0.4999999999514575)

        y_true = [1, 0]
        y_score = [1, 0]
        score, threshold = empa(y_true, y_score)
        self.assertAlmostEqual(score, 3725.0000000021705)
        self.assertAlmostEqual(threshold, 0.4999999999514575)

    def test_incorrect_prediction(self):
        """Completely incorrect prediction."""
        y_true = [0, 1]
        y_score = [1, 0]
        score, threshold = empa(y_true, y_score)
        self.assertAlmostEqual(score, 3700.000000006313)
        self.assertAlmostEqual(threshold, 0.999999999742547)

    def test_half_correct_prediction(self):
        """Half correct prediction."""
        y_true = [1, 0]
        y_score = [1, 1]
        score, threshold = empa(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_diagonal_roc_curve(self):
        """Generate QhullError, i.e., ROCCH is equal to diagonal line."""
        y_true = [1, 0]
        y_score = [0.5, 0.5]
        score, threshold = empa(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_all_y_true_0(self):
        """len(np.unique(y_true)) != 2 (all zero)"""
        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            empa(y_true, y_score)

    def test_all_y_true_1(self):
        """len(np.unique(y_true)) != 2 (all one)"""
        y_true = [1, 1]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            empa(y_true, y_score)

    def test_unequal_array_lengths(self):
        y = self.y_true[1:]
        yhat = self.y_pred
        with self.assertRaises(ValueError):
            empa(y_true=y, y_pred=yhat)

    def test_different_parameters_on_realistic_data(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_score = dat[:, 1]
        tests = [
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (2467.422967103394, 0.8774901985912619),
            ),
            (
                {
                    "alpha": 10,
                    "beta": 0.02,  # changed
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (10.257063531374918, 0.13641516097751377),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 100,  # changed
                    "sales_cost": 500,
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (2423.857763376393, 0.856980037413865),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 1000,  # changed
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (2300.009595128851, 0.8766236068510703),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 0.5,  # changed
                    "commission": 0.1,
                },
                (2417.1959571513835, 0.8775861680061278),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 0,  # changed
                    "commission": 0.5,  # changed
                },
                (1295.5538077904482, 0.8659949979925511),
            ),
            (
                {
                    "alpha": 10,  # changed
                    "beta": 0.0012,  # changed
                    "contact_cost": 80,  # changed
                    "sales_cost": 400,  # changed
                    "direct_selling": 0.3,  # changed
                    "commission": 0.2,  # changed
                },
                (2289.417003293194, 0.8673144783905945),
            ),
        ]

        for params, (true_score, true_threshold) in tests:
            score, threshold = empa(y_true, y_score, **params)
            self.assertAlmostEqual(score, true_score)
            self.assertAlmostEqual(threshold, true_threshold)

    def test_change_commission_direct_selling(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_pred = dat[:, 1]

        # change commission
        score1, threshold1 = empa(y_true, y_pred, direct_selling=1, commission=0.2)

        # do not change commission
        score2, threshold2 = empa(y_true, y_pred, direct_selling=1, commission=0.1)

        self.assertAlmostEqual(score1, score2)
        self.assertAlmostEqual(threshold1, threshold2)


class TestMPA(unittest.TestCase):
    # Class variable
    y_true = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    y_pred = [
        0.6125478,
        0.3642710,
        0.4321361,
        0.1402911,
        0.3848959,
        0.2444155,
        0.9706413,
        0.8901728,
        0.7817814,
        0.8687518,
    ]
    true_score = 4455.0
    true_threshold = 0.9
    params = {
        "contribution": 8000,
        "contact_cost": 50,
        "sales_cost": 500,
        "direct_selling": 1,
        "commission": 0.1,
    }
    path_predictions = "data/predictions.dat"

    def test_defaults(self):
        score, threshold = mpa(y_true=self.y_true, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_y_true_labels_float(self):
        y = [float(l) for l in self.y_true]
        score, threshold = mpa(y_true=y, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_percent_prediction1(self):
        """Perfect prediction (v1)."""
        y_true = [0, 1]
        y_score = [0, 1]
        score, threshold = mpa(y_true, y_score)
        self.assertAlmostEqual(score, 3725.0)
        self.assertAlmostEqual(threshold, 0.5)

    def test_perfect_prediction2(self):
        """Perfect prediction (v2)."""
        y_true = [1, 0]
        y_score = [1, 0]
        score, threshold = mpa(y_true, y_score)
        self.assertAlmostEqual(score, 3725.0)
        self.assertAlmostEqual(threshold, 0.5)

    def test_incorrect_prediction(self):
        """Completely incorrect prediction."""
        y_true = [0, 1]
        y_score = [1, 0]
        score, threshold = mpa(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_half_correct_prediction(self):
        """Half correct prediction."""
        y_true = [1, 0]
        y_score = [1, 1]
        score, threshold = mpa(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_diagonal_roc_curve(self):
        """Generate QhullError, i.e., ROCCH is equal to diagonal line."""
        y_true = [1, 0]
        y_score = [0.5, 0.5]
        score, threshold = mpa(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_all_y_true_0(self):
        """len(np.unique(y_true)) != 2 (all zero)"""
        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            mpa(y_true, y_score)

    def test_all_y_true_1(self):
        """len(np.unique(y_true)) != 2 (all one)"""
        y_true = [1, 1]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            mpa(y_true, y_score)

    def test_unequal_array_lengths(self):
        y = self.y_true[1:]
        yhat = self.y_pred
        with self.assertRaises(ValueError):
            mpa(y_true=y, y_pred=yhat)

    def test_different_parameters_on_realistic_data(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_score = dat[:, 1]

        tests = [
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (2467.420814479638, 0.8778280542986425),
            ),
            (
                {
                    "alpha": 10,
                    "beta": 0.02,  # changed
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (0.0, 0.0),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 100,  # changed
                    "sales_cost": 500,
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (2423.529411764706, 0.8778280542986425),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 1000,  # changed
                    "direct_selling": 1,
                    "commission": 0.1,
                },
                (2300.0, 0.8778280542986425),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 0.5,  # changed
                    "commission": 0.1,
                },
                (2417.1945701357467, 0.8778280542986425),
            ),
            (
                {
                    "alpha": 12,
                    "beta": 0.0015,
                    "contact_cost": 50,
                    "sales_cost": 500,
                    "direct_selling": 0,  # changed
                    "commission": 0.5,  # changed
                },
                (1295.475113122172, 0.8778280542986425),
            ),
            (
                {
                    "alpha": 10,  # changed
                    "beta": 0.0012,  # changed
                    "contact_cost": 80,  # changed
                    "sales_cost": 400,  # changed
                    "direct_selling": 0.3,  # changed
                    "commission": 0.2,  # changed
                },
                (2289.2911010558073, 0.8778280542986425),
            ),
        ]

        for params, (true_score, true_threshold) in tests:
            params['contribution'] = params['alpha'] / params['beta']
            del params['alpha']
            del params['beta']
            score, threshold = mpa(y_true, y_score, **params)
            self.assertAlmostEqual(score, true_score)
            self.assertAlmostEqual(threshold, true_threshold)

    def test_change_commission_direct_selling(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_pred = dat[:, 1]

        # change commission
        score1, threshold1 = mpa(y_true, y_pred, direct_selling=1, commission=0.2)

        # do not change commission
        score2, threshold2 = mpa(y_true, y_pred, direct_selling=1, commission=0.1)

        self.assertAlmostEqual(score1, score2)
        self.assertAlmostEqual(threshold1, threshold2)


class TestEMPAMPA(unittest.TestCase):
    path_predictions = "data/predictions.dat"

    def test_empa_greater_or_equal_than_mpa(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_score = dat[:, 1]

        test_params = [
            {
                "alpha": 12,
                "beta": 0.0015,
                "contact_cost": 50,
                "sales_cost": 500,
                "direct_selling": 1,
                "commission": 0.1,
            },
            {
                "alpha": 10,
                "beta": 0.02,  # changed
                "contact_cost": 50,
                "sales_cost": 500,
                "direct_selling": 1,
                "commission": 0.1,
            },
            {
                "alpha": 12,
                "beta": 0.0015,
                "contact_cost": 100,  # changed
                "sales_cost": 500,
                "direct_selling": 1,
                "commission": 0.1,
            },
            {
                "alpha": 12,
                "beta": 0.0015,
                "contact_cost": 50,
                "sales_cost": 1000,  # changed
                "direct_selling": 1,
                "commission": 0.1,
            },
            {
                "alpha": 12,
                "beta": 0.0015,
                "contact_cost": 50,
                "sales_cost": 500,
                "direct_selling": 0.5,  # changed
                "commission": 0.1,
            },
            {
                "alpha": 12,
                "beta": 0.0015,
                "contact_cost": 50,
                "sales_cost": 500,
                "direct_selling": 0,  # changed
                "commission": 0.5,  # changed
            },
            {
                "alpha": 10,  # changed
                "beta": 0.0012,  # changed
                "contact_cost": 80,  # changed
                "sales_cost": 400,  # changed
                "direct_selling": 0.3,  # changed
                "commission": 0.2,  # changed
            },
            {
                "alpha": 10,  # changed
                "beta": 0.0012,  # changed
                "contact_cost": 80,  # changed
                "sales_cost": 400,  # changed
                "direct_selling": 0.3,  # changed
                "commission": 0.8,  # changed
            },
        ]

        for params in test_params:
            empa_score, _ = empa(y_true, y_score, **params)
            params['contribution'] = params['alpha'] / params['beta']
            del params['alpha']
            del params['beta']
            mpa_score, _ = mpa(y_true, y_score, **params)
            self.assertGreaterEqual(empa_score, mpa_score)


if __name__ == "__main__":
    unittest.main()
