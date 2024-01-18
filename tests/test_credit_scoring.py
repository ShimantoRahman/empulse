import unittest
import numpy as np
from empulse.metrics import empcs, mpcs


class TestEMPCS(unittest.TestCase):
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
    true_score = 0.132364447
    true_threshold = 0.35410300000000006
    params = {
        "success_rate": 0.55,
        "default_rate": 0.1,
        "roi": 0.2644,
    }
    path_predictions = "data/predictions.dat"

    def test_defaults(self):
        score, threshold = empcs(y_true=self.y_true, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_y_true_labels_float(self):
        y = [float(l) for l in self.y_true]
        score, threshold = empcs(y_true=y, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_percent_prediction1(self):
        """Perfect prediction (v1)."""
        y_true = [0, 1]
        y_score = [0, 1]
        score, threshold = empcs(y_true, y_score)
        self.assertAlmostEqual(score, 0.1375)
        self.assertAlmostEqual(threshold, 0.22499999999999998)

    def test_perfect_prediction2(self):
        """Perfect prediction (v2)."""
        y_true = [1, 0]
        y_score = [1, 0]
        score, threshold = empcs(y_true, y_score)
        self.assertAlmostEqual(score, 0.1375)
        self.assertAlmostEqual(threshold, 0.22499999999999998)

    def test_incorrect_prediction(self):
        """Completely incorrect prediction."""
        y_true = [0, 1]
        y_score = [1, 0]
        score, threshold = empcs(y_true, y_score)
        self.assertAlmostEqual(score, 0.08412689400000001)
        self.assertAlmostEqual(threshold, 0.35746)

    def test_half_correct_prediction(self):
        """Half correct prediction."""
        y_true = [1, 0]
        y_score = [1, 1]
        score, threshold = empcs(y_true, y_score)
        self.assertAlmostEqual(score, 0.08412689400000001)
        self.assertAlmostEqual(threshold, 0.35746)

    def test_diagonal_roc_curve(self):
        """Generate QhullError, i.e., ROCCH is equal to diagonal line."""
        y_true = [1, 0]
        y_score = [0.5, 0.5]
        score, threshold = empcs(y_true, y_score)
        self.assertAlmostEqual(score, 0.08412689400000001)
        self.assertAlmostEqual(threshold, 0.35746)

    def test_all_y_true_0(self):
        """len(np.unique(y_true)) != 2 (all zero)"""
        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            empcs(y_true, y_score)

    def test_all_y_true_1(self):
        """len(np.unique(y_true)) != 2 (all one)"""
        y_true = [1, 1]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            empcs(y_true, y_score)

    def test_unequal_array_lengths(self):
        y = self.y_true[1:]
        yhat = self.y_pred
        with self.assertRaises(ValueError):
            empcs(y_true=y, y_pred=yhat)

    def test_different_parameters_on_realistic_data(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_score = dat[:, 1]

        tests = [
            (
                {
                    "success_rate": 0.55,
                    "default_rate": 0.1,
                    "roi": 0.2644,
                },
                (0.06275887634143887, 0.17452781458718136),
            ),
            (
                {
                    "success_rate": 0.7,  # changed
                    "default_rate": 0.1,
                    "roi": 0.2644,
                },
                (0.04653003664242532, 0.11873470465098149),
            ),
            (
                {
                    "success_rate": 0.55,
                    "default_rate": 0.01,  # changed
                    "roi": 0.2644,
                },
                (0.05009375482148049, 0.16809417828645784),
            ),
            (
                {
                    "success_rate": 0.55,
                    "default_rate": 0.1,
                    "roi": 0.1,  # changed
                },
                (0.07738088359641128, 0.25536875341114096),
            ),
            (
                {
                    "success_rate": 0.55,
                    "default_rate": 0.1,
                    "roi": 0.5,  # changed
                },
                (0.05012038050857254, 0.15223341009240662),
            ),
            (
                {
                    "success_rate": 0.45,  # changed
                    "default_rate": 0.2,  # changed
                    "roi": 0.2644,
                },
                (0.08765046005184612, 0.2188717059898963),
            ),
            (
                {
                    "success_rate": 0.01,  # changed
                    "default_rate": 0.7,  # changed
                    "roi": 0.7,  # changed
                },
                (0.14636925360203037, 0.38412016747189753),
            ),
        ]

        for params, (true_score, true_threshold) in tests:
            score, threshold = empcs(y_true, y_score, **params)
            self.assertAlmostEqual(score, true_score)
            self.assertAlmostEqual(threshold, true_threshold)


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
        score, threshold = mpcs(y_true=self.y_true, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_y_true_labels_float(self):
        y = [float(l) for l in self.y_true]
        score, threshold = mpcs(y_true=y, y_pred=self.y_pred, **self.params)
        self.assertAlmostEqual(score, self.true_score)
        self.assertAlmostEqual(threshold, self.true_threshold)

    def test_percent_prediction1(self):
        """Perfect prediction (v1)."""
        y_true = [0, 1]
        y_score = [0, 1]
        score, threshold = mpcs(y_true, y_score)
        self.assertAlmostEqual(score, 3725.0)
        self.assertAlmostEqual(threshold, 0.5)

    def test_perfect_prediction2(self):
        """Perfect prediction (v2)."""
        y_true = [1, 0]
        y_score = [1, 0]
        score, threshold = mpcs(y_true, y_score)
        self.assertAlmostEqual(score, 3725.0)
        self.assertAlmostEqual(threshold, 0.5)

    def test_incorrect_prediction(self):
        """Completely incorrect prediction."""
        y_true = [0, 1]
        y_score = [1, 0]
        score, threshold = mpcs(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_half_correct_prediction(self):
        """Half correct prediction."""
        y_true = [1, 0]
        y_score = [1, 1]
        score, threshold = mpcs(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_diagonal_roc_curve(self):
        """Generate QhullError, i.e., ROCCH is equal to diagonal line."""
        y_true = [1, 0]
        y_score = [0.5, 0.5]
        score, threshold = mpcs(y_true, y_score)
        self.assertAlmostEqual(score, 3700.0)
        self.assertAlmostEqual(threshold, 1.0)

    def test_all_y_true_0(self):
        """len(np.unique(y_true)) != 2 (all zero)"""
        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            mpcs(y_true, y_score)

    def test_all_y_true_1(self):
        """len(np.unique(y_true)) != 2 (all one)"""
        y_true = [1, 1]
        y_score = [0.25, 0.75]
        with self.assertRaises(ValueError):
            mpcs(y_true, y_score)

    def test_unequal_array_lengths(self):
        y = self.y_true[1:]
        yhat = self.y_pred
        with self.assertRaises(ValueError):
            mpcs(y_true=y, y_pred=yhat)

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
            score, threshold = mpcs(y_true, y_score, **params)
            self.assertAlmostEqual(score, true_score)
            self.assertAlmostEqual(threshold, true_threshold)

    def test_change_commission_direct_selling(self):
        dat = np.loadtxt(self.path_predictions)
        y_true = dat[:, 0]
        y_pred = dat[:, 1]

        # change commission
        score1, threshold1 = mpcs(y_true, y_pred, direct_selling=1, commission=0.2)

        # do not change commission
        score2, threshold2 = mpcs(y_true, y_pred, direct_selling=1, commission=0.1)

        self.assertAlmostEqual(score1, score2)
        self.assertAlmostEqual(threshold1, threshold2)


class TestEMPAMPA(unittest.TestCase):
    path_predictions = "data/predictions.dat"

    def test_empcs_greater_or_equal_than_mpcs(self):
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
            empcs_score, _ = empcs(y_true, y_score, **params)
            params['contribution'] = params['alpha'] / params['beta']
            del params['alpha']
            del params['beta']
            mpcs_score, _ = mpcs(y_true, y_score, **params)
            with self.subTest():
                self.assertGreaterEqual(empcs_score, mpcs_score)


if __name__ == "__main__":
    unittest.main()
