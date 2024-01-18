import unittest
import patsy
import numpy as np
import numpy.testing as npt
# from empulse.models import ProfLogitClassifier
from empulse.models.proflogit.proflogit import ProfLogitClassifier
from empulse.metrics import empc_score
from empulse.models.proflogit.common import load_data


class TestProfLogit(unittest.TestCase):
    path_one_predictor = "data/one-predictor.dat"
    path_two_predictors = "data/two-predictors.dat"
    path_patsy_demo_train = "data/patsy_demo_train.dat"
    path_patsy_demo_test = "data/patsy_demo_test.dat"


    # Class variable
    data = {
        "X1": [24, 19, 13, 40, 10, 27, 22, 2, 29, 15],
        "X2": [12, 3, 0, 2, 1, 0, 0, 1, 1, 1],
        "X3": [30760, 1750940, 0, 2536120, 21550, 0, 0, 635230, 849070, 83160],
        "Y": [0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    }

    def test_default_rga_kws(self):
        pfl = ProfLogitClassifier()
        target_value = {"niter": 1000, "niter_diff": 250}
        self.assertDictEqual(pfl.rga_kws, target_value)

    def test_simple_proflogit(self):
        """Use one predictor."""
        dat = np.loadtxt(self.path_one_predictor)
        y = dat[:, 0]
        x = dat[:, 1]
        m = np.mean(x)
        s = np.std(x)
        z = (x - m) / s
        z = z.reshape(len(x), 1)
        dmat = np.c_[np.ones_like(z), z]
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 10, "disp": False, "random_state": 42},
        )
        pfl.fit(X=dmat, y=y)
        npt.assert_array_almost_equal(pfl.rga.result.x, [-0.29866826, 0.00052175])
        self.assertAlmostEqual(pfl.rga.result.fun, 5.97213086792)
        empc_score = pfl.score(X=dmat, y_true=y)
        self.assertAlmostEqual(empc_score, 5.9721830425745903)

    def test_proflogitccp_access_fit_method_directly(self):
        dat = np.loadtxt(self.path_one_predictor)
        y = dat[:, 0]
        x = dat[:, 1]
        m = np.mean(x)
        s = np.std(x)
        z = (x - m) / s
        z = z.reshape(len(x), 1)
        dmat = np.c_[np.ones_like(z), z]
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 10, "disp": False, "random_state": 42},
        ).fit(X=dmat, y=y)
        npt.assert_array_almost_equal(pfl.rga.result.x, [-0.29866826, 0.00052175])
        self.assertAlmostEqual(pfl.rga.result.fun, 5.97213086792)
        empc_score = pfl.score(X=dmat, y_true=y)
        self.assertAlmostEqual(empc_score, 5.9721830425745903)

    # def test_simple_proflogit_str_label(self):
    #     """Use one predictor."""
    #     dat = np.loadtxt("../data/one-predictor.dat")
    #     y = ["yes" if v == 1 else "no" for v in dat[:, 0]]
    #     x = dat[:, 1]
    #     m = np.mean(x)
    #     s = np.std(x)
    #     z = (x - m) / s
    #     z = z.reshape(len(x), 1)
    #     dmat = np.c_[np.ones_like(z), z]
    #     pfl = ProfLogitCCP(
    #         rga_kws={"niter": 10, "disp": False, "random_state": 42},
    #         empc_kws={"case_label": "yes"},
    #     )
    #     pfl.fit(X=dmat, y=y)
    #     npt.assert_array_almost_equal(pfl.rga.res.x, [-0.29866826, 0.00052175])
    #     self.assertAlmostEqual(pfl.rga.res.fun, 5.97213086792)
    #     empc_score = pfl.score(X=dmat, y_true=y)
    #     self.assertAlmostEqual(empc_score, 5.9721830425745903)

    def test_raise_key_error_reg_kws(self):
        """
        Should raise KeyError when reg_kws keys are incorrect.
        """
        with self.assertRaises(KeyError):
            ProfLogitClassifier(
                rga_kws={"niter": 1},
                reg_kws={"non-sense": True},  # Incorrect
                intercept=False,
            )

    def test_raise_assertion_error_reg_kws(self):
        """
        Should raise AssertionError when reg_kws is no dict.
        """
        with self.assertRaises(AssertionError):
            ProfLogitClassifier(
                rga_kws={"niter": 1}, reg_kws=[], intercept=False,  # Incorrect
            )

    # def test_raise_assertion_error_empc_kws(self):
    #     """
    #     Should raise AssertionError when empc_kws is no dict.
    #     """
    #     with self.assertRaises(AssertionError):
    #         ProfLogitCCP(
    #             rga_kws={"niter": 1}, empc_kws=[], intercept=False,  # Incorrect
    #         )

    # def test_raise_type_error_empc_kws_wrong_param(self):
    #     """
    #     Should raise TypeError an unexpected keyword argument in empc_kws.
    #     """
    #     x = np.random.rand(5, 2)
    #     y = np.random.randint(2, size=x.shape[0])
    #     with self.assertRaises(TypeError):
    #         ProfLogitCCP(
    #             rga_kws={"niter": 10},
    #             empc_kws={"phi": 0.1},  # Incorrect
    #             intercept=False,
    #         ).fit(x, y)

    def test_raise_assertion_error_rga_kws(self):
        """
        Should raise AssertionError when rga_kws is no dict.
        """
        with self.assertRaises(AssertionError):
            ProfLogitClassifier(
                rga_kws=[], intercept=False,  # Incorrect
            )

    def test_raise_type_error_rga_kws_wrong_param(self):
        """
        Should raise TypeError an unexpected keyword argument in rga_kws.
        """
        x = np.random.rand(5, 2)
        y = np.random.randint(2, size=x.shape[0])
        with self.assertRaises(TypeError):
            ProfLogitClassifier(
                rga_kws={"maxiter": 10}, intercept=False,  # Incorrect
            ).fit(x, y)

    def test_proflogit_with_patsy_demo_data(self):
        """Test on simple categorical/numerical demo data from patsy."""
        # demo_data: returns a dict
        # Categorical variables are returned as a list of strings.
        # Numerical data sampled from a normal distribution (fixed seed)
        rng = np.random.RandomState(42)
        data = patsy.demo_data("a", "b", "x1", "x2", nlevels=3)
        y = rng.randint(2, size=len(data["a"]))
        # dmatrix: to create the design matrix alone (no left-hand side)
        # Important that `data` can be indexed like a Python dictionary,
        # e.g., `data[varname]`. It can also be a pandas.DataFrame
        X = patsy.dmatrix("a + b + x1 + x2", data)
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 10, "disp": False, "random_state": 42},
        )
        pfl.fit(X, y)
        npt.assert_array_almost_equal(
            pfl.rga.result.x,
            [
                0.26843982,  # Intercept
                0.0,  # Categorical variable 'a' - level a2
                -0.21947001,  # Categorical variable 'a' - level a3
                0.12036944,  # Categorical variable 'b' - level b2
                0.0,  # Categorical variable 'b' - level b3
                -0.47514314,  # Numeric variable 'x1'
                -0.08812723,  # Numeric variable 'x2'
            ],
        )
        self.assertAlmostEqual(pfl.rga.result.fun, 12.3541334628)
        empc_score = pfl.score(X, y)
        self.assertAlmostEqual(empc_score, 12.4444444445)

    def test_proflogit_with_patsy_demo_data_no_intercept(self):
        """
        Test on simple demo data from patsy w/o intercept.
        """
        # demo_data: returns a dict
        # categorical variables are returned as a list of strings.
        # Numerical data sampled from a normal distribution (fixed seed)
        rng = np.random.RandomState(42)
        data = patsy.demo_data("a", "b", "x1", "x2", nlevels=3)
        y = rng.randint(2, size=len(data["a"]))
        # dmatrix: to create the design matrix alone (no left-hand side)
        X = patsy.dmatrix("a + b + x1 + x2 - 1", data)
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 10, "disp": False, "random_state": 42},
            intercept=False,
        )
        pfl.fit(X, y)
        npt.assert_array_almost_equal(
            pfl.rga.result.x,
            [0.27466536, 0.0, -0.24030505, 0.0, 0.0, -0.82215168, 0.0],
        )
        self.assertAlmostEqual(pfl.rga.result.fun, 12.310732234783764)
        empc_score = pfl.score(X, y)
        self.assertAlmostEqual(empc_score, 12.4444444445)

    def test_proflogit_with_patsy_build_in_transformation_functions(self):
        """Test patsy build-in transformation functions."""
        # demo_data: returns a dict
        # Categorical variables are returned as a list of strings.
        # Numerical data sampled from a normal distribution (fixed seed)
        rng = np.random.RandomState(42)
        data = patsy.demo_data("a", "b", "x1", "x2", nlevels=3)
        y = rng.randint(2, size=len(data["a"]))
        # dmatrix: to create the design matrix alone (no left-hand side)
        # Important that `data` can be indexed like a Python dictionary,
        # e.g., `data[varname]`. It can also be a pandas.DataFrame
        # Strings and booleans are treated as categorical variables, where
        # the first level is the baseline.
        X = patsy.dmatrix("a + b + standardize(x1) + standardize(x2)", data,)
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 10, "disp": False, "random_state": 42},
        )
        pfl.fit(X, y)
        npt.assert_array_almost_equal(
            pfl.rga.result.x,
            [0.71321495, 0.0, -0.6815996, 0.0, 0.0, -0.92505635, 0.0],
        )
        self.assertAlmostEqual(pfl.rga.result.fun, 12.2837788495)
        empc_score = pfl.score(X, y)
        self.assertAlmostEqual(empc_score, 12.4444444445)

    # def test_load_data(self):
    #     ynm = "y"
    #     yix = 0
    #     X0, _ = load_data("../data/patsy_demo_train.dat")
    #     y0 = X0[ynm]
    #     del X0[ynm]
    #     X1, y1 = load_data("../data/patsy_demo_train.dat", ynm)
    #     X2, y2 = load_data("../data/patsy_demo_train.dat", yix)
    #     d = [X0, X1, X2]
    #     for dix, d0 in enumerate(d):
    #         for d1 in d[dix + 1 :]:
    #             for (k0, v0), (k1, v1) in zip(d0.items(), d1.items()):
    #                 self.assertEqual(k0, k1)
    #                 if isinstance(v0, list) and isinstance(v1, list):
    #                     self.assertListEqual(v0, v1)
    #                 else:
    #                     npt.assert_array_almost_equal(v0, v1)
    #     npt.assert_array_equal(y0, y1)
    #     npt.assert_array_equal(y0, y2)
    #     npt.assert_array_equal(y1, y2)

    def test_proflogit_patsy_demo_data(self):
        X_train, y_train = load_data(self.path_patsy_demo_train, "y")
        X_test, y_test = load_data(self.path_patsy_demo_test, 0)
        pfl = ProfLogitClassifier(rga_kws={"niter": 5, "random_state": 42},)
        pfl.fit(X_train, y_train)
        self.assertEqual(pfl.formula, "c + standardize(x)")
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)
        empc = pfl.score(X_test, y_test)
        self.assertAlmostEqual(empc, 25.5035418741)

    def test_proflogit_patsy(self):
        yix = 0
        X, y = load_data(self.path_two_predictors, yix, False)
        form = " + ".join(
            [
                "standardize(f{})".format(cix)
                for cix in range(len(X) + 1)
                if cix != yix
            ]
        )
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 5, "disp": False, "random_state": 42},
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.formula, form)
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)
        y_score = pfl.predict_proba(X)
        empc1 = empc_score(y, y_score)
        empc2 = pfl.score(X, y)
        self.assertAlmostEqual(empc1, empc2)

    def test_proflogit_on_class_data(self):
        data = self.data
        ynm = "Y"
        X = {k: v for k, v in data.items() if k != ynm}
        y = data[ynm]
        form = " + ".join(["standardize({})".format(k) for k in X if k != ynm])
        pfl = ProfLogitClassifier(
            rga_kws={"nfev": 500, "disp": False, "random_state": 2017},
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.formula, form)
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)
        empc = pfl.score(X, y)
        self.assertAlmostEqual(empc, 28.0)

    def test_proflogit_no_intercept_no_formula(self):
        ynm = "y"
        X_train, y_train = load_data(self.path_patsy_demo_train, ynm)
        X_test, y_test = load_data(self.path_patsy_demo_test, 0)
        pfl = ProfLogitClassifier(
            rga_kws={"niter": 5, "disp": False, "random_state": 42},
            intercept=False,
        )
        pfl.fit(X_train, y_train)
        self.assertEqual(pfl.formula, "c + standardize(x) - 1")
        self.assertFalse("Intercept" in pfl.design_info.column_names)
        self.assertFalse(pfl.intercept)
        empc = pfl.score(X_test, y_test)
        self.assertAlmostEqual(empc, 25.5035418741)

    def test_proflogit_no_intercept_through_formula(self):
        ynm = "y"
        X_train, y_train = load_data(self.path_patsy_demo_train, ynm)
        X_test, y_test = load_data(self.path_patsy_demo_test, 0)
        pfl = ProfLogitClassifier(
            formula="c + standardize(x) - 1",
            rga_kws={"niter": 5, "disp": False, "random_state": 42},
        )
        pfl.fit(X_train, y_train)
        self.assertEqual(pfl.formula, "c + standardize(x) - 1")
        self.assertFalse("Intercept" in pfl.design_info.column_names)
        self.assertFalse(pfl.intercept)
        empc = pfl.score(X_test, y_test)
        self.assertAlmostEqual(empc, 25.5035418741)

    def test_proflogit_on_class_data_no_intercept(self):
        data = self.data
        ynm = "Y"
        X = {k: v for k, v in data.items() if k != ynm}
        y = data[ynm]
        form = " + ".join(["standardize({})".format(k) for k in X if k != ynm])
        form += " - 1"
        pfl = ProfLogitClassifier(
            rga_kws={"nfev": 500, "disp": False, "random_state": 2017},
            intercept=False,
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.formula, form)
        self.assertFalse("Intercept" in pfl.design_info.column_names)
        self.assertFalse(pfl.intercept)
        empc = pfl.score(X, y)
        self.assertAlmostEqual(empc, 28.0)

    def test_proflogit_reg_kws(self):
        data = self.data
        ynm = "Y"
        X = {k: v for k, v in data.items() if k != ynm}
        y = data[ynm]
        form = " + ".join(["standardize({})".format(k) for k in X if k != ynm])
        pfl = ProfLogitClassifier(
            reg_kws={
                "lambda": 0.01,  # Arbitrary value, needs to be tuned
                "alpha": 1.0,  # By default, applying lasso penalty
                "soft-thd": True,  # Apply soft-thresholding
            },
            rga_kws={"nfev": 500, "random_state": 2017},
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.formula, form)
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)
        empc = pfl.score(X, y)
        self.assertAlmostEqual(empc, 28.0)

    def test_proflogit_reg_kws_change_only_lambda(self):
        data = self.data
        ynm = "Y"
        X = {k: v for k, v in data.items() if k != ynm}
        y = data[ynm]
        form = " + ".join(["standardize({})".format(k) for k in X if k != ynm])
        pfl = ProfLogitClassifier(
            reg_kws={"lambda": 0.01},  # Arbitrary value, needs to be tuned
            rga_kws={"nfev": 500, "random_state": 2017},
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.formula, form)
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)
        empc = pfl.score(X, y)
        self.assertAlmostEqual(empc, 28.0)

    # def test_proflogit_empc_kws(self):
    #     data = self.data
    #     ynm = "Y"
    #     X = {k: v for k, v in data.items() if k != ynm}
    #     y = data[ynm]
    #     pfl = ProfLogitClassifier(
    #         empc_kws={
    #             "alpha": 6,  # Alpha parameter of unimodal beta (alpha > 1)
    #             "beta": 14,  # Beta parameter of unimodal beta (beta > 1)
    #             "clv": 200,  # Constant CLV per retained customer (clv > d)
    #             "d": 10,  # Constant cost of retention offer (d > 0)
    #             "f": 2,  # Constant cost of contact (f > 0)
    #         },
    #         rga_kws={"nfev": 500, "random_state": 2017},
    #     )
    #     pfl.fit(X, y)
    #     empc = pfl.score(X, y)
    #     self.assertAlmostEqual(empc, 27.500000004757432)

    def test_default_bounds(self):
        data = self.data
        ynm = "Y"
        X = {k: v for k, v in data.items() if k != ynm}
        y = data[ynm]
        for t in [(-3, 3, 28.0), (-5, 5, 28.0)]:
            pfl = ProfLogitClassifier(
                rga_kws={"nfev": 500, "random_state": 2017},
                default_bounds=(t[0], t[1]),
            )
            pfl.fit(X, y)
            empc = pfl.score(X, y)
            self.assertAlmostEqual(empc, t[2])

    def test_bounds(self):
        data = self.data
        ynm = "Y"
        X = {k: v for k, v in data.items() if k != ynm}
        y = data[ynm]
        # n_params = intercept + #n_num_variables +
        # #(n_levels - 1) per cat_variable
        # For data, n_params = 4
        b = [
            (-3, 3),  # Intercept
            (-1, 1),  # X1
            (-3, 4),  # X2
            (-9, 0),  # X3
        ]
        pfl = ProfLogitClassifier(
            rga_kws={"bounds": b, "nfev": 500, "random_state": 2017},
        )
        pfl.fit(X, y)
        empc = pfl.score(X, y)
        self.assertAlmostEqual(empc, 25.8003164898)

    def test_raise_error_variable_not_in_formula(self):
        X, y = load_data(self.path_patsy_demo_train, "y")
        with self.assertRaises(patsy.PatsyError):
            ProfLogitClassifier("b + standardize(x)").fit(X, y)  # 'b' not in X

    def test_proflogit_access_fit_method_directly(self):
        X_train, y_train = load_data(self.path_patsy_demo_train, "y")
        X_test, y_test = load_data(self.path_patsy_demo_test, 0)
        pfl = ProfLogitClassifier(
            rga_kws={"niter_diff": 3, "disp": False, "random_state": 42},
        ).fit(X_train, y_train)
        self.assertEqual(pfl.formula, "c + standardize(x)")
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)
        empc = pfl.score(X_test, y_test)
        self.assertAlmostEqual(empc, 25.0827084152)

    def test_proflogit_nfev_correct_small(self):
        X, y = load_data(self.path_patsy_demo_train, "y")
        n_fev = 3
        pfl = ProfLogitClassifier(
            rga_kws={"nfev": n_fev, "disp": False, "random_state": 42},
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.rga.result.nfev, n_fev)
        self.assertEqual(pfl.formula, "c + standardize(x)")
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)

    def test_proflogit_nfev_correct_large(self):
        X, y = load_data(self.path_patsy_demo_train, "y")
        n_fev = 43
        pfl = ProfLogitClassifier(
            rga_kws={"nfev": n_fev, "disp": False, "random_state": 42},
        )
        pfl.fit(X, y)
        self.assertEqual(pfl.rga.result.nfev, n_fev)
        self.assertEqual(pfl.formula, "c + standardize(x)")
        self.assertTrue("Intercept" in pfl.design_info.column_names)
        self.assertTrue(pfl.intercept)


if __name__ == "__main__":
    unittest.main()