import unittest
import numpy as np
import numpy.testing as npt
from empulse.models.proflogit.rga import RGA


class TestRGA(unittest.TestCase):
    def test_rga_with_small_params_values(self):
        """Test simple 2D function with small parameter values."""
        rga = RGA(
            func=lambda x: -np.sum(x ** 2),
            bounds=[(-10, 10)] * 2,
            pop_size=10,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism=0.05,
            niter=3,
            niter_diff=np.inf,
            nfev=np.inf,
            disp=False,
            ftol=1e-4,
            random_state=101,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [0.32797255, 1.41335174])
        self.assertAlmostEqual(rga.res.fun, -2.10512912972)

    def test_simple_function(self):
        """Test simple 1D function."""
        rga = RGA(
            func=lambda x: (x ** 2 + x) * np.cos(x),
            bounds=[(-10, 10)],
            pop_size=100,
            niter=100,
            disp=False,
            random_state=None,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [6.5606], 3)
        self.assertAlmostEqual(rga.res.fun, 47.7056, 3)

    def test_differentiable_unimodal_func(self):
        """Test simple differentiable and uni-modal objective function."""

        def fobj(x, *args):
            a, b = args
            x1, x2 = x[0], x[1]
            z = x1 ** 2 + b * x2 ** 2
            fobj_val = a - np.exp(-z)
            return -fobj_val

        func_args = (10, 3)  # (a, b)
        rga = RGA(
            func=fobj,
            bounds=[(-1, 1)] * 2,
            args=func_args,
            nfev=int(5e2),
            disp=False,
            random_state=101,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [0.0] * 2, 2)
        self.assertAlmostEqual(rga.res.fun, -9.0, 3)

    def test_non_differentiable_unimodal_func(self):
        """Test simple non-differentiable and uni-modal objective function."""

        def fobj(x, *args):
            a, b = args
            x1, x2 = x[0], x[1]
            z = x1 ** 2 + b * x2 ** 2
            fobj_val = np.floor(a * (a - np.exp(-z))) / a
            return -fobj_val

        func_args = (10, 3)  # (a, b)
        rga = RGA(
            func=fobj,
            bounds=[(-1, 1)] * 2,
            args=func_args,
            niter_diff=5,
            disp=False,
            random_state=101,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [0.0] * 2, 1)
        self.assertAlmostEqual(rga.res.fun, -9.0, 3)

    def test_rotated_ellipse(self):
        """Test rotated ellipse."""

        def fobj(x):
            x1, x2 = x[0], x[1]
            fobj_val = 2 * (x1 ** 2 - x1 * x2 + x2 ** 2)
            return -fobj_val

        rga = RGA(
            func=fobj,
            bounds=[(-10, 10)] * 2,
            elitism=1,
            niter_diff=100,
            ftol=1e-12,
            disp=False,
            random_state=np.random.RandomState(101),
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [0.0] * 2, 3)
        self.assertAlmostEqual(rga.res.fun, 0.0, 4)

    def test_ackley(self):
        """Test niter_diff < niter with 2D Ackley."""

        def fobj(x, *args):
            a, b, c = args
            n = len(x)
            s1 = sum(np.power(x, 2))
            s2 = sum(np.cos(c * x))
            fobj_val = (
                    -a * np.exp(-b * np.sqrt(s1 / n))
                    - np.exp(s2 / n)
                    + a
                    + np.exp(1)
            )
            return -fobj_val

        func_args = (20, 0.2, 2)  # (a, b, c)
        n_dim = 2
        rga = RGA(
            func=fobj,
            bounds=[(-32.768, 32.768)] * n_dim,
            args=func_args,
            elitism=1,
            niter=int(1e4),
            niter_diff=250,
            disp=False,
            random_state=101,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [0.0] * n_dim, 3)
        self.assertAlmostEqual(rga.res.fun, 0.0, 3)

    def test_rastrigin(self):
        """Test niter_diff > niter with 2D Rastrigin."""

        def fobj(x, *args):
            a = args[0]
            d = len(x)
            s = np.power(x, 2) - a * np.cos(2 * np.pi * x)
            fobj_val = a * d + sum(s)
            return -fobj_val

        func_args = (10,)  # tuple(a)
        n_dim = 2
        rga = RGA(
            func=fobj,
            bounds=[(-5.12, 5.12)] * n_dim,
            args=func_args,
            niter=100,
            niter_diff=250,
            disp=False,
            random_state=101,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [0.0] * n_dim, 3)
        self.assertAlmostEqual(rga.res.fun, 0.0, 3)

    def test_easom(self):
        """Test nfev < niter with 2D Easom."""

        def fobj(x):
            x1, x2 = x[0], x[1]
            fobj_val = (
                    -np.cos(x1)
                    * np.cos(x2)
                    * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2)
            )
            return -fobj_val

        n_dim = 2
        rga = RGA(
            func=fobj,
            bounds=[(-100, 100)] * n_dim,
            niter=int(1e4),
            nfev=int(5e3),
            disp=False,
            random_state=101,
        )
        rga.solve()
        npt.assert_array_almost_equal(rga.res.x, [np.pi] * n_dim, 2)
        self.assertAlmostEqual(rga.res.fun, 1.0, 4)

    def test_raise_assertion_error_seed(self):
        """
        Should raise AssertionError when niter, niter_diff, and nfev not given.
        """
        with self.assertRaises(AssertionError):
            RGA(
                func=lambda x: (x ** 2 + x) * np.cos(x), bounds=[(-10, 10)],
            )

    def test_raise_value_error_seed(self):
        """Should raise ValueError when seed is incorrectly specified."""
        with self.assertRaises(ValueError):
            RGA(
                func=lambda x: (x ** 2 + x) * np.cos(x),
                bounds=[(-10, 10)],
                niter=10,
                random_state="seed_value",
            )


if __name__ == "__main__":
    unittest.main()
