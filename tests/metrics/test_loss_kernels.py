"""The compiled logistic loss kernels match their definition, including where the probability saturates."""

import mpmath
import numpy as np
import pytest
import sympy
from scipy.special import expit

from empulse.metrics import Cost, CostMatrix, Metric
from empulse.metrics._loss import cy_boost_grad_hess, cy_logit_gradient, cy_logit_loss, cy_logit_loss_gradient

RNG = np.random.default_rng(0)


def _reference(weights, features, grad_const, loss_const1, loss_const2, l1_weight, l2_weight, start_coef):
    probability = expit(features @ weights)
    coefficients = weights[start_coef:]
    loss = np.mean(probability * loss_const1 + (1 - probability) * loss_const2)
    loss += l1_weight * np.abs(coefficients).sum() + 0.5 * l2_weight * (coefficients**2).sum()
    gradient = grad_const.T @ (probability * (1 - probability)) / len(probability)
    gradient[start_coef:] += l1_weight * np.sign(coefficients) + l2_weight * coefficients
    return loss, gradient


def _exact_expit_derivatives(margin):
    """Return expit'(margin) and |expit''(margin)| to double precision, from 50-digit arithmetic."""
    with mpmath.workdps(50):
        # Not p * (1 - p): at any fixed precision, 1 - p rounds to 0 once the margin is large enough.
        exponential = mpmath.exp(-abs(mpmath.mpf(float(margin))))
        slope = exponential / (1 + exponential) ** 2
        return float(slope), float((1 - exponential) / (1 + exponential) * slope)


# Fewer than 128 samples take the C library's exp, more take numpy's.
@pytest.mark.parametrize(('n_samples', 'n_features'), [(1, 1), (50, 3), (127, 4), (128, 4), (2000, 20)])
@pytest.mark.parametrize(('l1_weight', 'l2_weight', 'start_coef'), [(0.0, 0.0, 1), (0.1, 0.2, 1), (0.3, 0.0, 0)])
def test_logit_kernels_match_their_definition(n_samples, n_features, l1_weight, l2_weight, start_coef):
    features = RNG.normal(size=(n_samples, n_features))
    weights = RNG.normal(size=n_features)
    grad_const = features * RNG.normal(size=(n_samples, 1))
    loss_const1, loss_const2 = RNG.normal(size=(2, n_samples))
    penalty = (l1_weight, l2_weight, start_coef)

    expected_loss, expected_gradient = _reference(weights, features, grad_const, loss_const1, loss_const2, *penalty)
    loss, gradient = cy_logit_loss_gradient(weights, features, grad_const, loss_const1, loss_const2, *penalty)
    assert loss == pytest.approx(expected_loss, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-15)
    # The kernels computing only one of the two agree with the combined one exactly.
    assert cy_logit_loss(weights, features, loss_const1, loss_const2, *penalty) == loss
    np.testing.assert_array_equal(cy_logit_gradient(weights, features, grad_const, *penalty), gradient)


@pytest.mark.parametrize('n_samples', [20, 400], ids=['c-library-exp', 'numpy-exp'])
def test_logit_kernels_stay_accurate_where_the_probability_saturates(n_samples):
    margins = np.linspace(-800, 800, n_samples)
    features = margins[:, None]
    ones = np.ones(n_samples)
    # With loss_const1 = 1 and loss_const2 = 0, the loss is the mean probability.
    loss, gradient = cy_logit_loss_gradient(np.ones(1), features, np.eye(n_samples, 1) * n_samples, ones, 0 * ones)
    assert loss == pytest.approx(np.mean(expit(margins)), rel=1e-14)
    # expit' of the first margin, -800, is about 1e-348, which is 0 in double precision.
    assert gradient[0] == pytest.approx(_exact_expit_derivatives(margins[0])[0], abs=1e-303)

    slopes = [cy_logit_gradient(np.ones(1), margins[[i]][:, None], np.ones((1, 1)))[0] for i in range(n_samples)]
    exact = np.array([_exact_expit_derivatives(margin)[0] for margin in margins])
    representable = np.abs(margins) < 700
    np.testing.assert_allclose(np.array(slopes)[representable], exact[representable], rtol=1e-14)
    assert np.all(np.array(slopes)[~representable] < 1e-303)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_boost_gradient_and_hessian_are_exact_to_double_precision(dtype):
    """``p * (1 - p)`` rounds to 0 once p rounds to 1, from a margin of about 37; this does not."""
    margins = np.concatenate([RNG.normal(size=500) * 20, [0.0, -0.0, 36.0, -36.0, 300.0, -300.0]]).astype(dtype)
    grad_const = RNG.normal(size=margins.size).astype(dtype)
    gradient, hessian = cy_boost_grad_hess(None, margins, grad_const)
    assert gradient.dtype == hessian.dtype == np.float64

    exact = np.array([_exact_expit_derivatives(margin) for margin in margins]) * np.abs(grad_const)[:, None]
    np.testing.assert_allclose(np.abs(gradient), exact[:, 0], rtol=1e-14)
    np.testing.assert_allclose(np.sign(gradient), np.sign(grad_const) * (exact[:, 0] > 0))
    # |1 - 2p| loses its relative precision near p = 1/2, as it would however it was computed.
    np.testing.assert_allclose(hessian, exact[:, 1], rtol=1e-13, atol=1e-16)


def test_logit_kernels_propagate_nan():
    features = np.ones((3, 2))
    ones = np.ones(3)
    loss, gradient = cy_logit_loss_gradient(np.array([np.nan, 1.0]), features, features, ones, ones)
    assert np.isnan(loss)
    assert np.isnan(gradient).all()
    gradient, hessian = cy_boost_grad_hess(None, np.array([np.nan, 1.0]), np.ones(2))
    assert np.isnan(gradient[0]) and np.isnan(hessian[0])
    assert np.isfinite(gradient[1]) and np.isfinite(hessian[1])


def test_logit_kernels_accept_read_only_arrays():
    features = RNG.normal(size=(200, 3))
    weights = RNG.normal(size=3)
    loss_const = RNG.normal(size=200)
    expected = cy_logit_loss(weights, features, loss_const, loss_const)
    for array in (features, weights, loss_const):
        array.flags.writeable = False
    assert cy_logit_loss(weights, features, loss_const, loss_const) == expected


def test_logit_kernels_handle_empty_inputs():
    loss, gradient = cy_logit_loss_gradient(np.zeros(0), np.zeros((5, 0)), np.zeros((5, 0)), np.ones(5), np.zeros(5))
    assert loss == 0.5
    assert gradient.shape == (0,)
    gradient, hessian = cy_boost_grad_hess(None, np.zeros(0), np.zeros(0))
    assert gradient.shape == hessian.shape == (0,)


@pytest.mark.parametrize(
    ('call', 'message'),
    [
        (lambda X: cy_logit_loss(np.ones(4), X, np.ones(50), np.ones(50)), 'weights has 4 entries'),
        (lambda X: cy_logit_loss(np.ones(3), X, np.ones(49), np.ones(50)), 'features has 50 rows'),
        (lambda X: cy_logit_gradient(np.ones(3), X, np.ones((50, 2))), r'grad_const has shape \(50, 2\)'),
        (lambda X: cy_boost_grad_hess(None, np.ones(3), np.ones(4)), 'y_score has 3 entries'),
    ],
)
def test_logit_kernels_reject_inputs_of_mismatched_shapes(call, message):
    with pytest.raises(ValueError, match=message):
        call(np.ones((50, 3)))


def test_cost_objective_accepts_features_in_fortran_order():
    fp, fn = sympy.symbols('fp fn')
    metric = Metric(CostMatrix().add_fp_cost(fp).add_fn_cost(fn).set_default(fp=1, fn=5), Cost())
    features = np.hstack((np.ones((300, 1)), RNG.normal(size=(300, 3))))
    y_true = (RNG.random(300) < 0.3).astype(int)
    weights = RNG.normal(size=4)
    arguments = {'y_true': y_true, 'C': 1.0, 'l1_ratio': 0.5, 'fit_intercept': True}
    c_order = metric._logit_objective(features=features, **arguments)
    fortran_order = metric._logit_objective(features=np.asfortranarray(features), **arguments)
    loss, gradient = fortran_order.logit_loss_gradient(np.repeat(weights, 2)[::2])
    expected_loss, expected_gradient = c_order.logit_loss_gradient(weights)
    assert loss == expected_loss
    np.testing.assert_array_equal(gradient, expected_gradient)
