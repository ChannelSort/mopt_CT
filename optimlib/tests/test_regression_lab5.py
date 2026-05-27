from __future__ import annotations

import numpy as np

import optimlib  # noqa: F401
import lab5.functions  # noqa: F401
from lab5.functions import Normalization, PolynomialRegressionObjective, mse, polynomial_design_matrix
from optimlib.core.config import OptimizerConfig
from optimlib.optimizers.regression import AnalyticalLinearRegression1D, LevenbergMarquardt, MiniBatchGradientDescent


def test_polynomial_design_matrix_uses_normalized_feature() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    matrix = polynomial_design_matrix(x, degree=3, normalization=Normalization(mean=2.0, scale=1.0))

    expected = np.array(
        [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(matrix, expected)


def test_mse() -> None:
    assert mse(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == 2.5


def test_analytical_linear_solution_has_small_gradient() -> None:
    func = PolynomialRegressionObjective(dataset_kind="near_linear", degree=1, n_points=120, noise_variance=0.0, seed=1)
    result = AnalyticalLinearRegression1D().minimize(func, OptimizerConfig(tol_grad=1e-8))

    assert result.converged
    assert result.n_iter == 1
    assert result.history
    assert result.history[-1].grad is not None
    assert float(np.linalg.norm(result.history[-1].grad)) <= 1e-8


def test_regression_gradient_matches_finite_difference() -> None:
    func = PolynomialRegressionObjective(
        dataset_kind="nonlinear",
        degree=3,
        n_points=120,
        noise_variance=0.01,
        seed=3,
        lambda_l1=0.01,
        lambda_l2=0.02,
    )
    w = np.array([0.2, -0.1, 0.3, -0.4], dtype=np.float64)
    analytic = func.gradient(w)
    h = 1e-6
    numeric = np.empty_like(w)
    for idx in range(w.size):
        step = np.zeros_like(w)
        step[idx] = h
        numeric[idx] = (func(w + step) - func(w - step)) / (2.0 * h)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_mini_batch_records_batch_size() -> None:
    func = PolynomialRegressionObjective(dataset_kind="near_linear", degree=2, n_points=120, noise_variance=0.01, seed=5)
    config = OptimizerConfig(max_iter=5, max_epochs=5, learning_rate=0.01, batch_size=8, step_rule="constant", tol_grad=1e-14)
    result = MiniBatchGradientDescent().minimize(func, config)

    assert result.n_iter == 5
    assert result.n_grad_calls > result.n_iter
    assert result.history[-1].extra_metrics["batch_size"] == 8


def test_levenberg_marquardt_converges_on_simple_regression() -> None:
    func = PolynomialRegressionObjective(dataset_kind="near_linear", degree=1, n_points=120, noise_variance=0.0, seed=7)
    config = OptimizerConfig(max_iter=20, tol_grad=1e-8, lm_damping_initial=1e-3)
    result = LevenbergMarquardt().minimize(func, config)

    assert result.converged
    assert result.history
    assert "damping" in result.history[-1].extra_metrics
