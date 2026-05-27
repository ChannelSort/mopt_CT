"""Regression datasets and objectives for Lab 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from optimlib.core.base import FloatArray
from optimlib.functions.base import MultivariateFunction
from optimlib.utils.registry import register_function
from optimlib.utils.validation import as_float_vector, ensure_gradient, ensure_hessian


IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Normalization:
    """Affine normalization parameters for a one-dimensional feature."""

    mean: float
    scale: float

    def transform(self, x: FloatArray) -> FloatArray:
        """Normalize input values."""
        return np.asarray((x - self.mean) / self.scale, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RegressionDataset:
    """Generated one-dimensional regression dataset."""

    x: FloatArray
    y: FloatArray
    y_true: FloatArray
    noise: FloatArray
    true_dependency: str
    x_range: tuple[float, float]
    noise_variance: float
    seed: int


def near_linear_true(x: FloatArray, a: float = 1.6, b: float = -0.7, c: float = 3.0) -> FloatArray:
    """Nearly linear target relation."""
    return np.asarray(a * x + b + 0.1 * np.sin(c * x), dtype=np.float64)


def nonlinear_true(x: FloatArray) -> FloatArray:
    """Strongly nonlinear target relation with oscillation, trend, and local peak."""
    peak = 1.5 * np.exp(-0.5 * ((x - 1.0) / 0.35) ** 2)
    return np.asarray(0.35 * x**3 - 0.8 * x + np.sin(2.8 * x) + peak, dtype=np.float64)


def generate_regression_dataset(
    dataset_kind: str,
    n_points: int,
    x_range: tuple[float, float],
    noise_variance: float,
    seed: int,
) -> RegressionDataset:
    """Generate a noisy one-dimensional regression dataset."""
    if n_points < 100:
        raise ValueError("Lab 5 datasets must contain at least 100 points.")
    if x_range[0] >= x_range[1]:
        raise ValueError("x_range must have increasing bounds.")
    if noise_variance < 0.0:
        raise ValueError("noise_variance must be non-negative.")
    rng = np.random.default_rng(seed)
    x = np.linspace(x_range[0], x_range[1], n_points, dtype=np.float64)
    if dataset_kind == "near_linear":
        true_values = near_linear_true(x)
        dependency = "f_true(x) = 1.6*x - 0.7 + 0.1*sin(3*x)"
    elif dataset_kind == "nonlinear":
        true_values = nonlinear_true(x)
        dependency = "f_true(x) = 0.35*x^3 - 0.8*x + sin(2.8*x) + 1.5*exp(-0.5*((x-1)/0.35)^2)"
    else:
        raise ValueError(f"Unknown dataset_kind: {dataset_kind}")
    noise = rng.normal(loc=0.0, scale=float(np.sqrt(noise_variance)), size=n_points).astype(np.float64)
    y = true_values + noise
    return RegressionDataset(
        x=x,
        y=np.asarray(y, dtype=np.float64),
        y_true=np.asarray(true_values, dtype=np.float64),
        noise=np.asarray(noise, dtype=np.float64),
        true_dependency=dependency,
        x_range=(float(x_range[0]), float(x_range[1])),
        noise_variance=float(noise_variance),
        seed=int(seed),
    )


def fit_normalization(x: FloatArray) -> Normalization:
    """Fit feature normalization parameters."""
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if scale <= 0.0:
        scale = 1.0
    return Normalization(mean=mean, scale=scale)


def polynomial_design_matrix(x: FloatArray, degree: int, normalization: Normalization | None = None) -> FloatArray:
    """Build a polynomial design matrix with an intercept column."""
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    normalized = values if normalization is None else normalization.transform(values)
    columns = [np.ones_like(normalized)]
    columns.extend(normalized**power for power in range(1, degree + 1))
    matrix: FloatArray = np.column_stack(columns).astype(np.float64)
    return matrix


def mse(y_pred: FloatArray, y_true: FloatArray) -> float:
    """Mean squared error."""
    residual = np.asarray(y_pred, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(np.mean(residual * residual))


class PolynomialRegressionObjective(MultivariateFunction):
    """Polynomial regression empirical risk with optional smooth regularization."""

    name = "PolynomialRegressionObjective"

    def __init__(
        self,
        dataset_kind: str = "near_linear",
        degree: int = 1,
        n_points: int = 120,
        x_range: list[float] | tuple[float, float] = (-3.0, 3.0),
        noise_variance: float = 0.04,
        seed: int = 0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        l1_delta: float = 1e-6,
        regularize_intercept: bool = False,
        x0: list[float] | None = None,
    ) -> None:
        if len(x_range) != 2:
            raise ValueError("x_range must contain exactly two bounds.")
        if lambda_l1 < 0.0 or lambda_l2 < 0.0:
            raise ValueError("Regularization parameters must be non-negative.")
        if l1_delta <= 0.0:
            raise ValueError("l1_delta must be positive.")
        super().__init__(degree + 1)
        self.dataset_kind = dataset_kind
        self.degree = int(degree)
        self.n_points = int(n_points)
        self.n_samples = self.n_points
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.l1_delta = float(l1_delta)
        self.regularize_intercept = bool(regularize_intercept)
        self.dataset = generate_regression_dataset(
            dataset_kind=dataset_kind,
            n_points=n_points,
            x_range=(float(x_range[0]), float(x_range[1])),
            noise_variance=noise_variance,
            seed=seed,
        )
        self.normalization = fit_normalization(self.dataset.x)
        self.design_matrix = polynomial_design_matrix(self.dataset.x, self.degree, self.normalization)
        self.x0 = np.zeros(self.dim, dtype=np.float64) if x0 is None else as_float_vector(x0, dim=self.dim)
        self.global_minimizers = ()
        self.f_min = None
        if self.lambda_l1 == 0.0:
            solution = self._least_squares_or_ridge_solution()
            self.global_minimizers = (solution,)
            self.f_min = self._evaluate(solution)

    def _regularized_slice(self, w: FloatArray) -> FloatArray:
        return w if self.regularize_intercept else w[1:]

    def _regularization_mask(self) -> FloatArray:
        mask = np.ones(self.dim, dtype=np.float64)
        if not self.regularize_intercept:
            mask[0] = 0.0
        return mask

    def _least_squares_or_ridge_solution(self) -> FloatArray:
        matrix = self.design_matrix
        rhs = matrix.T @ self.dataset.y
        normal = matrix.T @ matrix
        if self.lambda_l2 > 0.0:
            normal = normal + self.n_samples * self.lambda_l2 * np.diag(self._regularization_mask())
        try:
            solution = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        return as_float_vector(solution, dim=self.dim)

    def _evaluate(self, x: FloatArray) -> float:
        return self.loss_components(x)["loss"]

    def predictions(self, w: FloatArray, x_values: FloatArray | None = None) -> FloatArray:
        """Predict values at training points or custom raw x values."""
        weights = as_float_vector(w, dim=self.dim)
        matrix = self.design_matrix if x_values is None else polynomial_design_matrix(x_values, self.degree, self.normalization)
        return np.asarray(matrix @ weights, dtype=np.float64)

    def residuals(self, w: FloatArray) -> FloatArray:
        """Return residual vector ``Xw - y``."""
        return np.asarray(self.predictions(w) - self.dataset.y, dtype=np.float64)

    def jacobian(self, w: FloatArray) -> FloatArray:
        """Return residual Jacobian by model parameters."""
        as_float_vector(w, dim=self.dim)
        return np.array(self.design_matrix, dtype=np.float64, copy=True)

    def regularization_terms(self, w: FloatArray) -> tuple[float, float]:
        """Return weighted L1 and L2 regularization terms."""
        weights = self._regularized_slice(as_float_vector(w, dim=self.dim))
        l1 = self.lambda_l1 * float(np.sum(np.sqrt(weights * weights + self.l1_delta * self.l1_delta)))
        l2 = self.lambda_l2 * float(np.dot(weights, weights))
        return l1, l2

    def loss_components(self, w: FloatArray) -> dict[str, float]:
        """Return full loss decomposition."""
        weights = as_float_vector(w, dim=self.dim)
        residual = self.residuals(weights)
        empirical_risk = float(np.mean(residual * residual))
        l1_term, l2_term = self.regularization_terms(weights)
        total = empirical_risk + l1_term + l2_term
        return {
            "loss": total,
            "empirical_risk": empirical_risk,
            "mse": empirical_risk,
            "l1_term": l1_term,
            "l2_term": l2_term,
        }

    def regularization_gradient(self, w: FloatArray) -> FloatArray:
        """Gradient of the weighted regularization terms."""
        weights = as_float_vector(w, dim=self.dim)
        mask = self._regularization_mask()
        l1_grad = self.lambda_l1 * weights / np.sqrt(weights * weights + self.l1_delta * self.l1_delta)
        l2_grad = 2.0 * self.lambda_l2 * weights
        gradient = mask * (l1_grad + l2_grad)
        return ensure_gradient(gradient, dim=self.dim)

    def regularization_hessian_diag(self, w: FloatArray) -> FloatArray:
        """Diagonal Hessian of the smooth regularization approximation."""
        weights = as_float_vector(w, dim=self.dim)
        mask = self._regularization_mask()
        denom = (weights * weights + self.l1_delta * self.l1_delta) ** 1.5
        l1_diag = self.lambda_l1 * self.l1_delta * self.l1_delta / denom
        l2_diag = np.full(self.dim, 2.0 * self.lambda_l2, dtype=np.float64)
        diag = mask * (l1_diag + l2_diag)
        return np.asarray(diag, dtype=np.float64)

    def _gradient_no_count(self, w: FloatArray, indices: IntArray | None = None) -> FloatArray:
        weights = as_float_vector(w, dim=self.dim)
        if indices is None:
            matrix = self.design_matrix
            target = self.dataset.y
        else:
            matrix = self.design_matrix[indices]
            target = self.dataset.y[indices]
        residual = matrix @ weights - target
        gradient = (2.0 / residual.size) * (matrix.T @ residual)
        return ensure_gradient(gradient + self.regularization_gradient(weights), dim=self.dim)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        return self._gradient_no_count(as_float_vector(x, dim=self.dim))

    def sample_gradient(self, w: FloatArray, indices: FloatArray) -> FloatArray:
        """Return mini-batch gradient and count it as one gradient evaluation."""
        self._increment_grad_count()
        int_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        return self._gradient_no_count(as_float_vector(w, dim=self.dim), int_indices)

    def hessian(self, x: object) -> FloatArray:
        self._increment_hessian_count()
        as_float_vector(x, dim=self.dim)
        base = (2.0 / self.n_samples) * (self.design_matrix.T @ self.design_matrix)
        hessian = base + np.diag(self.regularization_hessian_diag(as_float_vector(x, dim=self.dim)))
        return ensure_hessian(hessian, dim=self.dim)

    def can_analytic_solution(self) -> bool:
        """Whether the mean-estimate analytical optimizer applies."""
        return self.degree == 1 and self.lambda_l1 == 0.0 and self.lambda_l2 == 0.0

    def analytic_solution(self) -> FloatArray:
        """Closed-form one-dimensional linear regression via sample means."""
        if not self.can_analytic_solution():
            raise ValueError("Analytical mean-estimate solution is only configured for unregularized degree-1 models.")
        z = self.normalization.transform(self.dataset.x)
        y = self.dataset.y
        z_mean = float(np.mean(z))
        y_mean = float(np.mean(y))
        numerator = float(np.dot(z - z_mean, y - y_mean))
        denominator = float(np.dot(z - z_mean, z - z_mean))
        slope = 0.0 if denominator <= 0.0 else numerator / denominator
        intercept = y_mean - slope * z_mean
        return np.array([intercept, slope], dtype=np.float64)

    def dense_prediction_table(self, w: FloatArray, n_grid: int = 400) -> dict[str, FloatArray]:
        """Return dense raw x, true y, and predicted y arrays for plotting."""
        grid = np.linspace(self.dataset.x_range[0], self.dataset.x_range[1], n_grid, dtype=np.float64)
        if self.dataset_kind == "near_linear":
            true_values = near_linear_true(grid)
        else:
            true_values = nonlinear_true(grid)
        return {"x": grid, "y_true": true_values, "y_pred": self.predictions(w, grid)}

    def metadata(self) -> dict[str, Any]:
        """Return dataset and model metadata for tables and plot labels."""
        regularization = "none"
        if self.lambda_l1 > 0.0 and self.lambda_l2 > 0.0:
            regularization = "elastic_net"
        elif self.lambda_l1 > 0.0:
            regularization = "l1"
        elif self.lambda_l2 > 0.0:
            regularization = "l2"
        return {
            "dataset_kind": self.dataset_kind,
            "degree": self.degree,
            "n_points": self.n_points,
            "x_range": self.dataset.x_range,
            "noise_variance": self.dataset.noise_variance,
            "seed": self.dataset.seed,
            "true_dependency": self.dataset.true_dependency,
            "normalization_mean": self.normalization.mean,
            "normalization_scale": self.normalization.scale,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "regularization": regularization,
            "regularize_intercept": self.regularize_intercept,
        }


register_function(PolynomialRegressionObjective.__name__, PolynomialRegressionObjective)
register_function("polynomial_regression", PolynomialRegressionObjective)
register_function("regression", PolynomialRegressionObjective)
