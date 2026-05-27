"""Objective functions for Lab 4."""

from __future__ import annotations

import math

import numpy as np

from optimlib.core.base import FloatArray
from optimlib.functions.base import MultivariateFunction
from optimlib.utils.registry import register_function
from optimlib.utils.validation import as_float_vector, ensure_gradient, ensure_hessian


class GeneratedQuadratic(MultivariateFunction):
    """Strictly convex quadratic ``0.5 * x.T A x - b.T x + c``."""

    name = "GeneratedQuadratic"

    def __init__(
        self,
        n: int = 10,
        k: float = 10.0,
        seed: int = 0,
        random_orthogonal: bool = True,
        x_star: list[float] | None = None,
        x0: list[float] | None = None,
        c: float | None = None,
        start_scale: float = 5.0,
    ) -> None:
        if k < 1.0:
            raise ValueError("Condition number k must be at least 1.")
        super().__init__(n)
        self.k = float(k)
        self.seed = int(seed)
        self.random_orthogonal = bool(random_orthogonal)
        rng = np.random.default_rng(self.seed)

        if self.k == 1.0:
            eigenvalues = np.ones(n, dtype=np.float64)
        else:
            eigenvalues = np.geomspace(1.0, self.k, n, dtype=np.float64)
        self.eigenvalues = eigenvalues

        if self.random_orthogonal:
            raw = rng.normal(size=(n, n))
            q_matrix, r_matrix = np.linalg.qr(raw)
            signs = np.sign(np.diag(r_matrix))
            signs[signs == 0.0] = 1.0
            q_matrix = q_matrix * signs
        else:
            q_matrix = np.eye(n, dtype=np.float64)
        self.q_matrix = q_matrix
        self.a_matrix = q_matrix.T @ np.diag(eigenvalues) @ q_matrix

        if x_star is None:
            self.x_star = rng.normal(size=n).astype(np.float64)
        else:
            self.x_star = as_float_vector(x_star, dim=n)
        self.b_vector = self.a_matrix @ self.x_star

        if c is None:
            self.c = 0.5 * float(np.dot(self.x_star, self.a_matrix @ self.x_star))
        else:
            self.c = float(c)
        self.f_min = float(self._evaluate(self.x_star))
        self.global_minimizers = (np.array(self.x_star, dtype=np.float64, copy=True),)

        if x0 is None:
            start = self.x_star + start_scale * rng.normal(size=n)
            if float(np.linalg.norm(start - self.x_star)) <= 1e-12:
                start = start + np.ones(n, dtype=np.float64)
            self.x0 = np.asarray(start, dtype=np.float64)
        else:
            self.x0 = as_float_vector(x0, dim=n)

    def _evaluate(self, x: FloatArray) -> float:
        return float(0.5 * np.dot(x, self.a_matrix @ x) - np.dot(self.b_vector, x) + self.c)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        vector = as_float_vector(x, dim=self.dim)
        return ensure_gradient(self.a_matrix @ vector - self.b_vector, dim=self.dim)

    def hessian(self, x: object) -> FloatArray:
        self._increment_hessian_count()
        as_float_vector(x, dim=self.dim)
        return ensure_hessian(np.array(self.a_matrix, dtype=np.float64, copy=True), dim=self.dim)


class Quadratic2DVisualization(GeneratedQuadratic):
    """Fixed two-dimensional quadratic used for trajectory plots."""

    name = "Quadratic2DVisualization"

    def __init__(
        self,
        seed: int = 42,
        random_orthogonal: bool = True,
        x0: list[float] | None = None,
        x_star: list[float] | None = None,
        start_scale: float = 5.0,
    ) -> None:
        super().__init__(
            n=2,
            k=10.0,
            seed=seed,
            random_orthogonal=random_orthogonal,
            x_star=x_star,
            x0=x0,
            start_scale=start_scale,
        )


class Lab4Rosenbrock(MultivariateFunction):
    """Two-dimensional Rosenbrock function."""

    name = "Rosenbrock"
    global_minimizers = (np.array([1.0, 1.0], dtype=np.float64),)
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        self.x0 = np.array([-1.2, 1.0] if x0 is None else x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            value = (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
        return float(value)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        vector = as_float_vector(x, dim=2)
        return np.array(
            [
                2.0 * (vector[0] - 1.0) - 400.0 * vector[0] * (vector[1] - vector[0] ** 2),
                200.0 * (vector[1] - vector[0] ** 2),
            ],
            dtype=np.float64,
        )

    def hessian(self, x: object) -> FloatArray:
        self._increment_hessian_count()
        vector = as_float_vector(x, dim=2)
        return np.array(
            [
                [2.0 - 400.0 * vector[1] + 1200.0 * vector[0] ** 2, -400.0 * vector[0]],
                [-400.0 * vector[0], 200.0],
            ],
            dtype=np.float64,
        )


class Lab4Himmelblau(MultivariateFunction):
    """Himmelblau function with four global minimizers."""

    name = "Himmelblau"
    global_minimizers = (
        np.array([3.0, 2.0], dtype=np.float64),
        np.array([-2.805118, 3.131312], dtype=np.float64),
        np.array([-3.779310, -3.283186], dtype=np.float64),
        np.array([3.584428, -1.848126], dtype=np.float64),
    )
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        self.x0 = np.array([2.0, 2.0] if x0 is None else x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        first = x[0] ** 2 + x[1] - 11.0
        second = x[0] + x[1] ** 2 - 7.0
        return float(first * first + second * second)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        vector = as_float_vector(x, dim=2)
        first = vector[0] ** 2 + vector[1] - 11.0
        second = vector[0] + vector[1] ** 2 - 7.0
        return np.array(
            [4.0 * vector[0] * first + 2.0 * second, 2.0 * first + 4.0 * vector[1] * second],
            dtype=np.float64,
        )

    def hessian(self, x: object) -> FloatArray:
        self._increment_hessian_count()
        vector = as_float_vector(x, dim=2)
        first = vector[0] ** 2 + vector[1] - 11.0
        second = vector[0] + vector[1] ** 2 - 7.0
        return np.array(
            [
                [4.0 * first + 8.0 * vector[0] ** 2 + 2.0, 4.0 * (vector[0] + vector[1])],
                [4.0 * (vector[0] + vector[1]), 4.0 * second + 8.0 * vector[1] ** 2 + 2.0],
            ],
            dtype=np.float64,
        )


class Lab4Ackley(MultivariateFunction):
    """Two-dimensional Ackley objective with finite-difference Hessian fallback."""

    name = "Ackley"
    global_minimizers = (np.array([0.0, 0.0], dtype=np.float64),)
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        self.x0 = np.array([1.0, 1.0] if x0 is None else x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        radius = math.sqrt(0.5 * float(np.dot(x, x)))
        cos_avg = 0.5 * float(np.cos(2.0 * math.pi * x[0]) + np.cos(2.0 * math.pi * x[1]))
        return float(-20.0 * math.exp(-0.2 * radius) - math.exp(cos_avg) + math.e + 20.0)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        vector = as_float_vector(x, dim=2)
        radius = math.sqrt(0.5 * float(np.dot(vector, vector)))
        if radius <= 1e-15:
            first = np.zeros(2, dtype=np.float64)
        else:
            first = 2.0 * math.exp(-0.2 * radius) * vector / radius
        cos_avg = 0.5 * float(np.cos(2.0 * math.pi * vector[0]) + np.cos(2.0 * math.pi * vector[1]))
        return first + math.pi * math.exp(cos_avg) * np.sin(2.0 * math.pi * vector)


for _cls in (GeneratedQuadratic, Quadratic2DVisualization, Lab4Rosenbrock, Lab4Himmelblau, Lab4Ackley):
    register_function(_cls.__name__, _cls)

register_function("generated_quadratic", GeneratedQuadratic)
register_function("quadratic_2d_visualization", Quadratic2DVisualization)
register_function("lab4_rosenbrock", Lab4Rosenbrock)
register_function("lab4_himmelblau", Lab4Himmelblau)
register_function("lab4_ackley", Lab4Ackley)
