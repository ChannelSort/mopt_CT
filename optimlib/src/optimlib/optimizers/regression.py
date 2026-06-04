"""Regression-oriented optimizers used by Lab 5."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from optimlib.core.base import FloatArray, ObjectiveFunction, OptimizationResult, StepState
from optimlib.core.callbacks import Callback, HistoryCallback
from optimlib.core.config import OptimizerConfig
from optimlib.exceptions import StopOptimization
from optimlib.functions.base import MultivariateFunction
from optimlib.utils.registry import register_optimizer
from optimlib.utils.validation import as_float_vector, ensure_finite, ensure_gradient


LossComponents = Mapping[str, float]


@runtime_checkable
class LossComponentsFunction(Protocol):
    """Objective with an explicit loss decomposition."""

    def loss_components(self, x: FloatArray) -> LossComponents:
        """Return full loss decomposition."""


@runtime_checkable
class SampleGradientFunction(Protocol):
    """Objective that can compute mini-batch gradients."""

    def sample_gradient(self, w: FloatArray, indices: FloatArray) -> FloatArray:
        """Return one mini-batch gradient estimate."""


@runtime_checkable
class LeastSquaresFunction(Protocol):
    """Objective exposing residuals and residual Jacobian."""

    def residuals(self, w: FloatArray) -> FloatArray:
        """Return residual vector."""

    def jacobian(self, w: FloatArray) -> FloatArray:
        """Return residual Jacobian."""


@runtime_checkable
class RegularizedLeastSquaresFunction(Protocol):
    """Objective exposing smooth regularization derivatives."""

    def regularization_gradient(self, w: FloatArray) -> FloatArray:
        """Return regularization gradient."""

    def regularization_hessian_diag(self, w: FloatArray) -> FloatArray:
        """Return diagonal of the regularization Hessian."""


@runtime_checkable
class AnalyticalRegressionFunction(Protocol):
    """Objective that supports a closed-form linear-regression solution."""

    def can_analytic_solution(self) -> bool:
        """Whether the analytical solution is applicable."""

    def analytic_solution(self) -> FloatArray:
        """Return analytical model parameters."""


@runtime_checkable
class RegressionDatasetFunction(Protocol):
    """Regression objective metadata used by stochastic optimizers."""

    n_samples: int
    dataset_kind: str
    degree: int
    lambda_l1: float
    lambda_l2: float


def _hessian_count(func: MultivariateFunction) -> int:
    return func.hessian_count


def _loss_components(func: MultivariateFunction, x: FloatArray) -> dict[str, float]:
    if isinstance(func, LossComponentsFunction):
        components = func.loss_components(x)
        return {str(key): float(value) for key, value in components.items()}
    value = func(x)
    return {"loss": value, "empirical_risk": value, "l1_term": 0.0, "l2_term": 0.0}


def _sample_gradient(func: MultivariateFunction, x: FloatArray, indices: FloatArray) -> FloatArray:
    if isinstance(func, SampleGradientFunction):
        gradient = func.sample_gradient(x, indices)
        return ensure_gradient(gradient, dim=func.dim)
    return ensure_gradient(func.gradient(x), dim=func.dim)


def _residuals(func: MultivariateFunction, x: FloatArray) -> FloatArray:
    if not isinstance(func, LeastSquaresFunction):
        raise TypeError("Gauss-Newton optimizers require residuals(x).")
    residuals = func.residuals(x)
    return np.asarray(residuals, dtype=np.float64).reshape(-1)


def _jacobian(func: MultivariateFunction, x: FloatArray) -> FloatArray:
    if not isinstance(func, LeastSquaresFunction):
        raise TypeError("Gauss-Newton optimizers require jacobian(x).")
    jacobian = np.asarray(func.jacobian(x), dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[1] != func.dim:
        raise ValueError(f"Expected Jacobian with {func.dim} columns, got {jacobian.shape}.")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian contains NaN or Inf.")
    return jacobian


def _regularization_gradient(func: MultivariateFunction, x: FloatArray) -> FloatArray:
    if isinstance(func, RegularizedLeastSquaresFunction):
        return ensure_gradient(func.regularization_gradient(x), dim=func.dim)
    return np.zeros(func.dim, dtype=np.float64)


def _regularization_hessian_diag(func: MultivariateFunction, x: FloatArray) -> FloatArray:
    if isinstance(func, RegularizedLeastSquaresFunction):
        diag = np.asarray(func.regularization_hessian_diag(x), dtype=np.float64).reshape(-1)
        if diag.shape != (func.dim,):
            raise ValueError(f"Expected regularization Hessian diagonal shape ({func.dim},), got {diag.shape}.")
        return diag
    return np.zeros(func.dim, dtype=np.float64)


class _RegressionOptimizer:
    """Shared result and callback handling for regression optimizers."""

    name = "RegressionOptimizer"

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def _as_multivariate(self, func: ObjectiveFunction) -> MultivariateFunction:
        if not isinstance(func, MultivariateFunction):
            raise TypeError(f"{self.name} requires a MultivariateFunction.")
        return func

    def _start(self) -> tuple[HistoryCallback, list[Callback], float]:
        history = HistoryCallback()
        callbacks: list[Callback] = [history, *self.callbacks]
        for callback in callbacks:
            callback.on_start()
        return history, callbacks, time.perf_counter()

    def _emit(self, callbacks: Sequence[Callback], state: StepState) -> None:
        for callback in callbacks:
            callback.on_step(state)

    def _result(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        n_iter: int,
        converged: bool,
        message: str,
        history: HistoryCallback,
        started_at: float,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        result_metadata = {"optimizer": self.name, "elapsed_seconds": time.perf_counter() - started_at}
        result_metadata.update(_loss_components(func, x))
        if metadata:
            result_metadata.update(metadata)
        return OptimizationResult(
            x=x,
            f=f_value,
            n_iter=n_iter,
            n_calls=func.call_count,
            n_grad_calls=func.grad_count,
            n_hessian_calls=_hessian_count(func),
            converged=converged,
            message=message,
            history=history.history,
            metadata=result_metadata,
        )

    def _finish(self, callbacks: Sequence[Callback], result: OptimizationResult) -> OptimizationResult:
        for callback in callbacks:
            callback.on_end(result)
        return result


class AnalyticalLinearRegression1D(_RegressionOptimizer):
    """Closed-form one-dimensional linear regression via sample means."""

    name = "AnalyticalLinearRegression1D"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks, started_at = self._start()
        if not isinstance(objective, AnalyticalRegressionFunction):
            raise TypeError("AnalyticalLinearRegression1D requires analytic_solution().")
        if not objective.can_analytic_solution():
            x0 = objective.initial_point()
            f0 = ensure_finite(objective(x0), "objective")
            grad0 = ensure_gradient(objective.gradient(x0), dim=objective.dim)
            state = StepState(
                iteration=0,
                x=x0,
                f=f0,
                grad=grad0,
                step_size=0.0,
                extra_metrics={
                    **_loss_components(objective, x0),
                    "grad_norm": float(np.linalg.norm(grad0)),
                    "epoch": 0,
                    "function_calls": objective.call_count,
                    "gradient_calls": objective.grad_count,
                },
            )
            self._emit(callbacks, state)
            result = self._result(
                objective,
                x0,
                f0,
                0,
                False,
                "analytical_solution_not_applicable",
                history,
                started_at,
                {"epochs": 0, "step_rule": "closed_form"},
            )
            return self._finish(callbacks, result)
        x = ensure_gradient(objective.analytic_solution(), dim=objective.dim)
        f_value = ensure_finite(objective(x), "objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        components = _loss_components(objective, x)
        state = StepState(
            iteration=0,
            x=x,
            f=f_value,
            grad=grad,
            step_size=0.0,
            extra_metrics={
                **components,
                "grad_norm": float(np.linalg.norm(grad)),
                "epoch": 0,
                "function_calls": objective.call_count,
                "gradient_calls": objective.grad_count,
            },
        )
        try:
            self._emit(callbacks, state)
        except StopOptimization:
            pass
        result = self._result(
            objective,
            x,
            f_value,
            1,
            float(np.linalg.norm(grad)) <= config.tol_grad,
            "Analytical one-dimensional linear regression solution.",
            history,
            started_at,
            {"epochs": 0, "step_rule": "closed_form"},
        )
        return self._finish(callbacks, result)


class StochasticGradientDescent(_RegressionOptimizer):
    """Stochastic gradient descent for regression objectives."""

    name = "StochasticGradientDescent"

    def _batch_size(self, func: MultivariateFunction, config: OptimizerConfig) -> int:
        return 1

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks, started_at = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history, started_at)
            return self._finish(callbacks, result)

        n_samples = config.n_samples or (objective.n_samples if isinstance(objective, RegressionDatasetFunction) else 1)
        max_epochs = config.max_epochs
        learning_rate = config.learning_rate
        step_rule = config.step_rule
        lr_decay = config.lr_decay
        seed = config.seed
        shuffle = config.shuffle
        rng = np.random.default_rng(seed)
        batch_size = max(1, min(n_samples, self._batch_size(objective, config)))
        converged = False
        message = "Maximum epochs reached."
        n_iter = 0
        update_count = 0

        try:
            for epoch in range(max_epochs):
                indices = np.arange(n_samples, dtype=np.float64)
                if shuffle:
                    rng.shuffle(indices)
                for start in range(0, n_samples, batch_size):
                    batch_indices = indices[start : start + batch_size]
                    if step_rule == "sqrt_decay":
                        alpha = learning_rate / np.sqrt(update_count + 1.0)
                    elif step_rule == "linear_decay":
                        alpha = learning_rate / (1.0 + lr_decay * update_count)
                    else:
                        alpha = learning_rate
                    batch_grad = _sample_gradient(objective, x, batch_indices)
                    x = as_float_vector(x - alpha * batch_grad, dim=objective.dim)
                    update_count += 1

                f_value = ensure_finite(objective(x), "objective")
                grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
                grad_norm = float(np.linalg.norm(grad))
                components = _loss_components(objective, x)
                state = StepState(
                    iteration=epoch,
                    x=x,
                    f=f_value,
                    grad=grad,
                    step_size=alpha,
                    extra_metrics={
                        **components,
                        "grad_norm": grad_norm,
                        "epoch": epoch + 1,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "effective_learning_rate": alpha,
                        "step_rule": step_rule,
                        "gradient_evaluations": objective.grad_count,
                        "function_calls": objective.call_count,
                    },
                )
                n_iter = epoch + 1
                self._emit(callbacks, state)
                if grad_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_norm:.3e}."
                    break
        except StopOptimization as exc:
            converged = True
            message = exc.message

        result = self._result(
            objective,
            x,
            f_value,
            n_iter,
            converged,
            message,
            history,
            started_at,
            {
                "epochs": n_iter,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "step_rule": step_rule,
                "updates": update_count,
            },
        )
        return self._finish(callbacks, result)


class MiniBatchGradientDescent(StochasticGradientDescent):
    """Mini-batch gradient descent for regression objectives."""

    name = "MiniBatchGradientDescent"

    def __init__(
        self,
        study_dataset_kind: str | None = None,
        study_degree: int | None = None,
        study_unregularized: bool = False,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        super().__init__(callbacks=callbacks)
        self.study_dataset_kind = study_dataset_kind
        self.study_degree = study_degree
        self.study_unregularized = study_unregularized

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        study_requires_metadata = self.study_dataset_kind is not None or self.study_degree is not None or self.study_unregularized
        if study_requires_metadata:
            if not isinstance(objective, RegressionDatasetFunction):
                return self._not_applicable(objective, config)
            regression_objective: RegressionDatasetFunction = objective
            if self.study_dataset_kind is not None and regression_objective.dataset_kind != self.study_dataset_kind:
                return self._not_applicable(objective, config)
            if self.study_degree is not None and regression_objective.degree != self.study_degree:
                return self._not_applicable(objective, config)
            if self.study_unregularized and (regression_objective.lambda_l1 != 0.0 or regression_objective.lambda_l2 != 0.0):
                return self._not_applicable(objective, config)
        return super().minimize(objective, config)

    def _not_applicable(self, objective: MultivariateFunction, config: OptimizerConfig) -> OptimizationResult:
        history, callbacks, started_at = self._start()
        x0 = objective.initial_point()
        f0 = ensure_finite(objective(x0), "objective")
        grad0 = ensure_gradient(objective.gradient(x0), dim=objective.dim)
        state = StepState(
            iteration=0,
            x=x0,
            f=f0,
            grad=grad0,
            step_size=0.0,
            extra_metrics={
                **_loss_components(objective, x0),
                "grad_norm": float(np.linalg.norm(grad0)),
                "epoch": 0,
                "batch_size": self._batch_size(objective, config),
                "function_calls": objective.call_count,
                "gradient_evaluations": objective.grad_count,
            },
        )
        self._emit(callbacks, state)
        result = self._result(
            objective,
            x0,
            f0,
            0,
            False,
            "mini_batch_study_not_applicable",
            history,
            started_at,
            {"epochs": 0, "batch_size": self._batch_size(objective, config), "step_rule": config.step_rule},
        )
        return self._finish(callbacks, result)

    def _batch_size(self, func: MultivariateFunction, config: OptimizerConfig) -> int:
        requested = config.batch_size
        n_samples = config.n_samples or (func.n_samples if isinstance(func, RegressionDatasetFunction) else requested)
        return max(1, min(n_samples, requested))


class GaussNewton(_RegressionOptimizer):
    """Gauss-Newton optimizer for least-squares regression objectives."""

    name = "GaussNewton"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        return self._minimize(func, config, use_lm=False)

    def _minimize(self, func: ObjectiveFunction, config: OptimizerConfig, *, use_lm: bool) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks, started_at = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        damping = config.lm_damping_initial
        damping_up = config.lm_damping_up
        damping_down = config.lm_damping_down
        min_damping = config.lm_damping_min
        max_damping = config.lm_damping_max
        converged = grad_norm <= config.tol_grad
        message = "Initial point satisfies gradient tolerance." if converged else "Maximum iterations reached."
        n_iter = 0
        accepted_steps = 0

        try:
            for iteration in range(config.max_iter):
                if converged:
                    break
                residuals = _residuals(objective, x)
                jacobian = _jacobian(objective, x)
                hessian_approx = (2.0 / residuals.size) * (jacobian.T @ jacobian)
                hessian_approx = hessian_approx + np.diag(_regularization_hessian_diag(objective, x))
                gradient = (2.0 / residuals.size) * (jacobian.T @ residuals) + _regularization_gradient(objective, x)
                system = hessian_approx + (damping if use_lm else 0.0) * np.eye(objective.dim, dtype=np.float64)
                try:
                    step = np.linalg.solve(system, -gradient)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(system, -gradient, rcond=None)[0]
                step = as_float_vector(step, dim=objective.dim)
                x_trial = as_float_vector(x + step, dim=objective.dim)
                f_trial = ensure_finite(objective(x_trial), "objective")
                accepted = (not use_lm) or f_trial <= f_value
                if accepted:
                    x = x_trial
                    f_value = f_trial
                    grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
                    grad_norm = float(np.linalg.norm(grad))
                    accepted_steps += 1
                    if use_lm:
                        damping = max(min_damping, damping * damping_down)
                elif use_lm:
                    damping = min(max_damping, damping * damping_up)

                components = _loss_components(objective, x)
                state = StepState(
                    iteration=iteration,
                    x=x,
                    f=f_value,
                    grad=grad,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={
                        **components,
                        "grad_norm": grad_norm,
                        "accepted": accepted,
                        "damping": damping if use_lm else 0.0,
                        "epoch": iteration + 1,
                        "gradient_evaluations": objective.grad_count,
                        "function_calls": objective.call_count,
                    },
                )
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
                    break
                if use_lm and damping >= max_damping:
                    message = "lm_damping_max_reached"
                    break
        except StopOptimization as exc:
            converged = True
            message = exc.message

        result = self._result(
            objective,
            x,
            f_value,
            n_iter,
            converged,
            message,
            history,
            started_at,
            {
                "epochs": n_iter,
                "accepted_steps": accepted_steps,
                "lm_damping": damping if use_lm else 0.0,
            },
        )
        return self._finish(callbacks, result)


class LevenbergMarquardt(GaussNewton):
    """Levenberg-Marquardt damped Gauss-Newton method."""

    name = "LevenbergMarquardt"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        return self._minimize(func, config, use_lm=True)


for _cls in (AnalyticalLinearRegression1D, StochasticGradientDescent, MiniBatchGradientDescent, GaussNewton, LevenbergMarquardt):
    register_optimizer(_cls.__name__, _cls)

register_optimizer("analytical_linear_regression_1d", AnalyticalLinearRegression1D)
register_optimizer("analytic_linear_regression", AnalyticalLinearRegression1D)
register_optimizer("sgd", StochasticGradientDescent)
register_optimizer("stochastic_gradient_descent", StochasticGradientDescent)
register_optimizer("mini_batch_gradient_descent", MiniBatchGradientDescent)
register_optimizer("mini_batch_gd", MiniBatchGradientDescent)
register_optimizer("gauss_newton", GaussNewton)
register_optimizer("levenberg_marquardt", LevenbergMarquardt)
register_optimizer("lm", LevenbergMarquardt)
