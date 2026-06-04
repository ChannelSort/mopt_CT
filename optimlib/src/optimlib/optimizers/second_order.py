"""Conjugate-direction, Newton, trust-region, and quasi-Newton methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from optimlib.core.base import FloatArray, ObjectiveFunction, OptimizationResult, StepState
from optimlib.core.callbacks import Callback, HistoryCallback
from optimlib.core.config import OptimizerConfig
from optimlib.exceptions import LineSearchError, StopOptimization
from optimlib.functions.base import MultivariateFunction
from optimlib.optimizers.gradient import StrongWolfe
from optimlib.utils.registry import register_optimizer
from optimlib.utils.validation import as_float_vector, ensure_finite, ensure_gradient, ensure_hessian


def _hessian_count(func: MultivariateFunction) -> int:
    return func.hessian_count


def _armijo_step(
    func: MultivariateFunction,
    x: FloatArray,
    f_value: float,
    grad: FloatArray,
    direction: FloatArray,
    config: OptimizerConfig,
    alpha_init: float | None = None,
) -> float:
    derphi0 = float(np.dot(grad, direction))
    if derphi0 >= 0.0:
        raise LineSearchError("Direction is not a descent direction.")
    alpha = config.alpha_init if alpha_init is None else alpha_init
    for _ in range(config.max_backtrack):
        trial = x + alpha * direction
        if func(trial) <= f_value + config.c1 * alpha * derphi0:
            return alpha
        alpha *= config.rho
        if alpha <= config.tol_step:
            break
    return max(alpha, config.tol_step)


def _line_search_step(
    func: MultivariateFunction,
    x: FloatArray,
    f_value: float,
    grad: FloatArray,
    direction: FloatArray,
    config: OptimizerConfig,
) -> float:
    mode = config.line_search.strip().lower()
    if mode in {"none", "full", "unit"}:
        return 1.0
    if mode == "armijo":
        return _armijo_step(func, x, f_value, grad, direction, config)
    try:
        alpha = StrongWolfe()._compute_step(func, x, f_value, grad, direction, config)
        if alpha > config.tol_step:
            return alpha
        return _armijo_step(func, x, f_value, grad, direction, config)
    except LineSearchError:
        return _armijo_step(func, x, f_value, grad, direction, config)


def _fallback_to_steepest_if_tiny(
    func: MultivariateFunction,
    x: FloatArray,
    f_value: float,
    grad: FloatArray,
    direction: FloatArray,
    alpha: float,
    config: OptimizerConfig,
) -> tuple[FloatArray, float, bool]:
    step_norm = float(np.linalg.norm(alpha * direction))
    if step_norm > config.tol_step or float(np.linalg.norm(grad)) <= config.tol_grad:
        return direction, alpha, False
    fallback = -grad
    if float(np.linalg.norm(fallback)) <= config.tol_step:
        return direction, alpha, False
    try:
        fallback_alpha = _armijo_step(func, x, f_value, grad, fallback, config)
    except LineSearchError:
        return direction, alpha, False
    fallback_step_norm = float(np.linalg.norm(fallback_alpha * fallback))
    if fallback_step_norm > step_norm:
        return fallback, fallback_alpha, True
    return direction, alpha, False


def _solve_cholesky(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    factor = np.linalg.cholesky(matrix)
    y = np.linalg.solve(factor, rhs)
    solution = np.linalg.solve(factor.T, y)
    return as_float_vector(solution, dim=rhs.size)


def _symmetrized(matrix: FloatArray) -> FloatArray:
    result: FloatArray = np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)
    return result


class _Lab4Optimizer:
    """Small shared shell for Lab 4 optimizers."""

    name = "Lab4Optimizer"

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def _as_multivariate(self, func: ObjectiveFunction) -> MultivariateFunction:
        if not isinstance(func, MultivariateFunction):
            raise TypeError(f"{self.name} requires a MultivariateFunction.")
        return func

    def _start(self) -> tuple[HistoryCallback, list[Callback]]:
        history = HistoryCallback()
        callbacks: list[Callback] = [history, *self.callbacks]
        for callback in callbacks:
            callback.on_start()
        return history, callbacks

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
        metadata: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        result_metadata = {"optimizer": self.name}
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


class QuadraticConjugateGradient(_Lab4Optimizer):
    """Linear conjugate-gradient method for strictly convex quadratics."""

    name = "QuadraticConjugateGradient"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_sq = float(np.dot(grad, grad))
        grad_norm = float(np.sqrt(grad_sq))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        hessian = ensure_hessian(objective.hessian(x), dim=objective.dim)
        direction = -grad
        converged = False
        message = "Maximum iterations reached."
        n_iter = 0
        curvature_eps = config.curvature_eps

        try:
            for iteration in range(config.max_iter):
                h_direction = hessian @ direction
                denom = float(np.dot(direction, h_direction))
                if denom <= curvature_eps:
                    message = "non_positive_curvature"
                    break
                alpha = grad_sq / denom
                step = alpha * direction
                x_next = as_float_vector(x + step, dim=objective.dim)
                f_next = ensure_finite(objective(x_next), "objective")
                grad_next = ensure_gradient(objective.gradient(x_next), dim=objective.dim)
                grad_next_sq = float(np.dot(grad_next, grad_next))
                grad_next_norm = float(np.sqrt(grad_next_sq))
                beta = 0.0 if grad_sq <= curvature_eps else grad_next_sq / grad_sq
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={"alpha": alpha, "beta": beta, "grad_norm": grad_next_norm},
                )
                x, f_value, grad, grad_sq = x_next, f_next, grad_next, grad_next_sq
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_next_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_next_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
                    break
                direction = -grad + beta * direction
        except StopOptimization as exc:
            converged = True
            message = exc.message

        result = self._result(objective, x, f_value, n_iter, converged, message, history)
        return self._finish(callbacks, result)


class _NonlinearConjugateGradient(_Lab4Optimizer, ABC):
    """Base loop for nonlinear conjugate-gradient methods."""

    name = "NonlinearConjugateGradient"

    @abstractmethod
    def _beta(self, grad: FloatArray, grad_next: FloatArray, config: OptimizerConfig) -> float:
        """Compute method-specific beta coefficient."""

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        direction = -grad
        converged = False
        message = "Maximum iterations reached."
        n_iter = 0
        restart_every = config.restart_every

        try:
            for iteration in range(config.max_iter):
                restarted = False
                if float(np.dot(grad, direction)) >= 0.0:
                    direction = -grad
                    restarted = True
                if float(np.linalg.norm(direction)) <= config.tol_step:
                    direction = -grad
                    restarted = True
                alpha = _line_search_step(objective, x, f_value, grad, direction, config)
                direction, alpha, fallback = _fallback_to_steepest_if_tiny(objective, x, f_value, grad, direction, alpha, config)
                restarted = restarted or fallback
                step = alpha * direction
                x_next = as_float_vector(x + step, dim=objective.dim)
                f_next = ensure_finite(objective(x_next), "objective")
                grad_next = ensure_gradient(objective.gradient(x_next), dim=objective.dim)
                grad_next_norm = float(np.linalg.norm(grad_next))
                beta = self._beta(grad, grad_next, config)
                if restart_every > 0 and (iteration + 1) % restart_every == 0:
                    beta = 0.0
                    restarted = True
                direction_next = -grad_next + beta * direction
                if float(np.dot(grad_next, direction_next)) >= 0.0:
                    direction_next = -grad_next
                    beta = 0.0
                    restarted = True
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={"alpha": alpha, "beta": beta, "grad_norm": grad_next_norm, "restart": restarted},
                )
                x, f_value, grad, direction = x_next, f_next, grad_next, direction_next
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_next_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_next_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
                    break
        except StopOptimization as exc:
            converged = True
            message = exc.message

        result = self._result(objective, x, f_value, n_iter, converged, message, history)
        return self._finish(callbacks, result)


class FletcherReeves(_NonlinearConjugateGradient):
    """Fletcher-Reeves nonlinear conjugate-gradient method."""

    name = "FletcherReeves"

    def _beta(self, grad: FloatArray, grad_next: FloatArray, config: OptimizerConfig) -> float:
        denom = float(np.dot(grad, grad))
        if denom <= config.curvature_eps:
            return 0.0
        return float(np.dot(grad_next, grad_next)) / denom


class PolakRibiere(_NonlinearConjugateGradient):
    """Polak-Ribiere nonlinear conjugate-gradient method."""

    name = "PolakRibiere"

    def _beta(self, grad: FloatArray, grad_next: FloatArray, config: OptimizerConfig) -> float:
        denom = float(np.dot(grad, grad))
        if denom <= config.curvature_eps:
            return 0.0
        beta = float(np.dot(grad_next, grad_next - grad)) / denom
        if config.polak_ribiere_plus:
            return max(0.0, beta)
        return beta


ConjugateGradientQuadratic = QuadraticConjugateGradient
FletcherReevesCG = FletcherReeves
PolakRibiereCG = PolakRibiere


class NewtonCholesky(_Lab4Optimizer):
    """Newton method solving the Newton system with Cholesky factorization."""

    name = "NewtonCholesky"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        converged = False
        message = "Maximum iterations reached."
        n_iter = 0

        try:
            for iteration in range(config.max_iter):
                hessian = ensure_hessian(objective.hessian(x), dim=objective.dim)
                try:
                    direction = _solve_cholesky(_symmetrized(hessian), -grad)
                except np.linalg.LinAlgError:
                    message = "non_positive_definite_hessian"
                    break
                if float(np.dot(grad, direction)) >= 0.0:
                    message = "non_descent_newton_direction"
                    break
                alpha = _line_search_step(objective, x, f_value, grad, direction, config)
                step = alpha * direction
                x_next = as_float_vector(x + step, dim=objective.dim)
                f_next = ensure_finite(objective(x_next), "objective")
                grad_next = ensure_gradient(objective.gradient(x_next), dim=objective.dim)
                grad_next_norm = float(np.linalg.norm(grad_next))
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={"alpha": alpha, "grad_norm": grad_next_norm},
                )
                x, f_value, grad = x_next, f_next, grad_next
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_next_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_next_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
                    break
        except StopOptimization as exc:
            converged = True
            message = exc.message

        result = self._result(objective, x, f_value, n_iter, converged, message, history)
        return self._finish(callbacks, result)


class NewtonDirectionChoice(_Lab4Optimizer):
    """Newton method with modified-Hessian and steepest-descent fallbacks."""

    name = "NewtonDirectionChoice"

    def _direction(self, hessian: FloatArray, grad: FloatArray, config: OptimizerConfig) -> tuple[FloatArray, str, float]:
        matrix = _symmetrized(hessian)
        try:
            direction = _solve_cholesky(matrix, -grad)
            if float(np.dot(grad, direction)) < 0.0:
                return direction, "newton", 0.0
        except np.linalg.LinAlgError:
            pass

        delta = config.hessian_shift
        max_attempts = config.hessian_shift_max_attempts
        min_eig = float(np.min(np.linalg.eigvalsh(matrix)))
        shift = max(delta, -min_eig + delta)
        identity = np.eye(grad.size, dtype=np.float64)
        for _ in range(max_attempts):
            try:
                direction = _solve_cholesky(matrix + shift * identity, -grad)
                if float(np.dot(grad, direction)) < 0.0:
                    return direction, "modified_newton", shift
            except np.linalg.LinAlgError:
                pass
            shift *= 10.0
        return -grad, "steepest_descent", shift

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        converged = False
        message = "Maximum iterations reached."
        n_iter = 0

        try:
            for iteration in range(config.max_iter):
                hessian = ensure_hessian(objective.hessian(x), dim=objective.dim)
                direction, direction_kind, shift = self._direction(hessian, grad, config)
                alpha = _line_search_step(objective, x, f_value, grad, direction, config)
                step = alpha * direction
                x_next = as_float_vector(x + step, dim=objective.dim)
                f_next = ensure_finite(objective(x_next), "objective")
                grad_next = ensure_gradient(objective.gradient(x_next), dim=objective.dim)
                grad_next_norm = float(np.linalg.norm(grad_next))
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={
                        "alpha": alpha,
                        "grad_norm": grad_next_norm,
                        "direction": direction_kind,
                        "hessian_shift": shift,
                    },
                )
                x, f_value, grad = x_next, f_next, grad_next
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_next_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_next_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
                    break
        except StopOptimization as exc:
            converged = True
            message = exc.message

        result = self._result(objective, x, f_value, n_iter, converged, message, history)
        return self._finish(callbacks, result)


class PowellDogLeg(_Lab4Optimizer):
    """Powell dog-leg trust-region method."""

    name = "PowellDogLeg"

    def _dog_leg_step(self, grad: FloatArray, hessian: FloatArray, radius: float) -> tuple[FloatArray, str]:
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= 0.0:
            return np.zeros_like(grad), "zero"
        hessian = _symmetrized(hessian)
        grad_h_grad = float(np.dot(grad, hessian @ grad))
        if grad_h_grad <= 0.0:
            return -radius * grad / grad_norm, "cauchy_boundary"

        alpha_cauchy = float(np.dot(grad, grad)) / grad_h_grad
        cauchy = -alpha_cauchy * grad
        cauchy_norm = float(np.linalg.norm(cauchy))
        if cauchy_norm >= radius:
            return radius * cauchy / cauchy_norm, "cauchy_boundary"

        try:
            newton = _solve_cholesky(hessian, -grad)
        except np.linalg.LinAlgError:
            return cauchy, "cauchy"
        if float(np.linalg.norm(newton)) <= radius:
            return newton, "newton"

        diff = newton - cauchy
        a = float(np.dot(diff, diff))
        b = 2.0 * float(np.dot(cauchy, diff))
        c = float(np.dot(cauchy, cauchy)) - radius * radius
        discriminant = max(0.0, b * b - 4.0 * a * c)
        tau = (-b + float(np.sqrt(discriminant))) / (2.0 * a)
        tau = float(np.clip(tau, 0.0, 1.0))
        return cauchy + tau * diff, "dog_leg"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        radius = config.trust_radius_initial
        max_radius = config.trust_radius_max
        eta = config.trust_eta
        shrink = config.trust_shrink
        expand = config.trust_expand

        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        converged = False
        message = "Maximum iterations reached."
        n_iter = 0
        accepted_steps = 0

        try:
            for iteration in range(config.max_iter):
                hessian = ensure_hessian(objective.hessian(x), dim=objective.dim)
                step, step_kind = self._dog_leg_step(grad, hessian, radius)
                step_norm = float(np.linalg.norm(step))
                predicted = -float(np.dot(grad, step) + 0.5 * np.dot(step, hessian @ step))
                if predicted <= config.predicted_reduction_eps:
                    message = "non_positive_predicted_reduction"
                    break
                f_trial = ensure_finite(objective(x + step), "objective")
                actual = f_value - f_trial
                rho = actual / predicted

                accepted = rho > eta
                if rho < 0.25:
                    radius *= shrink
                elif rho > 0.75 and step_norm >= 0.8 * radius:
                    radius = min(max_radius, expand * radius)

                if accepted:
                    x = as_float_vector(x + step, dim=objective.dim)
                    f_value = f_trial
                    grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
                    grad_norm = float(np.linalg.norm(grad))
                    accepted_steps += 1

                state = StepState(
                    iteration=iteration,
                    x=x,
                    f=f_value,
                    grad=grad,
                    step_size=step_norm,
                    extra_metrics={
                        "trust_radius": radius,
                        "rho": rho,
                        "accepted": accepted,
                        "actual_reduction": actual,
                        "predicted_reduction": predicted,
                        "step_kind": step_kind,
                        "grad_norm": grad_norm,
                    },
                )
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_norm:.3e}."
                    break
                if radius <= config.tol_step:
                    message = "trust_region_radius_too_small"
                    break
                if accepted and step_norm <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {step_norm:.3e}."
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
            {"accepted_steps": accepted_steps, "final_trust_radius": radius},
        )
        return self._finish(callbacks, result)


class _InverseQuasiNewton(_Lab4Optimizer, ABC):
    """Base class for inverse-Hessian quasi-Newton updates."""

    name = "InverseQuasiNewton"

    @abstractmethod
    def _update_inverse(
        self,
        inverse_hessian: FloatArray,
        s: FloatArray,
        y: FloatArray,
        ys: float,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, bool]:
        """Return updated inverse Hessian and whether the update was applied."""

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        inverse_hessian = np.eye(objective.dim, dtype=np.float64)
        converged = False
        message = "Maximum iterations reached."
        n_iter = 0
        skipped_updates = 0
        resets = 0

        try:
            for iteration in range(config.max_iter):
                direction = -inverse_hessian @ grad
                if float(np.dot(grad, direction)) >= 0.0 or float(np.linalg.norm(direction)) <= config.tol_step:
                    inverse_hessian = np.eye(objective.dim, dtype=np.float64)
                    direction = -grad
                    resets += 1
                alpha = _line_search_step(objective, x, f_value, grad, direction, config)
                direction, alpha, fallback = _fallback_to_steepest_if_tiny(objective, x, f_value, grad, direction, alpha, config)
                if fallback:
                    inverse_hessian = np.eye(objective.dim, dtype=np.float64)
                    resets += 1
                step = alpha * direction
                x_next = as_float_vector(x + step, dim=objective.dim)
                f_next = ensure_finite(objective(x_next), "objective")
                grad_next = ensure_gradient(objective.gradient(x_next), dim=objective.dim)
                y = grad_next - grad
                ys = float(np.dot(y, step))
                inverse_hessian, updated = self._update_inverse(inverse_hessian, step, y, ys, config)
                if not updated:
                    skipped_updates += 1
                grad_next_norm = float(np.linalg.norm(grad_next))
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={
                        "alpha": alpha,
                        "grad_norm": grad_next_norm,
                        "curvature": ys,
                        "update_applied": updated,
                    },
                )
                x, f_value, grad = x_next, f_next, grad_next
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_next_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_next_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
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
            {"skipped_updates": skipped_updates, "direction_resets": resets},
        )
        return self._finish(callbacks, result)


class DFP(_InverseQuasiNewton):
    """Davidon-Fletcher-Powell inverse-Hessian quasi-Newton method."""

    name = "DFP"

    def _update_inverse(
        self,
        inverse_hessian: FloatArray,
        s: FloatArray,
        y: FloatArray,
        ys: float,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, bool]:
        eps = config.curvature_eps
        if ys <= eps * max(1.0, float(np.linalg.norm(s)) * float(np.linalg.norm(y))):
            return inverse_hessian, False
        hy = inverse_hessian @ y
        yhy = float(np.dot(y, hy))
        if yhy <= eps:
            return inverse_hessian, False
        updated = inverse_hessian + np.outer(s, s) / ys - np.outer(hy, hy) / yhy
        return _symmetrized(updated), True


class BFGS(_InverseQuasiNewton):
    """BFGS inverse-Hessian quasi-Newton method."""

    name = "BFGS"

    def _update_inverse(
        self,
        inverse_hessian: FloatArray,
        s: FloatArray,
        y: FloatArray,
        ys: float,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, bool]:
        eps = config.curvature_eps
        if ys <= eps * max(1.0, float(np.linalg.norm(s)) * float(np.linalg.norm(y))):
            return inverse_hessian, False
        rho = 1.0 / ys
        identity = np.eye(s.size, dtype=np.float64)
        left = identity - rho * np.outer(s, y)
        right = identity - rho * np.outer(y, s)
        updated = left @ inverse_hessian @ right + rho * np.outer(s, s)
        return _symmetrized(updated), True


class LBFGS(_Lab4Optimizer):
    """Limited-memory BFGS with two-loop recursion."""

    name = "LBFGS"

    def __init__(self, m: int = 10, callbacks: Sequence[Callback] | None = None) -> None:
        super().__init__(callbacks=callbacks)
        if m <= 0:
            raise ValueError("m must be positive.")
        self.m = m

    def _memory_size(self, config: OptimizerConfig) -> int:
        memory = config.lbfgs_m if config.lbfgs_m is not None else config.m
        return max(1, memory)

    def _direction(
        self,
        grad: FloatArray,
        s_history: Sequence[FloatArray],
        y_history: Sequence[FloatArray],
        rho_history: Sequence[float],
    ) -> FloatArray:
        if not s_history:
            return -grad
        q = np.array(grad, dtype=np.float64, copy=True)
        alphas: list[float] = []
        for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history), strict=True):
            alpha = rho * float(np.dot(s, q))
            alphas.append(alpha)
            q = q - alpha * y
        s_last = s_history[-1]
        y_last = y_history[-1]
        yy = float(np.dot(y_last, y_last))
        gamma = 1.0 if yy <= 0.0 else float(np.dot(s_last, y_last)) / yy
        r = gamma * q
        for s, y, rho, alpha in zip(s_history, y_history, rho_history, reversed(alphas), strict=True):
            beta = rho * float(np.dot(y, r))
            r = r + s * (alpha - beta)
        return -r

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x = objective.initial_point()
        f_value = ensure_finite(objective(x), "initial objective")
        grad = ensure_gradient(objective.gradient(x), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= config.tol_grad:
            result = self._result(objective, x, f_value, 0, True, "Initial point satisfies gradient tolerance.", history)
            return self._finish(callbacks, result)

        memory = self._memory_size(config)
        s_history: list[FloatArray] = []
        y_history: list[FloatArray] = []
        rho_history: list[float] = []
        converged = False
        message = "Maximum iterations reached."
        n_iter = 0
        skipped_updates = 0
        resets = 0

        try:
            for iteration in range(config.max_iter):
                direction = self._direction(grad, s_history, y_history, rho_history)
                if float(np.dot(grad, direction)) >= 0.0 or float(np.linalg.norm(direction)) <= config.tol_step:
                    s_history.clear()
                    y_history.clear()
                    rho_history.clear()
                    direction = -grad
                    resets += 1
                alpha = _line_search_step(objective, x, f_value, grad, direction, config)
                direction, alpha, fallback = _fallback_to_steepest_if_tiny(objective, x, f_value, grad, direction, alpha, config)
                if fallback:
                    s_history.clear()
                    y_history.clear()
                    rho_history.clear()
                    resets += 1
                step = alpha * direction
                x_next = as_float_vector(x + step, dim=objective.dim)
                f_next = ensure_finite(objective(x_next), "objective")
                grad_next = ensure_gradient(objective.gradient(x_next), dim=objective.dim)
                y = grad_next - grad
                ys = float(np.dot(y, step))
                eps = config.curvature_eps
                updated = ys > eps * max(1.0, float(np.linalg.norm(step)) * float(np.linalg.norm(y)))
                if updated:
                    if len(s_history) == memory:
                        s_history.pop(0)
                        y_history.pop(0)
                        rho_history.pop(0)
                    s_history.append(step)
                    y_history.append(y)
                    rho_history.append(1.0 / ys)
                else:
                    skipped_updates += 1
                grad_next_norm = float(np.linalg.norm(grad_next))
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=float(np.linalg.norm(step)),
                    extra_metrics={
                        "alpha": alpha,
                        "grad_norm": grad_next_norm,
                        "curvature": ys,
                        "update_applied": updated,
                        "memory_size": len(s_history),
                        "m": memory,
                    },
                )
                x, f_value, grad = x_next, f_next, grad_next
                n_iter = iteration + 1
                self._emit(callbacks, state)
                if grad_next_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_next_norm:.3e}."
                    break
                if float(np.linalg.norm(step)) <= config.tol_step:
                    message = f"Step tolerance reached before gradient tolerance: {float(np.linalg.norm(step)):.3e}."
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
            {"m": memory, "skipped_updates": skipped_updates, "direction_resets": resets},
        )
        return self._finish(callbacks, result)


class ScipyNewtonCG(_Lab4Optimizer):
    """Wrapper around ``scipy.optimize.minimize(..., method="Newton-CG")``."""

    name = "ScipyNewtonCG"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        objective = self._as_multivariate(func)
        history, callbacks = self._start()
        x0 = objective.initial_point()

        def objective_wrapper(x: FloatArray) -> float:
            return objective(x)

        def gradient_wrapper(x: FloatArray) -> FloatArray:
            return ensure_gradient(objective.gradient(x), dim=objective.dim)

        def hessian_wrapper(x: FloatArray) -> FloatArray:
            return ensure_hessian(objective.hessian(x), dim=objective.dim)

        def callback(xk: FloatArray) -> None:
            x_vec = as_float_vector(xk, dim=objective.dim)
            grad = ensure_gradient(objective.gradient(x_vec), dim=objective.dim)
            state = StepState(
                iteration=len(history.history),
                x=x_vec,
                f=ensure_finite(objective(x_vec), "objective"),
                grad=grad,
                step_size=None,
                extra_metrics={"grad_norm": float(np.linalg.norm(grad))},
            )
            self._emit(callbacks, state)

        scipy_result = minimize(
            objective_wrapper,
            x0,
            method="Newton-CG",
            jac=gradient_wrapper,
            hess=hessian_wrapper,
            callback=callback,
            options={"xtol": config.tol_step, "maxiter": config.max_iter, "disp": False},
        )
        x_final = as_float_vector(scipy_result.x, dim=objective.dim)
        f_final = ensure_finite(float(scipy_result.fun), "objective")
        grad_final = ensure_gradient(objective.gradient(x_final), dim=objective.dim)
        grad_norm = float(np.linalg.norm(grad_final))
        converged = grad_norm <= config.tol_grad
        message = (
            f"Gradient norm tolerance reached: {grad_norm:.3e}."
            if converged
            else str(scipy_result.message)
        )
        result = self._result(
            objective,
            x_final,
            f_final,
            int(scipy_result.nit),
            converged,
            message,
            history,
            {"scipy_success": bool(scipy_result.success), "scipy_status": int(scipy_result.status)},
        )
        return self._finish(callbacks, result)


for _cls in (
    QuadraticConjugateGradient,
    FletcherReeves,
    PolakRibiere,
    NewtonCholesky,
    NewtonDirectionChoice,
    PowellDogLeg,
    DFP,
    BFGS,
    LBFGS,
    ScipyNewtonCG,
):
    register_optimizer(_cls.__name__, _cls)

register_optimizer("quadratic_conjugate_gradient", QuadraticConjugateGradient)
register_optimizer("conjugate_gradient_quadratic", QuadraticConjugateGradient)
register_optimizer("ConjugateGradientQuadratic", QuadraticConjugateGradient)
register_optimizer("fletcher_reeves", FletcherReeves)
register_optimizer("FletcherReevesCG", FletcherReeves)
register_optimizer("polak_ribiere", PolakRibiere)
register_optimizer("PolakRibiereCG", PolakRibiere)
register_optimizer("newton_cholesky", NewtonCholesky)
register_optimizer("newton_direction_choice", NewtonDirectionChoice)
register_optimizer("powell_dog_leg", PowellDogLeg)
register_optimizer("dog_leg", PowellDogLeg)
register_optimizer("dfp", DFP)
register_optimizer("bfgs", BFGS)
register_optimizer("lbfgs", LBFGS)
register_optimizer("l_bfgs", LBFGS)
register_optimizer("scipy_newton_cg", ScipyNewtonCG)
register_optimizer("newton_cg", ScipyNewtonCG)
