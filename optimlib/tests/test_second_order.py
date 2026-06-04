from __future__ import annotations

import numpy as np
import pytest

import optimlib  # noqa: F401
import lab4.functions  # noqa: F401
from lab4.functions import GeneratedQuadratic
from optimlib.core.base import OptimizationResult
from optimlib.core.config import OptimizerConfig
from optimlib.optimizers.second_order import (
    BFGS,
    DFP,
    LBFGS,
    ConjugateGradientQuadratic,
    FletcherReevesCG,
    NewtonCholesky,
    NewtonDirectionChoice,
    PolakRibiereCG,
    PowellDogLeg,
)


def _config(max_iter: int = 220) -> OptimizerConfig:
    return OptimizerConfig(
        max_iter=max_iter,
        tol_grad=1e-6,
        tol_step=1e-14,
        line_search="armijo",
        alpha_init=1.0,
        trust_radius_initial=1.0,
        m=5,
    )


def _quadratic(condition_number: float = 10.0) -> GeneratedQuadratic:
    return GeneratedQuadratic(n=5, k=condition_number, seed=2, random_orthogonal=True, start_scale=2.0)


def _assert_valid_history(result: OptimizationResult) -> None:
    assert len(result.history) == result.n_iter
    for state in result.history:
        assert np.all(np.isfinite(state.x))
        assert np.isfinite(state.f)
        assert state.grad is not None
        assert np.all(np.isfinite(state.grad))
        assert "grad_norm" in state.extra_metrics


@pytest.mark.parametrize(
    "optimizer",
    [
        ConjugateGradientQuadratic(),
        FletcherReevesCG(),
        PolakRibiereCG(),
        NewtonCholesky(),
        NewtonDirectionChoice(),
        PowellDogLeg(),
        DFP(),
        BFGS(),
        LBFGS(m=5),
    ],
)
def test_second_order_methods_converge_on_convex_quadratic(optimizer: object) -> None:
    func = _quadratic(condition_number=10.0)
    result = optimizer.minimize(func, _config())  # type: ignore[attr-defined]

    assert result.converged
    np.testing.assert_allclose(result.x, func.x_star, atol=2e-6)
    assert result.n_calls > 0
    assert result.n_grad_calls > 0
    _assert_valid_history(result)


@pytest.mark.parametrize("optimizer", [NewtonCholesky(), NewtonDirectionChoice(), PowellDogLeg()])
def test_hessian_based_methods_record_hessian_calls(optimizer: object) -> None:
    func = _quadratic(condition_number=100.0)
    result = optimizer.minimize(func, _config(max_iter=40))  # type: ignore[attr-defined]

    assert result.converged
    assert result.n_hessian_calls > 0
    assert result.n_hessian_calls == func.hessian_count
    _assert_valid_history(result)


@pytest.mark.parametrize(
    "optimizer",
    [
        ConjugateGradientQuadratic(),
        FletcherReevesCG(),
        PolakRibiereCG(),
        NewtonCholesky(),
        NewtonDirectionChoice(),
        PowellDogLeg(),
        DFP(),
        BFGS(),
        LBFGS(m=5),
    ],
)
def test_second_order_methods_do_not_degrade_on_ill_conditioned_quadratic(optimizer: object) -> None:
    func = _quadratic(condition_number=1000.0)
    initial_value = func(func.initial_point())
    func.reset_count()

    result = optimizer.minimize(func, _config(max_iter=120))  # type: ignore[attr-defined]

    assert np.isfinite(result.f)
    assert result.f <= initial_value
    assert result.n_calls > 0
    assert result.n_grad_calls > 0
    _assert_valid_history(result)
    if optimizer.name not in {"FletcherReeves", "PolakRibiere"}:  # type: ignore[attr-defined]
        assert np.linalg.norm(result.x - func.x_star) <= 1e-4


def test_powell_dog_leg_records_trust_region_diagnostics() -> None:
    func = _quadratic(condition_number=100.0)
    result = PowellDogLeg().minimize(func, _config(max_iter=40))

    assert result.converged
    assert result.history
    last_metrics = result.history[-1].extra_metrics
    assert "trust_radius" in last_metrics
    assert "rho" in last_metrics
    assert "accepted" in last_metrics


def test_newton_direction_choice_handles_indefinite_hessian() -> None:
    class IndefiniteQuadratic(GeneratedQuadratic):
        def __init__(self) -> None:
            super().__init__(n=2, k=1.0, seed=0, random_orthogonal=False, x_star=[0.0, 0.0], x0=[2.0, -1.0])
            self.a_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
            self.b_vector = np.zeros(2, dtype=np.float64)
            self.c = 0.0

    func = IndefiniteQuadratic()
    result = NewtonDirectionChoice().minimize(func, _config(max_iter=5))

    assert result.history
    assert result.n_hessian_calls > 0
    assert result.message != "non_positive_definite_hessian"
    assert all(np.all(np.isfinite(state.x)) for state in result.history)
