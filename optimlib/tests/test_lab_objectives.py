from __future__ import annotations

import numpy as np

import lab2.functions as lab2_functions
import lab3.functions as lab3_functions


FUNCTION_NAMES = (
    "WellConditionedQuadratic",
    "IllConditionedQuadratic",
    "Rosenbrock",
    "Ackley",
    "Himmelblau",
)

POINTS = (
    np.array([-2.0, -2.0], dtype=np.float64),
    np.array([1.0, 1.0], dtype=np.float64),
    np.array([2.0, -2.0], dtype=np.float64),
    np.array([2.0, 2.0], dtype=np.float64),
    np.array([4.0, 4.0], dtype=np.float64),
    np.array([0.0, 0.0], dtype=np.float64),
)


def test_lab2_and_lab3_use_same_objective_classes() -> None:
    for name in FUNCTION_NAMES:
        assert getattr(lab2_functions, name) is getattr(lab3_functions, name)


def test_lab2_and_lab3_objectives_match_on_reference_points() -> None:
    for name in FUNCTION_NAMES:
        lab2_func = getattr(lab2_functions, name)()
        lab3_func = getattr(lab3_functions, name)()
        for point in POINTS:
            assert lab2_func(point) == lab3_func(point)
            np.testing.assert_allclose(lab2_func.gradient(point), lab3_func.gradient(point), rtol=0.0, atol=0.0)
        assert len(lab2_func.global_minimizers) == len(lab3_func.global_minimizers)
        for lab2_min, lab3_min in zip(lab2_func.global_minimizers, lab3_func.global_minimizers, strict=True):
            np.testing.assert_allclose(lab2_min, lab3_min, rtol=0.0, atol=0.0)


def test_ackley_origin_gradient_is_handled_numerically() -> None:
    func = lab2_functions.Ackley()
    np.testing.assert_allclose(func.gradient(np.array([0.0, 0.0])), np.zeros(2), atol=0.0)
    np.testing.assert_allclose(func.gradient(np.array([1e-13, -1e-13])), np.zeros(2), atol=0.0)
    assert np.isfinite(func(np.array([0.0, 0.0])))
    assert np.isfinite(func(np.array([1e-13, -1e-13])))
