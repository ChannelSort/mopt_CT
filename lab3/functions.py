"""Lab 3 objective functions re-exported from the common benchmark module."""

from __future__ import annotations

from optimlib.functions.benchmarks import (
    ACKLEY_ORIGIN_TOL,
    Ackley,
    Himmelblau,
    IllConditionedQuadratic,
    Rosenbrock,
    WellConditionedQuadratic,
)

__all__ = [
    "ACKLEY_ORIGIN_TOL",
    "Ackley",
    "Himmelblau",
    "IllConditionedQuadratic",
    "Rosenbrock",
    "WellConditionedQuadratic",
]
