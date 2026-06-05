"""Objective function base classes and benchmark functions."""

from optimlib.functions.base import MultivariateFunction, UnivariateFunction
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
    "MultivariateFunction",
    "Rosenbrock",
    "UnivariateFunction",
    "WellConditionedQuadratic",
]
