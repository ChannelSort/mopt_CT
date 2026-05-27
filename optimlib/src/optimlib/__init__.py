"""optimlib core package for the optimization-course monorepo."""

from __future__ import annotations

from optimlib.core.base import OptimizationResult, StepState

# Import built-in optimizers so they register themselves.
from optimlib.optimizers import univariate as _univariate
from optimlib.optimizers import gradient as _gradient
from optimlib.optimizers import adaptive as _adaptive
from optimlib.optimizers import second_order as _second_order
from optimlib.optimizers import regression as _regression

__all__ = ["OptimizationResult", "StepState"]
