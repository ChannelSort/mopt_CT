"""Optimization algorithms."""

from optimlib.optimizers.univariate import BrentWrapper, Dichotomy, Fibonacci, GoldenSection, Parabola, PassiveSearch
from optimlib.optimizers.gradient import ArmijoBacktracking, ConstantStepGD, SteepestDescent, StrongWolfe
from optimlib.optimizers.adaptive import AdaDelta, AdaGrad, Adam, Momentum, Nesterov, RMSProp
from optimlib.optimizers.second_order import (
    BFGS,
    DFP,
    LBFGS,
    FletcherReeves,
    NewtonCholesky,
    NewtonDirectionChoice,
    PolakRibiere,
    PowellDogLeg,
    QuadraticConjugateGradient,
    ScipyNewtonCG,
)
from optimlib.optimizers.regression import (
    AnalyticalLinearRegression1D,
    GaussNewton,
    LevenbergMarquardt,
    MiniBatchGradientDescent,
    StochasticGradientDescent,
)

__all__ = [
    "AdaDelta",
    "AdaGrad",
    "Adam",
    "ArmijoBacktracking",
    "AnalyticalLinearRegression1D",
    "BFGS",
    "BrentWrapper",
    "ConstantStepGD",
    "DFP",
    "Dichotomy",
    "FletcherReeves",
    "Fibonacci",
    "GoldenSection",
    "LBFGS",
    "GaussNewton",
    "LevenbergMarquardt",
    "MiniBatchGradientDescent",
    "Momentum",
    "NewtonCholesky",
    "NewtonDirectionChoice",
    "Nesterov",
    "Parabola",
    "PassiveSearch",
    "PolakRibiere",
    "PowellDogLeg",
    "QuadraticConjugateGradient",
    "RMSProp",
    "ScipyNewtonCG",
    "SteepestDescent",
    "StochasticGradientDescent",
    "StrongWolfe",
]
