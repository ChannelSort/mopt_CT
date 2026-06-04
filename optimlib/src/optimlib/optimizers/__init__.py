"""Optimization algorithms."""

from optimlib.optimizers.univariate import BrentWrapper, Dichotomy, Fibonacci, GoldenSection, Parabola, PassiveSearch
from optimlib.optimizers.gradient import ArmijoBacktracking, ConstantStepGD, SteepestDescent, StrongWolfe
from optimlib.optimizers.adaptive import AdaDelta, AdaGrad, Adam, Momentum, Nesterov, RMSProp
from optimlib.optimizers.second_order import (
    BFGS,
    ConjugateGradientQuadratic,
    DFP,
    LBFGS,
    FletcherReeves,
    FletcherReevesCG,
    NewtonCholesky,
    NewtonDirectionChoice,
    PolakRibiere,
    PolakRibiereCG,
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
    "ConjugateGradientQuadratic",
    "ConstantStepGD",
    "DFP",
    "Dichotomy",
    "FletcherReeves",
    "FletcherReevesCG",
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
    "PolakRibiereCG",
    "PowellDogLeg",
    "QuadraticConjugateGradient",
    "RMSProp",
    "ScipyNewtonCG",
    "SteepestDescent",
    "StochasticGradientDescent",
    "StrongWolfe",
]
