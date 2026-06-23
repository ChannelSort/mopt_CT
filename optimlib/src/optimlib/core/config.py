"""Pydantic v2 configuration models."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from optimlib.exceptions import ConfigurationError


class OptimizerConfig(BaseModel):
    """Config passed to optimizers."""

    model_config = ConfigDict(frozen=True, extra="allow")

    max_iter: int = 1000
    max_evaluations: int = 1_000_000
    max_coordinate_abs: float = 1e12
    max_objective_abs: float = 1e300
    tol: float = 1e-8
    tol_grad: float = 1e-6
    tol_step: float = 1e-12
    interval: tuple[float, float] | None = None
    alpha: float = 1e-2
    alpha_init: float = 1.0
    c1: float = 1e-4
    c2: float = 0.9
    rho: float = 0.5
    delta: float = 1e-8
    max_backtrack: int = 50
    line_search_tol: float = 1e-8
    line_search_max_iter: int = 100
    beta: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    line_search: str = "strong_wolfe"
    curvature_eps: float = 1e-12
    restart_every: int = 0
    polak_ribiere_plus: bool = True
    hessian_shift: float = 1e-8
    hessian_shift_max_attempts: int = 8
    trust_radius_initial: float = 1.0
    initial_trust_radius: float = 1.0
    trust_radius_max: float = 100.0
    max_trust_radius: float = 100.0
    trust_eta: float = 0.1
    eta: float = 0.1
    trust_shrink: float = 0.25
    trust_expand: float = 2.0
    predicted_reduction_eps: float = 1e-16
    m: int = 10
    lbfgs_m: int | None = None
    n_samples: int | None = None
    max_epochs: int = 1000
    learning_rate: float = 1e-2
    step_rule: str = "constant"
    lr_decay: float = 0.0
    seed: int = 0
    shuffle: bool = True
    batch_size: int = 16
    lm_damping_initial: float = 1e-3
    lm_damping_up: float = 10.0
    lm_damping_down: float = 0.3
    lm_damping_min: float = 1e-12
    lm_damping_max: float = 1e12

    @field_validator("max_iter", "max_evaluations", "max_backtrack", "line_search_max_iter", "hessian_shift_max_attempts", "m", "max_epochs", "batch_size")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Iteration limits must be positive.")
        return value

    @field_validator(
        "tol",
        "tol_grad",
        "tol_step",
        "max_coordinate_abs",
        "max_objective_abs",
        "alpha",
        "alpha_init",
        "delta",
        "line_search_tol",
        "epsilon",
        "curvature_eps",
        "hessian_shift",
        "trust_radius_initial",
        "initial_trust_radius",
        "trust_radius_max",
        "max_trust_radius",
        "predicted_reduction_eps",
        "learning_rate",
        "lm_damping_initial",
        "lm_damping_up",
        "lm_damping_down",
        "lm_damping_min",
        "lm_damping_max",
    )
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("Expected positive float.")
        return value

    @field_validator("c1", "c2", "rho", "beta", "beta1", "beta2", "trust_eta", "eta", "trust_shrink")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("Expected value in (0, 1).")
        return value

    @field_validator("restart_every")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Expected non-negative integer.")
        return value

    @field_validator("lr_decay")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("Expected non-negative float.")
        return value

    @field_validator("trust_expand")
    @classmethod
    def _greater_than_one(cls, value: float) -> float:
        if value <= 1.0:
            raise ValueError("Expected value greater than 1.")
        return value

    @field_validator("lbfgs_m", "n_samples")
    @classmethod
    def _optional_positive_int(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("Expected positive integer or None.")
        return value


class FunctionConfig(BaseModel):
    """Objective function declaration."""

    model_config = ConfigDict(frozen=True)

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ParamGridConfig(BaseModel):
    """Grid of optimizer parameter overrides."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, list[Any]] = Field(default_factory=dict)

    def combinations(self) -> list[dict[str, Any]]:
        """Expand parameter axes so each optimizer run receives one override set."""
        if not self.values:
            return [{}]
        keys = tuple(self.values)
        return [dict(zip(keys, values, strict=True)) for values in product(*(self.values[key] for key in keys))]


class OptimizerSpec(BaseModel):
    """Optimizer with constructor/config params and optional grid."""

    model_config = ConfigDict(frozen=True)

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    param_grid: ParamGridConfig = Field(default_factory=ParamGridConfig)


class ExperimentConfig(BaseModel):
    """Experiment YAML schema."""

    model_config = ConfigDict(frozen=True)

    functions: list[str | FunctionConfig]
    optimizers: list[str | OptimizerSpec]
    base_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    stopping_tolerances: list[float] = Field(default_factory=lambda: [1e-6])
    parallel: bool = False
    output_dir: Path = Path("outputs")
    plots: dict[str, Any] = Field(default_factory=dict)
    experiment_grid: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stopping_tolerances")
    @classmethod
    def _positive_tolerances(cls, values: list[float]) -> list[float]:
        if not values or any(value <= 0.0 for value in values):
            raise ValueError("stopping_tolerances must contain positive values.")
        return values

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load and validate experiment config from YAML."""
        if not path.exists():
            raise ConfigurationError(f"Config does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ConfigurationError(f"Invalid config {path}: {exc}") from exc

    def normalized_functions(self) -> list[FunctionConfig]:
        """Return function declarations as ``FunctionConfig`` objects."""
        return [FunctionConfig(name=item) if isinstance(item, str) else item for item in self.functions]

    def normalized_optimizers(self) -> list[OptimizerSpec]:
        """Return optimizer declarations as ``OptimizerSpec`` objects."""
        return [OptimizerSpec(name=item) if isinstance(item, str) else item for item in self.optimizers]

    def config_for(self, tolerance: float, overrides: dict[str, Any]) -> OptimizerConfig:
        """Build an optimizer config for one tolerance and parameter override."""
        data = self.base_config.model_dump()
        data.update(overrides)
        data["tol"] = tolerance
        data["tol_grad"] = tolerance
        return OptimizerConfig.model_validate(data)
