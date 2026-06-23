"""Run Lab 4 conjugate-direction and Newton-method experiments."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIB_SRC = ROOT / "optimlib" / "src"
for path in (LIB_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import optimlib  # noqa: F401,E402
import lab4.functions  # noqa: F401,E402
from optimlib.experiment.runner import ExperimentRun, OptimizationExperiment  # noqa: E402
from optimlib.functions.base import MultivariateFunction  # noqa: E402
from optimlib.utils.registry import GLOBAL_REGISTRY  # noqa: E402
from optimlib.visualization.lab4 import (  # noqa: E402
    plot_lab4_grouped_trajectories,
    plot_lab4_lbfgs_memory,
    plot_lab4_trajectory_overview,
    save_lab4_report_tables,
)


def _variant_basename(function_name: str, function_params: dict[str, Any]) -> str:
    if not function_params:
        return function_name
    compact = json.dumps(function_params, sort_keys=True, separators=(",", ":"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", compact)
    return f"{function_name}_{safe}"


def _matching_target_runs(runs: list[ExperimentRun], target: object) -> list[ExperimentRun]:
    if not isinstance(target, dict):
        return runs
    name = target.get("name")
    params = target.get("params", {})
    if not isinstance(name, str) or not isinstance(params, dict):
        return runs
    return [run for run in runs if run.function_name == name and run.function_params == params]


def main() -> None:
    """Execute Lab 4, save tables, and build Lab 4 plots."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    if not experiment.config.output_dir.is_absolute():
        experiment.config = experiment.config.model_copy(update={"output_dir": ROOT / experiment.config.output_dir})
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    plots = experiment.config.plots
    plot_paths: dict[str, Path] = {}

    if bool(plots.get("lbfgs_memory", True)):
        lbfgs_runs = _matching_target_runs(runs, plots.get("lbfgs_memory_function"))
        plot_paths.update(plot_lab4_lbfgs_memory(lbfgs_runs, plot_dir, "n_iter"))

    if bool(plots.get("trajectories", True)):
        trajectory_names = plots.get(
            "trajectory_functions",
            ["Quadratic2DVisualization", "Lab4Rosenbrock", "Lab4Himmelblau", "Lab4Ackley"],
        )
        name_set = {str(name) for name in trajectory_names} if isinstance(trajectory_names, list) else set[str]()
        trajectory_tolerance = float(plots.get("trajectory_tolerance", 1e-8))
        tol_scale = 1e-12 * max(1.0, abs(trajectory_tolerance))
        for function_config in experiment.config.normalized_functions():
            if function_config.name not in name_set:
                continue
            func = GLOBAL_REGISTRY.get_function(function_config.name, **function_config.params)
            if not isinstance(func, MultivariateFunction) or func.dim != 2:
                continue
            family = [
                run
                for run in runs
                if run.function_name == function_config.name
                and run.function_params == dict(function_config.params)
                and abs(run.tolerance - trajectory_tolerance) <= tol_scale
            ]
            if not family:
                continue
            stem = _variant_basename(function_config.name, dict(function_config.params))
            plot_paths.update(plot_lab4_grouped_trajectories(func, family, plot_dir, stem))
            plot_paths.update(plot_lab4_trajectory_overview(func, family, plot_dir, stem))

    table_paths = experiment.save_tables()
    table_paths.update(save_lab4_report_tables(runs, experiment.config.output_dir))
    print(f"Lab 4 completed: {len(runs)} runs")
    for name, path in table_paths.items():
        print(f"{name}: {path}")
    if plot_paths:
        print(f"plots: {plot_dir}")


if __name__ == "__main__":
    main()
