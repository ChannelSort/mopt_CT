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
from optimlib.visualization.contour import plot_contours_and_trajectories  # noqa: E402
from optimlib.visualization.lab4 import (  # noqa: E402
    plot_lab4_lbfgs_memory,
    plot_lab4_metric_dependencies,
    plot_lab4_metric_tables,
    plot_lab4_optimizer_comparison,
)


def _variant_basename(function_name: str, function_params: dict[str, Any]) -> str:
    if not function_params:
        return function_name
    compact = json.dumps(function_params, sort_keys=True, separators=(",", ":"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", compact)
    return f"{function_name}_{safe}"


def _best_runs_per_optimizer(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    """Pick one run per optimizer; for L-BFGS this also picks the best memory size."""
    converged = [run for run in runs if run.result is not None and run.result.converged]
    pool = converged if converged else [run for run in runs if run.result is not None]
    by_optimizer: dict[str, ExperimentRun] = {}
    for run in pool:
        assert run.result is not None
        previous = by_optimizer.get(run.optimizer_name)
        previous_result = None if previous is None else previous.result
        if previous is None or previous_result is None or run.result.n_iter < previous_result.n_iter:
            by_optimizer[run.optimizer_name] = run
    return list(by_optimizer.values())


def _plot_metrics(plots: dict[str, Any]) -> list[str]:
    values = plots.get("metrics", ["n_iter", "n_calls", "n_grad_calls", "n_hessian_calls"])
    if not isinstance(values, list):
        return ["n_iter", "n_calls", "n_grad_calls", "n_hessian_calls"]
    return [str(value) for value in values]


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
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    plots = experiment.config.plots
    metrics = _plot_metrics(plots)
    plot_paths: dict[str, Path] = {}

    quadratic_runs = [run for run in runs if run.function_name == "GeneratedQuadratic"]
    if bool(plots.get("tables", True)):
        plot_paths.update(plot_lab4_metric_tables(quadratic_runs, plot_dir, metrics))
    if bool(plots.get("dependencies", True)):
        plot_paths.update(plot_lab4_metric_dependencies(quadratic_runs, plot_dir, metrics, fixed_k=10.0, fixed_n=10))
    if bool(plots.get("comparison", True)):
        plot_paths.update(plot_lab4_optimizer_comparison(quadratic_runs, plot_dir, metrics))
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
            best = _best_runs_per_optimizer(family)
            if not best:
                continue
            stem = _variant_basename(function_config.name, dict(function_config.params))
            plot_paths.update(plot_contours_and_trajectories(func, best, plot_dir, f"{stem}_trajectories"))

    table_paths = experiment.save_tables()
    print(f"Lab 4 completed: {len(runs)} runs")
    for name, path in table_paths.items():
        print(f"{name}: {path}")
    if plot_paths:
        print(f"plots: {plot_dir}")


if __name__ == "__main__":
    main()
