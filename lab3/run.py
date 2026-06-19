"""Run Lab 3 adaptive-method experiments."""

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
import lab3.functions  # noqa: F401,E402
from optimlib.experiment.runner import ExperimentRun, OptimizationExperiment
from optimlib.utils.registry import GLOBAL_REGISTRY
from optimlib.visualization.convergence import plot_param_sensitivity
from optimlib.visualization.lab3 import (
    save_lab3_complex_tables,
    save_lab3_quadratic_best_table,
    save_lab3_quadratic_sensitivity_tables,
    plot_lab3_quadratic_convergence,
    plot_lab3_grouped_trajectories,
)


def _variant_basename(function_name: str, function_params: dict[str, Any]) -> str:
    if not function_params:
        return function_name
    compact = json.dumps(function_params, sort_keys=True, separators=(",", ":"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", compact)
    return f"{function_name}_{safe}"


def _best_runs_per_optimizer(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    """Pick one run per optimizer (fewest iterations among converged, else best available)."""
    converged = [r for r in runs if r.result is not None and r.result.converged]
    pool = converged if converged else [r for r in runs if r.result is not None]
    by_opt: dict[str, ExperimentRun] = {}
    for r in pool:
        assert r.result is not None
        prev = by_opt.get(r.optimizer_name)
        prev_result = None if prev is None else prev.result
        if prev is None or prev_result is None or r.result.n_iter < prev_result.n_iter:
            by_opt[r.optimizer_name] = r
    return list(by_opt.values())


def main() -> None:
    """Execute Lab 3, save sensitivity plots and best-run trajectories per objective."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    if not experiment.config.output_dir.is_absolute():
        experiment.config = experiment.config.model_copy(update={"output_dir": ROOT / experiment.config.output_dir})
    traj_tol = float(experiment.config.plots.get("trajectory_tolerance", 1e-8))
    max_iter = int(experiment.config.base_config.max_iter)
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    tol_scale = 1e-12 * max(1.0, abs(traj_tol))

    for function_config in experiment.config.normalized_functions():
        func = GLOBAL_REGISTRY.get_function(function_config.name, **function_config.params)
        fparams = dict(function_config.params)
        stem = _variant_basename(function_config.name, fparams)
        family = [run for run in runs if run.function_name == function_config.name and run.function_params == fparams]

        for optimizer in ("Momentum", "Nesterov", "RMSProp"):
            subset = [run for run in family if run.optimizer_name == optimizer]
            if subset:
                y_param = "rho" if optimizer == "RMSProp" else "beta"
                plot_param_sensitivity(subset, plot_dir, "alpha", y_param, basename=f"{stem}_{optimizer}_heatmap")

        ada_grad = [run for run in family if run.optimizer_name == "AdaGrad"]
        if ada_grad:
            plot_param_sensitivity(
                ada_grad,
                plot_dir,
                "alpha",
                None,
                basename=f"{stem}_AdaGrad_sensitivity",
                title=f"{function_config.name}: AdaGrad n_iter ({max_iter} = max_iter reached)",
                max_iter=max_iter,
                y_label="iterations",
            )
            plot_param_sensitivity(
                ada_grad,
                plot_dir,
                "alpha",
                None,
                metric="f",
                basename=f"{stem}_AdaGrad_final_f",
                title=f"{function_config.name}: AdaGrad final f(x_k)",
                y_label=r"$f(x_k)$",
                y_log=True,
            )

        ada_delta = [run for run in family if run.optimizer_name == "AdaDelta"]
        if ada_delta:
            plot_param_sensitivity(
                ada_delta,
                plot_dir,
                "rho",
                None,
                basename=f"{stem}_AdaDelta_sensitivity",
                title=f"{function_config.name}: AdaDelta n_iter ({max_iter} = max_iter reached)",
                max_iter=max_iter,
                y_label="iterations",
            )
            plot_param_sensitivity(
                ada_delta,
                plot_dir,
                "rho",
                None,
                metric="f",
                basename=f"{stem}_AdaDelta_final_f",
                title=f"{function_config.name}: AdaDelta final f(x_k)",
                y_label=r"$f(x_k)$",
                y_log=True,
            )

        adam = [run for run in family if run.optimizer_name == "Adam"]
        if adam:
            adam_alpha = [run for run in adam if abs(float(run.params.get("alpha", 0.0)) - 1.0e-2) <= 1e-12]
            plot_param_sensitivity(adam_alpha or adam, plot_dir, "beta1", "beta2", basename=f"{stem}_Adam_heatmap")

        traj_runs = [
            run
            for run in family
            if run.result is not None
            and abs(run.tolerance - traj_tol) <= tol_scale
        ]
        plot_lab3_grouped_trajectories(func, traj_runs, plot_dir, stem)
        if function_config.name in {"WellConditionedQuadratic", "IllConditionedQuadratic"}:
            plot_lab3_quadratic_convergence(traj_runs, plot_dir, stem)

    print(f"Lab 3 completed: {len(runs)} runs")
    table_dir = experiment.config.output_dir / "tables"
    table_paths: dict[str, Path] = {}
    table_paths["quadratic_best"] = save_lab3_quadratic_best_table(runs, table_dir)
    table_paths["quadratic_sensitivity"] = save_lab3_quadratic_sensitivity_tables(runs, table_dir)
    table_paths.update({f"complex_{name}": path for name, path in save_lab3_complex_tables(runs, table_dir).items()})
    for name, path in {**experiment.save_tables(), **table_paths}.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
