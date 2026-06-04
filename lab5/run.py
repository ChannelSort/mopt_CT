"""Run Lab 5 regression experiments."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIB_SRC = ROOT / "optimlib" / "src"
for path in (LIB_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import optimlib  # noqa: F401,E402
import lab5.functions  # noqa: F401,E402
from optimlib.experiment.runner import ExperimentRun, OptimizationExperiment  # noqa: E402
from optimlib.functions.base import MultivariateFunction  # noqa: E402
from optimlib.utils.registry import GLOBAL_REGISTRY  # noqa: E402
from optimlib.visualization.regression import (  # noqa: E402
    plot_lab5_batch_size_comparison,
    plot_lab5_coefficients,
    plot_lab5_loss_by_gradient_calls,
    plot_lab5_loss_history,
    plot_lab5_method_comparison,
    plot_lab5_regression_fit,
    plot_lab5_regularization_comparison,
)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def _variant_basename(function_name: str, function_params: dict[str, Any]) -> str:
    compact = json.dumps(function_params, sort_keys=True, separators=(",", ":"))
    return _safe_stem(f"{function_name}_{compact}")


def _run_id(run: ExperimentRun, index: int) -> str:
    compact = json.dumps({"f": run.function_params, "o": run.optimizer_name, "p": run.params}, sort_keys=True, default=str)
    return f"run_{index:04d}_{_safe_stem(compact)[:90]}"


def _match_params(run: ExperimentRun, **expected: Any) -> bool:
    return all(run.function_params.get(key) == value for key, value in expected.items())


def _is_unregularized(run: ExperimentRun) -> bool:
    return float(run.function_params.get("lambda_l1", 0.0)) == 0.0 and float(run.function_params.get("lambda_l2", 0.0)) == 0.0


def _best_runs_per_optimizer(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    pool = [run for run in runs if run.result is not None]
    by_optimizer: dict[str, ExperimentRun] = {}
    for run in pool:
        assert run.result is not None
        previous = by_optimizer.get(run.optimizer_name)
        previous_result = None if previous is None else previous.result
        if previous is None or previous_result is None or run.result.f < previous_result.f:
            by_optimizer[run.optimizer_name] = run
    return list(by_optimizer.values())


def _save_tables(runs: list[ExperimentRun], experiment: OptimizationExperiment) -> dict[str, Path]:
    table_dir = experiment.config.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary = experiment.to_dataframe()
    summary_path = table_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    history_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    for index, run in enumerate(runs):
        rid = _run_id(run, index)
        if run.result is not None:
            for state in run.result.history:
                row: dict[str, object] = {
                    "run_id": rid,
                    "function": run.function_name,
                    "optimizer": run.optimizer_name,
                    "function_params": json.dumps(run.function_params, sort_keys=True, default=str),
                    "optimizer_params": json.dumps(run.params, sort_keys=True, default=str),
                    "iteration": state.iteration,
                    "f": state.f,
                    "grad_norm": np.nan if state.grad is None else float(np.linalg.norm(state.grad)),
                    "x": state.x.tolist(),
                    "n_calls": run.result.n_calls,
                    "n_grad_calls": run.result.n_grad_calls,
                    "message": run.result.message,
                }
                row.update({f"metric_{key}": value for key, value in state.extra_metrics.items()})
                history_rows.append(row)
            if not isinstance(run.result.x, float):
                weights = np.asarray(run.result.x, dtype=np.float64).reshape(-1)
                for coefficient_index, value in enumerate(weights):
                    coefficient_rows.append(
                        {
                            "run_id": rid,
                            "function": run.function_name,
                            "optimizer": run.optimizer_name,
                            "function_params": json.dumps(run.function_params, sort_keys=True, default=str),
                            "optimizer_params": json.dumps(run.params, sort_keys=True, default=str),
                            "coefficient": f"w{coefficient_index}",
                            "value": float(value),
                        }
                    )
        if run.function_params:
            dataset_rows.append(
                {
                    "run_id": rid,
                    "dataset_kind": run.function_params.get("dataset_kind"),
                    "degree": run.function_params.get("degree"),
                    "n_points": run.function_params.get("n_points"),
                    "x_range": json.dumps(run.function_params.get("x_range"), ensure_ascii=False),
                    "noise_variance": run.function_params.get("noise_variance"),
                    "seed": run.function_params.get("seed"),
                    "lambda_l1": run.function_params.get("lambda_l1", 0.0),
                    "lambda_l2": run.function_params.get("lambda_l2", 0.0),
                    "regularize_intercept": run.function_params.get("regularize_intercept", False),
                }
            )

    histories_path = table_dir / "histories.csv"
    pd.DataFrame(history_rows).to_csv(histories_path, index=False)
    paths["histories"] = histories_path

    coefficients_path = table_dir / "coefficients.csv"
    pd.DataFrame(coefficient_rows).to_csv(coefficients_path, index=False)
    paths["coefficients"] = coefficients_path

    datasets_path = table_dir / "datasets.csv"
    pd.DataFrame(dataset_rows).drop_duplicates().to_csv(datasets_path, index=False)
    paths["datasets"] = datasets_path
    return paths


def _instantiate_function(function_name: str, function_params: dict[str, Any]) -> MultivariateFunction | None:
    func = GLOBAL_REGISTRY.get_function(function_name, **function_params)
    return func if isinstance(func, MultivariateFunction) else None


def main() -> None:
    """Execute Lab 5, save readable tables and report-oriented plots."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    runs = experiment.execute()
    table_paths = _save_tables(runs, experiment)
    plot_dir = experiment.config.output_dir / "plots"
    plot_paths: dict[str, Path] = {}
    plots = experiment.config.plots

    if bool(plots.get("predictions", True)):
        selected_optimizer = str(plots.get("selected_optimizer_for_predictions", "LevenbergMarquardt"))
        for function_config in experiment.config.normalized_functions():
            if float(function_config.params.get("lambda_l1", 0.0)) != 0.0 or float(function_config.params.get("lambda_l2", 0.0)) != 0.0:
                continue
            func = _instantiate_function(function_config.name, dict(function_config.params))
            if func is None:
                continue
            family = [
                run
                for run in runs
                if run.function_name == function_config.name
                and run.function_params == dict(function_config.params)
                and run.optimizer_name == selected_optimizer
                and run.result is not None
            ]
            if family:
                stem = _variant_basename(function_config.name, dict(function_config.params))
                plot_paths.update(plot_lab5_regression_fit(func, family, plot_dir, f"{stem}_fit"))

    history_names = plots.get("history_optimizers", [])
    history_filter = {str(name) for name in history_names} if isinstance(history_names, list) else set[str]()
    for function_config in experiment.config.normalized_functions():
        if not _is_unregularized(ExperimentRun(function_config.name, "", 0.0, {}, None, dict(function_config.params))):
            continue
        family = [
            run
            for run in runs
            if run.function_name == function_config.name
            and run.function_params == dict(function_config.params)
            and (not history_filter or run.optimizer_name in history_filter)
            and run.result is not None
        ]
        if not family:
            continue
        best = _best_runs_per_optimizer(family)
        stem = _variant_basename(function_config.name, dict(function_config.params))
        if bool(plots.get("histories", True)):
            plot_paths.update(plot_lab5_loss_history(best, plot_dir, f"{stem}_history"))
            plot_paths.update(plot_lab5_loss_by_gradient_calls(best, plot_dir, f"{stem}_history"))
        if bool(plots.get("coefficients", True)):
            plot_paths.update(plot_lab5_coefficients(best, plot_dir, f"{stem}_coefficients"))

    batch_runs = [
        run
        for run in runs
        if run.optimizer_name == "MiniBatchGradientDescent"
        and _match_params(run, dataset_kind="nonlinear", degree=5)
        and _is_unregularized(run)
        and run.result is not None
    ]
    if bool(plots.get("batch_sizes", True)) and batch_runs:
        plot_paths.update(plot_lab5_batch_size_comparison(batch_runs, plot_dir))

    regularized_runs = [
        run
        for run in runs
        if _match_params(run, dataset_kind="nonlinear", degree=5)
        and run.optimizer_name == "LevenbergMarquardt"
        and run.result is not None
    ]
    if bool(plots.get("regularization", True)) and regularized_runs:
        plot_paths.update(plot_lab5_regularization_comparison(regularized_runs, plot_dir))
        plot_paths.update(plot_lab5_coefficients(regularized_runs, plot_dir, "lab5_regularized_coefficients"))

    if bool(plots.get("method_comparison", True)):
        comparison_runs = [run for run in runs if _is_unregularized(run) and run.result is not None]
        plot_paths.update(plot_lab5_method_comparison(comparison_runs, plot_dir))

    print(f"Lab 5 completed: {len(runs)} runs")
    for name, path in table_paths.items():
        print(f"{name}: {path}")
    if plot_paths:
        print(f"plots: {plot_dir}")


if __name__ == "__main__":
    main()
