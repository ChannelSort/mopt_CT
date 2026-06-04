"""Plotting functions for trajectories, convergence, and sensitivity."""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from optimlib.core.base import FloatArray
from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction


def _history_points(run: ExperimentRun) -> FloatArray:
    if run.result is None or not run.result.history:
        return np.empty((0, 2), dtype=np.float64)
    points = [state.x for state in run.result.history if state.x.size >= 2 and np.all(np.isfinite(state.x[:2]))]
    return np.vstack(points)[:, :2] if points else np.empty((0, 2), dtype=np.float64)


def _limits(func: MultivariateFunction, results: Iterable[ExperimentRun]) -> tuple[float, float, float, float]:
    points = [func.initial_point()]
    points.extend(func.global_minimizers)
    for run in results:
        history = _history_points(run)
        if history.size:
            points.extend(history)
        if run.result is not None and not isinstance(run.result.x, float):
            points.append(run.result.x[:2])
    stacked = np.vstack(points)
    lower = np.nanpercentile(stacked, 2.0, axis=0)
    upper = np.nanpercentile(stacked, 98.0, axis=0)
    span = np.maximum(upper - lower, 1.0)
    pad = 0.3 * span
    return float(lower[0] - pad[0]), float(upper[0] + pad[0]), float(lower[1] - pad[1]), float(upper[1] + pad[1])


def plot_contours_and_trajectories(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str | None = None,
) -> dict[str, Path]:
    """Plot contours with clipped optimization trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = _limits(func, results)
    xs = np.linspace(x_min, x_max, 220)
    ys = np.linspace(y_min, y_max, 220)
    xx, yy = np.meshgrid(xs, ys)
    values = np.array([func(np.array([x, y], dtype=np.float64)) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)]).reshape(xx.shape)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    use_log = positive.size > 0 and float(np.max(positive) / np.min(positive)) > 1e3

    fig, ax = plt.subplots(figsize=(8, 6))
    if use_log:
        contour = ax.contourf(xx, yy, np.maximum(values, np.min(positive)), levels=40, cmap="viridis", norm=LogNorm())
    else:
        contour = ax.contourf(xx, yy, values, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=ax)

    for run in results:
        points = _history_points(run)
        if points.size == 0:
            continue
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        points = points[mask]
        if points.size == 0:
            continue
        ax.plot(points[:, 0], points[:, 1], marker="o", markersize=2.5, linewidth=1.0, label=f"{run.optimizer_name} {run.params}")
        if points.shape[0] > 1:
            delta = np.diff(points, axis=0)
            ax.quiver(points[:-1, 0], points[:-1, 1], delta[:, 0], delta[:, 1], angles="xy", scale_units="xy", scale=1.0, width=0.002)
    for minimizer in func.global_minimizers:
        ax.scatter(minimizer[0], minimizer[1], marker="*", s=120, c="white", edgecolors="black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{func.name}: trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if results:
        ax.legend(fontsize=6)
    plt.tight_layout()
    stem = basename or f"{func.name}_trajectories"
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_convergence(results: list[ExperimentRun], output_dir: Path, basename: str = "convergence", f_min: float = 0.0) -> dict[str, Path]:
    """Plot ``||grad||``, ``|f-f*|``, and step sizes on log axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for run in results:
        if run.result is None or not run.result.history:
            continue
        iterations = np.array([state.iteration for state in run.result.history])
        grad_norm = np.array([np.linalg.norm(state.grad) if state.grad is not None else np.nan for state in run.result.history])
        f_error = np.array([abs(state.f - f_min) for state in run.result.history])
        steps = np.array([state.step_size if state.step_size is not None else np.nan for state in run.result.history])
        label = f"{run.optimizer_name} {run.params}"
        axes[0].semilogy(iterations, np.maximum(grad_norm, 1e-300), label=label)
        axes[1].semilogy(iterations, np.maximum(f_error, 1e-300), label=label)
        axes[2].semilogy(iterations, np.maximum(steps, 1e-300), label=label)
    axes[0].set_ylabel("||grad||")
    axes[1].set_ylabel("|f-f*|")
    axes[2].set_ylabel("alpha / step")
    axes[2].set_xlabel("iteration")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    if results:
        axes[0].legend(fontsize=6)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_param_sensitivity(
    results: list[ExperimentRun],
    output_dir: Path,
    x_param: str,
    y_param: str | None = None,
    metric: str = "n_iter",
    basename: str = "param_sensitivity",
) -> dict[str, Path]:
    """Plot line sensitivity for one parameter or heatmap for two parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in results:
        if run.result is None:
            continue
        rows.append({**run.params, metric: getattr(run.result, metric)})
    if not rows:
        return {}
    import pandas as pd

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    if y_param is None:
        sns.lineplot(data=data, x=x_param, y=metric, marker="o", ax=ax)
    else:
        pivot = data.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap="magma", ax=ax)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_lab2_constant_step_alpha(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "lab2_constant_step_iterations_vs_alpha",
    tolerance: float = 1e-8,
) -> dict[str, Path]:
    """Plot constant-step gradient descent iterations against ``alpha``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for run in results:
        if run.optimizer_name != "ConstantStepGD" or run.result is None:
            continue
        if run.function_name not in {"WellConditionedQuadratic", "IllConditionedQuadratic"}:
            continue
        if abs(run.tolerance - tolerance) > 1e-12 * max(1.0, abs(tolerance)):
            continue
        alpha = run.params.get("alpha")
        if alpha is None:
            continue
        rows.append(
            {
                "function": run.function_name,
                "alpha": float(alpha),
                "n_iter": run.result.n_iter,
                "converged": bool(run.result.converged),
            }
        )
    if not rows:
        return {}
    data = pd.DataFrame(rows).sort_values(["function", "alpha"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x="alpha", y="n_iter", hue="function", style="converged", marker="o", ax=ax)
    ax.set_xscale("log")
    ax.set_title("Constant-step GD: iterations vs alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("iterations")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_lab2_tolerance_dependencies(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "lab2_line_search_vs_tolerance",
    metrics: Sequence[str] = ("n_iter", "n_calls", "n_grad_calls"),
) -> dict[str, Path]:
    """Plot Lab 2 line-search metrics as functions of stopping tolerance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected_optimizers = {"ArmijoBacktracking", "StrongWolfe", "SteepestDescent"}
    selected_functions = {"WellConditionedQuadratic", "IllConditionedQuadratic"}
    for run in results:
        if run.optimizer_name not in selected_optimizers or run.result is None:
            continue
        if run.function_name not in selected_functions:
            continue
        for metric in metrics:
            value = getattr(run.result, metric, None)
            if isinstance(value, int | float | np.integer | np.floating):
                rows.append(
                    {
                        "function": run.function_name,
                        "optimizer": run.optimizer_name,
                        "tolerance": run.tolerance,
                        "metric": metric,
                        "value": float(value),
                        "converged": bool(run.result.converged),
                    }
                )
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_data = data[data["metric"] == metric]
        if metric_data.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, function_name in zip(axes, sorted(selected_functions), strict=True):
            subset = metric_data[metric_data["function"] == function_name]
            if subset.empty:
                continue
            sns.lineplot(data=subset, x="tolerance", y="value", hue="optimizer", marker="o", ax=ax)
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_title(function_name)
            ax.set_xlabel("tolerance")
            ax.set_ylabel(metric)
            ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{_safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths


def _safe_stem(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def _numeric_metric(run: ExperimentRun, metric: str) -> float | None:
    if run.result is None:
        return None
    value = getattr(run.result, metric, None)
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return None


def _lab4_metric_rows(results: list[ExperimentRun], metrics: Sequence[str] | None = None) -> list[dict[str, object]]:
    selected_metrics = tuple(metrics or ("n_iter", "n_calls", "n_grad_calls", "n_hessian_calls"))
    rows: list[dict[str, object]] = []
    for run in results:
        n = run.function_params.get("n")
        k = run.function_params.get("k")
        if n is None or k is None:
            continue
        for metric in selected_metrics:
            value = _numeric_metric(run, metric)
            if value is None:
                continue
            rows.append(
                {
                    "function": run.function_name,
                    "optimizer": run.optimizer_name,
                    "n": int(n),
                    "k": float(k),
                    "seed": run.function_params.get("seed"),
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def plot_lab4_metric_tables(
    results: list[ExperimentRun],
    output_dir: Path,
    metrics: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Save heatmap tables of Lab 4 metrics by dimension and condition number."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _lab4_metric_rows(results, metrics)
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    paths: dict[str, Path] = {}
    for metric in sorted(data["metric"].unique()):
        metric_data = data[data["metric"] == metric]
        for optimizer in sorted(metric_data["optimizer"].unique()):
            subset = metric_data[metric_data["optimizer"] == optimizer]
            pivot = subset.pivot_table(index="k", columns="n", values="value", aggfunc="mean")
            stem = f"lab4_table_{_safe_stem(str(metric))}_{_safe_stem(str(optimizer))}"
            csv_path = output_dir / f"{stem}.csv"
            pivot.to_csv(csv_path)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(pivot, annot=True, fmt=".3g", cmap="magma", ax=ax)
            ax.set_title(f"{optimizer}: {metric}")
            ax.set_xlabel("n")
            ax.set_ylabel("cond(A)")
            plt.tight_layout()
            png_path = output_dir / f"{stem}.png"
            fig.savefig(png_path, dpi=300)
            plt.close(fig)
            paths[f"{stem}_csv"] = csv_path
            paths[f"{stem}_png"] = png_path
    return paths


def plot_lab4_metric_dependencies(
    results: list[ExperimentRun],
    output_dir: Path,
    metrics: Sequence[str] | None = None,
    fixed_k: float | None = None,
    fixed_n: int | None = None,
) -> dict[str, Path]:
    """Plot Lab 4 metric dependencies on ``n`` and condition number ``k``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _lab4_metric_rows(results, metrics)
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    paths: dict[str, Path] = {}
    chosen_k = fixed_k if fixed_k is not None else float(sorted(data["k"].unique())[0])
    chosen_n = fixed_n if fixed_n is not None else int(sorted(data["n"].unique())[0])

    for metric in sorted(data["metric"].unique()):
        metric_data = data[data["metric"] == metric]
        by_n = metric_data[np.isclose(metric_data["k"].astype(float), chosen_k)]
        if not by_n.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=by_n, x="n", y="value", hue="optimizer", marker="o", errorbar=None, ax=ax)
            ax.set_title(f"{metric} vs n at k={chosen_k:g}")
            ax.set_xlabel("n")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            png_path = output_dir / f"lab4_{_safe_stem(str(metric))}_vs_n_k_{chosen_k:g}.png"
            fig.savefig(png_path, dpi=300)
            plt.close(fig)
            paths[f"{metric}_vs_n"] = png_path

        by_k = metric_data[metric_data["n"].astype(int) == chosen_n]
        if not by_k.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=by_k, x="k", y="value", hue="optimizer", marker="o", errorbar=None, ax=ax)
            ax.set_xscale("log")
            ax.set_title(f"{metric} vs cond(A) at n={chosen_n}")
            ax.set_xlabel("cond(A)")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            png_path = output_dir / f"lab4_{_safe_stem(str(metric))}_vs_k_n_{chosen_n}.png"
            fig.savefig(png_path, dpi=300)
            plt.close(fig)
            paths[f"{metric}_vs_k"] = png_path
    return paths


def plot_lab4_optimizer_comparison(
    results: list[ExperimentRun],
    output_dir: Path,
    metrics: Sequence[str] | None = None,
    reference_optimizer: str = "ScipyNewtonCG",
) -> dict[str, Path]:
    """Compare custom methods with the SciPy Newton-CG reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _lab4_metric_rows(results, metrics)
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    paths: dict[str, Path] = {}
    for metric in sorted(data["metric"].unique()):
        metric_data = data[data["metric"] == metric]
        if reference_optimizer not in set(metric_data["optimizer"]):
            continue
        grouped = metric_data.groupby("optimizer", as_index=False)["value"].mean()
        grouped["is_reference"] = grouped["optimizer"] == reference_optimizer
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=grouped, x="optimizer", y="value", hue="is_reference", dodge=False, palette=["#4C78A8", "#F58518"], ax=ax)
        ax.set_title(f"Mean {metric}: custom methods vs {reference_optimizer}")
        ax.set_xlabel("optimizer")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        plt.tight_layout()
        png_path = output_dir / f"lab4_comparison_{_safe_stem(str(metric))}.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        paths[f"comparison_{metric}"] = png_path
    return paths


def plot_lab4_lbfgs_memory(
    results: list[ExperimentRun],
    output_dir: Path,
    metric: str = "n_iter",
) -> dict[str, Path]:
    """Plot the influence of L-BFGS memory size on a selected metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for run in results:
        if run.optimizer_name != "LBFGS":
            continue
        memory = run.params.get("m", run.result.metadata.get("m") if run.result is not None else None)
        value = _numeric_metric(run, metric)
        if memory is None or value is None:
            continue
        rows.append(
            {
                "m": int(memory),
                "value": value,
                "n": run.function_params.get("n"),
                "k": run.function_params.get("k"),
                "function": run.function_name,
            }
        )
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x="m", y="value", marker="o", errorbar=None, ax=ax)
    ax.set_title(f"L-BFGS memory influence on {metric}")
    ax.set_xlabel("memory size m")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    png_path = output_dir / f"lab4_lbfgs_memory_{_safe_stem(metric)}.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {"png": png_path}


def _result_weights(run: ExperimentRun) -> FloatArray | None:
    if run.result is None or isinstance(run.result.x, float):
        return None
    return np.asarray(run.result.x, dtype=np.float64).reshape(-1)


def _function_metadata(func: MultivariateFunction) -> dict[str, Any]:
    method = getattr(func, "metadata", None)
    if callable(method):
        value = cast(Callable[[], Mapping[str, Any]], method)()
        return dict(value)
    return {}


def _history_dataframe(results: list[ExperimentRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in results:
        if run.result is None:
            continue
        for state in run.result.history:
            row: dict[str, object] = {
                "optimizer": run.optimizer_name,
                "params": str(run.params),
                "iteration": state.iteration,
                "f": state.f,
                "grad_norm": np.nan if state.grad is None else float(np.linalg.norm(state.grad)),
            }
            row.update({key: value for key, value in state.extra_metrics.items() if isinstance(value, int | float | str | bool)})
            rows.append(row)
    return pd.DataFrame(rows)


def _set_log_scale_if_positive(ax: Axes, data: pd.DataFrame, metric: str) -> None:
    if metric in data and bool((pd.to_numeric(data[metric], errors="coerce") > 0.0).any()):
        ax.set_yscale("log")


def plot_lab5_regression_fit(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
) -> dict[str, Path]:
    """Plot true relation, noisy observations, and fitted regression curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = getattr(func, "dataset", None)
    dense_method = getattr(func, "dense_prediction_table", None)
    if dataset is None or not callable(dense_method):
        return {}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dataset.x, dataset.y, s=14, alpha=0.55, label="noisy data")
    ax.plot(dataset.x, dataset.y_true, color="black", linewidth=2.0, label="true relation")
    for run in results:
        weights = _result_weights(run)
        if weights is None:
            continue
        table = cast(Callable[[FloatArray], Mapping[str, FloatArray]], dense_method)(weights)
        label = f"{run.optimizer_name} {run.params}".strip()
        ax.plot(table["x"], table["y_pred"], linewidth=1.4, label=label)
    meta = _function_metadata(func)
    ax.set_title(f"{meta.get('dataset_kind', func.name)} degree={meta.get('degree', '?')} regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_lab5_loss_history(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
    metrics: Sequence[str] = ("loss", "empirical_risk", "l1_term", "l2_term"),
) -> dict[str, Path]:
    """Plot optimization histories for selected loss components."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _history_dataframe(results)
    if data.empty:
        return {}
    paths: dict[str, Path] = {}
    for metric in metrics:
        if metric not in data:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=data, x="iteration", y=metric, hue="optimizer", style="params", errorbar=None, ax=ax)
        _set_log_scale_if_positive(ax, data, metric)
        ax.set_title(f"{metric} history")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{_safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths


def plot_lab5_loss_by_gradient_calls(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
    metric: str = "loss",
) -> dict[str, Path]:
    """Plot loss dynamics against gradient evaluation count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _history_dataframe(results)
    if data.empty or metric not in data or "gradient_evaluations" not in data:
        return {}
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x="gradient_evaluations", y=metric, hue="optimizer", style="params", errorbar=None, ax=ax)
    _set_log_scale_if_positive(ax, data, metric)
    ax.set_title(f"{metric} by gradient evaluations")
    ax.set_xlabel("gradient evaluations")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    png = output_dir / f"{basename}_{_safe_stem(metric)}_by_grad_calls.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_lab5_batch_size_comparison(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "lab5_batch_size_comparison",
) -> dict[str, Path]:
    """Compare mini-batch loss histories by batch size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _history_dataframe([run for run in results if run.optimizer_name == "MiniBatchGradientDescent"])
    if data.empty or "batch_size" not in data or "loss" not in data:
        return {}
    paths: dict[str, Path] = {}
    for x_axis, suffix in (("epoch", "by_epoch"), ("gradient_evaluations", "by_grad_calls")):
        if x_axis not in data:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=data, x=x_axis, y="loss", hue="batch_size", palette="viridis", errorbar=None, ax=ax)
        _set_log_scale_if_positive(ax, data, "loss")
        ax.set_title(f"Mini-batch loss {suffix.replace('_', ' ')}")
        ax.set_xlabel(x_axis)
        ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{suffix}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[suffix] = png
    return paths


def plot_lab5_regularization_comparison(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "lab5_regularization",
) -> dict[str, Path]:
    """Plot regularization loss components for comparable runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _history_dataframe(results)
    if data.empty:
        return {}
    paths: dict[str, Path] = {}
    for metric in ("loss", "empirical_risk", "l1_term", "l2_term"):
        if metric not in data:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=data, x="iteration", y=metric, hue="params", style="optimizer", errorbar=None, ax=ax)
        _set_log_scale_if_positive(ax, data, metric)
        ax.set_title(f"Regularization comparison: {metric}")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{_safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths


def plot_lab5_coefficients(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "lab5_coefficients",
) -> dict[str, Path]:
    """Bar plot of learned polynomial coefficients."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for run in results:
        weights = _result_weights(run)
        if weights is None:
            continue
        label = f"{run.optimizer_name} {run.params}".strip()
        for index, value in enumerate(weights):
            rows.append({"run": label, "coefficient": f"w{index}", "value": float(value)})
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x="coefficient", y="value", hue="run", ax=ax)
    ax.set_title("Learned coefficients")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=6)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_lab5_method_comparison(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "lab5_method_comparison",
) -> dict[str, Path]:
    """Compare final optimization metrics across methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for run in results:
        if run.result is None:
            continue
        row = {
            "optimizer": run.optimizer_name,
            "params": str(run.params),
            "loss": float(run.result.f),
            "n_iter": run.result.n_iter,
            "n_grad_calls": run.result.n_grad_calls,
            "elapsed_seconds": float(run.result.metadata.get("elapsed_seconds", np.nan)),
            "empirical_risk": float(run.result.metadata.get("empirical_risk", np.nan)),
            "converged": run.result.converged,
        }
        rows.append(row)
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    csv = output_dir / f"{basename}.csv"
    data.to_csv(csv, index=False)
    paths: dict[str, Path] = {"csv": csv}
    for metric in ("loss", "empirical_risk", "n_iter", "n_grad_calls", "elapsed_seconds"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=data, x="optimizer", y=metric, estimator="mean", errorbar=None, ax=ax)
        ax.set_title(f"Method comparison: {metric}")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{_safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths
