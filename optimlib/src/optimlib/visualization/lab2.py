"""Lab 2 specific plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.visualization.base import safe_stem
from optimlib.visualization.contour import _history_points


_ACKLEY_CONSTANT_ALPHAS = (0.1, 0.01, 0.001)


def _ackley_run_label(run: ExperimentRun) -> str:
    if run.optimizer_name == "ConstantStepGD":
        return f"Const alpha={float(run.params['alpha']):g}"
    return {
        "ArmijoBacktracking": "Armijo",
        "StrongWolfe": "Strong Wolfe",
        "SteepestDescent": "Steepest descent",
    }.get(run.optimizer_name, run.optimizer_name)


def _is_selected_ackley_run(run: ExperimentRun, tolerance: float) -> bool:
    if run.result is None or run.function_name != "Ackley":
        return False
    if dict(run.function_params) != {"x0": [-2.0, -2.0]}:
        return False
    if abs(run.tolerance - tolerance) > 1e-12 * max(1.0, abs(tolerance)):
        return False
    if run.optimizer_name == "ConstantStepGD":
        alpha = run.params.get("alpha")
        return alpha is not None and any(abs(float(alpha) - value) <= 1e-12 for value in _ACKLEY_CONSTANT_ALPHAS)
    return run.optimizer_name in {"ArmijoBacktracking", "StrongWolfe", "SteepestDescent"}


def plot_lab2_ackley_trajectories(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str = "ackley_trajectories",
    tolerance: float = 1e-8,
) -> dict[str, Path]:
    """Plot a readable Lab 2 Ackley trajectory comparison for x0=(-2, -2)."""
    selected = [run for run in results if _is_selected_ackley_run(run, tolerance)]
    if not selected:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = -3.0, 1.2, -3.0, 1.2
    xs = np.linspace(x_min, x_max, 360)
    ys = np.linspace(y_min, y_max, 360)
    xx, yy = np.meshgrid(xs, ys)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(
            [func(np.array([x, y], dtype=np.float64)) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)]
        ).reshape(xx.shape)
    z_plot = np.maximum(values, 1e-8)
    finite = z_plot[np.isfinite(z_plot)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    levels = np.geomspace(1e-8, max(vmax, 1e-7), 70)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    contour = ax.contourf(xx, yy, z_plot, levels=levels, cmap="viridis", norm=LogNorm(vmin=1e-8, vmax=max(vmax, 1e-7)))
    ax.contour(xx, yy, z_plot, levels=levels[::6], colors="black", linewidths=0.25, alpha=0.25, norm=LogNorm())
    fig.colorbar(contour, ax=ax, label="f(x)")

    order = {
        ("ConstantStepGD", 0.1): 0,
        ("ConstantStepGD", 0.01): 1,
        ("ConstantStepGD", 0.001): 2,
        ("ArmijoBacktracking", None): 3,
        ("StrongWolfe", None): 4,
        ("SteepestDescent", None): 5,
    }

    def sort_key(run: ExperimentRun) -> int:
        alpha = float(run.params["alpha"]) if run.optimizer_name == "ConstantStepGD" else None
        return order.get((run.optimizer_name, alpha), 99)

    styles = ["#d55e00", "#009e73", "#0072b2", "#cc79a7", "#f0e442", "#56b4e9"]
    linestyles = ["-", "-", "-", "--", "-.", ":"]
    for index, run in enumerate(sorted(selected, key=sort_key)):
        points = _history_points(run)
        if points.size == 0:
            continue
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        points = points[mask]
        if points.size == 0:
            continue
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=styles[index % len(styles)],
            linestyle=linestyles[index % len(linestyles)],
            linewidth=1.8,
            marker="o",
            markersize=3.0,
            markevery=max(1, points.shape[0] // 18),
            label=_ackley_run_label(run),
        )

    x0 = func.initial_point()
    ax.scatter(x0[0], x0[1], marker="X", s=90, c="white", edgecolors="black", linewidths=1.0, label="start")
    ax.scatter(0.0, 0.0, marker="*", s=160, c="gold", edgecolors="black", linewidths=0.9, label="min (0,0)")
    ax.text(x0[0] + 0.08, x0[1] - 0.16, r"$x_0=(-2,-2)$", color="white", fontsize=9, weight="bold")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Ackley: trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.88, ncol=2, borderpad=0.45, handlelength=2.0)
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
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths
