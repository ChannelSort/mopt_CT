"""Lab 2 specific plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimlib.experiment.runner import ExperimentRun
from optimlib.visualization.base import safe_stem


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
