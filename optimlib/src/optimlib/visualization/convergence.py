"""Generic convergence and parameter-sensitivity plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimlib.experiment.runner import ExperimentRun
from optimlib.utils.numerics import safe_norm


def plot_convergence(results: list[ExperimentRun], output_dir: Path, basename: str = "convergence", f_min: float = 0.0) -> dict[str, Path]:
    """Plot ``||grad||``, ``|f-f*|``, and step sizes on log axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for run in results:
        if run.result is None or not run.result.history:
            continue
        iterations = np.array([state.iteration for state in run.result.history])
        grad_norm = np.array([safe_norm(state.grad) if state.grad is not None else np.nan for state in run.result.history])
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
    title: str | None = None,
    max_iter: int | None = None,
    y_label: str | None = None,
    y_log: bool = False,
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

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    if y_param is None:
        sns.lineplot(data=data, x=x_param, y=metric, marker="o", ax=ax)
        if x_param == "alpha":
            ax.set_xscale("log")
        if metric == "n_iter":
            inferred_limit = max_iter or int(np.nanmax(data[metric]))
            ax.set_ylim(0.0, inferred_limit * 1.05)
            ax.axhline(inferred_limit, color="black", linestyle="--", linewidth=1.0, alpha=0.65, label=f"max_iter = {inferred_limit}")
            values = data[metric].dropna().to_numpy(dtype=float)
            if values.size > 0 and np.allclose(values, values[0]):
                message = "all runs reached max_iter" if max_iter is not None and np.allclose(values, max_iter) else "all runs have same n_iter"
                ax.text(
                    0.5,
                    0.08,
                    message,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.5"},
                )
            ax.legend(fontsize=8)
        elif y_log:
            positive = data[metric][data[metric] > 0]
            if not positive.empty:
                ax.set_yscale("log")
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_label or metric)
    else:
        pivot = data.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap="magma", ax=ax)
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25) if y_param is None else None
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}
