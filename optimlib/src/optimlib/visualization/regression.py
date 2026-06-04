"""Regression plots used by Lab 5."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimlib.core.base import FloatArray
from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.visualization.base import result_weights, safe_stem, set_log_scale_if_positive


@runtime_checkable
class RegressionPlotFunction(Protocol):
    """Function protocol needed for regression fit plots."""

    name: str
    dataset: Any

    def dense_prediction_table(self, w: FloatArray) -> Mapping[str, FloatArray]:
        """Return dense x, true y, and predicted y arrays."""

    def metadata(self) -> Mapping[str, Any]:
        """Return regression metadata."""


def _function_metadata(func: MultivariateFunction) -> dict[str, Any]:
    if isinstance(func, RegressionPlotFunction):
        return dict(func.metadata())
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


def plot_lab5_regression_fit(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
) -> dict[str, Path]:
    """Plot true relation, noisy observations, and fitted regression curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(func, RegressionPlotFunction):
        return {}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(func.dataset.x, func.dataset.y, s=14, alpha=0.55, label="noisy data")
    ax.plot(func.dataset.x, func.dataset.y_true, color="black", linewidth=2.0, label="true relation")
    for run in results:
        weights = result_weights(run)
        if weights is None:
            continue
        table = func.dense_prediction_table(weights)
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
        set_log_scale_if_positive(ax, data, metric)
        ax.set_title(f"{metric} history")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
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
    set_log_scale_if_positive(ax, data, metric)
    ax.set_title(f"{metric} by gradient evaluations")
    ax.set_xlabel("gradient evaluations")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    png = output_dir / f"{basename}_{safe_stem(metric)}_by_grad_calls.png"
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
        set_log_scale_if_positive(ax, data, "loss")
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
        set_log_scale_if_positive(ax, data, metric)
        ax.set_title(f"Regularization comparison: {metric}")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
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
        weights = result_weights(run)
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
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths
