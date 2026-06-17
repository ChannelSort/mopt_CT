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


LAB5_LINE_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#17becf",
    "#393b79",
    "#ad494a",
    "#a55194",
    "#e6550d",
    "#6b6ecf",
    "#f7b6d2",
)


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


def _format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6g}"


def _latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _compact_params(params: Mapping[str, Any]) -> str:
    if not params:
        return "{}"
    parts: list[str] = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, int | float):
            parts.append(f"{key}={float(value):g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _optimizer_label(name: str) -> str:
    labels = {
        "AnalyticalLinearRegression1D": "analytic",
        "StochasticGradientDescent": "SGD",
        "MiniBatchGradientDescent": "mini-batch",
        "GaussNewton": "GN",
        "LevenbergMarquardt": "LM",
    }
    return labels.get(name, name)


def _regularization_label(params: Mapping[str, Any], *, include_none: bool = False) -> str:
    l1 = float(params.get("lambda_l1", 0.0) or 0.0)
    l2 = float(params.get("lambda_l2", 0.0) or 0.0)
    if l1 == 0.0 and l2 == 0.0:
        return "none" if include_none else ""
    if l1 != 0.0 and l2 != 0.0:
        return f"Elastic Net l1={_format_float(l1)}, l2={_format_float(l2)}"
    if l1 != 0.0:
        return f"L1={_format_float(l1)}"
    return f"L2={_format_float(l2)}"


def _run_label(run: ExperimentRun, *, include_regularization: bool = True) -> str:
    label = _optimizer_label(run.optimizer_name)
    if run.optimizer_name == "MiniBatchGradientDescent" and "batch_size" in run.params:
        label = f"{label} b={run.params['batch_size']}"
    if include_regularization:
        reg = _regularization_label(run.function_params)
        if reg:
            label = f"{label}, {reg}"
    return label


def _palette_for(data: pd.DataFrame, column: str) -> dict[object, str]:
    levels = list(pd.Series(data[column]).dropna().unique())
    return {level: LAB5_LINE_COLORS[index % len(LAB5_LINE_COLORS)] for index, level in enumerate(levels)}


def _history_dataframe(results: list[ExperimentRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in results:
        if run.result is None:
            continue
        for state in run.result.history:
            row: dict[str, object] = {
                "optimizer": run.optimizer_name,
                "params": str(run.params),
                "label": _run_label(run),
                "regularization": _regularization_label(run.function_params, include_none=True),
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_prop_cycle(color=LAB5_LINE_COLORS)
    ax.scatter(func.dataset.x, func.dataset.y, s=14, alpha=0.55, label="noisy data")
    ax.plot(func.dataset.x, func.dataset.y_true, color="black", linewidth=2.0, label="true relation")
    for run in results:
        weights = result_weights(run)
        if weights is None:
            continue
        table = func.dense_prediction_table(weights)
        label = _run_label(run)
        ax.plot(table["x"], table["y_pred"], linewidth=1.4, label=label)
    meta = _function_metadata(func)
    ax.set_title(f"{meta.get('dataset_kind', func.name)} degree={meta.get('degree', '?')} regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x="iteration", y=metric, hue="label", palette=_palette_for(data, "label"), errorbar=None, ax=ax)
        set_log_scale_if_positive(ax, data, metric)
        ax.set_title(f"{metric} history")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="gradient_evaluations", y=metric, hue="label", palette=_palette_for(data, "label"), errorbar=None, ax=ax)
    set_log_scale_if_positive(ax, data, metric)
    ax.set_title(f"{metric} by gradient evaluations")
    ax.set_xlabel("gradient evaluations")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x=x_axis, y="loss", hue="batch_size", palette=_palette_for(data, "batch_size"), errorbar=None, ax=ax)
        set_log_scale_if_positive(ax, data, "loss")
        ax.set_title(f"Mini-batch loss {suffix.replace('_', ' ')}")
        ax.set_xlabel(x_axis)
        ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=data,
            x="iteration",
            y=metric,
            hue="regularization",
            palette=_palette_for(data, "regularization"),
            errorbar=None,
            ax=ax,
        )
        set_log_scale_if_positive(ax, data, metric)
        ax.set_title(f"Regularization comparison: {metric}")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
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
        label = _run_label(run)
        for index, value in enumerate(weights):
            rows.append({"run": label, "coefficient": f"w{index}", "value": float(value)})
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x="coefficient", y="value", hue="run", palette=_palette_for(data, "run"), ax=ax)
    ax.set_title("Learned coefficients")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7)
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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=data,
            x="optimizer",
            y=metric,
            hue="optimizer",
            estimator="mean",
            errorbar=None,
            palette=_palette_for(data, "optimizer"),
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Method comparison: {metric}")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths


def _empirical_risk(run: ExperimentRun) -> float | None:
    if run.result is None:
        return None
    value = run.result.metadata.get("empirical_risk")
    if isinstance(value, int | float):
        return float(value)
    if run.result.history:
        history_value = run.result.history[-1].extra_metrics.get("empirical_risk")
        if isinstance(history_value, int | float):
            return float(history_value)
    return None


def _short_stop(run: ExperimentRun) -> str:
    if run.result is None:
        return "error"
    message = run.result.message.lower()
    if "mini_batch_study_not_applicable" in message:
        return "skipped_batch_study"
    if "analytical_solution_not_applicable" in message:
        return "not_applicable"
    if "maximum epochs" in message:
        return "max_epochs"
    if "maximum iterations" in message:
        return "max_iter"
    if "gradient norm" in message:
        return "grad_tol"
    if "step tolerance" in message:
        return "step_tol"
    if run.result.converged:
        return "converged"
    return "stopped"


def _dataset_label(run: ExperimentRun) -> str:
    dataset = run.function_params.get("dataset_kind", "--")
    n_points = run.function_params.get("n_points", "--")
    noise = run.function_params.get("noise_variance", "--")
    return f"{dataset}, m={n_points}, noise={noise}"


def save_lab5_report_tables(runs: list[ExperimentRun], output_dir: Path, block_size: int = 34) -> dict[str, Path]:
    """Save compact LaTeX tables used by the Lab 5 report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary_report_scaled.tex"
    sorted_runs = sorted(
        runs,
        key=lambda run: (
            str(run.function_params.get("dataset_kind", "")),
            int(run.function_params.get("degree", 0) or 0),
            float(run.function_params.get("lambda_l1", 0.0) or 0.0),
            float(run.function_params.get("lambda_l2", 0.0) or 0.0),
            run.optimizer_name,
            str(run.params),
        ),
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
    ]
    for block_index, start in enumerate(range(0, len(sorted_runs), block_size), start=1):
        block = sorted_runs[start : start + block_size]
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{Полная таблица экспериментов, блок {block_index}}}",
                r"\fitreporttable{%",
                r"\begin{tabular}{lllrrrrrrrl}",
                r"\toprule",
                r"Data & Reg & Method & conv & $f$ & $Q$ & iter & $N_f$ & $N_g$ & time & Stop\\",
                r"\midrule",
            ]
        )
        for run in block:
            result = run.result
            row = [
                f"{_dataset_label(run)}, d={run.function_params.get('degree', '--')}",
                _regularization_label(run.function_params, include_none=True),
                f"{_optimizer_label(run.optimizer_name)} {_compact_params(run.params)}".strip(),
                "--" if result is None else ("yes" if result.converged else "no"),
                "--" if result is None else _format_float(result.f),
                _format_float(_empirical_risk(run)),
                "--" if result is None else str(result.n_iter),
                "--" if result is None else str(result.n_calls),
                "--" if result is None else str(result.n_grad_calls),
                "--" if result is None else _format_float(float(result.metadata.get("elapsed_seconds", np.nan))),
                _short_stop(run),
            ]
            lines.append(" & ".join(_latex_escape(item) for item in row) + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_report_scaled": path}
