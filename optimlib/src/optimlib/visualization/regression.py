"""Regression plots used by Lab 5."""

from __future__ import annotations

import json
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

METRIC_LABELS: dict[str, str] = {
    "loss": r"full objective $L(w)$",
    "empirical_risk": r"empirical risk $Q(w)$",
    "l1_term": r"L1 regularization term",
    "l2_term": r"L2 regularization term",
    "elapsed_seconds": "elapsed seconds",
}


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


def _regularization_family(params: Mapping[str, Any]) -> str:
    l1 = float(params.get("lambda_l1", 0.0) or 0.0)
    l2 = float(params.get("lambda_l2", 0.0) or 0.0)
    if l1 == 0.0 and l2 == 0.0:
        return "none"
    if l1 != 0.0 and l2 != 0.0:
        return "Elastic Net"
    if l1 != 0.0:
        return "L1"
    return "L2"


def _batch_label(value: object) -> str:
    try:
        batch = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if batch == 1:
        return "1 (SGD)"
    if batch == 120:
        return "120 (full)"
    return str(batch)


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
    if "lm_damping_max_reached" in message:
        return "lm_damping_max"
    if run.result.converged:
        return "converged"
    return "stopped"


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


def _run_label(run: ExperimentRun, *, include_regularization: bool = True, include_stats: bool = False) -> str:
    label = _optimizer_label(run.optimizer_name)
    if run.optimizer_name == "MiniBatchGradientDescent" and "batch_size" in run.params:
        batch = int(run.params["batch_size"])
        label = f"SGD-style b=1" if batch == 1 else f"{label} b={batch}"
    if include_regularization:
        reg = _regularization_label(run.function_params)
        if reg:
            label = f"{label}, {reg}"
    if include_stats and run.result is not None:
        risk = _empirical_risk(run)
        label = (
            f"{label}; L={_format_float(run.result.f)}, "
            f"Q={_format_float(risk)}, iter={run.result.n_iter}, {_short_stop(run)}"
        )
    return label


def _subplot_grid(count: int, *, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, count))
    rows = int(np.ceil(count / cols))
    return rows, cols


def _palette_for(data: pd.DataFrame, column: str) -> dict[object, str]:
    levels = list(pd.Series(data[column]).dropna().unique())
    return {level: LAB5_LINE_COLORS[index % len(LAB5_LINE_COLORS)] for index, level in enumerate(levels)}


def _history_dataframe(results: list[ExperimentRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in results:
        if run.result is None:
            continue
        n_points = int(run.function_params.get("n_points", 0) or 0)
        for state in run.result.history:
            epoch = state.extra_metrics.get("epoch", state.iteration)
            row: dict[str, object] = {
                "optimizer": run.optimizer_name,
                "params": str(run.params),
                "label": _run_label(run, include_stats=True),
                "regularization": _regularization_label(run.function_params, include_none=True),
                "iteration": state.iteration,
                "f": state.f,
                "grad_norm": np.nan if state.grad is None else float(np.linalg.norm(state.grad)),
                "processed_samples": int(epoch) * n_points if isinstance(epoch, int | float) and n_points > 0 else np.nan,
            }
            row.update({key: value for key, value in state.extra_metrics.items() if isinstance(value, int | float | str | bool)})
            rows.append(row)
    return pd.DataFrame(rows)


def _draw_regression_background(ax: plt.Axes, func: RegressionPlotFunction) -> None:
    ax.scatter(func.dataset.x, func.dataset.y, s=10, alpha=0.28, color="#4C78A8", label="noisy data")
    ax.plot(func.dataset.x, func.dataset.y_true, color="black", linewidth=1.8, label="true relation")


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
    _draw_regression_background(ax, func)
    for run in results:
        weights = result_weights(run)
        if weights is None:
            continue
        table = func.dense_prediction_table(weights)
        label = _run_label(run, include_stats=True)
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


def plot_lab5_regression_fit_panels(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(func, RegressionPlotFunction):
        return {}
    selected = [run for run in results if result_weights(run) is not None]
    if not selected:
        return {}
    rows, cols = _subplot_grid(len(selected), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.8 * rows), squeeze=False, sharex=True, sharey=True)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(selected):
            ax.axis("off")
            continue
        run = selected[index]
        weights = result_weights(run)
        assert weights is not None
        table = func.dense_prediction_table(weights)
        _draw_regression_background(ax, func)
        ax.plot(table["x"], table["y_pred"], linewidth=2.0, color="#D55E00", label="prediction")
        ax.set_title(_run_label(run, include_stats=True), fontsize=9)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=7, loc="best")
    meta = _function_metadata(func)
    fig.suptitle(f"{meta.get('dataset_kind', func.name)} degree={meta.get('degree', '?')}: fits by method", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    suffix = "panels" if basename.endswith("_fit") else "fit_panels"
    png = output_dir / f"{basename}_{suffix}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"panels": png}


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
        sns.lineplot(data=data, x="iteration", y=metric, hue="label", palette=_palette_for(data, "label"), marker=None, errorbar=None, ax=ax)
        final_points = data.sort_values("iteration").groupby("label", as_index=False).tail(1)
        sns.scatterplot(data=final_points, x="iteration", y=metric, hue="label", palette=_palette_for(data, "label"), legend=False, s=36, ax=ax)
        set_log_scale_if_positive(ax, data, metric)
        label = METRIC_LABELS.get(metric, metric)
        ax.set_title(f"{label} history")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        plt.tight_layout()
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths


def plot_lab5_loss_history_panels(
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
    metric: str = "loss",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _history_dataframe(results)
    if data.empty or metric not in data:
        return {}
    labels = list(data["label"].dropna().unique())
    rows, cols = _subplot_grid(len(labels), max_cols=2)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.8 * rows), squeeze=False, sharex=False, sharey=True)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(labels):
            ax.axis("off")
            continue
        label_value = labels[index]
        subset = data[data["label"] == label_value]
        ax.plot(subset["iteration"], subset[metric], marker="o" if len(subset) <= 4 else None, linewidth=2.0)
        if len(subset) > 4:
            last = subset.sort_values("iteration").tail(1)
            ax.scatter(last["iteration"], last[metric], s=40, color="#D55E00")
        set_log_scale_if_positive(ax, subset, metric)
        ax.set_title(str(label_value), fontsize=9)
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    png = output_dir / f"{basename}_{safe_stem(metric)}_panels.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {metric: png}


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
    sns.lineplot(data=data, x="gradient_evaluations", y=metric, hue="label", palette=_palette_for(data, "label"), marker=None, errorbar=None, ax=ax)
    final_points = data.sort_values("gradient_evaluations").groupby("label", as_index=False).tail(1)
    sns.scatterplot(data=final_points, x="gradient_evaluations", y=metric, hue="label", palette=_palette_for(data, "label"), legend=False, s=36, ax=ax)
    set_log_scale_if_positive(ax, data, metric)
    label = METRIC_LABELS.get(metric, metric)
    ax.set_title(f"{label} by gradient evaluations")
    ax.set_xlabel("gradient evaluations / optimizer updates")
    ax.set_ylabel(label)
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
    selected_runs = [run for run in results if run.optimizer_name == "MiniBatchGradientDescent" and "batch_size" in run.params]
    data = _history_dataframe(selected_runs)
    if data.empty or "batch_size" not in data or "loss" not in data:
        return {}
    data["batch_label"] = data["batch_size"].map(_batch_label)
    paths: dict[str, Path] = {}
    for x_axis, suffix in (("epoch", "by_epoch"), ("gradient_evaluations", "by_grad_calls"), ("processed_samples", "by_processed_samples")):
        if x_axis not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x=x_axis, y="loss", hue="batch_label", palette=_palette_for(data, "batch_label"), marker=None, linewidth=2.0, alpha=0.9, errorbar=None, ax=ax)
        final_points = data.sort_values(x_axis).groupby("batch_label", as_index=False).tail(1)
        sns.scatterplot(data=final_points, x=x_axis, y="loss", hue="batch_label", palette=_palette_for(data, "batch_label"), legend=False, s=32, ax=ax)
        set_log_scale_if_positive(ax, data, "loss")
        ax.set_title(f"Mini-batch full objective L(w) {suffix.replace('_', ' ')}")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(METRIC_LABELS["loss"])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        plt.tight_layout()
        png = output_dir / f"{basename}_{suffix}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[suffix] = png
    summary_rows: list[dict[str, object]] = []
    for run in selected_runs:
        if run.result is None or not run.result.history:
            continue
        batch = int(run.params.get("batch_size", run.result.metadata.get("batch_size", 0)))
        first_loss = float(run.result.history[0].extra_metrics.get("loss", run.result.history[0].f))
        final_loss = float(run.result.metadata.get("loss", run.result.f))
        grad_calls = max(1, int(run.result.n_grad_calls))
        processed = int(run.function_params.get("n_points", 0) or 0) * int(run.result.metadata.get("epochs", run.result.n_iter))
        n_samples = int(run.function_params.get("n_points", 0) or 0)
        gradient_noise_proxy = 1.0 / max(1, batch)
        if n_samples > 1:
            gradient_noise_proxy = max(0.0, (n_samples - batch) / (batch * (n_samples - 1)))
        summary_rows.append(
            {
                "batch": batch,
                "batch_label": _batch_label(batch),
                "final_loss": final_loss,
                "empirical_risk": _empirical_risk(run),
                "n_grad_calls": grad_calls,
                "processed_samples": processed,
                "gradient_noise_proxy": gradient_noise_proxy,
                "loss_reduction_per_1000_grad_calls": (first_loss - final_loss) / grad_calls * 1000.0,
                "cost_adjusted_loss": final_loss * grad_calls / 1000.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        csv = output_dir / f"{basename}_efficiency.csv"
        summary.sort_values("batch").to_csv(csv, index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
        sns.barplot(data=summary, x="batch_label", y="final_loss", hue="batch_label", palette=_palette_for(summary, "batch_label"), legend=False, ax=axes[0])
        axes[0].set_title("Final full objective L(w) after 80 epochs, lower is better")
        axes[0].set_xlabel("batch")
        axes[0].set_ylabel(METRIC_LABELS["loss"])
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].grid(True, axis="y", alpha=0.25)
        sns.barplot(data=summary, x="batch_label", y="loss_reduction_per_1000_grad_calls", hue="batch_label", palette=_palette_for(summary, "batch_label"), legend=False, ax=axes[1])
        axes[1].set_title("Total loss reduction per 1000 gradient calls, higher is better")
        axes[1].set_xlabel("batch")
        axes[1].set_ylabel(r"$\Delta L$ / 1000 grad calls")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_efficiency.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths["efficiency_csv"] = csv
        paths["efficiency"] = png

        # Fixed-gradient-budget comparison: which batch reaches the lowest loss for the same compute budget.
        budget_rows: list[dict[str, object]] = []
        targets = [0.70, 0.65, 0.62]
        budgets = [500, 1000, 2000]
        for run in selected_runs:
            if run.result is None or not run.result.history:
                continue
            batch = int(run.params.get("batch_size", run.result.metadata.get("batch_size", 0)))
            history = list(run.result.history)
            for budget in budgets:
                best_loss = None
                best_grads = None
                for state in history:
                    grads = state.extra_metrics.get("gradient_evaluations")
                    if grads is None:
                        continue
                    if grads <= budget and (best_grads is None or grads > best_grads):
                        best_loss = state.f
                        best_grads = grads
                if best_loss is not None:
                    budget_rows.append(
                        {
                            "batch": batch,
                            "batch_label": _batch_label(batch),
                            "budget": budget,
                            "loss_at_budget": float(best_loss),
                            "grad_evaluations_used": int(best_grads),
                        }
                    )
            for target in targets:
                reached_grads = None
                reached_epoch = None
                for state in history:
                    if state.f <= target:
                        reached_grads = state.extra_metrics.get("gradient_evaluations")
                        reached_epoch = state.extra_metrics.get("epoch", state.iteration)
                        break
                budget_rows.append(
                    {
                        "batch": batch,
                        "batch_label": _batch_label(batch),
                        "budget": f"target_{target}",
                        "loss_at_budget": float(target),
                        "grad_evaluations_used": reached_grads if reached_grads is not None else int(run.result.n_grad_calls),
                        "epochs_to_target": reached_epoch if reached_epoch is not None else None,
                        "reached": reached_grads is not None,
                    }
                )
        budget_df = pd.DataFrame(budget_rows)
        if not budget_df.empty:
            budget_csv = output_dir / f"{basename}_fixed_budget.csv"
            budget_df.sort_values(["budget", "batch"]).to_csv(budget_csv, index=False)
            paths["fixed_budget_csv"] = budget_csv
            numeric_budgets = budget_df[budget_df["budget"].isin(budgets)]
            if not numeric_budgets.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=numeric_budgets, x="batch_label", y="loss_at_budget", hue="budget", palette="viridis", ax=ax)
                ax.set_title(r"Loss after a fixed gradient-evaluation budget")
                ax.set_xlabel("batch size")
                ax.set_ylabel(METRIC_LABELS["loss"])
                ax.tick_params(axis="x", rotation=25)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(title="gradient budget", fontsize=8)
                plt.tight_layout()
                budget_png = output_dir / f"{basename}_fixed_budget.png"
                fig.savefig(budget_png, dpi=300)
                plt.close(fig)
                paths["fixed_budget"] = budget_png

            target_rows = budget_df[budget_df["budget"].astype(str).str.startswith("target_")]
            if not target_rows.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                target_rows = target_rows.copy()
                target_rows["target"] = target_rows["budget"].astype(str).str.replace("target_", "", regex=False).astype(float)
                sns.barplot(data=target_rows, x="batch_label", y="grad_evaluations_used", hue="target", palette="viridis", ax=ax)
                ax.set_title(r"Gradient evaluations needed to reach a target loss")
                ax.set_xlabel("batch size")
                ax.set_ylabel("gradient evaluations")
                ax.tick_params(axis="x", rotation=25)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(title="target loss", fontsize=8)
                plt.tight_layout()
                target_png = output_dir / f"{basename}_target_speed.png"
                fig.savefig(target_png, dpi=300)
                plt.close(fig)
                paths["target_speed"] = target_png

        tradeoff = summary.sort_values("batch")
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        ax.plot(
            tradeoff["batch"],
            tradeoff["gradient_noise_proxy"],
            color="#1f77b4",
            linewidth=2.2,
            marker="o",
            markersize=6.5,
        )
        ax.fill_between(
            tradeoff["batch"],
            tradeoff["gradient_noise_proxy"],
            np.zeros(len(tradeoff), dtype=np.float64),
            color="#1f77b4",
            alpha=0.12,
        )
        for _, row in tradeoff.iterrows():
            ax.annotate(
                f"b={row['batch_label']}\nL={float(row['final_loss']):.3g}",
                (float(row["batch"]), float(row["gradient_noise_proxy"])),
                textcoords="offset points",
                xytext=(6, 7 if float(row["gradient_noise_proxy"]) < 0.8 else -26),
                fontsize=8,
            )
        max_batch = float(tradeoff["batch"].max())
        max_noise = float(tradeoff["gradient_noise_proxy"].max())
        ax.annotate(
            "SGD\ncheap update,\nhigh variance",
            xy=(1.0, max_noise),
            xytext=(2.1, 0.78),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
        )
        ax.annotate(
            "mini-batch\ntrade-off zone",
            xy=(16.0, float(tradeoff.loc[tradeoff["batch"] == 16, "gradient_noise_proxy"].iloc[0])),
            xytext=(8.5, 0.38),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
        )
        ax.annotate(
            "full batch\nexpensive update,\nzero sampling variance",
            xy=(max_batch, 0.0),
            xytext=(max_batch * 0.28, 0.18),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
        )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.04, 1.08)
        ax.set_title("Batch size trade-off: update cost vs gradient variance")
        ax.set_xlabel("batch size b (computational cost per update)")
        ax.set_ylabel(r"gradient-noise proxy $\frac{m-b}{b(m-1)}$")
        ax.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        tradeoff_png = output_dir / f"{basename}_tradeoff.png"
        fig.savefig(tradeoff_png, dpi=300)
        plt.close(fig)
        paths["tradeoff"] = tradeoff_png
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
        data["regularization_panel"] = data["regularization"].map(lambda value: value.split("=")[0] if isinstance(value, str) and value != "none" else value)
        sns.lineplot(
            data=data,
            x="iteration",
            y=metric,
            hue="regularization",
            palette=_palette_for(data, "regularization"),
            marker=None,
            errorbar=None,
            ax=ax,
        )
        final_points = data.sort_values("iteration").groupby("regularization", as_index=False).tail(1)
        sns.scatterplot(data=final_points, x="iteration", y=metric, hue="regularization", palette=_palette_for(data, "regularization"), legend=False, s=30, ax=ax)
        set_log_scale_if_positive(ax, data, metric)
        label = METRIC_LABELS.get(metric, metric)
        ax.set_title(f"Regularization comparison: {label}")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        plt.tight_layout()
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
        if metric in {"l1_term", "l2_term"}:
            continue
        families = ["none", "L1", "L2", "Elastic Net"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), squeeze=False, sharex=False, sharey=True)
        for axis, family in zip(axes.ravel(), families, strict=True):
            if family == "none":
                subset = data[data["regularization"] == "none"]
            else:
                subset = data[data["regularization"].astype(str).str.startswith(family)]
            if subset.empty:
                axis.axis("off")
                continue
            sns.lineplot(data=subset, x="iteration", y=metric, hue="regularization", palette=_palette_for(subset, "regularization"), marker=None, errorbar=None, ax=axis)
            family_final = subset.sort_values("iteration").groupby("regularization", as_index=False).tail(1)
            sns.scatterplot(data=family_final, x="iteration", y=metric, hue="regularization", palette=_palette_for(subset, "regularization"), legend=False, s=28, ax=axis)
            set_log_scale_if_positive(axis, subset, metric)
            axis.set_title(family)
            axis.set_xlabel("epoch / iteration")
            axis.set_ylabel(label)
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize=7)
        plt.tight_layout()
        panel_png = output_dir / f"{basename}_{safe_stem(metric)}_panels.png"
        fig.savefig(panel_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths[f"{metric}_panels"] = panel_png
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
        label = _run_label(run, include_stats=True)
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
        if metric == "elapsed_seconds":
            positive = data[metric][data[metric] > 0]
            if not positive.empty:
                ax.set_yscale("log")
                ax.set_ylabel("elapsed seconds, log scale")
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        png = output_dir / f"{basename}_{safe_stem(metric)}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[metric] = png
    return paths


def _dataset_label(run: ExperimentRun) -> str:
    dataset = run.function_params.get("dataset_kind", "--")
    n_points = run.function_params.get("n_points", "--")
    noise = run.function_params.get("noise_variance", "--")
    return f"{dataset}, m={n_points}, noise={noise}"


def save_lab5_report_tables(runs: list[ExperimentRun], output_dir: Path, plot_dir: Path | None = None, block_size: int = 34) -> dict[str, Path]:
    """Save compact LaTeX tables used by the Lab 5 report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary_report_scaled.tex"
    paths: dict[str, Path] = {}
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
    paths["summary_report_scaled"] = path
    paths["fit_catalog"] = _save_fit_catalog(runs, output_dir / "fit_catalog.tex", plot_dir)
    paths["gn_lm_comparison"] = _save_gn_lm_comparison(runs, output_dir / "gn_lm_comparison.tex")
    paths["batch_size_summary"] = _save_batch_size_summary(runs, output_dir / "batch_size_summary.tex")
    return paths


def _fit_plot_name(run: ExperimentRun) -> str:
    compact = json.dumps(run.function_params, sort_keys=True, separators=(",", ":"), default=str)
    return f"{safe_stem(f'PolynomialRegressionObjective_{compact}')}_fit.png"


def _save_fit_catalog(runs: list[ExperimentRun], path: Path, plot_dir: Path | None) -> Path:
    selected = [
        run
        for run in runs
        if run.result is not None
        and float(run.function_params.get("lambda_l1", 0.0) or 0.0) == 0.0
        and float(run.function_params.get("lambda_l2", 0.0) or 0.0) == 0.0
        and "not_applicable" not in run.result.message.lower()
    ]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Каталог fit-графиков без регуляризации}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lllrll}",
        r"\toprule",
        r"dataset & degree & method & final $L$ & status & plot\\",
        r"\midrule",
    ]
    for run in sorted(selected, key=lambda item: (str(item.function_params.get("dataset_kind")), int(item.function_params.get("degree", 0)), item.optimizer_name)):
        result = run.result
        assert result is not None
        plot_name = _fit_plot_name(run)
        plot_path = f"../outputs/lab5/plots/{plot_name}" if plot_dir is not None else plot_name
        lines.append(
            " & ".join(
                [
                    _latex_escape(run.function_params.get("dataset_kind")),
                    str(run.function_params.get("degree")),
                    _latex_escape(_optimizer_label(run.optimizer_name)),
                    _format_float(result.f),
                    _latex_escape(_short_stop(run)),
                    _latex_escape(plot_path),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", r"\endgroup"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_gn_lm_comparison(runs: list[ExperimentRun], path: Path) -> Path:
    selected = [
        run
        for run in runs
        if run.optimizer_name in {"GaussNewton", "LevenbergMarquardt"}
        and run.result is not None
        and run.function_params.get("dataset_kind") == "nonlinear"
        and int(run.function_params.get("degree", 0) or 0) == 5
        and (
            (float(run.function_params.get("lambda_l1", 0.0) or 0.0) == 0.0 and float(run.function_params.get("lambda_l2", 0.0) or 0.0) == 0.0)
            or (float(run.function_params.get("lambda_l1", 0.0) or 0.0) == 0.01 and float(run.function_params.get("lambda_l2", 0.0) or 0.0) == 0.01)
        )
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Сравнение Gauss--Newton и Levenberg--Marquardt}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lllrrrrrl}",
        r"\toprule",
        r"dataset & Reg & Method & $L$ & $Q$ & iter & $N_f$ & $N_g$ & status\\",
        r"\midrule",
    ]
    for run in sorted(selected, key=lambda item: (_regularization_label(item.function_params, include_none=True), item.optimizer_name)):
        result = run.result
        assert result is not None
        lines.append(
            " & ".join(
                [
                    _latex_escape(run.function_params.get("dataset_kind")),
                    _latex_escape(_regularization_label(run.function_params, include_none=True)),
                    _latex_escape(_optimizer_label(run.optimizer_name)),
                    _format_float(result.f),
                    _format_float(_empirical_risk(run)),
                    str(result.n_iter),
                    str(result.n_calls),
                    str(result.n_grad_calls),
                    _latex_escape(_short_stop(run)),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_batch_size_summary(runs: list[ExperimentRun], path: Path) -> Path:
    selected = [
        run
        for run in runs
        if run.optimizer_name == "MiniBatchGradientDescent"
        and "batch_size" in run.params
        and run.result is not None
        and run.function_params.get("dataset_kind") == "nonlinear"
        and int(run.function_params.get("degree", 0) or 0) == 5
        and float(run.function_params.get("lambda_l1", 0.0) or 0.0) == 0.0
        and float(run.function_params.get("lambda_l2", 0.0) or 0.0) == 0.0
        and "not_applicable" not in run.result.message.lower()
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Сравнение batch size для nonlinear degree 5}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"batch & $L$ & $Q$ & epochs & $N_g$ & updates & $\Delta L/1000N_g$ & status\\",
        r"\midrule",
    ]
    for run in sorted(selected, key=lambda item: int(item.params.get("batch_size", item.result.metadata.get("batch_size", 0) if item.result else 0))):
        result = run.result
        assert result is not None
        batch = int(run.params.get("batch_size", result.metadata.get("batch_size", 0)))
        first_loss = float(result.history[0].extra_metrics.get("loss", result.history[0].f)) if result.history else result.f
        gain_per_1000 = (first_loss - result.f) / max(1, result.n_grad_calls) * 1000.0
        lines.append(
            " & ".join(
                [
                    _latex_escape(_batch_label(batch)),
                    _format_float(result.f),
                    _format_float(_empirical_risk(run)),
                    str(result.n_iter),
                    str(result.n_grad_calls),
                    str(result.metadata.get("updates", "--")),
                    _format_float(gain_per_1000),
                    _latex_escape(_short_stop(run)),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
