"""Lab 4 specific metric plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimlib.experiment.runner import ExperimentRun
from optimlib.visualization.base import numeric_result_metric, safe_stem


def _lab4_metric_rows(results: list[ExperimentRun], metrics: Sequence[str] | None = None) -> list[dict[str, object]]:
    selected_metrics = tuple(metrics or ("n_iter", "n_calls", "n_grad_calls", "n_hessian_calls"))
    rows: list[dict[str, object]] = []
    for run in results:
        n = run.function_params.get("n")
        k = run.function_params.get("k")
        if n is None or k is None:
            continue
        for metric in selected_metrics:
            value = numeric_result_metric(run, metric)
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
            stem = f"lab4_table_{safe_stem(str(metric))}_{safe_stem(str(optimizer))}"
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
            png_path = output_dir / f"lab4_{safe_stem(str(metric))}_vs_n_k_{chosen_k:g}.png"
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
            png_path = output_dir / f"lab4_{safe_stem(str(metric))}_vs_k_n_{chosen_n}.png"
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
        png_path = output_dir / f"lab4_comparison_{safe_stem(str(metric))}.png"
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
        value = numeric_result_metric(run, metric)
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
    png_path = output_dir / f"lab4_lbfgs_memory_{safe_stem(metric)}.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {"png": png_path}
