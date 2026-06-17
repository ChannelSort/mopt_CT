"""Lab 4 specific metric plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.visualization.base import numeric_result_metric, safe_stem
from optimlib.visualization.contour import plot_trajectory_contour_panels


LAB4_TRAJECTORY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CG", ("QuadraticConjugateGradient", "FletcherReeves", "PolakRibiere")),
    ("Newton / trust-region", ("NewtonCholesky", "NewtonDirectionChoice", "PowellDogLeg", "ScipyNewtonCG")),
    ("Quasi-Newton", ("DFP", "BFGS", "LBFGS")),
)


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


def _is_finite_result(run: ExperimentRun) -> bool:
    return run.result is not None and np.isfinite(run.result.f)


def _representative_run(runs: list[ExperimentRun], optimizer: str) -> ExperimentRun | None:
    candidates = [run for run in runs if run.optimizer_name == optimizer and _is_finite_result(run)]
    if not candidates:
        return None
    converged = [run for run in candidates if run.result is not None and run.result.converged]
    pool = converged if converged else candidates
    return min(pool, key=lambda run: run.result.n_iter if run.result is not None else 10**18)


def _compact_params(params: dict[str, Any]) -> str:
    if not params:
        return "{}"
    parts = []
    for key in sorted(params):
        value = params[key]
        parts.append(f"{key}={float(value):g}" if isinstance(value, int | float) else f"{key}={value}")
    return ", ".join(parts)


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


def _format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6g}"


def _short_stop(run: ExperimentRun) -> str:
    if run.result is None:
        return "error"
    message = run.result.message.lower()
    if run.result.converged and "gradient norm" in message:
        return "grad_tol"
    if "maximum iterations" in message:
        return "max_iter"
    if "step tolerance" in message:
        return "step_tol"
    if "non_positive_definite_hessian" in message:
        return "not_spd_hessian"
    if "non_descent_newton_direction" in message:
        return "non_descent"
    if run.result.converged:
        return "converged"
    return "stopped"


def _function_label(run: ExperimentRun) -> str:
    if run.function_name == "GeneratedQuadratic":
        n = run.function_params.get("n", "--")
        k = run.function_params.get("k", "--")
        seed = run.function_params.get("seed", "--")
        return f"Q(n={n}, k={k}, seed={seed})"
    x0 = run.function_params.get("x0")
    if x0 is not None:
        return f"{run.function_name}, x0={x0}"
    return run.function_name


def _direction_counts(run: ExperimentRun) -> tuple[int, int, int, float]:
    if run.result is None:
        return 0, 0, 0, 0.0
    newton = 0
    modified = 0
    steepest = 0
    max_shift = 0.0
    for state in run.result.history:
        direction = state.extra_metrics.get("direction")
        if direction == "newton":
            newton += 1
        elif direction == "modified_newton":
            modified += 1
        elif direction == "steepest_descent":
            steepest += 1
        shift = state.extra_metrics.get("hessian_shift")
        if isinstance(shift, int | float):
            max_shift = max(max_shift, float(shift))
    return newton, modified, steepest, max_shift


def plot_lab4_grouped_trajectories(
    func: MultivariateFunction,
    family: list[ExperimentRun],
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    """Plot Lab 4 trajectories as method-family panels."""
    groups: list[tuple[str, list[ExperimentRun]]] = []
    for title, optimizers in LAB4_TRAJECTORY_GROUPS:
        selected = [_representative_run(family, optimizer) for optimizer in optimizers]
        runs = [run for run in selected if run is not None]
        if runs:
            groups.append((title, runs))
    return plot_trajectory_contour_panels(
        func,
        groups,
        output_dir,
        f"{stem}_trajectory_panels",
        title=f"{func.name}: trajectories",
        run_label=lambda run: run.optimizer_name if run.optimizer_name != "LBFGS" else f"LBFGS m={run.params.get('m')}",
        show_arrows=True,
    )


def save_lab4_report_tables(runs: list[ExperimentRun], output_dir: Path) -> dict[str, Path]:
    """Save compact LaTeX tables used by the Lab 4 report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["summary_report_scaled"] = _save_lab4_summary_report(runs, output_dir / "summary_report_scaled.tex")
    paths["newton_direction_diagnostics"] = _save_newton_direction_diagnostics(runs, output_dir / "newton_direction_diagnostics.tex")
    return paths


def _save_lab4_summary_report(runs: list[ExperimentRun], path: Path, block_size: int = 38) -> Path:
    sorted_runs = sorted(
        runs,
        key=lambda run: (
            run.function_name,
            str(run.function_params),
            run.optimizer_name,
            str(run.params),
        ),
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
    ]
    for block_index, start in enumerate(range(0, len(sorted_runs), block_size), start=1):
        block = sorted_runs[start : start + block_size]
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{Полная таблица экспериментов, блок {block_index}}}",
                r"\fitreporttable{%",
                r"\begin{tabular}{llllrrrrrrl}",
                r"\toprule",
                r"Function & Optimizer & params & conv & $f(x_k)$ & $\|g_k\|$ & iter & $N_f$ & $N_g$ & $N_H$ & Stop\\",
                r"\midrule",
            ]
        )
        for run in block:
            result = run.result
            row = [
                _function_label(run),
                run.optimizer_name,
                _compact_params(run.params),
                "--" if result is None else ("yes" if result.converged else "no"),
                "--" if result is None else _format_float(result.f),
                _format_float(run.grad_norm),
                "--" if result is None else str(result.n_iter),
                "--" if result is None else str(result.n_calls),
                "--" if result is None else str(result.n_grad_calls),
                "--" if result is None else str(result.n_hessian_calls),
                _short_stop(run),
            ]
            lines.append(" & ".join(_latex_escape(item) for item in row) + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_newton_direction_diagnostics(runs: list[ExperimentRun], path: Path) -> Path:
    grouped: dict[str, dict[str, float | int]] = {}
    for run in runs:
        if run.optimizer_name != "NewtonDirectionChoice" or run.result is None:
            continue
        key = run.function_name
        item = grouped.setdefault(key, {"runs": 0, "newton": 0, "modified": 0, "steepest": 0, "max_shift": 0.0})
        newton, modified, steepest, max_shift = _direction_counts(run)
        item["runs"] = int(item["runs"]) + 1
        item["newton"] = int(item["newton"]) + newton
        item["modified"] = int(item["modified"]) + modified
        item["steepest"] = int(item["steepest"]) + steepest
        item["max_shift"] = max(float(item["max_shift"]), max_shift)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Диагностика выбора направления в NewtonDirectionChoice}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Функция & Запусков & newton & modified\_newton & steepest\_descent & max $\lambda$\\",
        r"\midrule",
    ]
    for name in sorted(grouped):
        item = grouped[name]
        row = [
            name,
            str(item["runs"]),
            str(item["newton"]),
            str(item["modified"]),
            str(item["steepest"]),
            _format_float(float(item["max_shift"])),
        ]
        lines.append(" & ".join(_latex_escape(value) for value in row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
