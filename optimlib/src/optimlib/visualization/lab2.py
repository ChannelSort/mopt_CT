"""Lab 2 specific plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.visualization.base import safe_stem
from optimlib.visualization.contour import plot_trajectory_contours


_ACKLEY_CONSTANT_ALPHAS = (0.1, 0.01, 0.001)
_COMPLEX_FUNCTIONS = ("Ackley", "Rosenbrock", "Himmelblau")
_OPTIMIZER_LABELS = {
    "ConstantStepGD": "ConstantStepGD",
    "ArmijoBacktracking": "Armijo",
    "StrongWolfe": "Strong Wolfe",
    "SteepestDescent": "Steepest Descent",
}


def _ackley_run_label(run: ExperimentRun) -> str:
    if run.optimizer_name == "ConstantStepGD":
        return f"Const alpha={float(run.params['alpha']):g}"
    return {
        "ArmijoBacktracking": "Armijo",
        "StrongWolfe": "Strong Wolfe",
        "SteepestDescent": "Steepest descent",
    }.get(run.optimizer_name, run.optimizer_name)


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
        return "--"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6g}"


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

    return plot_trajectory_contours(
        func,
        sorted(selected, key=sort_key),
        output_dir,
        basename,
        title="Ackley: trajectories",
        run_label=_ackley_run_label,
        grid_size=360,
        filled_levels=70,
        legend_fontsize=7.0,
        legend_loc="upper right",
        legend_ncol=2,
        start_text=r"$x_0=(-2,-2)$",
    )


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


def save_lab2_complex_summary_table(
    results: list[ExperimentRun],
    table_dir: Path,
    tolerance: float = 1e-8,
) -> Path:
    """Save compact Lab 2 summary table for non-quadratic functions."""
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "complex_summary.tex"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Результаты на сложных функциях при $\varepsilon=10^{-8}$}",
        r"\fitreporttable{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Функция & Метод & Успешных запусков & Мин. итераций & Мин. вызовов $f$ & Лучшее $f(x_k)$\\",
        r"\midrule",
    ]
    for function_name in _COMPLEX_FUNCTIONS:
        for optimizer_name in _OPTIMIZER_LABELS:
            subset = [
                run
                for run in results
                if run.function_name == function_name
                and run.optimizer_name == optimizer_name
                and run.result is not None
                and abs(run.tolerance - tolerance) <= 1e-12 * max(1.0, abs(tolerance))
            ]
            converged_count = sum(1 for run in subset if run.result is not None and run.result.converged)
            converged = [run for run in subset if run.result is not None and run.result.converged]
            iterations = min((run.result.n_iter for run in converged if run.result is not None), default=None)
            f_calls = min((run.result.n_calls for run in converged if run.result is not None), default=None)
            best_f = min((run.result.f for run in subset if run.result is not None and np.isfinite(run.result.f)), default=None)
            lines.append(
                " & ".join(
                    [
                        _latex_escape(function_name),
                        _latex_escape(_OPTIMIZER_LABELS[optimizer_name]),
                        str(converged_count),
                        "--" if iterations is None else str(iterations),
                        "--" if f_calls is None else str(f_calls),
                        _format_float(best_f),
                    ]
                )
                + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
