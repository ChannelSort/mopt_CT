"""Lab 4 specific metric plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.visualization.base import numeric_result_metric, safe_stem
from optimlib.visualization.contour import plot_trajectory_contour_panels, plot_trajectory_contours


LAB4_TRAJECTORY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CG", ("QuadraticConjugateGradient", "FletcherReeves", "PolakRibiere")),
    ("Newton / trust-region", ("NewtonCholesky", "NewtonDirectionChoice", "PowellDogLeg")),
    ("Quasi-Newton", ("DFP", "BFGS", "LBFGS")),
)


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


def _status(run: ExperimentRun) -> str:
    return _short_stop(run)


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
    if "precision loss" in message:
        return "precision_loss"
    if "non_positive_curvature" in message:
        return "non_pos_curv"
    if "non_positive_predicted_reduction" in message:
        return "non_pos_pred"
    if "optimization terminated successfully" in message:
        return "scipy_stop"
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


def _start_label(run: ExperimentRun) -> str:
    x0 = run.function_params.get("x0")
    if isinstance(x0, list | tuple) and len(x0) == 2:
        return f"({float(x0[0]):g}, {float(x0[1]):g})"
    return "--" if x0 is None else str(x0)


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


def plot_lab4_trajectory_overview(
    func: MultivariateFunction,
    family: list[ExperimentRun],
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    """Plot one combined trajectory overview with representative runs."""
    runs: list[ExperimentRun] = []
    for _, optimizers in LAB4_TRAJECTORY_GROUPS:
        for optimizer in optimizers:
            run = _representative_run(family, optimizer)
            if run is not None:
                runs.append(run)
    if not runs:
        return {}
    return plot_trajectory_contours(
        func,
        runs,
        output_dir,
        f"{stem}_trajectories",
        title=f"{func.name}: trajectories",
        run_label=lambda run: run.optimizer_name if run.optimizer_name != "LBFGS" else f"LBFGS {{'m': {run.params.get('m')}}}",
        legend_fontsize=6.0,
        show_arrows=True,
    )


def save_lab4_report_tables(runs: list[ExperimentRun], output_dir: Path) -> dict[str, Path]:
    """Save compact LaTeX tables used by the Lab 4 report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["summary_report_scaled"] = _save_lab4_summary_report(runs, output_dir / "summary_report_scaled.tex")
    paths["newton_direction_diagnostics"] = _save_newton_direction_diagnostics(runs, output_dir / "newton_direction_diagnostics.tex")
    paths["dependency_n_k10_split"] = _save_dependency_table(runs, output_dir / "dependency_n_k10_split.tex", fixed_key="k", fixed_value=10, x_key="n")
    paths["dependency_k_n10_split"] = _save_dependency_table(runs, output_dir / "dependency_k_n10_split.tex", fixed_key="n", fixed_value=10, x_key="k")
    paths["cost_generated_quadratic"] = _save_generated_quadratic_cost_table(runs, output_dir / "cost_generated_quadratic.tex")
    paths["lbfgs_memory_fixed"] = _save_lbfgs_memory_table(runs, output_dir / "lbfgs_memory_fixed.tex")
    paths["quadratic2d_starts"] = _save_quadratic2d_start_table(runs, output_dir / "quadratic2d_starts.tex")
    paths["complex_function_points"] = _save_complex_function_points_table(runs, output_dir / "complex_function_points.tex")
    paths["problem_cases"] = _save_problem_cases_table(runs, output_dir / "problem_cases.tex")
    paths.update(_save_report_dependency_plots(runs, output_dir / "plots"))
    paths["cg_linear_reference_plot"] = _save_cg_linear_reference_plot(runs, output_dir / "plots" / "lab4_cg_linear_reference.png")
    return paths


_REPORT_OPTIMIZER_ORDER = {
    "BFGS": 0,
    "DFP": 1,
    "FletcherReeves": 2,
    "LBFGS m=3": 3,
    "LBFGS m=5": 4,
    "LBFGS m=10": 5,
    "NewtonCholesky": 6,
    "NewtonDirectionChoice": 7,
    "PolakRibiere": 8,
    "PowellDogLeg": 9,
    "QuadraticConjugateGradient": 10,
    "ScipyNewtonCG": 11,
}

_REPORT_PLOT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CG methods", ("QuadraticConjugateGradient", "FletcherReeves", "PolakRibiere")),
    ("Newton / trust-region", ("NewtonCholesky", "NewtonDirectionChoice", "PowellDogLeg", "ScipyNewtonCG")),
    ("Quasi-Newton", ("DFP", "BFGS", "LBFGS")),
)

_REPORT_PLOT_COLORS = {
    "QuadraticConjugateGradient": "#1f77b4",
    "FletcherReeves": "#ff7f0e",
    "PolakRibiere": "#2ca02c",
    "NewtonCholesky": "#d62728",
    "NewtonDirectionChoice": "#9467bd",
    "PowellDogLeg": "#8c564b",
    "ScipyNewtonCG": "#17becf",
    "DFP": "#e377c2",
    "BFGS": "#7f7f7f",
    "LBFGS": "#bcbd22",
}

_REPORT_PLOT_MARKERS = {
    "QuadraticConjugateGradient": "o",
    "FletcherReeves": "s",
    "PolakRibiere": "^",
    "NewtonCholesky": "D",
    "NewtonDirectionChoice": "P",
    "PowellDogLeg": "X",
    "ScipyNewtonCG": "o",
    "DFP": "v",
    "BFGS": "<",
    "LBFGS": ">",
}

_REPORT_PLOT_LINESTYLES = {
    "QuadraticConjugateGradient": "-",
    "FletcherReeves": "--",
    "PolakRibiere": "-.",
    "NewtonCholesky": ":",
    "NewtonDirectionChoice": "--",
    "PowellDogLeg": "-",
    "ScipyNewtonCG": "-.",
    "DFP": "--",
    "BFGS": "-.",
    "LBFGS": "-",
}

_REPORT_METRIC_LABELS = {
    "n_iter": "iterations",
    "n_calls": "function calls",
    "n_grad_calls": "gradient calls",
    "n_hessian_calls": "Hessian calls",
}


def _report_optimizer_label(run: ExperimentRun, *, include_lbfgs_memory: bool = True) -> str:
    if run.optimizer_name == "LBFGS" and include_lbfgs_memory:
        return f"LBFGS m={int(run.params.get('m', 0))}"
    return run.optimizer_name


def _generated_quadratic_runs(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    return [run for run in runs if run.function_name == "GeneratedQuadratic" and run.result is not None]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _save_dependency_table(
    runs: list[ExperimentRun],
    path: Path,
    *,
    fixed_key: str,
    fixed_value: int,
    x_key: str,
    block_size: int = 18,
) -> Path:
    buckets: dict[tuple[str, int], list[ExperimentRun]] = defaultdict(list)
    for run in _generated_quadratic_runs(runs):
        if int(float(run.function_params.get(fixed_key, -1))) != fixed_value:
            continue
        x_value = int(float(run.function_params.get(x_key, -1)))
        buckets[(_report_optimizer_label(run), x_value)].append(run)

    table_rows: list[tuple[str, int, str]] = []
    for (optimizer, x_value), group in buckets.items():
        results = [run.result for run in group if run.result is not None]
        if not results:
            continue
        row = [
            _latex_escape(optimizer),
            str(x_value),
            _format_float(_mean([float(result.n_iter) for result in results])),
            _format_float(_mean([float(result.n_calls) for result in results])),
            _format_float(_mean([float(result.n_grad_calls) for result in results])),
            _format_float(_mean([float(result.n_hessian_calls) for result in results])),
            _format_float(_mean([1.0 if result.converged else 0.0 for result in results])),
        ]
        table_rows.append((optimizer, x_value, " & ".join(row) + r"\\"))

    table_rows.sort(key=lambda item: (_REPORT_OPTIMIZER_ORDER.get(item[0], 99), item[1]))
    fixed_latex = "$k=10$" if fixed_key == "k" else "$n=10$"
    varying_latex = "$n$" if x_key == "n" else "$k$"
    lines = [r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{2.0pt}"]
    for block_index, start in enumerate(range(0, len(table_rows), block_size), start=1):
        block = table_rows[start : start + block_size]
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{Зависимость метрик от {varying_latex} при фиксированном {fixed_latex}, блок {block_index}}}",
                r"\fitreporttable{%",
                r"\begin{tabular}{llrrrrr}",
                r"\toprule",
                rf"Optimizer & {varying_latex} & $n_{{iter}}$ & $N_f$ & $N_g$ & $N_H$ & success\\",
                r"\midrule",
                *[row for _, _, row in block],
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\end{table}",
                "",
            ]
        )
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_generated_quadratic_cost_table(runs: list[ExperimentRun], path: Path) -> Path:
    buckets: dict[str, list[ExperimentRun]] = defaultdict(list)
    for run in _generated_quadratic_runs(runs):
        buckets[_report_optimizer_label(run)].append(run)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Mean conditional cost $C=N_f+5N_g+20N_H$ on GeneratedQuadratic}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Optimizer & mean $C$ & mean $n_{iter}$ & mean $N_f$ & mean $N_g$ & mean $N_H$\\",
        r"\midrule",
    ]
    for optimizer in sorted(buckets, key=lambda item: _REPORT_OPTIMIZER_ORDER.get(item, 99)):
        results = [run.result for run in buckets[optimizer] if run.result is not None]
        costs = [float(result.n_calls + 5 * result.n_grad_calls + 20 * result.n_hessian_calls) for result in results]
        row = [
            _latex_escape(optimizer),
            _format_float(_mean(costs)),
            _format_float(_mean([float(result.n_iter) for result in results])),
            _format_float(_mean([float(result.n_calls) for result in results])),
            _format_float(_mean([float(result.n_grad_calls) for result in results])),
            _format_float(_mean([float(result.n_hessian_calls) for result in results])),
        ]
        lines.append(" & ".join(row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _format_vector(value: object) -> str:
    if value is None:
        return "--"
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return "(" + ", ".join(_format_float(float(item)) for item in array) + ")"


def _save_complex_function_points_table(runs: list[ExperimentRun], path: Path, block_size: int = 27) -> Path:
    selected = [
        run
        for run in runs
        if run.function_name in {"Lab4Rosenbrock", "Lab4Himmelblau", "Lab4Ackley"} and run.result is not None
    ]
    selected.sort(key=lambda item: (item.function_name, _start_label(item), item.optimizer_name, str(item.params)))
    table_rows: list[str] = []
    for run in selected:
        result = run.result
        assert result is not None
        optimizer = run.optimizer_name if run.optimizer_name != "LBFGS" else f"LBFGS m={run.params.get('m')}"
        row = [
            _latex_escape(run.function_name),
            _latex_escape(_start_label(run)),
            _latex_escape(optimizer),
            _latex_escape(_format_vector(result.x)),
            _format_float(result.f),
            _format_float(run.grad_norm),
            str(result.n_iter),
            str(result.n_calls),
            str(result.n_grad_calls),
            str(result.n_hessian_calls),
            _latex_escape(_status(run)),
        ]
        table_rows.append(" & ".join(row) + r"\\")

    lines = [r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{2.0pt}"]
    for block_index, start in enumerate(range(0, len(table_rows), block_size), start=1):
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{Найденные точки на сложных функциях, блок {block_index}}}",
                r"\fitreporttable{%",
                r"\begin{tabular}{llllrrrrrrl}",
                r"\toprule",
                r"Function & $x_0$ & Optimizer & $x_k$ & $f(x_k)$ & $\|g_k\|$ & iter & $N_f$ & $N_g$ & $N_H$ & Stop\\",
                r"\midrule",
                *table_rows[start : start + block_size],
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\end{table}",
                "",
            ]
        )
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _plot_dependency_data(
    runs: list[ExperimentRun],
    metric: str,
    *,
    fixed_key: str,
    fixed_value: int,
    x_key: str,
) -> dict[str, dict[int, float]]:
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run in _generated_quadratic_runs(runs):
        if int(float(run.function_params.get(fixed_key, -1))) != fixed_value:
            continue
        result = run.result
        if result is None:
            continue
        x_value = int(float(run.function_params.get(x_key, -1)))
        optimizer = run.optimizer_name
        value = getattr(result, metric)
        buckets[(optimizer, x_value)].append(float(value))
    data: dict[str, dict[int, float]] = defaultdict(dict)
    for (optimizer, x_value), values in buckets.items():
        data[optimizer][x_value] = _mean(values)
    return data


def _set_dependency_x_axis(ax: plt.Axes, x_values: list[int], log_x: bool) -> None:
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(x_values))
        ax.xaxis.set_major_formatter(FixedFormatter([str(value) for value in x_values]))
        ax.set_xlim(min(x_values) / 1.4, max(x_values) * 1.4)
    else:
        ax.set_xticks(x_values)


def _dependency_offsets(rows: list[tuple[str, list[int], list[float]]]) -> dict[str, float]:
    values = [value for _, _, series in rows for value in series]
    span = max(values) - min(values) if values else 1.0
    base = max(span * 0.018, 0.045)
    seen: dict[tuple[tuple[int, ...], tuple[float, ...]], int] = {}
    offsets: dict[str, float] = {}
    for optimizer, x_values, y_values in rows:
        key = (tuple(x_values), tuple(round(value, 10) for value in y_values))
        duplicate_index = seen.get(key, 0)
        seen[key] = duplicate_index + 1
        offsets[optimizer] = duplicate_index * base
    return offsets


def _save_dependency_plot(
    runs: list[ExperimentRun],
    metric: str,
    x_values: list[int],
    *,
    fixed_key: str,
    fixed_value: int,
    x_key: str,
    x_label: str,
    title: str,
    path: Path,
    log_x: bool = False,
) -> Path:
    data = _plot_dependency_data(runs, metric, fixed_key=fixed_key, fixed_value=fixed_value, x_key=x_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharex=False)
    fig.suptitle(f"{title} (overlapping lines offset for readability)", y=1.02, fontsize=13)
    for ax, (group_title, optimizers) in zip(axes, _REPORT_PLOT_GROUPS, strict=True):
        rows_for_group: list[tuple[str, list[int], list[float]]] = []
        for optimizer in optimizers:
            series = data.get(optimizer, {})
            xs = [value for value in x_values if value in series]
            if xs:
                rows_for_group.append((optimizer, xs, [series[value] for value in xs]))
        offsets = _dependency_offsets(rows_for_group)
        for optimizer, xs, ys in rows_for_group:
            ax.plot(
                xs,
                [value + offsets[optimizer] for value in ys],
                label=optimizer,
                color=_REPORT_PLOT_COLORS[optimizer],
                marker=_REPORT_PLOT_MARKERS[optimizer],
                linestyle=_REPORT_PLOT_LINESTYLES[optimizer],
                linewidth=2.0,
                markersize=5.5,
                alpha=0.95,
            )
        ax.set_title(group_title, fontsize=10.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(_REPORT_METRIC_LABELS[metric])
        _set_dependency_x_axis(ax, x_values, log_x)
        ax.grid(True, which="both", alpha=0.25)
        if rows_for_group:
            ax.legend(fontsize=7.2, framealpha=0.9, loc="best")
        ymin, ymax = ax.get_ylim()
        if ymin >= 0:
            ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_report_dependency_plots(runs: list[ExperimentRun], plot_dir: Path) -> dict[str, Path]:
    metrics = ["n_iter", "n_calls", "n_grad_calls", "n_hessian_calls"]
    paths: dict[str, Path] = {}
    for metric in metrics:
        paths[f"{metric}_vs_n_k10"] = _save_dependency_plot(
            runs,
            metric,
            [2, 10, 50, 100],
            fixed_key="k",
            fixed_value=10,
            x_key="n",
            x_label="dimension n",
            title=f"{metric} vs n at k=10",
            path=plot_dir / f"lab4_{metric}_vs_n_k_10.png",
        )
        paths[f"{metric}_vs_k_n10"] = _save_dependency_plot(
            runs,
            metric,
            [1, 10, 100, 1000],
            fixed_key="n",
            fixed_value=10,
            x_key="k",
            x_label="cond(A)",
            title=f"{metric} vs cond(A) at n=10",
            path=plot_dir / f"lab4_{metric}_vs_k_n_10.png",
            log_x=True,
        )
    return paths


def _save_cg_linear_reference_plot(runs: list[ExperimentRun], path: Path) -> Path:
    buckets: dict[int, list[float]] = defaultdict(list)
    for run in _generated_quadratic_runs(runs):
        if run.optimizer_name != "QuadraticConjugateGradient" or int(float(run.function_params.get("k", -1))) != 10:
            continue
        result = run.result
        if result is not None:
            buckets[int(float(run.function_params.get("n", -1)))].append(float(result.n_iter))
    ns = sorted(buckets)
    iterations = [_mean(buckets[n]) for n in ns]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(ns, iterations, marker="o", linewidth=2.4, markersize=7, color="#1f77b4", label="experiment: Quadratic CG, k=10")
    ax.plot(ns, ns, linestyle="--", linewidth=2.0, color="black", label="theoretical upper bound: n iterations")
    ax.fill_between(ns, 0, ns, color="#1f77b4", alpha=0.08, label="allowed region: <= n")
    ax.set_title("Linear conjugate gradient on quadratic functions")
    ax.set_xlabel("dimension n")
    ax.set_ylabel("iterations")
    ax.set_xticks(ns)
    ax.set_ylim(0, max(ns) * 1.08 if ns else 1)
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.92, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


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


def _save_lbfgs_memory_table(runs: list[ExperimentRun], path: Path) -> Path:
    selected = [
        run
        for run in runs
        if run.optimizer_name == "LBFGS"
        and run.function_name == "GeneratedQuadratic"
        and int(run.function_params.get("n", -1)) == 50
        and float(run.function_params.get("k", -1.0)) == 100.0
        and int(run.function_params.get("seed", -1)) == 0
        and run.result is not None
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{L-BFGS при фиксированных $n=50$, $k=100$, seed=0}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"$m$ & $n_{iter}$ & $N_f$ & $N_g$ & $\|\nabla f(x_k)\|$ & $f(x_k)$ & status\\",
        r"\midrule",
    ]
    for run in sorted(selected, key=lambda item: int(item.params.get("m", 0))):
        result = run.result
        assert result is not None
        lines.append(
            " & ".join(
                [
                    str(run.params.get("m", "--")),
                    str(result.n_iter),
                    str(result.n_calls),
                    str(result.n_grad_calls),
                    _format_float(run.grad_norm),
                    _format_float(result.f),
                    _latex_escape(_status(run)),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_quadratic2d_start_table(runs: list[ExperimentRun], path: Path, block_size: int = 24) -> Path:
    selected = [run for run in runs if run.function_name == "Quadratic2DVisualization" and run.result is not None]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
    ]
    table_rows: list[str] = []
    for run in sorted(selected, key=lambda item: (_start_label(item), item.optimizer_name, str(item.params))):
        result = run.result
        assert result is not None
        optimizer = run.optimizer_name if run.optimizer_name != "LBFGS" else f"LBFGS m={run.params.get('m')}"
        table_rows.append(
            " & ".join(
                [
                    _latex_escape(_start_label(run)),
                    _latex_escape(optimizer),
                    str(result.n_iter),
                    str(result.n_calls),
                    str(result.n_grad_calls),
                    str(result.n_hessian_calls),
                    _format_float(result.f),
                    _latex_escape(_status(run)),
                ]
            )
            + r"\\"
        )
    for block_index, start in enumerate(range(0, len(table_rows), block_size), start=1):
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{Запуски из пяти стартовых точек на двумерной квадратичной функции, блок {block_index}}}",
                r"\fitreporttable{%",
                r"\begin{tabular}{llrrrrrl}",
                r"\toprule",
                r"$x_0$ & Optimizer & $n_{iter}$ & $N_f$ & $N_g$ & $N_H$ & $f(x_k)$ & status\\",
                r"\midrule",
                *table_rows[start : start + block_size],
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\end{table}",
                "",
            ]
        )
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_problem_cases_table(runs: list[ExperimentRun], path: Path, limit: int = 42) -> Path:
    candidates = [
        run
        for run in runs
        if run.result is None
        or not run.result.converged
        or run.function_name in {"Lab4Himmelblau", "Lab4Ackley", "Lab4Rosenbrock"}
    ]
    priority = {
        "not_spd_hessian": 0,
        "non_pos_curv": 1,
        "non_pos_pred": 2,
        "precision_loss": 3,
        "scipy_stop": 4,
        "max_iter": 5,
        "non_descent": 6,
        "step_tol": 7,
        "stopped": 8,
        "grad_tol": 9,
    }
    selected = sorted(
        candidates,
        key=lambda run: (priority.get(_status(run), 9), run.function_name, _start_label(run), run.optimizer_name, str(run.params)),
    )[:limit]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Случаи несходимости и особенности остановки}",
        r"\fitreporttable{%",
        r"\begin{tabular}{llllrl}",
        r"\toprule",
        r"Function & $x_0$/id & Method & status & $f(x_k)$ & comment\\",
        r"\midrule",
    ]
    for run in selected:
        result = run.result
        if result is None:
            comment = run.error or "execution error"
            final_f = None
        else:
            comment = result.message
            final_f = result.f
        optimizer = run.optimizer_name if run.optimizer_name != "LBFGS" else f"LBFGS m={run.params.get('m')}"
        lines.append(
            " & ".join(
                [
                    _latex_escape(run.function_name),
                    _latex_escape(_start_label(run) if _start_label(run) != "--" else _function_label(run)),
                    _latex_escape(optimizer),
                    _latex_escape(_status(run)),
                    _format_float(final_f),
                    _latex_escape(comment),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", r"\endgroup"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    rows.sort(key=lambda item: int(item["m"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(row["m"]) for row in rows], [float(row["value"]) for row in rows], marker="o", linewidth=2.0)
    ax.set_title(f"L-BFGS memory influence on {metric}")
    ax.set_xlabel("memory size m")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    png_path = output_dir / f"lab4_lbfgs_memory_{safe_stem(metric)}.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {"png": png_path}


def plot_lab4_metric_dependencies(results: list[ExperimentRun], output_dir: Path) -> dict[str, Path]:
    """Backward-compatible wrapper for Lab 4 dependency plots."""
    return _save_report_dependency_plots(results, output_dir)


def plot_lab4_metric_tables(results: list[ExperimentRun], output_dir: Path) -> dict[str, Path]:
    """Backward-compatible wrapper for Lab 4 aggregate metric tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dependency_n_k10_split": _save_dependency_table(results, output_dir / "dependency_n_k10_split.tex", fixed_key="k", fixed_value=10, x_key="n"),
        "dependency_k_n10_split": _save_dependency_table(results, output_dir / "dependency_k_n10_split.tex", fixed_key="n", fixed_value=10, x_key="k"),
        "cost_generated_quadratic": _save_generated_quadratic_cost_table(results, output_dir / "cost_generated_quadratic.tex"),
    }


def plot_lab4_optimizer_comparison(results: list[ExperimentRun], output_dir: Path) -> dict[str, Path]:
    """Backward-compatible placeholder; report now uses grouped dependency plots."""
    return plot_lab4_metric_dependencies(results, output_dir)
