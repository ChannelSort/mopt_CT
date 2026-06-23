"""Lab 4 specific metric plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
    paths["lbfgs_memory_fixed"] = _save_lbfgs_memory_table(runs, output_dir / "lbfgs_memory_fixed.tex")
    paths["quadratic2d_starts"] = _save_quadratic2d_start_table(runs, output_dir / "quadratic2d_starts.tex")
    paths["problem_cases"] = _save_problem_cases_table(runs, output_dir / "problem_cases.tex")
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
