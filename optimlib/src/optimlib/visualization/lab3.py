"""Lab 3 report-specific plots and tables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.visualization.contour import plot_trajectory_contours


INERTIAL_OPTIMIZERS = ("Momentum", "Nesterov")
ADAPTIVE_OPTIMIZERS = ("AdaGrad", "RMSProp", "AdaDelta", "Adam")
QUADRATIC_FUNCTIONS = ("WellConditionedQuadratic", "IllConditionedQuadratic")
COMPLEX_FUNCTIONS = ("Rosenbrock", "Ackley", "Himmelblau")

COMPLEX_PARAM_SETS: dict[str, tuple[dict[str, float], ...]] = {
    "Momentum": ({"alpha": 1e-3, "beta": 0.85}, {"alpha": 3e-3, "beta": 0.9}),
    "Nesterov": ({"alpha": 1e-3, "beta": 0.85}, {"alpha": 3e-3, "beta": 0.9}),
    "AdaGrad": ({"alpha": 1e-2}, {"alpha": 1e-1}),
    "RMSProp": ({"alpha": 1e-3, "rho": 0.9}, {"alpha": 3e-3, "rho": 0.9}),
    "AdaDelta": ({"rho": 0.5}, {"rho": 0.9}),
    "Adam": ({"alpha": 1e-3, "beta1": 0.9, "beta2": 0.99}, {"alpha": 3e-3, "beta1": 0.85, "beta2": 0.99}),
}


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value).strip("_")


def _params_match(actual: dict[str, Any], expected: dict[str, float]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(actual_value, int | float) and isinstance(expected_value, int | float):
            if not math.isclose(float(actual_value), float(expected_value), rel_tol=1e-12, abs_tol=1e-12):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _is_finite_result(run: ExperimentRun) -> bool:
    return run.result is not None and np.isfinite(run.result.f)


def _representative_run(runs: list[ExperimentRun], optimizer: str) -> ExperimentRun | None:
    candidates = [run for run in runs if run.optimizer_name == optimizer and _is_finite_result(run)]
    if not candidates:
        return None
    converged = [run for run in candidates if run.result is not None and run.result.converged]
    if converged:
        return min(converged, key=lambda run: run.result.n_iter if run.result is not None else 10**18)
    return min(candidates, key=lambda run: run.result.f if run.result is not None else float("inf"))


def _compact_params(params: dict[str, Any]) -> str:
    parts = []
    for key in ("alpha", "beta", "rho", "beta1", "beta2"):
        if key in params:
            value = params[key]
            parts.append(f"{key}={float(value):g}" if isinstance(value, int | float) else f"{key}={value}")
    return ", ".join(parts) if parts else "{}"


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
    if run.result is None:
        message = (run.error or "failed").lower()
        if "overflow" in message:
            return "overflow"
        if "nan" in message:
            return "nan"
        return "diverged"
    if run.result.converged:
        return "true"
    message = run.result.message.lower()
    if "coordinate magnitude" in message or "objective magnitude" in message:
        return "diverged"
    if "numerical" in message or "overflow" in message:
        return "overflow"
    if "nan" in message:
        return "nan"
    if "maximum iterations" in message:
        return "false (max_iter reached)"
    return "false"


def _run_for_params(runs: list[ExperimentRun], optimizer: str, params: dict[str, float]) -> ExperimentRun | None:
    for run in runs:
        if run.optimizer_name == optimizer and _params_match(run.params, params):
            return run
    return None


def _iteration_cell(run: ExperimentRun | None) -> str:
    if run is None or run.result is None:
        return "--"
    suffix = "" if run.result.converged else "*"
    return f"{run.result.n_iter}{suffix}"


def _plot_group(
    func: MultivariateFunction,
    runs: list[ExperimentRun],
    output_dir: Path,
    basename: str,
    title: str,
) -> Path:
    paths = plot_trajectory_contours(
        func,
        runs,
        output_dir,
        basename,
        title=title,
        run_label=lambda run: f"{run.optimizer_name}: {_compact_params(run.params)}",
        legend_fontsize=6.6,
    )
    return paths["png"]


def plot_lab3_grouped_trajectories(
    func: MultivariateFunction,
    family: list[ExperimentRun],
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    """Plot one inertial and one adaptive trajectory figure for a function variant."""
    selected = {optimizer: _representative_run(family, optimizer) for optimizer in (*INERTIAL_OPTIMIZERS, *ADAPTIVE_OPTIMIZERS)}
    paths: dict[str, Path] = {}
    inertial = [selected[name] for name in INERTIAL_OPTIMIZERS if selected[name] is not None]
    adaptive = [selected[name] for name in ADAPTIVE_OPTIMIZERS if selected[name] is not None]
    if inertial:
        paths["inertial"] = _plot_group(func, inertial, output_dir, f"{stem}_inertial_trajectories", f"{func.name}: inertial trajectories")
    if adaptive:
        paths["adaptive"] = _plot_group(func, adaptive, output_dir, f"{stem}_adaptive_trajectories", f"{func.name}: adaptive trajectories")
    return paths


def save_lab3_quadratic_best_table(runs: list[ExperimentRun], table_dir: Path) -> Path:
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "quadratic_best.tex"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Лучшие параметры на квадратичных функциях при $\varepsilon=10^{-8}$}",
        r"\fitreporttable{%",
        r"\begin{tabular}{lllr}",
        r"\toprule",
        r"Функция & Метод & Лучший набор параметров & Итерации\\",
        r"\midrule",
    ]
    labels = {"WellConditionedQuadratic": "$f$", "IllConditionedQuadratic": "$g$"}
    for function_name in QUADRATIC_FUNCTIONS:
        family = [run for run in runs if run.function_name == function_name and dict(run.function_params) == {"x0": [-2.0, -2.0]}]
        for optimizer in (*INERTIAL_OPTIMIZERS, *ADAPTIVE_OPTIMIZERS):
            run = _representative_run(family, optimizer)
            params = "--" if run is None else _latex_escape(_compact_params(run.params))
            iterations = "--" if run is None or run.result is None else str(run.result.n_iter)
            lines.append(f"{labels[function_name]} & {optimizer} & {params} & {iterations}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_lab3_quadratic_sensitivity_tables(runs: list[ExperimentRun], table_dir: Path) -> Path:
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "quadratic_sensitivity_tables.tex"
    lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\textit{Примечание.} Число со звездочкой означает, что метод дошел до лимита итераций без выполнения критерия по норме градиента.",
        "",
    ]
    function_titles = {
        "WellConditionedQuadratic": "Хорошо обусловленная квадратичная функция, $x_0=(-2,-2)$",
        "IllConditionedQuadratic": "Плохо обусловленная квадратичная функция, $x_0=(-2,-2)$",
    }
    two_param_specs = {
        "Momentum": ("beta", "alpha", None),
        "Nesterov": ("beta", "alpha", None),
        "RMSProp": ("rho", "alpha", None),
        "Adam": ("beta1", "beta2", {"alpha": 1e-2}),
    }
    one_param_specs = {"AdaGrad": "alpha", "AdaDelta": "rho"}
    for function_name in QUADRATIC_FUNCTIONS:
        family = [run for run in runs if run.function_name == function_name and dict(run.function_params) == {"x0": [-2.0, -2.0]}]
        lines.append(rf"\subsubsection*{{{function_titles[function_name]}}}")
        for optimizer, (row_param, col_param, fixed) in two_param_specs.items():
            subset = [run for run in family if run.optimizer_name == optimizer]
            if fixed:
                subset = [run for run in subset if _params_match(run.params, fixed)]
            rows = sorted({float(run.params[row_param]) for run in subset})
            cols = sorted({float(run.params[col_param]) for run in subset})
            lines.extend([r"\begin{table}[H]", r"\centering", rf"\caption{{{optimizer}: число итераций}}", r"\fitreporttable{%", rf"\begin{{tabular}}{{l{'r' * len(cols)}}}", r"\toprule"])
            lines.append(_latex_escape(row_param) + " / " + _latex_escape(col_param) + " & " + " & ".join(f"{col:g}" for col in cols) + r"\\")
            lines.append(r"\midrule")
            for row_value in rows:
                cells = []
                for col_value in cols:
                    expected = {row_param: row_value, col_param: col_value}
                    if fixed:
                        expected.update(fixed)
                    cells.append(_iteration_cell(_run_for_params(subset, optimizer, expected)))
                lines.append(f"{row_value:g} & " + " & ".join(cells) + r"\\")
            lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
        for optimizer, param in one_param_specs.items():
            subset = sorted([run for run in family if run.optimizer_name == optimizer], key=lambda run: float(run.params[param]))
            lines.extend([r"\begin{table}[H]", r"\centering", rf"\caption{{{optimizer}: число итераций}}", r"\begin{tabular}{lrr}", r"\toprule"])
            lines.append(_latex_escape(param) + r" & Итерации & Сошелся\\")
            lines.append(r"\midrule")
            for run in subset:
                converged = "true" if run.result is not None and run.result.converged else "false"
                iterations = "--" if run.result is None else str(run.result.n_iter)
                lines.append(f"{float(run.params[param]):g} & {iterations} & {converged}\\\\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_lab3_complex_tables(runs: list[ExperimentRun], table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for function_name in COMPLEX_FUNCTIONS:
        path = table_dir / f"complex_{function_name}.tex"
        paths[function_name] = path
        function_runs = [run for run in runs if run.function_name == function_name]
        starts = sorted({str(dict(run.function_params).get("x0", [])) for run in function_runs})
        lines = [
            r"\begingroup",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
        ]
        for start in starts:
            lines.extend(
                [
                    r"\begin{table}[H]",
                    r"\centering",
                    rf"\caption{{Запуски на функции {function_name}, $x_0={_latex_escape(start)}$}}",
                    r"\fitreporttable{%",
                    r"\begin{tabular}{lllllllll}",
                    r"\toprule",
                    r"function & x0 & optimizer & params & converged & iterations & f\_calls & grad\_calls & $f(x_k)$\\",
                    r"\midrule",
                ]
            )
            start_runs = [run for run in function_runs if str(dict(run.function_params).get("x0", [])) == start]
            for optimizer, param_sets in COMPLEX_PARAM_SETS.items():
                for params in param_sets:
                    run = _run_for_params(start_runs, optimizer, params)
                    if run is None or run.result is None:
                        row = [function_name, start, optimizer, _compact_params(params), "diverged", "--", "--", "--", "nan"]
                    else:
                        row = [
                            function_name,
                            start,
                            optimizer,
                            _compact_params(run.params),
                            _status(run),
                            str(run.result.n_iter),
                            str(run.result.n_calls),
                            str(run.result.n_grad_calls),
                            _format_float(run.result.f),
                        ]
                    lines.append(" & ".join(_latex_escape(item) for item in row) + r"\\")
            lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
        lines.append(r"\endgroup")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
