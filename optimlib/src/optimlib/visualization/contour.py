"""Contour and trajectory plots for two-dimensional objectives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from optimlib.core.base import FloatArray
from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction
from optimlib.utils.numerics import max_abs


_MAX_PLOT_COORDINATE = 1e6
_CONTOUR_FLOOR = 1e-8
_FIXED_LIMITS: dict[str, tuple[float, float, float, float]] = {
    "Ackley": (-3.0, 3.0, -3.0, 3.0),
    "Himmelblau": (-5.0, 5.0, -5.0, 5.0),
}


def _history_points(run: ExperimentRun) -> FloatArray:
    if run.result is None or not run.result.history:
        return np.empty((0, 2), dtype=np.float64)
    points = [
        state.x
        for state in run.result.history
        if state.x.size >= 2 and np.all(np.isfinite(state.x[:2])) and max_abs(state.x[:2]) <= _MAX_PLOT_COORDINATE
    ]
    return np.vstack(points)[:, :2] if points else np.empty((0, 2), dtype=np.float64)


def _limits(func: MultivariateFunction, results: Iterable[ExperimentRun]) -> tuple[float, float, float, float]:
    points = [func.initial_point()]
    points.extend(func.global_minimizers)
    for run in results:
        history = _history_points(run)
        if history.size:
            points.extend(history)
        if run.result is not None and not isinstance(run.result.x, float) and max_abs(run.result.x[:2]) <= _MAX_PLOT_COORDINATE:
            points.append(run.result.x[:2])
    stacked = np.vstack(points)
    lower = np.nanpercentile(stacked, 2.0, axis=0)
    upper = np.nanpercentile(stacked, 98.0, axis=0)
    span = np.maximum(upper - lower, 1.0)
    pad = 0.3 * span
    return float(lower[0] - pad[0]), float(upper[0] + pad[0]), float(lower[1] - pad[1]), float(upper[1] + pad[1])


def contour_limits(func: MultivariateFunction, results: Iterable[ExperimentRun]) -> tuple[float, float, float, float]:
    """Return shared contour limits for Lab 2 and Lab 3 trajectory plots."""
    return _FIXED_LIMITS.get(func.name, _limits(func, results))


def _surface_values(
    func: MultivariateFunction,
    limits: tuple[float, float, float, float],
    grid_size: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    x_min, x_max, y_min, y_max = limits
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(
            [func(np.array([x, y], dtype=np.float64)) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)]
        ).reshape(xx.shape)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(xx, dtype=np.float64)
    else:
        values = np.where(np.isfinite(values), values, float(np.max(finite)))
    return xx, yy, values


def _draw_shared_contours(
    fig: plt.Figure,
    ax: plt.Axes,
    func: MultivariateFunction,
    xx: FloatArray,
    yy: FloatArray,
    values: FloatArray,
    *,
    filled_levels: int,
) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float64)
    positive = finite[finite > 0.0]
    force_log = func.name in {"Ackley", "Himmelblau"}
    use_log = force_log or (positive.size > 0 and float(np.max(positive) / np.min(positive)) > 1e3)

    if use_log:
        vmax = max(float(np.max(np.maximum(finite, _CONTOUR_FLOOR))), _CONTOUR_FLOOR * 10.0)
        levels = np.geomspace(_CONTOUR_FLOOR, vmax, filled_levels)
        z_plot = np.maximum(values, _CONTOUR_FLOOR)
        contour = ax.contourf(
            xx,
            yy,
            z_plot,
            levels=levels,
            cmap="viridis",
            norm=LogNorm(vmin=_CONTOUR_FLOOR, vmax=vmax),
            extend="max",
        )
        ax.contour(
            xx,
            yy,
            z_plot,
            levels=levels[:: max(1, filled_levels // 14)],
            colors="black",
            linewidths=0.25,
            alpha=0.25,
            norm=LogNorm(vmin=_CONTOUR_FLOOR, vmax=vmax),
        )
    else:
        contour = ax.contourf(xx, yy, values, levels=filled_levels, cmap="viridis")
        ax.contour(xx, yy, values, levels=14, colors="black", linewidths=0.25, alpha=0.25)
    fig.colorbar(contour, ax=ax, label="f(x)")


def plot_trajectory_contours(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str,
    *,
    title: str | None = None,
    run_label: Callable[[ExperimentRun], str] | None = None,
    grid_size: int = 260,
    filled_levels: int = 55,
    figsize: tuple[float, float] = (8.2, 6.2),
    legend_fontsize: float = 6.6,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    start_text: str | None = None,
    show_arrows: bool = False,
) -> dict[str, Path]:
    """Plot shared contour background with optimization trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = contour_limits(func, results)
    x_min, x_max, y_min, y_max = limits
    xx, yy, values = _surface_values(func, limits, grid_size)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_shared_contours(fig, ax, func, xx, yy, values, filled_levels=filled_levels)

    colors = ["#d55e00", "#0072b2", "#009e73", "#cc79a7", "#f0e442", "#56b4e9", "#332288", "#88ccee"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for index, run in enumerate(results):
        points = _history_points(run)
        if points.size == 0:
            continue
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        points = points[mask]
        if points.size == 0:
            continue
        color = colors[index % len(colors)]
        mark_every = max(1, points.shape[0] // 16)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linestyle=linestyles[index % len(linestyles)],
            linewidth=1.8,
            marker="o",
            markersize=2.8,
            markevery=mark_every,
            label=run_label(run) if run_label else f"{run.optimizer_name} {run.params}",
        )
        ax.scatter(points[-1, 0], points[-1, 1], marker="s", s=42, color=color, edgecolors="black", linewidths=0.7)
        if show_arrows and points.shape[0] > 1:
            delta = np.diff(points, axis=0)
            ax.quiver(
                points[:-1, 0],
                points[:-1, 1],
                delta[:, 0],
                delta[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.002,
                alpha=0.55,
            )

    start = func.initial_point()
    ax.scatter(start[0], start[1], marker="X", s=100, c="white", edgecolors="black", linewidths=1.1, label="start")
    if start_text is not None:
        ax.text(start[0] + 0.08, start[1] - 0.16, start_text, color="white", fontsize=9, weight="bold")
    for minimizer in func.global_minimizers:
        ax.scatter(minimizer[0], minimizer[1], marker="*", s=170, c="gold", edgecolors="black", linewidths=0.9)
    ax.scatter([], [], marker="*", s=130, c="gold", edgecolors="black", linewidths=0.9, label="global min")
    if results:
        ax.scatter([], [], marker="s", s=42, c="white", edgecolors="black", linewidths=0.7, label="final")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title or f"{func.name}: trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.18)
    if results:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.88, ncol=legend_ncol)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"png": png}


def plot_contours_and_trajectories(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str | None = None,
) -> dict[str, Path]:
    """Plot contours with clipped optimization trajectories."""
    stem = basename or f"{func.name}_trajectories"
    return plot_trajectory_contours(
        func,
        results,
        output_dir,
        stem,
        title=f"{func.name}: trajectories",
        legend_fontsize=6.0,
        show_arrows=True,
    )
