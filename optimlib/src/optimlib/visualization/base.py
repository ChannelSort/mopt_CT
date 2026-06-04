"""Shared visualization helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from optimlib.core.base import FloatArray
from optimlib.experiment.runner import ExperimentRun


def safe_stem(value: str) -> str:
    """Return a filesystem-safe stem fragment."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def numeric_result_metric(run: ExperimentRun, metric: str) -> float | None:
    """Return a numeric ``OptimizationResult`` metric by name."""
    if run.result is None:
        return None
    value = getattr(run.result, metric, None)
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return None


def result_weights(run: ExperimentRun) -> FloatArray | None:
    """Return final vector parameters for a run."""
    if run.result is None or isinstance(run.result.x, float):
        return None
    return np.asarray(run.result.x, dtype=np.float64).reshape(-1)


def set_log_scale_if_positive(ax: Axes, data: pd.DataFrame, metric: str) -> None:
    """Use a logarithmic y-axis only when the metric contains positive values."""
    if metric in data and bool((pd.to_numeric(data[metric], errors="coerce") > 0.0).any()):
        ax.set_yscale("log")
