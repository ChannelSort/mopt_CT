"""Benchmark summary helpers."""

from __future__ import annotations

from typing import cast

import pandas as pd

from optimlib.experiment.runner import ExperimentRun


def best_by_calls(runs: list[ExperimentRun]) -> pd.DataFrame:
    """Return successful runs sorted by objective and gradient calls."""
    rows: list[dict[str, object]] = []
    for run in runs:
        if run.result is None or not run.result.converged:
            continue
        rows.append(
            {
                "function": run.function_name,
                "optimizer": run.optimizer_name,
                "tolerance": run.tolerance,
                "n_calls": run.result.n_calls,
                "n_grad_calls": run.result.n_grad_calls,
                "f": run.result.f,
            }
        )
    dataframe = cast(pd.DataFrame, pd.DataFrame(rows).sort_values(["function", "n_calls", "n_grad_calls"]))
    return dataframe
