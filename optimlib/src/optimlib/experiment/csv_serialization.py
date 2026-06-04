"""CSV serialization helpers for experiment result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe_csv(dataframe: pd.DataFrame, output_dir: Path, stem: str = "summary") -> dict[str, Path]:
    """Save a DataFrame as a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    dataframe.to_csv(csv_path, index=False)
    return {"csv": csv_path}


def save_dataframe(dataframe: pd.DataFrame, output_dir: Path, stem: str = "summary") -> dict[str, Path]:
    """Backward-compatible alias for CSV-only DataFrame serialization."""
    return save_dataframe_csv(dataframe, output_dir, stem=stem)
