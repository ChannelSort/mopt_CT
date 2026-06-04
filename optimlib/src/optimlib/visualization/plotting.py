"""Backward-compatible plotting imports.

New code should import plotting helpers from focused modules such as
``optimlib.visualization.contour`` or ``optimlib.visualization.regression``.
This module is kept so existing laboratory scripts continue to work.
"""

from __future__ import annotations

from optimlib.visualization.contour import plot_contours_and_trajectories
from optimlib.visualization.convergence import plot_convergence, plot_param_sensitivity
from optimlib.visualization.lab2 import plot_lab2_constant_step_alpha, plot_lab2_tolerance_dependencies
from optimlib.visualization.lab4 import (
    plot_lab4_lbfgs_memory,
    plot_lab4_metric_dependencies,
    plot_lab4_metric_tables,
    plot_lab4_optimizer_comparison,
)
from optimlib.visualization.regression import (
    plot_lab5_batch_size_comparison,
    plot_lab5_coefficients,
    plot_lab5_loss_by_gradient_calls,
    plot_lab5_loss_history,
    plot_lab5_method_comparison,
    plot_lab5_regression_fit,
    plot_lab5_regularization_comparison,
)

__all__ = [
    "plot_contours_and_trajectories",
    "plot_convergence",
    "plot_lab2_constant_step_alpha",
    "plot_lab2_tolerance_dependencies",
    "plot_lab4_lbfgs_memory",
    "plot_lab4_metric_dependencies",
    "plot_lab4_metric_tables",
    "plot_lab4_optimizer_comparison",
    "plot_lab5_batch_size_comparison",
    "plot_lab5_coefficients",
    "plot_lab5_loss_by_gradient_calls",
    "plot_lab5_loss_history",
    "plot_lab5_method_comparison",
    "plot_lab5_regression_fit",
    "plot_lab5_regularization_comparison",
    "plot_param_sensitivity",
]
