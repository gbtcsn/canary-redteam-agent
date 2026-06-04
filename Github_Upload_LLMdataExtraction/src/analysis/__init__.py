"""
Analysis Module

This module provides utilities for analyzing experiment results. It loads
results from logs, computes statistics, and generates visualizations.

Purpose:
- Load and parse experiment results
- Compute aggregate statistics and metrics
- Compare different strategies and baselines
- Generate analysis reports and visualizations

Design Decisions:
- Pandas-based: Uses pandas for data manipulation
- Modular: Separate functions for loading, analysis, plotting
- Report generation: Creates markdown/HTML reports
- Visualization: Publication-quality figures

Technical Approach:
- Result loading: Parse JSONL and SQLite
- Statistical analysis: Basic stats, significance tests
- Comparative analysis: Strategy comparison
- Visualization: Matplotlib/Seaborn for plots
"""

# Export main components
from .analyze_results import (
    load_results,
    compute_statistics,
    compare_strategies,
    generate_report,
)
from .plots import (
    plot_success_rate_over_time,
    plot_attempts_to_success,
    plot_severity_distribution,
    plot_strategy_comparison,
    plot_failure_modes,
)

__all__ = [
    "load_results",
    "compute_statistics",
    "compare_strategies",
    "generate_report",
    "plot_success_rate_over_time",
    "plot_attempts_to_success",
    "plot_severity_distribution",
    "plot_strategy_comparison",
    "plot_failure_modes",
]
