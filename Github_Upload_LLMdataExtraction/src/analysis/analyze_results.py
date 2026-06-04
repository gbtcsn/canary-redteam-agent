"""
Result Analysis Utilities

This module provides utilities for analyzing experiment results. It loads
results from JSONL logs and SQLite databases, computes statistics, and
generates analysis reports.

Purpose:
- Load and parse experiment results
- Compute aggregate statistics
- Compare different strategies/baselines
- Generate analysis reports

Design Decisions:
- Data loading: Reads from JSONL logs and SQLite
- Pandas-based: Uses pandas for data manipulation
- Statistical analysis: Basic statistics (mean, median, CI)
- Comparative: Designed for comparing multiple experiments

Technical Approach:
- Result loading: Parse JSONL logs into pandas DataFrames
- Aggregation: Group by strategy, compute metrics
- Statistical tests: T-tests, Mann-Whitney U for significance
- Report generation: Markdown/HTML reports with tables and statistics

Analysis Functions:

1. load_results():
   - Load results from experiment directory
   - Parse JSONL logs
   - Query SQLite for metadata
   - Return structured data

2. compute_statistics():
   - Aggregate metrics across trials
   - Compute confidence intervals
   - Statistical significance tests

3. compare_strategies():
   - Compare agent vs baselines
   - Statistical significance testing
   - Effect size calculations

4. generate_report():
   - Create markdown report with results
   - Include tables, statistics, visualizations
   - Save to results/analysis/
"""

# This file will contain:
# - load_results(): Load experiment results
# - compute_statistics(): Statistical analysis
# - compare_strategies(): Strategy comparison
# - generate_report(): Report generation

# Example structure (commented out until implementation):
"""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from scipy import stats

def load_results(experiment_dir: Path) -> pd.DataFrame:
    # Load results from JSONL logs
    # Returns: DataFrame with all trial results
    results = []
    log_file = experiment_dir / "trials.jsonl"
    with open(log_file) as f:
        for line in f:
            results.append(json.loads(line))
    return pd.DataFrame(results)

def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    # Compute aggregate statistics
    # Returns: Dictionary with statistics
    stats = {
        "attack_success_rate": df["is_successful"].mean() * 100,
        "total_attempts": len(df),
        "successful_attempts": df["is_successful"].sum(),
        # ... more statistics
    }
    return stats

def compare_strategies(results_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    # Compare multiple strategies
    # results_dict: {"strategy_name": DataFrame}
    # Returns: Comparison statistics and significance tests
    pass

def generate_report(experiment_dir: Path, output_path: Path):
    # Generate analysis report
    # Loads results, computes statistics, creates report
    pass
"""
