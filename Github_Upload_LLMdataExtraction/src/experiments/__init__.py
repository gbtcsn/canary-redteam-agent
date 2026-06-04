"""
Experiments Module

This module contains experiment runners and hyperparameter sweep utilities.
It provides the main entry points for running exploit generation experiments.

Purpose:
- Execute single experiments with specific configurations
- Run hyperparameter sweeps to find optimal settings
- Provide command-line interfaces for experiment execution
- Coordinate all components for full experiment runs

Design Decisions:
- Separate runners: Single experiments vs sweeps
- Configuration-driven: All parameters from YAML
- Reproducible: Saves full configuration with results
- CLI-friendly: Easy to run from command line

Technical Approach:
- Reuses components from other modules (agent, evaluation, targetapp)
- Orchestrates experiment lifecycle
- Handles logging and result persistence
- Provides progress reporting
"""

# Export what is implemented
from .run_baselines import main as run_baselines_main
from .run_experiment import run_experiment, main as run_experiment_main

__all__ = ["run_baselines_main", "run_experiment", "run_experiment_main"]
