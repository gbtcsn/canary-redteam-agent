"""
LLM Exploit Generation Agent

This package implements an automated exploit generation agent for discovering
prompt injection vulnerabilities in locally hosted LLM applications.

Main Modules:
- targetapp: Sandboxed LLM application (target for exploits)
- agent: Exploit generation agent (core contribution)
- evaluation: Deterministic oracles and metrics
- experiments: Experiment runners and hyperparameter sweeps
- analysis: Result analysis and visualization utilities

This is an academic research project focused on security evaluation methodology.
All testing is performed on local, sandboxed systems only.
"""

__version__ = "0.1.0"
