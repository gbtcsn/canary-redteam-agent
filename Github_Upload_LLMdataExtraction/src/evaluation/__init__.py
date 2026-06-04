"""
Evaluation Module

This module implements deterministic evaluation oracles and metrics for assessing
whether an exploit attempt was successful. The focus is on deterministic checks
rather than LLM-based judges for reliability and reproducibility.

Purpose:
- Determine if an exploit attempt succeeded (detected vulnerability)
- Compute metrics (attack success rate, attempts to success, severity)
- Classify failure modes for analysis
- Provide reproducible, deterministic evaluation

Design Decisions:
- Deterministic oracles: No LLM judges (more reliable, reproducible)
- Multiple detection methods: Canary tokens, behavioral checks, response analysis
- Configurable thresholds: Adjustable sensitivity for different experiments
- Structured results: Consistent result format for analysis

Technical Approach:
- Canary detection: Check for secret tokens in responses
- Behavioral analysis: Detect unexpected tool calls, format violations
- Response analysis: Check for suspicious patterns, length anomalies
- Severity scoring: Assign severity levels to successful exploits

This module is critical for the research - it provides the ground truth for
whether exploits are successful, enabling fair comparison of different strategies.
"""

# Export main components (Step C: Oracle)
from .oracle import Oracle, Verdict

__all__ = [
    "Oracle",
    "Verdict",
]
