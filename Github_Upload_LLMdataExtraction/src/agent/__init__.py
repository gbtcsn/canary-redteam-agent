"""
Exploit Generation Agent Module

Core pieces of the 3-round mutation pipeline:

  - ``prompt_datasets`` — load JSONL prompt corpora (starting prompts, baselines)
  - ``mutators`` — LLM-driven prompt mutation (Round 1 / Round 2)
  - ``runner`` — orchestrates Round 0 → 1 → 2 and logs trials
"""

from .prompt_datasets import (
    FIXED_BASELINE_PROMPTS_FILE,
    STARTING_PROMPTS_FILE,
    build_random_mutation_baseline,
    list_fixed_baseline_prompts,
    list_starting_prompt_strings,
    load_benign_tasks,
    load_fixed_baseline_prompts,
    load_jsonl,
    load_starting_prompts,
)
from .mutators import Mutator, MutationOperator, mutate_random, LLMMutator
from .runner import AgentRunner, RoundsRunner

__all__ = [
    "STARTING_PROMPTS_FILE",
    "FIXED_BASELINE_PROMPTS_FILE",
    "load_jsonl",
    "load_starting_prompts",
    "load_fixed_baseline_prompts",
    "load_benign_tasks",
    "list_starting_prompt_strings",
    "list_fixed_baseline_prompts",
    "build_random_mutation_baseline",
    "Mutator",
    "MutationOperator",
    "mutate_random",
    "LLMMutator",
    "AgentRunner",
    "RoundsRunner",
]
