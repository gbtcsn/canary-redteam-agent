"""
Load prompt datasets (JSONL) and build comparison-baseline trial lists.

Data files (under ``data/``):

  - ``starting_prompts.jsonl`` — starting prompts for the 3-round agent pipeline
    (Round 0 input; Rounds 1–2 mutate these via LLMMutator in ``mutators.py``).
  - ``fixed_baseline_prompts.jsonl`` — fixed hand-written attacks for the
    *static* comparison baseline (no mutation pipeline).
  - ``benign_tasks.jsonl`` — non-attack tasks (optional evaluation).

The agent experiment only needs ``load_jsonl`` + ``starting_prompts.jsonl``.
The other helpers exist for ``run_baselines.py`` comparisons.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .mutators import mutate_random

# Default dataset paths (relative to project root).
STARTING_PROMPTS_FILE = "data/starting_prompts.jsonl"
FIXED_BASELINE_PROMPTS_FILE = "data/fixed_baseline_prompts.jsonl"


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file; one JSON object per non-empty line."""
    path = Path(path)
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_starting_prompts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load starting prompts for the agent pipeline (JSONL records with a ``prompt`` field)."""
    return load_jsonl(path)


def load_fixed_baseline_prompts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load fixed baseline prompt records (static comparison baseline)."""
    return load_jsonl(path)


def load_benign_tasks(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load benign task prompts from a JSONL file."""
    return load_jsonl(path)


def list_fixed_baseline_prompts(path: Union[str, Path]) -> List[str]:
    """Return prompt strings from ``fixed_baseline_prompts.jsonl``."""
    return [
        item.get("prompt", "")
        for item in load_fixed_baseline_prompts(path)
        if item.get("prompt")
    ]


def list_starting_prompt_strings(path: Union[str, Path]) -> List[str]:
    """Return prompt strings from ``starting_prompts.jsonl``."""
    return [
        item.get("prompt", "")
        for item in load_starting_prompts(path)
        if item.get("prompt")
    ]


def build_random_mutation_baseline(
    starting_prompts_path: Union[str, Path],
    llm: Any = None,
    num_trials: int = 20,
    num_mutations_per_trial: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Build prompts for the *random-mutation* comparison baseline.

    For each trial, pick a starting prompt at random (with replacement), apply
    ``num_mutations_per_trial`` LLM mutations (no 3-round structure).
    """
    if llm is None:
        raise ValueError(
            "build_random_mutation_baseline requires an `llm` argument (LLMMutator backend)."
        )

    records = load_starting_prompts(starting_prompts_path)
    pool = [item.get("prompt", "") for item in records if item.get("prompt")]
    if not pool:
        raise ValueError(f"No prompts found in: {starting_prompts_path}")

    rng = random.Random(seed)
    results: List[str] = []
    for i in range(num_trials):
        base = rng.choice(pool)
        trial_seed = (seed + i) if seed is not None else None
        mutated = mutate_random(
            base,
            llm=llm,
            num_mutations=num_mutations_per_trial,
            seed=trial_seed,
        )
        results.append(mutated)
    return results
