"""
Rounds Runner - 3-round mutation pipeline.

Pipeline:

  Round 0: feed each baseline prompt to the target as-is. Log every trial.
  Round 1 (fork by technique): for every Round-0 prompt, produce three
           mutated variants -- one per technique (logic_traps,
           roleplay_social, encoding) -- and feed each to the target.
  Round 2 (random follow-up): for every Round-1 result, apply ONE
           randomly-chosen technique and feed the result to the target.

Total trials per run: N + 3N + 3N = 7N, where N is the number of base
baseline prompts (capped via baseline.count in configs/experiment.yaml).

Outputs (in results/runs/<run_id>/):
  - trials.jsonl       : one JSON object per call (round, technique, path, ...)
  - summary.json       : aggregated metrics (per-round, per-technique,
                         per-combination ASR) from evaluation.metrics
  - config_snapshot.yaml : the resolved config used for the run
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Ensure project root is importable when this module is used standalone.
def _ensure_paths(base_path: Path) -> None:
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))


# Techniques are pinned here (matches LLMMutator.TECHNIQUES) so the runner
# can fail fast on unknown values without circular imports.
DEFAULT_TECHNIQUES = ("logic_traps", "roleplay_social", "encoding")


class RoundsRunner:
    """
    Orchestrates the 3-round mutation pipeline against a TargetApp.

    Constructor signature is intentionally identical to the previous
    AgentRunner so existing callers (run_experiment.py) keep working.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        base_path: Path,
        target_app: Any,
        oracle: Any,
        policy: Dict[str, Any],
        run_id: Optional[str] = None,
    ):
        self.config = config
        self.base_path = Path(base_path)
        self.target_app = target_app
        self.oracle = oracle
        self.policy = policy
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_path / "results" / "runs" / self.run_id
        self.trials: List[Dict[str, Any]] = []
        self._next_trial_id = 0

    # ------------------------------------------------------------------ I/O

    def _ensure_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _log_trial(self, trial: Dict[str, Any]) -> None:
        """Append one trial to trials.jsonl (incremental, crash-safe)."""
        self._ensure_run_dir()
        path = self.run_dir / "trials.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trial, ensure_ascii=False) + "\n")
        self.trials.append(trial)

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        self._ensure_run_dir()
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _write_config_snapshot(self) -> None:
        self._ensure_run_dir()
        import yaml

        with open(self.run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    # --------------------------------------------------------------- helpers

    def _record_trial(
        self,
        *,
        round_id: int,
        base_prompt_id: int,
        parent_trial_id: Optional[int],
        prompt: str,
        technique_applied: Optional[str],
        mutation_path: List[str],
    ) -> Dict[str, Any]:
        """
        Run a single (prompt -> target -> oracle) call, build the trial
        record, log it, and return it. Centralised so every trial row
        across rounds 0/1/2 has the exact same shape.
        """
        result = self.target_app.run(prompt)
        verdict = self.oracle.evaluate(result, self.policy)

        trial_id = self._next_trial_id
        self._next_trial_id += 1

        canaries_found = []
        details = getattr(verdict, "details", None) or {}
        if isinstance(details, dict):
            canaries_found = list(details.get("canaries_found") or [])

        trial: Dict[str, Any] = {
            "trial_id": trial_id,
            "round": round_id,
            "base_prompt_id": base_prompt_id,
            "parent_trial_id": parent_trial_id,
            "prompt": prompt,
            "prompt_preview": prompt[:100],
            "technique_applied": technique_applied,
            "mutation_path": list(mutation_path),
            "success": bool(verdict.success),
            "canaries_found": canaries_found,
            "output_preview": getattr(result, "output_text", "")[:200]
            if hasattr(result, "output_text")
            else "",
        }
        self._log_trial(trial)
        return trial

    @staticmethod
    def _load_baseline_prompts(base_path: Path, cfg: Dict[str, Any], max_override: Optional[int]) -> List[str]:
        """
        Load the baseline prompt set in file order (deterministic, no sampling).

        Prefers the new `baseline:` config block, falls back to the legacy
        `seeds:` block for backwards compatibility with older configs.

        Selection policy:
          - All prompts in the source file are used, in the order they appear.
          - `baseline.count` (or the `--max-base-prompts` CLI override) acts as
            an upper bound: when set, only the first `count` prompts are kept.
            When `count` is null / 0 / >= file size, every prompt is used.
          - There is NO random sampling: the same source file always yields
            the same baseline list across runs.
        """
        from .prompt_datasets import load_jsonl, STARTING_PROMPTS_FILE

        baseline_cfg = cfg.get("baseline") or cfg.get("seeds") or {}
        rel = baseline_cfg.get("file", STARTING_PROMPTS_FILE)
        count_cfg = baseline_cfg.get("count")  # may be None / null

        prompts_path = base_path / rel
        items = load_jsonl(prompts_path)
        prompts = [it.get("prompt", "") for it in items if it.get("prompt")]

        # Resolve the effective cap. None / 0 / negative -> "use everything".
        caps = [len(prompts)]
        if count_cfg is not None and int(count_cfg) > 0:
            caps.append(int(count_cfg))
        if max_override is not None and int(max_override) > 0:
            caps.append(int(max_override))
        cap = min(caps)
        return prompts[:cap]

    def _build_mutator(self, seed: int):
        """Build a single Mutator + its LocalChatModel for the whole run."""
        from .mutators import Mutator
        from src.targetapp.local_chat_model import LocalChatModel, ModelConfig
        import yaml as _yaml

        with open(self.base_path / "configs" / "targetapp.yaml") as _f:
            _ta_cfg = _yaml.safe_load(_f)
        mutator_llm = LocalChatModel(
            base_url=_ta_cfg["llm"].get("base_url"),
            config=ModelConfig(
                model=_ta_cfg["llm"]["model"],
                temperature=_ta_cfg["llm"].get("temperature", 0.7),
                max_tokens=_ta_cfg["llm"].get("max_tokens", 300),
            ),
            timeout=600,
        )
        return Mutator(llm=mutator_llm, rng=random.Random(seed)), mutator_llm

    # ------------------------------------------------------------------- run

    def run(self, max_base_prompts: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Round 0 -> Round 1 -> Round 2 and return the summary dict.

        Args:
            max_base_prompts: Optional cap on number of base prompts, useful
                              for fast smoke tests (overrides config).
        """
        _ensure_paths(self.base_path)
        from .mutators import LLMMutator
        from src.evaluation.metrics import aggregate_rounds

        repro = self.config.get("reproducibility", {})
        seed = int(repro.get("seed", 42))

        rounds_cfg = self.config.get("rounds", {}) or {}
        techniques = list(rounds_cfg.get("techniques") or DEFAULT_TECHNIQUES)
        enabled_rounds = set(int(r) for r in (rounds_cfg.get("enabled") or [0, 1, 2]))

        # Validate technique names against LLMMutator before any LLM call.
        unknown = [t for t in techniques if t not in LLMMutator.TECHNIQUES]
        if unknown:
            raise ValueError(
                f"Unknown techniques in rounds.techniques: {unknown}. "
                f"Allowed: {LLMMutator.TECHNIQUES}"
            )

        base_prompts = self._load_baseline_prompts(self.base_path, self.config, max_base_prompts)
        if not base_prompts:
            raise ValueError(
                "No baseline prompts loaded. Check baseline.file / baseline.count "
                "in configs/experiment.yaml."
            )

        self._ensure_run_dir()
        if repro.get("save_config", True):
            self._write_config_snapshot()

        # Single shared mutator + LLM for the whole run (deterministic rng).
        mutator, mutator_llm = self._build_mutator(seed)
        # Independent RNG for Round-2 random technique selection so it does
        # not interfere with the Mutator's internal sampling state.
        round2_rng = random.Random(seed + 1)

        round0_trials: List[Dict[str, Any]] = []
        round1_trials: List[Dict[str, Any]] = []

        # ---------------- Round 0: baseline ----------------
        if 0 in enabled_rounds:
            for base_id, prompt in enumerate(base_prompts):
                trial = self._record_trial(
                    round_id=0,
                    base_prompt_id=base_id,
                    parent_trial_id=None,
                    prompt=prompt,
                    technique_applied=None,
                    mutation_path=[],
                )
                round0_trials.append(trial)

        # ---------------- Round 1: fork by technique ----------------
        if 1 in enabled_rounds and round0_trials:
            # Pull the actual LLMMutator instance out of the mutator so we
            # can target a specific technique deterministically.
            llm_mutators = [op for op in mutator.operators if isinstance(op, LLMMutator)]
            if not llm_mutators:
                raise RuntimeError("Mutator has no LLMMutator operator; cannot run Round 1/2.")
            llm_mut: LLMMutator = llm_mutators[0]

            for parent in round0_trials:
                for technique in techniques:
                    mutated = llm_mut.mutate_with_technique(parent["prompt"], technique)
                    trial = self._record_trial(
                        round_id=1,
                        base_prompt_id=parent["base_prompt_id"],
                        parent_trial_id=parent["trial_id"],
                        prompt=mutated,
                        technique_applied=technique,
                        mutation_path=[technique],
                    )
                    round1_trials.append(trial)

        # ---------------- Round 2: random follow-up (STACKED) ----------------
        # Round 2 applies the new technique as an OUTER LAYER on top of the
        # Round-1 mutated prompt -- the Round-1 technique's structure is
        # preserved INSIDE. The R0 baseline is also passed as anchor so the
        # underlying attack goal does not drift across layers.
        if 2 in enabled_rounds and round1_trials:
            llm_mutators = [op for op in mutator.operators if isinstance(op, LLMMutator)]
            llm_mut = llm_mutators[0]

            for parent in round1_trials:
                technique = round2_rng.choice(techniques)
                base_id = parent["base_prompt_id"]
                base_text = base_prompts[base_id] if 0 <= base_id < len(base_prompts) else ""
                mutated = llm_mut.mutate_layered(
                    current_prompt=parent["prompt"],
                    new_technique=technique,
                    previous_technique=parent["technique_applied"] or "",
                    base_prompt=base_text,
                )
                self._record_trial(
                    round_id=2,
                    base_prompt_id=base_id,
                    parent_trial_id=parent["trial_id"],
                    prompt=mutated,
                    technique_applied=technique,
                    mutation_path=list(parent["mutation_path"]) + [technique],
                )

        # ---------------- Aggregate + write summary ----------------
        agg = aggregate_rounds(self.trials)
        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": self.config.get("experiment_name", "rounds_pipeline"),
            "description": self.config.get("description", ""),
            "strategy": "rounds",
            "num_base_prompts": len(base_prompts),
            "techniques": techniques,
            "enabled_rounds": sorted(enabled_rounds),
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            **agg,
        }
        self._write_summary(summary)
        return summary


# Backwards-compatible alias so existing imports keep working.
AgentRunner = RoundsRunner
