"""
Offline verifier for the 3-round mutation pipeline.

What this script checks (no Ollama, no network):

  1.  aggregate_rounds([]) returns a sane zero-trial summary.
  2.  aggregate_rounds(handcrafted_trials) returns the exact per-round,
      per-technique, and per-combination ASR we expect.
  3.  RoundsRunner with N=2 base prompts writes exactly 14 trials
      (2 R0 + 6 R1 + 6 R2).
  4.  Round-0 rows: round==0, technique_applied is None, mutation_path==[],
      parent_trial_id is None.
  5.  Round-1 rows: round==1, each technique in {logic_traps,
      roleplay_social, encoding} appears EXACTLY 2 times (one per base
      prompt), mutation_path has length 1.
  6.  Round-2 rows: round==2, mutation_path has length 2,
      mutation_path[0] == parent's mutation_path[0].
  7.  summary.json contains per_round/per_technique_round1/per_combination_round2
      and per_round["1"]["trials"] == 6.
  8.  Reproducibility: same seed yields the same Round-2 technique sequence.
  9.  ASR math: a FakeOracle that marks specific prompts as success produces
      aggregate ASR matching a hand-computed value.

Run from project root:

  python test_rounds_runner.py

Exit code is 0 only if every check passes; each check prints a single
PASS/FAIL line so failures are pinpointable.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------- fakes


class FakeLLM:
    """Stand-in for LocalChatModel used by the LLMMutator.

    Detects the active technique via the explicit [[TECHNIQUE:<name>]] marker
    that LLMMutator prepends to every meta-prompt. This is robust to layered
    prompts (Round 2) where multiple technique names appear in the text -- the
    marker always reflects the NEW (outer) technique being applied.
    """
    _MARKER_RE = re.compile(r"\[\[TECHNIQUE:\s*([a-z_]+)\s*\]\]", re.IGNORECASE)

    def generate(self, meta_prompt, **kwargs):
        m = self._MARKER_RE.search(meta_prompt)
        if m:
            technique = m.group(1).lower()
        else:
            lowered = meta_prompt.lower()
            if "logic trap" in lowered:
                technique = "logic_traps"
            elif "roleplay" in lowered or "social engineering" in lowered:
                technique = "roleplay_social"
            elif "encoding" in lowered:
                technique = "encoding"
            else:
                technique = "unknown"
        return {"response": f"[T:{technique}] mutated"}


class _FakeResult:
    def __init__(self, output_text="ok"):
        self.output_text = output_text


class FakeTargetApp:
    """Records every call; returns a deterministic fake result."""
    def __init__(self):
        self.calls = []

    def run(self, prompt: str):
        self.calls.append(prompt)
        return _FakeResult(output_text=f"echo:{prompt[:30]}")


class FakeVerdict:
    """Minimal Verdict-shaped object: only `success` matters for ASR."""
    def __init__(self, success=False, canaries_found=None):
        self.success = success
        self.details = {"canaries_found": list(canaries_found or [])}


class FakeOracle:
    """Always returns no-success (no canary leaked)."""
    def evaluate(self, result, policy):
        return FakeVerdict()


class TaggingOracle:
    """
    Marks success when the output text contains a "[T:<tech>]" tag that
    matches one of the techniques in `success_techniques`. Lets us test
    that aggregations correctly attribute successes to the right buckets.
    """
    def __init__(self, success_techniques):
        self.success_techniques = set(success_techniques)

    def evaluate(self, result, policy):
        text = getattr(result, "output_text", "") or ""
        for t in self.success_techniques:
            if f"[T:{t}]" in text:
                return FakeVerdict(success=True, canaries_found=[f"FAKE_CANARY_{t}"])
        return FakeVerdict()


# ------------------------------------------------------------- test scaffolding


PASS = 0
FAIL = 0


def _check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"PASS  {name}")
        PASS += 1
    except AssertionError as e:
        print(f"FAIL  {name}  --  {e}")
        FAIL += 1
    except Exception as e:
        print(f"ERROR {name}  --  {e.__class__.__name__}: {e}")
        traceback.print_exc()
        FAIL += 1


def _make_runner(run_id: str, *, target_app=None, oracle=None, base_prompts=None, seed: int = 42):
    """
    Build a RoundsRunner with all I/O subclassed so it never touches Ollama
    or the filesystem-defined baseline list. The fake mutator is wired in
    by overriding `_build_mutator`.
    """
    from src.agent.runner import RoundsRunner
    from src.agent.mutators import Mutator

    base_prompts = base_prompts if base_prompts is not None else ["prompt-A", "prompt-B"]
    target_app = target_app or FakeTargetApp()
    oracle = oracle or FakeOracle()

    class _Stubbed(RoundsRunner):
        def _build_mutator(self, seed_value):
            import random as _random
            llm = FakeLLM()
            return Mutator(llm=llm, rng=_random.Random(seed_value)), llm

        def _load_baseline_prompts(self, _base_path, _cfg, _override):
            return list(base_prompts)

    cfg = {
        "experiment_name": "offline_test",
        "reproducibility": {"seed": seed, "save_config": False},
        "rounds": {
            "enabled": [0, 1, 2],
            "techniques": ["logic_traps", "roleplay_social", "encoding"],
        },
    }
    return _Stubbed(
        config=cfg,
        base_path=PROJECT_ROOT,
        target_app=target_app,
        oracle=oracle,
        policy={},
        run_id=run_id,
    )


def _cleanup_run(run_id: str):
    run_dir = PROJECT_ROOT / "results" / "runs" / run_id
    if run_dir.exists():
        for p in run_dir.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        try:
            run_dir.rmdir()
        except OSError:
            pass


def _read_trials(run_id: str) -> List[Dict[str, Any]]:
    p = PROJECT_ROOT / "results" / "runs" / run_id / "trials.jsonl"
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_summary(run_id: str) -> Dict[str, Any]:
    p = PROJECT_ROOT / "results" / "runs" / run_id / "summary.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------- checks


def check_aggregate_empty():
    from src.evaluation.metrics import aggregate_rounds
    agg = aggregate_rounds([])
    assert agg["total_trials"] == 0
    assert agg["total_successes"] == 0
    for r in ("0", "1", "2"):
        assert agg["per_round"][r] == {"trials": 0, "successes": 0, "asr": 0.0}
    assert agg["per_technique_round1"] == {}
    assert agg["per_combination_round2"] == {}


def check_aggregate_handcrafted():
    """
    R0: 2 trials (1 success). R1: 6 trials, success on logic_traps only (2/6).
    R2: 4 trials with paths logic_traps -> {encoding x2, roleplay_social x2},
        success on both 'logic_traps -> encoding' rows.
    """
    from src.evaluation.metrics import aggregate_rounds

    trials = [
        {"round": 0, "technique_applied": None, "mutation_path": [], "success": True},
        {"round": 0, "technique_applied": None, "mutation_path": [], "success": False},

        {"round": 1, "technique_applied": "logic_traps", "mutation_path": ["logic_traps"], "success": True},
        {"round": 1, "technique_applied": "logic_traps", "mutation_path": ["logic_traps"], "success": True},
        {"round": 1, "technique_applied": "roleplay_social", "mutation_path": ["roleplay_social"], "success": False},
        {"round": 1, "technique_applied": "roleplay_social", "mutation_path": ["roleplay_social"], "success": False},
        {"round": 1, "technique_applied": "encoding", "mutation_path": ["encoding"], "success": False},
        {"round": 1, "technique_applied": "encoding", "mutation_path": ["encoding"], "success": False},

        {"round": 2, "technique_applied": "encoding", "mutation_path": ["logic_traps", "encoding"], "success": True},
        {"round": 2, "technique_applied": "encoding", "mutation_path": ["logic_traps", "encoding"], "success": True},
        {"round": 2, "technique_applied": "roleplay_social", "mutation_path": ["logic_traps", "roleplay_social"], "success": False},
        {"round": 2, "technique_applied": "roleplay_social", "mutation_path": ["logic_traps", "roleplay_social"], "success": False},
    ]
    agg = aggregate_rounds(trials)

    assert agg["per_round"]["0"] == {"trials": 2, "successes": 1, "asr": 50.0}
    assert agg["per_round"]["1"] == {"trials": 6, "successes": 2, "asr": round(100 * 2 / 6, 2)}
    assert agg["per_round"]["2"] == {"trials": 4, "successes": 2, "asr": 50.0}

    pt = agg["per_technique_round1"]
    assert pt["logic_traps"] == {"trials": 2, "successes": 2, "asr": 100.0}
    assert pt["roleplay_social"] == {"trials": 2, "successes": 0, "asr": 0.0}
    assert pt["encoding"] == {"trials": 2, "successes": 0, "asr": 0.0}

    pc = agg["per_combination_round2"]
    assert pc["logic_traps -> encoding"] == {"trials": 2, "successes": 2, "asr": 100.0}
    assert pc["logic_traps -> roleplay_social"] == {"trials": 2, "successes": 0, "asr": 0.0}


def check_runner_total_trials():
    run_id = "offline_total_trials"
    _cleanup_run(run_id)
    try:
        runner = _make_runner(run_id)
        summary = runner.run()
        rows = _read_trials(run_id)
        assert len(rows) == 14, f"Expected 14 trial rows (2 + 6 + 6), got {len(rows)}"
        assert summary["total_trials"] == 14
        assert summary["num_base_prompts"] == 2
    finally:
        _cleanup_run(run_id)


def check_round0_shape():
    run_id = "offline_round0_shape"
    _cleanup_run(run_id)
    try:
        runner = _make_runner(run_id)
        runner.run()
        rows = _read_trials(run_id)
        r0 = [r for r in rows if r["round"] == 0]
        assert len(r0) == 2
        for r in r0:
            assert r["technique_applied"] is None, r
            assert r["mutation_path"] == [], r
            assert r["parent_trial_id"] is None, r
    finally:
        _cleanup_run(run_id)


def check_round1_technique_balance():
    run_id = "offline_round1_balance"
    _cleanup_run(run_id)
    try:
        runner = _make_runner(run_id)
        runner.run()
        rows = _read_trials(run_id)
        r1 = [r for r in rows if r["round"] == 1]
        assert len(r1) == 6, f"Expected 6 Round-1 trials, got {len(r1)}"

        counts = {}
        for r in r1:
            t = r["technique_applied"]
            counts[t] = counts.get(t, 0) + 1
            assert len(r["mutation_path"]) == 1, r
            assert r["mutation_path"][0] == t, r
            assert r["parent_trial_id"] is not None, r
        for tech in ("logic_traps", "roleplay_social", "encoding"):
            assert counts.get(tech) == 2, f"Technique {tech} count = {counts.get(tech)} (expected 2)"
    finally:
        _cleanup_run(run_id)


def check_round2_path_extends_parent():
    run_id = "offline_round2_paths"
    _cleanup_run(run_id)
    try:
        runner = _make_runner(run_id)
        runner.run()
        rows = _read_trials(run_id)
        by_id = {r["trial_id"]: r for r in rows}
        r2 = [r for r in rows if r["round"] == 2]
        assert len(r2) == 6, f"Expected 6 Round-2 trials, got {len(r2)}"
        for r in r2:
            assert len(r["mutation_path"]) == 2, r
            parent = by_id[r["parent_trial_id"]]
            assert parent["round"] == 1
            assert r["mutation_path"][0] == parent["mutation_path"][0], r
            assert r["mutation_path"][1] == r["technique_applied"], r
    finally:
        _cleanup_run(run_id)


def check_summary_keys():
    run_id = "offline_summary_keys"
    _cleanup_run(run_id)
    try:
        runner = _make_runner(run_id)
        runner.run()
        summary = _read_summary(run_id)
        for k in ("per_round", "per_technique_round1", "per_combination_round2",
                  "total_trials", "total_successes", "num_base_prompts", "techniques", "seed"):
            assert k in summary, f"summary.json is missing key '{k}'"
        assert summary["per_round"]["1"]["trials"] == 6
    finally:
        _cleanup_run(run_id)


def check_reproducibility():
    """Same seed must give the same Round-2 technique sequence."""
    sequences = []
    for tag in ("repro_a", "repro_b"):
        run_id = f"offline_{tag}"
        _cleanup_run(run_id)
        try:
            runner = _make_runner(run_id, seed=12345)
            runner.run()
            rows = _read_trials(run_id)
            r2_techs = [r["technique_applied"] for r in rows if r["round"] == 2]
            sequences.append(r2_techs)
        finally:
            _cleanup_run(run_id)
    assert sequences[0] == sequences[1], (
        f"Round-2 technique sequence not reproducible:\n  run1={sequences[0]}\n  run2={sequences[1]}"
    )


def check_asr_math_with_tagging_oracle():
    """
    With the TaggingOracle marking only 'encoding'-tagged outputs as success,
    Round 1 ASR for 'encoding' should be 100% (2/2) and others 0%.
    """
    run_id = "offline_asr_math"
    _cleanup_run(run_id)
    try:
        runner = _make_runner(run_id, oracle=TaggingOracle(success_techniques=["encoding"]))
        summary = runner.run()
        pt = summary["per_technique_round1"]
        assert pt["encoding"] == {"trials": 2, "successes": 2, "asr": 100.0}, pt["encoding"]
        assert pt["logic_traps"]["successes"] == 0
        assert pt["roleplay_social"]["successes"] == 0
        # Round-0 success count must be 0 (no technique tag yet).
        assert summary["per_round"]["0"]["successes"] == 0
    finally:
        _cleanup_run(run_id)


# ---------------------------------------------------------------------- main


def main() -> int:
    print("=" * 70)
    print("  Rounds pipeline verifier (no Ollama required)")
    print("=" * 70)

    checks = [
        ("aggregate_rounds([]) gives a sane zero summary",         check_aggregate_empty),
        ("aggregate_rounds(handcrafted) matches expected ASRs",    check_aggregate_handcrafted),
        ("RoundsRunner produces exactly N + 3N + 3N trials",       check_runner_total_trials),
        ("Round-0 rows: technique=None, path=[], no parent",       check_round0_shape),
        ("Round-1 rows: balanced 2 per technique, path=[T]",       check_round1_technique_balance),
        ("Round-2 rows: path extends parent's Round-1 path",       check_round2_path_extends_parent),
        ("summary.json has per_round/per_technique/per_combo",     check_summary_keys),
        ("Same seed -> same Round-2 technique sequence",           check_reproducibility),
        ("ASR math: TaggingOracle attributes successes correctly", check_asr_math_with_tagging_oracle),
    ]
    for name, fn in checks:
        _check(name, fn)

    print("-" * 70)
    print(f"  Total: {PASS + FAIL}   PASS: {PASS}   FAIL: {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
