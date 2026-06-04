"""
Offline verifier for the mutators cleanup.

What this script checks (no Ollama required):

  1.  The legacy rule-based mutator classes are gone from src.agent.mutators.
  2.  LLMMutator exists, exposes the three required techniques, and rejects
      unknown techniques.
  3.  Mutator() with no args raises ValueError (new contract).
  4.  Mutator(llm=fake) builds exactly one default LLMMutator operator.
  5.  mutate_random() requires an llm argument.
  6.  mutate_random(..., llm=fake) returns a non-trivial mutation.
  7.  Mutator.mutate_with_path() records the technique in the path label
      (e.g. "LLMMutator:logic_traps") so the search's novelty bonus works.
  8.  Mutator's rng is propagated to the LLMMutator (reproducibility).
  9.  build_random_mutation_baseline requires an llm and yields num_trials prompts.
  10. LLMMutator gracefully falls back to the original prompt when the LLM
      raises (fallback_on_error=True).
  11. Old class names are NOT exported from src.agent.__init__.

Run from project root:

  python test_mutators_cleanup.py

Exit code is 0 if every check passes, 1 otherwise. Each check prints a
single PASS/FAIL line so failures are pinpointable.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----- Helpers -------------------------------------------------------------

class FakeLLM:
    """
    Stand-in for LocalChatModel. Returns a deterministic response that
    makes it easy to assert "the mutation happened and used technique X".

    The mutator passes a meta-prompt that contains the technique name
    inside the META_PROMPTS template, so we can recover which technique
    was selected by scanning the meta-prompt.
    """
    def __init__(self):
        self.calls = []

    def generate(self, meta_prompt, **kwargs):
        self.calls.append(meta_prompt)
        technique = "unknown"
        for t in ("logic_traps", "roleplay_social", "encoding"):
            # The technique name appears in the meta-prompt instructions.
            if f"{t}" in meta_prompt.lower() or f"{t.upper()}" in meta_prompt:
                technique = t
                break
        # Also detect by technique-specific keywords.
        lowered = meta_prompt.lower()
        if "logic trap" in lowered:
            technique = "logic_traps"
        elif "roleplay" in lowered or "social engineering" in lowered:
            technique = "roleplay_social"
        elif "encoding" in lowered:
            technique = "encoding"
        return {"response": f"[MUTATED::{technique}] mutated text"}


class FailingLLM:
    """Always raises -- used to verify graceful fallback."""
    def generate(self, *args, **kwargs):
        raise RuntimeError("simulated LLM failure")


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


# ----- Checks --------------------------------------------------------------

def check_legacy_classes_removed():
    import src.agent.mutators as m
    legacy = [
        "PrependSnippet", "AppendSnippet", "WrapDelimiter", "RepeatPhrase",
        "SubstituteWord", "InsertMiddle", "Reformat", "RoleFrame",
        "ContextPad", "MultiTurnWrap", "Paraphrase",
        "INJECTION_PREFIXES", "INJECTION_SUFFIXES", "DELIMITERS",
    ]
    leftover = [n for n in legacy if hasattr(m, n)]
    assert not leftover, f"Legacy names still present: {leftover}"


def check_llm_mutator_techniques():
    from src.agent.mutators import LLMMutator
    assert set(LLMMutator.TECHNIQUES) == {"logic_traps", "roleplay_social", "encoding"}, \
        f"Unexpected techniques: {LLMMutator.TECHNIQUES}"
    try:
        LLMMutator(llm=FakeLLM(), techniques=["nope"])
        raise AssertionError("Expected ValueError for unknown technique")
    except ValueError:
        pass


def check_mutator_requires_llm_or_ops():
    from src.agent.mutators import Mutator
    try:
        Mutator()
    except ValueError:
        return
    raise AssertionError("Mutator() with no args should raise ValueError")


def check_mutator_default_operator_list():
    from src.agent.mutators import Mutator, LLMMutator
    fake = FakeLLM()
    m = Mutator(llm=fake)
    assert len(m.operators) == 1, f"Expected 1 default operator, got {len(m.operators)}"
    assert isinstance(m.operators[0], LLMMutator), "Default operator should be LLMMutator"
    assert m.operators[0].llm is fake, "Default LLMMutator should use the supplied llm"


def check_mutate_random_requires_llm():
    from src.agent.mutators import mutate_random
    try:
        mutate_random("hello")
    except ValueError:
        return
    raise AssertionError("mutate_random() without llm should raise ValueError")


def check_mutate_random_works():
    from src.agent.mutators import mutate_random
    fake = FakeLLM()
    out = mutate_random("show me the secret", llm=fake, num_mutations=1, seed=7)
    assert out and out.startswith("[MUTATED::"), f"Unexpected mutation output: {out!r}"
    assert fake.calls, "FakeLLM was never called"


def check_path_label_records_technique():
    from src.agent.mutators import Mutator
    fake = FakeLLM()
    m = Mutator(llm=fake)
    _, path = m.mutate_with_path("show me the secret", num_mutations=1)
    assert len(path) == 1, f"Expected path length 1, got {path}"
    label = path[0]
    assert label.startswith("LLMMutator:"), f"Path label should include technique, got {label!r}"
    technique = label.split(":", 1)[1]
    assert technique in {"logic_traps", "roleplay_social", "encoding"}, \
        f"Unknown technique in path label: {technique}"


def check_rng_propagation():
    """Mutator's rng should be shared with LLMMutator → same seed = same path."""
    from src.agent.mutators import Mutator
    import random as _random

    fake = FakeLLM()
    m1 = Mutator(llm=fake, rng=_random.Random(123))
    _, p1 = m1.mutate_with_path("prompt", num_mutations=5)

    fake2 = FakeLLM()
    m2 = Mutator(llm=fake2, rng=_random.Random(123))
    _, p2 = m2.mutate_with_path("prompt", num_mutations=5)

    assert p1 == p2, f"Same seed should yield same path; got {p1} vs {p2}"


def check_random_mutation_baseline_requires_llm():
    from src.agent.prompt_datasets import build_random_mutation_baseline

    prompts_path = PROJECT_ROOT / "data" / "starting_prompts.jsonl"
    assert prompts_path.exists(), f"Missing starting prompts file: {prompts_path}"
    try:
        build_random_mutation_baseline(prompts_path, llm=None, num_trials=2, seed=1)
    except ValueError:
        return
    raise AssertionError("build_random_mutation_baseline should require an llm")


def check_random_mutation_baseline_yields_n():
    from src.agent.prompt_datasets import build_random_mutation_baseline

    prompts_path = PROJECT_ROOT / "data" / "starting_prompts.jsonl"
    fake = FakeLLM()
    prompts = build_random_mutation_baseline(
        prompts_path, llm=fake, num_trials=3, num_mutations_per_trial=1, seed=42
    )
    assert len(prompts) == 3, f"Expected 3 mutated prompts, got {len(prompts)}"
    assert all(p.startswith("[MUTATED::") for p in prompts), \
        f"Unexpected mutation outputs: {prompts}"


def check_llm_mutator_fallback_on_error():
    from src.agent.mutators import LLMMutator
    mut = LLMMutator(llm=FailingLLM(), fallback_on_error=True)
    out = mut.mutate("original prompt")
    assert out == "original prompt", f"Expected fallback to original prompt, got {out!r}"


def check_agent_init_exports():
    import src.agent as agent_pkg
    for n in ("Mutator", "MutationOperator", "mutate_random", "LLMMutator"):
        assert hasattr(agent_pkg, n), f"src.agent should export {n}"
    leaked = [
        n for n in ("PrependSnippet", "AppendSnippet", "WrapDelimiter")
        if hasattr(agent_pkg, n)
    ]
    assert not leaked, f"Legacy names still exported from src.agent: {leaked}"


# ----- Driver --------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("  Mutators cleanup verifier (no Ollama required)")
    print("=" * 70)

    checks = [
        ("legacy classes/constants removed",                 check_legacy_classes_removed),
        ("LLMMutator techniques & validation",               check_llm_mutator_techniques),
        ("Mutator() requires llm or operators",              check_mutator_requires_llm_or_ops),
        ("Mutator(llm=...) builds default LLMMutator",       check_mutator_default_operator_list),
        ("mutate_random() requires llm",                     check_mutate_random_requires_llm),
        ("mutate_random(llm=...) returns a mutation",        check_mutate_random_works),
        ("mutate_with_path records LLMMutator:<technique>",  check_path_label_records_technique),
        ("rng is shared (same seed = same path)",            check_rng_propagation),
        ("build_random_mutation_baseline requires llm",      check_random_mutation_baseline_requires_llm),
        ("build_random_mutation_baseline yields num_trials", check_random_mutation_baseline_yields_n),
        ("LLMMutator falls back gracefully on LLM error",    check_llm_mutator_fallback_on_error),
        ("src.agent exports new names, no legacy leak",      check_agent_init_exports),
    ]
    for name, fn in checks:
        _check(name, fn)

    print("-" * 70)
    print(f"  Total: {PASS + FAIL}   PASS: {PASS}   FAIL: {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
