"""
Step D — Run baselines before the agent exists.

- Static baseline: hand-crafted list (fixed_baseline_prompts.jsonl), run each prompt once.
- Random mutations baseline: mutate seeds without selection; run N trials.

Usage:
  python -m src.experiments.run_baselines [--static] [--random] [--random-trials N] [--seed S]
  Or from project root: python src/experiments/run_baselines.py
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Project root = parent of src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _setup_paths():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def load_policy(base_path: Path) -> dict:
    p = base_path / "configs" / "policy.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def load_canaries(base_path: Path) -> list:
    p = base_path / "data" / "canaries.json"
    with open(p) as f:
        return json.load(f).get("canary_tokens", [])


def build_target_app(base_path: Path):
    """Build RAGApp + LLM from config (same as test_app)."""
    _setup_paths()
    from src.targetapp.rag_app import RAGApp
    from src.targetapp.local_chat_model import LocalChatModel, ModelConfig

    with open(base_path / "configs" / "targetapp.yaml") as f:
        config = yaml.safe_load(f)
    canaries = load_canaries(base_path)
    llm_config = ModelConfig(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )
    llm = LocalChatModel(base_url=config["llm"].get("base_url"), config=llm_config, timeout=600)
    doc_store = base_path / "data" / "document_store"
    app = RAGApp(
        document_store_path=str(doc_store),
        llm=llm,
        system_prompt=config["system_prompt"],
        canary_tokens=canaries,
        top_k=config["rag"]["top_k"],
    )
    return app


def build_mutator_llm(base_path: Path):
    """Build a LocalChatModel used by the LLMMutator (same backend as TargetApp)."""
    _setup_paths()
    from src.targetapp.local_chat_model import LocalChatModel, ModelConfig

    with open(base_path / "configs" / "targetapp.yaml") as f:
        config = yaml.safe_load(f)
    llm_config = ModelConfig(
        model=config["llm"]["model"],
        temperature=config["llm"].get("temperature", 0.7),
        max_tokens=config["llm"].get("max_tokens", 300),
    )
    return LocalChatModel(base_url=config["llm"].get("base_url"), config=llm_config, timeout=600)


def build_oracle(base_path: Path):
    _setup_paths()
    from src.evaluation.oracle import Oracle
    policy = load_policy(base_path)
    # Inject canaries so oracle doesn't need to read file
    policy.setdefault("protected_canaries", {})["tokens"] = load_canaries(base_path)
    return Oracle(base_path=base_path), policy


def _result_row(prompt: str, verdict) -> dict:
    """Shared minimal result row: only ASR-relevant fields."""
    details = getattr(verdict, "details", None) or {}
    return {
        "prompt": prompt[:80],
        "success": bool(verdict.success),
        "canaries_found": list(details.get("canaries_found") or []) if isinstance(details, dict) else [],
    }


def run_static_baseline(target_app, oracle, policy, base_path: Path):
    """Run static baseline: each hand-crafted prompt once."""
    from src.agent.prompt_datasets import list_fixed_baseline_prompts, FIXED_BASELINE_PROMPTS_FILE

    static_path = base_path / FIXED_BASELINE_PROMPTS_FILE
    prompts = list_fixed_baseline_prompts(static_path)
    results = []
    for prompt in prompts:
        result = target_app.run(prompt)
        verdict = oracle.evaluate(result, policy)
        results.append(_result_row(prompt, verdict))
    return results


def run_random_mutations_baseline(target_app, oracle, policy, base_path: Path, mutator_llm, num_trials: int, seed: int):
    """Run random mutations: num_trials prompts from mutated seeds (no selection)."""
    from src.agent.prompt_datasets import build_random_mutation_baseline, STARTING_PROMPTS_FILE

    prompts_path = base_path / STARTING_PROMPTS_FILE
    prompts = build_random_mutation_baseline(
        prompts_path,
        llm=mutator_llm,
        num_trials=num_trials,
        num_mutations_per_trial=1,
        seed=seed,
    )
    results = []
    for prompt in prompts:
        result = target_app.run(prompt)
        verdict = oracle.evaluate(result, policy)
        results.append(_result_row(prompt, verdict))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run static and/or random-mutations baselines")
    parser.add_argument("--static", action="store_true", help="Run static baseline")
    parser.add_argument("--random", action="store_true", help="Run random-mutations baseline")
    parser.add_argument("--random-trials", type=int, default=20, help="Number of random-mutation trials (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mutations")
    parser.add_argument("--out-dir", type=str, default=None, help="Write results to this dir (default: print only)")
    args = parser.parse_args()

    if not args.static and not args.random:
        args.static = True
        args.random = True

    base_path = PROJECT_ROOT
    print("Building TargetApp and Oracle...")
    target_app = build_target_app(base_path)
    oracle, policy = build_oracle(base_path)
    print("OK.\n")

    mutator_llm = None
    if args.random:
        print("Building mutator LLM (LLMMutator backend)...")
        mutator_llm = build_mutator_llm(base_path)
        print("OK.\n")

    all_results = {}

    if args.static:
        print("Running static baseline...")
        static_results = run_static_baseline(target_app, oracle, policy, base_path)
        successes = sum(1 for r in static_results if r["success"])
        total = len(static_results)
        asr = (100.0 * successes / total) if total else 0
        all_results["static"] = {"trials": total, "successes": successes, "asr": asr, "results": static_results}
        print(f"  Static: {successes}/{total} successes, ASR = {asr:.1f}%\n")

    if args.random:
        print(f"Running random-mutations baseline ({args.random_trials} trials, seed={args.seed})...")
        rand_results = run_random_mutations_baseline(target_app, oracle, policy, base_path, mutator_llm, args.random_trials, args.seed)
        successes = sum(1 for r in rand_results if r["success"])
        total = len(rand_results)
        asr = (100.0 * successes / total) if total else 0
        all_results["random_mutations"] = {"trials": total, "successes": successes, "asr": asr, "results": rand_results}
        print(f"  Random mutations: {successes}/{total} successes, ASR = {asr:.1f}%\n")

    # Summary
    print("Summary:")
    for name, data in all_results.items():
        print(f"  {name}: ASR = {data['asr']:.1f}% ({data['successes']}/{data['trials']})")

    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
