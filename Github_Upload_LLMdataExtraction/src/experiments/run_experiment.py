"""
Experiment Runner - 3-round mutation pipeline, end-to-end.

Given a config and the baseline prompt set, runs:
  Round 0: baseline pass against the TargetApp (no mutation).
  Round 1: fork each prompt with each of the 3 mutator techniques.
  Round 2: apply one random technique on top of each Round-1 result.

Logs every call to results/runs/<run_id>/trials.jsonl and writes a summary.

Usage:
    python run_experiment.py --config configs/experiment.yaml
    python run_experiment.py --max-base-prompts 2     # fast smoke run
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_paths():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_policy(base_path: Path) -> dict:
    with open(base_path / "configs" / "policy.yaml") as f:
        return yaml.safe_load(f)


def load_canaries(base_path: Path) -> list:
    with open(base_path / "data" / "canaries.json") as f:
        return json.load(f).get("canary_tokens", [])


def build_target_app(base_path: Path):
    _setup_paths()
    from src.targetapp.rag_app import RAGApp
    from src.targetapp.local_chat_model import LocalChatModel, ModelConfig

    with open(base_path / "configs" / "targetapp.yaml") as f:
        cfg = yaml.safe_load(f)
    canaries = load_canaries(base_path)
    llm_config = ModelConfig(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    llm = LocalChatModel(base_url=cfg["llm"].get("base_url"), config=llm_config, timeout=600)
    app = RAGApp(
        document_store_path=str(base_path / "data" / "document_store"),
        llm=llm,
        system_prompt=cfg["system_prompt"],
        canary_tokens=canaries,
        top_k=cfg["rag"]["top_k"],
    )
    return app


def build_oracle(base_path: Path):
    _setup_paths()
    from src.evaluation.oracle import Oracle
    policy = load_policy(base_path)
    policy.setdefault("protected_canaries", {})["tokens"] = load_canaries(base_path)
    return Oracle(base_path=base_path), policy


def _print_round_table(per_round: dict) -> None:
    print("  Per-round ASR:")
    for r in ("0", "1", "2"):
        bucket = per_round.get(r, {})
        trials = bucket.get("trials", 0)
        successes = bucket.get("successes", 0)
        asr = bucket.get("asr", 0.0)
        print(f"    Round {r}: {successes}/{trials}  ASR = {asr:.2f}%")


def _print_technique_table(per_technique_round1: dict) -> None:
    if not per_technique_round1:
        return
    print("  Per-technique ASR (Round 1):")
    for tech in sorted(per_technique_round1.keys()):
        bucket = per_technique_round1[tech]
        trials = bucket.get("trials", 0)
        successes = bucket.get("successes", 0)
        asr = bucket.get("asr", 0.0)
        print(f"    {tech:20s}  {successes}/{trials}  ASR = {asr:.2f}%")


def _print_combination_table(per_combination_round2: dict) -> None:
    if not per_combination_round2:
        return
    print("  Per-combination ASR (Round 2, first -> second technique):")
    for combo in sorted(per_combination_round2.keys()):
        bucket = per_combination_round2[combo]
        trials = bucket.get("trials", 0)
        successes = bucket.get("successes", 0)
        asr = bucket.get("asr", 0.0)
        print(f"    {combo:40s}  {successes}/{trials}  ASR = {asr:.2f}%")


def run_experiment(
    config_path: Path,
    base_path: Path,
    run_id: str = None,
    max_base_prompts: int = None,
):
    config = load_config(config_path)
    run_id = run_id or config.get("experiment_id") or datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Building TargetApp and Oracle...")
    target_app = build_target_app(base_path)
    oracle, policy = build_oracle(base_path)
    print("OK.\n")

    print(f"Running 3-round mutation pipeline (run_id={run_id})...")
    from src.agent.runner import RoundsRunner
    runner = RoundsRunner(
        config=config,
        base_path=base_path,
        target_app=target_app,
        oracle=oracle,
        policy=policy,
        run_id=run_id,
    )
    summary = runner.run(max_base_prompts=max_base_prompts)

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Run ID:            {summary['run_id']}")
    print(f"  Base prompts:      {summary['num_base_prompts']}")
    print(f"  Total trials:      {summary['total_trials']}")
    print(f"  Total successes:   {summary['total_successes']}")
    print(f"  Output:            results/runs/{run_id}/")
    print()
    _print_round_table(summary.get("per_round", {}))
    print()
    _print_technique_table(summary.get("per_technique_round1", {}))
    print()
    _print_combination_table(summary.get("per_combination_round2", {}))
    print("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the 3-round mutation pipeline end-to-end")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--run-id", type=str, default=None, help="Override run ID")
    parser.add_argument(
        "--max-base-prompts",
        type=int,
        default=None,
        help="Cap the number of base baseline prompts (handy for quick smoke runs)",
    )
    args = parser.parse_args()

    base_path = PROJECT_ROOT
    config_path = base_path / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    run_experiment(
        config_path=config_path,
        base_path=base_path,
        run_id=args.run_id,
        max_base_prompts=args.max_base_prompts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
