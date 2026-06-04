"""
Smoke test for run_experiment (end-to-end 3-round mutation pipeline).

This test invokes the real run_experiment.py via subprocess, capped at a
small number of base prompts so it finishes in a reasonable time, and then
validates the output artifacts and their schema.

Requires Ollama to be running and the configured model to be installed.
For an offline (no-Ollama) verification of the same pipeline, use
test_rounds_runner.py instead.

Usage:
  python test_experiment.py
  python test_experiment.py --max-base-prompts 1 --run-id quick_smoke
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _validate_outputs(run_dir: Path, num_base_prompts: int) -> int:
    """Validate the schema of trials.jsonl and summary.json. Returns 0 / 1."""
    trials_path = run_dir / "trials.jsonl"
    summary_path = run_dir / "summary.json"

    missing = [str(p) for p in (trials_path, summary_path) if not p.exists()]
    if missing:
        print("\nFAIL: expected output files were not generated:")
        for m in missing:
            print(f" - {m}")
        return 1

    # ---- trials.jsonl: schema and counts ----
    rows = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    expected_total = num_base_prompts + 3 * num_base_prompts + 3 * num_base_prompts
    if len(rows) != expected_total:
        print(
            f"\nFAIL: trials.jsonl has {len(rows)} rows, expected "
            f"{expected_total} (N + 3N + 3N where N={num_base_prompts})"
        )
        return 1

    required_fields = {
        "trial_id", "round", "base_prompt_id", "parent_trial_id",
        "prompt", "technique_applied", "mutation_path",
        "success", "canaries_found",
    }
    for i, row in enumerate(rows):
        missing_fields = required_fields - row.keys()
        if missing_fields:
            print(f"\nFAIL: row {i} missing fields: {sorted(missing_fields)}")
            return 1
        if row["round"] not in (0, 1, 2):
            print(f"\nFAIL: row {i} has invalid round={row['round']}")
            return 1

    # Round-0 rows must have technique_applied=None and empty mutation_path.
    r0 = [r for r in rows if r["round"] == 0]
    if len(r0) != num_base_prompts:
        print(f"\nFAIL: expected {num_base_prompts} Round-0 rows, got {len(r0)}")
        return 1
    for r in r0:
        if r["technique_applied"] is not None or r["mutation_path"]:
            print(f"\nFAIL: Round-0 row has non-null technique or non-empty path: {r}")
            return 1

    # Round-1 rows: 3 per base prompt, mutation_path length 1.
    r1 = [r for r in rows if r["round"] == 1]
    if len(r1) != 3 * num_base_prompts:
        print(f"\nFAIL: expected {3 * num_base_prompts} Round-1 rows, got {len(r1)}")
        return 1
    for r in r1:
        if len(r["mutation_path"]) != 1:
            print(f"\nFAIL: Round-1 row mutation_path length != 1: {r}")
            return 1

    # Round-2 rows: 3 per base prompt, mutation_path length 2.
    r2 = [r for r in rows if r["round"] == 2]
    if len(r2) != 3 * num_base_prompts:
        print(f"\nFAIL: expected {3 * num_base_prompts} Round-2 rows, got {len(r2)}")
        return 1
    for r in r2:
        if len(r["mutation_path"]) != 2:
            print(f"\nFAIL: Round-2 row mutation_path length != 2: {r}")
            return 1

    # ---- summary.json: required aggregation sections ----
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for key in ("per_round", "per_technique_round1", "per_combination_round2",
                "total_trials", "num_base_prompts", "techniques"):
        if key not in summary:
            print(f"\nFAIL: summary.json missing key '{key}'")
            return 1

    if summary["total_trials"] != expected_total:
        print(f"\nFAIL: summary.total_trials = {summary['total_trials']}, expected {expected_total}")
        return 1
    if summary["per_round"].get("1", {}).get("trials") != 3 * num_base_prompts:
        print(f"\nFAIL: summary.per_round['1']['trials'] mismatch")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for run_experiment (3-round pipeline)")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--run-id", type=str, default="smoke_experiment")
    parser.add_argument(
        "--max-base-prompts",
        type=int,
        default=5,
        help="Cap number of base prompts to keep this smoke test fast (default 5 -> 35 total trials)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    runner = root / "src" / "experiments" / "run_experiment.py"

    cmd = [
        sys.executable,
        str(runner),
        "--config",
        args.config,
        "--run-id",
        args.run_id,
        "--max-base-prompts",
        str(args.max_base_prompts),
    ]

    print("Running experiment smoke test:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        print(f"\nFAIL: run_experiment exited with code {result.returncode}")
        return result.returncode

    run_dir = root / "results" / "runs" / args.run_id
    rc = _validate_outputs(run_dir, args.max_base_prompts)
    if rc != 0:
        return rc

    print(f"\nPASS: experiment runner worked. Output dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
