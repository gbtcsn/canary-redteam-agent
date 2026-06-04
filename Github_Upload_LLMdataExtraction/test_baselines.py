"""
Smoke test for run_baselines.

Purpose:
- Validate that the baselines runner executes without import/runtime errors.
- Validate that it writes a result JSON file when --out-dir is provided.

The evolutionary --agent mode has been removed from run_baselines.py.
The agent is now driven via run_experiment.py (3-round pipeline).

Usage examples:
  python test_baselines.py
  python test_baselines.py --random-trials 5 --seed 42
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for run_baselines")
    parser.add_argument("--random-trials", type=int, default=3, help="Small trial count for quick validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    runner = root / "src" / "experiments" / "run_baselines.py"
    out_dir = root / "results" / "runs"

    cmd = [
        sys.executable,
        str(runner),
        "--static",
        "--random",
        "--random-trials",
        str(args.random_trials),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(out_dir),
    ]

    print("Running baseline smoke test:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        print(f"\nFAIL: run_baselines exited with code {result.returncode}")
        return result.returncode

    generated = sorted(out_dir.glob("baselines_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not generated:
        print("\nFAIL: no baselines_*.json was generated in results/runs")
        return 1

    print(f"\nPASS: baseline runner worked. Latest output: {generated[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
