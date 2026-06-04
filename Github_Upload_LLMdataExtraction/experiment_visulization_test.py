"""
Smoke test for experiment_visualization.py.

We synthesize two fake pipeline runs (each with its own summary.json) plus a
baselines JSON, run the visualization script as a subprocess, and assert that
all expected tables/plots are produced.

The fake summaries match the schema written by RoundsRunner:
  total_trials, total_successes,
  per_round            -> {"0": {...}, "1": {...}, "2": {...}},
  per_technique_round1 -> {tech: {trials,successes,asr}},
  per_combination_round2 -> {"t1 -> t2": {trials,successes,asr}}

Usage:
  python experiment_visulization_test.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bucket(trials: int, successes: int) -> dict:
    asr = round((100.0 * successes / trials), 2) if trials > 0 else 0.0
    return {"trials": trials, "successes": successes, "asr": asr}


def _fake_summary(run_id: str, r0_n=4, r0_k=0, r1_per_tech=(1, 2, 0), r2_per_combo=None) -> dict:
    """Build a fake summary.json dict for a pipeline run.

    Args:
        r0_n, r0_k:         Round-0 trials and successes.
        r1_per_tech:        tuple (logic_traps_k, roleplay_social_k, encoding_k) of
                            successes for Round 1 (each technique runs `r0_n` trials).
        r2_per_combo:       optional dict {"t1 -> t2": successes}; defaults to all-zero
                            combinations except a couple to exercise the plot.
    """
    techniques = ["logic_traps", "roleplay_social", "encoding"]
    per_round = {
        "0": _bucket(r0_n, r0_k),
        "1": _bucket(r0_n * 3, sum(r1_per_tech)),
        "2": _bucket(r0_n * 3, 0),
    }
    per_technique_round1 = {
        tech: _bucket(r0_n, k) for tech, k in zip(techniques, r1_per_tech)
    }
    if r2_per_combo is None:
        r2_per_combo = {
            "logic_traps -> encoding": 1,
            "encoding -> roleplay_social": 0,
            "roleplay_social -> logic_traps": 1,
        }
    per_combination_round2 = {}
    r2_trials_per_combo = max(1, r0_n // 3 + 1)
    r2_total = 0
    for combo, k in r2_per_combo.items():
        per_combination_round2[combo] = _bucket(r2_trials_per_combo, k)
        r2_total += k
    per_round["2"] = _bucket(r0_n * 3, r2_total)

    total_trials = per_round["0"]["trials"] + per_round["1"]["trials"] + per_round["2"]["trials"]
    total_successes = per_round["0"]["successes"] + per_round["1"]["successes"] + per_round["2"]["successes"]

    return {
        "run_id": run_id,
        "experiment_name": "rounds_pipeline",
        "strategy": "rounds",
        "num_base_prompts": r0_n,
        "techniques": techniques,
        "enabled_rounds": [0, 1, 2],
        "seed": 42,
        "timestamp": "2026-05-15T12:00:00",
        "total_trials": total_trials,
        "total_successes": total_successes,
        "per_round": per_round,
        "per_technique_round1": per_technique_round1,
        "per_combination_round2": per_combination_round2,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    vis_script = root / "experiment_visualization.py"
    if not vis_script.exists():
        print(f"FAIL: missing script: {vis_script}")
        return 1

    test_root = root / "results" / "tmp_visualization_test"
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    runs_root = test_root / "runs"
    output_dir = test_root / "out"

    _write_json(
        runs_root / "run_alpha" / "summary.json",
        _fake_summary(
            "run_alpha",
            r0_n=4, r0_k=0,
            r1_per_tech=(1, 2, 0),
            r2_per_combo={
                "logic_traps -> encoding": 1,
                "encoding -> roleplay_social": 1,
                "roleplay_social -> logic_traps": 0,
            },
        ),
    )
    _write_json(
        runs_root / "run_beta" / "summary.json",
        _fake_summary(
            "run_beta",
            r0_n=4, r0_k=1,
            r1_per_tech=(0, 1, 1),
            r2_per_combo={
                "logic_traps -> encoding": 0,
                "encoding -> logic_traps": 2,
            },
        ),
    )

    baseline_json = test_root / "baselines_test.json"
    _write_json(
        baseline_json,
        {
            "static": {
                "trials": 4, "successes": 1, "asr": 25.0,
                "results": [
                    {"prompt": "p1", "success": False, "canaries_found": []},
                    {"prompt": "p2", "success": True,  "canaries_found": ["CANARY_X"]},
                ],
            },
            "random_mutations": {
                "trials": 5, "successes": 2, "asr": 40.0,
                "results": [
                    {"prompt": "p3", "success": False, "canaries_found": []},
                    {"prompt": "p4", "success": True,  "canaries_found": ["CANARY_Y"]},
                ],
            },
        },
    )

    cmd = [
        sys.executable,
        str(vis_script),
        "--runs-root", str(runs_root),
        "--baseline-json", str(baseline_json),
        "--output-dir", str(output_dir),
    ]
    print("Running visualization smoke test:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        print(f"\nFAIL: visualization script exited with code {result.returncode}")
        return result.returncode

    expected_files = [
        "summary_table.csv",
        "summary_table.md",
        "asr_overall.png",
        "asr_per_round.png",
        "asr_per_technique_r1.png",
        "asr_per_combination_r2.png",
    ]
    missing = [name for name in expected_files if not (output_dir / name).exists()]
    if missing:
        print("\nFAIL: expected files were not generated:")
        for name in missing:
            print(f" - {name}")
        return 1

    csv_text = (output_dir / "summary_table.csv").read_text(encoding="utf-8")
    for needle in ("run_alpha", "run_beta", "baseline:static", "baseline:random_mutations",
                   "r0_asr", "r1_asr", "r2_asr", "best_r1_technique", "best_r2_combination"):
        if needle not in csv_text:
            print(f"\nFAIL: summary_table.csv is missing expected token: {needle}")
            return 1

    print(f"\nPASS: visualization artifacts generated in: {output_dir}")
    print("Files:")
    for f in sorted(output_dir.glob("*")):
        print(f" - {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
