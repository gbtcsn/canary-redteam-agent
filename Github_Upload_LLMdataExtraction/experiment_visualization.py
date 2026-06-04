"""
Visualize results for the 3-round mutation pipeline.

Inputs (auto-discovered):
  - results/runs/<run_id>/summary.json
        Produced by RoundsRunner. Contains:
          total_trials, total_successes,
          per_round            -> {"0": {trials,successes,asr}, "1": ..., "2": ...},
          per_technique_round1 -> {tech: {trials,successes,asr}},
          per_combination_round2 -> {"t1 -> t2": {trials,successes,asr}}
  - results/runs/baselines_*.json (optional, via --baseline-json)
        Produced by run_baselines.py. Each top-level key is a strategy with
        {trials, successes, asr, results: [{"prompt","success","canaries_found"}, ...]}

Outputs (in --output-dir, auto-stamped if not provided):
  - summary_table.csv / summary_table.md
        One row per source (run or baseline) with overall + per-round ASR.
  - summary_table_average.csv / summary_table_average.md
        Trial-weighted pool across all pipeline runs (ASR = pooled successes / trials).
  - summary_average_breakdown.csv
        Pooled per-round, per-technique (R1), and per-combination (R2) stats.
  - asr_overall.png
        Top-level ASR comparison (runs + baselines side-by-side).
  - asr_per_round.png
        ASR per round (R0/R1/R2) per run -- does mutation help?
  - asr_per_technique_r1.png
        ASR per Round-1 technique per run -- which technique works best?
  - asr_per_combination_r2.png
        ASR per Round-2 combination per run -- which sequence works best?

Usage:
  python experiment_visualization.py
  python experiment_visualization.py --runs-root results/runs
  python experiment_visualization.py --baseline-json results/runs/baselines_<ts>.json
  python experiment_visualization.py --output-dir results/analysis/thesis_figures
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_run_summaries(runs_root: Path) -> List[Dict[str, Any]]:
    """Scan runs_root/*/summary.json and return them as a list of dicts.

    Each entry is the raw summary dict augmented with a "_run_id" key
    (= folder name) so downstream code can use it as a label.
    """
    summaries: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return summaries
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        try:
            data = _read_json(summary_path)
        except Exception as e:
            print(f"  WARN: could not parse {summary_path}: {e}")
            continue
        data["_run_id"] = summary_path.parent.name
        summaries.append(data)
    return summaries


def load_baselines_json(path: Path) -> List[Dict[str, Any]]:
    """Convert a baselines JSON (from run_baselines.py) into row dicts.

    The baseline file has no per-round / per-technique breakdown, so we just
    emit overall ASR rows that show up alongside the pipeline runs in the
    "overall" comparison.
    """
    if not path.exists():
        return []
    raw = _read_json(path)
    rows: List[Dict[str, Any]] = []
    for strategy, block in raw.items():
        trials = int(block.get("trials", 0))
        successes = int(block.get("successes", 0))
        asr = float(block.get("asr", (100.0 * successes / trials) if trials else 0.0))
        rows.append({
            "label": f"baseline:{strategy}",
            "source": "baseline",
            "trials": trials,
            "successes": successes,
            "asr": asr,
        })
    return rows


# ---------------------------------------------------------------------------
# Reshape summaries into tidy DataFrames for plotting
# ---------------------------------------------------------------------------

def _bucket_asr(bucket: Optional[Dict[str, Any]]) -> float:
    if not bucket:
        return 0.0
    return float(bucket.get("asr", 0.0))


def _bucket_trials(bucket: Optional[Dict[str, Any]]) -> int:
    if not bucket:
        return 0
    return int(bucket.get("trials", 0))


def _best_key(d: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Return the key with the highest ASR (None if d is empty)."""
    if not d:
        return None
    return max(d.items(), key=lambda kv: float(kv[1].get("asr", 0.0)))[0]


def build_run_summary_df(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per pipeline run with the headline numbers."""
    rows = []
    for s in summaries:
        per_round = s.get("per_round") or {}
        per_tech = s.get("per_technique_round1") or {}
        per_combo = s.get("per_combination_round2") or {}
        total_trials = int(s.get("total_trials", 0))
        total_successes = int(s.get("total_successes", 0))
        overall_asr = (100.0 * total_successes / total_trials) if total_trials else 0.0
        rows.append({
            "label": s.get("_run_id", "unknown"),
            "source": "run",
            "trials": total_trials,
            "successes": total_successes,
            "asr": round(overall_asr, 2),
            "r0_asr": round(_bucket_asr(per_round.get("0")), 2),
            "r1_asr": round(_bucket_asr(per_round.get("1")), 2),
            "r2_asr": round(_bucket_asr(per_round.get("2")), 2),
            "best_r1_technique": _best_key(per_tech) or "-",
            "best_r2_combination": _best_key(per_combo) or "-",
        })
    return pd.DataFrame(rows)


def build_per_round_long_df(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Tidy (run_id, round, asr, trials) for grouped bar plot."""
    rows = []
    for s in summaries:
        per_round = s.get("per_round") or {}
        for r in ("0", "1", "2"):
            b = per_round.get(r) or {}
            rows.append({
                "run_id": s.get("_run_id", "unknown"),
                "round": f"R{r}",
                "asr": round(_bucket_asr(b), 2),
                "trials": _bucket_trials(b),
            })
    return pd.DataFrame(rows)


def build_per_technique_long_df(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Tidy (run_id, technique, asr, trials) for Round-1 grouped bar plot."""
    rows = []
    for s in summaries:
        per_tech = s.get("per_technique_round1") or {}
        for tech, b in per_tech.items():
            rows.append({
                "run_id": s.get("_run_id", "unknown"),
                "technique": tech,
                "asr": round(_bucket_asr(b), 2),
                "trials": _bucket_trials(b),
            })
    return pd.DataFrame(rows)


def build_per_combination_long_df(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Tidy (run_id, combination, asr, trials) for Round-2 grouped bar plot."""
    rows = []
    for s in summaries:
        per_combo = s.get("per_combination_round2") or {}
        for combo, b in per_combo.items():
            rows.append({
                "run_id": s.get("_run_id", "unknown"),
                "combination": combo,
                "asr": round(_bucket_asr(b), 2),
                "trials": _bucket_trials(b),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_summary_tables(
    runs_df: pd.DataFrame,
    baseline_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write summary_table.csv and summary_table.md with runs + baselines."""
    if not runs_df.empty:
        runs_df = runs_df.sort_values("asr", ascending=False).reset_index(drop=True)

    baseline_df = pd.DataFrame(baseline_rows)
    if not baseline_df.empty:
        for col in ("r0_asr", "r1_asr", "r2_asr", "best_r1_technique", "best_r2_combination"):
            baseline_df[col] = "-"
        baseline_df["asr"] = baseline_df["asr"].round(2)

    columns = [
        "label", "source", "trials", "successes", "asr",
        "r0_asr", "r1_asr", "r2_asr",
        "best_r1_technique", "best_r2_combination",
    ]
    parts = [df for df in (runs_df, baseline_df) if not df.empty]
    if not parts:
        return
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.reindex(columns=columns)
    combined = combined.sort_values(["source", "asr"], ascending=[True, False]).reset_index(drop=True)

    out_csv = output_dir / "summary_table.csv"
    combined.to_csv(out_csv, index=False)

    out_md = output_dir / "summary_table.md"
    headers = list(combined.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in combined.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Trial-weighted averages across runs
# ---------------------------------------------------------------------------

def _asr_pct(successes: int, trials: int) -> float:
    return round((100.0 * successes / trials), 2) if trials else 0.0


def _merge_bucket(
    acc: Dict[str, Dict[str, int]],
    key: str,
    trials: int,
    successes: int,
) -> None:
    if key not in acc:
        acc[key] = {"trials": 0, "successes": 0}
    acc[key]["trials"] += trials
    acc[key]["successes"] += successes


def aggregate_summaries_weighted(
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pool trials and successes across runs; ASR is recomputed from counts."""
    per_round: Dict[str, Dict[str, int]] = {}
    per_technique: Dict[str, Dict[str, int]] = {}
    per_combination: Dict[str, Dict[str, int]] = {}
    total_trials = 0
    total_successes = 0

    for summary in summaries:
        total_trials += int(summary.get("total_trials", 0))
        total_successes += int(summary.get("total_successes", 0))
        for round_id, bucket in (summary.get("per_round") or {}).items():
            _merge_bucket(
                per_round,
                str(round_id),
                int(bucket.get("trials", 0)),
                int(bucket.get("successes", 0)),
            )
        for technique, bucket in (summary.get("per_technique_round1") or {}).items():
            _merge_bucket(
                per_technique,
                technique,
                int(bucket.get("trials", 0)),
                int(bucket.get("successes", 0)),
            )
        for combination, bucket in (summary.get("per_combination_round2") or {}).items():
            _merge_bucket(
                per_combination,
                combination,
                int(bucket.get("trials", 0)),
                int(bucket.get("successes", 0)),
            )

    return {
        "total_trials": total_trials,
        "total_successes": total_successes,
        "per_round": per_round,
        "per_technique_round1": per_technique,
        "per_combination_round2": per_combination,
    }


def _best_key_from_counts(buckets: Dict[str, Dict[str, int]]) -> str:
    if not buckets:
        return "-"
    return max(
        buckets.items(),
        key=lambda item: _asr_pct(item[1]["successes"], item[1]["trials"]),
    )[0]


def build_weighted_average_summary_row(
    summaries: List[Dict[str, Any]],
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """One summary_table-style row: trial-weighted across all runs."""
    pooled = aggregate_summaries_weighted(summaries)
    per_round = pooled["per_round"]
    num_runs = len(summaries)
    row_label = label or f"weighted_avg_{num_runs}_runs"

    def _round_asr(round_id: str) -> float:
        bucket = per_round.get(round_id, {"trials": 0, "successes": 0})
        return _asr_pct(bucket["successes"], bucket["trials"])

    return {
        "label": row_label,
        "source": "aggregate",
        "num_runs": num_runs,
        "trials": pooled["total_trials"],
        "successes": pooled["total_successes"],
        "asr": _asr_pct(pooled["total_successes"], pooled["total_trials"]),
        "r0_asr": _round_asr("0"),
        "r1_asr": _round_asr("1"),
        "r2_asr": _round_asr("2"),
        "best_r1_technique": _best_key_from_counts(pooled["per_technique_round1"]),
        "best_r2_combination": _best_key_from_counts(pooled["per_combination_round2"]),
    }


def build_weighted_average_breakdown_df(
    summaries: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Long table of pooled buckets: round, technique, or combination."""
    pooled = aggregate_summaries_weighted(summaries)
    rows: List[Dict[str, Any]] = []

    for round_id, bucket in sorted(pooled["per_round"].items(), key=lambda x: x[0]):
        rows.append({
            "metric_type": "per_round",
            "key": f"R{round_id}",
            "trials": bucket["trials"],
            "successes": bucket["successes"],
            "asr": _asr_pct(bucket["successes"], bucket["trials"]),
        })
    for technique, bucket in sorted(pooled["per_technique_round1"].items()):
        rows.append({
            "metric_type": "per_technique_round1",
            "key": technique,
            "trials": bucket["trials"],
            "successes": bucket["successes"],
            "asr": _asr_pct(bucket["successes"], bucket["trials"]),
        })
    for combination, bucket in sorted(pooled["per_combination_round2"].items()):
        rows.append({
            "metric_type": "per_combination_round2",
            "key": combination,
            "trials": bucket["trials"],
            "successes": bucket["successes"],
            "asr": _asr_pct(bucket["successes"], bucket["trials"]),
        })

    return pd.DataFrame(rows)


def load_summaries_for_summary_table(
    summary_csv: Path,
    runs_root: Path,
) -> List[Dict[str, Any]]:
    """Load summary.json for each pipeline run listed in summary_table.csv."""
    table = pd.read_csv(summary_csv)
    if table.empty:
        return []
    run_rows = table[table["source"] == "run"]
    summaries: List[Dict[str, Any]] = []
    for label in run_rows["label"].astype(str):
        summary_path = runs_root / label / "summary.json"
        if not summary_path.exists():
            print(f"  WARN: missing {summary_path}, skipping run {label}")
            continue
        data = _read_json(summary_path)
        data["_run_id"] = label
        summaries.append(data)
    return summaries


def save_weighted_average_tables(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
    label: Optional[str] = None,
) -> None:
    """Write trial-weighted average tables next to summary_table.csv."""
    if not summaries:
        print("  WARN: no run summaries to average; skipping weighted average tables.")
        return

    avg_row = build_weighted_average_summary_row(summaries, label=label)
    headline_cols = [
        "label", "source", "num_runs", "trials", "successes", "asr",
        "r0_asr", "r1_asr", "r2_asr",
        "best_r1_technique", "best_r2_combination",
    ]
    headline_df = pd.DataFrame([avg_row]).reindex(columns=headline_cols)
    breakdown_df = build_weighted_average_breakdown_df(summaries)

    csv_path = output_dir / "summary_table_average.csv"
    headline_df.to_csv(csv_path, index=False)

    md_path = output_dir / "summary_table_average.md"
    headers = list(headline_df.columns)
    lines = [
        "# Trial-weighted average across pipeline runs",
        "",
        "ASR columns are **pooled** (total successes / total trials), not the",
        "unweighted mean of per-run ASR percentages.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in headline_df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    breakdown_path = output_dir / "summary_average_breakdown.csv"
    breakdown_df.to_csv(breakdown_path, index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _annotate_bars(ax, fmt: str = "{:.1f}%") -> None:
    """Print numeric values on top of each bar."""
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h):
            continue
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points",
        )


def plot_overall_asr(
    runs_df: pd.DataFrame,
    baseline_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Top-level ASR comparison across runs (and baselines if available)."""
    parts: List[pd.DataFrame] = []
    if not runs_df.empty:
        parts.append(runs_df[["label", "source", "asr"]])
    if baseline_rows:
        parts.append(pd.DataFrame(baseline_rows)[["label", "source", "asr"]])
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True).sort_values("asr", ascending=False)

    plt.figure(figsize=(max(8, 0.6 * len(df) + 4), 5))
    ax = sns.barplot(data=df, x="label", y="asr", hue="source", dodge=False)
    ax.set_title("Overall Attack Success Rate (ASR)")
    ax.set_xlabel("")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, max(100.0, float(df["asr"].max()) + 10))
    plt.xticks(rotation=35, ha="right")
    _annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(output_dir / "asr_overall.png", dpi=220)
    plt.close()


def plot_asr_per_round(per_round_df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bars: x = round (R0/R1/R2), hue = run_id."""
    if per_round_df.empty:
        return
    plt.figure(figsize=(max(7, 0.7 * per_round_df["run_id"].nunique() + 5), 5))
    ax = sns.barplot(data=per_round_df, x="round", y="asr", hue="run_id")
    ax.set_title("ASR per Round (does mutation help?)")
    ax.set_xlabel("Round")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, max(100.0, float(per_round_df["asr"].max()) + 10))
    _annotate_bars(ax)
    plt.legend(title="run", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "asr_per_round.png", dpi=220)
    plt.close()


def plot_asr_per_technique_r1(per_tech_df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bars: x = technique (Round 1), hue = run_id."""
    if per_tech_df.empty:
        return
    plt.figure(figsize=(max(7, 0.7 * per_tech_df["run_id"].nunique() + 5), 5))
    ax = sns.barplot(data=per_tech_df, x="technique", y="asr", hue="run_id")
    ax.set_title("ASR per Mutation Technique (Round 1)")
    ax.set_xlabel("Technique")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, max(100.0, float(per_tech_df["asr"].max()) + 10))
    plt.xticks(rotation=20, ha="right")
    _annotate_bars(ax)
    plt.legend(title="run", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "asr_per_technique_r1.png", dpi=220)
    plt.close()


def plot_asr_per_combination_r2(per_combo_df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bars: x = combination (Round 2), hue = run_id."""
    if per_combo_df.empty:
        return
    n_combos = per_combo_df["combination"].nunique()
    plt.figure(figsize=(max(9, 0.8 * n_combos + 3), 5))
    ax = sns.barplot(data=per_combo_df, x="combination", y="asr", hue="run_id")
    ax.set_title("ASR per Technique Combination (Round 2)")
    ax.set_xlabel("Combination (first -> second)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, max(100.0, float(per_combo_df["asr"].max()) + 10))
    plt.xticks(rotation=30, ha="right")
    _annotate_bars(ax)
    plt.legend(title="run", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "asr_per_combination_r2.png", dpi=220)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tables and plots for the 3-round mutation pipeline."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="results/runs",
        help="Directory containing run folders with summary.json.",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default="",
        help="Optional baselines JSON (from run_baselines.py) to overlay in the overall plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for tables/plots. If empty, auto-generated.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help=(
            "Optional path to an existing summary_table.csv. When set with "
            "--averages-only, write weighted-average tables without regenerating plots."
        ),
    )
    parser.add_argument(
        "--averages-only",
        action="store_true",
        help="Only write trial-weighted average tables (requires --summary-csv or run summaries).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent

    runs_root = (root / args.runs_root).resolve()
    baseline_json = (root / args.baseline_json).resolve() if args.baseline_json else None

    if args.output_dir:
        output_dir = (root / args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (root / "results" / "analysis" / f"experiment_visualization_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (root / args.summary_csv).resolve() if args.summary_csv else None
    if summary_csv and summary_csv.exists():
        summaries = load_summaries_for_summary_table(summary_csv, runs_root)
        print(f"Loaded {len(summaries)} run summary file(s) from: {summary_csv}")
    else:
        summaries = load_run_summaries(runs_root)
        print(f"Loaded {len(summaries)} run summary file(s) from: {runs_root}")

    baseline_rows = load_baselines_json(baseline_json) if baseline_json else []
    if baseline_json:
        if baseline_rows:
            print(f"Loaded {len(baseline_rows)} baseline strateg(y/ies) from: {baseline_json}")
        else:
            print(f"WARN: no baseline rows could be loaded from: {baseline_json}")

    if not summaries and not baseline_rows:
        print("No data available. Nothing to visualize.")
        return 1

    save_weighted_average_tables(summaries, output_dir)

    if args.averages_only:
        print(f"\nWeighted-average tables written to: {output_dir}")
        for file in sorted(output_dir.glob("summary_*average*")):
            print(f" - {file.name}")
        return 0

    sns.set_theme(style="whitegrid")

    runs_df = build_run_summary_df(summaries)
    per_round_df = build_per_round_long_df(summaries)
    per_tech_df = build_per_technique_long_df(summaries)
    per_combo_df = build_per_combination_long_df(summaries)

    save_summary_tables(runs_df, baseline_rows, output_dir)
    plot_overall_asr(runs_df, baseline_rows, output_dir)
    plot_asr_per_round(per_round_df, output_dir)
    plot_asr_per_technique_r1(per_tech_df, output_dir)
    plot_asr_per_combination_r2(per_combo_df, output_dir)

    print(f"\nArtifacts written to: {output_dir}")
    for file in sorted(output_dir.glob("*")):
        print(f" - {file.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
