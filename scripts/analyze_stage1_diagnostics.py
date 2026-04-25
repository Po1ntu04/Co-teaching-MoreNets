#!/usr/bin/env python3
"""Analyze stage-1 target-utility diagnostic outputs.

The script is intentionally read-only: it parses local copies of remote
diagnostic JSON/JSONL files and emits a compact audit report.
"""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_training_log(run_dir: Path) -> Optional[Dict]:
    candidates = list(run_dir.glob("**/*_training_log.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    with candidates[0].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_values(rows: Iterable[Dict], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def summarize_training(log: Optional[Dict]) -> Dict:
    if not log or not log.get("epochs"):
        return {"best_test_acc": None, "best_epoch": None, "last_test_acc": None}
    epochs = log["epochs"]
    best = max(epochs, key=lambda row: row.get("test_acc", float("-inf")))
    last = epochs[-1]
    return {
        "best_test_acc": best.get("test_acc"),
        "best_epoch": best.get("epoch"),
        "last_test_acc": last.get("test_acc"),
        "last_epoch": last.get("epoch"),
        "last_q_mean": last.get("q_mean"),
        "last_q_std": last.get("q_std"),
        "last_pi_t": last.get("pi_t"),
    }


def classify_run(rows: List[Dict], training: Dict) -> Dict:
    clean_rows = [row for row in rows if row.get("target") == "clean"]
    noisy_rows = [row for row in rows if row.get("target") == "noisy"]
    clean_adam = finite_values(clean_rows, "auc_align_adam_clean")
    clean_raw = finite_values(clean_rows, "auc_align_raw_clean")
    noisy_adam = finite_values(noisy_rows, "auc_align_adam_clean")
    loss_auc = finite_values(clean_rows, "auc_loss_clean")
    selected_clean = finite_values(clean_rows, "selected_clean_rate")
    high_loss_high_align = finite_values(clean_rows, "high_loss_high_align_clean_rate")

    valid_clean_epochs = sum(1 for value in clean_adam if value >= 0.60)
    raw_valid_epochs = sum(1 for value in clean_raw if value >= 0.60)
    noisy_valid_epochs = sum(1 for value in noisy_adam if value >= 0.60)
    adam_beats_raw = (
        bool(clean_adam)
        and bool(clean_raw)
        and mean(clean_adam) > mean(clean_raw) + 0.03
    )

    diagnostics_ok = bool(clean_rows) and all(
        math.isfinite(float(row.get("auc_align_adam_clean", float("nan"))))
        for row in clean_rows
    )

    return {
        "diagnostics_ok": diagnostics_ok,
        "clean_epochs_ge_0_60": valid_clean_epochs,
        "raw_epochs_ge_0_60": raw_valid_epochs,
        "noisy_epochs_ge_0_60": noisy_valid_epochs,
        "adam_beats_raw_by_mean_gt_0_03": adam_beats_raw,
        "clean_auc_align_adam": summarize_values(clean_adam),
        "clean_auc_align_raw": summarize_values(clean_raw),
        "noisy_auc_align_adam": summarize_values(noisy_adam),
        "clean_auc_loss": summarize_values(loss_auc),
        "selected_clean_rate": summarize_values(selected_clean),
        "high_loss_high_align_clean_rate": summarize_values(high_loss_high_align),
        "training": training,
    }


def run_name_from_path(path: Path) -> str:
    name = path.name
    if name == "diag":
        return path.parent.name
    return name


def find_diag_dirs(root: Path) -> List[Path]:
    if root.is_file():
        return [root.parent]
    if (root / "alignment_summary.jsonl").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("alignment_summary.jsonl"))


def build_decision(run_summaries: Dict[str, Dict]) -> Dict:
    e1_runs = [summary for name, summary in run_summaries.items() if "stage1_e1" in name]
    e2_runs = [summary for name, summary in run_summaries.items() if "stage1_e2" in name]

    e1_support = sum(1 for run in e1_runs if run["clean_epochs_ge_0_60"] >= 2)
    e2_support = sum(1 for run in e2_runs if run["clean_epochs_ge_0_60"] >= 2)
    adam_over_raw = any(run["adam_beats_raw_by_mean_gt_0_03"] for run in run_summaries.values())

    if e1_support >= 2 or e2_support >= 2:
        conclusion = "support_optimizer_aware_utility_followup"
    elif len(e1_runs) >= 3 and e1_support == 0 and e2_support == 0:
        conclusion = "weak_alignment_signal_return_to_reliability_repair"
    else:
        conclusion = "insufficient_runs_continue_stage1"

    return {
        "e1_seed_runs_with_clean_adam_signal": e1_support,
        "e2_seed_runs_with_clean_adam_signal": e2_support,
        "any_adam_mean_beats_raw": adam_over_raw,
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="local result root or a diag directory")
    parser.add_argument("--output", default="", help="optional JSON report path")
    args = parser.parse_args()

    root = Path(args.root)
    diag_dirs = find_diag_dirs(root)
    run_summaries: Dict[str, Dict] = {}

    for diag_dir in diag_dirs:
        rows = read_jsonl(diag_dir / "alignment_summary.jsonl")
        run_dir = diag_dir.parent if diag_dir.name == "diag" else diag_dir
        training = summarize_training(read_training_log(run_dir))
        run_summaries[run_name_from_path(diag_dir)] = classify_run(rows, training)

    report = {
        "root": str(root),
        "num_runs": len(run_summaries),
        "runs": run_summaries,
        "decision": build_decision(run_summaries),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
