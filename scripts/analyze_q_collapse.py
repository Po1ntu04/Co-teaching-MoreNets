from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List


def finite_mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else float("nan")


def load_training_log(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_log(path: str, last_k: int) -> Dict[str, Any]:
    data = load_training_log(path)
    config = data.get("config", {})
    epochs = data.get("epochs", [])
    tail = epochs[-last_k:] if epochs else []
    best_epoch = max(epochs, key=lambda row: float(row.get("test_acc", float("-inf"))), default={})
    last_epoch = epochs[-1] if epochs else {}

    return {
        "path": path,
        "run": os.path.basename(os.path.dirname(path)),
        "seed": config.get("seed"),
        "dataset": config.get("dataset"),
        "noise_type": config.get("noise_type"),
        "noise_rate": config.get("noise_rate"),
        "num_models": config.get("num_models"),
        "q_mode": config.get("q_mode"),
        "q_usage_mode": config.get("q_usage_mode", "standard"),
        "mstep_mode": config.get("mstep_mode"),
        "sam_rho": config.get("sam_rho"),
        "replay_size": config.get("replay_size"),
        "epochs_completed": len(epochs),
        "best_epoch": best_epoch.get("epoch"),
        "best_acc": best_epoch.get("test_acc"),
        "last_epoch": last_epoch.get("epoch"),
        "last_acc": last_epoch.get("test_acc"),
        "last5_acc": finite_mean(row.get("test_acc") for row in tail),
        "last5_q_mean": finite_mean(row.get("q_mean") for row in tail),
        "last5_q_std": finite_mean(row.get("q_std") for row in tail),
        "last5_q_entropy": finite_mean(row.get("q_entropy") for row in tail),
        "last5_q_clean_auc": finite_mean(row.get("q_clean_auc") for row in tail),
        "last5_selected_clean_rate": finite_mean(row.get("selected_clean_rate") for row in tail),
        "last5_overlap": finite_mean(row.get("overlap") for row in tail),
        "last5_pi_t": finite_mean(row.get("pi_t") for row in tail),
    }


def find_logs(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "**", "*_training_log.json"),
        os.path.join(root, "**", "training_log.json"),
    ]
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="root directory containing training logs")
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    logs = find_logs(args.root)
    rows = [summarize_log(path, args.last_k) for path in logs]
    rows.sort(key=lambda row: (str(row.get("q_usage_mode")), str(row.get("num_models")), str(row.get("seed"))))

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.out:
        write_csv(rows, args.out)


if __name__ == "__main__":
    main()
