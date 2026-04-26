# -*- coding: utf-8 -*-
"""Summarize benchmark accuracy from this repository's JSON training logs."""

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def summarize_log(path: str, last_k: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    epochs = payload.get("epochs", [])
    if not epochs:
        return {
            "path": path,
            "error": "no epochs in log",
        }

    accs = finite([e.get("test_acc") for e in epochs])
    if not accs:
        return {
            "path": path,
            "error": "no finite test_acc values",
        }

    best_epoch = max(epochs, key=lambda e: float(e.get("test_acc", float("-inf"))))
    tail = accs[-last_k:]
    last10_mean_per_model = None
    per_model_tail = [e.get("test_accs_per_model") for e in epochs[-last_k:]]
    if per_model_tail and all(isinstance(row, list) for row in per_model_tail):
        width = min(len(row) for row in per_model_tail)
        if width > 0:
            last10_mean_per_model = [
                sum(float(row[i]) for row in per_model_tail) / len(per_model_tail)
                for i in range(width)
            ]

    return {
        "path": path,
        "config": payload.get("config", {}),
        "num_epochs_logged": len(epochs),
        "best_test_acc": float(best_epoch.get("test_acc")),
        "best_epoch": int(best_epoch.get("epoch")),
        "last_test_acc": float(accs[-1]),
        "last_epoch": int(epochs[-1].get("epoch")),
        "last_k": len(tail),
        "last_k_mean_test_acc": sum(tail) / len(tail),
        "last_k_min_test_acc": min(tail),
        "last_k_max_test_acc": max(tail),
        "last_k_mean_per_model": last10_mean_per_model,
    }


def find_logs(paths: List[str]) -> List[str]:
    logs: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in files:
                    if name.endswith("_training_log.json"):
                        logs.append(os.path.join(root, name))
        elif os.path.isfile(path):
            logs.append(path)
    return sorted(set(logs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="training log JSON files or result directories")
    parser.add_argument("--last-k", type=int, default=10, help="number of final logged epochs to average")
    parser.add_argument("--output", type=str, default="", help="optional output JSON path")
    args = parser.parse_args()

    summaries = [summarize_log(path, args.last_k) for path in find_logs(args.paths)]
    result = {"runs": summaries}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
