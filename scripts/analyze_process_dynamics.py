from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List


def finite_float(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f if math.isfinite(value_f) else float("nan")


def finite_mean(values: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_logs(root: str) -> List[str]:
    paths: List[str] = []
    for pattern in (
        os.path.join(root, "**", "*_training_log.json"),
        os.path.join(root, "**", "training_log.json"),
    ):
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def summarize_process(path: str, last_k: int) -> Dict[str, Any]:
    data = load_json(path)
    config = data.get("config", {})
    epochs = data.get("epochs", [])
    tail = epochs[-last_k:] if epochs else []
    first_epoch = epochs[0] if epochs else {}
    last_epoch = epochs[-1] if epochs else {}
    best_epoch = max(epochs, key=lambda row: finite_float(row.get("test_acc")), default={})

    def proc(row: Dict[str, Any], key: str) -> float:
        return finite_float(row.get("process_summary", {}).get(key))

    row: Dict[str, Any] = {
        "path": path,
        "seed": config.get("seed"),
        "num_models": config.get("num_models"),
        "q_mode": config.get("q_mode"),
        "q_usage_mode": config.get("q_usage_mode", "standard"),
        "mstep_mode": config.get("mstep_mode"),
        "q_gate_pool_mult": config.get("q_gate_pool_mult"),
        "sam_rho": config.get("sam_rho"),
        "epochs_completed": len(epochs),
        "best_epoch": best_epoch.get("epoch"),
        "best_acc": best_epoch.get("test_acc"),
        "last_acc": last_epoch.get("test_acc"),
        "last5_acc": finite_mean(e.get("test_acc") for e in tail),
        "epoch_q_mean_delta": finite_float(last_epoch.get("q_mean")) - finite_float(first_epoch.get("q_mean")),
        "epoch_q_std_delta": finite_float(last_epoch.get("q_std")) - finite_float(first_epoch.get("q_std")),
        "epoch_q_auc_delta": finite_float(last_epoch.get("q_clean_auc")) - finite_float(first_epoch.get("q_clean_auc")),
        "epoch_overlap_delta": finite_float(last_epoch.get("overlap")) - finite_float(first_epoch.get("overlap")),
        "last5_q_mean": finite_mean(e.get("q_mean") for e in tail),
        "last5_q_std": finite_mean(e.get("q_std") for e in tail),
        "last5_q_auc": finite_mean(e.get("q_clean_auc") for e in tail),
        "last5_selected_clean_rate": finite_mean(e.get("selected_clean_rate") for e in tail),
        "last5_overlap": finite_mean(e.get("overlap") for e in tail),
        "last5_grad_norm": finite_mean(e.get("grad_norm") for e in tail),
        "last5_update_norm": finite_mean(e.get("update_norm") for e in tail),
        "last5_update_to_param": finite_mean(e.get("update_to_param") for e in tail),
        "last5_within_q_mean_delta": finite_mean(proc(e, "q_mean_delta") for e in tail),
        "last5_within_q_std_delta": finite_mean(proc(e, "q_std_delta") for e in tail),
        "last5_within_overlap_delta": finite_mean(proc(e, "overlap_delta") for e in tail),
        "last5_within_selected_clean_delta": finite_mean(proc(e, "selected_clean_rate_delta") for e in tail),
        "last5_within_grad_norm_mean": finite_mean(proc(e, "grad_norm_mean_mean") for e in tail),
        "last5_within_update_to_param_mean": finite_mean(proc(e, "update_to_param_mean_mean") for e in tail),
        "last5_selected_q_gap": finite_mean(
            proc(e, "selected_q_mean_mean") - proc(e, "unselected_q_mean_mean") for e in tail
        ),
        "last5_selected_loss_gap": finite_mean(
            proc(e, "selected_loss_mean_mean") - proc(e, "unselected_loss_mean_mean") for e in tail
        ),
        "last5_gate_pool_frac": finite_mean(proc(e, "gate_pool_frac_mean") for e in tail),
        "last5_gate_pool_clean_rate": finite_mean(proc(e, "gate_pool_clean_rate_mean") for e in tail),
    }
    return row


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
    parser.add_argument("root", help="root directory containing process-enabled training logs")
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = [summarize_process(path, args.last_k) for path in find_logs(args.root)]
    rows = [row for row in rows if row.get("epochs_completed", 0)]
    rows.sort(key=lambda row: (str(row.get("q_usage_mode")), str(row.get("q_gate_pool_mult")), str(row.get("seed"))))
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.out:
        write_csv(rows, args.out)


if __name__ == "__main__":
    main()
