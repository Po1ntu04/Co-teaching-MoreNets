import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List


KEYS = [
    "source_size",
    "source_clean_rate",
    "source_positive_rate",
    "source_effective_size",
    "source_loss_mean",
    "source_loss_std",
    "source_confidence_mean",
    "oracle_mean",
    "oracle_positive_rate",
    "candidate_clean_rate",
    "auc_oracle_clean",
    "auc_proxy_adam_clean",
    "auc_loss_clean",
    "pearson_proxy_adam_oracle",
    "spearman_proxy_adam_oracle",
    "pearson_loss_oracle",
    "spearman_loss_oracle",
    "top25_oracle_mean_by_oracle",
    "top25_oracle_mean_by_proxy",
    "top25_oracle_mean_by_loss",
    "top25_oracle_lift_by_proxy",
    "top25_oracle_ratio_by_proxy",
    "top25_clean_rate_by_oracle",
    "top25_clean_rate_by_proxy",
    "top25_clean_rate_by_loss",
]


def finite_values(values: List[float]) -> List[float]:
    output = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(value)
    return output


def finite_mean(values: List[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def mean_label_hist(rows: List[Dict]) -> List[float]:
    hists = [r.get("source_label_hist") for r in rows if r.get("source_label_hist")]
    if not hists:
        return []
    width = max(len(hist) for hist in hists)
    totals = [0.0 for _ in range(width)]
    for hist in hists:
        for idx, value in enumerate(hist):
            totals[idx] += float(value)
    return [value / float(len(hists)) for value in totals]


def count_pass_epochs(rows: List[Dict]) -> Dict[str, int]:
    spearman_pass = 0
    lift_pass = 0
    for row in rows:
        spearman = row.get("spearman_proxy_adam_oracle")
        lift = row.get("top25_oracle_lift_by_proxy")
        ratio = row.get("top25_oracle_ratio_by_proxy")
        if spearman is not None and math.isfinite(float(spearman)) and float(spearman) >= 0.15:
            spearman_pass += 1
        if lift is not None and math.isfinite(float(lift)) and float(lift) > 0:
            lift_pass += 1
        elif ratio is not None and math.isfinite(float(ratio)) and float(ratio) >= 1.25:
            lift_pass += 1
    return {"spearman_pass_epochs": spearman_pass, "lift_pass_epochs": lift_pass}


def find_summary_files(paths: List[str]) -> List[str]:
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "**", "target_construction_summary.jsonl"), recursive=True))
        elif os.path.isfile(path):
            files.append(path)
    return sorted(set(files))


def load_records(files: List[str]) -> List[Dict]:
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_source_file"] = path
                records.append(record)
    return records


def summarize(records: List[Dict]) -> Dict:
    by_source = defaultdict(list)
    for record in records:
        by_source[record.get("source", "unknown")].append(record)

    output = {"num_records": len(records), "sources": {}}
    for source, rows in sorted(by_source.items()):
        available_rows = [r for r in rows if r.get("source_available", True) and int(r.get("num_records", 0)) > 0]
        summary = {
            "num_epochs": len(rows),
            "available_epochs": len(available_rows),
            "epochs": [r.get("epoch") for r in rows],
            "num_scored_samples": int(sum(int(r.get("num_records", 0)) for r in available_rows)),
        }
        for key in KEYS:
            summary[key] = finite_mean([r.get(key) for r in available_rows])
        summary["source_label_hist"] = mean_label_hist(available_rows)
        summary.update(count_pass_epochs(available_rows))
        summary["decision_hint"] = decision_hint(source, summary)
        output["sources"][source] = summary
    return output


def decision_hint(source: str, summary: Dict) -> str:
    if summary.get("available_epochs", 0) == 0:
        return "unavailable"
    if source == "clean_val":
        return "upper_bound_sanity_check"
    spearman = summary.get("spearman_proxy_adam_oracle", float("nan"))
    pass_epochs = summary.get("spearman_pass_epochs", 0)
    lift_epochs = summary.get("lift_pass_epochs", 0)
    if math.isfinite(float(spearman)) and float(spearman) >= 0.15 and pass_epochs >= 2:
        return "candidate_for_algorithmic_probe"
    if lift_epochs >= 2:
        return "weak_candidate_by_topk_lift"
    return "do_not_algorithmize_yet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="summary files or directories")
    parser.add_argument("--output", default="", help="optional JSON output path")
    args = parser.parse_args()

    files = find_summary_files(args.paths)
    records = load_records(files)
    summary = summarize(records)
    summary["files"] = files

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
