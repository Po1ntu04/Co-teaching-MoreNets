import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List


KEYS = [
    "oracle_mean",
    "oracle_positive_rate",
    "oracle_clean_mean",
    "oracle_noisy_mean",
    "auc_oracle_clean",
    "auc_loss_clean",
    "auc_sam_utility_clean",
    "auc_align_adam_clean",
    "spearman_loss_oracle",
    "spearman_sam_utility_oracle",
    "spearman_align_adam_oracle",
    "top25_oracle_mean_by_loss",
    "top25_oracle_mean_by_sam_utility",
    "top25_oracle_mean_by_align_adam",
    "top25_clean_rate_by_oracle",
    "top25_clean_rate_by_loss",
    "top25_clean_rate_by_sam_utility",
    "top25_clean_rate_by_align_adam",
]


def finite_mean(values: List[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def find_summary_files(paths: List[str]) -> List[str]:
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "**", "oracle_summary.jsonl"), recursive=True))
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
                record["_source"] = path
                records.append(record)
    return records


def summarize(records: List[Dict]) -> Dict:
    by_target = defaultdict(list)
    for record in records:
        by_target[record.get("target", "unknown")].append(record)

    output = {"num_records": len(records), "targets": {}}
    for target, rows in sorted(by_target.items()):
        target_summary = {
            "num_epochs": len(rows),
            "epochs": [r.get("epoch") for r in rows],
            "num_scored_samples": int(sum(int(r.get("num_records", 0)) for r in rows)),
        }
        for key in KEYS:
            target_summary[key] = finite_mean([r.get(key) for r in rows])
        output["targets"][target] = target_summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="oracle_summary.jsonl files or directories")
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
