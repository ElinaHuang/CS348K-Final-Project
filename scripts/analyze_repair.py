from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

from utils import read_csv, safe_div, write_csv

LABELS = {"pass", "fail", "ambiguous"}

TARGET_RESULT_FIELDS = [
    "repair_id", "target_constraint_id", "target_constraint_type",
    "before_label", "after_label", "target_fixed",
]
IMAGE_RESULT_FIELDS = [
    "repair_id", "source_image_id", "source_prompt_id", "repair_strategy",
    "before_image_label", "after_image_label", "image_fixed",
    "num_target_constraints", "num_targets_fixed", "num_regressions", "regressed_constraint_ids",
]
TYPE_FIELDS = [
    "target_constraint_type", "num_targets", "target_fixed_count", "target_fixed_rate",
    "num_repairs_with_regression", "avg_regression_count",
]
SUMMARY_FIELDS = ["metric", "value"]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    return label if label in LABELS else "ambiguous"


def aggregate_labels(labels: List[str]) -> str:
    labels = [normalize_label(x) for x in labels if (x or "").strip()]
    if not labels:
        return "ambiguous"
    if any(x == "fail" for x in labels):
        return "fail"
    if all(x == "pass" for x in labels):
        return "pass"
    return "ambiguous"


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def analyze_repair(
    repaired_prompts: List[Dict[str, str]],
    repair_targets: List[Dict[str, str]],
    repaired_labels: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    repair_by_id = {r["repair_id"]: r for r in repaired_prompts}
    labels_by_repair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in repaired_labels:
        if row.get("after_label", "").strip():
            labels_by_repair[row["repair_id"]].append(row)

    after_by_key = {(r["repair_id"], r["constraint_id"]): normalize_label(r.get("after_label", "")) for r in repaired_labels}

    target_results: List[Dict[str, str]] = []
    for target in repair_targets:
        repair_id = target["repair_id"]
        cid = target["target_constraint_id"]
        before = normalize_label(target.get("before_label", ""))
        after = after_by_key.get((repair_id, cid), "ambiguous")
        target_fixed = before in {"fail", "ambiguous"} and after == "pass"
        target_results.append({
            "repair_id": repair_id,
            "target_constraint_id": cid,
            "target_constraint_type": target["target_constraint_type"],
            "before_label": before,
            "after_label": after,
            "target_fixed": bool_str(target_fixed),
        })

    target_results_by_repair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in target_results:
        target_results_by_repair[row["repair_id"]].append(row)

    image_results: List[Dict[str, str]] = []
    for repair_id, labels in sorted(labels_by_repair.items()):
        repair = repair_by_id.get(repair_id, {})
        before_labels = [normalize_label(r.get("before_label", "")) for r in labels]
        after_labels = [normalize_label(r.get("after_label", "")) for r in labels]
        before_image = aggregate_labels(before_labels)
        after_image = aggregate_labels(after_labels)
        regressions = [
            r["constraint_id"] for r in labels
            if normalize_label(r.get("before_label", "")) == "pass" and normalize_label(r.get("after_label", "")) == "fail"
        ]
        target_rows = target_results_by_repair.get(repair_id, [])
        targets_fixed = sum(r["target_fixed"] == "True" for r in target_rows)
        image_results.append({
            "repair_id": repair_id,
            "source_image_id": repair.get("source_image_id", ""),
            "source_prompt_id": repair.get("source_prompt_id", ""),
            "repair_strategy": repair.get("repair_strategy", ""),
            "before_image_label": before_image,
            "after_image_label": after_image,
            "image_fixed": bool_str(before_image != "pass" and after_image == "pass"),
            "num_target_constraints": str(len(target_rows)),
            "num_targets_fixed": str(targets_fixed),
            "num_regressions": str(len(regressions)),
            "regressed_constraint_ids": ";".join(regressions),
        })

    # Repair success by constraint type.
    by_type: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in target_results:
        by_type[row["target_constraint_type"]].append(row)

    regressions_by_repair = {r["repair_id"]: int(r["num_regressions"]) for r in image_results}
    type_rows: List[Dict[str, str]] = []
    for ctype, rows in sorted(by_type.items()):
        repair_ids = {r["repair_id"] for r in rows}
        fixed = sum(r["target_fixed"] == "True" for r in rows)
        regression_counts = [regressions_by_repair.get(rid, 0) for rid in repair_ids]
        type_rows.append({
            "target_constraint_type": ctype,
            "num_targets": str(len(rows)),
            "target_fixed_count": str(fixed),
            "target_fixed_rate": f"{safe_div(fixed, len(rows)):.4f}",
            "num_repairs_with_regression": str(sum(x > 0 for x in regression_counts)),
            "avg_regression_count": f"{safe_div(sum(regression_counts), len(regression_counts)):.4f}",
        })

    num_targets = len(target_results)
    num_target_fixed = sum(r["target_fixed"] == "True" for r in target_results)
    num_repairs = len(image_results)
    num_image_fixed = sum(r["image_fixed"] == "True" for r in image_results)
    num_repairs_with_regression = sum(int(r["num_regressions"]) > 0 for r in image_results)

    summary = [
        {"metric": "num_repair_attempts", "value": str(num_repairs)},
        {"metric": "num_target_constraints", "value": str(num_targets)},
        {"metric": "target_fixed_rate", "value": f"{safe_div(num_target_fixed, num_targets):.4f}"},
        {"metric": "image_fixed_rate", "value": f"{safe_div(num_image_fixed, num_repairs):.4f}"},
        {"metric": "repair_regression_rate", "value": f"{safe_div(num_repairs_with_regression, num_repairs):.4f}"},
    ]

    return {
        "target_results": target_results,
        "image_results": image_results,
        "type_results": type_rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/checkpoint2_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    repair_cfg = config["repair"]

    repaired_prompts = read_csv(repair_cfg["repaired_prompts_csv"])
    repair_targets = read_csv(repair_cfg["repair_targets_csv"])
    repaired_labels = read_csv(repair_cfg["repaired_human_labels_csv"])

    results = analyze_repair(repaired_prompts, repair_targets, repaired_labels)

    write_csv(repair_cfg["repair_target_results_csv"], results["target_results"], TARGET_RESULT_FIELDS)
    write_csv(repair_cfg["repair_image_results_csv"], results["image_results"], IMAGE_RESULT_FIELDS)
    write_csv(repair_cfg["repair_success_by_type_csv"], results["type_results"], TYPE_FIELDS)
    write_csv(repair_cfg["repair_summary_metrics_csv"], results["summary"], SUMMARY_FIELDS)

    print(f"Wrote repair target results to {repair_cfg['repair_target_results_csv']}")
    print(f"Wrote repair image results to {repair_cfg['repair_image_results_csv']}")
    print(f"Wrote repair success by type to {repair_cfg['repair_success_by_type_csv']}")
    print(f"Wrote repair summary metrics to {repair_cfg['repair_summary_metrics_csv']}")


if __name__ == "__main__":
    main()
