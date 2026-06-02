from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from utils import read_csv, safe_div, write_csv

LABELS = {"pass", "fail", "ambiguous"}

TARGET_RESULT_FIELDS = [
    "repair_id",
    "source_image_id",
    "source_prompt_id",
    "target_constraint_id",
    "target_constraint_type",
    "before_label",
    "after_label",
    "target_fixed",
]

IMAGE_RESULT_FIELDS = [
    "repair_id",
    "source_image_id",
    "source_prompt_id",
    "repair_strategy",
    "trigger_label_source",
    "source_provider",
    "source_model_name",
    "prompt_family",
    "scene_context_type",
    "composition",
    "before_image_label",
    "after_image_label",
    "image_fixed",
    "is_unnecessary_repair",
    "unnecessary_repair_caused_regression",
    "num_constraints",
    "num_target_constraints",
    "num_targets_fixed",
    "num_regressions",
    "num_hard_regressions",
    "num_soft_regressions",
    "has_regression",
    "regressed_constraint_ids",
    "hard_regressed_constraint_ids",
    "soft_regressed_constraint_ids",
]

REPAIR_METRIC_FIELDS = [
    "dimension",
    "value",
    "num_examples",
    "before_pass_count",
    "before_pass_rate",
    "before_fail_count",
    "before_fail_rate",
    "before_ambiguous_count",
    "before_ambiguous_rate",
    "after_pass_count",
    "after_pass_rate",
    "after_fail_count",
    "after_fail_rate",
    "after_ambiguous_count",
    "after_ambiguous_rate",
    "pass_rate_delta",
    "nonpass_to_pass_count",
    "nonpass_to_pass_rate",
    "pass_to_nonpass_count",
    "pass_to_nonpass_rate",
    "hard_regression_count",
    "hard_regression_rate",
    "soft_regression_count",
    "soft_regression_rate",
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


def index(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {r[key]: r for r in rows if r.get(key)}


def repair_rates(rows: List[Dict[str, str]], dimension: str = "overall", value: str = "overall") -> Dict[str, str]:
    """Compute before/after repair metrics for rows with before_label/after_label.

    This helper is used for both constraint-level rows and image-level rows.
    """
    n = len(rows)
    before_pass = sum(normalize_label(r.get("before_label", "")) == "pass" for r in rows)
    before_fail = sum(normalize_label(r.get("before_label", "")) == "fail" for r in rows)
    before_amb = sum(normalize_label(r.get("before_label", "")) == "ambiguous" for r in rows)

    after_pass = sum(normalize_label(r.get("after_label", "")) == "pass" for r in rows)
    after_fail = sum(normalize_label(r.get("after_label", "")) == "fail" for r in rows)
    after_amb = sum(normalize_label(r.get("after_label", "")) == "ambiguous" for r in rows)

    nonpass_to_pass = sum(
        normalize_label(r.get("before_label", "")) != "pass"
        and normalize_label(r.get("after_label", "")) == "pass"
        for r in rows
    )
    before_nonpass = sum(normalize_label(r.get("before_label", "")) != "pass" for r in rows)

    pass_to_nonpass = sum(
        normalize_label(r.get("before_label", "")) == "pass"
        and normalize_label(r.get("after_label", "")) != "pass"
        for r in rows
    )
    before_pass_count = before_pass

    hard_regressions = sum(
        normalize_label(r.get("before_label", "")) == "pass"
        and normalize_label(r.get("after_label", "")) == "fail"
        for r in rows
    )
    soft_regressions = sum(
        normalize_label(r.get("before_label", "")) == "pass"
        and normalize_label(r.get("after_label", "")) == "ambiguous"
        for r in rows
    )

    before_pass_rate = safe_div(before_pass, n)
    after_pass_rate = safe_div(after_pass, n)

    return {
        "dimension": dimension,
        "value": value,
        "num_examples": str(n),
        "before_pass_count": str(before_pass),
        "before_pass_rate": f"{before_pass_rate:.4f}",
        "before_fail_count": str(before_fail),
        "before_fail_rate": f"{safe_div(before_fail, n):.4f}",
        "before_ambiguous_count": str(before_amb),
        "before_ambiguous_rate": f"{safe_div(before_amb, n):.4f}",
        "after_pass_count": str(after_pass),
        "after_pass_rate": f"{after_pass_rate:.4f}",
        "after_fail_count": str(after_fail),
        "after_fail_rate": f"{safe_div(after_fail, n):.4f}",
        "after_ambiguous_count": str(after_amb),
        "after_ambiguous_rate": f"{safe_div(after_amb, n):.4f}",
        "pass_rate_delta": f"{(after_pass_rate - before_pass_rate):.4f}",
        "nonpass_to_pass_count": str(nonpass_to_pass),
        "nonpass_to_pass_rate": f"{safe_div(nonpass_to_pass, before_nonpass):.4f}",
        "pass_to_nonpass_count": str(pass_to_nonpass),
        "pass_to_nonpass_rate": f"{safe_div(pass_to_nonpass, before_pass_count):.4f}",
        "hard_regression_count": str(hard_regressions),
        "hard_regression_rate": f"{safe_div(hard_regressions, before_pass_count):.4f}",
        "soft_regression_count": str(soft_regressions),
        "soft_regression_rate": f"{safe_div(soft_regressions, before_pass_count):.4f}",
    }


def metrics_by_group(rows: List[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r.get(key) or "unknown"].append(r)
    return [repair_rates(rs, key, value) for value, rs in sorted(grouped.items())]


def analyze_repair(
    repaired_prompts: List[Dict[str, str]],
    repair_targets: List[Dict[str, str]],
    repaired_labels: List[Dict[str, str]],
    prompts: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    repair_by_id = {r["repair_id"]: r for r in repaired_prompts}
    prompt_by_id = index(prompts or [], "prompt_id")

    labels_by_repair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in repaired_labels:
        if row.get("after_label", "").strip():
            labels_by_repair[row["repair_id"]].append(row)

    after_by_key = {
        (r["repair_id"], r["constraint_id"]): normalize_label(r.get("after_label", ""))
        for r in repaired_labels
    }

    target_results: List[Dict[str, str]] = []
    for target in repair_targets:
        repair_id = target["repair_id"]
        cid = target["target_constraint_id"]
        before = normalize_label(target.get("before_label", ""))
        after = after_by_key.get((repair_id, cid), "ambiguous")
        target_fixed = before in {"fail", "ambiguous"} and after == "pass"
        target_results.append({
            "repair_id": repair_id,
            "source_image_id": target.get("source_image_id", repair_by_id.get(repair_id, {}).get("source_image_id", "")),
            "source_prompt_id": target.get("source_prompt_id", repair_by_id.get(repair_id, {}).get("source_prompt_id", "")),
            "target_constraint_id": cid,
            "target_constraint_type": target["target_constraint_type"],
            "before_label": before,
            "after_label": after,
            "target_fixed": bool_str(target_fixed),
        })

    target_results_by_repair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in target_results:
        target_results_by_repair[row["repair_id"]].append(row)

    # Constraint-level rows contain all labeled constraints on repaired images, not
    # only the target constraints. These are the rows used for by-type/by-model/by-scene
    # repair metrics, aligned with the initial-generation analysis.
    constraint_rows: List[Dict[str, str]] = []
    image_results: List[Dict[str, str]] = []

    for repair_id, labels in sorted(labels_by_repair.items()):
        repair = repair_by_id.get(repair_id, {})
        source_prompt_id = repair.get("source_prompt_id", "")
        prompt_meta = prompt_by_id.get(source_prompt_id, {})
        prompt_family = repair.get("prompt_family", "") or prompt_meta.get("prompt_family", "")
        scene_context_type = repair.get("scene_context_type", "") or prompt_meta.get("scene_context_type", "")
        composition = repair.get("composition", "") or prompt_meta.get("composition", "")

        before_labels = [normalize_label(r.get("before_label", "")) for r in labels]
        after_labels = [normalize_label(r.get("after_label", "")) for r in labels]
        before_image = aggregate_labels(before_labels)
        after_image = aggregate_labels(after_labels)

        hard_regressions = [
            r["constraint_id"] for r in labels
            if normalize_label(r.get("before_label", "")) == "pass" and normalize_label(r.get("after_label", "")) == "fail"
        ]
        soft_regressions = [
            r["constraint_id"] for r in labels
            if normalize_label(r.get("before_label", "")) == "pass" and normalize_label(r.get("after_label", "")) == "ambiguous"
        ]
        regressions = hard_regressions + soft_regressions

        for r in labels:
            constraint_rows.append({
                "repair_id": repair_id,
                "source_image_id": repair.get("source_image_id", r.get("source_image_id", "")),
                "source_prompt_id": source_prompt_id,
                "constraint_id": r["constraint_id"],
                "constraint_type": r.get("constraint_type", ""),
                "before_label": normalize_label(r.get("before_label", "")),
                "after_label": normalize_label(r.get("after_label", "")),
                "provider": repair.get("source_provider", ""),
                "model_name": repair.get("source_model_name", ""),
                "prompt_family": prompt_family,
                "scene_context_type": scene_context_type,
                "composition": composition,
                "is_target_constraint": r.get("is_target_constraint", ""),
            })

        target_rows = target_results_by_repair.get(repair_id, [])
        targets_fixed = sum(r["target_fixed"] == "True" for r in target_rows)
        trigger_source = repair.get("trigger_label_source", "")

        # In the VLM-triggered setting, a repair can be unnecessary if the VLM
        # flags an image that human labels say was already fully correct.
        is_unnecessary_repair = trigger_source == "vlm" and before_image == "pass"
        unnecessary_repair_caused_regression = is_unnecessary_repair and after_image != "pass"

        image_results.append({
            "repair_id": repair_id,
            "source_image_id": repair.get("source_image_id", ""),
            "source_prompt_id": source_prompt_id,
            "repair_strategy": repair.get("repair_strategy", ""),
            "trigger_label_source": trigger_source,
            "source_provider": repair.get("source_provider", ""),
            "source_model_name": repair.get("source_model_name", ""),
            "prompt_family": prompt_family,
            "scene_context_type": scene_context_type,
            "composition": composition,
            "before_image_label": before_image,
            "after_image_label": after_image,
            "image_fixed": bool_str(before_image != "pass" and after_image == "pass"),
            "is_unnecessary_repair": bool_str(is_unnecessary_repair),
            "unnecessary_repair_caused_regression": bool_str(unnecessary_repair_caused_regression),
            "num_constraints": str(len(labels)),
            "num_target_constraints": str(len(target_rows)),
            "num_targets_fixed": str(targets_fixed),
            "num_regressions": str(len(regressions)),
            "num_hard_regressions": str(len(hard_regressions)),
            "num_soft_regressions": str(len(soft_regressions)),
            "has_regression": bool_str(len(regressions) > 0),
            "regressed_constraint_ids": ";".join(regressions),
            "hard_regressed_constraint_ids": ";".join(hard_regressions),
            "soft_regressed_constraint_ids": ";".join(soft_regressions),
        })

    image_metric_rows = [
        {
            "before_label": r["before_image_label"],
            "after_label": r["after_image_label"],
            "model_name": r.get("source_model_name", ""),
            "prompt_family": r.get("prompt_family", ""),
            "scene_context_type": r.get("scene_context_type", ""),
            "composition": r.get("composition", ""),
        }
        for r in image_results
    ]

    num_targets = len(target_results)
    num_target_fixed = sum(r["target_fixed"] == "True" for r in target_results)
    num_repairs = len(image_results)
    num_image_fixed = sum(r["image_fixed"] == "True" for r in image_results)
    num_repairs_with_regression = sum(r["has_regression"] == "True" for r in image_results)
    total_regressions = sum(int(r["num_regressions"]) for r in image_results)
    total_hard_regressions = sum(int(r["num_hard_regressions"]) for r in image_results)
    total_soft_regressions = sum(int(r["num_soft_regressions"]) for r in image_results)
    num_unnecessary_repairs = sum(r.get("is_unnecessary_repair") == "True" for r in image_results)
    num_unnecessary_repairs_with_regression = sum(r.get("unnecessary_repair_caused_regression") == "True" for r in image_results)

    overall_constraint_metrics = repair_rates(constraint_rows, "overall", "overall")
    overall_image_metrics = repair_rates(image_metric_rows, "overall", "overall")

    summary = [
        {"metric": "num_repair_attempts", "value": str(num_repairs)},
        {"metric": "num_target_constraints", "value": str(num_targets)},
        {"metric": "target_fixed_rate", "value": f"{safe_div(num_target_fixed, num_targets):.4f}"},
        {"metric": "image_fixed_rate", "value": f"{safe_div(num_image_fixed, num_repairs):.4f}"},
        {"metric": "constraint_before_pass_rate", "value": overall_constraint_metrics["before_pass_rate"]},
        {"metric": "constraint_after_pass_rate", "value": overall_constraint_metrics["after_pass_rate"]},
        {"metric": "constraint_pass_rate_delta", "value": overall_constraint_metrics["pass_rate_delta"]},
        {"metric": "image_before_pass_rate", "value": overall_image_metrics["before_pass_rate"]},
        {"metric": "image_after_pass_rate", "value": overall_image_metrics["after_pass_rate"]},
        {"metric": "image_pass_rate_delta", "value": overall_image_metrics["pass_rate_delta"]},
        {"metric": "repair_regression_rate", "value": f"{safe_div(num_repairs_with_regression, num_repairs):.4f}"},
        {"metric": "avg_regression_count", "value": f"{safe_div(total_regressions, num_repairs):.4f}"},
        {"metric": "total_regressions", "value": str(total_regressions)},
        {"metric": "total_hard_regressions", "value": str(total_hard_regressions)},
        {"metric": "total_soft_regressions", "value": str(total_soft_regressions)},
        {"metric": "unnecessary_repair_count", "value": str(num_unnecessary_repairs)},
        {"metric": "unnecessary_repair_rate", "value": f"{safe_div(num_unnecessary_repairs, num_repairs):.4f}"},
        {"metric": "unnecessary_repair_regression_count", "value": str(num_unnecessary_repairs_with_regression)},
        {"metric": "unnecessary_repair_regression_rate", "value": f"{safe_div(num_unnecessary_repairs_with_regression, num_unnecessary_repairs):.4f}"},
    ]

    return {
        "target_results": target_results,
        "image_results": image_results,
        "summary": summary,
        "constraint_metrics_by_type": metrics_by_group(constraint_rows, "constraint_type"),
        "constraint_metrics_by_model": metrics_by_group(constraint_rows, "model_name"),
        "constraint_metrics_by_scene_context_type": metrics_by_group(constraint_rows, "scene_context_type"),
        "image_metrics_by_prompt_family": metrics_by_group(image_metric_rows, "prompt_family"),
        "image_metrics_by_model": metrics_by_group(image_metric_rows, "model_name"),
        "image_metrics_by_scene_context_type": metrics_by_group(image_metric_rows, "scene_context_type"),
        # Backward-compatible alias for old tests/code.
        "type_results": metrics_by_group(target_results, "target_constraint_type"),
    }



def read_repair_inputs_for_source(
    repair_cfg: Dict,
    source: str,
    repaired_prompts_override: Optional[str] = None,
    repair_targets_override: Optional[str] = None,
    repaired_labels_override: Optional[str] = None,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load repair prompts, targets, and labels for one trigger source.

    For source="combined", concatenate the human-triggered and VLM-triggered repair
    inputs. This treats the combined analysis as an overall summary over all repair
    attempts, while preserving the original trigger_label_source field on each row.
    """
    if source in {"human", "vlm"}:
        repaired_prompts_path = repaired_prompts_override or repair_cfg.get(f"repaired_prompts_{source}_csv") or repair_cfg["repaired_prompts_csv"]
        repair_targets_path = repair_targets_override or repair_cfg.get(f"repair_targets_{source}_csv") or repair_cfg["repair_targets_csv"]
        repaired_labels_path = repaired_labels_override or repair_cfg.get(f"repaired_{source}_labels_csv") or repair_cfg.get(f"repaired_{source}_human_labels_csv") or repair_cfg["repaired_human_labels_csv"]
        return read_csv(repaired_prompts_path), read_csv(repair_targets_path), read_csv(repaired_labels_path)

    if source == "combined":
        if repaired_prompts_override or repair_targets_override or repaired_labels_override:
            if not (repaired_prompts_override and repair_targets_override and repaired_labels_override):
                raise ValueError(
                    "For --trigger-source combined, either provide all three explicit input paths "
                    "(--repaired-prompts, --repair-targets, --repaired-labels) or provide none."
                )
            return read_csv(repaired_prompts_override), read_csv(repair_targets_override), read_csv(repaired_labels_override)

        all_repaired_prompts: List[Dict[str, str]] = []
        all_repair_targets: List[Dict[str, str]] = []
        all_repaired_labels: List[Dict[str, str]] = []
        for subsource in ["human", "vlm"]:
            repaired_prompts, repair_targets, repaired_labels = read_repair_inputs_for_source(repair_cfg, subsource)
            all_repaired_prompts.extend(repaired_prompts)
            all_repair_targets.extend(repair_targets)
            all_repaired_labels.extend(repaired_labels)
        return all_repaired_prompts, all_repair_targets, all_repaired_labels

    raise ValueError(f"Unknown repair trigger source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/final_experiment_config.yaml")
    parser.add_argument("--trigger-source", choices=["human", "vlm", "combined"], default="human")
    parser.add_argument("--repaired-prompts", default=None)
    parser.add_argument("--repair-targets", default=None)
    parser.add_argument("--repaired-labels", default=None)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--out-dir", default="../results/final")
    args = parser.parse_args()
    config = load_config(args.config)
    repair_cfg = config["repair"]
    source = args.trigger_source

    repaired_prompts, repair_targets, repaired_labels = read_repair_inputs_for_source(
        repair_cfg,
        source,
        repaired_prompts_override=args.repaired_prompts,
        repair_targets_override=args.repair_targets,
        repaired_labels_override=args.repaired_labels,
    )
    prompts_path = args.prompts or config.get("prompt_subset", {}).get("out_prompts")
    prompts = read_csv(prompts_path) if prompts_path else []
    results = analyze_repair(repaired_prompts, repair_targets, repaired_labels, prompts=prompts)

    # Prefer explicit config paths when available; otherwise write source-specific
    # files under --out-dir to avoid human/vlm/combined runs overwriting each other.
    target_out = repair_cfg.get(f"repair_{source}_target_results_csv", f"{args.out_dir}/repair_{source}_target_results.csv")
    image_out = repair_cfg.get(f"repair_{source}_image_results_csv", f"{args.out_dir}/repair_{source}_image_results.csv")
    summary_out = repair_cfg.get(f"repair_{source}_summary_csv", f"{args.out_dir}/repair_{source}_summary.csv")
    constraint_type_out = repair_cfg.get(f"repair_{source}_constraint_metrics_by_type_csv", f"{args.out_dir}/repair_{source}_constraint_metrics_by_type.csv")
    image_family_out = repair_cfg.get(f"repair_{source}_image_metrics_by_prompt_family_csv", f"{args.out_dir}/repair_{source}_image_metrics_by_prompt_family.csv")
    image_model_out = repair_cfg.get(f"repair_{source}_image_metrics_by_model_csv", f"{args.out_dir}/repair_{source}_image_metrics_by_model.csv")
    image_scene_out = repair_cfg.get(f"repair_{source}_image_metrics_by_scene_context_type_csv", f"{args.out_dir}/repair_{source}_image_metrics_by_scene_context_type.csv")
    constraint_model_out = repair_cfg.get(f"repair_{source}_constraint_metrics_by_model_csv", f"{args.out_dir}/repair_{source}_constraint_metrics_by_model.csv")
    constraint_scene_out = repair_cfg.get(f"repair_{source}_constraint_metrics_by_scene_context_type_csv", f"{args.out_dir}/repair_{source}_constraint_metrics_by_scene_context_type.csv")

    write_csv(summary_out, results["summary"], SUMMARY_FIELDS)
    write_csv(constraint_type_out, results["constraint_metrics_by_type"], REPAIR_METRIC_FIELDS)
    write_csv(image_family_out, results["image_metrics_by_prompt_family"], REPAIR_METRIC_FIELDS)
    write_csv(image_model_out, results["image_metrics_by_model"], REPAIR_METRIC_FIELDS)
    write_csv(image_scene_out, results["image_metrics_by_scene_context_type"], REPAIR_METRIC_FIELDS)
    write_csv(constraint_model_out, results["constraint_metrics_by_model"], REPAIR_METRIC_FIELDS)
    write_csv(constraint_scene_out, results["constraint_metrics_by_scene_context_type"], REPAIR_METRIC_FIELDS)
    write_csv(target_out, results["target_results"], TARGET_RESULT_FIELDS)
    write_csv(image_out, results["image_results"], IMAGE_RESULT_FIELDS)

    print(f"Wrote {source}-triggered repair analysis results.")


if __name__ == "__main__":
    main()
