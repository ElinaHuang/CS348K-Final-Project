from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

from utils import read_csv, safe_div, write_csv


LABELS = ["pass", "fail", "ambiguous"]


''' 1 - Tool functions '''

def normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    return label if label in LABELS else "ambiguous"


def aggregate_labels(labels: List[str]) -> str:
    """Aggregate atomic constraint labels into one image-level label.

    Rule:
    - If any atomic constraint is fail -> fail.
    - Else if all atomic constraints are pass -> pass.
    - Else -> ambiguous.

    This conservative aggregation means an image is considered fully correct
    only when all of its required atomic constraints pass.
    """
    labels = [normalize_label(x) for x in labels if (x or "").strip()]
    if not labels:
        return "ambiguous"
    if any(x == "fail" for x in labels):
        return "fail"
    if all(x == "pass" for x in labels):
        return "pass"
    return "ambiguous"


def index_by_key(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {r[key]: r for r in rows if key in r and r[key]}


''' 2 - Label merging '''

def merge_labels(
    human_rows: List[Dict[str, str]],
    vlm_rows: List[Dict[str, str]],
    prompts: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """Merge human and VLM labels at the image-constraint level.

    Constraint-level rows are used mainly for diagnosis and VLM checker
    validation. Prompt-level metadata is attached so that image-level
    aggregation can later evaluate prompt-family / scene / composition effects.
    """
    vlm_by_key = {(r["image_id"], r["constraint_id"]): r for r in vlm_rows}
    prompts_by_id = index_by_key(prompts or [], "prompt_id")

    merged: List[Dict[str, str]] = []
    for h in human_rows:
        if not h.get("human_label", "").strip():
            continue

        key = (h["image_id"], h["constraint_id"])
        v = vlm_by_key.get(key)
        if not v:
            continue

        p = prompts_by_id.get(h.get("prompt_id", ""), {})
        merged.append({
            "image_id": h["image_id"],
            "prompt_id": h["prompt_id"],
            "constraint_id": h["constraint_id"],
            "constraint_type": h["constraint_type"],
            "prompt_family": p.get("prompt_family", ""),
            "scene_context_type": p.get("scene_context_type", ""),
            "scene_context": p.get("scene_context", ""),
            "composition": p.get("composition", ""),
            "human_label": normalize_label(h.get("human_label", "")),
            "vlm_label": normalize_label(v.get("vlm_label", "")),
            "human_notes": h.get("human_notes", ""),
            "vlm_reason": v.get("vlm_reason", ""),
            "parse_status": v.get("parse_status", ""),
        })

    return merged


''' 3 - General metrics computing '''

def overall_agreement(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(r["human_label"] == r["vlm_label"] for r in rows) / len(rows)


def human_pass_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r["human_label"] == "pass" for r in rows), len(rows))


def human_fail_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r["human_label"] == "fail" for r in rows), len(rows))


def human_ambiguous_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r["human_label"] == "ambiguous" for r in rows), len(rows))


def vlm_pass_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r["vlm_label"] == "pass" for r in rows), len(rows))


def vlm_fail_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r["vlm_label"] == "fail" for r in rows), len(rows))


def vlm_ambiguous_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r["vlm_label"] == "ambiguous" for r in rows), len(rows))


def confusion_matrix(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    counts = Counter((r["human_label"], r["vlm_label"]) for r in rows)
    out = []
    for h in LABELS:
        row = {"human_label": h}
        for v in LABELS:
            row[f"vlm_{v}"] = counts.get((h, v), 0)
        out.append(row)
    return out


def failure_detection_metrics(rows: List[Dict[str, str]], group_name: str = "overall") -> Dict[str, str]:
    """Treat human fail as positive and VLM fail as predicted positive."""
    tp = sum(r["human_label"] == "fail" and r["vlm_label"] == "fail" for r in rows)
    fp = sum(r["human_label"] != "fail" and r["vlm_label"] == "fail" for r in rows)
    fn = sum(r["human_label"] == "fail" and r["vlm_label"] != "fail" for r in rows)

    human_fail = tp + fn
    human_pass = sum(r["human_label"] == "pass" for r in rows)

    false_pass = sum(r["human_label"] == "fail" and r["vlm_label"] == "pass" for r in rows)
    false_fail = sum(r["human_label"] == "pass" and r["vlm_label"] == "fail" for r in rows)

    return {
        "group": group_name,
        "num_examples": len(rows),
        "human_pass_rate": f"{human_pass_rate(rows):.4f}",
        "human_fail_rate": f"{human_fail_rate(rows):.4f}",
        "human_ambiguous_rate": f"{human_ambiguous_rate(rows):.4f}",
        "vlm_pass_rate": f"{vlm_pass_rate(rows):.4f}",
        "vlm_fail_rate": f"{vlm_fail_rate(rows):.4f}",
        "vlm_ambiguous_rate": f"{vlm_ambiguous_rate(rows):.4f}",
        "agreement": f"{overall_agreement(rows):.4f}",
        "failure_precision": f"{safe_div(tp, tp + fp):.4f}",
        "failure_recall": f"{safe_div(tp, tp + fn):.4f}",
        "failure_f1": f"{safe_div(2 * tp, 2 * tp + fp + fn):.4f}",
        "false_pass_rate_among_human_fail": f"{safe_div(false_pass, human_fail):.4f}",
        "false_fail_rate_among_human_pass": f"{safe_div(false_fail, human_pass):.4f}",
    }


''' 4 - Grouping '''

def group_rows(rows: List[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        value = r.get(key, "") or "unknown"
        grouped[value].append(r)
    return grouped


def metrics_by_group(rows: List[Dict[str, str]], key: str, prefix: str | None = None) -> List[Dict[str, str]]:
    out = []
    for value, rs in sorted(group_rows(rows, key).items()):
        name = f"{key}={value}" if prefix is None else f"{prefix}:{key}={value}"
        metric = failure_detection_metrics(rs, name)
        metric["dimension"] = key
        metric["value"] = value
        out.append(metric)
    return out


def metrics_by_cross_group(rows: List[Dict[str, str]], keys: List[str]) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        group_key = tuple((r.get(k, "") or "unknown") for k in keys)
        grouped[group_key].append(r)

    out = []
    for group_key, rs in sorted(grouped.items()):
        group_name = ",".join(f"{k}={v}" for k, v in zip(keys, group_key))
        metric = failure_detection_metrics(rs, group_name)
        for k, v in zip(keys, group_key):
            metric[k] = v
        out.append(metric)
    return out


''' 5 - Constraint level checker analysis '''

def constraint_metrics_by_type(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Constraint-level diagnostics grouped only by atomic constraint type.

    This is intended for diagnosis and repair-trigger analysis, not for the main
    T2I controllability breakdown.
    """
    return metrics_by_group(rows, "constraint_type", prefix="constraint")

''' 6 - Image level checker analysis '''

def aggregate_to_image_level(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Aggregate image-constraint rows into image-level rows."""
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["image_id"]].append(r)

    image_rows: List[Dict[str, str]] = []
    for image_id, rs in sorted(grouped.items()):
        human_image_label = aggregate_labels([r["human_label"] for r in rs])
        vlm_image_label = aggregate_labels([r["vlm_label"] for r in rs])

        failed_human = [r["constraint_id"] for r in rs if r["human_label"] == "fail"]
        failed_vlm = [r["constraint_id"] for r in rs if r["vlm_label"] == "fail"]
        ambiguous_human = [r["constraint_id"] for r in rs if r["human_label"] == "ambiguous"]
        ambiguous_vlm = [r["constraint_id"] for r in rs if r["vlm_label"] == "ambiguous"]

        image_rows.append({
            "image_id": image_id,
            "prompt_id": rs[0]["prompt_id"],
            "prompt_family": rs[0].get("prompt_family", ""),
            "scene_context_type": rs[0].get("scene_context_type", ""),
            "scene_context": rs[0].get("scene_context", ""),
            "composition": rs[0].get("composition", ""),
            "num_constraints": len(rs),
            "human_image_label": human_image_label,
            "vlm_image_label": vlm_image_label,
            "image_label_agreement": str(human_image_label == vlm_image_label),
            "human_failed_constraints": ";".join(failed_human),
            "vlm_failed_constraints": ";".join(failed_vlm),
            "human_ambiguous_constraints": ";".join(ambiguous_human),
            "vlm_ambiguous_constraints": ";".join(ambiguous_vlm),
        })

    return image_rows


def image_level_agreement(image_rows: List[Dict[str, str]]) -> float:
    if not image_rows:
        return 0.0
    return sum(r["human_image_label"] == r["vlm_image_label"] for r in image_rows) / len(image_rows)


''' 7 - Image level controllability analysis '''

def image_rows_as_label_rows(image_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "human_label": r["human_image_label"],
            "vlm_label": r["vlm_image_label"],
            "prompt_family": r.get("prompt_family", ""),
            "scene_context_type": r.get("scene_context_type", ""),
            "scene_context": r.get("scene_context", ""),
            "composition": r.get("composition", ""),
            "constraint_type": "image_level",
        }
        for r in image_rows
    ]


def image_level_label_distribution(image_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    human_counts = Counter(r["human_image_label"] for r in image_rows)
    vlm_counts = Counter(r["vlm_image_label"] for r in image_rows)
    return [
        {"label": label, "human_count": human_counts.get(label, 0), "vlm_count": vlm_counts.get(label, 0)}
        for label in LABELS
    ]


def image_level_confusion_matrix(image_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return confusion_matrix(image_rows_as_label_rows(image_rows))


def image_level_failure_detection_metrics(image_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [failure_detection_metrics(image_rows_as_label_rows(image_rows), "image_level")]


def image_controllability_metrics_by_dimension(image_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Main T2I controllability metrics grouped by prompt-level design dimensions."""
    label_rows = image_rows_as_label_rows(image_rows)
    out = []
    out.extend(metrics_by_group(label_rows, "prompt_family", prefix="prompt"))
    out.extend(metrics_by_group(label_rows, "scene_context_type", prefix="scene"))
    out.extend(metrics_by_group(label_rows, "composition", prefix="composition"))
    return out


METRIC_FIELDS = [
    "group", "dimension", "value", "num_examples",
    "human_pass_rate", "human_fail_rate", "human_ambiguous_rate",
    "vlm_pass_rate", "vlm_fail_rate", "vlm_ambiguous_rate",
    "agreement", "failure_precision", "failure_recall", "failure_f1",
    "false_pass_rate_among_human_fail", "false_fail_rate_among_human_pass",
]


BASIC_METRIC_FIELDS = [
    "group", "num_examples",
    "human_pass_rate", "human_fail_rate", "human_ambiguous_rate",
    "vlm_pass_rate", "vlm_fail_rate", "vlm_ambiguous_rate",
    "agreement", "failure_precision", "failure_recall", "failure_f1",
    "false_pass_rate_among_human_fail", "false_fail_rate_among_human_pass",
]


SCENE_COMPOSITION_FIELDS = [
    "group", "scene_context_type", "composition", "num_examples",
    "human_pass_rate", "human_fail_rate", "human_ambiguous_rate",
    "vlm_pass_rate", "vlm_fail_rate", "vlm_ambiguous_rate",
    "agreement", "failure_precision", "failure_recall", "failure_f1",
    "false_pass_rate_among_human_fail", "false_fail_rate_among_human_pass",
]


FAMILY_SCENE_COMPOSITION_FIELDS = [
    "group", "prompt_family", "scene_context_type", "composition", "num_examples",
    "human_pass_rate", "human_fail_rate", "human_ambiguous_rate",
    "vlm_pass_rate", "vlm_fail_rate", "vlm_ambiguous_rate",
    "agreement", "failure_precision", "failure_recall", "failure_f1",
    "false_pass_rate_among_human_fail", "false_fail_rate_among_human_pass",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="../data/labels/human_labels.csv")
    parser.add_argument("--vlm", default="../data/labels/vlm_labels.csv")
    parser.add_argument("--prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--out-dir", default="../results/checkpoint1")
    args = parser.parse_args()

    human = read_csv(args.human)
    vlm = read_csv(args.vlm)
    prompts = read_csv(args.prompts)

    merged = merge_labels(human, vlm, prompts=prompts)
    image_rows = aggregate_to_image_level(merged)
    image_label_rows = image_rows_as_label_rows(image_rows)

    summary = [
        {"metric": "num_constraint_level_examples", "value": str(len(merged))},
        {"metric": "constraint_level_overall_agreement", "value": f"{overall_agreement(merged):.4f}"},
        {"metric": "constraint_level_human_pass_rate", "value": f"{human_pass_rate(merged):.4f}"},
        {"metric": "constraint_level_human_fail_rate", "value": f"{human_fail_rate(merged):.4f}"},
        {"metric": "constraint_level_human_ambiguous_rate", "value": f"{human_ambiguous_rate(merged):.4f}"},
        {"metric": "num_image_level_examples", "value": str(len(image_rows))},
        {"metric": "image_level_agreement", "value": f"{image_level_agreement(image_rows):.4f}"},
        {"metric": "image_level_human_pass_rate", "value": f"{human_pass_rate(image_label_rows):.4f}"},
        {"metric": "image_level_human_fail_rate", "value": f"{human_fail_rate(image_label_rows):.4f}"},
        {"metric": "image_level_human_ambiguous_rate", "value": f"{human_ambiguous_rate(image_label_rows):.4f}"},
    ]

    write_csv(f"{args.out_dir}/summary_metrics.csv", summary, ["metric", "value"])

    # 1. Constraint-level checker analysis:
    #    Used for atomic-constraint diagnosis, VLM checker validation, and repair triggers.
    write_csv(f"{args.out_dir}/constraint_checker_merged_labels.csv", merged, [
        "image_id", "prompt_id", "constraint_id", "constraint_type",
        "prompt_family", "scene_context_type", "scene_context", "composition",
        "human_label", "vlm_label", "human_notes", "vlm_reason", "parse_status"
    ])
    write_csv(f"{args.out_dir}/constraint_checker_metrics_by_type.csv", constraint_metrics_by_type(merged), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/constraint_checker_confusion_matrix.csv", confusion_matrix(merged), [
        "human_label", "vlm_pass", "vlm_fail", "vlm_ambiguous"
    ])
    write_csv(f"{args.out_dir}/constraint_checker_failure_detection_metrics.csv", [failure_detection_metrics(merged, "overall")], BASIC_METRIC_FIELDS)

    # 2. Image-level checker analysis:
    #    Used to evaluate whether the VLM checker can judge whole-image prompt success.
    write_csv(f"{args.out_dir}/image_checker_labels.csv", image_rows, [
        "image_id", "prompt_id", "prompt_family", "scene_context_type", "scene_context", "composition",
        "num_constraints", "human_image_label", "vlm_image_label", "image_label_agreement",
        "human_failed_constraints", "vlm_failed_constraints",
        "human_ambiguous_constraints", "vlm_ambiguous_constraints",
    ])
    write_csv(f"{args.out_dir}/image_checker_label_distribution.csv", image_level_label_distribution(image_rows), [
        "label", "human_count", "vlm_count"
    ])
    write_csv(f"{args.out_dir}/image_checker_confusion_matrix.csv", image_level_confusion_matrix(image_rows), [
        "human_label", "vlm_pass", "vlm_fail", "vlm_ambiguous"
    ])
    write_csv(f"{args.out_dir}/image_checker_failure_detection_metrics.csv", image_level_failure_detection_metrics(image_rows), BASIC_METRIC_FIELDS)

    # 3. Image-level controllability analysis:
    #    Used to evaluate T2I controllability under prompt-level design dimensions.
    write_csv(f"{args.out_dir}/image_controllability_by_dimension.csv", image_controllability_metrics_by_dimension(image_rows), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/image_controllability_by_scene_and_composition.csv", metrics_by_cross_group(image_label_rows, ["scene_context_type", "composition"]), SCENE_COMPOSITION_FIELDS)
    write_csv(f"{args.out_dir}/image_controllability_by_family_scene_composition.csv", metrics_by_cross_group(image_label_rows, ["prompt_family", "scene_context_type", "composition"]), FAMILY_SCENE_COMPOSITION_FIELDS)

    print(f"Merged {len(merged)} constraint-level labeled pairs.")
    print(f"Aggregated {len(image_rows)} image-level examples.")
    print(f"Constraint-level agreement: {overall_agreement(merged):.4f}")
    print(f"Image-level agreement: {image_level_agreement(image_rows):.4f}")
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
