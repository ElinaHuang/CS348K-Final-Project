from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

from utils import aggregate_labels, normalize_label, read_csv, safe_div, write_csv

LABELS = ["pass", "fail", "ambiguous"]
METRIC_FIELDS = ["group", "num_examples", "human_pass_rate", "human_fail_rate", "human_ambiguous_rate", "vlm_pass_rate", "vlm_fail_rate", "vlm_ambiguous_rate", "agreement", "nonpass_precision", "nonpass_recall", "nonpass_f1", "dimension", "value"]
CONFUSION_FIELDS = ["human_label", "vlm_pass", "vlm_fail", "vlm_ambiguous"]
IMAGE_FIELDS = ["image_id", "prompt_id", "provider", "model_name", "prompt_family", "scene_context_type", "composition", "num_constraints", "human_image_label", "vlm_image_label", "image_label_agreement", "human_failed_constraints", "vlm_failed_constraints", "human_ambiguous_constraints", "vlm_ambiguous_constraints"]


def index(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {r[key]: r for r in rows if r.get(key)}


def merge_labels(human_rows, vlm_rows, prompts=None, generations=None):
    vlm_by_key = {(r["image_id"], r["constraint_id"]): r for r in vlm_rows}
    prompt_by_id = index(prompts or [], "prompt_id")
    gen_by_image = index(generations or [], "image_id")
    out = []
    for h in human_rows:
        if not h.get("human_label", "").strip():
            continue
        v = vlm_by_key.get((h["image_id"], h["constraint_id"]))
        if not v:
            continue
        p = prompt_by_id.get(h["prompt_id"], {})
        g = gen_by_image.get(h["image_id"], {})
        out.append({
            "image_id": h["image_id"], "prompt_id": h["prompt_id"], "constraint_id": h["constraint_id"], "constraint_type": h["constraint_type"],
            "provider": g.get("provider", h.get("provider", "")), "model_name": g.get("model_name", h.get("model_name", "")),
            "prompt_family": p.get("prompt_family", ""), "scene_context_type": p.get("scene_context_type", ""), "composition": p.get("composition", ""),
            "human_label": normalize_label(h.get("human_label", "")), "vlm_label": normalize_label(v.get("vlm_label", "")),
            "human_notes": h.get("human_notes", ""), "vlm_reason": v.get("vlm_reason", ""),
        })
    return out


def rates(rows, group="overall"):
    n = len(rows)
    agreement = sum(r["human_label"] == r["vlm_label"] for r in rows)
    tp = sum(r["human_label"] != "pass" and r["vlm_label"] != "pass" for r in rows)
    fp = sum(r["human_label"] == "pass" and r["vlm_label"] != "pass" for r in rows)
    fn = sum(r["human_label"] != "pass" and r["vlm_label"] == "pass" for r in rows)
    return {
        "group": group, "num_examples": str(n),
        "human_pass_rate": f"{safe_div(sum(r['human_label']=='pass' for r in rows), n):.4f}",
        "human_fail_rate": f"{safe_div(sum(r['human_label']=='fail' for r in rows), n):.4f}",
        "human_ambiguous_rate": f"{safe_div(sum(r['human_label']=='ambiguous' for r in rows), n):.4f}",
        "vlm_pass_rate": f"{safe_div(sum(r['vlm_label']=='pass' for r in rows), n):.4f}",
        "vlm_fail_rate": f"{safe_div(sum(r['vlm_label']=='fail' for r in rows), n):.4f}",
        "vlm_ambiguous_rate": f"{safe_div(sum(r['vlm_label']=='ambiguous' for r in rows), n):.4f}",
        "agreement": f"{safe_div(agreement, n):.4f}",
        "nonpass_precision": f"{safe_div(tp, tp+fp):.4f}",
        "nonpass_recall": f"{safe_div(tp, tp+fn):.4f}",
        "nonpass_f1": f"{safe_div(2*tp, 2*tp+fp+fn):.4f}",
        "dimension": "overall", "value": "overall",
    }


def by_group(rows, key):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get(key) or "unknown"].append(r)
    out = []
    for value, rs in sorted(grouped.items()):
        row = rates(rs, f"{key}={value}"); row["dimension"] = key; row["value"] = value; out.append(row)
    return out


def confusion_matrix(rows):
    counts = Counter((r["human_label"], r["vlm_label"]) for r in rows)
    out = []
    for h in LABELS:
        row = {"human_label": h}
        for v in LABELS:
            row[f"vlm_{v}"] = counts.get((h, v), 0)
        out.append(row)
    return out


def aggregate_to_image_level(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["image_id"]].append(r)
    out = []
    for image_id, rs in sorted(grouped.items()):
        human = aggregate_labels([r["human_label"] for r in rs])
        vlm = aggregate_labels([r["vlm_label"] for r in rs])
        out.append({
            "image_id": image_id, "prompt_id": rs[0]["prompt_id"], "provider": rs[0].get("provider", ""), "model_name": rs[0].get("model_name", ""),
            "prompt_family": rs[0].get("prompt_family", ""), "scene_context_type": rs[0].get("scene_context_type", ""), "composition": rs[0].get("composition", ""),
            "num_constraints": str(len(rs)), "human_image_label": human, "vlm_image_label": vlm, "image_label_agreement": str(human == vlm),
            "human_failed_constraints": ";".join(r["constraint_id"] for r in rs if r["human_label"] == "fail"),
            "vlm_failed_constraints": ";".join(r["constraint_id"] for r in rs if r["vlm_label"] == "fail"),
            "human_ambiguous_constraints": ";".join(r["constraint_id"] for r in rs if r["human_label"] == "ambiguous"),
            "vlm_ambiguous_constraints": ";".join(r["constraint_id"] for r in rs if r["vlm_label"] == "ambiguous"),
        })
    return out


def image_rows_as_label_rows(image_rows):
    return [{
        "human_label": r["human_image_label"], "vlm_label": r["vlm_image_label"],
        "provider": r.get("provider", ""), "model_name": r.get("model_name", ""),
        "prompt_family": r.get("prompt_family", ""), "scene_context_type": r.get("scene_context_type", ""), "composition": r.get("composition", ""),
    } for r in image_rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="../data/labels/human_labels.csv")
    parser.add_argument("--vlm", default="../data/labels/vlm_labels.csv")
    parser.add_argument("--prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--generations", default="../data/generations/generations.csv")
    parser.add_argument("--out-dir", default="../results/final")
    args = parser.parse_args()
    human, vlm, prompts, gens = read_csv(args.human), read_csv(args.vlm), read_csv(args.prompts), read_csv(args.generations)
    merged = merge_labels(human, vlm, prompts, gens)
    image_rows = aggregate_to_image_level(merged)
    image_label_rows = image_rows_as_label_rows(image_rows)
    write_csv(f"{args.out_dir}/constraint_checker_metrics_overall.csv", [rates(merged)], METRIC_FIELDS)
    write_csv(f"{args.out_dir}/constraint_checker_metrics_by_type.csv", by_group(merged, "constraint_type"), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/constraint_checker_metrics_by_scene_context_type.csv", by_group(merged, "scene_context_type"), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/constraint_checker_metrics_by_model.csv", by_group(merged, "model_name"), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/constraint_confusion_matrix.csv", confusion_matrix(merged), CONFUSION_FIELDS)
    write_csv(f"{args.out_dir}/image_level_labels.csv", image_rows, IMAGE_FIELDS)
    write_csv(f"{args.out_dir}/image_metrics_overall.csv", [rates(image_label_rows)], METRIC_FIELDS)
    write_csv(f"{args.out_dir}/image_metrics_by_prompt_family.csv", by_group(image_label_rows, "prompt_family"), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/image_metrics_by_scene_context_type.csv", by_group(image_label_rows, "scene_context_type"), METRIC_FIELDS)
    write_csv(f"{args.out_dir}/image_metrics_by_model.csv", by_group(image_label_rows, "model_name"), METRIC_FIELDS)
    print(f"Analyzed {len(merged)} constraint rows and {len(image_rows)} images into {args.out_dir}")



# Backward-compatible helper names used by the earlier checkpoint tests.
def overall_agreement(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r.get("human_label") == r.get("vlm_label") for r in rows), len(rows))


def human_pass_rate(rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r.get("human_label") == "pass" for r in rows), len(rows))


def failure_detection_metrics(rows: List[Dict[str, str]], group_name: str = "overall") -> Dict[str, str]:
    row = rates(rows, group_name)
    # Keep old field names as aliases.
    row["failure_precision"] = row["nonpass_precision"]
    row["failure_recall"] = row["nonpass_recall"]
    row["failure_f1"] = row["nonpass_f1"]
    return row


def constraint_metrics_by_type(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = by_group(rows, "constraint_type")
    for r in out:
        r["failure_precision"] = r["nonpass_precision"]
        r["failure_recall"] = r["nonpass_recall"]
        r["failure_f1"] = r["nonpass_f1"]
    return out


def image_level_agreement(image_rows: List[Dict[str, str]]) -> float:
    return safe_div(sum(r.get("human_image_label") == r.get("vlm_image_label") for r in image_rows), len(image_rows))


def image_controllability_metrics_by_dimension(image_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows = image_rows_as_label_rows(image_rows)
    out = []
    for key in ["prompt_family", "scene_context_type", "composition", "model_name"]:
        out.extend(by_group(rows, key))
    for r in out:
        try:
            r["num_examples"] = int(r["num_examples"])
        except Exception:
            pass
    return out


def metrics_by_cross_group(rows: List[Dict[str, str]], keys: List[str]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for r in rows:
        group_key = tuple(r.get(k, "unknown") or "unknown" for k in keys)
        grouped[group_key].append(r)
    out = []
    for group_key, rs in sorted(grouped.items()):
        row = rates(rs, ",".join(f"{k}={v}" for k, v in zip(keys, group_key)))
        for k, v in zip(keys, group_key):
            row[k] = v
        out.append(row)
    return out

if __name__ == "__main__":
    main()
