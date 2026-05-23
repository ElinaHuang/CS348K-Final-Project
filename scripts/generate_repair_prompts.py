from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from utils import read_csv, write_csv

REPAIRED_PROMPT_FIELDS = [
    "repair_id", "prompt_id", "prompt", "source_image_id", "source_prompt_id",
    "repair_strategy", "trigger_label_source", "trigger_labels", "num_target_constraints",
    "target_constraint_ids", "original_prompt", "repaired_prompt",
]

REPAIR_TARGET_FIELDS = [
    "repair_id", "source_image_id", "source_prompt_id", "target_constraint_id",
    "target_constraint_type", "before_label", "repair_action", "target_constraint_text",
]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def index_by_key(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {r[key]: r for r in rows if r.get(key)}


def normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    return label if label in {"pass", "fail", "ambiguous"} else "ambiguous"


def repair_action_for_label(label: str) -> str:
    label = normalize_label(label)
    if label == "fail":
        return "corrective_rewrite"
    if label == "ambiguous":
        return "clarification_rewrite"
    return "none"


def get_object_name(constraint: Dict[str, str]) -> str:
    return constraint.get("target_object") or constraint.get("object_1") or "the target object"


def strengthened_instruction(constraint: Dict[str, str], label: str) -> str:
    """Return a stronger instruction for a failed or ambiguous constraint."""
    ctype = constraint.get("constraint_type", "")
    check = constraint.get("check_text", "")
    label = normalize_label(label)

    if label == "ambiguous":
        if ctype == "object_existence":
            obj = get_object_name(constraint)
            return f"Make {obj} clearly visible as a distinct, unobstructed object. Avoid cropping, overlap, or ambiguous shapes."
        if ctype == "cardinality":
            obj = constraint.get("target_object", "target object")
            count = constraint.get("target_count", "the requested number")
            return f"Make exactly {count} clearly separated {obj}(s) visible. Avoid overlap or partial objects that make counting uncertain."
        if ctype == "attribute":
            obj = get_object_name(constraint)
            attr = constraint.get("attribute") or constraint.get("attribute_1") or "the requested attribute"
            return f"Make the {attr} attribute of {obj} visually clear and unambiguous. Avoid lighting or occlusion that makes the attribute hard to judge."
        if ctype == "spatial_relation":
            obj1 = constraint.get("object_1", "object 1")
            obj2 = constraint.get("object_2", "object 2")
            rel = constraint.get("relation_text") or constraint.get("relation", "the requested spatial relation")
            return f"Make the spatial relation unambiguous: {obj1} is clearly {rel} {obj2}. Use a front-facing layout with separated objects and no overlap."
        return f"Make this requirement visually clear and unambiguous: {check}"

    # fail -> corrective rewrite
    if ctype == "object_existence":
        obj = get_object_name(constraint)
        return f"The {obj} must be clearly visible as a distinct object in the image. Do not omit it."
    if ctype == "cardinality":
        obj = constraint.get("target_object", "target object")
        count = constraint.get("target_count", "the requested number")
        return f"There must be exactly {count} clearly visible {obj}(s) total. Do not include extra {obj}(s), and do not hide or merge any of them."
    if ctype == "attribute":
        obj = get_object_name(constraint)
        attr = constraint.get("attribute") or constraint.get("attribute_1") or "the requested attribute"
        return f"The {obj} must clearly have the {attr} attribute. Do not apply this attribute to the wrong object."
    if ctype == "spatial_relation":
        obj1 = constraint.get("object_1", "object 1")
        obj2 = constraint.get("object_2", "object 2")
        rel = constraint.get("relation_text") or constraint.get("relation", "the requested spatial relation")
        return f"The {obj1} must be clearly {rel} the {obj2} in the 2D image. Keep the two objects separated so the relation is easy to verify."
    return f"This requirement must be clearly satisfied: {check}"


def base_instruction(constraint: Dict[str, str]) -> str:
    return constraint.get("check_text", "").strip().rstrip(".") + "."


def build_repaired_prompt(
    original_constraints: List[Dict[str, str]],
    target_labels_by_constraint_id: Dict[str, str],
) -> str:
    """Build one image-level structured repair prompt from the full constraint set.

    Non-target constraints are preserved. Target failed/ambiguous constraints are strengthened.
    The original prompt text is not pasted into the repaired prompt; the prompt is reconstructed
    from the original constraint metadata.
    """
    lines = [
        "Create a clear image satisfying all of the following visual requirements:",
    ]

    for idx, c in enumerate(original_constraints, start=1):
        cid = c["constraint_id"]
        if cid in target_labels_by_constraint_id:
            instruction = strengthened_instruction(c, target_labels_by_constraint_id[cid])
        else:
            instruction = base_instruction(c)
        lines.append(f"{idx}. {instruction}")

    lines.extend([
        "",
        "Composition guidelines:",
        "- Use a clear, front-facing composition.",
        "- Keep all relevant objects visible and separated.",
        "- Avoid occlusion, heavy overlap, cropped objects, or ambiguous object shapes.",
        "- Do not introduce extra objects that could be confused with the requested objects.",
    ])
    return "\n".join(lines)


def group_labels_by_image(labels: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in labels:
        grouped[row["image_id"]].append(row)
    return grouped


def generate_repair_records(config: Dict) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    prompt_cfg = config["prompt_subset"]
    repair_cfg = config["repair"]
    label_cfg = config["human_labels"]

    prompts = read_csv(prompt_cfg["out_prompts"])
    constraints = read_csv(prompt_cfg["out_constraints"])
    labels = read_csv(label_cfg["labels_csv"])

    prompt_by_id = index_by_key(prompts, "prompt_id")
    constraints_by_prompt: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in constraints:
        constraints_by_prompt[c["prompt_id"]].append(c)
    constraint_by_id = index_by_key(constraints, "constraint_id")

    trigger_labels = {normalize_label(x) for x in repair_cfg.get("trigger_labels", ["fail"])}
    strategies = repair_cfg.get("strategies", ["repair_all_failed"])
    max_repairs = int(repair_cfg.get("max_repairs", 10))

    labels_by_image = group_labels_by_image(labels)
    repaired_prompts: List[Dict[str, str]] = []
    repair_targets: List[Dict[str, str]] = []
    repair_count = 0

    for image_id, image_labels in sorted(labels_by_image.items()):
        prompt_id = image_labels[0]["prompt_id"]
        original_prompt = prompt_by_id.get(prompt_id, {}).get("prompt", "")
        original_constraints = constraints_by_prompt.get(prompt_id, [])
        candidate_targets = [
            row for row in image_labels
            if normalize_label(row.get("human_label", "")) in trigger_labels
            and row.get("constraint_id") in constraint_by_id
        ]
        if not candidate_targets:
            continue

        for strategy in strategies:
            if repair_count >= max_repairs:
                break
            if strategy == "repair_all_failed":
                target_groups = [candidate_targets]
            elif strategy == "repair_single_target":
                target_groups = [[row] for row in candidate_targets]
            else:
                raise ValueError(f"Unknown repair strategy: {strategy}")

            for target_group in target_groups:
                if repair_count >= max_repairs:
                    break
                repair_count += 1
                repair_id = f"repair_{repair_count:03d}_{strategy}_{image_id}"
                target_ids = [row["constraint_id"] for row in target_group]
                target_labels_by_id = {row["constraint_id"]: normalize_label(row.get("human_label", "")) for row in target_group}
                repaired_prompt = build_repaired_prompt(original_constraints, target_labels_by_id)

                repaired_prompts.append({
                    "repair_id": repair_id,
                    "prompt_id": repair_id,
                    "prompt": repaired_prompt,
                    "source_image_id": image_id,
                    "source_prompt_id": prompt_id,
                    "repair_strategy": strategy,
                    "trigger_label_source": repair_cfg.get("trigger_label_source", "human"),
                    "trigger_labels": ";".join(sorted(trigger_labels)),
                    "num_target_constraints": str(len(target_ids)),
                    "target_constraint_ids": ";".join(target_ids),
                    "original_prompt": original_prompt,
                    "repaired_prompt": repaired_prompt,
                })

                for row in target_group:
                    c = constraint_by_id[row["constraint_id"]]
                    before_label = normalize_label(row.get("human_label", ""))
                    repair_targets.append({
                        "repair_id": repair_id,
                        "source_image_id": image_id,
                        "source_prompt_id": prompt_id,
                        "target_constraint_id": row["constraint_id"],
                        "target_constraint_type": row["constraint_type"],
                        "before_label": before_label,
                        "repair_action": repair_action_for_label(before_label),
                        "target_constraint_text": c.get("check_text", ""),
                    })

    return repaired_prompts, repair_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/checkpoint2_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    repaired_prompts, repair_targets = generate_repair_records(config)
    repair_cfg = config["repair"]

    write_csv(repair_cfg["repaired_prompts_csv"], repaired_prompts, REPAIRED_PROMPT_FIELDS)
    write_csv(repair_cfg["repair_targets_csv"], repair_targets, REPAIR_TARGET_FIELDS)
    print(f"Wrote {len(repaired_prompts)} repaired prompts to {repair_cfg['repaired_prompts_csv']}")
    print(f"Wrote {len(repair_targets)} repair targets to {repair_cfg['repair_targets_csv']}")


if __name__ == "__main__":
    main()
