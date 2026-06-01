from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from utils import normalize_label, read_csv, write_csv, pluralize

REPAIRED_PROMPT_FIELDS = [
    "repair_id", "prompt_id", "prompt",
    "source_image_id", "source_prompt_id", "source_generation_job_id",
    "source_provider", "source_model_name", "source_image_path",
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


def index(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {r[key]: r for r in rows if r.get(key)}


def repair_action_for_label(label: str) -> str:
    label = normalize_label(label)
    return "strengthen_failed_constraint" if label == "fail" else "clarify_ambiguous_constraint"


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def readable_attribute_category(category: str) -> str:
    if category == "pattern_texture":
        return "pattern"
    if category == "material_finish":
        return "material"
    return "visual attribute"


def get_attribute_and_category(c: Dict[str, str]) -> tuple[str, str]:
    attr = (
        c.get("attribute_1", "").strip()
        or c.get("attribute", "").strip()
        or c.get("attribute_2", "").strip()
    )
    category = (
        c.get("attribute_1_category", "").strip()
        or c.get("attribute_category", "").strip()
        or c.get("attribute_2_category", "").strip()
    )
    return attr, category


def constraint_instruction(c: Dict[str, str], before_label: str) -> str:
    ctype = c.get("constraint_type", "")
    label = normalize_label(before_label)

    object_1 = c.get("object_1", "").strip()
    object_2 = c.get("object_2", "").strip()
    relation_text = c.get("relation_text", "").strip()
    target_object = c.get("target_object", "").strip()
    target_object_plural = c.get("target_object_plural", "").strip()
    object_1_plural = c.get("object_1_plural", "").strip()
    target_count = c.get("target_count", "").strip()
    attribute = c.get("attribute", "").strip()
    attribute_1 = c.get("attribute_1", "").strip()

    # Failed constraints ask for correctness. Ambiguous constraints ask for clarity
    # and easy visual judgment.
    if label == "ambiguous":
        action = "Make it clear and easy to judge that"
    else:
        action = "Make sure that"

    if ctype == "object_identity":
        obj = _first_nonempty(target_object, object_1, object_2, "the requested object")
        if label == "ambiguous":
            return (
                f"{action} the object generated is clearly recognizable as {obj}, "
                f"not a vague or ambiguous object-like shape."
            )
        return (
            f"{action} the object generated is exactly {obj}; "
            f"do not replace it with a different object category."
        )

    if ctype == "cardinality":
        obj = _first_nonempty(target_object_plural, object_1_plural, target_object, object_1, "target objects")
        count_phrase = f"exactly {target_count}" if target_count else "the requested number of"
        if label == "ambiguous":
            return (
                f"{action} there are {count_phrase} {pluralize(obj)}, "
                f"with each target object separated and easy to count."
            )
        return (
            f"{action} there are {count_phrase} {pluralize(obj)}; "
            f"do not generate more or fewer, and do not add extra target-like objects."
        )

    if ctype == "attribute":
        obj = object_1 or target_object
        attr, attr_category = get_attribute_and_category(c)
        readable_category = readable_attribute_category(attr_category)

        if label == "ambiguous":
            return (
                f"{action} the {obj} has {attr} {readable_category}."
            )

        return (
            f"{action} the {obj} has {attr} {readable_category}; "
            f"apply this {readable_category} to the correct object."
        )

    if ctype == "spatial_relation":
        obj1 = _first_nonempty(object_1, "the first object")
        obj2 = _first_nonempty(object_2, "the second object")
        relation = _first_nonempty(relation_text, "in the requested relation to")
        if label == "ambiguous":
            return (
                f"{action} the {obj1} is {relation} the {obj2}, "
                f"with both objects visible and the spatial relation easy to judge."
            )
        return f"{action} the {obj1} is clearly {relation} the {obj2}."

    check = c.get("check_text", "").strip().rstrip(".")
    if label == "ambiguous":
        return f"{action} this requirement is visually clear: {check}."
    return f"{action} this requirement is satisfied: {check}."


def build_repaired_prompt(
    original_prompt: str,
    original_constraints: List[Dict[str, str]],
    targets: Dict[str, str],
) -> str:
    # Repair prompts keep the original generation request first, then add
    # targeted instructions for failed or ambiguous constraints. Object identity
    # comes first because other constraints are easier to repair once the objects
    # are recognizable.
    priority = {
        "object_identity": 0,
        "cardinality": 1,
        "attribute": 2,
        "spatial_relation": 3,
    }

    target_constraints = [
        c for c in original_constraints
        if c.get("constraint_id") in targets
    ]
    target_constraints.sort(
        key=lambda c: (
            priority.get(c.get("constraint_type", ""), 99),
            c.get("constraint_id", ""),
        )
    )

    lines = [
        original_prompt.strip(),
        "",
        "Important generation instructions:",
    ]

    for c in target_constraints:
        cid = c["constraint_id"]
        lines.append(f"- {constraint_instruction(c, targets[cid])}")

    return "\n".join(lines)

def label_file_for_source(config: Dict, source: str) -> str:
    if source == "human":
        return config["human_labels"]["labels_csv"]
    if source == "vlm":
        return config["vlm_checker"]["labels_csv"]
    raise ValueError(f"Unknown trigger source: {source}")


def label_column_for_source(source: str) -> str:
    return "human_label" if source == "human" else "vlm_label"


def generation_by_image(config: Dict) -> Dict[str, Dict[str, str]]:
    gen_csv = config.get("generation", {}).get("generations_csv", "../data/generations/generations.csv")
    return index(read_csv(gen_csv), "image_id")


def generate_repair_records(config: Dict, trigger_source: str | None = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    prompt_cfg, repair_cfg = config["prompt_subset"], config["repair"]
    source = trigger_source or repair_cfg.get("trigger_label_source", "human")
    label_col = label_column_for_source(source)
    prompts = read_csv(prompt_cfg["out_prompts"])
    constraints = read_csv(prompt_cfg["out_constraints"])
    labels = read_csv(label_file_for_source(config, source))
    prompt_by_id = index(prompts, "prompt_id")
    constraint_by_id = index(constraints, "constraint_id")
    gen_by_image = generation_by_image(config)
    constraints_by_prompt: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in constraints:
        constraints_by_prompt[c["prompt_id"]].append(c)
    trigger_labels = {normalize_label(x) for x in repair_cfg.get("trigger_labels", ["fail", "ambiguous"])}
    labels_by_image: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in labels:
        if row.get(label_col, "").strip():
            labels_by_image[row["image_id"]].append(row)
    repaired_prompts, repair_targets = [], []
    count = 0
    for image_id, image_labels in sorted(labels_by_image.items()):
        prompt_id = image_labels[0]["prompt_id"]
        targets = [r for r in image_labels if normalize_label(r.get(label_col, "")) in trigger_labels and r.get("constraint_id") in constraint_by_id]
        if not targets:
            continue
        count += 1
        repair_id = f"repair_{source}_{count:03d}_{image_id}"
        target_labels = {r["constraint_id"]: normalize_label(r.get(label_col, "")) for r in targets}
        original_constraints = constraints_by_prompt[prompt_id]
        original_prompt = prompt_by_id.get(prompt_id, {}).get("prompt", "")
        repaired_prompt = build_repaired_prompt(original_prompt, original_constraints, target_labels)
        source_gen = gen_by_image.get(image_id, {})
        repaired_prompts.append({
            "repair_id": repair_id,
            "prompt_id": repair_id,
            "prompt": repaired_prompt,
            "source_image_id": image_id,
            "source_prompt_id": prompt_id,
            "source_generation_job_id": source_gen.get("generation_job_id", ""),
            "source_provider": source_gen.get("provider", ""),
            "source_model_name": source_gen.get("model_name", ""),
            "source_image_path": source_gen.get("image_path", ""),
            "repair_strategy": "repair_all_failed",
            "trigger_label_source": source,
            "trigger_labels": ";".join(sorted(trigger_labels)),
            "num_target_constraints": str(len(targets)),
            "target_constraint_ids": ";".join(r["constraint_id"] for r in targets),
            "original_prompt": prompt_by_id.get(prompt_id, {}).get("prompt", ""),
            "repaired_prompt": repaired_prompt,
        })
        for r in targets:
            c = constraint_by_id[r["constraint_id"]]
            before = normalize_label(r.get(label_col, ""))
            repair_targets.append({
                "repair_id": repair_id,
                "source_image_id": image_id,
                "source_prompt_id": prompt_id,
                "target_constraint_id": r["constraint_id"],
                "target_constraint_type": c.get("constraint_type", ""),
                "before_label": before,
                "repair_action": repair_action_for_label(before),
                "target_constraint_text": c.get("check_text", ""),
            })
    return repaired_prompts, repair_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/final_experiment_config.yaml")
    parser.add_argument("--trigger-source", choices=["human", "vlm"], default=None)
    parser.add_argument("--out-prompts", default=None)
    parser.add_argument("--out-targets", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    repaired_prompts, repair_targets = generate_repair_records(config, trigger_source=args.trigger_source)
    source = args.trigger_source or config["repair"].get("trigger_label_source", "human")
    out_prompts = args.out_prompts or config["repair"].get(f"repaired_prompts_{source}_csv") or config["repair"]["repaired_prompts_csv"]
    out_targets = args.out_targets or config["repair"].get(f"repair_targets_{source}_csv") or config["repair"]["repair_targets_csv"]
    write_csv(out_prompts, repaired_prompts, REPAIRED_PROMPT_FIELDS)
    write_csv(out_targets, repair_targets, REPAIR_TARGET_FIELDS)
    print(f"Wrote {len(repaired_prompts)} {source}-triggered repaired prompts to {out_prompts}")
    print(f"Wrote {len(repair_targets)} repair targets to {out_targets}")


if __name__ == "__main__":
    main()
