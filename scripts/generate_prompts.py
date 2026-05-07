from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from utils import pluralize, write_csv

PROMPT_FIELDS = [
    "prompt_id",
    "prompt",
    "prompt_family",
    "scene_context_type",
    "scene_context",
    "composition",
]

CONSTRAINT_FIELDS = [
    "constraint_id",
    "prompt_id",
    "constraint_type",
    "check_text",
    "object_1",
    "object_2",
    "relation",
    "relation_text",
    "target_object",
    "target_count",
    "attribute",
    "attribute_1",
    "attribute_2",
]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_scene(config: Dict, idx: int) -> Tuple[str, str, str]:
    scene_type = "simple" if idx % 2 == 0 else "natural"
    scene_cfg = config["scene_contexts"][scene_type]
    contexts = scene_cfg["contexts"]
    context = contexts[idx % len(contexts)]
    prefix = scene_cfg["prefix"]
    return scene_type, context, prefix


def choose_pair(objects: List[str], idx: int) -> Tuple[str, str]:
    obj1 = objects[idx % len(objects)]
    obj2 = objects[(idx + 1) % len(objects)]
    if obj1 == obj2:
        obj2 = objects[(idx + 2) % len(objects)]
    return obj1, obj2


def make_object_existence_constraint(prompt_id: str, obj: str, suffix: str) -> Dict:
    return {
        "constraint_id": f"{prompt_id}_exist_{suffix}",
        "prompt_id": prompt_id,
        "constraint_type": "object_existence",
        "check_text": f"The {obj} should be clearly visible.",
        "object_1": obj,
        "object_2": "",
        "relation": "",
        "relation_text": "",
        "target_object": obj,
        "target_count": "",
        "attribute": "",
        "attribute_1": "",
        "attribute_2": "",
    }


def make_cardinality_constraint(prompt_id: str, obj: str, count: int) -> Dict:
    return {
        "constraint_id": f"{prompt_id}_cardinality",
        "prompt_id": prompt_id,
        "constraint_type": "cardinality",
        "check_text": f"There should be exactly {count} clearly visible {pluralize(obj)}.",
        "object_1": "",
        "object_2": "",
        "relation": "",
        "relation_text": "",
        "target_object": obj,
        "target_count": str(count),
        "attribute": "",
        "attribute_1": "",
        "attribute_2": "",
    }


def make_attribute_constraint(prompt_id: str, obj: str, color: str, suffix: str) -> Dict:
    return {
        "constraint_id": f"{prompt_id}_attr_{suffix}",
        "prompt_id": prompt_id,
        "constraint_type": "attribute",
        "check_text": f"The {obj} should be {color}.",
        "object_1": obj,
        "object_2": "",
        "relation": "",
        "relation_text": "",
        "target_object": obj,
        "target_count": "",
        "attribute": color,
        "attribute_1": color,
        "attribute_2": "",
    }


def make_spatial_constraint(prompt_id: str, obj1: str, obj2: str, relation: str, relation_text: str) -> Dict:
    return {
        "constraint_id": f"{prompt_id}_spatial",
        "prompt_id": prompt_id,
        "constraint_type": "spatial_relation",
        "check_text": f"The {obj1} should be {relation_text} the {obj2} in the 2D image.",
        "object_1": obj1,
        "object_2": obj2,
        "relation": relation,
        "relation_text": relation_text,
        "target_object": "",
        "target_count": "",
        "attribute": "",
        "attribute_1": "",
        "attribute_2": "",
    }


def add_prompt(rows_p: List[Dict], rows_c: List[Dict], prompt_row: Dict, constraints: List[Dict]) -> None:
    rows_p.append(prompt_row)
    rows_c.extend(constraints)


def generate_spatial_single(config: Dict, n: int) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    relations = list(config["relations"].keys())
    template = config["templates"]["spatial_single"]

    for i in range(n):
        obj1, obj2 = choose_pair(objects, i)
        rel = relations[i % len(relations)]
        rel_text = config["relations"][rel]["text"]
        scene_type, scene_context, prefix = choose_scene(config, i)
        prompt_id = f"spatial_{scene_type}_single_{i+1:03d}"
        prompt = template.format(prefix=prefix, object_1=obj1, relation_text=rel_text, object_2=obj2, scene_context=scene_context)
        prompt_row = {"prompt_id": prompt_id, "prompt": prompt, "prompt_family": "spatial_layout", "scene_context_type": scene_type, "scene_context": scene_context, "composition": "single"}
        cons = [make_object_existence_constraint(prompt_id, obj1, "object_1"), make_object_existence_constraint(prompt_id, obj2, "object_2"), make_spatial_constraint(prompt_id, obj1, obj2, rel, rel_text)]
        add_prompt(prompts, constraints, prompt_row, cons)
    return prompts, constraints


def generate_cardinality_single(config: Dict, n: int) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    numbers = config["numbers"]
    template = config["templates"]["cardinality_single"]

    for i in range(n):
        obj = objects[i % len(objects)]
        count = numbers[i % len(numbers)]
        scene_type, scene_context, prefix = choose_scene(config, i)
        prompt_id = f"cardinality_{scene_type}_single_{i+1:03d}"
        prompt = template.format(prefix=prefix, number=count, object_plural=pluralize(obj), scene_context=scene_context)
        prompt_row = {"prompt_id": prompt_id, "prompt": prompt, "prompt_family": "cardinality", "scene_context_type": scene_type, "scene_context": scene_context, "composition": "single"}
        cons = [make_object_existence_constraint(prompt_id, obj, "target"), make_cardinality_constraint(prompt_id, obj, count)]
        add_prompt(prompts, constraints, prompt_row, cons)
    return prompts, constraints


def generate_attribute_single(config: Dict, n: int) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    colors = config["colors"]
    template = config["templates"]["attribute_single"]

    for i in range(n):
        obj1, obj2 = choose_pair(objects, i)
        c1 = colors[i % len(colors)]
        c2 = colors[(i + 1) % len(colors)]
        scene_type, scene_context, prefix = choose_scene(config, i)
        prompt_id = f"attribute_{scene_type}_single_{i+1:03d}"
        prompt = template.format(prefix=prefix, color_1=c1, object_1=obj1, color_2=c2, object_2=obj2, scene_context=scene_context)
        prompt_row = {"prompt_id": prompt_id, "prompt": prompt, "prompt_family": "attribute_binding", "scene_context_type": scene_type, "scene_context": scene_context, "composition": "single"}
        cons = [make_object_existence_constraint(prompt_id, obj1, "object_1"), make_object_existence_constraint(prompt_id, obj2, "object_2"), make_attribute_constraint(prompt_id, obj1, c1, "object_1"), make_attribute_constraint(prompt_id, obj2, c2, "object_2")]
        add_prompt(prompts, constraints, prompt_row, cons)
    return prompts, constraints


def generate_combined_attribute_spatial(config: Dict, n: int) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    colors = config["colors"]
    relations = list(config["relations"].keys())
    template = config["templates"]["combined_attribute_spatial"]

    for i in range(n):
        obj1, obj2 = choose_pair(objects, i)
        c1 = colors[i % len(colors)]
        c2 = colors[(i + 1) % len(colors)]
        rel = relations[i % len(relations)]
        rel_text = config["relations"][rel]["text"]
        scene_type, scene_context, prefix = choose_scene(config, i)
        prompt_id = f"combo_attr_spatial_{scene_type}_{i+1:03d}"
        prompt = template.format(prefix=prefix, color_1=c1, object_1=obj1, relation_text=rel_text, color_2=c2, object_2=obj2, scene_context=scene_context)
        prompt_row = {"prompt_id": prompt_id, "prompt": prompt, "prompt_family": "combined_attribute_spatial", "scene_context_type": scene_type, "scene_context": scene_context, "composition": "combined"}
        cons = [make_object_existence_constraint(prompt_id, obj1, "object_1"), make_object_existence_constraint(prompt_id, obj2, "object_2"), make_attribute_constraint(prompt_id, obj1, c1, "object_1"), make_attribute_constraint(prompt_id, obj2, c2, "object_2"), make_spatial_constraint(prompt_id, obj1, obj2, rel, rel_text)]
        add_prompt(prompts, constraints, prompt_row, cons)
    return prompts, constraints


def generate_combined_cardinality_spatial(config: Dict, n: int) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    numbers = config["numbers"]
    relations = list(config["relations"].keys())
    template = config["templates"]["combined_cardinality_spatial"]

    for i in range(n):
        obj1, obj2 = choose_pair(objects, i)
        count = numbers[i % len(numbers)]
        rel = relations[i % len(relations)]
        rel_text = config["relations"][rel]["text"]
        scene_type, scene_context, prefix = choose_scene(config, i)
        prompt_id = f"combo_card_spatial_{scene_type}_{i+1:03d}"
        prompt = template.format(prefix=prefix, number=count, object_1_plural=pluralize(obj1), relation_text=rel_text, object_2=obj2, scene_context=scene_context)
        prompt_row = {"prompt_id": prompt_id, "prompt": prompt, "prompt_family": "combined_cardinality_spatial", "scene_context_type": scene_type, "scene_context": scene_context, "composition": "combined"}
        cons = [make_object_existence_constraint(prompt_id, obj1, "object_1"), make_object_existence_constraint(prompt_id, obj2, "object_2"), make_cardinality_constraint(prompt_id, obj1, count), make_spatial_constraint(prompt_id, obj1, obj2, rel, rel_text)]
        add_prompt(prompts, constraints, prompt_row, cons)
    return prompts, constraints


def generate_all(config: Dict) -> Tuple[List[Dict], List[Dict]]:
    counts = config.get("checkpoint1_counts", {})
    all_prompts, all_constraints = [], []
    for key, fn in [
        ("spatial_single", generate_spatial_single),
        ("cardinality_single", generate_cardinality_single),
        ("attribute_single", generate_attribute_single),
        ("combined_attribute_spatial", generate_combined_attribute_spatial),
        ("combined_cardinality_spatial", generate_combined_cardinality_spatial),
    ]:
        n = int(counts.get(key, 0))
        if n > 0:
            prompts, constraints = fn(config, n)
            all_prompts.extend(prompts)
            all_constraints.extend(constraints)
    return all_prompts, all_constraints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grammar_config.yaml")
    parser.add_argument("--out-prompts", default="data/prompts/prompts.csv")
    parser.add_argument("--out-constraints", default="data/prompts/constraints.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts, constraints = generate_all(config)
    write_csv(args.out_prompts, prompts, PROMPT_FIELDS)
    write_csv(args.out_constraints, constraints, CONSTRAINT_FIELDS)
    print(f"Wrote {len(prompts)} prompts to {args.out_prompts}")
    print(f"Wrote {len(constraints)} constraints to {args.out_constraints}")


if __name__ == "__main__":
    main()
