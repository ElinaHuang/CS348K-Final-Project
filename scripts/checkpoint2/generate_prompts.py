from __future__ import annotations

import argparse
import random
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def get_seed(config: Dict, cli_seed: int | None = None) -> int:
    if cli_seed is not None:
        return cli_seed
    if "random_seed" in config:
        return int(config["random_seed"])
    return 348


def scene_options(config: Dict) -> List[Tuple[str, str, str]]:
    """Return all scene options as (scene_type, scene_context, prefix)."""
    options: List[Tuple[str, str, str]] = []
    for scene_type in ["simple", "natural"]:
        scene_cfg = config["scene_contexts"][scene_type]
        prefix = scene_cfg["prefix"]
        for context in scene_cfg["contexts"]:
            options.append((scene_type, context, prefix))
    return options


def object_pairs(objects: List[str]) -> List[Tuple[str, str]]:
    return [(obj1, obj2) for obj1 in objects for obj2 in objects if obj1 != obj2]


def color_pairs(colors: List[str]) -> List[Tuple[str, str]]:
    if len(colors) <= 1:
        return [(c, c) for c in colors]
    return [(c1, c2) for c1 in colors for c2 in colors if c1 != c2]


def sample_unique(
    candidates: List[Dict],
    n: int,
    rng: random.Random,
    prompt_key: str = "prompt",
) -> List[Dict]:
    """Randomly sample n unique prompt candidates.

    This avoids two earlier failure modes:
    1. deterministic modulo cycling could skip some contexts forever;
    2. taking the first n Cartesian-product candidates produced low diversity
       because early dimensions changed slowly.

    We deduplicate by prompt text, shuffle candidates with a fixed seed, and
    then take the first n unique candidates. This gives diversity while keeping
    results reproducible.
    """
    unique_by_prompt: Dict[str, Dict] = {}
    for c in candidates:
        key = c[prompt_key]
        if key not in unique_by_prompt:
            unique_by_prompt[key] = c

    unique_candidates = list(unique_by_prompt.values())
    rng.shuffle(unique_candidates)

    if len(unique_candidates) < n:
        raise ValueError(
            f"Only generated {len(unique_candidates)} unique prompts, but requested {n}. "
            "Increase the grammar vocabulary or reduce the requested count."
        )

    return unique_candidates[:n]


def sort_for_stable_ids(selected: List[Dict]) -> List[Dict]:
    """Give selected prompts stable ordering after random sampling.

    Sampling controls which prompts appear; sorting makes IDs deterministic and
    easier to inspect. Scene type is included so IDs still reflect simple/natural.
    """
    return sorted(
        selected,
        key=lambda x: (
            x.get("scene_type", ""),
            x.get("scene_context", ""),
            x.get("prompt", ""),
        ),
    )


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


def generate_spatial_single(config: Dict, n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    relations = list(config["relations"].keys())
    scenes = scene_options(config)
    template = config["templates"]["spatial_single"]

    candidates = []
    for (obj1, obj2), rel, (scene_type, scene_context, prefix) in product(object_pairs(objects), relations, scenes):
        rel_text = config["relations"][rel]["text"]
        prompt = template.format(
            prefix=prefix,
            object_1=obj1,
            relation_text=rel_text,
            object_2=obj2,
            scene_context=scene_context,
        )
        candidates.append({
            "prompt": prompt,
            "obj1": obj1,
            "obj2": obj2,
            "rel": rel,
            "rel_text": rel_text,
            "scene_type": scene_type,
            "scene_context": scene_context,
        })

    selected = sort_for_stable_ids(sample_unique(candidates, n, rng))

    for i, item in enumerate(selected):
        prompt_id = f"spatial_{item['scene_type']}_single_{i+1:03d}"
        prompt_row = {
            "prompt_id": prompt_id,
            "prompt": item["prompt"],
            "prompt_family": "spatial_layout",
            "scene_context_type": item["scene_type"],
            "scene_context": item["scene_context"],
            "composition": "single",
        }
        cons = [
            make_object_existence_constraint(prompt_id, item["obj1"], "object_1"),
            make_object_existence_constraint(prompt_id, item["obj2"], "object_2"),
            make_spatial_constraint(prompt_id, item["obj1"], item["obj2"], item["rel"], item["rel_text"]),
        ]
        add_prompt(prompts, constraints, prompt_row, cons)

    return prompts, constraints


def generate_cardinality_single(config: Dict, n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    numbers = config["numbers"]
    scenes = scene_options(config)
    template = config["templates"]["cardinality_single"]

    candidates = []
    for obj, count, (scene_type, scene_context, prefix) in product(objects, numbers, scenes):
        prompt = template.format(
            prefix=prefix,
            number=count,
            object_plural=pluralize(obj),
            scene_context=scene_context,
        )
        candidates.append({
            "prompt": prompt,
            "obj": obj,
            "count": count,
            "scene_type": scene_type,
            "scene_context": scene_context,
        })

    selected = sort_for_stable_ids(sample_unique(candidates, n, rng))

    for i, item in enumerate(selected):
        prompt_id = f"cardinality_{item['scene_type']}_single_{i+1:03d}"
        prompt_row = {
            "prompt_id": prompt_id,
            "prompt": item["prompt"],
            "prompt_family": "cardinality",
            "scene_context_type": item["scene_type"],
            "scene_context": item["scene_context"],
            "composition": "single",
        }
        cons = [
            make_object_existence_constraint(prompt_id, item["obj"], "target"),
            make_cardinality_constraint(prompt_id, item["obj"], item["count"]),
        ]
        add_prompt(prompts, constraints, prompt_row, cons)

    return prompts, constraints


def generate_attribute_single(config: Dict, n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    colors = config["colors"]
    scenes = scene_options(config)
    template = config["templates"]["attribute_single"]

    candidates = []
    for (obj1, obj2), (c1, c2), (scene_type, scene_context, prefix) in product(object_pairs(objects), color_pairs(colors), scenes):
        prompt = template.format(
            prefix=prefix,
            color_1=c1,
            object_1=obj1,
            color_2=c2,
            object_2=obj2,
            scene_context=scene_context,
        )
        candidates.append({
            "prompt": prompt,
            "obj1": obj1,
            "obj2": obj2,
            "c1": c1,
            "c2": c2,
            "scene_type": scene_type,
            "scene_context": scene_context,
        })

    selected = sort_for_stable_ids(sample_unique(candidates, n, rng))

    for i, item in enumerate(selected):
        prompt_id = f"attribute_{item['scene_type']}_single_{i+1:03d}"
        prompt_row = {
            "prompt_id": prompt_id,
            "prompt": item["prompt"],
            "prompt_family": "attribute_binding",
            "scene_context_type": item["scene_type"],
            "scene_context": item["scene_context"],
            "composition": "single",
        }
        cons = [
            make_object_existence_constraint(prompt_id, item["obj1"], "object_1"),
            make_object_existence_constraint(prompt_id, item["obj2"], "object_2"),
            make_attribute_constraint(prompt_id, item["obj1"], item["c1"], "object_1"),
            make_attribute_constraint(prompt_id, item["obj2"], item["c2"], "object_2"),
        ]
        add_prompt(prompts, constraints, prompt_row, cons)

    return prompts, constraints


def generate_combined_attribute_spatial(config: Dict, n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    colors = config["colors"]
    relations = list(config["relations"].keys())
    scenes = scene_options(config)
    template = config["templates"]["combined_attribute_spatial"]

    candidates = []
    for (obj1, obj2), (c1, c2), rel, (scene_type, scene_context, prefix) in product(
        object_pairs(objects), color_pairs(colors), relations, scenes
    ):
        rel_text = config["relations"][rel]["text"]
        prompt = template.format(
            prefix=prefix,
            color_1=c1,
            object_1=obj1,
            relation_text=rel_text,
            color_2=c2,
            object_2=obj2,
            scene_context=scene_context,
        )
        candidates.append({
            "prompt": prompt,
            "obj1": obj1,
            "obj2": obj2,
            "c1": c1,
            "c2": c2,
            "rel": rel,
            "rel_text": rel_text,
            "scene_type": scene_type,
            "scene_context": scene_context,
        })

    selected = sort_for_stable_ids(sample_unique(candidates, n, rng))

    for i, item in enumerate(selected):
        prompt_id = f"combo_attr_spatial_{item['scene_type']}_{i+1:03d}"
        prompt_row = {
            "prompt_id": prompt_id,
            "prompt": item["prompt"],
            "prompt_family": "combined_attribute_spatial",
            "scene_context_type": item["scene_type"],
            "scene_context": item["scene_context"],
            "composition": "combined",
        }
        cons = [
            make_object_existence_constraint(prompt_id, item["obj1"], "object_1"),
            make_object_existence_constraint(prompt_id, item["obj2"], "object_2"),
            make_attribute_constraint(prompt_id, item["obj1"], item["c1"], "object_1"),
            make_attribute_constraint(prompt_id, item["obj2"], item["c2"], "object_2"),
            make_spatial_constraint(prompt_id, item["obj1"], item["obj2"], item["rel"], item["rel_text"]),
        ]
        add_prompt(prompts, constraints, prompt_row, cons)

    return prompts, constraints


def generate_combined_cardinality_spatial(config: Dict, n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    prompts, constraints = [], []
    objects = config["objects"]
    numbers = config["numbers"]
    relations = list(config["relations"].keys())
    scenes = scene_options(config)
    template = config["templates"]["combined_cardinality_spatial"]

    candidates = []
    for (obj1, obj2), count, rel, (scene_type, scene_context, prefix) in product(
        object_pairs(objects), numbers, relations, scenes
    ):
        rel_text = config["relations"][rel]["text"]
        prompt = template.format(
            prefix=prefix,
            number=count,
            object_1_plural=pluralize(obj1),
            relation_text=rel_text,
            object_2=obj2,
            scene_context=scene_context,
        )
        candidates.append({
            "prompt": prompt,
            "obj1": obj1,
            "obj2": obj2,
            "count": count,
            "rel": rel,
            "rel_text": rel_text,
            "scene_type": scene_type,
            "scene_context": scene_context,
        })

    selected = sort_for_stable_ids(sample_unique(candidates, n, rng))

    for i, item in enumerate(selected):
        prompt_id = f"combo_card_spatial_{item['scene_type']}_{i+1:03d}"
        prompt_row = {
            "prompt_id": prompt_id,
            "prompt": item["prompt"],
            "prompt_family": "combined_cardinality_spatial",
            "scene_context_type": item["scene_type"],
            "scene_context": item["scene_context"],
            "composition": "combined",
        }
        cons = [
            make_object_existence_constraint(prompt_id, item["obj1"], "object_1"),
            make_object_existence_constraint(prompt_id, item["obj2"], "object_2"),
            make_cardinality_constraint(prompt_id, item["obj1"], item["count"]),
            make_spatial_constraint(prompt_id, item["obj1"], item["obj2"], item["rel"], item["rel_text"]),
        ]
        add_prompt(prompts, constraints, prompt_row, cons)

    return prompts, constraints


def generate_all(config: Dict, seed: int | None = None) -> Tuple[List[Dict], List[Dict]]:
    counts = config.get("checkpoint1_counts", {})
    base_seed = get_seed(config, seed)

    # Use separate RNG streams per family so changing one family count does not
    # completely reshuffle all later families.
    family_specs = [
        ("spatial_single", generate_spatial_single),
        ("cardinality_single", generate_cardinality_single),
        ("attribute_single", generate_attribute_single),
        ("combined_attribute_spatial", generate_combined_attribute_spatial),
        ("combined_cardinality_spatial", generate_combined_cardinality_spatial),
    ]

    all_prompts, all_constraints = [], []
    for family_idx, (key, fn) in enumerate(family_specs):
        n = int(counts.get(key, 0))
        if n <= 0:
            continue
        rng = random.Random(base_seed + family_idx * 1009)
        prompts, constraints = fn(config, n, rng)
        all_prompts.extend(prompts)
        all_constraints.extend(constraints)

    return all_prompts, all_constraints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/grammar_config.yaml")
    parser.add_argument("--out-prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--out-constraints", default="../data/prompts/constraints.csv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    prompts, constraints = generate_all(config, seed=args.seed)
    write_csv(args.out_prompts, prompts, PROMPT_FIELDS)
    write_csv(args.out_constraints, constraints, CONSTRAINT_FIELDS)
    print(f"Wrote {len(prompts)} prompts to {args.out_prompts}")
    print(f"Wrote {len(constraints)} constraints to {args.out_constraints}")


if __name__ == "__main__":
    main()
