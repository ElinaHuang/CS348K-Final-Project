from __future__ import annotations

import argparse
import random
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from utils import pluralize, slugify, write_csv

PROMPT_FIELDS = [
    "prompt_id", "prompt", "prompt_family", "scene_context_type", "scene_context", "composition",
]

CONSTRAINT_FIELDS = [
    "constraint_id", "prompt_id", "constraint_type", "check_text",
    "object_1", "object_2", "object_1_group", "object_2_group", "relation", "relation_text",
    "target_object", "target_object_group", "target_count", "attribute", "attribute_category",
    "attribute_1", "attribute_1_category", "attribute_2", "attribute_2_category",
    "object_slot", "attribute_slot",
]

GENERATION_PLAN_FIELDS = [
    "generation_job_id", "prompt_id", "provider", "model_name", "assignment_type",
    "image_dir", "size", "quality", "aspect_ratio",
]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_seed(config: Dict, cli_seed: int | None = None) -> int:
    return int(cli_seed if cli_seed is not None else config.get("random_seed", 348))


def scene_options(config: Dict) -> List[Dict[str, str]]:
    out = []
    for scene_type in ["simple", "natural"]:
        cfg = config["scene_contexts"][scene_type]
        for context in cfg["contexts"]:
            out.append({"scene_type": scene_type, "scene_context": context, "prefix": cfg["prefix"]})
    return out


def all_objects(config: Dict) -> List[Dict[str, str]]:
    out = []
    for group, names in config["object_groups"].items():
        for name in names:
            out.append({"object": name, "group": group})
    return out


def objects_from_groups(config: Dict, groups: Iterable[str]) -> List[Dict[str, str]]:
    group_set = set(groups)
    return [o for o in all_objects(config) if o["group"] in group_set]


def all_attributes(config: Dict) -> List[Dict[str, str]]:
    """Return all attributes with group/category metadata.

    Final grammar stores attributes under attribute_groups so the vocabulary has
    a logical structure. For backward compatibility, this helper also accepts
    the older per-attribute `attributes` format if present.
    """
    out = []
    if "attribute_groups" in config:
        for group_key, group_cfg in config["attribute_groups"].items():
            allowed_groups = group_cfg.get("allowed_object_groups", [])
            for attr in group_cfg.get("attributes", []):
                attr_key = attr.lower().replace(" ", "_").replace("-", "_")
                out.append({
                    "attribute_key": attr_key,
                    "attribute": attr,
                    "attribute_category": group_key,
                    "attribute_group": group_key,
                    "allowed_groups": allowed_groups,
                })
        return out

    # Backward-compatible fallback for older configs.
    for key, cfg in config.get("attributes", {}).items():
        out.append({
            "attribute_key": key,
            "attribute": cfg["text"],
            "attribute_category": cfg.get("category", "unknown"),
            "attribute_group": cfg.get("category", "unknown"),
            "allowed_groups": cfg.get("allowed_groups", []),
        })
    return out


def compatible_attributes(config: Dict, object_group: str) -> List[Dict[str, str]]:
    return [a for a in all_attributes(config) if object_group in a["allowed_groups"]]


def relation_pairs(config: Dict, relation_key: str) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    rel = config["relations"][relation_key]
    objs1 = objects_from_groups(config, rel["object_1_groups"])
    objs2 = objects_from_groups(config, rel["object_2_groups"])
    return [(o1, o2) for o1 in objs1 for o2 in objs2 if o1["object"] != o2["object"]]


def sample_balanced(candidates: List[Dict], n: int, rng: random.Random, balance_keys: List[str]) -> List[Dict]:
    """Deduplicate, then do lightweight stratified random sampling.

    The first balance key is used as the primary bucket. Within each bucket,
    candidates are shuffled. We then round-robin over buckets until n examples
    are selected. This avoids the old problem where some objects/relations never
    appeared while others repeated too often.
    """
    by_prompt = {}
    for c in candidates:
        by_prompt.setdefault(c["prompt"], c)
    unique = list(by_prompt.values())
    if len(unique) < n:
        raise ValueError(f"Only {len(unique)} unique candidates available, but {n} requested.")

    if not balance_keys:
        rng.shuffle(unique)
        return unique[:n]

    primary = balance_keys[0]
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for c in unique:
        buckets[str(c.get(primary, "unknown"))].append(c)
    for rows in buckets.values():
        rng.shuffle(rows)
    keys = list(buckets.keys())
    rng.shuffle(keys)

    selected = []
    while len(selected) < n and keys:
        progressed = False
        for key in list(keys):
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
                if len(selected) >= n:
                    break
            else:
                keys.remove(key)
        if not progressed:
            break
    if len(selected) < n:
        remaining = [c for rows in buckets.values() for c in rows]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    return selected[:n]


def sample_balanced_stream(candidates: Iterable[Dict], n: int, rng: random.Random, balance_keys: List[str]) -> List[Dict]:
    """Memory-safe stratified sampling from a large candidate iterator.

    Some typed grammar families have millions of valid combinations. Building the
    full list is unnecessary and can be slow or memory-heavy. This function keeps
    a bounded reservoir per primary balance bucket, then reuses sample_balanced
    on the retained candidates.
    """
    primary = balance_keys[0] if balance_keys else "balance_key"
    max_per_bucket = max(200, n * 10)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    seen_retained_prompts = set()

    for c in candidates:
        key = str(c.get(primary, "unknown"))
        counts[key] += 1
        bucket = buckets[key]
        if len(bucket) < max_per_bucket:
            if c["prompt"] not in seen_retained_prompts:
                bucket.append(c)
                seen_retained_prompts.add(c["prompt"])
        else:
            # Reservoir replacement within the bucket.
            j = rng.randrange(counts[key])
            if j < max_per_bucket and c["prompt"] not in seen_retained_prompts:
                old = bucket[j]
                seen_retained_prompts.discard(old["prompt"])
                bucket[j] = c
                seen_retained_prompts.add(c["prompt"])

    retained = [c for rows in buckets.values() for c in rows]
    if len(retained) < n:
        raise ValueError(f"Only retained {len(retained)} candidates, but {n} requested.")
    return sample_balanced(retained, n, rng, balance_keys)

def make_object_identity(prompt_id: str, obj: Dict[str, str], slot: str) -> Dict[str, str]:
    name = obj["object"]
    return {
        "constraint_id": f"{prompt_id}_identity_{slot}",
        "prompt_id": prompt_id,
        "constraint_type": "object_identity",
        "check_text": f"The object in {slot} should be identifiable as a {name}.",
        "object_1": name if slot == "object_1" else "", "object_2": name if slot == "object_2" else "",
        "object_1_group": obj["group"] if slot == "object_1" else "", "object_2_group": obj["group"] if slot == "object_2" else "",
        "target_object": name, "target_object_group": obj["group"], "object_slot": slot,
        "target_count": "", "relation": "", "relation_text": "", "attribute": "", "attribute_category": "",
        "attribute_1": "", "attribute_1_category": "", "attribute_2": "", "attribute_2_category": "", "attribute_slot": "",
    }


def make_cardinality(prompt_id: str, obj: Dict[str, str], count: int) -> Dict[str, str]:
    name = obj["object"]
    return {
        "constraint_id": f"{prompt_id}_cardinality",
        "prompt_id": prompt_id,
        "constraint_type": "cardinality",
        "check_text": f"The dominant generated object group corresponding to the requested {name} slot should contain exactly {count} visible instances.",
        "target_object": name, "target_object_group": obj["group"], "target_count": str(count), "object_slot": "object_1",
        "object_1": name, "object_1_group": obj["group"], "object_2": "", "object_2_group": "",
        "relation": "", "relation_text": "", "attribute": "", "attribute_category": "",
        "attribute_1": "", "attribute_1_category": "", "attribute_2": "", "attribute_2_category": "", "attribute_slot": "",
    }


def make_attribute_presence(prompt_id: str, attr: Dict[str, str], slot: str) -> Dict[str, str]:
    return {
        "constraint_id": f"{prompt_id}_attr_presence_{slot}",
        "prompt_id": prompt_id,
        "constraint_type": "attribute_presence",
        "check_text": f"The visual attribute '{attr['attribute']}' should be clearly present in the image.",
        "attribute": attr["attribute"], "attribute_category": attr["attribute_category"], "attribute_slot": slot,
        "attribute_1": attr["attribute"] if slot == "object_1" else "", "attribute_1_category": attr["attribute_category"] if slot == "object_1" else "",
        "attribute_2": attr["attribute"] if slot == "object_2" else "", "attribute_2_category": attr["attribute_category"] if slot == "object_2" else "",
        "object_1": "", "object_2": "", "object_1_group": "", "object_2_group": "", "relation": "", "relation_text": "",
        "target_object": "", "target_object_group": "", "target_count": "", "object_slot": "",
    }


def make_attribute_binding(prompt_id: str, obj: Dict[str, str], attr: Dict[str, str], slot: str) -> Dict[str, str]:
    name = obj["object"]
    return {
        "constraint_id": f"{prompt_id}_attr_binding_{slot}",
        "prompt_id": prompt_id,
        "constraint_type": "attribute_binding",
        "check_text": f"The {name} in {slot} should have the '{attr['attribute']}' visual attribute.",
        "object_1": name if slot == "object_1" else "", "object_2": name if slot == "object_2" else "",
        "object_1_group": obj["group"] if slot == "object_1" else "", "object_2_group": obj["group"] if slot == "object_2" else "",
        "target_object": name, "target_object_group": obj["group"], "object_slot": slot,
        "attribute": attr["attribute"], "attribute_category": attr["attribute_category"], "attribute_slot": slot,
        "attribute_1": attr["attribute"] if slot == "object_1" else "", "attribute_1_category": attr["attribute_category"] if slot == "object_1" else "",
        "attribute_2": attr["attribute"] if slot == "object_2" else "", "attribute_2_category": attr["attribute_category"] if slot == "object_2" else "",
        "relation": "", "relation_text": "", "target_count": "",
    }


def make_spatial(prompt_id: str, obj1: Dict[str, str], obj2: Dict[str, str], relation: str, relation_text: str) -> Dict[str, str]:
    return {
        "constraint_id": f"{prompt_id}_spatial",
        "prompt_id": prompt_id,
        "constraint_type": "spatial_relation",
        "check_text": f"The subject object for object_1 should be {relation_text} the subject object for object_2.",
        "object_1": obj1["object"], "object_2": obj2["object"],
        "object_1_group": obj1["group"], "object_2_group": obj2["group"],
        "relation": relation, "relation_text": relation_text,
        "target_object": "", "target_object_group": "", "target_count": "", "attribute": "", "attribute_category": "",
        "attribute_1": "", "attribute_1_category": "", "attribute_2": "", "attribute_2_category": "", "object_slot": "", "attribute_slot": "",
    }


def add(rows_p, rows_c, prompt_row, constraints):
    rows_p.append(prompt_row); rows_c.extend(constraints)


def prompt_row(pid, prompt, family, scene, composition):
    return {"prompt_id": pid, "prompt": prompt, "prompt_family": family, "scene_context_type": scene["scene_type"], "scene_context": scene["scene_context"], "composition": composition}


def candidates_spatial(config):
    template = config["templates"]["single_spatial"]
    scenes = scene_options(config)
    for rel_key, rel in config["relations"].items():
        for obj1, obj2 in relation_pairs(config, rel_key):
            for scene in scenes:
                yield {"prompt": template.format(prefix=scene["prefix"], object_1=obj1["object"], relation_text=rel["text"], object_2=obj2["object"], scene_context=scene["scene_context"]),
                       "family": "single_spatial", "relation": rel_key, "relation_text": rel["text"], "object_1": obj1, "object_2": obj2, "scene": scene, "scene_context_type": scene["scene_type"], "balance_key": rel_key}


def candidates_cardinality(config):
    template = config["templates"]["single_cardinality"]
    scenes = scene_options(config); objs = objects_from_groups(config, ["small_items"])
    for obj, count, scene in product(objs, config["numbers"], scenes):
        yield {"prompt": template.format(prefix=scene["prefix"], number=count, object_1_plural=pluralize(obj["object"]), scene_context=scene["scene_context"]),
               "family": "single_cardinality", "object_1": obj, "count": count, "scene": scene, "scene_context_type": scene["scene_type"], "balance_key": str(count)}


def compatible_object_attribute_pairs(config):
    for obj in all_objects(config):
        for attr in compatible_attributes(config, obj["group"]):
            yield obj, attr


def candidates_attribute(config):
    template = config["templates"]["single_attribute"]
    scenes = scene_options(config); pairs = list(compatible_object_attribute_pairs(config))
    for (obj1, attr1), (obj2, attr2), scene in product(pairs, pairs, scenes):
        if obj1["object"] == obj2["object"] or attr1["attribute"] == attr2["attribute"]:
            continue
        yield {"prompt": template.format(prefix=scene["prefix"], attribute_1=attr1["attribute"], object_1=obj1["object"], attribute_2=attr2["attribute"], object_2=obj2["object"], scene_context=scene["scene_context"]),
               "family": "single_attribute", "object_1": obj1, "object_2": obj2, "attribute_1": attr1, "attribute_2": attr2, "scene": scene, "scene_context_type": scene["scene_type"], "balance_key": attr1["attribute_category"]}


def candidates_spatial_attribute(config):
    template = config["templates"]["combined_spatial_attribute"]
    scenes = scene_options(config)
    for rel_key, rel in config["relations"].items():
        for obj1, obj2 in relation_pairs(config, rel_key):
            for attr1, attr2, scene in product(compatible_attributes(config, obj1["group"]), compatible_attributes(config, obj2["group"]), scenes):
                if attr1["attribute"] == attr2["attribute"]:
                    continue
                yield {"prompt": template.format(prefix=scene["prefix"], attribute_1=attr1["attribute"], object_1=obj1["object"], relation_text=rel["text"], attribute_2=attr2["attribute"], object_2=obj2["object"], scene_context=scene["scene_context"]),
                       "family": "combined_spatial_attribute", "relation": rel_key, "relation_text": rel["text"], "object_1": obj1, "object_2": obj2, "attribute_1": attr1, "attribute_2": attr2, "scene": scene, "scene_context_type": scene["scene_type"], "balance_key": rel_key}


def candidates_spatial_cardinality(config):
    template = config["templates"]["combined_spatial_cardinality"]
    scenes = scene_options(config)
    for rel_key, rel in config["relations"].items():
        for obj1, obj2 in relation_pairs(config, rel_key):
            if obj1["group"] != "small_items":
                continue
            for count, scene in product(config["numbers"], scenes):
                yield {"prompt": template.format(prefix=scene["prefix"], number=count, object_1_plural=pluralize(obj1["object"]), relation_text=rel["text"], object_2=obj2["object"], scene_context=scene["scene_context"]),
                       "family": "combined_spatial_cardinality", "relation": rel_key, "relation_text": rel["text"], "object_1": obj1, "object_2": obj2, "count": count, "scene": scene, "scene_context_type": scene["scene_type"], "balance_key": rel_key}


def candidates_attribute_cardinality(config):
    template = config["templates"]["combined_attribute_cardinality"]
    scenes = scene_options(config); objs1 = objects_from_groups(config, ["small_items"]); pairs2 = list(compatible_object_attribute_pairs(config))
    for obj1, count, (obj2, attr2), scene in product(objs1, config["numbers"], pairs2, scenes):
        if obj1["object"] == obj2["object"]:
            continue
        for attr1 in compatible_attributes(config, obj1["group"]):
            if attr1["attribute"] == attr2["attribute"]:
                continue
            yield {"prompt": template.format(prefix=scene["prefix"], number=count, attribute_1=attr1["attribute"], object_1_plural=pluralize(obj1["object"]), attribute_2=attr2["attribute"], object_2=obj2["object"], scene_context=scene["scene_context"]),
                   "family": "combined_attribute_cardinality", "object_1": obj1, "object_2": obj2, "attribute_1": attr1, "attribute_2": attr2, "count": count, "scene": scene, "scene_context_type": scene["scene_type"], "balance_key": attr1["attribute_category"]}


FAMILY_GENERATORS = {
    "single_spatial": candidates_spatial,
    "single_attribute": candidates_attribute,
    "single_cardinality": candidates_cardinality,
    "combined_spatial_attribute": candidates_spatial_attribute,
    "combined_spatial_cardinality": candidates_spatial_cardinality,
    "combined_attribute_cardinality": candidates_attribute_cardinality,
}


def constraints_for_candidate(pid: str, c: Dict) -> List[Dict[str, str]]:
    family = c["family"]
    obj1 = c.get("object_1"); obj2 = c.get("object_2")
    cons: List[Dict[str, str]] = []
    if obj1:
        cons.append(make_object_identity(pid, obj1, "object_1"))
    if obj2:
        cons.append(make_object_identity(pid, obj2, "object_2"))
    if "cardinality" in family:
        cons.append(make_cardinality(pid, obj1, c["count"]))
    if "attribute" in family:
        a1, a2 = c["attribute_1"], c["attribute_2"]
        cons.extend([make_attribute_presence(pid, a1, "object_1"), make_attribute_presence(pid, a2, "object_2")])
        cons.extend([make_attribute_binding(pid, obj1, a1, "object_1"), make_attribute_binding(pid, obj2, a2, "object_2")])
    if "spatial" in family:
        cons.append(make_spatial(pid, obj1, obj2, c["relation"], c["relation_text"]))
    return cons


def generate_all(config: Dict, seed: int | None = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(int(seed if seed is not None else config.get("random_seed", 348)))
    counts = config.get("dataset", {}).get("prompt_counts") or config.get("checkpoint1_counts") or {}
    prompts: List[Dict[str, str]] = []
    constraints: List[Dict[str, str]] = []
    for family, n in counts.items():
        if n <= 0:
            continue
        if family not in FAMILY_GENERATORS:
            raise ValueError(f"Unknown prompt family: {family}")
        sampled = sample_balanced_stream(FAMILY_GENERATORS[family](config), int(n), rng, ["balance_key", "scene_context_type"])
        sampled = sorted(sampled, key=lambda x: (x.get("balance_key", ""), x["scene"]["scene_type"], x["prompt"]))
        for i, cand in enumerate(sampled, start=1):
            pid = f"{family}_{i:03d}"
            composition = "combined" if family.startswith("combined") else "single"
            add(prompts, constraints, prompt_row(pid, cand["prompt"], family, cand["scene"], composition), constraints_for_candidate(pid, cand))
    return prompts, constraints


def build_generation_plan(prompts: List[Dict[str, str]], config: Dict, seed: int | None = None) -> List[Dict[str, str]]:
    rng = random.Random(int(seed if seed is not None else config.get("random_seed", 348)) + 999)
    models = config.get("dataset", {}).get("t2i_models", [])
    if not models:
        return []
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for p in prompts:
        grouped[p["prompt_family"]].append(p)
    rows: List[Dict[str, str]] = []
    idx = 1
    for family, ps in sorted(grouped.items()):
        ps = ps[:]
        rng.shuffle(ps)
        for j, p in enumerate(ps):
            m = models[j % len(models)]
            rows.append({
                "generation_job_id": f"job_{idx:04d}",
                "prompt_id": p["prompt_id"],
                "provider": m.get("provider", ""),
                "model_name": m.get("model_name", ""),
                "assignment_type": config.get("dataset", {}).get("assignment_type", "unpaired_stratified"),
                "image_dir": m.get("image_dir", "../data/images"),
                "size": m.get("size", ""),
                "quality": m.get("quality", ""),
                "aspect_ratio": m.get("aspect_ratio", ""),
            })
            idx += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/grammar_config.yaml")
    parser.add_argument("--out-prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--out-constraints", default="../data/prompts/constraints.csv")
    parser.add_argument("--out-generation-plan", default="../data/generations/generation_plan.csv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    prompts, constraints = generate_all(config, seed=args.seed)
    write_csv(args.out_prompts, prompts, PROMPT_FIELDS)
    write_csv(args.out_constraints, constraints, CONSTRAINT_FIELDS)
    plan = build_generation_plan(prompts, config, seed=args.seed)
    if args.out_generation_plan:
        write_csv(args.out_generation_plan, plan, GENERATION_PLAN_FIELDS)
    print(f"Wrote {len(prompts)} prompts to {args.out_prompts}")
    print(f"Wrote {len(constraints)} constraints to {args.out_constraints}")
    if plan:
        print(f"Wrote {len(plan)} generation jobs to {args.out_generation_plan}")


if __name__ == "__main__":
    main()
