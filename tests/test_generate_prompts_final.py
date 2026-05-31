from collections import Counter, defaultdict
from pathlib import Path

from generate_prompts import (
    build_generation_plan,
    compatible_attributes,
    generate_all,
    load_config,
)


EXPECTED_CONSTRAINT_TYPES = {
    "object_identity",
    "cardinality",
    "attribute",
    "spatial_relation",
}


def test_final_prompt_generation_counts_and_uniqueness(final_config, final_prompt_data):
    config = final_config
    prompts, constraints = final_prompt_data

    expected_counts = config["dataset"]["prompt_counts"]
    assert len(prompts) == sum(expected_counts.values())
    assert len({p["prompt_id"] for p in prompts}) == len(prompts)
    assert len({p["prompt"] for p in prompts}) == len(prompts)

    family_counts = Counter(p["prompt_family"] for p in prompts)
    assert family_counts == Counter(expected_counts)

    assert {c["constraint_type"] for c in constraints}.issubset(EXPECTED_CONSTRAINT_TYPES)
    assert {c["constraint_type"] for c in constraints} == EXPECTED_CONSTRAINT_TYPES


def test_simple_context_no_front_facing_language(final_prompt_data):
    prompts, _ = final_prompt_data
    simple_prompts = [p for p in prompts if p["scene_context_type"] == "simple"]
    assert simple_prompts
    assert all("front-facing" not in p["prompt"].lower() for p in simple_prompts)


def test_spatial_relations_use_compatible_object_groups(final_config, final_prompt_data):
    config = final_config
    _, constraints = final_prompt_data
    relation_cfg = config["relations"]

    spatial_constraints = [c for c in constraints if c["constraint_type"] == "spatial_relation"]
    assert spatial_constraints
    for c in spatial_constraints:
        rel = relation_cfg[c["relation"]]
        assert c["object_1_group"] in rel["object_1_groups"]
        assert c["object_2_group"] in rel["object_2_groups"]


def test_attribute_groups_are_used_and_compatible_with_object_groups(final_config, final_prompt_data):
    config = final_config
    _, constraints = final_prompt_data

    attribute_groups = config["attribute_groups"]
    group_names = set(attribute_groups.keys())
    plain_colors = {"red", "blue", "green", "yellow", "black", "white"}

    attribute_constraints = [c for c in constraints if c["constraint_type"] == "attribute"]
    assert attribute_constraints

    for c in attribute_constraints:
        assert c["attribute_category"] in group_names
        assert c["attribute"].lower() not in plain_colors
        allowed_groups = attribute_groups[c["attribute_category"]]["allowed_object_groups"]
        assert c["target_object_group"] in allowed_groups


def test_generation_plan_is_unpaired_and_stratified_by_family(final_config, final_prompt_data):
    config = final_config
    prompts, _ = final_prompt_data
    plan = build_generation_plan(prompts, config)

    assert len(plan) == len(prompts)
    assert {row["assignment_type"] for row in plan} == {"unpaired_stratified"}
    assert len({row["prompt_id"] for row in plan}) == len(prompts)

    prompt_by_id = {p["prompt_id"]: p for p in prompts}
    providers_by_family = defaultdict(Counter)
    for row in plan:
        family = prompt_by_id[row["prompt_id"]]["prompt_family"]
        providers_by_family[family][row["provider"]] += 1

    for family, counts in config["dataset"]["prompt_counts"].items():
        provider_counts = providers_by_family[family]
        assert sum(provider_counts.values()) == counts
        # All final family counts are even, so the split should be exactly balanced.
        assert len(set(provider_counts.values())) == 1
