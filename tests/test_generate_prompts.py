import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_prompts import generate_all, load_config


def test_generate_prompts_nonempty():
    config = load_config("configs/grammar_config.yaml")
    prompts, constraints = generate_all(config)
    assert len(prompts) > 0
    assert len(constraints) > 0


def test_prompt_ids_unique():
    config = load_config("configs/grammar_config.yaml")
    prompts, _ = generate_all(config)
    ids = [p["prompt_id"] for p in prompts]
    assert len(ids) == len(set(ids))


def test_each_prompt_has_constraints():
    config = load_config("configs/grammar_config.yaml")
    prompts, constraints = generate_all(config)
    prompt_ids = {p["prompt_id"] for p in prompts}
    constrained_prompt_ids = {c["prompt_id"] for c in constraints}
    assert prompt_ids.issubset(constrained_prompt_ids)


def test_combined_prompts_have_multiple_constraints():
    config = load_config("configs/grammar_config.yaml")
    prompts, constraints = generate_all(config)
    counts = {}
    for c in constraints:
        counts[c["prompt_id"]] = counts.get(c["prompt_id"], 0) + 1
    for p in prompts:
        if p["composition"] == "combined":
            assert counts[p["prompt_id"]] >= 4
