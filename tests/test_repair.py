import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_repair_prompts import build_repaired_prompt, strengthened_instruction
from analyze_repair import analyze_repair


def test_strengthened_spatial_fail_instruction():
    constraint = {
        "constraint_id": "p1_spatial",
        "constraint_type": "spatial_relation",
        "check_text": "The cup should be to the left of the book.",
        "object_1": "cup",
        "object_2": "book",
        "relation_text": "to the left of",
    }
    instruction = strengthened_instruction(constraint, "fail")
    assert "cup" in instruction
    assert "book" in instruction
    assert "left" in instruction
    assert "clearly" in instruction


def test_repaired_prompt_reconstructs_all_constraints_and_strengthens_target():
    constraints = [
        {"constraint_id": "p1_exist_1", "constraint_type": "object_existence", "check_text": "The cup should be clearly visible.", "target_object": "cup"},
        {"constraint_id": "p1_spatial", "constraint_type": "spatial_relation", "check_text": "The cup should be to the left of the book.", "object_1": "cup", "object_2": "book", "relation_text": "to the left of"},
    ]
    prompt = build_repaired_prompt(constraints, {"p1_spatial": "fail"})
    assert "Create a clear image" in prompt
    assert "The cup should be clearly visible" in prompt
    assert "must be clearly to the left of" in prompt
    assert "Original prompt" not in prompt


def test_analyze_repair_target_image_and_regression_metrics():
    repaired_prompts = [
        {"repair_id": "r1", "source_image_id": "img1", "source_prompt_id": "p1", "repair_strategy": "repair_all_failed"}
    ]
    repair_targets = [
        {"repair_id": "r1", "source_image_id": "img1", "source_prompt_id": "p1", "target_constraint_id": "c2", "target_constraint_type": "spatial_relation", "before_label": "fail"}
    ]
    repaired_labels = [
        {"repair_id": "r1", "constraint_id": "c1", "constraint_type": "object_existence", "before_label": "pass", "after_label": "pass"},
        {"repair_id": "r1", "constraint_id": "c2", "constraint_type": "spatial_relation", "before_label": "fail", "after_label": "pass"},
    ]
    results = analyze_repair(repaired_prompts, repair_targets, repaired_labels)
    assert results["target_results"][0]["target_fixed"] == "True"
    assert results["image_results"][0]["image_fixed"] == "True"
    assert results["image_results"][0]["num_regressions"] == "0"
    assert results["type_results"][0]["target_constraint_type"] == "spatial_relation"
    assert results["type_results"][0]["target_fixed_rate"] == "1.0000"
