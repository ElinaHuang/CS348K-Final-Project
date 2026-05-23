import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_repair_prompts import build_repaired_prompt, strengthened_instruction


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
