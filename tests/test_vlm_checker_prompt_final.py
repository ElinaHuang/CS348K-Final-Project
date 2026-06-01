from pathlib import Path

from run_vlm_checker import build_checker_prompt, task_sentence_for_constraint_type
from utils import parse_vlm_response


def test_task_sentence_for_constraint_type():
    assert task_sentence_for_constraint_type("object_identity") == "object existence"
    assert task_sentence_for_constraint_type("cardinality") == "count"
    assert task_sentence_for_constraint_type("attribute") == "attribute"
    assert task_sentence_for_constraint_type("spatial_relation") == "spatial relation"


def test_build_checker_prompt_for_one_constraint():
    constraint = {
        "constraint_id": "constraint_test_001",
        "prompt_id": "prompt_test_001",
        "constraint_type": "spatial_relation",
        "check_text": "Does the image show a small paper clip to the left of a cup?",
    }

    prompt = build_checker_prompt(constraint)

    # Print one generated VLM checker prompt for manual inspection.
    print("\n--- Example VLM checker prompt ---")
    print(prompt)
    print("--- End example VLM checker prompt ---\n")

    assert "You are a visual constraint checker." in prompt
    assert "Your job is to decide whether the image satisfies ONE visual constraint." in prompt

    assert "Constraint type: spatial relation" in prompt
    assert "Constraint: Does the image show a small paper clip to the left of a cup?" in prompt

    assert "Decision rules:" in prompt
    assert 'Return "pass" if the constraint is reasonably satisfied.' in prompt
    assert 'Return "fail" if the constraint is clearly violated.' in prompt
    assert 'Return "ambiguous" if the image does not provide enough visual evidence to confidently decide pass or fail.' in prompt

    assert "Ambiguous label guidance:" in prompt
    assert "Use ambiguous when the relevant object, count, attribute, or spatial relation is visually unclear, or when the target object's" in prompt
    assert "If a reasonable human annotator can still make a clear judgment, choose pass or fail rather than ambiguous." in prompt
    assert "Common ambiguous cases include heavy occlusion" in prompt

    assert "Return ONLY valid JSON" in prompt
    assert '"label": "pass" | "fail" | "ambiguous"' in prompt
    assert '"reason": "one short sentence"' in prompt


def test_build_checker_prompt_for_attribute_constraint():
    constraint = {
        "constraint_id": "constraint_test_002",
        "prompt_id": "prompt_test_002",
        "constraint_type": "attribute",
        "check_text": "Does the image show a blue-and-white striped notebook?",
    }

    prompt = build_checker_prompt(constraint)

    assert "Constraint type: attribute" in prompt
    assert "Constraint: Does the image show a blue-and-white striped notebook?" in prompt
    assert "Ambiguous label guidance:" in prompt
    assert "Return ONLY valid JSON" in prompt


def test_checker_prompt_defines_task_and_ambiguous_label():
    prompt = build_checker_prompt({
        "constraint_type": "spatial_relation",
        "check_text": "The paper clip should be inside the small box.",
    })
    lowered = prompt.lower()
    assert "visual constraint checker" in lowered
    assert "reasonable human annotator" in lowered
    assert "ambiguous" in lowered
    assert "not provide enough visual evidence" in lowered
    assert "occlusion" in lowered or "occluded" in lowered
    assert "cropping" in lowered or "cropped" in lowered
    assert "blurry" in lowered or "blur" in lowered
    assert "return only" in lowered and "json" in lowered


def test_parse_vlm_response_handles_json_and_fallback():
    label, reason, mode = parse_vlm_response('{"label": "fail", "reason": "The count is wrong."}')
    assert label == "fail"
    assert "count" in reason
    assert mode in {"json", "success"}

    label, reason, mode = parse_vlm_response("Label: ambiguous. The object is too blurry to identify.")
    assert label == "ambiguous"
    assert mode == "fallback"
