from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List

from utils import load_environment, parse_vlm_response, read_csv, write_csv

VLM_LABEL_FIELDS = [
    "image_id", "prompt_id", "provider", "model_name", "constraint_id", "constraint_type",
    "vlm_label", "vlm_reason", "parse_status", "raw_response",
]


def task_sentence_for_constraint_type(constraint_type: str) -> str:
    if constraint_type == "object_identity":
        return ("object existence")
    if constraint_type == "cardinality":
        return ("count")
    if constraint_type == "attribute":
        return ("attribute")
    if constraint_type == "spatial_relation":
        return ("spatial relation")


def build_checker_prompt(constraint: Dict[str, str]) -> str:
    check_text = constraint.get("check_text", "").strip()
    ctype = constraint.get("constraint_type", "").strip()
    task_sentence = task_sentence_for_constraint_type(ctype)

    rules = [
        "You are a visual constraint checker.",
        "Your job is to decide whether the image satisfies ONE visual constraint.",
        "",
        f"Constraint type: {task_sentence}",
        f"Constraint: {check_text}",
        "",
        "Decision rules:",
        '- Return "pass" if the constraint is reasonably satisfied.',
        '- Return "fail" if the constraint is clearly violated.',
        '- Return "ambiguous" if the image does not provide enough visual evidence to confidently decide pass or fail.',
        "",
        "Dependency-masked pass rule:",
        "- Some requirements depend on a target object existing first. For cardinality, attribute, and spatial-relation requirements, if the relevant target object is clearly missing or replaced by a clearly different object, mark this dependent visual constraint as pass rather than fail or ambiguous.",
        "- Use this rule only when the target object is clearly missing or wrong. If the target object is present but the count, attribute, or relation is unclear, return ambiguous.",
        "",
        "Ambiguous label guidance:",
        "- Use ambiguous when the relevant object, count, attribute, or relation is genuinely unclear, not merely imperfect.",
        "- If a reasonable human annotator can still make a clear judgment, choose pass or fail rather than ambiguous.",
        "- Common ambiguous cases include heavy occlusion, cropping, blur, very small objects, strongly overlapping objects, unclear object identity, unclear attribute visibility, or a spatial relation that cannot be determined from the image.",
        "",
        'Return ONLY valid JSON: {"label": "pass" | "fail" | "ambiguous", "reason": "one short sentence"}',
    ]

    return "\n".join(rules)


# def build_checker_prompt(constraint: Dict[str, str]) -> str:
#     check_text = constraint.get("check_text", "").strip()
#     ctype = constraint.get("constraint_type", "")
#     rules = [
#         "You are a visual constraint checker for a text-to-image evaluation pipeline.",
#         "Your job is to decide whether the image satisfies one specific visual requirement.",
#         "Use the judgment style of a reasonable human annotator.",
#         "Judge only the given requirement; do not evaluate overall image quality, aesthetics, or unrelated details.",
#         "",
#         f"Visual requirement: {check_text}",
#         "",
#         "Labels:",
#         "- pass: the requirement is reasonably satisfied.",
#         "- fail: the requirement is clearly violated.",
#         "- ambiguous: there is not enough visual evidence to confidently decide pass or fail.",
#         "",
#         "Dependency-masked pass rule:",
#         "- Some requirements depend on a target object existing first. For cardinality, attribute, and spatial-relation requirements, if the relevant target object is clearly missing or replaced by a clearly different object, mark this dependent requirement as pass rather than fail or ambiguous. The actionable failure is handled by a separate object_identity requirement.",
#         "- Use this masking rule only when the prerequisite object is clearly missing or clearly the wrong object. If the object is present but the count, attribute, or relation itself is unclear, use ambiguous.",
#         "",
#         "Ambiguous label guidance:",
#         "- Use ambiguous when the relevant object, count, attribute, or relation is genuinely unclear, not merely imperfect.",
#         "- Common ambiguous cases include heavy occlusion, cropping, blur, very small objects, strongly overlapping objects, unclear object identity, unclear attribute visibility, or a spatial relation that cannot be determined from the image.",
#         "- Do not use ambiguous just because the image is stylized, slightly messy, or has minor artifacts.",
#         "",
#         "Calibration rules:",
#         "- If a reasonable human annotator can still make a clear judgment, choose pass or fail rather than ambiguous.",
#         "- Minor artifacts or clutter are acceptable if the requirement is still clearly satisfied.",
#     ]
#     if ctype == "object_identity":
#         rules += ["- For object_identity, check only whether the requested object category is recognizable."]
#     elif ctype == "cardinality":
#         rules += ["- For cardinality, check whether the requested number of target objects is visible and countable, unless the target object itself is clearly missing or wrong, in which case use the dependency-masked pass rule."]
#     elif ctype == "attribute":
#         rules += ["- For attribute, check the full object-attribute requirement, such as whether the image shows a blue-and-white striped notebook, unless the target object itself is clearly missing or wrong, in which case use the dependency-masked pass rule."]
#     elif ctype == "spatial_relation":
#         rules += ["- For spatial_relation, check the complete relation in the requirement, such as whether the apple is to the left of the banana. If the required object is missing or clearly wrong, use the dependency-masked pass rule."]
#     rules += ["", 'Return ONLY valid JSON: {"label": "pass" | "fail" | "ambiguous", "reason": "one short sentence"}']
#     return "\n".join(rules)


def encode_image_as_data_url(image_path: str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().replace(".", "") or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_openai_vlm_checker(image_path: str, checker_prompt: str, model_name: str) -> str:
    load_environment()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK with: pip install openai") from exc
    client = OpenAI()
    data_url = encode_image_as_data_url(image_path)
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": [{"type": "input_text", "text": checker_prompt}, {"type": "input_image", "image_url": data_url}]}],
        temperature=0,
    )
    return getattr(response, "output_text", "") or str(response)


def call_vlm_checker(image_path: str, checker_prompt: str, model_name: str, provider: str) -> str:
    if provider == "openai":
        return call_openai_vlm_checker(image_path, checker_prompt, model_name)
    if provider == "stub":
        return json.dumps({"label": "ambiguous", "reason": "Stub VLM label."})
    raise NotImplementedError(f"Unsupported VLM provider: {provider}")


def run_checker(generations: List[Dict[str, str]], constraints: List[Dict[str, str]], model_name: str, provider: str = "openai", dry_run: bool = False) -> List[Dict[str, str]]:
    by_prompt: Dict[str, List[Dict[str, str]]] = {}
    for c in constraints:
        by_prompt.setdefault(c["prompt_id"], []).append(c)
    rows = []
    for g in generations:
        if g.get("generation_status", "success") not in ("success", "", None):
            continue
        for c in by_prompt.get(g["prompt_id"], []):
            prompt = build_checker_prompt(c)
            raw = json.dumps({"label": "ambiguous", "reason": "Dry run placeholder."}) if dry_run else call_vlm_checker(g["image_path"], prompt, model_name, provider)
            label, reason, status = parse_vlm_response(raw)
            rows.append({
                "image_id": g["image_id"], "prompt_id": g["prompt_id"], "provider": g.get("provider", ""), "model_name": g.get("model_name", ""),
                "constraint_id": c["constraint_id"], "constraint_type": c["constraint_type"], "vlm_label": label, "vlm_reason": reason, "parse_status": status, "raw_response": raw,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="data/generations/generations.csv")
    parser.add_argument("--constraints", default="data/prompts/constraints.csv")
    parser.add_argument("--out", default="data/labels/vlm_labels.csv")
    parser.add_argument("--model-name", default="gpt-4.1")
    parser.add_argument("--provider", default="openai", choices=["stub", "openai"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = run_checker(read_csv(args.generations), read_csv(args.constraints), args.model_name, provider=args.provider, dry_run=args.dry_run)
    write_csv(args.out, rows, VLM_LABEL_FIELDS)
    print(f"Wrote {len(rows)} VLM-label rows to {args.out}")


if __name__ == "__main__":
    main()
