from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List

from utils import parse_vlm_response, read_csv, write_csv, load_environment

VLM_LABEL_FIELDS = [
    "image_id",
    "prompt_id",
    "constraint_id",
    "constraint_type",
    "vlm_label",
    "vlm_reason",
    "parse_status",
    "raw_response",
]


def build_checker_prompt(constraint: Dict[str, str]) -> str:
    ctype = constraint["constraint_type"]
    check_text = constraint["check_text"]

    base = [
        "You are a visual constraint checker.",
        "",
        "Your task is to judge whether the image satisfies ONE visual constraint.",
        "",
        f"Constraint type: {ctype}",
        f"Constraint: {check_text}",
        "",
        "Decision rules:",
        '- Return "pass" only if the constraint is clearly satisfied.',
        '- Return "fail" if the constraint is clearly violated.',
        '- Return "ambiguous" if the image is unclear, objects are occluded, object identity is uncertain, or the constraint cannot be confidently judged.',
    ]

    if ctype == "cardinality":
        obj = constraint.get("target_object", "")
        count = constraint.get("target_count", "")
        base.extend([
            f'- For counting, count only clearly visible instances of the target object: {obj}.',
            f'- Do not count ambiguous shapes or partial objects unless they are clearly identifiable as {obj}.',
            f'- The expected count is exactly {count}.',
        ])
    elif ctype == "attribute":
        base.extend([
            "- For attributes, check whether the specified attribute is attached to the specified object.",
            "- Do not mark pass merely because the attribute appears somewhere else in the image.",
        ])
    elif ctype == "spatial_relation":
        base.extend([
            "- Use the 2D image coordinate system from the viewer's perspective.",
            '- "left" means the left side of the image.',
            '- "right" means the right side of the image.',
            '- "above" means higher in the image.',
            '- "below" means lower in the image.',
            "- The relevant objects must be clearly visible for a pass or fail judgment.",
        ])
    elif ctype == "object_existence":
        base.extend([
            "- For object existence, judge whether the target object is clearly visible.",
            "- If the object may be present but is unclear, use ambiguous.",
        ])

    base.extend([
        "",
        "Return ONLY valid JSON in the following format:",
        '{ "label": "pass" | "fail" | "ambiguous", "reason": "one short sentence" }',
    ])

    return "\n".join(base)


def call_vlm_checker_stub(image_path: str, checker_prompt: str, model_name: str) -> str:
    """Placeholder for a real VLM API call.

    Replace this with your chosen VLM API call later. The function should return
    the raw text response from the model.
    """
    raise NotImplementedError(
        "VLM checker API is not implemented yet. Replace call_vlm_checker_stub with the chosen VLM API call."
    )


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
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": checker_prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        temperature=0,
    )
    return getattr(response, "output_text", "") or str(response)


def call_vlm_checker(image_path: str, checker_prompt: str, model_name: str, provider: str) -> str:
    if provider == "openai":
        return call_openai_vlm_checker(image_path, checker_prompt, model_name)
    if provider == "stub":
        return call_vlm_checker_stub(image_path, checker_prompt, model_name=model_name)
    raise NotImplementedError(f"Unsupported VLM provider: {provider}")


def run_checker(
    generations: List[Dict[str, str]],
    constraints: List[Dict[str, str]],
    model_name: str,
    provider: str = "stub",
    dry_run: bool = False,
) -> List[Dict[str, str]]:
    constraints_by_prompt: Dict[str, List[Dict[str, str]]] = {}
    for c in constraints:
        constraints_by_prompt.setdefault(c["prompt_id"], []).append(c)

    rows: List[Dict[str, str]] = []
    for g in generations:
        if g.get("generation_status", "success") not in ("success", "", None):
            continue
        for c in constraints_by_prompt.get(g["prompt_id"], []):
            checker_prompt = build_checker_prompt(c)
            if dry_run:
                raw_response = json.dumps({"label": "ambiguous", "reason": "Dry run placeholder; no VLM call was made."})
            else:
                raw_response = call_vlm_checker(g["image_path"], checker_prompt, model_name, provider=provider)
            label, reason, parse_status = parse_vlm_response(raw_response)
            rows.append({
                "image_id": g["image_id"],
                "prompt_id": g["prompt_id"],
                "constraint_id": c["constraint_id"],
                "constraint_type": c["constraint_type"],
                "vlm_label": label,
                "vlm_reason": reason,
                "parse_status": parse_status,
                "raw_response": raw_response,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="../data/generations/generations.csv")
    parser.add_argument("--constraints", default="../data/prompts/constraints.csv")
    parser.add_argument("--out", default="../data/labels/vlm_labels.csv")
    parser.add_argument("--model-name", default="vlm_checker")
    parser.add_argument("--provider", default="stub", choices=["stub", "openai"])
    parser.add_argument("--dry-run", action="store_true", help="Write placeholder VLM labels without calling an API.")
    args = parser.parse_args()

    generations = read_csv(args.generations)
    constraints = read_csv(args.constraints)
    rows = run_checker(generations, constraints, args.model_name, provider=args.provider, dry_run=args.dry_run)
    write_csv(args.out, rows, VLM_LABEL_FIELDS)
    print(f"Wrote {len(rows)} VLM-label rows to {args.out}")


if __name__ == "__main__":
    main()
