from __future__ import annotations

import argparse
from typing import Dict, List

from utils import read_csv, write_csv

FIELDNAMES = [
    "image_id", "prompt_id", "provider", "model_name", "image_path", "prompt",
    "constraint_id", "constraint_type", "check_text", "human_label", "human_notes",
]


def build_template(generations: List[Dict[str, str]], constraints: List[Dict[str, str]]) -> List[Dict[str, str]]:
    constraints_by_prompt: Dict[str, List[Dict[str, str]]] = {}
    for c in constraints:
        constraints_by_prompt.setdefault(c["prompt_id"], []).append(c)
    rows: List[Dict[str, str]] = []
    for g in generations:
        if g.get("generation_status", "success") not in ("success", "", None):
            continue
        for c in constraints_by_prompt.get(g["prompt_id"], []):
            rows.append({
                "image_id": g["image_id"], "prompt_id": g["prompt_id"],
                "provider": g.get("provider", ""), "model_name": g.get("model_name", ""),
                "image_path": g.get("image_path", ""), "prompt": g.get("prompt", ""),
                "constraint_id": c["constraint_id"], "constraint_type": c["constraint_type"],
                "check_text": c["check_text"], "human_label": "", "human_notes": "",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="../data/generations/generations.csv")
    parser.add_argument("--constraints", default="../data/prompts/constraints.csv")
    parser.add_argument("--out", default="../data/labels/human_labels.csv")
    args = parser.parse_args()
    rows = build_template(read_csv(args.generations), read_csv(args.constraints))
    write_csv(args.out, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} human-label rows to {args.out}")


if __name__ == "__main__":
    main()
