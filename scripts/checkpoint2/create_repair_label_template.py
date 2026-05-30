from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

from utils import read_csv, write_csv

FIELDS = [
    "repair_id", "repaired_image_id", "source_image_id", "source_prompt_id",
    "constraint_id", "constraint_type", "check_text", "before_label",
    "after_label", "after_notes", "is_target_constraint",
]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/checkpoint2_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    prompt_cfg = config["prompt_subset"]
    repair_cfg = config["repair"]
    label_cfg = config["human_labels"]

    constraints = read_csv(prompt_cfg["out_constraints"])
    before_labels = read_csv(label_cfg["labels_csv"])
    repaired_prompts = read_csv(repair_cfg["repaired_prompts_csv"])
    repaired_generations = read_csv(repair_cfg["repaired_generations_csv"])

    constraints_by_prompt: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in constraints:
        constraints_by_prompt[c["prompt_id"]].append(c)

    before_by_key = {(r["image_id"], r["constraint_id"]): r.get("human_label", "") for r in before_labels}
    repair_by_id = {r["repair_id"]: r for r in repaired_prompts}

    rows: List[Dict[str, str]] = []
    for gen in repaired_generations:
        repair_id = gen["prompt_id"]
        repair = repair_by_id.get(repair_id)
        if not repair:
            continue
        source_image_id = repair["source_image_id"]
        source_prompt_id = repair["source_prompt_id"]
        target_ids = set(x for x in repair.get("target_constraint_ids", "").split(";") if x)
        for c in constraints_by_prompt.get(source_prompt_id, []):
            rows.append({
                "repair_id": repair_id,
                "repaired_image_id": gen["image_id"],
                "source_image_id": source_image_id,
                "source_prompt_id": source_prompt_id,
                "constraint_id": c["constraint_id"],
                "constraint_type": c["constraint_type"],
                "check_text": c.get("check_text", ""),
                "before_label": before_by_key.get((source_image_id, c["constraint_id"]), ""),
                "after_label": "",
                "after_notes": "",
                "is_target_constraint": str(c["constraint_id"] in target_ids),
            })

    write_csv(repair_cfg["repaired_human_labels_csv"], rows, FIELDS)
    print(f"Wrote {len(rows)} repaired-label template rows to {repair_cfg['repaired_human_labels_csv']}")


if __name__ == "__main__":
    main()
