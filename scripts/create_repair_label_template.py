from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

from utils import normalize_label, read_csv, write_csv

FIELDS = [
    "repair_id", "repaired_image_id", "source_image_id", "source_prompt_id", "constraint_id", "constraint_type", "check_text", "before_label", "after_label", "after_notes", "is_target_constraint",
]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/final_experiment_config.yaml")
    parser.add_argument("--repaired-generations", required=True)
    parser.add_argument("--repair-targets", required=True)
    parser.add_argument("--before-labels", required=True)
    parser.add_argument("--constraints", default="../data/prompts/constraints.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    repaired_gens, targets, before_labels, constraints = read_csv(args.repaired_generations), read_csv(args.repair_targets), read_csv(args.before_labels), read_csv(args.constraints)
    constraints_by_prompt: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in constraints:
        constraints_by_prompt[c["prompt_id"]].append(c)
    before_by_key = {(r["image_id"], r["constraint_id"]): normalize_label(r.get("human_label", "")) for r in before_labels}
    target_ids_by_repair: Dict[str, set] = defaultdict(set)
    source_by_repair: Dict[str, str] = {}
    for t in targets:
        target_ids_by_repair[t["repair_id"]].add(t["target_constraint_id"])
        source_by_repair[t["repair_id"]] = t["source_image_id"]
    rows = []
    for g in repaired_gens:
        repair_id = g["prompt_id"]
        source_image_id = source_by_repair.get(repair_id, "")
        source_prompt_id = ""
        if repair_id in source_by_repair:
            for t in targets:
                if t["repair_id"] == repair_id:
                    source_prompt_id = t["source_prompt_id"]; break
        for c in constraints_by_prompt.get(source_prompt_id, []):
            rows.append({
                "repair_id": repair_id, "repaired_image_id": g["image_id"], "source_image_id": source_image_id, "source_prompt_id": source_prompt_id,
                "constraint_id": c["constraint_id"], "constraint_type": c["constraint_type"], "check_text": c["check_text"],
                "before_label": before_by_key.get((source_image_id, c["constraint_id"]), ""), "after_label": "", "after_notes": "",
                "is_target_constraint": str(c["constraint_id"] in target_ids_by_repair.get(repair_id, set())),
            })
    write_csv(args.out, rows, FIELDS)
    print(f"Wrote {len(rows)} repaired-label template rows to {args.out}")


if __name__ == "__main__":
    main()
