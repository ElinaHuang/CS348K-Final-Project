from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from utils import ensure_dir, read_csv, write_csv


PROMPT_FIELDS = ["prompt_id", "prompt", "prompt_family", "scene_context_type", "scene_context", "composition"]
CONSTRAINT_FIELDS = [
    "constraint_id", "prompt_id", "constraint_type", "check_text",
    "object_1", "object_2", "relation", "relation_text",
    "target_object", "target_count", "attribute", "attribute_1", "attribute_2",
]


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def select_prompt_subset(config: Dict) -> None:
    cfg = config["prompt_subset"]
    prompts = read_csv(cfg["source_prompts"])
    constraints = read_csv(cfg["source_constraints"])
    mode = cfg.get("selection_mode", "explicit_prompt_ids")

    if mode != "explicit_prompt_ids":
        raise ValueError("Checkpoint 2 currently supports selection_mode: explicit_prompt_ids")

    selected_ids = cfg.get("prompt_ids", [])
    selected_set = set(selected_ids)
    prompt_by_id = {p["prompt_id"]: p for p in prompts}

    missing = [pid for pid in selected_ids if pid not in prompt_by_id]
    if missing:
        raise ValueError(f"Prompt ids not found in source prompts: {missing}")

    selected_prompts = [prompt_by_id[pid] for pid in selected_ids]
    selected_constraints = [c for c in constraints if c.get("prompt_id") in selected_set]

    write_csv(cfg["out_prompts"], selected_prompts, PROMPT_FIELDS)
    write_csv(cfg["out_constraints"], selected_constraints, CONSTRAINT_FIELDS)
    print(f"Wrote {len(selected_prompts)} checkpoint2 prompts to {cfg['out_prompts']}")
    print(f"Wrote {len(selected_constraints)} checkpoint2 constraints to {cfg['out_constraints']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/checkpoint2_config.yaml")
    parser.add_argument("--generate-full-prompts", action="store_true")
    parser.add_argument("--select-prompts", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    py = sys.executable

    prompt_cfg = config["prompt_subset"]
    
    if args.generate_full_prompts:
        run([
            py, "scripts/generate_prompts.py",
            "--config", "configs/grammar_config.yaml",
            "--out-prompts", prompt_cfg["source_prompts"],
            "--out-constraints", prompt_cfg["source_constraints"],
        ])

    if args.select_prompts:
        select_prompt_subset(config)

if __name__ == "__main__":
    main()
