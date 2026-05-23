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
    parser.add_argument("--generate-images", action="store_true")
    parser.add_argument("--dry-run-images", action="store_true")
    parser.add_argument("--create-human-template", action="store_true")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--dry-run-vlm", action="store_true")
    parser.add_argument("--all-pre-label", action="store_true", help="Generate prompt subset, generate images, and create human-label template.")
    args = parser.parse_args()

    config = load_config(args.config)
    py = sys.executable

    prompt_cfg = config["prompt_subset"]
    gen_cfg = config["generation"]
    label_cfg = config["human_labels"]
    vlm_cfg = config["vlm_checker"]

    if args.all_pre_label:
        args.generate_full_prompts = True
        args.select_prompts = True
        args.generate_images = True
        args.create_human_template = True
    
    if args.generate_full_prompts:
        run([
            py, "generate_prompts.py",
            "--config", "../configs/grammar_config.yaml",
            "--out-prompts", prompt_cfg["source_prompts"],
            "--out-constraints", prompt_cfg["source_constraints"],
        ])

    if args.select_prompts:
        select_prompt_subset(config)

    if args.generate_images:
        cmd = [
            py, "generate_images.py",
            "--prompts", prompt_cfg["out_prompts"],
            "--out", gen_cfg["generations_csv"],
            "--image-dir", gen_cfg["image_dir"],
            "--model-name", gen_cfg["model_name"],
            "--samples-per-prompt", str(gen_cfg.get("samples_per_prompt", 1)),
            "--provider", gen_cfg.get("provider", "stub"),
            "--size", gen_cfg.get("size", "1024x1024"),
            "--quality", gen_cfg.get("quality", "low"),
        ]
        if args.dry_run_images:
            cmd.append("--dry-run")
        run(cmd)
    
    if args.create_human_template:
        run([
            py, "create_human_label_template.py",
            "--generations", gen_cfg["generations_csv"],
            "--constraints", prompt_cfg["out_constraints"],
            "--out", label_cfg["labels_csv"],
        ])

    if args.run_vlm:
        cmd = [
            py, "run_vlm_checker.py",
            "--generations", gen_cfg["generations_csv"],
            "--constraints", prompt_cfg["out_constraints"],
            "--out", vlm_cfg["labels_csv"],
            "--model-name", vlm_cfg["model_name"],
            "--provider", vlm_cfg.get("provider", "stub"),
        ]
        if args.dry_run_vlm:
            cmd.append("--dry-run")
        run(cmd)


if __name__ == "__main__":
    main()
