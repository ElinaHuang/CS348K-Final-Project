from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/final_experiment_config.yaml")
    parser.add_argument("--stage", required=True, choices=[
        "generate-prompts", "generate-images", "create-human-template"
    ])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    py = sys.executable

    if args.stage == "generate-prompts":
        run([py, "generate_prompts.py", "--config", cfg["grammar"]["config"], "--out-prompts", cfg["prompt_subset"]["out_prompts"], "--out-constraints", cfg["prompt_subset"]["out_constraints"], "--out-generation-plan", cfg["prompt_subset"]["generation_plan"]])
    elif args.stage == "generate-images":
        cmd = [py, "generate_images.py", "--prompts", cfg["prompt_subset"]["out_prompts"], "--generation-plan", cfg["prompt_subset"]["generation_plan"], "--out", cfg["generation"]["generations_csv"], "--samples-per-prompt", str(cfg["generation"].get("samples_per_prompt", 1))]
        if args.dry_run: cmd.append("--dry-run")
        run(cmd)
    elif args.stage == "create-human-template":
        run([py, "create_human_label_template.py", "--generations", cfg["generation"]["generations_csv"], "--constraints", cfg["prompt_subset"]["out_constraints"], "--out", cfg["human_labels"]["labels_csv"]])


if __name__ == "__main__":
    main()
