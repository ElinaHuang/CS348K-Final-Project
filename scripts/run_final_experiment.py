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
    parser.add_argument("--stage", required=True, choices=["generate-prompts"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    py = sys.executable

    if args.stage == "generate-prompts":
        run([py, "generate_prompts.py", "--config", cfg["grammar"]["config"], "--out-prompts", cfg["prompt_subset"]["out_prompts"], "--out-constraints", cfg["prompt_subset"]["out_constraints"], "--out-generation-plan", cfg["prompt_subset"]["generation_plan"]])
    
    
if __name__ == "__main__":
    main()
