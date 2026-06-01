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
        "generate-prompts", "generate-images", "create-human-template", "run-vlm", "analyze-initial",
        "generate-repairs-human", "generate-repairs-vlm", "generate-repair-images-human", "generate-repair-images-vlm",
        "create-repair-label-template-human", "create-repair-label-template-vlm"
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
    elif args.stage == "run-vlm":
        cmd = [py, "run_vlm_checker.py", "--generations", cfg["generation"]["generations_csv"], "--constraints", cfg["prompt_subset"]["out_constraints"], "--out", cfg["vlm_checker"]["labels_csv"], "--provider", cfg["vlm_checker"]["provider"], "--model-name", cfg["vlm_checker"]["model_name"]]
        if args.dry_run: cmd.append("--dry-run")
        run(cmd)
    elif args.stage == "analyze-initial":
        run([py, "analyze_checker.py", "--human", cfg["human_labels"]["labels_csv"], "--vlm", cfg["vlm_checker"]["labels_csv"], "--prompts", cfg["prompt_subset"]["out_prompts"], "--generations", cfg["generation"]["generations_csv"], "--out-dir", cfg["analysis"]["out_dir"]])
    elif args.stage in {"generate-repairs-human", "generate-repairs-vlm"}:
        source = "human" if args.stage.endswith("human") else "vlm"
        run([py, "generate_repair_prompts.py", "--config", args.config, "--trigger-source", source])
    elif args.stage in {"generate-repair-images-human", "generate-repair-images-vlm"}:
        source = "human" if args.stage.endswith("human") else "vlm"
        r = cfg["repair"]
        prompts = r[f"repaired_prompts_{source}_csv"]
        # Create a repair generation plan that keeps each repaired image on the same
        # T2I provider/model as the source image. This makes before/after repair
        # comparisons isolate the prompt repair rather than changing generators.
        plan = f"../data/repaired/{source}_triggered/repaired_generation_plan.csv"
        import csv
        grammar_cfg = load_config(cfg["grammar"]["config"])
        model_cfgs = {
            (m.get("provider", ""), m.get("model_name", "")): m
            for m in grammar_cfg.get("dataset", {}).get("t2i_models", [])
        }
        Path(plan).parent.mkdir(parents=True, exist_ok=True)
        with open(prompts, newline='', encoding='utf-8') as f, open(plan, 'w', newline='', encoding='utf-8') as out:
            reader = list(csv.DictReader(f))
            writer = csv.DictWriter(out, fieldnames=["generation_job_id","prompt_id","provider","model_name","assignment_type","image_dir","size","quality","aspect_ratio"])
            writer.writeheader()
            for i, row in enumerate(reader, 1):
                provider = row.get("source_provider", "")
                model_name = row.get("source_model_name", "")
                model_cfg = model_cfgs.get((provider, model_name), {})
                writer.writerow({
                    "generation_job_id": f"repair_job_{source}_{i:04d}",
                    "prompt_id": row["prompt_id"],
                    "provider": provider,
                    "model_name": model_name,
                    "assignment_type": f"repair_{source}_same_model",
                    "image_dir": r[f"repaired_{source}_image_dir"],
                    "size": model_cfg.get("size", ""),
                    "quality": model_cfg.get("quality", ""),
                    "aspect_ratio": model_cfg.get("aspect_ratio", ""),
                })
        cmd = [py, "generate_images.py", "--prompts", prompts, "--generation-plan", plan, "--out", r[f"repaired_{source}_generations_csv"]]
        if args.dry_run: cmd.append("--dry-run")
        run(cmd)
    elif args.stage in {"create-repair-label-template-human", "create-repair-label-template-vlm"}:
        source = "human" if args.stage.endswith("human") else "vlm"; r = cfg["repair"]
        run([py, "create_repair_label_template.py", "--config", args.config, "--repaired-generations", r[f"repaired_{source}_generations_csv"], "--repair-targets", r[f"repair_targets_{source}_csv"], "--before-labels", cfg["human_labels"]["labels_csv"], "--constraints", cfg["prompt_subset"]["out_constraints"], "--out", r[f"repaired_{source}_labels_csv"]])


if __name__ == "__main__":
    main()
