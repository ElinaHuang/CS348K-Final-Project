from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-prompts", action="store_true")
    parser.add_argument("--create-human-template", action="store_true")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--dry-run-vlm", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all non-image-generation checkpoint steps.")
    args = parser.parse_args()

    py = sys.executable

    if args.all or args.generate_prompts:
        run([py, "generate_prompts.py"])

    if args.all or args.create_human_template:
        run([py, "create_human_label_template.py"])

    if args.all or args.run_vlm:
        cmd = [py, "run_vlm_checker.py"]
        if args.dry_run_vlm:
            cmd.append("--dry-run")
        run(cmd)

    if args.all or args.analyze:
        run([py, "analyze_checker.py"])


if __name__ == "__main__":
    main()
