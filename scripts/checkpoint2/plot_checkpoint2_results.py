from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

from utils import ensure_dir, read_csv


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def plot_bar(rows: List[Dict[str, str]], label_key: str, value_key: str, title: str, ylabel: str, out_path: str) -> None:
    if not rows:
        print(f"No rows available for {out_path}")
        return
    labels = [r.get(label_key, "") for r in rows]
    values = [to_float(r.get(value_key, "0")) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results/checkpoint2")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    ensure_dir(fig_dir)

    image_dim = read_csv(results_dir / "image_controllability_by_dimension.csv")
    prompt_family_rows = [r for r in image_dim if r.get("dimension") == "prompt_family"]
    composition_rows = [r for r in image_dim if r.get("dimension") == "composition"]

    plot_bar(
        prompt_family_rows, "value", "human_pass_rate",
        "Image-level pass rate by prompt family", "Human pass rate",
        str(fig_dir / "image_pass_rate_by_prompt_family.png"),
    )
    plot_bar(
        composition_rows, "value", "human_pass_rate",
        "Image-level pass rate by composition", "Human pass rate",
        str(fig_dir / "image_pass_rate_by_composition.png"),
    )

    constraint_rows = read_csv(results_dir / "constraint_checker_metrics_by_type.csv")
    plot_bar(
        constraint_rows, "value", "agreement",
        "VLM-human agreement by constraint type", "Agreement",
        str(fig_dir / "checker_agreement_by_constraint_type.png"),
    )

    repair_type_rows = read_csv(results_dir / "repair_success_by_constraint_type.csv")
    plot_bar(
        repair_type_rows, "target_constraint_type", "target_fixed_rate",
        "Repair target fixed rate by constraint type", "Target fixed rate",
        str(fig_dir / "repair_success_by_constraint_type.png"),
    )


if __name__ == "__main__":
    main()
