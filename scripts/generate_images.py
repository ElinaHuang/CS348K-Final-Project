from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from utils import ensure_dir, read_csv, write_csv

GENERATION_FIELDS = [
    "image_id",
    "prompt_id",
    "model_name",
    "sample_id",
    "image_path",
    "generation_status",
    "seed",
    "timestamp",
]


def call_t2i_model_stub(prompt: str, model_name: str) -> bytes:
    """Placeholder for a real T2I API call.

    For checkpoint 1, you may skip this script and manually populate
    generations.csv after generating images elsewhere.
    """
    raise NotImplementedError(
        "T2I generation is not implemented yet. Generate images externally and fill "
        "data/generations/generations.csv, or replace call_t2i_model_stub with an API call."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--out", default="../data/generations/generations.csv")
    parser.add_argument("--image-dir", default="../data/images/model_stub")
    parser.add_argument("--model-name", default="model_stub")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Create planned generation records without calling an API.")
    args = parser.parse_args()

    prompts = read_csv(args.prompts)
    ensure_dir(args.image_dir)

    rows: List[Dict[str, str]] = []
    for p in prompts:
        for sample_id in range(args.samples_per_prompt):
            image_id = f"{p['prompt_id']}_{args.model_name}_{sample_id}"
            image_path = str(Path(args.image_dir) / f"{image_id}.png")
            status = "planned" if args.dry_run else "success"

            if not args.dry_run:
                _image_bytes = call_t2i_model_stub(p["prompt"], args.model_name)
                # Save bytes here after implementing a real API.
                # with open(image_path, "wb") as f:
                #     f.write(_image_bytes)

            rows.append({
                "image_id": image_id,
                "prompt_id": p["prompt_id"],
                "model_name": args.model_name,
                "sample_id": str(sample_id),
                "image_path": image_path,
                "generation_status": status,
                "seed": "",
                "timestamp": "",
            })

    write_csv(args.out, rows, GENERATION_FIELDS)
    print(f"Wrote {len(rows)} generation records to {args.out}")
    if args.dry_run:
        print("Dry run only: no images were generated.")


if __name__ == "__main__":
    main()
