from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from utils import ensure_dir, read_csv, write_csv, load_environment

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


def call_openai_image_api(prompt: str, model_name: str, size: str, quality: str) -> bytes:
    """Generate one image using the OpenAI Images API.

    Requires OPENAI_API_KEY in the environment and the openai Python package.
    """
    load_environment()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK with: pip install openai") from exc

    client = OpenAI()
    response = client.images.generate(
        model=model_name,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    item = response.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    if getattr(item, "url", None):
        # Avoid adding a requests dependency; urllib is enough here.
        from urllib.request import urlopen
        with urlopen(item.url) as f:
            return f.read()
    raise RuntimeError("OpenAI image response did not contain b64_json or url.")


def call_t2i_model(prompt: str, provider: str, model_name: str, size: str, quality: str) -> bytes:
    if provider == "openai":
        return call_openai_image_api(prompt, model_name=model_name, size=size, quality=quality)
    if provider == "stub":
        call_t2i_model_stub(prompt, model_name=model_name)
    raise ValueError(f"Unsupported T2I provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--out", default="../data/generations/generations.csv")
    parser.add_argument("--image-dir", default="../data/images/model_stub")
    parser.add_argument("--provider", default="stub", choices=["stub", "openai"])
    parser.add_argument("--model-name", default="model_stub", choices=["model_stub", "gpt-image-1"])
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--quality", default="low")
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
            timestamp = datetime.now(timezone.utc).isoformat()

            if not args.dry_run:
                image_bytes = call_t2i_model(
                    prompt=p["prompt"],
                    provider=args.provider,
                    model_name=args.model_name,
                    size=args.size,
                    quality=args.quality,
                )
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

            rows.append({
                "image_id": image_id,
                "prompt_id": p["prompt_id"],
                "model_name": args.model_name,
                "sample_id": str(sample_id),
                "image_path": image_path,
                "generation_status": status,
                "seed": "",
                "timestamp": timestamp,
            })

    write_csv(args.out, rows, GENERATION_FIELDS)
    print(f"Wrote {len(rows)} generation records to {args.out}")
    if args.dry_run:
        print("Dry run only: no images were generated.")


if __name__ == "__main__":
    main()
