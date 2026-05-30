from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from utils import ensure_dir, load_environment, read_csv, slugify, write_csv

GENERATION_FIELDS = [
    "image_id", "generation_job_id", "prompt_id", "provider", "model_name", "sample_id",
    "image_path", "generation_status", "prompt", "timestamp", "error_message",
]


def call_openai_image_api(prompt: str, model_name: str, size: str = "1024x1024", quality: str = "low") -> bytes:
    load_environment()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK with: pip install openai") from exc
    client = OpenAI()
    response = client.images.generate(model=model_name, prompt=prompt, size=size or "1024x1024", quality=quality or "low", n=1)
    return base64.b64decode(response.data[0].b64_json)


def call_google_image_api(prompt: str, model_name: str, aspect_ratio: str = "1:1") -> bytes:
    load_environment()
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Install the Google GenAI SDK with: pip install google-genai") from exc
    client = genai.Client()
    # Gemini native image models use generate_content and return an inline image part.
    if "gemini" in model_name.lower() or "banana" in model_name.lower():
        kwargs = {"response_modalities": ["TEXT", "IMAGE"]}
        # Some SDK versions support image_config; if not, retry without it.
        try:
            kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio or "1:1")
            config = types.GenerateContentConfig(**kwargs)
        except Exception:
            config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
        response = client.models.generate_content(model=model_name, contents=prompt, config=config)
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                return part.inline_data.data
        raise RuntimeError(f"No image returned. Text response: {getattr(response, 'text', '')}")
    # Imagen models use generate_images.
    response = client.models.generate_images(
        model=model_name,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=aspect_ratio or "1:1"),
    )
    return response.generated_images[0].image.image_bytes


def call_t2i_model(prompt: str, provider: str, model_name: str, size: str = "", quality: str = "", aspect_ratio: str = "1:1") -> bytes:
    if provider == "openai":
        return call_openai_image_api(prompt, model_name=model_name, size=size or "1024x1024", quality=quality or "low")
    if provider == "google":
        return call_google_image_api(prompt, model_name=model_name, aspect_ratio=aspect_ratio or "1:1")
    if provider == "stub":
        return b""
    raise ValueError(f"Unsupported T2I provider: {provider}")


def generate_from_plan(prompts: List[Dict[str, str]], plan: List[Dict[str, str]], samples_per_prompt: int = 1, dry_run: bool = False) -> List[Dict[str, str]]:
    prompt_by_id = {p["prompt_id"]: p for p in prompts}
    rows: List[Dict[str, str]] = []
    for job in plan:
        prompt_row = prompt_by_id.get(job["prompt_id"])
        if not prompt_row:
            continue
        provider = job.get("provider", "")
        model_name = job.get("model_name", "")
        image_dir = Path(job.get("image_dir") or "data/images")
        ensure_dir(image_dir)
        for sample_id in range(samples_per_prompt):
            image_id = f"{job['generation_job_id']}_{slugify(model_name)}_{sample_id:02d}"
            image_path = image_dir / f"{image_id}.png"
            status, error = "success", ""
            if not dry_run:
                try:
                    image_bytes = call_t2i_model(
                        prompt=prompt_row["prompt"], provider=provider, model_name=model_name,
                        size=job.get("size", ""), quality=job.get("quality", ""), aspect_ratio=job.get("aspect_ratio", "1:1"),
                    )
                    image_path.write_bytes(image_bytes)
                except Exception as exc:
                    status, error = "error", str(exc)
            rows.append({
                "image_id": image_id,
                "generation_job_id": job["generation_job_id"],
                "prompt_id": job["prompt_id"],
                "provider": provider,
                "model_name": model_name,
                "sample_id": str(sample_id),
                "image_path": str(image_path),
                "generation_status": status,
                "prompt": prompt_row["prompt"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error_message": error,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="../data/prompts/prompts.csv")
    parser.add_argument("--generation-plan", default="../data/generations/generation_plan.csv")
    parser.add_argument("--out", default="../data/generations/generations.csv")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prompts = read_csv(args.prompts)
    plan = read_csv(args.generation_plan)
    if not prompts:
        raise FileNotFoundError(f"No prompts found at {args.prompts}")
    if not plan:
        raise FileNotFoundError(f"No generation plan found at {args.generation_plan}")
    rows = generate_from_plan(prompts, plan, samples_per_prompt=args.samples_per_prompt, dry_run=args.dry_run)
    write_csv(args.out, rows, GENERATION_FIELDS)
    print(f"Wrote {len(rows)} generation records to {args.out}")


if __name__ == "__main__":
    main()
