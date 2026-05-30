from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

VALID_LABELS = {"pass", "fail", "ambiguous"}
NON_PASS_LABELS = {"fail", "ambiguous"}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})


def append_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "item"


def pluralize(obj: str) -> str:
    irregular = {
        "die": "dice",
        "box": "boxes",
        "small box": "small boxes",
        "glass bowl": "glass bowls",
        "ceramic cup": "ceramic cups",
        "plastic container": "plastic containers",
        "tray": "trays",
        "sheet of paper": "sheets of paper",
        "cutting board": "cutting boards",
        "safety pin": "safety pins",
        "paper clip": "paper clips",
        "bottle cap": "bottle caps",
        "rubber band": "rubber bands",
    }
    if obj in irregular:
        return irregular[obj]
    if obj.endswith("s"):
        return obj
    return obj + "s"


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
    