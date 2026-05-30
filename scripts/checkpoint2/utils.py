from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

VALID_LABELS = {"pass", "fail", "ambiguous"}


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
    return value


def pluralize(obj: str) -> str:
    if obj == "box":
        return "boxes"
    if obj == "toy car":
        return "toy cars"
    if obj.endswith("s"):
        return obj
    return obj + "s"


def normalize_label(label: str | None) -> str:
    if label is None:
        return "ambiguous"
    label = label.strip().lower()
    label = label.replace('"', "").replace("'", "")
    if label in VALID_LABELS:
        return label
    if "pass" in label:
        return "pass"
    if "fail" in label:
        return "fail"
    if "ambiguous" in label or "unclear" in label or "unsure" in label or "uncertain" in label:
        return "ambiguous"
    return "ambiguous"


def parse_vlm_response(raw_response: str) -> Tuple[str, str, str]:
    """Parse a VLM checker response.

    Returns:
        (label, reason, parse_status)
        parse_status in {"success", "fallback", "parse_error"}
    """
    raw = (raw_response or "").strip()
    if not raw:
        return "ambiguous", "Empty response.", "parse_error"

    # Try direct JSON.
    try:
        data = json.loads(raw)
        label = normalize_label(str(data.get("label", "")))
        reason = str(data.get("reason", "")).strip()
        return label, reason, "success"
    except Exception:
        pass

    # Try to extract JSON substring.
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start : end + 1])
            label = normalize_label(str(data.get("label", "")))
            reason = str(data.get("reason", "")).strip()
            return label, reason, "success"
    except Exception:
        pass

    # Fallback label search.
    lowered = raw.lower()
    m = re.search(r"label\s*[:=]\s*(pass|fail|ambiguous)", lowered)
    if m:
        label = normalize_label(m.group(1))
        return label, raw[:300], "fallback"

    found = [lab for lab in VALID_LABELS if re.search(rf"\b{lab}\b", lowered)]
    if len(found) == 1:
        return found[0], raw[:300], "fallback"

    return "ambiguous", raw[:300], "parse_error"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0

def load_environment() -> None:
    """Load local environment variables from a .env file if python-dotenv is installed.

    This is useful for local development. In deployed or shell-based settings,
    environment variables can still be provided directly through the shell.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
