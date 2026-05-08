import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from utils import parse_vlm_response


def test_parse_valid_json():
    label, reason, status = parse_vlm_response('{"label": "pass", "reason": "Looks correct."}')
    assert label == "pass"
    assert reason == "Looks correct."
    assert status == "success"


def test_parse_json_substring():
    raw = 'Here is the answer: {"label": "fail", "reason": "Only two cups."}'
    label, reason, status = parse_vlm_response(raw)
    assert label == "fail"
    assert "two cups" in reason
    assert status == "success"


def test_fallback_label_line():
    label, reason, status = parse_vlm_response("Label: ambiguous\nReason: The objects overlap.")
    assert label == "ambiguous"
    assert status == "fallback"


def test_parse_error_defaults_ambiguous():
    label, reason, status = parse_vlm_response("I cannot provide that.")
    assert label == "ambiguous"
    assert status == "parse_error"
