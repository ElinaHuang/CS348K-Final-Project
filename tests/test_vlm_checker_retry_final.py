from pathlib import Path
import pytest

import run_vlm_checker


def test_run_checker_retries_rate_limit_then_succeeds(monkeypatch):
    generations = [
        {
            "image_id": "image_001",
            "prompt_id": "prompt_001",
            "provider": "openai",
            "model_name": "gpt-image-1",
            "image_path": "dummy_image.png",
            "generation_status": "success",
        }
    ]

    constraints = [
        {
            "constraint_id": "constraint_001",
            "prompt_id": "prompt_001",
            "constraint_type": "object_identity",
            "check_text": "Does the image contain a small paper clip?",
        }
    ]

    call_count = {"n": 0}

    def fake_call_vlm_checker(image_path, checker_prompt, model_name, provider):
        call_count["n"] += 1

        if call_count["n"] <= 2:
            raise Exception("Rate limit reached. Please try again in 114ms.")

        return '{"label": "pass", "reason": "The small paper clip is visible."}'

    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(run_vlm_checker, "call_vlm_checker", fake_call_vlm_checker)
    monkeypatch.setattr(run_vlm_checker.time, "sleep", fake_sleep)

    rows = run_vlm_checker.run_checker(
        generations=generations,
        constraints=constraints,
        model_name="gpt-4.1",
        provider="openai",
        dry_run=False,
        max_retries=4,
    )

    assert call_count["n"] == 3
    assert len(sleep_calls) == 2

    # The retry wait should be at least the fallback wait.
    # attempt 0 fallback = 0.5
    # attempt 1 fallback = 1.0
    assert sleep_calls[0] >= 0.5
    assert sleep_calls[1] >= 1.0

    assert len(rows) == 1
    assert rows[0]["image_id"] == "image_001"
    assert rows[0]["constraint_id"] == "constraint_001"
    assert rows[0]["vlm_label"] == "pass"
    assert rows[0]["parse_status"] == "success"


def test_run_checker_stops_after_max_retries(monkeypatch):
    generations = [
        {
            "image_id": "image_001",
            "prompt_id": "prompt_001",
            "provider": "openai",
            "model_name": "gpt-image-1",
            "image_path": "dummy_image.png",
            "generation_status": "success",
        }
    ]

    constraints = [
        {
            "constraint_id": "constraint_001",
            "prompt_id": "prompt_001",
            "constraint_type": "object_identity",
            "check_text": "Does the image contain a small paper clip?",
        }
    ]

    call_count = {"n": 0}

    def always_rate_limited(image_path, checker_prompt, model_name, provider):
        call_count["n"] += 1
        raise Exception("Rate limit reached. Please try again in 114ms.")

    monkeypatch.setattr(run_vlm_checker, "call_vlm_checker", always_rate_limited)
    monkeypatch.setattr(run_vlm_checker.time, "sleep", lambda seconds: None)

    with pytest.raises(Exception, match="Rate limit reached"):
        run_vlm_checker.run_checker(
            generations=generations,
            constraints=constraints,
            model_name="gpt-4.1",
            provider="openai",
            dry_run=False,
            max_retries=3,
        )

    assert call_count["n"] == 3


def test_run_checker_does_not_retry_non_retryable_error(monkeypatch):
    generations = [
        {
            "image_id": "image_001",
            "prompt_id": "prompt_001",
            "provider": "openai",
            "model_name": "gpt-image-1",
            "image_path": "dummy_image.png",
            "generation_status": "success",
        }
    ]

    constraints = [
        {
            "constraint_id": "constraint_001",
            "prompt_id": "prompt_001",
            "constraint_type": "object_identity",
            "check_text": "Does the image contain a small paper clip?",
        }
    ]

    call_count = {"n": 0}

    def fake_code_error(image_path, checker_prompt, model_name, provider):
        call_count["n"] += 1
        raise ValueError("Bad local code path.")

    monkeypatch.setattr(run_vlm_checker, "call_vlm_checker", fake_code_error)
    monkeypatch.setattr(run_vlm_checker.time, "sleep", lambda seconds: None)

    with pytest.raises(ValueError):
        run_vlm_checker.run_checker(
            generations=generations,
            constraints=constraints,
            model_name="gpt-4.1",
            provider="openai",
            dry_run=False,
            max_retries=5,
        )

    assert call_count["n"] == 1