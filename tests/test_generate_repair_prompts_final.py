from pathlib import Path

import yaml

from generate_repair_prompts import generate_repair_records
from utils import write_csv


def _write_config(tmp_path: Path, source: str) -> Path:
    prompts_path = tmp_path / "prompts.csv"
    constraints_path = tmp_path / "constraints.csv"
    generations_path = tmp_path / "generations.csv"
    human_labels_path = tmp_path / "human_labels.csv"
    vlm_labels_path = tmp_path / "vlm_labels.csv"

    write_csv(prompts_path, [{"prompt_id": "p1", "prompt": "Original prompt."}], ["prompt_id", "prompt"])
    write_csv(constraints_path, [
        {"constraint_id": "c1", "prompt_id": "p1", "constraint_type": "object_identity", "check_text": "The object should be a paper clip.", "object_1": "paper clip", "object_2": "", "relation_text": ""},
        {"constraint_id": "c2", "prompt_id": "p1", "constraint_type": "cardinality", "check_text": "There should be exactly 5 objects.", "object_1": "paper clip", "object_2": "", "relation_text": ""},
    ], ["constraint_id", "prompt_id", "constraint_type", "check_text", "object_1", "object_2", "relation_text"])
    write_csv(generations_path, [{
        "image_id": "img1", "generation_job_id": "job1", "prompt_id": "p1",
        "provider": "google", "model_name": "gemini-2.5-flash-image", "image_path": "data/images/img1.png",
    }], ["image_id", "generation_job_id", "prompt_id", "provider", "model_name", "image_path"])
    write_csv(human_labels_path, [
        {"image_id": "img1", "prompt_id": "p1", "constraint_id": "c1", "human_label": "fail"},
        {"image_id": "img1", "prompt_id": "p1", "constraint_id": "c2", "human_label": "pass"},
    ], ["image_id", "prompt_id", "constraint_id", "human_label"])
    write_csv(vlm_labels_path, [
        {"image_id": "img1", "prompt_id": "p1", "constraint_id": "c1", "vlm_label": "pass"},
        {"image_id": "img1", "prompt_id": "p1", "constraint_id": "c2", "vlm_label": "ambiguous"},
    ], ["image_id", "prompt_id", "constraint_id", "vlm_label"])

    config = {
        "prompt_subset": {"out_prompts": str(prompts_path), "out_constraints": str(constraints_path)},
        "generation": {"generations_csv": str(generations_path)},
        "human_labels": {"labels_csv": str(human_labels_path)},
        "vlm_checker": {"labels_csv": str(vlm_labels_path)},
        "repair": {"trigger_labels": ["fail", "ambiguous"], "trigger_label_source": source},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_human_triggered_repair_inherits_source_model(tmp_path):
    config_path = _write_config(tmp_path, "human")
    config = yaml.safe_load(config_path.read_text())
    repaired_prompts, repair_targets = generate_repair_records(config, trigger_source="human")

    assert len(repaired_prompts) == 1
    assert len(repair_targets) == 1
    row = repaired_prompts[0]
    assert row["source_provider"] == "google"
    assert row["source_model_name"] == "gemini-2.5-flash-image"
    assert row["trigger_label_source"] == "human"
    assert "Important generation instructions" in row["repaired_prompt"]
    assert repair_targets[0]["target_constraint_id"] == "c1"


def test_vlm_triggered_repair_uses_vlm_labels(tmp_path):
    config_path = _write_config(tmp_path, "vlm")
    config = yaml.safe_load(config_path.read_text())
    repaired_prompts, repair_targets = generate_repair_records(config, trigger_source="vlm")

    assert len(repaired_prompts) == 1
    assert len(repair_targets) == 1
    assert repaired_prompts[0]["trigger_label_source"] == "vlm"
    assert repair_targets[0]["target_constraint_id"] == "c2"
    assert repair_targets[0]["before_label"] == "ambiguous"
