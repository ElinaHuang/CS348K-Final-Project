import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_checker import (
    aggregate_labels,
    aggregate_to_image_level,
    confusion_matrix,
    constraint_metrics_by_type,
    failure_detection_metrics,
    human_pass_rate,
    image_controllability_metrics_by_dimension,
    image_level_agreement,
    image_rows_as_label_rows,
    metrics_by_cross_group,
    merge_labels,
    overall_agreement,
)


def test_overall_agreement():
    rows = [
        {"human_label": "pass", "vlm_label": "pass", "constraint_type": "spatial_relation"},
        {"human_label": "fail", "vlm_label": "pass", "constraint_type": "spatial_relation"},
        {"human_label": "ambiguous", "vlm_label": "ambiguous", "constraint_type": "cardinality"},
    ]
    assert abs(overall_agreement(rows) - 2 / 3) < 1e-9


def test_failure_detection_metrics_overall():
    rows = [
        {"human_label": "fail", "vlm_label": "fail", "constraint_type": "spatial_relation"},
        {"human_label": "pass", "vlm_label": "fail", "constraint_type": "spatial_relation"},
        {"human_label": "fail", "vlm_label": "pass", "constraint_type": "spatial_relation"},
        {"human_label": "pass", "vlm_label": "pass", "constraint_type": "spatial_relation"},
    ]
    metrics = failure_detection_metrics(rows)
    assert metrics["group"] == "overall"
    assert metrics["human_pass_rate"] == "0.5000"
    assert metrics["human_fail_rate"] == "0.5000"
    assert metrics["failure_precision"] == "0.5000"
    assert metrics["failure_recall"] == "0.5000"


def test_confusion_matrix_shape():
    rows = [
        {"human_label": "pass", "vlm_label": "pass"},
        {"human_label": "fail", "vlm_label": "ambiguous"},
    ]
    cm = confusion_matrix(rows)
    assert len(cm) == 3
    assert set(cm[0].keys()) == {"human_label", "vlm_pass", "vlm_fail", "vlm_ambiguous"}


def test_aggregate_labels_all_pass():
    assert aggregate_labels(["pass", "pass", "pass"]) == "pass"


def test_aggregate_labels_any_fail():
    assert aggregate_labels(["pass", "fail", "ambiguous"]) == "fail"


def test_aggregate_labels_ambiguous_when_no_fail_but_some_ambiguous():
    assert aggregate_labels(["pass", "ambiguous"]) == "ambiguous"


def test_merge_labels_adds_prompt_metadata():
    human_rows = [
        {
            "image_id": "img1",
            "prompt_id": "p1",
            "constraint_id": "c1",
            "constraint_type": "spatial_relation",
            "human_label": "pass",
        }
    ]
    vlm_rows = [
        {
            "image_id": "img1",
            "prompt_id": "p1",
            "constraint_id": "c1",
            "constraint_type": "spatial_relation",
            "vlm_label": "pass",
        }
    ]
    prompts = [
        {
            "prompt_id": "p1",
            "prompt_family": "spatial_layout",
            "scene_context_type": "simple",
            "scene_context": "on a clean tabletop",
            "composition": "single",
        }
    ]
    merged = merge_labels(human_rows, vlm_rows, prompts)
    assert merged[0]["prompt_family"] == "spatial_layout"
    assert merged[0]["scene_context_type"] == "simple"
    assert merged[0]["composition"] == "single"


def test_constraint_metrics_by_type_groups_only_constraint_type():
    rows = [
        {"constraint_type": "spatial_relation", "human_label": "pass", "vlm_label": "pass"},
        {"constraint_type": "spatial_relation", "human_label": "fail", "vlm_label": "pass"},
        {"constraint_type": "attribute", "human_label": "fail", "vlm_label": "fail"},
    ]
    metrics = constraint_metrics_by_type(rows)
    values = {m["value"] for m in metrics}
    assert values == {"spatial_relation", "attribute"}


def test_aggregate_to_image_level_preserves_prompt_dimensions():
    rows = [
        {
            "image_id": "img1",
            "prompt_id": "p1",
            "constraint_id": "c1",
            "constraint_type": "object_existence",
            "prompt_family": "spatial_layout",
            "scene_context_type": "simple",
            "scene_context": "on a clean tabletop",
            "composition": "single",
            "human_label": "pass",
            "vlm_label": "pass",
        },
        {
            "image_id": "img1",
            "prompt_id": "p1",
            "constraint_id": "c2",
            "constraint_type": "spatial_relation",
            "prompt_family": "spatial_layout",
            "scene_context_type": "simple",
            "scene_context": "on a clean tabletop",
            "composition": "single",
            "human_label": "fail",
            "vlm_label": "pass",
        },
    ]

    image_rows = aggregate_to_image_level(rows)
    assert image_rows[0]["human_image_label"] == "fail"
    assert image_rows[0]["vlm_image_label"] == "pass"
    assert image_rows[0]["prompt_family"] == "spatial_layout"
    assert image_rows[0]["scene_context_type"] == "simple"
    assert image_rows[0]["composition"] == "single"
    assert image_rows[0]["human_failed_constraints"] == "c2"


def test_image_level_agreement():
    image_rows = [
        {"human_image_label": "pass", "vlm_image_label": "pass"},
        {"human_image_label": "fail", "vlm_image_label": "pass"},
        {"human_image_label": "ambiguous", "vlm_image_label": "ambiguous"},
    ]
    assert abs(image_level_agreement(image_rows) - 2 / 3) < 1e-9


def test_image_rows_as_label_rows_keeps_prompt_dimensions():
    image_rows = [
        {
            "human_image_label": "pass",
            "vlm_image_label": "pass",
            "prompt_family": "spatial_layout",
            "scene_context_type": "simple",
            "composition": "single",
        }
    ]
    rows = image_rows_as_label_rows(image_rows)
    assert rows[0]["human_label"] == "pass"
    assert rows[0]["prompt_family"] == "spatial_layout"
    assert rows[0]["scene_context_type"] == "simple"
    assert rows[0]["composition"] == "single"


def test_image_controllability_metrics_by_dimension():
    image_rows = [
        {
            "human_image_label": "pass",
            "vlm_image_label": "pass",
            "prompt_family": "spatial_layout",
            "scene_context_type": "simple",
            "composition": "single",
        },
        {
            "human_image_label": "fail",
            "vlm_image_label": "pass",
            "prompt_family": "spatial_layout",
            "scene_context_type": "natural",
            "composition": "single",
        },
        {
            "human_image_label": "fail",
            "vlm_image_label": "fail",
            "prompt_family": "attribute_binding",
            "scene_context_type": "simple",
            "composition": "combined",
        },
    ]
    metrics = image_controllability_metrics_by_dimension(image_rows)
    by_dim_val = {(m["dimension"], m["value"]): m for m in metrics}
    assert by_dim_val[("scene_context_type", "simple")]["num_examples"] == 2
    assert by_dim_val[("scene_context_type", "simple")]["human_pass_rate"] == "0.5000"
    assert by_dim_val[("composition", "single")]["num_examples"] == 2


def test_metrics_by_cross_group_family_scene_composition_on_image_rows():
    rows = [
        {
            "prompt_family": "spatial_layout",
            "scene_context_type": "simple",
            "composition": "single",
            "human_label": "pass",
            "vlm_label": "pass",
        },
        {
            "prompt_family": "spatial_layout",
            "scene_context_type": "natural",
            "composition": "single",
            "human_label": "fail",
            "vlm_label": "pass",
        },
        {
            "prompt_family": "attribute_binding",
            "scene_context_type": "simple",
            "composition": "combined",
            "human_label": "fail",
            "vlm_label": "fail",
        },
    ]
    metrics = metrics_by_cross_group(rows, ["prompt_family", "scene_context_type", "composition"])
    assert len(metrics) == 3
    target = [
        m for m in metrics
        if m["prompt_family"] == "spatial_layout"
        and m["scene_context_type"] == "natural"
        and m["composition"] == "single"
    ][0]
    assert target["human_fail_rate"] == "1.0000"


def test_human_pass_rate():
    rows = [
        {"human_label": "pass"},
        {"human_label": "fail"},
        {"human_label": "ambiguous"},
        {"human_label": "pass"},
    ]
    assert human_pass_rate(rows) == 0.5
