import analyze_repair as analyze_repair_module
from analyze_repair import analyze_repair, read_repair_inputs_for_source


def _summary_dict(results):
    return {r["metric"]: r["value"] for r in results["summary"]}


def _by_value(rows, value_key="value"):
    return {r[value_key]: r for r in rows}


def test_analyze_repair_tracks_fixed_targets_regressions_and_breakdowns():
    repaired_prompts = [
        {
            "repair_id": "r1",
            "source_image_id": "img_good",
            "source_prompt_id": "p1",
            "repair_strategy": "repair_all_failed",
            "trigger_label_source": "vlm",
            "source_provider": "openai",
            "source_model_name": "gpt-image-1",
        },
        {
            "repair_id": "r2",
            "source_image_id": "img_bad",
            "source_prompt_id": "p2",
            "repair_strategy": "repair_all_failed",
            "trigger_label_source": "human",
            "source_provider": "google",
            "source_model_name": "gemini-2.5-flash-image",
        },
    ]
    prompts = [
        {
            "prompt_id": "p1",
            "prompt_family": "single_spatial",
            "scene_context_type": "simple",
            "composition": "single",
        },
        {
            "prompt_id": "p2",
            "prompt_family": "combined_attribute_cardinality",
            "scene_context_type": "natural",
            "composition": "combined",
        },
    ]
    repair_targets = [
        {
            "repair_id": "r1",
            "source_image_id": "img_good",
            "source_prompt_id": "p1",
            "target_constraint_id": "c1",
            "target_constraint_type": "spatial_relation",
            "before_label": "fail",
        },
        {
            "repair_id": "r2",
            "source_image_id": "img_bad",
            "source_prompt_id": "p2",
            "target_constraint_id": "c3",
            "target_constraint_type": "cardinality",
            "before_label": "fail",
        },
    ]
    repaired_labels = [
        # r1 is unnecessary under human labels: before image was fully pass, after regresses.
        {
            "repair_id": "r1",
            "constraint_id": "c1",
            "constraint_type": "spatial_relation",
            "before_label": "pass",
            "after_label": "fail",
        },
        {
            "repair_id": "r1",
            "constraint_id": "c2",
            "constraint_type": "object_identity",
            "before_label": "pass",
            "after_label": "pass",
        },
        # r2 is a successful human-triggered repair with no regression.
        {
            "repair_id": "r2",
            "constraint_id": "c3",
            "constraint_type": "cardinality",
            "before_label": "fail",
            "after_label": "pass",
        },
        {
            "repair_id": "r2",
            "constraint_id": "c4",
            "constraint_type": "attribute",
            "before_label": "pass",
            "after_label": "pass",
        },
    ]

    results = analyze_repair(repaired_prompts, repair_targets, repaired_labels, prompts=prompts)
    image_by_id = {r["repair_id"]: r for r in results["image_results"]}
    target_by_id = {r["repair_id"]: r for r in results["target_results"]}
    summary = _summary_dict(results)

    assert target_by_id["r1"]["target_fixed"] == "False"
    assert target_by_id["r2"]["target_fixed"] == "True"
    assert image_by_id["r1"]["is_unnecessary_repair"] == "True"
    assert image_by_id["r1"]["unnecessary_repair_caused_regression"] == "True"
    assert image_by_id["r1"]["num_hard_regressions"] == "1"
    assert image_by_id["r1"]["prompt_family"] == "single_spatial"
    assert image_by_id["r1"]["scene_context_type"] == "simple"
    assert image_by_id["r2"]["image_fixed"] == "True"
    assert image_by_id["r2"]["has_regression"] == "False"
    assert image_by_id["r2"]["prompt_family"] == "combined_attribute_cardinality"
    assert image_by_id["r2"]["scene_context_type"] == "natural"

    assert summary["num_repair_attempts"] == "2"
    assert summary["target_fixed_rate"] == "0.5000"
    assert summary["repair_regression_rate"] == "0.5000"
    assert summary["unnecessary_repair_count"] == "1"
    assert summary["unnecessary_repair_regression_count"] == "1"

    # Constraint-level metrics should align with initial generation dimensions.
    by_type = _by_value(results["constraint_metrics_by_type"])
    assert set(by_type) == {"attribute", "cardinality", "object_identity", "spatial_relation"}
    assert by_type["cardinality"]["before_pass_count"] == "0"
    assert by_type["cardinality"]["before_fail_count"] == "1"
    assert by_type["cardinality"]["after_pass_count"] == "1"
    assert by_type["cardinality"]["after_fail_count"] == "0"
    assert by_type["cardinality"]["before_pass_rate"] == "0.0000"
    assert by_type["cardinality"]["after_pass_rate"] == "1.0000"
    assert by_type["spatial_relation"]["before_pass_count"] == "1"
    assert by_type["spatial_relation"]["after_fail_count"] == "1"
    assert by_type["spatial_relation"]["hard_regression_count"] == "1"

    by_constraint_model = _by_value(results["constraint_metrics_by_model"])
    assert set(by_constraint_model) == {"gpt-image-1", "gemini-2.5-flash-image"}

    by_constraint_scene = _by_value(results["constraint_metrics_by_scene_context_type"])
    assert set(by_constraint_scene) == {"simple", "natural"}

    # Image-level metrics should also align with initial generation dimensions.
    by_family = _by_value(results["image_metrics_by_prompt_family"])
    assert set(by_family) == {"single_spatial", "combined_attribute_cardinality"}
    assert by_family["combined_attribute_cardinality"]["before_fail_count"] == "1"
    assert by_family["combined_attribute_cardinality"]["after_pass_count"] == "1"
    assert by_family["combined_attribute_cardinality"]["nonpass_to_pass_rate"] == "1.0000"
    assert by_family["single_spatial"]["before_pass_count"] == "1"
    assert by_family["single_spatial"]["after_fail_count"] == "1"
    assert by_family["single_spatial"]["pass_to_nonpass_rate"] == "1.0000"

    by_image_model = _by_value(results["image_metrics_by_model"])
    assert set(by_image_model) == {"gpt-image-1", "gemini-2.5-flash-image"}

    by_image_scene = _by_value(results["image_metrics_by_scene_context_type"])
    assert set(by_image_scene) == {"simple", "natural"}


def test_analyze_repair_keeps_backward_compatible_type_results_alias():
    repaired_prompts = [
        {
            "repair_id": "r1",
            "source_image_id": "img1",
            "source_prompt_id": "p1",
            "trigger_label_source": "human",
        }
    ]
    repair_targets = [
        {
            "repair_id": "r1",
            "source_image_id": "img1",
            "source_prompt_id": "p1",
            "target_constraint_id": "c1",
            "target_constraint_type": "attribute",
            "before_label": "ambiguous",
        }
    ]
    repaired_labels = [
        {
            "repair_id": "r1",
            "constraint_id": "c1",
            "constraint_type": "attribute",
            "before_label": "ambiguous",
            "after_label": "pass",
        }
    ]

    results = analyze_repair(repaired_prompts, repair_targets, repaired_labels)

    assert "type_results" in results
    type_rows = _by_value(results["type_results"])
    assert type_rows["attribute"]["nonpass_to_pass_rate"] == "1.0000"



def test_read_repair_inputs_for_combined_concatenates_human_and_vlm(monkeypatch):
    repair_cfg = {
        "repaired_prompts_human_csv": "human_prompts.csv",
        "repair_targets_human_csv": "human_targets.csv",
        "repaired_human_labels_csv": "human_labels.csv",
        "repaired_prompts_vlm_csv": "vlm_prompts.csv",
        "repair_targets_vlm_csv": "vlm_targets.csv",
        "repaired_vlm_labels_csv": "vlm_labels.csv",
    }
    fake_files = {
        "human_prompts.csv": [{"repair_id": "repair_human_001"}],
        "human_targets.csv": [{"repair_id": "repair_human_001", "target_constraint_id": "c1"}],
        "human_labels.csv": [{"repair_id": "repair_human_001", "constraint_id": "c1"}],
        "vlm_prompts.csv": [{"repair_id": "repair_vlm_001"}],
        "vlm_targets.csv": [{"repair_id": "repair_vlm_001", "target_constraint_id": "c2"}],
        "vlm_labels.csv": [{"repair_id": "repair_vlm_001", "constraint_id": "c2"}],
    }

    def fake_read_csv(path):
        return fake_files[path]

    monkeypatch.setattr(analyze_repair_module, "read_csv", fake_read_csv)

    prompts, targets, labels = read_repair_inputs_for_source(repair_cfg, "combined")

    assert [p["repair_id"] for p in prompts] == ["repair_human_001", "repair_vlm_001"]
    assert [t["target_constraint_id"] for t in targets] == ["c1", "c2"]
    assert [l["constraint_id"] for l in labels] == ["c1", "c2"]
