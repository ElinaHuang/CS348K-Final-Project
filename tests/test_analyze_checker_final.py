from analyze_checker import aggregate_to_image_level, by_group, image_rows_as_label_rows, merge_labels, rates


def test_analyze_checker_tracks_scene_context_and_image_level_labels():
    prompts = [
        {"prompt_id": "p1", "prompt_family": "single_spatial", "scene_context_type": "simple", "composition": "single"},
        {"prompt_id": "p2", "prompt_family": "combined_spatial_attribute", "scene_context_type": "natural", "composition": "combined"},
    ]
    generations = [
        {"image_id": "img1", "prompt_id": "p1", "provider": "openai", "model_name": "gpt-image-1"},
        {"image_id": "img2", "prompt_id": "p2", "provider": "google", "model_name": "gemini-2.5-flash-image"},
    ]
    human = [
        {"image_id": "img1", "prompt_id": "p1", "constraint_id": "c1", "constraint_type": "spatial_relation", "human_label": "pass"},
        {"image_id": "img1", "prompt_id": "p1", "constraint_id": "c2", "constraint_type": "object_identity", "human_label": "pass"},
        {"image_id": "img2", "prompt_id": "p2", "constraint_id": "c3", "constraint_type": "attribute", "human_label": "fail"},
        {"image_id": "img2", "prompt_id": "p2", "constraint_id": "c4", "constraint_type": "object_identity", "human_label": "pass"},
    ]
    vlm = [
        {"image_id": "img1", "constraint_id": "c1", "vlm_label": "pass"},
        {"image_id": "img1", "constraint_id": "c2", "vlm_label": "pass"},
        {"image_id": "img2", "constraint_id": "c3", "vlm_label": "pass"},
        {"image_id": "img2", "constraint_id": "c4", "vlm_label": "pass"},
    ]

    merged = merge_labels(human, vlm, prompts, generations)
    assert len(merged) == 4
    assert {r["scene_context_type"] for r in merged} == {"simple", "natural"}

    scene_rows = by_group(merged, "scene_context_type")
    assert {r["value"] for r in scene_rows} == {"simple", "natural"}

    image_rows = aggregate_to_image_level(merged)
    by_image = {r["image_id"]: r for r in image_rows}
    assert by_image["img1"]["human_image_label"] == "pass"
    assert by_image["img2"]["human_image_label"] == "fail"
    assert by_image["img2"]["vlm_image_label"] == "pass"

    image_metric = rates(image_rows_as_label_rows(image_rows))
    assert image_metric["num_examples"] == "2"
    assert image_metric["agreement"] == "0.5000"
