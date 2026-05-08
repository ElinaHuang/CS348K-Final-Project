# Checkpoint 1: Evaluation Pipeline

## Goal of this checkpoint

The goal of Checkpoint 1 is to make the project evaluation concrete and runnable.

This checkpoint does **not** focus on introducing a new generation method yet. Instead, it focuses on defining and implementing the evaluation pipeline for constraint-level prompt adherence in text-to-image generation.

At this stage, the main goal is to answer:

- What exactly should be measured?
- What prompt/constraint structure will be tested?
- How will generated images be labeled as successful, failed, or ambiguous?
- How will a VLM checker be evaluated against human labels?
- How will image-level controllability be analyzed across prompt conditions?

For this checkpoint, I validate the evaluation pipeline using mock images and mock labels. These mock files are only used to make sure the pipeline runs end-to-end before replacing them with real T2I generations and real VLM checker outputs.

## Project questions

This project aims to answer two main questions.

1. **Can a VLM checker automatically evaluate large batches of T2I outputs?**
   Instead of manually inspecting every generated image, the system uses a VLM checker to judge whether each image satisfies the expected visual constraints. I evaluate this by comparing VLM checker outputs against human labels at both the atomic constraint level and the whole-image level.

2. **Can a constraint grammar improve T2I controllability and repair?**
   The project constructs prompts using a constraint grammar over object existence, cardinality, attributes, spatial relations, scene context, and constraint composition. This structure makes failures more measurable: when an image fails, the system can identify which constraint failed. In later stages, I will use these failed constraints to guide targeted prompt repair and evaluate whether constraint-aware repair improves success rate.

For Checkpoint 1, I focus on the first part of this system: building the evaluation pipeline, defining the prompt/constraint schema, and producing metrics that can later be used to compare initial generation accuracy and repair success.

## Prompt and constraint setup

The prompt grammar varies along three main dimensions:

- `prompt_family`
  - spatial layout
  - cardinality
  - attribute binding
  - combined constraints

- `scene_context_type`
  - simple scenes
  - natural scenes

- `composition`
  - single-constraint prompts
  - combined-constraint prompts

Each prompt is decomposed into atomic constraints. The current atomic constraint types are:

- `object_existence`
- `cardinality`
- `attribute`
- `spatial_relation`

The generated prompt and constraint files are:

- `data/prompts/prompts.csv`
- `data/prompts/constraints.csv`

The grammar design and configuration are stored in:

- `docs/grammar.md`
- `configs/grammar_config.yaml`

## Evaluation design

The evaluation is organized into three parts.

### 1. Constraint-level checker evaluation

**Goal:** evaluate whether the VLM checker can correctly judge each atomic visual constraint.

**Unit of analysis:** one `image × constraint` pair.

This level is mainly used for diagnosis and for identifying failed constraints that can later drive prompt repair.

**Metrics include:**

- VLM-human agreement
- confusion matrix
- failure precision / recall
- false-pass rate
- false-fail rate

### 2. Image-level checker evaluation

**Goal:** evaluate whether the VLM checker can judge whether a whole generated image satisfies the prompt.

Atomic constraints are aggregated into one image-level label using the following rule:

```text
all atomic constraints pass        → image-level pass
any atomic constraint fails        → image-level fail
no failures, but some ambiguous    → image-level ambiguous
```

**Metrics include:**

- image-level VLM-human agreement
- image-level confusion matrix
- image-level failure detection metrics

### 3. Image-level controllability evaluation

**Goal:** evaluate how controllable the T2I generator is under different prompt conditions.

This is the main evaluation for the project. It uses human-labeled image-level results to analyze how well the generator follows prompts under different conditions.

**Main breakdowns:**

- `prompt_family`
- `scene_context_type`
- `composition`
- `scene_context_type × composition`
- `prompt_family × scene_context_type × composition`

**Main metrics:**

- human-labeled image-level pass rate
- human-labeled image-level fail rate
- human-labeled image-level ambiguous rate

## What has been implemented

The following parts are implemented for Checkpoint 1:

- Grammar-based prompt and constraint generation
- Prompt-level metadata generation
- Atomic constraint metadata generation
- Mock image generation records
- Human label file format
- VLM checker label file format
- VLM response parsing utilities
- Constraint-level checker metrics
- Image-level label aggregation
- Image-level checker metrics
- Image-level controllability metrics by prompt dimensions
- Unit tests for prompt generation, parsing, and metric logic

The main scripts are:

- `scripts/generate_prompts.py`
- `scripts/create_human_label_template.py`
- `scripts/run_vlm_checker.py`
- `scripts/analyze_checker.py`
- `scripts/utils.py`

The T2I and VLM API calls are currently represented by stubs. This keeps the evaluation pipeline testable before API-specific details are added.

## Mock baseline used for this checkpoint

For this checkpoint, I use mock schematic images and mock labels to verify that the evaluation pipeline runs end-to-end.

The mock data includes:

- `data/generations/generations.csv`
- `data/images/mock_model/`
- `data/labels/human_labels.csv`
- `data/labels/vlm_labels.csv`

These mock images and labels are **not final T2I results**. They are used as a trivial baseline to ensure that the evaluation code can correctly process image records, human labels, VLM labels, constraint-level metrics, image-level aggregation, and controllability breakdowns.

This matches the checkpoint goal of validating the evaluation pipeline before running larger experiments on real generated images.

## Current results

The checkpoint pipeline has been run on the mock data, and the result CSV files are saved under:

- `results/checkpoint1/summary_metrics.csv`

### Constraint-level checker analysis

- `results/checkpoint1/constraint_checker_merged_labels.csv`
- `results/checkpoint1/constraint_checker_metrics_by_type.csv`
- `results/checkpoint1/constraint_checker_confusion_matrix.csv`
- `results/checkpoint1/constraint_checker_failure_detection_metrics.csv`

### Image-level checker analysis

- `results/checkpoint1/image_checker_labels.csv`
- `results/checkpoint1/image_checker_label_distribution.csv`
- `results/checkpoint1/image_checker_confusion_matrix.csv`
- `results/checkpoint1/image_checker_failure_detection_metrics.csv`

### Image-level controllability analysis

- `results/checkpoint1/image_controllability_by_dimension.csv`
- `results/checkpoint1/image_controllability_by_scene_and_composition.csv`
- `results/checkpoint1/image_controllability_by_family_scene_composition.csv`

The tests have also been run successfully:

```bash
pytest tests
```

## How to reproduce the checkpoint pipeline

From the repository root, run:

```bash
python scripts/run_checkpoint1.py --all
```

This script runs the checkpoint pipeline using the current mock data and writes the evaluation outputs to:

```
results/checkpoint1/
```

The individual steps are also available as separate scripts:

```bash
python scripts/generate_prompts.py
python scripts/create_human_label_template.py
python scripts/analyze_checker.py
```

If using the mock labels directly, `run_vlm_checker.py` does not need to be called. After real VLM API support is added, the intended command will be:

```bash
python scripts/run_vlm_checker.py
```

## Next steps

After Checkpoint 1, I plan to:

1. Replace the mock images and labels with real T2I-generated images and real VLM checker outputs.

2. Run the prompt suite on one or more basic image generators, with particular attention to spatial-relation prompts because spatial judgments are a key risk for VLM-based checking.

3. Expand the real evaluation across the planned constraint families and prompt settings, including counting, attribute binding, spatial relations, simple vs. natural scenes, and single vs. combined constraints.

4. Use failed constraints from the evaluation pipeline to drive targeted prompt repair for Checkpoint 2.
