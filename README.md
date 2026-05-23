# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project Proposal**

**Student:** Yiling Huang 

**SUNET ID:** yilhuang

## Summary

This project builds a grammar-driven diagnose-and-repair pipeline for prompt adherence in text-to-image generation, focusing on structured visual constraints such as object cardinality, attribute binding, and spatial layout. The core system will synthesize controlled prompts across simple and natural scene contexts, including both single-constraint and combined-constraint prompts, then evaluate generated images using human labels and a VLM-based constraint checker. By the end of the project, I will report constraint-level failure rates, VLM-human agreement, and a small-scale before/after analysis of whether constraint-aware prompt rewrites repair failed requirements without introducing regressions.

## Checkpoint 1

Checkpoint 1 focuses on concretizing the evaluation pipeline. The current version includes grammar-based prompt/constraint generation, mock image generations, human/VLM label schemas, constraint-level checker metrics, image-level checker metrics, and image-level controllability metrics.

The Checkpoint 1 data is archived under:

```text
data/checkpoint1/
```

See [`docs/checkpoint1/checkpoint1_README.md`](docs/checkpoint1/checkpoint1_README.md) for the full checkpoint report.

## Current Status: Checkpoint 2

Checkpoint 2 moves the project from mock data to a small real-data pilot. The current submission includes:

- real T2I generations using the OpenAI image API
- human labels on a selected subset of generated images
- VLM checker outputs on the same labeled subset
- constraint-level and image-level evaluation results
- a repair pipeline that converts failed constraints into structured repair prompts
- a small repair pilot with before/after human labels

See [`docs/checkpoint2/checkpoint2_README.md`](docs/checkpoint2/checkpoint2_README.md) for the Checkpoint 2 report.

## Evaluation Overview

The evaluation has three levels.

### 1. Constraint-level checker evaluation

Each generated image is decomposed into atomic visual constraints, such as:

- `object_existence`
- `cardinality`
- `attribute`
- `spatial_relation`

The unit of analysis is one `image × constraint` pair. This level evaluates whether the VLM checker agrees with human labels on each atomic requirement. It is mainly used for diagnosis and for identifying failed constraints that can later drive repair.

### 2. Image-level checker evaluation

Atomic constraint labels are aggregated into one image-level judgment:

```text
all atomic constraints pass        → image-level pass
any atomic constraint fails        → image-level fail
no failures, but some ambiguous    → image-level ambiguous
```

This level evaluates whether the VLM checker can correctly judge whether a whole generated image satisfies the prompt.

### 3. Image-level controllability evaluation

This is the main project-level evaluation. It measures how well the T2I generator follows prompts under different prompt conditions, including:

- `prompt_family`
- `scene_context_type`
- `composition`
- `scene_context_type × composition`
- `prompt_family × scene_context_type × composition`

The main metrics are human-labeled image-level pass, fail, and ambiguous rates.

For Checkpoint 2, the quantitative pilot is intentionally small, so these numbers are treated as preliminary rather than final conclusions.

### 4. Repair evaluation

Repair is triggered by failed atomic constraints, but organized as image-level prompt reconstruction. The repair analysis measures:

- `target-constraint fixed rate`
- `image-level fixed rate`
- `regression rate`
- `repair success by constraint type`

## Project Documents

- [Project proposal](docs/proposal/proposal.md)
- [Grammar design](docs/grammar.md)
- [Checkpoint 1 report](docs/checkpoint1/checkpoint1_README.md)
- [Checkpoint 2 report](docs/checkpoint2/checkpoint2_README.md)

## Repository Structure

```text
configs/        machine-readable grammar/config files
docs/           human-readable project documents
scripts/        pipeline scripts
data/           prompts, constraints, mock images, labels
results/        generated evaluation results
tests/          unit tests
```

## Data Organization

Checkpoint 1 mock/easy data is archived in:

```text
data/checkpoint1/
```

The current active project data uses the default data folders:

```text
data/prompts/
data/generations/
data/images/
data/labels/
data/repaired/
```

Checkpoint 2 selected prompt subsets and labels use checkpoint-specific filenames, for example:

```text
data/prompts/prompts_checkpoint2.csv
data/prompts/constraints_checkpoint2.csv
data/generations/generations_checkpoint2.csv
data/labels/human_labels_checkpoint2.csv
data/labels/vlm_labels_checkpoint2.csv
```

Generated image files are stored under:

```text
data/images/openai_checkpoint2/
```

Results are organized by checkpoint:

```text
results/checkpoint1/
results/checkpoint2/
```

## How to Run

### Checkpoint 1

From the repository root, run:

```bash
python scripts/run_checkpoint1.py
```

The main outputs are written to:

```text
results/checkpoint1/
```

### Checkpoint 2

Checkpoint 2 uses staged commands because some steps require API calls or manual human labeling.

From the `scripts/` directory:

```bash
python run_checkpoint2.py --select-prompts
python run_checkpoint2.py --generate-images
python run_checkpoint2.py --create-human-template
python run_checkpoint2.py --run-vlm
python run_checkpoint2.py --analyze
python run_checkpoint2.py --generate-repairs
python run_checkpoint2.py --generate-repair-images
python run_checkpoint2.py --create-repair-label-template
python run_checkpoint2.py --analyze-repair
```

The main outputs are written to:

```text
results/checkpoint2/
```

## Tests

From the repository, run:

```bash
pytest tests
```

The tests cover prompt generation, VLM response parsing, evaluation metrics, and repair logic.

## Notes

The current Checkpoint 2 results are a small real-data pilot rather than the final experiment. Human labeling revealed that cardinality and attribute/material constraints already produce meaningful pass/fail/ambiguous cases, while the current spatial grammar still needs revision. In the next iteration, the spatial grammar will be updated to use more physically meaningful relations before rerunning the broader generation and repair experiment.
