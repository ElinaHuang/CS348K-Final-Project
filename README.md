# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project Proposal**

**Student:** Yiling Huang 

**SUNET ID:** yilhuang

## Summary

This project builds a grammar-driven diagnose-and-repair pipeline for prompt adherence in text-to-image generation, focusing on structured visual constraints such as object cardinality, attribute binding, and spatial layout. The core system will synthesize controlled prompts across simple and natural scene contexts, including both single-constraint and combined-constraint prompts, then evaluate generated images using human labels and a VLM-based constraint checker. By the end of the project, I will report constraint-level failure rates, VLM-human agreement, and a small-scale before/after analysis of whether constraint-aware prompt rewrites repair failed requirements without introducing regressions.

## Current Status: Checkpoint 1

Checkpoint 1 focuses on concretizing the evaluation pipeline. The current version includes grammar-based prompt/constraint generation, mock image generations, human/VLM label schemas, constraint-level checker metrics, image-level checker metrics, and image-level controllability metrics.

See [`docs/checkpoint1/checkpoint1_README.md`](docs/checkpoint1/checkpoint1_README.md) for the full checkpoint report.

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

## Project Documents

- [Project proposal](docs/proposal/proposal.md)
- [Grammar design](docs/grammar.md)
- [Checkpoint 1 report](docs/checkpoint1/checkpoint1_README.md)

## Repository Structure

```text
configs/        machine-readable grammar/config files
docs/           human-readable project documents
scripts/        pipeline scripts
data/           prompts, constraints, mock images, labels
results/        generated evaluation results
tests/          unit tests
```

## How to Run

From the repository root, run:

```bash
python scripts/run_checkpoint1.py
```

This runs the current checkpoint pipeline and writes outputs to:

```text
results/checkpoint1/
```

## Tests

From the repository, run:

```bash
pytest tests
```

All tests currently pass.

## Mock Data Notice

The current Checkpoint 1 data is a mock baseline:

- images in `data/images/mock_model/` and relative metadata in `/data/generations/generations.csv` are not real T2I outputs
- `data/labels/human_labels.csv` is not manually labeled final data
- `data/labels/vlm_labels.csv` is not real VLM API output

These files are included only to test the checkpoint evaluation pipeline. The next step is to replace them with real T2I-generated images and real VLM checker outputs.
