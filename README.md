# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project Proposal**

**Student:** Yiling Huang 

**SUNET ID:** yilhuang



## Summary

This project builds a grammar-driven diagnose-and-repair pipeline for prompt adherence in text-to-image generation, focusing on structured visual constraints such as object cardinality, attribute binding, and spatial layout. The core system will synthesize controlled prompts across simple and natural scene contexts, including both single-constraint and combined-constraint prompts, then evaluate generated images using human labels and a VLM-based constraint checker. By the end of the project, I will report constraint-level failure rates, VLM-human agreement, and a small-scale before/after analysis of whether constraint-aware prompt rewrites repair failed requirements without introducing regressions.

## Evaluation

The evaluation has three parts:

### 1. Constraint-level checker evaluation

**Goal:** evaluate whether the VLM checker can correctly judge each atomic visual constraint.

**Unit of analysis:** one `image × constraint` pair.

Atomic constraint types include:

- `object_existence`
- `cardinality`
- `attribute`
- `spatial_relation`

**Metrics:**

- VLM-human agreement
- confusion matrix
- failure precision / recall
- false-pass rate
- false-fail rate

This level is mainly used for diagnosis and for identifying failed constraints that can later drive prompt repair.

---

### 2. Image-level checker evaluation

**Goal:** evaluate whether the VLM checker can judge whether a whole generated image satisfies the prompt.

**Aggregation rule:**

```text
all atomic constraints pass        → image-level pass
any atomic constraint fails        → image-level fail
no failures, but some ambiguous    → image-level ambiguous
```

**Metrics:**

- image-level VLM-human agreement
- image-level confusion matrix
- image-level failure detection metrics

---

### 3. Image-level controllability evaluation

**Goal:** evaluate how controllable the T2I generator is under different prompt conditions.

**Unit of analysis:** one generated image.

The main breakdowns are:

- `prompt_family`
- `scene_context_type`
- `composition`
- `scene_context_type × composition`
- `prompt_family × scene_context_type × composition`

**Main metrics:**

- human-labeled image-level pass rate
- human-labeled image-level fail rate
- human-labeled image-level ambiguous rate

This is the main evaluation for understanding how prompt family, scene complexity, and constraint composition affect text-to-image prompt adherence.

## Test
Run `pytest tests` command at the root directary for testing.

## Mock Image and Labels for Checkpoint 1 Pipeline Testing

The images in `/data/images/mock_model` and relative metadata in `/data/generations/generations.csv` are **not** real T2I outputs. The labels in `/data/labels/human_labels.csv` are **not** human-labeled and the labels in `/data/labels/vlm_labels.csv` are **not** real VLM outputs. They are intended only to test the following functions for checkpoint 1.

- `create_human_label_template.py`
- `run_vlm_checker.py`
- `analyze_checker.py`