# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project Proposal**

**Student:** Yiling Huang 

**SUNET ID:** yilhuang



## Summary

This project builds a grammar-driven diagnose-and-repair pipeline for prompt adherence in text-to-image generation, focusing on structured visual constraints such as object cardinality, attribute binding, and spatial layout. The core system will synthesize controlled prompts across simple and natural scene contexts, including both single-constraint and combined-constraint prompts, then evaluate generated images using human labels and a VLM-based constraint checker. By the end of the project, I will report constraint-level failure rates, VLM-human agreement, and a small-scale before/after analysis of whether constraint-aware prompt rewrites repair failed requirements without introducing regressions.

## Test
Use `pytest tests` at the root directary to run all the testing functions.

## Mock Image Generation

The images in `/data/images/mock_model` and relative metadata in `/data/generations/generations.csv` are **not** real T2I outputs. They are intended only to test the following functions for checkpoint 1.

- `create_human_label_template.py`
- `run_vlm_checker.py`
- `analyze_checker.py`