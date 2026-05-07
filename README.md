# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project Proposal**

**Student:** Yiling Huang 

**SUNET ID:** yilhuang



## Summary

This project builds a grammar-driven diagnose-and-repair pipeline for prompt adherence in text-to-image generation, focusing on structured visual constraints such as object cardinality, attribute binding, and spatial layout. The core system will synthesize controlled prompts across simple and natural scene contexts, including both single-constraint and combined-constraint prompts, then evaluate generated images using human labels and a VLM-based constraint checker. By the end of the project, I will report constraint-level failure rates, VLM-human agreement, and a small-scale before/after analysis of whether constraint-aware prompt rewrites repair failed requirements without introducing regressions.

## Test
Use `pytest tests` at the root directary to run all the testing functions.