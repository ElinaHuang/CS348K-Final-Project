# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

CS348K Project Final Report

Yiling Huang

yilhuang@stanford.edu

## Summary

Text-to-image (T2I) models often generate images that look plausible overall while still failing specific parts of a prompt, such as object identity, exact count, object attributes, or spatial relations. This project studies whether decomposing T2I prompts into atomic visual constraints can make these partial prompt-following failures measurable, repairable, and scalable with VLM-assisted checking.

I build an end-to-end workflow that starts from a controlled prompt grammar, generates images with two T2I models, decomposes each prompt into atomic constraints, labels each image-constraint pair with human and VLM checkers, generates targeted repair prompts, and evaluates whether repair improves human-labeled correctness. The main result is that constraint-aware repair improves both image-level and constraint-level pass rates. Human-triggered repair improves image pass rate by 51.9%, while VLM-triggered repair improves image pass rate by 28.0% under human evaluation. These results support the main claim that constraint-level structure is a useful systems abstraction for making partial T2I prompt-following failures measurable and actionable.

## 1. Background and Setup

### 1.1 Partial Prompt-Following Failures

Text-to-image (T2I) models are increasingly good at producing images that look visually plausible at a glance. However, a visually plausible image does not necessarily satisfy all of the requirements in the input prompt. In many cases, the model does not completely ignore the prompt; instead, it satisfies some parts of the prompt while failing others.

For example, consider a prompt such as:

> A realistic image of exactly 4 dice inside a cup on a shelf full of books and decorations.

<img src="./assets/figure1_dice_cup_count_failure.png" width="50%">

**Figure 1**. Example of a partial prompt-following failure. The image satisfies the object and spatial requirements, since dice and a cup are visible and the dice are inside the cup. However, it violates the cardinality requirement because the prompt asks for exactly 4 dice while the image contains 5 dice.

This type of partial prompt-following failure is difficult to capture with only image-level evaluation. If I label the entire image as failed, I lose information about which parts of the prompt were actually satisfied. If I only judge whether the image looks realistic, I may miss the prompt-following error entirely. Therefore, this project focuses on evaluating and improving T2I outputs at the level of **individual visual requirements**, not only at the level of the whole image.

### 1.2 Project Goal and Questions

The main goal of this project is to make partial T2I prompt-following failures measurable and repairable. More specifically, I want a system that can identify which parts of a prompt were satisfied, which parts failed, and which parts are uncertain, so that the next generation attempt can be guided toward an image that better satisfies the original prompt.

The **main question** I ask is:

> How can we make partial T2I prompt-following failures measurable and repairable?

There is also a **practical scaling question**. Human judgment is useful for evaluating whether an image satisfies a prompt, but if the goal is to generate and improve many images, manually checking every requirement in every image becomes expensive. Therefore, I also ask:

> Can a VLM checker scale this process by identifying repair targets automatically?

In this framing, the VLM is not meant to replace human judgment. Instead, the goal is to test whether a VLM can provide useful automatic signals for improving generation at scale, while human judgment is still used to evaluate whether the final images actually satisfy the prompt.

### 1.3 Inputs, Outputs, and Success Criteria

The basic task setting is a text-to-image generation problem. The **input** is a text prompt that contains multiple visual requirements, and the **output** is an image that should satisfy as many of those requirements as possible. These requirements may involve object identity, exact count, object attributes, or spatial relations.

A successful system should not only generate an image, but also make the image generation process easier to evaluate and improve. I define success along three dimensions:

- First, the system should **make failures measurable.** Instead of only asking whether the whole image passes or fails, it should expose which specific visual requirements were satisfied, failed, or uncertain.
- Second, the system should **make failures repairable**. If one part of the prompt fails, the system should be able to turn that failure into a targeted repair instruction, rather than rewriting the entire prompt blindly.
- Third, the system should **be reasonably stable and practical**. The same representation should support generation, checking, repair, and analysis. It should also work across different prompt families and T2I models, rather than depending on a single hand-written example.

I do not frame this project as a benchmark competition against an existing repair algorithm. Instead, I evaluate whether a structured representation of prompt requirements is useful enough to support an end-to-end generate-check-repair workflow.

### 1.4 Technical Crux and Conceptual Motivation

The technical crux of this project is deciding how to represent prompt requirements so that they are measurable, repairable, and stable across the full workflow. A requirement that is too vague is difficult for a human to label, difficult for a VLM to check, and difficult to turn into a repair instruction. A requirement that is too rigid may be easy to check, but may not correspond well to natural T2I prompts. The challenge is to define visual requirements at a level that is specific enough to evaluate, but still natural enough to generate useful images.

This project is motivated by a broader theme in generative systems: output quality alone is not enough when the output is supposed to satisfy structured user requirements. SceneEval [1] makes a related observation for text-conditioned 3D indoor scene synthesis, where semantic coherence should be evaluated against explicit user requirements and scene-level expectations rather than only visual realism. My project explores a similar constraint-level idea for 2D text-to-image generation, but extends it into a generate-check-repair workflow.

The project is also influenced by course readings that emphasize structured representations for generation and verification. Grammar-based design systems, such as those discussed in Design for Descent [2], motivated the idea of using a structured design space rather than only free-form natural language prompts. Verification-oriented work such as MoVer [3] also motivated the separation between generation and checking: the system should not only produce an output, but also expose properties that can be checked.

Given this problem setting, I designed a constraint-level grammar as the central abstraction for the project. The grammar is introduced in the next section. It defines controlled prompt families, decomposes prompts into atomic visual constraints, supports human and VLM checking, and converts failed constraints into targeted repair instructions.

## 2. Approach



## 3. Evaluation and Results



## 4. Discussion and Takeaways



## 5. Team Responsibilities



## 6. References

1. Hou In Ivan Tam, Hou In Derek Pun, Austin T. Wang, Angel X. Chang, and Manolis Savva. **SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis.** arXiv:2503.14756, 2025. https://arxiv.org/abs/2503.14756
2. Milin Kodnongbua, Zihan Jack Zhang, Nicholas Sharp, and Adriana Schulz. **Design for Descent: What Makes a Shape Grammar Easy to Optimize?** SIGGRAPH Asia Conference Papers, 2025. https://www.computationaldesign.group/publications/design-for-descent
3. Jiaju Ma and Maneesh Agrawala. **MoVer: Motion Verification for Motion Graphics Animations.** arXiv:2502.13372, 2025. https://arxiv.org/abs/2502.13372
