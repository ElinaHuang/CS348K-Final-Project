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

<img src="./assets/figure1_dice_cup_count_failure.png" width="45%">

**Figure 1.** Example of a partial prompt-following failure. The image satisfies the object and spatial requirements, since dice and a cup are visible and the dice are inside the cup. However, it violates the cardinality requirement because the prompt asks for exactly 4 dice while the image contains 5 dice.

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

### 2.1 A Shared Grammar for Generation, Checking, Repair, and Evaluation

The key abstraction in this project is one shared grammar. The grammar does not only generate the original text prompt. It also defines the atomic visual constraints implied by that prompt, and these constraints are then reused for labeling, repair, and evaluation.

This design is important because partial prompt-following failures are only useful if they can be localized. For example, if an image generated from the prompt “exactly 4 dice inside a cup” contains dice and a cup, but generates the wrong number of dice, the system should not only say that the whole image failed. It should be able to identify that the object identity and spatial relation were satisfied, while the cardinality requirement failed.

<img src="./assets/figure2_approach_pipeline.png" width="85%">

**Figure 2.** Example of the shared grammar.

The grammar is therefore used as a shared representation across the project:

1. It generates controlled prompts.
2. It decomposes each prompt into atomic constraints.
3. It provides the units that humans and VLMs label.
4. It converts failed or ambiguous constraints into targeted repair instructions.
5. It defines the metrics used to analyze initial generation and repair results.

In other words, the grammar turns a free-form prompt-following problem into a structured generate-check-repair-evaluate problem.

### 2.2 Grammar Design for Controlled Prompt Generation

The grammar is designed to generate prompts with controlled visual requirements. It is not intended to cover every possible user prompt; instead, it creates a manageable testbed where prompt requirements can be systematically generated, labeled, repaired, and analyzed. This controlled setting is important because the project goal is not to maximize prompt diversity, but to study whether partial prompt-following failures can be localized and improved in a stable way.

A prompt is created by filling typed slots such as object category, number, attribute, spatial relation, and scene context. A representative template is:

```text
{prefix} exactly {number} {object_1_plural} {relation_text} a {object_2} {scene_context}.
```

For example, this template can be instantiated as:

```text
A realistic image of exactly 4 dice inside a cup on a shelf full of books and decorations.
```

This example makes the main grammar components easier to see: the template includes a number, a target object, a spatial relation, a reference object, and a scene context. Other prompt families use similar typed slots to generate attribute-only, cardinality-only, spatial-only, and combined prompts.

The main grammar components are summarized below.

| Grammar component | Examples in this project                                     | Purpose                                                      |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Prompt family     | single spatial, single attribute, single cardinality, spatial+attribute, spatial+cardinality, attribute+cardinality | Controls which types of visual requirements appear in a prompt |
| Scene context     | simple, natural                                              | Tests whether clutter and context complexity affect prompt-following |
| Object groups     | small items, containers, surfaces                            | Keeps object choices typed and compatible with relations     |
| Attribute groups  | pattern/texture, material/finish                             | Avoids unnatural object-attribute combinations               |
| Spatial relations | left/right, on top of/under, inside/outside                  | Uses relations that are natural and labelable                |
| Numbers           | 3, 4, 5, 6                                                   | Tests exact cardinality control                              |

The full grammar configuration is stored in [grammar_configs.yaml](../../configs/grammar_config.yaml), with additional design notes in [grammar.md](../grammar.md). In the writeup, I focus on the design choices and the resulting evaluation rather than listing every grammar template.

### 2.3 From Prompt Instances to Atomic Constraints

After a prompt is generated, the same grammar instance is used to derive atomic constraints. These constraints are the basic units for checking, repair, and evaluation.

I use four representative constraint types in this project:

| Constraint type  | Meaning                                                      | Example                                           |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| object_identity  | Whether the requested object category is clearly present and recognizable | Does the image contain clearly identifiable dice? |
| cardinality      | Whether the requested number of target objects is visible and countable | Does the image contain exactly 4 dice?            |
| attribute        | Whether the requested object has the required pattern, texture, material, or finish | Does the image show a glossy white cup?           |
| spatial_relation | Whether two requested objects satisfy the required relation  | Are the dice inside the cup?                      |

For the example prompt:

```text
A realistic image of exactly 4 dice inside a cup on a shelf full of books and decorations.
```

the grammar produces the following atomic constraints:

| Prompt element  | Atomic constraint type | Constraint                                              |
| --------------- | ---------------------- | ------------------------------------------------------- |
| dice            | object_identity        | The image should contain clearly identifiable dice      |
| cup             | object_identity        | The image should contain a clearly identifiable cup     |
| exactly 4 dice  | cardinality            | The image should contain exactly 4 clearly visible dice |
| dice inside cup | spatial_relation       | The dice should be inside the cup                       |

Attribute constraints are generated in the same way when the prompt includes material, finish, pattern, or texture requirements. For example, a prompt containing “a glossy white cup” produces an attribute constraint checking whether the cup has the requested glossy white material or surface finish.

This decomposition is what makes the problem measurable. Instead of assigning one label to the entire image, the system can ask which specific requirements were satisfied and which were not.

### 2.4 Label Design and Human/VLM Checking

Each atomic constraint receives one of three labels:

| Label     | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| pass      | The constraint is reasonably satisfied                   |
| fail      | The constraint is clearly violated                       |
| ambiguous | The constraint is uncertain or not confidently judgeable |

The ambiguous label is important for generated images. In this project, ambiguity does not only mean that the image is blurry or occluded. It also includes cases where the model generates an object-like shape that looks plausible, but whose identity cannot be confidently determined.

For example, the prompt below asks for thumbtacks:

```text
A realistic image of exactly 4 thumbtacks outside a food container on a picnic blanket with many small objects.
```

<img src="./assets/figure3_ambiguous_thumbtack.png" width="45%">

**Figure 3.** Example of an ambiguous object-identity case. The image contains four red object-like shapes in the expected location, but it is difficult to confidently identify them as thumbtacks. I label this kind of case as ambiguous rather than forcing a pass/fail decision.

Human labels are used as the ground-truth evaluation signal. A VLM checker is used to test whether the checking process can be automated at scale. Both humans and the VLM judge the same image-constraint pairs, which makes it possible to compare their agreement. However, VLM labels are not treated as final truth; they are used as automatic signals that can guide the next generation attempt.

### 2.5 Constraint-Aware Repair Prompt Generation

The repair stage uses the same constraint representation. When a constraint is labeled as fail or ambiguous, the system converts that constraint into a targeted repair instruction.

The repaired prompt is constructed by keeping the original prompt and appending repair instructions for the failed or ambiguous constraints. For example, if the cardinality constraint fails for the dice prompt, the repair prompt keeps the original description and adds an instruction such as:

```text
Make sure there are exactly 4 dice; do not generate more or fewer, and keep them separated and countable.
```

This design avoids rewriting the entire prompt from scratch. Instead, the grammar identifies which part of the prompt failed and produces an instruction that directly targets that requirement.

The repair instruction is generated according to the failed or ambiguous constraint type:

| Constraint type  | Repair focus                                                 | Example repair instruction                                   |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| object_identity  | Make the requested object recognizable and avoid replacing it with a different category | Make sure the image clearly contains identifiable dice.      |
| cardinality      | Enforce the exact count and keep target objects separated and countable | Make sure there are exactly 4 dice; do not generate more or fewer, and keep them separated and countable. |
| attribute        | Apply the requested pattern, texture, material, or finish to the correct object | Make sure the cup has the glossy white material or surface finish. |
| spatial_relation | Make the requested relation visually clear                   | Make sure the dice are clearly inside the cup.               |

For constraints labeled as `ambiguous`, the repair instruction is phrased to make the requirement clear and easy to judge, rather than only asking the model to correct a clear failure. This is useful because ambiguous cases are not always clearly wrong; sometimes the generated image simply does not provide enough visual evidence.

Overall, the repair approach follows the same grammar-level principle as the rest of the system: the unit of repair is not the entire image, but the specific visual requirement that failed or was uncertain.

## 3. Evaluation and Results



## 4. Discussion and Takeaways



## 5. Team Responsibilities



## 6. References

1. Hou In Ivan Tam, Hou In Derek Pun, Austin T. Wang, Angel X. Chang, and Manolis Savva. **SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis.** arXiv:2503.14756, 2025. https://arxiv.org/abs/2503.14756
2. Milin Kodnongbua, Zihan Jack Zhang, Nicholas Sharp, and Adriana Schulz. **Design for Descent: What Makes a Shape Grammar Easy to Optimize?** SIGGRAPH Asia Conference Papers, 2025. https://www.computationaldesign.group/publications/design-for-descent
3. Jiaju Ma and Maneesh Agrawala. **MoVer: Motion Verification for Motion Graphics Animations.** arXiv:2502.13372, 2025. https://arxiv.org/abs/2502.13372
