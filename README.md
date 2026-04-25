# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project Proposal**

**Student:** Yiling Huang 

**SUNET ID:** yilhuang



## Summary

This project builds a grammar-driven diagnose-and-repair pipeline for prompt adherence in text-to-image generation, focusing on structured visual constraints such as object cardinality, attribute binding, and spatial layout. The core system will synthesize controlled prompts across simple and natural scene contexts, including both single-constraint and combined-constraint prompts, then evaluate generated images using human labels and a VLM-based constraint checker. By the end of the project, I will report constraint-level failure rates, VLM-human agreement, and a small-scale before/after analysis of whether constraint-aware prompt rewrites repair failed requirements without introducing regressions.



## Problem Setup and Motivation

Text-to-image systems can often generate images that are visually plausible overall but fail specific prompt requirements. A single generated image can be inspected by a human, but batch evaluation becomes slow and unstructured when a user wants to generate many candidates, compare prompt variants, or build a larger automated image generation workflow. In those settings, the useful question is not only whether an image “looks good,” but whether it satisfies the specific visual requirements encoded in the prompt.

This project asks whether structured constraint-level feedback can support scalable evaluation and prompt debugging for text-to-image generation. Given a controlled prompt and a generated image, the system identifies which visual requirements are satisfied, violated, or ambiguous. It then uses the type of failed constraint to guide a targeted prompt rewrite and evaluates whether the repaired prompt improves the failed requirement.

The project is not intended to characterize all capabilities of text-to-image models. Instead, it focuses on a controlled subset of structured visual constraints that are common in user prompts, human-checkable, and systematically generatable from templates. This makes the evaluation tractable while still capturing common prompt-following failures in generated images.



## System Pipeline

Rather than a single-step input-output task, the project is organized as a diagnose-and-repair pipeline:

```text
Constraint Grammar
        ↓
Controlled Prompts + Expected Constraint Metadata
        ↓
Text-to-Image Generation
        ↓
Generated Images
        ↓
Constraint-Level Checking
        ↓
Human Labels + VLM Checker Outputs
        ↓
Failure Analysis
        ↓
Constraint-Aware Prompt Repair on Failed Examples
        ↓
Regenerated Images
        ↓
Before / After Repair Evaluation
```

The key design idea is that the same constraint grammar supports both evaluation and repair. For example, a counting constraint defines how the initial prompt states cardinality, how the generated image is checked, and how the prompt is rewritten if the count fails.



## Inputs and Outputs

### External Inputs

The system requires the following external inputs:

1. **Constraint grammar design**: A small set of structured visual constraint families, including object cardinality, attribute binding, and spatial layout.
2. **Existing text-to-image model**: An existing text-to-image model or API used to generate images from controlled prompts. The project does not require training a new image generation model.
3. **Existing VLM model**: A vision-language model used as an automatic constraint checker. Its outputs will be compared against human labels rather than treated as ground truth.
4. **Human annotations**: Human pass / fail / ambiguous labels for whether generated images satisfy the expected constraints.

### Intermediate Data Products

The pipeline produces several intermediate artifacts:

1. Controlled prompts generated from the grammar.
2. Expected constraint metadata for each prompt.
3. Generated images from the text-to-image model.
4. Human labels for each image-constraint pair.
5. VLM checker judgments for each image-constraint pair.
6. Failed-constraint records used to drive prompt repair.
7. Rewritten prompts and regenerated images for the repair subset.

### Final Outputs

The final project outputs are:

1. A small structured prompt-image-constraint dataset.
2. Constraint-level failure statistics by constraint type.
3. VLM-human agreement analysis for automatic constraint checking.
4. Before / after repair results on failed examples.
5. Qualitative examples showing successful repairs, failed repairs, and repair-induced regressions.



## Design Goals and Constraints

### Goals

The project has three main goals:

1. **Controlled evaluation** 

   Build a small prompt generation system that produces prompts with explicit, known visual constraints.

2. **Constraint-level diagnosis** 

   Evaluate generated images not with a single holistic score, but with pass / fail / ambiguous judgments for individual constraints.

3. **Targeted prompt repair** 

   Use the failed constraint type to select a corresponding rewrite strategy, then test whether the repaired prompt improves the pass rate of the failed requirement.

### Constraints

The project is designed for one student, so it avoids large-scale model training or heavy infrastructure. The main constraints are:

1. **Human labeling time**

   The dataset must be small enough to label manually but structured enough to support meaningful analysis.

2. **Evaluation reliability** 

   Human labels will be treated as the primary ground truth. VLM-based checking will be evaluated against human labels rather than assumed correct.

3. **Scope control** 

   The project focuses on a representative subset of structured visual constraints rather than attempting to cover all possible text-to-image prompts.

4. **Automation** 

   Prompt generation and prompt repair should be systematic rather than fully manual. The grammar should make it possible to expand the evaluation set if time permits.

5. **Interpretability** 

   The diagnosis and repair process should be explainable. If a counting constraint fails, the repair should be visibly related to counting; if a spatial relation fails, the repair should target spatial layout.



## Constraint Grammar

The project uses a small template-based constraint grammar. Each constraint family defines:

1. how initial prompts are generated;
2. what metadata is stored as the expected visual constraint;
3. how generated images are checked;
4. how failed prompts are repaired.

The core constraint families are object cardinality, attribute binding, and spatial layout. Their simplicity is intentional: they are controlled visual primitives that can be composed into richer prompts, checked by humans, and repaired with constraint-type-specific rewrite rules.

| Constraint Type      | Initial Prompt Pattern                                       | Example Prompt                                               | Check Criterion                                              | Repair Pattern                                               |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Object cardinality   | `A {style} photo of exactly {number} {object_plural} on/in {background}.` | `A clean photo of exactly three cups on a table.`            | The image contains exactly `{number}` visible instances of `{object}`. | Repeat the exact count, add “total,” and add a negative constraint such as “no extra {object_plural} are visible.” |
| Attribute binding    | `A {style} photo of a {attribute_1} {object_1} next to a {attribute_2} {object_2} on/in {background}.` | `A clean photo of a red cup next to a blue book on a table.` | Both objects appear, and each attribute is attached to the correct object. | Split object-attribute pairs into separate sentences: “The cup must be red. The book must be blue.” |
| Spatial layout       | `A {style} photo of a {object_1} {spatial_relation} a {object_2} on/in {background}.` | `A clean photo of a cup to the left of a book on a table.`   | Both objects appear, and `{object_1}` satisfies the specified spatial relation with respect to `{object_2}`. | Rewrite the relation as an explicit image layout: “The cup is on the left side of the image. The book is on the right side.” |
| Combined constraints | A prompt composed from multiple atomic constraint templates. | `A clean photo of exactly two red cups to the left of a blue book on a table.` | Each atomic constraint is checked separately, and full prompt success requires all relevant constraints to pass. | Rewrite the prompt as numbered atomic visual requirements.   |

### Grammar Variables

The grammar will use a restricted vocabulary to keep labeling manageable and reduce ambiguity.

- `{number}`: 2, 3, 4
- `{object}`: cup, book, ball, apple, box, toy car
- `{attribute}`: red, blue, green, yellow, black, white
- `{spatial_relation}`: to the left of, to the right of, above, below
- `{background}`: table, plain background, shelf, floor, kitchen counter, park, campus walkway
- `{style}`: clean, simple, front-facing, realistic

The core grammar will vary prompts along three dimensions:

1. **Constraint family:** object cardinality, attribute binding, and spatial layout.
2. **Scene context:** simple scenes versus natural scenes.
3. **Constraint composition:** single-constraint prompts versus combined-constraint prompts.

This supports the evaluation goal: measuring not only whether individual constraint families are followed, but also whether adherence changes when the same requirements are placed in richer scene contexts or combined with other constraints.



## Task List

### Task 1: Build the prompt and constraint generation grammar

I will implement a small template-based grammar that generates prompts with explicit expected constraints. The core grammar will cover object cardinality, attribute binding, and spatial layout.

The grammar will also include simple versus natural scene contexts and single versus combined constraints as part of the core scope. For example, the system should be able to generate prompts such as:

- a simple counting prompt;
- a counting prompt in a natural scene;
- an attribute-binding prompt;
- a spatial-layout prompt;
- a combined prompt with both attribute and spatial constraints.

A feasible initial target is:

> 3 constraint families × 2 scene contexts × 2 composition levels × 3–5 prompt instances × 1 image per prompt

This gives around 36–60 initial generated images. If time permits, I may generate a second image per prompt to measure variability.

### Task 2: Generate images from an existing text-to-image model

I will use an existing text-to-image model or API to generate images from the prompts. The project does not involve training a new image generation model.

For each prompt, I will store:

1. the original prompt;
2. the expected constraint metadata;
3. the generated image;
4. the scene context;
5. the composition level;
6. the constraint family or families involved.

### Task 3: Human-label constraint satisfaction

For each generated image, I will label whether each expected constraint is:

1. **Pass** — the constraint is clearly satisfied;
2. **Fail** — the constraint is clearly violated;
3. **Ambiguous** — the image is unclear or the constraint cannot be reliably judged.

Human labels will serve as the primary ground truth.

This labeling process is important because some constraints, especially spatial or combined constraints, can be ambiguous. The ambiguous category prevents the evaluation from forcing uncertain cases into binary labels.

### Task 4: Build a VLM-based constraint checker

I will use a vision-language model to automatically judge whether each generated image satisfies each constraint. The VLM checker will receive:

1. the generated image;
2. the constraint text;
3. a fixed instruction to output pass / fail / ambiguous;
4. a short explanation.

The checker output will be compared against human labels. The goal is not to assume that the VLM checker is correct, but to evaluate whether it is reliable enough to support batch evaluation and debugging.

### Task 5: Analyze failure patterns

I will compute failure statistics across the generated dataset.

The analysis will be broken down by:

1. constraint type;
2. scene context;
3. single versus combined constraints;
4. simple versus natural prompts.

Example questions include:

- Are counting constraints more fragile than attribute-binding constraints?
- Do spatial-layout constraints fail more often in natural scenes than in simple scenes?
- Do combined prompts have lower full-prompt success rates than single-constraint prompts?
- Does the VLM checker agree more with humans on attributes than on spatial relations?

### Task 6: Run a small-scale constraint-aware repair loop

For failed examples, I will apply repair rules based on the failed constraint type. This repair step is not a free-form prompt rewrite. It is guided by the same constraint grammar used for prompt generation.

Examples:

- Counting failures trigger cardinality-focused rewrites.
- Attribute-binding failures trigger explicit object-attribute binding rewrites.
- Spatial-layout failures trigger explicit layout rewrites.
- Combined failures may be rewritten as numbered atomic requirements.

For the core project, I will run this repair loop on a small subset of failed examples, aiming for 3–5 failed cases per core constraint family when available.

### Task 7: Evaluate before / after prompt repair

After generating images from the repaired prompts, I will compare constraint pass rates before and after repair.

The key metric is:

> Does the originally failed constraint pass after targeted repair?

I will also check whether repair introduces regressions in other constraints. For example, a repaired prompt may fix a spatial relation but cause the object count to fail. This is an important system tradeoff in prompt debugging.



## Expected Deliverables and Evaluation

### Deliverables

The final project will include:

1. A prompt and constraint generation grammar.
2. A generated image dataset with structured metadata.
3. Human labels for constraint satisfaction.
4. A VLM-based constraint checker.
5. Quantitative analysis of checker reliability.
6. Quantitative analysis of text-to-image failure patterns.
7. A small-scale constraint-aware prompt repair loop on failed examples.
8. Before / after evaluation of repaired prompts.
9. A final report and short demo showing the full debugging pipeline.

### Evaluation Plan

#### Evaluation 1: Constraint failure rate

I will measure how often each constraint type fails under human labels.

Example table:

| Constraint Type    | Simple Context | Natural Context | Single Constraint | Combined Constraint |
| ------------------ | -------------: | --------------: | ----------------: | ------------------: |
| Object cardinality |            TBD |             TBD |               TBD |                 TBD |
| Attribute binding  |            TBD |             TBD |               TBD |                 TBD |
| Spatial layout     |            TBD |             TBD |               TBD |                 TBD |

This answers:

> Which structured visual requirements are most fragile in generated images, and how do scene context and constraint composition affect failure rates?

#### Evaluation 2: VLM checker agreement with human labels

I will compare VLM judgments with human labels.

Metrics may include:

1. overall accuracy;
2. accuracy by constraint type;
3. precision and recall for detecting failures;
4. ambiguous rate;
5. confusion matrix.

Example table:

| Constraint Type    | VLM-Human Agreement | Failure Detection Precision | Failure Detection Recall |
| ------------------ | ------------------: | --------------------------: | -----------------------: |
| Object cardinality |                 TBD |                         TBD |                      TBD |
| Attribute binding  |                 TBD |                         TBD |                      TBD |
| Spatial layout     |                 TBD |                         TBD |                      TBD |

This answers:

> Can VLM-based checking approximate human judgment well enough to support batch prompt-following evaluation?

#### Evaluation 3: Prompt repair effectiveness

For failed examples selected for repair, I will compare the original and repaired prompts.

Metrics:

1. target constraint repair success rate;
2. full prompt success rate;
3. non-target constraint regression rate.

Example table:

| Repair Type      | Target Constraint Pass Rate Before | Target Constraint Pass Rate After | Regression in Other Constraints |
| ---------------- | ---------------------------------: | --------------------------------: | ------------------------------: |
| Counting repair  |                                TBD |                               TBD |                             TBD |
| Attribute repair |                                TBD |                               TBD |                             TBD |
| Spatial repair   |                                TBD |                               TBD |                             TBD |

This answers:

> Does constraint-type-aware prompt repair improve the specific failed visual requirement, and what tradeoffs does it introduce?

#### Evaluation 4: Qualitative failure analysis

The final report will include representative examples where:

1. the original generation failed a constraint;
2. the VLM checker correctly diagnosed the failure;
3. the VLM checker disagreed with the human label;
4. the repair successfully fixed the failed constraint;
5. the repair fixed one constraint but broke another;
6. the failure persisted after repair.

This qualitative analysis is important because aggregate statistics may hide why certain constraints are difficult.



## Definition of Success

The project will be successful if it demonstrates a working constraint-level debugging loop for text-to-image prompt adherence.

A strong outcome would show that:

1. the grammar can generate controlled prompts with explicit expected constraints;
2. constraint-level labels reveal systematic prompt-following failure patterns;
3. VLM checking is reliable for some constraint families but less reliable for others;
4. targeted prompt repair improves the pass rate of failed constraints on at least some examples;
5. the system exposes tradeoffs, such as repairing one constraint while introducing regressions elsewhere.

The project does not need to solve text-to-image generation or produce a complete benchmark of all visual capabilities. The goal is to build and evaluate a small but principled system for scalable, structured prompt-following evaluation and repair.



## Nice-to-Have Extensions

### Nice-to-have 1: Action and Interaction Constraints

If time permits, I will extend the constraint grammar beyond cardinality, attribute binding, and spatial layout to include action or interaction constraints, such as one person handing an object to another person, a person throwing a ball to another person, or an object being placed inside another object. These constraints are more visually ambiguous and harder to label reliably, so they are not part of the core scope.

### Nice-to-have 2: Larger-Scale Repair Evaluation and Repair Ablations

The core project will include a small-scale constraint-aware repair loop on a subset of failed examples. If time permits, I will expand the repair evaluation to more failed cases and compare different repair strategies, such as generic LLM rewriting versus constraint-type-aware rewriting.



## Biggest Risks

### Risk 1: Human labeling cost

Manual constraint-level labeling may become time-consuming as the prompt suite grows.

**Mitigation:** I will keep the initial prompt suite small, use structured constraint metadata, and expand the dataset only after the labeling format is stable.

### Risk 2: VLM checker reliability

The VLM-based checker may be unreliable for some constraints, especially spatial layout or ambiguous generations.

**Mitigation:** Human labels will be treated as the primary ground truth. The VLM checker will be evaluated against human labels rather than assumed to be correct.

### Risk 3: Ambiguous generated images

Some generated images may be difficult to label because objects are occluded, unclear, or only partially satisfy a constraint.

**Mitigation:** I will use a three-way label: pass, fail, and ambiguous. The ambiguous rate will be reported separately instead of forcing all cases into binary labels.

### Risk 4: Prompt repair may not improve constraint adherence

Constraint-aware prompt repair may not consistently fix failed constraints, or it may fix one constraint while breaking another.

**Mitigation:** The repair loop will be evaluated on a small subset of failed examples. I will measure both target-constraint improvement and non-target regressions.

### Risk 5: Evaluation matrix size

Varying constraint type, scene context, and constraint composition may create too many experimental conditions for a solo project.

**Mitigation:** I will start with a small number of prompt instances per condition and prioritize the core analysis over larger-scale expansion.



## What I Need Help With

I would like feedback from the course staff on the following questions:

1. Is the proposed scope reasonable for a solo project?
2. Is the proposed scale of around 36–60 initial generated images reasonable, with repair evaluated on a smaller failed subset?
3. Is it reasonable to use an existing text-to-image model and focus on evaluation, diagnosis, and prompt repair rather than training a new model?
4. Are there related papers or systems on text-to-image evaluation, prompt-following benchmarks, VLM-based visual checking, or prompt repair that I should look at?