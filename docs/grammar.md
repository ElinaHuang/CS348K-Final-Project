# Constraint Grammar Design

This document defines the first version of the constraint grammar for the CS348K final project. The grammar is intentionally controlled and may be revised after the first batch of image generation, human labeling, and VLM-checker experiments.

The grammar has three connected roles:

1. **Prompt generation:** generate controlled text-to-image prompts.
2. **Constraint labeling:** define the atomic constraints that humans and VLM checkers should evaluate.
3. **Prompt repair:** define constraint-type-aware rewrite rules for failed constraints.

The main design principle is that the same constraint grammar should connect the whole pipeline:

```text
prompt grammar
→ expected atomic constraints
→ human / VLM checking criteria
→ failed constraint types
→ targeted repair rules
```

---

## 1. Core Grammar Dimensions

The core grammar varies prompts along three dimensions.

### 1.1 Constraint Family

The core constraint families are:

1. **Object cardinality**: whether the image contains exactly the requested number of target objects.
2. **Attribute binding**: whether a specified attribute, usually color, is attached to the intended object.
3. **Spatial layout**: whether one object appears in the specified 2D spatial relation to another object.

### 1.2 Scene Context

The grammar includes two scene context levels:

1. **Simple scenes**: controlled settings such as a clean tabletop or plain background.
2. **Natural scenes**: richer scenes such as a kitchen, park, or living room.

Simple scenes are used to reduce ambiguity. Natural scenes test whether the same constraints become harder in more realistic contexts.

### 1.3 Constraint Composition

The grammar includes:

1. **Single-constraint prompts**: prompts primarily testing one constraint family.
2. **Combined-constraint prompts**: prompts containing multiple atomic requirements, such as cardinality + attribute + spatial layout.

Combined prompts are important because a generated image may satisfy some requirements while violating others.

---

## 2. Shared Vocabulary

The first version uses a small controlled vocabulary. This is not meant to cover all possible user prompts; it is meant to create a manageable, human-checkable testbed.

### 2.1 Objects

Initial object vocabulary:

- cup
- book
- apple
- ball
- box
- toy car

These objects are chosen because they are common, visually recognizable, and easy to use in spatial and attribute constraints.

### 2.2 Attributes

Initial attribute vocabulary:

- red
- blue
- green
- yellow

The first version focuses on color attributes because they are easy to check visually. More subtle attributes can be added later if needed.

### 2.3 Numbers

Initial cardinality vocabulary:

- 2
- 3
- 4

The grammar avoids larger counts in the first version because they are harder for both image generators and human annotators.

### 2.4 Spatial Relations

Initial spatial relation vocabulary:

- left of
- right of
- above
- below

Spatial relations are interpreted in the 2D image coordinate system from the viewer's perspective.

### 2.5 Scene Contexts

Simple contexts:

- on a clean tabletop
- on a plain white background
- on a clean floor

Natural contexts:

- in a kitchen
- in a park
- in a living room

Style is not treated as a separate grammar dimension. Instead, the prompt prefix is tied to scene context:

- simple scene: `A clean front-facing image of ...`
- natural scene: `A realistic image of ...`

---

## 3. Prompt Generation Grammar

Prompt generation produces two structured artifacts:

1. `prompts.csv`: one row per generated prompt.
2. `constraints.csv`: one row per expected atomic constraint associated with each prompt.

The generated image itself is not labeled at this stage. Labels are added later after image generation.

---

### 3.1 Object Cardinality Prompts

#### Template

```text
{prefix} exactly {number} {object_plural} {scene_context}.
```

#### Examples

```text
A clean front-facing image of exactly three cups on a clean tabletop.
A clean front-facing image of exactly two apples on a plain white background.
A realistic image of exactly four balls in a park.
```

#### Expected Atomic Constraints

A cardinality prompt generates at least two atomic constraints:

1. `object_existence`: the target object should be visible.
2. `cardinality`: the number of visible target objects should equal the requested count.

Example constraints:

```text
The cup should be clearly visible.
There should be exactly three visible cups.
```

---

### 3.2 Attribute Binding Prompts

#### Template

```text
{prefix} a {color_1} {object_1} next to a {color_2} {object_2} {scene_context}.
```

#### Examples

```text
A clean front-facing image of a red cup next to a blue book on a clean tabletop.
A clean front-facing image of a green apple next to a yellow box on a plain white background.
A realistic image of a blue ball next to a red toy car in a park.
```

#### Expected Atomic Constraints

An attribute-binding prompt generates atomic constraints such as:

1. `object_existence`: object 1 should be visible.
2. `object_existence`: object 2 should be visible.
3. `attribute`: object 1 should have attribute 1.
4. `attribute`: object 2 should have attribute 2.

Example constraints:

```text
The cup should be clearly visible.
The book should be clearly visible.
The cup should be red.
The book should be blue.
```

The important point is object-specific binding. A red object and a blue object appearing somewhere in the image is not enough; the red attribute should apply to the cup and the blue attribute should apply to the book.

---

### 3.3 Spatial Layout Prompts

#### Template

```text
{prefix} a {object_1} {relation_text} a {object_2} {scene_context}.
```

#### Examples

```text
A clean front-facing image of a cup to the left of a book on a clean tabletop.
A clean front-facing image of an apple to the right of a box on a plain white background.
A realistic image of a ball above a toy car in a park.
A realistic image of a book below an apple in a living room.
```

#### Expected Atomic Constraints

A spatial-layout prompt generates atomic constraints such as:

1. `object_existence`: object 1 should be visible.
2. `object_existence`: object 2 should be visible.
3. `spatial_relation`: object 1 should satisfy the specified spatial relation with respect to object 2.

Example constraints:

```text
The cup should be clearly visible.
The book should be clearly visible.
The cup should be to the left of the book in the 2D image.
```

Spatial relations are judged from the viewer's perspective in the image plane:

- left = left side of the image
- right = right side of the image
- above = higher in the image
- below = lower in the image

---

## 4. Combined-Constraint Prompt Grammar

Combined prompts contain multiple atomic requirements in one prompt. They are used to test partial prompt adherence.

### 4.1 Attribute + Spatial

#### Template

```text
{prefix} a {color_1} {object_1} {relation_text} a {color_2} {object_2} {scene_context}.
```

#### Example

```text
A clean front-facing image of a red cup to the left of a blue book on a clean tabletop.
```

#### Expected Atomic Constraints

```text
The cup should be clearly visible.
The book should be clearly visible.
The cup should be red.
The book should be blue.
The cup should be to the left of the book in the 2D image.
```

---

### 4.2 Cardinality + Attribute

#### Template

```text
{prefix} exactly {number} {color} {object_plural} {scene_context}.
```

#### Example

```text
A clean front-facing image of exactly three red cups on a clean tabletop.
```

#### Expected Atomic Constraints

```text
The cup should be clearly visible.
There should be exactly three visible cups.
The cups should be red.
```

---

### 4.3 Cardinality + Spatial

#### Template

```text
{prefix} exactly {number} {object_1_plural} {relation_text} a {object_2} {scene_context}.
```

#### Example

```text
A clean front-facing image of exactly three cups to the left of a book on a clean tabletop.
```

#### Expected Atomic Constraints

```text
The cup should be clearly visible.
The book should be clearly visible.
There should be exactly three visible cups.
The cups should be to the left of the book in the 2D image.
```

---

### 4.4 Cardinality + Attribute + Spatial

#### Template

```text
{prefix} exactly {number} {color_1} {object_1_plural} {relation_text} a {color_2} {object_2} {scene_context}.
```

#### Example

```text
A clean front-facing image of exactly three red cups to the left of a blue book on a clean tabletop.
```

#### Expected Atomic Constraints

```text
The cup should be clearly visible.
The book should be clearly visible.
There should be exactly three visible cups.
The cups should be red.
The book should be blue.
The cups should be to the left of the book in the 2D image.
```

This combined case is more complex, so it may be used in smaller quantity than the simpler prompt families.

---

## 5. Constraint Labeling Grammar

Labeling is performed at the atomic constraint level. The unit of labeling is:

```text
image_id × constraint_id
```

Labels are not assigned only at the whole-image level.

### 5.1 Universal Labels

All atomic constraints use the same three labels:

- `pass`: the constraint is clearly satisfied.
- `fail`: the constraint is clearly violated.
- `ambiguous`: the constraint cannot be judged confidently because of unclear object identity, occlusion, overlap, image quality, or visual ambiguity.

For combined prompts, prompt-level success is derived from atomic labels:

```text
full prompt success = all required atomic constraints are pass
prompt failure = at least one required atomic constraint is fail
prompt ambiguous = no fails, but at least one required atomic constraint is ambiguous
```

---

### 5.2 Object Existence Labeling

#### Check

```text
The {object} should be clearly visible.
```

#### Pass

The object is clearly visible and identifiable.

#### Fail

The object is clearly absent.

#### Ambiguous

The object may be present but is unclear, occluded, partially visible, or not confidently identifiable.

---

### 5.3 Cardinality Labeling

#### Check

```text
There should be exactly {number} visible {object_plural}.
```

#### Pass

Exactly the requested number of target objects are clearly visible.

#### Fail

The number of clearly visible target objects is clearly different from the requested number.

#### Ambiguous

The target objects are partially occluded, merged, unclear, or not visually distinct enough to count confidently.

---

### 5.4 Attribute Labeling

#### Check

```text
The {object} should be {attribute}.
```

#### Pass

The specified object is clearly visible and has the specified attribute.

#### Fail

The object is visible but does not have the specified attribute, or the attribute is clearly attached to the wrong object.

#### Ambiguous

The object or attribute is unclear because of lighting, occlusion, color ambiguity, or image quality.

---

### 5.5 Spatial Relation Labeling

#### Check

```text
The {object_1} should be {relation_text} the {object_2} in the 2D image.
```

#### Pass

Both objects are clearly visible, and the specified spatial relation clearly holds in the 2D image plane.

#### Fail

Both objects are clearly visible, but the specified relation is clearly wrong, reversed, or not satisfied.

#### Ambiguous

Either object is unclear, missing, occluded, overlapping, or the viewpoint makes the spatial relation difficult to judge confidently.

Spatial judgments use the viewer's image coordinate system.

---

## 6. VLM Checker Prompt Grammar

The VLM checker uses the same atomic constraints as the human labeling grammar. The difference is that the labeling rule is converted into an API prompt.

The checker should be given one image and one atomic constraint at a time. The checker output should be parsed and written to `vlm_labels.csv`.

### 6.1 Universal VLM Checker Instruction

```text
You are a visual constraint checker.

Your task is to judge whether the image satisfies ONE visual constraint.

Return exactly one label:
- pass
- fail
- ambiguous

Use "pass" only if the constraint is clearly satisfied.
Use "fail" if the constraint is clearly violated.
Use "ambiguous" if the image is unclear, objects are occluded, object identity is uncertain, or the constraint cannot be confidently judged.

Return ONLY valid JSON in the following format:
{
  "label": "pass" | "fail" | "ambiguous",
  "reason": "one short sentence"
}
```

### 6.2 Object Existence Checker

```text
Constraint type: object existence
Constraint: A {object} should be clearly visible in the image.

Decision rules:
- Return "pass" if the object is clearly visible.
- Return "fail" if the object is clearly absent.
- Return "ambiguous" if the object may be present but is unclear, occluded, or not confidently identifiable.
```

### 6.3 Cardinality Checker

```text
Constraint type: object cardinality
Constraint: There should be exactly {number} clearly visible {object_plural}.

Decision rules:
- Return "pass" if exactly {number} clearly identifiable {object_plural} are visible.
- Return "fail" if the number of clearly visible {object_plural} is clearly different from {number}.
- Return "ambiguous" if the objects are unclear, occluded, partially visible, or cannot be counted confidently.
- Count only clearly visible instances of the target object.
```

### 6.4 Attribute Checker

```text
Constraint type: object attribute
Constraint: The {object} should be {attribute}.

Decision rules:
- Return "pass" if the specified object is clearly visible and has the specified attribute.
- Return "fail" if the object is clearly visible but does not have the specified attribute, or if the attribute is clearly attached to the wrong object.
- Return "ambiguous" if the object or attribute is unclear because of lighting, occlusion, or image quality.
```

### 6.5 Spatial Relation Checker

```text
Constraint type: spatial relation
Constraint: The {object_1} should be {relation_text} the {object_2}.

Decision rules:
- Use the 2D image coordinate system from the viewer's perspective.
- "left" means the left side of the image.
- "right" means the right side of the image.
- "above" means higher in the image.
- "below" means lower in the image.
- Return "pass" if both objects are clearly visible and the relation clearly holds.
- Return "fail" if both objects are clearly visible and the relation is clearly violated.
- Return "ambiguous" if either object is unclear, missing, occluded, overlapping, or the relation cannot be confidently judged.
```

---

## 7. Repair Grammar

The repair grammar is designed around one principle:

```text
constraint-level failure triggers repair,
image-level prompt is reconstructed for repair,
both constraint-level and image-level repair success are evaluated.
```

A failed atomic constraint is used to diagnose what went wrong, but the repaired prompt is generated at the image level. The repaired prompt should preserve all original requirements while strengthening failed or ambiguous constraints.

### 7.1 Repair Trigger Labels

The grammar distinguishes between `fail` and `ambiguous` cases.

#### Fail: Corrective Repair

A `fail` label means the constraint is clearly violated. Repair should use a **corrective rewrite** that explicitly states what must be true and, when useful, what should be avoided.

Example:

```text
The cup must be clearly to the left of the book in the 2D image. Keep the two objects separated so the relation is easy to verify.
```

#### Ambiguous: Clarification Repair

An `ambiguous` label means the image does not provide enough visual evidence to confidently mark the constraint as pass or fail. Repair should use a **clarification rewrite** that improves visibility, separability, or judgeability.

Example:

```text
Make the cup and the book clearly visible, separated, and not overlapping. Use a front-facing layout so the spatial relation is easy to judge.
```

For Checkpoint 2, the repair pilot uses `fail` as the trigger label. The grammar defines ambiguous repair as well, but ambiguous-triggered repair is left as a final-report extension.

---

### 7.2 Repair Strategy

The code supports two repair strategies.

#### `repair_all_failed`

One repaired prompt is generated for each failed image. All failed constraints in that image are strengthened together.

This strategy is image-level: it asks whether repairing all diagnosed failures can make the whole generated image satisfy all original constraints.

#### `repair_single_target`

One repaired prompt is generated for each target failed constraint. If a source image has multiple failed constraints, each failed constraint can produce its own repaired prompt.

This strategy supports a more detailed final analysis of repair difficulty by constraint type.

Checkpoint 2 uses `repair_all_failed` by default, while keeping `repair_single_target` available in the code.

---

### 7.3 Repaired Prompt Construction

The repaired prompt is not created by pasting the original prompt and adding a vague instruction such as "fix the image." Instead, it is reconstructed from the original constraint metadata:

```text
same original constraint set
+ strengthened failed constraints
+ clarity / anti-regression instructions
```

The format is:

```text
Create a clear image satisfying all of the following visual requirements:
1. <requirement from original constraint 1>
2. <requirement from original constraint 2>
3. <strengthened instruction for failed constraint>
...

Composition guidelines:
- Use a clear, front-facing composition.
- Keep all relevant objects visible and separated.
- Avoid occlusion, heavy overlap, cropped objects, or ambiguous object shapes.
- Do not introduce extra objects that could be confused with the requested objects.
```

This makes the repair prompt comparable to the original prompt: both come from the same underlying constraint set, but the repair prompt expresses the constraints in a more explicit and structured way.

---

### 7.4 Constraint-Type-Specific Repair Rules

#### Object Existence

Fail repair:

```text
The <object> must be clearly visible as a distinct object in the image. Do not omit it.
```

Ambiguous repair:

```text
Make <object> clearly visible as a distinct, unobstructed object. Avoid cropping, overlap, or ambiguous shapes.
```

#### Cardinality

Fail repair:

```text
There must be exactly <N> clearly visible <object_plural> total. Do not include extra <object_plural>, and do not hide or merge any of them.
```

Ambiguous repair:

```text
Make exactly <N> clearly separated <object_plural> visible. Avoid overlap or partial objects that make counting uncertain.
```

#### Attribute Binding

Fail repair:

```text
The <object> must clearly have the <attribute> attribute. Do not apply this attribute to the wrong object.
```

Ambiguous repair:

```text
Make the <attribute> attribute of <object> visually clear and unambiguous. Avoid lighting or occlusion that makes the attribute hard to judge.
```

#### Spatial Relation

Fail repair:

```text
The <object_1> must be clearly <relation_text> the <object_2> in the 2D image. Keep the two objects separated so the relation is easy to verify.
```

Ambiguous repair:

```text
Make the spatial relation unambiguous: <object_1> is clearly <relation_text> <object_2>. Use a front-facing layout with separated objects and no overlap.
```

Relation-specific layout guidance can be used when available:

```text
left_of:
  object_1 → left side of the image
  object_2 → right side of the image

right_of:
  object_1 → right side of the image
  object_2 → left side of the image

above:
  object_1 → upper part of the image
  object_2 → lower part of the image

below:
  object_1 → lower part of the image
  object_2 → upper part of the image
```

---

### 7.5 Combined-Constraint Repair

For combined prompts, repair is still generated at the image level.

The repaired prompt should:

1. Preserve all original constraints.
2. Strengthen the failed or ambiguous constraints.
3. Add clarity and anti-regression guidance.

Example:

```text
Create a clear image satisfying all of the following visual requirements:
1. The cup is clearly visible.
2. The book is clearly visible.
3. The cup is red.
4. The book is blue.
5. The cup must be clearly to the left of the book in the 2D image.

Composition guidelines:
- Use a clear, front-facing composition.
- Keep the cup and book separated and not overlapping.
- Do not swap the colors.
- Do not introduce extra confusing objects.
```

---

### 7.6 Repair Output Tables

Repair uses two linked tables.

#### `repaired_prompts_checkpoint2.csv`

One row per repair attempt:

```text
repair_id
source_image_id
source_prompt_id
repair_strategy
trigger_label_source
trigger_labels
num_target_constraints
target_constraint_ids
original_prompt
repaired_prompt
```

This table represents image-level repair attempts.

#### `repair_targets_checkpoint2.csv`

One row per target constraint:

```text
repair_id
source_image_id
source_prompt_id
target_constraint_id
target_constraint_type
before_label
repair_action
target_constraint_text
```

This table makes it possible to analyze repair success by constraint type.

---

### 7.7 Repair Evaluation

Repair is evaluated at four levels:

1. **Target-constraint repair success:** whether the targeted failed constraint becomes `pass` after repair.
2. **Image-level repair success:** whether the repaired image satisfies all original constraints.
3. **Regression analysis:** whether repair introduces new failures among constraints that previously passed.
4. **Repair success by constraint type:** whether some constraint types are easier or harder to repair.

The main repair metrics are:

```text
target_fixed_rate
image_fixed_rate
regression_rate
target_fixed_rate_by_constraint_type
```

---

## 8. Expected Files Produced by the Grammar Pipeline

The grammar design supports the following files:

```text
prompts.csv        # prompt-level information
constraints.csv    # expected atomic constraints for each prompt
generations.csv    # generated image metadata
human_labels.csv   # human labels for image × constraint pairs
vlm_labels.csv     # VLM checker labels for image × constraint pairs
```

For repair experiments, it also supports:

```text
repaired_prompts_checkpoint2.csv
repair_targets_checkpoint2.csv
repaired_generations_checkpoint2.csv
repaired_human_labels_checkpoint2.csv
repair_target_results.csv
repair_image_results.csv
repair_success_by_constraint_type.csv
repair_summary_metrics.csv
```

The grammar itself is stored in two forms:

```text
docs/grammar.md              # human-readable design document
configs/grammar_config.yaml  # machine-readable config for scripts
```
