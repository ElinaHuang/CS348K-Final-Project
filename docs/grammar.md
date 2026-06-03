# Constraint Grammar Design

This document describes the final constraint grammar used in the CS348K final project, **Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following**.

The grammar is designed to generate prompts with controlled visual requirements. It is not intended to cover every possible user prompt; instead, it creates a manageable testbed where prompt requirements can be systematically generated, labeled, repaired, and analyzed.

The machine-readable version of the grammar is:

```text
configs/grammar_config.yaml
```

## 1. Design Goal

The central idea of the project is to use one shared grammar across the full generate-check-repair-evaluate workflow.

The grammar does not only generate the original text prompt. It also defines the atomic visual constraints implied by that prompt, and those constraints are reused for:

1. human labeling,
2. VLM checking,
3. targeted repair prompt generation,
4. image-level and constraint-level evaluation.

The shared representation is:

```text
grammar instance
→ prompt text
→ atomic constraints
→ human / VLM labels
→ targeted repair instructions
→ metrics
```

For example:

```text
A realistic image of exactly 4 dice inside a cup on a shelf full of books and decorations.
```

is generated from a grammar instance with:

```text
number = 4
object_1 = die / dice
relation = inside
object_2 = cup
scene_context = on a shelf full of books and decorations
```

and decomposes into constraints such as:

```text
dice exist
cup exists
count = exactly 4 dice
dice inside cup
```

This decomposition makes partial prompt-following failures measurable and repairable.

## 2. Grammar Components

The final grammar varies prompts along six main dimensions.

| Component         | Values / examples                                            | Purpose                                                      |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Prompt family     | single spatial, single attribute, single cardinality, spatial+attribute, spatial+cardinality, attribute+cardinality | controls which visual requirements appear in a prompt        |
| Scene context     | simple, natural                                              | tests whether clutter/context complexity affects prompt-following |
| Object groups     | small items, containers, surfaces                            | keeps object choices typed and compatible with relations     |
| Attribute groups  | pattern/texture, material/finish                             | keeps object-attribute combinations meaningful               |
| Spatial relations | left/right, on top of/under, inside/outside                  | uses relations that are natural and labelable                |
| Numbers           | 3, 4, 5, 6                                                   | tests exact count control                                    |

## 3. Object Vocabulary

Objects are grouped by functional/visual role. These groups are used to make relation and attribute choices more physically plausible.

### 3.1 Small Items

```text
small paper clip
standard sewing button
coin
die
marble
standard safety pin
thumbtack
small flat washer
bottle cap
thin rubber band
```

### 3.2 Containers

```text
small box
serving bowl
cup
food container
tray
```

### 3.3 Surfaces

```text
notebook
sheet of paper
napkin
index card
plate
cutting board
```

## 4. Attribute Vocabulary

Plain single-color attributes are intentionally excluded from the final grammar because pilot generations made them too easy. Instead, color is combined with pattern, texture, material, or finish.

### 4.1 Pattern / Texture Attributes

Allowed object groups:

```text
surfaces, containers
```

Attributes:

```text
blue-and-white striped
red-and-white striped
black-and-white checkered
yellow-and-green polka-dotted
white speckled
multicolored
```

### 4.2 Material / Finish Attributes

Allowed object groups:

```text
small_items, containers
```

Attributes:

```text
reflective metallic
transparent glass
glossy white
matte black
plastic
ceramic
```

The grammar uses attribute groups so that generated prompts are systematic rather than arbitrary. For example, `blue-and-white striped index card` is treated as a pattern/texture requirement, while `matte black die` is treated as a material/finish requirement.

## 5. Spatial Relation Vocabulary

Spatial relations are grouped by relation type and use object-group constraints to avoid unnatural combinations.

| Relation key | Text            | Category    | Object 1 groups                   | Object 2 groups                   |
| ------------ | --------------- | ----------- | --------------------------------- | --------------------------------- |
| `left_of`    | to the left of  | lateral     | small_items, containers, surfaces | small_items, containers, surfaces |
| `right_of`   | to the right of | lateral     | small_items, containers, surfaces | small_items, containers, surfaces |
| `on_top_of`  | on top of       | support     | small_items                       | containers, surfaces              |
| `under`      | under           | support     | surfaces                          | small_items                       |
| `inside`     | inside          | containment | small_items                       | containers                        |
| `outside`    | outside         | containment | small_items                       | containers                        |

The final grammar uses physically meaningful relations such as `inside`, `on top of`, and `under`. Earlier pilot versions used more image-plane-style relations such as `above` and `below`, but those were harder to label consistently because a vertical image position does not always imply a physical spatial relation.

## 6. Scene Contexts

The grammar uses two scene context types.

### 6.1 Simple Contexts

Prefix:

```text
A clear image of
```

Contexts:

```text
on a lightly cluttered tabletop
on a plain desk with soft shadows
on a clean surface with a few unrelated objects
```

### 6.2 Natural Contexts

Prefix:

```text
A realistic image of
```

Contexts:

```text
on a cluttered desk with many office supplies
on a crowded kitchen counter with utensils nearby
on a picnic blanket with many small objects
on a shelf full of books and decorations
on a workbench with tools
on a craft table with beads and art supplies
```

Simple contexts reduce ambiguity. Natural contexts introduce more clutter and test whether the same constraints remain controllable and labelable in more realistic scenes.

## 7. Prompt Families and Templates

The final dataset contains 180 prompts across six prompt families. Prompt counts are balanced by family and assigned to T2I models using unpaired stratified assignment.

| Prompt family                    | Count | Template                                                     |
| -------------------------------- | ----: | ------------------------------------------------------------ |
| `single_spatial`                 |    24 | `{prefix} a {object_1} {relation_text} a {object_2} {scene_context}.` |
| `single_attribute`               |    24 | `{prefix} a {attribute_1} {object_1} and a {attribute_2} {object_2} {scene_context}.` |
| `single_cardinality`             |    24 | `{prefix} exactly {number} {object_1_plural} {scene_context}.` |
| `combined_spatial_attribute`     |    36 | `{prefix} a {attribute_1} {object_1} {relation_text} a {attribute_2} {object_2} {scene_context}.` |
| `combined_spatial_cardinality`   |    36 | `{prefix} exactly {number} {object_1_plural} {relation_text} a {object_2} {scene_context}.` |
| `combined_attribute_cardinality` |    36 | `{prefix} exactly {number} {attribute_1} {object_1_plural} and one {attribute_2} {object_2} {scene_context}.` |

Example prompt:

```text
A realistic image of exactly 4 dice inside a cup on a shelf full of books and decorations.
```

## 8. Atomic Constraint Types

Each generated prompt is decomposed into atomic constraints. These are the units used for human labeling, VLM checking, repair targeting, and metric aggregation.

The final grammar uses four representative constraint types.

| Constraint type    | Meaning                                                      | Example                                           |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| `object_identity`  | whether the requested object category is clearly present and recognizable | Does the image contain clearly identifiable dice? |
| `cardinality`      | whether the requested number of target objects is visible and countable | Does the image contain exactly 4 dice?            |
| `attribute`        | whether the requested object has the required pattern, texture, material, or finish | Does the image show a glossy white cup?           |
| `spatial_relation` | whether two requested objects satisfy the requested relation | Are the dice inside the cup?                      |

For the prompt:

```text
A realistic image of exactly 4 dice inside a cup on a shelf full of books and decorations.
```

the generated constraints are:

| Prompt element  | Constraint type    | Constraint                                              |
| --------------- | ------------------ | ------------------------------------------------------- |
| dice            | `object_identity`  | the image should contain clearly identifiable dice      |
| cup             | `object_identity`  | the image should contain a clearly identifiable cup     |
| exactly 4 dice  | `cardinality`      | the image should contain exactly 4 clearly visible dice |
| dice inside cup | `spatial_relation` | the dice should be inside the cup                       |

Attribute prompts generate analogous attribute constraints. For example, `a matte black die` becomes an `attribute` constraint checking whether the die has the requested material/finish.

## 9. Label Semantics

Each image-constraint pair receives one of three labels.

| Label       | Meaning                                                      |
| ----------- | ------------------------------------------------------------ |
| `pass`      | the constraint is reasonably satisfied                       |
| `fail`      | the constraint is clearly violated                           |
| `ambiguous` | the image does not provide enough evidence to confidently decide pass or fail |

The `ambiguous` label is intentionally broader than blur or occlusion. It also includes cases where the image contains an object-like shape whose identity or required property cannot be confidently determined.

This is important for T2I outputs because models often produce plausible-looking shapes that are not clearly the requested object.

## 10. VLM Checker Prompt Design

The VLM checker receives one image and one atomic constraint at a time. It is asked to decide whether the image satisfies that single visual constraint.

The checker prompt emphasizes:

1. judge only one visual constraint,
2. return `pass`, `fail`, or `ambiguous`,
3. use `ambiguous` only when the relevant object, count, attribute, or relation cannot be confidently judged,
4. prefer `pass` or `fail` when a reasonable human annotator could make a clear judgment.

VLM labels are not treated as ground truth. They are used to test whether repair target selection can be automated. Human labels remain the final evaluation signal.

## 11. Constraint-Aware Repair Rules

Repair uses the same atomic constraint structure. A repaired prompt is constructed as:

```text
original prompt
+
targeted repair instructions for failed/ambiguous constraints
```

The repair instruction depends on the constraint type.

| Constraint type    | Repair focus                                                 | Example repair instruction                                   |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `object_identity`  | make the requested object recognizable and avoid replacing it with a different category | Make sure the image clearly contains identifiable dice.      |
| `cardinality`      | enforce the exact count and keep target objects separated and countable | Make sure there are exactly 4 dice; do not generate more or fewer, and keep them separated and countable. |
| `attribute`        | apply the requested pattern, texture, material, or finish to the correct object | Make sure the cup has the glossy white material or surface finish. |
| `spatial_relation` | make the requested relation visually clear                   | Make sure the dice are clearly inside the cup.               |

For constraints labeled `ambiguous`, the repair instruction emphasizes clarity and easy judgment rather than only correcting a known failure. This reflects the fact that ambiguous cases are sometimes not clearly wrong; they may simply lack enough visual evidence.

## 12. Generated Files

Prompt generation produces three main files:

```text
data/prompts/prompts.csv
data/prompts/constraints.csv
data/generations/generation_plan.csv
```

`prompts.csv` contains prompt-level metadata such as:

```text
prompt_id
prompt
prompt_family
scene_context_type
scene_context
composition
```

`constraints.csv` contains one row per atomic constraint, with fields such as:

```text
constraint_id
prompt_id
constraint_type
check_text
object_1
object_2
relation
relation_text
target_object
target_count
attribute
attribute_category
object_slot
attribute_slot
```

`generation_plan.csv` assigns each prompt to a T2I provider/model using stratified assignment by prompt family.

## 13. Design Notes

The grammar was revised during the project based on pilot generations and human labeling.

Important changes include:

- replacing broad or ambiguous object names with more specific ones, such as `small flat washer` instead of `washer`;
- using `index card` rather than the broader `card`;
- separating attributes into `pattern_texture` and `material_finish`;
- removing plain single-color attributes because they were too easy;
- replacing image-plane spatial relations with more physically meaningful relations;
- keeping relative object scale as a prompt wording concern rather than adding a separate constraint type.

The final grammar is therefore not only a prompt generator. It is the shared representation that makes prompt-following failures measurable, repairable, and analyzable.