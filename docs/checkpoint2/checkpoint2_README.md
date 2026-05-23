# Checkpoint 2: Real-data Pilot and Repair Pipeline Integration

## Checkpoint Goal

Checkpoint 2 extends the Checkpoint 1 evaluation framework from mock data to a small real-data pilot. The goal is to show meaningful intermediate progress toward the final project: real generated images, human labels, VLM checker results, preliminary metrics, and repair-pipeline integration.

The scope of this checkpoint is:

```text
small real-data generation/evaluation pilot
+ VLM checker comparison against human labels
+ repair pipeline integration
+ small before/after repair pilot
```

The quantitative results are intentionally limited to the subset of generated images that has completed human labeling and uses constraints that are currently well-defined.

---

## What is included in Checkpoint 2

This checkpoint includes:

- real T2I generation using the OpenAI image API
- a selected prompt subset with 20 generated images included in the quantitative analysis
- 60 human-labeled atomic constraint examples
- VLM checker outputs using `gpt-4.1`
- constraint-level and image-level checker analysis
- image-level controllability analysis by prompt family, scene type, and composition
- repair prompt generation from failed human-labeled constraints
- a small repair pilot with 3 repair attempts and 4 target constraints

The generated image files are stored under:

```text
data/images/openai_checkpoint2/
```

The uploaded archive used for discussion may omit images due to file size, but the repository structure expects generated images in that folder.

---

## Scope of the Checkpoint 2 Quantitative Pilot

For this checkpoint, I define the scope as a small real-data evaluation pilot plus repair-pipeline integration. The quantitative results focus on the subset of generated images that have completed human labels and whose constraints are currently well-defined. In parallel, the repair module is integrated at the pipeline level: failed atomic constraints can be converted into structured repair prompts and repair target records, and a small repair pilot is included.

During preliminary human labeling, I found that the current cardinality and attribute/material constraints produce meaningful pass, fail, and ambiguous cases. In contrast, the current spatial grammar requires further revision. In particular, image-plane vertical relations such as “higher/lower in the image” do not always match natural human interpretations of physical spatial relations. Because the spatial definition is still being revised, I restrict the Checkpoint 2 quantitative results to the subset of prompts whose constraints are already well-defined and fully labeled.

The excluded spatial-heavy combined prompts are kept as generated examples, but they are not included in the current quantitative metrics. In the next iteration, I will revise the spatial grammar to use more physically meaningful relations such as “on top of,” “under,” and “inside,” then rerun generation and repair on the updated prompt set.

---

## Ambiguous Label Definition

In response to Checkpoint 1 feedback, I clarified the meaning of `ambiguous`.

The label definitions are:

- `pass`: the constraint is clearly satisfied.
- `fail`: the constraint is clearly violated.
- `ambiguous`: the image does not provide enough visual evidence to confidently mark the constraint as pass or fail.

Ambiguous cases include unclear object identity, occlusion, heavy overlap, partial visibility, uncertain counts, unclear attributes/materials, or spatial relations that cannot be judged reliably.

For pass labels, notes are optional and usually left blank. For fail or ambiguous labels, notes are used to briefly explain the reason.

---

## Pipeline Status

### 1. Prompt generation and real image generation

Checkpoint 2 uses a selected subset of prompts from the active hard grammar.

Important files:

```text
data/prompts/prompts_checkpoint2.csv
data/prompts/constraints_checkpoint2.csv
data/generations/generations_checkpoint2.csv
data/images/openai_checkpoint2/
```

The prompt subset used for quantitative analysis contains:

```text
20 images
60 atomic constraints
```

### 2. Human labeling

Human labels are stored in:

```text
data/labels/human_labels_checkpoint2.csv
```

The labeled subset includes:

```text
60 image × constraint examples
20 image-level examples after aggregation
```

### 3. VLM checker

The current VLM checker uses:

```text
gpt-4.1
```

VLM outputs are stored in:

```text
data/labels/vlm_labels_checkpoint2.csv
```

I also experimented with other VLM choices during development. The current checkpoint keeps one checker fixed for the reported metrics.

### 4. Repair pipeline

The repair pipeline takes failed human-labeled atomic constraints and generates:

```text
data/repaired/repaired_prompts_checkpoint2.csv
data/repaired/repair_targets_checkpoint2.csv
```

It then generates repaired images and prepares repaired-label templates:

```text
data/repaired/repaired_generations_checkpoint2.csv
data/repaired/repaired_human_labels_checkpoint2.csv
```

The current repair pilot includes:

```text
3 repair attempts
4 target constraints
```

---

## Intermediate Results

### Overall checker metrics

From `results/checkpoint2/summary_metrics.csv`:

| Metric | Value |
|---|---:|
| Constraint-level examples | 60 |
| Constraint-level VLM-human agreement | 0.8000 |
| Human constraint pass rate | 0.8833 |
| Human constraint fail rate | 0.0667 |
| Human constraint ambiguous rate | 0.0500 |
| Image-level examples | 20 |
| Image-level VLM-human agreement | 0.6500 |
| Human image pass rate | 0.7500 |
| Human image fail rate | 0.1500 |
| Human image ambiguous rate | 0.1000 |

### Constraint-level observations

From `results/checkpoint2/constraint_checker_metrics_by_type.csv`:

| Constraint type | Examples | Human pass rate | Human fail rate | Human ambiguous rate | VLM-human agreement |
|---|---:|---:|---:|---:|---:|
| object existence | 35 | 0.9143 | 0.0571 | 0.0286 | 0.8000 |
| cardinality | 5 | 0.6000 | 0.4000 | 0.0000 | 0.8000 |
| attribute/material | 10 | 0.8000 | 0.0000 | 0.2000 | 0.6000 |
| spatial relation | 10 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |

These numbers are preliminary. They suggest that cardinality and attribute/material prompts are useful for producing non-pass examples, while the current spatial subset is too easy under the current image-plane definition.

### Image-level controllability observations

From `results/checkpoint2/image_controllability_by_dimension.csv`:

| Prompt family | Examples | Human image pass rate | Human image fail rate | Human image ambiguous rate |
|---|---:|---:|---:|---:|
| spatial layout | 10 | 1.0000 | 0.0000 | 0.0000 |
| attribute binding | 5 | 0.6000 | 0.2000 | 0.2000 |
| cardinality | 5 | 0.4000 | 0.4000 | 0.2000 |

This supports the current next step: keep cardinality and attribute/material settings, but revise the spatial grammar.

---

## Repair Pilot Results

The repair pilot is intentionally small. It is mainly used to verify that the repair pipeline is connected end-to-end.

From `results/checkpoint2/repair_summary_metrics.csv`:

| Metric | Value |
|---|---:|
| Repair attempts | 3 |
| Target constraints | 4 |
| Target fixed rate | 0.7500 |
| Image fixed rate | 0.6667 |
| Repair regression rate | 0.0000 |

From `results/checkpoint2/repair_success_by_constraint_type.csv`:

| Target constraint type | Targets | Fixed | Target fixed rate |
|---|---:|---:|---:|
| object existence | 2 | 2 | 1.0000 |
| cardinality | 2 | 1 | 0.5000 |

The repair results are encouraging but too small to support final conclusions. They show that the pipeline can generate repair prompts, produce repaired images, and compute before/after metrics.

---

## Current Findings Relative to Project Goals

### What has been answered successfully so far

1. **The evaluation pipeline works on real T2I outputs.**  
   The project now runs beyond mock data and produces human/VLM label comparisons on real generated images.

2. **The ambiguous label is now operationally defined.**  
   Ambiguous means the image does not provide enough evidence for confident pass/fail judgment.

3. **Cardinality and attribute/material constraints are useful stress tests.**  
   These categories produce a mix of pass, fail, and ambiguous cases.

4. **The VLM checker can approximate human labels, but it is not perfect.**  
   Constraint-level agreement is 0.8000, while image-level agreement is 0.6500 on the current pilot subset.

5. **The repair pipeline is integrated.**  
   Failed constraints can be converted into repair prompts and evaluated with target-level, image-level, and regression metrics.

### What is still not up to par

1. **The current spatial grammar needs revision.**  
   Image-plane vertical relations do not always match natural human interpretation. Under the current definition, many spatial examples are also too easy.

2. **The quantitative pilot is small.**  
   The current results are useful for debugging and checkpoint evaluation, but not enough for final statistical conclusions.

3. **The repair experiment needs to be rerun after grammar revision.**  
   The current repair pilot verifies the pipeline, but the final repair experiment should use the revised spatial grammar and a larger prompt set.

4. **VLM checker calibration remains open.**  
   Different VLMs show different tendencies toward under-detecting or over-detecting non-pass cases. The final report will use the human-labeled subset to calibrate the checker prompt and model choice.

---

## Output Files

Important Checkpoint 2 files:

```text
configs/checkpoint2_config.yaml

data/prompts/prompts_checkpoint2.csv
data/prompts/constraints_checkpoint2.csv
data/generations/generations_checkpoint2.csv
data/labels/human_labels_checkpoint2.csv
data/labels/vlm_labels_checkpoint2.csv

data/repaired/repaired_prompts_checkpoint2.csv
data/repaired/repair_targets_checkpoint2.csv
data/repaired/repaired_generations_checkpoint2.csv
data/repaired/repaired_human_labels_checkpoint2.csv

results/checkpoint2/summary_metrics.csv
results/checkpoint2/constraint_checker_metrics_by_type.csv
results/checkpoint2/image_controllability_by_dimension.csv
results/checkpoint2/repair_summary_metrics.csv
results/checkpoint2/repair_success_by_constraint_type.csv
```

---

## How to Run

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

API calls require `OPENAI_API_KEY` to be set in the environment.

---

## Next Steps

Before the final report, I plan to:

1. revise the spatial grammar to use more physically meaningful relations such as `on top of`, `under`, and `inside`;
2. update the constraint decomposition so object identity, count, attribute presence, attribute binding, and spatial relation are more clearly separated;
3. rerun generation with the revised grammar;
4. rerun VLM checking and repair evaluation on the revised prompt set;
5. add a second T2I generator to reduce generator-specific bias;
6. improve the presentation of generated image examples and summary plots for the final report.
