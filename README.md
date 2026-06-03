# Constraint-Level Evaluation and Repair for Text-to-Image Prompt Following

**CS348K Final Project**

**Student:** Yiling Huang

**SUNet ID:** yilhuang

## Summary

Text-to-image (T2I) models often generate images that look plausible overall while still failing specific parts of a prompt, such as object identity, exact count, object attributes, or spatial relations. This project studies whether decomposing T2I prompts into atomic visual constraints can make these partial prompt-following failures measurable, repairable, and scalable with VLM-assisted checking.

The project implements an end-to-end constraint-level workflow:

```text
controlled grammar
→ prompts + atomic constraints
→ T2I image generation
→ human labels + VLM checker labels
→ targeted repair prompts
→ repaired images
→ before/after evaluation
```

The main result is positive: constraint-aware repair improves both image-level and constraint-level pass rates under human evaluation. Human-triggered repair improves image pass rate by **51.9%**, while VLM-triggered repair improves image pass rate by **28.0%**. These results support the main claim that constraint-level structure is a useful systems abstraction for making partial T2I prompt-following failures measurable, actionable, and analyzable.

## Final Deliverables

- [Final report](docs/final_report/final_report.md)
- [Final presentation slides](docs/final_report/Presentation_Slides.pdf)
- [Grammar design document](docs/grammar.md)
- [Final grammar config](configs/grammar_config.yaml)
- [Final experiment config](configs/final_experiment_config.yaml)
- [Final result CSVs](results/final/)

Older proposal and checkpoint materials are kept under `docs/proposal/`, `docs/checkpoint1/`, and `docs/checkpoint2/` for project history.

## Project Scope

The grammar is not intended to cover every possible user prompt. Instead, it creates a controlled testbed where prompt requirements can be systematically generated, labeled, repaired, and analyzed.

The final grammar covers six prompt families:

- `single_spatial`
- `single_attribute`
- `single_cardinality`
- `combined_spatial_attribute`
- `combined_spatial_cardinality`
- `combined_attribute_cardinality`

It uses four representative atomic constraint types:

- `object_identity`
- `cardinality`
- `attribute`
- `spatial_relation`

The purpose of this setup is to test whether partial prompt-following failures can be localized and used as actionable repair signals.

## Final Experiment Overview

| Component           | Setting                                               |
| ------------------- | ----------------------------------------------------- |
| Prompts             | 180                                                   |
| Atomic constraints  | 720                                                   |
| T2I models          | OpenAI `gpt-image-1`, Google `gemini-2.5-flash-image` |
| VLM checker         | GPT-4.1                                               |
| Labels              | `pass`, `fail`, `ambiguous`                           |
| Repair triggers     | human-triggered, VLM-triggered, combined              |
| Scene context types | `simple`, `natural`                                   |

### Initial Generation

| Level      | Human pass | Human fail | Human ambiguous |
| ---------- | ---------: | ---------: | --------------: |
| Image      |      55.0% |      33.3% |           11.7% |
| Constraint |      85.8% |       9.4% |            4.7% |

The gap between image-level and constraint-level pass rates shows that many images are partial failures rather than total failures. Constraint-level evaluation reveals which specific requirement failed.

### VLM-Human Agreement

| Level      | Agreement | Non-pass precision | Non-pass recall | Non-pass F1 |
| ---------- | --------: | -----------------: | --------------: | ----------: |
| Image      |     69.4% |              72.0% |           82.7% |       77.0% |
| Constraint |     80.4% |              42.9% |           71.6% |       53.7% |

The VLM checker is useful as an automatic repair trigger, but human labels remain the final evaluation ground truth.

### Repair Results

| Trigger source  | Attempts | Constraint pass Δ | Image pass Δ | Image fixed rate | Regression rate |
| --------------- | -------: | ----------------: | -----------: | ---------------: | --------------: |
| Human-triggered |       81 |         +16.7 pts |    +51.9 pts |            51.9% |           11.1% |
| VLM-triggered   |       93 |          +9.7 pts |    +28.0 pts |            34.4% |           15.1% |
| Combined        |      174 |         +12.9 pts |    +39.1 pts |            42.5% |           13.2% |

The repair results show that the same constraint grammar used for checking can also turn failed or ambiguous constraints into useful targeted repair instructions.

## Repository Structure

```text
configs/
  grammar_config.yaml              # final grammar and dataset specification
  final_experiment_config.yaml     # paths and stage configuration

scripts/
  generate_prompts.py              # grammar → prompts, constraints, generation plan
  generate_images.py               # T2I API image generation
  create_human_label_template.py   # label template for initial generations
  run_vlm_checker.py               # VLM checking for image-constraint pairs
  analyze_checker.py               # initial generation + VLM agreement analysis
  generate_repair_prompts.py       # failed constraints → repaired prompts
  create_repair_label_template.py  # label template for repaired images
  analyze_repair.py                # before/after repair analysis
  run_final_experiment.py          # staged final experiment runner
  utils.py                         # shared CSV, parsing, and label utilities

data/
  prompts/                         # prompts.csv, constraints.csv
  generations/                     # generation plan and generation metadata
  images/                          # generated initial images
  labels/                          # human and VLM labels
  repaired/                        # repaired prompts, generations, labels, images

results/
  final/                           # final evaluation and repair result CSVs

docs/
  grammar.md                       # human-readable grammar design
  final_report/                    # final report, slides, and report figures
  checkpoint1/, checkpoint2/       # archived checkpoint materials

tests/
  pytest tests for prompt generation, VLM prompts, retry logic,
  repair prompt generation, and analysis metrics
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For API-backed generation/checking, create a `.env` file or export the required API keys for the providers you use. The scripts load environment variables through `python-dotenv`.

Typical variables:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

You can use `--dry-run` for image generation or VLM checking stages when testing the pipeline without making API calls.

## Running the Final Experiment

The final experiment is staged because some steps require API calls and manual human labeling.

Run commands from the `scripts/` directory:

```bash
cd scripts
```

### 1. Generate prompts and constraints

```bash
python run_final_experiment.py --stage generate-prompts
```

Outputs:

```text
data/prompts/prompts.csv
data/prompts/constraints.csv
data/generations/generation_plan.csv
```

### 2. Generate initial images

```bash
python run_final_experiment.py --stage generate-images
```

For a no-API test run:

```bash
python run_final_experiment.py --stage generate-images --dry-run
```

### 3. Create human label template

```bash
python run_final_experiment.py --stage create-human-template
```

Fill in `data/labels/human_labels.csv` manually.

### 4. Run VLM checker

```bash
python run_final_experiment.py --stage run-vlm
```

For a no-API test run:

```bash
python run_final_experiment.py --stage run-vlm --dry-run
```

### 5. Analyze initial generation and VLM agreement

```bash
python run_final_experiment.py --stage analyze-initial
```

Outputs are written to:

```text
results/final/
```

### 6. Generate repair prompts

```bash
python run_final_experiment.py --stage generate-repairs-human
python run_final_experiment.py --stage generate-repairs-vlm
```

### 7. Generate repaired images

```bash
python run_final_experiment.py --stage generate-repair-images-human
python run_final_experiment.py --stage generate-repair-images-vlm
```

### 8. Create repaired-image label templates

```bash
python run_final_experiment.py --stage create-repair-label-template-human
python run_final_experiment.py --stage create-repair-label-template-vlm
```

Fill in the repaired-image human labels manually.

### 9. Analyze repair

```bash
python run_final_experiment.py --stage analyze-repair-human
python run_final_experiment.py --stage analyze-repair-vlm
python run_final_experiment.py --stage analyze-repair-combined
```

## Tests

Run all tests from the repository root:

```bash
pytest tests
```

The tests cover:

- final prompt and constraint generation
- VLM checker prompt construction
- VLM retry logic
- repair prompt generation
- initial analysis metrics
- repair analysis metrics

## Notes on Labels

Labels use three values:

- `pass`: the visual requirement is reasonably satisfied.
- `fail`: the visual requirement is clearly violated.
- `ambiguous`: the image does not provide enough evidence to confidently decide pass or fail.

In this project, `ambiguous` includes not only blur, occlusion, or small objects, but also generated object-like shapes whose identity or required property cannot be confidently determined.

## Project History

The repository also contains earlier project stages:

- `docs/proposal/`: original proposal.
- `docs/checkpoint1/`: mock-data/checker pipeline checkpoint.
- `docs/checkpoint2/`: real-data pilot checkpoint.

The final experiment supersedes the checkpoint results and uses the final grammar in `configs/grammar_config.yaml`.