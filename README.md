# CoGA-GRPO

**CoGA-GRPO** (Correctness-Gated Allocation GRPO) is a training method for multimodal video reasoning. It uses one shared correctness-derived allocation score to control adaptive KL regularization and safe response-length shaping.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10--3.12](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](pyproject.toml)

This is the anonymous review artifact accompanying the manuscript. It contains
the training, evaluation, configuration, and release-checking code needed to
inspect the reported method. Licensed datasets, media, initialization weights,
checkpoints, and generated outputs are not redistributed.

This repository is based on [Multimodal Open R1](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), distributed under Apache-2.0. Attribution is retained in [NOTICE](NOTICE).

## Highlights

- A complete two-stage pipeline: supervised fine-tuning followed by CoGA-GRPO.
- A shared correctness-gated allocation score for adaptive KL and response-length shaping.
- Rule-based format and accuracy rewards without online LLM judging.
- Deterministic evaluation on IntentBench, Daily-Omni, and WorldSense.
- Explicit negative controls for incorrect-length rewards and reversed KL allocation.
- Portable foreground launchers with visible training and evaluation progress.

## Method overview

For every prompt, the policy samples a complete GRPO response group. The trainer derives model-relative hardness and disagreement from rule-based correctness, optionally mixes them with a bounded multimodal input-load proxy, and uses the shared score to:

1. weaken KL regularization for groups requiring more policy freedom;
2. apply bounded, group-relative length shaping only to eligible easy and correct responses, rewarding shorter-than-group-mean completions and penalizing longer ones;
3. avoid rewarding incorrect responses for becoming longer.

## Repository structure

```text
data_config/stage1.yaml             SFT data manifest
data_config/stage2.yaml             GRPO data manifest
eval/evaluate_coga.py               Complete benchmark evaluator
eval/analyze_ib_axes.py            IntentBench axis analysis
run_scripts/run_sft.sh              Stage 1 supervised fine-tuning
run_scripts/run_e0_e6.sh           Controlled training launcher
run_scripts/eval_one.sh            Deterministic evaluation launcher
run_scripts/zero2_offload.json      DeepSpeed configuration
src/open_r1/grpo_qwenomni.py        Dataset, reward, and training entry point
src/open_r1/trainer/                GRPO allocation and optimization
src/open_r1/vlm_modules/            Qwen2.5-Omni adapter
```

## Environment

| Component | Tested configuration |
|---|---|
| Operating system | Ubuntu 22.04 |
| Python | 3.12 |
| PyTorch | 2.5.1 |
| CUDA | 12.4 |

## Installation

The experiments were developed with Python 3.12, CUDA 12.4, and CUDA-enabled PyTorch 2.5+. Install the PyTorch build suitable for your CUDA runtime first:

```bash
git clone <repository-url>
cd CoGA-GRPO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Private data

Training JSON files, media, weights, outputs, and logs are excluded from Git. The expected layout is:

```text
data_config/
|-- emer_rewrite.json
|-- social_iq_v2_rewrite.json
|-- Video-R1_rewrite.json
|-- stage1.yaml
`-- stage2.yaml
data/
|-- MER24/...
|-- social_iq/...
`-- Video-R1-data/...
```

Use symbolic links for `data/` or edit `data_root` in the YAML manifests. Dataset annotations, media, model weights, checkpoints, and generated outputs are intentionally excluded from Git.

## Training pipeline

```text
Qwen2.5-Omni base
        |
        v
Stage 1: multimodal SFT
        |
        v
Stage 2: CoGA-GRPO
        |
        v
Deterministic benchmark evaluation
```

## Stage 1: supervised fine-tuning

The original SFT stage is retained. It trains on the held-out halves of the emotion and social-reasoning data plus 30% of Video-R1, freezes the vision and audio towers, keeps audio extracted from video enabled, and runs for two epochs by default.

```bash
BASE_MODEL=/path/to/Qwen2.5-Omni-base \
DATASET_CONFIG=data_config/stage1.yaml \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=16666 \
bash run_scripts/run_sft.sh
```

Set `NPROC_PER_NODE` to use multiple GPUs. The output defaults to `output/coga-grpo-sft`.

## Stage 2: controlled GRPO training

Each experiment uses one process on one GPU so that all six generations for a prompt remain in one GRPO group.

```bash
BASE_MODEL=output/coga-grpo-sft \
DATASET_CONFIG=data_config/stage2.yaml \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=17100 \
bash run_scripts/run_e0_e6.sh E3 full 42
```

| Internal ID | Public method name | Dynamic KL | Input load | Safe length shaping | Purpose |
|---|---|---:|---:|---:|---|
| E0 | Standard GRPO | no | no | no | baseline |
| E1 | Dynamic KL only | yes | no | no | component ablation |
| E2 | Safe length shaping | no | no | yes | component ablation |
| E3 | Ours (CoGA-GRPO) | yes | no | yes | proposed method |
| E4 | Input-load augmentation | yes | yes | yes | optional ablation |
| E5 | Incorrect-length reward | yes | yes | yes | negative stress test |
| E6 | Reverse-KL allocation | reversed | yes | yes | negative stress test |

Use semantic names rather than internal IDs in the paper and public result tables.

## Reproduction scope

The public artifact supports inspection of the controller implementation,
training launchers, evaluation logic, and configuration mapping. Exact numerical
reproduction additionally requires the licensed benchmark media, rewritten
annotations, the initialization checkpoint, and substantial GPU compute. These
items are not included in this review repository. The reported manuscript results
should therefore be interpreted within the data and verification boundaries stated
in the paper.

## Complete evaluation

```bash
BASE_MODEL=/path/to/base-model \
DATA_ROOT=/path/to/benchmarks \
CUDA_VISIBLE_DEVICES=0 \
bash run_scripts/eval_one.sh \
  coga_grpo \
  output/controlled_ablation/e3_full_seed42 \
  ib
```

The evaluation launcher matches the manuscript protocol: four frames at 1 FPS,
pixel bounds of 50,176 and 75,264, greedy decoding, a 1,024-new-token cap, and
strict completeness. Missing samples and CUDA out-of-memory failures terminate
the run instead of being silently omitted from the reported benchmark result.
