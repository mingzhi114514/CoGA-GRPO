# CoGA-GRPO

**CoGA-GRPO** (Correctness-Gated Allocation GRPO) is a training method for efficient multimodal video reasoning. It uses one shared allocation score to control adaptive KL regularization and safe response-length shaping.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10--3.12](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](pyproject.toml)
[![Release checks](https://github.com/mingzhi114514/CoGA-GRPO/actions/workflows/ci.yml/badge.svg)](https://github.com/mingzhi114514/CoGA-GRPO/actions/workflows/ci.yml)

This repository is based on [Multimodal Open R1](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), distributed under Apache-2.0. Attribution is retained in [NOTICE](NOTICE).

## Highlights

- A complete two-stage pipeline: supervised fine-tuning followed by controlled GRPO.
- A shared correctness-gated allocation score for adaptive KL and response-length shaping.
- Rule-based format and accuracy rewards without online LLM judging.
- Deterministic evaluation on IntentBench, Daily-Omni, and WorldSense.
- Explicit negative controls for incorrect-length rewards and reversed KL allocation.
- Portable foreground launchers with visible training and evaluation progress.

## Method overview

For every prompt, the policy samples a complete GRPO response group. The trainer derives model-relative hardness and disagreement from rule-based correctness, optionally mixes them with a bounded multimodal input-load proxy, and uses the shared score to:

1. weaken KL regularization for groups requiring more policy freedom;
2. apply a small length penalty only to easy, correct responses;
3. avoid rewarding incorrect responses for becoming longer.

Core experiments use deterministic format and answer-accuracy rewards. Online LLM judging is not part of the controlled protocol. This release supports Qwen2.5-Omni checkpoints only.

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
| GPU | NVIDIA GPU with 48 GB memory |

## Installation

The experiments were developed with Python 3.12, CUDA 12.4, and CUDA-enabled PyTorch 2.5+. Install the PyTorch build suitable for your CUDA runtime first:

```bash
git clone https://github.com/mingzhi114514/CoGA-GRPO.git
cd CoGA-GRPO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The controlled launcher defaults to SDPA, so FlashAttention is not required.

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
bash run_scripts/run_e0_e6.sh E4 full 42
```

Modes are `smoke` (5 steps), `pilot` (50 steps), and `full` (one epoch). Use different GPUs and ports for concurrent jobs.

| Internal ID | Public method name | Dynamic KL | Input load | Safe length shaping | Purpose |
|---|---|---:|---:|---:|---|
| E0 | Standard GRPO | no | no | no | baseline |
| E1 | Dynamic KL only | yes | no | no | component ablation |
| E2 | Safe length shaping | no | no | yes | component ablation |
| E3 | Model-relative allocation | yes | no | yes | allocation ablation |
| E4 | Full shared allocation | yes | yes | yes | proposed method |
| E5 | Incorrect-length reward | yes | yes | yes | negative stress test |
| E6 | Reverse-KL allocation | reversed | yes | yes | negative stress test |

Use semantic names rather than internal IDs in the paper and public result tables.

## Complete evaluation

```bash
BASE_MODEL=/path/to/base-model \
DATA_ROOT=/path/to/benchmarks \
CUDA_VISIBLE_DEVICES=0 \
bash run_scripts/eval_one.sh \
  full_shared_allocation \
  output/controlled_ablation/e4_full_seed42 \
  ib
```

Supported identifiers are `ib`, `daily`, and `world`. CUDA OOM samples are skipped, recorded in the output `oom_skips` list, and included in `final_acc.oom_skipped`. Always report the evaluated and expected denominators; results with skipped samples are marked `complete=false`. Compare models only when effective evaluation settings match.

Report strict accuracy and denominator, token mean/median/P90/standard deviation, lengths split by correctness, missing/malformed/no-EOS/OOM counts, paired confidence intervals, McNemar comparisons, seed, and code revision.

## Validation

Run the lightweight release checks before opening a pull request:

```bash
python -m compileall -q src eval tools tests
python tests/test_release_structure.py
python tools/check_release.py
```

The checker rejects private server paths, credentials, datasets, model artifacts, legacy project naming, and non-English CJK text.

## Limitations

The current protocol targets explicit multiple-choice video reasoning with one multimodal backbone. Output-token reduction is not equivalent to end-to-end latency reduction because video encoding can dominate runtime. Audio adaptation is not claimed when audio input is disabled.

## Citation

The BibTeX entry will be added after the manuscript title, author list, and public paper identifier are finalized.
