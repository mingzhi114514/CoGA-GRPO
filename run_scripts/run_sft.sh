#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${BASE_MODEL:?Set BASE_MODEL to a Qwen2.5-Omni checkpoint}"

DATASET_CONFIG="${DATASET_CONFIG:-data_config/stage1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
RUN_NAME="${RUN_NAME:-coga-grpo-sft}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-16666}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-run_scripts/zero2_offload.json}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"

mkdir -p "$OUTPUT_DIR"

echo "[CoGA-GRPO SFT] project_root=$PROJECT_ROOT"
echo "[CoGA-GRPO SFT] base_model=$BASE_MODEL"
echo "[CoGA-GRPO SFT] dataset=$DATASET_CONFIG"
echo "[CoGA-GRPO SFT] output=$OUTPUT_DIR"

"$TORCHRUN_BIN" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  src/open_r1/sft.py \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_name "$DATASET_CONFIG" \
  --freeze_vision_modules true \
  --use_audio_in_video true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --learning_rate 2.0e-5 \
  --bf16 \
  --torch_dtype bfloat16 \
  --data_seed 42 \
  --seed 42 \
  --report_to none \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --run_name "$RUN_NAME" \
  --save_steps "$SAVE_STEPS" \
  --log_level info \
  --save_only_model true 2>&1 | tee "$OUTPUT_DIR/train.log"
