#!/usr/bin/env bash
set -euo pipefail

# Controlled single-GPU ablations. Run from any directory.
# Usage:
#   BASE_MODEL=/path/to/model \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash run_scripts/run_e0_e6.sh E3 full 42

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

EXP_ID="${1:?experiment ID required: E0..E6}"
MODE="${2:-smoke}"
SEED="${3:-42}"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the local base-model directory or Hub identifier}"
DATASET_CONFIG="${DATASET_CONFIG:-data_config/stage2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/controlled_ablation}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MIN_PIXELS="${MIN_PIXELS:-75264}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-6}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
MASTER_PORT="${MASTER_PORT:-16666}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
INPUT_COMPLEXITY_MIX="${INPUT_COMPLEXITY_MIX:-0.25}"
LENGTH_REWARD_WEIGHT="${LENGTH_REWARD_WEIGHT:-0.25}"
VERBOSE_DIAGNOSTICS="${VERBOSE_DIAGNOSTICS:-false}"
REWARD_PROTOCOL="rule_based_format_accuracy"

if (( PER_DEVICE_TRAIN_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE must be divisible by NUM_GENERATIONS=$NUM_GENERATIONS" >&2
  exit 2
fi

USE_DYNAMIC_BETA=false
ENABLE_LENGTH_REWARD=false
USE_INTRINSIC_DIFFICULTY=false
REWARD_LONG_INCORRECT=false
BETA_DIRECTION=lower_for_hard

case "$EXP_ID" in
  E0) ;;
  E1) USE_DYNAMIC_BETA=true ;;
  E2) ENABLE_LENGTH_REWARD=true ;;
  E3) USE_DYNAMIC_BETA=true; ENABLE_LENGTH_REWARD=true ;;
  E4) USE_DYNAMIC_BETA=true; ENABLE_LENGTH_REWARD=true; USE_INTRINSIC_DIFFICULTY=true ;;
  E5) USE_DYNAMIC_BETA=true; ENABLE_LENGTH_REWARD=true; USE_INTRINSIC_DIFFICULTY=true; REWARD_LONG_INCORRECT=true ;;
  E6) USE_DYNAMIC_BETA=true; ENABLE_LENGTH_REWARD=true; USE_INTRINSIC_DIFFICULTY=true; BETA_DIRECTION=higher_for_hard ;;
  *) echo "Unknown experiment: $EXP_ID" >&2; exit 2 ;;
esac

case "$MODE" in
  smoke) TRAIN_LIMIT_ARGS=(--max_steps 5 --save_steps 5) ;;
  pilot) TRAIN_LIMIT_ARGS=(--max_steps 50 --save_steps 50) ;;
  full) TRAIN_LIMIT_ARGS=(--num_train_epochs 1 --save_steps 100) ;;
  *) echo "Unknown mode: $MODE (expected smoke, pilot or full)" >&2; exit 2 ;;
esac

RUN_NAME="${EXP_ID,,}_${MODE}_seed${SEED}"
OUT="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "$OUT"
cp "$0" "$OUT/launch_script.sh"

export GRPO_ASSISTANT_PREFILL=$'\n<context>\n'
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

{
  echo "experiment_id=$EXP_ID"
  echo "mode=$MODE"
  echo "seed=$SEED"
  echo "use_dynamic_beta=$USE_DYNAMIC_BETA"
  echo "enable_length_reward=$ENABLE_LENGTH_REWARD"
  echo "use_intrinsic_difficulty=$USE_INTRINSIC_DIFFICULTY"
  echo "reward_long_incorrect=$REWARD_LONG_INCORRECT"
  echo "beta_difficulty_direction=$BETA_DIRECTION"
  echo "num_generations=$NUM_GENERATIONS"
  echo "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "grpo_groups_per_step=$((PER_DEVICE_TRAIN_BATCH_SIZE / NUM_GENERATIONS))"
  echo "gradient_checkpointing=$GRADIENT_CHECKPOINTING"
  echo "input_complexity_mix=$INPUT_COMPLEXITY_MIX"
  echo "length_reward_weight=$LENGTH_REWARD_WEIGHT"
  echo "max_completion_length=$MAX_COMPLETION_LEN"
  echo "reward_protocol=$REWARD_PROTOCOL"
  echo "reward_funcs=format accuracy"
  echo "llm_judge=false"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "project_root=$PROJECT_ROOT"
  echo "base_model=$BASE_MODEL"
  echo "dataset_config=$DATASET_CONFIG"
  echo "python_bin=$PYTHON_BIN"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo uncommitted)"
  fi
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())'
} | tee "$OUT/run_manifest.txt"

# One process per experiment preserves complete GRPO groups. Use the two GPUs
# for two independent experiments rather than splitting one G=6 group over DDP.
"$TORCHRUN_BIN" --nproc_per_node 1 --master_port "$MASTER_PORT" \
  src/open_r1/grpo_qwenomni.py \
  --deepspeed run_scripts/zero2_offload.json \
  --output_dir "$OUT" \
  --model_name_or_path "$BASE_MODEL" \
  --use_peft true \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --dataset_name "$DATASET_CONFIG" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LEN" \
  --num_generations "$NUM_GENERATIONS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --beta 0.04 \
  --use_dynamic_beta "$USE_DYNAMIC_BETA" \
  --beta_min 0.0001 \
  --beta_max 0.04 \
  --beta_mix_lambda 0.8 \
  --input_complexity_mix "$INPUT_COMPLEXITY_MIX" \
  --beta_difficulty_direction "$BETA_DIRECTION" \
  --use_uncertainty_in_beta true \
  --enable_length_reward "$ENABLE_LENGTH_REWARD" \
  --length_reward_weight "$LENGTH_REWARD_WEIGHT" \
  --length_reward_margin 0.05 \
  --length_reward_alpha 2.0 \
  --reward_long_incorrect "$REWARD_LONG_INCORRECT" \
  --difficulty_percentile 85 \
  --difficulty_window_size 32 \
  --use_intrinsic_difficulty "$USE_INTRINSIC_DIFFICULTY" \
  --require_modality_complexity true \
  --strict_group_generation true \
  --complexity_weight_image 0.25 \
  --complexity_weight_audio 0.05 \
  --complexity_weight_video 0.55 \
  --complexity_weight_text 0.15 \
  --freeze_vision_modules true \
  --reward_funcs format accuracy \
  --max_pixels "$MAX_PIXELS" \
  --min_pixels "$MIN_PIXELS" \
  --video_max_frames "$VIDEO_MAX_FRAMES" \
  --video_reader_fps 1.0 \
  --use_audio_in_video false \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --attn_implementation sdpa \
  --bf16 \
  --torch_dtype bfloat16 \
  --seed "$SEED" \
  --data_seed "$SEED" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --logging_steps 1 \
  --log_completions "$VERBOSE_DIAGNOSTICS" \
  --log_reward_to_console "$VERBOSE_DIAGNOSTICS" \
  --log_length_reward_details "$VERBOSE_DIAGNOSTICS" \
  --log_length_reward_groups 3 \
  --scale_rewards false \
  --report_to none \
  --save_only_model false \
  --run_name "$RUN_NAME" \
  "${TRAIN_LIMIT_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"

exit "${PIPESTATUS[0]}"
