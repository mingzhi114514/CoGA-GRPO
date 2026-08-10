#!/usr/bin/env bash
set -euo pipefail

# Deterministic single-GPU evaluation.
# Usage:
#   BASE_MODEL=/path/to/model DATA_ROOT=/path/to/benchmarks \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash run_scripts/eval_one.sh full_shared_allocation /path/to/adapter ib

RUN_LABEL="${1:?run label required}"
ADAPTER_PATH="${2:?adapter path required}"
DATASET="${3:-ib}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the local base-model directory or Hub identifier}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing benchmark folders}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-output/unified_eval}"

if [[ ! -e "$ADAPTER_PATH" ]]; then
  echo "Adapter path does not exist: $ADAPTER_PATH" >&2
  exit 2
fi

mkdir -p "$EVAL_OUTPUT_ROOT/results" "$EVAL_OUTPUT_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRPO_ASSISTANT_PREFILL=$'\n<context>\n'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"$PYTHON_BIN" eval/evaluate_coga.py \
  --model-path "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output-dir "$EVAL_OUTPUT_ROOT/results" \
  --file-name "$RUN_LABEL" \
  --max-samples 0 \
  --eval-seed 42 \
  --attn-implementation auto \
  --device-map auto \
  --max-new-tokens 1024 \
  --no-do-sample \
  --repetition-penalty 1.0 \
  --max-input-tokens 0 \
  --assistant-prefill "$GRPO_ASSISTANT_PREFILL" \
  --video-max-frames 6 \
  --video-reader-fps 1.0 \
  --max-pixels 100352 \
  --min-pixels 75264 \
  --num-workers 0 \
  --stop-after-first-answer \
  --truncate-at-first-answer \
  --qa-output \
  --no-strict-completeness \
  --skip-oom \
  --no-skip-missing-video-samples \
  2>&1 | tee "$EVAL_OUTPUT_ROOT/logs/${DATASET}_${RUN_LABEL}.log"

exit "${PIPESTATUS[0]}"
