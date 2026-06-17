#!/usr/bin/env bash
set -euo pipefail

# Train two matched adapters that differ only in the online memory backend:
#   1. delta_rule: original delta-Mem associative update
#   2. rwkv_ms: multi-state RWKV-style update with fixed chunk routing
#
# This script intentionally reuses the repo trainer/evaluator entry points.
# Override paths from the shell before running.

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29571}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/models/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/models/rwkv_ms_delta_rule_comparison}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/outputs/rwkv_ms_delta_rule_comparison}"

ADAPTER_RANK="${ADAPTER_RANK:-8}"
ALPHA="${ALPHA:-16.0}"
DELTA_HEADS="${DELTA_HEADS:-q,o}"
NUM_STATE_HEADS="${NUM_STATE_HEADS:-1}"
MEMORY_WRITE_GRANULARITY="${MEMORY_WRITE_GRANULARITY:-token}"
RWKV_MS_NUM_STATES="${RWKV_MS_NUM_STATES:-4}"
RWKV_MS_CHUNK_SIZE="${RWKV_MS_CHUNK_SIZE:-1024}"

COMMON_TRAIN_ARGS=(
  --model-path "${BASE_MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --training-mode episode
  --assistant-loss-mode final_assistant_only
  --episode-recent-messages 1
  --max-length 512
  --max-write-length 8192
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 4
  --learning-rate 2e-4
  --lr-scheduler-type cosine
  --warmup-ratio 0.10
  --weight-decay 0.0
  --num-train-epochs 1.0
  --logging-steps 1
  --save-steps 200
  --rank "${ADAPTER_RANK}"
  --alpha "${ALPHA}"
  --num-state-heads "${NUM_STATE_HEADS}"
  --beta-bias-init -1.5
  --couple-lambda
  --state-update-mode standard
  --output-init base_slice_fixed
  --base-slice-ref-width 8
  --delta-heads "${DELTA_HEADS}"
  --online-gain 0.05
  --rankwise-gates
  --target-layers off
  --memory-readout-mode delta
  --memory-write-source learned_hidden
  --memory-write-granularity "${MEMORY_WRITE_GRANULARITY}"
  --memory-contrast-weight 1.0
  --memory-kl-weight 0.02
  --memory-margin 0.01
  --memory-causal-weight 1.0
  --memory-anchor-weight 1.0
  --memory-anchor-margin 0.005
  --memory-recover-weight 0.25
  --memory-need-floor 0.15
  --memory-dropout-state-only-prob 0.2
)

train_backend() {
  local backend="$1"
  local port="$2"
  local output_dir="${OUTPUT_ROOT}/${backend}"
  local extra_args=()
  if [[ "${backend}" == "rwkv_ms" ]]; then
    extra_args=(
      --rwkv-ms-num-states "${RWKV_MS_NUM_STATES}"
      --rwkv-ms-chunk-size "${RWKV_MS_CHUNK_SIZE}"
      --rwkv-ms-boundary-mode fixed_chunk
      --rwkv-ms-erase-gate 1.0
      --rwkv-ms-read-top-k 0
    )
  fi

  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${port}" \
    -m deltamem.train.delta_sft_experimental \
    "${COMMON_TRAIN_ARGS[@]}" \
    --output-dir "${output_dir}" \
    --memory-backend "${backend}" \
    "${extra_args[@]}"
}

mkdir -p "${OUTPUT_ROOT}" "${RESULTS_ROOT}"
train_backend delta_rule "${BASE_MASTER_PORT}"
train_backend rwkv_ms "$((BASE_MASTER_PORT + 1))"

cat <<EOF
Matched adapters trained under: ${OUTPUT_ROOT}

Run the existing benchmark suite against each checkpoint, for example:

BASE_MODEL_PATH=${BASE_MODEL_PATH} \\
TSW_ADAPTER_DIR=${OUTPUT_ROOT}/delta_rule/trainer/checkpoint-<step> \\
SUITE_ROOT=${RESULTS_ROOT}/delta_rule \\
BENCHMARK_VARIANTS_STRING=TSW_rank8_qasper_write8192 \\
bash scripts/run_qasper_multimodel_write8192_benchmark_suite.sh

BASE_MODEL_PATH=${BASE_MODEL_PATH} \\
TSW_ADAPTER_DIR=${OUTPUT_ROOT}/rwkv_ms/trainer/checkpoint-<step> \\
SUITE_ROOT=${RESULTS_ROOT}/rwkv_ms \\
BENCHMARK_VARIANTS_STRING=TSW_rank8_qasper_write8192 \\
bash scripts/run_qasper_multimodel_write8192_benchmark_suite.sh
EOF
