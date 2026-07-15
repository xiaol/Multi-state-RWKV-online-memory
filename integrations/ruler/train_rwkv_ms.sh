#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/aiuser/X/.venv-wkvm/bin/python}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-/home/aiuser/X/.rwkv-memory-deps}"
MODEL_PATH="${MODEL_PATH:-/home/aiuser/X/models/gemma-4-E4B-it}"
TRAIN_FILE="${TRAIN_FILE:-/home/aiuser/X/results/ruler-gemma4/train/episodes-v1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/aiuser/X/results/ruler-gemma4/train/rwkv-ms-v1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TOKENIZED_DATASET_ROOT="${TOKENIZED_DATASET_ROOT:-/home/aiuser/X/results/ruler-gemma4/tokenized-cache}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29641}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SEED="${SEED:-3407}"

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#CUDA_DEVICE_LIST[@]}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training file is missing: ${TRAIN_FILE}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Base model is missing: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  && [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Output directory must be absent or empty unless RESUME_FROM_CHECKPOINT is set: ${OUTPUT_DIR}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${DEPENDENCY_ROOT}:${ROOT}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" "${ROOT}/integrations/ruler/verify_training_data.py" \
  --training-file "${TRAIN_FILE}"

mkdir -p "${OUTPUT_DIR}" "${TOKENIZED_DATASET_ROOT}"

TRAIN_ARGS=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --tokenized-dataset-root "${TOKENIZED_DATASET_ROOT}"
  --dtype bfloat16
  --bf16
  --attn-implementation sdpa
  --memory-backend rwkv_ms
  --rwkv-ms-num-states 4
  --rwkv-ms-chunk-size 1024
  --rwkv-ms-boundary-mode fixed_chunk
  --rwkv-ms-erase-gate 1.0
  --rwkv-ms-read-top-k 0
  --rank 8
  --alpha 16
  --num-state-heads 1
  --beta-bias-init 0.0
  --couple-lambda
  --state-update-mode standard
  --output-init base_slice_fixed
  --base-slice-ref-width 8
  --delta-heads q,o
  --online-gain 0.2
  --rankwise-gates
  --target-layers 0,1,2,3,4,5
  --training-mode episode
  --assistant-loss-mode final_assistant_only
  --episode-recent-messages 1
  --max-length 512
  --max-write-length 8192
  --memory-readout-mode delta
  --memory-write-source learned_hidden
  --memory-write-granularity token
  --memory-contrast-weight 1.0
  --memory-kl-weight 0.02
  --memory-margin 0.01
  --memory-causal-weight 1.0
  --memory-anchor-weight 1.0
  --memory-anchor-margin 0.005
  --memory-recover-weight 0.25
  --memory-need-floor 0.15
  --memory-dropout-state-only-prob 0.2
  --per-device-train-batch-size 1
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --lr-scheduler-type cosine
  --warmup-ratio 0.1
  --weight-decay 0.0
  --optim adamw_torch_fused
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-steps -1
  --logging-steps 1
  --save-steps "${SAVE_STEPS}"
  --dataset-num-proc 8
  --dataloader-num-workers 2
  --seed "${SEED}"
  --data-seed "${SEED}"
  --tf32
  --log-delta-debug-stats
)

TEE_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
  TEE_ARGS=(-a)
fi

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -m deltamem.train.delta_sft_experimental \
  "${TRAIN_ARGS[@]}" 2>&1 | tee "${TEE_ARGS[@]}" "${OUTPUT_DIR}/train.log"
