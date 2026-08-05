#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/rwkv_py312/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/local_artifacts/synthetic_associative_retrieval_canary_v2}"
TRAIN_FILE="${DATA_ROOT}/train.jsonl"
SOURCE_MANIFEST="${DATA_ROOT}/source_manifest.json"
GATE0_RECEIPT="${DATA_ROOT}/gate0_receipt.json"
PREFLIGHT_RECEIPT="${DATA_ROOT}/projected_kv_preflight_receipt.json"
RUN_NAME="${RUN_NAME:-synthetic_associative_projected_kv_s2_k32_t16_u1_b4_lr2e4_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/local_artifacts/synthetic_associative_retrieval_runs/${RUN_NAME}}"
INITIAL_ADAPTER_DIR="${OUTPUT_DIR}/initial_adapter"
HF_CACHE_DIR="${HF_CACHE_DIR:-/root/X/.cache/hf/runtime}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_STEPS="${MAX_STEPS:-64}"
SAVE_STEPS="${SAVE_STEPS:-8}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ -x "${PYTHON_BIN}" ]] || fail "python_missing path=${PYTHON_BIN}"
[[ -f "${MODEL_PATH}/model.safetensors" ]] || fail "model_missing path=${MODEL_PATH}"
[[ -f "${TRAIN_FILE}" ]] || fail "train_file_missing path=${TRAIN_FILE}"
[[ -f "${SOURCE_MANIFEST}" ]] || fail "source_manifest_missing path=${SOURCE_MANIFEST}"
[[ -f "${GATE0_RECEIPT}" ]] || fail "gate0_receipt_missing path=${GATE0_RECEIPT}"
[[ -f "${PREFLIGHT_RECEIPT}" ]] || fail "preflight_receipt_missing path=${PREFLIGHT_RECEIPT}"
[[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]] || fail "CUDA_DEVICE must be a non-negative integer"
[[ "${MAX_STEPS}" == "64" ]] || fail "The locked associative run requires MAX_STEPS=64"
[[ "${SAVE_STEPS}" == "8" ]] || fail "The locked associative run requires SAVE_STEPS=8"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "fresh_output_required path=${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${HF_HOME:-/root/X/.cache/hf/runtime/home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/X/.cache/hf/runtime/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/X/.cache/hf/runtime/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p -- "${HF_CACHE_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_synthetic_associative_retrieval_gate0.py" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --model-path "${MODEL_PATH}" \
  --validate-receipt "${GATE0_RECEIPT}" >/dev/null

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_synthetic_associative_retrieval_preflight.py" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --gate0-receipt "${GATE0_RECEIPT}" \
  --model-path "${MODEL_PATH}" \
  --validate-receipt "${PREFLIGHT_RECEIPT}" >/dev/null

SOURCE_MANIFEST_SHA256="$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')"
TARGET_LAYERS="$(seq -s, 0 41)"

printf 'Launching associative projected-KV canary: output=%s steps=%s gpu=%s slots=2 key_dim=32 temperature=16 update_threshold=1 batch=4 lr=2e-4 seed=42\n' \
  "${OUTPUT_DIR}" "${MAX_STEPS}" "${CUDA_DEVICE}"

exec "${PYTHON_BIN}" -m deltamem.train.delta_sft \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --initial-adapter-output-dir "${INITIAL_ADAPTER_DIR}" \
  --hf-cache-dir "${HF_CACHE_DIR}" \
  --no-tokenized-cache \
  --device cuda:0 \
  --dtype bfloat16 \
  --bf16 \
  --attn-implementation sdpa \
  --memory-backend rwkv_ms \
  --rwkv-ms-num-states 2 \
  --rwkv-ms-chunk-size 128 \
  --rwkv-ms-boundary-mode fixed_chunk \
  --rwkv-ms-write-mode recurrent \
  --rwkv-ms-erase-gate 1.0 \
  --rwkv-ms-read-top-k 0 \
  --rwkv-ms-output-init-scale 0.02 \
  --rwkv-ms-semantics-version 2 \
  --projected-kv-key-dim 32 \
  --projected-kv-temperature 16.0 \
  --projected-kv-update-cosine-threshold 1.0 \
  --rank 4 \
  --alpha 8 \
  --num-state-heads 1 \
  --beta-bias-init 0.0 \
  --couple-lambda \
  --state-update-mode standard \
  --output-init base_slice_fixed \
  --base-slice-ref-width 8 \
  --delta-heads q,o \
  --no-delta-o-rmsnorm \
  --memory-fusion-mode add \
  --memory-fusion-placement attention_output \
  --memory-fusion-residual-scale 1.0 \
  --memory-fusion-residual-scale-max 1.0 \
  --trainable-delta-scale \
  --delta-scale-init 0.1 \
  --delta-scale-max 0.5 \
  --delta-scale-granularity head \
  --delta-scale-parameterization alpha_over_rank \
  --online-gain 0.2 \
  --target-layers "${TARGET_LAYERS}" \
  --memory-readout-mode projected_kv_slots \
  --memory-write-source learned_hidden \
  --memory-write-granularity message_mean \
  --training-mode episode \
  --assistant-loss-mode final_assistant_only \
  --episode-recent-messages 1 \
  --max-length 128 \
  --max-write-length 128 \
  --no-episode-read-write-enabled \
  --memory-loss-mode scene_state_identity_ce \
  --scene-state-identity-margin 0.5 \
  --scene-state-source-manifest "${SOURCE_MANIFEST}" \
  --expected-scene-state-source-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --scene-boundary-payload-ce-weight 0 \
  --memory-dropout-no-memory-prob 0 \
  --memory-dropout-state-only-prob 0 \
  --memory-base-kl-weight 0 \
  --memory-contrast-weight 0 \
  --memory-representation-weight 0 \
  --memory-kl-weight 0 \
  --memory-causal-weight 0 \
  --memory-anchor-weight 0 \
  --memory-recover-weight 0 \
  --write-sparsity-weight 0 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --lr-scheduler-type constant \
  --warmup-ratio 0 \
  --warmup-steps 0 \
  --weight-decay 0 \
  --max-grad-norm 1.0 \
  --optim adamw_torch_fused \
  --num-train-epochs 1 \
  --max-steps "${MAX_STEPS}" \
  --logging-steps 1 \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 8 \
  --eval-steps 1000 \
  --validation-split-ratio 0 \
  --no-load-best-model-at-end \
  --dataset-num-proc 1 \
  --dataloader-num-workers 0 \
  --frozen-mlp-activation-checkpointing \
  --seed 42 \
  --data-seed 42 \
  --train-sampler-seed 42 \
  --tf32 \
  --log-delta-debug-stats \
  --rankwise-gates
