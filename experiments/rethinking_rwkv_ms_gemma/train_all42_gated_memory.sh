#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

: "${PYTHON_BIN:?Set PYTHON_BIN to the Python executable with Delta-Mem dependencies}"
: "${MODEL_PATH:?Set MODEL_PATH to the local Gemma model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the prepared 32-row JSONL dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new training output directory}"
: "${HF_CACHE_DIR:?Set HF_CACHE_DIR to the Hugging Face datasets cache}"
: "${TOKENIZED_DATASET_ROOT:?Set TOKENIZED_DATASET_ROOT to the tokenized-cache root}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
MAX_STEPS="${MAX_STEPS:-128}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TARGET_LAYERS="$(seq -s, 0 41)"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR must not already exist: $OUTPUT_DIR" >&2
  exit 1
fi

resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  resume_args=(
    --resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"
    --resume-mode extend
  )
fi

cd -- "$REPO"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
  -m deltamem.train.delta_sft \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  "${resume_args[@]}" \
  --hf-cache-dir "$HF_CACHE_DIR" \
  --tokenized-dataset-root "$TOKENIZED_DATASET_ROOT" \
  --tokenized-cache \
  --device cuda:0 \
  --dtype bfloat16 \
  --bf16 \
  --attn-implementation sdpa \
  --memory-backend rwkv_ms \
  --rwkv-ms-num-states 4 \
  --rwkv-ms-chunk-size 128 \
  --rwkv-ms-boundary-mode fixed_chunk \
  --rwkv-ms-erase-gate 1.0 \
  --rwkv-ms-read-top-k 0 \
  --rwkv-ms-output-init-scale 0.02 \
  --rwkv-ms-semantics-version 2 \
  --rank 4 \
  --alpha 8 \
  --num-state-heads 1 \
  --beta-bias-init 0.0 \
  --couple-lambda \
  --state-update-mode standard \
  --output-init base_slice_fixed \
  --base-slice-ref-width 8 \
  --delta-heads o \
  --no-delta-o-rmsnorm \
  --delta-o-rmsnorm-eps 1e-6 \
  --memory-fusion-mode content_gated_add \
  --memory-fusion-gate-init 0.1 \
  --trainable-delta-scale \
  --delta-scale-init 0.1 \
  --delta-scale-max 0.5 \
  --delta-scale-granularity head \
  --delta-scale-parameterization alpha_over_rank \
  --online-gain 0.2 \
  --target-layers "$TARGET_LAYERS" \
  --memory-readout-mode delta \
  --memory-write-source learned_hidden \
  --memory-write-granularity token \
  --max-length 256 \
  --training-mode episode \
  --episode-recent-messages 1 \
  --max-write-length 512 \
  --no-episode-read-write-enabled \
  --memory-loss-mode context_dropout_ce \
  --memory-contrast-weight 0 \
  --memory-kl-weight 0 \
  --memory-causal-weight 0 \
  --memory-anchor-weight 0 \
  --memory-recover-weight 0 \
  --memory-dropout-no-memory-prob 0 \
  --memory-dropout-state-only-prob 0 \
  --memory-base-kl-weight 0 \
  --context-ablation-mode mixed \
  --context-ablation-no-state-prob 0.2 \
  --context-ablation-state-only-prob 0.2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 0.001 \
  --seed 42 \
  --data-seed 42 \
  --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.0625 \
  --weight-decay 0 \
  --optim adamw_torch_fused \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --logging-steps 1 \
  --save-steps 16 \
  --eval-steps 1000 \
  --validation-split-ratio 0 \
  --save-total-limit 8 \
  --no-load-best-model-at-end \
  --dataset-num-proc 1 \
  --dataloader-num-workers 0 \
  --frozen-mlp-checkpointing \
  --tf32 \
  --write-sparsity-weight 0 \
  --write-sparsity-target 0.05 \
  --log-delta-debug-stats \
  --assistant-loss-mode final_assistant_only \
  --rankwise-gates
