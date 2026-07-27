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
RESUME_MODE="${RESUME_MODE:-extend}"
WARM_START_FROM_CHECKPOINT="${WARM_START_FROM_CHECKPOINT:-}"
WARM_START_MODE="${WARM_START_MODE:-residual_hybrid_w8_ablation}"
MEMORY_FUSION_PLACEMENT="${MEMORY_FUSION_PLACEMENT:-attention_output}"
MEMORY_FUSION_RESIDUAL_SCALE="${MEMORY_FUSION_RESIDUAL_SCALE:-1.0}"
MEMORY_FUSION_RESIDUAL_SCALE_MAX="${MEMORY_FUSION_RESIDUAL_SCALE_MAX:-1.0}"
MEMORY_FUSION_MODE="${MEMORY_FUSION_MODE:-content_gated_add}"
MEMORY_FUSION_GATE_INIT="${MEMORY_FUSION_GATE_INIT:-0.1}"
MEMORY_LOSS_MODE="${MEMORY_LOSS_MODE:-context_dropout_ce}"
MEMORY_CONTRAST_WEIGHT="${MEMORY_CONTRAST_WEIGHT:-0}"
MEMORY_MARGIN="${MEMORY_MARGIN:-0.1}"
MEMORY_REPRESENTATION_WEIGHT="${MEMORY_REPRESENTATION_WEIGHT:-0}"
MEMORY_REPRESENTATION_MARGIN="${MEMORY_REPRESENTATION_MARGIN:-0.1}"
TARGET_LAYERS="$(seq -s, 0 41)"

if [[ -e "$OUTPUT_DIR" ]]; then
  if [[ -z "$RESUME_FROM_CHECKPOINT" || "$RESUME_MODE" != "exact" ]]; then
    echo "OUTPUT_DIR may already exist only for an exact checkpoint resume: $OUTPUT_DIR" >&2
    exit 1
  fi
fi

resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" && -n "$WARM_START_FROM_CHECKPOINT" ]]; then
  echo "RESUME_FROM_CHECKPOINT and WARM_START_FROM_CHECKPOINT are mutually exclusive" >&2
  exit 1
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  resume_args=(
    --resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"
    --resume-mode "$RESUME_MODE"
  )
fi

warm_start_args=()
if [[ -n "$WARM_START_FROM_CHECKPOINT" ]]; then
  warm_start_args=(
    --warm-start-from-checkpoint "$WARM_START_FROM_CHECKPOINT"
    --warm-start-mode "$WARM_START_MODE"
  )
fi

cd -- "$REPO"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
  -m deltamem.train.delta_sft \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  "${resume_args[@]}" \
  "${warm_start_args[@]}" \
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
  --memory-fusion-mode "$MEMORY_FUSION_MODE" \
  --memory-fusion-gate-init "$MEMORY_FUSION_GATE_INIT" \
  --memory-fusion-placement "$MEMORY_FUSION_PLACEMENT" \
  --memory-fusion-residual-scale "$MEMORY_FUSION_RESIDUAL_SCALE" \
  --memory-fusion-residual-scale-max "$MEMORY_FUSION_RESIDUAL_SCALE_MAX" \
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
  --memory-loss-mode "$MEMORY_LOSS_MODE" \
  --memory-contrast-weight "$MEMORY_CONTRAST_WEIGHT" \
  --memory-margin "$MEMORY_MARGIN" \
  --memory-representation-weight "$MEMORY_REPRESENTATION_WEIGHT" \
  --memory-representation-margin "$MEMORY_REPRESENTATION_MARGIN" \
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
