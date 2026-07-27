#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${WARM_START_FROM_CHECKPOINT:?Set WARM_START_FROM_CHECKPOINT to V14 checkpoint-416}"

unset RESUME_FROM_CHECKPOINT
export WARM_START_MODE=residual_hybrid_w8_ablation
export MEMORY_FUSION_PLACEMENT=post_attention_residual_hybrid
export MEMORY_FUSION_RESIDUAL_SCALE=0.01
export MEMORY_FUSION_RESIDUAL_SCALE_MAX=0.02
export MEMORY_LOSS_MODE=content_contrast_ce
export MEMORY_CONTRAST_WEIGHT=0.25
export MEMORY_MARGIN=0.5
export MEMORY_REPRESENTATION_WEIGHT=0.1
export MEMORY_REPRESENTATION_MARGIN=0.1
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=32

exec "$SCRIPT_DIR/train_all42_gated_memory.sh"
