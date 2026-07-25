#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${RESUME_FROM_CHECKPOINT:?Set RESUME_FROM_CHECKPOINT to the completed checkpoint-128}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh post-attention-norm ablation directory}"

export RESUME_MODE=placement_ablation
export MEMORY_FUSION_PLACEMENT=post_attention_norm
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
export MAX_STEPS="${MAX_STEPS:-160}"

exec "$SCRIPT_DIR/train_all42_gated_memory.sh"
