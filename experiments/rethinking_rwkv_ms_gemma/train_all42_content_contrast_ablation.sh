#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${RESUME_FROM_CHECKPOINT:?Set RESUME_FROM_CHECKPOINT to the source checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh content-contrast ablation directory}"

export RESUME_MODE="${RESUME_MODE:-objective_ablation}"
export MEMORY_LOSS_MODE=content_contrast_ce
export MEMORY_CONTRAST_WEIGHT="${MEMORY_CONTRAST_WEIGHT:-0.25}"
export MEMORY_MARGIN="${MEMORY_MARGIN:-0.5}"
export MEMORY_REPRESENTATION_WEIGHT="${MEMORY_REPRESENTATION_WEIGHT:-0.1}"
export MEMORY_REPRESENTATION_MARGIN="${MEMORY_REPRESENTATION_MARGIN:-0.1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-13}"
export MAX_STEPS="${MAX_STEPS:-416}"

exec "$SCRIPT_DIR/train_all42_gated_memory.sh"
