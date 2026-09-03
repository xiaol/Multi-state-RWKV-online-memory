#!/usr/bin/env bash
# Write / clear / read on frozen Qwen3-1.7B: single-state vs multi-state at matched 16 slots per layer.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-/root/x/models/Qwen3-1.7B}
COMMON="--model $MODEL --dataset synthetic --facts 8 --entities 4 --eval-facts 4,16 --layers auto --mem-dim 128 --steps 3000 --batch-size 16 --lr 1e-3 --eval-every 500 --eval-rows 256 --eval-batch-size 32 --save-adapter"
run() { # name gpu extra...
  local name=$1 gpu=$2; shift 2
  mkdir -p runs/$name
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train.py $COMMON --out runs/$name "$@" > runs/$name/stdout.log 2>&1 &
  echo "$name pid $! gpu $gpu"
}
run qwen1p7b_k8_single16  2 --n-states 1 --slots-per-state 16 --routing single
run qwen1p7b_k8_chunk4x4  2 --n-states 4 --slots-per-state 4  --routing chunk
run qwen1p7b_k8_cosine4x4 3 --n-states 4 --slots-per-state 4  --routing cosine
