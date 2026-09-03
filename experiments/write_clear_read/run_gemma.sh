#!/usr/bin/env bash
# Write / clear / read on frozen Gemma4 E4B, memory on the seven full-attention layers.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-/root/x/models/gemma-4-E4B-it}
COMMON="--model $MODEL --dataset synthetic --facts 8 --entities 4 --eval-facts 4,16 --layers full --mem-dim 128 --steps 3000 --batch-size 16 --lr 1e-3 --eval-every 500 --eval-rows 256 --eval-batch-size 16 --save-adapter"
run() { local name=$1 gpu=$2; shift 2; mkdir -p runs/$name
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train.py $COMMON --out runs/$name "$@" > runs/$name/stdout.log 2>&1 &
  echo "$name pid $! gpu $gpu"; }
run gemma4e4b_k8_single16 ${GPU:-0} --n-states 1 --slots-per-state 16 --routing single
