#!/usr/bin/env bash
set -euo pipefail

# Production recipe for the second novel-memory run. Every tuning value can be
# overridden from the environment, but the data checksum and trainer feature
# checks stay enabled so an old trainer or changed corpus cannot start silently.

REPO="${REPO:-/home/xiaol/X/Multi-state-RWKV-online-memory}"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it}"
TRAIN_FILE="${TRAIN_FILE:-/run/media/xiaol/B214449214445C0B/delta_mem_data/novel_agent_memory/novel_memory_suffix112_dedup.jsonl}"
TRAIN_SHA256="${TRAIN_SHA256:-e196ba27f7529ff0e5d15c4d36dc1ed1317fea46a47717699488f4053af92f94}"

RUN_ROOT="${RUN_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory}"
RUN_NAME="${RUN_NAME:-v2_l0_3_r8_read256_write1024_preserve}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${RUN_ROOT}/${RUN_NAME}.log}"
STATUS_FILE="${STATUS_FILE:-${RUN_ROOT}/${RUN_NAME}.status}"
CACHE_ROOT="${CACHE_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_cache}"
TOKENIZED_DATASET_ROOT="${TOKENIZED_DATASET_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_tokenized/novel_agent_memory/v2_read256_write1024_recent1}"

TARGET_LAYERS="${TARGET_LAYERS:-0,1,2,3}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-8}"
NUM_STATES="${NUM_STATES:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
BETA_BIAS_INIT="${BETA_BIAS_INIT:--1.5}"
ONLINE_GAIN="${ONLINE_GAIN:-0.05}"
DELTA_SCALE_INIT="${DELTA_SCALE_INIT:-0.10}"
DELTA_SCALE_MAX="${DELTA_SCALE_MAX:-0.50}"

MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_WRITE_LENGTH="${MAX_WRITE_LENGTH:-1024}"
EPISODE_RECENT_MESSAGES="${EPISODE_RECENT_MESSAGES:-1}"
EPISODE_READ_WRITE_ENABLED="${EPISODE_READ_WRITE_ENABLED:-1}"
MEMORY_STATE_ONLY_PROB="${MEMORY_STATE_ONLY_PROB:-0.20}"
MEMORY_NO_MEMORY_PROB="${MEMORY_NO_MEMORY_PROB:-0.0}"
MEMORY_BASE_KL_WEIGHT="${MEMORY_BASE_KL_WEIGHT:-0.50}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
VALIDATION_SPLIT_RATIO="${VALIDATION_SPLIT_RATIO:-0.01}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-auto}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

timestamp() {
  date --iso-8601=seconds
}

record_status() {
  mkdir -p "$(dirname "${STATUS_FILE}")"
  printf 'updated_at=%s\nstate=%s\n%s\n' "$(timestamp)" "$1" "${2:-}" >"${STATUS_FILE}"
}

fail() {
  record_status failed "reason=$1"
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable path=${PYTHON_BIN}"
[[ -d "${REPO}/deltamem" ]] || fail "repository_missing path=${REPO}"
[[ -d "${MODEL_PATH}" ]] || fail "model_missing path=${MODEL_PATH}"
[[ -f "${TRAIN_FILE}" ]] || fail "training_file_missing path=${TRAIN_FILE}"

actual_sha256="$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')"
[[ "${actual_sha256}" == "${TRAIN_SHA256}" ]] \
  || fail "dataset_checksum_mismatch expected=${TRAIN_SHA256} actual=${actual_sha256}"

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

# These flags are the minimum corrected trainer contract. Refuse to fall back
# to the v1 behavior, where preservation/validation flags were unavailable.
trainer_help="$("${PYTHON_BIN}" -m deltamem.train.delta_sft --help)"
required_flags=(
  --memory-loss-mode
  --memory-dropout-no-memory-prob
  --memory-dropout-state-only-prob
  --memory-base-kl-weight
  --episode-read-write-enabled
  --validation-split-ratio
  --eval-steps
  --save-total-limit
  --load-best-model-at-end
)
for required_flag in "${required_flags[@]}"; do
  if ! rg --quiet --fixed-strings -- "${required_flag}" <<<"${trainer_help}"; then
    fail "trainer_missing_required_flag flag=${required_flag}"
  fi
done

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [[ -s "${OUTPUT_DIR}/training_summary.json" \
     && -s "${OUTPUT_DIR}/delta_mem_adapter.pt" \
     && -s "${OUTPUT_DIR}/delta_mem_config.json" ]]; then
  record_status complete "output=${OUTPUT_DIR}"
  printf 'Training output is already complete: %s\n' "${OUTPUT_DIR}"
  exit 0
elif compgen -G "${OUTPUT_DIR}/trainer/checkpoint-*" >/dev/null; then
  resume_args=(--resume-from-checkpoint latest)
elif [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  fail "output_nonempty_without_checkpoint path=${OUTPUT_DIR}"
fi

load_best_args=()
case "${LOAD_BEST_MODEL_AT_END}" in
  auto)
    if awk -v ratio="${VALIDATION_SPLIT_RATIO}" 'BEGIN { exit !(ratio > 0) }'; then
      load_best_args=(--load-best-model-at-end)
    fi
    ;;
  0)
    ;;
  1)
    if ! awk -v ratio="${VALIDATION_SPLIT_RATIO}" 'BEGIN { exit !(ratio > 0) }'; then
      fail "load_best_model_at_end_requires_validation"
    fi
    load_best_args=(--load-best-model-at-end)
    ;;
  *)
    fail "invalid_load_best_model_at_end value=${LOAD_BEST_MODEL_AT_END} expected=auto,0,1"
    ;;
esac

episode_read_write_args=()
case "${EPISODE_READ_WRITE_ENABLED}" in
  0)
    episode_read_write_args=(--no-episode-read-write-enabled)
    ;;
  1)
    episode_read_write_args=(--episode-read-write-enabled)
    ;;
  *)
    fail "invalid_episode_read_write_enabled value=${EPISODE_READ_WRITE_ENABLED} expected=0,1"
    ;;
esac

mkdir -p \
  "${OUTPUT_DIR}" \
  "$(dirname "${LOG_FILE}")" \
  "${HF_HOME}/datasets" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${TOKENIZED_DATASET_ROOT}"

train_args=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --hf-cache-dir "${HF_HOME}/datasets"
  --tokenized-dataset-root "${TOKENIZED_DATASET_ROOT}"
  --device cuda:0
  --dtype bfloat16
  --bf16
  --attn-implementation sdpa
  --memory-backend rwkv_ms
  --rwkv-ms-num-states "${NUM_STATES}"
  --rwkv-ms-chunk-size "${CHUNK_SIZE}"
  --rwkv-ms-boundary-mode fixed_chunk
  --rwkv-ms-erase-gate 1.0
  --rwkv-ms-read-top-k 0
  --rank "${RANK}"
  --alpha "${ALPHA}"
  --num-state-heads 1
  --beta-bias-init "${BETA_BIAS_INIT}"
  --couple-lambda
  --state-update-mode standard
  --output-init base_slice_fixed
  --delta-heads q,o
  --online-gain "${ONLINE_GAIN}"
  --rankwise-gates
  --target-layers "${TARGET_LAYERS}"
  --trainable-delta-scale
  --delta-scale-init "${DELTA_SCALE_INIT}"
  --delta-scale-max "${DELTA_SCALE_MAX}"
  --delta-scale-granularity head
  --delta-scale-parameterization alpha_over_rank
  --memory-readout-mode delta
  --memory-write-source learned_hidden
  --memory-write-granularity token
  --training-mode episode
  --assistant-loss-mode final_assistant_only
  --episode-recent-messages "${EPISODE_RECENT_MESSAGES}"
  --max-length "${MAX_LENGTH}"
  --max-write-length "${MAX_WRITE_LENGTH}"
  "${episode_read_write_args[@]}"
  --memory-loss-mode context_dropout_ce
  --memory-dropout-no-memory-prob "${MEMORY_NO_MEMORY_PROB}"
  --memory-dropout-state-only-prob "${MEMORY_STATE_ONLY_PROB}"
  --memory-base-kl-weight "${MEMORY_BASE_KL_WEIGHT}"
  --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --lr-scheduler-type "${LR_SCHEDULER_TYPE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --weight-decay 0.0
  --optim adamw_torch_fused
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --validation-split-ratio "${VALIDATION_SPLIT_RATIO}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  "${load_best_args[@]}"
  --logging-steps "${LOGGING_STEPS}"
  --dataset-num-proc 1
  --dataloader-num-workers 0
  --seed "${SEED}"
  --data-seed "${DATA_SEED}"
  --tf32
  --log-delta-debug-stats
  "${resume_args[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  record_status dry_run "output=${OUTPUT_DIR}"
  printf 'Validated v2 command (not started):\n'
  printf '%q ' "${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}"
  printf '\n'
  exit 0
fi

record_status training "output=${OUTPUT_DIR} resume=${resume_args[*]:-none}"
printf '[%s] Starting novel-memory v2 training; output=%s resume=%s\n' \
  "$(timestamp)" "${OUTPUT_DIR}" "${resume_args[*]:-none}" | tee -a "${LOG_FILE}"

set +e
"${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
train_status="${PIPESTATUS[0]}"
set -e
if (( train_status != 0 )); then
  record_status failed "phase=training exit_code=${train_status}"
  exit "${train_status}"
fi

record_status complete "output=${OUTPUT_DIR}"
printf '[%s] Novel-memory v2 training completed successfully.\n' "$(timestamp)" | tee -a "${LOG_FILE}"
