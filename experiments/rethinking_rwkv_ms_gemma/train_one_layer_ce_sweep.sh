#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: train_one_layer_ce_sweep.sh [--dry-run] [--resume]

Runs six independent CE-only probes for layers 4, 5, 10, 11, 22, and 23.

Options:
  --dry-run  Validate inputs and print every command without writing or training.
  --resume   Permit nonempty layer output directories and resume their latest
             complete checkpoint. Completed layer runs are skipped.
  --help     Show this help text.

Selected environment overrides:
  SWEEP_ROOT, MODEL_PATH, TRAIN_FILE, TOKENIZED_DATASET_ROOT, PYTHON_BIN,
  MAX_STEPS, NUM_TRAIN_EPOCHS, WARMUP_RATIO, SAVE_STEPS, DRY_RUN, RESUME.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

timestamp() {
  date --iso-8601=seconds
}

DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --resume)
      RESUME=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
  shift
done

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || fail "DRY_RUN must be 0 or 1"
[[ "${RESUME}" == "0" || "${RESUME}" == "1" ]] || fail "RESUME must be 0 or 1"

REPO="${REPO:-/home/xiaol/X/Multi-state-RWKV-online-memory}"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it}"
TRAIN_FILE="${TRAIN_FILE:-/run/media/xiaol/B214449214445C0B/delta_mem_data/novel_agent_memory/novel_memory_content_control_probe_seed20260724_n32.jsonl}"
EXPECTED_TRAIN_SHA256="0aa7472d3c7fe3b5501801fc380f570b82a048c6e535e800263c6e1c2ee08a2d"
EXPECTED_TRAIN_ROWS=32

RUN_ROOT="${RUN_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory}"
SWEEP_ROOT="${SWEEP_ROOT:-${RUN_ROOT}/v4_one_layer_ce_gate128_content_control_seed20260724_n32}"
TOKENIZED_DATASET_ROOT="${TOKENIZED_DATASET_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_tokenized/novel_agent_memory/loss_probe/v4_one_layer_ce_gate128_content_control_seed20260724_n32_read256_write512_recent1}"
SOURCE_VALIDATOR="${REPO}/experiments/rethinking_rwkv_ms_gemma/validate_one_layer_ce_source.py"

MAX_STEPS="${MAX_STEPS:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0625}"
SAVE_STEPS="${SAVE_STEPS:-32}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SUMMARIZE_AFTER="${SUMMARIZE_AFTER:-1}"

LAYERS=(4 5 10 11 22 23)
CHECKPOINT_FILES=(
  delta_mem_adapter.pt
  delta_mem_config.json
  optimizer.pt
  rng_state.pth
  scheduler.pt
  trainer_state.json
  training_protocol.json
)

[[ -x "${PYTHON_BIN}" ]] || fail "python is not executable: ${PYTHON_BIN}"
[[ -d "${REPO}/deltamem" ]] || fail "repository is missing deltamem: ${REPO}"
[[ -d "${MODEL_PATH}" ]] || fail "model directory is missing: ${MODEL_PATH}"
[[ -f "${TRAIN_FILE}" ]] || fail "training file is missing: ${TRAIN_FILE}"
[[ -f "${SOURCE_VALIDATOR}" ]] || fail "source validator is missing: ${SOURCE_VALIDATOR}"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be a positive integer"
[[ "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "SAVE_STEPS must be a positive integer"
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || fail "SAVE_TOTAL_LIMIT must be a positive integer"
[[ "${LOGGING_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "LOGGING_STEPS must be a positive integer"
[[ "${SUMMARIZE_AFTER}" == "0" || "${SUMMARIZE_AFTER}" == "1" ]] \
  || fail "SUMMARIZE_AFTER must be 0 or 1"

actual_train_sha256="$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')"
[[ "${actual_train_sha256}" == "${EXPECTED_TRAIN_SHA256}" ]] \
  || fail "dataset checksum mismatch: expected=${EXPECTED_TRAIN_SHA256} actual=${actual_train_sha256}"
actual_train_rows="$(awk 'NF {count += 1} END {print count + 0}' "${TRAIN_FILE}")"
[[ "${actual_train_rows}" == "${EXPECTED_TRAIN_ROWS}" ]] \
  || fail "dataset row-count mismatch: expected=${EXPECTED_TRAIN_ROWS} actual=${actual_train_rows}"
source_contract_json="$(
  "${PYTHON_BIN}" "${SOURCE_VALIDATOR}" \
    --source "${TRAIN_FILE}" \
    --expected-sha256 "${EXPECTED_TRAIN_SHA256}"
)"

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/run/media/xiaol/B214449214445C0B/delta_mem_cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/run/media/xiaol/B214449214445C0B/delta_mem_cache/xdg}"
export TMPDIR="${TMPDIR:-/run/media/xiaol/B214449214445C0B/delta_mem_cache/tmp}"

trainer_help="$("${PYTHON_BIN}" -m deltamem.train.delta_sft --help)"
required_flags=(
  --episode-read-write-enabled
  --no-episode-read-write-enabled
  --memory-base-kl-weight
  --memory-dropout-no-memory-prob
  --memory-dropout-state-only-prob
  --resume-from-checkpoint
  --tokenized-cache
)
for required_flag in "${required_flags[@]}"; do
  if ! rg --quiet --fixed-strings -- "${required_flag}" <<<"${trainer_help}"; then
    fail "trainer is missing required flag: ${required_flag}"
  fi
done

directory_is_nonempty() {
  local directory="$1"
  [[ -d "${directory}" && -n "$(find "${directory}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

checkpoint_is_complete() {
  local checkpoint="$1"
  local filename
  [[ -d "${checkpoint}" ]] || return 1
  for filename in "${CHECKPOINT_FILES[@]}"; do
    [[ -s "${checkpoint}/${filename}" ]] || return 1
  done
}

latest_complete_checkpoint() {
  local output_dir="$1"
  local checkpoint
  while IFS= read -r checkpoint; do
    if checkpoint_is_complete "${checkpoint}"; then
      printf '%s\n' "${checkpoint}"
      return 0
    fi
  done < <(
    find "${output_dir}/trainer" -mindepth 1 -maxdepth 1 -type d \
      -name 'checkpoint-*' -print 2>/dev/null | sort -Vr
  )
  return 1
}

run_is_complete() {
  local output_dir="$1"
  [[ -s "${output_dir}/delta_mem_adapter.pt" \
    && -s "${output_dir}/delta_mem_config.json" \
    && -s "${output_dir}/training_protocol.json" \
    && -s "${output_dir}/training_summary.json" ]]
}

declare -a OUTPUT_DIRS ACTIONS RESUME_PATHS
for index in "${!LAYERS[@]}"; do
  layer="${LAYERS[index]}"
  printf -v layer_tag 'layer_%02d' "${layer}"
  output_dir="${SWEEP_ROOT}/${layer_tag}"
  OUTPUT_DIRS[index]="${output_dir}"
  ACTIONS[index]=fresh
  RESUME_PATHS[index]=""

  if ! directory_is_nonempty "${output_dir}"; then
    continue
  fi
  if [[ "${RESUME}" != "1" ]]; then
    fail "output directory is nonempty; rerun with --resume only if continuation is intended: ${output_dir}"
  fi
  if run_is_complete "${output_dir}"; then
    ACTIONS[index]=skip
    continue
  fi
  checkpoint="$(latest_complete_checkpoint "${output_dir}" || true)"
  [[ -n "${checkpoint}" ]] \
    || fail "explicit resume requested but no complete checkpoint exists: ${output_dir}"
  ACTIONS[index]=resume
  RESUME_PATHS[index]="${checkpoint}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'source_contract=%s\n' "${source_contract_json}"
else
  mkdir -p "${SWEEP_ROOT}/_metadata"
  printf '%s\n' "${source_contract_json}" >"${SWEEP_ROOT}/_metadata/source_contract.json"
fi

build_train_command() {
  local layer="$1"
  local output_dir="$2"
  local resume_path="$3"
  TRAIN_COMMAND=(
    "${PYTHON_BIN}" -m deltamem.train.delta_sft
    --model-path "${MODEL_PATH}"
    --train-file "${TRAIN_FILE}"
    --output-dir "${output_dir}"
    --hf-cache-dir "${HF_HOME}/datasets"
    --tokenized-dataset-root "${TOKENIZED_DATASET_ROOT}"
    --tokenized-cache
    --device cuda:0
    --dtype bfloat16
    --bf16
    --attn-implementation sdpa
    --memory-backend rwkv_ms
    --rwkv-ms-num-states 4
    --rwkv-ms-chunk-size 128
    --rwkv-ms-boundary-mode fixed_chunk
    --rwkv-ms-erase-gate 1.0
    --rwkv-ms-read-top-k 0
    --rank 8
    --alpha 8
    --num-state-heads 1
    --beta-bias-init 0.0
    --couple-lambda
    --state-update-mode standard
    --output-init base_slice_fixed
    --delta-heads q,o
    --online-gain 0.2
    --rankwise-gates
    --target-layers "${layer}"
    --trainable-delta-scale
    --delta-scale-init 0.10
    --delta-scale-max 0.50
    --delta-scale-granularity head
    --delta-scale-parameterization alpha_over_rank
    --memory-readout-mode delta
    --memory-write-source learned_hidden
    --memory-write-granularity token
    --training-mode episode
    --assistant-loss-mode final_assistant_only
    --episode-recent-messages 1
    --max-length 256
    --max-write-length 512
    --no-episode-read-write-enabled
    --memory-loss-mode context_dropout_ce
    --memory-dropout-no-memory-prob 0
    --memory-dropout-state-only-prob 0
    --memory-kl-weight 0
    --memory-base-kl-weight 0
    --per-device-train-batch-size 1
    --per-device-eval-batch-size 1
    --gradient-accumulation-steps 1
    --learning-rate 1e-3
    --lr-scheduler-type constant_with_warmup
    --warmup-ratio "${WARMUP_RATIO}"
    --weight-decay 0.0
    --optim adamw_torch_fused
    --num-train-epochs "${NUM_TRAIN_EPOCHS}"
    --max-steps "${MAX_STEPS}"
    --validation-split-ratio 0
    --eval-steps 1000
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --logging-steps "${LOGGING_STEPS}"
    --dataset-num-proc 1
    --dataloader-num-workers 0
    --seed 42
    --data-seed 42
    --tf32
    --log-delta-debug-stats
  )
  if [[ -n "${resume_path}" ]]; then
    TRAIN_COMMAND+=(--resume-from-checkpoint "${resume_path}")
  fi
}

for index in "${!LAYERS[@]}"; do
  layer="${LAYERS[index]}"
  output_dir="${OUTPUT_DIRS[index]}"
  action="${ACTIONS[index]}"
  resume_path="${RESUME_PATHS[index]}"
  if [[ "${action}" == "skip" ]]; then
    printf 'layer=%s action=skip_complete output=%s\n' "${layer}" "${output_dir}"
    continue
  fi

  build_train_command "${layer}" "${output_dir}" "${resume_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'layer=%s action=%s output=%s\n' "${layer}" "${action}" "${output_dir}"
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
    continue
  fi

  mkdir -p "${SWEEP_ROOT}/_logs"
  printf -v layer_tag 'layer_%02d' "${layer}"
  log_file="${SWEEP_ROOT}/_logs/${layer_tag}.log"
  status_file="${SWEEP_ROOT}/_logs/${layer_tag}.status"
  printf 'updated_at=%s\nstate=training\nlayer=%s\naction=%s\noutput=%s\n' \
    "$(timestamp)" "${layer}" "${action}" "${output_dir}" >"${status_file}"
  printf '[%s] layer=%s action=%s output=%s\n' \
    "$(timestamp)" "${layer}" "${action}" "${output_dir}" | tee -a "${log_file}"

  set +e
  "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${log_file}"
  train_status="${PIPESTATUS[0]}"
  set -e
  if ((train_status != 0)); then
    printf 'updated_at=%s\nstate=failed\nlayer=%s\nexit_code=%s\noutput=%s\n' \
      "$(timestamp)" "${layer}" "${train_status}" "${output_dir}" >"${status_file}"
    exit "${train_status}"
  fi
  printf 'updated_at=%s\nstate=complete\nlayer=%s\noutput=%s\n' \
    "$(timestamp)" "${layer}" "${output_dir}" >"${status_file}"
done

if [[ "${DRY_RUN}" == "0" && "${SUMMARIZE_AFTER}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO}/experiments/rethinking_rwkv_ms_gemma/summarize_one_layer_ce_sweep.py" \
    --sweep-root "${SWEEP_ROOT}" \
    --expected-steps "${MAX_STEPS}"
fi
