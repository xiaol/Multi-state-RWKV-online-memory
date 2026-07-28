#!/usr/bin/env bash
set -euo pipefail

# Fresh identity-proof launcher. RUN_MODE must be prepare, smoke, or proof.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="/home/xiaol/X/Multi-state-RWKV-online-memory"
PYTHON_BIN="/home/xiaol/X/delta-Mem/.venv/bin/python"
VALIDATION_PYTHON_BIN="python3"
MODEL_PATH="/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"
PAIR_ROOT="/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/pairs_candidate64_failure32_holdout32_v1"
PAIR_MANIFEST="${PAIR_ROOT}/manifest.json"
PAIR_MANIFEST_SHA256="2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008"
TRAIN_FILE="${PAIR_ROOT}/train.jsonl"
TRAIN_SHA256="5f35f6ed41a2edaf88afee83626f17c34da38f5cb61cf4b6796a03eaae38f897"

SOURCE_LOCK="${SCRIPT_DIR}/scene_memory_v6_source_lock.json"
TOKENIZATION_LOCK="${SCRIPT_DIR}/scene_memory_v6_tokenized_cache_lock.json"
DATA_CONTRACT_TOOL="${SCRIPT_DIR}/scene_memory_v6_data_contract.py"
LAUNCH_CONTRACT_TOOL="${SCRIPT_DIR}/scene_memory_v6_launch_contract.py"
RUN_AUDIT_TOOL="${SCRIPT_DIR}/scene_memory_v6_run_audit.py"

EXTERNAL_ROOT="/run/media/xiaol/B214449214445C0B"
RUN_ROOT="${EXTERNAL_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory"
CACHE_ROOT="${EXTERNAL_ROOT}/delta_mem_cache/scene_memory_v6_identity"
HF_HOME_LOCKED="${CACHE_ROOT}/huggingface"
HF_CACHE_DIR="${HF_HOME_LOCKED}/datasets"
XDG_CACHE_HOME_LOCKED="${CACHE_ROOT}/xdg"
TMPDIR_LOCKED="${CACHE_ROOT}/tmp"

: "${RUN_MODE:?Set RUN_MODE to one of: prepare, smoke, proof}"
RUN_ATTEMPT="${RUN_ATTEMPT:-run1}"
PREPARE_AUTH_ATTEMPT="${PREPARE_AUTH_ATTEMPT:-run1}"
SMOKE_AUTH_ATTEMPT="${SMOKE_AUTH_ATTEMPT:-run1}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

for attempt_name in RUN_ATTEMPT PREPARE_AUTH_ATTEMPT SMOKE_AUTH_ATTEMPT; do
  attempt_value="${!attempt_name}"
  [[ "${attempt_value}" =~ ^run[1-9][0-9]*$ ]] \
    || fail "${attempt_name,,}_must_match_runN actual=${attempt_value}"
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "dry_run_must_be_0_or_1 actual=${DRY_RUN}"
[[ -z "${RESUME_FROM_CHECKPOINT:-}" ]] \
  || fail "resume_is_forbidden_for_scene_memory_v6_identity_proof"
[[ -z "${WARM_START_FROM_CHECKPOINT:-}" ]] \
  || fail "warm_start_is_forbidden_for_scene_memory_v6_identity_proof"

case "${RUN_MODE}" in
  prepare)
    RUN_NAME="scene_memory_v6_identityproof_all42_qo_r4_fail32_s32_${RUN_ATTEMPT}_prepare"
    MAX_STEPS=32
    SAVE_STEPS=16
    SAVE_TOTAL_LIMIT=2
    WARMUP_RATIO=0.0625
    ;;
  smoke)
    RUN_NAME="scene_memory_v6_identityproof_all42_qo_r4_fail32_smoke1_${RUN_ATTEMPT}"
    MAX_STEPS=1
    SAVE_STEPS=1
    SAVE_TOTAL_LIMIT=1
    WARMUP_RATIO=0
    ;;
  proof)
    RUN_NAME="scene_memory_v6_identityproof_all42_qo_r4_fail32_s32_${RUN_ATTEMPT}"
    MAX_STEPS=32
    SAVE_STEPS=16
    SAVE_TOTAL_LIMIT=2
    WARMUP_RATIO=0.0625
    ;;
  *)
    fail "run_mode_must_be_prepare_smoke_or_proof actual=${RUN_MODE}"
    ;;
esac

REFERENCE_PREPARE_ROOT="${RUN_ROOT}/scene_memory_v6_identityproof_all42_qo_r4_fail32_s32_${PREPARE_AUTH_ATTEMPT}_prepare"
REFERENCE_PREPARE_RECEIPT="${REFERENCE_PREPARE_ROOT}/prepare_receipt.json"
SMOKE_ROOT="${RUN_ROOT}/scene_memory_v6_identityproof_all42_qo_r4_fail32_smoke1_${SMOKE_AUTH_ATTEMPT}"
SMOKE_RECEIPT="${SMOKE_ROOT}/run_audit_receipt.json"

OUTPUT_DIR="${RUN_ROOT}/${RUN_NAME}"
INITIAL_ADAPTER_DIR="${OUTPUT_DIR}/initial_adapter"
DATA_MANIFEST="${OUTPUT_DIR}/data_contract_manifest.json"
LAUNCH_MANIFEST="${OUTPUT_DIR}/launch_manifest.json"
PREPARE_RECEIPT="${OUTPUT_DIR}/prepare_receipt.json"
RUN_AUDIT_RECEIPT="${OUTPUT_DIR}/run_audit_receipt.json"
LOG_FILE="${RUN_ROOT}/${RUN_NAME}.log"
WATCHER_LOG="${RUN_ROOT}/${RUN_NAME}.checkpoint_watcher.log"
EXPECTED_TARGET_LAYERS="$(seq -s, 0 41)"

[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable path=${PYTHON_BIN}"
command -v "${VALIDATION_PYTHON_BIN}" >/dev/null 2>&1 \
  || fail "validation_python_not_found command=${VALIDATION_PYTHON_BIN}"
[[ "$(realpath -m "${SCRIPT_DIR}/../..")" == "${REPO}" ]] \
  || fail "launcher_repository_path_differs_from_lock"
[[ -d "${MODEL_PATH}" ]] || fail "model_missing path=${MODEL_PATH}"
for required_file in \
  "${PAIR_MANIFEST}" \
  "${TRAIN_FILE}" \
  "${SOURCE_LOCK}" \
  "${TOKENIZATION_LOCK}" \
  "${DATA_CONTRACT_TOOL}" \
  "${LAUNCH_CONTRACT_TOOL}" \
  "${RUN_AUDIT_TOOL}"; do
  [[ -f "${required_file}" && ! -L "${required_file}" ]] \
    || fail "required_file_missing_or_symlink path=${required_file}"
done
[[ "$(sha256sum "${PAIR_MANIFEST}" | awk '{print $1}')" == "${PAIR_MANIFEST_SHA256}" ]] \
  || fail "pair_manifest_hash_differs_from_lock"
[[ "$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')" == "${TRAIN_SHA256}" ]] \
  || fail "training_file_hash_differs_from_lock"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "fresh_output_path_must_not_exist path=${OUTPUT_DIR}"
[[ ! -e "${LOG_FILE}" ]] || fail "fresh_log_path_must_not_exist path=${LOG_FILE}"
[[ ! -e "${WATCHER_LOG}" ]] || fail "fresh_watcher_log_path_must_not_exist path=${WATCHER_LOG}"
for distributed_variable in WORLD_SIZE LOCAL_RANK RANK MASTER_ADDR MASTER_PORT; do
  [[ -z "${!distributed_variable:-}" ]] \
    || fail "distributed_environment_is_forbidden variable=${distributed_variable}"
done
if [[ "${RUN_MODE}" != "prepare" ]]; then
  [[ -s "${REFERENCE_PREPARE_RECEIPT}" ]] \
    || fail "reviewed_prepare_receipt_missing path=${REFERENCE_PREPARE_RECEIPT}"
fi
if [[ "${RUN_MODE}" == "proof" ]]; then
  [[ -s "${SMOKE_RECEIPT}" ]] \
    || fail "validated_smoke_receipt_missing path=${SMOKE_RECEIPT}"
fi

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME_LOCKED}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_LOCKED}"
export TMPDIR="${TMPDIR_LOCKED}"

if ! data_summary="$("${VALIDATION_PYTHON_BIN}" "${DATA_CONTRACT_TOOL}" --summary)"; then
  fail "official_scene_identity_data_contract_invalid"
fi

trainer_help="$("${PYTHON_BIN}" -m deltamem.train.delta_sft --help)"
required_trainer_flags=(
  --prepare-only
  --initial-adapter-output-dir
  --scene-state-identity-margin
  --scene-state-source-manifest
  --expected-scene-state-source-manifest-sha256
  --scene-boundary-payload-ce-weight
  --train-sampler-seed
  --rwkv-ms-semantics-version
  --delta-heads
  --target-layers
)
for required_flag in "${required_trainer_flags[@]}"; do
  if ! rg --quiet --fixed-strings -- "${required_flag}" <<<"${trainer_help}"; then
    fail "trainer_missing_required_flag flag=${required_flag}"
  fi
done

train_args=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --initial-adapter-output-dir "${INITIAL_ADAPTER_DIR}"
  --hf-cache-dir "${HF_CACHE_DIR}"
  --no-tokenized-cache
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
  --rwkv-ms-output-init-scale 0.02
  --rwkv-ms-semantics-version 2
  --rank 4
  --alpha 8
  --num-state-heads 1
  --beta-bias-init 0.0
  --couple-lambda
  --state-update-mode standard
  --output-init base_slice_fixed
  --base-slice-ref-width 8
  --delta-heads q,o
  --no-delta-o-rmsnorm
  --memory-fusion-mode add
  --memory-fusion-placement attention_output
  --trainable-delta-scale
  --delta-scale-init 0.1
  --delta-scale-max 0.5
  --delta-scale-granularity head
  --delta-scale-parameterization alpha_over_rank
  --online-gain 0.2
  --target-layers "${EXPECTED_TARGET_LAYERS}"
  --memory-readout-mode delta
  --memory-write-source learned_hidden
  --memory-write-granularity token
  --training-mode episode
  --assistant-loss-mode final_assistant_only
  --episode-recent-messages 0
  --max-length 256
  --max-write-length 1280
  --no-episode-read-write-enabled
  --memory-loss-mode scene_state_identity_ce
  --scene-state-identity-margin 0.5
  --scene-state-source-manifest "${PAIR_MANIFEST}"
  --expected-scene-state-source-manifest-sha256 "${PAIR_MANIFEST_SHA256}"
  --scene-boundary-payload-ce-weight 0
  --memory-dropout-no-memory-prob 0
  --memory-dropout-state-only-prob 0
  --memory-base-kl-weight 0
  --memory-contrast-weight 0
  --memory-representation-weight 0
  --memory-kl-weight 0
  --memory-causal-weight 0
  --memory-anchor-weight 0
  --memory-recover-weight 0
  --write-sparsity-weight 0
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 1
  --gradient-accumulation-steps 1
  --learning-rate 5e-4
  --lr-scheduler-type constant_with_warmup
  --warmup-ratio "${WARMUP_RATIO}"
  --weight-decay 0
  --optim adamw_torch_fused
  --num-train-epochs 1
  --max-steps "${MAX_STEPS}"
  --logging-steps 1
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-steps 1000
  --validation-split-ratio 0
  --no-load-best-model-at-end
  --dataset-num-proc 1
  --dataloader-num-workers 0
  --frozen-mlp-activation-checkpointing
  --seed 42
  --data-seed 42
  --train-sampler-seed 42
  --tf32
  --log-delta-debug-stats
  --rankwise-gates
)
if [[ "${RUN_MODE}" == "prepare" ]]; then
  train_args+=(--prepare-only)
fi
train_command=("${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}")

printf 'Validated scene_memory_v6 identity data: %s\n' "${data_summary}"
printf 'Locked mode=%s train_rows=32 max_steps=%s save_steps=%s fresh_run=true output=%s\n' \
  "${RUN_MODE}" "${MAX_STEPS}" "${SAVE_STEPS}" "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" "${LAUNCH_CONTRACT_TOOL}" validate-source-lock \
    --repo "${REPO}" \
    --source-lock "${SOURCE_LOCK}"
  "${PYTHON_BIN}" "${LAUNCH_CONTRACT_TOOL}" validate-tokenization-lock \
    --tokenization-lock "${TOKENIZATION_LOCK}"
  if [[ "${RUN_MODE}" != "prepare" ]]; then
    "${PYTHON_BIN}" "${LAUNCH_CONTRACT_TOOL}" validate-prepare-authorization \
      --prepare-receipt "${REFERENCE_PREPARE_RECEIPT}"
  fi
  if [[ "${RUN_MODE}" == "proof" ]]; then
    "${PYTHON_BIN}" "${LAUNCH_CONTRACT_TOOL}" validate-smoke-authorization \
      --smoke-receipt "${SMOKE_RECEIPT}"
  fi
  "${PYTHON_BIN}" "${LAUNCH_CONTRACT_TOOL}" validate-command \
    --run-mode "${RUN_MODE}" \
    --python-bin "${PYTHON_BIN}" \
    --output-dir "${OUTPUT_DIR}" \
    --initial-adapter-dir "${INITIAL_ADAPTER_DIR}" \
    --hf-cache-dir "${HF_CACHE_DIR}" \
    "${train_command[@]}"
  printf 'Validated scene_memory_v6 identity command (not started):\n'
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${OUTPUT_DIR}" \
  "${HF_CACHE_DIR}" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "$(dirname -- "${LOG_FILE}")"

"${VALIDATION_PYTHON_BIN}" "${DATA_CONTRACT_TOOL}" --output "${DATA_MANIFEST}"
"${PYTHON_BIN}" "${LAUNCH_CONTRACT_TOOL}" write-launch-manifest \
  --repo "${REPO}" \
  --source-lock "${SOURCE_LOCK}" \
  --tokenization-lock "${TOKENIZATION_LOCK}" \
  --prepare-receipt "${REFERENCE_PREPARE_RECEIPT}" \
  --smoke-receipt "${SMOKE_RECEIPT}" \
  --data-manifest "${DATA_MANIFEST}" \
  --launch-manifest "${LAUNCH_MANIFEST}" \
  --run-mode "${RUN_MODE}" \
  --run-attempt "${RUN_ATTEMPT}" \
  --python-bin "${PYTHON_BIN}" \
  --output-dir "${OUTPUT_DIR}" \
  --initial-adapter-dir "${INITIAL_ADAPTER_DIR}" \
  --hf-cache-dir "${HF_CACHE_DIR}" \
  "${train_command[@]}"

printf 'Starting locked scene_memory_v6 identity mode=%s output=%s\n' \
  "${RUN_MODE}" "${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

watcher_pid=""
if [[ "${RUN_MODE}" != "prepare" ]]; then
  "${PYTHON_BIN}" "${RUN_AUDIT_TOOL}" watch-checkpoints \
    --run-mode "${RUN_MODE}" \
    --run-root "${OUTPUT_DIR}" \
    --source-lock "${SOURCE_LOCK}" \
    --timeout-seconds 14400 \
    --poll-seconds 2 \
    >"${WATCHER_LOG}" 2>&1 &
  watcher_pid="$!"
fi

set +e
"${train_command[@]}" 2>&1 | tee -a "${LOG_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
train_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
set -e
if (( train_status != 0 || tee_status != 0 )); then
  if [[ -n "${watcher_pid}" ]] && kill -0 "${watcher_pid}" 2>/dev/null; then
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
  fi
  (( train_status == 0 )) \
    || fail "scene_memory_v6_identity_failed mode=${RUN_MODE} exit_code=${train_status}"
  fail "scene_memory_v6_identity_log_failed mode=${RUN_MODE} exit_code=${tee_status}"
fi

if [[ "${RUN_MODE}" == "prepare" ]]; then
  "${PYTHON_BIN}" "${RUN_AUDIT_TOOL}" audit-prepare \
    --run-root "${OUTPUT_DIR}" \
    --source-lock "${SOURCE_LOCK}" \
    --receipt "${PREPARE_RECEIPT}"
  [[ -s "${PREPARE_RECEIPT}" ]] \
    || fail "prepare_receipt_missing path=${PREPARE_RECEIPT}"
  printf 'Prepare-only identity proof completed: %s\n' "${PREPARE_RECEIPT}" | tee -a "${LOG_FILE}"
  exit 0
fi

if ! wait "${watcher_pid}"; then
  tail -n 100 "${WATCHER_LOG}" >&2 || true
  fail "checkpoint_receipt_watcher_failed mode=${RUN_MODE}"
fi
for checkpoint_step in $(seq "${SAVE_STEPS}" "${SAVE_STEPS}" "${MAX_STEPS}"); do
  checkpoint_receipt="${OUTPUT_DIR}/trainer/checkpoint-${checkpoint_step}/checkpoint_receipt.json"
  [[ -s "${checkpoint_receipt}" ]] \
    || fail "checkpoint_receipt_missing path=${checkpoint_receipt}"
done
[[ -s "${OUTPUT_DIR}/training_summary.json" ]] \
  || fail "training_summary_missing_after_completed_run"
"${PYTHON_BIN}" "${RUN_AUDIT_TOOL}" audit-run \
  --run-mode "${RUN_MODE}" \
  --run-root "${OUTPUT_DIR}" \
  --log-file "${LOG_FILE}" \
  --source-lock "${SOURCE_LOCK}" \
  --receipt "${RUN_AUDIT_RECEIPT}" \
  --trainer-exit-code "${train_status}" \
  --tee-exit-code "${tee_status}"
[[ -s "${RUN_AUDIT_RECEIPT}" ]] \
  || fail "run_audit_receipt_missing path=${RUN_AUDIT_RECEIPT}"
printf 'scene_memory_v6 identity %s completed successfully.\n' "${RUN_MODE}" | tee -a "${LOG_FILE}"
