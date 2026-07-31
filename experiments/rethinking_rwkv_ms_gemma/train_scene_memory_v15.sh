#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SSD_ROOT="/run/media/xiaol/B214449214445C0B"
MODEL_PATH="${SSD_ROOT}/models/gemma/gemma-4-E4B-it"
DATA_ROOT="${SSD_ROOT}/delta_mem_data/scene_failure_state/scene_memory_v15/all32_pair64_v1"
SOURCE_LOCK="${SCRIPT_DIR}/scene_memory_v15_source_lock.json"
WARM_START_LOCK="${SCRIPT_DIR}/scene_memory_v14_v13_checkpoint4_lock.json"
PINNED_WARM_START_CHECKPOINT="${SSD_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v13/scene_memory_v13_production_value14_dense_20260731_070142_step4/trainer/checkpoint-4"
RUN_ROOT="${SSD_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v15"
CACHE_ROOT="${SSD_ROOT}/delta_mem_cache/scene_memory_v15"

PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
VALIDATION_PYTHON_BIN="${VALIDATION_PYTHON_BIN:-${REPO}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-}"
SMOKE_RUN="${SMOKE_RUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_STEP="${TARGET_STEP:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
GATE_RECEIPT="${GATE_RECEIPT:-}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fail "run_name_must_be_a_safe_nonempty_component actual=${RUN_NAME:-unset}"
[[ "${SMOKE_RUN}" == "0" || "${SMOKE_RUN}" == "1" ]] \
  || fail "smoke_run_must_be_0_or_1 actual=${SMOKE_RUN}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "dry_run_must_be_0_or_1 actual=${DRY_RUN}"
if [[ -z "${TARGET_STEP}" ]]; then
  [[ "${SMOKE_RUN}" == "1" ]] && TARGET_STEP=1 || TARGET_STEP=4
fi
if [[ "${SMOKE_RUN}" == "1" ]]; then
  [[ "${TARGET_STEP}" == "1" ]] || fail "v15_smoke_target_step_must_be_one"
else
  [[ "${TARGET_STEP}" == "4" ]] || fail "v15_target_step_must_be_four"
fi
[[ -z "${RESUME_FROM_CHECKPOINT}" ]] || fail "v15_resume_is_forbidden"
[[ -z "${GATE_RECEIPT}" ]] || fail "v15_gate_receipt_cannot_authorize_training"
[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable path=${PYTHON_BIN}"
[[ -x "${VALIDATION_PYTHON_BIN}" ]] \
  || fail "validation_python_not_executable path=${VALIDATION_PYTHON_BIN}"

for distributed_variable in \
  WORLD_SIZE LOCAL_RANK RANK MASTER_ADDR MASTER_PORT \
  SLURM_PROCID PMI_RANK OMPI_COMM_WORLD_RANK; do
  [[ -z "${!distributed_variable:-}" ]] \
    || fail "distributed_environment_is_forbidden variable=${distributed_variable}"
done
for forbidden_variable in \
  HARD32 HARD32_FILE HARD32_PATH HARD32_DIR \
  FULL170 FULL170_FILE FULL170_PATH FULL170_DIR \
  BENCHMARK BENCHMARK_FILE BENCHMARK_PATH BENCHMARK_DIR \
  EVAL_DATASET EVAL_FILE EVAL_PATH EVAL_DIR DO_EVAL \
  VALIDATION_DATASET VALIDATION_FILE VALIDATION_PATH VALIDATION_DIR \
  TEST_DATASET TEST_FILE TEST_PATH TEST_DIR; do
  [[ -z "${!forbidden_variable:-}" ]] \
    || fail "benchmark_or_evaluation_access_is_forbidden variable=${forbidden_variable}"
done

run_kind="production"
[[ "${SMOKE_RUN}" == "0" ]] || run_kind="smoke"
RUN_ID="scene_memory_v15_${run_kind}_${RUN_NAME}_step${TARGET_STEP}"
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
LAUNCH_RECEIPT="${LOG_DIR}/${RUN_ID}.launch.json"
COMPLETION_RECEIPT="${LOG_DIR}/${RUN_ID}.completion.json"

HF_HOME_LOCKED="${CACHE_ROOT}/huggingface"
HF_CACHE_DIR="${HF_HOME_LOCKED}/datasets"
XDG_CACHE_HOME_LOCKED="${CACHE_ROOT}/xdg"
TOKENIZED_DATASET_ROOT="${CACHE_ROOT}/tokenized"
TMPDIR_LOCKED="${CACHE_ROOT}/tmp/${RUN_ID}"
TORCH_CACHE_ROOT="${CACHE_ROOT}/torch"
HF_ASSETS_CACHE_LOCKED="${HF_HOME_LOCKED}/assets"
PYTHONPYCACHEPREFIX_LOCKED="${CACHE_ROOT}/python"
TORCHINDUCTOR_CACHE_DIR_LOCKED="${CACHE_ROOT}/torchinductor"
CUDA_CACHE_PATH_LOCKED="${CACHE_ROOT}/cuda"
NUMBA_CACHE_DIR_LOCKED="${CACHE_ROOT}/numba"
MPLCONFIGDIR_LOCKED="${CACHE_ROOT}/matplotlib"
WANDB_DIR_LOCKED="${CACHE_ROOT}/wandb"

require_no_symlink_components() {
  local raw="$1"
  local current="/"
  local component
  local -a components
  IFS='/' read -r -a components <<<"${raw}"
  for component in "${components[@]}"; do
    [[ -n "${component}" ]] || continue
    current="${current%/}/${component}"
    [[ ! -L "${current}" ]] \
      || fail "symlink_path_component_is_forbidden path=${current}"
  done
}

require_ssd_path() {
  local raw="$1"
  local resolved
  [[ "${raw}" == /* ]] || fail "ssd_path_must_be_absolute path=${raw}"
  case "/${raw}/" in
    */../*) fail "parent_path_alias_is_forbidden path=${raw}" ;;
  esac
  require_no_symlink_components "${raw}"
  resolved="$(realpath -m -- "${raw}")"
  [[ "${resolved}" == "${raw}" ]] \
    || fail "path_must_be_canonical path=${raw} resolved=${resolved}"
  [[ "${resolved}" == "${SSD_ROOT}" || "${resolved}" == "${SSD_ROOT}/"* ]] \
    || fail "path_must_stay_on_2t_ssd path=${raw} resolved=${resolved}"
}

require_under_root() {
  local raw="$1"
  local root="$2"
  require_ssd_path "${raw}"
  [[ "${raw}" == "${root}" || "${raw}" == "${root}/"* ]] \
    || fail "path_outside_locked_root path=${raw} root=${root}"
}

for locked_path in \
  "${MODEL_PATH}" "${DATA_ROOT}" "${PINNED_WARM_START_CHECKPOINT}" \
  "${RUN_ROOT}" "${OUTPUT_DIR}" "${LOG_DIR}" "${LOG_FILE}" \
  "${LAUNCH_RECEIPT}" "${COMPLETION_RECEIPT}" "${CACHE_ROOT}" \
  "${HF_HOME_LOCKED}" "${HF_CACHE_DIR}" "${XDG_CACHE_HOME_LOCKED}" \
  "${TOKENIZED_DATASET_ROOT}" "${TMPDIR_LOCKED}" "${TORCH_CACHE_ROOT}" \
  "${HF_ASSETS_CACHE_LOCKED}" "${PYTHONPYCACHEPREFIX_LOCKED}" \
  "${TORCHINDUCTOR_CACHE_DIR_LOCKED}" "${CUDA_CACHE_PATH_LOCKED}" \
  "${NUMBA_CACHE_DIR_LOCKED}" "${MPLCONFIGDIR_LOCKED}" "${WANDB_DIR_LOCKED}"; do
  require_ssd_path "${locked_path}"
done
for run_path in \
  "${OUTPUT_DIR}" "${LOG_DIR}" "${LOG_FILE}" \
  "${LAUNCH_RECEIPT}" "${COMPLETION_RECEIPT}"; do
  require_under_root "${run_path}" "${RUN_ROOT}"
done
for cache_path in \
  "${HF_HOME_LOCKED}" "${HF_CACHE_DIR}" "${XDG_CACHE_HOME_LOCKED}" \
  "${TOKENIZED_DATASET_ROOT}" "${TMPDIR_LOCKED}" "${TORCH_CACHE_ROOT}" \
  "${HF_ASSETS_CACHE_LOCKED}" "${PYTHONPYCACHEPREFIX_LOCKED}" \
  "${TORCHINDUCTOR_CACHE_DIR_LOCKED}" "${CUDA_CACHE_PATH_LOCKED}" \
  "${NUMBA_CACHE_DIR_LOCKED}" "${MPLCONFIGDIR_LOCKED}" "${WANDB_DIR_LOCKED}"; do
  require_under_root "${cache_path}" "${CACHE_ROOT}"
done

[[ -d "${MODEL_PATH}" && ! -L "${MODEL_PATH}" ]] \
  || fail "locked_model_missing_or_symlink"
[[ -d "${DATA_ROOT}" && ! -L "${DATA_ROOT}" ]] \
  || fail "locked_data_root_missing_or_symlink"
[[ -f "${SOURCE_LOCK}" && ! -L "${SOURCE_LOCK}" ]] \
  || fail "source_lock_missing_or_symlink"
[[ -f "${WARM_START_LOCK}" && ! -L "${WARM_START_LOCK}" ]] \
  || fail "warm_start_lock_missing_or_symlink"
[[ -d "${PINNED_WARM_START_CHECKPOINT}" && ! -L "${PINNED_WARM_START_CHECKPOINT}" ]] \
  || fail "warm_start_checkpoint_missing_or_symlink"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "fresh_output_collision path=${OUTPUT_DIR}"
[[ ! -e "${LOG_FILE}" ]] || fail "fresh_log_collision path=${LOG_FILE}"
[[ ! -e "${LAUNCH_RECEIPT}" ]] || fail "fresh_launch_receipt_collision"
[[ ! -e "${COMPLETION_RECEIPT}" ]] || fail "fresh_completion_receipt_collision"

critical_tracked_files=(
  "deltamem/scene_boundary.py"
  "deltamem/train/cached_prefix_replay.py"
  "deltamem/train/delta_sft_experimental.py"
  "deltamem/train/scene_state_generation_alignment.py"
  "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v15_data.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v15_data_contract.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v15_source_lock.json"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_v13_checkpoint4_lock.json"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_warm_start.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v15_launch_contract.py"
  "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v15.sh"
)
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]] \
    || fail "tracked_worktree_must_be_clean_before_v15_training"
  for critical_file in "${critical_tracked_files[@]}"; do
    git -C "${REPO}" ls-files --error-unmatch -- "${critical_file}" >/dev/null 2>&1 \
      || fail "critical_v15_source_must_be_tracked path=${critical_file}"
  done
fi

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME_LOCKED}"
export HF_HUB_CACHE="${HF_HOME_LOCKED}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE_LOCKED}"
export TRANSFORMERS_CACHE="${HF_HOME_LOCKED}/transformers"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_LOCKED}"
export TMPDIR="${TMPDIR_LOCKED}"
export TMP="${TMPDIR_LOCKED}"
export TEMP="${TMPDIR_LOCKED}"
export TORCH_HOME="${TORCH_CACHE_ROOT}"
export TORCH_EXTENSIONS_DIR="${TORCH_CACHE_ROOT}/extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX_LOCKED}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR_LOCKED}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH_LOCKED}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR_LOCKED}"
export MPLCONFIGDIR="${MPLCONFIGDIR_LOCKED}"
export WANDB_DIR="${WANDB_DIR_LOCKED}"

contract_args=(
  -m experiments.rethinking_rwkv_ms_gemma.scene_memory_v15_launch_contract
  --target-step "${TARGET_STEP}"
  --run-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --cache-root "${CACHE_ROOT}"
  --data-root "${DATA_ROOT}"
  --source-lock "${SOURCE_LOCK}"
  --warm-start-lock "${WARM_START_LOCK}"
  --warm-start-checkpoint "${PINNED_WARM_START_CHECKPOINT}"
  --base-model "${MODEL_PATH}"
  --ssd-root "${SSD_ROOT}"
  --format tsv
)
[[ "${SMOKE_RUN}" == "0" ]] || contract_args+=(--smoke)
if ! contract_values="$(PYTHONDONTWRITEBYTECODE=1 "${VALIDATION_PYTHON_BIN}" "${contract_args[@]}")"; then
  fail "v15_launch_contract_validation_failed"
fi

IFS=$'\t' read -r \
  TRAIN_FILE TRAIN_SHA256 SOURCE_MANIFEST SOURCE_MANIFEST_SHA256 \
  SCHEDULE_FILE SCHEDULE_SHA256 WARM_START_CHECKPOINT \
  WARM_START_ADAPTER_SHA256 WARM_START_MODE VALIDATED_OUTPUT_DIR \
  VALIDATED_LOG_FILE VALIDATED_LAUNCH_RECEIPT VALIDATED_COMPLETION_RECEIPT \
  RUN_MODE GRADIENT_ACCUMULATION_STEPS MAX_STEPS SAVE_TOTAL_LIMIT \
  TOTAL_PAIR_PRESENTATIONS FIRST_PAIR_LOW FIRST_PAIR_HIGH SOURCE_LOCK_SHA256 \
  <<<"${contract_values}"
[[ "${VALIDATED_OUTPUT_DIR}" == "${OUTPUT_DIR}" ]] || fail "validated_output_dir_differs"
[[ "${VALIDATED_LOG_FILE}" == "${LOG_FILE}" ]] || fail "validated_log_file_differs"
[[ "${VALIDATED_LAUNCH_RECEIPT}" == "${LAUNCH_RECEIPT}" ]] \
  || fail "validated_launch_receipt_differs"
[[ "${VALIDATED_COMPLETION_RECEIPT}" == "${COMPLETION_RECEIPT}" ]] \
  || fail "validated_completion_receipt_differs"
[[ "${WARM_START_CHECKPOINT}" == "${PINNED_WARM_START_CHECKPOINT}" ]] \
  || fail "validated_warm_start_checkpoint_differs"
require_ssd_path "${TRAIN_FILE}"
require_under_root "${SOURCE_MANIFEST}" "${DATA_ROOT}"
require_under_root "${SCHEDULE_FILE}" "${DATA_ROOT}"

TARGET_LAYERS="$(seq -s, 0 41)"
train_args=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --warm-start-from-checkpoint "${WARM_START_CHECKPOINT}"
  --warm-start-mode "${WARM_START_MODE}"
  --resume-mode exact
  --hf-cache-dir "${HF_CACHE_DIR}"
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
  --memory-fusion-residual-scale 1.0
  --memory-fusion-residual-scale-max 1.0
  --trainable-delta-scale
  --delta-scale-init 0.1
  --delta-scale-max 0.5
  --delta-scale-granularity head
  --delta-scale-parameterization alpha_over_rank
  --online-gain 0.2
  --target-layers "${TARGET_LAYERS}"
  --memory-readout-mode delta
  --memory-write-source learned_hidden
  --memory-write-granularity token
  --training-mode episode
  --assistant-loss-mode final_assistant_only
  --episode-recent-messages 0
  --max-length 256
  --max-write-length 2048
  --no-episode-read-write-enabled
  --memory-loss-mode scene_state_generation_ce
  --scene-state-source-manifest "${SOURCE_MANIFEST}"
  --expected-scene-state-source-manifest-sha256 "${SOURCE_MANIFEST_SHA256}"
  --scene-state-generation-objective-version scene_state_generation_ce_symmetric_cached_prefix_identity_v15
  --scene-state-generated-prefix-correction-weight 0
  --scene-state-generated-unlikelihood-weight 0
  --scene-state-generated-unlikelihood-max-wrong-tokens 0
  --scene-state-generated-rollout-extra-tokens 4
  --scene-state-generated-rollout-max-tokens 24
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
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate 1e-4
  --lr-scheduler-type constant
  --warmup-ratio 0
  --warmup-steps 0
  --weight-decay 0
  --max-grad-norm 1.0
  --optim adamw_torch_fused
  --num-train-epochs 1
  --max-steps "${MAX_STEPS}"
  --logging-steps 1
  --save-steps 1
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --validation-split-ratio 0
  --no-load-best-model-at-end
  --no-ignore-data-skip
  --dataset-num-proc 1
  --dataloader-num-workers 0
  --frozen-mlp-activation-checkpointing
  --seed 42
  --data-seed 42
  --tf32
  --log-delta-debug-stats
  --rankwise-gates
)
if [[ "${SMOKE_RUN}" == "1" ]]; then
  train_args+=(--scene-state-v15-one-pair-smoke)
fi
train_command=("${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}")

printf 'Validated V15 mode=%s optimizer_steps=%s pair_presentations=%s accumulation=%s; first pair=%s,%s\n' \
  "${RUN_MODE}" "${MAX_STEPS}" "${TOTAL_PAIR_PRESENTATIONS}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${FIRST_PAIR_LOW}" "${FIRST_PAIR_HIGH}"
printf 'Locked train_sha256=%s schedule_sha256=%s source_lock_sha256=%s warm_adapter_sha256=%s\n' \
  "${TRAIN_SHA256}" "${SCHEDULE_SHA256}" "${SOURCE_LOCK_SHA256}" \
  "${WARM_START_ADAPTER_SHA256}"
printf 'Resume, validation, Hard32, Full170, test, and every other benchmark are forbidden during training.\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Validated V15 training command (not started):\n'
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${RUN_ROOT}" "${LOG_DIR}" "${HF_CACHE_DIR}" "${HF_HUB_CACHE}" \
  "${XDG_CACHE_HOME}" "${TOKENIZED_DATASET_ROOT}" "${TMPDIR}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${HF_ASSETS_CACHE}" \
  "${PYTHONPYCACHEPREFIX}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${WANDB_DIR}"

CODE_COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
"${VALIDATION_PYTHON_BIN}" - \
  "${LAUNCH_RECEIPT}" "${RUN_NAME}" "${OUTPUT_DIR}" "${CACHE_ROOT}" \
  "${TARGET_STEP}" "${SMOKE_RUN}" "${CODE_COMMIT}" <<'PY'
from pathlib import Path
import sys

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v15_launch_contract as launch

output, run_name, output_dir, cache_root, target_step, smoke_raw, commit = sys.argv[1:]
smoke = smoke_raw == "1"
contract = launch.validate_launch_contract(
    target_step=int(target_step),
    run_name=run_name,
    output_dir=Path(output_dir),
    cache_root=Path(cache_root),
    smoke=smoke,
)
launch.write_launch_receipt(
    Path(output),
    contract,
    git_commit=commit,
    critical_files=launch.critical_training_code_bindings(),
)
PY

mkdir "${OUTPUT_DIR}"

{
  printf 'Launch receipt: %s\n' "${LAUNCH_RECEIPT}"
  printf 'Starting attached V15 training. Command: '
  printf '%q ' "${train_command[@]}"
  printf '\n'
} | tee "${LOG_FILE}"

set +e
"${train_command[@]}" 2>&1 | tee -a "${LOG_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
(( pipeline_status[0] == 0 )) \
  || fail "v15_training_failed exit_code=${pipeline_status[0]} log=${LOG_FILE}"
(( pipeline_status[1] == 0 )) \
  || fail "v15_training_log_failed exit_code=${pipeline_status[1]}"

TRAINING_SUMMARY="${OUTPUT_DIR}/training_summary.json"
[[ -s "${TRAINING_SUMMARY}" ]] || fail "completed_training_summary_missing"
checkpoint_args=()
for checkpoint_step in $(seq 1 "${TARGET_STEP}"); do
  checkpoint="${OUTPUT_DIR}/trainer/checkpoint-${checkpoint_step}"
  [[ -d "${checkpoint}" ]] || fail "completed_checkpoint_missing step=${checkpoint_step}"
  checkpoint_args+=("${checkpoint}")
done

"${VALIDATION_PYTHON_BIN}" - \
  "${COMPLETION_RECEIPT}" "${LAUNCH_RECEIPT}" "${TRAINING_SUMMARY}" \
  "${LOG_FILE}" "${SMOKE_RUN}" "${checkpoint_args[@]}" <<'PY'
from pathlib import Path
import sys

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v15_launch_contract as launch

output, launch_receipt, summary, log, smoke_raw, *checkpoint_raw = sys.argv[1:]
launch.write_completion_receipt(
    Path(output),
    launch_receipt=Path(launch_receipt),
    training_summary=Path(summary),
    log_file=Path(log),
    checkpoints=[Path(value) for value in checkpoint_raw],
    smoke=smoke_raw == "1",
)
launch.validate_completion_receipt(Path(output))
PY

if [[ "${SMOKE_RUN}" == "1" ]]; then
  printf 'Completed V15 one-pair real-backward smoke at checkpoint-1: %s\n' "${OUTPUT_DIR}"
  printf 'Smoke is not production-eligible and cannot authorize any benchmark.\n'
else
  printf 'Completed V15 checkpoints 1 through 4: %s\n' "${OUTPUT_DIR}"
  printf 'Use train-only checkpoint selection before the single frozen Hard32 run.\n'
fi
