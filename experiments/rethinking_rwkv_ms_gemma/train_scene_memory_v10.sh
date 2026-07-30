#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SSD_ROOT="/run/media/xiaol/B214449214445C0B"
MODEL_PATH="${SSD_ROOT}/models/gemma/gemma-4-E4B-it"
DATA_ROOT="${SSD_ROOT}/delta_mem_data/scene_failure_state/scene_memory_v9/value14_pair28_v1"
SOURCE_LOCK="${SCRIPT_DIR}/scene_memory_v9_source_lock.json"
WARM_START_LOCK="${SCRIPT_DIR}/scene_memory_v9_v8_checkpoint56_lock.json"
RUN_ROOT="${SSD_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v10"
GATES_ROOT="${RUN_ROOT}/gates"
CACHE_ROOT="${SSD_ROOT}/delta_mem_cache/scene_memory_v10"
PINNED_WARM_START_CHECKPOINT="${SSD_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v8/scene_memory_v8_production_v8_value56_20260729_1931_step56/trainer/checkpoint-56"
PINNED_HISTORICAL_TRAIN32="${SSD_ROOT}/delta_mem_data/scene_failure_state/scene_memory_v7_fixed_hard32_aligned_train32_v1/train32.jsonl"

PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
VALIDATION_PYTHON_BIN="${VALIDATION_PYTHON_BIN:-${REPO}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-}"
TARGET_STEP="${TARGET_STEP:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
GATE_RECEIPT="${GATE_RECEIPT:-}"
SMOKE_RUN="${SMOKE_RUN:-0}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fail "run_name_must_be_a_safe_nonempty_component actual=${RUN_NAME:-unset}"
[[ "${TARGET_STEP}" =~ ^[1-4]$ ]] \
  || fail "target_step_must_be_v10_endpoint_1_to_4 actual=${TARGET_STEP}"
[[ "${SMOKE_RUN}" == "0" || "${SMOKE_RUN}" == "1" ]] \
  || fail "smoke_run_must_be_0_or_1 actual=${SMOKE_RUN}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "dry_run_must_be_0_or_1 actual=${DRY_RUN}"
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
  EVAL_DATASET EVAL_FILE EVAL_PATH EVAL_DIR DO_EVAL \
  VALIDATION_DATASET VALIDATION_FILE VALIDATION_PATH VALIDATION_DIR \
  TEST_DATASET TEST_FILE TEST_PATH TEST_DIR; do
  [[ -z "${!forbidden_variable:-}" ]] \
    || fail "hard32_or_evaluation_access_is_forbidden variable=${forbidden_variable}"
done

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  [[ "${RESUME_FROM_CHECKPOINT}" == /* ]] \
    || fail "resume_checkpoint_must_be_absolute"
  [[ "${RESUME_FROM_CHECKPOINT}" != "auto" && \
     "${RESUME_FROM_CHECKPOINT}" != "latest" && \
     "${RESUME_FROM_CHECKPOINT}" != "last" ]] \
    || fail "resume_checkpoint_must_be_explicit"
  [[ "${GATE_RECEIPT}" == /* ]] \
    || fail "resume_requires_absolute_v10_gate_receipt"
elif [[ -n "${GATE_RECEIPT}" ]]; then
  fail "fresh_launch_forbids_gate_receipt"
fi
[[ "${SMOKE_RUN}" == "0" || -z "${RESUME_FROM_CHECKPOINT}" ]] \
  || fail "smoke_launch_forbids_resume"

run_kind="production"
[[ "${SMOKE_RUN}" == "0" ]] || run_kind="smoke"
RUN_ID="scene_memory_v10_${run_kind}_${RUN_NAME}_step${TARGET_STEP}"
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
    */../*)
      fail "parent_path_alias_is_forbidden path=${raw}"
      ;;
  esac
  case "${raw,,}" in
    *hard32*|*eval*|*validation*|*holdout*)
      [[ "${raw}" == "${PINNED_HISTORICAL_TRAIN32}" ]] || \
      fail "hard32_or_evaluation_path_is_forbidden path=${raw}"
      ;;
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
  "${MODEL_PATH}" "${DATA_ROOT}" "${RUN_ROOT}" "${GATES_ROOT}" \
  "${OUTPUT_DIR}" "${LOG_DIR}" \
  "${LOG_FILE}" "${LAUNCH_RECEIPT}" "${COMPLETION_RECEIPT}" \
  "${HF_HOME_LOCKED}" "${HF_CACHE_DIR}" "${XDG_CACHE_HOME_LOCKED}" \
  "${TOKENIZED_DATASET_ROOT}" "${TMPDIR_LOCKED}" "${TORCH_CACHE_ROOT}" \
  "${HF_ASSETS_CACHE_LOCKED}" "${PYTHONPYCACHEPREFIX_LOCKED}" \
  "${TORCHINDUCTOR_CACHE_DIR_LOCKED}" "${CUDA_CACHE_PATH_LOCKED}" \
  "${NUMBA_CACHE_DIR_LOCKED}"; do
  require_ssd_path "${locked_path}"
done
for run_path in \
  "${OUTPUT_DIR}" "${LOG_DIR}" "${LOG_FILE}" "${LAUNCH_RECEIPT}" \
  "${COMPLETION_RECEIPT}"; do
  require_under_root "${run_path}" "${RUN_ROOT}"
done
for cache_path in \
  "${HF_HOME_LOCKED}" "${HF_CACHE_DIR}" "${XDG_CACHE_HOME_LOCKED}" \
  "${TOKENIZED_DATASET_ROOT}" "${TMPDIR_LOCKED}" "${TORCH_CACHE_ROOT}" \
  "${HF_ASSETS_CACHE_LOCKED}" "${PYTHONPYCACHEPREFIX_LOCKED}" \
  "${TORCHINDUCTOR_CACHE_DIR_LOCKED}" "${CUDA_CACHE_PATH_LOCKED}" \
  "${NUMBA_CACHE_DIR_LOCKED}"; do
  require_under_root "${cache_path}" "${CACHE_ROOT}"
done
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  require_under_root "${RESUME_FROM_CHECKPOINT}" "${RUN_ROOT}"
  require_under_root "${GATE_RECEIPT}" "${GATES_ROOT}"
fi

[[ -d "${MODEL_PATH}" && ! -L "${MODEL_PATH}" ]] \
  || fail "locked_model_missing_or_symlink"
[[ -d "${DATA_ROOT}" && ! -L "${DATA_ROOT}" ]] \
  || fail "locked_data_root_missing_or_symlink"
[[ -f "${SOURCE_LOCK}" && ! -L "${SOURCE_LOCK}" ]] \
  || fail "source_lock_missing_or_symlink"
[[ -f "${WARM_START_LOCK}" && ! -L "${WARM_START_LOCK}" ]] \
  || fail "warm_start_lock_missing_or_symlink"
require_no_symlink_components "${SOURCE_LOCK}"
require_no_symlink_components "${WARM_START_LOCK}"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "fresh_output_collision path=${OUTPUT_DIR}"
[[ ! -e "${LOG_FILE}" ]] || fail "fresh_log_collision path=${LOG_FILE}"
[[ ! -e "${LAUNCH_RECEIPT}" ]] || fail "fresh_launch_receipt_collision"
[[ ! -e "${COMPLETION_RECEIPT}" ]] || fail "fresh_completion_receipt_collision"
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]] \
  || fail "tracked_worktree_must_be_clean_before_v10_training"

critical_tracked_files=(
  "deltamem/train/delta_sft_experimental.py"
  "deltamem/train/scene_state_generation_alignment.py"
  "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v9_data.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_data_contract.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_source_lock.json"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_v8_checkpoint56_lock.json"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_warm_start.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_launch_contract.py"
  "experiments/rethinking_rwkv_ms_gemma/run_scene_memory_v10_gate.py"
  "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v10.sh"
)
for critical_file in "${critical_tracked_files[@]}"; do
  git -C "${REPO}" ls-files --error-unmatch -- "${critical_file}" >/dev/null 2>&1 \
    || fail "critical_v10_source_must_be_tracked path=${critical_file}"
done

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
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

contract_args=(
  -m experiments.rethinking_rwkv_ms_gemma.scene_memory_v10_launch_contract
  --target-step "${TARGET_STEP}"
  --data-root "${DATA_ROOT}"
  --source-lock "${SOURCE_LOCK}"
  --warm-start-lock "${WARM_START_LOCK}"
  --ssd-root "${SSD_ROOT}"
  --format tsv
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  contract_args+=(--resume-checkpoint "${RESUME_FROM_CHECKPOINT}")
  contract_args+=(--gate-receipt "${GATE_RECEIPT}")
fi
[[ "${SMOKE_RUN}" == "0" ]] || contract_args+=(--smoke)
if ! contract_values="$("${VALIDATION_PYTHON_BIN}" "${contract_args[@]}")"; then
  fail "v10_launch_contract_validation_failed"
fi

IFS=$'\t' read -r \
  TRAIN_FILE TRAIN_SHA256 SOURCE_MANIFEST SOURCE_MANIFEST_SHA256 \
  SCHEDULE_FILE SCHEDULE_SHA256 WARM_START_CHECKPOINT LAUNCH_MODE \
  SOURCE_STEP VALIDATED_TARGET_STEP RESUME_SCHEDULE_CURSOR \
  NEXT_PAIR_LOW_ORDINAL NEXT_PAIR_HIGH_ORDINAL NEXT_SCHEDULE_ENTRY_SHA256 \
  SAVE_STEPS GATE_AUTHORIZATION_KIND VALIDATED_GATE_RECEIPT \
  GATE_RECEIPT_FILE_SHA256 GATE_RECEIPT_SHA256 <<<"${contract_values}"
[[ "${VALIDATED_TARGET_STEP}" == "${TARGET_STEP}" ]] \
  || fail "validated_target_step_differs"
[[ "${RESUME_SCHEDULE_CURSOR}" -eq $((SOURCE_STEP * 7)) ]] \
  || fail "v10_cursor_must_equal_global_step_times_seven"
[[ "${TRAIN_FILE}" == "${PINNED_HISTORICAL_TRAIN32}" ]] \
  || fail "validated_train_file_differs_from_pinned_historical_train32"
[[ "${WARM_START_CHECKPOINT}" == "${PINNED_WARM_START_CHECKPOINT}" ]] \
  || fail "validated_warm_start_checkpoint_differs_from_pin"
require_ssd_path "${TRAIN_FILE}"
require_under_root "${SOURCE_MANIFEST}" "${DATA_ROOT}"
require_under_root "${SCHEDULE_FILE}" "${DATA_ROOT}"
require_ssd_path "${WARM_START_CHECKPOINT}"

TARGET_LAYERS="$(seq -s, 0 41)"
checkpoint_args=()
if [[ "${LAUNCH_MODE}" == "warm_start" || "${LAUNCH_MODE}" == "warm_start_smoke" ]]; then
  checkpoint_args=(
    --warm-start-from-checkpoint "${WARM_START_CHECKPOINT}"
    --warm-start-mode scene_memory_v10_v8_checkpoint56_adapter_only
    --resume-mode exact
  )
elif [[ "${LAUNCH_MODE}" == "resume" ]]; then
  checkpoint_args=(
    --resume-from-checkpoint "$(realpath -- "${RESUME_FROM_CHECKPOINT}")"
    --resume-mode extend
  )
else
  fail "validated_launch_mode_differs actual=${LAUNCH_MODE}"
fi

train_args=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  "${checkpoint_args[@]}"
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
  --scene-state-generation-objective-version scene_state_generation_ce_symmetric_cycle_retention_v4
  --scene-state-generated-prefix-correction-weight 0.5
  --scene-state-generated-unlikelihood-weight 0
  --scene-state-generated-unlikelihood-max-wrong-tokens 4
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
  --per-device-eval-batch-size 1
  --gradient-accumulation-steps 7
  --learning-rate 2e-4
  --lr-scheduler-type constant
  --warmup-ratio 0
  --warmup-steps 0
  --weight-decay 0
  --optim adamw_torch_fused
  --num-train-epochs 1
  --max-steps "${TARGET_STEP}"
  --logging-steps 1
  --save-steps "${SAVE_STEPS}"
  --save-total-limit 1
  --eval-steps 1000
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
train_command=("${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}")

printf 'Validated V10 mode=%s source_step=%s target_step=%s cursor=%s cycle=%s..%s\n' \
  "${LAUNCH_MODE}" "${SOURCE_STEP}" "${TARGET_STEP}" \
  "${RESUME_SCHEDULE_CURSOR}" "${RESUME_SCHEDULE_CURSOR}" \
  "$((RESUME_SCHEDULE_CURSOR + 6))"
printf 'Locked train_sha256=%s schedule_sha256=%s next_entry=%s\n' \
  "${TRAIN_SHA256}" "${SCHEDULE_SHA256}" "${NEXT_SCHEDULE_ENTRY_SHA256}"
printf 'Hard32 and evaluation access forbidden; attached output: %s\n' "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Validated V10 training command (not started):\n'
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${RUN_ROOT}" "${LOG_DIR}" "${HF_CACHE_DIR}" "${HF_HUB_CACHE}" \
  "${XDG_CACHE_HOME}" "${TOKENIZED_DATASET_ROOT}" "${TMPDIR}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${HF_ASSETS_CACHE}" \
  "${PYTHONPYCACHEPREFIX}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}"
mkdir "${OUTPUT_DIR}"
[[ -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || fail "trainer_output_must_be_empty_before_entry"

CODE_COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
TRAINER_CODE="${REPO}/deltamem/train/delta_sft_experimental.py"
LAUNCHER="${SCRIPT_DIR}/train_scene_memory_v10.sh"
LAUNCH_CONTRACT="${SCRIPT_DIR}/scene_memory_v10_launch_contract.py"
GATE_CODE="${SCRIPT_DIR}/run_scene_memory_v10_gate.py"
"${VALIDATION_PYTHON_BIN}" - \
  "${LAUNCH_RECEIPT}" "${RUN_NAME}" "${LAUNCH_MODE}" "${SOURCE_STEP}" \
  "${TARGET_STEP}" "${RESUME_FROM_CHECKPOINT}" "${RESUME_SCHEDULE_CURSOR}" \
  "${OUTPUT_DIR}" "${LOG_FILE}" "${GATE_AUTHORIZATION_KIND}" \
  "${VALIDATED_GATE_RECEIPT}" "${GATE_RECEIPT_FILE_SHA256}" \
  "${GATE_RECEIPT_SHA256}" "${CODE_COMMIT}" "${TRAINER_CODE}" \
  "${LAUNCHER}" "${LAUNCH_CONTRACT}" "${GATE_CODE}" <<'PY'
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import sys

(output_raw, run_name, launch_mode, source_step, target_step,
 resume_checkpoint, cursor, trainer_output, log_file, authorization_kind,
 gate_receipt, gate_file_sha, gate_sha, code_commit, trainer_code, launcher,
 launch_contract, gate_code) = sys.argv[1:]

def sha(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

cursor_value = int(cursor)
payload = {
    "schema": "rwkv_ms_scene_memory_v10_attached_launch.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_name": run_name,
    "attached_foreground_execution": True,
    "launch_mode": launch_mode,
    "source_step": int(source_step),
    "target_step": int(target_step),
    "resume_checkpoint": resume_checkpoint or None,
    "trainer_output": trainer_output,
    "log_file": log_file,
    "fixed_schedule_cursor": {
        "consumed_pair_presentations": cursor_value,
        "next_schedule_index": cursor_value,
        "cycle_stop_exclusive": cursor_value + 7,
        "cursor_formula": "global_step_times_7",
        "ignore_data_skip": False,
    },
    "progression_gate": {
        "authorization_kind": authorization_kind,
        "receipt": None if gate_receipt == "not_required_fresh_start" else gate_receipt,
        "receipt_file_sha256": None if gate_file_sha == "not_required_fresh_start" else gate_file_sha,
        "receipt_sha256": None if gate_sha == "not_required_fresh_start" else gate_sha,
        "required": launch_mode == "resume",
    },
    "objective": {
        "version": "scene_state_generation_ce_symmetric_cycle_retention_v4",
        "selected_full_vocab_ce_in_total": False,
        "gradient_accumulation_steps": 7,
        "pair_presentations_per_optimizer_step": 7,
    },
    "optimization": {
        "learning_rate": 2e-4,
        "scheduler": "constant",
        "warmup_steps": 0,
        "physical_batch_size": 1,
        "gradient_accumulation_steps": 7,
        "optimizer": "adamw_torch_fused",
        "ignore_data_skip": False,
    },
    "hard32_access": "forbidden_not_resolved_opened_or_hashed",
    "evaluation_access": "forbidden",
    "training_code": {
        "git_commit": code_commit,
        "tracked_worktree_clean": True,
        "trainer": {"path": trainer_code, "sha256": sha(trainer_code)},
        "launcher": {"path": launcher, "sha256": sha(launcher)},
        "launch_contract": {"path": launch_contract, "sha256": sha(launch_contract)},
        "progression_gate": {"path": gate_code, "sha256": sha(gate_code)},
    },
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
with Path(output_raw).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

{
  printf 'Launch receipt: %s\n' "${LAUNCH_RECEIPT}"
  printf 'Starting attached V10 mode=%s source_step=%s target_step=%s cursor=%s\n' \
    "${LAUNCH_MODE}" "${SOURCE_STEP}" "${TARGET_STEP}" \
    "${RESUME_SCHEDULE_CURSOR}"
  printf 'Command: '
  printf '%q ' "${train_command[@]}"
  printf '\n'
} | tee "${LOG_FILE}"

set +e
"${train_command[@]}" 2>&1 | tee -a "${LOG_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
(( pipeline_status[0] == 0 )) \
  || fail "v10_training_failed exit_code=${pipeline_status[0]} log=${LOG_FILE}"
(( pipeline_status[1] == 0 )) \
  || fail "v10_training_log_failed exit_code=${pipeline_status[1]}"
[[ -s "${OUTPUT_DIR}/training_summary.json" ]] \
  || fail "completed_training_summary_missing"
CHECKPOINT_DIR="${OUTPUT_DIR}/trainer/checkpoint-${TARGET_STEP}"
[[ -d "${CHECKPOINT_DIR}" ]] || fail "completed_checkpoint_missing"

"${VALIDATION_PYTHON_BIN}" - \
  "${COMPLETION_RECEIPT}" "${LAUNCH_RECEIPT}" "${LOG_FILE}" \
  "${OUTPUT_DIR}/training_summary.json" "${CHECKPOINT_DIR}" \
  "${TARGET_STEP}" <<'PY'
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import sys

output_raw, launch_raw, log_raw, summary_raw, checkpoint_raw, step_raw = sys.argv[1:]
def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
checkpoint = Path(checkpoint_raw).resolve()
required = (
    "delta_mem_adapter.pt", "delta_mem_config.json", "optimizer.pt",
    "scheduler.pt", "trainer_state.json", "training_protocol.json",
    "scene_state_identity_pairing_manifest.json",
)
missing = [name for name in required if not (checkpoint / name).is_file()]
if missing:
    raise SystemExit("completion checkpoint missing: " + ", ".join(missing))
rng = sorted(checkpoint.glob("rng_state*.pth"))
if not rng:
    raise SystemExit("completion checkpoint missing rng state")
step = int(step_raw)
payload = {
    "schema": "rwkv_ms_scene_memory_v10_attached_completion.v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "completed",
    "optimizer_step": step,
    "consumed_pair_presentations": step * 7,
    "cursor_formula": "global_step_times_7",
    "ignore_data_skip": False,
    "launch_receipt": {"path": str(Path(launch_raw).resolve()), "sha256": sha(Path(launch_raw))},
    "log": {"path": str(Path(log_raw).resolve()), "sha256": sha(Path(log_raw))},
    "training_summary": {"path": str(Path(summary_raw).resolve()), "sha256": sha(Path(summary_raw))},
    "checkpoint": str(checkpoint),
    "checkpoint_artifacts": {name: sha(checkpoint / name) for name in required},
    "rng_state_artifacts": {path.name: sha(path) for path in rng},
    "hard32_access": "forbidden_not_resolved_opened_or_hashed",
    "evaluation_access": "forbidden",
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
with Path(output_raw).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'Completed attached V10 optimizer checkpoint-%s: %s\n' \
  "${TARGET_STEP}" "${OUTPUT_DIR}"
