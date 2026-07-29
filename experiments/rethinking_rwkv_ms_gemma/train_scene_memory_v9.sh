#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SSD_ROOT="/run/media/xiaol/B214449214445C0B"
MODEL_PATH="${SSD_ROOT}/models/gemma/gemma-4-E4B-it"
DATA_ROOT="${SSD_ROOT}/delta_mem_data/scene_failure_state/scene_memory_v9/value14_pair28_v1"
SOURCE_LOCK="${SCRIPT_DIR}/scene_memory_v9_source_lock.json"
WARM_START_LOCK="${SCRIPT_DIR}/scene_memory_v9_v8_checkpoint56_lock.json"
RUN_ROOT="${SSD_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v9"
CACHE_ROOT="${SSD_ROOT}/delta_mem_cache/scene_memory_v9"

PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
VALIDATION_PYTHON_BIN="${VALIDATION_PYTHON_BIN:-${REPO}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-}"
TARGET_STEP="${TARGET_STEP:-}"
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
[[ "${SMOKE_RUN}" == "0" || "${SMOKE_RUN}" == "1" ]] \
  || fail "smoke_run_must_be_0_or_1 actual=${SMOKE_RUN}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "dry_run_must_be_0_or_1 actual=${DRY_RUN}"
if [[ -z "${TARGET_STEP}" ]]; then
  if [[ "${SMOKE_RUN}" == "1" ]]; then
    TARGET_STEP=1
  else
    TARGET_STEP=7
  fi
fi
[[ "${TARGET_STEP}" =~ ^[1-9][0-9]*$ ]] \
  || fail "target_step_must_be_positive actual=${TARGET_STEP}"
[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable path=${PYTHON_BIN}"
[[ -x "${VALIDATION_PYTHON_BIN}" ]] \
  || fail "validation_python_not_executable path=${VALIDATION_PYTHON_BIN}"
[[ -d "${MODEL_PATH}" && ! -L "${MODEL_PATH}" ]] \
  || fail "locked_model_missing_or_symlink path=${MODEL_PATH}"
[[ -d "${DATA_ROOT}" && ! -L "${DATA_ROOT}" ]] \
  || fail "locked_data_root_missing_or_symlink path=${DATA_ROOT}"
[[ -f "${SOURCE_LOCK}" && ! -L "${SOURCE_LOCK}" ]] \
  || fail "source_lock_missing_or_symlink path=${SOURCE_LOCK}"
[[ -f "${WARM_START_LOCK}" && ! -L "${WARM_START_LOCK}" ]] \
  || fail "warm_start_lock_missing_or_symlink path=${WARM_START_LOCK}"

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

[[ -z "${RESUME_MODE:-}" ]] \
  || fail "resume_mode_is_derived_not_user_configurable"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  [[ "${RESUME_FROM_CHECKPOINT}" != "auto" && \
     "${RESUME_FROM_CHECKPOINT}" != "latest" && \
     "${RESUME_FROM_CHECKPOINT}" != "last" ]] \
    || fail "resume_checkpoint_must_be_explicit"
  [[ "${RESUME_FROM_CHECKPOINT}" == /* ]] \
    || fail "resume_checkpoint_must_be_absolute path=${RESUME_FROM_CHECKPOINT}"
  [[ -n "${GATE_RECEIPT}" ]] \
    || fail "resume_requires_explicit_v9_gate_receipt"
  [[ "${GATE_RECEIPT}" == /* ]] \
    || fail "gate_receipt_must_be_absolute path=${GATE_RECEIPT}"
elif [[ -n "${GATE_RECEIPT}" ]]; then
  fail "fresh_launch_forbids_gate_receipt"
fi
if [[ "${SMOKE_RUN}" == "1" && -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  fail "smoke_launch_forbids_resume"
fi

run_kind="production"
if [[ "${SMOKE_RUN}" == "1" ]]; then
  run_kind="smoke"
fi
RUN_ID="scene_memory_v9_${run_kind}_${RUN_NAME}_step${TARGET_STEP}"
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

require_ssd_path() {
  local resolved
  resolved="$(realpath -m -- "$1")"
  [[ "${resolved}" == "${SSD_ROOT}" || "${resolved}" == "${SSD_ROOT}/"* ]] \
    || fail "path_must_stay_on_2t_ssd path=$1 resolved=${resolved}"
  case "/${resolved,,}/" in
    */hard32/*|*/eval/*|*/evaluation/*|*/validation/*)
      fail "hard32_or_evaluation_path_is_forbidden path=${resolved}"
      ;;
  esac
}

for locked_path in \
  "${MODEL_PATH}" "${DATA_ROOT}" "${OUTPUT_DIR}" "${LOG_DIR}" \
  "${LOG_FILE}" "${LAUNCH_RECEIPT}" "${COMPLETION_RECEIPT}" \
  "${HF_HOME_LOCKED}" "${HF_CACHE_DIR}" "${XDG_CACHE_HOME_LOCKED}" \
  "${TOKENIZED_DATASET_ROOT}" "${TMPDIR_LOCKED}" "${TORCH_CACHE_ROOT}"; do
  require_ssd_path "${locked_path}"
done
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  require_ssd_path "${RESUME_FROM_CHECKPOINT}"
  require_ssd_path "${GATE_RECEIPT}"
fi

[[ ! -e "${OUTPUT_DIR}" ]] || fail "fresh_output_collision path=${OUTPUT_DIR}"
[[ ! -e "${LOG_FILE}" ]] || fail "fresh_log_collision path=${LOG_FILE}"
[[ ! -e "${LAUNCH_RECEIPT}" ]] \
  || fail "fresh_launch_receipt_collision path=${LAUNCH_RECEIPT}"
[[ ! -e "${COMPLETION_RECEIPT}" ]] \
  || fail "fresh_completion_receipt_collision path=${COMPLETION_RECEIPT}"
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]] \
  || fail "tracked_worktree_must_be_clean_before_v9_training"
critical_tracked_files=(
  "deltamem/train/delta_sft_experimental.py"
  "deltamem/train/scene_state_generation_alignment.py"
  "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v9_data.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_data_contract.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_source_lock.json"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_warm_start.py"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_v8_checkpoint56_lock.json"
  "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_launch_contract.py"
  "experiments/rethinking_rwkv_ms_gemma/run_scene_memory_v9_gate.py"
  "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v9.sh"
)
for critical_file in "${critical_tracked_files[@]}"; do
  git -C "${REPO}" ls-files --error-unmatch -- "${critical_file}" >/dev/null 2>&1 \
    || fail "critical_v9_source_must_be_tracked path=${critical_file}"
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
export TRANSFORMERS_CACHE="${HF_HOME_LOCKED}/transformers"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_LOCKED}"
export TMPDIR="${TMPDIR_LOCKED}"
export TMP="${TMPDIR_LOCKED}"
export TEMP="${TMPDIR_LOCKED}"
export TORCH_HOME="${TORCH_CACHE_ROOT}"
export TORCH_EXTENSIONS_DIR="${TORCH_CACHE_ROOT}/extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

contract_args=(
  -m experiments.rethinking_rwkv_ms_gemma.scene_memory_v9_launch_contract
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
if [[ "${SMOKE_RUN}" == "1" ]]; then
  contract_args+=(--smoke)
fi
if ! contract_values="$("${VALIDATION_PYTHON_BIN}" "${contract_args[@]}")"; then
  fail "v9_launch_contract_validation_failed"
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
for contract_path in \
  "${TRAIN_FILE}" "${SOURCE_MANIFEST}" "${SCHEDULE_FILE}" \
  "${WARM_START_CHECKPOINT}"; do
  require_ssd_path "${contract_path}"
done

TARGET_LAYERS="$(seq -s, 0 41)"
checkpoint_args=()
if [[ "${LAUNCH_MODE}" == "warm_start" || \
      "${LAUNCH_MODE}" == "warm_start_smoke" ]]; then
  checkpoint_args=(
    --warm-start-from-checkpoint "${WARM_START_CHECKPOINT}"
    --warm-start-mode scene_memory_v9_v8_checkpoint56_adapter_only
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
  --scene-state-generation-objective-version scene_state_generation_ce_symmetric_full_gold_selected_vocab_aligned_prefix_v3
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
  --gradient-accumulation-steps 1
  --learning-rate 2e-4
  --lr-scheduler-type constant_with_warmup
  --warmup-ratio 0
  --warmup-steps 4
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

printf 'Validated V9 mode=%s source_step=%s target_step=%s cursor=%s next_pair=%s:%s\n' \
  "${LAUNCH_MODE}" "${SOURCE_STEP}" "${TARGET_STEP}" \
  "${RESUME_SCHEDULE_CURSOR}" "${NEXT_PAIR_LOW_ORDINAL}" \
  "${NEXT_PAIR_HIGH_ORDINAL}"
printf 'Locked train_sha256=%s schedule_sha256=%s next_entry=%s\n' \
  "${TRAIN_SHA256}" "${SCHEDULE_SHA256}" "${NEXT_SCHEDULE_ENTRY_SHA256}"
printf 'Hard32 and evaluation access forbidden; attached output: %s\n' "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Validated V9 training command (not started):\n'
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${RUN_ROOT}" "${LOG_DIR}" "${HF_CACHE_DIR}" "${HF_HUB_CACHE}" \
  "${XDG_CACHE_HOME}" "${TOKENIZED_DATASET_ROOT}" "${TMPDIR}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
mkdir "${OUTPUT_DIR}"
[[ -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || fail "trainer_output_must_be_empty_before_entry"

CODE_COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
TRAINER_CODE="${REPO}/deltamem/train/delta_sft_experimental.py"
LAUNCHER="${SCRIPT_DIR}/train_scene_memory_v9.sh"
LAUNCH_CONTRACT="${SCRIPT_DIR}/scene_memory_v9_launch_contract.py"
GATE_CODE="${SCRIPT_DIR}/run_scene_memory_v9_gate.py"
TRAINER_CODE_SHA256="$(sha256sum "${TRAINER_CODE}" | cut -d' ' -f1)"
LAUNCHER_SHA256="$(sha256sum "${LAUNCHER}" | cut -d' ' -f1)"
LAUNCH_CONTRACT_SHA256="$(sha256sum "${LAUNCH_CONTRACT}" | cut -d' ' -f1)"
GATE_CODE_SHA256="$(sha256sum "${GATE_CODE}" | cut -d' ' -f1)"
"${VALIDATION_PYTHON_BIN}" - \
  "${LAUNCH_RECEIPT}" "${RUN_NAME}" "${LAUNCH_MODE}" "${SOURCE_STEP}" \
  "${TARGET_STEP}" "${RESUME_FROM_CHECKPOINT}" "${RESUME_SCHEDULE_CURSOR}" \
  "${NEXT_PAIR_LOW_ORDINAL}" "${NEXT_PAIR_HIGH_ORDINAL}" \
  "${NEXT_SCHEDULE_ENTRY_SHA256}" "${OUTPUT_DIR}" "${LOG_FILE}" \
  "${GATE_AUTHORIZATION_KIND}" "${VALIDATED_GATE_RECEIPT}" \
  "${GATE_RECEIPT_FILE_SHA256}" "${GATE_RECEIPT_SHA256}" \
  "${CODE_COMMIT}" "${TRAINER_CODE}" "${TRAINER_CODE_SHA256}" \
  "${LAUNCHER}" "${LAUNCHER_SHA256}" "${LAUNCH_CONTRACT}" \
  "${LAUNCH_CONTRACT_SHA256}" "${GATE_CODE}" "${GATE_CODE_SHA256}" \
  "${SOURCE_LOCK}" "${WARM_START_LOCK}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

(
    output_raw, run_name, launch_mode, source_step, target_step,
    resume_checkpoint, cursor, pair_low, pair_high, next_entry_sha256,
    trainer_output, log_file, gate_authorization_kind, gate_receipt,
    gate_receipt_file_sha256, gate_receipt_sha256, code_commit,
    trainer_code, trainer_sha256,
    launcher, launcher_sha256, launch_contract, launch_contract_sha256,
    gate_code, gate_code_sha256, source_lock, warm_start_lock,
) = sys.argv[1:]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema": "rwkv_ms_scene_memory_v9_attached_launch.v1",
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
        "consumed_pair_steps": int(cursor),
        "next_schedule_index": int(cursor),
        "next_canonical_pair_ordinals": [int(pair_low), int(pair_high)],
        "next_entry_sha256": next_entry_sha256,
        "ignore_data_skip": False,
    },
    "progression_gate": {
        "authorization_kind": gate_authorization_kind,
        "receipt": None if gate_receipt == "not_required_fresh_start" else gate_receipt,
        "receipt_file_sha256": None if gate_receipt_file_sha256 == "not_required_fresh_start" else gate_receipt_file_sha256,
        "receipt_sha256": None if gate_receipt_sha256 == "not_required_fresh_start" else gate_receipt_sha256,
        "required": launch_mode == "resume",
    },
    "pair_execution": {
        "physical_batch_size": 1,
        "logical_reciprocal_pair_size": 2,
        "directional_exposures_per_step": 2,
    },
    "fresh_start": {
        "adapter_only_from_pinned_v8_checkpoint56": launch_mode.startswith("warm_start"),
        "optimizer": "fresh_adamw_after_adapter_load" if launch_mode.startswith("warm_start") else "exact_checkpoint_restore",
        "scheduler": "fresh_constant_with_warmup" if launch_mode.startswith("warm_start") else "exact_checkpoint_restore",
        "trainer_state": "fresh_global_step_0" if launch_mode.startswith("warm_start") else "exact_checkpoint_restore",
        "rng": "fresh_seed_42" if launch_mode.startswith("warm_start") else "exact_checkpoint_restore",
    },
    "objective": {
        "version": "scene_state_generation_ce_symmetric_full_gold_selected_vocab_aligned_prefix_v3",
        "generated_prefix_correction_weight": 0.5,
        "legacy_generated_unlikelihood_weight": 0.0,
    },
    "optimization": {
        "learning_rate": 2e-4,
        "scheduler": "constant_with_warmup",
        "warmup_steps": 4,
        "physical_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw_torch_fused",
    },
    "hard32_access": "forbidden",
    "evaluation_access": "forbidden",
    "training_code": {
        "git_commit": code_commit,
        "tracked_worktree_clean": True,
        "trainer": {"path": trainer_code, "sha256": trainer_sha256},
        "launcher": {"path": launcher, "sha256": launcher_sha256},
        "launch_contract": {
            "path": launch_contract,
            "sha256": launch_contract_sha256,
        },
        "progression_gate": {
            "path": gate_code,
            "sha256": gate_code_sha256,
        },
    },
    "locks": {
        "source": {"path": str(Path(source_lock).resolve()), "sha256": sha256_file(Path(source_lock))},
        "warm_start": {"path": str(Path(warm_start_lock).resolve()), "sha256": sha256_file(Path(warm_start_lock))},
    },
}
unsigned = dict(payload)
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
with Path(output_raw).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

{
  printf 'Launch receipt: %s\n' "${LAUNCH_RECEIPT}"
  printf 'Starting attached V9 mode=%s source_step=%s target_step=%s cursor=%s\n' \
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
  || fail "v9_training_failed exit_code=${pipeline_status[0]} log=${LOG_FILE}"
(( pipeline_status[1] == 0 )) \
  || fail "v9_training_log_failed exit_code=${pipeline_status[1]} log=${LOG_FILE}"
[[ -s "${OUTPUT_DIR}/training_summary.json" ]] \
  || fail "completed_training_summary_missing path=${OUTPUT_DIR}/training_summary.json"
CHECKPOINT_DIR="${OUTPUT_DIR}/trainer/checkpoint-${TARGET_STEP}"
[[ -d "${CHECKPOINT_DIR}" ]] \
  || fail "completed_checkpoint_missing step=${TARGET_STEP}"

"${VALIDATION_PYTHON_BIN}" - \
  "${COMPLETION_RECEIPT}" "${LAUNCH_RECEIPT}" "${LOG_FILE}" \
  "${OUTPUT_DIR}/training_summary.json" "${CHECKPOINT_DIR}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

output_raw, launch_raw, log_raw, summary_raw, checkpoint_raw = sys.argv[1:]

def sha256_file(path: Path) -> str:
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
rng_files = sorted(path for path in checkpoint.glob("rng_state*.pth") if path.is_file())
if not rng_files:
    raise SystemExit("completion checkpoint missing rng_state*.pth")
launch = Path(launch_raw).resolve()
summary = Path(summary_raw).resolve()
payload = {
    "schema": "rwkv_ms_scene_memory_v9_attached_completion.v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "completed",
    "launch_receipt": {"path": str(launch), "sha256": sha256_file(launch)},
    "log": {"path": str(Path(log_raw).resolve()), "sha256": sha256_file(Path(log_raw))},
    "training_summary": {"path": str(summary), "sha256": sha256_file(summary)},
    "checkpoint": str(checkpoint),
    "checkpoint_artifacts": {
        name: sha256_file(checkpoint / name) for name in required
    },
    "rng_state_artifacts": {
        path.name: sha256_file(path) for path in rng_files
    },
    "hard32_access": "forbidden",
    "evaluation_access": "forbidden",
}
unsigned = dict(payload)
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
with Path(output_raw).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'Completed attached V9 block at checkpoint-%s: %s\n' \
  "${TARGET_STEP}" "${OUTPUT_DIR}"
printf 'Completion receipt: %s\n' "${COMPLETION_RECEIPT}"
