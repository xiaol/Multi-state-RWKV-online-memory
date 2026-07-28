#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SSD_ROOT="/run/media/xiaol/B214449214445C0B"
MODEL_PATH="${SSD_ROOT}/models/gemma/gemma-4-E4B-it"
SOURCE_LOCK="${SCRIPT_DIR}/scene_memory_v7_source_lock.json"
RUN_ROOT="${SSD_ROOT}/delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v7"
CACHE_ROOT="${SSD_ROOT}/delta_mem_cache/scene_memory_v7"

PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
VALIDATION_PYTHON_BIN="${VALIDATION_PYTHON_BIN:-python3}"
DATASET_KIND="${DATASET_KIND:-}"
RUN_NAME="${RUN_NAME:-}"
BLOCK_STEPS="${BLOCK_STEPS:-32}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_MODE="${RESUME_MODE:-}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

[[ "${DATASET_KIND}" == "tiny2" || "${DATASET_KIND}" == "train32" ]] \
  || fail "dataset_kind_must_be_tiny2_or_train32 actual=${DATASET_KIND:-unset}"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fail "run_name_must_be_a_safe_nonempty_component actual=${RUN_NAME:-unset}"
[[ "${BLOCK_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || fail "block_steps_must_be_positive actual=${BLOCK_STEPS}"
(( BLOCK_STEPS <= 4096 )) || fail "block_steps_exceeds_safety_limit actual=${BLOCK_STEPS}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "dry_run_must_be_0_or_1 actual=${DRY_RUN}"
[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable path=${PYTHON_BIN}"
command -v "${VALIDATION_PYTHON_BIN}" >/dev/null 2>&1 \
  || fail "validation_python_not_found command=${VALIDATION_PYTHON_BIN}"
[[ -d "${MODEL_PATH}" && ! -L "${MODEL_PATH}" ]] \
  || fail "locked_model_missing_or_symlink path=${MODEL_PATH}"
[[ -f "${SOURCE_LOCK}" && ! -L "${SOURCE_LOCK}" ]] \
  || fail "source_lock_missing_or_symlink path=${SOURCE_LOCK}"

for distributed_variable in \
  WORLD_SIZE LOCAL_RANK RANK MASTER_ADDR MASTER_PORT \
  SLURM_PROCID PMI_RANK OMPI_COMM_WORLD_RANK; do
  [[ -z "${!distributed_variable:-}" ]] \
    || fail "distributed_environment_is_forbidden variable=${distributed_variable}"
done

resume_kind="fresh"
if [[ -n "${RESUME_FROM_CHECKPOINT}" || -n "${RESUME_MODE}" ]]; then
  [[ -n "${RESUME_FROM_CHECKPOINT}" ]] \
    || fail "resume_mode_requires_explicit_checkpoint"
  [[ "${RESUME_MODE}" == "extend" ]] \
    || fail "resume_mode_must_be_extend actual=${RESUME_MODE:-unset}"
  [[ "${RESUME_FROM_CHECKPOINT}" != "auto" && "${RESUME_FROM_CHECKPOINT}" != "latest" ]] \
    || fail "resume_checkpoint_must_be_explicit"
  [[ "${RESUME_FROM_CHECKPOINT}" == /* ]] \
    || fail "resume_checkpoint_must_be_absolute path=${RESUME_FROM_CHECKPOINT}"
  resume_kind="extend"
fi

OUTPUT_DIR="${RUN_ROOT}/scene_memory_v7_${DATASET_KIND}_${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/scene_memory_v7_${DATASET_KIND}_${RUN_NAME}.log"
HF_HOME_LOCKED="${CACHE_ROOT}/huggingface"
HF_CACHE_DIR="${HF_HOME_LOCKED}/datasets"
XDG_CACHE_HOME_LOCKED="${CACHE_ROOT}/xdg"
TOKENIZED_DATASET_ROOT="${CACHE_ROOT}/tokenized"
TMPDIR_LOCKED="${CACHE_ROOT}/tmp/${DATASET_KIND}_${RUN_NAME}"
TORCH_CACHE_ROOT="${CACHE_ROOT}/torch"

require_ssd_path() {
  local resolved
  resolved="$(realpath -m -- "$1")"
  [[ "${resolved}" == "${SSD_ROOT}" || "${resolved}" == "${SSD_ROOT}/"* ]] \
    || fail "path_must_stay_on_2t_ssd path=$1 resolved=${resolved}"
}

for locked_path in \
  "${MODEL_PATH}" "${OUTPUT_DIR}" "${LOG_DIR}" "${LOG_FILE}" "${HF_HOME_LOCKED}" \
  "${HF_CACHE_DIR}" "${XDG_CACHE_HOME_LOCKED}" "${TOKENIZED_DATASET_ROOT}" \
  "${TMPDIR_LOCKED}" "${TORCH_CACHE_ROOT}"; do
  require_ssd_path "${locked_path}"
done
[[ ! -e "${OUTPUT_DIR}" ]] \
  || fail "fresh_output_collision path=${OUTPUT_DIR}"
[[ ! -e "${LOG_FILE}" ]] \
  || fail "fresh_log_collision path=${LOG_FILE}"

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

if ! contract_values="$("${VALIDATION_PYTHON_BIN}" - \
    "${SOURCE_LOCK}" "${DATASET_KIND}" "${SSD_ROOT}" \
    "${RESUME_FROM_CHECKPOINT}" "${BLOCK_STEPS}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


source_lock_path = Path(sys.argv[1])
dataset_kind = sys.argv[2]
ssd_root = Path(sys.argv[3]).resolve()
resume_raw = sys.argv[4]
block_steps = int(sys.argv[5])


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_object(path: Path, description: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        fail(f"{description}_missing_or_symlink path={path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description}_invalid path={path}") from error
    if not isinstance(payload, dict):
        fail(f"{description}_must_be_object path={path}")
    return payload


def require_ssd(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != ssd_root and ssd_root not in resolved.parents:
        fail(f"{description}_must_stay_on_2t_ssd path={resolved}")
    return resolved


lock = load_object(source_lock_path, "source_lock")
if lock.get("schema") != "rwkv_ms_scene_memory_v7_source_lock.v1":
    fail("source_lock_schema_differs")
unsigned_lock = dict(lock)
recorded_lock_sha256 = unsigned_lock.pop("lock_sha256", None)
if recorded_lock_sha256 != canonical_sha256(unsigned_lock):
    fail("source_lock_self_hash_differs")
artifacts = lock.get("artifacts")
if not isinstance(artifacts, dict):
    fail("source_lock_artifacts_missing")

artifact_keys = (
    "bundle_manifest",
    dataset_kind,
    f"{dataset_kind}_rows",
    f"{dataset_kind}_pair_manifest",
    f"{dataset_kind}_source_manifest",
)
resolved: dict[str, tuple[Path, str]] = {}
for key in artifact_keys:
    binding = artifacts.get(key)
    if not isinstance(binding, dict):
        fail(f"source_lock_artifact_missing key={key}")
    path = require_ssd(Path(str(binding.get("path", ""))), key)
    expected_sha256 = str(binding.get("sha256", ""))
    if not path.is_file() or path.is_symlink():
        fail(f"locked_artifact_missing_or_symlink key={key} path={path}")
    if sha256_file(path) != expected_sha256:
        fail(f"locked_artifact_hash_differs key={key} path={path}")
    resolved[key] = (path, expected_sha256)

train_path, train_sha256 = resolved[dataset_kind]
rows_path, rows_sha256 = resolved[f"{dataset_kind}_rows"]
pair_path, pair_sha256 = resolved[f"{dataset_kind}_pair_manifest"]
source_path, source_sha256 = resolved[f"{dataset_kind}_source_manifest"]
source = load_object(source_path, "source_manifest")
if source.get("schema") != "rwkv_ms_scene_memory_v7_source.v1":
    fail("source_manifest_schema_differs")
train_partition = source.get("partitions", {}).get("train", {})
data_binding = train_partition.get("data", {})
row_binding = train_partition.get("row_manifest", {})
pair_binding = source.get("v7_pairing", {}).get("pair_manifest", {})
expected_rows = 2 if dataset_kind == "tiny2" else 32
if train_partition.get("rows") != expected_rows:
    fail("source_manifest_row_count_differs")
if Path(str(data_binding.get("path", ""))).resolve() != train_path:
    fail("source_manifest_train_path_differs")
if data_binding.get("sha256") != train_sha256:
    fail("source_manifest_train_hash_differs")
if Path(str(row_binding.get("path", ""))).resolve() != rows_path:
    fail("source_manifest_rows_path_differs")
if row_binding.get("sha256") != rows_sha256:
    fail("source_manifest_rows_hash_differs")
if Path(str(pair_binding.get("path", ""))).resolve() != pair_path:
    fail("source_manifest_pair_path_differs")
if pair_binding.get("sha256") != pair_sha256:
    fail("source_manifest_pair_hash_differs")
if sum(bool(line.strip()) for line in train_path.read_text(encoding="utf-8").splitlines()) != expected_rows:
    fail("locked_train_row_count_differs")

source_step = 0
target_max_steps = block_steps
if resume_raw:
    checkpoint = require_ssd(Path(resume_raw), "resume_checkpoint")
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        fail(f"resume_checkpoint_missing_or_symlink path={checkpoint}")
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint.name[11:].isdigit():
        fail("resume_checkpoint_must_be_checkpoint_N")
    required = (
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_protocol.json",
        "scene_state_identity_pairing_manifest.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        fail("resume_checkpoint_incomplete missing=" + ",".join(missing))
    state = load_object(checkpoint / "trainer_state.json", "trainer_state")
    protocol = load_object(checkpoint / "training_protocol.json", "training_protocol")
    try:
        source_step = int(state["global_step"])
        source_max_steps = int(state["max_steps"])
        protocol_max_steps = int(protocol["max_steps"])
        protocol_save_steps = int(protocol["save_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume_checkpoint_horizon_invalid") from error
    checkpoint_step = int(checkpoint.name[11:])
    if source_step <= 0 or not (
        source_step == source_max_steps == protocol_max_steps == checkpoint_step
    ):
        fail("resume_checkpoint_is_not_a_completed_horizon")
    if protocol_save_steps != block_steps or source_step % block_steps:
        fail("resume_checkpoint_block_size_differs")
    if protocol.get("memory_loss_mode") != "scene_state_generation_ce":
        fail("resume_checkpoint_objective_differs")
    if Path(str(protocol.get("train_file", ""))).resolve() != train_path:
        fail("resume_checkpoint_dataset_differs")
    source_identity = protocol.get("scene_state_source_manifest", {})
    if source_identity.get("file_sha256") != source_sha256:
        fail("resume_checkpoint_source_manifest_differs")
    if protocol.get("per_device_train_batch_size") != 1:
        fail("resume_checkpoint_batch_size_differs")
    if protocol.get("gradient_accumulation_steps") != 1:
        fail("resume_checkpoint_gradient_accumulation_differs")
    if protocol.get("frozen_mlp_activation_checkpointing") is not True:
        fail("resume_checkpoint_mlp_checkpointing_differs")
    target_max_steps = source_step + block_steps

fields = (
    str(train_path),
    train_sha256,
    str(source_path),
    source_sha256,
    str(pair_path),
    pair_sha256,
    str(source_step),
    str(target_max_steps),
)
if any("\t" in field or "\n" in field for field in fields):
    fail("contract_value_contains_control_character")
print("\t".join(fields))
PY
)"; then
  fail "v7_launch_contract_validation_failed"
fi

IFS=$'\t' read -r \
  TRAIN_FILE TRAIN_SHA256 SOURCE_MANIFEST SOURCE_MANIFEST_SHA256 \
  PAIR_MANIFEST PAIR_MANIFEST_SHA256 SOURCE_STEP MAX_STEPS \
  <<<"${contract_values}"

TARGET_LAYERS="$(seq -s, 0 41)"
resume_args=()
initial_adapter_args=(--initial-adapter-output-dir "${OUTPUT_DIR}/initial_adapter")
if [[ "${resume_kind}" == "extend" ]]; then
  resume_args=(
    --resume-from-checkpoint "$(realpath -- "${RESUME_FROM_CHECKPOINT}")"
    --resume-mode extend
  )
  initial_adapter_args=()
fi

train_args=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  "${initial_adapter_args[@]}"
  "${resume_args[@]}"
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
  --warmup-ratio 0.0625
  --weight-decay 0
  --optim adamw_torch_fused
  --num-train-epochs 1
  --max-steps "${MAX_STEPS}"
  --logging-steps 1
  --save-steps "${BLOCK_STEPS}"
  --save-total-limit 1
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
train_command=("${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}")

printf 'Validated V7 dataset=%s train_sha256=%s source_sha256=%s pair_sha256=%s\n' \
  "${DATASET_KIND}" "${TRAIN_SHA256}" "${SOURCE_MANIFEST_SHA256}" \
  "${PAIR_MANIFEST_SHA256}"
printf 'Locked mode=%s source_step=%s block_steps=%s target_max_steps=%s output=%s\n' \
  "${resume_kind}" "${SOURCE_STEP}" "${BLOCK_STEPS}" "${MAX_STEPS}" "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Validated V7 training command (not started):\n'
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${RUN_ROOT}" "${LOG_DIR}" "${HF_CACHE_DIR}" "${HF_HUB_CACHE}" \
  "${XDG_CACHE_HOME}" "${TOKENIZED_DATASET_ROOT}" "${TMPDIR}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
mkdir "${OUTPUT_DIR}"

{
  printf 'Starting V7 dataset=%s mode=%s source_step=%s target_max_steps=%s\n' \
    "${DATASET_KIND}" "${resume_kind}" "${SOURCE_STEP}" "${MAX_STEPS}"
  printf 'Command: '
  printf '%q ' "${train_command[@]}"
  printf '\n'
} | tee "${LOG_FILE}"

set +e
"${train_command[@]}" 2>&1 | tee -a "${LOG_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
(( pipeline_status[0] == 0 )) \
  || fail "v7_training_failed exit_code=${pipeline_status[0]} log=${LOG_FILE}"
(( pipeline_status[1] == 0 )) \
  || fail "v7_training_log_failed exit_code=${pipeline_status[1]} log=${LOG_FILE}"
[[ -s "${OUTPUT_DIR}/training_summary.json" ]] \
  || fail "completed_training_summary_missing path=${OUTPUT_DIR}/training_summary.json"
[[ -d "${OUTPUT_DIR}/trainer/checkpoint-${MAX_STEPS}" ]] \
  || fail "completed_checkpoint_missing step=${MAX_STEPS}"
printf 'Completed V7 block at checkpoint-%s: %s\n' "${MAX_STEPS}" "${OUTPUT_DIR}"
