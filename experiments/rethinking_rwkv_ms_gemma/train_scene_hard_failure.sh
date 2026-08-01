#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODEL_PATH="/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"
DATA_ROOT="/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/scene_hard_failure_curriculum_base64_pairs16_v1"
TRAIN_FILE="${DATA_ROOT}/train.jsonl"
SOURCE_MANIFEST="${DATA_ROOT}/source_manifest.json"
SOURCE_MANIFEST_SHA256="a3b1e0a255f2e7440971e81337d9648a1dbf96da8a1211aa3e103f2d01f052d8"
SOURCE_LOCK="${SCRIPT_DIR}/scene_hard_failure_source_lock.json"
RUN_ROOT="/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory/scene_hard_failure"
CACHE_ROOT="/run/media/xiaol/B214449214445C0B/delta_mem_cache/scene_hard_failure"
HF_CACHE_DIR="/run/media/xiaol/B214449214445C0B/huggingface_cache"
RUN_MODE="${RUN_MODE:-smoke}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

for variable in WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT; do
  [[ -z "${!variable+x}" ]] \
    || fail "distributed_environment_is_forbidden variable=${variable}"
done
while IFS='=' read -r variable _; do
  case "${variable}" in
    HARD32*|VALIDATION*|TEST*|BENCHMARK*)
      fail "protected_evaluation_environment_is_forbidden variable=${variable}"
      ;;
  esac
done < <(env)

[[ -x "${PYTHON_BIN}" ]] || fail "python_missing path=${PYTHON_BIN}"
[[ "${RUN_MODE}" == "smoke" || "${RUN_MODE}" == "production" ]] \
  || fail "RUN_MODE must be smoke or production"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "DRY_RUN must be 0 or 1"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fail "RUN_NAME contains unsupported characters"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  MAX_STEPS=1
  SAVE_TOTAL_LIMIT=1
  RUN_KIND="one_pair_real_update"
else
  MAX_STEPS=64
  SAVE_TOTAL_LIMIT=64
  RUN_KIND="four_cycle_pair64"
fi

OUTPUT_DIR="${RUN_ROOT}/scene_hard_failure_${RUN_KIND}_${RUN_NAME}_step${MAX_STEPS}"
INITIAL_ADAPTER_DIR="${OUTPUT_DIR}/initial_adapter"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/$(basename -- "${OUTPUT_DIR}").log"
COMPLETION_RECEIPT="${LOG_DIR}/$(basename -- "${OUTPUT_DIR}").completion.json"
TARGET_LAYERS="$(seq -s, 0 41)"

[[ ! -e "${OUTPUT_DIR}" ]] || fail "fresh_output_already_exists path=${OUTPUT_DIR}"
[[ ! -e "${LOG_FILE}" ]] || fail "fresh_log_already_exists path=${LOG_FILE}"
[[ ! -e "${COMPLETION_RECEIPT}" ]] \
  || fail "fresh_completion_receipt_already_exists path=${COMPLETION_RECEIPT}"

"${PYTHON_BIN}" - "${RUN_MODE}" "${OUTPUT_DIR}" "${CACHE_ROOT}" "${DRY_RUN}" <<'PY'
from pathlib import Path
import sys

from experiments.rethinking_rwkv_ms_gemma import scene_hard_failure_train_contract as contract

mode, output_dir, cache_root, dry_run = sys.argv[1:]
smoke = mode == "smoke"
values = {
    "objective_version": contract.OBJECTIVE_VERSION,
    "max_steps": 1 if smoke else contract.TOTAL_OPTIMIZER_STEPS,
    "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
    "save_steps": contract.SAVE_STEPS,
    "save_total_limit": 1 if smoke else contract.SAVE_TOTAL_LIMIT,
    "target_layers": contract.TARGET_LAYERS,
    "rank": contract.RANK,
    "alpha": contract.ALPHA,
    "delta_heads": contract.DELTA_HEADS,
    "rwkv_ms_num_states": contract.RWKV_MS_NUM_STATES,
    "rwkv_ms_semantics_version": contract.RWKV_MS_SEMANTICS_VERSION,
    "rwkv_ms_chunk_size": contract.RWKV_MS_CHUNK_SIZE,
    "rwkv_ms_boundary_mode": contract.RWKV_MS_BOUNDARY_MODE,
    "state_reset_per_row": contract.STATE_RESET_PER_ROW,
    "episode_read_write_enabled": contract.READ_SIDE_WRITES_ENABLED,
    "memory_fusion_mode": contract.MEMORY_FUSION_MODE,
    "memory_fusion_placement": contract.MEMORY_FUSION_PLACEMENT,
    "per_device_train_batch_size": 1,
    "memory_kl_weight": 0.0,
    "memory_base_kl_weight": 0.0,
    "validation_split_ratio": 0.0,
    "smoke": smoke,
    "train_file": contract.TRAIN_FILE,
    "data_root": contract.DATA_ROOT,
    "output_dir": Path(output_dir),
    "cache_root": Path(cache_root),
    "argv": (),
    "lineage": {
        "source_checkpoint": None,
        "optimizer_state_imported": False,
        "scheduler_state_imported": False,
        "trainer_state_imported": False,
        "rng_state_imported": False,
    },
}
contract.validate_launch_contract(values)
if dry_run != "1":
    contract.validate_critical_worktree()
PY

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
  --scene-state-generation-objective-version scene_state_generation_ce_symmetric_cached_prefix_identity_hard_failure_v1
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
  --per-device-eval-batch-size 1
  --gradient-accumulation-steps 1
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
if [[ "${RUN_MODE}" == "smoke" ]]; then
  train_args+=(--scene-state-hard-failure-one-pair-smoke)
fi
train_command=("${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}")

printf 'Validated hard-scene mode=%s optimizer_steps=%s accumulation=1 checkpoints=%s\n' \
  "${RUN_MODE}" "${MAX_STEPS}" "${SAVE_TOTAL_LIMIT}"
printf 'Fresh adapter=%s objective=%s\n' \
  "${INITIAL_ADAPTER_DIR}" \
  'scene_state_generation_ce_symmetric_cached_prefix_identity_hard_failure_v1'

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p -- "${LOG_DIR}" "${CACHE_ROOT}" "${HF_CACHE_DIR}"
set +e
(
  cd -- "${REPO_ROOT}"
  "${train_command[@]}"
) 2>&1 | tee -- "${LOG_FILE}"
trainer_status=${PIPESTATUS[0]}
tee_status=${PIPESTATUS[1]}
set -e
[[ "${trainer_status}" == "0" ]] \
  || fail "trainer_failed status=${trainer_status} log=${LOG_FILE}"
[[ "${tee_status}" == "0" ]] \
  || fail "tee_failed status=${tee_status} log=${LOG_FILE}"

audit_args=()
if [[ "${RUN_MODE}" == "smoke" ]]; then
  audit_args+=(--smoke)
fi
for checkpoint_step in $(seq 1 "${MAX_STEPS}"); do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scene_hard_failure_run_audit.py" \
    --run-root "${OUTPUT_DIR}" \
    --checkpoint-step "${checkpoint_step}" \
    "${audit_args[@]}"
done

"${PYTHON_BIN}" - \
  "${OUTPUT_DIR}" "${LOG_FILE}" "${COMPLETION_RECEIPT}" \
  "${RUN_MODE}" "${MAX_STEPS}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

output_dir, log_file, receipt_file, run_mode, max_steps = sys.argv[1:]
root = Path(output_dir).resolve()
log = Path(log_file).resolve()
receipt = Path(receipt_file).resolve()
steps = list(range(1, int(max_steps) + 1))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

audits = []
for step in steps:
    path = root / "trainer" / f"checkpoint-{step}" / "scene_hard_failure_checkpoint_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_optimizer_step") != step:
        raise SystemExit(f"checkpoint audit step differs: {step}")
    audits.append({"step": step, "path": str(path), "sha256": sha256_file(path)})

payload = {
    "schema": "rwkv_ms_scene_hard_failure_completion.v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "run_mode": run_mode,
    "run_root": str(root),
    "global_step": int(max_steps),
    "checkpoint_steps": steps,
    "checkpoint_audits": audits,
    "log": {"path": str(log), "sha256": sha256_file(log)},
    "training_complete": True,
    "evaluation_accessed": False,
}
unsigned = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
payload["receipt_sha256"] = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
receipt.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{receipt.name}.", dir=receipt.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, receipt)
finally:
    temporary.unlink(missing_ok=True)
print(f"completion_receipt={receipt} sha256={payload['receipt_sha256']}")
PY
