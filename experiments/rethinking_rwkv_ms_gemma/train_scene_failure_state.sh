#!/usr/bin/env bash
set -euo pipefail

# Fresh all-layer RWKV-MS run for scene failures mined only from the official
# training split. The paired-data manifest is a required provenance boundary.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaol/X/delta-Mem/.venv/bin/python}"
VALIDATION_PYTHON_BIN="${VALIDATION_PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it}"

: "${TRAIN_FILE:?Set TRAIN_FILE to the prepared scene-failure train.jsonl}"
: "${TRAIN_SHA256:?Set TRAIN_SHA256 to the lowercase SHA-256 of TRAIN_FILE}"

PAIR_MANIFEST="${PAIR_MANIFEST:-$(dirname -- "${TRAIN_FILE}")/manifest.json}"
RUN_ROOT="${RUN_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory}"
RUN_MODE="${RUN_MODE:-production}"
case "${RUN_MODE}" in
  production)
    MAX_STEPS=128
    WARMUP_RATIO=0.0625
    DEFAULT_RUN_NAME=scene_failure_state_all42_qo_r4_n32_p4_lr5e4
    ;;
  smoke)
    MAX_STEPS=1
    WARMUP_RATIO=0
    DEFAULT_RUN_NAME=scene_failure_state_all42_qo_r4_n32_smoke1_lr5e4
    ;;
  *)
    printf 'ERROR: invalid_run_mode value=%s expected=production,smoke\n' "${RUN_MODE}" >&2
    exit 2
    ;;
esac
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${RUN_NAME}}"
INITIAL_ADAPTER_DIR="${OUTPUT_DIR}/initial_adapter"
LAUNCH_MANIFEST="${OUTPUT_DIR}/launch_manifest.json"
LOG_FILE="${LOG_FILE:-${RUN_ROOT}/${RUN_NAME}.log}"
CACHE_ROOT="${CACHE_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_cache}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${CACHE_ROOT}/huggingface/datasets}"
TOKENIZED_DATASET_ROOT="${TOKENIZED_DATASET_ROOT:-/run/media/xiaol/B214449214445C0B/delta_mem_tokenized/scene_failure_state/${TRAIN_SHA256:0:16}}"
DRY_RUN="${DRY_RUN:-0}"
DATA_SEED="${DATA_SEED:-42}"

EXPECTED_TARGET_LAYERS="$(seq -s, 0 41)"
TARGET_LAYERS="${TARGET_LAYERS:-${EXPECTED_TARGET_LAYERS}}"
DELTA_HEADS="${DELTA_HEADS:-q,o}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable path=${PYTHON_BIN}"
command -v "${VALIDATION_PYTHON_BIN}" >/dev/null 2>&1 \
  || fail "validation_python_not_found command=${VALIDATION_PYTHON_BIN}"
[[ -d "${REPO}/deltamem" ]] || fail "repository_missing path=${REPO}"
[[ -d "${MODEL_PATH}" ]] || fail "model_missing path=${MODEL_PATH}"
[[ -f "${TRAIN_FILE}" ]] || fail "training_file_missing path=${TRAIN_FILE}"
[[ -f "${PAIR_MANIFEST}" ]] || fail "pair_manifest_missing path=${PAIR_MANIFEST}"
[[ "${TRAIN_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid_train_sha256 expected=64_lowercase_hex"
[[ "${TARGET_LAYERS}" == "${EXPECTED_TARGET_LAYERS}" ]] \
  || fail "target_layers_must_be_all_42 actual=${TARGET_LAYERS}"
[[ "${DELTA_HEADS}" == "q,o" ]] \
  || fail "delta_heads_must_be_q_o actual=${DELTA_HEADS}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || fail "invalid_dry_run value=${DRY_RUN} expected=0,1"
[[ "${DATA_SEED}" =~ ^[0-9]+$ ]] \
  || fail "invalid_data_seed value=${DATA_SEED} expected=nonnegative_integer"
if [[ "${RUN_MODE}" == "production" && "${DATA_SEED}" != "42" ]]; then
  fail "production_data_seed_must_be_42 actual=${DATA_SEED}"
fi
[[ -z "${RESUME_FROM_CHECKPOINT:-}" ]] \
  || fail "resume_is_forbidden_for_fresh_scene_failure_run"
[[ -z "${WARM_START_FROM_CHECKPOINT:-}" ]] \
  || fail "warm_start_is_forbidden_for_fresh_scene_failure_run"

if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
  fail "output_path_is_not_directory path=${OUTPUT_DIR}"
fi
case "$(realpath -m "${LOG_FILE}")" in
  "$(realpath -m "${OUTPUT_DIR}")"/*)
    fail "log_file_must_be_outside_output_dir path=${LOG_FILE}"
    ;;
esac
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  fail "output_directory_must_be_empty path=${OUTPUT_DIR}"
fi

actual_sha256="$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')"
[[ "${actual_sha256}" == "${TRAIN_SHA256}" ]] \
  || fail "dataset_checksum_mismatch expected=${TRAIN_SHA256} actual=${actual_sha256}"

if ! validation_summary="$(
  "${VALIDATION_PYTHON_BIN}" - \
    "${PAIR_MANIFEST}" "${TRAIN_FILE}" "${TRAIN_SHA256}" "${MODEL_PATH}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


BASE_RUNTIME_ARTIFACT_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def model_artifact_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def local_base_model_artifacts(model_path: Path) -> dict[str, object]:
    weight_paths = sorted(
        (path for path in model_path.rglob("*.safetensors") if path.is_file()),
        key=lambda path: path.relative_to(model_path).as_posix(),
    )
    require(bool(weight_paths), "MODEL_PATH contains no safetensors weights")
    runtime_paths = sorted(
        (
            path
            for path in model_path.rglob("*")
            if path.is_file()
            and (
                path.name in BASE_RUNTIME_ARTIFACT_NAMES
                or path.name.startswith("tokenizer.")
                or path.name.startswith("chat_template.")
            )
            and path.suffix != ".safetensors"
        ),
        key=lambda path: path.relative_to(model_path).as_posix(),
    )
    runtime_names = {path.name for path in runtime_paths}
    require("config.json" in runtime_names, "MODEL_PATH runtime artifacts omit config.json")
    require(
        bool({"tokenizer.json", "tokenizer.model", "spiece.model"} & runtime_names),
        "MODEL_PATH is missing a tokenizer payload",
    )
    weights = [model_artifact_record(path, root=model_path) for path in weight_paths]
    runtime = [model_artifact_record(path, root=model_path) for path in runtime_paths]
    aggregate_payload = {"weights": weights, "runtime_artifacts": runtime}
    return {
        "root": str(model_path),
        **aggregate_payload,
        "aggregate_sha256": canonical_json_sha256(aggregate_payload),
    }


def declared_path(raw_path: object, manifest_dir: Path, label: str) -> Path:
    require(isinstance(raw_path, str) and bool(raw_path), f"{label}.path is missing")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def read_jsonl_lines(path: Path) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_line = line.rstrip("\n")
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            rows.append((raw_line, payload))
    return rows


def validate() -> int:
    manifest_path = Path(sys.argv[1]).expanduser().resolve()
    train_path = Path(sys.argv[2]).expanduser().resolve()
    expected_sha256 = sys.argv[3]
    model_path = Path(sys.argv[4]).expanduser().resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None, "invalid expected SHA-256")

    model_config_path = model_path / "config.json"
    require(model_config_path.is_file(), "MODEL_PATH/config.json is missing")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    text_config = model_config.get("text_config", model_config)
    require(isinstance(text_config, dict), "model text_config is invalid")
    require(text_config.get("num_hidden_layers") == 42, "MODEL_PATH must contain a 42-layer text model")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "pair manifest must be a JSON object")
    require(manifest.get("schema") == "rwkv_ms_scene_failure_pairs.v1", "unexpected pair manifest schema")
    require(manifest.get("task") == "scene-v4-current", "unexpected pair manifest task")

    contract = manifest.get("contract")
    require(isinstance(contract, dict), "pair manifest contract is missing")
    require(contract.get("failure_mining_split") == "train", "failure mining must use train")
    require(contract.get("holdout_source_split") == "val", "holdout must use val")
    require("never emitted" in str(contract.get("test_policy", "")), "test policy must forbid emission")
    require(contract.get("candidate_count") == 64, "contract.candidate_count must be 64")
    require(contract.get("train_failure_count") == 32, "contract.train_failure_count must be 32")
    episode = contract.get("episode_contract")
    require(isinstance(episode, dict), "episode contract is missing")
    require(episode.get("messages") == ["system", "user", "assistant"], "unexpected episode roles")
    require(episode.get("episode_recent_messages") == 0, "episode_recent_messages must be zero")

    config = manifest.get("config")
    require(isinstance(config, dict), "pair manifest config is missing")
    require(config.get("candidate_count") == 64, "config.candidate_count must be 64")
    require(config.get("train_failure_count") == 32, "config.train_failure_count must be 32")

    validation = manifest.get("validation")
    require(isinstance(validation, dict), "pair manifest validation is missing")
    required_true = (
        "row_sha256_pairwise_disjoint",
        "exact_user_prompt_sha256_pairwise_disjoint",
        "all_base_records_joined_to_train_by_row_sha256",
        "base_gold_matches_train_source",
        "train_holdout_row_sha256_disjoint",
        "train_holdout_exact_user_prompt_sha256_disjoint",
        "output_rows_preserve_source_serialization",
        "output_rows_have_exactly_three_messages",
        "candidate_count_matches_protocol",
        "train_failure_count_matches_protocol",
        "base_records_match_producer_selection",
        "base_records_share_producer_fingerprint",
        "producer_summary_complete",
    )
    for field in required_true:
        require(validation.get(field) is True, f"validation.{field} must be true")
    require(validation.get("holdout_selection_uses_model_output") is False, "holdout selection consulted model output")
    require(
        validation.get("failure_selection_uses_eval_record_order") is False,
        "failure selection consulted evaluation record order",
    )
    require(validation.get("test_rows_emitted") == 0, "test rows must never be emitted")

    sources = manifest.get("sources")
    require(isinstance(sources, dict), "source provenance is missing")
    for split in ("train", "val", "test"):
        require(isinstance(sources.get(split), dict), f"sources.{split} is missing")
    require(sources["train"].get("emitted_for_training") is True, "train source is not marked for training")
    require(sources["val"].get("emitted_for_training") is False, "validation source is marked for training")
    require(sources["test"].get("emitted_for_training") is False, "test source is marked for training")
    require(sources["test"].get("emitted_for_holdout") is False, "test source is marked for holdout")

    partitions = manifest.get("partitions")
    require(isinstance(partitions, dict), "partitions are missing")
    train_partition = partitions.get("train")
    holdout_partition = partitions.get("holdout")
    require(isinstance(train_partition, dict), "train partition is missing")
    require(isinstance(holdout_partition, dict), "holdout partition is missing")
    require(train_partition.get("source_split") == "train", "train partition source_split must be train")
    require(holdout_partition.get("source_split") == "val", "holdout partition source_split must be val")
    expected_rows = train_partition.get("rows")
    require(
        isinstance(expected_rows, int)
        and not isinstance(expected_rows, bool)
        and expected_rows == 32,
        "train partition must contain exactly 32 failures",
    )

    data = train_partition.get("data")
    require(isinstance(data, dict), "train partition data is missing")
    declared_train_path = declared_path(data.get("path"), manifest_path.parent, "partitions.train.data")
    require(declared_train_path == train_path, "TRAIN_FILE differs from partitions.train.data.path")
    require(data.get("sha256") == expected_sha256, "TRAIN_SHA256 differs from manifest train hash")
    require(sha256_file(train_path) == expected_sha256, "TRAIN_FILE hash differs after manifest validation")

    train_rows = read_jsonl_lines(train_path)
    require(len(train_rows) == expected_rows, "train JSONL row count differs from manifest")
    roles = ["system", "user", "assistant"]
    row_hashes: list[str] = []
    prompt_hashes: list[str] = []
    for row_number, (raw_line, payload) in enumerate(train_rows, start=1):
        require(isinstance(payload, dict) and set(payload) == {"messages"}, f"train row {row_number} is not messages-only")
        messages = payload.get("messages")
        require(isinstance(messages, list) and len(messages) == 3, f"train row {row_number} must have three messages")
        require(all(isinstance(message, dict) for message in messages), f"train row {row_number} has a non-object message")
        require([message.get("role") for message in messages] == roles, f"train row {row_number} has invalid roles")
        require(all(isinstance(message.get("content"), str) for message in messages), f"train row {row_number} has invalid content")
        row_hashes.append(sha256_text(raw_line))
        prompt_hashes.append(sha256_text(messages[1]["content"]))
    require(len(set(row_hashes)) == len(row_hashes), "train rows contain duplicate hashes")
    require(len(set(prompt_hashes)) == len(prompt_hashes), "train rows contain duplicate prompts")
    require(train_partition.get("row_hashes_sha256") == canonical_json_sha256(row_hashes), "train row hash aggregate differs")
    require(train_partition.get("prompt_hashes_sha256") == canonical_json_sha256(prompt_hashes), "train prompt hash aggregate differs")

    row_manifest = train_partition.get("row_manifest")
    require(isinstance(row_manifest, dict), "train row manifest is missing")
    row_manifest_path = declared_path(row_manifest.get("path"), manifest_path.parent, "partitions.train.row_manifest")
    require(row_manifest_path.is_file(), "train row manifest file is missing")
    require(row_manifest.get("sha256") == sha256_file(row_manifest_path), "train row manifest hash differs")
    row_records = read_jsonl_lines(row_manifest_path)
    require(len(row_records) == expected_rows, "train row manifest count differs")
    manifest_row_hashes: list[str] = []
    manifest_prompt_hashes: list[str] = []
    for row_number, (_, record) in enumerate(row_records, start=1):
        require(isinstance(record, dict), f"train manifest row {row_number} is not an object")
        require(record.get("partition") == "train", f"train manifest row {row_number} has invalid partition")
        require(record.get("source_split") == "train", f"train manifest row {row_number} has non-train source_split")
        manifest_row_hashes.append(record.get("row_sha256"))
        manifest_prompt_hashes.append(record.get("prompt_sha256"))
    require(manifest_row_hashes == row_hashes, "train data and row manifest hashes differ")
    require(manifest_prompt_hashes == prompt_hashes, "train data and row manifest prompt hashes differ")

    base_eval = manifest.get("base_train_evaluation")
    require(isinstance(base_eval, dict), "base train evaluation provenance is missing")
    require(base_eval.get("selected_task_records") == 64, "base evaluation must contain exactly 64 selected task rows")
    eligible_failures = base_eval.get("eligible_failures")
    require(
        isinstance(eligible_failures, int)
        and not isinstance(eligible_failures, bool)
        and eligible_failures >= 32,
        "base evaluation must contain at least 32 eligible failures",
    )
    require(base_eval.get("selected_failures") == expected_rows, "selected failure count differs from train rows")

    producer_bundle = base_eval.get("producer_bundle")
    require(isinstance(producer_bundle, dict), "base producer bundle is missing")
    producer_model = producer_bundle.get("base_model")
    require(isinstance(producer_model, dict), "base producer model identity is missing")
    producer_model_path = Path(str(producer_model.get("path", ""))).expanduser().resolve()
    require(producer_model_path == model_path, "MODEL_PATH differs from the failure-mining base model")
    actual_model_artifacts = local_base_model_artifacts(model_path)
    require(
        producer_model.get("artifact_aggregate_sha256")
        == actual_model_artifacts["aggregate_sha256"],
        "MODEL_PATH artifact aggregate differs from the failure-mining base model",
    )

    producer_manifest_record = producer_bundle.get("manifest")
    require(isinstance(producer_manifest_record, dict), "base producer manifest record is missing")
    producer_manifest_path = declared_path(
        producer_manifest_record.get("path"),
        manifest_path.parent,
        "base_train_evaluation.producer_bundle.manifest",
    )
    require(producer_manifest_path.is_file(), "base producer manifest file is missing")
    require(
        producer_manifest_record.get("sha256") == sha256_file(producer_manifest_path),
        "base producer manifest SHA-256 differs",
    )
    producer_manifest = json.loads(producer_manifest_path.read_text(encoding="utf-8"))
    require(isinstance(producer_manifest, dict), "base producer manifest must be an object")
    require(
        producer_manifest.get("schema") == "rwkv_ms_scene_train_base_eval.v1",
        "unexpected base producer manifest schema",
    )
    producer_fingerprint_payload = producer_manifest.get("fingerprint_payload")
    require(
        isinstance(producer_fingerprint_payload, dict),
        "base producer fingerprint payload is missing",
    )
    require(
        producer_manifest.get("fingerprint")
        == canonical_json_sha256(producer_fingerprint_payload),
        "base producer fingerprint payload differs from its fingerprint",
    )
    manifest_model_path = Path(
        str(producer_fingerprint_payload.get("base_model", ""))
    ).expanduser().resolve()
    require(manifest_model_path == model_path, "base producer manifest model path differs")
    manifest_model_artifacts = producer_fingerprint_payload.get("base_model_artifacts")
    require(
        isinstance(manifest_model_artifacts, dict),
        "base producer manifest model artifacts are missing",
    )
    require(
        manifest_model_artifacts.get("aggregate_sha256")
        == actual_model_artifacts["aggregate_sha256"],
        "base producer manifest model aggregate differs from MODEL_PATH",
    )
    return expected_rows


try:
    rows = validate()
except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
print(f"rows={rows} schema=rwkv_ms_scene_failure_pairs.v1")
PY
)"; then
  fail "training_data_contract_invalid manifest=${PAIR_MANIFEST}"
fi

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

train_args=(
  --model-path "${MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --initial-adapter-output-dir "${INITIAL_ADAPTER_DIR}"
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
  --delta-heads "${DELTA_HEADS}"
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
  --max-write-length 1280
  --no-episode-read-write-enabled
  --memory-loss-mode context_dropout_ce
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
  --save-steps 32
  --save-total-limit 4
  --eval-steps 1000
  --validation-split-ratio 0
  --no-load-best-model-at-end
  --dataset-num-proc 1
  --dataloader-num-workers 0
  --frozen-mlp-activation-checkpointing
  --seed 42
  --data-seed "${DATA_SEED}"
  --tf32
  --log-delta-debug-stats
  --rankwise-gates
)
train_command=("${PYTHON_BIN}" -m deltamem.train.delta_sft "${train_args[@]}")

write_launch_manifest() {
  "${VALIDATION_PYTHON_BIN}" - \
    "${LAUNCH_MANIFEST}" \
    "${RUN_MODE}" \
    "${SCRIPT_DIR}/train_scene_failure_state.sh" \
    "${REPO}/deltamem/train/delta_sft.py" \
    "${REPO}/deltamem/train/delta_sft_experimental.py" \
    "${REPO}/deltamem/core/delta.py" \
    "${REPO}/deltamem/core/delta_impl.py" \
    "${REPO}/deltamem/core/hrm_rwkv7.py" \
    "${REPO}/deltamem/core/backbone_compat.py" \
    "${REPO}/deltamem/kernels/affine_scan.py" \
    "${REPO}/deltamem/chat_templates.py" \
    "${TRAIN_FILE}" \
    "${PAIR_MANIFEST}" \
    "${MODEL_PATH}" \
    "${INITIAL_ADAPTER_DIR}" \
    "32" \
    "${MAX_STEPS}" \
    "${WARMUP_RATIO}" \
    "${train_command[@]}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


manifest_path = Path(sys.argv[1]).expanduser().resolve()
run_mode = sys.argv[2]
launcher_path = Path(sys.argv[3]).expanduser().resolve()
trainer_entrypoint = Path(sys.argv[4]).expanduser().resolve()
trainer_implementation = Path(sys.argv[5]).expanduser().resolve()
behavior_source_labels = (
    "delta_entrypoint",
    "delta_implementation",
    "rwkv_ms_core",
    "backbone_compatibility",
    "affine_scan",
    "chat_templates",
)
behavior_sources = {
    label: Path(sys.argv[6 + index]).expanduser().resolve()
    for index, label in enumerate(behavior_source_labels)
}
train_path = Path(sys.argv[12]).expanduser().resolve()
pair_manifest_path = Path(sys.argv[13]).expanduser().resolve()
model_path = Path(sys.argv[14]).expanduser().resolve()
model_config_path = model_path / "config.json"
initial_adapter_dir = Path(sys.argv[15]).expanduser().resolve()
train_rows = int(sys.argv[16])
max_steps = int(sys.argv[17])
warmup_ratio_raw = sys.argv[18]
command = sys.argv[19:]
if manifest_path.exists():
    raise SystemExit(f"Launch manifest already exists: {manifest_path}")
for required_path in (
    launcher_path,
    trainer_entrypoint,
    trainer_implementation,
    *behavior_sources.values(),
    train_path,
    pair_manifest_path,
):
    if not required_path.is_file():
        raise SystemExit(f"Launch provenance input is missing: {required_path}")
if not model_path.is_dir():
    raise SystemExit(f"Launch provenance model directory is missing: {model_path}")
if not model_config_path.is_file():
    raise SystemExit(f"Launch provenance model config is missing: {model_config_path}")

model_weight_paths = sorted(
    (
        path
        for path in model_path.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or (path.name.startswith("pytorch_model") and path.name.endswith(".bin"))
        )
    ),
    key=lambda path: path.name,
)
if not model_weight_paths:
    raise SystemExit(
        "Launch provenance requires at least one local .safetensors or pytorch_model*.bin weight file"
    )
model_weight_files = [
    {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(model_path).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for path in model_weight_paths
]
model_weight_content = [
    {
        "relative_path": record["relative_path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }
    for record in model_weight_files
]
if train_rows != 32:
    raise SystemExit(f"Launch manifest requires exactly 32 train rows, got {train_rows}")
expected_max_steps = {"production": 128, "smoke": 1}.get(run_mode)
if expected_max_steps is None:
    raise SystemExit(f"Unsupported run mode for launch provenance: {run_mode}")
if max_steps != expected_max_steps:
    raise SystemExit(
        f"Launch max_steps differs from locked {run_mode} horizon: "
        f"expected={expected_max_steps} actual={max_steps}"
    )
expected_warmup_ratio = {"production": "0.0625", "smoke": "0"}[run_mode]
if warmup_ratio_raw != expected_warmup_ratio:
    raise SystemExit(
        f"Launch warmup ratio differs from locked {run_mode} protocol: "
        f"expected={expected_warmup_ratio} actual={warmup_ratio_raw}"
    )
warmup_ratio = 0 if warmup_ratio_raw == "0" else float(warmup_ratio_raw)
effective_passes = (
    max_steps // train_rows
    if max_steps % train_rows == 0
    else max_steps / train_rows
)
if run_mode == "production" and effective_passes != 4:
    raise SystemExit(
        f"Production launch requires exactly 4 effective passes, got {effective_passes}"
    )
payload = {
    "schema": "rwkv_ms_scene_failure_launch.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_mode": run_mode,
    "fresh_run": True,
    "train_rows": train_rows,
    "max_steps": max_steps,
    "warmup_ratio": warmup_ratio,
    "effective_passes": effective_passes,
    "production_reference": {
        "max_steps": 128,
        "warmup_ratio": 0.0625,
        "save_steps": 32,
        "checkpoint_steps": [32, 64, 96, 128],
        "effective_passes": 4,
    },
    "output_dir": str(manifest_path.parent),
    "initial_adapter": {
        "required": True,
        "path": str(initial_adapter_dir),
        "expected_global_step": 0,
    },
    "artifacts": {
        "launcher": {"path": str(launcher_path), "sha256": sha256_file(launcher_path)},
        "trainer_entrypoint": {
            "path": str(trainer_entrypoint),
            "sha256": sha256_file(trainer_entrypoint),
        },
        "trainer_implementation": {
            "path": str(trainer_implementation),
            "sha256": sha256_file(trainer_implementation),
        },
        "behavior_sources": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in behavior_sources.items()
        },
        "train_file": {"path": str(train_path), "sha256": sha256_file(train_path)},
        "pair_manifest": {
            "path": str(pair_manifest_path),
            "sha256": sha256_file(pair_manifest_path),
        },
        "model_config": {
            "path": str(model_config_path),
            "sha256": sha256_file(model_config_path),
        },
        "model_weights": {
            "model_path": str(model_path),
            "patterns": ["*.safetensors", "pytorch_model*.bin"],
            "file_count": len(model_weight_files),
            "files": model_weight_files,
            "aggregate_sha256": canonical_sha256(model_weight_content),
        },
    },
    "command": {
        "argv": command,
        "argv_sha256": canonical_sha256(command),
        "shell": shlex.join(command),
    },
}
payload["manifest_sha256"] = canonical_sha256(payload)
temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
try:
    with temporary_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, manifest_path)
except BaseException:
    temporary_path.unlink(missing_ok=True)
    raise
PY
}

printf 'Validated scene-failure data: %s\n' "${validation_summary}"
printf 'Validated run mode: %s max_steps=%s train_rows=32 effective_passes=%s data_seed=%s\n' \
  "${RUN_MODE}" "${MAX_STEPS}" \
  "$(awk -v steps="${MAX_STEPS}" 'BEGIN { print steps / 32 }')" "${DATA_SEED}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Validated scene-failure training command (not started):\n'
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${OUTPUT_DIR}" \
  "${HF_CACHE_DIR}" \
  "${TOKENIZED_DATASET_ROOT}" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "$(dirname -- "${LOG_FILE}")"

write_launch_manifest
printf 'Wrote launch manifest: %s\n' "${LAUNCH_MANIFEST}"

printf 'Starting fresh all-42-layer scene-failure training; mode=%s output=%s\n' \
  "${RUN_MODE}" "${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
set +e
"${train_command[@]}" 2>&1 | tee -a "${LOG_FILE}"
train_status="${PIPESTATUS[0]}"
set -e
if (( train_status != 0 )); then
  printf 'ERROR: training_failed exit_code=%s\n' "${train_status}" >&2
  exit "${train_status}"
fi
for required_initial_artifact in \
  delta_mem_adapter.pt \
  delta_mem_config.json \
  training_protocol.json \
  initial_adapter_manifest.json; do
  [[ -s "${INITIAL_ADAPTER_DIR}/${required_initial_artifact}" ]] \
    || fail "initial_adapter_artifact_missing path=${INITIAL_ADAPTER_DIR}/${required_initial_artifact}"
done
printf 'Scene-failure training completed successfully.\n' | tee -a "${LOG_FILE}"
