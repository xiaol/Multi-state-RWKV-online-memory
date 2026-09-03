"""Shared contracts for recurrent-routed projected-value post-training."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    HFDeltaMemConfig,
    attach_delta_mem,
    freeze_non_delta_mem_params,
    iter_delta_mem_modules,
    reset_delta_mem_states,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "576537822ca7079d15fc6d0ce618a94b8631286c47008f136ef1b6ed725d191d"
)
BASE_MODEL = Path("/root/x/models/gemma-4-E4B-it")
BASE_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
BASE_MODEL_WEIGHTS_SHA256 = (
    "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
)
BASE_CONFIG_SHA256 = (
    "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
)
TOKENIZER_SHA256 = (
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
)
WARMSTART_ADAPTER = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"
)
WARMSTART_WEIGHTS_SHA256 = (
    "b063940a9be0712f830a992e9114055e4488297b0842245e6e26563b303545a9"
)
WARMSTART_CONFIG_SHA256 = (
    "94b4649a2b14f178dfd2b2de18bcc77894a5606f8b67426bf753f033690273f4"
)
SPLIT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_recurrent_routed_posttrain_split_v1"
)
SPLIT_MANIFEST_RECEIPT = (
    "05314bfcaa3f4c6febe860f33bf7867af8d57a80e9e1b9020b1cc318bceebc96"
)
FINAL_COMMITMENT_RECEIPT = (
    "c8c106a00e1379e26bbae5b774f0fe831de2c2527c3195a1e42881a33b2b2fae"
)
OPEN_SPLIT_RECEIPT = (
    "159cf93c913715f0c90e03ca659bf3bd4f1deb9d3e12c64f923d7b5b71340ad8"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
TASKS = ("attribution", "narrative", "scene")
EXPECTED_LAYERS = 42
HYBRID_MODE = "recurrent_routed_projected_value"
HYBRID_GAIN = 0.125
READ_TEMPERATURE = 16.0
READ_TOP_K = 1
RECURRENT_ATTRIBUTES = (
    "delta_state",
    "rwkv_ms_positions",
    "rwkv_ms_previous_source",
)
PROJECTED_ATTRIBUTES = (
    "projected_kv_keys",
    "projected_kv_values",
    "projected_kv_occupied",
    "projected_kv_surprise",
)
CONDITIONS = (
    "correct_recurrent_state",
    "zero_recurrent_state",
    "matched_donor_recurrent_state",
    "slot_shuffled_recurrent_state",
    "layer_permuted_recurrent_state",
)
PROMPT_VARIANTS = {
    "attribution": (
        None,
        "阅读给出的中文小说上下文和候选角色，判断目标对话最可能由谁说；若证据不足则标记不确定。角色必须来自候选列表。仅返回符合要求的 JSON。",
        "请完成说话者归因：依据上下文，在候选角色中选择目标台词的最佳归属，并给出不确定性判断。禁止输出候选外角色或解释，只输出 JSON。",
        "把下面内容当作中文小说对话归因题。选出候选集中最可能的说话者，同时判断是否无法确定；答案必须是 JSON，不能附加说明。",
    ),
    "narrative": (
        None,
        "为下列已切分的中文小说叙事单元逐一分类。每个 unit_id 只能标为 dialogue、narration、thought、action 或 scene_description；不要改写原文，只输出标准 JSON。输入对话使用 「」 引号。",
        "请标注每个叙事 unit_id 的类型，合法标签仅有 dialogue, narration, thought, action, scene_description。保留单元内容不变，不要解释，并用标准双引号输出 JSON。",
        "这是中文小说叙事单元分类任务。对每个 unit_id 从 dialogue、narration、thought、action、scene_description 中选一个标签；禁止改写和额外文字，答案只含 JSON。",
    ),
    "scene": (
        None,
        "阅读这些中文小说段落，找出确有场景切换的段落边界。情绪变化本身不算切换。只用 JSON 返回 boundaries，不要说明理由。",
        "请判断输入段落之间哪些位置发生了明显 scene 转换；不要把单纯的情绪或语气变化当作边界。输出仅限包含 boundaries 的 JSON。",
        "完成小说场景边界检测：只标真正发生时间、地点或场景转换的边界，不能因情绪变化而切分。不要解释，直接返回 boundaries JSON。",
    ),
}


@dataclass(frozen=True)
class SourceRow:
    task: str
    source_ordinal: int
    row_sha256: str
    raw_line: str
    assistant_identity: str
    user_characters: int


@dataclass(frozen=True)
class ScheduledRow:
    step: int
    position: int
    target: SourceRow
    donor: SourceRow
    prompt_variant: int


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_signed_json(path: Path, expected_receipt: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"Signed JSON receipt is missing: {path}")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != expected_receipt or receipt.get("payload_sha256") != digest:
        raise ValueError(f"Signed JSON payload differs: {path}")
    return value


def validate_protocol() -> Mapping[str, Any]:
    protocol = validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    frozen = protocol.get("frozen_inputs", {})
    expected = {
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_weights_sha256": BASE_MODEL_WEIGHTS_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "warm_start_weights_sha256": WARMSTART_WEIGHTS_SHA256,
        "warm_start_config_sha256": WARMSTART_CONFIG_SHA256,
        "split_manifest_receipt": SPLIT_MANIFEST_RECEIPT,
        "final_commitment_receipt": FINAL_COMMITMENT_RECEIPT,
        "open_split_receipt": OPEN_SPLIT_RECEIPT,
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("Recurrent-routed protocol frozen inputs differ")
    architecture = protocol.get("architecture", {})
    if (
        architecture.get("rwkv_ms_hybrid_mode") != HYBRID_MODE
        or architecture.get("rwkv_ms_hybrid_gain") != HYBRID_GAIN
        or architecture.get("task_router") is not False
        or architecture.get("template_matcher") is not False
        or architecture.get("dual_pass_selector") is not False
        or architecture.get("benchmark_specific_decoder") is not False
    ):
        raise ValueError("Recurrent-routed architecture protocol differs")
    return protocol


def validate_split_artifacts() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = validate_signed_json(
        SPLIT_ROOT / "manifest.json",
        SPLIT_MANIFEST_RECEIPT,
    )
    final_commitment = validate_signed_json(
        SPLIT_ROOT / "final_commitment.json",
        FINAL_COMMITMENT_RECEIPT,
    )
    open_receipt = validate_signed_json(
        SPLIT_ROOT / "open_split_receipt.json",
        OPEN_SPLIT_RECEIPT,
    )
    if (
        manifest.get("final_commitment_payload_sha256")
        != FINAL_COMMITMENT_RECEIPT
        or open_receipt.get("manifest_receipt") != SPLIT_MANIFEST_RECEIPT
        or open_receipt.get("final_files_written") != []
        or open_receipt.get("materialized_splits") != ["train", "development"]
        or final_commitment.get("semantic_content_opened_during_commitment")
        is not False
    ):
        raise ValueError("Recurrent-routed split bindings differ")
    for relative_path, metadata in open_receipt["files"].items():
        path = SPLIT_ROOT / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != metadata.get("bytes")
            or sha256_file(path) != metadata.get("sha256")
        ):
            raise ValueError(f"Open split file differs: {path}")
    return manifest, open_receipt


def load_open_rows(
    split: str,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, tuple[SourceRow, ...]]:
    if split not in {"train", "development"}:
        raise ValueError("Only committed open splits may be loaded before final authorization")
    rows_by_task: dict[str, tuple[SourceRow, ...]] = {}
    for task in TASKS:
        path = SPLIT_ROOT / "open" / task / f"{split}.jsonl"
        raw_lines = tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)
        committed = manifest["tasks"][task]["splits"][split]["rows"]
        if len(raw_lines) != len(committed):
            raise ValueError(f"Open {task}/{split} row count differs")
        loaded: list[SourceRow] = []
        for raw_line, metadata in zip(raw_lines, committed):
            digest = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            if digest != metadata["row_sha256"]:
                raise ValueError(f"Open {task}/{split} row hash differs")
            value = json.loads(raw_line)
            messages = value.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 3
                or [message.get("role") for message in messages]
                != ["system", "user", "assistant"]
            ):
                raise ValueError(f"Open {task}/{split} messages differ")
            loaded.append(
                SourceRow(
                    task=task,
                    source_ordinal=int(metadata["source_ordinal"]),
                    row_sha256=digest,
                    raw_line=raw_line,
                    assistant_identity=str(messages[-1]["content"]),
                    user_characters=len(str(messages[1]["content"])),
                )
            )
        rows_by_task[task] = tuple(loaded)
    return rows_by_task


def paraphrased_raw_line(row: SourceRow, prompt_variant: int) -> str:
    variants = PROMPT_VARIANTS[row.task]
    if not 0 <= prompt_variant < len(variants):
        raise ValueError("Prompt paraphrase variant is outside the locked set")
    if prompt_variant == 0:
        return row.raw_line
    value = json.loads(row.raw_line)
    messages = [dict(message) for message in value["messages"]]
    messages[0]["content"] = variants[prompt_variant]
    value["messages"] = messages
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def encode_row(
    tokenizer: Any,
    row: SourceRow,
    *,
    prompt_variant: int,
) -> evolution.NativeFullRowExample:
    return evolution.encode_native_full_row(
        tokenizer,
        task=row.task,
        source_ordinal=row.source_ordinal,
        raw_line=paraphrased_raw_line(row, prompt_variant),
    )


def build_training_schedule(
    rows_by_task: Mapping[str, Sequence[SourceRow]],
    *,
    updates: int,
) -> tuple[tuple[ScheduledRow, ...], list[dict[str, Any]]]:
    patterns = (
        {"attribution": 3, "narrative": 3, "scene": 2},
        {"attribution": 2, "narrative": 3, "scene": 3},
        {"attribution": 3, "narrative": 2, "scene": 3},
    )
    ordered = {
        task: sorted(
            rows_by_task[task],
            key=lambda row: (
                hashlib.sha256(
                    (
                        "rwkv-ms-recurrent-routed-train-v1:"
                        + task
                        + ":"
                        + row.row_sha256
                    ).encode("utf-8")
                ).hexdigest(),
                row.source_ordinal,
            ),
        )
        for task in TASKS
    }
    cursors = {task: 0 for task in TASKS}
    schedule: list[ScheduledRow] = []
    payload: list[dict[str, Any]] = []
    for step in range(1, updates + 1):
        step_rows: list[SourceRow] = []
        for task, count in patterns[(step - 1) % len(patterns)].items():
            start = cursors[task]
            selected = ordered[task][start : start + count]
            if len(selected) != count:
                raise RuntimeError(f"Insufficient locked training rows for {task}")
            step_rows.extend(selected)
            cursors[task] += count
        step_rows.sort(
            key=lambda row: hashlib.sha256(
                f"rwkv-ms-recurrent-routed-step-v1:{step}:{row.row_sha256}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        step_payload: list[dict[str, Any]] = []
        for position, target in enumerate(step_rows):
            candidates = [
                candidate
                for candidate in rows_by_task[target.task]
                if candidate.source_ordinal != target.source_ordinal
                and candidate.assistant_identity != target.assistant_identity
            ]
            if not candidates:
                raise ValueError(f"Training row has no different-answer donor: {target}")
            donor = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate.user_characters - target.user_characters),
                    candidate.row_sha256,
                    candidate.source_ordinal,
                ),
            )
            variant = (step + position - 1) % 4
            scheduled = ScheduledRow(
                step=step,
                position=position,
                target=target,
                donor=donor,
                prompt_variant=variant,
            )
            schedule.append(scheduled)
            step_payload.append(
                {
                    "position": position,
                    "task": target.task,
                    "source_ordinal": target.source_ordinal,
                    "source_row_sha256": target.row_sha256,
                    "donor_source_ordinal": donor.source_ordinal,
                    "donor_row_sha256": donor.row_sha256,
                    "prompt_variant": variant,
                }
            )
        payload.append(
            {
                "step": step,
                "rows": step_payload,
                "payload_sha256": canonical_sha256(step_payload),
            }
        )
    return tuple(schedule), payload


def build_config() -> HFDeltaMemConfig:
    data = json.loads(
        (WARMSTART_ADAPTER / "delta_mem_config.json").read_text(encoding="utf-8")
    )
    config = HFDeltaMemConfig.from_dict(data)
    return replace(
        config,
        memory_readout_mode="projected_kv_rwkv_hybrid",
        rwkv_ms_hybrid_mode=HYBRID_MODE,
        rwkv_ms_hybrid_gain=HYBRID_GAIN,
        rwkv_ms_read_temperature=READ_TEMPERATURE,
        rwkv_ms_read_top_k=READ_TOP_K,
        rwkv_ms_detach_read_scores=False,
    )


def _warmstart_common_parameters(model: torch.nn.Module) -> Mapping[str, Any]:
    if sha256_file(WARMSTART_ADAPTER / "delta_mem_adapter.pt") != WARMSTART_WEIGHTS_SHA256:
        raise ValueError("V9 warm-start weights differ")
    if sha256_file(WARMSTART_ADAPTER / "delta_mem_config.json") != WARMSTART_CONFIG_SHA256:
        raise ValueError("V9 warm-start config differs")
    source = torch.load(
        WARMSTART_ADAPTER / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected = {
        f"{module_name}.{parameter_name}"
        for module_name, module in iter_delta_mem_modules(model)
        for parameter_name, _ in module.named_parameters()
        if not parameter_name.startswith("base.")
    }
    route_keys = {
        key
        for key in expected
        if key.endswith((".rwkv_route_query_proj", ".rwkv_route_state_proj"))
    }
    pair_value_keys = {
        key for key in expected if key.endswith(".rwkv_pair_value_proj")
    }
    recurrent_value_keys = {
        key for key in expected if key.endswith(".rwkv_recurrent_value_proj")
    }
    pair_gate_keys = {
        key
        for key in expected
        if key.endswith((".rwkv_pair_gate_weight", ".rwkv_pair_gate_bias"))
    }
    fresh_keys = pair_value_keys | recurrent_value_keys | pair_gate_keys
    if set(source) != expected - route_keys - fresh_keys or len(route_keys) != EXPECTED_LAYERS * 2:
        missing = sorted(expected - route_keys - set(source))
        extra = sorted(set(source) - expected)
        raise ValueError(
            f"V9 warm-start topology differs: missing={missing[:4]} extra={extra[:4]}"
        )
    named_parameters = dict(model.named_parameters())
    for key, value in source.items():
        parameter = named_parameters[key]
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(f"V9 warm-start shape differs: {key}")
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    for key in pair_value_keys | recurrent_value_keys:
        parameter = named_parameters[key]
        if not torch.equal(parameter.detach(), torch.zeros_like(parameter.detach())):
            raise ValueError(f"Fresh recurrent residual parameter is not zero: {key}")
    for key in pair_gate_keys:
        parameter = named_parameters[key]
        if key.endswith(".rwkv_pair_gate_weight"):
            passed = torch.equal(
                parameter.detach(),
                torch.zeros_like(parameter.detach()),
            )
        else:
            passed = torch.allclose(
                torch.sigmoid(parameter.detach().float()),
                torch.full_like(parameter.detach().float(), 0.01),
                atol=5e-4,
                rtol=0.0,
            )
        if not passed:
            raise ValueError(f"Fresh recurrent pair gate initialization differs: {key}")
    for name, module in iter_delta_mem_modules(model):
        if not torch.equal(
            module.rwkv_route_query_proj.detach().cpu(),
            torch.eye(module.rank),
        ) or not torch.equal(
            module.rwkv_route_state_proj.detach().cpu(),
            torch.eye(module.rank),
        ):
            raise ValueError(f"Route warm start is not identity: {name}")
    return {
        "source_parameter_tensors": len(source),
        "identity_route_parameter_tensors": len(route_keys),
        "source_weights_sha256": WARMSTART_WEIGHTS_SHA256,
        "source_config_sha256": WARMSTART_CONFIG_SHA256,
    }


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    freeze_non_delta_mem_params(model)
    trainable_suffixes = (
        ".rwkv_route_query_proj",
        ".rwkv_route_state_proj",
        ".hrm_rwkv7_core.output.weight",
        ".rwkv_pair_value_proj",
        ".rwkv_recurrent_value_proj",
        ".rwkv_pair_gate_weight",
        ".rwkv_pair_gate_bias",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith(trainable_suffixes))
    runtime._promote_trainable_parameters_to_fp32(model)
    selected = distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    names = [name for name, _ in selected]
    query_routes = [name for name in names if name.endswith(".rwkv_route_query_proj")]
    state_routes = [name for name in names if name.endswith(".rwkv_route_state_proj")]
    readouts = [name for name in names if name.endswith(".hrm_rwkv7_core.output.weight")]
    pair_values = [name for name in names if name.endswith(".rwkv_pair_value_proj")]
    recurrent_values = [
        name for name in names if name.endswith(".rwkv_recurrent_value_proj")
    ]
    pair_gate_weights = [
        name for name in names if name.endswith(".rwkv_pair_gate_weight")
    ]
    pair_gate_biases = [
        name for name in names if name.endswith(".rwkv_pair_gate_bias")
    ]
    dead_projected_keys = [
        name for name in names if name.endswith(".projected_kv_key_proj")
    ]
    passed = (
        bool(selected)
        and len(query_routes) == EXPECTED_LAYERS
        and len(state_routes) == EXPECTED_LAYERS
        and len(readouts) == EXPECTED_LAYERS
        and len(pair_values)
        == (
            EXPECTED_LAYERS
            if HYBRID_MODE
            in {
                "recurrent_routed_query_value",
                "recurrent_routed_residual_query_value",
                "recurrent_routed_gated_query_value",
            }
            else 0
        )
        and len(recurrent_values)
        == (
            EXPECTED_LAYERS
            if HYBRID_MODE == "recurrent_routed_residual_query_value"
            else 0
        )
        and len(pair_gate_weights)
        == (
            EXPECTED_LAYERS
            if HYBRID_MODE == "recurrent_routed_gated_query_value"
            else 0
        )
        and len(pair_gate_biases)
        == (
            EXPECTED_LAYERS
            if HYBRID_MODE == "recurrent_routed_gated_query_value"
            else 0
        )
        and not dead_projected_keys
        and all(parameter.dtype == torch.float32 for _, parameter in selected)
    )
    audit = {
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": canonical_sha256(names),
        "route_query_tensors": len(query_routes),
        "route_state_tensors": len(state_routes),
        "rwkv_readout_tensors": len(readouts),
        "rwkv_pair_value_tensors": len(pair_values),
        "rwkv_recurrent_value_tensors": len(recurrent_values),
        "rwkv_pair_gate_weight_tensors": len(pair_gate_weights),
        "rwkv_pair_gate_bias_tensors": len(pair_gate_biases),
        "trainable_parameter_suffixes": list(trainable_suffixes),
        "projected_key_router_trainable_tensors": len(dead_projected_keys),
        "all_trainable_fp32": all(
            parameter.dtype == torch.float32 for _, parameter in selected
        ),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Recurrent-routed trainable isolation failed: {audit!r}")
    return selected, audit


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    trainable: bool,
    configure_trainables: Callable[
        [torch.nn.Module],
        tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]],
    ]
    | None = None,
) -> tuple[torch.nn.Module, Any, HFDeltaMemConfig, Mapping[str, Any]]:
    base_model = base_model.expanduser().resolve(strict=True)
    if sha256_file(base_model / "model.safetensors") != BASE_MODEL_WEIGHTS_SHA256:
        raise ValueError("Pinned Gemma weights differ")
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Pinned Gemma config differs")
    if sha256_file(base_model / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Pinned Gemma tokenizer differs")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    runtime._disable_training_cache(model)
    delta_config = build_config()
    replaced = attach_delta_mem(model, delta_config)
    warmstart = _warmstart_common_parameters(model)
    named_trainable: tuple[tuple[str, torch.nn.Parameter], ...] = ()
    trainable_audit: Mapping[str, Any] = {"passed": True, "parameter_tensors": 0}
    if trainable:
        configure_trainables = (
            configure_trainable_parameters
            if configure_trainables is None
            else configure_trainables
        )
        named_trainable, trainable_audit = configure_trainables(model)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    configured = (
        len(replaced) == EXPECTED_LAYERS
        and len(modules) == EXPECTED_LAYERS
        and all(
            module.memory_backend == "rwkv_ms"
            and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == HYBRID_MODE
            and module.rwkv_ms_hybrid_gain == HYBRID_GAIN
            and module.rwkv_ms_read_temperature == READ_TEMPERATURE
            and module.rwkv_ms_read_top_k == READ_TOP_K
            and module.rwkv_ms_detach_read_scores is False
            for _, module in modules
        )
    )
    audit = {
        "replaced_layers": len(replaced),
        "wrapped_layers": len(modules),
        "configured": configured,
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "gradient_checkpointing": bool(model.is_gradient_checkpointing),
        "warmstart": warmstart,
        "trainables": trainable_audit,
        "named_trainable": named_trainable,
    }
    if not configured:
        raise RuntimeError(f"Recurrent-routed model attachment failed: {audit!r}")
    return model, tokenizer, delta_config, audit


def ordered_modules(model: torch.nn.Module) -> tuple[tuple[str, Any], ...]:
    modules = tuple(
        sorted(
            iter_delta_mem_modules(model),
            key=lambda item: int(item[0].split(".layers.", 1)[1].split(".", 1)[0]),
        )
    )
    if len(modules) != EXPECTED_LAYERS:
        raise ValueError("Recurrent-routed interventions require 42 wrapped layers")
    return modules


def capture_online_state_references(
    modules: Sequence[tuple[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    captured: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in modules:
        attributes: dict[str, torch.Tensor] = {}
        for attribute in (*RECURRENT_ATTRIBUTES, *PROJECTED_ATTRIBUTES):
            value = getattr(module, attribute)
            if value is None:
                raise RuntimeError(f"Online write omitted {name}.{attribute}")
            attributes[attribute] = value
        captured[name] = attributes
    return captured


def install_condition_state(
    modules: Sequence[tuple[str, Any]],
    *,
    correct: Mapping[str, Mapping[str, torch.Tensor]],
    donor: Mapping[str, Mapping[str, torch.Tensor]] | None,
    condition: str,
) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown recurrent-routed condition: {condition}")
    names = [name for name, _ in modules]
    for index, (name, module) in enumerate(modules):
        for attribute in PROJECTED_ATTRIBUTES:
            setattr(module, attribute, correct[name][attribute])
        source_name = name
        recurrent_source = correct
        if condition == "matched_donor_recurrent_state":
            if donor is None:
                raise ValueError("Matched donor condition requires donor state")
            recurrent_source = donor
        elif condition == "layer_permuted_recurrent_state":
            source_name = names[(index + 1) % len(names)]
        for attribute in RECURRENT_ATTRIBUTES:
            source = recurrent_source[source_name][attribute]
            if condition == "zero_recurrent_state":
                source = torch.zeros_like(source)
            elif (
                condition == "slot_shuffled_recurrent_state"
                and attribute == "delta_state"
            ):
                source = source.roll(shifts=1, dims=2)
            setattr(module, attribute, source)
    return all(
        getattr(module, attribute) is correct[name][attribute]
        for name, module in modules
        for attribute in PROJECTED_ATTRIBUTES
    )


def direct_condition_logits(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    *,
    condition: str,
    donor: evolution.NativeFullRowBatch | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Mapping[str, bool]]:
    modules = ordered_modules(model)
    evolution._native_write(model, target, dtype=dtype)
    correct = capture_online_state_references(modules)
    donor_state = None
    if donor is not None:
        evolution._native_write(model, donor, dtype=dtype)
        donor_state = capture_online_state_references(modules)
    references_fixed = install_condition_state(
        modules,
        correct=correct,
        donor=donor_state,
        condition=condition,
    )
    logits = evolution._native_read(model, target, dtype=dtype)
    bytes_fixed = all(
        torch.equal(getattr(module, attribute), correct[name][attribute])
        for name, module in modules
        for attribute in PROJECTED_ATTRIBUTES
    )
    return logits, {
        "projected_carrier_references_fixed": references_fixed,
        "projected_carrier_bytes_fixed": bytes_fixed,
    }


def checkpointed_condition_logits(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    *,
    condition: str,
    donor: evolution.NativeFullRowBatch | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Mapping[str, bool]]:
    audit = {
        "projected_carrier_references_fixed": True,
        "projected_carrier_bytes_fixed": True,
    }

    def write_read(*tensors: torch.Tensor) -> torch.Tensor:
        recompute_target = evolution.NativeFullRowBatch(
            examples=target.examples,
            write_input_ids=tensors[0],
            write_attention_mask=tensors[1],
            read_input_ids=tensors[2],
            read_attention_mask=tensors[3],
            labels=target.labels,
        )
        recompute_donor = None
        if donor is not None:
            recompute_donor = evolution.NativeFullRowBatch(
                examples=target.examples,
                write_input_ids=tensors[4],
                write_attention_mask=tensors[5],
                read_input_ids=tensors[2],
                read_attention_mask=tensors[3],
                labels=target.labels,
            )
        logits, branch_audit = direct_condition_logits(
            model,
            recompute_target,
            condition=condition,
            donor=recompute_donor,
            dtype=dtype,
        )
        for key in audit:
            audit[key] = bool(audit[key] and branch_audit[key])
        return logits

    inputs = [
        target.write_input_ids,
        target.write_attention_mask,
        target.read_input_ids,
        target.read_attention_mask,
    ]
    if donor is not None:
        inputs.extend((donor.write_input_ids, donor.write_attention_mask))
    logits = checkpoint(write_read, *inputs, use_reentrant=False)
    return logits, audit


def audit_gradient_family(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    suffix: str,
) -> Mapping[str, Any]:
    rows = []
    for name, parameter in named_trainable:
        if not name.endswith(suffix):
            continue
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        norm = (
            0.0
            if gradient is None or not finite
            else float(gradient.detach().float().norm().item())
        )
        rows.append(
            {
                "name": name,
                "gradient_present": gradient is not None,
                "gradient_finite": finite,
                "gradient_l2_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        )
    passed = (
        len(rows) == EXPECTED_LAYERS
        and all(row["gradient_finite"] for row in rows)
        and all(row["gradient_nonzero"] for row in rows)
    )
    return {
        "suffix": suffix,
        "parameter_tensors": len(rows),
        "parameter_names_sha256": canonical_sha256([row["name"] for row in rows]),
        "minimum_l2_norm": min(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "maximum_l2_norm": max(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "layers": rows,
        "passed": passed,
    }


def audit_joint_routing_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    families = {
        "route_query": audit_gradient_family(
            named_trainable,
            ".rwkv_route_query_proj",
        ),
        "route_state": audit_gradient_family(
            named_trainable,
            ".rwkv_route_state_proj",
        ),
        "rwkv_readout": audit_gradient_family(
            named_trainable,
            ".hrm_rwkv7_core.output.weight",
        ),
    }
    if HYBRID_MODE in {
        "recurrent_routed_query_value",
        "recurrent_routed_gated_query_value",
    }:
        families["pair_value"] = audit_gradient_family(
            named_trainable,
            ".rwkv_pair_value_proj",
        )
    if HYBRID_MODE == "recurrent_routed_gated_query_value":
        families["pair_gate_weight"] = audit_gradient_family(
            named_trainable,
            ".rwkv_pair_gate_weight",
        )
        families["pair_gate_bias"] = audit_gradient_family(
            named_trainable,
            ".rwkv_pair_gate_bias",
        )
    return {
        "families": families,
        "passed": all(family["passed"] for family in families.values()),
    }
