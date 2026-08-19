#!/usr/bin/env python3
"""Run the joint query/state CrossGLU mechanics gate on open native rows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as candidate,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_query_state_bilinear_crossfit as crossfit,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as endpoint,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_screen as hardware,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_projected_value_identity as value_identity,
)
from experiments.rethinking_rwkv_ms_gemma.rwkv_joint_pair_crossglu import (  # noqa: E402
    JointPairGatedCrossGLU,
)
from experiments.rethinking_rwkv_ms_gemma.rwkv_query_state_bilinear import (  # noqa: E402
    ResidualBilinearIdentity,
)


SCHEMA = "rwkv_ms_natural_memory_native_joint_pair_crossglu_mechanics.v1"
WORLD_SIZE = 4
SEED = 116
STATE_DIM = 32
LAYERS = 42
MAX_GAIN = 0.25
HYBRID_GAIN = 0.125
CROSSFIT_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_query_state_bilinear_crossfit_v1"
CROSSFIT_RESULT_SHA256 = "5e41c4569273fd5841381fcb6c5738b26212dd326b4f7cf56b589528df346ba3"
CROSSFIT_RECEIPT = "89392eaeffa50c0bed9109fd8db3d33a5625eb4ff7117f81d710cf9b5be93945"
CROSSFIT_PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_query_state_bilinear_crossfit_protocol_v1.json"
CROSSFIT_PROTOCOL_PAYLOAD_SHA256 = "3e0c450275ed2d979d599bf29169ab89528ca673c2df5b44ffa8c01443edaddf"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_joint_pair_crossglu_mechanics_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "ca4d747496a2a6084b6182cd736e2a2de8aaa2440475d2309994d4da9f639e6f"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_crossfit_head(
    context: Any | None = None,
) -> crossfit.LayerwiseBilinear:
    protocol = json.loads(CROSSFIT_PROTOCOL.read_text(encoding="utf-8"))
    unsigned_protocol = dict(protocol)
    unsigned_protocol.pop("receipt", None)
    if (
        candidate.shared.distributed.canonical_sha256(unsigned_protocol)
        != CROSSFIT_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("receipt", {}).get("payload_sha256") != CROSSFIT_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("generation_authorized") is not False
        or protocol.get("adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise RuntimeError("Cross-fit protocol hash or stopping rule differs")
    if sha256_file(CROSSFIT_ROOT / "result.json") != CROSSFIT_RESULT_SHA256:
        raise RuntimeError("Cross-fit result hash differs from signed input")
    result = json.loads((CROSSFIT_ROOT / "result.json").read_text(encoding="utf-8"))
    unsigned_result = dict(result)
    unsigned_result.pop("receipt", None)
    if (
        result.get("passed") is not True
        or result.get("receipt", {}).get("payload_sha256") != CROSSFIT_RECEIPT
        or candidate.shared.distributed.canonical_sha256(unsigned_result) != CROSSFIT_RECEIPT
    ):
        raise RuntimeError("Cross-fit identity result does not authorize mechanics")
    payload: list[Mapping[str, torch.Tensor] | None] = [None]
    if context is None or context.is_primary:
        records, _ = crossfit.load_feature_records(CROSSFIT_ROOT, result["crossfit_split"])
        ordered = sorted(records, key=lambda row: int(row["source_index"]))
        feature = {
            name: torch.tensor([row[name] for row in ordered], dtype=torch.float32)
            for name in ("query", "correct", "matched_donor", "layer_permuted")
        }
        train_index = torch.tensor(
            [index for index, row in enumerate(ordered) if row["split"] == "train"],
            dtype=torch.long,
        )
        torch.manual_seed(crossfit.SEED)
        head = crossfit.LayerwiseBilinear()
        train = {name: value.index_select(0, train_index) for name, value in feature.items()}
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=crossfit.LEARNING_RATE,
            weight_decay=crossfit.WEIGHT_DECAY,
        )
        for _ in range(crossfit.TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)
            correct = head.score(train["query"], train["correct"])
            donor = head.score(train["query"], train["matched_donor"])
            permuted = head.score(train["query"], train["layer_permuted"])
            loss = torch.relu(crossfit.IDENTITY_MARGIN - correct + donor).mean()
            loss = loss + torch.relu(crossfit.IDENTITY_MARGIN - correct + permuted).mean()
            loss.backward()
            optimizer.step()
        if crossfit.train_and_evaluate(records) != result["analysis"]:
            raise RuntimeError("Cross-fit reconstruction differs from signed analysis")
        payload[0] = {
            name: parameter.detach().cpu()
            for name, parameter in head.state_dict().items()
        }
    if context is not None:
        dist.broadcast_object_list(payload, src=0, group=context.control_group)
    if payload[0] is None:
        raise RuntimeError("Cross-fit head broadcast returned no state")
    if context is None or context.is_primary:
        return head
    head = crossfit.LayerwiseBilinear()
    head.load_state_dict(payload[0])
    return head


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    if (
        candidate.shared.distributed.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("receipt", {}).get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("generation_authorized") is not False
        or protocol.get("causal_endpoint_authorized") is not False
        or protocol.get("adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise RuntimeError("Joint pair mechanics protocol hash or stopping rule differs")
    return protocol


def ordered_modules(model: torch.nn.Module) -> tuple[tuple[str, Any], ...]:
    modules = tuple(causal_train.ordered_modules(model))
    if len(modules) != LAYERS:
        raise RuntimeError(f"Expected {LAYERS} wrappers, got {len(modules)}")
    return modules


def install_bridges(
    model: torch.nn.Module,
    head: crossfit.LayerwiseBilinear,
) -> Mapping[str, Any]:
    modules = ordered_modules(model)
    for index, (_, module) in enumerate(modules):
        identity = ResidualBilinearIdentity(STATE_DIM, bottleneck=crossfit.BOTTLENECK)
        identity.load_state_dict(head.heads[index].state_dict())
        bridge = JointPairGatedCrossGLU(
            STATE_DIM,
            identity=identity,
            max_gain=MAX_GAIN,
        ).to(device=module.memory_v_proj.device, dtype=module.memory_v_proj.dtype)
        for parameter in bridge.identity.parameters():
            parameter.requires_grad_(False)
        module.rwkv_joint_pair_crossglu = bridge
        module.rwkv_joint_pair_crossglu_original_hybrid_mode = module.rwkv_ms_hybrid_mode
        module.rwkv_joint_pair_crossglu_original_hybrid_gain = module.rwkv_ms_hybrid_gain
        module.rwkv_ms_outer_ffn_enabled = False
    return {
        "layers": len(modules),
        "mode": "joint_pair_crossglu",
        "state_dim": STATE_DIM,
        "identity_bottleneck": crossfit.BOTTLENECK,
        "identity_maps_frozen": True,
        "max_gain": MAX_GAIN,
        "hybrid_gain": HYBRID_GAIN,
        "projected_carrier_unchanged": True,
    }


def freeze_and_select_bridges(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if "rwkv_joint_pair_crossglu" in name and ".identity." not in name:
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    if len(selected) != LAYERS * 3:
        raise RuntimeError(f"Expected {LAYERS * 3} bridge tensors, got {len(selected)}")
    return tuple(sorted(selected)), {
        "trainable_tensors": len(selected),
        "trainable_elements": sum(parameter.numel() for _, parameter in selected),
        "identity_parameters_frozen": True,
    }


def _set_joint_read_mode(modules: Sequence[tuple[str, Any]]) -> None:
    for _, module in modules:
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.rwkv_ms_hybrid_mode = "joint_pair_crossglu"
        module.rwkv_ms_hybrid_gain = HYBRID_GAIN


def _set_original_write_mode(modules: Sequence[tuple[str, Any]]) -> None:
    for _, module in modules:
        module.rwkv_ms_hybrid_mode = module.rwkv_joint_pair_crossglu_original_hybrid_mode
        module.rwkv_ms_hybrid_gain = module.rwkv_joint_pair_crossglu_original_hybrid_gain


def zero_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: torch.zeros_like(values[attribute])
            for attribute in causal_train.RECURRENT_ATTRIBUTES
        }
        for name, values in state.items()
    }


def random_norm_matched(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in state.items():
        recurrent = values["delta_state"]
        random = torch.randn_like(recurrent.float())
        random = random * (
            recurrent.float().norm(dim=-1, keepdim=True)
            / random.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        result[name] = {
            "delta_state": random.to(dtype=recurrent.dtype),
            "rwkv_ms_positions": values["rwkv_ms_positions"],
            "rwkv_ms_previous_source": values["rwkv_ms_previous_source"],
        }
    return result


def _set_queries(
    modules: Sequence[tuple[str, Any]],
    query_values: Mapping[str, torch.Tensor],
    seq_len: int,
    *,
    gate_override: Mapping[str, torch.Tensor] | None = None,
    gate_shuffle: bool = False,
) -> None:
    for name, module in modules:
        query = query_values[name]
        module.rwkv_joint_pair_crossglu_query = query.expand(-1, seq_len, -1)
        module.rwkv_joint_pair_crossglu_gate_override = (
            None if gate_override is None else gate_override[name]
        )
        module.rwkv_joint_pair_crossglu_gate_shuffle = gate_shuffle


def _capture_last_gates(modules: Sequence[tuple[str, Any]]) -> dict[str, torch.Tensor]:
    gates: dict[str, torch.Tensor] = {}
    for name, module in modules:
        gate = module.rwkv_joint_pair_crossglu_last_gate
        if gate is None:
            raise RuntimeError(f"Bridge gate was not captured for {name}")
        gates[name] = gate.detach()
    return gates


def _capture_last_corrections(modules: Sequence[tuple[str, Any]]) -> dict[str, torch.Tensor]:
    corrections: dict[str, torch.Tensor] = {}
    for name, module in modules:
        correction = module.rwkv_joint_pair_crossglu_last_correction
        if correction is None:
            raise RuntimeError(f"Bridge correction was not captured for {name}")
        corrections[name] = correction.detach()
    return corrections


def _capture_last_values(modules: Sequence[tuple[str, Any]]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, module in modules:
        value = module.rwkv_joint_pair_crossglu_last_value
        if value is None:
            raise RuntimeError(f"Bridge value was not captured for {name}")
        values[name] = value.detach()
    return values


@torch.no_grad()
def _read_condition(
    model: torch.nn.Module,
    target: Any,
    modules: Sequence[tuple[str, Any]],
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    query_values: Mapping[str, torch.Tensor],
    *,
    seq_len: int,
    rotate_recurrent_layers: bool = False,
    gate_override: Mapping[str, torch.Tensor] | None = None,
    gate_shuffle: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], bool]:
    reset_delta_mem_states(model)
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=rotate_recurrent_layers,
    )
    _set_queries(
        modules,
        query_values,
        seq_len,
        gate_override=gate_override,
        gate_shuffle=gate_shuffle,
    )
    with torch.inference_mode():
        logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    return (
        logits,
        _capture_last_gates(modules),
        _capture_last_values(modules),
        _capture_last_corrections(modules),
        bool(fixed),
    )


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> float | None:
    try:
        if logits.ndim == 2 and labels.ndim == 2 and logits.shape[0] == labels.shape[1]:
            logits = logits.unsqueeze(0)
        value, _ = contrast.detached_answer_ce(logits, labels)
        return float(value)
    except ValueError as error:
        if "do not cover target predictors" not in str(error):
            raise
        return None


def controls(
    model: torch.nn.Module,
    target: Any,
    donor: Any,
) -> Mapping[str, Any]:
    modules = ordered_modules(model)
    _set_original_write_mode(modules)
    value_identity.clear(model)
    reset_delta_mem_states(model)
    with torch.inference_mode():
        evolution._native_write(model, target, dtype=torch.bfloat16)
    target_state = causal_train.capture_online_state_references(modules)
    target_values = value_identity.capture_write_values(model)
    value_identity.clear(model)
    reset_delta_mem_states(model)
    with torch.inference_mode():
        evolution._native_write(model, donor, dtype=torch.bfloat16)
    donor_state = causal_train.capture_online_state_references(modules)
    donor_values = value_identity.capture_write_values(model)
    _set_joint_read_mode(modules)
    seq_len = int(target.read_input_ids.shape[1])
    value_identity.set_fixed_target_values(model, target_values)
    branches: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], bool]] = {}
    branches["correct"], correct_gates, correct_values, correct_corrections, correct_fixed = _read_condition(
        model, target, modules, target_state, target_state, target_values, seq_len=seq_len
    )
    branches["matched_donor"], donor_gates, donor_values_read, donor_corrections, donor_fixed = _read_condition(
        model, target, modules, target_state, donor_state, target_values, seq_len=seq_len
    )
    branches["donor_query_target_state"], donor_query_gates, _, _, donor_query_fixed = _read_condition(
        model, target, modules, target_state, target_state, donor_values, seq_len=seq_len
    )
    branches["donor_query_donor_state"], _, _, _, donor_both_fixed = _read_condition(
        model, target, modules, target_state, donor_state, donor_values, seq_len=seq_len
    )
    branches["layer_permuted"], _, _, _, permuted_fixed = _read_condition(
        model, target, modules, target_state, target_state, target_values,
        seq_len=seq_len, rotate_recurrent_layers=True,
    )
    zero = zero_recurrent(target_state)
    branches["zero_state"], _, _, _, zero_fixed = _read_condition(
        model, target, modules, target_state, zero, target_values, seq_len=seq_len
    )
    random_state = random_norm_matched(target_state)
    branches["norm_matched_random"], _, _, _, random_fixed = _read_condition(
        model, target, modules, target_state, random_state, target_values, seq_len=seq_len
    )
    branches["target_gate_donor_value"], fixed_gates, fixed_gate_values, fixed_gate_corrections, fixed_gate_fixed = _read_condition(
        model, target, modules, target_state, donor_state, target_values,
        seq_len=seq_len, gate_override=correct_gates,
    )
    branches["shuffled_gate"], shuffled_gates, shuffled_gate_values, shuffled_gate_corrections, shuffled_gate_fixed = _read_condition(
        model, target, modules, target_state, target_state, target_values,
        seq_len=seq_len, gate_shuffle=True,
    )
    reset_delta_mem_states(model)
    causal_train.install_intervened_state(
        modules,
        projected=target_state,
        recurrent=zero_recurrent(target_state),
        rotate_recurrent_layers=False,
    )
    for _, module in modules:
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.rwkv_ms_hybrid_mode = "residual"
    with torch.inference_mode():
        projected_only = evolution._native_read(model, target, dtype=torch.bfloat16)
    projected_only_fixed = True
    for _, module in modules:
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.rwkv_ms_hybrid_mode = "joint_pair_crossglu"
    ce = {name: _ce(branch[0], target.labels) for name, branch in branches.items()}
    ce["projected_only"] = _ce(projected_only, target.labels)
    zero_logits = branches["zero_state"][0]
    projected_only_logits = projected_only
    if zero_logits.ndim + 1 == projected_only_logits.ndim and projected_only_logits.size(0) == 1:
        projected_only_logits = projected_only_logits.squeeze(0)
    elif projected_only_logits.ndim + 1 == zero_logits.ndim and zero_logits.size(0) == 1:
        zero_logits = zero_logits.squeeze(0)
    if tuple(zero_logits.shape) != tuple(projected_only_logits.shape):
        raise RuntimeError(
            "Zero-state and projected-only logits differ beyond a singleton batch axis"
        )
    zero_projected_difference = (zero_logits.float() - projected_only_logits.float()).abs()
    logits = [branch[0] for branch in branches.values()] + [projected_only]
    finite = all(bool(torch.isfinite(value).all()) for value in logits)
    gate_values = list(correct_gates.values()) + list(donor_gates.values())
    gate_tensor = torch.cat([value.reshape(-1) for value in gate_values])
    value_tensor = torch.cat(
        [value.reshape(-1) for value in (*correct_values.values(), *donor_values_read.values())]
    )
    gate_activity = {
        "minimum": float(gate_tensor.min().item()),
        "maximum": float(gate_tensor.max().item()),
        "mean": float(gate_tensor.mean().item()),
        "fraction_below_001": float(gate_tensor.lt(0.01).float().mean().item()),
        "fraction_above_099": float(gate_tensor.gt(0.99).float().mean().item()),
        "non_saturated_fraction": float(
            gate_tensor.ge(0.01)
            .logical_and(gate_tensor.le(MAX_GAIN - 0.001))
            .float()
            .mean()
            .item()
        ),
    }
    correct_matrix = torch.stack(list(correct_corrections.values()), dim=1).float()
    donor_matrix = torch.stack(list(donor_corrections.values()), dim=1).float()
    relative_l2 = (
        (donor_matrix - correct_matrix).norm(dim=-1).mean(dim=(1, 2))
        / correct_matrix.norm(dim=-1).mean(dim=(1, 2)).clamp_min(1e-6)
    )
    donor_row_changed = relative_l2.ge(0.05)
    fixed_gate_value_changed = any(
        bool((fixed_gate_corrections[name] - donor_corrections[name]).abs().gt(0.0).any())
        for name in fixed_gate_corrections
    )
    shuffled_gate_value_changed = any(
        bool((shuffled_gate_corrections[name] - correct_corrections[name]).abs().gt(0.0).any())
        for name in shuffled_gate_corrections
    )
    return {
        "ce": ce,
        "all_logits_finite": finite,
        "all_bridge_values_finite": bool(torch.isfinite(value_tensor).all()),
        "carrier_fixed_all_conditions": bool(
            correct_fixed
            and donor_fixed
            and donor_query_fixed
            and donor_both_fixed
            and permuted_fixed
            and zero_fixed
            and random_fixed
            and fixed_gate_fixed
            and shuffled_gate_fixed
            and projected_only_fixed
        ),
        "zero_equals_projected_only": bool(torch.equal(zero_logits, projected_only_logits)),
        "zero_projected_max_abs_difference": float(zero_projected_difference.max().item()),
        "zero_projected_changed_fraction": float(
            zero_projected_difference.gt(0.0).float().mean().item()
        ),
        "zero_projected_max_abs_delta": float(
            (branches["zero_state"][0] - projected_only).abs().max().item()
        ),
        "zero_projected_metadata": {
            "zero_dtype": str(branches["zero_state"][0].dtype),
            "projected_only_dtype": str(projected_only.dtype),
            "zero_shape": list(branches["zero_state"][0].shape),
            "projected_only_shape": list(projected_only.shape),
            "zero_device": str(branches["zero_state"][0].device),
            "projected_only_device": str(projected_only.device),
        },
        "gate_activity": gate_activity,
        "donor_bridge_row_fraction_changed": float(donor_row_changed.float().mean().item()),
        "donor_bridge_row_relative_l2_mean": float(relative_l2.mean().item()),
        "fixed_gate_changes_donor_value": fixed_gate_value_changed,
        "fixed_gate_correction_max_abs_delta": float(
            max(
                (fixed_gate_corrections[name] - donor_corrections[name]).abs().max().item()
                for name in fixed_gate_corrections
            )
        ),
        "fixed_gate_value_max_abs_delta": float(
            max(
                (fixed_gate_values[name] - donor_values_read[name]).abs().max().item()
                for name in fixed_gate_values
            )
        ),
        "shuffled_gate_changes_value": shuffled_gate_value_changed,
        "shuffled_gate_correction_max_abs_delta": float(
            max(
                (shuffled_gate_corrections[name] - correct_corrections[name]).abs().max().item()
                for name in shuffled_gate_corrections
            )
        ),
        "shuffled_gate_value_max_abs_delta": float(
            max(
                (shuffled_gate_values[name] - correct_values[name]).abs().max().item()
                for name in shuffled_gate_values
            )
        ),
        "gate_changes_with_state": bool(
            any(
                not torch.equal(correct_gates[name], donor_gates[name])
                for name in correct_gates
            )
        ),
        "query_changes_with_state_fixed": bool(
            any(
                not torch.equal(correct_gates[name], donor_query_gates[name])
                for name in correct_gates
            )
        ),
        "all_values_finite": bool(
            torch.isfinite(correct_matrix).all()
            and torch.isfinite(donor_matrix).all()
            and torch.isfinite(gate_tensor).all()
        ),
    }


def gradient_audit(
    model: torch.nn.Module,
    selected: Sequence[tuple[str, torch.nn.Parameter]],
    target: Any,
) -> Mapping[str, Any]:
    value_identity.clear(model)
    reset_delta_mem_states(model)
    with torch.inference_mode():
        evolution._native_write(model, target, dtype=torch.bfloat16)
    target_values = value_identity.capture_write_values(model)
    modules = ordered_modules(model)
    _set_joint_read_mode(modules)
    value_identity.set_fixed_target_values(model, target_values)
    _set_queries(modules, target_values, int(target.read_input_ids.shape[1]))
    logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    loss_sum, tokens, _ = evolution.checkpointed_native_answer_loss_sum_and_count(
        logits,
        target.labels,
        chunk_tokens=contrast.CE_CHUNK_TOKENS,
    )
    loss = loss_sum / tokens
    loss.backward()
    nonzero = [
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.abs().max().gt(0))
        for _, parameter in selected
    ]
    return {
        "loss": float(loss.detach().item()),
        "trainable_tensors": len(selected),
        "finite_nonzero_tensors": int(sum(nonzero)),
        "passed": bool(all(nonzero)),
    }


def run(*, base_model: Path, dataset_root: Path, output_dir: Path) -> Mapping[str, Any]:
    context = candidate.shared.distributed.initialize_distributed_training("cuda")
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol = validate_protocol()
        if context.world_size != WORLD_SIZE:
            raise RuntimeError("Joint pair mechanics requires four ranks")
        if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
            raise RuntimeError("HF_ENDPOINT must be exactly https://hf-mirror.com")
        if not hardware.four_distinct_a100s(context.rank_devices):
            raise RuntimeError("Joint pair mechanics requires four distinct A100s")
        if context.is_primary and output_dir.exists():
            raise ValueError(f"Output must be fresh: {output_dir}")
        candidate.shared.distributed.phase_consensus(
            context,
            phase="joint-pair-output-freshness",
            error=None,
        )
        if context.is_primary:
            output_dir.mkdir(parents=True, exist_ok=False)
        candidate.shared.distributed.phase_consensus(
            context,
            phase="joint-pair-output-creation",
            error=None,
        )
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        head = reconstruct_crossfit_head(context)
        model, tokenizer, model_audit = candidate.load_model(base_model, device=context.device)
        installation = install_bridges(model, head)
        selected, trainable_audit = freeze_and_select_bridges(model)
        rows = contrast.load_scene_rows(tokenizer, dataset_root)
        mapping, _, _ = contrast.build_donor_mapping(rows)
        schedule, _ = contrast.build_schedule(rows, mapping, contrast.build_donor_mapping(rows)[1])
        source = int(schedule[0].source_ordinals[context.process_rank])
        donor_source = int(schedule[0].donor_ordinals[context.process_rank])
        target = evolution.collate_native_examples(
            [rows[source].example],
            pad_token_id=int(tokenizer.pad_token_id),
            device=context.device,
        )
        donor = contrast.build_donor_batch(target, rows[donor_source].example, device=context.device)
        model.eval()
        diagnostics = controls(model, target, donor)
        model.train()
        gradients = gradient_audit(model, selected, target)
        local = {
            "rank": context.process_rank,
            "source_ordinal": source,
            "donor_ordinal": donor_source,
            "diagnostics": diagnostics,
            "gradients": gradients,
        }
        gathered = candidate.shared.distributed.gather_objects(context, local)
        checks = {
            "four_a100_ranks": True,
            "all_logits_finite": all(row["diagnostics"]["all_logits_finite"] for row in gathered),
            "carrier_fixed_all_conditions": all(row["diagnostics"]["carrier_fixed_all_conditions"] for row in gathered),
            "zero_equals_projected_only": all(row["diagnostics"]["zero_equals_projected_only"] for row in gathered),
            "pair_gate_changes_with_state": all(row["diagnostics"]["gate_changes_with_state"] for row in gathered),
            "matched_donor_row_fraction": all(row["diagnostics"]["donor_bridge_row_fraction_changed"] >= 0.95 for row in gathered),
            "matched_donor_mean_relative_l2": all(row["diagnostics"]["donor_bridge_row_relative_l2_mean"] >= 0.05 for row in gathered),
            "fixed_gate_donor_value_path": all(row["diagnostics"]["fixed_gate_changes_donor_value"] for row in gathered),
            "gate_shuffle_path": all(row["diagnostics"]["shuffled_gate_changes_value"] for row in gathered),
            "pair_gate_non_saturated": all(row["diagnostics"]["gate_activity"]["non_saturated_fraction"] >= 0.5 for row in gathered),
            "all_values_finite": all(row["diagnostics"]["all_values_finite"] for row in gathered),
            "bridge_gradients_nonzero": all(row["gradients"]["passed"] for row in gathered),
        }
        result = {
            "schema": SCHEMA,
            "status": "joint_pair_crossglu_mechanics_passed_generation_blocked" if all(checks.values()) else "joint_pair_crossglu_mechanics_failed_causal_endpoint_blocked",
            "passed": all(checks.values()),
            "generation_authorized": False,
            "causal_endpoint_authorized": bool(all(checks.values())),
            "crossfit_result_sha256": sha256_file(CROSSFIT_ROOT / "result.json"),
            "crossfit_result": str(CROSSFIT_ROOT / "result.json"),
            "installation": installation,
            "trainable_audit": trainable_audit,
            "model_audit": model_audit,
            "checks": checks,
            "rank_rows": list(gathered),
            "protected_splits_opened": [],
            "no_adapter_weights_saved": True,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "protocol_objective": protocol["objective"],
        }
        if context.is_primary:
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": candidate.shared.distributed.canonical_sha256(result),
            }
            output_dir.joinpath("result.json").write_text(
                json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dist.barrier()
        return result
    finally:
        candidate.shared.distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
