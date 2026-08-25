#!/usr/bin/env python3
"""Train a source-bound outer FFN on the explicitly open residual split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.cumulative_rwkv_residual import (  # noqa: E402
    SourceBoundOuterFFN,
    SourceCumulativeResidualRouter,
)
from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_source_cumulative_residual_development as development_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_source_cumulative_residual_development_screen as screen,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_source_bound_outer_ffn_development.v3"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_bound_outer_ffn_development.v1.split"
)
PROTOCOL_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_bound_outer_ffn_development.v3.protocol"
)
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_source_bound_outer_ffn_development_protocol_v3.json"
)
PROTOCOL_FILE_SHA256 = "7680e8fca730889343936378468db36fb0df023b3d2852828f4d838a48a8d4ae"
PROTOCOL_PAYLOAD_SHA256 = "33b37d02558e4a4384cf1bf227e93d24d359523191d89c97d0f400060ce4ade8"
DEFAULT_BASE_MODEL = screen.DEFAULT_BASE_MODEL
DEFAULT_MATERIALIZATION = screen.DEFAULT_DEVELOPMENT_MATERIALIZATION
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_bound_outer_ffn_development_train_v3"
)

WORLD_SIZE = 4
HF_ENDPOINT = "https://hf-mirror.com"
TIMEOUT_SECONDS = 1800
ANCHORS = (5, 11, 17)
TERMINAL_ANCHOR = ANCHORS[-1]
COMPATIBILITY_SCALE = 1.0
RESIDUAL_GAIN = 1.0 / 32.0
NATIVE_READ_DIM = 32
HIDDEN_DIM = 2560
BOTTLENECK_DIM = 32
SEED = 20260825
SPLIT_SALT = "rwkv-source-bound-outer-ffn-open-pair-split-v1:"
TRAIN_PAIRS = 16
HELDOUT_PAIRS = 16
TRAIN_ROWS = 32
HELDOUT_ROWS = 32
GLOBAL_BATCH_ROWS = 4
LOCAL_BATCH_ROWS = 1
UPDATES = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
DONOR_MARGIN = 0.02
LAYER_MARGIN = 0.01
CONTRAST_TEMPERATURE = 0.02
CORRECT_CE_WEIGHT = 0.5
SINGLE_CE_WEIGHT = 0.5
DONOR_CONTRAST_WEIGHT = 1.0
LAYER_CONTRAST_WEIGHT = 0.5
TRAIN_CONTROLS = (
    "correct_four_way",
    "single_target",
    "matched_donor_address_and_state",
    "layer_rolled_address_and_state",
)
TRAIN_CONTROL_INDEX = {
    name: index for index, name in enumerate(TRAIN_CONTROLS)
}
HELDOUT_DONOR_POSITIVE_MINIMUM = 0.75
HELDOUT_DONOR_MEAN_MINIMUM = 0.02
HELDOUT_LAYER_POSITIVE_MINIMUM = 0.75
HELDOUT_CORRECT_GAIN_MINIMUM = 0.0
TRAINABLE_TENSORS = 3
TRAINABLE_ELEMENTS = (
    NATIVE_READ_DIM * BOTTLENECK_DIM
    + HIDDEN_DIM * BOTTLENECK_DIM
    + BOTTLENECK_DIM * HIDDEN_DIM
)
TRAINING_TARGET_MODE = "first_target_donor_divergent_supervised_token"
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
    "f5589ee6a37d9fa044d0e4629dd2c0499dda73554c5e291aad845a72ad645076"
)
DISCRIMINATIVE_CONTROLS = (
    "correct_four_way",
    "single_target",
    "matched_donor_address_and_state",
    "layer_rolled_address_and_state",
    "zero_state",
)
DISCRIMINATIVE_CONTROL_INDEX = {
    name: index for index, name in enumerate(DISCRIMINATIVE_CONTROLS)
}


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Source-bound artifact must be fresh: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_receipt(
    value: Mapping[str, Any], *, scope: str, description: str
) -> None:
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    expected = {
        "algorithm": "sha256",
        "payload_scope": scope,
        "payload_sha256": canonical_sha256(unsigned),
    }
    if dict(receipt) != expected:
        raise ValueError(f"{description} receipt differs")


def validate_protocol() -> Mapping[str, Any]:
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Source-bound development protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Source-bound development protocol schema differs")
    validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Source-bound development protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Source-bound development protocol receipt differs")
    if (
        protocol.get("open_development_only") is not True
        or protocol.get("protected_mechanics_authorized") is not False
        or protocol.get("protected_causal_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Source-bound development access policy differs")
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    gate = protocol.get("heldout_gate", {})
    if architecture != {
        "anchor_layers": list(ANCHORS),
        "bottleneck_dim": BOTTLENECK_DIM,
        "compatibility_scale": COMPATIBILITY_SCALE,
        "exact_zero_state_path": True,
        "hidden_dim": HIDDEN_DIM,
        "initialization": "zero correction over the frozen native residual",
        "native_read_dim": NATIVE_READ_DIM,
        "query_only_or_hidden_only_path": False,
        "residual_gain": RESIDUAL_GAIN,
        "selected_native_rwkv_read_is_value": True,
        "trainable_parameter_elements": TRAINABLE_ELEMENTS,
        "trainable_parameter_tensors": TRAINABLE_TENSORS,
    }:
        raise ValueError("Source-bound development architecture contract differs")
    if training != {
        "contrast_temperature": CONTRAST_TEMPERATURE,
        "correct_ce_weight": CORRECT_CE_WEIGHT,
        "donor_contrast_weight": DONOR_CONTRAST_WEIGHT,
        "donor_margin": DONOR_MARGIN,
        "first_update_gradient_contract": {
            "outer_ffn.output_up.weight": True,
            "outer_ffn.query_gate.weight": False,
            "outer_ffn.state_down.weight": False,
        },
        "global_batch_rows": GLOBAL_BATCH_ROWS,
        "gradient_clip": GRADIENT_CLIP,
        "heldout_pairs": HELDOUT_PAIRS,
        "layer_contrast_weight": LAYER_CONTRAST_WEIGHT,
        "layer_margin": LAYER_MARGIN,
        "learning_rate": LEARNING_RATE,
        "local_batch_rows": LOCAL_BATCH_ROWS,
        "optimizer": "fused AdamW with rank-averaged gradients",
        "single_ce_weight": SINGLE_CE_WEIGHT,
        "subsequent_update_gradient_contract": (
            "all three trainable tensors active"
        ),
        "target_mode": TRAINING_TARGET_MODE,
        "target_payload_sha256": DISCRIMINATIVE_TARGET_PAYLOAD_SHA256,
        "train_controls": list(TRAIN_CONTROLS),
        "train_pairs": TRAIN_PAIRS,
        "updates": UPDATES,
        "weight_decay": WEIGHT_DECAY,
    }:
        raise ValueError("Source-bound development training contract differs")
    if gate != {
        "correct_gain_vs_provider_off_mean_minimum": HELDOUT_CORRECT_GAIN_MINIMUM,
        "donor_both_minus_target_mean_minimum": HELDOUT_DONOR_MEAN_MINIMUM,
        "donor_both_positive_row_fraction_minimum": HELDOUT_DONOR_POSITIVE_MINIMUM,
        "layer_both_positive_row_fraction_minimum": HELDOUT_LAYER_POSITIVE_MINIMUM,
        "mechanics_must_pass": True,
    }:
        raise ValueError("Source-bound development heldout gate differs")
    discriminative_gate = protocol.get("discriminative_heldout_gate", {})
    if discriminative_gate != gate:
        raise ValueError("Source-bound discriminative heldout gate differs")
    return protocol


def _pair_key(row: Mapping[str, Any]) -> tuple[int, int]:
    source = int(row["source_index"])
    donor = int(row["donor_source_index"])
    if source == donor:
        raise ValueError("Development row cannot donate to itself")
    return tuple(sorted((source, donor)))


def split_open_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any]]:
    if len(rows) != TRAIN_ROWS + HELDOUT_ROWS:
        raise ValueError("Source-bound development row count differs")
    rows_by_source = {int(row["source_index"]): row for row in rows}
    if len(rows_by_source) != len(rows):
        raise ValueError("Source-bound development source IDs are not unique")
    pair_keys = sorted({_pair_key(row) for row in rows})
    if len(pair_keys) != TRAIN_PAIRS + HELDOUT_PAIRS:
        raise ValueError("Source-bound development pair count differs")
    for source, donor in pair_keys:
        if (
            int(rows_by_source[source]["donor_source_index"]) != donor
            or int(rows_by_source[donor]["donor_source_index"]) != source
        ):
            raise ValueError("Source-bound development donor pairing is not reciprocal")
    ranked_pairs = sorted(
        pair_keys,
        key=lambda pair: (
            hashlib.sha256(
                (SPLIT_SALT + canonical_sha256(list(pair))).encode("ascii")
            ).hexdigest(),
            pair,
        ),
    )
    train_pair_keys = set(ranked_pairs[:TRAIN_PAIRS])
    heldout_pair_keys = set(ranked_pairs[TRAIN_PAIRS:])
    if train_pair_keys & heldout_pair_keys:
        raise RuntimeError("Source-bound train/heldout pairs overlap")
    train_rows = sorted(
        [row for row in rows if _pair_key(row) in train_pair_keys],
        key=lambda row: int(row["source_index"]),
    )
    heldout_rows = sorted(
        [row for row in rows if _pair_key(row) in heldout_pair_keys],
        key=lambda row: int(row["source_index"]),
    )
    train_sources = [int(row["source_index"]) for row in train_rows]
    heldout_sources = [int(row["source_index"]) for row in heldout_rows]
    if (
        len(train_rows) != TRAIN_ROWS
        or len(heldout_rows) != HELDOUT_ROWS
        or set(train_sources) & set(heldout_sources)
        or set(train_sources) | set(heldout_sources) != set(rows_by_source)
    ):
        raise RuntimeError("Source-bound train/heldout row partition differs")
    payload = {
        "schema": SPLIT_SCHEMA,
        "salt": SPLIT_SALT,
        "split_unit": "reciprocal donor pair with globally component-disjoint rows",
        "train_pairs": [list(pair) for pair in ranked_pairs[:TRAIN_PAIRS]],
        "heldout_pairs": [list(pair) for pair in ranked_pairs[TRAIN_PAIRS:]],
        "train_sources": train_sources,
        "heldout_sources": heldout_sources,
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "source_disjoint": True,
        "pair_disjoint": True,
        "component_disjoint_from_parent_reservation": True,
    }
    return train_rows, heldout_rows, payload


def build_discriminative_targets(
    rows: Sequence[Mapping[str, Any]], examples: Mapping[int, Any]
) -> tuple[dict[int, Mapping[str, int]], Mapping[str, Any]]:
    rows_by_source = {int(row["source_index"]): row for row in rows}
    if len(rows_by_source) != TRAIN_ROWS + HELDOUT_ROWS:
        raise ValueError("Discriminative target row coverage differs")
    targets = {}
    payload_rows = []
    for source in sorted(rows_by_source):
        donor = int(rows_by_source[source]["donor_source_index"])
        target_labels = tuple(int(value) for value in examples[source].labels)
        donor_labels = tuple(int(value) for value in examples[donor].labels)
        target_positions = tuple(
            index for index, value in enumerate(target_labels) if value != -100
        )
        donor_positions = tuple(
            index for index, value in enumerate(donor_labels) if value != -100
        )
        target_tokens = tuple(target_labels[index] for index in target_positions)
        donor_tokens = tuple(donor_labels[index] for index in donor_positions)
        divergence = next(
            (
                index
                for index, (target_token, donor_token) in enumerate(
                    zip(target_tokens, donor_tokens)
                )
                if target_token != donor_token
            ),
            None,
        )
        if divergence is None:
            raise ValueError("Reciprocal donor answers have no aligned divergent token")
        label_index = target_positions[divergence]
        predictor = label_index - 1
        if predictor < 1 or target_labels[label_index] == donor_tokens[divergence]:
            raise RuntimeError("Discriminative target geometry differs")
        target = {
            "source_index": source,
            "donor_source_index": donor,
            "answer_offset": divergence,
            "predictor_index": predictor,
            "label_index": label_index,
            "target_token_id": target_labels[label_index],
            "donor_token_id": donor_tokens[divergence],
        }
        targets[source] = target
        payload_rows.append(target)
    payload = {
        "schema": f"{SCHEMA}.discriminative_targets",
        "mode": TRAINING_TARGET_MODE,
        "rows": payload_rows,
        "row_count": len(payload_rows),
        "all_target_tokens_differ_from_donor": all(
            row["target_token_id"] != row["donor_token_id"]
            for row in payload_rows
        ),
    }
    return targets, payload


def make_router(maps: Mapping[int, Any], device: torch.device) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundOuterFFN(
        state_dim=NATIVE_READ_DIM,
        query_dim=HIDDEN_DIM,
        bottleneck_dim=BOTTLENECK_DIM,
    )
    router = SourceCumulativeResidualRouter(
        maps=maps,
        anchor_layers=ANCHORS,
        compatibility_scale=COMPATIBILITY_SCALE,
        residual_gain=RESIDUAL_GAIN,
        required_receptance_calls=2,
        outer_ffn=outer_ffn,
    ).to(device)
    return router


def named_trainable(
    router: SourceCumulativeResidualRouter,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    if router.outer_ffn is None:
        raise RuntimeError("Source-bound router has no outer FFN")
    values = []
    for name, parameter in router.outer_ffn.named_parameters():
        parameter.requires_grad_(True)
        parameter.data = parameter.data.float()
        values.append((f"outer_ffn.{name}", parameter))
    result = tuple(values)
    if (
        len(result) != TRAINABLE_TENSORS
        or sum(parameter.numel() for _, parameter in result) != TRAINABLE_ELEMENTS
        or any(parameter.dtype != torch.float32 for _, parameter in result)
    ):
        raise RuntimeError("Source-bound trainable parameter inventory differs")
    return result


def parameter_digest(named: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named:
        digest.update(name.encode("utf-8"))
        value = parameter.detach().float().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_controls(
    banks: tuple[Mapping[int, torch.Tensor], ...],
    controls: Sequence[str],
) -> tuple[dict[int, torch.Tensor], ...]:
    indices = torch.tensor(
        [screen.CONTROL_INDEX[name] for name in controls],
        dtype=torch.long,
        device=next(iter(banks[0].values())).device,
    )
    return tuple(
        {
            int(layer): value.index_select(0, indices)
            for layer, value in bank.items()
        }
        for bank in banks
    )


def routed_predictor_logits(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    target_state: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    router: SourceCumulativeResidualRouter,
    banks: tuple[Mapping[int, torch.Tensor], ...],
    predictor: int,
    memory_mass_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...]]:
    if predictor < 1:
        raise ValueError("Source-bound predictor requires a nonempty prefill")
    batch_size = int(next(iter(banks[0].values())).size(0))
    screen.parent_runner.install_target_state(model, modules, target_state, batch_size)
    input_ids = batch.read_input_ids.repeat(batch_size, 1)
    attention_mask = batch.read_attention_mask.repeat(batch_size, 1)
    prefix_positions = torch.arange(
        predictor, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    query_positions = torch.full(
        (batch_size, 1), predictor, device=input_ids.device, dtype=torch.long
    )
    screen.clear_providers(modules_by_layer)
    try:
        with torch.no_grad(), screen.mechanics.evolution.runtime._autocast_context(
            input_ids.device, torch.bfloat16
        ):
            prefill = model(
                input_ids=input_ids[:, :predictor],
                attention_mask=attention_mask[:, :predictor],
                position_ids=prefix_positions,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        cache = prefill.past_key_values
        router.begin_forward(
            states={layer: banks[0][layer] for layer in ANCHORS},
            address_keys={layer: banks[1][layer] for layer in ANCHORS},
            occupied={layer: banks[2][layer] for layer in ANCHORS},
            source_ids={layer: banks[3][layer] for layer in ANCHORS},
            memory_mass_override=memory_mass_override,
        )
        for layer in ANCHORS:
            modules_by_layer[layer].set_source_cumulative_residual_provider(
                router.provider_for(layer)
            )
        with screen.mechanics.evolution.runtime._autocast_context(
            input_ids.device, torch.bfloat16
        ):
            output = model(
                input_ids=input_ids[:, predictor : predictor + 1],
                attention_mask=attention_mask[:, : predictor + 1],
                position_ids=query_positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        diagnostics = router.end_forward()
        if tuple(item["layer"] for item in diagnostics) != ANCHORS:
            raise RuntimeError("Source-bound router lifecycle differs")
        return output.logits[:, -1].float(), diagnostics
    finally:
        screen.clear_providers(modules_by_layer)
        if router.active or router.completed:
            router.abort_forward()


def provider_off_predictor_logits(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    target_state: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
    predictor: int,
) -> torch.Tensor:
    if predictor < 1:
        raise ValueError("Source-bound provider-off predictor requires a nonempty prefill")
    screen.parent_runner.install_target_state(model, modules, target_state, batch_size)
    input_ids = batch.read_input_ids.repeat(batch_size, 1)
    attention_mask = batch.read_attention_mask.repeat(batch_size, 1)
    prefix_positions = torch.arange(
        predictor, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    query_positions = torch.full(
        (batch_size, 1), predictor, device=input_ids.device, dtype=torch.long
    )
    screen.clear_providers(modules_by_layer)
    with screen.mechanics.evolution.runtime._autocast_context(
        input_ids.device, torch.bfloat16
    ):
        prefill = model(
            input_ids=input_ids[:, :predictor],
            attention_mask=attention_mask[:, :predictor],
            position_ids=prefix_positions,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        output = model(
            input_ids=input_ids[:, predictor : predictor + 1],
            attention_mask=attention_mask[:, : predictor + 1],
            position_ids=query_positions,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    return output.logits[:, -1].float()


def training_loss(
    logits: torch.Tensor, target_token: int
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    if logits.ndim != 2 or logits.size(0) != len(TRAIN_CONTROLS):
        raise ValueError("Source-bound training logit geometry differs")
    labels = torch.full(
        (len(TRAIN_CONTROLS),),
        int(target_token),
        dtype=torch.long,
        device=logits.device,
    )
    ce = F.cross_entropy(logits, labels, reduction="none")
    correct = ce[TRAIN_CONTROL_INDEX["correct_four_way"]]
    single = ce[TRAIN_CONTROL_INDEX["single_target"]]
    donor = ce[TRAIN_CONTROL_INDEX["matched_donor_address_and_state"]]
    layer = ce[TRAIN_CONTROL_INDEX["layer_rolled_address_and_state"]]
    donor_contrast = CONTRAST_TEMPERATURE * F.softplus(
        (DONOR_MARGIN - (donor - single)) / CONTRAST_TEMPERATURE
    )
    layer_contrast = CONTRAST_TEMPERATURE * F.softplus(
        (LAYER_MARGIN - (layer - single)) / CONTRAST_TEMPERATURE
    )
    loss = (
        CORRECT_CE_WEIGHT * correct
        + SINGLE_CE_WEIGHT * single
        + DONOR_CONTRAST_WEIGHT * donor_contrast
        + LAYER_CONTRAST_WEIGHT * layer_contrast
    )
    return loss, {
        "correct_ce": correct,
        "single_ce": single,
        "donor_ce": donor,
        "layer_ce": layer,
        "donor_minus_single": donor - single,
        "layer_minus_single": layer - single,
        "donor_contrast": donor_contrast,
        "layer_contrast": layer_contrast,
    }


def synchronize_gradients(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    world_size: int,
    update: int,
) -> Mapping[str, Any]:
    finite = True
    maximum = 0.0
    activity = {}
    for name, parameter in named:
        if parameter.grad is None:
            raise RuntimeError("Source-bound trainable parameter has no gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        local_maximum = float(parameter.grad.abs().max().item())
        activity[name] = local_maximum > 0.0
        maximum = max(maximum, local_maximum)
    if update == 0:
        expected_activity = {
            "outer_ffn.state_down.weight": False,
            "outer_ffn.query_gate.weight": False,
            "outer_ffn.output_up.weight": True,
        }
    else:
        expected_activity = {name: True for name, _ in named}
    gradient_contract_passed = activity == expected_activity
    if not finite or not gradient_contract_passed:
        raise RuntimeError("Source-bound synchronized gradients are invalid")
    return {
        "all_trainable_gradients_finite": finite,
        "tensor_activity": activity,
        "expected_tensor_activity": expected_activity,
        "gradient_contract_passed": gradient_contract_passed,
        "maximum_absolute_gradient": maximum,
    }


def _float_metrics(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().float().item()) for name, value in values.items()}


def train_outer_ffn(
    model: torch.nn.Module,
    router: SourceCumulativeResidualRouter,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    train_rows: Sequence[Mapping[str, Any]],
    examples: Mapping[int, Any],
    natural_cache: Mapping[int, Any],
    candidates: Mapping[int, Sequence[int]],
    discriminative_targets: Mapping[int, Mapping[str, int]],
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    names_by_layer: Mapping[int, str],
    ordered_names: Sequence[str],
    *,
    context: Any,
    pad_token_id: int,
    output_dir: Path,
) -> Mapping[str, Any]:
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    initial_digest = parameter_digest(named)
    distributed.require_consensus(
        context, initial_digest, description="source-bound initial outer FFN"
    )
    model.eval()
    router.train()
    records = []
    for update in range(UPDATES):
        global_offset = (update * GLOBAL_BATCH_ROWS) % len(train_rows)
        global_indices = tuple(
            (global_offset + rank) % len(train_rows) for rank in range(WORLD_SIZE)
        )
        local_row = train_rows[global_indices[context.process_rank]]
        source = int(local_row["source_index"])
        batch = screen.mechanics.evolution.collate_native_examples(
            [examples[source]], pad_token_id=pad_token_id, device=context.device
        )
        full_banks = screen.control_banks(
            natural_cache,
            candidates[source],
            names_by_layer,
            ordered_names,
            context.device,
        )
        banks = select_controls(full_banks, TRAIN_CONTROLS)
        target_binding = discriminative_targets[source]
        predictor = int(target_binding["predictor_index"])
        target_token = int(target_binding["target_token_id"])
        optimizer.zero_grad(set_to_none=True)
        logits, diagnostics = routed_predictor_logits(
            model,
            batch,
            modules,
            modules_by_layer,
            natural_cache[source]["state"],
            router=router,
            banks=banks,
            predictor=predictor,
        )
        loss, metrics = training_loss(logits, target_token)
        loss.backward()
        gradient_audit = synchronize_gradients(
            named, world_size=WORLD_SIZE, update=update
        )
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], GRADIENT_CLIP
            ).item()
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError("Source-bound gradient norm is nonfinite")
        optimizer.step()
        terminal = diagnostics[-1]
        local_record = {
            "rank": context.process_rank,
            "source_index": source,
            "target_token_id": target_token,
            "answer_offset": int(target_binding["answer_offset"]),
            "predictor_index": predictor,
            "loss": float(loss.detach().float().item()),
            "metrics": _float_metrics(metrics),
            "selected_sources": [
                int(value)
                for value in terminal["source_ids"]
                .gather(1, terminal["selected_slot"])
                .detach()
                .cpu()
                .flatten()
            ],
            "gradient_audit": gradient_audit,
            "gradient_norm_before_clip": gradient_norm,
        }
        rank_records = distributed.gather_objects(context, local_record)
        step = {
            "schema": STEP_SCHEMA,
            "update": update + 1,
            "global_indices": list(global_indices),
            "global_sources": [
                int(train_rows[index]["source_index"]) for index in global_indices
            ],
            "mean_loss": sum(item["loss"] for item in rank_records) / WORLD_SIZE,
            "mean_metrics": {
                name: sum(item["metrics"][name] for item in rank_records) / WORLD_SIZE
                for name in rank_records[0]["metrics"]
            },
            "maximum_gradient_norm_before_clip": max(
                item["gradient_norm_before_clip"] for item in rank_records
            ),
            "all_trainable_gradients_finite": all(
                item["gradient_audit"]["all_trainable_gradients_finite"]
                for item in rank_records
            ),
            "gradient_contract_passed": all(
                item["gradient_audit"]["gradient_contract_passed"]
                for item in rank_records
            ),
            "rank_rows": list(rank_records),
        }
        step["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_step_without_receipt",
            "payload_sha256": canonical_sha256(step),
        }
        records.append(step)
        if context.is_primary:
            with (output_dir / "training_progress.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(step, ensure_ascii=True, sort_keys=True) + "\n"
                )
        print(
            f"SOURCE_BOUND_OUTER_FFN_TRAIN rank={context.process_rank} "
            f"update={update + 1}/{UPDATES} source={source} "
            f"loss={local_record['loss']:.6f}",
            flush=True,
        )
        reset_delta_mem_states(model)
        screen.mechanics.evolution.release_native_row_allocator_cache(context.device)
    final_digest = parameter_digest(named)
    distributed.require_consensus(
        context, final_digest, description="source-bound trained outer FFN"
    )
    return {
        "updates": len(records),
        "global_batch_rows": GLOBAL_BATCH_ROWS,
        "initial_parameter_sha256": initial_digest,
        "final_parameter_sha256": final_digest,
        "trainable_subset_changed": final_digest != initial_digest,
        "all_step_gradient_contracts_passed": all(
            step["all_trainable_gradients_finite"]
            and step["gradient_contract_passed"]
            for step in records
        ),
        "first_step": records[0],
        "last_step": records[-1],
    }


@torch.no_grad()
def evaluate_heldout(
    model: torch.nn.Module,
    router: SourceCumulativeResidualRouter,
    heldout_rows: Sequence[Mapping[str, Any]],
    examples: Mapping[int, Any],
    natural_cache: Mapping[int, Any],
    candidates: Mapping[int, Sequence[int]],
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    names_by_layer: Mapping[int, str],
    ordered_names: Sequence[str],
    compatibility_maps: Mapping[int, Any],
    *,
    context: Any,
    pad_token_id: int,
) -> Mapping[str, Any]:
    model.eval()
    router.eval()
    assigned = heldout_rows[context.process_rank :: WORLD_SIZE]
    if len(assigned) != HELDOUT_ROWS // WORLD_SIZE:
        raise RuntimeError("Source-bound heldout rank assignment differs")
    local_results = []
    for ordinal, row in enumerate(assigned, start=1):
        source = int(row["source_index"])
        batch = screen.mechanics.evolution.collate_native_examples(
            [examples[source]], pad_token_id=pad_token_id, device=context.device
        )
        first_label, _ = screen.retrieval.first_prompt_boundary(batch.labels)
        target_token = int(batch.labels[0, first_label].item())
        banks = screen.control_banks(
            natural_cache,
            candidates[source],
            names_by_layer,
            ordered_names,
            context.device,
        )
        screen.clear_terminal_hooks(modules)
        full_null = screen.full_null_predictor(
            model,
            batch,
            modules,
            modules_by_layer,
            natural_cache[source]["state"],
        )
        baseline = screen.predictor_pass(
            model,
            batch,
            modules,
            modules_by_layer,
            natural_cache[source]["state"],
            batch_size=len(screen.CONTROL_NAMES),
            router=None,
            banks=None,
        )
        baseline_replay = screen.predictor_pass(
            model,
            batch,
            modules,
            modules_by_layer,
            natural_cache[source]["state"],
            batch_size=len(screen.CONTROL_NAMES),
            router=None,
            banks=None,
        )
        screen.bind_terminal_hook(model, modules, terminal_layer=TERMINAL_ANCHOR)
        routed = screen.predictor_pass(
            model,
            batch,
            modules,
            modules_by_layer,
            natural_cache[source]["state"],
            batch_size=len(screen.CONTROL_NAMES),
            router=router,
            banks=banks,
            compatibility_maps=compatibility_maps,
        )
        local_results.append(
            screen.development_row_result(
                source=source,
                sources=candidates[source],
                target=target_token,
                anchor_layers=ANCHORS,
                compatibility_scale=COMPATIBILITY_SCALE,
                routed=routed,
                baseline=baseline,
                baseline_replay=baseline_replay,
                full_null=full_null,
            )
        )
        screen.clear_terminal_hooks(modules)
        reset_delta_mem_states(model)
        screen.mechanics.evolution.release_native_row_allocator_cache(context.device)
        print(
            f"SOURCE_BOUND_OUTER_FFN_HELDOUT rank={context.process_rank} "
            f"row={source} ordinal={ordinal}/{len(assigned)}",
            flush=True,
        )
    gathered = distributed.gather_objects(context, local_results)
    rows = sorted(
        [row for rank_rows in gathered for row in rank_rows],
        key=lambda row: int(row["source_index"]),
    )
    if len(rows) != HELDOUT_ROWS:
        raise RuntimeError("Source-bound heldout result coverage differs")
    aggregate = screen.aggregate_development_variant(rows, ANCHORS)
    margins = aggregate["target_ce_margins"]
    checks = {
        "heldout_rows_complete": len(rows) == HELDOUT_ROWS,
        "mechanics_pass": aggregate["mechanics_pass"],
        "correct_gain_vs_provider_off_mean": (
            margins["gain_vs_provider_off"]["mean"]
            > HELDOUT_CORRECT_GAIN_MINIMUM
        ),
        "donor_both_mean_margin": (
            margins["donor_both_minus_target"]["mean"]
            >= HELDOUT_DONOR_MEAN_MINIMUM
        ),
        "donor_both_positive_row_fraction": (
            margins["donor_both_minus_target"]["positive_fraction"]
            >= HELDOUT_DONOR_POSITIVE_MINIMUM
        ),
        "layer_both_positive_row_fraction": (
            margins["layer_both_minus_target"]["positive_fraction"]
            >= HELDOUT_LAYER_POSITIVE_MINIMUM
        ),
        "zero_controls_exact_provider_off": all(
            row["invariants"]["zero_controls_exact_provider_off"] for row in rows
        ),
    }
    return {
        "rows": rows,
        "aggregate": aggregate,
        "checks": checks,
        "passed": all(checks.values()),
    }


@torch.no_grad()
def evaluate_discriminative_heldout(
    model: torch.nn.Module,
    router: SourceCumulativeResidualRouter,
    heldout_rows: Sequence[Mapping[str, Any]],
    examples: Mapping[int, Any],
    natural_cache: Mapping[int, Any],
    candidates: Mapping[int, Sequence[int]],
    discriminative_targets: Mapping[int, Mapping[str, int]],
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    names_by_layer: Mapping[int, str],
    ordered_names: Sequence[str],
    *,
    context: Any,
    pad_token_id: int,
) -> Mapping[str, Any]:
    model.eval()
    router.eval()
    assigned = heldout_rows[context.process_rank :: WORLD_SIZE]
    if len(assigned) != HELDOUT_ROWS // WORLD_SIZE:
        raise RuntimeError("Discriminative heldout rank assignment differs")
    screen.clear_terminal_hooks(modules)
    screen.bind_terminal_hook(model, modules, terminal_layer=TERMINAL_ANCHOR)
    local_rows = []
    try:
        for ordinal, row in enumerate(assigned, start=1):
            source = int(row["source_index"])
            target_binding = discriminative_targets[source]
            predictor = int(target_binding["predictor_index"])
            target_token = int(target_binding["target_token_id"])
            batch = screen.mechanics.evolution.collate_native_examples(
                [examples[source]],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            full_banks = screen.control_banks(
                natural_cache,
                candidates[source],
                names_by_layer,
                ordered_names,
                context.device,
            )
            banks = select_controls(full_banks, DISCRIMINATIVE_CONTROLS)
            baseline = provider_off_predictor_logits(
                model,
                batch,
                modules,
                modules_by_layer,
                natural_cache[source]["state"],
                batch_size=len(DISCRIMINATIVE_CONTROLS),
                predictor=predictor,
            )
            routed, diagnostics = routed_predictor_logits(
                model,
                batch,
                modules,
                modules_by_layer,
                natural_cache[source]["state"],
                router=router,
                banks=banks,
                predictor=predictor,
            )
            labels = torch.full(
                (len(DISCRIMINATIVE_CONTROLS),),
                target_token,
                dtype=torch.long,
                device=routed.device,
            )
            routed_ce = F.cross_entropy(routed, labels, reduction="none")
            baseline_ce = F.cross_entropy(baseline, labels, reduction="none")
            correct_index = DISCRIMINATIVE_CONTROL_INDEX["correct_four_way"]
            single_index = DISCRIMINATIVE_CONTROL_INDEX["single_target"]
            donor_index = DISCRIMINATIVE_CONTROL_INDEX[
                "matched_donor_address_and_state"
            ]
            layer_index = DISCRIMINATIVE_CONTROL_INDEX[
                "layer_rolled_address_and_state"
            ]
            zero_index = DISCRIMINATIVE_CONTROL_INDEX["zero_state"]
            terminal = diagnostics[-1]
            selected_slot = int(
                terminal["selected_slot"][correct_index, 0].item()
            )
            selected_source = (
                int(
                    terminal["source_ids"][correct_index, selected_slot].item()
                )
                if selected_slot >= 0
                else -1
            )
            local_rows.append(
                {
                    **dict(target_binding),
                    "selected_source_index": selected_source,
                    "target_ce": {
                        "provider_off": float(baseline_ce[correct_index].item()),
                        "correct_four_way": float(routed_ce[correct_index].item()),
                        "single_target": float(routed_ce[single_index].item()),
                        "matched_donor_address_and_state": float(
                            routed_ce[donor_index].item()
                        ),
                        "layer_rolled_address_and_state": float(
                            routed_ce[layer_index].item()
                        ),
                    },
                    "target_ce_margins": {
                        "gain_vs_provider_off": float(
                            baseline_ce[correct_index].item()
                            - routed_ce[correct_index].item()
                        ),
                        "donor_both_minus_target": float(
                            routed_ce[donor_index].item()
                            - routed_ce[single_index].item()
                        ),
                        "layer_both_minus_target": float(
                            routed_ce[layer_index].item()
                            - routed_ce[single_index].item()
                        ),
                    },
                    "zero_logits_byte_exact_provider_off": bool(
                        torch.equal(routed[zero_index], baseline[zero_index])
                    ),
                    "all_logits_finite": bool(
                        torch.isfinite(routed).all()
                        and torch.isfinite(baseline).all()
                    ),
                    "residual_finite_and_bounded": bool(
                        torch.isfinite(terminal["residual"]).all()
                        and terminal["residual"].abs().max().item()
                        <= RESIDUAL_GAIN
                    ),
                }
            )
            reset_delta_mem_states(model)
            screen.mechanics.evolution.release_native_row_allocator_cache(
                context.device
            )
            print(
                f"SOURCE_BOUND_OUTER_FFN_DIVERGENT rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/{len(assigned)}",
                flush=True,
            )
    finally:
        screen.clear_terminal_hooks(modules)
    gathered = distributed.gather_objects(context, local_rows)
    rows = sorted(
        [row for rank_rows in gathered for row in rank_rows],
        key=lambda row: int(row["source_index"]),
    )
    if len(rows) != HELDOUT_ROWS:
        raise RuntimeError("Discriminative heldout result coverage differs")
    margin_names = tuple(rows[0]["target_ce_margins"])
    margins = {
        name: {
            "mean": sum(row["target_ce_margins"][name] for row in rows)
            / len(rows),
            "positive_fraction": sum(
                row["target_ce_margins"][name] > 0.0 for row in rows
            )
            / len(rows),
        }
        for name in margin_names
    }
    selected_fraction = sum(
        row["selected_source_index"] == row["source_index"] for row in rows
    ) / len(rows)
    checks = {
        "heldout_rows_complete": len(rows) == HELDOUT_ROWS,
        "target_selected_fraction": selected_fraction >= 0.75,
        "correct_gain_vs_provider_off_mean": (
            margins["gain_vs_provider_off"]["mean"]
            > HELDOUT_CORRECT_GAIN_MINIMUM
        ),
        "donor_both_mean_margin": (
            margins["donor_both_minus_target"]["mean"]
            >= HELDOUT_DONOR_MEAN_MINIMUM
        ),
        "donor_both_positive_row_fraction": (
            margins["donor_both_minus_target"]["positive_fraction"]
            >= HELDOUT_DONOR_POSITIVE_MINIMUM
        ),
        "layer_both_positive_row_fraction": (
            margins["layer_both_minus_target"]["positive_fraction"]
            >= HELDOUT_LAYER_POSITIVE_MINIMUM
        ),
        "zero_controls_exact_provider_off": all(
            row["zero_logits_byte_exact_provider_off"] for row in rows
        ),
        "all_logits_and_residuals_valid": all(
            row["all_logits_finite"] and row["residual_finite_and_bounded"]
            for row in rows
        ),
    }
    return {
        "rows": rows,
        "target_ce_margins": margins,
        "target_selected_fraction": selected_fraction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def save_checkpoint(
    named: Sequence[tuple[str, torch.nn.Parameter]], path: Path
) -> Mapping[str, Any]:
    from safetensors.torch import save_file

    state = {
        name: parameter.detach().float().cpu().contiguous()
        for name, parameter in named
    }
    save_file(
        state,
        str(path),
        metadata={
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        },
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tensors": len(state),
        "elements": sum(value.numel() for value in state.values()),
    }


def prepare_output(context: Any, output_dir: Path) -> None:
    error: BaseException | None = None
    if context.is_primary:
        try:
            if output_dir.exists():
                raise ValueError(f"Source-bound output must be fresh: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=False)
        except BaseException as caught:
            error = caught
    distributed.phase_consensus(context, phase="source-bound-output", error=error)


def run(
    *,
    base_model: Path,
    materialization_root: Path,
    output_dir: Path,
    preflight_only: bool = False,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda", required_world_size=WORLD_SIZE, timeout_seconds=TIMEOUT_SECONDS
    )
    if context is None:
        raise RuntimeError("Run source-bound training with torchrun --nproc_per_node=4")
    try:
        if (
            context.backend != "nccl"
            or context.control_backend != "gloo"
            or not screen.mechanics.hardware.four_distinct_a100s(context.rank_devices)
        ):
            raise RuntimeError("Source-bound training requires four distinct A100s")
        if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
            raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
        protocol = validate_protocol()
        manifest = screen.consensual_operation(
            context,
            phase="source-bound-open-manifest",
            operation=lambda: development_materializer.load_manifest(
                materialization_root / "manifest.json"
            ),
        )
        primary_rows = screen.consensual_operation(
            context,
            phase="source-bound-open-rows",
            operation=lambda: (
                development_materializer.read_open_development(
                    materialization_root, manifest
                )
                if context.is_primary
                else None
            ),
        )
        rows = screen.retrieval._broadcast_primary_object(context, primary_rows)
        train_rows, heldout_rows, split_payload = split_open_rows(rows)
        if canonical_sha256(split_payload) != protocol["split"]["payload_sha256"]:
            raise RuntimeError("Source-bound open train/heldout split differs")
        distributed.require_consensus(
            context,
            canonical_sha256(split_payload),
            description="source-bound open split",
        )
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, model_audit = screen.exact_v5.load_exact_v5_model(
            base_model, device=context.device
        )
        model.eval()
        modules = screen.mechanics.causal_train.ordered_modules(model)
        ordered_names = tuple(name for name, _ in modules)
        maps = screen.mechanics.load_frozen_maps(ordered_names)
        screen.integration.install(
            model,
            rank=screen.mechanics.MAP_RANK,
            seed=screen.mechanics.SEED,
            k_gain=screen.mechanics.K_GAIN,
            a_gain=screen.mechanics.A_GAIN,
            b_gain=screen.mechanics.B_GAIN,
            trainable_map=False,
        )
        for name, module in modules:
            module.rwkv_continuous_write_conditioner.load_frozen_map(
                maps[name].down, maps[name].up
            )
        screen.integration.set_mode(model, screen.integration.CONTINUOUS_MODE)
        screen.integration.set_capture(model, True)
        screen.mechanics.install_feature_observer(modules)
        modules_by_layer = {int(module.layer_idx): module for _, module in modules}
        names_by_layer = {int(module.layer_idx): name for name, module in modules}
        compatibility_maps = {
            layer: maps[names_by_layer[layer]] for layer in ANCHORS
        }
        terminal_module = modules_by_layer[TERMINAL_ANCHOR]
        delta_o_proj = getattr(terminal_module, "delta_o_proj", None)
        if (
            int(compatibility_maps[TERMINAL_ANCHOR].up.size(0))
            != NATIVE_READ_DIM
            or not isinstance(delta_o_proj, torch.Tensor)
            or tuple(delta_o_proj.shape) != (HIDDEN_DIM, NATIVE_READ_DIM)
        ):
            raise RuntimeError("Source-bound terminal read geometry differs")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model_versions_before = screen.parameter_versions(model)
        router = make_router(compatibility_maps, context.device)
        named = named_trainable(router)
        initial_parameter_sha256 = parameter_digest(named)
        examples = screen.retrieval._encode_rows(tokenizer, rows)
        discriminative_targets, discriminative_target_payload = (
            build_discriminative_targets(rows, examples)
        )
        discriminative_target_sha256 = canonical_sha256(
            discriminative_target_payload
        )
        if discriminative_target_sha256 != DISCRIMINATIVE_TARGET_PAYLOAD_SHA256:
            raise RuntimeError("Source-bound discriminative target payload differs")
        distributed.require_consensus(
            context,
            discriminative_target_sha256,
            description="source-bound discriminative targets",
        )
        preflight = {
            "schema": f"{SCHEMA}.preflight",
            "passed": True,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "development_manifest_sha256": (
                development_materializer.SEALED_MANIFEST_SHA256
            ),
            "development_rows_opened": len(rows),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "split_payload_sha256": canonical_sha256(split_payload),
            "trainable_tensors": len(named),
            "trainable_elements": sum(parameter.numel() for _, parameter in named),
            "initial_parameter_sha256": initial_parameter_sha256,
            "training_target_mode": TRAINING_TARGET_MODE,
            "discriminative_target_payload_sha256": (
                discriminative_target_sha256
            ),
            "hardware": {
                "world_size": context.world_size,
                "devices": list(context.rank_devices),
                "four_distinct_a100s": True,
                "hf_endpoint": os.environ["HF_ENDPOINT"],
            },
            "protected_mechanics_rows_opened": 0,
            "protected_causal_rows_opened": 0,
            "native_benchmark_opened": False,
        }
        distributed.require_consensus(
            context, canonical_sha256(preflight), description="source-bound preflight"
        )
        if preflight_only:
            return preflight
        prepare_output(context, output_dir)
        input_binding = {
            "schema": INPUT_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "development_manifest_sha256": (
                development_materializer.SEALED_MANIFEST_SHA256
            ),
            "development_manifest_receipt": manifest["receipt"]["payload_sha256"],
            "development_split_receipt": manifest["split_contract"]["receipt"][
                "payload_sha256"
            ],
            "split": split_payload,
            "split_payload_sha256": canonical_sha256(split_payload),
            "base_model": str(base_model.expanduser().resolve(strict=True)),
            "materialization_root": str(materialization_root.resolve(strict=True)),
            "model_audit": model_audit,
            "hardware": preflight["hardware"],
            "runner_sha256": sha256_file(Path(__file__)),
            "initial_parameter_sha256": initial_parameter_sha256,
            "training_target_mode": TRAINING_TARGET_MODE,
            "discriminative_target_payload": discriminative_target_payload,
            "discriminative_target_payload_sha256": (
                discriminative_target_sha256
            ),
            "protected_splits_opened": [],
        }
        input_binding["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_input_without_receipt",
            "payload_sha256": canonical_sha256(input_binding),
        }
        if context.is_primary:
            signed_json(output_dir / "input_binding.json", input_binding)
        dist.barrier(group=context.control_group)

        assigned_rows = rows[context.process_rank :: WORLD_SIZE]
        if len(assigned_rows) != len(rows) // WORLD_SIZE:
            raise RuntimeError("Source-bound natural-cache rank assignment differs")
        local_cache = {}
        for ordinal, row in enumerate(assigned_rows, start=1):
            source = int(row["source_index"])
            batch = screen.mechanics.evolution.collate_native_examples(
                [examples[source]],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            state, audit, address = screen.mechanics.capture_write_condition(
                model,
                batch,
                modules,
                mode=screen.integration.CONTINUOUS_MODE,
                override=None,
                reference_mode="none",
            )
            screen.mechanics._clear_feature_references(modules)
            if (
                audit.get("formula_byte_exact_all_modules") is not True
                or audit.get("all_state_tensors_finite") is not True
            ):
                raise RuntimeError("Source-bound natural write audit differs")
            local_cache[source] = {"state": state, "address": address}
            print(
                f"SOURCE_BOUND_OUTER_FFN_WRITE rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/{len(assigned_rows)}",
                flush=True,
            )
        natural_cache = screen.gather_development_natural_cache(
            context, local_cache, expected_rows=len(rows)
        )
        candidates = screen.parent_runner.candidate_sources(rows)
        screen.clear_terminal_hooks(modules)
        screen.bind_terminal_hook(model, modules, terminal_layer=TERMINAL_ANCHOR)
        training = train_outer_ffn(
            model,
            router,
            named,
            train_rows,
            examples,
            natural_cache,
            candidates,
            discriminative_targets,
            modules,
            modules_by_layer,
            names_by_layer,
            ordered_names,
            context=context,
            pad_token_id=int(tokenizer.pad_token_id),
            output_dir=output_dir,
        )
        screen.clear_terminal_hooks(modules)
        heldout = evaluate_heldout(
            model,
            router,
            heldout_rows,
            examples,
            natural_cache,
            candidates,
            modules,
            modules_by_layer,
            names_by_layer,
            ordered_names,
            compatibility_maps,
            context=context,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        discriminative_heldout = evaluate_discriminative_heldout(
            model,
            router,
            heldout_rows,
            examples,
            natural_cache,
            candidates,
            discriminative_targets,
            modules,
            modules_by_layer,
            names_by_layer,
            ordered_names,
            context=context,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        if screen.parameter_versions(model) != model_versions_before:
            raise RuntimeError("Frozen model parameters changed during source-bound training")
        result: dict[str, Any] = {}
        save_error: BaseException | None = None
        if context.is_primary:
            try:
                checkpoint = save_checkpoint(named, output_dir / "outer_ffn.safetensors")
                passed = bool(
                    training["updates"] == UPDATES
                    and training["trainable_subset_changed"]
                    and training["all_step_gradient_contracts_passed"]
                    and heldout["passed"]
                    and discriminative_heldout["passed"]
                )
                result = {
                    "schema": SCHEMA,
                    "status": (
                        "open_heldout_passed_protected_mechanics_candidate"
                        if passed
                        else "open_heldout_failed_not_promoted"
                    ),
                    "passed": passed,
                    "protected_mechanics_authorized": passed,
                    "protected_causal_authorized": False,
                    "native_benchmark_authorized": False,
                    "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                    "input_binding": input_binding,
                    "architecture": protocol["architecture"],
                    "training": training,
                    "heldout": heldout,
                    "discriminative_heldout": discriminative_heldout,
                    "checkpoint": checkpoint,
                    "development_rows_opened": len(rows),
                    "protected_mechanics_rows_opened": 0,
                    "protected_causal_rows_opened": 0,
                    "native_benchmark_opened": False,
                }
                result["receipt"] = {
                    "algorithm": "sha256",
                    "payload_scope": "canonical_result_without_receipt",
                    "payload_sha256": canonical_sha256(result),
                }
                signed_json(output_dir / "result.json", result)
            except BaseException as caught:
                save_error = caught
        distributed.phase_consensus(
            context, phase="source-bound-result", error=save_error
        )
        passed_values = distributed.gather_objects(
            context, heldout["passed"] and discriminative_heldout["passed"]
        )
        worker_passed = all(bool(value) for value in passed_values)
        del model, examples, natural_cache
        gc.collect()
        torch.cuda.empty_cache()
        return result if context.is_primary else {
            "status": "worker_complete",
            "passed": worker_passed,
            "protected_mechanics_authorized": worker_passed,
        }
    finally:
        distributed.destroy_distributed_training(context)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        base_model=args.base_model,
        materialization_root=args.materialization_root,
        output_dir=args.output_dir,
        preflight_only=args.preflight_only,
    )
    print(
        json.dumps(
            {
                "status": result["status"] if "status" in result else "preflight_passed",
                "passed": result["passed"],
                "protected_mechanics_authorized": result.get(
                    "protected_mechanics_authorized", False
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
