#!/usr/bin/env python3
"""Cross-fit identity using untouched shadows from the trained DeepEmbed v5 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
SIGNED_SOURCE_ROOT_ENV = "RWKV_V5_EXACT_SOURCE_ROOT"
_signed_source_value = os.environ.get(SIGNED_SOURCE_ROOT_ENV)
SIGNED_SOURCE_ROOT = (
    None
    if not _signed_source_value
    else Path(_signed_source_value).expanduser().resolve()
)
if SIGNED_SOURCE_ROOT is not None:
    if not SIGNED_SOURCE_ROOT.is_dir():
        raise RuntimeError(
            f"{SIGNED_SOURCE_ROOT_ENV} is not a directory: {SIGNED_SOURCE_ROOT}"
        )
    try:
        sys.path.remove(str(SIGNED_SOURCE_ROOT))
    except ValueError:
        pass
    sys.path.insert(0, str(SIGNED_SOURCE_ROOT))

from common import load_model_and_tokenizer  # noqa: E402
from deltamem.core import delta_impl as core_impl  # noqa: E402
from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval as v5_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_query_state_bilinear_crossfit as source,
)


SCHEMA = "rwkv_ms_natural_memory_native_v5_shadow_crossfit.v1"
FEATURE_SCHEMA = "rwkv_ms_natural_memory_native_v5_shadow_features.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_v5_shadow_crossfit_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "faa6d1125a13506820485148885904fc7a1b35eef14bf57c6190106acc2f70bf"
WORLD_SIZE = 4
SEED = 118
HEAD_SEED = source.SEED
SPLIT_SALT = "rwkv-v5-shadow-crossfit-v1:"
TRAIN_ROWS = 176
HELDOUT_ROWS = 44
STATE_DIM = source.STATE_DIM
LAYERS = source.LAYERS
BOTTLENECK = source.BOTTLENECK
TRAIN_STEPS = source.TRAIN_STEPS
LEARNING_RATE = source.LEARNING_RATE
WEIGHT_DECAY = source.WEIGHT_DECAY
IDENTITY_MARGIN = source.IDENTITY_MARGIN
DONOR_FRACTION_GATE = source.DONOR_FRACTION_GATE
MEAN_GAP_GATE = source.MEAN_GAP_GATE
PERMUTED_FRACTION_GATE = source.PERMUTED_FRACTION_GATE
HF_ENDPOINT = "https://hf-mirror.com"
SIGNED_V5_COMMIT = "cd7deb91a3dbbf15b7f82cf7bb445e3d7664d631"
SIGNED_V5_DELTA_IMPL_SHA256 = "88c495a417fcff62b295b70971f1c02991aac3962f35dae3afa5affb0a808788"
V5_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5_r1"
V5_RESULT = V5_ROOT / "result.json"
V5_ADAPTER = V5_ROOT / "adapter"
V5_RESULT_SHA256 = "95376ed78da98cf36183146ce56a3623988e94645723c9b34aee0510e0457545"
V5_RESULT_RECEIPT = "7afee3fd1d88c7db91c86dd3f7febfd80656a35d54971fd824623a29883dba8e"
V5_ADAPTER_WEIGHTS_SHA256 = "87e7e2d0ee1db91ef59fb283c176e1a2838ebd74ddb7255f9fb41c05d5d42162"
V5_ADAPTER_CONFIG_SHA256 = "69d18784bb400fb51f38d8e073ed6acb83be54428bfd18644d9e4b833933be44"

distributed = source.distributed
evolution = source.evolution
causal_train = source.causal_train
contrast = source.candidate.shared.contrast
hardware = source.hardware
endpoint = source.endpoint
value_identity = source.value_identity
geometry = source.geometry
bilinear = source.bilinear


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    digest = canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    loader = protocol.get("exact_source_loader", {})
    capture = protocol.get("shadow_feature_capture", {})
    split = protocol.get("crossfit_split", {})
    training = protocol.get("offline_training", {})
    gates = protocol.get("heldout_gates", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("v5_result_sha256") != V5_RESULT_SHA256
        or authorization.get("v5_result_receipt") != V5_RESULT_RECEIPT
        or authorization.get("v5_adapter_weights_sha256")
        != V5_ADAPTER_WEIGHTS_SHA256
        or authorization.get("v5_adapter_config_sha256")
        != V5_ADAPTER_CONFIG_SHA256
        or frozen.get("capture_seed") != SEED
        or frozen.get("head_seed") != HEAD_SEED
        or frozen.get("signed_v5_source_commit") != SIGNED_V5_COMMIT
        or frozen.get("signed_v5_delta_impl_sha256")
        != SIGNED_V5_DELTA_IMPL_SHA256
        or loader.get("required_source_root_environment")
        != SIGNED_SOURCE_ROOT_ENV
        or loader.get("learned_write_installed") is not False
        or loader.get("config_overrides") != []
        or loader.get("initialize_missing_parameters") is not False
        or capture.get("model_output_changed") is not False
        or capture.get("binder_or_bridge_installed") is not False
        or split.get("selection_salt") != SPLIT_SALT
        or split.get("train_rows") != TRAIN_ROWS
        or split.get("heldout_rows") != HELDOUT_ROWS
        or training.get("steps") != TRAIN_STEPS
        or training.get("bottleneck") != BOTTLENECK
        or gates.get("donor_pairwise_positive_fraction_minimum")
        != DONOR_FRACTION_GATE
        or gates.get("donor_mean_gap_minimum") != MEAN_GAP_GATE
        or gates.get("layer_permuted_pairwise_positive_fraction_minimum")
        != PERMUTED_FRACTION_GATE
        or protocol.get("causal_mechanics_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Exact-v5 shadow cross-fit protocol differs")
    if sha256_file(V5_RESULT) != V5_RESULT_SHA256:
        raise ValueError("Exact-v5 training result differs")
    if sha256_file(V5_ADAPTER / "delta_mem_adapter.pt") != V5_ADAPTER_WEIGHTS_SHA256:
        raise ValueError("Exact-v5 adapter weights differ")
    if sha256_file(V5_ADAPTER / "delta_mem_config.json") != V5_ADAPTER_CONFIG_SHA256:
        raise ValueError("Exact-v5 adapter config differs")
    result = v5_eval.validate_train_result(V5_RESULT, adapter_dir=V5_ADAPTER)
    if (
        result.get("receipt", {}).get("payload_sha256") != V5_RESULT_RECEIPT
        or result.get("passed") is not True
        or result.get("open_native_generation_authorized") is not True
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Exact-v5 result does not authorize shadow capture")
    return protocol, result


def validate_execution_source() -> Mapping[str, Any]:
    if SIGNED_SOURCE_ROOT is None:
        raise RuntimeError(
            f"Set {SIGNED_SOURCE_ROOT_ENV} to a detached worktree at "
            f"{SIGNED_V5_COMMIT} before launching exact-v5 capture"
        )
    imported_core = Path(core_impl.__file__).resolve()
    if not imported_core.is_relative_to(SIGNED_SOURCE_ROOT):
        raise RuntimeError(
            "Imported Delta-Mem core is outside the signed source root: "
            f"root={SIGNED_SOURCE_ROOT} imported={imported_core}"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(SIGNED_SOURCE_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SIGNED_V5_COMMIT:
        raise RuntimeError(
            "Exact-v5 source worktree commit differs: "
            f"expected={SIGNED_V5_COMMIT} actual={commit}"
        )
    actual = sha256_file(Path(core_impl.__file__).resolve())
    if actual != SIGNED_V5_DELTA_IMPL_SHA256:
        raise RuntimeError(
            "Exact-v5 shadow capture requires the signed v5 Delta-Mem source "
            f"from commit {SIGNED_V5_COMMIT}; expected={SIGNED_V5_DELTA_IMPL_SHA256} "
            f"actual={actual}"
        )
    return {
        "signed_source_root": str(SIGNED_SOURCE_ROOT),
        "signed_v5_source_commit": SIGNED_V5_COMMIT,
        "signed_source_head": commit,
        "imported_delta_impl": str(imported_core),
        "delta_impl_sha256": actual,
    }


def crossfit_split(mapping: Mapping[int, int]) -> tuple[dict[int, str], Mapping[str, Any]]:
    components = source.donor_components(mapping)
    ordered = sorted(
        components,
        key=lambda component: (
            hashlib.sha256(
                (SPLIT_SALT + canonical_sha256(list(component))).encode("ascii")
            ).hexdigest(),
            component,
        ),
    )
    paths: dict[int, tuple[int, ...]] = {0: ()}
    for index, component in enumerate(ordered):
        for total, selected in tuple(sorted(paths.items(), reverse=True)):
            next_total = total + len(component)
            if next_total <= HELDOUT_ROWS and next_total not in paths:
                paths[next_total] = (*selected, index)
    if HELDOUT_ROWS not in paths:
        raise RuntimeError("Cannot construct exact v5 shadow heldout partition")
    heldout_components = set(paths[HELDOUT_ROWS])
    heldout = {
        source_index
        for index, component in enumerate(ordered)
        if index in heldout_components
        for source_index in component
    }
    split = {
        source_index: ("heldout" if source_index in heldout else "train")
        for source_index in sorted(mapping)
    }
    train = {source_index for source_index, name in split.items() if name == "train"}
    if len(train) != TRAIN_ROWS or len(heldout) != HELDOUT_ROWS:
        raise RuntimeError("Exact-v5 shadow cross-fit split sizes differ")
    if any(split[source_index] != split[donor] for source_index, donor in mapping.items()):
        raise RuntimeError("A donor edge crosses the v5 shadow cross-fit partition")
    return split, {
        "selection_salt": SPLIT_SALT,
        "component_count": len(ordered),
        "component_sizes": [len(component) for component in ordered],
        "heldout_component_indices": sorted(heldout_components),
        "train_sources": sorted(train),
        "heldout_sources": sorted(heldout),
    }


def authorized_examples(tokenizer: Any, dataset_root: Path) -> tuple[
    dict[int, Any], dict[int, Mapping[str, Any]], dict[int, int], Mapping[str, Any]
]:
    path = dataset_root / endpoint.DATASET_RELATIVE_PATH
    if sha256_file(path) != endpoint.DATASET_SHA256:
        raise ValueError("Authorized native development dataset differs")
    metadata = endpoint.raw_line_metadata(path)
    rows = endpoint.parse_authorized_rows(metadata)
    counts = endpoint.prompt_token_counts(tokenizer, rows)
    mapping = endpoint.build_donor_mapping(rows, counts)
    split, split_payload = crossfit_split(mapping)
    selected = {
        int(row["source_index"]): row for row in endpoint.authorized_metadata(metadata)
    }
    examples = {
        source_index: evolution.encode_native_full_row(
            tokenizer,
            task="scene",
            source_ordinal=source_index,
            raw_line=bytes(record["raw_line"]).decode("utf-8"),
        )
        for source_index, record in selected.items()
    }
    rows_by_source = {int(row["source_index"]): row for row in rows}
    if set(examples) != set(mapping) or len(examples) != endpoint.EVALUATION_ROWS:
        raise RuntimeError("Authorized exact-v5 example coverage differs")
    return examples, rows_by_source, mapping, {
        **dict(split_payload),
        "rows": [
            {
                "source_index": source_index,
                "row_sha256": rows_by_source[source_index]["row_sha256"],
                "donor_source_index": mapping[source_index],
                "donor_row_sha256": rows_by_source[mapping[source_index]]["row_sha256"],
                "split": split[source_index],
            }
            for source_index in sorted(mapping)
        ],
    }


def clone_online_state(
    modules: Sequence[tuple[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    captured = causal_train.capture_online_state_references(modules)
    return {
        name: {
            attribute: tensor.detach().clone()
            for attribute, tensor in values.items()
        }
        for name, values in captured.items()
    }


def aggregate_vectors(
    captured: Sequence[Any], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return geometry._aggregate_answer_vectors(captured, labels)


def _capture_condition(
    model: torch.nn.Module,
    target: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    target_values: Mapping[str, torch.Tensor],
    rotate_recurrent_layers: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    reset_delta_mem_states(model)
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=rotate_recurrent_layers,
    )
    value_identity.set_fixed_target_values(model, dict(target_values))
    logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    query, shadow = aggregate_vectors(value_identity.capture(model), target.labels)
    del logits
    return query, shadow, bool(fixed)


def capture_row(
    model: torch.nn.Module,
    target_example: Any,
    donor_example: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    target = evolution.collate_native_examples(
        [target_example], pad_token_id=pad_token_id, device=device
    )
    donor = contrast.build_donor_batch(target, donor_example, device=device)
    try:
        with torch.inference_mode():
            value_identity.clear(model)
            evolution._native_write(model, target, dtype=torch.bfloat16)
            correct_state = clone_online_state(modules)
            target_values = {
                name: value.detach().clone()
                for name, value in value_identity.capture_write_values(model).items()
            }
            query, correct, correct_fixed = _capture_condition(
                model,
                target,
                modules,
                projected=correct_state,
                recurrent=correct_state,
                target_values=target_values,
                rotate_recurrent_layers=False,
            )

            value_identity.clear(model)
            evolution._native_write(model, donor, dtype=torch.bfloat16)
            donor_state = clone_online_state(modules)
            donor_query, donor_shadow, donor_fixed = _capture_condition(
                model,
                target,
                modules,
                projected=correct_state,
                recurrent=donor_state,
                target_values=target_values,
                rotate_recurrent_layers=False,
            )
            permuted_query, permuted_shadow, permuted_fixed = _capture_condition(
                model,
                target,
                modules,
                projected=correct_state,
                recurrent=correct_state,
                target_values=target_values,
                rotate_recurrent_layers=True,
            )
        if not (
            correct_fixed
            and donor_fixed
            and permuted_fixed
            and torch.equal(query, donor_query)
            and torch.equal(query, permuted_query)
        ):
            raise RuntimeError("Exact-v5 projected query or carrier changed")
        tensors = (query, correct, donor_shadow, permuted_shadow)
        if any(tuple(tensor.shape) != (LAYERS, STATE_DIM) for tensor in tensors):
            raise RuntimeError("Exact-v5 shadow feature shape differs")
        if not bool(torch.isfinite(torch.stack(tensors)).all().item()):
            raise RuntimeError("Exact-v5 shadow feature is non-finite")
        return {
            "query": query.cpu().tolist(),
            "correct": correct.cpu().tolist(),
            "matched_donor": donor_shadow.cpu().tolist(),
            "layer_permuted": permuted_shadow.cpu().tolist(),
            "projected_carrier_fixed": True,
            "state_snapshots_detached_and_cloned": True,
            "binder_or_bridge_installed": False,
        }
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


LayerwiseBilinear = source.LayerwiseBilinear
score_metrics = source.score_metrics


def train_and_evaluate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["source_index"]))
    feature = {
        name: torch.tensor([row[name] for row in ordered], dtype=torch.float32)
        for name in ("query", "correct", "matched_donor", "layer_permuted")
    }
    train_index = torch.tensor(
        [index for index, row in enumerate(ordered) if row["split"] == "train"],
        dtype=torch.long,
    )
    heldout_index = torch.tensor(
        [index for index, row in enumerate(ordered) if row["split"] == "heldout"],
        dtype=torch.long,
    )
    if train_index.numel() != TRAIN_ROWS or heldout_index.numel() != HELDOUT_ROWS:
        raise RuntimeError("Exact-v5 shadow analyzer row counts differ")
    torch.manual_seed(HEAD_SEED)
    head = LayerwiseBilinear()
    train = {name: value.index_select(0, train_index) for name, value in feature.items()}
    heldout = {
        name: value.index_select(0, heldout_index) for name, value in feature.items()
    }
    initial = {
        "donor": score_metrics(
            head, heldout["query"], heldout["correct"], heldout["matched_donor"]
        ),
        "layer_permuted": score_metrics(
            head, heldout["query"], heldout["correct"], heldout["layer_permuted"]
        ),
    }
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    losses: list[float] = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        correct_score = head.score(train["query"], train["correct"])
        donor_score = head.score(train["query"], train["matched_donor"])
        permuted_score = head.score(train["query"], train["layer_permuted"])
        loss = F.relu(IDENTITY_MARGIN - correct_score + donor_score).mean()
        loss = loss + F.relu(
            IDENTITY_MARGIN - correct_score + permuted_score
        ).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Exact-v5 shadow cross-fit loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    final = {
        "train": {
            "donor": score_metrics(
                head, train["query"], train["correct"], train["matched_donor"]
            ),
            "layer_permuted": score_metrics(
                head, train["query"], train["correct"], train["layer_permuted"]
            ),
        },
        "heldout": {
            "donor": score_metrics(
                head,
                heldout["query"],
                heldout["correct"],
                heldout["matched_donor"],
            ),
            "layer_permuted": score_metrics(
                head,
                heldout["query"],
                heldout["correct"],
                heldout["layer_permuted"],
            ),
        },
    }
    donor = final["heldout"]["donor"]
    permuted = final["heldout"]["layer_permuted"]
    checks = {
        "heldout_donor_pairwise_positive_fraction": (
            donor["pairwise_positive_fraction"] >= DONOR_FRACTION_GATE
        ),
        "heldout_donor_mean_gap": donor["mean_gap"] >= MEAN_GAP_GATE,
        "heldout_layer_permuted_pairwise_positive_fraction": (
            permuted["pairwise_positive_fraction"] >= PERMUTED_FRACTION_GATE
        ),
        "all_heldout_scores_finite": donor["finite"] and permuted["finite"],
    }
    return {
        "head": bilinear.audit_payload(STATE_DIM, BOTTLENECK),
        "optimizer": {
            "name": "AdamW",
            "seed": HEAD_SEED,
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "identity_margin": IDENTITY_MARGIN,
        },
        "initial_heldout": initial,
        "loss": {"initial": losses[0], "final": losses[-1]},
        "metrics": final,
        "checks": checks,
        "passed": all(checks.values()),
        "weights_saved": False,
    }


def load_feature_records(
    output_dir: Path, split_payload: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    split_by_source = {
        int(row["source_index"]): str(row["split"])
        for row in split_payload["rows"]
    }
    records: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    for shard_index in range(WORLD_SIZE):
        path = output_dir / f"shard-{shard_index}.jsonl"
        shard_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records.extend(shard_rows)
        provenance.append(
            {"path": str(path), "rows": len(shard_rows), "sha256": sha256_file(path)}
        )
    sources = [int(row["source_index"]) for row in records]
    if len(records) != endpoint.EVALUATION_ROWS or len(set(sources)) != len(sources):
        raise RuntimeError("Exact-v5 shadow shard coverage differs")
    for row in records:
        source_index = int(row["source_index"])
        if (
            row.get("schema") != FEATURE_SCHEMA
            or row.get("split") != split_by_source[source_index]
            or row.get("projected_carrier_fixed") is not True
            or row.get("state_snapshots_detached_and_cloned") is not True
            or row.get("binder_or_bridge_installed") is not False
        ):
            raise RuntimeError("Exact-v5 shadow feature row contract differs")
    return records, provenance


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_exact_v5_model(
    base_model: Path, *, device: torch.device
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=V5_ADAPTER,
        device=str(device),
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    v5_eval._assert_deepembed_modules(model)
    modules = causal_train.ordered_modules(model)
    learned_write_modules = sum(
        hasattr(module, "rwkv_learned_write_k_down") for _, module in modules
    )
    active_outer_ffn_layers = [
        int(module.layer_idx) for _, module in modules if module.rwkv_ms_outer_ffn_enabled
    ]
    if learned_write_modules or active_outer_ffn_layers != [10, 21, 31, 41]:
        raise RuntimeError("Strict-loaded v5 source topology differs")
    return model, tokenizer, {
        "loader": "strict_saved_config_and_complete_adapter",
        "adapter_weights_sha256": V5_ADAPTER_WEIGHTS_SHA256,
        "adapter_config_sha256": V5_ADAPTER_CONFIG_SHA256,
        "wrapped_layers": len(modules),
        "learned_write_modules": learned_write_modules,
        "outer_ffn_layers": active_outer_ffn_layers,
        "deepembed_pre_hooks": sum(
            module._deepembed_ffn_pre_hook_handle is not None for _, module in modules
        ),
        "deepembed_down_hooks": sum(
            module._deepembed_ffn_down_pre_hook_handle is not None
            for _, module in modules
        ),
    }


def run(
    *, base_model: Path, dataset_root: Path, output_dir: Path
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training("cuda")
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol, _ = validate_protocol()
        source_audit = validate_execution_source()
        if context.world_size != WORLD_SIZE or not hardware.four_distinct_a100s(
            context.rank_devices
        ):
            raise RuntimeError("Exact-v5 shadow capture requires four distinct A100s")
        if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
            raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
        fresh_error: BaseException | None = None
        if context.is_primary and output_dir.exists():
            fresh_error = ValueError(f"Exact-v5 shadow output must be fresh: {output_dir}")
        distributed.phase_consensus(
            context, phase="v5-shadow-crossfit-fresh", error=fresh_error
        )
        create_error: BaseException | None = None
        if context.is_primary:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                create_error = error
        distributed.phase_consensus(
            context, phase="v5-shadow-crossfit-create", error=create_error
        )

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, model_audit = load_exact_v5_model(
            base_model, device=context.device
        )
        capture_audit = value_identity.install(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        examples, rows, mapping, split_payload = authorized_examples(
            tokenizer, dataset_root
        )
        split = {
            int(row["source_index"]): str(row["split"])
            for row in split_payload["rows"]
        }
        shard_path = output_dir / f"shard-{context.process_rank}.jsonl"
        shard_sources = [
            source_index
            for source_index in sorted(examples)
            if source_index % WORLD_SIZE == context.process_rank
        ]
        for ordinal, source_index in enumerate(shard_sources, start=1):
            donor = mapping[source_index]
            feature = capture_row(
                model,
                examples[source_index],
                examples[donor],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            append_jsonl(
                shard_path,
                {
                    "schema": FEATURE_SCHEMA,
                    "source_index": source_index,
                    "row_sha256": rows[source_index]["row_sha256"],
                    "donor_source_index": donor,
                    "donor_row_sha256": rows[donor]["row_sha256"],
                    "split": split[source_index],
                    **feature,
                },
            )
            print(
                f"V5_SHADOW_CROSSFIT rank={context.process_rank} "
                f"row={source_index} ordinal={ordinal}/{len(shard_sources)}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()
        dist.barrier()

        result: dict[str, Any] = {}
        if context.is_primary:
            records, provenance = load_feature_records(output_dir, split_payload)
            analysis = train_and_evaluate(records)
            passed = bool(analysis["passed"])
            result = {
                "schema": SCHEMA,
                "status": (
                    "v5_shadow_crossfit_passed_mechanics_design_authorized"
                    if passed
                    else "v5_shadow_crossfit_failed_identity_family_retired"
                ),
                "passed": passed,
                "causal_mechanics_design_authorized": passed,
                "training_authorized": False,
                "generation_authorized": False,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "protocol_objective": protocol["objective"],
                "capture_seed": SEED,
                "head_seed": HEAD_SEED,
                "base_model": str(base_model),
                "dataset_root": str(dataset_root),
                "dataset_file": str(dataset_root / endpoint.DATASET_RELATIVE_PATH),
                "dataset_sha256": endpoint.DATASET_SHA256,
                "authorized_rows_payload_sha256": endpoint.AUTHORIZED_ROWS_PAYLOAD_SHA256,
                "rows": len(records),
                "crossfit_split": {
                    **dict(split_payload),
                    "payload_sha256": canonical_sha256(split_payload),
                },
                "feature_provenance": provenance,
                "analysis": analysis,
                "hardware": {
                    "world_size": WORLD_SIZE,
                    "rank_devices": list(context.rank_devices),
                },
                "source_audit": source_audit,
                "model_audit": {
                    **dict(model_audit),
                    "capture": capture_audit,
                    "output_changed_by_capture": False,
                    "binder_or_bridge_installed": False,
                },
                "v5_provenance": {
                    "result_sha256": V5_RESULT_SHA256,
                    "result_receipt": V5_RESULT_RECEIPT,
                    "adapter_weights_sha256": V5_ADAPTER_WEIGHTS_SHA256,
                    "adapter_config_sha256": V5_ADAPTER_CONFIG_SHA256,
                },
                "no_adapter_weights_saved": True,
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "bilinear_helper_sha256": sha256_file(Path(bilinear.__file__)),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            (output_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dist.barrier()
        return result
    finally:
        distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
