#!/usr/bin/env python3
"""Fail-fast prompt-latched identity cross-fit over signed predictor features."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics as parent,
)


SCHEMA = "rwkv_ms_natural_memory_native_plat_prompt_latch_crossfit.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_plat_prompt_latch_crossfit_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "25f486f4242ada2dd1043fdaf1682663b812b2a4b3d4bcbdd95bd760181bc1bd"
WORLD_SIZE = 4
DISTRIBUTED_TIMEOUT_SECONDS = 1800
SPLIT_SALT = "rwkv-plat-prompt-latch-crossfit-v1:"
TRAIN_ROWS = 132
HELDOUT_ROWS = 44
EXCLUDED_ROWS = 44
LAYERS = parent.LAYERS
STATE_DIM = parent.STATE_DIM
HEAD_SEED = parent.HEAD_SEED
TRAIN_STEPS = parent.TRAIN_STEPS
LEARNING_RATE = parent.LEARNING_RATE
WEIGHT_DECAY = parent.WEIGHT_DECAY
IDENTITY_MARGIN = parent.IDENTITY_MARGIN
DONOR_FRACTION_GATE = parent.DONOR_FRACTION_GATE
MEAN_GAP_GATE = parent.MEAN_GAP_GATE
PERMUTED_FRACTION_GATE = parent.PERMUTED_FRACTION_GATE
HF_ENDPOINT = "https://hf-mirror.com"

PARENT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics_v1"
)
PARENT_RESULT = PARENT_ROOT / "result.json"
PARENT_RESULT_SHA256 = "44e4b22c6db8b9c9e98a947ec9baf829291bb49efd2b5dc5b21ca98574ca9cbb"
PARENT_RECEIPT = "3489154f6bae3feadd3510ca2aeddce31dea3fb3d5e5f995c043bf466c544959"
PARENT_RUNNER_SHA256 = "819e1ed28310a010cf03c0dbae1e6e77c991fd610e998fc3bcc2e2b4d0933275"
PARENT_SHADOW_RUNNER_SHA256 = "0fd85e3e415a8705d3e37efd47d88f5ac6aa8feb25af039abdd1e4d704151ed6"
PARENT_HEAD_RUNNER_SHA256 = "207ea694ebe96aaf4a5f4e9bb0170ce15ef79c0777473e49d690b1ab098d762f"
PARENT_BILINEAR_HELPER_SHA256 = "1c63bdbcfb9db9dec5b225131c83d6ec1774459ef2fa09d18c2688ccc1fc6be9"
PARENT_PROTOCOL_PAYLOAD_SHA256 = "c6f8393baf0b8ef778f6b9125c9b26887ac844bbd4f62213c0e2e39d7ff2630c"
FEATURE_SCHEMA = parent.FEATURE_SCHEMA
SHARD_BINDINGS = (
    (0, 54, "667ed132c44021b3e27b52011271811cf4b57f8579e5ad77f03d5d8d42a6b586"),
    (1, 63, "528803ab052aaac2a186d66a6fc87d784bb457ec798185c45ee2fedee42119fc"),
    (2, 48, "437f097f8b5299baae42f9bc0eac366f06c9b6d4cf00682e338f79f39b39df61"),
    (3, 55, "3c1773836354abfbef32f21d885ee169053a9fab5929d7ac12c53b2c7b34493e"),
)

LayerwiseBilinear = parent.LayerwiseBilinear
bilinear = parent.bilinear


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def donor_components(
    mapping: Mapping[int, int],
    eligible: set[int],
) -> tuple[tuple[int, ...], ...]:
    if set(mapping) != eligible or set(mapping.values()) - eligible:
        raise ValueError("PLAT donor mapping differs from the eligible source set")
    adjacency = {source: set() for source in eligible}
    for source, donor in mapping.items():
        adjacency[source].add(donor)
        adjacency[donor].add(source)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in sorted(eligible):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            source = stack.pop()
            component.append(source)
            for neighbor in sorted(adjacency[source], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    if seen != eligible:
        raise RuntimeError("PLAT donor component coverage differs")
    return tuple(components)


def nested_split(
    mapping: Mapping[int, int],
    eligible: set[int],
    excluded: set[int],
) -> tuple[dict[int, str], Mapping[str, Any]]:
    if eligible & excluded or len(eligible) != TRAIN_ROWS + HELDOUT_ROWS:
        raise ValueError("PLAT eligible and excluded source sets differ")
    components = donor_components(mapping, eligible)
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
        raise RuntimeError("Cannot construct the precommitted PLAT heldout split")
    heldout_component_indices = set(paths[HELDOUT_ROWS])
    heldout = {
        source
        for index, component in enumerate(ordered)
        if index in heldout_component_indices
        for source in component
    }
    train = eligible - heldout
    split = {
        source: ("heldout" if source in heldout else "train")
        for source in sorted(eligible)
    }
    if len(train) != TRAIN_ROWS or len(heldout) != HELDOUT_ROWS:
        raise RuntimeError("PLAT nested split sizes differ")
    if any(split[source] != split[donor] for source, donor in mapping.items()):
        raise RuntimeError("A donor edge crosses the PLAT nested split")
    payload = {
        "selection_salt": SPLIT_SALT,
        "component_count": len(ordered),
        "component_sizes": [len(component) for component in ordered],
        "heldout_component_indices": sorted(heldout_component_indices),
        "train_sources": sorted(train),
        "heldout_sources": sorted(heldout),
        "excluded_prior_heldout_sources": sorted(excluded),
    }
    return split, payload


def _expected_shard_payload() -> list[Mapping[str, Any]]:
    return [
        {
            "basename": f"stage1-shard-{shard_index}.jsonl",
            "rows": rows,
            "sha256": digest,
        }
        for shard_index, rows, digest in SHARD_BINDINGS
    ]


def _expected_dependency_payload() -> list[Mapping[str, str]]:
    return [
        {
            "role": "predictor_feature_and_metric_runner",
            "basename": Path(parent.__file__).name,
            "sha256": PARENT_RUNNER_SHA256,
        },
        {
            "role": "exact_v5_shadow_crossfit_runner",
            "basename": Path(parent.shadow.__file__).name,
            "sha256": PARENT_SHADOW_RUNNER_SHA256,
        },
        {
            "role": "layerwise_head_definition",
            "basename": Path(parent.shadow.source.__file__).name,
            "sha256": PARENT_HEAD_RUNNER_SHA256,
        },
        {
            "role": "bilinear_audit_helper",
            "basename": Path(parent.bilinear.__file__).name,
            "sha256": PARENT_BILINEAR_HELPER_SHA256,
        },
    ]


def validate_parent_result() -> Mapping[str, Any]:
    current_dependencies = (
        (Path(parent.__file__), PARENT_RUNNER_SHA256),
        (Path(parent.shadow.__file__), PARENT_SHADOW_RUNNER_SHA256),
        (Path(parent.shadow.source.__file__), PARENT_HEAD_RUNNER_SHA256),
        (Path(parent.bilinear.__file__), PARENT_BILINEAR_HELPER_SHA256),
    )
    for path, expected in current_dependencies:
        if sha256_file(path) != expected:
            raise ValueError(f"Imported PLAT code dependency differs: {path}")
    if sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise ValueError("Signed predictor parent result differs")
    result = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt", {})
    if (
        result.get("schema") != parent.SCHEMA
        or result.get("status") != "predictor_crossfit_failed_stage2_not_run"
        or result.get("passed") is not False
        or result.get("stage2_executed") is not False
        or result.get("stage2") is not None
        or result.get("protected_splits_opened") != []
        or result.get("model_or_adapter_training_authorized") is not False
        or result.get("generation_authorized") is not False
        or result.get("protocol_payload_sha256") != PARENT_PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PARENT_RECEIPT
        or canonical_sha256(unsigned) != PARENT_RECEIPT
        or result.get("code_bindings", {}).get("runner_sha256")
        != PARENT_RUNNER_SHA256
    ):
        raise ValueError("Signed predictor parent result contract differs")
    provenance = [
        {
            "basename": Path(item["path"]).name,
            "rows": int(item["rows"]),
            "sha256": item["sha256"],
        }
        for item in result.get("feature_provenance", [])
    ]
    if provenance != _expected_shard_payload():
        raise ValueError("Signed predictor shard provenance differs")
    for item in provenance:
        path = PARENT_ROOT / item["basename"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Signed predictor feature shard differs: {path}")
    return result


def validate_protocol() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    digest = canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    source_scope = protocol.get("source_scope", {})
    split_spec = protocol.get("precommitted_nested_split", {})
    transform = protocol.get("prompt_latch_transform", {})
    fit = protocol.get("offline_fit", {})
    gates = protocol.get("heldout_gates", {})
    execution = protocol.get("execution", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("parent_result_sha256") != PARENT_RESULT_SHA256
        or authorization.get("parent_result_receipt") != PARENT_RECEIPT
        or authorization.get("parent_runner_sha256") != PARENT_RUNNER_SHA256
        or authorization.get("code_dependencies")
        != _expected_dependency_payload()
        or authorization.get("parent_protocol_payload_sha256")
        != PARENT_PROTOCOL_PAYLOAD_SHA256
        or authorization.get("feature_shards") != _expected_shard_payload()
        or source_scope.get("eligible_rows") != TRAIN_ROWS + HELDOUT_ROWS
        or source_scope.get("excluded_rows") != EXCLUDED_ROWS
        or split_spec.get("selection_salt") != SPLIT_SALT
        or split_spec.get("train_rows") != TRAIN_ROWS
        or split_spec.get("heldout_rows") != HELDOUT_ROWS
        or transform.get("query_source") != "first_causal_predictor_query"
        or transform.get("expansion") != "byte_identical_across_all_predictor_tokens"
        or transform.get("recurrent_features_changed") is not False
        or fit.get("head_seed") != HEAD_SEED
        or fit.get("steps") != TRAIN_STEPS
        or fit.get("learning_rate") != LEARNING_RATE
        or fit.get("weight_decay") != WEIGHT_DECAY
        or fit.get("identity_margin") != IDENTITY_MARGIN
        or fit.get("heldout_used_for_fit_thresholds_or_selection") is not False
        or gates.get("donor_token_pairwise_positive_fraction_minimum")
        != DONOR_FRACTION_GATE
        or gates.get("donor_row_pairwise_positive_fraction_minimum")
        != DONOR_FRACTION_GATE
        or gates.get("donor_mean_gap_minimum") != MEAN_GAP_GATE
        or gates.get("layer_permuted_token_pairwise_positive_fraction_minimum")
        != PERMUTED_FRACTION_GATE
        or gates.get("layer_permuted_row_pairwise_positive_fraction_minimum")
        != PERMUTED_FRACTION_GATE
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("timeout_seconds") != DISTRIBUTED_TIMEOUT_SECONDS
        or execution.get("rank0_only_fit") is not True
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or protocol.get("model_or_adapter_training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("stage2_authorized") is not False
        or protocol.get("weights_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Signed PLAT prompt-latch protocol differs")
    parent_result = validate_parent_result()
    prior_train = set(parent_result["crossfit_split"]["train_sources"])
    excluded = set(parent_result["crossfit_split"]["heldout_sources"])
    rows = parent_result["crossfit_split"]["rows"]
    mapping = {
        int(row["source_index"]): int(row["donor_source_index"])
        for row in rows
        if int(row["source_index"]) in prior_train
    }
    split, split_payload = nested_split(mapping, prior_train, excluded)
    del split
    signed_payload = {
        key: split_spec[key]
        for key in (
            "selection_salt",
            "component_count",
            "component_sizes",
            "heldout_component_indices",
            "train_sources",
            "heldout_sources",
            "excluded_prior_heldout_sources",
        )
    }
    if (
        signed_payload != split_payload
        or split_spec.get("payload_sha256") != canonical_sha256(split_payload)
        or source_scope.get("excluded_prior_heldout_sources")
        != sorted(excluded)
    ):
        raise ValueError("Precommitted PLAT nested split differs")
    return protocol, parent_result, split_payload


def _validate_feature_shape(value: Any, *, predictor_tokens: int) -> None:
    if not isinstance(value, list) or len(value) != predictor_tokens:
        raise ValueError("PLAT feature predictor-token axis differs")
    if any(
        not isinstance(token, list)
        or len(token) != LAYERS
        or any(not isinstance(layer, list) or len(layer) != STATE_DIM for layer in token)
        for token in value
    ):
        raise ValueError("PLAT feature layer/state axes differ")


def load_eligible_records(
    parent_result: Mapping[str, Any],
    split_payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    train_sources = set(split_payload["train_sources"])
    heldout_sources = set(split_payload["heldout_sources"])
    eligible = train_sources | heldout_sources
    excluded = set(split_payload["excluded_prior_heldout_sources"])
    metadata = {
        int(row["source_index"]): row
        for row in parent_result["crossfit_split"]["rows"]
    }
    records: list[Mapping[str, Any]] = []
    observed: set[int] = set()
    for shard_index, expected_rows, _ in SHARD_BINDINGS:
        path = PARENT_ROOT / f"stage1-shard-{shard_index}.jsonl"
        shard_rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                shard_rows += 1
                row = json.loads(line)
                source = int(row["source_index"])
                if source in observed:
                    raise RuntimeError("Duplicate signed predictor feature source")
                observed.add(source)
                if source % WORLD_SIZE != shard_index:
                    raise RuntimeError("Signed predictor feature shard assignment differs")
                expected = metadata[source]
                if (
                    row.get("schema") != FEATURE_SCHEMA
                    or row.get("row_sha256") != expected["row_sha256"]
                    or int(row.get("donor_source_index"))
                    != int(expected["donor_source_index"])
                    or row.get("donor_row_sha256") != expected["donor_row_sha256"]
                    or row.get("split") != expected["split"]
                    or row.get("feature_positions")
                    != "labels[:,1:]_shifted_one_token_left"
                    or row.get("projected_carrier_fixed") is not True
                    or row.get("state_snapshots_detached_and_cloned") is not True
                    or row.get("binder_or_feedback_installed") is not False
                ):
                    raise RuntimeError("Signed predictor feature metadata differs")
                if source in excluded:
                    continue
                if source not in eligible or row.get("split") != "train":
                    raise RuntimeError("PLAT source scope includes a forbidden row")
                predictor_tokens = int(row["predictor_tokens"])
                if predictor_tokens < 1:
                    raise ValueError("PLAT row has no causal predictor token")
                for name in ("query", "correct", "matched_donor", "layer_permuted"):
                    _validate_feature_shape(row[name], predictor_tokens=predictor_tokens)
                records.append(
                    {
                        **row,
                        "plat_split": (
                            "train" if source in train_sources else "heldout"
                        ),
                    }
                )
        if shard_rows != expected_rows:
            raise RuntimeError("Signed predictor feature shard row count differs")
    if observed != set(metadata) or len(records) != TRAIN_ROWS + HELDOUT_ROWS:
        raise RuntimeError("PLAT signed feature coverage differs")
    if {int(row["source_index"]) for row in records} != eligible:
        raise RuntimeError("PLAT eligible feature sources differ")
    return records


def prompt_latch_query(query: torch.Tensor) -> torch.Tensor:
    if query.ndim != 3 or tuple(query.shape[1:]) != (LAYERS, STATE_DIM):
        raise ValueError("PLAT query must be [predictor, layer, state]")
    if query.size(0) < 1 or not bool(torch.isfinite(query).all().item()):
        raise ValueError("PLAT query is empty or non-finite")
    latch = query[0:1].expand(query.size(0), -1, -1)
    if any(not torch.equal(latch[index], query[0]) for index in range(query.size(0))):
        raise RuntimeError("PLAT latch expansion is not byte-identical")
    return latch


def _feature_tensors(
    records: Sequence[Mapping[str, Any]],
    split: str,
) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
    selected = [
        row
        for row in sorted(records, key=lambda item: int(item["source_index"]))
        if row["plat_split"] == split
    ]
    expected_rows = TRAIN_ROWS if split == "train" else HELDOUT_ROWS
    if len(selected) != expected_rows:
        raise RuntimeError(f"PLAT {split} row count differs")
    lengths = tuple(int(row["predictor_tokens"]) for row in selected)
    features: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("query", "correct", "matched_donor", "layer_permuted")
    }
    for row in selected:
        query = torch.tensor(row["query"], dtype=torch.float32)
        features["query"].append(prompt_latch_query(query))
        for name in ("correct", "matched_donor", "layer_permuted"):
            value = torch.tensor(row[name], dtype=torch.float32)
            if tuple(value.shape) != tuple(query.shape) or not bool(
                torch.isfinite(value).all().item()
            ):
                raise RuntimeError("PLAT recurrent feature shape or finiteness differs")
            features[name].append(value)
    return {
        name: torch.cat(values, dim=0) for name, values in features.items()
    }, lengths


def derive_train_only_thresholds(
    head: LayerwiseBilinear,
    train: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        correct = head.score(train["query"], train["correct"])
        donor = head.score(train["query"], train["matched_donor"])
        permuted = head.score(train["query"], train["layer_permuted"])
        negative = torch.maximum(donor, permuted)
        thresholds = 0.5 * (
            torch.quantile(correct, 0.05, dim=0)
            + torch.quantile(negative, 0.95, dim=0)
        )
    if tuple(thresholds.shape) != (LAYERS,) or not bool(
        torch.isfinite(thresholds).all().item()
    ):
        raise RuntimeError("PLAT train-only thresholds differ")
    return thresholds


def fit_train_only(
    records: Sequence[Mapping[str, Any]],
) -> tuple[LayerwiseBilinear, torch.Tensor, Mapping[str, Any]]:
    train, train_lengths = _feature_tensors(records, "train")
    torch.manual_seed(HEAD_SEED)
    head = LayerwiseBilinear()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    losses: list[float] = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        correct = head.score(train["query"], train["correct"])
        donor = head.score(train["query"], train["matched_donor"])
        permuted = head.score(train["query"], train["layer_permuted"])
        loss = F.relu(IDENTITY_MARGIN - correct + donor).mean()
        loss = loss + F.relu(IDENTITY_MARGIN - correct + permuted).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("PLAT prompt-latch fit loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    thresholds = derive_train_only_thresholds(head, train)
    train_metrics = {
        name: parent.predictor_score_metrics(
            head,
            train["query"],
            train["correct"],
            train[field],
            train_lengths,
        )
        for name, field in (
            ("donor", "matched_donor"),
            ("layer_permuted", "layer_permuted"),
        )
    }
    return head, thresholds, {
        "optimizer": {
            "name": "AdamW",
            "seed": HEAD_SEED,
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "identity_margin": IDENTITY_MARGIN,
        },
        "loss": {"initial": losses[0], "final": losses[-1]},
        "train_metrics": train_metrics,
        "thresholds": {
            "source": "nested_132_train_rows_only",
            "method": "midpoint_q05_correct_q95_max_negative_per_layer",
            "values": thresholds.tolist(),
        },
    }


def evaluate_heldout(
    head: LayerwiseBilinear,
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    heldout, heldout_lengths = _feature_tensors(records, "heldout")
    metrics = {
        name: parent.predictor_score_metrics(
            head,
            heldout["query"],
            heldout["correct"],
            heldout[field],
            heldout_lengths,
        )
        for name, field in (
            ("donor", "matched_donor"),
            ("layer_permuted", "layer_permuted"),
        )
    }
    donor = metrics["donor"]
    permuted = metrics["layer_permuted"]
    checks = {
        "heldout_donor_token_fraction": (
            donor["token_pairwise_positive_fraction"] >= DONOR_FRACTION_GATE
        ),
        "heldout_donor_row_fraction": (
            donor["row_pairwise_positive_fraction"] >= DONOR_FRACTION_GATE
        ),
        "heldout_donor_mean_gap": donor["mean_gap"] >= MEAN_GAP_GATE,
        "heldout_layer_permuted_token_fraction": (
            permuted["token_pairwise_positive_fraction"]
            >= PERMUTED_FRACTION_GATE
        ),
        "heldout_layer_permuted_row_fraction": (
            permuted["row_pairwise_positive_fraction"]
            >= PERMUTED_FRACTION_GATE
        ),
        "all_heldout_scores_finite": donor["finite"] and permuted["finite"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def analyze(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    head, thresholds, train = fit_train_only(records)
    heldout = evaluate_heldout(head, records)
    return {
        "head": bilinear.audit_payload(STATE_DIM, parent.shadow.BOTTLENECK),
        "prompt_latch": {
            "source": "first_causal_predictor_query",
            "expanded_byte_identically_across_predictor_tokens": True,
            "recurrent_features_changed": False,
        },
        **dict(train),
        "heldout": heldout,
        "passed": bool(heldout["passed"]),
        "head_weights_saved": False,
        "threshold_tensor_returned_only_for_runtime_audit": int(thresholds.numel()),
    }


def _initialize_distributed() -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise RuntimeError("Run PLAT prompt-latch cross-fit with torchrun --nproc_per_node=4")
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=DISTRIBUTED_TIMEOUT_SECONDS),
    )
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("PLAT prompt-latch cross-fit requires exactly four ranks")
    return dist.get_rank()


def _require_consensus(value: Any, description: str) -> None:
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, value)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise RuntimeError(f"Distributed PLAT {description} differs: {gathered!r}")


def _consensual_operation(operation: Any, description: str) -> Any:
    value: Any = None
    local_error: Mapping[str, Any] | None = None
    try:
        value = operation()
    except BaseException as error:
        trace = traceback.format_exc()
        local_error = {
            "rank": dist.get_rank(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
        }
    gathered_errors: list[Mapping[str, Any] | None] = [
        None
    ] * dist.get_world_size()
    dist.all_gather_object(gathered_errors, local_error)
    failures = [error for error in gathered_errors if error is not None]
    if failures:
        if any(
            set(error) != {"rank", "type", "message", "traceback_sha256"}
            for error in failures
        ) or len({int(error["rank"]) for error in failures}) != len(failures):
            raise RuntimeError(
                f"Distributed PLAT {description} returned malformed errors"
            )
        raise RuntimeError(
            f"Distributed PLAT {description} failed: "
            f"{json.dumps(failures, ensure_ascii=True, sort_keys=True)}"
        )
    return value


def _validated_source_state() -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
]:
    protocol, parent_result, split_payload = validate_protocol()
    source_audit = {
        "parent_result_sha256": sha256_file(PARENT_RESULT),
        "parent_result_receipt": parent_result["receipt"]["payload_sha256"],
        "parent_runner_sha256": parent_result["code_bindings"]["runner_sha256"],
        "feature_shards": _expected_shard_payload(),
        "code_dependencies": _expected_dependency_payload(),
        "split_payload_sha256": canonical_sha256(split_payload),
    }
    return (
        protocol,
        parent_result,
        split_payload,
        source_audit,
        canonical_sha256(source_audit),
    )


def run(*, output_dir: Path) -> Mapping[str, Any]:
    rank = _initialize_distributed()
    try:
        protocol, parent_result, split_payload, source_audit, source_digest = (
            _consensual_operation(
                _validated_source_state,
                "signed source and protocol validation",
            )
        )
        _require_consensus(source_digest, "source binding")
        output_error: list[str | None] = [None]
        if rank == 0:
            try:
                if output_dir.exists():
                    raise ValueError(f"PLAT output must be fresh: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                output_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(output_error, src=0)
        if output_error[0] is not None:
            raise ValueError(output_error[0])
        dist.barrier()

        result_payload: list[Mapping[str, Any] | None] = [None]
        analysis_error: list[str | None] = [None]
        if rank == 0:
            try:
                records = load_eligible_records(parent_result, split_payload)
                analysis = analyze(records)
                passed = bool(analysis["passed"])
                result: dict[str, Any] = {
                    "schema": SCHEMA,
                    "status": (
                        "plat_prompt_latch_crossfit_passed_mechanics_design_only"
                        if passed
                        else "plat_prompt_latch_crossfit_failed_family_retired"
                    ),
                    "passed": passed,
                    "plat_mechanics_design_authorized": passed,
                    "stage2_authorized": False,
                    "model_or_adapter_training_authorized": False,
                    "generation_authorized": False,
                    "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                    "protocol_objective": protocol["objective"],
                    "parent_binding": source_audit,
                    "nested_split": {
                        **dict(split_payload),
                        "payload_sha256": canonical_sha256(split_payload),
                    },
                    "explicitly_excluded_prior_heldout_sources": split_payload[
                        "excluded_prior_heldout_sources"
                    ],
                    "analysis": analysis,
                    "execution": {
                        "world_size": WORLD_SIZE,
                        "backend": "gloo",
                        "timeout_seconds": DISTRIBUTED_TIMEOUT_SECONDS,
                        "rank0_only_fit": True,
                        "other_ranks_verified_source_binding": True,
                        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                    },
                    "no_model_or_adapter_weights_loaded": True,
                    "no_weights_saved": True,
                    "protected_splits_opened": [],
                    "code_bindings": {
                        "runner_sha256": sha256_file(Path(__file__)),
                        "parent_runner_sha256": PARENT_RUNNER_SHA256,
                        "parent_shadow_runner_sha256": PARENT_SHADOW_RUNNER_SHA256,
                        "parent_head_runner_sha256": PARENT_HEAD_RUNNER_SHA256,
                        "parent_bilinear_helper_sha256": PARENT_BILINEAR_HELPER_SHA256,
                    },
                }
                result["receipt"] = {
                    "algorithm": "sha256",
                    "payload_scope": "canonical_result_without_receipt",
                    "payload_sha256": canonical_sha256(result),
                }
                (output_dir / "result.json").write_text(
                    json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                result_payload[0] = result
            except BaseException as error:
                analysis_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(analysis_error, src=0)
        if analysis_error[0] is not None:
            raise RuntimeError(f"PLAT rank-zero analysis failed: {analysis_error[0]}")
        dist.broadcast_object_list(result_payload, src=0)
        if result_payload[0] is None:
            raise RuntimeError("PLAT rank-zero analysis did not broadcast a result")
        _require_consensus(
            result_payload[0]["receipt"]["payload_sha256"],
            "result receipt",
        )
        dist.barrier()
        return result_payload[0]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(output_dir=args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
