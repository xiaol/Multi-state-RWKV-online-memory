#!/usr/bin/env python3
"""Strict open-row cross-fit screen for learned RWKV state identity.

Exactly four A100 ranks capture frozen answer-position projected-value/RWKV
features.  Rank zero then trains only tiny per-layer bilinear compatibility
heads on a deterministic donor-component-disjoint 176/44 split.  The runner
never changes model output, saves no adapter, opens no protected split, and
never runs generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_projected_value_identity as value_identity,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_query_state_bilinear as bilinear,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as endpoint,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as candidate,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_screen as hardware,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_query_state_infonce_screen as geometry,
)


SCHEMA = "rwkv_ms_natural_memory_native_query_state_bilinear_crossfit.v1"
FEATURE_SCHEMA = "rwkv_ms_natural_memory_native_query_state_features.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_query_state_bilinear_crossfit_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "3e0c450275ed2d979d599bf29169ab89528ca673c2df5b44ffa8c01443edaddf"
WORLD_SIZE = 4
SEED = 114
SPLIT_SALT = "rwkv-query-state-bilinear-component-crossfit-v1:"
TRAIN_ROWS = 176
HELDOUT_ROWS = 44
STATE_DIM = 32
LAYERS = 42
BOTTLENECK = 4
TRAIN_STEPS = 512
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
IDENTITY_MARGIN = 0.2
DONOR_FRACTION_GATE = 0.95
MEAN_GAP_GATE = 0.05
PERMUTED_FRACTION_GATE = 0.95
GEOMETRY_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_query_state_infonce_screen_v4/result.json"
GEOMETRY_RESULT_SHA256 = "3df2df65e0e5144046eda60f23bde5693a27c0d5a43995f526ce9b59505654c9"
PREFLIGHT_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_query_state_infonce_causal_preflight_v2/result.json"
PREFLIGHT_RESULT_SHA256 = "47407af50042765c2f71557c19b3a7f9df9af118fcc11bc053981725c39f9139"

distributed = candidate.shared.distributed
evolution = candidate.shared.evolution
causal_train = candidate.causal_train


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    digest = canonical_sha256(unsigned)
    training = protocol.get("offline_training", {})
    gates = protocol.get("heldout_gates", {})
    split = protocol.get("crossfit_split", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or protocol.get("generation_authorized") is not False
        or protocol.get("adapter_saved") is not False
        or split.get("train_rows") != TRAIN_ROWS
        or split.get("heldout_rows") != HELDOUT_ROWS
        or split.get("selection_salt") != SPLIT_SALT
        or training.get("steps") != TRAIN_STEPS
        or training.get("bottleneck") != BOTTLENECK
        or gates.get("donor_pairwise_positive_fraction_minimum") != DONOR_FRACTION_GATE
        or gates.get("donor_mean_gap_minimum") != MEAN_GAP_GATE
        or gates.get("layer_permuted_pairwise_positive_fraction_minimum")
        != PERMUTED_FRACTION_GATE
    ):
        raise ValueError("Bilinear cross-fit protocol differs")
    for path, expected in (
        (GEOMETRY_RESULT, GEOMETRY_RESULT_SHA256),
        (PREFLIGHT_RESULT, PREFLIGHT_RESULT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Authorization result differs: {path}")
    return protocol


def donor_components(mapping: Mapping[int, int]) -> tuple[tuple[int, ...], ...]:
    nodes = set(mapping)
    if set(mapping.values()) - nodes:
        raise ValueError("Donor mapping leaves the authorized row set")
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source, donor in mapping.items():
        adjacency[int(source)].add(int(donor))
        adjacency[int(donor)].add(int(source))
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    if seen != nodes:
        raise ValueError("Donor component coverage differs")
    return tuple(components)


def crossfit_split(mapping: Mapping[int, int]) -> tuple[dict[int, str], Mapping[str, Any]]:
    components = donor_components(mapping)
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
        raise RuntimeError("Cannot construct exact heldout component partition")
    heldout_components = set(paths[HELDOUT_ROWS])
    heldout = {
        source
        for index, component in enumerate(ordered)
        if index in heldout_components
        for source in component
    }
    split = {
        source: ("heldout" if source in heldout else "train")
        for source in sorted(mapping)
    }
    train = {source for source, name in split.items() if name == "train"}
    if len(train) != TRAIN_ROWS or len(heldout) != HELDOUT_ROWS:
        raise RuntimeError("Cross-fit split sizes differ")
    if any(split[source] != split[donor] for source, donor in mapping.items()):
        raise RuntimeError("A donor edge crosses the cross-fit partition")
    payload = {
        "selection_salt": SPLIT_SALT,
        "component_count": len(ordered),
        "component_sizes": [len(component) for component in ordered],
        "heldout_component_indices": sorted(heldout_components),
        "train_sources": sorted(train),
        "heldout_sources": sorted(heldout),
    }
    return split, payload


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
        source: evolution.encode_native_full_row(
            tokenizer,
            task="scene",
            source_ordinal=source,
            raw_line=bytes(record["raw_line"]).decode("utf-8"),
        )
        for source, record in selected.items()
    }
    rows_by_source = {int(row["source_index"]): row for row in rows}
    if set(examples) != set(mapping) or len(examples) != endpoint.EVALUATION_ROWS:
        raise RuntimeError("Authorized example coverage differs")
    split_payload = {
        **dict(split_payload),
        "rows": [
            {
                "source_index": source,
                "row_sha256": rows_by_source[source]["row_sha256"],
                "donor_source_index": mapping[source],
                "donor_row_sha256": rows_by_source[mapping[source]]["row_sha256"],
                "split": split[source],
            }
            for source in sorted(mapping)
        ],
    }
    return examples, rows_by_source, mapping, split_payload


def aggregate_vectors(captured: Sequence[Any], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return geometry._aggregate_answer_vectors(captured, labels)


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
    donor = candidate.shared.contrast.build_donor_batch(
        target, donor_example, device=device
    )
    try:
        with torch.no_grad():
            value_identity.clear(model)
            evolution._native_write(model, target, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            target_values = value_identity.capture_write_values(model)
            value_identity.set_fixed_target_values(model, target_values)
            correct_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            query, correct = aggregate_vectors(value_identity.capture(model), target.labels)
            del correct_logits

            value_identity.clear(model)
            evolution._native_write(model, donor, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            donor_carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=donor_state,
                rotate_recurrent_layers=False,
            )
            value_identity.set_fixed_target_values(model, target_values)
            donor_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            donor_query, donor_read = aggregate_vectors(
                value_identity.capture(model), target.labels
            )
            del donor_logits

            value_identity.clear(model)
            permuted_carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=correct_state,
                rotate_recurrent_layers=True,
            )
            value_identity.set_fixed_target_values(model, target_values)
            permuted_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            permuted_query, permuted = aggregate_vectors(
                value_identity.capture(model), target.labels
            )
            del permuted_logits
        if not (
            donor_carrier_fixed
            and permuted_carrier_fixed
            and torch.equal(query, donor_query)
            and torch.equal(query, permuted_query)
        ):
            raise RuntimeError("Projected query/carrier changed across interventions")
        tensors = (query, correct, donor_read, permuted)
        if any(tuple(tensor.shape) != (LAYERS, STATE_DIM) for tensor in tensors):
            raise RuntimeError("Captured query/state feature shape differs")
        if not bool(torch.isfinite(torch.stack(tensors)).all()):
            raise RuntimeError("Captured query/state feature is non-finite")
        return {
            "query": query.cpu().tolist(),
            "correct": correct.cpu().tolist(),
            "matched_donor": donor_read.cpu().tolist(),
            "layer_permuted": permuted.cpu().tolist(),
            "projected_carrier_fixed": True,
        }
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


class LayerwiseBilinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            bilinear.ResidualBilinearIdentity(STATE_DIM, bottleneck=BOTTLENECK)
            for _ in range(LAYERS)
        )

    def score(self, query: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if query.shape != state.shape or tuple(query.shape[1:]) != (LAYERS, STATE_DIM):
            raise ValueError("Layerwise bilinear feature shapes differ")
        return torch.stack(
            [
                head.score(query[:, layer], state[:, layer])
                for layer, head in enumerate(self.heads)
            ],
            dim=1,
        )


def score_metrics(
    head: LayerwiseBilinear,
    query: torch.Tensor,
    correct: torch.Tensor,
    negative: torch.Tensor,
) -> Mapping[str, Any]:
    with torch.no_grad():
        gap = (head.score(query, correct) - head.score(query, negative)).mean(dim=1)
    return {
        "rows": int(gap.numel()),
        "mean_gap": float(gap.mean().item()),
        "median_gap": float(gap.median().item()),
        "minimum_gap": float(gap.min().item()),
        "maximum_gap": float(gap.max().item()),
        "pairwise_positive_fraction": float(gap.gt(0.0).float().mean().item()),
        "finite": bool(torch.isfinite(gap).all()),
    }


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
        raise RuntimeError("Analyzer cross-fit row counts differ")
    torch.manual_seed(SEED)
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
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    losses: list[float] = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        correct_score = head.score(train["query"], train["correct"])
        donor_score = head.score(train["query"], train["matched_donor"])
        permuted_score = head.score(train["query"], train["layer_permuted"])
        donor_hinge = F.relu(IDENTITY_MARGIN - correct_score + donor_score)
        permuted_hinge = F.relu(IDENTITY_MARGIN - correct_score + permuted_score)
        loss = donor_hinge.mean() + permuted_hinge.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Bilinear cross-fit loss is non-finite")
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


def load_feature_records(output_dir: Path, split_payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    split_by_source = {
        int(row["source_index"]): str(row["split"])
        for row in split_payload["rows"]
    }
    records: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    for shard_index in range(WORLD_SIZE):
        path = output_dir / f"shard-{shard_index}.jsonl"
        shard_rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records.extend(shard_rows)
        provenance.append(
            {"path": str(path), "rows": len(shard_rows), "sha256": sha256_file(path)}
        )
    sources = [int(row["source_index"]) for row in records]
    if len(records) != endpoint.EVALUATION_ROWS or len(set(sources)) != len(sources):
        raise RuntimeError("Feature shard coverage differs")
    for row in records:
        source = int(row["source_index"])
        if (
            row.get("schema") != FEATURE_SCHEMA
            or row.get("split") != split_by_source[source]
            or row.get("projected_carrier_fixed") is not True
        ):
            raise RuntimeError("Feature row contract differs")
    return records, provenance


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(
    *,
    base_model: Path,
    dataset_root: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training("cuda")
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol = validate_protocol()
        if context.world_size != WORLD_SIZE or not hardware.four_distinct_a100s(
            context.rank_devices
        ):
            raise RuntimeError("Cross-fit capture requires exactly four distinct A100s")
        if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
            raise RuntimeError("HF_ENDPOINT must be exactly https://hf-mirror.com")
        fresh_error: BaseException | None = None
        if context.is_primary and output_dir.exists():
            fresh_error = ValueError(f"Cross-fit output must be fresh: {output_dir}")
        distributed.phase_consensus(context, phase="bilinear-crossfit-fresh", error=fresh_error)
        create_error: BaseException | None = None
        if context.is_primary:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                create_error = error
        distributed.phase_consensus(context, phase="bilinear-crossfit-create", error=create_error)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, model_audit = candidate.load_model(
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
            source for source in sorted(examples)
            if source % WORLD_SIZE == context.process_rank
        ]
        for ordinal, source in enumerate(shard_sources, start=1):
            donor = mapping[source]
            feature = capture_row(
                model,
                examples[source],
                examples[donor],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            append_jsonl(
                shard_path,
                {
                    "schema": FEATURE_SCHEMA,
                    "source_index": source,
                    "row_sha256": rows[source]["row_sha256"],
                    "donor_source_index": donor,
                    "donor_row_sha256": rows[donor]["row_sha256"],
                    "split": split[source],
                    **feature,
                },
            )
            print(
                f"BILINEAR_CROSSFIT rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/{len(shard_sources)}",
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
                    "bilinear_crossfit_passed_causal_training_design_authorized"
                    if passed
                    else "bilinear_crossfit_failed_donor_identity_branch_blocked"
                ),
                "passed": passed,
                "causal_training_design_authorized": passed,
                "generation_authorized": False,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "protocol_objective": protocol["objective"],
                "seed": SEED,
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
                "model_audit": {
                    "wrapped_layers": model_audit.get("wrapped_layers"),
                    "learned_write": model_audit.get("learned_write"),
                    "capture": capture_audit,
                    "output_changed_by_identity_head": False,
                },
                "authorization_results": {
                    "geometry_result_sha256": GEOMETRY_RESULT_SHA256,
                    "preflight_result_sha256": PREFLIGHT_RESULT_SHA256,
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
    result = run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0 if not result or result.get("passed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
