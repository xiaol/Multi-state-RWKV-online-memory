#!/usr/bin/env python3
"""Diagnose writer-state and reader-output specificity for episode memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deltamem.core.delta import (
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
)
from experiments.rethinking_rwkv_ms_gemma.common import (
    load_model_and_tokenizer,
    read_jsonl,
    write_json,
)
from experiments.rethinking_rwkv_ms_gemma.eval_episode_memory_ce import (
    load_protocol,
    make_mismatched_donors,
    prime_write,
    reset_runtime,
    sha256_file,
    source_identity,
    supervised_token_nll,
    tensor_row,
    validate_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenized-dataset", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-row-index", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--delta-mem-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def default_output_path(checkpoint: Path) -> Path:
    if checkpoint.parent.name != "trainer":
        raise ValueError(
            "--output is required unless checkpoint is RUN_ROOT/trainer/checkpoint-N"
        )
    return (
        checkpoint.parent.parent
        / "representation_diagnostic"
        / f"{checkpoint.name}_writer_reader_representation.json"
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError(
            f"Pair tensors must have the same flattened shape: "
            f"{tuple(left_flat.shape)} != {tuple(right_flat.shape)}"
        )
    if not torch.isfinite(left_flat).all() or not torch.isfinite(right_flat).all():
        raise ValueError("Pair tensors must be finite")
    left_norm = left_flat.norm()
    right_norm = right_flat.norm()
    distance = (left_flat - right_flat).norm()
    mean_norm = (left_norm + right_norm) * 0.5
    rms_norm = torch.sqrt((left_norm.square() + right_norm.square()) * 0.5)
    cosine = F.cosine_similarity(left_flat, right_flat, dim=0, eps=1e-12)
    return {
        "cosine": float(cosine.item()),
        "l2_distance": float(distance.item()),
        "left_l2_norm": float(left_norm.item()),
        "right_l2_norm": float(right_norm.item()),
        "relative_l2_mean_norm": float((distance / mean_norm.clamp_min(1e-12)).item()),
        "relative_l2_rms_norm": float((distance / rms_norm.clamp_min(1e-12)).item()),
    }


def _spectral_summary(matrix: torch.Tensor) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    total_energy = energy.sum()
    if singular_values.numel() == 0 or float(total_energy.item()) <= 1e-24:
        return {
            "singular_values": [float(value) for value in singular_values.tolist()],
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "top1_energy_fraction": 0.0,
        }
    probabilities = energy / total_energy
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    tolerance = (
        max(matrix.shape)
        * torch.finfo(matrix.dtype).eps
        * singular_values.max()
    )
    return {
        "singular_values": [float(value) for value in singular_values.tolist()],
        "numerical_rank": int((singular_values > tolerance).sum().item()),
        "effective_rank": float(torch.exp(entropy).item()),
        "stable_rank": float((total_energy / energy.max()).item()),
        "top1_energy_fraction": float((energy.max() / total_energy).item()),
    }


def representation_summary(rows: torch.Tensor) -> dict[str, Any]:
    matrix = rows.detach().double()
    if matrix.ndim < 2:
        raise ValueError(
            f"Expected [rows, ...] representation tensor, got {tuple(matrix.shape)}"
        )
    matrix = matrix.reshape(matrix.size(0), -1)
    if matrix.size(0) < 2:
        raise ValueError("Representation summary requires at least two rows")
    if not torch.isfinite(matrix).all():
        raise ValueError("Representation tensor must be finite")

    row_norms = matrix.norm(dim=1)
    mean_vector = matrix.mean(dim=0)
    centered = matrix - mean_vector
    centered_row_norms = centered.norm(dim=1)
    normalized = F.normalize(matrix, dim=1, eps=1e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    pairwise_distance = torch.cdist(matrix, matrix)
    pairwise_denominator = (
        row_norms[:, None] + row_norms[None, :]
    ) * 0.5
    off_diagonal = ~torch.eye(matrix.size(0), dtype=torch.bool)
    off_diagonal_cosine = cosine[off_diagonal]
    off_diagonal_relative_l2 = (
        pairwise_distance / pairwise_denominator.clamp_min(1e-12)
    )[off_diagonal]
    centered_rms = centered_row_norms.square().mean().sqrt()
    mean_norm = mean_vector.norm()
    return {
        "rows": int(matrix.size(0)),
        "features": int(matrix.size(1)),
        "row_l2_norm_mean": float(row_norms.mean().item()),
        "row_l2_norm_min": float(row_norms.min().item()),
        "row_l2_norm_max": float(row_norms.max().item()),
        "mean_vector_l2_norm": float(mean_norm.item()),
        "centered_variation_rms": float(centered_rms.item()),
        "centered_variation_to_mean_norm": float(
            (centered_rms / mean_norm.clamp_min(1e-12)).item()
        ),
        "off_diagonal_cosine_mean": float(off_diagonal_cosine.mean().item()),
        "off_diagonal_cosine_min": float(off_diagonal_cosine.min().item()),
        "off_diagonal_cosine_max": float(off_diagonal_cosine.max().item()),
        "off_diagonal_relative_l2_mean": float(off_diagonal_relative_l2.mean().item()),
        "off_diagonal_relative_l2_min": float(off_diagonal_relative_l2.min().item()),
        "off_diagonal_relative_l2_max": float(off_diagonal_relative_l2.max().item()),
        "uncentered_spectrum": _spectral_summary(matrix),
        "centered_spectrum": _spectral_summary(centered),
    }


def paired_representation_summary(
    rows: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Paired representation summary requires at least one pair")
    metrics = [pair_metrics(rows[left], rows[right]) for left, right in pairs]
    summarized: dict[str, Any] = {"pairs": len(metrics)}
    for key in metrics[0]:
        values = [item[key] for item in metrics]
        summarized[f"{key}_mean"] = statistics.fmean(values)
        summarized[f"{key}_min"] = min(values)
        summarized[f"{key}_max"] = max(values)
    return summarized


def causal_supervised_features(
    values: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    values = values.detach().cpu()
    labels = labels.detach().cpu()
    attention_mask = attention_mask.detach().cpu()
    if values.ndim != 3 or labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("Expected values [B,T,D] and labels/mask [B,T]")
    if values.shape[:2] != labels.shape or labels.shape != attention_mask.shape:
        raise ValueError(
            "Captured values and labels must align: "
            f"values={tuple(values.shape)} labels={tuple(labels.shape)} "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    supervised = labels[:, 1:].ne(-100) & attention_mask[:, 1:].ne(0)
    selected = values[:, :-1].masked_select(supervised.unsqueeze(-1))
    return selected.reshape(-1, values.size(-1)).float()


class ReadPathCapture:
    """Temporarily capture low-rank reads and the exact attention-output residual."""

    def __init__(self, modules: list[tuple[str, torch.nn.Module]]) -> None:
        self.modules = modules
        self.records: dict[str, dict[str, torch.Tensor]] = {}
        self.capture_fused_delta = False
        self._readout_originals: list[tuple[torch.nn.Module, Any]] = []
        self._fusion_originals: list[tuple[torch.nn.Module, Any]] = []

    def arm(self, *, capture_fused_delta: bool) -> None:
        self.records = {}
        self.capture_fused_delta = capture_fused_delta

    def __enter__(self) -> ReadPathCapture:
        for name, module in self.modules:
            core = getattr(module, "hrm_rwkv7_core", None)
            if core is None:
                raise RuntimeError(f"{name} does not have an RWKV-MS core")
            original_readout = core.readout
            original_fusion = module._fuse_delta_o_output
            self._readout_originals.append((core, original_readout))
            self._fusion_originals.append((module, original_fusion))

            def capture_readout(
                core_self,
                reads: torch.Tensor,
                gate: torch.Tensor,
                *,
                module_name: str = name,
                bound_original=original_readout,
            ) -> torch.Tensor:
                output = bound_original(reads, gate)
                record = self.records.setdefault(module_name, {})
                record["raw_state_read"] = reads.detach().float().cpu()
                record["post_readout"] = output.detach().float().cpu()
                return output

            def capture_fusion(
                owner,
                base_o_output: torch.Tensor,
                delta_o: torch.Tensor | None,
                hidden_states: torch.Tensor,
                reads: torch.Tensor,
                token_mask: torch.Tensor | None,
                *,
                module_name: str = name,
                bound_original=original_fusion,
            ) -> torch.Tensor:
                record = self.records.setdefault(module_name, {})
                fusion_gate = owner._memory_fusion_gate(hidden_states, reads)
                record["fusion_gate"] = fusion_gate.detach().float().cpu()
                if self.capture_fused_delta and delta_o is not None:
                    typed_delta = owner._apply_delta_o_rmsnorm(
                        delta_o.to(dtype=hidden_states.dtype)
                    )
                    fused_delta = typed_delta * fusion_gate.to(dtype=typed_delta.dtype)
                    record["fused_delta_o"] = fused_delta.detach().float().cpu()
                return bound_original(
                    base_o_output,
                    delta_o,
                    hidden_states,
                    reads,
                    token_mask,
                )

            core.readout = MethodType(capture_readout, core)
            module._fuse_delta_o_output = MethodType(capture_fusion, module)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for core, original in self._readout_originals:
            core.readout = original
        for module, original in self._fusion_originals:
            module._fuse_delta_o_output = original
        self._readout_originals.clear()
        self._fusion_originals.clear()

    def finish(self) -> dict[str, dict[str, torch.Tensor]]:
        expected = {name for name, _ in self.modules}
        missing = sorted(expected.difference(self.records))
        if missing:
            raise RuntimeError(f"Read capture missed Delta-Mem modules: {missing}")
        for name, record in self.records.items():
            required = {"raw_state_read", "post_readout", "fusion_gate"}
            absent = sorted(required.difference(record))
            if absent:
                raise RuntimeError(f"Read capture for {name} is missing: {absent}")
            if self.capture_fused_delta and "fused_delta_o" not in record:
                raise RuntimeError(f"Read capture for {name} is missing fused_delta_o")
        return self.records


def replay_fixed_target(
    *,
    model,
    target_row: dict[str, Any],
    online_state: dict[str, torch.Tensor],
    device: str,
    capture: ReadPathCapture,
    capture_fused_delta: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    reset_runtime(model, write_enabled=False)
    load_delta_mem_online_state(model, online_state)
    input_ids = tensor_row(target_row, "input_ids", device)
    attention_mask = tensor_row(target_row, "attention_mask", device)
    labels = tensor_row(target_row, "labels", device)
    read_context_mask = labels.eq(-100) & attention_mask.ne(0)
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_read_context_mask(model, read_context_mask)
    capture.arm(capture_fused_delta=capture_fused_delta)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    token_nll = supervised_token_nll(outputs.logits, labels, attention_mask)
    if not torch.isfinite(token_nll).all():
        raise RuntimeError("Fixed-target replay produced non-finite token NLL")
    records = capture.finish()
    metrics = {
        "token_count": int(token_nll.numel()),
        "nll_sum": float(token_nll.sum().item()),
        "ce": float(token_nll.mean().item()),
    }
    del outputs, token_nll
    return metrics, records


def _stack_snapshot_field(
    snapshots: list[dict[str, torch.Tensor]],
    key: str,
) -> torch.Tensor:
    return torch.stack([snapshot[key].reshape(-1).float() for snapshot in snapshots])


def _aggregate_layer_metric(
    layer_metrics: dict[str, dict[str, Any]],
    path: tuple[str, ...],
) -> dict[str, float]:
    values = []
    for metrics in layer_metrics.values():
        value: Any = metrics
        for key in path:
            value = value[key]
        values.append(float(value))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenized_path = args.tokenized_dataset.expanduser().resolve()
    source_path = args.source_jsonl.expanduser().resolve()
    output_path = (
        default_output_path(checkpoint)
        if args.output is None
        else args.output.expanduser().resolve()
    )
    raw_path = output_path.with_name(f"{output_path.stem}_raw.pt")

    protocol = load_protocol(checkpoint)
    tokenized: Dataset = load_from_disk(str(tokenized_path))
    source_rows = read_jsonl(source_path)
    ready_metadata = validate_artifacts(
        checkpoint=checkpoint,
        tokenized=tokenized,
        tokenized_path=tokenized_path,
        source_path=source_path,
        source_rows=source_rows,
        protocol=protocol,
    )
    if args.target_row_index < 0 or args.target_row_index >= len(tokenized):
        raise ValueError(
            f"--target-row-index must be in [0, {len(tokenized) - 1}], "
            f"got {args.target_row_index}"
        )
    donors = make_mismatched_donors(tokenized, args.shuffle_seed)
    shuffled_row_index = donors[args.target_row_index]

    model, _ = load_model_and_tokenizer(
        base_model=str(args.base_model.expanduser().resolve()),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=checkpoint,
    )
    model.eval()
    modules = list(iter_delta_mem_modules(model))
    if not modules:
        raise RuntimeError("Loaded model has no attached Delta-Mem modules")
    unsupported = {
        name: module.memory_fusion_placement
        for name, module in modules
        if module.memory_backend != "rwkv_ms"
        or module.memory_fusion_placement != "attention_output"
    }
    if unsupported:
        raise ValueError(
            "This diagnostic currently requires RWKV-MS attention_output fusion: "
            f"{unsupported}"
        )

    started_at = time.time()
    writer_snapshots: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for row_index in range(len(tokenized)):
            reset_runtime(model, write_enabled=True)
            prime_write(model, tokenized[row_index], args.device)
            snapshot = get_delta_mem_online_state(model)
            writer_snapshots.append(snapshot)
            print(
                f"writer {row_index + 1:02d}/{len(tokenized)} "
                f"tokens={len(tokenized[row_index]['write_input_ids'])}",
                flush=True,
            )

    expected_state_keys = {
        key
        for name, _ in modules
        for key in (
            name,
            f"{name}.__rwkv_ms_positions",
            f"{name}.__rwkv_ms_previous_source",
        )
    }
    for row_index, snapshot in enumerate(writer_snapshots):
        if set(snapshot) != expected_state_keys:
            raise RuntimeError(
                f"Writer snapshot {row_index} has unexpected keys: "
                f"missing={sorted(expected_state_keys.difference(snapshot))} "
                f"extra={sorted(set(snapshot).difference(expected_state_keys))}"
            )

    paired_indices = list(enumerate(donors))
    writer_layer_metrics: dict[str, dict[str, Any]] = {}
    for name, _ in modules:
        state_rows = _stack_snapshot_field(writer_snapshots, name)
        previous_source_key = f"{name}.__rwkv_ms_previous_source"
        previous_source_rows = _stack_snapshot_field(
            writer_snapshots,
            previous_source_key,
        )
        position_key = f"{name}.__rwkv_ms_positions"
        positions = _stack_snapshot_field(writer_snapshots, position_key).squeeze(-1)
        writer_layer_metrics[name] = {
            "layer_index": int(dict(modules)[name].layer_idx),
            "delta_state_shape_per_row": list(writer_snapshots[0][name].shape),
            "delta_state": {
                "representation": representation_summary(state_rows),
                "shuffled_pairs": paired_representation_summary(
                    state_rows,
                    paired_indices,
                ),
                "fixed_target_correct_vs_shuffled": pair_metrics(
                    state_rows[args.target_row_index],
                    state_rows[shuffled_row_index],
                ),
            },
            "previous_source": {
                "representation": representation_summary(previous_source_rows),
                "shuffled_pairs": paired_representation_summary(
                    previous_source_rows,
                    paired_indices,
                ),
                "fixed_target_correct_vs_shuffled": pair_metrics(
                    previous_source_rows[args.target_row_index],
                    previous_source_rows[shuffled_row_index],
                ),
            },
            "positions": {
                "values": [int(value) for value in positions.tolist()],
                "all_equal": bool(torch.equal(positions, positions[0].expand_as(positions))),
            },
        }

    target_row = tokenized[args.target_row_index]
    target_labels = tensor_row(target_row, "labels", args.device)
    target_attention_mask = tensor_row(target_row, "attention_mask", args.device)
    fixed_target_replays: list[dict[str, Any]] = []
    full_records: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    low_dim_raw_records: list[dict[str, dict[str, torch.Tensor]]] = []
    with torch.inference_mode(), ReadPathCapture(modules) as capture:
        for memory_row_index, snapshot in enumerate(writer_snapshots):
            capture_full = memory_row_index in {
                args.target_row_index,
                shuffled_row_index,
            }
            replay_metrics, records = replay_fixed_target(
                model=model,
                target_row=target_row,
                online_state=snapshot,
                device=args.device,
                capture=capture,
                capture_fused_delta=capture_full,
            )
            for name, module in modules:
                routes = module.last_read_routes
                if routes is None:
                    raise RuntimeError(f"Read replay did not produce routes for {name}")
                records[name]["read_routes"] = routes.detach().float().cpu()
            low_dim_raw_records.append(
                {
                    name: {
                        key: value
                        for key, value in records[name].items()
                        if key != "fused_delta_o"
                    }
                    for name, _ in modules
                }
            )
            if capture_full:
                full_records[memory_row_index] = records
            fixed_target_replays.append(
                {
                    "memory_row_index": memory_row_index,
                    "memory_identity": source_identity(
                        source_rows[memory_row_index],
                        memory_row_index,
                    ),
                    **replay_metrics,
                }
            )
            print(
                f"read {memory_row_index + 1:02d}/{len(writer_snapshots)} "
                f"fixed_target={args.target_row_index} ce={replay_metrics['ce']:.6f}",
                flush=True,
            )
    reset_runtime(model, write_enabled=True)

    read_layer_metrics: dict[str, dict[str, Any]] = {}
    for name, _ in modules:
        stage_rows: dict[str, torch.Tensor] = {}
        for stage in ("raw_state_read", "post_readout", "fusion_gate", "read_routes"):
            stage_rows[stage] = torch.stack(
                [
                    causal_supervised_features(
                        records[name][stage],
                        target_labels,
                        target_attention_mask,
                    ).reshape(-1)
                    for records in low_dim_raw_records
                ]
            )
        correct_record = full_records[args.target_row_index][name]
        shuffled_record = full_records[shuffled_row_index][name]
        correct_fused = causal_supervised_features(
            correct_record["fused_delta_o"],
            target_labels,
            target_attention_mask,
        )
        shuffled_fused = causal_supervised_features(
            shuffled_record["fused_delta_o"],
            target_labels,
            target_attention_mask,
        )
        read_layer_metrics[name] = {
            "layer_index": int(dict(modules)[name].layer_idx),
        }
        read_layer_metrics[name].update(
            {
                stage: {
                    "representation_across_writer_states": representation_summary(rows),
                    "correct_vs_shuffled": pair_metrics(
                        rows[args.target_row_index],
                        rows[shuffled_row_index],
                    ),
                }
                for stage, rows in stage_rows.items()
            }
        )
        read_layer_metrics[name]["fused_delta_o"] = {
            "shape_per_condition": list(correct_fused.shape),
            "correct_vs_shuffled": pair_metrics(correct_fused, shuffled_fused),
            "correct_sha256": tensor_sha256(correct_fused),
            "shuffled_sha256": tensor_sha256(shuffled_fused),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "schema": "rwkv_ms_memory_representation_raw.v1",
        "checkpoint": str(checkpoint),
        "target_row_index": args.target_row_index,
        "shuffled_row_index": shuffled_row_index,
        "writer_snapshots": writer_snapshots,
        "fixed_target_low_dim_replays": low_dim_raw_records,
    }
    torch.save(raw_payload, raw_path)

    correct_replay = fixed_target_replays[args.target_row_index]
    shuffled_replay = fixed_target_replays[shuffled_row_index]
    result = {
        "schema": "rwkv_ms_memory_representation_diagnostic.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started_at,
        "provenance": {
            "base_model": str(args.base_model.expanduser().resolve()),
            "checkpoint": str(checkpoint),
            "tokenized_dataset": str(tokenized_path),
            "tokenized_fingerprint": getattr(tokenized, "_fingerprint", None),
            "tokenized_ready_metadata": ready_metadata,
            "source_jsonl": str(source_path),
            "source_jsonl_sha256": sha256_file(source_path),
            "training_protocol": protocol,
            "shuffle_seed": args.shuffle_seed,
            "shuffled_donor_indices": donors,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "torch_version": torch.__version__,
            "model_load_count": 1,
            "raw_artifact": str(raw_path),
            "raw_artifact_sha256": sha256_file(raw_path),
        },
        "fixed_target": {
            "target": source_identity(
                source_rows[args.target_row_index],
                args.target_row_index,
            ),
            "shuffled_memory": source_identity(
                source_rows[shuffled_row_index],
                shuffled_row_index,
            ),
            "write_tokens_all_rows": [
                len(tokenized[index]["write_input_ids"])
                for index in range(len(tokenized))
            ],
            "read_tokens": len(target_row["input_ids"]),
            "supervised_tokens": int(correct_replay["token_count"]),
            "correct_memory_ce": float(correct_replay["ce"]),
            "shuffled_memory_ce": float(shuffled_replay["ce"]),
            "shuffled_minus_correct_ce": float(
                shuffled_replay["ce"] - correct_replay["ce"]
            ),
            "all_memory_replays": fixed_target_replays,
        },
        "writer_layer_metrics": writer_layer_metrics,
        "read_layer_metrics": read_layer_metrics,
        "cross_layer_summary": {
            "writer_delta_state_off_diagonal_cosine": _aggregate_layer_metric(
                writer_layer_metrics,
                ("delta_state", "representation", "off_diagonal_cosine_mean"),
            ),
            "writer_delta_state_relative_l2": _aggregate_layer_metric(
                writer_layer_metrics,
                ("delta_state", "representation", "off_diagonal_relative_l2_mean"),
            ),
            "writer_delta_state_content_to_mean": _aggregate_layer_metric(
                writer_layer_metrics,
                ("delta_state", "representation", "centered_variation_to_mean_norm"),
            ),
            "writer_delta_state_centered_effective_rank": _aggregate_layer_metric(
                writer_layer_metrics,
                ("delta_state", "representation", "centered_spectrum", "effective_rank"),
            ),
            "raw_state_read_correct_vs_shuffled_cosine": _aggregate_layer_metric(
                read_layer_metrics,
                ("raw_state_read", "correct_vs_shuffled", "cosine"),
            ),
            "post_readout_correct_vs_shuffled_cosine": _aggregate_layer_metric(
                read_layer_metrics,
                ("post_readout", "correct_vs_shuffled", "cosine"),
            ),
            "fused_delta_correct_vs_shuffled_cosine": _aggregate_layer_metric(
                read_layer_metrics,
                ("fused_delta_o", "correct_vs_shuffled", "cosine"),
            ),
            "fused_delta_correct_vs_shuffled_relative_l2": _aggregate_layer_metric(
                read_layer_metrics,
                ("fused_delta_o", "correct_vs_shuffled", "relative_l2_mean_norm"),
            ),
            "post_readout_centered_effective_rank": _aggregate_layer_metric(
                read_layer_metrics,
                (
                    "post_readout",
                    "representation_across_writer_states",
                    "centered_spectrum",
                    "effective_rank",
                ),
            ),
        },
        "metric_notes": {
            "relative_l2_mean_norm": "||a-b|| / ((||a||+||b||)/2)",
            "centered_effective_rank": (
                "exp(entropy) of normalized squared singular values after subtracting "
                "the across-row mean"
            ),
            "raw_state_read": "Slot-routed state lookup before RWKV readout GroupNorm/g/output.",
            "post_readout": "Four-dimensional RWKV read after GroupNorm, g, and output projection.",
            "fused_delta_o": (
                "Exact content-gated delta_o injected by attention_output fusion at supervised "
                "causal source positions."
            ),
        },
    }
    write_json(output_path, result)
    print(json.dumps(result["fixed_target"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(result["cross_layer_summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {raw_path}", flush=True)


if __name__ == "__main__":
    main()
