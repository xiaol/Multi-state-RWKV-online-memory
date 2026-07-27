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


def correction_reference_metrics(
    correction: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    correction_rows = correction.detach().double().reshape(-1, correction.size(-1))
    reference_rows = reference.detach().double().reshape(-1, reference.size(-1))
    if correction_rows.shape != reference_rows.shape:
        raise ValueError(
            "Correction and reference tensors must have the same shape: "
            f"{tuple(correction_rows.shape)} != {tuple(reference_rows.shape)}"
        )
    if correction_rows.size(0) == 0:
        raise ValueError("Correction/reference metrics require at least one row")
    if not torch.isfinite(correction_rows).all() or not torch.isfinite(reference_rows).all():
        raise ValueError("Correction and reference tensors must be finite")
    correction_norms = correction_rows.norm(dim=-1)
    reference_norms = reference_rows.norm(dim=-1)
    ratios = correction_norms / reference_norms.clamp_min(1e-12)
    cosine = F.cosine_similarity(
        correction_rows,
        reference_rows,
        dim=-1,
        eps=1e-12,
    )
    return {
        "correction_l2_norm": float(correction_rows.reshape(-1).norm().item()),
        "reference_l2_norm": float(reference_rows.reshape(-1).norm().item()),
        "global_norm_ratio": float(
            (
                correction_rows.reshape(-1).norm()
                / reference_rows.reshape(-1).norm().clamp_min(1e-12)
            ).item()
        ),
        "token_norm_ratio_mean": float(ratios.mean().item()),
        "token_norm_ratio_median": float(ratios.median().item()),
        "token_norm_ratio_min": float(ratios.min().item()),
        "token_norm_ratio_max": float(ratios.max().item()),
        "token_cosine_mean": float(cosine.mean().item()),
        "token_cosine_min": float(cosine.min().item()),
        "token_cosine_max": float(cosine.max().item()),
    }


def causal_read_context_mask(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if labels.ndim != 2 or attention_mask.shape != labels.shape:
        raise ValueError("Read-context labels and attention mask must be matching 2D tensors")
    valid_tokens = attention_mask.ne(0)
    read_mask = labels.eq(-100) & valid_tokens
    read_mask[:, :-1] |= (
        labels[:, 1:].ne(-100)
        & valid_tokens[:, :-1]
        & valid_tokens[:, 1:]
    )
    return read_mask


def module_online_state_keys(module_name: str) -> tuple[str, str, str]:
    return (
        module_name,
        f"{module_name}.__rwkv_ms_positions",
        f"{module_name}.__rwkv_ms_previous_source",
    )


def replace_module_online_state(
    base_state: dict[str, torch.Tensor],
    replacement_state: dict[str, torch.Tensor],
    module_name: str,
) -> dict[str, torch.Tensor]:
    keys = module_online_state_keys(module_name)
    missing_base = [key for key in keys if key not in base_state]
    missing_replacement = [key for key in keys if key not in replacement_state]
    if missing_base or missing_replacement:
        raise ValueError(
            f"Cannot replace online state for {module_name}: "
            f"missing_base={missing_base} missing_replacement={missing_replacement}"
        )
    mixed_state = dict(base_state)
    for key in keys:
        mixed_state[key] = replacement_state[key]
    return mixed_state


def causal_state_swap_metrics(
    *,
    correct_ce: float,
    donor_ce: float,
    donor_with_correct_layer_ce: float,
    correct_with_donor_layer_ce: float,
) -> dict[str, float | bool]:
    donor_to_correct_gain = donor_ce - donor_with_correct_layer_ce
    correct_to_donor_damage = correct_with_donor_layer_ce - correct_ce
    return {
        "donor_with_correct_layer_ce": donor_with_correct_layer_ce,
        "correct_with_donor_layer_ce": correct_with_donor_layer_ce,
        "donor_to_correct_ce_gain": donor_to_correct_gain,
        "correct_to_donor_ce_damage": correct_to_donor_damage,
        "bidirectional_mean_ce_effect": (
            donor_to_correct_gain + correct_to_donor_damage
        )
        * 0.5,
        "bidirectional_positive": bool(
            donor_to_correct_gain > 0.0 and correct_to_donor_damage > 0.0
        ),
    }


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson correlation requires matching lists with at least two values")
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(right, dtype=torch.float64)
    left_centered = left_tensor - left_tensor.mean()
    right_centered = right_tensor - right_tensor.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator.item()) <= 1e-24:
        return None
    return float((left_centered @ right_centered / denominator).item())


def load_pairing_donors(
    checkpoint: Path,
    *,
    split_name: str,
    row_count: int,
    fallback_seed: int,
    tokenized: Dataset,
) -> tuple[list[int], dict[str, Any]]:
    manifest_path = checkpoint / "content_contrast_pairing_manifest.json"
    if not manifest_path.is_file():
        donors = make_mismatched_donors(tokenized, fallback_seed)
        return donors, {
            "source": "seeded_derangement_fallback",
            "shuffle_seed": fallback_seed,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest.get("splits", {}).get(split_name)
    if not isinstance(split, dict) or not isinstance(split.get("pairs"), list):
        raise ValueError(
            f"Pairing manifest does not contain split {split_name!r}: {manifest_path}"
        )
    pairs = split["pairs"]
    if len(pairs) != row_count:
        raise ValueError(
            f"Pairing manifest row count mismatch: expected={row_count} actual={len(pairs)}"
        )
    donors = [-1] * row_count
    for pair in pairs:
        source_index = int(pair["source_index"])
        partner_index = int(pair["partner_index"])
        if source_index < 0 or source_index >= row_count:
            raise ValueError(f"Pairing manifest source index is out of range: {source_index}")
        if partner_index < 0 or partner_index >= row_count:
            raise ValueError(f"Pairing manifest partner index is out of range: {partner_index}")
        if source_index == partner_index:
            raise ValueError(f"Pairing manifest contains a self-pair at row {source_index}")
        if donors[source_index] != -1:
            raise ValueError(f"Pairing manifest repeats source row {source_index}")
        donors[source_index] = partner_index
    if any(index < 0 for index in donors):
        raise ValueError("Pairing manifest does not cover every tokenized row")
    return donors, {
        "source": "checkpoint_pairing_manifest",
        "path": str(manifest_path),
        "file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "split_manifest_sha256": split.get("manifest_sha256"),
        "pairing_version": split.get("pairing_version"),
        "split": split_name,
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

    def __init__(
        self,
        modules: list[tuple[str, torch.nn.Module]],
        post_attention_norms: dict[str, torch.nn.Module],
    ) -> None:
        self.modules = modules
        self.post_attention_norms = post_attention_norms
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
            post_attention_norm = self.post_attention_norms[name]
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
                layernorm=post_attention_norm,
            ) -> torch.Tensor:
                record = self.records.setdefault(module_name, {})
                fusion_gate = owner._memory_fusion_gate(hidden_states, reads)
                record["fusion_gate"] = fusion_gate.detach().float().cpu()
                fused_output = bound_original(
                    base_o_output,
                    delta_o,
                    hidden_states,
                    reads,
                    token_mask,
                )
                if self.capture_fused_delta and delta_o is not None:
                    typed_delta = owner._apply_delta_o_rmsnorm(
                        delta_o.to(dtype=hidden_states.dtype)
                    )
                    fused_delta = typed_delta * fusion_gate.to(dtype=typed_delta.dtype)
                    record["base_o_output"] = base_o_output.detach().float().cpu()
                    record["fused_delta_o"] = fused_delta.detach().float().cpu()
                    record["applied_delta_o"] = (
                        fused_output.float() - base_o_output.float()
                    ).detach().cpu()
                    base_post_norm = layernorm.forward(base_o_output)
                    memory_post_norm = layernorm.forward(fused_output)
                    record["base_post_attention_norm"] = (
                        base_post_norm.detach().float().cpu()
                    )
                    record["post_attention_residual_correction"] = (
                        memory_post_norm.float() - base_post_norm.float()
                    ).detach().cpu()
                    del base_post_norm, memory_post_norm
                return fused_output

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
            if self.capture_fused_delta and "base_o_output" not in record:
                raise RuntimeError(f"Read capture for {name} is missing base_o_output")
            if self.capture_fused_delta and "applied_delta_o" not in record:
                raise RuntimeError(f"Read capture for {name} is missing applied_delta_o")
            if self.capture_fused_delta and "base_post_attention_norm" not in record:
                raise RuntimeError(
                    f"Read capture for {name} is missing base_post_attention_norm"
                )
            if (
                self.capture_fused_delta
                and "post_attention_residual_correction" not in record
            ):
                raise RuntimeError(
                    f"Read capture for {name} is missing post_attention_residual_correction"
                )
        return self.records


def replay_fixed_target(
    *,
    model,
    target_row: dict[str, Any],
    online_state: dict[str, torch.Tensor],
    device: str,
    capture: ReadPathCapture | None,
    capture_fused_delta: bool = False,
    include_token_nll: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    reset_runtime(model, write_enabled=False)
    load_delta_mem_online_state(model, online_state)
    input_ids = tensor_row(target_row, "input_ids", device)
    attention_mask = tensor_row(target_row, "attention_mask", device)
    labels = tensor_row(target_row, "labels", device)
    read_context_mask = causal_read_context_mask(labels, attention_mask)
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_read_context_mask(model, read_context_mask)
    if capture is not None:
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
    records = {} if capture is None else capture.finish()
    metrics = {
        "token_count": int(token_nll.numel()),
        "nll_sum": float(token_nll.sum().item()),
        "ce": float(token_nll.mean().item()),
    }
    if include_token_nll:
        metrics["token_nll"] = [float(value) for value in token_nll.detach().cpu().tolist()]
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
    donors, pairing_provenance = load_pairing_donors(
        checkpoint,
        split_name=str(protocol.get("dataset_split", "train")),
        row_count=len(tokenized),
        fallback_seed=args.shuffle_seed,
        tokenized=tokenized,
    )
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
    named_modules = dict(model.named_modules())
    post_attention_norms: dict[str, torch.nn.Module] = {}
    for name, _ in modules:
        parent_name = name.rsplit(".", 1)[0]
        parent = named_modules.get(parent_name)
        layernorm = getattr(parent, "post_attention_layernorm", None)
        if not isinstance(layernorm, torch.nn.Module):
            raise ValueError(f"Delta-Mem module {name} has no post_attention_layernorm")
        post_attention_norms[name] = layernorm

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
    high_dim_capture_keys = {
        "fused_delta_o",
        "base_o_output",
        "applied_delta_o",
        "base_post_attention_norm",
        "post_attention_residual_correction",
    }
    with torch.inference_mode(), ReadPathCapture(modules, post_attention_norms) as capture:
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
                        if key not in high_dim_capture_keys
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
        correct_base = causal_supervised_features(
            correct_record["base_o_output"],
            target_labels,
            target_attention_mask,
        )
        shuffled_base = causal_supervised_features(
            shuffled_record["base_o_output"],
            target_labels,
            target_attention_mask,
        )
        correct_applied = causal_supervised_features(
            correct_record["applied_delta_o"],
            target_labels,
            target_attention_mask,
        )
        shuffled_applied = causal_supervised_features(
            shuffled_record["applied_delta_o"],
            target_labels,
            target_attention_mask,
        )
        correct_post_norm_base = causal_supervised_features(
            correct_record["base_post_attention_norm"],
            target_labels,
            target_attention_mask,
        )
        shuffled_post_norm_base = causal_supervised_features(
            shuffled_record["base_post_attention_norm"],
            target_labels,
            target_attention_mask,
        )
        correct_post_norm_correction = causal_supervised_features(
            correct_record["post_attention_residual_correction"],
            target_labels,
            target_attention_mask,
        )
        shuffled_post_norm_correction = causal_supervised_features(
            shuffled_record["post_attention_residual_correction"],
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
            "correct_vs_base_attention": correction_reference_metrics(
                correct_fused,
                correct_base,
            ),
            "shuffled_vs_base_attention": correction_reference_metrics(
                shuffled_fused,
                shuffled_base,
            ),
            "correct_sha256": tensor_sha256(correct_fused),
            "shuffled_sha256": tensor_sha256(shuffled_fused),
        }
        read_layer_metrics[name]["base_o_output"] = {
            "correct_vs_shuffled": pair_metrics(correct_base, shuffled_base),
        }
        read_layer_metrics[name]["applied_delta_o"] = {
            "shape_per_condition": list(correct_applied.shape),
            "correct_vs_shuffled": pair_metrics(correct_applied, shuffled_applied),
            "correct_vs_base_attention": correction_reference_metrics(
                correct_applied,
                correct_base,
            ),
            "shuffled_vs_base_attention": correction_reference_metrics(
                shuffled_applied,
                shuffled_base,
            ),
            "correct_sha256": tensor_sha256(correct_applied),
            "shuffled_sha256": tensor_sha256(shuffled_applied),
        }
        read_layer_metrics[name]["post_attention_residual_correction"] = {
            "shape_per_condition": list(correct_post_norm_correction.shape),
            "correct_vs_shuffled": pair_metrics(
                correct_post_norm_correction,
                shuffled_post_norm_correction,
            ),
            "correct_vs_post_norm_base": correction_reference_metrics(
                correct_post_norm_correction,
                correct_post_norm_base,
            ),
            "shuffled_vs_post_norm_base": correction_reference_metrics(
                shuffled_post_norm_correction,
                shuffled_post_norm_base,
            ),
            "correct_sha256": tensor_sha256(correct_post_norm_correction),
            "shuffled_sha256": tensor_sha256(shuffled_post_norm_correction),
        }

    correct_snapshot = writer_snapshots[args.target_row_index]
    shuffled_snapshot = writer_snapshots[shuffled_row_index]
    correct_replay = fixed_target_replays[args.target_row_index]
    shuffled_replay = fixed_target_replays[shuffled_row_index]
    with torch.inference_mode():
        for module_index, (name, _) in enumerate(modules):
            donor_with_correct_layer, _ = replay_fixed_target(
                model=model,
                target_row=target_row,
                online_state=replace_module_online_state(
                    shuffled_snapshot,
                    correct_snapshot,
                    name,
                ),
                device=args.device,
                capture=None,
            )
            correct_with_donor_layer, _ = replay_fixed_target(
                model=model,
                target_row=target_row,
                online_state=replace_module_online_state(
                    correct_snapshot,
                    shuffled_snapshot,
                    name,
                ),
                device=args.device,
                capture=None,
            )
            read_layer_metrics[name]["causal_state_swap"] = causal_state_swap_metrics(
                correct_ce=float(correct_replay["ce"]),
                donor_ce=float(shuffled_replay["ce"]),
                donor_with_correct_layer_ce=float(donor_with_correct_layer["ce"]),
                correct_with_donor_layer_ce=float(correct_with_donor_layer["ce"]),
            )
            bidirectional_effect = read_layer_metrics[name]["causal_state_swap"][
                "bidirectional_mean_ce_effect"
            ]
            print(
                f"swap {module_index + 1:02d}/{len(modules)} "
                f"layer={read_layer_metrics[name]['layer_index']} "
                f"effect={bidirectional_effect:.6f}",
                flush=True,
            )

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

    correction_distances = [
        float(metrics["applied_delta_o"]["correct_vs_shuffled"]["relative_l2_mean_norm"])
        for metrics in read_layer_metrics.values()
    ]
    correction_ratios = [
        float(metrics["applied_delta_o"]["correct_vs_base_attention"]["token_norm_ratio_mean"])
        for metrics in read_layer_metrics.values()
    ]
    post_norm_distances = [
        float(
            metrics["post_attention_residual_correction"]["correct_vs_shuffled"][
                "relative_l2_mean_norm"
            ]
        )
        for metrics in read_layer_metrics.values()
    ]
    causal_effects = [
        float(metrics["causal_state_swap"]["bidirectional_mean_ce_effect"])
        for metrics in read_layer_metrics.values()
    ]
    ranked_layers = sorted(
        (
            {
                "layer_index": int(metrics["layer_index"]),
                "module_name": name,
                "bidirectional_mean_ce_effect": float(
                    metrics["causal_state_swap"]["bidirectional_mean_ce_effect"]
                ),
                "donor_to_correct_ce_gain": float(
                    metrics["causal_state_swap"]["donor_to_correct_ce_gain"]
                ),
                "correct_to_donor_ce_damage": float(
                    metrics["causal_state_swap"]["correct_to_donor_ce_damage"]
                ),
                "correction_relative_l2": float(
                    metrics["applied_delta_o"]["correct_vs_shuffled"][
                        "relative_l2_mean_norm"
                    ]
                ),
                "correction_base_ratio": float(
                    metrics["applied_delta_o"]["correct_vs_base_attention"][
                        "token_norm_ratio_mean"
                    ]
                ),
                "post_norm_correction_relative_l2": float(
                    metrics["post_attention_residual_correction"][
                        "correct_vs_shuffled"
                    ]["relative_l2_mean_norm"]
                ),
                "post_norm_correction_base_ratio": float(
                    metrics["post_attention_residual_correction"][
                        "correct_vs_post_norm_base"
                    ]["token_norm_ratio_mean"]
                ),
            }
            for name, metrics in read_layer_metrics.items()
        ),
        key=lambda item: item["bidirectional_mean_ce_effect"],
        reverse=True,
    )
    result = {
        "schema": "rwkv_ms_memory_representation_diagnostic.v2",
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
            "pairing": pairing_provenance,
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
            "fused_delta_correct_base_ratio": _aggregate_layer_metric(
                read_layer_metrics,
                (
                    "fused_delta_o",
                    "correct_vs_base_attention",
                    "token_norm_ratio_mean",
                ),
            ),
            "applied_delta_correct_vs_shuffled_relative_l2": _aggregate_layer_metric(
                read_layer_metrics,
                ("applied_delta_o", "correct_vs_shuffled", "relative_l2_mean_norm"),
            ),
            "applied_delta_correct_base_ratio": _aggregate_layer_metric(
                read_layer_metrics,
                (
                    "applied_delta_o",
                    "correct_vs_base_attention",
                    "token_norm_ratio_mean",
                ),
            ),
            "post_norm_correction_correct_vs_shuffled_relative_l2": _aggregate_layer_metric(
                read_layer_metrics,
                (
                    "post_attention_residual_correction",
                    "correct_vs_shuffled",
                    "relative_l2_mean_norm",
                ),
            ),
            "post_norm_correction_base_ratio": _aggregate_layer_metric(
                read_layer_metrics,
                (
                    "post_attention_residual_correction",
                    "correct_vs_post_norm_base",
                    "token_norm_ratio_mean",
                ),
            ),
            "causal_donor_to_correct_ce_gain": _aggregate_layer_metric(
                read_layer_metrics,
                ("causal_state_swap", "donor_to_correct_ce_gain"),
            ),
            "causal_correct_to_donor_ce_damage": _aggregate_layer_metric(
                read_layer_metrics,
                ("causal_state_swap", "correct_to_donor_ce_damage"),
            ),
            "causal_bidirectional_mean_ce_effect": _aggregate_layer_metric(
                read_layer_metrics,
                ("causal_state_swap", "bidirectional_mean_ce_effect"),
            ),
            "causal_bidirectional_positive_layers": sum(
                bool(metrics["causal_state_swap"]["bidirectional_positive"])
                for metrics in read_layer_metrics.values()
            ),
            "correction_distance_vs_causal_effect_pearson": pearson_correlation(
                correction_distances,
                causal_effects,
            ),
            "correction_base_ratio_vs_causal_effect_pearson": pearson_correlation(
                correction_ratios,
                causal_effects,
            ),
            "post_norm_distance_vs_causal_effect_pearson": pearson_correlation(
                post_norm_distances,
                causal_effects,
            ),
            "ranked_layers_by_causal_effect": ranked_layers,
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
                "Pre-add content-gated delta_o used by the representation objective at "
                "supervised causal source positions."
            ),
            "applied_delta_o": (
                "Actual bfloat16 attention-output change after adding fused_delta_o to the "
                "base attention output."
            ),
            "post_attention_residual_correction": (
                "Exact downstream change after Gemma post-attention RMSNorm, before the "
                "decoder residual addition."
            ),
            "causal_state_swap": (
                "Downstream supervised CE change when one layer's complete RWKV-MS online "
                "state is swapped between the correct and shuffled writer snapshots."
            ),
            "bidirectional_mean_ce_effect": (
                "Mean of donor CE improvement after inserting the correct layer state and "
                "correct CE damage after inserting the donor layer state; positive is desired."
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
