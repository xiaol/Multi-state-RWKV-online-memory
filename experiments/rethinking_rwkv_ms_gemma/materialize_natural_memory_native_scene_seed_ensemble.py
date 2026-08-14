#!/usr/bin/env python3
"""Materialize the locked mean of three robust native-scene seed deltas."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_ensemble_materialization.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_ensemble_patch.v1"
CANDIDATE_ID = "seed_delta_mean"


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def gate_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    selected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
        and any(name.endswith(f".{family}") for family in training.contrast.GATE_FAMILIES)
    }
    if len(selected) != 126 or sum(tensor.numel() for tensor in selected.values()) != 108906:
        raise ValueError("Seed-ensemble gate state differs")
    return selected


def load_v9(memory_dir: Path) -> dict[str, torch.Tensor]:
    files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(files) != training.contrast.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Seed-ensemble V9 adapter differs")
    adapter_path = memory_dir / "delta_mem_adapter.pt"
    if sha256_file(adapter_path) != training.contrast.V9_ADAPTER_WEIGHTS_SHA256:
        raise ValueError("Seed-ensemble V9 adapter weights differ")
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Seed-ensemble V9 payload differs")
    return gate_state(payload)


def load_seed_root(root: Path, *, seed: int) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    result = probe.validate_signed_json(
        root / "result.json",
        description=f"Seed-ensemble training result {seed}",
    )
    training_block = result.get("training")
    if (
        result.get("schema") != training.SCHEMA
        or result.get("status") != "training_complete_evaluation_pending"
        or result.get("protected_splits_opened") != []
        or not isinstance(training_block, Mapping)
        or training_block.get("seed") != seed
        or training_block.get("updates") != training.TRAIN_UPDATES
        or training_block.get("rows") != 256
        or training_block.get("non_gate_unchanged") is not True
    ):
        raise ValueError(f"Seed-ensemble training result differs: {seed}")
    manifest = probe.validate_signed_json(
        root / f"checkpoint-{training.TRAIN_UPDATES}" / "manifest.json",
        description=f"Seed-ensemble checkpoint manifest {seed}",
    )
    if (
        manifest != training_block.get("checkpoint")
        or manifest.get("schema") != training.PATCH_SCHEMA
        or manifest.get("protocol_payload_sha256") != training.PROTOCOL_PAYLOAD_SHA256
        or manifest.get("seed") != seed
        or manifest.get("step") != training.TRAIN_UPDATES
        or manifest.get("parameter_tensors") != 126
        or manifest.get("parameter_elements") != 108906
    ):
        raise ValueError(f"Seed-ensemble checkpoint manifest differs: {seed}")
    patch_path = root / f"checkpoint-{training.TRAIN_UPDATES}" / "gate_patch.pt"
    patch_file = manifest.get("patch_file")
    if (
        not isinstance(patch_file, Mapping)
        or patch_file.get("bytes") != patch_path.stat().st_size
        or patch_file.get("sha256") != sha256_file(patch_path)
    ):
        raise ValueError(f"Seed-ensemble checkpoint file differs: {seed}")
    payload = torch.load(patch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError(f"Seed-ensemble checkpoint payload differs: {seed}")
    required = {
        "schema": training.PATCH_SCHEMA,
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "source_adapter_files_sha256": training.contrast.V9_ADAPTER_FILES_SHA256,
        "seed": seed,
        "step": training.TRAIN_UPDATES,
        "gate_state_sha256": manifest["gate_state_sha256"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError(f"Seed-ensemble checkpoint metadata differs: {seed}")
    state = gate_state(payload["state_dict"])
    if runtime._state_dict_sha256(state) != manifest["gate_state_sha256"]:
        raise ValueError(f"Seed-ensemble checkpoint state differs: {seed}")
    return state, {
        "seed": seed,
        "training_root": str(root),
        "training_result_sha256": sha256_file(root / "result.json"),
        "training_result_receipt_sha256": result["receipt"]["payload_sha256"],
        "gate_state_sha256": manifest["gate_state_sha256"],
        "manifest_sha256": sha256_file(
            root / f"checkpoint-{training.TRAIN_UPDATES}" / "manifest.json"
        ),
        "patch_sha256": patch_file["sha256"],
    }


def mean_seed_delta(
    v9: Mapping[str, torch.Tensor],
    seeds: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if tuple(sorted(seeds)) != training.SEEDS:
        raise ValueError("Seed-ensemble source seed set differs")
    names = set(v9)
    if any(set(state) != names for state in seeds.values()):
        raise ValueError("Seed-ensemble source parameter names differ")
    mixed: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        anchor = v9[name]
        delta_sum = torch.zeros_like(anchor, dtype=torch.float64)
        for seed in training.SEEDS:
            source = seeds[seed][name]
            if source.shape != anchor.shape:
                raise ValueError(f"Seed-ensemble source shape differs: {name}")
            delta_sum.add_(source.to(torch.float64) - anchor.to(torch.float64))
        mixed[name] = (
            anchor.to(torch.float64) + delta_sum / len(training.SEEDS)
        ).to(anchor.dtype)
    return mixed


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Seed-ensemble output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize(
    *,
    memory_dir: Path,
    training_roots: Mapping[int, Path],
    output_root: Path,
) -> Mapping[str, Any]:
    training.validate_protocol()
    if output_root.exists():
        raise ValueError(f"Seed-ensemble output must be fresh: {output_root}")
    v9 = load_v9(memory_dir)
    states: dict[int, dict[str, torch.Tensor]] = {}
    sources: list[Mapping[str, Any]] = []
    for seed in training.SEEDS:
        state, source = load_seed_root(training_roots[seed], seed=seed)
        states[seed] = state
        sources.append(source)
    candidate = mean_seed_delta(v9, states)
    candidate_sha256 = runtime._state_dict_sha256(candidate)
    source_hashes = {
        "v9": runtime._state_dict_sha256(v9),
        **{str(seed): runtime._state_dict_sha256(states[seed]) for seed in training.SEEDS},
    }
    output_root.mkdir(parents=True, exist_ok=False)
    patch_path = output_root / "gate_patch.pt"
    torch.save(
        {
            "schema": PATCH_SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "candidate_id": CANDIDATE_ID,
            "source_gate_state_sha256": source_hashes,
            "seed_weights": {str(seed): 1 for seed in training.SEEDS},
            "denominator": len(training.SEEDS),
            "gate_state_sha256": candidate_sha256,
            "state_dict": candidate,
        },
        patch_path,
    )
    manifest: dict[str, Any] = {
        "schema": PATCH_SCHEMA,
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "candidate_id": CANDIDATE_ID,
        "source_gate_state_sha256": source_hashes,
        "seed_weights": {str(seed): 1 for seed in training.SEEDS},
        "denominator": len(training.SEEDS),
        "gate_state_sha256": candidate_sha256,
        "parameter_tensors": len(candidate),
        "parameter_elements": sum(tensor.numel() for tensor in candidate.values()),
        "patch_file": {
            "path": str(patch_path),
            "bytes": patch_path.stat().st_size,
            "sha256": sha256_file(patch_path),
        },
    }
    manifest["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_manifest_without_receipt",
        "payload_sha256": canonical_sha256(manifest),
    }
    write_json(output_root / "manifest.json", manifest)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "candidate_id": CANDIDATE_ID,
        "source_gate_state_sha256": source_hashes,
        "source_training_artifacts": sources,
        "candidate_manifest": manifest,
        "candidate_fixed_before_generation": True,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_materialization_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    write_json(output_root / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", type=Path, required=True)
    for seed in training.SEEDS:
        parser.add_argument(f"--seed-{seed}-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = {
        seed: getattr(args, f"seed_{seed}_root").expanduser().resolve(strict=True)
        for seed in training.SEEDS
    }
    result = materialize(
        memory_dir=args.memory_dir.expanduser().resolve(strict=True),
        training_roots=roots,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "candidate_id": result["candidate_id"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
