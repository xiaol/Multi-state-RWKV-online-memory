#!/usr/bin/env python3
"""Materialize hash-bound convex gate soups around contrast checkpoint 16."""

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
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_checkpoint_soup_materialization.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_checkpoint_soup_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_checkpoint_soup_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "775ee366ce65fd7530011c0ef16b6104c8fd50661a2f4c6cd1da685840d09466"
RECIPE_PAYLOAD_SHA256 = "62f988374feec0ae51cbe53a967d102463c0a804e837cb3d0b1295316df000e0"
TRAINING_RESULT_FILE_SHA256 = "471bfb011c0b62833814af506acfa22f1093d34637c74cdc1d21954d70d5f6cc"
TRAINING_RESULT_RECEIPT_SHA256 = "ef72604d84a379413e8c518b4d41ce4a844eaa536df05d30163ac6bf51f5c9cc"
SOURCE_ORDER = ("v9", "checkpoint_8", "checkpoint_16", "checkpoint_32")


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Checkpoint-soup protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Checkpoint-soup protocol hash differs")
    recipes = value.get("candidate_materialization", {}).get("candidates")
    if not isinstance(recipes, list) or canonical_sha256(recipes) != RECIPE_PAYLOAD_SHA256:
        raise ValueError("Checkpoint-soup recipe payload differs")
    return value


def gate_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    selected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
        and any(name.endswith(f".{family}") for family in training.GATE_FAMILIES)
    }
    if len(selected) != 126 or sum(tensor.numel() for tensor in selected.values()) != 108906:
        raise ValueError("Checkpoint-soup gate state differs")
    return selected


def load_sources(
    *,
    memory_dir: Path,
    training_root: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[Mapping[str, Any]]]:
    adapter_files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(adapter_files) != training.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Checkpoint-soup V9 adapter differs")
    adapter_path = memory_dir / "delta_mem_adapter.pt"
    if sha256_file(adapter_path) != training.V9_ADAPTER_WEIGHTS_SHA256:
        raise ValueError("Checkpoint-soup V9 adapter weights differ")
    raw_v9 = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_v9, Mapping):
        raise ValueError("Checkpoint-soup V9 payload differs")
    sources = {"v9": gate_state(raw_v9)}
    manifests = probe.validate_training_root(training_root)
    if sha256_file(training_root / "result.json") != TRAINING_RESULT_FILE_SHA256:
        raise ValueError("Checkpoint-soup training result file differs")
    result = probe.validate_signed_json(
        training_root / "result.json",
        description="Checkpoint-soup source training result",
    )
    if result["receipt"].get("payload_sha256") != TRAINING_RESULT_RECEIPT_SHA256:
        raise ValueError("Checkpoint-soup training result receipt differs")
    for manifest in manifests:
        step = int(manifest["step"])
        payload = torch.load(
            training_root / f"checkpoint-{step}" / "gate_patch.pt",
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise ValueError(f"Checkpoint-soup source patch differs: {step}")
        state = gate_state(payload["state_dict"])
        if runtime._state_dict_sha256(state) != manifest["gate_state_sha256"]:
            raise ValueError(f"Checkpoint-soup source state hash differs: {step}")
        sources[f"checkpoint_{step}"] = state
    if tuple(sorted(sources)) != tuple(sorted(SOURCE_ORDER)):
        raise ValueError("Checkpoint-soup source set differs")
    names = set(sources["v9"])
    if any(set(state) != names for state in sources.values()):
        raise ValueError("Checkpoint-soup source parameter names differ")
    return sources, manifests


def mix_state(
    sources: Mapping[str, Mapping[str, torch.Tensor]],
    recipe: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    weights = recipe.get("weights")
    denominator = recipe.get("denominator")
    if not isinstance(weights, Mapping) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("Checkpoint-soup recipe structure differs")
    integer_weights = {name: weights.get(name) for name in SOURCE_ORDER}
    if (
        any(not isinstance(weight, int) or weight < 0 for weight in integer_weights.values())
        or sum(integer_weights.values()) != denominator
        or set(weights) != set(SOURCE_ORDER)
    ):
        raise ValueError("Checkpoint-soup recipe is not convex")
    mixed: dict[str, torch.Tensor] = {}
    for name in sorted(sources["v9"]):
        reference = sources["v9"][name]
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for source_name in SOURCE_ORDER:
            source = sources[source_name][name]
            if source.shape != reference.shape:
                raise ValueError(f"Checkpoint-soup source shape differs: {name}")
            accumulator.add_(source.to(torch.float64), alpha=integer_weights[source_name])
        mixed[name] = (accumulator / denominator).to(reference.dtype)
    return mixed


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Checkpoint-soup output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize(
    *,
    memory_dir: Path,
    training_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    protocol = validate_protocol()
    recipes = protocol["candidate_materialization"]["candidates"]
    sources, manifests = load_sources(memory_dir=memory_dir, training_root=training_root)
    source_hashes = {
        name: runtime._state_dict_sha256(state) for name, state in sources.items()
    }
    candidates: list[dict[str, Any]] = []
    for recipe in recipes:
        candidate_id = str(recipe["candidate_id"])
        state = mix_state(sources, recipe)
        state_sha256 = runtime._state_dict_sha256(state)
        candidate_dir = output_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        patch_path = candidate_dir / "gate_patch.pt"
        torch.save(
            {
                "schema": PATCH_SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "candidate_id": candidate_id,
                "recipe": recipe,
                "source_gate_state_sha256": source_hashes,
                "gate_state_sha256": state_sha256,
                "state_dict": state,
            },
            patch_path,
        )
        manifest: dict[str, Any] = {
            "schema": PATCH_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "candidate_id": candidate_id,
            "recipe": recipe,
            "source_gate_state_sha256": source_hashes,
            "gate_state_sha256": state_sha256,
            "parameter_tensors": len(state),
            "parameter_elements": sum(tensor.numel() for tensor in state.values()),
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
        write_json(candidate_dir / "manifest.json", manifest)
        candidates.append(manifest)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "recipe_payload_sha256": RECIPE_PAYLOAD_SHA256,
        "source_gate_state_sha256": source_hashes,
        "source_checkpoint_manifests": [
            {
                "step": int(manifest["step"]),
                "gate_state_sha256": manifest["gate_state_sha256"],
                "patch_sha256": manifest["patch_file"]["sha256"],
            }
            for manifest in manifests
        ],
        "candidates": candidates,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "all_convex": True,
        "non_gate_parameters_materialized": 0,
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
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(
        memory_dir=args.memory_dir.expanduser().resolve(strict=True),
        training_root=args.training_root.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "candidates": result["candidate_ids"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
