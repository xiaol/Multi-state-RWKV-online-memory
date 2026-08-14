#!/usr/bin/env python3
"""Materialize convex bridges between frozen repair endpoints."""

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
    run_natural_memory_native_scene_dualpath_repair_eval as dualpath_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_onpolicy_repair_eval as onpolicy_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_repair_bridge_materialization.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_repair_bridge_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_repair_bridge_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "44a97af8f0249f9c0cf129d6566bf0882f8ca1fc8f868c334cd2c7ae98fb0d7f"
RECIPE_PAYLOAD_SHA256 = "1ed39e66f38bbeeae8957080b9cb877a08222682a61147eae6e873344ac802c1"
SOURCE_ORDER = ("onpolicy", "dualpath")


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Repair-bridge protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Repair-bridge protocol hash differs")
    recipes = value.get("candidate_materialization", {}).get("candidates")
    if not isinstance(recipes, list) or canonical_sha256(recipes) != RECIPE_PAYLOAD_SHA256:
        raise ValueError("Repair-bridge recipe payload differs")
    return value


def gate_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    selected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
        and any(name.endswith(f".{family}") for family in training.GATE_FAMILIES)
    }
    if len(selected) != 126 or sum(tensor.numel() for tensor in selected.values()) != 108906:
        raise ValueError("Repair-bridge gate state differs")
    return selected


def load_sources(
    *,
    onpolicy_root: Path,
    dualpath_root: Path,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    list[Mapping[str, Any]],
]:
    source_specs = (
        ("onpolicy", onpolicy_root, onpolicy_eval, "onpolicy_repair_endpoint"),
        ("dualpath", dualpath_root, dualpath_eval, "dualpath_repair_endpoint"),
    )
    sources: dict[str, dict[str, torch.Tensor]] = {}
    provenance: list[Mapping[str, Any]] = []
    for source_name, root, evaluator, expected_id in source_specs:
        result, manifest = evaluator.validate_materialization(root)
        patch_path = root / "gate_patch.pt"
        payload = torch.load(patch_path, map_location="cpu", weights_only=True)
        if (
            result.get("candidate_id") != expected_id
            or manifest.get("candidate_id") != expected_id
            or not isinstance(payload, Mapping)
            or not isinstance(payload.get("state_dict"), Mapping)
            or payload.get("candidate_id") != expected_id
            or payload.get("gate_state_sha256") != manifest.get("gate_state_sha256")
            or sha256_file(patch_path) != manifest.get("patch_file", {}).get("sha256")
        ):
            raise ValueError(f"Repair-bridge source binding differs: {source_name}")
        state = gate_state(payload["state_dict"])
        if runtime._state_dict_sha256(state) != manifest["gate_state_sha256"]:
            raise ValueError(f"Repair-bridge source state differs: {source_name}")
        sources[source_name] = state
        provenance.append(
            {
                "source_name": source_name,
                "candidate_id": expected_id,
                "materialization_root": str(root),
                "materialization_result_sha256": sha256_file(root / "result.json"),
                "materialization_receipt_sha256": result["receipt"]["payload_sha256"],
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "patch_sha256": manifest["patch_file"]["sha256"],
                "gate_state_sha256": manifest["gate_state_sha256"],
            }
        )
    names = set(sources[SOURCE_ORDER[0]])
    if any(set(state) != names for state in sources.values()):
        raise ValueError("Repair-bridge source parameter names differ")
    return sources, provenance


def mix_state(
    sources: Mapping[str, Mapping[str, torch.Tensor]],
    recipe: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    weights = recipe.get("weights")
    denominator = recipe.get("denominator")
    if not isinstance(weights, Mapping) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("Repair-bridge recipe structure differs")
    integer_weights = {name: weights.get(name) for name in SOURCE_ORDER}
    if (
        any(not isinstance(weight, int) or weight < 0 for weight in integer_weights.values())
        or sum(integer_weights.values()) != denominator
        or set(weights) != set(SOURCE_ORDER)
    ):
        raise ValueError("Repair-bridge recipe is not convex")
    mixed: dict[str, torch.Tensor] = {}
    for name in sorted(sources[SOURCE_ORDER[0]]):
        reference = sources[SOURCE_ORDER[0]][name]
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for source_name in SOURCE_ORDER:
            source = sources[source_name][name]
            if source.shape != reference.shape:
                raise ValueError(f"Repair-bridge source shape differs: {name}")
            accumulator.add_(source.to(torch.float64), alpha=integer_weights[source_name])
        mixed[name] = (accumulator / denominator).to(reference.dtype)
    return mixed


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Repair-bridge output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize(
    *,
    onpolicy_root: Path,
    dualpath_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    protocol = validate_protocol()
    recipes = protocol["candidate_materialization"]["candidates"]
    sources, provenance = load_sources(
        onpolicy_root=onpolicy_root,
        dualpath_root=dualpath_root,
    )
    frozen = protocol["frozen_inputs"]
    expected_sources = {
        "onpolicy": frozen["onpolicy_materialization"],
        "dualpath": frozen["dualpath_materialization"],
    }
    for source in provenance:
        expected = expected_sources[source["source_name"]]
        required = {
            "candidate_id": expected["candidate_id"],
            "materialization_result_sha256": expected["result_file_sha256"],
            "materialization_receipt_sha256": expected["result_receipt_sha256"],
            "manifest_sha256": expected["manifest_sha256"],
            "patch_sha256": expected["patch_sha256"],
            "gate_state_sha256": expected["gate_state_sha256"],
        }
        if any(source.get(key) != value for key, value in required.items()):
            raise ValueError(
                f"Repair-bridge frozen source differs: {source['source_name']}"
            )
    if output_root.exists():
        raise ValueError(f"Repair-bridge output must be fresh: {output_root}")
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
        "source_materializations": provenance,
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
    parser.add_argument("--onpolicy-root", type=Path, required=True)
    parser.add_argument("--dualpath-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(
        onpolicy_root=args.onpolicy_root.expanduser().resolve(strict=True),
        dualpath_root=args.dualpath_root.expanduser().resolve(strict=True),
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
