#!/usr/bin/env python3
"""Materialize the locked on-policy repair endpoint unchanged."""

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
    run_natural_memory_native_scene_onpolicy_repair as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_onpolicy_repair_materialization.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_onpolicy_repair_patch.v1"
CANDIDATE_ID = "onpolicy_repair_endpoint"


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
        raise ValueError("On-policy repair gate state differs")
    return selected


def load_training_endpoint(
    root: Path,
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    result = probe.validate_signed_json(
        root / "result.json",
        description="On-policy repair training result",
    )
    block = result.get("training")
    if (
        result.get("schema") != training.SCHEMA
        or result.get("status") != "training_complete_evaluation_pending"
        or result.get("protected_splits_opened") != []
        or not isinstance(block, Mapping)
        or block.get("seed") != training.SEED
        or block.get("updates") != training.TRAIN_UPDATES
        or block.get("rows") != training.TRAIN_UPDATES * training.GLOBAL_BATCH_SIZE
        or block.get("generated_rows") != training.TRAIN_UPDATES * training.GLOBAL_BATCH_SIZE
        or block.get("full_sequence_gold_ce_weight") != 0.0
        or block.get("synthetic_negative_unlikelihood_weight") != 0.0
        or block.get("initial_gate_state_sha256")
        != result.get("input_binding", {}).get("starting_checkpoint", {}).get(
            "runtime_gate_state_sha256"
        )
        or block.get("non_gate_unchanged") is not True
    ):
        raise ValueError("On-policy repair training result differs")
    manifest = probe.validate_signed_json(
        root / f"checkpoint-{training.TRAIN_UPDATES}" / "manifest.json",
        description="On-policy repair endpoint manifest",
    )
    if (
        manifest != block.get("checkpoint")
        or manifest.get("schema") != training.PATCH_SCHEMA
        or manifest.get("protocol_payload_sha256") != training.PROTOCOL_PAYLOAD_SHA256
        or manifest.get("seed") != training.SEED
        or manifest.get("step") != training.TRAIN_UPDATES
        or manifest.get("parameter_tensors") != 126
        or manifest.get("parameter_elements") != 108906
    ):
        raise ValueError("On-policy repair endpoint manifest differs")
    patch_path = root / f"checkpoint-{training.TRAIN_UPDATES}" / "gate_patch.pt"
    patch_file = manifest.get("patch_file")
    if (
        not isinstance(patch_file, Mapping)
        or patch_file.get("bytes") != patch_path.stat().st_size
        or patch_file.get("sha256") != sha256_file(patch_path)
    ):
        raise ValueError("On-policy repair endpoint patch differs")
    payload = torch.load(patch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("On-policy repair endpoint payload differs")
    state = gate_state(payload["state_dict"])
    if runtime._state_dict_sha256(state) != manifest["gate_state_sha256"]:
        raise ValueError("On-policy repair endpoint state differs")
    return state, {
        "training_root": str(root),
        "training_result_sha256": sha256_file(root / "result.json"),
        "training_result_receipt_sha256": result["receipt"]["payload_sha256"],
        "gate_state_sha256": manifest["gate_state_sha256"],
        "manifest_sha256": sha256_file(
            root / f"checkpoint-{training.TRAIN_UPDATES}" / "manifest.json"
        ),
        "patch_sha256": patch_file["sha256"],
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"On-policy repair output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize(
    *,
    training_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    training.validate_protocol()
    if output_root.exists():
        raise ValueError(f"On-policy repair output must be fresh: {output_root}")
    candidate, source = load_training_endpoint(training_root)
    candidate_sha256 = runtime._state_dict_sha256(candidate)
    source_hashes = {
        "checkpoint_16": training.STARTING_GATE_STATE_SHA256,
        str(training.SEED): candidate_sha256,
    }
    seed_weights = {str(training.SEED): 1}
    output_root.mkdir(parents=True, exist_ok=False)
    patch_path = output_root / "gate_patch.pt"
    torch.save(
        {
            "schema": PATCH_SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "candidate_id": CANDIDATE_ID,
            "source_gate_state_sha256": source_hashes,
            "seed_weights": seed_weights,
            "denominator": 1,
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
        "seed_weights": seed_weights,
        "denominator": 1,
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
        "source_training_artifact": source,
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
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(
        training_root=args.training_root.expanduser().resolve(strict=True),
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
