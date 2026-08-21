#!/usr/bin/env python3
"""Build the signed protocol for the four-A100 bidirectional-sign dev gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_bidirectional_sign_open_fit as open_fit,
)
from experiments.rethinking_rwkv_ms_gemma.rwkv_diagonal_sign_binding import (
    deterministic_projection,
)


HF_ENDPOINT = "https://hf-mirror.com"
SCHEMA = "rwkv_ms_bidirectional_sign_development_gate.v1"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BASE_MODEL_ID = "google/gemma-4-E4B-it"
BASE_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
SIGNED_SOURCE_COMMIT = "cd7deb91a3dbbf15b7f82cf7bb445e3d7664d631"
PLMSC_PROTOCOL_FILE_SHA256 = (
    "15fd83f0cc9eb636f6264d5d2fb80a830e612ac144a123a4b4e7be5d483ed5ed"
)
PLMSC_PROTOCOL_PAYLOAD_SHA256 = (
    "a66d2b855491e0c814d0c524d9d66f70cd03164c9aac14b30da1fb47e769142e"
)
PLMSC_RESULT_FILE_SHA256 = (
    "b7dce00737c928abc13729b19e24ccfe803b9dce6dde62b9d9d944971a295544"
)
PLMSC_RESULT_RECEIPT = (
    "23c7cfdf0cdf0fb747010615cfe271ae7d7c0cddd7bd9a90401179033100fda7"
)
V5_ADAPTER = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5_r1/"
    "adapter"
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_row(
    *,
    role: str,
    scope: str,
    root: Path,
    relative: str,
) -> dict[str, str]:
    path = (root / relative).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"Protocol manifest path escapes {scope}: {relative}")
    return {
        "role": role,
        "scope": scope,
        "path": relative,
        "sha256": sha256_file(path),
    }


def _project_rows(open_fit_root: Path) -> list[dict[str, str]]:
    roles = {
        "binding": "experiments/rethinking_rwkv_ms_gemma/rwkv_bidirectional_sign_binding.py",
        "diagonal_sign_dependency": "experiments/rethinking_rwkv_ms_gemma/rwkv_diagonal_sign_binding.py",
        "integration": "experiments/rethinking_rwkv_ms_gemma/rwkv_bidirectional_sign_integration.py",
        "gate_core": "experiments/rethinking_rwkv_ms_gemma/rwkv_bidirectional_sign_development_gate_core.py",
        "open_fit_materializer": "experiments/rethinking_rwkv_ms_gemma/materialize_natural_memory_native_rwkv_bidirectional_sign_open_fit.py",
        "plmsc_runner": "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_plmsc_code_alignment_v2.py",
        "exact_v5_loader": "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_v5_shadow_crossfit.py",
        "shadow_model_loader": "experiments/rethinking_rwkv_ms_gemma/common.py",
    }
    open_fit_relative = open_fit_root.resolve(strict=True).relative_to(PROJECT_ROOT)
    roles["open_fit_manifest"] = (open_fit_relative / "manifest.json").as_posix()
    roles["open_fit_development_bundle"] = (
        open_fit_relative / "development.jsonl"
    ).as_posix()
    return [
        _manifest_row(
            role=role,
            scope="project",
            root=PROJECT_ROOT,
            relative=relative,
        )
        for role, relative in roles.items()
    ]


def _projection_manifest() -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for layer in range(42):
        name = f"model.language_model.layers.{layer}.self_attn"
        result[name] = {
            "left": _tensor_sha256(deterministic_projection(64, 131 + 2 * layer, 32)),
            "right": _tensor_sha256(
                deterministic_projection(64, 131 + 2 * layer + 1, 32)
            ),
        }
    return result


def build_protocol(
    *,
    base_model: Path,
    signed_source_root: Path,
    open_fit_root: Path,
) -> Mapping[str, Any]:
    if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
        raise RuntimeError(f"HF_ENDPOINT must be explicitly set to {HF_ENDPOINT}")
    if _git_output(signed_source_root, "rev-parse", "HEAD") != SIGNED_SOURCE_COMMIT:
        raise ValueError("Signed exact-v5 source commit differs")
    if _git_output(
        signed_source_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("Signed exact-v5 source has tracked changes")
    open_fit_validated = open_fit.validate_materialization(
        open_fit_root,
        bundles=("development",),
    )
    manifest = open_fit_validated["manifest"]
    splits = manifest["splits"]

    manifest_rows = _project_rows(open_fit_root)
    for role, relative in (
        ("signed_delta_api", "deltamem/core/delta.py"),
        ("signed_delta_implementation", "deltamem/core/delta_impl.py"),
        ("chat_template_runtime", "deltamem/chat_templates.py"),
        (
            "signed_model_loader",
            "experiments/rethinking_rwkv_ms_gemma/common.py",
        ),
        (
            "distributed_runtime",
            "experiments/rethinking_rwkv_ms_gemma/natural_memory_distributed.py",
        ),
        (
            "native_evolution_runtime",
            "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_evolution.py",
        ),
        (
            "causal_state_runtime",
            "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_value_causal_train.py",
        ),
        (
            "dataset_endpoint_runtime",
            "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval.py",
        ),
        (
            "hardware_runtime",
            "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_value_screen.py",
        ),
    ):
        manifest_rows.append(
            _manifest_row(
                role=role,
                scope="signed_source",
                root=signed_source_root,
                relative=relative,
            )
        )
    for relative in (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        manifest_rows.append(
            _manifest_row(
                role=f"base_model_{relative.replace('.', '_')}",
                scope="base_model",
                root=base_model,
                relative=relative,
            )
        )
    for relative in ("delta_mem_adapter.pt", "delta_mem_config.json"):
        manifest_rows.append(
            _manifest_row(
                role=f"v5_adapter_{relative.replace('.', '_')}",
                scope="v5_adapter",
                root=V5_ADAPTER,
                relative=relative,
            )
        )

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": "2026-08-21T00:00:00+08:00",
        "objective": (
            "Run one exact four-A100 development64 gate for full-key bidirectional "
            "diagonal-sign RWKV state binding using only the signed publisher-TRAIN-"
            "derived open-fit development replay."
        ),
        "authorization_basis": {
            "plmsc_protocol_file_sha256": PLMSC_PROTOCOL_FILE_SHA256,
            "plmsc_protocol_payload_sha256": PLMSC_PROTOCOL_PAYLOAD_SHA256,
            "plmsc_result_file_sha256": PLMSC_RESULT_FILE_SHA256,
            "plmsc_result_receipt": PLMSC_RESULT_RECEIPT,
            "plmsc_status": "plmsc_code_alignment_failed_family_retired",
            "scope": "new independent identity mechanics family; development only",
        },
        "architecture": {
            "state_encoding": "S_bound=D_value(address)@S@D_key(address)",
            "write_features": "v-left and k/a/b-right",
            "read_features": "right-coded native receptance then left decode",
            "state_dim": 32,
            "address_dim": 64,
            "projection_seed": 131,
            "frequency": 64.0,
            "state_rebase_required": True,
            "projections_trainable": False,
            "native_routing_changed": False,
        },
        "execution": {
            "world_size": 4,
            "hf_endpoint": HF_ENDPOINT,
            "attempts": 1,
            "resume": False,
            "fit_or_training": False,
            "fresh_output_required": True,
            "python": "3.12.13",
            "torch": "2.6.0+cu128",
            "cuda": "12.8",
        },
        "firewall": {
            "manifest_metadata_opened": True,
            "development_bundle": "development.jsonl",
            "development_rows_opened": 64,
            "mechanics_bundle_opened": False,
            "mechanics_rows_opened": 0,
            "causal_bundle_opened": False,
            "causal_rows_opened": 0,
        },
        "frozen_inputs": {
            "base_model": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "signed_source_commit": SIGNED_SOURCE_COMMIT,
            "required_source_root_environment": "RWKV_V5_EXACT_SOURCE_ROOT",
            "open_fit_manifest_sha256": sha256_file(open_fit_root / "manifest.json"),
            "open_fit_manifest_receipt": manifest["receipt"]["payload_sha256"],
            "open_fit_source_sha256": open_fit.SOURCE_SHA256,
            "v5_adapter": V5_ADAPTER.relative_to(PROJECT_ROOT).as_posix(),
        },
        "split": {
            "source_namespace": open_fit.SOURCE_NAMESPACE,
            "materialization_schema": open_fit.MANIFEST_SCHEMA,
            "source_set_sha256": open_fit.EXPECTED_SOURCE_SET_SHA256,
            "mapping_pairs_sha256": open_fit.EXPECTED_GLOBAL_MAPPING_SHA256,
            "ordered_components_sha256": open_fit.EXPECTED_ORDERED_COMPONENTS_SHA256,
            "development_sources": splits["development"]["source_indices"],
            "development_sha256": open_fit.EXPECTED_SOURCE_SHA256["development"],
            "development_mapping_pairs": splits["development"]["mapping_pairs"],
            "development_mapping_sha256": open_fit.EXPECTED_MAPPING_SHA256[
                "development"
            ],
            "mechanics_sources": splits["mechanics"]["source_indices"],
            "mechanics_sha256": open_fit.EXPECTED_SOURCE_SHA256["mechanics"],
            "mechanics_mapping_sha256": open_fit.EXPECTED_MAPPING_SHA256["mechanics"],
            "causal_sources": splits["causal"]["source_indices"],
            "causal_sha256": open_fit.EXPECTED_SOURCE_SHA256["causal"],
            "causal_mapping_sha256": open_fit.EXPECTED_MAPPING_SHA256["causal"],
        },
        "projection_sha256": _projection_manifest(),
        "manifests": {"files": manifest_rows},
        "mechanics_stage_authorized": False,
        "model_or_adapter_training_authorized": False,
        "generation_authorized": False,
        "benchmark_authorized": False,
        "protected_splits_opened_by_this_protocol": [],
    }
    protocol["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(protocol),
    }
    return protocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--signed-source-root", type=Path, required=True)
    parser.add_argument("--open-fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Protocol output must be fresh: {output}")
    protocol = build_protocol(
        base_model=args.base_model.expanduser().resolve(strict=True),
        signed_source_root=args.signed_source_root.expanduser().resolve(strict=True),
        open_fit_root=args.open_fit_root.expanduser().resolve(strict=True),
    )
    output.write_text(
        json.dumps(protocol, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"protocol_file_sha256={sha256_file(output)}")
    print(f"protocol_payload_sha256={protocol['receipt']['payload_sha256']}")


if __name__ == "__main__":
    main()
