#!/usr/bin/env python3
"""Build the preregistered reflected recurrent-routing checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_posttrain as stage1,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage4.v1"
STATUS = "stage4_checkpoint_built_development_evaluation_authorized"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage4_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "854e466f3e683925d5079ce78a45444b9920c665a6b7aeae293a056854fb935d"
STAGE2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
STAGE3_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage3_train10_v1"
STAGE2_DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_development_v1"
STAGE3_DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage3_development_v1"
STAGE2_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
STAGE3_RESULT_RECEIPT = "75fe0d877c3748f13629bcb0796076fcec4c63c97ebca89638acbc52bea9cb38"
STAGE2_DEVELOPMENT_RECEIPT = "88129262892d28795b23752d44289133c2f5416245847d944af17c9b4853a47a"
STAGE3_DEVELOPMENT_RECEIPT = "8abc7c1954f85352b1bdd763d6c794938cef423768a37df785ed661c917a62c4"
STAGE2_WEIGHTS_SHA256 = "7af3769fa34631329a54fb8caf44797a3a5598344e104680b6aa2cb108339248"
STAGE3_WEIGHTS_SHA256 = "25d487e8c2b2e5a14a98df65e9f29fd565448e50dc9d44b44d2fef6c55c6ee44"
CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
ALPHA = -1.0
TRAINED_SUFFIXES = (
    ".rwkv_route_query_proj",
    ".rwkv_route_state_proj",
    ".hrm_rwkv7_core.output.weight",
)


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    stage2 = common.validate_signed_json(STAGE2_ROOT / "result.json", STAGE2_RESULT_RECEIPT)
    stage3 = common.validate_signed_json(STAGE3_ROOT / "result.json", STAGE3_RESULT_RECEIPT)
    stage2_development = common.validate_signed_json(
        STAGE2_DEVELOPMENT_ROOT / "result.json",
        STAGE2_DEVELOPMENT_RECEIPT,
    )
    stage3_development = common.validate_signed_json(
        STAGE3_DEVELOPMENT_ROOT / "result.json",
        STAGE3_DEVELOPMENT_RECEIPT,
    )
    common.validate_split_artifacts()
    stage2_weights = STAGE2_ROOT / "adapter/delta_mem_adapter.pt"
    stage3_weights = STAGE3_ROOT / "adapter/delta_mem_adapter.pt"
    stage2_config = STAGE2_ROOT / "adapter/delta_mem_config.json"
    stage3_config = STAGE3_ROOT / "adapter/delta_mem_config.json"
    if (
        stage2.get("passed") is not True
        or stage3.get("passed") is not True
        or stage2_development.get("passed") is not False
        or stage3_development.get("passed") is not False
        or stage2_development.get("final_rows_opened") is not False
        or stage3_development.get("final_rows_opened") is not False
        or stage3_development.get("summary", {}).get("gates", {}).get(
            "overall_correct_over_all_controls"
        )
        is not True
        or stage3_development.get("summary", {}).get("gates", {}).get(
            "projected_carriers_fixed"
        )
        is not True
        or common.sha256_file(stage2_weights) != STAGE2_WEIGHTS_SHA256
        or common.sha256_file(stage3_weights) != STAGE3_WEIGHTS_SHA256
        or common.sha256_file(stage2_config) != CONFIG_SHA256
        or common.sha256_file(stage3_config) != CONFIG_SHA256
    ):
        raise ValueError("Stage-4 recurrent-routing lineage differs")
    return protocol, stage2, stage3


def build_checkpoint(output_dir: Path) -> Mapping[str, Any]:
    protocol, stage2_result, stage3_result = validate_lineage()
    resolved_output = output_dir.resolve()
    if resolved_output.exists():
        raise ValueError(f"Stage-4 output must be fresh: {resolved_output}")
    stage2_path = STAGE2_ROOT / "adapter/delta_mem_adapter.pt"
    stage3_path = STAGE3_ROOT / "adapter/delta_mem_adapter.pt"
    stage2_state = torch.load(stage2_path, map_location="cpu", weights_only=True)
    stage3_state = torch.load(stage3_path, map_location="cpu", weights_only=True)
    if stage2_state.keys() != stage3_state.keys():
        raise ValueError("Stage-2 and stage-3 adapter keys differ")
    output_state = {}
    trained_tensors = 0
    changed_trained_tensors = 0
    unchanged_nontrained_tensors = 0
    for name, stage2_tensor in stage2_state.items():
        stage3_tensor = stage3_state[name]
        trained = name.endswith(TRAINED_SUFFIXES)
        if not torch.equal(stage2_tensor, stage3_tensor) and not trained:
            raise ValueError(f"Nontrained adapter tensor changed in stage 3: {name}")
        if trained:
            trained_tensors += 1
            reflected = stage2_tensor + ALPHA * (stage3_tensor - stage2_tensor)
            output_state[name] = reflected.to(dtype=stage2_tensor.dtype)
            changed_trained_tensors += int(not torch.equal(reflected, stage2_tensor))
        else:
            output_state[name] = stage2_tensor
            unchanged_nontrained_tensors += 1
    if trained_tensors != 126 or changed_trained_tensors != 126:
        raise ValueError("Stage-4 trained tensor audit differs")
    adapter_dir = resolved_output / "adapter"
    adapter_dir.mkdir(parents=True)
    torch.save(output_state, adapter_dir / "delta_mem_adapter.pt")
    shutil.copyfile(
        STAGE2_ROOT / "adapter/delta_mem_config.json",
        adapter_dir / "delta_mem_config.json",
    )
    adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
    input_binding = {
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "stage2_training_result_receipt": STAGE2_RESULT_RECEIPT,
        "stage3_training_result_receipt": STAGE3_RESULT_RECEIPT,
        "stage2_adapter_files_sha256": stage2_result["adapter_files_sha256"],
        "stage3_adapter_files_sha256": stage3_result["adapter_files_sha256"],
        "stage2_development_result_receipt": STAGE2_DEVELOPMENT_RECEIPT,
        "stage3_development_result_receipt": STAGE3_DEVELOPMENT_RECEIPT,
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "final_commitment_receipt": common.FINAL_COMMITMENT_RECEIPT,
        "formula": "stage4 = stage2 + alpha * (stage3 - stage2)",
        "alpha": ALPHA,
        "trained_parameter_suffixes": list(TRAINED_SUFFIXES),
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    stage1.write_fresh_json(resolved_output / "input_binding.json", input_binding)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "passed": True,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "input_binding": input_binding,
        "checkpoint_audit": {
            "adapter_tensors": len(output_state),
            "trained_tensors": trained_tensors,
            "changed_trained_tensors": changed_trained_tensors,
            "unchanged_nontrained_tensors": unchanged_nontrained_tensors,
            "nontrained_stage2_stage3_equal": True,
            "alpha": ALPHA,
        },
        "adapter_files": adapter_files,
        "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
        "open_development_evaluation_authorized": True,
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": common.canonical_sha256(result),
    }
    stage1.write_fresh_json(resolved_output / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_checkpoint(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
