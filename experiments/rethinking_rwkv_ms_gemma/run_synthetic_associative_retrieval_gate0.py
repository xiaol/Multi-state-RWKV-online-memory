#!/usr/bin/env python3
"""Run and validate frozen full-context Gate 0 for the associative canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_synthetic_associative_retrieval_canary as canary,
)
from experiments.rethinking_rwkv_ms_gemma.run_synthetic_state_identity_gate0 import (
    GENERATION_CONTRACT,
    HF_MIRROR_ENDPOINT,
    evaluate_case,
)


RECEIPT_SCHEMA = "rwkv_ms_synthetic_associative_retrieval_gate0.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision(cases: list[dict[str, Any]]) -> dict[str, Any]:
    exact_count = sum(bool(case["generation"]["exact"]) for case in cases)
    margins = [
        float(case["pair_target"]["source_minus_donor_logit_margin"])
        for case in cases
    ]
    finite = all(math.isfinite(value) for value in margins)
    criteria = {
        "exact_full_context_generation_4_of_4": exact_count == len(canary.CASE_LAYOUT),
        "finite_pair_target_margins": finite,
        "pair_target_margin_at_least_5_all_rows": (
            finite
            and len(margins) == len(canary.CASE_LAYOUT)
            and min(margins) >= canary.GATE0_MIN_PAIR_LOGIT_MARGIN
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "exact_generation_count": exact_count,
        "required_exact_generation_count": len(canary.CASE_LAYOUT),
        "pair_target_logit_margins": margins,
        "minimum_pair_target_logit_margin": min(margins) if margins else None,
        "required_minimum_pair_target_logit_margin": (
            canary.GATE0_MIN_PAIR_LOGIT_MARGIN
        ),
    }


def _validate_cases(
    cases: Any,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) != len(canary.CASE_LAYOUT):
        raise ValueError("Associative Gate 0 must contain exactly four cases")
    for index, case in enumerate(cases):
        donor_index = canary.DONOR_INDICES[index]
        if not isinstance(case, dict):
            raise ValueError(f"Associative Gate 0 case {index} is not an object")
        target_content = rows[index]["messages"][-1]["content"]
        donor_content = rows[donor_index]["messages"][-1]["content"]
        generation = case.get("generation")
        pair = case.get("pair_target")
        target_nll = case.get("target_response_nll")
        donor_nll = case.get("donor_response_nll")
        margin = (
            pair.get("source_minus_donor_logit_margin")
            if isinstance(pair, dict)
            else None
        )
        if (
            case.get("target_content") != target_content
            or case.get("donor_content") != donor_content
            or not isinstance(generation, dict)
            or generation.get("exact") is not True
            or generation.get("decoded_text") != target_content
            or not isinstance(pair, dict)
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or not isinstance(target_nll, dict)
            or not isinstance(donor_nll, dict)
            or float(target_nll.get("mean_nll", math.inf))
            >= float(donor_nll.get("mean_nll", -math.inf))
        ):
            raise ValueError(f"Associative Gate 0 evidence differs at row {index}")
        target_ids = case.get("target_response_token_ids")
        donor_ids = case.get("donor_response_token_ids")
        if (
            not isinstance(target_ids, list)
            or not target_ids
            or not isinstance(donor_ids, list)
            or not donor_ids
            or target_ids == donor_ids
        ):
            raise ValueError(f"Associative Gate 0 token evidence differs at row {index}")
    for index, donor_index in enumerate(canary.DONOR_INDICES):
        if (
            cases[index]["target_response_token_ids"]
            != cases[donor_index]["donor_response_token_ids"]
        ):
            raise ValueError("Associative Gate 0 target/donor tokens are not reciprocal")
    return cases


def run_gate0(
    source_manifest: Path,
    model_path: Path,
    output: Path,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Associative Gate 0 output must be fresh: {output}")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"Gate 0 requires HF_ENDPOINT={HF_MIRROR_ENDPOINT}")
    source = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=True,
    )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": {"": device},
        "local_files_only": True,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
    model.requires_grad_(False)

    rows = source["rows"]
    cases = [
        evaluate_case(
            model,
            tokenizer,
            rows[index],
            rows[canary.DONOR_INDICES[index]],
            device=device,
        )
        for index in range(len(rows))
    ]
    _validate_cases(cases, rows)
    decision = _decision(cases)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": _utc_now(),
        "source": {
            "schema": canary.SOURCE_SCHEMA,
            "manifest_path": str(source["manifest_path"]),
            "manifest_file_sha256": source["manifest_file_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "train_path": str(source["train_path"]),
            "train_sha256": source["train_sha256"],
            "rows_path": str(source["rows_path"]),
            "rows_sha256": source["rows_sha256"],
        },
        "model": source["model"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "adapter_loaded": False,
            "grad_enabled": False,
        },
        "generation_contract": GENERATION_CONTRACT,
        "identity_donor_indices": list(canary.DONOR_INDICES),
        "cases": cases,
        "gate": decision,
    }
    receipt["receipt_sha256"] = canary.canonical_sha256(receipt)
    canary.atomic_write(
        output,
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )
    receipt["receipt_path"] = str(output)
    receipt["receipt_file_sha256"] = canary.sha256_file(output)
    return receipt


def validate_receipt(
    receipt_path: Path,
    source_manifest: Path,
    model_path: Path,
    *,
    verify_model_hashes: bool,
) -> dict[str, Any]:
    path = receipt_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Associative Gate 0 receipt is invalid: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Associative Gate 0 receipt must be an object")
    unsigned = dict(receipt)
    declared_hash = unsigned.pop("receipt_sha256", None)
    if declared_hash != canary.canonical_sha256(unsigned):
        raise ValueError("Associative Gate 0 receipt SHA-256 differs")
    source = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=verify_model_hashes,
    )
    expected_source = {
        "schema": canary.SOURCE_SCHEMA,
        "manifest_path": str(source["manifest_path"]),
        "manifest_file_sha256": source["manifest_file_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "train_path": str(source["train_path"]),
        "train_sha256": source["train_sha256"],
        "rows_path": str(source["rows_path"]),
        "rows_sha256": source["rows_sha256"],
    }
    runtime = receipt.get("runtime")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("source") != expected_source
        or receipt.get("model") != source["model"]
        or receipt.get("identity_donor_indices") != list(canary.DONOR_INDICES)
        or not isinstance(runtime, dict)
        or runtime.get("hf_endpoint") != HF_MIRROR_ENDPOINT
        or runtime.get("adapter_loaded") is not False
        or runtime.get("grad_enabled") is not False
    ):
        raise ValueError("Associative Gate 0 receipt binding differs")
    cases = _validate_cases(receipt.get("cases"), source["rows"])
    decision = _decision(cases)
    if receipt.get("gate") != decision or decision.get("passed") is not True:
        raise ValueError("Associative Gate 0 receipt does not contain a passing gate")
    return {
        "valid": True,
        "receipt_path": str(path),
        "receipt_file_sha256": canary.sha256_file(path),
        "receipt_sha256": declared_hash,
        "gate": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--validate-receipt", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--verify-model-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_receipt is not None:
        result = validate_receipt(
            args.validate_receipt,
            args.source_manifest,
            args.model_path,
            verify_model_hashes=args.verify_model_hashes,
        )
    else:
        result = run_gate0(
            args.source_manifest,
            args.model_path,
            args.output,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
