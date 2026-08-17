#!/usr/bin/env python3
"""Screen DeepEmbed-style recurrent ChannelMix modulation on four GPUs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules, snapshot_delta_mem_weights  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as state_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_alignment_residual_screen as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_deepembed_ffn_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "0fd7434b2b999042ca662fb2d9291b2bd6837d465e5be68056c646ff484146c4"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 101
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
MIN_FFN_MATERIAL_LOGIT_DELTA = 1e-3
PASS_STATUS = "addressed_moe_deepembed_ffn_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_deepembed_ffn_screen_failed_training_blocked"
CANDIDATES = (
    {
        "candidate_id": "deepembed_ffn_t16_k2_ag03125_fg0001220703125",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 8192.0,
    },
    {
        "candidate_id": "deepembed_ffn_t16_k2_ag03125_fg000244140625",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 4096.0,
    },
    {
        "candidate_id": "deepembed_ffn_t16_k2_ag03125_fg00048828125",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 2048.0,
    },
)
ACTIVE_CANDIDATE: Mapping[str, Any] = CANDIDATES[0]
RUNNER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("DeepEmbed FFN protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("candidate_grid") != list(CANDIDATES)
        or architecture.get("hybrid_mode") != "addressed_moe_deepembed_ffn"
        or architecture.get("attention_hybrid_gain") != 0.03125
        or architecture.get("outer_ffn_gains")
        != [candidate["outer_ffn_gain"] for candidate in CANDIDATES]
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("DeepEmbed FFN screen protocol differs")
    return protocol


def build_config(candidate: Mapping[str, Any]) -> Any:
    return replace(
        top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="addressed_moe_deepembed_ffn",
        rwkv_ms_outer_ffn_gain=float(candidate["outer_ffn_gain"]),
    )


def configure_candidate(model: torch.nn.Module, candidate: Mapping[str, Any]) -> None:
    hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=str(candidate["hybrid_mode"]),
        hybrid_gain=float(candidate["hybrid_gain"]),
    )
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_ms_read_temperature = float(candidate["read_temperature"])
        module.rwkv_ms_read_top_k = int(candidate["read_top_k"])
        module.rwkv_ms_detach_read_scores = bool(candidate["detach_read_scores"])
        module.rwkv_ms_outer_ffn_gain = float(candidate["outer_ffn_gain"])
        probability = float(candidate["fusion_gate_probability"])
        if not hasattr(module, "memory_fusion_bias"):
            raise RuntimeError("DeepEmbed candidate requires a learned content gate")
        with torch.no_grad():
            module.memory_fusion_bias.fill_(
                torch.tensor(probability / (1.0 - probability)).log().item()
            )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    candidate = ACTIVE_CANDIDATE
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=build_config(candidate),
    )
    configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    deepembed_pre_hooks = sum(
        module._deepembed_ffn_pre_hook_handle is not None
        for _, module in modules
    )
    deepembed_down_hooks = sum(
        module._deepembed_ffn_down_pre_hook_handle is not None
        for _, module in modules
    )
    family_counts = {
        family: sum(hasattr(module, family) for _, module in modules)
        for family in (
            "rwkv_outer_ffn_down_weight",
            "rwkv_outer_ffn_gate_weight",
            "rwkv_outer_ffn_up_weight",
        )
    }
    configured = (
        len(modules) == preflight.EXPECTED_LAYERS
        and deepembed_pre_hooks == preflight.EXPECTED_LAYERS
        and deepembed_down_hooks == preflight.EXPECTED_LAYERS
        and all(count == preflight.EXPECTED_LAYERS for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == "addressed_moe_deepembed_ffn"
            and module.rwkv_ms_hybrid_gain == float(candidate["hybrid_gain"])
            and module.rwkv_ms_outer_ffn_gain == float(candidate["outer_ffn_gain"])
            and module.rwkv_ms_read_temperature == float(candidate["read_temperature"])
            and module.rwkv_ms_read_top_k == int(candidate["read_top_k"])
            and module.rwkv_ms_detach_read_scores is True
            and module.memory_fusion_mode == "content_gated_add"
            and hasattr(module, "rwkv_moe_bias")
            for _, module in modules
        )
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_addressed_moe_deepembed_ffn": configured,
        "deepembed_ffn_pre_hook_count": deepembed_pre_hooks,
        "deepembed_ffn_down_hook_count": deepembed_down_hooks,
        "deepembed_ffn_family_counts": family_counts,
        "deepembed_ffn_gain": float(candidate["outer_ffn_gain"]),
    }
    if not configured:
        raise RuntimeError(f"DeepEmbed FFN attachment failed: {audit!r}")
    return model, tokenizer, audit


def _set_outer_ffn_gain(model: torch.nn.Module, gain: float) -> list[float]:
    previous = []
    for _, module in iter_delta_mem_modules(model):
        previous.append(float(module.rwkv_ms_outer_ffn_gain))
        module.rwkv_ms_outer_ffn_gain = float(gain)
    return previous


def _restore_outer_ffn_gain(model: torch.nn.Module, previous: Sequence[float]) -> None:
    for (_, module), gain in zip(iter_delta_mem_modules(model), previous):
        module.rwkv_ms_outer_ffn_gain = float(gain)


def local_evidence(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    candidate = ACTIVE_CANDIDATE
    projected_only = hybrid_screen.read_logits(
        model,
        batch,
        states["correct"],
        readout_mode="projected_kv_slots",
    )
    condition_logits = {
        condition: hybrid_screen.read_logits(
            model,
            batch,
            state,
            readout_mode="projected_kv_rwkv_hybrid",
            hybrid_mode=str(candidate["hybrid_mode"]),
            hybrid_gain=float(candidate["hybrid_gain"]),
        )
        for condition, state in states.items()
    }
    previous = _set_outer_ffn_gain(model, 0.0)
    try:
        ffn_gain_zero = hybrid_screen.read_logits(
            model,
            batch,
            states["correct"],
            readout_mode="projected_kv_rwkv_hybrid",
            hybrid_mode=str(candidate["hybrid_mode"]),
            hybrid_gain=float(candidate["hybrid_gain"]),
        )
    finally:
        _restore_outer_ffn_gain(model, previous)
    comparisons = {
        "correct_vs_zero": hybrid_screen.compare_logits(
            condition_logits["correct"], condition_logits["zero"]
        ),
        "correct_vs_matched_donor": hybrid_screen.compare_logits(
            condition_logits["correct"], condition_logits["matched_donor"]
        ),
        "correct_vs_layer_permuted": hybrid_screen.compare_logits(
            condition_logits["correct"], condition_logits["layer_permuted"]
        ),
        "correct_vs_projected_only": hybrid_screen.compare_logits(
            condition_logits["correct"], projected_only
        ),
        "correct_vs_ffn_gain_zero": hybrid_screen.compare_logits(
            condition_logits["correct"], ffn_gain_zero
        ),
        "zero_vs_projected_only": hybrid_screen.compare_logits(
            condition_logits["zero"], projected_only
        ),
    }
    checks = {
        "zero_recurrent_exactly_equals_projected_only": torch.equal(
            condition_logits["zero"], projected_only
        ),
        "correct_vs_zero_material": comparisons["correct_vs_zero"][
            "max_abs_logit_delta"
        ]
        >= MIN_MATERIAL_LOGIT_DELTA,
        "correct_vs_matched_donor_material": comparisons[
            "correct_vs_matched_donor"
        ]["max_abs_logit_delta"]
        >= MIN_MATERIAL_LOGIT_DELTA,
        "correct_vs_layer_permuted_material": comparisons[
            "correct_vs_layer_permuted"
        ]["max_abs_logit_delta"]
        >= MIN_MATERIAL_LOGIT_DELTA,
        "correct_vs_projected_bounded": comparisons[
            "correct_vs_projected_only"
        ]["max_abs_logit_delta"]
        <= MAX_BOUNDED_LOGIT_DELTA,
        "deepembed_ffn_ablation_material": comparisons[
            "correct_vs_ffn_gain_zero"
        ]["max_abs_logit_delta"]
        >= MIN_FFN_MATERIAL_LOGIT_DELTA,
        "all_condition_logits_finite": all(
            metrics["all_finite"] for metrics in comparisons.values()
        ),
    }
    return {
        "candidate": dict(candidate),
        "checks": checks,
        "passed": all(checks.values()),
        "comparisons": comparisons,
    }


@contextmanager
def screen_bindings(candidate: Mapping[str, Any]) -> Iterator[None]:
    global ACTIVE_CANDIDATE
    previous_candidate = ACTIVE_CANDIDATE
    ACTIVE_CANDIDATE = candidate
    names = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "HF_MIRROR_ENDPOINT": HF_MIRROR_ENDPOINT,
        "BASE_MODEL": BASE_MODEL,
        "DATASET_ROOT": DATASET_ROOT,
        "WORLD_SIZE": WORLD_SIZE,
        "SEED": SEED,
        "MIN_MATERIAL_LOGIT_DELTA": MIN_MATERIAL_LOGIT_DELTA,
        "MAX_BOUNDED_LOGIT_DELTA": MAX_BOUNDED_LOGIT_DELTA,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "MODEL_AUDIT_KEY": "all_wrappers_addressed_moe_deepembed_ffn",
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "PRIOR_RESULT": RUNNER_BINDING_PATH,
        "PRIOR_RESULT_CODE_BINDING_KEY": "deepembed_ffn_runner_self_sha256",
        "SELECTED_CANDIDATE": candidate,
        "validate_protocol": validate_protocol,
        "load_model": load_model,
        "local_evidence": local_evidence,
    }
    previous = {name: getattr(shared, name) for name in names}
    try:
        for name, value in names.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)
        ACTIVE_CANDIDATE = previous_candidate


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("DeepEmbed FFN screen requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"DeepEmbed FFN screen output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="deepembed-ffn-output-freshness",
        error=freshness_error,
    )
    results: list[Mapping[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_dir = output_dir / str(candidate["candidate_id"])
        with screen_bindings(candidate):
            results.append(
                shared.run(
                    context=context,
                    output_dir=candidate_dir,
                    base_model=base_model,
                    dataset_root=dataset_root,
                )
            )
    passing = [result for result in results if result.get("passed") is True]
    selected = (
        min(
            passing,
            key=lambda result: (
                float(result["selected_candidate"]["outer_ffn_gain"]),
                float(
                    result["rank_evidence"][0]["comparisons"][
                        "correct_vs_projected_only"
                    ]["max_abs_logit_delta"]
                ),
                str(result["selected_candidate"]["candidate_id"]),
            ),
        )
        if passing
        else None
    )
    final: dict[str, Any] = {
        "schema": SCHEMA,
        "status": PASS_STATUS if selected is not None else FAIL_STATUS,
        "passed": selected is not None,
        "checks": {
            "four_distinct_a100_ranks": (
                len(context.rank_devices) == WORLD_SIZE
                and all("A100" in str(device["device_name"]) for device in context.rank_devices)
            ),
            "candidate_selected": selected is not None,
        },
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "base_model": str(base_model.expanduser().resolve()),
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "world_size": WORLD_SIZE,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "candidate_results": [
            {
                "candidate": result["selected_candidate"],
                "passed": result["passed"],
                "status": result["status"],
                "result_receipt": result["receipt"]["payload_sha256"],
                "result_dir": str(
                    output_dir / str(result["selected_candidate"]["candidate_id"])
                ),
            }
            for result in results
        ],
        "selected_candidate": None if selected is None else selected["selected_candidate"],
        "selected_candidate_result_receipt": (
            None if selected is None else selected["receipt"]["payload_sha256"]
        ),
        "training_authorized": selected is not None,
        "native_generation_authorized": False,
        "protected_splits_opened": [],
        "code_bindings": {
            "runner_sha256": sha256_file(RUNNER_BINDING_PATH),
            "protocol_file_sha256": sha256_file(PROTOCOL),
            "delta_impl_sha256": sha256_file(PROJECT_ROOT / "deltamem/core/delta_impl.py"),
            "shared_screen_runner_sha256": sha256_file(Path(shared.__file__)),
        },
    }
    final["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(final),
    }
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "result.json", final)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="deepembed-ffn-result-save",
        error=save_error,
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("DeepEmbed FFN screen requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            base_model=args.base_model,
            dataset_root=args.dataset_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "selected_candidate": result["selected_candidate"],
                "result_receipt": result["receipt"]["payload_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
