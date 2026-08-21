from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma"
    / "rwkv_bidirectional_sign_development_gate_core.py"
)
OPEN_FIT_MANIFEST = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts"
    / "natural_memory_native_rwkv_bidirectional_sign_open_fit_v1/manifest.json"
)


@pytest.fixture
def gate_core(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("RWKV_V5_EXACT_SOURCE_ROOT", str(REPO_ROOT))
    original_sys_path = list(sys.path)
    module_name = "rwkv_bidirectional_sign_development_gate_validation_test_module"
    spec = importlib.util.spec_from_file_location(module_name, CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_sys_path
        sys.modules.pop(module_name, None)


@pytest.fixture
def development_sources(gate_core: ModuleType) -> list[int]:
    manifest = json.loads(OPEN_FIT_MANIFEST.read_text(encoding="utf-8"))
    unsigned_manifest = dict(manifest)
    receipt = unsigned_manifest.pop("receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_manifest_without_receipt",
        "payload_sha256": gate_core.canonical_sha256(unsigned_manifest),
    }
    return [int(value) for value in manifest["splits"]["development"]["source_indices"]]


def _rank_shards(gate_core: ModuleType, development_sources: list[int]) -> list[dict]:
    shards = []
    for rank in range(gate_core.WORLD_SIZE):
        sources = development_sources[rank :: gate_core.WORLD_SIZE]
        rows = [
            {
                "source_index": source,
                "checks": {key: True for key in gate_core.ROW_CHECK_KEYS},
                "selected_codes": {
                    f"module_{module_index:02d}": {
                        "left": source,
                        "right": source ^ (module_index + 1),
                    }
                    for module_index in range(42)
                },
                "rebase_events": 0,
                "logit_max_abs": 0.0,
            }
            for source in sources
        ]
        shards.append(
            {
                "rank": rank,
                "sources": sources,
                "rows": rows,
                "data_audit": {
                    "local_target_sources": sources,
                    "local_decoded_sources": sources,
                    "development_row_hashes_verified": gate_core.DEVELOPMENT_ROWS,
                    "development_rows_decoded": len(sources),
                    "development_rows_tokenized": len(sources),
                    "development_rows_forwarded": len(sources),
                    "manifest_metadata_opened": True,
                    "development_bundle_opened": True,
                    "mechanics_bundle_opened": False,
                    "causal_bundle_opened": False,
                    "mechanics_rows_decoded": 0,
                    "mechanics_rows_tokenized": 0,
                    "mechanics_rows_forwarded": 0,
                    "causal_rows_decoded": 0,
                    "causal_rows_tokenized": 0,
                    "causal_rows_forwarded": 0,
                },
            }
        )
    return shards


def test_exact_four_rank_stride_assignment_is_accepted(
    gate_core: ModuleType,
    development_sources: list[int],
) -> None:
    gate_core._validate_rank_shards(
        _rank_shards(gate_core, development_sources),
        development_sources,
    )


def test_missing_rank_shard_is_rejected(
    gate_core: ModuleType,
    development_sources: list[int],
) -> None:
    shards = _rank_shards(gate_core, development_sources)

    with pytest.raises(ValueError, match="rank-shard coverage differs"):
        gate_core._validate_rank_shards(shards[:-1], development_sources)


@pytest.mark.parametrize("malformation", ["schema", "rank", "sources"])
def test_malformed_rank_shard_assignment_is_rejected(
    gate_core: ModuleType,
    development_sources: list[int],
    malformation: str,
) -> None:
    shards = _rank_shards(gate_core, development_sources)
    if malformation == "schema":
        shards[0]["unexpected"] = True
    elif malformation == "rank":
        shards[0]["rank"] = 1
    else:
        shards[0]["sources"] = list(reversed(shards[0]["sources"]))

    with pytest.raises(ValueError, match="schema or assignment differs"):
        gate_core._validate_rank_shards(shards, development_sources)


@pytest.mark.parametrize(
    "opened_key",
    ["mechanics_bundle_opened", "causal_bundle_opened"],
)
def test_protected_bundle_open_is_rejected(
    gate_core: ModuleType,
    development_sources: list[int],
    opened_key: str,
) -> None:
    shards = _rank_shards(gate_core, development_sources)
    shards[0]["data_audit"][opened_key] = True

    with pytest.raises(ValueError, match="rank-shard firewall differs"):
        gate_core._validate_rank_shards(shards, development_sources)


def test_result_receipt_tampering_is_rejected(gate_core: ModuleType) -> None:
    unsigned = {
        "schema": gate_core.RESULT_SCHEMA,
        "status": gate_core.FAIL_STATUS,
        "passed": False,
        "protocol_file_sha256": "0" * 64,
        "protocol_payload_sha256": "0" * 64,
        "launcher_sha256": "0" * 64,
        "gate_core_sha256": "0" * 64,
        "source_audit": {},
        "manifest_audit": {},
        "open_fit_audit": {},
        "model_audit": {},
        "installation": {},
        "runtime": {},
        "development_sources": [],
        "rank_shards": [],
        "mapping_pairs": [],
        "code_separation": {},
        "firewall": {},
        "fit_executed": False,
        "model_updates": 0,
        "adapter_saved": False,
        "generation_executed": False,
        "mechanics_protocol_authorized": False,
        "generation_authorized": False,
        "benchmark_authorized": False,
        "protected_splits_opened": [],
        "checks": {},
    }
    payload = {
        **unsigned,
        "receipt": {
            "algorithm": "sha256",
            "payload_scope": "canonical_result_without_receipt",
            "payload_sha256": gate_core.canonical_sha256(unsigned),
        },
    }
    payload["model_updates"] = 1

    with pytest.raises(ValueError, match="Development result receipt differs"):
        gate_core.validate_result_payload(payload)


def test_run_row_preserves_write_codes_until_paired_dual_read_capture(
    gate_core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        rwkv_bidirectional_sign_pending_rebase=None,
        rwkv_bidirectional_sign_read_kind=None,
        rwkv_bidirectional_sign_read_sequence=[],
        rwkv_bidirectional_sign_query_address=None,
        rwkv_bidirectional_sign_write_address=None,
        rwkv_bidirectional_sign_write_left_code=None,
        rwkv_bidirectional_sign_write_right_code=None,
        rwkv_bidirectional_sign_captures={},
        rwkv_bidirectional_sign_rebase_capture=None,
        rwkv_rotary_write_address=None,
        rwkv_rotary_read_captures={},
        enabled=False,
    )
    batch = SimpleNamespace(write_attention_mask=torch.ones(1, 2, dtype=torch.bool))
    expected_left = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    expected_right = -expected_left
    observed_reads: list[tuple[torch.Tensor, torch.Tensor]] = []

    monkeypatch.setattr(
        gate_core.sign,
        "iter_delta_mem_modules",
        lambda native_model: (("module", native_model),),
    )
    monkeypatch.setattr(
        gate_core.evolution,
        "collate_native_examples",
        lambda examples, pad_token_id, device: batch,
    )

    def reset_mode(native_model: SimpleNamespace, enabled: bool) -> None:
        gate_core.sign.clear_transient(native_model)
        native_model.enabled = enabled

    def native_write(
        native_model: SimpleNamespace,
        native_batch: SimpleNamespace,
        *,
        dtype: torch.dtype,
    ) -> None:
        assert native_batch is batch
        assert dtype is torch.bfloat16
        native_model.rwkv_bidirectional_sign_write_address = torch.ones(1, 2, 2)
        native_model.rwkv_bidirectional_sign_write_left_code = expected_left.clone()
        native_model.rwkv_bidirectional_sign_write_right_code = expected_right.clone()

    def snapshot_write(native_model: SimpleNamespace) -> dict[str, torch.Tensor]:
        return {
            "left": native_model.rwkv_bidirectional_sign_write_left_code.clone(),
            "right": native_model.rwkv_bidirectional_sign_write_right_code.clone(),
        }

    def native_read(
        native_model: SimpleNamespace,
        native_batch: SimpleNamespace,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert native_batch is batch
        assert dtype is torch.bfloat16
        assert native_model.rwkv_bidirectional_sign_captures == {}
        left = native_model.rwkv_bidirectional_sign_write_left_code
        right = native_model.rwkv_bidirectional_sign_write_right_code
        assert left is not None and right is not None
        observed_reads.append((left.clone(), right.clone()))
        native_model.rwkv_bidirectional_sign_captures = {
            kind: {
                "write_codes": left.clone(),
                "right_write_codes": right.clone(),
            }
            for kind in ("addressed", "global")
        }
        native_model.rwkv_bidirectional_sign_read_sequence = []
        return torch.tensor([[1.0, 2.0]])

    def snapshot_reads(native_model: SimpleNamespace) -> dict[str, dict[str, torch.Tensor]]:
        assert set(native_model.rwkv_bidirectional_sign_captures) == {
            "addressed",
            "global",
        }
        assert native_model.rwkv_bidirectional_sign_read_sequence == []
        return native_model.rwkv_bidirectional_sign_captures

    monkeypatch.setattr(gate_core, "_reset_mode", reset_mode)
    monkeypatch.setattr(gate_core.evolution, "_native_write", native_write)
    monkeypatch.setattr(gate_core, "_snapshot_state", lambda native_model: {})
    monkeypatch.setattr(gate_core, "_snapshot_write", snapshot_write)
    monkeypatch.setattr(gate_core.evolution, "_native_read", native_read)
    monkeypatch.setattr(gate_core, "_snapshot_reads", snapshot_reads)
    monkeypatch.setattr(
        gate_core,
        "_write_state_checks",
        lambda *args: (
            {"write_lifecycle_valid": True},
            {
                "selected_codes": {"module": {"left": 1, "right": 2}},
                "rebase_events": 0,
            },
        ),
    )
    monkeypatch.setattr(
        gate_core,
        "_read_checks",
        lambda baseline, bound: {"read_lifecycle_valid": True},
    )
    monkeypatch.setattr(gate_core, "reset_delta_mem_states", lambda native_model: None)
    monkeypatch.setattr(
        gate_core.evolution,
        "release_native_row_allocator_cache",
        lambda device: None,
    )

    row = gate_core._run_row(
        model,
        SimpleNamespace(pad_token_id=0),
        SimpleNamespace(source_ordinal=7),
        device=torch.device("cpu"),
    )

    assert row["source_index"] == 7
    assert row["checks"]["write_lifecycle_valid"] is True
    assert row["checks"]["read_lifecycle_valid"] is True
    assert len(observed_reads) == 2
    for left, right in observed_reads:
        assert torch.equal(left.view(torch.uint8), expected_left.view(torch.uint8))
        assert torch.equal(right.view(torch.uint8), expected_right.view(torch.uint8))
    assert model.rwkv_bidirectional_sign_write_left_code is None
    assert model.rwkv_bidirectional_sign_write_right_code is None
    assert model.rwkv_bidirectional_sign_captures == {}
    assert model.rwkv_bidirectional_sign_read_sequence == []


def test_output_nested_in_immutable_input_root_is_rejected(
    gate_core: ModuleType,
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "base-model"
    open_fit_root = tmp_path / "open-fit"
    protocol_path = tmp_path / "protocol.json"
    launcher_path = tmp_path / "launcher.py"
    base_model.mkdir()
    open_fit_root.mkdir()
    protocol_path.write_text("{}\n", encoding="utf-8")
    launcher_path.write_text("\n", encoding="utf-8")
    output_dir = base_model / "nested-output"

    with pytest.raises(ValueError, match="overlaps an immutable input root"):
        gate_core._validate_output_boundary(
            protocol_path=protocol_path,
            launcher_path=launcher_path,
            base_model=base_model,
            open_fit_root=open_fit_root,
            output_dir=output_dir,
        )
