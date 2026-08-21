from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_continuous_write_retrieval as retrieval,
)
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment


def _read_basis_result() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    receptance = torch.arange(96, dtype=torch.float32).reshape(1, 3, 2, 16)
    slot_reads = torch.arange(384, dtype=torch.float32).reshape(1, 3, 2, 4, 16)
    gate = torch.arange(48, dtype=torch.bfloat16).reshape(1, 3, 16)
    return receptance, slot_reads, gate


def _observer_module(
    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> SimpleNamespace:
    def original(
        state: torch.Tensor,
        memory_source_seq: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del state, memory_source_seq, token_mask
        return results.pop(0)

    return SimpleNamespace(
        rwkv_continuous_retrieval_original_read_basis=original,
        rwkv_continuous_retrieval_predictor_index=1,
        rwkv_continuous_retrieval_read_basis_calls=0,
        rwkv_continuous_retrieval_first_result=None,
        rwkv_continuous_retrieval_receptance=None,
        rwkv_continuous_retrieval_full_bytes_identical=False,
        rwkv_continuous_retrieval_result_shapes=None,
        rwkv_continuous_retrieval_result_dtypes=None,
    )


def test_read_basis_observer_requires_two_byte_identical_calls() -> None:
    first = _read_basis_result()
    second = tuple(tensor.clone() for tensor in first)
    module = _observer_module([first, second])
    state = torch.zeros(1)
    source = torch.zeros(1)

    retrieval._observed_read_basis(module, state, source, None)
    retrieval._observed_read_basis(module, state, source, None)

    assert module.rwkv_continuous_retrieval_read_basis_calls == 2
    assert module.rwkv_continuous_retrieval_full_bytes_identical is True
    assert module.rwkv_continuous_retrieval_first_result is None
    assert tuple(module.rwkv_continuous_retrieval_receptance.shape) == (1, 32)


def test_read_basis_observer_rejects_one_byte_difference() -> None:
    first = _read_basis_result()
    second = tuple(tensor.clone() for tensor in first)
    second[1].view(torch.uint8).flatten()[0] ^= 1
    module = _observer_module([first, second])
    state = torch.zeros(1)
    source = torch.zeros(1)

    retrieval._observed_read_basis(module, state, source, None)
    with pytest.raises(RuntimeError, match="read-basis bytes differ"):
        retrieval._observed_read_basis(module, state, source, None)


def test_first_prompt_boundary_is_first_label_minus_one() -> None:
    labels = torch.tensor([[-100, -100, -100, 7, 8]], dtype=torch.long)

    assert retrieval.first_prompt_boundary(labels) == (3, 2)


def test_retrieval_analysis_runs_one_locked_evaluation() -> None:
    module_names = ("layer.0", "layer.1")
    down = torch.zeros(16, 64)
    down[:, :16] = torch.eye(16)
    up = torch.zeros(32, 16)
    up[:16] = torch.eye(16)
    maps = {
        name: rwkv_continuous_write_alignment.FrozenMapWeights(
            down=down.clone(), up=up.clone()
        )
        for name in module_names
    }
    records = []
    for source_index in range(retrieval.RETRIEVAL_ROWS):
        sign = 1.0 if source_index % 2 == 0 else -1.0
        address = torch.zeros(2, 64)
        address[0, :16] = sign
        address[1, :16] = -sign
        receptance = torch.zeros(2, 32)
        receptance[0, :16] = sign
        receptance[1, :16] = -sign
        records.append(
            {
                "split": "retrieval",
                "source_index": source_index,
                "donor_source_index": source_index ^ 1,
                "write_address_full64": address.tolist(),
                "causal_prompt_boundary_receptance32": receptance.tolist(),
            }
        )

    result = retrieval.retrieval_analysis(records, module_names, maps)

    assert result["evaluation_calls"] == 1
    assert result["passed"] is True
    assert len(result["per_row"]) == retrieval.RETRIEVAL_ROWS
    assert result["aggregate"]["donor_positive_row_fraction"] == 1.0
    assert result["aggregate"]["layer_permuted_positive_row_fraction"] == 1.0


def test_fit_maps_rejects_non_fit_rows() -> None:
    records = [
        {
            "split": "retrieval",
            "write_address_full64": torch.zeros(1, 64).tolist(),
            "causal_prompt_boundary_receptance32": torch.zeros(1, 32).tolist(),
        }
    ] * retrieval.FIT_ROWS

    with pytest.raises(ValueError, match="only 64 FIT rows"):
        retrieval.fit_maps(records, ("layer.0",))


def test_protocol_has_canonical_nonplaceholder_receipt() -> None:
    protocol_path = Path(retrieval.PROTOCOL)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt")

    assert receipt["payload_sha256"] == retrieval.canonical_sha256(unsigned)
    assert receipt["payload_sha256"] == retrieval.PROTOCOL_PAYLOAD_SHA256
    assert retrieval.sha256_file(protocol_path) == retrieval.PROTOCOL_FILE_SHA256
    assert "PLACEHOLDER" not in protocol_path.read_text(encoding="utf-8")
