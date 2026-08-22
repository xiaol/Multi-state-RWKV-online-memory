from __future__ import annotations

import inspect

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_decoded_reconstruction as reconstruction,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_address_decoded_token_replacement as ad_rtr,
)
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment as alignment


def _rows(split: str, count: int) -> list[dict[str, object]]:
    rows = []
    for source in range(count):
        donor = source + count // 2 if source < count // 2 else source - count // 2
        rows.append(
            {
                "split": split,
                "source_index": source,
                "donor_source_index": donor,
            }
        )
    return rows


def test_unsigned_protocol_fails_before_any_input_access(monkeypatch) -> None:
    monkeypatch.setattr(reconstruction, "PROTOCOL_PAYLOAD_SHA256", "TO_BE_FILLED")
    monkeypatch.setattr(reconstruction, "PROTOCOL_FILE_SHA256", "TO_BE_FILLED")

    with pytest.raises(RuntimeError, match="not signed"):
        reconstruction.validate_protocol()


def test_row_loader_allows_only_open_splits_and_requires_reciprocal_donors(
    monkeypatch,
) -> None:
    fit_rows = _rows("fit", reconstruction.FIT_ROWS)
    monkeypatch.setattr(
        reconstruction.materializer,
        "_read_bundle",
        lambda *_args: fit_rows,
    )

    loaded = reconstruction._load_rows(
        reconstruction.DEFAULT_MATERIALIZATION,
        {},
        "fit",
    )
    assert len(loaded) == reconstruction.FIT_ROWS
    with pytest.raises(PermissionError, match="only FIT and retrieval"):
        reconstruction._load_rows(
            reconstruction.DEFAULT_MATERIALIZATION,
            {},
            "mechanics",
        )

    fit_rows[0]["donor_source_index"] = 1
    with pytest.raises(ValueError, match="row contract differs"):
        reconstruction._load_rows(
            reconstruction.DEFAULT_MATERIALIZATION,
            {},
            "fit",
        )


def test_capture_source_proves_effective_address_is_occupied_key() -> None:
    source = inspect.getsource(reconstruction._extract_row_features)

    assert "torch.equal(effective_address, selected_keys)" in source
    assert "write_formula_byte_exact_all_modules" in source
    assert "write_value_same_object_and_bytes_all_modules" in source


def test_capture_loop_reaches_consensus_before_collective_gather() -> None:
    source = inspect.getsource(reconstruction._capture_split)

    assert source.index("distributed.phase_consensus(") < source.index(
        "distributed.gather_objects(context, captured)"
    )


def test_run_freezes_decoder_before_opening_retrieval() -> None:
    source = inspect.getsource(reconstruction.run)

    assert source.index('phase="ad-rtr-fit-decoder-validation"') < source.index(
        'phase="ad-rtr-open-retrieval-bundle-after-decoder-freeze"'
    )


def test_control_slots_gather_source_active_and_scatter_to_target() -> None:
    value = torch.zeros(2, 3, 1, 4, 2, 2)
    source_occupied = torch.zeros(2, 3, 4, dtype=torch.bool)
    target_occupied = torch.zeros_like(source_occupied)
    for row in range(2):
        for module in range(3):
            source_slot = (row + module) % 4
            target_slot = (row + module + 2) % 4
            source_occupied[row, module, source_slot] = True
            target_occupied[row, module, target_slot] = True
            value[row, module, :, source_slot] = row * 10 + module + 1

    aligned = ad_rtr.canonicalize_active_slots(
        value,
        source_occupied,
        target_occupied,
        slot_dim=3,
    )

    assert torch.equal(
        aligned.ne(0).any(dim=(2, 4, 5)),
        target_occupied,
    )
    assert torch.equal(
        aligned.masked_select(
            target_occupied[:, :, None, :, None, None].expand_as(aligned)
        ),
        value.masked_select(
            source_occupied[:, :, None, :, None, None].expand_as(value)
        ),
    )


def _frozen_map() -> alignment.FrozenMapWeights:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(181)
    return alignment.FrozenMapWeights(
        down=torch.randn(ad_rtr.MAP_RANK, ad_rtr.ADDRESS_DIM, generator=generator),
        up=torch.randn(ad_rtr.STATE_DIM, ad_rtr.MAP_RANK, generator=generator),
    )


def _analysis_fixture() -> tuple[
    dict[str, torch.Tensor],
    list[dict[str, object]],
    tuple[str, ...],
    dict[str, alignment.FrozenMapWeights],
    dict[str, ad_rtr.FullRankRidgeDecoder],
    dict[str, object],
]:
    rows = reconstruction.RETRIEVAL_ROWS
    modules = reconstruction.MODULES
    generator = torch.Generator(device="cpu")
    generator.manual_seed(191)
    base_state = torch.randn(rows // 2, 1, 4, 32, 32, generator=generator)
    base_keys = torch.randn(rows // 2, 4, 64, generator=generator)
    state = torch.empty(rows, modules, 1, 4, 32, 32)
    keys = torch.empty(rows, modules, 4, 64)
    occupied = torch.zeros(rows, modules, 4, dtype=torch.bool)
    occupied[:, :, 0] = True
    for row in range(rows):
        pair = row % (rows // 2)
        row_sign = 1.0 if row < rows // 2 else -1.0
        for module in range(modules):
            module_sign = 1.0 if module % 2 == 0 else -1.0
            state[row, module] = row_sign * module_sign * base_state[pair]
            keys[row, module] = row_sign * module_sign * base_keys[pair]
    state[:, :, :, 1:] = 0.0
    keys[:, :, 1:] = 0.0
    module_names = tuple(f"layer.{index}" for index in range(modules))
    frozen_map = _frozen_map()
    maps = {name: frozen_map for name in module_names}
    values = torch.empty(rows, modules, 4, 32)
    for index, name in enumerate(module_names):
        values[:, index] = ad_rtr.address_decoded_slots(
            state[:, index],
            keys[:, index],
            torch.zeros(rows, 4, 32),
            occupied[:, index],
            maps[name],
        ).contracted
    decoders = {
        name: ad_rtr.FullRankRidgeDecoder(
            weight=torch.eye(ad_rtr.STATE_DIM),
            ridge=reconstruction.RIDGE,
        )
        for name in module_names
    }
    protocol = {
        "required_gates": {
            "correct_mean_cosine_minimum": 0.9,
            "correct_row_mean_cosine_minimum": 0.8,
            "correct_row_fraction_at_or_above_minimum": 0.95,
            "correct_minus_matched_donor_state_mean_cosine_gap_minimum": 0.05,
            "correct_minus_matched_donor_state_positive_row_fraction_minimum": 0.95,
            "correct_minus_wrong_address_mean_cosine_gap_minimum": 0.05,
            "correct_minus_wrong_address_positive_row_fraction_minimum": 0.95,
            "correct_minus_layer_rolled_state_mean_cosine_gap_minimum": 0.05,
            "correct_minus_layer_rolled_state_positive_row_fraction_minimum": 0.95,
            "module_correct_mean_cosine_minimum": 0.8,
            "module_control_mean_cosine_gap_minimum": 0.05,
            "module_control_positive_row_fraction_minimum": 0.95,
            "module_identity_pass_fraction_minimum": 0.95,
        }
    }
    return (
        {"state": state, "keys": keys, "values": values, "occupied": occupied},
        _rows("retrieval", rows),
        module_names,
        maps,
        decoders,
        protocol,
    )


def test_every_active_target_vector_must_be_nonzero() -> None:
    heldout, rows, module_names, maps, decoders, protocol = _analysis_fixture()
    heldout["values"][0, 0, 0] = 0.0

    analysis = reconstruction._analyze_reconstruction(
        heldout,
        rows,
        module_names,
        maps,
        decoders,
        protocol,
    )

    assert analysis["checks"][
        "all_active_addresses_states_targets_decodes_nonzero"
    ] is False
    assert analysis["passed"] is False


def test_module_coverage_gate_catches_failures_hidden_by_aggregate() -> None:
    heldout, rows, module_names, maps, decoders, protocol = _analysis_fixture()
    partial = torch.diag(torch.tensor([1.0] * 16 + [0.0] * 16))
    for name in module_names[:3]:
        decoders[name] = ad_rtr.FullRankRidgeDecoder(
            weight=partial,
            ridge=reconstruction.RIDGE,
        )

    analysis = reconstruction._analyze_reconstruction(
        heldout,
        rows,
        module_names,
        maps,
        decoders,
        protocol,
    )

    assert analysis["checks"]["correct_mean_cosine"] is True
    assert analysis["module_gate"]["passed_modules"] == 39
    assert analysis["checks"]["module_identity_pass_fraction"] is False
    assert analysis["passed"] is False


def test_decoder_artifact_is_reloaded_and_digest_checked(tmp_path) -> None:
    module_names = tuple(f"layer.{index}" for index in range(reconstruction.MODULES))
    decoders = {
        name: ad_rtr.FullRankRidgeDecoder(
            weight=torch.eye(ad_rtr.STATE_DIM),
            ridge=reconstruction.RIDGE,
        )
        for name in module_names
    }
    path = tmp_path / "address-decoded-value-decoders.pt"

    saved = reconstruction._save_decoders(path, decoders, module_names)
    validated = reconstruction._validate_decoder_artifact(
        path,
        module_names=module_names,
        expected_digest=saved["decoder_digest"],
        expected_sha256=saved["sha256"],
    )

    assert validated == saved
