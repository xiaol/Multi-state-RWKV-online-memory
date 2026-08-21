from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_plmsc_code_alignment as v1,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_plmsc_code_alignment_v2 as screen,
)


V1_PROTOCOL_FILE_SHA256 = (
    "8428742275084b901d677395514228e2410323322d5041d92af6b6578f0cb93d"
)
V1_PROTOCOL_PAYLOAD_SHA256 = (
    "5f849ec1c6fbc590f4bb8df6c976136213c928dc3270acc9e2291ec88ca76400"
)
V1_RUNNER_SHA256 = (
    "2a30f50e921f018e7e3cedb0641cb582c3ffdf308f7261e54765ec07b6c0b6d9"
)
V1_FAILURE_FILE_SHA256 = (
    "f402b6e78dffe69bf6c2233e2beaee3791b3f179b2f927a3c338cc97798d0968"
)
V1_FAILURE_PAYLOAD_SHA256 = (
    "283adb5c5d8ed021eb4301156fd40a1ea41db101c49cc97c0b7a4127003ba1d5"
)
V1_FAILURE = (
    screen.SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_plmsc_code_alignment_v1"
    / "operational_failure.json"
)


@pytest.fixture(scope="module")
def signed_contract() -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[int, Mapping[str, Any]],
]:
    return screen.validate_protocol()


def _unsigned_signed_json(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    receipt = unsigned.pop("receipt")
    return unsigned, receipt


def test_protocol_and_operational_failure_are_canonically_bound(
    signed_contract,
) -> None:
    protocol = signed_contract[0]
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt")
    authorization = protocol["authorization_basis"]
    failure_unsigned, failure_receipt = _unsigned_signed_json(V1_FAILURE)

    assert screen.PROTOCOL_PAYLOAD_SHA256 != "TO_BE_SIGNED"
    assert screen.PROTOCOL_FILE_SHA256 != "TO_BE_SIGNED"
    assert screen.sha256_file(screen.PROTOCOL) == screen.PROTOCOL_FILE_SHA256
    assert screen.canonical_sha256(unsigned) == screen.PROTOCOL_PAYLOAD_SHA256
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
    }
    assert screen.sha256_file(v1.PROTOCOL) == V1_PROTOCOL_FILE_SHA256
    assert screen.sha256_file(Path(v1.__file__)) == V1_RUNNER_SHA256
    assert screen.sha256_file(V1_FAILURE) == V1_FAILURE_FILE_SHA256
    assert screen.canonical_sha256(failure_unsigned) == V1_FAILURE_PAYLOAD_SHA256
    assert failure_receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_operational_failure_without_receipt",
        "payload_sha256": V1_FAILURE_PAYLOAD_SHA256,
    }
    assert authorization["v1_protocol_file_sha256"] == V1_PROTOCOL_FILE_SHA256
    assert authorization["v1_runner_sha256"] == V1_RUNNER_SHA256
    assert authorization["v1_operational_failure_sha256"] == V1_FAILURE_FILE_SHA256
    assert (
        authorization["v1_operational_failure_receipt"]
        == V1_FAILURE_PAYLOAD_SHA256
    )
    assert authorization["v2_authorization_parent_commit"] == (
        "a3f63da49ddba4d021d77498dbf80917a35789c7"
    )
    assert failure_unsigned["bindings"]["protocol_payload_sha256"] == (
        V1_PROTOCOL_PAYLOAD_SHA256
    )
    assert failure_unsigned["status"] == (
        "plmsc_v1_observer_call_count_contract_failed_before_feature_save"
    )
    assert failure_unsigned["mechanics_gate_evaluated"] is False
    assert failure_unsigned["root_cause"]["observed_calls_per_anchor_before_failure"] == 2
    assert failure_unsigned["data_firewall"]["saved_feature_rows"] == 0
    assert failure_unsigned["data_firewall"]["code_map_fit_started"] is False
    assert failure_unsigned["data_firewall"]["mechanics_metrics_computed"] is False
    assert failure_unsigned["data_firewall"]["causal_rows_opened"] is False
    assert failure_unsigned["data_firewall"]["protected_splits_opened"] == []
    assert failure_unsigned["authorization"]["v1_retry_authorized"] is False
    assert failure_unsigned["authorization"]["bounded_v2_protocol_draft_authorized"]


def _protocol_differences(
    left: Any,
    right: Any,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add((*prefix, str(key)))
                continue
            differences |= _protocol_differences(
                left[key], right[key], (*prefix, str(key))
            )
        return differences
    if left != right:
        return {prefix}
    return set()


def test_v2_changes_only_the_allowlisted_observer_and_parent_bindings(
    signed_contract,
) -> None:
    v2_protocol = signed_contract[0]
    v1_protocol = json.loads(v1.PROTOCOL.read_text(encoding="utf-8"))
    differences = _protocol_differences(v1_protocol, v2_protocol)
    exact_allowed = {
        ("schema",),
        ("created_at",),
        ("objective",),
        ("authorization_basis", "authorization_scope"),
        ("authorization_basis", "v2_authorization_parent_commit"),
        ("exact_v5_capture", "canonical_read_basis_call_role"),
        ("exact_v5_capture", "duplicate_prompt_boundary_raw_byte_identity_required"),
        ("exact_v5_capture", "full_return_shapes_and_dtypes_recorded_per_call"),
        ("exact_v5_capture", "full_state_or_sequence_hashed"),
        ("exact_v5_capture", "per_call_prompt_boundary_sha256_recorded"),
        ("exact_v5_capture", "prompt_boundary_sha256_scope"),
        ("execution", "output_directory"),
        ("receipt", "payload_sha256"),
    }
    prefix_allowed = {
        ("authorization_basis", "v1_"),
        ("exact_v5_capture", "read_query_observer"),
        ("exact_v5_capture", "read_query"),
        ("exact_v5_capture", "read_basis"),
    }

    def allowed(path: tuple[str, ...]) -> bool:
        if path in exact_allowed:
            return True
        return any(
            len(path) >= len(prefix)
            and path[: len(prefix) - 1] == prefix[:-1]
            and path[len(prefix) - 1].startswith(prefix[-1])
            for prefix in prefix_allowed
        )

    unexpected = sorted(path for path in differences if not allowed(path))
    assert not unexpected, unexpected
    assert v2_protocol["precommitted_three_way_split"] == v1_protocol[
        "precommitted_three_way_split"
    ]
    assert v2_protocol["code_alignment_fit"] == v1_protocol["code_alignment_fit"]
    assert v2_protocol["locked_mechanics_gates"] == v1_protocol[
        "locked_mechanics_gates"
    ]
    assert v2_protocol["causal_firewall"] == v1_protocol["causal_firewall"]
    assert v2_protocol["stopping_rule"] == v1_protocol["stopping_rule"]


class _SyntheticReadBasis(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer_idx = layer
        self.state_read_dim = screen.STATE_WIDTH
        self.responses: list[torch.Tensor] = []
        self.last_original_return: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _rwkv_ms_token_state_read_basis(
        self,
        state: torch.Tensor,
        memory_source_seq: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del token_mask
        if not self.responses:
            raise AssertionError("synthetic read-basis response exhausted")
        result = (self.responses.pop(0), state, memory_source_seq)
        self.last_original_return = result
        return result


def _base_receptance(layer: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (
        torch.arange(4 * screen.STATE_WIDTH, dtype=torch.float32)
        .reshape(1, 4, 4, 8)
        .add(float(layer))
        .to(dtype=dtype)
    )


def _configure_synthetic_capture(monkeypatch) -> tuple[
    torch.nn.Module,
    list[tuple[str, _SyntheticReadBasis]],
    dict[str, torch.Tensor],
]:
    model = torch.nn.Module()
    modules = [
        (f"anchor_{layer}", _SyntheticReadBasis(layer)) for layer in screen.ANCHORS
    ]
    selected_stored_projected_key = {
        "value": torch.ones(1, 1, screen.STATE_WIDTH)
    }
    batch = SimpleNamespace(labels=torch.tensor([[-100, -100, 7, 8]]))
    monkeypatch.setattr(screen.causal_train, "ordered_modules", lambda _: modules)
    screen.install_anchor_read_capture(model)
    monkeypatch.setattr(
        screen.evolution, "collate_native_examples", lambda *args, **kwargs: batch
    )
    monkeypatch.setattr(screen.shadow, "reset_delta_mem_states", lambda _: None)
    monkeypatch.setattr(screen.evolution, "_native_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        screen.write_address_capture,
        "capture_write_addresses",
        lambda _: {
            name: selected_stored_projected_key["value"].clone()
            for name, _ in modules
        },
    )
    monkeypatch.setattr(screen.predictor, "reads_are_write_disabled", lambda _: True)
    monkeypatch.setattr(
        screen.evolution, "release_native_row_allocator_cache", lambda _: None
    )
    return model, modules, selected_stored_projected_key


def _set_responses(
    modules: Sequence[tuple[str, _SyntheticReadBasis]],
    responses: Sequence[torch.Tensor],
) -> None:
    for _, module in modules:
        module.responses = [value.clone() for value in responses]


def _native_read_calls(
    modules: Sequence[tuple[str, _SyntheticReadBasis]], calls: int
) -> torch.Tensor:
    for _, module in modules:
        for _ in range(calls):
            returned = module._rwkv_ms_token_state_read_basis(
                torch.ones(1), torch.ones(1), None
            )
            assert returned is module.last_original_return
    return torch.ones(1)


def test_two_identical_calls_emit_one_first_call_canonical_query_and_hashes(
    monkeypatch,
) -> None:
    model, modules, _ = _configure_synthetic_capture(monkeypatch)
    first = _base_receptance(0)
    second = first.clone()
    _set_responses(modules, [first, second])
    monkeypatch.setattr(
        screen.evolution,
        "_native_read",
        lambda *args, **kwargs: _native_read_calls(modules, 2),
    )

    feature = screen.capture_row(
        model, object(), pad_token_id=0, device=torch.device("cpu")
    )

    assert feature["read_basis_calls_per_anchor"] == [2, 2, 2, 2]
    assert feature["read_basis_call_roles"] == [
        "addressed_recurrent",
        "global_recurrent",
    ]
    assert feature["read_basis_observations_byte_identical_per_anchor"] == [True] * 4
    hashes = feature["read_basis_prompt_boundary_sha256_per_anchor_per_call"]
    assert len(hashes) == 4
    assert all(len(pair) == 2 and pair[0] == pair[1] for pair in hashes)
    assert all(len(digest) == 64 for pair in hashes for digest in pair)
    dtypes = feature["read_basis_return_dtypes_per_anchor_per_call"]
    expected_dtypes = ["torch.float32", "torch.float32", "torch.float32"]
    assert dtypes == [[expected_dtypes, expected_dtypes]] * 4
    shapes = feature["read_basis_return_shapes_per_anchor_per_call"]
    expected_shapes = [list(first.shape), [1], [1]]
    assert shapes == [[expected_shapes, expected_shapes]] * 4
    expected = first[:, 1].flatten(1)[0].tolist()
    assert feature["prompt_boundary_rwkv_receptance"] == [expected] * 4
    for _, module in modules:
        assert module.rwkv_plmsc_prompt_boundary_predictor_index is None
        assert module.rwkv_plmsc_prompt_boundary_r_seq_captures == []
        assert module.rwkv_plmsc_read_basis_calls == 0


def test_first_call_capture_is_detached_immutable_and_never_overwritten(
    monkeypatch,
) -> None:
    model, modules, _ = _configure_synthetic_capture(monkeypatch)
    original = _base_receptance(0)
    _set_responses(modules, [original, original])

    def native_read(*args, **kwargs) -> torch.Tensor:
        del args, kwargs
        for _, module in modules:
            first_result = module._rwkv_ms_token_state_read_basis(
                torch.ones(1), torch.ones(1), None
            )
            first_capture = module.rwkv_plmsc_prompt_boundary_r_seq_captures[0]
            expected = original[:, 1].flatten(1).clone()
            assert torch.equal(first_capture, expected)
            first_result[0][:, 1] += 100000.0
            assert torch.equal(first_capture, expected)
            module._rwkv_ms_token_state_read_basis(
                torch.ones(1), torch.ones(1), None
            )
            assert len(module.rwkv_plmsc_prompt_boundary_r_seq_captures) == 2
            assert module.rwkv_plmsc_prompt_boundary_r_seq_captures[0] is first_capture
            assert torch.equal(first_capture, expected)
        return torch.ones(1)

    monkeypatch.setattr(screen.evolution, "_native_read", native_read)
    feature = screen.capture_row(
        model, object(), pad_token_id=0, device=torch.device("cpu")
    )

    expected = original[:, 1].flatten(1)[0].tolist()
    assert feature["prompt_boundary_rwkv_receptance"] == [expected] * 4


@pytest.mark.parametrize("calls", (0, 1, 3))
def test_observer_rejects_every_call_count_other_than_two(
    monkeypatch,
    calls: int,
) -> None:
    model, modules, _ = _configure_synthetic_capture(monkeypatch)
    _set_responses(modules, [_base_receptance(0)] * max(calls, 1))
    monkeypatch.setattr(
        screen.evolution,
        "_native_read",
        lambda *args, **kwargs: _native_read_calls(modules, calls),
    )

    with pytest.raises(RuntimeError, match=r"exactly (?:two|twice)"):
        screen.capture_row(
            model, object(), pad_token_id=0, device=torch.device("cpu")
        )


def _flip_one_float32_bit(value: torch.Tensor) -> torch.Tensor:
    changed = value.clone().contiguous()
    bits = changed.view(torch.int32)
    bits[0, 1, 0, 0] ^= 1
    return changed


def test_raw_byte_helpers_distinguish_signed_zero_and_one_bit_changes() -> None:
    positive_zero = torch.tensor([0.0, 1.0], dtype=torch.float32)
    negative_zero = positive_zero.clone()
    negative_zero[0] = -0.0
    changed_bit = positive_zero.clone()
    changed_bit.view(torch.int32)[1] ^= 1

    assert torch.equal(positive_zero, negative_zero)
    assert not screen.tensor_raw_bytes_equal(positive_zero, negative_zero)
    assert screen.tensor_raw_bytes_sha256(positive_zero) != (
        screen.tensor_raw_bytes_sha256(negative_zero)
    )
    assert not screen.tensor_raw_bytes_equal(positive_zero, changed_bit)


def test_prompt_boundary_digest_binds_dtype_shape_and_exact_bytes() -> None:
    value = torch.arange(4, dtype=torch.float32)
    same_bytes_new_shape = value.reshape(2, 2)
    same_bytes_new_dtype = value.view(torch.int32)

    assert value.numpy().tobytes() == same_bytes_new_shape.numpy().tobytes()
    assert value.numpy().tobytes() == same_bytes_new_dtype.numpy().tobytes()
    assert screen.tensor_raw_bytes_sha256(value) != screen.tensor_raw_bytes_sha256(
        same_bytes_new_shape
    )
    assert screen.tensor_raw_bytes_sha256(value) != screen.tensor_raw_bytes_sha256(
        same_bytes_new_dtype
    )


@pytest.mark.parametrize("mismatch", ("one_bit", "signed_zero"))
def test_observer_requires_raw_byte_identity_not_numeric_equality(
    monkeypatch,
    mismatch: str,
) -> None:
    model, modules, _ = _configure_synthetic_capture(monkeypatch)
    first = _base_receptance(0)
    if mismatch == "one_bit":
        second = _flip_one_float32_bit(first)
    else:
        first[0, 1, 0, 0] = 0.0
        second = first.clone()
        second[0, 1, 0, 0] = -0.0
        assert torch.equal(first, second)
        assert first.numpy().tobytes() != second.numpy().tobytes()
    _set_responses(modules, [first, second])
    monkeypatch.setattr(
        screen.evolution,
        "_native_read",
        lambda *args, **kwargs: _native_read_calls(modules, 2),
    )

    with pytest.raises(RuntimeError, match="differ|byte-identical"):
        screen.capture_row(
            model, object(), pad_token_id=0, device=torch.device("cpu")
        )


@pytest.mark.parametrize("invalid", ("dtype", "full_shape", "nan_first", "inf_second"))
def test_observer_rejects_dtype_shape_and_nonfinite_changes(
    monkeypatch,
    invalid: str,
) -> None:
    model, modules, _ = _configure_synthetic_capture(monkeypatch)
    first = _base_receptance(0)
    second = first.clone()
    if invalid == "dtype":
        second = second.double()
    elif invalid == "full_shape":
        second = second.reshape(1, 4, 2, 16)
    elif invalid == "nan_first":
        first[0, 1, 0, 0] = torch.nan
    else:
        second[0, 1, 0, 0] = torch.inf
    _set_responses(modules, [first, second])
    monkeypatch.setattr(
        screen.evolution,
        "_native_read",
        lambda *args, **kwargs: _native_read_calls(modules, 2),
    )

    with pytest.raises(RuntimeError, match="shape|dtype|nonfinite|non-finite|differ"):
        screen.capture_row(
            model, object(), pad_token_id=0, device=torch.device("cpu")
        )


def test_selected_key_and_later_positions_cannot_change_live_boundary_query(
    monkeypatch,
) -> None:
    model, modules, selected_key = _configure_synthetic_capture(monkeypatch)

    def run_capture() -> Mapping[str, Any]:
        responses = _base_receptance(0)
        _set_responses(modules, [responses, responses])
        monkeypatch.setattr(
            screen.evolution,
            "_native_read",
            lambda *args, **kwargs: _native_read_calls(modules, 2),
        )
        return screen.capture_row(
            model, object(), pad_token_id=0, device=torch.device("cpu")
        )

    first = run_capture()
    selected_key["value"] = (
        torch.arange(screen.STATE_WIDTH, dtype=torch.float32).reshape(1, 1, -1)
        + 3.0
    )
    second = run_capture()
    original_factory = globals()["_base_receptance"]

    def changed_later(layer: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        value = original_factory(layer, dtype=dtype)
        value[:, 2:] += 100000.0
        return value

    monkeypatch.setitem(globals(), "_base_receptance", changed_later)
    third = run_capture()

    assert first["write_slot_address"] != second["write_slot_address"]
    assert first["prompt_boundary_rwkv_receptance"] == second[
        "prompt_boundary_rwkv_receptance"
    ]
    assert second["prompt_boundary_rwkv_receptance"] == third[
        "prompt_boundary_rwkv_receptance"
    ]
    assert second["read_basis_prompt_boundary_sha256_per_anchor_per_call"] == third[
        "read_basis_prompt_boundary_sha256_per_anchor_per_call"
    ]


def _valid_feature_row() -> tuple[
    Mapping[str, Any], Mapping[int, str], Mapping[int, Mapping[str, Any]]
]:
    query = torch.ones(4, 32, dtype=torch.float32)
    digest = screen.tensor_raw_bytes_sha256(query[0:1])
    unsigned = {
        "schema": screen.ROW_SCHEMA,
        "capture_rank": 0,
        "source_index": 0,
        "row_sha256": "1" * 64,
        "donor_source_index": 0,
        "donor_row_sha256": "1" * 64,
        "split": "fit",
        "anchors": list(screen.ANCHORS),
        "write_slot_address": torch.ones(4, 32).tolist(),
        "prompt_boundary_rwkv_receptance": query.tolist(),
        "first_supervised_label_index": 2,
        "prompt_boundary_predictor_index": 1,
        "predictor_definition": "first_supervised_label_index_minus_one",
        "predictor_vectors_per_row": 1,
        "answer_or_later_predictor_features_captured": False,
        "write_passes": 1,
        "read_passes": 1,
        "read_basis_calls_per_anchor": [2, 2, 2, 2],
        "read_basis_call_roles": ["addressed_recurrent", "global_recurrent"],
        "read_basis_observations_byte_identical_per_anchor": [True] * 4,
        "read_basis_prompt_boundary_sha256_per_anchor_per_call": [
            [digest, digest] for _ in screen.ANCHORS
        ],
        "read_basis_return_shapes_per_anchor_per_call": [
            [
                [[1, 4, 4, 8], [1], [1]],
                [[1, 4, 4, 8], [1], [1]],
            ]
            for _ in screen.ANCHORS
        ],
        "read_basis_return_dtypes_per_anchor_per_call": [
            [
                ["torch.float32", "torch.float32", "torch.float32"],
                ["torch.float32", "torch.float32", "torch.float32"],
            ]
            for _ in screen.ANCHORS
        ],
        "read_writes_enabled": False,
        "features_detached_and_cloned": True,
        "model_output_changed_by_capture": False,
        "binder_bridge_or_code_module_installed_during_capture": False,
    }
    return (
        screen._signed_feature_row(unsigned),
        {0: "fit"},
        {
            0: {
                "source_index": 0,
                "donor_source_index": 0,
                "row_sha256": "1" * 64,
                "donor_row_sha256": "1" * 64,
            }
        },
    )


@pytest.mark.parametrize(
    ("field", "mutation"),
    (
        ("read_basis_calls_per_anchor", lambda value: [1, 2, 2, 2]),
        (
            "read_basis_call_roles",
            lambda value: ["global_recurrent", "addressed_recurrent"],
        ),
        (
            "read_basis_observations_byte_identical_per_anchor",
            lambda value: [False, True, True, True],
        ),
        (
            "read_basis_prompt_boundary_sha256_per_anchor_per_call",
            lambda value: [["0" * 64, value[0][1]], *value[1:]],
        ),
        (
            "read_basis_return_shapes_per_anchor_per_call",
            lambda value: [
                [[[1, 4, 2, 16], [1], [1]], value[0][1]],
                *value[1:],
            ],
        ),
        (
            "read_basis_return_dtypes_per_anchor_per_call",
            lambda value: [
                [
                    ["torch.float64", "torch.float32", "torch.float32"],
                    value[0][1],
                ],
                *value[1:],
            ],
        ),
    ),
)
def test_feature_schema_rejects_unproved_duplicate_observations(
    field: str,
    mutation: Any,
) -> None:
    row, split, signed_rows = _valid_feature_row()
    screen._validate_feature_row(row, split, signed_rows)
    unsigned = dict(row)
    unsigned.pop("receipt")
    unsigned[field] = mutation(copy.deepcopy(unsigned[field]))
    mutated = screen._signed_feature_row(unsigned)

    with pytest.raises(ValueError, match="feature row differs"):
        screen._validate_feature_row(mutated, split, signed_rows)


def _feature_record(
    source: int, donor: int, split: str, *, offset: float
) -> Mapping[str, Any]:
    generator = torch.Generator().manual_seed(900 + source)
    write = torch.randn(
        len(screen.ANCHORS), screen.STATE_WIDTH, generator=generator
    ) + offset
    query = write + 0.01 * torch.randn(write.shape, generator=generator)
    return {
        "source_index": source,
        "donor_source_index": donor,
        "split": split,
        "write_slot_address": write.tolist(),
        "prompt_boundary_rwkv_receptance": query.tolist(),
    }


def test_mechanics_cannot_tune_maps_or_change_the_fit_digest(monkeypatch) -> None:
    monkeypatch.setattr(screen, "FIT_ROWS", 2)
    monkeypatch.setattr(screen, "TRAIN_STEPS", 3)
    records = [
        _feature_record(0, 1, "fit", offset=0.5),
        _feature_record(1, 0, "fit", offset=-0.5),
        _feature_record(2, 3, "mechanics", offset=1.5),
        _feature_record(3, 2, "mechanics", offset=-1.5),
    ]
    mutated = copy.deepcopy(records)
    for row in mutated:
        if row["split"] != "mechanics":
            continue
        for field in ("write_slot_address", "prompt_boundary_rwkv_receptance"):
            value = torch.tensor(row[field], dtype=torch.float32)
            row[field] = value.mul(-997.0).add(113.0).tolist()

    first_maps, first_audit = screen.train_fit_only(records)
    second_maps, second_audit = screen.train_fit_only(mutated)

    assert first_audit == second_audit
    assert first_audit["mechanics_or_causal_rows_used"] is False
    assert screen._code_map_sha256(first_maps) == screen._code_map_sha256(second_maps)
    for name, value in first_maps.state_dict().items():
        assert torch.equal(value, second_maps.state_dict()[name])


def test_split_gates_firewall_and_fit_constants_are_unchanged(signed_contract) -> None:
    protocol = signed_contract[0]
    v1_protocol = json.loads(v1.PROTOCOL.read_text(encoding="utf-8"))
    capture = protocol["exact_v5_capture"]

    assert screen.ANCHORS == v1.ANCHORS == (10, 21, 31, 41)
    assert screen.FIT_ROWS == v1.FIT_ROWS == 64
    assert screen.MECHANICS_ROWS == v1.MECHANICS_ROWS == 34
    assert screen.CAUSAL_ROWS == v1.CAUSAL_ROWS == 34
    assert screen.PRIOR_EXCLUDED_ROWS == v1.PRIOR_EXCLUDED_ROWS == 44
    assert screen.PLAT_EXCLUDED_ROWS == v1.PLAT_EXCLUDED_ROWS == 44
    assert screen.SPLIT_PAYLOAD_SHA256 == v1.SPLIT_PAYLOAD_SHA256
    assert screen.TRAIN_STEPS == v1.TRAIN_STEPS == 512
    assert screen.LOSS_WEIGHTS == v1.LOSS_WEIGHTS
    assert screen.CORRECT_ANCHOR_GATE == v1.CORRECT_ANCHOR_GATE
    assert screen.MINIMUM_DISTINCT_CODES == v1.MINIMUM_DISTINCT_CODES == 8
    assert screen.MAXIMUM_CODE_FRACTION == v1.MAXIMUM_CODE_FRACTION == 0.25
    assert screen.READ_BASIS_CALLS_PER_ANCHOR == 2
    assert list(screen.READ_BASIS_CALL_ROLES) == [
        "addressed_recurrent",
        "global_recurrent",
    ]
    assert capture["read_basis_calls_per_anchor"] == 2
    assert capture["read_basis_call_roles"] == list(screen.READ_BASIS_CALL_ROLES)
    assert capture["canonical_read_basis_call_role"] == "addressed_recurrent"
    assert capture["duplicate_prompt_boundary_raw_byte_identity_required"] is True
    assert capture["full_return_shapes_and_dtypes_recorded_per_call"] is True
    assert capture["prompt_boundary_sha256_scope"] == (
        "dtype_shape_and_exact_raw_bytes_of_only_the_selected_prompt_boundary_vector"
    )
    assert capture["full_state_or_sequence_hashed"] is False
    assert protocol["locked_mechanics_gates"] == v1_protocol[
        "locked_mechanics_gates"
    ]
    assert protocol["causal_firewall"] == v1_protocol["causal_firewall"]


def test_causal_and_both_excluded_sets_remain_opaque(signed_contract) -> None:
    _, _, _, split_payload, signed_rows = signed_contract
    causal = set(split_payload["causal_sources"])
    prior = set(split_payload["excluded_prior_heldout_sources"])
    plat = set(split_payload["excluded_plat_heldout_sources"])

    assert len(causal) == 34
    assert len(prior) == len(plat) == 44
    assert causal.isdisjoint(prior | plat)
    assert prior.isdisjoint(plat)
    assert all(
        set(signed_rows[source]) == {"source_index", "donor_source_index"}
        for source in causal
    )


def test_dataset_loader_never_decodes_tokenizes_or_retains_protected_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fit = list(range(0, 64))
    mechanics = list(range(64, 98))
    causal = list(range(98, 132))
    prior = list(range(132, 176))
    plat = list(range(176, 220))
    protected = set(causal + prior + plat)
    rows = [
        json.dumps(
            {
                "category": "TOP_SECRET" if source in protected else "capture",
                "source_index": source,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for source in range(361)
    ]
    dataset_path = tmp_path / screen.endpoint.DATASET_RELATIVE_PATH
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"\n".join(rows) + b"\n")
    signed_rows = {
        source: {
            "source_index": source,
            "donor_source_index": source,
            "row_sha256": hashlib.sha256(rows[source]).hexdigest(),
            "donor_row_sha256": hashlib.sha256(rows[source]).hexdigest(),
        }
        for source in fit + mechanics
    }
    signed_rows.update(
        {
            source: {"source_index": source, "donor_source_index": source}
            for source in causal
        }
    )
    split_payload = {
        "fit_sources": fit,
        "mechanics_sources": mechanics,
        "causal_sources": causal,
        "excluded_prior_heldout_sources": prior,
        "excluded_plat_heldout_sources": plat,
    }
    encoded: list[tuple[int, str]] = []

    def encode_row(
        tokenizer: Any,
        *,
        task: str,
        source_ordinal: int,
        raw_line: str,
    ) -> Any:
        del tokenizer, task
        encoded.append((source_ordinal, raw_line))
        return SimpleNamespace(
            row_sha256=hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
        )

    monkeypatch.setattr(screen.endpoint, "DATASET_SHA256", screen.sha256_file(dataset_path))
    monkeypatch.setattr(screen.evolution, "encode_native_full_row", encode_row)
    examples, audit = screen._load_local_examples(
        object(), tmp_path, signed_rows, split_payload, process_rank=0
    )

    expected_local = {source for source in fit + mechanics if source % 4 == 0}
    assert set(examples) == expected_local
    assert {source for source, _ in encoded} == expected_local
    assert all("TOP_SECRET" not in raw_line for _, raw_line in encoded)
    assert audit["capture_row_hashes_verified"] == 98
    assert audit["causal_rows_opaque_skipped"] == 34
    assert audit["excluded_prior_rows_opaque_skipped"] == 44
    assert audit["excluded_plat_rows_opaque_skipped"] == 44
    assert audit["retained_causal_or_excluded_payloads"] == 0
    assert audit["causal_rows_decoded"] == 0
    assert audit["causal_rows_tokenized"] == 0
    assert audit["causal_rows_model_forwarded"] == 0
    assert audit["causal_features_captured"] == 0


def test_observer_failure_reaches_consensus_before_write_or_later_collective(
    monkeypatch,
) -> None:
    context = object()
    observed: list[tuple[str, BaseException | None]] = []
    writes: list[Path] = []

    def consensus(
        actual_context: Any, *, phase: str, error: BaseException | None
    ) -> None:
        assert actual_context is context
        observed.append((phase, error))

    monkeypatch.setattr(screen.distributed, "phase_consensus", consensus)
    monkeypatch.setattr(
        screen,
        "_write_signed_json",
        lambda path, value: writes.append(path),
    )
    with pytest.raises(RuntimeError, match="observer mismatch"):
        screen._consensual_operation(
            context,
            phase="plmsc-v2-observer-capture",
            operation=lambda: (_ for _ in ()).throw(
                RuntimeError("observer mismatch")
            ),
        )

    assert len(observed) == 1
    assert observed[0][0] == "plmsc-v2-observer-capture"
    assert isinstance(observed[0][1], RuntimeError)
    assert writes == []


def test_protocol_and_runner_forbid_launch_training_generation_and_saves(
    signed_contract,
) -> None:
    protocol = signed_contract[0]
    source = Path(screen.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert protocol["stage2_authorized"] is False
    assert protocol["model_or_adapter_training_authorized"] is False
    assert protocol["generation_authorized"] is False
    assert protocol["adapter_saved"] is False
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert protocol["code_alignment_fit"]["code_map_weights_saved"] is False
    assert called_names.isdisjoint(
        {
            "run_stage2",
            "train_model",
            "train_adapter",
            "generate",
            "generate_native",
            "save_adapter",
            "save_pretrained",
            "save_state_dict",
            "torch_save",
            "linear_sum_assignment",
            "check_call",
            "check_output",
            "Popen",
        }
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "subprocess" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"stage2", "monomial_transform"}
        for node in ast.walk(tree)
    )
