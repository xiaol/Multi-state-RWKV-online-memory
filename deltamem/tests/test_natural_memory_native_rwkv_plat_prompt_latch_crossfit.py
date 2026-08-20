from __future__ import annotations

import ast
import copy
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_plat_prompt_latch_crossfit as screen,
)


EXPECTED_SPLIT_PAYLOAD_SHA256 = (
    "c1c1d27335563b47845633f5c3d33ea5409d5dfaf279dea9f084fd0b472c6f25"
)
EXPECTED_HELDOUT_SOURCES = (
    26,
    37,
    45,
    46,
    49,
    60,
    61,
    65,
    71,
    79,
    84,
    89,
    93,
    115,
    119,
    122,
    125,
    129,
    132,
    154,
    160,
    161,
    165,
    177,
    188,
    196,
    224,
    247,
    250,
    259,
    263,
    274,
    276,
    279,
    283,
    290,
    296,
    300,
    305,
    323,
    333,
    349,
    352,
    359,
)
EXPECTED_EXCLUDED_SOURCES = (
    4,
    12,
    24,
    28,
    47,
    59,
    66,
    70,
    74,
    80,
    98,
    101,
    103,
    106,
    121,
    126,
    128,
    144,
    145,
    155,
    164,
    171,
    172,
    182,
    200,
    219,
    221,
    223,
    231,
    234,
    236,
    244,
    248,
    258,
    260,
    261,
    270,
    280,
    281,
    288,
    330,
    331,
    338,
    341,
)


@pytest.fixture(scope="module")
def signed_contract() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    return screen.validate_protocol()


def _synthetic_record(
    source_index: int,
    plat_split: str,
    *,
    predictor_tokens: int = 2,
    offset: float = 0.0,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(700 + source_index)
    query = torch.randn(
        predictor_tokens,
        screen.LAYERS,
        screen.STATE_DIM,
        generator=generator,
    ) + offset
    correct = query + 0.01 * torch.randn(
        query.shape,
        generator=generator,
    )
    donor = -query + 0.01 * torch.randn(
        query.shape,
        generator=generator,
    )
    permuted = correct.roll(1, dims=1)
    return {
        "source_index": source_index,
        "plat_split": plat_split,
        "predictor_tokens": predictor_tokens,
        "query": query.tolist(),
        "correct": correct.tolist(),
        "matched_donor": donor.tolist(),
        "layer_permuted": permuted.tolist(),
    }


def test_signed_parent_artifacts_and_protocol_are_bound(signed_contract) -> None:
    protocol, parent_result, _ = signed_contract
    unsigned_protocol = dict(protocol)
    protocol_receipt = unsigned_protocol.pop("receipt")
    unsigned_parent = dict(parent_result)
    parent_receipt = unsigned_parent.pop("receipt")

    assert screen.PROTOCOL_PAYLOAD_SHA256 != "TO_BE_SIGNED"
    assert screen.canonical_sha256(unsigned_protocol) == screen.PROTOCOL_PAYLOAD_SHA256
    assert protocol_receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
    }
    assert screen.sha256_file(screen.PARENT_RESULT) == screen.PARENT_RESULT_SHA256
    assert screen.canonical_sha256(unsigned_parent) == screen.PARENT_RECEIPT
    assert parent_receipt["payload_sha256"] == screen.PARENT_RECEIPT
    assert protocol["authorization_basis"]["feature_shards"] == (
        screen._expected_shard_payload()
    )
    assert protocol["authorization_basis"]["code_dependencies"] == (
        screen._expected_dependency_payload()
    )
    dependency_paths = (
        Path(screen.parent.__file__),
        Path(screen.parent.shadow.__file__),
        Path(screen.parent.shadow.source.__file__),
        Path(screen.parent.bilinear.__file__),
    )
    assert [screen.sha256_file(path) for path in dependency_paths] == [
        item["sha256"] for item in screen._expected_dependency_payload()
    ]
    assert parent_result["protected_splits_opened"] == []


def test_nested_split_and_prior_heldout_exclusion_are_exact(signed_contract) -> None:
    protocol, parent_result, split_payload = signed_contract
    prior_rows = parent_result["crossfit_split"]["rows"]
    prior_train = set(parent_result["crossfit_split"]["train_sources"])
    prior_heldout = set(parent_result["crossfit_split"]["heldout_sources"])
    mapping = {
        int(row["source_index"]): int(row["donor_source_index"])
        for row in prior_rows
        if int(row["source_index"]) in prior_train
    }
    split, recomputed = screen.nested_split(mapping, prior_train, prior_heldout)

    assert recomputed == split_payload
    assert screen.canonical_sha256(split_payload) == EXPECTED_SPLIT_PAYLOAD_SHA256
    assert protocol["precommitted_nested_split"]["payload_sha256"] == (
        EXPECTED_SPLIT_PAYLOAD_SHA256
    )
    assert tuple(split_payload["heldout_component_indices"]) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        14,
    )
    assert tuple(split_payload["heldout_sources"]) == EXPECTED_HELDOUT_SOURCES
    assert tuple(split_payload["excluded_prior_heldout_sources"]) == (
        EXPECTED_EXCLUDED_SOURCES
    )
    assert set(split_payload["train_sources"]) == (
        prior_train - set(EXPECTED_HELDOUT_SOURCES)
    )
    assert prior_heldout == set(EXPECTED_EXCLUDED_SOURCES)
    assert len(split_payload["train_sources"]) == screen.TRAIN_ROWS
    assert len(split_payload["heldout_sources"]) == screen.HELDOUT_ROWS
    assert all(split[source] == split[donor] for source, donor in mapping.items())


def test_prompt_latch_expands_first_query_byte_identically() -> None:
    query = torch.arange(
        3 * screen.LAYERS * screen.STATE_DIM,
        dtype=torch.float32,
    ).reshape(3, screen.LAYERS, screen.STATE_DIM)

    latched = screen.prompt_latch_query(query)

    expected_bytes = query[0].contiguous().numpy().tobytes()
    assert latched.shape == query.shape
    assert all(
        latched[token].contiguous().numpy().tobytes() == expected_bytes
        for token in range(latched.size(0))
    )
    assert latched.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()


def test_feature_transform_leaves_all_recurrent_tensors_unchanged(
    monkeypatch,
) -> None:
    monkeypatch.setattr(screen, "TRAIN_ROWS", 1)
    record = _synthetic_record(0, "train", predictor_tokens=3)
    expected = {
        name: torch.tensor(record[name], dtype=torch.float32)
        for name in ("correct", "matched_donor", "layer_permuted")
    }

    features, lengths = screen._feature_tensors([record], "train")

    assert lengths == (3,)
    assert torch.equal(features["query"], features["query"][0:1].expand_as(features["query"]))
    for name, value in expected.items():
        assert torch.equal(features[name], value)


def test_heldout_mutation_cannot_change_train_head_or_thresholds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(screen, "TRAIN_ROWS", 1)
    monkeypatch.setattr(screen, "HELDOUT_ROWS", 1)
    monkeypatch.setattr(screen, "TRAIN_STEPS", 2)
    records = [
        _synthetic_record(0, "train"),
        _synthetic_record(1, "heldout"),
    ]
    mutated_records = copy.deepcopy(records)
    for name in ("query", "correct", "matched_donor", "layer_permuted"):
        heldout = torch.tensor(mutated_records[1][name], dtype=torch.float32)
        mutated_records[1][name] = (heldout.mul(-997.0).add(113.0)).tolist()

    first_head, first_thresholds, first_audit = screen.fit_train_only(records)
    second_head, second_thresholds, second_audit = screen.fit_train_only(
        mutated_records
    )

    assert torch.equal(first_thresholds, second_thresholds)
    assert first_audit == second_audit
    for name, value in first_head.state_dict().items():
        assert torch.equal(value, second_head.state_dict()[name])


def _gate_metric(**overrides: Any) -> dict[str, Any]:
    return {
        "tokens": 1,
        "rows": 1,
        "mean_gap": 0.05,
        "token_mean_gap": 0.05,
        "token_pairwise_positive_fraction": 0.95,
        "row_pairwise_positive_fraction": 0.95,
        "finite": True,
        **overrides,
    }


def _synthetic_gate_result(
    monkeypatch,
    donor: Mapping[str, Any],
    permuted: Mapping[str, Any],
) -> Mapping[str, Any]:
    feature = torch.zeros(1, screen.LAYERS, screen.STATE_DIM)
    features = {
        "query": feature,
        "correct": feature,
        "matched_donor": feature,
        "layer_permuted": feature,
    }
    monkeypatch.setattr(
        screen,
        "_feature_tensors",
        lambda records, split: (features, (1,)),
    )
    metrics = iter((donor, permuted))
    monkeypatch.setattr(
        screen.parent,
        "predictor_score_metrics",
        lambda *args, **kwargs: next(metrics),
    )
    return screen.evaluate_heldout(object(), [])


def test_synthetic_heldout_gates_pass_at_locked_boundaries(monkeypatch) -> None:
    result = _synthetic_gate_result(
        monkeypatch,
        _gate_metric(),
        _gate_metric(),
    )

    assert result["passed"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("expected_failed_check", "donor_override", "permuted_override"),
    (
        (
            "heldout_donor_token_fraction",
            {"token_pairwise_positive_fraction": 0.949},
            {},
        ),
        (
            "heldout_donor_row_fraction",
            {"row_pairwise_positive_fraction": 0.949},
            {},
        ),
        ("heldout_donor_mean_gap", {"mean_gap": 0.049}, {}),
        (
            "heldout_layer_permuted_token_fraction",
            {},
            {"token_pairwise_positive_fraction": 0.949},
        ),
        (
            "heldout_layer_permuted_row_fraction",
            {},
            {"row_pairwise_positive_fraction": 0.949},
        ),
        ("all_heldout_scores_finite", {"finite": False}, {}),
    ),
)
def test_each_synthetic_heldout_gate_fails_closed(
    monkeypatch,
    expected_failed_check: str,
    donor_override: Mapping[str, Any],
    permuted_override: Mapping[str, Any],
) -> None:
    result = _synthetic_gate_result(
        monkeypatch,
        _gate_metric(**donor_override),
        _gate_metric(**permuted_override),
    )

    assert result["passed"] is False
    assert result["checks"][expected_failed_check] is False


def test_protocol_never_authorizes_model_stage2_or_generation(signed_contract) -> None:
    protocol, parent_result, _ = signed_contract
    source = Path(screen.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert protocol["execution"]["model_loaded"] is False
    assert protocol["prompt_latch_transform"]["model_forward_required"] is False
    assert protocol["stage2_authorized"] is False
    assert protocol["model_or_adapter_training_authorized"] is False
    assert protocol["generation_authorized"] is False
    assert protocol["weights_saved"] is False
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert parent_result["stage2_executed"] is False
    assert parent_result["stage2"] is None
    assert called_names.isdisjoint(
        {"from_pretrained", "load_model_and_tokenizer", "load_state_dict", "load"}
    )


def test_execution_contract_requires_four_gloo_ranks_and_1800_seconds(
    monkeypatch,
    signed_contract,
) -> None:
    protocol, _, _ = signed_contract
    init_arguments: dict[str, Any] = {}

    assert screen.WORLD_SIZE == 4
    assert screen.DISTRIBUTED_TIMEOUT_SECONDS == 1800
    assert protocol["execution"] == {
        "world_size": 4,
        "backend": "gloo",
        "timeout_seconds": 1800,
        "rank0_only_fit": True,
        "other_ranks_verify_and_receive_source_and_result_bindings": True,
        "hf_endpoint": "https://hf-mirror.com",
        "fresh_output_required": True,
        "model_loaded": False,
    }

    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setattr(
        screen.dist,
        "init_process_group",
        lambda **kwargs: pytest.fail("three ranks must fail before initialization"),
    )
    with pytest.raises(RuntimeError, match="nproc_per_node=4"):
        screen._initialize_distributed()

    monkeypatch.setenv("WORLD_SIZE", "4")

    def capture_init(**kwargs: Any) -> None:
        init_arguments.update(kwargs)

    monkeypatch.setattr(screen.dist, "init_process_group", capture_init)
    monkeypatch.setattr(screen.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(screen.dist, "get_rank", lambda: 2)

    assert screen._initialize_distributed() == 2
    assert init_arguments["backend"] == "gloo"
    assert init_arguments["timeout"] == timedelta(seconds=1800)


def test_rank_local_validation_failure_reaches_consensus(monkeypatch) -> None:
    def gather_errors(gathered, local_error) -> None:
        gathered[:] = [
            local_error,
            None,
            {
                "rank": 2,
                "type": "ValueError",
                "message": "remote shard differs",
                "traceback_sha256": "0" * 64,
            },
            None,
        ]

    monkeypatch.setattr(screen.dist, "all_gather_object", gather_errors)
    monkeypatch.setattr(screen.dist, "get_world_size", lambda: 4)

    with pytest.raises(RuntimeError, match="remote shard differs"):
        screen._consensual_operation(lambda: "locally-valid", "validation")


def test_validated_source_state_includes_a_consensus_digest(signed_contract) -> None:
    protocol, parent_result, split_payload, source_audit, digest = (
        screen._validated_source_state()
    )

    assert protocol == signed_contract[0]
    assert parent_result == signed_contract[1]
    assert split_payload == signed_contract[2]
    assert digest == screen.canonical_sha256(source_audit)
    assert source_audit["code_dependencies"] == screen._expected_dependency_payload()
