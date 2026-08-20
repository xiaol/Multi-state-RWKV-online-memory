from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_plmsc_code_alignment as screen,
)


EXPECTED_SPLIT_PAYLOAD_SHA256 = (
    "ab2a225c4a4710316a88b61fa48b5a48bb5ff0772c28d04455db20370fd0f737"
)
EXPECTED_DONOR_MAPPING_SHA256 = (
    "a15184300c04d707a135fd6f1ffd69c460985ddf07357a543fb3ee063530f6c6"
)
EXPECTED_CAPTURE_SOURCES_SHA256 = (
    "2d84030580d983ad6ab41956ebc7c9d5c92322de79dee346c5ea1e1adf6e6381"
)
EXPECTED_PRIOR_EXCLUDED_SHA256 = (
    "8f1fb3ec2fee2e8d01d7dec0b081d78d9b6b628f243b481dd082305b88a66eea"
)
EXPECTED_PLAT_EXCLUDED_SHA256 = (
    "bd7b7ddec5beab243a56b964140432ad18041c203dfbb0d430399e847f03f938"
)
EXPECTED_DISTRIBUTED_SHA256 = (
    "09fd08b4750469c1364c28a935b443895f988f17ccc8102ede3ecce6bed6f44d"
)
EXPECTED_SIGNED_DELTA_API_SHA256 = (
    "144023cf0ef24970f93d2ffaf80a7265dfb137a32c1326df84e43aee80898f0c"
)
EXPECTED_SIGNED_RUNTIME_BINDINGS = {
    "native_capture_runtime": (
        "run_natural_memory_native_evolution.py",
        "6abbe06f249ec8fba942f0f865ff9d92485a13fe95d8c9eab0f308e0c0e258e7",
    ),
    "anchor_module_runtime": (
        "run_natural_memory_native_rwkv_addressed_value_causal_train.py",
        "3036e7c75c1dedd31ab7f3d8aa79126c849a5c64e5e37395bbe3e2c43822fbc7",
    ),
    "dataset_endpoint_contract": (
        "run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval.py",
        "ddb2da83266967ab877c928d7d21662f9bcfbd225c84fdbdd619fcbc2c159756",
    ),
    "a100_hardware_contract": (
        "run_natural_memory_native_rwkv_addressed_value_screen.py",
        "b42cb3455679b1799a72b262958df5fd5a85211bc5b37730c24c70309309d654",
    ),
}


@pytest.fixture(scope="module")
def signed_contract() -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[int, Mapping[str, Any]],
]:
    return screen.validate_protocol()


def _artifact_paths() -> Mapping[str, Path]:
    return {
        "plat_parent_result_sha256": screen.PLAT_RESULT,
        "plat_parent_protocol_sha256": screen.plat.PROTOCOL,
        "predictor_parent_result_sha256": screen.PREDICTOR_RESULT,
        "v5_result_sha256": screen.shadow.V5_RESULT,
        "v5_adapter_weights_sha256": (
            screen.shadow.V5_ADAPTER / "delta_mem_adapter.pt"
        ),
        "v5_adapter_config_sha256": (
            screen.shadow.V5_ADAPTER / "delta_mem_config.json"
        ),
    }


def test_protocol_receipt_artifacts_and_every_dependency_are_byte_bound(
    signed_contract,
) -> None:
    protocol = signed_contract[0]
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt")
    authorization = protocol["authorization_basis"]

    assert screen.PROTOCOL_PAYLOAD_SHA256 != "TO_BE_SIGNED"
    assert screen.PROTOCOL_FILE_SHA256 != "TO_BE_SIGNED"
    assert screen.sha256_file(screen.PROTOCOL) == screen.PROTOCOL_FILE_SHA256
    assert screen.canonical_sha256(unsigned) == screen.PROTOCOL_PAYLOAD_SHA256
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
    }
    for field, path in _artifact_paths().items():
        assert screen.sha256_file(path) == authorization[field]

    dependencies = screen._dependency_payload()
    assert authorization["code_dependencies"] == dependencies
    dependency_roles = [item["role"] for item in dependencies]
    assert dependency_roles[:9] == [
        "plat_split_and_parent_contract",
        "predictor_metadata_and_causal_boundary_contract",
        "exact_v5_loader_and_capture_contract",
        "v5_endpoint_topology_contract",
        "model_and_tokenizer_loader",
        "native_capture_runtime",
        "anchor_module_runtime",
        "dataset_endpoint_contract",
        "a100_hardware_contract",
    ]
    assert dependency_roles.index("signed_exact_v5_delta_api") < dependency_roles.index(
        "signed_exact_v5_delta_implementation"
    )
    dependency_paths = {
        Path(module.__file__).name: Path(module.__file__)
        for module in (
            screen.plat,
            screen.predictor,
            screen.shadow,
            screen.shadow.core_impl,
            screen.write_address_capture,
            screen.distributed,
            screen.evolution,
            screen.causal_train,
            screen.endpoint,
            screen.hardware,
        )
    }
    dependency_paths[screen.V5_TOPOLOGY_PATH.name] = screen.V5_TOPOLOGY_PATH
    dependency_paths[screen.MODEL_COMMON_PATH.name] = screen.MODEL_COMMON_PATH
    delta_api_path = Path(screen.signed_delta_api.__file__).resolve()
    dependency_paths[delta_api_path.name] = delta_api_path
    assert {item["basename"] for item in dependencies} == set(dependency_paths)
    for item in dependencies:
        assert screen.sha256_file(dependency_paths[item["basename"]]) == item["sha256"]
    distributed_binding = next(
        item for item in dependencies if item["role"] == "distributed_runtime"
    )
    assert distributed_binding == {
        "role": "distributed_runtime",
        "basename": "natural_memory_distributed.py",
        "sha256": EXPECTED_DISTRIBUTED_SHA256,
    }
    delta_api_binding = next(
        item for item in dependencies if item["role"] == "signed_exact_v5_delta_api"
    )
    assert delta_api_binding == {
        "role": "signed_exact_v5_delta_api",
        "basename": "delta.py",
        "sha256": EXPECTED_SIGNED_DELTA_API_SHA256,
    }
    signed_root = screen.shadow.SIGNED_SOURCE_ROOT.resolve()
    assert delta_api_path.is_relative_to(signed_root)
    assert Path(screen.shadow.core_impl.__file__).resolve().is_relative_to(signed_root)
    for role, (basename, expected_sha256) in EXPECTED_SIGNED_RUNTIME_BINDINGS.items():
        binding = next(item for item in dependencies if item["role"] == role)
        assert binding == {
            "role": role,
            "basename": basename,
            "sha256": expected_sha256,
        }
        assert dependency_paths[basename].resolve().is_relative_to(signed_root)


def test_signed_source_setup_import_precedes_workspace_deltamem_hazards() -> None:
    tree = ast.parse(Path(screen.__file__).read_text(encoding="utf-8"))
    project_imports: list[tuple[int, set[str]]] = []
    for position, node in enumerate(tree.body):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "experiments.rethinking_rwkv_ms_gemma":
            continue
        project_imports.append((position, {alias.name for alias in node.names}))

    plat_name = "run_natural_memory_native_rwkv_plat_prompt_latch_crossfit"
    hazardous = {
        "common",
        "run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train",
        "rwkv_query_state_identity",
    }
    plat_position = next(
        position for position, names in project_imports if plat_name in names
    )
    hazard_positions = [
        position for position, names in project_imports if names & hazardous
    ]

    assert hazard_positions
    assert plat_position < min(hazard_positions)
    assert project_imports[0][1] == {plat_name}


def test_three_way_split_and_both_failed_heldouts_are_exact(
    signed_contract,
) -> None:
    protocol, predictor_result, plat_result, split_payload, signed_rows = (
        signed_contract
    )
    split, recomputed, recomputed_rows = screen.derive_three_way_split(
        predictor_result, plat_result
    )
    signed_split = protocol["precommitted_three_way_split"]
    fit = set(split_payload["fit_sources"])
    mechanics = set(split_payload["mechanics_sources"])
    causal = set(split_payload["causal_sources"])
    prior_excluded = set(split_payload["excluded_prior_heldout_sources"])
    plat_excluded = set(split_payload["excluded_plat_heldout_sources"])

    assert recomputed == split_payload
    assert recomputed_rows == signed_rows
    assert screen.canonical_sha256(split_payload) == EXPECTED_SPLIT_PAYLOAD_SHA256
    assert signed_split["payload_sha256"] == EXPECTED_SPLIT_PAYLOAD_SHA256
    assert signed_split["eligible_donor_mapping_pairs_sha256"] == (
        EXPECTED_DONOR_MAPPING_SHA256
    )
    assert len(fit) == 64
    assert len(mechanics) == 34
    assert len(causal) == 34
    assert len(prior_excluded) == 44
    assert len(plat_excluded) == 44
    assert len(signed_rows) == 132
    assert not (fit & mechanics or fit & causal or mechanics & causal)
    assert not ((fit | mechanics | causal) & (prior_excluded | plat_excluded))
    assert not prior_excluded & plat_excluded
    assert screen.canonical_sha256(sorted(prior_excluded)) == (
        EXPECTED_PRIOR_EXCLUDED_SHA256
    )
    assert screen.canonical_sha256(sorted(plat_excluded)) == (
        EXPECTED_PLAT_EXCLUDED_SHA256
    )
    assert screen.canonical_sha256(sorted(fit | mechanics)) == (
        EXPECTED_CAPTURE_SOURCES_SHA256
    )
    assert all(split[source] == "fit" for source in fit)
    assert all(split[source] == "mechanics" for source in mechanics)
    assert all(split[source] == "causal" for source in causal)


def test_no_donor_edge_crosses_and_causal_contracts_are_metadata_only(
    signed_contract,
) -> None:
    _, _, _, split_payload, signed_rows = signed_contract
    split = {
        **{source: "fit" for source in split_payload["fit_sources"]},
        **{source: "mechanics" for source in split_payload["mechanics_sources"]},
        **{source: "causal" for source in split_payload["causal_sources"]},
    }
    causal = set(split_payload["causal_sources"])

    assert all(
        split[source] == split[int(contract["donor_source_index"])]
        for source, contract in signed_rows.items()
    )
    assert all(
        set(signed_rows[source]) == {"source_index", "donor_source_index"}
        for source in causal
    )
    assert all(
        "row_sha256" in signed_rows[source]
        and "donor_row_sha256" in signed_rows[source]
        for source in set(split) - causal
    )


def _raw_row(category: str, source_index: int) -> bytes:
    return json.dumps(
        {"category": category, "source_index": source_index},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_dataset_loader_hashes_only_capture_rows_and_keeps_others_opaque(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fit = list(range(0, 64))
    mechanics = list(range(64, 98))
    causal = list(range(98, 132))
    prior_excluded = list(range(132, 176))
    plat_excluded = list(range(176, 220))
    categories = {
        **{source: "capture" for source in fit + mechanics},
        **{source: "TOP_SECRET_CAUSAL" for source in causal},
        **{source: "TOP_SECRET_PRIOR" for source in prior_excluded},
        **{source: "TOP_SECRET_PLAT" for source in plat_excluded},
    }
    rows = [
        _raw_row(categories.get(source, "unprotected"), source)
        for source in range(361)
    ]
    dataset_path = tmp_path / screen.endpoint.DATASET_RELATIVE_PATH
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"\n".join(rows) + b"\n")
    signed_rows: dict[int, Mapping[str, Any]] = {
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
            source: {
                "source_index": source,
                "donor_source_index": source,
            }
            for source in causal
        }
    )
    split_payload = {
        "fit_sources": fit,
        "mechanics_sources": mechanics,
        "causal_sources": causal,
        "excluded_prior_heldout_sources": prior_excluded,
        "excluded_plat_heldout_sources": plat_excluded,
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

    monkeypatch.setattr(
        screen.endpoint,
        "DATASET_SHA256",
        screen.sha256_file(dataset_path),
    )
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


def test_first_query_index_is_immediately_before_first_nonignored_label() -> None:
    labels = torch.tensor([[-100, -100, -100, 17, 18]])

    assert screen.first_prompt_boundary(labels) == (3, 2)

    with pytest.raises(RuntimeError, match="first-label-minus-one"):
        screen.first_prompt_boundary(torch.tensor([[17, 18]]))
    with pytest.raises(ValueError, match="no supervised label"):
        screen.first_prompt_boundary(torch.full((1, 4), -100))


def test_only_four_locked_anchors_and_paired_maps_are_bias_free(
    signed_contract,
) -> None:
    protocol = signed_contract[0]
    code_maps = screen.PairedAnchorCodeMaps()
    projections = tuple(code_maps.write_maps) + tuple(code_maps.query_maps)

    assert screen.ANCHORS == (10, 21, 31, 41)
    assert protocol["exact_v5_capture"]["anchors"] == [10, 21, 31, 41]
    assert protocol["code_alignment_fit"]["anchors"] == [10, 21, 31, 41]
    assert len(code_maps.write_maps) == len(code_maps.query_maps) == 4
    assert all(projection.bias is None for projection in projections)
    assert all(
        tuple(projection.weight.shape) == (screen.CODEBOOK_SIZE, screen.STATE_WIDTH)
        for projection in projections
    )
    assert sum(parameter.numel() for parameter in code_maps.parameters()) == 16384


class _SyntheticReadBasis(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer_idx = layer
        self.state_read_dim = screen.STATE_WIDTH
        self.live_receptance = torch.arange(
            4 * screen.STATE_WIDTH, dtype=torch.float32
        ).reshape(1, 4, 4, 8) + float(layer)

    def _rwkv_ms_token_state_read_basis(
        self,
        state: torch.Tensor,
        memory_source_seq: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del token_mask
        return self.live_receptance, state, memory_source_seq


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


def test_selected_stored_projected_key_cannot_change_live_read_query(
    monkeypatch,
) -> None:
    model, modules, selected_stored_projected_key = _configure_synthetic_capture(
        monkeypatch
    )

    def native_read(*args, **kwargs) -> torch.Tensor:
        del args, kwargs
        for _, module in modules:
            assert module.rwkv_plmsc_prompt_boundary_predictor_index == 1
            expected_boundary = module.live_receptance[:, 1].flatten(1).clone()
            module._rwkv_ms_token_state_read_basis(
                torch.ones(1), torch.ones(1), None
            )
            captured = module.rwkv_plmsc_prompt_boundary_r_seq
            assert tuple(captured.shape) == (1, 32)
            assert torch.equal(captured, expected_boundary)
            module.live_receptance[:, 1] += 100000.0
            assert torch.equal(captured, expected_boundary)
            module.live_receptance[:, 1] -= 100000.0
        return torch.ones(1)

    monkeypatch.setattr(screen.evolution, "_native_read", native_read)

    first = screen.capture_row(
        model, object(), pad_token_id=0, device=torch.device("cpu")
    )
    selected_stored_projected_key["value"] = (
        torch.arange(screen.STATE_WIDTH, dtype=torch.float32).reshape(1, 1, -1)
        + 3.0
    )
    second = screen.capture_row(
        model, object(), pad_token_id=0, device=torch.device("cpu")
    )
    for _, module in modules:
        module.live_receptance[:, 2:] += 100000.0
    third = screen.capture_row(
        model, object(), pad_token_id=0, device=torch.device("cpu")
    )

    assert first["write_slot_address"] != second["write_slot_address"]
    assert (
        first["prompt_boundary_rwkv_receptance"]
        == second["prompt_boundary_rwkv_receptance"]
    )
    assert (
        second["prompt_boundary_rwkv_receptance"]
        == third["prompt_boundary_rwkv_receptance"]
    )
    for feature in (first, second, third):
        assert feature["prompt_boundary_predictor_index"] == 1
        assert feature["read_basis_calls_per_anchor"] == [1, 1, 1, 1]
        assert feature["write_passes"] == 1
        assert feature["read_passes"] == 1
        assert "write_calls" not in feature
    for _, module in modules:
        assert module.rwkv_plmsc_prompt_boundary_predictor_index is None
        assert module.rwkv_plmsc_prompt_boundary_r_seq is None
        assert module.rwkv_plmsc_read_basis_calls == 0


def test_double_read_basis_invocation_fails_closed(monkeypatch) -> None:
    model, modules, _ = _configure_synthetic_capture(monkeypatch)

    def double_read(*args, **kwargs) -> torch.Tensor:
        del args, kwargs
        for _, module in modules:
            module._rwkv_ms_token_state_read_basis(
                torch.ones(1), torch.ones(1), None
            )
            module._rwkv_ms_token_state_read_basis(
                torch.ones(1), torch.ones(1), None
            )
        return torch.ones(1)

    monkeypatch.setattr(screen.evolution, "_native_read", double_read)

    with pytest.raises(RuntimeError, match="exactly once"):
        screen.capture_row(
            model, object(), pad_token_id=0, device=torch.device("cpu")
        )
    for _, module in modules:
        assert module.rwkv_plmsc_prompt_boundary_predictor_index is None
        assert module.rwkv_plmsc_prompt_boundary_r_seq is None
        assert module.rwkv_plmsc_read_basis_calls == 0


def _valid_feature_row() -> tuple[
    Mapping[str, Any],
    Mapping[int, str],
    Mapping[int, Mapping[str, Any]],
]:
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
        "prompt_boundary_rwkv_receptance": torch.ones(4, 32).tolist(),
        "first_supervised_label_index": 2,
        "prompt_boundary_predictor_index": 1,
        "predictor_definition": "first_supervised_label_index_minus_one",
        "predictor_vectors_per_row": 1,
        "answer_or_later_predictor_features_captured": False,
        "write_passes": 1,
        "read_passes": 1,
        "read_basis_calls_per_anchor": [1, 1, 1, 1],
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


def test_feature_schema_rejects_unobserved_write_call_claim() -> None:
    row, split, signed_rows = _valid_feature_row()

    screen._validate_feature_row(row, split, signed_rows)
    mutated = dict(row)
    mutated.pop("receipt")
    mutated["write_calls"] = 1
    mutated = screen._signed_feature_row(mutated)

    with pytest.raises(ValueError, match="feature row differs"):
        screen._validate_feature_row(mutated, split, signed_rows)


def _feature_record(
    source: int,
    donor: int,
    split: str,
    *,
    offset: float,
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


def test_mechanics_mutations_cannot_change_fit_maps(monkeypatch) -> None:
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
        for field in (
            "write_slot_address",
            "prompt_boundary_rwkv_receptance",
        ):
            value = torch.tensor(row[field], dtype=torch.float32)
            row[field] = value.mul(-997.0).add(113.0).tolist()

    first_maps, first_audit = screen.train_fit_only(records)
    second_maps, second_audit = screen.train_fit_only(mutated)

    assert first_audit == second_audit
    assert first_audit["mechanics_or_causal_rows_used"] is False
    assert first_audit["thresholds"] is None
    for name, value in first_maps.state_dict().items():
        assert torch.equal(value, second_maps.state_dict()[name])


def _branch_metric(**overrides: Any) -> Mapping[str, Any]:
    return {
        "anchor_match_or_collision_count": 0,
        "anchor_match_or_collision_fraction": 0.95,
        "complete_row_match_or_collision_count": 0,
        "complete_row_match_or_collision_fraction": 0.95,
        "mean_probability_dot_affinity": 0.1,
        "per_anchor_match_or_collision_fraction": [0.0] * 4,
        "finite": True,
        **overrides,
    }


def _usage(
    *, distinct: int = 8, maximum_fraction: float = 0.25
) -> list[Mapping[str, Any]]:
    return [
        {
            "layer": layer,
            "distinct_codes": distinct,
            "maximum_single_code_count": 8,
            "maximum_single_code_fraction": maximum_fraction,
        }
        for layer in screen.ANCHORS
    ]


class _ZeroCodeMaps:
    def write_logits(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(value.size(0), len(screen.ANCHORS), screen.CODEBOOK_SIZE)

    def query_logits(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(value.size(0), len(screen.ANCHORS), screen.CODEBOOK_SIZE)


def _synthetic_mechanics_gate(
    monkeypatch,
    *,
    correct_overrides: Mapping[str, Any] | None = None,
    donor_overrides: Mapping[str, Any] | None = None,
    permuted_overrides: Mapping[str, Any] | None = None,
    usage_distinct: int = 8,
    usage_maximum_fraction: float = 0.25,
    finite: bool = True,
) -> Mapping[str, Any]:
    rows = screen.MECHANICS_ROWS
    feature = torch.ones(rows, len(screen.ANCHORS), screen.STATE_WIDTH)
    monkeypatch.setattr(
        screen,
        "_record_tensors",
        lambda records, split: ([], feature, feature),
    )
    monkeypatch.setattr(screen, "_donor_indices", lambda _: torch.arange(rows))
    donor_row_fraction = screen.DONOR_ROW_COLLISION_COUNT_GATE / rows
    permuted_row_fraction = screen.LAYER_PERMUTED_ROW_COLLISION_COUNT_GATE / rows
    donor_values = {
        "anchor_match_or_collision_fraction": 0.03,
        "complete_row_match_or_collision_count": 1,
        "complete_row_match_or_collision_fraction": donor_row_fraction,
        **dict(donor_overrides or {}),
    }
    permuted_values = {
        "anchor_match_or_collision_fraction": 0.03,
        "complete_row_match_or_collision_count": 1,
        "complete_row_match_or_collision_fraction": permuted_row_fraction,
        **dict(permuted_overrides or {}),
    }
    metrics = iter(
        (
            _branch_metric(**dict(correct_overrides or {})),
            _branch_metric(**donor_values),
            _branch_metric(**permuted_values),
        )
    )
    usages = iter(
        (
            _usage(
                distinct=usage_distinct,
                maximum_fraction=usage_maximum_fraction,
            ),
            _usage(
                distinct=usage_distinct,
                maximum_fraction=usage_maximum_fraction,
            ),
        )
    )
    monkeypatch.setattr(screen, "_branch_metrics", lambda *args: next(metrics))
    monkeypatch.setattr(screen, "_usage_metrics", lambda *args: next(usages))
    monkeypatch.setattr(screen, "_finite_tree", lambda _: finite)
    return screen.evaluate_mechanics_once(_ZeroCodeMaps(), [])


def test_locked_mechanics_gates_pass_exactly_at_boundaries(monkeypatch) -> None:
    result = _synthetic_mechanics_gate(monkeypatch)

    assert screen.MINIMUM_DISTINCT_CODES == 8
    assert screen.MAXIMUM_CODE_FRACTION == 0.25
    assert result["passed"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("failed_check", "kwargs"),
    (
        (
            "correct_anchor_match_fraction",
            {"correct_overrides": {"anchor_match_or_collision_fraction": 0.949}},
        ),
        (
            "correct_complete_row_match_fraction",
            {
                "correct_overrides": {
                    "complete_row_match_or_collision_fraction": 0.949
                }
            },
        ),
        (
            "donor_anchor_collision_fraction",
            {"donor_overrides": {"anchor_match_or_collision_fraction": 0.031}},
        ),
        (
            "donor_complete_row_collision_count",
            {"donor_overrides": {"complete_row_match_or_collision_count": 2}},
        ),
        (
            "donor_complete_row_collision_fraction",
            {
                "donor_overrides": {
                    "complete_row_match_or_collision_fraction": 1 / 34 + 1e-6
                }
            },
        ),
        (
            "layer_permuted_anchor_collision_fraction",
            {
                "permuted_overrides": {
                    "anchor_match_or_collision_fraction": 0.031
                }
            },
        ),
        (
            "layer_permuted_complete_row_collision_count",
            {
                "permuted_overrides": {
                    "complete_row_match_or_collision_count": 2
                }
            },
        ),
        (
            "layer_permuted_complete_row_collision_fraction",
            {
                "permuted_overrides": {
                    "complete_row_match_or_collision_fraction": 1 / 34 + 1e-6
                }
            },
        ),
        (
            "all_logits_probabilities_losses_and_metrics_finite",
            {"finite": False},
        ),
        (
            "correct_write_and_query_codes_noncollapsed",
            {"usage_distinct": 7},
        ),
        (
            "correct_write_and_query_codes_noncollapsed",
            {"usage_maximum_fraction": 0.251},
        ),
    ),
)
def test_each_locked_mechanics_gate_fails_closed_past_boundary(
    monkeypatch,
    failed_check: str,
    kwargs: Mapping[str, Any],
) -> None:
    result = _synthetic_mechanics_gate(monkeypatch, **kwargs)

    assert result["passed"] is False
    assert result["checks"][failed_check] is False


def test_local_and_remote_rank_errors_reach_consensus_before_return(
    monkeypatch,
) -> None:
    context = object()
    observed: list[tuple[str, BaseException | None]] = []

    def record_consensus(
        actual_context: Any,
        *,
        phase: str,
        error: BaseException | None,
    ) -> None:
        assert actual_context is context
        observed.append((phase, error))

    monkeypatch.setattr(screen.distributed, "phase_consensus", record_consensus)
    with pytest.raises(ValueError, match="malformed local shard"):
        screen._consensual_operation(
            context,
            phase="feature-validation",
            operation=lambda: (_ for _ in ()).throw(
                ValueError("malformed local shard")
            ),
        )
    assert len(observed) == 1
    assert observed[0][0] == "feature-validation"
    assert isinstance(observed[0][1], ValueError)

    later_collective_reached = False

    def remote_failure(
        actual_context: Any,
        *,
        phase: str,
        error: BaseException | None,
    ) -> None:
        assert actual_context is context
        assert phase == "remote-validation"
        assert error is None
        raise RuntimeError("rank 2: malformed remote shard")

    monkeypatch.setattr(screen.distributed, "phase_consensus", remote_failure)
    with pytest.raises(RuntimeError, match="rank 2: malformed remote shard"):
        screen._consensual_operation(
            context,
            phase="remote-validation",
            operation=lambda: "locally valid",
        )
        later_collective_reached = True
    assert later_collective_reached is False


def test_execution_uses_nccl_model_data_and_gloo_control(signed_contract) -> None:
    execution = signed_contract[0]["execution"]

    assert execution["world_size"] == 4
    assert execution["model_backend"] == "nccl"
    assert execution["control_backend"] == "gloo"
    assert execution["timeout_seconds"] == 1800
    assert execution["four_rank_failures_reach_consensus_before_later_collectives"]


def test_protocol_and_runner_forbid_training_stage2_generation_and_saves(
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
    assert "monomial transform" in protocol["code_alignment_fit"][
        "posthoc_monomial_transform"
    ]
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
        }
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"stage2", "monomial_transform"}
        for node in ast.walk(tree)
    )
