from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_continuous_write_causal_train as training,
)


def test_signed_source_bootstrap_precedes_delta_import() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")

    assert source.index(
        "run_natural_memory_native_rwkv_continuous_write_mechanics as mechanics"
    ) < source.index("from deltamem.core.delta import reset_delta_mem_states")
    if training.mechanics.SIGNED_SOURCE_ROOT is not None:
        assert Path(training.mechanics.core_impl.__file__).resolve().is_relative_to(
            training.mechanics.SIGNED_SOURCE_ROOT
        )


def _fit_rows() -> list[dict[str, int]]:
    rows = []
    for source in range(training.FIT_ROWS):
        donor = source + training.FIT_PAIRS if source < training.FIT_PAIRS else source - training.FIT_PAIRS
        rows.append({"source_index": source, "donor_source_index": donor})
    return rows


def test_pair_schedule_uses_every_symmetric_pair_once() -> None:
    schedule = training.build_pair_schedule(_fit_rows())

    assert len(schedule) == training.UPDATES
    assert all(len(step.rank_pairs) == training.WORLD_SIZE for step in schedule)
    pairs = [pair for step in schedule for pair in step.rank_pairs]
    assert len(pairs) == training.FIT_PAIRS
    assert len(set(pairs)) == training.FIT_PAIRS
    assert {source for pair in pairs for source in pair} == set(range(training.FIT_ROWS))


def test_pair_schedule_rejects_non_symmetric_donor_mapping() -> None:
    rows = _fit_rows()
    rows[0]["donor_source_index"] = 1

    with pytest.raises(ValueError, match="not symmetric"):
        training.build_pair_schedule(rows)


def test_smooth_hinge_wrong_coefficient_is_bounded_and_directional() -> None:
    violated = training.smooth_hinge_wrong_coefficient(
        correct_ce=2.0,
        wrong_ce=1.9,
        margin=0.02,
        weight=0.5,
    )
    satisfied = training.smooth_hinge_wrong_coefficient(
        correct_ce=2.0,
        wrong_ce=2.5,
        margin=0.02,
        weight=0.5,
    )

    assert -0.5 <= violated < satisfied < 0.0
    assert abs(violated) > 0.45
    assert abs(satisfied) < 0.001


def test_correct_branch_coefficient_includes_every_hinge_derivative() -> None:
    branch_ce = {
        "correct": 2.0,
        "paired_donor_recurrent": 1.9,
        "layer_rolled_recurrent": 2.1,
        "zero_recurrent": 2.2,
    }

    coefficients = training.serialized_branch_coefficients(branch_ce)

    assert coefficients["correct"] == pytest.approx(
        1.0
        - coefficients["paired_donor_recurrent"]
        - coefficients["layer_rolled_recurrent"]
        - coefficients["zero_recurrent"]
    )


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = torch.nn.Linear(2, 2, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hrm_rwkv7_core = _Core()
        self.delta_o_proj = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
        self.writer = torch.nn.Parameter(torch.ones(2, 2))
        self.receptance = torch.nn.Parameter(torch.ones(2, 2))
        self.continuous_map = torch.nn.Parameter(torch.ones(2, 2))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Layer() for _ in range(training.EXPECTED_TENSORS_PER_SUFFIX)]
        )


def test_trainable_isolation_selects_only_read_path_tensors() -> None:
    model = _Model()

    selected, audit = training.configure_trainable_parameters(model)

    assert audit["passed"] is True
    assert len(selected) == training.EXPECTED_TRAINABLE_TENSORS
    assert all(
        name.endswith(training.TRAINABLE_SUFFIXES) for name, _ in selected
    )
    assert all(parameter.dtype == torch.float32 for _, parameter in selected)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.endswith(training.TRAINABLE_SUFFIXES)
    )


def test_serialized_branch_accumulator_requires_complete_gradients() -> None:
    model = _Model()
    selected, _ = training.configure_trainable_parameters(model)
    for _, parameter in selected:
        parameter.grad = torch.ones_like(parameter)
    accumulator: dict[str, torch.Tensor] = {}

    training._accumulate_reduced_branch(selected, accumulator)
    training._accumulate_reduced_branch(selected, accumulator)
    training._install_accumulated_gradients(selected, accumulator)

    assert all(torch.equal(parameter.grad, torch.full_like(parameter, 2.0)) for _, parameter in selected)


def test_fit_loader_has_no_generic_or_causal_materialization_path() -> None:
    source = inspect.getsource(training._load_fit_rows)

    assert "validate_materialization" not in source
    assert "allow_causal" not in source
    assert '"fit"' in source
    assert '"causal"' not in source


def test_unsigned_protocol_fails_before_file_access(monkeypatch) -> None:
    monkeypatch.setattr(training, "PROTOCOL_PAYLOAD_SHA256", "TO_BE_FILLED")
    monkeypatch.setattr(training, "PROTOCOL_FILE_SHA256", "TO_BE_FILLED")

    with pytest.raises(RuntimeError, match="not signed"):
        training.validate_protocol()


def _causal_rows() -> list[dict[str, object]]:
    rows = []
    for source in range(training.CAUSAL_ROWS):
        donor = source + 16 if source < 16 else source - 16
        rows.append(
            {
                "source_index": source,
                "donor_source_index": donor,
                "row_sha256": f"row-{source}",
                "donor_row_sha256": f"row-{donor}",
            }
        )
    return rows


def test_causal_assignment_keeps_four_complete_pairs_per_rank() -> None:
    assignment = training.build_causal_pair_assignment(_causal_rows())

    assert len(assignment) == training.WORLD_SIZE
    assert all(len(rank_pairs) == 4 for rank_pairs in assignment)
    assert all(len(training.causal_sources_for_rank(assignment, rank)) == 8 for rank in range(4))
    assert {
        source for rank_pairs in assignment for pair in rank_pairs for source in pair
    } == set(range(training.CAUSAL_ROWS))
    assert "_capture_natural_snapshot" not in inspect.getsource(training.evaluate_causal_row)


def _analyzer_row(source: int, *, margin: float, tokens: int = 1) -> dict[str, object]:
    correct_ce = 1.0
    conditions = {
        "correct": correct_ce,
        "paired_donor_recurrent": correct_ce + margin,
        "layer_rolled_recurrent": correct_ce + margin,
        "zero_recurrent": correct_ce + margin,
        "projected_only": correct_ce + margin,
    }
    return {
        "source_index": source,
        "all_condition_ce_finite": True,
        "projected_carrier_fixed_all_conditions": True,
        "zero_recurrent_logits_byte_equal_projected_only": True,
        "answer_token_count_identical": True,
        "condition_loss_sum": {
            name: value * tokens for name, value in conditions.items()
        },
        "condition_token_count": {name: tokens for name in conditions},
        "ce_margins": {
            "paired_donor_recurrent": margin,
            "layer_rolled_recurrent": margin,
            "zero_recurrent": margin,
        },
    }


def test_causal_analysis_uses_token_weighted_answer_ce() -> None:
    rows = [_analyzer_row(source, margin=0.1) for source in range(31)]
    rows.append(_analyzer_row(31, margin=-0.01, tokens=100))

    analysis = training.analyze_causal_rows(rows)

    donor = analysis["aggregate"]["paired_donor_recurrent"]
    assert donor["unweighted_row_mean_ce_margin"] > 0.09
    assert donor["token_weighted_mean_ce_margin"] == pytest.approx(2.1 / 131)
    assert analysis["checks"]["paired_donor_recurrent_token_weighted_mean"] is False


def test_projected_only_exact_equality_is_a_hard_gate() -> None:
    rows = [_analyzer_row(source, margin=0.1) for source in range(32)]
    rows[0]["zero_recurrent_logits_byte_equal_projected_only"] = False

    analysis = training.analyze_causal_rows(rows)

    assert analysis["checks"]["zero_recurrent_equals_projected_only_every_row"] is False
    assert analysis["passed"] is False


def test_training_receipt_signature_gates_causal_loader(tmp_path, monkeypatch) -> None:
    accessed = False

    def forbidden_read(*_args, **_kwargs):
        nonlocal accessed
        accessed = True
        raise AssertionError("causal reader must remain unopened")

    monkeypatch.setattr(training.materializer, "_read_bundle", forbidden_read)
    receipt = {
        "status": "continuous_write_fit_training_frozen_causal_open_authorized",
        "causal_bytes_open_authorized": True,
        "parameters_frozen_before_causal_open": True,
    }

    with pytest.raises(ValueError, match="receipt differs"):
        training._load_causal_rows_after_receipt(tmp_path, {}, receipt)
    assert accessed is False


def test_signed_shards_follow_pair_assignment(tmp_path) -> None:
    causal_rows = _causal_rows()
    assignment = training.build_causal_pair_assignment(causal_rows)
    binding = {"assignment": training.causal_assignment_binding(causal_rows, assignment)}
    for rank in range(training.WORLD_SIZE):
        training._write_causal_shard(
            tmp_path,
            rank=rank,
            rows=[
                {
                    "source_index": source,
                    "donor_source_index": int(causal_rows[source]["donor_source_index"]),
                }
                for source in training.causal_sources_for_rank(assignment, rank)
            ],
            binding=binding,
        )

    evaluated, inventory = training._load_causal_shards(
        tmp_path, binding=binding, assignment=assignment
    )

    assert len(evaluated) == training.CAUSAL_ROWS
    assert len(inventory) == training.WORLD_SIZE

    shard_path = tmp_path / "causal-shard-0.json"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["rows"].reverse()
    unsigned = dict(shard)
    unsigned.pop("receipt")
    shard["receipt"]["payload_sha256"] = training.canonical_sha256(unsigned)
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(ValueError, match="shard contract differs"):
        training._load_causal_shards(tmp_path, binding=binding, assignment=assignment)
