from __future__ import annotations

import hashlib

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_seed_ensemble as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_seed_ensemble as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_seed_ensemble as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)


def make_row(index: int) -> contrast.SceneContrastRow:
    example = evolution.NativeFullRowExample(
        row_id=f"row-{index}",
        task="scene",
        source_ordinal=index,
        row_sha256=f"{index:064x}",
        write_input_ids=(1, 2),
        write_attention_mask=(1, 1),
        read_input_ids=(1, 2, 3),
        read_attention_mask=(1, 1, 1),
        labels=(-100, -100, 3),
        assistant_target_tokens=1,
    )
    return contrast.SceneContrastRow(example=example, assistant_identity=str(index))


def test_seed_ensemble_protocol_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert [item["seed"] for item in protocol["training_data"]["seeds"]] == [17, 29, 43]
    assert protocol["intervention"]["global_batch_size"] == 16
    assert protocol["intervention"]["learning_rate"] == 5e-5
    assert protocol["authorization"]["publisher_validation_replication_authorized"] is False


def test_seed_schedule_has_locked_large_batch_and_dropout(monkeypatch) -> None:
    seed = 17
    rows = [make_row(index) for index in range(256)]
    mapping = {index: (index + 1) % len(rows) for index in range(len(rows))}
    deltas = {index: 0 for index in range(len(rows))}
    train_salt = f"{runner.TRAIN_SALT_PREFIX}{seed}:"
    selected = sorted(
        range(len(rows)),
        key=lambda index: (
            hashlib.sha256(
                (train_salt + rows[index].example.row_sha256).encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    payload = []
    for offset in range(0, len(selected), runner.GLOBAL_BATCH_SIZE):
        step = offset // runner.GLOBAL_BATCH_SIZE + 1
        group = selected[offset : offset + runner.GLOBAL_BATCH_SIZE]
        no_state = set(
            sorted(
                group,
                key=lambda index: (
                    hashlib.sha256(
                        (
                            f"{runner.DROP_SALT_PREFIX}{seed}:{step}:"
                            + rows[index].example.row_sha256
                        ).encode("utf-8")
                    ).hexdigest(),
                    index,
                ),
            )[:4]
        )
        payload.append(
            {
                "step": step,
                "rows": [
                    {
                        "source_ordinal": index,
                        "source_row_sha256": rows[index].example.row_sha256,
                        "donor_ordinal": mapping[index],
                        "positive_condition": "no_state" if index in no_state else "correct_state",
                    }
                    for index in group
                ],
            }
        )
    monkeypatch.setattr(runner, "EXPECTED_ELIGIBLE_ROWS", len(rows))
    monkeypatch.setitem(
        runner.SEED_BINDINGS,
        seed,
        {
            "selected_rows_payload_sha256": runner.canonical_sha256(
                [rows[index].example.row_sha256 for index in selected]
            ),
            "schedule_payload_sha256": runner.canonical_sha256(payload),
        },
    )

    schedule, actual_payload = runner.build_schedule(rows, mapping, deltas, seed=seed)

    assert actual_payload == payload
    assert len(schedule) == 16
    assert all(len(step.source_ordinals) == 16 for step in schedule)
    assert all(len(step.no_state_ordinals) == 4 for step in schedule)


def test_proximal_shrinkage_moves_delta_toward_anchor(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, -1.0]))
    anchor = torch.tensor([1.0, 1.0])
    monkeypatch.setattr(runner, "POST_STEP_DELTA_RETENTION", 0.5)

    audit = runner.apply_proximal_shrinkage([("gate", parameter)], {"gate": anchor})

    assert torch.equal(parameter.detach(), torch.tensor([2.0, 0.0]))
    assert audit["observed_l2_retention"] == 0.5


def test_materialization_averages_signed_seed_deltas() -> None:
    v9 = {"gate": torch.tensor([10.0, 20.0])}
    states = {
        17: {"gate": torch.tensor([11.0, 18.0])},
        29: {"gate": torch.tensor([13.0, 20.0])},
        43: {"gate": torch.tensor([9.0, 25.0])},
    }

    mixed = materializer.mean_seed_delta(v9, states)

    assert torch.allclose(mixed["gate"], torch.tensor([11.0, 21.0]))


def test_analysis_thresholds_match_protocol() -> None:
    protocol = runner.validate_protocol()
    gates = protocol["evaluation"]["gates"]

    assert gates["coverage_minimum"] == analyzer.GATE_THRESHOLDS["coverage"]
    assert (
        gates["candidate_minus_checkpoint_16_micro_f1_minimum"]
        == analyzer.GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"]
    )
