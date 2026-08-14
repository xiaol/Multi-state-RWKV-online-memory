from __future__ import annotations

from dataclasses import replace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)


def make_example(
    ordinal: int,
    *,
    write_tokens: int,
    row_sha256: str,
) -> evolution.NativeFullRowExample:
    return evolution.NativeFullRowExample(
        row_id=f"row-{ordinal}",
        task="scene",
        source_ordinal=ordinal,
        row_sha256=row_sha256,
        write_input_ids=tuple(range(write_tokens)),
        write_attention_mask=(1,) * write_tokens,
        read_input_ids=(1, 2, 3),
        read_attention_mask=(1, 1, 1),
        labels=(-100, -100, 3),
        assistant_target_tokens=1,
    )


def test_protocol_receipt_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False
    assert protocol["intervention"]["checkpoint_updates"] == [8, 16, 32]


def test_donor_mapping_prefers_length_then_hash() -> None:
    rows = [
        runner.SceneContrastRow(make_example(0, write_tokens=10, row_sha256="c" * 64), "a"),
        runner.SceneContrastRow(make_example(1, write_tokens=11, row_sha256="b" * 64), "b"),
        runner.SceneContrastRow(make_example(2, write_tokens=9, row_sha256="a" * 64), "c"),
        runner.SceneContrastRow(make_example(3, write_tokens=10, row_sha256="d" * 64), "a"),
    ]

    mapping, deltas, payload = runner.build_donor_mapping(rows)

    assert mapping[0] == 2
    assert deltas[0] == 1
    assert payload[0]["donor_row_sha256"] == "a" * 64
    assert mapping[3] == 2


def test_schedule_has_fixed_dropout_balance(monkeypatch) -> None:
    rows = [
        runner.SceneContrastRow(
            make_example(index, write_tokens=10, row_sha256=f"{index:064x}"),
            str(index),
        )
        for index in range(256)
    ]
    mapping = {index: (index + 1) % len(rows) for index in range(len(rows))}
    deltas = {index: 0 for index in range(len(rows))}
    selected = sorted(
        range(len(rows)),
        key=lambda index: (
            runner.hashlib.sha256(
                (runner.TRAIN_SALT + rows[index].example.row_sha256).encode("utf-8")
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
                    runner.hashlib.sha256(
                        (
                            f"{runner.DROP_SALT}{step}:"
                            + rows[index].example.row_sha256
                        ).encode("utf-8")
                    ).hexdigest(),
                    index,
                ),
            )[:2]
        )
        payload.append(
            {
                "step": step,
                "rows": [
                    {
                        "source_ordinal": index,
                        "source_row_sha256": rows[index].example.row_sha256,
                        "donor_ordinal": mapping[index],
                        "positive_condition": (
                            "no_state" if index in no_state else "correct_state"
                        ),
                    }
                    for index in group
                ],
            }
        )
    monkeypatch.setattr(runner, "EXPECTED_ELIGIBLE_ROWS", len(rows))
    monkeypatch.setattr(
        runner,
        "SELECTED_ROWS_SHA256",
        runner.canonical_sha256([rows[index].example.row_sha256 for index in selected]),
    )
    monkeypatch.setattr(runner, "FULL_SCHEDULE_SHA256", runner.canonical_sha256(payload))

    schedule, actual_payload = runner.build_schedule(rows, mapping, deltas)

    assert actual_payload == payload
    assert len(schedule) == 32
    assert all(len(step.no_state_ordinals) == 2 for step in schedule)
    assert all(len(step.source_ordinals) == 8 for step in schedule)


def test_gate_only_training_freezes_every_other_parameter() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for _ in range(42):
                layer = torch.nn.Module()
                layer.memory_fusion_hidden_weight = torch.nn.Parameter(torch.ones(1))
                layer.memory_fusion_read_weight = torch.nn.Parameter(torch.ones(1))
                layer.memory_fusion_bias = torch.nn.Parameter(torch.ones(1))
                layer.other = torch.nn.Parameter(torch.ones(1))
                self.layers.append(layer)

    model = Model()
    selected, audit = runner.configure_gate_only_training(model)

    assert audit["passed"] is True
    assert len(selected) == 126
    assert all(parameter.requires_grad for _, parameter in selected)
    assert all(not layer.other.requires_grad for layer in model.layers)


def test_donor_batch_keeps_target_read_and_uses_donor_write() -> None:
    target = make_example(0, write_tokens=2, row_sha256="a" * 64)
    donor = replace(
        make_example(1, write_tokens=3, row_sha256="b" * 64),
        write_input_ids=(7, 8, 9),
    )
    target_batch = evolution.collate_native_examples(
        [target],
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    donor_batch = runner.build_donor_batch(
        target_batch,
        donor,
        device=torch.device("cpu"),
    )

    assert donor_batch.write_input_ids.tolist() == [[7, 8, 9]]
    assert torch.equal(donor_batch.read_input_ids, target_batch.read_input_ids)
    assert torch.equal(donor_batch.labels, target_batch.labels)


def test_detached_answer_ce_matches_cross_entropy() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    labels = torch.tensor([[-100, 0, 1]])

    mean, count = runner.detached_answer_ce(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits.view(-1, 2),
        torch.tensor([0, 1]),
    )

    assert count == 2
    assert mean == expected.item()
