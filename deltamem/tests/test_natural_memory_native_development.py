from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_natural_memory_native_development as native,
)


def test_evolution_protocol_receipt_is_bound() -> None:
    protocol = native.load_evolution_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        native.EVOLUTION_PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["native_data_contract"]["split_seed"] == native.SPLIT_SEED


def test_native_component_split_is_deterministic_atomic_and_balanced() -> None:
    component_rows = {
        f"component-{index}": (
            f"row-{index}-a",
            f"row-{index}-b",
        )
        for index in range(30)
    }
    row_task = {
        row_id: ("attribution", "narrative", "scene")[index % 3]
        for index, rows in enumerate(component_rows.values())
        for row_id in rows
    }

    first = native.assign_native_component_splits(component_rows, row_task)
    second = native.assign_native_component_splits(component_rows, row_task)

    assert first == second
    assert set(first) == set(component_rows)
    assert set(first.values()) == {"fit", "development"}
    for task in set(row_task.values()):
        total = sum(value == task for value in row_task.values())
        fit = sum(
            row_task[row_id] == task and first[component] == "fit"
            for component, rows in component_rows.items()
            for row_id in rows
        )
        assert abs(fit / total - native.FIT_FRACTION) <= 0.1
