from __future__ import annotations

from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (
    screen_natural_memory_split_seeds as screen,
)


def test_screen_selects_first_feasible_seeds_without_generation(
    monkeypatch,
) -> None:
    items = [object()]
    monkeypatch.setattr(
        screen,
        "component_inputs",
        lambda paths: (
            items,
            {"component": ["row"]},
            {"component": {"scene": 1}},
            {
                "sources": [{"task": "scene", "sha256": "a" * 64}],
                "row_count": 1,
                "component_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        screen.source,
        "assign_component_splits",
        lambda component_rows, component_task_weights, *, seed: {"component": seed},
    )
    errors = {10: 0.2, 11: 0.03, 12: 0.01, 13: 0.02}
    monkeypatch.setattr(
        screen.source,
        "_split_audit",
        lambda current_items, assignments: {
            "maximum_item_fraction_abs_error": errors[assignments["component"]]
        },
    )
    monkeypatch.setattr(screen.source, "sha256_file", lambda path: "b" * 64)

    payload = screen.screen_seeds(
        paths={"scene": Path("train.jsonl")},
        start_seed=10,
        count=4,
        selected_count=2,
    )

    assert payload["selected_seeds"] == [11, 12]
    assert payload["screening_scope"]["episodes_generated"] == 0
    assert payload["screening_scope"]["sealed_rows_generated"] == 0
    assert payload["screened"][0]["feasible"] is False
    assert payload["receipt"]["payload_sha256"] == screen.sha256_text(
        screen.canonical_json({key: value for key, value in payload.items() if key != "receipt"})
    )
