from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import diagnose_residual_hybrid_scales as diagnostic


class FakeModule:
    memory_backend = "rwkv_ms"
    active_delta_heads = frozenset({"o"})

    def __init__(self, layer_idx: int, raw_gain: float) -> None:
        self.layer_idx = layer_idx
        self.memory_fusion_placement = "post_attention_residual_hybrid"
        self.memory_fusion_mode = "content_gated_add"
        self.memory_fusion_residual_scale = 0.01
        self.memory_fusion_residual_scale_max = 0.02
        self.memory_fusion_residual_gain_raw = torch.nn.Parameter(
            torch.tensor([raw_gain], dtype=torch.float32)
        )
        self._post_attention_norm_hook_handle = object()

    def set_memory_fusion_residual_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= self.memory_fusion_residual_scale_max:
            raise ValueError("gain outside cap")
        with torch.no_grad():
            self.memory_fusion_residual_gain_raw.fill_(gain)

    def _resolved_memory_fusion_residual_gain(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.memory_fusion_residual_gain_raw.detach().clamp(
            0.0, self.memory_fusion_residual_scale_max
        ).to(device=device, dtype=dtype)[0]


def make_modules() -> list[tuple[str, FakeModule]]:
    return [
        ("model.layers.0.self_attn", FakeModule(0, -0.00282517)),
        ("model.layers.1.self_attn", FakeModule(1, 0.0175)),
    ]


def test_parse_condition_names_deduplicates_and_validates() -> None:
    assert diagnostic.parse_condition_names(
        "native, native-gate-open,native"
    ) == ["native", "native_gate_open"]
    with pytest.raises(ValueError, match="Unsupported condition"):
        diagnostic.parse_condition_names("attention_output")


def test_initial_screen_contains_the_bounded_five_conditions() -> None:
    conditions = diagnostic.initial_condition_screen(
        list(diagnostic.SUPPORTED_CONDITIONS)
    )

    assert [condition["name"] for condition in conditions] == list(
        diagnostic.SUPPORTED_CONDITIONS
    )
    assert conditions[0] == {
        "name": "native",
        "description": "Checkpoint-native placement, content gate, and learned gains.",
    }
    assert conditions[1]["memory_fusion_mode"] == "add"
    assert conditions[2]["memory_fusion_residual_gain"] == 0.0
    assert conditions[3]["memory_fusion_residual_gain"] == 0.01
    assert conditions[4]["memory_fusion_placement"] == "post_attention_norm"


def test_restore_fusion_settings_restores_exact_raw_learned_gains() -> None:
    modules = make_modules()
    saved = diagnostic.capture_fusion_settings(modules)

    for _, module in modules:
        module.memory_fusion_placement = "post_attention_norm"
        module.memory_fusion_mode = "add"
        module.memory_fusion_residual_scale = 0.5
        module.memory_fusion_residual_scale_max = 1.0
        with torch.no_grad():
            module.memory_fusion_residual_gain_raw.fill_(0.75)

    diagnostic.restore_fusion_settings(modules, saved)

    for name, module in modules:
        state = saved[name]
        assert module.memory_fusion_placement == state["memory_fusion_placement"]
        assert module.memory_fusion_mode == state["memory_fusion_mode"]
        assert (
            module.memory_fusion_residual_scale
            == state["memory_fusion_residual_scale"]
        )
        assert (
            module.memory_fusion_residual_scale_max
            == state["memory_fusion_residual_scale_max"]
        )
        assert torch.equal(
            module.memory_fusion_residual_gain_raw.detach(),
            state["memory_fusion_residual_gain_raw"],
        )
    assert modules[0][1].memory_fusion_residual_gain_raw.item() < 0.0


def test_snapshot_bank_sha256_is_canonical_and_content_sensitive() -> None:
    first = [
        {
            "b": torch.tensor([[2.0]], dtype=torch.bfloat16),
            "a": torch.tensor([1], dtype=torch.long),
        }
    ]
    reordered = [{"a": first[0]["a"].clone(), "b": first[0]["b"].clone()}]
    changed = [{"a": torch.tensor([3]), "b": first[0]["b"].clone()}]

    assert diagnostic.snapshot_bank_sha256(first) == diagnostic.snapshot_bank_sha256(
        reordered
    )
    assert diagnostic.snapshot_bank_sha256(first) != diagnostic.snapshot_bank_sha256(
        changed
    )


def make_empty_pairing_manifest() -> dict[str, Any]:
    split = {
        "split": "train",
        "pairing_version": "post_split_half_rotation_v1",
        "sample_count": 0,
        "rotation": 0,
        "target_mode": "first_differing_supervised_target_span_v1",
        "target_span_tokens": 8,
        "target_token_count": 0,
        "source_fingerprint": "fingerprint",
        "paired_fingerprint": "paired",
        "pairs_sha256": diagnostic.canonical_json_sha256([]),
        "pairs": [],
    }
    split["manifest_sha256"] = diagnostic.canonical_json_sha256(split)
    manifest = {
        "schema_version": 1,
        "objective_version": "content_contrast_ce_v6",
        "pairing_version": "post_split_half_rotation_v1",
        "pairing_scope": "within_post_split_partition",
        "target_mode": "first_differing_supervised_target_span_v1",
        "target_span_tokens": 8,
        "target_token_count": 0,
        "data_seed": 42,
        "tokenized_fingerprint": "fingerprint",
        "splits": {"train": split},
    }
    manifest["manifest_sha256"] = diagnostic.canonical_json_sha256(manifest)
    return manifest


def test_pairing_integrity_rejects_a_tampered_self_hash(tmp_path: Path) -> None:
    manifest = make_empty_pairing_manifest()
    protocol = {
        "tokenized_fingerprint": "fingerprint",
        "content_contrast_pairing": diagnostic.pairing_protocol_summary(manifest),
    }
    manifest["data_seed"] = 43
    (tmp_path / "content_contrast_pairing_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top-level self hash"):
        diagnostic.validate_pairing_manifest_integrity(
            checkpoint=tmp_path,
            protocol=protocol,
            tokenized=[],
            split_name="train",
            donors=[],
        )


def test_pairing_integrity_rejects_protocol_summary_mismatch(tmp_path: Path) -> None:
    manifest = make_empty_pairing_manifest()
    protocol_summary = diagnostic.pairing_protocol_summary(manifest)
    protocol_summary["data_seed"] = 99
    (tmp_path / "content_contrast_pairing_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match training_protocol"):
        diagnostic.validate_pairing_manifest_integrity(
            checkpoint=tmp_path,
            protocol={
                "tokenized_fingerprint": "fingerprint",
                "content_contrast_pairing": protocol_summary,
            },
            tokenized=[],
            split_name="train",
            donors=[],
        )


def test_condition_sweep_freshly_primes_and_accounts_for_all_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = make_modules()
    native = diagnostic.capture_fusion_settings(modules)
    tokenized = [{"row_index": index} for index in range(32)]
    condition_order = [
        "gate_open_gamma_0",
        "native",
        "native_gate_open",
        "gate_open_gamma_0p01",
        "post_attention_norm_gate_open_0p01",
    ]
    conditions = diagnostic.initial_condition_screen(condition_order)
    prime_observations: list[dict[str, Any]] = []
    evaluated_bank_ids: list[int] = []
    reset_calls = 0

    def fake_reset(model, *, write_enabled: bool) -> None:
        nonlocal reset_calls
        assert write_enabled is True
        reset_calls += 1

    def fake_prime(*, model, tokenized, modules, device: str):
        call_index = len(prime_observations)
        prime_observations.append(
            {
                "mode": [module.memory_fusion_mode for _, module in modules],
                "placement": [
                    module.memory_fusion_placement for _, module in modules
                ],
                "raw": [
                    module.memory_fusion_residual_gain_raw.detach().clone()
                    for _, module in modules
                ],
            }
        )
        return [
            {"sentinel": torch.tensor([call_index, row_index])}
            for row_index in range(len(tokenized))
        ]

    def fake_evaluate(*, tokenized, snapshots, **kwargs):
        evaluated_bank_ids.append(id(snapshots))
        row_count = len(tokenized)
        return (
            [],
            {},
            {
                "correct_replay_count": row_count,
                "donor_replay_count": row_count,
                "replay_count": row_count * 2,
            },
        )

    monkeypatch.setattr(diagnostic, "reset_runtime", fake_reset)
    monkeypatch.setattr(diagnostic, "prime_writer_snapshots", fake_prime)
    monkeypatch.setattr(diagnostic, "evaluate_condition", fake_evaluate)

    results, totals = diagnostic.run_condition_sweep(
        model=object(),
        modules=modules,
        native_settings=native,
        conditions=conditions,
        tokenized=tokenized,
        source_rows=[{} for _ in tokenized],
        donors=list(reversed(range(len(tokenized)))),
        device="cpu",
        primary_span_tokens=8,
    )

    assert len(prime_observations) == 5
    assert len(set(evaluated_bank_ids)) == 5
    assert len({row["writer_snapshot_bank_sha256"] for row in results}) == 5
    assert results[0]["writer_snapshot_scope"] == "condition_local_fresh_prime"
    assert results[0]["writer_prime_count"] == 32
    assert results[0]["replay_count"] == 64
    assert totals == {
        "condition_count": 5,
        "writer_prime_count": 160,
        "correct_replay_count": 160,
        "donor_replay_count": 160,
        "replay_count": 320,
    }

    assert all(raw.item() == 0.0 for raw in prime_observations[0]["raw"])
    native_raw = [
        native[name]["memory_fusion_residual_gain_raw"] for name, _ in modules
    ]
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(
            prime_observations[1]["raw"], native_raw, strict=True
        )
    )
    assert reset_calls == 6
    for name, module in modules:
        assert module.memory_fusion_placement == native[name]["memory_fusion_placement"]
        assert module.memory_fusion_mode == native[name]["memory_fusion_mode"]
        assert torch.equal(
            module.memory_fusion_residual_gain_raw.detach(),
            native[name]["memory_fusion_residual_gain_raw"],
        )


def test_evaluate_condition_records_actual_correct_and_donor_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenized = [{"row_index": 0}, {"row_index": 1}]
    snapshots = [
        {"marker": torch.tensor([0])},
        {"marker": torch.tensor([1])},
    ]
    calls: list[tuple[int, int]] = []

    def fake_selection(*args, **kwargs):
        return {
            "target_supervised_token_count": 1,
            "windows": {"1": {}, "8": {}, "16": {}, "32": {}},
        }

    def fake_replay(*, target_row, online_state, **kwargs):
        calls.append(
            (int(target_row["row_index"]), int(online_state["marker"].item()))
        )
        return {"token_nll": [0.0]}

    def fake_condition_metrics(replay, selection):
        return {"marker": replay["token_nll"][0]}

    def fake_directional(*args, **kwargs):
        return {
            "full_answer": {"ce_effect": 0.0},
            "history_windows": {
                key: {"ce_effect": 0.0} for key in ("1", "8", "16", "32")
            },
        }

    monkeypatch.setattr(diagnostic, "history_token_selection", fake_selection)
    monkeypatch.setattr(diagnostic, "replay_with_token_nll", fake_replay)
    monkeypatch.setattr(diagnostic, "condition_metrics", fake_condition_metrics)
    monkeypatch.setattr(diagnostic, "directional_row_effect", fake_directional)
    monkeypatch.setattr(diagnostic, "source_identity", lambda row, index: index)
    monkeypatch.setattr(
        diagnostic, "summarize_condition_rows", lambda rows, condition: {}
    )
    monkeypatch.setattr(diagnostic, "baseline_gap_summary", lambda rows: {})

    rows, summary, counts = diagnostic.evaluate_condition(
        model=object(),
        tokenized=tokenized,
        source_rows=[{}, {}],
        donors=[1, 0],
        snapshots=snapshots,
        device="cpu",
        primary_span_tokens=8,
    )

    assert calls == [(0, 0), (0, 1), (1, 1), (1, 0)]
    assert len(rows) == 2
    assert summary == {
        "correct_memory": {},
        "exact_donor_memory": {},
        "donor_minus_correct": {},
    }
    assert counts == {
        "correct_replay_count": 2,
        "donor_replay_count": 2,
        "replay_count": 4,
    }


def test_rank_conditions_rejects_an_outlier_driven_mean() -> None:
    def make_condition(
        name: str,
        gaps: list[float],
        correct_ce: float,
        correct_w1_ce: float = 5.0,
    ) -> dict:
        mean_gap = sum(gaps) / len(gaps)
        positive_fraction = sum(gap > 0.0 for gap in gaps) / len(gaps)
        return {
            "name": name,
            "effective_fusion_settings": [
                {
                    "memory_fusion_placement": "post_attention_residual_hybrid",
                    "memory_fusion_mode": "add",
                    "memory_fusion_residual_gain_effective": 0.01,
                }
            ],
            "summary": {
                "correct_memory": {
                    "full_answer": {"token_weighted_ce": correct_ce},
                    "history_windows": {
                        "1": {"token_weighted_ce": correct_w1_ce}
                    },
                },
                "donor_minus_correct": {
                    "full_answer": {"token_weighted_ce_effect": 0.0},
                    "history_windows": {
                        key: {
                            "token_weighted_ce_effect": mean_gap,
                            "positive_row_fraction": positive_fraction,
                        }
                        for key in ("1", "8")
                    },
                },
            },
            "rows": [
                {
                    "donor_minus_correct": {
                        "history_windows": {
                            key: {"ce_effect": gap} for key in ("1", "8")
                        }
                    }
                }
                for gap in gaps
            ],
        }

    stable = make_condition("stable", [0.1, 0.1, 0.1, -0.01], 1.5)
    outlier = make_condition("outlier", [2.0, -0.1, -0.1, -0.1], 1.0)
    native = make_condition("native", [0.05, 0.05, 0.05, -0.01], 1.4)

    ranking = diagnostic.rank_conditions([outlier, stable, native], 8)

    assert [row["condition"] for row in ranking] == [
        "stable",
        "native",
        "outlier",
    ]
    assert ranking[0]["stable_memory_signal"] is True
    assert ranking[2]["stable_memory_signal"] is False


def test_rank_conditions_rejects_destructive_correct_history_ce() -> None:
    def make_condition(name: str, gap: float, correct_ce: float) -> dict:
        return {
            "name": name,
            "effective_fusion_settings": [
                {
                    "memory_fusion_placement": "post_attention_residual_hybrid",
                    "memory_fusion_mode": "add",
                    "memory_fusion_residual_gain_effective": 0.01,
                }
            ],
            "summary": {
                "correct_memory": {
                    "full_answer": {"token_weighted_ce": correct_ce},
                    "history_windows": {"1": {"token_weighted_ce": 5.0}},
                },
                "donor_minus_correct": {
                    "full_answer": {"token_weighted_ce_effect": gap},
                    "history_windows": {
                        key: {
                            "token_weighted_ce_effect": gap,
                            "positive_row_fraction": 1.0,
                        }
                        for key in ("1", "8")
                    },
                },
            },
            "rows": [
                {
                    "donor_minus_correct": {
                        "history_windows": {
                            key: {"ce_effect": gap} for key in ("1", "8")
                        }
                    }
                }
                for _ in range(4)
            ],
        }

    native = make_condition("native", 0.05, 1.4)
    useful = make_condition("useful", 0.1, 1.5)
    destructive = make_condition("destructive", 1.0, 8.0)

    ranking = diagnostic.rank_conditions([destructive, native, useful], 8)

    assert ranking[0]["condition"] == "useful"
    assert ranking[-1]["condition"] == "destructive"
    assert ranking[-1]["stable_memory_signal"] is True
    assert ranking[-1]["correct_history_quality_constraint"]["passed"] is False
    assert ranking[-1]["selection_eligible"] is False


def test_effect_distribution_reports_stability_statistics() -> None:
    summary = diagnostic.effect_distribution([-0.1, 0.1, 0.2, 4.0])

    assert summary["count"] == 4
    assert summary["median"] == pytest.approx(0.15)
    assert summary["population_std"] > 1.0
    assert summary["absolute_gap_gt_0p2_count"] == 1
