from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset

from experiments.rethinking_rwkv_ms_gemma import eval_episode_memory_ce as evaluator


def test_parse_args_accepts_memory_fusion_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_episode_memory_ce.py",
            "--base-model",
            "/models/gemma",
            "--checkpoint",
            "/runs/trainer/checkpoint-384",
            "--tokenized-dataset",
            "/data/tokenized",
            "--source-jsonl",
            "/data/source.jsonl",
            "--memory-fusion-placement",
            "normalized_residual_correction",
            "--memory-fusion-residual-scale",
            "0.875",
        ],
    )

    args = evaluator.parse_args()

    assert args.memory_fusion_placement == "normalized_residual_correction"
    assert args.memory_fusion_residual_scale == 0.875


def test_default_output_path_is_unique_across_residual_scale_sweep(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "trainer" / "checkpoint-384"
    checkpoint.mkdir(parents=True)
    scales = (0.0, 0.75, 0.875, 0.95, 1.0)

    outputs = {
        scale: evaluator.default_output_path(
            checkpoint,
            memory_fusion_placement="normalized_residual_correction",
            memory_fusion_residual_scale=scale,
        )
        for scale in scales
    }

    assert len(set(outputs.values())) == len(scales)
    assert outputs[0.0].name.endswith(
        "_normalized_residual_correction_scale0.json"
    )
    assert outputs[0.875].name.endswith(
        "_normalized_residual_correction_scale0p875.json"
    )
    assert outputs[1.0].name.endswith(
        "_normalized_residual_correction_scale1.json"
    )
    assert evaluator.default_output_path(checkpoint) not in outputs.values()


def test_supervised_token_nll_uses_causal_shift_and_mask_order() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 3.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 0, -100, 2, 1]])
    attention_mask = torch.ones_like(labels)

    actual = evaluator.supervised_token_nll(logits, labels, attention_mask)
    expected = F.cross_entropy(
        torch.stack((logits[0, 0], logits[0, 2], logits[0, 3])),
        torch.tensor([0, 2, 1]),
        reduction="none",
    )

    assert torch.equal(actual, expected)


def test_summarize_token_nll_uses_chronological_prefixes() -> None:
    token_nll = torch.arange(1, 41, dtype=torch.float32)

    summary = evaluator.summarize_token_nll(token_nll)

    assert summary["token_count"] == 40
    assert summary["nll_sum"] == pytest.approx(820.0)
    assert summary["prefixes"]["1"]["ce"] == pytest.approx(1.0)
    assert summary["prefixes"]["8"]["nll_sum"] == pytest.approx(36.0)
    assert summary["prefixes"]["16"]["ce"] == pytest.approx(8.5)
    assert summary["prefixes"]["32"]["ce"] == pytest.approx(16.5)


def test_mismatched_donors_are_deterministic_deranged_permutation() -> None:
    tokenized = Dataset.from_dict(
        {"write_input_ids": [[index, index + 100] for index in range(12)]}
    )

    first = evaluator.make_mismatched_donors(tokenized, seed=71)
    second = evaluator.make_mismatched_donors(tokenized, seed=71)

    assert first == second
    assert sorted(first) == list(range(len(tokenized)))
    assert all(target_index != donor_index for target_index, donor_index in enumerate(first))
    assert all(
        tokenized[target_index]["write_input_ids"]
        != tokenized[donor_index]["write_input_ids"]
        for target_index, donor_index in enumerate(first)
    )


def test_legacy_presave_fingerprint_requires_matching_ready_cache_key(tmp_path) -> None:
    tokenized_path = tmp_path / "cache-key"
    tokenized_path.mkdir()
    ready = {"cache_key": "cache-key"}

    validation = evaluator.validate_tokenized_fingerprint(
        expected="before-save",
        actual="after-load",
        tokenized_path=tokenized_path,
        ready_metadata=ready,
    )

    assert validation["status"] == "legacy_presave_fingerprint_mismatch"
    with pytest.raises(ValueError, match="fingerprint does not match"):
        evaluator.validate_tokenized_fingerprint(
            expected="before-save",
            actual="after-load",
            tokenized_path=tokenized_path,
            ready_metadata={"cache_key": "another-cache"},
        )


def test_condition_summary_is_token_weighted() -> None:
    rows = [
        {
            "conditions": {
                evaluator.CONDITION_CORRECT: {
                    "ce": 1.0,
                    "nll_sum": 1.0,
                    "token_count": 1,
                    "prefixes": {
                        "8": {"ce": 2.0, "nll_sum": 2.0, "token_count": 1}
                    },
                }
            }
        },
        {
            "conditions": {
                evaluator.CONDITION_CORRECT: {
                    "ce": 3.0,
                    "nll_sum": 9.0,
                    "token_count": 3,
                    "prefixes": {
                        "8": {"ce": 4.0, "nll_sum": 12.0, "token_count": 3}
                    },
                }
            }
        },
    ]

    summary = evaluator.summarize_condition(rows, evaluator.CONDITION_CORRECT)
    prefix_summary = evaluator.summarize_condition(
        rows,
        evaluator.CONDITION_CORRECT,
        prefix_token_count=8,
    )

    assert summary["token_weighted_ce"] == pytest.approx(2.5)
    assert summary["mean_row_ce"] == pytest.approx(2.0)
    assert prefix_summary["token_weighted_ce"] == pytest.approx(3.5)
    assert prefix_summary["mean_row_ce"] == pytest.approx(3.0)


def test_read_configures_writes_before_context_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        evaluator,
        "reset_delta_mem_states",
        lambda model: events.append(("reset", None)),
    )
    monkeypatch.setattr(
        evaluator,
        "set_delta_mem_read_context_mask",
        lambda model, mask: events.append(("mask", mask)),
    )
    monkeypatch.setattr(
        evaluator,
        "set_delta_mem_write_message_ids",
        lambda model, ids: events.append(("message_ids", ids)),
    )
    monkeypatch.setattr(
        evaluator,
        "set_delta_mem_write_sentence_ids",
        lambda model, ids: events.append(("sentence_ids", ids)),
    )
    monkeypatch.setattr(
        evaluator,
        "set_delta_mem_write_enabled",
        lambda model, enabled: events.append(("write", enabled)),
    )
    monkeypatch.setattr(
        evaluator,
        "collect_delta_mem_state_stats",
        lambda model: {"nonzero_modules": 1},
    )
    monkeypatch.setattr(
        evaluator,
        "collect_delta_mem_output_ratio_stats",
        lambda model: {"normalized_residual_correction_fusion_modules": 1},
    )
    monkeypatch.setattr(evaluator, "logits_to_keep_kwargs", lambda model, value: {})

    class FakeModel:
        def __call__(self, *, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            events.append(("model", int(input_ids.size(1))))
            return SimpleNamespace(
                logits=torch.zeros(input_ids.size(0), input_ids.size(1), 4)
            )

    row = {
        "write_input_ids": [1, 2],
        "write_attention_mask": [1, 1],
        "write_message_ids": [0, 0],
        "write_sentence_ids": [-1, -1],
        "input_ids": [1, 2, 3, 0],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 3, 0],
    }

    evaluator.evaluate_condition(
        model=FakeModel(),
        target_row=row,
        write_row=row,
        device="cpu",
        read_write_enabled=True,
    )

    read_model_index = max(index for index, event in enumerate(events) if event[0] == "model")
    read_write_index = max(
        index
        for index, event in enumerate(events[:read_model_index])
        if event == ("write", True)
    )
    read_mask_index = max(
        index
        for index, event in enumerate(events[:read_model_index])
        if event[0] == "mask" and torch.is_tensor(event[1])
    )
    assert read_write_index < read_mask_index < read_model_index
    assert torch.equal(
        events[read_mask_index][1],
        torch.tensor([[True, True, False, False]]),
    )
