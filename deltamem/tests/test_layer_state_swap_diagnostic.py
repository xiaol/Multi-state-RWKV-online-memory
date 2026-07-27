from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import diagnose_layer_state_swaps as diagnostic


def make_row(answer_ids: list[int], row_index: int = 0) -> dict:
    return {
        "row_index": row_index,
        "input_ids": [7, *answer_ids],
        "attention_mask": [1] * (len(answer_ids) + 1),
        "labels": [-100, *answer_ids],
        "write_input_ids": [1000 + row_index],
        "write_attention_mask": [1],
        "write_message_ids": [0],
        "write_sentence_ids": [0],
    }


def make_snapshot(row_index: int, module_names: list[str]) -> dict[str, torch.Tensor]:
    snapshot = {}
    for name in module_names:
        snapshot[name] = torch.tensor([float(row_index)])
        snapshot[f"{name}.__rwkv_ms_positions"] = torch.tensor([1])
        snapshot[f"{name}.__rwkv_ms_previous_source"] = torch.tensor([float(row_index)])
    return snapshot


def test_default_output_path_uses_checkpoint_run_root(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "trainer" / "checkpoint-416"
    checkpoint.mkdir(parents=True)

    output = diagnostic.default_output_path(checkpoint)

    assert output == (
        tmp_path
        / "run"
        / "layer_state_swap_diagnostic"
        / "checkpoint-416_all_rows_layer_state_swaps.json"
    )


def test_parse_args_defaults_to_single_pre_leak_history_token(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostic.sys,
        "argv",
        [
            "diagnose_layer_state_swaps.py",
            "--base-model",
            "base",
            "--checkpoint",
            "checkpoint",
            "--tokenized-dataset",
            "tokenized",
            "--source-jsonl",
            "source.jsonl",
        ],
    )

    args = diagnostic.parse_args()

    assert args.history_span_tokens == 1
    assert args.symmetric_top_k == 6
    assert args.baseline_only is False
    assert args.expected_layer_count == 42
    assert args.expected_row_count == 32


def test_parse_args_accepts_baseline_only(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostic.sys,
        "argv",
        [
            "diagnose_layer_state_swaps.py",
            "--base-model",
            "base",
            "--checkpoint",
            "checkpoint",
            "--tokenized-dataset",
            "tokenized",
            "--source-jsonl",
            "source.jsonl",
            "--baseline-only",
            "--history-span-tokens",
            "8",
        ],
    )

    args = diagnostic.parse_args()

    assert args.baseline_only is True
    assert args.history_span_tokens == 8


def test_validate_layer_modules_accepts_saved_residual_hybrid() -> None:
    modules = [
        (
            f"model.layers.{layer_index}.self_attn",
            SimpleNamespace(
                layer_idx=layer_index,
                memory_backend="rwkv_ms",
                memory_fusion_placement="post_attention_residual_hybrid",
            ),
        )
        for layer_index in range(2)
    ]

    assert diagnostic.validate_layer_modules(modules, expected_layer_count=2) == modules


def test_history_token_selection_uses_first_positional_mismatch() -> None:
    target_ids = [10, 11, 12, 20, *range(30, 70)]
    donor_ids = [10, 11, 12, 99, *range(130, 170)]

    selection = diagnostic.history_token_selection(
        make_row(target_ids),
        make_row(donor_ids),
        primary_span_tokens=8,
        unaligned_policy="error",
    )

    assert selection["status"] == "aligned"
    assert selection["first_history_ordinal"] == 3
    assert selection["first_history_label_position"] == 4
    assert selection["first_history_predictor_position"] == 3
    assert selection["causal_prefix_identical"] is True
    assert selection["primary_window_key"] == "8"
    assert selection["windows"]["1"]["supervised_ordinals"] == [3]
    assert selection["windows"]["8"]["supervised_ordinals"] == list(range(3, 11))
    assert selection["windows"]["32"]["target_predictor_positions"][0] == 3


@pytest.mark.parametrize(
    "target_ids,donor_ids,reason",
    [
        ([10, 11], [10], "counts differ"),
        ([10, 11], [10, 11], "identical"),
    ],
)
def test_history_token_selection_errors_on_unaligned_pairs(
    target_ids: list[int],
    donor_ids: list[int],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        diagnostic.history_token_selection(
            make_row(target_ids),
            make_row(donor_ids),
            primary_span_tokens=8,
            unaligned_policy="error",
        )


def test_history_token_selection_records_full_answer_fallback() -> None:
    selection = diagnostic.history_token_selection(
        make_row([10, 11, 12]),
        make_row([10, 11]),
        primary_span_tokens=8,
        unaligned_policy="full_answer",
    )

    assert selection["fallback_used"] is True
    assert selection["first_history_ordinal"] is None
    assert selection["windows"]["8"]["selection_kind"] == "full_answer_fallback"
    assert selection["windows"]["8"]["supervised_ordinals"] == [0, 1, 2]


def test_summarize_effect_scopes_uses_token_weighted_nll() -> None:
    summary = diagnostic.summarize_effect_scopes(
        [
            {
                "token_count": 1,
                "ce_effect": 2.0,
                "nll_effect": 2.0,
                "positive": True,
            },
            {
                "token_count": 3,
                "ce_effect": 0.0,
                "nll_effect": 0.0,
                "positive": False,
            },
        ]
    )

    assert summary["mean_ce_effect"] == pytest.approx(1.0)
    assert summary["median_ce_effect"] == pytest.approx(1.0)
    assert summary["min_ce_effect"] == pytest.approx(0.0)
    assert summary["max_ce_effect"] == pytest.approx(2.0)
    assert summary["token_weighted_ce_effect"] == pytest.approx(0.5)
    assert summary["positive_row_count"] == 1
    assert summary["positive_row_fraction"] == pytest.approx(0.5)


def test_prime_writer_snapshots_primes_each_row_once(monkeypatch) -> None:
    module_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    modules = [
        (name, SimpleNamespace(layer_idx=layer_index))
        for layer_index, name in enumerate(module_names)
    ]
    rows = [make_row([10, 11], row_index) for row_index in range(3)]
    calls = {"reset": 0, "prime": 0, "state": 0}

    def fake_reset(model, *, write_enabled: bool) -> None:
        assert write_enabled is True
        calls["reset"] += 1

    def fake_prime(model, row, device: str) -> None:
        assert device == "cpu"
        calls["prime"] += 1

    def fake_state(model) -> dict[str, torch.Tensor]:
        row_index = calls["state"]
        calls["state"] += 1
        return make_snapshot(row_index, module_names)

    monkeypatch.setattr(diagnostic, "reset_runtime", fake_reset)
    monkeypatch.setattr(diagnostic, "prime_write", fake_prime)
    monkeypatch.setattr(diagnostic, "get_delta_mem_online_state", fake_state)

    snapshots = diagnostic.prime_writer_snapshots(
        model=object(),
        tokenized=rows,
        modules=modules,
        device="cpu",
    )

    assert len(snapshots) == 3
    assert calls == {"reset": 3, "prime": 3, "state": 3}


def test_dataset_swaps_rank_history_window_and_run_symmetric_top_layer(
    monkeypatch,
) -> None:
    module_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    modules = [
        (name, SimpleNamespace(layer_idx=layer_index))
        for layer_index, name in enumerate(module_names)
    ]
    shared_prefix = [10, 11, 12]
    tokenized = [
        make_row(shared_prefix + [20] * 37, row_index=0),
        make_row(shared_prefix + [30] * 37, row_index=1),
    ]
    donors = [1, 0]
    snapshots = [make_snapshot(row_index, module_names) for row_index in range(2)]
    source_rows = [{}, {}]
    replay_calls = []

    def fake_replay_fixed_target(**kwargs):
        assert kwargs["include_token_nll"] is True
        target_index = int(kwargs["target_row"]["row_index"])
        online_state = kwargs["online_state"]
        token_nll = [1.0] * 40
        layer_effects = (0.4, 0.1)
        for name, layer_effect in zip(module_names, layer_effects):
            state_row = int(online_state[name].item())
            if state_row != target_index:
                for ordinal in range(3, 11):
                    token_nll[ordinal] += layer_effect
        state_rows = tuple(int(online_state[name].item()) for name in module_names)
        replay_calls.append((target_index, state_rows))
        nll_sum = sum(token_nll)
        return (
            {
                "token_count": len(token_nll),
                "nll_sum": nll_sum,
                "ce": nll_sum / len(token_nll),
                "token_nll": token_nll,
            },
            {},
        )

    monkeypatch.setattr(diagnostic, "replay_fixed_target", fake_replay_fixed_target)

    baselines, layers = diagnostic.evaluate_donor_to_correct_swaps(
        model=object(),
        tokenized=tokenized,
        source_rows=source_rows,
        donors=donors,
        snapshots=snapshots,
        modules=modules,
        device="cpu",
        primary_span_tokens=8,
        unaligned_policy="error",
        progress_layer_interval=2,
    )
    rankings = diagnostic.rank_forward_layers(layers, primary_span_tokens=8)

    assert len(replay_calls) == 8
    assert baselines[0]["history_token_selection"]["first_history_ordinal"] == 3
    assert rankings["by_primary_token_weighted_ce_gain"][0]["layer_index"] == 0
    assert rankings["by_primary_token_weighted_ce_gain"][0][
        "token_weighted_ce_gain"
    ] == pytest.approx(0.4)
    assert layers[module_names[0]]["donor_to_correct"]["aggregate"][
        "full_answer"
    ]["token_weighted_ce_effect"] == pytest.approx(0.08)

    diagnostic.evaluate_correct_to_donor_swaps(
        model=object(),
        tokenized=tokenized,
        donors=donors,
        snapshots=snapshots,
        baselines=baselines,
        layer_results=layers,
        selected_module_names=[module_names[0]],
        device="cpu",
        primary_span_tokens=8,
    )
    bidirectional = diagnostic.rank_bidirectional_layers(
        layers,
        [module_names[0]],
        primary_span_tokens=8,
    )

    assert len(replay_calls) == 10
    assert bidirectional[0]["bidirectional_token_weighted_ce_effect"] == pytest.approx(0.4)
    assert bidirectional[0]["bidirectional_positive_row_count"] == 2


def test_baseline_only_scores_exact_w1_w8_without_layer_swaps(monkeypatch) -> None:
    module_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    modules = [
        (name, SimpleNamespace(layer_idx=layer_index))
        for layer_index, name in enumerate(module_names)
    ]
    shared_prefix = [10, 11, 12]
    tokenized = [
        make_row(shared_prefix + [20] * 37, row_index=0),
        make_row(shared_prefix + [30] * 37, row_index=1),
    ]
    donors = [1, 0]
    snapshots = [make_snapshot(row_index, module_names) for row_index in range(2)]
    replay_calls = []

    def fake_replay_fixed_target(**kwargs):
        target_index = int(kwargs["target_row"]["row_index"])
        state_index = int(kwargs["online_state"][module_names[0]].item())
        token_nll = [1.0] * 40
        if state_index != target_index:
            token_nll[3] += 0.5
            for ordinal in range(4, 11):
                token_nll[ordinal] += 0.25
        replay_calls.append((target_index, state_index))
        return (
            {
                "token_count": len(token_nll),
                "nll_sum": sum(token_nll),
                "ce": sum(token_nll) / len(token_nll),
                "token_nll": token_nll,
            },
            {},
        )

    monkeypatch.setattr(diagnostic, "replay_fixed_target", fake_replay_fixed_target)
    monkeypatch.setattr(
        diagnostic,
        "evaluate_donor_to_correct_swaps",
        lambda **kwargs: pytest.fail("baseline-only must not run layer swaps"),
    )
    monkeypatch.setattr(
        diagnostic,
        "evaluate_correct_to_donor_swaps",
        lambda **kwargs: pytest.fail("baseline-only must not run symmetric swaps"),
    )

    baselines, layers, forward, selected, bidirectional = (
        diagnostic.run_diagnostic_evaluation(
            model=object(),
            tokenized=tokenized,
            source_rows=[{}, {}],
            donors=donors,
            snapshots=snapshots,
            modules=modules,
            device="cpu",
            primary_span_tokens=8,
            unaligned_policy="error",
            symmetric_top_k=0,
            baseline_only=True,
        )
    )
    summary = diagnostic.baseline_gap_summary(baselines)

    assert len(replay_calls) == 4
    assert summary["history_windows"]["1"]["token_weighted_ce_effect"] == pytest.approx(0.5)
    assert summary["history_windows"]["8"]["token_weighted_ce_effect"] == pytest.approx(
        0.28125
    )
    assert layers == {}
    assert forward == diagnostic.empty_forward_rankings()
    assert selected == []
    assert bidirectional == []


def test_bidirectional_positive_requires_both_directions() -> None:
    combined = diagnostic._bidirectional_scope(
        {
            "token_count": 8,
            "ce_effect": 0.2,
            "nll_effect": 1.6,
            "positive": True,
        },
        {
            "token_count": 8,
            "ce_effect": -0.1,
            "nll_effect": -0.8,
            "positive": False,
        },
    )

    assert combined["ce_effect"] == pytest.approx(0.05)
    assert combined["positive"] is False
