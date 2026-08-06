from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    profile_natural_memory_gate_cuda as profiler,
)


GIB = 1024**3


def _example(
    row_id: str,
    *,
    write_lengths: tuple[int, int, int, int],
    read_length: int,
    condition: str = "correct_state",
    predictor_start: int | None = None,
    predictor_length: int = 1,
) -> SimpleNamespace:
    labels = [-100] * read_length
    if predictor_start is not None:
        if (
            predictor_start < 0
            or predictor_length <= 0
            or predictor_start + predictor_length >= read_length
        ):
            raise ValueError("Synthetic predictor span does not fit the read")
        for predictor_index in range(
            predictor_start, predictor_start + predictor_length
        ):
            labels[predictor_index + 1] = predictor_index + 100
    return SimpleNamespace(
        row_id=row_id,
        episode_id=f"episode-{row_id}",
        task="narrative",
        condition=condition,
        write_records=tuple(
            {"input_ids": tuple(range(length))} for length in write_lengths
        ),
        read_input_ids=tuple(range(read_length)),
        labels=tuple(labels),
    )


def test_signed_payload_detects_tampering() -> None:
    receipt = profiler.signed_payload(
        {"schema": "test", "value": [1, 2, 3]}, "receipt_sha256"
    )

    assert profiler.verify_signed_payload(receipt, "receipt_sha256")
    receipt["value"].append(4)
    assert not profiler.verify_signed_payload(receipt, "receipt_sha256")


def test_select_longest_examples_is_deterministic_and_tie_breaks_by_row() -> None:
    examples = [
        _example("z", write_lengths=(10, 10, 10, 10), read_length=20),
        _example("b", write_lengths=(12, 12, 12, 12), read_length=20),
        _example("a", write_lengths=(12, 12, 12, 12), read_length=20),
        _example("short", write_lengths=(5, 5, 5, 5), read_length=5),
    ]

    selected, evidence = profiler.select_longest_examples(examples, 3)

    assert [example.row_id for example in selected] == ["a", "b", "z"]
    assert [item["row_id"] for item in evidence] == ["a", "b", "z"]
    assert evidence[0]["total_unpadded_token_positions"] == 68


def test_select_longest_examples_rejects_non_correct_state() -> None:
    examples = [
        _example(
            "wrong",
            write_lengths=(10, 10, 10, 10),
            read_length=20,
            condition="donor_state",
        )
    ]

    with pytest.raises(ValueError, match="only correct_state"):
        profiler.select_longest_examples(examples, 1)


def test_selection_exactly_maximizes_batch_padded_workload() -> None:
    examples = [
        _example("cover", write_lengths=(90, 90, 1, 1), read_length=1),
        _example("redundant", write_lengths=(89, 89, 1, 1), read_length=1),
        _example("write-0", write_lengths=(100, 1, 1, 1), read_length=1),
        _example("write-1", write_lengths=(1, 100, 1, 1), read_length=1),
        _example("write-2", write_lengths=(1, 1, 100, 1), read_length=1),
        _example("write-3", write_lengths=(1, 1, 1, 100), read_length=1),
        _example("read", write_lengths=(1, 1, 1, 1), read_length=100),
    ]

    selected, _, audit = profiler.select_padded_workload_examples(examples, 4)

    assert {example.row_id for example in selected} == {
        "cover",
        "write-2",
        "write-3",
        "read",
    }
    assert audit["selected"]["total_padded_token_positions"] == 1920
    assert (
        audit["unconstrained_per_dimension_upper_bound"][
            "total_padded_token_positions"
        ]
        == 2000
    )
    assert audit["upper_bound_coverage_fraction"] == pytest.approx(0.96)
    assert audit["selected_batch_is_exact_constrained_optimum"] is True


def test_selection_matches_brute_force_on_random_small_corpora() -> None:
    rng = random.Random(1701)
    for trial in range(40):
        maximum_length = 3 if trial % 2 else 24
        examples = [
            _example(
                f"{trial:02d}-{row:02d}",
                write_lengths=tuple(
                    rng.randint(1, maximum_length) for _ in range(4)
                ),
                read_length=rng.randint(1, maximum_length),
            )
            for row in range(7)
        ]
        for batch_size in range(1, len(examples) + 1):
            selected, _, audit = profiler.select_padded_workload_examples(
                examples, batch_size
            )
            brute_force_score = max(
                profiler._padded_workload(
                    [profiler._example_token_profile(example) for example in subset],
                    batch_size,
                )["total_padded_token_positions"]
                for subset in combinations(examples, batch_size)
            )
            repeated, _, _ = profiler.select_padded_workload_examples(
                list(reversed(examples)), batch_size
            )

            assert (
                audit["selected"]["total_padded_token_positions"]
                == brute_force_score
            )
            assert [example.row_id for example in selected] == [
                example.row_id for example in repeated
            ]


def test_answer_logit_selection_matches_brute_force_interval_union() -> None:
    rng = random.Random(2718)
    for trial in range(30):
        examples = []
        for row in range(7):
            predictor_length = rng.randint(1, 5)
            predictor_start = rng.randint(0, 28 - predictor_length)
            examples.append(
                _example(
                    f"logit-{trial:02d}-{row:02d}",
                    write_lengths=tuple(rng.randint(1, 12) for _ in range(4)),
                    read_length=30,
                    predictor_start=predictor_start,
                    predictor_length=predictor_length,
                )
            )
        for batch_size in range(1, len(examples) + 1):
            selected, _, audit = profiler.select_answer_logit_examples(
                examples, batch_size
            )
            brute_force_score = max(
                len(
                    profiler._answer_predictor_union_indices(
                        [
                            {
                                **profiler._example_token_profile(example),
                                **profiler._answer_predictor_profile(example),
                            }
                            for example in subset
                        ]
                    )
                )
                for subset in combinations(examples, batch_size)
            )
            repeated, _, _ = profiler.select_answer_logit_examples(
                list(reversed(examples)), batch_size
            )

            assert (
                audit["selected_answer_predictor_union_positions"]
                == brute_force_score
            )
            assert [example.row_id for example in selected] == [
                example.row_id for example in repeated
            ]


def _phase(peak_reserved: int, peak_allocated: int, free_after: int) -> dict:
    return {
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_bytes": peak_allocated,
        "after": {"free_bytes": free_after},
    }


def test_memory_gate_uses_conservative_post_peak_headroom() -> None:
    initial = {
        "total_bytes": 40 * GIB,
        "free_bytes": 39 * GIB,
        "reserved_bytes": 0,
    }

    result = profiler.build_memory_gate(
        initial_snapshot=initial,
        phases=[
            _phase(10 * GIB, 9 * GIB, 29 * GIB),
            _phase(30 * GIB, 27 * GIB, 9 * GIB),
        ],
    )

    assert result["peak_reserved_bytes"] == 30 * GIB
    assert result["isolated_reserved_headroom_bytes"] == 10 * GIB
    assert result["environment_adjusted_reserved_headroom_bytes"] == 9 * GIB
    assert result["post_peak_reserved_memory_headroom_bytes"] == 9 * GIB
    assert result["headroom_passed"] is True


def test_memory_gate_fails_below_five_gib() -> None:
    result = profiler.build_memory_gate(
        initial_snapshot={
            "total_bytes": 40 * GIB,
            "free_bytes": 39 * GIB,
            "reserved_bytes": 0,
        },
        phases=[_phase(36 * GIB, 34 * GIB, 3 * GIB)],
    )

    assert result["post_peak_reserved_memory_headroom_bytes"] == 3 * GIB
    assert result["headroom_passed"] is False


def test_profiled_local_batch_sizes_and_explicit_cuda_device_are_locked() -> None:
    assert profiler.PROFILED_LOCAL_BATCH_SIZES == (1, 2, 4)
    assert profiler.REQUIRED_LOCAL_BATCH_SIZES == (1,)
    assert profiler.EXPLORATORY_LOCAL_BATCH_SIZES == (2, 4)
    assert profiler._parse_batch_sizes("1,2,4") == (1, 2, 4)
    assert profiler._parse_cuda_device("cuda:3") == torch.device("cuda:3")
    with pytest.raises(
        ValueError,
        match="requires local batch sizes 1,2,4 in that order",
    ):
        profiler._parse_batch_sizes("1,4,2")
    with pytest.raises(ValueError, match="must be explicit"):
        profiler._parse_cuda_device("cuda")


def test_execute_optimizer_step_uses_real_update_sequence(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    target_slots = torch.tensor([0, 1])
    batch = SimpleNamespace(
        examples=[object(), object()],
        labels=torch.tensor([[1, 1], [1, 1]]),
        query_mask=torch.ones((2, 2), dtype=torch.bool),
        target_slots=target_slots,
    )
    calls: list[str] = []
    prior_liveness = profiler.OptimizerStepLiveness(
        write_audit=object(),
        logits=object(),
        route_logits=object(),
        answer_loss=object(),
        route_loss=object(),
        route_predictions=object(),
        total_loss=object(),
        grad_norm=object(),
        router_audit=object(),
    )
    rollover_holder = [prior_liveness]

    def fake_write(model, batch, *, dtype):
        assert prior_liveness.write_audit is not None
        calls.append("write")
        return {
            "full_occupancy_count": 2,
            "full_occupancy_total": 2,
            "forced_write_route_match_count": 8,
            "forced_write_route_total": 8,
        }

    def fake_read(model, batch, *, dtype):
        assert prior_liveness.write_audit is None
        assert prior_liveness.logits is not None
        calls.append("read")
        logits = model.weight.reshape(1, 1, 1).expand(2, 1, 1)
        return logits, {"layer": logits}

    def fake_answer_loss(logits, labels):
        assert prior_liveness.logits is None
        assert prior_liveness.route_logits is None
        assert prior_liveness.answer_loss is not None
        calls.append("answer_loss")
        return (model.weight - 3.0).square().mean()

    def fake_route_loss(route_logits, query_mask, slots):
        assert prior_liveness.answer_loss is None
        assert prior_liveness.route_loss is not None
        calls.append("route_loss")
        return (model.weight - 2.0).square().mean(), {"layer": slots.clone()}

    def fake_router_audit(model, route_loss):
        assert prior_liveness.route_loss is None
        assert prior_liveness.route_predictions is None
        assert prior_liveness.total_loss is None
        calls.append("router_audit")
        return {"all_modules_finite_nonzero": True}

    monkeypatch.setattr(profiler.gate, "_write_episode_batch", fake_write)
    monkeypatch.setattr(profiler.gate, "_read_episode_batch", fake_read)
    monkeypatch.setattr(profiler.gate, "causal_answer_loss", fake_answer_loss)
    monkeypatch.setattr(
        profiler.gate, "route_loss_and_predictions", fake_route_loss
    )
    monkeypatch.setattr(
        profiler.gate.runtime, "_router_gradient_audit", fake_router_audit
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    result, liveness = profiler._execute_optimizer_step(
        model,
        batch,
        optimizer,
        list(model.parameters()),
        dtype=torch.float32,
        include_router_gradient_audit=True,
        rollover_liveness_holder=rollover_holder,
    )

    assert calls == ["write", "read", "answer_loss", "route_loss", "router_audit"]
    assert not torch.equal(model.weight.detach(), before)
    assert result["rows"] == 2
    assert result["route_correct"] == result["route_total"] == 2
    assert liveness.logits.shape == (2, 1, 1)
    assert rollover_holder == []


def test_measure_optimizer_step_preserves_cuda_oom_peak_evidence(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cuda:3")

    def raise_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(profiler, "_execute_optimizer_step", raise_oom)
    monkeypatch.setattr(
        profiler.gate,
        "collate_examples",
        lambda examples, pad_token_id, device: SimpleNamespace(),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        profiler,
        "_cuda_snapshot",
        lambda device: {
            "allocated_bytes": 30 * GIB,
            "reserved_bytes": 31 * GIB,
            "free_bytes": 8 * GIB,
            "total_bytes": 40 * GIB,
        },
    )
    monkeypatch.setattr(
        profiler,
        "_phase_result",
        lambda device, before, elapsed_seconds: {
            "before": dict(before),
            "after": dict(before),
            "peak_allocated_bytes": 31 * GIB,
            "peak_reserved_bytes": 32 * GIB,
            "elapsed_seconds": elapsed_seconds,
        },
    )

    result, liveness, batch = profiler._measure_optimizer_step(
        model,
        [SimpleNamespace()],
        optimizer,
        list(model.parameters()),
        pad_token_id=0,
        device=device,
        dtype=torch.bfloat16,
        include_router_gradient_audit=True,
    )

    assert result["status"] == "cuda_out_of_memory"
    assert result["step"] is None
    assert result["peak_reserved_bytes"] == 32 * GIB
    assert result["error"]["type"] == "OutOfMemoryError"
    assert len(result["error"]["traceback_sha256"]) == 64
    assert liveness is None
    assert isinstance(batch, SimpleNamespace)


def test_worker_preserves_setup_cuda_oom_telemetry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    initial = {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "free_bytes": 39 * GIB,
        "total_bytes": 40 * GIB,
    }

    def raise_setup_oom(**kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic setup OOM")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(profiler, "_cuda_snapshot", lambda device: dict(initial))
    monkeypatch.setattr(
        profiler,
        "_phase_result",
        lambda device, before, elapsed_seconds: {
            "before": dict(before),
            "after": dict(initial),
            "peak_allocated_bytes": 35 * GIB,
            "peak_reserved_bytes": 36 * GIB,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    monkeypatch.setattr(
        profiler,
        "_device_evidence",
        lambda device: {"requested": str(device)},
    )
    monkeypatch.setattr(
        profiler,
        "_profile_worker_on_initialized_device",
        raise_setup_oom,
    )

    result = profiler._profile_worker(
        source_manifest=manifest,
        batch_size=4,
        device_name="cuda:3",
    )

    assert result["status"] == "failed"
    assert result["gate_passed"] is False
    assert result["cuda_oom_telemetry"]["active_phase"]["status"] == (
        "cuda_out_of_memory"
    )
    assert result["memory_gate"]["peak_reserved_bytes"] == 36 * GIB
    assert profiler.verify_signed_payload(result, "worker_receipt_sha256")


def test_measure_step_keeps_prior_batch_and_gradients_live_during_collation(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cuda:3")
    prior_batch = object()
    prior_batch_holder = [prior_batch]

    def fake_collate(examples, *, pad_token_id, device):
        assert prior_batch_holder == [prior_batch]
        assert model.weight.grad is not None
        return SimpleNamespace(examples=list(examples))

    def fake_execute(*args, **kwargs):
        assert prior_batch_holder == []
        assert model.weight.grad is None
        return {"rows": 1}, (object(),)

    monkeypatch.setattr(profiler.gate, "collate_examples", fake_collate)
    monkeypatch.setattr(profiler, "_execute_optimizer_step", fake_execute)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        profiler,
        "_cuda_snapshot",
        lambda device: {
            "allocated_bytes": 2 * GIB,
            "reserved_bytes": 3 * GIB,
            "free_bytes": 37 * GIB,
            "total_bytes": 40 * GIB,
        },
    )
    monkeypatch.setattr(
        profiler,
        "_phase_result",
        lambda device, before, elapsed_seconds: {
            "before": dict(before),
            "after": dict(before),
            "peak_allocated_bytes": 2 * GIB,
            "peak_reserved_bytes": 3 * GIB,
            "elapsed_seconds": elapsed_seconds,
        },
    )

    result, liveness, batch = profiler._measure_optimizer_step(
        model,
        [SimpleNamespace()],
        optimizer,
        list(model.parameters()),
        pad_token_id=0,
        device=device,
        dtype=torch.bfloat16,
        include_router_gradient_audit=False,
        prior_batch_holder=prior_batch_holder,
    )

    assert result["status"] == "passed"
    assert liveness is not None
    assert isinstance(batch, SimpleNamespace)
    assert prior_batch_holder == []


def test_stress_sequence_profiles_both_cross_batch_rollover_orders() -> None:
    calls: list[tuple[str, str | None, str | None, bool]] = []

    def fake_measure(
        model,
        examples,
        optimizer,
        trainable,
        *,
        include_router_gradient_audit,
        rollover_liveness_holder,
        prior_batch_holder,
        **kwargs,
    ):
        role = examples[0]
        prior_liveness_role = (
            None
            if not rollover_liveness_holder
            else rollover_liveness_holder[0].router_audit
        )
        prior_batch_role = (
            None if not prior_batch_holder else prior_batch_holder[0].role
        )
        calls.append(
            (
                role,
                prior_liveness_role,
                prior_batch_role,
                include_router_gradient_audit,
            )
        )
        liveness = profiler.OptimizerStepLiveness(
            write_audit=None,
            logits=None,
            route_logits=None,
            answer_loss=None,
            route_loss=None,
            route_predictions=None,
            total_loss=None,
            grad_norm=None,
            router_audit=role,
        )
        return {"status": "passed"}, liveness, SimpleNamespace(role=role)

    phases, completed = profiler._measure_stress_sequence(
        SimpleNamespace(),
        activation_examples=["activation"],
        answer_logit_examples=["answer_logit"],
        optimizer=SimpleNamespace(),
        trainable=[],
        pad_token_id=0,
        device=torch.device("cuda:3"),
        dtype=torch.bfloat16,
        measure_step=fake_measure,
    )

    assert completed == profiler.PRODUCTION_PROFILE_OPTIMIZER_STEPS
    assert calls == [
        ("activation", None, None, True),
        ("answer_logit", "activation", "activation", True),
        ("activation", "answer_logit", "answer_logit", False),
    ]
    assert all(phase["status"] == "passed" for phase in phases.values())


def _fake_worker_payload(
    batch_size: int,
    pid: int,
    manifest_file_sha256: str,
    *,
    manifest_path: Path = Path("/formal/natural-memory/manifest.json"),
) -> dict:
    initial = {
        "allocated_bytes": 0,
        "total_bytes": 40 * GIB,
        "free_bytes": 39 * GIB,
        "reserved_bytes": 0,
    }
    load_phase = {
        "before": dict(initial),
        "after": {
            "allocated_bytes": 9 * GIB,
            "total_bytes": 40 * GIB,
            "free_bytes": 29 * GIB,
            "reserved_bytes": 10 * GIB,
        },
        "peak_allocated_bytes": 9 * GIB,
        "peak_reserved_bytes": 10 * GIB,
        "elapsed_seconds": 1.0,
        "status": "passed",
    }
    def optimizer_phase(
        *,
        before: dict,
        include_router_gradient_audit: bool,
    ) -> dict:
        answer_predictor_positions = 8
        route_total = len(profiler.PRODUCTION_TARGET_LAYERS) * batch_size
        full_occupancy_total = route_total
        forced_write_route_total = (
            profiler.gate.RECORDS_PER_EPISODE * route_total
        )
        router_gradient_audit = (
            {
                "modules": len(profiler.PRODUCTION_TARGET_LAYERS),
                "finite_nonzero_modules": len(profiler.PRODUCTION_TARGET_LAYERS),
                "all_modules_finite_nonzero": True,
                "records": [
                    {
                        "module": f"model.layers.{layer}.self_attn",
                        "layer": layer,
                        "projected_kv_key_route_grad_norm": 0.1,
                        "finite_nonzero": True,
                    }
                    for layer in profiler.PRODUCTION_TARGET_LAYERS
                ],
            }
            if include_router_gradient_audit
            else None
        )
        return {
            "before": dict(before),
            "after": {
                "allocated_bytes": 20 * GIB,
                "total_bytes": 40 * GIB,
                "free_bytes": 18 * GIB,
                "reserved_bytes": 21 * GIB,
            },
            "peak_allocated_bytes": 20 * GIB,
            "peak_reserved_bytes": 21 * GIB,
            "elapsed_seconds": 2.0,
            "status": "passed",
            "step": {
                "rows": batch_size,
                "answer_predictor_positions": answer_predictor_positions,
                "answer_logit_shape": [
                    batch_size,
                    answer_predictor_positions,
                    262144,
                ],
                "answer_logit_dtype": "torch.bfloat16",
                "answer_loss": 1.25,
                "route_loss": 0.75,
                "total_loss": 2.0,
                "gradient_norm": 0.5,
                "route_correct": route_total,
                "route_total": route_total,
                "full_occupancy_count": full_occupancy_total,
                "full_occupancy_total": full_occupancy_total,
                "forced_write_route_match_count": forced_write_route_total,
                "forced_write_route_total": forced_write_route_total,
                "router_gradient_audit": router_gradient_audit,
            },
            "includes_first_step_router_gradient_audit": (
                include_router_gradient_audit
            ),
        }

    optimizer_phases = {}
    phase_before = dict(load_phase["after"])
    for name, _, include_router_audit in profiler.PROFILE_STRESS_SEQUENCE:
        phase = optimizer_phase(
            before=phase_before,
            include_router_gradient_audit=include_router_audit,
        )
        optimizer_phases[name] = phase
        phase_before = dict(phase["after"])
    phases = {"fresh_model_load": load_phase, **optimizer_phases}
    memory_gate = profiler.build_memory_gate(
        initial_snapshot=initial,
        phases=list(phases.values()),
    )
    source_files = {
        str(manifest_path): {
            "bytes": 128,
            "sha256": manifest_file_sha256,
        },
        str(manifest_path.parent / "train.jsonl"): {
            "bytes": 1024,
            "sha256": "1" * 64,
        },
        str(manifest_path.parent / "development.jsonl"): {
            "bytes": 1024,
            "sha256": "2" * 64,
        },
    }
    model_path = Path("/formal/models/gemma-4-e4b")
    model_files = {
        str(model_path / "config.json"): {
            "bytes": 256,
            "sha256": "3" * 64,
        },
        str(model_path / "model.safetensors"): {
            "bytes": 4096,
            "sha256": "4" * 64,
        },
        str(model_path / "tokenizer.json"): {
            "bytes": 512,
            "sha256": "5" * 64,
        },
    }
    trainable_names_sha256 = "6" * 64
    activation_profiles = [
        {
            "row_id": f"activation-max-{index}",
            "episode_id": f"activation-episode-{index}",
            "task": "narrative",
            "condition": "correct_state",
            "write_token_lengths": [64, 64, 64, 64],
            "read_token_length": 128,
            "total_unpadded_token_positions": 384,
            "maximum_sequence_token_length": 128,
            "answer_predictor_start": 120,
            "answer_predictor_end_exclusive": 128,
            "answer_predictor_positions": 8,
        }
        for index in range(batch_size)
    ]
    answer_logit_profiles = [
        {
            **profile,
            "row_id": f"answer-logit-max-{index}",
            "episode_id": f"answer-logit-episode-{index}",
            "answer_predictor_start": 120,
            "answer_predictor_end_exclusive": 128,
            "answer_predictor_positions": 8,
        }
        for index, profile in enumerate(activation_profiles)
    ]
    selected_padded_workload = profiler._padded_workload(
        activation_profiles, batch_size
    )
    return profiler.signed_payload(
        {
            "schema": profiler.WORKER_SCHEMA,
            "status": "passed",
            "pid": pid,
            "hf_endpoint": profiler.HF_MIRROR_ENDPOINT,
            "device": {
                "requested": "cuda:3",
                "index": 3,
                "name": "NVIDIA A100-PCIE-40GB",
                "capability": [8, 0],
                "reported_total_memory_bytes": 40 * GIB,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            },
            "source_manifest_path": str(manifest_path),
            "configuration": profiler._production_configuration(
                batch_size=batch_size
            ),
            "model_binding_sha256": "a" * 64,
            "source_manifest_payload_sha256": "b" * 64,
            "source_manifest_file_sha256": manifest_file_sha256,
            "model_path": str(model_path),
            "source_files_before": {
                path: dict(fingerprint) for path, fingerprint in source_files.items()
            },
            "source_files_after": {
                path: dict(fingerprint) for path, fingerprint in source_files.items()
            },
            "model_files_before": {
                path: dict(fingerprint) for path, fingerprint in model_files.items()
            },
            "model_files_after": {
                path: dict(fingerprint) for path, fingerprint in model_files.items()
            },
            "profiler_file_sha256": profiler.sha256_file(
                Path(profiler.__file__).resolve()
            ),
            "natural_runner_file_sha256": profiler.sha256_file(
                Path(profiler.gate.__file__).resolve()
            ),
            "shared_runtime_file_sha256": profiler.sha256_file(
                Path(profiler.gate.runtime.__file__).resolve()
            ),
            "delta_api_file_sha256": profiler.sha256_file(
                Path(profiler.delta_core.__file__).resolve()
            ),
            "delta_impl_file_sha256": profiler.sha256_file(
                Path(profiler.delta_impl.__file__).resolve()
            ),
            "replaced_layers": list(profiler.PRODUCTION_TARGET_LAYERS),
            "checkpointed_frozen_mlps": list(profiler.PRODUCTION_TARGET_LAYERS),
            "trainable_audit": {
                "actual_trainable_parameter_tensors": 1176,
                "allowed_trainable_parameter_tensors": 1176,
                "expected_trainable_parameter_tensors": 1176,
                "actual_trainable_names_sha256": trainable_names_sha256,
                "allowed_trainable_names_sha256": trainable_names_sha256,
                "expected_trainable_names_sha256": trainable_names_sha256,
                "only_delta_mem_parameters_trainable": True,
                "trainable_name_binding_passed": True,
                "nonempty_trainable_set": True,
                "passed": True,
            },
            "selection_policy": {
                "activation_stress": (
                    "exactly maximize total batch-padded token positions across the four "
                    "write invocations and one read invocation"
                ),
                "answer_logit_stress": (
                    "exactly maximize the union of supervised causal answer-predictor "
                    "positions projected through the vocabulary head"
                ),
            },
            "activation_selection_audit": {
                "schema": "rwkv_ms_natural_memory_gate_padded_selection.v1",
                "method": "exact constrained padded-workload maximization",
                "dimensions": [
                    "write_0",
                    "write_1",
                    "write_2",
                    "write_3",
                    "read",
                ],
                "selected": dict(selected_padded_workload),
                "unconstrained_per_dimension_upper_bound": dict(
                    selected_padded_workload
                ),
                "upper_bound_coverage_fraction": 1.0,
                "selected_batch_is_exact_constrained_optimum": True,
            },
            "answer_logit_selection_audit": {
                "schema": "rwkv_ms_natural_memory_gate_answer_logit_selection.v1",
                "method": "exact constrained answer-predictor union maximization",
                "selected_answer_predictor_union_indices": list(range(120, 128)),
                "selected_answer_predictor_union_positions": 8,
                "compact_logit_batch_position_factor": batch_size * 8,
                "sum_of_top_individual_widths_upper_bound": batch_size * 8,
                "upper_bound_coverage_fraction": 1.0,
                "selected_batch_is_exact_constrained_optimum": True,
                "selected_padded_workload": profiler._padded_workload(
                    answer_logit_profiles, batch_size
                ),
            },
            "training_examples_considered": 384,
            "activation_stress_examples": activation_profiles,
            "activation_stress_examples_sha256": profiler.sha256_text(
                profiler.canonical_json(activation_profiles)
            ),
            "answer_logit_stress_examples": answer_logit_profiles,
            "answer_logit_stress_examples_sha256": profiler.sha256_text(
                profiler.canonical_json(answer_logit_profiles)
            ),
            "adapter_state_sha256_before": "7" * 64,
            "adapter_state_sha256_after": "8" * 64,
            "adapter_changed": True,
            "phases": phases,
            "memory_gate": memory_gate,
            "execution_gate": {
                "required_optimizer_steps": (
                    profiler.PRODUCTION_PROFILE_OPTIMIZER_STEPS
                ),
                "completed_optimizer_steps": (
                    profiler.PRODUCTION_PROFILE_OPTIMIZER_STEPS
                ),
                "phase_completion": {
                    name: True for name in optimizer_phases
                },
                "error": None,
                "passed": True,
            },
            "gate_passed": True,
        },
        "worker_receipt_sha256",
    )


def _resign_worker_payload(payload: dict) -> dict:
    unsigned = dict(payload)
    unsigned.pop("worker_receipt_sha256")
    return profiler.signed_payload(unsigned, "worker_receipt_sha256")


def _as_cuda_oom_worker_payload(payload: dict) -> dict:
    phase_name = profiler.PROFILE_PHASE_NAMES[1]
    error = {
        "type": "OutOfMemoryError",
        "message": "synthetic CUDA out of memory",
        "traceback_sha256": "9" * 64,
    }
    phase = payload["phases"][phase_name]
    phase["status"] = "cuda_out_of_memory"
    phase["step"] = None
    phase["error"] = dict(error)
    for later_phase_name in profiler.PROFILE_PHASE_NAMES[2:]:
        payload["phases"][later_phase_name] = None
    payload["execution_gate"] = {
        "required_optimizer_steps": profiler.PRODUCTION_PROFILE_OPTIMIZER_STEPS,
        "completed_optimizer_steps": 0,
        "phase_completion": {
            name: False for name, _, _ in profiler.PROFILE_STRESS_SEQUENCE
        },
        "error": dict(error),
        "passed": False,
    }
    payload["memory_gate"] = profiler.build_memory_gate(
        initial_snapshot=payload["phases"]["fresh_model_load"]["before"],
        phases=[payload["phases"]["fresh_model_load"], phase],
    )
    payload["status"] = "failed"
    payload["gate_passed"] = False
    return _resign_worker_payload(payload)


def _as_headroom_failure_worker_payload(payload: dict) -> dict:
    phase = payload["phases"][profiler.PROFILE_PHASE_NAMES[-1]]
    phase["after"] = {
        "allocated_bytes": 34 * GIB,
        "total_bytes": 40 * GIB,
        "free_bytes": 4 * GIB,
        "reserved_bytes": 35 * GIB,
    }
    phase["peak_allocated_bytes"] = 34 * GIB
    phase["peak_reserved_bytes"] = 35 * GIB
    payload["memory_gate"] = profiler.build_memory_gate(
        initial_snapshot=payload["phases"]["fresh_model_load"]["before"],
        phases=[payload["phases"][name] for name in profiler.PROFILE_PHASE_NAMES],
    )
    assert payload["memory_gate"]["headroom_passed"] is False
    payload["status"] = "failed"
    payload["gate_passed"] = False
    return _resign_worker_payload(payload)


def _assert_strict_worker_rejection(result: dict, failed_check: str) -> None:
    assert result["worker_evidence_passed"] is False
    assert failed_check in result["worker_evidence_failed_checks"]
    assert result["gate_passed"] is False


def _invocation_evidence(
    tmp_path: Path,
    payload: dict,
    *,
    batch_size: int = 4,
    subprocess_pid: int = 1004,
    subprocess_returncode: int = 0,
    command_matches: bool = True,
) -> dict:
    result_path = tmp_path / "result.json"
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    expected_command = profiler.build_worker_command(
        source_manifest=Path(payload["source_manifest_path"]),
        worker_output=result_path,
        batch_size=batch_size,
        device_name="cuda:3",
    )
    return profiler._worker_invocation_evidence(
        batch_size=batch_size,
        result_path=result_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        completed=SimpleNamespace(
            args=(
                expected_command
                if command_matches
                else [*expected_command, "--unexpected"]
            ),
            returncode=subprocess_returncode,
            pid=subprocess_pid,
        ),
        parent_pid=999,
        device_name="cuda:3",
        expected_command=expected_command,
    )


def test_worker_invocation_rejects_spoofed_subprocess_pid(tmp_path: Path) -> None:
    payload = _fake_worker_payload(4, 2004, "c" * 64)

    result = _invocation_evidence(tmp_path, payload, subprocess_pid=1004)

    assert result["fresh_process_isolation_passed"] is False
    assert result["gate_passed"] is False


def test_worker_invocation_rejects_failed_nested_memory_gate(
    tmp_path: Path,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    payload["memory_gate"]["headroom_passed"] = False
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    assert result["memory_headroom_passed"] is False
    assert result["gate_passed"] is False


def test_worker_invocation_recomputes_memory_gate_from_phases(
    tmp_path: Path,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    payload["phases"]["fresh_model_load"]["peak_reserved_bytes"] = 38 * GIB
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    assert result["memory_gate_recomputation_passed"] is False
    assert result["gate_passed"] is False


@pytest.mark.parametrize("drift", ["command", "endpoint", "device"])
def test_worker_invocation_rejects_launch_binding_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    if drift == "endpoint":
        payload["hf_endpoint"] = "https://huggingface.co"
    elif drift == "device":
        payload["device"]["requested"] = "cuda:2"
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(
        tmp_path,
        payload,
        command_matches=drift != "command",
    )

    assert result["gate_passed"] is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("dtype", "float16"),
        ("learning_rate", 1e-3),
        ("optimizer_fused", False),
        ("training_conditions", ["correct_state", "donor_state"]),
    ],
)
def test_worker_invocation_rejects_production_configuration_drift(
    tmp_path: Path,
    field: str,
    invalid_value,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    payload["configuration"][field] = invalid_value
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    assert result["configuration_passed"] is False
    assert result["gate_passed"] is False


def test_worker_invocation_rejects_delta_configuration_drift(
    tmp_path: Path,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    payload["configuration"]["delta_mem_config"]["rank"] = 16
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    assert result["configuration_passed"] is False
    assert result["gate_passed"] is False


def test_worker_invocation_accepts_complete_production_receipt(
    tmp_path: Path,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)

    result = _invocation_evidence(tmp_path, payload)

    assert result["device_binding_passed"] is True
    assert result["memory_gate_recomputation_passed"] is True
    assert result["execution_passed"] is True
    assert result["worker_evidence_failed_checks"] == []
    assert result["worker_evidence_passed"] is True
    assert result["gate_passed"] is True


@pytest.mark.parametrize("returncode", [1, 124])
def test_worker_invocation_rejects_nonzero_exit_with_passing_receipt(
    tmp_path: Path,
    returncode: int,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)

    result = _invocation_evidence(
        tmp_path,
        payload,
        subprocess_returncode=returncode,
    )

    assert result["worker_evidence_passed"] is True
    assert result["subprocess_returncode"] == returncode
    assert result["gate_passed"] is False


@pytest.mark.parametrize("phase_change", ["missing", "extra"])
def test_worker_invocation_rejects_inexact_phase_set(
    tmp_path: Path,
    phase_change: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    if phase_change == "missing":
        del payload["phases"][profiler.PROFILE_PHASE_NAMES[-1]]
    else:
        payload["phases"]["unexpected_optimizer_step"] = dict(
            payload["phases"][profiler.PROFILE_PHASE_NAMES[-1]]
        )
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "phase_set")


def test_worker_invocation_rejects_failed_optimizer_phase(tmp_path: Path) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    phase = payload["phases"][profiler.PROFILE_PHASE_NAMES[2]]
    phase["status"] = "cuda_out_of_memory"
    phase["step"] = None
    phase["error"] = {
        "type": "OutOfMemoryError",
        "message": "synthetic OOM",
        "traceback_sha256": "9" * 64,
    }
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "phase_status")


@pytest.mark.parametrize("peak_kind", ["allocated", "reserved"])
def test_worker_invocation_rejects_after_snapshot_above_phase_peak(
    tmp_path: Path,
    peak_kind: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    load_phase = payload["phases"]["fresh_model_load"]
    load_phase["after"][f"{peak_kind}_bytes"] = (
        load_phase[f"peak_{peak_kind}_bytes"] + 1
    )
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "phase_memory_evidence")


def test_worker_invocation_rejects_contradictory_phase_completion(
    tmp_path: Path,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    optimizer_phase = profiler.PROFILE_PHASE_NAMES[1]
    payload["execution_gate"]["phase_completion"][optimizer_phase] = False
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "execution_gate")


@pytest.mark.parametrize("step_change", ["missing_field", "noncompact_shape"])
def test_worker_invocation_rejects_incomplete_optimizer_step_evidence(
    tmp_path: Path,
    step_change: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    step = payload["phases"][profiler.PROFILE_PHASE_NAMES[1]]["step"]
    if step_change == "missing_field":
        del step["gradient_norm"]
    else:
        step["answer_logit_shape"][1] = step["answer_predictor_positions"] + 1
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "optimizer_step_evidence")


@pytest.mark.parametrize(
    ("artifact_kind", "snapshot_change"),
    [
        ("source", "missing"),
        ("source", "changed"),
        ("model", "missing"),
        ("model", "changed"),
    ],
)
def test_worker_invocation_rejects_missing_or_changed_immutable_snapshots(
    tmp_path: Path,
    artifact_kind: str,
    snapshot_change: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    before_field = f"{artifact_kind}_files_before"
    after_field = f"{artifact_kind}_files_after"
    if snapshot_change == "missing":
        del payload[before_field]
    else:
        first_path = next(iter(payload[after_field]))
        payload[after_field][first_path]["sha256"] = "9" * 64
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "immutable_snapshots")


def test_worker_invocation_rejects_unchanged_adapter(tmp_path: Path) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    payload["adapter_state_sha256_after"] = payload[
        "adapter_state_sha256_before"
    ]
    payload["adapter_changed"] = False
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "adapter_change")


@pytest.mark.parametrize("audit_change", ["failed_flag", "name_hash_drift"])
def test_worker_invocation_rejects_failed_trainable_boundary(
    tmp_path: Path,
    audit_change: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    if audit_change == "failed_flag":
        payload["trainable_audit"]["only_delta_mem_parameters_trainable"] = False
        payload["trainable_audit"]["passed"] = False
    else:
        payload["trainable_audit"]["actual_trainable_names_sha256"] = "9" * 64
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    _assert_strict_worker_rejection(result, "trainable_boundary")


@pytest.mark.parametrize("device_change", ["index", "total_memory"])
def test_worker_invocation_rejects_device_identity_drift(
    tmp_path: Path,
    device_change: str,
) -> None:
    payload = _fake_worker_payload(4, 1004, "c" * 64)
    if device_change == "index":
        payload["device"]["index"] = 2
    else:
        payload["device"]["reported_total_memory_bytes"] = 39 * GIB
    payload = _resign_worker_payload(payload)

    result = _invocation_evidence(tmp_path, payload)

    assert result["device_binding_passed"] is False
    _assert_strict_worker_rejection(result, "device_evidence")


def test_orchestrator_binds_three_distinct_worker_processes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_file_sha256 = profiler.sha256_file(manifest)
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(
        profiler,
        "_parent_protocol_bindings",
        lambda manifest_path: {
            "source_manifest_payload_sha256": "b" * 64,
            "model_binding_sha256": "a" * 64,
        },
    )

    def fake_runner(*, source_manifest, worker_output, batch_size, device_name):
        calls.append((batch_size, device_name))
        worker_output.write_text(
            json.dumps(
                _fake_worker_payload(
                    batch_size,
                    1000 + batch_size,
                    manifest_file_sha256,
                    manifest_path=source_manifest,
                )
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            args=profiler.build_worker_command(
                source_manifest=source_manifest,
                worker_output=worker_output,
                batch_size=batch_size,
                device_name=device_name,
            ),
            returncode=0,
            stdout="ok\n",
            stderr="",
            pid=1000 + batch_size,
        )

    result = profiler.run_orchestrator(
        source_manifest=manifest,
        output_dir=tmp_path / "profile",
        device_name="cuda:3",
        worker_runner=fake_runner,
    )

    assert calls == [(1, "cuda:3"), (2, "cuda:3"), (4, "cuda:3")]
    receipt = result["receipt"]
    assert receipt["gate_passed"] is True
    assert receipt["gate"]["observed_local_batch_sizes"] == [1, 2, 4]
    assert receipt["gate"]["launch_gate"]["passed"] is True
    assert receipt["gate"]["exploration"] == {
        "complete": True,
        "outcomes_by_local_batch_size": {"2": "passed", "4": "passed"},
        "passing_local_batch_sizes": [2, 4],
        "cuda_oom_local_batch_sizes": [],
        "insufficient_headroom_local_batch_sizes": [],
        "malformed_or_unclassified_local_batch_sizes": [],
    }
    assert all(
        worker["worker_evidence_passed"] is True
        for worker in receipt["workers"]
    )
    assert profiler.verify_signed_payload(receipt, "profile_receipt_sha256")
    assert [worker["subprocess_pid"] for worker in receipt["workers"]] == [
        1001,
        1002,
        1004,
    ]
    assert Path(result["receipt_path"]).is_file()


def _run_fake_orchestrator(
    tmp_path: Path,
    monkeypatch,
    transform,
) -> dict:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_file_sha256 = profiler.sha256_file(manifest)
    monkeypatch.setattr(
        profiler,
        "_parent_protocol_bindings",
        lambda manifest_path: {
            "source_manifest_payload_sha256": "b" * 64,
            "model_binding_sha256": "a" * 64,
        },
    )

    def fake_runner(*, source_manifest, worker_output, batch_size, device_name):
        payload = _fake_worker_payload(
            batch_size,
            1000 + batch_size,
            manifest_file_sha256,
            manifest_path=source_manifest,
        )
        payload, returncode = transform(batch_size, payload)
        worker_output.write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        return SimpleNamespace(
            args=profiler.build_worker_command(
                source_manifest=source_manifest,
                worker_output=worker_output,
                batch_size=batch_size,
                device_name=device_name,
            ),
            returncode=returncode,
            stdout="",
            stderr="",
            pid=1000 + batch_size,
        )

    return profiler.run_orchestrator(
        source_manifest=manifest,
        output_dir=tmp_path / "profile",
        device_name="cuda:3",
        worker_runner=fake_runner,
    )["receipt"]


def test_signed_exploratory_oom_receipts_do_not_fail_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def transform(batch_size: int, payload: dict) -> tuple[dict, int]:
        if batch_size in profiler.EXPLORATORY_LOCAL_BATCH_SIZES:
            return _as_cuda_oom_worker_payload(payload), 1
        return payload, 0

    receipt = _run_fake_orchestrator(tmp_path, monkeypatch, transform)

    assert profiler.verify_signed_payload(receipt, "profile_receipt_sha256")
    assert all(
        worker["cuda_oom_evidence_passed"] is True
        for worker in receipt["workers"][1:]
    )
    assert receipt["gate_passed"] is True
    assert receipt["gate"]["launch_gate"]["passed"] is True
    assert receipt["gate"]["exploration"] == {
        "complete": True,
        "outcomes_by_local_batch_size": {
            "2": "cuda_out_of_memory",
            "4": "cuda_out_of_memory",
        },
        "passing_local_batch_sizes": [],
        "cuda_oom_local_batch_sizes": [2, 4],
        "insufficient_headroom_local_batch_sizes": [],
        "malformed_or_unclassified_local_batch_sizes": [],
    }


@pytest.mark.parametrize(
    "evidence_change",
    ["phase_order", "phase_tail", "execution_gate", "error", "memory"],
)
def test_malformed_exploratory_oom_evidence_fails_overall_gate(
    tmp_path: Path,
    monkeypatch,
    evidence_change: str,
) -> None:
    def transform(batch_size: int, payload: dict) -> tuple[dict, int]:
        if batch_size != 2:
            return payload, 0
        payload = _as_cuda_oom_worker_payload(payload)
        if evidence_change == "phase_order":
            failed_phase = payload["phases"][profiler.PROFILE_PHASE_NAMES[1]]
            payload["phases"][profiler.PROFILE_PHASE_NAMES[1]] = None
            payload["phases"][profiler.PROFILE_PHASE_NAMES[2]] = failed_phase
        elif evidence_change == "phase_tail":
            payload["phases"][profiler.PROFILE_PHASE_NAMES[2]] = dict(
                _fake_worker_payload(2, 1002, "c" * 64)["phases"][
                    profiler.PROFILE_PHASE_NAMES[2]
                ]
            )
        elif evidence_change == "execution_gate":
            payload["execution_gate"]["completed_optimizer_steps"] = 1
        elif evidence_change == "error":
            payload["execution_gate"]["error"]["message"] = "different error"
        else:
            failed_phase = payload["phases"][profiler.PROFILE_PHASE_NAMES[1]]
            failed_phase["peak_reserved_bytes"] = (
                failed_phase["after"]["reserved_bytes"] - 1
            )
        return _resign_worker_payload(payload), 1

    receipt = _run_fake_orchestrator(tmp_path, monkeypatch, transform)

    worker = receipt["workers"][1]
    assert worker["cuda_out_of_memory_observed"] is True
    assert worker["cuda_oom_evidence_passed"] is False
    assert receipt["gate"]["launch_gate"]["passed"] is True
    assert receipt["gate"]["exploration"][
        "malformed_or_unclassified_local_batch_sizes"
    ] == [2]
    assert receipt["gate"]["exploration"]["complete"] is False
    assert receipt["gate_passed"] is False
    assert receipt["status"] == "failed"


def test_signed_exploratory_headroom_failure_does_not_fail_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def transform(batch_size: int, payload: dict) -> tuple[dict, int]:
        if batch_size == 2:
            return _as_headroom_failure_worker_payload(payload), 1
        return payload, 0

    receipt = _run_fake_orchestrator(tmp_path, monkeypatch, transform)

    headroom_worker = receipt["workers"][1]
    assert headroom_worker["profiled_local_batch_size"] == 2
    assert headroom_worker["result_valid"] is True
    assert headroom_worker["cuda_out_of_memory_observed"] is False
    assert headroom_worker["memory_gate_recomputation_passed"] is True
    assert headroom_worker["memory_headroom_passed"] is False
    assert headroom_worker["execution_passed"] is True
    assert headroom_worker["gate_passed"] is False
    assert receipt["gate"]["launch_gate"]["passed"] is True
    assert receipt["gate_passed"] is True
    exploration = receipt["gate"]["exploration"]
    assert exploration["complete"] is True
    assert exploration["outcomes_by_local_batch_size"] == {
        "2": "insufficient_headroom",
        "4": "passed",
    }
    assert exploration["insufficient_headroom_local_batch_sizes"] == [2]
    assert exploration["malformed_or_unclassified_local_batch_sizes"] == []


def test_signed_required_batch_one_oom_fails_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def transform(batch_size: int, payload: dict) -> tuple[dict, int]:
        if batch_size == profiler.DISTRIBUTED_LOCAL_BATCH_SIZE:
            return _as_cuda_oom_worker_payload(payload), 1
        return payload, 0

    receipt = _run_fake_orchestrator(tmp_path, monkeypatch, transform)

    assert receipt["gate_passed"] is False
    assert receipt["status"] == "failed"
    launch = receipt["gate"]["launch_gate"]
    assert launch["passed"] is False
    assert "required_worker_invocation_binding" in launch["failed_checks"]
    assert "required_memory_headroom" in launch["failed_checks"]
    assert "required_optimizer_execution" in launch["failed_checks"]
    assert "required_worker_gate" in launch["failed_checks"]


@pytest.mark.parametrize("invalidity", ["signature", "artifact_binding"])
def test_invalid_exploratory_receipt_is_a_separate_incomplete_audit(
    tmp_path: Path,
    monkeypatch,
    invalidity: str,
) -> None:
    def transform(batch_size: int, payload: dict) -> tuple[dict, int]:
        if batch_size != 2:
            return payload, 0
        if invalidity == "signature":
            payload["worker_receipt_sha256"] = "0" * 64
        else:
            payload["delta_impl_file_sha256"] = "0" * 64
            payload = _resign_worker_payload(payload)
        return payload, 0

    receipt = _run_fake_orchestrator(tmp_path, monkeypatch, transform)

    assert receipt["gate_passed"] is False
    assert receipt["status"] == "failed"
    assert receipt["gate"]["launch_gate"]["passed"] is True
    assert "exploration_complete" in receipt["gate"]["failed_checks"]
    exploration = receipt["gate"]["exploration"]
    assert exploration["complete"] is False
    assert exploration["outcomes_by_local_batch_size"] == {
        "2": "malformed_or_unclassified",
        "4": "passed",
    }
    assert exploration["malformed_or_unclassified_local_batch_sizes"] == [2]


def test_main_returns_nonzero_for_incomplete_exploratory_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt = {
        "gate_passed": False,
        "gate": {
            "launch_gate": {"passed": True},
            "exploration": {"complete": False},
        },
    }
    monkeypatch.setattr(
        profiler,
        "run_orchestrator",
        lambda **kwargs: {
            "output_dir": str(tmp_path / "profile"),
            "receipt_path": str(tmp_path / "profile" / "profile_receipt.json"),
            "receipt": receipt,
        },
    )

    returncode = profiler.main(
        [
            "--source-manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "profile"),
            "--device",
            "cuda:3",
        ]
    )

    assert returncode == 1


def _valid_profile_workers() -> list[dict]:
    workers = []
    for batch_size in profiler.PROFILED_LOCAL_BATCH_SIZES:
        workers.append(
            {
                "profiled_local_batch_size": batch_size,
                "result_valid": True,
                "status": "passed",
                "subprocess_returncode": 0,
                "subprocess_pid": 1000 + batch_size,
                "fresh_process_isolation_passed": True,
                "worker_command_passed": True,
                "source_manifest_path_passed": True,
                "hf_endpoint_passed": True,
                "device_binding_passed": True,
                "cuda_out_of_memory_observed": False,
                "cuda_oom_evidence_passed": False,
                "worker_evidence_passed": True,
                "worker_evidence_failed_checks": [],
                "worker_evidence_checks": {
                    "immutable_snapshots": True,
                    "trainable_boundary": True,
                },
                "model_binding_sha256": "a" * 64,
                "source_manifest_payload_sha256": "b" * 64,
                "source_manifest_file_sha256": "c" * 64,
                "profiler_file_sha256": "d" * 64,
                "natural_runner_file_sha256": "e" * 64,
                "shared_runtime_file_sha256": "f" * 64,
                "delta_api_file_sha256": "1" * 64,
                "delta_impl_file_sha256": "2" * 64,
                "memory_headroom_passed": True,
                "memory_gate_recomputation_passed": True,
                "execution_passed": True,
                "configuration_passed": True,
                "gate_passed": True,
            }
        )
    return workers


def test_distributed_profile_target_locks_four_gpu_global_batch_four() -> None:
    target = profiler._distributed_training_target()
    workers = _valid_profile_workers()

    result = profiler.build_profile_gate(workers)

    assert target["world_size"] == 4
    assert target["local_batch_size"] == 1
    assert target["global_batch_size"] == 4
    assert target["world_size"] * target["local_batch_size"] == target[
        "global_batch_size"
    ]
    assert result["launch_gate"]["selected_world_size"] == 4
    assert result["launch_gate"]["selected_local_batch_size"] == 1
    assert result["launch_gate"]["selected_global_batch_size"] == 4
    assert result["passed"] is True


@pytest.mark.parametrize("evidence_change", ["missing", "duplicate", "reordered"])
def test_profile_gate_fails_closed_on_inexact_exploratory_evidence(
    evidence_change: str,
) -> None:
    workers = _valid_profile_workers()
    if evidence_change == "missing":
        workers = [
            worker
            for worker in workers
            if worker["profiled_local_batch_size"] != 2
        ]
    elif evidence_change == "duplicate":
        duplicate = dict(workers[1])
        duplicate["subprocess_pid"] = 2002
        workers.insert(2, duplicate)
    else:
        workers[1:] = reversed(workers[1:])

    result = profiler.build_profile_gate(workers)

    assert result["launch_gate"]["passed"] is True
    assert result["profile_set_complete"] is False
    assert result["exploration"]["complete"] is False
    assert "exploration_complete" in result["failed_checks"]
    assert result["passed"] is False


@pytest.mark.parametrize(
    "binding_field",
    [
        "model_binding_sha256",
        "shared_runtime_file_sha256",
        "delta_impl_file_sha256",
    ],
)
def test_profile_gate_rejects_cross_worker_binding_drift(
    binding_field: str,
) -> None:
    workers = _valid_profile_workers()
    workers[0][binding_field] = "9" * 64

    result = profiler.build_profile_gate(
        workers,
        expected_bindings={binding_field: "a" * 64}
        if binding_field == "model_binding_sha256"
        else {binding_field: workers[1][binding_field]},
    )

    assert result["required_artifact_binding_passed"] is False
    assert result["launch_gate"]["passed"] is False
    assert result["passed"] is False
    assert "required_artifact_binding" in result["failed_checks"]


def test_profile_gate_rejects_missing_worker_binding() -> None:
    workers = _valid_profile_workers()
    del workers[0]["delta_api_file_sha256"]

    result = profiler.build_profile_gate(workers)

    assert result["required_artifact_binding_passed"] is False
    assert result["launch_gate"]["passed"] is False
    assert result["passed"] is False
    assert "required_artifact_binding" in result["failed_checks"]


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_orchestrator_emits_failed_receipt_when_worker_cannot_launch(
    tmp_path: Path,
    error_type: type[Exception],
    monkeypatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        profiler,
        "_parent_protocol_bindings",
        lambda manifest_path: {
            "source_manifest_payload_sha256": "b" * 64,
            "model_binding_sha256": "a" * 64,
        },
    )

    def failed_runner(**kwargs):
        raise error_type("worker executable is unavailable")

    result = profiler.run_orchestrator(
        source_manifest=manifest,
        output_dir=tmp_path / "failed-profile",
        device_name="cuda:3",
        worker_runner=failed_runner,
    )

    assert result["receipt"]["gate_passed"] is False
    assert result["receipt"]["status"] == "failed"
    assert profiler.verify_signed_payload(
        result["receipt"], "profile_receipt_sha256"
    )
    assert all(
        worker["subprocess_returncode"] == 127
        for worker in result["receipt"]["workers"]
    )
