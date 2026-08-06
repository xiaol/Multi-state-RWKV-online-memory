from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as runner


def test_torchrun_environment_is_absent_complete_or_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in distributed.TORCHRUN_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    assert distributed.torchrun_environment() is None

    monkeypatch.setenv("RANK", "2")
    with pytest.raises(ValueError, match="missing LOCAL_RANK, WORLD_SIZE"):
        distributed.torchrun_environment()

    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert distributed.torchrun_environment() == {
        "RANK": 2,
        "LOCAL_RANK": 2,
        "WORLD_SIZE": 4,
    }


def test_global_schedule_is_deterministic_complete_and_disjoint() -> None:
    row_ids = [f"row-{index:03d}" for index in range(384)]
    schedule, schedule_hash = distributed.build_global_training_schedule(
        row_ids,
        seed=42,
        epochs=8,
        max_steps=768,
        world_size=4,
        local_batch_size=1,
    )
    repeated, repeated_hash = distributed.build_global_training_schedule(
        row_ids,
        seed=42,
        epochs=8,
        max_steps=768,
        world_size=4,
        local_batch_size=1,
    )

    assert schedule == repeated
    assert schedule_hash == repeated_hash
    assert len(schedule) == 768
    assert schedule[0].step == 1
    assert schedule[-1].step == 768
    assert schedule[0].epoch == 0
    assert schedule[-1].epoch == 7
    assert all(len(step.global_indices) == 4 for step in schedule)
    for step in schedule:
        rank_indices = [
            distributed.local_step_indices(
                step,
                process_rank=rank,
                world_size=4,
                local_batch_size=1,
            )
            for rank in range(4)
        ]
        assert tuple(index for indices in rank_indices for index in indices) == (
            step.global_indices
        )
        assert len({indices[0] for indices in rank_indices}) == 4
    for epoch in range(8):
        epoch_indices = [
            index
            for step in schedule
            if step.epoch == epoch
            for index in step.global_indices
        ]
        assert sorted(epoch_indices) == list(range(384))


def test_global_schedule_rejects_partial_or_underfilled_runs() -> None:
    with pytest.raises(ValueError, match="complete distributed global batches"):
        distributed.build_global_training_schedule(
            [f"row-{index}" for index in range(6)],
            seed=1,
            epochs=1,
            max_steps=None,
            world_size=4,
            local_batch_size=1,
        )
    with pytest.raises(ValueError, match="epochs provide only 2"):
        distributed.build_global_training_schedule(
            [f"row-{index}" for index in range(8)],
            seed=1,
            epochs=1,
            max_steps=3,
            world_size=4,
            local_batch_size=1,
        )


def _objective_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    answer_logits = torch.randn(4, 8, 11, generator=generator)
    labels = torch.full((4, 8), -100, dtype=torch.long)
    answer_lengths = (1, 2, 4, 7)
    for row, length in enumerate(answer_lengths):
        labels[row, 1 : length + 1] = torch.arange(length) % 11
    route_logits = torch.randn(4, 8, 4, generator=generator)
    query_mask = torch.zeros(4, 8, dtype=torch.bool)
    query_mask[:, :2] = True
    return answer_logits, labels, route_logits, query_mask


def test_sharded_objective_and_gradients_match_monolithic_reference() -> None:
    answer_logits, labels, route_logits, query_mask = _objective_inputs()
    targets = torch.tensor([0, 1, 2, 3])
    monolithic_answer = answer_logits.clone().requires_grad_(True)
    monolithic_route = route_logits.clone().requires_grad_(True)
    answer_sum, answer_count = distributed.answer_loss_sum_and_count(
        monolithic_answer, labels
    )
    route_sum, route_count, _ = distributed.route_loss_sum_and_predictions(
        {"layer": monolithic_route}, query_mask, targets
    )
    monolithic_loss = answer_sum / answer_count + route_sum / route_count
    monolithic_loss.backward()

    sharded_answer_gradients = []
    sharded_route_gradients = []
    sharded_loss = 0.0
    for row in range(4):
        local_answer = answer_logits[row : row + 1].clone().requires_grad_(True)
        local_route = route_logits[row : row + 1].clone().requires_grad_(True)
        local_answer_sum, local_answer_count = distributed.answer_loss_sum_and_count(
            local_answer, labels[row : row + 1]
        )
        local_route_sum, local_route_count, _ = (
            distributed.route_loss_sum_and_predictions(
                {"layer": local_route},
                query_mask[row : row + 1],
                targets[row : row + 1],
            )
        )
        assert local_answer_count == (1, 2, 4, 7)[row]
        assert local_route_count == 1
        local_loss = local_answer_sum / answer_count + local_route_sum / route_count
        local_loss.backward()
        sharded_loss += float(local_loss.detach().item())
        sharded_answer_gradients.append(local_answer.grad)
        sharded_route_gradients.append(local_route.grad)

    assert sharded_loss == pytest.approx(float(monolithic_loss.detach().item()))
    assert torch.allclose(
        torch.cat(sharded_answer_gradients), monolithic_answer.grad, atol=1e-6
    )
    assert torch.allclose(
        torch.cat(sharded_route_gradients), monolithic_route.grad, atol=1e-6
    )


def test_objective_statistics_are_validated_before_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collectives: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        assert op == distributed.dist.ReduceOp.SUM
        collectives.append(tensor.clone())
        tensor.mul_(4.0)

    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)
    prepared = distributed.prepare_objective_statistics(
        answer_loss_sum=torch.tensor(6.0),
        answer_token_count=3,
        route_loss_sum=torch.tensor(2.0),
        route_row_count=1,
    )
    reduced = distributed.reduce_objective_statistics(_fake_context(), prepared)

    assert len(collectives) == 1
    assert collectives[0].tolist() == [6.0, 2.0, 3.0, 1.0]
    assert reduced == {
        "answer_loss_sum": 24.0,
        "route_loss_sum": 8.0,
        "answer_token_count": 12,
        "route_row_count": 4,
    }
    with pytest.raises(ValueError, match="Every rank must contribute"):
        distributed.prepare_objective_statistics(
            answer_loss_sum=torch.tensor(1.0),
            answer_token_count=0,
            route_loss_sum=torch.tensor(1.0),
            route_row_count=1,
        )
    with pytest.raises(ValueError, match="must be finite"):
        distributed.prepare_objective_statistics(
            answer_loss_sum=torch.tensor(float("nan")),
            answer_token_count=1,
            route_loss_sum=torch.tensor(1.0),
            route_row_count=1,
        )
    assert len(collectives) == 1


def _fake_context() -> distributed.DistributedTrainingContext:
    return distributed.DistributedTrainingContext(
        process_rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        backend="gloo",
        control_backend="gloo",
        control_group=object(),
        rank_devices=(),
    )


def _mock_control_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, BaseException | None]]:
    phases: list[tuple[str, BaseException | None]] = []

    def fake_phase_consensus(
        context: distributed.DistributedTrainingContext,
        *,
        phase: str,
        error: BaseException | None,
    ) -> None:
        assert context.process_rank == 0
        assert context.world_size == 4
        assert context.device == torch.device("cpu")
        phases.append((phase, error))
        if error is not None:
            raise distributed.DistributedTrainingError(
                f"Distributed phase {phase!r} failed"
            )

    def fake_require_consensus(
        context: distributed.DistributedTrainingContext,
        value: Any,
        *,
        description: str,
    ) -> tuple[Any, ...]:
        assert description
        return (value,) * context.world_size

    monkeypatch.setattr(distributed, "phase_consensus", fake_phase_consensus)
    monkeypatch.setattr(distributed, "require_consensus", fake_require_consensus)
    return phases


def test_complete_adapter_broadcast_includes_frozen_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainable = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
    seen: list[torch.Tensor] = []

    def fake_broadcast(tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0
        seen.append(tensor.clone())

    _mock_control_consensus(monkeypatch)
    monkeypatch.setattr(distributed.dist, "broadcast", fake_broadcast)
    evidence = distributed.broadcast_named_parameters(
        _fake_context(),
        [("adapter.trainable", trainable), ("adapter.frozen", frozen)],
        bucket_bytes=1024,
    )

    assert evidence["parameter_tensors"] == 2
    assert evidence["broadcast_bytes"] == 12
    assert len(seen) == 1
    assert seen[0].tolist() == [3.0, 1.0, 2.0]


def test_gradient_collective_sums_before_global_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = torch.nn.Parameter(torch.tensor([0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([0.4])
    second.grad = torch.tensor([0.3])
    collectives: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        assert op == distributed.dist.ReduceOp.SUM
        collectives.append(tensor.clone())
        tensor.mul_(4.0)

    _mock_control_consensus(monkeypatch)
    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)
    evidence = distributed.sum_gradients(
        _fake_context(),
        [("second", second), ("first", first)],
    )
    pre_clip_norm = torch.nn.utils.clip_grad_norm_([first, second], max_norm=1.0)

    assert evidence["gradient_tensors"] == 2
    assert len(collectives) == 1
    assert collectives[0].tolist() == pytest.approx([0.4, 0.3])
    assert float(pre_clip_norm) == pytest.approx(2.0)
    assert float(torch.linalg.vector_norm(torch.stack([first.grad, second.grad]))) == (
        pytest.approx(1.0)
    )


def test_broadcast_preparation_failure_prevents_data_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _mock_control_consensus(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    def fail_bucket_preparation(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected bucket preparation failure")

    def forbidden_broadcast(*args: Any, **kwargs: Any) -> None:
        pytest.fail("broadcast must not run after a local preparation failure")

    monkeypatch.setattr(distributed, "_tensor_buckets", fail_bucket_preparation)
    monkeypatch.setattr(distributed.dist, "broadcast", forbidden_broadcast)

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="broadcast-named-parameters-preparation",
    ):
        distributed.broadcast_named_parameters(
            _fake_context(), [("adapter.weight", parameter)]
        )

    assert len(phases) == 1
    assert phases[0][0] == "broadcast-named-parameters-preparation"
    assert isinstance(phases[0][1], RuntimeError)


def test_gradient_flatten_failure_prevents_data_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _mock_control_consensus(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([1.0])

    def fail_flatten(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("injected flatten failure")

    def forbidden_all_reduce(*args: Any, **kwargs: Any) -> None:
        pytest.fail("all_reduce must not run after a local flatten failure")

    monkeypatch.setattr(distributed, "_flatten_collective_bucket", fail_flatten)
    monkeypatch.setattr(distributed.dist, "all_reduce", forbidden_all_reduce)

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="sum-gradients-bucket-0-flatten-readiness",
    ):
        distributed.sum_gradients(
            _fake_context(), [("adapter.weight", parameter)]
        )

    assert [phase for phase, _ in phases] == [
        "sum-gradients-preparation",
        "sum-gradients-bucket-0-flatten-readiness",
    ]
    assert phases[0][1] is None
    assert isinstance(phases[1][1], RuntimeError)


def test_postcollective_broadcast_failure_is_consensused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _mock_control_consensus(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    collective_calls = 0

    def corrupting_broadcast(tensor: torch.Tensor, *, src: int) -> None:
        nonlocal collective_calls
        assert src == 0
        collective_calls += 1
        tensor.fill_(float("nan"))

    monkeypatch.setattr(distributed.dist, "broadcast", corrupting_broadcast)

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="bucket-0-post-collective-apply",
    ):
        distributed.broadcast_named_parameters(
            _fake_context(), [("adapter.weight", parameter)]
        )

    assert collective_calls == 1
    assert [phase for phase, _ in phases] == [
        "broadcast-named-parameters-preparation",
        "broadcast-named-parameters-bucket-0-flatten-readiness",
        "broadcast-named-parameters-bucket-0-collective",
        "broadcast-named-parameters-bucket-0-post-collective-apply",
    ]
    assert isinstance(phases[-1][1], RuntimeError)


def test_postcollective_gradient_failure_is_consensused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _mock_control_consensus(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([1.0])
    collective_calls = 0

    def corrupting_all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        nonlocal collective_calls
        assert op == distributed.dist.ReduceOp.SUM
        collective_calls += 1
        tensor.fill_(float("nan"))

    monkeypatch.setattr(distributed.dist, "all_reduce", corrupting_all_reduce)

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="bucket-0-post-collective-apply",
    ):
        distributed.sum_gradients(
            _fake_context(), [("adapter.weight", parameter)]
        )

    assert collective_calls == 1
    assert [phase for phase, _ in phases] == [
        "sum-gradients-preparation",
        "sum-gradients-bucket-0-flatten-readiness",
        "sum-gradients-bucket-0-collective",
        "sum-gradients-bucket-0-post-collective-apply",
    ]
    assert isinstance(phases[-1][1], RuntimeError)


def test_gradient_validation_rejects_missing_nonfinite_and_non_fp32() -> None:
    missing = torch.nn.Parameter(torch.tensor([0.0]))
    nonfinite = torch.nn.Parameter(torch.tensor([0.0]))
    nonfinite.grad = torch.tensor([float("nan")])
    half = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float16))
    half.grad = torch.tensor([1.0], dtype=torch.float16)

    evidence = distributed.validate_local_gradients(
        [("missing", missing), ("nonfinite", nonfinite), ("half", half)]
    )

    assert evidence["passed"] is False
    assert evidence["missing"] == ["missing"]
    assert evidence["nonfinite"] == ["nonfinite"]
    assert evidence["non_fp32"] == ["half"]


def test_tensor_mapping_hash_is_stable_and_sensitive() -> None:
    value = {
        "state": {
            0: {
                "step": torch.tensor(3.0),
                "exp_avg": torch.tensor([1.0, 2.0]),
            }
        },
        "param_groups": [{"lr": 2e-4, "params": [0]}],
    }
    repeated = {
        "param_groups": [{"params": [0], "lr": 2e-4}],
        "state": {0: {"exp_avg": torch.tensor([1.0, 2.0]), "step": torch.tensor(3.0)}},
    }
    changed = replace_tensor(value, torch.tensor([1.0, 2.1]))

    assert distributed.tensor_mapping_sha256(value) == (
        distributed.tensor_mapping_sha256(repeated)
    )
    assert distributed.tensor_mapping_sha256(value) != (
        distributed.tensor_mapping_sha256(changed)
    )


def replace_tensor(value: dict, tensor: torch.Tensor) -> dict:
    return {
        "state": {0: {"step": torch.tensor(3.0), "exp_avg": tensor}},
        "param_groups": [{"lr": 2e-4, "params": [0]}],
    }


def _gloo_precollective_failure_worker(
    process_rank: int,
    world_size: int,
    initialization_path: str,
    output_dir: str,
) -> None:
    torch_dist.init_process_group(
        backend="gloo",
        init_method=f"file://{initialization_path}",
        rank=process_rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        context = distributed.DistributedTrainingContext(
            process_rank=process_rank,
            local_rank=process_rank,
            world_size=world_size,
            device=torch.device("cpu"),
            backend="gloo",
            control_backend="gloo",
            control_group=torch_dist.group.WORLD,
            rank_devices=(),
        )
        local_error: BaseException | None = None
        prepared = None
        try:
            prepared = distributed.prepare_objective_statistics(
                answer_loss_sum=torch.tensor(1.0),
                answer_token_count=0 if process_rank == 2 else 1,
                route_loss_sum=torch.tensor(1.0),
                route_row_count=1,
            )
        except BaseException as error:
            local_error = error
        collective_entered = False
        propagated = False
        message = ""
        try:
            distributed.phase_consensus(
                context,
                phase="objective-preparation",
                error=local_error,
            )
            collective_entered = True
            distributed.reduce_objective_statistics(context, prepared)
        except distributed.DistributedTrainingError as error:
            propagated = True
            message = str(error)
        Path(output_dir, f"failure-rank-{process_rank}.json").write_text(
            json.dumps(
                {
                    "rank": process_rank,
                    "collective_entered": collective_entered,
                    "propagated": propagated,
                    "message": message,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        torch_dist.destroy_process_group()


def test_rank_failure_propagates_before_objective_collective(tmp_path: Path) -> None:
    initialization_path = tmp_path / "failure-initialization"
    torch_mp.spawn(
        _gloo_precollective_failure_worker,
        args=(4, str(initialization_path), str(tmp_path)),
        nprocs=4,
        join=True,
    )
    payloads = [
        json.loads(
            (tmp_path / f"failure-rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(4)
    ]

    assert all(payload["propagated"] is True for payload in payloads)
    assert all(payload["collective_entered"] is False for payload in payloads)
    assert all("objective-preparation" in payload["message"] for payload in payloads)
    assert all('"rank":2' in payload["message"] for payload in payloads)


def _valid_preflight_training() -> dict:
    trainable_metadata = [
        {
            "name": "adapter.weight",
            "shape": [4],
            "dtype": "torch.float32",
            "requires_grad": True,
        }
    ]
    trainable_metadata_sha256 = distributed.canonical_sha256(trainable_metadata)
    trainable_names_sha256 = distributed.canonical_sha256(["adapter.weight"])
    complete_adapter_names_sha256 = distributed.canonical_sha256(
        ["adapter.frozen", "adapter.weight"]
    )
    step_evidence = []
    collective_evidence_by_step = []
    for step in range(1, runner.DISTRIBUTED_PREFLIGHT_STEPS + 1):
        adapter_hash = f"{step:064x}"
        optimizer_hash = f"{step + 16:064x}"
        global_rows = [f"step-{step}-rank-{rank}" for rank in range(4)]
        gradient_collective = {
            "gradient_tensors": 1,
            "parameter_names_sha256": trainable_names_sha256,
            "bucket_plan_sha256": f"{step + 32:064x}",
            "collective_buckets": 1,
            "all_reduce_bytes": 16,
        }
        collective_evidence_by_step.append(gradient_collective)
        ranks = [
            {
                "rank": rank,
                "local_row_ids": [global_rows[rank]],
                "local_online_state_sha256": [f"{step * 16 + rank:064x}"],
                "local_answer_tokens": rank + 1,
                "local_route_rows": 1,
                "trainable_metadata_sha256": trainable_metadata_sha256,
                "trainable_names_sha256": trainable_names_sha256,
                "gradient_validation": {
                    "parameter_tensors": 1,
                    "missing": [],
                    "nonfinite": [],
                    "non_fp32": [],
                    "passed": True,
                },
                "gradient_collective": gradient_collective,
                "adapter_state_sha256": adapter_hash,
                "optimizer_state_sha256": optimizer_hash,
            }
            for rank in range(4)
        ]
        step_evidence.append(
            {
                "step": step,
                "global_row_ids": global_rows,
                "global_answer_tokens": 10,
                "global_route_rows": 4,
                "phase_order": list(runner.DISTRIBUTED_STEP_PHASE_ORDER),
                "ranks": ranks,
                "adapter_state_sha256": adapter_hash,
                "optimizer_state_sha256": optimizer_hash,
                "trainable_metadata_sha256": trainable_metadata_sha256,
                "trainable_names_sha256": trainable_names_sha256,
            }
        )
    return {
        "steps": runner.DISTRIBUTED_PREFLIGHT_STEPS,
        "max_steps": runner.DISTRIBUTED_PREFLIGHT_STEPS,
        "adapter_changed": True,
        "progress_sha256": "a" * 64,
        "router_gradient_audit": {
            "all_ranks_all_modules_finite_nonzero": True,
        },
        "distributed": {
            "backend": "nccl",
            "control_backend": "gloo",
            "world_size": 4,
            "local_batch_size": 1,
            "global_batch_size": 4,
            "gradient_synchronization": "sum",
            "gradient_clip_order": "after_sum_before_adamw",
            "answer_loss_normalization": "global_supervised_answer_token_count",
            "route_loss_normalization": "global_row_count_after_layer_mean",
            "online_memory_state": "rank_local_never_reduced",
            "rank_devices": [
                {
                    "process_rank": rank,
                    "device_uuid": f"GPU-{rank}",
                    "pid": 1000 + rank,
                }
                for rank in range(4)
            ],
            "initialization": {
                "complete_adapter_metadata_sha256": "e" * 64,
                "complete_adapter_names_sha256": complete_adapter_names_sha256,
                "hashes_after_broadcast": ["b" * 64] * 4,
                "synchronized_adapter_state_sha256": "b" * 64,
                "broadcast": {
                    "parameter_tensors": 2,
                    "parameter_names_sha256": complete_adapter_names_sha256,
                    "bucket_plan_sha256": "f" * 64,
                    "collective_buckets": 1,
                    "broadcast_bytes": 32,
                },
            },
            "trainable_metadata": trainable_metadata,
            "trainable_metadata_sha256": trainable_metadata_sha256,
            "trainable_names_sha256": trainable_names_sha256,
            "collective_evidence": collective_evidence_by_step[-1],
            "collective_evidence_by_step": collective_evidence_by_step,
            "step_evidence": step_evidence,
            "final_adapter_state_sha256": step_evidence[-1][
                "adapter_state_sha256"
            ],
            "final_optimizer_state_sha256": step_evidence[-1][
                "optimizer_state_sha256"
            ],
            "rank_memory": [
                {
                    "after_training": {
                        "process_rank": rank,
                        "total_bytes": 40 * 1024**3,
                        "free_bytes": 18 * 1024**3,
                        "peak_reserved_bytes": 21 * 1024**3,
                    }
                }
                for rank in range(4)
            ],
            "rank_input_immutability": [
                {
                    "rank": rank,
                    "source_snapshot_sha256": "c" * 64,
                    "model_snapshot_sha256": "d" * 64,
                }
                for rank in range(4)
            ],
        },
    }


def test_distributed_preflight_gate_passes_only_complete_evidence() -> None:
    training = _valid_preflight_training()
    gate = runner.build_distributed_preflight_gate(training)

    assert gate["passed"] is True
    assert gate["failed_checks"] == []
    assert len(gate["headroom_by_rank"]) == 4
    assert all(
        value["conservative_headroom_bytes"] >= 18 * 1024**3
        for value in gate["headroom_by_rank"]
    )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda training: training["distributed"]["step_evidence"][0].update(
                {"global_answer_tokens": 9}
            ),
            "global_objective_and_row_ownership",
        ),
        (
            lambda training: [
                rank.update({"local_online_state_sha256": ["e" * 64]})
                for rank in training["distributed"]["step_evidence"][0]["ranks"]
            ],
            "global_objective_and_row_ownership",
        ),
        (
            lambda training: training["distributed"]["rank_memory"][2][
                "after_training"
            ].update({"free_bytes": 4 * 1024**3}),
            "communication_inclusive_memory_headroom",
        ),
        (
            lambda training: training["distributed"]["step_evidence"][2]["ranks"][
                1
            ].update({"optimizer_state_sha256": "f" * 64}),
            "global_objective_and_row_ownership",
        ),
    ],
)
def test_distributed_preflight_gate_fails_closed(
    mutation: Callable[[dict], Any],
    failed_check: str,
) -> None:
    training = _valid_preflight_training()
    mutation(training)
    gate = runner.build_distributed_preflight_gate(training)

    assert gate["passed"] is False
    assert failed_check in gate["failed_checks"]


def test_distributed_preflight_gate_rejects_malformed_evidence_without_crashing() -> None:
    malformed_values: list[Any] = [
        None,
        [],
        {"distributed": {"step_evidence": [None, None, None]}},
    ]
    unhashable_device = _valid_preflight_training()
    unhashable_device["distributed"]["rank_devices"][0]["device_uuid"] = []
    malformed_values.append(unhashable_device)
    malformed_local_rows = _valid_preflight_training()
    malformed_local_rows["distributed"]["step_evidence"][0]["ranks"][0][
        "local_row_ids"
    ] = 7
    malformed_values.append(malformed_local_rows)
    malformed_router_audit = _valid_preflight_training()
    malformed_router_audit["router_gradient_audit"] = ["not", "a", "mapping"]
    malformed_values.append(malformed_router_audit)

    for value in malformed_values:
        gate = runner.build_distributed_preflight_gate(value)
        assert gate["passed"] is False
        assert gate["failed_checks"]


def test_distributed_preflight_gate_binds_collectives_to_trainable_names() -> None:
    training = _valid_preflight_training()
    training["distributed"]["collective_evidence_by_step"][1][
        "parameter_names_sha256"
    ] = "9" * 64

    gate = runner.build_distributed_preflight_gate(training)

    assert gate["passed"] is False
    assert "collective_parameter_binding" in gate["failed_checks"]


def _gloo_reference_worker(
    process_rank: int,
    world_size: int,
    initialization_path: str,
    output_dir: str,
) -> None:
    torch_dist.init_process_group(
        backend="gloo",
        init_method=f"file://{initialization_path}",
        rank=process_rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        context = distributed.DistributedTrainingContext(
            process_rank=process_rank,
            local_rank=process_rank,
            world_size=world_size,
            device=torch.device("cpu"),
            backend="gloo",
            control_backend="gloo",
            control_group=torch_dist.group.WORLD,
            rank_devices=(),
        )
        weight = torch.nn.Parameter(torch.tensor([0.5 + process_rank]))
        distributed.broadcast_named_parameters(context, [("weight", weight)])
        optimizer = torch.optim.AdamW([weight], lr=0.1, weight_decay=0.0)
        x = torch.tensor([float(process_rank + 1)])
        target = 2.0 * x
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            local_loss = (weight * x - target).square().sum() / world_size
            local_loss.backward()
            distributed.sum_gradients(context, [("weight", weight)])
            torch.nn.utils.clip_grad_norm_([weight], max_norm=1.0)
            optimizer.step()
        payload = {
            "rank": process_rank,
            "pid": __import__("os").getpid(),
            "weight": float(weight.detach().item()),
            "model_sha256": distributed.tensor_mapping_sha256(
                {"weight": weight.detach()}
            ),
            "optimizer_sha256": distributed.tensor_mapping_sha256(
                optimizer.state_dict()
            ),
        }
        Path(output_dir, f"rank-{process_rank}.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    finally:
        torch_dist.destroy_process_group()


def test_real_four_process_gloo_matches_serial_three_step_reference(
    tmp_path: Path,
) -> None:
    initialization_path = tmp_path / "gloo-initialization"
    torch_mp.spawn(
        _gloo_reference_worker,
        args=(4, str(initialization_path), str(tmp_path)),
        nprocs=4,
        join=True,
    )
    rank_payloads = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(4)
    ]

    serial_weight = torch.nn.Parameter(torch.tensor([0.5]))
    serial_optimizer = torch.optim.AdamW(
        [serial_weight], lr=0.1, weight_decay=0.0
    )
    x = torch.arange(1, 5, dtype=torch.float32)
    target = 2.0 * x
    for _ in range(3):
        serial_optimizer.zero_grad(set_to_none=True)
        serial_loss = (serial_weight * x - target).square().mean()
        serial_loss.backward()
        torch.nn.utils.clip_grad_norm_([serial_weight], max_norm=1.0)
        serial_optimizer.step()

    assert len({payload["pid"] for payload in rank_payloads}) == 4
    assert len({payload["model_sha256"] for payload in rank_payloads}) == 1
    assert len({payload["optimizer_sha256"] for payload in rank_payloads}) == 1
    assert [payload["weight"] for payload in rank_payloads] == pytest.approx(
        [float(serial_weight.detach().item())] * 4,
        abs=1e-7,
    )
