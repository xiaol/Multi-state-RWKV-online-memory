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


def test_hard_negative_route_objective_matches_single_process_helper() -> None:
    _, _, route_logits, query_mask = _objective_inputs()
    targets = torch.tensor([0, 1, 2, 3])
    logits_by_layer = {
        "layer-0": route_logits,
        "layer-1": route_logits.flip(-1),
    }

    single_loss, single_predictions = runner.route_loss_and_predictions(
        logits_by_layer,
        query_mask,
        targets,
        hard_negative_margin=0.5,
        hard_negative_weight=0.1,
    )
    loss_sum, row_count, distributed_predictions = (
        distributed.route_loss_sum_and_predictions(
            logits_by_layer,
            query_mask,
            targets,
            hard_negative_margin=0.5,
            hard_negative_weight=0.1,
        )
    )

    assert torch.allclose(single_loss, loss_sum / row_count)
    assert distributed_predictions.keys() == single_predictions.keys()
    assert all(
        torch.equal(distributed_predictions[name], single_predictions[name])
        for name in single_predictions
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
        assert 0 <= context.process_rank < context.world_size
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

    def fake_gather_objects(
        context: distributed.DistributedTrainingContext,
        value: Any,
    ) -> tuple[Any, ...]:
        if isinstance(value, dict) and "active_names" in value:
            return tuple(
                dict(value, rank=rank) for rank in range(context.world_size)
            )
        return (value,) * context.world_size

    monkeypatch.setattr(distributed, "phase_consensus", fake_phase_consensus)
    monkeypatch.setattr(distributed, "require_consensus", fake_require_consensus)
    monkeypatch.setattr(distributed, "gather_objects", fake_gather_objects)
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
    assert evidence["trainable_parameter_tensors"] == 2
    assert evidence["global_active_parameter_indices"] == [0, 1]
    assert evidence["global_inactive_parameter_indices"] == []
    assert evidence["materialized_zero_gradient_tensors_by_rank"] == [0, 0, 0, 0]
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
        "sum-gradients-active-union-validation",
        "sum-gradients-zero-materialization",
        "sum-gradients-collective-preparation",
        "sum-gradients-bucket-0-flatten-readiness",
    ]
    assert phases[0][1] is None
    assert all(error is None for _, error in phases[:-1])
    assert isinstance(phases[-1][1], RuntimeError)


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
        "sum-gradients-active-union-validation",
        "sum-gradients-zero-materialization",
        "sum-gradients-collective-preparation",
        "sum-gradients-bucket-0-flatten-readiness",
        "sum-gradients-bucket-0-collective",
        "sum-gradients-bucket-0-post-collective-apply",
    ]
    assert isinstance(phases[-1][1], RuntimeError)


def test_gradient_validation_permits_missing_and_rejects_invalid_present_gradients() -> None:
    missing = torch.nn.Parameter(torch.tensor([0.0]))
    nonfinite = torch.nn.Parameter(torch.tensor([0.0]))
    nonfinite.grad = torch.tensor([float("nan")])
    half = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float16))
    half.grad = torch.tensor([1.0], dtype=torch.float16)

    evidence = distributed.validate_local_gradients(
        [("missing", missing), ("nonfinite", nonfinite), ("half", half)]
    )

    assert evidence["passed"] is False
    assert evidence["missing_gradient_tensors"] == 1
    assert evidence["nonfinite_gradient_tensors"] == 1
    assert evidence["nonfinite_preview"] == ["nonfinite"]
    assert evidence["non_fp32_gradient_tensors"] == 1
    assert evidence["non_fp32_preview"] == ["half"]

    missing_only = distributed.validate_local_gradients([("missing", missing)])
    assert missing_only["passed"] is True
    assert missing_only["active_gradient_tensors"] == 0
    assert missing_only["missing_gradient_tensors"] == 1


def test_global_inactive_gradient_remains_none_and_optimizer_skips_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = torch.nn.Parameter(torch.tensor([1.0]))
    inactive = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW([active, inactive], lr=0.1, weight_decay=0.0)
    active.grad = torch.tensor([0.5])
    inactive_before = inactive.detach().clone()

    _mock_control_consensus(monkeypatch)
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda tensor, *, op: None)
    evidence = distributed.sum_gradients(
        _fake_context(), [("inactive", inactive), ("active", active)]
    )
    optimizer.step()

    assert evidence["global_active_parameter_indices"] == [0]
    assert evidence["global_inactive_parameter_indices"] == [1]
    assert evidence["gradient_tensors"] == 1
    assert inactive.grad is None
    assert torch.equal(inactive.detach(), inactive_before)
    assert inactive not in optimizer.state


def test_rank_missing_global_active_gradient_is_zero_filled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _fake_context()
    context = distributed.DistributedTrainingContext(
        process_rank=1,
        local_rank=1,
        world_size=context.world_size,
        device=context.device,
        backend=context.backend,
        control_backend=context.control_backend,
        control_group=context.control_group,
        rank_devices=context.rank_devices,
    )
    common = torch.nn.Parameter(torch.tensor([0.0]))
    exclusive = torch.nn.Parameter(torch.tensor([0.0]))
    dormant = torch.nn.Parameter(torch.tensor([0.0]))
    common.grad = torch.tensor([0.5])
    _mock_control_consensus(monkeypatch)

    def gather_active(
        current: distributed.DistributedTrainingContext,
        value: Any,
    ) -> tuple[Any, ...]:
        del value
        assert current is context
        names_by_rank = [
            ["common", "exclusive"],
            ["common"],
            ["common"],
            ["common"],
        ]
        return tuple(
            {
                "rank": rank,
                "active_names": names,
                "active_gradient_tensors": len(names),
                "active_names_sha256": distributed.canonical_sha256(names),
            }
            for rank, names in enumerate(names_by_rank)
        )

    def sum_from_all_ranks(tensor: torch.Tensor, *, op: object) -> None:
        assert op == distributed.dist.ReduceOp.SUM
        assert tensor.tolist() == [0.5, 0.0]
        tensor.copy_(torch.tensor([2.0, 4.0]))

    monkeypatch.setattr(distributed, "gather_objects", gather_active)
    monkeypatch.setattr(distributed.dist, "all_reduce", sum_from_all_ranks)
    evidence = distributed.sum_gradients(
        context,
        [("common", common), ("exclusive", exclusive), ("dormant", dormant)],
    )

    assert common.grad.tolist() == [2.0]
    assert exclusive.grad.tolist() == [4.0]
    assert dormant.grad is None
    assert evidence["global_active_parameter_indices"] == [0, 2]
    assert evidence["global_inactive_parameter_indices"] == [1]
    assert evidence["materialized_zero_gradient_tensors_by_rank"] == [0, 1, 1, 1]


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


def _valid_preflight_training_dataset_audit() -> dict:
    conditions = list(runner.DEFAULT_TRAINING_CONDITIONS)
    tasks = list(runner.PRODUCTION_TASKS)
    rows_per_condition_task = {
        condition: {
            task: runner.PRODUCTION_ROWS_PER_CONDITION_TASK for task in tasks
        }
        for condition in conditions
    }
    audit = {
        "schema": runner.TRAINING_DATASET_AUDIT_SCHEMA,
        "training_conditions": conditions,
        "tasks": tasks,
        "rows": runner.PRODUCTION_TRAINING_ROWS,
        "unique_row_ids": True,
        "row_id_policy": runner.TRAINING_ROW_ID_POLICY,
        "row_id_policy_passed": True,
        "sampling_policy": runner.TRAINING_SAMPLING_POLICY,
        "payload_digest_policy": runner.TRAINING_PAYLOAD_DIGEST_POLICY,
        "family_invariant_policy": runner.TRAINING_FAMILY_INVARIANT_POLICY,
        "condition_set_exact": True,
        "condition_task_strata_exact": True,
        "condition_task_strata_balanced": True,
        "rows_per_condition_task": rows_per_condition_task,
        "answer_tokens_per_condition_task": rows_per_condition_task,
        "rows_by_condition": {
            condition: runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
            for condition in conditions
        },
        "rows_by_task": {
            task: runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(conditions)
            for task in tasks
        },
        "source_query_condition_families": (
            runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
        ),
        "complete_source_query_condition_families": (
            runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
        ),
        "paired_condition_coverage": True,
        "family_invariants_passed": True,
        "family_invariant_failure_count": 0,
        "training_row_id_set_sha256": "1" * 64,
        "ordered_training_examples_sha256": "2" * 64,
        "passed": True,
    }
    return runner.bind_production_training_contract(
        audit,
        epochs=runner.PRODUCTION_EPOCHS,
        global_batch_size=runner.distributed.REQUIRED_GLOBAL_BATCH_SIZE,
        requested_max_steps=runner.DISTRIBUTED_PREFLIGHT_STEPS,
        schedule_mode="preflight",
    )


def _valid_preflight_training() -> dict:
    trainable_metadata = [
        {
            "name": name,
            "shape": [4],
            "dtype": "torch.float32",
            "requires_grad": True,
        }
        for name in ("adapter.active", "adapter.dormant", "adapter.rank_only")
    ]
    trainable_metadata_sha256 = distributed.canonical_sha256(trainable_metadata)
    trainable_names = [value["name"] for value in trainable_metadata]
    trainable_names_sha256 = distributed.canonical_sha256(trainable_names)
    global_active_names = ["adapter.active", "adapter.rank_only"]
    global_inactive_names = ["adapter.dormant"]
    active_names_by_rank = [
        global_active_names,
        ["adapter.active"],
        ["adapter.active"],
        ["adapter.active"],
    ]
    complete_adapter_names_sha256 = distributed.canonical_sha256(
        ["adapter.active", "adapter.dormant", "adapter.frozen", "adapter.rank_only"]
    )
    step_evidence = []
    collective_evidence_by_step = []
    for step in range(1, runner.DISTRIBUTED_PREFLIGHT_STEPS + 1):
        adapter_hash = f"{step:064x}"
        optimizer_hash = f"{step + 16:064x}"
        global_rows = [f"step-{step}-rank-{rank}" for rank in range(4)]
        gradient_collective = {
            "trainable_parameter_tensors": len(trainable_names),
            "trainable_names_sha256": trainable_names_sha256,
            "gradient_tensors": len(global_active_names),
            "global_active_parameter_indices": [0, 2],
            "global_active_names_sha256": distributed.canonical_sha256(
                global_active_names
            ),
            "global_inactive_parameter_indices": [1],
            "global_inactive_names_sha256": distributed.canonical_sha256(
                global_inactive_names
            ),
            "per_rank_active_gradients": [
                {
                    "rank": rank,
                    "active_gradient_tensors": len(names),
                    "active_names_sha256": distributed.canonical_sha256(names),
                }
                for rank, names in enumerate(active_names_by_rank)
            ],
            "materialized_zero_gradient_tensors_by_rank": [0, 1, 1, 1],
            "bucket_plan_sha256": f"{step + 32:064x}",
            "collective_buckets": 1,
            "all_reduce_bytes": 32,
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
                    "parameter_tensors": len(trainable_names),
                    "parameter_names_sha256": trainable_names_sha256,
                    "active_gradient_tensors": len(active_names_by_rank[rank]),
                    "active_names_sha256": distributed.canonical_sha256(
                        active_names_by_rank[rank]
                    ),
                    "missing_gradient_tensors": (
                        len(trainable_names) - len(active_names_by_rank[rank])
                    ),
                    "missing_names_sha256": distributed.canonical_sha256(
                        [
                            name
                            for name in trainable_names
                            if name not in set(active_names_by_rank[rank])
                        ]
                    ),
                    "nonfinite_gradient_tensors": 0,
                    "nonfinite_names_sha256": distributed.canonical_sha256([]),
                    "nonfinite_preview": [],
                    "non_fp32_gradient_tensors": 0,
                    "non_fp32_names_sha256": distributed.canonical_sha256([]),
                    "non_fp32_preview": [],
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
        "training_dataset_audit": _valid_preflight_training_dataset_audit(),
        "distributed": {
            "backend": "nccl",
            "control_backend": "gloo",
            "world_size": 4,
            "local_batch_size": 1,
            "global_batch_size": 4,
            "gradient_synchronization": "sum",
            "unused_gradient_policy": (
                "global_active_union_zero_fill_rank_missing_skip_global_inactive"
            ),
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
                    "parameter_tensors": 4,
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
        (
            lambda training: training["training_dataset_audit"].update(
                {
                    "rows": 4,
                    "passed": True,
                    "production_contract_passed": True,
                }
            ),
            "compositional_training_dataset",
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
        "trainable_names_sha256"
    ] = "9" * 64

    gate = runner.build_distributed_preflight_gate(training)

    assert gate["passed"] is False
    assert "collective_parameter_binding" in gate["failed_checks"]


def test_distributed_preflight_gate_resolves_active_indices_against_metadata() -> None:
    training = _valid_preflight_training()
    training["distributed"]["collective_evidence_by_step"][1][
        "global_active_parameter_indices"
    ] = [0, 1]
    training["distributed"]["collective_evidence_by_step"][1][
        "global_inactive_parameter_indices"
    ] = [2]

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
        rank_only = torch.nn.Parameter(torch.tensor([0.25 + process_rank]))
        dormant = torch.nn.Parameter(torch.tensor([-0.75 + process_rank]))
        named_parameters = [
            ("dormant", dormant),
            ("rank_only", rank_only),
            ("weight", weight),
        ]
        distributed.broadcast_named_parameters(context, named_parameters)
        optimizer = torch.optim.AdamW(
            [weight, rank_only, dormant], lr=0.1, weight_decay=0.0
        )
        x = torch.tensor([float(process_rank + 1)])
        target = 2.0 * x
        collective_hashes = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            local_loss = (weight * x - target).square().sum() / world_size
            if process_rank == 0:
                local_loss = local_loss + (rank_only - 1.0).square().sum() / world_size
            local_loss.backward()
            collective = distributed.sum_gradients(context, named_parameters)
            collective_hashes.append(distributed.canonical_sha256(collective))
            torch.nn.utils.clip_grad_norm_(
                [weight, rank_only, dormant], max_norm=1.0
            )
            optimizer.step()
        payload = {
            "rank": process_rank,
            "pid": __import__("os").getpid(),
            "weight": float(weight.detach().item()),
            "rank_only": float(rank_only.detach().item()),
            "dormant": float(dormant.detach().item()),
            "dormant_grad_is_none": dormant.grad is None,
            "collective_hashes": collective_hashes,
            "last_collective": collective,
            "model_sha256": distributed.tensor_mapping_sha256(
                {
                    "weight": weight.detach(),
                    "rank_only": rank_only.detach(),
                    "dormant": dormant.detach(),
                }
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
    serial_rank_only = torch.nn.Parameter(torch.tensor([0.25]))
    serial_dormant = torch.nn.Parameter(torch.tensor([-0.75]))
    serial_optimizer = torch.optim.AdamW(
        [serial_weight, serial_rank_only, serial_dormant],
        lr=0.1,
        weight_decay=0.0,
    )
    x = torch.arange(1, 5, dtype=torch.float32)
    target = 2.0 * x
    for _ in range(3):
        serial_optimizer.zero_grad(set_to_none=True)
        serial_loss = (
            (serial_weight * x - target).square().mean()
            + (serial_rank_only - 1.0).square().sum() / 4
        )
        serial_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [serial_weight, serial_rank_only, serial_dormant], max_norm=1.0
        )
        serial_optimizer.step()

    assert len({payload["pid"] for payload in rank_payloads}) == 4
    assert len({payload["model_sha256"] for payload in rank_payloads}) == 1
    assert len({payload["optimizer_sha256"] for payload in rank_payloads}) == 1
    assert len(
        {
            tuple(payload["collective_hashes"])
            for payload in rank_payloads
        }
    ) == 1
    assert [payload["weight"] for payload in rank_payloads] == pytest.approx(
        [float(serial_weight.detach().item())] * 4,
        abs=1e-7,
    )
    assert [payload["rank_only"] for payload in rank_payloads] == pytest.approx(
        [float(serial_rank_only.detach().item())] * 4,
        abs=1e-7,
    )
    assert [payload["dormant"] for payload in rank_payloads] == pytest.approx(
        [float(serial_dormant.detach().item())] * 4,
        abs=1e-7,
    )
    assert all(payload["dormant_grad_is_none"] is True for payload in rank_payloads)
    assert all(
        payload["last_collective"]["global_active_parameter_indices"] == [1, 2]
        for payload in rank_payloads
    )
    assert all(
        payload["last_collective"][
            "materialized_zero_gradient_tensors_by_rank"
        ]
        == [0, 1, 1, 1]
        for payload in rank_payloads
    )
