from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as runner


class ToyOnlineMemoryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Parameter(
            torch.tensor([0.2, -0.1, 0.3, 0.0], dtype=torch.float32)
        )
        self.register_buffer("online_memory_keys", torch.ones(2, 4))
        self.register_buffer("online_memory_values", torch.full((2, 4), 2.0))
        self.register_buffer("online_memory_occupancy", torch.ones(2))

    @property
    def online_memory_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.online_memory_keys,
            self.online_memory_values,
            self.online_memory_occupancy,
        )


def _context(process_rank: int = 0) -> distributed.DistributedTrainingContext:
    return distributed.DistributedTrainingContext(
        process_rank=process_rank,
        local_rank=process_rank,
        world_size=4,
        device=torch.device("cpu"),
        backend="gloo",
        control_backend="gloo",
        control_group=object(),
        rank_devices=(),
    )


def _examples() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            row_id=f"family-{family_index}:q{member_order}",
            condition="correct_state",
            episode_id=f"family-{family_index}",
            semantic_target_slot=member_order,
        )
        for family_index in range(4)
        for member_order in range(4)
    ]


def _batch(batch_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        labels=torch.tensor([[-100, 1]], dtype=torch.long).repeat(batch_size, 1),
        query_mask=torch.tensor([[True]]).repeat(batch_size, 1),
        target_slots=torch.zeros(batch_size, dtype=torch.long),
    )


def _contains_identity(value: Any, targets: tuple[torch.Tensor, ...]) -> bool:
    if any(value is target for target in targets):
        return True
    if isinstance(value, Mapping):
        return any(_contains_identity(child, targets) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(child, targets) for child in value)
    return False


def _install_toy_training_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: ToyOnlineMemoryModel,
    context: distributed.DistributedTrainingContext,
    gathered_step_mismatch: bool = False,
) -> tuple[list[torch.Tensor], list[Any]]:
    collective_tensors: list[torch.Tensor] = []
    gathered_values: list[Any] = []

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        distributed,
        "cuda_memory_snapshot",
        lambda current: {
            "process_rank": current.process_rank,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "collate_examples",
        lambda examples, *args, **kwargs: _batch(len(examples)),
    )

    def write_episode_batch(
        current_model: ToyOnlineMemoryModel,
        batch: SimpleNamespace,
        *,
        dtype: torch.dtype,
    ) -> dict[str, int]:
        del batch, dtype
        with torch.no_grad():
            current_model.online_memory_keys.add_(1.0)
            current_model.online_memory_values.add_(1.0)
            current_model.online_memory_occupancy.fill_(1.0)
        return {
            "full_occupancy_count": 1,
            "full_occupancy_total": 1,
            "forced_write_route_match_count": 1,
            "forced_write_route_total": 1,
        }

    def read_episode_batch(
        current_model: ToyOnlineMemoryModel,
        batch: SimpleNamespace,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del dtype
        batch_size = int(batch.labels.size(0))
        logits = current_model.adapter[:3].view(1, 1, 3).expand(batch_size, -1, -1)
        route_logits = {
            "toy.route": current_model.adapter.view(1, 1, 4).expand(
                batch_size, -1, -1
            )
        }
        return logits, route_logits

    monkeypatch.setattr(runner, "_write_episode_batch", write_episode_batch)
    monkeypatch.setattr(runner, "_read_episode_batch", read_episode_batch)
    monkeypatch.setattr(
        runner,
        "_answer_exact_predictions",
        lambda logits, labels: ([True] * int(labels.size(0)), None, None),
    )
    monkeypatch.setattr(
        runner.runtime,
        "_router_gradient_audit",
        lambda current_model, loss: {
            "modules": 1,
            "all_modules_finite_nonzero": True,
        },
    )

    def reset_online_memory(current_model: ToyOnlineMemoryModel) -> None:
        with torch.no_grad():
            for tensor in current_model.online_memory_tensors:
                tensor.zero_()

    monkeypatch.setattr(runner, "reset_delta_mem_states", reset_online_memory)

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        collective_tensors.append(tensor)
        tensor.mul_(context.world_size)

    monkeypatch.setattr(runner.torch_dist, "all_reduce", all_reduce)

    def gather_objects(
        current: distributed.DistributedTrainingContext,
        value: Any,
    ) -> tuple[Any, ...]:
        assert current is context
        gathered_values.append(value)
        values = [value for _ in range(context.world_size)]
        if isinstance(value, Mapping) and "active_names" in value:
            values = [dict(value, rank=rank) for rank in range(context.world_size)]
        elif (
            isinstance(value, Mapping)
            and "local_row_ids" in value
            and "adapter_state_sha256" in value
        ):
            values = [dict(value, rank=rank) for rank in range(context.world_size)]
            if gathered_step_mismatch:
                values[2]["adapter_state_sha256"] = "f" * 64
        return tuple(values)

    monkeypatch.setattr(distributed, "gather_objects", gather_objects)
    return collective_tensors, gathered_values


def _train_one_step(
    model: ToyOnlineMemoryModel,
    context: distributed.DistributedTrainingContext,
    progress_path: Path,
    *,
    capture_step_evidence: bool = False,
) -> dict[str, Any]:
    return runner.train_model_distributed(
        model,
        _examples(),
        context=context,
        seed=11,
        epochs=1,
        max_steps=1,
        global_batch_size=16,
        learning_rate=0.01,
        answer_weight=1.0,
        route_weight=1.0,
        max_grad_norm=1.0,
        pad_token_id=0,
        dtype=torch.float32,
        progress_path=progress_path,
        capture_step_evidence=capture_step_evidence,
    )


def test_injected_forward_failure_reaches_consensus_before_data_collective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(process_rank=2)
    model = ToyOnlineMemoryModel()
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(distributed, "cuda_memory_snapshot", lambda current: {})

    def gather_objects(
        current: distributed.DistributedTrainingContext,
        value: Any,
    ) -> tuple[Any, ...]:
        if isinstance(value, Mapping) and "passed" in value:
            return tuple(
                value
                if rank == context.process_rank
                else {"rank": rank, "passed": True, "error": None}
                for rank in range(context.world_size)
            )
        return tuple(value for _ in range(context.world_size))

    monkeypatch.setattr(distributed, "gather_objects", gather_objects)
    monkeypatch.setattr(
        runner,
        "collate_examples",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected rank-two forward failure")
        ),
    )
    monkeypatch.setattr(
        runner.torch_dist,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail(
            "data collective ran after a pre-collective rank failure"
        ),
    )

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="step-1-microbatch-preparation.*injected rank-two forward failure",
    ):
        _train_one_step(model, context, tmp_path / "progress.jsonl")


def test_invalid_local_gradients_are_consensused_before_gradient_sum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(process_rank=1)
    model = ToyOnlineMemoryModel()
    _install_toy_training_runtime(monkeypatch, model=model, context=context)
    monkeypatch.setattr(
        distributed,
        "validate_local_gradients",
        lambda named_trainable: {
            "parameter_tensors": 1,
            "parameter_names_sha256": distributed.canonical_sha256(["adapter"]),
            "active_gradient_tensors": 1,
            "active_names_sha256": distributed.canonical_sha256(["adapter"]),
            "missing_gradient_tensors": 0,
            "missing_names_sha256": distributed.canonical_sha256([]),
            "nonfinite_gradient_tensors": 1,
            "nonfinite_names_sha256": distributed.canonical_sha256(["adapter"]),
            "nonfinite_preview": ["adapter"],
            "non_fp32_gradient_tensors": 0,
            "non_fp32_names_sha256": distributed.canonical_sha256([]),
            "non_fp32_preview": [],
            "passed": False,
        },
    )
    gradient_sum_called = False

    def forbidden_gradient_sum(*args, **kwargs):
        nonlocal gradient_sum_called
        gradient_sum_called = True
        pytest.fail("gradient SUM ran after failed local gradient validation")

    monkeypatch.setattr(distributed, "sum_gradients", forbidden_gradient_sum)

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="step-1-gradient-validation.*Invalid local gradients",
    ):
        _train_one_step(model, context, tmp_path / "progress.jsonl")

    assert gradient_sum_called is False


def test_update_order_is_backward_sum_clip_reset_then_adamw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    model = ToyOnlineMemoryModel()
    _install_toy_training_runtime(monkeypatch, model=model, context=context)
    operations: list[str] = []
    model.adapter.register_hook(
        lambda gradient: operations.append("backward") or gradient
    )

    original_sum_gradients = distributed.sum_gradients

    def ordered_sum_gradients(*args, **kwargs):
        operations.append("sum")
        return original_sum_gradients(*args, **kwargs)

    monkeypatch.setattr(distributed, "sum_gradients", ordered_sum_gradients)
    original_clip = torch.nn.utils.clip_grad_norm_

    def ordered_clip(*args, **kwargs):
        operations.append("clip")
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", ordered_clip)

    def ordered_reset(current_model: ToyOnlineMemoryModel) -> None:
        operations.append("reset")
        with torch.no_grad():
            for tensor in current_model.online_memory_tensors:
                tensor.zero_()

    monkeypatch.setattr(runner, "reset_delta_mem_states", ordered_reset)
    original_adamw_step = torch.optim.AdamW.step

    def ordered_adamw_step(optimizer, *args, **kwargs):
        operations.append("adamw")
        return original_adamw_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", ordered_adamw_step)

    _train_one_step(model, context, tmp_path / "progress.jsonl")

    assert operations == [
        "backward",
        "reset",
        "backward",
        "reset",
        "sum",
        "clip",
        "adamw",
    ]


def test_capture_evidence_replica_mismatch_raises_distributed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    model = ToyOnlineMemoryModel()
    _install_toy_training_runtime(
        monkeypatch,
        model=model,
        context=context,
        gathered_step_mismatch=True,
    )

    progress_path = tmp_path / "progress.jsonl"
    with pytest.raises(
        distributed.DistributedTrainingError,
        match="Replica consensus failed after step 1",
    ):
        _train_one_step(
            model,
            context,
            progress_path,
            capture_step_evidence=True,
        )
    assert not progress_path.exists()


def test_online_memory_runtime_tensors_never_enter_collectives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    model = ToyOnlineMemoryModel()
    collective_tensors, gathered_values = _install_toy_training_runtime(
        monkeypatch, model=model, context=context
    )
    original_sum_gradients = distributed.sum_gradients

    def checked_sum_gradients(
        current: distributed.DistributedTrainingContext,
        named_trainable,
        **kwargs,
    ):
        assert [name for name, _ in named_trainable] == ["adapter"]
        assert [parameter for _, parameter in named_trainable] == [model.adapter]
        return original_sum_gradients(current, named_trainable, **kwargs)

    monkeypatch.setattr(distributed, "sum_gradients", checked_sum_gradients)

    _train_one_step(model, context, tmp_path / "progress.jsonl")

    runtime_tensors = model.online_memory_tensors
    runtime_pointers = {tensor.data_ptr() for tensor in runtime_tensors}
    assert collective_tensors
    assert all(tensor.data_ptr() not in runtime_pointers for tensor in collective_tensors)
    assert all(
        not _contains_identity(value, runtime_tensors) for value in gathered_values
    )


@pytest.mark.parametrize("process_rank", [0, 1])
def test_only_primary_rank_writes_training_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_rank: int,
) -> None:
    context = _context(process_rank=process_rank)
    model = ToyOnlineMemoryModel()
    _install_toy_training_runtime(monkeypatch, model=model, context=context)
    progress_path = tmp_path / "training_progress.jsonl"

    result = _train_one_step(model, context, progress_path)

    assert progress_path.exists() is (process_rank == 0)
    assert (result["progress_sha256"] is not None) is (process_rank == 0)
    if process_rank == 0:
        assert len(progress_path.read_text(encoding="utf-8").splitlines()) == 1


def test_metric_preparation_failure_stops_before_metric_collective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(process_rank=2)
    model = ToyOnlineMemoryModel()
    collective_tensors, _ = _install_toy_training_runtime(
        monkeypatch, model=model, context=context
    )
    monkeypatch.setattr(
        runner,
        "_answer_exact_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected metric preparation failure")
        ),
    )

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="step-1-microbatch-1-metrics.*injected metric preparation failure",
    ):
        _train_one_step(model, context, tmp_path / "progress.jsonl")

    assert len(collective_tensors) == 1
    assert not (tmp_path / "progress.jsonl").exists()


def test_rank_zero_progress_failure_is_consensused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(process_rank=0)
    model = ToyOnlineMemoryModel()
    _install_toy_training_runtime(monkeypatch, model=model, context=context)
    progress_path = tmp_path / "progress.jsonl"
    monkeypatch.setattr(
        runner,
        "_append_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected progress write failure")
        ),
    )

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="rank-zero-progress-commit.*injected progress write failure",
    ):
        _train_one_step(model, context, progress_path)

    assert not progress_path.exists()


def test_final_cuda_snapshot_failure_is_consensused_before_memory_gather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(process_rank=3)
    model = ToyOnlineMemoryModel()
    _, gathered_values = _install_toy_training_runtime(
        monkeypatch, model=model, context=context
    )
    calls = 0

    def cuda_memory_snapshot(
        current: distributed.DistributedTrainingContext,
    ) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected final CUDA snapshot failure")
        return {
            "process_rank": current.process_rank,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        }

    monkeypatch.setattr(distributed, "cuda_memory_snapshot", cuda_memory_snapshot)

    with pytest.raises(
        distributed.DistributedTrainingError,
        match="final-cuda-memory-preparation.*injected final CUDA snapshot failure",
    ):
        _train_one_step(model, context, tmp_path / "progress.jsonl")

    assert calls == 2
    assert not any(
        isinstance(value, Mapping) and "before_training" in value
        for value in gathered_values
    )


def test_captured_phase_order_is_the_observed_production_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(process_rank=0)
    model = ToyOnlineMemoryModel()
    _install_toy_training_runtime(monkeypatch, model=model, context=context)

    result = _train_one_step(
        model,
        context,
        tmp_path / "progress.jsonl",
        capture_step_evidence=True,
    )

    evidence = result["distributed"]["step_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["phase_order"] == list(runner.DISTRIBUTED_STEP_PHASE_ORDER)
    assert evidence[0]["trainable_names_sha256"] == (
        result["distributed"]["trainable_names_sha256"]
    )
