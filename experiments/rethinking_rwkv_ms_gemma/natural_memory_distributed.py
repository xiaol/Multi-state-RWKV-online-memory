"""Fail-closed distributed primitives for the natural outer-memory proof."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
import random
import socket
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


REQUIRED_WORLD_SIZE = 4
REQUIRED_LOCAL_BATCH_SIZE = 4
REQUIRED_LOCAL_MICROBATCH_SIZE = 2
REQUIRED_GRADIENT_ACCUMULATION_STEPS = 2
REQUIRED_GLOBAL_BATCH_SIZE = REQUIRED_WORLD_SIZE * REQUIRED_LOCAL_BATCH_SIZE
if (
    REQUIRED_LOCAL_MICROBATCH_SIZE * REQUIRED_GRADIENT_ACCUMULATION_STEPS
    != REQUIRED_LOCAL_BATCH_SIZE
):
    raise RuntimeError("Distributed microbatch contract does not fill the local batch")
GRADIENT_BUCKET_BYTES = 64 * 1024 * 1024
TORCHRUN_ENVIRONMENT = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


class DistributedTrainingError(RuntimeError):
    """A phase failed on at least one training rank."""


@dataclass(frozen=True)
class DistributedTrainingContext:
    process_rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    control_backend: str
    control_group: Any
    rank_devices: tuple[Mapping[str, Any], ...]

    @property
    def is_primary(self) -> bool:
        return self.process_rank == 0


@dataclass(frozen=True)
class GlobalTrainingStep:
    step: int
    epoch: int
    global_indices: tuple[int, ...]
    global_row_ids: tuple[str, ...]
    step_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def torchrun_environment() -> Mapping[str, int] | None:
    present = {name: os.environ.get(name) for name in TORCHRUN_ENVIRONMENT}
    populated = {name: value for name, value in present.items() if value is not None}
    if not populated:
        return None
    if len(populated) != len(TORCHRUN_ENVIRONMENT):
        missing = sorted(set(TORCHRUN_ENVIRONMENT) - set(populated))
        raise ValueError(
            "Distributed training requires a complete torchrun environment; missing "
            + ", ".join(missing)
        )
    try:
        parsed = {name: int(value) for name, value in present.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Torchrun rank variables must be integers") from error
    if parsed["WORLD_SIZE"] <= 1:
        raise ValueError("Distributed training requires WORLD_SIZE greater than one")
    if not 0 <= parsed["RANK"] < parsed["WORLD_SIZE"]:
        raise ValueError("RANK is outside WORLD_SIZE")
    if parsed["LOCAL_RANK"] < 0:
        raise ValueError("LOCAL_RANK must be nonnegative")
    return parsed


def _local_device_evidence(
    *, process_rank: int, local_rank: int, device: torch.device
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "process_rank": process_rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "device_index": int(device.index),
        "device_name": properties.name,
        "device_uuid": str(properties.uuid),
        "device_total_memory_bytes": int(properties.total_memory),
    }


def _validated_rank_device_evidence(
    gathered: Sequence[Any], *, world_size: int
) -> tuple[Mapping[str, Any], ...]:
    if len(gathered) != world_size or any(
        not isinstance(value, Mapping) for value in gathered
    ):
        raise RuntimeError("Distributed device evidence is malformed")
    devices = tuple(dict(value) for value in gathered)
    if [value.get("process_rank") for value in devices] != list(range(world_size)):
        raise RuntimeError("Distributed rank identity gathering is inconsistent")
    if len({value.get("pid") for value in devices}) != world_size:
        raise RuntimeError("Distributed ranks do not have distinct process IDs")
    if len({value.get("device_uuid") for value in devices}) != world_size:
        raise RuntimeError("Distributed ranks do not own distinct CUDA devices")
    return devices


def initialize_distributed_training(
    device_name: str,
    *,
    required_world_size: int = REQUIRED_WORLD_SIZE,
    timeout_seconds: int = 300,
) -> DistributedTrainingContext | None:
    environment = torchrun_environment()
    if environment is None:
        return None
    if environment["WORLD_SIZE"] != required_world_size:
        raise ValueError(
            f"Distributed training requires exactly {required_world_size} ranks"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed natural-memory training requires CUDA")
    local_rank = environment["LOCAL_RANK"]
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK {local_rank} has no visible CUDA device")
    requested = torch.device(device_name)
    if requested.type != "cuda":
        raise ValueError("Distributed natural-memory training requires a CUDA device")
    if requested.index is not None and requested.index != local_rank:
        raise ValueError("Explicit CUDA index must equal LOCAL_RANK under torchrun")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    timeout = timedelta(seconds=timeout_seconds)
    if dist.is_initialized():
        raise RuntimeError("A process group is already initialized")
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
    control_group = None
    try:
        control_group = dist.new_group(
            ranks=list(range(environment["WORLD_SIZE"])),
            backend="gloo",
            timeout=timeout,
        )
        provisional_context = DistributedTrainingContext(
            process_rank=environment["RANK"],
            local_rank=local_rank,
            world_size=environment["WORLD_SIZE"],
            device=device,
            backend="nccl",
            control_backend="gloo",
            control_group=control_group,
            rank_devices=(),
        )
        local_device = _consensual_operation(
            provisional_context,
            phase="distributed-device-evidence-preparation",
            operation=lambda: _local_device_evidence(
                process_rank=environment["RANK"],
                local_rank=local_rank,
                device=device,
            ),
        )
        gathered: list[Any] = [None] * environment["WORLD_SIZE"]
        dist.all_gather_object(gathered, local_device, group=control_group)
        devices = _consensual_operation(
            provisional_context,
            phase="distributed-device-evidence-validation",
            operation=lambda: _validated_rank_device_evidence(
                gathered, world_size=environment["WORLD_SIZE"]
            ),
        )
        return DistributedTrainingContext(
            process_rank=environment["RANK"],
            local_rank=local_rank,
            world_size=environment["WORLD_SIZE"],
            device=device,
            backend="nccl",
            control_backend="gloo",
            control_group=control_group,
            rank_devices=devices,
        )
    except BaseException:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


def destroy_distributed_training(context: DistributedTrainingContext | None) -> None:
    if context is not None and dist.is_initialized():
        dist.destroy_process_group()


def gather_objects(
    context: DistributedTrainingContext,
    value: Any,
) -> tuple[Any, ...]:
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, value, group=context.control_group)
    return tuple(gathered)


def require_consensus(
    context: DistributedTrainingContext,
    value: Any,
    *,
    description: str,
) -> tuple[Any, ...]:
    gathered = gather_objects(context, value)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise DistributedTrainingError(
            f"Distributed {description} differs across ranks: {gathered!r}"
        )
    return gathered


def phase_consensus(
    context: DistributedTrainingContext,
    *,
    phase: str,
    error: BaseException | None,
) -> None:
    if error is None:
        local = {"rank": context.process_rank, "passed": True, "error": None}
    else:
        rendered = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        local = {
            "rank": context.process_rank,
            "passed": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            },
        }
    statuses = gather_objects(context, local)
    failed = [status for status in statuses if status.get("passed") is not True]
    if failed:
        raise DistributedTrainingError(
            f"Distributed phase {phase!r} failed: {_canonical_json(failed)}"
        )


def _consensual_operation(
    context: DistributedTrainingContext,
    *,
    phase: str,
    operation: Callable[[], Any],
) -> Any:
    result: Any = None
    error: BaseException | None = None
    try:
        result = operation()
    except BaseException as caught:
        error = caught
    phase_consensus(context, phase=phase, error=error)
    if error is not None:
        raise error
    return result


def build_global_training_schedule(
    row_ids: Sequence[str],
    *,
    seed: int,
    epochs: int,
    max_steps: int | None,
    world_size: int,
    local_batch_size: int,
) -> tuple[tuple[GlobalTrainingStep, ...], str]:
    if not row_ids or len(set(row_ids)) != len(row_ids):
        raise ValueError("Training row IDs must be nonempty and unique")
    if epochs <= 0 or world_size <= 1 or local_batch_size <= 0:
        raise ValueError("Distributed schedule dimensions must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when supplied")
    global_batch_size = world_size * local_batch_size
    if len(row_ids) % global_batch_size:
        raise ValueError(
            "Every epoch must divide into complete distributed global batches"
        )
    rng = random.Random(seed)
    steps: list[GlobalTrainingStep] = []
    stop = False
    for epoch in range(epochs):
        indices = list(range(len(row_ids)))
        rng.shuffle(indices)
        for start in range(0, len(indices), global_batch_size):
            selected = tuple(indices[start : start + global_batch_size])
            if len(selected) != global_batch_size:
                raise RuntimeError("Distributed schedule emitted a partial batch")
            selected_rows = tuple(row_ids[index] for index in selected)
            step = len(steps) + 1
            payload = {
                "step": step,
                "epoch": epoch,
                "global_indices": list(selected),
                "global_row_ids": list(selected_rows),
            }
            steps.append(
                GlobalTrainingStep(
                    step=step,
                    epoch=epoch,
                    global_indices=selected,
                    global_row_ids=selected_rows,
                    step_sha256=canonical_sha256(payload),
                )
            )
            if max_steps is not None and len(steps) >= max_steps:
                stop = True
                break
        if stop:
            break
    if max_steps is not None and len(steps) != max_steps:
        raise ValueError(
            f"Requested {max_steps} updates but epochs provide only {len(steps)}"
        )
    schedule_payload = [
        {
            "step": step.step,
            "epoch": step.epoch,
            "global_indices": list(step.global_indices),
            "global_row_ids": list(step.global_row_ids),
            "step_sha256": step.step_sha256,
        }
        for step in steps
    ]
    return tuple(steps), canonical_sha256(schedule_payload)


def local_step_indices(
    step: GlobalTrainingStep,
    *,
    process_rank: int,
    world_size: int,
    local_batch_size: int,
) -> tuple[int, ...]:
    if not 0 <= process_rank < world_size:
        raise ValueError("Process rank is outside the schedule world size")
    if len(step.global_indices) != world_size * local_batch_size:
        raise ValueError("Schedule step has the wrong global batch size")
    start = process_rank * local_batch_size
    return step.global_indices[start : start + local_batch_size]


def answer_loss_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or labels.ndim != 2 or logits.size(0) != labels.size(0):
        raise ValueError("Answer logits and labels are misaligned")
    if labels.size(1) < 2:
        raise ValueError("Answer labels have no causal sequence axis")
    supervised = labels[:, 1:].ne(-100)
    if not bool(supervised.any().item()):
        raise ValueError("Answer labels contain no supervised targets")
    predictor_indices = supervised.any(dim=0).nonzero(as_tuple=False).flatten()
    if logits.size(1) == labels.size(1):
        selected_logits = logits.index_select(1, predictor_indices)
    elif logits.size(1) == predictor_indices.numel():
        selected_logits = logits
    else:
        raise ValueError("Answer logits do not cover supervised predictors")
    selected_labels = labels.index_select(1, predictor_indices + 1)
    count = int(selected_labels.ne(-100).sum().item())
    loss_sum = F.cross_entropy(
        selected_logits.contiguous().float().view(-1, logits.size(-1)),
        selected_labels.contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, count


def selected_route_logits(
    logits: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != query_mask.shape:
        raise ValueError("Route logits and query mask are misaligned")
    counts = query_mask.sum(dim=1, keepdim=True)
    if bool(counts.eq(0).any().item()):
        raise ValueError("Every row must select at least one query token")
    return torch.einsum(
        "btc,bt->bc", logits.float(), query_mask.to(dtype=torch.float32)
    ) / counts.to(dtype=torch.float32)


def route_loss_sum_and_predictions(
    logits_by_module: Mapping[str, torch.Tensor],
    query_mask: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    hard_negative_margin: float = 0.0,
    hard_negative_weight: float = 0.0,
) -> tuple[torch.Tensor, int, dict[str, torch.Tensor]]:
    if not logits_by_module:
        raise RuntimeError("No graph-connected projected-KV route logits were exposed")
    if bool(target_slots.lt(0).any().item()):
        raise ValueError("Route loss requires a target slot for every row")
    if hard_negative_margin < 0.0 or hard_negative_weight < 0.0:
        raise ValueError("Hard-negative margin and weight must be nonnegative")
    losses: list[torch.Tensor] = []
    predictions: dict[str, torch.Tensor] = {}
    for name in sorted(logits_by_module):
        selected = selected_route_logits(logits_by_module[name], query_mask)
        loss = F.cross_entropy(selected, target_slots, reduction="sum")
        if hard_negative_weight > 0.0:
            target = selected.gather(1, target_slots.unsqueeze(1)).squeeze(1)
            negative = selected.masked_fill(
                F.one_hot(target_slots, selected.size(-1)).to(dtype=torch.bool),
                -torch.inf,
            ).amax(dim=1)
            hard_negative_loss = F.relu(
                hard_negative_margin + negative - target
            ).sum()
            loss = loss + hard_negative_weight * hard_negative_loss
        losses.append(loss)
        predictions[name] = selected.argmax(dim=-1)
    return torch.stack(losses).mean(), int(target_slots.numel()), predictions


def prepare_objective_statistics(
    *,
    answer_loss_sum: torch.Tensor,
    answer_token_count: int,
    route_loss_sum: torch.Tensor,
    route_row_count: int,
) -> torch.Tensor:
    if answer_token_count <= 0 or route_row_count <= 0:
        raise ValueError("Every rank must contribute answer tokens and route rows")
    if answer_loss_sum.numel() != 1 or route_loss_sum.numel() != 1:
        raise ValueError("Local objective loss sums must be scalar tensors")
    values = torch.tensor(
        [
            float(answer_loss_sum.detach().float().item()),
            float(route_loss_sum.detach().float().item()),
            float(answer_token_count),
            float(route_row_count),
        ],
        dtype=torch.float64,
        device=answer_loss_sum.device,
    )
    if values.device != route_loss_sum.device:
        raise ValueError("Answer and route loss sums must share one device")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Local objective statistics must be finite")
    return values


def reduce_objective_statistics(
    context: DistributedTrainingContext,
    values: torch.Tensor,
) -> Mapping[str, float | int]:
    if values.shape != (4,) or values.dtype != torch.float64:
        raise ValueError("Prepared objective statistics have the wrong shape or dtype")
    if values.device != context.device:
        raise ValueError("Prepared objective statistics are on the wrong device")
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("Global objective statistics must be finite")
    answer_tokens = int(values[2].item())
    route_rows = int(values[3].item())
    if answer_tokens <= 0 or route_rows <= 0:
        raise RuntimeError("Global objective denominators must be positive")
    return {
        "answer_loss_sum": float(values[0].item()),
        "route_loss_sum": float(values[1].item()),
        "answer_token_count": answer_tokens,
        "route_row_count": route_rows,
    }


def stable_named_parameters(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    ordered = tuple(sorted(named_parameters, key=lambda item: item[0]))
    names = [name for name, _ in ordered]
    if not names or len(names) != len(set(names)):
        raise ValueError("Named parameter collection must be nonempty and unique")
    return ordered


def named_tensor_metadata(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in stable_named_parameters(named_parameters)
    )


def _tensor_buckets(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    bucket_bytes: int,
) -> tuple[tuple[tuple[str, torch.Tensor], ...], ...]:
    if bucket_bytes <= 0:
        raise ValueError("Collective bucket size must be positive")
    buckets: list[list[tuple[str, torch.Tensor]]] = []
    current: list[tuple[str, torch.Tensor]] = []
    current_bytes = 0
    current_dtype: torch.dtype | None = None
    current_device: torch.device | None = None
    for name, tensor in named_tensors:
        tensor_bytes = tensor.numel() * tensor.element_size()
        incompatible = current and (
            tensor.dtype != current_dtype
            or tensor.device != current_device
            or current_bytes + tensor_bytes > bucket_bytes
        )
        if incompatible:
            buckets.append(current)
            current = []
            current_bytes = 0
        if not current:
            current_dtype = tensor.dtype
            current_device = tensor.device
        current.append((name, tensor))
        current_bytes += tensor_bytes
        if current_bytes >= bucket_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
            current_dtype = None
            current_device = None
    if current:
        buckets.append(current)
    return tuple(tuple(bucket) for bucket in buckets)


def _collective_bucket_plan(
    buckets: Sequence[Sequence[tuple[str, torch.Tensor]]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "names": [name for name, _ in bucket],
            "tensors": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device_type": tensor.device.type,
                    "numel": tensor.numel(),
                }
                for name, tensor in bucket
            ],
        }
        for bucket in buckets
    )


def _validate_collective_tensors(
    context: DistributedTrainingContext,
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> None:
    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Collective tensor {name!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"Collective tensor {name!r} must be strided")
        if tensor.device != context.device:
            raise ValueError(
                f"Collective tensor {name!r} is on {tensor.device}, "
                f"expected {context.device}"
            )


def _flatten_collective_bucket(
    bucket: Sequence[tuple[str, torch.Tensor]],
    *,
    require_finite: bool,
) -> torch.Tensor:
    flat = torch.cat([tensor.reshape(-1) for _, tensor in bucket])
    expected_numel = sum(tensor.numel() for _, tensor in bucket)
    if flat.numel() != expected_numel or not flat.is_contiguous():
        raise RuntimeError("Collective bucket flattening produced an invalid buffer")
    if require_finite and not bool(torch.isfinite(flat).all().item()):
        raise RuntimeError("Collective bucket contains non-finite values")
    return flat


def _apply_collective_bucket(
    flat: torch.Tensor,
    bucket: Sequence[tuple[str, torch.Tensor]],
) -> int:
    if not bool(torch.isfinite(flat).all().item()):
        raise RuntimeError("Collective bucket produced non-finite values")
    offset = 0
    total_bytes = 0
    for name, tensor in bucket:
        count = tensor.numel()
        tensor.copy_(flat[offset : offset + count].view_as(tensor))
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(
                f"Collective bucket application made tensor {name!r} non-finite"
            )
        offset += count
        total_bytes += count * tensor.element_size()
    if offset != flat.numel():
        raise RuntimeError("Collective bucket application consumed the wrong size")
    return total_bytes


def broadcast_named_parameters(
    context: DistributedTrainingContext,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    source_rank: int = 0,
    bucket_bytes: int = GRADIENT_BUCKET_BYTES,
) -> Mapping[str, Any]:
    def prepare() -> tuple[
        tuple[tuple[str, torch.nn.Parameter], ...],
        tuple[tuple[tuple[str, torch.Tensor], ...], ...],
        str,
    ]:
        if not isinstance(source_rank, int) or not 0 <= source_rank < context.world_size:
            raise ValueError("Broadcast source rank is outside the process group")
        ordered_parameters = stable_named_parameters(named_parameters)
        if any(
            not isinstance(parameter, torch.nn.Parameter)
            for _, parameter in ordered_parameters
        ):
            raise TypeError("Named parameter collection contains a non-parameter")
        named_tensors = tuple(
            (name, parameter.data) for name, parameter in ordered_parameters
        )
        _validate_collective_tensors(context, named_tensors)
        prepared_buckets = _tensor_buckets(
            named_tensors,
            bucket_bytes=bucket_bytes,
        )
        plan_sha256 = canonical_sha256(
            {
                "operation": "broadcast_named_parameters",
                "source_rank": source_rank,
                "bucket_bytes": bucket_bytes,
                "requires_grad": [
                    [name, bool(parameter.requires_grad)]
                    for name, parameter in ordered_parameters
                ],
                "buckets": _collective_bucket_plan(prepared_buckets),
            }
        )
        return ordered_parameters, prepared_buckets, plan_sha256

    ordered, buckets, plan_sha256 = _consensual_operation(
        context,
        phase="broadcast-named-parameters-preparation",
        operation=prepare,
    )
    require_consensus(
        context,
        plan_sha256,
        description="broadcast parameter bucket plan",
    )
    total_bytes = 0
    with torch.no_grad():
        for bucket_index, bucket in enumerate(buckets):
            flat = _consensual_operation(
                context,
                phase=(
                    "broadcast-named-parameters-"
                    f"bucket-{bucket_index}-flatten-readiness"
                ),
                operation=lambda bucket=bucket: _flatten_collective_bucket(
                    bucket,
                    require_finite=context.process_rank == source_rank,
                ),
            )
            _consensual_operation(
                context,
                phase=(
                    "broadcast-named-parameters-"
                    f"bucket-{bucket_index}-collective"
                ),
                operation=lambda flat=flat: dist.broadcast(flat, src=source_rank),
            )
            total_bytes += _consensual_operation(
                context,
                phase=(
                    "broadcast-named-parameters-"
                    f"bucket-{bucket_index}-post-collective-apply"
                ),
                operation=lambda flat=flat, bucket=bucket: _apply_collective_bucket(
                    flat, bucket
                ),
            )
    return {
        "parameter_tensors": len(ordered),
        "parameter_names_sha256": canonical_sha256(
            [name for name, _ in ordered]
        ),
        "bucket_plan_sha256": plan_sha256,
        "collective_buckets": len(buckets),
        "broadcast_bytes": total_bytes,
    }


def validate_local_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    ordered = stable_named_parameters(named_trainable)
    parameter_names = [name for name, _ in ordered]
    active = [name for name, parameter in ordered if parameter.grad is not None]
    missing = [name for name, parameter in ordered if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in ordered
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    non_fp32 = [
        name
        for name, parameter in ordered
        if parameter.grad is not None and parameter.grad.dtype != torch.float32
    ]
    return {
        "parameter_tensors": len(ordered),
        "parameter_names_sha256": canonical_sha256(parameter_names),
        "active_gradient_tensors": len(active),
        "active_names_sha256": canonical_sha256(active),
        "missing_gradient_tensors": len(missing),
        "missing_names_sha256": canonical_sha256(missing),
        "nonfinite_gradient_tensors": len(nonfinite),
        "nonfinite_names_sha256": canonical_sha256(nonfinite),
        "nonfinite_preview": nonfinite[:8],
        "non_fp32_gradient_tensors": len(non_fp32),
        "non_fp32_names_sha256": canonical_sha256(non_fp32),
        "non_fp32_preview": non_fp32[:8],
        "passed": not nonfinite and not non_fp32,
    }


def _validated_active_gradient_union(
    gathered: Sequence[Any],
    *,
    parameter_names: Sequence[str],
    world_size: int,
) -> Mapping[str, Any]:
    if len(gathered) != world_size or any(
        not isinstance(value, Mapping) for value in gathered
    ):
        raise RuntimeError("Distributed active-gradient evidence is malformed")
    ordered_parameter_names = tuple(parameter_names)
    parameter_name_set = set(ordered_parameter_names)
    per_rank: list[dict[str, Any]] = []
    active_sets: list[set[str]] = []
    for expected_rank, value in enumerate(gathered):
        rank = value.get("rank")
        active_names = value.get("active_names")
        if rank != expected_rank or not isinstance(active_names, list):
            raise RuntimeError("Distributed active-gradient rank evidence is malformed")
        if any(not isinstance(name, str) for name in active_names):
            raise RuntimeError("Distributed active-gradient names are malformed")
        active_set = set(active_names)
        expected_order = [
            name for name in ordered_parameter_names if name in active_set
        ]
        if (
            len(active_names) != len(active_set)
            or active_set - parameter_name_set
            or active_names != expected_order
            or value.get("active_gradient_tensors") != len(active_names)
            or value.get("active_names_sha256") != canonical_sha256(active_names)
        ):
            raise RuntimeError("Distributed active-gradient evidence is inconsistent")
        active_sets.append(active_set)
        per_rank.append(
            {
                "rank": expected_rank,
                "active_gradient_tensors": len(active_names),
                "active_names_sha256": canonical_sha256(active_names),
            }
        )

    global_active_set = set().union(*active_sets)
    global_active_names = [
        name for name in ordered_parameter_names if name in global_active_set
    ]
    global_inactive_names = [
        name for name in ordered_parameter_names if name not in global_active_set
    ]
    materialized_by_rank = [
        len(global_active_set - active_set) for active_set in active_sets
    ]
    return {
        "global_active_names": global_active_names,
        "global_active_names_sha256": canonical_sha256(global_active_names),
        "global_inactive_names": global_inactive_names,
        "global_inactive_names_sha256": canonical_sha256(global_inactive_names),
        "per_rank_active_gradients": per_rank,
        "materialized_zero_gradient_tensors_by_rank": materialized_by_rank,
    }


def sum_gradients(
    context: DistributedTrainingContext,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    bucket_bytes: int = GRADIENT_BUCKET_BYTES,
) -> Mapping[str, Any]:
    def prepare() -> tuple[
        tuple[tuple[str, torch.nn.Parameter], ...], dict[str, Any]
    ]:
        ordered_parameters = stable_named_parameters(named_trainable)
        if any(
            not isinstance(parameter, torch.nn.Parameter)
            for _, parameter in ordered_parameters
        ):
            raise TypeError("Named trainable collection contains a non-parameter")
        validation = validate_local_gradients(ordered_parameters)
        if validation["passed"] is not True:
            raise RuntimeError(f"Invalid local gradients: {validation!r}")
        active_names = [
            name for name, parameter in ordered_parameters if parameter.grad is not None
        ]
        local_active = {
            "rank": context.process_rank,
            "active_names": active_names,
            "active_gradient_tensors": len(active_names),
            "active_names_sha256": canonical_sha256(active_names),
        }
        return ordered_parameters, local_active

    ordered, local_active = _consensual_operation(
        context,
        phase="sum-gradients-preparation",
        operation=prepare,
    )
    gathered_active = gather_objects(context, local_active)
    parameter_names = [name for name, _ in ordered]
    union = _consensual_operation(
        context,
        phase="sum-gradients-active-union-validation",
        operation=lambda: _validated_active_gradient_union(
            gathered_active,
            parameter_names=parameter_names,
            world_size=context.world_size,
        ),
    )
    global_active_names = tuple(union["global_active_names"])
    global_active_set = set(global_active_names)

    def materialize_missing_active_gradients() -> int:
        materialized = 0
        with torch.no_grad():
            for name, parameter in ordered:
                if name in global_active_set and parameter.grad is None:
                    parameter.grad = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    materialized += 1
        expected = union["materialized_zero_gradient_tensors_by_rank"][
            context.process_rank
        ]
        if materialized != expected:
            raise RuntimeError(
                "Materialized zero-gradient count differs from active-union evidence"
            )
        return materialized

    _consensual_operation(
        context,
        phase="sum-gradients-zero-materialization",
        operation=materialize_missing_active_gradients,
    )

    def prepare_collective() -> tuple[
        tuple[tuple[tuple[str, torch.Tensor], ...], ...], str
    ]:
        named_gradients = tuple(
            (name, parameter.grad)
            for name, parameter in ordered
            if name in global_active_set
        )
        _validate_collective_tensors(context, named_gradients)
        prepared_buckets = _tensor_buckets(
            named_gradients,
            bucket_bytes=bucket_bytes,
        )
        plan_sha256 = canonical_sha256(
            {
                "operation": "sum_gradients",
                "bucket_bytes": bucket_bytes,
                "trainable_parameter_names_sha256": canonical_sha256(parameter_names),
                "global_active_names_sha256": union[
                    "global_active_names_sha256"
                ],
                "buckets": _collective_bucket_plan(prepared_buckets),
            }
        )
        return prepared_buckets, plan_sha256

    buckets, plan_sha256 = _consensual_operation(
        context,
        phase="sum-gradients-collective-preparation",
        operation=prepare_collective,
    )
    require_consensus(
        context,
        plan_sha256,
        description="gradient SUM bucket plan",
    )
    total_bytes = 0
    with torch.no_grad():
        for bucket_index, bucket in enumerate(buckets):
            flat = _consensual_operation(
                context,
                phase=f"sum-gradients-bucket-{bucket_index}-flatten-readiness",
                operation=lambda bucket=bucket: _flatten_collective_bucket(
                    bucket,
                    require_finite=True,
                ),
            )
            _consensual_operation(
                context,
                phase=f"sum-gradients-bucket-{bucket_index}-collective",
                operation=lambda flat=flat: dist.all_reduce(
                    flat, op=dist.ReduceOp.SUM
                ),
            )
            total_bytes += _consensual_operation(
                context,
                phase=(
                    f"sum-gradients-bucket-{bucket_index}-post-collective-apply"
                ),
                operation=lambda flat=flat, bucket=bucket: _apply_collective_bucket(
                    flat, bucket
                ),
            )

    def validate_applied_gradients() -> None:
        validation = validate_local_gradients(ordered)
        if validation["passed"] is not True:
            raise RuntimeError(f"Invalid globally summed gradients: {validation!r}")
        active_after_sum = tuple(
            name for name, parameter in ordered if parameter.grad is not None
        )
        if active_after_sum != global_active_names:
            raise RuntimeError(
                "Globally summed gradients differ from the active-gradient union"
            )

    _consensual_operation(
        context,
        phase="sum-gradients-final-validation",
        operation=validate_applied_gradients,
    )
    global_active_indices = [
        index for index, name in enumerate(parameter_names) if name in global_active_set
    ]
    global_inactive_indices = [
        index for index, name in enumerate(parameter_names) if name not in global_active_set
    ]
    return {
        "trainable_parameter_tensors": len(ordered),
        "trainable_names_sha256": canonical_sha256(parameter_names),
        "gradient_tensors": len(global_active_names),
        "global_active_parameter_indices": global_active_indices,
        "global_active_names_sha256": union["global_active_names_sha256"],
        "global_inactive_parameter_indices": global_inactive_indices,
        "global_inactive_names_sha256": union[
            "global_inactive_names_sha256"
        ],
        "per_rank_active_gradients": list(union["per_rank_active_gradients"]),
        "materialized_zero_gradient_tensors_by_rank": list(
            union["materialized_zero_gradient_tensors_by_rank"]
        ),
        "bucket_plan_sha256": plan_sha256,
        "collective_buckets": len(buckets),
        "all_reduce_bytes": total_bytes,
    }


def cuda_memory_snapshot(context: DistributedTrainingContext) -> Mapping[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(context.device)
    return {
        "process_rank": context.process_rank,
        "local_rank": context.local_rank,
        "device_index": int(context.device.index),
        "allocated_bytes": int(torch.cuda.memory_allocated(context.device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(context.device)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(context.device)
        ),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(context.device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def tensor_mapping_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(_canonical_json(list(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(str(key))
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                update(child)
            return
        if isinstance(item, (str, int, float, bool)) or item is None:
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Cannot hash a non-finite scalar")
            digest.update(_canonical_json(item).encode("ascii"))
            digest.update(b"\0")
            return
        raise TypeError(f"Unsupported optimizer-state value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()
