from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_MEMORY_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/from_gguf/"
    "gemma-4-E4B-it-rwkv-ms-memory"
)
DEFAULT_OUTPUT = Path(".openresearch/artifacts/rwkv_ms_math_fixture.json")

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from hrm_rwkv7 import HRMRWKV7LowRankCore  # noqa: E402


TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, obj: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return TORCH_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_adapter(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def raw_tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bool:
        raw = tensor.numpy().tobytes()
    else:
        raw = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    flat = tensor.reshape(-1)
    if tensor.dtype == torch.bool:
        data = [bool(item) for item in flat.tolist()]
    elif tensor.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        data = [int(item) for item in flat.tolist()]
    else:
        data = [float(item) for item in flat.float().tolist()]
    return {
        "shape": list(tensor.shape),
        "dtype": dtype_name(tensor.dtype),
        "sha256": raw_tensor_sha256(tensor),
        "data": data,
    }


def tensor_from_record(record: dict[str, Any]) -> torch.Tensor:
    dtype_text = str(record["dtype"])
    if dtype_text == "bool":
        dtype = torch.bool
    elif dtype_text in {"int8", "int16", "int32", "int64", "uint8"}:
        dtype = getattr(torch, dtype_text)
    else:
        dtype = dtype_from_name(dtype_text)
    return torch.tensor(record["data"], dtype=dtype).reshape(tuple(record["shape"])).contiguous()


def flatten_tensor_records(obj: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(obj, dict) and {"shape", "dtype", "data"}.issubset(obj):
        return {prefix: obj}
    result: dict[str, dict[str, Any]] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_tensor_records(value, name))
    return result


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    numeric = tensor.float() if tensor.dtype != torch.bool else tensor.to(torch.float32)
    return {
        "shape": list(tensor.shape),
        "dtype": dtype_name(tensor.dtype),
        "numel": int(tensor.numel()),
        "sum": float(numeric.sum().item()) if tensor.numel() else 0.0,
        "sum_abs": float(numeric.abs().sum().item()) if tensor.numel() else 0.0,
        "max_abs": float(numeric.abs().max().item()) if tensor.numel() else 0.0,
        "sha256": raw_tensor_sha256(tensor),
    }


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{int(layer)}.self_attn."


def require_tensor(adapter: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    tensor = adapter.get(name)
    if not torch.is_tensor(tensor):
        raise KeyError(f"Missing adapter tensor: {name}")
    return tensor.detach().cpu().contiguous()


def load_layer_artifacts(memory_dir: Path, layer: int, dtype: torch.dtype) -> dict[str, Any]:
    memory_dir = memory_dir.expanduser().resolve()
    config = read_json(memory_dir / "delta_mem_config.json")
    metadata = read_json(memory_dir / "adapter_metadata.json")
    adapter = load_adapter(memory_dir / "delta_mem_adapter.pt")
    if not isinstance(adapter, dict):
        raise TypeError(f"Expected a flat adapter state dict, got {type(adapter)!r}")

    prefix = layer_prefix(layer)
    rank = int(config["rank"])
    num_state_heads = int(config.get("num_state_heads", 1))
    state_read_dim = rank * num_state_heads
    target_layers = [int(item) for item in config.get("target_layers", [])]
    n_layer = max(target_layers) + 1 if target_layers else int(layer) + 1

    core = HRMRWKV7LowRankCore(
        dim=state_read_dim,
        head_size=rank,
        layer_id=int(layer),
        n_layer=n_layer,
    )
    core_prefix = prefix + "hrm_rwkv7_core."
    core_state: dict[str, torch.Tensor] = {}
    for name, tensor in adapter.items():
        if name.startswith(core_prefix):
            core_state[name[len(core_prefix):]] = tensor.detach().cpu().to(dtype=dtype).contiguous()
    missing, unexpected = core.load_state_dict(core_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Could not load RWKV core tensors: missing={missing}, unexpected={unexpected}")
    core.to(dtype=dtype)
    core.eval()

    params = {
        "memory_q_proj": require_tensor(adapter, prefix + "memory_q_proj").to(dtype=dtype),
        "memory_k_proj": require_tensor(adapter, prefix + "memory_k_proj").to(dtype=dtype),
        "memory_v_proj": require_tensor(adapter, prefix + "memory_v_proj").to(dtype=dtype),
        "beta_proj": require_tensor(adapter, prefix + "beta_proj").to(dtype=dtype),
        "beta_bias": require_tensor(adapter, prefix + "beta_bias").to(dtype=dtype),
        "delta_q_proj": require_tensor(adapter, prefix + "delta_q_proj").to(dtype=dtype),
        "delta_k_proj": require_tensor(adapter, prefix + "delta_k_proj").to(dtype=dtype),
        "delta_v_proj": require_tensor(adapter, prefix + "delta_v_proj").to(dtype=dtype),
        "delta_o_proj": require_tensor(adapter, prefix + "delta_o_proj").to(dtype=dtype),
    }
    if not bool(config.get("couple_lambda", True)):
        params["lambda_proj"] = require_tensor(adapter, prefix + "lambda_proj").to(dtype=dtype)
        params["lambda_bias"] = require_tensor(adapter, prefix + "lambda_bias").to(dtype=dtype)

    tensor_names = sorted(name for name in adapter if name.startswith(prefix))
    tensor_hashes = {
        name: raw_tensor_sha256(require_tensor(adapter, name))
        for name in tensor_names
    }
    return {
        "memory_dir": memory_dir,
        "config": config,
        "metadata": metadata,
        "core": core,
        "params": params,
        "tensor_names": tensor_names,
        "tensor_hashes": tensor_hashes,
        "hidden_size": int(params["memory_v_proj"].shape[1]),
        "state_read_dim": state_read_dim,
        "rank": rank,
        "num_state_heads": num_state_heads,
    }


def parse_token_mask(value: str, *, batch_size: int, seq_len: int) -> torch.Tensor | None:
    if value == "none":
        return None
    if value == "auto":
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        if seq_len >= 3:
            mask[:, seq_len // 2] = False
        if batch_size > 1 and seq_len >= 2:
            mask[1::2, -1] = False
        return mask
    bits = [item.strip() for item in value.split(",") if item.strip()]
    if len(bits) != seq_len:
        raise ValueError(f"Token mask length {len(bits)} does not match seq_len {seq_len}")
    row = torch.tensor([item not in {"0", "false", "False"} for item in bits], dtype=torch.bool)
    return row.view(1, seq_len).expand(batch_size, seq_len).clone()


def make_deterministic_inputs(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_state_heads: int,
    num_states: int,
    rank: int,
    dtype: torch.dtype,
    token_mask: torch.Tensor | None,
    initial_position: int,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    hidden = torch.randn(batch_size, seq_len, hidden_size, generator=generator, dtype=torch.float32) * 0.125
    feature_axis = torch.linspace(-1.0, 1.0, hidden_size, dtype=torch.float32).view(1, 1, hidden_size)
    time_axis = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    hidden = hidden + 0.025 * torch.sin(feature_axis * 7.0 + time_axis)
    initial_state = (
        torch.randn(
            batch_size,
            num_state_heads,
            num_states,
            rank,
            rank,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.03125
    )
    positions = torch.arange(batch_size, dtype=torch.long) + int(initial_position)
    return {
        "hidden_states": hidden.to(dtype=dtype),
        "initial_state": initial_state.to(dtype=dtype),
        "initial_positions": positions,
        "token_mask": token_mask,
    }


def normalize_memory_projection(
    projected: torch.Tensor,
    *,
    normalize_qk: bool,
    num_state_heads: int,
    rank: int,
) -> torch.Tensor:
    if not normalize_qk:
        return projected
    if num_state_heads > 1 and projected.size(-1) == num_state_heads * rank:
        shaped = projected.view(*projected.shape[:-1], num_state_heads, rank)
        shaped = F.normalize(torch.tanh(shaped), dim=-1, eps=1e-6)
        return shaped.reshape(*projected.shape[:-1], num_state_heads * rank)
    return F.normalize(torch.tanh(projected), dim=-1, eps=1e-6)


def memory_sequence_projections(
    hidden_states: torch.Tensor,
    params: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    num_state_heads = int(config.get("num_state_heads", 1))
    rank = int(config["rank"])
    gate_weights = [params["beta_proj"]]
    split_sizes = [int(params["beta_proj"].shape[0])]
    if not bool(config.get("couple_lambda", True)):
        gate_weights.append(params["lambda_proj"])
        split_sizes.append(int(params["lambda_proj"].shape[0]))
    packed_gates = F.linear(hidden_states, torch.cat(gate_weights, dim=0))
    gate_splits = torch.split(packed_gates, split_sizes, dim=-1)
    packed_memory = F.linear(
        hidden_states,
        torch.cat(
            [params["memory_q_proj"], params["memory_k_proj"], params["memory_v_proj"]],
            dim=0,
        ),
    )
    memory_q, memory_k, memory_v = torch.split(
        packed_memory,
        [rank * num_state_heads, rank * num_state_heads, rank * num_state_heads],
        dim=-1,
    )
    memory_q = normalize_memory_projection(
        memory_q,
        normalize_qk=bool(config.get("normalize_qk", True)),
        num_state_heads=num_state_heads,
        rank=rank,
    )
    memory_k = normalize_memory_projection(
        memory_k,
        normalize_qk=bool(config.get("normalize_qk", True)),
        num_state_heads=num_state_heads,
        rank=rank,
    )
    beta = torch.sigmoid(gate_splits[0] + params["beta_bias"].view(1, 1, -1)).unsqueeze(-1)
    if str(config.get("state_update_mode", "standard")) == "no_lambda":
        lam = torch.ones_like(beta)
    elif bool(config.get("couple_lambda", True)):
        lam = 1.0 - beta
    else:
        lam = torch.sigmoid(gate_splits[1] + params["lambda_bias"].view(1, 1, -1)).unsqueeze(-1)
    return {
        "memory_q": memory_q,
        "memory_k": memory_k,
        "memory_v": memory_v,
        "beta": beta,
        "lambda": lam,
    }


def project_heads(projected: torch.Tensor, *, num_state_heads: int, rank: int) -> torch.Tensor:
    return projected.view(projected.size(0), projected.size(1), num_state_heads, rank)


def update_coefficients(
    beta_seq: torch.Tensor,
    lambda_seq: torch.Tensor,
    *,
    rank: int,
    num_state_heads: int,
    rankwise_gates: bool,
    state_update_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta_rows = beta_seq.squeeze(-1) if beta_seq.ndim == 4 else beta_seq
    lambda_rows = lambda_seq.squeeze(-1) if lambda_seq.ndim == 4 else lambda_seq
    if num_state_heads > 1:
        gate_dim_per_head = rank if rankwise_gates else 1
        beta_rows = beta_rows.view(beta_rows.size(0), beta_rows.size(1), num_state_heads, gate_dim_per_head)
        lambda_rows = lambda_rows.view(
            lambda_rows.size(0),
            lambda_rows.size(1),
            num_state_heads,
            gate_dim_per_head,
        )
        if gate_dim_per_head == 1:
            beta_rows = beta_rows.expand(-1, -1, -1, rank)
            lambda_rows = lambda_rows.expand(-1, -1, -1, rank)
    else:
        if beta_rows.size(-1) == 1:
            beta_rows = beta_rows.expand(beta_rows.size(0), beta_rows.size(1), rank)
        if lambda_rows.size(-1) == 1:
            lambda_rows = lambda_rows.expand(lambda_rows.size(0), lambda_rows.size(1), rank)

    if state_update_mode == "standard":
        keep_seq = lambda_rows
        erase_seq = beta_rows
        write_seq = beta_rows
    elif state_update_mode == "lambda_outside":
        keep_seq = lambda_rows
        erase_seq = lambda_rows * beta_rows
        write_seq = beta_rows
    elif state_update_mode == "no_lambda":
        keep_seq = torch.ones_like(beta_rows)
        erase_seq = beta_rows
        write_seq = beta_rows
    else:
        raise ValueError(f"Unsupported state_update_mode: {state_update_mode}")

    if num_state_heads == 1:
        keep_seq = keep_seq.unsqueeze(2)
        erase_seq = erase_seq.unsqueeze(2)
        write_seq = write_seq.unsqueeze(2)
    return keep_seq, erase_seq, write_seq


def slot_indices(positions: torch.Tensor, *, chunk_size: int, num_states: int) -> torch.Tensor:
    return torch.div(positions, int(chunk_size), rounding_mode="floor").remainder(int(num_states))


def read_routes(
    slot_reads: torch.Tensor,
    query: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    rank: int,
    read_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = (slot_reads * query.unsqueeze(2)).sum(dim=-1) / math.sqrt(float(rank))
    if 0 < int(read_top_k) < scores.size(-1):
        top_scores, top_indices = torch.topk(scores, k=int(read_top_k), dim=-1)
        masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        scores = masked_scores.scatter(-1, top_indices, top_scores)
    routes = F.softmax(scores, dim=-1)
    if valid is not None:
        routes = routes * valid.view(valid.size(0), 1, 1).to(dtype=routes.dtype)
    return scores, routes


def rwkv_ms_scan(
    *,
    core: HRMRWKV7LowRankCore,
    state: torch.Tensor,
    positions: torch.Tensor,
    memory_source_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    lambda_seq: torch.Tensor,
    config: dict[str, Any],
    scan_config: dict[str, Any],
    token_mask: torch.Tensor | None,
    include_step_trace: bool,
) -> dict[str, torch.Tensor]:
    rank = int(config["rank"])
    num_state_heads = int(config.get("num_state_heads", 1))
    num_states = int(scan_config["rwkv_ms_num_states"])
    chunk_size = int(scan_config["rwkv_ms_chunk_size"])
    erase_gate = float(scan_config["rwkv_ms_erase_gate"])
    read_top_k = int(scan_config["rwkv_ms_read_top_k"])
    batch_size, seq_len, _ = memory_source_seq.shape

    with torch.no_grad():
        features = core.project(memory_source_seq)
        r_seq = project_heads(features.r, num_state_heads=num_state_heads, rank=rank)
        w_seq = project_heads(features.w, num_state_heads=num_state_heads, rank=rank)
        k_seq = project_heads(features.k, num_state_heads=num_state_heads, rank=rank)
        v_seq = project_heads(features.v, num_state_heads=num_state_heads, rank=rank)
        a_seq = project_heads(features.a, num_state_heads=num_state_heads, rank=rank)
        b_seq = project_heads(features.b, num_state_heads=num_state_heads, rank=rank)
        keep_seq, erase_seq, write_seq = update_coefficients(
            beta_seq,
            lambda_seq,
            rank=rank,
            num_state_heads=num_state_heads,
            rankwise_gates=bool(config.get("rankwise_gates", True)),
            state_update_mode=str(config.get("state_update_mode", "standard")),
        )

        current_state = state.clone()
        current_positions = positions.clone()
        raw_read_steps: list[torch.Tensor] = []
        read_route_steps: list[torch.Tensor] = []
        read_route_mean_steps: list[torch.Tensor] = []
        write_route_steps: list[torch.Tensor] = []
        slot_index_steps: list[torch.Tensor] = []
        w_decay_steps: list[torch.Tensor] = []
        trace: dict[str, list[torch.Tensor]] = {
            "state_before": [],
            "slot_reads": [],
            "read_scores": [],
            "candidate_state": [],
            "state_after": [],
            "positions_before": [],
            "positions_after": [],
        }

        for token_idx in range(seq_len):
            r_t = r_seq[:, token_idx]
            w_t = torch.exp(-torch.exp(w_seq[:, token_idx].float())).to(dtype=memory_source_seq.dtype)
            k_t = k_seq[:, token_idx]
            v_t = v_seq[:, token_idx]
            a_t = a_seq[:, token_idx]
            b_t = b_seq[:, token_idx]
            keep_t = keep_seq[:, token_idx]
            erase_t = erase_seq[:, token_idx]
            write_t = write_seq[:, token_idx]
            valid_t = None if token_mask is None else token_mask[:, token_idx]

            if include_step_trace:
                trace["state_before"].append(current_state.clone())
                trace["positions_before"].append(current_positions.clone())
            slot_reads = torch.einsum("bhsij,bhj->bhsi", current_state, r_t)
            scores, routes = read_routes(
                slot_reads,
                r_t,
                valid_t,
                rank=rank,
                read_top_k=read_top_k,
            )
            read_t = torch.einsum("bhs,bhsi->bhi", routes, slot_reads)
            raw_read_steps.append(read_t.reshape(batch_size, num_state_heads * rank))
            read_route_steps.append(routes)
            read_route_mean_steps.append(routes.mean(dim=1))

            slot_idx = slot_indices(current_positions, chunk_size=chunk_size, num_states=num_states)
            slot_mask = F.one_hot(slot_idx, num_classes=num_states).to(dtype=current_state.dtype)
            if valid_t is not None:
                slot_mask = slot_mask * valid_t.to(dtype=current_state.dtype).unsqueeze(-1)
            write_route_steps.append(slot_mask)
            slot_index_steps.append(slot_idx)
            w_decay_steps.append(w_t)

            correction_read = torch.einsum("bhsij,bhj->bhsi", current_state, a_t)
            write_outer = v_t.unsqueeze(2).unsqueeze(-1) * k_t.unsqueeze(2).unsqueeze(-2)
            correction_outer = correction_read.unsqueeze(-1) * b_t.unsqueeze(2).unsqueeze(-2)
            candidate_state = (
                keep_t.unsqueeze(2).unsqueeze(-1)
                * w_t.unsqueeze(2).unsqueeze(-2)
                * current_state
                + write_t.unsqueeze(2).unsqueeze(-1) * write_outer
                + erase_gate
                * erase_t.unsqueeze(2).unsqueeze(-1)
                * correction_outer
            )
            state_mask = slot_mask.view(batch_size, 1, num_states, 1, 1)
            current_state = candidate_state * state_mask + current_state * (1.0 - state_mask)
            if valid_t is None:
                current_positions = current_positions + 1
            else:
                current_positions = current_positions + valid_t.to(dtype=torch.long)

            if include_step_trace:
                trace["slot_reads"].append(slot_reads)
                trace["read_scores"].append(scores)
                trace["candidate_state"].append(candidate_state)
                trace["state_after"].append(current_state.clone())
                trace["positions_after"].append(current_positions.clone())

        raw_reads = torch.stack(raw_read_steps, dim=1)
        reads = core.readout(raw_reads, features.g)
        result = {
            "memory_source": memory_source_seq,
            "feature_r": features.r,
            "feature_w": features.w,
            "feature_k": features.k,
            "feature_v": features.v,
            "feature_a": features.a,
            "feature_b": features.b,
            "feature_g": features.g,
            "r_heads": r_seq,
            "w_heads": w_seq,
            "k_heads": k_seq,
            "v_heads": v_seq,
            "a_heads": a_seq,
            "b_heads": b_seq,
            "w_decay": torch.stack(w_decay_steps, dim=1),
            "keep": keep_seq,
            "erase": erase_seq,
            "write": write_seq,
            "raw_reads": raw_reads,
            "reads": reads,
            "read_routes_per_head": torch.stack(read_route_steps, dim=1),
            "read_routes": torch.stack(read_route_mean_steps, dim=1),
            "write_routes": torch.stack(write_route_steps, dim=1),
            "slot_indices": torch.stack(slot_index_steps, dim=1),
            "final_state": current_state,
            "final_positions": current_positions,
        }
        if include_step_trace:
            result.update({name: torch.stack(values, dim=1) for name, values in trace.items()})
        return result


def project_delta_heads(
    reads: torch.Tensor,
    params: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    active = set(config.get("delta_heads", []))
    scaling = float(config["alpha"]) / float(config["rank"])
    if bool(config.get("trainable_delta_scale", False)):
        raise ValueError("Fixture generator does not support trainable_delta_scale checkpoints yet")
    outputs: dict[str, torch.Tensor] = {}
    for head_name, param_name in (
        ("q", "delta_q_proj"),
        ("k", "delta_k_proj"),
        ("v", "delta_v_proj"),
        ("o", "delta_o_proj"),
    ):
        if head_name in active:
            outputs[f"delta_{head_name}"] = F.linear(reads, params[param_name]) * scaling
    return outputs


def compute_expected_tensors(
    *,
    layer_artifacts: dict[str, Any],
    inputs: dict[str, torch.Tensor | None],
    scan_config: dict[str, Any],
    include_step_trace: bool,
) -> dict[str, torch.Tensor]:
    config = layer_artifacts["config"]
    params = layer_artifacts["params"]
    projections = memory_sequence_projections(inputs["hidden_states"], params, config)  # type: ignore[arg-type]
    scan = rwkv_ms_scan(
        core=layer_artifacts["core"],
        state=inputs["initial_state"],  # type: ignore[arg-type]
        positions=inputs["initial_positions"],  # type: ignore[arg-type]
        memory_source_seq=projections["memory_v"],
        beta_seq=projections["beta"],
        lambda_seq=projections["lambda"],
        config=config,
        scan_config=scan_config,
        token_mask=inputs["token_mask"],  # type: ignore[arg-type]
        include_step_trace=include_step_trace,
    )
    delta = project_delta_heads(scan["reads"], params, config)
    return {**projections, **scan, **delta}


def records_from_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {name: tensor_record(tensor) for name, tensor in sorted(tensors.items())}


def tensors_from_records(records: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {name: tensor_from_record(record) for name, record in records.items()}


def build_summary(*sections: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for section in sections:
        records.update(flatten_tensor_records(section))
    return {
        "tensor_count": len(records),
        "total_values": sum(int(math.prod(record["shape"])) for record in records.values()),
        "sha256": {name: record["sha256"] for name, record in sorted(records.items())},
    }


def compare_tensor_records(
    expected_records: dict[str, dict[str, Any]],
    actual_tensors: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
    max_errors: int,
) -> dict[str, Any]:
    errors: list[str] = []
    max_abs_diff = 0.0
    compared = 0
    for name, expected_record in sorted(expected_records.items()):
        expected = tensor_from_record(expected_record)
        actual = actual_tensors.get(name)
        if actual is None:
            errors.append(f"missing actual tensor: {name}")
            if len(errors) >= max_errors:
                break
            continue
        actual = actual.detach().cpu().to(dtype=expected.dtype).contiguous()
        if tuple(expected.shape) != tuple(actual.shape):
            errors.append(f"shape mismatch {name}: expected={tuple(expected.shape)} actual={tuple(actual.shape)}")
            if len(errors) >= max_errors:
                break
            continue
        if expected.dtype == torch.bool:
            equal = torch.equal(expected, actual)
            diff = 0.0 if equal else 1.0
        elif expected.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            equal = torch.equal(expected, actual)
            diff = float((expected.to(torch.float32) - actual.to(torch.float32)).abs().max().item()) if expected.numel() else 0.0
        else:
            delta = (expected.float() - actual.float()).abs()
            diff = float(delta.max().item()) if delta.numel() else 0.0
            equal = torch.allclose(expected.float(), actual.float(), rtol=rtol, atol=atol)
        max_abs_diff = max(max_abs_diff, diff)
        compared += 1
        if not equal:
            errors.append(f"tensor mismatch {name}: max_abs_diff={diff:.9g}")
            if len(errors) >= max_errors:
                break
    extra = sorted(set(actual_tensors) - set(expected_records))
    if extra:
        errors.extend(f"extra actual tensor: {name}" for name in extra[: max(0, max_errors - len(errors))])
    return {
        "ok": not errors,
        "compared_tensors": compared,
        "max_abs_diff": max_abs_diff,
        "errors": errors,
    }
