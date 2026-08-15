"""Fused CUDA write scan for the recurrent RWKV-MS state.

The CUDA implementation is deliberately narrow: it updates the recurrent state
without materializing the write-phase reads, which the projected-KV hybrid does
not consume. The reference path keeps CPU and unsupported CUDA environments
functional and is used by parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

import torch


@dataclass(frozen=True)
class RWKVWriteScanSupport:
    supported: bool
    reason: str = ""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: BaseException | None = None
_EXTENSION_WARNING_EMITTED = False


def _load_extension() -> Any:
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("RWKV write scan CUDA extension is unavailable") from _EXTENSION_ERROR
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("rwkv_ms_write_scan_cuda.cu")
        _EXTENSION = load(
            name="deltamem_rwkv_ms_write_scan_cuda_v1",
            sources=[str(source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("DELTA_MEM_VERBOSE_EXTENSIONS") == "1",
        )
        return _EXTENSION
    except BaseException as error:  # pragma: no cover - environment-specific
        _EXTENSION_ERROR = error
        raise RuntimeError("Failed to build RWKV write scan CUDA extension") from error


def support(
    state: torch.Tensor,
    w: torch.Tensor,
    *,
    rank: int,
) -> RWKVWriteScanSupport:
    if os.environ.get("DELTA_MEM_DISABLE_RWKV_WRITE_SCAN") == "1":
        return RWKVWriteScanSupport(False, "disabled by environment")
    if not state.is_cuda:
        return RWKVWriteScanSupport(False, "state tensor is not on CUDA")
    if state.dtype != torch.float32:
        return RWKVWriteScanSupport(False, "state tensor must be float32")
    if rank > 32:
        return RWKVWriteScanSupport(False, "fused scan supports rank <= 32")
    if w.device != state.device or w.dtype != torch.float32:
        return RWKVWriteScanSupport(False, "feature tensors must match CUDA float32 state")
    return RWKVWriteScanSupport(True)


def write_slot_indices(
    token_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    positions: torch.Tensor,
    chunk_size: int,
    num_slots: int,
) -> torch.Tensor:
    if token_mask is None:
        valid = torch.ones((batch_size, seq_len), device=positions.device, dtype=torch.bool)
    else:
        valid = token_mask.to(device=positions.device, dtype=torch.bool)
    offsets = valid.to(torch.long).cumsum(dim=1) - 1
    absolute = positions.view(batch_size, 1) + offsets
    slots = torch.div(absolute, chunk_size, rounding_mode="floor").remainder(num_slots)
    return torch.where(valid, slots, slots.new_full((), -1)).contiguous()


def _reference(
    state: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    keep: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    slots: torch.Tensor,
    erase_gate: float,
) -> torch.Tensor:
    current = state
    batch_size, seq_len, num_heads, rank = w.shape
    for token in range(seq_len):
        selected = slots[:, token]
        valid = selected.ge(0)
        slot_mask = torch.nn.functional.one_hot(
            selected.clamp_min(0), num_classes=current.size(2)
        ).to(dtype=current.dtype)
        slot_mask = slot_mask * valid.unsqueeze(-1).to(dtype=current.dtype)
        current_slot = current.gather(
            2,
            selected.clamp_min(0).view(batch_size, 1, 1, 1, 1)
            .expand(-1, num_heads, 1, rank, rank),
        ).squeeze(2)
        corr = torch.einsum("bhij,bhj->bhi", current_slot, a[:, token])
        write_outer = v[:, token].unsqueeze(-1) * k[:, token].unsqueeze(-2)
        correction_outer = corr.unsqueeze(-1) * b[:, token].unsqueeze(-2)
        candidate = (
            keep[:, token].unsqueeze(-1) * w[:, token].unsqueeze(-2) * current_slot
            + write[:, token].unsqueeze(-1) * write_outer
            + erase_gate
            * erase[:, token].unsqueeze(-1)
            * correction_outer
        )
        candidate = candidate * valid.view(batch_size, 1, 1, 1).to(candidate.dtype)
        updated = current * (1.0 - slot_mask.view(batch_size, 1, -1, 1, 1))
        updated = updated + candidate.unsqueeze(2) * slot_mask[:, None, :, None, None]
        current = updated
    return current


class _FusedWriteScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, w, k, v, a, b, keep, erase, write, slots, erase_gate):
        extension = _load_extension()
        final_state, history = extension.forward(
            state.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(),
            a.contiguous(), b.contiguous(), keep.contiguous(), erase.contiguous(),
            write.contiguous(), slots.contiguous(), float(erase_gate), True,
        )
        ctx.save_for_backward(history, w, k, v, a, b, keep, erase, write, slots)
        ctx.erase_gate = float(erase_gate)
        return final_state

    @staticmethod
    def backward(ctx, grad_final_state):
        history, w, k, v, a, b, keep, erase, write, slots = ctx.saved_tensors
        gradients = _load_extension().backward(
            grad_final_state.contiguous(), history, w, k, v, a, b, keep,
            erase, write, slots, ctx.erase_gate,
        )
        return (*gradients, None, None)


def fused_write_scan(
    state: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    keep: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    slots: torch.Tensor,
    erase_gate: float,
) -> torch.Tensor:
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (state, w, k, v, a, b, keep, erase, write)
    ):
        return _FusedWriteScan.apply(
            state, w, k, v, a, b, keep, erase, write, slots, erase_gate
        )
    return _load_extension().forward(
        state.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(),
        a.contiguous(), b.contiguous(), keep.contiguous(), erase.contiguous(),
        write.contiguous(), slots.contiguous(), float(erase_gate), False,
    )[0]


def rwkv_ms_write_scan(
    state: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    keep: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    slots: torch.Tensor,
    erase_gate: float,
) -> torch.Tensor:
    global _EXTENSION_WARNING_EMITTED
    support_info = support(state, w, rank=int(state.size(-1)))
    if support_info.supported:
        try:
            return fused_write_scan(
                state, w, k, v, a, b, keep, erase, write, slots, erase_gate
            )
        except RuntimeError:
            if _EXTENSION_ERROR is None:
                raise
            if not _EXTENSION_WARNING_EMITTED:  # pragma: no cover - environment-specific
                warnings.warn(
                    "RWKV write scan CUDA extension could not be built; using the "
                    "PyTorch reference scan",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _EXTENSION_WARNING_EMITTED = True
    return _reference(
        state, w, k, v, a, b, keep, erase, write, slots, erase_gate
    )
