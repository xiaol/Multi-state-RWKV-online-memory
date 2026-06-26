from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HRMRWKV7Features(NamedTuple):
    r: Tensor
    w: Tensor
    k: Tensor
    v: Tensor
    a: Tensor
    b: Tensor
    g: Tensor


def _ortho_init_(x: Tensor, scale: float = 1.0) -> Tensor:
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(x, gain=gain * scale)
        else:
            raise ValueError(f"Unsupported tensor shape for RWKV-7 orthogonal init: {shape}")
    return x


def _time_shift_delta(x: Tensor) -> Tensor:
    xx = torch.empty_like(x)
    xx[:, 0] = -x[:, 0]
    if x.size(1) > 1:
        xx[:, 1:] = x[:, :-1] - x[:, 1:]
    return xx


class HRMRWKV7LowRankCore(nn.Module):
    """HRM-Text-derived RWKV-7 memory core.

    This is a minimal, local port of HRM-Text's `RWKV7TimeMix` projection stack
    and `RWKVStateMemory` read normalization/output path. Delta-Mem owns the
    persistent multi-state routing around it.
    """

    def __init__(
        self,
        *,
        dim: int,
        head_size: int,
        layer_id: int = 0,
        n_layer: int = 1,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if head_size < 1:
            raise ValueError("head_size must be >= 1")
        if dim % head_size != 0:
            raise ValueError(f"dim ({dim}) must be divisible by head_size ({head_size})")
        if n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if layer_id < 0:
            raise ValueError("layer_id must be >= 0")
        self.dim = int(dim)
        self.head_size = int(head_size)
        self.n_head = self.dim // self.head_size
        # Layer-depth position drives the official RWKV-7 init schedules below.
        # Defaults (layer_id=0, n_layer=1) reproduce the shallowest-layer init,
        # which matches the pre-fix behavior for backward compatibility.
        self.layer_id = int(layer_id)
        self.n_layer = int(n_layer)

        decay_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        aaa_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        gate_lora_dim = max(32, int(round((5.0 * (dim**0.5)) / 32) * 32))

        self.x_r = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_w = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_k = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_v = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_a = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_g = nn.Parameter(torch.empty(dim, dtype=torch.float32))

        self.w1 = nn.Parameter(torch.empty(dim, decay_lora_dim, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty(decay_lora_dim, dim, dtype=torch.float32))
        self.w0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.a1 = nn.Parameter(torch.empty(dim, aaa_lora_dim, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.empty(aaa_lora_dim, dim, dtype=torch.float32))
        self.a0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.g1 = nn.Parameter(torch.empty(dim, gate_lora_dim, dtype=torch.float32))
        self.g2 = nn.Parameter(torch.empty(gate_lora_dim, dim, dtype=torch.float32))
        self.k_k = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.k_a = nn.Parameter(torch.empty(dim, dtype=torch.float32))

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=64e-5)
        self.reset_parameters(dim**-0.5)

    def reset_parameters(self, init_std: float) -> None:
        device = self.x_r.device
        dim = self.dim
        # Official RWKV-7 layer-depth schedules (RWKV-LM v7 train_temp/src/model.py).
        # ratio_0_to_1 ramps 0->1 with depth; ratio_1_to_almost0 ramps 1->~0.
        ratio_0_to_1 = self.layer_id / max(self.n_layer - 1, 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / max(self.n_layer, 1))
        ddd = torch.arange(dim, device=device, dtype=torch.float32) / dim
        linear = torch.arange(dim, device=device, dtype=torch.float32) / max(dim - 1, 1) - 0.5
        zigzag = torch.arange(dim, device=device, dtype=torch.float32) % self.head_size
        zigzag = (zigzag - ((self.head_size - 1) / 2)) / max((self.head_size - 1) / 2, 1.0)
        zigzag = zigzag * zigzag.abs()
        # Depth-dependent decay exponent: deeper layers get longer memory horizons.
        decay_base = torch.arange(dim, device=device, dtype=torch.float32) / max(dim - 1, 1)
        decay = -6 + 6 * torch.pow(decay_base, 1.0 + ratio_0_to_1**0.3)
        with torch.no_grad():
            self.x_r.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.w0.copy_(decay + 0.5 + zigzag * 2.5)
            self.a0.copy_(torch.zeros_like(linear) - 0.19 + zigzag * 0.3 + linear * 0.4)
            self.k_k.copy_(torch.zeros_like(linear) + 0.71 - linear * 0.1)
            self.k_a.fill_(1.02)
            self.w1.zero_()
            _ortho_init_(self.w2, 0.1)
            self.a1.zero_()
            _ortho_init_(self.a2, 0.1)
            self.g1.zero_()
            _ortho_init_(self.g2, 0.1)
        # Official r/k/v scale: receptance/value ~uniform(+-0.5/sqrt(dim)),
        # key 10x smaller (+-0.05/sqrt(dim)); init_std is passed as dim**-0.5.
        nn.init.uniform_(self.receptance.weight, -0.5 * init_std, 0.5 * init_std)
        nn.init.uniform_(self.key.weight, -0.05 * init_std, 0.05 * init_std)
        nn.init.uniform_(self.value.weight, -0.5 * init_std, 0.5 * init_std)
        nn.init.zeros_(self.output.weight)
        self.ln_x.reset_parameters()

    def project(self, x: Tensor) -> HRMRWKV7Features:
        batch_size, seq_len, dim = x.shape
        xx = _time_shift_delta(x)
        xr = x + xx * self.x_r.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        xw = x + xx * self.x_w.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        xk = x + xx * self.x_k.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        xv = x + xx * self.x_v.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        xa = x + xx * self.x_a.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        xg = x + xx * self.x_g.to(dtype=x.dtype, device=x.device).view(1, 1, -1)

        r = self.receptance(xr)
        w = self.w0.to(dtype=x.dtype, device=x.device).view(1, 1, -1) + (
            torch.tanh(xw @ self.w1.to(dtype=x.dtype, device=x.device))
            @ self.w2.to(dtype=x.dtype, device=x.device)
        )
        k = self.key(xk)
        v = self.value(xv)
        w = -F.softplus(-w.float()).to(dtype=x.dtype) - 0.5
        a_gate = torch.sigmoid(
            self.a0.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
            + (xa @ self.a1.to(dtype=x.dtype, device=x.device))
            @ self.a2.to(dtype=x.dtype, device=x.device)
        )
        g = torch.sigmoid(xg @ self.g1.to(dtype=x.dtype, device=x.device)) @ self.g2.to(
            dtype=x.dtype,
            device=x.device,
        )
        kk = k * self.k_k.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        kk = F.normalize(
            kk.reshape(batch_size, seq_len, self.n_head, self.head_size),
            dim=-1,
            p=2.0,
        ).reshape(batch_size, seq_len, dim)
        k = k * (1 + (a_gate - 1) * self.k_a.to(dtype=x.dtype, device=x.device).view(1, 1, -1))
        return HRMRWKV7Features(r=r, w=w, k=k, v=v, a=-kk, b=kk * a_gate, g=g)

    def readout(self, reads: Tensor, g: Tensor) -> Tensor:
        batch_size, seq_len, dim = reads.shape
        if seq_len == 0:
            return reads
        normalized = F.group_norm(
            reads.reshape(batch_size * seq_len, dim),
            num_groups=self.n_head,
            weight=self.ln_x.weight.to(device=reads.device, dtype=reads.dtype),
            bias=None,
            eps=self.ln_x.eps,
        ).reshape(batch_size, seq_len, dim)
        return self.output(normalized * g)
