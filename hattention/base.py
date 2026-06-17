import os
import math
import torch
from enum import Enum
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from typing_extensions import Self


class HType(Enum):
    WEAK = 0
    STRONG = 1


class HStruct(Enum):
    MAMBA2 = 0
    GDELTA = 1


CACHED_LEVELS_MATRICES = {
    (2048 , 2, HType.WEAK  , -1): "/export/share/experiments/20250202/llut/llut.length-2048.base-2.pth",
    (16384, 2, HType.WEAK  , -1): "/export/share/experiments/20250202/llut/llut.length-16384.base-2.pth",
    (16384, 2, HType.STRONG, -1): "/export/share/experiments/20250402/llut/llut.length-16384.base-2.strong.pth",
}


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


def floor_log(x: int, b: int) -> int:
    return math.floor(math.log(x, b))


def get_num_levels(length: int, base: int) -> int:
    return ceil_log(length, base) + 1


def get_level_index_weak(t: int, s: int, b: int) -> int:
    l = floor_log(t, b)
    v = b ** l
    if s < v:
        return l + 1
    return get_level_index_weak(t - v, s - v, b)


def get_level_index_strong(t: int, s: int, b: int) -> int:
    if b != 2:
        raise NotImplementedError
    l = floor_log(t, b)
    v = b ** max(l - 1, 0)
    r = t - b ** l
    if s < v:
        return l + 1
    elif s < v * 2 and r >= v:
        return l + 1
    elif s < v * 2 and r < v:
        return get_level_index_strong(t - v * 1, s - v * 1, b)
    else:
        return get_level_index_strong(t - v * 2, s - v * 2, b)


def get_level_index(t: int, s: int, b: int, htype: HType) -> int:
    if t <= s:
        raise ValueError
    if htype == HType.WEAK:
        return get_level_index_weak(t=t, s=s, b=b)
    if htype == HType.STRONG:
        return get_level_index_strong(t=t, s=s, b=b)
    raise ValueError


def make_masked_H_matrix(
    A: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch num_heads target_length source_length"]:
    batch, length, num_heads = A.shape
    H = torch.zeros(
        (batch, num_heads, length, length),
        dtype=A.dtype,
        device=A.device)

    for target in range(length):
        H[..., target, target] = L[:, target, :, 0]

        for source in range(target):
            # hierarchical matrix
            l = get_level_index(t=target, s=source, b=base, htype=htype)
            h_H = L[:, target, :, l]
            # sequentially semi-separable matrix
            h_SSS = torch.prod(A[:, source + 1: target + 1, :], dim=1)
            H[..., target, source] = h_H * h_SSS

    return H


@lru_cache(maxsize=10)
def make_levels_matrix(
    length: int,
    base: int,
    htype: HType,
    dtype: torch.dtype,
    device: torch.device,
    default: int = -1,
    clamp_min: Optional[int] = None,
    file_name: Optional[str] = None,
    cached_length: Optional[int] = 16384,
) -> Int[torch.Tensor, "length length"]:

    if cached_length is not None:
        cached_levels = make_levels_matrix(
            length=cached_length,
            base=base,
            htype=htype,
            dtype=dtype,
            device=device,
            default=default,
            clamp_min=clamp_min,
            file_name=file_name,
            cached_length=None)
        if not all([
            length <= cached_length,
            length <= cached_levels.shape[0],
            length <= cached_levels.shape[1]]):
            raise ValueError
        return cached_levels[:length, :length].contiguous()

    if (length, base, htype, default) in CACHED_LEVELS_MATRICES.keys():
        levels = torch.load(
            CACHED_LEVELS_MATRICES[(length, base, htype, default)],
            map_location=device,
            weights_only=True)
    else:
        levels = torch.full(
            (length, length),
            fill_value=default,
            dtype=dtype,
            device=device)
        for target in range(length):
            levels[target, target] = 0
            for source in range(target):
                levels[target, source] = get_level_index(
                    t=target,
                    s=source,
                    b=base,
                    htype=htype)

        if file_name is not None:
            if os.path.exists(file_name):
                raise ValueError(f"{file_name} exists.")
            torch.save(levels, file_name)
            print(f"Saved to {file_name}")

    if clamp_min is not None:
        levels = torch.clamp(levels, min=clamp_min)

    return levels


def check_make_levels_matrix_cached_length() -> None:
    for length in [1, 97, 100, 512, 777, 3333, 7777, 11111, 16383, 16384]:
        llut0 = make_levels_matrix(
            length=length,
            base=2,
            htype=HType.WEAK,
            dtype=torch.int64,
            device="cuda",
            clamp_min=0,
            cached_length=None)
        llut1 = make_levels_matrix(
            length=length,
            base=2,
            htype=HType.WEAK,
            dtype=torch.int64,
            device="cuda",
            clamp_min=0,
            cached_length=16384)
        print(f"{length}: {(llut0 == llut1).all()} {llut0.is_contiguous()} {llut1.is_contiguous()}")


def make_masked_log_H_matrix_v2(
    log_A: Float[torch.Tensor, "batch length num_heads"],
    log_L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch num_heads target_length source_length"]:
    length = log_A.shape[1]
    levels = make_levels_matrix(
        length=length,
        base=base,
        htype=htype,
        dtype=torch.int64,
        device=log_A.device)
    lengths = torch.arange(
        length,
        dtype=levels.dtype,
        device=levels.device)
    lengths = torch.unsqueeze(lengths, dim=1)
    lengths = lengths.expand_as(levels)
    levels = rearrange(levels, "t s -> (t s)")
    lengths = rearrange(lengths, "t s -> (t s)")

    log_A = rearrange(log_A, "b t h -> b h t")
    log_A_cumsum = torch.cumsum(log_A, dim=-1)
    log_A_segsum = log_A_cumsum[..., :, None] - log_A_cumsum[..., None, :]

    log_L_gather = log_L[:, lengths, :, levels]
    log_L_gather = rearrange(log_L_gather, "(t s) b h -> b h t s", t=length, s=length)

    log_H = log_A_segsum + log_L_gather

    mask = torch.ones(length, length, dtype=torch.bool, device=log_A.device)
    mask = torch.tril(mask, diagonal=0)
    return torch.masked_fill(log_H, ~mask, -torch.inf)


def make_masked_H_matrix_v3(
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch target_length source_length num_heads"]:
    length = G.shape[1]
    device = G.device
    levels = make_levels_matrix(
        length=length,
        base=base,
        htype=htype,
        dtype=torch.int64,
        device=device)
    lengths = torch.arange(
        length,
        dtype=levels.dtype,
        device=levels.device)
    lengths = torch.unsqueeze(lengths, dim=1)
    lengths = lengths.expand_as(levels)
    levels = rearrange(levels, "t s -> (t s)")
    lengths = rearrange(lengths, "t s -> (t s)")

    G_cumsum = torch.cumsum(G, dim=1)
    G_segsum = rearrange(G_cumsum, "b t h -> b t 1 h") - rearrange(G_cumsum, "b s h -> b 1 s h")
    G_segsum = G_segsum.exp()

    L_gather = L[:, lengths, :, levels]
    L_gather = rearrange(L_gather, "(t s) b h -> b t s h", t=length, s=length)

    H = G_segsum * L_gather
    mask = torch.ones(length, length, dtype=torch.bool, device=device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "t s -> 1 t s 1")
    return torch.masked_fill(H, ~mask, 0.)


def make_masked_H_tensor_dplr(
    K: Float[torch.Tensor, "batch length num_heads num_units"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch target_length source_length num_heads num_units num_units"]:
    # 1. each `A` is a (scaled) identity + (scaled) low-rank matrix (aI + bUU^T)
    # 2. each `L` is still a scalar
    # 3. `A` and `L` are no longer in log domain.
    batch, length, num_heads, num_units = K.shape
    dtype = K.dtype
    device = K.device
    levels = make_levels_matrix(
        length=length,
        base=base,
        htype=htype,
        dtype=torch.int64,
        device=device)
    lengths = torch.arange(
        length,
        dtype=levels.dtype,
        device=levels.device)
    lengths = torch.unsqueeze(lengths, dim=1)
    lengths = lengths.expand_as(levels)
    levels = rearrange(levels, "t s -> (t s)")
    lengths = rearrange(lengths, "t s -> (t s)")

    I = torch.eye(num_units, dtype=dtype, device=device)
    I = rearrange(I, "dm dn -> 1 1 1 dm dn")
    BKKT = einsum(B, K, K, "b t h, b t h dm, b t h dn -> b t h dm dn")
    IBKKT = I - BKKT

    # Since `A` are matrices now, we cannot use cumsum. Looping
    # over target and source times are too inefficient, so we will
    # compute one subdiagonal at a time using batch matmul.
    H = torch.zeros(
        (batch, num_heads, num_units, num_units, length, length),
        dtype=dtype,
        device=device)
    for t in range(length):
        if t == 0:
            H_subdiag = repeat(
                torch.eye(num_units, dtype=dtype, device=device),
                "dm dn -> b h dm dn t",
                b=batch, h=num_heads, t=length)
        elif t == 1:
            H_subdiag = einsum(
                H_subdiag[..., 1:],
                IBKKT[:, 1:, ...],
                "b h dm dk t, b t h dk dn -> b h dm dn t")
        else:
            H_subdiag = einsum(
                H_subdiag[..., 1:],
                IBKKT[:, 1:-(t - 1), ...],
                "b h dm dk t, b t h dk dn -> b h dm dn t")
        H = H + torch.diag_embed(H_subdiag, offset=-t, dim1=-2, dim2=-1)

    G_cumsum = torch.cumsum(G, dim=1)
    G_segsum = rearrange(G_cumsum, "b t h -> b t 1 h") - rearrange(G_cumsum, "b s h -> b 1 s h")
    G_segsum = rearrange(G_segsum, "b t s h -> b t s h 1 1")
    G_segsum = G_segsum.exp()

    L_gather = L[:, lengths, :, levels]
    L_gather = rearrange(L_gather, "(t s) b h -> b t s h 1 1", t=length, s=length)

    H = rearrange(H, "b h dm dn t s -> b t s h dm dn")
    H = H * G_segsum * L_gather * rearrange(B, "b s h -> b 1 s h 1 1")

    mask = torch.ones(length, length, dtype=torch.bool, device=device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "t s -> 1 t s 1 1 1")
    return torch.masked_fill(H, ~mask, 0.)


def hattention_materialized(
    Q: Float[torch.Tensor, "batch length num_heads num_units"],
    K: Float[torch.Tensor, "batch length num_heads num_units"],
    V: Float[torch.Tensor, "batch length num_heads num_units"],
    A: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Tuple[Float[torch.Tensor, "batch length num_heads num_units"], Dict]:
    # H * (QK^T) V
    H = make_masked_H_matrix(A, L, base=base, htype=htype)
    QKT = einsum(Q, K, "b t h d, b s h d -> b h t s")
    MQKT = H * QKT
    Y = einsum(MQKT, V, "b h t s, b s h d -> b t h d")
    data = {"H": H, "QKT": QKT, "MQKT": MQKT}
    return Y, data


def hattention_materialized_v2(
    Q: Float[torch.Tensor, "batch length num_heads num_units"],
    K: Float[torch.Tensor, "batch length num_heads num_units"],
    V: Float[torch.Tensor, "batch length num_heads num_units"],
    log_A: Float[torch.Tensor, "batch length num_heads"],
    log_L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch length num_heads num_units"]:
    # H * (QK^T) V
    log_H = make_masked_log_H_matrix_v2(log_A, log_L, base=base, htype=htype)
    QKT = einsum(Q, K, "b t h d, b s h d -> b h t s")
    MQKT = torch.exp(log_H).to(dtype=QKT.dtype) * QKT
    return einsum(MQKT, V, "b h t s, b s h d -> b t h d")


def hattention_materialized_dplr(
    Q: Float[torch.Tensor, "batch length num_heads num_units"],
    K: Float[torch.Tensor, "batch length num_heads num_units"],
    V: Float[torch.Tensor, "batch length num_heads num_units"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch length num_heads num_units"]:
    # (Q H K^T) V
    H = make_masked_H_tensor_dplr(K=K, B=B, G=G, L=L, base=base, htype=htype)
    QHKT = einsum(Q, H, K, "b t h dkm, b t s h dkm dkn, b s h dkn -> b t s h")
    return einsum(QHKT, V, "b t s h, b s h dv -> b t h dv")


def hattention_materialized_dplr_v2(
    Q: Float[torch.Tensor, "batch length num_heads num_units"],
    K: Float[torch.Tensor, "batch length num_heads num_units"],
    V: Float[torch.Tensor, "batch length num_heads num_units"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
) -> Float[torch.Tensor, "batch length num_heads num_units"]:
    dtype = G.dtype
    device = G.device
    length = G.shape[1]
    mask_ = torch.ones(length, length, dtype=torch.bool, device=device)
    mask0 = torch.triu(mask_, diagonal=0)
    mask1 = torch.tril(mask_, diagonal=0)
    mask0 = rearrange(mask0, "s0 s1 -> 1 s0 s1 1")
    mask1 = rearrange(mask1, "t  s  -> 1 t  s  1")

    T = einsum(B, K, K, "b s0 h, b s0 h d, b s1 h d -> b s0 s1 h")
    T = -T.masked_fill(mask0, 0)
    for t in range(1, length):
        T[:, t, :t, :] = T[:, t, :t, :] + einsum(
            T[:, t, :,     :],
            T[:,    :, :t, :],
            "b s0 h, b s0 s1 h -> b s1 h")
    I = torch.eye(length, dtype=dtype, device=device)
    I = rearrange(I, "s0 s1 -> 1 s0 s1 1")
    T = T + I

    H = make_masked_H_matrix_v3(G=G, L=L, base=base, htype=htype)
    QKT = einsum(Q, K, "b t h d, b s h d -> b t s h")
    QKT = torch.masked_fill(QKT, ~mask1, 0.)
    QKTT = einsum(QKT, T, "b t s0 h, b s0 s1 h -> b t s1 h")
    QKTTH = QKTT * H
    BV = einsum(B, V, "b s h, b s h d -> b s h d")
    return einsum(QKTTH, BV, "b t s h, b s h dv -> b t h dv")


# 2/4