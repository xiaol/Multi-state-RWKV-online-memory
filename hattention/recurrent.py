import torch
from einops import einsum, rearrange
from jaxtyping import Float
from typing import List, Tuple, Optional, Union
from typing_extensions import Self
from hattention.base import HType, HStruct


class HState(object):

    def __init__(
        self,
        base: int,
        htype: HType,
        hstruct: HStruct,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if htype != HType.WEAK and base != 2:
            raise NotImplementedError
        if not isinstance(shape, tuple):
            shape = tuple(shape)
        self.base = base
        self.htype = htype
        self.hstruct = hstruct
        self.shape = shape
        self.dtype = dtype
        self.device = device

        self.max_num_levels = shape[-1]
        self.states = torch.zeros(
            self.shape,
            dtype=self.dtype,
            device=self.device)
        self.counts = torch.zeros(
            self.max_num_levels,
            dtype=torch.int32,
            device=self.device)

    def cascade_weak(self) -> None:
        for level in range(self.max_num_levels):
            capacity = self.base ** level
            if self.counts[level] == capacity:
                level_next = level + 1
                self.states[..., level_next] = self.states[..., level_next] + self.states[..., level]
                self.counts[     level_next] = self.counts[     level_next] + self.counts[     level]
                self.states[..., level] = torch.zeros_like(self.states[..., level])
                self.counts[     level] = 0
            elif self.counts[level] < capacity:
                break
            else:
                raise ValueError

    def cascade_strong(self) -> None:
        # 0: carry states and counts from last level
        # 1: overflow states and counts from current level
        temp_states = torch.zeros_like(self.states[..., :2])
        temp_counts = torch.zeros_like(self.counts[     :2])
        for level in range(self.max_num_levels):
            capacity = self.base ** max(level - 1, 0)
            if self.counts[level] + temp_counts[0] > capacity or level == 0:
                temp_states[..., 1] = self.states[..., level]
                temp_counts[     1] = self.counts[     level]
                self.states[..., level] = temp_states[..., 0]
                self.counts[     level] = temp_counts[     0]
                temp_states[..., 0] = temp_states[..., 1]
                temp_counts[     0] = temp_counts[     1]
            else:
                self.states[..., level] = self.states[..., level] + temp_states[..., 0]
                self.counts[     level] = self.counts[     level] + temp_counts[     0]
                temp_states[..., 0] = torch.zeros_like(temp_states[..., 0])
                temp_counts[     0] = torch.zeros_like(temp_counts[     0])

    def insert(
        self,
        gate: Union[Float[torch.Tensor, "batch num_heads"],
                    Float[torch.Tensor, "batch num_heads num_units_head num_units_head"]],
        value: Float[torch.Tensor, "batch num_heads num_units_state num_units_head"],
    ) -> None:
        if self.htype == HType.WEAK:
            self.cascade_weak()
        else:
            self.cascade_strong()

        if self.hstruct == HStruct.MAMBA2:
            gate = rearrange(gate, "b h -> b h 1 1 1")
            self.states = gate * self.states
        else:
            self.states = einsum(gate, self.states, "b h dkm dkn, b h dkn dv l -> b h dkm dv l")

        self.states[..., 0] = value
        self.counts[     0] = self.counts[0] + 1

    def dot(
        self,
        other: Float[torch.Tensor, "batch num_heads num_units_state num_units_head num_levels"],
    ) -> Float[torch.Tensor, "batch num_heads num_units_state num_units_head"]:
        return einsum(self.states, other, "b h dk dv l, b h dk dv l -> b h dk dv")

    def reset_states(self) -> None:
        self.states = torch.zeros_like(self.states)

    def replace(self, other: "HState") -> None:
        if not all([
            tuple(self.shape) == tuple(other.shape),
            self.base == other.base,
            self.htype == other.htype,
            self.dtype == other.dtype,
            self.device == other.device]):
            raise ValueError("Shape, dtype or device mismatch")

        # self.reset_states()
        self.states = other.states

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> "HState":
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        new_state = HState(
            base=self.base,
            htype=self.htype,
            hstruct=self.hstruct,
            shape=self.shape,
            dtype=dtype,
            device=device)
        new_state.states = self.states.to(dtype=dtype, device=device)
        new_state.counts = self.counts.to(dtype=torch.int32, device=device)
        return new_state


def step_state(
    S: HState,
    k: Float[torch.Tensor, "batch num_heads num_units"],
    v: Float[torch.Tensor, "batch num_heads num_units"],
    a: Float[torch.Tensor, "batch num_heads"],
    b: Optional[Float[torch.Tensor, "batch num_heads"]] = None,
) -> HState:
    if S.hstruct == HStruct.MAMBA2:
        if b is not None:
            raise ValueError
        gate = a
        value = einsum(k, v, "b h dk, b h dv -> b h dk dv")
    else:
        if b is None:
            raise ValueError
        I = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
        I = rearrange(I, "dm dn -> 1 1 dm dn")
        bkkt = einsum(b, k, k, "b h, b h dm, b h dn -> b h dm dn")
        gate = rearrange(a, "b h -> b h 1 1")
        gate = rearrange(gate * (I - bkkt), "b h dm dn -> b h dm dn")
        value = einsum(b, k, v, "b h, b h dk, b h dv -> b h dk dv")
    S.insert(gate=gate, value=value)
    return S


def step_output(
    S: HState,
    q: Float[torch.Tensor, "batch num_heads num_units"],
    l: Float[torch.Tensor, "batch num_heads num_levels"],
) -> Float[torch.Tensor, "batch num_heads num_units"]:
    l = rearrange(l, "b h l -> b h 1 1 l")
    Sl = S.dot(l)
    return einsum(q, Sl, "b h d, b h d dv -> b h dv")


def hattention_recurrent(
    Q: Float[torch.Tensor, "batch length num_heads num_units_state"],
    K: Float[torch.Tensor, "batch length num_heads num_units_state"],
    V: Float[torch.Tensor, "batch length num_heads num_units_head"],
    A: Float[torch.Tensor, "batch length num_heads"],
    B: Optional[Float[torch.Tensor, "batch length num_heads"]],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    base: int,
    htype: HType,
    hstruct: HStruct,
) -> Tuple[Float[torch.Tensor, "batch length num_heads num_units_head"], HState]:
    if not all([
        Q.shape == K.shape,
        Q.shape[:-1] == V.shape[:-1],
        Q.shape[:-1] == A.shape,
        Q.shape[:-1] == L.shape[:-1],
        A.dtype == torch.float32,
        Q.dtype == K.dtype,
        Q.dtype == V.dtype,
        Q.dtype == L.dtype]):
        raise ValueError("Invalid shape")

    length = Q.shape[1]
    state_dtype = A.dtype
    output_dtype = V.dtype

    S = HState(
        base=base,
        htype=htype,
        hstruct=hstruct,
        shape=(Q.shape[0], V.shape[2], Q.shape[3], V.shape[3], L.shape[-1]),
        dtype=state_dtype,
        device=Q.device)
    Ys = []
    for t in range(length):
        S = step_state(
            S,
            K[:, t, ...].to(dtype=state_dtype),
            V[:, t, ...].to(dtype=state_dtype),
            A[:, t, ...],
            B[:, t, ...] if B is not None else None)
        Y = step_output(
            S.to(dtype=output_dtype),
            Q[:, t, ...],
            L[:, t, ...])
        Ys.append(Y)

    return torch.stack(Ys, dim=1), S
