"""Delta-rule / RWKV-style multi-state memory read through attention KV slots on a frozen base.

Mechanism
---------
Write pass: the passage is run through the frozen model once. At every wrapped layer the
residual-stream input of that layer is projected into per-head keys, values, a decay and a
write strength, and folded into one or more matrix states with the gated delta rule

    S_t = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t * k_t v_t^T .

Multi-state: tokens are routed to one of `n_states` states per head, either by contiguous
chunks of the passage ("chunk"), by cosine similarity to learned anchors ("cosine"), or
there is a single state ("single").

Clear: the passage never enters the read context. Only the states survive.

Read pass: each state is queried with a learned query bank, and every retrieved vector is
mapped into the frozen layer's key and value space. Those vectors are appended as extra
key/value slots that the frozen attention attends to alongside the real tokens. Slot keys
are matched against the *unrotated* query (captured after `q_norm`), which is equivalent
to a rotary relative distance of zero, so the score does not depend on the query position.

No projection, norm, rotary, KV cache, or KV-sharing code in the base model is touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

ATTN_NAME = "memkv"
ROUTINGS = ("single", "chunk", "cosine")
READ_MODES = ("bank", "query")
MEMORY_KINDS = ("delta", "kvbank")
WRITE_SOURCES = ("residual", "attn_input")


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


@dataclass
class LayerContext:
    """Per-layer mutable runtime context shared between hooks and the attention function."""

    adapter: "LayerMemory"
    layer_idx: int
    capture: bool = False
    captured: torch.Tensor | None = None
    q_unrot: torch.Tensor | None = None
    slots: tuple[torch.Tensor, torch.Tensor] | None = None
    state: torch.Tensor | None = None
    stats: dict = field(default_factory=dict)


class LayerMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        mem_dim: int = 128,
        n_states: int = 1,
        slots_per_state: int = 16,
        routing: str = "single",
        route_temperature: float = 0.1,
        decay_bias: float = 4.0,
        beta_bias: float = 0.0,
        slot_bias: float = -6.0,
        chunk_size: int = 16,
        read_mode: str = "bank",
        memory_kind: str = "delta",
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        if read_mode not in READ_MODES:
            raise ValueError(f"read_mode must be one of {READ_MODES}")
        if memory_kind not in MEMORY_KINDS:
            raise ValueError(f"memory_kind must be one of {MEMORY_KINDS}")
        if memory_kind == "kvbank" and read_mode != "query":
            raise ValueError("kvbank memory requires read_mode='query'")
        self.read_mode = read_mode
        self.memory_kind = memory_kind
        if routing not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}")
        if routing == "single" and n_states != 1:
            raise ValueError("routing='single' requires n_states=1")
        if routing != "single" and n_states < 2:
            raise ValueError("multi-state routing requires n_states>=2")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mem_dim = mem_dim
        self.n_states = n_states
        self.slots_per_state = slots_per_state
        self.routing = routing
        self.route_temperature = route_temperature

        self.w_k = nn.Linear(hidden_size, num_heads * mem_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, num_heads * mem_dim, bias=False)
        self.w_gate = nn.Linear(hidden_size, num_heads * 2, bias=True)
        with torch.no_grad():
            nn.init.normal_(self.w_k.weight, std=hidden_size**-0.5)
            nn.init.normal_(self.w_v.weight, std=hidden_size**-0.5)
            nn.init.zeros_(self.w_gate.weight)
            bias = self.w_gate.bias.view(num_heads, 2)
            bias[:, 0].fill_(decay_bias)
            bias[:, 1].fill_(beta_bias)
        if routing == "cosine":
            self.anchors = nn.Parameter(torch.randn(num_heads, n_states, mem_dim))
        else:
            self.register_parameter("anchors", None)

        self.query_bank = nn.Parameter(torch.randn(n_states, num_heads, slots_per_state, mem_dim))
        # Query-conditioned read: map each frozen head's unrotated query into the memory key space.
        self.w_q = nn.Parameter(torch.randn(num_heads, head_dim, mem_dim) * (head_dim**-0.5))
        # kvbank diagnostic: softmax retrieval over the uncompressed passage tokens (log inverse temperature).
        self.kv_log_scale = nn.Parameter(torch.full((num_heads,), math.log(10.0)))
        self.read_scale = nn.Parameter(torch.ones(mem_dim))
        # Zero-init slot keys: every slot starts at exactly `slot_bias` logits regardless of the
        # base model's attention scaling, so an untrained memory is invisible to the frozen model.
        self.w_k_out = nn.Parameter(torch.zeros(num_heads, mem_dim, head_dim))
        self.w_v_out = nn.Parameter(torch.randn(num_heads, mem_dim, head_dim) * (0.1 * mem_dim**-0.5))
        self.slot_bias = nn.Parameter(torch.full((num_heads,), float(slot_bias)))

    @property
    def n_slots(self) -> int:
        return self.n_states * self.slots_per_state

    def zero_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch, self.num_heads, self.n_states, self.mem_dim, self.mem_dim, device=device
        )

    def routing_weights(
        self, k: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return [B, T, H, S] soft assignment of every token to a state."""
        b, t, h, _ = k.shape
        if self.routing == "single":
            w = torch.ones(b, t, h, 1, device=k.device)
        elif self.routing == "chunk":
            valid = mask.float()
            idx = valid.cumsum(dim=1) - 1.0
            length = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            state = torch.floor(idx * self.n_states / length).clamp(0, self.n_states - 1).long()
            w = F.one_hot(state, self.n_states).float()[:, :, None, :].expand(b, t, h, self.n_states)
        else:
            anchors = F.normalize(self.anchors, dim=-1)
            cos = torch.einsum("bthk,hsk->bths", k, anchors)
            w = torch.softmax(cos / self.route_temperature, dim=-1)
        return w * mask[:, :, None, None].float()

    def write_inputs(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden [B, T, hidden_size] into keys, values and per-state effective gates."""
        h32 = hidden.float()
        b, t, _ = h32.shape
        k = F.normalize(self.w_k(h32).view(b, t, self.num_heads, self.mem_dim), dim=-1)
        v = self.w_v(h32).view(b, t, self.num_heads, self.mem_dim)
        gate = self.w_gate(h32).view(b, t, self.num_heads, 2)
        alpha = torch.sigmoid(gate[..., 0])
        beta = torch.sigmoid(gate[..., 1])
        w = self.routing_weights(k, mask)
        # Effective per-state gates: a token routed with weight w decays its state by
        # 1 - w (1 - alpha) and writes with strength w * beta; unrouted states are untouched.
        a_eff = 1.0 - w * (1.0 - alpha[..., None])  # [B, T, H, S]
        b_eff = w * beta[..., None]
        return k, v, a_eff, b_eff

    def write(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, hidden_size] (any dtype), mask: [B, T] bool. Returns [B, H, S, dk, dv]."""
        k, v, a_eff, b_eff = self.write_inputs(hidden, mask)
        state = self.zero_state(hidden.shape[0], hidden.device)
        return scan(state, k, v, a_eff, b_eff, self.chunk_size)

    def read(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """state: [B, H, S, dk, dv] -> keys, values [B, H, S*M, head_dim] in fp32."""
        q = F.normalize(self.query_bank, dim=-1)  # [S, H, M, dk]
        r = torch.einsum("shmk,bhskv->bhsmv", q, state.float())
        b = r.shape[0]
        r = r.reshape(b, self.num_heads, self.n_slots, self.mem_dim)
        r = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * self.read_scale
        keys = torch.einsum("bhnv,hvd->bhnd", r, self.w_k_out)
        values = torch.einsum("bhnv,hvd->bhnd", r, self.w_v_out)
        return keys, values

    def query_read(self, state: torch.Tensor, q_unrot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Content-addressed read. state [B,H,S,dk,dv], q_unrot [B,H,Q,D] -> keys, values [B,H,Q,S,D].

        Every query position addresses each state with its own key (linear attention over the
        written passage) and receives one slot per state. The frozen attention still decides,
        through the slot key, how much of that retrieval to use.
        """
        q_mem = torch.einsum("bhqd,hdk->bhqk", q_unrot.float(), self.w_q)
        q_mem = F.normalize(q_mem, dim=-1)
        if self.memory_kind == "kvbank":
            keys_t, values_t, mask = state  # [B, T, H, dk], [B, T, H, dv], [B, T]
            logits = torch.einsum("bhqk,bthk->bhqt", q_mem, keys_t) * self.kv_log_scale.exp()[None, :, None, None]
            logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
            p = torch.softmax(logits, dim=-1)
            r = torch.einsum("bhqt,bthv->bhqv", p, values_t)[:, :, :, None, :]  # one slot per position
        else:
            r = torch.einsum("bhqk,bhskv->bhqsv", q_mem, state.float())
        r = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * self.read_scale
        keys = torch.einsum("bhqsv,hvd->bhqsd", r, self.w_k_out)
        values = torch.einsum("bhqsv,hvd->bhqsd", r, self.w_v_out)
        return keys, values


def scan(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a_eff: torch.Tensor,
    b_eff: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Gated delta rule over T tokens with chunk-wise activation checkpointing."""
    t = k.shape[1]
    for start in range(0, t, chunk_size):
        end = min(t, start + chunk_size)
        args = (state, k[:, start:end], v[:, start:end], a_eff[:, start:end], b_eff[:, start:end])
        if torch.is_grad_enabled() and any(x.requires_grad for x in args):
            state = checkpoint(_scan_chunk, *args, use_reentrant=False)
        else:
            state = _scan_chunk(*args)
    return state


def _scan_chunk(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a_eff: torch.Tensor,
    b_eff: torch.Tensor,
) -> torch.Tensor:
    """One chunk of the gated delta rule. state [B,H,S,dk,dv]; k,v [B,C,H,d]; gates [B,C,H,S]."""
    for step in range(k.shape[1]):
        kt = k[:, step]  # [B, H, dk]
        vt = v[:, step]  # [B, H, dv]
        a = a_eff[:, step][..., None, None]  # [B, H, S, 1, 1]
        bb = b_eff[:, step][..., None, None]
        k_state = torch.matmul(kt[:, :, None, None, :], state)  # k^T S -> [B, H, S, 1, dv]
        # S <- a S + b k (v - a k^T S)^T  ==  a (S - b k k^T S) + b k v^T
        update = vt[:, :, None, None, :] - a * k_state
        state = a * state + bb * kt[:, :, None, :, None] * update
    return state


def memkv_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if scaling is None:
        scaling = module.head_dim**-0.5
    n_rep = getattr(module, "num_key_value_groups", 1)
    k = repeat_kv(key, n_rep)
    v = repeat_kv(value, n_rep)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    q_len, k_len = query.shape[2], k.shape[2]
    if attention_mask is not None:
        mask = attention_mask[..., :k_len]
        if mask.dtype == torch.bool:
            mask = torch.where(mask, 0.0, torch.finfo(scores.dtype).min).to(scores.dtype)
        scores = scores + mask
    elif q_len > 1:
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device).tril(k_len - q_len)
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

    ctx: LayerContext | None = getattr(module, "_memkv", None)
    if ctx is None or (ctx.slots is None and ctx.state is None):
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=dropout, training=module.training)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).contiguous(), None

    if ctx.state is not None:
        q_unrot = ctx.q_unrot
        if q_unrot is None:
            raise RuntimeError(f"layer {ctx.layer_idx}: unrotated query was not captured")
        q_unrot = q_unrot.transpose(1, 2)  # [B, H, Q, D]
        if q_unrot.shape[2] != q_len:
            raise RuntimeError(f"layer {ctx.layer_idx}: captured query length {q_unrot.shape[2]} != {q_len}")
        k_mem, v_mem = ctx.adapter.query_read(ctx.state, q_unrot)  # [B, H, Q, S, D]
        mem_scores = torch.einsum("bhqd,bhqsd->bhqs", q_unrot.float(), k_mem) * scaling
        mem_scores = mem_scores + ctx.adapter.slot_bias.float()[None, :, None, None]
        full = torch.cat([scores.float(), mem_scores], dim=-1)
        probs = torch.softmax(full, dim=-1)
        p_real = probs[..., :k_len].to(v.dtype)
        p_mem = probs[..., k_len:]
        ctx.stats["mem_mass"] = p_mem.sum(-1).mean().detach()
        out = torch.matmul(p_real, v) + torch.einsum("bhqs,bhqsd->bhqd", p_mem, v_mem).to(v.dtype)
        return out.transpose(1, 2).contiguous(), None

    k_mem, v_mem = ctx.slots  # [B, H, N, D] fp32
    q_unrot = ctx.q_unrot
    if q_unrot is None:
        raise RuntimeError(f"layer {ctx.layer_idx}: unrotated query was not captured")
    q_unrot = q_unrot.transpose(1, 2)  # [B, H, Q, D]
    if q_unrot.shape[2] != q_len:
        raise RuntimeError(
            f"layer {ctx.layer_idx}: captured query length {q_unrot.shape[2]} != {q_len}"
        )
    mem_scores = torch.einsum("bhqd,bhnd->bhqn", q_unrot.float(), k_mem) * scaling
    mem_scores = mem_scores + ctx.adapter.slot_bias.float()[None, :, None, None]
    full = torch.cat([scores.float(), mem_scores], dim=-1)
    probs = torch.softmax(full, dim=-1)
    p_real = probs[..., :k_len].to(v.dtype)
    p_mem = probs[..., k_len:]
    ctx.stats["mem_mass"] = p_mem.sum(-1).mean().detach()
    out = torch.matmul(p_real, v) + torch.matmul(p_mem, v_mem).to(v.dtype)
    return out.transpose(1, 2).contiguous(), None


_REGISTERED = False


def register_attention() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    AttentionInterface.register(ATTN_NAME, memkv_attention)
    AttentionMaskInterface.register(ATTN_NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    _REGISTERED = True


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the text decoder layer stack (multimodal checkpoints also carry vision/audio stacks)."""
    expected = model.config.get_text_config().num_hidden_layers
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) == expected and hasattr(module[0], "self_attn")
    ]
    if not candidates:
        raise RuntimeError("could not locate decoder layers")
    for name, module in candidates:
        if "language_model" in name or "text" in name:
            return module
    return candidates[0][1]


class MemoryModel:
    """Attach per-layer memories to a frozen causal LM and drive write / read passes."""

    def __init__(
        self,
        model: nn.Module,
        layer_ids: Sequence[int],
        *,
        mem_dim: int,
        n_states: int,
        slots_per_state: int,
        routing: str,
        route_temperature: float = 0.1,
        decay_bias: float = 4.0,
        beta_bias: float = 0.0,
        slot_bias: float = -6.0,
        chunk_size: int = 16,
        read_mode: str = "bank",
        memory_kind: str = "delta",
        write_source: str = "residual",
    ) -> None:
        self.model = model
        self.read_mode = read_mode
        self.memory_kind = memory_kind
        if write_source not in WRITE_SOURCES:
            raise ValueError(f"write_source must be one of {WRITE_SOURCES}")
        self.write_source = write_source
        self.layers = find_decoder_layers(model)
        self.layer_ids = list(layer_ids)
        self.contexts: dict[int, LayerContext] = {}
        self.adapters = nn.ModuleDict()
        hidden_size = model.config.get_text_config().hidden_size
        device = next(model.parameters()).device
        for idx in self.layer_ids:
            attn = self.layers[idx].self_attn
            adapter = LayerMemory(
                hidden_size,
                attn.config.num_attention_heads,
                attn.head_dim,
                mem_dim=mem_dim,
                n_states=n_states,
                slots_per_state=slots_per_state,
                routing=routing,
                route_temperature=route_temperature,
                decay_bias=decay_bias,
                beta_bias=beta_bias,
                slot_bias=slot_bias,
                chunk_size=chunk_size,
                read_mode=read_mode,
                memory_kind=memory_kind,
            ).to(device=device, dtype=torch.float32)
            self.adapters[str(idx)] = adapter
            ctx = LayerContext(adapter=adapter, layer_idx=idx)
            self.contexts[idx] = ctx
            attn._memkv = ctx
            attn.q_norm.register_forward_hook(self._make_q_hook(ctx))
            # residual: raw layer input; attn_input: after input_layernorm (what the frozen attention sees)
            target = attn if write_source == "attn_input" else self.layers[idx]
            target.register_forward_pre_hook(self._make_capture_hook(ctx), with_kwargs=True)

    @staticmethod
    def _make_q_hook(ctx: LayerContext):
        def hook(module, inputs, output):
            ctx.q_unrot = output

        return hook

    @staticmethod
    def _make_capture_hook(ctx: LayerContext):
        def hook(module, args, kwargs):
            if not ctx.capture:
                return None
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            ctx.captured = hidden.detach()
            return None

        return hook

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.adapters.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())

    def set_slots(self, states: dict[int, torch.Tensor] | None) -> None:
        for idx, ctx in self.contexts.items():
            ctx.slots = None
            ctx.state = None
            if states is None:
                continue
            if self.read_mode == "query":
                ctx.state = states[idx]
            else:
                ctx.slots = ctx.adapter.read(states[idx])

    def write(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the passage through the frozen model and fold it into every layer state."""
        self.set_slots(None)
        for ctx in self.contexts.values():
            ctx.capture = True
            ctx.captured = None
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        mask = attention_mask.bool()
        inputs = []
        for idx, ctx in self.contexts.items():
            ctx.capture = False
            if ctx.captured is None:
                raise RuntimeError(f"layer {idx} did not capture hidden states")
            inputs.append(ctx.adapter.write_inputs(ctx.captured, mask))
            ctx.captured = None
        if self.memory_kind == "kvbank":
            return {idx: (inp[0], inp[1], mask) for idx, inp in zip(self.layer_ids, inputs)}
        # All wrapped layers share one geometry, so run a single recurrence over the
        # layer-concatenated batch instead of one token loop per layer.
        first = self.contexts[self.layer_ids[0]].adapter
        batch = input_ids.shape[0]
        stacked = [torch.cat(parts, dim=0) for parts in zip(*inputs)]
        state = first.zero_state(batch * len(self.layer_ids), input_ids.device)
        state = scan(state, *stacked, first.chunk_size)
        states: dict[int, torch.Tensor] = {}
        for pos, idx in enumerate(self.layer_ids):
            states[idx] = state[pos * batch : (pos + 1) * batch]
        return states

    def mem_mass(self) -> dict[int, float]:
        return {
            idx: float(ctx.stats["mem_mass"]) for idx, ctx in self.contexts.items() if "mem_mass" in ctx.stats
        }

    def state_dict(self) -> dict:
        return self.adapters.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.adapters.load_state_dict(state)


def roll_states(states: dict[int, torch.Tensor], shift: int = 1) -> dict[int, torch.Tensor]:
    """Donor control: give every row the state written from another row's passage."""
    out = {}
    for idx, s in states.items():
        if isinstance(s, tuple):
            out[idx] = tuple(torch.roll(x, shifts=shift, dims=0) for x in s)
        else:
            out[idx] = torch.roll(s, shifts=shift, dims=0)
    return out


def default_layer_ids(model: nn.Module, spec: str) -> list[int]:
    text_config = model.config.get_text_config()
    n = text_config.num_hidden_layers
    if spec == "full":
        types = getattr(text_config, "layer_types", None)
        if not types:
            raise ValueError("model has no layer_types; use explicit layer ids")
        return [i for i, t in enumerate(types) if t == "full_attention"]
    if spec == "auto":
        count = 6
        lo, hi = n // 6, (5 * n) // 6
        return [int(round(lo + (hi - lo) * i / (count - 1))) for i in range(count)]
    return [int(x) for x in spec.split(",") if x.strip()]
