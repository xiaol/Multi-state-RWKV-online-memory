from __future__ import annotations

import copy

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
)

from deltamem.core.delta import (
    DeltaMemAttention,
    HFDeltaMemConfig,
    get_delta_mem_online_state,
    load_delta_mem_online_state,
)
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention


def _module(
    *,
    hybrid_mode: str = "residual",
    hybrid_gain: float = 0.125,
    outer_ffn_gain: float = 0.03125,
    outer_ffn_layers: tuple[int, ...] = (),
    value_adapter: bool = False,
    write_address_gain: float = 0.0,
) -> DeltaMemAttention:
    torch.manual_seed(0)
    return DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=2,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
            output_init="base_slice_fixed",
            base_slice_ref_width=2,
            rwkv_ms_output_init_scale=0.02,
            rwkv_ms_hybrid_mode=hybrid_mode,
            rwkv_ms_hybrid_gain=hybrid_gain,
            rwkv_ms_outer_ffn_gain=outer_ffn_gain,
            rwkv_ms_outer_ffn_layers=outer_ffn_layers,
            rwkv_ms_write_address_gain=write_address_gain,
            rwkv_ms_write_address_value_adapter=value_adapter,
            rwkv_ms_write_address_value_adapter_rank=2,
        ),
    )


def _ple_module() -> tuple[DeltaMemAttention, Gemma4TextDecoderLayer]:
    torch.manual_seed(11)
    backbone_config = Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        hidden_size_per_layer_input=4,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        layer_types=["full_attention"],
    )
    attention = Gemma4TextAttention(backbone_config, layer_idx=0)
    module = DeltaMemAttention(
        attention,
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=2,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
            output_init="base_slice_fixed",
            base_slice_ref_width=2,
            rwkv_ms_output_init_scale=0.02,
            rwkv_ms_hybrid_mode="address_keyed_moe_ple",
            rwkv_ms_hybrid_gain=0.125,
            rwkv_ms_ple_rank=2,
            rwkv_ms_ple_gain=0.125,
            delta_heads="none",
        ),
    )
    return module, Gemma4TextDecoderLayer(backbone_config, layer_idx=0)


class _TinyDeepEmbedMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


@pytest.mark.parametrize(
    "mode",
    (
        "residual",
        "alignment_residual",
        "aligned_vector_gate",
        "addressed_affine",
        "address_bound_write",
        "addressed_route_agreement",
        "addressed_query_state_gate",
        "addressed_vector_gate",
        "vector_gate",
        "scalar_gate",
    ),
)
def test_hybrid_modes_preserve_projected_carrier_for_zero_rwkv_state(
    mode: str,
) -> None:
    module = _module(hybrid_mode=mode)
    projected = torch.randn(2, 3, module.state_read_dim)

    kwargs = (
        {"route_agreement": torch.ones(*projected.shape[:-1], 1)}
        if mode == "addressed_route_agreement"
        else {"query_state_gate": torch.full((*projected.shape[:-1], 1), 0.5)}
        if mode == "addressed_query_state_gate"
        else {}
    )
    fused = module._fuse_projected_rwkv_reads(
        projected,
        torch.zeros_like(projected),
        **kwargs,
    )

    assert torch.equal(fused, projected)


def test_addressed_value_requires_recurrent_state() -> None:
    module = _module(hybrid_mode="addressed_value", hybrid_gain=0.25)
    projected = torch.randn(2, 3, module.state_read_dim)

    fused = module._fuse_projected_rwkv_reads(
        projected,
        torch.zeros_like(projected),
    )

    assert torch.equal(fused, torch.zeros_like(projected))


@pytest.mark.parametrize(
    "mode",
    (
        "residual",
        "alignment_residual",
        "aligned_vector_gate",
        "addressed_affine",
        "address_bound_write",
        "addressed_route_agreement",
        "addressed_query_state_gate",
        "addressed_vector_gate",
        "vector_gate",
        "scalar_gate",
        "addressed_value",
        "chunk_addressed_value",
        "recurrent_value",
    ),
)
def test_hybrid_modes_are_sensitive_to_nonzero_rwkv_read(mode: str) -> None:
    module = _module(hybrid_mode=mode, hybrid_gain=0.25)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    kwargs = (
        {"route_agreement": torch.ones(*projected.shape[:-1], 1)}
        if mode == "addressed_route_agreement"
        else {"query_state_gate": torch.full((*projected.shape[:-1], 1), 0.5)}
        if mode == "addressed_query_state_gate"
        else {}
    )
    fused = module._fuse_projected_rwkv_reads(projected, recurrent, **kwargs)

    assert torch.isfinite(fused).all()
    assert not torch.equal(fused, projected)


def test_addressed_value_is_bounded_and_ignores_projected_values() -> None:
    module = _module(hybrid_mode="addressed_value", hybrid_gain=0.25)
    projected = torch.randn(2, 3, module.state_read_dim) * 1000.0
    recurrent = torch.randn_like(projected) * 1000.0

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    assert float(fused.abs().max()) <= 0.25


def test_alignment_residual_matches_locked_fusion_equation() -> None:
    module = _module(hybrid_mode="alignment_residual", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    projected_fp32 = projected.float()
    recurrent_fp32 = recurrent.float()
    carrier_rms = projected_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    recurrent_rms = recurrent_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent_fp32 / recurrent_rms.clamp_min(1e-6))
    alignment = (
        torch.nn.functional.normalize(projected_fp32, dim=-1, eps=1e-6)
        * torch.nn.functional.normalize(recurrent_fp32, dim=-1, eps=1e-6)
    ).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    expected = projected_fp32 + 0.125 * carrier_rms * alignment * direction

    assert torch.equal(fused, expected.to(dtype=projected.dtype))


def test_aligned_vector_gate_matches_locked_fusion_equation() -> None:
    module = _module(hybrid_mode="aligned_vector_gate", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    projected_fp32 = projected.float()
    recurrent_fp32 = recurrent.float()
    recurrent_rms = recurrent_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent_fp32 / recurrent_rms.clamp_min(1e-6))
    alignment = (
        torch.nn.functional.normalize(projected_fp32, dim=-1, eps=1e-6)
        * torch.nn.functional.normalize(recurrent_fp32, dim=-1, eps=1e-6)
    ).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    expected = projected_fp32 * (1.0 + 0.125 * alignment * direction)

    assert torch.equal(fused, expected.to(dtype=projected.dtype))


def test_addressed_vector_gate_matches_locked_fusion_equation() -> None:
    module = _module(hybrid_mode="addressed_vector_gate", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    projected_fp32 = projected.float()
    recurrent_fp32 = recurrent.float()
    recurrent_rms = recurrent_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent_fp32 / recurrent_rms.clamp_min(1e-6))
    expected = projected_fp32 * (1.0 + 0.125 * direction)

    assert torch.equal(fused, expected.to(dtype=projected.dtype))


def test_addressed_affine_matches_locked_fusion_equation() -> None:
    module = _module(hybrid_mode="addressed_affine", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    projected_fp32 = projected.float()
    recurrent_fp32 = recurrent.float()
    carrier_rms = projected_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    recurrent_rms = recurrent_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent_fp32 / recurrent_rms.clamp_min(1e-6))
    expected = (
        projected_fp32 * (1.0 + 0.125 * direction)
        + 0.03125 * carrier_rms * direction
    )

    assert torch.equal(fused, expected.to(dtype=projected.dtype))


def test_address_bound_write_matches_addressed_read_equation() -> None:
    module = _module(hybrid_mode="address_bound_write", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    projected_fp32 = projected.float()
    recurrent_fp32 = recurrent.float()
    carrier_rms = projected_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    recurrent_rms = recurrent_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent_fp32 / recurrent_rms.clamp_min(1e-6))
    expected = (
        projected_fp32 * (1.0 + 0.125 * direction)
        + 0.03125 * carrier_rms * direction
    )

    assert torch.equal(fused, expected.to(dtype=projected.dtype))


def test_addressed_route_agreement_matches_locked_fusion_equation() -> None:
    module = _module(hybrid_mode="addressed_route_agreement", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])
    agreement = torch.tensor([[[0.75], [0.0]]])

    fused = module._fuse_projected_rwkv_reads(
        projected,
        recurrent,
        route_agreement=agreement,
    )

    recurrent_rms = recurrent.float().square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent.float() / recurrent_rms.clamp_min(1e-6))
    expected = projected.float() * (1.0 + 0.125 * agreement * direction)

    assert torch.equal(fused, expected.to(dtype=projected.dtype))
    assert torch.equal(fused[:, 1], projected[:, 1])


def test_addressed_route_agreement_requires_route_agreement() -> None:
    module = _module(hybrid_mode="addressed_route_agreement")
    projected = torch.randn(1, 2, module.state_read_dim)

    with pytest.raises(ValueError, match="requires route agreement"):
        module._fuse_projected_rwkv_reads(projected, torch.randn_like(projected))


def test_addressed_query_state_gate_matches_locked_fusion_equation() -> None:
    module = _module(hybrid_mode="addressed_query_state_gate", hybrid_gain=0.125)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])
    gate = torch.tensor([[[0.75], [0.0]]])

    fused = module._fuse_projected_rwkv_reads(
        projected,
        recurrent,
        query_state_gate=gate,
    )

    recurrent_rms = recurrent.float().square().mean(dim=-1, keepdim=True).sqrt()
    direction = torch.tanh(recurrent.float() / recurrent_rms.clamp_min(1e-6))
    expected = projected.float() * (1.0 + 0.125 * gate * direction)

    assert torch.equal(fused, expected.to(dtype=projected.dtype))
    assert torch.equal(fused[:, 1], projected[:, 1])


def test_addressed_query_state_gate_requires_gate() -> None:
    module = _module(hybrid_mode="addressed_query_state_gate")
    projected = torch.randn(1, 2, module.state_read_dim)

    with pytest.raises(ValueError, match="requires a supervised query-state gate"):
        module._fuse_projected_rwkv_reads(projected, torch.randn_like(projected))


@pytest.mark.parametrize(
    "hybrid_mode",
    ("address_bound_write", "address_bound_moe_controller"),
)
def test_address_bound_modes_broadcast_projected_slot_to_recurrent_tokens(
    monkeypatch,
    hybrid_mode: str,
) -> None:
    module = _module(hybrid_mode=hybrid_mode)
    module.set_write_enabled(True)
    hidden = torch.randn(2, 3, module.hidden_size)
    token_mask = torch.tensor([[True, True, True], [True, True, False]])
    projected_routes = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]],
    )
    captured: dict[str, torch.Tensor | None] = {}

    def write_slots(*args) -> None:
        module.last_write_routes = projected_routes

    def backend_scan(*args, **kwargs):
        captured["routes"] = kwargs["rwkv_write_route_seq"]
        return args[0], torch.zeros(
            hidden.size(0),
            hidden.size(1),
            module.state_read_dim,
        )

    monkeypatch.setattr(module, "_write_projected_kv_slots", write_slots)
    monkeypatch.setattr(module, "_memory_backend_scan", backend_scan)

    module._projected_rwkv_hybrid_step(
        torch.zeros(
            hidden.size(0),
            module.num_state_heads,
            module.rwkv_ms_num_states,
            module.rank,
            module.rank,
        ),
        hidden,
        token_mask,
    )

    expected = projected_routes.expand(-1, hidden.size(1), -1) * token_mask.unsqueeze(-1)
    assert torch.equal(captured["routes"], expected)


def test_address_bound_moe_zero_state_is_exactly_projected_only() -> None:
    module = _module(hybrid_mode="address_bound_moe_controller")
    projected = torch.randn(
        2, 3, module.state_read_dim, requires_grad=True
    )
    recurrent = torch.zeros_like(projected, requires_grad=True)
    global_recurrent = torch.zeros_like(projected, requires_grad=True)
    hidden = torch.randn(2, 3, module.hidden_size, requires_grad=True)

    fused = module._fuse_projected_rwkv_reads(
        projected,
        recurrent,
        global_recurrent_reads=global_recurrent,
        hidden_states=hidden,
    )
    fused.sum().backward()

    assert hasattr(module, "rwkv_moe_bias")
    assert torch.equal(fused, projected)
    for tensor in (projected, recurrent, global_recurrent, hidden):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_addressed_moe_outer_ffn_zero_state_is_exactly_projected_only() -> None:
    module = _module(hybrid_mode="addressed_moe_outer_ffn")
    projected = torch.randn(2, 3, module.state_read_dim)
    recurrent = torch.zeros_like(projected)
    hidden = torch.randn(2, 3, module.hidden_size)

    fused = module._fuse_projected_rwkv_reads(
        projected,
        recurrent,
        global_recurrent_reads=recurrent,
        hidden_states=hidden,
    )

    assert torch.equal(fused, projected)


def test_addressed_moe_outer_ffn_is_bounded_zero_preserving_and_trainable() -> None:
    module = _module(
        hybrid_mode="addressed_moe_outer_ffn",
        hybrid_gain=0.03125,
        outer_ffn_gain=1.0 / 2048.0,
    )
    hidden = torch.randn(2, 3, module.hidden_size)
    zero_control = torch.zeros(2, 3, module.state_read_dim)

    zero_residual = module._outer_ffn_residual(hidden, zero_control, None)
    assert torch.equal(zero_residual, torch.zeros_like(hidden))

    control = torch.randn_like(zero_control)
    residual = module._outer_ffn_residual(hidden, control, None)
    residual.square().mean().backward()

    assert torch.isfinite(residual).all()
    assert float(residual.abs().max()) <= 1.0 / 2048.0
    assert module.rwkv_outer_ffn_down_weight.grad is not None
    assert module.rwkv_outer_ffn_gate_weight.grad is not None
    assert module.rwkv_outer_ffn_up_weight.grad is not None
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in (
            module.rwkv_outer_ffn_down_weight,
            module.rwkv_outer_ffn_gate_weight,
            module.rwkv_outer_ffn_up_weight,
        )
    )


def test_addressed_moe_outer_ffn_bf16_activations_keep_fp32_gradients() -> None:
    module = _module(
        hybrid_mode="addressed_moe_outer_ffn",
        hybrid_gain=0.03125,
        outer_ffn_gain=1.0 / 2048.0,
    )
    hidden = torch.randn(2, 3, module.hidden_size, dtype=torch.bfloat16)
    control = torch.randn(2, 3, module.state_read_dim, dtype=torch.bfloat16)

    residual = module._outer_ffn_residual(hidden, control, None)
    residual.float().square().mean().backward()

    assert residual.dtype == torch.bfloat16
    for parameter in (
        module.rwkv_outer_ffn_down_weight,
        module.rwkv_outer_ffn_gate_weight,
        module.rwkv_outer_ffn_up_weight,
    ):
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.float().norm().item() > 0.0


def test_addressed_moe_outer_ffn_hook_adds_and_consumes_pending_residual() -> None:
    module = _module(hybrid_mode="addressed_moe_outer_ffn")
    layernorm = torch.nn.Identity()
    module.bind_post_feedforward_layernorm(layernorm)
    source = torch.randn(2, 3, module.hidden_size)
    residual = torch.full_like(source, 0.125)
    module._pending_outer_ffn_delta = residual

    output = layernorm(source)

    assert torch.equal(output, source + residual)
    assert module._pending_outer_ffn_delta is None
    module.remove_post_feedforward_layernorm_hook()


def test_addressed_moe_outer_ffn_can_be_sparse_across_depth() -> None:
    inactive = _module(
        hybrid_mode="addressed_moe_outer_ffn",
        outer_ffn_layers=(1, 3),
    )
    layernorm = torch.nn.Identity()
    inactive.bind_post_feedforward_layernorm(layernorm)
    source = torch.randn(2, 3, inactive.hidden_size)

    assert inactive.layer_idx == 0
    assert inactive.rwkv_ms_outer_ffn_enabled is False
    assert not hasattr(inactive, "rwkv_outer_ffn_up_weight")
    assert torch.equal(layernorm(source), source)
    inactive.remove_post_feedforward_layernorm_hook()


def test_deepembed_ffn_mode_is_zero_preserving_and_multiplicative() -> None:
    module = _module(
        hybrid_mode="addressed_moe_deepembed_ffn",
        hybrid_gain=0.03125,
        outer_ffn_gain=1.0 / 2048.0,
    )
    mlp = _TinyDeepEmbedMLP(module.hidden_size, 7)
    module.bind_deepembed_ffn(mlp)
    hidden = torch.randn(2, 3, module.hidden_size)
    control = torch.randn(2, 3, module.state_read_dim)
    module._pending_deepembed_ffn_control = torch.zeros_like(control)
    baseline = mlp(hidden)

    module._pending_deepembed_ffn_control = torch.zeros_like(control)
    zero = mlp(hidden)
    assert torch.equal(zero, baseline)

    module._pending_deepembed_ffn_control = control
    changed = mlp(hidden)
    assert torch.isfinite(changed).all()
    assert not torch.equal(changed, baseline)
    module.remove_deepembed_ffn_hooks()
    assert module._deepembed_ffn_pre_hook_handle is None
    assert module._deepembed_ffn_down_pre_hook_handle is None


def test_deepembed_ffn_mode_keeps_bf16_gradients_finite() -> None:
    module = _module(
        hybrid_mode="addressed_moe_deepembed_ffn",
        hybrid_gain=0.03125,
        outer_ffn_gain=1.0 / 2048.0,
    )
    mlp = _TinyDeepEmbedMLP(module.hidden_size, 7).to(dtype=torch.bfloat16)
    module.bind_deepembed_ffn(mlp)
    hidden = torch.randn(2, 3, module.hidden_size, dtype=torch.bfloat16)
    control = torch.randn(2, 3, module.state_read_dim, dtype=torch.bfloat16)
    module._pending_deepembed_ffn_control = control
    output = mlp(hidden)
    output.float().square().mean().backward()

    for parameter in (
        module.rwkv_outer_ffn_down_weight,
        module.rwkv_outer_ffn_gate_weight,
        module.rwkv_outer_ffn_up_weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
    module.remove_deepembed_ffn_hooks()


def test_rwkv_ple_projection_is_exactly_zero_at_identity_initialization() -> None:
    module, _ = _ple_module()
    control = torch.randn(2, 3, module.state_read_dim)

    delta = module._rwkv_ple_memory_delta(control)
    zero_delta = module._rwkv_ple_memory_delta(torch.zeros_like(control))

    assert delta.shape == (2, 3, module.rwkv_ple_dim)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(zero_delta, torch.zeros_like(zero_delta))


def test_rwkv_ple_projection_has_live_low_rank_gradients() -> None:
    module, _ = _ple_module()
    control = torch.randn(2, 3, module.state_read_dim)

    delta = module._rwkv_ple_memory_delta(control)
    delta.sum().backward()

    for parameter in (module.rwkv_ple_down_weight, module.rwkv_ple_up_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().gt(0).any()


def test_rwkv_ple_hook_injects_before_native_per_layer_projection() -> None:
    module, layer = _ple_module()
    module.bind_ple_input(layer.per_layer_projection)
    native = torch.randn(2, 3, module.rwkv_ple_dim)
    control = torch.randn(2, 3, module.state_read_dim)
    with torch.no_grad():
        module.rwkv_ple_up_weight.add_(0.05)
    delta = module._rwkv_ple_memory_delta(control)
    module.remove_ple_input_hook()
    expected = layer.per_layer_projection(native + delta)

    module.bind_ple_input(layer.per_layer_projection)
    module._pending_ple_memory = control
    actual = layer.per_layer_projection(native)

    torch.testing.assert_close(actual, expected)
    assert module._pending_ple_memory is None
    module.remove_ple_input_hook()


def test_rwkv_ple_multiplicative_fusion_matches_deepembed_shape() -> None:
    module, layer = _ple_module()
    module.rwkv_ms_ple_fusion = "multiplicative"
    module.bind_ple_input(layer.per_layer_projection)
    native = torch.randn(2, 3, module.rwkv_ple_dim)
    control = torch.randn(2, 3, module.state_read_dim)
    with torch.no_grad():
        module.rwkv_ple_up_weight.add_(0.05)
    delta = module._rwkv_ple_memory_delta(control)
    module.remove_ple_input_hook()
    expected = layer.per_layer_projection(native * (1.0 + delta))

    module.bind_ple_input(layer.per_layer_projection)
    module._pending_ple_memory = control
    actual = layer.per_layer_projection(native)

    torch.testing.assert_close(actual, expected)
    assert module._pending_ple_memory is None
    module.remove_ple_input_hook()


def test_rwkv_ple_attach_preserves_frozen_gemma_output_at_initialization() -> None:
    torch.manual_seed(12)
    backbone_config = Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        hidden_size_per_layer_input=4,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        layer_types=["full_attention", "full_attention"],
    )
    baseline = Gemma4TextModel(backbone_config)
    candidate = Gemma4TextModel(backbone_config)
    candidate.load_state_dict(baseline.state_dict())
    from deltamem.core.delta import attach_delta_mem

    attach_delta_mem(
        candidate,
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=2,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
            output_init="base_slice_fixed",
            base_slice_ref_width=2,
            rwkv_ms_output_init_scale=0.02,
            rwkv_ms_hybrid_mode="address_keyed_moe_ple",
            rwkv_ms_hybrid_gain=0.125,
            rwkv_ms_ple_rank=2,
            rwkv_ms_ple_gain=0.125,
            delta_heads="none",
            target_modules=("self_attn",),
            target_layers=(0, 1),
        ),
    )
    input_ids = torch.tensor([[2, 7, 11]])
    expected = baseline(input_ids=input_ids).last_hidden_state
    actual = candidate(input_ids=input_ids).last_hidden_state

    assert torch.equal(actual, expected)


def test_addressed_route_agreement_step_abstains_at_chance_overlap(
    monkeypatch,
) -> None:
    module = _module(hybrid_mode="addressed_route_agreement", hybrid_gain=0.125)
    module.set_write_enabled(False)
    hidden = torch.randn(1, 2, module.hidden_size)
    token_mask = torch.ones(1, 2, dtype=torch.bool)
    state = torch.randn(
        1,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
    )
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    projected_routes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    def projected_read(_: torch.Tensor) -> torch.Tensor:
        module.last_read_routes = projected_routes
        return projected

    recurrent = torch.full_like(projected, 0.5)
    recurrent_routes = torch.full_like(projected_routes, 0.5)
    monkeypatch.setattr(module, "_projected_kv_slot_token_reads", projected_read)
    monkeypatch.setattr(
        module,
        "_rwkv_ms_route_agreement_token_state_reads",
        lambda *args: (recurrent, recurrent_routes),
    )

    _, reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert torch.equal(reads, projected)
    assert torch.equal(module.last_read_routes, projected_routes)


def test_addressed_route_agreement_step_uses_matching_routes(monkeypatch) -> None:
    module = _module(hybrid_mode="addressed_route_agreement", hybrid_gain=0.125)
    module.set_write_enabled(False)
    hidden = torch.randn(1, 2, module.hidden_size)
    token_mask = torch.ones(1, 2, dtype=torch.bool)
    state = torch.randn(
        1,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
    )
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    projected_routes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    def projected_read(_: torch.Tensor) -> torch.Tensor:
        module.last_read_routes = projected_routes
        return projected

    monkeypatch.setattr(module, "_projected_kv_slot_token_reads", projected_read)
    monkeypatch.setattr(
        module,
        "_rwkv_ms_route_agreement_token_state_reads",
        lambda *args: (torch.full_like(projected, 0.5), projected_routes),
    )

    _, reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert not torch.equal(reads, projected)
    assert torch.equal(module.last_read_routes, projected_routes)


@pytest.mark.parametrize("mode", ("addressed_affine", "addressed_vector_gate"))
def test_addressed_gate_modes_use_projected_routes(monkeypatch, mode: str) -> None:
    module = _module(hybrid_mode=mode, hybrid_gain=0.125)
    module.set_write_enabled(False)
    hidden = torch.randn(1, 3, module.hidden_size)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    state = torch.randn(
        1,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
    )
    module.projected_kv_keys = torch.randn(
        1,
        module.rwkv_ms_num_states,
        module.projected_kv_key_dim,
    )
    module.projected_kv_values = torch.randn(
        1,
        module.rwkv_ms_num_states,
        module.state_read_dim,
    )
    module.projected_kv_occupied = torch.ones(
        1,
        module.rwkv_ms_num_states,
        dtype=torch.bool,
    )
    module.projected_kv_surprise = torch.ones(1, module.rwkv_ms_num_states)
    captured: dict[str, torch.Tensor] = {}
    original = module._rwkv_ms_addressed_token_state_reads

    def addressed_read(
        recurrent_state: torch.Tensor,
        memory_source: torch.Tensor,
        projected_routes: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        captured["routes"] = projected_routes.detach().clone()
        return original(recurrent_state, memory_source, projected_routes, mask)

    monkeypatch.setattr(module, "_rwkv_ms_addressed_token_state_reads", addressed_read)
    monkeypatch.setattr(
        module,
        "_memory_backend_token_reads",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("independent RWKV routing must not run")
        ),
    )

    _, reads, _, _ = module._projected_rwkv_hybrid_step(state, hidden, token_mask)

    assert torch.equal(module.last_read_routes, captured["routes"])
    assert torch.count_nonzero(reads).item() > 0


def test_addressed_value_step_uses_projected_routes_not_projected_values() -> None:
    module = _module(hybrid_mode="addressed_value", hybrid_gain=0.5)
    module.set_write_enabled(False)
    hidden = torch.randn(1, 3, module.hidden_size)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    state = torch.randn(
        1,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
    )
    module.projected_kv_keys = torch.randn(
        1,
        module.rwkv_ms_num_states,
        module.projected_kv_key_dim,
    )
    module.projected_kv_occupied = torch.ones(
        1,
        module.rwkv_ms_num_states,
        dtype=torch.bool,
    )
    module.projected_kv_surprise = torch.ones(1, module.rwkv_ms_num_states)
    module.projected_kv_values = torch.randn(
        1,
        module.rwkv_ms_num_states,
        module.state_read_dim,
    )

    _, first_reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )
    first_routes = module.last_read_routes.detach().clone()
    module.projected_kv_values = torch.randn_like(module.projected_kv_values) * 1000.0
    _, second_reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert torch.equal(module.last_read_routes, first_routes)
    assert torch.equal(second_reads, first_reads)
    assert torch.count_nonzero(first_reads).item() > 0


def test_recurrent_value_step_ignores_complete_projected_bundle() -> None:
    module = _module(hybrid_mode="recurrent_value", hybrid_gain=0.5)
    module.set_write_enabled(False)
    hidden = torch.randn(1, 3, module.hidden_size)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    state = torch.randn(
        1,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
    )
    module.projected_kv_keys = torch.randn(
        1,
        module.rwkv_ms_num_states,
        module.projected_kv_key_dim,
    )
    module.projected_kv_values = torch.randn(
        1,
        module.rwkv_ms_num_states,
        module.state_read_dim,
    )
    module.projected_kv_occupied = torch.ones(
        1,
        module.rwkv_ms_num_states,
        dtype=torch.bool,
    )
    module.projected_kv_surprise = torch.ones(1, module.rwkv_ms_num_states)

    _, first_reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )
    first_routes = module.last_read_routes.detach().clone()
    module.projected_kv_keys = torch.randn_like(module.projected_kv_keys) * 1000.0
    module.projected_kv_values = torch.randn_like(module.projected_kv_values) * 1000.0
    module.projected_kv_occupied.zero_()
    module.projected_kv_surprise = torch.randn_like(module.projected_kv_surprise)
    _, second_reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert torch.equal(second_reads, first_reads)
    assert torch.equal(module.last_read_routes, first_routes)
    assert torch.count_nonzero(first_reads).item() > 0


def test_rwkv_read_temperature_sharpens_internal_routes() -> None:
    module = _module(hybrid_mode="recurrent_value", hybrid_gain=0.5)
    module.rwkv_ms_mask_empty_slots = True
    slot_reads = torch.tensor([[[[1.0, 0.0], [0.8, 0.6]]]])
    query = torch.tensor([[[1.0, 0.0]]])
    valid = torch.ones(1, dtype=torch.bool)

    module.rwkv_ms_read_temperature = 1.0
    baseline = module._rwkv_ms_read_routes(slot_reads, query, valid)
    module.rwkv_ms_read_temperature = 16.0
    sharpened = module._rwkv_ms_read_routes(slot_reads, query, valid)

    assert sharpened[0, 0, 0] > baseline[0, 0, 0]
    assert torch.equal(sharpened.argmax(dim=-1), baseline.argmax(dim=-1))


def test_rwkv_read_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="rwkv_ms_read_temperature"):
        HFDeltaMemConfig(rwkv_ms_read_temperature=0.0)


def test_top_one_rwkv_route_is_hard_forward_soft_backward() -> None:
    module = _module(hybrid_mode="recurrent_value", hybrid_gain=0.5)
    module.rwkv_ms_mask_empty_slots = True
    module.rwkv_ms_read_top_k = 1
    slot_reads = torch.tensor(
        [[[[1.0, 0.0], [0.8, 0.6]]]],
        requires_grad=True,
    )
    query = torch.tensor([[[1.0, 0.0]]])

    routes = module._rwkv_ms_read_routes(
        slot_reads,
        query,
        torch.ones(1, dtype=torch.bool),
    )
    weighted = (routes * torch.tensor([[[1.0, -1.0]]])).sum()
    weighted.backward()

    assert torch.equal(routes.detach(), torch.tensor([[[1.0, 0.0]]]))
    assert slot_reads.grad is not None
    assert torch.count_nonzero(slot_reads.grad).item() > 0


def test_detached_rwkv_scores_preserve_routes_without_router_gradient() -> None:
    module = _module(hybrid_mode="recurrent_value", hybrid_gain=0.5)
    module.rwkv_ms_mask_empty_slots = True
    slot_reads = torch.tensor(
        [[[[1.0, 0.0], [0.8, 0.6]]]],
        requires_grad=True,
    )
    query = torch.tensor([[[1.0, 0.0]]])
    valid = torch.ones(1, dtype=torch.bool)

    baseline = module._rwkv_ms_read_routes(slot_reads, query, valid)
    module.rwkv_ms_detach_read_scores = True
    detached = module._rwkv_ms_read_routes(slot_reads, query, valid)

    assert torch.equal(detached, baseline)
    assert detached.requires_grad is False


def test_hybrid_write_populates_projected_and_recurrent_state() -> None:
    module = _module()
    hidden = torch.randn(1, 4, module.hidden_size)
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    state = module._ensure_state(1, hidden.device, hidden.dtype)

    next_state, reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert module.projected_kv_keys is not None
    assert module.projected_kv_values is not None
    assert module.projected_kv_occupied is not None
    assert module.projected_kv_occupied.any()
    assert torch.count_nonzero(next_state).item() > 0
    assert module.rwkv_ms_positions is not None
    assert torch.equal(module.rwkv_ms_positions, torch.tensor([4]))
    assert torch.count_nonzero(reads).item() == 0


def test_chunk_addressed_write_aligns_keys_with_rwkv_slots() -> None:
    module = _module(hybrid_mode="chunk_addressed_value")
    hidden = torch.randn(1, 5, module.hidden_size)
    token_mask = torch.ones(1, 5, dtype=torch.bool)
    state = module._ensure_state(1, hidden.device, hidden.dtype)
    module.rwkv_ms_positions = torch.tensor([1])
    expected_hidden = hidden[:, [4, 2]]
    expected_keys, _ = module._projected_kv_project_hidden(expected_hidden)

    next_state, reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert module.projected_kv_keys is not None
    assert torch.equal(module.projected_kv_keys, expected_keys)
    assert module.projected_kv_values is not None
    assert torch.count_nonzero(module.projected_kv_values).item() == 0
    assert module.projected_kv_occupied is not None
    assert torch.equal(module.projected_kv_occupied, torch.ones(1, 2, dtype=torch.bool))
    assert module.last_write_routes is not None
    assert torch.equal(
        module.last_write_routes.argmax(dim=-1),
        torch.tensor([[0, 1, 1, 0, 0]]),
    )
    assert torch.count_nonzero(next_state).item() > 0
    assert module.rwkv_ms_positions is not None
    assert torch.equal(module.rwkv_ms_positions, torch.tensor([6]))
    assert torch.count_nonzero(reads).item() == 0


def test_hybrid_online_state_round_trip_contains_both_carriers() -> None:
    module = _module()
    model = torch.nn.Module()
    model.add_module("attn", module)
    hidden = torch.randn(1, 3, module.hidden_size)
    state = module._ensure_state(1, hidden.device, hidden.dtype)
    next_state, _, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        torch.ones(1, 3, dtype=torch.bool),
    )
    module.delta_state = next_state
    saved = get_delta_mem_online_state(model)

    module.reset_state()
    load_delta_mem_online_state(model, saved)

    assert "attn" in saved
    assert "attn.__projected_kv_keys" in saved
    assert "attn.__projected_kv_values" in saved
    assert module.delta_state is not None
    assert torch.equal(module.delta_state.cpu(), saved["attn"])
    assert module.projected_kv_values is not None
    assert torch.equal(
        module.projected_kv_values.cpu(),
        saved["attn.__projected_kv_values"],
    )


@pytest.mark.parametrize("semantics_version", (1, 2))
def test_vectorized_recurrent_read_matches_token_loop(semantics_version: int) -> None:
    module = _module()
    module.rwkv_ms_semantics_version = semantics_version
    module.rwkv_ms_mask_empty_slots = True
    module.rwkv_ms_read_top_k = 1
    batch_size, seq_len = 2, 5
    source = torch.randn(
        batch_size,
        seq_len,
        module.state_read_dim,
        requires_grad=True,
    )
    state = torch.randn(
        batch_size,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
        requires_grad=True,
    )
    state = state.clone()
    state[:, :, -1] = 0.0
    token_mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, False]]
    )

    features = module.hrm_rwkv7_core.project(
        source,
        previous_x=None,
        token_mask=token_mask,
        advance_within_sequence=False,
    )
    r_seq = module._rwkv_ms_project_heads(features.r).float()
    current_state = state.float()
    occupied = current_state.ne(0).any(dim=(-1, -2))
    reference_reads = []
    reference_routes = []
    for token_index in range(seq_len):
        r_t = r_seq[:, token_index]
        routes = module._rwkv_ms_read_routes(
            torch.einsum("bhsij,bhj->bhsi", current_state, r_t),
            r_t,
            token_mask[:, token_index],
            occupied_slots=occupied,
        )
        slot_reads = torch.einsum("bhsij,bhj->bhsi", current_state, r_t)
        reference_reads.append(
            torch.einsum("bhs,bhsi->bhi", routes, slot_reads).reshape(
                batch_size,
                module.state_read_dim,
            )
        )
        reference_routes.append(routes.mean(dim=1))
    reference = module.hrm_rwkv7_core.readout(
        torch.stack(reference_reads, dim=1).to(features.g.dtype),
        features.g,
    )
    targets = (state, source, module.hrm_rwkv7_core.output.weight)
    reference_grads = torch.autograd.grad(
        reference.float().square().mean(),
        targets,
        retain_graph=True,
    )

    vectorized = module._rwkv_ms_token_state_reads(state, source, token_mask)
    vectorized_routes = module.last_read_routes
    vectorized_grads = torch.autograd.grad(
        vectorized.float().square().mean(),
        targets,
    )

    assert torch.equal(vectorized, reference)
    assert torch.allclose(
        vectorized_routes,
        torch.stack(reference_routes, dim=1),
        atol=1e-7,
        rtol=0.0,
    )
    for vectorized_grad, reference_grad in zip(
        vectorized_grads,
        reference_grads,
    ):
        assert torch.allclose(
            vectorized_grad,
            reference_grad,
            atol=1e-8,
            rtol=1e-6,
        ), {
            "maximum_absolute_delta": float(
                (vectorized_grad - reference_grad).abs().max()
            ),
            "reference_maximum": float(reference_grad.abs().max()),
        }


def test_hybrid_write_only_scan_matches_full_rwkv_state_exactly() -> None:
    torch.manual_seed(17)
    full_module = _module()
    write_only_module = copy.deepcopy(full_module)
    batch_size, seq_len = 2, 7
    source = torch.randn(batch_size, seq_len, full_module.state_read_dim)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1, 1))
    decay = torch.sigmoid(torch.randn(batch_size, seq_len, 1, 1))
    token_mask = torch.tensor(
        [[True, True, False, True, True, True, False], [True, False, True, True, True, False, True]]
    )
    state = torch.randn(
        batch_size,
        full_module.num_state_heads,
        full_module.rwkv_ms_num_states,
        full_module.rank,
        full_module.rank,
    )
    positions = torch.tensor([0, 1022])
    for module in (full_module, write_only_module):
        module.rwkv_ms_positions = positions.clone()
        module.rwkv_ms_previous_source = None

    full_state, _ = full_module._rwkv_ms_scan(
        state,
        source,
        beta,
        decay,
        token_mask,
        update_positions=True,
        write_only=False,
    )
    write_only_state, _ = write_only_module._rwkv_ms_scan(
        state,
        source,
        beta,
        decay,
        token_mask,
        update_positions=True,
        write_only=True,
    )

    assert torch.equal(write_only_state, full_state)
    assert torch.equal(
        write_only_module.rwkv_ms_positions,
        full_module.rwkv_ms_positions,
    )
    assert torch.equal(
        write_only_module.rwkv_ms_previous_source,
        full_module.rwkv_ms_previous_source,
    )
    assert torch.equal(
        write_only_module.last_write_routes,
        full_module.last_write_routes,
    )


def test_hybrid_configuration_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="rwkv_ms_hybrid_gain"):
        HFDeltaMemConfig(rwkv_ms_hybrid_gain=1.01)
    with pytest.raises(ValueError, match="hybrid mode"):
        HFDeltaMemConfig(rwkv_ms_hybrid_mode="unknown")
    with pytest.raises(ValueError, match="memory_write_granularity='token'"):
        HFDeltaMemConfig(
            memory_backend="rwkv_ms",
            memory_readout_mode="projected_kv_rwkv_hybrid",
            memory_write_granularity="message_mean",
        )
    with pytest.raises(ValueError, match="address_keyed_moe_ple requires recurrent"):
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=2,
            rwkv_ms_hybrid_mode="address_keyed_moe_ple",
            rwkv_ms_write_mode="last_token_overwrite",
        )


def test_address_value_adapter_is_zero_effect_at_initialization() -> None:
    torch.manual_seed(19)
    legacy = _module(
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        write_address_gain=0.25,
    )
    adapted = _module(
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        value_adapter=True,
        write_address_gain=0.25,
    )
    features = tuple(torch.randn(1, 3, adapted.state_read_dim) for _ in range(4))
    address = torch.randn(1, 3, adapted.state_read_dim)
    mask = torch.ones(1, 3, dtype=torch.bool)
    legacy_result = legacy._rwkv_ms_address_conditioned_write_features(
        *features, address, mask
    )
    adapted_result = adapted._rwkv_ms_address_conditioned_write_features(
        *features, address, mask
    )
    for expected, actual in zip(legacy_result, adapted_result):
        assert torch.equal(expected, actual)


def test_address_value_adapter_only_changes_value_direction() -> None:
    module = _module(
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        value_adapter=True,
        write_address_gain=0.25,
    )
    with torch.no_grad():
        module.rwkv_ms_write_address_value_up.fill_(0.25)
    features = tuple(torch.randn(1, 2, module.state_read_dim) for _ in range(4))
    address = torch.randn(1, 2, module.state_read_dim)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = module._rwkv_ms_address_conditioned_write_features(
        *features, address, mask
    )
    legacy = _module(
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        write_address_gain=0.25,
    )
    legacy_result = legacy._rwkv_ms_address_conditioned_write_features(
        *features, address, mask
    )
    assert torch.equal(result[0], legacy_result[0])
    assert not torch.equal(result[1], legacy_result[1])
    assert torch.equal(result[2], legacy_result[2])
    assert torch.equal(result[3], legacy_result[3])


def test_address_value_adapter_zero_address_preserves_all_features() -> None:
    module = _module(
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        value_adapter=True,
        write_address_gain=0.25,
    )
    features = tuple(torch.randn(1, 2, module.state_read_dim) for _ in range(4))
    address = torch.zeros(1, 2, module.state_read_dim)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = module._rwkv_ms_address_conditioned_write_features(
        *features, address, mask
    )
    for expected, actual in zip(features, result):
        assert torch.equal(expected, actual)


def test_address_value_adapter_has_finite_value_gradients() -> None:
    module = _module(
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        value_adapter=True,
        write_address_gain=0.25,
    )
    features = tuple(torch.randn(1, 2, module.state_read_dim) for _ in range(4))
    address = torch.randn(1, 2, module.state_read_dim)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = module._rwkv_ms_address_conditioned_write_features(
        *features, address, mask
    )
    result[1].sum().backward()
    for name in (
        "rwkv_ms_write_address_value_down",
        "rwkv_ms_write_address_value_up",
    ):
        gradient = getattr(module, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
