from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from deltamem.core.delta import DeltaMemAttention, HFDeltaMemConfig
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5
    as causal_train_v5,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_execution,
)
from experiments.rethinking_rwkv_ms_gemma import learned_rwkv_write
from experiments.rethinking_rwkv_ms_gemma import rwkv_query_state_identity
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as learned_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_query_state_identity_causal_train as identity_train,
)


def _module(
    *,
    write_address_gain: float,
    projected_key_dim: int = 2,
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
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
            projected_kv_key_dim=projected_key_dim,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
            output_init="base_slice_fixed",
            base_slice_ref_width=2,
            rwkv_ms_output_init_scale=0.02,
            rwkv_ms_hybrid_mode="address_keyed_moe_deepembed_ffn",
            rwkv_ms_hybrid_gain=1.0 / 64.0,
            rwkv_ms_write_address_gain=write_address_gain,
            rwkv_ms_outer_ffn_gain=1.0 / 128.0,
            rwkv_ms_outer_ffn_layers=(0,),
        ),
    )


def test_address_keyed_config_locks_gain_and_feature_dimensions() -> None:
    config = _module(write_address_gain=0.25).rwkv_ms_write_address_gain
    assert config == 0.25

    with pytest.raises(ValueError, match="rwkv_ms_write_address_gain"):
        HFDeltaMemConfig(rwkv_ms_write_address_gain=1.01)
    with pytest.raises(ValueError, match="only active"):
        HFDeltaMemConfig(rwkv_ms_write_address_gain=0.25)
    with pytest.raises(ValueError, match="projected_kv_key_dim"):
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=3,
            rwkv_ms_hybrid_mode="address_keyed_moe_deepembed_ffn",
        )
    with pytest.raises(ValueError, match="recurrent RWKV writes"):
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=2,
            rwkv_ms_write_mode="last_token_overwrite",
            rwkv_ms_hybrid_mode="address_keyed_moe_deepembed_ffn",
        )


def test_address_conditioner_has_exact_identity_controls() -> None:
    module = _module(write_address_gain=0.25)
    shape = (2, 3, module.state_read_dim)
    features = tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(4))
    token_mask = torch.tensor([[True, True, False], [True, False, True]])

    zero_outputs = module._rwkv_ms_address_conditioned_write_features(
        *features,
        torch.zeros(shape, dtype=torch.bfloat16),
        token_mask,
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(zero_outputs, features))

    zero_gain = copy.deepcopy(module)
    zero_gain.rwkv_ms_write_address_gain = 0.0
    address = torch.randn(shape, dtype=torch.bfloat16)
    gain_outputs = zero_gain._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        token_mask,
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(gain_outputs, features))


def test_address_conditioner_changes_only_valid_keyed_tokens() -> None:
    module = _module(write_address_gain=0.25)
    shape = (2, 3, module.state_read_dim)
    features = tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(4))
    address = torch.randn(shape, dtype=torch.bfloat16)
    token_mask = torch.tensor([[True, True, False], [True, False, True]])

    outputs = module._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        token_mask,
    )

    assert all(not torch.equal(actual[token_mask], expected[token_mask]) for actual, expected in zip(outputs, features))
    assert all(torch.equal(actual[~token_mask], expected[~token_mask]) for actual, expected in zip(outputs, features))
    assert all(torch.isfinite(actual).all() for actual in outputs)


def test_address_conditioner_zero_address_backward_is_finite() -> None:
    module = _module(write_address_gain=0.25)
    shape = (2, 3, module.state_read_dim)
    features = tuple(
        torch.randn(shape, dtype=torch.float32, requires_grad=True) for _ in range(4)
    )
    address = torch.zeros(shape, dtype=torch.float32, requires_grad=True)
    token_mask = torch.tensor([[True, True, False], [True, False, True]])

    outputs = module._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        token_mask,
    )
    sum(output.sum() for output in outputs).backward()

    assert all(torch.equal(actual, expected) for actual, expected in zip(outputs, features))
    assert all(feature.grad is not None for feature in features)
    assert all(torch.isfinite(feature.grad).all() for feature in features)
    assert address.grad is not None
    assert torch.isfinite(address.grad).all()


def test_address_conditioner_nonzero_carriers_have_finite_feature_gradients() -> None:
    module = _module(write_address_gain=0.25)
    shape = (2, 3, module.state_read_dim)
    features = tuple(
        torch.randn(shape, dtype=torch.float32, requires_grad=True) for _ in range(4)
    )
    address = torch.randn(shape, dtype=torch.float32, requires_grad=True)

    outputs = module._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        None,
    )
    sum(output.square().mean() for output in outputs).backward()

    assert all(feature.grad is not None for feature in features)
    assert all(torch.isfinite(feature.grad).all() for feature in features)
    assert address.grad is not None
    assert torch.isfinite(address.grad).all()


def test_learned_write_conditioner_starts_as_exact_noop_and_trains() -> None:
    module = _module(write_address_gain=0.25)
    model = torch.nn.Module()
    model.attention = module
    audit = learned_rwkv_write.install(model, rank=2)
    assert audit["parameter_tensors"] == 8
    shape = (2, 3, module.state_read_dim)
    features = tuple(torch.randn(shape) for _ in range(4))
    address = torch.randn(shape)
    initial = module._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        None,
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(initial, features))
    loss = sum(output.square().mean() for output in initial)
    loss.backward()
    assert all(
        getattr(module, name[1:]).grad is not None
        for name in learned_rwkv_write.parameter_suffixes()
    )
    with torch.no_grad():
        module.rwkv_learned_write_k_up[0, 0] = 0.25
    changed = module._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        None,
    )
    assert not torch.equal(changed[0], features[0])
    assert torch.isfinite(torch.stack(changed)).all()


def test_query_state_identity_capture_preserves_forward_and_trains_state() -> None:
    module = _module(write_address_gain=0.25)
    model = torch.nn.Module()
    model.attention = module
    shape = (1, 3, module.state_read_dim)
    projected = torch.randn(shape)
    positive_state = torch.randn(shape, requires_grad=True)
    donor_state = torch.randn(shape, requires_grad=True)
    global_state = torch.randn(shape)
    hidden = torch.randn(1, 3, module.hidden_size)
    projected_keys = torch.randn(1, module.rwkv_ms_num_states, 2, requires_grad=True)
    routes = torch.nn.functional.one_hot(
        torch.tensor([[0, 1, 0]]),
        num_classes=module.rwkv_ms_num_states,
    ).float()
    module.projected_kv_keys = projected_keys
    module.last_read_routes = routes
    module.last_write_routes = torch.nn.functional.one_hot(
        torch.tensor([[1, 0, 1]]),
        num_classes=module.rwkv_ms_num_states,
    ).float()
    baseline = module._fuse_projected_rwkv_reads(
        projected,
        positive_state,
        global_recurrent_reads=global_state,
        hidden_states=hidden,
    )
    audit = rwkv_query_state_identity.install(model)
    write_addresses = rwkv_query_state_identity.capture_write_addresses(model)
    rwkv_query_state_identity.set_fixed_query_addresses(model, write_addresses)
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
    module.projected_kv_surprise = torch.zeros(1, module.rwkv_ms_num_states)
    module._projected_kv_slot_token_reads(hidden)
    actual = module._fuse_projected_rwkv_reads(
        projected,
        positive_state,
        global_recurrent_reads=global_state,
        hidden_states=hidden,
    )
    positive = rwkv_query_state_identity.capture(model)
    expected_address = (
        (projected_keys[:, 0:1] + projected_keys[:, 1:2] * 2.0) / 3.0
    ).expand_as(positive_state)
    assert torch.allclose(positive[0].query_address, expected_address)
    assert positive[0].query_address.requires_grad is False
    module._fuse_projected_rwkv_reads(
        projected,
        donor_state,
        global_recurrent_reads=global_state,
        hidden_states=hidden,
    )
    donor = rwkv_query_state_identity.capture(model)
    _, _, loss = rwkv_query_state_identity.donor_hinge(
        positive,
        donor,
        torch.tensor([[-100, 1, 2]]),
        margin=3.0,
    )
    loss.backward()
    assert audit["forward_output_changed"] is False
    assert torch.equal(actual, baseline)
    assert positive_state.grad is not None and bool(positive_state.grad.ne(0).any())
    assert donor_state.grad is not None and bool(donor_state.grad.ne(0).any())
    assert projected_keys.grad is None


def test_selected_projected_key_expands_over_only_valid_write_tokens() -> None:
    module = _module(write_address_gain=0.25)
    hidden = torch.randn(2, 3, module.hidden_size)
    token_mask = torch.tensor([[True, True, False], [True, False, True]])
    module.projected_kv_keys = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]]
    )
    module.last_write_routes = torch.tensor(
        [[[0.0, 1.0]], [[1.0, 0.0]]]
    )

    address_seq = module._projected_rwkv_write_address_sequence(hidden, token_mask)

    expected = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            [[-1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
        ]
    )
    assert torch.equal(address_seq, expected)


def test_wider_projected_key_folds_both_halves_into_rwkv_features() -> None:
    module = _module(write_address_gain=0.25, projected_key_dim=4)
    hidden = torch.randn(1, 2, module.hidden_size)
    module.projected_kv_keys = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.0] * 4]])
    module.last_write_routes = torch.tensor([[[1.0, 0.0]]])

    address_seq = module._projected_rwkv_write_address_sequence(hidden, None)

    expected = torch.tensor([[(1.0 + 3.0) / 2**0.5, (2.0 + 4.0) / 2**0.5]])
    assert torch.allclose(address_seq[:, :1], expected)
    assert torch.equal(address_seq[:, 0], address_seq[:, 1])


def test_address_key_changes_recurrent_write_state_but_not_slot_route() -> None:
    keyed = _module(write_address_gain=0.25)
    baseline = copy.deepcopy(keyed)
    baseline.rwkv_ms_write_address_gain = 0.0
    batch_size, seq_len = 2, 4
    source = torch.randn(batch_size, seq_len, keyed.state_read_dim)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1, 1))
    decay = torch.sigmoid(torch.randn(batch_size, seq_len, 1, 1))
    token_mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    routes = torch.zeros(batch_size, seq_len, keyed.rwkv_ms_num_states)
    routes[0, :, 0] = token_mask[0]
    routes[1, :, 1] = token_mask[1]
    address = torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]]).expand(-1, seq_len, -1)
    address = address * token_mask.unsqueeze(-1)
    state = torch.zeros(
        batch_size,
        keyed.num_state_heads,
        keyed.rwkv_ms_num_states,
        keyed.rank,
        keyed.rank,
    )

    baseline_state, _ = baseline._rwkv_ms_scan(
        state,
        source,
        beta,
        decay,
        token_mask,
        write_only=True,
        write_route_seq=routes,
        write_address_seq=address,
    )
    keyed_state, _ = keyed._rwkv_ms_scan(
        state,
        source,
        beta,
        decay,
        token_mask,
        write_only=True,
        write_route_seq=routes,
        write_address_seq=address,
    )

    assert not torch.equal(keyed_state, baseline_state)
    assert torch.equal(keyed.last_write_routes, baseline.last_write_routes)
    assert torch.count_nonzero(keyed_state[0, :, 1]).item() == 0
    assert torch.count_nonzero(keyed_state[1, :, 0]).item() == 0


def test_address_keyed_screen_protocol_and_bindings_are_locked() -> None:
    protocol = screen.validate_protocol()
    candidate = screen.CANDIDATES[0]

    assert protocol["architecture"]["projected_address_dim"] == 64
    assert protocol["architecture"]["rwkv_feature_dim"] == 32
    assert protocol["execution"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert candidate["hybrid_mode"] == "address_keyed_moe_deepembed_ffn"
    assert candidate["write_address_gain"] == 0.25

    sparse = screen.base
    deepembed = sparse.base
    original_candidates = sparse.CANDIDATES
    original_evidence = deepembed.local_evidence
    original_write = deepembed.hybrid_screen.write_state
    with screen.screen_bindings():
        assert sparse.CANDIDATES == screen.CANDIDATES
        assert deepembed.local_evidence is screen.local_evidence
        assert deepembed.hybrid_screen.write_state is screen.write_state
        assert screen.build_config(candidate).rwkv_ms_write_address_gain == 0.25
    assert sparse.CANDIDATES is original_candidates
    assert deepembed.local_evidence is original_evidence
    assert deepembed.hybrid_screen.write_state is original_write


def test_address_keyed_causal_protocol_and_native_write_are_locked() -> None:
    protocol = causal_train.validate_protocol()
    original_write = causal_train.evolution._native_write
    original_runner = causal_train.SHARED_TRAINER.RUNNER_BINDING_PATH

    assert protocol["training"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["training"]["optimizer_updates"] == 16
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        causal_train.HELDOUT_ORDINALS
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    with causal_train.bindings():
        assert causal_train.evolution._native_write is causal_train.keyed_native_write
        assert causal_train.SHARED_TRAINER.RUNNER_BINDING_PATH == causal_train.Path(
            causal_train.__file__
        )
    assert causal_train.evolution._native_write is original_write
    assert causal_train.SHARED_TRAINER.RUNNER_BINDING_PATH == original_runner


def test_learned_write_protocol_and_signed_endpoint_are_locked() -> None:
    protocol = learned_train.validate_protocol()
    assert protocol["architecture"]["learned_write_conditioner"] == "rank2_per_feature_low_rank"
    assert protocol["training"]["global_batch_rows"] == 4
    assert learned_train.SELECTED_CANDIDATE["learned_write_rank"] == 2
    result = learned_train.json.loads(
        (
            learned_train.SCRIPT_DIR
            / "local_artifacts/natural_memory_native_rwkv_address_keyed_learned_write_causal_train_v3/result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "address_keyed_learned_write_heldout_failed_generation_blocked"
    assert result["heldout_causal_endpoint"]["checks"]["donor_minus_correct_mean_ce_positive"] is False
    assert result["heldout_causal_endpoint"]["checks"]["projected_carrier_fixed_every_row"] is True


def test_query_state_identity_protocol_and_fresh_endpoint_are_locked() -> None:
    protocol = identity_train.validate_protocol()
    assert protocol["architecture"]["identity_probe_parameters"] == 0
    assert protocol["architecture"]["forward_output_changed"] is False
    assert protocol["training"]["identity_margin"] == 0.2
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        identity_train.HELDOUT_ORDINALS
    )
    assert set(identity_train.HELDOUT_ORDINALS).isdisjoint(
        learned_train.shared.HELDOUT_ORDINALS
    )


def test_query_state_identity_backward_serializes_scalar_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    model = torch.nn.Module()
    positive = torch.tensor(0.1, requires_grad=True)
    donor = torch.tensor(0.2, requires_grad=True)
    positive.register_hook(lambda gradient: events.append(("positive", gradient.item())))
    donor.register_hook(lambda gradient: events.append(("donor", gradient.item())))
    monkeypatch.setattr(
        identity_train,
        "_ORIGINAL_BACKWARD_LOGITS",
        lambda *args, **kwargs: events.append("answer") or (1, 1),
    )
    monkeypatch.setattr(
        identity_train.identity,
        "clear",
        lambda *args, **kwargs: events.append("clear"),
    )
    monkeypatch.setattr(
        identity_train,
        "reset_delta_mem_states",
        lambda *args, **kwargs: events.append("reset"),
    )
    monkeypatch.setattr(
        identity_train.evolution,
        "release_native_row_allocator_cache",
        lambda *args, **kwargs: events.append("release"),
    )
    identity_train._reset_identity_metrics()
    identity_train._pending_identity = (model, positive, donor)
    scale = identity_train.IDENTITY_WEIGHT / identity_train.causal_train.GLOBAL_BATCH_SIZE

    identity_train._backward_logits_with_identity(
        torch.tensor([0.0]),
        torch.tensor([0]),
        coefficient=1.0,
    )

    assert events == [
        "answer",
        "clear",
        "reset",
        "release",
        ("positive", -scale),
        "clear",
        "reset",
        "release",
        ("donor", scale),
        "clear",
        "reset",
        "release",
    ]
    assert identity_train._pending_identity is None


def test_serialized_graph_protocol_validates_inside_active_bindings() -> None:
    protocol = causal_train_v5.validate_protocol()

    assert protocol["training"]["optimizer_state_cpu_offload_enabled"] is True
    assert protocol["training"]["control_branch_graph_serialization_enabled"] is True
    assert protocol["training"]["maximum_simultaneous_autograd_graphs_per_rank"] == 1
    assert causal_train_v5.validate_v4_failure()["heldout_causal_endpoint_opened"] is False
    with causal_train_v5.bindings():
        assert causal_execution.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS is True
        assert causal_execution.SERIALIZE_CONTROL_BRANCH_GRAPHS is True
        rebound_protocol = causal_train_v5.validate_protocol()
    assert rebound_protocol["receipt"] == protocol["receipt"]


def test_serialized_control_metric_forward_has_no_graph_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    batch = SimpleNamespace(labels=torch.tensor([[1, 2]]))

    def intervened(*args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, bool]]:
        del args, kwargs
        events.append(("forward_grad_enabled", torch.is_grad_enabled()))
        return torch.tensor([[[1.0, 2.0]]]), {
            "projected_carrier_references_fixed": True
        }

    def detached(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
        del logits, labels
        events.append(("metric_grad_enabled", torch.is_grad_enabled()))
        return 0.75, 1

    monkeypatch.setattr(
        causal_execution,
        "checkpointed_intervened_write_read",
        intervened,
    )
    monkeypatch.setattr(causal_execution.contrast, "detached_answer_ce", detached)
    monkeypatch.setattr(
        causal_execution,
        "reset_delta_mem_states",
        lambda model: events.append(("reset", model)),
    )
    monkeypatch.setattr(
        causal_execution.evolution,
        "release_native_row_allocator_cache",
        lambda device: events.append(("release", device)),
    )

    mean_ce, tokens, audit = (
        causal_execution.evaluate_intervened_condition_without_grad(
            "model",
            batch,
            donor_batch=None,
            rotate_recurrent_layers=True,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    )

    assert (mean_ce, tokens) == (0.75, 1)
    assert audit["projected_carrier_references_fixed"] is True
    assert events == [
        ("forward_grad_enabled", False),
        ("metric_grad_enabled", False),
        ("reset", "model"),
        ("release", torch.device("cpu")),
    ]


def test_serialized_active_control_backwards_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    batch = SimpleNamespace(labels=torch.tensor([[1, 2]]))
    parameter = torch.tensor(2.0, requires_grad=True)

    def intervened(*args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, bool]]:
        del args, kwargs
        events.append(("forward_grad_enabled", torch.is_grad_enabled()))
        return parameter.reshape(1, 1, 1), {
            "projected_carrier_references_fixed": True
        }

    def backward(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        coefficient: float,
    ) -> tuple[int, int]:
        del labels
        events.append(("backward_coefficient", coefficient))
        logits.sum().backward()
        return 1, 1

    monkeypatch.setattr(
        causal_execution,
        "checkpointed_intervened_write_read",
        intervened,
    )
    monkeypatch.setattr(causal_execution, "backward_logits", backward)
    monkeypatch.setattr(
        causal_execution,
        "reset_delta_mem_states",
        lambda model: events.append(("reset", model)),
    )
    monkeypatch.setattr(
        causal_execution.evolution,
        "release_native_row_allocator_cache",
        lambda device: events.append(("release", device)),
    )

    tokens, chunks, audit = (
        causal_execution.backward_serialized_intervened_condition(
            "model",
            batch,
            donor_batch=batch,
            rotate_recurrent_layers=False,
            coefficient=-1.0,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    )

    assert (tokens, chunks) == (1, 1)
    assert audit["projected_carrier_references_fixed"] is True
    assert parameter.grad is not None and parameter.grad.item() == 1.0
    assert events == [
        ("forward_grad_enabled", True),
        ("backward_coefficient", -1.0),
        ("reset", "model"),
        ("release", torch.device("cpu")),
    ]


def test_keyed_runtime_configuration_does_not_reset_learned_fusion_bias() -> None:
    module = _module(write_address_gain=0.25)
    with torch.no_grad():
        module.memory_fusion_bias.fill_(1.75)

    causal_train.configure_keyed_runtime(module)

    assert module.rwkv_ms_hybrid_mode == "address_keyed_moe_deepembed_ffn"
    assert module.rwkv_ms_write_address_gain == 0.25
    assert module.memory_fusion_bias.item() == 1.75
