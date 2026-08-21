from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_bidirectional_sign_integration as integration,
)
from experiments.rethinking_rwkv_ms_gemma.rwkv_bidirectional_sign_binding import (
    BidirectionalDiagonalSignBinding,
)


def _binding() -> BidirectionalDiagonalSignBinding:
    width = 4
    left_projection = torch.eye(width)
    right_projection = torch.roll(torch.eye(width), shifts=1, dims=1)
    return BidirectionalDiagonalSignBinding(
        width,
        address_dim=width,
        left_projection=left_projection,
        right_projection=right_projection,
        frequency=0.5,
        trainable_projection=False,
    )


def _assert_byte_identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def test_matching_two_axis_encode_and_read_cancel_exactly() -> None:
    binding = _binding()
    address = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]
    )
    state = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4).sub(11.0)
    receptance = torch.tensor(
        [[0.25, -0.5, 1.5, 2.0], [-1.0, 0.75, 0.5, -0.25]]
    )

    encoded = binding.encode_state(address, state)
    decoded = binding.decoded_read(address, encoded, receptance)
    reference = torch.matmul(state, receptance.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(decoded, reference, atol=0.0, rtol=0.0)


def test_wrong_donor_distorts_both_state_axes() -> None:
    binding = _binding()
    target_address = torch.ones(1, 4)
    donor_address = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
    state = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 7.0, 11.0, 13.0],
                [17.0, 19.0, 23.0, 29.0],
                [31.0, 37.0, 41.0, 43.0],
            ]
        ]
    )
    receptance = torch.tensor([[0.5, -1.0, 1.5, 2.0]])

    target_left, target_right = binding.codes(target_address)
    donor_left, donor_right = binding.codes(donor_address)
    assert bool(target_left.ne(donor_left).any())
    assert bool(target_right.ne(donor_right).any())

    donor_encoded = binding.encode_state(donor_address, state)
    wrong_read = binding.decoded_read(target_address, donor_encoded, receptance)
    relative_left = target_left * donor_left
    relative_right = target_right * donor_right
    expected_wrong = relative_left * torch.matmul(
        state,
        (relative_right * receptance).unsqueeze(-1),
    ).squeeze(-1)
    correct_read = torch.matmul(state, receptance.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(wrong_read, expected_wrong, atol=0.0, rtol=0.0)
    assert not torch.equal(wrong_read, correct_read)


def test_zero_address_is_byte_identity_for_state_and_write_features() -> None:
    binding = _binding()
    address = torch.zeros(2, 4)
    state = torch.tensor(
        [
            [[-0.0, 1.0, -2.0, 3.0]] * 4,
            [[4.0, -5.0, 6.0, -7.0]] * 4,
        ]
    )
    features = tuple(
        torch.arange(8, dtype=torch.float32).reshape(2, 4).sub(offset)
        for offset in (1.0, 3.0, 5.0, 7.0)
    )

    left_code, right_code = binding.codes(address)
    assert torch.equal(left_code, torch.ones_like(left_code))
    assert torch.equal(right_code, torch.ones_like(right_code))
    _assert_byte_identical(binding.encode_state(address, state), state)
    for actual, expected in zip(
        binding.bind_features(address, *features),
        features,
        strict=True,
    ):
        _assert_byte_identical(actual, expected)


def test_state_rebase_matches_fresh_encoding_and_new_address_read() -> None:
    binding = _binding()
    old_address = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    new_address = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
    state = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).sub(8.0)
    receptance = torch.tensor([[1.0, -0.5, 0.25, 2.0]])

    old_encoded = binding.encode_state(old_address, state)
    rebased = binding.rebase_state(old_address, new_address, old_encoded)
    freshly_encoded = binding.encode_state(new_address, state)

    torch.testing.assert_close(rebased, freshly_encoded, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        binding.decoded_read(new_address, rebased, receptance),
        torch.matmul(state, receptance.unsqueeze(-1)).squeeze(-1),
        atol=0.0,
        rtol=0.0,
    )


def test_bind_features_places_kab_right_and_v_left() -> None:
    binding = _binding()
    address = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
    k = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    v = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    a = torch.tensor([[9.0, 10.0, 11.0, 12.0]])
    b = torch.tensor([[13.0, 14.0, 15.0, 16.0]])
    left_code, right_code = binding.codes(address)
    assert not torch.equal(left_code, right_code)

    bound_k, bound_v, bound_a, bound_b = binding.bind_features(
        address,
        k,
        v,
        a,
        b,
    )

    torch.testing.assert_close(bound_k, k * right_code, atol=0.0, rtol=0.0)
    torch.testing.assert_close(bound_v, v * left_code, atol=0.0, rtol=0.0)
    torch.testing.assert_close(bound_a, a * right_code, atol=0.0, rtol=0.0)
    torch.testing.assert_close(bound_b, b * right_code, atol=0.0, rtol=0.0)


def test_write_hook_records_full_address_and_independent_right_code() -> None:
    binding = _binding()
    keys = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]],
            [[1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
        ]
    )
    routes = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        ]
    )
    module = SimpleNamespace(
        projected_kv_keys=keys,
        projected_kv_key_dim=4,
        last_write_routes=routes,
        rwkv_bidirectional_sign_binding=binding,
        rwkv_bidirectional_sign_enabled=True,
        rwkv_bidirectional_sign_original_write_features=(
            lambda k, v, a, b, address_seq, token_mask: (k, v, a, b)
        ),
        rwkv_bidirectional_sign_write_address=None,
        rwkv_bidirectional_sign_write_left_code=None,
        rwkv_bidirectional_sign_write_right_code=None,
        rwkv_rotary_write_address=None,
    )
    features = tuple(torch.randn(2, 3, 4) for _ in range(4))
    address_seq = torch.zeros(2, 3, 4)

    actual = integration._write_features(
        module,
        *features,
        address_seq,
        None,
    )
    full_address = torch.einsum("bts,bsd->btd", routes, keys)
    expected_left, expected_right = binding.codes(full_address)
    expected = binding.bind_features(full_address, *features)

    for actual_feature, expected_feature in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_feature,
            expected_feature,
            atol=0.0,
            rtol=0.0,
        )
    assert tuple(module.rwkv_bidirectional_sign_write_right_code.shape) == (2, 3, 4)
    torch.testing.assert_close(
        module.rwkv_bidirectional_sign_write_left_code,
        expected_left,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        module.rwkv_bidirectional_sign_write_right_code,
        expected_right,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(expected_left, expected_right)
    assert module.rwkv_rotary_write_address is module.rwkv_bidirectional_sign_write_address


def test_slot_read_integration_cancels_right_code_with_expected_shapes() -> None:
    binding = _binding()
    keys = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]],
            [[1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
        ]
    )
    module = SimpleNamespace(
        projected_kv_keys=keys,
        rwkv_bidirectional_sign_binding=binding,
    )
    state = torch.arange(64, dtype=torch.float32).reshape(2, 1, 2, 4, 4).sub(20.0)
    receptance = torch.tensor(
        [
            [[[1.0, -0.5, 0.25, 2.0]], [[-1.0, 1.5, 0.5, 0.75]]],
            [[[0.5, 1.0, -1.0, 0.25]], [[2.0, -0.25, 1.0, -0.5]]],
        ]
    )
    slot_left, slot_right = binding.codes(keys)
    assert bool(slot_right.eq(-1.0).any())
    assert bool(slot_left.ne(slot_right).any())
    encoded_state = (
        slot_left.unsqueeze(1).unsqueeze(-1)
        * state
        * slot_right.unsqueeze(1).unsqueeze(-2)
    )

    decoded, expanded_left, expanded_right = integration._decoded_slot_reads(
        module,
        encoded_state,
        receptance,
    )
    reference = torch.einsum("bhsij,bthj->bthsi", state, receptance)

    assert tuple(decoded.shape) == (2, 2, 1, 2, 4)
    assert tuple(expanded_left.shape) == (2, 2, 2, 4)
    assert tuple(expanded_right.shape) == (2, 2, 2, 4)
    torch.testing.assert_close(decoded, reference, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        expanded_right,
        slot_right.unsqueeze(1).expand(-1, 2, -1, -1),
        atol=0.0,
        rtol=0.0,
    )


def _read_basis_fixture() -> tuple[
    SimpleNamespace,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    binding = _binding()
    keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]]]
    )
    unbound_state = torch.arange(32, dtype=torch.float32).reshape(1, 1, 2, 4, 4)
    slot_left, slot_right = binding.codes(keys)
    encoded_state = (
        slot_left.unsqueeze(1).unsqueeze(-1)
        * unbound_state
        * slot_right.unsqueeze(1).unsqueeze(-2)
    )
    native_receptance = torch.tensor(
        [[[[1.0, -0.5, 0.25, 2.0]], [[-1.0, 1.5, 0.5, 0.75]]]]
    )
    native_slots = torch.full((1, 2, 1, 2, 4), -919.0)
    native_gate = torch.tensor(
        [[[0.125, 0.25, 0.5, 1.0], [1.0, 0.5, 0.25, 0.125]]]
    )
    memory_source = torch.randn(1, 2, 4)
    expected_decoded = torch.einsum(
        "bhsij,bthj->bthsi",
        unbound_state,
        native_receptance,
    )
    module = SimpleNamespace(
        projected_kv_keys=keys,
        projected_kv_key_dim=4,
        rwkv_bidirectional_sign_binding=binding,
        rwkv_bidirectional_sign_enabled=True,
        rwkv_bidirectional_sign_capture_enabled=False,
        rwkv_bidirectional_sign_read_kind="addressed",
        rwkv_bidirectional_sign_query_address=keys[:, :1].expand(-1, 2, -1),
        rwkv_bidirectional_sign_original_read_basis=(
            lambda state, memory_source_seq, token_mask: (
                native_receptance,
                native_slots,
                native_gate,
            )
        ),
    )
    return (
        module,
        encoded_state,
        memory_source,
        native_receptance,
        native_gate,
        expected_decoded,
    )


def test_read_basis_decodes_slots_and_preserves_native_receptance_and_gate() -> None:
    (
        module,
        encoded_state,
        memory_source,
        native_receptance,
        native_gate,
        expected_decoded,
    ) = _read_basis_fixture()

    returned_receptance, decoded_slots, returned_gate = integration._read_basis(
        module,
        encoded_state,
        memory_source,
        None,
    )

    assert returned_receptance is native_receptance
    assert returned_gate is native_gate
    assert tuple(decoded_slots.shape) == (1, 2, 1, 2, 4)
    torch.testing.assert_close(decoded_slots, expected_decoded, atol=0.0, rtol=0.0)


def test_read_basis_requires_an_explicit_call_site_tag() -> None:
    module, encoded_state, memory_source, _, _, _ = _read_basis_fixture()
    module.rwkv_bidirectional_sign_read_kind = None

    with pytest.raises(
        RuntimeError,
        match="no explicit call-site tag",
    ):
        integration._read_basis(
            module,
            encoded_state,
            memory_source,
            None,
        )


def test_outer_read_wrappers_only_scope_tags_around_native_math() -> None:
    keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]]]
    )
    state = torch.randn(1, 1, 2, 4, 4)
    memory_source = torch.randn(1, 3, 4)
    routes = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]]])
    token_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    observed_tags: list[str | None] = []
    module = SimpleNamespace(
        projected_kv_keys=keys,
        projected_kv_key_dim=4,
        rwkv_bidirectional_sign_read_kind=None,
        rwkv_bidirectional_sign_query_address=None,
    )

    def native_addressed(
        native_state: torch.Tensor,
        native_memory: torch.Tensor,
        native_routes: torch.Tensor,
        native_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        assert native_state is state
        assert native_memory is memory_source
        assert native_routes is routes
        assert native_mask is token_mask
        observed_tags.append(module.rwkv_bidirectional_sign_read_kind)
        return native_memory + native_routes.sum(dim=-1, keepdim=True)

    def native_global(
        native_state: torch.Tensor,
        native_memory: torch.Tensor,
        native_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        assert native_state is state
        assert native_memory is memory_source
        assert native_mask is token_mask
        observed_tags.append(module.rwkv_bidirectional_sign_read_kind)
        return native_memory.square().sub(0.25)

    module.rwkv_bidirectional_sign_original_addressed_reads = native_addressed
    module.rwkv_bidirectional_sign_original_global_reads = native_global

    addressed = integration._addressed_reads(
        module,
        state,
        memory_source,
        routes,
        token_mask,
    )
    expected_address = torch.einsum("bts,bsd->btd", routes, keys)
    torch.testing.assert_close(
        addressed,
        memory_source + routes.sum(dim=-1, keepdim=True),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        module.rwkv_bidirectional_sign_query_address,
        expected_address,
        atol=0.0,
        rtol=0.0,
    )
    assert module.rwkv_bidirectional_sign_read_kind is None

    global_read = integration._global_reads(
        module,
        state,
        memory_source,
        token_mask,
    )
    torch.testing.assert_close(
        global_read,
        memory_source.square().sub(0.25),
        atol=0.0,
        rtol=0.0,
    )
    assert module.rwkv_bidirectional_sign_read_kind is None
    assert observed_tags == ["addressed", "global"]


def _lifecycle_module(
    *,
    keys: torch.Tensor | None,
    occupied: torch.Tensor | None,
) -> SimpleNamespace:
    module = SimpleNamespace(
        projected_kv_keys=None if keys is None else keys.detach().clone(),
        projected_kv_occupied=(
            None if occupied is None else occupied.detach().clone().to(torch.bool)
        ),
        last_write_routes=None,
        rwkv_bidirectional_sign_binding=_binding(),
        rwkv_bidirectional_sign_enabled=True,
        rwkv_bidirectional_sign_pending_rebase=None,
        rwkv_bidirectional_sign_rebase_events=0,
        rwkv_bidirectional_sign_capture_enabled=True,
        rwkv_bidirectional_sign_rebase_capture=None,
        next_projected_kv_keys=None,
        next_projected_kv_occupied=None,
        next_write_routes=None,
        backend_transform=None,
        backend_scan_states=[],
    )

    def native_projected_slot_write(
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> None:
        assert hidden_states.ndim == 3
        assert token_mask is None or token_mask.shape == hidden_states.shape[:2]
        assert module.next_projected_kv_keys is not None
        assert module.next_projected_kv_occupied is not None
        assert module.next_write_routes is not None
        module.projected_kv_keys = module.next_projected_kv_keys.detach().clone()
        module.projected_kv_occupied = (
            module.next_projected_kv_occupied.detach().clone().to(torch.bool)
        )
        module.last_write_routes = module.next_write_routes.detach().clone()

    def native_backend_scan(
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del memory_k_seq, memory_v_seq, beta_seq, lambda_seq, kwargs
        module.backend_scan_states.append(state.detach().clone())
        next_state = (
            state
            if module.backend_transform is None
            else module.backend_transform(state)
        )
        reads = memory_q_seq.new_zeros(
            memory_q_seq.shape[0],
            memory_q_seq.shape[1],
            memory_q_seq.shape[2],
        )
        return next_state, reads

    module.rwkv_bidirectional_sign_original_projected_slot_write = (
        native_projected_slot_write
    )
    module.rwkv_bidirectional_sign_original_backend_scan = native_backend_scan
    return module


def _stage_projected_slot_write(
    module: SimpleNamespace,
    *,
    keys: torch.Tensor,
    occupied: torch.Tensor,
    selected_slots: tuple[int, ...],
) -> None:
    routes = keys.new_zeros(keys.shape[0], 1, keys.shape[1])
    for slot_index in selected_slots:
        routes[:, 0, slot_index] = 1.0
    module.next_projected_kv_keys = keys
    module.next_projected_kv_occupied = occupied
    module.next_write_routes = routes
    integration._projected_slot_write(
        module,
        torch.zeros(keys.shape[0], 1, keys.shape[-1]),
        torch.ones(keys.shape[0], 1, dtype=torch.bool),
    )


def _run_backend_scan(
    module: SimpleNamespace,
    state: torch.Tensor,
    *,
    write_only: bool = True,
) -> torch.Tensor:
    memory = torch.zeros(state.shape[0], 1, state.shape[-1])
    gates = torch.zeros(state.shape[0], 1, 1, 1)
    next_state, _ = integration._backend_scan(
        module,
        state,
        memory,
        memory,
        memory,
        gates,
        gates,
        write_only=write_only,
    )
    return next_state


def _encoded_slot(
    binding: BidirectionalDiagonalSignBinding,
    address: torch.Tensor,
    logical_state: torch.Tensor,
) -> torch.Tensor:
    return binding.encode_state(
        address.reshape(1, -1),
        logical_state.reshape(1, logical_state.shape[-2], logical_state.shape[-1]),
    )[0]


def test_empty_insert_preserves_the_zero_recurrent_slot_byte_for_byte() -> None:
    new_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    occupied = torch.tensor([[True, False]])
    module = _lifecycle_module(keys=None, occupied=None)
    state = torch.zeros(1, 1, 2, 4, 4)
    state[0, 0, 0, 0, 0] = -0.0
    original = state.detach().clone()

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(0,),
    )
    next_state = _run_backend_scan(module, state)

    assert module.rwkv_bidirectional_sign_rebase_events == 0
    assert module.rwkv_bidirectional_sign_pending_rebase is None
    _assert_byte_identical(module.backend_scan_states[-1], original)
    _assert_byte_identical(next_state, original)


def test_occupied_matching_key_update_rebases_to_the_new_code_exactly() -> None:
    old_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]]
    )
    new_keys = old_keys.detach().clone()
    new_keys[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    occupied = torch.tensor([[True, True]])
    module = _lifecycle_module(keys=old_keys, occupied=occupied)
    logical_state = torch.arange(16, dtype=torch.float32).reshape(4, 4).sub(7.0)
    state = torch.zeros(1, 1, 2, 4, 4)
    state[0, 0, 0] = _encoded_slot(
        module.rwkv_bidirectional_sign_binding,
        old_keys[0, 0],
        logical_state,
    )

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(0,),
    )
    next_state = _run_backend_scan(module, state)
    expected = _encoded_slot(
        module.rwkv_bidirectional_sign_binding,
        new_keys[0, 0],
        logical_state,
    )

    assert module.rwkv_bidirectional_sign_rebase_events == 1
    _assert_byte_identical(module.backend_scan_states[-1][0, 0, 0], expected)
    _assert_byte_identical(next_state[0, 0, 0], expected)


def test_disabled_binding_consumes_key_change_without_rebasing_state() -> None:
    old_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    new_keys = old_keys.detach().clone()
    new_keys[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    occupied = torch.tensor([[True, False]])
    module = _lifecycle_module(keys=old_keys, occupied=occupied)
    module.rwkv_bidirectional_sign_enabled = False
    state = torch.randn(1, 1, 2, 4, 4)
    original = state.detach().clone()

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(0,),
    )
    next_state = _run_backend_scan(module, state)

    assert module.rwkv_bidirectional_sign_pending_rebase is None
    assert module.rwkv_bidirectional_sign_rebase_events == 0
    _assert_byte_identical(module.backend_scan_states[-1], original)
    _assert_byte_identical(next_state, original)


def test_eviction_rebases_selected_state_and_preserves_other_slots() -> None:
    old_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0]]]
    )
    new_keys = old_keys.detach().clone()
    new_keys[:, 1] = torch.tensor([1.0, -1.0, -1.0, 1.0])
    occupied = torch.tensor([[True, True]])
    module = _lifecycle_module(keys=old_keys, occupied=occupied)
    logical_state = torch.arange(16, dtype=torch.float32).reshape(4, 4).add(1.0)
    state = torch.randn(1, 1, 2, 4, 4)
    state[0, 0, 1] = _encoded_slot(
        module.rwkv_bidirectional_sign_binding,
        old_keys[0, 1],
        logical_state,
    )
    unselected = state[:, :, 0].detach().clone()

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(1,),
    )
    next_state = _run_backend_scan(module, state)
    expected = _encoded_slot(
        module.rwkv_bidirectional_sign_binding,
        new_keys[0, 1],
        logical_state,
    )

    assert module.rwkv_bidirectional_sign_rebase_events == 1
    _assert_byte_identical(next_state[0, 0, 1], expected)
    _assert_byte_identical(next_state[:, :, 0], unselected)


def test_repeated_key_changes_preserve_logical_state_across_episodes() -> None:
    first_key = torch.tensor([1.0, 1.0, 1.0, 1.0])
    second_key = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    third_key = torch.tensor([1.0, -1.0, -1.0, 1.0])
    empty_key = torch.zeros(4)
    module = _lifecycle_module(keys=None, occupied=None)
    occupied = torch.tensor([[True, False]])
    logical_first = torch.arange(16, dtype=torch.float32).reshape(4, 4).add(2.0)
    logical_second = logical_first + torch.eye(4)
    state = torch.zeros(1, 1, 2, 4, 4)

    def first_native_write(scan_state: torch.Tensor) -> torch.Tensor:
        assert bool(scan_state[0, 0, 0].eq(0.0).all())
        updated = scan_state.detach().clone()
        updated[0, 0, 0] = _encoded_slot(
            module.rwkv_bidirectional_sign_binding,
            first_key,
            logical_first,
        )
        return updated

    module.backend_transform = first_native_write
    _stage_projected_slot_write(
        module,
        keys=torch.stack((first_key, empty_key)).unsqueeze(0),
        occupied=occupied,
        selected_slots=(0,),
    )
    state = _run_backend_scan(module, state)

    def second_native_write(scan_state: torch.Tensor) -> torch.Tensor:
        expected = _encoded_slot(
            module.rwkv_bidirectional_sign_binding,
            second_key,
            logical_first,
        )
        _assert_byte_identical(scan_state[0, 0, 0], expected)
        updated = scan_state.detach().clone()
        updated[0, 0, 0] = _encoded_slot(
            module.rwkv_bidirectional_sign_binding,
            second_key,
            logical_second,
        )
        return updated

    module.backend_transform = second_native_write
    _stage_projected_slot_write(
        module,
        keys=torch.stack((second_key, empty_key)).unsqueeze(0),
        occupied=occupied,
        selected_slots=(0,),
    )
    state = _run_backend_scan(module, state)

    def third_native_write(scan_state: torch.Tensor) -> torch.Tensor:
        expected = _encoded_slot(
            module.rwkv_bidirectional_sign_binding,
            third_key,
            logical_second,
        )
        _assert_byte_identical(scan_state[0, 0, 0], expected)
        return scan_state

    module.backend_transform = third_native_write
    _stage_projected_slot_write(
        module,
        keys=torch.stack((third_key, empty_key)).unsqueeze(0),
        occupied=occupied,
        selected_slots=(0,),
    )
    state = _run_backend_scan(module, state)
    receptance = torch.tensor([[0.5, -1.0, 1.5, 2.0]])
    decoded = module.rwkv_bidirectional_sign_binding.decoded_read(
        third_key.reshape(1, -1),
        state[0, 0, 0].reshape(1, 4, 4),
        receptance,
    )
    reference = torch.matmul(
        logical_second,
        receptance[0].unsqueeze(-1),
    ).squeeze(-1)

    assert module.rwkv_bidirectional_sign_rebase_events == 2
    torch.testing.assert_close(decoded[0], reference, atol=0.0, rtol=0.0)


def test_unselected_signed_zero_bytes_survive_a_neighbor_rebase() -> None:
    old_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]]]
    )
    new_keys = old_keys.detach().clone()
    new_keys[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    occupied = torch.tensor([[True, True]])
    module = _lifecycle_module(keys=old_keys, occupied=occupied)
    state = torch.ones(1, 1, 2, 4, 4)
    signed_zero = torch.zeros(4, 4)
    signed_zero[::2, 1::2] = -0.0
    signed_zero[1::2, ::2] = -0.0
    state[0, 0, 1] = signed_zero
    original_unselected = state[:, :, 1].detach().clone()

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(0,),
    )
    next_state = _run_backend_scan(module, state)

    _assert_byte_identical(next_state[:, :, 1], original_unselected)


def test_stale_recurrent_state_in_an_empty_insert_is_rejected() -> None:
    old_keys = torch.zeros(1, 2, 4)
    new_keys = old_keys.detach().clone()
    new_keys[:, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    old_occupied = torch.tensor([[False, False]])
    new_occupied = torch.tensor([[True, False]])
    module = _lifecycle_module(keys=old_keys, occupied=old_occupied)
    state = torch.zeros(1, 1, 2, 4, 4)
    state[0, 0, 0, 2, 3] = 1.0

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=new_occupied,
        selected_slots=(0,),
    )
    with pytest.raises(RuntimeError, match="inserted slot contains stale"):
        _run_backend_scan(module, state)


def test_delayed_rebase_is_rejected_before_another_write_or_read_scan() -> None:
    old_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    new_keys = old_keys.detach().clone()
    new_keys[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    occupied = torch.tensor([[True, False]])
    module = _lifecycle_module(keys=old_keys, occupied=occupied)
    state = torch.zeros(1, 1, 2, 4, 4)

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(0,),
    )
    with pytest.raises(RuntimeError, match="delayed past its scan"):
        _stage_projected_slot_write(
            module,
            keys=new_keys,
            occupied=occupied,
            selected_slots=(0,),
        )
    with pytest.raises(RuntimeError, match="non-write scan"):
        _run_backend_scan(module, state, write_only=False)


def test_pending_rebase_is_consumed_once_and_cannot_double_apply() -> None:
    old_keys = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    new_keys = old_keys.detach().clone()
    new_keys[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    occupied = torch.tensor([[True, False]])
    module = _lifecycle_module(keys=old_keys, occupied=occupied)
    logical_state = torch.arange(16, dtype=torch.float32).reshape(4, 4).add(1.0)
    state = torch.zeros(1, 1, 2, 4, 4)
    state[0, 0, 0] = _encoded_slot(
        module.rwkv_bidirectional_sign_binding,
        old_keys[0, 0],
        logical_state,
    )

    _stage_projected_slot_write(
        module,
        keys=new_keys,
        occupied=occupied,
        selected_slots=(0,),
    )
    once = _run_backend_scan(module, state)
    events_after_first_scan = module.rwkv_bidirectional_sign_rebase_events
    twice = _run_backend_scan(module, once)

    assert module.rwkv_bidirectional_sign_pending_rebase is None
    assert events_after_first_scan == 1
    assert module.rwkv_bidirectional_sign_rebase_events == events_after_first_scan
    _assert_byte_identical(twice, once)


def test_binding_toggle_with_live_state_is_rejected(monkeypatch) -> None:
    module = SimpleNamespace(
        rwkv_bidirectional_sign_enabled=True,
        delta_state=torch.ones(1, 1, 1, 4, 4),
    )
    monkeypatch.setattr(
        integration,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )

    with pytest.raises(RuntimeError, match="live recurrent state"):
        integration.set_enabled(object(), False)

    module.delta_state.zero_()
    integration.set_enabled(object(), False)
    assert module.rwkv_bidirectional_sign_enabled is False


def test_install_rejects_trainable_bidirectional_projections() -> None:
    with pytest.raises(ValueError, match="projections are frozen"):
        integration.install(torch.nn.Module(), trainable_projection=True)
