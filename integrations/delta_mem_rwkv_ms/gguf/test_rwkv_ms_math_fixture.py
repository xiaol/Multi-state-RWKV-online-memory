from __future__ import annotations

import sys
from pathlib import Path

import torch


GGUF_DIR = Path(__file__).resolve().parent
if str(GGUF_DIR) not in sys.path:
    sys.path.insert(0, str(GGUF_DIR))

from rwkv_ms_math_fixture_common import rwkv_ms_scan  # noqa: E402


def test_rwkv_ms_scan_two_chunks_match_full_stream() -> None:
    torch.manual_seed(17)
    from hrm_rwkv7 import HRMRWKV7LowRankCore

    batch_size = 2
    seq_len = 7
    rank = 2
    num_states = 3
    split = 3
    core = HRMRWKV7LowRankCore(
        dim=rank,
        head_size=rank,
        layer_id=1,
        n_layer=4,
    )
    with torch.no_grad():
        core.output.weight.normal_(mean=0.0, std=0.2)
    core.eval()

    memory_source = torch.randn(batch_size, seq_len, rank)
    initial_state = torch.randn(batch_size, 1, num_states, rank, rank) * 0.1
    initial_positions = torch.tensor([3, 4], dtype=torch.long)
    initial_previous_source = torch.randn(batch_size, rank)
    token_mask = torch.tensor(
        [
            [True, False, True, False, True, True, False],
            [False, True, False, True, False, True, True],
        ]
    )
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, rank, 1))
    lam = 1.0 - beta
    config = {
        "rank": rank,
        "num_state_heads": 1,
        "rankwise_gates": True,
        "state_update_mode": "standard",
    }
    scan_config = {
        "rwkv_ms_num_states": num_states,
        "rwkv_ms_chunk_size": 2,
        "rwkv_ms_erase_gate": 1.0,
        "rwkv_ms_read_top_k": 0,
    }

    def scan(
        start: int,
        end: int,
        *,
        state: torch.Tensor,
        positions: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return rwkv_ms_scan(
            core=core,
            state=state,
            positions=positions,
            memory_source_seq=memory_source[:, start:end],
            beta_seq=beta[:, start:end],
            lambda_seq=lam[:, start:end],
            config=config,
            scan_config=scan_config,
            token_mask=token_mask[:, start:end],
            previous_source=previous_source,
            include_step_trace=False,
        )

    full = scan(
        0,
        seq_len,
        state=initial_state,
        positions=initial_positions,
        previous_source=initial_previous_source,
    )
    first = scan(
        0,
        split,
        state=initial_state,
        positions=initial_positions,
        previous_source=initial_previous_source,
    )
    second = scan(
        split,
        seq_len,
        state=first["final_state"],
        positions=first["final_positions"],
        previous_source=first["final_previous_source"],
    )

    chunked_reads = torch.cat([first["reads"], second["reads"]], dim=1)
    expected_positions = initial_positions + token_mask.sum(dim=1)
    expected_previous_source = torch.stack(
        [memory_source[0, 5], memory_source[1, 6]],
        dim=0,
    )
    assert torch.count_nonzero(full["reads"]) > 0
    assert torch.allclose(second["final_state"], full["final_state"], atol=1e-6, rtol=1e-5)
    assert torch.equal(second["final_positions"], full["final_positions"])
    assert torch.equal(full["final_positions"], expected_positions)
    assert torch.equal(full["final_previous_source"], expected_previous_source)
    assert torch.allclose(
        second["final_previous_source"],
        full["final_previous_source"],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(chunked_reads, full["reads"], atol=1e-6, rtol=1e-5)


def test_rwkv_ms_scan_legacy_input_does_not_emit_previous_source() -> None:
    from hrm_rwkv7 import HRMRWKV7LowRankCore

    core = HRMRWKV7LowRankCore(dim=2, head_size=2)
    result = rwkv_ms_scan(
        core=core,
        state=torch.zeros(1, 1, 2, 2, 2),
        positions=torch.zeros(1, dtype=torch.long),
        memory_source_seq=torch.zeros(1, 2, 2),
        beta_seq=torch.full((1, 2, 2, 1), 0.5),
        lambda_seq=torch.full((1, 2, 2, 1), 0.5),
        config={
            "rank": 2,
            "num_state_heads": 1,
            "rankwise_gates": True,
            "state_update_mode": "standard",
        },
        scan_config={
            "rwkv_ms_num_states": 2,
            "rwkv_ms_chunk_size": 2,
            "rwkv_ms_erase_gate": 1.0,
            "rwkv_ms_read_top_k": 0,
        },
        token_mask=torch.tensor([[True, False]]),
        previous_source=None,
        include_step_trace=False,
    )

    assert "final_previous_source" not in result
