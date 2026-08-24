from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed


PROTOCOL = Path(__file__).parents[2] / (
    "experiments/rethinking_rwkv_ms_gemma/"
    "natural_memory_native_rwkv_virtual_kv_identity_protocol_v1.json"
)


def test_virtual_kv_identity_protocol_is_signed_and_fail_closed() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = dict(protocol.pop("receipt"))
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": natural_memory_distributed.canonical_sha256(protocol),
    }
    architecture = protocol["architecture"]
    assert architecture["anchor_layers"] == [5, 11, 17, 23]
    assert architecture["virtual_slots"] == 4
    assert architecture["query_length"] == 1
    assert architecture["attention_implementation"] == "eager only"
    assert architecture["model_parameters_updated"] is False
    assert architecture["full_bandwidth_feedback_installed"] is False
    lifecycle = protocol["data_lifecycle"]
    assert lifecycle["already_open_bundles_only"] == ["fit", "retrieval"]
    assert lifecycle["protected_bytes_tokenized_or_forwarded"] is False
    assert protocol["execution"]["world_size"] == 4
    assert protocol["execution"]["hf_endpoint"] == "https://hf-mirror.com"
