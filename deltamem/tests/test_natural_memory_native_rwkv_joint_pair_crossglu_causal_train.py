from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_joint_pair_crossglu_causal_train as training,
)


def test_crossglu_causal_protocol_is_signed_and_generation_blocked():
    protocol, mechanics, crossfit = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert protocol["generation_authorized"] is False
    assert protocol["causal_endpoint_authorized"] is True
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert mechanics["status"] == "joint_pair_crossglu_mechanics_passed_generation_blocked"
    assert crossfit["status"] == "bilinear_crossfit_passed_causal_training_design_authorized"


def test_crossglu_causal_contract_keeps_small_serialized_update_budget():
    protocol, _, _ = training.validate_protocol()
    architecture = protocol["architecture"]
    train = protocol["training"]

    assert architecture["hybrid_mode"] == "joint_pair_crossglu"
    assert architecture["trainable_parameter_tensors"] == 126
    assert architecture["trainable_parameter_elements"] == 172032
    assert train["optimizer_updates"] == 8
    assert train["global_batch_rows"] == 4
    assert train["local_rows_per_rank"] == 1
    assert train["control_graph_serialization"] == "serialized"
    assert train["optimizer_state_cpu_offload_enabled"] is True


def test_crossglu_causal_endpoint_remains_on_open_development_rows():
    protocol, _, _ = training.validate_protocol()
    endpoint = protocol["heldout_causal_endpoint"]

    assert endpoint["rows"] == len(training.HELDOUT_SOURCES)
    assert endpoint["source_indices"] == list(training.HELDOUT_SOURCES)
    assert endpoint["required_ce_margins"]["minimum_donor_positive_row_fraction"] == 0.75
    assert protocol["claim_policy"]["native_gain"].startswith("Forbidden")
