from __future__ import annotations

from collections import Counter

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_recurrent_routed_residual_query_value as residual,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_recurrent_routed_residual_query_value_v2 as residual_v2,
)
from experiments.rethinking_rwkv_ms_gemma import (
    evaluate_natural_memory_native_recurrent_routed_residual_query_value_v2 as residual_v2_development,
)


def test_protocol_forbids_benchmark_time_selection() -> None:
    protocol, _ = residual.validate_protocol(residual.PREFLIGHT_UPDATES)

    architecture = protocol["architecture"]
    criteria = protocol["development_promotion_criteria"]
    assert architecture["dynamic_pair_gate"] is False
    assert architecture["task_router"] is False
    assert architecture["template_matcher"] is False
    assert architecture["dual_pass_selector"] is False
    assert architecture["baseline_fallback"] is False
    assert architecture["benchmark_time_parameter_override"] is False
    assert criteria["candidate_is_saved_checkpoint_only"] is True
    assert criteria["answer_selection_or_fallback"] is False


def test_v2_protocol_treats_slot_shuffle_as_invariance() -> None:
    protocol = residual.common.validate_signed_json(
        residual_v2.PROTOCOL,
        residual_v2.PROTOCOL_PAYLOAD_SHA256,
    )

    assert protocol["architecture"]["dynamic_pair_gate"] is False
    assert protocol["training"]["slot_shuffle_role"].startswith(
        "permutation-invariance"
    )
    criteria = protocol["development_promotion_criteria"]
    assert criteria["slot_shuffle_is_exact_permutation_invariance"] is True
    assert criteria["recovered_scoring_is_diagnostic_only"] is True
    assert protocol["final_benchmark"]["exact_raw_json_primary_scoring"] is True


def test_v2_training_focuses_narrative_and_strengthens_decay_updates() -> None:
    assert residual_v2.PREFLIGHT_UPDATES == 3
    assert residual_v2.PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES is True
    assert residual_v2.TARGET_COUNTS == {
        "attribution": 8,
        "narrative": 48,
        "scene": 8,
    }
    assert residual_v2.CONTROL_WEIGHTS["matched_donor_recurrent_state"] == 1.0
    assert residual_v2.BASELINE_ANCHOR_WEIGHT == 2.0
    assert residual_v2.FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES == {
        ".hrm_rwkv7_core.x_w",
        ".hrm_rwkv7_core.x_a",
        ".hrm_rwkv7_core.x_g",
    }
    assert not (
        residual_v2.FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
        & set(residual_v2.TRAINABLE_SUFFIXES)
    )
    protocol = residual.common.validate_signed_json(
        residual_v2.PROTOCOL,
        residual_v2.PROTOCOL_PAYLOAD_SHA256,
    )
    assert protocol["training"][
        "preflight_requires_all_trainable_families_changed"
    ] is True
    assert protocol["training"]["frozen_negligible_gradient_suffixes"] == sorted(
        residual_v2.FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
    )


def test_schedule_is_distinct_balanced_and_development_sealed() -> None:
    manifest, _ = common.validate_split_artifacts()
    rows_by_task = common.load_open_rows("train", manifest=manifest)
    schedule, payload = residual.build_schedule(rows_by_task)
    development = residual.load_v2_manifest()["development_source_ordinals"]
    targets = {row.target.row_sha256 for row in schedule}

    assert len(schedule) == residual.TRAIN_UPDATES * 8
    assert len(payload) == residual.TRAIN_UPDATES
    assert len(targets) == sum(residual.TARGET_COUNTS.values())
    assert Counter(row.target.task for row in schedule) == {
        task: count * 4 for task, count in residual.TARGET_COUNTS.items()
    }
    assert Counter(row.prompt_variant for row in schedule) == {
        variant: 64 for variant in range(4)
    }
    assert all(
        row.target.source_ordinal
        not in set(development[row.target.task])
        for row in schedule
    )


def test_first_update_requires_only_residual_projection_gradients() -> None:
    named_trainable = []
    for layer in range(common.EXPECTED_LAYERS):
        for suffix in residual.TRAINABLE_SUFFIXES:
            parameter = torch.nn.Parameter(torch.ones(1))
            parameter.grad = (
                torch.zeros_like(parameter)
                if suffix in residual.FIRST_STEP_ZERO_ALLOWED
                else torch.ones_like(parameter)
            )
            named_trainable.append(
                (f"model.layers.{layer}.self_attn{suffix}", parameter)
            )

    audit = residual.audit_gradients(named_trainable)

    assert audit["passed"] is True
    assert audit["families"][".rwkv_recurrent_value_proj"][
        "requires_nonzero_on_first_update"
    ] is True
    assert audit["families"][".rwkv_pair_value_proj"][
        "requires_nonzero_on_first_update"
    ] is True
    assert audit["families"][".rwkv_route_query_proj"][
        "requires_nonzero_on_first_update"
    ] is False


def test_protocol_uses_training_only_zero_state_anchor() -> None:
    protocol, _ = residual.validate_protocol(residual.PREFLIGHT_UPDATES)

    anchor = protocol["training"]["baseline_anchor"]
    assert anchor == {
        "weight": residual.BASELINE_ANCHOR_WEIGHT,
        "temperature": residual.BASELINE_ANCHOR_TEMPERATURE,
        "top_k": residual.BASELINE_ANCHOR_TOP_K,
        "teacher_condition": "zero_recurrent_state",
    }
    assert protocol["architecture"]["baseline_fallback"] is False


def test_v2_development_requires_invariance_and_all_effective_families() -> None:
    residual_v2_development.configure()

    assert residual_v2_development.development_v2.SLOT_SHUFFLE_EXPECTATION == (
        "invariance"
    )
    assert residual_v2_development.evaluator.TRAINED_SUFFIXES == (
        residual_v2.TRAINABLE_SUFFIXES
    )
    assert residual_v2_development.TRAINING_RESULT_RECEIPT == (
        "6667c5733e10f5a304f4907f009ca7ef01e65c81f642fe6247ae0520d0cb1ee9"
    )


def test_v3_is_balanced_and_sealed() -> None:
    from experiments.rethinking_rwkv_ms_gemma import (
        run_natural_memory_native_recurrent_routed_residual_query_value_v3 as residual_v3,
    )

    residual_v3.configure()
    protocol = common.validate_signed_json(
        residual_v3.PROTOCOL,
        residual_v3.PROTOCOL_PAYLOAD_SHA256,
    )
    assert residual_v3.TARGET_COUNTS == {
        "attribution": 16,
        "narrative": 32,
        "scene": 16,
    }
    assert protocol["architecture"]["dynamic_pair_gate"] is False
    assert protocol["architecture"]["task_router"] is False
    assert protocol["training"]["final_rows_opened_during_training"] is False
    assert protocol["frozen_inputs"]["failed_v2_development_receipt"] == (
        "f5f2703755b190ce78c07bfb1f042a0dad51647e36ac84be4feaa26c87fbf32a"
    )


def test_v3_development_binds_saved_checkpoint() -> None:
    from experiments.rethinking_rwkv_ms_gemma import (
        evaluate_natural_memory_native_recurrent_routed_residual_query_value_v3 as development_v3,
    )

    development_v3.configure()
    assert development_v3.TRAINING_RESULT_RECEIPT == (
        "b48ea22238a2c60d35dfd54dc3aa26a6ac3bf37d7c0608a5cd009bdf29811609"
    )
    assert development_v3.evaluator.TRAINED_SUFFIXES == (
        development_v3.residual_v3.TRAINABLE_SUFFIXES
    )
    assert development_v3.development_v2.SLOT_SHUFFLE_EXPECTATION == (
        "invariance"
    )
