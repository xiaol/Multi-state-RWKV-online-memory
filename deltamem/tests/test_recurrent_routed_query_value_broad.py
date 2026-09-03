from __future__ import annotations

from collections import defaultdict

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_recurrent_routed_query_value_broad as broad,
)


def test_protocol_binds_ungated_checkpoint_only_evaluation() -> None:
    protocol, _ = broad.validate_protocol(broad.PREFLIGHT_UPDATES)

    architecture = protocol["architecture"]
    criteria = protocol["development_promotion_criteria"]
    assert architecture["rwkv_pair_gate"] is False
    assert architecture["task_router"] is False
    assert architecture["dual_pass_selector"] is False
    assert architecture["baseline_fallback"] is False
    assert architecture["benchmark_time_parameter_override"] is False
    assert criteria["candidate_is_saved_checkpoint_only"] is True
    assert criteria["answer_selection_or_fallback"] is False


def test_schedule_excludes_development_and_covers_all_paraphrases() -> None:
    manifest, _ = common.validate_split_artifacts()
    rows_by_task = common.load_open_rows("train", manifest=manifest)
    schedule, payload = broad.build_schedule(rows_by_task)
    development = broad.load_v2_manifest()["development_source_ordinals"]
    targets: dict[str, set[int]] = defaultdict(set)
    variants: dict[tuple[str, int], set[int]] = defaultdict(set)

    for row in schedule:
        assert row.target.source_ordinal not in set(
            development[row.target.task]
        )
        targets[row.target.task].add(row.target.source_ordinal)
        variants[(row.target.task, row.target.source_ordinal)].add(
            row.prompt_variant
        )

    assert len(schedule) == broad.TRAIN_UPDATES * 8
    assert len(payload) == broad.TRAIN_UPDATES
    assert {task: len(values) for task, values in targets.items()} == (
        broad.TARGET_COUNTS
    )
    assert all(values == set(range(4)) for values in variants.values())


def test_first_update_audits_every_trainable_family() -> None:
    named_trainable = []
    for layer in range(common.EXPECTED_LAYERS):
        for suffix in broad.TRAINABLE_SUFFIXES:
            parameter = torch.nn.Parameter(torch.ones(1))
            parameter.grad = (
                torch.zeros_like(parameter)
                if suffix in broad.FIRST_STEP_ZERO_ALLOWED
                else torch.ones_like(parameter)
            )
            named_trainable.append(
                (f"model.layers.{layer}.self_attn{suffix}", parameter)
            )

    audit = broad.audit_broad_gradients(named_trainable)

    assert audit["audited_parameter_families"] == len(
        broad.TRAINABLE_SUFFIXES
    )
    assert audit["passed"] is True
    assert all(
        family["parameter_tensors"] == common.EXPECTED_LAYERS
        for family in audit["families"].values()
    )


def test_gradient_audit_rejects_disconnected_active_family() -> None:
    named_trainable = []
    for layer in range(common.EXPECTED_LAYERS):
        for suffix in broad.TRAINABLE_SUFFIXES:
            parameter = torch.nn.Parameter(torch.ones(1))
            parameter.grad = torch.ones_like(parameter)
            if layer == 0 and suffix == ".memory_v_proj":
                parameter.grad = None
            named_trainable.append(
                (f"model.layers.{layer}.self_attn{suffix}", parameter)
            )

    audit = broad.audit_broad_gradients(named_trainable)

    assert audit["passed"] is False
    assert audit["families"][".memory_v_proj"]["passed"] is False
