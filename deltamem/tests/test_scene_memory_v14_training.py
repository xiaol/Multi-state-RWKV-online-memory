from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch

import deltamem.train.delta_sft_experimental as experimental_train


class _CachedReplayModeModel(torch.nn.Module):
    def __init__(
        self,
        *,
        module_dropout: float = 0.0,
        configured_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout_layer = torch.nn.Dropout(module_dropout)
        self.config = SimpleNamespace(attention_dropout=configured_dropout)


class _CachedReplayModeWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module


def _function_ast(function) -> ast.FunctionDef:
    parsed = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = parsed.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _called_attribute_names(node: ast.AST) -> set[str]:
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        name.id
        for name in ast.walk(node)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
    }


def _nodes_as_module(nodes: list[ast.stmt]) -> ast.Module:
    return ast.Module(body=nodes, type_ignores=[])


def _failed_branch_inputs() -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor | int | bool],
    dict[str, torch.Tensor | int],
    torch.Tensor,
]:
    labels = torch.tensor([[-100, -100, 5, 4, 3, 1]], dtype=torch.long)
    generated = torch.tensor([5, 4, 2], dtype=torch.long)
    failed_alignment: dict[str, torch.Tensor | int] = {
        "generated_cursor": 2,
        "selected_position": torch.tensor(4),
        "selected_decision_ordinal": 1,
        "competitor_id": torch.tensor(2),
        "alignment_kind_code": 0,
        "selected_is_termination": torch.tensor(False),
    }
    replay_logits = torch.full((1, 3, 7), -3.0)
    replay_logits[0, 0, 5] = 5.0
    replay_logits[0, 1, 4] = 5.0
    replay_logits[0, 2, 2] = 2.0
    replay_logits[0, 2, 3] = 0.25
    return (
        {"labels": labels},
        {"generated_token_ids": generated},
        failed_alignment,
        replay_logits.requires_grad_(),
    )


def test_v14_objective_identity_and_four_cycle_contract_are_distinct() -> None:
    objective = experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION

    assert objective == (
        "scene_state_generation_ce_symmetric_cached_prefix_boundary_v14"
    )
    assert (
        experimental_train._SCENE_STATE_CACHED_PREFIX_TRAINING_PROTOCOL_SCHEMA_VERSION
        == 17
    )
    assert objective in experimental_train._SCENE_STATE_CYCLE_OBJECTIVE_VERSIONS
    assert objective in experimental_train._SCENE_STATE_RECIPROCAL_OBJECTIVE_VERSIONS
    assert objective != experimental_train._SCENE_STATE_DENSE_SEMANTIC_OBJECTIVE_VERSION
    assert experimental_train._SCENE_STATE_CACHED_PREFIX_CHECKPOINT_STEPS == (
        1,
        2,
        3,
        4,
    )
    assert experimental_train._SCENE_STATE_CACHED_PREFIX_PRESENTATION_CHECKPOINT_STEPS == (
        7,
        14,
        21,
        28,
    )
    assert (
        experimental_train._SCENE_STATE_V14_FOUR_CYCLE_PAIRS
        == experimental_train._SCENE_STATE_V13_FOUR_CYCLE_PAIRS
    )
    assert experimental_train._V14_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE == (
        "explicit_ordered_v14_four_canonical_seven_pair_cycles_v1"
    )
    formula = experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_FORMULA
    assert formula.startswith("symmetric_pair_mean(if(parsed_boundary_exact,")
    assert "selected_pair_and_zero_hinges=telemetry_only" in formula


def test_v14_exact_cached_retention_uses_only_decision_rows_and_detaches_competitors(
) -> None:
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4, -100]], dtype=torch.long)
    decision_mask = torch.tensor(
        [[False, False, False, True, False, True, False]]
    )
    replay_logits = torch.full((1, 4, 7), -4.0)
    # Non-decision rows are deliberately more adversarial than supervised rows.
    replay_logits[0, 0, 6] = 8.0
    replay_logits[0, 2, 6] = 9.0
    replay_logits[0, 1, 2] = 0.0
    replay_logits[0, 1, 0] = 0.5
    replay_logits[0, 3, 4] = 0.0
    replay_logits[0, 3, 5] = -0.5
    replay_logits.requires_grad_()

    loss, stats = (
        experimental_train.DeltaMemTrainer._scene_state_v14_exact_cached_retention_branch(
            {"labels": labels},
            rollout={
                "gold_token_ids": torch.tensor([1, 2, 3, 4]),
                "generation_start": 2,
            },
            decision_mask=decision_mask,
            cached_replay_logits=replay_logits,
        )
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert stats["scene_generation_v14_cached_replay_token_count"] == 4.0
    assert stats["scene_generation_v14_cached_decision_token_count"] == 2.0
    assert stats["scene_generation_v14_cached_replay_selected_cursor"] == 1.0
    assert stats["scene_generation_v14_cached_selected_label_position"] == 3.0
    assert replay_logits.grad is not None
    assert replay_logits.grad[0, 1, 2].item() == pytest.approx(-0.5)
    assert replay_logits.grad[0, 3, 4].item() == pytest.approx(-0.5)
    assert replay_logits.grad[0, 1, 0].item() == 0.0
    assert replay_logits.grad[0, 3, 5].item() == 0.0
    assert replay_logits.grad[0, [0, 2]].abs().sum().item() == 0.0


def test_v14_failed_cached_repair_selects_cursor_and_binds_actual_greedy_top1(
) -> None:
    model_inputs, rollout, failed_alignment, replay_logits = _failed_branch_inputs()

    loss, stats = (
        experimental_train.DeltaMemTrainer._scene_state_v14_failed_cached_repair_branch(
            model_inputs,
            rollout=rollout,
            failed_alignment=failed_alignment,
            cached_replay_logits=replay_logits,
        )
    )
    loss.backward()

    assert stats["scene_generation_v14_cached_replay_token_count"] == 3.0
    assert stats["scene_generation_v14_cached_replay_selected_cursor"] == 2.0
    assert stats["scene_generation_v14_cached_selected_competitor_id"] == 2.0
    assert stats["scene_generation_v14_cached_replay_top1_matches_actual"] == 1.0
    assert stats["scene_generation_v14_cached_replay_top1_match_count"] == 3.0
    assert replay_logits.grad is not None
    assert replay_logits.grad[0, 2].abs().sum().item() > 0.0
    assert replay_logits.grad[0, :2].abs().sum().item() == 0.0

    with pytest.raises(ValueError, match="tensors do not align"):
        experimental_train.DeltaMemTrainer._scene_state_v14_failed_cached_repair_branch(
            model_inputs,
            rollout=rollout,
            failed_alignment=failed_alignment,
            cached_replay_logits=replay_logits.detach()[:, :2],
        )

    wrong_prefix_top1 = replay_logits.detach().clone()
    wrong_prefix_top1[0, 0, 6] = 8.0
    with pytest.raises(
        RuntimeError,
        match="top-1 differs from the rollout actual prefix: cursor=0",
    ):
        experimental_train.DeltaMemTrainer._scene_state_v14_failed_cached_repair_branch(
            model_inputs,
            rollout=rollout,
            failed_alignment=failed_alignment,
            cached_replay_logits=wrong_prefix_top1,
        )

    wrong_selected_top1 = replay_logits.detach().clone()
    wrong_selected_top1[0, 2, 6] = 8.0
    with pytest.raises(
        RuntimeError,
        match="top-1 differs from the rollout actual prefix: cursor=2",
    ):
        experimental_train.DeltaMemTrainer._scene_state_v14_failed_cached_repair_branch(
            model_inputs,
            rollout=rollout,
            failed_alignment=failed_alignment,
            cached_replay_logits=wrong_selected_top1,
        )


def test_v14_sequential_dispatch_includes_cursor_and_excludes_legacy_repair() -> None:
    function = _function_ast(
        experimental_train.DeltaMemTrainer._scene_state_generation_symmetric_sequential_backward
    )
    cached_dispatch = next(
        branch
        for branch in ast.walk(function)
        if isinstance(branch, ast.If)
        and {
            "_scene_state_v14_cached_replay_logits",
            "_scene_state_v14_exact_cached_retention_branch",
            "_scene_state_v14_failed_cached_repair_branch",
        }
        <= _called_attribute_names(branch)
    )
    cached_calls = _called_attribute_names(cached_dispatch)

    assert "cached_prefix_objective" in _loaded_names(cached_dispatch.test)
    assert "_scene_state_v12_failed_replay_logits" not in cached_calls
    assert "_scene_state_v13_failed_semantic_repair_branch" not in cached_calls

    inclusive_slice = next(
        subscript
        for subscript in ast.walk(cached_dispatch)
        if isinstance(subscript, ast.Subscript)
        and isinstance(subscript.value, ast.Name)
        and subscript.value.id == "rollout"
        and isinstance(subscript.slice, ast.Constant)
        and subscript.slice.value == "generated_token_ids"
    )
    parent_slice = next(
        subscript
        for subscript in ast.walk(cached_dispatch)
        if isinstance(subscript, ast.Subscript)
        and subscript.value is inclusive_slice
        and isinstance(subscript.slice, ast.Slice)
    )
    assert ast.dump(parent_slice.slice.upper, include_attributes=False) == ast.dump(
        ast.BinOp(
            left=ast.Name(id="generated_cursor", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        ),
        include_attributes=False,
    )

    legacy_dispatch = next(
        branch
        for branch in ast.walk(function)
        if isinstance(branch, ast.If)
        and "_scene_state_v12_failed_replay_logits" in _called_attribute_names(branch)
    )
    assert "cached_prefix_objective" in _loaded_names(legacy_dispatch.test)
    assert any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "cached_prefix_objective"
        for node in ast.walk(legacy_dispatch.test)
    )


def test_v14_teacher_telemetry_has_no_gradient_or_backward_path() -> None:
    function = _function_ast(
        experimental_train.DeltaMemTrainer._scene_state_generation_symmetric_sequential_backward
    )
    run_side = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_side"
    )
    teacher_context = next(
        node
        for node in ast.walk(run_side)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "set_grad_enabled"
            for item in node.items
        )
    )
    grad_call = next(
        item.context_expr
        for item in teacher_context.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "set_grad_enabled"
    )
    assert len(grad_call.args) == 1
    assert isinstance(grad_call.args[0], ast.UnaryOp)
    assert isinstance(grad_call.args[0].op, ast.Not)
    assert isinstance(grad_call.args[0].operand, ast.Name)
    assert grad_call.args[0].operand.id == "cached_prefix_objective"

    teacher_backward_dispatch = next(
        node
        for node in ast.walk(run_side)
        if isinstance(node, ast.If)
        and _loaded_names(node.test) == {"cached_prefix_objective"}
        and "backward" in _called_attribute_names(_nodes_as_module(node.orelse))
    )
    assert "backward" not in _called_attribute_names(
        _nodes_as_module(teacher_backward_dispatch.body)
    )
    assert "backward" in _called_attribute_names(
        _nodes_as_module(teacher_backward_dispatch.orelse)
    )

    cached_teacher_branch = next(
        node
        for node in ast.walk(teacher_context)
        if isinstance(node, ast.If)
        and _loaded_names(node.test) == {"cached_prefix_objective"}
        and any(
            isinstance(value, ast.Constant)
            and value.value == "scene_generation_v14_auxiliary_loss"
            for value in ast.walk(node)
        )
    )
    zero_assignments = [
        assignment.value
        for assignment in ast.walk(cached_teacher_branch)
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "scene_generation_v14_auxiliary_loss"
            for target in assignment.targets
        )
    ]
    assert len(zero_assignments) == 1
    assert isinstance(zero_assignments[0], ast.Constant)
    assert zero_assignments[0].value == 0.0


def test_v14_cached_replay_requires_single_logit_projection_in_source() -> None:
    function = _function_ast(
        experimental_train.DeltaMemTrainer._scene_state_v14_cached_replay_logits
    )
    replay_call = next(
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "cached_prefix_replay_logits"
    )
    keyword_values = {keyword.arg: keyword.value for keyword in replay_call.keywords}

    assert isinstance(
        keyword_values["require_single_logit_projection"],
        ast.Constant,
    )
    assert keyword_values["require_single_logit_projection"].value is True
    called_names = _called_attribute_names(function)
    assert "_scene_state_v14_validate_cached_replay_model_mode" in called_names
    assert "eval" not in called_names


def test_v14_cached_replay_model_mode_is_fail_closed() -> None:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    validate = (
        trainer._scene_state_v14_validate_cached_replay_model_mode
    )
    valid_model = _CachedReplayModeModel().train()
    validate(valid_model)

    with pytest.raises(RuntimeError, match="must run in training mode"):
        validate(valid_model.eval())

    with torch.no_grad(), pytest.raises(RuntimeError, match="gradient mode"):
        validate(valid_model.train())

    with pytest.raises(ValueError, match="zero dropout modules"):
        validate(_CachedReplayModeModel(module_dropout=0.1).train())

    wrapped_config_dropout = _CachedReplayModeWrapper(
        _CachedReplayModeModel(configured_dropout=0.1)
    ).train()
    with pytest.raises(ValueError, match="zero configured dropout"):
        validate(wrapped_config_dropout)


def test_v14_training_protocol_binds_cached_replay_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from deltamem.tests.test_scene_memory_v13_artifacts import (
        _v13_protocol_fixture,
    )
    from experiments.rethinking_rwkv_ms_gemma import (
        scene_memory_v13_launch_contract as v13_launch,
    )

    objective = experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
    monkeypatch.setattr(v13_launch, "OBJECTIVE_VERSION", objective)
    protocol, _ = _v13_protocol_fixture(monkeypatch, tmp_path)

    assert protocol["schema_version"] == 17
    assert protocol["memory_objective_version"] == objective
    assert protocol["train_sampler_mode"] == (
        "explicit_ordered_v14_four_canonical_seven_pair_cycles_v1"
    )
    assert protocol["scene_generation_objective_formula"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_FORMULA
    )
    assert protocol["scene_generation_backward_mode"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_BACKWARD_MODE
    )
    assert protocol["scene_generation_generated_prefix_correction_mode"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_MODE
    )
    assert protocol["scene_generation_failed_replay_mode"] == (
        "use_cache_true_logits_to_keep_1_actual_greedy_prefix_v2"
    )
    assert protocol["scene_generation_exact_replay_mode"] == (
        "use_cache_true_logits_to_keep_1_gold_prefix_v1"
    )
    assert protocol["scene_generation_exact_retention_scope"] == (
        "all_boundary_decision_mask_tokens_cached_gold_prefix_v1"
    )
    assert protocol["scene_generation_cached_replay_logits_to_keep"] == 1
    assert protocol["scene_generation_cached_replay_use_cache"] is True
    assert protocol["scene_generation_cached_replay_model_mode"] == (
        "train_grad_enabled_activation_checkpointing_zero_dropout_v1"
    )
    assert protocol["scene_generation_failed_replay_top1_parity_scope"] == (
        "every_actual_greedy_prefix_token_through_selected_cursor_v1"
    )
    assert protocol["scene_generation_cached_prefix_semantic_mode"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_MODE
    )
    assert protocol["scene_generation_cached_decision_token_overlap_policy"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_DECISION_TOKEN_OVERLAP_POLICY
    )
    assert protocol["scene_generation_exact_retention_hinge_weight"] == 1.0
    assert protocol["scene_generation_exact_retention_hinge_mode"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_EXACT_RETENTION_HINGE_MODE
    )
    assert protocol["scene_generation_exact_retention_margin"] == 1.0
    assert protocol["scene_generation_failed_semantic_repair_ce_weight"] == 1.0
    assert protocol["scene_generation_failed_semantic_repair_hinge_weight"] == 1.0
    assert protocol["scene_generation_failed_semantic_repair_margin"] == 1.0
    assert protocol["scene_generation_teacher_forced_full_forward_mode"] == (
        "no_grad_telemetry_only_v1"
    )
    assert protocol["scene_generation_selected_pair_auxiliary_optimization_weight"] == 0.0
    assert protocol["scene_generation_zero_state_auxiliary_optimization_weight"] == 0.0
    assert protocol["scene_generation_row_objective_audit_filename"] == (
        "scene_memory_v14_row_objective.json"
    )
    assert protocol["scene_generation_row_objective_audit_schema"] == (
        "rwkv_ms_scene_memory_v14_row_objective.v1"
    )


def test_v14_one_pair_smoke_flag_is_rejected_outside_cached_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "delta-sft",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path / "output"),
            "--dataset-name",
            "synthetic",
            "--scene-state-v14-one-pair-smoke",
        ],
    )

    with pytest.raises(ValueError, match="requires the V14 cached-prefix objective"):
        experimental_train.parse_args()


def test_v14_one_pair_smoke_warm_start_contract_normalizes_to_v13_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized_args: list[SimpleNamespace] = []
    monkeypatch.setattr(
        experimental_train,
        "_validate_scene_v13_warm_start_args",
        lambda args: normalized_args.append(args),
    )
    args = SimpleNamespace(
        scene_state_v14_one_pair_smoke=True,
        warm_start_mode=experimental_train._SCENE_V14_WARM_START_MODE,
        scene_state_generation_objective_version=(
            experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
        ),
        scene_state_generated_prefix_correction_weight=0.0,
        scene_state_generated_unlikelihood_max_wrong_tokens=0,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        max_steps=1,
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
    )

    experimental_train._validate_scene_v14_warm_start_args(args)

    assert len(normalized_args) == 1
    normalized = normalized_args[0]
    assert normalized.warm_start_mode == experimental_train._SCENE_V13_WARM_START_MODE
    assert normalized.scene_state_generation_objective_version == (
        experimental_train._SCENE_STATE_DENSE_SEMANTIC_OBJECTIVE_VERSION
    )
    assert normalized.gradient_accumulation_steps == 7
    assert normalized.max_steps == 4
    assert normalized.save_total_limit == 4

    args.gradient_accumulation_steps = 7
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        experimental_train._validate_scene_v14_warm_start_args(args)


def _v14_full_schedule_binding() -> dict[str, object]:
    pairs = experimental_train._SCENE_STATE_V14_FOUR_CYCLE_PAIRS
    return {
        "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
        "canonical_value14_pairs": [
            list(pair) for pair in experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS
        ],
        "ordered_pairs_sha256": experimental_train._canonical_json_sha256(
            [list(pair) for pair in pairs]
        ),
        "total_steps": 28,
        "checkpoint_steps": [7, 14, 21, 28],
        "pair_indices": pairs,
        "indices": tuple(source for source, _ in pairs),
    }


def test_v14_one_pair_smoke_schedule_is_sliced_from_verified_full_schedule() -> None:
    full_binding = _v14_full_schedule_binding()
    smoke_binding = experimental_train._scene_state_v14_one_pair_smoke_binding(
        full_binding
    )
    first_pair = experimental_train._SCENE_STATE_V14_FOUR_CYCLE_PAIRS[0]

    assert smoke_binding["source_total_steps"] == 28
    assert smoke_binding["source_checkpoint_steps"] == [7, 14, 21, 28]
    assert smoke_binding["source_ordered_pairs_sha256"] == (
        full_binding["ordered_pairs_sha256"]
    )
    assert smoke_binding["total_steps"] == 1
    assert smoke_binding["checkpoint_steps"] == [1]
    assert smoke_binding["pair_indices"] == (first_pair,)
    assert smoke_binding["indices"] == (first_pair[0],)
    experimental_train._validate_scene_state_v14_one_pair_smoke_schedule(
        smoke_binding
    )

    wrong_source_hash = dict(smoke_binding)
    wrong_source_hash["source_ordered_pairs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="one-pair smoke schedule differs"):
        experimental_train._validate_scene_state_v14_one_pair_smoke_schedule(
            wrong_source_hash
        )

    smoke_binding["pair_indices"] = (experimental_train._SCENE_STATE_V14_FOUR_CYCLE_PAIRS[1],)
    with pytest.raises(ValueError, match="one-pair smoke schedule differs"):
        experimental_train._validate_scene_state_v14_one_pair_smoke_schedule(
            smoke_binding
        )


def test_v14_smoke_and_production_protocol_modes_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from deltamem.tests.test_scene_memory_v13_artifacts import (
        _v13_protocol_fixture,
    )
    from experiments.rethinking_rwkv_ms_gemma import (
        scene_memory_v13_launch_contract as v13_launch,
    )

    objective = experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
    monkeypatch.setattr(v13_launch, "OBJECTIVE_VERSION", objective)
    original_build = experimental_train.build_training_protocol
    captured: dict[str, object] = {}

    def capture_build(args, tokenized, **kwargs):
        captured.update({"args": args, "tokenized": tokenized, "kwargs": kwargs})
        return original_build(args, tokenized, **kwargs)

    monkeypatch.setattr(experimental_train, "build_training_protocol", capture_build)
    production, _ = _v13_protocol_fixture(monkeypatch, tmp_path)

    assert production["scene_generation_v14_run_mode"] == (
        experimental_train._SCENE_STATE_V14_PRODUCTION_RUN_MODE
    )
    assert production["scene_generation_v14_production_eligible"] is True
    assert production["gradient_accumulation_steps"] == 7
    assert production["max_steps"] == 4
    assert production["train_schedule"]["microbatch_cycle_size"] == 7
    assert experimental_train._scene_memory_v10_protocol_checkpoint_steps(
        production
    ) == (1, 2, 3, 4)

    args = captured["args"]
    assert isinstance(args, object)
    args.scene_state_v14_one_pair_smoke = True
    args.gradient_accumulation_steps = 1
    args.max_steps = 1
    args.save_total_limit = 1
    kwargs = dict(captured["kwargs"])
    full_binding = dict(kwargs["train_schedule_binding"])
    full_binding["ordered_pairs_sha256"] = (
        experimental_train._canonical_json_sha256(
            [
                list(pair)
                for pair in experimental_train._SCENE_STATE_V14_FOUR_CYCLE_PAIRS
            ]
        )
    )
    kwargs["train_schedule_binding"] = (
        experimental_train._scene_state_v14_one_pair_smoke_binding(
            full_binding
        )
    )
    smoke = original_build(args, captured["tokenized"], **kwargs)

    assert smoke["scene_generation_v14_run_mode"] == (
        experimental_train._SCENE_STATE_V14_ONE_PAIR_SMOKE_RUN_MODE
    )
    assert smoke["scene_generation_v14_production_eligible"] is False
    assert smoke["train_sampler_mode"] == (
        experimental_train._V14_ONE_PAIR_SMOKE_SAMPLER_MODE
    )
    assert smoke["gradient_accumulation_steps"] == 1
    assert smoke["max_steps"] == 1
    assert smoke["save_total_limit"] == 1
    assert smoke["train_schedule"]["checkpoint_steps"] == [1]
    assert smoke["train_schedule"]["optimizer_checkpoint_steps"] == [1]
    assert smoke["train_schedule"]["microbatch_cycle_size"] == 1
    assert experimental_train._scene_memory_v10_protocol_checkpoint_steps(smoke) == (
        1,
    )

    smoke["scene_generation_v14_production_eligible"] = True
    with pytest.raises(ValueError, match="V14 cached-prefix protocol differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(smoke)


def _v14_exact_audit_stats() -> dict[str, float]:
    stats: dict[str, float] = {}
    for role in ("source", "donor"):
        common = f"scene_generation_{role}_"
        v14 = f"scene_generation_v14_"
        stats.update(
            {
                f"{common}selected_top_hinge": 0.3,
                f"{common}zero_hinge": 0.4,
                f"{v14}parsed_boundary_exact_{role}": 1.0,
                f"{v14}raw_token_exact_{role}": 1.0,
                f"{v14}first_divergence_{role}": 3.0,
                f"{v14}rollout_token_count_{role}": 3.0,
                f"{v14}cached_branch_kind_code_{role}": 0.0,
                f"{v14}cached_replay_use_cache_{role}": 1.0,
                f"{v14}cached_replay_logits_to_keep_{role}": 1.0,
                f"{v14}cached_replay_token_count_{role}": 3.0,
                f"{v14}cached_replay_selected_cursor_{role}": 1.0,
                f"{v14}cached_decision_token_count_{role}": 2.0,
                f"{v14}cached_selected_decision_ordinal_{role}": 0.0,
                f"{v14}cached_selected_label_position_{role}": 3.0,
                f"{v14}cached_selected_gold_token_id_{role}": 5.0,
                f"{v14}cached_selected_competitor_id_{role}": 6.0,
                f"{v14}cached_competitor_is_actual_greedy_{role}": 0.0,
                f"{v14}cached_replay_top1_matches_actual_{role}": 0.0,
                f"{v14}cached_replay_top1_match_count_{role}": 0.0,
                f"{v14}cached_ce_{role}": 0.0,
                f"{v14}cached_failed_competitor_hinge_{role}": 0.0,
                f"{v14}cached_exact_retention_hinge_{role}": 0.2,
                f"{v14}cached_selected_gold_vs_competitor_margin_{role}": 0.5,
                f"{v14}cached_gold_top1_fraction_{role}": 1.0,
                f"{v14}cached_alignment_kind_code_{role}": -1.0,
                f"{v14}cached_selected_is_termination_{role}": 0.0,
                f"{v14}cached_branch_loss_{role}": 0.2,
                f"{v14}auxiliary_loss_{role}": 0.0,
                f"{v14}auxiliary_telemetry_loss_{role}": 0.7,
                f"{v14}total_side_loss_{role}": 0.2,
            }
        )
    stats.update(
        {
            "scene_generation_v14_pair_mean_cached_branch_loss": 0.2,
            "scene_generation_v14_pair_mean_cached_exact_retention_hinge": 0.2,
            "scene_generation_v14_pair_mean_cached_failed_ce": 0.0,
            "scene_generation_v14_pair_mean_cached_failed_competitor_hinge": 0.0,
            "scene_generation_v14_pair_mean_auxiliary_loss": 0.0,
            "scene_generation_v14_pair_mean_selected_top_competitor_hinge": 0.3,
            "scene_generation_v14_pair_mean_selected_correct_vs_zero_hinge": 0.4,
            "scene_generation_v14_pair_mean_total_side_loss": 0.2,
            "scene_generation_v14_objective_total_loss": 0.2,
            "scene_generation_v14_recomputed_objective_total_loss": 0.2,
        }
    )
    return stats


def test_v14_one_pair_smoke_aggregates_and_emits_two_row_audit_observations() -> None:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
    )
    trainer.scene_state_v14_one_pair_smoke = True
    trainer._scene_state_cycle_retention_metric_sums = {}
    trainer._scene_state_cycle_retention_metric_presentations = 0
    trainer._scene_state_v14_cycle_pairs = []
    trainer._scene_state_v14_completed_cycles = 0
    trainer._scene_state_v14_row_observations = []
    trainer._scene_state_v14_pair_observations = []
    source, donor = experimental_train._SCENE_STATE_V14_FOUR_CYCLE_PAIRS[0]
    manifest_pairs: list[dict[str, object]] = [{} for _ in range(32)]
    manifest_pairs[source] = {
        "source_index": source,
        "donor_index": donor,
        "source_row_sha256": bytes([source] * 32).hex(),
        "donor_row_sha256": bytes([donor] * 32).hex(),
    }
    trainer.scene_state_identity_pairing_manifest = {
        "splits": {"train": {"pairs": manifest_pairs}}
    }
    stats = _v14_exact_audit_stats()
    row_hash = lambda ordinal: torch.full(
        (1, 32), ordinal, dtype=torch.uint8
    )

    trainer._scene_state_v14_record_pair_presentation(
        torch.tensor([source]),
        torch.tensor([donor]),
        row_hash(source),
        row_hash(donor),
        stats,
    )
    averaged = trainer._scene_state_cycle_retention_aggregate_memory_stats(stats)
    payload = trainer._scene_state_v14_row_audit_payload()

    assert averaged["scene_generation_v14_cycle_pair_presentations"] == 1.0
    assert trainer._scene_state_v14_completed_cycles == 1
    assert payload["run_mode"] == (
        experimental_train._SCENE_STATE_V14_ONE_PAIR_SMOKE_RUN_MODE
    )
    assert payload["production_eligible"] is False
    assert payload["checkpoint_optimizer_step"] == 1
    assert payload["completed_pair_presentations"] == 1
    assert payload["phases"] == ["smoke_input"]
    assert len(payload["pair_presentations"]) == 1
    assert len(payload["rows"]) == 2
    assert [row["row_ordinal"] for row in payload["rows"]] == [source, donor]

    with pytest.raises(RuntimeError, match="missing its ordered pair presentation"):
        trainer._scene_state_cycle_retention_aggregate_memory_stats(stats)


def test_v14_smoke_runtime_uses_one_pair_accumulation() -> None:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
    )
    trainer.scene_state_v14_one_pair_smoke = True
    trainer.current_gradient_accumulation_steps = 1
    trainer.episode_read_write_enabled = False
    trainer.memory_kl_weight = 0.0
    trainer.memory_base_kl_weight = 0.0
    trainer.args = SimpleNamespace(ignore_data_skip=False, optim="adamw_torch")
    trainer.accelerator = SimpleNamespace(
        gradient_accumulation_steps=1,
        distributed_type=None,
    )

    trainer._validate_scene_state_generation_sequential_runtime()

    trainer.scene_state_v14_one_pair_smoke = False
    with pytest.raises(ValueError, match="gradient_accumulation_steps=7"):
        trainer._validate_scene_state_generation_sequential_runtime()
    trainer.current_gradient_accumulation_steps = 7
    trainer._validate_scene_state_generation_sequential_runtime()


@pytest.mark.parametrize(
    ("objective", "expected_v13", "expected_v14"),
    [
        (
            experimental_train._SCENE_STATE_DENSE_SEMANTIC_OBJECTIVE_VERSION,
            True,
            False,
        ),
        (
            experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION,
            False,
            True,
        ),
    ],
)
def test_v13_v14_checkpoint_reload_requires_versioned_row_audit(
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
    expected_v13: bool,
    expected_v14: bool,
) -> None:
    captured: dict[str, object] = {}

    class StopValidation(Exception):
        pass

    def capture_validation(checkpoint: Path, **kwargs):
        captured.update(kwargs)
        raise StopValidation

    monkeypatch.setattr(
        experimental_train,
        "_validate_resume_checkpoint",
        capture_validation,
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.resume_mode = "exact"
    trainer.training_protocol = {}
    trainer.content_contrast_pairing_manifest = None
    trainer.scene_state_identity_pairing_manifest = None
    trainer.scene_state_generation_objective_version = objective

    with pytest.raises(StopValidation):
        trainer._load_from_checkpoint("checkpoint-1")

    assert captured["require_scene_state_v13_audit"] is expected_v13
    assert captured["require_scene_state_v14_audit"] is expected_v14
    expected_filename = (
        experimental_train._SCENE_STATE_V13_ROW_AUDIT_FILENAME
        if expected_v13
        else experimental_train._SCENE_STATE_V14_ROW_AUDIT_FILENAME
    )
    missing = experimental_train._missing_resume_checkpoint_files(
        Path("checkpoint-1"),
        require_scene_state_v13_audit=expected_v13,
        require_scene_state_v14_audit=expected_v14,
    )
    assert expected_filename in missing
