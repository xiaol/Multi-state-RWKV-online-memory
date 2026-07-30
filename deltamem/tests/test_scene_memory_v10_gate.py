from __future__ import annotations

from argparse import Namespace
import copy
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v10_gate as gate


def _v9_result(*, step: int, passed: bool = True) -> dict[str, Any]:
    return {
        "status": "pass" if passed else "fail",
        "all_gates_passed": passed,
        "contract": gate.v9.GATE_CONTRACT,
        "checkpoint_step": step,
        "comparison": {
            "kind": "v8_checkpoint56_frozen_baseline",
            "checkpoint_step": None if step == 7 else step - 7,
            "metrics": dict(gate.V8_CHECKPOINT56_BASELINE),
        },
        "requirements": dict(gate.PROGRESSION_REQUIREMENTS),
        "metrics": {},
        "gates": {"synthetic": passed},
        "training_continuation_authorized": passed and step < 28,
        "next_checkpoint_step": None if step == 28 else step + 7,
        "final_checkpoint_reached": step == 28,
        "final_benchmark_candidate": False,
        "hard32_access": gate.v9.HARD32_ACCESS_POLICY,
        "hard32_authorized": False,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def test_v10_first_gate_maps_optimizer_step_to_v9_metric_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def built(**kwargs):
        calls.append(kwargs)
        return _v9_result(step=kwargs["checkpoint_step"])

    monkeypatch.setattr(gate.v9, "build_v9_gate", built)
    result = gate.build_v10_gate(
        records_by_condition={},
        pairing={},
        checkpoint_step=1,
    )

    assert calls[0]["checkpoint_step"] == 7
    assert calls[0]["previous_gate"] is None
    assert result["contract"] == gate.GATE_CONTRACT
    assert result["checkpoint_step"] == 1
    assert result["consumed_pair_presentations"] == 7
    assert result["next_checkpoint_step"] == 2
    assert result["training_continuation_authorized"] is True
    assert result["hard32_authorized"] is False


def test_v10_later_gate_maps_immediate_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = gate.build_v10_gate
    del previous
    prior = {
        **_v9_result(step=7),
        "contract": gate.GATE_CONTRACT,
        "checkpoint_step": 1,
        "next_checkpoint_step": 2,
    }
    captured: dict[str, Any] = {}

    def built(**kwargs):
        captured.update(kwargs)
        return _v9_result(step=14)

    monkeypatch.setattr(gate.v9, "build_v9_gate", built)
    result = gate.build_v10_gate(
        records_by_condition={},
        pairing={},
        checkpoint_step=2,
        previous_gate=prior,
    )

    assert captured["checkpoint_step"] == 14
    assert captured["previous_gate"]["contract"] == gate.v9.GATE_CONTRACT
    assert captured["previous_gate"]["checkpoint_step"] == 7
    assert captured["previous_gate"]["next_checkpoint_step"] == 14
    assert result["comparison"]["checkpoint_step"] == 1
    assert result["next_checkpoint_step"] == 3
    assert result["consumed_pair_presentations"] == 14


def test_v10_gate_retains_v8_plus_one_requirements() -> None:
    assert gate.V8_CHECKPOINT56_BASELINE == {
        "correct_strict_exact_rows": 3,
        "donor_identity_strict_exact_rows": 3,
        "bidirectional_identity_switch_rows": 6,
        "correct_state_prefers_source_token_rows": 10,
        "donor_state_prefers_donor_token_rows": 10,
        "correct_state_beats_donor_state_on_source_token_rows": 13,
        "correct_state_beats_zero_on_source_token_rows": 11,
    }
    assert gate.PROGRESSION_REQUIREMENTS["correct_strict_exact_rows"] == 4
    assert gate.PROGRESSION_REQUIREMENTS["donor_identity_strict_exact_rows"] == 4
    assert gate.PROGRESSION_REQUIREMENTS["bidirectional_identity_switch_rows"] == 7
    assert gate.V10_OBJECTIVE["selected_full_vocab_ce_in_total"] is False
    assert gate.V10_OBJECTIVE["generated_prefix_correction_mode"] == (
        gate.launch.GENERATED_PREFIX_MODE
    )
    assert gate.V10_OBJECTIVE["cycle_retention_mode"] == (
        gate.launch.CYCLE_RETENTION_MODE
    )
    assert gate.V10_OBJECTIVE["generated_replay_state_gradient"] is True
    assert gate.V10_OBJECTIVE["generated_replay_read_path_gradient"] is True


def test_v10_rejects_non_immediate_previous_gate() -> None:
    prior = {
        **_v9_result(step=7),
        "contract": gate.GATE_CONTRACT,
        "checkpoint_step": 1,
        "next_checkpoint_step": 2,
    }
    with pytest.raises(
        gate.V10EvaluationContractError,
        match="immediate predecessor",
    ):
        gate.build_v10_gate(
            records_by_condition={},
            pairing={},
            checkpoint_step=3,
            previous_gate=prior,
        )


def test_v10_gate_path_rejects_protected_name_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "evaluation" / "gate"
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, *args, **kwargs):
        if "evaluation" in {part.lower() for part in path.parts}:
            raise AssertionError("protected gate path was resolved")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(gate.V10EvaluationContractError, match="protected paths"):
        gate._ssd_path(
            protected,
            description="gate output",
            ssd_root=tmp_path,
        )


def test_v10_gate_source_never_mentions_hard32_locator() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")

    assert "holdout.jsonl" not in source
    assert gate.HARD32_ACCESS_POLICY == "forbidden_not_resolved_opened_or_hashed"


def test_v10_base_model_guard_rejects_protected_path_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "evaluation" / "gemma-4-E4B-it"
    pinned = tmp_path / "models" / "gemma" / "gemma-4-E4B-it"
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, *args, **kwargs):
        if "evaluation" in {part.lower() for part in path.parts}:
            raise AssertionError("protected base-model path was resolved")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(
        gate.V10EvaluationContractError,
        match="exact canonical pinned",
    ):
        gate.validate_base_model_path(
            protected,
            ssd_root=tmp_path,
            pinned_base_model=pinned,
        )


def test_v10_base_model_requires_exact_canonical_pinned_directory(
    tmp_path: Path,
) -> None:
    pinned = tmp_path / "models" / "gemma" / "gemma-4-E4B-it"
    pinned.mkdir(parents=True)
    other = tmp_path / "models" / "gemma" / "other"
    other.mkdir()

    assert gate.validate_base_model_path(
        pinned,
        ssd_root=tmp_path,
        pinned_base_model=pinned,
    ) == pinned.resolve()
    with pytest.raises(gate.V10EvaluationContractError, match="exact canonical"):
        gate.validate_base_model_path(
            other,
            ssd_root=tmp_path,
            pinned_base_model=pinned,
        )
    lexical_alias = pinned.parent / ".." / "gemma" / pinned.name
    with pytest.raises(gate.V10EvaluationContractError, match="exact canonical"):
        gate.validate_base_model_path(
            lexical_alias,
            ssd_root=tmp_path,
            pinned_base_model=pinned,
        )

    alias = tmp_path / "models" / "gemma" / "alias"
    alias.symlink_to(pinned, target_is_directory=True)
    with pytest.raises(gate.V10EvaluationContractError):
        gate.validate_base_model_path(
            alias,
            ssd_root=tmp_path,
            pinned_base_model=pinned,
        )
    assert str(gate.launch.PINNED_BASE_MODEL) == (
        "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"
    )


def test_v10_preflight_validates_base_model_without_hashing_or_loading(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned = gate.launch.PINNED_BASE_MODEL
    called: list[Path | str] = []
    args = Namespace(
        base_model=str(pinned),
        memory_dir=Path("/ssd/checkpoint-1"),
        output_dir=Path("/ssd/gate"),
        previous_gate_receipt=None,
        delta_mem_root=str(gate.PROJECT_ROOT),
        expected_memory_layer_count=42,
        max_new_tokens=gate.v9.DEFAULT_MAX_NEW_TOKENS,
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation="sdpa",
        normal_fusion_profile="native",
        overwrite=False,
        preflight_only=True,
    )
    monkeypatch.setattr(gate, "_parse_args", lambda _argv=None: args)
    monkeypatch.setattr(
        gate,
        "validate_base_model_path",
        lambda path: called.append(path) or pinned,
    )
    monkeypatch.setattr(gate, "_gate_path", lambda path, **_kwargs: Path(path))
    monkeypatch.setattr(gate, "validate_v10_train_inputs", lambda: {"artifacts": {}})
    monkeypatch.setattr(gate.launch, "validate_warm_start_contract", lambda: {})
    monkeypatch.setattr(
        gate,
        "validate_v10_checkpoint",
        lambda *_args, **_kwargs: {"global_step": 1},
    )
    monkeypatch.setattr(gate, "validate_previous_gate_receipt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        gate.v9,
        "base_model_weight_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight hashed base-model weights")
        ),
    )

    assert gate.main([]) == 0
    assert called == [str(pinned)]
    assert '"model_loaded": false' in capsys.readouterr().out


def test_v10_embedded_previous_receipt_binding_is_exact(
    tmp_path: Path,
) -> None:
    receipt_path = gate.launch.v10_gates_root_for(tmp_path) / "previous-receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text('{"receipt":"content"}\n', encoding="utf-8")
    validated = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": "a" * 64,
        "checkpoint": {"memory_dir": "/ssd/checkpoint-1", "global_step": 1},
    }
    embedded = {
        "artifact": gate._artifact_binding(
            receipt_path,
            description="test previous receipt",
        ),
        "receipt_sha256": validated["receipt_sha256"],
        "checkpoint": validated["checkpoint"],
    }
    gate._validate_embedded_previous_gate_receipt_binding(
        embedded,
        validated_previous=validated,
        ssd_root=tmp_path,
    )

    drifts = []
    for path in (
        ("artifact", "bytes"),
        ("artifact", "sha256"),
        ("receipt_sha256",),
        ("checkpoint", "global_step"),
    ):
        drifted = copy.deepcopy(embedded)
        target = drifted
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = 2 if path[-1] in {"bytes", "global_step"} else "b" * 64
        drifts.append(drifted)
    extra = copy.deepcopy(embedded)
    extra["unexpected"] = True
    drifts.append(extra)

    for drifted in drifts:
        with pytest.raises(gate.V10EvaluationContractError, match="validated predecessor"):
            gate._validate_embedded_previous_gate_receipt_binding(
                drifted,
                validated_previous=validated,
                ssd_root=tmp_path,
            )


def test_v10_existing_manifest_rejects_payload_drift_with_same_fingerprint() -> None:
    payload = {
        "training_sources": {"train": {"sha256": "a" * 64}},
        "runtime": {"dtype": "bfloat16", "device": "cuda:0"},
        "code": {"gate": {"sha256": "b" * 64}},
        "hard32_access": gate.HARD32_ACCESS_POLICY,
    }
    fingerprint = gate.v9.fingerprint_payload_sha256(payload)
    manifest = {
        "schema": gate.GATE_MANIFEST_SCHEMA,
        "created_at": "2026-07-30T00:00:00Z",
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "hard32_access": gate.HARD32_ACCESS_POLICY,
    }
    assert gate.validate_existing_manifest(
        manifest,
        expected_fingerprint=fingerprint,
        expected_fingerprint_payload=payload,
    ) == manifest

    for field, value in (
        ("training_sources", {"train": {"sha256": "c" * 64}}),
        ("runtime", {"dtype": "float16", "device": "cuda:0"}),
        ("code", {"gate": {"sha256": "d" * 64}}),
        ("hard32_access", "authorized"),
    ):
        drifted = copy.deepcopy(manifest)
        drifted["fingerprint_payload"][field] = value
        with pytest.raises(gate.V10EvaluationContractError, match="payload differs"):
            gate.validate_existing_manifest(
                drifted,
                expected_fingerprint=fingerprint,
                expected_fingerprint_payload=payload,
            )

    drifted = copy.deepcopy(manifest)
    drifted["hard32_access"] = "authorized"
    with pytest.raises(gate.V10EvaluationContractError, match="Hard32"):
        gate.validate_existing_manifest(
            drifted,
            expected_fingerprint=fingerprint,
            expected_fingerprint_payload=payload,
        )


def test_v10_fingerprint_payload_is_rebuilt_from_all_live_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "weights": {"model": {"sha256": "a" * 64}},
        "prompts": {"tokenizer": {"sha256": "b" * 64}},
        "packages": {"torch": "2.test"},
        "code": {"gate": {"sha256": "c" * 64}},
    }
    monkeypatch.setattr(
        gate,
        "validate_base_model_path",
        lambda _path: gate.launch.PINNED_BASE_MODEL,
    )
    monkeypatch.setattr(gate.v9, "base_model_weight_identity", lambda _path: state["weights"])
    monkeypatch.setattr(gate.v9, "base_model_prompt_identity", lambda _path: state["prompts"])
    monkeypatch.setattr(gate.v9, "runtime_package_versions", lambda: state["packages"])
    monkeypatch.setattr(gate, "evaluator_code_binding", lambda: state["code"])
    inputs = {"artifacts": {"train32": {"sha256": "d" * 64}}}
    checkpoint = {
        "memory_dir": "/v10/checkpoint-1",
        "global_step": 1,
        "architecture": {"target_layers": list(range(42))},
    }

    baseline = gate.build_evaluation_fingerprint_payload(
        input_contract=inputs,
        checkpoint=checkpoint,
    )
    baseline_hash = gate.v9.fingerprint_payload_sha256(baseline)
    assert baseline["training_sources"] == inputs["artifacts"]
    assert baseline["checkpoint"] == checkpoint
    assert baseline["objective"] == gate.V10_OBJECTIVE
    assert baseline["hard32_access"] == gate.HARD32_ACCESS_POLICY
    assert baseline["runtime"] == {
        "conditions": list(gate.CONDITIONS),
        "semantic_selected_token_ordinals": list(gate.VALUE14_ORDINALS),
        "max_new_tokens": gate.GATE_MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache_generation": True,
        "prime_use_cache": False,
        "device": gate.GATE_DEVICE,
        "dtype": gate.GATE_DTYPE,
        "attn_implementation": gate.GATE_ATTN_IMPLEMENTATION,
        "normal_fusion_profile": gate.GATE_NORMAL_FUSION_PROFILE,
        "packages": state["packages"],
    }

    for field, changed in (
        ("weights", {"model": {"sha256": "1" * 64}}),
        ("prompts", {"tokenizer": {"sha256": "2" * 64}}),
        ("packages", {"torch": "different"}),
        ("code", {"gate": {"sha256": "3" * 64}}),
    ):
        original = state[field]
        state[field] = changed
        rebuilt = gate.build_evaluation_fingerprint_payload(
            input_contract=inputs,
            checkpoint=checkpoint,
        )
        assert gate.v9.fingerprint_payload_sha256(rebuilt) != baseline_hash
        state[field] = original

    changed_inputs = copy.deepcopy(inputs)
    changed_inputs["artifacts"]["train32"]["sha256"] = "4" * 64
    changed_checkpoint = copy.deepcopy(checkpoint)
    changed_checkpoint["global_step"] = 2
    for rebuilt in (
        gate.build_evaluation_fingerprint_payload(
            input_contract=changed_inputs,
            checkpoint=checkpoint,
        ),
        gate.build_evaluation_fingerprint_payload(
            input_contract=inputs,
            checkpoint=changed_checkpoint,
        ),
    ):
        assert gate.v9.fingerprint_payload_sha256(rebuilt) != baseline_hash


def test_v10_receipt_validation_reconstructs_fingerprint_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {"global_step": 1}
    inputs = {"artifacts": {}}
    receipt = {
        "schema": gate.GATE_RECEIPT_SCHEMA,
        "contract": gate.GATE_CONTRACT,
        "objective": dict(gate.V10_OBJECTIVE),
        "status": "pass",
        "checkpoint": checkpoint,
        "training_sources": {},
    }
    receipt["receipt_sha256"] = gate.self_hash_payload(
        receipt,
        hash_field="receipt_sha256",
    )
    monkeypatch.setattr(
        gate,
        "validate_v10_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )

    class Rebuilt(RuntimeError):
        pass

    monkeypatch.setattr(
        gate,
        "build_evaluation_fingerprint_payload",
        lambda **_kwargs: (_ for _ in ()).throw(Rebuilt("live identity rebuilt")),
    )
    with pytest.raises(Rebuilt, match="live identity rebuilt"):
        gate.validate_gate_receipt_for_checkpoint(
            receipt,
            memory_dir=Path("/unused"),
            input_contract=inputs,
            warm_contract={},
        )


def test_v10_postload_manifest_resume_validates_runtime_before_reuse() -> None:
    payload = {
        "training_sources": {},
        "runtime": {"dtype": "bfloat16"},
        "code": {},
        "hard32_access": gate.HARD32_ACCESS_POLICY,
    }
    fingerprint = gate.v9.fingerprint_payload_sha256(payload)
    preload = {
        "schema": gate.GATE_MANIFEST_SCHEMA,
        "created_at": "2026-07-30T00:00:00Z",
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "hard32_access": gate.HARD32_ACCESS_POLICY,
    }
    prefixes = {"system_prefix_sha256": "a" * 64}
    profile = {"mode": "native", "layers": 42}
    postload = gate.bind_or_validate_manifest_runtime(
        preload,
        runtime_prefixes=prefixes,
        runtime_fusion_profile=profile,
    )
    assert "runtime_prefixes" not in preload
    assert gate.validate_existing_manifest(
        postload,
        expected_fingerprint=fingerprint,
        expected_fingerprint_payload=payload,
        require_postload=True,
    ) == postload
    with pytest.raises(gate.V10EvaluationContractError, match="post-load"):
        gate.validate_existing_manifest(
            preload,
            expected_fingerprint=fingerprint,
            expected_fingerprint_payload=payload,
            require_postload=True,
        )
    assert gate.bind_or_validate_manifest_runtime(
        postload,
        runtime_prefixes=prefixes,
        runtime_fusion_profile=profile,
    ) == postload

    for changed_prefixes, changed_profile, message in (
        ({"system_prefix_sha256": "b" * 64}, profile, "prefixes differ"),
        (prefixes, {"mode": "other", "layers": 42}, "profile differs"),
    ):
        with pytest.raises(gate.V10EvaluationContractError, match=message):
            gate.bind_or_validate_manifest_runtime(
                postload,
                runtime_prefixes=changed_prefixes,
                runtime_fusion_profile=changed_profile,
            )
        assert postload["runtime_prefixes"] == prefixes
        assert postload["runtime_fusion_profile"] == profile
