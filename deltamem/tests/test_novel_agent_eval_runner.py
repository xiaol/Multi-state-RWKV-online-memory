from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import run_novel_agent_eval as evaluator
from experiments.rethinking_rwkv_ms_gemma import diagnose_residual_hybrid_scales as diagnostic
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v6_run_audit as audit


class FakeModule:
    def __init__(self, layer_idx: int, raw_gain: float) -> None:
        self.layer_idx = layer_idx
        self.memory_fusion_mode = "content_gated_add"
        self.memory_fusion_placement = "post_attention_residual_hybrid"
        self.memory_fusion_residual_scale = 0.01
        self.memory_fusion_residual_scale_max = 0.02
        self.memory_fusion_residual_gain_raw = torch.nn.Parameter(
            torch.tensor([raw_gain], dtype=torch.float32)
        )
        self._post_attention_norm_hook_handle = object()

    def set_memory_fusion_residual_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= self.memory_fusion_residual_scale_max:
            raise ValueError("gain outside cap")
        with torch.no_grad():
            self.memory_fusion_residual_gain_raw.fill_(gain)

    def _resolved_memory_fusion_residual_gain(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.memory_fusion_residual_gain_raw.detach().clamp(
            0.0, self.memory_fusion_residual_scale_max
        ).to(device=device, dtype=dtype)[0]


def make_modules() -> list[tuple[str, FakeModule]]:
    return [
        ("model.layers.0.self_attn", FakeModule(0, -0.002)),
        ("model.layers.1.self_attn", FakeModule(1, 0.018)),
    ]


@pytest.mark.parametrize(
    ("profile", "mode", "placement", "gain"),
    [
        (
            "native",
            "content_gated_add",
            "post_attention_residual_hybrid",
            None,
        ),
        (
            "native_gate_open",
            "add",
            "post_attention_residual_hybrid",
            None,
        ),
        ("gate_open_gamma_0", "add", "post_attention_residual_hybrid", 0.0),
        (
            "gate_open_gamma_0p01",
            "add",
            "post_attention_residual_hybrid",
            0.01,
        ),
        (
            "post_attention_norm_gate_open_0p01",
            "add",
            "post_attention_norm",
            None,
        ),
    ],
)
def test_apply_normal_fusion_profiles(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    mode: str,
    placement: str,
    gain: float | None,
) -> None:
    modules = make_modules()
    native_raw = [module.memory_fusion_residual_gain_raw.item() for _, module in modules]
    monkeypatch.setattr(evaluator, "iter_delta_mem_modules", lambda model: modules)

    runtime = evaluator.apply_normal_fusion_profile(
        object(),
        profile_name=profile,
        expected_layer_count=2,
    )

    assert runtime["profile"] == profile
    assert runtime["layer_count"] == 2
    assert runtime["layer_indices"] == [0, 1]
    assert len(runtime["effective_settings_sha256"]) == 64
    assert all(module.memory_fusion_mode == mode for _, module in modules)
    assert all(module.memory_fusion_placement == placement for _, module in modules)
    if gain is None:
        assert [
            module.memory_fusion_residual_gain_raw.item() for _, module in modules
        ] == pytest.approx(native_raw)
    else:
        assert all(
            module.memory_fusion_residual_gain_raw.item() == pytest.approx(gain)
            for _, module in modules
        )
    if placement == "post_attention_norm":
        assert all(
            setting["memory_fusion_residual_gain_effective"] is None
            for setting in runtime["effective_settings"]
        )


def test_apply_profile_rejects_missing_norm_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = make_modules()
    modules[1][1]._post_attention_norm_hook_handle = None
    monkeypatch.setattr(evaluator, "iter_delta_mem_modules", lambda model: modules)

    with pytest.raises(ValueError, match="requires existing Gemma"):
        evaluator.apply_normal_fusion_profile(
            object(),
            profile_name="native_gate_open",
            expected_layer_count=2,
        )


def test_native_profile_accepts_legacy_attention_output_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = make_modules()
    for _, module in modules:
        module.memory_fusion_placement = "attention_output"
        module._post_attention_norm_hook_handle = None
    monkeypatch.setattr(evaluator, "iter_delta_mem_modules", lambda model: modules)

    runtime = evaluator.apply_normal_fusion_profile(
        object(),
        profile_name="native",
        expected_layer_count=2,
    )

    assert runtime["profile"] == "native"
    assert [
        setting["memory_fusion_placement"]
        for setting in runtime["effective_settings"]
    ] == ["attention_output", "attention_output"]
    assert all(
        not setting["post_attention_norm_hook_bound"]
        for setting in runtime["effective_settings"]
    )


def test_profile_fingerprint_changes_with_profile() -> None:
    native = evaluator.normal_fusion_fingerprint_fields("native", 42)
    gate_open = evaluator.normal_fusion_fingerprint_fields(
        "native_gate_open", 42
    )

    assert native["normal_fusion_profile"] == "native"
    assert gate_open["normal_fusion_profile"] == "native_gate_open"
    assert native["profile_definition_sha256"] != gate_open[
        "profile_definition_sha256"
    ]


def test_benchmark_profiles_match_the_causal_screen() -> None:
    assert evaluator.NORMAL_FUSION_PROFILES == diagnostic.SUPPORTED_CONDITIONS
    for name in evaluator.NORMAL_FUSION_PROFILES:
        assert evaluator.normal_fusion_profile_definition(name) == (
            diagnostic.initial_condition_screen([name])[0]
        )


def test_normal_model_loads_checkpoint_before_applying_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model = object()
    tokenizer = object()

    def fake_load(**kwargs):
        events.append("load")
        assert kwargs["memory_dir"] == "/checkpoint"
        return model, tokenizer

    def fake_apply(active_model, *, profile_name: str, expected_layer_count: int):
        events.append("apply")
        assert active_model is model
        assert profile_name == "native_gate_open"
        assert expected_layer_count == 42
        return {"profile": profile_name}

    monkeypatch.setattr(evaluator, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(evaluator, "apply_normal_fusion_profile", fake_apply)
    args = SimpleNamespace(
        base_model="/base",
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        delta_mem_root="/repo",
        memory_dir="/checkpoint",
        normal_fusion_profile="native_gate_open",
        expected_memory_layer_count=42,
    )

    loaded_model, loaded_tokenizer, runtime = evaluator.load_normal_model(args)

    assert events == ["load", "apply"]
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert runtime == {"profile": "native_gate_open"}


def test_selected_conditions_accepts_no_write_and_rejects_duplicates() -> None:
    assert evaluator.selected_conditions("base,normal,no_write") == [
        "base",
        "normal",
        "no_write",
    ]
    with pytest.raises(ValueError, match="must not contain duplicates"):
        evaluator.selected_conditions("normal,normal")


def make_scene_v6_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    run_root = tmp_path / "scene_memory_v6_stage1_run1"
    checkpoint = run_root / "trainer" / "checkpoint-128"
    initial_dir = run_root / "initial_adapter"
    checkpoint.mkdir(parents=True)
    initial_dir.mkdir()

    protocol = {"protocol": "scene-v6-stage1"}
    protocol_path = checkpoint / "training_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_TRAINING_PROTOCOL_FILE_SHA256",
        evaluator.sha256_file(protocol_path),
    )
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_TRAINING_PROTOCOL_CANONICAL_SHA256",
        evaluator.canonical_object_sha256(protocol),
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 128}),
        encoding="utf-8",
    )

    source_lock_path = (
        evaluator.SCRIPT_DIR / "scene_memory_v6_source_lock.json"
    )
    launch = {
        "schema": "rwkv_ms_scene_memory_v6_launch.v1",
        "experiment": "scene_memory_v6",
        "run_mode": "stage1",
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "warm_start_from_checkpoint": None,
        "paths": {"output_dir": str(run_root)},
        "stage": {
            "checkpoint_steps": [128, 256, 384, 512],
            "max_steps": 512,
        },
        "topology": {
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 4,
            "rwkv_ms_semantics_version": 2,
            "memory_backend": "rwkv_ms",
        },
        "data_contract": {
            "training_partition": {
                "source_split": "train",
                "rows": 1804,
                "sha256": evaluator.SCENE_V6_TRAIN_SHA256,
                "val_or_test_rows_emitted_for_training": 0,
            }
        },
        "sampling_audit": {
            "seed": 42,
            "data_seed": 42,
            "stage1_updates": 512,
            "sample_without_replacement": True,
            "extension_beyond_update_512_allowed": False,
        },
        "source_lock": {
            "file_sha256": evaluator.sha256_file(source_lock_path),
            "payload": json.loads(source_lock_path.read_text(encoding="utf-8")),
        },
    }
    launch["manifest_sha256"] = evaluator.canonical_object_sha256(launch)
    launch_path = run_root / "launch_manifest.json"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    initial_manifest = {
        "schema": "deltamem.seeded_initial_adapter.v1",
        "fresh_run": True,
        "global_step": 0,
        "training_started": False,
        "launch_manifest": {
            "path": str(launch_path),
            "sha256": evaluator.sha256_file(launch_path),
        },
    }
    initial_manifest["manifest_sha256"] = evaluator.canonical_object_sha256(
        initial_manifest
    )
    initial_manifest_path = initial_dir / "initial_adapter_manifest.json"
    initial_manifest_path.write_text(json.dumps(initial_manifest), encoding="utf-8")
    (run_root / "training_summary.json").write_text(
        json.dumps(
            {
                "resume_from_checkpoint": None,
                "warm_start_from_checkpoint": None,
                "initial_adapter_output_dir": str(initial_dir),
                "initial_adapter_manifest_sha256": initial_manifest[
                    "manifest_sha256"
                ],
                "train_samples": 1804,
                "seed": 42,
                "data_seed": 42,
                "train_sampler_seed": 42,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_scene_v6_training_lineage_accepts_locked_stage1_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = make_scene_v6_lineage(tmp_path, monkeypatch)
    lineage = evaluator.scene_v6_training_lineage(checkpoint)
    assert lineage["checkpoint_step"] == 128
    assert lineage["training_protocol_file_sha256"] == (
        evaluator.SCENE_V6_TRAINING_PROTOCOL_FILE_SHA256
    )

    launch_path = checkpoint.parent.parent / "launch_manifest.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["fresh_run"] = False
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        evaluator.scene_v6_training_lineage(checkpoint)


def test_scene_v6_training_lineage_rejects_old_unproven_pilot(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "old_scene_pilot" / "trainer" / "checkpoint-128"
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError, match="provenance file is missing"):
        evaluator.scene_v6_training_lineage(checkpoint)


@pytest.mark.parametrize(
    ("stale_section", "stale_field", "stale_value"),
    [
        (
            "objective",
            "backward_mode",
            "sequential_replayed_donor_zero_diagnostic_exact_first_order_v2",
        ),
        ("objective", "zero_margin_weight", 1.0),
        ("history", "finite", True),
        (
            "receipt",
            "adapter_change_from_step_zero",
            {"changed_tensor_count": 1},
        ),
    ],
)
def test_scene_v6_identity_lineage_accepts_current_receipt_and_rejects_stale_contract_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_section: str,
    stale_field: str,
    stale_value: object,
) -> None:
    run_root = tmp_path / "identity_proof"
    checkpoint = run_root / "trainer" / "checkpoint-32"
    checkpoint.mkdir(parents=True)

    def write_file(path: Path, content: bytes = b"artifact") -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": str(path.resolve()),
            "file_sha256": evaluator.sha256_file(path),
        }

    launch = write_file(run_root / "launch_manifest.json", b"{}")
    data_contract = write_file(run_root / "data_contract_manifest.json", b"{}")
    source_lock = write_file(run_root / "source_lock.json", b"{}")
    pair_manifest = write_file(run_root / "pair_manifest.json", b"{}")
    train_partition = write_file(run_root / "train.jsonl", b"train\n")
    indices = write_file(run_root / "holdout_source_indices.json", b"indices\n")
    holdout = write_file(run_root / "holdout.jsonl", b"holdout\n")
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_IDENTITY_PAIR_MANIFEST_SHA256",
        pair_manifest["file_sha256"],
    )
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_IDENTITY_TRAIN_SHA256",
        train_partition["file_sha256"],
    )
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_IDENTITY_HARD32_SELECTION_SHA256",
        indices["file_sha256"],
    )
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_IDENTITY_HARD32_HOLDOUT_SHA256",
        holdout["file_sha256"],
    )
    artifacts = {
        "adapter": write_file(checkpoint / "delta_mem_adapter.pt"),
        "config": write_file(checkpoint / "delta_mem_config.json", b"{}"),
        "protocol": write_file(checkpoint / "training_protocol.json", b"{}"),
        "trainer_state": write_file(
            checkpoint / "trainer_state.json", b'{"global_step":32}'
        ),
        "optimizer": write_file(checkpoint / "optimizer.pt"),
        "scheduler": write_file(checkpoint / "scheduler.pt"),
        "rng": [write_file(checkpoint / "rng_state.pth")],
    }
    receipt = {
        "schema": evaluator.SCENE_V6_IDENTITY_CHECKPOINT_RECEIPT_SCHEMA,
        "experiment": "scene_memory_v6_identity_proof",
        "run_mode": "proof",
        "checkpoint_step": 32,
        "complete": True,
        "issued_at": "2026-07-28T00:00:00Z",
        "run_root": str(run_root.resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "launch": launch,
        "data_contract": data_contract,
        "source_lock": source_lock,
        "pair_manifest": pair_manifest,
        "objective": dict(audit.OBJECTIVE_PROTOCOL),
        "train_partition": {
            **train_partition,
            "rows": 32,
            "source_split": "train",
        },
        "hard32_selection": {
            "indices": indices,
            "holdout": holdout,
            "rows": 32,
            "source_split": "val",
            "test_rows": 0,
        },
        "trainer_state": {"global_step": 32},
        "checkpoint_artifacts": artifacts,
        "history": {
            "records": 32,
            "first_step": 1,
            "last_step": 32,
            "identity_metrics_finite": True,
        },
        "adapter_change": {"changed_tensor_count": 1},
    }
    receipt["receipt_sha256"] = evaluator.canonical_object_sha256(receipt)
    (checkpoint / "checkpoint_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    lineage = evaluator.scene_v6_training_lineage(checkpoint)

    assert lineage["lineage_kind"] == "identity_checkpoint_receipt"
    assert lineage["checkpoint_step"] == 32
    assert lineage["objective"]["objective_version"] == (
        "scene_state_identity_ce_v2"
    )
    assert lineage["objective"] == dict(audit.OBJECTIVE_PROTOCOL)
    assert lineage["objective"]["zero_diagnostic_weight"] == 0.0
    assert not (run_root / "training_summary.json").exists()

    if stale_section == "objective":
        receipt["objective"][stale_field] = stale_value
    elif stale_section == "history":
        receipt["history"][stale_field] = stale_value
    else:
        receipt[stale_field] = stale_value
    receipt["receipt_sha256"] = evaluator.canonical_object_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    (checkpoint / "checkpoint_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        evaluator.scene_v6_training_lineage(checkpoint)


def test_scene_v6_identity_objective_matches_audited_protocol() -> None:
    assert evaluator.SCENE_V6_IDENTITY_OBJECTIVE_EXPECTED == dict(
        audit.OBJECTIVE_PROTOCOL
    )


def _write_scene_v6_hard32_pass_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict, dict]:
    def bind_file(path: Path, content: bytes) -> dict[str, str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": str(path.resolve()),
            "sha256": evaluator.sha256_file(path),
        }

    memory_dir = tmp_path / "checkpoint-32"
    memory_dir.mkdir()
    (memory_dir / "delta_mem_adapter.pt").write_bytes(b"adapter")
    (memory_dir / "delta_mem_config.json").write_bytes(b"config")
    selection = bind_file(tmp_path / "holdout_source_indices.json", b"selection")
    monkeypatch.setattr(
        evaluator,
        "SCENE_V6_IDENTITY_HARD32_SELECTION_SHA256",
        selection["sha256"],
    )
    output_dir = tmp_path / "hard32"
    conditions = {
        condition: bind_file(
            output_dir / f"{condition}.jsonl",
            f"{condition}\n".encode("utf-8"),
        )
        for condition in (
            "base_full",
            "normal_full",
            "no_write_full",
            "state_only",
            "state_only_donor",
            "state_only_no_write",
        )
    }
    candidate_lineage = {
        "lineage_kind": "identity_checkpoint_receipt",
        "checkpoint_step": 32,
    }
    receipt = {
        "schema": evaluator.SCENE_V6_IDENTITY_HARD32_RECEIPT_SCHEMA,
        "status": "pass",
        "evaluation_fingerprint": "f" * 64,
        "contract": {
            "name": "scene_v6_identity_hard32",
            "rows": 32,
        },
        "checkpoint": {
            "memory_dir": str(memory_dir.resolve()),
            "adapter_sha256": evaluator.sha256_file(
                memory_dir / "delta_mem_adapter.pt"
            ),
            "config_sha256": evaluator.sha256_file(
                memory_dir / "delta_mem_config.json"
            ),
            "candidate_lineage": candidate_lineage,
        },
        "selection": {
            **selection,
            "holdout_sha256": (
                evaluator.SCENE_V6_IDENTITY_HARD32_HOLDOUT_SHA256
            ),
            "pair_manifest_sha256": (
                evaluator.SCENE_V6_IDENTITY_PAIR_MANIFEST_SHA256
            ),
        },
        "outputs": {
            "manifest": bind_file(output_dir / "manifest.json", b"manifest"),
            "summary": bind_file(output_dir / "summary.json", b"summary"),
            "conditions": conditions,
        },
        "gate": {
            "status": "pass",
            "all_gates_passed": True,
            "full170_authorized_for_bound_checkpoint": True,
        },
    }
    receipt["receipt_sha256"] = evaluator.canonical_object_sha256(receipt)
    receipt_path = output_dir / "hard32_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, memory_dir, candidate_lineage, receipt


def test_scene_v6_hard32_receipt_accepts_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, candidate_lineage, receipt = (
        _write_scene_v6_hard32_pass_receipt(tmp_path, monkeypatch)
    )

    authorization = evaluator.validate_scene_v6_hard32_pass_receipt(
        receipt_path,
        memory_dir=memory_dir,
        candidate_lineage=candidate_lineage,
    )

    assert receipt["schema"] == "scene_v6_identity_hard32_receipt.v2"
    assert authorization["payload_sha256"] == receipt["receipt_sha256"]
    assert authorization["evaluation_fingerprint"] == "f" * 64


def test_scene_v6_hard32_receipt_rejects_correctly_rehashed_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, candidate_lineage, receipt = (
        _write_scene_v6_hard32_pass_receipt(tmp_path, monkeypatch)
    )
    stale_receipt = {
        **receipt,
        "schema": "scene_v6_identity_hard32_receipt.v1",
    }
    stale_unsigned = {
        key: value
        for key, value in stale_receipt.items()
        if key != "receipt_sha256"
    }
    stale_receipt["receipt_sha256"] = evaluator.canonical_object_sha256(
        stale_unsigned
    )
    receipt_path.write_text(json.dumps(stale_receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a passed hard32 receipt"):
        evaluator.validate_scene_v6_hard32_pass_receipt(
            receipt_path,
            memory_dir=memory_dir,
            candidate_lineage=candidate_lineage,
        )


def test_scene_v6_lineage_cannot_bypass_contract_with_generic_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = make_scene_v6_lineage(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot use the generic evaluation contract"):
        evaluator.validate_contract_lineage_mode(
            contract="generic",
            memory_dir=checkpoint,
        )

    copied_checkpoint = tmp_path / "copied_checkpoint"
    copied_checkpoint.mkdir()
    (copied_checkpoint / "training_protocol.json").write_bytes(
        (checkpoint / "training_protocol.json").read_bytes()
    )
    with pytest.raises(ValueError, match="cannot use the generic evaluation contract"):
        evaluator.validate_contract_lineage_mode(
            contract="generic",
            memory_dir=copied_checkpoint,
        )

    unrelated = tmp_path / "unrelated_all42_qo" / "trainer" / "checkpoint-128"
    unrelated.mkdir(parents=True)
    evaluator.validate_contract_lineage_mode(
        contract="generic",
        memory_dir=unrelated,
    )


def test_read_records_repairs_only_a_partial_final_line(tmp_path: Path) -> None:
    path = tmp_path / "normal.jsonl"
    valid = {"key": "scene-v4-current:0"}
    path.write_bytes(json.dumps(valid).encode("utf-8") + b"\n{\"partial\"")

    assert evaluator.read_records(path) == [valid]
    assert path.read_bytes() == json.dumps(valid).encode("utf-8") + b"\n"


def test_read_records_terminates_a_valid_final_record(tmp_path: Path) -> None:
    path = tmp_path / "normal.jsonl"
    valid = {"key": "scene-v4-current:0"}
    path.write_bytes(json.dumps(valid).encode("utf-8"))

    assert evaluator.read_records(path) == [valid]
    assert path.read_bytes() == json.dumps(valid).encode("utf-8") + b"\n"


def test_protected_outputs_require_their_locked_manifest(tmp_path: Path) -> None:
    records_path = tmp_path / "normal.jsonl"
    records_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without their locked manifest"):
        evaluator.validate_protected_output_manifest_presence(
            contract_name="scene_v6_validation",
            manifest_path=tmp_path / "manifest.json",
            output_paths=[records_path],
        )


def test_resume_record_contract_binds_fingerprint_and_recomputed_score() -> None:
    spec = next(
        spec for spec in evaluator.TASK_SPECS if spec.name == "scene-v4-current"
    )
    sample = {
        "line_index": 0,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [1]},
    }
    parsed = {"boundaries": [1]}
    record = {
        "status": "ok",
        "fingerprint": "f" * 64,
        "key": "scene-v4-current:0",
        "condition": "normal",
        "normal_fusion_profile": "native",
        "task": "scene-v4-current",
        "task_kind": "scene",
        "split": "val",
        "line_index": 0,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [1]},
        "max_new_tokens": 128,
        "raw_generation": json.dumps(parsed),
        "parsed_json": parsed,
        "score": evaluator.score_prediction("scene", parsed, sample["gold"]),
        "input_tokens": 64,
        "output_tokens": 2,
        "hit_max_new_tokens": False,
        "elapsed_seconds": 0.25,
    }

    evaluator.validate_resume_record_contract(
        record,
        condition="normal",
        spec=spec,
        sample=sample,
        split="val",
        fingerprint="f" * 64,
        normal_fusion_profile="native",
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        evaluator.validate_resume_record_contract(
            {**record, "fingerprint": "0" * 64},
            condition="normal",
            spec=spec,
            sample=sample,
            split="val",
            fingerprint="f" * 64,
            normal_fusion_profile="native",
        )

    invalid_diagnostics = (
        ({"input_tokens": True}, "input_tokens is invalid"),
        ({"input_tokens": 0}, "input_tokens is invalid"),
        ({"output_tokens": -1}, "output_tokens is invalid"),
        ({"output_tokens": 129}, "output_tokens is invalid"),
        ({"output_tokens": 128}, "hit_max_new_tokens is inconsistent"),
        ({"hit_max_new_tokens": "false"}, "hit_max_new_tokens is inconsistent"),
        ({"elapsed_seconds": -0.1}, "elapsed_seconds is invalid"),
        ({"elapsed_seconds": float("nan")}, "elapsed_seconds is invalid"),
    )
    for mutation, message in invalid_diagnostics:
        with pytest.raises(ValueError, match=message):
            evaluator.validate_resume_record_contract(
                {**record, **mutation},
                condition="normal",
                spec=spec,
                sample=sample,
                split="val",
                fingerprint="f" * 64,
                normal_fusion_profile="native",
            )


def test_existing_manifest_binds_fingerprinted_references() -> None:
    references = {"source_hashes": {"reference.json": "a" * 64}}
    payload = {
        "evaluation_contract": {"name": "scene_v6_validation"},
        "references": references,
    }
    fingerprint = evaluator.fingerprint_payload_sha256(payload)
    manifest = {
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "references": references,
    }

    assert evaluator.validate_existing_manifest(
        manifest,
        expected_fingerprint=fingerprint,
    ) is manifest
    with pytest.raises(ValueError, match="references differ"):
        evaluator.validate_existing_manifest(
            {**manifest, "references": {}},
            expected_fingerprint=fingerprint,
        )


def test_no_write_condition_freezes_writes_around_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    model = object()

    @contextmanager
    def fake_memory_condition(active_model, condition: str):
        events.append(("enter", (active_model, condition)))
        yield
        events.append(("exit", (active_model, condition)))

    def fake_generate_one(**kwargs):
        events.append(("generate", kwargs["model"]))
        return {"raw_generation": "{}"}

    monkeypatch.setattr(evaluator, "memory_condition", fake_memory_condition)
    monkeypatch.setattr(evaluator, "generate_one", fake_generate_one)

    result = evaluator.generate_for_condition(
        "no_write",
        model=model,
        tokenizer=object(),
        messages=[{"role": "user", "content": "prompt"}],
        max_new_tokens=8,
        device="cpu",
    )

    assert result == {"raw_generation": "{}"}
    assert events == [
        ("enter", (model, "no_write")),
        ("generate", model),
        ("exit", (model, "no_write")),
    ]


def test_scene_v6_evaluation_contracts_require_full_official_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = next(spec for spec in evaluator.TASK_SPECS if spec.name == "scene-v4-current")
    val_rows = [object()] * 170
    test_rows = [object()] * 149
    val_path = SimpleNamespace(name="val.jsonl")
    test_path = SimpleNamespace(name="test.jsonl")
    monkeypatch.setattr(
        evaluator,
        "sha256_file",
        lambda path: evaluator.OFFICIAL_SCENE_V4_SHA256[path.name.removesuffix(".jsonl")],
    )

    validation = evaluator.validate_evaluation_contract(
        contract="scene_v6_validation",
        split="val",
        specs=[spec],
        conditions=["base", "normal", "no_write"],
        task_data={spec.name: (val_path, val_rows)},
        limit_per_task=None,
        overwrite=False,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
        memory_target_layers=list(range(42)),
        memory_delta_heads=["q", "o"],
        memory_rank=4,
        rwkv_ms_semantics_version=2,
        memory_backend="rwkv_ms",
        hard32_receipt_authorization={"status": "pass"},
    )
    with pytest.raises(ValueError, match="validation-selection receipt"):
        evaluator.validate_evaluation_contract(
            contract="scene_v6_final_test",
            split="test",
            specs=[spec],
            conditions=["base", "normal", "no_write"],
            task_data={spec.name: (test_path, test_rows)},
            limit_per_task=None,
            overwrite=False,
            normal_fusion_profile="native",
            expected_memory_layer_count=42,
            memory_target_layers=list(range(42)),
            memory_delta_heads=["q", "o"],
            memory_rank=4,
            rwkv_ms_semantics_version=2,
            memory_backend="rwkv_ms",
        )

    assert validation["phase"] == "validation_selection"
    assert validation["rows"] == 170


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"split": "test"}, "requires split=val"),
        ({"conditions": ["base", "normal"]}, "exact order"),
        ({"limit_per_task": 170}, "forbids --limit-per-task"),
        ({"rows": 169}, "exactly 170 official rows"),
        ({"normal_fusion_profile": "native_gate_open"}, "requires --normal-fusion-profile native"),
        ({"expected_memory_layer_count": 41}, "requires --expected-memory-layer-count 42"),
        ({"memory_target_layers": list(range(41))}, "requires checkpoint target_layers=0..41"),
        ({"memory_delta_heads": ["o"]}, "requires checkpoint delta_heads=q,o"),
        ({"memory_rank": 8}, "requires checkpoint rank=4"),
        ({"rwkv_ms_semantics_version": 1}, "requires checkpoint rwkv_ms_semantics_version=2"),
        ({"memory_backend": "delta"}, "requires checkpoint memory_backend=rwkv_ms"),
    ],
)
def test_scene_v6_validation_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message,
) -> None:
    spec = next(spec for spec in evaluator.TASK_SPECS if spec.name == "scene-v4-current")
    values = {
        "contract": "scene_v6_validation",
        "split": "val",
        "specs": [spec],
        "conditions": ["base", "normal", "no_write"],
        "limit_per_task": None,
        "overwrite": False,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(range(42)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "hard32_receipt_authorization": {"status": "pass"},
        "rows": 170,
    }
    values.update(mutation)
    rows = [object()] * values.pop("rows")

    monkeypatch.setattr(
        evaluator,
        "sha256_file",
        lambda path: evaluator.OFFICIAL_SCENE_V4_SHA256["val"],
    )
    with pytest.raises(ValueError, match=message):
        evaluator.validate_evaluation_contract(
            task_data={spec.name: (SimpleNamespace(), rows)},
            **values,
        )


def test_scene_v6_final_test_is_unavailable_without_receipts() -> None:
    spec = next(spec for spec in evaluator.TASK_SPECS if spec.name == "scene-v4-current")

    with pytest.raises(ValueError, match="validation-selection receipt"):
        evaluator.validate_evaluation_contract(
            contract="scene_v6_final_test",
            split="test",
            specs=[spec],
            conditions=["base", "normal", "no_write"],
            task_data={spec.name: (SimpleNamespace(), [object()] * 149)},
            limit_per_task=None,
            overwrite=False,
            normal_fusion_profile="native",
            expected_memory_layer_count=42,
            memory_target_layers=list(range(42)),
            memory_delta_heads=["q", "o"],
            memory_rank=4,
            rwkv_ms_semantics_version=2,
            memory_backend="rwkv_ms",
        )
