from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import run_novel_agent_eval as evaluator
from experiments.rethinking_rwkv_ms_gemma import diagnose_residual_hybrid_scales as diagnostic


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
