from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from deltamem.core.delta import HFDeltaMemConfig
import deltamem.train.delta_sft_experimental as experimental_train


def _parse_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *extra_args: str,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            *extra_args,
        ],
    )
    return experimental_train.parse_args()


def test_initial_adapter_snapshot_cli_is_default_off_and_accepts_fresh_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = _parse_args(monkeypatch, tmp_path)
    assert defaults.initial_adapter_output_dir is None
    assert defaults.prepare_only is False

    snapshot_path = tmp_path / "output" / "initial_adapter"
    args = _parse_args(
        monkeypatch,
        tmp_path,
        "--initial-adapter-output-dir",
        str(snapshot_path),
    )
    assert args.initial_adapter_output_dir == snapshot_path

    prepare_args = _parse_args(
        monkeypatch,
        tmp_path,
        "--initial-adapter-output-dir",
        str(snapshot_path),
        "--prepare-only",
    )
    assert prepare_args.prepare_only is True


def test_prepare_only_cli_requires_initial_adapter_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires --initial-adapter-output-dir"):
        _parse_args(monkeypatch, tmp_path, "--prepare-only")


@pytest.mark.parametrize(
    "checkpoint_args",
    [
        ("--resume-from-checkpoint", "checkpoint-32"),
        (
            "--warm-start-from-checkpoint",
            "checkpoint-32",
            "--warm-start-mode",
            "residual_hybrid_w8_ablation",
        ),
    ],
)
def test_initial_adapter_snapshot_cli_rejects_resume_and_warm_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_args: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="valid only for fresh runs"):
        _parse_args(
            monkeypatch,
            tmp_path,
            "--initial-adapter-output-dir",
            str(tmp_path / "output" / "initial_adapter"),
            *checkpoint_args,
        )


def _snapshot_args(tmp_path: Path) -> SimpleNamespace:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"messages":[]}\n', encoding="utf-8")
    return SimpleNamespace(
        output_dir=tmp_path / "output",
        initial_adapter_output_dir=tmp_path / "output" / "initial_adapter",
        train_file=train_path,
        model_path=str(model_path),
        dataset_name=None,
        dataset_split="train",
        seed=42,
        data_seed=43,
    )


def test_resolve_initial_adapter_output_dir_enforces_fresh_single_process_path(
    tmp_path: Path,
) -> None:
    args = _snapshot_args(tmp_path)
    expected = args.initial_adapter_output_dir.resolve()

    assert experimental_train.resolve_initial_adapter_output_dir(
        args,
        resume_from_checkpoint=None,
        warm_start_from_checkpoint=None,
        world_size=1,
    ) == expected

    args.output_dir.mkdir()
    (args.output_dir / "launch_manifest.json").write_text("{}\n", encoding="utf-8")
    (args.output_dir / "data_contract_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    assert experimental_train.resolve_initial_adapter_output_dir(
        args,
        resume_from_checkpoint=None,
        warm_start_from_checkpoint=None,
        world_size=1,
    ) == expected

    (args.output_dir / "unexpected.txt").write_text("partial\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires a fresh training output"):
        experimental_train.resolve_initial_adapter_output_dir(
            args,
            resume_from_checkpoint=None,
            warm_start_from_checkpoint=None,
            world_size=1,
        )


@pytest.mark.parametrize(
    ("requested_suffix", "resume", "warm_start", "world_size", "message"),
    [
        ("elsewhere", None, None, 1, "must be exactly"),
        ("initial_adapter", "checkpoint-32", None, 1, "only for fresh runs"),
        ("initial_adapter", None, Path("checkpoint-32"), 1, "only for fresh runs"),
        ("initial_adapter", None, None, 2, "single-process"),
    ],
)
def test_resolve_initial_adapter_output_dir_rejects_unsafe_modes(
    tmp_path: Path,
    requested_suffix: str,
    resume: str | None,
    warm_start: Path | None,
    world_size: int,
    message: str,
) -> None:
    args = _snapshot_args(tmp_path)
    args.initial_adapter_output_dir = args.output_dir / requested_suffix

    with pytest.raises(ValueError, match=message):
        experimental_train.resolve_initial_adapter_output_dir(
            args,
            resume_from_checkpoint=resume,
            warm_start_from_checkpoint=warm_start,
            world_size=world_size,
        )


@pytest.mark.parametrize(
    "filename",
    ["launch_manifest.json", "data_contract_manifest.json"],
)
def test_resolve_initial_adapter_output_dir_rejects_non_file_provenance(
    tmp_path: Path,
    filename: str,
) -> None:
    args = _snapshot_args(tmp_path)
    args.output_dir.mkdir()
    (args.output_dir / filename).mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        experimental_train.resolve_initial_adapter_output_dir(
            args,
            resume_from_checkpoint=None,
            warm_start_from_checkpoint=None,
            world_size=1,
        )


def _install_fake_adapter_serializer(
    monkeypatch: pytest.MonkeyPatch,
    adapter_state: dict[str, torch.Tensor],
    *,
    fail: bool = False,
) -> None:
    monkeypatch.setattr(
        experimental_train,
        "get_delta_mem_state_dict",
        lambda model: {name: value.clone() for name, value in adapter_state.items()},
    )

    def fake_save_adapter(model, output_dir, config) -> None:
        del model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(output_path)
        torch.save(adapter_state, output_path / "delta_mem_adapter.pt")
        torch.rand(8)
        if fail:
            raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(
        experimental_train,
        "save_delta_mem_adapter",
        fake_save_adapter,
    )


def test_seeded_initial_adapter_snapshot_is_atomic_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _snapshot_args(tmp_path)
    args.output_dir.mkdir()
    launch_manifest_path = args.output_dir / "launch_manifest.json"
    launch_manifest_path.write_text('{"schema":"launch-test"}\n', encoding="utf-8")
    data_contract_manifest_path = args.output_dir / "data_contract_manifest.json"
    data_contract_manifest_path.write_text(
        '{"schema":"data-contract-test"}\n',
        encoding="utf-8",
    )
    adapter_state = {
        "model.layers.0.self_attn.delta_q_A": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "model.layers.0.self_attn.delta_q_B": torch.ones(3, 2),
    }
    _install_fake_adapter_serializer(monkeypatch, adapter_state)
    delta_config = HFDeltaMemConfig(rank=2)
    training_protocol = {
        "schema_version": experimental_train._TRAINING_PROTOCOL_SCHEMA_VERSION,
        "tokenized_fingerprint": "snapshot-test-fingerprint",
        "train_samples": 32,
    }
    protocol_sha256 = experimental_train._protocol_sha256(training_protocol)
    model = nn.Identity()
    torch.manual_seed(123456)
    rng_before = torch.random.get_rng_state().clone()

    manifest = experimental_train.save_seeded_initial_adapter_snapshot(
        model,
        args.initial_adapter_output_dir,
        delta_config,
        args=args,
        training_protocol=training_protocol,
        training_protocol_sha256=protocol_sha256,
        train_samples=32,
        replaced_modules=["model.layers.0.self_attn"],
        trainable_names=list(adapter_state),
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    snapshot_dir = args.initial_adapter_output_dir
    assert sorted(path.name for path in snapshot_dir.iterdir()) == [
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
        "initial_adapter_manifest.json",
        "training_protocol.json",
    ]
    assert not any(path.name.startswith(".initial_adapter.tmp-") for path in args.output_dir.iterdir())

    saved_manifest = json.loads(
        (snapshot_dir / "initial_adapter_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest == manifest
    assert manifest["schema"] == "deltamem.seeded_initial_adapter.v1"
    assert manifest["global_step"] == 0
    assert manifest["fresh_run"] is True
    assert manifest["training_started"] is False
    assert manifest["optimizer_created"] is False
    assert manifest["optimizer_state_included"] is False
    assert manifest["seed"] == 42
    assert manifest["data_seed"] == 43
    assert manifest["rng_state_after_attachment"]["cpu_sha256"] == (
        experimental_train._rng_tensor_sha256(rng_before)
    )
    assert manifest["dataset"]["train_samples"] == 32
    assert manifest["dataset"]["tokenized_fingerprint"] == "snapshot-test-fingerprint"
    assert manifest["topology"]["adapter_tensor_count"] == 2
    assert manifest["topology"]["adapter_parameter_count"] == 12
    assert manifest["training_protocol"]["canonical_sha256"] == protocol_sha256
    assert manifest["launch_manifest"]["sha256"] == experimental_train._sha256_file(
        launch_manifest_path
    )
    assert manifest["data_contract_manifest"]["sha256"] == (
        experimental_train._sha256_file(data_contract_manifest_path)
    )
    for file_record in manifest["files"].values():
        artifact_path = snapshot_dir / file_record["path"]
        assert file_record["sha256"] == experimental_train._sha256_file(artifact_path)

    manifest_sha256 = saved_manifest.pop("manifest_sha256")
    assert manifest_sha256 == experimental_train._protocol_sha256(saved_manifest)

    with pytest.raises(ValueError, match="already exists"):
        experimental_train.save_seeded_initial_adapter_snapshot(
            nn.Linear(1, 1),
            snapshot_dir,
            delta_config,
            args=args,
            training_protocol=training_protocol,
            training_protocol_sha256=protocol_sha256,
            train_samples=32,
            replaced_modules=[],
            trainable_names=[],
        )


def test_seeded_initial_adapter_snapshot_cleans_failed_atomic_write_and_rng(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _snapshot_args(tmp_path)
    adapter_state = {"adapter.weight": torch.ones(2, 2)}
    _install_fake_adapter_serializer(monkeypatch, adapter_state, fail=True)
    model = nn.Identity()
    torch.manual_seed(654321)
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="injected serialization failure"):
        experimental_train.save_seeded_initial_adapter_snapshot(
            model,
            args.initial_adapter_output_dir,
            HFDeltaMemConfig(rank=2),
            args=args,
            training_protocol={"tokenized_fingerprint": "failure-test"},
            training_protocol_sha256="0" * 64,
            train_samples=32,
            replaced_modules=[],
            trainable_names=[],
        )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not args.initial_adapter_output_dir.exists()
    assert not any(
        path.name.startswith(".initial_adapter.tmp-")
        for path in args.output_dir.iterdir()
    )


def test_prepare_only_main_returns_before_training_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "output" / "initial_adapter"
    args = _parse_args(
        monkeypatch,
        tmp_path,
        "--initial-adapter-output-dir",
        str(snapshot_dir),
        "--prepare-only",
        "--no-tokenized-cache",
    )
    monkeypatch.setattr(experimental_train, "parse_args", lambda: args)
    monkeypatch.setattr(
        experimental_train,
        "resolve_resume_checkpoint",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "resolve_adapter_warm_start_checkpoint",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "prepare_adapter_warm_start",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "prepare_training_continuation",
        lambda *unused_args, **unused_kwargs: None,
    )

    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    monkeypatch.setattr(
        experimental_train.AutoTokenizer,
        "from_pretrained",
        lambda *unused_args, **unused_kwargs: tokenizer,
    )

    class FakeDataset:
        _fingerprint = "prepare-only-test"

        def __len__(self) -> int:
            return 1

    dataset = FakeDataset()
    monkeypatch.setattr(
        experimental_train,
        "load_or_prepare_tokenized_dataset",
        lambda *unused_args, **unused_kwargs: (
            dataset,
            {"training_mode": "episode"},
        ),
    )
    monkeypatch.setattr(
        experimental_train,
        "split_tokenized_dataset",
        lambda *unused_args, **unused_kwargs: (dataset, None),
    )
    monkeypatch.setattr(
        experimental_train,
        "resolve_attn_implementation",
        lambda *unused_args, **unused_kwargs: "sdpa",
    )
    model = nn.Identity()
    monkeypatch.setattr(
        experimental_train.AutoModelForCausalLM,
        "from_pretrained",
        lambda *unused_args, **unused_kwargs: model,
    )
    monkeypatch.setattr(experimental_train, "_disable_training_cache", lambda unused: None)
    monkeypatch.setattr(experimental_train, "attach_delta_mem", lambda *unused: [])
    monkeypatch.setattr(experimental_train, "iter_delta_mem_modules", lambda unused: [])
    monkeypatch.setattr(experimental_train, "freeze_non_delta_mem_params", lambda unused: [])
    monkeypatch.setattr(
        experimental_train,
        "_promote_trainable_parameters_to_fp32",
        lambda unused: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "compute_warmup_steps",
        lambda **unused_kwargs: 0,
    )
    monkeypatch.setattr(
        experimental_train,
        "build_training_protocol",
        lambda *unused_args, **unused_kwargs: {
            "tokenized_fingerprint": "prepare-only-test",
        },
    )

    def fake_snapshot(*unused_args, **unused_kwargs):
        snapshot_dir.mkdir(parents=True)
        for filename in (
            "delta_mem_adapter.pt",
            "delta_mem_config.json",
            "training_protocol.json",
        ):
            (snapshot_dir / filename).write_text("prepared\n", encoding="utf-8")
        manifest = {
            "global_step": 0,
            "fresh_run": True,
            "training_started": False,
            "optimizer_created": False,
            "optimizer_state_included": False,
            "manifest_sha256": "a" * 64,
        }
        (snapshot_dir / "initial_adapter_manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        return manifest

    monkeypatch.setattr(
        experimental_train,
        "save_seeded_initial_adapter_snapshot",
        fake_snapshot,
    )

    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError("prepare-only crossed into training construction")

    monkeypatch.setattr(experimental_train, "TrainingArguments", forbidden)
    monkeypatch.setattr(experimental_train, "DeltaMemTrainer", forbidden)

    experimental_train.main()

    assert snapshot_dir.is_dir()
    assert not (args.output_dir / "trainer").exists()
