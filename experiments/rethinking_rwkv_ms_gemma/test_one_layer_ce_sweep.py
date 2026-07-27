from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma.summarize_one_layer_ce_sweep import (
    LAYERS,
    summarize_sweep,
)
from experiments.rethinking_rwkv_ms_gemma.validate_one_layer_ce_source import (
    PROVENANCE_SHA256,
    VISIBLE_FINAL_USER,
    VISIBLE_SYSTEM,
    sha256_text,
    validate_source,
)


def write_controlled_source(path: Path) -> str:
    rows = []
    for row_index in range(32):
        original_system = f"original system {row_index}"
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": VISIBLE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"{original_system}\n\nstory history {row_index}",
                    },
                    {"role": "assistant", "content": f"middle continuation {row_index}"},
                    {"role": "user", "content": VISIBLE_FINAL_USER},
                    {"role": "assistant", "content": f"target continuation {row_index}"},
                ],
                "content_control_probe": {
                    "schema_version": 1,
                    "source_sha256": PROVENANCE_SHA256,
                    "source_row_index": row_index,
                    "original_system_sha256": sha256_text(original_system),
                    "original_final_user_sha256": sha256_text(VISIBLE_FINAL_USER),
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(train_file: Path) -> dict:
    return {
        "train_file": str(train_file.resolve()),
        "tokenized_fingerprint": "shared-fingerprint",
        "tokenized_samples": 32,
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "max_length": 256,
        "max_write_length": 512,
        "teacher_max_length": 768,
        "episode_recent_messages": 1,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "context_dropout_ce",
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "validation_split_ratio": 0.0,
        "seed": 42,
        "data_seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.001,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 8,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "num_train_epochs": 4.0,
        "max_steps": 128,
        "save_steps": 32,
    }


def config(layer: int) -> dict:
    return {
        "rank": 8,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "beta_bias_init": 0.0,
        "output_init": "base_slice_fixed",
        "online_gain": 0.2,
        "target_layers": [layer],
        "delta_heads": ["q", "o"],
        "trainable_delta_scale": True,
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "memory_readout_mode": "delta",
        "memory_write_source": "learned_hidden",
        "memory_write_granularity": "token",
    }


def trainer_state(final_epoch_ce: float) -> dict:
    history = []
    for step in range(1, 129):
        epoch = (step - 1) // 32 + 1
        task_ce = final_epoch_ce + (4 - epoch) * 0.1
        history.append(
            {
                "step": step,
                "epoch": step / 32,
                "loss": task_ce,
                "delta/memory_keep_loss": task_ce,
                "delta/memory_kl_loss": 0.0,
                "delta/memory_teacher_loss": 0.0,
                "delta/memory_wmem": 1.0,
                "delta/max_state_norm": float(step),
                "grad_norm": 0.25,
            }
        )
    return {"global_step": 128, "log_history": history}


def write_run(
    sweep_root: Path,
    train_file: Path,
    layer: int,
    final_epoch_ce: float,
) -> Path:
    checkpoint = sweep_root / f"layer_{layer:02d}" / "trainer" / "checkpoint-128"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_protocol.json").write_text(
        json.dumps(protocol(train_file)), encoding="utf-8"
    )
    (checkpoint / "delta_mem_config.json").write_text(
        json.dumps(config(layer)), encoding="utf-8"
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer_state(final_epoch_ce)), encoding="utf-8"
    )
    return checkpoint


def test_controlled_source_and_complete_ranking(tmp_path: Path) -> None:
    train_file = tmp_path / "controlled.jsonl"
    expected_sha256 = write_controlled_source(train_file)
    contract = validate_source(source=train_file, expected_sha256=expected_sha256)
    assert contract["visible_system_unique_count"] == 1
    assert contract["visible_final_user_unique_count"] == 1
    assert contract["read_phase_writes_required"] is False
    assert contract["provenance"]["source_sha256"] == PROVENANCE_SHA256

    sweep_root = tmp_path / "sweep"
    final_losses = {4: 4.4, 5: 4.1, 10: 4.3, 11: 4.0, 22: 4.2, 23: 3.9}
    for layer in LAYERS:
        write_run(sweep_root, train_file, layer, final_losses[layer])
    summary = summarize_sweep(
        sweep_root=sweep_root,
        train_file=train_file,
        expected_sha256=expected_sha256,
        expected_steps=128,
    )
    assert summary["ranking_complete"] is True
    assert [row["layer"] for row in summary["ranking"]] == [23, 11, 5, 22, 10, 4]
    assert all(run["status"] == "complete" for run in summary["runs"])


def test_protocol_drift_blocks_ranking(tmp_path: Path) -> None:
    train_file = tmp_path / "controlled.jsonl"
    expected_sha256 = write_controlled_source(train_file)
    sweep_root = tmp_path / "sweep"
    checkpoints = {
        layer: write_run(sweep_root, train_file, layer, 4.0 + layer / 100)
        for layer in LAYERS
    }
    drifted = protocol(train_file)
    drifted["episode_read_write_enabled"] = True
    drifted["memory_base_kl_weight"] = 0.5
    (checkpoints[11] / "training_protocol.json").write_text(
        json.dumps(drifted), encoding="utf-8"
    )
    drifted_config = config(11)
    drifted_config["output_init"] = "zero"
    (checkpoints[11] / "delta_mem_config.json").write_text(
        json.dumps(drifted_config), encoding="utf-8"
    )
    summary = summarize_sweep(
        sweep_root=sweep_root,
        train_file=train_file,
        expected_sha256=expected_sha256,
        expected_steps=128,
    )
    layer_11 = next(run for run in summary["runs"] if run["layer"] == 11)
    assert layer_11["status"] == "invalid"
    assert any("episode_read_write_enabled" in error for error in layer_11["errors"])
    assert any("memory_base_kl_weight" in error for error in layer_11["errors"])
    assert any("output_init" in error for error in layer_11["errors"])
    assert summary["ranking_complete"] is False


def test_visible_prompt_drift_is_rejected(tmp_path: Path) -> None:
    train_file = tmp_path / "controlled.jsonl"
    write_controlled_source(train_file)
    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines()]
    rows[7]["messages"][0]["content"] = "different visible system"
    train_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    expected_sha256 = hashlib.sha256(train_file.read_bytes()).hexdigest()
    try:
        validate_source(source=train_file, expected_sha256=expected_sha256)
    except ValueError as error:
        assert "visible system prompt" in str(error)
    else:
        raise AssertionError("visible prompt drift should fail validation")
