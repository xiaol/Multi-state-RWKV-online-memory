from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

import predict as predictor
import predict_matrix
from predict_matrix import (
    RunConfig,
    build_predict_command,
    build_work_units,
    expected_output_identity,
    git_identity,
    load_input_matrix,
    model_identity,
    output_manifest_path,
    output_status,
    prepare_run,
    run_work_queue,
    run_work_unit,
    sha256_file,
    verify_matrix_coverage,
    verify_score_output,
)


def write_rows(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"input": f"prompt {index}", "outputs": [f"answer {index}"]}
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_generation_manifest(
    root: Path,
    lengths: tuple[int, ...] = (4096,),
    tasks: tuple[str, ...] = ("niah_single_1",),
    rows: int = 5,
) -> Path:
    jobs = []
    for context_length in lengths:
        for task in tasks:
            input_file = root / str(context_length) / task / "validation.jsonl"
            write_rows(input_file, rows)
            jobs.append(
                {
                    "context_length": context_length,
                    "task": task,
                    "seed": 42,
                    "rows": rows,
                    "template_name": "gemma4-chat",
                    "subset": "validation",
                    "path": str(input_file),
                    "sha256": sha256_file(input_file),
                }
            )
    manifest_file = root / "generation_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "mode": "eval",
                "subset": "validation",
                "template_name": "gemma4-chat",
                "lengths": list(lengths),
                "tasks": list(tasks),
                "seeds": [42] * len(jobs),
                "num_samples_per_task_length": rows,
                "total_rows": len(jobs) * rows,
                "jobs": jobs,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_file


def base_config(tmp_path: Path) -> RunConfig:
    predict_script = tmp_path / "predict.py"
    predict_script.write_text("# test predictor\n", encoding="utf-8")
    score_script = tmp_path / "score.py"
    score_script.write_text("# test scorer\n", encoding="utf-8")
    return RunConfig(
        variant="base",
        model_path="/models/gemma",
        model_identity={"sha256": "m" * 64},
        adapter_dir=None,
        adapter_sha256=None,
        adapter_config_sha256=None,
        runtime_root=tmp_path,
        runtime_git_identity={"revision": "r" * 40, "tree_sha256": "t" * 64},
        dtype="bfloat16",
        attn_implementation="sdpa",
        max_new_tokens=0,
        python_bin=sys.executable,
        predict_script=predict_script,
        predict_script_sha256=sha256_file(predict_script),
        score_script=score_script,
        score_script_sha256=sha256_file(score_script),
        overwrite_output=False,
    )


def write_completed_output(unit, config: RunConfig) -> None:
    unit.output_file.parent.mkdir(parents=True, exist_ok=True)
    source_rows = [
        json.loads(line)
        for line in unit.job.input_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [
        {
            **source_rows[ordinal],
            "sample_ordinal": ordinal,
            "task": unit.job.task,
            "variant": config.variant,
            "pred": "answer",
        }
        for ordinal in unit.selected_ordinals
    ]
    unit.output_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        **expected_output_identity(unit, config),
        "status": "complete",
        "completed_rows": len(rows),
        "output_sha256": sha256_file(unit.output_file),
    }
    output_manifest_path(unit.output_file).write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )


def test_matrix_shards_keep_exact_inputs_and_cover_every_row(tmp_path: Path) -> None:
    manifest_file = write_generation_manifest(
        tmp_path / "data",
        tasks=("niah_single_1", "qa_1"),
    )
    matrix = load_input_matrix(manifest_file)
    units = build_work_units(matrix, tmp_path / "predictions", chunks_per_input=2)

    assert len(units) == 4
    assert len({unit.output_file for unit in units}) == len(units)
    for job in matrix.jobs:
        job_units = [unit for unit in units if unit.job == job]
        assert {unit.job.input_file for unit in job_units} == {job.input_file}
        covered = [ordinal for unit in job_units for ordinal in unit.selected_ordinals]
        assert sorted(covered) == list(range(job.rows))


def test_matrix_rejects_an_input_changed_after_generation(tmp_path: Path) -> None:
    manifest_file = write_generation_manifest(tmp_path / "data")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    input_file = Path(payload["jobs"][0]["path"])
    with input_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"input": "changed", "outputs": ["changed"]}) + "\n")

    with pytest.raises(ValueError, match="Input hash mismatch"):
        load_input_matrix(manifest_file)


def test_resume_requires_output_identity_and_complete_hash(tmp_path: Path) -> None:
    matrix = load_input_matrix(write_generation_manifest(tmp_path / "data"))
    unit = build_work_units(matrix, tmp_path / "predictions", 1)[0]
    config = base_config(tmp_path)
    write_completed_output(unit, config)

    assert output_status(unit, config) == "complete"

    manifest_file = output_manifest_path(unit.output_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["input_sha256"] = "0" * 64
    manifest_file.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_sha256"):
        output_status(unit, config)


def test_orphan_outputs_require_overwrite_or_matching_top_manifest(tmp_path: Path) -> None:
    matrix = load_input_matrix(write_generation_manifest(tmp_path / "data"))
    output_root = tmp_path / "predictions"
    unit = build_work_units(matrix, output_root, 1)[0]
    config = base_config(tmp_path)
    write_completed_output(unit, config)

    with pytest.raises(ValueError, match="without a matching top-level run manifest"):
        prepare_run(matrix, (unit,), output_root, ("0",), config)


def test_hybrid_output_requires_every_replaced_module_state_nonzero(tmp_path: Path) -> None:
    matrix = load_input_matrix(write_generation_manifest(tmp_path / "data"))
    unit = build_work_units(matrix, tmp_path / "predictions", 1)[0]
    config = replace(
        base_config(tmp_path),
        variant="hybrid",
        adapter_dir=tmp_path / "adapter",
        adapter_sha256="a" * 64,
        adapter_config_sha256="c" * 64,
    )
    write_completed_output(unit, config)
    rows = predictor.load_jsonl(unit.output_file)
    for row in rows:
        row["state_stats"] = {"num_modules": 1, "nonzero_modules": 0}
    unit.output_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_file = output_manifest_path(unit.output_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["replaced_modules"] = ["layer.0"]
    manifest["output_sha256"] = sha256_file(unit.output_file)
    manifest_file.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not nonzero in every module"):
        output_status(unit, config)


def test_model_and_runtime_identities_change_with_file_contents(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type":"gemma"}\n', encoding="utf-8")
    weight_file = model_root / "model.safetensors"
    weight_file.write_bytes(b"weights-v1")
    first_model = model_identity(str(model_root))
    weight_file.write_bytes(b"weights-v2")
    second_model = model_identity(str(model_root))
    assert first_model["sha256"] != second_model["sha256"]
    assert first_model["weight_files"][0]["sha256"] != second_model["weight_files"][0]["sha256"]

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime_file = runtime_root / "runtime.py"
    runtime_file.write_text("VALUE = 1\n", encoding="utf-8")
    commands = (
        ("git", "init", "-q"),
        ("git", "add", "runtime.py"),
        ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"),
    )
    for command in commands:
        predict_matrix.subprocess.run(command, cwd=runtime_root, check=True)
    first_runtime = git_identity(runtime_root)
    runtime_file.write_text("VALUE = 2\n", encoding="utf-8")
    second_runtime = git_identity(runtime_root)
    assert first_runtime["revision"] == second_runtime["revision"]
    assert first_runtime["tree_sha256"] != second_runtime["tree_sha256"]
    assert second_runtime["dirty"] is True


def test_coverage_and_score_gate_bind_outputs_to_source_rows(tmp_path: Path) -> None:
    matrix = load_input_matrix(write_generation_manifest(tmp_path / "data"))
    units = build_work_units(matrix, tmp_path / "predictions", 2)
    config = base_config(tmp_path)
    for unit in units:
        write_completed_output(unit, config)

    coverage = verify_matrix_coverage(matrix, units, config)
    assert coverage[0]["generation_manifest_sha256"] == matrix.manifest_sha256
    assert coverage[0]["observed_rows"] == matrix.rows_per_job
    assert coverage[0]["complete"] is True

    output_json = tmp_path / "scores" / "4096.json"
    output_csv = tmp_path / "scores" / "4096.csv"
    output_json.parent.mkdir()
    output_json.write_text(
        json.dumps(
            {
                "variant": "base",
                "context_length": 4096,
                "macro_score": 100.0,
                "task_count": 1,
                "complete_task_matrix": False,
                "missing_tasks": [
                    task
                    for task in predict_matrix.EVAL_TASKS
                    if task != "niah_single_1"
                ],
                "tasks": {
                    "niah_single_1": {
                        "rows": matrix.rows_per_job,
                        "files": [
                            {
                                "path": str(unit.output_file),
                                "sha256": sha256_file(unit.output_file),
                                "rows": len(unit.selected_ordinals),
                            }
                            for unit in units
                        ],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_csv.write_text("task,score,rows,null_predictions\n", encoding="utf-8")
    score_record = verify_score_output(
        matrix,
        units,
        config,
        4096,
        output_json,
        output_csv,
    )
    assert score_record["macro_score"] == 100.0
    assert score_record["rows"] == matrix.rows_per_job

    tampered = predictor.load_jsonl(units[0].output_file)
    tampered[0]["input"] = "another prompt"
    units[0].output_file.write_text(
        "".join(json.dumps(row) + "\n" for row in tampered),
        encoding="utf-8",
    )
    child_manifest_file = output_manifest_path(units[0].output_file)
    child_manifest = json.loads(child_manifest_file.read_text(encoding="utf-8"))
    child_manifest["output_sha256"] = sha256_file(units[0].output_file)
    child_manifest_file.write_text(json.dumps(child_manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match source input"):
        verify_matrix_coverage(matrix, units, config)


def test_work_unit_isolates_physical_gpu_and_uses_logical_cuda_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = load_input_matrix(write_generation_manifest(tmp_path / "data"))
    unit = build_work_units(matrix, tmp_path / "predictions", 2)[1]
    adapter_dir = tmp_path / "adapter"
    config = base_config(tmp_path)
    config = RunConfig(
        **{
            **config.__dict__,
            "variant": "hybrid",
            "adapter_dir": adapter_dir,
            "adapter_sha256": "a" * 64,
        }
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(predict_matrix.subprocess, "run", fake_run)
    run_work_unit(unit, "GPU-deadbeef", config, log_handle=io.StringIO())

    command = captured["command"]
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-deadbeef"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--variant") + 1] == "hybrid"
    assert command[command.index("--input-file") + 1] == str(unit.job.input_file)
    assert command[command.index("--chunk-index") + 1] == "1"
    assert command[command.index("--chunk-count") + 1] == "2"
    assert command[command.index("--adapter-dir") + 1] == str(adapter_dir)


def test_work_queue_never_overlaps_processes_on_one_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = load_input_matrix(
        write_generation_manifest(
            tmp_path / "data",
            tasks=("niah_single_1", "niah_single_2"),
        )
    )
    units = build_work_units(matrix, tmp_path / "predictions", 3)
    config = base_config(tmp_path)
    active: set[str] = set()
    used: set[str] = set()
    lock = threading.Lock()

    def fake_run_work_unit(unit, device, run_config):
        with lock:
            assert device not in active
            active.add(device)
            used.add(device)
        time.sleep(0.01)
        with lock:
            active.remove(device)

    monkeypatch.setattr(predict_matrix, "run_work_unit", fake_run_work_unit)
    monkeypatch.setattr(predict_matrix, "output_status", lambda unit, run_config: "complete")
    monkeypatch.setattr(predict_matrix, "assert_runtime_unchanged", lambda run_config: None)
    run_work_queue(units, ("0", "1"), config)

    assert used == {"0", "1"}
    assert not active


class FakeTensor:
    def __init__(self, length: int = 3, value: int = 0):
        self.length = length
        self.value = value

    def to(self, device):
        return self

    def size(self, dimension: int) -> int:
        assert dimension == 1
        return self.length

    def item(self) -> int:
        return self.value


class FakeLogits:
    def __getitem__(self, index):
        return self

    def argmax(self, dim: int, keepdim: bool) -> FakeTensor:
        return FakeTensor(value=99)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=8192, use_cache=False)
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        return iter((SimpleNamespace(device="cpu"),))

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=FakeLogits(), past_key_values=object())


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, prompt: str, **kwargs):
        return SimpleNamespace(input_ids=FakeTensor(), attention_mask=FakeTensor())

    def decode(self, generated_ids, skip_special_tokens: bool) -> str:
        return ""


def install_fake_prediction_modules(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeModel,
    reset_calls: list[FakeModel],
) -> None:
    torch_module = ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    torch_module.inference_mode = nullcontext
    torch_module.zeros_like = lambda tensor: FakeTensor(tensor.length)
    torch_module.full_like = lambda tensor, value: FakeTensor(tensor.length, value)
    torch_module.cuda = SimpleNamespace(is_available=lambda: False, synchronize=lambda device: None)

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    transformers_module.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: model
    )

    deltamem_module = ModuleType("deltamem")
    core_module = ModuleType("deltamem.core")
    delta_module = ModuleType("deltamem.core.delta")
    delta_impl_module = ModuleType("deltamem.core.delta_impl")
    delta_module.HFDeltaMemConfig = SimpleNamespace(
        from_pretrained=lambda adapter_dir: object()
    )
    delta_module.attach_delta_mem = lambda model, config: ["layer"]
    delta_impl_module.collect_delta_mem_state_stats = lambda model: {}
    delta_impl_module.load_delta_mem_adapter = lambda model, adapter_dir: None
    delta_impl_module.reset_delta_mem_states = lambda current_model: reset_calls.append(
        current_model
    )
    delta_impl_module.set_delta_mem_write_enabled = lambda model, enabled: None
    delta_impl_module.set_delta_mem_write_message_ids = lambda model, message_ids: None
    delta_impl_module.set_delta_mem_write_sentence_ids = lambda model, sentence_ids: None

    for name, module in {
        "torch": torch_module,
        "transformers": transformers_module,
        "deltamem": deltamem_module,
        "deltamem.core": core_module,
        "deltamem.core.delta": delta_module,
        "deltamem.core.delta_impl": delta_impl_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_hybrid_predictor_resets_state_before_every_selected_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "validation.jsonl"
    output_file = tmp_path / "predictions.jsonl"
    write_rows(input_file, 3)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "delta_mem_adapter.pt").write_bytes(b"adapter")
    runtime_root = tmp_path / "runtime"
    (runtime_root / "deltamem").mkdir(parents=True)
    model = FakeModel()
    reset_calls: list[FakeModel] = []
    install_fake_prediction_modules(monkeypatch, model, reset_calls)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
            "--task",
            "niah_single_1",
            "--variant",
            "hybrid",
            "--model-path",
            str(tmp_path / "model"),
            "--adapter-dir",
            str(adapter_dir),
            "--runtime-root",
            str(runtime_root),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "1",
            "--overwrite-output",
        ],
    )

    predictor.main()

    assert reset_calls == [model, model, model]
    assert model.calls == 3
    assert len(predictor.load_jsonl(output_file)) == 3
