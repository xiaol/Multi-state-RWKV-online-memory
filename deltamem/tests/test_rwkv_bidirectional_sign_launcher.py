from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma"
    / "run_natural_memory_native_rwkv_bidirectional_sign_development_gate.py"
)


def _load_launcher(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("RWKV_V5_EXACT_SOURCE_ROOT", str(REPO_ROOT))
    module_name = "bidirectional_sign_launcher_test_module"
    spec = importlib.util.spec_from_file_location(module_name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_protocol(path: Path, *, files: list[dict[str, str]]) -> tuple[str, str]:
    unsigned = {
        "schema": "rwkv_ms_bidirectional_sign_development_gate.v1",
        "manifests": {"files": files},
    }
    canonical = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(canonical).hexdigest()
    protocol = {
        **unsigned,
        "receipt": {
            "algorithm": "sha256",
            "payload_scope": "canonical_protocol_without_receipt",
            "payload_sha256": payload_sha256,
        },
    }
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _file_sha256(path), payload_sha256


def test_launcher_does_not_import_gate_core_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_launcher(monkeypatch)

    assert "experiments.rethinking_rwkv_ms_gemma.rwkv_bidirectional_sign_development_gate_core" not in sys.modules
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert source.index("validate_launcher_contract(PROTOCOL)") < source.index(
        "rwkv_bidirectional_sign_development_gate_core as gate"
    )
    assert module.EXPECTED_PROTOCOL_FILE_SHA256 == _file_sha256(module.PROTOCOL)
    assert module.EXPECTED_CORE_SHA256 == _file_sha256(module.CORE)
    assert (
        module.validate_launcher_contract(module.PROTOCOL)["receipt"][
            "payload_sha256"
        ]
        == module.EXPECTED_PROTOCOL_PAYLOAD_SHA256
    )


def test_launcher_validates_receipt_and_core_before_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_launcher(monkeypatch)
    protocol_path = tmp_path / "protocol.json"
    protocol_file_sha256, payload_sha256 = _write_protocol(protocol_path, files=[])
    monkeypatch.setattr(module, "PROTOCOL", protocol_path)
    monkeypatch.setattr(module, "EXPECTED_PROTOCOL_FILE_SHA256", protocol_file_sha256)
    monkeypatch.setattr(module, "EXPECTED_PROTOCOL_PAYLOAD_SHA256", payload_sha256)
    monkeypatch.setattr(module, "EXPECTED_CORE_SHA256", _file_sha256(module.CORE))

    assert module.validate_launcher_contract(protocol_path)["schema"] == module.SCHEMA


def test_launcher_rejects_itself_in_protocol_dag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_launcher(monkeypatch)
    launcher_relative = LAUNCHER_PATH.relative_to(REPO_ROOT).as_posix()
    protocol_path = tmp_path / "protocol.json"
    protocol_file_sha256, payload_sha256 = _write_protocol(
        protocol_path,
        files=[
            {
                "role": "gate_core",
                "scope": "project",
                "path": launcher_relative,
                "sha256": "f" * 64,
            }
        ],
    )
    monkeypatch.setattr(module, "PROTOCOL", protocol_path)
    monkeypatch.setattr(module, "EXPECTED_PROTOCOL_FILE_SHA256", protocol_file_sha256)
    monkeypatch.setattr(module, "EXPECTED_PROTOCOL_PAYLOAD_SHA256", payload_sha256)
    monkeypatch.setattr(module, "EXPECTED_CORE_SHA256", _file_sha256(module.CORE))

    with pytest.raises(ValueError, match="must not be in protocol DAG"):
        module.validate_launcher_contract(protocol_path)
