#!/usr/bin/env python3
"""Health-check a patched llama.cpp Gemma4 RWKV-MS GGUF sidecar server."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-e4b-it-rwkv-ms-q8"
DEFAULT_TRACE = ".openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json"
DEFAULT_OUTPUT = ".openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json"
DEFAULT_SLOT_SAVE_PATH = ".openresearch/artifacts/gguf_ui/slots"
DEFAULT_BASE_GGUF = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-Q8_0.gguf"
)
DEFAULT_SIDECAR = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)

GGUF_TYPE_SIZE = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def read_u32(handle: Any) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle: Any) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_gguf_string(handle: Any) -> str:
    size = read_u64(handle)
    return read_exact(handle, size).decode("utf-8")


def skip_gguf_value(handle: Any, value_type: int) -> None:
    if value_type == 8:  # STRING
        handle.seek(read_u64(handle), 1)
        return
    if value_type == 9:  # ARRAY
        array_type = read_u32(handle)
        n_items = read_u64(handle)
        if array_type == 8:
            for _ in range(n_items):
                handle.seek(read_u64(handle), 1)
            return
        item_size = GGUF_TYPE_SIZE.get(array_type)
        if item_size is None:
            raise ValueError(f"unsupported GGUF array type {array_type}")
        handle.seek(item_size * n_items, 1)
        return
    size = GGUF_TYPE_SIZE.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    handle.seek(size, 1)


def read_gguf_string_field(path: Path, key: str) -> str:
    with path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32(handle)
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version} in {path}")
        _n_tensors = read_u64(handle)
        n_kv = read_u64(handle)
        for _ in range(n_kv):
            cur_key = read_gguf_string(handle)
            value_type = read_u32(handle)
            if cur_key == key:
                if value_type != 8:
                    raise ValueError(f"GGUF metadata key {key} is not a string")
                return read_gguf_string(handle)
            skip_gguf_value(handle, value_type)
    raise KeyError(f"missing GGUF metadata key {key}")


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as response:  # noqa: S310 - local dev server health check.
        return json.loads(response.read().decode("utf-8"))


def decode_http_error(exc: HTTPError) -> dict[str, Any]:
    body = exc.read().decode("utf-8", errors="replace")
    result: dict[str, Any] = {
        "status": exc.code,
        "reason": exc.reason,
        "body": body,
    }
    try:
        result["json"] = json.loads(body)
    except json.JSONDecodeError:
        pass
    return result


def http_error_contains(error: dict[str, Any], needle: str) -> bool:
    body = error.get("body", "")
    if needle in body:
        return True
    parsed = error.get("json")
    return parsed is not None and needle in json.dumps(parsed, ensure_ascii=False)


def corrupt_rwkv_ms_state_fingerprint(src: Path, dst: Path) -> None:
    data = bytearray(src.read_bytes())
    marker = b"RMVS\x02\x00\x00\x00"
    offset = data.find(marker)
    if offset < 0:
        raise ValueError(f"RWKV-MS v2 state header not found in {src}")
    fingerprint_offset = offset + 28
    if fingerprint_offset >= len(data):
        raise ValueError(f"RWKV-MS v2 state header is truncated in {src}")
    data[fingerprint_offset] ^= 0x01
    dst.write_bytes(data)


def root_url_from_base_url(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def assistant_text(raw: dict[str, Any]) -> str:
    message = raw.get("choices", [{}])[0].get("message", {})
    return message.get("content") or message.get("reasoning_content") or ""


def load_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    if trace.get("schema") != "delta_mem_rwkv_ms_reference_trace.v1":
        raise ValueError(f"Unexpected trace schema in {path}")
    return trace


def check_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "rwkv_ms_active": "RWKV-MS runtime is active" in text,
        "constraint_warning": "RWKV-MS sidecar runtime is constrained" in text,
        "one_slot": "n_slots = 1" in text,
        "prompt_cache_disabled": "prompt cache is disabled" in text,
        "ctx_checkpoints_disabled": "context checkpoints disabled" in text,
        "warmup_disabled": "does not support warmup" in text,
        "slot_save_restore_enabled": "slot save/restore is experimental" in text,
        "exact_prefix_reuse": "RWKV-MS exact-prefix slot reuse" in text,
        "transactional_restore_rollback": "rolled context state back" in text,
    }
    checks["ok"] = all(checks.values())
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace", type=Path, default=Path(DEFAULT_TRACE))
    parser.add_argument("--base-gguf", type=Path, default=Path(DEFAULT_BASE_GGUF))
    parser.add_argument("--sidecar", type=Path, default=Path(DEFAULT_SIDECAR))
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--slot-save-path", type=Path, default=Path(DEFAULT_SLOT_SAVE_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def run_health_check(
    *,
    base_url: str,
    model: str,
    trace: Path,
    base_gguf: Path | None,
    sidecar: Path,
    server_log: Path,
    slot_save_path: Path,
    timeout: float = 60.0,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    root_url = root_url_from_base_url(base_url)
    base_gguf = base_gguf or Path(DEFAULT_BASE_GGUF)
    result: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "base_url": base_url,
        "root_url": root_url,
        "model": model,
        "base_gguf": str(base_gguf),
        "sidecar": str(sidecar),
        "trace": str(trace),
        "server_log": str(server_log),
        "checks": {},
    }

    try:
        if not base_gguf.is_file():
            raise FileNotFoundError(f"missing base GGUF: {base_gguf}")
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing RWKV-MS sidecar: {sidecar}")
        if not server_log.is_file():
            raise FileNotFoundError(f"missing server log: {server_log}")

        sidecar_base_sha = read_gguf_string_field(sidecar, "delta_mem.base_gguf_sha256").lower()
        base_sha = sha256_file(base_gguf)
        result["sidecar_base_gguf_sha256"] = sidecar_base_sha
        result["base_gguf_sha256"] = base_sha
        result["checks"]["base_hash_binding"] = {
            "ok": len(sidecar_base_sha) == 64 and sidecar_base_sha == base_sha,
            "sidecar_base_gguf_sha256": sidecar_base_sha,
            "base_gguf_sha256": base_sha,
        }
        if not result["checks"]["base_hash_binding"]["ok"]:
            raise ValueError("RWKV-MS sidecar base GGUF SHA mismatch")

        models = request_json("GET", f"{base_url}/models", timeout=timeout)
        model_ids = [item.get("id") for item in models.get("data", [])]
        result["checks"]["model_list"] = {"ok": model in model_ids, "model_ids": model_ids}

        smoke_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "temperature": 0,
            "top_k": 1,
            "seed": 123,
        }
        started = time.perf_counter()
        smoke_raw = request_json("POST", f"{base_url}/chat/completions", smoke_payload, timeout=timeout)
        result["checks"]["chat_smoke"] = {
            "ok": bool(assistant_text(smoke_raw).strip()),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "text": assistant_text(smoke_raw),
            "finish_reason": smoke_raw.get("choices", [{}])[0].get("finish_reason"),
            "usage": smoke_raw.get("usage"),
        }

        trace_data = load_trace(trace)
        generation = trace_data["generation"]
        trace_payload = {
            "model": model,
            "messages": [{"role": "user", "content": generation["prompt"]}],
            "max_tokens": int(generation.get("max_new_tokens", 64)),
            "temperature": float(generation.get("temperature", 1.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": int(generation.get("top_k", 0)),
            "repeat_penalty": 1.0,
        }
        seed = generation.get("sample_seed")
        if seed is not None:
            trace_payload["seed"] = int(seed)
        started = time.perf_counter()
        trace_raw = request_json("POST", f"{base_url}/chat/completions", trace_payload, timeout=timeout)
        reference_text = trace_data["result"]["assistant_display"].strip()
        trace_text = assistant_text(trace_raw).strip()
        result["checks"]["reference_trace"] = {
            "ok": trace_text == reference_text,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "reference_text": reference_text,
            "gguf_text": trace_text,
            "usage": trace_raw.get("usage"),
        }

        prefix_prompt = "A reliable RWKV-MS slot prefix: alpha beta gamma."
        prefix_raw = request_json(
            "POST",
            f"{root_url}/completion",
            {
                "prompt": prefix_prompt,
                "n_predict": 0,
                "temperature": 0,
                "top_k": 1,
                "seed": 123,
                "cache_prompt": True,
                "response_fields": ["content", "tokens_cached", "timings/prompt_n"],
            },
            timeout=timeout,
        )
        slot_filename = f"rwkv-ms-health-{result['run_id']}.bin"
        slot_save = request_json("POST", f"{root_url}/slots/0?action=save", {"filename": slot_filename}, timeout=timeout)
        slot_file = slot_save_path / slot_filename
        bad_slot_filename = f"{slot_filename}.bad"
        bad_slot_file = slot_save_path / bad_slot_filename
        corrupt_rwkv_ms_state_fingerprint(slot_file, bad_slot_file)
        bad_restore: dict[str, Any]
        try:
            unexpected = request_json("POST", f"{root_url}/slots/0?action=restore", {"filename": bad_slot_filename}, timeout=timeout)
            bad_restore = {"ok": False, "unexpected_restore": unexpected}
        except HTTPError as exc:
            http_error = decode_http_error(exc)
            bad_restore = {
                "ok": http_error.get("status") == 400 and http_error_contains(http_error, "sidecar identity mismatch"),
                **http_error,
            }
        mutation_payload = {
            "prompt": "Completely different prompt to mutate the live RWKV-MS slot.",
            "n_predict": 1,
            "temperature": 0,
            "top_k": 1,
            "seed": 321,
        }
        request_json("POST", f"{root_url}/completion", mutation_payload, timeout=timeout)
        slot_restore = request_json("POST", f"{root_url}/slots/0?action=restore", {"filename": slot_filename}, timeout=timeout)
        continuation_raw = request_json(
            "POST",
            f"{root_url}/completion",
            {
                "prompt": prefix_prompt + " Continue with one short sentence.",
                "n_predict": 4,
                "temperature": 0,
                "top_k": 1,
                "seed": 123,
                "cache_prompt": True,
                "response_fields": ["content", "tokens_cached", "timings/prompt_n"],
            },
            timeout=timeout,
        )
        prompt_n = continuation_raw.get("timings/prompt_n")
        result["checks"]["slot_save_restore"] = {
            "ok": (
                slot_save.get("n_written", 0) > 0
                and bad_restore.get("ok") is True
                and slot_restore.get("n_read") == slot_save.get("n_written")
                and slot_restore.get("n_restored") == slot_save.get("n_saved")
                and isinstance(prompt_n, int)
                and prompt_n < slot_save.get("n_saved", 0)
            ),
            "filename": slot_filename,
            "prefix": prefix_raw,
            "save": slot_save,
            "bad_restore": bad_restore,
            "restore": slot_restore,
            "continuation": continuation_raw,
        }

        result["checks"]["server_log"] = check_log(server_log)
        result["ok"] = all(section.get("ok") is True for section in result["checks"].values())
    except (OSError, URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result["ok"] = False
        result["error"] = repr(exc)

    return result


def write_health_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = run_health_check(
        base_url=args.base_url,
        model=args.model,
        trace=args.trace,
        base_gguf=args.base_gguf,
        sidecar=args.sidecar,
        server_log=args.server_log,
        slot_save_path=args.slot_save_path,
        timeout=args.timeout,
    )
    write_health_result(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
