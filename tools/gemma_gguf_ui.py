#!/usr/bin/env python3
"""Gradio test UI for Gemma4 GGUF served by llama.cpp."""

from __future__ import annotations

import json
import inspect
import os
import shlex
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import gradio as gr
from openai import OpenAI

from check_rwkv_ms_gguf_runtime import run_health_check, write_health_result


DEFAULT_SYSTEM_PROMPT = (
    "You are Gemma running from a local GGUF model. Answer directly and preserve "
    "the user's requested format."
)
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-e4b-it-q8"
DEFAULT_RWKV_MS_MODEL = "gemma-4-e4b-it-rwkv-ms-q8"
DEFAULT_MODEL_PATH = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-Q8_0.gguf"
)
DEFAULT_MMPROJ_PATH = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "mmproj-gemma-4-E4B-it-Q8_0.gguf"
)
DEFAULT_RWKV_MS_SIDECAR_PATH = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_REFERENCE_TRACE = os.environ.get("GGUF_REFERENCE_TRACE", ".openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json")
DEFAULT_SLOT_SAVE_PATH = ".openresearch/artifacts/gguf_ui/slots"
DEFAULT_SERVER_LOG = ".openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log"
DEFAULT_HEALTH_OUTPUT = ".openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/") + "/", api_key=env("OPENAI_API_KEY", "llama.cpp"))


def log_dir() -> Path:
    path = Path(env("GGUF_LOG_DIR", ".openresearch/artifacts/gguf_ui"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_path_text(path: str) -> str:
    return str(Path(path.strip()).expanduser().resolve(strict=False)) if path.strip() else ""


def health_max_age_seconds() -> int:
    return int(env("GGUF_RWKV_MS_HEALTH_MAX_AGE_SECONDS", "3600"))


def health_summary(result: dict[str, Any]) -> dict[str, Any]:
    checks = result.get("checks", {})
    return {
        "ok": result.get("ok"),
        "run_id": result.get("run_id"),
        "timestamp": result.get("timestamp"),
        "base_url": result.get("base_url"),
        "model": result.get("model"),
        "base_gguf": result.get("base_gguf"),
        "base_gguf_sha256": result.get("base_gguf_sha256"),
        "sidecar_base_gguf_sha256": result.get("sidecar_base_gguf_sha256"),
        "sidecar": result.get("sidecar"),
        "server_log": result.get("server_log"),
        "checks": {name: section.get("ok") for name, section in checks.items()},
        "error": result.get("error"),
    }


def load_health_file(path: str) -> dict[str, Any]:
    health_path = Path(path.strip() or DEFAULT_HEALTH_OUTPUT)
    with health_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rwkv_ms_verification_gate(
    *,
    rwkv_ms_enabled: bool,
    require_verification: bool,
    base_url: str,
    model: str,
    model_path: str,
    sidecar_path: str,
    server_log_path: str,
    health_output_path: str,
) -> dict[str, Any]:
    if not rwkv_ms_enabled:
        return {"ok": True, "required": False}
    if not require_verification:
        return {"ok": True, "required": False, "warning": "RWKV-MS runtime verification was disabled"}

    issues: list[str] = []
    try:
        health = load_health_file(health_output_path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "required": True, "issues": [f"health file unavailable: {exc!r}"]}

    if health.get("ok") is not True:
        issues.append("last RWKV-MS runtime health check failed")
    if health.get("base_url", "").rstrip("/") != base_url.rstrip("/"):
        issues.append("health check base URL does not match the selected backend")
    if health.get("model") != model:
        issues.append("health check model does not match the selected model")
    if normalize_path_text(str(health.get("base_gguf", ""))) != normalize_path_text(model_path):
        issues.append("health check base GGUF does not match the selected model path")
    if normalize_path_text(str(health.get("sidecar", ""))) != normalize_path_text(sidecar_path):
        issues.append("health check sidecar does not match the selected sidecar")
    if normalize_path_text(str(health.get("server_log", ""))) != normalize_path_text(server_log_path):
        issues.append("health check server log does not match the selected log")

    try:
        timestamp = datetime.fromisoformat(str(health.get("timestamp")))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > health_max_age_seconds():
            issues.append(f"health check is stale: {int(age)} seconds old")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"health check timestamp is invalid: {exc!r}")

    return {
        "ok": not issues,
        "required": True,
        "issues": issues,
        "health": health_summary(health),
    }


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as response:  # noqa: S310 - local llama.cpp UI helper.
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


def assistant_message_text(raw_response: dict[str, Any]) -> tuple[str, str | None]:
    message = raw_response.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content")
    return content or reasoning or "", reasoning


def messages_for_request(system_prompt: str, history: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend({"role": item["role"], "content": item["content"]} for item in history)
    messages.append({"role": "user", "content": user_text})
    return messages


def generation_params(
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_params: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    extra_body: dict[str, Any] = {
        "top_k": int(top_k),
        "repeat_penalty": float(repeat_penalty),
    }
    if int(seed) >= 0:
        extra_body["seed"] = int(seed)
    return request_params, extra_body


def call_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    seed: int,
) -> tuple[str, dict[str, Any], float]:
    request_params, extra_body = generation_params(max_tokens, temperature, top_p, top_k, repeat_penalty, seed)
    started = time.perf_counter()
    response = make_client(base_url).chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **request_params,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    raw = response.model_dump(mode="json")
    text, _reasoning = assistant_message_text(raw)
    return text, raw, latency_ms


def chat_once(
    user_text: str,
    history: list[dict[str, str]],
    system_prompt: str,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    seed: int,
    rwkv_ms_enabled: bool,
    model_path: str,
    sidecar_path: str,
    server_log_path: str,
    health_output_path: str,
    require_rwkv_ms_verification: bool,
) -> tuple[list[dict[str, str]], str, str]:
    history = history or []
    if not user_text.strip():
        return history, "", "Empty prompt."

    messages = messages_for_request(system_prompt, history, user_text.strip())
    run_id = str(uuid.uuid4())
    params, extra_body = generation_params(max_tokens, temperature, top_p, top_k, repeat_penalty, seed)
    row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": utc_now(),
        "backend": "llama.cpp_openai_compatible",
        "base_url": base_url,
        "model": model,
        "model_path": model_path,
        "mmproj_path": env("GGUF_MMPROJ_PATH", DEFAULT_MMPROJ_PATH),
        "backend_mode": "rwkv_ms_sidecar" if rwkv_ms_enabled else "base_gguf",
        "rwkv_ms_requested": bool(rwkv_ms_enabled),
        "rwkv_ms_sidecar_path": sidecar_path.strip() if rwkv_ms_enabled else "",
        "rwkv_ms_runtime_verification_required": bool(require_rwkv_ms_verification),
        "messages": messages,
        "params": params,
        "extra_body": extra_body,
    }

    gate = rwkv_ms_verification_gate(
        rwkv_ms_enabled=rwkv_ms_enabled,
        require_verification=require_rwkv_ms_verification,
        base_url=base_url,
        model=model,
        model_path=model_path,
        sidecar_path=sidecar_path,
        server_log_path=server_log_path,
        health_output_path=health_output_path,
    )
    row["rwkv_ms_runtime_verification"] = gate
    if not gate.get("ok"):
        row.update({"error": "RWKV-MS runtime verification is required before sidecar chat"})
        append_jsonl(log_dir() / "chat.jsonl", row)
        return history, "", json.dumps(row, indent=2, ensure_ascii=False)

    try:
        text, raw_response, latency_ms = call_chat_completion(
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )
        row.update(
            {
                "latency_ms": latency_ms,
                "response_text": text,
                "reasoning_content": raw_response.get("choices", [{}])[0].get("message", {}).get("reasoning_content"),
                "raw_response": raw_response,
                "finish_reason": raw_response.get("choices", [{}])[0].get("finish_reason"),
                "usage": raw_response.get("usage"),
            }
        )
        history = history + [{"role": "user", "content": user_text.strip()}, {"role": "assistant", "content": text}]
        status = f"Saved run {run_id} in {log_dir() / 'chat.jsonl'}"
    except Exception as exc:  # noqa: BLE001 - show backend errors in the UI.
        row.update({"error": repr(exc)})
        status = f"Backend error: {exc!r}"

    append_jsonl(log_dir() / "chat.jsonl", row)
    return history, "", json.dumps(row, indent=2, ensure_ascii=False)


def check_backend(base_url: str) -> str:
    try:
        models = make_client(base_url).models.list()
        return json.dumps(models.model_dump(mode="json"), indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return f"Backend check failed: {exc!r}"


def shell_env(name: str, value: str | int) -> str:
    return f"{name}={shlex.quote(str(value))}"


def host_port_from_base_url(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return host, parsed.port
    if parsed.scheme == "https":
        return host, 443
    return host, 80


def root_url_from_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def chatbot_component() -> gr.Chatbot:
    kwargs: dict[str, Any] = {"label": "Conversation", "height": 460}
    if "type" in inspect.signature(gr.Chatbot).parameters:
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


def server_command(
    base_url: str,
    model_path: str,
    mmproj_path: str,
    sidecar_path: str,
    slot_save_path: str,
    server_log_path: str,
    alias: str,
    rwkv_ms_enabled: bool,
) -> str:
    host, port = host_port_from_base_url(base_url)
    env_lines = [
        shell_env("GGUF_MODEL_PATH", model_path),
        shell_env("GGUF_MMPROJ_PATH", mmproj_path),
        shell_env("LLAMA_MODEL_ALIAS", alias),
        shell_env("LLAMA_HOST", host),
        shell_env("LLAMA_PORT", port),
        "LLAMA_REASONING=off",
    ]
    if rwkv_ms_enabled:
        env_lines.extend(
            [
                "LLAMA_RWKV_MS=1",
                shell_env("GGUF_RWKV_MS_SIDECAR_PATH", sidecar_path),
                "LLAMA_BATCH_SIZE=2",
                "LLAMA_UBATCH_SIZE=1",
                "LLAMA_PARALLEL=1",
                "LLAMA_CONT_BATCHING=0",
                "LLAMA_CONTEXT_SHIFT=0",
                "LLAMA_CACHE_PROMPT=0",
                "LLAMA_CACHE_IDLE_SLOTS=0",
                "LLAMA_CACHE_RAM=0",
                "LLAMA_CTX_CHECKPOINTS=0",
                shell_env("LLAMA_SLOT_SAVE_PATH", slot_save_path.strip() or DEFAULT_SLOT_SAVE_PATH),
                "LLAMA_TEXT_ONLY=1",
            ]
        )
    env_prefix = " \\\n".join(env_lines)
    return "\n".join(
        [
            "# Start llama.cpp before using the UI:",
            f"{env_prefix} \\",
            f"bash tools/llama_server_gemma4.sh 2>&1 | tee {shlex.quote(server_log_path.strip() or DEFAULT_SERVER_LOG)}",
            "",
            "# Direct smoke for the patched runtime:",
            f"LLAMA_RWKV_MS=1 {shell_env('LLAMA_HOST', host)} {shell_env('LLAMA_PORT', port)} bash tools/llama_server_gemma4.sh",
        ]
    )


def clear_chat() -> tuple[list[dict[str, str]], str]:
    return [], ""


def slot_action(base_url: str, action: str, filename: str) -> str:
    run_id = str(uuid.uuid4())
    action = action.strip().lower()
    filename = filename.strip()
    row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": utc_now(),
        "base_url": base_url,
        "slot_id": 0,
        "action": action,
        "filename": filename,
    }
    try:
        if action not in {"save", "restore"}:
            raise ValueError(f"Unsupported slot action: {action}")
        if not filename:
            raise ValueError("Slot filename is required")
        payload = {"filename": filename}
        raw = request_json("POST", f"{root_url_from_base_url(base_url)}/slots/0?action={action}", payload)
        row["raw_response"] = raw
    except HTTPError as exc:
        row["http_error"] = decode_http_error(exc)
        row["error"] = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - show backend errors in the UI.
        row["error"] = repr(exc)
    append_jsonl(log_dir() / "slot_actions.jsonl", row)
    return json.dumps(row, indent=2, ensure_ascii=False)


def verify_rwkv_ms_runtime(
    base_url: str,
    model: str,
    model_path: str,
    sidecar_path: str,
    trace_path: str,
    server_log_path: str,
    slot_save_path: str,
    health_output_path: str,
) -> str:
    output_path = Path(health_output_path.strip() or DEFAULT_HEALTH_OUTPUT)
    try:
        result = run_health_check(
            base_url=base_url,
            model=model,
            trace=Path(trace_path.strip() or DEFAULT_REFERENCE_TRACE),
            base_gguf=Path(model_path.strip() or DEFAULT_MODEL_PATH),
            sidecar=Path(sidecar_path.strip() or DEFAULT_RWKV_MS_SIDECAR_PATH),
            server_log=Path(server_log_path.strip() or DEFAULT_SERVER_LOG),
            slot_save_path=Path(slot_save_path.strip() or DEFAULT_SLOT_SAVE_PATH),
            timeout=float(env("GGUF_RWKV_MS_HEALTH_TIMEOUT", "60")),
        )
        write_health_result(result, output_path)
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "run_id": str(uuid.uuid4()),
            "timestamp": utc_now(),
            "base_url": base_url,
            "model": model,
            "base_gguf": model_path,
            "sidecar": sidecar_path,
            "trace": trace_path,
            "server_log": server_log_path,
            "error": repr(exc),
        }
        write_health_result(result, output_path)
    append_jsonl(log_dir() / "runtime_health.jsonl", result)
    return json.dumps(result, indent=2, ensure_ascii=False)


def load_reference_trace(path: str) -> dict[str, Any]:
    trace_path = Path(path).expanduser().resolve()
    with trace_path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    if trace.get("schema") != "delta_mem_rwkv_ms_reference_trace.v1":
        raise ValueError(f"Unexpected reference trace schema in {trace_path}")
    return trace


def compare_to_reference_trace(
    trace_path: str,
    base_url: str,
    model: str,
    repeat_penalty: float,
    rwkv_ms_enabled: bool,
    model_path: str,
    sidecar_path: str,
    server_log_path: str,
    health_output_path: str,
    require_rwkv_ms_verification: bool,
) -> tuple[str, str, str]:
    run_id = str(uuid.uuid4())
    row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": utc_now(),
        "backend": "llama.cpp_openai_compatible",
        "comparison": "gguf_rwkv_ms_sidecar_vs_delta_mem_reference_trace" if rwkv_ms_enabled else "gguf_base_vs_delta_mem_reference_trace",
        "trace_path": trace_path,
        "base_url": base_url,
        "model": model,
        "model_path": model_path,
        "backend_mode": "rwkv_ms_sidecar" if rwkv_ms_enabled else "base_gguf",
        "rwkv_ms_requested": bool(rwkv_ms_enabled),
        "rwkv_ms_sidecar_path": sidecar_path.strip() if rwkv_ms_enabled else "",
        "rwkv_ms_runtime_verification_required": bool(require_rwkv_ms_verification),
    }
    gate = rwkv_ms_verification_gate(
        rwkv_ms_enabled=rwkv_ms_enabled,
        require_verification=require_rwkv_ms_verification,
        base_url=base_url,
        model=model,
        model_path=model_path,
        sidecar_path=sidecar_path,
        server_log_path=server_log_path,
        health_output_path=health_output_path,
    )
    row["rwkv_ms_runtime_verification"] = gate
    if not gate.get("ok"):
        row.update({"error": "RWKV-MS runtime verification is required before sidecar trace comparison"})
        append_jsonl(log_dir() / "trace_compare.jsonl", row)
        return "", "", json.dumps(row, indent=2, ensure_ascii=False)
    try:
        trace = load_reference_trace(trace_path)
        generation = trace["generation"]
        prompt = generation["prompt"]
        reference_text = trace["result"]["assistant_display"]
        max_tokens = int(generation.get("max_new_tokens", 64))
        temperature = float(generation.get("temperature", 1.0))
        top_p = float(generation.get("top_p", 1.0))
        top_k = int(generation.get("top_k", 0))
        seed = generation.get("sample_seed")
        seed_arg = -1 if seed is None else int(seed)
        messages = [{"role": "user", "content": prompt}]
        text, raw_response, latency_ms = call_chat_completion(
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed_arg,
        )
        row.update(
            {
                "reference_text": reference_text,
                "gguf_text": text,
                "gguf_content": raw_response.get("choices", [{}])[0].get("message", {}).get("content") or "",
                "gguf_reasoning_content": raw_response.get("choices", [{}])[0].get("message", {}).get("reasoning_content"),
                "exact_match": text.strip() == reference_text.strip(),
                "latency_ms": latency_ms,
                "generation": generation,
                "raw_response": raw_response,
                "usage": raw_response.get("usage"),
                "reference_source": trace.get("source"),
                "reference_turn_stats": trace.get("result", {}).get("turn_stats"),
                "reference_state_stats": trace.get("result", {}).get("state_stats"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"error": repr(exc)})
        append_jsonl(log_dir() / "trace_compare.jsonl", row)
        return "", "", json.dumps(row, indent=2, ensure_ascii=False)
    append_jsonl(log_dir() / "trace_compare.jsonl", row)
    return row["reference_text"], row["gguf_text"], json.dumps(row, indent=2, ensure_ascii=False)


def build_demo() -> gr.Blocks:
    base_url_default = env("LLAMA_BASE_URL", DEFAULT_BASE_URL)
    sidecar_path_default = env("GGUF_RWKV_MS_SIDECAR_PATH", DEFAULT_RWKV_MS_SIDECAR_PATH)
    rwkv_ms_default = env_flag("LLAMA_RWKV_MS", bool(os.environ.get("GGUF_RWKV_MS_SIDECAR_PATH")))
    model_default = env("LLAMA_MODEL", DEFAULT_RWKV_MS_MODEL if rwkv_ms_default else DEFAULT_MODEL)
    model_path_default = env("GGUF_MODEL_PATH", DEFAULT_MODEL_PATH)
    mmproj_path_default = env("GGUF_MMPROJ_PATH", DEFAULT_MMPROJ_PATH)
    slot_save_path_default = env("LLAMA_SLOT_SAVE_PATH", DEFAULT_SLOT_SAVE_PATH)
    server_log_default = env("LLAMA_SERVER_LOG", DEFAULT_SERVER_LOG)
    health_output_default = env("GGUF_RWKV_MS_HEALTH_OUTPUT", DEFAULT_HEALTH_OUTPUT)
    require_health_default = env_flag("GGUF_UI_REQUIRE_RWKV_MS_HEALTH", True)

    with gr.Blocks(title="Gemma4 GGUF Test UI") as demo:
        gr.Markdown("# Gemma4 GGUF Test UI")

        with gr.Row():
            base_url = gr.Textbox(label="OpenAI base URL", value=base_url_default, scale=2)
            model = gr.Textbox(label="Served model", value=model_default, scale=1)

        with gr.Accordion("Backend launch", open=False):
            model_path = gr.Textbox(label="GGUF model path", value=model_path_default)
            mmproj_path = gr.Textbox(label="MM projector path", value=mmproj_path_default)
            sidecar_path = gr.Textbox(label="RWKV-MS sidecar path", value=sidecar_path_default)
            slot_save_path = gr.Textbox(label="Slot save path", value=slot_save_path_default)
            server_log_path = gr.Textbox(label="Server log path", value=server_log_default)
            health_output_path = gr.Textbox(label="Runtime health file", value=health_output_default)
            rwkv_ms_enabled = gr.Checkbox(label="RWKV-MS sidecar", value=rwkv_ms_default)
            require_rwkv_ms_verification = gr.Checkbox(label="Require RWKV-MS verification", value=require_health_default)
            command = gr.Code(
                label="llama-server command",
                value=server_command(
                    base_url_default,
                    model_path_default,
                    mmproj_path_default,
                    sidecar_path_default,
                    slot_save_path_default,
                    server_log_default,
                    model_default,
                    rwkv_ms_default,
                ),
                language="shell",
            )
            refresh_command = gr.Button("Refresh command")
            backend_check = gr.Button("Check backend")
            verify_runtime = gr.Button("Verify RWKV-MS runtime")
            backend_status = gr.Code(label="Backend status", language="json")
            runtime_status = gr.Code(label="RWKV-MS runtime status", language="json")

        chatbot = chatbot_component()
        user_text = gr.Textbox(label="Prompt", lines=5, placeholder="Ask the local GGUF model...")

        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Reset conversation")

        with gr.Accordion("Generation controls", open=True):
            system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
            with gr.Row():
                max_tokens = gr.Slider(1, 2048, value=256, step=1, label="Max tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.2, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
            with gr.Row():
                top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k")
                repeat_penalty = gr.Slider(0.8, 1.5, value=1.05, step=0.01, label="Repeat penalty")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")

        raw = gr.Code(label="Last request/response log row", language="json")

        with gr.Accordion("Slot state", open=False):
            slot_filename = gr.Textbox(label="Slot filename", value="manual-slot.bin")
            with gr.Row():
                save_slot = gr.Button("Save slot 0")
                restore_slot = gr.Button("Restore slot 0")
            slot_status = gr.Code(label="Slot action result", language="json")

        with gr.Accordion("Reference Trace Comparison", open=False):
            trace_path = gr.Textbox(label="Trace JSON", value=DEFAULT_REFERENCE_TRACE)
            compare = gr.Button("Compare")
            with gr.Row():
                reference_output = gr.Textbox(label="RWKV-MS reference", lines=6)
                gguf_output = gr.Textbox(label="GGUF backend", lines=6)
            compare_raw = gr.Code(label="Comparison log row", language="json")

        inputs = [
            user_text,
            chatbot,
            system_prompt,
            base_url,
            model,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            seed,
            rwkv_ms_enabled,
            model_path,
            sidecar_path,
            server_log_path,
            health_output_path,
            require_rwkv_ms_verification,
        ]
        submit.click(chat_once, inputs=inputs, outputs=[chatbot, user_text, raw])
        user_text.submit(chat_once, inputs=inputs, outputs=[chatbot, user_text, raw])
        clear.click(clear_chat, outputs=[chatbot, raw])
        backend_check.click(check_backend, inputs=base_url, outputs=backend_status)
        refresh_command.click(
            server_command,
            inputs=[base_url, model_path, mmproj_path, sidecar_path, slot_save_path, server_log_path, model, rwkv_ms_enabled],
            outputs=command,
        )
        verify_runtime.click(
            verify_rwkv_ms_runtime,
            inputs=[base_url, model, model_path, sidecar_path, trace_path, server_log_path, slot_save_path, health_output_path],
            outputs=runtime_status,
        )
        save_slot.click(slot_action, inputs=[base_url, gr.State("save"), slot_filename], outputs=slot_status)
        restore_slot.click(slot_action, inputs=[base_url, gr.State("restore"), slot_filename], outputs=slot_status)
        compare.click(
            compare_to_reference_trace,
            inputs=[
                trace_path,
                base_url,
                model,
                repeat_penalty,
                rwkv_ms_enabled,
                model_path,
                sidecar_path,
                server_log_path,
                health_output_path,
                require_rwkv_ms_verification,
            ],
            outputs=[reference_output, gguf_output, compare_raw],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(default_concurrency_limit=int(env("GGUF_UI_CONCURRENCY", "1")))
    demo.launch(
        server_name=env("GGUF_UI_HOST", "127.0.0.1"),
        server_port=int(env("GGUF_UI_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
