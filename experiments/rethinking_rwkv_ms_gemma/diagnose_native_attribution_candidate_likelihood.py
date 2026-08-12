#!/usr/bin/env python3
"""Compare frozen-base and online-memory likelihood over attribution candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import (  # noqa: E402
    load_model_and_tokenizer,
    reset_delta_state,
    set_delta_write_enabled,
)
from deltamem.chat_templates import apply_chat_template  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as analysis,
)


SCHEMA = "rwkv_ms_native_attribution_candidate_likelihood.v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError("Attribution row messages are invalid")
            candidates = analysis.parse_candidates(str(messages[-2]["content"]))
            gold = analysis.extract_json(str(messages[-1]["content"]))
            if len(candidates) < 2 or not isinstance(gold, Mapping):
                raise ValueError("Attribution candidates or gold are invalid")
            rows.append(
                {
                    "line_index": len(rows),
                    "messages": messages[:-1],
                    "candidates": list(candidates),
                    "gold": dict(gold),
                    "row_sha256": hashlib.sha256(
                        raw_line.rstrip("\n").encode("utf-8")
                    ).hexdigest(),
                }
            )
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"Expected {limit} attribution rows, found {len(rows)}")
    return rows


def continuation_nll(
    model,
    tokenizer,
    *,
    messages: list[dict[str, str]],
    candidate: str,
    device: str,
    use_online_memory: bool,
) -> Mapping[str, Any]:
    reset_delta_state(model)
    try:
        if use_online_memory:
            write_rendered = apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            write = tokenizer(
                write_rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            set_delta_write_enabled(model, True)
            model(
                input_ids=write.input_ids.to(device),
                attention_mask=write.attention_mask.to(device),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            set_delta_write_enabled(model, False)
        prompt = apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix = prompt + '{"best_candidate":"'
        completion = candidate + '"}'
        prefix_ids = tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        full_ids = tokenizer(
            prefix + completion,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        prefix_tokens = int(prefix_ids.size(1))
        if not torch.equal(prefix_ids, full_ids[:, :prefix_tokens]):
            raise ValueError("Candidate likelihood prefix is not token-stable")
        input_ids = full_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        targets = input_ids[:, 1:]
        token_losses = F.cross_entropy(
            outputs.logits[:, :-1].float().transpose(1, 2),
            targets,
            reduction="none",
        )
        start = prefix_tokens - 1
        selected = token_losses[:, start:]
        if selected.numel() == 0:
            raise RuntimeError("Candidate continuation has no scored tokens")
        return {
            "candidate": candidate,
            "continuation_tokens": int(selected.numel()),
            "nll_sum": float(selected.sum().item()),
            "nll_mean": float(selected.mean().item()),
        }
    finally:
        reset_delta_state(model)
        set_delta_write_enabled(model, True)


def score_condition(
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    use_online_memory: bool,
) -> Mapping[str, Any]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        candidates = [
            continuation_nll(
                model,
                tokenizer,
                messages=list(row["messages"]),
                candidate=str(candidate),
                device=device,
                use_online_memory=use_online_memory,
            )
            for candidate in row["candidates"]
        ]
        selected = min(candidates, key=lambda item: float(item["nll_mean"]))[
            "candidate"
        ]
        is_correct = selected == row["gold"].get("best_candidate")
        correct += int(is_correct)
        scored_rows.append(
            {
                "line_index": row["line_index"],
                "row_sha256": row["row_sha256"],
                "gold": row["gold"].get("best_candidate"),
                "selected": selected,
                "correct": is_correct,
                "candidates": candidates,
            }
        )
        print(
            f"ATTR_LIKELIHOOD condition={'memory' if use_online_memory else 'base'} "
            f"row={row['line_index']} selected={selected!r} correct={is_correct}",
            flush=True,
        )
    return {
        "rows": scored_rows,
        "correct": correct,
        "total": len(scored_rows),
        "accuracy": correct / len(scored_rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output must be fresh: {output}")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset = args.dataset.expanduser().resolve(strict=True)
    rows = read_rows(dataset, args.limit)
    conditions: dict[str, Any] = {}
    for name, use_online_memory in (("base", False), ("memory", True)):
        model, tokenizer = load_model_and_tokenizer(
            base_model=str(base_model),
            memory_dir=str(memory_dir) if use_online_memory else None,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        conditions[name] = score_condition(
            model,
            tokenizer,
            rows,
            device=args.device,
            use_online_memory=use_online_memory,
        )
        del model, tokenizer
        torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": {
            "dataset": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "rows": args.limit,
            "selection": "first_rows_already_opened_by_v9_probe",
            "protected_test_opened": False,
            "hard32_opened": False,
        },
        "model": {
            "base": str(base_model),
            "memory_dir": str(memory_dir),
            "memory_adapter_sha256": sha256_file(
                memory_dir / "delta_mem_adapter.pt"
            ),
        },
        "scoring": {
            "candidate_template": '{"best_candidate":"<candidate>"}',
            "selection": "minimum_mean_continuation_nll",
            "online_memory_protocol": "write_then_read",
        },
        "conditions": conditions,
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
