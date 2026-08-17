#!/usr/bin/env python3
"""Locate the first nonfinite branch in address-keyed DeepEmbed training."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v2
    as base,
)


CAUSAL_TRAINER = base.base.SHARED_TRAINER.__dict__["causal_train"]
SHARED_TRAINER = base.base.SHARED_TRAINER
_named_trainable: tuple[tuple[str, torch.nn.Parameter], ...] = ()
_row = 0
_branch = 0


def _output_dir(argv: Sequence[str]) -> Path:
    index = argv.index("--output-dir")
    return Path(argv[index + 1]).expanduser().resolve()


def _family(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1]


def _gradient_evidence() -> Mapping[str, Any]:
    nonfinite: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    active = 0
    for name, parameter in _named_trainable:
        gradient = parameter.grad
        if gradient is None:
            continue
        active += 1
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        family = _family(name)
        family_counts[family] = family_counts.get(family, 0) + 1
        nonfinite.append(
            {
                "name": name,
                "nan": int(torch.isnan(gradient).sum().item()),
                "positive_infinity": int(torch.isposinf(gradient).sum().item()),
                "negative_infinity": int(torch.isneginf(gradient).sum().item()),
            }
        )
    return {
        "active_gradient_tensors": active,
        "nonfinite_gradient_tensors": len(nonfinite),
        "nonfinite_family_counts": dict(sorted(family_counts.items())),
        "nonfinite_preview": nonfinite[:16],
    }


def _write_event(event: Mapping[str, Any]) -> None:
    rank = int(os.environ["RANK"])
    path = _output_dir(sys.argv) / f"gradient_diagnostic_rank{rank}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=True, sort_keys=True) + "\n")


def configure_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    global _named_trainable
    selected, audit = _original_configurer(model)
    _named_trainable = tuple(selected)
    return _named_trainable, audit


def diagnostic_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    coefficient: float,
) -> tuple[int, int]:
    global _branch, _row
    if coefficient > 0.0:
        _row += 1
        _branch = 0
        label = "correct"
    else:
        _branch += 1
        label = f"negative_{_branch}"
    logits_finite = bool(torch.isfinite(logits).all().item())
    result = _original_backward(logits, labels, coefficient=coefficient)
    _write_event(
        {
            "event": "after_backward",
            "local_row": _row,
            "branch": label,
            "coefficient": coefficient,
            "logits_finite": logits_finite,
            **_gradient_evidence(),
        }
    )
    return result


def diagnostic_accumulate(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    clean_gradients: dict[str, torch.Tensor],
) -> Mapping[str, Any]:
    evidence = dict(_gradient_evidence())
    evidence.update({"event": "row_complete", "local_row": _row})
    _write_event(evidence)
    return _original_accumulate(named_trainable, clean_gradients)


_original_configurer = SHARED_TRAINER.TRAINABLE_CONFIGURER
_original_backward = CAUSAL_TRAINER.backward_logits
_original_accumulate = CAUSAL_TRAINER.accumulate_finite_row_gradients


@contextmanager
def bindings() -> Iterator[None]:
    global _original_accumulate, _original_backward, _original_configurer
    with base.bindings():
        _original_configurer = SHARED_TRAINER.TRAINABLE_CONFIGURER
        _original_backward = CAUSAL_TRAINER.backward_logits
        _original_accumulate = CAUSAL_TRAINER.accumulate_finite_row_gradients
        previous_runner = SHARED_TRAINER.RUNNER_BINDING_PATH
        previous_minimum_rows = CAUSAL_TRAINER.MIN_ACCEPTED_ROWS_PER_UPDATE
        try:
            SHARED_TRAINER.TRAINABLE_CONFIGURER = configure_parameters
            SHARED_TRAINER.RUNNER_BINDING_PATH = Path(__file__)
            CAUSAL_TRAINER.backward_logits = diagnostic_backward
            CAUSAL_TRAINER.accumulate_finite_row_gradients = diagnostic_accumulate
            CAUSAL_TRAINER.MIN_ACCEPTED_ROWS_PER_UPDATE = 9
            yield
        finally:
            CAUSAL_TRAINER.MIN_ACCEPTED_ROWS_PER_UPDATE = previous_minimum_rows
            CAUSAL_TRAINER.accumulate_finite_row_gradients = _original_accumulate
            CAUSAL_TRAINER.backward_logits = _original_backward
            SHARED_TRAINER.RUNNER_BINDING_PATH = previous_runner
            SHARED_TRAINER.TRAINABLE_CONFIGURER = _original_configurer


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return SHARED_TRAINER.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
