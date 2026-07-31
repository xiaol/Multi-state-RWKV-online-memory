from __future__ import annotations

import inspect
from collections.abc import Mapping

import torch


def _validate_cached_replay_inputs(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    replay_token_ids: torch.Tensor,
) -> None:
    for name, value in (
        ("prompt_input_ids", prompt_input_ids),
        ("prompt_attention_mask", prompt_attention_mask),
        ("replay_token_ids", replay_token_ids),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, sequence]")

    if prompt_input_ids.dtype != torch.long:
        raise TypeError("prompt_input_ids must use torch.long token IDs")
    if replay_token_ids.dtype != torch.long:
        raise TypeError("replay_token_ids must use torch.long token IDs")
    if prompt_input_ids.size(0) == 0 or prompt_input_ids.size(1) == 0:
        raise ValueError("cached-prefix replay requires a nonempty prompt")
    if replay_token_ids.size(1) == 0:
        raise ValueError("cached-prefix replay requires at least one replay token")
    if replay_token_ids.size(0) != prompt_input_ids.size(0):
        raise ValueError("prompt and replay token batches must have the same size")
    if prompt_attention_mask.shape != prompt_input_ids.shape:
        raise ValueError("prompt_attention_mask must match prompt_input_ids")
    if not (
        prompt_input_ids.device
        == prompt_attention_mask.device
        == replay_token_ids.device
    ):
        raise ValueError("cached-prefix replay tensors must share one device")
    if bool(prompt_input_ids.lt(0).any()) or bool(replay_token_ids.lt(0).any()):
        raise ValueError("cached-prefix replay token IDs must be nonnegative")
    if not bool(prompt_attention_mask.eq(1).all()):
        raise ValueError("cached-prefix replay requires an unpadded prompt")


def _cached_forward_outputs(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logit_limit_kwargs: dict[str, int],
    require_single_logit_projection: bool,
    past_key_values=None,
) -> tuple[torch.Tensor, object]:
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
        **logit_limit_kwargs,
    }
    if past_key_values is not None:
        forward_kwargs["past_key_values"] = past_key_values
    outputs = model(**forward_kwargs)
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
        next_past_key_values = outputs.get("past_key_values")
    else:
        logits = getattr(outputs, "logits", None)
        next_past_key_values = getattr(outputs, "past_key_values", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.size(0) != input_ids.size(0)
        or logits.size(1) not in (1, input_ids.size(1))
        or logits.size(2) <= 0
    ):
        raise RuntimeError(
            "cached-prefix replay model logits must have shape "
            "[batch, one-or-input-sequence, vocabulary]"
        )
    if require_single_logit_projection and logits.size(1) != 1:
        raise RuntimeError(
            "cached-prefix replay model did not honor single-logit projection"
        )
    if next_past_key_values is None:
        raise RuntimeError("cached-prefix replay model did not return past_key_values")
    return logits[:, -1, :], next_past_key_values


def _unwrap_forward_model(model):
    while True:
        wrapped_model = getattr(model, "module", None)
        if wrapped_model is not None and wrapped_model is not model:
            model = wrapped_model
            continue
        original_model = getattr(model, "_orig_mod", None)
        if original_model is not None and original_model is not model:
            model = original_model
            continue
        return model


def _single_logit_projection_kwargs(
    model,
    *,
    required: bool,
) -> dict[str, int]:
    base_model = _unwrap_forward_model(model)
    try:
        call_parameters = inspect.signature(model.forward).parameters.values()
        base_parameters = inspect.signature(base_model.forward).parameters
    except (AttributeError, TypeError, ValueError) as exc:
        if required:
            raise RuntimeError(
                "cached-prefix replay could not verify single-logit projection "
                "support"
            ) from exc
        return {}
    call_accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in call_parameters
    )
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name not in base_parameters:
            continue
        if call_accepts_kwargs or any(
            parameter.name == name for parameter in call_parameters
        ):
            return {name: 1}
    if required:
        raise RuntimeError(
            "cached-prefix replay requires explicit logits_to_keep or "
            "num_logits_to_keep support"
        )
    return {}


def cached_prefix_replay_logits(
    model,
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    replay_token_ids: torch.Tensor,
    require_single_logit_projection: bool = True,
) -> torch.Tensor:
    """Return cached-path logits that selected each replay token.

    The prompt prefill predicts ``replay_token_ids[:, 0]``. Each subsequent
    replay token is predicted after feeding only the preceding token together
    with the live KV cache. By default, the model must explicitly expose a
    supported single-logit projection keyword so the replay protocol cannot
    silently run uncapped. No tensors are detached and no gradient context is
    changed, so losses on the returned ``[batch, replay, vocabulary]`` tensor
    remain differentiable through every cached forward.
    """

    _validate_cached_replay_inputs(
        prompt_input_ids,
        prompt_attention_mask,
        replay_token_ids,
    )
    if not isinstance(require_single_logit_projection, bool):
        raise TypeError("require_single_logit_projection must be a bool")
    logit_limit_kwargs = _single_logit_projection_kwargs(
        model,
        required=require_single_logit_projection,
    )
    next_token_logits, past_key_values = _cached_forward_outputs(
        model,
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        logit_limit_kwargs=logit_limit_kwargs,
        require_single_logit_projection=require_single_logit_projection,
    )
    vocabulary_size = int(next_token_logits.size(1))
    if bool(replay_token_ids.ge(vocabulary_size).any()):
        raise ValueError(
            "cached-prefix replay token ID exceeds the model vocabulary"
        )

    selection_logits = [next_token_logits]
    running_attention_mask = prompt_attention_mask
    for replay_index in range(replay_token_ids.size(1) - 1):
        running_attention_mask = torch.cat(
            (
                running_attention_mask,
                torch.ones(
                    (prompt_input_ids.size(0), 1),
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
            ),
            dim=1,
        )
        next_token_logits, past_key_values = _cached_forward_outputs(
            model,
            input_ids=replay_token_ids[:, replay_index : replay_index + 1],
            attention_mask=running_attention_mask,
            logit_limit_kwargs=logit_limit_kwargs,
            require_single_logit_projection=require_single_logit_projection,
            past_key_values=past_key_values,
        )
        if next_token_logits.size(1) != vocabulary_size:
            raise RuntimeError(
                "cached-prefix replay vocabulary changed between decode steps"
            )
        selection_logits.append(next_token_logits)
    return torch.stack(selection_logits, dim=1)
