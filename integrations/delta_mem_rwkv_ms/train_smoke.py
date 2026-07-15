from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DELTA_MEM_ROOT = ROOT.parent / "delta-Mem"
DEFAULT_TRAIN_FILE = ROOT / "configs" / "rwkv_ms_smoke_train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / ".openresearch" / "artifacts" / "rwkv_ms_train_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a verified two-step Gemma4 RWKV-MS adapter training smoke test."
    )
    parser.add_argument("--delta-mem-root", type=Path, default=DEFAULT_DELTA_MEM_ROOT)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--target-layers", default="0")
    parser.add_argument("--delta-heads", default="q,o")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--num-states", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-write-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow adapter artifacts in a non-empty output directory to be overwritten.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Training file is empty: {path}")
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Training file has no examples: {path}")
    return rows


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("target-layers must contain at least one layer index")
    return parsed


def main() -> None:
    args = parse_args()
    delta_mem_root = args.delta_mem_root.resolve()
    if not (delta_mem_root / "deltamem").is_dir():
        raise FileNotFoundError(f"Not a delta-Mem checkout: {delta_mem_root}")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {args.output_dir}")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite_output
    ):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose another path or pass --overwrite-output."
        )
    sys.path.insert(0, str(delta_mem_root))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from deltamem.core.delta import (
        HFDeltaMemConfig,
        attach_delta_mem,
        freeze_non_delta_mem_params,
    )
    from deltamem.core.delta_impl import (
        collect_delta_mem_state_stats,
        get_delta_mem_state_dict,
        load_delta_mem_adapter,
        save_delta_mem_adapter,
    )
    from deltamem.train.delta_sft_experimental import (
        DeltaMemTrainer,
        EpisodeCausalLMCollator,
        build_episode_training_examples,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The Gemma4 smoke train requires a CUDA GPU")
    if args.max_steps < 1:
        raise ValueError("max-steps must be >= 1")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    episodes: list[dict] = []
    for row in load_rows(args.train_file):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Each training row must contain a messages list")
        episodes.extend(
            build_episode_training_examples(
                tokenizer,
                messages,
                args.max_length,
                assistant_loss_mode="final_assistant_only",
                episode_recent_messages=1,
                max_write_length=args.max_write_length,
                include_sentence_ids=False,
            )
        )
    if not episodes:
        raise ValueError("Training data did not produce any episode examples")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    delta_config = HFDeltaMemConfig(
        rank=args.rank,
        alpha=args.alpha,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=args.num_states,
        rwkv_ms_chunk_size=args.chunk_size,
        rwkv_ms_boundary_mode="fixed_chunk",
        rwkv_ms_erase_gate=1.0,
        rwkv_ms_read_top_k=0,
        num_state_heads=1,
        beta_bias_init=0.0,
        couple_lambda=True,
        state_update_mode="standard",
        rankwise_gates=True,
        output_init="base_slice_fixed",
        base_slice_ref_width=8,
        delta_heads=tuple(item.strip() for item in args.delta_heads.split(",") if item.strip()),
        online_gain=0.2,
        target_layers=parse_csv_ints(args.target_layers),
        memory_readout_mode="delta",
        memory_write_source="learned_hidden",
        memory_write_granularity="token",
    )
    replaced = attach_delta_mem(model, delta_config)
    trainable_names = freeze_non_delta_mem_params(model)

    # Keep optimizer-owned adapter weights in FP32 while the frozen base runs in BF16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer-runtime"),
        per_device_train_batch_size=1,
        learning_rate=args.learning_rate,
        bf16=model_dtype == torch.bfloat16,
        report_to=["none"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    collator = EpisodeCausalLMCollator(tokenizer)
    trainer = DeltaMemTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        data_collator=collator,
        delta_config=delta_config,
        memory_loss_mode="context_dropout_ce",
        memory_contrast_weight=0.1,
        memory_kl_weight=0.1,
        memory_margin=0.1,
        memory_causal_weight=1.0,
        memory_anchor_weight=1.0,
        memory_anchor_margin=0.005,
        memory_recover_weight=0.25,
        memory_need_floor=0.15,
        memory_dropout_no_memory_prob=0.0,
        memory_dropout_state_only_prob=0.0,
        context_ablation_mode="mixed",
        context_ablation_no_state_prob=0.2,
        context_ablation_state_only_prob=0.2,
    )
    model = trainer.model
    model.train()
    device = next(model.parameters()).device
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.0)
    initial = get_delta_mem_state_dict(model)
    step_records = []

    for step in range(1, args.max_steps + 1):
        episode = episodes[(step - 1) % len(episodes)]
        batch = {name: tensor.to(device) for name, tensor in collator([episode]).items()}
        optimizer.zero_grad(set_to_none=True)
        trainer._reset_online_state(model)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if model_dtype == torch.bfloat16
            else nullcontext()
        )
        with autocast:
            loss = trainer.compute_loss(model, batch)
        loss.backward()

        grad_sq = 0.0
        max_abs_grad = 0.0
        nonzero_grad_tensors = 0
        for parameter in trainable_params:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            grad_sq += float(grad.square().sum().item())
            local_max = float(grad.abs().max().item()) if grad.numel() else 0.0
            max_abs_grad = max(max_abs_grad, local_max)
            if local_max > 0.0:
                nonzero_grad_tensors += 1
        grad_norm = math.sqrt(grad_sq)
        clip_input_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm).item()
        )
        optimizer.step()
        state_stats = collect_delta_mem_state_stats(model)
        record = {
            "step": step,
            "loss": float(loss.detach().float().item()),
            "grad_norm": grad_norm,
            "clip_input_norm": clip_input_norm,
            "max_abs_grad": max_abs_grad,
            "nonzero_grad_tensors": nonzero_grad_tensors,
            "nonzero_state_modules": state_stats["nonzero_modules"],
            "max_state_norm": state_stats["max_state_norm"],
        }
        step_records.append(record)
        print("SMOKE_STEP=" + json.dumps(record, sort_keys=True), flush=True)
        if nonzero_grad_tensors == 0 or state_stats["nonzero_modules"] == 0:
            raise RuntimeError("RWKV-MS smoke train produced a disconnected memory path")

    final = get_delta_mem_state_dict(model)
    changed = []
    max_abs_change = 0.0
    l2_change_sq = 0.0
    for name in sorted(set(initial) & set(final)):
        diff = final[name].float() - initial[name].float()
        local_max = float(diff.abs().max().item()) if diff.numel() else 0.0
        if local_max > 0.0:
            changed.append(name)
        max_abs_change = max(max_abs_change, local_max)
        l2_change_sq += float(diff.square().sum().item())
    if not changed:
        raise RuntimeError("Training completed but no adapter tensor changed")

    save_delta_mem_adapter(model, args.output_dir, delta_config)
    saved = torch.load(
        args.output_dir / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    if set(saved) != set(final):
        raise RuntimeError(
            "Saved adapter keys differ from the in-memory adapter: "
            f"missing={sorted(set(final) - set(saved))}, "
            f"unexpected={sorted(set(saved) - set(final))}"
        )
    save_max_abs_error = max(
        float((saved[name].float() - final[name].float()).abs().max().item())
        for name in saved
    )
    if save_max_abs_error != 0.0:
        raise RuntimeError(f"Saved adapter mismatch: {save_max_abs_error}")

    module_map = dict(model.named_modules())
    with torch.no_grad():
        for name in saved:
            module_name, parameter_name = name.rsplit(".", 1)
            getattr(module_map[module_name], parameter_name).add_(1.0)
    loaded_config = load_delta_mem_adapter(model, args.output_dir)
    if loaded_config.to_dict() != delta_config.to_dict():
        raise RuntimeError("Reloaded adapter config differs from the training config")
    reloaded = get_delta_mem_state_dict(model)
    if set(reloaded) != set(saved):
        raise RuntimeError("Reloaded adapter keys differ from the saved adapter")
    reload_max_abs_error = max(
        float((reloaded[name].float() - saved[name].float()).abs().max().item())
        for name in saved
    )
    if reload_max_abs_error != 0.0:
        raise RuntimeError(f"Adapter reload mismatch: {reload_max_abs_error}")

    summary = {
        "status": "passed",
        "model_path": args.model_path,
        "train_file": str(args.train_file.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "base_dtype": str(model_dtype),
        "trainable_dtype": "torch.float32",
        "seed": args.seed,
        "replaced_modules": replaced,
        "trainable_tensor_count": len(trainable_names),
        "adapter_tensor_count": len(final),
        "changed_tensor_count": len(changed),
        "max_abs_parameter_change": max_abs_change,
        "l2_parameter_change": math.sqrt(l2_change_sq),
        "save_max_abs_error": save_max_abs_error,
        "reload_config_matches": True,
        "reload_max_abs_error": reload_max_abs_error,
        "steps": step_records,
    }
    summary_path = args.output_dir / "smoke_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SMOKE_TRAINING_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
