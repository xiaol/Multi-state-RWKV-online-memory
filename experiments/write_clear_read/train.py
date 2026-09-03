#!/usr/bin/env python
"""Train and evaluate a delta-rule multi-state memory read through KV slots on a frozen LM.

Protocol per example: write(passage) -> clear context -> read(question only).
Controls at evaluation: correct state, no memory (zero), donor state (another row's passage),
and the frozen base with the passage in context (upper bound).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from memkv import MemoryModel, default_layer_ids, register_attention, roll_states  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", choices=("synthetic", "squad"), default="synthetic")
    p.add_argument("--squad-root", type=Path, default=Path("/root/x/data/squad"))
    p.add_argument("--facts", type=int, default=4, help="synthetic facts per passage")
    p.add_argument("--entities", type=int, default=4, help="synthetic distinct people per passage")
    p.add_argument("--eval-facts", type=str, default="", help="extra synthetic fact counts to evaluate, e.g. 8,16")
    p.add_argument("--layers", type=str, default="auto", help="'auto', 'full', or comma list")
    p.add_argument("--mem-dim", type=int, default=128)
    p.add_argument("--n-states", type=int, default=1)
    p.add_argument("--slots-per-state", type=int, default=16)
    p.add_argument("--routing", choices=("single", "chunk", "cosine"), default="single")
    p.add_argument("--route-temperature", type=float, default=0.1)
    p.add_argument("--decay-bias", type=float, default=4.0)
    p.add_argument("--beta-bias", type=float, default=0.0)
    p.add_argument("--slot-bias", type=float, default=-6.0)
    p.add_argument("--chunk-size", type=int, default=0, help="recurrence checkpoint chunk; 0 = 16 // n_states")
    p.add_argument("--read-mode", choices=("bank", "query"), default="bank", help="bank: static learned query bank; query: each frozen query addresses the state")
    p.add_argument("--memory", choices=("delta", "kvbank"), default="delta", help="delta: gated delta-rule matrix state; kvbank: uncompressed passage tokens (diagnostic)")
    p.add_argument("--write-source", choices=("residual", "attn_input"), default="residual", help="hidden states the write projections read: raw layer input or post-input-layernorm attention input")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-rows", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--train-pool", type=int, default=50000)
    p.add_argument("--max-passage-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--equivalence-check", action="store_true", help="verify memkv attention == eager with no slots")
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--save-adapter", action="store_true")
    return p.parse_args()


class Tok:
    def __init__(self, tokenizer, device: torch.device) -> None:
        self.tk = tokenizer
        self.device = device
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        try:
            return self.tk.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
            )
        except TypeError:
            return self.tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def prompt_and_answer_ids(self, user: str, answer: str) -> tuple[list[int], list[int]]:
        messages = [{"role": "user", "content": user}]
        prompt = self.render(messages, True)
        full = self.render(messages + [{"role": "assistant", "content": answer}], False)
        if full.startswith(prompt):
            answer_text = full[len(prompt) :]
        else:
            answer_text = answer + (self.tk.eos_token or "")
        answer_text = answer_text.rstrip("\n")
        p_ids = self.tk(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tk(answer_text, add_special_tokens=False)["input_ids"]
        return p_ids, a_ids

    def pad(self, seqs: list[list[int]], side: str) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            if side == "left":
                ids[i, width - len(s) :] = torch.tensor(s)
                mask[i, width - len(s) :] = 1
            else:
                ids[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    def write_batch(self, examples: list[D.Example], max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.tk(e.passage, add_special_tokens=True)["input_ids"][:max_tokens] for e in examples]
        return self.pad(seqs, "right")

    def read_batch(
        self, examples: list[D.Example], *, in_context: bool, dataset: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Left-padded prompt+answer ids, mask, labels (-100 outside answer), max answer length."""
        seqs, labels = [], []
        for e in examples:
            p_ids, a_ids = self.prompt_and_answer_ids(D.read_prompt(e, in_context=in_context, dataset=dataset), e.answer)
            seqs.append(p_ids + a_ids)
            labels.append([-100] * len(p_ids) + a_ids)
        ids, mask = self.pad(seqs, "left")
        lab = torch.full_like(ids, -100)
        for i, l in enumerate(labels):
            lab[i, ids.shape[1] - len(l) :] = torch.tensor(l, device=ids.device)
        max_answer = max(len(self.prompt_and_answer_ids("x", e.answer)[1]) for e in examples)
        return ids, mask, lab, max_answer

    def prompt_batch(self, examples: list[D.Example], *, in_context: bool, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [
            self.prompt_and_answer_ids(D.read_prompt(e, in_context=in_context, dataset=dataset), e.answer)[0]
            for e in examples
        ]
        return self.pad(seqs, "left")


def answer_loss(model, ids, mask, labels, max_answer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean per-token CE over answer tokens and per-row summed CE."""
    keep = max_answer + 1
    out = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep)
    logits = out.logits[:, :-1].float()
    target = labels[:, -max_answer:]
    if logits.shape[1] != target.shape[1]:
        raise RuntimeError(f"logits {logits.shape} vs target {target.shape}")
    per_tok = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction="none")
    valid = (target != -100).float()
    per_row = (per_tok * valid).sum(1)
    mean = per_row.sum() / valid.sum().clamp(min=1.0)
    return mean, per_row / valid.sum(1).clamp(min=1.0)


@torch.no_grad()
def generate(model, tok: Tok, ids, mask, max_new_tokens: int) -> list[str]:
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_id,
    )
    texts = tok.tk.batch_decode(out[:, ids.shape[1] :], skip_special_tokens=True)
    return [t.strip() for t in texts]


@torch.no_grad()
def evaluate(
    mm: MemoryModel,
    tok: Tok,
    examples: list[D.Example],
    *,
    dataset: str,
    batch_size: int,
    max_passage_tokens: int,
    max_new_tokens: int,
    with_in_context: bool = True,
    log=print,
) -> dict:
    model = mm.model
    conditions = ["memory_correct", "memory_zero", "memory_donor"]
    if with_in_context:
        conditions.append("base_in_context")
    agg = {c: {"ce": 0.0, "em": 0.0, "f1": 0.0, "n": 0} for c in conditions}
    mem_mass = {}
    samples = []
    def eval_batch(batch: list[D.Example]) -> None:
        mm.set_slots(None)
        for ctx in mm.contexts.values():
            ctx.capture = False
        w_ids, w_mask = tok.write_batch(batch, max_passage_tokens)
        states = mm.write(w_ids, w_mask)
        donor = roll_states(states, 1)
        r_ids, r_mask, r_lab, max_ans = tok.read_batch(batch, in_context=False, dataset=dataset)
        p_ids, p_mask = tok.prompt_batch(batch, in_context=False, dataset=dataset)
        local = {c: [] for c in conditions}
        local_mass = {}
        for cond in conditions:
            if cond == "memory_correct":
                mm.set_slots(states)
                ids, mask, lab, ma, pids, pmask = r_ids, r_mask, r_lab, max_ans, p_ids, p_mask
            elif cond == "memory_donor":
                mm.set_slots(donor)
                ids, mask, lab, ma, pids, pmask = r_ids, r_mask, r_lab, max_ans, p_ids, p_mask
            elif cond == "memory_zero":
                mm.set_slots(None)
                ids, mask, lab, ma, pids, pmask = r_ids, r_mask, r_lab, max_ans, p_ids, p_mask
            else:
                mm.set_slots(None)
                ids, mask, lab, ma = tok.read_batch(batch, in_context=True, dataset=dataset)
                pids, pmask = tok.prompt_batch(batch, in_context=True, dataset=dataset)
            _, per_row = answer_loss(model, ids, mask, lab, ma)
            if cond == "memory_correct":
                local_mass = mm.mem_mass()
            preds = generate(model, tok, pids, pmask, max_new_tokens)
            for i, (e, pred) in enumerate(zip(batch, preds)):
                local[cond].append((float(per_row[i]), D.exact_match(pred, e), D.f1_score(pred, e), pred))
        mm.set_slots(None)
        # commit only after the whole batch succeeded, so an OOM retry cannot double count
        for cond in conditions:
            for ce, em, f1, pred in local[cond]:
                agg[cond]["ce"] += ce
                agg[cond]["em"] += em
                agg[cond]["f1"] += f1
                agg[cond]["n"] += 1
        for k, v in local_mass.items():
            mem_mass[k] = mem_mass.get(k, 0.0) + v * len(batch)
        for e, (_, _, _, pred) in zip(batch, local["memory_correct"]):
            if len(samples) < 8:
                samples.append({"question": e.question, "gold": e.answer, "pred": pred})

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        with_oom_retry(lambda: eval_batch(batch), log, "eval_batch")
    result = {
        c: {"ce": v["ce"] / v["n"], "em": v["em"] / v["n"], "f1": v["f1"] / v["n"], "n": v["n"]}
        for c, v in agg.items()
    }
    result["mem_mass"] = {str(k): v / len(examples) for k, v in mem_mass.items()}
    result["samples"] = samples
    return result


def equivalence_check(mm: MemoryModel, tok: Tok, examples: list[D.Example], dataset: str) -> float:
    """Max abs logit difference between memkv attention (no slots) and stock eager attention."""
    model = mm.model
    ids, mask, _, _ = tok.read_batch(examples, in_context=True, dataset=dataset)
    mm.set_slots(None)
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        configs = [model.config, model.config.get_text_config()]
        prev = [c._attn_implementation for c in configs]
        for c in configs:
            c._attn_implementation = "eager"
        try:
            b = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        finally:
            for c, p in zip(configs, prev):
                c._attn_implementation = p
    valid = mask.bool()[..., None]
    return float(((a - b).abs() * valid).max())


def with_oom_retry(fn, log, what: str, *, retries: int = 240, wait: float = 30.0):
    """Shared GPUs: another job's memory spike must not kill a run. Retry the unit of work."""
    for attempt in range(retries):
        try:
            return fn()
        except torch.OutOfMemoryError:
            log({"event": "oom_retry", "what": what, "attempt": attempt})
            torch.cuda.empty_cache()
            time.sleep(wait)
    raise RuntimeError(f"persistent out-of-memory in {what}")


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.out / "log.jsonl"
    log_file = open(log_path, "a")

    def log(record: dict) -> None:
        record["time"] = time.time()
        line = json.dumps(record)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    register_attention()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="memkv", local_files_only=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer_ids = default_layer_ids(model, args.layers)
    mm = MemoryModel(
        model,
        layer_ids,
        mem_dim=args.mem_dim,
        n_states=args.n_states,
        slots_per_state=args.slots_per_state,
        routing=args.routing,
        route_temperature=args.route_temperature,
        decay_bias=args.decay_bias,
        beta_bias=args.beta_bias,
        slot_bias=args.slot_bias,
        chunk_size=args.chunk_size or max(2, 16 // args.n_states),
        read_mode=args.read_mode,
        memory_kind=args.memory,
        write_source=args.write_source,
    )
    tok = Tok(tokenizer, device)
    config = {**vars(args), "out": str(args.out), "squad_root": str(args.squad_root), "layer_ids": layer_ids,
              "adapter_params": mm.num_parameters(), "model_type": model.config.model_type}
    log({"event": "config", **config})

    if args.dataset == "synthetic":
        train = D.synthetic_examples(args.train_pool, seed=args.seed, facts=args.facts, entities=args.entities, split="train")
        eval_sets = {f"synthetic_k{args.facts}": D.synthetic_examples(args.eval_rows, seed=args.seed + 1, facts=args.facts, entities=args.entities, split="eval")}
        for k in [int(x) for x in args.eval_facts.split(",") if x.strip()]:
            eval_sets[f"synthetic_k{k}"] = D.synthetic_examples(args.eval_rows, seed=args.seed + 1, facts=k, entities=min(args.entities, k), split="eval")
    else:
        train = D.squad_examples(args.squad_root, "train", seed=args.seed)
        eval_sets = {"squad_val": D.squad_examples(args.squad_root, "validation", limit=args.eval_rows, seed=args.seed + 1)}
    log({"event": "data", "train_rows": len(train), "eval": {k: len(v) for k, v in eval_sets.items()},
         "example": vars(train[0])})

    if args.equivalence_check:
        diff = equivalence_check(mm, tok, train[:4], args.dataset)
        log({"event": "equivalence_check", "max_abs_logit_diff": diff})

    history = []

    def run_eval(step: int) -> None:
        for name, rows in eval_sets.items():
            res = evaluate(mm, tok, rows, dataset=args.dataset, batch_size=args.eval_batch_size,
                           max_passage_tokens=args.max_passage_tokens, max_new_tokens=args.max_new_tokens, log=log)
            res.update({"event": "eval", "step": step, "set": name})
            history.append(res)
            log(res)

    if not args.skip_initial_eval:
        run_eval(0)

    params = list(mm.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 7)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        lr = lr_at(step - 1, args.steps, args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        def train_step():
            opt.zero_grad(set_to_none=True)
            mm.set_slots(None)
            for ctx in mm.contexts.values():
                ctx.capture = False
            w_ids, w_mask = tok.write_batch(batch, args.max_passage_tokens)
            states = mm.write(w_ids, w_mask)
            mm.set_slots(states)
            r_ids, r_mask, r_lab, max_ans = tok.read_batch(batch, in_context=False, dataset=args.dataset)
            loss, _ = answer_loss(model, r_ids, r_mask, r_lab, max_ans)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            mm.set_slots(None)
            return loss.detach(), gnorm

        loss, gnorm = with_oom_retry(train_step, log, "train_step")
        if step % 10 == 0 or step == 1:
            mass = mm.mem_mass()
            log({"event": "train", "step": step, "loss": float(loss), "lr": lr, "grad_norm": float(gnorm),
                 "mem_mass_mean": sum(mass.values()) / max(1, len(mass)), "elapsed": time.time() - t0})
        if step % args.eval_every == 0 or step == args.steps:
            run_eval(step)

    final = {h["set"]: {k: h[k] for k in ("memory_correct", "memory_zero", "memory_donor", "base_in_context") if k in h}
             for h in history if h["step"] == args.steps}
    result = {"config": config, "final": final, "history": history}
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    if args.save_adapter:
        torch.save(mm.state_dict(), args.out / "adapter.pt")
    log({"event": "done", "final": final})


if __name__ == "__main__":
    main()
