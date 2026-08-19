# Next direction: learned query/state compatibility

The direct query-state identity experiment is now a useful negative result,
not a reason to increase the existing gain.  Its score was

```text
cosine(folded projected-slot write address, addressed RWKV read)
```

and the v3 endpoint produced mean positive-minus-donor `-0.002735` with only
`0.4375` positive rows.  The projected key is trained for slot routing while
the RWKV read is produced by recurrent `k/v/a/b` dynamics; comparing those
vectors directly assumes a shared basis that has not been learned.

## Candidate

Use the frozen projected route only to select/capture the target address, but
learn a tiny per-layer compatibility map before scoring:

```text
q' = q + Uq silu(Dq q)
s' = s + Us silu(Ds s)
identity_score(q,s) = cosine(q', s')
```

`q` is the answer-position query feature (`memory_q_proj(rms(hidden))`) and
`s` is the addressed RWKV read.  `Dq/Ds` have rank 4 and `Uq/Us` start at zero,
so the maps are exact identity at initialization.  With state width 32 this is
512 trainable values per layer (21,504 over 42 layers), plus no new Gemma
parameters.  The implementation is
[`rwkv_query_state_bilinear.py`](rwkv_query_state_bilinear.py).

## Safe two-phase screen

1. **Offline cross-fit identity screen (cheap, no adapter write).** On the same
   220 already-authorized native rows, capture answer-position query/state
   features for correct and matched-donor states while retaining the target
   projected carrier.  Use a deterministic 176/44 source-and-donor-disjoint
   split.  Train only the 42 low-rank heads for eight optimizer updates, with
   `max(0, 0.2 - score_correct + score_donor)`.  Report donor and
   layer-permutation controls.  This phase changes no model output.

2. **Causal identity gate (only if phase 1 passes).** Install the same heads
   in the addressed recurrent correction.  Let `base` be the previously
   tested recurrent correction and use

   ```text
   fused = projected + g(q,s) * (base - projected)
   g = sigmoid(4 * (identity_score - threshold) - 6)
   ```

   The projected carrier remains fixed; zero recurrent state gives an exactly
   zero correction regardless of `g`.  Train with the existing answer CE plus
   the matched-donor hinge, serializing active control graphs on exactly four
   A100s.  Do not run native generation until this causal endpoint passes.

## Gates and stopping rule

The offline phase must satisfy all of:

- heldout matched-donor pairwise-positive fraction `>= 0.95`;
- heldout mean `score_correct - score_donor >= 0.05`;
- layer-permuted fraction `>= 0.95` and finite all rows;
- projected-carrier identity is true for every intervention and protected
  splits remain unopened.

If the donor gate fails, retire this compatibility family without a native
benchmark.  If it passes, require the causal endpoint's donor CE margin and
identity margin before authorizing generation.  This prevents another expensive
run that only learns state presence or layer placement.

## Commands

The bounded unit contract is:

```bash
PYTHONPATH=. pytest -q deltamem/tests/test_rwkv_query_state_bilinear.py
```

The eventual four-GPU causal runner must set `HF_ENDPOINT` to
`https://hf-mirror.com`, use only `train_derived_development.jsonl`, and write
all adapters under `local_artifacts` without staging `delta_mem_adapter.pt`.
