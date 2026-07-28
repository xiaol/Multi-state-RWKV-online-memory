# Scene Memory V6 State-Identity Proof

## Purpose

This proof answers one question before any full benchmark run:

> Can the RWKV-MS state learn scene-boundary cases that the frozen Gemma base
> model fails, and does the answer depend on the contents of the current state?

Only `scene-v4-current` is in scope. Aggregate training loss is diagnostic and
is never an acceptance criterion.

The previous payload-weighted objective has a state-ignoring empirical
prefix-entropy floor of `1.2566` on its selected 512 rows. A target such as
`1.7` therefore cannot establish that the state contains the current scene.

## Frozen data

The local Hugging Face dataset checkout is pinned to revision
`5d3040d21f51b3ce90b9396b058e552c47f43cd5`.

Training uses 32 frozen-base failures mined only from the official scene-v4
training split:

- `train.jsonl`: `5f35f6ed41a2edaf88afee83626f17c34da38f5cb61cf4b6796a03eaae38f897`
- source pair-bundle manifest: `2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008`

That source-bundle hash is distinct from the trainer-generated Objective-V2
source/donor pairing manifest
`f4fb3b9611c5996518490588297d83099c8aaccad6ced6bea1c9dfd51e1dbbc6`.
The latter contains 24 presence pairs and 8 same-cardinality value pairs, no
cross-cardinality pairs, a maximum write-length delta of 61, and a unique-pair
total delta of 313.

The hard proof uses the existing 32-row official-validation selection. Its
selection did not inspect labels, base outputs, or adapter outputs:

- `holdout.jsonl`: `b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a`
- selection manifest: `76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db`

The test split remains untouched. Full 170-row validation is forbidden until a
hard-proof receipt binds a passing checkpoint.

## Benchmark contract

The official dataset rows, gold `boundaries`, and benchmark metric are not
modified. For generated output, the protected primary score is the existing
literal `boundaries` TP/FP/FN micro-F1. Boundary values are compared without
coercion; for example, integer `2` and string `"2"` are different predictions.

Format recovery remains a separately named diagnostic and is never substituted
for the strict benchmark score or its state-versus-control comparisons.

## Training contract

The topology remains all 42 Gemma layers, rank 4, Q+O delta heads, RWKV-MS
semantics v2, and attention-output additive fusion. Every row resets state.
`[system,user]` is the write and `[system,assistant]` is the state-only read;
read-side writes are disabled.

Objective V2 optimizes the row-mean loss:

```text
L = CE_full(correct_state)
  + mean_row CE_all_semantic(correct_state)
  + mean_row relu(
      0.5 - (
        CE_pair_target(donor_state)
        - CE_pair_target(correct_state)
      )
    )
```

Full correct-state CE covers the whole answer. All-semantic correct-state CE
covers every non-whitespace token inside the literal `boundaries` array. Donor
contrast covers exactly the first pair-distinguishing semantic token, and its
full causal token prefix must be identical for source and donor. The semantic
and donor coefficients are fixed at `1.0`, and the donor margin is `0.5`.

Zero-state all-semantic CE is diagnostic-only during training: it has no
objective gradient and no backward replay. Correct-versus-zero remains a
mandatory held-out proof gate. KL and representation-distance losses are zero.

Donors are deterministic, symmetric, nearest-write-length paired, and
exact-label distinct. The proof separately reports empty/nonempty,
same-cardinality nonempty, and different-cardinality nonempty donor strata.
The zero branch keeps the adapter active but disables all state writes.

Hard32 evaluation uses this frozen global length-optimal donor map:

```text
[[3,112],[6,33],[16,141],[21,88],[24,47],[30,102],
 [50,56],[59,70],[63,67],[64,71],[66,74],[75,79],
 [87,128],[113,166],[132,151],[144,159]]
```

- pair-list SHA-256: `e772e8c77210537234df4b584b7bf5f762a228362d56eb644baffd33d16c9aea`
- directed mapping-row SHA-256: `a531552ef876479a7462fe290dc61f50168fb01926be47727d177337ad13b0cf`
- maximum unique-pair write-length delta: 85
- total unique-pair write-length delta: 329
- strata: 18 empty/nonempty, 10 same-cardinality nonempty, 4 different-cardinality nonempty rows

The fresh proof run has 32 updates, two warmup updates, and checkpoints at steps
16 and 32. It cannot resume or warm-start from a trained adapter.

## Hard-proof conditions

Evaluate exactly these conditions on the fixed 32 rows:

1. `base_full`
2. `normal_full`
3. `no_write_full`
4. `state_only`
5. `state_only_donor`
6. `state_only_no_write`

The checkpoint passes only if all gates pass:

- Donor-minus-correct **pair-target** NLL gap is positive on at least 20/32 rows.
- Zero-minus-correct **all-semantic** NLL gap is positive on at least 20/32 rows.
- Donor-minus-correct pair-target NLL is positive on at least 8/10 fixed
  nonempty, same-cardinality donor rows.
- `state_only` strict benchmark micro-F1 exceeds `state_only_donor` by at least 0.05.
- `state_only` strict benchmark micro-F1 exceeds zero state by at least 0.05.
- `normal_full` strict micro-F1 exceeds the stronger of base and no-write by at
  least 0.05.
- Normal and state-only predicted/gold boundary density ratios are at most 2.0.
- State-only generation recovers at least 8 of the 32 gold boundary indices.
- State-only canonical empty-list exact accuracy is at least 6/9.
- At least 31/32 state-only outputs are recoverable and canonical.

The receipt also reports recovery of frozen-base failures and regressions on the
base-success sentinel. These diagnostics cannot replace the state-identity
gates. Objective-V2 receipts use schema
`scene_v6_identity_hard32_receipt.v2`; stale V1 receipts with the former flat
semantic-NLL evidence are rejected.

## Escalation

Evaluate checkpoints 16 and 32 only. If neither passes, change the objective or
topology before spending more training compute. If one passes, bind the selected
checkpoint in the receipt and use the complete 170-row official validation for
`scene-v4-current` as the separate final benchmark stage. Do not run the other
Novel Agent tasks. The test split remains unavailable until validation selection
is complete.
