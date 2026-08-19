# Conditional fallback: headwise rotary address-value binding

This is not the current winner.  The source-and-donor-component-disjoint
bilinear cross-fit passed: held-out donor pairwise separation was `0.954545`,
mean donor gap was `0.118183`, and layer-permuted separation was `1.0`.
The bilinear output-coupling route therefore proceeds first.  Rotary binding
stays inactive unless that route fails a locked mechanics or causal gate.

The repeated failure is matched-donor identity, not insufficient recurrent
gain.  The learned address-conditioned write changed RWKV `k/v/a/b`, yet its
held-out donor CE margin was `-0.0000795`; the state-scalar screen separated
only `0.586364` of matched donors.  Both mechanisms allow the recurrent value
path to remain effectively address agnostic.

## Candidate

Bind the address into the value basis before it enters RWKV state:

```text
theta_h(a) = pi * tanh(P_h * rmsnorm(a_h))
v_bound_h  = blockdiag(R(theta_h(a_write))) * v_h
v_read_h   = blockdiag(R(-theta_h(a_query))) * raw_state_read_h
```

Each `R` is a two-dimensional rotation.  The active adapter has one RWKV head
of width 32, so its phase map has 16 complex pairs and 512 parameters per
layer (21,504 across 42 layers).  Binding stays inside the RWKV head, so it
commutes with the `v outer k` write and matrix read.  A correct write/read
address cancels exactly before group normalization and output projection.  A
donor carrying a separated address code is written in the donor basis and is
decoded with the target inverse.

This is compressed tensor binding, not another cosine score.  It changes the
stored representation and the inverse read operation; there is no scalar
identity gate.  The pure implementation is
[`rwkv_headwise_rotary_binding.py`](rwkv_headwise_rotary_binding.py).

## Priority

The bilinear route ranks first because it now has held-out component-disjoint
identity evidence.  Rotary ranks second as a conditional architectural
fallback.  It has an attractive structural hypothesis—identity lives in the
stored value basis—but only algebraic unit evidence, the same 512 parameters
per layer, and higher integration risk.  Do not combine the routes until each
has independent causal evidence.

## Gates

The next rotary action is no action while the bilinear route remains active.
If that winner fails a locked downstream gate, the fallback starts with a
separately finalized mechanics-only integration.  It must use exactly four A100s and
`HF_ENDPOINT=https://hf-mirror.com`, update no model weights, retain the
projected carrier, and prove on open fit rows that:

- the correct bound read matches the unbound raw RWKV read within `1e-5`;
- zero recurrence is exactly projected-only;
- at least `0.95` of matched donors change decoded basis, with mean normalized
  L2 change at least `0.05`;
- all writes use one stable address per routed slot and all values are finite.

A mechanics failure retires the branch.  A pass permits a separately locked
eight-update causal run.  Before generation, that run must reach donor CE
margin `>=0.02`, donor-positive row fraction `>=0.75`, positive zero and
layer-permutation margins, and held-out address-code separation `>=0.95`.

Only then may a separately locked native generation benchmark run.  Its
micro-F1 must beat projected-only, matched donor, and layer permutation by at
least `0.005` each.  This establishes open native gain, not publisher-test
SOTA; validation/test, Hard32, and strength holdouts remain closed.
