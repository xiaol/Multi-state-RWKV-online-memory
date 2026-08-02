# HOLA-on-RWKV: a hippocampus for the RWKV-7 multi-state neocortex

Goal: take HOLA's semiparametric memory (arXiv 2607.02303 — GDN state + bounded exact-KV
cache evicted by the delta-rule write magnitude beta*||e||, read via a sharpened softmax)
and replace its neocortex with this repo's RWKV-7 multi-state online memory
(`hrm_rwkv7_state` / `hrm_rwkv7_multistate_recall_score` in `dla_poc.py`).

## Why the mapping is exact, not analogical

The repo's read-before-write RWKV-7 recurrence (unit key kk, erase_gate g):

    correction_read = S_{t-1} @ (-kk)
    S_t = decay * S_{t-1} + v_t k_t^T + outer(correction_read, g*kk)

For unit-norm k_t (= kk) this is

    S_t = decay * S_{t-1} + (v_t - g * S_{t-1} kk) k_t^T

i.e. a delta rule whose residual is e_t = v_t - g * S_{t-1} k_t, with erase_gate playing
the role of HOLA's write strength beta (in the trained adapter it is the learned
in-context gate). HOLA's surprise score beta*||e|| therefore transfers verbatim:

    m_t = || v_t k_t^T + outer(correction_read, g*kk) ||_F      (the actual write)
        = || v_t - g * S_{t-1} k_t ||        for unit k_t
        = "how much this token changed the RWKV state"

The decay term is the RWKV analog of GDN's alpha gate — HOLA already argues the decay
gate is orthogonal to the method.

## Components

- **Neocortex**: the existing per-block RWKV-7 states (multi-state; boundaries from any
  policy: oracle / dla / fixed / noisy_dla / low_k_dla). Untouched.
- **Hippocampus**: ONE global cache of capacity w. During the same online pass that
  builds the block states, each token's m_t is computed read-before-write against its
  block's current state; the cache keeps the global top-w tokens by m_t as exact (k,v)
  copies. m_t is fixed at write time (same order-free semantics as HOLA).
- **Read** (for query q = k*): sharpened softmax over cache entries,
  logits = sqrt(d) * (q . k_j)  — the RMSNorm-gamma effect (norms sqrt(d), scale 1/sqrt(d))
  in this unit-norm synthetic setting. The flat control uses HOLA's measured naive scale,
  logits = 0.83 * (q . k_j), which makes a perfect match get ~uniform mass.
- **Combine**: o = rms_unit(state_read) + lambda * rms_unit(cache_read), lambda=1.
  Both reads are normalized to unit RMS before mixing so the cosine recall metric
  compares direction, mirroring HOLA's decoupled-normalization spirit at PoC level.

## Variants measured (all on the state-only ablation grid of dla_poc.py)

| variant | neocortex | cache eviction | cache read | tests |
|---|---|---|---|---|
| ms                | multi-state | none | - | baseline (existing) |
| ms+hola           | multi-state | top-w by m_t | sharp | the proposal |
| ms+recency        | multi-state | last w tokens | sharp | HOLA's matched control (H1) |
| ms+hola-flat      | multi-state | top-w by m_t | flat 0.83cos | read ablation (H2) |
| single+hola       | 1 global state | top-w by m_t | sharp | pure neocortex+hippocampus |

Grid: (needles, filler/seg, K) in {(8,12,16),(12,10,16),(16,8,12)} x policies
{oracle, dla, fixed, noisy_dla, low_k_dla} x 5 seeds x w in {8, 16}. Metric: the repo's
cosine needle recall. Note w=8 < needle count in all configs, so eviction quality is
genuinely load-bearing (the cache cannot hold everything).

## Hypotheses (keel ledger)

- H1 surprise > recency under imperfect boundaries.
- H2 sharp read required; flat read ~ no cache.
- H3 ms+hola at low_k (K/2 states) recovers most of the oracle gap.

## Expected failure mode to watch

Filler tokens arrive right after each theme switch with a then-empty/decayed state, so the
FIRST filler of a segment also has a large residual (theme is new). The cache may spend
slots on first-of-theme filler. This is faithful to HOLA ("surprise the state acted on")
— needle keys are still more distinctive than theme keys, but if it costs recall we will
see it in the w=8 rows.
