# MARCH and RWKV-MS: Historical Anchors over Online Multi-State Memory

This note compares this repository with **MARCH: Scaling Recurrent Memory with
Content-Routed State Anchors** (arXiv:2608.12435) and documents the optional
historical-anchor extension implemented here.

## Short Answer

The two systems are close in motivation but not identical in memory geometry.

| Axis | MARCH | Existing RWKV-MS | Anchored RWKV-MS extension |
| --- | --- | --- | --- |
| Recurrent object | One cumulative Gated DeltaNet state | A fixed bank of RWKV-7 states | The same RWKV-7 state bank |
| Expansion direction | Historical snapshots over time | Parallel slots at the current time | Parallel slots and historical snapshots |
| Write allocation | Continuous base recurrence | Fixed-chunk rotating slot | Unchanged rotating-slot write |
| Retrieval unit | A cumulative state checkpoint | A current RWKV-MS slot | A historical snapshot of the full slot bank |
| Router | Compact anchor key, token query, null route | Cosine routing over current slot readouts | Compact snapshot key, token query, null route, then slot routing |
| Growth | One anchor per checkpoint | Fixed number of slots | Growing or capacity-bounded anchor bank |
| Base path | Current-state read plus historical residual | Current multi-state read | Current multi-state read plus historical residual |

RWKV-MS answers, “Which present slot should receive or answer this token?” MARCH
answers, “Which earlier version of recurrent memory still contains what later
writes erased?” The extension composes both questions instead of replacing one
with the other.

## Paper Derivation

MARCH starts from a recurrent matrix state and preserves cumulative checkpoints
at boundaries `b_m`:

```math
A^{(m,\ell)} = S^{(\ell)}_{b_m}.
```

A token produces a routing query `rho_t`, each checkpoint has a compact key
`kappa_m`, and a learned null route lets the model avoid historical memory:

```math
a_{t,m} = \rho_t^\top \kappa_m,
\qquad
\pi_{t,j} = \frac{\exp(s_{t,j})}{\sum_r \exp(s_{t,r})}.
```

The current recurrent read remains intact. Historical states form an auxiliary
residual branch:

```math
o_t = S_t q_t + \sum_j \pi_{t,j} A^{(j)} q_t.
```

This is the key architectural idea transferred into this repository.

## RWKV-MS Adaptation

RWKV-MS does not have one state. For `H` state heads and `N` rotating slots, its
online state is

```math
\mathcal{S}_t \in \mathbb{R}^{H \times N \times R \times R}.
```

The existing current-state path reads every slot with the RWKV receptance vector
`r_t`, routes over those slot readouts, and combines them:

```math
c_t = \sum_{s=1}^{N} \gamma_{t,s}\,\mathcal{S}_{t,s}r_t.
```

At every configured anchor interval, the extension snapshots the *entire slot
bank*:

```math
\mathcal{A}^{(m)} = \mathcal{S}_{b_m}.
```

A learned probe `p` compresses the bank into a descriptor. The implementation
uses a learned linear key projection:

```math
\kappa_m = \operatorname{norm}\!\left(
W_K\,\operatorname{vec}\!\left[
\frac{1}{N}\sum_s \mathcal{A}^{(m)}_s p
\right]\right).
```

The current token source produces `rho_t = norm(W_R u_t)`. Softmax routing is
performed jointly over historical snapshots and a learned null logit. Inside
each selected snapshot, the existing cosine slot router chooses the useful
RWKV-MS slots. The final memory read is

```math
o_t^{\text{mem}} = c_t
+ \lambda_A \sum_m \pi_{t,m}
  \sum_s \gamma^{(m)}_{t,s}\,\mathcal{A}^{(m)}_s r_t.
```

This preserves the live RWKV-MS path and adds a temporal-expansion residual. It
is therefore a two-level router:

1. choose a historical snapshot;
2. choose slots inside that snapshot.

The implementation is strictly causal: token `t` reads only anchors created
before `t`; a boundary snapshot is appended after the token's recurrent update.

## Configuration

Historical anchors are disabled by default, so existing configs and checkpoints
retain their behavior. Enable them only with `memory_backend=rwkv_ms`.

```bash
python -m deltamem.train.delta_sft \
  --memory-backend rwkv_ms \
  --rwkv-ms-anchor-interval 128 \
  --rwkv-ms-anchor-capacity 32 \
  --rwkv-ms-anchor-route-dim 64 \
  --rwkv-ms-anchor-top-k 4 \
  --rwkv-ms-anchor-residual-scale 1.0 \
  --rwkv-ms-anchor-null-bias-init 2.0 \
  ...
```

Parameter meanings:

- `rwkv_ms_anchor_interval=0` disables the extension. A positive value creates
  a snapshot after that many RWKV-MS write steps.
- `rwkv_ms_anchor_capacity=0` keeps every snapshot. A positive value evicts the
  oldest snapshots beyond the budget.
- `rwkv_ms_anchor_route_dim` controls compact key/query width.
- `rwkv_ms_anchor_top_k=0` uses dense historical routing. A positive value keeps
  only the highest-scoring anchors before the null-route softmax.
- `rwkv_ms_anchor_residual_scale` scales the historical branch.
- `rwkv_ms_anchor_null_bias_init` controls the initial preference for bypassing
  untrained historical memory.

Online session snapshots now persist anchor states, keys, and masks together
with the current RWKV-MS state, position, and streaming predecessor.

## Recommended Experiments

1. **Matched-state ablation:** compare anchors off/on at identical RWKV-MS slot
   count, rank, data, and training tokens.
2. **Capacity frontier:** sweep anchor capacity `{8, 16, 32, 64}` and report
   quality, memory, tokens/s, and null-route mass.
3. **Temporal versus slot expansion:** compare more current slots against fewer
   slots plus historical anchors at matched bytes.
4. **Adaptive anchoring:** replace the fixed interval with state novelty or
   update magnitude. This directly addresses MARCH's stated limitation and fits
   the surprise signal already explored by the repository's HOLA experiment.
5. **Consolidation:** demote or merge anchors that the current RWKV-MS bank can
   already answer, combining MARCH retrieval with the HOLA-style consolidation
   rule.
6. **Scene-aware checkpoints:** for online agent memory, anchor at message,
   scene, tool-result, or failure-recovery boundaries instead of raw token
   intervals.
7. **Hierarchical routing:** route first by memory type or timescale, then by
   historical anchor, then by RWKV-MS slot.

## Evidence Boundary

The paper reports strong results for MARCH over Gated DeltaNet after 50B-token
pretraining, including LongBench and retrieval gains. Those numbers do not
transfer automatically to RWKV-MS. The code in this repository is a tested
architectural prototype: causality, chunk invariance, gradients, capacity
bounding, and online-state round trips are covered by unit tests. It still needs
matched training and long-context evaluation before any quality claim can be
made for anchored RWKV-MS.

## Primary Source

- MARCH, arXiv:2608.12435, especially Equations 8-14, Figure 2, Tables 4-5, and
  the limitations section.
