# Full-bandwidth transformer review for RWKV memory

Source: [arXiv:2608.08888v1](https://arxiv.org/abs/2608.08888v1),
*Full-bandwidth transformer* (Wang et al., submitted 2026-08-09).  This review
uses the official arXiv source. The follow-on mechanics screen is recorded
separately below; no protected split is opened by either document.

## Exact paper mechanism

The paper separates horizontal access across positions from vertical access
through model depth.  A normal cached transformer preserves earlier layer
states, but a state produced at layer `l` is readable only by later layers
above `l`; in particular, the previous top-layer output is not placed in the
KV cache.  The paper calls this persistence without depth renewal.  Its
full-bandwidth transformer returns the previous top-layer state to layer 0 at
the next generated position (Section 3.1, Eqs. 2--3 and Figure 1).

The central mechanism is not a new state score.  It makes the carried state
the mandatory value pathway and uses the sampled token only as a gate:

```text
x_t = (W_U h^L_{t-1}) * sigmoid(W_G e_t)
h^L_t = transformer(x_t, kv_cache)
```

Both `W_U` and `W_G` are `D x D`.  The state is the value; the token identity
survives only as the `D`-dimensional multiplicative gating pattern.  The paper
argues that an additive fusion such as `e + W h` leaves a shortcut: the model
can suppress the hidden-state path and recover the ordinary input.  The
asymmetric GLU closes that shortcut because suppressing the state also removes
the entire layer-0 input (Section 3.1, Eq. 4).  The fused vector keeps width
`D`, so the transformer blocks and KV-cache layout do not change.  At decode
time the only new arithmetic is the two dense projections, reported as under
`1%` per token.  Older latent states are not maintained as explicit registers:
each is folded into a position's fused input and then into the cache; only the
latest top state is the recurrence variable (Section 3.2).

The evaluated backbone is a decoder-only roughly 1B model with a tied
`100,352`-token embedding/readout, 24 layers, hidden width `1,536`, a `6,656`
SiLU-GLU FFN, 16 query heads, 8 shared KV heads, QK RMSNorm, headwise attention
gates, and rotary positions.  Context length is `8,192`; most layers use a
`2,048` sliding window and every sixth layer uses full attention (Appendix A).
The fusion therefore adds `2 * 1536 * 1536 = 4,718,592` weights, about `0.47%`
of a 1B model.

## Exact training objective

The causal decode recurrence is sequential across positions.  The paper
approximates it during teacher forcing with Jacobi-style temporal parallelism
(Section 3.3, Eqs. 8--10 and Figure 2):

```text
h^(1) = model(embed(tokens))
loss = NTP(h^(1))
for k = 2..K:
    x^(k) = cross(embed(tokens), shift_right(h^(k-1)))
    h^(k) = model(x^(k))
    loss += NTP(h^(k))
```

Pass 1 is ordinary teacher forcing.  Each later pass shifts the previous
pass's top states one position to the right, fuses every position in parallel,
and runs the full transformer again.  The objective is

```text
L^K = L_NTP(pass 1) + lambda/(K-1) * sum_{k=2..K} L_NTP(pass k), lambda = 1
```

The first-pass loss preserves standard no-feedback operation for ordinary
prompt prefill.  The state is not detached during training: losses from later
passes backpropagate through the earlier pass that produced their carried
state.  This makes each earlier top state useful to future predictions rather
than only to its immediate vocabulary projection, at the cost of a larger
activation footprint.

The remaining training details are:

- train with parallel Jacobi-style multi-pass forwards: pass 1 is ordinary,
  and pass `k` shifts pass `k-1` hidden states, fuses them, and reruns the
  sequence;
- apply the ordinary next-token loss to every pass and keep gradients through
  the passes rather than detaching the feedback state;
- introduce recurrence late, with the reported `75%` one-pass, `22%` two-pass,
  `3%` three-pass mixture; the small three-pass fraction is reported to turn
  an unstable two-pass map into a contraction;
- use a random prefix mixin so plain prompt inputs and fused generated inputs
  share a training distribution;
- RMS-normalize the fused input, keep carried-state scale stationary, and add
  small state jitter (`Uniform[-0.02, 0.02]`) for local robustness.

The random prefix mixin is important: each feedback pass samples a boundary,
keeps the prefix as ordinary embeddings, and fuses only the suffix.  This
matches one-pass prompt prefill followed by recurrent generation.  The
stability recipe additionally uses depth scaling so the carried top-state norm
stays `O(1)`, ties input embedding and LM-head weights to encourage a shared
basis, and applies RMSNorm after fusion (Section 3.3 and Appendix Figure 9).

The experiment optimizer uses NorMuon for matrix parameters (`lr=1e-2`,
weight decay `0.01`) and Adam for other parameters (`lr=5e-4`, no weight
decay), with a WSD schedule, 200 warmup steps, a 25% cooldown, cooldown z-loss
`1e-5`, and AdamC-style joint learning-rate/weight-decay decay.  The data is
the Phi-4 mixture at context length `8,192`, normally with a 300K-token global
batch (Section 4).  The paper does not report exact accelerator hardware.

## Reported evidence

The recurrent schedules and their token-equivalent compute are reported as:

| training tokens | feedback-pass mixture | token-equivalent compute |
| ---: | --- | ---: |
| 100B | 75% one-pass, 25% three-pass | 150B |
| 200B | 75% one-pass, 22% two-pass, 3% three-pass | 256B |
| 400B | 75% one-pass, 22% two-pass, 3% three-pass | 512B |

The most informative ablation is recurrence-depth stability, not a fusion-form
ablation.  On the 1B/200B model, `75%` one-pass plus `25%` two-pass batches
works at its trained horizon but validation loss rises and state-update norms
oscillate when iterated further.  Replacing only `3%` of the mixture with
three-pass batches (`75/22/3`) keeps validation loss flat through 30 feedback
passes while `||h^(k)-h^(k-1)||` decays toward a plateau (Figure 3).  Appendix
Figure 10 reports stability through 1,000 passes.  The third pass is the first
one whose input is itself produced from a fused input, so this is evidence
that a small amount of genuine self-composition changes long-horizon behavior.

Figure 4 reports front-loaded prefill gains: most improvement arrives with the
first fused prefill, and two fused passes let the 100B feedback model reach the
200B standard baseline and the 200B feedback model reach the 400B standard
baseline on validation perplexity and averaged 5-shot LM Eval.  Appendix
Table 2 gives an exact 0-shot subset for the 200B model: average PIQA/OBQA/
ARC-E/ARC-C rises from `52.66` at zero feedback passes to `53.58` at one pass.

The strongest causal evidence is same-weight decoding, because it separates
the effect of using feedback from the effect of feedback training.  Figure 5
compares standard decoding, soft decoding (ordinary prefill plus recurrent
generation), and fused decoding (one extra fused prefill plus recurrent
generation).  At the 200B scale, the text reports MATH-500 approximately
`0.27 -> 0.37` under soft decoding, HumanEval `0.31 -> 0.34` under fused
decoding, and MBPP `0.38 -> 0.40` under fused decoding.  Soft decoding improves
over standard decoding across all four reported generation tasks at both
tested feedback-training scales; the 200B model approaches or exceeds standard
baselines trained on `2x`--`5x` as many tokens on selected tasks.

After 12B tokens of 8K-to-32K context extension and 6B tokens of instruction
tuning, Table 1 reports these exact percentages:

| scale / task | standard | soft feedback | fused prefill + feedback |
| --- | ---: | ---: | ---: |
| 200B GSM8K Pass@1 | 64.52 | **67.93** | 67.55 |
| 200B MATH-500 Pass@1 | 43.80 | **45.60** | **45.60** |
| 200B HumanEval Pass@3 | 42.54 | 45.06 | **45.92** |
| 200B MBPP Pass@3 | 38.39 | 39.80 | **41.22** |
| 400B GSM8K Pass@1 | 67.90 | 71.00 | **71.80** |
| 400B MATH-500 Pass@1 | 46.00 | 45.40 | **48.40** |
| 400B HumanEval Pass@3 | 46.50 | 47.20 | **47.60** |
| 400B MBPP Pass@3 | 40.50 | 40.60 | **41.70** |

The base model also produces shorter MATH-500 traces at equal or better
accuracy (Figures 6 and 8), but that effect disappears after instruction
tuning; the authors attribute this to off-policy verbose target traces.

Finally, Figure 7 probes accessibility rather than task output.  One recurrent
prefill step raises the layer-0 linear-probe accuracy to `99.6%` for completion
tracking and `100%` for delayed memory.  Multi-register experiments show that
full recurrence helps most as overwrite interference increases.  The paper
correctly cautions that linear accessibility does not prove causal use.

## Claim boundary and caveats

- The paper carries one temporally adjacent state.  It has no multi-slot
  address selection and therefore does not test our matched-donor identity
  problem.
- It reports no zero-state, wrong-state, matched-donor, or layer-permuted
  causal controls.  Standard versus soft decoding proves usefulness of the
  feedback regime, not identity specificity.
- The mandatory asymmetric GLU is motivated by its bypass structure, but the
  paper provides no additive-versus-GLU fusion ablation.  Depth scaling,
  RMSNorm, weight tying, and jitter also are not independently ablated.
- The model scale is limited to roughly 1B, and the pass schedule is explicitly
  described as heuristic.  Exact recurrence-onset timing is not specified.
- Exact accelerator hardware and a code/checkpoint release are not reported
  in the official v1 source.
- The 10B schedule row says `100%` three-pass but assigns 40B token-equivalent
  compute.  That is inconsistent with the paper's own definition, under which
  three passes cost `3x`, so it should not be used to infer scaling efficiency.
- Multi-pass prefill can double or further multiply prefill compute.  The
  under-`1%` claim applies to the per-token two-projection decode fusion, not
  to optional repeated prefilling.
- The stable fixed-point diagnostic is a proxy for decode-time behavior.  It
  is necessary evidence for self-composition but not a matched-state causal
  result.

## Comparison with existing DeepEmbed evidence

Our `addressed_moe_deepembed_ffn` branch already implements the paper's most
important local interaction pattern inside frozen Gemma's FFN:

```text
state = silu(W_down * rms(recurrent_control))
gate = sigmoid(W_gate * rms(hidden_query))
modulation = W_up * (state * gate)
channel_scale = 1 + gain * tanh(Gemma_up_proj(rms(modulation)))
```

Like full-bandwidth fusion, the recurrent state is a vector value and the
current hidden state is a vector gate.  DeepEmbed differs in three decisive
ways.  It modulates intermediate FFN channels rather than replacing layer-0
input; it preserves an exact unit-scale/projected-only bypass at zero state;
and it is installed sparsely at layers `(10, 21, 31, 41)` rather than renewing
a deep state's entire depth budget.

The locked causal endpoint demonstrates why the paper is an architectural
clue rather than an identity solution.  DeepEmbed made zero state much worse
than correct state (`+1.221520` CE) and layer permutation worse by `+0.231559`
CE, but the matched donor was slightly *better* than correct (`-0.003740` CE;
7/11 correct-positive rows).  Thus vector-valued state-by-query gating can
force state presence and layer sensitivity while remaining donor-neutral.  A
plain FBT GLU transplanted at another hook would repeat that unresolved
failure unless it receives explicit state-identity supervision.

## RWKV translation

### Retired: plain separable CrossGLU

A plain deep-to-shallow CrossGLU is not a new identity mechanism:

```text
value = U_s RMSNorm(state)
gate  = sigmoid(U_q RMSNorm(query))
bridge = W_o(value * gate)
```

This is the same separable state-times-query family already tested by
DeepEmbed.  Moving it earlier renews the state's downstream depth budget, but
does not make the gate depend jointly on the query/state pair.  A content- or
norm-matched donor can therefore use the same value channel without being
recognized as the wrong state.  The locked DeepEmbed donor failure is direct
evidence against spending another causal run on this form.  Plain separable
CrossGLU is retired and is not ranked as an active route.

### Retained rank-3 fallback: joint pair-gated CrossGLU

The only CrossGLU variant worth retaining makes the vector gate a function of
the query/state pair.  Seed its coordinate system from the passed
source/donor-component-disjoint bilinear maps, but do not collapse the mapped
vectors to one scalar cosine:

```text
q'_l     = map_query_l(answer_query_l)
s'_l     = map_state_l(addressed_raw_rwkv_read_l)
qhat_l   = RMSNorm(q'_l)
shat_l   = RMSNorm(s'_l)
joint_l  = concat(qhat_l * shat_l, abs(qhat_l - shat_l))
value_l  = U^s_l shat_l
gate_l   = sigmoid(G^joint_l joint_l)
z_l      = W^o_l (value_l * gate_l)
early_hidden = early_hidden + alpha_l * bounded(z_l)
```

`map_query_l` and `map_state_l` should be deterministically reconstructed from
the passed component-disjoint compatibility fit and frozen for the first
mechanics screen.  `map_state_l`, `U^s_l`, and `W^o_l` must be bias-free, and
the bridge must contain no query-only, hidden-only, or projected-value bypass.
An exactly zero recurrent state then maps to an exactly zero value and exactly
zero bridge correction even though the joint gate remains defined.  The target
projected carrier must remain byte-identical under every state intervention.

The addressed RWKV read remains the value path; the joint interaction only
controls which components of that value can pass.  `W^o_l` lifts the state-width
bridge back into Gemma's hidden width.  Injection at layer 0, or at the earliest
feasible frozen-residual anchor, gives a deep memory result a renewed downstream
depth budget.  This is the main benefit the current output gate and sparse
DeepEmbed path lack.

An exact replacement of Gemma's token embedding would be closest to the paper,
but it is too large a distribution shift for the present frozen-model,
eight-update regime.  The first bounded implementation should therefore use a
zero-initialized or tightly bounded residual lift.  This restores a projected-
only bypass, so the paper's structural anti-shortcut guarantee no longer
applies.  Compensate explicitly in the objective, not by increasing gain:

```text
L = answer_CE(correct)
  + lambda_d * relu(m_ce - CE(donor) + CE(correct))
  + lambda_p * relu(m_ce - CE(permuted) + CE(correct))
  + lambda_i * InfoNCE(q', correct_state, donor/permuted_states)
```

Gradients must reach the CrossGLU, the mapped query/state heads, and the RWKV
writer/reader selected by the finalized protocol; detaching the carried state
would remove the paper's future-loss supervision.  Zero state must still
produce an exact zero CrossGLU correction, leaving the locked projected-only
control byte-identical.  Randomize the memory-onset boundary in training, and
add small normalized state jitter only after the noiseless mechanics contract
passes.

This bridge is not an inverse binding guarantee.  Unlike rotary binding, a
donor state is not algebraically decoded in a wrong basis.  It instead combines
the bilinear route's empirical coordinate alignment with the paper's
full-vector, deep-to-shallow causal use.  Explicit donor and component-disjoint
contrast remains mandatory.

### Optional write extension

Do not change the online write in the first CrossGLU implementation.  If the
read-only bridge passes, a later write experiment may test:

```text
v'_t = (U^v_l v_t) * sigmoid(G^v_l a_t)
```

while leaving RWKV `k/a/b` and slot routing fixed.  This keeps the experiment
bounded and isolates whether address gating at storage helps.  A simultaneous
write/read change would confound state identity with altered state dynamics.

## Ranking against current routes

The component-disjoint bilinear cross-fit passed its internal screen, but its
authorized output-coupled causal endpoint failed: matched-donor CE was
`-0.000080` relative to correct and donor-positive rows were `0.386364`.
That family is retired; the cross-fit result is not a causal or benchmark gain.

Headwise rotary binding then failed its signed mechanics gate because its dense
rotation did not commute with RWKV's channelwise value-axis update.  The
algebra-safe diagonal-sign fallback was tested with the full 64-dimensional
projected slot key.  Its signed full-key run passed finite controls, carrier
invariance, write/read code matching, exact zero-state bypass, and BF16
cancellation (`0.000553`--`0.000816` maximum), but changed only `0.750`--`0.769`
of donor decoded rows (required `>=0.95`).  The diagonal-sign family is
retired without causal training.

The joint pair-gated CrossGLU bridge was the first Full-Bandwidth-inspired
causal candidate. Its signed four-A100 mechanics screen passed, but its locked
held-out causal endpoint failed donor identity: matched-donor CE was
`-0.001360` relative to correct, donor-positive rows were `0.045455`, and the
layer-permuted and zero controls were also slightly better than correct. The
route is therefore retired without gain, batch-size, learning-rate, or
duration tuning. Native generation was not authorized.

The learned identity-bound DeepEmbed follow-up also completed all eight
four-A100 updates after serializing positive and donor backward graphs. It
trained only 168 binder tensors while freezing the inherited address-keyed
writer and DeepEmbed adapter. Zero-minus-correct and layer-permuted-minus-
correct CE were strongly positive (`+1.301495` and `+0.285514`), but donor-
minus-correct CE was `-0.007757`; both donor-positive CE rows and binder-score
rows were only `0.4375`. The binder score itself preferred the donor by
`0.002040`. This closes another gate-shape route: mandatory state use and layer
sensitivity still do not imply source identity.

The exact-v5 answer-position shadow head then passed, but its predeclared
causal-predictor replication failed before recurrent mechanics. On the same
donor-component-disjoint `176/44` split, held-out donor-positive row fraction
was `0.954545`, while donor-positive token fraction was `0.878327` and mean
donor gap was `0.047801`; the locked gates were `0.95`, `0.95`, and `0.05`.
Layer-permuted token and row fractions were both `1.0`. This is a position
boundary, not evidence for threshold tuning: source identity is recoverable
after teacher-forced answer tokens but is not uniformly recoverable from each
preceding causal predictor state. Stage 2 was skipped and no weights were
trained.

| priority | route | evidence | main upside | main risk |
| --- | --- | --- | --- | --- |
| retired | projected-value/RWKV identity-bound DeepEmbed | eight updates passed; causal and binder donor gates failed | direct pair supervision on the strongest sparse FFN path | learned score remains donor-neutral |
| retired | joint pair-gated deep-to-shallow CrossGLU | mechanics passed; causal donor gate failed | pair-dependent full-vector causal use with renewed depth | donor-neutral endpoint and early-residual bypass |
| retired | bilinear compatibility + output gate | cross-fit passed, causal donor gate failed | held-out score alignment | donor-neutral causal endpoint |
| retired | exact-v5 per-token predictor shadow | row identity `0.954545`; token identity `0.878327`; mean gap `0.047801` | exact causal-position test with immutable live state | each predictor must rediscover prompt identity |
| retired | rotary / diagonal-sign binding | rotary non-commutation; sign donor rows `0.750`--`0.769` | algebraic cancellation in limited controls | insufficient donor specificity |

### Next directions after the CrossGLU failure

The paper's useful contribution is now a training recipe, not a drop-in
identity module. The best next move is an exact-source detached shadow-replay
mechanics screen on the causal-passing `address_keyed_moe_deepembed_ffn`
branch. Earlier combined routes did not always replay the same learned writer
and DeepEmbed feature generator, so the diagnostic must execute the learned
writer once, snapshot both RWKV and projected state, and keep those snapshots
immutable. Feed the previous pass's detached RWKV read back before query/read
formation, rerun only the read path through the same sparse DeepEmbed anchors,
and measure recurrence contraction (`delta_k`) through eight passes.

The required controls are target state plus target shadow, target state plus
matched-donor shadow, donor state plus donor shadow while retaining the target
answer, zero, cyclic layer permutation, row-shuffled shadow, norm-matched
random shadow, and shadow disabled. This is Jacobi-inspired mechanics, not the
paper's live-gradient Jacobi training: detachment deliberately isolates whether
renewed read depth can create donor-specific behavior without mutating or
double-writing memory. Only if correct replay beats every donor control and the
tail contracts may a separately locked two-pass experiment keep gradients
through pass 1 and apply answer losses on both passes plus per-layer donor
InfoNCE.

Other bounded directions, in descending priority, are:

1. a slot-codebook bridge that binds the projected slot value to the RWKV read
   with a learned discrete identity code and a donor/code swap control;
2. a confidence/abstention controller that falls back to projected-only when
   query-state agreement is low, evaluated on the locked 220-row native set
   only after causal donor specificity passes;
3. a multi-slot or chunked RWKV read that reduces overwrite interference while
   preserving the original writer and isolates state capacity from identity;
4. a write-side address code experiment, changing only the RWKV value feature
   with a frozen read path, followed by the same matched-donor endpoint.

Do not combine these directions in one run. Each must first pass mechanics,
then an eight-update four-A100 causal endpoint, before the protected native
benchmark can be reopened.

Do not combine the joint CrossGLU bridge with bilinear output gating or rotary
binding before an independent causal result.  Combining gates would make a
donor margin uninterpretable.

## Safe experiment gates

The mechanics-only fallback has now been run under the following sequence. If
the causal endpoint fails, retire this route without gain, batch-size, or
duration tuning.

### Mechanics, no model update

- exactly four distinct A100s and `HF_ENDPOINT=https://hf-mirror.com`;
- already-open fit rows only, with source and matched donor components split
  before capture;
- freeze the projected route/key, model, writer, and RWKV state dynamics;
- install only the finalized joint CrossGLU maps and bounded early lift, with no
  adapter saved;
- prove zero-state exact-zero memory output, finite values, non-saturated joint
  gates, and finite nonzero gradients through both state-value and joint-gate
  paths;
- require held-out donor pairwise separation at least `0.95`, mean state
  score gap at least `0.05`, and layer-permuted separation at least `0.95`;
- verify that projected-key tensors are byte-identical across correct, zero,
  donor, and layer-permuted interventions.

The mechanics screen must evaluate this factorial control matrix while keeping
the target answer and projected carrier fixed:

1. target query plus target state;
2. target query plus matched-donor state;
3. donor query/address plus target state;
4. donor query/address plus donor state while the target answer remains fixed;
5. target joint gate held fixed plus donor value;
6. joint gates shuffled across rows plus correct values;
7. norm-matched random or orthogonal state;
8. zero state;
9. cyclic layer-permuted state; and
10. bridge disabled / projected-only.

The donor-query/donor-state condition is critical: it is an internally
compatible but target-wrong pair, so worsening it rules out mere pair
consistency.  Fixed-gate donor-value and gate-shuffle controls separately test
whether both the RWKV value and joint gate are causally used.  Retire the route
if the wrong matched pair is not worse, either component swap is neutral, zero
state changes the projected-only output, the carrier changes, gates saturate,
or any output or gradient is non-finite.

The branch is retired if any mechanics gate fails.  Do not compensate by
raising gain, batch size, or training duration.

### Causal endpoint

Only a mechanics pass may authorize an eight-update causal run.  Use a fresh
source-and-donor-disjoint open-fit endpoint, serialized control graphs, and the
full factorial controls above.  Require finite logits, positive zero and
layer-permutation CE margins, donor CE margin at least `0.02`, donor-positive
row fraction at least `0.75`, and the same held-out identity gates.  The
donor-query/donor-state pair, fixed-gate donor value, and shuffled-gate correct
value must each be worse than the correct pair.  Native generation remains
blocked until all pass.

### Current mechanics receipt

The v22 run used exactly four A100s and `HF_ENDPOINT=https://hf-mirror.com`:

- result: `local_artifacts/natural_memory_native_rwkv_joint_pair_crossglu_mechanics_v22/result.json`;
- result SHA-256: `dba313f9a2a441ed1c81fac0a83cc42b4ca6f94975d8e8b5b483db654296bcc3`;
- receipt payload: `ba9a986ef0bc937c652e82204c8edb3efc80855136e1261c30c9fe15e3954ddd`;
- all mechanics checks pass, including exact zero/projected-only element equality,
  `1.0` matched-donor changed-row fraction, non-saturated gates, fixed-gate and
  shuffled-gate value paths, finite outputs, fixed carrier references, and live
  bridge gradients.

### Current causal receipt

The authorized v1 endpoint also used exactly four A100s and
`HF_ENDPOINT=https://hf-mirror.com`, with eight serialized optimizer updates.
Training invariants passed: all 126 bridge tensors had finite nonzero global
gradients on every update, all projected carriers stayed fixed, and zero-state
logits exactly matched the projected-only bypass. The signed endpoint failed
only the identity and causal-preference gates; its result is
`local_artifacts/natural_memory_native_rwkv_joint_pair_crossglu_causal_train_v1/result.json`
with receipt payload
`e7e007540e30976da22af006b4cab46dcdc0d3c02bbfec45685a6662931e1e3d`.

### Feedback-depth stability

The paper's multi-pass idea is transferable only if the RWKV branch actually
feeds its own state-derived representation into a later computation.  A
one-pass read gate is not a full-bandwidth recurrence.  If a later experiment
adds read feedback, measure

```text
delta_k = mean(||r^(k) - r^(k-1)||)
```

for `k=1..8` on open rows.  Require finite values and a contracting tail
(`delta_(k+1) <= 0.95 * delta_k` for the final three steps), with no CE
divergence.  Use prefix mixin at the write/read boundary and state jitter only
after the no-noise mechanics path passes.  Current RWKV state scans already
update online; blindly repeating the scan would double-write slots and is not
an equivalent FBT pass.  A read-only second pass or a separately snapshotted
state is required.

## Bottom line

Full-bandwidth transformer gives us two valuable design principles: identity
information should gate a full-vector state value path, and a deep memory
result should re-enter early enough to receive renewed computation depth.
Repeated self-composition should be trained and measured for contraction.
Plain and joint CrossGLU gates and the learned identity-bound DeepEmbed binder
all failed matched-donor causal gates, so another one-pass gate shape is not
the next move. The exact-source v5 shadow cross-fit screen subsequently passed
on a donor-component-disjoint `176/44` split: held-out matched-donor pairwise
accuracy was `0.954545`, its mean score gap was `0.103092`, and layer-permuted
accuracy was `1.0`. The signed result receipt is
`4ba137387216a8f2bc2c5562a764b4f340afa795cc4dbc88d4d2cf0ea470443c`.

That result establishes learnable identity in untouched v5 shadow features at
teacher-forced answer positions, not as causal model use. The causal-predictor
replication subsequently failed its token and mean-gap gates, so the detached
per-token shadow family is retired and recurrent mechanics remain unexecuted.

The next paper-inspired move is **identity transport rather than identity
recovery**. Derive a latch once from the causal prompt boundary, shift it across
answer predictors, and use that stable address to query immutable exact-v5
RWKV state. RWKV remains the material online key/value store; a sparse outer
FFN transforms the live read before early-layer fusion and shifted feedback
renews its computation depth. This is distinct from the paper's token gate and
must be established independently with wrong-latch, wrong-state, paired-wrong,
zero, permutation, shuffle, random, disabled, immutability, and eight-pass
contraction controls on a new pre-signed donor-disjoint split. The paper does
not establish matched-state identity, so any eventual native benchmark claim
remains separate from its single-state feedback evidence.
