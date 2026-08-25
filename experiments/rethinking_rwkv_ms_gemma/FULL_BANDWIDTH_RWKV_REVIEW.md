# Full-bandwidth transformer review for RWKV memory

Source: [arXiv:2608.08888v1](https://arxiv.org/abs/2608.08888v1),
*Full-bandwidth transformer* (Wang et al., submitted 2026-08-09).  This review
uses the official arXiv source. The follow-on mechanics screen is recorded
separately below; no protected split is opened by either document.

The official v1 PDF and source were rechecked on 2026-08-25. Their SHA-256
digests were `7ddb5869843aea15d72cc3fc94d69d15217a67c07effcd7a5e4631320cd52ee3`
and `6ac4149a537a5427019c39def1fc4647dd6a92db360ef7e8d1676c7602219bf8`,
respectively.

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
(Section 3.3, Eqs. 8--12 and Figure 2):

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
  `3%` three-pass heuristic mixture; the small three-pass fraction is reported
  to turn an unstable two-pass map into a contraction;
- use a random prefix mixin so plain prompt inputs and fused generated inputs
  share a training distribution;
- RMS-normalize both the token embedding entering the gate projection and the
  fused input entering the model, keep carried-state scale stationary, and add
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
Table 2 gives an exact 0-shot subset for the 200B model: the five-task average
over WinoGrande/PIQA/OBQA/ARC-E/ARC-C rises from `52.66` at zero feedback
passes to `53.58` at one pass.  The four-task average that excludes WinoGrande
is `50.715 -> 51.3325`.

The strongest causal evidence is same-weight decoding, because it separates
the effect of using feedback from the effect of feedback training.  Figure 5
compares standard decoding, soft decoding (ordinary prefill plus recurrent
generation), and fused decoding (one extra fused prefill plus recurrent
generation).  At the 200B scale, the text reports MATH-500 approximately
`0.27 -> 0.37` under soft decoding, HumanEval `0.31 -> 0.34` under fused
decoding, and MBPP `0.38 -> 0.40` under fused decoding.  Figure 5 plots 100B,
200B, and 400B full-bandwidth series, while the nearby prose says "both
scales"; across the plotted pretraining scales, the paper reports soft decoding
above standard decoding on every shown generation task.  The 200B model
approaches or exceeds standard baselines trained on `2x`--`5x` as many tokens
on selected tasks.

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

The paper's prose says both feedback modes improve all four post-tuning tasks,
but the table contradicts that claim at 400B MATH-500: soft feedback is
`45.40`, below the `46.00` standard result.  The coding temperatures are
selected separately for each method, and HumanEval/MBPP Pass@3 is estimated
from ten rollouts; Figure 5 provides no uncertainty bars.  Context extension
and instruction tuning use three forward passes throughout rather than the
pretraining `75/22/3` mixture.

The base model also produces shorter MATH-500 traces at equal or better
accuracy (Figures 6 and 8), but that effect disappears after instruction
tuning; the authors attribute this to off-policy verbose target traces.

Finally, Figure 7 probes accessibility rather than task output.  One recurrent
prefill step raises the layer-0 linear-probe accuracy to `99.6%` for completion
tracking and `100%` for delayed memory.  Multi-register experiments show that
full recurrence helps most as overwrite interference increases.  That probe's
"full recurrent" prefill is fully sequential across prompt tokens, not the
ordinary parallel multi-pass approximation.  The paper correctly cautions
that linear accessibility does not prove causal use.

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
  under-`1%` claim is the authors' analytical FLOP estimate for the per-token
  two-projection decode fusion, not a measured latency or throughput result;
  it does not apply to optional repeated prefilling.
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
| retired | two-axis bidirectional diagonal-sign binding | 64-row development exactness `0/64`; donor code separation `0.976190` per axis | intended exact full-matrix basis cancellation | write/state basis mismatch before mechanics |
| retired | PLMSC discrete write/query code | correct anchor agreement `0.433824`; complete-row agreement `0.058824`; donor collision `0.132353` | explicit write/read identity code | categorical collapse and no causal authorization |

### Historical directions after the CrossGLU failure

At that checkpoint, the paper's useful contribution had become a training
recipe rather than a drop-in identity module. An exact-source detached
shadow-replay mechanics screen was proposed on the causal-passing
`address_keyed_moe_deepembed_ffn` branch. The later exact-v5 shadow,
causal-predictor, prompt-latch, and PLMSC results supersede that priority; the
description is retained to document the sequence of decisions. The diagnostic
would execute the learned writer once, snapshot both RWKV and projected state,
and keep those snapshots immutable. It would feed the previous pass's detached
RWKV read back before query/read formation, rerun only the read path through
the same sparse DeepEmbed anchors, and measure recurrence contraction
(`delta_k`) through eight passes.

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

## Post-PLMSC decision

PLMSC tested the most direct discrete identity transfer between the exact-v5
projected write address and the causal prompt-boundary RWKV-7 receptance. Its
single locked four-A100 mechanics run failed before training: correct
write/query code agreement was `0.433824` per anchor and `0.058824` per
complete row against `0.95` gates. Matched-donor and cyclic layer-permuted
anchor collisions were `0.132353` and `0.139706` against maximum `0.03` gates.
The layer-10 query used only three codes, with one code covering `64.71%` of
rows. The causal 34-row split remained unopened, un-tokenized, and
un-forwarded. The signed result SHA-256 is
`b7dce00737c928abc13729b19e24ccfe803b9dce6dde62b9d9d944971a295544` and
the receipt is
`23c7cfdf0cdf0fb747010615cfe271ae7d7c0cddd7bd9a90401179033100fda7`.

This failure strengthens the boundary around Full-Bandwidth transfer. The
paper's asymmetric GLU can force a full vector to be used, but it cannot make
the write address and causal read query identify the same state. Adding its
feedback loop now would give a donor-neutral state more depth and make the
result harder to interpret. Identity mechanics must pass first.

### Priority 1: continuous query-aligned write conditioning

The next independent family should be continuous rather than categorical.
Fit a regularized low-rank map on fresh fit-only rows between the projected
write address `A` and the causal RWKV receptance `r`, freeze it, and map `A` to
a direction `d(A)` in RWKV's native 32-dimensional key axis. Use that direction
only in the right-axis write features:

```text
k^A = k + g_k * RMS(k) * d(A)
a^A = a * (1 + g_a * tanh(d(A)))
b^A = b * (1 + g_b * tanh(d(A)))
v^A = v
```

The native `r` remains unchanged, `v` remains the material value, and an
exactly zero address must be an exact no-op. The direct identity score is
`r^T d(A)`, so a target query is tested against the write identity on the same
axis that reads the recurrent matrix. This is smoother for unseen addresses
and directly targets the `r`-to-write geometry that PLMSC tried to quantize.
The precommitted map is bias-free reduced-rank ridge with rank `16` and ridge
`1.0`. No reusable prior-open diagnostic artifact exists in the worktree, so
the signed protocol treats these values as a disclosed precommitted design
choice rather than making a stronger provenance claim. They remain frozen
through the fresh retrieval gate.

This is distinct from the retired learned rank-2 `k/v/a/b` writer: it uses one
shared query-aligned latent map, preserves `v`, must pass retrieval before any
model mechanics, and freezes the map before the causal path. It nevertheless
remains high risk because it is adjacent to that failed family and still
depends on generalizing the write-address/query relation that collapsed under
PLMSC.

Pre-sign four passage-component-disjoint FIT partitions before capture: `64`
fit, `32` retrieval, `32` mechanics, and `32` causal source rows.  The FIT
dataset has its own SHA-qualified namespace, so PLMSC numeric row indices must
never be mixed into it. Exclude the complete normalized-passage/32-character-
shingle component closure touching any of the `98` sources already selected by
the bidirectional manifest, and keep every source/donor component in exactly
one partition. Capture only fit and retrieval rows. Make the projected-address
lifecycle explicit before fitting: immediately after
`_write_projected_kv_slots`, snapshot the selected keys and routes, materialize
one immutable write-address sequence, and pass that exact tensor to both the
`k/a/b` conditioner and its audit.  Never recompute the address from mutable
slot state after the scan.  A synthetic old-key-to-new-key mutation regression
must prove that conditioning and audit both use the new latched key.  First
require component-held-out target-versus-donor and layer-permuted retrieval
gates without changing the state. Only a retrieval pass may authorize one
four-A100 exact-update mechanics gate with target address/state, target address
plus donor state, donor address/state with the target answer fixed, address
permutation, zero address, state-only, prompt-only, row shuffle, norm-matched
random, disabled, finite outputs, and byte-identical projected carriers. A
mechanics pass authorizes only a separately locked causal endpoint; it does not
authorize native generation.

That reservation is now materialized as
`local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1`.
The manifest file SHA-256 is
`c437a7d1f2b850a730fe5b28a08ae32ba02678561bb1265a4eef55bda7f4d468`
and its canonical receipt is
`99a878493c3848c96624e2ad658842c99e69769b4a1721b5854ad25af8d0bee2`.
Default validation byte-read only `manifest.json`, `fit.jsonl`, and
`retrieval.jsonl`; mechanics and causal remained inventory-only.

### Continuous-write causal decision

The retrieval and mechanics stages passed, but the separately signed causal
endpoint rejected this family. Exactly four A100s completed eight FIT updates
over all `32` symmetric pairs, froze the `84` selected read-path tensors, and
opened the `32` causal rows once. Zero recurrence was worse than correct by
`+0.050419` token-weighted CE with `0.875000` positive rows, passing both locked
gates. Layer roll reached only `+0.019981` and `0.718750`; matched donor was
better than correct by `0.007690` CE and only `0.406250` of rows preferred the
correct state. The result receipt is
`5660251fc35005ee6cc054587d83bdd3069f22c52b9d9d7b440912fdbf71c0d0`.

This closes the unchanged-`v`, train-readout-only continuous-write family.
Address-conditioned `k/a/b` created material and identity-sensitive state
geometry, but that geometry did not become source-specific causal answer use.
Do not add a plain FBT loop to this failed snapshot: renewing depth would
amplify both correct and matched-donor states without supplying the missing
identity operation. Any future Full-Bandwidth transfer must follow a new
identity-bearing value path, consume one frozen RWKV snapshot, rerun only the
read path, and start a fresh mechanics/causal sequence before native access.

### Priority 2: exact monomial binding fallback

The algebraic fallback is a projected-address-derived signed permutation, or
monomial matrix, `M(A) = D(A) P(A)` on RWKV's value axis. For one state matrix
`S`, encode `S^A = M(A) S` and make the exact recurrent update commute with
that basis change:

```text
v^A     = M(A) v
keep^A  = P(A) keep
erase^A = P(A) erase
write^A = P(A) write
k^A, a^A, b^A, w^A = k, a, b, w
read(A) = M(A)^T S^A r = S r
```

Unlike the retired dense rotary binding, a monomial transform preserves the
diagonal left-axis coefficients by permutation. Unlike diagonal signs, it also
reorders value coordinates, so a wrong decoder applies a full signed
permutation rather than only sign flips. Correct cancellation is exact in real
arithmetic and can be checked in BF16 before any model run.

Its limitation determines the ranking. A paired donor state with its own donor
decoder also cancels exactly and returns to the existing donor-content path;
monomial binding therefore guarantees address mismatch sensitivity but does
not itself solve the paired-donor neutrality that has blocked the causal gate.
It is best used as an independent fallback and intervention-harness control if
continuous right-axis identity fails. Any claim would still be
"projected-address-bound RWKV value memory," not autonomous RWKV addressing.

### Priority 3: Full-Bandwidth read feedback

Only after one identity family passes both mechanics and its causal endpoint
should the paper's mechanism be added. Feed the full decoded RWKV vector into
an early layer as the mandatory value path, use the current token/query only as
the vector gate, keep gradients live across two- and occasional three-pass
training, randomize the plain-prefix/fused-suffix boundary, RMS-normalize the
fusion, and add jitter only after the noiseless path passes. Snapshot memory
and rerun the read path only; repeating the online scan would double-write the
state and is not Full-Bandwidth recurrence.

Measure both identity margins and long-horizon self-composition. The final
three of eight read-feedback deltas must contract without CE divergence, and
all wrong-address, paired-donor, zero, permutation, shuffle, random, and
disabled controls must remain interpretable. Four GPUs improve data-parallel
throughput, but the feedback passes remain sequential and their live-gradient
activations constrain per-GPU batch size.

## Bottom line

Full-bandwidth transformer gives us two valuable design principles: the
current token/query should gate a full-vector carried-state value path, and a
deep memory result should re-enter early enough to receive renewed computation
depth. An identity-dependent pair gate is our extrapolation, not paper evidence.
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

Prompt-latched identity transport was tested once on a new nested `132/44`
donor-component split that excluded the earlier held-out rows. It raised the
held-out donor mean gap to `0.054514` and retained `0.954545` row separation,
but token separation reached only `0.919414` against the precommitted `0.95`
gate. The family is retired without model mechanics or training.

PLMSC then tested write-time slot codes exactly once and failed its signed
mechanics gates with categorical collapse; no causal rows or model weights were
opened. The best next move is **continuous query-aligned `k/a/b` write
conditioning with `v` unchanged**.  The subsequent full-matrix bidirectional
diagonal-sign fallback also failed before mechanics: routed-code and
encoded-state exactness were `0/64`, while each axis separated `0.976190` of
donor pairs against the exact `1.0` gate.  Its receipt-valid result is
`local_artifacts/natural_memory_native_rwkv_bidirectional_sign_development_gate_v2/result.json`.

The continuous alignment's fresh retrieval stage subsequently passed. It fit
all `42` rank-`16` full64-to-causal32 maps on `64` source/donor-component-safe
FIT rows before opening the `32` retrieval rows. Donor-positive row fraction
was `1.0`, mean correct-minus-donor gap was `0.069070`, and both
layer-permuted row fraction and mean gap passed at `1.0` and `0.868476`.
Receipt
`cf001ac0f06afeb58b96084d656e5a22521a7d2229d68436b09572e231e0a6dd`
binds the result. This proves causal-boundary identity geometry under the
unchanged inherited exact-v5 write, not causal use by the continuous writer.
The subsequent separately signed mechanics gate has now passed on all `32`
mechanics rows using exactly four A100s. Correct continuous state differed
materially from raw (`0.793498` mean normalized L2), matched-donor address only
(`0.366313`), layer-rolled address only (`1.048170`), and target address on
donor content (`0.351101`); every comparison had positive row fraction `1.0`.
All exact lifecycle, override-byte, fixed-carrier, unchanged-`v`, zero/raw,
projected-only, read-basis, finiteness, and no-update gates passed. Result
receipt
`2621b0d7773f7931fda80676774697fcc4c059abf49f8ebbad683f19f34c1a95`
binds that mechanics pass.

The continuous-write causal endpoint subsequently failed matched-donor and
layer-roll gates, retiring that readout family. A four-A100 address-decoded
screen then tested `S d(A) -> projected value` on already-open FIT/retrieval
rows. Correct reconstruction was high (`0.912683` mean cosine), but donor and
wrong-address gaps were only `0.008718` and `0.009482`, and no module passed the
locked identity gate. Its signed receipt is
`c7df99f673e55b749b88e7b9f8a71967ed2fd4da28aa99d21fa9daf9b563c93a`.

That failure does not eliminate explicit address-keyed virtual K/V. Cosine is
blind to positive rescaling, so a rank-one-like state can contract a wrong
address into the same value direction. The next bounded architecture should
therefore use separate roles: an address-derived virtual key is tested by
attention-logit identity margins, while an RMS-normalized RWKV contraction is
the virtual value. Inject four ephemeral positions after cache retrieval and
before the attention softmax at full-attention layers `5/11/17/23`; never cache
them or mutate shared K/V. Exact-zero state must fall back to projected-only.
Only a fresh mechanics and causal pass can make FBT feedback eligible. Native
benchmark, Full-Bandwidth gain, and SOTA claims remain unauthorized.

The first explicit virtual-KV identity screen has now completed on the
already-open FIT/retrieval rows. A fixed post-RoPE/no-position bias-free ridge
map from the continuous address to Gemma K-space failed all four anchors.
Layer `23` was close but below the locked row gate: correct beat the strongest
wrong candidate on `0.718750` of rows with `+0.224652` mean attention-mass
margin. Layers `5/11/17` reached `0.531250/0.562500/0.375000`. Zero RWKV state
disabled the virtual path exactly on every row. This result does not reject
virtual K/V generally; it rejects the assumption that one position-agnostic
linear map can predict a post-RoPE attention key across varying prompt
positions. The next bounded key family should map address to an unrotated key
and co-rotate it at the current query position. Co-rotation makes the virtual
key/query relative phase zero, gives an explicit prompt-shift invariance, and
can be paired with a nonlinear contrastive FIT objective without mixing the
RWKV value payload into the selector.

The nonlinear co-rotated follow-up has now failed on frozen open-split
captures. Every seed reached `1.0` FIT top-1 at every anchor, but held-out
strict top-1 was only `0.125000`--`0.250000` at layer `5`,
`0.625000`--`0.718750` at layer `11`, `0.625000`--`0.750000` at layer `17`,
and `0.375000`--`0.500000` at layer `23`. No anchor passed the locked
all-seed gate. The result SHA-256 is
`30b4ba8c3be1b96da919acdb8f4980e96d98fe82677f65f844ef988656710d3a`
and its receipt is
`d2e4460f55b80b4c92ef76caacf84cfe9f8d2bae8cfaf82383fb949ad4d7a466`.
RoPE phase was therefore not the remaining bottleneck: the frozen native
attention query does not consistently encode the online-memory address, and
increasing address-only key-map capacity is retired.

This result sharpens where Full-Bandwidth feedback belongs. The paper's GLU
can force a selected state vector to become the next shallow input, and its
multi-pass objective can teach that vector to survive repeated
self-composition. Neither operation selects among source identities. The next
selector should instead reuse the already-passed continuous-write
receptance/address compatibility as an explicit additive bias only on the
ephemeral virtual-logit suffix. Real attention logits and cache state remain
unchanged; the RWKV contraction remains the virtual value. Call this an
explicit routed virtual KV, not native attention addressing. If its
wrong-address, matched-donor, permutation, zero, and causal gates pass, then a
Full-Bandwidth-style mandatory deep-to-shallow GLU becomes the next depth
renewal experiment, with the paper's `75/22/3` pass mixture, prefix mixin, and
contraction diagnostics. Until then, feedback would only give an
identity-ambiguous state more computation.

The first explicit local compatibility-bias screen then ran once on exactly
four A100s and failed its precommitted `3/4`-anchor gate with `2/4` passing.
Layer `5` reached `0.937500` strict four-way top-1 and `0.108299` mean margin;
layer `23` reached `0.750000` and `0.191626`. Layers `11/17` reached only
`0.656250/0.593750` strict top-1. All layer-permutation, candidate-permutation,
zero-address, and finite controls passed. Result SHA-256 is
`bf33dba5091ed9e9fc75e6289c5bb3d0da67a6fd6421c4afbf260419097993b8`
and receipt is
`cf5a1869a8ea770ab57d85aa19d48905fb2d6244fa0ab574ca2a2f7a4d1fd5a2`.
This closes same-layer frozen compatibility bias without authorizing live
attention. It also suggests a paper-compatible distinction: evidence can be
accumulated causally while moving upward through depth before a selected state
is renewed downward. A separately signed cumulative-depth router may test
that architecture, but post-selecting only layers `5/23` is not claim-grade.

The separately signed causal prefix-depth accumulator has now passed all four
anchors without changing maps, candidates, thresholds, or anchor inventory.
It averages compatibility evidence over `[5]`, `[5,11]`, `[5,11,17]`, and
`[5,11,17,23]` as the forward pass moves upward. Strict four-way top-1 was
`0.937500/0.875000/0.937500/1.000000`, mean strongest-wrong margins were
`0.108299/0.095830/0.098911/0.134506`, and every layer-permutation row passed.
The result SHA-256 is
`0ea1627415f0319931b86d0ce5ba5a255c5e738f6f22b88aec73c742f9dad73b`
with receipt
`fdf41009269b2cb71ca14285bbe687e50bb30207e5acc699c555cd9122262d59`.

This is the first identity result that changes the Full-Bandwidth ordering.
The selector may now enter a live mechanics screen as an ephemeral running
belief over source identity. Only after live wrong-state/wrong-address and
causal output gates pass should its selected RWKV value be returned to a
shallow layer through the paper's mandatory GLU. The current pass proves
open-tensor routing geometry, not causal model use or benchmark improvement.

## Post-live-mechanics decision

The authorized live eager-Q1 run preserved the selector's identity result but
failed the virtual-carrier invariants. All four anchors passed identity and
every declared state/address intervention materially changed predictor logits.
However, `11/32` inactive zero controls differed from provider-off when active
and inactive rows shared one virtual-suffix batch, and joint slot permutation
changed final logits on `5/32` rows (maximum absolute delta `0.828125`). Six
later-anchor bias-permutation checks also failed. Cached-Q1 replay, cache bytes,
native RWKV state, and the projected carrier stayed exact. The signed result
SHA-256 is
`23ce8601a84c388b8b1ea0a2bee527c9eb677dd67cf5b80a5f49f44e9deb7d58`
with receipt
`6dff6c59c7ee03d2a2ae775bc88ebd94877e8c80e06c9808f0b59e7eeb302a27`.

This retires the exact virtual-suffix carrier, not the cumulative identity
signal. The next distinct route should keep the identity computation outside
Gemma attention, canonically order candidates by source identity, combine the
selected raw RWKV read into a bounded zero-exact residual, and leave attention
width unchanged. If that route later passes both mechanics and a donor-specific
causal endpoint, the paper-faithful Full-Bandwidth experiment remains the same:
feed the router-enriched top hidden state through the mandatory top-to-bottom
GLU, snapshot RWKV memory so feedback never double-writes, use the `75/22/3`
pass mixture plus prefix mixin, and require contraction before native
benchmark access. The present failure authorizes none of those later stages.

## Source-residual depth-renewal development screen

The source-canonical residual route was screened on a new explicitly open
64-row development reservation before any fresh protected mechanics access.
The reservation excludes all 94 historical components, all 160 parent-selected
components, and all 64 components in the sealed fresh mechanics/causal
reservation. Its manifest SHA-256 is
`59b9926fa1023fa39f5616bdfa3f0bf4d1d4549d2f7e948dd628536a1bdb9f38`;
the protected mechanics and causal bundles remained unopened.

The first four-A100 screen held the selected-score gate at scale `32` and
compared residual injection after layers `5`, `11`, `17`, and `23`. This is a
bounded test of the paper's useful depth-renewal idea, not Full-Bandwidth
feedback: no state was returned across generated positions and no extra model
pass ran. Target-source top-1 was `0.875000` after layers `5/11` and `0.953125`
after layers `17/23`. Layer `23` had the largest mean strongest-wrong score
margin (`0.134644` versus `0.101257` at layer `17`), while layer `17` preserved
six more downstream transformer layers. Scale `32` saturated the selected-score
sigmoid: donor-address-only interventions were material on at most `0.046875`
of rows, so this setting was not eligible for mechanics.

The corrected scale screen therefore fixed injection after layer `17`, swept
`0.5/1/2/4/8/16`, and retained one layer-23 scale-4 comparison. Scales
`0.5/1/2/4/8` passed the open mechanics checks after applying the predeclared
half-BF16-step mass tolerance (`1/512`); the maximum observed selected-mass
equation error was `0.001941`. Scale `1` was the best diagnostic setting: every
declared intervention was materially visible, correct memory improved mean
target CE over provider-off by `0.012150`, and matched donor state+address was
worse than the target by `0.017373` mean CE. But donor-positive row fraction
was only `0.609375`, below the locked `0.75` gate. Scale `16` increased the
mean donor margin to `0.027597` but reduced donor-positive rows to `0.562500`
and donor-address materiality to `0.328125`. The layer-23 scale-4 control was
worse still: donor mean margin was `-0.012367` and address materiality was
`0.906250`.

The signed raw result SHA-256 is
`5eba745db6b245d5df5e8a2f16d058f41a162b2f551f958041e412ab1cd1abf9`
with receipt
`768c88849c97017469d72af928eaef1dc01b934ec0004b69f89a983a9992e831`.
The BF16-corrected analysis SHA-256 is
`02e02590556adebdcd1c9ea89acb1aab7f3bb2b71efadb322f42ddd4e81ee9b0`
with receipt
`1aebad03bb9610e86533fd16683a319d64f30cf2b01cbba03a4d322465b69029`.

This closes pure depth placement and scalar-gate calibration. The router has
strong source selection, and Full-Bandwidth-style earlier renewal makes the
selected state materially influential, but neither makes the untrained native
RWKV read consistently helpful for the target answer. The next bounded route
should freeze the layer-17, scale-1 selector and train only a small state-valued
outer FFN with correct-versus-donor, layer-roll, and zero-state contrast on open
development data. It must retain an exact-zero state path and may reach the
fresh mechanics gate only if held-out donor-positive rows pass `0.75`. Exact
top-to-bottom Full-Bandwidth feedback and its multi-pass schedule remain later
experiments, after source-specific causal use exists.

## Source-bound outer-FFN open-heldout result

The first outer-FFN experiment split the 32 explicitly open reciprocal donor
pairs into source- and pair-disjoint `16/16` train/heldout halves. It froze the
backbone, adapter, RWKV writer, compatibility maps, projected carrier, hard
selector, and scalar memory mass. Only a bias-free `32 -> 32 -> 2560`
state-valued correction was trained. The correction used the selected native
RWKV read as its mandatory value and the layer-17 hidden query as a vector gate;
zero state remained an exact zero residual. Exactly four A100s averaged one
training row per rank for 32 optimizer updates. The signed result SHA-256 is
`7ead31468793225f71110215ab59425a5ada977adede3d83d4da94683ee5e592`;
the checkpoint SHA-256 is
`c8fa785788b3eb1922b56d704a3bdc17e62c4593b25a8b7c77a211d6ea48d37c`.
No protected mechanics, protected causal, or native benchmark row was opened.

The heldout mechanics evidence improved substantially. All mechanics and
exact-zero checks passed, every declared intervention was material, target
selection was `0.9375`, correct memory improved mean target CE over provider-off
by `0.230696`, matched donor state+address was worse by `0.045821` mean CE, and
layer-rolled state+address was worse by `0.191366`. But matched-donor-positive
rows were only `0.59375`, below the locked `0.75` gate, so the family was not
promoted.

The training trace exposed a more specific limitation than outer-FFN capacity:
every update optimized token ID `105`, the common opening token of the serialized
JSON answer. In reciprocal pairs, the first target/donor answer difference
usually occurs six to eight supervised tokens later at `[]` versus `[` or at a
boundary-number token. The next bounded run must retain the same split and
architecture but place correct/donor/layer contrast at each pair's first
divergent supervised answer token. It must still pass the original first-token
heldout gate as well as a separately reported divergent-token gate before any
protected access.

That divergent-token run completed all 32 four-A100 updates with the same
frozen architecture and exact-zero residual contract. Every declared trainable
tensor received finite gradients under the step-1/step-2 contract. The signed
result SHA-256 is
`ecd0d14d0b1ea2f295777a349862b885d435fec02545ec7edc04edeec37f86ab`;
its receipt is
`e367f7560044ed8f3a53e19d218690de0389d73f63024c2953a58a398df8dd5d`,
and the checkpoint SHA-256 is
`bedfb5a92bcf0e4482f73d7e582e69038c4f1bbf457a110a05f7b88790eae563`.
No protected mechanics, protected causal, generation, or native benchmark row
was opened.

Moving the objective to the first divergent token helped that token but did not
solve identity. On the divergent-token heldout view, correct memory improved CE
over provider-off by `0.028627`, but matched donor was worse by only `0.010514`
mean CE with `0.59375` positive rows, and layer roll was worse by `0.029577`
with `0.625` positive rows. More importantly, the live selector chose the target
source on only `0.40625` of rows. On the original first-token view, target
selection remained `0.9375`, but correct memory regressed below provider-off by
`-0.017853`; donor margin was `0.016081` with `0.625` positive rows and layer
margin was `0.005397` with `0.46875` positive rows. Both heldout views therefore
failed the locked gates and the route was not promoted.

This localizes the next problem to answer-phase identity transport. The source
selector is accurate at the prompt/answer boundary, but recomputing it several
teacher-forced answer tokens later loses the selected source. Full-Bandwidth
Transformer suggests carrying a deep result across autoregressive steps, but a
generic previous-top-state GLU would carry an already donor-neutral mixture and
would not identify which memory was selected. The next distinct architecture
should instead latch the prompt-boundary selected address and native RWKV read,
then make their answer-token use depend jointly on the current causal query:

```text
a_star, m_star = latch_prompt_boundary_selection(addresses, rwkv_reads)
q_t            = map_query(answer_hidden_t)
joint_t        = concat(q_t * a_star, abs(q_t - a_star))
gate_t         = sigmoid(G_joint(joint_t))
residual_t     = W_out(U_state(m_star) * gate_t)
```

The latched RWKV read remains the only material value path; all state-side maps
remain bias-free so zero state gives an exact-zero correction. Correct latch,
matched-donor latch, address-only swap, state-only swap, layer roll, shuffled
latch, zero state, and provider-off must be evaluated while the projected
carrier stays fixed. A fresh component-disjoint open split must require target
selection, donor-positive rows, and layer-positive rows of at least `0.75`, plus
mean donor margin of at least `0.02`, before protected access. This is a
paper-guided persistent identity carrier, not yet Full-Bandwidth feedback: it
does not feed the model's top hidden state to layer 0, run extra transformer
passes, or claim the paper's depth-renewal result. Exact temporal feedback and
the `75/22/3` schedule remain gated on first establishing donor-specific causal
use of the latched carrier.

## Prompt-latched joint-identity open-heldout result

The proposed prompt latch and joint gate were then tested on the already-open
64-row development reservation. The route latched the prompt-boundary source,
masked every other source during answer-token use, and trained only three
bias-free tensors. Per-anchor identity features were
`concat(qhat * ahat, abs(qhat - ahat))` at layers `5/11/17`; the selected
32-dimensional native RWKV read remained the only material value. The first
launch exposed a runtime defect rather than an architectural result: the
post-FFN hook returned the base tensor when a trainable residual was
numerically zero, severing the zero-initialized correction graph before update
1. Commit `55fd3f2` makes the hook bypass only graphless zero residuals and
adds a regression test; inference-time exact-zero behavior is unchanged.

The fresh `v2` launch used exactly four distinct A100s through
`HF_ENDPOINT=https://hf-mirror.com`, completed all 32 updates, and passed every
staged-gradient, finiteness, bounded-residual, cache/state, mechanics, and
exact-zero control. The result SHA-256 is
`819cb586acbbc4f391256048cf1bd38a774237d46ae3398fbd9c204983cb5746`,
its receipt is
`25fc989427c5acb56f3c78f681b3ee508dc3d2f24268c7f8b464648915548563`,
and the 89,088-parameter checkpoint SHA-256 is
`a8f4012020c355b751872865c72ea34d441c7dbd624fa0fa30bd56497e9b6e24`.
No protected mechanics, protected causal, generation, or native benchmark row
was opened.

Prompt latching repaired answer-phase source transport but not causal value
identity. Target selection was `0.937500` on both heldout views. On the
original first-token view, however, correct memory was worse than provider-off
by `0.053084` mean CE; matched donor was better than the target by `0.020373`
with only `0.375000` target-positive rows, and layer roll was better by
`0.021550` with `0.437500` target-positive rows. On the divergent-token view,
correct memory was worse than provider-off by `0.004891`; matched donor was
better than the target by `0.000756` with `0.468750` target-positive rows, and
layer roll was better by `0.005354` with `0.593750` target-positive rows. Both
locked gates failed, so this terminal-read joint-identity family is retired
without gain, duration, batch-size, threshold, or learning-rate tuning.

This result sharpens the Full-Bandwidth transfer boundary. Carrying a selected
source across answer tokens solves the observed selector drift, while a joint
address/receptance gate over one terminal native read still does not supply a
donor-specific answer value. A paper-faithful top-hidden feedback loop would
renew depth for that unreliable value and cannot be justified as an identity
repair. The next open-only family should move the learned operation inside the
associative read rather than add another gate after a bad terminal read. Latch
the prompt-selected full RWKV matrices at layers `5/11/17`; at each anchor use
a low-rank hidden-to-32 delta to form a learned native query, contract that
query with the selected matrix, concatenate the three block-normalized reads,
and make that 96-dimensional bundle the mandatory value side of a bias-free
bounded lift. Hidden state may form the query but must have no output bypass,
so exactly zero matrices still give exact provider-off behavior. A frozen-query
multi-anchor bundle is the required bandwidth ablation, not the primary route.
Both heldout views must pass before temporal feedback, a fresh protected split,
or native benchmark access.
