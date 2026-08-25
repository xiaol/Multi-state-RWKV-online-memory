# Multi-State RWKV Online Memory

Mechanism-level experiments for comparing Dynamic Linear Attention (DLA) with
RWKV-style online memory under controlled state and boundary policies.

HF checkpoint:
[`xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1`](https://huggingface.co/xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1)

## Latest Trained-Model Result

The frozen Gemma4 + projected-slot outer-memory system now passes its
preregistered native publisher-validation gate. The adapter is implemented in
the repository's RWKV-MS-capable Delta-Mem runtime, but its active
`memory_readout_mode=projected_kv_slots` bypasses the recurrent RWKV matrix
scan. The decoder and all task routing rules were locked on publisher-TRAIN-
derived development data before the validation split was opened. Evaluation
used the identical frozen
`google/gemma-4-E4B-it` comparator, greedy decoding, write-then-read online
memory, the HF mirror, and four A100 GPUs.

| Native task | Frozen Gemma base | Locked outer-memory system | Delta |
| --- | ---: | ---: | ---: |
| Attribution candidate accuracy (29 rows) | 0.8966 | 0.8966 | +0.0000 |
| Narrative unit accuracy (39 rows, 1,449 units) | 0.6432 | 0.6467 | +0.0035 |
| Scene-boundary micro-F1 (170 rows) | 0.1820 | 0.2727 | **+0.0907** |

All three tasks met the `>=0.95` coverage floor, no task regressed, and two
tasks improved. The scene result is the main effect: memory reduced false
positives from 698 to 171, while true positives changed from 87 to 54 and false
negatives from 84 to 117. Narrative gained five correct units. Attribution is
preserved exactly by using the frozen-base candidate-likelihood scorer.

This is a system-level result for the locked decoder, not a claim that raw
memory improves every task. Its fixed task policy is:

- attribution: frozen-base candidate likelihood;
- narrative: use memory only for the preregistered
  `base=narration, memory=scene_description` label pair;
- scene: use the projected-slot memory generation directly.

The reported scope is 238 rows. Attribution source row 0 was excluded in the
protocol before the final run because it had already been touched by historical
runtime diagnostics. Publisher test and Hard32 remain unopened.

### Native Mechanism Accounting

The native validation, checkpoint-16, repair, and consistency-router results
above and below establish a system-level gain from learned online outer memory.
They do **not** establish that RWKV recurrence caused that gain. The frozen V9
adapter declares `memory_backend=rwkv_ms`, but its
`memory_readout_mode=projected_kv_slots` forward branch writes and reads four
content-addressed projected key/value slots per wrapped layer without invoking
the recurrent `_rwkv_ms_scan` path. In the native one-shot benchmark, one
last-valid-token proposal is written per layer; the four-slot capacity matters
for sessions containing multiple write calls.

Accordingly, throughout the native-result sections:

- “online state” means the complete captured adapter state, whose active signal
  is the projected-slot key/value, occupancy, and surprise bundle;
- “memory gain” means a gain from that projected-slot Q/O adapter and its fixed
  task policy;
- “RWKV recurrence gain” is not claimed and requires a new matched experiment
  in which recurrent state materially contributes to the readout, with verified
  recurrent-state mutation and correct-state versus zero/donor/permuted-
  recurrent controls that hold projected slots fixed.

This correction changes the mechanism attribution, not the signed predictions,
metrics, validation split discipline, or accepted system-level result. The
CPU RWKV-7 studies, recurrent tau2 experiments, and recurrent GGUF runtime
documented later are separate paths and are not reinterpreted by this note.

Reproducibility evidence:

- [Locked publisher-validation protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_publisher_validation_protocol_v1.json)
- [Signed validation decision](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_publisher_validation_v1/decision.json)
- [Signed metrics and artifact hashes](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_publisher_validation_v1/result.json)
- [Validation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_publisher_validation.py)
  and [hash-bound analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_publisher_validation.py)
- Independent replication seeds R12 and R13 each passed all 52 sealed checks:
  [R12](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_gate_replication_r12_sealed_run_split20260825_seed53/evaluation.json)
  and [R13](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_gate_replication_r13_sealed_run_split20260826_seed54/evaluation.json)

### Next Recurrent Goal

The recurrent-only candidate remains rejected: despite finite nonzero output
gradients in all 42 layers, its one-update BF16 correct-versus-zero and correct-
versus-donor final-logit deltas were exactly `0.0`. The replacement is now a
carrier-controller hybrid. Projected KV slots remain the material retrieval
carrier, while recurrent RWKV state modulates that carrier through a bounded
readout that is exactly projected-only when recurrent state is zero.

A locked four-A100 screen tested residual, vector-gate, and scalar-gate
equations at gains `0.03125`, `0.0625`, and `0.125`. All nine candidates passed
the fixed-carrier BF16 causal gates. The preregistered rule selected
`scalar_gate_g003125`: the lowest gain and, at that gain, the smallest worst-
rank perturbation from projected-only (`1.3789` maximum absolute logit delta).
Its minimum correct-versus-matched-donor recurrent delta was `1.3438`.

The selected hybrid then passed a separately locked one-update BF16 calibration.
All 42 recurrent output gradients were finite and nonzero, the global gradient
norm was `0.2388`, and both the full adapter and recurrent output weights
changed. After the update, correct-versus-zero recurrent deltas ranged from
`1.2812` to `1.9375` and correct-versus-donor deltas from `1.2969` to `1.8750`
across the four ranks. Zero recurrent state remained exactly equal to the
projected-only readout, with projected state byte-identical across interventions.

This established BF16-visible, trainable perturbation by RWKV state, but the
subsequent three-seed native benchmark showed that perturbation was not the
cause of the measured gain. The selected hybrid beat fresh projected-only
controls by mean scene micro-F1 `+0.00730`; two of three seeds were nonnegative,
and `8.48%` of paired outputs changed. The per-seed gains were `+0.02236`,
`+0.00281`, and `-0.00326` for seeds 57, 58, and 59.

The recurrent causal gates failed. Mean correct-state minus zero-state micro-F1
was `-0.00192`, correct-state minus matched-donor was `-0.00227`, and correct-
state minus layer-permuted was only `+0.00052`, below the locked `+0.005`
margin. Zero recurrent state exactly reproduced the projected-only bypass and
the projected carrier remained byte-identical across interventions. The valid
claim is therefore: **the trained carrier-controller hybrid improves over a
fresh projected-slot control on this authorized native benchmark, but correct
RWKV recurrent state did not cause the improvement.** The signed result status
is `native_benchmark_gain_without_recurrent_causal_pass`; its canonical receipt
is `7cd97cf939012c831bff96cdcc5fcfcf52ad3f626409d339814798cfa3c0d397`.

The first information-bottleneck replacement, `addressed_value`, has now been
tested. Projected keys supplied address/location information, projected values
were excluded from the output path, and the complete bounded memory value came
from the selected RWKV matrix. Zero recurrent state therefore produced exactly
zero memory read, and zeroing every projected value was bit-exactly inert. The
lowest-gain candidate (`0.03125`) passed the four-A100 structural screen: the
minimum correct-versus-zero, donor, and layer-permuted maximum logit deltas were
`1.375`, `1.28125`, and `1.3125`. Its separately locked one-update calibration
also passed, with finite nonzero recurrent-readout gradients in all 42 layers
and global gradient norm `0.008994`.

Eight causal-contrast updates completed, but their mean training margins did
not stabilize in the desired direction: zero-minus-correct CE was `-0.00467`,
donor-minus-correct was `-0.00596`, and layer-permuted-minus-correct was
`+0.00148`. On the authorized 220-row publisher-TRAIN-derived native
development partition, correct recurrent state scored `0.19287` micro-F1,
versus `0.19135` for zero/empty, `0.18873` for a matched donor, and `0.19215`
for layer-permuted recurrence. Every comparison was directionally positive,
but the margins (`+0.00152`, `+0.00414`, and `+0.00073`) all missed the locked
`+0.005` threshold. The valid status is therefore
`addressed_value_native_gain_not_established`, not recurrent native success.

The likely limitation was address/write misalignment rather than insufficient
readout magnitude. A native write normally created one projected proposal in
projected slot 0, while the recurrent scan wrote successive 128-token chunks
to RWKV slots 0 through 3. The addressed read consequently tended to query
RWKV slot 0 even when later chunks occupied other matrices.

The replacement `chunk_addressed_value` now creates one projected key for every
RWKV slot touched by a write. Each key comes from the last contextual hidden
state assigned to that exact recurrent slot; projected values are zero and
remain excluded from the output path. An initial screen execution correctly
failed its alignment audit because an inherited helper reset the write mode to
the old residual hybrid. The corrected fresh execution placed two projected
keys alongside the two nonempty recurrent chunks in every layer and rank. The
lowest gain (`0.03125`) passed with minimum correct-versus-zero, donor, and
layer-permuted maximum logit deltas of `1.28906`, `1.21875`, and `1.21875`.

The separately locked one-update calibration also passed. All 42 recurrent
readout gradients were finite and nonzero, the global gradient norm was
`0.008295`, recurrent output weights changed, and exact slot alignment,
zero projected values, zero-state equivalence, and projected-value independence
all survived the update.

Matched eight-update causal training then completed. Relative to the earlier
single-address method, its mean donor-minus-correct CE improved from `-0.00596`
to `-0.00078` and layer-permuted-minus-correct improved from `+0.00148` to
`+0.00403`; zero-minus-correct remained negative at `-0.00471`. These internal
improvements did not transfer to generation. On the same authorized 220-row
native development partition, correct recurrent state scored `0.19222`
micro-F1, versus `0.19135` for zero/empty, `0.19350` for the matched donor, and
`0.19314` for layer-permuted recurrence. The locked margins were therefore
`+0.00087`, `-0.00128`, and `-0.00092`, all below `+0.005`. The signed status is
`chunk_addressed_value_native_gain_not_established`.

Exact external chunk addressing was therefore not the way out. The next hybrid,
`recurrent_value`, removed projected addressing from the read path entirely.
RWKV's own cosine router scored all nonempty recurrent slots, and the bounded
read was `0.03125 * tanh(read / rms(read))`. Zeroing projected keys, values,
occupancy, and surprise was bit-exactly inert, while zero, donor, and layer-
permuted recurrent interventions were material on every A100 rank.

The one-update calibration passed with finite nonzero recurrent-output
gradients in all 42 layers and global gradient norm `0.009567`. The first
causal-training preflight then correctly rejected 42 inactive
`projected_kv_key_proj` tensors: they had remained in the optimizer even though
the architecture could not read them. Protocol v2 froze exactly that read-inert
family. Its fresh preflight had zero inactive trainables, and the fixed eight-
update run completed with every recurrent gradient and carrier audit intact.
The mean teacher-forced margins nevertheless remained negative:
zero-minus-correct CE was `-0.00406`, donor-minus-correct was `-0.00549`, and
layer-permuted-minus-correct was `-0.00243`.

The authorized 220-row native generation result also failed. Correct recurrent
state scored `0.18967` micro-F1, versus `0.19135` for zero/empty, `0.19195` for
the matched donor, and `0.18936` for layer-permuted recurrence. The locked
margins were `-0.00169`, `-0.00229`, and `+0.00031`, all below `+0.005`; the
signed status is `recurrent_value_native_gain_not_established` with receipt
`1eecbb4a345e4bee390025089082757f7981e7147965e655d7c382952cf078b7`.

The internal-router follow-up tested that hypothesis. A top-1 straight-through
router became essentially one-hot, but its discrete forward route collapsed the
useful optimization path. Top-2 routing preserved a differentiable mixture and
passed a one-update contrastive calibration. Longer AdamW variants then failed
deterministically at update 2 with non-finite per-row gradients, while an SPSA
variant completed all eight updates but failed the held-out causal endpoint:
zero-minus-correct, donor-minus-correct, and layer-permuted-minus-correct CE were
`-0.00102`, `+0.00454`, and `-0.00510`.

Row-isolated gradient checks found one reproducible numerical culprit, source
ordinal `1291`. Filtering that whole row removed the update-2 instability.
Positive-only training completed eight updates with 63/64 accepted rows but
failed all three fresh endpoint margins (`-0.00825`, `-0.00748`, `-0.00904`).
Filtered direct recurrent-value contrast training also completed, yet only the
zero control was weakly positive: `+0.000085` zero-minus-correct, versus
`-0.00545` donor-minus-correct and `-0.00843` layer-permuted-minus-correct.
This is evidence that the direct recurrent vector is not a reliable standalone
material carrier, even after its numerical training failure is isolated.

The current candidate therefore restores projected KV as the material carrier
and gives RWKV a narrower, causally testable role:

```text
prompt hidden states
  |-- projected K/V writer --> four content-addressed carrier slots --|
  |-- RWKV-7 scan ----------> four recurrent 128-token states --------|-->
query hidden state --> temperature-16 detached top-2 RWKV router ------|
                                                                        |
projected_read * (1 + 0.03125 * clamp(cos(projected_read,
                                           recurrent_read), -1, 1))
  --> learned content gate --> Gemma attention output --> decoder
```

Thus RWKV is active, but it is not the retrieved value template: it writes the
recurrent matrices, routes over them, produces `recurrent_read`, and controls a
bounded scalar rescaling of the projected carrier. Zero recurrent state makes
the cosine term zero and is exactly projected-only. Projected and recurrent
states are separately captured so zero, matched-donor, and cyclic layer-
permutation interventions can change RWKV while leaving every projected carrier
byte-identical.

The selected scalar-agreement configuration uses temperature `16`, top-2
routing, detached route scores, gain `0.03125`, and content-gate initialization
`0.25`. Projected carrier and recurrent router tensors were frozen; 210 stable
readout tensors were trained for eight row-isolated contrastive updates. All
64 rows were accepted. On a fresh 32-row teacher-forced endpoint it passed:
zero-minus-correct CE was `+0.29294`, donor-minus-correct was `+0.00746`, and
layer-permuted-minus-correct was `+0.000474`. This establishes held-out causal
preference, not yet native generation gain.

The signed adapter config retained the constructor's earlier
`rwkv_ms_hybrid_mode=recurrent_value` enum even though the signed training model
audit ran `scalar_gate`; the candidate switch changed runtime attributes rather
than the serialized config object. The locked generation evaluator discloses
and hash-binds this mismatch, restores only that non-parameter runtime enum, and
does not reinitialize any learned tensor.

The matched 220-row open native generation benchmark has now completed under
exact zero-state/projected-bypass batch-shape controls. Correct recurrent state
scored `0.18986` micro-F1, versus `0.18979` for zero/projected-only, `0.18710`
for the matched donor, and `0.19149` for layer-permuted recurrence. The locked
margins were therefore `+0.000064`, `+0.00275`, and `-0.00163`; all missed the
required `+0.005` gate. The valid signed status is
`scalar_agreement_native_gain_not_established`, with receipt
`9590428d136660e378b0b92ce79fcde5d46b7518314f9a629a04d8a9966c2e9e`.
Correct recurrence changed `9.55%` of outputs relative to projected-only, but
the changes did not improve native F1 or beat layer permutation, so no native
RWKV recurrence gain is claimed.

The elementwise vector-FiLM follow-up then trained and passed its locked
teacher-forced causal endpoint. Its zero-minus-correct, donor-minus-correct, and
layer-permuted-minus-correct CE margins were `+0.30025`, `+0.00393`, and
`+0.01053`. On the matched 220-row native generation benchmark it established
a real gain over its own fixed carrier: correct recurrence scored `0.19426`
micro-F1 versus `0.18763` for both zero recurrence and the explicit
projected-only bypass, a `+0.00663` margin above the locked `+0.005` gate.

That is not yet a recurrent causal pass. The matched donor scored `0.19054`, so
correct-minus-donor was only `+0.00372`; cyclic layer permutation scored
`0.19495`, beating correct recurrence by `0.00069`. Correct recurrence changed
`10.00%` of outputs relative to projected-only, and zero/projected-only outputs
were exact while every projected carrier stayed byte-identical. The valid
status is therefore `vector_gate_native_gain_without_full_causal_pass`, not
native RWKV causal gain. Its signed receipt is
`9fcbbd11ba502fdab77bee6c1177a5f5296cd4ff6ebecd0acdb3ce02b4cd10af`.

The locked generation protocol contains two non-operative wording errors: one
disclosure sentence says `scalar_gate` instead of the executed `vector_gate`,
and another says generation sets the gain even though `0.125` was already
serialized and only verified. The signed result carries both errata. Operative
architecture fields, the fusion equation, training audit, evaluator assertions,
and every prediction record consistently bind the executed mode to
`vector_gate`; no learned tensor changed during restoration.

The result narrowed the remaining problem from carrier gain to state
specificity. The next bounded hybrid tested an alignment-gated residual:

```text
a = clamp(cos(projected_read, recurrent_read), -1, 1)
projected_read
  + gain * rms(projected_read) * a
         * tanh(recurrent_read / rms(recurrent_read))
```

The locked four-A100 structural screen passed. Zero recurrence was bit-exactly
projected-only, every projected carrier stayed fixed, and all four ranks showed
material correct-versus-zero, donor, and layer-permuted logit changes. The
separately locked eight-update run also passed every training-integrity gate:
63/64 rows were accepted, only the previously identified unstable ordinal
`1291` was discarded, all 42 recurrent output tensors had finite nonzero first-
update gradients, and both the recurrent and full trainable subsets changed.

The fresh 32-row teacher-forced endpoint rejected the equation. Correct
recurrence beat zero recurrence by `+0.289878` CE, but lost to the matched donor
by `-0.002166` and to cyclic layer permutation by `-0.007990`. The signed status
is `alignment_residual_heldout_failed_generation_blocked`, with receipt
`5f9a2b9ce10e5cd55537cb1bebdfe249494627dc3709e8006402c21e44b867f5`.
This is a useful negative result: cosine alignment amplified dependence on a
nonzero recurrent vector but did not identify the correct row or layer. No
native generation benchmark was opened for this method.

The next bounded hybrid keeps the vector-FiLM operation that produced native
carrier gain and uses alignment only as its verifier:

```text
a = clamp(cos(projected_read, recurrent_read), -1, 1)
projected_read *
  (1 + gain * a * tanh(recurrent_read / rms(recurrent_read)))
```

This `aligned_vector_gate` preserved exact projected-only identity for zero
recurrence and passed its locked four-A100 structural screen on every rank. Its
minimum correct-versus-zero, donor, and layer-permuted maximum logit deltas
were `1.3125`, `1.1875`, and `1.328125`, with every perturbation bounded below
`2.0`.

The eight-update training run passed all integrity checks with 63/64 accepted
rows and only ordinal `1291` filtered. On its new 32-row endpoint, correct
recurrence again beat zero by a large `+0.259662` CE margin, but lost to the
matched donor by `-0.004416` and to cyclic layer permutation by `-0.004879`.
The signed status is `aligned_vector_gate_heldout_failed_generation_blocked`,
with receipt
`5f4a0f233e8b74a7d5ca17376954144e235e65db9bc2330ef11c8afc7581fff4`.
No native generation benchmark was opened.

Two different alignment fusions therefore reproduced the same initial
diagnosis: the recurrent branch strongly signaled memory presence, but eight
updates did not make its correction identify the correct row and layer. A
preregistered specificity-focused run kept the aligned vector-FiLM equation,
doubled training to 16 updates, and raised the active-control contrast weight
from `0.25` to `1.0`. It passed every training-integrity gate with 127/128
accepted rows; ordinal `1291` was the only filtered row.

On a new source/donor-disjoint 32-row endpoint, correct recurrence scored
`2.932965` CE versus `3.872879` for zero, `2.937126` for a matched donor, and
`2.947671` for cyclic layer permutation. All three locked margins are positive:
`+0.939914`, `+0.004161`, and `+0.014706`. The signed status is
`aligned_vector_gate_specificity_heldout_passed_generation_authorized`, with
receipt `96ccb53dbcf9f8927125a2b9ff0d6118007c11f54770d54c5fc4a3cdfee915a7`.
This establishes held-out teacher-forced state specificity for the aligned
hybrid and authorizes a separately locked native generation benchmark. It does
not itself establish native generation gain. Publisher validation, publisher
test, Hard32, and the unused strength holdout remain unopened and unauthorized.

The separately locked 220-row native generation benchmark did not transfer
that teacher-forced result into autoregressive gain. Correct recurrent state
scored `0.18964` micro-F1, versus `0.18859` for both zero recurrence and the
explicit projected-only bypass, `0.19206` for the matched donor, and `0.19010`
for cyclic layer permutation. The four correct-state margins were therefore
only `+0.00105` over zero/projected-only, `-0.00242` against the donor, and
`-0.00045` against layer permutation; every locked `+0.005` causal margin
failed. Correct recurrence changed `10.91%` of outputs relative to the fixed
carrier, so the recurrent controller was active, but its changes were not
useful or state-specific in native generation. Zero and projected-only outputs
were exact, every projected carrier remained byte-identical, and coverage
passed in every condition. The signed status is
`aligned_vector_gate_native_gain_not_established`, with receipt
`b2c3e76cebc7744021393dd5028d568e8ea334b2008414e6ce0a014ccc9c8c65`.

This rejects more training of the same independently routed aligned gate as the
next move. The next bounded hybrid should instead use the projected slot route
to address the corresponding RWKV recurrent matrix, then apply that addressed
RWKV read as a bounded vector-FiLM controller over the projected value. This
`addressed_vector_gate` keeps the proven projected carrier while forcing its
carrier and recurrent controller to refer to the same memory chunk. It must
pass fresh zero, donor, and layer-permutation teacher-forced controls before any
further native generation benchmark is opened.

That addressed route was tested in two stages. A pure addressed vector gate
remained donor-ambiguous, so the next `addressed_affine` hybrid combined its
bounded FiLM controller with a quarter-strength recurrent value residual. On a
fresh source/donor-disjoint 32-row teacher-forced endpoint, correct recurrence
scored `2.887219` CE versus `3.725045` for zero, `2.896725` for a matched donor,
and `2.926837` for layer permutation. The positive `+0.837826`, `+0.009506`,
and `+0.039618` margins passed the locked causal gate and authorized native
generation.

The locked 220-row native benchmark did not transfer. Correct recurrence
scored `0.189507` micro-F1 versus `0.198083` for zero/projected-only,
`0.195004` for the matched donor, and `0.196269` for layer permutation. Thus
all causal margins were negative: `-0.008576`, `-0.005497`, and `-0.006763`.
Correct recurrence changed `13.64%` of projected-only outputs, but increased
false positives and reduced both precision and recall. Exact zero/projected
identity, fixed projected carriers, coverage, and protected-split gates all
held. The signed status is `addressed_affine_native_gain_not_established`, with
receipt `c18d190c8fffcbac142c4d95cce6899129df637685cc551ae0e78214710ecdde`.

This establishes a pattern boundary: positive teacher-forced state preference
is insufficient when recurrent perturbations are injected unconditionally
during autoregressive decoding. Two bounded addressed controllers were then
tested under fresh source/donor-disjoint causal endpoints. `addressed_route_agreement`
used the overlap between the projected and RWKV slot routes as its scalar
abstention signal; `addressed_query_state_gate` learned a query/read gate. Both
kept the projected carrier fixed and reproduced exact projected-only behavior
when recurrence was zero, but neither acquired the required donor-specific
positive margin. Their causal endpoints were therefore rejected before native
generation was opened.

The bounded mixture-of-experts follow-up also passed its four-A100 structural
screen: it combined addressed and global recurrent reads with a three-way
query-conditioned softmax and an explicit projected-only abstention arm. The
zero-recurrent identity and fixed projected carrier held, with minimum donor
and layer-permuted perturbations of `1.390625` and `1.392578`. Its fresh
16-update causal run completed all integrity gates, but the held-out endpoint
still failed the full specificity gate: zero-minus-correct CE was `+0.882820`,
layer-permuted-minus-correct was `-0.000107`, and donor-minus-correct was
`+0.002533`. Native generation remains closed. The signed status is
`addressed_moe_controller_heldout_failed_generation_blocked`.

This is now a write/read identity boundary rather than a missing readout gate:
three addressed controllers all make recurrence visible, but none makes the
correct state reliably preferable to a matched donor. The next experiment
should bind the recurrent write itself to the projected address, with a
bounded key-conditioned write/value adapter and an explicit donor-contrast
objective. Readout-only hybrids should not be promoted to native generation
without a positive donor margin.

An outer-FFN variant then tested whether RWKV information should enter after
Gemma's frozen MLP rather than only through attention. It kept the addressed /
global MoE and projected carrier unchanged, adding a gated RMS-normalized FFN
residual at sparse decoder anchors `(10, 21, 31, 41)`. The corrected same-mode
zero-gain screen held routing, state, carrier, and attention gain fixed while
changing only the outer-FFN gain. Recurrent and carrier controls passed, but
outer-on versus outer-zero logit deltas were `1.46875`, `1.375`, `1.65625`, and
`1.125` across the four A100 ranks, above the preregistered `0.5` cap. The
signed status is `addressed_moe_outer_ffn_gain_ablation_screen_failed_branch_stopped`;
no causal endpoint or native generation was opened for this branch.

The outer hook is live, but its small per-layer residual is amplified by the
frozen stack. A DeepEmbed-style follow-up therefore moved the FFN interaction
inside Gemma's frozen gated MLP. The all-layer addressed/global RWKV MoE
attention remains active, but at decoder anchors `(10, 21, 31, 41)` a recurrent
ChannelMix maps the RWKV control and current hidden state to a bounded
multiplicative scale on Gemma's native intermediate activation before the
frozen `down_proj`:

```text
state = silu(W_down * rms(recurrent_control))
gate = sigmoid(W_gate * rms(hidden))
channel_scale = 1 + ffn_gain * tanh(
    Gemma_up_proj(rms(W_up * (state * gate)))
)
Gemma_down_proj(native_gated_mlp_activation * channel_scale)
```

Zero recurrent state produces a unit channel scale and exact projected-only
behavior. The first `1/8192` through `1/2048` gain grid was exactly invisible
at BF16 final logits. A corrected BF16 grid selected `ffn_gain=1/128` with
attention gain `1/64`. The prior all-layer FFN training layout was not
memory-safe for the 40 GiB A100 budget, so the qualified sparse design retained
attention on every layer and instantiated only 12 ChannelMix tensors at the
four anchors. Its structural screen passed with result SHA-256
`079be7f01b2c8e53199c1db4efeda4d66e4428b8fd2dc5ad4f41a1bbf61a3844`.

The exact four-A100 causal run then completed all 16 updates, accepted 127 of
128 rows under the preregistered non-finite-row filter, trained all 390 selected
tensors with zero globally inactive tensors, and stayed within the per-device
memory budget. On its fresh 11-row endpoint, zero-minus-correct CE was
`+1.221520` and layer-permuted-minus-correct was `+0.231559`, both positive on
all 11 rows. Matched-donor-minus-correct CE was `-0.003740`, however, and was
positive on only 7 of 11 rows. Thus the deeper FFN path made recurrence and
layer placement strongly visible but still did not encode the correct donor
identity. The signed status is
`addressed_moe_deepembed_ffn_sparse_heldout_failed_generation_blocked`, result
SHA-256 is
`5067878c838b55ad953b606563ce0e21a290efe55d054bc3136fad9239b488ef`, and
receipt is
`980123e096fb4125ebe0c8da98e25a0333404df4772131bf2b732b95effa4af7`.
No native generation benchmark was authorized and no native benchmark gain is
claimed.

The result rules out "add a stronger readout" as the immediate next move. An
address-keyed follow-up therefore injected the selected projected slot key into
RWKV's own `k/v/a/b` write features while retaining all-layer recurrent MoE
attention and the four sparse DeepEmbed anchors. This is a real RWKV write/read
path, not a template lookup: projected routing chooses and conditions the
recurrent write, RWKV-7 evolves four matrix states per layer, and query-time
projected routing selects the recurrent read used by attention and ChannelMix.

The first four-A100 execution completed five optimizer updates before update 6
ran out of memory while retaining three control graphs. The execution-only v5
retry serialized those graphs, then completed all 16 updates and accepted all
128 rows. Its fresh 11-row teacher-forced endpoint passed:
zero-minus-correct CE was `+1.469673`, layer-permuted-minus-correct was
`+0.335185`, and matched-donor-minus-correct was `+0.006768`. The signed
training status is
`address_keyed_moe_deepembed_ffn_serialized_graphs_heldout_passed_generation_authorized`.

The authorized 220-row publisher-TRAIN-derived native generation benchmark was
then run as four deterministic A100 shards against the exact projected-only
bypass:

| recurrent condition | micro-F1 | margin from correct |
| --- | ---: | ---: |
| correct | 0.192212 | - |
| zero / projected-only | 0.194250 | **-0.002038** |
| matched donor | 0.195688 | **-0.003476** |
| layer-permuted | 0.187192 | +0.005020 |

Coverage passed for every condition, every projected carrier stayed
byte-identical, and zero recurrence matched the explicit attention-plus-FFN
bypass exactly in both parsed predictions and raw generations. Native gain did
not pass: correct recurrence lost to the projected-only carrier and to matched
donor recurrence. The signed status is
`address_keyed_deepembed_native_gain_not_established`; result SHA-256 is
`9435980573f845ee0fde3abff987ea39ffa106e597a4ee47d5c8f2e00f7f6aba` and
receipt is
`1c79acb43b7ee6fea75dc3579bccb06c9cf81fb4485d6d82e893f58d33fdae71`.
No native RWKV/DeepEmbed benchmark gain is claimed.

The failure is informative. Correct recurrence reduced false positives from
`922` to `851`, but also reduced true positives from `125` to `116`; matched
donor state was slightly better than correct state. The current fixed
address-to-write perturbation is acting mainly as a conservative decoder
calibrator, not identity-specific memory. The best next hybrid is a learned
low-rank address-to-RWKV write transform for `k/v/a/b`, trained with an
internal query/state InfoNCE or hinge objective before answer CE. A final
state-conditioned boundary-logit adapter is the secondary direction. Merely
increasing attention or FFN gain is not supported by these results.

The next rank-2 learned-write experiment kept the same projected carrier,
RWKV-MS readout, and four sparse DeepEmbed anchors, but added eight no-op-at-
initialization low-rank tensors per layer to learn separate address transforms
for RWKV `k/v/a/b`. The four-A100 run used one row per rank to stay below the
40-GB allocator ceiling and completed all eight updates with zero globally
inactive tensors. Its 32-row causal endpoint passed the zero and layer-
permutation checks, but matched-donor-minus-correct CE was `-0.0000795`, so
the donor-identity gate failed. The signed status is
`address_keyed_learned_write_heldout_failed_generation_blocked`, result receipt
`022d28748453e5743a24a52b3b7eaa1144d29f4e96cdaf13da8a8c8c6b14a3a8`. This
isolates the remaining problem: the learned write is active and trainable, but
answer CE plus state-intervention contrasts still does not bind the correct
address strongly enough. The next goal is direct internal query/state
InfoNCE or hinge supervision, with the answer objective retained as a
secondary loss.

That direct identity experiment has now completed on four A100s. It retained
the learned rank-2 `k/v/a/b` writes, addressed MoE readout, and four sparse
DeepEmbed anchors, added no probe parameters, and trained a cosine hinge
between a detached projected-slot key address and the addressed RWKV read. All
eight updates completed, all 726 selected tensors were active, and the frozen
projected carrier remained fixed. On the fresh 32-row endpoint,
zero-minus-correct CE was `+0.499466` and layer-permuted-minus-correct was
`+0.248370`, but donor-minus-correct CE was `-0.001470`. More importantly, the
training identity advantage was only `+0.000504` with a mean hinge of
`0.199496` against the `0.2` margin; held-out correct-minus-donor cosine then
reversed to `-0.002735`, positive on only `43.75%` of rows. The signed status
is `query_state_identity_heldout_failed_generation_blocked`, result SHA-256 is
`e1acc2d492339540dc89abcf2a1cfe619f5b55d9f8c459396bead19f84199b1a`,
and receipt is
`45684bccf63bd46908fff632790b8e0484e87c3bdfdfc0e5f293439df499221a`.
Generation remains blocked and no native benchmark gain is claimed.

This failure narrows the problem to representation compatibility and loss
reduction. The projected key describes location, while the RWKV read is a
value-space vector; their raw cosine need not encode identity. The next
controlled method will freeze the target projected slot **value** as the
same-space target and apply the donor hinge independently at each answer
position and layer before reduction. Correct and donor branches will remain
separate checkpoints with a fixed detached active mask and serialized
backward. This retains the proven hybrid carrier and changes only the failed
compatibility objective; another gain increase or larger batch would not
address the observed geometry.

That projected-value target and its learned identity-bound DeepEmbed follow-up
have now both completed. The final binder run froze the causal-passing learned
writer and sparse DeepEmbed adapter, trained only 168 binder tensors on exactly
four A100s, and completed all eight updates without OOM. On its fresh 16-row
endpoint, zero-minus-correct and layer-permuted-minus-correct CE were
`+1.301495` and `+0.285514`, but matched-donor-minus-correct was `-0.007757`.
The learned binder score also preferred the donor by `0.002040`, with correct
positive on only `43.75%` of rows. The signed status is
`identity_bound_deepembed_heldout_failed_generation_blocked`, result SHA-256
is `90aa3aba6fd7ec885f16f344801ea8575825dce07103a37dc90b821d6fc9ba46`,
and receipt is
`96f782b35e2d920c72a07cc01d323f3fd4a9c1177d338d93e2b5561518871c96`.
No native benchmark or SOTA gain is claimed.

The exact-source v5 shadow cross-fit has now passed on all 220 open development
rows. It executed the causal-passing adapter under its signed `cd7deb91` core,
changed no model output, and fit a disposable identity head on a source-and-
donor-component-disjoint `176/44` split. Held-out matched-donor pairwise
accuracy was `0.954545` with a `0.103092` mean score gap; cyclic layer-
permutation accuracy was `1.0`. The result SHA-256 is
`c3607fbc6f42b6a2ebcdfab7d5cdf399e5b8e4c8ab52a1c707e8f1d19d44108d`
and receipt is
`4ba137387216a8f2bc2c5562a764b4f340afa795cc4dbc88d4d2cf0ea470443c`.
This establishes learnable identity in untouched teacher-forced answer-position
RWKV shadows, not identity at the preceding causal predictor positions, causal
use, or native gain; training, generation, and the native benchmark remain
blocked.

The separately signed causal-predictor replication has now rejected that
identity family. It recaptured all 220 open rows one token before each
teacher-forced answer position on exactly four A100s and reused the same
donor-component-disjoint `176/44` split. The held-out row-level donor fraction
was `0.954545`, but token-level donor separation was only `0.878327` and the
mean donor gap was `0.047801`, missing the locked `0.95` and `0.05` gates.
Layer-permuted separation remained `1.0`. Stage 2 recurrent mechanics did not
run; no weights were trained, no protected split was opened, and native
generation remains blocked. The result SHA-256 is
`44e4b22c6db8b9c9e98a947ec9baf829291bb49efd2b5dc5b21ca98574ca9cbb`
and receipt is
`3489154f6bae3feadd3510ca2aeddce31dea3fb3d5e5f995c043bf466c544959`.

The pre-signed prompt-latched follow-up also failed before model mechanics. It
excluded the earlier 44 held-out rows, formed a new donor-component-disjoint
`132/44` split from the former fit rows, and expanded only the first causal
predictor query across each answer. Held-out donor row fraction (`0.954545`),
mean gap (`0.054514`), layer-permuted separation (`1.0`), and finiteness passed,
but donor token fraction was only `0.919414` against the locked `0.95` gate.
No model was loaded and no weights, Stage 2, generation, or protected split
were opened. Result SHA-256 is
`4f7c4aa1f715157e95cd753842b79d28f94b3356e318ef5c7c09e911456f8aac`
and receipt is
`de1a677ce8b77e5c1f16eb9f9601d93c1140feffdf8adc60922e8cf671b01979`.

Read-side identity recovery, prompt-latch transport, and discrete PLMSC are
therefore retired. PLMSC was executed exactly once on four A100s. On its locked
34-row mechanics split, correct write/query code agreement was only `0.433824`
per anchor and `0.058824` per complete row against locked `0.95` gates. Matched
donor and cyclic layer-permuted anchor collisions were `0.132353` and
`0.139706` against a maximum `0.03`; the layer-10 query collapsed to three
codes with one code used by `64.71%` of rows. The signed status is
`plmsc_code_alignment_failed_family_retired`, result SHA-256 is
`b7dce00737c928abc13729b19e24ccfe803b9dce6dde62b9d9d944971a295544`,
and receipt is
`23c7cfdf0cdf0fb747010615cfe271ae7d7c0cddd7bd9a90401179033100fda7`.
The untouched causal 34 remained unopened, un-tokenized, and un-forwarded; no
model weights, code maps, generation, native benchmark, or SOTA claim were
authorized.

An independent two-axis algebraic fallback then bound the complete RWKV matrix
as `S_bound = D_value(A) S D_key(A)`, with `v` coded on the value axis and
`k/a/b/r` coded on the key axis.  Its single authorized four-A100 development
run completed all 64 open rows and preserved final logits bit-exactly on every
row.  The family nevertheless failed before mechanics: routed write codes and
the expected encoded state matched on `0/64` rows, while left and right donor
code separation were each `0.976190` against an exact `1.0` gate.  Mechanics,
causal, generation, and native benchmark data remained unopened.  The signed
status is `bidirectional_sign_development_failed_family_retired`, result
SHA-256 is `05a7797cece7ce32aa8407119900405d324eb6b9142516c7c08d5f1783f43144`,
and receipt is
`63a61d9814cd3327591808db6cab977bf883df8033863cb122fb64b181f7e97f`.
The launcher returned nonzero only after writing this valid result because its
final in-memory/persisted comparison treated a tuple and its JSON list as
different; independent schema and receipt validation passes, so this is a
recorded architecture failure rather than authorization for another retry.

The next distinct family removes the discrete codebook. A continuous vector
derived from the outer write address will condition the exact-v5 RWKV write
key/update, while the full RWKV state remains the material value through the
sparse outer FFN and early fusion. The frozen bias-free address map is locked
to reduced-rank ridge with rank `16` and ridge `1.0`, fixed as a disclosed
precommitted design choice before the fresh retrieval gate. Its SHA-qualified
FIT dataset must first be
split by normalized-passage/32-character-shingle connected components into
`64` fit, `32` retrieval, `32` mechanics, and `32` causal source rows. Exclude
the full component closure touching all `98` sources in the bidirectional
manifest; PLMSC numeric indices belong to another dataset namespace and must
not be mixed into this split. Capture only fit and retrieval, and require their
matched-donor and layer-permutation retrieval pass before opening mechanics.
Mechanics must then pass address-permutation, donor-address/state, zero-address,
layer-permutation, prompt-only, state-only, inherited exact-v5, raw-disabled,
and immutability controls before any training or native benchmark is
authorized. The implementation must snapshot the selected projected keys and
routes once immediately after the projected-slot write, materialize one
immutable address sequence, and pass that same tensor to both the `k/a/b`
conditioner and its audit.  A synthetic old-key-to-new-key mutation regression
must pass before capture or GPU execution; recomputing an address later from
mutable slot state is prohibited. Contraction belongs only to a later
Full-Bandwidth read-feedback stage after identity mechanics and causal gates,
not to this one-pass write screen. See the
[paper review](experiments/rethinking_rwkv_ms_gemma/FULL_BANDWIDTH_RWKV_REVIEW.md)
for the transfer boundary and stopping gates.
The component-safe v2 inventory is materialized under
`experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1`.
Its manifest SHA-256 is
`c437a7d1f2b850a730fe5b28a08ae32ba02678561bb1265a4eef55bda7f4d468`
and receipt is
`99a878493c3848c96624e2ad658842c99e69769b4a1721b5854ad25af8d0bee2`;
fit/retrieval are validated, while mechanics/causal remain sealed.

The one-shot continuous alignment retrieval gate now passes. On exactly four
A100s, it captured all `42` exact-v5 RWKV modules for the `64` FIT rows, froze
the rank-`16` maps in memory, and only then opened and captured the `32`
retrieval rows. Every addressed/global read-basis pair was byte-identical. The
retrieval donor-positive row fraction is `1.000000` against `0.95`, the mean
correct-minus-donor gap is `0.069070` against `0.05`, the layer-permuted
positive row fraction is `1.000000`, and its mean gap is `0.868476`. Exact-zero
addresses remain exact zero and every active mapped direction is finite and
nonzero. The signed result SHA-256 is
`5103a66475b7a596e53a58b8c7cb554e7e400f5825d42ca141bc342dfef8784b`
with receipt
`cf001ac0f06afeb58b96084d656e5a22521a7d2229d68436b09572e231e0a6dd`.
This establishes source/donor-disjoint full64-to-causal32 identity geometry;
capture ran in inherited exact-v5 mode, so the retrieval result alone did not
establish live continuous-write mechanics, causal donor specificity,
generation, native benchmark gain, or SOTA. At that stage, mechanics and causal
bytes remained unopened and the pass authorized only a separately signed
continuous-mode mechanics protocol.

The separately signed continuous-write mechanics gate has now passed on exactly
four A100s. All `32` mechanics rows ran once, exactly `8/GPU`, across all `42`
modules, `10` write conditions, and `17` read conditions. Mean normalized state
L2 was `0.793498` for correct continuous versus raw, `0.366313` versus matched-
donor address only, `1.048170` versus layer-rolled address only, and `0.351101`
versus target address on donor content. Every comparison had positive row
fraction `1.0`, passing the locked `0.05` mean and `0.95` row-fraction gates.
The full-address override matched the requested float32 bytes, projected
carriers and recurrent metadata stayed fixed, `v` stayed the same object and
bytes, exact-zero address reproduced raw features/state/logits, zero recurrence
reproduced projected-only logits, and projected-only made zero underlying RWKV
read-basis calls. No parameters were updated.

The signed status is
`continuous_write_mechanics_passed_causal_protocol_draft_authorized`; result
SHA-256 is
`a7215ff987f06a369e19ea5b62e54ae2e99b018b9dbed15616f964806e811456`,
and receipt is
`2621b0d7773f7931fda80676774697fcc4c059abf49f8ebbad683f19f34c1a95`.
This proves material, identity-sensitive continuous `k/a/b` write mechanics
with `v` unchanged. It does **not** prove causal answer preference, generation
quality, native benchmark gain, or SOTA. The pass authorized only a separately
signed causal protocol; Full-Bandwidth read feedback remained deferred until
that causal donor-identity gate.

The separately signed continuous-write causal endpoint has now completed. Its
first launch stopped in preflight before causal access because the workspace
`delta_impl` was imported before the signed exact-v5 source root; a regression-
tested import-order repair was squashed into a replacement code commit and a
fresh launch-only child before the one protected open. The corrected run used
exactly four A100s, completed all eight FIT updates over all `32` symmetric
source/donor pairs, froze the `84` selected read-path tensors, signed the
checkpoint and training receipt, and then read the `32` causal rows exactly
once. All logits were finite, projected carriers stayed fixed, and zero
recurrence was byte-identical to projected-only on every row.

The causal identity result failed. Zero-minus-correct passed at `+0.050419` CE
with `0.875000` positive rows, but layer-roll-minus-correct reached only
`+0.019981` with `0.718750` positive rows, and matched-donor-minus-correct was
negative at `-0.007690` with only `0.406250` positive rows. The signed status is
`continuous_write_causal_failed_readout_family_retired`; result SHA-256 is
`71d738ab63ae893c79b42e2cb1a93e25fee5e64daa6bf0d9d3eceb7dff572a09`,
and receipt is
`5660251fc35005ee6cc054587d83bdd3069f22c52b9d9d7b440912fdbf71c0d0`.
This exact readout family is retired without gain, batch-size, learning-rate,
or duration tuning. Generation and the native benchmark remain unauthorized
and unopened. Full-Bandwidth recurrence is still a depth-renewal candidate,
not a repair for the failed matched-state identity gate.

The next locked stage tested address-decoded RWKV token replacement before any
new protected split. A bias-free ridge decoder was fit on the `64` already-open
FIT rows, frozen and persisted, and only then evaluated on the `32` already-open
retrieval rows using exactly four A100s. Correct reconstruction reached
`0.912683` mean cosine, but matched-donor and wrong-address separation reached
only `0.008718`/`0.468750` and `0.009482`/`0.812500` for mean gap/positive-row
fraction; `0/42` modules passed the locked identity gate. Layer roll separated
strongly at `1.094052` with positive-row fraction `1.0`. The signed status is
`address_decoded_reconstruction_failed_linear_decoder_family_retired`; result
SHA-256 is `d6992d50ef60f70e2dc503b9b752c36c723f4ac3ed3ade45fa41583b6ee8e5bd`
and receipt is `c7df99f673e55b749b88e7b9f8a71967ed2fd4da28aa99d21fa9daf9b563c93a`.

This retires only the linear `S d(A) -> value` cosine decoder. Cosine is
scale-invariant, so an approximately rank-one state can map a wrong address to
a rescaled vector in the same payload direction and receive nearly identical
score. The stronger next route separates the roles: an explicit address-derived
virtual key carries identity, while an RMS-normalized RWKV state contraction
supplies the virtual value inside Gemma attention. It must first pass key-logit,
matched-donor, wrong-key, zero-carrier, and cache-immutability mechanics gates.
Plain Full-Bandwidth feedback remains deferred until that causal identity path
passes.

A separate Full-Bandwidth-inspired source-residual line has now reached the
same boundary more directly. A layer-17 cumulative router selected the target
source on `93.75%` of open heldout rows, and a prompt latch preserved that rate
at the first target/donor-divergent answer token. The final joint-identity
candidate used per-anchor address/receptance product-and-distance features but
kept the selected 32-dimensional native RWKV read as its only material value.
Exactly four A100s, configured through `HF_ENDPOINT=https://hf-mirror.com`,
completed all 32 signed updates. Mechanics, finiteness, staged gradients, and
exact-zero/provider-off controls passed, but causal identity did not. On the
original heldout view, donor-minus-target CE was `-0.020373` with `0.375000`
positive rows; on the divergent-token view it was `-0.000756` with `0.468750`
positive rows. Correct memory was also worse than provider-off on both views.
The signed result SHA-256 is
`819cb586acbbc4f391256048cf1bd38a774237d46ae3398fbd9c204983cb5746`
with receipt
`25fc989427c5acb56f3c78f681b3ee508dc3d2f24268c7f8b464648915548563`.
No protected or native benchmark data was opened, and no benchmark gain or
SOTA claim is made. The paper's depth-renewal loop remains a later candidate;
the next open-only test widens the RWKV value path to a prompt-latched
multi-anchor read bundle before adding temporal feedback.

Evidence: [continuous retrieval protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_continuous_write_retrieval_protocol_v1.json),
[continuous retrieval runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_continuous_write_retrieval.py),
[signed retrieval pass](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_continuous_write_retrieval_v1/result.json),
[continuous mechanics protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_continuous_write_mechanics_protocol_v1.json),
[exact launch binding](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_continuous_write_mechanics_launch_v1.json),
[continuous mechanics runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_continuous_write_mechanics.py),
[signed mechanics pass](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_continuous_write_mechanics_v1/result.json),
[continuous causal protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_continuous_write_causal_train_protocol_v1.json),
[continuous causal launch](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_continuous_write_causal_train_launch_v1.json),
[continuous causal runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_continuous_write_causal_train.py),
[signed continuous causal failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_continuous_write_causal_train_v1/result.json),
[address-decoded protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_address_decoded_reconstruction_protocol_v1.json),
[address-decoded launch](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_address_decoded_reconstruction_launch_v1.json),
[address-decoded runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_address_decoded_reconstruction.py),
[signed address-decoded failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_address_decoded_reconstruction_v1/result.json),
[prompt-latched joint-identity protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_prompt_latched_joint_identity_development_protocol_v1.json),
[prompt-latched joint-identity runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_prompt_latched_joint_identity_development.py),
and [signed prompt-latched joint-identity failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_prompt_latched_joint_identity_development_v2/result.json).

Evidence: [recurrent-only protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_recurrent_rwkv_protocol_v1.json),
[signed recurrent-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_recurrent_rwkv_bf16_calibration_v1/result.json),
[hybrid screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_projected_rwkv_hybrid_screen_protocol_v1.json),
[signed hybrid screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_projected_rwkv_hybrid_screen_v1/result.json),
[hybrid calibration protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_projected_rwkv_hybrid_bf16_calibration_protocol_v1.json),
[signed hybrid calibration](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_projected_rwkv_hybrid_bf16_calibration_v1/result.json),
[hybrid screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_projected_rwkv_hybrid_screen.py),
[hybrid calibration runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_projected_rwkv_hybrid_bf16_calibration.py),
[locked native benchmark protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_projected_rwkv_hybrid_benchmark_protocol_v1.json),
[signed native benchmark result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_projected_rwkv_hybrid_benchmark_eval_fee6ae1/result.json),
[benchmark training runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_projected_rwkv_hybrid_benchmark_train.py),
[benchmark evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval.py),
[hash-bound benchmark analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_projected_rwkv_hybrid_benchmark.py),
[addressed-value screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_value_screen_protocol_v1.json),
[signed addressed-value screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_value_screen_v1/result.json),
[addressed-value calibration protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_value_calibration_protocol_v1.json),
[signed addressed-value calibration](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_value_calibration_v1/result.json),
[causal-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_value_causal_train_protocol_v1.json),
[signed causal-training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_value_causal_train_v1/result.json),
[signed addressed-value native result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_value_eval_batched_2way_v1/result.json),
[addressed-value runners](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_value_screen.py),
[hash-bound addressed-value analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_addressed_value_eval.py),
[chunk-addressed screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_chunk_addressed_value_screen_protocol_v1.json),
[signed corrected chunk-addressed screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_chunk_addressed_value_screen_v2/result.json),
[chunk-addressed calibration protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_chunk_addressed_value_calibration_protocol_v1.json),
[signed chunk-addressed calibration](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_chunk_addressed_value_calibration_v1/result.json),
[chunk-addressed causal-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_chunk_addressed_value_causal_train_protocol_v1.json),
[signed chunk-addressed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_chunk_addressed_value_causal_train_v1/result.json),
[signed chunk-addressed native result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_chunk_addressed_value_eval_v1/result.json),
[chunk-addressed evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_chunk_addressed_value_eval.py),
[hash-bound chunk-addressed analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_chunk_addressed_value_eval.py),
[focused chunk-addressed tests](deltamem/tests/test_natural_memory_native_rwkv_chunk_addressed_value_screen.py),
[addressed route-agreement screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_route_agreement_screen_protocol_v1.json),
[signed addressed route-agreement screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_route_agreement_screen_v1/result.json),
[addressed route-agreement causal protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_route_agreement_causal_train_protocol_v1.json),
[signed addressed route-agreement causal endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_route_agreement_causal_train_v1/result.json),
[addressed route-agreement runners](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_route_agreement_screen.py),
[addressed query-state-gate screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_query_state_gate_screen_protocol_v1.json),
[signed addressed query-state-gate screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_query_state_gate_screen_v1/result.json),
[addressed query-state-gate causal protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_query_state_gate_causal_train_protocol_v1.json),
[signed addressed query-state-gate causal endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_query_state_gate_causal_train_v1/result.json),
[addressed query-state-gate runners](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_query_state_gate_screen.py),
[addressed MoE-controller screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_controller_screen_protocol_v1.json),
[signed addressed MoE-controller screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_controller_screen_v4/result.json),
[addressed MoE-controller causal protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_controller_causal_train_protocol_v1.json),
[signed addressed MoE-controller causal endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_controller_causal_train_v4/result.json),
[addressed MoE-controller runners](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_moe_controller_screen.py),
[outer-FFN architecture screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen_protocol_v1.json),
[signed outer-FFN screen failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen_v1/result.json),
[outer-FFN screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen.py),
[DeepEmbed initial screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_protocol_v1.json),
[signed DeepEmbed BF16-invisible screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v1/result.json),
[DeepEmbed BF16 screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_protocol_v2.json),
[signed DeepEmbed BF16 screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v2/result.json),
[sparse DeepEmbed screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen_protocol_v1.json),
[signed sparse DeepEmbed screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen_v1/result.json),
[sparse DeepEmbed causal protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train_protocol_v1.json),
[signed sparse DeepEmbed causal failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train_v1/result.json),
[DeepEmbed screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen.py),
[DeepEmbed BF16 screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v2.py),
[sparse DeepEmbed screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen.py),
[sparse DeepEmbed causal runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train.py),
[address-keyed DeepEmbed v5 training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v5.json),
[signed address-keyed DeepEmbed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5_r1/result.json),
[exact-v5 shadow cross-fit protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_v5_shadow_crossfit_protocol_v1.json),
[exact-v5 shadow cross-fit runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_v5_shadow_crossfit.py),
[signed exact-v5 shadow identity pass](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_v5_shadow_crossfit_v1/result.json),
[causal-predictor recurrent protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics_protocol_v1.json),
[causal-predictor recurrent runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics.py),
[signed causal-predictor identity failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics_v1/result.json),
[PLAT prompt-latch protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_plat_prompt_latch_crossfit_protocol_v1.json),
[PLAT prompt-latch runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_plat_prompt_latch_crossfit.py),
[signed PLAT identity failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_plat_prompt_latch_crossfit_v1/result.json),
[PLMSC v2 protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_plmsc_code_alignment_protocol_v2.json),
[PLMSC v2 runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_plmsc_code_alignment_v2.py),
[signed PLMSC retirement](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_plmsc_code_alignment_v2/result.json),
[locked address-keyed native protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_generation_protocol_v1.json),
[signed address-keyed native failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval_v1/result.json),
[address-keyed native evaluator](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval.py),
[address-keyed native analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval.py),
[learned-write causal protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_address_keyed_learned_write_causal_train_protocol_v1.json),
[learned-write causal runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train.py),
[signed learned-write causal result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_address_keyed_learned_write_causal_train_v3/result.json),
[query-state identity protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_query_state_identity_causal_train_protocol_v1.json),
[query-state identity runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_query_state_identity_causal_train.py),
[signed query-state identity failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_query_state_identity_causal_train_v3/result.json),
[recurrent-value screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_recurrent_value_screen_protocol_v1.json),
[signed recurrent-value screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_recurrent_value_screen_v1/result.json),
[recurrent-value calibration protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_recurrent_value_calibration_protocol_v1.json),
[signed recurrent-value calibration](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_recurrent_value_calibration_v1/result.json),
[corrected causal-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_recurrent_value_causal_train_protocol_v2.json),
[signed recurrent-value training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_recurrent_value_causal_train_v1/result.json),
[signed recurrent-value native result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_recurrent_value_eval_v1/result.json),
[recurrent-value evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_recurrent_value_eval.py),
[hash-bound recurrent-value analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_recurrent_value_eval.py),
[focused recurrent-value tests](deltamem/tests/test_natural_memory_native_rwkv_recurrent_value_eval.py),
[sharp-router screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_sharp_router_screen_protocol_v1.json),
[signed corrected sharp-router screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_sharp_router_screen_v2/result.json),
[top-2 contrast calibration protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_top2_abstention_contrast_calibration_protocol_v1.json),
[signed top-2 calibration](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_top2_abstention_contrast_calibration_v1/result.json),
[SPSA causal-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_spsa_gate_causal_train_protocol_v1.json),
[signed SPSA failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_spsa_gate_causal_train_v1/result.json),
[positive-only filtered protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_positive_only_filtered_causal_train_protocol_v1.json),
[signed positive-only filtered failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_positive_only_filtered_causal_train_v1/result.json),
[filtered causal-contrast protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_filtered_contrast_causal_train_protocol_v1.json),
[signed filtered causal failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_filtered_contrast_causal_train_v1/result.json),
[scalar-agreement training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_scalar_agreement_causal_train_protocol_v1.json),
[signed scalar-agreement causal endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_scalar_agreement_causal_train_v1/result.json),
[locked scalar native protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_scalar_agreement_generation_protocol_v1.json),
[signed scalar native failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_scalar_agreement_eval_v2/result.json),
[scalar native evaluator](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_scalar_agreement_eval.py),
[hash-bound scalar analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_scalar_agreement_eval.py),
[focused scalar tests](deltamem/tests/test_natural_memory_native_rwkv_scalar_agreement_eval.py),
[vector-FiLM training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_vector_gate_causal_train_protocol_v1.json),
[signed vector-FiLM causal endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_vector_gate_causal_train_v1/result.json),
[locked vector-FiLM native protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_vector_gate_generation_protocol_v1.json),
[signed vector-FiLM native result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_vector_gate_eval_v1/result.json),
[vector-FiLM evaluator](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_vector_gate_eval.py),
[hash-bound vector-FiLM analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_vector_gate_eval.py),
[focused vector-FiLM tests](deltamem/tests/test_natural_memory_native_rwkv_vector_gate_eval.py),
[alignment-residual screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_alignment_residual_screen_protocol_v1.json),
[signed alignment-residual screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_alignment_residual_screen_v1/result.json),
[alignment-residual training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_alignment_residual_causal_train_protocol_v1.json),
[signed alignment-residual causal failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_alignment_residual_causal_train_v1/result.json),
[alignment-residual screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_alignment_residual_screen.py),
[alignment-residual training runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_alignment_residual_causal_train.py),
[focused alignment-residual tests](deltamem/tests/test_natural_memory_native_rwkv_alignment_residual_causal_train.py),
[aligned-vector screen protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_aligned_vector_gate_screen_protocol_v1.json),
[signed aligned-vector screen](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_screen_v1/result.json),
[aligned-vector training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_aligned_vector_gate_causal_train_protocol_v1.json),
[signed aligned-vector causal failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_causal_train_v1/result.json),
[aligned-vector screen runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_aligned_vector_gate_screen.py),
[aligned-vector training runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_aligned_vector_gate_causal_train.py),
[focused aligned-vector tests](deltamem/tests/test_natural_memory_native_rwkv_aligned_vector_gate_causal_train.py),
[specificity-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_aligned_vector_gate_specificity_train_protocol_v1.json),
[signed specificity endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_specificity_train_v1/result.json),
[specificity-training runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_aligned_vector_gate_specificity_train.py),
[focused specificity tests](deltamem/tests/test_natural_memory_native_rwkv_aligned_vector_gate_specificity_train.py),
[locked specificity-generation protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_rwkv_aligned_vector_gate_specificity_generation_protocol_v1.json),
[signed specificity-generation failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_specificity_eval_v1/result.json),
[specificity-generation evaluator](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_aligned_vector_gate_specificity_eval.py),
[hash-bound specificity-generation analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_rwkv_aligned_vector_gate_specificity_eval.py),
and [focused specificity-generation tests](deltamem/tests/test_natural_memory_native_rwkv_aligned_vector_gate_specificity_eval.py).

### Post-Validation Mechanism Study

A preregistered 357-row publisher-TRAIN-derived study has now tested three
online-state counterfactuals and 16 conservative scene routers. Correct state
scored 0.2901 scene micro-F1, versus 0.1909 with zero state and 0.1923 after
cyclically moving complete state bundles across the 42 wrapped layers. This is
strong evidence that structured online state matters. The stricter causal gate
did not pass, however: a different-gold, write-length-matched donor state scored
0.3001, beating the row-correct state by 0.0101. We therefore do not claim that
row-specific episodic content uniquely causes the scene gain.

The router screen selected `memory_plus_small_base_2`: union frozen-base and
memory boundaries only when the base predicts at most two. It improved fit
micro-F1 from 0.2844 to 0.3190 and held-out development micro-F1 from 0.3093 to
0.3103. That held-out gain is real but only +0.0011, so this router is a future
replication candidate and does not replace the accepted validation decoder.

A subsequent preregistered label-free state-retrieval study also failed its
materiality gate. Four rules selected external states from a 1,443-row
TRAIN-derived bank. Deterministic hash-random state won the 289-row fit screen,
improving 0.2795 to 0.3093 micro-F1, but improved the exactly-once 68-row
intervention holdout only from 0.3333 to 0.3352 (+0.0019, below the required
+0.005). Semantic character-TF-IDF and token-length retrieval did not win. This
points to weak generic state-induced regularization, not reliable semantic
state retrieval.

Effective Q/O memory-strength calibration has now also completed. A signed
excluded-row preflight proved that `0x`, `0.5x`, and `1x` produce different
outputs, then four A100s generated 1,136 candidates on the 284-row fit
partition. Full strength remained best at 0.2915 micro-F1. The `0.5x` and
`0.75x` candidates scored 0.2853; lower strengths scored 0.2195 and 0.2112.
No intermediate strength passed the `+0.005` gate, so no selection was written
and the 73-row intervention holdout remains unopened. This rules out a fixed
global amplitude as the next boundary, not memory learning itself.

Evidence:

- [Locked causal/router protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_causal_router_protocol_v1.json)
- [Signed causal/router decision](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_causal_router_v1/decision.json)
- [Signed result and raw-artifact hashes](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_causal_router_v1/result.json)
- [Signed state-retrieval failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_state_retrieval_v1/decision.json)
- [State-retrieval fit and holdout receipts](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_state_retrieval_v1/holdout_result.json)
- [Effective-strength protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_strength_controller_protocol_v2.json)
- [Signed effective-strength decision](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_strength_controller_v2_r2/decision.json)
- [Effective-strength fit receipt](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_strength_controller_v2_r2/fit_result.json)

### Contrast-Trained Scene Memory

The recommended training intervention has now succeeded on publisher-TRAIN-
derived scene data. Only the 126 shared Q/O content-gate tensors were updated;
all other adapter tensors remained bit-identical. Training used correct/no-
state positives and different-gold, write-length-matched donor negatives on
four A100 GPUs. A locked 64-row checkpoint probe selected step 16; step 32 was
rejected because it no longer beat the donor control.

Checkpoint 16 then generalized to all 220 remaining open fit rows. On the
combined 284-row fit partition it reached `0.3197` scene micro-F1, versus
`0.2915` for frozen V9, `0.3058` for matched-donor state, and `0.1980` for zero
state. The output changed on 25.7% of rows versus V9, and every preregistered
coverage, native-gain, and causal-control gate passed. This is stronger than a
generic state effect: the row-correct state now beats the matched donor.

This result does not replace the accepted publisher-validation number. It is a
TRAIN-derived candidate result, and no validation predictions were used for
checkpoint selection or analysis.

Evidence:

- [Locked contrast-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_contrast_dropout_protocol_v1.json)
- [Signed training result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_contrast_dropout_train_v1/result.json)
- [Signed checkpoint-16 selection](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_contrast_probe_v1/selection.json)
- [Locked full-fit progression](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_contrast_progression_protocol_v1.json)
- [Signed full-fit result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_contrast_progression_v1/result.json)

The subsequent multitask-preservation gate passed scene progression and exact
attribution reuse, but failed its narrative comparator narrowly. Checkpoint 16
reached `0.5987` routed narrative unit accuracy on the untouched 114-row
remainder, above frozen base (`0.5847`) but below V9's routed comparator
(`0.6007`) by `0.0020`. The complete signed failure is archived in the
[preservation result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_multitask_preservation_v1/result.json).

### Hybrid Validation Replication

The failure is isolated to 12 narrative unit-level disagreements, so replacing
V9 narrative behavior with checkpoint-16 output is rejected. The signed hybrid
candidate keeps the proven task-wise contracts: frozen-base candidate
likelihood for attribution, exact V9 routed output for narrative, and
checkpoint-16 correct-state generation for scene. On the combined open
TRAIN-derived fit rows this gives attribution accuracy `0.6966`, V9 routed
narrative accuracy `0.6007`, and checkpoint-16 scene micro-F1 `0.3197`.

The TRAIN-derived hybrid gates passed without opening any protected split and
authorized one separately preregistered publisher-validation replication. That
replication generated every condition again from raw validation rows on four
A100 GPUs; it did not read or reuse prior validation predictions.

| Native task | Fresh frozen base | Hybrid candidate | Delta |
| --- | ---: | ---: | ---: |
| Attribution candidate accuracy | 0.8966 | 0.8966 | +0.0000 |
| Narrative unit accuracy | 0.6432 | 0.6467 | +0.0035 |
| Scene-boundary micro-F1 | 0.1820 | 0.2711 | +0.0891 |

The candidate improved two tasks over base, but the stricter training-gain gate
failed: freshly regenerated V9 reached `0.2727` scene micro-F1, so checkpoint 16
was lower by `0.0016` instead of exceeding V9 by the required `0.005`. The
contrast-trained checkpoint therefore does not replace V9. The accepted V9
publisher-validation result at the top of this README remains authoritative.
No publisher test, Hard32, or unused strength holdout evaluation is authorized.

Evidence:

- [Locked hybrid protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_multitask_hybrid_protocol_v1.json)
- [Signed hybrid result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_multitask_hybrid_v1.json)
- [Hash-bound hybrid analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_multitask_hybrid.py)
- [Locked fresh-validation protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_hybrid_publisher_validation_protocol_v1.json)
- [Signed fresh-validation failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_hybrid_publisher_validation_v1/result.json)
- [Fresh-validation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_hybrid_publisher_validation.py)
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_hybrid_publisher_validation.py)

### Cross-Fitted Scene Router

A subsequent TRAIN-only study tested a different method without reopening any
protected split. Eight fixed V9/checkpoint-16 set-combination rules were
selected independently inside five hash folds and scored only on each held-out
fold. Four folds selected `v9_if_subset_else_checkpoint`; one selected raw
checkpoint 16.

The cross-fitted router reached `0.3191` scene micro-F1, above frozen V9
(`0.2915`) but just below checkpoint 16 (`0.3197`) by `0.0006`. It removed four
false positives but also lost one true positive. The preregistered gate required
a `+0.005` gain over both inputs, so this method failed and no external
replication is authorized. Simple set routing is therefore not the next
boundary; further work must improve training robustness or expose calibrated
token-level confidence while remaining publisher-TRAIN-derived.

Evidence:

- [Locked cross-fit protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_crossfit_router_protocol_v1.json)
- [Signed cross-fit failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_crossfit_router_v1/result.json)
- [Cross-fit analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_crossfit_router.py)

### Checkpoint-Soup Failure

A second publisher-TRAIN-only study tested weight-space averaging instead of
another output router. Seven convex recipes mixed only the 126 learned content-
gate tensors from frozen V9 and contrast checkpoints 8, 16, and 32. The recipe
bytes were signed and pushed before any new generation. Five hash folds then
selected among those recipes with unchanged checkpoint 16 available as a
fallback, using only the other four folds for each held-out decision.

The best single recipe, `trajectory_centered` (`25%` step 8, `50%` step 16,
`25%` step 32), reached `0.3175` scene micro-F1 on the 220 post-probe fit rows.
That is only `+0.0021` over checkpoint 16 (`0.3154`), below the locked `+0.005`
requirement. Fold selection was also unstable: the five winners were
`trajectory_centered`, `s16_75_s32_25`, checkpoint 16,
`trajectory_centered`, and `v9_25_s16_75`. Their combined out-of-fold score was
`0.3012`, or `-0.0142` versus checkpoint 16, despite remaining `+0.0108` above
V9 (`0.2904`). The method therefore failed and authorizes no external
replication. Neither prediction-set routing nor convex checkpoint averaging is
the next boundary; the next study must change training robustness itself.

Evidence:

- [Locked checkpoint-soup protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_checkpoint_soup_protocol_v1.json)
- [Signed candidate materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_checkpoint_soup_materialization_v1/result.json)
- [Signed checkpoint-soup failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_checkpoint_soup_v1/result.json)
- [Four-GPU runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_checkpoint_soup.py)
  and [cross-fit analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_checkpoint_soup.py)

### Independent-Seed Robust Training Failure

The next publisher-TRAIN-only study changed training rather than selecting
among outputs. Three independently hashed 256-row schedules trained the same
126 content-gate tensors on four A100 GPUs with global batch 16, half the
original learning rate, and an explicit `0.995` post-step pull toward V9. All
three runs were numerically clean and tightly matched in endpoint delta norm
(`0.1001`, `0.1039`, and `0.0973`). Their pairwise delta cosines were
`0.611`-`0.659`. The only preregistered candidate was the equal mean of the
three signed V9-relative deltas, fixed and pushed before generation.

That candidate reached `0.3059` scene micro-F1 on the same 220 open fit rows.
It remained above V9 (`0.2904`) by `+0.0155`, but fell below checkpoint 16
(`0.3154`) by `-0.0094`. Relative to checkpoint 16 it gained one true positive
but added 21 false positives, so the locked `+0.005` gate failed. The averaged
delta was only `0.0873` from V9 versus `0.1847` for checkpoint 16 and had
cosine `0.566` with checkpoint 16's delta. The lower-rate V9-centered ensemble
therefore stabilized an underpowered direction rather than preserving the
single checkpoint's precision. It authorizes no external replication. A next
training study should anchor small independent residual updates at checkpoint
16 instead of pulling every run back toward V9.

Evidence:

- [Locked seed-ensemble protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_seed_ensemble_protocol_v1.json)
- [Signed seed-delta materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_seed_ensemble_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_seed_ensemble_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_seed_ensemble.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_seed_ensemble_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_seed_ensemble.py)

### Checkpoint-16 Residual Training Failure

A follow-up publisher-TRAIN-only study preserved checkpoint 16 as the anchor
instead of restarting from V9. Three four-GPU runs used disjoint sets of 128
previously unused rows, eight global-batch-16 updates, learning rate `2.5e-5`,
and a `0.995` post-step pull toward checkpoint 16. The runs were numerically
clean, changed no frozen parameters, and ended only `0.0251`, `0.0269`, and
`0.0258` from the anchor. Their residual directions were weakly aligned,
however: pairwise cosines were `0.049`-`0.189`. The locked equal residual mean
therefore had norm `0.0169` and was fixed and pushed before generation.

The candidate reached `0.3129` scene micro-F1 on the same 220 open fit rows.
It preserved checkpoint 16's 79 true positives and 161 false negatives, but
added four false positives, reducing micro-F1 by `-0.0025` from checkpoint 16
(`0.3154`). It still remained `+0.0225` above V9 (`0.2904`) and changed `6.36%`
of checkpoint-16 outputs, but failed the preregistered `+0.005` improvement
gate. It authorizes no external replication, and publisher validation, test,
Hard32, and the unused 73-row holdout remain unopened. Repeated endpoint
averaging is no longer the useful boundary: the next training intervention
must directly suppress false-positive scene labels while preserving
checkpoint 16's true positives.

Evidence:

- [Locked checkpoint-16 residual protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_c16_residual_protocol_v1.json)
- [Signed residual materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_c16_residual_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_c16_residual_v1/result.json)
- [Four-GPU residual trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_c16_residual.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_c16_residual_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_c16_residual.py)

### Precision-Unlikelihood Training Failure

The next publisher-TRAIN-only intervention started from checkpoint 16 and used
256 previously untouched eligible rows. One four-A100 run made 16 global-batch-
16 updates at learning rate `1.5e-5`, with a `0.995` post-step pull toward the
starting checkpoint. Each row combined unit-weight gold teacher-forced CE with
weight-`0.5` unlikelihood on only the decimal token or tokens of one inserted
false boundary under the same correct online state. JSON syntax was never a
negative target. Of the 256 negatives, 199 inserted boundary `1`; the loss
penalized 270 false-boundary digit tokens in total. The run changed only the
126 content-gate tensors, ended `0.0499` L2 from checkpoint 16, and was fixed
and pushed before generation.

| TRAIN-derived scene candidate | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Precision-unlikelihood endpoint | 76 | 202 | 164 | 0.2934 |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |

The candidate lost `0.0219` micro-F1 from checkpoint 16 and missed the locked
gain over V9 by `0.0019`. It changed 23 of 220 outputs: nine additions, seven
removals, and seven substitutions. Those changes added 30 false boundaries
while removing only 10, lost four true boundaries while adding one, and
included one unstable row that added all 15 boundaries from `2` through `16`.
Boundary `1` itself was not calibrated: five false instances were removed but
five new false instances appeared, while two true instances were lost and one
was added. Teacher-forced digit-only unlikelihood therefore did not transfer
to greedy-set precision; combined gold CE also failed to preserve the local
checkpoint-16 decision surface. The endpoint is archived without external
replication. Publisher validation, publisher test, Hard32, and the unused
73-row strength holdout remain unopened.

Evidence:

- [Locked precision-unlikelihood protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_precision_unlikelihood_protocol_v1.json)
- [Signed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_precision_unlikelihood_train_v1/result.json)
- [Signed materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_precision_unlikelihood_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_precision_unlikelihood_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_precision_unlikelihood.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_precision_unlikelihood_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_precision_unlikelihood.py)

### On-Policy First-Divergence Repair Failure

A materially different publisher-TRAIN-only run mined 96 previously untouched
eligible rows from frozen checkpoint 16 before optimization. It updated only
the 126 content-gate tensors for six global-batch-16 steps on four A100 GPUs.
Rows with a generated false-positive boundary contributed one pairwise loss at
the first divergence from gold; there was no full-sequence gold CE and no
synthetic-negative unlikelihood. Learning rate `5e-6`, gradient clipping at
`0.05`, and a `0.99` checkpoint-relative retention kept the final move to only
`0.00586` L2. The run found 53 actionable repairs representing 62 false
boundaries, changed no non-gate state, and was fixed and pushed before
generation.

| TRAIN-derived scene candidate | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| On-policy repair endpoint | 82 | 200 | 158 | 0.3142 |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |

The endpoint changed 14 of 220 outputs and improved recall, gaining three true
boundaries without losing any. It nevertheless added 18 net false positives
and missed checkpoint 16 by `0.00119` micro-F1. One row caused almost the whole
failure: source row 187 changed `[1]` into every boundary `[1, ..., 16]`, adding
15 false positives. Excluding that diagnostic row, the endpoint would score
`0.3235`, but that is not the benchmark result and no row is excluded from the
signed verdict. First-divergence repair found a useful local direction but did
not constrain the rest of the generated trajectory. The next intervention
must preserve checkpoint-16 continuation behavior while applying repairs,
rather than adding stronger global precision pressure. No protected split is
authorized or opened.

Evidence:

- [Locked on-policy repair protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_onpolicy_repair_protocol_v1.json)
- [Signed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_onpolicy_repair_train_v1/result.json)
- [Signed materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_onpolicy_repair_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_onpolicy_repair_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_onpolicy_repair.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_onpolicy_repair_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_onpolicy_repair.py)

### Adaptive Gold-Suffix Repair Failure

A follow-up development study reused the same 96 publisher-TRAIN rows after
observing the on-policy result, so it is explicitly adaptive and cannot
authorize external replication or protected evaluation. It retained the
first-divergence pairwise objective and added weight-`0.25` teacher-forced CE
on as many as four gold tokens beginning at the divergence. Six four-A100
updates used learning rate `2.5e-6`, clipping at `0.025`, and `0.995`
checkpoint-relative retention. The final content-gate move was only `0.00302`
L2 from checkpoint 16 and was fixed before generation.

| Adaptive TRAIN-derived scene candidate | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Gold-suffix repair endpoint | 80 | 197 | 160 | 0.3095 |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |

The endpoint changed 12 of 220 outputs, for a net gain of one true positive
but 15 false positives. It trailed checkpoint 16 by `0.00589` micro-F1 while
remaining `0.01910` above V9. Crucially, source row 187 still changed `[1]`
into `[1, ..., 16]`: gold-suffix CE trained the corrected branch, but did not
constrain the generated continuation after the model still selected the
wrong `[1` branch. The signed result receipt is
`75bff5880533653f8748bffb5dda8386335a7bee7df668fbfa2c853555caff02`.
Publisher validation, publisher test, Hard32, and the unused 73-row holdout
remain sealed.

Evidence:

- [Locked suffix-repair protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_suffix_repair_protocol_v1.json)
- [Signed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_suffix_repair_train_v1/result.json)
- [Signed materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_suffix_repair_materialization_v1/result.json)
- [Signed adaptive TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_suffix_repair_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_suffix_repair.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_suffix_repair_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_suffix_repair.py)

### Adaptive Dual-Path Continuation Repair Failure

The next adaptive study kept the original first-divergence pairwise correction
but added a second forward/backward path for each actionable row. That path
conditioned on the checkpoint-16 prefix through the original wrong token and
distilled as many as three subsequent checkpoint tokens with weight `0.25`.
This directly constrained behavior after a wrong branch instead of training
only the corrected branch. The same reused 96 TRAIN rows produced 53 repairs
over six four-A100 updates at learning rate `5e-6`, clipping `0.05`, and
retention `0.99`. The final content-gate move was `0.00591` L2.

| Adaptive TRAIN-derived scene candidate | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Dual-path repair endpoint | 79 | 186 | 161 | 0.3129 |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |

The intervention solved its targeted failure: source row 187 remained the
checkpoint output `[1]` instead of expanding to `[1, ..., 16]`. It changed
only seven of 220 outputs and removed false boundaries on two rows, including
an exact correction from `[1, 10]` to `[10]` on source row 190. However, it
added six false boundaries on five other rows, changed no true-positive
count, and ended four false positives above checkpoint 16. Micro-F1 therefore
remained `0.00250` below checkpoint 16, while exceeding V9 by `0.02249`.
Wrong-branch continuation distillation is a useful stability constraint, but
at this weight it suppressed the on-policy endpoint's three-TP recall gain.
The signed result receipt is
`08d8359f4aaeb7f9e5367c0999abd6ce2e74a4cb73dfd9ce9b7a32c3887e5f58`.
No external replication or protected evaluation is authorized.

Evidence:

- [Locked dual-path protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_dualpath_repair_protocol_v1.json)
- [Signed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_dualpath_repair_train_v1/result.json)
- [Signed materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_dualpath_repair_materialization_v1/result.json)
- [Signed adaptive TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_dualpath_repair_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_dualpath_repair.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_dualpath_repair_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_dualpath_repair.py)

### Adaptive Convex Repair-Bridge Failure

The next TRAIN-only study tested whether the recall and continuation behavior
could be separated in weight space without another training run. It
materialized exact convex gate-only bridges between the frozen on-policy and
dual-path endpoints at 25%, 50%, and 75% on-policy weight. All three recipes,
source artifacts, 126 tensors, and 108,906 values were hash-bound before four
A100s generated any outputs. A fresh five-fold cross-fit selected among the
three bridges and unchanged checkpoint 16 using only the other four folds.

| Adaptive TRAIN-derived scene candidate | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 25% on-policy / 75% dual-path | 78 | 183 | 162 | 0.3114 |
| 50% on-policy / 50% dual-path | 79 | 198 | 161 | 0.3056 |
| 75% on-policy / 25% dual-path | 81 | 196 | 159 | 0.3133 |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |
| Five-fold out-of-fold selection | 79 | 196 | 161 | 0.3068 |

The interpolation exposed a discrete generation threshold rather than a
smooth precision/recall frontier. Source row 187 stayed at `[1]` with 25%
on-policy weight, but both 50% and 75% abruptly regenerated every boundary
`[1, ..., 16]`. The 75% bridge recovered two of the on-policy endpoint's three
true positives, yet its 14 extra false positives left it `0.00202` micro-F1
below checkpoint 16. Cross-fit selected a learned bridge in only one of five
folds and trailed checkpoint 16 by `0.00857`; the signed study therefore
failed. The result rules out global linear interpolation as the next step and
motivates conditional inference on the small set of rows where the endpoints
disagree. Its receipt is
`98eb982c239dd7e845e597d7294aef5aeeab3e18c906505205d0373521f44717`.
No protected split was opened or authorized.

Evidence:

- [Locked bridge protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_repair_bridge_protocol_v1.json)
- [Signed materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_repair_bridge_materialization_v1/result.json)
- [Four-GPU raw generations](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_repair_bridge_eval_v1)
- [Signed adaptive TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_repair_bridge_v1/result.json)
- [Materializer](experiments/rethinking_rwkv_ms_gemma/materialize_natural_memory_native_scene_repair_bridges.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_repair_bridge.py),
  and [cross-fit analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_repair_bridge.py)

### Adaptive Consistency-Router Success

The first method to clear the locked `+0.005` improvement threshold on the
native TRAIN-derived scene benchmark is a conservative dual-pass consistency
router rather than another globally trained endpoint. It compares checkpoint
16 with the recall-heavier 75% on-policy bridge and accepts the proposal only
when it is a strict subset of the checkpoint prediction, or when checkpoint
abstains and the proposal is a single boundary. The rule never sees gold at
inference. It was designed after inspecting the bridge failure, so it is
explicitly post-hoc adaptive evidence, not an independent benchmark claim.

| Adaptive TRAIN-derived scene policy | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Strict-subset routing only | 79 | 178 | 161 | 0.3179 |
| Abstention-singleton routing only | 81 | 182 | 159 | 0.3221 |
| Combined consistency router | **81** | **178** | **159** | **0.3246** |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |

A fresh five-fold cross-fit selected the combined rule on all five fit
partitions, so its out-of-fold result equals the aggregate result. The router
changed six of 220 outputs: two singleton proposals repaired missed true
boundaries on rows 89 and 178, while four strict-subset proposals removed one
false boundary each on rows 190, 305, 321, and 354. This gives `+2` TP, `-4`
FP, and `+0.00928` micro-F1 over checkpoint 16. It also avoids the row-187
cascade because the 16-boundary proposal is neither a strict subset nor a
singleton after abstention.

This is a real success on the open publisher-TRAIN-derived native development
benchmark, but it does not yet establish generalization: the rule family was
created after observing those rows, requires two generation passes, and has
not been tested on any protected split. The next boundary is a pre-registered
replication on genuinely new data, followed by distilling the decision into a
single-pass confidence head only if replication succeeds. The signed receipt
is `c7f2d6af754c843cea5abbc8cf96415182d4ae97838586fdbea8d46a15ba049a`.

Evidence:

- [Locked adaptive router protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_consistency_router_protocol_v1.json)
- [Signed TRAIN-only success](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_consistency_router_v1/result.json)
- [Router and cross-fit analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_consistency_router.py)

This repository starts from the Log-Linear Attention codebase and adds a
CPU-only proof of concept in `dla_poc.py`. It reproduces the core DLA mechanism
from arXiv 2606.10650 and adds HRM-Text-inspired memory baselines:

- `rwkv_mem(delta_rule)`: single online delta-rule associative memory.
- `rwkv_mem(rwkv7)`: single read-before-write RWKV-7 state.
- `rwkv_mem(rwkv7 multi-state)`: same RWKV-7 state update, but one state per
  adaptive memory block.
- State-only ablation: fixes the exact same boundaries for linear/DLA states
  and RWKV-7 states, so the comparison isolates the state update.

## Quick Start

```bash
python3.12 -m venv .venv
PATH="$PWD/.venv/bin:$PATH" bash run.sh
```

If the environment is already set up:

```bash
.venv/bin/python dla_poc.py
```

Outputs are written to:

```text
EVAL.md
.openresearch/artifacts/dla_summary.json
.openresearch/artifacts/dla_trials.jsonl
.openresearch/artifacts/dla_comparison.png
.openresearch/artifacts/run_log.txt
```

## Mechanism-Level Result

The main DLA reproduction still passes:

- DLA lowers the Theorem 3.1 deviation bound in every tested config.
- DLA beats fixed Log-Linear blocking on needle recall at matched state count.
- The repo Log-Linear attention smoke test passes on CPU.

Mechanism recall comparison:

| needles | filler/seg | K | states | fixed | rwkv_mem(delta_rule) | rwkv_mem(rwkv7) | rwkv_mem(rwkv7 multi-state) | DLA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8 | 16 | 12.0 | 0.920 | 0.229 | 0.797 | 1.000 | 1.000 |
| 10 | 6 | 24 | 20.0 | 0.934 | 0.122 | 0.682 | 1.000 | 1.000 |
| 8 | 10 | 20 | 16.0 | 0.887 | 0.046 | 0.626 | 1.000 | 1.000 |

State-update-only comparison, with identical boundaries for both states:

| boundary policy | needles | filler/seg | K | states | linear/DLA state | RWKV-7 state | RWKV - linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | 8 | 12 | 16 | 16.0 | 1.000 | 1.000 | +0.000 |
| dla | 8 | 12 | 16 | 16.0 | 1.000 | 1.000 | +0.000 |
| fixed | 8 | 12 | 16 | 16.0 | 0.848 | 0.980 | +0.133 |
| noisy_dla | 8 | 12 | 16 | 16.0 | 0.874 | 0.987 | +0.112 |
| low_k_dla | 8 | 12 | 16 | 8.0 | 0.640 | 0.952 | +0.313 |
| oracle | 12 | 10 | 16 | 24.0 | 1.000 | 1.000 | +0.000 |
| dla | 12 | 10 | 16 | 16.0 | 0.792 | 0.982 | +0.190 |
| fixed | 12 | 10 | 16 | 16.0 | 0.763 | 0.991 | +0.228 |
| noisy_dla | 12 | 10 | 16 | 16.0 | 0.691 | 0.973 | +0.282 |
| low_k_dla | 12 | 10 | 16 | 8.0 | 0.516 | 0.889 | +0.373 |
| oracle | 16 | 8 | 12 | 32.0 | 1.000 | 1.000 | +0.000 |
| dla | 16 | 8 | 12 | 12.0 | 0.556 | 0.827 | +0.272 |
| fixed | 16 | 8 | 12 | 12.0 | 0.649 | 0.972 | +0.324 |
| noisy_dla | 16 | 8 | 12 | 12.0 | 0.509 | 0.819 | +0.311 |
| low_k_dla | 16 | 8 | 12 | 6.0 | 0.371 | 0.669 | +0.299 |

This table fixes the exact same token blocks for both methods. `linear/DLA state`
uses the standard block sum `sum k_t v_t^T`; `RWKV-7 state` uses the RWKV-7
recurrence inside each same block. Therefore each row compares state
update/readout only, not boundary quality.

Interpretation:

- With perfect or near-perfect boundaries, linear/DLA state and RWKV-7 state tie
  on this synthetic recall task.
- When boundaries are fixed, noisy, or compressed to low K, RWKV-7 state is more
  robust in this task.
- DLA's main advantage is adaptive boundary/state allocation; RWKV-7's advantage
  appears in the state update when boundaries are held fixed and imperfect.

Full tables are in `EVAL.md`.

## What Is Compared

`dla_poc.py` runs four groups of checks.

1. Codebase smoke test
   - Loads the original Log-Linear Attention pure PyTorch path directly.
   - Avoids CUDA-only Triton/Mamba dependencies.

2. DLA deviation-bound check
   - Implements Algorithm 1: information-aware dynamic state merging.
   - Implements Algorithm 2: capacity-bounded adjacent state merging.
   - Compares DLA blocks against fixed contiguous blocks at matched state count.

3. Needle associative recall
   - Uses synthetic rare needle tokens mixed with redundant filler tokens.
   - Compares fixed blocks, DLA blocks, delta-rule memory, RWKV-7 memory, and
     multi-state RWKV-7 memory.

4. State-update-only ablation
   - Uses the same block boundaries for both linear/DLA and RWKV-7 state update.
   - Boundary policies: `oracle`, `dla`, `fixed`, `noisy_dla`, `low_k_dla`.
   - This isolates whether the state update/readout is stronger, independent of
     boundary selection.

## Scope

The top-level DLA comparison is a training-free mechanism reproduction. It does
not reproduce 50B-token pretraining or trained HRM-Text checkpoints. The
separate Gemma4 online-memory experiments documented above and under
`experiments/rethinking_rwkv_ms_gemma/` include both projected-slot and
recurrent RWKV-MS readouts; the native validated V9 line is the projected-slot
variant described in “Native Mechanism Accounting.”

The HRM/RWKV baselines are self-contained ports of the memory recurrence ideas,
not full imports of HRM-Text:

- `rwkv_mem(delta_rule)` follows the read-before-write delta-rule associative
  state from HRM-Text's `models/rwkv_memory.py`.
- `rwkv_mem(rwkv7)` follows the latest read-before-write RWKV-7 recurrence from
  HRM-Text's `models/rwkv7.py`, specialized to the synthetic key/value stream.

## Repository Layout

```text
dla_poc.py                         # Main reproduction and comparisons
run.sh                             # CPU dependency install + run
EVAL.md                            # Generated report from latest run
.openresearch/artifacts/           # JSONL, JSON, figure, run log
hattention/                        # Log-Linear Attention implementation used for smoke test
figs/                              # Original figure asset
deltamem/                          # bundled patched HF online-memory runtime
integrations/delta_mem_rwkv_ms/    # launchers, docs, GGUF tools, optional upstream patch
integrations/delta_mem_rwkv_ms/gguf/ # GGUF sidecar, fixture, and parity helpers
```

## HOLA Hippocampus on RWKV-7 Multi-State

`experiments/hola_hippocampus/` replaces the neocortex of HOLA (arXiv 2607.02303,
semiparametric memory = compressive state + bounded exact-KV cache) with this repo's
read-before-write RWKV-7 multi-state and re-tests HOLA's design claims on the
state-only ablation grid. The mapping is exact: for unit keys the RWKV-7 correction
term makes the update a delta rule, so HOLA's surprise score beta*||e|| becomes the
write magnitude `m_t = ||Delta_t||_F` already computed by the recurrence.

Result summary (5 seeds; full tables in `experiments/hola_hippocampus/REPORT.md`):

- The weakest state-only cell (16 needles, `low_k_dla`, 0.669 above) rises to
  **0.880** with a 16-slot surprise cache; a matched recency cache stays at 0.665.
- HOLA's two claims reproduce on RWKV-7: recency caching is dead weight for far
  needles, and a flat softmax read (0.83*cos) equals no cache at all.
- One correction was required: raw surprise admission fails with an untrained
  constant gate; an online CLS-style consolidation rule (demote cache entries whose
  key the state later predicts well) plus a read-confidence gate makes the cache
  strictly non-harmful. Hypothesis ledger and run provenance live in `.keel/`.

Run: `.venv/bin/python experiments/hola_hippocampus/hola_rwkv_ms.py`

## MARCH-Inspired Historical State Anchors

The RWKV-MS runtime now includes an optional, default-off historical-anchor
residual inspired by MARCH (arXiv:2608.12435). Existing RWKV-MS keeps a fixed
bank of current-time slots; the extension periodically snapshots the full slot
bank, assigns it a compact learned key, and retrieves earlier snapshots through
a token-conditioned router with a learned null route. This composes temporal
state expansion with the repository's existing parallel-slot expansion.

See [`docs/MARCH_RWKV_MS_COMPARISON.md`](docs/MARCH_RWKV_MS_COMPARISON.md) for
the mathematical mapping, exact differences, configuration, evidence boundary,
and proposed experiments. The feature remains disabled when
`rwkv_ms_anchor_interval=0`.

## Delta-Mem RWKV-MS Online Memory

The practical RWKV-MS online-memory integration is self-contained in this
repository. The patched Python runtime is bundled at top-level `deltamem/`, so
normal Qwen/Gemma HF training and inference do not require another delta-Mem
checkout. `integrations/delta_mem_rwkv_ms/` contains HF inference and verified
manual training-smoke entry points, a matched delta-rule/RWKV-MS launcher, GGUF
tools, and an optional upstream patch export. The runtime supports Qwen3,
Qwen3.5/Qwen3.6, SmolLM3, and Gemma4 text attention;
for `google/gemma-4-E4B-it` it wraps the non-KV-shared attention layers and
skips the KV-shared tail layers.

Fresh RWKV-MS configs use semantics v2: FP32 recurrent matrices, bounded
per-head write sources, RWKV decay without a second lambda decay, cosine slot
routing, and a bias-free empty-state readout. Checkpoints without an explicit
`rwkv_ms_semantics_version` load as legacy v1 and must not be resumed as v2;
start a fresh run with `--rwkv-ms-semantics-version 2`.

Transformers exposes Qwen3.6 as `qwen3_5`. Its 64-layer hybrid stack has 16
full-attention layers at physical indices `3,7,11,...,63`; the other 48 Gated
DeltaNet layers are not wrapped. Use layer `3` for a smoke run or
`3,7,11,15,19,23` for the six early eligible layers. This Qwen path is the HF
integration and is separate from the Gemma-only GGUF sidecar runtime.

The bundled `deltamem/` package provides the wrapper/session machinery:
attaching online-memory modules to a Transformers model, loading
`delta_mem_adapter.pt`, keeping RWKV-MS state synchronized with the KV cache,
and applying the chat template. The optional
`integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch` exports these changes
for upstream delta-Mem revision `5cd5d9153c7f408764728d953565201e198c39e2`;
it is not needed for normal use of this repository.
See [bundled runtime provenance](integrations/delta_mem_rwkv_ms/BUNDLED_RUNTIME.md)
for the source snapshot and local integration revision.

For HF workflows, install the bundled package from the repository root before
running the commands below:

```bash
pip install -r requirements.txt
pip install -e .
```

### Gemma tau2 status

The active Gemma + RWKV-MS tau2 recipe is documented in
`GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md`. For reproducibility, the benchmark
artifacts record this historical source integration commit (it is not a current
external runtime dependency):

```text
bec8330 Add RWKV-MS memory backend for Gemma tau2
```

Current best learned no-rule online-memory checkpoint:

```text
xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
```

Local source checkpoint:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100
```

The table below keeps learned online-memory runs separate from rule-assisted diagnostic
runs. "No-rule" means no eval-time `--mobile-data-rule-planner` and no parser
format-repair patch.

Release framing: "One can have both the fish and the bear's paw." The base
Gemma checkpoint remains frozen to preserve original behavior, while the learned
RWKV-MS path adds a small recurrent memory surface that can be adapted to local
domain data.

| Run / condition | Layers / rank / length | pass^1 | Takeaway |
| --- | --- | ---: | --- |
| Base checkpoint `google/gemma-4-E4B-it`, focused tools + line verify + autostop | none | 4/20 (0.20) | Current base-only baseline for the accepted setup |
| Base checkpoint `google/gemma-4-E4B-it`, checklist prompt | none | 7/20 (0.35) | Prompt-only baseline, still below learned best |
| Original 82-row Phase 1 | `0,1` / r8 / len256 | 1/20 (0.05) | Dataset/format mismatch; reject |
| Generated action SFT | `0,1` / r8 / len256 | 9/20 (0.45) | 2 layers help but are not enough |
| Generated action SFT | `0-5` / r8 / len256 | 10/20 (0.50) | Shallow 6-layer band is better |
| Generated action SFT | all eligible / r4 / len256 | 1/20 (0.05) | All-layer memory path over-perturbs |
| Format-refresh continuation, final | `0-5` / r8 / len192 | 12/20 (0.60) | Good final checkpoint |
| Format-refresh continuation, `step-100` | `0-5` / r8 / len192 | **14/20 (0.70)** | Best learned no-rule checkpoint |

Memory-path size from saved checkpoints:

| Memory-path shape | Trainable memory params |
| --- | ---: |
| 2 layers, r8 `q,o` | 257,744 |
| 6 layers, r8 `q,o` | 797,808 |
| 24 eligible layers, r4 `q,o` | 1,594,080 |

Local training-cost notes:

- Experiments were local CUDA bf16 runs on an RTX 4090 24 GB setup.
- Generated mobile-data action SFT used 3,519 turn rows for 656 optimizer steps.
- Format-refresh continuation used 5,027 turn rows for 200 optimizer steps.
- Exact wall time and VRAM vary with local hardware, sequence length, layer
  count, rank, cache location, and fragmentation; adapt the frozen-base
  online-memory recipe to your own data.

Status interpretation:

- The original tau2 data was the problem: the 82-row run trained for 656
  optimizer steps and its loss moved, but the benchmark collapsed to 1/20.
- Generated mobile-data action SFT transfers better, and the 6-layer shallow
  online-memory path is the current useful capacity point.
- The 200-step format-refresh continuation overtrains relative to its
  `step-100` checkpoint, so checkpoint selection matters.
- The eval-time rule planner / float-format fix is excluded from the comparison
  table because it is benchmark-specific control logic, not model behavior.
- The next benchmark should run the `step-100` checkpoint on at least 50 tasks,
  preferably the full telecom split, before treating 14/20 as robust.

Recommended HF online-memory inference command:

```bash
python integrations/delta_mem_rwkv_ms/inference.py \
  --memory-repo xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --base-model google/gemma-4-E4B-it \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa
```

## Gemma4 GGUF First Step

A base Gemma4 E4B GGUF has been downloaded for llama.cpp testing on the 2 TB SSD:

```text
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/mmproj-gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
```

The first two files are normal base-model inference artifacts. The RWKV-MS
memory file is a GGUF sidecar containing the adapter tensors and metadata. The
local llama.cpp branch can now consume that sidecar in an experimental Gemma4
runtime path: model load owns the sidecar tensors in CPU buffers, the Gemma4
graph applies RWKV-MS `q,o` deltas on target layers `0-5`, and a mutable
RWKV-MS state buffer is updated during prompt/generation scans. The current
runtime is intentionally constrained to one sequence. The server/UI path keeps
physical microbatches serial (`-ub 1`) for the best-tested state behavior; the
CLI graph can build experimental graph-unrolled multi-token prompt scans, but
that path still needs stronger state-level parity coverage before it should be
treated as production-ready. See
`GGUF_EXTERNAL_MEMORY_FEASIBILITY.md` and
`integrations/delta_mem_rwkv_ms/GGUF_PORT_PLAN.md`.
At llama.cpp model load time, the sidecar path now performs semantic validation
before runtime use: it verifies `delta_mem.base_gguf_sha256` against the exact
loaded base GGUF file, then rejects unsupported `num_state_heads != 1`,
duplicate compact tensor names, missing required tensors, and wrong ggml-order
tensor shapes.

Patched llama.cpp fork:

```text
https://github.com/xiaol/llama.cpp-online-memory
branch: main
commit: 85da0c63b Add Gemma4 RWKV-MS GGUF sidecar runtime
base upstream: ggml-org/llama.cpp 1ec44d1
```

Current sidecar identity:

```text
sha256: 0c646a776b5b12c9d3657ffd2e5e581be1eb46e858f1f404afeaa7077c02974e
bound base GGUF sha256: fb8f0c032de00b18c710824af3c7e5777c71e5fb60b13f13575f0a9e92ddecd0
size: 1,663,840 bytes
tensor name format: compact_with_source_name_manifest
tensors: 186 BF16
```

Start a recent llama.cpp server:

```bash
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh
```

For the experimental RWKV-MS sidecar runtime through `llama-server`, use the
patched llama.cpp build and the constrained sidecar mode:

```bash
mkdir -p .openresearch/artifacts/gguf_ui
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_PORT=18083 \
LLAMA_RWKV_MS=1 \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh 2>&1 | tee .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log
```

The helper sets `--rwkv-ms-sidecar`, `--batch-size 2`, `--ubatch-size 1`,
`--parallel 1`, disables continuous batching/context shift/prompt-cache reuse,
disables server prompt-cache RAM and context checkpoints, enables a slot-save
directory for manual slot 0 save/restore, and uses text-only mode for the
current one-sequence runtime.
The patched llama.cpp context rejects sidecar runs with more than one sequence.
The server/helper keep `--ubatch-size 1`, reject speculative decoding, and
preflight unsafe slot/cache/batch overrides before starting `llama-server`;
model load also rejects malformed or unsupported sidecars before any RWKV-MS
graph consumes their tensors. A sidecar exported for a different base GGUF now
fails model load with a hash mismatch instead of running against the wrong
weights.

For the best-tested experimental runtime path, pass the sidecar and use serial
physical microbatches:

```bash
/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-completion \
  -m /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf \
  --rwkv-ms-sidecar /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  -p "Hi" -n 32 -c 96 -b 2 -ub 1 -ngl 99 --no-warmup --no-display-prompt \
  --no-perf -no-cnv -s 123 --temp 0 --top-k 1
```

With the same seed and greedy sampling, the base and sidecar paths now diverge.
The sidecar path produced `! I'm excited to chat with you. What's on your mind
today? ...`, while the base path continued `! I'm excited to chat with you. I'm
here to help ...`. Treat this as a smoke signal consistent with the sidecar path;
confirm runtime use with server logs and the reference-trace health check.

The local CUDA build is from the online-memory fork commit `85da0c63b`, based
on upstream llama.cpp `1ec44d1`, and detects the RTX 4090 as `CUDA0`. CUDA 13.1
plus GCC 15 needed a local header shim during build; the resulting binary is
under the SSD tool directory above.

Then launch the local testing UI:

```bash
python3.12 -m venv .venv-ui
.venv-ui/bin/pip install -r requirements-ui.txt
LLAMA_BASE_URL=http://127.0.0.1:18083/v1 \
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
GGUF_RWKV_MS_SIDECAR_PATH=/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
LLAMA_SERVER_LOG=.openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
GGUF_RWKV_MS_HEALTH_OUTPUT=.openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json \
GGUF_UI_REQUIRE_RWKV_MS_HEALTH=1 \
GGUF_UI_PORT=7861 \
.venv-ui/bin/python tools/gemma_gguf_ui.py
```

Before comparing prompts, verify that the endpoint is really the patched
sidecar runtime. The UI exposes the same check through its RWKV-MS runtime
button, writes the health file, and blocks sidecar chat/trace comparison while
the selected endpoint/model/sidecar/log do not match a recent successful check.

```bash
.venv-ui/bin/python tools/check_rwkv_ms_gguf_runtime.py \
  --base-url http://127.0.0.1:18083/v1 \
  --server-log .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
  --output .openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json
```

The check requires the server log because API output alone cannot prove that
llama.cpp loaded the RWKV-MS sidecar. It verifies model listing, a chat smoke
request, the saved reference trace, slot 0 save/restore with exact-prefix
continuation, corrupted slot restore rejection, and log evidence for RWKV-MS
activation, one server slot, disabled prompt cache, disabled context
checkpoints, and exact-prefix slot reuse. The sidecar server also rejects
speculative decoding options.

For repeatable prompt checks against the same server:

```bash
.venv-ui/bin/python tools/eval_gguf_prompts.py configs/gguf_rwkv_ms_prompt_suite.jsonl \
  --base-url http://127.0.0.1:18083/v1 \
  --model gemma-4-e4b-it-rwkv-ms-q8 \
  --rwkv-ms \
  --temperature 0 \
  --seed 42
```

For the RWKV-MS side of the future port, inspect the PyTorch memory checkpoint
into a tensor/config manifest:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_checkpoint.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --output .openresearch/artifacts/gguf_memory_manifest.json
```

To regenerate and validate the GGUF memory sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/export_memory_gguf.py \
  --manifest-output .openresearch/artifacts/rwkv_ms_memory_sidecar_manifest.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_gguf.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/materialize_memory_gguf.py --force
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/compare_memory_checkpoints.py
```

To generate and validate the isolated RWKV-MS math fixture from the
sidecar-rebuilt checkpoint:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/generate_rwkv_ms_math_fixture.py \
  --output .openresearch/artifacts/rwkv_ms_math_fixture.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/validate_rwkv_ms_math_fixture.py \
  --fixture .openresearch/artifacts/rwkv_ms_math_fixture.json \
  --json
```

The current fixture uses real layer-0 adapter tensors, covers projection,
read-before-write state update, readout, and active `q,o` delta heads, and
validates with `max_abs_diff: 0.0`. It is a PyTorch golden math fixture for a
future GGML port, not stock llama.cpp memory execution.

The local llama.cpp checkout has an isolated C++ fixture for the compact
sidecar:

```bash
/run/media/xiaol/B214449214445C0B/tools/cmake-4.3.3/bin/cmake \
  --build /run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda \
  --target test-rwkv-ms-fixture -j 8

/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/test-rwkv-ms-fixture \
  .openresearch/artifacts/rwkv_ms_math_fixture.json \
  /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  1e-5 1e-5
```

Current strict sidecar result: `{"ok":true,"compared":51,"sidecar":true,"max_abs_diff":1.37090683e-06}`.
The no-sidecar run also passes with `compared=11` and `max_abs_diff=5.96046448e-08`.
This covers `tests/test-rwkv-ms-fixture.cpp` in llama.cpp parsing the compact
sidecar, computing memory projections, `HRMRWKV7LowRankCore` feature
projections, driving a second C++ read-before-write scan from those
sidecar/GGML tensors, graph readout from the scan `raw_reads` plus graph
`feature_g`, and `delta_q`/`delta_o` from the graph-produced readout. This
fixture remains the isolated math parity check; the separate `llama-completion`
smoke above is the Gemma4 generation runtime check.

The local llama.cpp checkout also has `tests/test-rwkv-ms-state.cpp` for the
RWKV-MS recurrent state payload. It checks v2 state metadata, deterministic
sidecar fingerprint validation, staged sidecar-local restore, and rejection for
metadata/fingerprint/length mismatches. The fingerprint now includes the bound
base GGUF hash, so slot files created before that binding should be regenerated.
Full and sequence state restore now snapshot the current context before
RWKV-MS-enabled loads and roll back that snapshot if the normal memory portion
loads but the RWKV-MS sub-state fails. Failed server slot restore still clears
the affected slot/context state after the library rollback and returns the
exact state-load error.
Context-owned memory mutation now uses llama.cpp `llama_context_memory_*`
wrappers in the patched paths: clear and supported full-sequence removal keep
RWKV-MS state synchronized, while unsupported sequence copy, keep, shift, and
division fail explicitly under RWKV-MS instead of mutating only KV cache.

To generate the first PyTorch golden trace from the sidecar-rebuilt checkpoint:

```bash
.venv/bin/python \
  integrations/delta_mem_rwkv_ms/gguf/generate_reference_trace.py \
  --max-new-tokens 64 \
  --output .openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json \
  --save-snapshot-dir .openresearch/artifacts/gguf_reference_snapshot_from_sidecar_64
```

To compare the running GGUF backend against that reference trace:

```bash
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
.venv-ui/bin/python tools/compare_gguf_to_reference_trace.py \
  --output .openresearch/artifacts/gguf_ui/trace_compare_reasoning_off.jsonl
```

With `LLAMA_REASONING=off`, the comparison harness can log either base-GGUF or
RWKV-MS-sidecar runs. Stock llama.cpp still does not execute RWKV-MS memory; the
sidecar mode requires the local patched branch.

## Acknowledgement

This work builds on the Log-Linear Attention repository and uses local
HRM-Text/RWKV memory ideas as mechanism baselines. The added experiments are
intended for controlled research exploration, not as a trained-model benchmark.
