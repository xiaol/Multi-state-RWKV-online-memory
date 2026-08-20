# Rethinking RWKV-MS Gemma Experiments

This folder adapts the experiment logic from "Rethinking the Role of Efficient
Attention in Hybrid Architectures" to the local Gemma + RWKV-MS online-memory
hybrid.

The target question is:

> Does RWKV-MS carry long-range information, or does it mostly perturb/shape
> Gemma attention while full attention still carries retrieval?

## Current Architecture And Claim Boundary

The current native-memory candidate is not a pure RWKV retrieval system. It is
a hybrid with two persistent memories in every wrapped Gemma attention layer:

1. A projected key/value slot sidecar stores the material value and chooses a
   slot from the current query by cosine routing.
2. An RWKV-7 recurrent matrix state is written online into the corresponding
   fixed-chunk slots.
3. Candidate hybrids use the projected route to address the matching RWKV state
   slot. The recurrent read is normalized and used as a bounded elementwise
   controller:

   ```text
   direction = tanh(rwkv_read / rms(rwkv_read))
   output = projected_read * (1 + gain * controller * direction)
   ```

4. The fused read passes through the learned delta output path and a
   content-gated residual into frozen Gemma.

The projected key/value sidecar remains the primary material carrier. The
recurrent contribution is active under teacher forcing, but the causal
endpoints do not jointly establish correct-state preference over both donor
and layer-permuted controls. Native generation is blocked for these candidates.

### FFN branches

The addressed-MoE candidate also has an optional post-MLP path. At sparse
decoder anchors `(10, 21, 31, 41)`, the addressed/global RWKV correction is
normalized and passed through a small gated FFN, then added by a forward hook
after Gemma's frozen MLP and before the layer residual addition:

```text
control = (hybrid_read - projected_read) /
          (attention_gain * rms(projected_read))
outer = outer_gain * tanh(rms_norm(
    up(silu(down(rms_norm(control))) * sigmoid(gate(rms_norm(query))))
))
```

The four-A100 same-mode ablation kept the projected carrier, recurrent state,
routing, and attention gain fixed and changed only `outer_gain` from `1/8192`
to zero. Recurrent and carrier identity checks passed, but outer-on versus
outer-zero logit deltas were `1.46875`, `1.375`, `1.65625`, and `1.125`, above
the locked `0.5` bound. The signed result is
`local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen_v1/result.json`
(file SHA-256
`645f2dbe5098b23366797de154065677cd5fb57ad7e7bc5d733380ef0168c285`, receipt
`5c0e2babcd68aaeeb7db381f137c9abca183f716ddfe723fb35d42061c293278`). Its
status is `addressed_moe_outer_ffn_gain_ablation_screen_failed_branch_stopped`;
the causal endpoint and native generation remain closed.

The subsequent `addressed_moe_deepembed_ffn` branch does not add a residual
after the MLP. It preserves the all-layer addressed/global MoE attention path
and uses its normalized RWKV control to modulate Gemma's native MLP channels:

```text
state = silu(W_down * rms(recurrent_control))
gate = sigmoid(W_gate * rms(hidden))
modulation = rms(W_up * (state * gate))
scale = 1 + ffn_gain * tanh(Gemma_up_proj(modulation))
output = Gemma_down_proj(native_gated_mlp_activation * scale)
```

Gemma's weights stay frozen. Zero recurrent state gives a unit scale, so both
the attention fusion and ChannelMix path are exactly projected-only. The first
gain grid (`1/8192`, `1/4096`, `1/2048`) produced exactly zero BF16 final-logit
change versus FFN-gain zero. The BF16-resolvable grid selected the lowest
passing gain, `1/128`, together with attention gain `1/64`.

All-layer FFN training exceeded the 40 GiB A100 budget at update 6. The sparse
design therefore keeps recurrent attention in all 42 layers and instantiates
the three ChannelMix tensors only at anchors `(10, 21, 31, 41)`. Its four-A100
screen passed; the signed top-level result has SHA-256
`079be7f01b2c8e53199c1db4efeda4d66e4428b8fd2dc5ad4f41a1bbf61a3844` and
receipt
`bfdb7ef9d40683a87404243811548764cc4a7bdec8c3993d5409505ace04d275`.

The locked sparse causal run completed 16 updates in 602 seconds. It accepted
127 of 128 rows after filtering known non-finite ordinal `1291`, exercised all
390 selected trainable tensors with zero globally inactive tensors, and peaked
at `41,354,340,352` bytes on the busiest rank. The fresh 11-row endpoint was:

| recurrent condition | mean CE | margin versus correct | positive rows |
| --- | ---: | ---: | ---: |
| correct | 2.881166 | - | - |
| zero | 4.102686 | +1.221520 | 11 / 11 |
| matched donor | 2.877426 | **-0.003740** | 7 / 11 |
| layer-permuted | 3.112725 | +0.231559 | 11 / 11 |

Lower CE is better. DeepEmbed learned a strong dependence on state presence and
layer placement, but not on matched-donor identity. The endpoint status is
`addressed_moe_deepembed_ffn_sparse_heldout_failed_generation_blocked`; the
result SHA-256 is
`5067878c838b55ad953b606563ce0e21a290efe55d054bc3136fad9239b488ef` and
receipt is
`980123e096fb4125ebe0c8da98e25a0333404df4772131bf2b732b95effa4af7`.
Native generation stayed closed, so this branch establishes no native
benchmark gain.

Provenance limitation: this v1 result binds the delegated shared causal engine,
protocol, and Delta-Mem core, but the thin DeepEmbed wrapper did not include its
own file SHA in the signed payload. The repository commit pins that wrapper;
future result schemas should self-bind the top-level wrapper as well.

The subsequent `address_keyed_moe_deepembed_ffn` branch moved the intervention
upstream into RWKV itself. The selected projected slot key conditions recurrent
RWKV `k/v/a/b` write features, while the same all-layer addressed/global MoE
readout and sparse ChannelMix anchors remain active. The projected sidecar is
still the material carrier, but RWKV now has an explicit write-side identity
signal rather than only a routed state slot.

The v4 training execution stopped after five updates because three active
control graphs exceeded a 40 GiB A100 during update 6. The signed failure kept
the endpoint closed. The v5 execution serialized the donor and layer-permuted
control graphs without changing the candidate, schedule, data, objective, or
optimizer. It completed all 16 updates, accepted 128/128 rows, and passed its
fresh 11-row endpoint:

| recurrent condition | mean CE | margin versus correct | positive rows |
| --- | ---: | ---: | ---: |
| correct | 2.828835 | - | - |
| zero | 4.298508 | +1.469673 | 11 / 11 |
| matched donor | 2.835603 | +0.006768 | 6 / 11 |
| layer-permuted | 3.164019 | +0.335185 | 11 / 11 |

The separately locked native benchmark then evaluated 220 open
publisher-TRAIN-derived rows on four A100s. It compared the same checkpoint and
projected carrier under correct, zero, matched-donor, layer-permuted, and exact
projected-only conditions:

| recurrent condition | micro-F1 | precision | recall |
| --- | ---: | ---: | ---: |
| correct | 0.192212 | 0.119959 | 0.483333 |
| zero / projected-only | 0.194250 | 0.119389 | 0.520833 |
| matched donor | 0.195688 | 0.122153 | 0.491667 |
| layer-permuted | 0.187192 | 0.116564 | 0.475000 |

Coverage, fixed-carrier, and exact zero/bypass identity gates passed. Only the
correct-minus-layer-permuted margin passed (`+0.005020`). Correct-minus-bypass
was `-0.002038`, and correct-minus-donor was `-0.003476`, so the status is
`address_keyed_deepembed_native_gain_not_established`. The signed result file
SHA-256 is
`9435980573f845ee0fde3abff987ea39ffa106e597a4ee47d5c8f2e00f7f6aba`, and
its receipt is
`1c79acb43b7ee6fea75dc3579bccb06c9cf81fb4485d6d82e893f58d33fdae71`.
No native benchmark gain is established.

The correct state suppressed false positives (`851` versus `922`) but also
suppressed true positives (`116` versus `125`), while a matched donor slightly
outperformed the correct state. The fixed address perturbation therefore acts
like a generic decoding calibrator rather than a learned identity binding. The
next candidate should replace that parameter-free perturbation with learned
low-rank address-to-RWKV `k/v/a/b` transforms and directly supervise
query-to-state identity with an internal contrastive loss. A final
state-conditioned boundary-logit adapter is a separate fallback; another gain
increase in the existing attention or FFN path is not justified.

## Current Evidence

Earlier locked open-fit experiments used four A100 GPUs, 16 optimizer
updates, 128 accepted training rows, and fresh source/donor-disjoint 32-row
teacher-forced endpoints. All training and integrity gates passed, with no row
rejections and no protected split access.

| candidate / recurrent condition | mean CE | margin versus correct |
| --- | ---: | ---: |
| route agreement / correct | 2.636815 | - |
| route agreement / zero | 3.414512 | +0.777697 |
| route agreement / donor | 2.636736 | **-0.000079** |
| route agreement / layer-permuted | 2.642873 | +0.006058 |
| query-state gate / correct | 2.981633 | - |
| query-state gate / zero | 3.824999 | +0.843366 |
| query-state gate / donor | 2.974421 | **-0.007213** |
| query-state gate / layer-permuted | 2.962663 | **-0.018970** |

Lower CE is better. Zero recurrence is consistently worse, proving the RWKV
path is active. Route agreement is donor-neutral, while the query-state gate
is donor- and layer-unfavorable on its fresh endpoint. The signed results are
`local_artifacts/natural_memory_native_rwkv_addressed_route_agreement_causal_train_v1/result.json`
(file SHA-256 `fa665edfa75620de412c12f85e453cfdbb4f6f19fc15088fc0babc3bd1be2ca8`,
receipt `d4f2ff8f105fbba9d29c64d1e5e56c33e067f7a674babf4805be40144aa9f622`)
and
`local_artifacts/natural_memory_native_rwkv_addressed_query_state_gate_causal_train_v1/result.json`
(file SHA-256 `7bc255f1784c2e36df9ef2abd53903e634d3164e51f29631911fa2235b736343`,
receipt `a4f9bb3943a9e47cfd8457d58aea52ba9403998b5eddd440ae8d657b1d6b0512`).

The final addressed/global MoE rerun also completed 16 updates and all
integrity gates. On its fresh 32-row endpoint, zero-minus-correct CE was
`+0.882820` and donor-minus-correct was `+0.002533`, but
layer-permuted-minus-correct was `-0.000107`. The full causal gate therefore
failed and native generation stayed closed. The signed result is
`local_artifacts/natural_memory_native_rwkv_addressed_moe_controller_causal_train_v4/result.json`
(file SHA-256
`6b4f835e487eb01bdc4058013b00df0ec4e364e7a8bf19dada182c59e4e18df2`, receipt
`aaec56edc52ab77a207617685e8dc3c8ead1b550614763012fe0595e222e9e29`).

The separately locked 220-row native generation benchmark failed every causal
gain gate:

| recurrent condition | micro-F1 | margin from correct |
| --- | ---: | ---: |
| correct | 0.189507 | - |
| zero / projected-only | 0.198083 | -0.008576 |
| matched donor | 0.195004 | -0.005497 |
| layer-permuted | 0.196269 | -0.006763 |

Correct recurrence changed 13.64% of outputs from projected-only but reduced
precision and recall, so the recurrent effect was active and harmful under
autoregressive decoding. Coverage passed, zero and projected-only generations
were exact, and every projected carrier stayed byte-identical. The signed
native result is
`local_artifacts/natural_memory_native_rwkv_addressed_affine_eval_v1/result.json`
(file SHA-256
`43097d4bceef4eb5a4a760f146bf4e5f697e5294e3ba929b948c9fd12f4b6d73`,
receipt
`c18d190c8fffcbac142c4d95cce6899129df637685cc551ae0e78214710ecdde`).
No native RWKV gain is established.

The fixed address-keyed write plus DeepEmbed experiment above has now tested
write-side identity and failed native transfer. The remaining write-side move
is learned identity binding: train low-rank address projections into RWKV's
`k/v/a/b` features and enforce a direct query/state contrast before decoding.
The MoE attention and sparse DeepEmbed paths can remain controlled readout
ablations, but another gate or a larger gain does not target the repeated
donor-neutral failure.

The rank-2 learned-write follow-up added separate low-rank projected-address
transforms for RWKV `k/v/a/b`, initialized as an exact no-op. A four-A100,
global-batch-four run completed eight updates with all 726 selected tensors
active. The 32-row causal endpoint produced zero-minus-correct CE `+0.500323`
and layer-permuted-minus-correct `+0.211665`, but donor-minus-correct was
`-0.0000795`. The donor-specific gate therefore failed and native generation
remains blocked. The next experiment must supervise internal query/state
identity directly; scaling the learned write or outer FFN would not address
this near-zero donor margin.

The direct query/state identity run then retained that full hybrid and added a
parameter-free cosine hinge between a detached target projected-key address
and the addressed RWKV read. It completed all eight four-A100 updates with all
726 trainable tensors active and no carrier mutation. Training never opened a
meaningful identity margin: correct-minus-donor cosine was `+0.000504`, the
mean hinge remained `0.199496` against its `0.2` initialization, and every
training row kept the hinge active. The fresh 32-row endpoint was:

| condition / identity metric | held-out result |
| --- | ---: |
| zero-minus-correct CE | +0.499466 |
| donor-minus-correct CE | **-0.001470** |
| layer-permuted-minus-correct CE | +0.248370 |
| correct-minus-donor identity cosine | **-0.002735** |
| positive identity rows | **43.75%** |

The signed status is
`query_state_identity_heldout_failed_generation_blocked`, result SHA-256 is
`e1acc2d492339540dc89abcf2a1cfe619f5b55d9f8c459396bead19f84199b1a`,
and receipt is
`45684bccf63bd46908fff632790b8e0484e87c3bdfdfc0e5f293439df499221a`.
The result proves recurrence and layer placement remain material but rejects
this key/read compatibility target. Projected keys represent slot addresses;
RWKV reads live in the value space.

The next locked candidate should instead freeze the selected projected slot
**value** and contrast it with correct and donor RWKV reads in the same
32-dimensional space. Its hinge must be formed per answer token and layer,
then reduced, rather than averaging 42 layers before the hinge. Independent
correct/donor checkpoints, a detached active mask, serialized backward, the
learned rank-2 write, addressed MoE, and sparse DeepEmbed anchors remain fixed.
Only a fresh endpoint pass may reopen native generation.

That same-space projected-value run completed after the control-graph binding
was corrected. All eight four-A100 updates completed with serialized control
graphs, CPU optimizer-state offload, 726 active trainable tensors, and 32/32
identity rows. The training audit passed, but the fresh 16-row endpoint did not:

| held-out metric | result |
| --- | ---: |
| zero-minus-correct CE | +0.514081 |
| donor-minus-correct CE | **+0.002462** |
| layer-permuted-minus-correct CE | +0.211908 |
| correct-minus-donor projected-value cosine | **+0.000095** |
| positive identity rows | **43.75%** |

The signed status is
`projected_value_identity_heldout_failed_generation_blocked`, with result
receipt `c3f7faa1e286a4990ef1299624f3b325f279712981905fc364501c84e6b24944`.
This closes the raw projected-value identity branch: it makes layer placement
visible, but does not make a matched donor preferable. Native generation stays
blocked.

The next goal is therefore a different loss and boundary, not a larger RWKV
gain. Start from the native-active aligned vector-gate checkpoint, freeze the
projected carrier and RWKV controller, and train only the content gate against
gold scene-boundary CE plus a deterministic wrong-boundary unlikelihood term.
The gate should learn to abstain on recurrent changes that increase false
positives while preserving the existing `+0.00663` correct-versus-projected
native F1 signal. A fresh four-A100 native benchmark is authorized only after
the gate patch passes exact zero/projected identity, fixed-carrier, coverage,
and donor/layer-permutation controls. This tests the native objective directly
without reopening the failed key/value identity geometry.

The precision-unlikelihood run is now locked and complete. It started from the
passed specificity adapter
`natural_memory_native_rwkv_aligned_vector_gate_specificity_train_v1`, kept the
projected carrier and RWKV controller frozen, and updated only the 126 content
gate tensors on 256 untouched fit rows. All 16 updates on four A100s completed:
all rows were finite, all gate tensors were active, the non-gate state hash was
unchanged, and the proximal retention stayed at `0.995`. The signed training
receipt is in
`local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood_train_v1/result.json`.

The separately locked 220-row native benchmark completed under
`natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood_eval_v1`.
Coverage, fixed-carrier, and exact zero/projected identity passed, but the
causal margins did not reach the locked `0.005` threshold:

| condition | micro-F1 | margin versus correct |
| --- | ---: | ---: |
| correct recurrent | 0.195356 | — |
| zero / projected-only | 0.191975 | +0.003381 |
| matched donor | 0.194815 | +0.000541 |
| layer-permuted | 0.194145 | +0.001211 |

The signed status is
`aligned_vector_gate_precision_unlikelihood_native_gain_not_established`;
native generation gain remains blocked. The loss improved the correct state
over the projected-only carrier, but not enough to establish donor or layer
specificity. The prescribed next move was a small frozen-readout calibration,
not more updates to this gate. That screen is now complete. It used the same
locked 220 open rows, exactly four A100s, and 42 per-layer recurrent-state L2
norms. A deterministic ridge scalar separated all layer-permuted controls
(pairwise positive fraction `1.0`) but separated a matched donor on only
`0.586364` of rows, below the locked `0.95` threshold; zero-state separation
was `0.531818`. The signed status is
`state_scalar_screen_failed_donor_separation_blocked` (result SHA-256
`bd066174c8b994c9d3174025216b328c4f9a6cba12084d0cd875394cfcb76ca0`, receipt
`7ae6ba19c915fdf9243d4d595ce9ebbfd57a58ae0fb9e53f97075e9edc930881`). Because
the donor gate failed, no native generation calibration was authorized or
run. The aligned-vector branch is retired; increasing its gain, batch size, or
training duration is not justified.

The next goal is a genuinely learned state-identity mechanism: address-
conditioned low-rank transforms into RWKV `k/v/a/b`, or direct query-to-state
contrastive supervision, with a causal donor-separation gate before any native
generation. The projected carrier and exact control checks remain fixed, and
the HF endpoint remains `https://hf-mirror.com`.

The rank-4 learned projected-value-to-RWKV-read InfoNCE compatibility screen
then passed its four-A100 geometry check on four already-open fit rows: its
mean loss fell from `1.831140` to `0.522844`, and correct-minus-hardest-donor
margin moved from `-0.920785` to `+1.471007`. All 42 zero-initialized `Up`
matrices had finite nonzero first-step gradients; `Down` matrices correctly
had zero cold-start gradients. This is only an internal representation fit,
not a causal result.

The separately preregistered one-update fixed-carrier mechanics preflight also
completed on four A100s. It preserved the target projected carrier for zero,
donor, and layer-permuted recurrent interventions; all control logits and the
compatibility gradients were finite. But the compatibility head is output-
inactive, so its update cannot affect the frozen answer CE controls. Those
mechanics-only CE margins were negative for matched donor (`-0.003903`) and
layer permutation (`-0.000127`), and the pre-update InfoNCE donor margin was
`-0.846677`. It does **not** test or establish causal preference, and neither
eight-update causal training nor generation is authorized.

The strict source-and-donor-component-disjoint cross-fit screen is now
complete. It captured correct, matched-donor, and cyclic layer-permuted
answer-position features for all 220 authorized open rows on exactly four
A100s. The undirected matched-donor graph had 78 connected components; whole
components formed an exact 176-row train / 44-row heldout split, so no source
or its donor crossed partitions. Model outputs and adapter weights remained
frozen, no head or adapter weights were saved, and no generation ran.

Before learning, heldout donor separation was `0.386364` with mean gap
`-0.002108`; layer-permuted separation was `0.886364` with mean gap
`+0.021160`. After 512 CPU AdamW updates to only the 21,504-parameter
two-sided rank-4 compatibility head, the locked heldout result was:

| recurrent control | pairwise positive fraction | mean correct-minus-control score gap |
| --- | ---: | ---: |
| matched donor | **0.954545** | **+0.118183** |
| layer-permuted | **1.000000** | **+0.581383** |

All preregistered gates passed: donor pairwise separation was at least `0.95`,
donor mean gap exceeded `0.05`, layer-permuted separation was at least `0.95`,
and every heldout score was finite. The signed status is
`bilinear_crossfit_passed_causal_training_design_authorized` (result SHA-256
`5e41c4569273fd5841381fcb6c5738b26212dd326b4f7cf56b589528df346ba3`,
receipt
`89392eaeffa50c0bed9109fd8db3d33a5625eb4ff7117f81d710cf9b5be93945`).

This is the first heldout internal donor-identity pass for the learned
compatibility family. It authorizes designing a separately preregistered
causal run that uses the score as a bounded recurrent-correction gate. It does
**not** establish answer-CE preference, native benchmark gain, or authorize
generation.

The signed output-coupled causal endpoint then failed its preregistered donor
gate after all eight updates on exactly four A100s. Training itself was valid:
all 168 gate tensors had finite nonzero global gradients, the projected carrier
was fixed on every row, and zero recurrent logits were byte-identical to the
explicit projected-only bypass. The 44-row endpoint had positive zero and
layer-permuted CE margins (`+0.003578` and `+0.006910`), but matched-donor CE
was slightly **better** than correct (`-0.000080`), donor-positive rows were
`0.386364`, and learned donor identity was `-0.003583` with only `0.386364`
positive rows. The signed status is
`output_identity_gate_heldout_causal_failed_generation_blocked` (result
SHA-256 `b1b71c3a8efb3c9c3b5eaed27bb286e495ec913e26c15c57665ff435b5eff27`).
Per the stopping rule, the bilinear output-gate family is retired without gain,
batch-size, learning-rate, or duration tuning; generation and native benchmark
claims remain blocked.

Evidence: [InfoNCE screen protocol](natural_memory_native_rwkv_query_state_infonce_screen_protocol_v1.json),
[signed screen result](local_artifacts/natural_memory_native_rwkv_query_state_infonce_screen_v4/result.json),
[causal mechanics protocol](natural_memory_native_rwkv_query_state_infonce_causal_preflight_protocol_v1.json),
[signed mechanics result](local_artifacts/natural_memory_native_rwkv_query_state_infonce_causal_preflight_v2/result.json),
[strict cross-fit protocol](natural_memory_native_rwkv_query_state_bilinear_crossfit_protocol_v1.json),
[strict cross-fit runner](run_natural_memory_native_rwkv_query_state_bilinear_crossfit.py),
and [signed strict cross-fit result](local_artifacts/natural_memory_native_rwkv_query_state_bilinear_crossfit_v1/result.json).

Evidence: [output-gate mechanics protocol](natural_memory_native_rwkv_output_identity_gate_mechanics_protocol_v1.json),
[signed mechanics result](local_artifacts/natural_memory_native_rwkv_output_identity_gate_mechanics_v2/result.json),
[output-gate causal protocol](natural_memory_native_rwkv_output_identity_gate_causal_train_protocol_v1.json),
[output-gate causal runner](run_natural_memory_native_rwkv_output_identity_gate_causal_train.py),
and [signed output-gate causal failure](local_artifacts/natural_memory_native_rwkv_output_identity_gate_causal_train_v1/result.json).

The review of arXiv `2608.08888`, *Full Bandwidth Transformer*, supplies two
useful principles: make the carried state the mandatory full-vector value path
and reinject it early enough to receive renewed computation depth. Its
single-adjacent-state experiments report no matched-donor, wrong-state,
zero-state, or layer-permutation controls, so they do not establish our
identity claim. A plain state-times-query CrossGLU repeats the separable family
already tested by DeepEmbed and is retired.

The later identity routes are now closed by signed gates. Bilinear output
gating failed its causal donor endpoint (`-0.000080` donor CE margin;
`0.386364` donor-positive rows). Dense rotary binding failed RWKV
channelwise-update commutation. Full-key diagonal-sign binding passed
cancellation and all finite/zero/carrier controls, but changed only
`0.750`--`0.769` donor decoded rows against the required `0.95`.

The Full-Bandwidth-inspired joint pair-gated CrossGLU then passed its signed
v22 four-A100 mechanics screen, including exact zero/projected-only equality
and fixed-gate/shuffled-gate value controls. Its authorized eight-update causal
endpoint failed donor identity: matched-donor CE was `-0.001360` relative to
correct and only `0.045455` of donor rows were positive. The route is retired,
native generation was never authorized, and no native benchmark claim follows.
The signed causal result is
`local_artifacts/natural_memory_native_rwkv_joint_pair_crossglu_causal_train_v1/result.json`.

The subsequent identity-bound DeepEmbed run kept the causal-passing
address-keyed writer and sparse DeepEmbed adapter frozen and trained only 168
binder tensors (`107,856` parameters). The four-A100 execution completed all
eight updates and 64 rows with finite nonzero gradients, serialized control
graphs, CPU gradient accumulation, and no OOM. Its fresh 16-row endpoint still
rejected state identity:

| held-out metric | result |
| --- | ---: |
| zero-minus-correct CE | +1.301495 |
| donor-minus-correct CE | **-0.007757** |
| layer-permuted-minus-correct CE | +0.285514 |
| correct-minus-donor binder score | **-0.002040** |
| donor-positive CE / binder rows | **43.75% / 43.75%** |

The signed status is
`identity_bound_deepembed_heldout_failed_generation_blocked`, result SHA-256
is `90aa3aba6fd7ec885f16f344801ea8575825dce07103a37dc90b821d6fc9ba46`,
and receipt is
`96f782b35e2d920c72a07cc01d323f3fd4a9c1177d338d93e2b5561518871c96`.
Zero and layer-permutation controls prove that the state path and layer layout
matter, but a matched donor remains better than the correct state. Native
generation was not opened, and this route establishes no native or SOTA gain.

The exact-source v5 shadow cross-fit screen has now passed. It strict-loaded
the causal-passing v5 adapter under its signed `cd7deb91` Delta-Mem source,
captured all 220 open development rows on four A100s without changing model
output, and trained only a disposable 21,504-parameter compatibility head on a
source-and-donor-component-disjoint `176/44` split. The predeclared held-out
gates all pass:

| held-out shadow identity metric | result | gate |
| --- | ---: | ---: |
| matched-donor pairwise-positive fraction | **0.954545** | >= 0.95 |
| matched-donor mean score gap | **0.103092** | >= 0.05 |
| layer-permuted pairwise-positive fraction | **1.000000** | >= 0.95 |

The result SHA-256 is
`c3607fbc6f42b6a2ebcdfab7d5cdf399e5b8e4c8ab52a1c707e8f1d19d44108d`
and its receipt is
`4ba137387216a8f2bc2c5562a764b4f340afa795cc4dbc88d4d2cf0ea470443c`.
This proves that untouched v5 RWKV shadows contain cross-source identity that
generalizes across the locked open split. It does not prove that the model
causally uses that identity, and it does not authorize training, generation,
the native benchmark, or a SOTA claim.

The authorized next goal is a separately signed exact-v5 recurrent mechanics
diagnostic. Execute the learned writer once, keep the live RWKV read as the
material value, use the detached shadow only to certify identity, and run
read-only feedback passes through the unchanged sparse DeepEmbed feature path.
Compare correct, matched-donor, donor-state-plus-donor-shadow, zero, layer-
permuted, row-shuffled, random, and disabled-shadow controls while measuring
`delta_k` through eight passes. This is Jacobi-inspired rather than true Jacobi
training because the mechanics screen detaches the replay state. Only a donor-
specific, contracting mechanics pass may authorize a separately locked two-
pass live-gradient causal run. The full paper-to-RWKV review lists that route
and the other bounded alternatives.
See [FULL_BANDWIDTH_RWKV_REVIEW.md](FULL_BANDWIDTH_RWKV_REVIEW.md) for
equations, reported results, caveats, factorial controls, and stopping gates.

Evidence: [initial DeepEmbed protocol](natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_protocol_v1.json),
[initial signed result](local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v1/result.json),
[BF16 protocol](natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_protocol_v2.json),
[BF16 signed result](local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v2/result.json),
[sparse screen protocol](natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen_protocol_v1.json),
[sparse screen result](local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen_v1/result.json),
[sparse causal protocol](natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train_protocol_v1.json),
[sparse causal result](local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train_v1/result.json),
[address-keyed v5 training protocol](natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v5.json),
[address-keyed v5 endpoint](local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5_r1/result.json),
[exact-v5 shadow cross-fit protocol](natural_memory_native_rwkv_v5_shadow_crossfit_protocol_v1.json),
[exact-v5 shadow cross-fit runner](run_natural_memory_native_rwkv_v5_shadow_crossfit.py),
[signed exact-v5 shadow identity pass](local_artifacts/natural_memory_native_rwkv_v5_shadow_crossfit_v1/result.json),
[locked address-keyed generation protocol](natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_generation_protocol_v1.json),
[signed address-keyed native failure](local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval_v1/result.json),
[learned-write causal protocol](natural_memory_native_rwkv_address_keyed_learned_write_causal_train_protocol_v1.json),
[learned-write causal runner](run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train.py),
[signed learned-write causal failure](local_artifacts/natural_memory_native_rwkv_address_keyed_learned_write_causal_train_v3/result.json),
[query-state identity protocol](natural_memory_native_rwkv_query_state_identity_causal_train_protocol_v1.json),
[query-state identity runner](run_natural_memory_native_rwkv_query_state_identity_causal_train.py),
and [signed query-state identity failure](local_artifacts/natural_memory_native_rwkv_query_state_identity_causal_train_v3/result.json).

Full-key diagonal-sign mechanics evidence:
[protocol](natural_memory_native_rwkv_diagonal_sign_binding_fullkey_mechanics_protocol_v1.json),
[runner](run_natural_memory_native_rwkv_diagonal_sign_binding_mechanics.py),
and [signed donor-specificity failure](local_artifacts/natural_memory_native_rwkv_diagonal_sign_binding_fullkey_mechanics_v2/result.json).

The state-scalar screen is documented by
[its protocol](natural_memory_native_rwkv_aligned_vector_gate_state_scalar_screen_protocol_v1.json),
[its probe](probe_native_state_scalar.py),
[its analyzer](analyze_native_state_scalar.py), and
[its signed result](local_artifacts/natural_memory_native_state_scalar_probe_v1/result.json).

Use the PyTorch/HF delta-Mem path for these diagnostics. The GGUF sidecar path
is useful for serving, but it does not expose the hidden states, gradients, and
RWKV-MS routing tensors needed here.

## Experiment Surface

The scripts inspect these existing delta-Mem hooks:

- `last_read_routes`: RWKV-MS read distribution over state slots.
- `last_write_routes`: slot written by each token under fixed-chunk routing.
- `last_delta_o_ratio`: size of memory output relative to base attention output.
- `delta_state`: online RWKV-MS state tensor.
- `rwkv_ms_chunk_size` and `rwkv_ms_num_states`: slot layout.

## Quick Start

Generate a small NIAH-style classification set:

```bash
python experiments/rethinking_rwkv_ms_gemma/make_niah.py \
  --output .openresearch/niah_rwkv_ms/niah.jsonl \
  --num-samples 64 \
  --num-records 96 \
  --num-candidates 8 \
  --seed 7
```

Run answer-token ablations:

```bash
python experiments/rethinking_rwkv_ms_gemma/run_ablation_eval.py \
  --delta-mem-root ../delta-Mem \
  --base-model /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
  --memory-dir /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100 \
  --dataset .openresearch/niah_rwkv_ms/niah.jsonl \
  --conditions base,normal,no_write,no_delta,reset_1024 \
  --output-dir .openresearch/niah_rwkv_ms/ablation
```

Trace whether query tokens read from the slot that contains the needle:

```bash
python experiments/rethinking_rwkv_ms_gemma/trace_rwkv_ms_slots.py \
  --delta-mem-root ../delta-Mem \
  --base-model /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
  --memory-dir /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100 \
  --dataset .openresearch/niah_rwkv_ms/niah.jsonl \
  --output-dir .openresearch/niah_rwkv_ms/trace
```

Extract final-token hidden states for layer-wise probing:

```bash
python experiments/rethinking_rwkv_ms_gemma/extract_layer_probes.py \
  --delta-mem-root ../delta-Mem \
  --base-model /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
  --memory-dir /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100 \
  --dataset .openresearch/niah_rwkv_ms/niah.jsonl \
  --condition normal \
  --output-npz .openresearch/niah_rwkv_ms/probes_normal.npz

python experiments/rethinking_rwkv_ms_gemma/train_probe.py \
  --input-npz .openresearch/niah_rwkv_ms/probes_normal.npz \
  --output-json .openresearch/niah_rwkv_ms/probes_normal.json
```

Compute token-distance gradient influence:

```bash
python experiments/rethinking_rwkv_ms_gemma/gradient_influence.py \
  --delta-mem-root ../delta-Mem \
  --base-model /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
  --memory-dir /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100 \
  --dataset .openresearch/niah_rwkv_ms/niah.jsonl \
  --condition normal \
  --output-npz .openresearch/niah_rwkv_ms/gradient_normal.npz
```

## Conditions

- `base`: frozen Gemma without RWKV-MS.
- `normal`: Gemma + RWKV-MS checkpoint.
- `no_write`: RWKV-MS state reads from zero state because writes are disabled.
- `no_delta`: RWKV-MS state still updates, but q/k/v/o delta heads are disabled.
- `reset_N`: process with KV cache while resetting RWKV-MS every `N` prompt
  tokens, for example `reset_1024`.

The first interpretation pass should compare `base`, `normal`, `no_write`, and
`no_delta`. If `normal` beats `no_write` and `no_delta`, memory readout is doing
real work. If `normal` is close to `no_delta`, the memory path is mostly a
training-time or regularization effect, not an inference-time retrieval carrier.

## Main Metrics

Answer ablation:

- `accuracy`: predicted candidate letter equals the gold letter.
- `gold_margin`: gold candidate logit minus best non-gold candidate logit.

Slot trace:

- `needle_slot_mass`: query-token read probability assigned to the slot that
  contains the needle marker.
- `read_entropy`: entropy of the query-token read distribution over RWKV-MS
  slots.
- `delta_o_ratio`: memory-output magnitude relative to base attention output.
- `state_norm`: norm of the online RWKV-MS state.

Probe:

- Per-layer classifier accuracy from final query-token hidden states.

Gradient:

- Mean input-embedding gradient norm by distance from the query token.

## Research Readout

Evidence for RWKV-MS as a true retrieval carrier:

- high `needle_slot_mass` at query time,
- lower `read_entropy` after training,
- strong drop under `no_write` or `reset_N`,
- layer probes improve immediately after wrapped layers,
- gradient influence reaches the needle through the memory path.

Evidence for "State-Slot Laziness" or over-perturbation:

- `normal` improves task format but not NIAH retrieval,
- `delta_o_ratio` grows while `needle_slot_mass` stays low,
- all-layer/adapted checkpoints show worse probe separability or higher read
  entropy,
- `no_delta` behaves close to `normal`.
