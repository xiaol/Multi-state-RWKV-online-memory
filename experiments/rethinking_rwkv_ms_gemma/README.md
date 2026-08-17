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
[locked address-keyed generation protocol](natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_generation_protocol_v1.json),
and [signed address-keyed native failure](local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval_v1/result.json).

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
