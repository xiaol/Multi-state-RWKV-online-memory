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

### Outer-FFN branch

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

The next hybrid should use a bounded mixture-of-experts recurrent controller:
small addressed and global-state readout experts compete under a normalized
query-conditioned gate, with donor/layer contrast and an explicit projected-only
abstention arm. This targets example-specificity without opening native
generation for an unproven recurrent readout.

The next direction is write-side identity: condition the RWKV write/value
update on the projected slot address, then train with a donor-contrast loss.
This targets the repeated donor-neutral result directly. A smaller or more
sparsely applied outer FFN can remain a diagnostic control, but it is not an
authorized training or generation path after the bound failure.

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
