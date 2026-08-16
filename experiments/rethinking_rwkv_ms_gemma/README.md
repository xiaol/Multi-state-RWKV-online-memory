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
3. The projected route addresses the matching RWKV state slot. The RWKV read is
   normalized and used as a bounded elementwise FiLM controller:

   ```text
   output = projected_read * (1 + 0.125 * tanh(rwkv_read / rms(rwkv_read)))
   ```

4. The fused read passes through the learned delta output path and a
   content-gated residual into frozen Gemma.

The projected key/value sidecar is therefore still the material carrier. RWKV
has a causal controller role only when correct recurrent state beats zero,
matched-donor, and layer-permuted state while the projected carrier is held
fixed. The repository does not currently establish that full causal claim or a
native benchmark gain attributable to RWKV.

## Current Evidence

The latest locked open-fit experiment used four A100 GPUs, 16 optimizer
updates, 128 accepted training rows, and a fresh source/donor-disjoint 32-row
teacher-forced endpoint. All training and integrity gates passed, with no row
rejections and no protected split access.

| recurrent condition | mean CE | margin versus correct |
| --- | ---: | ---: |
| correct | 3.018380 | - |
| zero / projected-only | 3.929799 | +0.911419 |
| matched donor | 3.015786 | -0.002594 |
| layer-permuted | 3.036787 | +0.018407 |

Lower CE is better, so the zero and layer-permuted controls show that the RWKV
path is active and layer/order-sensitive. The matched donor is slightly better
than the correct state, so example-specific recurrent retrieval is not
established. Native generation remains blocked. The signed result is
`local_artifacts/natural_memory_native_rwkv_addressed_vector_gate_causal_train_v1/result.json`
(file SHA-256
`e9a115cd7864e0c31738478c0393aed21c6db40e3ab1adccc50ba76cb8a898e4`,
receipt
`e13f1a45139e28178b0e7ca28b3b647d42a427a23180bc2b7f695fda9dc109c3`).

The next architecture must add state/content identity rather than merely
increasing recurrent gain. A larger gain would amplify the same donor ambiguity.

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
