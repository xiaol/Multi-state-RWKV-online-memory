# Delta-Mem RWKV-MS Online Memory

This repository contains a self-contained delta-Mem integration for the RWKV
multi-state online-memory module used by the mechanism comparison. The patched
Python runtime is bundled at top-level `deltamem/`; normal Qwen/Gemma HF
training and inference do not require a separate delta-Mem checkout.

The implementation uses the verified HRM-Text RWKV-7 read-before-write memory
core as the source reference. It adds a second online memory backend and
supports both the Qwen3.6 HF model and the current practical Gemma target,
`google/gemma-4-E4B-it`.

| Backend | Flag | State | Readout contract |
| --- | --- | --- | --- |
| Delta rule | `--memory-backend delta_rule` | one associative matrix per state head | existing q/k/v/o delta heads |
| RWKV-MS | `--memory-backend rwkv_ms` | `rwkv_ms_num_states` RWKV-7 matrices per state head | same q/k/v/o delta heads |

Supported attention backbones in the bundled runtime:

| Backbone | Status |
| --- | --- |
| Qwen3 | supported |
| Qwen3.5/Qwen3.6 (`qwen3_5` in Transformers) | supported, including gated q/output projection and partial RoPE |
| SmolLM3 | supported |
| Gemma4 text attention | supported for non-KV-shared layers |

Gemma4 E4B has KV-shared tail layers that do not own k/v projections. The
runtime skips those layers and wraps the non-shared sliding/full attention
layers.

## Files

| File | Purpose |
| --- | --- |
| `../../deltamem/` | Bundled HF online-memory runtime used for normal training and inference. |
| `delta_mem_rwkv_ms.patch` | Optional export of the integration for the pinned upstream delta-Mem revision. |
| `hrm_rwkv7.py` | Minimal HRM-Text-derived RWKV-7 projection/readout core mirrored by the bundled runtime and patch. |
| `inference.py` | HF online-memory inference entry point for a compatible RWKV-MS checkpoint. |
| `train_smoke.py` | Manual two-step adapter train with gradient, tensor-delta, and reload checks. |
| `run_rwkv_ms_delta_rule_comparison.sh` | Matched delta-rule vs RWKV-MS training launcher. |
| `check_delta_mem_patch.sh` | Validates the bundled runtime; optionally verifies the patch against an upstream copy. |

## Bundled HF Runtime

The HF checkpoint is only the learned online-memory weights and
`delta_mem_config.json`.
Inference also needs runtime code, which this repository now provides in
`deltamem/`:

- the attention wrapper that attaches online-memory read/write modules to Gemma4
  text attention layers;
- `HFDeltaMemConfig`, checkpoint loading, and state reset/save/load helpers;
- the chat/session runtime that keeps `past_key_values` and RWKV-MS online
  memory state synchronized across turns;
- tokenizer/chat-template handling and message/span IDs used by write routing.

The normal layout is therefore self-contained:

```text
Multi-state-RWKV-online-memory/
├── deltamem/                         # bundled patched Python runtime
├── integrations/delta_mem_rwkv_ms/  # launchers, docs, GGUF tools, optional patch export
└── HF memory checkpoint/             # downloaded weights + config, local or Hub cache
```

Install from the repository root and run the commands below from that root:

```bash
git clone https://github.com/xiaol/Multi-state-RWKV-online-memory.git
cd Multi-state-RWKV-online-memory
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Optional Upstream Patch Export

`delta_mem_rwkv_ms.patch` is retained for review, provenance, and exporting the
same integration to upstream delta-Mem. It is not required when using this
repository's bundled `deltamem/` package. The export is pinned to upstream
delta-Mem revision `5cd5d9153c7f408764728d953565201e198c39e2`.
See [`BUNDLED_RUNTIME.md`](BUNDLED_RUNTIME.md) for the exact source and local
integration provenance.

From a clean or reviewable `delta-Mem` checkout:

```bash
git apply \
  /path/to/Multi-state-RWKV-online-memory/integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch
python -m py_compile \
  deltamem/core/hrm_rwkv7.py \
  deltamem/core/delta_impl.py \
  deltamem/core/delta.py \
  deltamem/train/delta_sft_experimental.py
bash -n scripts/run_rwkv_ms_delta_rule_comparison.sh
```

With no arguments, the checker validates the bundled runtime and runs the Qwen,
Gemma, RWKV-MS, and artifact-independent GGUF streaming-math regressions:

```bash
PYTHON_BIN=/path/to/python \
  integrations/delta_mem_rwkv_ms/check_delta_mem_patch.sh
```

Passing a clean upstream checkout additionally applies the optional patch and
proves that its patched `deltamem/` tree is byte-identical to the bundled one:

```bash
integrations/delta_mem_rwkv_ms/check_delta_mem_patch.sh /path/to/delta-Mem
```

## Qwen3.6 HF Path

Qwen3.6 is implemented by Transformers as `qwen3_5` and requires
`transformers>=5.12.0`. The 27B checkpoint has 64 physical layers. Only the 16
layers at `3,7,11,...,63` use full attention; the other 48 are Gated DeltaNet
layers and do not contain a wrappable attention module. `target_layers` always
refers to these physical model indices.

Use layer `3` for a small attachment/training smoke:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  integrations/delta_mem_rwkv_ms/train_smoke.py \
  --model-path /home/aiuser/X/models/Qwen3.6-27B \
  --target-layers 3 \
  --attn-implementation auto \
  --output-dir .openresearch/artifacts/qwen36_rwkv_ms_train_smoke
```

For a matched backend comparison over the first six eligible layers:

```bash
BASE_MODEL_PATH=/home/aiuser/X/models/Qwen3.6-27B \
TARGET_LAYERS=3,7,11,15,19,23 \
TRAIN_FILE=/path/to/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
OUTPUT_ROOT=/path/to/qwen36_rwkv_ms_delta_rule_comparison \
NPROC_PER_NODE=8 \
bash integrations/delta_mem_rwkv_ms/run_rwkv_ms_delta_rule_comparison.sh
```

The wrapper handles Qwen3.6's gated query projection and output gate without
injecting deltas into the gate branch, and applies the model's partial RoPE
helper. Stateful RWKV time mixing is carried across cached decode calls and is
included in online-state snapshots. The predecessor is detached between
separate write calls, giving value-continuous state with truncated BPTT at call
boundaries.

## Gemma4 HF Online Memory

Recommended entry point:

```bash
cd Multi-state-RWKV-online-memory
.venv/bin/python integrations/delta_mem_rwkv_ms/inference.py \
  --memory-repo xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --base-model google/gemma-4-E4B-it \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --prompt "Help me troubleshoot a mobile data issue where the customer has no usable data."
```

The same script also accepts `--memory-dir /path/to/local/model-repo` if the HF
model repo has already been cloned. For backward compatibility, `--adapter-repo`
and `--adapter-dir` are accepted as delta-Mem API aliases. It uses
`DeltaMemChatSession.generate_reply(...)` from the bundled runtime.

Then train a matched pair:

```bash
BASE_MODEL_PATH=google/gemma-4-E4B-it \
TARGET_LAYERS=0,1,2,3,4,5 \
TRAIN_FILE=/path/to/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
OUTPUT_ROOT=/path/to/rwkv_ms_delta_rule_comparison \
NPROC_PER_NODE=8 \
bash integrations/delta_mem_rwkv_ms/run_rwkv_ms_delta_rule_comparison.sh
```

Keep base model, data, rank, alpha, delta heads, target layers, write policy,
training budget, and eval scripts identical across both runs. The comparison is
valid only if the controlled variable is the online state backend.

For a small Gemma end-to-end training check, use one visible GPU and the bundled
two-example fixture. Override the Qwen-oriented smoke default with a Gemma
physical layer:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  integrations/delta_mem_rwkv_ms/train_smoke.py \
  --model-path /path/to/gemma-4-E4B-it \
  --target-layers 0 \
  --output-dir .openresearch/artifacts/rwkv_ms_train_smoke
```

The smoke trainer keeps the frozen base in BF16, keeps the optimizer-owned
adapter parameters in FP32, and performs manual `loss.backward()` / `AdamW.step()`
updates. It fails unless RWKV-MS state is nonzero, gradients are nonzero,
adapter tensors change, and every saved tensor plus the adapter config reloads
exactly. To protect checkpoints, the output directory must be absent or empty;
use `--overwrite-output` only when replacing its contents is intentional.

## Verification Already Run

- HRM-Text `.venv` imports and runs the RWKV memory stack.
- HRM-Text RWKV memory manual tests pass.
- HRM-Text tiny CPU and CUDA RWKV memory benchmark paths pass.
- The patch checker applies to the pinned delta-Mem base, verifies bundled-file
  parity, and runs focused regressions.
- Gemma4 text attention zero-init RWKV-MS smoke passes against base attention.
- Tiny Gemma4 text model attach smoke wraps non-shared layers and skips
  KV-shared layers.
- RWKV-MS forward smoke passes with Qwen3 eager attention:
  zero-init output matches base attention, state shape is
  `[batch, num_state_heads, rwkv_ms_num_states, rank, rank]`, fixed chunk routes
  match expectation, and read-only mode preserves state, positions, and the
  streaming time-mix predecessor.
- Qwen3.6 actual-config meta attach wraps exactly the 16 physical full-attention
  layers at `3,7,...,63`.
- Qwen3.6 BF16 on an A800 passes a prompt forward plus cached one-token decode
  with SDPA, adapter save/load, and online-state snapshot restore.
- All 29 Qwen3.6 checkpoint files match mirror revision `6a9e13bd...`; all 15
  safetensor shard hashes and headers validate.
- Changed-area Qwen/Gemma/RWKV delta-Mem regressions: 78 passed.
- The GGUF math helper preserves legacy v1 fixtures and proves that a nonzero
  streaming predecessor produces identical state, position, predecessor, and
  read results in one full scan or two successive chunks.
- Gemma4 E4B A800 training smoke: two optimizer steps, nonzero recurrent state
  and gradients, 21 changed adapter tensors, exact checkpoint reload.
