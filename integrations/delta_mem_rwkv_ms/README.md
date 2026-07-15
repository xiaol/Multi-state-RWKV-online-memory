# Delta-Mem RWKV-MS Online Memory

This folder contains the practical delta-Mem integration for the RWKV multi-state
online-memory module used by the mechanism comparison in this repo.

The implementation was built against the local `delta-Mem` repo and uses the
verified HRM-Text RWKV-7 read-before-write memory core as the source reference.
It adds a second online memory backend and supports both the Qwen3.6 HF model
and the current practical Gemma target, `google/gemma-4-E4B-it`.

| Backend | Flag | State | Readout contract |
| --- | --- | --- | --- |
| Delta rule | `--memory-backend delta_rule` | one associative matrix per state head | existing q/k/v/o delta heads |
| RWKV-MS | `--memory-backend rwkv_ms` | `rwkv_ms_num_states` RWKV-7 matrices per state head | same q/k/v/o delta heads |

Supported attention backbones in this patch:

| Backbone | Status |
| --- | --- |
| Qwen3 | supported |
| Qwen3.5/Qwen3.6 (`qwen3_5` in Transformers) | supported, including gated q/output projection and partial RoPE |
| SmolLM3 | supported |
| Gemma4 text attention | supported for non-KV-shared layers |

Gemma4 E4B has KV-shared tail layers that do not own k/v projections. The patch
skips those layers and wraps the non-shared sliding/full attention layers.

## Files

| File | Purpose |
| --- | --- |
| `delta_mem_rwkv_ms.patch` | Patch for a delta-Mem checkout. |
| `hrm_rwkv7.py` | Minimal HRM-Text-derived RWKV-7 projection/readout core included in the patch. |
| `inference.py` | HF online-memory inference entry point for a compatible RWKV-MS checkpoint. |
| `train_smoke.py` | Manual two-step adapter train with gradient, tensor-delta, and reload checks. |
| `run_rwkv_ms_delta_rule_comparison.sh` | Matched delta-rule vs RWKV-MS training launcher. |
| `check_delta_mem_patch.sh` | Applies the patch to a temporary copy and runs focused regressions. |

## What delta-Mem Provides

The HF checkpoint is only the learned online-memory weights and
`delta_mem_config.json`.
Inference still needs the delta-Mem runtime because delta-Mem provides:

- the attention wrapper that attaches online-memory read/write modules to Gemma4
  text attention layers;
- `HFDeltaMemConfig`, checkpoint loading, and state reset/save/load helpers;
- the chat/session runtime that keeps `past_key_values` and RWKV-MS online
  memory state synchronized across turns;
- tokenizer/chat-template handling and message/span IDs used by write routing.

This repository owns the RWKV-MS mechanism, patch, training notes, benchmark
comparison, and inference entry point. It does not vendor the full delta-Mem
runtime as a forked package, because that would duplicate the upstream runtime
surface and make memory-checkpoint compatibility harder to track. The intended
layout is:

```text
Multi-state-RWKV-online-memory/      # this repo: patch, docs, inference entry point
delta-Mem/                           # patched runtime dependency
HF memory checkpoint repo/           # weights + config only
```

## Apply To Delta-Mem

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

For a fresh local inference setup:

```bash
git clone https://github.com/xiaol/Multi-state-RWKV-online-memory.git
git clone https://github.com/declare-lab/delta-Mem.git

cd delta-Mem
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -U "huggingface_hub>=1.0.0"

git apply \
  ../Multi-state-RWKV-online-memory/integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch
```

If the patch reports that hunks are already applied, use a clean delta-Mem
checkout or a delta-Mem revision that already contains the RWKV-MS integration.

The patch targets delta-Mem revision `5cd5d9153c7f408764728d953565201e198c39e2`.
The bundled checker verifies that revision, applies the patch in a temporary
clone, compares the bundled core and launcher byte-for-byte, and runs the Qwen,
Gemma, RWKV-MS, and artifact-independent GGUF streaming-math regressions:

```bash
PYTHON_BIN=/path/to/python \
  integrations/delta_mem_rwkv_ms/check_delta_mem_patch.sh ../delta-Mem
```

## Qwen3.6 HF Path

Qwen3.6 is implemented by Transformers as `qwen3_5` and requires
`transformers>=5.12.0`. The 27B checkpoint has 64 physical layers. Only the 16
layers at `3,7,11,...,63` use full attention; the other 48 are Gated DeltaNet
layers and do not contain a wrappable attention module. `target_layers` always
refers to these physical model indices.

Use layer `3` for a small attachment/training smoke:

```bash
CUDA_VISIBLE_DEVICES=0 ../delta-Mem/.venv/bin/python \
  integrations/delta_mem_rwkv_ms/train_smoke.py \
  --delta-mem-root ../delta-Mem \
  --model-path /home/aiuser/X/models/Qwen3.6-27B \
  --target-layers 3 \
  --attn-implementation auto \
  --output-dir .openresearch/artifacts/qwen36_rwkv_ms_train_smoke
```

For a matched backend comparison over the first six eligible layers:

```bash
(
  cd ../delta-Mem
  BASE_MODEL_PATH=/home/aiuser/X/models/Qwen3.6-27B \
  TARGET_LAYERS=3,7,11,15,19,23 \
  TRAIN_FILE=/path/to/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
  OUTPUT_ROOT=/path/to/qwen36_rwkv_ms_delta_rule_comparison \
  NPROC_PER_NODE=8 \
  bash scripts/run_rwkv_ms_delta_rule_comparison.sh
)
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
../delta-Mem/.venv/bin/python integrations/delta_mem_rwkv_ms/inference.py \
  --delta-mem-root ../delta-Mem \
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
`DeltaMemChatSession.generate_reply(...)` from the patched delta-Mem runtime.

Then train a matched pair:

```bash
(
  cd ../delta-Mem
  BASE_MODEL_PATH=google/gemma-4-E4B-it \
  TARGET_LAYERS=0,1,2,3,4,5 \
  TRAIN_FILE=/path/to/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
  OUTPUT_ROOT=/path/to/rwkv_ms_delta_rule_comparison \
  NPROC_PER_NODE=8 \
  bash scripts/run_rwkv_ms_delta_rule_comparison.sh
)
```

Keep base model, data, rank, alpha, delta heads, target layers, write policy,
training budget, and eval scripts identical across both runs. The comparison is
valid only if the controlled variable is the online state backend.

For a small Gemma end-to-end training check, use one visible GPU and the bundled
two-example fixture. Override the Qwen-oriented smoke default with a Gemma
physical layer:

```bash
CUDA_VISIBLE_DEVICES=0 ../delta-Mem/.venv/bin/python \
  integrations/delta_mem_rwkv_ms/train_smoke.py \
  --delta-mem-root ../delta-Mem \
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
- Changed-area Qwen/Gemma/RWKV delta-Mem regressions: 77 passed.
- The GGUF math helper preserves legacy v1 fixtures and proves that a nonzero
  streaming predecessor produces identical state, position, predecessor, and
  read results in one full scan or two successive chunks.
- Gemma4 E4B A800 training smoke: two optimizer steps, nonzero recurrent state
  and gradients, 21 changed adapter tensors, exact checkpoint reload.
