# Delta-Mem RWKV-MS Online Memory

This folder contains the practical delta-Mem integration for the RWKV multi-state
online-memory module used by the mechanism comparison in this repo.

The implementation was built against the local `delta-Mem` repo and uses the
verified HRM-Text RWKV-7 read-before-write memory core as the source reference.
It adds a second online memory backend and extends the adapter wrapper to the
current practical Gemma target, `google/gemma-4-E4B-it`.

| Backend | Flag | State | Readout contract |
| --- | --- | --- | --- |
| Delta rule | `--memory-backend delta_rule` | one associative matrix per state head | existing q/k/v/o delta heads |
| RWKV-MS | `--memory-backend rwkv_ms` | `rwkv_ms_num_states` RWKV-7 matrices per state head | same q/k/v/o delta heads |

Supported attention backbones in this patch:

| Backbone | Status |
| --- | --- |
| Qwen3 | supported |
| SmolLM3 | supported |
| Gemma4 text attention | supported for non-KV-shared layers |

Gemma4 E4B has KV-shared tail layers that do not own k/v projections. The patch
skips those layers and wraps the non-shared sliding/full attention layers.

## Files

| File | Purpose |
| --- | --- |
| `delta_mem_rwkv_ms.patch` | Patch for a delta-Mem checkout. |
| `hrm_rwkv7.py` | Minimal HRM-Text-derived RWKV-7 projection/readout core included in the patch. |
| `inference.py` | HF online-memory inference entry point for the Gemma4 E4B RWKV-MS checkpoint. |
| `run_rwkv_ms_delta_rule_comparison.sh` | Matched delta-rule vs RWKV-MS training launcher. |
| `check_delta_mem_patch.sh` | Applies the patch to a temporary copy and runs syntax checks. |

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
git apply --unidiff-zero --whitespace=nowarn \
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

git apply --unidiff-zero --whitespace=nowarn \
  ../Multi-state-RWKV-online-memory/integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch
```

If the patch reports that hunks are already applied, use a clean delta-Mem
checkout or a delta-Mem revision that already contains the RWKV-MS integration.

## Run The HF Online Memory

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
BASE_MODEL_PATH=google/gemma-4-E4B-it \
TRAIN_FILE=/path/to/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
OUTPUT_ROOT=/path/to/rwkv_ms_delta_rule_comparison \
NPROC_PER_NODE=8 \
bash scripts/run_rwkv_ms_delta_rule_comparison.sh
```

Keep base model, data, rank, alpha, delta heads, target layers, write policy,
training budget, and eval scripts identical across both runs. The comparison is
valid only if the controlled variable is the online state backend.

## Verification Already Run

- HRM-Text `.venv` imports and runs the RWKV memory stack.
- HRM-Text RWKV memory manual tests pass.
- HRM-Text tiny CPU and CUDA RWKV memory benchmark paths pass.
- delta-Mem patched files pass `py_compile`.
- Gemma4 text attention zero-init RWKV-MS smoke passes against base attention.
- Tiny Gemma4 text model attach smoke wraps non-shared layers and skips
  KV-shared layers.
- RWKV-MS forward smoke passes with Qwen3 eager attention:
  zero-init output matches base attention, state shape is
  `[batch, num_state_heads, rwkv_ms_num_states, rank, rank]`, fixed chunk routes
  match expectation, and read-only mode preserves state and positions.

`pytest` was not available in the existing local environments, so the full
delta-Mem pytest suite was not run from this bundle.
