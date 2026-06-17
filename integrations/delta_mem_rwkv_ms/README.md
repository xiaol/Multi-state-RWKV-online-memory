# Delta-Mem RWKV-MS Adapter

This folder contains the practical delta-Mem integration for the RWKV multi-state
memory adapter used by the mechanism comparison in this repo.

The implementation was built against the local `delta-Mem` repo and uses the
verified HRM-Text RWKV-7 read-before-write memory core as the source reference.
It adds a second online memory backend:

| Backend | Flag | State | Readout contract |
| --- | --- | --- | --- |
| Delta rule | `--memory-backend delta_rule` | one associative matrix per state head | existing q/k/v/o delta heads |
| RWKV-MS | `--memory-backend rwkv_ms` | `rwkv_ms_num_states` RWKV-7 matrices per state head | same q/k/v/o delta heads |

## Files

| File | Purpose |
| --- | --- |
| `delta_mem_rwkv_ms.patch` | Patch for a delta-Mem checkout. |
| `hrm_rwkv7.py` | Minimal HRM-Text-derived RWKV-7 projection/readout core included in the patch. |
| `run_rwkv_ms_delta_rule_comparison.sh` | Matched delta-rule vs RWKV-MS training launcher. |
| `check_delta_mem_patch.sh` | Applies the patch to a temporary copy and runs syntax checks. |

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

Then train a matched pair:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
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
- RWKV-MS forward smoke passes with Qwen3 eager attention:
  zero-init output matches base attention, state shape is
  `[batch, num_state_heads, rwkv_ms_num_states, rank, rank]`, fixed chunk routes
  match expectation, and read-only mode preserves state and positions.

`pytest` was not available in the existing local environments, so the full
delta-Mem pytest suite was not run from this bundle.
