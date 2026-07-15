# GGUF / GGML RWKV-MS Port Plan

This is the implementation plan for turning the current PyTorch delta-Mem
RWKV-MS checkpoint into a real llama.cpp/GGML runtime. The base Gemma4 E4B GGUF
is already useful for normal inference tests, but it is not the external-memory
runtime.

## Current Working Baselines

PyTorch online-memory reference:

```bash
python integrations/delta_mem_rwkv_ms/inference.py \
  --delta-mem-root /path/to/patched/delta-Mem \
  --memory-repo xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --base-model google/gemma-4-E4B-it \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa
```

Base GGUF reference:

```bash
LLAMA_SERVER_BIN=/path/to/llama-server \
bash tools/llama_server_gemma4.sh
```

Memory GGUF sidecar:

```text
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
```

The base GGUF path should be used for llama.cpp health checks, tokenizer/chat
template checks, and baseline prompt behavior. The PyTorch path remains the
authoritative RWKV-MS reference, while the local llama.cpp branch now has the
first constrained Gemma4 sidecar runtime for serial single-token scans.

Patched llama.cpp fork:

```text
https://github.com/xiaol/llama.cpp-online-memory
branch: main
commit: 85da0c63b Add Gemma4 RWKV-MS GGUF sidecar runtime
base upstream: ggml-org/llama.cpp 1ec44d1
```

## Runtime Requirements

1. Load memory metadata:
   - memory backend `rwkv_ms`;
   - target layers, currently Gemma4 text layers `0-5`;
   - target projections, current best checkpoint uses `q,o`;
   - rank, alpha, chunk length, state count, and read/write policy.

2. Load trained tensors:
   - q/o low-rank adapter weights;
   - RWKV-7 projection/readout parameters from `HRMRWKV7LowRankCore`;
   - any learned state-routing parameters in the delta-Mem checkpoint.

3. Add Gemma4 attention hooks:
   - wrap only non-KV-shared Gemma4 text attention layers;
   - skip KV-shared tail layers;
   - read selected hidden states before attention projection injection;
   - inject learned deltas into `q` and `o` at the same point as PyTorch.

4. Implement online state:
   - recurrent RWKV-MS state buffers with shape equivalent to the PyTorch state;
   - token-by-token read-before-write update;
   - state reset/save/load;
   - slot/KV-cache synchronization for multi-turn chat and batched serving.

5. Validate against PyTorch:
   - isolated RWKV-MS math fixture before full Gemma integration;
   - projection, read-before-write update, readout, and `q,o` delta parity;
   - identical prompt and chat template;
   - fixed seed / greedy decoding;
   - captured token IDs;
   - per-layer delta summaries;
   - state statistics before and after each turn;
   - generated text and tool-call format.

## Suggested Work Order

1. Add a memory checkpoint inspector that emits a JSON manifest of config values
   and tensor names/shapes/dtypes. Done.
2. Export the RWKV-MS adapter tensors to a GGUF sidecar. Done.
3. Round-trip the GGUF sidecar back into a delta-Mem checkpoint directory. Done.
4. Generate golden PyTorch traces from `DeltaMemChatSession.generate_reply`. Done.
5. Add GGUF-vs-reference trace comparison harness. Done.
6. Draft a session-state serialization format.
7. Add an isolated PyTorch RWKV-MS math fixture from the sidecar-rebuilt
   checkpoint. Done.
8. Port sidecar projection, `HRMRWKV7LowRankCore` feature projection,
   sidecar-driven scan inputs, graph readout, and active q/o delta math to an
   isolated GGML/C++ fixture. Done.
9. Load sidecar tensors into `llama_model`. Done.
10. Add Gemma4 layer hooks in a llama.cpp branch and keep sidecar state keyed by
    logical layer, not `has_kv(il)`. Done for the current target layers `0-5`.
11. Add the first stateful llama.cpp RWKV-MS runtime. Done for one sequence,
    recommended serial server/UI scans, `num_state_heads=1`, and `q,o` deltas;
    an experimental graph-unrolled multi-token prompt path exists for CLI
    testing.
12. Add state-level parity coverage for multi-token prompt scans, then replace
    the unrolled graph with a production fused scan op/path so long prompts do
    not require excessive graph construction.
13. Harden llama.cpp memory mutation boundaries so context-owned side state
    cannot desynchronize from KV cache. Done for conservative guardrails:
    application/server call sites route through context-aware clear/remove/
    copy/keep/shift/divide APIs, and unsupported RWKV-MS copy/position
    mutation fails explicitly.
14. Add stateful slot/cache mutation semantics and tests for actual RWKV-MS
    cache reuse/context-shift support, then compare detailed traces against the
    PyTorch golden traces.

## Non-Goals For This Repo

This repo should not silently present the base GGUF as an RWKV-MS model. Runtime
code that changes llama.cpp/GGML belongs in a llama.cpp fork or branch. This repo
can own the checkpoint inspector, manifest spec, PyTorch math fixtures, PyTorch
golden traces, launch scripts, UI, and comparison harness.

## Checkpoint Inspector

Generate a tensor/config inventory for the current RWKV-MS memory checkpoint:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_checkpoint.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --output .openresearch/artifacts/gguf_memory_manifest.json
```

The manifest is useful input to a future GGUF sidecar schema. It is not a GGUF
conversion and is not consumed by llama.cpp today.

Current generated summary:

```text
tensors: 186 bf16
params: 797,808
bytes: 1,595,616
target layers inferred from tensor names: 0,1,2,3,4,5
```

## GGUF Sidecar Export

Export and validate the current RWKV-MS adapter as a GGUF sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/export_memory_gguf.py \
  --manifest-output .openresearch/artifacts/rwkv_ms_memory_sidecar_manifest.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_gguf.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
```

Current sidecar:

```text
path: /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
sha256: 0c646a776b5b12c9d3657ffd2e5e581be1eb46e858f1f404afeaa7077c02974e
size: 1,663,840 bytes
tensor name format: compact_with_source_name_manifest
tensors: 186 BF16
params: 797,808
raw tensor bytes: 1,595,616
checkpoint_validation_ok: True
```

The sidecar embeds:

- `delta_mem.config_json`;
- `delta_mem.adapter_metadata_json`;
- `delta_mem.tensor_manifest_json`;
- `delta_mem.base_gguf` and `delta_mem.base_gguf_sha256`;
- `delta_mem.mmproj_gguf` and `delta_mem.mmproj_gguf_sha256`;
- target layers `0-5`;
- active delta heads `q,o`;
- per-layer q/k/v/o output dimensions, including layer `5` full-attention
  widths `q=4096`, `k=1024`, `v=1024`, `o=2560`.

Stock llama.cpp can parse the sidecar as a GGUF file but does not execute it
with Gemma. The local llama.cpp branch now loads the sidecar tensors into
`llama_model`, adds Gemma4 attention hooks, and executes a constrained
single-token RWKV-MS stateful runtime.

## Sidecar Round Trip

Rebuild a standard delta-Mem checkpoint directory from the GGUF sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/materialize_memory_gguf.py --force
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/compare_memory_checkpoints.py
```

Current round-trip target:

```text
/run/media/xiaol/B214449214445C0B/models/delta_mem/from_gguf/gemma-4-E4B-it-rwkv-ms-memory
```

Validation result:

```text
tensor_count_left: 186
tensor_count_right: 186
total_numel_left: 797,808
total_numel_right: 797,808
ok: true
```

This proves the GGUF sidecar preserves the original adapter tensors exactly
enough to reconstruct the PyTorch runtime checkpoint.

## RWKV-MS Math Fixture

Generate and validate a small deterministic fixture using the checkpoint rebuilt
from the GGUF sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/generate_rwkv_ms_math_fixture.py \
  --output .openresearch/artifacts/rwkv_ms_math_fixture.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/validate_rwkv_ms_math_fixture.py \
  --fixture .openresearch/artifacts/rwkv_ms_math_fixture.json \
  --json
```

Current fixture:

```text
schema: delta_mem_rwkv_ms_math_fixture.v1
path: .openresearch/artifacts/rwkv_ms_math_fixture.json
sha256: 40d6f621522344e6116b5a0213abc74695e86a18243472c72c8e11baa7ecdfad
size: 849,921 bytes
layer: 0
dtype: float32
tensor records: 37
slot_indices: [0, 0, 1, 1]
final_positions: [1025]
validation ok: true
max_abs_diff: 0.0
```

The default fixture uses real layer-0 adapter tensors and starts positions just
before the configured fixed-chunk boundary, so a four-token sequence exercises
slot rollover while keeping `rwkv_ms_chunk_size=1024`. It covers:

- hidden-to-memory projections and beta/lambda gates;
- HRM/RWKV-7 `r,w,k,v,a,b,g` projection features;
- read-before-write routing and slot update;
- readout normalization/output;
- active `delta_q` and `delta_o` heads.

The checked-in 37-record fixture is the original v1 artifact and has no
streaming-predecessor input. The validator keeps that representation
backward-compatible. Newly generated v1 fixtures also record a deterministic
nonzero `initial_previous_source` and the resulting `final_previous_source`;
the patch checker runs an artifact-independent test proving that state,
positions, predecessor, and reads match between one full scan and two chunks.
This Python fixture extension does not change the Gemma-only scope of the GGUF
sidecar runtime or claim C++ coverage for the new optional records.

This is the golden input for the isolated GGML/C++ math test. Passing this
fixture validates the sidecar math kernel, not the full llama.cpp/Gemma runtime.

Current llama.cpp isolated fixture status:

```bash
/run/media/xiaol/B214449214445C0B/tools/cmake-4.3.3/bin/cmake \
  --build /run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda \
  --target test-rwkv-ms-fixture -j 8

/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/test-rwkv-ms-fixture \
  /home/xiaol/X/Multi-state-RWKV-online-memory/.openresearch/artifacts/rwkv_ms_math_fixture.json \
  /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  1e-5 1e-5
```

```text
{"ok":true,"compared":51,"sidecar":true,"max_abs_diff":1.37090683e-06}
no-sidecar pass: compared=11 max_abs_diff=5.96046448e-08
```

The C++ fixture now parses the compact GGUF sidecar through llama.cpp's C GGUF
reader, computes memory projections, computes `HRMRWKV7LowRankCore` feature
projections, drives a second C++ read-before-write scan from those
sidecar/GGML tensors, forms the graph readout from the scan `raw_reads` plus
graph `feature_g`, and derives `delta_q`/`delta_o` from the graph-produced
readout.
The llama.cpp branch also has a constrained Gemma4 generation runtime that
consumes RWKV-MS memory during serial prompt/generation scans.

Current llama.cpp sidecar runtime status:

- `--rwkv-ms-sidecar FNAME` is registered in common CLI/server params.
- `llama_model_params` carries the sidecar path into model load.
- Gemma4 model load validates the compact RWKV-MS sidecar metadata and owns all
  186 BF16 sidecar tensors in CPU buffers.
- Semantic sidecar validation at model load verifies the sidecar's
  `delta_mem.base_gguf_sha256` against the loaded base GGUF file, then rejects
  unsupported `num_state_heads != 1`, duplicate compact tensor names, missing
  required tensors, and wrong ggml-order tensor shapes before runtime use.
- Per-layer `llama_layer::rwkv_ms` pointers are assigned for layers `0-5`.
- The Gemma4 graph applies RWKV-MS `delta_q` and `delta_o` from those sidecar
  tensors, and the runtime updates mutable RWKV-MS state and previous-value
  tensors.
- `tools/llama_server_gemma4.sh` and `tools/gemma_gguf_ui.py` expose an
  explicit sidecar test path with `LLAMA_RWKV_MS=1`, `--ubatch-size 1`, one
  server slot, disabled server prompt-cache/checkpoints, and sidecar-aware
  JSONL logging plus manual slot 0 save/restore controls.
- `tools/check_rwkv_ms_gguf_runtime.py` verifies the running endpoint, saved
  reference trace, server slot 0 exact-prefix save/restore, corrupted slot
  restore rejection, and required llama-server log evidence before prompt
  testing.
- The Gradio UI can run that runtime health check directly and, by default,
  blocks RWKV-MS-labeled chat and trace comparison unless the selected
  endpoint/model/sidecar/log match a recent successful health file.
- llama.cpp now rejects RWKV-MS sidecar contexts with more than one sequence.
  The CLI graph can build experimental graph-unrolled multi-token prompt scans,
  while the recommended server/UI path keeps physical ubatches serial (`-ub 1`).
- The server launch helper preflights sidecar settings before invoking
  `llama-server`, so the recommended path fails before model load for unsafe
  slot/cache/batch/speculative overrides.
- llama.cpp context-owned memory mutation APIs now synchronize or protect the
  RWKV-MS side state. Full clear and supported full-sequence removal clear the
  RWKV-MS state; unsupported sequence copy, keep, position shift, and position
  division are rejected under RWKV-MS instead of mutating only KV cache.
- RWKV-MS recurrent state uses a v2 payload with runtime metadata and a
  deterministic sidecar fingerprint that includes the bound base GGUF hash.
  RWKV-MS tensor payload restore is staged before commit; failed server slot
  restore clears the affected context state and exposes the exact state-load
  error in the HTTP 400 response. Full and sequence state restore snapshot the
  current context before RWKV-MS-enabled loads and roll back that snapshot if
  the normal memory portion loads but the RWKV-MS sub-state fails. Slot files
  saved before base-hash binding should be regenerated.
- Current constraints: one sequence, recommended serial server/UI scans
  (`-ub 1`), `num_state_heads=1`, no production fused multi-token prompt-batch
  graph, no speculative decoding, experimental host/file plus slot 0
  exact-prefix save/load for RWKV-MS recurrent state, and conservative rejection
  rather than semantic support for sequence copy/position mutation, cache reuse,
  and context shift.

## Reference Trace

Generate a PyTorch delta-Mem reference trace using the checkpoint rebuilt from
the GGUF sidecar:

```bash
/home/xiaol/X/delta-Mem/.venv/bin/python \
  integrations/delta_mem_rwkv_ms/gguf/generate_reference_trace.py \
  --max-new-tokens 64 \
  --output .openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json \
  --save-snapshot-dir .openresearch/artifacts/gguf_reference_snapshot_from_sidecar_64
```

Current trace output:

```text
[ACTION]
get_customer_by_phone(phone_number="555-123-2002")
[/ACTION]
```

This trace is the first golden fixture for detailed llama.cpp/GGML parity:
the implementation should match the prompt tokens, generated text, state-shape
summaries, and turn stats before broader testing.

Compare the running GGUF backend against the trace:

```bash
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
.venv-ui/bin/python tools/compare_gguf_to_reference_trace.py \
  --output .openresearch/artifacts/gguf_ui/trace_compare_reasoning_off.jsonl
```

Current result with `LLAMA_REASONING=off`:

```text
exact_match: true
[ACTION]
get_customer_by_phone(phone_number="555-123-2002")
[/ACTION]
```
