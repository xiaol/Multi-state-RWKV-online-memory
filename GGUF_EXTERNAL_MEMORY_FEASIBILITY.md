# GGUF External Memory Feasibility

## Short Answer

The current RWKV-MS / delta-Mem online memory checkpoint still does **not** work
with a normal stock GGUF conversion alone. It now has a local experimental
llama.cpp sidecar runtime path for Gemma4, but that path requires the custom
`--rwkv-ms-sidecar` branch and serial single-token physical microbatches.

A plain GGUF Gemma model plus `delta_mem_adapter.pt` is not enough. Standard
GGUF runners such as llama.cpp load a static GGML graph and do not know how to
run delta-Mem's per-layer online memory hooks.

## Current First-Step GGUF Artifact

The useful first step is ready: a base Gemma4 E4B GGUF for normal llama.cpp
inference on the 2 TB SSD. This is for tokenizer/chat-template, latency, prompt,
and baseline behavior testing. The base GGUF by itself does **not** activate
RWKV-MS online memory; use the sidecar command below for the experimental local
runtime.

Selected repo:

```text
ggml-org/gemma-4-E4B-it-GGUF
```

Downloaded files:

```text
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/mmproj-gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
```

Observed file sizes:

```text
gemma-4-E4B-it-Q8_0.gguf          8,031,240,160 bytes
mmproj-gemma-4-E4B-it-Q8_0.gguf     559,874,528 bytes
gemma-4-E4B-it-rwkv-ms-memory.gguf    1,663,840 bytes
```

SHA256:

```text
fb8f0c032de00b18c710824af3c7e5777c71e5fb60b13f13575f0a9e92ddecd0  gemma-4-E4B-it-Q8_0.gguf
51d4b7fd825e4569f746b200fccc5332bf914e8ef7cbe447272ce4fec6df3db6  mmproj-gemma-4-E4B-it-Q8_0.gguf
0c646a776b5b12c9d3657ffd2e5e581be1eb46e858f1f404afeaa7077c02974e  gemma-4-E4B-it-rwkv-ms-memory.gguf
```

Reproduce the download:

```bash
bash tools/download_gemma4_e4b_gguf.sh
```

Serve it with a recent llama.cpp build:

```bash
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh
```

Serve the experimental RWKV-MS sidecar runtime with the patched llama.cpp build:

```bash
mkdir -p .openresearch/artifacts/gguf_ui
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_PORT=18083 \
LLAMA_RWKV_MS=1 \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh 2>&1 | tee .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log
```

Patched llama.cpp fork:

```text
https://github.com/xiaol/llama.cpp-online-memory
branch: main
commit: 85da0c63b Add Gemma4 RWKV-MS GGUF sidecar runtime
base upstream: ggml-org/llama.cpp 1ec44d1
```

In `LLAMA_RWKV_MS=1` mode, `tools/llama_server_gemma4.sh` passes the sidecar,
uses `--batch-size 2 --ubatch-size 1`, serves one slot, disables continuous
batching/context shift/prompt-cache reuse, disables server prompt-cache RAM and
context checkpoints, creates a slot-save directory for manual slot 0
save/restore, and switches to text-only mode. Those settings match the current
runtime constraints.
The patched llama.cpp context now rejects RWKV-MS sidecar runs with more than
one sequence instead of silently skipping the memory graph. The recommended
server path keeps `--ubatch-size 1` and rejects speculative decoding; the launch
helper also preflights unsafe sidecar settings before invoking `llama-server`.

Verified local backend:

```text
/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server
fork commit: 85da0c63b
base upstream: 1ec44d1
device: CUDA0 NVIDIA GeForce RTX 4090
```

Text-only mode:

```bash
LLAMA_TEXT_ONLY=1 LLAMA_SERVER_BIN=/path/to/llama-server \
bash tools/llama_server_gemma4.sh
```

Launch the local test UI after `llama-server` is running:

```bash
python3.12 -m venv .venv-ui
.venv-ui/bin/pip install -r requirements-ui.txt
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
.venv-ui/bin/python tools/gemma_gguf_ui.py
```

The UI writes generation logs to:

```text
.openresearch/artifacts/gguf_ui/chat.jsonl
```

The RWKV-MS memory adapter is also exported as a GGUF sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/export_memory_gguf.py \
  --manifest-output .openresearch/artifacts/rwkv_ms_memory_sidecar_manifest.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_gguf.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
```

Current sidecar validation:

```text
schema: delta_mem_rwkv_ms_memory_gguf_sidecar.v1
metadata status: sidecar_only_runtime_port_required
local runtime status: experimental_llama_cpp_sidecar_runtime_available
base_gguf_sha256: fb8f0c032de00b18c710824af3c7e5777c71e5fb60b13f13575f0a9e92ddecd0
tensor name format: compact_with_source_name_manifest
tensors: 186 BF16
tensor bytes: 1,595,616
target layers: 0,1,2,3,4,5
delta heads: q,o
checkpoint_validation_ok: True
```

The sidecar can also rebuild a normal delta-Mem checkpoint directory:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/materialize_memory_gguf.py --force
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/compare_memory_checkpoints.py
```

Current round-trip result:

```text
tensor_count_left: 186
tensor_count_right: 186
total_numel_left: 797,808
total_numel_right: 797,808
ok: true
```

Isolated RWKV-MS math fixture from the sidecar-rebuilt checkpoint:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/generate_rwkv_ms_math_fixture.py \
  --output .openresearch/artifacts/rwkv_ms_math_fixture.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/validate_rwkv_ms_math_fixture.py \
  --fixture .openresearch/artifacts/rwkv_ms_math_fixture.json \
  --json
```

Current fixture result:

```text
schema: delta_mem_rwkv_ms_math_fixture.v1
layer: 0
dtype: float32
fixture size: 849,921 bytes
fixture sha256: 40d6f621522344e6116b5a0213abc74695e86a18243472c72c8e11baa7ecdfad
tensor records: 37
slot_indices: [0, 0, 1, 1]
final_positions: [1025]
validation ok: true
max_abs_diff: 0.0
```

This fixture covers the real adapter tensors for the RWKV-MS projection stack,
read-before-write scan, slot update, readout, and active `q,o` delta heads. It
is a PyTorch golden math fixture for a future GGML implementation; it is not
stock llama.cpp execution.

The tracked 37-record v1 fixture predates streaming predecessor persistence and
remains valid for compatibility. The current generator can additionally carry
a nonzero `initial_previous_source` and record `final_previous_source`; an
artifact-independent Python regression compares one full scan with two chunks.
The optional records do not broaden this experimental GGUF path beyond Gemma4.

The local llama.cpp checkout also has an isolated C++ fixture target:

```bash
/run/media/xiaol/B214449214445C0B/tools/cmake-4.3.3/bin/cmake \
  --build /run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda \
  --target test-rwkv-ms-fixture -j 8

/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/test-rwkv-ms-fixture \
  .openresearch/artifacts/rwkv_ms_math_fixture.json \
  /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  1e-5 1e-5
```

Current result:

```text
{"ok":true,"compared":51,"sidecar":true,"max_abs_diff":1.37090683e-06}
no-sidecar pass: compared=11 max_abs_diff=5.96046448e-08
```

That target now proves `tests/test-rwkv-ms-fixture.cpp` in llama.cpp can parse
the compact sidecar, compute memory projections, compute
`HRMRWKV7LowRankCore` feature projections, drive a second C++
read-before-write scan from those sidecar/GGML tensors, form the graph readout
from the scan `raw_reads` plus graph `feature_g`, and derive `delta_q`/`delta_o`
from the graph-produced readout.

The local llama.cpp branch now has an experimental Gemma4 runtime slice:

```bash
/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-completion \
  -m /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf \
  --rwkv-ms-sidecar /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  -p "Hi" -n 32 -c 96 -b 2 -ub 1 -ngl 99 --no-warmup --no-display-prompt \
  --no-perf -no-cnv -s 123 --temp 0 --top-k 1
```

This path adds `--rwkv-ms-sidecar`, validates the compact sidecar metadata, loads
all 186 BF16 sidecar tensors into model-owned CPU buffers, and assigns per-layer
`llama_layer::rwkv_ms` tensor pointers for layers `0-5`. It also creates mutable
RWKV-MS state and previous-value tensors, wires the sidecar projections into the
Gemma4 graph, applies `delta_q` and `delta_o`, and updates state on single-token
microbatches. The graph can also build an experimental unrolled multi-token
prompt scan, but the server/UI path intentionally keeps serial physical
microbatches (`-ub 1`) until state-level parity coverage is stronger. Current
constraints: one sequence, recommended serial server/UI scans, `num_state_heads=1`,
no production fused prompt-batch scan, and no speculative decoding. Host/file
session state and server slot 0 save/restore now serialize the RWKV-MS recurrent
state, but server restore is only supported for exact-prefix continuation and
remains experimental. The RWKV-MS state payload is now versioned with sidecar
metadata and a deterministic sidecar fingerprint, and the RWKV-MS tensor payload
is staged before committing so a bad sidecar sub-payload does not partially
overwrite RWKV-MS tensors. Failed server slot restore clears the affected
slot/context state and returns the exact state-load error, including sidecar
identity mismatches. Full and sequence state restore now snapshot the current
context before RWKV-MS-enabled loads and roll back that snapshot if the normal
memory portion loads but the RWKV-MS sub-state fails.
The runtime has guardrails for the first two constraints: unsafe context
settings fail during context creation, and the server launch path rejects unsafe
slot/cache/batch/speculative settings before graph construction.
llama.cpp also routes context-owned memory clear/remove/copy/keep/shift/divide
operations through `llama_context_memory_*` APIs. With RWKV-MS enabled, full
clear and safe full-sequence removal synchronize the side state; unsupported
sequence copy, keep, position-shift, and position-division paths are rejected
instead of mutating only the KV cache.
Model load also performs semantic sidecar validation before runtime use: it
verifies the sidecar's `delta_mem.base_gguf_sha256` against the loaded base
GGUF file, then rejects unsupported `num_state_heads != 1`, duplicate compact
tensor names, missing required tensors, and wrong ggml-order tensor shapes.

The same constrained runtime is available through `llama-server` via:

```bash
mkdir -p .openresearch/artifacts/gguf_ui
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_RWKV_MS=1 \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh 2>&1 | tee .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log
```

Then run the UI with:

```bash
LLAMA_RWKV_MS=1 LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
LLAMA_BASE_URL=http://127.0.0.1:18083/v1 \
LLAMA_SERVER_LOG=.openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
GGUF_RWKV_MS_HEALTH_OUTPUT=.openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json \
GGUF_UI_REQUIRE_RWKV_MS_HEALTH=1 \
GGUF_UI_PORT=7861 \
.venv-ui/bin/python tools/gemma_gguf_ui.py
```

Verify the running endpoint before prompt testing. The Gradio UI has the same
RWKV-MS runtime verification action and, by default, refuses sidecar chat or
trace comparison unless the selected endpoint/model/sidecar/log match a recent
successful health file.

```bash
.venv-ui/bin/python tools/check_rwkv_ms_gguf_runtime.py \
  --base-url http://127.0.0.1:18083/v1 \
  --server-log .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
  --output .openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json
```

Golden PyTorch trace from the sidecar-rebuilt checkpoint:

```bash
/home/xiaol/X/delta-Mem/.venv/bin/python \
  integrations/delta_mem_rwkv_ms/gguf/generate_reference_trace.py \
  --max-new-tokens 64 \
  --output .openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json \
  --save-snapshot-dir .openresearch/artifacts/gguf_reference_snapshot_from_sidecar_64
```

Observed assistant text:

```text
[ACTION]
get_customer_by_phone(phone_number="555-123-2002")
[/ACTION]
```

The running GGUF backend can be compared to this trace:

```bash
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
.venv-ui/bin/python tools/compare_gguf_to_reference_trace.py \
  --output .openresearch/artifacts/gguf_ui/trace_compare_reasoning_off.jsonl
```

Current first-fixture comparison with `LLAMA_REASONING=off`:

```text
exact_match: true
GGUF output:
[ACTION]
get_customer_by_phone(phone_number="555-123-2002")
[/ACTION]
```

## Why It Does Not Work Directly

The RWKV-MS memory is not just an external vector database or a static sidecar
tensor file. In the current PyTorch runtime it needs to:

- attach online memory read/write modules inside Gemma text-attention layers;
- read hidden states from selected layers, currently Gemma text layers `0-5`;
- update recurrent RWKV-MS state token by token;
- inject learned deltas into attention `q` and `o`;
- keep RWKV-MS online state synchronized with KV cache/session state;
- preserve/reset/save/load memory state across chat turns.

The current implementation depends on the patched delta-Mem runtime for those
operations. GGUF stores model tensors and metadata for GGML executors, but a
standard GGUF engine does not automatically expose arbitrary per-layer read/write
hooks or delta-Mem session state.

## What A GGUF Version Would Require

A real GGUF path is possible, but it is a custom runtime port, not a simple
quantization/export step.

Required work:

1. Convert the base Gemma checkpoint to GGUF normally.
2. Convert the RWKV-MS memory weights into GGUF-compatible tensors or a
   sidecar file.
3. Patch llama.cpp / GGML to implement the RWKV-MS online memory module:
   - recurrent RWKV-MS state buffers;
   - per-token read-before-write update;
   - layer `0-5` hooks;
   - `q,o` delta injection;
   - state reset/save/load;
   - synchronization with the KV cache.
4. Match the delta-Mem tokenizer and chat-template behavior.
5. Validate isolated GGML math against the RWKV-MS math fixture.
6. Validate the GGUF runtime against the PyTorch delta-Mem reference on fixed
   prompts and state-debug outputs.

## Honest Milestone Checklist

Completed first-step artifacts:

- [x] identify a Gemma4 E4B GGUF matching the PyTorch base model target;
- [x] download the GGUF model and multimodal projector to the 2 TB SSD;
- [x] add a reproducible download helper;
- [x] add a llama.cpp server launch helper;
- [x] add a Gradio UI for controlled local GGUF testing;
- [x] add a JSONL batch evaluator for repeatable prompt checks.
- [x] add a memory-checkpoint inspector for future GGUF/GGML port manifests.
- [x] export the RWKV-MS adapter tensors to a GGUF sidecar and validate the
      sidecar against the original PyTorch checkpoint.
- [x] round-trip the GGUF sidecar back into a delta-Mem checkpoint directory and
      verify tensor-level equality.
- [x] generate and validate an isolated PyTorch RWKV-MS math fixture from the
      sidecar-rebuilt checkpoint.
- [x] add a llama.cpp C++ fixture that reads the compact sidecar and validates
      memory projections, `HRMRWKV7LowRankCore` feature projections, graph
      sidecar-driven scan/readout, and `delta_q`/`delta_o` parity.
- [x] add an experimental llama.cpp Gemma4 runtime path for `--rwkv-ms-sidecar`
      that owns the compact sidecar tensors and consumes them during serial
      prompt/generation scans.
- [x] generate a PyTorch golden trace from the checkpoint rebuilt from GGUF.
- [x] add a UI/CLI comparison harness and verify the running GGUF backend
      matches generated text for the first sidecar-derived reference trace with
      reasoning disabled.
- [x] update the server helper and Gradio UI so the patched sidecar runtime has
      an explicit one-sequence, `-ub 1` test path.
- [x] serialize RWKV-MS recurrent state in llama.cpp host/file state and server
      slot 0 save/restore paths.
- [x] add v2 RWKV-MS state identity checks, sidecar-local staged restore, and a
      corrupted server-slot restore health check that expects a 400 sidecar
      identity mismatch.
- [x] bind the sidecar runtime to the exact base GGUF SHA-256 at llama.cpp model
      load and include that identity in the RWKV-MS state fingerprint.
- [x] make RWKV-MS-enabled full and sequence state restore transactional across
      normal memory plus RWKV-MS state on load failure, with health-log evidence
      from corrupted slot restore.
- [x] harden llama.cpp memory mutation APIs so application/server call sites use
      context-aware clear/remove/copy/keep/shift/divide wrappers; unsupported
      RWKV-MS sequence copy and position mutation now fail explicitly instead
      of desynchronizing KV and RWKV-MS state.

Still required before calling this "RWKV-MS GGUF":

- [ ] generalize recurrent RWKV-MS state update/readout beyond the current
      single-sequence runtime path, including state-level parity tests for the
      experimental multi-token graph and a production fused prompt scan;
- [ ] implement real RWKV-MS semantics for sequence copy, position mutation,
      cache reuse, and context shift instead of the current conservative
      rejection guardrails;
- [ ] compare detailed llama.cpp/GGML state traces and session behavior against
      the PyTorch delta-Mem reference, beyond the current deterministic
      generated-text smoke.

## Practical Recommendation

Use the current PyTorch + patched delta-Mem runtime as the reference
implementation.

Treat GGUF as a next engineering milestone:

- first make the base model GGUF for normal inference;
- then prototype a llama.cpp/GGML RWKV-MS runtime hook;
- only call it a GGUF release once the online memory behavior matches the
  PyTorch reference.

Suggested model-card wording:

> GGUF is planned, but this checkpoint is not GGUF-ready yet. A real GGUF path
> requires GGML/llama.cpp runtime support for online RWKV-MS state, per-layer
> read/write hooks, `q,o` delta injection, and session-state synchronization.
