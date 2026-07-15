# Multi-State RWKV Online Memory

Mechanism-level experiments for comparing Dynamic Linear Attention (DLA) with
RWKV-style online memory under controlled state and boundary policies.

HF checkpoint:
[`xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1`](https://huggingface.co/xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1)

This repository starts from the Log-Linear Attention codebase and adds a
CPU-only proof of concept in `dla_poc.py`. It reproduces the core DLA mechanism
from arXiv 2606.10650 and adds HRM-Text-inspired memory baselines:

- `rwkv_mem(delta_rule)`: single online delta-rule associative memory.
- `rwkv_mem(rwkv7)`: single read-before-write RWKV-7 state.
- `rwkv_mem(rwkv7 multi-state)`: same RWKV-7 state update, but one state per
  adaptive memory block.
- State-only ablation: fixes the exact same boundaries for linear/DLA states
  and RWKV-7 states, so the comparison isolates the state update.

## Quick Start

```bash
python3.12 -m venv .venv
PATH="$PWD/.venv/bin:$PATH" bash run.sh
```

If the environment is already set up:

```bash
.venv/bin/python dla_poc.py
```

Outputs are written to:

```text
EVAL.md
.openresearch/artifacts/dla_summary.json
.openresearch/artifacts/dla_trials.jsonl
.openresearch/artifacts/dla_comparison.png
.openresearch/artifacts/run_log.txt
```

## Current Result

The main DLA reproduction still passes:

- DLA lowers the Theorem 3.1 deviation bound in every tested config.
- DLA beats fixed Log-Linear blocking on needle recall at matched state count.
- The repo Log-Linear attention smoke test passes on CPU.

Mechanism recall comparison:

| needles | filler/seg | K | states | fixed | rwkv_mem(delta_rule) | rwkv_mem(rwkv7) | rwkv_mem(rwkv7 multi-state) | DLA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8 | 16 | 12.0 | 0.920 | 0.229 | 0.797 | 1.000 | 1.000 |
| 10 | 6 | 24 | 20.0 | 0.934 | 0.122 | 0.682 | 1.000 | 1.000 |
| 8 | 10 | 20 | 16.0 | 0.887 | 0.046 | 0.626 | 1.000 | 1.000 |

State-update-only comparison, with identical boundaries for both states:

| boundary policy | needles | filler/seg | K | states | linear/DLA state | RWKV-7 state | RWKV - linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | 8 | 12 | 16 | 16.0 | 1.000 | 1.000 | +0.000 |
| dla | 8 | 12 | 16 | 16.0 | 1.000 | 1.000 | +0.000 |
| fixed | 8 | 12 | 16 | 16.0 | 0.848 | 0.980 | +0.133 |
| noisy_dla | 8 | 12 | 16 | 16.0 | 0.874 | 0.987 | +0.112 |
| low_k_dla | 8 | 12 | 16 | 8.0 | 0.640 | 0.952 | +0.313 |
| oracle | 12 | 10 | 16 | 24.0 | 1.000 | 1.000 | +0.000 |
| dla | 12 | 10 | 16 | 16.0 | 0.792 | 0.982 | +0.190 |
| fixed | 12 | 10 | 16 | 16.0 | 0.763 | 0.991 | +0.228 |
| noisy_dla | 12 | 10 | 16 | 16.0 | 0.691 | 0.973 | +0.282 |
| low_k_dla | 12 | 10 | 16 | 8.0 | 0.516 | 0.889 | +0.373 |
| oracle | 16 | 8 | 12 | 32.0 | 1.000 | 1.000 | +0.000 |
| dla | 16 | 8 | 12 | 12.0 | 0.556 | 0.827 | +0.272 |
| fixed | 16 | 8 | 12 | 12.0 | 0.649 | 0.972 | +0.324 |
| noisy_dla | 16 | 8 | 12 | 12.0 | 0.509 | 0.819 | +0.311 |
| low_k_dla | 16 | 8 | 12 | 6.0 | 0.371 | 0.669 | +0.299 |

This table fixes the exact same token blocks for both methods. `linear/DLA state`
uses the standard block sum `sum k_t v_t^T`; `RWKV-7 state` uses the RWKV-7
recurrence inside each same block. Therefore each row compares state
update/readout only, not boundary quality.

Interpretation:

- With perfect or near-perfect boundaries, linear/DLA state and RWKV-7 state tie
  on this synthetic recall task.
- When boundaries are fixed, noisy, or compressed to low K, RWKV-7 state is more
  robust in this task.
- DLA's main advantage is adaptive boundary/state allocation; RWKV-7's advantage
  appears in the state update when boundaries are held fixed and imperfect.

Full tables are in `EVAL.md`.

## What Is Compared

`dla_poc.py` runs four groups of checks.

1. Codebase smoke test
   - Loads the original Log-Linear Attention pure PyTorch path directly.
   - Avoids CUDA-only Triton/Mamba dependencies.

2. DLA deviation-bound check
   - Implements Algorithm 1: information-aware dynamic state merging.
   - Implements Algorithm 2: capacity-bounded adjacent state merging.
   - Compares DLA blocks against fixed contiguous blocks at matched state count.

3. Needle associative recall
   - Uses synthetic rare needle tokens mixed with redundant filler tokens.
   - Compares fixed blocks, DLA blocks, delta-rule memory, RWKV-7 memory, and
     multi-state RWKV-7 memory.

4. State-update-only ablation
   - Uses the same block boundaries for both linear/DLA and RWKV-7 state update.
   - Boundary policies: `oracle`, `dla`, `fixed`, `noisy_dla`, `low_k_dla`.
   - This isolates whether the state update/readout is stronger, independent of
     boundary selection.

## Scope

This is a training-free mechanism reproduction. It does not reproduce 50B-token
pretraining, downstream language-model evaluations, or trained HRM-Text
checkpoints.

The HRM/RWKV baselines are self-contained ports of the memory recurrence ideas,
not full imports of HRM-Text:

- `rwkv_mem(delta_rule)` follows the read-before-write delta-rule associative
  state from HRM-Text's `models/rwkv_memory.py`.
- `rwkv_mem(rwkv7)` follows the latest read-before-write RWKV-7 recurrence from
  HRM-Text's `models/rwkv7.py`, specialized to the synthetic key/value stream.

## Repository Layout

```text
dla_poc.py                         # Main reproduction and comparisons
run.sh                             # CPU dependency install + run
EVAL.md                            # Generated report from latest run
.openresearch/artifacts/           # JSONL, JSON, figure, run log
hattention/                        # Log-Linear Attention implementation used for smoke test
figs/                              # Original figure asset
deltamem/                          # bundled patched HF online-memory runtime
integrations/delta_mem_rwkv_ms/    # launchers, docs, GGUF tools, optional upstream patch
integrations/delta_mem_rwkv_ms/gguf/ # GGUF sidecar, fixture, and parity helpers
```

## Delta-Mem RWKV-MS Online Memory

The practical RWKV-MS online-memory integration is self-contained in this
repository. The patched Python runtime is bundled at top-level `deltamem/`, so
normal Qwen/Gemma HF training and inference do not require another delta-Mem
checkout. `integrations/delta_mem_rwkv_ms/` contains HF inference and verified
manual training-smoke entry points, a matched delta-rule/RWKV-MS launcher, GGUF
tools, and an optional upstream patch export. The runtime supports Qwen3,
Qwen3.5/Qwen3.6, SmolLM3, and Gemma4 text attention;
for `google/gemma-4-E4B-it` it wraps the non-KV-shared attention layers and
skips the KV-shared tail layers.

Transformers exposes Qwen3.6 as `qwen3_5`. Its 64-layer hybrid stack has 16
full-attention layers at physical indices `3,7,11,...,63`; the other 48 Gated
DeltaNet layers are not wrapped. Use layer `3` for a smoke run or
`3,7,11,15,19,23` for the six early eligible layers. This Qwen path is the HF
integration and is separate from the Gemma-only GGUF sidecar runtime.

The bundled `deltamem/` package provides the wrapper/session machinery:
attaching online-memory modules to a Transformers model, loading
`delta_mem_adapter.pt`, keeping RWKV-MS state synchronized with the KV cache,
and applying the chat template. The optional
`integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch` exports these changes
for upstream delta-Mem revision `5cd5d9153c7f408764728d953565201e198c39e2`;
it is not needed for normal use of this repository.
See [bundled runtime provenance](integrations/delta_mem_rwkv_ms/BUNDLED_RUNTIME.md)
for the source snapshot and local integration revision.

For HF workflows, install the bundled package from the repository root before
running the commands below:

```bash
pip install -r requirements.txt
pip install -e .
```

### Gemma tau2 status

The active Gemma + RWKV-MS tau2 recipe is documented in
`GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md`. For reproducibility, the benchmark
artifacts record this historical source integration commit (it is not a current
external runtime dependency):

```text
bec8330 Add RWKV-MS memory backend for Gemma tau2
```

Current best learned no-rule online-memory checkpoint:

```text
xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
```

Local source checkpoint:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100
```

The table below keeps learned online-memory runs separate from rule-assisted diagnostic
runs. "No-rule" means no eval-time `--mobile-data-rule-planner` and no parser
format-repair patch.

Release framing: "One can have both the fish and the bear's paw." The base
Gemma checkpoint remains frozen to preserve original behavior, while the learned
RWKV-MS path adds a small recurrent memory surface that can be adapted to local
domain data.

| Run / condition | Layers / rank / length | pass^1 | Takeaway |
| --- | --- | ---: | --- |
| Base checkpoint `google/gemma-4-E4B-it`, focused tools + line verify + autostop | none | 4/20 (0.20) | Current base-only baseline for the accepted setup |
| Base checkpoint `google/gemma-4-E4B-it`, checklist prompt | none | 7/20 (0.35) | Prompt-only baseline, still below learned best |
| Original 82-row Phase 1 | `0,1` / r8 / len256 | 1/20 (0.05) | Dataset/format mismatch; reject |
| Generated action SFT | `0,1` / r8 / len256 | 9/20 (0.45) | 2 layers help but are not enough |
| Generated action SFT | `0-5` / r8 / len256 | 10/20 (0.50) | Shallow 6-layer band is better |
| Generated action SFT | all eligible / r4 / len256 | 1/20 (0.05) | All-layer memory path over-perturbs |
| Format-refresh continuation, final | `0-5` / r8 / len192 | 12/20 (0.60) | Good final checkpoint |
| Format-refresh continuation, `step-100` | `0-5` / r8 / len192 | **14/20 (0.70)** | Best learned no-rule checkpoint |

Memory-path size from saved checkpoints:

| Memory-path shape | Trainable memory params |
| --- | ---: |
| 2 layers, r8 `q,o` | 257,744 |
| 6 layers, r8 `q,o` | 797,808 |
| 24 eligible layers, r4 `q,o` | 1,594,080 |

Local training-cost notes:

- Experiments were local CUDA bf16 runs on an RTX 4090 24 GB setup.
- Generated mobile-data action SFT used 3,519 turn rows for 656 optimizer steps.
- Format-refresh continuation used 5,027 turn rows for 200 optimizer steps.
- Exact wall time and VRAM vary with local hardware, sequence length, layer
  count, rank, cache location, and fragmentation; adapt the frozen-base
  online-memory recipe to your own data.

Status interpretation:

- The original tau2 data was the problem: the 82-row run trained for 656
  optimizer steps and its loss moved, but the benchmark collapsed to 1/20.
- Generated mobile-data action SFT transfers better, and the 6-layer shallow
  online-memory path is the current useful capacity point.
- The 200-step format-refresh continuation overtrains relative to its
  `step-100` checkpoint, so checkpoint selection matters.
- The eval-time rule planner / float-format fix is excluded from the comparison
  table because it is benchmark-specific control logic, not model behavior.
- The next benchmark should run the `step-100` checkpoint on at least 50 tasks,
  preferably the full telecom split, before treating 14/20 as robust.

Recommended HF online-memory inference command:

```bash
python integrations/delta_mem_rwkv_ms/inference.py \
  --memory-repo xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --base-model google/gemma-4-E4B-it \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa
```

## Gemma4 GGUF First Step

A base Gemma4 E4B GGUF has been downloaded for llama.cpp testing on the 2 TB SSD:

```text
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/mmproj-gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
```

The first two files are normal base-model inference artifacts. The RWKV-MS
memory file is a GGUF sidecar containing the adapter tensors and metadata. The
local llama.cpp branch can now consume that sidecar in an experimental Gemma4
runtime path: model load owns the sidecar tensors in CPU buffers, the Gemma4
graph applies RWKV-MS `q,o` deltas on target layers `0-5`, and a mutable
RWKV-MS state buffer is updated during prompt/generation scans. The current
runtime is intentionally constrained to one sequence. The server/UI path keeps
physical microbatches serial (`-ub 1`) for the best-tested state behavior; the
CLI graph can build experimental graph-unrolled multi-token prompt scans, but
that path still needs stronger state-level parity coverage before it should be
treated as production-ready. See
`GGUF_EXTERNAL_MEMORY_FEASIBILITY.md` and
`integrations/delta_mem_rwkv_ms/GGUF_PORT_PLAN.md`.
At llama.cpp model load time, the sidecar path now performs semantic validation
before runtime use: it verifies `delta_mem.base_gguf_sha256` against the exact
loaded base GGUF file, then rejects unsupported `num_state_heads != 1`,
duplicate compact tensor names, missing required tensors, and wrong ggml-order
tensor shapes.

Patched llama.cpp fork:

```text
https://github.com/xiaol/llama.cpp-online-memory
branch: main
commit: 85da0c63b Add Gemma4 RWKV-MS GGUF sidecar runtime
base upstream: ggml-org/llama.cpp 1ec44d1
```

Current sidecar identity:

```text
sha256: 0c646a776b5b12c9d3657ffd2e5e581be1eb46e858f1f404afeaa7077c02974e
bound base GGUF sha256: fb8f0c032de00b18c710824af3c7e5777c71e5fb60b13f13575f0a9e92ddecd0
size: 1,663,840 bytes
tensor name format: compact_with_source_name_manifest
tensors: 186 BF16
```

Start a recent llama.cpp server:

```bash
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh
```

For the experimental RWKV-MS sidecar runtime through `llama-server`, use the
patched llama.cpp build and the constrained sidecar mode:

```bash
mkdir -p .openresearch/artifacts/gguf_ui
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_PORT=18083 \
LLAMA_RWKV_MS=1 \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh 2>&1 | tee .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log
```

The helper sets `--rwkv-ms-sidecar`, `--batch-size 2`, `--ubatch-size 1`,
`--parallel 1`, disables continuous batching/context shift/prompt-cache reuse,
disables server prompt-cache RAM and context checkpoints, enables a slot-save
directory for manual slot 0 save/restore, and uses text-only mode for the
current one-sequence runtime.
The patched llama.cpp context rejects sidecar runs with more than one sequence.
The server/helper keep `--ubatch-size 1`, reject speculative decoding, and
preflight unsafe slot/cache/batch overrides before starting `llama-server`;
model load also rejects malformed or unsupported sidecars before any RWKV-MS
graph consumes their tensors. A sidecar exported for a different base GGUF now
fails model load with a hash mismatch instead of running against the wrong
weights.

For the best-tested experimental runtime path, pass the sidecar and use serial
physical microbatches:

```bash
/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-completion \
  -m /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf \
  --rwkv-ms-sidecar /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  -p "Hi" -n 32 -c 96 -b 2 -ub 1 -ngl 99 --no-warmup --no-display-prompt \
  --no-perf -no-cnv -s 123 --temp 0 --top-k 1
```

With the same seed and greedy sampling, the base and sidecar paths now diverge.
The sidecar path produced `! I'm excited to chat with you. What's on your mind
today? ...`, while the base path continued `! I'm excited to chat with you. I'm
here to help ...`. Treat this as a smoke signal consistent with the sidecar path;
confirm runtime use with server logs and the reference-trace health check.

The local CUDA build is from the online-memory fork commit `85da0c63b`, based
on upstream llama.cpp `1ec44d1`, and detects the RTX 4090 as `CUDA0`. CUDA 13.1
plus GCC 15 needed a local header shim during build; the resulting binary is
under the SSD tool directory above.

Then launch the local testing UI:

```bash
python3.12 -m venv .venv-ui
.venv-ui/bin/pip install -r requirements-ui.txt
LLAMA_BASE_URL=http://127.0.0.1:18083/v1 \
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
GGUF_RWKV_MS_SIDECAR_PATH=/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
LLAMA_SERVER_LOG=.openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
GGUF_RWKV_MS_HEALTH_OUTPUT=.openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json \
GGUF_UI_REQUIRE_RWKV_MS_HEALTH=1 \
GGUF_UI_PORT=7861 \
.venv-ui/bin/python tools/gemma_gguf_ui.py
```

Before comparing prompts, verify that the endpoint is really the patched
sidecar runtime. The UI exposes the same check through its RWKV-MS runtime
button, writes the health file, and blocks sidecar chat/trace comparison while
the selected endpoint/model/sidecar/log do not match a recent successful check.

```bash
.venv-ui/bin/python tools/check_rwkv_ms_gguf_runtime.py \
  --base-url http://127.0.0.1:18083/v1 \
  --server-log .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
  --output .openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json
```

The check requires the server log because API output alone cannot prove that
llama.cpp loaded the RWKV-MS sidecar. It verifies model listing, a chat smoke
request, the saved reference trace, slot 0 save/restore with exact-prefix
continuation, corrupted slot restore rejection, and log evidence for RWKV-MS
activation, one server slot, disabled prompt cache, disabled context
checkpoints, and exact-prefix slot reuse. The sidecar server also rejects
speculative decoding options.

For repeatable prompt checks against the same server:

```bash
.venv-ui/bin/python tools/eval_gguf_prompts.py configs/gguf_rwkv_ms_prompt_suite.jsonl \
  --base-url http://127.0.0.1:18083/v1 \
  --model gemma-4-e4b-it-rwkv-ms-q8 \
  --rwkv-ms \
  --temperature 0 \
  --seed 42
```

For the RWKV-MS side of the future port, inspect the PyTorch memory checkpoint
into a tensor/config manifest:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_checkpoint.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --output .openresearch/artifacts/gguf_memory_manifest.json
```

To regenerate and validate the GGUF memory sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/export_memory_gguf.py \
  --manifest-output .openresearch/artifacts/rwkv_ms_memory_sidecar_manifest.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_gguf.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/materialize_memory_gguf.py --force
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/compare_memory_checkpoints.py
```

To generate and validate the isolated RWKV-MS math fixture from the
sidecar-rebuilt checkpoint:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/generate_rwkv_ms_math_fixture.py \
  --output .openresearch/artifacts/rwkv_ms_math_fixture.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/validate_rwkv_ms_math_fixture.py \
  --fixture .openresearch/artifacts/rwkv_ms_math_fixture.json \
  --json
```

The current fixture uses real layer-0 adapter tensors, covers projection,
read-before-write state update, readout, and active `q,o` delta heads, and
validates with `max_abs_diff: 0.0`. It is a PyTorch golden math fixture for a
future GGML port, not stock llama.cpp memory execution.

The local llama.cpp checkout has an isolated C++ fixture for the compact
sidecar:

```bash
/run/media/xiaol/B214449214445C0B/tools/cmake-4.3.3/bin/cmake \
  --build /run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda \
  --target test-rwkv-ms-fixture -j 8

/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/test-rwkv-ms-fixture \
  .openresearch/artifacts/rwkv_ms_math_fixture.json \
  /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  1e-5 1e-5
```

Current strict sidecar result: `{"ok":true,"compared":51,"sidecar":true,"max_abs_diff":1.37090683e-06}`.
The no-sidecar run also passes with `compared=11` and `max_abs_diff=5.96046448e-08`.
This covers `tests/test-rwkv-ms-fixture.cpp` in llama.cpp parsing the compact
sidecar, computing memory projections, `HRMRWKV7LowRankCore` feature
projections, driving a second C++ read-before-write scan from those
sidecar/GGML tensors, graph readout from the scan `raw_reads` plus graph
`feature_g`, and `delta_q`/`delta_o` from the graph-produced readout. This
fixture remains the isolated math parity check; the separate `llama-completion`
smoke above is the Gemma4 generation runtime check.

The local llama.cpp checkout also has `tests/test-rwkv-ms-state.cpp` for the
RWKV-MS recurrent state payload. It checks v2 state metadata, deterministic
sidecar fingerprint validation, staged sidecar-local restore, and rejection for
metadata/fingerprint/length mismatches. The fingerprint now includes the bound
base GGUF hash, so slot files created before that binding should be regenerated.
Full and sequence state restore now snapshot the current context before
RWKV-MS-enabled loads and roll back that snapshot if the normal memory portion
loads but the RWKV-MS sub-state fails. Failed server slot restore still clears
the affected slot/context state after the library rollback and returns the
exact state-load error.
Context-owned memory mutation now uses llama.cpp `llama_context_memory_*`
wrappers in the patched paths: clear and supported full-sequence removal keep
RWKV-MS state synchronized, while unsupported sequence copy, keep, shift, and
division fail explicitly under RWKV-MS instead of mutating only KV cache.

To generate the first PyTorch golden trace from the sidecar-rebuilt checkpoint:

```bash
.venv/bin/python \
  integrations/delta_mem_rwkv_ms/gguf/generate_reference_trace.py \
  --max-new-tokens 64 \
  --output .openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json \
  --save-snapshot-dir .openresearch/artifacts/gguf_reference_snapshot_from_sidecar_64
```

To compare the running GGUF backend against that reference trace:

```bash
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
.venv-ui/bin/python tools/compare_gguf_to_reference_trace.py \
  --output .openresearch/artifacts/gguf_ui/trace_compare_reasoning_off.jsonl
```

With `LLAMA_REASONING=off`, the comparison harness can log either base-GGUF or
RWKV-MS-sidecar runs. Stock llama.cpp still does not execute RWKV-MS memory; the
sidecar mode requires the local patched branch.

## Acknowledgement

This work builds on the Log-Linear Attention repository and uses local
HRM-Text/RWKV memory ideas as mechanism baselines. The added experiments are
intended for controlled research exploration, not as a trained-model benchmark.
