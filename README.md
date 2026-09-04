# Multi-State RWKV Online Memory

**Goal.** Train a small recurrent memory (gated delta rule, RWKV-7 style, one or more matrix
states per layer) that a **frozen** LLM can write to and read from, so the model can answer from
text that is no longer in its context. The base model (Gemma4 E4B, Qwen3) is never modified; only
the memory is trained. Task routers, rule planners, and benchmark-specific decoders do not count
as progress. Only the trained memory does.

HF checkpoint of the earlier tau2 adapter:
[`xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1`](https://huggingface.co/xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1)
(see "What we tried before" for why it is not yet evidence of memory).

## Result (2026-09-04)

Write a passage into the memory, delete the passage from the context, ask a question about it.
Exact match of the greedy answer on 256 held-out passages with 8 facts about invented people,
after 3000 adapter updates at batch 16. Full tables: [RESULTS.md](experiments/write_clear_read/RESULTS.md).

| frozen base | memory | correct | donor | no memory | in context |
|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | delta state, query-conditioned read, attention-input write | **0.586** | 0.074 | 0.031 | 1.000 |
| Qwen3-1.7B | uncompressed KV bank, same read (diagnostic) | 0.602 | 0.074 | 0.031 | 1.000 |
| Qwen3-1.7B | delta state, static 16-slot bank read | 0.438 | 0.094 | 0.031 | 1.000 |
| Qwen3-1.7B | delta state, cosine-routed 4 states x 4 slots | 0.449 | 0.086 | 0.031 | 1.000 |
| Qwen3-4B | delta state, query-conditioned read | 0.492 | 0.051 | 0.047 | 0.996 |
| Gemma4 E4B | delta state, query-conditioned read, attention-input write, 7 full-attention layers | **0.230** | 0.066 | 0.020 | 1.000 |
| Gemma4 E4B | uncompressed KV bank, 8 sliding-window layers (diagnostic) | 0.508 | 0.059 | 0.020 | 1.000 |

How to read a row:

* **correct**: the prompt holds only the question; the memory holds this row's passage. This is the memory.
* **donor**: the prompt holds only the question; the memory holds a *different* row's passage (same
  task and format, different names and values). The control. `correct - donor` is the content the
  state actually carries; whatever both share is answer format the adapter learned.
* **no memory**: question only, nothing else. Chance level for the frozen base.
* **in context**: passage and question both in the prompt, no adapter at all. The frozen model
  reading the passage with its own attention. This is the ceiling, not a competitor: memory is for
  the case where the passage cannot stay in the context (beyond the window, earlier session, cleared history).

A working memory shows `no memory <= donor < correct <= in context`.

## Key findings

1. **The memory only shows up when the text is gone from the context.** Every earlier benchmark in
   this repository wrote the memory from the prompt and then read with that same prompt still in the
   context (`run_natural_memory_native_evolution.py:315-348`). The frozen model's attention already
   had everything, so the optimal adapter ignored the state, and 17 readout variants all reported a
   zero donor margin. That was the benchmark, not the readout. With the passage removed, the same
   kind of state separates from the donor control on both Qwen3 and Gemma4.
2. **Read the state through the interface the frozen model already uses: attention key/value
   slots.** Each wrapped layer's state is turned into extra K/V entries that the frozen attention
   attends to next to the real tokens. No base weight, projection, rotary, KV-cache or KV-sharing
   code is changed; with no slots the hook reproduces stock attention exactly. Residual, gate and
   per-layer-embedding injections are off-manifold inputs and were the families that failed before.
3. **Let the frozen query address the state.** A static learned query bank (16 fixed reads per
   layer) works but is weaker (0.438). Mapping each frozen head's query into the memory key space and
   reading the state with it (linear-attention style, one slot per query position) gives 0.551 to 0.586.
4. **Write from the post-layernorm attention input, not the raw residual.** In both models one
   residual dimension carries 70 to 80 percent of the layer-input energy (Gemma dim 611 has RMS 89
   against a median of 0.7). Linear key/value projections of the raw residual see nearly the same
   vector for every token. On Gemma nothing bound from the residual (0.074 vs 0.051); from the
   attention input the delta state reaches 0.230 and the KV bank 0.293 on the same layers.
5. **Compression is nearly free.** The delta-rule matrix state trails an uncompressed softmax KV
   bank over all passage tokens by less than 0.02 on Qwen. The remaining gap to the in-context
   ceiling is in the write and read projections, not in the recurrence.
6. **Binding takes 1000 to 2000 updates and looks like failure before that.** For the first 500 to
   1000 updates every variant learns only the answer format: CE drops from 4.8 to 1.2 with correct
   *and* donor states, and answers are right type, wrong instance ("July" for October). This is the
   exact signature the earlier experiments saw at 32 updates and then chased with new readouts.
7. **Multi-state did not beat one state at matched slots.** Cosine routing (4 states x 4 slots) was
   on par with a single state (0.449 vs 0.438 with the bank read); fixed chunk routing was clearly
   worse (0.297). Model size did not matter much (Qwen3-4B 0.492), and 10k updates plateau at 0.586.
8. **Layer choice matters on Gemma.** Its sliding-window layers bind much better than its
   full-attention layers (head dim 512, partial rotary): KV bank 0.508 vs 0.293. A delta-state run on
   the sliding layers is the obvious next experiment.

Not yet working: natural text. SQuAD on frozen Qwen3-1.7B (3000 updates, batch 8, residual write)
shows no separation (0.086 vs 0.074; in-context ceiling 0.523). It needs the attention-input write,
more updates and probably a larger memory dimension.

## The mechanism

```
write   passage --> frozen model (one pass) --> at each wrapped layer, the attention input
        h_t is projected to k_t, v_t, alpha_t, beta_t and folded into the matrix state
        S <- alpha (S - beta k k^T S) + beta k v^T            (gated delta rule, fp32)
clear   the passage is dropped; only S survives
read    question --> frozen model; in each wrapped layer the frozen query q (after q_norm,
        before rotary) addresses the state: r = S^T W_q q, then r is mapped into that layer's
        key and value space and appended as extra K/V slots that the frozen attention
        attends to together with the real tokens
```

Multi-state: tokens are routed to one of S states per head (contiguous chunks, or cosine to learned
anchors), and every state is read. Trained parameters: 48M on Gemma4 E4B (7 layers), 55M on
Qwen3-1.7B (6 layers). Loss: next-token cross-entropy on the answer, nothing else. No donor or
zero-state contrast terms are needed once the passage is out of the read context.

Code: [`experiments/write_clear_read/memkv.py`](experiments/write_clear_read/memkv.py) (adapter and
attention hook), [`train.py`](experiments/write_clear_read/train.py) (trainer with the four
controls), [`FINDINGS.md`](experiments/write_clear_read/FINDINGS.md) (discussion),
[`README.md`](experiments/write_clear_read/README.md) (metric definitions).

## Run it

```bash
pip install -r requirements.txt && pip install -e .
cd experiments/write_clear_read
./run_qwen.sh                       # frozen Qwen3-1.7B, single-state vs multi-state
./run_gemma.sh                      # frozen Gemma4 E4B
python train.py --model /path/to/Qwen3-1.7B --out runs/my_run \
  --facts 8 --entities 4 --eval-facts 4,16 --layers auto \
  --read-mode query --write-source attn_input --steps 3000 --batch-size 16
python report.py                    # regenerate RESULTS.md from runs/*/log.jsonl
```

Useful flags: `--n-states S --slots-per-state M --routing {single,chunk,cosine}` (multi-state),
`--memory kvbank` (uncompressed diagnostic), `--dataset squad --squad-root /path/to/parquet`,
`--equivalence-check` (asserts the hook equals stock attention with no slots). Runs survive other
jobs' GPU memory spikes by retrying on out-of-memory.

## What we tried before, and why it is not memory evidence yet

The full chronological record is in [HISTORY.md](HISTORY.md). In short:

* **Native Gemma4 tasks (attribution, narrative, scene).** A projected-slot adapter passed a
  preregistered validation gate (scene F1 +0.09), but its readout bypasses the recurrent scan, and
  every recurrent variant showed a zero donor margin because the read prompt contained the written
  text (finding 1). Those runs also trained for only 32 updates (finding 6).
* **tau2 telecom (14/20 vs 4/20 base).** The adapter there is 0.8M trainable q/o parameters
  trained by SFT; no run has evaluated it with the recurrent state zeroed, so the gain is not yet
  attributable to memory. Recipe: [GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md](GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md).
* **Mechanism studies that do hold.** The DLA adaptive-state reproduction ([EVAL.md](EVAL.md),
  `dla_poc.py`): multi-state RWKV-7 recall 1.0 on synthetic associative recall, adaptive merging
  beats fixed blocks at matched state count. HOLA hippocampus cache on RWKV-7 multi-state
  (`experiments/hola_hippocampus/`). MARCH-style anchors ([docs/MARCH_RWKV_MS_COMPARISON.md](docs/MARCH_RWKV_MS_COMPARISON.md)).
* **Runtime.** The bundled `deltamem/` package and `integrations/delta_mem_rwkv_ms/` provide the
  RWKV-MS-capable HF runtime, tau2 inference, and the Gemma4 GGUF sidecar work
  ([GGUF_EXTERNAL_MEMORY_FEASIBILITY.md](GGUF_EXTERNAL_MEMORY_FEASIBILITY.md)).

## Next

1. Delta state on Gemma's sliding-window layers with the attention-input write.
2. Close the gap to the in-context ceiling: nonlinear write keys, larger memory dimension, sharper read.
3. Natural text: SQuAD with the attention-input write and a larger budget; then multi-turn and
   cross-session memory, where keeping the text in context is not an option.
4. Port the write/clear/read protocol to the native Gemma tasks and rerun tau2 with the state zeroed.

## Repository layout

```text
experiments/write_clear_read/      # the current experiment: adapter, trainer, runs, findings
experiments/rethinking_rwkv_ms_gemma/  # earlier native-task experiments and signed artifacts
experiments/hola_hippocampus/      # HOLA cache on RWKV-7 multi-state
deltamem/                          # bundled HF online-memory runtime (RWKV-MS backend)
integrations/delta_mem_rwkv_ms/    # tau2 inference/training entry points, GGUF tools, upstream patch
integrations/ruler/                # RULER data and scoring helpers
dla_poc.py, EVAL.md, hattention/   # DLA mechanism reproduction
HISTORY.md                         # the previous README, kept verbatim
```

## Acknowledgement

This work builds on the Log-Linear Attention repository and uses HRM-Text/RWKV memory ideas as
mechanism baselines. The frozen bases are Google's Gemma4 E4B and Alibaba's Qwen3.
