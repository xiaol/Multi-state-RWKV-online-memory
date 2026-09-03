# Findings: does a delta-rule / RWKV-style memory work on a frozen base?

Date: 2026-09-04. Numbers are exact match (EM) on 256 held-out synthetic passages with 8 facts about
invented people (fresh names and values, disjoint name pools from training), question asked with the
passage **absent** from the context. `correct` / `donor` = state written from this row's / another
row's passage. Full tables: `RESULTS.md` (regenerate with `python report.py`).

## Short answer

Yes, with two conditions that the rest of this repository never met.

1. The read context must not contain the written text. On the native benchmark rows the passage is
   in the read prompt, so the frozen model's own attention already has everything and the optimal
   adapter ignores the state. Here the passage is removed before the read, and the donor control
   separates as soon as the adapter has trained for more than about a thousand updates.
2. The state has to be read through an interface the frozen model already consumes: extra key/value
   slots inside attention, addressed by the frozen query. Residual, gate and PLE injections were
   never tried here, but the KV-slot interface works on both Qwen3 and Gemma4 without touching any
   base weight.

## What worked, in order of effect size (frozen Qwen3-1.7B, 3000 updates, batch 16)

| variant | correct EM | donor EM | note |
|---|---:|---:|---|
| frozen base, passage in context (upper bound) | 1.000 | | |
| frozen base, question only (lower bound) | 0.031 | | |
| delta state, static 16-slot query bank, raw residual write | 0.438 | 0.094 | first working config |
| delta state, query-conditioned read, raw residual write | 0.551 | 0.066 | frozen query addresses the state |
| delta state, query-conditioned read, **normalized attention input** write | **0.586** | 0.074 | best delta config |
| uncompressed KV bank (softmax over passage tokens), same read | 0.602 | 0.074 | compression costs almost nothing |
| delta state, cosine-routed 4 states x 4 slots | 0.449 | 0.086 | multi-state, matched 16 slots |
| delta state, chunk-routed 4 states x 4 slots | 0.297 | 0.082 | fixed-block routing is worst |
| delta state, trained on 2-fact passages, tested on 8 facts | 0.418 | 0.074 | capacity transfers |

Learning is slow to start: at 500 updates every variant has learned only the answer *format*
(CE drops from 4.8 to 1.2 with correct **and** donor states, generations are right-type
wrong-instance such as "July" for "October"). Binding appears between 1000 and 2000 updates. That
early phase is exactly the signature the native experiments in this repository reported and then
chased with new readouts; here it resolves by simply training longer on a task where the state matters.

## Gemma4 E4B

Gemma is harder. With the raw residual as write source the query-conditioned read never separated
(0.074 vs 0.051 at 3000 updates) and the static bank only weakly (0.180 vs 0.113). The cause is in
the write features: one residual dimension carries about 80 percent of the layer-input energy (dim
611, RMS 89 versus a median of 0.7), so linear key/value projections of the raw residual see almost
the same vector for every token. Writing from the post-layernorm attention input (the tensor the
frozen attention itself projects) fixes it (frozen Gemma4 E4B, 3000 updates):

| variant (write from attention input) | correct EM | donor EM |
|---|---:|---:|
| delta state, query read, 7 full-attention layers (5,11,...,41) | 0.230 | 0.066 |
| uncompressed KV bank, query read, same 7 layers | 0.293 | 0.082 |
| uncompressed KV bank, query read, 8 sliding-window layers (4,8,...,32) | 0.508 | 0.059 |

Layer choice matters on Gemma: the sliding-window layers bind much faster than the full-attention
layers (head dim 512, partial rotary), which is where a delta-state run should go next.

## What did not work

* SQuAD (natural passages, frozen Qwen3-1.7B, 3000 updates at batch 8): no separation (0.086 vs
  0.074 EM; in-context upper bound 0.523). Natural text needs many more bindings per passage and
  the answer is a span rather than a value; this needs a bigger training budget and probably the
  attention-input write source, which this run predates.
* Multi-state routing did not beat a single state at matched slot count. Cosine routing was on par,
  chunk routing was clearly worse.
* Model size did not matter much: frozen Qwen3-4B reached 0.492 vs 0.051 at 3000 updates, close to
  1.7B (0.551) with the same configuration.
* Training longer did not help: the 10k-update run plateaued at 0.586 vs 0.070 from about 3000
  updates on. The remaining gap to the in-context upper bound (1.000) is not a matter of steps; it
  needs a better write (nonlinear keys, more memory dimensions) or a sharper read.

## What this says about the rest of the repository

* The 17 retired readout families were not readout failures. Their donor margins were zero because
  the benchmark rows put the written text in the read context. No readout can pass that gate.
* The tau2 result (14/20) still has no state-zeroed control and should not be cited as evidence for
  the recurrent path. That control could not be run here because the tau2 harness and data live on
  the other workstation.
* If the goal is online memory, the next step is to port this protocol to the native tasks: write
  the passage or prior context, clear it, and ask. The projected-slot adapter and the RWKV-MS state
  can then be compared on equal terms.
