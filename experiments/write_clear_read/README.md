# Write / clear / read: delta-rule multi-state memory through KV slots on a frozen base

This directory holds the deciding experiment for the question "can a delta-rule / RWKV-style
recurrent memory add anything to a frozen transformer?" It replaces the native benchmark
rows used elsewhere in this repository, whose read pass already contains the written text
through attention (see `run_natural_memory_native_evolution.py:315-348`), with a task in
which the state is the only carrier.

## Protocol

1. **Write.** The passage is run through the frozen model once. At each wrapped layer the
   residual-stream input is projected into per-head keys, values, a decay `alpha` and a write
   strength `beta`, and folded into one or more `mem_dim x mem_dim` matrix states with the
   gated delta rule `S <- alpha (S - beta k k^T S) + beta k v^T`.
2. **Clear.** The passage is discarded. Nothing but the states survives.
3. **Read.** The question is asked alone. Each state is queried with a learned query bank and
   every retrieved vector is mapped into the frozen layer's key and value space, then appended
   as extra key/value slots that the frozen attention attends to next to the real tokens. Slot
   keys are scored against the unrotated query (captured after `q_norm`), which equals a rotary
   relative distance of zero, so scores do not depend on the query position.

Only the adapter (`LayerMemory` in `memkv.py`) is trained, with next-token loss on the answer.
The base model's projections, norms, rotary, KV cache and KV sharing are untouched; the only
hook is a registered attention function (`memkv`) that concatenates the slots. With no slots
it reproduces stock eager attention exactly (`--equivalence-check`, max logit diff 0.0 on
both Qwen3 and Gemma4).

## Controls

| Condition | What the read pass sees |
| --- | --- |
| `memory_correct` | question + slots from this row's passage |
| `memory_donor` | question + slots from another row's passage |
| `memory_zero` | question only (no slots) |
| `base_in_context` | passage + question in the prompt, no slots (upper bound) |

Because the passage is never in the read context, `memory_correct` above `memory_donor` is
only possible if the state carries row-specific content. On the native rows in the rest of
this repository that separation cannot occur by construction.

## Reading the results

Every evaluation row in `RESULTS.md` and `log.jsonl` reports the same frozen model under four
read conditions. The adapter is the same trained object in the three memory conditions; the
fourth uses no adapter at all.

| column | what the read pass sees | what it measures |
| --- | --- | --- |
| `correct` (`memory_correct`) | question only + memory slots computed from the state written from **this row's** passage | the actual memory: can the state carry the fact into the answer? |
| `donor` (`memory_donor`) | question only + slots from the state written from **another row's** passage (the previous row in the eval batch, same task and format, different names and values) | the control: anything `correct` gains over `donor` is row-specific content; anything they share is just format or prior learned by the adapter |
| `zero` (`memory_zero`) | question only, no slots | the frozen base with no information at all (chance level for the task) |
| `in context` (`base_in_context`) | passage **and** question in the prompt, no slots, frozen base only | the upper bound: what the frozen model does when it can read the passage with its own attention |

So the ordering to look for is `zero` <= `donor` < `correct` <= `in context`. A memory that only
teaches the answer format shows `correct == donor` far above `zero`; a memory that stores content
shows `correct` clearly above `donor`.

Metrics per condition:

* **EM** (exact match): the model generates greedily (up to `--max-new-tokens`), the first line of
  the output is lower-cased, punctuation and articles are stripped, and it must equal the gold
  value (any of the gold aliases for SQuAD). Fraction of rows, 0 to 1. This is the primary metric.
* **CE** (answer cross-entropy): teacher-forced mean per-token negative log-likelihood of the gold
  answer tokens (including the end-of-turn token) given the prompt; lower is better. Comparable
  across the three memory conditions because they share the trained adapter. It is **not** directly
  comparable between the memory conditions and `in context`: the trained adapter has learned the
  answer style, so its CE can be lower than the frozen base's even when its EM is far worse (see the
  SQuAD row, where `in context` has CE 3.40 but EM 0.52).
* **F1** (in `log.jsonl` only): SQuAD-style token F1 between prediction and gold.
* **mem mass**: mean attention probability that the wrapped layers put on the memory slots during
  the `correct` read pass, averaged over heads, query positions and layers. Diagnostic only; it
  shows the frozen model is using the slots, not that the slots are useful.
* **step**: optimizer updates completed when the evaluation ran. **eval set**: `synthetic_kN`
  means N facts per passage (training used K=8 unless noted), `squad_val` is SQuAD v1.1 validation.

## Multi-state

`--n-states S --slots-per-state M --routing {single,chunk,cosine}` keeps the total slot count
`S*M` fixed across variants. `chunk` routes contiguous passage segments to different states
(the fixed-block baseline of the DLA study), `cosine` routes each token to the state whose
learned anchor is closest to its key (soft, temperature `--route-temperature`).

## Data

* `synthetic`: K facts about invented people with distinct entity/attribute pairs, fresh
  entities and values per example, disjoint first-name pools for train and eval. Also
  evaluated at other K (`--eval-facts`) for capacity generalisation.
* `squad`: SQuAD v1.1 context / question / answer span (`--squad-root`, parquet files).

## Running

```bash
./run_qwen.sh    # frozen Qwen3-1.7B: single-state vs chunk vs cosine at 16 slots per layer
./run_gemma.sh   # frozen Gemma4 E4B: memory on the seven full-attention layers
```

Each run writes `runs/<name>/log.jsonl`, `result.json` (config, per-eval metrics for every
condition, attention mass on memory slots per layer, sample generations) and optionally
`adapter.pt`.
