# Weighted Renewal Route Ablation

This is an open-development route ablation inspired by the multi-depth renewal
hypothesis in Full-Bandwidth Transformer (arXiv `2608.08888`). It is not a
reproduction of that paper and does not open protected mechanics, protected
causal data, or the native benchmark.

## Frozen setup

- The RWKV value bundle, adapter, writer, compatibility maps, prompt latch, and
  training schedule are unchanged from the prior multi-anchor gate.
- The router aggregates source scores from native RWKV anchors `(5, 11, 17)`.
- Weighted run uses normalized route weights `(0.25, 0.25, 0.5)`.
- Equal control uses `(1/3, 1/3, 1/3)` on the same fresh rows.
- Both runs use four distinct A100s and `HF_ENDPOINT=https://hf-mirror.com`.

## Fresh split

The split contains 40 reciprocal pairs (48 train rows, 32 held-out rows). It
excludes 462 passage components: historical 94, parent 160, protected 64,
prior cumulative development 64, and prior multi-anchor development 80. The
remaining component count before selection is 246, with zero overlap.

## Results

Both runs pass preflight, exact-zero/provider-off controls, prompt source and
confidence latching, finite residual checks, and the terminal-selection gate.
Neither passes the causal held-out gate:

| route | original-view gain vs off | donor-both margin | layer margin | decision |
| --- | ---: | ---: | ---: | --- |
| weighted `(0.25, 0.25, 0.5)` | `-0.05739` | `-0.03406` | `-0.02075` | failed, not promoted |
| equal `(1/3, 1/3, 1/3)` | `-0.03950` | `+0.01609` | `-0.00977` | failed, not promoted |

Equal weighting is directionally better, but remains below the locked donor
margin (`0.02`) and correct-gain requirements. The weighted route is therefore
retired as a candidate architecture; no protected or native evaluation is
authorized. The next useful experiment must change the value/write mechanism,
not tune these route weights on the same data.
