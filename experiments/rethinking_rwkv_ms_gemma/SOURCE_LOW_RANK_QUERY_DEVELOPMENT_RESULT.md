# Source-Bound Low-Rank Query Development Result

The locked open development run for the address/query-conditioned low-rank RWKV read completed on four distinct A100 GPUs with `HF_ENDPOINT=https://hf-mirror.com`.

- Protocol: `natural_memory_native_rwkv_low_rank_query_development_protocol_v1.json`
- Open materialization: `natural_memory_native_rwkv_low_rank_query_development_v1`
- Updates: 48; train rows: 48; held-out rows: 32
- Trainable route: six tensors, 198,400 elements; frozen base model and maps
- Protected mechanics rows opened: 0
- Protected causal rows opened: 0
- Native benchmark rows opened: 0

## Decision

`open_heldout_failed_not_promoted`

Prompt/source/confidence latching, exact-zero controls, finite residuals, target selection, and all staged gradient contracts passed. The ordinary held-out aggregate failed: correct-vs-provider-off mean `-0.06685`, donor-both-vs-target mean `-0.02520`, and layer-both-vs-target mean `-0.03970`. The discriminative held-out view improved to correct-vs-off mean `0.09042`, donor-both-vs-target mean `0.02893`, and layer-both-vs-target mean `0.12474`, but donor-positive-row fraction was only `0.59375` against the locked `0.75` gate.

This exact low-rank-query family is not promoted or replicated. The result does not authorize protected mechanics, protected causal evaluation, native benchmark access, or an SOTA claim. Full-Bandwidth Transformer remains a separate decode-time latent-feedback method and is not reproduced by this route.
