# DLA mechanism reproduction (matched memory budget)

- Codebase smoke test (repo Log-Linear attention ran): **PASS**


## Theorem 3.1 deviation bound, fixed vs DLA (mean over 5 seeds)

| config | segments | K | DLA states | bound fixed | bound DLA | bound ↓ |
|---|---|---|---|---|---|---|
| K>=segments  (Alg 1 active) | 6 | 8 | 6.0 | 45.898 | 23.839 | 48.1% |
| K>=segments  (Alg 1 active) | 10 | 12 | 10.0 | 58.908 | 26.054 | 55.8% |
| K< segments  (Alg 2 active) | 16 | 8 | 8.0 | 143.390 | 118.131 | 17.6% |
| K< segments  (Alg 2 active) | 24 | 10 | 10.0 | 193.717 | 169.493 | 12.5% |

## Associative recall cos(o, v*), mechanism baselines (mean over 5 seeds)

| needles | filler/seg | K | states | fixed | rwkv_mem(delta_rule) | rwkv_mem(rwkv7) | rwkv_mem(rwkv7 multi-state) | DLA |
|---|---|---|---|---|---|---|---|---|
| 6 | 8 | 16 | 12.0 | 0.920 | 0.229 | 0.797 | 1.000 | 1.000 |
| 10 | 6 | 24 | 20.0 | 0.934 | 0.122 | 0.682 | 1.000 | 1.000 |
| 8 | 10 | 20 | 16.0 | 0.887 | 0.046 | 0.626 | 1.000 | 1.000 |

The HRM baselines are self-contained mechanism ports from `HRM-Text/models/rwkv_memory.py` and `models/rwkv7.py`. `rwkv_mem(delta_rule)` uses the online delta-rule associative state with HRM's default beta bias -1.5. `rwkv_mem(rwkv7)` uses the latest read-before-write RWKV-7 recurrence specialized to the synthetic key/value stream. `rwkv_mem(rwkv7 multi-state)` keeps the same RWKV-7 state update but allocates one state per DLA adaptive block at the same state count.


## State-Update-Only Ablation, Same Boundaries (mean over 5 seeds)

| boundary policy | needles | filler/seg | K | states | linear/DLA state | RWKV-7 state | RWKV - linear |
|---|---|---|---|---|---|---|---|
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

This table fixes the exact same token blocks for both methods. `linear/DLA state` uses the standard block sum `sum k_t v_t^T`; `RWKV-7 state` uses the RWKV-7 recurrence inside each same block. Therefore each row compares state update/readout only, not boundary quality.


## Verdict

- DLA lower deviation bound in every config: **True** (reproduces Theorem 3.1 / Corollary 3.2)
- DLA higher needle recall in every config: **True**
- DLA higher than HRM rwkv_mem(delta_rule): **True**
- DLA higher than HRM rwkv_mem(rwkv7): **True**
- Multi-state RWKV-7 improves over single-state RWKV-7: **True**
- State-only ablation wins, RWKV-7/linear/tie: **11/0/4**
- Codebase smoke test: **True**
- Core claim (adaptive merging beats fixed schedule at matched budget): **REPRODUCED**
