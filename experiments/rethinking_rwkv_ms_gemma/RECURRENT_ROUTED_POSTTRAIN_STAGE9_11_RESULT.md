# Recurrent-Routed Post-Training: Stages 9–11

The active goal remains model-only post-training. No task router, template matcher, dual-pass selector, or benchmark-specific decoder was added.

## Development-v2 split

- Manifest receipt: `2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6`
- Rows: attribution 8, narrative 32, scene 32; all four locked prompt variants.
- Final rows remain unopened.
- The split is disjoint from the stage-2 predecessor target schedule. It should not be described as independent of every discarded continuation in the historical workspace.

## Candidates

| Candidate | Model change | Training receipt | Development receipt | Result |
|---|---|---|---|---|
| Stage 9 | Always-on generic donor-state contrast; donor weight 1.5 | `160b0abe0b5ce79d56c9df7d833e189a736616633582841179fc15b8c08addba` | `1a3e1208cda389650be40a6634b0515981e904b0084c28c887c53d0418677805` | blocked |
| Stage 10 | Narrative-heavy target schedule; always-on donor contrast | `dac5ad358fbfb2a78de274ce76a930409e243dd4061f556242d9d76afb4e30e2` | `202ba6ff015fce526d7cec4d515756370925be844f19be42bf90c2afdc6bc67b` | blocked |
| Stage 11 | Stage 10 objective with recurrent hybrid gain 0.25 | `849ee91a0c5c548fbfd8a095dba29cff8ea045e92e7d2c5d4d8eb64c9ce8b34d` | `b52dcb6f86db3527731a1939ee1da18410ac0bd65c7e04609022466213fdbf6f` | blocked |

All three candidates passed distributed training audits and fixed projected-carrier audits. The stage-2 checkpoint itself was also evaluated on v2 as a reference.

## Gate diagnosis

The overall causal gate passes for every v2 evaluation, but the per-task and all-prompt-variant gates fail on narrative donor-state separation:

- Stage 2 reference: `-0.00038088` mean donor-minus-correct CE.
- Stage 9: `-0.00110483`.
- Stage 10: `-0.00034924`.
- Stage 11: `-0.00087911`.

Attribution and scene margins stay positive. Since the strict development gate fails, no final rows were opened and no final benchmark improvement is claimed.

## Next experiment

Diagnose narrative recurrent-state identifiability at the state/readout objective level with a pre-registered objective and a truly unused holdout, or replenish the open split before claiming a fresh final benchmark. Do not promote any stage-9–11 adapter to final evaluation.
