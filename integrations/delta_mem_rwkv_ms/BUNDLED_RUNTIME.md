# Bundled Delta-Mem Runtime

The top-level `deltamem/` package is a source snapshot derived from
`declare-lab/delta-Mem` commit `5cd5d9153c7f408764728d953565201e198c39e2`,
plus the Qwen3.6 and RWKV-MS compatibility work recorded locally as delta-Mem
commit `f358cc8`.

It is bundled here so Qwen3.6 online-memory inference, training, serialization,
and tests do not depend on a second Git checkout or write access to the upstream
repository. `delta_mem_rwkv_ms.patch` remains as an optional export for users who
want to apply the same changes to the pinned upstream revision.
