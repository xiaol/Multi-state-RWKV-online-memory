"""HOLA-on-RWKV: surprise-evicted exact-KV hippocampus over RWKV-7 multi-state.

Replaces HOLA's (arXiv 2607.02303) GDN neocortex with this repo's read-before-write
RWKV-7 multi-state online memory and re-tests HOLA's two design claims at mechanism
level, on the same state-only ablation grid as dla_poc.py:

  H1 what to store: top-w by the actual RWKV-7 write magnitude m_t = ||Delta_t||_F
     (= ||v - g*S k|| for unit keys; erase_gate g plays HOLA's beta) beats a matched
     recency cache under imperfect boundaries.
  H2 how to read: the cache needs a sharpened read (sqrt(d)*cos logits); HOLA's
     measured naive scale (0.83*cos) degenerates toward no cache.
  H3 the hippocampus recovers most of the low-K state-compression gap.

Usage:
  .venv/bin/python experiments/hola_hippocampus/hola_rwkv_ms.py --smoke   # parity + quick pass
  .venv/bin/python experiments/hola_hippocampus/hola_rwkv_ms.py          # full grid + REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dla_poc as poc  # noqa: E402

HERE = Path(__file__).resolve().parent
DECAY = 0.98
ERASE_GATE = 1.0
SHARP = None  # set to sqrt(d) at runtime
FLAT = 0.83   # HOLA's measured naive cache logit scale (tau/sqrt(d) * cos ~ 0.83 cos)


def rms_unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm() + eps)


def build_states_and_scores(Kt, Vt, blocks, decay=DECAY, erase_gate=ERASE_GATE,
                            kind="rwkv7"):
    """One online pass in token order, per-block states, plus per-token surprise
    score m_t = ||actual write||_F measured read-before-write against the token's
    own block state.

    kind:
      rwkv7  — the repo recurrence (identical to poc.hrm_rwkv7_state).
      gdn    — HOLA's actual backbone: e = v - alpha*S k; S = alpha*S + beta*e k^T
               (alpha=decay, beta=erase_gate). For unit keys this differs from
               rwkv7 only in the erase term's alpha scaling.
      linear — plain block sum S += v k^T (the repo's linear/DLA state). Its write
               norm is ~||v||~1 for every token: no delta rule, no surprise signal.
    """
    T, d = Kt.shape
    tok2blk = {}
    for bi, b in enumerate(blocks):
        for t in b:
            tok2blk[t] = bi
    states = [torch.zeros(Vt.shape[1], d, dtype=torch.float32) for _ in blocks]
    scores = torch.zeros(T)
    for t in range(T):
        k_t = Kt[t].float()
        v_t = Vt[t].float()
        kk = k_t / (k_t.norm() + 1e-8)
        S = states[tok2blk[t]]
        if kind == "rwkv7":
            correction_read = S @ (-kk)
            write = torch.outer(v_t, k_t) + torch.outer(correction_read, erase_gate * kk)
            new_S = decay * S + write
        elif kind == "gdn":
            e_t = v_t - decay * (S @ k_t)
            write = erase_gate * torch.outer(e_t, k_t)
            new_S = decay * S + write
        elif kind == "linear":
            write = torch.outer(v_t, k_t)
            new_S = S + write
        else:
            raise ValueError(f"unknown state kind: {kind}")
        scores[t] = write.norm()
        states[tok2blk[t]] = new_S
    return states, scores, tok2blk


def cache_indices(scores: torch.Tensor, w: int, eviction: str) -> list[int]:
    T = scores.shape[0]
    w = min(w, T)
    if eviction == "surprise":
        return sorted(torch.topk(scores, w).indices.tolist())
    if eviction == "recency":
        return list(range(T - w, T))
    raise ValueError(f"unknown eviction: {eviction}")


CONSOLIDATE_THRESH = 0.5  # m_t below this = state predicted the token well
CONSOLIDATE_COS = 0.9     # key-direction match for "same association"
CONSOLIDATE_DECAY = 0.5   # demotion factor per consolidation event


def consolidation_cache(Kt, scores, w):
    """Online top-w cache with CLS-style systems consolidation.

    Admission score is the write magnitude m_t (HOLA's beta*||e|| analog). The one
    addition: when a later token with the SAME key direction is predicted well by
    the state (small m), matching cache entries are demoted — the neocortex has
    absorbed that association, so the hippocampus releases it. Needles never repeat,
    so they are never demoted. O(w) per token, no learning, order-online.
    """
    T = Kt.shape[0]
    w = min(w, T)
    idx: list[int] = []
    score: dict[int, float] = {}
    for t in range(T):
        k_t = Kt[t].float()
        k_t = k_t / (k_t.norm() + 1e-8)
        m_t = float(scores[t])
        if m_t < CONSOLIDATE_THRESH and idx:
            for e in idx:
                k_e = Kt[e].float()
                k_e = k_e / (k_e.norm() + 1e-8)
                if abs(float(k_e @ k_t)) > CONSOLIDATE_COS:
                    score[e] *= CONSOLIDATE_DECAY
        if len(idx) < w:
            idx.append(t)
            score[t] = m_t
        else:
            weakest = min(idx, key=lambda e: score[e])
            if m_t > score[weakest]:
                idx.remove(weakest)
                score.pop(weakest)
                idx.append(t)
                score[t] = m_t
    return sorted(idx)


def cache_read(q, Kt, Vt, idx, scale):
    """Returns (read vector, confidence gate lambda).

    lambda = (p1 - 1/w) / (1 - 1/w): the untrained analog of HOLA's learned read
    gate — 0 when the softmax is uniform (nothing in cache matches the query),
    ~1 when one entry takes almost all mass (a sharp retrieval hit).
    """
    Kc = Kt[idx].float()
    Vc = Vt[idx].float()
    logits = scale * (Kc @ q.float())
    weights = torch.softmax(logits, dim=0)
    n = len(idx)
    p1 = float(weights.max())
    lam = max(0.0, (p1 - 1.0 / n) / (1.0 - 1.0 / n)) if n > 1 else p1
    return weights @ Vc, lam


def needle_cosines(outs, needle_val):
    cos = []
    for o, v in zip(outs, needle_val.float()):
        cos.append(float((o @ v) / ((o.norm() * v.norm()) + 1e-8)))
    return float(np.mean(cos))


def run_variants(Kt, Vt, blocks, npos, nval, w):
    d = Kt.shape[1]
    sharp = float(np.sqrt(d))
    T = Kt.shape[0]

    states, scores, tok2blk = build_states_and_scores(Kt, Vt, blocks)
    single_states, single_scores, _ = build_states_and_scores(
        Kt, Vt, [list(range(T))]
    )
    idx_hola = consolidation_cache(Kt, scores, w)
    idx_plain = cache_indices(scores, w, "surprise")
    idx_recency = cache_indices(scores, w, "recency")
    idx_single = consolidation_cache(Kt, single_scores, w)

    out = {}
    ms_reads = [states[tok2blk[p]] @ Kt[p].float() for p in npos]
    out["ms"] = needle_cosines(ms_reads, nval)

    def combined(state_reads, idx, scale):
        reads = []
        for p, s in zip(npos, state_reads):
            c, lam = cache_read(Kt[p], Kt, Vt, idx, scale)
            reads.append(rms_unit(s) + lam * rms_unit(c))
        return needle_cosines(reads, nval)

    out["ms+hola"] = combined(ms_reads, idx_hola, sharp)
    out["ms+hola-plain"] = combined(ms_reads, idx_plain, sharp)
    out["ms+recency"] = combined(ms_reads, idx_recency, sharp)
    out["ms+hola-flat"] = combined(ms_reads, idx_hola, FLAT)

    single_reads = [single_states[0] @ Kt[p].float() for p in npos]
    out["single"] = needle_cosines(single_reads, nval)
    out["single+hola"] = combined(single_reads, idx_single, sharp)

    # cache hygiene diagnostic: fraction of needles held verbatim in the cache
    out["_needle_in_cache"] = float(
        np.mean([1.0 if p in set(idx_hola) else 0.0 for p in npos])
    )
    return out


VARIANTS = ["ms", "ms+hola", "ms+hola-plain", "ms+recency", "ms+hola-flat",
            "single", "single+hola"]
BACKBONES = ["gdn", "rwkv7", "linear"]


def run_backbone_comparison(Kt, Vt, blocks, npos, nval, w):
    """HOLA vs HOLA-RWKV vs linear: same blocks, same cache machinery, only the
    neocortex recurrence (and hence its surprise signal) changes."""
    d = Kt.shape[1]
    sharp = float(np.sqrt(d))
    out = {}
    for kind in BACKBONES:
        states, scores, tok2blk = build_states_and_scores(Kt, Vt, blocks, kind=kind)
        idx = consolidation_cache(Kt, scores, w)
        state_reads = [states[tok2blk[p]] @ Kt[p].float() for p in npos]
        out[f"{kind}"] = needle_cosines(state_reads, nval)
        reads = []
        for p, s in zip(npos, state_reads):
            c, lam = cache_read(Kt[p], Kt, Vt, idx, sharp)
            reads.append(rms_unit(s) + lam * rms_unit(c))
        out[f"{kind}+hola"] = needle_cosines(reads, nval)
        out[f"_{kind}_needle_in_cache"] = float(
            np.mean([1.0 if p in set(idx) else 0.0 for p in npos])
        )
    return out


def run_comparison_grid(w, seeds, configs, policies):
    d, dv, tau = 32, 32, 0.6
    rows = []
    for nseg, nfill, K in configs:
        for policy in policies:
            acc = None
            nstates = []
            for seed in range(seeds):
                gen = torch.Generator().manual_seed(500 + 31 * seed + nseg)
                Kt, Vt, Smat, npos, nval = poc.make_recall_sequence(nseg, nfill, d, dv, gen)
                blocks = blocks_for(policy, Kt, Smat, npos, K, tau, gen)
                res = run_backbone_comparison(Kt, Vt, blocks, npos, nval, w)
                if acc is None:
                    acc = {k: [] for k in res}
                for k, v in res.items():
                    acc[k].append(v)
                nstates.append(len(blocks))
            row = dict(w=w, nseg=nseg, nfill=nfill, K=K, policy=policy,
                       states=float(np.mean(nstates)),
                       **{k: float(np.mean(vs)) for k, vs in acc.items()})
            rows.append(row)
            print(f"[cmp] w={w} needles={nseg:2d} {policy:9s} "
                  f"gdn={row['gdn']:.3f}/{row['gdn+hola']:.3f} "
                  f"rwkv7={row['rwkv7']:.3f}/{row['rwkv7+hola']:.3f} "
                  f"linear={row['linear']:.3f}/{row['linear+hola']:.3f}")
    return rows


def comparison_verdicts(rows):
    absdiff = float(np.mean([abs(r["gdn+hola"] - r["rwkv7+hola"]) for r in rows]))
    bad = [r for r in rows if r["policy"] in ("fixed", "noisy_dla", "low_k_dla")]
    gain_delta = float(np.mean(
        [r["gdn+hola"] - r["gdn"] for r in bad] + [r["rwkv7+hola"] - r["rwkv7"] for r in bad]
    ))
    gain_linear = float(np.mean([r["linear+hola"] - r["linear"] for r in bad]))
    return dict(
        h4_gdn_vs_rwkv_absdiff=absdiff,
        h4_pass=absdiff < 0.02,
        h5_cache_gain_delta_backbones=gain_delta,
        h5_cache_gain_linear=gain_linear,
        h5_pass=gain_linear < 0.5 * gain_delta,
    )
POLICIES = ["oracle", "dla", "fixed", "noisy_dla", "low_k_dla"]
CONFIGS = [(8, 12, 16), (12, 10, 16), (16, 8, 12)]


def blocks_for(policy, Kt, Smat, npos, K, tau, gen):
    T = Kt.shape[0]
    bnds, info = poc.dla_dynamic_merge(Smat, tau=tau)
    dla_blocks = poc.dla_capacity_bound(poc.boundaries_to_blocks(bnds, T), info, K)
    if policy == "oracle":
        return poc.oracle_needle_blocks(T, npos)
    if policy == "dla":
        return dla_blocks
    if policy == "fixed":
        return poc.fixed_blocks(T, len(dla_blocks))
    if policy == "noisy_dla":
        return poc.jitter_blocks(dla_blocks, T, max_shift=2, gen=gen)
    if policy == "low_k_dla":
        return poc.dla_capacity_bound(poc.boundaries_to_blocks(bnds, T), info, max(1, K // 2))
    raise ValueError(policy)


def run_grid(widths, seeds, configs, policies):
    d, dv, tau = 32, 32, 0.6
    rows, per_trial = [], []
    for w in widths:
        for nseg, nfill, K in configs:
            for policy in policies:
                acc = {v: [] for v in VARIANTS}
                acc["_needle_in_cache"] = []
                nstates = []
                for seed in range(seeds):
                    gen = torch.Generator().manual_seed(500 + 31 * seed + nseg)
                    Kt, Vt, Smat, npos, nval = poc.make_recall_sequence(nseg, nfill, d, dv, gen)
                    blocks = blocks_for(policy, Kt, Smat, npos, K, tau, gen)
                    res = run_variants(Kt, Vt, blocks, npos, nval, w)
                    for key in acc:
                        acc[key].append(res[key])
                    nstates.append(len(blocks))
                    per_trial.append(dict(w=w, nseg=nseg, nfill=nfill, K=K,
                                          policy=policy, seed=seed,
                                          states=len(blocks), **res))
                row = dict(w=w, nseg=nseg, nfill=nfill, K=K, policy=policy,
                           states=float(np.mean(nstates)),
                           **{k: float(np.mean(vs)) for k, vs in acc.items()})
                rows.append(row)
                print(f"w={w:2d} needles={nseg:2d} K={K:2d} {policy:9s} "
                      f"ms={row['ms']:.3f} +hola={row['ms+hola']:.3f} "
                      f"+plain={row['ms+hola-plain']:.3f} "
                      f"+rec={row['ms+recency']:.3f} +flat={row['ms+hola-flat']:.3f} "
                      f"single+hola={row['single+hola']:.3f} "
                      f"needle_in_cache={row['_needle_in_cache']:.2f}")
    return rows, per_trial


def verdicts(rows):
    bad = [r for r in rows if r["policy"] in ("fixed", "noisy_dla", "low_k_dla")]
    h1_gap = float(np.mean([r["ms+hola"] - r["ms+recency"] for r in bad]))
    h2_gap = float(np.mean([r["ms+hola"] - r["ms+hola-flat"] for r in bad]))
    flat_vs_ms = float(np.mean([r["ms+hola-flat"] - r["ms"] for r in bad]))
    rec = []
    for r in rows:
        if r["policy"] != "low_k_dla":
            continue
        oracle = next(x for x in rows if x["w"] == r["w"] and x["nseg"] == r["nseg"]
                      and x["policy"] == "oracle")
        gap = oracle["ms"] - r["ms"]
        if gap > 0.02:
            rec.append((r["ms+hola"] - r["ms"]) / gap)
    h3_frac = float(np.mean(rec)) if rec else float("nan")
    return dict(
        h1_gap_bad_boundaries=h1_gap,
        h1_pass=h1_gap > 0.03,
        h2_gap_sharp_vs_flat=h2_gap,
        h2_flat_minus_no_cache=flat_vs_ms,
        h2_pass=h2_gap > 0.03,
        h3_lowk_gap_recovered=h3_frac,
        h3_pass=(not np.isnan(h3_frac)) and h3_frac > 0.5,
    )


def write_report(rows, verd, widths, seeds, cmp_rows, cmp_verd):
    lines = [
        "# HOLA-on-RWKV mechanism results",
        "",
        "Neocortex = RWKV-7 multi-state (repo recurrence, decay 0.98, erase_gate 1.0);",
        "hippocampus = global top-w exact-KV cache by write magnitude m_t = ||Delta_t||_F;",
        "read = softmax over cache with sqrt(d)*cos logits (flat control: 0.83*cos);",
        f"combined read = rms_unit(state) + rms_unit(cache). seeds={seeds}, w in {widths}.",
        "Metric: cosine needle recall on the dla_poc state-only ablation grid.",
        "",
        "| w | needles | K | policy | states | ms | ms+hola | ms+hola-plain | ms+recency | ms+hola-flat | single | single+hola | needle-in-cache |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['w']} | {r['nseg']} | {r['K']} | {r['policy']} | {r['states']:.1f} "
            f"| {r['ms']:.3f} | {r['ms+hola']:.3f} | {r['ms+hola-plain']:.3f} "
            f"| {r['ms+recency']:.3f} "
            f"| {r['ms+hola-flat']:.3f} | {r['single']:.3f} | {r['single+hola']:.3f} "
            f"| {r['_needle_in_cache']:.2f} |"
        )
    lines += [
        "",
        "## HOLA vs HOLA-RWKV vs linear (same blocks, same cache, only the neocortex changes)",
        "",
        "gdn = HOLA's delta-rule backbone (alpha=0.98, beta=1); rwkv7 = repo recurrence;",
        "linear = plain block sum. `x/y` = state-only / state+hola-cache. w=16.",
        "",
        "| needles | K | policy | gdn | gdn+hola | rwkv7 | rwkv7+hola | linear | linear+hola | needle-in-cache g/r/l |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cmp_rows:
        lines.append(
            f"| {r['nseg']} | {r['K']} | {r['policy']} "
            f"| {r['gdn']:.3f} | {r['gdn+hola']:.3f} "
            f"| {r['rwkv7']:.3f} | {r['rwkv7+hola']:.3f} "
            f"| {r['linear']:.3f} | {r['linear+hola']:.3f} "
            f"| {r['_gdn_needle_in_cache']:.2f}/{r['_rwkv7_needle_in_cache']:.2f}/{r['_linear_needle_in_cache']:.2f} |"
        )
    lines += [
        "",
        f"- H4 (HOLA vs HOLA-RWKV): mean |gdn+hola - rwkv7+hola| = "
        f"{cmp_verd['h4_gdn_vs_rwkv_absdiff']:.4f} -> "
        f"{'PASS (within noise)' if cmp_verd['h4_pass'] else 'FAIL (backbones differ)'}",
        f"- H5 (surprise needs the delta rule): cache gain on bad boundaries — "
        f"delta backbones {cmp_verd['h5_cache_gain_delta_backbones']:+.3f} vs linear "
        f"{cmp_verd['h5_cache_gain_linear']:+.3f} -> "
        f"{'PASS' if cmp_verd['h5_pass'] else 'FAIL'}",
        "",
        "## Hypothesis verdicts",
        "",
        f"- H1 (surprise vs recency, bad boundaries): mean gap "
        f"{verd['h1_gap_bad_boundaries']:+.3f} -> {'PASS' if verd['h1_pass'] else 'FAIL'}",
        f"- H2 (sharp vs flat read, bad boundaries): mean gap "
        f"{verd['h2_gap_sharp_vs_flat']:+.3f} (flat vs no-cache "
        f"{verd['h2_flat_minus_no_cache']:+.3f}) -> {'PASS' if verd['h2_pass'] else 'FAIL'}",
        f"- H3 (low-K gap recovered by cache): "
        f"{verd['h3_lowk_gap_recovered']:.2f} of the oracle gap -> "
        f"{'PASS' if verd['h3_pass'] else 'FAIL'}",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (HERE / "results.json").write_text(
        json.dumps(dict(rows=rows, verdicts=verd,
                        comparison_rows=cmp_rows, comparison_verdicts=cmp_verd),
                   indent=2),
        encoding="utf-8",
    )


def smoke() -> int:
    # Parity: our per-block pass must reproduce dla_poc's multi-state recall exactly.
    gen = torch.Generator().manual_seed(1234)
    Kt, Vt, Smat, npos, nval = poc.make_recall_sequence(6, 8, 32, 32, gen)
    blocks = poc.fixed_blocks(Kt.shape[0], 8)
    states, _, tok2blk = build_states_and_scores(Kt, Vt, blocks)
    ours = poc.recall_from_states(Kt, blocks, states, npos, nval)
    ref = poc.hrm_rwkv7_multistate_recall_score(Kt, Vt, blocks, npos, nval)
    assert abs(ours - ref) < 1e-6, f"state parity broken: {ours} vs {ref}"
    rows, _ = run_grid(widths=[8], seeds=2, configs=[(8, 12, 16)],
                       policies=["fixed", "low_k_dla"])
    assert all(0.0 <= r["ms+hola"] <= 1.0 for r in rows)
    print("SMOKE_OK parity and quick grid pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    widths = [8, 16]
    rows, per_trial = run_grid(widths, args.seeds, CONFIGS, POLICIES)
    verd = verdicts(rows)
    cmp_rows = run_comparison_grid(16, args.seeds, CONFIGS, POLICIES)
    cmp_verd = comparison_verdicts(cmp_rows)
    write_report(rows, verd, widths, args.seeds, cmp_rows, cmp_verd)
    (HERE / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in per_trial), encoding="utf-8"
    )
    print(json.dumps(dict(**verd, **cmp_verd), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
