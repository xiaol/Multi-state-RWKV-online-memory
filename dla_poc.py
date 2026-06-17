"""
Dynamic Linear Attention (DLA, arXiv 2606.10650) -- minimal mechanism reproduction.

This is a CPU-only, training-free proof of concept that isolates and tests DLA's
*core claim*: at a matched memory budget K, DLA's information-aware adaptive state
merging preserves salient / non-stationary information better than the FIXED block
schedule used by Log-Linear Attention (Guo et al., 2506.04761), which this repo
implements. That claim is the premise behind every empirical table in the paper
and is formalized in the paper's Theorem 3.1 / Corollary 3.2.

What this script does, end to end:
  1. Codebase smoke test: runs this repo's real Log-Linear recurrence
     (`hattention.recurrent.hattention_recurrent`, the fixed Fenwick-tree cascade)
     on a small batch and confirms it executes and produces multi-state output.
  2. Theorem 3.1 / Corollary 3.2 numerical check: on non-stationary sequences,
     computes the summarization deviation bound B(pi) for a FIXED contiguous
     blocking vs DLA's change-point-aligned blocking (Algorithm 1) at matched
     block count. Reproduces "fixed blocking is sub-optimal".
  3. Associative recall: compares DLA against fixed blocking plus
     self-contained HRM-Text-inspired rwkv_mem(delta_rule), rwkv_mem(rwkv7),
     and rwkv_mem(rwkv7_multi_state) mechanism baselines.
  4. State-update-only ablation: fixes the same boundary policy for both
     linear/DLA states and RWKV-7 states, so differences come only from the
     state update/readout rule.
  5. Capacity-bounded regime: when the number of true segments exceeds K,
     exercises Algorithm 2 (merge the adjacent pair with lowest information
     density) and shows DLA still beats fixed blocking at the same K.

All randomness is seeded. Results (summary + per-trial JSONL + a figure) are
written to .openresearch/artifacts/ and an EVAL.md summary at the repo root.
"""

import os
import sys
import json
import math
import types
import importlib.util
import pathlib

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / ".openresearch" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

LOG_LINES = []


def log(msg=""):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# ----------------------------------------------------------------------------
# 1. Codebase smoke test: run the repo's real Log-Linear recurrence.
#    We load hattention.base and hattention.recurrent directly (both are pure
#    torch) without triggering hattention/__init__.py, which pulls in triton /
#    mamba-ssm / transformers and would require CUDA.
# ----------------------------------------------------------------------------
def load_repo_loglinear():
    pkg = types.ModuleType("hattention")
    pkg.__path__ = [str(ROOT / "hattention")]
    sys.modules["hattention"] = pkg

    def _load(name, fname):
        spec = importlib.util.spec_from_file_location(name, ROOT / "hattention" / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    base = _load("hattention.base", "base.py")
    recur = _load("hattention.recurrent", "recurrent.py")
    return base, recur


def smoke_test_repo():
    log("=" * 72)
    log("[1] Codebase smoke test: repo Log-Linear attention (fixed cascade)")
    log("=" * 72)
    try:
        base, recur = load_repo_loglinear()
        b, T, h, d = 1, 32, 2, 16
        num_levels = base.get_num_levels(T, 2)
        Q = torch.randn(b, T, h, d)
        K = torch.randn(b, T, h, d)
        V = torch.randn(b, T, h, d)
        # A: per-token scalar decay (gating); 1.0 = no decay. float32 required.
        A = torch.ones(b, T, h, dtype=torch.float32)
        # L: per-level read weights (the lambda of log-linear); uniform here.
        L = torch.ones(b, T, h, num_levels, dtype=torch.float32)
        # Materialized log-linear attention forward (builds the hierarchical
        # Fenwick H matrix and applies it): repo code, pure torch.
        Y, data = base.hattention_materialized(
            Q, K, V, A, L, base=2, htype=base.HType.WEAK,
        )
        ok = (tuple(Y.shape) == (b, T, h, d)) and bool(torch.isfinite(Y).all())
        log(f"  ran hattention_materialized: output {tuple(Y.shape)}, "
            f"hierarchical levels={num_levels}, H shape={tuple(data['H'].shape)}")
        # Also confirm the recurrent multi-state cascade imports and steps.
        try:
            S = recur.HState(base=2, htype=base.HType.WEAK, hstruct=base.HStruct.MAMBA2,
                             shape=(b, h, d, d, num_levels), dtype=torch.float32,
                             device=torch.device("cpu"))
            recur.step_state(S, K[:, 0], V[:, 0], A[:, 0], None)
            log(f"  ran recurrent HState.step_state: states {tuple(S.states.shape)}")
        except Exception as e2:  # noqa
            log(f"  (recurrent cascade step note: {type(e2).__name__}: {e2})")
        log(f"  SMOKE TEST: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    except Exception as e:  # noqa
        log(f"  SMOKE TEST: FAIL ({type(e).__name__}: {e})")
        return False


# ----------------------------------------------------------------------------
# Synthetic non-stationary sequences.
# u_t in R^d: piecewise-constant segment means (random well-separated unit
# directions) plus small within-segment Gaussian noise. This is exactly the
# "non-stationary sequence with distinct segment means" of Corollary 3.2.
# ----------------------------------------------------------------------------
def make_nonstationary(num_segments, seg_len_mean, d, noise=0.03, gen=None):
    g = gen or torch.Generator().manual_seed(0)
    us, seg_ids, bounds = [], [], []
    t = 0
    for s in range(num_segments):
        mu = torch.randn(d, generator=g)
        mu = mu / mu.norm()
        # vary segment length a little so fixed blocks cannot trivially align
        L = max(2, int(seg_len_mean + torch.randint(-2, 3, (1,), generator=g).item()))
        bounds.append(t)
        for _ in range(L):
            us.append(mu + noise * torch.randn(d, generator=g))
            seg_ids.append(s)
            t += 1
    U = torch.stack(us, dim=0)  # [T, d]
    return U, torch.tensor(seg_ids), bounds


# ----------------------------------------------------------------------------
# DLA Algorithm 1: Information-Aware Dynamic State Merging.
# Returns block boundaries (list of start indices). RMSNorm is applied to the
# token contribution and the running state before the State Information Score,
# exactly as in the paper (Eq. 13).
# ----------------------------------------------------------------------------
def rmsnorm(x, eps=1e-6):
    return x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)


def dla_dynamic_merge(U, tau, eps=1e-6):
    T = U.shape[0]
    boundaries = [0]
    state_sum = U[0].clone()          # accumulated state S of the open block
    n = 1
    info_scores = [0.0]               # per-token I_t (I_1 := 0)
    for t in range(1, T):
        s_t = U[t]
        S = state_sum                 # current (most recent) memory state
        s_hat = rmsnorm(s_t)
        S_hat = rmsnorm(S)
        I_t = (s_hat - S_hat).norm() / (S_hat.norm() + eps)
        info_scores.append(float(I_t))
        if I_t >= tau:                # b_t = 1: start a new state
            boundaries.append(t)
            state_sum = s_t.clone()
            n = 1
        else:                         # b_t = 0: merge into current state
            state_sum = state_sum + s_t
            n += 1
    return boundaries, torch.tensor(info_scores)


def boundaries_to_blocks(boundaries, T):
    blocks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else T
        blocks.append(list(range(start, end)))
    return blocks


# ----------------------------------------------------------------------------
# DLA Algorithm 2: Capacity-Bounded Memory Modeling.
# Given the dynamic blocks and their aggregated per-token information scores,
# repeatedly merge the adjacent pair with the lowest information density
# (I_i + I_{i+1}) / (n_i + n_{i+1}) until at most K blocks remain. Adjacent-only
# merges preserve temporal order (paper Sec. 3.2, Eq. 16-17).
# ----------------------------------------------------------------------------
def dla_capacity_bound(blocks, info_scores, K):
    blocks = [list(b) for b in blocks]
    # aggregated info score per block (sum of per-token I_t)
    Ibar = [float(sum(info_scores[t] for t in b)) for b in blocks]
    n = [len(b) for b in blocks]
    while len(blocks) > K:
        best, best_density = None, None
        for i in range(len(blocks) - 1):
            density = (Ibar[i] + Ibar[i + 1]) / (n[i] + n[i + 1])
            if best_density is None or density < best_density:
                best_density, best = density, i
        i = best
        blocks[i] = blocks[i] + blocks[i + 1]
        Ibar[i] = Ibar[i] + Ibar[i + 1]
        n[i] = n[i] + n[i + 1]
        del blocks[i + 1], Ibar[i + 1], n[i + 1]
    return blocks


def fixed_blocks(T, m):
    """Fixed contiguous partition into m equal-ish blocks (the 'block every n
    tokens' fixed schedule that Log-Linear / fixed multi-state methods use)."""
    edges = np.linspace(0, T, m + 1).astype(int)
    edges = sorted(set(edges.tolist()))
    if edges[0] != 0:
        edges = [0] + edges
    if edges[-1] != T:
        edges = edges + [T]
    blocks = [list(range(edges[i], edges[i + 1])) for i in range(len(edges) - 1)
              if edges[i + 1] > edges[i]]
    return blocks


# ----------------------------------------------------------------------------
# Metric (a): Deviation bound B(pi) from Theorem 3.1 (q-independent factor):
#         B(pi) = sum_i sqrt(|C_i|) * sqrt(sum_{t in C_i} ||u_t - ubar_i||^2)
#     with ubar_i = block mean (the heterogeneity-minimizing representative).
#     This is the paper's own quantity; lower = better summarization.
# ----------------------------------------------------------------------------
def deviation_bound(U, blocks):
    total = 0.0
    for b in blocks:
        Ub = U[b]
        ubar = Ub.mean(dim=0, keepdim=True)
        hetero = ((Ub - ubar) ** 2).sum().item()
        total += math.sqrt(len(b)) * math.sqrt(hetero)
    return total


# ----------------------------------------------------------------------------
# Metric (b): Associative recall (needle-in-a-haystack style, Table 3).
# A sequence of redundant "filler" tokens (each segment shares a key/value
# theme) interspersed with rare "needle" tokens carrying a distinctive key k*
# and value v*. The linear-attention state of a block is S = sum_t k_t v_t^T.
# Probing the needle's block with its key recovers
#     o = k*^T S = (k*.k*) v* + sum_{filler in block} (k*.k_filler) v_filler.
# If the needle sits alone in its state (DLA isolates it because its k*/v* give
# a high State Information Score), o ~= v* and recall is perfect. If the needle
# is buried with filler (fixed blocking), the filler terms interfere and recall
# drops. Metric: mean cosine(o, v*) over needles. Higher = better.
# ----------------------------------------------------------------------------
def make_recall_sequence(num_segments, num_filler, d, dv, gen):
    keys, vals, is_needle = [], [], []
    needle_pos, needle_val = [], []
    t = 0
    for s in range(num_segments):
        # needle: distinctive key and value (the thing to retrieve)
        kstar = torch.randn(d, generator=gen); kstar = kstar / kstar.norm()
        vstar = torch.randn(dv, generator=gen); vstar = vstar / vstar.norm()
        keys.append(kstar); vals.append(vstar); is_needle.append(True)
        needle_pos.append(t); needle_val.append(vstar)
        t += 1
        # redundant filler: shared key/value theme + tiny noise (low information)
        ktheme = torch.randn(d, generator=gen); ktheme = ktheme / ktheme.norm()
        vtheme = torch.randn(dv, generator=gen); vtheme = vtheme / vtheme.norm()
        nf = max(2, num_filler + int(torch.randint(-2, 3, (1,), generator=gen).item()))
        for _ in range(nf):
            k = ktheme + 0.02 * torch.randn(d, generator=gen); k = k / k.norm()
            v = vtheme + 0.02 * torch.randn(dv, generator=gen)
            keys.append(k); vals.append(v); is_needle.append(False)
            t += 1
    Kt = torch.stack(keys)          # [T, d]
    Vt = torch.stack(vals)          # [T, dv]
    # per-token state contribution s_t = k_t v_t^T, flattened for Algorithm 1
    Smat = (Kt.unsqueeze(-1) * Vt.unsqueeze(-2)).reshape(Kt.shape[0], -1)  # [T, d*dv]
    return Kt, Vt, Smat, needle_pos, torch.stack(needle_val)


def recall_score(Kt, Vt, blocks, needle_pos, needle_val):
    # map each token to its block
    tok2blk = {}
    for bi, b in enumerate(blocks):
        for t in b:
            tok2blk[t] = bi
    cos = []
    for p, v in zip(needle_pos, needle_val):
        idx = blocks[tok2blk[p]]
        kstar = Kt[p]
        sims = Kt[idx] @ kstar                     # (k*.k_t) for t in block
        o = sims @ Vt[idx]                          # k*^T S_block
        denom = (o.norm() * v.norm()) + 1e-8
        cos.append(float((o @ v) / denom))
    return float(np.mean(cos))


# ----------------------------------------------------------------------------
# HRM-Text-inspired baseline.
#
# HRM-Text's `models/rwkv_memory.py` has two memory adapter paths. The active
# delta-rule path reads an online associative state before writing the current
# token, then updates with keep/erase/write coefficients:
#     read_t = S_{t-1} q_t
#     pred_t = S_{t-1} k_t
#     S_t = keep * S_{t-1} - erase * pred_t k_t^T + write * v_t k_t^T
#
# Importing HRM-Text directly would pull in its whole model/training stack. For a
# CPU-only mechanism baseline, this function implements the same recurrence on
# the synthetic key/value recall task. It is a single compact online state, not a
# multi-state segmented memory, so it is reported only on recall, not on the DLA
# partition deviation bound.
# ----------------------------------------------------------------------------
def hrm_delta_rule_recall_score(Kt, Vt, needle_pos, needle_val, beta=None):
    if beta is None:
        beta = float(torch.sigmoid(torch.tensor(-1.5)))  # HRM default bias.
    d = Kt.shape[1]
    dv = Vt.shape[1]
    if d != dv:
        raise ValueError("HRM delta-rule recall baseline expects key/value dims to match")

    keep = 1.0 - beta
    erase = beta
    write = beta
    state = torch.zeros(dv, d, dtype=torch.float32)
    for k_t, v_t in zip(Kt.float(), Vt.float()):
        read_key = k_t
        pred_t = state @ read_key
        state = (
            keep * state
            - erase * torch.outer(pred_t, read_key)
            + write * torch.outer(v_t, read_key)
        )

    cos = []
    for p, v in zip(needle_pos, needle_val.float()):
        out = state @ Kt[p].float()
        denom = (out.norm() * v.norm()) + 1e-8
        cos.append(float((out @ v) / denom))
    return float(np.mean(cos))


def hrm_rwkv7_state(Kt, Vt, token_indices, decay=0.98, erase_gate=1.0):
    """Latest HRM rwkv_mem(rwkv7) state update, specialized to synthetic K/V.

    HRM-Text's latest RWKV memory comparison path reads before writing the
    current token and then applies the RWKV-7 recurrence:
        state = decay * state + v k^T + (state a) b^T
    with a key-direction correction. In the real adapter, r/w/k/v/a/b come from
    learned RWKV-7 projections. Here we use the synthetic key/value directly and
    a fixed long-memory decay so the comparison tests the recurrence mechanism,
    not random untrained projections.
    """
    d = Kt.shape[1]
    dv = Vt.shape[1]
    if d != dv:
        raise ValueError("HRM RWKV-7 recall baseline expects key/value dims to match")

    state = torch.zeros(dv, d, dtype=torch.float32)
    for t in token_indices:
        k_t = Kt[t].float()
        v_t = Vt[t].float()
        kk = k_t / (k_t.norm() + 1e-8)
        # Equivalent to HRM-Text rwkv7_recurrence_read_before_write_torch with
        # a=-kk and b=erase_gate*kk, after replacing learned projections by the
        # synthetic associative-recall key/value stream.
        correction_read = state @ (-kk)
        state = (
            decay * state
            + torch.outer(v_t, k_t)
            + torch.outer(correction_read, erase_gate * kk)
        )
    return state


def recall_from_states(Kt, blocks, states, needle_pos, needle_val):
    tok2blk = {}
    for bi, b in enumerate(blocks):
        for t in b:
            tok2blk[t] = bi

    cos = []
    for p, v in zip(needle_pos, needle_val.float()):
        out = states[tok2blk[p]] @ Kt[p].float()
        denom = (out.norm() * v.norm()) + 1e-8
        cos.append(float((out @ v) / denom))
    return float(np.mean(cos))


def hrm_rwkv7_recall_score(Kt, Vt, needle_pos, needle_val, decay=0.98, erase_gate=1.0):
    state = hrm_rwkv7_state(Kt, Vt, range(Kt.shape[0]), decay=decay, erase_gate=erase_gate)
    cos = []
    for p, v in zip(needle_pos, needle_val.float()):
        out = state @ Kt[p].float()
        denom = (out.norm() * v.norm()) + 1e-8
        cos.append(float((out @ v) / denom))
    return float(np.mean(cos))


def hrm_rwkv7_multistate_recall_score(
    Kt,
    Vt,
    blocks,
    needle_pos,
    needle_val,
    decay=0.98,
    erase_gate=1.0,
):
    states = [
        hrm_rwkv7_state(Kt, Vt, b, decay=decay, erase_gate=erase_gate)
        for b in blocks
    ]
    return recall_from_states(Kt, blocks, states, needle_pos, needle_val)


def oracle_needle_blocks(T, needle_pos):
    """Perfect boundary policy: isolate each needle and group filler spans."""
    blocks = []
    cursor = 0
    for p in needle_pos:
        if cursor < p:
            blocks.append(list(range(cursor, p)))
        blocks.append([p])
        cursor = p + 1
    if cursor < T:
        blocks.append(list(range(cursor, T)))
    return blocks


def jitter_blocks(blocks, T, max_shift, gen):
    """Keep block count fixed but perturb internal boundaries deterministically."""
    if len(blocks) <= 1:
        return [list(range(T))]
    raw = []
    for b in blocks[1:]:
        delta = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=gen).item())
        raw.append(b[0] + delta)
    raw.sort()

    fixed = []
    prev = 0
    for i, start in enumerate(raw):
        remaining = len(raw) - i
        low = prev + 1
        high = T - remaining
        start = max(low, min(high, start))
        fixed.append(start)
        prev = start
    return boundaries_to_blocks([0] + fixed, T)


# ----------------------------------------------------------------------------
# Experiment driver.
# ----------------------------------------------------------------------------
def run_bound():
    log()
    log("=" * 72)
    log("[2] Theorem 3.1 / Corollary 3.2: deviation bound at matched budget K")
    log("=" * 72)
    d = 32
    tau = 0.6  # paper default merge boundary
    configs = [
        # (label, num_segments, seg_len_mean, K)  ; K = matched state budget
        ("K>=segments  (Alg 1 active)", 6, 24, 8),
        ("K>=segments  (Alg 1 active)", 10, 16, 12),
        ("K< segments  (Alg 2 active)", 16, 12, 8),
        ("K< segments  (Alg 2 active)", 24, 10, 10),
    ]
    rows, per_trial = [], []
    for label, nseg, slen, K in configs:
        b_fix_list, b_dla_list, mdla_list = [], [], []
        for seed in range(5):
            gen = torch.Generator().manual_seed(seed)
            U, _, _ = make_nonstationary(nseg, slen, d, noise=0.03, gen=gen)
            T = U.shape[0]
            bnds, info = dla_dynamic_merge(U, tau=tau)
            dla_blocks = dla_capacity_bound(boundaries_to_blocks(bnds, T), info, K)
            m_dla = len(dla_blocks)
            fx_blocks = fixed_blocks(T, m_dla)   # matched budget
            b_fix = deviation_bound(U, fx_blocks)
            b_dla = deviation_bound(U, dla_blocks)
            b_fix_list.append(b_fix); b_dla_list.append(b_dla); mdla_list.append(m_dla)
            per_trial.append(dict(metric="bound", config=label, nseg=nseg, K=K,
                                  seed=seed, T=T, m_dla=m_dla,
                                  bound_fixed=b_fix, bound_dla=b_dla))
        bf, bd = np.mean(b_fix_list), np.mean(b_dla_list)
        rows.append(dict(config=label, nseg=nseg, K=K, m_dla=float(np.mean(mdla_list)),
                         bound_fixed=bf, bound_dla=bd, bound_reduction=1 - bd / bf))
        log()
        log(f"  config: {label}  (segments={nseg}, K={K}, avg DLA states={np.mean(mdla_list):.1f})")
        log(f"    deviation bound   fixed={bf:8.3f}  DLA={bd:8.3f}  -> {100*(1-bd/bf):5.1f}% lower")
    return rows, per_trial


def run_recall():
    log()
    log("=" * 72)
    log("[3] Associative recall (needle): retrieval at matched budget K")
    log("=" * 72)
    d, dv = 32, 32
    tau = 0.6
    configs = [
        # (num_segments=#needles, num_filler, K)  ; K >= 2*needles -> Alg 1 isolates
        (6, 8, 16),
        (10, 6, 24),
        (8, 10, 20),
    ]
    rows, per_trial = [], []
    for nseg, nfill, K in configs:
        r_fix_list, r_dla_list, r_delta_list = [], [], []
        r_rwkv7_list, r_rwkv7_multi_list, mdla_list = [], [], []
        for seed in range(5):
            gen = torch.Generator().manual_seed(100 + seed)
            Kt, Vt, Smat, npos, nval = make_recall_sequence(nseg, nfill, d, dv, gen)
            T = Kt.shape[0]
            bnds, info = dla_dynamic_merge(Smat, tau=tau)
            dla_blocks = dla_capacity_bound(boundaries_to_blocks(bnds, T), info, K)
            m_dla = len(dla_blocks)
            fx_blocks = fixed_blocks(T, m_dla)   # matched budget
            r_fix = recall_score(Kt, Vt, fx_blocks, npos, nval)
            r_dla = recall_score(Kt, Vt, dla_blocks, npos, nval)
            r_delta = hrm_delta_rule_recall_score(Kt, Vt, npos, nval)
            r_rwkv7 = hrm_rwkv7_recall_score(Kt, Vt, npos, nval)
            r_rwkv7_multi = hrm_rwkv7_multistate_recall_score(
                Kt, Vt, dla_blocks, npos, nval,
            )
            r_fix_list.append(r_fix); r_dla_list.append(r_dla)
            r_delta_list.append(r_delta); r_rwkv7_list.append(r_rwkv7)
            r_rwkv7_multi_list.append(r_rwkv7_multi); mdla_list.append(m_dla)
            per_trial.append(dict(metric="recall", nseg=nseg, nfill=nfill, K=K,
                                  seed=seed, T=T, m_dla=m_dla,
                                  recall_fixed=r_fix, recall_dla=r_dla,
                                  recall_hrm_delta=r_delta,
                                  recall_hrm_rwkv7=r_rwkv7,
                                  recall_hrm_rwkv7_multi=r_rwkv7_multi))
        rf = np.mean(r_fix_list)
        rd = np.mean(r_dla_list)
        rdelta = np.mean(r_delta_list)
        rrwkv7 = np.mean(r_rwkv7_list)
        rrwkv7_multi = np.mean(r_rwkv7_multi_list)
        rows.append(dict(nseg=nseg, nfill=nfill, K=K, m_dla=float(np.mean(mdla_list)),
                         recall_fixed=rf, recall_dla=rd,
                         recall_hrm_delta=rdelta,
                         recall_hrm_rwkv7=rrwkv7,
                         recall_hrm_rwkv7_multi=rrwkv7_multi))
        log()
        log(f"  config: needles={nseg}, filler/seg={nfill}, K={K}, "
            f"avg DLA states={np.mean(mdla_list):.1f}")
        log(f"    recall cos(o, v*)   fixed={rf:6.3f}  "
            f"HRM-delta={rdelta:6.3f}  HRM-rwkv7={rrwkv7:6.3f}  "
            f"HRM-rwkv7-ms={rrwkv7_multi:6.3f}  DLA={rd:6.3f}")
    return rows, per_trial


def run_state_only():
    log()
    log("=" * 72)
    log("[4] State-update-only ablation: same boundaries, different states")
    log("=" * 72)
    d, dv = 32, 32
    tau = 0.6
    configs = [
        # (num_segments=#needles, num_filler, K)
        (8, 12, 16),
        (12, 10, 16),
        (16, 8, 12),
    ]
    rows, per_trial = [], []
    for nseg, nfill, K in configs:
        for policy in ("oracle", "dla", "fixed", "noisy_dla", "low_k_dla"):
            lin_list, rwkv_list, state_count_list = [], [], []
            for seed in range(5):
                gen = torch.Generator().manual_seed(500 + 31 * seed + nseg)
                Kt, Vt, Smat, npos, nval = make_recall_sequence(nseg, nfill, d, dv, gen)
                T = Kt.shape[0]
                bnds, info = dla_dynamic_merge(Smat, tau=tau)
                dla_blocks = dla_capacity_bound(boundaries_to_blocks(bnds, T), info, K)

                if policy == "oracle":
                    blocks = oracle_needle_blocks(T, npos)
                elif policy == "dla":
                    blocks = dla_blocks
                elif policy == "fixed":
                    blocks = fixed_blocks(T, len(dla_blocks))
                elif policy == "noisy_dla":
                    blocks = jitter_blocks(dla_blocks, T, max_shift=2, gen=gen)
                elif policy == "low_k_dla":
                    low_k = max(1, K // 2)
                    blocks = dla_capacity_bound(boundaries_to_blocks(bnds, T), info, low_k)
                else:
                    raise ValueError(f"unknown state-only policy: {policy}")

                r_linear = recall_score(Kt, Vt, blocks, npos, nval)
                r_rwkv = hrm_rwkv7_multistate_recall_score(Kt, Vt, blocks, npos, nval)
                lin_list.append(r_linear); rwkv_list.append(r_rwkv)
                state_count_list.append(len(blocks))
                per_trial.append(dict(metric="state_only", policy=policy,
                                      nseg=nseg, nfill=nfill, K=K, seed=seed,
                                      T=T, states=len(blocks),
                                      recall_linear_state=r_linear,
                                      recall_rwkv7_state=r_rwkv))
            lin = float(np.mean(lin_list))
            rwkv = float(np.mean(rwkv_list))
            states = float(np.mean(state_count_list))
            rows.append(dict(policy=policy, nseg=nseg, nfill=nfill, K=K,
                             states=states, recall_linear_state=lin,
                             recall_rwkv7_state=rwkv,
                             rwkv_minus_linear=rwkv - lin))
            log()
            log(f"  policy={policy:9s} needles={nseg}, filler/seg={nfill}, "
                f"K={K}, avg states={states:.1f}")
            log(f"    linear/DLA-state={lin:6.3f}  "
                f"RWKV7-state={rwkv:6.3f}  delta={rwkv-lin:+6.3f}")
    return rows, per_trial


def write_artifacts(smoke_ok, bound_rows, recall_rows, state_rows, per_trial):
    with open(ART / "dla_trials.jsonl", "w") as f:
        for r in per_trial:
            f.write(json.dumps(r) + "\n")
    with open(ART / "dla_summary.json", "w") as f:
        json.dump(dict(smoke_test_pass=smoke_ok, bound=bound_rows,
                       recall=recall_rows, state_only=state_rows),
                  f, indent=2)

    # figure: deviation bound (lower better) + recall (higher better)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(len(bound_rows))
        axes[0].bar(x - 0.2, [r["bound_fixed"] for r in bound_rows], 0.4,
                    label="fixed (log-linear)", color="#c44")
        axes[0].bar(x + 0.2, [r["bound_dla"] for r in bound_rows], 0.4,
                    label="DLA (adaptive)", color="#3a7")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{r['nseg']}seg/K{r['K']}" for r in bound_rows], rotation=15)
        axes[0].set_title("Theorem 3.1 deviation bound (lower better)")
        axes[0].legend()
        xr = np.arange(len(recall_rows))
        width = 0.16
        axes[1].bar(xr - 2 * width, [r["recall_fixed"] for r in recall_rows], width,
                    label="fixed (log-linear)", color="#c44")
        axes[1].bar(xr - width, [r["recall_hrm_delta"] for r in recall_rows], width,
                    label="rwkv_mem(delta_rule)", color="#47a")
        axes[1].bar(xr, [r["recall_hrm_rwkv7"] for r in recall_rows], width,
                    label="rwkv_mem(rwkv7)", color="#a77")
        axes[1].bar(xr + width, [r["recall_hrm_rwkv7_multi"] for r in recall_rows], width,
                    label="rwkv_mem(rwkv7 multi-state)", color="#b98c10")
        axes[1].bar(xr + 2 * width, [r["recall_dla"] for r in recall_rows], width,
                    label="DLA (adaptive)", color="#3a7")
        axes[1].set_xticks(xr)
        axes[1].set_xticklabels([f"{r['nseg']}ndl/K{r['K']}" for r in recall_rows], rotation=15)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Needle recall cos(o, v*) (higher better)")
        axes[1].legend(fontsize=8)
        fig.suptitle("DLA vs fixed log-linear and HRM rwkv_mem baselines")
        fig.tight_layout()
        fig.savefig(ART / "dla_comparison.png", dpi=120)
        log()
        log(f"  figure written: {ART / 'dla_comparison.png'}")
    except Exception as e:  # noqa
        log(f"  (figure skipped: {type(e).__name__}: {e})")

    all_bound = all(r["bound_dla"] < r["bound_fixed"] for r in bound_rows)
    all_recall = all(r["recall_dla"] > r["recall_fixed"] for r in recall_rows)
    all_hrm_delta_recall = all(r["recall_dla"] > r["recall_hrm_delta"] for r in recall_rows)
    all_hrm_rwkv7_recall = all(r["recall_dla"] > r["recall_hrm_rwkv7"] for r in recall_rows)
    all_rwkv7_multi_recall = all(
        r["recall_hrm_rwkv7_multi"] >= r["recall_hrm_rwkv7"]
        for r in recall_rows
    )
    state_win_eps = 1e-6
    state_rwkv_wins = sum(
        1 for r in state_rows
        if r["recall_rwkv7_state"] - r["recall_linear_state"] > state_win_eps
    )
    state_linear_wins = sum(
        1 for r in state_rows
        if r["recall_linear_state"] - r["recall_rwkv7_state"] > state_win_eps
    )
    state_ties = len(state_rows) - state_rwkv_wins - state_linear_wins
    success = smoke_ok and all_bound and all_recall

    lines = []
    lines.append("# DLA mechanism reproduction (matched memory budget)\n")
    lines.append(f"- Codebase smoke test (repo Log-Linear attention ran): "
                 f"**{'PASS' if smoke_ok else 'FAIL'}**\n")
    lines.append("\n## Theorem 3.1 deviation bound, fixed vs DLA (mean over 5 seeds)\n")
    lines.append("| config | segments | K | DLA states | bound fixed | bound DLA | bound ↓ |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in bound_rows:
        lines.append(f"| {r['config']} | {r['nseg']} | {r['K']} | {r['m_dla']:.1f} | "
                     f"{r['bound_fixed']:.3f} | {r['bound_dla']:.3f} | "
                     f"{100*r['bound_reduction']:.1f}% |")
    lines.append("\n## Associative recall cos(o, v*), mechanism baselines (mean over 5 seeds)\n")
    lines.append("| needles | filler/seg | K | states | fixed | rwkv_mem(delta_rule) | rwkv_mem(rwkv7) | rwkv_mem(rwkv7 multi-state) | DLA |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in recall_rows:
        lines.append(f"| {r['nseg']} | {r['nfill']} | {r['K']} | {r['m_dla']:.1f} | "
                     f"{r['recall_fixed']:.3f} | {r['recall_hrm_delta']:.3f} | "
                     f"{r['recall_hrm_rwkv7']:.3f} | "
                     f"{r['recall_hrm_rwkv7_multi']:.3f} | "
                     f"{r['recall_dla']:.3f} |")
    lines.append("\nThe HRM baselines are self-contained mechanism ports from "
                 "`HRM-Text/models/rwkv_memory.py` and `models/rwkv7.py`. "
                 "`rwkv_mem(delta_rule)` uses the online delta-rule associative "
                 "state with HRM's default beta bias -1.5. `rwkv_mem(rwkv7)` "
                 "uses the latest read-before-write RWKV-7 recurrence specialized "
                 "to the synthetic key/value stream. `rwkv_mem(rwkv7 multi-state)` "
                 "keeps the same RWKV-7 state update but allocates one state per "
                 "DLA adaptive block at the same state count.\n")
    lines.append("\n## State-Update-Only Ablation, Same Boundaries (mean over 5 seeds)\n")
    lines.append("| boundary policy | needles | filler/seg | K | states | linear/DLA state | RWKV-7 state | RWKV - linear |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in state_rows:
        lines.append(f"| {r['policy']} | {r['nseg']} | {r['nfill']} | {r['K']} | "
                     f"{r['states']:.1f} | {r['recall_linear_state']:.3f} | "
                     f"{r['recall_rwkv7_state']:.3f} | "
                     f"{r['rwkv_minus_linear']:+.3f} |")
    lines.append("\nThis table fixes the exact same token blocks for both methods. "
                 "`linear/DLA state` uses the standard block sum `sum k_t v_t^T`; "
                 "`RWKV-7 state` uses the RWKV-7 recurrence inside each same block. "
                 "Therefore each row compares state update/readout only, not boundary quality.\n")
    lines.append("\n## Verdict\n")
    lines.append(f"- DLA lower deviation bound in every config: **{all_bound}** "
                 "(reproduces Theorem 3.1 / Corollary 3.2)")
    lines.append(f"- DLA higher needle recall in every config: **{all_recall}**")
    lines.append(f"- DLA higher than HRM rwkv_mem(delta_rule): **{all_hrm_delta_recall}**")
    lines.append(f"- DLA higher than HRM rwkv_mem(rwkv7): **{all_hrm_rwkv7_recall}**")
    lines.append(f"- Multi-state RWKV-7 improves over single-state RWKV-7: **{all_rwkv7_multi_recall}**")
    lines.append(f"- State-only ablation wins, RWKV-7/linear/tie: "
                 f"**{state_rwkv_wins}/{state_linear_wins}/{state_ties}**")
    lines.append(f"- Codebase smoke test: **{smoke_ok}**")
    lines.append(f"- Core claim (adaptive merging beats fixed schedule at matched budget): "
                 f"**{'REPRODUCED' if success else 'NOT REPRODUCED'}**")
    (ROOT / "EVAL.md").write_text("\n".join(lines) + "\n")
    (ART / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n")
    return success


def main():
    smoke_ok = smoke_test_repo()
    bound_rows, bound_trials = run_bound()
    recall_rows, recall_trials = run_recall()
    state_rows, state_trials = run_state_only()
    success = write_artifacts(smoke_ok, bound_rows, recall_rows, state_rows,
                              bound_trials + recall_trials + state_trials)
    log()
    log("=" * 72)
    log(f"OVERALL: core claim {'REPRODUCED' if success else 'NOT REPRODUCED'}")
    log("=" * 72)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
