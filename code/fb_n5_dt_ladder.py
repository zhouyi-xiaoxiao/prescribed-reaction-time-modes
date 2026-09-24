#!/usr/bin/env python3
"""N5: time-step convergence with common random numbers and exact OU transitions.

GAP_CLOSURE_PLAN.md section 3, N5 (closes G09).  The stored SM dt-halving pairs
(robustness/dt_halving_summary.json) used independent streams at the two
resolutions and one of them, the boundary cell (m, eps, B) = (3, 0.175, 0.5),
flipped 3 -> 2 classified modes at dt/2.  This driver replaces those pairs by
a nested time-step ladder on COMMON Brownian increments, evaluated with the
exact-law Feynman--Kac weights of the N0 estimator (no kill coins, no
thresholds).

Schemes (all driven by the same standard-normal increments xi_k drawn at the
finest step h = 5e-4; the initial state is drawn once and shared):
* ``em_r{1,2,4}``: the production Euler--Maruyama scheme at dt = r h, r = 1, 2, 4
  (dt = 5e-4, 1e-3 [= production], 2e-3).  The coarse Brownian increment is the
  pair (quad) sum of the fine ones, sqrt(h) (xi_1 + ... + xi_r).
* ``ex_r{1,2,4}``: EXACT Gaussian OU transitions for the midpoint Z and the
  longitudinal separation R_par (the transverse separation is Brownian motion
  mod W, exact in every scheme).  Exact transitions compose, so the dt = r h
  chain is the fine exact chain subsampled every r steps: the ex_r ladder
  isolates the error of the end-of-step (right-point Riemann) killing
  quadrature, the em_r ladder the full production-scheme error.
Every scheme kills at the END of each of its steps with probability
1 - exp(-kappa(X_end) dt) (production convention), and the exact discrete-time
law is evaluated by the FK identity P(T = bin) = E[e^{-B X(a)}(1 - e^{-B (X(b)-X(a))})]
on right-closed 0.02 bins (e_i, e_{i+1}] that nest exactly for every dt (the
production float-edge histogram coincides with this at 147 of 151 window
edges; see N0 notes).  No Brownian-bridge correction for contact episodes
inside a step is applied.

Cells (the four SM dt pairs; the boundary cell is one of them):
    m3_eps0.1      B = 1 (m3_anchor), 3.57 (m3_threshold), equal weights
    m3_eps0.175    B = 0.5 (m3_phase_boundary), equal weights
    m5_z08_eps0.1  B = 1 (m5_anchor), z0 = 8, centres (2.8, 2.2, 1.6, 1.0, 0.4), w = 0.2
Local specification: 3 seed replicates x 2e5 paths per cell (seed tag 84,
base 20260923; replicate r, chunk i -> SeedSequence(entropy(r)).spawn(n_chunks)[i],
Philox).  Isambard specification (3 x 1e6 x 3 dt) is not run here.

Usage
-----
    python3 fb_n5_dt_ladder.py simulate --cell m3_eps0.1 --replicates 0,1,2 [--chunks 0,1]
    python3 fb_n5_dt_ladder.py analyze
    python3 fb_n5_dt_ladder.py figure
Chunk accumulators (batch sums, ~0.4 MB each) go to ~/.local-build/prr_fb_n5/;
results to artifacts/data/exact_m_fixed_budget/N5/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402

OUT = fk.FB_DATA / "N5"
CHUNK_ROOT = Path(os.environ.get("PRR_FB_N5_ROOT", str(Path.home() / ".local-build" / "prr_fb_n5")))
TAG = fk.TAGS["N5"]
H_FINE = 5e-4
FACTORS = (1, 2, 4)
SCHEMES = tuple(f"{k}_r{r}" for k in ("em", "ex") for r in FACTORS)
BIN = 0.02
TMAX = 4.0
CHUNK = 50_000
BATCH = 2_500
N_PER_REP = 200_000
M5_CENTRES = (2.8, 2.2, 1.6, 1.0, 0.4)

CELLS = {
    "m3_eps0.1": dict(spec=dict(m=3, eps=0.1), w=(1 / 3, 1 / 3, 1 / 3),
                      budgets=(1.0, 3.57), labels=("m3_anchor", "m3_threshold")),
    "m3_eps0.175": dict(spec=dict(m=3, eps=0.175), w=(1 / 3, 1 / 3, 1 / 3),
                        budgets=(0.5,), labels=("m3_phase_boundary",)),
    "m5_z08_eps0.1": dict(spec=dict(m=5, eps=0.1, z0=8.0, centres_z=M5_CENTRES),
                          w=(0.2,) * 5, budgets=(1.0,), labels=("m5_anchor",)),
}
SM_PAIRS = {  # robustness/dt_halving_summary.json#rows (independent streams, 5e5 walkers)
    "m3_anchor": {"mode_count_dt": 3, "mode_count_dt_half": 3},
    "m3_threshold": {"mode_count_dt": 2, "mode_count_dt_half": 2},
    "m3_phase_boundary": {"mode_count_dt": 3, "mode_count_dt_half": 2},
    "m5_anchor": {"mode_count_dt": 5, "mode_count_dt_half": 5},
}


def cell_spec(cell: str, dt: float = H_FINE) -> fk.EnsembleSpec:
    return fk.EnsembleSpec(dt=dt, tmax=TMAX, **CELLS[cell]["spec"])


# ----------------------------------------------------------------------------
# Chunk simulator
# ----------------------------------------------------------------------------


def _exposure(F, gate_d2, spec_h, centres, w, a2):
    """Unit-budget exposure over one fine step h at positions F (w folded)."""
    act, comps = fk._slab_components(F, centres, spec_h)
    if act.size == 0:
        return act, None
    e = fk._wsum(comps, w)
    e *= (gate_d2[act] < a2)
    return act, e


def run_chunk(task: dict) -> dict:
    cell = task["cell"]
    cfg = CELLS[cell]
    spec = cell_spec(cell)
    if spec.n_perp != 1:
        raise NotImplementedError("N5 ladder implemented for d = 2")
    n = int(task["size"])
    h = H_FINE
    steps = int(round(TMAX / h))
    per_bin = int(round(BIN / h))
    n_bins = steps // per_bin
    budgets = np.asarray(cfg["budgets"], float)
    nB = budgets.size
    w = np.asarray(cfg["w"], float)
    centres = spec.centres()
    a2 = spec.contact_a ** 2
    W = spec.torus_w
    g, zb, d0, eps = spec.gamma, spec.z_bar, spec.d0, spec.eps
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()

    # initial law (same draw order as the production simulator)
    z_init = spec.z0 + math.sqrt(spec.var_z0_scale * eps**2 * d0 / (2.0 * g)) * rng.standard_normal(n)
    rp_init = spec.r_par0 + eps * spec.u0 * rng.standard_normal(n)
    perp = np.mod(spec.r_perp0 + eps * spec.sigma_perp0 * rng.standard_normal(n), W)

    # exact chain
    zx = z_init.copy(); px = rp_init.copy()
    ax_ = math.exp(-g * h)
    sx_z = eps * math.sqrt(d0 * (1.0 - math.exp(-2.0 * g * h)) / (2.0 * g))
    sx_p = 2.0 * sx_z
    # EM chains r = 1, 2, 4
    ze = {r: z_init.copy() for r in FACTORS}
    pe = {r: rp_init.copy() for r in FACTORS}
    acc_z = {r: np.zeros(n) for r in FACTORS if r > 1}
    acc_p = {r: np.zeros(n) for r in FACTORS if r > 1}
    nz_h = eps * math.sqrt(d0 * h)          # EM noise per fine increment for Z
    nr_h = 2.0 * eps * math.sqrt(d0 * h)    # for R (par and perp)

    X = {s: np.zeros(n) for s in SCHEMES}
    Xlast = {s: np.zeros(n) for s in SCHEMES}
    nbatch = -(-n // BATCH)
    bid = np.arange(n) // BATCH
    bm = np.zeros((len(SCHEMES), nB, nbatch, n_bins))
    sq = np.zeros((len(SCHEMES), nB, n_bins))
    contact_frac = np.zeros(len(SCHEMES))

    for k in range(1, steps + 1):
        xi = rng.standard_normal((3, n))
        # transverse separation: exact Brownian motion mod W (common to all schemes)
        perp += nr_h * xi[2]
        np.mod(perp, W, out=perp)
        pm = np.minimum(perp, W - perp)
        pm2 = pm * pm
        # exact OU
        zx -= zb
        zx *= ax_
        zx += zb + sx_z * xi[0]
        px *= ax_
        px += sx_p * xi[1]
        d2x = px * px + pm2
        act, e = _exposure(zx, d2x, spec, centres, w, a2)
        if act.size:
            X["ex_r1"][act] += e
            if k % 2 == 0:
                X["ex_r2"][act] += 2.0 * e
            if k % 4 == 0:
                X["ex_r4"][act] += 4.0 * e
        # EM r = 1
        ze[1] *= (1.0 - g * h)
        ze[1] += g * zb * h + nz_h * xi[0]
        pe[1] *= (1.0 - g * h)
        pe[1] += nr_h * xi[1]
        d2 = pe[1] * pe[1] + pm2
        act, e = _exposure(ze[1], d2, spec, centres, w, a2)
        if act.size:
            X["em_r1"][act] += e
        for r in (2, 4):
            acc_z[r] += xi[0]
            acc_p[r] += xi[1]
            if k % r == 0:
                ze[r] *= (1.0 - g * r * h)
                ze[r] += g * zb * r * h + nz_h * acc_z[r]
                pe[r] *= (1.0 - g * r * h)
                pe[r] += nr_h * acc_p[r]
                acc_z[r][:] = 0.0
                acc_p[r][:] = 0.0
                d2 = pe[r] * pe[r] + pm2
                act, e = _exposure(ze[r], d2, spec, centres, w, a2)
                if act.size:
                    X[f"em_r{r}"][act] += r * e
        if k % per_bin == 0:
            b = k // per_bin - 1
            for si, s in enumerate(SCHEMES):
                inc = X[s] - Xlast[s]
                nzr = np.flatnonzero(inc > 0)
                if nzr.size:
                    for ib, B in enumerate(budgets):
                        y = np.exp(-B * Xlast[s][nzr]) * (-np.expm1(-B * inc[nzr]))
                        bm[si, ib, :, b] += np.bincount(bid[nzr], weights=y, minlength=nbatch)
                        sq[si, ib, b] += float((y * y).sum())
                Xlast[s][:] = X[s]
            if b == n_bins // 2:
                # contact diagnostic at t = 2
                contact_frac[:] = [float(np.mean(pe[r] * pe[r] + pm2 < a2)) for r in FACTORS] + \
                    [float(np.mean(px * px + pm2 < a2))] * 3
    runtime = time.perf_counter() - t0
    surv_tail = np.zeros((len(SCHEMES), nB))
    for si, s in enumerate(SCHEMES):
        for ib, B in enumerate(budgets):
            surv_tail[si, ib] = float(np.exp(-B * X[s]).sum())
    fpath = Path(task["file"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, bm=bm, sq=sq, surv_tail_sum=surv_tail, n=n,
                        contact_frac_t2=contact_frac)
    os.replace(tmp, fpath)
    return {"chunk": task["chunk"], "replicate": task["replicate"], "size": n,
            "runtime_seconds": runtime, "file": str(fpath)}


def replicate_entropy(cell: str, replicate: int) -> list[int]:
    spec = cell_spec(cell)
    ent = fk.path_entropy(spec, seed=fk.BASE_SEED, tag=TAG, replicate=replicate, chunk=CHUNK)
    # distinguish the ladder simulator from an fk.simulate_ensemble path set
    return ent + [5_000_000 + 1]


def cmd_simulate(args):
    cell = args.cell
    reps = [int(x) for x in args.replicates.split(",")]
    n_chunks = N_PER_REP // CHUNK
    chunks = list(range(n_chunks)) if args.chunks is None else [int(x) for x in args.chunks.split(",")]
    tasks = []
    for r in reps:
        ent = replicate_entropy(cell, r)
        children = np.random.SeedSequence(ent).spawn(n_chunks)
        for i in chunks:
            f = CHUNK_ROOT / cell / f"rep{r}_chunk{i:02d}.npz"
            if f.exists() and not args.force:
                continue
            tasks.append({"cell": cell, "replicate": r, "chunk": i, "size": CHUNK if args.size is None
                          else int(args.size), "seedseq": children[i], "file": str(f),
                          "entropy": ent})
    if not tasks:
        print("[n5] nothing to do", flush=True)
        return
    info = fk.wait_for_cpu_slot()
    print(f"[n5] {cell}: {len(tasks)} chunks; cpu wait {info}", flush=True)
    t0 = time.time()
    results = []
    if args.workers <= 1:
        for t in tasks:
            results.append(run_chunk(t))
            print(f"[n5] {cell} rep{results[-1]['replicate']} chunk {results[-1]['chunk']} "
                  f"{results[-1]['runtime_seconds']:.1f}s", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, fk.MAX_WORKERS, len(tasks))) as pool:
            for r in pool.imap_unordered(run_chunk, tasks):
                results.append(r)
                print(f"[n5] {cell} rep{r['replicate']} chunk {r['chunk']} {r['runtime_seconds']:.1f}s",
                      flush=True)
    log = CHUNK_ROOT / cell / "runs.jsonl"
    with open(log, "a") as fh:
        for r, t in zip(sorted(results, key=lambda x: (x["replicate"], x["chunk"])),
                        sorted(tasks, key=lambda x: (x["replicate"], x["chunk"]))):
            rec = dict(r)
            rec["seed_entropy"] = t["entropy"]
            rec["spawn_key"] = list(t["seedseq"].spawn_key)
            fh.write(json.dumps(rec) + "\n")
    print(f"[n5] wall {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def load_cell(cell: str):
    files = sorted((CHUNK_ROOT / cell).glob("rep*_chunk*.npz"))
    if not files:
        return None
    bms, sqs, tails, ns, contact = [], [], [], [], []
    for f in files:
        with np.load(f) as d:
            bms.append(d["bm"])
            sqs.append(d["sq"])
            tails.append(d["surv_tail_sum"])
            ns.append(int(d["n"]))
            contact.append(d["contact_frac_t2"])
    bm = np.concatenate(bms, axis=2)     # (S, nB, nbatch_total, bins)
    return {"files": [str(f) for f in files], "bm": bm, "sq": np.sum(sqs, axis=0),
            "tail": np.sum(tails, axis=0), "N": int(sum(ns)),
            "batch_size": BATCH, "contact_frac_t2": np.mean(contact, axis=0)}


def window_bins():
    edges_all = BIN * np.arange(int(round(TMAX / BIN)) + 1)
    lo = int(round(fk.WINDOW[0] / BIN))
    hi = int(round(fk.WINDOW[1] / BIN))
    return lo, hi, edges_all[lo:hi + 1]


def functionals(p_win: np.ndarray, edges: np.ndarray, cuts_idx: list, after_time: float,
                A: np.ndarray, m: int, with_protocol: bool = True) -> dict:
    """Scalar functionals of one window bin-mass vector."""
    t = 0.5 * (edges[:-1] + edges[1:])
    dens = p_win / BIN
    sm = np.einsum("ij,j->i", A, dens)
    out = {"window_mass": float(p_win.sum())}
    basins = [float(p_win[a:b].sum()) for a, b in zip(cuts_idx[:-1], cuts_idx[1:])]
    out["basin_masses"] = basins
    peaks = []
    for a, b in zip(cuts_idx[:-1], cuts_idx[1:]):
        seg = sm[a:b]
        i = a + int(np.argmax(seg))
        if 0 < i < sm.size - 1:
            y0, y1, y2 = sm[i - 1], sm[i], sm[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            off = max(-0.5, min(0.5, off))
        else:
            off = 0.0
        peaks.append(float(t[i] + off * BIN))
    out["peak_times_smoothed"] = peaks
    import fb_n6_visibility_thresholds as n6
    out["late_rel_prominence_smoothed"] = n6.late_prominence(sm, t, after_time)[0]
    out["late_rel_prominence_unsmoothed"] = n6.late_prominence(dens, t, after_time)[0]
    if with_protocol:
        for N in (500_000, 1_000_000):
            cls = fk.classify_expected(p_win, edges, walkers=N, bandwidth=0.04)
            late = [r for r in cls["rows"] if r["time"] > after_time]
            best = max(late, key=lambda r: r["z"]) if late else None
            out[f"protocol_mode_count_N{N}"] = int(cls["mode_count"])
            out[f"protocol_late_z_N{N}"] = float(best["z"]) if best else 0.0
    return out


def flatten(fd: dict) -> dict:
    out = {}
    for k, v in fd.items():
        if isinstance(v, list):
            for i, x in enumerate(v):
                out[f"{k}[{i}]"] = x
        else:
            out[k] = v
    return out


def analyze_budget(cell: str, ib: int, L: dict, groups: int = 60) -> dict:
    cfg = CELLS[cell]
    spec_prod = cell_spec(cell, dt=1e-3)
    m = len(cfg["w"])
    B = cfg["budgets"][ib]
    lo, hi, edges = window_bins()
    K = hi - lo
    A = fk._smoothing_operator(K, round(BIN, 14), 0.04)[2]
    valleys = fk.g_valley_times(spec_prod, cfg["w"])
    cuts = [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]]
    cuts_idx = [int(round((c - fk.WINDOW[0]) / BIN)) for c in cuts]
    after_time = valleys[-1]
    bm = L["bm"][:, ib, :, lo:hi]        # (S, nb, K)
    nb = bm.shape[1]
    N = L["N"]
    bs = L["batch_size"]
    # batch -> groups (contiguous)
    gidx = (np.arange(nb) * groups) // nb
    Gs = np.zeros((len(SCHEMES), groups, K))
    for gi in range(groups):
        Gs[:, gi] = bm[:, gidx == gi].sum(1)
    gsize = np.bincount(gidx, minlength=groups) * bs
    tot = Gs.sum(1)
    P = tot / N
    # iid per-bin SE
    sq = L["sq"][:, ib, lo:hi]
    se_bin = np.sqrt(np.maximum(sq / N - P * P, 0.0) / N)

    def funcs_for(Pmat, prot=True):
        return [flatten(functionals(Pmat[s], edges, cuts_idx, after_time, A, m, prot))
                for s in range(len(SCHEMES))]

    full = funcs_for(P)
    keys = list(full[0].keys())
    val = np.array([[f[k] for k in keys] for f in full])        # (S, nk)
    jack = np.zeros((groups,) + val.shape)
    for gi in range(groups):
        Pj = (tot - Gs[:, gi]) / (N - gsize[gi])
        fj = funcs_for(Pj, prot=True)
        jack[gi] = np.array([[f[k] for k in keys] for f in fj])
    si = {s: i for i, s in enumerate(SCHEMES)}

    def jse(x_full, x_jack):
        G = x_jack.shape[0]
        return float(math.sqrt((G - 1) / G * np.sum((x_jack - x_jack.mean(0)) ** 2)))

    rows = {}
    for kk, k in enumerate(keys):
        v = {s: float(val[si[s], kk]) for s in SCHEMES}
        jv = {s: jack[:, si[s], kk] for s in SCHEMES}
        row = {"values": v, "se": {s: jse(v[s], jv[s]) for s in SCHEMES}}
        for fam in ("em", "ex"):
            f1, f2, f4 = v[f"{fam}_r1"], v[f"{fam}_r2"], v[f"{fam}_r4"]
            j1, j2, j4 = jv[f"{fam}_r1"], jv[f"{fam}_r2"], jv[f"{fam}_r4"]
            R1 = 2 * f1 - f2
            R2 = (8 * f1 - 6 * f2 + f4) / 3.0
            row[f"{fam}_richardson1"] = R1
            row[f"{fam}_richardson1_se"] = jse(R1, 2 * j1 - j2)
            row[f"{fam}_richardson2"] = R2
            row[f"{fam}_richardson2_se"] = jse(R2, (8 * j1 - 6 * j2 + j4) / 3.0)
            d21, d42 = f2 - f1, f4 - f2
            row[f"{fam}_diff_r2_minus_r1"] = d21
            row[f"{fam}_diff_r2_minus_r1_se"] = jse(d21, j2 - j1)
            row[f"{fam}_diff_r4_minus_r2"] = d42
            row[f"{fam}_diff_r4_minus_r2_se"] = jse(d42, j4 - j2)
            row[f"{fam}_order_ratio"] = (d42 / d21) if d21 != 0 else None
        # CRN-paired SEs of each scheme's offset from the EM second-order limit
        R2j = (8 * jv["em_r1"] - 6 * jv["em_r2"] + jv["em_r4"]) / 3.0
        row["offset_from_em_R2"] = {s: v[s] - row["em_richardson2"] for s in SCHEMES}
        row["offset_from_em_R2_se_paired"] = {s: jse(v[s] - row["em_richardson2"], jv[s] - R2j)
                                              for s in SCHEMES}
        # production (em_r2, dt = 1e-3) vs Richardson-extrapolated continuum limit
        D = row["em_richardson1"] - v["em_r2"]
        Dj = (2 * jv["em_r1"] - jv["em_r2"]) - jv["em_r2"]
        row["prod_bias_D"] = D
        row["prod_bias_D_se"] = jse(D, Dj)
        row["prod_bias_D_ci95"] = [D - 1.96 * row["prod_bias_D_se"], D + 1.96 * row["prod_bias_D_se"]]
        row["prod_value_ci95_halfwidth"] = 1.96 * row["se"]["em_r2"]
        row["accept_abs_D_lt_ci95_of_dt1e-3_value"] = bool(abs(D) < 1.96 * row["se"]["em_r2"])
        row["accept_D_ci_contains_0"] = bool(abs(D) < 1.96 * row["prod_bias_D_se"])
        # EM vs exact-OU extrapolations must agree (same continuum limit)
        C = row["em_richardson1"] - row["ex_richardson1"]
        Cj = (2 * jv["em_r1"] - jv["em_r2"]) - (2 * jv["ex_r1"] - jv["ex_r2"])
        row["em_vs_ex_richardson1_diff"] = C
        row["em_vs_ex_richardson1_z"] = C / jse(C, Cj) if jse(C, Cj) > 0 else None
        C2 = row["em_richardson2"] - row["ex_richardson2"]
        Cj2 = (8 * jv["em_r1"] - 6 * jv["em_r2"] + jv["em_r4"]) / 3 - \
            (8 * jv["ex_r1"] - 6 * jv["ex_r2"] + jv["ex_r4"]) / 3
        row["em_vs_ex_richardson2_z"] = C2 / jse(C2, Cj2) if jse(C2, Cj2) > 0 else None
        rows[k] = row
    return {"cell": cell, "label": cfg["labels"][ib], "B": B, "weights": list(cfg["w"]),
            "m": m, "N_paths": N, "groups": groups, "g_valleys": valleys,
            "cuts_idx": cuts_idx, "after_time": after_time,
            "bin_mass": {s: P[si[s]].tolist() for s in SCHEMES},
            "bin_se_iid": {s: se_bin[si[s]].tolist() for s in SCHEMES},
            "edges": edges.tolist(), "functionals": rows,
            "survival_t4": {s: float(L["tail"][si[s], ib] / N) for s in SCHEMES},
            "contact_fraction_t2": {s: float(L["contact_frac_t2"][si[s]]) for s in SCHEMES}}


def protocol_resampling(P: np.ndarray, edges: np.ndarray, walkers: int, n_rep: int,
                        seed_seq, m: int) -> dict:
    """Multinomial histograms of `walkers` walkers from the exact law -> classify_both."""
    import reclassify_covariance_aware as recl
    rng = np.random.Generator(np.random.Philox(seed_seq))
    probs = np.r_[P, max(0.0, 1.0 - P.sum())]
    counts = rng.multinomial(walkers, probs, size=n_rep)[:, :-1]
    mc = np.zeros(n_rep, int)
    for i in range(n_rep):
        mc[i] = recl.classify_both(counts[i], edges, walkers, bandwidth=0.04)["mode_count_covariance_aware"]
    k = int((mc == m).sum())
    return {"walkers": walkers, "replicas": n_rep, "n_mode_count_eq_m": k,
            "p_mode_count_eq_m": k / n_rep, "wilson95": list(core.wilson_ci(k, n_rep)),
            "mode_count_histogram": {str(int(u)): int((mc == u).sum()) for u in np.unique(mc)}}


def stored_pair_reclassify() -> dict:
    """Late-candidate diagnostics of the stored independent-stream boundary pair."""
    import reclassify_covariance_aware as recl
    p = core.REPORT / "artifacts/data/exact_m_prr_upgrade/robustness/dt_halving/m3_phase_boundary.json"
    d = json.loads(p.read_text())
    out = {"source": str(p.relative_to(core.REPORT))}
    for lab in ("dt", "dt_half"):
        c = d["runs"][lab]["classifier"]
        walkers = int(d["runs"][lab]["kills"] + d["runs"][lab]["survivors"])
        res = recl.classify_both(c["counts"], c["edges"], walkers, bandwidth=0.04)
        late = [r for r in res["rows"] if r["time"] > 2.034]
        best = max(late, key=lambda r: r["z_covariance_aware"]) if late else None
        out[lab] = {"walkers": walkers, "mode_count_covariance_aware": res["mode_count_covariance_aware"],
                    "late_time": best["time"] if best else None,
                    "late_relative_prominence": best["relative_prominence"] if best else None,
                    "late_z_covariance_aware": best["z_covariance_aware"] if best else None}
    return out


def cmd_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"analysis": "N5 dt ladder on common random numbers (EM and exact-OU), exact FK law",
               "schemes": {s: {"family": s[:2], "dt": H_FINE * int(s[-1])} for s in SCHEMES},
               "binning": "right-closed 0.02 bins (e_i, e_{i+1}], nested for all dt",
               "seed": {"base_seed": fk.BASE_SEED, "tag": TAG,
                        "entropy_rule": "fk.path_entropy(spec(dt=5e-4), seed, tag=84, replicate, chunk=5e4) + [5000001]",
                        "rng": "Philox; chunk i = SeedSequence(entropy).spawn(4)[i]"},
               "richardson": {"R1": "2 f(5e-4) - f(1e-3) (first order)",
                              "R2": "(8 f(5e-4) - 6 f(1e-3) + f(2e-3))/3 (first+second order)"},
               "acceptance_reading": ("|R1 - f_EM(1e-3)| < 1.96 SE of the dt=1e-3 value (FK SE at this N); "
                                      "also reported: whether the CI of the bias D contains 0"),
               "cells": {}}
    for cell in CELLS:
        L = load_cell(cell)
        if L is None:
            continue
        cfg = CELLS[cell]
        results["cells"][cell] = {"N_paths": L["N"], "n_chunk_files": len(L["files"]),
                                  "budgets": {}}
        for ib, B in enumerate(cfg["budgets"]):
            r = analyze_budget(cell, ib, L)
            results["cells"][cell]["budgets"][cfg["labels"][ib]] = r
            print(f"[n5] {cell} B={B}: done", flush=True)
    # boundary cell: flip explanation
    bc = results["cells"].get("m3_eps0.175", {}).get("budgets", {}).get("m3_phase_boundary")
    if bc is not None:
        lo, hi, edges = window_bins()
        ss = np.random.SeedSequence([fk.BASE_SEED, TAG, 777]).spawn(8)
        flip = {"stored_pair": stored_pair_reclassify(), "resampling": {}}
        k = 0
        for s in ("em_r2", "em_r1", "ex_r1"):
            Pm = np.array(bc["bin_mass"][s])
            for N in (500_000, 1_000_000):
                flip["resampling"][f"{s}_N{N}"] = protocol_resampling(Pm, edges, N, args.replicas,
                                                                      ss[k], 3)
                k += 1
        a = flip["resampling"]["em_r2_N500000"]["p_mode_count_eq_m"]
        b = flip["resampling"]["em_r1_N500000"]["p_mode_count_eq_m"]
        flip["p_observed_pattern_3_then_2_at_5e5"] = a * (1 - b)
        flip["seed"] = {"entropy": [fk.BASE_SEED, TAG, 777], "spawned": 8}
        results["boundary_flip"] = flip
    results["seed"]["replicate_entropies"] = {c: {str(r): replicate_entropy(c, r) for r in range(3)}
                                              for c in CELLS}
    results["seed"]["chunk_run_logs"] = {c: str(CHUNK_ROOT / c / "runs.jsonl") for c in CELLS}
    results["validation_vs_n0"] = validate_vs_n0(results)
    results["peak_time_bias_model"] = peak_bias_model(results)
    core.write_json(OUT / "n5_dt_ladder.json", fk._jsonable(results))
    print("[n5] wrote", OUT / "n5_dt_ladder.json")


def validate_vs_n0(results: dict) -> dict:
    """Ladder em_r2 (EM, dt = 1e-3) basin masses vs the N0 FK ensembles (independent paths,
    production float-edge binning; the conventions differ only at window edges 0.50-0.56)."""
    pairs = (("m3_anchor", "m3_eps0.1", "n0_m3_eps0.1"), ("m3_threshold", "m3_eps0.1", "n0_m3_eps0.1"),
             ("m5_anchor", "m5_z08_eps0.1", "n0_m5_z08_eps0.1"))
    out = {}
    for lab, cell, name in pairs:
        r = results["cells"].get(cell, {}).get("budgets", {}).get(lab)
        if r is None or not fk.index_path(name).exists():
            continue
        ens = fk.load_ensemble(name)
        cuts = [fk.WINDOW[0]] + r["g_valleys"] + [fk.WINDOW[1]]
        bm = fk.basin_masses(ens, r["B"], r["weights"], cuts=cuts)
        rows = []
        for j in range(len(cuts) - 1):
            F = r["functionals"][f"basin_masses[{j}]"]
            v, se = F["values"]["em_r2"], F["se"]["em_r2"]
            z = (v - float(bm["masses"][j])) / math.hypot(se, float(bm["se_iid"][j]))
            rows.append({"ladder_em_dt1e-3": v, "ladder_se": se, "n0_fk": float(bm["masses"][j]),
                         "n0_se_iid": float(bm["se_iid"][j]), "z": z})
        out[lab] = {"n0_ensemble": name, "n0_cuts_edges": bm["cuts_edges"], "basins": rows,
                    "max_abs_z": max(abs(x["z"]) for x in rows)}
    return out


def peak_bias_model(results: dict) -> dict:
    """Compare the production peak-time bias D = R1 - f_EM(1e-3) with (dt/2)(1 + gamma t_peak):
    dt/2 from the end-of-step (right-point) kill-time stamp, gamma t dt/2 from the EM mean
    trajectory (1 - gamma dt)^n vs e^{-gamma t}."""
    rows = []
    for cell, c in results["cells"].items():
        for lab, r in c["budgets"].items():
            for k, F in r["functionals"].items():
                if not k.startswith("peak_times_smoothed"):
                    continue
                tp = F["em_richardson1"]
                pred = 0.5e-3 * (1.0 + tp)
                ex_bias = F["ex_richardson1"] - F["values"]["ex_r2"]
                rows.append({"cell": lab, "peak": k, "t_peak": tp, "D_em": F["prod_bias_D"],
                             "D_em_se": F["prod_bias_D_se"], "predicted_dt2_1_plus_t": pred,
                             "D_exactOU_same_dt": ex_bias, "predicted_exactOU_dt2": 0.5e-3})
    return {"rows": rows,
            "max_abs_D_em_minus_pred": max(abs(x["D_em"] - x["predicted_dt2_1_plus_t"]) for x in rows),
            "max_abs_D_ex_minus_dt2": max(abs(x["D_exactOU_same_dt"] - 0.5e-3) for x in rows)}


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    R = json.loads((OUT / "n5_dt_ladder.json").read_text())
    order = ["m3_anchor", "m3_threshold", "m3_phase_boundary", "m5_anchor"]
    rows = {lab: r for c in R["cells"].values() for lab, r in c["budgets"].items()}
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 6.0), layout="constrained")
    dts = np.array([5e-4, 1e-3, 2e-3]) * 1e3
    cols = [core.OI_BLUE, core.OI_ORANGE, core.OI_GREEN, core.OI_VERMILLION, core.OI_PURPLE]
    specs = [("basin_masses", 1e3, r"basin mass $-$ limit ($10^{-3}$)"),
             ("peak_times_smoothed", 1e3, r"peak time $-$ limit ($10^{-3}$)"),
             ("late_rel_prominence_smoothed", 1e2, "last-mode rel. prom. $-$ limit (%)")]
    for col, lab in enumerate(order):
        r = rows.get(lab)
        if r is None:
            continue
        F = r["functionals"]
        m = r["m"]
        for rowi, (base, scale, ylab) in enumerate(specs):
            ax = axes[rowi, col]
            keys = [f"{base}[{j}]" for j in range(m)] if base != "late_rel_prominence_smoothed" else [base]
            band = []
            for j, key in enumerate(keys):
                off = F[key]["offset_from_em_R2"]
                se = F[key]["offset_from_em_R2_se_paired"]
                for fam, mk, ls in (("em", "o", "-"), ("ex", "s", "--")):
                    y = [off[f"{fam}_r{q}"] * scale for q in FACTORS]
                    e = [1.96 * se[f"{fam}_r{q}"] * scale for q in FACTORS]
                    ax.errorbar(dts, y, yerr=e, color=cols[j % 5], marker=mk, ms=2.6, lw=0.9, ls=ls,
                                capsize=1.0, mfc=cols[j % 5] if fam == "em" else "white")
                band.append(1.96 * F[key]["se"]["em_r2"] * scale)
            hb = float(np.median(band))
            ax.axhspan(-hb, hb, color="0.85", lw=0, zorder=0)
            if base == "peak_times_smoothed":
                tt = np.linspace(0, 2.2, 10)
                for j, key in enumerate(keys):
                    tp = F[key]["em_richardson2"]
                    ax.plot(tt, -0.5 * tt * (1 + tp), color=cols[j % 5], lw=0.5, ls=":")
                ax.plot(tt, -0.5 * tt, color="k", lw=0.5, ls=":")
            ax.axhline(0, color="0.4", lw=0.6)
            ax.axvline(1.0, color="0.6", lw=0.6, ls=":")
            ax.set_xlim(0, 2.2)
            if rowi == 0:
                ax.set_title(f"{lab.replace('_', ' ')} ($B={r['B']:g}$)", fontsize=7.2)
            if rowi == 2:
                ax.set_xlabel(r"time step $\Delta t$ ($10^{-3}$)")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=7.0)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="k", marker="o", ls="-", ms=3, label="Euler–Maruyama (production scheme)"),
               Line2D([], [], color="k", marker="s", mfc="white", ls="--", ms=3, label="exact OU transitions"),
               Line2D([], [], color="k", ls=":", lw=0.6, label=r"peak shift $-(\Delta t/2)(1+\gamma t_{\rm peak})$ / $-\Delta t/2$"),
               Patch(color="0.85", label=r"95% sampling CI of the $\Delta t=10^{-3}$ value (median)")]
    handles += [Line2D([], [], color=cols[j], lw=1.2, label=f"basin/peak {j + 1}") for j in range(5)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.0, frameon=False)
    written = core.save_figure(fig, core.FIGURES / "fb_n5_dt_ladder")
    plt.close(fig)
    print(written)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--cell", required=True, choices=list(CELLS))
    s.add_argument("--replicates", default="0,1,2")
    s.add_argument("--chunks", default=None)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--size", default=None, help="override chunk size (tests only)")
    s.add_argument("--force", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--replicas", type=int, default=2000)
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
