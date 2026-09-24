#!/usr/bin/env python3
"""HPC item A (HPC_N5full): full N5 time-step ladder on Isambard 3.

GAP_CLOSURE_PLAN.md section 3 N5, "Isambard specification" (GPT-6 G7):
3 seeds x 1e6 paths per cell x nested dt in {2e-3, 1e-3, 5e-4}, exact OU
transitions for Z and R_par, R_perp Brownian mod W, common Brownian increments
generated at the finest step and pair/quad-summed, optional Brownian-bridge
contact correction.  The local driver fb_n5_dt_ladder.py ran 3 x 2e5.

Schemes (all on the SAME paths; main normals from the chunk's Philox stream in
exactly the draw order of fb_n5_dt_ladder.run_chunk):
* em_r{1,2,4}: production Euler--Maruyama at dt = r h (h = 5e-4), coarse
  increments = sums of the fine ones.
* ex_r{1,2,4}: exact Gaussian OU transitions for Z and R_par (subsampled
  every r fine steps), end-of-step (right-point) exposure.
* ex_b{2,4}: Brownian-bridge refinement of the exact chain.  Inside every fine
  step [t_{k-1}, t_k] exact bridge points are drawn at t_{k-1} + h/2
  (b2) and additionally at h/4, 3h/4 (b4), CONDITIONAL on the two exact
  endpoints (OU bridge for Z and R_par, Brownian bridge for the unwrapped
  R_perp), from a SEPARATE Philox stream (so em/ex values are unaffected).
  The exposure over the step uses the right-point rule on the refined grid,
  i.e. ex_b2 / ex_b4 are the exact-OU schemes at dt = h/2, h/4 on the same
  coarse path: this resolves contact (gate) episodes that begin and end
  inside one fine step, which the end-point rule of every other scheme
  misses.  The b-ladder (1.25e-4, 2.5e-4, 5e-4) is nested with ex_r1.
Kills: exact discrete-time FK law of each scheme, P(T in (e_i, e_{i+1}]) =
E[e^{-B X(e_i)} (1 - e^{-B (X(e_{i+1}) - X(e_i))})] on right-closed 0.02
bins nested for every dt (same convention as fb_n5_dt_ladder).

Cells: fb_n5_dt_ladder.CELLS (the four SM dt pairs, which include the m = 5
anchor (z0 = 8) and the boundary cell (3, 0.175, 0.5)).

Seeds: base 20260923, tag 95; replicate r of a cell uses entropy
fk.path_entropy(spec(dt=5e-4), seed, tag=95, replicate=r, chunk=25000) + [5000001];
chunk i main stream = SeedSequence(entropy).spawn(40)[i]; bridge stream =
SeedSequence(entropy + [7]).spawn(40)[i]; Philox.

Usage (from code/):
    python3 fb_hpc_n5full.py simulate --workers 140 [--cells ...] [--reps 0,1,2]
    python3 fb_hpc_n5full.py analyze
Chunk accumulators: $PRR_HPC_WORK/N5full/<cell>/rep<r>_chunk<i>.npz (outside repo);
summary: artifacts/data/exact_m_fixed_budget/HPC_N5full/hpc_n5full.json
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import fb_hpc_common as hc  # noqa: E402
import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402
import fb_n5_dt_ladder as n5  # noqa: E402

ITEM = "N5full"
TAG = hc.HPC_TAGS[ITEM]
H = n5.H_FINE
BIN = n5.BIN
TMAX = n5.TMAX
FACTORS = (1, 2, 4)
SCHEMES = ("em_r1", "em_r2", "em_r4", "ex_r1", "ex_r2", "ex_r4", "ex_b2", "ex_b4")
SCHEME_DT = {"em_r1": H, "em_r2": 2 * H, "em_r4": 4 * H, "ex_r1": H, "ex_r2": 2 * H,
             "ex_r4": 4 * H, "ex_b2": H / 2, "ex_b4": H / 4}
LADDERS = {  # (finest, middle, coarsest), each a factor 2 apart
    "em": ("em_r1", "em_r2", "em_r4"),
    "ex": ("ex_r1", "ex_r2", "ex_r4"),
    "exb": ("ex_b4", "ex_b2", "ex_r1"),
}
CHUNK = 25_000
BATCH = 2_500
N_PER_REP = 1_000_000
REPS = (0, 1, 2)
CELLS = n5.CELLS
GROUPS = 60


def _ou_bridge(step: float, g: float, sig2: float) -> tuple:
    """Midpoint of an OU bridge over an interval of length ``step`` (centred coords):
    Y_mid | y0, y1 ~ N(c (y0 + y1), s^2), c = a / (1 + a^2), s^2 = v / (1 + a^2),
    a = exp(-g step / 2), v = sig2 (1 - a^2) / (2 g)."""
    a = math.exp(-g * step / 2.0)
    v = sig2 * (1.0 - a * a) / (2.0 * g)
    return a / (1.0 + a * a), math.sqrt(v / (1.0 + a * a))


def run_chunk(task: dict) -> dict:
    cell = task["cell"]
    cfg = CELLS[cell]
    spec = n5.cell_spec(cell)                     # dt = h = 5e-4
    if spec.n_perp != 1:
        raise NotImplementedError("d = 2 only")
    n = int(task["size"])
    h = H
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
    bridges = bool(task.get("bridges", True))
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    rngb = np.random.Generator(np.random.Philox(task["seedseq_bridge"]))
    t0 = time.perf_counter()

    # initial law (draw order of fb_n5_dt_ladder.run_chunk / production simulator)
    z_init = spec.z0 + math.sqrt(spec.var_z0_scale * eps**2 * d0 / (2.0 * g)) * rng.standard_normal(n)
    rp_init = spec.r_par0 + eps * spec.u0 * rng.standard_normal(n)
    perp = np.mod(spec.r_perp0 + eps * spec.sigma_perp0 * rng.standard_normal(n), W)

    zx = z_init.copy()
    px = rp_init.copy()
    ax_ = math.exp(-g * h)
    sx_z = eps * math.sqrt(d0 * (1.0 - math.exp(-2.0 * g * h)) / (2.0 * g))
    sx_p = 2.0 * sx_z
    ze = {r: z_init.copy() for r in FACTORS}
    pe = {r: rp_init.copy() for r in FACTORS}
    acc_z = {r: np.zeros(n) for r in FACTORS if r > 1}
    acc_p = {r: np.zeros(n) for r in FACTORS if r > 1}
    nz_h = eps * math.sqrt(d0 * h)
    nr_h = 2.0 * eps * math.sqrt(d0 * h)
    # bridge constants: sigma^2 = eps^2 D0 (Z), 4 eps^2 D0 (R_par, R_perp)
    s2z, s2r = eps**2 * d0, 4.0 * eps**2 * d0
    cH_z, sH_z = _ou_bridge(h, g, s2z)
    cH_r, sH_r = _ou_bridge(h, g, s2r)
    cQ_z, sQ_z = _ou_bridge(h / 2, g, s2z)
    cQ_r, sQ_r = _ou_bridge(h / 2, g, s2r)
    sH_perp = math.sqrt(s2r * h / 4.0)
    sQ_perp = math.sqrt(s2r * (h / 2) / 4.0)

    X = {s: np.zeros(n) for s in SCHEMES}
    Xlast = {s: np.zeros(n) for s in SCHEMES}
    batch = int(task.get("batch", BATCH))
    nbatch = -(-n // batch)
    bid = np.arange(n) // batch
    bm = np.zeros((len(SCHEMES), nB, nbatch, n_bins))
    sq = np.zeros((len(SCHEMES), nB, n_bins))
    contact_frac = np.zeros(len(SCHEMES))

    def expo(F, d2):
        return n5._exposure(F, d2, spec, centres, w, a2)

    def wrapped_d2(par, perp_unwrapped):
        q = np.mod(perp_unwrapped, W)
        q = np.minimum(q, W - q)
        return par * par + q * q

    for k in range(1, steps + 1):
        xi = rng.standard_normal((3, n))
        perp_prev = perp.copy() if bridges else None
        incr = nr_h * xi[2]
        perp += incr
        np.mod(perp, W, out=perp)
        pm = np.minimum(perp, W - perp)
        pm2 = pm * pm
        if bridges:
            zprev = zx.copy()
            pprev = px.copy()
        zx -= zb
        zx *= ax_
        zx += zb + sx_z * xi[0]
        px *= ax_
        px += sx_p * xi[1]
        d2x = px * px + pm2
        act, e = expo(zx, d2x)
        if act.size:
            X["ex_r1"][act] += e
            if k % 2 == 0:
                X["ex_r2"][act] += 2.0 * e
            if k % 4 == 0:
                X["ex_r4"][act] += 4.0 * e
        if bridges:
            eta = rngb.standard_normal((9, n))
            # midpoint (h/2) given both endpoints
            zM = zb + cH_z * ((zprev - zb) + (zx - zb)) + sH_z * eta[0]
            pM = cH_r * (pprev + px) + sH_r * eta[1]
            qM = perp_prev + 0.5 * incr + sH_perp * eta[2]          # unwrapped
            actM, eM = expo(zM, wrapped_d2(pM, qM))
            # quarter points given (start, mid) and (mid, end)
            zQ1 = zb + cQ_z * ((zprev - zb) + (zM - zb)) + sQ_z * eta[3]
            pQ1 = cQ_r * (pprev + pM) + sQ_r * eta[4]
            qQ1 = perp_prev + 0.5 * (qM - perp_prev) + sQ_perp * eta[5]
            zQ3 = zb + cQ_z * ((zM - zb) + (zx - zb)) + sQ_z * eta[6]
            pQ3 = cQ_r * (pM + px) + sQ_r * eta[7]
            qQ3 = qM + 0.5 * (perp_prev + incr - qM) + sQ_perp * eta[8]
            act1, e1 = expo(zQ1, wrapped_d2(pQ1, qQ1))
            act3, e3 = expo(zQ3, wrapped_d2(pQ3, qQ3))
            if act.size:
                X["ex_b2"][act] += 0.5 * e
                X["ex_b4"][act] += 0.25 * e
            if actM.size:
                X["ex_b2"][actM] += 0.5 * eM
                X["ex_b4"][actM] += 0.25 * eM
            if act1.size:
                X["ex_b4"][act1] += 0.25 * e1
            if act3.size:
                X["ex_b4"][act3] += 0.25 * e3
        # EM r = 1
        ze[1] *= (1.0 - g * h)
        ze[1] += g * zb * h + nz_h * xi[0]
        pe[1] *= (1.0 - g * h)
        pe[1] += nr_h * xi[1]
        d2 = pe[1] * pe[1] + pm2
        act, e = expo(ze[1], d2)
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
                act, e = expo(ze[r], d2)
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
                contact_frac[:] = [float(np.mean(pe[r] * pe[r] + pm2 < a2)) for r in FACTORS] + \
                    [float(np.mean(px * px + pm2 < a2))] * 5
    runtime = time.perf_counter() - t0
    surv_tail = np.array([[float(np.exp(-B * X[s]).sum()) for B in budgets] for s in SCHEMES])
    fpath = Path(task["file"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, bm=bm, sq=sq, surv_tail_sum=surv_tail, n=n,
                        contact_frac_t2=contact_frac, schemes=np.array(SCHEMES),
                        bridges=bridges)
    os.replace(tmp, fpath)
    return {"cell": cell, "chunk": task["chunk"], "replicate": task["replicate"], "size": n,
            "runtime_seconds": runtime, "file": str(fpath), "host": os.uname().nodename}


def replicate_entropy(cell: str, replicate: int) -> list[int]:
    spec = n5.cell_spec(cell)
    return fk.path_entropy(spec, seed=hc.BASE_SEED, tag=TAG, replicate=replicate,
                           chunk=CHUNK) + [5_000_000 + 1]


def build_tasks(cells, reps, n_per_rep=N_PER_REP, chunk=CHUNK, size_override=None,
                bridges=True, force=False, root=None) -> list:
    root = hc.work_dir(ITEM) if root is None else Path(root)
    n_chunks = n_per_rep // chunk
    tasks = []
    for cell in cells:
        for r in reps:
            ent = replicate_entropy(cell, r)
            main = np.random.SeedSequence(ent).spawn(n_chunks)
            brid = np.random.SeedSequence(ent + [7]).spawn(n_chunks)
            for i in range(n_chunks):
                f = root / cell / f"rep{r}_chunk{i:03d}.npz"
                if f.exists() and not force:
                    continue
                tasks.append({"cell": cell, "replicate": r, "chunk": i,
                              "size": chunk if size_override is None else int(size_override),
                              "seedseq": main[i], "seedseq_bridge": brid[i], "file": str(f),
                              "entropy": ent, "bridges": bridges, "batch": BATCH})
    return tasks


def cmd_simulate(args):
    cells = list(CELLS) if not args.cells else args.cells.split(",")
    reps = [int(x) for x in args.reps.split(",")]
    tasks = build_tasks(cells, reps, force=args.force)
    # largest cells first (m = 5 is the slowest) for load balance
    order = {c: i for i, c in enumerate(["m5_z08_eps0.1", "m3_eps0.175", "m3_eps0.1"])}
    tasks.sort(key=lambda t: (order.get(t["cell"], 9), t["replicate"], t["chunk"]))
    print(f"[n5full] {len(tasks)} chunk tasks, env {hc.env_info()}", flush=True)
    deadline = time.time() + 60 * args.deadline_min if args.deadline_min else None
    logs = {}

    def on_result(r):
        logs.setdefault(r["cell"], []).append(r)

    t0 = time.time()
    hc.run_pool(run_chunk, tasks, args.workers, on_result, label="n5full", deadline=deadline)
    by_key = {(t["cell"], t["replicate"], t["chunk"]): t for t in tasks}
    for cell, rs in logs.items():
        log = hc.work_dir(ITEM) / cell / "runs.jsonl"
        with open(log, "a") as fh:
            for r in sorted(rs, key=lambda x: (x["replicate"], x["chunk"])):
                t = by_key[(cell, r["replicate"], r["chunk"])]
                rec = dict(r)
                rec["seed_entropy"] = t["entropy"]
                rec["spawn_key"] = list(t["seedseq"].spawn_key)
                fh.write(json.dumps(rec) + "\n")
    print(f"[n5full] simulate wall {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def load_cell(cell: str, reps=REPS):
    root = hc.work_dir(ITEM) / cell
    files = sorted(f for r in reps for f in root.glob(f"rep{r}_chunk*.npz"))
    if not files:
        return None
    bms, sqs, tails, ns, contact, rep_of_batch = [], [], [], [], [], []
    batch_sizes = set()
    runtimes = []
    for f in files:
        with np.load(f) as d:
            if tuple(str(x) for x in d["schemes"]) != SCHEMES:
                raise ValueError(f"{f}: scheme list differs")
            bms.append(d["bm"])
            sqs.append(d["sq"])
            tails.append(d["surv_tail_sum"])
            ns.append(int(d["n"]))
            contact.append(d["contact_frac_t2"])
            rep = int(f.name.split("_")[0][3:])
            rep_of_batch += [rep] * d["bm"].shape[2]
            if int(d["n"]) % d["bm"].shape[2]:
                raise ValueError(f"{f}: chunk size not a multiple of the batch count")
            batch_sizes.add(int(d["n"]) // d["bm"].shape[2])
    if len(batch_sizes) != 1:
        raise ValueError(f"mixed batch sizes {batch_sizes}")
    bm = np.concatenate(bms, axis=2)
    log = root / "runs.jsonl"
    if log.exists():
        runtimes = [json.loads(ln)["runtime_seconds"] for ln in log.read_text().splitlines() if ln.strip()]
    return {"files": [str(f) for f in files], "bm": bm, "sq": np.sum(sqs, axis=0),
            "tail": np.sum(tails, axis=0), "N": int(sum(ns)), "batch_size": int(batch_sizes.pop()),
            "contact_frac_t2": np.mean(contact, axis=0), "rep_of_batch": np.array(rep_of_batch),
            "chunk_runtime_seconds": {"n": len(runtimes), "mean": float(np.mean(runtimes)) if runtimes else None,
                                      "sum_core_hours": float(np.sum(runtimes) / 3600) if runtimes else None}}


def _rich(f1, f2, f4):
    """R1 = 2 f1 - f2 (first order), R2 = (8 f1 - 6 f2 + f4)/3 (first + second order)."""
    return 2 * f1 - f2, (8 * f1 - 6 * f2 + f4) / 3.0


def analyze_budget(cell: str, ib: int, L: dict, groups: int = GROUPS) -> dict:
    cfg = CELLS[cell]
    spec_prod = n5.cell_spec(cell, dt=1e-3)
    m = len(cfg["w"])
    B = cfg["budgets"][ib]
    lo, hi, edges = n5.window_bins()
    K = hi - lo
    A = fk._smoothing_operator(K, round(BIN, 14), 0.04)[2]
    valleys = fk.g_valley_times(spec_prod, cfg["w"])
    cuts = [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]]
    cuts_idx = [int(round((c - fk.WINDOW[0]) / BIN)) for c in cuts]
    after_time = valleys[-1]
    bm = L["bm"][:, ib, :, lo:hi]
    nb = bm.shape[1]
    N = L["N"]
    bs = L["batch_size"]
    gidx = (np.arange(nb) * groups) // nb
    Gs = np.zeros((len(SCHEMES), groups, K))
    for gi in range(groups):
        Gs[:, gi] = bm[:, gidx == gi].sum(1)
    gsize = np.bincount(gidx, minlength=groups) * bs
    tot = Gs.sum(1)
    P = tot / N
    sq = L["sq"][:, ib, lo:hi]
    se_bin = np.sqrt(np.maximum(sq / N - P * P, 0.0) / N)
    si = {s: i for i, s in enumerate(SCHEMES)}

    def funcs_for(Pmat, prot=True):
        return [n5.flatten(n5.functionals(Pmat[s], edges, cuts_idx, after_time, A, m, prot))
                for s in range(len(SCHEMES))]

    full = funcs_for(P)
    keys = list(full[0].keys())
    val = np.array([[f[k] for k in keys] for f in full])
    jack = np.zeros((groups,) + val.shape)
    for gi in range(groups):
        Pj = (tot - Gs[:, gi]) / (N - gsize[gi])
        fj = funcs_for(Pj, prot=True)
        jack[gi] = np.array([[f[k] for k in keys] for f in fj])
    # per-replicate (independent seeds) values
    reps = sorted(set(int(x) for x in L["rep_of_batch"]))
    rep_vals = {}
    for r in reps:
        msk = L["rep_of_batch"] == r
        Pr = bm[:, msk].sum(1) / (msk.sum() * bs)
        fr = funcs_for(Pr, prot=False)
        rep_vals[r] = {s: fr[si[s]] for s in SCHEMES}

    rows = {}
    for kk, k in enumerate(keys):
        v = {s: float(val[si[s], kk]) for s in SCHEMES}
        jv = {s: jack[:, si[s], kk] for s in SCHEMES}
        row = {"values": v, "se": {s: float(hc.jackknife_se(jv[s])) for s in SCHEMES}}
        rich = {}
        for fam, (s1, s2, s4) in LADDERS.items():
            R1, R2 = _rich(v[s1], v[s2], v[s4])
            J1, J2 = _rich(jv[s1], jv[s2], jv[s4])
            d21, d42 = v[s2] - v[s1], v[s4] - v[s2]
            row[f"{fam}_richardson1"] = R1
            row[f"{fam}_richardson1_se"] = float(hc.jackknife_se(J1))
            row[f"{fam}_richardson2"] = R2
            row[f"{fam}_richardson2_se"] = float(hc.jackknife_se(J2))
            row[f"{fam}_diff_mid_minus_fine"] = d21
            row[f"{fam}_diff_mid_minus_fine_se"] = float(hc.jackknife_se(jv[s2] - jv[s1]))
            row[f"{fam}_diff_coarse_minus_mid"] = d42
            row[f"{fam}_diff_coarse_minus_mid_se"] = float(hc.jackknife_se(jv[s4] - jv[s2]))
            row[f"{fam}_order_ratio"] = (d42 / d21) if d21 != 0 else None
            rich[fam] = (R1, R2, J1, J2)
        # production scheme (EM, dt = 1e-3) vs its first-order Richardson limit (N5 criterion)
        D = rich["em"][0] - v["em_r2"]
        Dj = rich["em"][2] - jv["em_r2"]
        row["prod_bias_D"] = D
        row["prod_bias_D_se"] = float(hc.jackknife_se(Dj))
        row["prod_value_ci95_halfwidth"] = 1.96 * row["se"]["em_r2"]
        row["accept_abs_D_lt_ci95_of_dt1e-3_value"] = bool(abs(D) < 1.96 * row["se"]["em_r2"])
        row["accept_D_ci_contains_0"] = bool(abs(D) < 1.96 * row["prod_bias_D_se"])
        # best continuum estimate: second-order limit of the bridge-refined exact ladder
        Db = rich["exb"][1] - v["em_r2"]
        Dbj = rich["exb"][3] - jv["em_r2"]
        row["prod_bias_vs_exb_R2"] = Db
        row["prod_bias_vs_exb_R2_se"] = float(hc.jackknife_se(Dbj))
        row["accept_abs_Dexb_lt_ci95_of_dt1e-3_value"] = bool(abs(Db) < 1.96 * row["se"]["em_r2"])
        # continuum-limit consistency between ladders (paired jackknife z)
        for a_, b_ in (("em", "ex"), ("em", "exb"), ("ex", "exb")):
            for o, lab in ((0, "richardson1"), (1, "richardson2")):
                C = rich[a_][o] - rich[b_][o]
                Cse = float(hc.jackknife_se(rich[a_][o + 2] - rich[b_][o + 2]))
                row[f"{a_}_vs_{b_}_{lab}_diff"] = C
                row[f"{a_}_vs_{b_}_{lab}_z"] = (C / Cse) if Cse > 0 else None
        # bridge correction at fixed step: ex_b2 - ex_r1 (sub-step contact episodes, dt 5e-4 -> 2.5e-4)
        row["bridge_correction_b2_minus_r1"] = v["ex_b2"] - v["ex_r1"]
        row["bridge_correction_b2_minus_r1_se"] = float(hc.jackknife_se(jv["ex_b2"] - jv["ex_r1"]))
        # independent-seed values of the production scheme and of the best limit
        row["per_seed"] = {str(r): {"em_r2": rep_vals[r]["em_r2"].get(k),
                                    "exb_R2": (8 * rep_vals[r]["ex_b4"].get(k, np.nan)
                                               - 6 * rep_vals[r]["ex_b2"].get(k, np.nan)
                                               + rep_vals[r]["ex_r1"].get(k, np.nan)) / 3.0
                                    if k in rep_vals[r]["ex_b4"] else None}
                           for r in reps}
        rows[k] = row
    return {"cell": cell, "label": cfg["labels"][ib], "B": B, "weights": list(cfg["w"]),
            "m": m, "N_paths": N, "groups": groups, "g_valleys": valleys,
            "cuts_idx": cuts_idx, "after_time": after_time,
            "bin_mass": {s: P[si[s]].tolist() for s in SCHEMES},
            "bin_se_iid": {s: se_bin[si[s]].tolist() for s in SCHEMES},
            "edges": edges.tolist(), "functionals": rows,
            "survival_t4": {s: float(L["tail"][si[s], ib] / N) for s in SCHEMES},
            "contact_fraction_t2": {s: float(L["contact_frac_t2"][si[s]]) for s in SCHEMES}}


def acceptance_summary(results: dict) -> dict:
    """N5 acceptance: Richardson-extrapolated basin masses, peak times and last-mode
    prominence differ from the dt = 1e-3 production values by < their 95% CI."""
    out = {}
    for cell, c in results["cells"].items():
        for lab, r in c["budgets"].items():
            F = r["functionals"]
            rows = {}
            for fam in ("basin_masses", "peak_times_smoothed", "late_rel_prominence_smoothed"):
                ks = [k for k in F if k.startswith(fam)]
                rows[fam] = {k: {"D_em_R1": F[k]["prod_bias_D"], "D_se": F[k]["prod_bias_D_se"],
                                 "ci95_halfwidth_dt1e-3": F[k]["prod_value_ci95_halfwidth"],
                                 "pass": F[k]["accept_abs_D_lt_ci95_of_dt1e-3_value"],
                                 "D_exb_R2": F[k]["prod_bias_vs_exb_R2"],
                                 "D_exb_R2_se": F[k]["prod_bias_vs_exb_R2_se"],
                                 "pass_vs_exb_R2": F[k]["accept_abs_Dexb_lt_ci95_of_dt1e-3_value"]}
                             for k in ks}
            out[lab] = rows
    n_all = sum(len(v2) for v in out.values() for v2 in v.values())
    n_pass = sum(x["pass"] for v in out.values() for v2 in v.values() for x in v2.values())
    n_pass_b = sum(x["pass_vs_exb_R2"] for v in out.values() for v2 in v.values() for x in v2.values())
    fam_counts = {}
    for fam in ("basin_masses", "peak_times_smoothed", "late_rel_prominence_smoothed"):
        xs = [x for v in out.values() for k2, v2 in v.items() if k2 == fam for x in v2.values()]
        fam_counts[fam] = {"n": len(xs), "pass_em_R1": sum(x["pass"] for x in xs),
                           "pass_vs_exb_R2": sum(x["pass_vs_exb_R2"] for x in xs)}
    return {"by_cell": out, "n_functionals": n_all, "n_pass_em_R1": n_pass,
            "n_pass_vs_exb_R2": n_pass_b, "by_family": fam_counts}


def cmd_analyze(args):
    out = hc.out_dir(ITEM)
    results = {"item": "HPC_N5full (GAP_CLOSURE_PLAN N5 Isambard specification)",
               "driver": HERE.name, "env": hc.env_info(),
               "schemes": {s: {"family": s[:2], "dt": SCHEME_DT[s]} for s in SCHEMES},
               "ladders": {k: [list(v), [SCHEME_DT[s] for s in v]] for k, v in LADDERS.items()},
               "bridge_correction": ("ex_b2/ex_b4: exact OU / Brownian bridge points at h/2 (and h/4, 3h/4) "
                                     "inside every fine step h = 5e-4, conditional on the exact endpoints; "
                                     "right-point exposure on the refined grid (resolves sub-step contact "
                                     "episodes); separate Philox stream"),
               "binning": "right-closed 0.02 bins (e_i, e_{i+1}], nested for all dt",
               "seed": {"base_seed": hc.BASE_SEED, "tag": TAG,
                        "entropy_rule": ("fk.path_entropy(spec(dt=5e-4), seed, tag=95, replicate, "
                                         "chunk=25000) + [5000001]; bridge stream entropy + [7]"),
                        "rng": "Philox; chunk i = SeedSequence(entropy).spawn(40)[i]"},
               "richardson": {"R1": "2 f(fine) - f(mid)", "R2": "(8 f(fine) - 6 f(mid) + f(coarse))/3"},
               "acceptance_reading": ("|R1_em - f_EM(1e-3)| < 1.96 SE of the dt=1e-3 value (as local N5); "
                                      "also vs the second-order bridge-refined exact limit exb_R2"),
               "cells": {}}
    for cell in CELLS:
        L = load_cell(cell)
        if L is None:
            continue
        cfg = CELLS[cell]
        results["cells"][cell] = {"N_paths": L["N"], "n_chunk_files": len(L["files"]),
                                  "chunk_runtime": L["chunk_runtime_seconds"], "budgets": {}}
        for ib, B in enumerate(cfg["budgets"]):
            results["cells"][cell]["budgets"][cfg["labels"][ib]] = analyze_budget(cell, ib, L)
            print(f"[n5full] analyzed {cell} B={B}", flush=True)
        del L
    bc = results["cells"].get("m3_eps0.175", {}).get("budgets", {}).get("m3_phase_boundary")
    if bc is not None:
        lo, hi, edges = n5.window_bins()
        ss = np.random.SeedSequence([hc.BASE_SEED, TAG, 777]).spawn(8)
        flip = {"stored_pair": None, "resampling": {}}
        try:
            flip["stored_pair"] = n5.stored_pair_reclassify()
        except Exception as exc:  # input JSON not synced
            flip["stored_pair"] = {"error": repr(exc)}
        k = 0
        for s in ("em_r2", "em_r1", "ex_r1", "ex_b4"):
            Pm = np.array(bc["bin_mass"][s])
            for Nw in (500_000, 1_000_000):
                flip["resampling"][f"{s}_N{Nw}"] = n5.protocol_resampling(Pm, edges, Nw, args.replicas,
                                                                          ss[k], 3)
                k += 1
        a = flip["resampling"]["em_r2_N500000"]["p_mode_count_eq_m"]
        b = flip["resampling"]["em_r1_N500000"]["p_mode_count_eq_m"]
        flip["p_observed_pattern_3_then_2_at_5e5"] = a * (1 - b)
        flip["seed"] = {"entropy": [hc.BASE_SEED, TAG, 777], "spawned": 8}
        results["boundary_flip"] = flip
    results["seed"]["replicate_entropies"] = {c: {str(r): replicate_entropy(c, r) for r in REPS}
                                              for c in CELLS}
    results["peak_time_bias_model"] = peak_bias_model(results)
    results["acceptance"] = acceptance_summary(results)
    results["local_N5_comparison"] = compare_local(results)
    hc.write_json(out / "hpc_n5full.json", results)
    print("[n5full] wrote", out / "hpc_n5full.json", flush=True)


def peak_bias_model(results: dict) -> dict:
    rows = []
    for cell, c in results["cells"].items():
        for lab, r in c["budgets"].items():
            for k, F in r["functionals"].items():
                if not k.startswith("peak_times_smoothed"):
                    continue
                tp = F["exb_richardson2"]
                rows.append({"cell": lab, "peak": k, "t_peak_exb_R2": tp,
                             "D_em_R1": F["prod_bias_D"], "D_em_R1_se": F["prod_bias_D_se"],
                             "D_em_vs_exb_R2": F["prod_bias_vs_exb_R2"],
                             "D_em_vs_exb_R2_se": F["prod_bias_vs_exb_R2_se"],
                             "predicted_dt2_1_plus_t": 0.5e-3 * (1.0 + tp),
                             "D_ex_r2_vs_exb_R2": F["exb_richardson2"] - F["values"]["ex_r2"],
                             "predicted_exactOU_dt2": 0.5e-3})
    if not rows:
        return {"rows": []}
    return {"rows": rows,
            "max_abs_D_em_vs_exb_minus_pred": max(abs(x["D_em_vs_exb_R2"] - x["predicted_dt2_1_plus_t"])
                                                  for x in rows),
            "max_abs_D_ex_vs_exb_minus_dt2": max(abs(x["D_ex_r2_vs_exb_R2"] - 0.5e-3) for x in rows)}


def compare_local(results: dict) -> dict:
    """HPC values vs the local N5 run (3 x 2e5, tag 84; independent paths)."""
    p = fk.FB_DATA / "N5" / "n5_dt_ladder.json"
    if not p.exists():
        return {"note": "local N5 JSON not present"}
    loc = json.loads(p.read_text())
    out = {}
    for cell, c in results["cells"].items():
        for lab, r in c["budgets"].items():
            lr = loc["cells"].get(cell, {}).get("budgets", {}).get(lab)
            if lr is None:
                continue
            rows = {}
            for k, F in r["functionals"].items():
                LF = lr["functionals"].get(k)
                if LF is None or not isinstance(F["values"]["em_r2"], (int, float)):
                    continue
                if k.startswith("protocol"):
                    continue
                zs = {}
                for s in ("em_r2", "ex_r2", "em_r1"):
                    se = math.hypot(F["se"][s], LF["se"][s])
                    zs[s] = (F["values"][s] - LF["values"][s]) / se if se > 0 else None
                rows[k] = {"hpc_em_r2": F["values"]["em_r2"], "local_em_r2": LF["values"]["em_r2"],
                           "z_independent": zs,
                           "hpc_em_R1": F["em_richardson1"], "local_em_R1": LF["em_richardson1"]}
            out[lab] = {"rows": rows,
                        "max_abs_z_em_r2": max((abs(v["z_independent"]["em_r2"]) for v in rows.values()
                                                if v["z_independent"]["em_r2"] is not None), default=None)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--cells", default=None)
    s.add_argument("--reps", default="0,1,2")
    s.add_argument("--workers", type=int, default=140)
    s.add_argument("--deadline-min", type=float, default=None)
    s.add_argument("--force", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--replicas", type=int, default=2000)
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
