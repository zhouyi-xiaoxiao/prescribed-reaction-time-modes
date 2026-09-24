#!/usr/bin/env python3
"""HPC item B (HPC_N3full): GPT-6 full specification of the N3 contact/field factorial.

GAP_CLOSURE_PLAN.md section 3 N3: "GPT-6's full specification adds exact OU
steps, 3 nested dt and 3 seeds x 1e6" (not run locally; the local N3 used the
production Euler--Maruyama chain at dt = 1e-3, 1e6 paths, one seed).

Arms (all on common paths per anchor, so every contrast is paired):
  anchors (m, eps) in {(2, 0.1), (3, 0.1)}, equal weights, B in {1, 4, 8, 20};
  14 kernel variants = field {pair midpoint Z; particle 1, Z + R_par/2} x
  gate {contact |R|_mi < a, a in {0.4 (production), 0.2, 0.15};
        deterministic mean contact c(t) = P(|R_t|_mi < a), same radii; no gate}
  (fb_n3_contact_factorial.factorial_variants()).
Time discretisation (levels, all on the same Brownian increments drawn at
h = 5e-4):
  ex_r1, ex_r2, ex_r4: EXACT Gaussian OU transitions for Z and R_par (R_perp is
      Brownian mod W, exact), subsampled at dt = 5e-4, 1e-3, 2e-3 (nested),
      end-of-step exposure (production kill convention);
  em_r2: the production Euler--Maruyama chain at dt = 1e-3 driven by the pair
      sums of the same increments (links to the local N3 numbers).
  Richardson limits of the exact ladder: R1 = 2 ex_r1 - ex_r2, R2 =
  (8 ex_r1 - 6 ex_r2 + ex_r4)/3, formed batch by batch (paired SEs).
Law: exact discrete-time FK law of each level on right-closed 0.02 bins
(e_i, e_{i+1}] nested for all dt; mean contact uses the continuous-time
c(t) (exact for the exact-OU chain at grid times; as in the local N3).

Seeds: base 20260923, tag 96; replicate r of anchor (m, eps) uses entropy
fk.path_entropy(EnsembleSpec(m, eps, dt=5e-4), seed, tag=96, replicate=r,
chunk=25000) + [6000001]; chunk i = SeedSequence(entropy).spawn(40)[i], Philox.

Usage (from code/):
    python3 fb_hpc_n3full.py simulate --workers 140
    python3 fb_hpc_n3full.py analyze
Chunk accumulators: $PRR_HPC_WORK/N3full/<anchor>/rep<r>_chunk<i>.npz;
summary: artifacts/data/exact_m_fixed_budget/HPC_N3full/hpc_n3full.json
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
import fb_n3_contact_factorial as n3  # noqa: E402

ITEM = "N3full"
TAG = hc.HPC_TAGS[ITEM]
H = 5e-4
TMAX = 4.0
BIN = 0.02
LEVELS = ("ex_r1", "ex_r2", "ex_r4", "em_r2")
LEVEL_R = {"ex_r1": 1, "ex_r2": 2, "ex_r4": 4, "em_r2": 2}
LEVEL_DT = {k: H * r for k, r in LEVEL_R.items()}
ANCHORS = {"m2_eps0.1": (2, 0.1), "m3_eps0.1": (3, 0.1)}
BUDGETS = (1.0, 4.0, 8.0, 20.0)
RADII = n3.RADII
VARIANT_STRINGS = tuple(n3.factorial_variants())
CHUNK = 25_000
BATCH = 2_500
N_PER_REP = 1_000_000
REPS = (0, 1, 2)


def anchor_spec(anchor: str) -> fk.EnsembleSpec:
    m, eps = ANCHORS[anchor]
    return fk.EnsembleSpec(m=m, eps=eps, dt=H, tmax=TMAX)


def variants_for(spec) -> list:
    return [fk.parse_variant(v, spec) for v in VARIANT_STRINGS]


def run_chunk(task: dict) -> dict:
    anchor = task["anchor"]
    spec = anchor_spec(anchor)
    n = int(task["size"])
    h = H
    steps = int(round(TMAX / h))
    per_bin = int(round(BIN / h))
    n_bins = steps // per_bin
    Bs = np.asarray(BUDGETS, float)
    nB = Bs.size
    centres = spec.centres()
    m = centres.size
    w = np.full(m, 1.0 / m)
    W = spec.torus_w
    g, zb, d0, eps = spec.gamma, spec.z_bar, spec.d0, spec.eps
    variants = variants_for(spec)
    nv = len(variants)
    nL = len(LEVELS)
    nch = nL * nv
    fields = ("mid", "p1")
    mean_c = {float(a): np.asarray(c) for a, c in task["mean_contact"].items()}
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()

    z_init = spec.z0 + math.sqrt(spec.var_z0_scale * eps**2 * d0 / (2.0 * g)) * rng.standard_normal(n)
    rp_init = spec.r_par0 + eps * spec.u0 * rng.standard_normal(n)
    perp = np.mod(spec.r_perp0 + eps * spec.sigma_perp0 * rng.standard_normal(n), W)
    zx = z_init.copy()
    px = rp_init.copy()
    ax_ = math.exp(-g * h)
    sx_z = eps * math.sqrt(d0 * (1.0 - math.exp(-2.0 * g * h)) / (2.0 * g))
    sx_p = 2.0 * sx_z
    ze = z_init.copy()
    pe = rp_init.copy()
    acc_z = np.zeros(n)
    acc_p = np.zeros(n)
    nz_h = eps * math.sqrt(d0 * h)
    nr_h = 2.0 * eps * math.sqrt(d0 * h)

    X = np.zeros((nch, n))
    Xlast = np.zeros((nch, n))
    batch = int(task.get("batch", BATCH))
    nb = n // batch
    if n % batch:
        raise ValueError("chunk size must be a multiple of the batch size")
    bid = np.arange(n) // batch
    P = np.zeros((nch, nB, n_bins))
    P2 = np.zeros((nch, nB, n_bins))
    Pb = np.zeros((nch, nB, nb, n_bins))
    EX = np.zeros((nch, n_bins + 1))
    tj_steps = [int(round(tj / h)) for tj in spec.times()]
    chi_code = np.zeros((len(RADII), n), np.int64)
    vfield = [v["field"] for v in variants]
    vgate = [v["gate"] for v in variants]
    va = [float(v["a"]) for v in variants]

    def gated(zpos, ppos, d2, k):
        """Per-variant (active indices, unit-budget exposure over one fine step h)."""
        comp = {}
        for fld in fields:
            F = zpos if fld == "mid" else zpos + fk.FIELD_SHIFT[fld] * ppos
            act, comps = fk._slab_components(F, centres, spec)
            comp[fld] = (act, fk._wsum(comps, w) if act.size else None)
        out = []
        for iv in range(nv):
            act, e = comp[vfield[iv]]
            if act.size:
                if vgate[iv] == "contact":
                    e = e * (d2[act] < va[iv] * va[iv])
                elif vgate[iv] == "mean":
                    e = e * mean_c[va[iv]][k - 1]
            out.append((act, e))
        return out

    def add(li, ge, r):
        for iv, (act, e) in enumerate(ge):
            if act.size:
                X[li * nv + iv, act] += r * e

    for k in range(1, steps + 1):
        xi = rng.standard_normal((3, n))
        perp += nr_h * xi[2]
        np.mod(perp, W, out=perp)
        pm = np.minimum(perp, W - perp)
        pm2 = pm * pm
        zx -= zb
        zx *= ax_
        zx += zb + sx_z * xi[0]
        px *= ax_
        px += sx_p * xi[1]
        d2x = px * px + pm2
        # exact levels share the positions at step k; level r contributes r*h exposure
        # at steps divisible by r.  Evaluate the kernel once and add with multipliers.
        ge = gated(zx, px, d2x, k)
        add(0, ge, 1.0)
        if k % 2 == 0:
            add(1, ge, 2.0)
        if k % 4 == 0:
            add(2, ge, 4.0)
        for jj, sj in enumerate(tj_steps):
            if k == sj:
                for ir, rr in enumerate(RADII):
                    chi_code[ir] += (d2x < rr * rr).astype(np.int64) << jj
        acc_z += xi[0]
        acc_p += xi[1]
        if k % 2 == 0:
            ze *= (1.0 - g * 2 * h)
            ze += g * zb * 2 * h + nz_h * acc_z
            pe *= (1.0 - g * 2 * h)
            pe += nr_h * acc_p
            acc_z[:] = 0.0
            acc_p[:] = 0.0
            add(3, gated(ze, pe, pe * pe + pm2, k), 2.0)
        if k % per_bin == 0:
            b = k // per_bin - 1
            for ch in range(nch):
                inc = X[ch] - Xlast[ch]
                nzr = np.flatnonzero(inc > 0)
                if nzr.size:
                    xl = Xlast[ch, nzr]
                    ic = inc[nzr]
                    bb = bid[nzr]
                    for ib in range(nB):
                        B = Bs[ib]
                        y = np.exp(-B * xl) * (-np.expm1(-B * ic))
                        P[ch, ib, b] += y.sum()
                        P2[ch, ib, b] += (y * y).sum()
                        Pb[ch, ib, :, b] += np.bincount(bb, weights=y, minlength=nb)
                EX[ch, b + 1] += X[ch].sum()
                Xlast[ch] = X[ch]
    npat = 1 << m
    chi_b = np.stack([np.bincount(bid * npat + chi_code[ir], minlength=nb * npat).reshape(nb, npat)
                      for ir in range(len(RADII))])
    surv = np.array([[float(np.exp(-B * X[ch]).sum()) for B in Bs] for ch in range(nch)])
    runtime = time.perf_counter() - t0
    fpath = Path(task["file"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, P=P, P2=P2, Pb=Pb, EX=EX, chi_b=chi_b, surv_tail_sum=surv, n=n, batch=batch,
                        levels=np.array(LEVELS), variants=np.array([v["name"] for v in variants]))
    os.replace(tmp, fpath)
    return {"anchor": anchor, "replicate": task["replicate"], "chunk": task["chunk"], "size": n,
            "runtime_seconds": runtime, "file": str(fpath), "host": os.uname().nodename}


def replicate_entropy(anchor: str, replicate: int) -> list[int]:
    return fk.path_entropy(anchor_spec(anchor), seed=hc.BASE_SEED, tag=TAG, replicate=replicate,
                           chunk=CHUNK) + [6_000_000 + 1]


def mean_contact_table(anchor: str) -> dict:
    spec = anchor_spec(anchor)
    T = fk.step_times(H, int(round(TMAX / H)))
    return {float(a): fk.mean_contact_curve(spec, a, T) for a in RADII}


def build_tasks(anchors, reps, n_per_rep=N_PER_REP, chunk=CHUNK, size_override=None, force=False):
    root = hc.work_dir(ITEM)
    n_chunks = n_per_rep // chunk
    tasks = []
    for an in anchors:
        mc = mean_contact_table(an)
        for r in reps:
            ent = replicate_entropy(an, r)
            ch = np.random.SeedSequence(ent).spawn(n_chunks)
            for i in range(n_chunks):
                f = root / an / f"rep{r}_chunk{i:03d}.npz"
                if f.exists() and not force:
                    continue
                tasks.append({"anchor": an, "replicate": r, "chunk": i,
                              "size": chunk if size_override is None else int(size_override),
                              "seedseq": ch[i], "file": str(f), "entropy": ent,
                              "mean_contact": mc, "batch": BATCH})
    return tasks


def cmd_simulate(args):
    anchors = list(ANCHORS) if not args.anchors else args.anchors.split(",")
    reps = [int(x) for x in args.reps.split(",")]
    tasks = build_tasks(anchors, reps, force=args.force)
    tasks.sort(key=lambda t: (-ANCHORS[t["anchor"]][0], t["replicate"], t["chunk"]))
    print(f"[n3full] {len(tasks)} chunk tasks, env {hc.env_info()}", flush=True)
    deadline = time.time() + 60 * args.deadline_min if args.deadline_min else None
    logs = {}
    t0 = time.time()
    hc.run_pool(run_chunk, tasks, args.workers, lambda r: logs.setdefault(r["anchor"], []).append(r),
                label="n3full", deadline=deadline)
    by_key = {(t["anchor"], t["replicate"], t["chunk"]): t for t in tasks}
    for an, rs in logs.items():
        with open(hc.work_dir(ITEM) / an / "runs.jsonl", "a") as fh:
            for r in sorted(rs, key=lambda x: (x["replicate"], x["chunk"])):
                t = by_key[(an, r["replicate"], r["chunk"])]
                rec = dict(r)
                rec["seed_entropy"] = t["entropy"]
                rec["spawn_key"] = list(t["seedseq"].spawn_key)
                fh.write(json.dumps(rec) + "\n")
    print(f"[n3full] simulate wall {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def load_anchor(anchor: str) -> dict | None:
    root = hc.work_dir(ITEM) / anchor
    files = sorted(root.glob("rep*_chunk*.npz"))
    if not files:
        return None
    acc = None
    Pbs, chis, reps = [], [], []
    N = 0
    for f in files:
        with np.load(f) as d:
            if acc is None:
                acc = {k: d[k].astype(float).copy() for k in ("P", "P2", "EX", "surv_tail_sum")}
                names = [str(x) for x in d["variants"]]
                levels = [str(x) for x in d["levels"]]
            else:
                for k in ("P", "P2", "EX", "surv_tail_sum"):
                    acc[k] += d[k]
            Pbs.append(d["Pb"])
            bsz = int(d["batch"])
            chis.append(d["chi_b"])
            N += int(d["n"])
            reps += [int(f.name.split("_")[0][3:])] * d["Pb"].shape[2]
    if tuple(levels) != LEVELS:
        raise ValueError("level list differs")
    runtimes = []
    log = root / "runs.jsonl"
    if log.exists():
        runtimes = [json.loads(ln)["runtime_seconds"] for ln in log.read_text().splitlines() if ln.strip()]
    return {"P": acc["P"], "P2": acc["P2"], "EX": acc["EX"], "surv": acc["surv_tail_sum"], "batch": bsz,
            "Pb": np.concatenate(Pbs, axis=2), "chi_b": np.concatenate(chis, axis=1),
            "N": N, "names": names, "rep_of_batch": np.array(reps), "files": len(files),
            "runtime": {"n_chunks_logged": len(runtimes),
                        "core_hours": float(np.sum(runtimes) / 3600) if runtimes else None,
                        "mean_chunk_s": float(np.mean(runtimes)) if runtimes else None}}


EDGES_ALL = BIN * np.arange(int(round(TMAX / BIN)) + 1)
WLO = int(round(fk.WINDOW[0] / BIN))
WHI = int(round(fk.WINDOW[1] / BIN))


def channel(D: dict, level: str, var: str, ib: int):
    """(bin masses over all 200 bins, batch bin masses (nb, 200), E X at edges) for a level
    or a Richardson combination ('R1', 'R2') of the exact ladder."""
    iv = D["names"].index(var)
    nv = len(D["names"])
    comb = {"R1": {"ex_r1": 2.0, "ex_r2": -1.0},
            "R2": {"ex_r1": 8 / 3, "ex_r2": -2.0, "ex_r4": 1 / 3}}.get(level, {level: 1.0})
    N, bs = D["N"], D["batch"]
    P = sum(c * D["P"][LEVELS.index(lv) * nv + iv, ib] for lv, c in comb.items()) / N
    Pb = sum(c * D["Pb"][LEVELS.index(lv) * nv + iv, ib] for lv, c in comb.items()) / bs
    EX = sum(c * D["EX"][LEVELS.index(lv) * nv + iv] for lv, c in comb.items()) / N
    return P, Pb, EX


def law_window(P, Pb):
    edges = EDGES_ALL[WLO:WHI + 1]
    mass = P[WLO:WHI]
    bmass = Pb[:, WLO:WHI]
    bw = np.diff(edges)
    return {"t": 0.5 * (edges[:-1] + edges[1:]), "edges": edges, "bin_mass": mass,
            "batch_mass": bmass, "density": mass / bw, "batch_density": bmass / bw[None, :],
            "cov": np.cov(bmass, rowvar=False) / bmass.shape[0]}


def basins(P, Pb, EX, B, cut_times):
    ci = [int(round(c / BIN)) for c in cut_times]
    M = np.array([P[a:b].sum() for a, b in zip(ci[:-1], ci[1:])])
    Mb = np.array([Pb[:, a:b].sum(1) for a, b in zip(ci[:-1], ci[1:])]).T
    se = Mb.std(axis=0, ddof=1) / math.sqrt(Mb.shape[0])
    L = B * EX[ci]
    mf = np.exp(-L[:-1]) - np.exp(-L[1:])
    return {"cuts": [float(EDGES_ALL[k]) for k in ci], "masses": M, "se_batch": se,
            "batch_masses": Mb, "mean_field": mf}


def frozen_surrogate(D, spec, radius, B, w):
    m = spec.centres().size
    ir = list(RADII).index(radius)
    cb = D["chi_b"][ir].astype(float)
    lam = fk.lambdas(B, w, spec)
    pats = np.arange(1 << m)
    chi = ((pats[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    ex = chi * lam[None, :]
    cum = np.concatenate([np.zeros((pats.size, 1)), np.cumsum(ex, axis=1)], axis=1)
    Mpat = np.exp(-cum[:, :-1]) * (1.0 - np.exp(-ex))
    tot = cb.sum(0) / cb.sum()
    return {"masses": tot @ Mpat, "contact_prob_tj": (tot @ chi).tolist()}


def classify(law):
    cls = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000)
    clsc = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000, cov=law["cov"])
    sig = [{"time": r["time"], "relative_prominence": r["relative_prominence"],
            "z_protocol_1e6": r["z"], "z_fk_estimate": rc["z"],
            "vertex": n3._vertex(law["t"], law["density"], r["time"], BIN)}
           for r, rc in zip(cls["rows"], clsc["rows"]) if r["significant"] or rc["significant"]]
    return cls["mode_count"], clsc["mode_count"], sig


def summarize(D, spec, level, var, ib, cuts_win, cuts_ext, w, vinfo, full=True):
    B = BUDGETS[ib]
    P, Pb, EX = channel(D, level, var, ib)
    bw = basins(P, Pb, EX, B, cuts_win)
    be = basins(P, Pb, EX, B, cuts_ext)
    out = {"level": level, "variant": var, "B": B,
           "basins_window": {"cuts": bw["cuts"], "masses": bw["masses"], "se_batch": bw["se_batch"],
                             "mean_field": bw["mean_field"],
                             "abs_err_mean_field_sum": float(np.abs(bw["masses"] - bw["mean_field"]).sum())},
           "basins_extended": {"cuts": be["cuts"], "masses": be["masses"], "se_batch": be["se_batch"],
                               "mean_field": be["mean_field"],
                               "abs_err_mean_field_sum": float(np.abs(be["masses"] - be["mean_field"]).sum())}}
    lam, prod = fk.limit_masses(B, w, spec)
    out["product_limit_masses"] = prod
    out["abs_err_product_sum_extended"] = float(np.abs(be["masses"] - prod).sum())
    if vinfo["gate"] == "contact" and float(vinfo["a"]) in RADII:
        fs = frozen_surrogate(D, spec, float(vinfo["a"]), B, w)
        out["frozen_surrogate_masses"] = fs["masses"]
        out["contact_prob_tj"] = fs["contact_prob_tj"]
        out["abs_err_frozen_surrogate_sum_extended"] = float(np.abs(be["masses"] - fs["masses"]).sum())
    if full:
        law = law_window(P, Pb)
        mc, mcf, sig = classify(law)
        der = fk.derivative_signs(law["t"], law["density"], law["batch_density"], zthr=5.0)
        out.update({"mode_count_protocol_1e6": mc, "mode_count_fk_estimate": mcf,
                    "significant_peaks": sig,
                    "derivative_census_z5": {"n_maxima": der["n_maxima"], "n_minima": der["n_minima"],
                                             "maxima_t": der["maxima_t"], "minima_t": der["minima_t"]},
                    "window_mass": float(law["bin_mass"].sum())})
    return out


def contrast(D, level, v1, v2, ib, cuts):
    B = BUDGETS[ib]
    P1, Pb1, EX1 = channel(D, level, v1, ib)
    P2, Pb2, EX2 = channel(D, level, v2, ib)
    b1 = basins(P1, Pb1, EX1, B, cuts)
    b2 = basins(P2, Pb2, EX2, B, cuts)
    d = b1["batch_masses"] - b2["batch_masses"]
    diff = b1["masses"] - b2["masses"]
    se = d.std(axis=0, ddof=1) / math.sqrt(d.shape[0])
    L1 = float(np.abs(P1[WLO:WHI] - P2[WLO:WHI]).sum())
    return {"level": level, "minus": [v1, v2], "B": B, "basin_mass_diff": diff, "se_batch": se,
            "ci95": [(diff - 1.96 * se).tolist(), (diff + 1.96 * se).tolist()],
            "significant_any_basin": bool(np.any(np.abs(diff) > 1.96 * se)),
            "window_L1_distance": L1}


def dt_effect(D, var, ib, cuts):
    """Paired dt effects on basin masses: ex_r2 - R1 (production dt, exact OU),
    em_r2 - ex_r2 (EM vs exact at dt = 1e-3), R2 - R1."""
    B = BUDGETS[ib]
    res = {}
    vals = {}
    for lv in ("ex_r4", "ex_r2", "ex_r1", "R1", "R2", "em_r2"):
        P, Pb, EX = channel(D, lv, var, ib)
        vals[lv] = basins(P, Pb, EX, B, cuts)
    for a_, b_ in (("ex_r2", "R1"), ("em_r2", "ex_r2"), ("em_r2", "R1"), ("R2", "R1"), ("ex_r4", "ex_r2")):
        d = vals[a_]["batch_masses"] - vals[b_]["batch_masses"]
        diff = vals[a_]["masses"] - vals[b_]["masses"]
        se = d.std(axis=0, ddof=1) / math.sqrt(d.shape[0])
        res[f"{a_}_minus_{b_}"] = {"diff": diff, "se": se,
                                   "max_abs_z": float(np.max(np.abs(diff) / np.maximum(se, 1e-300)))}
    res["masses_by_level"] = {lv: vals[lv]["masses"] for lv in vals}
    res["se_by_level"] = {lv: vals[lv]["se_batch"] for lv in vals}
    return res


def s1_widths(D, spec, w):
    """Particle-1 vs midpoint self-centred widths at B = 1 (ex_r2 and R1), vs the S1 record."""
    p = core.UPGRADE_DATA / "w6_single_particle" / "width_diagnostic_selfcentred.json"
    m = spec.centres().size
    key = {2: "m2_eps0.1_B1_p1.json", 3: "m3_eps0.1_B1_p1.json"}[m]
    if not p.exists():
        return {"note": "S1 width diagnostic not present"}
    wd = json.loads(p.read_text())["cells"][key]
    out = {"s1_record": key, "rows": {}}
    ib = BUDGETS.index(1.0)
    for lv in ("ex_r2", "R1", "em_r2"):
        Pp, Pbp, _ = channel(D, lv, n3.vname("contact", 0.4, "p1"), ib)
        Pm, Pbm, _ = channel(D, lv, n3.vname("contact", 0.4, "mid"), ib)
        t = 0.5 * (EDGES_ALL[:-1] + EDGES_ALL[1:])
        rows = []
        nb = Pbp.shape[0]
        grp = np.array_split(np.arange(nb), 20)
        for row in wd:
            tp0, hw = row["t_G1"], row["hw"]
            try:
                s1, v1 = n3._selfc(t, Pp, tp0, hw)
                s0, v0 = n3._selfc(t, Pm, tp0, hw)
                rb, sb = [], []
                for g_ in grp:
                    a1, b1 = n3._selfc(t, Pbp[g_].mean(0), tp0, hw)
                    a0, b0 = n3._selfc(t, Pbm[g_].mean(0), tp0, hw)
                    rb.append(a1 / a0)
                    sb.append(b1 - b0)
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                rows.append({"t_G1": tp0, "hw": hw, "error": repr(exc)})
                continue
            rows.append({"t_G1": tp0, "hw": hw, "ratio": s1 / s0,
                         "ratio_se": float(np.std(rb, ddof=1) / math.sqrt(len(grp))),
                         "vertex_shift": v1 - v0,
                         "vertex_shift_se": float(np.std(sb, ddof=1) / math.sqrt(len(grp))),
                         "s1_ratio": row["ratio"], "s1_ratio_se": row["se_ratio"],
                         "s1_vertex_shift": row["vertex_shift"], "s1_shift_se": row["se_shift"]})
        out["rows"][lv] = rows
    return out


def local_n3_link(D, anchor, cuts_win):
    """em_r2 (production EM, dt = 1e-3, tag 96) vs the local N3 anchors (EM dt = 1e-3, tag 82,
    independent paths, production float-edge grid): basin-mass z per variant and B."""
    p = fk.FB_DATA / "N3" / "n3_summary.json"
    if not p.exists():
        return {"note": "local n3_summary.json not present"}
    loc = json.loads(p.read_text())["anchors"].get(anchor.split("_")[0])
    if loc is None:
        return {}
    rows = []
    for c in loc["cells"]:
        if c["variant"] not in D["names"] or float(c["B"]) not in BUDGETS:
            continue
        ib = BUDGETS.index(float(c["B"]))
        P, Pb, EX = channel(D, "em_r2", c["variant"], ib)
        b = basins(P, Pb, EX, c["B"], c["basin_cuts"])
        z = (b["masses"] - np.asarray(c["basin_masses"])) / np.hypot(b["se_batch"], c["basin_se_batch"])
        rows.append({"variant": c["variant"], "B": c["B"], "hpc_em_r2": b["masses"],
                     "local": c["basin_masses"], "z": z, "local_mode_count_protocol_1e6":
                     c["mode_count_protocol_1e6"]})
    return {"rows": rows, "max_abs_z": max(float(np.max(np.abs(r["z"]))) for r in rows) if rows else None,
            "n_abs_z_gt_3": sum(int(np.sum(np.abs(r["z"]) > 3)) for r in rows),
            "n_basins": sum(len(r["z"]) for r in rows),
            "note": ("independent path sets; local grid uses production float edges, HPC nested "
                     "right-closed bins (differ by <= one step of mass per edge)")}


def analyze_anchor(anchor: str) -> dict:
    D = load_anchor(anchor)
    if D is None:
        return None
    spec = anchor_spec(anchor)
    m = spec.centres().size
    w = np.full(m, 1.0 / m)
    vinfo = {v["name"]: v for v in variants_for(spec)}
    spec_prod = fk.EnsembleSpec(m=m, eps=spec.eps)
    valleys = fk.g_valley_times(spec_prod, tuple(w))
    cuts_win = [fk.WINDOW[0]] + [round(v / BIN) * BIN for v in valleys] + [fk.WINDOW[1]]
    import fb_n1_allocation_design as n1
    geo = n1.geometric_cuts(spec_prod)
    cuts_ext = [0.0] + [round(v / BIN) * BIN for v in geo] + [TMAX]
    out = {"anchor": anchor, "m": m, "eps": spec.eps, "N_paths": D["N"], "chunk_files": D["files"],
           "batches": int(D["Pb"].shape[2]), "batch_size": D["batch"], "runtime": D["runtime"],
           "cuts_window": cuts_win, "cuts_extended": cuts_ext, "g_valleys": valleys,
           "seed": {"base_seed": hc.BASE_SEED, "tag": TAG,
                    "replicate_entropies": {str(r): replicate_entropy(anchor, r) for r in REPS}},
           "variants": D["names"], "budgets": list(BUDGETS), "levels": list(LEVELS) + ["R1", "R2"],
           "cells": [], "contrasts": [], "dt_effects": []}
    for ib, B in enumerate(BUDGETS):
        for var in D["names"]:
            for lv in ("ex_r2", "R1", "em_r2", "ex_r1", "ex_r4", "R2"):
                out["cells"].append(summarize(D, spec, lv, var, ib, cuts_win, cuts_ext, w, vinfo[var],
                                              full=lv in ("ex_r2", "R1", "em_r2", "R2")))
            out["dt_effects"].append({"variant": var, "B": B, "window": dt_effect(D, var, ib, cuts_win),
                                      "extended": dt_effect(D, var, ib, cuts_ext)})
        for lv in ("ex_r2", "R1", "em_r2"):
            for field in ("mid", "p1"):
                for a in RADII:
                    c, mn = n3.vname("contact", a, field), n3.vname("mean", a, field)
                    out["contrasts"].append(dict(kind="contact_contribution(full-meancontact)",
                                                 **contrast(D, lv, c, mn, ib, cuts_win)))
                    out["contrasts"].append(dict(kind="gate_effect(contact-nogate)",
                                                 **contrast(D, lv, c, n3.vname("none", 0.4, field), ib,
                                                            cuts_win)))
            for a in RADII:
                out["contrasts"].append(dict(kind="field_effect(p1-mid)",
                                             **contrast(D, lv, n3.vname("contact", a, "p1"),
                                                        n3.vname("contact", a, "mid"), ib, cuts_win)))
        print(f"[n3full] {anchor} B={B} analyzed", flush=True)
    out["s1_widths_B1"] = s1_widths(D, spec, w)
    out["local_n3_link"] = local_n3_link(D, anchor, cuts_win)
    # compact headline tables
    heads = []
    for c in out["cells"]:
        if c["level"] in ("ex_r2", "R1", "em_r2") and "mode_count_protocol_1e6" in c:
            heads.append({"variant": c["variant"], "B": c["B"], "level": c["level"],
                          "modes_protocol_1e6": c["mode_count_protocol_1e6"],
                          "modes_fk": c["mode_count_fk_estimate"],
                          "deriv_maxima": c["derivative_census_z5"]["n_maxima"],
                          "basins_window": c["basins_window"]["masses"]})
    out["mode_count_table"] = heads
    return out


def cmd_analyze(args):
    res = {"item": "HPC_N3full (GAP_CLOSURE_PLAN N3, GPT-6 full specification)",
           "driver": HERE.name, "env": hc.env_info(),
           "levels": {k: {"dt": LEVEL_DT[k], "scheme": "exact OU" if k.startswith("ex") else "Euler-Maruyama"}
                      for k in LEVELS},
           "richardson": {"R1": "2 ex_r1 - ex_r2", "R2": "(8 ex_r1 - 6 ex_r2 + ex_r4)/3"},
           "variants": list(VARIANT_STRINGS), "budgets": list(BUDGETS),
           "binning": "right-closed 0.02 bins nested for all dt; window basins [0.5, G valleys, 3.5] "
                      "(local N3 convention), extended basins [0, geometric cuts, 4] (paper convention)",
           "anchors": {}}
    for an in ANCHORS:
        r = analyze_anchor(an)
        if r is not None:
            res["anchors"][an] = r
    hc.write_json(hc.out_dir(ITEM) / "hpc_n3full.json", res)
    print("[n3full] wrote", hc.out_dir(ITEM) / "hpc_n3full.json", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--anchors", default=None)
    s.add_argument("--reps", default="0,1,2")
    s.add_argument("--workers", type=int, default=140)
    s.add_argument("--deadline-min", type=float, default=None)
    s.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
