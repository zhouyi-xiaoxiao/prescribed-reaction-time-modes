#!/usr/bin/env python3
"""N1 -- fixed-budget allocation-design campaign (GAP_CLOSURE_PLAN.md §3 N1; tests TH-4).

Question: at a FIXED total integrated reactivity B, does re-allocating the budget
across the m slabs with the inverse-design law of TH-4

    lambda_j = ln(S_j / S_{j+1}),   S_j = 1 - sum_{i<j} p_j,
    w_j proportional to |mu'(t_j)| lambda_j        (W^{d-1} = 1 here)

produce the prescribed basin masses in the exact finite-noise law, and does it
keep all m modes where the equal allocation loses one?

Cells: m in {2,3} x eps in {0.05, 0.1, 0.15} x B in {0.5, 1, 2, 4, 8}, each under
three allocations: equal weights; the max-min (equal-mass) design p*(B,m);
one unequal target SHAPE r = (0.2, 0.8) (m=2) / (0.5, 0.3, 0.2) (m=3), scaled by
the largest common factor theta*(B, r) attainable at that budget
(fb_allocation_law.max_common_scale; the targets theta* r sum to < 1).  Plus
m = 5 in the W4 stretched geometry (z0 = 8, centres 2.8..0.4) at eps = 0.1,
B in {1, 2, 4}.

Method: one Feynman--Kac stored ensemble of 5e5 unkilled paths per (m, eps)
(exact_m_prr_fk_exact_law, seed tag 81 = N1); every (B, allocation) is a
reweighting of the same paths.  Confirmation: direct-kill simulator at 1e6
walkers (core.simulate_chunk_general, seed tag 81) for 10 designed cells.

Subcommands
-----------
  simulate   build/resume the FK ensembles (<= 3 workers; waits for CPU slot)
  directkill run the direct-kill confirmation cells (<= 3 workers)
  analyze    evaluate every cell -> artifacts/data/exact_m_fixed_budget/N1/n1_allocation_design.json
  figure     fb_n1_* figures from the analysis JSON

Seeds: base 20260923, tag 81; FK entropy per exact_m_prr_fk_exact_law.path_entropy,
direct-kill entropy [20260923, 81, m, eps*1e9, B*1e9, round(w_j*1e9)..., z0*1e9, walkers].
All seeds / entropies are recorded in the output JSONs.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

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
import fb_allocation_law as alloc  # noqa: E402

OUT_DIR = fk.FB_DATA / "N1"
OUT_JSON = OUT_DIR / "n1_allocation_design.json"
DK_DIR = OUT_DIR / "directkill"
TAG = fk.TAGS["N1"]  # 81
SEED = fk.BASE_SEED

EPS_LIST = (0.05, 0.1, 0.15)
M_LIST = (2, 3)
B_LIST = (0.5, 1.0, 2.0, 4.0, 8.0)
SHAPES = {2: (0.2, 0.8), 3: (0.5, 0.3, 0.2)}
M5_CENTRES = (2.8, 2.2, 1.6, 1.0, 0.4)
M5_Z0 = 8.0
M5_EPS = 0.1
M5_B = (1.0, 2.0, 4.0)
N_PATHS = 500_000

# direct-kill confirmation cells: (m, eps, B, allocation)
DK_CELLS = [
    (2, 0.1, 8.0, "maxmin"),   # Opus demonstration cell (w* = (0.1275, 0.8725))
    (3, 0.1, 4.0, "maxmin"),   # Opus demonstration cell (w* = (0.1822, 0.1399, 0.6779))
    (2, 0.05, 2.0, "maxmin"), (2, 0.05, 8.0, "maxmin"),
    (2, 0.1, 2.0, "maxmin"),
    (3, 0.05, 2.0, "maxmin"), (3, 0.05, 8.0, "maxmin"),
    (3, 0.1, 2.0, "maxmin"), (3, 0.1, 8.0, "maxmin"),
    (5, 0.1, 2.0, "maxmin"),
]


# ---------------------------------------------------------------------------
# Cells, specs and designs
# ---------------------------------------------------------------------------


def ens_name(m: int, eps: float) -> str:
    return f"n1_m5_z08_eps{eps:g}" if m == 5 else f"n1_m{m}_eps{eps:g}"


def spec_for(m: int, eps: float) -> fk.EnsembleSpec:
    if m == 5:
        return fk.EnsembleSpec(m=5, eps=eps, z0=M5_Z0, centres_z=M5_CENTRES)
    return fk.EnsembleSpec(m=m, eps=eps)


def ensemble_list():
    out = [(m, e) for m in M_LIST for e in EPS_LIST]
    out.append((5, M5_EPS))
    return out


def speeds_for(spec: fk.EnsembleSpec) -> list[float]:
    return [float(v) for v in spec.mu_prime_abs(np.asarray(spec.times()))]


def area_for(spec: fk.EnsembleSpec) -> float:
    return float(spec.torus_w ** spec.n_perp)


def design(spec: fk.EnsembleSpec, B: float, allocation: str) -> dict:
    """Weights, limit-law exposures and target masses for one allocation."""
    v = speeds_for(spec)
    A = area_for(spec)
    m = len(v)
    if allocation == "equal":
        w = [1.0 / m] * m
        lam = alloc.exposures(B, w, v, A)
        target = alloc.masses_from_exposures(lam)
        extra = {}
    elif allocation == "maxmin":
        res = alloc.p_star(B, v, A)
        w = res["weights"]
        lam = res["exposures"]
        target = res["masses"]
        extra = {"p_star": res["p_star"], "theta": res["theta"]}
    elif allocation == "shape":
        shape = SHAPES[m]
        res = alloc.max_common_scale(B, list(shape), v, A)
        w = res["weights"]
        lam = res["exposures"]
        target = res["masses"]
        extra = {"shape": list(shape), "theta": res["theta"]}
    else:
        raise ValueError(allocation)
    # forward check: the stick-breaking law of (B, w) reproduces the target
    fwd = alloc.masses(B, w, v, A)
    if max(abs(a - b) for a, b in zip(fwd, target)) > 1e-9:
        raise AssertionError("inverse design does not reproduce its target")
    return {"allocation": allocation, "B": float(B), "w": [float(x) for x in w],
            "lambda": [float(x) for x in lam], "target": [float(x) for x in target],
            "speeds": v, **extra}


def cell_list():
    cells = []
    for m in M_LIST:
        for e in EPS_LIST:
            for B in B_LIST:
                for a in ("equal", "maxmin", "shape"):
                    cells.append((m, e, B, a))
    for B in M5_B:
        for a in ("equal", "maxmin"):
            cells.append((5, M5_EPS, B, a))
    return cells


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def cmd_simulate(args) -> None:
    todo = ensemble_list()
    if args.only:
        keep = set(args.only.split(","))
        todo = [(m, e) for (m, e) in todo if ens_name(m, e) in keep]
    for m, e in todo:
        spec = spec_for(m, e)
        name = ens_name(m, e)
        subset = None
        if args.chunks:
            a, b = (int(x) for x in args.chunks.split(":"))
            subset = list(range(a, b))
        t0 = time.time()
        ens = fk.simulate_ensemble(spec, int(args.paths), name=name, tag=TAG, seed=SEED,
                                   variants=("full",), workers=args.workers,
                                   chunks_subset=subset,
                                   note="N1 allocation-design ensemble (tag 81)")
        print(f"[n1] {name}: {ens.n_paths} paths stored, complete="
              f"{ens.index.get('complete')}, wall {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# direct kill
# ---------------------------------------------------------------------------


def _dk_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    p = spec.model()
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    out = core.simulate_chunk_general(
        rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]),
        weights=tuple(task["w"]), centres_z=spec.centres(), dt=spec.dt,
        step_count=spec.steps(), p=p, n_perp=spec.n_perp)
    return {"chunk": task["chunk"], "kill_times": out["kill_times"],
            "survivors": out["survivors"], "kill_probability_max": out["kill_probability_max"],
            "walker_steps": out["walker_steps"]}


def dk_entropy(m, eps, B, w, z0, walkers):
    q = lambda x: int(round(float(x) * 1e9)) % (1 << 63)  # noqa: E731
    return [SEED, TAG, int(m), q(eps), q(B)] + [q(x) for x in w] + [q(z0), int(walkers)]


def cmd_directkill(args) -> None:
    import multiprocessing as mp
    DK_DIR.mkdir(parents=True, exist_ok=True)
    cells = DK_CELLS
    if args.only:
        idx = [int(x) for x in args.only.split(",")]
        cells = [DK_CELLS[i] for i in idx]
    walkers = int(float(args.walkers))
    chunk = int(float(args.chunk))
    ctx = mp.get_context("spawn")
    for (m, eps, B, a) in cells:
        spec = spec_for(m, eps)
        d = design(spec, B, a)
        tagname = f"dk_m{m}_eps{eps:g}_B{B:g}_{a}"
        outp = DK_DIR / f"{tagname}.json"
        if outp.exists() and not args.overwrite:
            print(f"[dk] {tagname} exists, skip", flush=True)
            continue
        fk.wait_for_cpu_slot()
        ent = dk_entropy(m, eps, B, d["w"], spec.z0, walkers)
        sizes = [chunk] * (walkers // chunk) + ([walkers % chunk] if walkers % chunk else [])
        children = np.random.SeedSequence(ent).spawn(len(sizes))
        tasks = [{"spec": spec.to_dict(), "size": s, "seedseq": c, "B": B, "w": d["w"],
                  "chunk": i} for i, (s, c) in enumerate(zip(sizes, children))]
        t0 = time.time()
        with ctx.Pool(processes=min(args.workers, fk.MAX_WORKERS)) as pool:
            res = sorted(pool.map(_dk_chunk, tasks), key=lambda r: r["chunk"])
        kt = np.concatenate([r["kill_times"] for r in res])
        surv = int(sum(r["survivors"] for r in res))
        assert kt.size + surv == walkers
        edges = fk.production_window_edges()
        counts, _ = np.histogram(kt[(kt >= edges[0]) & (kt <= edges[-1])], bins=edges)
        # full-range 0.02 histogram on [0, tmax] (np.histogram semantics, last edge inclusive)
        full_edges = np.round(np.arange(0.0, spec.tmax + 1e-9, 0.02), 10)
        full_counts, _ = np.histogram(kt, bins=full_edges)
        payload = {"cell": {"m": m, "eps": eps, "B": B, "allocation": a},
                   "design": d, "walkers": walkers, "chunk": chunk,
                   "seed": SEED, "tag": TAG, "seed_entropy": ent,
                   "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
                   "simulator": "exact_m_prr_upgrade_core.simulate_chunk_general (EM + end-of-step Doi kill)",
                   "spec": spec.to_dict(),
                   "window_edges": edges.tolist(), "window_counts": counts.tolist(),
                   "full_edges": full_edges.tolist(), "full_counts": full_counts.tolist(),
                   "survivors_at_tmax": surv,
                   "kill_probability_max": max(r["kill_probability_max"] for r in res),
                   "walker_steps": int(sum(r["walker_steps"] for r in res)),
                   "wall_seconds": time.time() - t0, "workers": args.workers}
        core.write_json(outp, payload)
        print(f"[dk] {tagname}: wall {time.time() - t0:.1f}s, window mass "
              f"{counts.sum() / walkers:.4f}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--paths", type=float, default=N_PATHS)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--only", default="")
    s.add_argument("--chunks", default="")
    d = sub.add_parser("directkill")
    d.add_argument("--walkers", default="1e6")
    d.add_argument("--chunk", default="1e5")
    d.add_argument("--workers", type=int, default=3)
    d.add_argument("--only", default="")
    d.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--only", default="")
    a.add_argument("--no-scan", action="store_true")
    sub.add_parser("assemble")
    f = sub.add_parser("figure")
    args = ap.parse_args(argv)
    if args.cmd == "simulate":
        cmd_simulate(args)
    elif args.cmd == "directkill":
        cmd_directkill(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "assemble":
        cmd_assemble(args)
    elif args.cmd == "figure":
        cmd_figure(args)
    return 0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

GROUPS = fk.DEFAULT_GROUPS          # 20 contiguous path groups (batch means)
T975 = {19: 2.093}                  # Student t quantile for 95% batch-means CIs
DK_WALKERS_NOMINAL = 1_000_000      # protocol verdict at the confirmation size
SCAN_B = tuple(float(x) for x in np.round(np.geomspace(0.5, 64.0, 22), 6))
PARTS_DIR = OUT_DIR / "parts"
# per-item bin-mass covariance matrices are large: kept outside the repository
CURVES_DIR = fk.ENSEMBLE_ROOT / "_n1_parts"
CURVES_JSON = OUT_DIR / "n1_curves.json"


def geometric_cuts(spec: fk.EnsembleSpec) -> list[float]:
    """eps- and w-independent basin cuts: times at which mu(t) is midway between centres."""
    c = spec.centres()
    mids = 0.5 * (c[:-1] + c[1:])
    return [float(-math.log((x - spec.z_bar) / (spec.z0 - spec.z_bar)) / spec.gamma) for x in mids]


def cut_indices(ens: fk.Ensemble) -> dict:
    """Checkpoint indices for the two basin conventions.

    window:   [0.5, s_1, ..., s_{m-1}, 3.5]  (TH-2 definition, s_0 = tau, s_m = T)
    extended: [0,   s_1, ..., s_{m-1}, tmax] (all mass of each passage up to tmax)
    Interior cuts s_j are the geometric midpoints snapped to the nearest stored edge.
    """
    edges = ens.edges
    s = geometric_cuts(ens.spec)
    inner = [int(np.argmin(np.abs(edges - x))) for x in s]
    wi = ens.window_index
    return {"window": [int(wi[0])] + inner + [int(wi[-1])],
            "extended": [0] + inner + [int(edges.size - 1)],
            "cuts_requested": s, "cuts_snapped": [float(edges[k]) for k in inner]}


def frozen_gate_masses(B: float, w, spec: fk.EnsembleSpec, chi: np.ndarray) -> np.ndarray:
    """TH-7 frozen-gate law E[exp(-sum_{i<j} lam_i chi_i)(1 - exp(-lam_j chi_j))]."""
    return fk.limit_masses(B, w, spec, chi=chi)[1]


def gate_maxmin_design(B: float, spec: fk.EnsembleSpec, chi: np.ndarray) -> dict:
    """Max-min (equal-mass) allocation under the frozen-gate law (TH-7 correction of TH-4).

    With binary chi_j, M_j = a_j (1 - e^{-lam_j}), a_j = E[prod_{i<j} e^{-lam_i chi_i} chi_j].
    For a common mass p the first m-1 exposures follow sequentially, lam_j = -ln(1 - p/a_j).
    Parametrised by the LAST exposure lam_m (so that large budgets, where p approaches the
    gate-avoidance cap and lam_m grows without bound, stay representable): for given lam_m, p
    solves p = a_m(p) (1 - e^{-lam_m}) (bisection; the right side decreases in p); the budget
    A sum_j v_j lam_j increases with lam_m; bisect lam_m in log space.
    """
    v = np.asarray(speeds_for(spec))
    A = area_for(spec)
    m = np.asarray(chi).shape[1]
    # binary contact patterns: all expectations are sums over <= 2^m patterns
    pat, cnt = np.unique(np.asarray(chi).astype(np.int8), axis=0, return_counts=True)
    chi = pat.astype(float)
    pw = cnt / cnt.sum()

    def first_lams(p):
        lam, surv = [], np.ones(chi.shape[0])
        for j in range(m - 1):
            a = float(np.dot(pw, surv * chi[:, j]))
            if p >= a:
                return None, None
            lj = -math.log1p(-p / a)
            lam.append(lj)
            surv = surv * np.exp(-lj * chi[:, j])
        return lam, float(np.dot(pw, surv * chi[:, m - 1]))

    def p_for(lam_m):
        lo, hi = 0.0, 1.0 / m
        fac = -math.expm1(-lam_m)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lam, am = first_lams(mid)
            if lam is None or am * fac < mid:
                hi = mid
            else:
                lo = mid
        return lo

    def budget(lam_m):
        p = p_for(lam_m)
        lam, _ = first_lams(p)
        lam = np.asarray(lam + [lam_m])
        return A * float(np.dot(v, lam)), p, lam

    lo, hi = math.log(1e-12), math.log(1e4)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if budget(math.exp(mid))[0] > B:
            hi = mid
        else:
            lo = mid
    bud, p, lam = budget(math.exp(0.5 * (lo + hi)))
    w = A * v * lam / bud
    w = w / w.sum()
    return {"allocation": "maxmin_gate", "B": float(B), "w": w.tolist(),
            "lambda_gate": lam.tolist(), "p_star_gate": float(p), "budget_used": float(bud),
            "target": [float(p)] * m,
            "note": "equal-mass design under the frozen-gate law with chi measured on this ensemble"}



def slab_unit_exposure(spec: fk.EnsembleSpec) -> np.ndarray:
    """L[i, n]: exact (EM-chain) mean exposure of slab i alone (w = e_i, unit budget, EM contact
    probability) accumulated over the first n steps, n = 0..steps (deterministic quadrature)."""
    m = spec.centres().size
    out = []
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        fe = fk.free_exposure_discrete(spec, e, gate="contact")
        out.append(np.concatenate([[0.0], np.cumsum(fe["G"]) * spec.dt]))
    return np.asarray(out)


def mf_design(B: float, shape, Lcut: np.ndarray, spec: fk.EnsembleSpec, label: str) -> dict:
    """Finite-eps (mean-field-corrected) inverse design.

    Chooses u = B w >= 0 such that the mean-field basin masses on the window cuts s_0..s_m,
    exp(-sum_i u_i L_i(s_{j-1})) - exp(-sum_i u_i L_i(s_j)), equal theta * shape_j with the
    largest attainable theta at total sum_i u_i = B.  L_i(s) is the exact EM-chain mean
    exposure of slab i (contact factor and O(eps) smearing included); for eps -> 0 this
    reduces to the TH-4 limit-law design.  Deterministic (no sampled paths).
    Parametrised by the total log-survival Ltot = sum_i u_i L_i(s_m) (theta = S0 - e^{-Ltot}),
    so that large budgets (theta -> S0) stay representable; bisection on ln Ltot.
    """
    r = np.asarray(shape, float)
    R = np.cumsum(r)
    Mj = Lcut[:, 1:].T          # (j, i)
    L0 = Lcut[:, 0]

    def solve(Ltot):
        S0 = 1.0
        u = None
        for _ in range(200):
            theta = S0 - math.exp(-Ltot)
            if theta <= 0:
                return None, None, None
            rem = S0 - theta * R[:-1]
            if np.any(rem <= 0):
                return None, None, None
            y = np.concatenate([-np.log(rem), [Ltot]])
            u = np.linalg.solve(Mj, y)
            S0n = math.exp(-float(L0 @ u))
            if abs(S0n - S0) < 1e-15:
                S0 = S0n
                break
            S0 = S0n
        if u is None or np.any(u < 0):
            return None, None, None
        return u, S0, S0 - math.exp(-Ltot)

    lo, hi = math.log(1e-12), math.log(700.0)
    for _ in range(150):
        mid = 0.5 * (lo + hi)
        u, _, _ = solve(math.exp(mid))
        if u is None or float(u.sum()) > B:
            hi = mid
        else:
            lo = mid
    u, S0, theta = solve(math.exp(lo))
    w = u / u.sum()
    return {"allocation": label, "B": float(B), "w": w.tolist(), "theta_mf": float(theta),
            "target": (theta * r).tolist(), "mf_survival_at_window_start": float(S0),
            "budget_used": float(u.sum()),
            "feasible_at_B": bool(float(u.sum()) >= B * (1 - 1e-6)),
            "note": "mean-field-corrected design on window cuts from the exact EM-chain free clock; "
                    "if feasible_at_B is false the equal-mass condition with u >= 0 cannot absorb "
                    "the whole budget (a later slab's tail overfills an earlier basin) and w is the "
                    "design at budget_used, applied at B"}



def ensemble_pass(ens: fk.Ensemble, items: list, cuts: dict, *, groups: int = GROUPS,
                  block: int = 25_000) -> dict:
    """One pass over the stored chunks for many (B, w) items.

    Per item: window bin masses (sum, square sum, group sums, optional covariance),
    basin masses for each cut convention (sum, square sum, group sums), survival at every
    checkpoint (optional).  Per distinct w: first two moments of X at every checkpoint.
    """
    wi = np.asarray(ens.window_index)
    K = ens.cps.size
    N = ens.n_paths
    gb = fk._group_bounds(N, groups)
    nb = wi.size - 1
    wmap: dict = {}
    for it in items:
        wmap.setdefault(tuple(float(x) for x in it["w"]), []).append(it)
    accw = {wk: {"X_sum": np.zeros(K), "X_sq": np.zeros(K)} for wk in wmap}
    acc = {}
    for it in items:
        a = {"P_sum": np.zeros(nb), "P_sq": np.zeros(nb), "P_g": np.zeros((groups, nb))}
        if it.get("cov"):
            a["C"] = np.zeros((nb, nb))
        if it.get("survival"):
            a["S_sum"] = np.zeros(K)
        for cn, ci in cuts.items():
            if cn in ("window", "extended"):
                nc = len(ci) - 1
                a[f"M_{cn}_sum"] = np.zeros(nc)
                a[f"M_{cn}_sq"] = np.zeros(nc)
                a[f"M_{cn}_g"] = np.zeros((groups, nc))
        acc[it["key"]] = a
    chi_list = []
    a_sq = ens.spec.contact_a ** 2
    for start, dX, d2 in ens.iter_chunks("full"):
        chi_list.append(np.asarray(d2) < a_sq)
        n = dX.shape[0]
        for b0 in range(0, n, block):
            blk = dX[b0:b0 + block].astype(np.float64)
            nbk = blk.shape[0]
            gid = fk._group_ids(gb, start + b0, nbk)
            for wk, its in wmap.items():
                wv = np.asarray(wk)
                dXw = np.einsum("imk,m->ik", blk, wv)
                X = np.cumsum(dXw, axis=1)
                accw[wk]["X_sum"] += X.sum(0)
                accw[wk]["X_sq"] += (X * X).sum(0)
                Xw = X[:, wi]
                segw = np.add.reduceat(dXw[:, : wi[-1] + 1], wi[:-1] + 1, axis=1)
                segs = {}
                for cn in ("window", "extended"):
                    ci = np.asarray(cuts[cn])
                    segs[cn] = (X[:, ci], np.add.reduceat(dXw[:, : ci[-1] + 1], ci[:-1] + 1, axis=1))
                for it in its:
                    a = acc[it["key"]]
                    B = float(it["B"])
                    y = np.exp(-B * Xw[:, :-1]) * (-np.expm1(-B * segw))
                    a["P_sum"] += y.sum(0)
                    a["P_sq"] += (y * y).sum(0)
                    a["P_g"] += fk._group_sums(y, gid, groups)
                    if "C" in a:
                        with np.errstate(all="ignore"):
                            a["C"] += y.T @ y
                    if "S_sum" in a:
                        a["S_sum"] += np.exp(-B * X).sum(0)
                    for cn, (Xc, sc) in segs.items():
                        yb = np.exp(-B * Xc[:, :-1]) * (-np.expm1(-B * sc))
                        a[f"M_{cn}_sum"] += yb.sum(0)
                        a[f"M_{cn}_sq"] += (yb * yb).sum(0)
                        a[f"M_{cn}_g"] += fk._group_sums(yb, gid, groups)
    return {"acc": acc, "accw": accw, "chi": np.concatenate(chi_list, axis=0), "N": N,
            "group_sizes": np.diff(gb).astype(float)}


def _stats(s, sq, g, N, sizes):
    mean = s / N
    se = np.sqrt(np.maximum(sq / N - mean * mean, 0.0) / N)
    seb = fk._batch_se(g / sizes.reshape((-1,) + (1,) * (g.ndim - 1)), sizes)
    return mean, se, seb


def _mf_bins(Lam, B, idx):
    E = np.exp(-B * Lam[idx])
    return E[:-1] - E[1:]


def _basin_of(t, cut_times):
    for j in range(len(cut_times) - 1):
        if cut_times[j] < t <= cut_times[j + 1]:
            return j
    return None


def summarize_item(ens, it, res, cuts, spec, chi, mf_Lam) -> tuple[dict, dict]:
    a = res["acc"][it["key"]]
    N = res["N"]
    sizes = res["group_sizes"]
    G = sizes.size
    tq = T975.get(G - 1, 1.96)
    wi = np.asarray(ens.window_index)
    edges = ens.edges[wi]
    t = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)
    P, se, seb = _stats(a["P_sum"], a["P_sq"], a["P_g"], N, sizes)
    dens = P / bw
    gdens = (a["P_g"] / sizes[:, None]) / bw[None, :]
    cov = None
    if "C" in a:
        cov = (a["C"] / N - np.outer(P, P)) / N
    B = float(it["B"])
    w = np.asarray(it["w"], float)
    m = w.size
    target = np.asarray(it["target"], float)
    # classifier (expected law at the protocol size), FK-significance, derivative census
    cls = fk.classify_expected(P, edges, walkers=DK_WALKERS_NOMINAL)
    cut_times_w = [float(ens.edges[k]) for k in cuts["window"]]
    sig = [r for r in cls["rows"] if r["significant"]]
    basins_hit = sorted({_basin_of(r["time"], cut_times_w) for r in sig} - {None})
    per_basin_prom = []
    for j in range(m):
        vals = [r["relative_prominence"] for r in cls["rows"]
                if _basin_of(r["time"], cut_times_w) == j and r["z"] >= 5.0]
        per_basin_prom.append(max(vals) if vals else 0.0)
    out = {"key": it["key"], "m": int(m), "eps": float(spec.eps), "B": B,
           "allocation": it["allocation"], "w": w.tolist(), "target": target.tolist(),
           "design": it.get("design", {}),
           "protocol_mode_count_1e6": int(cls["mode_count"]),
           "protocol_significant_times": [r["time"] for r in sig],
           "protocol_basins_with_significant_mode": basins_hit,
           "relative_prominence_best_per_basin": per_basin_prom,
           "weakest_basin_relative_prominence": float(min(per_basin_prom)) if per_basin_prom else 0.0}
    if cov is not None:
        cfk = fk.classify_expected(P, edges, walkers=N, cov=cov)
        out["fk_cov_mode_count"] = int(cfk["mode_count"])
        out["fk_cov_significant_times"] = [r["time"] for r in cfk["rows"] if r["significant"]]
    for bwd in (0.0, 0.04):
        ds = fk.derivative_signs(t, dens, gdens, bandwidth=bwd, zthr=5.0)
        out[f"derivative_census_bw{bwd:g}"] = {k: ds[k] for k in
                                                ("n_maxima", "n_minima", "maxima_t", "minima_t",
                                                 "n_undecided_points")}
    # basins
    lam, lim = fk.limit_masses(B, w, spec)
    gate = frozen_gate_masses(B, w, spec, chi)
    for cn in ("window", "extended"):
        M, Mse, Mseb = _stats(a[f"M_{cn}_sum"], a[f"M_{cn}_sq"], a[f"M_{cn}_g"], N, sizes)
        mf = _mf_bins(mf_Lam, B, np.asarray(cuts[cn]))
        dev = M - target
        out[f"basins_{cn}"] = {
            "cut_times": [float(ens.edges[k]) for k in cuts[cn]],
            "mass": M.tolist(), "se_iid": Mse.tolist(), "se_batch": Mseb.tolist(),
            "ci95_batch": [[float(x - tq * s), float(x + tq * s)] for x, s in zip(M, Mseb)],
            "target_limit_law": target.tolist(), "deviation_from_target": dev.tolist(),
            "max_abs_deviation": float(np.max(np.abs(dev))),
            "z_vs_target_batch": (dev / np.where(Mseb > 0, Mseb, np.inf)).tolist(),
            "mean_field_sampled_Lambda": mf.tolist(),
            "frozen_gate_law": gate.tolist(),
            "deviation_from_frozen_gate": (M - gate).tolist(),
            "max_abs_deviation_frozen_gate": float(np.max(np.abs(M - gate))),
            "total": float(M.sum()), "total_target": float(target.sum())}
    out["lambda_limit"] = lam.tolist()
    out["window_mass"] = float(P.sum())
    out["mean_field_protocol_mode_count_1e6"] = int(fk.classify_expected(
        _mf_bins(mf_Lam, B, wi), edges, walkers=DK_WALKERS_NOMINAL)["mode_count"])
    curve = {"t": t, "density": dens, "density_se_iid": se / bw, "density_se_batch": seb / bw,
             "mean_field_density": _mf_bins(mf_Lam, B, wi) / bw}
    if cov is not None:
        curve["_cov"] = cov
        curve["_bin_mass"] = P
    return out, curve


def build_items(spec: fk.EnsembleSpec, chi: np.ndarray, scan: bool, Lcut=None) -> list:
    m = spec.centres().size
    items = []

    def mf_items(B, grid):
        if Lcut is None:
            return []
        res = []
        shapes = [("maxmin_mf", [1.0 / m] * m)]
        if grid and m in SHAPES:
            shapes.append(("shape_mf", list(SHAPES[m])))
        for lab, shp in shapes:
            dmf = mf_design(B, shp, Lcut, spec, lab)
            res.append({"key": (f"B{B:g}_" if grid else f"scan_B{B:g}_") + lab, "B": B,
                        "w": dmf["w"], "target": dmf["target"], "allocation": lab, "design": dmf,
                        "cov": grid, "survival": False, "grid": grid})
        return res

    Bs = M5_B if m == 5 else B_LIST
    allocs = ("equal", "maxmin") if m == 5 else ("equal", "maxmin", "shape")
    for B in Bs:
        for a in allocs:
            d = design(spec, B, a)
            items.append({"key": f"B{B:g}_{a}", "B": B, "w": d["w"], "target": d["target"],
                          "allocation": a, "design": d, "cov": True, "survival": False,
                          "grid": True})
        g = gate_maxmin_design(B, spec, chi)
        items.append({"key": f"B{B:g}_maxmin_gate", "B": B, "w": g["w"], "target": g["target"],
                      "allocation": "maxmin_gate", "design": g, "cov": True, "survival": False,
                      "grid": True})
        items.extend(mf_items(B, True))
    if scan:
        for B in SCAN_B:
            for a in ("equal", "maxmin"):
                d = design(spec, B, a)
                items.append({"key": f"scan_B{B:g}_{a}", "B": B, "w": d["w"],
                              "target": d["target"], "allocation": a, "design": d,
                              "cov": False, "survival": False, "grid": False})
            g = gate_maxmin_design(B, spec, chi)
            items.append({"key": f"scan_B{B:g}_maxmin_gate", "B": B, "w": g["w"],
                          "target": g["target"], "allocation": "maxmin_gate", "design": g,
                          "cov": False, "survival": False, "grid": False})
            items.extend(mf_items(B, False))
    return items


def load_chi(ens: fk.Ensemble) -> np.ndarray:
    a_sq = ens.spec.contact_a ** 2
    out = []
    for c in ens.chunks:
        with np.load(c["file"]) as d:
            out.append(np.asarray(d["dist2_tj"]) < a_sq)
    return np.concatenate(out, axis=0)


def analyze_ensemble(m: int, eps: float, scan: bool = True) -> dict:
    t0 = time.time()
    ens = fk.load_ensemble(ens_name(m, eps))
    if not ens.index.get("complete"):
        raise RuntimeError(f"{ens.name} incomplete")
    spec = ens.spec
    cuts = cut_indices(ens)
    chi = load_chi(ens)
    L = slab_unit_exposure(spec)
    Lcut = L[:, ens.cps[np.asarray(cuts["window"])]]
    items = build_items(spec, chi, scan, Lcut=Lcut)
    res = ensemble_pass(ens, items, cuts)
    wkeys = {}
    for wk, a in res["accw"].items():
        wkeys[wk] = a["X_sum"] / res["N"]
    rows, curves = [], {}
    for it in items:
        Lam = wkeys[tuple(float(x) for x in it["w"])]
        r, c = summarize_item(ens, it, res, cuts, spec, chi, Lam)
        r["grid"] = it["grid"]
        rows.append(r)
        if it["grid"]:
            curves[it["key"]] = c
    # equal-allocation prominence-floor crossing on this ensemble (retention reference)
    bp = None
    if m in (2, 3):
        try:
            bp = fk.prominence_floor_crossing(ens, w=[1.0 / m] * m, p=0.05, B_lo=0.5, B_hi=64.0,
                                              n_coarse=15, n_fine=17)
            bp = {k: bp[k] for k in ("B_p", "p", "jackknife_se", "ci95", "after_time", "status")
                  if k in bp}
        except Exception as exc:  # noqa: BLE001
            bp = {"error": repr(exc)}
    chi_rates = chi.mean(axis=0).tolist()
    part = {"ensemble": ens.name, "m": m, "eps": eps, "n_paths": ens.n_paths,
            "seed": ens.index["seed"], "tag": ens.index["tag"],
            "replicate": ens.index["replicate"], "seed_entropy": ens.index["seed_entropy"],
            "spec": ens.index["spec"], "cuts": {k: v for k, v in cuts.items()},
            "contact_probability_at_t_j": chi_rates, "equal_weight_B5": bp,
            "rows": rows, "analysis_seconds": time.time() - t0}
    return fk._jsonable(part), curves


def cmd_analyze(args) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    todo = ensemble_list()
    if args.only:
        keep = set(args.only.split(","))
        todo = [(m, e) for (m, e) in todo if ens_name(m, e) in keep]
    for m, e in todo:
        part, curves = analyze_ensemble(m, e, scan=not args.no_scan)
        core.write_json(PARTS_DIR / f"{ens_name(m, e)}.json", part)
        CURVES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CURVES_DIR / f"{ens_name(m, e)}_curves.npz",
                            **{f"{k}__{kk}": np.asarray(vv) for k, c in curves.items()
                               for kk, vv in c.items()})
        print(f"[n1] analyzed {ens_name(m, e)} in {part['analysis_seconds']:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# assemble: direct-kill comparisons, acceptance, p* curves
# ---------------------------------------------------------------------------


def _load_part(m, e):
    p = PARTS_DIR / f"{ens_name(m, e)}.json"
    if not p.exists():
        return None, None
    part = json.loads(p.read_text())
    cz = CURVES_DIR / f"{ens_name(m, e)}_curves.npz"
    curves = {}
    if cz.exists():
        with np.load(cz) as d:
            for k in d.files:
                key, kk = k.split("__", 1)
                curves.setdefault(key, {})[kk] = d[k]
    return part, curves


def compare_directkill(dk: dict, row: dict, curve: dict) -> dict:
    import reclassify_covariance_aware as R
    from exact_m_prr_fk_n0_validation import _chi2_sf, _cov_chi2, merge_groups
    N = int(dk["walkers"])
    cnt = np.asarray(dk["window_counts"], float)
    edges = np.asarray(dk["window_edges"], float)
    if not np.allclose(edges, fk.production_window_edges()):
        raise ValueError("direct-kill edges differ from production edges")
    p = np.asarray(curve["_bin_mass"], float)
    C = np.asarray(curve["_cov"], float)
    pdk = cnt / N
    grp = merge_groups(N * p, 10.0)
    A = np.zeros((len(grp), p.size))
    for gi, g in enumerate(grp):
        A[gi, g] = 1.0
    pm, pdm = A @ p, A @ pdk
    Cm = A @ C @ A.T
    zm = (pm - pdm) / np.sqrt(np.diag(Cm) + pm * (1 - pm) / N)
    chi2, dof = _cov_chi2(pm - pdm, Cm + (np.diag(pm) - np.outer(pm, pm)) / N)
    # basin masses (window cuts, identical checkpoints: production edges)
    cut_t = row["basins_window"]["cut_times"]
    ci = [int(np.argmin(np.abs(edges - c))) for c in cut_t]
    cum = np.concatenate([[0.0], np.cumsum(cnt)])
    Mdk = np.array([(cum[ci[j + 1]] - cum[ci[j]]) / N for j in range(len(ci) - 1)])
    Mfk = np.asarray(row["basins_window"]["mass"])
    sefk = np.asarray(row["basins_window"]["se_iid"])
    sedk = np.sqrt(Mdk * (1 - Mdk) / N)
    zb = (Mfk - Mdk) / np.sqrt(sefk**2 + sedk**2)
    cls = R.classify_both(cnt.astype(np.int64), edges, N, bandwidth=fk.base.DEFAULT_BANDWIDTH)
    target = np.asarray(row["target"])
    tq = 1.96
    return {"cell": dk["cell"], "walkers": N, "seed_entropy": dk["seed_entropy"],
            "tag": dk["tag"], "w": dk["design"]["w"],
            "dk_mode_count_covariance_aware": int(cls["mode_count_covariance_aware"]),
            "dk_significant_times": cls.get("significant_times_covariance_aware"),
            "fk_protocol_mode_count_1e6": row["protocol_mode_count_1e6"],
            "n_merged_bins": len(grp), "merged_max_abs_z": float(np.max(np.abs(zm))),
            "chi2_cov_merged": chi2, "chi2_cov_dof": dof,
            "chi2_cov_p_value_wilson_hilferty": _chi2_sf(chi2, dof),
            "basin_masses_dk": Mdk.tolist(), "basin_se_dk": sedk.tolist(),
            "basin_ci95_dk": [[float(x - tq * s), float(x + tq * s)] for x, s in zip(Mdk, sedk)],
            "basin_masses_fk": Mfk.tolist(), "basin_z_fk_vs_dk": zb.tolist(),
            "basin_target": target.tolist(),
            "dk_max_abs_deviation_from_target": float(np.max(np.abs(Mdk - target))),
            "window_mass_dk": float(cnt.sum() / N),
            "survivors_at_tmax_dk": dk["survivors_at_tmax"],
            "wall_seconds": dk["wall_seconds"],
            "_plot": {"t": (0.5 * (edges[:-1] + edges[1:])).tolist(),
                      "dk_density": (pdk / np.diff(edges)).tolist()}}


def pstar_curves() -> dict:
    out = {}
    Bs = np.geomspace(0.1, 100.0, 121)
    for m in (2, 3):
        spec = spec_for(m, 0.1)
        v = speeds_for(spec)
        A = area_for(spec)
        out[f"m{m}"] = {"B": Bs.tolist(),
                        "p_star": [alloc.p_star(float(B), v, A)["p_star"] for B in Bs],
                        "equal_min_limit_mass": [min(alloc.masses(float(B), [1 / m] * m, v, A))
                                                 for B in Bs]}
    spec = spec_for(5, 0.1)
    v = speeds_for(spec)
    out["m5_z08"] = {"B": Bs.tolist(),
                     "p_star": [alloc.p_star(float(B), v, 1.0)["p_star"] for B in Bs]}
    return out


def acceptance(parts: dict, dks: list) -> dict:
    acc = {}
    plan_allocs = ("equal", "maxmin", "shape")
    # (a) eps = 0.05: every basin within 0.02 absolute of target, B <= 8
    rows05 = [r for (m, e), p in parts.items() if e == 0.05 for r in p["rows"]
              if r["grid"] and r["allocation"] in plan_allocs]
    for cn in ("window", "extended"):
        worst = max(rows05, key=lambda r: r[f"basins_{cn}"]["max_abs_deviation"])
        acc[f"eps0.05_max_abs_dev_{cn}"] = {
            "value": worst[f"basins_{cn}"]["max_abs_deviation"], "at": worst["key"],
            "m": worst["m"], "n_cells": len(rows05),
            "n_cells_within_0.02": sum(r[f"basins_{cn}"]["max_abs_deviation"] <= 0.02
                                       for r in rows05)}
    # (b) eps = 0.1: <= max(0.07, 3 SE); frozen-gate law closes >= half of the residual
    rows10 = [r for (m, e), p in parts.items() if e == 0.1 and m in (2, 3) for r in p["rows"]
              if r["grid"] and r["allocation"] in plan_allocs]
    for cn in ("window", "extended"):
        ok, worst, res_lim, res_gate, res_mf = 0, None, 0.0, 0.0, 0.0
        failing = []
        for r in rows10:
            b = r[f"basins_{cn}"]
            dev = np.abs(np.asarray(b["deviation_from_target"]))
            thr = np.maximum(0.07, 3 * np.asarray(b["se_batch"]))
            ok += int(np.all(dev <= thr))
            if not np.all(dev <= thr):
                failing.append({"m": r["m"], "B": r["B"], "allocation": r["allocation"],
                                "max_abs_deviation": float(dev.max()),
                                "basin": int(np.argmax(dev)) + 1})
            if worst is None or b["max_abs_deviation"] > worst[1]:
                worst = (r["key"] + f"_m{r['m']}", b["max_abs_deviation"])
            res_lim += float(np.sum(dev))
            res_gate += float(np.sum(np.abs(b["deviation_from_frozen_gate"])))
            res_mf += float(np.sum(np.abs(np.asarray(b["mass"]) - np.asarray(b["mean_field_sampled_Lambda"]))))
        acc[f"eps0.1_{cn}"] = {"n_cells": len(rows10), "n_cells_within_max_0.07_3se": ok,
                               "failing_cells": failing,
                               "worst": worst, "sum_abs_residual_vs_limit": res_lim,
                               "sum_abs_residual_vs_frozen_gate": res_gate,
                               "fraction_of_residual_closed_by_frozen_gate":
                                   1.0 - res_gate / res_lim if res_lim > 0 else None,
                               "sum_abs_residual_vs_mean_field_exact_G": res_mf,
                               "fraction_of_residual_closed_by_mean_field_exact_G":
                                   1.0 - res_mf / res_lim if res_lim > 0 else None}
    # frozen-gate closure split by basin position (first basin vs later basins), eps = 0.1, window
    first, later = [0.0, 0.0], [0.0, 0.0]
    for r in rows10:
        b = r["basins_window"]
        d0 = np.abs(np.asarray(b["deviation_from_target"]))
        dg = np.abs(np.asarray(b["deviation_from_frozen_gate"]))
        first[0] += d0[0]; first[1] += dg[0]
        later[0] += d0[1:].sum(); later[1] += dg[1:].sum()
    acc["eps0.1_frozen_gate_by_basin_window"] = {
        "first_basin_fraction_closed": 1 - first[1] / first[0],
        "later_basins_fraction_closed": 1 - later[1] / later[0]}
    # residuals of every allocation family, per eps (window convention)
    fam = {}
    for (m, e), p in parts.items():
        for r in p["rows"]:
            if not r["grid"]:
                continue
            k = f"eps{e:g}_{r['allocation']}" + ("_m5" if m == 5 else "")
            f = fam.setdefault(k, {"n_cells": 0, "max_abs_deviation": 0.0, "sum_abs_deviation": 0.0,
                                   "n_cells_protocol_count_m": 0, "n_cells_derivative_count_m": 0})
            b = r["basins_window"]
            f["n_cells"] += 1
            f["max_abs_deviation"] = max(f["max_abs_deviation"], b["max_abs_deviation"])
            f["sum_abs_deviation"] += float(np.sum(np.abs(b["deviation_from_target"])))
            f["n_cells_protocol_count_m"] += int(r["protocol_mode_count_1e6"] == m)
            f["n_cells_derivative_count_m"] += int(r["derivative_census_bw0.04"]["n_maxima"] == m)
    acc["families_window"] = fam
    # (c) retention: designed keeps m modes wherever equal loses one (eps <= 0.1)
    ret = []
    for (m, e), p in parts.items():
        if e > 0.1:
            continue
        byk = {r["key"]: r for r in p["rows"]}
        for r in p["rows"]:
            if r["allocation"] != "equal":
                continue
            if r["protocol_mode_count_1e6"] < m:
                kd = r["key"].replace("_equal", "_maxmin")
                d = byk.get(kd)
                kg = r["key"].replace("_equal", "_maxmin_gate")
                g = byk.get(kg)
                ret.append({"m": m, "eps": e, "B": r["B"], "grid": r["grid"],
                            "equal_count": r["protocol_mode_count_1e6"],
                            "maxmin_count": d["protocol_mode_count_1e6"] if d else None,
                            "maxmin_derivative_maxima_bw0.04":
                                d["derivative_census_bw0.04"]["n_maxima"] if d else None,
                            "maxmin_gate_count": g["protocol_mode_count_1e6"] if g else None})
    acc["retention_cases"] = ret
    acc["retention_all_designed_keep_m"] = all(x["maxmin_count"] == x["m"] for x in ret)
    acc["retention_grid_cases"] = [x for x in ret if x["grid"]]
    acc["directkill"] = {"n": len(dks),
                         "mode_count_agreement": sum(d["dk_mode_count_covariance_aware"]
                                                     == d["fk_protocol_mode_count_1e6"]
                                                     for d in dks),
                         "max_merged_abs_z": max((d["merged_max_abs_z"] for d in dks), default=None),
                         "max_basin_abs_z": max((max(abs(x) for x in d["basin_z_fk_vs_dk"])
                                                 for d in dks), default=None)}
    return acc


def cmd_assemble(args) -> None:
    parts, curves = {}, {}
    for m, e in ensemble_list():
        p, c = _load_part(m, e)
        if p is not None:
            parts[(m, e)] = p
            curves[(m, e)] = c
    dks = []
    for f in sorted(DK_DIR.glob("dk_*.json")):
        dk = json.loads(f.read_text())
        c = dk["cell"]
        key = (c["m"], c["eps"])
        if key not in parts:
            continue
        rk = f"B{c['B']:g}_{c['allocation']}"
        row = next(r for r in parts[key]["rows"] if r["key"] == rk)
        dks.append(compare_directkill(dk, row, curves[key][rk]))
    payload = {
        "item": "N1", "plan": "notes/gap_diagnosis_20260923/GAP_CLOSURE_PLAN.md §3 N1",
        "driver": HERE.name, "seed": SEED, "tag": TAG,
        "model": core.model_payload(),
        "allocations": {"equal": "w_j = 1/m",
                        "maxmin": "TH-4 max-min (equal-mass) design p*(B,m) (fb_allocation_law.p_star)",
                        "shape": "target theta*(B,r) r, r = (0.2,0.8) / (0.5,0.3,0.2), theta* the largest "
                                 "common scale attainable at B (fb_allocation_law.max_common_scale)",
                        "maxmin_gate": "extra: equal-mass design under the frozen-gate law (TH-7) "
                                       "with contact indicators measured on the same ensemble"},
        "basin_conventions": {"window": "cuts [0.5, s_1..s_{m-1}, 3.5] (TH-2 definition)",
                              "extended": "cuts [0, s_1..s_{m-1}, tmax=4]",
                              "interior_cuts": "times where mu(t) is midway between adjacent "
                                               "slab centres, snapped to the 0.02 grid "
                                               "(independent of eps and w)"},
        "statistics": "FK exact law of the EM chain; iid per-path SE and batch means over 20 "
                      "contiguous path groups; 95% CI = mean +- t_{0.975,19} SE_batch",
        "protocol_mode_count": "covariance-aware 5 sigma + 5% classifier applied to the expected "
                               "histogram at 1e6 walkers (fk.classify_expected)",
        "ensembles": {f"m{m}_eps{e:g}": {k: p[k] for k in ("ensemble", "n_paths", "seed", "tag",
                                                            "replicate", "seed_entropy", "cuts",
                                                            "contact_probability_at_t_j",
                                                            "equal_weight_B5", "analysis_seconds")}
                      for (m, e), p in parts.items()},
        "cells": [dict(r, ensemble=p["ensemble"]) for (m, e), p in parts.items()
                  for r in p["rows"] if r["grid"]],
        "scan": [{k: r[k] for k in ("m", "eps", "B", "allocation", "w", "target",
                                    "protocol_mode_count_1e6", "weakest_basin_relative_prominence",
                                    "relative_prominence_best_per_basin")}
                 | {"basin_mass_window": r["basins_window"]["mass"],
                    "basin_se_batch_window": r["basins_window"]["se_batch"],
                    "derivative_maxima_bw0.04": r["derivative_census_bw0.04"]["n_maxima"],
                    "derivative_maxima_bw0": r["derivative_census_bw0"]["n_maxima"]}
                 for (m, e), p in parts.items() for r in p["rows"] if not r["grid"]],
        "directkill": [{k: v for k, v in d.items() if k != "_plot"} for d in dks],
        "p_star_curves": pstar_curves(),
        "acceptance": acceptance(parts, dks),
    }
    core.write_json(OUT_JSON, fk._jsonable(payload))
    # plot helpers
    plot = {"directkill": {f"m{d['cell']['m']}_eps{d['cell']['eps']:g}_B{d['cell']['B']:g}": d["_plot"]
                           for d in dks},
            "curves": {f"m{m}_eps{e:g}__{k}": {kk: [float(f"{x:.6g}") for x in np.asarray(vv).ravel()]
                                               for kk, vv in c.items() if not kk.startswith("_")}
                       for (m, e), cc in curves.items() for k, c in cc.items()}}
    core.write_json(CURVES_JSON, plot)
    print(f"[n1] wrote {OUT_JSON}", flush=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    core.apply_prr_style()
    d = json.loads(OUT_JSON.read_text())
    pl = json.loads(CURVES_JSON.read_text())
    written = []
    col = {"equal": core.OI_ORANGE, "maxmin": core.OI_BLUE, "shape": core.OI_GREEN}
    mk = {2: "o", 3: "s", 5: "^"}
    lab = {"equal": "equal weights", "maxmin": "max-min design", "shape": "target shape design"}

    # --- Fig. A: target vs measured basin masses --------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0]}, layout="constrained")
    for ci, eps in enumerate(EPS_LIST):
        ax, axr = axes[0, ci], axes[1, ci]
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.7)
        tol = 0.02 if eps == 0.05 else (0.07 if eps == 0.1 else None)
        for c in d["cells"]:
            if c["eps"] != eps or c["allocation"] not in col:
                continue
            b = c["basins_window"]
            T = np.asarray(b["target_limit_law"]); M = np.asarray(b["mass"])
            face = col[c["allocation"]] if c["m"] != 5 else core.OI_PURPLE
            ax.plot(T, M, mk[c["m"]], ms=3.2, mfc=face, mec="k", mew=0.3, alpha=0.9)
            axr.plot(T, M - T, mk[c["m"]], ms=3.2, mfc=face, mec="k", mew=0.3, alpha=0.9)
        for c in d["cells"]:
            if c["eps"] != eps or c["allocation"] != "maxmin_mf":
                continue
            b = c["basins_window"]
            T = np.asarray(b["target_limit_law"]); M = np.asarray(b["mass"])
            axr.plot(T, M - T, mk[c["m"]], ms=3.4, mfc="none", mec=core.OI_SKY, mew=0.9)
        axr.axhline(0, color="0.5", lw=0.7)
        if tol:
            axr.axhspan(-tol, tol, color="0.85", lw=0, zorder=0)
            axr.text(0.98, 0.04, f"$\\pm${tol:g}", transform=axr.transAxes, ha="right",
                     va="bottom", fontsize=6.5, color="0.35")
        ax.set_title(f"$\\varepsilon$ = {eps:g}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        lim_r = {0.05: 0.03, 0.1: 0.11, 0.15: 0.17}[eps]
        axr.set_ylim(-lim_r, lim_r)
        axr.set_xlabel("target (limit-law) basin mass $M_j^{\\rm lim}$")
        if ci == 0:
            ax.set_ylabel("exact basin mass $M_j$")
            axr.set_ylabel("$M_j - M_j^{\\rm lim}$")
    handles = [Line2D([], [], ls="", marker="o", mfc=col[a], mec="k", mew=0.3, label=lab[a])
               for a in ("equal", "maxmin", "shape")]
    handles += [Line2D([], [], ls="", marker="o", mfc="none", mec=core.OI_SKY, mew=0.9,
                       label="max-min, finite-$\\varepsilon$ corrected (own target)")]
    handles += [Line2D([], [], ls="", marker="o", mfc="w", mec="k", label="$m$ = 2"),
                Line2D([], [], ls="", marker="s", mfc="w", mec="k", label="$m$ = 3"),
                Line2D([], [], ls="", marker="^", mfc=core.OI_PURPLE, mec="k", mew=0.3,
                       label="$m$ = 5 ($z_0$ = 8)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    core.stream_tag(fig, "fb N1")
    written += core.save_figure(fig, fk.FIGURES / "fb_n1_design_masses")
    plt.close(fig)

    # --- Fig. B: densities at (2, 0.1, 8) and (3, 0.1, 4) --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained")
    for ax, (m, eps, B) in zip(axes, [(2, 0.1, 8.0), (3, 0.1, 4.0)]):
        for a in ("equal", "maxmin"):
            c = pl["curves"][f"m{m}_eps{eps:g}__B{B:g}_{a}"]
            t = np.asarray(c["t"]); f = np.asarray(c["density"]); s = np.asarray(c["density_se_batch"])
            row = next(r for r in d["cells"] if r["m"] == m and r["eps"] == eps and r["B"] == B
                       and r["allocation"] == a)
            ww = ", ".join(f"{x:.3f}" for x in row["w"])
            ax.fill_between(t, f - 2 * s, f + 2 * s, color=col[a], alpha=0.3, lw=0)
            ax.plot(t, f, color=col[a], lw=1.1,
                    label=f"{lab[a]}, $w$ = ({ww}): {row['protocol_mode_count_1e6']} "
                          + ("mode" if row['protocol_mode_count_1e6'] == 1 else "modes"))
        dkp = pl["directkill"].get(f"m{m}_eps{eps:g}_B{B:g}")
        if dkp:
            ax.plot(dkp["t"], dkp["dk_density"], ".", ms=2.0, color="k",
                    label="direct kill, max-min design ($10^6$ walkers)")
        spec = spec_for(m, eps)
        for tj in spec.times():
            ax.axvline(tj, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"$m$ = {m}, $\\varepsilon$ = {eps:g}, $B$ = {B:g}")
        ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6.0, frameon=False, loc="upper right")
    axes[0].set_ylabel("reaction-time density $f(t)$")
    core.stream_tag(fig, "fb N1")
    written += core.save_figure(fig, fk.FIGURES / "fb_n1_design_densities")
    plt.close(fig)

    # --- Fig. C: p*(B, m) and the smallest basin mass vs B ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained", sharey=True)
    ecol = {0.05: core.OI_BLUE, 0.1: core.OI_GREEN, 0.15: core.OI_VERMILLION}
    for ax, m in zip(axes, (2, 3)):
        pc = d["p_star_curves"][f"m{m}"]
        ax.plot(pc["B"], pc["p_star"], color="k", lw=1.1, label="$p^*(B,m)$ (limit law)")
        ax.plot(pc["B"], pc["equal_min_limit_mass"], color=core.OI_ORANGE, lw=1.0, ls="--",
                label="equal weights, limit law")
        for eps in EPS_LIST:
            for a, off in (("maxmin", 1.0), ("equal", 1.0)):
                rows = [r for r in d["scan"] if r["m"] == m and r["eps"] == eps
                        and r["allocation"] == a]
                if not rows:
                    continue
                Bs = np.array([r["B"] for r in rows])
                mn = np.array([min(r["basin_mass_window"]) for r in rows])
                ok = np.array([r["protocol_mode_count_1e6"] == m for r in rows])
                mark = "o" if a == "maxmin" else "v"
                ax.plot(Bs[ok], mn[ok], mark, ms=3.0, color=ecol[eps], mec=ecol[eps])
                ax.plot(Bs[~ok], mn[~ok], mark, ms=3.0, mfc="w", mec=ecol[eps])
        ax.set_xscale("log")
        ax.set_xlabel("budget $B$")
        ax.set_title(f"$m$ = {m}")
        ax.set_xlim(0.4, 80)
    axes[0].set_ylabel("smallest basin mass $\\min_j M_j$")
    handles = [Line2D([], [], color="k", lw=1.1, label="$p^*(B,m)$, limit law"),
               Line2D([], [], color=core.OI_ORANGE, lw=1.0, ls="--", label="equal weights, limit law")]
    handles += [Line2D([], [], ls="", marker="o", ms=4, color=ecol[e],
                       label=f"exact law, $\\varepsilon$ = {e:g}") for e in EPS_LIST]
    handles += [Line2D([], [], ls="", marker="o", ms=4, color="0.3", label="circles: max-min design"),
                Line2D([], [], ls="", marker="v", ms=4, color="0.3", label="triangles: equal weights"),
                Line2D([], [], ls="", marker="o", ms=4, mfc="w", mec="0.3",
                       label="open: fewer than $m$ modes (protocol, $10^6$)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    core.stream_tag(fig, "fb N1")
    written += core.save_figure(fig, fk.FIGURES / "fb_n1_pstar")
    plt.close(fig)
    print("\n".join(written))


if __name__ == "__main__":
    sys.exit(main())
