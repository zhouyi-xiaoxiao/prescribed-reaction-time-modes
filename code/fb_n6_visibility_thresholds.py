#!/usr/bin/env python3
"""N6: visibility thresholds B_p on the EXACT reaction-time law (GAP_CLOSURE_PLAN.md, section 3, N6).

Replaces the headline B_op bisection (a prominence-floor crossing of noisy
direct-kill histograms under the classifier protocol) by a functional of the
exact expected law of the production process, computed with the N0
Feynman--Kac / Rao--Blackwell estimator (exact_m_prr_fk_exact_law.py).

Definitions (all on the production window [0.5, 3.5], 0.02 bins, equal weights)
-------------------------------------------------------------------------------
* r_s(B): largest relative prominence (contour base = higher of the two side
  minima, as in the classifier; relative to the global maximum of the smoothed
  window density) among local maxima later than the last window valley of the
  semi-analytic free-exposure clock G ("last mode"), of the bandwidth-0.04
  Gaussian-smoothed EXACT expected density.  No significance gate.
* r_u(B): the same on the unsmoothed 0.02-bin exact density.
* r_P(B): protocol version (as in N0): the smoothed exact law judged by the
  covariance-aware 5-sigma rule at 1e6 EXPECTED walkers; a late maximum counts
  only if z >= 5.  This is "prominence-floor crossing under protocol P" and is
  the quantity the S2 seed envelopes bracket.
* B_p = smallest B at which r falls to p (first down-crossing, log-linear
  interpolation), p in {1, 5, 10} %.  Status flags: "below_at_B_lo" (r < p at
  the smallest budget: the late mode is never p-visible), "censored" (no
  crossing up to B_hi).

Uncertainty: path-group (40 groups) delete-one jackknife SE and group
bootstrap (1000 replicates, percentile 95 % CI) of B_p.  Effective sample size
of the late-basin FK weights, ESS = (sum y)^2 / sum y^2, is reported at large B
(plan: fall back to direct-kill probes if ESS < 1e3).

Cells: m in {2, 3} x eps in {0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25};
B in [0.02, 64] for every cell (this includes the requested extension of m=2
at eps = 0.15, 0.2 to B = 64).  Ensembles: 2e5 unkilled paths each, seed tag
85 (base 20260923); the four cells (m, eps) with eps in {0.05, 0.1} reuse the
N0 ensembles (tag 80).  NB paths depend only on the path law, so the m=2 and
m=3 ensembles at the same eps share paths (common random numbers).

Usage
-----
    python3 fb_n6_visibility_thresholds.py simulate [--cells m2_eps0.075,...]
    python3 fb_n6_visibility_thresholds.py analyze  [--cells ...] [--boot 1000]
    python3 fb_n6_visibility_thresholds.py figure
Outputs: artifacts/data/exact_m_fixed_budget/N6/n6_cell_<cell>.json,
n6_visibility_thresholds.json; figures fb_n6_visibility_thresholds.{pdf,png},
fb_n6_prominence_curves.{pdf,png}.
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

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402

OUT = fk.FB_DATA / "N6"
TAG = fk.TAGS["N6"]
EPS_LIST = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25)
M_LIST = (2, 3)
N_PATHS = 200_000
N0_REUSE = {(2, 0.05): "n0_m2_eps0.05", (3, 0.05): "n0_m3_eps0.05",
            (2, 0.1): "n0_m2_eps0.1", (3, 0.1): "n0_m3_eps0.1"}
P_LEVELS = (0.01, 0.05, 0.10)
B_LO, B_HI = 0.02, 64.0
N_COARSE = 57
GROUPS = 40
BANDWIDTH = 0.04
PROTOCOL_WALKERS = 1_000_000
S2_ENVELOPES = {  # w2_seed_repeat_summary.json#cells[*].three_seed.envelope
    "m2_eps0.05": [6.874477192489912, 7.025008641493198],
    "m3_eps0.1": [3.5507529850530424, 3.5894181500062143],
}
S2_SOURCE = ("artifacts/data/exact_m_prr_upgrade/robustness/w2_seed_repeat_summary.json"
             "#cells[*].three_seed.envelope")


def cell_label(m: int, eps: float) -> str:
    return f"m{m}_eps{eps:g}"


def ensemble_name(m: int, eps: float) -> str:
    return N0_REUSE.get((m, round(eps, 6)), f"n6_m{m}_eps{eps:g}")


def parse_cells(text: str | None):
    allc = [(m, e) for m in M_LIST for e in EPS_LIST]
    if not text:
        return allc
    want = set(text.split(","))
    return [c for c in allc if cell_label(*c) in want]


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------


def cmd_simulate(args):
    for m, eps in parse_cells(args.cells):
        name = ensemble_name(m, eps)
        if name.startswith("n0_"):
            print(f"[n6] {cell_label(m, eps)}: reuse N0 ensemble {name}", flush=True)
            continue
        ip = fk.index_path(name)
        if ip.exists() and json.loads(ip.read_text()).get("complete") and not args.force:
            print(f"[n6] {name} complete; skip", flush=True)
            continue
        t0 = time.time()
        spec = fk.EnsembleSpec(m=m, eps=eps)
        fk.simulate_ensemble(spec, N_PATHS, name=name, tag=TAG, variants=("full",),
                             workers=args.workers, overwrite=args.force,
                             note="N6 visibility thresholds (fb_n6_visibility_thresholds.py)")
        print(f"[n6] {name} done in {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Prominence functionals (vectorised over rows)
# ----------------------------------------------------------------------------


def smoothing_matrix(n: int, bin_width: float, bandwidth: float) -> np.ndarray:
    return fk._smoothing_operator(n, round(bin_width, 14), bandwidth)[2]


def late_maxima(y: np.ndarray, t: np.ndarray, after_time: float) -> list:
    """Local maxima after ``after_time`` with classifier contour-base prominence.

    Mirrors the classifier's maxima scan (strict left, weak right) and contour
    base (higher of the two side minima before the curve exceeds the peak;
    the window edge terminates the probe).  Returns [(prom, i, base_bin)].
    """
    n = y.size
    left = np.r_[False, y[1:-1] > y[:-2], False]
    right = np.r_[False, y[1:-1] >= y[2:], False]
    out = []
    for i in np.flatnonzero(left & right):
        if t[i] <= after_time:
            continue
        h = y[i]
        sides = []
        for step in (-1, 1):
            j = i + step
            lo, lb = h, i
            while 0 <= j < n and y[j] <= h:
                if y[j] < lo:
                    lo, lb = y[j], j
                j += step
            sides.append((lo, lb))
        base, bb = max(sides, key=lambda pr: pr[0])
        out.append((float(h - base), int(i), int(bb)))
    return out


def late_prominence(dens: np.ndarray, t: np.ndarray, after_time: float) -> tuple:
    """(largest relative prominence after after_time, its time, number of late maxima)."""
    y = np.asarray(dens, float)
    gmax = float(y.max())
    if gmax <= 0:
        return 0.0, None, 0
    lm = late_maxima(y, t, after_time)
    if not lm:
        return 0.0, None, 0
    prom, i, _ = max(lm)
    return prom / gmax, float(t[i]), len(lm)


def r_versions(P: np.ndarray, edges: np.ndarray, after_time: float, A: np.ndarray,
               with_protocol: bool = True, protocol_modes: bool = False) -> dict:
    """P: (nB, K) bin masses -> r_s (smoothed), r_u (unsmoothed), r_P (protocol, z>=5 at 1e6).

    r_P reproduces fk.last_mode_prominence(fk.classify_expected(..., walkers=1e6)):
    the largest relative prominence among late maxima whose covariance-aware
    z = prominence / sqrt(sum_k p_k c_k^2 / N), c = (A[i] - A[base]) / bin, is >= 5.
    """
    bw = np.diff(edges)
    b0 = float(bw[0])
    t = 0.5 * (edges[:-1] + edges[1:])
    dens = P / bw[None, :]
    sm = np.einsum("bj,ij->bi", dens, A)
    nB = P.shape[0]
    out = {"r_s": np.zeros(nB), "r_u": np.zeros(nB), "t_s": [None] * nB, "t_u": [None] * nB}
    if with_protocol:
        out["r_P"] = np.zeros(nB)
    if protocol_modes:
        out["modes_P"] = np.zeros(nB, int)
        out["r_P_classify_expected"] = np.zeros(nB)
    for k in range(nB):
        g_s = float(sm[k].max())
        lm = late_maxima(sm[k], t, after_time) if g_s > 0 else []
        if lm:
            prom, i, _ = max(lm)
            out["r_s"][k], out["t_s"][k] = prom / g_s, float(t[i])
        if with_protocol and lm:
            best = 0.0
            for prom, i, bb in lm:
                c = (A[i] - A[bb]) / b0
                sig = math.sqrt(max(float(np.sum(P[k] * c * c)) / PROTOCOL_WALKERS, 0.0))
                z = prom / sig if sig > 0 else math.inf
                if z >= 5.0 and prom / g_s > best:
                    best = prom / g_s
            out["r_P"][k] = best
        out["r_u"][k], out["t_u"][k], _ = late_prominence(dens[k], t, after_time)
        if protocol_modes:
            cls = fk.classify_expected(P[k], edges, walkers=PROTOCOL_WALKERS, bandwidth=BANDWIDTH)
            out["modes_P"][k] = int(cls["mode_count"])
            out["r_P_classify_expected"][k] = fk.last_mode_prominence(cls, after_time)
    return out


def crossing(Bs: np.ndarray, r: np.ndarray, p: float, rise_first: bool = False) -> tuple:
    """First down-crossing of level p (log-linear in B). Returns (B_p, i, status).

    rise_first (protocol version r_P): at B << 1 the expected kill count is too
    small for the 5-sigma gate, so r_P starts at 0; the crossing is then the
    first down-crossing after r_P first reaches p.
    """
    i0 = 0
    if rise_first:
        above = np.flatnonzero(np.asarray(r) >= p)
        if above.size == 0:
            return None, None, "never_passes_gate_and_floor"
        i0 = int(above[0])
    elif r[0] < p:
        return None, None, "below_at_B_lo"
    for i in range(i0, Bs.size - 1):
        if r[i] >= p and r[i + 1] < p:
            x0, x1 = math.log(Bs[i]), math.log(Bs[i + 1])
            y0, y1 = r[i] - p, r[i + 1] - p
            return float(math.exp(x0 + (x1 - x0) * y0 / (y0 - y1))), i, "interpolated"
    return None, None, "censored"


# ----------------------------------------------------------------------------
# Analysis of one cell
# ----------------------------------------------------------------------------


def late_ess(ens, w, Bs, after_edge_index: int, hi_edge_index: int) -> dict:
    """ESS of per-path late-basin FK weights y = e^{-B X(a)} (1 - e^{-B (X(b)-X(a))})."""
    s1 = np.zeros(len(Bs))
    s2 = np.zeros(len(Bs))
    idx = np.array([after_edge_index, hi_edge_index])
    for _, dX, _ in ens.iter_chunks("full"):
        Xs, seg = fk._selected_exposure(dX, w, idx)
        for ib, B in enumerate(Bs):
            y = np.exp(-B * Xs[:, 0]) * (-np.expm1(-B * seg[:, 0]))
            s1[ib] += y.sum()
            s2[ib] += (y * y).sum()
    N = ens.n_paths
    ess = np.where(s2 > 0, s1 * s1 / np.where(s2 > 0, s2, 1.0), 0.0)
    return {"B": [float(b) for b in Bs], "late_basin_mass": (s1 / N).tolist(),
            "ess": ess.tolist(), "ess_fraction": (ess / N).tolist()}


def analyze_cell(m: int, eps: float, n_boot: int, rng_seed: int) -> dict:
    t_start = time.time()
    name = ensemble_name(m, eps)
    ens = fk.load_ensemble(name)
    ens.cache_in_memory = True
    spec = ens.spec
    w = np.full(m, 1.0 / m)
    valleys = fk.g_valley_times(spec, w)
    res = {"cell": cell_label(m, eps), "m": m, "eps": eps, "ensemble": name,
           "ensemble_tag": int(ens.index["tag"]), "seed": int(ens.index["seed"]),
           "seed_entropy": ens.index["seed_entropy"], "n_paths": ens.n_paths,
           "weights": w.tolist(), "g_window_valleys": valleys}
    if len(valleys) < m - 1:
        res["status"] = "G_has_fewer_than_m_minus_1_window_valleys"
    after_time = valleys[-1] if valleys else fk.WINDOW[0]
    res["after_time_last_G_valley"] = after_time
    edges = ens.edges[ens.window_index]
    t = 0.5 * (edges[:-1] + edges[1:])
    K = edges.size - 1
    A = smoothing_matrix(K, fk.WINDOW_BIN, BANDWIDTH)

    coarse = np.geomspace(B_LO, B_HI, N_COARSE)
    tab = fk.survival_table(ens, coarse, w, index=ens.window_index, groups=GROUPS)
    N = tab["N"]
    Pg = tab["P_group_sums"]            # (G, nB, K)
    sizes = tab["group_sizes"]
    P = Pg.sum(0) / N
    rc = r_versions(P, edges, after_time, A, protocol_modes=True)
    res["coarse"] = {"B": coarse.tolist(), "r_s": rc["r_s"].tolist(), "r_u": rc["r_u"].tolist(),
                     "r_P": rc["r_P"].tolist(), "modes_protocol_1e6": rc["modes_P"].tolist(),
                     "r_P_check_max_abs_diff_vs_classify_expected": float(
                         np.max(np.abs(rc["r_P"] - rc["r_P_classify_expected"]))),
                     "late_time_s": rc["t_s"], "late_time_u": rc["t_u"],
                     "window_mass": P.sum(1).tolist()}
    # smoothed-law window maxima count (no gate) at each coarse B, for context
    # brackets per (version, p)
    versions = ("r_s", "r_u", "r_P")
    brackets = {}
    fine_pts = []
    for ver in versions:
        for p in P_LEVELS:
            bx, i, st = crossing(coarse, rc[ver], p, rise_first=(ver == "r_P"))
            if len(valleys) < m - 1:
                bx, i, st = None, None, "undefined_G_lacks_last_valley"
            brackets[(ver, p)] = (bx, i, st)
            if bx is not None:
                lo = coarse[max(i - 1, 0)]
                hi = coarse[min(i + 2, coarse.size - 1)]
                fine_pts.append(np.geomspace(lo, hi, 25))
    fine = np.unique(np.concatenate(fine_pts)) if fine_pts else np.array([])
    out_bp = {}
    if fine.size:
        tabf = fk.survival_table(ens, fine, w, index=ens.window_index, groups=GROUPS)
        Pgf = tabf["P_group_sums"]
        Pf = Pgf.sum(0) / N
        rf = r_versions(Pf, edges, after_time, A, protocol_modes=True)
        # jackknife + bootstrap replicates on fine and coarse grids
        rng = np.random.Generator(np.random.PCG64(rng_seed))
        G = GROUPS
        boots = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)   # (nboot, G)
        jack = np.ones((G, G)) - np.eye(G)
        rep_w = np.concatenate([jack, boots.astype(float)], axis=0)    # (G+nboot, G)
        need = {ver for (ver, p), (bx, _, _) in brackets.items() if bx is not None}
        rep_fine = {v: np.zeros((rep_w.shape[0], fine.size)) for v in need}
        # coarse replicates only in the coarse brackets (fallback when a replicate's
        # crossing leaves the fine grid)
        cmask = np.zeros(coarse.size, bool)
        for (ver, p), (bx, i, _) in brackets.items():
            if bx is not None:
                cmask[max(i - 3, 0): min(i + 5, coarse.size)] = True
        cidx = np.flatnonzero(cmask)
        rep_coarse = {v: np.full((rep_w.shape[0], coarse.size), np.nan) for v in need}
        for k in range(rep_w.shape[0]):
            cw = rep_w[k]
            nn = float(cw @ sizes)
            Prf = np.einsum("g,gbk->bk", cw, Pgf) / nn
            Prc = np.einsum("g,gbk->bk", cw, Pg[:, cidx, :]) / nn
            rr_f = r_versions(Prf, edges, after_time, A, with_protocol=("r_P" in need))
            rr_c = r_versions(Prc, edges, after_time, A, with_protocol=("r_P" in need))
            for v in need:
                rep_fine[v][k] = rr_f[v]
                rep_coarse[v][k, cidx] = rr_c[v]
        for ver in versions:
            for p in P_LEVELS:
                bx, i, st = brackets[(ver, p)]
                key = f"{ver}_p{int(round(100 * p))}"
                if bx is None:
                    out_bp[key] = {"B_p": None, "status": st}
                    continue
                bf, _, stf = crossing(fine, rf[ver], p, rise_first=(ver == "r_P"))
                reps = []
                for k in range(rep_w.shape[0]):
                    b_k, _, _ = crossing(fine, rep_fine[ver][k], p, rise_first=(ver == "r_P"))
                    if b_k is None:
                        rc_k = rep_coarse[ver][k]
                        ok = ~np.isnan(rc_k)
                        b_k, _, _ = crossing(coarse[ok], rc_k[ok], p, rise_first=(ver == "r_P"))
                    reps.append(b_k)
                jr = np.array([x for x in reps[:G] if x is not None])
                br = np.array([x for x in reps[G:] if x is not None])
                se_j = float(math.sqrt((jr.size - 1) / jr.size * np.sum((jr - jr.mean()) ** 2))) \
                    if jr.size > 1 else None
                ci = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))] \
                    if br.size > 10 else None
                out_bp[key] = {"B_p": bf if bf is not None else bx, "status": stf if bf is not None
                               else "coarse_only", "B_p_coarse": bx, "jackknife_se": se_j,
                               "bootstrap_ci95": ci, "bootstrap_se": float(br.std(ddof=1))
                               if br.size > 2 else None,
                               "n_boot_valid": int(br.size), "n_boot": int(n_boot),
                               "n_jack_valid": int(jr.size)}
        res["fine"] = {"B": fine.tolist(), "r_s": rf["r_s"].tolist(), "r_u": rf["r_u"].tolist(),
                       "r_P": rf["r_P"].tolist(), "modes_protocol_1e6": rf["modes_P"].tolist()}
    else:
        for ver in versions:
            for p in P_LEVELS:
                bx, i, st = brackets[(ver, p)]
                out_bp[f"{ver}_p{int(round(100 * p))}"] = {"B_p": None, "status": st}
    res["B_p"] = out_bp
    # effective sample size of late-basin weights at large B
    ai = int(np.argmin(np.abs(ens.edges - after_time)))
    hi = int(ens.window_index[-1])
    ess_B = [4.0, 8.0, 16.0, 32.0, 64.0]
    bp5 = out_bp.get("r_s_p5", {}).get("B_p")
    if bp5:
        ess_B = sorted(set(ess_B + [float(bp5)]))
    res["late_basin_ess"] = late_ess(ens, w, ess_B, ai, hi)
    res["late_basin_ess"]["after_edge"] = float(ens.edges[ai])
    res["late_basin_ess"]["min_ess"] = float(min(res["late_basin_ess"]["ess"]))
    res["late_basin_ess"]["fallback_needed_ess_lt_1e3"] = bool(res["late_basin_ess"]["min_ess"] < 1e3)
    res["bootstrap_rng"] = {"generator": "PCG64", "seed": int(rng_seed)}
    res["wall_seconds"] = time.time() - t_start
    ens.clear_cache()
    return res


# ----------------------------------------------------------------------------
# eps -> 0 limits (H1 fixed-B law)
# ----------------------------------------------------------------------------


def h1_limit_ratio_smoothed(m: int, B: float) -> float:
    """eps -> 0 at fixed smoothing bandwidth: peaks become kernel-shaped with
    masses M_j (stick-breaking law); well-separated peaks => r -> M_m / max_j M_j."""
    spec = fk.EnsembleSpec(m=m, eps=0.05)
    _, M = fk.limit_masses(B, np.full(m, 1.0 / m), spec)
    return float(M[-1] / M.max())


def h1_limit_ratio_unsmoothed(m: int, B: float, n_y: int = 4001) -> float:
    """eps -> 0 without smoothing: ratio of passage-profile peak heights of the H1
    limit law, f_j(t) ~ e^{-sum_{i<j} lam_i} g_j((t - t_j)/eps)/eps with
    g(tau) = lam v int phi_rho(y) e^{-lam Phi_rho(y)} phi_s(v tau - y) dy."""
    spec = fk.EnsembleSpec(m=m, eps=0.05)
    p = spec.model()
    lam = fk.lambdas(B, np.full(m, 1.0 / m), spec)
    s = math.sqrt(p.d0 / (2.0 * p.gamma))
    rho = p.rho
    y = np.linspace(-12 * max(rho, s), 12 * max(rho, s), n_y)
    dy = y[1] - y[0]
    erf = np.vectorize(math.erf)
    Phi = 0.5 * (1.0 + erf(y / (rho * math.sqrt(2.0))))
    phi_r = np.exp(-0.5 * (y / rho) ** 2) / (math.sqrt(2 * math.pi) * rho)
    heights = []
    surv = 1.0
    for j, tj in enumerate(spec.times()):
        v = float(spec.mu_prime_abs(tj))
        h = lam[j] * phi_r * np.exp(-lam[j] * Phi)
        tau = np.linspace(-10 * (rho + s) / v, 10 * (rho + s) / v, 4001)
        g = v * (np.exp(-0.5 * ((v * tau[:, None] - y[None, :]) / s) ** 2) @ h) * dy \
            / (math.sqrt(2 * math.pi) * s)
        heights.append(surv * g.max())
        surv *= math.exp(-lam[j])
    heights = np.array(heights)
    return float(heights[-1] / heights.max())


def h1_limit_thresholds() -> dict:
    out = {}
    for m in M_LIST:
        for kind, fn in (("smoothed", h1_limit_ratio_smoothed),
                         ("unsmoothed", h1_limit_ratio_unsmoothed)):
            for p in P_LEVELS:
                lo, hi = 1e-3, 1e3
                f = lambda B: fn(m, B) - p  # noqa: E731
                if f(lo) < 0:
                    out[f"m{m}_{kind}_p{int(round(100 * p))}"] = None
                    continue
                for _ in range(80):
                    mid = math.sqrt(lo * hi)
                    if f(mid) >= 0:
                        lo = mid
                    else:
                        hi = mid
                out[f"m{m}_{kind}_p{int(round(100 * p))}"] = math.sqrt(lo * hi)
    out["definition"] = ("eps -> 0 limit at fixed B and fixed smoothing bandwidth: smoothed r -> "
                         "M_m/max_j M_j with stick-breaking masses M_j (lambda_j = B w_j/(W|mu'(t_j)|)); "
                         "unsmoothed r -> ratio of H1 passage-profile peak heights; contact -> 1 "
                         "as eps -> 0 in the production geometry (|r_par0| = 0.1 < a = 0.4)")
    return out


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def cmd_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (m, eps) in enumerate(parse_cells(args.cells)):
        path = OUT / f"n6_cell_{cell_label(m, eps)}.json"
        if path.exists() and not args.force:
            print(f"[n6] {path.name} exists; skip", flush=True)
            continue
        name = ensemble_name(m, eps)
        ip = fk.index_path(name)
        if not ip.exists() or not json.loads(ip.read_text()).get("complete"):
            print(f"[n6] ensemble {name} missing/incomplete; skip", flush=True)
            continue
        seed = fk.BASE_SEED * 100 + TAG * 10 + (m * 17 + int(round(eps * 1000))) % 10
        r = analyze_cell(m, eps, args.boot, seed)
        core.write_json(path, fk._jsonable(r))
        bp = {kk: (vv.get("B_p"), vv.get("jackknife_se"), vv.get("status"))
              for kk, vv in r["B_p"].items()}
        print(f"[n6] {cell_label(m, eps)} {r['wall_seconds']:.0f}s  {bp}", flush=True)


def cmd_summary(args):
    cells = {}
    for m, eps in parse_cells(None):
        path = OUT / f"n6_cell_{cell_label(m, eps)}.json"
        if path.exists():
            cells[cell_label(m, eps)] = json.loads(path.read_text())
    lim = h1_limit_thresholds()
    s2 = {}
    for lab, env in S2_ENVELOPES.items():
        c = cells.get(lab)
        if c is None:
            continue
        row = {"envelope": env, "source": S2_SOURCE}
        for ver in ("r_P", "r_s", "r_u"):
            b = c["B_p"].get(f"{ver}_p5", {})
            bp = b.get("B_p")
            row[f"{ver}_B5"] = bp
            row[f"{ver}_B5_ci95"] = b.get("bootstrap_ci95")
            row[f"{ver}_inside_envelope"] = (bp is not None and env[0] <= bp <= env[1])
        s2[lab] = row
    table = []
    for lab, c in cells.items():
        row = {"cell": lab, "m": c["m"], "eps": c["eps"], "ensemble": c["ensemble"],
               "tag": c["ensemble_tag"], "min_late_ess": c["late_basin_ess"]["min_ess"]}
        for key, v in c["B_p"].items():
            row[key] = v.get("B_p")
            row[key + "_status"] = v.get("status")
            row[key + "_ci95"] = v.get("bootstrap_ci95")
            row[key + "_se_jack"] = v.get("jackknife_se")
        row["r_s_at_B_lo"] = c["coarse"]["r_s"][0]
        table.append(row)
    summary = {
        "analysis": "N6 visibility thresholds B_p on the exact law (FK estimator, N0 API)",
        "definitions": {
            "r_s": "last-mode relative prominence of the bandwidth-0.04 smoothed exact expected density (no significance gate)",
            "r_u": "same on the unsmoothed 0.02-bin exact density",
            "r_P": "protocol P: smoothed exact law, late maximum counted only if covariance-aware z >= 5 at 1e6 expected walkers (prominence-floor crossing under protocol P, formerly B_op)",
            "last_mode": "maxima later than the last window valley of the semi-analytic free-exposure clock G",
            "B_p": "first down-crossing of r(B) through p, log-linear interpolation on a 25-point geometric fine grid",
            "CI": "path-group bootstrap (40 groups, percentile 95%) and delete-one-group jackknife SE",
            "budget_range": [B_LO, B_HI],
        },
        "p_levels": list(P_LEVELS),
        "table": table,
        "s2_cross_check": s2,
        "eps_to_zero_limits": lim,
        "scout_h1_values_bop_h1_json": {
            "source": "notes/gap_diagnosis_20260923/theory_scout/bop_h1.json (protocol classifier on the H1 law at finite eps)",
            "m2": {"0.05": 6.364640975463803, "0.025": 7.3138258658398065,
                   "0.01": 8.478339979733498, "0.005": 8.837075837599956},
            "m3": {"0.05": 3.786359405333882, "0.025": 4.445866818685324,
                   "0.01": 5.2797202162554, "0.005": 5.612307592678264}},
        "seeds": {"base_seed": fk.BASE_SEED, "tag_new_ensembles": TAG,
                  "tag_reused_n0": fk.TAGS["N0"]},
    }
    core.write_json(OUT / "n6_visibility_thresholds.json", fk._jsonable(summary))
    print(json.dumps({"s2": s2, "limits": lim}, indent=1, default=str))
    return summary


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    S = json.loads((OUT / "n6_visibility_thresholds.json").read_text())
    lim = S["eps_to_zero_limits"]
    cells = {}
    for m, eps in parse_cells(None):
        path = OUT / f"n6_cell_{cell_label(m, eps)}.json"
        if path.exists():
            cells[(m, eps)] = json.loads(path.read_text())
    colors = {0.01: core.OI_BLUE, 0.05: core.OI_VERMILLION, 0.10: core.OI_GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")
    for ax, m in zip(axes, M_LIST):
        for p in P_LEVELS:
            key = f"r_s_p{int(round(100 * p))}"
            ku = f"r_u_p{int(round(100 * p))}"
            xs, ys, lo, hi, xu, yu, xc = [], [], [], [], [], [], []
            for eps in EPS_LIST:
                c = cells.get((m, eps))
                if c is None:
                    continue
                v = c["B_p"].get(key, {})
                u = c["B_p"].get(ku, {})
                xs.append(eps)
                xu.append(eps)
                if v.get("B_p") is not None:
                    ys.append(v["B_p"])
                    ci = v.get("bootstrap_ci95") or [v["B_p"], v["B_p"]]
                    lo.append(v["B_p"] - ci[0])
                    hi.append(ci[1] - v["B_p"])
                else:
                    ys.append(np.nan); lo.append(0.0); hi.append(0.0)
                    if v.get("status") == "censored":
                        xc.append(eps)
                yu.append(u["B_p"] if u.get("B_p") is not None else np.nan)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colors[p], marker="o", ms=3.2, lw=1.0,
                        capsize=1.5, label=f"$p={100 * p:g}\\%$, smoothed (95% CI)")
            ax.plot(xu, yu, color=colors[p], marker="o", mfc="white", ms=3.0, lw=0.7, ls=":",
                    label=f"$p={100 * p:g}\\%$, unsmoothed")
            if xc:
                ax.plot(xc, [B_HI] * len(xc), ls="none", marker="^", ms=5, color=colors[p],
                        label=f"$p={100 * p:g}\\%$: $B_p>{B_HI:g}$ (censored)")
            lv = lim.get(f"m{m}_smoothed_p{int(round(100 * p))}")
            if lv:
                ax.plot([0.0], [lv], marker="<", color=colors[p], ms=5, clip_on=False, ls="none")
        for lab, env in S2_ENVELOPES.items():
            if lab.startswith(f"m{m}_"):
                e = float(lab.split("eps")[1])
                ax.plot([e - 0.008, e - 0.008], env, color="k", lw=3.5, alpha=0.55,
                        solid_capstyle="butt", label="S2 seed envelope (protocol P)")
        ax.axhline(B_HI, color="0.5", lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_xlim(-0.008, 0.262)
        ax.set_xlabel(r"noise amplitude $\varepsilon$ (dimensionless)")
        ax.set_ylabel(r"visibility threshold $B_p$ (dimensionless budget)")
        ax.set_title(f"$m={m}$, equal weights ($\\blacktriangleleft$: $\\varepsilon\\to0$ limit; dashed: scan limit $B={B_HI:g}$)",
                     fontsize=6.8)
        ax.grid(alpha=0.3, which="both")
        if m == 3:
            ax.set_ylim(0.1, 90)
            ax.annotate("$\\varepsilon=0.2$: $r_s<5\\%$\nat all $B$", xy=(0.2, 0.13), fontsize=5.8, ha="center",
                        color="0.3")
            ax.annotate("$\\varepsilon=0.25$: no\nseparated late\npassage in $G$", xy=(0.245, 0.55), fontsize=5.8,
                        ha="center", color="0.3")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6.0, frameon=False)
    written = core.save_figure(fig, core.FIGURES / "fb_n6_visibility_thresholds")
    plt.close(fig)
    # prominence curves
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for ax, m in zip(axes, M_LIST):
        for k, eps in enumerate(EPS_LIST):
            c = cells.get((m, eps))
            if c is None or str(c.get("status", "")).startswith("G_has_fewer"):
                continue
            B = np.array(c["coarse"]["B"])
            r = np.array(c["coarse"]["r_s"])
            ax.plot(B, np.maximum(r, 1e-4), color=cmap(k / (len(EPS_LIST) - 1)), lw=1.0,
                    label=f"$\\varepsilon={eps:g}$")
        for p in P_LEVELS:
            ax.axhline(p, color="0.4", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.set_xlabel(r"budget $B$ (dimensionless)")
        ax.set_ylabel(r"last-mode relative prominence $r_s(B)$")
        ax.set_title(f"$m={m}$, smoothed exact law (bandwidth 0.04)", fontsize=7.5)
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(loc="lower left", fontsize=6.0, ncol=2)
    written += core.save_figure(fig, core.FIGURES / "fb_n6_prominence_curves")
    plt.close(fig)
    print(written)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--cells", default=None)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--force", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--cells", default=None)
    a.add_argument("--boot", type=int, default=1000)
    a.add_argument("--force", action="store_true")
    sub.add_parser("summary")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze, "summary": cmd_summary,
     "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
