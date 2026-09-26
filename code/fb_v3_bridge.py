#!/usr/bin/env python3
"""V3 theory--numerics bridge: how close are the simulated noise levels to the asymptotic regime?

Question (referee item, every simulation): Theorems 2 and 3 are limits eps -> 0 with an eps_0 that is
proved to exist but not computed.  At the anchor designs, how fast do the exact-law basin masses and
peak times approach their limits, and are the empirical rates consistent with the orders the theorems
state?

What the theorems state (read from SM-A, sm/S3_fixed_budget.tex; nothing here is new theory):
  * masses  (Theorem th2:main(a) = main Theorem 2(a)):  |M_j - M_j^lim| <= C (B^2 eps + B eps^2),
    i.e. order eps at fixed B, and the order is sharp (Remark th2:shape(i)).  The leading term is
    explicit (Proposition th2:correction together with |M^mf - M^lim| <= C B eps^2, Lemma th2:mean):
        M_j - M_j^lim = eps * c1_j + O((B^2 + B^3) eps^{3/2} + B eps^2),
        c1_j = -(D0 B^2 / (4 sqrt(pi) rho)) [Pi_{j+1} sum_{i<=j} k_i^2/v_i^3 - Pi_j sum_{i<j} k_i^2/v_i^3],
    k_i = w_i / W^{d-1}; the constants hide exp(-c/eps^2) contact-gate terms whose exponent is small
    at the anchors (Remark th2:scope(i)).
  * peak times (Corollary th3:peaks, Theorem th6:thm-main = main Theorems 2(c), 3):
        t*_j = t_j + eps rho y_*(lambda_j)/v_j + o(eps)          (no rate is proved;
    Remark th3:rate records, without proof, a C^0 profile rate eps^{1/2}).

Data (every point says where it came from; see bridge.json#sources_used_as_primary):
  * masses at eps = 0.1, 0.05: N1 stored FK ensembles (5e5 paths, dt = 1e-3; the published values,
    artifacts/data/exact_m_fixed_budget/N1/n1_allocation_design.json#cells[*].basins_window);
  * equal weights, B = 8: N9a declared FK ensembles (5e5 paths, step resolution; raw accumulators
    under ~/.local-build/prr_fk_ensembles/n9a_*_decl) -- masses at eps = 0.035, 0.025 and peak times
    at eps = 0.05, 0.035, 0.025;
  * everything else: NEW declared-mode FK ensembles of this driver (tag 301, 1e6 paths per (m, eps),
    one ensemble serves the four anchor designs of that m), step-resolution exact densities.  N1
    stores exposures on the 0.02 grid only, so it gives no step-level peak times;
  * mass decomposition M - M^lim = (M - M^mf) + (M^mf - M^lim): the new ensembles at every eps
    (M and M^mf share paths; joint jackknife SEs over the 20 path groups);
  * cross-checks only (not plotted): HPC_headline direct kill (2e7 walkers, eps 0.05/0.1; its peak
    times are vertices of a 0.04-bandwidth smoothed histogram, not exact-density peaks, so they are not
    used for the peak panel), and new ensembles against N1 / N9a where both exist.
  Seeds: the fk module derives the seed entropy from the spec without the target times, so the m = 2
  and m = 3 ensembles at the same (eps, dt) use common random numbers; different eps (or dt) are
  independent streams.

Anchor designs: m in {2, 3} (targets (1.0, 2.5) and (0.8, 1.6, 2.8)), allocation in {equal, max-min w*},
B in {2, 8} -- the 8 cells of HPC_headline / N1 DK_CELLS.  Basins: theorem convention (window cuts
tau = 0.5, T = 3.5; interior cuts = N1 geometric midpoints snapped to the 0.02 grid).

Time-step bias: every FK law is exact for the Euler--Maruyama chain.  Peak times are critical points
of the step-level law p_n/dt at the stamps t_n = n dt; for such point evaluations the end-of-step stamp
gives no dt/2 shift (p_n = S(t_{n-1})(1 - e^{-K_n dt}) = f(t_n) dt + O(dt^2); the dt/2 shift of SM-B
Sections N5 / HPC_N5full, (dt/2)(1 + gamma t), belongs to laws binned on right-closed bins), and only the
Euler--Maruyama mean path runs ahead, by gamma t dt/2.  Peak times are therefore corrected by
+gamma t dt/2, which the dt ladder at eps = 0.0125 (subcommand dtcheck, dt in {1e-3, 5e-4, 2.5e-4})
tests; its bound gives the systematic PEAK_DT_SYS * dt added in quadrature.  Masses: the chain's mean
path moves by gamma dt (z - zbar) per step, so every exposure is a Riemann sum on a geometric grid and
is low by the relative factor gamma dt/2 (checked on the exact EM mean exposure); masses are corrected by
+(gamma dt/2) B dM^lim_j/dB, with half of it as systematic sigma (the ladder tests the slope too).

Subcommands (from code/):
    python3 fb_v3_bridge.py simulate [--only 2:0.025,3:0.025] [--paths 1e6] [--workers 3]
    python3 fb_v3_bridge.py dtcheck      # dt ladder at eps = 0.0125 (validates the peak-time correction)
    python3 fb_v3_bridge.py analyze      # -> artifacts/data/exact_m_fixed_budget/V3_bridge/bridge.json
                                         #    (fits, decomposition, dt ladder, verdict)
    python3 fb_v3_bridge.py figure       # -> artifacts/figures/fb_v3_bridge.{pdf,png}
    python3 fb_v3_bridge.py snippet      # -> <scratch>/prep/bridge_snippet.tex
Python: Xcode /usr/bin/python3 (numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4).  Machine etiquette: at
most 3 worker processes (fk.MAX_WORKERS).
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import fb_n1_allocation_design as n1  # noqa: E402

REPORT = HERE.parents[1]
DATA = fk.FB_DATA
OUT_DIR = DATA / "V3_bridge"
OUT_JSON = OUT_DIR / "bridge.json"
INDEX_DIR_V3 = OUT_DIR / "fk_index"
FIG_STEM = REPORT / "artifacts" / "figures" / "fb_v3_bridge"
SCRATCH = Path("<scratch>/"
               "prep")
SNIPPET = SCRATCH / "bridge_snippet.tex"

TAG = 301                       # new stream (unused elsewhere: data tags 72..98, 201..217)
SEED = fk.BASE_SEED
N_PATHS = 1_000_000
CHUNK = 50_000
BATCH = 50_000                  # 20 path groups of 5e4 -> jackknife / batch-means SEs
EPS_LADDER = (0.1, 0.07, 0.05, 0.035, 0.025, 0.0125)
DT = {0.1: 1e-3, 0.07: 1e-3, 0.05: 1e-3, 0.035: 5e-4, 0.025: 2.5e-4, 0.0125: 2.5e-4}
M_LIST = (2, 3)
ALLOCS = ("equal", "maxmin")
B_LIST = (2.0, 8.0)
# run order: cheap eps = 0.1 first (pipeline check), then the costly small-eps points
RUN_ORDER = [(2, 0.1), (3, 0.1), (2, 0.0125), (3, 0.0125), (2, 0.025), (3, 0.025),
             (2, 0.035), (3, 0.035), (2, 0.05), (3, 0.05), (2, 0.07), (3, 0.07)]

# dt-bias model (SM-B N5 / HPC_N5full)
PEAK_DT_SYS = 0.2               # systematic sigma of the gamma t dt/2 peak-time correction, in units of dt
MASS_DT_BIAS_PER_DT = 0.15      # SM-B N5 bound (1.5e-4 at dt = 1e-3), kept for the record only
# Euler--Maruyama exposure bias: the chain's mean path moves by gamma dt (z - zbar) per step, so every
# passage exposure is a Riemann sum on a geometric grid, low by the relative factor gamma dt / 2 (exact
# EM mean exposure at eps = 0.0125: 0.3396521, 0.3397371, 0.3397795, 0.3398008 at dt = 1e-3 ... 1.25e-4,
# i.e. -gamma dt/2 relative).  The chain law is the law at the budget B (1 - gamma dt/2), so masses are
# corrected by + (gamma dt / 2) B dM^lim_j/dB, with half the correction as systematic sigma.
MASS_DT_REL_SYS = 0.5


def ens_name(m: int, eps: float) -> str:
    return f"v3b_m{m}_eps{eps:g}"


def spec_for(m: int, eps: float) -> fk.EnsembleSpec:
    return fk.EnsembleSpec(m=m, eps=eps, dt=DT[eps])


def cells():
    return [(m, a, B) for m in M_LIST for a in ALLOCS for B in B_LIST]


def cell_key(m, a, B) -> str:
    return f"m{m}_{a}_B{B:g}"


def declared_items(spec: fk.EnsembleSpec) -> list:
    out = []
    for a in ALLOCS:
        for B in B_LIST:
            ds = n1.design(spec, B, a)
            out.append({"B": float(B), "w": [float(x) for x in ds["w"]], "variant": "full",
                        "label": f"{a}_B{B:g}"})
    return out


def _use_v3_index():
    fk.INDEX_DIR = INDEX_DIR_V3
    INDEX_DIR_V3.mkdir(parents=True, exist_ok=True)


# ============================================================================
# simulate
# ============================================================================


def cmd_simulate(args) -> None:
    _use_v3_index()
    todo = RUN_ORDER
    if args.only:
        want = {(int(x.split(":")[0]), float(x.split(":")[1])) for x in args.only.split(",")}
        todo = [p for p in RUN_ORDER if p in want]
    for m, eps in todo:
        spec = spec_for(m, eps)
        name = ens_name(m, eps)
        t0 = time.time()
        print(f"[v3] simulate {name} dt={spec.dt} paths={int(args.paths)}", flush=True)
        ens = fk.simulate_ensemble(spec, int(args.paths), name=name, tag=TAG, seed=SEED,
                                   mode="declared", declared=declared_items(spec),
                                   workers=args.workers, chunk=CHUNK, batch=BATCH,
                                   note="V3 bridge (fb_v3_bridge.py): anchor designs, "
                                        "eq/maxmin x B in {2, 8}")
        print(f"[v3] {name} complete={ens.index['complete']} wall={time.time() - t0:.0f}s "
              f"core={ens.index['process_seconds_total']:.0f}s", flush=True)


# ============================================================================
# theory (limit law, predicted peak offsets, first-order mass coefficient)
# ============================================================================


def _Phi(x):
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(x, float) / math.sqrt(2.0))


_U = np.arange(-16.0, 16.0, 0.002)
_PHI_U = np.exp(-0.5 * _U * _U) / math.sqrt(2.0 * math.pi)
_CDF_U = _Phi(_U)


def ystar(lam: float, theta: float) -> float:
    """Unique critical point y_*(lambda) of F_lambda = p_lambda * phi_theta (SM-A Lemma th3:profile).

    Quadrature on u in [-16, 16] (du = 0.002) and bisection on F'_lambda; reproduces
    TH3/local_profile_check.json#profile.rows[*].y_star to 1e-12.
    """
    if lam <= 0:
        return 0.0
    p = lam * _PHI_U * np.exp(-lam * _CDF_U)

    def dF(y):
        z = (y - _U) / theta
        return float(np.sum(p * (-z / theta) * np.exp(-0.5 * z * z)))

    lo, hi = -12.0, 1.0
    flo, fhi = dF(lo), dF(hi)
    if not (flo > 0 > fhi):
        raise RuntimeError("y_* not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = dF(mid)
        if fm > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def theory_cell(m: int, alloc: str, B: float) -> dict:
    spec = fk.EnsembleSpec(m=m, eps=0.05)          # eps enters nothing below
    ds = n1.design(spec, B, alloc)
    w = np.asarray(ds["w"], float)
    v = np.asarray(ds["speeds"], float)
    lam = np.asarray(ds["lambda"], float)
    A = spec.torus_w ** spec.n_perp
    k = w / A
    cum = np.concatenate([[0.0], np.cumsum(lam)])
    Pi = np.exp(-cum)                               # Pi[j] = Pi_{j+1} (0-based j)
    Mlim = Pi[:-1] * (1.0 - np.exp(-lam))
    rho, D0 = spec.rho, spec.d0
    sZ = math.sqrt(D0 / (2.0 * spec.gamma))
    theta = sZ / rho
    ys = np.array([ystar(float(l), theta) for l in lam])
    s_star = rho * ys / v
    K = np.concatenate([[0.0], np.cumsum(k * k / v**3)])
    pref = D0 * B * B / (4.0 * math.sqrt(math.pi) * rho)
    c1 = np.array([-pref * (Pi[j + 1] * K[j + 1] - Pi[j] * K[j]) for j in range(m)])
    # eps^2 term of the mean-field masses: Lambda_eps(s_j) = Lambda^0_j + gamma^2 S_*^2 eps^2
    # sum_{i<=j} k_i/v_i^3 + ... (SM-A Lemma th2:mean; stated in Remark th2:shape(i))
    S2 = sZ * sZ + rho * rho
    dl = np.concatenate([[0.0], np.cumsum(spec.gamma**2 * S2 * k / v**3)])
    c2mf = np.array([B * (Pi[j + 1] * dl[j + 1] - Pi[j] * dl[j]) for j in range(m)])
    # B dM^lim_j/dB (all lambda_i proportional to B): used for the Euler--Maruyama exposure bias
    BdMdB = np.array([-cum[j] * Mlim[j] + Pi[j] * lam[j] * math.exp(-lam[j]) for j in range(m)])
    return {"m": m, "allocation": alloc, "B": float(B), "w": w.tolist(), "speeds": v.tolist(),
            "t_j": list(spec.times()), "z_j": [float(c) for c in spec.centres()],
            "lambda": lam.tolist(), "Pi": Pi[:-1].tolist(), "M_lim": Mlim.tolist(),
            "theta": theta, "y_star": ys.tolist(), "s_star": s_star.tolist(),
            "c1_first_order_mass": c1.tolist(),
            "c2_mean_field_mass": c2mf.tolist(),
            "c2_formula": ("M_j^mf - M_j^lim = eps^2 c2_j + ...: c2_j = B [Pi_{j+1} delta_j - Pi_j "
                           "delta_{j-1}], delta_j = gamma^2 S_*^2 sum_{i<=j} k_i/v_i^3 (leading eps^2 "
                           "term of Lambda_eps, SM-A Lemma th2:mean / Remark th2:shape(i); contact-gate "
                           "terms exp(-c/eps^2) omitted)"),
            "B_dMlim_dB": BdMdB.tolist(),
            "c1_formula": ("SM-A Prop. th2:correction + Lemma th2:mean: M_j - M_j^lim = eps*c1_j + "
                           "O((B^2+B^3) eps^{3/2}); c1_j = -(D0 B^2/(4 sqrt(pi) rho)) [Pi_{j+1} "
                           "sum_{i<=j} k_i^2/v_i^3 - Pi_j sum_{i<j} k_i^2/v_i^3], k_i = w_i/W^{d-1}")}


def contact_at_targets(m: int, eps: float) -> list:
    spec = fk.EnsembleSpec(m=m, eps=eps)
    return [float(x) for x in fk.mean_contact_curve(spec, spec.contact_a,
                                                    np.asarray(spec.times(), float))]


PEAK_DT_COEF = 0.5          # validated by the dt ladder (cmd dtcheck)

# ============================================================================
# estimates from declared (step-resolution) FK ensembles
# ============================================================================

CUTS = {2: [0.5, 1.5, 3.5], 3: [0.5, 1.12, 2.02, 3.5]}   # = N1 basins_window.cut_times


def _groups(batch_p: np.ndarray, G: int = 20) -> np.ndarray:
    nb = batch_p.shape[0]
    if nb % G:
        G = nb
    return batch_p.reshape(G, nb // G, -1).mean(axis=1)


def _vertex(t: np.ndarray, p: np.ndarray, i0: int, frac: float = 0.75, deg: int = 4):
    """Critical point of the step density near i0: degree-4 LSQ fit on the contiguous
    region where p >= frac * p[i0] (at least +-4 steps); root of the fitted derivative."""
    pk = p[i0]
    lo = i0
    while lo > 0 and p[lo - 1] >= frac * pk:
        lo -= 1
    hi = i0
    while hi < p.size - 1 and p[hi + 1] >= frac * pk:
        hi += 1
    lo = min(lo, i0 - 4)
    hi = max(hi, i0 + 4)
    lo, hi = max(lo, 0), min(hi, p.size - 1)
    tt = t[lo:hi + 1]
    yy = p[lo:hi + 1] / pk
    c, h = t[i0], max(t[hi] - t[i0], t[i0] - t[lo])
    x = (tt - c) / h
    co = np.polyfit(x, yy, deg)
    dr = np.roots(np.polyder(co))
    dr = dr[np.abs(dr.imag) < 1e-9].real
    dr = dr[(dr > -1.0) & (dr < 1.0)]
    if dr.size == 0:
        return float("nan"), (lo, hi)
    x0 = dr[np.argmin(np.abs(dr))]
    return float(c + h * x0), (lo, hi)


def declared_estimates(res: dict, item: int, m: int, eps: float, G: int = 20) -> dict:
    """Window basin masses and exact-density peak times for one declared item."""
    t = np.asarray(res["t"], float)
    dt = float(res["dt"])
    p = np.asarray(res["p_step"][item], float)
    bp = _groups(np.asarray(res["batch_p_step"][item], float), G)
    Gg = bp.shape[0]
    cuts = CUTS[m]
    masses, ses, peaks = [], [], []
    for j in range(m):
        a, b = cuts[j], cuts[j + 1]
        last = j == m - 1
        msk = (t >= a - 1e-12) & ((t <= b + 1e-12) if last else (t < b - 1e-12))
        Mj = float(p[msk].sum())
        gm = bp[:, msk].sum(axis=1)
        se = float(gm.std(ddof=1) / math.sqrt(Gg))
        masses.append(Mj)
        ses.append(se)
        idx = np.flatnonzero(msk)
        i0 = int(idx[np.argmax(p[idx])])
        interior = (i0 - idx[0] >= 5) and (idx[-1] - i0 >= 5) and \
            (p[i0] > p[i0 - 5]) and (p[i0] > p[i0 + 5])
        if not interior:
            peaks.append({"exists": False, "note": "maximum of the step density at a basin edge: "
                                                   "no interior peak in this basin"})
            continue
        tv, (lo, hi) = _vertex(t, p, i0)
        jack = []
        for g in range(Gg):
            pg = (Gg * p - bp[g]) / (Gg - 1)
            ig = int(idx[np.argmax(pg[idx])])
            jack.append(_vertex(t, pg, ig)[0])
        jack = np.asarray(jack)
        se_jk = float(math.sqrt((Gg - 1) / Gg * np.sum((jack - jack.mean()) ** 2)))
        # method check: quadratic vertex on the top 95 %
        tq, _ = _vertex(t, p, i0, frac=0.95, deg=2)
        gamma = 1.0
        # point evaluations p_n/dt at the stamps t_n carry no end-of-step dt/2 shift (that shift
        # belongs to binned laws); the Euler--Maruyama mean path runs ahead by gamma t dt/2
        corr = PEAK_DT_COEF * dt * gamma * tv
        peaks.append({"exists": True, "t_peak_raw": tv, "t_peak_quadratic_top95": tq,
                      "fit_window": [float(t[lo]), float(t[hi])], "se_jackknife": se_jk,
                      "dt_correction": corr, "t_peak": tv + corr,
                      "se_total": math.sqrt(se_jk**2 + (PEAK_DT_SYS * dt) ** 2)})
    return {"masses": masses, "se": ses, "se_batch": ses, "groups": Gg, "dt": dt, "N": int(res["N"]),
            "mass_dt_bias_bound": MASS_DT_BIAS_PER_DT * dt, "peaks": peaks,
            "window_mass": float(sum(masses))}


def mean_field_from_declared(res: dict, pair: int, B: float, m: int) -> list:
    t = np.asarray(res["t"], float)
    X = np.asarray(res["mean_X"][pair], float)
    cuts = CUTS[m]
    # unit exposure accumulated up to (and including) the last step before each cut
    L = []
    for c in cuts:
        i = int(np.searchsorted(t, c - 1e-12, side="left")) - 1
        L.append(0.0 if i < 0 else float(X[i]))
    L = np.asarray(L)
    mf = np.exp(-B * L[:-1]) - np.exp(-B * L[1:])
    return mf.tolist()


def _masses_and_mf(t: np.ndarray, p: np.ndarray, X: np.ndarray, B: float, cuts) -> np.ndarray:
    """Window basin masses (step sums, as in declared_estimates) followed by the mean-field
    masses exp(-B X(s_{j-1})) - exp(-B X(s_j)) from the sampled mean exposure X."""
    m = len(cuts) - 1
    out = np.empty(2 * m)
    for j in range(m):
        a, b = cuts[j], cuts[j + 1]
        last = j == m - 1
        msk = (t >= a - 1e-12) & ((t <= b + 1e-12) if last else (t < b - 1e-12))
        out[j] = p[msk].sum()
    L = np.empty(len(cuts))
    for k, c in enumerate(cuts):
        i = int(np.searchsorted(t, c - 1e-12, side="left")) - 1
        L[k] = 0.0 if i < 0 else X[i]
    out[m:] = np.exp(-B * L[:-1]) - np.exp(-B * L[1:])
    return out


def decomposition_from_declared(res: dict, item: int, m: int) -> dict:
    """M_j, M_j^mf and M_j - M_j^mf with JOINT jackknife SEs over the path groups.

    M and M^mf come from the same paths, so the SE of the fluctuation part M - M^mf is much
    smaller than the SE of M; batch_mean_X and batch_p_step are aligned by path group."""
    t = np.asarray(res["t"], float)
    d = res["declared"][item]
    B, pair = float(d["B"]), int(d["pair"])
    p = np.asarray(res["p_step"][item], float)
    bp = np.asarray(res["batch_p_step"][item], float)
    X = np.asarray(res["mean_X"][pair], float)
    bX = np.asarray(res["batch_mean_X"][pair], float)
    G = bp.shape[0]
    if bX.shape[0] != G:
        raise AssertionError("path groups of p and X differ")
    cuts = CUTS[m]
    full = _masses_and_mf(t, p, X, B, cuts)
    jk = np.array([_masses_and_mf(t, (G * p - bp[g]) / (G - 1), (G * X - bX[g]) / (G - 1), B, cuts)
                   for g in range(G)])
    fl_jk = jk[:, :m] - jk[:, m:]

    def jse(a):
        return np.sqrt((G - 1) / G * np.sum((a - a.mean(axis=0)) ** 2, axis=0))

    se = jse(jk)
    return {"M": full[:m].tolist(), "se_M": se[:m].tolist(),
            "M_mf": full[m:].tolist(), "se_M_mf": se[m:].tolist(),
            "fluct": (full[:m] - full[m:]).tolist(), "se_fluct": jse(fl_jk).tolist(),
            "groups": G, "dt": float(res["dt"]), "N": int(res["N"])}


# ============================================================================
# assembly of all sources
# ============================================================================

N1_JSON = DATA / "N1" / "n1_allocation_design.json"
HPC_JSON = DATA / "HPC_headline" / "hpc_headline.json"
N9A_ENS = {(m, e): f"n9a_m{m}_eps{e:g}_decl" for m in (2, 3) for e in (0.05, 0.035, 0.025)}
REL = lambda p: str(Path(p).resolve().relative_to(REPORT))  # noqa: E731


def _load_declared(name: str, v3: bool):
    fk.INDEX_DIR = INDEX_DIR_V3 if v3 else DATA / "fk_ensembles"
    try:
        ens = fk.load_ensemble(name)
    except FileNotFoundError:
        return None, None
    if not ens.index.get("complete", False):
        return None, None
    return ens, ens.declared()


def _item_index(res: dict, B: float, w) -> int | None:
    for i, d in enumerate(res["declared"]):
        if d.get("variant", "full") == "full" and abs(d["B"] - B) < 1e-12 and \
                np.allclose(d["w"], w, atol=1e-9, rtol=0):
            return i
    return None


def collect() -> dict:
    """Every (cell, eps) point from every source; primary selection per the docstring rule."""
    n1j = json.loads(N1_JSON.read_text())
    n1_cells = {(c["m"], c["eps"], c["B"], c["allocation"]): c for c in n1j["cells"]}
    hpc = json.loads(HPC_JSON.read_text())["cells"]
    out = {}
    declared_cache = {}

    def get_decl(kind, m, eps):
        key = (kind, m, eps)
        if key not in declared_cache:
            if kind == "v3b":
                ens, res = _load_declared(ens_name(m, eps), True)
            else:
                nm = N9A_ENS.get((m, eps))
                ens, res = _load_declared(nm, False) if nm else (None, None)
            declared_cache[key] = (ens, res)
        return declared_cache[key]

    for (m, a, B) in cells():
        th = theory_cell(m, a, B)
        w = th["w"]
        ck = cell_key(m, a, B)
        pts = {}
        for eps in EPS_LADDER:
            rec = {"eps": eps, "mass_sources": {}, "peak_sources": {}}
            # N1 stored FK (published)
            c = n1_cells.get((m, eps, B, a))
            if c is not None:
                bw = c["basins_window"]
                if not np.allclose(bw["cut_times"], CUTS[m], atol=1e-9):
                    raise AssertionError("N1 cuts differ")
                rec["mass_sources"]["N1"] = {
                    "masses": bw["mass"], "se": bw["se_batch"], "se_iid": bw["se_iid"],
                    "dt": 1e-3, "N": 500000,
                    "mass_dt_bias_bound": MASS_DT_BIAS_PER_DT * 1e-3,
                    "mean_field": bw.get("mean_field_sampled_Lambda"),
                    "frozen_gate_law": bw.get("frozen_gate_law"),
                    "src": f"{REL(N1_JSON)}#cells[m={m},eps={eps:g},B={B:g},allocation={a}]"
                           ".basins_window.{{mass,se_batch}}"}
            # N9a declared FK (equal weights only)
            if a == "equal":
                ens, res = get_decl("n9a", m, eps)
                if res is not None:
                    ii = _item_index(res, B, w)
                    if ii is not None:
                        est = declared_estimates(res, ii, m, eps)
                        pair = res["declared"][ii]["pair"]
                        est["mean_field"] = mean_field_from_declared(res, pair, B, m)
                        est["src"] = (f"artifacts/data/exact_m_fixed_budget/fk_ensembles/"
                                      f"{N9A_ENS[(m, eps)]}.json (declared item "
                                      f"{res['declared'][ii]['label']}; raw accumulators "
                                      f"~/.local-build/prr_fk_ensembles/{N9A_ENS[(m, eps)]})")
                        rec["mass_sources"]["N9a"] = est
                        rec["peak_sources"]["N9a"] = est
            # new V3 declared FK
            ens, res = get_decl("v3b", m, eps)
            if res is not None:
                ii = _item_index(res, B, w)
                if ii is not None:
                    est = declared_estimates(res, ii, m, eps)
                    pair = res["declared"][ii]["pair"]
                    est["mean_field"] = mean_field_from_declared(res, pair, B, m)
                    est["src"] = (f"artifacts/data/exact_m_fixed_budget/V3_bridge/fk_index/"
                                  f"{ens_name(m, eps)}.json (declared item "
                                  f"{res['declared'][ii]['label']}; tag {TAG})")
                    est["seed_entropy"] = ens.index["seed_entropy"]
                    rec["mass_sources"]["V3"] = est
                    rec["peak_sources"]["V3"] = est
                    dec = decomposition_from_declared(res, ii, m)
                    dec["src"] = est["src"]
                    rec["decomposition_source"] = dec
            # HPC direct kill (cross-check only)
            hk = f"m{m}_eps{eps:g}_B{B:g}_{a}"
            if hk in hpc:
                pw = hpc[hk]["pooled"]["basins_window"]
                rec["mass_sources"]["HPC_dk"] = {
                    "masses": pw["mass"], "se": pw["se_batch"], "N": 20000000, "dt": 1e-3,
                    "peak_times_smoothed_vertex": hpc[hk]["pooled"]["peaks_extended_basins"],
                    "src": f"{REL(HPC_JSON)}#cells.{hk}.pooled.basins_window"}
            # primary selection
            ms = rec["mass_sources"]
            prim_m = next((s for s in ("N1", "N9a", "V3") if s in ms), None)
            ps = rec["peak_sources"]
            prim_p = next((s for s in ("N9a", "V3") if s in ps), None)
            rec["mass_primary"] = prim_m
            rec["peak_primary"] = prim_p
            rec["contact_prob_at_t_j"] = contact_at_targets(m, eps)
            pts[f"{eps:g}"] = rec
        out[ck] = {"theory": th, "points": pts}
    return out


# ============================================================================
# derived quantities and log-log fits
# ============================================================================

RESOLVED_Z = 3.0
FIT_RANGES = {"eps_le_0.035": 0.035 + 1e-12, "eps_le_0.05": 0.05 + 1e-12, "all": 1.0}


def derive(data: dict) -> None:
    for ck, cell in data.items():
        th = cell["theory"]
        m = th["m"]
        for ek, rec in cell["points"].items():
            eps = rec["eps"]
            ms = rec["mass_sources"].get(rec["mass_primary"]) if rec["mass_primary"] else None
            if ms is not None:
                dtm = float(ms["dt"])
                corr = 0.5 * 1.0 * dtm * np.asarray(th["B_dMlim_dB"])     # gamma = 1
                M = np.asarray(ms["masses"], float) + corr
                se = np.asarray(ms["se"], float)
                sig = np.sqrt(se**2 + (MASS_DT_REL_SYS * corr) ** 2)
                dM = M - np.asarray(th["M_lim"])
                pred = eps * np.asarray(th["c1_first_order_mass"])
                pred2 = pred + eps * eps * np.asarray(th["c2_mean_field_mass"])
                d = {"dM": dM.tolist(), "abs_dM": np.abs(dM).tolist(), "sigma": sig.tolist(),
                     "z": (dM / sig).tolist(),
                     "resolved": (np.abs(dM) >= RESOLVED_Z * sig).tolist(),
                     "first_order_prediction": pred.tolist(),
                     "ratio_to_first_order": (dM / pred).tolist(),
                     "ratio_to_first_order_se": (sig / np.abs(pred)).tolist(),
                     "second_order_prediction": pred2.tolist(),
                     "ratio_to_second_order": (dM / pred2).tolist(),
                     "dt_correction_applied": corr.tolist(), "dt": dtm,
                     "masses_dt_corrected": M.tolist(),
                     "dM_over_eps": (dM / eps).tolist()}
                if ms.get("mean_field") is not None:
                    mf = np.asarray(ms["mean_field"], float) + corr
                    d["fluctuation_part_M_minus_Mmf"] = (M - mf).tolist()
                    d["mean_field_part_Mmf_minus_Mlim"] = (mf - np.asarray(th["M_lim"])).tolist()
                rec["mass_derived"] = d
            dec = rec.get("decomposition_source")
            if dec is not None:
                # M - M^lim = (M - M^mf) + (M^mf - M^lim): fluctuation part (Prop. th2:correction:
                # eps c1 + O((B^2 + B^3) eps^{3/2})) + mean-field part (Lemma th2:mean: O(B eps^2),
                # leading term eps^2 c2).  The Euler--Maruyama exposure correction is common to M
                # and M^mf, so it cancels in the fluctuation part and is applied to M^mf only.
                c1 = np.asarray(th["c1_first_order_mass"])
                c2 = np.asarray(th["c2_mean_field_mass"])
                corr = 0.5 * float(dec["dt"]) * np.asarray(th["B_dMlim_dB"])
                fl = np.asarray(dec["fluct"])
                sfl = np.asarray(dec["se_fluct"])
                rem = fl - eps * c1
                mfp = np.asarray(dec["M_mf"]) + corr - np.asarray(th["M_lim"])
                smf = np.sqrt(np.asarray(dec["se_M_mf"]) ** 2 + (MASS_DT_REL_SYS * corr) ** 2)
                rec["mass_decomposition"] = {
                    "source": "V3", "src": dec["src"],
                    "fluct_M_minus_Mmf": fl.tolist(), "se_fluct_joint_jackknife": sfl.tolist(),
                    "fluct_over_eps_c1": (fl / (eps * c1)).tolist(),
                    "fluct_over_eps_c1_se": (sfl / np.abs(eps * c1)).tolist(),
                    "remainder_fluct_minus_eps_c1": rem.tolist(),
                    "remainder_over_eps_1p5": (rem / eps**1.5).tolist(),
                    "remainder_over_eps_1p5_se": (sfl / eps**1.5).tolist(),
                    "remainder_z": (rem / sfl).tolist(),
                    "meanfield_part_Mmf_minus_Mlim": mfp.tolist(), "se_meanfield_part": smf.tolist(),
                    "meanfield_part_over_eps2_c2": (mfp / (eps * eps * c2)).tolist(),
                    "meanfield_part_over_eps2_c2_se": (smf / np.abs(eps * eps * c2)).tolist(),
                    "meanfield_part_z": (mfp / smf).tolist(),
                    "dt_correction_applied_to_Mmf": corr.tolist()}
            ps = rec["peak_sources"].get(rec["peak_primary"]) if rec["peak_primary"] else None
            if ps is not None:
                rows = []
                for j, pk in enumerate(ps["peaks"]):
                    tj, s = th["t_j"][j], th["s_star"][j]
                    if not pk.get("exists"):
                        rows.append({"exists": False})
                        continue
                    off = pk["t_peak"] - tj
                    pred = eps * s
                    r = off - pred
                    rows.append({"exists": True, "t_peak": pk["t_peak"], "se": pk["se_total"],
                                 "offset": off, "predicted_offset": pred,
                                 "ratio_offset_to_prediction": off / pred,
                                 "ratio_se": pk["se_total"] / abs(pred),
                                 "normalized_offset": off / eps, "s_star": s,
                                 "residual": r, "residual_over_eps": r / eps,
                                 "resolved": abs(r) >= RESOLVED_Z * pk["se_total"]})
                rec["peak_derived"] = rows


def _tq(dof: int) -> float:
    from scipy.stats import t as tdist
    return float(tdist.ppf(0.975, dof)) if dof > 0 else float("inf")


def wls(X: np.ndarray, Y: np.ndarray, sY: np.ndarray) -> dict:
    with np.errstate(all="ignore"):     # numpy 2 + Accelerate: spurious FP flags in matmul
        return _wls(X, Y, sY)


def _wls(X: np.ndarray, Y: np.ndarray, sY: np.ndarray) -> dict:
    W = 1.0 / sY
    Xw, Yw = X * W[:, None], Y * W
    beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
    res = Yw - Xw @ beta
    n, k = X.shape
    dof = n - k
    chi2 = float(res @ res)
    cov = np.linalg.inv(Xw.T @ Xw)
    scale = max(1.0, chi2 / dof) if dof > 0 else 1.0
    se = np.sqrt(np.diag(cov) * scale)
    return {"beta": beta, "se": se, "dof": dof, "chi2": chi2, "scale": scale, "n": n}


def loglog_fit(series: dict, eps_max: float, need: int = 2) -> dict:
    """Common-slope fit of log|y| on log eps with one intercept per series (pooled), and
    per-series fits (>= 3 points).  series: {name: [(eps, y, sigma), ...]} (resolved points)."""
    used, rows = {}, []
    for name, pts in series.items():
        pp = [p for p in pts if p[0] <= eps_max]
        if len(pp) < need:
            continue
        sg = {np.sign(p[1]) for p in pp}
        if len(sg) > 1:
            continue                      # sign change inside the range: no power law
        used[name] = pp
    out = {"eps_max": eps_max, "series_used": sorted(used), "per_series": {}}
    if not used:
        out["pooled"] = None
        return out
    names = sorted(used)
    X, Y, S = [], [], []
    for i, nm in enumerate(names):
        for e, y, s in used[nm]:
            row = np.zeros(1 + len(names))
            row[0] = math.log(e)
            row[1 + i] = 1.0
            X.append(row)
            Y.append(math.log(abs(y)))
            S.append(s / abs(y))
    f = wls(np.array(X), np.array(Y), np.array(S))
    tq = _tq(f["dof"])
    out["pooled"] = {"slope": float(f["beta"][0]), "se": float(f["se"][0]),
                     "ci95": [float(f["beta"][0] - tq * f["se"][0]),
                              float(f["beta"][0] + tq * f["se"][0])],
                     "dof": f["dof"], "chi2": f["chi2"], "se_inflation": math.sqrt(f["scale"]),
                     "n_points": f["n"], "n_series": len(names)}
    for nm in names:
        pp = used[nm]
        if len(pp) < 3:
            continue
        X1 = np.array([[math.log(e), 1.0] for e, _, _ in pp])
        Y1 = np.array([math.log(abs(y)) for _, y, _ in pp])
        S1 = np.array([s / abs(y) for _, y, s in pp])
        g = wls(X1, Y1, S1)
        tq1 = _tq(g["dof"])
        out["per_series"][nm] = {"slope": float(g["beta"][0]), "se": float(g["se"][0]),
                                 "ci95": [float(g["beta"][0] - tq1 * g["se"][0]),
                                          float(g["beta"][0] + tq1 * g["se"][0])],
                                 "n_points": len(pp), "eps": [p[0] for p in pp]}
    return out


def two_term_fits(data: dict, eps_max: float = 0.035 + 1e-12) -> dict:
    """dM_j = a_j eps + b_j eps^2 over eps <= eps_max (all points, resolved or not); a_j vs c1_j."""
    rows = {}
    for ck, cell in data.items():
        th = cell["theory"]
        for j in range(th["m"]):
            pts = [(rec["eps"], rec["mass_derived"]["dM"][j], rec["mass_derived"]["sigma"][j])
                   for rec in cell["points"].values()
                   if rec.get("mass_derived") and rec["eps"] <= eps_max]
            if len(pts) < 3:
                continue
            e = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            sg = np.array([p[2] for p in pts])
            f = wls(np.column_stack([e, e * e]), y, sg)
            c1 = th["c1_first_order_mass"][j]
            tq = _tq(f["dof"])
            rows[f"{ck}_j{j + 1}"] = {"a": float(f["beta"][0]), "a_se": float(f["se"][0]),
                                     "b": float(f["beta"][1]), "b_se": float(f["se"][1]),
                                     "c1": c1, "a_over_c1": float(f["beta"][0] / c1),
                                     "a_over_c1_ci95": sorted([float((f["beta"][0] - tq * f["se"][0]) / c1),
                                                               float((f["beta"][0] + tq * f["se"][0]) / c1)]),
                                     "dof": f["dof"], "eps": e.tolist()}
    a = np.array([r["a_over_c1"] for r in rows.values()]) if rows else np.array([np.nan])
    return {"model": "dM_j = a_j eps + b_j eps^2 (weighted LSQ, sigma = SE and dt bound in quadrature)",
            "eps_max": eps_max, "rows": rows,
            "a_over_c1_range": [float(np.nanmin(a)), float(np.nanmax(a))]}


def fits(data: dict) -> dict:
    mass_series, peak_series = {}, {}
    fluct_series, rem_series, mf_series = {}, {}, {}
    unresolved = {"mass": [], "peak": [], "fluct": [], "remainder": [], "meanfield": []}
    for ck, cell in data.items():
        m = cell["theory"]["m"]
        for j in range(m):
            ms, ps, fs, rs, mfs = [], [], [], [], []
            tag = f"{ck}_j{j + 1}"
            for ek, rec in cell["points"].items():
                e = rec["eps"]
                d = rec.get("mass_derived")
                if d:
                    if d["resolved"][j]:
                        ms.append((e, d["dM"][j], d["sigma"][j]))
                    else:
                        unresolved["mass"].append(f"{tag}@eps{ek}")
                dc = rec.get("mass_decomposition")
                if dc:
                    for key, se_key, lst, nm in (
                            ("fluct_M_minus_Mmf", "se_fluct_joint_jackknife", fs, "fluct"),
                            ("remainder_fluct_minus_eps_c1", "se_fluct_joint_jackknife", rs, "remainder"),
                            ("meanfield_part_Mmf_minus_Mlim", "se_meanfield_part", mfs, "meanfield")):
                        y, sy = dc[key][j], dc[se_key][j]
                        if abs(y) >= RESOLVED_Z * sy:
                            lst.append((e, y, sy))
                        else:
                            unresolved[nm].append(f"{tag}@eps{ek}")
                pr = rec.get("peak_derived")
                if pr and pr[j].get("exists"):
                    if pr[j]["resolved"]:
                        ps.append((e, pr[j]["residual"], pr[j]["se"]))
                    else:
                        unresolved["peak"].append(f"{tag}@eps{ek}")
            mass_series[tag] = sorted(ms)
            fluct_series[tag] = sorted(fs)
            rem_series[tag] = sorted(rs)
            mf_series[tag] = sorted(mfs)
            peak_series[tag] = sorted(ps)
    out = {"resolved_rule": (f"|value| >= {RESOLVED_Z} sigma (masses: batch SE and half the "
                             "Euler--Maruyama exposure correction in quadrature; decomposition: joint "
                             "jackknife SE over the 20 path groups; peaks: jackknife SE and the "
                             "time-step systematic in quadrature)"),
           "unresolved_points": unresolved, "mass": {}, "mass_fluctuation_part": {},
           "mass_fluctuation_remainder": {}, "mass_meanfield_part": {}, "peak_residual": {},
           "notes": {"mass": "log|M_j - M_j^lim| on log eps; common slope, one intercept per basin",
                     "mass_fluctuation_part": ("log|M_j - M_j^mf| (M^mf: mean-field law with the "
                                               "sampled mean exposure of the same paths, gate "
                                               "included; V3 ensembles); predicted: slope 1 with "
                                               "coefficient c1_j (SM-A Prop. th2:correction)"),
                     "mass_fluctuation_remainder": ("log|M_j - M_j^mf - eps c1_j|; SM-A Prop. "
                                                    "th2:correction bounds it by C(B^2 + B^3) "
                                                    "eps^{3/2} (an upper bound: slope >= 1.5 or a "
                                                    "bounded |r'|/eps^{3/2} is consistent)"),
                     "mass_meanfield_part": ("log|M_j^mf - M_j^lim|; SM-A Lemma th2:mean bounds it by "
                                             "C B eps^2, leading term eps^2 c2_j (Remark th2:shape(i))"),
                     "peak_residual": ("log|t_j^max - t_j - eps rho y_*/v_j| (time units); the "
                                       "theorems give o(eps), i.e. slope > 1, no rate"),
                     "sign_rule": "series whose sign changes inside the range are left out",
                     "dependence": ("the four designs of one (m, eps) share paths and the m = 2 and "
                                    "m = 3 ensembles at the same (eps, dt) share the seed (target times "
                                    "are not part of the seed entropy): pooled CIs treat the series as "
                                    "independent and are therefore optimistic; the WLS SEs are inflated "
                                    "by sqrt(chi2/dof) when chi2/dof > 1")}}
    for nm, emax in FIT_RANGES.items():
        out["mass"][nm] = loglog_fit(mass_series, emax)
        out["mass_fluctuation_part"][nm] = loglog_fit(fluct_series, emax)
        out["mass_fluctuation_remainder"][nm] = loglog_fit(rem_series, emax)
        out["mass_meanfield_part"][nm] = loglog_fit(mf_series, emax)
        out["peak_residual"][nm] = loglog_fit(peak_series, emax)
    out["mass_two_term"] = two_term_fits(data)
    out["remainder_scaling"] = remainder_scaling(data)
    return out


def remainder_scaling(data: dict) -> dict:
    """Per eps: max over basins of |fluct/(eps c1) - 1| and of |r'|/eps^{3/2}.  An O(eps^{3/2})
    remainder keeps max|r'|/eps^{3/2} bounded as eps decreases; a wrong first-order coefficient
    would make it grow like eps^{-1/2}."""
    by = {}
    for ck, cell in data.items():
        for ek, rec in cell["points"].items():
            dc = rec.get("mass_decomposition")
            if not dc:
                continue
            b = by.setdefault(ek, {"rel": [], "r15": [], "mfr": []})
            for j in range(cell["theory"]["m"]):
                b["rel"].append((abs(dc["fluct_over_eps_c1"][j] - 1.0), f"{ck}_j{j + 1}"))
                b["r15"].append((abs(dc["remainder_over_eps_1p5"][j]), f"{ck}_j{j + 1}"))
                if abs(dc["meanfield_part_z"][j]) >= RESOLVED_Z:
                    b["mfr"].append(dc["meanfield_part_over_eps2_c2"][j])
    out = {}
    for ek in sorted(by, key=float):
        b = by[ek]
        rel = max(b["rel"])
        r15 = max(b["r15"])
        out[ek] = {"max_abs_fluct_ratio_minus_1": rel[0], "argmax": rel[1],
                   "fluct_ratio_range": None,
                   "max_abs_remainder_over_eps_1p5": r15[0], "argmax_remainder": r15[1],
                   "meanfield_ratio_range_resolved": ([float(min(b["mfr"])), float(max(b["mfr"]))]
                                                      if b["mfr"] else None),
                   "n_meanfield_resolved": len(b["mfr"])}
    for ck, cell in data.items():
        for ek, rec in cell["points"].items():
            dc = rec.get("mass_decomposition")
            if not dc:
                continue
            r = out[ek]
            lo = min(dc["fluct_over_eps_c1"])
            hi = max(dc["fluct_over_eps_c1"])
            if r["fluct_ratio_range"] is None:
                r["fluct_ratio_range"] = [lo, hi]
            else:
                r["fluct_ratio_range"] = [min(r["fluct_ratio_range"][0], lo),
                                          max(r["fluct_ratio_range"][1], hi)]
    return out


# ============================================================================
# cross-checks between independent sources
# ============================================================================


def cross_checks(data: dict) -> dict:
    rows = []
    for ck, cell in data.items():
        for ek, rec in cell["points"].items():
            ms = rec["mass_sources"]
            prim = rec["mass_primary"]
            for other in ("V3", "N9a", "N1", "HPC_dk"):
                if prim is None or other == prim or other not in ms:
                    continue
                a = np.asarray(ms[other]["masses"], float)
                b = np.asarray(ms[prim]["masses"], float)
                sa = np.asarray(ms[other]["se"], float)
                sb = np.asarray(ms[prim]["se"], float)
                z = (a - b) / np.sqrt(sa**2 + sb**2)
                rows.append({"cell": ck, "eps": rec["eps"], "quantity": "basin masses",
                             "source": other, "vs": prim, "diff": (a - b).tolist(),
                             "z": z.tolist(), "max_abs_z": float(np.max(np.abs(z)))})
            ps = rec["peak_sources"]
            pp = rec["peak_primary"]
            for other in ("V3",):
                if pp is None or other == pp or other not in ps:
                    continue
                zz, dd = [], []
                for x, y in zip(ps[other]["peaks"], ps[pp]["peaks"]):
                    if x.get("exists") and y.get("exists"):
                        d = x["t_peak"] - y["t_peak"]
                        s = math.hypot(x["se_jackknife"], y["se_jackknife"])
                        dd.append(d)
                        zz.append(d / s)
                rows.append({"cell": ck, "eps": rec["eps"], "quantity": "peak times (dt-corrected)",
                             "source": other, "vs": pp, "diff": dd, "z": zz,
                             "max_abs_z": float(np.max(np.abs(zz))) if zz else None,
                             "note": "sampling SEs only; the two ensembles have different dt at "
                                     "eps = 0.035 (N9a 5e-4 = V3) and 0.025 (both 2.5e-4)"})
    by = {}
    for r in rows:
        k = f"{r['source']}_vs_{r['vs']}_{r['quantity'].split()[0]}"
        by.setdefault(k, []).append(r["max_abs_z"])
    summary = {k: {"n_cells": len(v), "max_abs_z": float(np.nanmax([x for x in v if x is not None]))}
               for k, v in by.items() if any(x is not None for x in v)}
    return {"rows": rows, "summary": summary}


# ============================================================================
# figure
# ============================================================================

MARK = {2: "o", 3: "s"}


GATE_UNSAT = 1e-3       # shading rule: 1 - c(t_m) > GATE_UNSAT at the last target time


def gate_boundary(data: dict) -> float:
    """Geometric midpoint between the largest eps with 1 - c(t_m) <= GATE_UNSAT in every cell and
    the next eps of the ladder (contact probabilities from fk.mean_contact_curve)."""
    worst = {}
    for cell in data.values():
        for ek, rec in cell["points"].items():
            u = 1.0 - min(rec["contact_prob_at_t_j"])
            worst[rec["eps"]] = max(worst.get(rec["eps"], 0.0), u)
    eps = sorted(worst)
    sat = [e for e in eps if worst[e] <= GATE_UNSAT]
    uns = [e for e in eps if worst[e] > GATE_UNSAT]
    if not sat or not uns or max(sat) > min(uns):
        raise AssertionError(f"gate saturation not monotone in eps: {worst}")
    return math.sqrt(max(sat) * min(uns))


def make_figure(data: dict, stem: Path) -> list:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1758844800")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import fb_make_paper_figures as pf
    matplotlib.rcParams.update(pf.PAPER_RC)
    col = {"equal": pf.C_EQUAL, "maxmin": pf.C_MAXMIN}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pf.TEXT_WIDTH_IN, 2.45),
                                   gridspec_kw={"wspace": 0.34})
    xlim = (0.0105, 0.118)
    gate_edge = gate_boundary(data)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.axvspan(gate_edge, xlim[1], color="0.93", zorder=0, lw=0)
        ax.set_xticks([0.0125, 0.025, 0.05, 0.1])
        ax.set_xticklabels(["0.0125", "0.025", "0.05", "0.1"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"noise $\varepsilon$")
    for ck, cell in data.items():
        th = cell["theory"]
        m, a, B = th["m"], th["allocation"], th["B"]
        c = col[a]
        face = c if B == 8.0 else "white"
        for j in range(m):
            xs, ys, xo, yo, xs2, ys2, es2, xf, yf, xfo, yfo = ([] for _ in range(11))
            for ek, rec in cell["points"].items():
                d = rec.get("mass_derived")
                if d:
                    val = abs(d["dM"][j]) / abs(th["c1_first_order_mass"][j])
                    same = np.sign(d["dM"][j]) == np.sign(th["c1_first_order_mass"][j])
                    (xs if same else xo).append(rec["eps"])
                    (ys if same else yo).append(val)
                dc = rec.get("mass_decomposition")
                if dc:
                    fj, cj = dc["fluct_M_minus_Mmf"][j], th["c1_first_order_mass"][j]
                    same_f = np.sign(fj) == np.sign(cj)
                    (xf if same_f else xfo).append(rec["eps"])
                    (yf if same_f else yfo).append(abs(fj) / abs(cj))
                pr = rec.get("peak_derived")
                if pr and pr[j].get("exists"):
                    xs2.append(rec["eps"])
                    ys2.append(pr[j]["ratio_offset_to_prediction"])
                    es2.append(pr[j]["ratio_se"])
            allx = sorted(zip(xs + xo, ys + yo))
            if allx:
                ax1.plot([p[0] for p in allx], [p[1] for p in allx], "-", color=c, lw=0.6,
                         alpha=0.55, zorder=2)
            ax1.plot(xs, ys, MARK[m], ms=3.6, mfc=face, mec=c, mew=0.8, ls="none", zorder=3)
            if xf:
                ax1.plot(xf, yf, ".", ms=2.2, color="0.1", ls="none", zorder=4)
            if xfo:
                ax1.plot(xfo, yfo, "x", ms=2.2, mew=0.6, color="0.1", ls="none", zorder=4)
            if xo:
                ax1.plot(xo, yo, "x", ms=3.6, color=c, mew=0.9, ls="none", zorder=3)
            if xs2:
                o = np.argsort(xs2)
                ax2.plot(np.asarray(xs2)[o], np.asarray(ys2)[o], "-", color=c, lw=0.6,
                         alpha=0.55, zorder=2)
                ax2.errorbar(xs2, ys2, yerr=2 * np.asarray(es2), fmt=MARK[m], ms=3.6, mfc=face,
                             mec=c, mew=0.8, ecolor=c, elinewidth=0.6, capsize=0, zorder=3)
    e = np.array(xlim)
    ax1.plot(e, e, "-", color="0.25", lw=0.9, zorder=1)
    ax1.set_yscale("log")
    ax1.legend(handles=[Line2D([], [], color="0.25", lw=0.9,
                               label=r"first order: $\varepsilon$"),
                        Line2D([], [], marker=".", ls="none", color="0.1", ms=4.0,
                               label=r"$|M_j-M_j^{\rm mf}|/|c_{1,j}|$"),
                        Line2D([], [], marker="x", ls="none", color="0.4", ms=3.6, mew=0.9,
                               label="opposite sign")],
               loc="upper left", frameon=False, fontsize=7.0, handlelength=1.4)
    ax1.text(gate_edge * 1.06, 0.97, "gate not\nsaturated", transform=ax1.get_xaxis_transform(),
             ha="left", va="top", fontsize=6.5, color="0.35")
    ax1.set_ylabel(r"$|M_j-M_j^{\rm lim}|\,/\,|c_{1,j}|$")
    ax1.set_title(r"(a) basin masses")
    ax2.axhline(1.0, color="0.25", lw=0.9, zorder=1)
    ax2.set_ylabel(r"$(t^{\max}_j-t_j)\,/\,(\varepsilon\rho\,y_*(\lambda_j)/v_j)$")
    ax2.set_title(r"(b) peak times")
    h = [Line2D([], [], color=col["equal"], lw=1.2, label="equal $w$"),
         Line2D([], [], marker="o", ls="none", mfc="0.4", mec="0.4", ms=3.6, label="$m=2$"),
         Line2D([], [], marker="o", ls="none", mfc="0.4", mec="0.4", ms=3.6,
                label="filled: $B=8$"),
         Line2D([], [], color=col["maxmin"], lw=1.2, label=r"max-min $w^*$"),
         Line2D([], [], marker="s", ls="none", mfc="0.4", mec="0.4", ms=3.6, label="$m=3$"),
         Line2D([], [], marker="o", ls="none", mfc="white", mec="0.4", ms=3.6,
                label="open: $B=2$")]
    ax2.legend(handles=h, loc="lower left", ncol=2, frameon=False, fontsize=7.0,
               handletextpad=0.3, columnspacing=0.8)
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.17, top=0.9)
    out = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suf in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suf), dpi=300,
                    metadata=({"CreationDate": None, "ModDate": None} if suf == ".pdf" else None))
        out.append(str(stem.with_suffix(suf)))
    plt.close(fig)
    return out


# ============================================================================
# analyze -> bridge.json
# ============================================================================


def _strip(est: dict) -> dict:
    return {k: v for k, v in est.items() if k not in ("seed_entropy",)}


def summaries(data: dict, ft: dict) -> dict:
    by_eps = {}
    for ck, cell in data.items():
        for ek, rec in cell["points"].items():
            b = by_eps.setdefault(ek, {"mass_ratio_to_first_order": [], "abs_dM": [],
                                       "mass_ratio_to_second_order": [],
                                       "fluct_ratio_to_first_order": [],
                                       "peak_ratio": [], "peak_ratio_first_passages": [],
                                       "peak_ratio_last_passages": [], "missing_peaks": [],
                                       "cells_with_mass": 0, "cells_with_peaks": 0})
            d = rec.get("mass_derived")
            if d:
                b["cells_with_mass"] += 1
                b["mass_ratio_to_first_order"] += d["ratio_to_first_order"]
                b["mass_ratio_to_second_order"] += d["ratio_to_second_order"]
                b["abs_dM"] += d["abs_dM"]
            dc = rec.get("mass_decomposition")
            if dc:
                b["fluct_ratio_to_first_order"] += dc["fluct_over_eps_c1"]
            pr = rec.get("peak_derived")
            if pr:
                b["cells_with_peaks"] += 1
                m = cell["theory"]["m"]
                for j, x in enumerate(pr):
                    if not x.get("exists"):
                        b["missing_peaks"].append(f"{ck}_j{j + 1}")
                        continue
                    b["peak_ratio"].append(x["ratio_offset_to_prediction"])
                    (b["peak_ratio_last_passages"] if j == m - 1 else
                     b["peak_ratio_first_passages"]).append(x["ratio_offset_to_prediction"])
    out = {}
    for ek, b in by_eps.items():
        def rng(v):
            return [float(np.min(v)), float(np.max(v))] if v else None
        out[ek] = {"cells_with_mass": b["cells_with_mass"], "cells_with_peaks": b["cells_with_peaks"],
                   "mass_ratio_to_first_order_range": rng(b["mass_ratio_to_first_order"]),
                   "fluctuation_part_ratio_range": rng(b["fluct_ratio_to_first_order"]),
                   "fluctuation_part_source": "V3 ensembles (mass_decomposition)",
                   "mass_ratio_to_second_order_range": rng(b["mass_ratio_to_second_order"]),
                   "max_abs_dM": float(np.max(b["abs_dM"])) if b["abs_dM"] else None,
                   "peak_ratio_range": rng(b["peak_ratio"]),
                   "peak_ratio_range_non_last": rng(b["peak_ratio_first_passages"]),
                   "peak_ratio_range_last": rng(b["peak_ratio_last_passages"]),
                   "missing_peaks": b["missing_peaks"]}
    return out


def analyze() -> dict:
    data = collect()
    derive(data)
    ft = fits(data)
    xc = cross_checks(data)
    sm = summaries(data, ft)
    new_runs = []
    for m, eps in RUN_ORDER:
        p = INDEX_DIR_V3 / f"{ens_name(m, eps)}.json"
        if p.exists():
            ix = json.loads(p.read_text())
            new_runs.append({"name": ix["name"], "m": m, "eps": eps, "dt": ix["spec"]["dt"],
                             "n_paths": ix["n_paths"], "tag": ix["tag"], "seed": ix["seed"],
                             "seed_entropy": ix["seed_entropy"], "complete": ix["complete"],
                             "batch_groups": ix["n_paths"] // ix["batch"],
                             "process_seconds_total": ix["process_seconds_total"],
                             "index": REL(p), "declared": ix["declared"]})
    used = {"mass": {}, "peak": {}}
    for ck, cell in data.items():
        for ek, rec in cell["points"].items():
            if rec["mass_primary"]:
                used["mass"].setdefault(rec["mass_primary"], []).append(f"{ck}@eps{ek}")
            if rec["peak_primary"]:
                used["peak"].setdefault(rec["peak_primary"], []).append(f"{ck}@eps{ek}")
    cells_out = {}
    for ck, cell in data.items():
        pts = {}
        for ek, rec in cell["points"].items():
            r = dict(rec)
            r["mass_sources"] = {k: _strip(v) for k, v in rec["mass_sources"].items()}
            r["peak_sources"] = {k: {"peaks": v["peaks"], "src": v["src"]}
                                 for k, v in rec["peak_sources"].items()}
            pts[ek] = r
        cells_out[ck] = {"theory": cell["theory"], "points": pts}
    out = {
        "item": "V3 theory-numerics bridge (distance of the simulated noise levels to the "
                "asymptotic regime of Theorems 2-3)",
        "driver": "code/fb_v3_bridge.py analyze",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sm_a_numbering": {"th2:mean": "Lemma A3.17", "th2:varasym": "Proposition A3.19",
                           "th2:main": "Theorem A3.20", "th2:correction": "Proposition A3.21",
                           "th2:shape": "Remark A3.22", "th2:scope": "Remark A3.23",
                           "th3:main": "Theorem A3.32", "th3:peaks": "Corollary A3.33",
                           "th3:rate": "Remark A3.34", "th6:thm-main": "Theorem A4.1",
                           "source": "manuscript/cnsns_submission/supplement_A_cnsns.aux"},
        "predicted_orders": {
            "masses": ("SM-A Theorem th2:main(a) (main Theorem 2(a)): |M_j - M_j^lim| <= C(B^2 eps + "
                       "B eps^2), i.e. order eps at fixed B (not eps^{1/2}); sharp (SM-A Remark "
                       "th2:shape(i)); leading coefficient c1_j explicit (SM-A Prop. th2:correction with "
                       "Lemma th2:mean); constants hide exp(-c/eps^2) contact-gate terms (SM-A Remark "
                       "th2:scope(i))"),
            "peak_times": ("SM-A Corollary th3:peaks / Theorem th6:thm-main (main Theorems 2(c), 3): "
                           "t_j^max = t_j + eps rho y_*(lambda_j)/v_j + o(eps); no rate is proved "
                           "(SM-A Remark th3:rate records an unproved C^0 profile rate eps^{1/2})")},
        "anchor": {"model": "production anchor (gamma = D0 = rho = W = 1, z0 = 4, zbar = 0, d = 2, "
                            "a = 0.4, r_par0 = 0.1, u0 = 0.3, Sigma_perp0 = 0.3^2)",
                   "targets": {"2": [1.0, 2.5], "3": [0.8, 1.6, 2.8]},
                   "cells": [cell_key(*c) for c in cells()],
                   "eps_ladder": list(EPS_LADDER),
                   "basin_cuts_window": {str(k): v for k, v in CUTS.items()},
                   "basin_convention": "theorem convention: s_0 = tau = 0.5, s_m = T = 3.5; interior "
                                       "cuts = N1 geometric midpoints on the 0.02 grid",
                   "peak_estimator": ("critical point of the exact step density p_n/dt at the stamps "
                                      "t_n (degree-4 LSQ fit on the contiguous region >= 0.75 x peak "
                                      "height, at least +-4 steps), jackknife over 20 path groups; "
                                      "+ gamma t dt/2 Euler--Maruyama mean-path correction (point "
                                      "evaluations carry no dt/2 stamp shift; the (dt/2)(1 + gamma t) "
                                      "of SM-B N5 belongs to binned laws; see dt_ladder), systematic "
                                      "0.2 dt in quadrature"),
                   "mass_sigma": ("batch-means SE and half the Euler--Maruyama exposure correction "
                                  "(gamma dt/2) B dM^lim/dB in quadrature; the correction is applied"),
                   "decomposition": ("M - M^lim = (M - M^mf) + (M^mf - M^lim) from the V3 ensembles at "
                                     "every eps (joint jackknife over the 20 path groups: M and M^mf "
                                     "share paths); M^mf from the sampled mean exposure at the cuts"),
                   "common_random_numbers": ("the four designs of one (m, eps) ensemble share paths; the "
                                             "m = 2 and m = 3 ensembles at the same (eps, dt) share the "
                                             "seed entropy (see new_runs[*].seed_entropy), i.e. common "
                                             "random numbers; different eps or dt are independent"),
                   "dt_by_eps_new_runs": {f"{k:g}": v for k, v in DT.items()}},
        "selection_rule": ("masses: N1 (published stored FK, eps 0.1/0.05) > N9a (declared FK, equal "
                           "weights B = 8, eps 0.035/0.025) > V3 new; peaks: N9a > V3 new (N1 and HPC "
                           "have no step-resolution densities; HPC peak times are smoothed vertices)"),
        "sources_used_as_primary": used,
        "sources_not_used": {
            "HPC_headline": "direct kill, eps 0.05/0.1: masses used as cross-check only; its peak "
                            "times are vertices of a 0.04-bandwidth smoothed 0.02-bin histogram",
            "N6 stored eps 0.075": "not on the eps ladder (0.07 run anew with step resolution)",
            "n3_tangent eps 0.0125": "boundary-tangent geometry (r_par0 = 0, r_perp0 = a), not the anchor",
            "V2_design horizon eps 0.01-0.1": "designed stripe times per eps (not fixed designs)",
            "TH3 fk_peaks": "gate-free scout kernel on a 1e-3 grid"},
        "new_runs": new_runs,
        "cells": cells_out,
        "fits": ft,
        "cross_checks": xc,
        "summary_by_eps": sm,
        "dt_ladder": dt_ladder(),
        "variance_check": variance_check(),
    }
    out["verdict"] = verdict(out)
    lad = out["dt_ladder"].get("summary")
    if lad:
        out["dt_ladder"]["systematic_used"] = {
            "peak_time_sigma_per_dt": PEAK_DT_SYS,
            "ladder_bound_per_dt": lad["peak_sys_bound_per_dt"],
            "ladder_bound_rule": lad["peak_sys_bound_rule"],
            "covered": lad["peak_sys_bound_per_dt"] <= PEAK_DT_SYS,
            "per_row_worst_case_per_dt": lad["max_abs_slope_minus_model_plus_2se"],
            "per_row_note": ("max over the rows of |slope - model| + 2 se: dominated by the noisy "
                             "slopes of the slow last passages (2e5 ladder paths); row z against the "
                             "point-evaluated model in [%.2f, %.2f], against the binned model in "
                             "[%.1f, %.1f]" % (*lad["peak_slope_z_vs_point_EM_model"],
                                               *lad["peak_slope_z_vs_binned_model"]))}
    return out


def _rng_eps(sm: dict, keys, kind: str):
    lo, hi = [], []
    for ek in keys:
        r = sm[ek][kind]
        if r:
            lo.append(r[0])
            hi.append(r[1])
    return [float(min(lo)), float(max(hi))] if lo else None


def verdict(out: dict) -> dict:
    """Plain consistency statements (numbers from this file only)."""
    sm = out["summary_by_eps"]
    ft = out["fits"]
    rs = ft["remainder_scaling"]
    small = [k for k in sm if float(k) <= 0.035]
    gate = [k for k in sm if float(k) >= 0.05]
    fl = ft["mass_fluctuation_part"]["eps_le_0.035"]["pooled"]
    rem = ft["mass_fluctuation_remainder"]["eps_le_0.035"]["pooled"]
    mf = ft["mass_meanfield_part"]["eps_le_0.035"]["pooled"]
    tot = ft["mass"]["eps_le_0.035"]["pooled"]
    pk = ft["peak_residual"]["eps_le_0.035"]["pooled"]
    pk5 = ft["peak_residual"]["eps_le_0.05"]["pooled"]
    r15 = {ek: rs[ek]["max_abs_remainder_over_eps_1p5"] for ek in small}
    rel = {ek: rs[ek]["max_abs_fluct_ratio_minus_1"] for ek in small}
    e_min = min(small, key=float)
    # criteria (fixed here, stated in the output)
    c_fl = abs(fl["slope"] - 1.0) <= 0.05 and max(rel.values()) <= 0.10
    c_rem = r15[e_min] <= 1.2 * max(v for k, v in r15.items() if k != e_min)
    c_mf = mf["ci95"][0] <= 2.0 <= mf["ci95"][1] or mf["ci95"][0] > 1.5
    c_pk = pk["ci95"][0] > 1.0
    n_opp, n_bas = 0, 0
    for cell in out["cells"].values():
        th = cell["theory"]
        for a, b in zip(th["c1_first_order_mass"], th["c2_mean_field_mass"]):
            n_bas += 1
            n_opp += int(a * b < 0)
    return {
        "criteria": {
            "masses_first_order": ("|pooled slope of |M - M^mf| - 1| <= 0.05 for eps <= 0.035 and the "
                                   "fluctuation part within 10% of eps c1 there"),
            "masses_remainder": ("max_j |M_j - M_j^mf - eps c1_j| / eps^{3/2} at the smallest eps not "
                                 "above 1.2 x its maximum at the larger eps <= 0.035 (an O(eps) "
                                 "remainder would grow like eps^{-1/2}: x1.67 from 0.035 to 0.0125)"),
            "masses_meanfield": "pooled slope of |M^mf - M^lim| (eps <= 0.035): CI contains 2 or lies above 1.5",
            "peaks_o_eps": "pooled slope of the peak-time residual (eps <= 0.035): CI lower end > 1"},
        "masses": {
            "theorem_order": ("|M_j - M_j^lim| <= C(B^2 eps + B eps^2) (main Thm 2(a) = SM-A Thm "
                              "th2:main(a)), order eps, sharp; leading term eps c1_j explicit (SM-A Prop. "
                              "th2:correction, remainder C(B^2 + B^3) eps^{3/2}); mean-field part "
                              "<= C B eps^2 (SM-A Lemma th2:mean). NOT eps^{1/2}."),
            "fluct_slope_eps_le_0.035": {k: fl[k] for k in ("slope", "ci95", "n_points", "n_series")},
            "fluct_ratio_range_eps_le_0.035": _rng_eps(sm, small, "fluctuation_part_ratio_range"),
            "fluct_max_rel_dev_by_eps": rel,
            "remainder_max_over_eps_1p5_by_eps": r15,
            "remainder_slope_eps_le_0.035": ({k: rem[k] for k in ("slope", "ci95", "n_points", "n_series")}
                                             if rem else None),
            "meanfield_slope_eps_le_0.035": {k: mf[k] for k in ("slope", "ci95", "n_points", "n_series")},
            "total_slope_eps_le_0.035": {k: tot[k] for k in ("slope", "ci95", "n_points", "n_series",
                                                              "se_inflation")},
            "total_ratio_to_first_order_eps_le_0.035": _rng_eps(sm, small, "mass_ratio_to_first_order_range"),
            "total_ratio_to_two_term_eps_le_0.035": _rng_eps(sm, small, "mass_ratio_to_second_order_range"),
            "total_max_abs_dM_eps_le_0.035": float(max(sm[k]["max_abs_dM"] for k in small)),
            "basins_with_c1_c2_opposite_sign": [n_opp, n_bas],
            "gate_regime_ratio_to_first_order": {k: sm[k]["mass_ratio_to_first_order_range"] for k in gate},
            "gate_regime_max_abs_dM": {k: sm[k]["max_abs_dM"] for k in gate},
            "consistent": bool(c_fl and c_rem and c_mf),
            "statement": (
                "Consistent with the theorem orders for eps <= 0.035: the exposure-fluctuation part "
                "M_j - M_j^mf is first order (pooled log-log slope %.3f, 95%% CI %.3f-%.3f) and equals "
                "eps c1_j to within %.0f%% (%.0f%% at eps = %s); its remainder stays below %.3f eps^{3/2} "
                "(slope %s); the mean-field part scales as eps^{%.2f} (CI %.2f-%.2f; theorem: B eps^2). "
                "The total deviation has local slope %.2f (CI %.2f-%.2f) because the explicit eps^2 "
                "mean-field term has the opposite sign to eps c1_j (in %d of the %d basins); the two "
                "explicit terms together give %.2f-%.2f of it. For eps >= 0.05 the contact gate is not "
                "saturated and the deviation reaches %.1f times the first-order term (%.3f in mass at "
                "eps = 0.1): the masses are not in the asymptotic regime there." % (
                    fl["slope"], fl["ci95"][0], fl["ci95"][1],
                    100 * max(rel.values()), 100 * rel[e_min], e_min, max(r15.values()),
                    ("%.2f, CI %.2f-%.2f" % (rem["slope"], rem["ci95"][0], rem["ci95"][1])) if rem else "n/a",
                    mf["slope"], mf["ci95"][0], mf["ci95"][1],
                    tot["slope"], tot["ci95"][0], tot["ci95"][1], n_opp, n_bas,
                    *_rng_eps(sm, small, "mass_ratio_to_second_order_range"),
                    max(sm[k]["mass_ratio_to_first_order_range"][1] for k in gate),
                    sm["0.1"]["max_abs_dM"]))},
        "peaks": {
            "theorem_order": ("t_j^max = t_j + eps rho y_*(lambda_j)/v_j + o(eps) (main Thms 2(c), 3 = "
                              "SM-A Cor. th3:peaks, Thm th6:thm-main); no rate proved"),
            "residual_slope_eps_le_0.035": {k: pk[k] for k in ("slope", "ci95", "n_points", "n_series")},
            "residual_slope_eps_le_0.05": {k: pk5[k] for k in ("slope", "ci95", "n_points", "n_series")},
            "ratio_range_by_eps": {k: sm[k]["peak_ratio_range"] for k in sorted(sm, key=float)},
            "ratio_range_non_last_by_eps": {k: sm[k]["peak_ratio_range_non_last"] for k in sorted(sm, key=float)},
            "ratio_range_last_by_eps": {k: sm[k]["peak_ratio_range_last"] for k in sorted(sm, key=float)},
            "n_unresolved_residuals": len(ft["unresolved_points"]["peak"]),
            "consistent": bool(c_pk),
            "statement": (
                "Consistent with o(eps): where the residual t_j^max - t_j - eps rho y_*/v_j is resolved "
                "(mostly the slow last passages) it falls as eps^{%.2f} (95%% CI %.2f-%.2f, eps <= "
                "0.035), and elsewhere it is within 3 SE of zero. The predicted shift is reached to "
                "%.2f-%.2f at eps = 0.0125 and to %.2f-%.2f for the non-last passages already at eps "
                "= 0.1; the last passages reach only %.2f-%.2f of it at eps = 0.1." % (
                    pk["slope"], pk["ci95"][0], pk["ci95"][1],
                    *sm["0.0125"]["peak_ratio_range"], *sm["0.1"]["peak_ratio_range_non_last"],
                    *sm["0.1"]["peak_ratio_range_last"]))},
    }


# ============================================================================
# snippet (main-text paragraph + caption; NOT inserted into the manuscript)
# ============================================================================


def write_snippet(res: dict, path: Path) -> str:
    """Draft main-text paragraph (<= 8 printed lines) + figure caption, every number with a
    % src comment into bridge.json.  NOT inserted into the manuscript."""
    J = "artifacts/data/exact_m_fixed_budget/V3_bridge/bridge.json"
    v = res["verdict"]
    vm, vp = v["masses"], v["peaks"]
    sm = res["summary_by_eps"]
    ft = res["fits"]
    fl = ft["mass_fluctuation_part"]["eps_le_0.035"]["pooled"]
    mf = ft["mass_meanfield_part"]["eps_le_0.035"]["pooled"]
    pk = ft["peak_residual"]["eps_le_0.035"]["pooled"]
    tot = ft["mass"]["eps_le_0.035"]["pooled"]
    rel = 100 * max(vm["fluct_max_rel_dev_by_eps"].values())
    gmax = max(r[1] for r in vm["gate_regime_ratio_to_first_order"].values())
    p0125 = sm["0.0125"]["peak_ratio_range"]
    p10l = sm["0.1"]["peak_ratio_range_last"]
    p10n = sm["0.1"]["peak_ratio_range_non_last"]
    npaths = sorted({r["n_paths"] for r in res["new_runs"]})
    c2 = res["cells"]["m2_equal_B2"]["points"]["0.05"]["contact_prob_at_t_j"][-1]
    c3 = res["cells"]["m3_equal_B2"]["points"]["0.05"]["contact_prob_at_t_j"][-1]
    c2b = res["cells"]["m2_equal_B2"]["points"]["0.035"]["contact_prob_at_t_j"][-1]
    c3b = res["cells"]["m3_equal_B2"]["points"]["0.035"]["contact_prob_at_t_j"][-1]

    def f2(x):
        return f"{x:.2f}"

    def ci(fit, nd=2):
        return f"${fit['slope']:.{nd}f}$ (${fit['ci95'][0]:.{nd}f}$--${fit['ci95'][1]:.{nd}f}$)"

    lines = [
        "% ---------------------------------------------------------------------------",
        "% V3 theory--numerics bridge: draft paragraph + figure (NOT inserted in the manuscript).",
        "% Suggested place: end of Sec. 4.1 (after 'Error control'), or as the last paragraph of",
        "% Sec. 3.2 after the Caveat of Theorem 3. Figure file: artifacts/figures/fb_v3_bridge.pdf",
        "% (found through \\graphicspath{../../artifacts/figures/} of main_cnsns.tex; copy it to figures/",
        "% for the submission bundle). Macros: \\Mlim, \\Mmf, \\smref (macros.tex); new label fig:bridge.",
        "% Generated by code/fb_v3_bridge.py snippet from " + J + ".",
        "% ---------------------------------------------------------------------------",
        "\\emph{Distance to the limits.} As $\\varepsilon_0$ is not computed, Fig.~\\ref{fig:bridge}",
        "follows eight anchor designs from $\\varepsilon=0.1$ to $0.0125$. For",
        f"$\\varepsilon\\le0.035$, $M_j-\\Mmf{{j}}$ is within ${rel:.0f}\\%$ of the explicit first-order term",
        f"% src: {J}#verdict.masses.fluct_max_rel_dev_by_eps (max over eps <= 0.035 and 20 basins: {rel:.2f}%); #cells.*.theory.c1_first_order_mass",
        "$\\varepsilon c_{1,j}$ (SM-A Proposition~\\smref{th2:correction}) and $\\Mmf{j}-\\Mlim{j}$ is",
        "of order $\\varepsilon^2$, as in Theorem~\\ref{thm:fixedB}(a); for $\\varepsilon\\ge0.05$",
        f"(contact gate unsaturated) $|M_j-\\Mlim{{j}}|$ reaches ${gmax:.0f}\\,\\varepsilon|c_{{1,j}}|$.",
        f"% src: {J}#fits.mass_meanfield_part.eps_le_0.035.pooled.{{slope,ci95}} ({mf['slope']:.3f}, {mf['ci95'][0]:.3f}-{mf['ci95'][1]:.3f}); #verdict.masses.gate_regime_ratio_to_first_order (max {gmax:.2f} at eps 0.1); gate: #cells.m{{2,3}}_equal_B2.points[0.05].contact_prob_at_t_j[-1] ({c2:.4f}, {c3:.4f}) vs [0.035] ({c2b:.4f}, {c3b:.4f})",
        f"Peak shifts reach ${f2(p0125[0])}$--${f2(p0125[1])}$ of $\\varepsilon\\rho\\,y_*(\\lambda_j)/v_j$ at",
        f"$\\varepsilon=0.0125$, the remainder falling as $\\varepsilon^{{{pk['slope']:.2f}}}$, i.e.\\ $o(\\varepsilon)$",
        f"% src: {J}#summary_by_eps[0.0125].peak_ratio_range; #fits.peak_residual.eps_le_0.035.pooled.slope ({pk['slope']:.3f}, CI {pk['ci95'][0]:.3f}-{pk['ci95'][1]:.3f}, {pk['n_points']} resolved points)",
        f"(Theorem~\\ref{{thm:exactcount}}); at $\\varepsilon=0.1$ only the slow last passages lag",
        f"(${f2(p10l[0])}$--${f2(p10l[1])}$). The masses thus follow the two-term expansion from",
        f"% src: {J}#summary_by_eps[0.1].peak_ratio_range_last; non-last passages at eps 0.1: #summary_by_eps[0.1].peak_ratio_range_non_last ({f2(p10n[0])}-{f2(p10n[1])}: 'within 10%')",
        "$\\varepsilon\\approx0.035$ on, the other peaks to within $10\\%$ at $\\varepsilon=0.1$.",
        "",
        "\\begin{figure}[tbp]",
        "  \\centering",
        "  \\includegraphics[width=\\textwidth]{fb_v3_bridge}",
        "  \\caption{Approach to the limits of Theorems~\\ref{thm:fixedB} and~\\ref{thm:exactcount}",
        "    at the eight anchor designs: $m=2$ (circles) and $3$ (squares), equal (orange) and",
        "    max--min (blue) weights, $B=8$ (filled) and $2$ (open); exact laws of the",
        "    time-discretised process ($\\Delta t=10^{-3}$ for $\\varepsilon\\ge0.05$, $5\\times10^{-4}$ at",
        "    $0.035$, $2.5\\times10^{-4}$ below), $5\\times10^5$--$10^6$ paths per point, window basins.",
        "    (a)~$|M_j-\\Mlim{j}|$ over the first-order coefficient",
        "    $|c_{1,j}|$ of SM-A Proposition~\\smref{th2:correction}; line: the first-order law",
        "    $\\varepsilon$; dots: the fluctuation part $|M_j-\\Mmf{j}|/|c_{1,j}|$; crosses: sign",
        "    opposite to $c_{1,j}$. For $\\varepsilon\\le0.035$ the coloured points lie below the line",
        "    because $\\Mmf{j}-\\Mlim{j}$, of order $\\varepsilon^2$, has the opposite sign.",
        "    (b)~Peak shift over its predicted leading term (bars: $\\pm2$ standard errors; time-step",
        "    correction included). Pooled log--log slopes, $\\varepsilon\\le0.035$ (95\\% CI):",
        f"    $M_j-\\Mmf{{j}}$ {ci(fl, 3)}, $\\Mmf{{j}}-\\Mlim{{j}}$ {ci(mf)}, peak-time remainder",
        f"    {ci(pk)}. Shaded: contact gate not saturated ($1-c(t_m)>10^{{-3}}$).}}",
        f"  % src: {J}#fits.mass_fluctuation_part.eps_le_0.035.pooled.{{slope,ci95}} ({fl['n_points']} points, {fl['n_series']} basins); #fits.mass_meanfield_part.eps_le_0.035.pooled ({mf['n_points']} resolved points); #fits.peak_residual.eps_le_0.035.pooled ({pk['n_points']} resolved points)",
        f"  % src: opposite sign: {J}#verdict.masses.basins_with_c1_c2_opposite_sign ({vm['basins_with_c1_c2_opposite_sign'][0]}/{vm['basins_with_c1_c2_opposite_sign'][1]}); local slope of |M_j - M_j^lim| at eps <= 0.035: #fits.mass.eps_le_0.035.pooled ({tot['slope']:.2f}, CI {tot['ci95'][0]:.2f}-{tot['ci95'][1]:.2f}); two explicit terms give #verdict.masses.total_ratio_to_two_term_eps_le_0.035 ({f2(vm['total_ratio_to_two_term_eps_le_0.035'][0])}-{f2(vm['total_ratio_to_two_term_eps_le_0.035'][1])})",
        f"  % src: data: {J}#sources_used_as_primary (masses: N1 at eps 0.1, 0.05 (5e5 paths); N9a equal B = 8 at 0.035, 0.025 (5e5); new runs otherwise ({len(res['new_runs'])} x {npaths[-1]:.0e} paths, #new_runs)); peaks: N9a (equal B = 8, eps 0.05-0.025), new runs otherwise; dots: #cells.*.points.*.mass_decomposition (new runs)",
        f"  % src: time step: {J}#anchor.dt_by_eps_new_runs (1e-3 for eps >= 0.05, 5e-4 at 0.035, 2.5e-4 below); peak correction + gamma t dt/2 validated by #dt_ladder.summary.pooled_kappa ({res['dt_ladder']['summary']['pooled_kappa']['kappa']:.3f} +- {res['dt_ladder']['summary']['pooled_kappa']['se']:.3f}); shading rule: code/fb_v3_bridge.py gate_boundary (GATE_UNSAT = 1e-3)",
        "  % figure: code/fb_v3_bridge.py figure -> artifacts/figures/fb_v3_bridge.pdf",
        "  \\label{fig:bridge}",
        "\\end{figure}",
    ]
    txt = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
    return txt


# ============================================================================
# time-step ladder at the smallest noise (validates the peak-time correction)
# ============================================================================

DTCHECK_EPS = 0.0125
DTCHECK_DTS = (1e-3, 5e-4)          # + the production 2.5e-4 ensemble
DTCHECK_PATHS = 200_000


def dtcheck_name(m: int, dt: float) -> str:
    return f"v3b_dt{dt:g}_m{m}_eps{DTCHECK_EPS:g}"


def cmd_dtcheck(args) -> None:
    _use_v3_index()
    for m in M_LIST:
        for dt in DTCHECK_DTS:
            spec = fk.EnsembleSpec(m=m, eps=DTCHECK_EPS, dt=dt)
            name = dtcheck_name(m, dt)
            t0 = time.time()
            print(f"[v3] dtcheck {name}", flush=True)
            ens = fk.simulate_ensemble(spec, int(args.paths), name=name, tag=TAG, seed=SEED,
                                       mode="declared", declared=declared_items(spec),
                                       workers=args.workers, chunk=CHUNK, batch=BATCH,
                                       note="V3 bridge dt ladder (fb_v3_bridge.py dtcheck)")
            print(f"[v3] {name} complete={ens.index['complete']} wall={time.time() - t0:.0f}s",
                  flush=True)


def dt_ladder() -> dict:
    rows = []
    for m in M_LIST:
        runs = {}
        for dt in DTCHECK_DTS:
            ens, res = _load_declared(dtcheck_name(m, dt), True)
            if res is not None:
                runs[dt] = res
        ens, res = _load_declared(ens_name(m, DTCHECK_EPS), True)
        if res is not None:
            runs[DT[DTCHECK_EPS]] = res
        if len(runs) < 2:
            continue
        for (mm, a, B) in cells():
            if mm != m:
                continue
            thc = theory_cell(m, a, B)
            w = thc["w"]
            per_dt = {}
            for dt, res in runs.items():
                ii = _item_index(res, B, w)
                per_dt[dt] = declared_estimates(res, ii, m, DTCHECK_EPS)
            dts = sorted(per_dt)
            for j in range(m):
                pk = [per_dt[d]["peaks"][j] for d in dts]
                if not all(p.get("exists") for p in pk):
                    continue
                tr = np.array([p["t_peak_raw"] for p in pk])
                se = np.array([max(p["se_jackknife"], 1e-7) for p in pk])
                X = np.column_stack([np.ones(len(dts)), dts])
                f = wls(X, tr, se)
                c, sc = float(f["beta"][1]), float(f["se"][1])
                tp = float(f["beta"][0])
                Ms = np.array([per_dt[d]["masses"][j] for d in dts])
                Ss = np.array([per_dt[d]["se"][j] for d in dts])
                g = wls(X, Ms, Ss)
                rows.append({"cell": cell_key(m, a, B), "j": j + 1, "dt": dts,
                             "t_peak_raw": tr.tolist(), "se_jackknife": se.tolist(),
                             "slope_dt": c, "slope_se": sc, "t_extrapolated": tp,
                             "model_point_EM": -0.5 * tp, "model_binned_SMB": -0.5 * (1 + tp),
                             "slope_over_model_point_EM": c / (-0.5 * tp),
                             "mass_slope_dt": float(g["beta"][1]),
                             "mass_slope_se": float(g["se"][1]),
                             "mass_slope_model_EM_exposure": -0.5 * thc["B_dMlim_dB"][j],
                             "mass_z_vs_model": float((g["beta"][1] + 0.5 * thc["B_dMlim_dB"][j])
                                                      / g["se"][1])})
    if not rows:
        return {"rows": []}
    ratio = np.array([r["slope_over_model_point_EM"] for r in rows])
    dev = np.array([abs(r["slope_dt"] - r["model_point_EM"]) + 2 * r["slope_se"] for r in rows])
    msl = np.array([abs(r["mass_slope_dt"]) + 2 * r["mass_slope_se"] for r in rows])
    mz = np.array([r["mass_z_vs_model"] for r in rows])
    pz_em = np.array([(r["slope_dt"] - r["model_point_EM"]) / r["slope_se"] for r in rows])
    pz_bin = np.array([(r["slope_dt"] - r["model_binned_SMB"]) / r["slope_se"] for r in rows])
    # pooled scale factor: slope_dt = kappa * (-gamma t / 2)  (kappa = 1: point-evaluated EM model;
    # the binned SM-B model corresponds to kappa = (1 + gamma t) / (gamma t) >= 1.36 here)
    xm = np.array([r["model_point_EM"] for r in rows])
    ys = np.array([r["slope_dt"] for r in rows])
    ws = 1.0 / np.array([r["slope_se"] for r in rows]) ** 2
    kap = float(np.sum(ws * xm * ys) / np.sum(ws * xm * xm))
    kap_se = float(1.0 / math.sqrt(np.sum(ws * xm * xm)))
    kap_chi2 = float(np.sum(ws * (ys - kap * xm) ** 2))
    tmax = float(max(r["t_extrapolated"] for r in rows))
    sys_bound = (abs(kap - 1.0) + 2.0 * kap_se) * 0.5 * tmax
    return {"eps": DTCHECK_EPS, "dts": sorted(set(sum([r["dt"] for r in rows], []))),
            "paths": {"ladder": DTCHECK_PATHS, "production": N_PATHS},
            "question": ("do step-level (point-evaluated) peak times carry the SM-B binned-law shift "
                         "(dt/2)(1+gamma t) or only the Euler--Maruyama mean-path advance gamma t dt/2?"),
            "rows": rows,
            "summary": {"n_peaks": len(rows),
                        "slope_over_gamma_t_half_range": [float(ratio.min()), float(ratio.max())],
                        "max_abs_slope_minus_model_plus_2se": float(dev.max()),
                        "max_abs_mass_slope_plus_2se": float(msl.max()),
                        "peak_slope_z_vs_point_EM_model": [float(pz_em.min()), float(pz_em.max())],
                        "peak_slope_z_vs_binned_model": [float(pz_bin.min()), float(pz_bin.max())],
                        "mass_slope_z_vs_EM_exposure_model": [float(mz.min()), float(mz.max())],
                        "pooled_kappa": {"kappa": kap, "se": kap_se, "chi2": kap_chi2,
                                         "dof": len(rows) - 1,
                                         "model": "slope_dt = kappa * (-gamma t_peak / 2)",
                                         "binned_model_kappa_min": float(min(
                                             (1 + r["t_extrapolated"]) / r["t_extrapolated"]
                                             for r in rows))},
                        "peak_sys_bound_per_dt": sys_bound,
                        "peak_sys_bound_rule": ("(|kappa - 1| + 2 se) * gamma t_max / 2: the largest "
                                                "residual bias of the corrected peak times, in units "
                                                "of dt"),
                        "note": ("20 rows share paths within each (m, dt) ensemble (4 designs), so "
                                 "the z are not independent")}}


# ============================================================================
# ingredient check: exposure variance at the cuts (SM-A Prop. th2:varasym)
# ============================================================================


def variance_check() -> dict:
    """Var X_{s_j} / eps from the sampled exposure moments of the new ensembles against the
    leading term D0/(2 sqrt(pi) rho) sum_{i<=j} k_i^2/v_i^3 (gate included in the samples)."""
    rows = []
    for m, eps in RUN_ORDER:
        ens, res = _load_declared(ens_name(m, eps), True)
        if res is None:
            continue
        t = np.asarray(res["t"], float)
        for ip, pr in enumerate(res["pairs"]):
            w = np.asarray(pr["w"], float)
            spec = fk.EnsembleSpec(m=m, eps=eps)
            v = spec.mu_prime_abs(np.asarray(spec.times()))
            k = w / spec.torus_w ** spec.n_perp
            pref = spec.d0 / (2.0 * math.sqrt(math.pi) * spec.rho)
            for jc, c in enumerate(CUTS[m][1:], start=1):
                i = int(np.searchsorted(t, c - 1e-12, side="left")) - 1
                pred = pref * float(np.sum(k[:jc] ** 2 / v[:jc] ** 3))
                vx = float(res["var_X"][ip][i])
                sx = float(res["var_X_se_batch"][ip][i])
                rows.append({"m": m, "eps": eps, "w": w.tolist(), "cut": c,
                             "var_X_over_eps": vx / eps, "se": sx / eps,
                             "leading_term": pred, "ratio": vx / eps / pred})
    return {"statement": "SM-A Prop. th2:varasym: Var X_{s_j} = eps D0/(2 sqrt(pi) rho) "
                         "sum_{i<=j} k_i^2/v_i^3 + O(eps^{3/2})", "rows": rows}


# ============================================================================
# CLI
# ============================================================================


def cmd_analyze(args) -> None:
    res = analyze()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    os.replace(tmp, OUT_JSON)
    print(f"[v3] wrote {OUT_JSON}")


def _load_json() -> dict:
    return json.loads(OUT_JSON.read_text())


def cmd_figure(args) -> None:
    data = _load_json()["cells"]
    for p in make_figure(data, FIG_STEM):
        print(f"[v3] wrote {p}")


def cmd_snippet(args) -> None:
    write_snippet(_load_json(), Path(args.out))
    print(f"[v3] wrote {args.out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--only", default="")
    s.add_argument("--paths", type=float, default=N_PATHS)
    s.add_argument("--workers", type=int, default=fk.MAX_WORKERS)
    s.set_defaults(func=cmd_simulate)
    s = sub.add_parser("dtcheck")
    s.add_argument("--paths", type=float, default=DTCHECK_PATHS)
    s.add_argument("--workers", type=int, default=fk.MAX_WORKERS)
    s.set_defaults(func=cmd_dtcheck)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("figure").set_defaults(func=cmd_figure)
    s = sub.add_parser("snippet")
    s.add_argument("--out", default=str(SNIPPET))
    s.set_defaults(func=cmd_snippet)
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
