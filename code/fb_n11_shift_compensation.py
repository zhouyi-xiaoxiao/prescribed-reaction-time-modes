#!/usr/bin/env python3
"""N11 -- shift-compensated timing design (answers referee Opus F3).

Question.  Theorem 2(c) puts the j-th reaction-time peak at t_j + eps*rho*y_*(lambda_j)/v_j
+ o(eps): early, and by a lot for a heavily loaded slow passage (the Box 2 max-min designs
peak at 2.09 against 2.5 at (m, eps, B) = (2, 0.1, 8)).  Can the design be inverted so that
the PEAK TIMES land on the targets T_j while the basin masses stay on the equal-mass (max-min)
target at the same fixed budget B?

Design variables: stripe positions c_j = mu(t'_j) (nominal passage times t'_j) and the
allocation w (sum w_j = 1, budget B fixed).  Targets: peak times T_j = the paper's target
times (1.0, 2.5) / (0.8, 1.6, 2.8); masses: all m whole-axis basin masses equal (max-min).
Basins: s_0 = 0, s_m = tmax = 4 and, between stripes, the valleys of the design's own free
exposure clock G (TH-2 cuts strictly between passages); the mass design and the evaluation use
the same cuts (referee F24).  A design-independent convention (cuts at the time midpoints of
consecutive targets) is reported as a secondary check.

Design loop (deterministic, no random numbers):
  masses : finite-eps mean-field max-min design on the exact EM-chain free clock
           (fb_n1_allocation_design.mf_design on the cuts above), given the stripes;
  shifts : predictor A = Theorem 2(c) leading order, peak_j = t'_j + eps rho y_*(lambda_j)/v(t'_j);
           predictor B = argmax of the mean-field density f_1 = B G e^{-B Lambda} (exact
           EM-chain clock, EM contact probability) inside basin j;
  update : t' <- t' + (T - peak) (first iteration, unit gain), then per-component secant
           steps; iterate to 1e-6 (history of iterations 1..3 recorded).
Designs tested per cell: box2 (stripes at T, limit-law max-min weights = the paper's Box 2),
mf_nominal (stripes at T, mean-field max-min weights), compA, compB (compensated), and
nwt1, nwt2 (one and two Newton steps from compB with the model Jacobian -- predictor B peaks and mean-field
mass imbalances w.r.t. (t', w) -- and the MEASURED Feynman--Kac residual; stripes that would leave
the window are pinned at t' = 3.5 and their peak equations dropped); compB_ext / nwt*_ext: the
same with stripes allowed beyond the window (t' <= 6) where the in-window target is unattainable
(nwt1_ext, nwt2_ext).

Test: exact discrete-time law by the Feynman--Kac estimator in declared mode (per-step exact
kill probabilities, exact_m_prr_fk_exact_law, 2.1e5 paths in 3 chunks, tag 90; all designs at one eps share
paths), and direct-kill confirmations with 1e6 walkers (tag 91) of the final compensated design
at (2, 0.1, 8) (the referee's anchor) and (3, 0.05, 8).
Peak time (measured): per basin, vertex of the least-squares parabola through the raw per-step
density on the region around the maximum where the density smoothed over 5 steps is >= 90 % of
its maximum (fit_peak; a smooth functional once the window is fixed, so the delete-one-batch
jackknife over 42 batches of 5000 paths, window held fixed, is valid).  Its bias relative to the
exact argmax, measured on the noiseless mean-field curves of these designs, is <= 0.002.  The
argmax of the 5-step-smoothed density is reported as a secondary value; it is unreliable on flat
late peaks (first-pass Newton designs compFK/compFK2 used it and are kept as superseded records).
Model peaks (predictors A, B) are exact argmaxes of noiseless curves.

Subcommands: design | fk | fkcorr [--second] | dk [--design nwt2] | analyze | figure
Seeds: base 20260923; tag 90 (FK), tag 91 (direct kill).
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
import exact_m_prr_upgrade_core as core  # noqa: E402
import fb_allocation_law as alloc  # noqa: E402
import fb_n1_allocation_design as n1  # noqa: E402

OUT_DIR = fk.FB_DATA / "N11_shift_compensation"
FK_INDEX_DIR = OUT_DIR / "fk_index"
DESIGN_JSON = OUT_DIR / "n11_designs.json"
OUT_JSON = OUT_DIR / "n11_shift_compensation.json"
DK_DIR = OUT_DIR / "directkill"
FIG = fk.FIGURES / "fb_n11_shift_compensation.pdf"
SEED = fk.BASE_SEED
TAG_FK = 90
TAG_DK = 91
N_PATHS = 210_000
FK_CHUNK = 70_000               # 3 chunks = one round of 3 workers
DK_WALKERS = 1_000_000
CELLS = [(2, 0.1, 8.0), (3, 0.1, 8.0), (3, 0.05, 8.0), (2, 0.05, 4.0), (3, 0.1, 4.0)]
DK_CELLS = [(2, 0.1, 8.0), (3, 0.05, 8.0)]   # referee anchor + an m = 3 cell where the target is attainable
TARGETS = {m: tuple(float(x) for x in fk.TARGET_TIMES[m]) for m in (2, 3)}
P = fk.MODEL
GAM, Z0, ZB, RHO = P.gamma, P.z0, P.z_bar, P.rho
AREA = P.torus_w ** (P.dim - 1)
DT = fk.base.DEFAULT_DT
TMAX = fk.base.DEFAULT_TMAX
SMOOTH_STEPS = 5.0

# fk writes ensemble indexes into its module-level INDEX_DIR: redirect them to this item.
fk.INDEX_DIR = FK_INDEX_DIR


def cell_key(m, eps, B) -> str:
    return f"m{m}_eps{eps:g}_B{B:g}"


def mu(t):
    return ZB + (Z0 - ZB) * np.exp(-GAM * np.asarray(t, float))


def t_of_centre(c):
    return -np.log((np.asarray(c, float) - ZB) / (Z0 - ZB)) / GAM


def speed(t):
    return GAM * abs(Z0 - ZB) * np.exp(-GAM * np.asarray(t, float))




# ---------------------------------------------------------------------------
# Theorem 2(c): passage-profile peak y_*(lambda)  (Lemma th3:profile: unique critical point)
# ---------------------------------------------------------------------------
_DU = 0.002
_U = np.arange(-16.0, 16.0 + _DU / 2, _DU)
_PHI = np.exp(-0.5 * _U * _U) / math.sqrt(2 * math.pi)
_CDF = 0.5 * (1.0 + np.vectorize(math.erf)(_U / math.sqrt(2.0)))
THETA = math.sqrt(P.d0 / (2 * GAM)) / RHO          # vartheta = s_Z / rho
_YCACHE: dict = {}


def ystar(lam: float) -> float:
    """argmax_y of F_lam = p_lam * phi_theta (bisection on the unique zero of F')."""
    key = round(float(lam), 12)
    if key in _YCACHE:
        return _YCACHE[key]
    hp = _PHI * np.exp(-lam * _CDF)

    def d1(y):
        x = y - _U
        return float(np.sum(hp * (-x / THETA**2) * np.exp(-0.5 * (x / THETA) ** 2)))

    lo, hi = -12.0, 3.0
    assert d1(lo) > 0 > d1(hi), lam
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if d1(mid) > 0:
            lo = mid
        else:
            hi = mid
    _YCACHE[key] = 0.5 * (lo + hi)
    return _YCACHE[key]




# ---------------------------------------------------------------------------
# Exact EM-chain free clock (deterministic): G_i,n = c_n * N(mean_z,n - c_i; var_z,n + sd^2) / A
# ---------------------------------------------------------------------------
def _wsum(w, rows) -> np.ndarray:
    out = np.zeros(rows.shape[1])
    for wi, r in zip(w, rows):
        out += float(wi) * r
    return out


class Clock:
    def __init__(self, eps: float):
        spec = fk.EnsembleSpec(m=2, eps=eps)
        mom = fk.em_moments(spec)
        self.eps = eps
        self.t = mom["t"]                       # step times n dt, n = 1..steps
        self.mz = mom["mean_z"]
        self.vz = mom["var_z"]
        self.c = fk.contact_probability_em(spec)
        self.sd = spec.sd()
        self.dt = spec.dt
        self.steps = spec.steps()

    def g_slab(self, centre: float) -> np.ndarray:
        v = self.vz + self.sd**2
        return self.c * np.exp(-(self.mz - centre) ** 2 / (2 * v)) / np.sqrt(2 * math.pi * v) / AREA

    def slabs(self, centres) -> np.ndarray:
        return np.array([self.g_slab(float(c)) for c in centres])

    def mf_pstep(self, centres, w, B) -> np.ndarray:
        """Mean-field per-step reaction probability e^{-B Lam_{n-1}} (1 - e^{-B G_n dt})."""
        G = _wsum(w, self.slabs(centres))
        Lam = np.concatenate([[0.0], np.cumsum(G) * self.dt])
        return np.exp(-B * Lam[:-1]) * (-np.expm1(-B * G * self.dt))

    def g_valleys(self, tp, w) -> list[float]:
        """Minimum of G = sum_i w_i G_i between consecutive stripe times (parabolic refine)."""
        G = _wsum(w, self.slabs(mu(tp)))
        out = []
        for a, b in zip(tp[:-1], tp[1:]):
            sel = np.flatnonzero((self.t > a) & (self.t < b))
            i = sel[int(np.argmin(G[sel]))]
            y0, y1, y2 = G[i - 1], G[i], G[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
            out.append(float(self.t[i] + off * self.dt))
        return out

    def cut_exposures(self, centres, cuts) -> np.ndarray:
        """L[i, k] = unit mean exposure of slab i accumulated over the steps n dt <= cut k."""
        Gs = self.slabs(centres)
        L = np.concatenate([np.zeros((Gs.shape[0], 1)), np.cumsum(Gs, axis=1) * self.dt], axis=1)
        idx = [cut_step(s) for s in cuts]
        return L[:, idx]


def cut_step(s: float) -> int:
    """Number of steps whose kill time n*dt is <= s (cuts snapped to the step grid)."""
    return int(math.floor(s / DT + 1e-9))


_CLOCKS: dict = {}


def clock(eps: float) -> Clock:
    if eps not in _CLOCKS:
        _CLOCKS[eps] = Clock(eps)
    return _CLOCKS[eps]


def cuts_of(eps, tp, w) -> list[float]:
    return [0.0] + clock(eps).g_valleys(np.asarray(tp, float), w) + [TMAX]


def target_cuts(m) -> list[float]:
    T = TARGETS[m]
    return [0.0] + [0.5 * (a + b) for a, b in zip(T[:-1], T[1:])] + [TMAX]


def mf_weights(m, eps, B, tp, iters: int = 20) -> dict:
    """Finite-eps mean-field max-min (equal-mass) allocation for stripes at mu(tp), on the
    G-valley cuts of the design itself (fixed point in (w, cuts))."""
    ck = clock(eps)
    tp = np.asarray(tp, float)
    w = [1.0 / m] * m
    cuts = cuts_of(eps, tp, w)
    for _ in range(iters):
        Lcut = ck.cut_exposures(mu(tp), cuts)
        d = n1.mf_design(B, [1.0 / m] * m, Lcut, None, "maxmin_mf")
        w = d["w"]
        new = cuts_of(eps, tp, w)
        if max(abs(a - b) for a, b in zip(new, cuts)) < 0.5 * DT:
            cuts = new
            break
        cuts = new
    return {"w": [float(x) for x in w], "target_mass_mf": float(d["target"][0]),
            "feasible_at_B": bool(d["feasible_at_B"]), "budget_used": float(d["budget_used"]),
            "cuts": cuts}


def limit_weights(m, B, tp) -> dict:
    """Limit-law max-min allocation (Proposition 1 / Box 2) for stripes at mu(tp)."""
    res = alloc.p_star(B, [float(x) for x in speed(tp)], AREA)
    return {"w": [float(x) for x in res["weights"]], "p_star": float(res["p_star"]),
            "lambda": [float(x) for x in res["exposures"]]}


# ---------------------------------------------------------------------------
# peak locations and basin masses
# ---------------------------------------------------------------------------
def gauss_smooth(y: np.ndarray, h: float) -> np.ndarray:
    """Gaussian smoothing (sd h steps) along the last axis, reflecting edges."""
    if h <= 0:
        return np.asarray(y, float)
    k = np.arange(-int(5 * h), int(5 * h) + 1)
    ker = np.exp(-0.5 * (k / h) ** 2)
    ker /= ker.sum()
    pad = k.size // 2
    yp = np.pad(y, [(0, 0)] * (np.ndim(y) - 1) + [(pad, pad)], mode="reflect")
    return np.apply_along_axis(lambda r: np.convolve(r, ker, mode="valid"), -1, yp)


def peak_in(t: np.ndarray, dens: np.ndarray, lo: float, hi: float) -> tuple[float, bool]:
    """Maximum of dens on (lo, hi], 3-point parabola; flag False if it sits on the boundary."""
    sel = np.flatnonzero((t > lo) & (t <= hi))
    k = int(np.argmax(dens[sel]))
    i = sel[k]
    interior = 0 < k < sel.size - 1
    if i <= 0 or i >= t.size - 1:
        return float(t[i]), interior
    y0, y1, y2 = dens[i - 1], dens[i], dens[i + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return float(t[i] + off * (t[1] - t[0])), interior


def peaks_of(t, dens, cuts, h=SMOOTH_STEPS) -> tuple[list[float], list[bool]]:
    s = gauss_smooth(dens, h)
    res = [peak_in(t, s, cuts[j], cuts[j + 1]) for j in range(len(cuts) - 1)]
    return [r[0] for r in res], [r[1] for r in res]


PEAK_FRAC = 0.9


def fit_peak(t, dens, lo, hi, win=None, frac: float = PEAK_FRAC, h: float = SMOOTH_STEPS):
    """Measured peak time: vertex of the least-squares parabola through the RAW per-step density
    on the contiguous region around the maximum where the density smoothed over h steps is
    >= frac * its maximum (at least +-10 steps).  With the window held fixed the vertex is a
    smooth functional of the data, so jackknife replicates reuse the full-sample window.
    Returns (vertex, window, interior flag)."""
    s = gauss_smooth(dens, h)
    sel = np.flatnonzero((t > lo) & (t <= hi))
    k = int(np.argmax(s[sel]))
    i0 = sel[k]
    interior = 0 < k < sel.size - 1
    if win is None:
        thr = frac * s[i0]
        a = i0
        while a > sel[0] and s[a - 1] >= thr:
            a -= 1
        b = i0
        while b < sel[-1] and s[b + 1] >= thr:
            b += 1
        win = (int(min(a, i0 - 10)), int(max(b, i0 + 10)))
    a, b = win
    tc = 0.5 * (t[a] + t[b])
    c = np.polyfit(t[a:b + 1] - tc, dens[a:b + 1], 2)
    if c[0] >= 0:
        return float(t[i0]), win, False
    return float(tc - c[1] / (2 * c[0])), win, interior


def peaks_fit(t, dens, cuts, wins=None):
    out = [fit_peak(t, dens, cuts[j], cuts[j + 1], None if wins is None else wins[j])
           for j in range(len(cuts) - 1)]
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]


def masses_of(pstep, cuts) -> list[float]:
    """Basin masses from per-step probabilities p_n (n = 1..steps), cuts on the step grid."""
    c = np.concatenate([[0.0], np.cumsum(pstep)])
    idx = [cut_step(s) for s in cuts]
    return [float(c[idx[j + 1]] - c[idx[j]]) for j in range(len(idx) - 1)]


def predict_A(m, eps, B, tp, w) -> list[float]:
    v = speed(tp)
    lam = B * np.asarray(w) / (AREA * v)
    return [float(tp[j] + eps * RHO * ystar(lam[j]) / v[j]) for j in range(m)]


def predict_B(m, eps, B, tp, w, cuts=None) -> list[float]:
    ck = clock(eps)
    cuts = cuts_of(eps, tp, w) if cuts is None else cuts
    ps = ck.mf_pstep(mu(tp), w, B)
    return peaks_of(ck.t, ps / ck.dt, cuts, h=0.0)[0]


# ---------------------------------------------------------------------------
# design loop
# ---------------------------------------------------------------------------
PREDICTORS = {"A": predict_A, "B": predict_B}


def evaluate_model(m, eps, B, tp, w, predictor) -> list[float]:
    return PREDICTORS[predictor](m, eps, B, np.asarray(tp, float), w)


def design_compensated(m, eps, B, predictor: str, *, max_iter: int = 60, tol: float = 1e-6,
                       tp0=None, t_cap: float = 3.5) -> dict:
    """Fixed-point / secant iteration for stripe times tp with peaks(tp, w(tp)) = T.

    Iteration 1 is the plain fixed-point step tp <- tp + (T - peak) (unit gain); later
    iterations use per-component secant slopes (clipped to [0.05, 2]).  w(tp) is the
    mean-field max-min allocation on the design's own G-valley cuts.
    """
    T = np.asarray(TARGETS[m], float)
    tp = T.copy() if tp0 is None else np.asarray(tp0, float).copy()
    hist = []
    prev = None
    for it in range(max_iter + 1):
        mw = mf_weights(m, eps, B, tp)
        pk = np.asarray(evaluate_model(m, eps, B, tp, mw["w"], predictor))
        err = T - pk
        hist.append({"iter": it, "tp": tp.tolist(), "w": mw["w"], "model_peaks": pk.tolist(),
                     "max_abs_model_err": float(np.max(np.abs(err))),
                     "target_mass_mf": mw["target_mass_mf"], "cuts": mw["cuts"]})
        if np.max(np.abs(err)) < tol or it == max_iter:
            break
        if prev is None:
            slope = np.ones(m)
        else:
            dtp = tp - prev[0]
            dpk = pk - prev[1]
            slope = np.where(np.abs(dtp) > 1e-12, dpk / np.where(np.abs(dtp) > 1e-12, dtp, 1), 1.0)
            slope = np.clip(slope, 0.05, 2.0)
        prev = (tp.copy(), pk.copy())
        tp = tp + err / slope
        tp = np.minimum(tp, t_cap)
    last = hist[-1]
    return {"predictor": predictor, "t_cap": t_cap,
            "converged": bool(last["max_abs_model_err"] < tol),
            "tp": last["tp"], "centres_z": mu(last["tp"]).tolist(), "w": last["w"],
            "model_peaks": last["model_peaks"], "target_mass_mf": last["target_mass_mf"],
            "cuts": last["cuts"], "iterations": len(hist) - 1,
            "history_first4": hist[:4], "model_err_by_iter": [h["max_abs_model_err"] for h in hist]}


def peak_map(m, eps, B, predictor, j, grid) -> list[list[float]]:
    """Model peak j as a function of stripe time t'_j (other stripes at their targets,
    weights re-designed): monotonicity / attainability of the target."""
    T = np.asarray(TARGETS[m], float)
    out = []
    for x in grid:
        tp = T.copy()
        tp[j] = x
        mw = mf_weights(m, eps, B, tp)
        out.append([float(x), float(evaluate_model(m, eps, B, tp, mw["w"], predictor)[j])])
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=1))
    os.replace(tmp, path)


def model_summary(m, eps, B, tp, w) -> dict:
    """Both predictors and the mean-field masses of one design (on its own G-valley cuts)."""
    tp = np.asarray(tp, float)
    cuts = cuts_of(eps, tp, w)
    ps = clock(eps).mf_pstep(mu(tp), w, B)
    return {"tp": tp.tolist(), "centres_z": mu(tp).tolist(), "w": [float(x) for x in w],
            "cuts": cuts, "peaks_A": predict_A(m, eps, B, tp, w),
            "peaks_B": predict_B(m, eps, B, tp, w, cuts),
            "masses_mf": masses_of(ps, cuts), "masses_mf_target_cuts": masses_of(ps, target_cuts(m)),
            "lambda_limit": (B * np.asarray(w) / (AREA * speed(tp))).tolist()}


def designs_for_cell(m, eps, B) -> dict:
    T = np.asarray(TARGETS[m], float)
    out = {"cell": {"m": m, "eps": eps, "B": B}, "targets": T.tolist(), "designs": {}}
    lw = limit_weights(m, B, T)
    mwn = mf_weights(m, eps, B, T)
    out["p_star_limit_nominal"] = lw["p_star"]
    D = out["designs"]
    D["box2"] = {**model_summary(m, eps, B, T, lw["w"]), "rule": "stripes at T; limit-law max-min w (Box 2)",
                 "target_mass": lw["p_star"]}
    D["mf_nominal"] = {**model_summary(m, eps, B, T, mwn["w"]),
                       "rule": "stripes at T; mean-field max-min w", "target_mass": mwn["target_mass_mf"]}
    # referee's proposal: one leading-order pre-shift step (Theorem 2(c)), w re-designed
    pa = np.asarray(predict_A(m, eps, B, T, mwn["w"]))
    tp1 = np.minimum(T + (T - pa), 3.5)
    mw1 = mf_weights(m, eps, B, tp1)
    D["preA"] = {**model_summary(m, eps, B, tp1, mw1["w"]), "target_mass": mw1["target_mass_mf"],
                 "rule": "one unit-gain step t' = T - eps rho y*(lambda)/v (Theorem 2(c)); mean-field w"}
    for P in ("A", "B"):
        d = design_compensated(m, eps, B, P, t_cap=3.5)
        D[f"comp{P}"] = {**model_summary(m, eps, B, d["tp"], d["w"]), "target_mass": d["target_mass_mf"],
                         "rule": f"predictor {P} fixed point, stripes inside the window (t' <= 3.5)",
                         "converged": d["converged"], "iterations": d["iterations"],
                         "model_err_by_iter": d["model_err_by_iter"], "history_first4": d["history_first4"]}
        if not d["converged"] and P == "B":
            e = design_compensated(m, eps, B, P, t_cap=6.0)
            D["compB_ext"] = {**model_summary(m, eps, B, e["tp"], e["w"]), "target_mass": e["target_mass_mf"],
                              "rule": "predictor B fixed point, stripes allowed beyond the window (t' <= 6)",
                              "converged": e["converged"], "iterations": e["iterations"],
                              "model_err_by_iter": e["model_err_by_iter"]}
    grid = np.round(np.arange(2.5, 6.001, 0.125), 4)
    out["late_peak_map"] = {P: peak_map(m, eps, B, P, m - 1, grid) for P in ("A", "B")}
    for P in ("A", "B"):
        arr = np.asarray(out["late_peak_map"][P])
        win = arr[arr[:, 0] <= 3.5 + 1e-9]
        out[f"late_peak_max_in_window_{P}"] = [float(win[np.argmax(win[:, 1]), 0]), float(win[:, 1].max())]
    return out


def cmd_design(args) -> None:
    res = {"item": "N11", "driver": HERE.name, "targets": {str(k): v for k, v in TARGETS.items()},
           "model": "gamma=1, D0=1, z_bar=0, z0=4, W=1, a=0.4, rho=1, u0=0.3, sigma_perp0=0.3, "
                    "r_par0=0.1, r_perp0=0; dt=1e-3, tmax=4", "cells": {}}
    t0 = time.time()
    for (m, eps, B) in CELLS:
        res["cells"][cell_key(m, eps, B)] = designs_for_cell(m, eps, B)
        print(f"[design] {cell_key(m, eps, B)} done {time.time() - t0:.1f}s", flush=True)
    write_json(DESIGN_JSON, res)
    print(f"wrote {DESIGN_JSON}")


# ---------------------------------------------------------------------------
# Feynman--Kac evaluation (declared mode: exact per-step kill probabilities)
# ---------------------------------------------------------------------------
def compute_python_count(threshold: float = 20.0) -> int:
    """Python processes currently using > threshold % CPU (idle MCP servers excluded)."""
    import subprocess
    out = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True).stdout
    n = 0
    for ln in out.splitlines()[1:]:
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and "ython" in parts[1]:
            try:
                if float(parts[0]) > threshold:
                    n += 1
            except ValueError:
                pass
    return n


def wait_compute(limit: int = 6, poll: float = 60.0, max_wait: float = 1500.0) -> dict:
    t0 = time.time()
    while True:
        n = compute_python_count()
        if n < limit or time.time() - t0 > max_wait:
            return {"busy_python_processes": n, "waited_seconds": time.time() - t0}
        print(f"[wait] {n} busy python processes; sleeping {poll:.0f}s", flush=True)
        time.sleep(poll)


def fk_groups(cell: dict, names) -> list[tuple[str, list[str]]]:
    """Group designs with identical stripes into one declared ensemble."""
    groups: list[tuple[str, list[str]]] = []
    for dn in names:
        tp = cell["designs"][dn]["tp"]
        for gname, members in groups:
            if np.allclose(cell["designs"][members[0]]["tp"], tp, atol=1e-12, rtol=0):
                members.append(dn)
                break
        else:
            groups.append((dn, [dn]))
    return groups


def phase1_names(cell: dict) -> list[str]:
    D = cell["designs"]
    names = ["box2", "mf_nominal", "preA"]
    if D["compA"].get("converged"):
        names.append("compA")
    names.append("compB")
    if "compB_ext" in D:
        names.append("compB_ext")
    return names


def run_fk_designs(ck: str, cell: dict, names, n_paths: int, workers: int = 3) -> list[str]:
    m, eps, B = cell["cell"]["m"], cell["cell"]["eps"], cell["cell"]["B"]
    done = []
    for gname, members in fk_groups(cell, names):
        d0 = cell["designs"][members[0]]
        spec = fk.EnsembleSpec(m=m, eps=eps, centres_z=tuple(float(c) for c in d0["centres_z"]))
        items = [dict(B=float(B), w=[float(x) for x in cell["designs"][dn]["w"]], label=dn)
                 for dn in members]
        name = f"n11_{ck}_{gname}"
        info = wait_compute()
        t0 = time.time()
        fk.simulate_ensemble(spec, n_paths, name=name, tag=TAG_FK, mode="declared", declared=items,
                             workers=workers, chunk=FK_CHUNK, wait=False, verbose=True,
                             note=f"N11 {ck}: designs {members}; cpu wait {info}")
        print(f"[fk] {name} ({members}) {time.time() - t0:.0f}s", flush=True)
        done.append(name)
    return done


def cmd_fk(args) -> None:
    D = json.loads(DESIGN_JSON.read_text())
    keys = list(D["cells"]) if not args.only else args.only.split(",")
    for ck in keys:
        cell = D["cells"][ck]
        names = phase1_names(cell) if not args.designs else args.designs.split(",")
        run_fk_designs(ck, cell, names, int(float(args.paths)), workers=args.workers)


def fk_item_stats(p: np.ndarray, pb: np.ndarray, t: np.ndarray, cuts, extra_cuts=None) -> dict:
    """Peaks and basin masses of one exact per-step law p (steps,) with batch laws pb (nb, steps);
    delete-one-batch jackknife SEs (equal batch sizes)."""
    nb = pb.shape[0]
    dens = p / DT
    pk, wins, interior = peaks_fit(t, dens, cuts)
    pk_argmax = peaks_of(t, dens, cuts)[0]
    M = masses_of(p, cuts)
    loo = (nb * p[None, :] - pb) / (nb - 1)
    pk_j = np.array([peaks_fit(t, r / DT, cuts, wins)[0] for r in loo])
    M_j = np.array([masses_of(r, cuts) for r in loo])

    def jse(a):
        return np.sqrt((nb - 1) / nb * np.sum((a - a.mean(0)) ** 2, axis=0))

    out = {"peaks": pk, "peaks_interior": interior, "peaks_se_jk": jse(pk_j).tolist(),
           "peak_windows": [[float(t[a]), float(t[b])] for a, b in wins],
           "peaks_argmax_smoothed5": pk_argmax,
           "masses": M, "masses_se_jk": jse(M_j).tolist(), "cuts": list(cuts),
           "mass_before_first_cut_after_tmax": float(1.0 - p.sum())}
    if extra_cuts is not None:
        out["masses_target_cuts"] = masses_of(p, extra_cuts)
        out["masses_target_cuts_se_jk"] = jse(np.array([masses_of(r, extra_cuts) for r in loo])).tolist()
    return out


def load_fk_items(name: str) -> dict:
    ens = fk.load_ensemble(name)
    res = ens.declared()
    out = {}
    for k, d in enumerate(res["declared"]):
        out[d["label"]] = {"p": res["p_step"][k], "pb": res["batch_p_step"][k], "t": res["t"],
                           "N": res["N"], "seed_entropy": ens.index["seed_entropy"],
                           "tag": ens.index["tag"], "ensemble": name}
    return out


def _x_to_design(m, x):
    tp = np.asarray(x[:m], float)
    w = np.concatenate([x[m:], [1.0 - float(np.sum(x[m:]))]])
    return tp, w


def model_outputs(m, eps, B, x) -> np.ndarray:
    """(predictor-B peaks, mean-field mass imbalance M_j - mean M, j < m) on the design's cuts."""
    tp, w = _x_to_design(m, x)
    cuts = cuts_of(eps, tp, w)
    pk = predict_B(m, eps, B, tp, w, cuts)
    M = np.asarray(masses_of(clock(eps).mf_pstep(mu(tp), w, B), cuts))
    return np.concatenate([pk, (M - M.mean())[: m - 1]])


def model_jacobian(m, eps, B, x, dt_step=2e-3, dw_step=1e-3) -> np.ndarray:
    x = np.asarray(x, float)
    J = np.zeros((2 * m - 1, 2 * m - 1))
    for k in range(2 * m - 1):
        h = dt_step if k < m else dw_step
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        J[:, k] = (model_outputs(m, eps, B, xp) - model_outputs(m, eps, B, xm)) / (2 * h)
    return J


def fk_residual(m, cell, dn, st) -> np.ndarray:
    T = np.asarray(cell["targets"], float)
    M = np.asarray(st["masses"], float)
    return np.concatenate([np.asarray(st["peaks"]) - T, (M - M.mean())[: m - 1]])


def newton_from_fk(ck, cell, base: str, t_cap: float) -> dict:
    """One Newton step x1 = x0 - J_model(x0)^{-1} r_FK(x0) (model Jacobian, measured residual)."""
    m, eps, B = cell["cell"]["m"], cell["cell"]["eps"], cell["cell"]["B"]
    d0 = cell["designs"][base]
    name = next(f"n11_{ck}_{g}" for g, mem in fk_groups(cell, [base]))
    # the ensemble holding `base` may be named after another member of its group
    for g, mem in fk_groups(cell, list(cell["designs"])):
        if base in mem and (FK_INDEX_DIR / f"n11_{ck}_{g}.json").exists():
            name = f"n11_{ck}_{g}"
    x = load_fk_items(name)[base]
    st = fk_item_stats(x["p"], x["pb"], x["t"], d0["cuts"])
    x0 = np.concatenate([d0["tp"], d0["w"][: m - 1]])
    r = fk_residual(m, cell, base, st)
    J = model_jacobian(m, eps, B, x0)
    dx = -np.linalg.solve(J, r)
    # stripes that would leave [0, t_cap]: pin them at the cap and drop their peak equations
    pinned = [k for k in range(m) if x0[k] + dx[k] > t_cap]
    if pinned:
        free = [k for k in range(2 * m - 1) if k not in pinned]
        rows = [k for k in range(2 * m - 1) if k not in pinned]
        xp = x0.copy()
        xp[pinned] = t_cap
        # residual after moving the pinned stripes (linear model), then solve the reduced system
        rp = r + J[:, pinned] @ (xp[pinned] - x0[pinned])
        dx = np.zeros_like(x0)
        dx[pinned] = xp[pinned] - x0[pinned]
        dx[free] = -np.linalg.solve(J[np.ix_(rows, free)], rp[rows])
    x1 = x0 + dx
    tp1, w1 = _x_to_design(m, x1)
    if np.any(w1 <= 0):
        raise AssertionError("Newton step left the simplex")
    return {**model_summary(m, eps, B, tp1, w1), "rule": f"one Newton step from {base}: model (predictor B, "
            "mean-field masses) Jacobian, Feynman--Kac residual", "base": base, "t_cap": t_cap,
            "fk_residual_at_base": r.tolist(), "jacobian": J.tolist(), "step": dx.tolist(),
            "pinned_at_cap": pinned, "target_mass": float(np.mean(st["masses"]))}


def cmd_fkcorr(args) -> None:
    D = json.loads(DESIGN_JSON.read_text())
    keys = list(D["cells"]) if not args.only else args.only.split(",")
    for ck in keys:
        cell = D["cells"][ck]
        new = []
        for base, cap, lab in (("compB", 3.5, "nwt1"), ("compB_ext", 6.0, "nwt1_ext"),
                               ("nwt1", 3.5, "nwt2"), ("nwt1_ext", 6.0, "nwt2_ext")):
            if base not in cell["designs"] or lab in cell["designs"]:
                continue
            if lab.startswith("nwt2") and not args.second:
                continue
            cell["designs"][lab] = newton_from_fk(ck, cell, base, cap)
            new.append(lab)
            print(f"[fkcorr] {ck} {lab}: tp {np.round(cell['designs'][lab]['tp'], 4)} "
                  f"w {np.round(cell['designs'][lab]['w'], 4)}", flush=True)
        write_json(DESIGN_JSON, D)
        if new:
            run_fk_designs(ck, cell, new, int(float(args.paths)), workers=args.workers)


# ---------------------------------------------------------------------------
# direct-kill confirmation (independent simulator, 1e6 walkers, tag 91)
# ---------------------------------------------------------------------------
def _dk_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    out = core.simulate_chunk_general(
        rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]), weights=tuple(task["w"]),
        centres_z=spec.centres(), dt=spec.dt, step_count=spec.steps(), p=spec.model(),
        n_perp=spec.n_perp)
    steps = spec.steps()
    k = np.rint(out["kill_times"] / spec.dt).astype(np.int64)      # kill step n (time n dt)
    counts = np.bincount(k - 1, minlength=steps)[:steps]
    return {"chunk": task["chunk"], "counts": counts, "survivors": out["survivors"],
            "kill_probability_max": out["kill_probability_max"]}


def dk_entropy(m, eps, B, w, centres, walkers):
    q = lambda x: int(round(float(x) * 1e9)) % (1 << 63)  # noqa: E731
    return [SEED, TAG_DK, int(m), q(eps), q(B)] + [q(x) for x in w] + [q(c) for c in centres] + [int(walkers)]


def cmd_dk(args) -> None:
    import multiprocessing as mp
    D = json.loads(DESIGN_JSON.read_text())
    cells = DK_CELLS if not args.only else [tuple(float(v) if i else int(v) for i, v in enumerate(c.split(":")))
                                            for c in args.only.split(",")]
    walkers = int(float(args.walkers))
    chunk = int(float(args.chunk))
    DK_DIR.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    for (m, eps, B) in cells:
        ck = cell_key(int(m), eps, B)
        d = D["cells"][ck]["designs"][args.design]
        outp = DK_DIR / f"dk_{ck}_{args.design}.json"
        if outp.exists():
            print(f"[dk] {outp.name} exists, skip", flush=True)
            continue
        spec = fk.EnsembleSpec(m=int(m), eps=eps, centres_z=tuple(float(c) for c in d["centres_z"]))
        ent = dk_entropy(m, eps, B, d["w"], d["centres_z"], walkers)
        sizes = [chunk] * (walkers // chunk) + ([walkers % chunk] if walkers % chunk else [])
        children = np.random.SeedSequence(ent).spawn(len(sizes))
        tasks = [{"spec": spec.to_dict(), "size": sz, "seedseq": c, "B": B, "w": d["w"], "chunk": i}
                 for i, (sz, c) in enumerate(zip(sizes, children))]
        info = wait_compute()
        t0 = time.time()
        with ctx.Pool(processes=min(args.workers, 3)) as pool:
            res = sorted(pool.map(_dk_chunk, tasks), key=lambda r: r["chunk"])
        counts = np.array([r["counts"] for r in res])
        surv = int(sum(r["survivors"] for r in res))
        assert int(counts.sum()) + surv == walkers
        payload = {"cell": {"m": int(m), "eps": eps, "B": B}, "design": args.design,
                   "tp": d["tp"], "centres_z": d["centres_z"], "w": d["w"], "cuts": d["cuts"],
                   "walkers": walkers, "chunk": chunk, "seed": SEED, "tag": TAG_DK, "seed_entropy": ent,
                   "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
                   "simulator": "exact_m_prr_upgrade_core.simulate_chunk_general (EM + end-of-step Doi kill)",
                   "counts_per_step_per_chunk": counts.tolist(), "survivors_at_tmax": surv,
                   "kill_probability_max": max(r["kill_probability_max"] for r in res),
                   "wall_seconds": time.time() - t0, "cpu_wait": info}
        write_json(outp, payload)
        print(f"[dk] {ck} {args.design}: {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def ensemble_of(ck, cell, dn):
    for g, mem in fk_groups(cell, list(cell["designs"])):
        if dn in mem and (FK_INDEX_DIR / f"n11_{ck}_{g}.json").exists():
            return f"n11_{ck}_{g}"
    p = FK_INDEX_DIR / f"n11_{ck}_{dn}.json"
    return f"n11_{ck}_{dn}" if p.exists() else None


def dk_stats(dk: dict) -> dict:
    C = np.asarray(dk["counts_per_step_per_chunk"], float)
    N = float(dk["walkers"])
    nc = C.shape[0]
    sizes = np.full(nc, N / nc)
    p = C.sum(0) / N
    t = (np.arange(p.size) + 1) * DT
    cuts = dk["cuts"]
    pk, wins, interior = peaks_fit(t, p / DT, cuts)
    M = masses_of(p, cuts)
    loo = (C.sum(0)[None, :] - C) / (N - sizes[:, None])
    pk_j = np.array([peaks_fit(t, r / DT, cuts, wins)[0] for r in loo])
    M_j = np.array([masses_of(r, cuts) for r in loo])

    def jse(a):
        return np.sqrt((nc - 1) / nc * np.sum((a - a.mean(0)) ** 2, axis=0))

    return {"peaks": pk, "peaks_interior": interior, "peaks_se_jk": jse(pk_j).tolist(),
            "peaks_argmax_smoothed5": peaks_of(t, p / DT, cuts)[0],
            "masses": M, "masses_se_jk": jse(M_j).tolist(), "_p": p}


def cmd_analyze(args) -> None:
    D = json.loads(DESIGN_JSON.read_text())
    res = {"item": "N11", "driver": HERE.name, "designs_json": str(DESIGN_JSON.relative_to(fk.REPORT)),
           "definitions": {
               "peak": "vertex of the least-squares parabola through the raw per-step density on the "
                       "region (inside the design's basin) where the 5-step-smoothed density is >= 90% of "
                       "its maximum; SE = delete-one-batch jackknife with the window held fixed "
                       "(FK: 42 batches of 5000 paths; DK: 20 chunks of 5e4 walkers); "
                       "peaks_argmax_smoothed5 = secondary argmax estimator",
               "basins": "cuts 0 < valleys of the design's EM-exact free clock G < tmax = 4",
               "peak_error": "achieved peak time - target time T_j",
               "mass_err_equal": "max_j |M_j - mean_j M_j| (equal-mass target at the achieved total)",
               "mass_err_pstar": "max_j |M_j - p*(B,m)| (Box 2 limit-law target at the nominal stripes)"},
           "seeds": {"base": SEED, "fk_tag": TAG_FK, "dk_tag": TAG_DK}, "cells": {}}
    for ck, cell in D["cells"].items():
        T = np.asarray(cell["targets"], float)
        pstar = cell["p_star_limit_nominal"]
        rows = {}
        for dn, d in cell["designs"].items():
            name = ensemble_of(ck, cell, dn)
            if name is None:
                continue
            x = load_fk_items(name).get(dn)
            if x is None:
                continue
            st = fk_item_stats(x["p"], x["pb"], x["t"], d["cuts"], target_cuts(cell["cell"]["m"]))
            pk = np.asarray(st["peaks"])
            M = np.asarray(st["masses"])
            rows[dn] = {"ensemble": name, "n_paths": x["N"], "seed_entropy": x["seed_entropy"],
                        "tp": d["tp"], "w": d["w"], "cuts": d["cuts"], "rule": d.get("rule"),
                        "fk": st, "peak_error": (pk - T).tolist(),
                        "max_abs_peak_error": float(np.max(np.abs(pk - T))),
                        "mass_err_equal": float(np.max(np.abs(M - M.mean()))),
                        "mass_err_pstar": float(np.max(np.abs(M - pstar))),
                        "model_peaks_A": d["peaks_A"], "model_peaks_B": d["peaks_B"],
                        "model_error_A_vs_fk": (np.asarray(d["peaks_A"]) - pk).tolist(),
                        "model_error_B_vs_fk": (np.asarray(d["peaks_B"]) - pk).tolist(),
                        "masses_mf_model": d["masses_mf"],
                        "converged_model": d.get("converged")}
        dks = {}
        for f in sorted(DK_DIR.glob(f"dk_{ck}_*.json")):
            dk = json.loads(f.read_text())
            st = dk_stats(dk)
            dn = dk["design"]
            fkr = rows.get(dn, {}).get("fk")
            ent = {"design": dn, "walkers": dk["walkers"], "seed_entropy": dk["seed_entropy"],
                   "tag": dk["tag"], "peaks": st["peaks"], "peaks_se_jk": st["peaks_se_jk"],
                   "masses": st["masses"], "masses_se_jk": st["masses_se_jk"],
                   "peak_error": (np.asarray(st["peaks"]) - T).tolist(),
                   "survivors_at_tmax": dk["survivors_at_tmax"]}
            if fkr is not None:
                dp = np.asarray(st["peaks"]) - np.asarray(fkr["peaks"])
                sp = np.sqrt(np.asarray(st["peaks_se_jk"]) ** 2 + np.asarray(fkr["peaks_se_jk"]) ** 2)
                dm = np.asarray(st["masses"]) - np.asarray(fkr["masses"])
                sm = np.sqrt(np.asarray(st["masses_se_jk"]) ** 2 + np.asarray(fkr["masses_se_jk"]) ** 2)
                ent.update({"peak_dk_minus_fk": dp.tolist(), "peak_z": (dp / sp).tolist(),
                            "mass_dk_minus_fk": dm.tolist(), "mass_z": (dm / sm).tolist()})
            dks[dn] = ent
        res["cells"][ck] = {"cell": cell["cell"], "targets": T.tolist(), "p_star_limit_nominal": pstar,
                            "late_peak_max_in_window_A": cell["late_peak_max_in_window_A"],
                            "late_peak_max_in_window_B": cell["late_peak_max_in_window_B"],
                            "late_peak_B_stripe_far_beyond_window": peak_map(
                                cell["cell"]["m"], cell["cell"]["eps"], cell["cell"]["B"], "B",
                                cell["cell"]["m"] - 1, [5.0, 6.0, 8.0, 10.0]),
                            "designs": rows, "directkill": dks}
    # summary: before (box2) vs after (best in-window compensated design)
    summ = {}
    for ck, c in res["cells"].items():
        r = c["designs"]
        after = next((k for k in ("nwt2", "nwt1", "compB") if k in r), None)
        summ[ck] = {k: {"max_abs_peak_error": r[k]["max_abs_peak_error"],
                        "peak_error": [round(v, 4) for v in r[k]["peak_error"]],
                        "mass_err_equal": r[k]["mass_err_equal"], "mass_err_pstar": r[k]["mass_err_pstar"]}
                    for k in ("box2", "mf_nominal", "preA", "compA", "compB", "nwt1", "nwt2",
                              "compB_ext", "nwt1_ext", "nwt2_ext", "compFK", "compFK2",
                              "compFK_ext", "compFK2_ext") if k in r}
        summ[ck]["after_in_window"] = after
    res["summary"] = summ
    write_json(OUT_JSON, res)
    print(f"wrote {OUT_JSON}")


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    core.apply_prr_style()
    res = json.loads(OUT_JSON.read_text())
    D = json.loads(DESIGN_JSON.read_text())
    sty = {"box2": dict(marker="o", mfc="0.6", mec="0.25", label="Box 2 (stripes at targets)"),
           "preA": dict(marker="v", mfc=core.OI_ORANGE, mec="k", label="one-step Thm 2(c) pre-shift"),
           "compB": dict(marker="s", mfc=core.OI_SKY, mec="k", label="compensated (mean-field model)"),
           "final": dict(marker="D", mfc=core.OI_BLUE, mec="k", label="+ measured Newton steps (final)"),
           "final_ext": dict(marker="D", mfc="w", mec=core.OI_BLUE, label="final, last stripe beyond window")}

    def pick(r, key):
        if key == "final":
            return next((r[k] for k in ("nwt2", "nwt1") if k in r), None)
        if key == "final_ext":
            return next((r[k] for k in ("nwt2_ext", "nwt1_ext") if k in r), None)
        return r.get(key)

    fig = plt.figure(figsize=(7.0, 4.9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    dens_cells = [("m2_eps0.1_B8", "(a)"), ("m3_eps0.05_B8", "(b)")]
    for k, (ck, tag) in enumerate(dens_cells):
        ax = fig.add_subplot(gs[0, k])
        cell = D["cells"][ck]
        after = res["summary"][ck]["after_in_window"]
        for dn, col, lw in (("box2", "0.55", 1.0), (after, core.OI_BLUE, 1.2)):
            x = load_fk_items(ensemble_of(ck, cell, dn))[dn]
            ax.plot(x["t"], gauss_smooth(x["p"] / DT, SMOOTH_STEPS), color=col, lw=lw,
                    label="Box 2" if dn == "box2" else "compensated")
            for tpj in cell["designs"][dn]["tp"]:
                ax.plot([tpj], [0], marker="^", ms=4, color=col, clip_on=False, zorder=5)
        for Tj in cell["targets"]:
            ax.axvline(Tj, color="k", ls=":", lw=0.7)
        c = cell["cell"]
        ax.set_title(f"{tag} $m$={c['m']}, $\\varepsilon$={c['eps']:g}, $B$={c['B']:g}", fontsize=8)
        ax.set_xlim(0.4, 3.8)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("$t$")
        if k == 0:
            ax.set_ylabel("exact density $f(t)$")
            ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    keys = list(res["cells"])
    for k, (qty, ylab, tag) in enumerate((("peak_error", "peak time $-$ target", "(c)"),
                                           ("mass_dev", "$M_j-\\bar M$", "(d)"))):
        ax = fig.add_subplot(gs[1, k])
        xpos, xt, xl = 0, [], []
        for ck in keys:
            c = res["cells"][ck]
            m = c["cell"]["m"]
            xs = np.arange(m) + xpos
            for dn, st in sty.items():
                r = pick(c["designs"], dn)
                if r is None:
                    continue
                if qty == "peak_error":
                    y = np.asarray(r["peak_error"])
                    e = np.asarray(r["fk"]["peaks_se_jk"])
                else:
                    M = np.asarray(r["fk"]["masses"])
                    y = M - M.mean()
                    e = np.asarray(r["fk"]["masses_se_jk"])
                ax.errorbar(xs, y, yerr=e, ls="", marker=st["marker"], ms=3.6, mfc=st["mfc"],
                            mec=st["mec"], mew=0.6, ecolor="0.4", elinewidth=0.6)
            dk = c["directkill"].get(res["summary"][ck]["after_in_window"])
            if dk is not None:
                if qty == "peak_error":
                    y = np.asarray(dk["peak_error"])
                    e = np.asarray(dk["peaks_se_jk"])
                else:
                    M = np.asarray(dk["masses"])
                    y = M - M.mean()
                    e = np.asarray(dk["masses_se_jk"])
                ax.errorbar(xs + 0.3, y, yerr=e, ls="", marker="x", ms=4, color="k", mew=0.9,
                            elinewidth=0.6)
            xt.append(xpos + (m - 1) / 2)
            xl.append(f"({m},{c['cell']['eps']:g},{c['cell']['B']:g})")
            xpos += m + 1
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=6.5)
        ax.set_xlabel("cell $(m,\\varepsilon,B)$; one marker per peak $j$")
        ax.set_ylabel(ylab)
        ax.set_title(tag, fontsize=8, loc="left")
    handles = [Line2D([], [], ls="", marker=st["marker"], mfc=st["mfc"], mec=st["mec"], label=st["label"])
               for st in sty.values()]
    handles.append(Line2D([], [], ls="", marker="x", color="k", mew=0.9,
                          label="final, direct-kill check ($10^6$ walkers)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
    core.stream_tag(fig, "fb N11")
    written = core.save_figure(fig, fk.FIGURES / "fb_n11_shift_compensation")
    plt.close(fig)
    print("wrote", written)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("design")
    a = sub.add_parser("fk")
    a.add_argument("--only", default="")
    a.add_argument("--designs", default="")
    a.add_argument("--paths", default=str(N_PATHS))
    a.add_argument("--workers", type=int, default=3)
    a = sub.add_parser("fkcorr")
    a.add_argument("--only", default="")
    a.add_argument("--second", action="store_true")
    a.add_argument("--paths", default=str(N_PATHS))
    a.add_argument("--workers", type=int, default=3)
    a = sub.add_parser("dk")
    a.add_argument("--only", default="")
    a.add_argument("--design", default="nwt2")
    a.add_argument("--walkers", default=str(DK_WALKERS))
    a.add_argument("--chunk", default="50000")
    a.add_argument("--workers", type=int, default=3)
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    {"design": cmd_design, "fk": cmd_fk, "fkcorr": lambda a: cmd_fkcorr(a),
     "dk": lambda a: cmd_dk(a), "analyze": lambda a: cmd_analyze(a),
     "figure": lambda a: cmd_figure(a)}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
