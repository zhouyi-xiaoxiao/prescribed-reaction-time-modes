#!/usr/bin/env python3
"""Hero demonstration (uplift2 item 4): five programmed reaction-time peaks in an idealised
electro-osmotic (plug-flow) channel, with a Poiseuille contrast.  Idealised model channel; no experiment.

Model (N14 plug channel, fb_n14_universality; nondimensional U = D0 = rho = A = 1, X_0 = 0)
------------------------------------------------------------------------------------------
    X_n = X_{n-1} + U dt + eps sqrt(2 D0 dt) N_n,   stripes phi_s(x - c_j), s = eps rho (Gaussian),
    e_n = dt sum_j w_j phi_s(X_n - c_j) / A,   kill at the end of step n w.p. 1 - exp(-B e_n).
The exact count of the continuous-time limit is SM Thm gx:brownian-theorem (unconditional for
eps < eps_0).  Physical units: length L, time L/U, eps^2 = D/(U L) (Peclet 1/eps^2); B = k_tot/U with
k_tot = int kappa dx the total areal rate constant (m/s) of the pattern, split k_j = w_j k_tot.

Design problem (formulation F1 of item 1, fixed B)
--------------------------------------------------
Unknowns (c_1..c_5, u_1..u_4) (stripe positions and simplex tangent coordinates of w), outputs the
peak times t*_j (maximisers of f_{h_j}, per-passage bandwidths h_j = clip(0.1 sigma_j, 0.003, 0.05),
sigma_j = eps sqrt(2 D0 T_j / U + rho^2) / U) and the conditional ratios r_1..r_4 of the valley-cut
basin masses, exactly the functionals of fb_v2_saa_newton.functionals.  Targets T = (1, 1.3, 1.8, 2.2, 2.8) L/U and
ratio profiles rising (~j), falling (~6-j), equal, all at B = 8.

Three evaluations of the SAME discrete-time law
-----------------------------------------------
  quad  deterministic transfer operator in the co-moving frame xi = x - U t_n: the Gaussian step kernel is
        applied exactly in Fourier space (periodic grid, |xi| <= 13 sigma sqrt(tmax), dx <= s/6) and the
        kill is a pointwise factor; forward tangents give dp_n/d(c, w).  No sampling error: the exact
        root x_inf of the design map, the dt-bias at dt, dt/2, dt/4 and the SAA error x_N - x_inf.
  fk    Feynman--Kac exact-law estimator on unkilled paths with pathwise gradients (item 1's estimator
        for this transport); SAA Newton (item 1, Box 3) on one ensemble of N paths.
  dk    direct kill: each walker is removed at the first step at which B X_n (X_n = sum_{k<=n} e_k)
        exceeds an independent Exp(1) threshold -- the same law as removal w.p. 1 - exp(-B e_n) at the
        end of each step.  Levels dt, dt/2, dt/4 for the Total-Error Protocol (fb_v2_tep.richardson).
Step windows: a stripe is evaluated only on steps with |U t_n - c_j| <= 12 s + 9 sigma sqrt(t_n); between
windows the path jumps with the exact Gaussian increment (exact for this law up to < 1e-15).

Seeds: base 20260923; tag 206 (FK design / out-of-sample; replicate 0 / 1), 213 (DK), model code 1 plug /
2 Poiseuille; chunk k of an ensemble = child k of SeedSequence([base, tag, code, eps*1e9, dt*1e12,
replicate, level, n_paths]), Philox.

CLI (outputs under artifacts/data/exact_m_fixed_budget/V2_hero/)
    python3 fb_v2_hero_channel.py selftest         # quadrature vs FK, grid refinement, FD gradients
    python3 fb_v2_hero_channel.py design           # exact roots (quad) + SAA Newton (FK), both eps
    python3 fb_v2_hero_channel.py tep --eps 0.03   # direct kill at dt, dt/2, dt/4 + FK out of sample
    python3 fb_v2_hero_channel.py pois             # Poiseuille contrast with the same designs
    python3 fb_v2_hero_channel.py summary | figure
Multiprocessing uses the spawn context: the entry point is guarded by ``if __name__ == "__main__"``.
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
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import fb_allocation_law as alloc  # noqa: E402
import fb_v2_design_estimator as de  # noqa: E402
import fb_v2_saa_newton as sn  # noqa: E402

OUT = de.FB_DATA / "V2_hero"
FIG_STEM = de.FIGURES / "fb_v2_hero_channel"

# ----------------------------------------------------------------------------
# Model constants (N14 plug channel) and the design problem
# ----------------------------------------------------------------------------
U = 1.0
D0 = 1.0
RHO = 1.0
AREA = 1.0
M = 5
T_TARGET = (1.0, 1.3, 1.8, 2.2, 2.8)
B_BUDGET = 8.0
PROFILES = ("rising", "falling", "equal")
EPS_LIST = (0.03, 0.01)
TMAX = 4.0
DT_PER_EPS = 1.0 / 40.0            # N14 channel step: dt = eps/40
STRIPE_CUT = 12.0                  # stripe negligible beyond 12 s (exp(-72))
PATH_SD_CUT = 9.0                  # walkers within 9 sigma sqrt(t) of U t (P(outside) ~ 2e-19 each)
BASE_SEED = 20260923
TAG_FK = de.TAGS["HERO"]           # 206
TAG_DK = de.TAGS["HERO_DK"]        # 213
MODEL_CODE = {"plug": 1, "pois": 2}

# physical units (idealised channel; no experiment)
PHYS = {"L_m": 1.0e-3, "U_m_per_s": 1.0e-4}


def dt_of(eps: float) -> float:
    return eps * DT_PER_EPS


def sigma_of(eps: float) -> float:
    return eps * math.sqrt(2.0 * D0)


def passage_sd(T, eps: float) -> np.ndarray:
    """Time width of passage j: sd of X at T_j combined with the stripe sd, over U."""
    T = np.asarray(T, float)
    return np.sqrt(sigma_of(eps) ** 2 * T / U + (eps * RHO) ** 2) / U


def bandwidths(T, eps: float) -> np.ndarray:
    return np.clip(sn.KAPPA * passage_sd(T, eps), sn.H_LO, sn.H_HI)


KAPPA_VAL = 0.30                   # validation bandwidth for direct-kill histograms (KDE noise ~ (h/sigma)^-3/2)


def val_bandwidths(T, eps: float) -> np.ndarray:
    return np.clip(KAPPA_VAL * passage_sd(T, eps), sn.H_LO, sn.H_HI)


def phys_units(eps: float) -> dict:
    L, Uu = PHYS["L_m"], PHYS["U_m_per_s"]
    D = eps ** 2 * Uu * L
    return {"L_mm": L * 1e3, "U_um_per_s": Uu * 1e6, "time_unit_s": L / Uu, "D_m2_per_s": D,
            "peclet_UL_over_D": Uu * L / D, "stripe_sd_um": eps * RHO * L * 1e6,
            "stripe_tophat_equiv_width_um": math.sqrt(12.0) * eps * RHO * L * 1e6,
            "k_tot_um_per_s": B_BUDGET * Uu * 1e6,
            "k_tot_definition": "B = k_tot / U, k_tot = int kappa dx (total areal rate constant); k_j = w_j k_tot"}


def prop1_plug(r, B: float = B_BUDGET) -> dict:
    """Prop. 1 limit allocation for plug flow (all speeds U): lambda_j = B w_j / (A U)."""
    out = alloc.max_common_scale(float(B), list(map(float, r)), [U] * len(r), AREA)
    return {"w": np.array(out["weights"]), "yield": out["theta"], "masses": out["masses"],
            "lambda": out["exposures"]}


# ----------------------------------------------------------------------------
# Deterministic transfer-operator ("quadrature") law of the discrete-time model
# ----------------------------------------------------------------------------


@dataclass
class QLaw:
    """Law-like container accepted by fb_v2_saa_newton.functionals / analyse (no groups)."""
    t: np.ndarray
    p: np.ndarray
    dp: np.ndarray | None
    meta: dict = field(default_factory=dict)
    sizes: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    @property
    def n_paths(self) -> int:
        return 0


def _phi(u, s):
    return np.exp(-0.5 * (u / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)


def quad_law(c, w, B: float, eps: float, *, dt: float | None = None, tmax: float = TMAX,
             grad: bool = True, n_grid: int | None = None, half_width_sd: float = 13.0,
             t_start: float = 0.05) -> QLaw:
    """Exact discrete-time law p_n = P(kill at step n) by the Fourier transfer operator.

    Co-moving frame xi = X_n - U t_n: q_n(xi) (surviving sub-density) evolves by the exact Gaussian
    step kernel (variance sigma^2 dt, applied in Fourier space) and the pointwise kill factor
    exp(-B e(xi + U t_n)).  Before t_start the kill is below 1e-30 (checked) and q is the exact Gaussian
    N(0, sigma^2 t).  Tangents dq/dtheta (theta = c_1..c_m, w_1..w_m) are propagated alongside."""
    c = np.asarray(c, float)
    w = np.asarray(w, float)
    m = c.size
    dt = dt_of(eps) if dt is None else float(dt)
    S = int(round(tmax / dt))
    sig = sigma_of(eps)
    s = eps * RHO
    L = half_width_sd * sig * math.sqrt(tmax)
    if n_grid is None:
        n_grid = 1 << int(math.ceil(math.log2(2.0 * L / (s / 6.0))))
    dx = 2.0 * L / n_grid
    xi = -L + dx * np.arange(n_grid)
    k = 2.0 * math.pi * np.fft.rfftfreq(n_grid, d=dx)
    G = np.exp(-0.5 * sig ** 2 * dt * k ** 2)
    t_start = max(float(t_start), (4.0 * dx / sig) ** 2)          # initial Gaussian resolved (sd >= 4 dx)
    n0 = max(1, int(math.ceil(t_start / dt)))
    t0 = n0 * dt
    # neglected kill mass before n0 (first-order bound; e <= exact exposure)
    tt = dt * np.arange(1, n0 + 1)
    var = s ** 2 + sig ** 2 * tt
    pre = B * dt * np.sum([wj * np.exp(-0.5 * (cj - U * tt) ** 2 / var) / np.sqrt(2 * np.pi * var)
                           for wj, cj in zip(w, c)])
    q = np.exp(-0.5 * xi ** 2 / (sig ** 2 * t0)) / math.sqrt(2.0 * math.pi * sig ** 2 * t0)
    if sig * math.sqrt(t0) < 3.0 * dx or pre > 1e-30:
        raise ValueError(f"start not admissible: sd/dx {sig * math.sqrt(t0) / dx:.2f}, neglected kill {pre:.1e}")
    K = 2 * m
    dq = np.zeros((K, n_grid)) if grad else None
    p = np.zeros(S)
    dp = np.zeros((K, S)) if grad else None
    ph = np.empty((m, n_grid))
    for n in range(n0 + 1, S + 1):
        tn = n * dt
        q = np.fft.irfft(np.fft.rfft(q) * G, n_grid)
        x = xi + U * tn
        e = np.zeros(n_grid)
        act = np.abs(U * tn - c) <= L + STRIPE_CUT * s
        for j in np.flatnonzero(act):
            ph[j] = _phi(x - c[j], s)
            e += w[j] * ph[j]
        e *= dt / AREA
        E = np.exp(-B * e)
        p[n - 1] = dx * float(np.sum(q * (1.0 - E)))
        if grad:
            dq = np.fft.irfft(np.fft.rfft(dq, axis=1) * G[None, :], n_grid, axis=1)
            de_ = np.zeros((K, n_grid))
            for j in np.flatnonzero(act):
                de_[m + j] = ph[j] * dt / AREA
                de_[j] = w[j] * ph[j] * (x - c[j]) / s ** 2 * dt / AREA
            dE = -B * E[None, :] * de_
            dp[:, n - 1] = dx * np.sum(dq * (1.0 - E)[None, :] - q[None, :] * dE, axis=1)
            dq = dq * E[None, :] + q[None, :] * dE
        q = q * E
    survivors = dx * float(np.sum(q))
    t = dt * np.arange(1, S + 1)
    meta = {"kind": "quadrature", "eps": eps, "dt": dt, "tmax": tmax, "n_grid": n_grid, "dx": dx,
            "half_width": L, "n0": n0, "neglected_kill_before_n0_bound": float(pre),
            "survivors_at_tmax": survivors, "mass_balance_error": float(p.sum() + survivors - 1.0),
            "edge_density_rel": float(max(abs(q[0]), abs(q[-1])) / max(q.max(), 1e-300))}
    return QLaw(t, p, dp, meta)


# ----------------------------------------------------------------------------
# Path engines: FK (plug, with pathwise gradients), direct kill (plug), FK (Poiseuille)
# ----------------------------------------------------------------------------


def u_pois(y):
    return 1.5 * U * (1.0 - (2.0 * np.asarray(y) - 1.0) ** 2)


POIS_Y0 = (0.15, 0.85)             # N14 injection band: u in [0.765, 1.5]
POIS_TMAX = 5.5


def seed_seq(model: str, eps: float, dt: float, *, tag: int, replicate: int, level: int,
             n_paths: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([BASE_SEED, int(tag), MODEL_CODE[model], int(round(eps * 1e9)),
                                   int(round(dt * 1e12)), int(replicate), int(level), int(n_paths)])


def stripe_windows(c_all, eps: float, dt: float, S: int) -> np.ndarray:
    """(n_stripes, S) bool: stripe j is evaluated at step n (t_n = n dt)."""
    t = dt * np.arange(1, S + 1)
    rad = STRIPE_CUT * eps * RHO + PATH_SD_CUT * sigma_of(eps) * np.sqrt(t)
    return np.abs(U * t[None, :] - np.asarray(c_all, float)[:, None]) <= rad[None, :]


def _plug_chunk(task: dict) -> dict:
    """One chunk of plug-flow paths: FK law (+ gradients) or direct kill, for several designs.
    Group sums over consecutive blocks of task['group'] paths (jackknife groups)."""
    mode, eps, dt, tmax = task["mode"], task["eps"], task["dt"], task["tmax"]
    designs = task["designs"]            # list of (c, w, B)
    n = int(task["n"])
    gsz = int(task.get("group", 5000))
    G = int(math.ceil(n / gsz))
    gid = np.minimum(np.arange(n) // gsz, G - 1)
    sizes = np.bincount(gid, minlength=G).astype(float)
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(**task["seed"])))
    S = int(round(tmax / dt))
    sig, s = sigma_of(eps), eps * RHO
    cs = [np.asarray(d[0], float) for d in designs]
    ws = [np.asarray(d[1], float) for d in designs]
    Bs = [float(d[2]) for d in designs]
    m = cs[0].size
    K = 2 * m
    win = [stripe_windows(c, eps, dt, S) for c in cs]
    steps = np.flatnonzero(np.any(np.vstack(win), axis=0)) + 1
    X = np.zeros(n)
    last = 0
    nd = len(designs)
    out = {"Y": [np.zeros((G, S)) for _ in range(nd)], "n": n, "sizes": sizes, "max_std_excursion": 0.0}

    def gsum(v):                           # per-group sums of a path vector (all paths present)
        return np.bincount(gid, weights=v, minlength=G)

    if mode == "fk":
        grad = bool(task.get("grad", True))
        Xc = [np.zeros(n) for _ in range(nd)]
        dXc = [np.zeros((K, n)) for _ in range(nd)] if grad else None
        out["dY"] = [np.zeros((G, K, S)) for _ in range(nd)] if grad else None
        full = (n % gsz == 0)
    else:
        thr = [rng.exponential(1.0, n) for _ in range(nd)]
        H = [np.zeros(n) for _ in range(nd)]
        alive = [np.ones(n, bool) for _ in range(nd)]
    for n_ in steps:
        k = n_ - last
        X += U * k * dt + sig * math.sqrt(k * dt) * rng.standard_normal(X.size)
        last = n_
        tn = n_ * dt
        if X.size:
            ex = float(np.max(np.abs(X - U * tn))) / (sig * math.sqrt(tn))
            out["max_std_excursion"] = max(out["max_std_excursion"], ex)
        for d in range(nd):
            act = np.flatnonzero(win[d][:, n_ - 1])
            if act.size == 0:
                continue
            c, w, B = cs[d], ws[d], Bs[d]
            phs = {j: _phi(X - c[j], s) for j in act}
            e = sum(w[j] * phs[j] for j in act) * (dt / AREA)
            if mode == "fk":
                A_ = np.exp(-B * Xc[d])
                Q = -np.expm1(-B * e)
                y = A_ * Q
                out["Y"][d][:, n_ - 1] = y.reshape(G, -1).sum(1) if full else gsum(y)
                if grad:
                    a1 = B * A_ * (1.0 - Q)
                    aq = -B * y
                    if full:
                        g = np.einsum("gi,kgi->gk", aq.reshape(G, -1), dXc[d].reshape(K, G, -1))
                    else:
                        g = np.stack([gsum(aq * dXc[d][kk]) for kk in range(K)], axis=1)
                    for j in act:
                        dew = phs[j] * (dt / AREA)
                        dec = w[j] * phs[j] * (X - c[j]) / s ** 2 * (dt / AREA)
                        g[:, m + j] += (a1 * dew).reshape(G, -1).sum(1) if full else gsum(a1 * dew)
                        g[:, j] += (a1 * dec).reshape(G, -1).sum(1) if full else gsum(a1 * dec)
                        dXc[d][m + j] += dew
                        dXc[d][j] += dec
                    out["dY"][d][:, :, n_ - 1] = g
                Xc[d] += e
            else:
                H[d] += B * e
                kill = alive[d] & (H[d] >= thr[d])
                if kill.any():
                    out["Y"][d][:, n_ - 1] = np.bincount(gid[kill], minlength=G)
                    alive[d] &= ~kill
        if mode == "dk" and (n_ % 200 == 0):
            keep = np.zeros(X.size, bool)
            for d in range(nd):
                keep |= alive[d]
            if keep.sum() < 0.8 * X.size:
                X, gid = X[keep], gid[keep]
                for d in range(nd):
                    thr[d], H[d], alive[d] = thr[d][keep], H[d][keep], alive[d][keep]
                if X.size == 0:
                    break
    if mode == "dk":
        out["survivors"] = [int(a.sum()) for a in alive]
    return out


def _pois_chunk(task: dict) -> dict:
    """Poiseuille channel (N14 pois): unkilled (x, y) paths, FK law and position-cell masses."""
    eps, dt, tmax = task["eps"], task["dt"], task["tmax"]
    designs = task["designs"]
    n = int(task["n"])
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(**task["seed"])))
    S = int(round(tmax / dt))
    sig, s = sigma_of(eps), eps * RHO
    x = np.zeros(n)
    y = rng.uniform(POIS_Y0[0], POIS_Y0[1], n)
    nd = len(designs)
    cs = [np.asarray(d[0], float) for d in designs]
    ws = [np.asarray(d[1], float) for d in designs]
    Bs = [float(d[2]) for d in designs]
    m = cs[0].size
    mids = [0.5 * (c[:-1] + c[1:]) for c in cs]
    Xc = [np.zeros(n) for _ in range(nd)]
    at = [np.full((m - 1, n), np.nan) for _ in range(nd)]       # exposure at first passage of midpoints
    Y = [np.zeros(S) for _ in range(nd)]
    sq = sig * math.sqrt(dt)
    for n_ in range(1, S + 1):
        x += u_pois(y) * dt + sq * rng.standard_normal(n)
        y += sq * rng.standard_normal(n)
        y = np.abs(y)
        y = np.where(y > 1.0, 2.0 - y, y)
        xlo, xhi = float(x.min()), float(x.max())
        for d in range(nd):
            c = cs[d]
            act = np.flatnonzero((c >= xlo - STRIPE_CUT * s) & (c <= xhi + STRIPE_CUT * s))
            if act.size:
                e = sum(ws[d][j] * _phi(x - c[j], s) for j in act) * (dt / AREA)
                Y[d][n_ - 1] = float((np.exp(-Bs[d] * Xc[d]) * -np.expm1(-Bs[d] * e)).sum())
                Xc[d] += e
            for j in range(m - 1):
                newly = np.isnan(at[d][j]) & (x >= mids[d][j])
                if newly.any():
                    at[d][j][newly] = Xc[d][newly]
    cells = []
    for d in range(nd):
        a = at[d]
        a = np.where(np.isnan(a), Xc[d][None, :], a)
        ex = np.vstack([np.zeros(n), a, Xc[d][None, :]])
        cells.append((np.exp(-Bs[d] * ex[:-1]) - np.exp(-Bs[d] * ex[1:])).sum(axis=1))
    return {"Y": Y, "cells": cells, "n": n,
            "y_range_final": [float(y.min()), float(y.max())]}


def run_chunks(fn, tasks: list, workers: int = 3) -> list:
    workers = max(1, min(int(workers), de.MAX_LOCAL_WORKERS, len(tasks)))
    if workers == 1:
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=1)


def _seed_dict(ss: np.random.SeedSequence) -> dict:
    return {"entropy": ss.entropy, "spawn_key": tuple(ss.spawn_key)}


def ensemble_tasks(mode: str, eps: float, designs: list, n_paths: int, *, tag: int, replicate: int = 0,
                   level: int = 1, chunk: int = 25_000, tmax: float = TMAX, grad: bool = True,
                   model: str = "plug") -> list:
    dt = dt_of(eps) / level
    nch = int(math.ceil(n_paths / chunk))
    root = seed_seq(model, eps, dt, tag=tag, replicate=replicate, level=level, n_paths=n_paths)
    kids = root.spawn(nch)
    des = [(list(map(float, c)), list(map(float, w)), float(B)) for c, w, B in designs]
    return [{"mode": mode, "eps": eps, "dt": dt, "tmax": tmax, "designs": des, "grad": grad,
             "n": min(chunk, n_paths - i * chunk), "seed": _seed_dict(kids[i])} for i in range(nch)]


@dataclass
class GLaw:
    """Grouped law (one group per chunk) in the format of fb_v2_design_estimator.Law."""
    t: np.ndarray
    p: np.ndarray
    dp: np.ndarray | None
    gY: np.ndarray
    gdY: np.ndarray | None
    sizes: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return int(self.sizes.sum())

    def loo(self, g: int) -> tuple:
        if "_tot" not in self.meta:
            self.meta["_tot"] = self.gY.sum(0)
            self.meta["_dtot"] = None if self.gdY is None else self.gdY.sum(0)
        N = self.n_paths - self.sizes[g]
        p = (self.meta["_tot"] - self.gY[g]) / N
        dp = None if self.gdY is None else (self.meta["_dtot"] - self.gdY[g]) / N
        return p, dp


def assemble(results: list, d: int, dt: float, extra: dict | None = None) -> GLaw:
    gY = np.concatenate([np.atleast_2d(r["Y"][d]) for r in results]).astype(float)
    sizes = np.concatenate([np.atleast_1d(r.get("sizes", r["n"])) for r in results]).astype(float)
    grad = results[0].get("dY") is not None
    gdY = np.concatenate([r["dY"][d] for r in results]).astype(float) if grad else None
    N = sizes.sum()
    t = dt * np.arange(1, gY.shape[1] + 1)
    return GLaw(t, gY.sum(0) / N, None if gdY is None else gdY.sum(0) / N, gY, gdY, sizes, dict(extra or {}))


# ----------------------------------------------------------------------------
# Functionals (item 1's per-passage functionals) and design coordinates
# ----------------------------------------------------------------------------


def cuts0_of(c, tmax: float = TMAX) -> list:
    c = np.sort(np.asarray(c, float))
    return [0.0] + [float(0.5 * (a + b) / U) for a, b in zip(c[:-1], c[1:])] + [float(tmax)]


def analyse(law, eps: float, c, *, jackknife: bool = False, tmax: float = TMAX, kappa: str = "design") -> dict:
    """item 1 functionals; kappa 'design' (h_j = 0.1 sigma_j, the design functional) or 'val' (0.3 sigma_j)."""
    hs = bandwidths(T_TARGET, eps) if kappa == "design" else val_bandwidths(T_TARGET, eps)
    return sn.analyse(law, hs, cuts0_of(c, tmax), jackknife=jackknife)


def f1(res: dict):
    y, J = sn.f1_outputs(res, M)
    if J is not None:
        J = np.hstack([J[:, :M], J[:, M:2 * M - 1] - J[:, 2 * M - 1:2 * M]])   # (c, u) coordinates
    return y, J


def targets_of(kind: str) -> np.ndarray:
    return np.concatenate([np.array(T_TARGET), sn.profile(M, kind)[:M - 1]])


def step_x(c, w, dx, a: float = 1.0):
    cn = np.asarray(c, float) + a * dx[:M]
    wn = np.asarray(w, float).copy()
    wn[:M - 1] += a * dx[M:]
    wn[M - 1] -= a * float(np.sum(dx[M:]))
    return cn, wn


def admissible(c, w) -> bool:
    return bool(np.all(w > 1e-7) and np.all(np.diff(c) > 0.05) and c[0] > 0.2)


def raw_peaks(p, t, cuts) -> list:
    """Unsmoothed peak times: argmax of p_n in each basin, refined by a parabola through 3 points."""
    out = []
    dt = float(t[1] - t[0])
    for a, b in zip(cuts[:-1], cuts[1:]):
        sel = np.flatnonzero((t > a) & (t <= b))
        i = sel[int(np.argmax(p[sel]))]
        y0, y1, y2 = p[i - 1], p[i], p[i + 1]
        den = y0 - 2 * y1 + y2
        out.append(float(t[i] + (0.5 * (y0 - y2) / den * dt if den < 0 else 0.0)))
    return out


# ----------------------------------------------------------------------------
# selftest: quadrature law vs FK / DK, grid refinement, gradients
# ----------------------------------------------------------------------------


def _binned(p, t, lo=0.5, hi=5.8, width=0.02):
    edges = np.arange(lo, hi + 1e-12, width)
    idx = np.digitize(t, edges)
    return np.array([p[idx == k].sum() for k in range(1, edges.size)])


def cmd_selftest(args) -> dict:
    eps = float(args.eps)
    kind = "equal"
    pr = prop1_plug(sn.profile(M, kind))
    c0, w0 = np.array(T_TARGET, float), pr["w"]
    out = {"item": "V2 hero selftest", "eps": eps, "design": {"c": c0, "w": w0, "B": B_BUDGET}}
    q = quad_law(c0, w0, B_BUDGET, eps, grad=True)
    qf = quad_law(c0, w0, B_BUDGET, eps, grad=False, n_grid=2 * q.meta["n_grid"], half_width_sd=16.0)
    out["quad_meta"] = q.meta
    out["grid_refinement_max_abs_dp"] = float(np.max(np.abs(q.p - qf.p)))
    out["grid_refinement_rel_to_pmax"] = float(np.max(np.abs(q.p - qf.p)) / q.p.max())
    # finite-difference check of the tangents (c3, w2, w5)
    fd = {}
    for name, k, hstep in (("c3", 2, 1e-5), ("w2", M + 1, 1e-6), ("w5", 2 * M - 1, 1e-6)):
        th = np.concatenate([c0, w0])
        tp, tm = th.copy(), th.copy()
        tp[k] += hstep
        tm[k] -= hstep
        lp = quad_law(tp[:M], tp[M:], B_BUDGET, eps, grad=False)
        lm = quad_law(tm[:M], tm[M:], B_BUDGET, eps, grad=False)
        g_fd = (lp.p - lm.p) / (2 * hstep)
        fd[name] = {"max_abs_diff": float(np.max(np.abs(g_fd - q.dp[k]))),
                    "rel_to_max": float(np.max(np.abs(g_fd - q.dp[k])) / np.max(np.abs(q.dp[k])))}
    out["fd_tangent_check"] = fd
    # FK and DK on the same design
    n = int(float(args.n))
    for mode, tag in (("fk", TAG_FK), ("dk", TAG_DK)):
        tasks = ensemble_tasks(mode, eps, [(c0, w0, B_BUDGET)], n, tag=tag, replicate=90, chunk=25_000)
        t0 = time.time()
        rs = run_chunks(_plug_chunk, tasks, workers=args.workers)
        law = assemble(rs, 0, dt_of(eps))
        bq = _binned(q.p, q.t)
        G = law.gY.shape[0]
        gb = np.array([_binned(law.gY[g] / law.sizes[g], law.t) for g in range(G)])
        mb = gb.mean(0)
        se = gb.std(0, ddof=1) / math.sqrt(G)
        ok = (se > 0) & (bq > 1e-4 * bq.max())
        z = (mb[ok] - bq[ok]) / se[ok]
        kap = "design" if mode == "fk" else "val"
        rq = analyse(q, eps, c0, kappa=kap)
        rl = analyse(law, eps, c0, jackknife=True, kappa=kap)
        zpk = (np.array(rl["peaks"]) - np.array(rq["peaks"])) / rl["se"]["peaks"]
        zr = (np.array(rl["ratios"]) - np.array(rq["ratios"])) / rl["se"]["ratios"]
        # basin-level tests (valley cuts of the quadrature law): mass and mean kill time per basin.
        cuts = rq["cuts"]
        bz_m, bz_t = [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            sel = (law.t > a) & (law.t <= b)
            gm = np.array([law.gY[g][sel].sum() / law.sizes[g] for g in range(G)])
            gt = np.array([(law.gY[g][sel] * law.t[sel]).sum() / max(law.gY[g][sel].sum(), 1e-300) for g in range(G)])
            mq = q.p[sel].sum()
            tq = (q.p[sel] * q.t[sel]).sum() / mq
            mm = law.p[sel].sum()
            tm = (law.p[sel] * law.t[sel]).sum() / mm
            bz_m.append(float((mm - mq) / (gm.std(ddof=1) / math.sqrt(G))))
            bz_t.append(float((tm - tq) / (gt.std(ddof=1) / math.sqrt(G))))
        rec = {"bandwidth": kap, "n": n, "seconds": time.time() - t0, "bins": int(ok.sum()),
               "chi2_naive_bins": float(np.sum(z ** 2)),
               "chi2_note": "naive: bins of one basin are strongly correlated for FK (one path feeds many bins)",
               "basin_mass_z": bz_m, "basin_mean_time_z": bz_t,
               "max_abs_z_bins": float(np.max(np.abs(z))), "peaks_z": zpk, "ratios_z": zr,
               "max_std_excursion": max(r["max_std_excursion"] for r in rs)}
        if mode == "fk":
            J = law.dp
            zg = []
            for k in (2, M + 1, 2 * M - 1):
                gg = np.array([_binned(law.gdY[g][k] / law.sizes[g], law.t) for g in range(G)])
                sg = gg.std(0, ddof=1) / math.sqrt(G)
                bqk = _binned(q.dp[k], q.t)
                okk = sg > 1e-3 * np.max(np.abs(bqk))
                zg.append(float(np.sum(((gg.mean(0)[okk] - bqk[okk]) / sg[okk]) ** 2) / max(okk.sum(), 1)))
            rec["grad_chi2_per_bin_c3_w2_w5"] = zg
            _ = J
        out[mode] = rec
        print(f"[selftest] {mode}: {rec['seconds']:.0f}s basin mass z {np.round(bz_m, 2)} mean-t z {np.round(bz_t, 2)} "
              f"peaks z {np.round(zpk, 2)} ratios z {np.round(zr, 2)}", flush=True)
    de.write_json(OUT / f"selftest_eps{eps:g}.json", out)
    return out


# ----------------------------------------------------------------------------
# design: exact roots on the quadrature law + SAA Newton (item 1, Box 3) on an FK ensemble
# ----------------------------------------------------------------------------


def _eval_quad(c, w, eps, *, dt=None, grad=True, kappa="design"):
    law = quad_law(c, w, B_BUDGET, eps, dt=dt, grad=grad)
    res = analyse(law, eps, c, kappa=kappa)
    return law, res


def summarise_law(law, res, eps, c) -> dict:
    """Peaks, ratios, masses, yield, raw peaks (unsmoothed), exactly-m check of one evaluated law."""
    hs = bandwidths(T_TARGET, eps)
    mc = sn.mode_check(law, hs, res)
    return {"peaks": res["peaks"], "ratios": res["ratios"], "masses": res["masses"], "yield": res["yield"],
            "valleys": res["valleys"], "f_peak": res["f_peak"], "f_valley": res["f_valley"],
            "valley_to_peak": [float(res["f_valley"][j] / min(res["f_peak"][j], res["f_peak"][j + 1]))
                               for j in range(M - 1)],
            "raw_peaks": raw_peaks(law.p, law.t, res["cuts"]), "mode_check": mc}


def quad_newton(kind: str, eps: float, c0, w0, *, tol: float = 1e-12, it_max: int = 25) -> dict:
    tgt = targets_of(kind)
    c, w = np.asarray(c0, float), np.asarray(w0, float)
    table = []
    for it in range(it_max + 1):
        law, res = _eval_quad(c, w, eps)
        y, J = f1(res)
        F = y - tgt
        r = float(np.max(np.abs(F)))
        table.append({"iter": it, "res_inf": r, "c": c.tolist(), "w": w.tolist()})
        if r < tol:
            break
        dx = np.linalg.solve(J, -F)
        a = 1.0
        while True:
            cn, wn = step_x(c, w, dx, a)
            if admissible(cn, wn):
                break
            a *= 0.5
        c, w = cn, wn
    for k in range(1, len(table)):
        prev = table[k - 1]["res_inf"]
        table[k]["ratio_to_prev_sq"] = table[k]["res_inf"] / prev ** 2 if prev > 0 else None
    out = {"residual_table": table, "c": c, "w": w, "cond_J": float(np.linalg.cond(J)), "J": J,
           "converged": bool(table[-1]["res_inf"] < tol), **summarise_law(law, res, eps, c)}
    return out


def saa_newton(eps: float, kinds: list, starts: dict, n_paths: int, *, workers: int = 3, tag: int = TAG_FK,
               replicate: int = 0, tol: float = 1e-11, it_max: int = 8, log=print) -> dict:
    """Lockstep SAA Newton for several profiles on ONE FK ensemble (paths regenerated every pass)."""
    tg = {k: targets_of(k) for k in kinds}
    cur = {k: (np.asarray(starts[k][0], float), np.asarray(starts[k][1], float)) for k in kinds}
    table = {k: [] for k in kinds}
    last = {}
    done = set()
    for it in range(it_max + 1):
        todo = [k for k in kinds if k not in done] if it < it_max else list(kinds)
        des = [(cur[k][0], cur[k][1], B_BUDGET) for k in todo]
        t0 = time.time()
        tasks = ensemble_tasks("fk", eps, des, n_paths, tag=tag, replicate=replicate, chunk=25_000, grad=True)
        rs = run_chunks(_plug_chunk, tasks, workers)
        for i, k in enumerate(todo):
            law = assemble(rs, i, dt_of(eps), {"kind": "fk", "tag": tag, "replicate": replicate})
            c, w = cur[k]
            res = analyse(law, eps, c, jackknife=True)
            y, J = f1(res)
            F = y - tg[k]
            r = float(np.max(np.abs(F)))
            prev = table[k][-1]["res_inf"] if table[k] else None
            table[k].append({"pass": it, "res_inf": r, "c": c.tolist(), "w": w.tolist(),
                             "ratio_to_prev_sq": (r / prev ** 2 if prev else None),
                             "seconds": time.time() - t0})
            last[k] = (law, res, J, F)
            log(f"[saa eps={eps:g}] pass {it} {k}: res_inf {r:.3e} ({time.time() - t0:.0f}s)")
            if r < tol:
                done.add(k)
                continue
            dx = np.linalg.solve(J, -F)
            a = 1.0
            while True:
                cn, wn = step_x(c, w, dx, a)
                if admissible(cn, wn):
                    break
                a *= 0.5
            cur[k] = (cn, wn)
        if len(done) == len(kinds):
            break
    out = {}
    for k in kinds:
        law, res, J, F = last[k]
        c, w = np.array(table[k][-1]["c"]), np.array(table[k][-1]["w"])
        # delta-method design covariance from the jackknife replicates of y at fixed x
        reps = np.hstack([res["jk_reps"]["peaks"], res["jk_reps"]["ratios"][:, :M - 1]])
        G = reps.shape[0]
        Cy = (G - 1) / G * (reps - reps.mean(0)).T @ (reps - reps.mean(0))
        Ji = np.linalg.inv(J)
        Cx = Ji @ Cy @ Ji.T
        out[k] = {"residual_table": table[k], "c": c, "w": w, "converged": bool(table[k][-1]["res_inf"] < tol),
                  "cond_J": float(np.linalg.cond(J)), "n_paths": law.n_paths, "groups": int(law.sizes.size),
                  "se": {q: res["se"][q] for q in ("peaks", "ratios", "masses", "yield")},
                  "design_se_delta": np.sqrt(np.diag(Cx)), "design_cov_delta": Cx,
                  **summarise_law(law, res, eps, c)}
    return out


def design_payload(eps: float, kind: str, c, w) -> dict:
    return {"label": f"hero_m5_eps{eps:g}_B{B_BUDGET:g}_{kind}", "model": "plug", "m": M, "eps": eps,
            "B": B_BUDGET, "c": list(map(float, c)), "w": list(map(float, w)), "tmax": TMAX,
            "dt_design": dt_of(eps), "h": bandwidths(T_TARGET, eps).tolist(),
            "h_val": val_bandwidths(T_TARGET, eps).tolist(),
            "targets": {"peaks": list(T_TARGET), "ratios": sn.profile(M, kind).tolist()}}


def cmd_design(args) -> dict:
    eps_list = [float(e) for e in str(args.eps).split(",")] if args.eps != "all" else list(EPS_LIST)
    n = int(float(args.n))
    for eps in eps_list:
        log = sn.Logger(OUT / "logs" / f"design_eps{eps:g}.log")
        out = {"item": "V2 hero design (uplift2 item 4)", "driver": HERE.name, "eps": eps, "B": B_BUDGET,
               "targets_T": T_TARGET, "tmax": TMAX, "dt": dt_of(eps), "phys": phys_units(eps),
               "h_design": bandwidths(T_TARGET, eps), "h_val": val_bandwidths(T_TARGET, eps),
               "definitions": {
                   "design_map": "F1: y = (t*_1..t*_5, r_1..r_4) of the exact discrete-time law at dt = eps/40; "
                                 "x = (c_1..c_5, u_1..u_4), B = 8 fixed",
                   "quad": "deterministic transfer-operator law (no sampling error): exact root x_inf",
                   "saa": "Newton on the FK sample-average map (item 1 Box 3), N paths, tag 206 replicate "
                          f"{args.replicate}; start = Prop. 1 limit (stripes at U T_j)",
                   "naive": "Prop. 1 limit design (c_j = U T_j, limit weights) evaluated exactly"},
               "cells": {}}
        starts = {}
        for kind in PROFILES:
            pr = prop1_plug(sn.profile(M, kind))
            c0, w0 = np.array(T_TARGET) * U, pr["w"]
            starts[kind] = (c0, w0)
            lawn, resn = _eval_quad(c0, w0, eps)
            naive = summarise_law(lawn, resn, eps, c0)
            naive["dev_peaks"] = (np.array(naive["peaks"]) - np.array(T_TARGET)).tolist()
            naive["dev_ratios"] = (np.array(naive["ratios"]) - sn.profile(M, kind)).tolist()
            t0 = time.time()
            qn = quad_newton(kind, eps, c0, w0)
            log(f"[quad eps={eps:g}] {kind}: {len(qn['residual_table']) - 1} steps, res "
                f"{qn['residual_table'][-1]['res_inf']:.2e}, c {np.round(qn['c'], 4)}, w {np.round(qn['w'], 4)} "
                f"({time.time() - t0:.0f}s)")
            out["cells"][kind] = {"target_ratios": sn.profile(M, kind), "prop1_limit": {"w": w0, "c": c0,
                                  "lambda": pr["lambda"], "yield": pr["yield"]},
                                  "naive": naive, "quad": qn}
        de.write_json(OUT / f"design_eps{eps:g}.json", out)
        if n > 0:
            saa = saa_newton(eps, list(PROFILES), starts, n, workers=args.workers, replicate=args.replicate, log=log)
            for kind in PROFILES:
                s_ = saa[kind]
                qn = out["cells"][kind]["quad"]
                xs = np.concatenate([s_["c"], s_["w"][:M - 1]])
                xi = np.concatenate([qn["c"], qn["w"][:M - 1]])
                s_["x_minus_xinf"] = xs - xi
                s_["x_minus_xinf_z"] = (xs - xi) / s_["design_se_delta"]
                lq, rq = _eval_quad(s_["c"], s_["w"], eps)
                ex = summarise_law(lq, rq, eps, s_["c"])
                ex["dev_peaks"] = (np.array(ex["peaks"]) - np.array(T_TARGET)).tolist()
                ex["dev_ratios"] = (np.array(ex["ratios"]) - sn.profile(M, kind)).tolist()
                s_["exact_eval_of_saa_design"] = ex
                out["cells"][kind]["saa"] = s_
                pay = design_payload(eps, kind, s_["c"], s_["w"])
                pay["fk_prediction"] = {"peaks": s_["peaks"], "ratios": s_["ratios"], "masses": s_["masses"],
                                        "se": s_["se"]}
                pay["quad_prediction"] = {"peaks": ex["peaks"], "ratios": ex["ratios"], "masses": ex["masses"],
                                          "raw_peaks": ex["raw_peaks"]}
                de.write_json(OUT / "designs" / f"{pay['label']}.json", pay)
            out["saa_meta"] = {"n_paths": n, "tag": TAG_FK, "replicate": args.replicate, "chunk": 25_000,
                               "group": 5000}
            de.write_json(OUT / f"design_eps{eps:g}.json", out)
    return {}


# ----------------------------------------------------------------------------
# tep: direct kill at dt, dt/2, dt/4 (Total-Error Protocol) + FK out of sample
# ----------------------------------------------------------------------------

QUANT = ("peaks", "ratios", "masses")


def load_designs(eps: float) -> dict:
    out = {}
    for kind in PROFILES:
        p = OUT / "designs" / f"hero_m5_eps{eps:g}_B{B_BUDGET:g}_{kind}.json"
        out[kind] = json.loads(p.read_text())
    return out


def _level_summary(law, eps, c, kappa) -> dict:
    r = analyse(law, eps, c, jackknife=True, kappa=kappa)
    return {"peaks": r["peaks"], "ratios": r["ratios"], "masses": r["masses"], "yield": r["yield"],
            "se": {q: r["se"][q] for q in QUANT + ("yield",)}, "cuts": r["cuts"],
            "n_paths": law.n_paths, "groups": int(law.sizes.size)}


def cmd_tep(args) -> dict:
    import fb_v2_tep as tep
    eps = float(args.eps)
    n = int(float(args.n))
    levels = [int(x) for x in str(args.levels).split(",")]
    D = load_designs(eps)
    kinds = list(PROFILES)
    des = [(D[k]["c"], D[k]["w"], D[k]["B"]) for k in kinds]
    log = sn.Logger(OUT / "logs" / f"tep_eps{eps:g}.log")
    out = {"item": "V2 hero TEP (direct kill, uplift2 section 3)", "driver": HERE.name, "eps": eps,
           "walkers_per_level": n, "levels": levels, "tag": TAG_DK, "replicate": args.replicate,
           "dt_design": dt_of(eps), "h_val": val_bandwidths(T_TARGET, eps), "h_design": bandwidths(T_TARGET, eps),
           "definitions": {
               "dk": "direct kill: removal at the first step with B X_n >= Exp(1) threshold (same law as per-step "
                     "removal w.p. 1-exp(-B e_n)); 3 designs share the Brownian paths, independent thresholds",
               "functionals": "item 1 functionals with validation bandwidths h'_j = clip(0.3 sigma_j, 0.003, 0.05); "
                              "valley cuts; jackknife over groups of 5000 walkers",
               "smoothing_correction": "delta_L = t*_h(quad_L) - t*_h'(quad_L) (exact); corrected peak = DK peak + delta_L "
                                       "estimates the design functional (h_j = 0.1 sigma_j)",
               "quad_z": "(DK_L - quad_L) / SE_L with quad_L the exact discrete law at dt_L (same functional)",
               "tep": "fb_v2_tep.richardson: R1 = 2Q(dt/4)-Q(dt/2), R2 = (8Q4-6Q2+Q1)/3; TE_L = |Q_L-R1|+2SE_L, "
                      "TE_c = |R2-R1|+2SE(R1); peaks pass if |Q-T| <= max(0.01, TE), ratios/masses if |Q-target| <= TE"},
           "cells": {k: {"design": D[k], "levels": {}} for k in kinds}}
    per = {k: {} for k in kinds}
    for L in levels:
        dtL = dt_of(eps) / L
        t0 = time.time()
        tasks = ensemble_tasks("dk", eps, des, n, tag=TAG_DK, replicate=args.replicate, level=L, chunk=25_000)
        rs = run_chunks(_plug_chunk, tasks, args.workers)
        log(f"[tep eps={eps:g}] level {L} (dt {dtL:.3e}) {n} walkers: {time.time() - t0:.0f}s")
        for i, k in enumerate(kinds):
            c = np.asarray(D[k]["c"])
            law = assemble(rs, i, dtL, {"kind": "dk", "level": L})
            sdk = _level_summary(law, eps, c, "val")
            ql = quad_law(c, D[k]["w"], B_BUDGET, eps, dt=dtL, grad=False)
            qv = analyse(ql, eps, c, kappa="val")
            qd = analyse(ql, eps, c, kappa="design")
            delta = np.array(qd["peaks"]) - np.array(qv["peaks"])
            rec = {"dt": dtL, "dk": sdk,
                   "dk_survivors": int(sum(r["survivors"][i] for r in rs)),
                   "max_std_excursion": max(r["max_std_excursion"] for r in rs),
                   "quad_val": {q: qv[q] for q in QUANT}, "quad_design": {q: qd[q] for q in QUANT},
                   "quad_raw_peaks": raw_peaks(ql.p, ql.t, qd["cuts"]), "smoothing_delta": delta,
                   "peaks_corrected": np.array(sdk["peaks"]) + delta,
                   "quad_z": {q: ((np.array(sdk[q]) - np.array(qv[q])) / np.array(sdk["se"][q])) for q in QUANT}}
            out["cells"][k]["levels"][L] = rec
            per[k][L] = rec
            log(f"   {k}: peaks {np.round(rec['peaks_corrected'], 4)} (se {np.round(sdk['se']['peaks'], 4)}) "
                f"ratios {np.round(sdk['ratios'], 4)} quad-z peaks {np.round(rec['quad_z']['peaks'], 2)} "
                f"ratios {np.round(rec['quad_z']['ratios'], 2)}")
        de.write_json(OUT / f"tep_eps{eps:g}.json", out)
    if all(L in levels for L in (1, 2, 4)):
        for k in kinds:
            lv = per[k]
            vals = {"peaks": {L: lv[L]["peaks_corrected"] for L in (1, 2, 4)}}
            for q in ("ratios", "masses"):
                vals[q] = {L: np.array(lv[L]["dk"][q]) for L in (1, 2, 4)}
            ses = {q: {L: np.array(lv[L]["dk"]["se"][q]) for L in (1, 2, 4)} for q in QUANT}
            R = {q: tep.richardson(vals[q], ses[q]) for q in QUANT}
            tg = D[k]["targets"]
            checks = {}
            for q, floor, tv in (("peaks", 0.01, tg["peaks"]), ("ratios", 0.0, tg["ratios"])):
                tv = np.asarray(tv, float)
                checks[q] = {"target": tv,
                             "design_dt": tep._target_checks(R[q]["levels"][1]["value"], R[q]["levels"][1]["total_error"],
                                                             tv, floor),
                             "continuum": tep._target_checks(R[q]["R1"], R[q]["total_error_continuum"], tv, floor)}
            # quadrature Richardson (exact dt-bias of the design functional)
            qR = {q: {L: np.array(lv[L]["quad_design"][q]) for L in (1, 2, 4)} for q in ("peaks", "ratios")}
            qrich = {q: {"R1": 2 * qR[q][4] - qR[q][2], "R2": (8 * qR[q][4] - 6 * qR[q][2] + qR[q][1]) / 3.0,
                         "bias_design_dt": qR[q][1] - (2 * qR[q][4] - qR[q][2])} for q in qR}
            allz = np.concatenate([np.ravel(lv[L]["quad_z"][q]) for L in (1, 2, 4) for q in ("peaks", "ratios")])
            out["cells"][k]["richardson"] = R
            out["cells"][k]["target_checks"] = checks
            out["cells"][k]["quad_richardson"] = qrich
            out["cells"][k]["quad_z_summary"] = {"n": int(allz.size), "max_abs": float(np.max(np.abs(allz))),
                                                 "rms": float(np.sqrt(np.mean(allz ** 2)))}
            out["cells"][k]["all_pass"] = bool(checks["peaks"]["design_dt"]["all_pass"] and
                                               checks["ratios"]["design_dt"]["all_pass"] and
                                               checks["peaks"]["continuum"]["all_pass"] and
                                               checks["ratios"]["continuum"]["all_pass"])
            log(f"[tep eps={eps:g}] {k}: all_pass {out['cells'][k]['all_pass']}; peak dev (dt) "
                f"{np.round(checks['peaks']['design_dt']['deviation'], 4)} allowed "
                f"{np.round(checks['peaks']['design_dt']['allowed'], 4)}; ratio dev "
                f"{np.round(checks['ratios']['design_dt']['deviation'], 4)} TE "
                f"{np.round(checks['ratios']['design_dt']['allowed'], 4)}")
        out["G2_all_pass"] = bool(all(out["cells"][k]["all_pass"] for k in kinds))
        de.write_json(OUT / f"tep_eps{eps:g}.json", out)
    return out


def cmd_oos(args) -> dict:
    """FK out-of-sample evaluation of the SAA designs (tag 206, replicate args.replicate >= 1)."""
    eps = float(args.eps)
    n = int(float(args.n))
    if args.replicate < 1:
        raise SystemExit("out-of-sample needs --replicate >= 1 (replicate 0 is the design ensemble)")
    D = load_designs(eps)
    kinds = list(PROFILES)
    des = [(D[k]["c"], D[k]["w"], D[k]["B"]) for k in kinds]
    t0 = time.time()
    tasks = ensemble_tasks("fk", eps, des, n, tag=TAG_FK, replicate=args.replicate, chunk=25_000, grad=False)
    rs = run_chunks(_plug_chunk, tasks, args.workers)
    out = {"item": "V2 hero FK out of sample", "eps": eps, "n_paths": n, "tag": TAG_FK, "replicate": args.replicate,
           "seconds": time.time() - t0, "cells": {}}
    for i, k in enumerate(kinds):
        law = assemble(rs, i, dt_of(eps))
        c = np.asarray(D[k]["c"])
        r = analyse(law, eps, c, jackknife=True)
        tg = D[k]["targets"]
        dT = np.array(r["peaks"]) - np.array(tg["peaks"])
        dr = np.array(r["ratios"]) - np.array(tg["ratios"])
        qp = D[k]["quad_prediction"]
        zx_p = (np.array(r["peaks"]) - np.array(qp["peaks"])) / r["se"]["peaks"]
        zx_r = (np.array(r["ratios"]) - np.array(qp["ratios"])) / r["se"]["ratios"]
        # same comparison with the validation bandwidth h' = 0.3 sigma_j (exact law evaluated with h')
        rv = analyse(law, eps, c, jackknife=True, kappa="val")
        qv = analyse(quad_law(c, D[k]["w"], B_BUDGET, eps, grad=False), eps, c, kappa="val")
        zv_p = (np.array(rv["peaks"]) - np.array(qv["peaks"])) / rv["se"]["peaks"]
        zv_r = (np.array(rv["ratios"]) - np.array(qv["ratios"])) / rv["se"]["ratios"]
        out["cells"][k] = {"peaks": r["peaks"], "ratios": r["ratios"], "masses": r["masses"], "yield": r["yield"],
                           "z_vs_exact_law_peaks": zx_p, "z_vs_exact_law_ratios": zx_r,
                           "val_bandwidth": {"peaks": rv["peaks"], "se_peaks": rv["se"]["peaks"],
                                             "z_vs_exact_law_peaks": zv_p, "z_vs_exact_law_ratios": zv_r},
                           "se": {q: r["se"][q] for q in ("peaks", "ratios", "masses", "yield")},
                           "dev_peaks": dT, "dev_ratios": dr, "z_peaks": dT / r["se"]["peaks"],
                           "z_ratios": dr / r["se"]["ratios"],
                           "mode_check": sn.mode_check(law, bandwidths(T_TARGET, eps), r)}
        print(f"[oos eps={eps:g}] {k}: z_exact(h) {np.round(zx_p, 2)} z_exact(h') {np.round(zv_p, 2)} "
              f"dT {np.round(dT, 5)} z {np.round(dT / r['se']['peaks'], 2)} "
              f"dr {np.round(dr, 5)} z {np.round(dr / r['se']['ratios'], 2)}", flush=True)
    de.write_json(OUT / f"oos_eps{eps:g}_rep{args.replicate}.json", out)
    return out


def cmd_oos_study(args) -> dict:
    """Calibration of the FK standard errors: K independent FK replicates (tag 206, replicates 11..10+K) of the
    designs; z of peaks and ratios against the exact (quadrature) law of each design."""
    eps = float(args.eps)
    n = int(float(args.n))
    K = int(args.levels) if str(args.levels).isdigit() else 8
    D = load_designs(eps)
    kinds = list(PROFILES)
    des = [(D[k]["c"], D[k]["w"], D[k]["B"]) for k in kinds]
    exact = {k: analyse(quad_law(D[k]["c"], D[k]["w"], B_BUDGET, eps, grad=False), eps, np.asarray(D[k]["c"]))
             for k in kinds}
    rows = []
    for rep in range(11, 11 + K):
        tasks = ensemble_tasks("fk", eps, des, n, tag=TAG_FK, replicate=rep, chunk=25_000, grad=False)
        rs = run_chunks(_plug_chunk, tasks, args.workers)
        row = {"replicate": rep}
        for i, k in enumerate(kinds):
            law = assemble(rs, i, dt_of(eps))
            r = analyse(law, eps, np.asarray(D[k]["c"]), jackknife=True)
            row[k] = {"z_peaks": (np.array(r["peaks"]) - np.array(exact[k]["peaks"])) / r["se"]["peaks"],
                      "z_ratios": (np.array(r["ratios"]) - np.array(exact[k]["ratios"])) / r["se"]["ratios"],
                      "se_peaks": r["se"]["peaks"], "dev_peaks": np.array(r["peaks"]) - np.array(exact[k]["peaks"])}
        rows.append(row)
        print(f"[oos-study eps={eps:g}] rep {rep}: z peaks (equal) {np.round(row['equal']['z_peaks'], 2)}", flush=True)
    out = {"item": "V2 hero FK SE calibration", "eps": eps, "n_paths": n, "replicates": K, "tag": TAG_FK, "rows": rows}
    for k in kinds:
        zp = np.array([r_[k]["z_peaks"] for r_ in rows])
        zr = np.array([r_[k]["z_ratios"] for r_ in rows])
        dv = np.array([r_[k]["dev_peaks"] for r_ in rows])
        se = np.array([r_[k]["se_peaks"] for r_ in rows])
        out[k] = {"rms_z_peaks_per_peak": np.sqrt((zp ** 2).mean(0)), "rms_z_ratios_per_ratio": np.sqrt((zr ** 2).mean(0)),
                  "rms_z_peaks_all": float(np.sqrt((zp ** 2).mean())), "rms_z_ratios_all": float(np.sqrt((zr ** 2).mean())),
                  "empirical_sd_over_mean_se_peaks": dv.std(0, ddof=1) / se.mean(0),
                  "mean_dev_over_se_of_mean_peaks": dv.mean(0) / (dv.std(0, ddof=1) / math.sqrt(K))}
    de.write_json(OUT / f"oos_study_eps{eps:g}.json", out)
    return out


# ----------------------------------------------------------------------------
# pois: Poiseuille contrast with the SAME designs (N14 pois channel)
# ----------------------------------------------------------------------------


def mixture_masses(w, B: float = B_BUDGET, n_quad: int = 400) -> dict:
    """TH13 mixture law (frozen transverse position): M_j = G(K_{j-1}) - G(K_j), K_j = B sum_{i<=j} w_i / A,
    G(K) = E exp(-K / u(Y0)), Y0 ~ U[0.15, 0.85] (Gauss-Legendre in y0)."""
    x, wq = np.polynomial.legendre.leggauss(n_quad)
    a, b = POIS_Y0
    y = 0.5 * (b - a) * x + 0.5 * (a + b)
    wy = 0.5 * wq                                   # uniform density 1/(b-a) times Jacobian (b-a)/2
    K = np.concatenate([[0.0], B * np.cumsum(w) / AREA])
    G = np.array([np.sum(wy * np.exp(-k / u_pois(y))) for k in K])
    return {"masses": -np.diff(G), "G": G, "K": K}


def prominent_maxima(t, f, rel_prominence: float = 1e-2, t_lo: float = 0.3) -> list:
    """Local maxima of f with topographic prominence > rel_prominence * max f: the base on each side is the
    minimum of f between the peak and the nearest higher point on that side (or the end of the range)."""
    t, f = np.asarray(t, float), np.asarray(f, float)
    sel = np.flatnonzero(t > t_lo)
    t, f = t[sel], f[sel]
    top = float(f.max())
    out = []
    for i in range(1, f.size - 1):
        if not (f[i] > f[i - 1] and f[i] >= f[i + 1]):
            continue
        hl = np.flatnonzero(f[:i] > f[i])
        hr = np.flatnonzero(f[i + 1:] > f[i])
        lb = f[(hl[-1] if hl.size else 0):i + 1].min()
        rb = f[i:(i + 1 + hr[0] + 1) if hr.size else f.size].min()
        if f[i] - max(lb, rb) > rel_prominence * top:
            out.append(float(t[i]))
    return out


def count_modes(p, t, h: float, rel_prominence: float = 1e-2, t_lo: float = 0.3) -> dict:
    f = de.smooth_grid(p, float(t[1] - t[0]), h, 0)
    peaks = prominent_maxima(t, f, rel_prominence, t_lo)
    return {"n_modes": len(peaks), "mode_times": peaks, "h": h, "rel_prominence": rel_prominence,
            "definition": "topographic prominence (base = min between the peak and the nearest higher point)"}


def cmd_pois_recount(args) -> dict:
    """Recount the Poiseuille modes on the stored smoothed curves with the topographic prominence (the first
    run used the global minima on each side, which counts every noise wiggle of a monotone tail)."""
    pj = OUT / "pois_contrast.json"
    P = json.loads(pj.read_text())
    for e, rec in P["eps"].items():
        for k, c in rec["cells"].items():
            if "modes_v0_global_min_prominence" not in c:
                c["modes_v0_global_min_prominence"] = c["modes"]
            cur = c["curve"]
            out = {}
            for rel in (1e-2, 5e-2):
                pk = prominent_maxima(cur["t"], cur["f_h"], rel)
                out[f"{rel:g}"] = {"n_modes": len(pk), "mode_times": pk}
            c["modes"] = {"n_modes": out["0.01"]["n_modes"], "mode_times": out["0.01"]["mode_times"],
                          "rel_prominence": 1e-2, "by_prominence": out, "h": cur["h"],
                          "grid": "stored curve (every 0.004 time units)",
                          "definition": "topographic prominence (base = min between the peak and the nearest higher point)"}
            print(f"[recount eps={e}] {k}: {out}", flush=True)
    de.write_json(pj, P)
    return P


def cmd_pois(args) -> dict:
    eps_list = [float(e) for e in str(args.eps).split(",")]
    n = int(float(args.n))
    out = {"item": "V2 hero Poiseuille contrast (same designs)", "driver": HERE.name, "n_paths": n,
           "tag": TAG_FK, "replicate": 3, "tmax": POIS_TMAX, "injection_band_y0": POIS_Y0,
           "flow": "u(y) = 1.5 U (1 - (2y-1)^2), mean U over the section; y reflects at 0, 1; same D; stripes span the section",
           "definitions": {"cells": "position cells: kills before first passage of the stripe midpoints (FK)",
                           "mixture": "TH13 mixture law with frozen transverse position",
                           "modes": "local maxima of f_h (h = 0.3 sigma_1) with prominence > 1e-2 max (FK tail noise gives spurious maxima at 1e-3)"},
           "eps": {}}
    for eps in eps_list:
        D = load_designs(eps)
        kinds = list(PROFILES)
        des = [(D[k]["c"], D[k]["w"], D[k]["B"]) for k in kinds]
        dt = dt_of(eps)
        t0 = time.time()
        tasks = ensemble_tasks("pois", eps, des, n, tag=TAG_FK, replicate=3, chunk=25_000, tmax=POIS_TMAX,
                               model="pois")
        rs = run_chunks(_pois_chunk, tasks, args.workers)
        rec = {"seconds": time.time() - t0, "dt": dt, "cells": {}}
        h = float(val_bandwidths(T_TARGET, eps)[0])
        for i, k in enumerate(kinds):
            gY = np.array([r["Y"][i] for r in rs])
            sizes = np.array([r["n"] for r in rs], float)
            p = gY.sum(0) / sizes.sum()
            t = dt * np.arange(1, p.size + 1)
            gc = np.array([r["cells"][i] / r["n"] for r in rs])
            cells = (gc * sizes[:, None]).sum(0) / sizes.sum()
            G = gc.shape[0]
            jk = np.array([(cells * sizes.sum() - gc[g] * sizes[g]) / (sizes.sum() - sizes[g]) for g in range(G)])
            se = np.sqrt((G - 1) / G * np.sum((jk - jk.mean(0)) ** 2, axis=0))
            mix = mixture_masses(np.asarray(D[k]["w"]))
            plug_masses = np.asarray(D[k].get("quad_prediction", {}).get("masses", [np.nan] * M))
            modes = count_modes(p, t, h)
            f = de.smooth_grid(p, dt, h, 0)                  # density per model time unit (kernel sum)
            stride = max(1, int(round(0.004 / dt)))
            rec["cells"][k] = {"cell_masses": cells, "cell_se": se, "mixture_masses": mix["masses"],
                               "cell_minus_mixture_z": (cells - mix["masses"]) / se,
                               "plug_design_masses": plug_masses, "yield": float(p.sum()),
                               "modes": modes, "centreline_arrival_times": (np.asarray(D[k]["c"]) / float(u_pois(0.5))).tolist(),
                               "curve": {"t": t[::stride], "f_h": f[::stride], "h": h}}
            print(f"[pois eps={eps:g}] {k}: {modes['n_modes']} modes at {np.round(modes['mode_times'], 3)}; "
                  f"cells {np.round(cells, 4)} mixture {np.round(mix['masses'], 4)} plug {np.round(plug_masses, 4)}",
                  flush=True)
        out["eps"][f"{eps:g}"] = rec
        de.write_json(OUT / "pois_contrast.json", out)
    return out


# ----------------------------------------------------------------------------
# figure: fb_v2_hero_channel.pdf (legible at graphical-abstract size, 2.6:1)
# ----------------------------------------------------------------------------


def cmd_figure(args) -> list:
    """(a) plug (electro-osmotic) flow: exact densities of the three designs, stripe pattern, direct-kill
    histogram; (b) Poiseuille flow, same patterns (FK); (c) achieved vs target ratios (direct kill, TEP)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exact_m_prr_upgrade_core as core
    core.apply_prr_style()
    eps = float(args.eps)
    tu = PHYS["L_m"] / PHYS["U_m_per_s"]                 # seconds per model time unit
    D = load_designs(eps)
    tep = {e: json.loads((OUT / f"tep_eps{e:g}.json").read_text()) for e in EPS_LIST
           if (OUT / f"tep_eps{e:g}.json").exists()}
    pois = json.loads((OUT / "pois_contrast.json").read_text()) if (OUT / "pois_contrast.json").exists() else None
    col = {"rising": core.OI_BLUE, "falling": core.OI_VERMILLION, "equal": core.OI_GREEN}
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=(6.3, 2.45), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 0.62], hspace=0.0)
    xlo, xhi = 0.75 * tu, 3.35 * tu
    axes_a = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_b = [fig.add_subplot(gs[i, 1], sharex=None) for i in range(3)]
    def stripes(ax, c, w, top):
        # stripe pattern: position in mm = time in s x U (plug flow); bar height ~ sqrt(k_j / k_max)
        for cj, wj in zip(c, w):
            ax.add_patch(matplotlib.patches.Rectangle((cj * tu - 0.25, -0.30 * top), 0.5,
                                                      0.26 * top * math.sqrt(wj / w.max()), color="0.25", lw=0))

    for i, kind in enumerate(PROFILES):
        c = np.asarray(D[kind]["c"])
        w = np.asarray(D[kind]["w"])
        ql = quad_law(c, w, B_BUDGET, eps, grad=False)
        f = ql.p / ql.meta["dt"] / tu                       # per second
        tt = fp = None
        if pois and f"{eps:g}" in pois["eps"]:
            cur = pois["eps"][f"{eps:g}"]["cells"][kind]["curve"]
            tt = np.asarray(cur["t"]) * tu
            fp = np.asarray(cur["f_h"]) / tu
        top = max(float(f.max()), float(fp.max()) if fp is not None else 0.0)
        ax = axes_a[i]
        ax.fill_between(ql.t * tu, f, color=col[kind], alpha=0.35, lw=0)
        ax.plot(ql.t * tu, f, color=col[kind], lw=0.9)
        for tj in T_TARGET:
            ax.axvline(tj * tu, color="0.35", lw=0.6, ls=":", zorder=0)
        stripes(ax, c, w, top)
        ax.set_ylim(-0.32 * top, 1.08 * top)
        ax.set_xlim(xlo, xhi)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.text(0.995, 0.9, kind, transform=ax.transAxes, va="top", ha="right", color=col[kind], fontsize=9)
        if i < 2:
            ax.set_xticklabels([])
        axp = axes_b[i]
        if fp is not None:
            axp.fill_between(tt, fp, color=col[kind], alpha=0.35, lw=0)
            axp.plot(tt, fp, color=col[kind], lw=0.9)
        stripes(axp, c, w, top)
        axp.set_ylim(-0.32 * top, 1.08 * top)
        for tj in T_TARGET:
            axp.axvline(tj * tu, color="0.35", lw=0.6, ls=":", zorder=0)
        axp.set_xlim(0.5 * tu, 3.9 * tu)
        axp.set_yticks([])
        axp.spines[["left", "right", "top"]].set_visible(False)
        if i < 2:
            axp.set_xticklabels([])
    axes_b[0].set_title(r"(b) Poiseuille, same pattern", fontsize=9, loc="left")
    axes_a[-1].set_xlabel("reaction time (s)")
    axes_b[-1].set_xlabel("reaction time (s)")
    axes_a[1].set_ylabel("density", fontsize=9)
    pu = phys_units(eps)
    axes_a[0].set_title(f"(a) plug flow (EOF), Pe$\\,={pu['peclet_UL_over_D']:.0f}$", fontsize=9, loc="left")
    # (c) achieved vs target ratios under the TEP (design dt, direct kill)
    axc = fig.add_subplot(gs[:, 2])
    off = {"rising": -0.012, "falling": 0.012, "equal": 0.0}
    for e, T_ in tep.items():
        for kind in PROFILES:
            ch = T_["cells"][kind].get("target_checks")
            if not ch:
                continue
            tv = np.asarray(ch["ratios"]["target"])
            dev = np.asarray(ch["ratios"]["design_dt"]["deviation"])
            te = np.asarray(ch["ratios"]["design_dt"]["allowed"])
            filled = abs(e - 0.03) < 1e-12
            xo = off[kind] + (-0.003 if filled else 0.003)
            axc.errorbar(tv + xo, tv + dev, yerr=te, fmt="o" if filled else "s", ms=3.0, color=col[kind],
                         mfc=col[kind] if filled else "white", mec=col[kind], elinewidth=0.6, capsize=0, lw=0)
    axc.plot([0, 0.36], [0, 0.36], color="0.5", lw=0.6, zorder=0)
    axc.set_xlim(0, 0.36)
    axc.set_ylim(0, 0.36)
    axc.set_xlabel("target ratio $r_j$")
    axc.set_ylabel("achieved $r_j$ (direct kill)")
    axc.set_title("(c) on target", fontsize=9, loc="left")
    devs = [np.max(np.abs(np.asarray(T_["cells"][k]["target_checks"]["peaks"][cv]["deviation"])))
            for T_ in tep.values() for k in PROFILES for cv in ("design_dt", "continuum")
            if T_["cells"][k].get("target_checks")]
    npk = sum(M for T_ in tep.values() for k in PROFILES if T_["cells"][k].get("target_checks"))
    peak_txt = f"\n{npk}/{npk} peaks\nwithin {math.ceil(max(devs) * tu * 100) / 100:.2f} s" if devs else ""
    axc.text(0.97, 0.03, r"$\varepsilon=0.03$ (filled)" + "\n" + r"$\varepsilon=0.01$ (open)" + peak_txt,
             transform=axc.transAxes, va="bottom", ha="right", fontsize=7)
    axc.text(0.03, 0.97, "idealised model\nchannel;\nno experiment", transform=axc.transAxes, ha="left", va="top",
             fontsize=7, style="italic", color="0.3")
    outs = []
    for suffix in (".pdf", ".png"):
        fig.savefig(FIG_STEM.with_suffix(suffix), dpi=300)
        outs.append(str(FIG_STEM.with_suffix(suffix)))
    plt.close(fig)
    print("[figure]", outs, flush=True)
    return outs


# ----------------------------------------------------------------------------
# summary: key numbers of all hero outputs (-> V2_hero/summary.json) + LaTeX table rows
# ----------------------------------------------------------------------------


def _mx(a) -> float:
    return float(np.max(np.abs(np.asarray(a, float))))


def cmd_summary(args) -> dict:
    tu = PHYS["L_m"] / PHYS["U_m_per_s"]
    S = {"item": "V2 hero summary", "time_unit_s": tu, "eps": {}}
    rows_design, rows_tep = [], []
    for eps in EPS_LIST:
        e = f"{eps:g}"
        dz = json.loads((OUT / f"design_eps{e}.json").read_text())
        rec = {"phys": dz["phys"], "cells": {}}
        tp = OUT / f"tep_eps{e}.json"
        tj = json.loads(tp.read_text()) if tp.exists() else None
        ojs = {rep: json.loads((OUT / f"oos_eps{e}_rep{rep}.json").read_text()) for rep in (1, 2)
               if (OUT / f"oos_eps{e}_rep{rep}.json").exists()}
        for kind in PROFILES:
            c = dz["cells"][kind]
            s_, q_, nv = c["saa"], c["quad"], c["naive"]
            ex = s_["exact_eval_of_saa_design"]
            r = {"c_mm": np.asarray(s_["c"]) * PHYS["L_m"] * 1e3, "w": s_["w"],
                 "k_um_per_s": np.asarray(s_["w"]) * B_BUDGET * PHYS["U_m_per_s"] * 1e6,
                 "lambda": np.asarray(s_["w"]) * B_BUDGET,
                 "naive_max_dT_s": _mx(nv["dev_peaks"]) * tu, "naive_max_dr": _mx(nv["dev_ratios"]),
                 "quad_residuals": [t["res_inf"] for t in q_["residual_table"]],
                 "saa_residuals": [t["res_inf"] for t in s_["residual_table"]],
                 "saa_q": [t["ratio_to_prev_sq"] for t in s_["residual_table"]],
                 "cond_J": s_["cond_J"], "max_z_x_minus_xinf": _mx(s_["x_minus_xinf_z"]),
                 "exact_max_dT_s": _mx(ex["dev_peaks"]) * tu, "exact_max_dr": _mx(ex["dev_ratios"]),
                 "exact_raw_minus_design_peak_s": _mx(np.asarray(ex["raw_peaks"]) - np.asarray(ex["peaks"])) * tu,
                 "yield_exact": ex["yield"], "exactly_5": ex["mode_check"]["exactly_m"],
                 "valley_to_peak_max": max(ex["valley_to_peak"])}
            if tj and "target_checks" in tj["cells"][kind]:
                cc = tj["cells"][kind]
                tc = cc["target_checks"]
                qr = cc["quad_richardson"]
                r["tep"] = {
                    "peaks_dev_dt_s": np.asarray(tc["peaks"]["design_dt"]["deviation"]) * tu,
                    "peaks_TE_dt_s": np.asarray(cc["richardson"]["peaks"]["levels"]["1"]["total_error"]) * tu,
                    "peaks_dev_cont_s": np.asarray(tc["peaks"]["continuum"]["deviation"]) * tu,
                    "peaks_TE_cont_s": np.asarray(cc["richardson"]["peaks"]["total_error_continuum"]) * tu,
                    "ratios_dev_dt": tc["ratios"]["design_dt"]["deviation"],
                    "ratios_TE_dt": tc["ratios"]["design_dt"]["allowed"],
                    "ratios_dev_cont": tc["ratios"]["continuum"]["deviation"],
                    "ratios_TE_cont": tc["ratios"]["continuum"]["allowed"],
                    "max_abs_peak_dev_s": max(_mx(tc["peaks"]["design_dt"]["deviation"]),
                                              _mx(tc["peaks"]["continuum"]["deviation"])) * tu,
                    "max_ratio_dev_over_TE": max(_mx(np.asarray(tc["ratios"]["design_dt"]["deviation"]) /
                                                     np.asarray(tc["ratios"]["design_dt"]["allowed"])),
                                                 _mx(np.asarray(tc["ratios"]["continuum"]["deviation"]) /
                                                     np.asarray(tc["ratios"]["continuum"]["allowed"]))),
                    "all_pass": cc["all_pass"], "quad_z": cc["quad_z_summary"],
                    "exact_dt_bias_peaks_s": _mx(qr["peaks"]["bias_design_dt"]) * tu,
                    "exact_dt_bias_ratios": _mx(qr["ratios"]["bias_design_dt"]),
                    "smoothing_delta_max_s": max(_mx(cc["levels"][L]["smoothing_delta"]) for L in cc["levels"]) * tu,
                    "dk_peak_se_level1_s": np.asarray(cc["levels"]["1"]["dk"]["se"]["peaks"]) * tu}
            for rep, oj in ojs.items():
                o = oj["cells"][kind]
                dsg = json.loads((OUT / "designs" / f"hero_m5_eps{e}_B{B_BUDGET:g}_{kind}.json").read_text())
                qp = dsg["quad_prediction"]
                zxp = (np.asarray(o["peaks"]) - np.asarray(qp["peaks"])) / np.asarray(o["se"]["peaks"])
                zxr = (np.asarray(o["ratios"]) - np.asarray(qp["ratios"])) / np.asarray(o["se"]["ratios"])
                r.setdefault("oos", {})[f"rep{rep}"] = {"n_paths": oj["n_paths"], "z_vs_exact_law_peaks": zxp, "z_vs_exact_law_ratios": zxr,
                            "max_dT_s": _mx(o["dev_peaks"]) * tu, "max_z_peaks": _mx(o["z_peaks"]),
                            "max_dr": _mx(o["dev_ratios"]), "max_z_ratios": _mx(o["z_ratios"]),
                            "exactly_5": o["mode_check"]["exactly_m"]}
            rec["cells"][kind] = r
            r["offset_um"] = (np.asarray(s_["c"]) - U * np.asarray(T_TARGET)) * PHYS["L_m"] * 1e6
            rows_design.append(f"{e} & {kind} & " + " & ".join(f"{x:.1f}" for x in r["offset_um"]) + " & " +
                               " & ".join(f"{x:.1f}" for x in r["k_um_per_s"]) + r" \\")
            if "tep" in r:
                t = r["tep"]
                rows_tep.append(f"{e} & {kind} & {r['naive_max_dT_s']:.2f} & {_mx(t['peaks_dev_dt_s']):.3f} & "
                                f"{np.max(t['peaks_TE_dt_s']):.3f} & {_mx(t['peaks_dev_cont_s']):.3f} & "
                                f"{_mx(t['ratios_dev_dt']):.4f} & {t['max_ratio_dev_over_TE']:.2f} & "
                                f"{'yes' if t['all_pass'] else 'no'}" + r" \\")
        if tj:
            rec["G2_all_pass"] = tj.get("G2_all_pass")
            pool = {}
            for q in ("peaks", "ratios"):
                z = np.concatenate([np.ravel(tj["cells"][k]["levels"][L]["quad_z"][q]) for k in PROFILES
                                    for L in tj["cells"][k]["levels"]])
                pool[q] = {"n": int(z.size), "rms": float(np.sqrt(np.mean(z ** 2))), "max_abs": float(np.max(np.abs(z)))}
            rec["dk_vs_quad_pooled"] = pool
        S["eps"][e] = rec
    pj = OUT / "pois_contrast.json"
    if pj.exists():
        P = json.loads(pj.read_text())
        S["pois"] = {e: {k: {"n_modes": v["modes"]["n_modes"], "mode_times_s": np.asarray(v["modes"]["mode_times"]) * tu,
                             "mode_over_target": (np.asarray(v["modes"]["mode_times"]) / np.asarray(T_TARGET)).tolist()
                             if v["modes"]["n_modes"] == M else None,
                             "n_modes_5pct": v["modes"].get("by_prominence", {}).get("0.05", {}).get("n_modes"),
                             "centreline_s": np.asarray(v["centreline_arrival_times"]) * tu,
                             "cells": v["cell_masses"], "mixture": v["mixture_masses"],
                             "max_z_cells_vs_mixture": _mx(v["cell_minus_mixture_z"]),
                             "max_abs_cells_minus_mixture": _mx(np.asarray(v["cell_masses"]) - np.asarray(v["mixture_masses"])),
                             "max_abs_cells_minus_plug": _mx(np.asarray(v["cell_masses"]) - np.asarray(v["plug_design_masses"]))}
                         for k, v in r_["cells"].items()} for e, r_ in P["eps"].items()}
    S["latex_rows_design"] = rows_design
    S["latex_rows_tep"] = rows_tep
    de.write_json(OUT / "summary.json", S)
    print("\n".join(rows_design))
    print("\n".join(rows_tep))
    return S


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {k[4:].replace("_", "-"): v for k, v in globals().items() if k.startswith("cmd_") and callable(v)}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--eps", default="0.03")
        p.add_argument("--n", default="1e5")
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--replicate", type=int, default=0)
        p.add_argument("--chunk", default="25000")
        p.add_argument("--levels", default="1,2,4")
        p.add_argument("--start", default="prop1")
        p.add_argument("--no-wait", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_wait:
        print(f"[cpu] {de.wait_for_cpu()}", flush=True)
    cmds[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
