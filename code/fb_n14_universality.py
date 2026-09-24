#!/usr/bin/env python3
"""N14 -- universality demos for the fixed-budget stick-breaking law (numerics for TH-13).

Question
--------
Theorem 2 / Prop. 1 are proved for the harmonic-trap pair model.  Do the
fixed-budget stick-breaking masses

    M_j = exp(-sum_{i<j} lambda_i) (1 - exp(-lambda_j)),   lambda_j = B w_j / (A v_j),

(v_j = normal crossing speed of the deterministic path at stripe j, A = transverse
normalisation) and the inverse-design law of Prop. 1 hold for OTHER transport
mechanisms?  Three demos, each a single reactant with a static catalyst field of
total integrated reactivity B split over m thin stripes of width eps*rho:

  plug   2D channel of width A = 1, reflecting walls, plug flow U = 1, stripes at
         x = (1, 3, 6), diffusion
         D = eps^2 D0 (D0 = 1).  Stripes span the channel, so y is irrelevant and the
         longitudinal coordinate is an exact drift-diffusion X_t = U t + eps sqrt(2 D0) W_t.
         Stripe profiles: Gaussian (sd eps*rho) and top-hat matched in mass and sd
         (half-width sqrt(3) eps*rho).  Limit: lambda_j = B w_j / (A U).
  pois   same channel with Poiseuille flow u(y) = 1.5 U (1 - (2y-1)^2) (mean U over the
         section), particles injected uniformly in the band y0 in [0.15, 0.85]
         (u in [0.765, 1.5]); y diffuses with the same D and reflects at y = 0, 1.
         Stripes at x = (0.5, 1.25, 3.125): ratio 2.5 > u_max/u_min = 1.96, so the limiting
         crossing-time windows [x_j/u_max, x_j/u_min] do not overlap.
         Limit (frozen transverse position): M_j = G(K_{j-1}) - G(K_j),
         K_j = B sum_{i<=j} w_i / A,  G(K) = E exp(-K / u(Y0))  (Laplace transform of the
         slowness 1/u); adapted design K_j = G^{-1}(1 - sum_{i<=j} p_i).
  nonlin 1D gradient flow dZ = -(Z + Z^3) dt + eps sqrt(2 D0) dW  (V = z^2/2 + z^4/4),
         Z0 = 3; Gaussian stripes at z = (2.4, 1.6, 0.8); v_j = |b(z_j)| = z_j + z_j^3.
         Naive comparison: the harmonic (linearised, gamma = -b'(0) = 1) speeds v_j = z_j.

Kill rate (all demos): kappa(x) = (B / A) sum_j w_j phi_j(x), phi_j a unit-mass stripe
profile in the normal coordinate; A = 1 (channel width; 1 for the 1D demo).

Method (exact discrete law, Feynman--Kac reweighting)
----------------------------------------------------
Unkilled Euler--Maruyama paths with step dt_n; a walker surviving to the end of step
n is killed with probability 1 - exp(-B w.e_n), e_n^{(j)} = phi_j(x_n) dt_n / A
(end-of-step rule, as in the production simulator).  For that discrete-time law
P(kill in steps (a, b]) = E[ exp(-B w.X_a) (1 - exp(-B w.(X_b - X_a))) ] exactly, so a
single path ensemble serves every (B, w) and every stripe profile (common random
numbers).  Stored per path: the unit-budget stripe exposures X_j at the basin cut
times and at T, and at the first passage of the midpoints between stripes
("position cells"); for Poiseuille also y0 and y at the first passage of each stripe centre
(per-path "local-speed law" lambda_j = B w_j / (A u(y_cross_j))).  Accumulated online, for a
declared list of (profile, m, B, w)
cells: per-group bin masses on an output grid (every OUT_EVERY steps), giving the
exact-law density with batch-means SEs (20 groups of 1e4 paths).
Analytic first-order check (plug, nonlin): Var X_j = 2 D0 eps int psi^2 / (rho A^2 v_j^3)
(psi = stripe profile in slab units) and S_j = exp(-Lambda_j)(1 + V_j/2).
Time steps: channel dt = eps/40 (a stripe sd is crossed in >= 26 steps);
nonlin dt_n = (eps/40) / max(|b(phi(t_n))|, 0.3) (deterministic, speed-adapted; the
path moves ~eps/40 per step).  dt check: dt/2 at eps = 0.05 (1e5 paths).

Seeds: base 20260923, tag 97; chunk c of ensemble (demo, eps, dt_factor) uses child c
of SeedSequence([20260923, 97, demo_code, round(eps*1e9), round(dt_factor*1e6),
n_paths]).spawn(n_chunks), Philox generator.  Recorded in every output.

Subcommands
-----------
  simulate  run/resume the chunk tasks (<= 3 worker processes, deadline-bounded)
  chunk     one chunk (internal; used by simulate)
  analyze   -> artifacts/data/exact_m_fixed_budget/N14_universality/n14_universality.json
  figure    -> artifacts/figures/fb_n14_universality.{pdf,png}
  law       print the deterministic predictions (no simulation)
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import fb_allocation_law as alloc  # noqa: E402

REPORT = core.REPORT
OUT_DIR = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "N14_universality"
OUT_JSON = OUT_DIR / "n14_universality.json"
FIG_STEM = core.FIGURES / "fb_n14_universality"
DEFAULT_CACHE = Path(
    "<scratch>/"
    "cnsns_general/n14_cache")

BASE_SEED = 20260923
TAG = 97
DEMO_CODE = {"plug": 1, "pois": 2, "nonlin": 3}

D0 = 1.0
RHO = 1.0
U = 1.0
AREA = 1.0                      # A: channel width (transverse normalisation)
EPS_LIST = (0.1, 0.05, 0.025)
B_LIST = (1.0, 4.0, 8.0)
N_PATHS = 200_000
CHUNK = 50_000
GROUPS_PER_CHUNK = 5
OUT_EVERY = 4
DT_PER_EPS = 1.0 / 40.0
DTCHECK_EPS = 0.05
DTCHECK_PATHS = 100_000

# channel geometry (per flow profile)
CH_X = {"plug": (1.0, 3.0, 6.0), "pois": (0.5, 1.25, 3.125)}   # stripe positions
CH_T = {"plug": 8.0, "pois": 5.25}
CH_CUTS = {"plug": (2.0, 4.5), "pois": (0.74, 1.84)}           # time-basin cuts
CH_MIDS = {"plug": (2.0, 4.5), "pois": (0.875, 2.1875)}        # position cells
POIS_Y0 = (0.15, 0.85)          # injection band: u in [0.765, 1.5], ratio 1.96 < x_{j+1}/x_j
TOPHAT_HALF = math.sqrt(3.0)    # half-width in units of eps*rho (sd-matched)

# nonlinear relaxation geometry
NL_Z0 = 3.0
NL_Z = (2.4, 1.6, 0.8)
NL_MIDS = (2.0, 1.2)
NL_T = 1.5
NL_VFLOOR = 0.3

SUBSETS = {3: (0, 1, 2), 2: (0, 2)}     # m = 2 uses the first and last stripe
CUT_INDEX = {3: (0, 1), 2: (1,)}         # which stored cuts delimit the basins
PY_ACTIVE_LIMIT = 9


# ============================================================================
# Deterministic geometry and limit laws
# ============================================================================


def u_pois(y):
    return 1.5 * U * (1.0 - (2.0 * np.asarray(y) - 1.0) ** 2)


def b_nl(z):
    return -(z + z ** 3)


def phi_nl(t):
    """Exact deterministic path of dz/dt = -(z + z^3), z(0) = NL_Z0."""
    return 1.0 / np.sqrt((1.0 + 1.0 / NL_Z0 ** 2) * np.exp(2.0 * np.asarray(t)) - 1.0)


def t_nl(z):
    """Hitting time of level z < z0 by the deterministic path."""
    return 0.5 * math.log((1.0 + 1.0 / z ** 2) / (1.0 + 1.0 / NL_Z0 ** 2))


def nl_offset_var(z):
    """Linearised offset variance at level z (start deterministic):
    sigma^2(t) = 2 D0 b(z_t)^2 int_{z_t}^{z0} dz / |b(z)|^3  (closed form of the
    variance ODE d s2/dt = 2 b'(phi) s2 + 2 D0)."""
    zz = np.linspace(z, NL_Z0, 20001)
    f = 1.0 / np.abs(b_nl(zz)) ** 3
    integral = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(zz)))
    return 2.0 * D0 * b_nl(z) ** 2 * integral


def geometry(demo: str) -> dict:
    windows = None
    if demo in ("plug", "pois"):
        centres = CH_X[demo]
        cuts = CH_CUTS[demo]
        mids = CH_MIDS[demo]
        T = CH_T[demo]
        if demo == "plug":
            speeds = [U] * 3
            times = [x / U for x in centres]
            offset_var = [2.0 * D0 * t for t in times]
        else:
            speeds = None     # random (transverse position)
            times = None
            offset_var = None
            umin, umax = float(u_pois(POIS_Y0[0])), float(u_pois(0.5))
            windows = [[x / umax, x / umin] for x in centres]
        shapes = ("gauss", "tophat")
    else:
        centres = NL_Z
        times = [t_nl(z) for z in NL_Z]
        cuts = tuple(t_nl(z) for z in NL_MIDS)
        mids = NL_MIDS
        T = NL_T
        speeds = [abs(float(b_nl(z))) for z in NL_Z]
        offset_var = [nl_offset_var(z) for z in NL_Z]
        shapes = ("gauss",)
    return {"demo": demo, "centres": list(centres), "cuts": list(cuts), "mids": list(mids),
            "T": T, "speeds": speeds, "cross_times": times, "offset_var": offset_var,
            "shapes": list(shapes), "area": AREA, "rho": RHO, "D0": D0,
            "pois_crossing_windows": windows}


# -- Poiseuille averaged law -------------------------------------------------

_GL = {}


def pois_slowness_nodes(n: int = 200):
    if n not in _GL:
        x, wq = np.polynomial.legendre.leggauss(n)
        a, b = POIS_Y0
        y = 0.5 * (a + b) + 0.5 * (b - a) * x
        _GL[n] = (1.0 / u_pois(y), 0.5 * wq)      # uniform law on [a, b]
    return _GL[n]


def G_pois(K: float) -> float:
    s, wq = pois_slowness_nodes()
    return float(np.sum(wq * np.exp(-K * s / AREA)))


def G_pois_inv(g: float) -> float:
    if not 0.0 < g <= 1.0:
        raise ValueError("G^{-1} needs 0 < g <= 1")
    lo, hi = 0.0, 1.0
    while G_pois(hi) > g:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if G_pois(mid) > g:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pois_law(B: float, w) -> list[float]:
    K = np.concatenate([[0.0], B * np.cumsum(w)])
    return [G_pois(K[j]) - G_pois(K[j + 1]) for j in range(len(w))]


def pois_design(B: float, m: int) -> dict:
    """Adapted Prop. 1 for a random common slowness: equal masses p* = (1 - G(B))/m,
    cumulative budgets K_j = G^{-1}(1 - j p*), w_j = (K_j - K_{j-1}) / B."""
    p = (1.0 - G_pois(B)) / m
    K = [0.0] + [G_pois_inv(1.0 - j * p) for j in range(1, m)] + [B]
    w = [(K[j + 1] - K[j]) / B for j in range(m)]
    return {"p_star": p, "weights": w, "K": K}


def pois_mean_speed() -> float:
    s, wq = pois_slowness_nodes()
    return float(np.sum(wq / s))


def pois_harmonic_speed() -> float:
    s, wq = pois_slowness_nodes()
    return float(1.0 / np.sum(wq * s))


# -- declared cells ------------------------------------------------------------


def law_masses(demo: str, B: float, w, idx) -> list[float]:
    g = geometry(demo)
    if demo == "pois":
        return pois_law(B, w)
    v = [g["speeds"][i] for i in idx]
    return alloc.masses(B, w, v, AREA)


def naive_speeds(demo: str, idx):
    if demo == "pois":
        return [pois_mean_speed()] * len(idx)
    if demo == "nonlin":
        return [NL_Z[i] for i in idx]        # harmonic speeds gamma |z_j|, gamma = 1
    return None


def declared_cells(demo: str) -> list[dict]:
    g = geometry(demo)
    cells = []
    for shape in g["shapes"]:
        for m in (2, 3):
            idx = SUBSETS[m]
            for B in B_LIST:
                allocs = {"equal": [1.0 / m] * m}
                if demo == "pois":
                    allocs["design"] = pois_design(B, m)["weights"]
                else:
                    v = [g["speeds"][i] for i in idx]
                    allocs["design"] = alloc.p_star(B, v, AREA)["weights"]
                nv = naive_speeds(demo, idx)
                if nv is not None:
                    allocs["naive_design"] = alloc.p_star(B, nv, AREA)["weights"]
                for name, w in allocs.items():
                    if demo == "pois" and shape != "gauss":
                        continue
                    cells.append({"shape": shape, "m": m, "idx": list(idx), "B": B,
                                  "alloc": name, "w": [float(x) for x in w]})
    return cells


def component_index(demo: str, shape: str, stripe: int) -> int:
    g = geometry(demo)
    return g["shapes"].index(shape) * len(g["centres"]) + stripe


# ============================================================================
# Simulation (one chunk)
# ============================================================================


def time_grid(demo: str, eps: float, dt_factor: float) -> np.ndarray:
    c = eps * DT_PER_EPS * dt_factor
    if demo in ("plug", "pois"):
        n = int(math.ceil(CH_T[demo] / c - 1e-9))
        dts = np.full(n, c)
    else:
        out, t = [], 0.0
        while t < NL_T - 1e-12:
            v = max(abs(float(b_nl(phi_nl(t)))), NL_VFLOOR)
            out.append(c / v)
            t += c / v
        dts = np.array(out)
    pad = (-len(dts)) % OUT_EVERY
    if pad:
        dts = np.concatenate([dts, np.full(pad, dts[-1])])
    return dts


def chunk_path(cache: Path, demo: str, eps: float, dt_factor: float, n_total: int, c: int) -> Path:
    return cache / f"{demo}_e{eps:g}_dt{dt_factor:g}_N{n_total}_c{c:02d}.npz"


def seed_entropy(demo: str, eps: float, dt_factor: float, n_total: int) -> list[int]:
    return [BASE_SEED, TAG, DEMO_CODE[demo], int(round(eps * 1e9)),
            int(round(dt_factor * 1e6)), int(n_total)]


def run_chunk(demo: str, eps: float, dt_factor: float, n_total: int, c: int,
              cache: Path) -> dict:
    t0 = time.time()
    n_chunks = n_total // CHUNK
    ent = seed_entropy(demo, eps, dt_factor, n_total)
    ss = np.random.SeedSequence(ent).spawn(n_chunks)[c]
    rng = np.random.Generator(np.random.Philox(ss))
    g = geometry(demo)
    centres = np.array(g["centres"])
    n_str = len(centres)
    shapes = g["shapes"]
    ncomp = n_str * len(shapes)
    cells = declared_cells(demo)
    ncell = len(cells)
    Wmat = np.zeros((ncomp, ncell))
    for k, cell in enumerate(cells):
        for wj, sj in zip(cell["w"], cell["idx"]):
            # X (unit-budget exposure) already carries 1/A (see the accumulation below);
            # fixer 2026-09-23: a second 1/A removed here (AREA = 1.0, no stored number changes)
            Wmat[component_index(demo, cell["shape"], sj), k] = cell["B"] * wj

    dts = time_grid(demo, eps, dt_factor)
    times = np.cumsum(dts)
    nsteps = len(dts)
    ck_times = list(g["cuts"]) + [g["T"]]
    ck_steps = [int(np.searchsorted(times, tc - 1e-12)) for tc in ck_times]   # 0-based step idx
    ck_steps[-1] = nsteps - 1
    ck_lookup = {s: i for i, s in enumerate(ck_steps)}
    n_out = nsteps // OUT_EVERY
    out_edges = np.concatenate([[0.0], times[OUT_EVERY - 1::OUT_EVERY]])

    N = CHUNK
    G = GROUPS_PER_CHUNK
    sw = eps * RHO
    gnorm = 1.0 / (math.sqrt(2.0 * math.pi) * sw)
    inv2 = 1.0 / (2.0 * sw * sw)
    th_half = TOPHAT_HALF * sw
    th_h = 1.0 / (2.0 * th_half)

    if demo == "nonlin":
        x = np.full(N, NL_Z0)
    else:
        x = np.zeros(N)
    if demo == "pois":
        y = rng.uniform(POIS_Y0[0], POIS_Y0[1], N)
        y0 = y.copy()
        ycross = np.full((N, n_str), np.nan)     # y at first passage of stripe centre j
    X = np.zeros((N, ncomp))
    ck = np.zeros((N, len(ck_steps), ncomp))
    mids = np.array(g["mids"])
    Xmid = np.zeros((N, len(mids), ncomp))
    crossed = np.zeros((N, len(mids)), dtype=bool)
    Yprev = np.zeros((N, ncell))
    Sprev = np.ones((N, ncell))
    dens = np.zeros((G, ncell, n_out))
    xmax_dev = 0.0
    a = 0
    for n in range(nsteps):
        dt = dts[n]
        sq = eps * math.sqrt(2.0 * D0 * dt)
        if demo == "plug":
            x += U * dt + sq * rng.standard_normal(N)
        elif demo == "pois":
            nz = rng.standard_normal((2, N))
            x += u_pois(y) * dt + sq * nz[0]
            y += sq * nz[1]
            y = np.abs(y)
            y = 1.0 - np.abs(1.0 - y)
        else:
            x += b_nl(x) * dt + sq * rng.standard_normal(N)
        for j in range(n_str):
            d = x - centres[j]
            for si, shape in enumerate(shapes):
                col = si * n_str + j
                if shape == "gauss":
                    X[:, col] += dt * gnorm * np.exp(-d * d * inv2) / AREA
                else:
                    X[:, col] += dt * th_h * (np.abs(d) <= th_half) / AREA
        if demo == "pois":
            for j in range(n_str):
                newj = np.isnan(ycross[:, j]) & (x >= centres[j])
                if newj.any():
                    ycross[newj, j] = y[newj]
        for k, xm in enumerate(mids):
            if demo == "nonlin":
                newly = (~crossed[:, k]) & (x <= xm)
            else:
                newly = (~crossed[:, k]) & (x >= xm)
            if newly.any():
                Xmid[newly, k, :] = X[newly]
                crossed[newly, k] = True
        if n in ck_lookup:
            ck[:, ck_lookup[n], :] = X
        if (n + 1) % OUT_EVERY == 0:
            with np.errstate(all="ignore"):     # Accelerate matmul raises spurious FP flags
                Ynew = X @ Wmat
            dY = Ynew - Yprev
            mass = Sprev * (-np.expm1(-dY))
            dens[:, :, a] = mass.reshape(G, N // G, ncell).sum(axis=1)
            Sprev *= np.exp(-dY)
            Yprev = Ynew
            a += 1
    if not (np.isfinite(dens).all() and np.isfinite(ck).all() and np.isfinite(Xmid).all()):
        raise FloatingPointError("non-finite exposure or density")
    # paths that never reached a midpoint by T: their cell boundary is T
    for k in range(len(mids)):
        Xmid[~crossed[:, k], k, :] = X[~crossed[:, k]]
    out = chunk_path(cache, demo, eps, dt_factor, n_total, c)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"demo": demo, "eps": eps, "dt_factor": dt_factor, "n_total": n_total, "chunk": c,
            "n_chunks": n_chunks, "N": N, "groups": G, "entropy": ent, "spawn_child": c,
            "nsteps": nsteps, "dt_min": float(dts.min()), "dt_max": float(dts.max()),
            "ck_times_nominal": ck_times, "ck_times_actual": [float(times[s]) for s in ck_steps],
            "T_actual": float(times[-1]), "cells": cells, "wall_s": time.time() - t0,
            "crossed_frac": crossed.mean(axis=0).tolist(),
            "geometry": g}
    tmp = out.with_suffix(".tmp.npz")
    extra = {"y0": y0, "ycross": ycross} if demo == "pois" else {}
    np.savez(tmp, ck=ck, Xmid=Xmid, crossed=crossed, dens=dens, out_edges=out_edges,
             meta=np.array(json.dumps(meta)), **extra)
    os.replace(tmp, out)
    return meta


# ============================================================================
# Analysis
# ============================================================================


def load_ensemble(cache: Path, demo: str, eps: float, dt_factor: float, n_total: int) -> dict:
    parts = [chunk_path(cache, demo, eps, dt_factor, n_total, c) for c in range(n_total // CHUNK)]
    missing = [str(p) for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} chunk(s) missing, e.g. {missing[0]}")
    ck, Xmid, crossed, dens, metas, y0, ycross = [], [], [], [], [], [], []
    for p in parts:
        d = np.load(p)
        if "ycross" in d.files:
            y0.append(d["y0"])
            ycross.append(d["ycross"])
        ck.append(d["ck"])
        Xmid.append(d["Xmid"])
        crossed.append(d["crossed"])
        dens.append(d["dens"])
        edges = d["out_edges"]
        metas.append(json.loads(str(d["meta"])))
    return {"ck": np.concatenate(ck), "Xmid": np.concatenate(Xmid),
            "crossed": np.concatenate(crossed), "dens": np.concatenate(dens),
            "edges": edges, "metas": metas, "N": sum(m["N"] for m in metas),
            "groups": sum(m["groups"] for m in metas),
            "y0": np.concatenate(y0) if y0 else None,
            "ycross": np.concatenate(ycross) if ycross else None}


def unit_exposure(ens: dict, demo: str, shape: str, idx, w, B: float, which: str) -> np.ndarray:
    arr = ens["ck"] if which == "ck" else ens["Xmid"]
    Y = np.zeros(arr.shape[:2])
    for wj, sj in zip(w, idx):
        Y += (B * wj) * arr[:, :, component_index(demo, shape, sj)]  # arr already has 1/A
    return Y


def basin_stats(Yb: np.ndarray) -> dict:
    """Yb: (N, m+1) cumulative exposures at basin ends (last column = T), start 0."""
    N, k = Yb.shape
    Y0 = np.concatenate([np.zeros((N, 1)), Yb], axis=1)
    per = np.exp(-Y0[:, :-1]) * (-np.expm1(-(Y0[:, 1:] - Y0[:, :-1])))
    surv = np.exp(-Yb[:, -1])
    ess = per.sum(0) ** 2 / np.maximum((per ** 2).sum(0), 1e-300)     # Kish effective size
    return {"mass": per.mean(0).tolist(), "se": (per.std(0, ddof=1) / math.sqrt(N)).tolist(),
            "kish_ess": ess.tolist(),
            "surv_T": float(surv.mean()), "surv_T_se": float(surv.std(ddof=1) / math.sqrt(N))}


def cumulant_masses(Yb: np.ndarray) -> dict:
    """Mean-exposure law exp(-E Y) and its second-cumulant (Jensen) correction."""
    mu = Yb.mean(0)
    var = Yb.var(0, ddof=1)
    S_mf = np.concatenate([[1.0], np.exp(-mu)])
    S_j = np.concatenate([[1.0], np.exp(-mu + 0.5 * var)])
    return {"mean_exposure_law": (S_mf[:-1] - S_mf[1:]).tolist(),
            "jensen_law": (S_j[:-1] - S_j[1:]).tolist(),
            "mean_Y": mu.tolist(), "var_Y": var.tolist()}


def peak_census(edges: np.ndarray, gdens: np.ndarray, per_group: int, zthr: float) -> dict:
    """Derivative sign census of the exact-law density (batch-means SE over groups)."""
    width = np.diff(edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    fg = gdens / per_group / width               # (G, n)
    f = fg.mean(0)
    dfg = np.diff(fg, axis=1) / np.diff(centres)
    d = dfg.mean(0)
    se = dfg.std(0, ddof=1) / math.sqrt(fg.shape[0])
    z = np.where(se > 0, d / np.where(se > 0, se, 1.0), 0.0)
    sgn = np.where(z > zthr, 1, np.where(z < -zthr, -1, 0))
    tm = 0.5 * (centres[1:] + centres[:-1])
    nz = np.flatnonzero(sgn)
    maxima, minima = [], []
    for a, b in zip(nz[:-1], nz[1:]):
        if sgn[a] == 1 and sgn[b] == -1:
            maxima.append(float(0.5 * (tm[a] + tm[b])))
        elif sgn[a] == -1 and sgn[b] == 1:
            minima.append(float(0.5 * (tm[a] + tm[b])))
    return {"n_maxima": len(maxima), "n_minima": len(minima), "maxima_t": maxima,
            "minima_t": minima, "zthr": zthr, "f": f, "centres": centres}


# -- limiting local passage profile (TH-3 form, general offset variance) ----------

def passage_profile(lam: float, theta: float, shape: str = "gauss"):
    """F_{lam,theta} = p_lam * phi_theta on a grid (slab units rho); returns (y, F)."""
    du = 0.002
    L = 14.0 + 9.0 * theta
    u = np.arange(-L, L + du, du)
    if shape == "gauss":
        dens = np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(u / math.sqrt(2.0)))
    else:
        h = TOPHAT_HALF
        dens = np.where(np.abs(u) <= h, 1.0 / (2.0 * h), 0.0)
        cdf = np.clip((u + h) / (2.0 * h), 0.0, 1.0)
    p = lam * dens * np.exp(-lam * cdf)
    if theta <= 0:
        return u, p
    n = len(u)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    k = (np.arange(nfft) - nfft // 2) * du          # kernel centred at index nfft // 2
    ker = np.exp(-0.5 * (k / theta) ** 2) / (math.sqrt(2.0 * math.pi) * theta) * du
    ker = np.fft.ifftshift(ker)                       # centre -> index 0 (circular)
    F = np.fft.irfft(np.fft.rfft(p, nfft) * np.fft.rfft(ker), nfft)[:n]
    return u, F


def limit_density(demo: str, eps: float, B: float, w, idx, t: np.ndarray, shape: str = "gauss"):
    g = geometry(demo)
    v = [g["speeds"][i] for i in idx]
    lam = [B * wj / (AREA * vj) for wj, vj in zip(w, v)]
    f = np.zeros_like(t)
    peaks = []
    log_s = 0.0
    for lj, vj, sj in zip(lam, v, idx):
        theta = math.sqrt(g["offset_var"][sj]) / RHO
        yy, F = passage_profile(lj, theta, shape)
        tj = g["cross_times"][sj]
        arg = vj * (t - tj) / (eps * RHO)
        f += math.exp(-log_s) * (vj / (eps * RHO)) * np.interp(arg, yy, F, left=0.0, right=0.0)
        ystar = float(yy[int(np.argmax(F))])
        peaks.append({"t_j": tj, "lambda": lj, "theta": theta, "y_star": ystar,
                      "t_peak_limit": tj + eps * RHO * ystar / vj,
                      "height_limit": math.exp(-log_s) * vj / (eps * RHO) * float(F.max())})
        log_s += lj
    return f, peaks


def profile_check(demo: str, eps: float, cell: dict, cen: np.ndarray, f_fk: np.ndarray,
                  edges: np.ndarray, basins: list) -> dict:
    f_lim, peaks = limit_density(demo, eps, cell["B"], cell["w"], cell["idx"], cen, cell["shape"])
    width = np.diff(edges)
    out = []
    for k, (lo, hi) in enumerate(basins):
        msk = (cen > lo) & (cen <= hi)
        l1 = float(np.sum(np.abs(f_fk[msk] - f_lim[msk]) * width[msk]))
        mass_lim = float(np.sum(f_lim[msk] * width[msk]))
        i_fk = int(np.argmax(np.where(msk, f_fk, -1.0)))
        out.append({"rel_L1": l1 / mass_lim if mass_lim > 0 else None,
                    "t_peak_fk": float(cen[i_fk]), "height_fk": float(f_fk[i_fk]),
                    **peaks[k],
                    "height_ratio_fk_over_limit": float(f_fk[i_fk]) / peaks[k]["height_limit"]})
    return {"passages": out}


def analyze_ensemble(demo: str, eps: float, ens: dict, *, with_density: bool = True) -> list[dict]:
    g = geometry(demo)
    cells = declared_cells(demo)
    per_group = ens["N"] // ens["groups"]
    rows = []
    for k, cell in enumerate(cells):
        m, idx, w, B, shape = cell["m"], cell["idx"], cell["w"], cell["B"], cell["shape"]
        Yck = unit_exposure(ens, demo, shape, idx, w, B, "ck")
        Ymid = unit_exposure(ens, demo, shape, idx, w, B, "mid")
        cut_cols = list(CUT_INDEX[m])
        tb = basin_stats(np.concatenate([Yck[:, cut_cols], Yck[:, -1:]], axis=1))
        pc = basin_stats(np.concatenate([Ymid[:, cut_cols], Yck[:, -1:]], axis=1))
        law = law_masses(demo, B, w, idx)
        row = {"shape": shape, "m": m, "idx": idx, "B": B, "alloc": alloc_name(cell),
               "w": w, "law": law,
               "law_surv": float(1.0 - sum(law)),
               "fk_time_basins": tb, "fk_position_cells": pc,
               "cells_minus_basins_max": float(np.max(np.abs(np.array(tb["mass"]) -
                                                            np.array(pc["mass"]))))}
        nv = naive_speeds(demo, idx)
        if nv is not None:
            row["naive_speeds"] = nv
            row["naive_prediction"] = alloc.masses(B, w, nv, AREA)
        if cell["alloc"] == "design":
            row["target"] = (pois_design(B, m)["p_star"] if demo == "pois" else
                             alloc.p_star(B, [g["speeds"][i] for i in idx], AREA)["p_star"])
        if cell["alloc"] == "naive_design":
            row["naive_target"] = alloc.p_star(B, nv, AREA)["p_star"]
        if demo == "pois" and ens.get("ycross") is not None:
            # per-path stick-breaking with the transverse position AT each crossing
            # (tests lambda = kappa / local speed; isolates the frozen-y approximation)
            yc = ens["ycross"][:, idx]
            lam = np.where(np.isnan(yc), 0.0,
                           (B * np.asarray(w) / AREA) / u_pois(np.nan_to_num(yc, nan=0.5)))
            L0 = np.concatenate([np.zeros((lam.shape[0], 1)), np.cumsum(lam, axis=1)], axis=1)
            per = np.exp(-L0[:, :-1]) * (-np.expm1(-lam))
            row["local_speed_law"] = {"mass": per.mean(0).tolist(),
                                      "se": (per.std(0, ddof=1) / math.sqrt(len(per))).tolist()}
            lam0 = (B * np.asarray(w) / AREA)[None, :] / u_pois(ens["y0"])[:, None]
            L0 = np.concatenate([np.zeros((lam0.shape[0], 1)), np.cumsum(lam0, axis=1)], axis=1)
            per0 = np.exp(-L0[:, :-1]) * (-np.expm1(-lam0))
            row["frozen_y0_sampled_law"] = {"mass": per0.mean(0).tolist(),
                                            "se": (per0.std(0, ddof=1) / math.sqrt(len(per0))).tolist()}
        if demo != "pois":
            row["first_order_law"] = first_order_law(demo, eps, B, w, idx, shape)
            row["cumulant"] = cumulant_masses(np.concatenate([Yck[:, cut_cols], Yck[:, -1:]],
                                                             axis=1))
        if with_density:
            gd = ens["dens"][:, k, :]
            dens_total = float(gd.sum() / ens["N"])
            row["density_total_mass"] = dens_total
            row["density_total_mass_minus_basins"] = dens_total - float(sum(tb["mass"]))
            for zthr in (5.0, 3.0):
                pk = peak_census(ens["edges"], gd, per_group, zthr)
                row[f"peaks_z{zthr:g}"] = {kk: pk[kk] for kk in ("n_maxima", "n_minima",
                                                                "maxima_t", "minima_t", "zthr")}
            if demo != "pois":
                cen, f = pk["centres"], pk["f"]
                cuts = [0.0] + [g["cuts"][c] for c in cut_cols] + [g["T"]]
                basins = list(zip(cuts[:-1], cuts[1:]))
                row["profile_check"] = profile_check(demo, eps, cell, cen, f, ens["edges"],
                                                     basins)
        rows.append(row)
    return rows


def profile_l2(shape: str) -> float:
    """int g^2 for the unit-mass, unit-sd stripe profile g (slab units)."""
    return 1.0 / (2.0 * math.sqrt(math.pi)) if shape == "gauss" else 1.0 / (2.0 * TOPHAT_HALF)


def exposure_var_pred(eps: float, v: float, shape: str) -> float:
    """Leading exposure variance of one crossing at speed v (first order in the Brownian
    displacement during the O(eps) passage): Var X = 2 D0 eps int g^2 / (rho A^2 v^3)."""
    return 2.0 * D0 * eps * profile_l2(shape) / (RHO * AREA ** 2 * v ** 3)


def first_order_law(demo: str, eps: float, B: float, w, idx, shape: str) -> list[float]:
    """Limit law plus the exposure-fluctuation (Jensen) term:
    S_j = exp(-Lambda_j) (1 + V_j / 2),  V_j = sum_{i<=j} (B w_i)^2 Var X_i.
    X_i is the unit-budget exposure, which already contains the factor 1/A, so
    I_i = B w_i X_i and Var I_i = (B w_i)^2 Var X_i (TH13 Prop. th13:jensen(i) with
    a_i = 2 D0, beta_i = B w_i / A).  [Fixer 2026-09-23: an extra 1/A^2 was removed
    here; all runs use AREA = 1.0, so no stored number changes.]"""
    g = geometry(demo)
    v = [g["speeds"][i] for i in idx]
    lam = np.cumsum([B * wj / (AREA * vj) for wj, vj in zip(w, v)])
    V = np.cumsum([(B * wj) ** 2 * exposure_var_pred(eps, vj, shape)
                   for wj, vj in zip(w, v)])
    S = np.concatenate([[1.0], np.exp(-lam) * (1.0 + 0.5 * V)])
    return (S[:-1] - S[1:]).tolist()


def exposure_moments(demo: str, ens: dict, eps: float) -> dict:
    """Unit-budget exposure of each stripe accrued by T: mean and variance versus the limit."""
    g = geometry(demo)
    out = {}
    for shape in g["shapes"]:
        for j in range(len(g["centres"])):
            x = ens["ck"][:, -1, component_index(demo, shape, j)]
            row = {"mean": float(x.mean()), "mean_se": float(x.std(ddof=1) / math.sqrt(len(x))),
                   "var": float(x.var(ddof=1))}
            if demo == "pois":
                row["limit_frozen_E_1_over_u_Y0"] = float(np.mean(1.0 / u_pois(ens["y0"])))  \
                    if ens.get("y0") is not None else None
                if ens.get("ycross") is not None:
                    yc = ens["ycross"][:, j]
                    ok = ~np.isnan(yc)
                    row["E_1_over_u_Ycross"] = float(np.mean(1.0 / u_pois(yc[ok])))
                    row["crossed_frac"] = float(ok.mean())
                s_nodes, wq = pois_slowness_nodes()
                row["limit_frozen_quadrature"] = float(np.sum(wq * s_nodes)) / AREA
            else:
                row["limit_1_over_Av"] = 1.0 / (AREA * g["speeds"][j])
                row["var_pred"] = exposure_var_pred(eps, g["speeds"][j], shape)
                row["var_over_pred"] = row["var"] / row["var_pred"]
            out[f"{shape}|{j}"] = row
    return out


def alloc_name(cell: dict) -> str:
    return cell["alloc"]


def density_curve(demo: str, eps: float, ens: dict, sel: dict, n_keep: int = 1500) -> dict:
    cells = declared_cells(demo)
    k = next(i for i, c in enumerate(cells) if all(c[kk] == vv for kk, vv in sel.items()))
    per_group = ens["N"] // ens["groups"]
    pk = peak_census(ens["edges"], ens["dens"][:, k, :], per_group, 5.0)
    cen, f = pk["centres"], pk["f"]
    cell = cells[k]
    f_lim = None
    if demo != "pois":
        tt = np.linspace(cen.min(), cen.max(), 20001) if demo == "plug" else \
            np.exp(np.linspace(math.log(max(cen.min(), 1e-3)), math.log(cen.max()), 20001))
        fl, _ = limit_density(demo, eps, cell["B"], cell["w"], cell["idx"], tt, cell["shape"])
        f_lim = {"t": tt[::4].tolist(), "f": fl[::4].tolist()}
    step = max(1, len(cen) // n_keep)
    return {"cell": cell, "eps": eps, "t": cen[::step].tolist(), "f": f[::step].tolist(),
            "t_full_len": int(len(cen)), "limit": f_lim}


def convergence_table(results: dict) -> dict:
    """Per (demo, shape, m, B, alloc, basin): Delta(eps) = FK - law, local orders."""
    out = {}
    for demo, per_eps in results.items():
        keys = {}
        for eps_s, rows in per_eps.items():
            for r in rows:
                key = f"{r['shape']}|m{r['m']}|B{r['B']:g}|{r['alloc']}"
                keys.setdefault(key, {})[float(eps_s)] = r
        tab = {}
        for key, byeps in keys.items():
            eps_sorted = sorted(byeps, reverse=True)
            m = byeps[eps_sorted[0]]["m"]
            obs = "fk_position_cells" if demo == "pois" else "fk_time_basins"
            series = []
            for j in range(m):
                d = [byeps[e][obs]["mass"][j] - byeps[e]["law"][j] for e in eps_sorted]
                se = [byeps[e][obs]["se"][j] for e in eps_sorted]
                orders = []
                for a in range(len(d) - 1):
                    sig = abs(d[a]) > 3 * se[a] and abs(d[a + 1]) > 3 * se[a + 1] and \
                        np.sign(d[a]) == np.sign(d[a + 1])
                    orders.append(math.log(abs(d[a] / d[a + 1])) /
                                  math.log(eps_sorted[a] / eps_sorted[a + 1]) if sig else None)
                series.append({"basin": j + 1, "eps": eps_sorted, "delta": d, "se": se,
                               "local_orders": orders,
                               "delta_over_eps": [dd / e for dd, e in zip(d, eps_sorted)]})
            tab[key] = series
        # pooled: max |Delta| per eps and pooled log-log slope over all basins
        pooled = {}
        for e in sorted({float(x) for x in per_eps}, reverse=True):
            vals = [abs(sr["delta"][sr["eps"].index(e)]) for ser in tab.values() for sr in ser]
            pooled[f"{e:g}"] = {"max_abs_delta": max(vals), "median_abs_delta": float(np.median(vals))}
        allorders = [o for ser in tab.values() for sr in ser for o in sr["local_orders"]
                     if o is not None]
        out[demo] = {"series": tab, "pooled": pooled,
                     "local_orders_summary": {
                         "n": len(allorders),
                         "median": float(np.median(allorders)) if allorders else None,
                         "q25": float(np.percentile(allorders, 25)) if allorders else None,
                         "q75": float(np.percentile(allorders, 75)) if allorders else None}}
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def cmd_analyze(args) -> int:
    cache = Path(args.cache)
    results, ens_meta, dens_curves, dtcheck, expo = {}, {}, {}, {}, {}
    for demo in ("plug", "pois", "nonlin"):
        results[demo] = {}
        for eps in EPS_LIST:
            try:
                ens = load_ensemble(cache, demo, eps, 1.0, N_PATHS)
            except FileNotFoundError as exc:
                print(f"skip {demo} eps={eps}: {exc}")
                continue
            results[demo][f"{eps:g}"] = analyze_ensemble(demo, eps, ens)
            expo.setdefault(demo, {})[f"{eps:g}"] = exposure_moments(demo, ens, eps)
            mt = ens["metas"][0]
            ens_meta[f"{demo}|{eps:g}"] = {
                "N": ens["N"], "groups": ens["groups"], "nsteps": mt["nsteps"],
                "dt_min": mt["dt_min"], "dt_max": mt["dt_max"], "entropy": mt["entropy"],
                "chunks": len(ens["metas"]), "ck_times_actual": mt["ck_times_actual"],
                "T_actual": mt["T_actual"],
                "crossed_frac": np.mean([m["crossed_frac"] for m in ens["metas"]], axis=0),
                "wall_s_total": sum(m["wall_s"] for m in ens["metas"])}
            if demo in ("plug", "nonlin"):
                dens_curves[f"{demo}|{eps:g}"] = density_curve(
                    demo, eps, ens, {"shape": "gauss", "m": 3, "B": 8.0, "alloc": "design"})
            else:
                dens_curves[f"{demo}|{eps:g}"] = density_curve(
                    demo, eps, ens, {"shape": "gauss", "m": 3, "B": 8.0, "alloc": "design"})
            del ens
        # dt check at eps = 0.05, dt/2, 1e5 paths
        try:
            ens_h = load_ensemble(cache, demo, DTCHECK_EPS, 0.5, DTCHECK_PATHS)
        except FileNotFoundError as exc:
            print(f"skip dtcheck {demo}: {exc}")
            continue
        rows_h = analyze_ensemble(demo, DTCHECK_EPS, ens_h, with_density=False)
        rows_f = results[demo].get(f"{DTCHECK_EPS:g}")
        if rows_f:
            obs = "fk_position_cells" if demo == "pois" else "fk_time_basins"
            zs, ds = [], []
            for rh, rf in zip(rows_h, rows_f):
                for j in range(rh["m"]):
                    dd = rh[obs]["mass"][j] - rf[obs]["mass"][j]
                    se = math.hypot(rh[obs]["se"][j], rf[obs]["se"][j])
                    ds.append(dd)
                    zs.append(dd / se if se > 0 else 0.0)
            dtcheck[demo] = {"eps": DTCHECK_EPS, "dt_factor": 0.5, "N_half": ens_h["N"],
                             "observable": obs, "max_abs_diff": float(np.max(np.abs(ds))),
                             "max_abs_z": float(np.max(np.abs(zs))),
                             "rms_z": float(np.sqrt(np.mean(np.square(zs)))), "n": len(zs)}
    conv = convergence_table(results)
    payload = {
        "meta": {
            "driver": "code/fb_n14_universality.py", "seed": BASE_SEED, "tag": TAG,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "n_paths_per_eps": N_PATHS, "chunk": CHUNK, "groups_per_chunk": GROUPS_PER_CHUNK,
            "out_every": OUT_EVERY, "dt_rule": {"channel": "eps/40",
                                                "nonlin": "(eps/40)/max(|b(phi(t))|,0.3)"},
            "D0": D0, "rho": RHO, "U": U, "area_A": AREA, "eps_list": EPS_LIST, "B_list": B_LIST,
            "kill_rule": "end-of-step, P(kill) = 1 - exp(-(B/A) sum_j w_j phi_j(x_n) dt_n)",
            "geometry": {d: geometry(d) for d in ("plug", "pois", "nonlin")},
            "pois": {"y0_band": POIS_Y0, "mean_speed": pois_mean_speed(),
                     "harmonic_speed": pois_harmonic_speed(),
                     "law": "M_j = G(K_{j-1}) - G(K_j), K_j = B sum_{i<=j} w_i / A, "
                            "G(K) = E exp(-K/u(Y0))",
                     "design": "K_j = G^{-1}(1 - j p*), p* = (1 - G(B))/m"},
            "subsets": SUBSETS, "cut_index": CUT_INDEX,
            "tophat_half_width_over_eps_rho": TOPHAT_HALF,
            "ensembles": ens_meta},
        "results": results, "convergence": conv, "dt_check": dtcheck,
        "exposure_moments": expo,
        "density_curves": dens_curves,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(_clean(payload), fh, indent=1)
    print(f"wrote {OUT_JSON}")
    return 0


# ============================================================================
# Figure
# ============================================================================


def cmd_figure(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core.apply_prr_style()
    with open(args.json) as fh:
        P = json.load(fh)
    R = P["results"]
    eps_col = {"0.1": core.OI_ORANGE, "0.05": core.OI_SKY, "0.025": core.OI_BLUE}
    text_w = 390.0 / 72.27
    fig, axes = plt.subplots(1, 2, figsize=(text_w, 2.75), layout="constrained")

    def parity(ax, demo, obs, marker, label_done, face_full=True, allocs=("equal", "design"),
               shape="gauss"):
        for eps_s in ("0.1", "0.05", "0.025"):
            rows = R.get(demo, {}).get(eps_s, [])
            xs, ys = [], []
            for r in rows:
                if r["alloc"] not in allocs or r["shape"] != shape:
                    continue
                xs += r["law"]
                ys += r[obs]["mass"]
            if not xs:
                continue
            lab = None
            if (eps_s, "eps") not in label_done:
                lab = rf"$\varepsilon={eps_s}$"
                label_done.add((eps_s, "eps"))
            ax.plot(xs, ys, marker, ms=3.2 if eps_s == "0.025" else 2.6, ls="none",
                    mfc=eps_col[eps_s] if face_full else "none", mec=eps_col[eps_s],
                    mew=0.7, label=lab, alpha=0.95)

    from matplotlib.lines import Line2D

    def eps_legend(ax, extra_handles=(), extra_labels=(), loc="upper left"):
        h = [Line2D([], [], marker="o", ls="none", mfc=eps_col[e], mec=eps_col[e], ms=3)
             for e in ("0.1", "0.05", "0.025")]
        ax.legend(h + list(extra_handles),
                  [r"$\varepsilon=0.1$", r"$\varepsilon=0.05$", r"$\varepsilon=0.025$"]
                  + list(extra_labels), loc=loc, fontsize=6.3, handletextpad=0.2,
                  borderaxespad=0.3, labelspacing=0.2, handlelength=1.2)

    def density_inset(ax, key, logt):
        dc = P["density_curves"].get(key)
        if not dc:
            return
        # R1 fix editor (finding O15): inset raised (0.075 -> 0.13) and its x label given
        # a positive labelpad, so that "t" no longer touches the tick labels at print size.
        ins = ax.inset_axes([0.54, 0.13, 0.43, 0.34])
        t = np.array(dc["t"])
        f = np.array(dc["f"])
        yv = t * f if logt else f
        ins.plot(t, yv, color=core.OI_BLUE, lw=0.9)
        if dc["limit"]:
            tl = np.array(dc["limit"]["t"])
            fl = np.array(dc["limit"]["f"])
            ins.plot(tl, tl * fl if logt else fl, color="k", lw=0.6, ls="--")
        if logt:
            ins.set_xscale("log")
            ins.set_xlim(0.015, 1.2)
            ins.set_ylabel(r"$t\,f(t)$", fontsize=6.3, labelpad=0)
        else:
            ins.set_xlim(0, 7.5)
            ins.set_ylabel(r"$f(t)$", fontsize=6.3, labelpad=0)
        ins.set_ylim(bottom=0)
        ins.set_xlabel(r"$t$", fontsize=6.3, labelpad=1.5)
        ins.tick_params(labelsize=5.8, pad=1, length=1.5)

    # ---- (a) channel ------------------------------------------------------
    ax = axes[0]
    done = set()
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, zorder=0)
    parity(ax, "plug", "fk_time_basins", "o", done)
    parity(ax, "plug", "fk_time_basins", "s", done, face_full=False, shape="tophat")
    parity(ax, "pois", "fk_position_cells", "^", done)
    # single-speed (mean injected speed) Poiseuille design: x = its own target, y = FK
    xs, ys = [], []
    for r in R.get("pois", {}).get("0.025", []):
        if r["alloc"] == "naive_design":
            xs += [r["naive_target"]] * r["m"]
            ys += r["fk_position_cells"]["mass"]
    ax.plot(xs, ys, "x", color="0.3", ms=3.5, mew=0.8, ls="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"predicted mass $M_j$")
    ax.set_ylabel(r"exact-law basin mass")
    ax.set_title("(a) channel flow", fontsize=7.5, loc="left")
    eps_legend(ax,
               [Line2D([], [], marker="o", ls="none", mfc="0.55", mec="0.55", ms=3),
                Line2D([], [], marker="s", ls="none", mfc="none", mec="0.4", ms=3),
                Line2D([], [], marker="^", ls="none", mfc="0.55", mec="0.55", ms=3),
                Line2D([], [], marker="x", ls="none", color="0.3", ms=3.5)],
               ["plug, Gaussian", "plug, top-hat", "Poiseuille", "single-speed design"])
    density_inset(ax, "plug|0.025", logt=False)
    # ---- (b) nonlinear relaxation ----------------------------------------
    ax = axes[1]
    done = set()
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, zorder=0)
    parity(ax, "nonlin", "fk_time_basins", "o", done)
    xs, ys = [], []
    for r in R.get("nonlin", {}).get("0.025", []):
        if r["alloc"] == "naive_design":
            xs += [r["naive_target"]] * r["m"]
            ys += r["fk_time_basins"]["mass"]
    ax.plot(xs, ys, "x", color="0.3", ms=3.5, mew=0.8, ls="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"predicted mass $M_j$")
    ax.set_title(r"(b) anharmonic relaxation", fontsize=7.5, loc="left")
    eps_legend(ax, [Line2D([], [], marker="x", ls="none", color="0.3", ms=3.5)],
               ["harmonic design"], loc="upper right")
    density_inset(ax, "nonlin|0.025", logt=True)
    core.stream_tag(fig, "fb N14")
    written = core.save_figure(fig, FIG_STEM)
    print("\n".join(written))
    return 0


# ============================================================================
# Orchestration
# ============================================================================


def active_python_processes() -> int:
    """Number of python processes using > 20% CPU (idle MCP servers excluded)."""
    try:
        txt = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return 0
    n = 0
    for line in txt.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "python" in parts[1].lower():
            try:
                if float(parts[0]) > 20.0:
                    n += 1
            except ValueError:
                pass
    return n


def task_list(demos, eps_list, with_dtcheck: bool) -> list[tuple]:
    tasks = []
    for demo in demos:
        for eps in eps_list:
            for c in range(N_PATHS // CHUNK):
                tasks.append((demo, eps, 1.0, N_PATHS, c))
        if with_dtcheck:
            for c in range(DTCHECK_PATHS // CHUNK):
                tasks.append((demo, DTCHECK_EPS, 0.5, DTCHECK_PATHS, c))
    # cheapest first is irrelevant; put the long (small-eps) tasks first for packing
    tasks.sort(key=lambda t: (t[1] * (1.0 / t[2]) if t[2] else 0.0))
    return tasks


def cmd_simulate(args) -> int:
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    demos = args.demos.split(",")
    eps_list = [float(e) for e in args.eps.split(",")]
    tasks = [t for t in task_list(demos, eps_list, not args.no_dtcheck)
             if not chunk_path(cache, t[0], t[1], t[2], t[3], t[4]).exists()]
    deadline = time.time() + 60.0 * args.max_minutes
    running: list[tuple] = []
    log = cache / "simulate.log"
    print(f"{len(tasks)} tasks pending", flush=True)
    while tasks or running:
        running = [(p, t, ts) for (p, t, ts) in running if p.poll() is None]
        can_launch = (tasks and len(running) < args.workers
                      and time.time() + args.est_minutes * 60.0 < deadline)
        if can_launch:
            if active_python_processes() - len(running) >= PY_ACTIVE_LIMIT:
                time.sleep(30)
                continue
            t = tasks.pop(0)
            cmd = [sys.executable, str(HERE), "chunk", "--demo", t[0], "--eps", str(t[1]),
                   "--dt-factor", str(t[2]), "--n-total", str(t[3]), "--chunk-index", str(t[4]),
                   "--cache", str(cache)]
            with open(log, "a") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} launch {t}\n")
            p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT)
            running.append((p, t, time.time()))
            continue
        if not running and tasks:
            print(f"deadline reached; {len(tasks)} tasks left", flush=True)
            break
        time.sleep(5)
    print("simulate: done", flush=True)
    return 0


def cmd_chunk(args) -> int:
    meta = run_chunk(args.demo, args.eps, args.dt_factor, args.n_total, args.chunk_index,
                     Path(args.cache))
    print(f"{time.strftime('%H:%M:%S')} done {args.demo} eps={args.eps} dtf={args.dt_factor} "
          f"c={args.chunk_index} steps={meta['nsteps']} wall={meta['wall_s']:.1f}s", flush=True)
    return 0


def cmd_law(args) -> int:
    for demo in ("plug", "pois", "nonlin"):
        g = geometry(demo)
        print(demo, json.dumps({k: g[k] for k in ("centres", "cuts", "speeds", "cross_times",
                                                   "offset_var")}))
        for cell in declared_cells(demo):
            if cell["shape"] != "gauss":
                continue
            M = law_masses(demo, cell["B"], cell["w"], cell["idx"])
            print(f"  m={cell['m']} B={cell['B']:g} {cell['alloc']:13s} w={np.round(cell['w'], 4)} "
                  f"M={np.round(M, 4)}")
    print("pois mean speed", pois_mean_speed(), "harmonic", pois_harmonic_speed())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--demos", default="plug,pois,nonlin")
    s.add_argument("--eps", default=",".join(str(e) for e in EPS_LIST))
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--max-minutes", type=float, default=24.0)
    s.add_argument("--est-minutes", type=float, default=4.0)
    s.add_argument("--no-dtcheck", action="store_true")
    s.add_argument("--cache", default=str(DEFAULT_CACHE))
    c = sub.add_parser("chunk")
    c.add_argument("--demo", required=True)
    c.add_argument("--eps", type=float, required=True)
    c.add_argument("--dt-factor", type=float, default=1.0)
    c.add_argument("--n-total", type=int, default=N_PATHS)
    c.add_argument("--chunk-index", type=int, required=True)
    c.add_argument("--cache", default=str(DEFAULT_CACHE))
    sub.add_parser("law")
    a = sub.add_parser("analyze")
    a.add_argument("--cache", default=str(DEFAULT_CACHE))
    f = sub.add_parser("figure")
    f.add_argument("--json", default=str(OUT_JSON))
    args = ap.parse_args(argv)
    if args.cmd == "simulate":
        args.workers = min(args.workers, 3)
        return cmd_simulate(args)
    if args.cmd == "chunk":
        return cmd_chunk(args)
    if args.cmd == "law":
        return cmd_law(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "figure":
        return cmd_figure(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
