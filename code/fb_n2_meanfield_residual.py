#!/usr/bin/env python3
"""N2 -- decomposing the mean-field residual and checking the TH-1 sandwich.

GAP_CLOSURE_PLAN.md §3 N2 (closes G05, tests TH-1).  Cells (2, 0.05), (2, 0.1),
(3, 0.1), (3, 0.15) at B in {0.5, 1, 2, 4, 8}, equal weights (the paper's model).

Exact laws (Feynman--Kac, common paths) of three kernels:
  full         V = 1{|R|_mi < a} sum_j w_j phi_j(Z) / W^(d-1)   (the model)
  meancontact  gate replaced by its mean c(t)                      (no gating noise)
  nogate       gate removed                                        (c = 1)
Mean-field laws f_1 = B G exp(-B Lambda):
  EM-exact     G = E V of the Euler--Maruyama chain (exact Gaussian law; per kernel)
  continuum    G of validate_exact_m_offlattice.free_exposure_clock (the paper's f_1,
               from which B_op^mf = 7.358 at (2, 0.05) was computed)

Decomposition of the residual of the exact law (per window bin, per basin, and for the
5 % prominence-floor crossing B_5%):
  f_full - f_1^cont = (f_full - f_mc)                 contact-gating fluctuation
                    + (f_mc - f_1^mc,EM)               midpoint-exposure fluctuation
                    + (f_1^mc,EM - f_1^full,EM)        c(t) quadrature vs EM contact (O(dt))
                    + (f_1^full,EM - f_1^cont)         time step (Euler--Maruyama bias of f_1)
TH-1 checks: sandwich e^{-B Lam} <= S_B <= e^{-B Lam} + B^2 h(B Lam) Var X at every stored
time; the density-level term -B^2 e^{-B Lam} Cov(V, X) and its remainder bound; Var X_t by
exact two-time Gaussian (Mehler) sums for the EM chain vs the FK sample variance.

Ensembles: N0 ensembles n0_m2_eps0.05, n0_m2_eps0.1, n0_m3_eps0.1 (tag 80, 2e5 paths,
variants full/nogate/meancontact on common paths) and one new ensemble n2_m3_eps0.15
(tag 81, 2e5 paths = the first four chunks of the N1 ensemble n1_m3_eps0.15, same seeds).

Subcommands: simulate | analyze [--only CELL] | assemble | figure
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

OUT_DIR = fk.FB_DATA / "N2"
PARTS = OUT_DIR / "parts"
OUT_JSON = OUT_DIR / "n2_meanfield_residual.json"
B_LIST = (0.5, 1.0, 2.0, 4.0, 8.0)
VARIANTS = ("full", "meancontact", "nogate")
GATE_OF = {"full": "contact", "meancontact": "mean", "nogate": "none"}
CELLS = {
    "m2_eps0.05": (2, 0.05, "n0_m2_eps0.05"),
    "m2_eps0.1": (2, 0.1, "n0_m2_eps0.1"),
    "m3_eps0.1": (3, 0.1, "n0_m3_eps0.1"),
    "m3_eps0.15": (3, 0.15, "n2_m3_eps0.15"),
}
BP_CELLS = {"m2_eps0.05": (5.0, 9.5), "m3_eps0.1": (2.5, 5.0)}
GROUPS = fk.DEFAULT_GROUPS
TAG_NEW = fk.TAGS["N1"]  # 81: same paths as n1_m3_eps0.15 (chunks 0-3)


# ---------------------------------------------------------------------------
# small numerics
# ---------------------------------------------------------------------------


def h_fun(u):
    """h(u) = (1 - (1 + u) e^{-u}) / u^2, h(0) = 1/2 (TH-1), stable for small u."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    small = u < 1e-3
    us = u[small]
    out[small] = 0.5 - us / 3.0 + us * us / 8.0
    ul = u[~small]
    out[~small] = (-np.expm1(-ul) - ul * np.exp(-ul)) / (ul * ul)
    return out


def _erfc(x):
    """Numerical-Recipes erfc (fractional error < 1.2e-7), vectorised."""
    x = np.asarray(x, float)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, ans, 2.0 - ans)


def _Phi(x):
    return 0.5 * _erfc(-np.asarray(x) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# EM-exact moments: G, Lambda and Var X (two-time Gaussian sums)
# ---------------------------------------------------------------------------


def c2_table(spec: fk.EnsembleSpec, k_stride: int = 20, n_r: int = 24, n_phi: int = 48,
             images: int = 4, lags=None, ks=None, u_mult: float = 1.0) -> dict:
    """Two-time contact probability c2(k, k+D) = P(chi_k = 1, chi_{k+D} = 1) of the EM chain.

    Exact Gaussian law of the relative EM chain: R_par is AR(1) (A = 1 - gamma dt),
    R_perp a Gaussian random walk wrapped on the torus (a < W/2), independent.  Outer
    integral over the contact disk at step k in polar coordinates (Gauss--Legendre in r,
    split at a - L to resolve the boundary layer of width ~ the transition s.d.; trapezoid
    in the angle); inner probability that step k+D is in contact, given the state at k,
    as a 1-D integral over the transverse position u = a sin(theta) of step k+D (image
    sums) of the exact longitudinal Gaussian chord probability.  Lags below D_min = 16
    steps are not tabulated: c2 is interpolated in sqrt(D) between D = 0 (c2 = c_k, the
    exact EM contact probability) and D = 16.
    """
    if spec.n_perp != 1 or spec.r_perp0 != 0.0:
        raise NotImplementedError("c2_table: d = 2, r_perp0 = 0 only")
    mom = fk.em_moments(spec)
    steps = spec.steps()
    a = spec.contact_a
    W = spec.torus_w
    A = 1.0 - spec.gamma * spec.dt
    if lags is None:
        lags = [0, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640,
                800, 1024, 1280, 1600, 2048, 2560, 3200, steps - 1]
    if ks is None:
        ks = list(range(0, steps, k_stride))
    c_em = fk.contact_probability_em(spec)
    tab = np.full((len(ks), len(lags)), np.nan)
    rn, rw = np.polynomial.legendre.leggauss(n_r)
    phi = 2 * math.pi * np.arange(n_phi) / n_phi
    wphi = 2 * math.pi / n_phi
    nimg = np.arange(-images, images + 1) * W
    for ik, k in enumerate(ks):
        Vk = mom["var_perp"][k]
        Uk = mom["var_par"][k]
        mk = mom["mean_par"][k]
        for il, D in enumerate(lags):
            l = k + D
            if l >= steps:
                continue
            if D == 0:
                tab[ik, il] = c_em[k]
                continue
            dV = mom["var_perp"][l] - Vk
            sxi = math.sqrt(max(mom["var_par"][l] - A ** (2 * D) * Uk, 1e-300))
            AD = A ** D
            sb = math.sqrt(dV + sxi * sxi)
            L = min(0.5 * a, 6.0 * sb)
            r1 = 0.5 * (a - L) * (rn + 1.0); w1 = 0.5 * (a - L) * rw
            r2 = (a - L) + 0.5 * L * (rn + 1.0); w2 = 0.5 * L * rw
            r = np.concatenate([r1, r2]); wr = np.concatenate([w1, w2])
            R, PH = np.meshgrid(r, phi, indexing="ij")
            p = (R * np.cos(PH)).ravel()
            y = (R * np.sin(PH)).ravel()
            wout = (np.repeat(wr * r, n_phi) * wphi
                    * np.exp(-(p - mk) ** 2 / (2 * Uk)) / math.sqrt(2 * math.pi * Uk)
                    * (np.exp(-(y[:, None] + nimg[None, :]) ** 2 / (2 * Vk)).sum(1)
                       / math.sqrt(2 * math.pi * Vk)))
            sd_u = math.sqrt(dV)
            n_u = int(u_mult * (320 if sd_u < 0.02 else (200 if sd_u < 0.05 else 96)))
            xg, wg = np.polynomial.legendre.leggauss(n_u)
            thu = 0.5 * math.pi * xg
            u = a * np.sin(thu)
            ju = a * np.cos(thu) * 0.5 * math.pi * wg
            hu = a * np.cos(thu)
            nimg_d = np.arange(-2, 3) * W
            keep = wout > 1e-14 * wout.max()
            pk, yk, wk = p[keep], y[keep], wout[keep]
            # transverse transition density to u (mod W), summed over images
            dd = u[None, :] - yk[:, None]
            tr = np.zeros_like(dd)
            for n_ in nimg_d:
                tr += np.exp(-(dd + n_) ** 2 / (2 * dV))
            tr /= math.sqrt(2 * math.pi * dV)
            mp = AD * pk[:, None]
            lp = _Phi((hu[None, :] - mp) / sxi) - _Phi((-hu[None, :] - mp) / sxi)
            pin = np.einsum("ij,j->i", tr * lp, ju)
            tab[ik, il] = float(np.sum(wk * pin))
    return {"ks": np.asarray(ks), "lags": np.asarray(lags), "c2": tab, "c_em": c_em}


def cached_c2_table(spec: fk.EnsembleSpec) -> dict:
    """c2 depends only on the relative-process law (eps, a, u0, sigma_perp0, r_par0, dt): cache by eps."""
    d = OUT_DIR / "c2_tables"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"c2_eps{spec.eps:g}_a{spec.contact_a:g}_dt{spec.dt:g}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    t0 = time.time()
    tab = c2_table(spec)
    np.savez_compressed(f, **tab, seconds=np.asarray(time.time() - t0))
    return tab


def c2_interp(tabd: dict, k: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of c2 in (k, sqrt(D)); k, D integer arrays (same shape)."""
    ks = tabd["ks"].astype(float)
    sl = np.sqrt(tabd["lags"].astype(float))
    tab = tabd["c2"]
    kf = np.clip((k - ks[0]) / (ks[1] - ks[0]), 0, ks.size - 1 - 1e-9)
    i0 = np.floor(kf).astype(int)
    fk_ = kf - i0
    sD = np.sqrt(D.astype(float))
    j0 = np.clip(np.searchsorted(sl, sD, side="right") - 1, 0, sl.size - 2)
    fd = (sD - sl[j0]) / (sl[j0 + 1] - sl[j0])
    fd = np.clip(fd, 0.0, 1.0)

    def val(ii, jj):
        v = tab[ii, jj]
        return v

    v00 = val(i0, j0); v01 = val(i0, j0 + 1)
    v10 = val(np.minimum(i0 + 1, ks.size - 1), j0); v11 = val(np.minimum(i0 + 1, ks.size - 1), j0 + 1)
    # rows near the end may miss long lags (NaN): fall back to the available row
    v10 = np.where(np.isnan(v10), v00, v10); v11 = np.where(np.isnan(v11), v01, v11)
    v01 = np.where(np.isnan(v01), v00, v01); v11 = np.where(np.isnan(v11), v10, v11)
    out = (1 - fk_) * ((1 - fd) * v00 + fd * v01) + fk_ * ((1 - fd) * v10 + fd * v11)
    return out


def em_var_exposure(spec: fk.EnsembleSpec, w, gate: str, c2tab: dict | None = None,
                    block: int = 250) -> dict:
    """Exact (EM-chain) Lambda_n and Var X_n at every step, unit budget.

    e_n = g_n sum_j w_j phi_j(Z_n) dt / W^(d-1);  Z_n exactly Gaussian with
    Cov(Z_k, Z_l) = A^{|l-k|} Var Z_min(k,l), A = 1 - gamma dt.
    gate 'none' (g = 1), 'mean' (g = c(t_n), deterministic) or 'contact'
    (E[g_k g_l] = c2(k, l) from ``c2_table``).
    """
    mom = fk.em_moments(spec)
    steps = spec.steps()
    dt = spec.dt
    A = 1.0 - spec.gamma * dt
    w = np.asarray(w, float)
    cen = spec.centres()
    s2 = spec.sd() ** 2
    mz, vz = mom["mean_z"], mom["var_z"]
    pref = dt / spec.torus_w ** spec.n_perp
    # one-time factors
    ev = np.zeros(steps)
    for wj, c in zip(w, cen):
        v = vz + s2
        ev += wj * np.exp(-(mz - c) ** 2 / (2 * v)) / np.sqrt(2 * math.pi * v)
    ev *= pref
    T = mom["t"]
    if gate == "none":
        g = np.ones(steps)
    elif gate == "mean":
        g = fk.mean_contact_curve(spec, spec.contact_a, T)
    elif gate == "contact":
        g = fk.contact_probability_em(spec)
    else:
        raise ValueError(gate)
    mean_e = g * ev
    Lam = np.cumsum(mean_e)
    # Var X_n = sum_{k,l<=n} Cov(e_k, e_l): rows k, columns l <= k
    rs = np.zeros(steps)   # sum_{l<k} Cov(e_k, e_l)
    dg = np.zeros(steps)   # Var e_k
    idx = np.arange(steps)
    for k0 in range(0, steps, block):
        k1 = min(steps, k0 + block)
        kk = idx[k0:k1][:, None]
        ll = idx[None, :k1]
        mask = ll <= kk
        D = np.where(mask, kk - ll, 0)
        vmin = vz[np.minimum(kk, ll)]
        cz = (A ** D) * vmin
        E2 = np.zeros((k1 - k0, k1))
        for wi, ci in zip(w, cen):
            for wj, cj in zip(w, cen):
                v1 = vz[kk] + s2
                v2 = vz[ll] + s2
                det = v1 * v2 - cz * cz
                x1 = ci - mz[kk]
                x2 = cj - mz[ll]
                q = (v2 * x1 * x1 - 2 * cz * x1 * x2 + v1 * x2 * x2) / det
                E2 += wi * wj * np.exp(-0.5 * q) / (2 * math.pi * np.sqrt(det))
        E2 *= pref * pref
        if gate == "contact":
            gg = c2_interp(c2tab, np.minimum(kk, ll) + 0 * D, D)
        else:
            gg = g[kk] * g[ll]
        cov = gg * E2 - mean_e[kk] * mean_e[ll]
        cov = np.where(mask, cov, 0.0)
        diag = cov[np.arange(k1 - k0), idx[k0:k1]]
        dg[k0:k1] = diag
        rs[k0:k1] = cov.sum(axis=1) - diag
    var = np.cumsum(2 * rs + dg)
    return {"t": T, "Lambda": Lam, "var_X": var, "G": mean_e / dt, "gate_factor": g}


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def cmd_simulate(args) -> None:
    spec = fk.EnsembleSpec(m=3, eps=0.15)
    ens = fk.simulate_ensemble(spec, int(float(args.paths)), name="n2_m3_eps0.15", tag=TAG_NEW,
                               variants=("full", "nogate", "meancontact"), workers=args.workers,
                               note="N2 (3,0.15) three-kernel ensemble; same seeds as n1_m3_eps0.15 "
                                    "chunks 0-3")
    print(ens, flush=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def geometric_cut_indices(ens: fk.Ensemble) -> list[int]:
    c = ens.spec.centres()
    s = [-math.log((0.5 * (a + b) - ens.spec.z_bar) / (ens.spec.z0 - ens.spec.z_bar)) / ens.spec.gamma
         for a, b in zip(c[:-1], c[1:])]
    wi = ens.window_index
    return [int(wi[0])] + [int(np.argmin(np.abs(ens.edges - x))) for x in s] + [int(wi[-1])]


def variant_pass(ens: fk.Ensemble, variant: str, w, Bs, groups: int = GROUPS,
                 block: int = 25_000) -> dict:
    K = ens.cps.size
    N = ens.n_paths
    gb = fk._group_bounds(N, groups)
    wi = np.asarray(ens.window_index)
    nb = wi.size - 1
    w = np.asarray(w, float)
    acc = {"X1": np.zeros(K), "X2": np.zeros(K), "X3": np.zeros(K),
           "gX1": np.zeros((groups, K)), "gX2": np.zeros((groups, K)), "gX3": np.zeros((groups, K))}
    per = {B: {"S": np.zeros(K), "S2": np.zeros(K), "gS": np.zeros((groups, K)),
               "Y": np.zeros(K), "Y2": np.zeros(K),
               "P": np.zeros(nb), "P2": np.zeros(nb), "gP": np.zeros((groups, nb))} for B in Bs}
    for start, dX, _ in ens.iter_chunks(variant):
        n = dX.shape[0]
        for b0 in range(0, n, block):
            blk = dX[b0:b0 + block].astype(np.float64)
            gid = fk._group_ids(gb, start + b0, blk.shape[0])
            dXw = np.einsum("imk,m->ik", blk, w)
            X = np.cumsum(dXw, axis=1)
            X2 = X * X
            X3 = X2 * X
            acc["X1"] += X.sum(0); acc["X2"] += X2.sum(0); acc["X3"] += X3.sum(0)
            acc["gX1"] += fk._group_sums(X, gid, groups)
            acc["gX2"] += fk._group_sums(X2, gid, groups)
            acc["gX3"] += fk._group_sums(X3, gid, groups)
            Xw = X[:, wi]
            segw = np.add.reduceat(dXw[:, : wi[-1] + 1], wi[:-1] + 1, axis=1)
            for B in Bs:
                E = np.exp(-B * X)
                a = per[B]
                a["S"] += E.sum(0); a["S2"] += (E * E).sum(0)
                Yc = -np.expm1(-B * X)          # 1 - e^{-BX} without cancellation
                a["Y"] += Yc.sum(0); a["Y2"] += (Yc * Yc).sum(0)
                a["gS"] += fk._group_sums(E, gid, groups)
                y = E[:, wi[:-1]] * (-np.expm1(-B * segw))
                a["P"] += y.sum(0); a["P2"] += (y * y).sum(0)
                a["gP"] += fk._group_sums(y, gid, groups)
    return {"acc": acc, "per": per, "N": N, "sizes": np.diff(gb).astype(float)}


def _moments(acc, N, sizes):
    m1 = acc["X1"] / N
    m2 = acc["X2"] / N
    m3 = acc["X3"] / N
    var_pop = np.maximum(m2 - m1 * m1, 0.0)
    k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    g1 = acc["gX1"] / sizes[:, None]
    g2 = acc["gX2"] / sizes[:, None]
    gv = g2 - g1 * g1
    return {"Lambda": m1, "var_pop": var_pop, "var_unbiased": var_pop * N / (N - 1),
            "kappa3": k3, "var_se_batch": fk._batch_se(gv, sizes),
            "Lambda_se_batch": fk._batch_se(g1, sizes)}


def mf_bins(Lam_edges, B):
    E = np.exp(-B * np.asarray(Lam_edges))
    return E[:-1] - E[1:]


def mf_crossing(Lam_edges, edges, after_time, *, p=0.05, B_lo, B_hi, walkers=1e6,
                n_scan=81, iters=60) -> dict:
    """Deterministic B_p of a mean-field law (last-mode relative prominence falls to p)."""
    def r(B):
        cls = fk.classify_expected(mf_bins(Lam_edges, B), edges, walkers=walkers)
        return fk.last_mode_prominence(cls, after_time)

    grid = np.geomspace(B_lo, B_hi, n_scan)
    rv = np.array([r(B) for B in grid])
    idx = [i for i in range(n_scan - 1) if rv[i] >= p and rv[i + 1] < p]
    if not idx:
        return {"B_p": None, "status": "no_crossing_in_range", "scan_B": grid.tolist(),
                "scan_r": rv.tolist()}
    i = idx[0]
    lo, hi = grid[i], grid[i + 1]
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if r(mid) >= p:
            lo = mid
        else:
            hi = mid
    return {"B_p": float(math.sqrt(lo * hi)), "status": "bisected",
            "bracket": [float(lo), float(hi)], "n_down_crossings_in_scan": len(idx)}


def analyze_cell(label: str, with_c2: bool = True) -> dict:
    t0 = time.time()
    m, eps, name = CELLS[label]
    ens = fk.load_ensemble(name)
    spec = ens.spec
    w = [1.0 / m] * m
    wi = np.asarray(ens.window_index)
    edges_w = ens.edges[wi]
    cps = ens.cps
    ci = geometric_cut_indices(ens)
    # deterministic mean-field clocks: EM-exact per kernel, and the paper's continuum clock
    em = {}
    for v in VARIANTS:
        gate = GATE_OF[v]
        c2tab = None
        if gate == "contact" and with_c2:
            c2tab = cached_c2_table(spec)
        if gate == "contact" and not with_c2:
            fe = fk.free_exposure_discrete(spec, w, gate="contact")
            lam = np.cumsum(fe["G"]) * spec.dt
            em[v] = {"Lambda": lam, "var_X": None}
            continue
        em[v] = em_var_exposure(spec, w, gate, c2tab)
    lam_q = {v: np.concatenate([[0.0], em[v]["Lambda"]])[cps] for v in VARIANTS}
    varq = {v: (np.concatenate([[0.0], em[v]["var_X"]])[cps] if em[v]["var_X"] is not None
                else None) for v in VARIANTS}
    import exact_m_prr_mean_field_boundary as mfb
    clock = mfb.clock_for(m, eps, tuple(w))
    lam_cont_w = clock.lam[clock.edge_index]            # continuum Lambda at nominal window edges
    after = fk.g_valley_times(spec, w)[-1]
    out = {"label": label, "m": m, "eps": eps, "ensemble": name, "n_paths": ens.n_paths,
           "seed": ens.index["seed"], "tag": ens.index["tag"], "seed_entropy": ens.index["seed_entropy"],
           "cuts_window_edges": [float(ens.edges[k]) for k in ci], "after_time_last_G_valley": after,
           "variants": {}}
    passes = {}
    for v in VARIANTS:
        passes[v] = variant_pass(ens, v, w, B_LIST)
    N = ens.n_paths
    sizes = passes["full"]["sizes"]
    G = sizes.size
    for v in VARIANTS:
        ps = passes[v]
        mo = _moments(ps["acc"], N, sizes)
        rec = {"var_X_at_cuts_sample": mo["var_unbiased"][ci].tolist(),
               "var_X_at_cuts_se_batch": mo["var_se_batch"][ci].tolist(),
               "Lambda_at_cuts_sample": mo["Lambda"][ci].tolist(),
               "Lambda_at_cuts_em_exact": lam_q[v][ci].tolist()}
        # (criterion 2) quadrature vs sample variance over the window checkpoints
        if varq[v] is not None:
            vs, se = mo["var_unbiased"][wi], mo["var_se_batch"][wi]
            vq = varq[v][wi]
            # exclude times where Var X is negligible (< 1e-6 of its window maximum): there the
            # simulation's 12-sd kernel truncation, absent from the quadrature, dominates
            ok = (se > 0) & (vq > 1e-6 * vq.max())
            z = np.where(ok, (vs - vq) / np.where(ok, se, 1), 0.0)
            rec["var_quadrature_check"] = {
                "var_X_at_cuts_em_exact": varq[v][ci].tolist(),
                "z_at_cuts": ((mo["var_unbiased"][ci] - varq[v][ci])
                              / np.where(mo["var_se_batch"][ci] > 0, mo["var_se_batch"][ci], np.inf)).tolist(),
                "max_abs_z_window": float(np.max(np.abs(z))),
                "t_at_max": float(edges_w[int(np.argmax(np.abs(z)))]),
                "n_window_points": int(ok.sum()),
                "n_abs_z_gt_3": int(np.sum(np.abs(z) > 3)),
                "var_floor_rule": "points with Var_q > 1e-6 max_window Var_q",
                "max_rel_diff_where_var_gt_1pct_of_max": float(np.max(
                    np.abs(vs - vq)[vq > 0.01 * vq.max()] / vq[vq > 0.01 * vq.max()]))}
            # the lambda check (sampled vs EM-exact)
        lz = (mo["Lambda"][wi] - lam_q[v][wi]) / np.where(mo["Lambda_se_batch"][wi] > 0,
                                                         mo["Lambda_se_batch"][wi], np.inf)
        rec["Lambda_sample_vs_em_exact_max_abs_z_window"] = float(np.max(np.abs(lz)))
        rec["budgets"] = {}
        for B in B_LIST:
            a = ps["per"][B]
            Ym = a["Y"] / N
            S = 1.0 - Ym
            S_se = np.sqrt(np.maximum(a["Y2"] / N - Ym * Ym, 0.0) / N)
            P = a["P"] / N
            Lam = mo["Lambda"]
            vp = mo["var_pop"]
            # sandwich with the ensemble's own moments (exact for the empirical law)
            Delta = S - np.exp(-B * Lam)
            delta = B * B * h_fun(B * Lam) * vp
            sand_emp = bool(np.all(Delta >= -1e-12) and np.all(Delta <= delta + 1e-12))
            # sandwich with deterministic EM-exact moments (sampling error: 3 SE)
            lo_q = np.exp(-B * lam_q[v])
            TOL = 1e-12   # float64 rounding of S ~ 1
            # compare only where the reaction probability is resolvable (1 - e^{-B Lam_q} >= 1e-6);
            # below that, the 12-sd kernel truncation of the simulation (absent from the
            # quadrature) and float rounding dominate
            act = -np.expm1(-B * lam_q[v]) >= 1e-6
            viol_lo = int(np.sum((S < lo_q - 3 * S_se - TOL) & act))
            b = {"sandwich_empirical_holds_all_times": sand_emp,
                 "n_times": int(S.size),
                 "lower_bound_em_exact_violations_3se": viol_lo,
                 "tightness_Delta_over_delta_at_cuts": [float(Delta[k] / delta[k]) if delta[k] > 0 else None
                                                        for k in ci],
                 "Delta_at_cuts": [float(Delta[k]) for k in ci],
                 "delta_at_cuts": [float(delta[k]) for k in ci]}
            if varq[v] is not None:
                up_q = lo_q + B * B * h_fun(B * lam_q[v]) * varq[v]
                b["upper_bound_em_exact_violations_3se"] = int(np.sum((S > up_q + 3 * S_se + TOL) & act))
                b["upper_bound_em_exact_min_margin_over_se"] = float(np.min(
                    np.where(act & (S_se > 0), (up_q - S) / np.where(S_se > 0, S_se, 1), np.inf)))
            b["lower_bound_em_exact_min_margin_over_se"] = float(np.min(
                np.where(act & (S_se > 0), (S - lo_q) / np.where(S_se > 0, S_se, 1), np.inf)))
            b["n_times_compared_em_exact"] = int(act.sum())
            b["sandwich_tolerance_abs"] = TOL
            # density-level identity at bin level (window): D = P - P_mf(sampled Lambda)
            Lw = Lam[wi]
            Vw = vp[wi]
            K3 = mo["kappa3"][wi]
            D = P - mf_bins(Lw, B)
            Tcov = -(B * B / 2.0) * (Vw[1:] - Vw[:-1]) * 0.5 * (np.exp(-B * Lw[:-1]) + np.exp(-B * Lw[1:]))
            rho = D - Tcov
            U = B ** 3 * h_fun(B * Lw[:-1]) * ((K3[1:] - K3[:-1]) / 3.0 + (Lw[1:] - Lw[:-1]) * 0.5 * (Vw[1:] + Vw[:-1]))
            sD = float(np.sum(np.abs(D)))
            # Gaussian (second-cumulant) closure S_2 = exp(-B Lam + B^2 Var / 2)
            S2c = np.exp(-B * Lw + 0.5 * B * B * Vw)
            D2 = P - (S2c[:-1] - S2c[1:])
            b["density_level"] = {
                "L1_f_minus_f1": sD,
                "L1_minus_B2cov_term": float(np.sum(np.abs(Tcov))),
                "L1_remainder": float(np.sum(np.abs(rho))),
                "fraction_explained_L1": 1.0 - float(np.sum(np.abs(rho))) / sD if sD > 0 else None,
                "fraction_explained_L2": 1.0 - float(np.sum(rho * rho)) / float(np.sum(D * D)) if sD > 0 else None,
                "cumulant2_closure_fraction_explained_L1": 1.0 - float(np.sum(np.abs(D2))) / sD if sD > 0 else None,
                "cumulant2_closure_fraction_explained_L2": 1.0 - float(np.sum(D2 * D2)) / float(np.sum(D * D)) if sD > 0 else None,
                "remainder_min": float(rho.min()), "remainder_bound_min_margin": float(np.min(U - rho)),
                "n_bins_remainder_negative_beyond_1e-3_of_maxabsD": int(np.sum(rho < -1e-3 * np.max(np.abs(D)))),
                "n_bins_remainder_above_bound_beyond_1e-3_of_maxabsD": int(np.sum(rho > U + 1e-3 * np.max(np.abs(D))))}
            # basins
            cum = lambda arr: np.array([arr[ci[j]:ci[j + 1]].sum() for j in range(len(ci) - 1)])  # noqa: E731
            wpos = {int(k): i for i, k in enumerate(wi)}
            cw = [wpos[k] for k in ci]
            Mb = np.array([P[cw[j]:cw[j + 1]].sum() for j in range(len(cw) - 1)])
            gPb = np.stack([a["gP"][:, cw[j]:cw[j + 1]].sum(1) for j in range(len(cw) - 1)], axis=1) / sizes[:, None]
            b["basins"] = {"mass": Mb.tolist(), "se_batch": fk._batch_se(gPb, sizes).tolist(),
                           "mean_field_em_exact": mf_bins(lam_q[v][ci], B).tolist(),
                           "mean_field_sampled": mf_bins(Lam[ci], B).tolist()}
            b["_P"] = P
            b["_gP"] = a["gP"] / sizes[:, None]
            rec["budgets"][f"{B:g}"] = b
        rec["_lam_q_w"] = lam_q[v][wi]
        out["variants"][v] = rec
    # ---- decomposition of f_full - f_1^cont (window bins) ------------------------------
    dec = {}
    for B in B_LIST:
        kb = f"{B:g}"
        Pf = out["variants"]["full"]["budgets"][kb]["_P"]
        Pm = out["variants"]["meancontact"]["budgets"][kb]["_P"]
        Pn = out["variants"]["nogate"]["budgets"][kb]["_P"]
        gf = out["variants"]["full"]["budgets"][kb]["_gP"]
        gm = out["variants"]["meancontact"]["budgets"][kb]["_gP"]
        P1_full = mf_bins(lam_q["full"][wi], B)
        P1_mc = mf_bins(lam_q["meancontact"][wi], B)
        P1_ng = mf_bins(lam_q["nogate"][wi], B)
        P1_cont = mf_bins(lam_cont_w, B)
        comps = {"gating_full_minus_mc": Pf - Pm, "exposure_mc_minus_f1mc": Pm - P1_mc,
                 "cquad_f1mc_minus_f1full": P1_mc - P1_full, "dt_f1full_minus_f1cont": P1_full - P1_cont}
        total = Pf - P1_cont
        resid = total - sum(comps.values())
        wpos = {int(k): i for i, k in enumerate(wi)}
        cw = [wpos[k] for k in ci]
        basin = lambda arr: [float(arr[cw[j]:cw[j + 1]].sum()) for j in range(len(cw) - 1)]  # noqa: E731
        gdiff = gf - gm
        gdb = np.stack([gdiff[:, cw[j]:cw[j + 1]].sum(1) for j in range(len(cw) - 1)], axis=1)
        dec[kb] = {"L1_total": float(np.abs(total).sum()),
                   "L1": {k: float(np.abs(c).sum()) for k, c in comps.items()},
                   "basin_total": basin(total), "basin": {k: basin(c) for k, c in comps.items()},
                   "basin_gating_se_batch": fk._batch_se(gdb, sizes).tolist(),
                   "telescoping_max_abs_error": float(np.max(np.abs(resid))),
                   "nogate_exposure_L1": float(np.abs(Pn - P1_ng).sum()),
                   "nogate_exposure_basin": basin(Pn - P1_ng)}
    out["decomposition"] = dec
    # ---- visibility-threshold decomposition --------------------------------------------
    if label in BP_CELLS:
        blo, bhi = BP_CELLS[label]
        bp = {}
        for v in VARIANTS:
            r = fk.prominence_floor_crossing(ens, w=w, p=0.05, B_lo=blo, B_hi=bhi, variant=v)
            bp[f"fk_{v}"] = {k: r.get(k) for k in ("B_p", "jackknife_se", "ci95", "status",
                                                   "jackknife_replicates")}
        for v in VARIANTS:
            bp[f"mf_em_{v}"] = mf_crossing(lam_q[v][wi], edges_w, after, B_lo=blo, B_hi=bhi)
        bp["mf_continuum"] = mf_crossing(lam_cont_w, clock.window_edges, after, B_lo=blo, B_hi=bhi)
        try:
            pr = mfb.find_upper_crossing(clock, after, 1e6)
            bp["mf_continuum_pass_rule"] = {k: pr.get(k) for k in pr if not isinstance(pr.get(k), (list, dict))}
        except Exception as exc:  # noqa: BLE001
            bp["mf_continuum_pass_rule"] = {"error": repr(exc)}

        def jdiff(a, b):
            ra, rb = bp[a]["jackknife_replicates"], bp[b]["jackknife_replicates"]
            d = np.array([x - y for x, y in zip(ra, rb) if x is not None and y is not None])
            g = d.size
            return float(math.sqrt((g - 1) / g * np.sum((d - d.mean()) ** 2))) if g > 1 else None

        f_full, f_mc, f_ng = (bp[f"fk_{v}"]["B_p"] for v in VARIANTS)
        m_full, m_mc, m_ng = (bp[f"mf_em_{v}"]["B_p"] for v in VARIANTS)
        m_cont = bp["mf_continuum"]["B_p"]
        if None not in (f_full, f_mc, m_full, m_mc, m_cont):
            bp["decomposition"] = {
                "offset_mf_continuum_minus_fk_full": m_cont - f_full,
                "dt_part_mf_cont_minus_mf_em_full": m_cont - m_full,
                "cquad_part_mf_em_full_minus_mf_em_mc": m_full - m_mc,
                "exposure_part_mf_em_mc_minus_fk_mc": m_mc - f_mc,
                "gating_part_fk_mc_minus_fk_full": f_mc - f_full,
                "gating_part_se_paired_jackknife": jdiff("fk_meancontact", "fk_full"),
                "exposure_part_se": bp["fk_meancontact"]["jackknife_se"],
                "fk_full_se": bp["fk_full"]["jackknife_se"],
                "nogate_exposure_part_mf_em_nogate_minus_fk_nogate":
                    (m_ng - f_ng) if (m_ng is not None and f_ng is not None) else None}
        for v in VARIANTS:
            bp[f"fk_{v}"].pop("jackknife_replicates", None)
        out["visibility_threshold"] = bp
    # strip private arrays
    for v in VARIANTS:
        rec = out["variants"][v]
        rec.pop("_lam_q_w", None)
        for kb in rec["budgets"]:
            rec["budgets"][kb].pop("_P", None)
            rec["budgets"][kb].pop("_gP", None)
    out["analysis_seconds"] = time.time() - t0
    return fk._jsonable(out)


def cmd_analyze(args) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    labels = list(CELLS) if not args.only else args.only.split(",")
    for lab in labels:
        res = analyze_cell(lab, with_c2=not args.no_c2)
        core.write_json(PARTS / f"n2_{lab}.json", res)
        print(f"[n2] {lab} done in {res['analysis_seconds']:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# assemble + figure
# ---------------------------------------------------------------------------


def cmd_assemble(args) -> None:
    parts = {}
    for lab in CELLS:
        f = PARTS / f"n2_{lab}.json"
        if f.exists():
            parts[lab] = json.loads(f.read_text())
    acc = {}
    # criterion 2: EM-exact Mehler quadrature vs FK sample variance
    c2 = {}
    for lab, p in parts.items():
        for v, rec in p["variants"].items():
            q = rec.get("var_quadrature_check")
            if q:
                c2[f"{lab}_{v}"] = {"max_abs_z_at_cuts": max(abs(x) for x in q["z_at_cuts"][1:]),
                                    "max_abs_z_window": q["max_abs_z_window"],
                                    "n_abs_z_gt_3_window": q["n_abs_z_gt_3"],
                                    "n_window_points": q["n_window_points"]}
    acc["var_quadrature"] = c2
    # criterion 3: sandwich
    s3 = {"empirical_all_hold": all(b["sandwich_empirical_holds_all_times"]
                                    for p in parts.values() for rec in p["variants"].values()
                                    for b in rec["budgets"].values()),
          "em_exact_lower_violations_3se": sum(b["lower_bound_em_exact_violations_3se"]
                                               for p in parts.values() for rec in p["variants"].values()
                                               for b in rec["budgets"].values()),
          "em_exact_upper_violations_3se": sum(b.get("upper_bound_em_exact_violations_3se", 0)
                                               for p in parts.values() for rec in p["variants"].values()
                                               for b in rec["budgets"].values()),
          "n_checks": sum(b["n_times_compared_em_exact"] for p in parts.values()
                          for rec in p["variants"].values() for b in rec["budgets"].values()),
          "min_lower_margin_over_se": min(b["lower_bound_em_exact_min_margin_over_se"]
                                          for p in parts.values() for rec in p["variants"].values()
                                          for b in rec["budgets"].values()),
          "min_upper_margin_over_se": min(b.get("upper_bound_em_exact_min_margin_over_se", np.inf)
                                          for p in parts.values() for rec in p["variants"].values()
                                          for b in rec["budgets"].values())}
    acc["sandwich"] = s3
    # criterion 4: share of f - f1 explained by -B^2 Cov at B <= 1 (full kernel)
    s4 = {}
    for lab, p in parts.items():
        for kb, b in p["variants"]["full"]["budgets"].items():
            s4[f"{lab}_B{kb}"] = {k: b["density_level"][k] for k in
                                 ("fraction_explained_L1", "fraction_explained_L2",
                                  "cumulant2_closure_fraction_explained_L1",
                                  "cumulant2_closure_fraction_explained_L2",
                                  "n_bins_remainder_negative_beyond_1e-3_of_maxabsD",
                                  "n_bins_remainder_above_bound_beyond_1e-3_of_maxabsD")}
    acc["cov_share"] = s4
    acc["cov_share_min_at_B_le_1"] = min(v["fraction_explained_L1"] for k, v in s4.items()
                                         if k.endswith("B0.5") or k.endswith("B1"))
    # criterion 5: sign of the basin-1 bias (exact <= mean field)
    s5 = []
    for lab, p in parts.items():
        for kb, b in p["variants"]["full"]["budgets"].items():
            bs = b["basins"]
            s5.append({"cell": lab, "B": float(kb), "M1_exact": bs["mass"][0],
                       "M1_mf_em_exact": bs["mean_field_em_exact"][0],
                       "diff": bs["mass"][0] - bs["mean_field_em_exact"][0],
                       "se": bs["se_batch"][0]})
    acc["basin1_sign"] = {"rows": s5, "n_rows": len(s5),
                          "n_exact_below_mf": sum(r["diff"] < 0 for r in s5),
                          "max_diff_over_se": max(r["diff"] / r["se"] for r in s5)}
    # offset decomposition
    acc["visibility"] = {lab: p.get("visibility_threshold", {}).get("decomposition")
                         for lab, p in parts.items() if "visibility_threshold" in p}
    rep = OUT_DIR / "n2_var_replicate.json"
    if rep.exists():
        acc["var_quadrature_independent_replicate_n1"] = json.loads(rep.read_text())
    payload = {"item": "N2", "driver": HERE.name, "plan": "GAP_CLOSURE_PLAN.md §3 N2",
               "model": core.model_payload(), "cells": parts, "acceptance": acc}
    core.write_json(OUT_JSON, fk._jsonable(payload))
    print(f"[n2] wrote {OUT_JSON}")


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exact_m_prr_mean_field_boundary as mfb
    core.apply_prr_style()
    d = json.loads(OUT_JSON.read_text())
    written = []
    panels = [("m2_eps0.05", 1.0), ("m2_eps0.05", 4.0), ("m3_eps0.1", 1.0), ("m3_eps0.1", 4.0)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), layout="constrained", sharex=True)
    cols = {"total": "k", "gating_full_minus_mc": core.OI_VERMILLION,
            "exposure_mc_minus_f1mc": core.OI_BLUE, "dt_f1full_minus_f1cont": core.OI_GREEN}
    labs = {"total": "$f - f_1$ (total, vs paper's $f_1$)",
            "gating_full_minus_mc": "contact gating: $f - f_{\\bar c}$",
            "exposure_mc_minus_f1mc": "exposure fluctuation: $f_{\\bar c} - f_1^{\\rm EM}$",
            "dt_f1full_minus_f1cont": "time step: $f_1^{\\rm EM} - f_1$"}
    for ax, (lab, B) in zip(axes.ravel(), panels):
        m, eps, name = CELLS[lab]
        ens = fk.load_ensemble(name)
        w = [1.0 / m] * m
        wi = np.asarray(ens.window_index)
        ew = ens.edges[wi]
        t = 0.5 * (ew[:-1] + ew[1:])
        bw = np.diff(ew)
        law = {v: fk.exact_law(ens, B, w, variant=v)["bin_mass"] for v in ("full", "meancontact")}
        fe = fk.free_exposure_discrete(ens.spec, w, gate="contact")
        lam_f = np.concatenate([[0.0], np.cumsum(fe["G"]) * ens.spec.dt])[ens.cps][wi]
        fm = fk.free_exposure_discrete(ens.spec, w, gate="mean")
        lam_m = np.concatenate([[0.0], np.cumsum(fm["G"]) * ens.spec.dt])[ens.cps][wi]
        clock = mfb.clock_for(m, eps, tuple(w))
        lam_c = clock.lam[clock.edge_index]
        comp = {"total": law["full"] - mf_bins(lam_c, B),
                "gating_full_minus_mc": law["full"] - law["meancontact"],
                "exposure_mc_minus_f1mc": law["meancontact"] - mf_bins(lam_m, B),
                "dt_f1full_minus_f1cont": mf_bins(lam_f, B) - mf_bins(lam_c, B)}
        for k, c in comp.items():
            ax.plot(t, c / bw, color=cols[k], lw=1.3 if k == "total" else 1.0,
                    ls="-" if k != "dt_f1full_minus_f1cont" else "--", label=labs[k])
        ax.axhline(0, color="0.6", lw=0.5)
        for tj in ens.spec.times():
            ax.axvline(tj, color="0.7", lw=0.6, ls=":")
        ax.set_title(f"$m$ = {m}, $\\varepsilon$ = {eps:g}, $B$ = {B:g}")
        ax.set_xlim(0.5, 3.5)
    for ax in axes[1]:
        ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density difference ($\\gamma$ units)")
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
               columnspacing=1.2)
    core.stream_tag(fig, "fb N2")
    written += core.save_figure(fig, fk.FIGURES / "fb_n2_residual_decomposition")
    plt.close(fig)
    print("\n".join(written))



def cmd_replicate(args) -> None:
    """Criterion-2 cross-check on independent path sets: gated-kernel EM-exact Var X vs the
    sample variance of the N1 ensembles (tag 81, 5e5 paths, independent of the tag-80 N0 paths)."""
    out = {}
    for lab, name in (("m2_eps0.05", "n1_m2_eps0.05"), ("m2_eps0.1", "n1_m2_eps0.1"),
                      ("m3_eps0.1", "n1_m3_eps0.1"), ("m3_eps0.15", "n1_m3_eps0.15")):
        ens = fk.load_ensemble(name)
        m = ens.spec.centres().size
        w = [1.0 / m] * m
        em = em_var_exposure(ens.spec, w, "contact", cached_c2_table(ens.spec))
        vq = np.concatenate([[0.0], em["var_X"]])[ens.cps]
        mom = fk.exposure_moments(ens, w)
        wi = np.asarray(ens.window_index)
        vs, se, q = mom["var_X"][wi], mom["var_X_se_batch"][wi], vq[wi]
        ok = (se > 0) & (q > 1e-6 * q.max())
        z = (vs[ok] - q[ok]) / se[ok]
        ci = geometric_cut_indices(ens)
        out[lab] = {"ensemble": name, "n_paths": ens.n_paths, "tag": ens.index["tag"],
                    "seed_entropy": ens.index["seed_entropy"],
                    "max_abs_z_window": float(np.max(np.abs(z))), "n_abs_z_gt_3": int(np.sum(np.abs(z) > 3)),
                    "n_points": int(ok.sum()),
                    "z_at_cuts": ((mom["var_X"][ci] - vq[ci]) / mom["var_X_se_batch"][ci]).tolist()[1:],
                    "max_abs_rel_diff": float(np.max(np.abs(vs[ok] - q[ok]) / q[ok]))}
        print(lab, out[lab]["max_abs_z_window"], flush=True)
    core.write_json(OUT_DIR / "n2_var_replicate.json", fk._jsonable(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--paths", default="2e5")
    s.add_argument("--workers", type=int, default=3)
    a = sub.add_parser("analyze")
    a.add_argument("--only", default="")
    a.add_argument("--no-c2", action="store_true")
    sub.add_parser("assemble")
    sub.add_parser("figure")
    sub.add_parser("replicate")
    args = ap.parse_args()
    if args.cmd == "simulate":
        cmd_simulate(args)
    else:
        globals()[f"cmd_{args.cmd}"](args)
