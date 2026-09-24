#!/usr/bin/env python3
"""N14 fixer checks (deterministic, no random numbers) for
manuscript/cnsns_submission/theory/TH13_universality.tex, Remarks th13:rem-counterex and
th13:rem-check-mc.

1. euler: time-discretisation bias of the Monte Carlo driver code/fb_n14_general_mc.py.
   That driver uses Euler--Maruyama with dt = eps*rho/(30 v_max), i.e. dt is proportional to
   eps, so the deterministic Euler error O(dt) = O(eps) does NOT vanish on the rescaled
   scale (Z - phi)/eps.  We run the noise-free recurrence (sigma = 0, xi = 0) on exactly the
   grids of the Monte Carlo runs (models A, B, C) and record
     * the scaled offset bias  n_j.(Z^Euler_k - phi(t_k))/eps  at the sampling index of t_j
       (to be compared with the recorded offset_mean of the MC runs);
     * the unit-budget passage exposure X_j = int_{s_{j-1}}^{s_j} kappa_j/(B w_j) dt with the
       driver's trapezoidal rule along the Euler path, against the same quantity along an
       accurately integrated path (RK4 with step dt/64) -> leading-order discretisation bias
       of the mean passage exposure, delta mu_j = B w_j (X_j^Euler - X_j^ref) (equal weights);
     * the resulting deterministic mass bias max_j |M_j(Euler exposures) - M_j(ref exposures)|.
2. corehalo: two-maximum beta-intervals of the core+halo passage profile (q, s1, s2) =
   (0.5, 0.3, 3) on the beta x theta grid of code/fb_n14_unimodality.py, together with the
   sub-interval on which the valley between the two maxima is at least 1% deep
   (valley / lower maximum <= 0.99).

Output: artifacts/data/exact_m_fixed_budget/N14_universality/n14_fix_checks.json
Run:    python3 code/fb_n14_fix_checks.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb_n14_general_mc as mc  # noqa: E402
import fb_n14_unimodality as um  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "artifacts", "data", "exact_m_fixed_budget", "N14_universality", "n14_fix_checks.json")


def drift_fn(spec):
    nm = spec["name"]
    if nm in ("A", "H"):
        return lambda z: np.array([spec["U"]])
    if nm == "B":
        return lambda z: -(z + z ** 3)
    om = spec["omega"]
    return lambda z: np.array([-z[0] - om * z[1], -z[1] + om * z[0]])


def unit_rates_fn(spec, psi_name, eps):
    """Same unit-budget stripe rates as fb_n14_general_mc.simulate.unit_rates (one path)."""
    pdf, _ = mc.PROFILES[psi_name]
    nm, rho = spec["name"], spec["rho"]

    def f(z):
        if nm in ("A", "H"):
            return np.array([pdf((z[0] - xj) / (eps * rho)) / (eps * rho) / spec["Aarea"] for xj in spec["xs"]])
        if nm == "B":
            return np.array([pdf((z[0] - zj) / (eps * rho)) / (eps * rho) / spec["Aarea"] for zj in spec["zs"]])
        r = np.hypot(z[0], z[1])
        out = []
        for j, rj in enumerate(spec["rs"]):
            ell = (r * r - rj * rj) / (2 * rj)
            cosang = (z[0] * np.cos(spec["thetas"][j]) + z[1] * np.sin(spec["thetas"][j])) / max(r, 1e-300)
            eta = (1 + spec["alpha"] * cosang) / (2 * np.pi * rj)
            out.append(pdf(ell / (eps * rho)) / (eps * rho) * eta)
        return np.array(out)

    return f


def exposures_on_path(path, dt, rates, cut_idx):
    """Trapezoidal unit exposures X_j(cut) along a discrete path (list of states)."""
    m = len(cut_idx) - 1
    X = np.zeros(m)
    Xcut = np.zeros((m, m + 1))
    ci = 1
    r_prev = rates(path[0])
    for k in range(1, len(path)):
        r = rates(path[k])
        X += 0.5 * dt * (r + r_prev)
        r_prev = r
        while ci <= m and cut_idx[ci] == k:
            Xcut[:, ci] = X
            ci += 1
    return Xcut


def euler_check(model, psi_name, eps, n_per_width=30, refine=64):
    spec = mc.model_spec(model)
    geo = mc.geometry(spec)
    rho, T = spec["rho"], spec["T"]
    dt = eps * rho / (n_per_width * geo["v"].max())
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps
    tj = geo["t"]
    m = len(tj)
    cuts = np.concatenate([[0.0], 0.5 * (tj[:-1] + tj[1:]), [T]])
    cut_idx = np.clip(np.round(cuts / dt).astype(int), 0, nsteps)
    tj_idx = np.round(tj / dt).astype(int)
    b = drift_fn(spec)
    if model in ("A", "H"):
        z0 = np.array([0.0])
    elif model == "B":
        z0 = np.array([spec["z0"]])
    else:
        z0 = np.array([spec["r0"], 0.0])
    # Euler path on the MC grid
    eul = [z0.copy()]
    z = z0.copy()
    for _ in range(nsteps):
        z = z + b(z) * dt
        eul.append(z.copy())
    # reference path: RK4 with step dt/refine, sampled on the fine grid
    h = dt / refine
    ref = [z0.copy()]
    z = z0.copy()
    for _ in range(nsteps * refine):
        k1 = b(z); k2 = b(z + 0.5 * h * k1); k3 = b(z + 0.5 * h * k2); k4 = b(z + h * k3)
        z = z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ref.append(z.copy())
    rates = unit_rates_fn(spec, psi_name, eps)
    Xe = exposures_on_path(eul, dt, rates, cut_idx)
    Xr = exposures_on_path(ref, h, rates, cut_idx * refine)
    Xe_j = np.array([Xe[j, j + 1] - Xe[j, j] for j in range(m)])
    Xr_j = np.array([Xr[j, j + 1] - Xr[j, j] for j in range(m)])
    # scaled offsets at the sampling index of t_j (reference path at the same grid time)
    offs = []
    for j in range(m):
        k = tj_idx[j]
        ze, zr = eul[k], ref[k * refine]
        nvec = zr / np.linalg.norm(zr) if model == "C" else np.array([1.0])
        offs.append(float(nvec @ (ze - zr) / eps))
    w = np.full(m, 1.0 / m)
    Bv = spec["B"]
    lam_e, lam_r = Bv * w * Xe_j, Bv * w * Xr_j
    Me, _ = mc.stick_breaking(lam_e)
    Mr, _ = mc.stick_breaking(lam_r)
    lam_exact = Bv * w / geo["cost"]
    return dict(model=model, psi=psi_name, eps=eps, dt=dt, nsteps=nsteps,
                scaled_offset_bias=offs,
                unit_exposure_euler=Xe_j.tolist(), unit_exposure_ref=Xr_j.tolist(),
                rel_exposure_bias=(Xe_j / Xr_j - 1).tolist(),
                ref_minus_exact_lambda=(lam_r - lam_exact).tolist(),
                delta_mu_equal=(lam_e - lam_r).tolist(),
                mass_bias_equal=float(np.max(np.abs(Me - Mr))))


def corehalo_intervals():
    fam = um.FAMILIES["corehalo_q0.5_s0.3_S3"]()
    U, h = fam["U"], fam["h"]
    u = np.arange(-U, U + h / 2, h)
    psi, Psi = fam["pdf"](u), fam["cdf"](u)
    out = {}
    for th in um.THETAS:
        two, deep = [], []
        cells = []
        for beta in um.BETAS:
            p = beta * psi * np.exp(-beta * Psi)
            F = p if th == 0 else um.fft_convolve(p, th, h)
            Fp = np.concatenate([[0.0], F, [0.0]])
            im = um.count_maxima(Fp) - 1
            dip = None
            if im.size >= 2:
                i1, i2 = im[0], im[1]
                jmin = i1 + int(np.argmin(F[i1:i2 + 1]))
                dip = float(F[jmin] / min(F[i1], F[i2]))
                two.append(float(beta))
                if dip <= 0.99:
                    deep.append(float(beta))
            cells.append(dict(beta=float(beta), num_maxima=int(im.size), dip_ratio=dip))
        out[str(th)] = dict(two_maxima_beta=[min(two), max(two)] if two else None,
                            valley_ge_1pct_beta=[min(deep), max(deep)] if deep else None,
                            n_two=len(two), n_deep=len(deep), cells=cells)
    return out


def main():
    t0 = time.time()
    res = dict(script="code/fb_n14_fix_checks.py", random_numbers=False, euler=[], corehalo=None)
    runs = [("A", "logistic", [0.1, 0.05, 0.025, 0.0125]),
            ("B", "gauss", [0.1, 0.05, 0.025, 0.0125]),
            ("B", "logistic", [0.025, 0.0125]),
            ("C", "gauss", [0.1, 0.05, 0.025, 0.0125])]
    for model, psi, epss in runs:
        for eps in epss:
            r = euler_check(model, psi, eps)
            res["euler"].append(r)
            print(model, psi, eps, "off", np.round(r["scaled_offset_bias"], 5), "dmu",
                  np.round(r["delta_mu_equal"], 6), "relX", np.round(r["rel_exposure_bias"], 6),
                  "refminusexact", np.round(r["ref_minus_exact_lambda"], 8), "Mbias", round(r["mass_bias_equal"], 7),
                  flush=True)
    res["corehalo"] = corehalo_intervals()
    for th, v in res["corehalo"].items():
        print("corehalo theta", th, v["two_maxima_beta"], v["valley_ge_1pct_beta"], v["n_two"], v["n_deep"])
    res["runtime_s"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", OUT, round(res["runtime_s"], 1), "s")


if __name__ == "__main__":
    main()
