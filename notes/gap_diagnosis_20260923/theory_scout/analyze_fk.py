#!/usr/bin/env python3
"""Analysis of FK path ensembles (theory scout): H1 mass law, limit bump shape, mode counts."""
import json
import math
import sys

import numpy as np

GAMMA, D0, Z0, W, RHO = 1.0, 1.0, 4.0, 1.0, 1.0
SZ = math.sqrt(D0 / (2 * GAMMA))
TARGETS = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}


def vel(t):
    return GAMMA * Z0 * math.exp(-GAMMA * t)  # |mu'(t)|


def ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def limit_shape(tau, lam, v, rho=RHO, s=SZ):
    """g(tau) = lam*v*int phi_rho(y) e^{-lam Phi_rho(y)} phi_s(v tau - y) dy (mass 1-e^{-lam})."""
    y = np.linspace(-12 * max(rho, s), 12 * max(rho, s), 6001)
    dy = y[1] - y[0]
    h = lam * np.exp(-0.5 * (y / rho) ** 2) / (math.sqrt(2 * math.pi) * rho) * np.exp(-lam * ncdf(y / rho))
    out = np.empty_like(tau)
    for i, ta in enumerate(tau):
        out[i] = v * np.sum(h * np.exp(-0.5 * ((v * ta - y) / s) ** 2)) * dy / (math.sqrt(2 * math.pi) * s)
    return out


def limit_masses(m, B, w=None, chi=None):
    w = np.full(m, 1.0 / m) if w is None else np.asarray(w)
    lam = np.array([B * w[j] / (W * vel(tj)) for j, tj in enumerate(TARGETS[m])])
    if chi is None:
        cum = np.concatenate([[0.0], np.cumsum(lam)])
        return lam, np.exp(-cum[:-1]) * (1 - np.exp(-lam))
    # frozen contact: chi (N x m) indicators
    ex = chi * lam[None, :]
    cum = np.concatenate([np.zeros((chi.shape[0], 1)), np.cumsum(ex, axis=1)], axis=1)
    return lam, np.mean(np.exp(-cum[:, :-1]) * (1 - np.exp(-ex)), axis=0)


def exact_basin_masses(lamck, B):
    L = lamck.astype(float)
    S = np.exp(-B * L)  # survival at checkpoints
    Sm = S.mean(0)
    # basins: between successive checkpoints (0.5, valleys..., 3.5); last ck (tmax) dropped
    return Sm[:-2] - Sm[1:-1], Sm


def meanfield_basin_masses(lamck, B):
    Lm = lamck.astype(float).mean(0)
    S = np.exp(-B * Lm)
    return S[:-2] - S[1:-1]


def smooth(y, h_steps):
    if h_steps < 1:
        return y.copy()
    k = np.arange(-int(5 * h_steps), int(5 * h_steps) + 1)
    ker = np.exp(-0.5 * (k / h_steps) ** 2)
    ker /= ker.sum()
    return np.convolve(y, ker, mode="same")


def count_modes(t, f, fb, window=(0.5, 3.5), h_steps=3, zsig=5.0):
    """Local maxima of smoothed density; prominence (vs higher-side contour base) + MC significance."""
    fs = smooth(f, h_steps)
    nb = fb.shape[0]
    fbs = np.array([smooth(x, h_steps) for x in fb])
    se = fbs.std(axis=0, ddof=1) / math.sqrt(nb)
    msk = (t >= window[0]) & (t <= window[1])
    idx = np.flatnonzero(msk)
    y = fs[idx]
    maxima = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]
    gmax = y.max()
    res = []
    for i in maxima:
        # topographic prominence: min over both sides down to higher terrain
        left = y[:i][::-1]; right = y[i + 1:]
        def base(side):
            hi = np.flatnonzero(side > y[i])
            seg = side if hi.size == 0 else side[:hi[0]]
            return seg.min() if seg.size else y[i]
        b = max(base(left), base(right))
        prom = y[i] - b
        sig = prom / max(se[idx[i]], 1e-300)
        res.append(dict(t=float(t[idx[i]]), height=float(y[i]), rel_height=float(y[i] / gmax),
                        prom=float(prom), rel_prom=float(prom / gmax), z=float(sig)))
    return res


def main(path):
    d = np.load(path)
    m = int(d["m"]); eps = float(d["eps"]); dt = float(d["dt"]); t = d["tgrid"]
    budgets = d["budgets"]
    out = dict(m=m, eps=eps, N=int(d["N"]), dt=dt, checkpoints=d["checkpoints"].tolist())
    # --- exposure concentration per slab ---
    conc = {}
    for key in ("free", "full"):
        sx = d[f"slabexp_{key}"].astype(float)
        lam1 = np.array([(1.0 / m) / (W * vel(tj)) for tj in TARGETS[m]])
        r = sx / lam1[None, :]
        conc[key] = dict(mean_ratio=r.mean(0).tolist(), std_ratio=r.std(0).tolist(),
                         frac_below_half=(r < 0.5).mean(0).tolist())
    out["exposure_ratio_to_limit"] = conc
    out["pred_rel_std_free"] = [0.5311 * math.sqrt(eps * D0 / (RHO * vel(tj))) for tj in TARGETS[m]]
    chi = d["chi_tj"].astype(float)
    out["contact_prob_at_tj"] = chi.mean(0).tolist()
    # --- basin masses ---
    Bgrid = [0.5, 1, 2, 4, 8, 20]
    masses = []
    for B in Bgrid:
        row = dict(B=B)
        for key in ("free", "full"):
            ex, Sm = exact_basin_masses(d[f"lamck_{key}"], B)
            mf = meanfield_basin_masses(d[f"lamck_{key}"], B)
            row[f"exact_{key}"] = ex.tolist()
            row[f"mf_{key}"] = mf.tolist()
        lam, p = limit_masses(m, B)
        row["limit_c1"] = p.tolist()
        _, pchi = limit_masses(m, B, chi=chi)
        row["limit_frozen_contact"] = pchi.tolist()
        row["lambda_c1"] = lam.tolist()
        masses.append(row)
    out["basin_masses"] = masses
    # --- modes of exact FK density per budget ---
    modes = {}
    vmax = max(vel(tj) for tj in TARGETS[m])
    width_t = eps * math.sqrt(SZ**2 + RHO**2) / vmax  # narrowest bump sd in time
    h_steps = max(1.0, 0.15 * width_t / dt)
    for key in ("free", "full"):
        FK = d[f"FK_{key}"] / dt; FKb = d[f"FKb_{key}"] / dt
        modes[key] = {}
        for bi, B in enumerate(budgets):
            res = count_modes(t, FK[bi], FKb[:, bi, :], h_steps=h_steps)
            modes[key][str(float(B))] = res
    out["modes"] = modes
    out["h_steps"] = h_steps
    # --- peak shifts vs limit shape (free kernel) ---
    shifts = []
    for B in (1.0, 2.0, 8.0):
        bi = int(np.flatnonzero(np.isclose(budgets, B))[0])
        f = smooth(d["FK_free"][bi] / dt, h_steps)
        lam, _ = limit_masses(m, B)
        row = dict(B=B, peaks=[])
        cks = d["checkpoints"]
        for j, tj in enumerate(TARGETS[m]):
            lo, hi = cks[j], cks[j + 1]
            msk = (t > lo) & (t < hi)
            tt = t[msk]; ff = f[msk]
            tpk = float(tt[np.argmax(ff)])
            tau = np.linspace(-6, 3, 1801)
            g = limit_shape(tau, lam[j], vel(tj))
            taustar = float(tau[np.argmax(g)])
            row["peaks"].append(dict(j=j, t_target=tj, t_peak_exact=tpk, shift_exact=tpk - tj,
                                     shift_limit=eps * taustar / 1.0, lam=float(lam[j]),
                                     peak_height_exact=float(ff.max()),
                                     peak_height_limit=float(math.exp(-sum(lam[:j])) * g.max() / eps)))
        shifts.append(row)
    out["peak_shifts_free"] = shifts
    return out


if __name__ == "__main__":
    res = [main(p) for p in sys.argv[1:]]
    print(json.dumps(res, indent=1))
