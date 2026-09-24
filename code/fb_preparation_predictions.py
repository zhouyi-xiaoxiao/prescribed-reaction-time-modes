#!/usr/bin/env python3
"""TH-10 predictions: preparation (initial midpoint variance) sensitivity.

General Gaussian preparation Z_0 ~ N(z0, eps^2 s0^2).  Then
Var Z_t = eps^2 s^2(t),  s^2(t) = aZ + (s0^2 - aZ) exp(-2 gamma t),  aZ = D0/(2 gamma).

Deterministic predictions for the preparation-sensitivity runs (item N4):
  * fixed-budget limit masses (independent of s0),
  * passage-profile width ratio theta_j^2 = s^2(t_j)/rho^2, the location y of the
    maximum of F_lambda = p_lambda * phi_theta, the predicted peak-time shift
    eps*rho*y/v_j and the predicted peak s.d. in time (profile s.d. * eps*rho/v_j),
  * the weighted-space admissibility of the preparation for the small-budget
    transfer of Theorem 1 (needs Var Z_t < 2 eps^2 aZ at the restart time),
  * the width-variation ratio omega_j = sup_{t in [t_j, t_{j+1}]}
    Delta_j |(log S)'(t)| / (2 x'(t)),  S^2 = s^2 + rho^2, which enters the
    interior-gap sector inequalities of the (sketched) small-budget count with a
    time-dependent mixture width (sufficient: omega < 2/3),
  * the outer-sector ratios omega_L = sup_{t in [tau, t_1]} |(log S)'(t)| |mu(t) - mu(t_1)| / |mu'(t)|
    and omega_R = sup_{t in [t_m, T]} |(log S)'(t)| |mu(t) - mu(t_m)| / |mu'(t)| (sufficient: < 1),
    which control the two window-end sectors not covered by omega.

Pure numpy, deterministic, seconds.  Run from code/:
    python3 fb_preparation_predictions.py
Output: ../artifacts/data/exact_m_fixed_budget/TH10/preparation_predictions.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GAMMA, D0, Z0, ZBAR, W, RHO = 1.0, 1.0, 4.0, 0.0, 1.0, 1.0
AZ = D0 / (2 * GAMMA)
TAU, TEND = 0.5, 3.5  # window I = [tau, T] of Theorem 1 at the anchor
TARGETS = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}
PREPS = {"point_release": 0.0, "stationary": AZ, "four_x_stationary": 4 * AZ}
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "TH10" / "preparation_predictions.json"

U = np.linspace(-14.0, 14.0, 28001)
DU = U[1] - U[0]
PHI_U = np.exp(-0.5 * U**2) / math.sqrt(2 * math.pi)
CDF_U = np.concatenate([[0.0], np.cumsum(0.5 * (PHI_U[1:] + PHI_U[:-1]) * DU)])


def s2(t, s0sq):
    return AZ + (s0sq - AZ) * math.exp(-2 * GAMMA * t)


def mu(t):
    return ZBAR + (Z0 - ZBAR) * math.exp(-GAMMA * t)


def vel(t):
    return GAMMA * abs(Z0 - ZBAR) * math.exp(-GAMMA * t)


def profile_stats(lam, theta):
    """F = p_lam * phi_theta on a y-grid: argmax y*, s.d. of F/(1-e^-lam)."""
    p = lam * PHI_U * np.exp(-lam * CDF_U)
    y = U
    # convolution with a centred Gaussian of s.d. theta (FFT-free direct sum on a coarse y grid)
    ker_x = np.arange(-int(8 * theta / DU) - 1, int(8 * theta / DU) + 2) * DU
    ker = np.exp(-0.5 * (ker_x / theta) ** 2)
    ker /= ker.sum()
    F = np.convolve(p, ker, mode="same")
    i = int(np.argmax(F))
    # parabolic refinement of the argmax
    if 0 < i < len(F) - 1:
        a, b, c = F[i - 1], F[i], F[i + 1]
        den = a - 2 * b + c
        ystar = y[i] + (0.5 * (a - c) / den * DU if den != 0 else 0.0)
    else:  # pragma: no cover
        ystar = y[i]
    mass = F.sum() * DU
    mean = (y * F).sum() * DU / mass
    sd = math.sqrt(((y - mean) ** 2 * F).sum() * DU / mass)
    return float(ystar), float(sd), float(mass)


def main():
    res = dict(script="code/fb_preparation_predictions.py", item="TH-10", deterministic=True, seeds=None,
               geometry=dict(gamma=GAMMA, D0=D0, z0=Z0, zbar=ZBAR, W=W, rho=RHO, aZ=AZ,
                             targets={str(k): v for k, v in TARGETS.items()}),
               preparations={k: dict(s0sq=v, var_Z0_over_eps2=v) for k, v in PREPS.items()})
    # admissibility for the weighted space X_pi (Theorem 1 transfer): Var Z_t / eps^2 < 2 aZ
    adm = {}
    for name, s0sq in PREPS.items():
        if s0sq < 2 * AZ:
            adm[name] = dict(admissible_at_t0=bool(s0sq > 0 or False), earliest_restart_time=0.0 if s0sq > 0 else "any t>0",
                             note="q0 in X_pi" if s0sq > 0 else "point mass: not in X_pi at t=0, admissible after any t>0")
        else:
            tadm = math.log((s0sq - AZ) / AZ) / (2 * GAMMA)
            adm[name] = dict(admissible_at_t0=False, earliest_restart_time=tadm,
                             note="free state enters X_pi only for t > earliest_restart_time")
    res["weighted_space_admissibility"] = adm
    rows = {}
    for m, tj in TARGETS.items():
        w = [1.0 / m] * m
        v = [vel(t) for t in tj]
        cz = [mu(t) for t in tj]
        for B in (1.0, 4.0):
            lam = [B * w[j] / (W * v[j]) for j in range(m)]
            cum = np.concatenate([[0.0], np.cumsum(lam)])
            masses = (np.exp(-cum[:-1]) * (1 - np.exp(-np.array(lam)))).tolist()
            for name, s0sq in PREPS.items():
                recs = []
                for j in range(m):
                    th = math.sqrt(s2(tj[j], s0sq)) / RHO
                    ystar, sd, mass = profile_stats(lam[j], th)
                    rec = dict(j=j + 1, t_j=tj[j], s2_tj=s2(tj[j], s0sq), theta=th, lam=lam[j],
                               y_star=ystar, profile_sd_y=sd, profile_mass=mass, profile_mass_exact=1 - math.exp(-lam[j]))
                    for eps in (0.1, 0.05):
                        rec[f"peak_shift_eps{eps}"] = eps * RHO * ystar / v[j]
                        rec[f"peak_sd_time_eps{eps}"] = eps * RHO * sd / v[j]
                    recs.append(rec)
                rows[f"m{m}_B{B:g}_{name}"] = dict(m=m, B=B, preparation=name, limit_masses=masses, passages=recs)
        # width-variation ratio omega_j on each gap
        om = {}
        for name, s0sq in PREPS.items():
            vals = []
            for j in range(m - 1):
                ts = np.linspace(tj[j], tj[j + 1], 20001)
                S2 = AZ + (s0sq - AZ) * np.exp(-2 * GAMMA * ts) + RHO**2
                dlogS = 0.5 * (-2 * GAMMA * (s0sq - AZ) * np.exp(-2 * GAMMA * ts)) / S2
                xp = GAMMA * abs(Z0 - ZBAR) * np.exp(-GAMMA * ts)
                delta = abs(cz[j] - cz[j + 1])
                vals.append(float(np.max(delta * np.abs(dlogS) / (2 * xp))))
            om[name] = dict(omega_per_gap=vals, omega_max=max(vals) if vals else 0.0)
        res[f"width_variation_ratio_m{m}"] = om
        # outer sectors [tau, t_1] and [t_m, T]: log-slope ~ (x'/sigma^2)(c_1 - x)[1 + kappa~ (c_1 - x)], kappa~ = (log S)'/x'
        outer = {}
        for name, s0sq in PREPS.items():
            recs = {}
            for side, (ta, tb, cref) in (("left", (TAU, tj[0], cz[0])), ("right", (tj[-1], TEND, cz[-1]))):
                ts = np.linspace(ta, tb, 20001)
                S2 = AZ + (s0sq - AZ) * np.exp(-2 * GAMMA * ts) + RHO**2
                dlogS = 0.5 * (-2 * GAMMA * (s0sq - AZ) * np.exp(-2 * GAMMA * ts)) / S2
                xp = GAMMA * abs(Z0 - ZBAR) * np.exp(-GAMMA * ts)
                dist = np.abs(ZBAR + (Z0 - ZBAR) * np.exp(-GAMMA * ts) - cref)
                val = np.abs(dlogS) * dist / xp
                k = int(np.argmax(val))
                recs[side] = dict(sup=float(val[k]), at_t=float(ts[k]), interval=[ta, tb])
            outer[name] = dict(omega_L=recs["left"]["sup"], omega_L_at=recs["left"]["at_t"],
                               omega_R=recs["right"]["sup"], omega_R_at=recs["right"]["at_t"],
                               sufficient_condition="omega_L < 1 and omega_R < 1")
        res[f"outer_sector_ratio_m{m}"] = outer
    res["predictions"] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
    print(json.dumps(res["weighted_space_admissibility"], indent=0))
    for m in (2, 3):
        print(m, res[f"width_variation_ratio_m{m}"])
    for k, r in rows.items():
        print(k, [round(x, 4) for x in r["limit_masses"]],
              [(round(p["theta"], 4), round(p["y_star"], 4), round(p["peak_shift_eps0.1"], 4), round(p["peak_sd_time_eps0.1"], 4))
               for p in r["passages"]])


if __name__ == "__main__":
    main()
