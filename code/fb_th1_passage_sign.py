#!/usr/bin/env python3
"""TH-1 audit fix (GPT-6 finding TH1-F1): sign of the exact-minus-mean-field density inside a passage.

Deterministic quadrature, no random numbers.  In the fixed-budget small-noise limit (TH-3) the
rescaled passage-j densities are, in the local coordinate y0 = v_j s / rho (s = (t - t_j)/eps) and up to
the common factor Pi_j v_j / rho,

    exact:       F_ex(y0) = lambda * E[ phi(Y) exp(-lambda Phi(Y)) ],        Y ~ N(y0, vartheta^2),
    mean field:  F_mf(y0) = lambda * E[phi(Y)] * exp(-lambda E[Phi(Y)])
                          = p_lambda(y0 / sqrt(1+vartheta^2)) / sqrt(1+vartheta^2),

with p_lambda(u) = lambda phi(u) exp(-lambda Phi(u)).  Their difference is
    F_ex - F_mf = -lambda^2 exp(-lambda E Phi(Y)) Cov(phi(Y), Phi(Y)) + (one-signed remainder >= 0),
the limiting form of Proposition th1:density.  This script tabulates, for several lambda at the anchor
vartheta^2 = s_Z^2/rho^2 = 1/2:
  * c(y0) = Cov(phi(Y), Phi(Y)) (odd in y0), e.g. c(-1), c(+1), c(-0.1);
  * F_ex, F_mf and their difference on a y0 grid, the sign changes of F_ex - F_mf on the grid and
    the values at y0 = -0.1 and 0 (GPT-6 counterexample: at lambda = 1 the exact density exceeds the
    mean field at y0 = -0.1 although c(-0.1) > 0; at y0 = 0, c = 0 but F_ex > F_mf by strict Jensen);
  * the local CDF order 1 - E exp(-lambda Phi(Y)) <= 1 - exp(-lambda E Phi(Y)) (Jensen), i.e. the mean
    field front-loads every passage in the sense of stochastic order, and the masses of both profiles.
All grid statements are finite-grid observations (step 0.001 on [-6, 6]), not certified root counts.

Output: artifacts/data/exact_m_fixed_budget/TH_core/th1_passage_sign.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
OUT_JSON = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "TH_core" / "th1_passage_sign.json"

_erf = np.vectorize(math.erf, otypes=[float])


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / math.sqrt(2.0)))


def profiles(lam: float, theta2: float, y0: np.ndarray, n_gh: int = 160) -> dict:
    """Exact and mean-field limiting passage densities and c(y0) on the grid y0 (Gauss-Hermite in Y)."""
    x, wgt = np.polynomial.hermite_e.hermegauss(n_gh)
    wgt = wgt / math.sqrt(2.0 * math.pi)  # E g(N) = sum wgt g(x)
    th = math.sqrt(theta2)
    Y = y0[:, None] + th * x[None, :]
    ph, PH = phi(Y), Phi(Y)
    e_ph = (ph * wgt).sum(axis=1)
    e_PH = (PH * wgt).sum(axis=1)
    e_ph_PH = (ph * PH * wgt).sum(axis=1)
    e_ph_exp = (ph * np.exp(-lam * PH) * wgt).sum(axis=1)
    e_exp = (np.exp(-lam * PH) * wgt).sum(axis=1)
    cov = e_ph_PH - e_ph * e_PH
    f_ex = lam * e_ph_exp
    f_mf = lam * e_ph * np.exp(-lam * e_PH)
    lead = -lam**2 * np.exp(-lam * e_PH) * cov
    return {"f_ex": f_ex, "f_mf": f_mf, "cov": cov, "lead": lead,
            "surv_ex": e_exp, "surv_mf": np.exp(-lam * e_PH)}


def main() -> int:
    theta2 = 0.5  # anchor: s_Z^2 / rho^2 = (D0 / (2 gamma)) / rho^2 with gamma = D0 = rho = 1
    dy = 1e-3
    y0 = np.round(np.arange(-6.0, 6.0 + dy / 2, dy), 10)
    # closed-form cross-check of the mean-field profile
    s = math.sqrt(1.0 + theta2)
    rows = []
    for lam in (0.01, 0.1, 0.5, 1.0, 2.0, math.e, 6.795704571147613):
        pr = profiles(lam, theta2, y0)
        diff = pr["f_ex"] - pr["f_mf"]
        sg = np.sign(diff)
        nz = np.flatnonzero(sg != 0)
        chg = [float(0.5 * (y0[nz[i]] + y0[nz[i + 1]])) for i in range(nz.size - 1) if sg[nz[i]] != sg[nz[i + 1]]]
        mf_closed = lam * phi(y0 / s) * np.exp(-lam * Phi(y0 / s)) / s
        i0 = int(np.argmin(np.abs(y0 - 0.0)))
        im = int(np.argmin(np.abs(y0 + 0.1)))
        ip1 = int(np.argmin(np.abs(y0 - 1.0)))
        im1 = int(np.argmin(np.abs(y0 + 1.0)))
        cdf_gap = (1.0 - pr["surv_mf"]) - (1.0 - pr["surv_ex"])  # >= 0 by Jensen
        rows.append({
            "lambda": lam,
            "mass_exact_profile": float(np.trapezoid(pr["f_ex"], y0)),
            "mass_mean_field_profile": float(np.trapezoid(pr["f_mf"], y0)),
            "mass_expected_1_minus_exp_minus_lambda": float(-math.expm1(-lam)),
            "max_abs_mf_vs_closed_form": float(np.max(np.abs(pr["f_mf"] - mf_closed))),
            "cov_at_y0_minus1": float(pr["cov"][im1]), "cov_at_y0_plus1": float(pr["cov"][ip1]),
            "cov_at_y0_minus0p1": float(pr["cov"][im]), "cov_at_y0_0": float(pr["cov"][i0]),
            "f_exact_at_y0_minus0p1": float(pr["f_ex"][im]), "f_mf_at_y0_minus0p1": float(pr["f_mf"][im]),
            "f_exact_at_y0_0": float(pr["f_ex"][i0]), "f_mf_at_y0_0": float(pr["f_mf"][i0]),
            "exact_minus_mf_at_y0_minus0p1": float(diff[im]), "exact_minus_mf_at_y0_0": float(diff[i0]),
            "leading_term_at_y0_minus0p1": float(pr["lead"][im]),
            "sign_changes_exact_minus_mf_on_grid": chg,
            "n_sign_changes_on_grid": len(chg),
            "min_local_cdf_gap_mf_minus_exact": float(cdf_gap.min()),
            "max_local_cdf_gap_mf_minus_exact": float(cdf_gap.max()),
            "argmax_exact": float(y0[int(np.argmax(pr["f_ex"]))]),
            "argmax_mean_field": float(y0[int(np.argmax(pr["f_mf"]))]),
        })
        print(f"lambda={lam:.4g}: changes {chg}  diff(-0.1)={diff[im]:+.3e}  diff(0)={diff[i0]:+.3e}  "
              f"cov(-1)={pr['cov'][im1]:.7f}", flush=True)
    payload = {
        "schema_version": 1,
        "item": "TH-1 fix TH1-F1: exact vs mean-field limiting passage density (deterministic quadrature)",
        "script": "code/fb_th1_passage_sign.py",
        "vartheta2": theta2,
        "y0_grid": [float(y0[0]), float(y0[-1]), dy],
        "gauss_hermite_nodes": 160,
        "seeds": "none (deterministic)",
        "note": ("Densities in the local coordinate y0 = v_j s / rho, up to the common factor Pi_j v_j / rho. "
                 "Sign changes are finite-grid observations. lambda = e and 2.5 e are the first-passage "
                 "exposures lambda_1 = B w_1 e^{t_1}/4 at (m, B) = (2, 8) and (2, 20) of the anchor "
                 "(w_1 = 1/2, t_1 = 1); see TH2 Remark th2:shape."),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
