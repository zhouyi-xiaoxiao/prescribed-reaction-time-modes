#!/usr/bin/env python3
"""Mean-field level-set threshold asymptotics (theory item TH-5): deterministic check.

The mean-field hazard-survival law f1 = B G exp(-B Lambda), Lambda' = G, obeys
f1' = B e^{-B Lambda} G^2 (r - B) with r = G'/G^2, so its interior maxima are the
downward crossings of the level B by r.  On a rising flank of the free clock G
between gap j (centres c_j < c_{j+1}, weights w_j, w_{j+1}) the maximum of r gives

  ln B_top^mf = Delta^2/(8 sigma^2) + ln(1/eps) + C0 + C1 eps^2 - ln c_eps(t*) + O(eps^4),

  C0 = ln[ x'(tbar) Delta ell0^2 W^(d-1) sqrt(2 pi) / (8 S sqrt(w_j w_{j+1})) ],
  C1 = (S^2/(ell0^2 Delta^2)) K,  K = (l + u*)^2/2 - (2 sqrt 2 + gamma Delta/x'(tbar)) (l + u*),
  u* = 2 asinh(1),  l = ln(w_j/w_{j+1}),  S^2 = D0/(2 gamma) + rho^2,  sigma = eps S/ell0,
  tbar: x(tbar) = (c_j + c_{j+1})/2,  t*: x(t*) = xi + sigma^2 u*/Delta,
  xi = (c_j + c_{j+1})/2 + (sigma^2/Delta) l   (weighted equality point),

and B_top^mf is the minimum of this level over the interior gaps.

This script (no randomness):
  1. reads the stored D1 output artifacts/data/exact_m_prr_upgrade/mean_field_topology.json
     (bisected B_top^mf with the contact factor; level-set cross-check);
  2. evaluates C0, C1 and the contact correction -ln c_eps(t*) with the frozen contact
     quadrature of validate_exact_m_offlattice.py (d=2) or exact_m_prr_upgrade_core (d=3);
  3. recomputes the contact-free (c = 1) threshold max_flank r exactly in the log domain,
     to isolate the O(eps^2) and O(eps^4) terms of the expansion;
  4. collects the diagnostics quoted in TH-5 (prominence at B_op, survival at B_top^mf).

Output: artifacts/data/exact_m_fixed_budget/TH5/meanfield_levelset_asymptotics.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import validate_exact_m_offlattice as base  # noqa: E402

D1_JSON = core.UPGRADE_DATA / "mean_field_topology.json"
OUT_JSON = core.REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "TH5" / "meanfield_levelset_asymptotics.json"
U_STAR = 2.0 * math.asinh(1.0)


def gap_constants(p, eps: float, centres_x, weights, j: int) -> dict:
    """Asymptotic constants for the rising flank of component j+1 (0-based gap j)."""
    s2 = p.d0 / (2.0 * p.gamma) + p.rho**2
    s = math.sqrt(s2)
    sigma = eps * s / p.ell0
    area = p.torus_w ** (p.dim - 1)
    cj, ck = centres_x[j], centres_x[j + 1]
    delta = ck - cj
    lw = math.log(weights[j] / weights[j + 1])
    mid = 0.5 * (cj + ck)
    x_inf = base.sign_mu_prime(p) * p.z_bar / p.ell0
    xprime_mid = p.gamma * (x_inf - mid)  # x'(t) = gamma (x_inf - x) for the OU mean
    c0 = math.log(xprime_mid * delta * p.ell0**2 * area * math.sqrt(2.0 * math.pi)
                  / (8.0 * s * math.sqrt(weights[j] * weights[j + 1])))
    kk = 0.5 * (lw + U_STAR) ** 2 - (2.0 * math.sqrt(2.0) + p.gamma * delta / xprime_mid) * (lw + U_STAR)
    c1 = s2 / (p.ell0**2 * delta**2) * kk
    xi = mid + sigma**2 / delta * lw
    x_star = xi + sigma**2 * U_STAR / delta
    # invert x(t) = sgn(mu') mu(t)/ell0  ->  mu(t*) = sgn * ell0 * x*
    mu_star = base.sign_mu_prime(p) * p.ell0 * x_star
    t_star = -math.log((mu_star - p.z_bar) / (p.z0 - p.z_bar)) / p.gamma
    lead = delta**2 / (8.0 * sigma**2) + math.log(1.0 / eps)
    return {
        "gap_index_1based": j + 1, "Delta": delta, "ln_w_ratio": lw, "xprime_mid": xprime_mid,
        "C0": c0, "K": kk, "C1": c1, "t_star": t_star, "leading_ln": lead,
        "pred_ln_C0": lead + c0, "pred_ln_C0_C1": lead + c0 + c1 * eps**2,
    }


def contact_at(p, eps: float, t: float, n_perp: int) -> float:
    ts = np.array([t])
    if n_perp == 1:
        return float(base.contact_probability(ts, eps, p)[0])
    return float(core.contact_probability_d3(ts, eps, p)[0])


def contact_free_threshold(p, eps: float, centres_x, weights, j: int) -> dict:
    """max over the rising flank of component j+1 of r = G'/G^2 with c = 1 (log domain)."""
    s = math.sqrt(p.d0 / (2.0 * p.gamma) + p.rho**2)
    sigma = eps * s / p.ell0
    area = p.torus_w ** (p.dim - 1)
    a_const = 1.0 / (area * math.sqrt(2.0 * math.pi) * eps * s)
    cx = np.asarray(centres_x, float)
    w = np.asarray(weights, float)
    delta = cx[j + 1] - cx[j]
    xs = np.linspace(0.5 * (cx[j] + cx[j + 1]) - 0.25 * delta, cx[j + 1], 400001)
    e = np.log(w)[:, None] - (xs[None, :] - cx[:, None]) ** 2 / (2.0 * sigma**2)
    emax = e.max(axis=0)
    pw = np.exp(e - emax)
    hs = pw.sum(axis=0)
    lslope = (pw * (cx[:, None] - xs[None, :])).sum(axis=0) / hs / sigma**2  # H'/H
    x_inf = base.sign_mu_prime(p) * p.z_bar / p.ell0
    xprime = p.gamma * (x_inf - xs)
    ok = lslope > 0.0
    logr = np.full(xs.shape, -np.inf)
    logr[ok] = np.log(xprime[ok] * lslope[ok]) - math.log(a_const) - (emax[ok] + np.log(hs[ok]))
    k = int(np.argmax(logr))
    # parabolic refinement on the grid
    if 0 < k < xs.size - 1:
        y0, y1, y2 = logr[k - 1], logr[k], logr[k + 1]
        den = y0 - 2.0 * y1 + y2
        peak = y1 - 0.125 * (y0 - y2) ** 2 / den if den != 0 else y1
    else:
        peak = logr[k]
    n_local_max = int(np.sum((logr[1:-1] > logr[:-2]) & (logr[1:-1] >= logr[2:]) & np.isfinite(logr[1:-1])))
    return {"ln_max_r_contact_free": float(peak), "u_at_max": float((xs[k] - 0.5 * (cx[j] + cx[j + 1])) * delta / sigma**2),
            "n_local_maxima_of_r_on_flank": n_local_max}


def main() -> int:
    d1 = json.loads(D1_JSON.read_text())
    rows = []
    for cfg in d1["configurations"]:
        if cfg.get("status") != "bisected":
            rows.append({"label": cfg["label"], "status": cfg.get("status"), "skipped": True})
            continue
        eps = float(cfg["eps"])
        p = core.make_model(z0=float(cfg["z0"]))
        n_perp = int(cfg["n_perp"])
        if n_perp == 2:
            p = core.make_model(z0=float(cfg["z0"]), dim=3)
        cx = [float(v) for v in cfg["centres_x"]]
        w = [float(v) for v in cfg["weights"]]
        m = len(cx)
        gaps = []
        for j in range(m - 1):
            g = gap_constants(p, eps, cx, w, j)
            c_star = contact_at(p, eps, g["t_star"], n_perp)
            g["contact_at_t_star"] = c_star
            g["pred_ln_full"] = g["pred_ln_C0_C1"] - math.log(c_star)
            g.update(contact_free_threshold(p, eps, cx, w, j))
            gaps.append(g)
        jstar = min(range(m - 1), key=lambda j: gaps[j]["pred_ln_full"])
        gb = gaps[jstar]
        ln_b = float(cfg["log10_B_top_mf"]) * math.log(10.0)
        resid = ln_b - gb["leading_ln"]
        diag = cfg["diagnostics_at_threshold"]
        lmp = diag.get("last_mode_relative_prominence", {})
        lsc = cfg["level_set_cross_check"]
        d1_flank_max = [
            {"rising_flank_index": r["rising_flank_index"], "time": r["time"], "ln_r": math.log(r["r_parabolic"])}
            for r in lsc["local_maxima_of_r"]
        ]
        rows.append({
            "label": cfg["label"], "m": m, "eps": eps, "dim": p.dim, "z0": p.z0, "weights": w,
            "d1_log10_B_top_mf": cfg["log10_B_top_mf"], "d1_ln_B_top_mf": ln_b,
            "d1_first_loss_time": cfg["first_loss"]["lost_maximum_time"],
            "d1_level_set_rel_diff": lsc["relative_difference_level_set_vs_bisection"],
            "d1_rising_flank_r_maxima": d1_flank_max,
            "binding_gap_1based_predicted": jstar + 1,
            "binding_flank_d1": lsc["predicted_first_loss_flank"],
            "gaps": gaps,
            "residual_d1": resid,
            "residual_minus_C0": resid - gb["C0"],
            "residual_minus_C0_C1": resid - gb["C0"] - gb["C1"] * eps**2,
            "residual_minus_full_prediction": ln_b - gb["pred_ln_full"],
            "contact_free_residual": gb["ln_max_r_contact_free"] - gb["leading_ln"],
            "contact_free_minus_C0_C1": gb["ln_max_r_contact_free"] - gb["pred_ln_C0_C1"],
            "t_star_predicted": gb["t_star"],
            "prominence_at_B_op_stored": lmp.get("B_op_stored", {}).get("relative_prominence"),
            "prominence_at_B_op_mf_stored": lmp.get("B_op_mf_stored", {}).get("relative_prominence"),
            "log10_survival_at_last_valley_at_B_top_mf": diag.get("log10_survival_at_last_valley_at_B_top_mf"),
            "stored_B_op": cfg["stored_comparators"].get("B_op"),
            "stored_B_op_mf": cfg["stored_comparators"].get("B_op_mf"),
        })

    def pick(label):
        return next(r for r in rows if r["label"] == label)

    slope = {}
    for m in (2, 3):
        a, b = pick(f"m{m}_eps0.05"), pick(f"m{m}_eps0.1")
        g = a["gaps"][a["binding_gap_1based_predicted"] - 1]
        s2 = 1.5
        slope[f"m{m}"] = {
            "d1_slope_lnB_vs_inv_eps2_between_0.05_and_0.1": (a["d1_ln_B_top_mf"] - b["d1_ln_B_top_mf"]) / (1 / 0.05**2 - 1 / 0.1**2),
            "Delta2_over_8S2": g["Delta"] ** 2 / (8.0 * s2),
        }
    acceptance = {}
    for m, tol in ((2, 0.1), (3, 0.11)):
        devs = {r["label"]: abs(r["residual_minus_C0"]) for r in rows
                if not r.get("skipped") and r["m"] == m and r["dim"] == 2 and r["z0"] == 4.0 and r["eps"] <= 0.1 + 1e-12}
        acceptance[f"m{m}_max_abs_residual_minus_C0_eps_le_0.1"] = {"values": devs, "max": max(devs.values()),
                                                                   "tolerance": tol, "pass": max(devs.values()) <= tol}
    payload = {
        "schema_version": 1,
        "item": "TH-5 mean-field level-set threshold asymptotics (deterministic check of stored D1 output)",
        "source": str(D1_JSON.relative_to(core.REPORT)),
        "formula": __doc__.split("\n\n")[1],
        "u_star": U_STAR,
        "rows": rows,
        "slope_check": slope,
        "acceptance": acceptance,
        "not_a_theorem_about_exact_process": "B_top^mf is a threshold of the mean-field law f1 only; it is not the exact-process B_top",
    }
    core.write_json(OUT_JSON, payload)
    for r in rows:
        if r.get("skipped"):
            print(r["label"], "skipped", r["status"])
            continue
        print(f"{r['label']:>20s} gap*={r['binding_gap_1based_predicted']} (D1 flank {r['binding_flank_d1']}) "
              f"resid={r['residual_d1']:+.4f}  -C0={r['residual_minus_C0']:+.4f}  -C0-C1e2={r['residual_minus_C0_C1']:+.4f}  "
              f"-full={r['residual_minus_full_prediction']:+.4f}  c(t*)={r['gaps'][r['binding_gap_1based_predicted']-1]['contact_at_t_star']:.4f} "
              f"t*={r['t_star_predicted']:.4f} (D1 {r['d1_first_loss_time']})  cfree-C0C1={r['contact_free_minus_C0_C1']:+.2e}")
    print(json.dumps(slope, indent=1))
    print(json.dumps(acceptance, indent=1))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
