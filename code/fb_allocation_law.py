#!/usr/bin/env python3
"""Fixed-budget allocation law (theory item TH-4): deterministic evaluation.

Small-noise limit law of the basin masses at fixed budget B (TH-2):

    lambda_j = B w_j / (W^(d-1) v_j),     v_j = |mu'(t_j)| = gamma |z0 - zbar| e^(-gamma t_j),
    M_j      = exp(-sum_{i<j} lambda_i) (1 - exp(-lambda_j)).

Inverse design (TH-4): for target masses p_j > 0 with sum_j p_j < 1, put
S_j = 1 - sum_{i<j} p_i (S_1 = 1).  Then

    lambda_j = ln(S_j / S_{j+1}),   B = W^(d-1) sum_j v_j lambda_j,   w_j = W^(d-1) v_j lambda_j / B.

Max-min optimum: the equal-mass level p*(B, m) solves
    B = W^(d-1) sum_j v_j ln[(1 - (j-1) p) / (1 - j p)],
and more generally the largest common scale theta*(B, r) of a target shape r
(r_j > 0, sum r_j = 1) solves B = C(theta r).  Both are computed here in the
variable u = 1 - theta (so that tiny deficits near theta = 1 are resolved).

Outputs (no randomness; nothing is simulated):
  artifacts/data/exact_m_fixed_budget/allocation_law.json
  artifacts/figures/fb_allocation_law.{png,pdf}   (with --figure)

Reference numbers reproduced (asserted):
  GPT-6 (referee_gpt6.md, "illustrative new calculation"): m=3, B=1:
      p* = 0.247968, w_opt = (0.512193, 0.323096, 0.164711);
      equal weights -> masses (0.169279, 0.280928, 0.410140).
  Opus (opus_referee_scripts/equal_mass_design.py): m=3 p*(B=0.5,1,2) =
      0.148 / 0.248 / 0.328 and ~1/3 at B>=4; designed weights
      m=2,B=8: (0.127, 0.873); m=3,B=4: (0.182, 0.14, 0.678);
      m=5 (z0=8), B=1: (0.296, 0.26, 0.215, 0.155, 0.073), p* = 0.1004.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_upgrade_core as core  # noqa: E402  (anchor MODEL / TARGET_TIMES)

OUT_DIR = core.REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
OUT_JSON = OUT_DIR / "allocation_law.json"
FIG_STEM = core.FIGURES / "fb_allocation_law"

# W4 stretched geometry (exact_m_prr_upgrade_w4.py): z0 = 8, physical centres.
W4_Z0 = 8.0
W4_CENTRES = (2.8, 2.2, 1.6, 1.0, 0.4)

B_TABLE = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 50.0)
B_CURVE = tuple(10 ** (k / 20.0) for k in range(-40, 41))  # 1e-2 .. 1e2


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def speeds(target_times, p=core.MODEL) -> list[float]:
    """Crossing speeds v_j = |mu'(t_j)| = gamma |z0 - zbar| exp(-gamma t_j)."""
    return [p.gamma * abs(p.z0 - p.z_bar) * math.exp(-p.gamma * t) for t in target_times]


def area_factor(p=core.MODEL) -> float:
    """A = W^(d-1): transverse normalisation of the killing field."""
    return p.torus_w ** (p.dim - 1)


# --------------------------------------------------------------------------
# Forward and inverse stick-breaking law
# --------------------------------------------------------------------------


def exposures(budget: float, weights, v, area: float) -> list[float]:
    return [budget * w / (area * vj) for w, vj in zip(weights, v)]


def masses_from_exposures(lams) -> list[float]:
    out, log_surv = [], 0.0
    for lam in lams:
        out.append(math.exp(-log_surv) * (-math.expm1(-lam)))
        log_surv += lam
    return out


def masses(budget: float, weights, v, area: float) -> list[float]:
    return masses_from_exposures(exposures(budget, weights, v, area))


def inverse_design(target, v, area: float) -> dict:
    """Exposures, budget and allocation that realise the target masses exactly."""
    if min(target) <= 0.0 or sum(target) >= 1.0:
        raise ValueError("target masses must be positive with sum < 1")
    surv, lams = 1.0, []
    for pj in target:
        lams.append(-math.log1p(-pj / surv))  # ln(S_j / S_{j+1})
        surv -= pj
    budget = area * sum(vj * lam for vj, lam in zip(v, lams))
    weights = [area * vj * lam / budget for vj, lam in zip(v, lams)]
    return {"exposures": lams, "budget": budget, "weights": weights}


def budget_of_shape(u: float, shape, v, area: float) -> float:
    """C(theta r) with theta = 1 - u, evaluated stably for tiny u.

    S_j = 1 - theta R_{j-1} = (1 - R_{j-1}) + u R_{j-1},  R_k = r_1 + ... + r_k,
    so S_{m+1} = u exactly when R_m = 1 (no cancellation for tiny u).
    """
    total, cum = 0.0, 0.0
    for rj, vj in zip(shape, v):
        cum_next = cum + rj
        s_j = (1.0 - cum) + u * cum
        s_next = max(1.0 - cum_next, 0.0) + u * cum_next
        total += vj * math.log(s_j / s_next)
        cum = cum_next
    return area * total


def max_common_scale(budget: float, shape, v, area: float) -> dict:
    """Largest theta with theta*shape attainable at this budget (bisection in ln u)."""
    if abs(sum(shape) - 1.0) > 1e-12:
        raise ValueError("shape must sum to one")
    lo, hi = -700.0, 0.0  # ln u; C is decreasing in u
    if budget_of_shape(math.exp(lo), shape, v, area) < budget:
        raise ValueError("budget beyond double-precision range")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if budget_of_shape(math.exp(mid), shape, v, area) > budget:
            lo = mid
        else:
            hi = mid
    u = math.exp(0.5 * (lo + hi))
    theta = 1.0 - u
    target = [theta * rj for rj in shape]
    # Inverse design evaluated in the stable u-parametrisation (identical to
    # inverse_design(target) whenever theta < 1 is representable).
    lams, cum = [], 0.0
    for rj in shape:
        cum_next = cum + rj
        lams.append(math.log(((1.0 - cum) + u * cum) / (max(1.0 - cum_next, 0.0) + u * cum_next)))
        cum = cum_next
    bud = area * sum(vj * lam for vj, lam in zip(v, lams))
    weights = [area * vj * lam / bud for vj, lam in zip(v, lams)]
    return {"theta": theta, "deficit_1_minus_theta": u, "masses": target,
            "exposures": lams, "budget": bud, "weights": weights}


def p_star(budget: float, v, area: float) -> dict:
    m = len(v)
    res = max_common_scale(budget, [1.0 / m] * m, v, area)
    res["p_star"] = res["theta"] / m
    res["deficit_1_over_m_minus_p_star"] = res["deficit_1_minus_theta"] / m
    return res


def cost_gradient(target, v, area: float) -> list[float]:
    """dC/dp_k = A [v_k / S_{k+1} + sum_{j>k} v_j (1/S_{j+1} - 1/S_j)] (all > 0)."""
    m = len(target)
    surv = [1.0]
    for pj in target:
        surv.append(surv[-1] - pj)  # surv[j] = S_{j+1} in 1-based notation
    grad = []
    for k in range(m):
        g = v[k] / surv[k + 1]
        for j in range(k + 1, m):
            g += v[j] * (1.0 / surv[j + 1] - 1.0 / surv[j])
        grad.append(area * g)
    return grad


def small_budget_asymptote(budget: float, v, area: float) -> float:
    v1 = sum(v)
    v2 = sum(vj * (2 * (j + 1) - 1) / 2.0 for j, vj in enumerate(v))
    x = budget / (area * v1)
    return x - (v2 / v1) * x * x


def large_budget_deficit_asymptote(budget: float, v, area: float) -> float:
    """1/m - p* ~ (1/m^2) exp(-(B/A - K_m)/v_m), K_m = sum_{j<m} v_j ln((m-j+1)/(m-j))."""
    m = len(v)
    k_m = sum(v[j] * math.log((m - j) / (m - j - 1.0)) for j in range(m - 1))
    return math.exp(-(budget / area - k_m) / v[-1]) / m**2


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def geometry_block(name: str, target_times, v, area: float, z0: float) -> dict:
    return {
        "name": name,
        "z0": z0,
        "target_times": list(target_times),
        "speeds_v": v,
        "budget_price_per_unit_log_survival": [area * vj for vj in v],
        "area_factor_W_pow_d_minus_1": area,
    }


def equal_weight_row(budget, v, area):
    m = len(v)
    w = [1.0 / m] * m
    lam = exposures(budget, w, v, area)
    ms = masses_from_exposures(lam)
    return {"weights": w, "exposures": lam, "masses": ms, "min_mass": min(ms),
            "total_mass": sum(ms)}


def optimum_row(budget, v, area):
    res = p_star(budget, v, area)
    return {
        "budget": budget,
        "p_star": res["p_star"],
        "deficit_1_over_m_minus_p_star": res["deficit_1_over_m_minus_p_star"],
        "total_mass_m_p_star": res["theta"],
        "w_opt": res["weights"],
        "exposures_opt": res["exposures"],
        "budget_identity_residual": res["budget"] - budget,
        "forward_check_masses": masses(budget, res["weights"], v, area),
        "small_budget_asymptote": small_budget_asymptote(budget, v, area),
        "large_budget_deficit_asymptote": large_budget_deficit_asymptote(budget, v, area),
        "equal_weights": equal_weight_row(budget, v, area),
    }


def reachable_curve_m2(budget, v, area, n=101):
    rows = []
    for k in range(1, n):
        w1 = k / n
        ms = masses(budget, [w1, 1.0 - w1], v, area)
        rows.append({"w1": w1, "M1": ms[0], "M2": ms[1]})
    lam_max1 = budget / (area * v[0])
    return {
        "budget": budget,
        "parametrisation": "w1 in (0,1); hypersurface v1 ln(1/(1-M1)) + v2 ln((1-M1)/(1-M1-M2)) = B/A",
        "samples": rows,
        "endpoint_w1_to_1": {"M1": -math.expm1(-lam_max1), "M2": 0.0},
        "endpoint_w1_to_0": {"M1": 0.0, "M2": -math.expm1(-budget / (area * v[1]))},
    }


def reachable_surface_m3(budget, v, area, step=0.05):
    rows = []
    n = int(round(1.0 / step))
    for a in range(1, n):
        for b in range(1, n - a):
            w = [a * step, b * step, 1.0 - (a + b) * step]
            ms = masses(budget, w, v, area)
            surv = [1.0, 1.0 - ms[0], 1.0 - ms[0] - ms[1], 1.0 - sum(ms)]
            resid = area * sum(v[j] * math.log(surv[j] / surv[j + 1]) for j in range(3)) - budget
            rows.append({"w": w, "M": ms, "budget_identity_residual": resid})
    return {"budget": budget, "grid_step": step, "samples": rows}


# --------------------------------------------------------------------------
# Reference checks
# --------------------------------------------------------------------------


def run_checks(g2, g3, g5) -> dict:
    checks = {}
    v2, v3, v5, a = g2["speeds_v"], g3["speeds_v"], g5["speeds_v"], g2["area_factor_W_pow_d_minus_1"]

    r = p_star(1.0, v3, a)
    ew = masses(1.0, [1 / 3] * 3, v3, a)
    ew2 = masses(1.0, [0.5, 0.5], v2, a)
    checks["gpt6_m3_B1"] = {
        "p_star": r["p_star"], "expected": 0.247968,
        "w_opt": r["weights"], "expected_w_opt": [0.512193, 0.323096, 0.164711],
        "equal_weight_masses": ew, "expected_equal_weight_masses": [0.169279, 0.280928, 0.410140],
        "pass": (abs(r["p_star"] - 0.247968) < 5e-7
                 and all(abs(x - y) < 5e-7 for x, y in zip(r["weights"], [0.512193, 0.323096, 0.164711]))
                 and all(abs(x - y) < 5e-7 for x, y in zip(ew, [0.169279, 0.280928, 0.410140]))),
    }
    checks["operator_m2_B1_equal_weights"] = {
        "masses": ew2, "expected": [0.288077, 0.556654],
        "pass": all(abs(x - y) < 5e-7 for x, y in zip(ew2, [0.288077, 0.556654])),
    }
    opus = {0.5: 0.148, 1.0: 0.248, 2.0: 0.328}
    got = {b: p_star(b, v3, a)["p_star"] for b in opus}
    checks["opus_m3_p_star"] = {
        "p_star": {str(b): got[b] for b in opus}, "expected_3dp": {str(b): opus[b] for b in opus},
        "p_star_B4": p_star(4.0, v3, a)["p_star"], "p_star_B8": p_star(8.0, v3, a)["p_star"],
        "pass": all(round(got[b], 3) == opus[b] for b in opus) and round(p_star(4.0, v3, a)["p_star"], 2) == 0.33,
    }
    w_m2_b8 = p_star(8.0, v2, a)["weights"]
    w_m3_b4 = p_star(4.0, v3, a)["weights"]
    r5 = p_star(1.0, v5, a)
    checks["opus_designed_weights"] = {
        "m2_B8": w_m2_b8, "expected_m2_B8": [0.127, 0.873],
        "m3_B4": w_m3_b4, "expected_m3_B4": [0.182, 0.14, 0.678],
        "m5_B1": r5["weights"], "expected_m5_B1": [0.296, 0.26, 0.215, 0.155, 0.073],
        "m5_B1_p_star": r5["p_star"], "expected_m5_B1_p_star": 0.1004,
        "pass": (all(abs(x - y) < 6e-4 for x, y in zip(w_m2_b8, [0.127, 0.873]))
                 and all(abs(x - y) < 6e-3 for x, y in zip(w_m3_b4, [0.182, 0.14, 0.678]))
                 and all(abs(x - y) < 6e-3 for x, y in zip(r5["weights"], [0.296, 0.26, 0.215, 0.155, 0.073]))
                 and abs(r5["p_star"] - 0.1004) < 5e-5),
    }
    # Monotonicity and limits of p*(B, m); positivity of the cost gradient.
    mono = {}
    for name, v in (("m2", v2), ("m3", v3), ("m5_z0_8", v5)):
        rows = [p_star(b, v, a) for b in B_CURVE]
        vals = [r["p_star"] for r in rows]
        deficits = [r["deficit_1_over_m_minus_p_star"] for r in rows]  # 1/m - p*, resolved to ~1e-300
        grad_ok = all(
            min(cost_gradient(p_star(b, v, a)["masses"], v, a)) > 0.0 for b in (0.1, 1.0, 10.0)
        )
        big = [(b, d, large_budget_deficit_asymptote(b, v, a)) for b, d in zip(B_CURVE, deficits) if b >= 20.0]
        mono[name] = {
            "deficit_strictly_decreasing_on_B_curve": all(y < x for x, y in zip(deficits, deficits[1:])),
            "p_star_nondecreasing_in_double_precision": all(y >= x for x, y in zip(vals, vals[1:])),
            "p_star_at_B_1e-2": vals[0],
            "small_budget_asymptote_at_B_1e-2": small_budget_asymptote(B_CURVE[0], v, a),
            "small_budget_relative_error_at_B_1e-2": abs(vals[0] / small_budget_asymptote(B_CURVE[0], v, a) - 1.0),
            "deficit_at_B_100": deficits[-1],
            "large_budget_deficit_asymptote_at_B_100": big[-1][2],
            "large_budget_asymptote_ratio_B_20_to_100": [d / asy for _, d, asy in big],
            "cost_gradient_positive_at_optimum": grad_ok,
        }
    checks["monotonicity_and_limits"] = mono
    return checks


def build_payload() -> dict:
    p = core.MODEL
    a = area_factor(p)
    t2, t3 = core.TARGET_TIMES[2], core.TARGET_TIMES[3]
    v2, v3 = speeds(t2, p), speeds(t3, p)
    p5 = core.make_model(z0=W4_Z0)
    t5 = tuple(math.log(W4_Z0 / c) for c in W4_CENTRES)
    v5 = speeds(t5, p5)
    g2 = geometry_block("anchor m=2", t2, v2, a, p.z0)
    g3 = geometry_block("anchor m=3", t3, v3, a, p.z0)
    g5 = geometry_block("W4 stretched m=5 (z0=8)", t5, v5, area_factor(p5), W4_Z0)

    payload = {
        "schema_version": 1,
        "item": "TH-4 fixed-budget allocation law (small-noise limit; deterministic)",
        "not_simulated": "all numbers are closed-form evaluations of the TH-2 limit law; no random numbers",
        "model": core.model_payload(p),
        "formulas": {
            "exposure": "lambda_j = B w_j / (W^(d-1) v_j), v_j = gamma |z0 - zbar| exp(-gamma t_j)",
            "masses": "M_j = exp(-sum_{i<j} lambda_i) (1 - exp(-lambda_j))",
            "inverse": "S_j = 1 - sum_{i<j} p_i; lambda_j = ln(S_j/S_{j+1}); B = W^(d-1) sum_j v_j lambda_j; w_j = W^(d-1) v_j lambda_j / B",
            "equal_mass_budget": "B = W^(d-1) sum_j v_j ln[(1-(j-1)p)/(1-jp)]",
            "small_budget": "p* = x - (V2/V1) x^2 + O(x^3), x = B/(W^(d-1) V1), V1 = sum v_j, V2 = sum v_j (2j-1)/2",
            "large_budget": "1/m - p* ~ m^-2 exp(-(B/W^(d-1) - K_m)/v_m), K_m = sum_{j<m} v_j ln((m-j+1)/(m-j))",
        },
        "geometries": {"m2": g2, "m3": g3, "m5_z0_8": g5},
        "optimum_table": {
            "m2": [optimum_row(b, v2, a) for b in B_TABLE],
            "m3": [optimum_row(b, v3, a) for b in B_TABLE],
            "m5_z0_8": [optimum_row(b, v5, a) for b in (0.5, 1.0, 2.0, 4.0)],
        },
        "p_star_curve": {
            "budgets": list(B_CURVE),
            "m2": [p_star(b, v2, a)["p_star"] for b in B_CURVE],
            "m3": [p_star(b, v3, a)["p_star"] for b in B_CURVE],
            "m5_z0_8": [p_star(b, v5, a)["p_star"] for b in B_CURVE],
            "m3_w_opt": [p_star(b, v3, a)["weights"] for b in B_CURVE],
            "m3_equal_weight_min_mass": [min(masses(b, [1 / 3] * 3, v3, a)) for b in B_CURVE],
            "m2_equal_weight_min_mass": [min(masses(b, [0.5, 0.5], v2, a)) for b in B_CURVE],
        },
        "weighted_maxmin_N1_shapes": {
            "m2_shape_0.2_0.8": [
                {"budget": b, **max_common_scale(b, [0.2, 0.8], v2, a)} for b in (0.5, 1.0, 2.0, 4.0, 8.0)
            ],
            "m3_shape_0.5_0.3_0.2": [
                {"budget": b, **max_common_scale(b, [0.5, 0.3, 0.2], v3, a)} for b in (0.5, 1.0, 2.0, 4.0, 8.0)
            ],
        },
        "reachable_hypersurface": {
            # B = 0.5 appended (final fix pass, 2026-09-23) for Fig. 2(c) of the CNSNS paper,
            # which plots B in {0.5, 1, 8}: at B = 4 the curve and the max-min square coincide
            # with those of B = 8, and at B = 2 the equal-weight point, the max-min square
            # and the B = 8 square overlap; appended last so that the campaign figure
            # (first three rows) is unchanged.
            "m2": [reachable_curve_m2(b, v2, a) for b in (1.0, 4.0, 8.0, 0.5)],
            "m3": [reachable_surface_m3(b, v3, a) for b in (1.0, 4.0)],
            "total_mass_range_open_interval": {
                name: {str(b): [-math.expm1(-b / (a * max(v))), -math.expm1(-b / (a * min(v)))]
                       for b in (1.0, 4.0, 8.0, 0.5)}
                for name, v in (("m2", v2), ("m3", v3))
            },
        },
    }
    payload["checks"] = run_checks(g2, g3, g5)
    payload["all_checks_pass"] = all(
        c.get("pass", True) for c in payload["checks"].values() if isinstance(c, dict) and "pass" in c
    ) and all(
        all(v for k, v in row.items() if isinstance(v, bool))
        for row in payload["checks"]["monotonicity_and_limits"].values()
    )
    return payload


def make_figure(payload: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core.apply_prr_style()
    curve = payload["p_star_curve"]
    b = curve["budgets"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), layout="constrained")
    ax = axes[0]
    for key, col, m, lab in (("m2", core.OI_BLUE, 2, "$m=2$"), ("m3", core.OI_VERMILLION, 3, "$m=3$"),
                             ("m5_z0_8", core.OI_GREEN, 5, "$m=5$ ($z_0=8$)")):
        ax.plot(b, [m * y for y in curve[key]], color=col, label=lab)
    ax.plot(b, [3 * y for y in curve["m3_equal_weight_min_mass"]], color=core.OI_VERMILLION, ls="--", lw=0.9,
            label="$m=3$, equal $w$: $3\\,\\mathrm{min}_j M_j$")
    ax.set_xscale("log")
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("$m\\,p^*(B,m)$")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", frameon=True, fontsize=6.5)
    ax = axes[1]
    wopt = curve["m3_w_opt"]
    for j, col in enumerate((core.OI_BLUE, core.OI_ORANGE, core.OI_VERMILLION)):
        ax.plot(b, [w[j] for w in wopt], color=col, label=f"$w_{j + 1}$")
    ax.axhline(1 / 3, color=core.OI_GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("max–min weights ($m=3$)")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left")
    ax = axes[2]
    for curve_row, col in zip(payload["reachable_hypersurface"]["m2"], (core.OI_BLUE, core.OI_ORANGE, core.OI_VERMILLION)):
        m1 = [r["M1"] for r in curve_row["samples"]]
        m2 = [r["M2"] for r in curve_row["samples"]]
        ax.plot(m1, m2, color=col, label=f"$B={curve_row['budget']:g}$")
        eq = masses(curve_row["budget"], [0.5, 0.5], payload["geometries"]["m2"]["speeds_v"],
                    payload["geometries"]["m2"]["area_factor_W_pow_d_minus_1"])
        ax.plot([eq[0]], [eq[1]], marker="o", ms=3, color=col)
        ps = p_star(curve_row["budget"], payload["geometries"]["m2"]["speeds_v"],
                    payload["geometries"]["m2"]["area_factor_W_pow_d_minus_1"])["p_star"]
        ax.plot([ps], [ps], marker="s", ms=3, color=col, mfc="white")
    ax.plot([0, 0.5], [0, 0.5], color=core.OI_GREY, lw=0.6, ls=":")
    ax.set_xlabel("$M_1$")
    ax.set_ylabel("$M_2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", title="reachable ($m=2$)")
    return core.save_figure(fig, FIG_STEM)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figure", action="store_true", help="also write fb_allocation_law.{png,pdf}")
    args = ap.parse_args()
    payload = build_payload()
    if args.figure:
        payload["figure"] = make_figure(payload)
    core.write_json(OUT_JSON, payload)
    chk = payload["checks"]
    print("GPT-6 m3 B1: p* = %.6f  w_opt = %s  equal-weight masses = %s" % (
        chk["gpt6_m3_B1"]["p_star"], [round(x, 6) for x in chk["gpt6_m3_B1"]["w_opt"]],
        [round(x, 6) for x in chk["gpt6_m3_B1"]["equal_weight_masses"]]))
    print("Opus m3 p*(0.5,1,2) =", {k: round(v, 4) for k, v in chk["opus_m3_p_star"]["p_star"].items()},
          " p*(4) = %.4f" % chk["opus_m3_p_star"]["p_star_B4"])
    print("Opus designed weights:", {k: [round(x, 4) for x in v] if isinstance(v, list) else v
                                     for k, v in chk["opus_designed_weights"].items() if not k.startswith("expected")})
    print("all_checks_pass =", payload["all_checks_pass"])
    print("wrote", OUT_JSON)
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
