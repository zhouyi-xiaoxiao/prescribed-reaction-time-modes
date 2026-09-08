#!/usr/bin/env python3
"""Semi-analytic preflight for the exact-m PRR upgrade campaign.

No simulation.  Uses the spine free-exposure factorization to
  (a) count G window maxima across the W1 epsilon grid (theory boundary),
  (b) vet W4 m=5 stretched-window candidate designs including a first-order
      depletion estimate f1(t) = B G(t) exp(-B Lambda(t)) and the classifier
      relative-floor risk on the last peak,
  (c) quote the d=3 contact factor / event-rate estimate for W5,
  (d) list which W2 epsilon values retain m=3 G maxima.

Writes artifacts/data/exact_m_prr_upgrade/preflight_theory.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import validate_exact_m_offlattice as base

W1_EPS_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25)
W2_EPS_GRID = (0.05, 0.10, 0.15, 0.20)

W4_CANDIDATES = {
    "A_mu_2.8_to_0.4_step0.6": (2.8, 2.2, 1.6, 1.0, 0.4),
    "B_mu_2.4_to_0.4_step0.5": (2.4, 1.9, 1.4, 0.9, 0.4),
    "C_mu_3.0_to_0.6_step0.6": (3.0, 2.4, 1.8, 1.2, 0.6),
}
W4_Z0 = 8.0
W4_EPS = 0.10
W4_B = 1.0


def first_order_density(ts: np.ndarray, g: np.ndarray, budget: float) -> np.ndarray:
    lam = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(ts))))
    return budget * g * np.exp(-budget * lam)


def analyze_curve(ts: np.ndarray, vals: np.ndarray) -> dict:
    diag = core.g_stationary_diagnostics(ts, vals)
    peaks = diag["peaks"]
    out = {
        "n_window_maxima": diag["n_window_maxima"],
        "peak_times": [p["time"] for p in peaks],
        "peak_heights": [p["g"] for p in peaks],
    }
    if peaks:
        gmax = max(p["g"] for p in peaks)
        # prominence proxy for the last peak: height minus deepest valley
        # between it and its higher neighbourhood (use adjacent valley).
        last = peaks[-1]
        valleys = diag["valleys"]
        left_valleys = [v for v in valleys if v["time"] < last["time"]]
        saddle = left_valleys[-1]["g"] if left_valleys else 0.0
        out["last_peak_height_over_max"] = last["g"] / gmax
        out["last_peak_prominence_over_max"] = (last["g"] - saddle) / gmax
        out["valley_to_peak_ratios"] = [
            v["valley_to_peak_ratio"] for v in diag["valley_depths"]
        ]
    return out


def main() -> None:
    report: dict = {}

    # (a) + (d): G maxima across the eps grid, standard model.
    ts = np.linspace(base.THEORY_T_MIN, base.DEFAULT_TMAX, 1600)
    grid_rows = []
    for m, weights in ((2, (0.5, 0.5)), (3, (1 / 3, 1 / 3, 1 / 3))):
        for eps in W1_EPS_GRID:
            diag = core.g_window_maxima_count(m=m, eps=eps, weights=weights)
            grid_rows.append(
                {
                    "m": m,
                    "eps": eps,
                    "n_window_maxima": diag["n_window_maxima"],
                    "g_has_m_window_maxima": diag["g_has_m_window_maxima"],
                    "sigma_x_space": diag["sigma_x_space"],
                    "valley_to_peak_ratios": [
                        v["valley_to_peak_ratio"] for v in diag["valley_depths"]
                    ],
                }
            )
            print(
                f"G theory m={m} eps={eps:g}: maxima={diag['n_window_maxima']} "
                f"(want {m}) valley/peak="
                + ",".join(
                    f"{v['valley_to_peak_ratio']:.3f}" for v in diag["valley_depths"]
                )
            )
    report["w1_theory_grid"] = grid_rows
    report["w2_m3_eps_retaining_3_maxima"] = [
        row["eps"]
        for row in grid_rows
        if row["m"] == 3 and row["g_has_m_window_maxima"] and row["eps"] in W2_EPS_GRID
    ]

    # (b) W4 candidates.
    p4 = core.make_model(z0=W4_Z0)
    w4_rows = {}
    for name, mus in W4_CANDIDATES.items():
        times = tuple(float(np.log(W4_Z0 / mu)) for mu in mus)
        centres = core.centres_z_for(5, p=p4, target_times=times) if False else np.array(mus, float)
        theory = core.free_exposure_general(
            ts, centres_z=centres, eps=W4_EPS, weights=(0.2,) * 5, p=p4, n_perp=1
        )
        g_diag = analyze_curve(ts, theory["g"])
        f1 = first_order_density(ts, theory["g"], W4_B)
        f1_diag = analyze_curve(ts, f1)
        contact_check = core.deterministic_contact_check(p4)
        w4_rows[name] = {
            "mu_centres": list(mus),
            "target_times": list(times),
            "z0": W4_Z0,
            "eps": W4_EPS,
            "budget": W4_B,
            "g": g_diag,
            "first_order_depleted": f1_diag,
            "contact_interior_check": contact_check,
            "total_exposure_Lambda": float(
                np.trapezoid(theory["g"], ts)
            ),
        }
        print(
            f"W4 {name}: times={['%.3f' % t for t in times]} "
            f"G maxima={g_diag['n_window_maxima']} "
            f"f1 maxima={f1_diag['n_window_maxima']} "
            f"f1 last-peak prom/max={f1_diag.get('last_peak_prominence_over_max', 0):.3f} "
            f"Lambda={w4_rows[name]['total_exposure_Lambda']:.3f}"
        )
    report["w4_candidates"] = w4_rows

    # (c) W5 d=3 estimate at the anchor (m=2, eps=0.1, B=1).
    theory_d2 = core.free_exposure_general(
        ts, centres_z=core.centres_z_for(2), eps=0.10, weights=(0.5, 0.5), n_perp=1
    )
    theory_d3 = core.free_exposure_general(
        ts, centres_z=core.centres_z_for(2), eps=0.10, weights=(0.5, 0.5), n_perp=2
    )
    lam_d2 = float(np.trapezoid(theory_d2["g"], ts))
    lam_d3 = float(np.trapezoid(theory_d3["g"], ts))
    d3_diag = analyze_curve(ts, theory_d3["g"])
    f1_d3 = analyze_curve(ts, first_order_density(ts, theory_d3["g"], 1.0))
    report["w5_d3_estimate"] = {
        "eps": 0.10,
        "budget": 1.0,
        "Lambda_total_d2": lam_d2,
        "Lambda_total_d3": lam_d3,
        "contact_factor_ratio_d3_over_d2_at_t2.5": float(
            theory_d3["contact_factor"][np.searchsorted(ts, 2.5)]
            / theory_d2["contact_factor"][np.searchsorted(ts, 2.5)]
        ),
        "first_order_kill_fraction_at_B1_d3": float(1.0 - np.exp(-lam_d3)),
        "g_d3": d3_diag,
        "f1_d3": f1_d3,
    }
    print(
        f"W5 d=3: Lambda_d3={lam_d3:.3f} (d2 {lam_d2:.3f}); first-order kill "
        f"fraction at B=1 ~ {1.0 - np.exp(-lam_d3):.3f}; "
        f"G maxima={d3_diag['n_window_maxima']}"
    )

    out = core.UPGRADE_DATA / "preflight_theory.json"
    core.write_json(out, report)
    print(f"preflight -> {out}")


if __name__ == "__main__":
    main()
