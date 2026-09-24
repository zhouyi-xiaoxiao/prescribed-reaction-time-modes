#!/usr/bin/env python3
"""Derived numbers quoted in the CNSNS manuscript that are not a single JSON value.

R1 fix editor (2026-09-23, referee finding O16).  Before this script, the src comments of
these numbers pointed at OUTLINE.md, at an old table of the PRR manuscript, at a notes
JSON and at temporary working notes.  This script recomputes each of them from the stored
script outputs (no new random numbers) and writes

    artifacts/data/exact_m_fixed_budget/derived_numbers.json

Quantities (keys of the JSON):
  pooled_B5_m2_eps0.1        inverse-variance pooled B_5% at (m, eps) = (2, 0.1) from the two
                             independent ensembles (N6 tag 80, N1 tag 81), with the
                             heterogeneity z, chi^2_1 and the Birge-inflated 95% interval
                             (main Table 2, Sec. 4.5; OUTLINE.md ruling R3).
  mean_field_over_exact_B5   B_top^mf / B_5% at (2, 0.1) [pooled], (3, 0.1) and (3, 0.15)
                             (main Sec. 1 item (iv), Sec. 4.5).
  contact_factor_anchor      c(t) = P(|R_t|_mi < a) of the continuum relative process at the
                             window ends tau = 0.5 and T = 3.5 at the eps = 0.1 anchor
                             (main Sec. 2.2; exact quadrature of the Gaussian law, as in
                             exact_m_prr_fk_exact_law.mean_contact_curve), and the
                             Euler--Maruyama chain value at the same steps.
  fk_per_path_variance_factor  (N_dk / N_fk) / (se_fk / se_dk)^2 with the median SE ratio of
                             the three primary validation cells (main Sec. 4.1).
  frozen_gate_equal_eps0.1   number of equal-weight cells at eps = 0.1 in which the frozen-gate
                             law with the measured contact states lowers the summed absolute
                             basin-mass deviation (main Sec. 4.2).
  n9a_sign_changes_total     total number of significant sign changes of f' (maxima + minima)
                             over the 72 census cases and five resolutions (main Sec. 4.6;
                             R3 finding O34).
  n4_width_ratio_max_abs_dev largest |measured - predicted| ratio of a passage's time standard
                             deviation to its stationary-preparation value, over the point-release
                             and four-times-stationary preparations (whole-axis basins; main
                             Sec. 4.7; R3 finding O34).

Usage: python3 code/fb_derived_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402

DATA = fk.FB_DATA
OUT = DATA / "derived_numbers.json"


def load(rel: str) -> dict:
    return json.loads((DATA / rel).read_text())


def n6_row(n6: dict, cell: str) -> dict:
    for r in n6["table"]:
        if r.get("cell") == cell:
            return r
    raise KeyError(cell)


def pooled_b5(n6: dict, n1: dict) -> dict:
    r6 = n6_row(n6, "m2_eps0.1")
    b1 = n1["ensembles"]["m2_eps0.1"]["equal_weight_B5"]
    x = np.array([r6["r_P_p5"], b1["B_p"]])
    se = np.array([r6["r_P_p5_se_jack"], b1["jackknife_se"]])
    wts = 1.0 / se**2
    mean = float(np.sum(wts * x) / np.sum(wts))
    se_plain = float(1.0 / math.sqrt(np.sum(wts)))
    chi2 = float(np.sum(wts * (x - mean) ** 2))
    z = float((x[1] - x[0]) / math.sqrt(np.sum(se**2)))
    birge = math.sqrt(chi2 / (x.size - 1))
    se_infl = se_plain * max(1.0, birge)
    half = 1.959963984540054 * se_infl
    return {
        "inputs": {
            "N6_tag80": {"value": float(x[0]), "se_jack": float(se[0]),
                         "src": "N6/n6_visibility_thresholds.json#table[cell=m2_eps0.1].{r_P_p5,r_P_p5_se_jack}"},
            "N1_tag81": {"value": float(x[1]), "se_jack": float(se[1]),
                         "src": "N1/n1_allocation_design.json#ensembles.m2_eps0.1.equal_weight_B5.{B_p,jackknife_se}"},
        },
        "pooled": mean,
        "se_plain": se_plain,
        "z_between_inputs": z,
        "chi2_1dof": chi2,
        "birge_ratio": birge,
        "se_birge_inflated": se_infl,
        "ci95_birge_inflated": [mean - half, mean + half],
        "method": "inverse-variance weighted mean; SE inflated by the Birge ratio sqrt(chi2/dof) because the "
                  "two independent ensembles differ by more than their SEs allow",
    }


def mean_field_ratios(i4: dict, n6: dict, pooled: float) -> dict:
    lg = i4["headline"]["log10_B_top_mf_by_label"]
    out = {}
    for cell, b5, b5src in (("m2_eps0.1", pooled, "pooled_B5_m2_eps0.1.pooled"),
                            ("m3_eps0.1", n6_row(n6, "m3_eps0.1")["r_P_p5"],
                             "N6/n6_visibility_thresholds.json#table[cell=m3_eps0.1].r_P_p5"),
                            ("m3_eps0.15", n6_row(n6, "m3_eps0.15")["r_P_p5"],
                             "N6/n6_visibility_thresholds.json#table[cell=m3_eps0.15].r_P_p5")):
        btop = 10.0 ** lg[cell]
        out[cell] = {"B_top_mf": btop, "B_5": float(b5), "ratio": btop / float(b5),
                     "src_B_top_mf": f"I4_mean_field_topology/mean_field_topology.json#headline.log10_B_top_mf_by_label.{cell}",
                     "src_B_5": b5src}
    return out


def contact_factor() -> dict:
    meta = json.loads((DATA / "fk_ensembles" / "n0_m3_eps0.1.json").read_text())
    spec = fk.EnsembleSpec.from_dict(meta["spec"])
    tt = np.array([0.5, 3.5])
    c_cont = fk.mean_contact_curve(spec, spec.contact_a, tt)
    c_em_all = fk.contact_probability_em(spec)
    steps = np.rint(tt / spec.dt).astype(int) - 1          # step n has time (n + 1) dt
    return {
        "spec_src": "fk_ensembles/n0_m3_eps0.1.json#spec",
        "eps": spec.eps, "contact_a": spec.contact_a, "r_par0": spec.r_par0, "r_perp0": spec.r_perp0,
        "u0": spec.u0, "sigma_perp0": spec.sigma_perp0, "torus_w": spec.torus_w,
        "t": tt.tolist(),
        "c_continuum": [float(v) for v in c_cont],
        "c_euler_maruyama_chain": [float(c_em_all[s]) for s in steps],
        "note": "c(t) depends only on the relative process, so it is the same for m = 2 and m = 3 at eps = 0.1",
    }


def per_path_factor(n0: dict) -> dict:
    rows = {}
    for r in n0["primary"]:
        ratio = float(r["se_ratio_fk_to_dk_median"])
        fac = (r["walkers_direct_kill"] / r["fk_paths"]) / ratio**2
        rows[r["label"]] = {"walkers_direct_kill": r["walkers_direct_kill"], "fk_paths": r["fk_paths"],
                            "se_ratio_fk_to_dk_median": ratio, "per_path_variance_factor": fac}
    facs = [v["per_path_variance_factor"] for v in rows.values()]
    return {"src": "n0_validation.json#primary[*].{walkers_direct_kill,fk_paths,se_ratio_fk_to_dk_median}",
            "formula": "(N_dk / N_fk) / (median se_fk/se_dk)^2", "cells": rows,
            "min": min(facs), "max": max(facs)}


def frozen_gate_count(n1: dict) -> dict:
    rows = []
    for c in n1["cells"]:
        if abs(c["eps"] - 0.1) < 1e-12 and c["allocation"] == "equal":
            be = c["basins_extended"]
            s_t = float(sum(abs(x) for x in be["deviation_from_target"]))
            s_g = float(sum(abs(x) for x in be["deviation_from_frozen_gate"]))
            rows.append({"m": c["m"], "B": c["B"], "key": c["key"], "sum_abs_dev_limit_law": s_t,
                         "sum_abs_dev_frozen_gate": s_g, "frozen_gate_better": s_g < s_t})
    return {"src": "N1/n1_allocation_design.json#cells[eps=0.1, allocation=equal].basins_extended."
                   "{deviation_from_target,deviation_from_frozen_gate}",
            "n_cells": len(rows), "n_frozen_gate_better": sum(r["frozen_gate_better"] for r in rows),
            "exceptions": [(r["m"], r["B"]) for r in rows if not r["frozen_gate_better"]], "rows": rows}


def n9a_sign_changes(n9a: dict) -> dict:
    rows = n9a["summary_rows"]
    tot = 0
    for r in rows:
        for _h, (n_max, n_min) in r["counts_by_resolution"].items():
            tot += int(n_max) + int(n_min)
    return {"src": "N9a/n9a_census.json#summary_rows[*].counts_by_resolution.* ([n_max, n_min])",
            "n_cases": len(rows), "n_resolutions": len(n9a["resolutions_h_over_eps"]),
            "total": tot}


def n4_width_ratio(n4: dict) -> dict:
    rows = n4["rows"]
    stat = {(r["m"], r["B"]): r for r in rows if r["preparation"] == "stationary"}
    devs = []
    for r in rows:
        if r["preparation"] == "stationary":
            continue
        s = stat[(r["m"], r["B"])]
        for j in range(r["m"]):
            meas = r["basins_extended"][j]["sd_time"] / s["basins_extended"][j]["sd_time"]
            pred = (r["th10_prediction"]["peak_sd_time_eps0.1"][j]
                    / s["th10_prediction"]["peak_sd_time_eps0.1"][j])
            devs.append({"m": r["m"], "B": r["B"], "preparation": r["preparation"], "passage": j + 1,
                         "measured_ratio": meas, "predicted_ratio": pred, "abs_dev": abs(meas - pred)})
    worst = max(devs, key=lambda d: d["abs_dev"])
    return {"src": "N4/n4_preparation.json#rows[*].{basins_extended[j].sd_time,th10_prediction.peak_sd_time_eps0.1}",
            "definition": "max over non-stationary preparations and passages of |sd/sd_stationary (measured) - "
                          "sd/sd_stationary (predicted from s^2(t_j))|",
            "max_abs_dev": worst["abs_dev"], "worst": worst, "n_comparisons": len(devs)}


def main() -> int:
    n6 = load("N6/n6_visibility_thresholds.json")
    n1 = load("N1/n1_allocation_design.json")
    i4 = load("I4_mean_field_topology/mean_field_topology.json")
    n0 = load("n0_validation.json")
    pb = pooled_b5(n6, n1)
    res = {
        "item": "derived numbers (R1 finding O16)",
        "driver": HERE.name,
        "note": "recomputed from stored script outputs; no new random numbers",
        "pooled_B5_m2_eps0.1": pb,
        "mean_field_over_exact_B5": mean_field_ratios(i4, n6, pb["pooled"]),
        "contact_factor_anchor": contact_factor(),
        "fk_per_path_variance_factor": per_path_factor(n0),
        "frozen_gate_equal_eps0.1": frozen_gate_count(n1),
        "n9a_sign_changes_total": n9a_sign_changes(load("N9a/n9a_census.json")),
        "n4_width_ratio_max_abs_dev": n4_width_ratio(load("N4/n4_preparation.json")),
    }
    OUT.write_text(json.dumps(res, indent=1, default=fk._json_default) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("frozen_gate_equal_eps0.1",)}, indent=1,
                     default=fk._json_default)[:3000])
    print("frozen gate:", res["frozen_gate_equal_eps0.1"]["n_frozen_gate_better"], "/",
          res["frozen_gate_equal_eps0.1"]["n_cells"], res["frozen_gate_equal_eps0.1"]["exceptions"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
