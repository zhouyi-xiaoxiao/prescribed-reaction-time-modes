"""V2 item 3 / CV-2 companion: float64 scan of the m = 3 max-min designs (NOT a certificate).

Question: along the max-min allocation w*(B) of Prop. 3(iii) (equal target masses, anchor targets
(0.8, 1.6, 2.8), v_j = 4 exp(-t_j), A = 1), which fold binds for the fixed-width limit law f_det, and
does the design keep all three peaks?  The codimension-2 switch b_2 = b_3 of fb_v2_codim2_switch.py
(F3S) needs w_1/w_2 of order 10^3; the max-min designs have w_1/w_2 = v_1 lam_1/(v_2 lam_2) < v_1/v_2
= e^{0.8} because lam_1 = -ln(1-p) < lam_2 = ln((1-p)/(1-2p)) ((1-p)^2 > 1-2p).  This scan checks
directly, on a budget grid, that the last flank binds wherever (D1)-(D3) hold.

Arithmetic: float64 grid analysis of fb_finite_width_det.analyse (n = 200001 grid points on I); the
numbers are diagnostics, not bounds.  Deterministic, no seeds.

Run (from code/):  python fb_v2_maxmin_m3_scan.py
Output: artifacts/data/exact_m_fixed_budget/V2_bifurcation/maxmin_m3_float_scan.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import fb_allocation_law as AL
import fb_finite_width_det as F

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "V2_bifurcation" / "maxmin_m3_float_scan.json"
TARGETS = F.TARGETS[3]
V = [4.0 * math.exp(-t) for t in TARGETS]
RHOS = (0.15, 0.2, 0.25, 0.3)
BUDGETS = [10 ** (k / 20.0) for k in range(-40, 81)]  # 1e-2 .. 1e4


def point(rho, B):
    try:
        w = AL.p_star(B, V, 1.0)["weights"]
    except ValueError:  # 1 - theta below double precision (u < e^-700): outside the float design law
        return {"B": B, "w": None, "w1_over_w2": None, "D1": None, "note": "beyond double-precision design law"}
    a = F.analyse(3, rho, w=w, n=200_001)
    row = {"B": B, "w": w, "w1_over_w2": w[0] / w[1], "D1": a["D1_signature_ok"]}
    if not a["D1_signature_ok"]:
        return row
    c = a["candidates"]
    row.update({"D2": a["D2_first_flank_monotone"], "D3": a["D3_flank_unimodal"], "candidates": c,
                "binding": a["binding"], "B_top_det": a["B_top_det"],
                "b2_over_b3": c.get("flank_2", math.nan) / c.get("flank_3", math.nan),
                "all_peaks_kept": bool(B < a["B_top_det"])})
    return row


def summarise(rows):
    rows = [r for r in rows if r["w"] is not None]
    ok = [r for r in rows if r["D1"] and r.get("D2") and r.get("D3")]
    lost = [r for r in ok if not r["all_peaks_kept"]]
    return {
        "n_budgets_in_float_range": len(rows), "B_max_in_float_range": max(r["B"] for r in rows), "n_D1_D3_ok": len(ok),
        "B_range_D1_D3_ok": [min(r["B"] for r in ok), max(r["B"] for r in ok)] if ok else None,
        "D1_D3_ok_contiguous": bool(ok) and [r["B"] for r in rows if r in ok] == [
            r["B"] for r in rows[rows.index(ok[0]):rows.index(ok[-1]) + 1]],
        "bindings_where_ok": sorted({r["binding"] for r in ok}),
        "min_b2_over_b3": min(r["b2_over_b3"] for r in ok) if ok else None,
        "max_w1_over_w2": max(r["w1_over_w2"] for r in rows),
        "B_grid_where_D1_D3_ok_and_a_peak_is_lost (B > B_top_det)": [r["B"] for r in lost],
        "B_grid_where_D1_D3_ok_and_all_peaks_kept": [r["B"] for r in ok if r["all_peaks_kept"]],
        "binding_where_lost": sorted({r["binding"] for r in lost}),
    }


def main():
    t0 = time.time()
    res = {"script": "code/fb_v2_maxmin_m3_scan.py", "item": "V2 item 3, CV-2 (companion float scan)",
           "status": "float64 diagnostic (fb_finite_width_det.analyse, n = 200001); NOT certified",
           "deterministic": True, "seeds": None,
           "design_law": "Prop. 3(iii) max-min (equal masses), A = 1, v_j = 4 exp(-t_j), targets (0.8, 1.6, 2.8); "
                         "fb_allocation_law.p_star",
           "e^{0.8} = v1/v2": V[0] / V[1], "budgets": "10^(k/20), k = -40..80", "by_rho": {}}
    for rho in RHOS:
        rows = [point(rho, B) for B in BUDGETS]
        res["by_rho"][str(rho)] = {"summary": summarise(rows), "rows": rows}
        print(rho, res["by_rho"][str(rho)]["summary"], flush=True)
    res["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
