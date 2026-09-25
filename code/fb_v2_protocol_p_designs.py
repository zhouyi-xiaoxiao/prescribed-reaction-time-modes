#!/usr/bin/env python3
"""Protocol-P visibility of the V2 demonstration designs (NU-E follow-up).

The V2 design check "every density has exactly m peaks" (V2_design/demo_eps*.json#cells.*.mode_check)
counts local maxima of the per-passage smoothed densities f_{h_j}, one per basin, with a relative
prominence floor of 1e-3.  The paper's visibility protocol P (covariance-aware 5 sigma + 5 % relative
prominence, bandwidth 0.04, 0.02 bins on I = [0.5, 3.5], at 10^6 walkers) is stricter.  This script applies
protocol P to the EXPECTED histogram of each design's exact law (the independent out-of-sample FK ensemble,
V2_design/demo_eps*.json#out_of_sample.cells.*.density_bins, 0.01 bins aggregated to the production 0.02
bins) with exact_m_prr_fk_exact_law.classify_expected(walkers = 1e6), and records every maximum's relative
prominence and z.  (The direct-kill counts under protocol P at dt, dt/2, dt/4 are in the TEP JSONs,
#protocol_P_count_by_level.)

Run from code/:  python3 fb_v2_protocol_p_designs.py
Output: ../artifacts/data/exact_m_fixed_budget/V2_TEP/protocol_P_design_law.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exact_m_prr_fk_exact_law as fk  # noqa: E402

DATA = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
OUT = DATA / "V2_TEP" / "protocol_P_design_law.json"


def main() -> int:
    edges = fk.production_window_edges()
    cells = {}
    for e in ("0.05", "0.025"):
        D = json.loads((DATA / "V2_design" / f"demo_eps{e}.json").read_text())
        for k, c in D["out_of_sample"]["cells"].items():
            db = c["density_bins"]
            w, t0 = float(db["width"]), float(db["t_left"])
            dens = np.asarray(db["density"], float)
            i0, i1 = int(round((0.5 - t0) / w)), int(round((3.5 - t0) / w))
            k2 = int(round(0.02 / w))
            bm = (dens[i0:i1] * w).reshape(-1, k2).sum(1)
            r = fk.classify_expected(bm, edges, walkers=1e6)
            m = len(c["peaks"])
            rows = [{"time": round(x["time"], 4), "relative_prominence": x["relative_prominence"], "z": x["z"],
                     "significant": x["significant"]} for x in r["rows"]]
            cells[k] = {"m": m, "protocol_P_count_1e6": int(r["mode_count"]),
                        "equals_m": int(r["mode_count"]) == m,
                        "targets_r": D["cells"][k]["targets_r"], "maxima": rows,
                        "source": f"V2_design/demo_eps{e}.json#out_of_sample.cells.{k}.density_bins"}
    short = {k: v for k, v in cells.items() if not v["equals_m"]}
    out = {"item": "V2 NU-E: protocol-P visibility of the design laws", "driver": "code/fb_v2_protocol_p_designs.py",
           "classifier": "exact_m_prr_fk_exact_law.classify_expected (covariance-aware 5 sigma + 5% relative "
                         "prominence, bandwidth 0.04, 0.02 bins on [0.5, 3.5], walkers 1e6)",
           "n_designs": len(cells), "n_count_equals_m": sum(v["equals_m"] for v in cells.values()),
           "fewer_than_m": {k: {"count": v["protocol_P_count_1e6"], "m": v["m"], "targets_r": v["targets_r"],
                                "insignificant_maxima": [x for x in v["maxima"] if not x["significant"]]}
                            for k, v in short.items()},
           "cells": cells}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_designs", "n_count_equals_m")}, indent=1))
    for k, v in out["fewer_than_m"].items():
        print(k, v["count"], "/", v["m"], [(x["time"], round(x["relative_prominence"], 4), round(x["z"], 1))
                                           for x in v["insignificant_maxima"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
