#!/usr/bin/env python3
"""Second independent FK out-of-sample check of stored SAA designs (uplift2 item 1, NU-C).

Re-evaluates converged designs of a V2_design JSON on a fresh ensemble (tag 204, replicate R >= 2;
replicate 0 = design ensemble, 1 = first out-of-sample ensemble) and reports achieved - target with
z = dev / sqrt(SE_oos^2 + SE_design^2).  Seeds disjoint from every earlier ensemble by the replicate
index in the SeedSequence entropy.

CLI: python3 fb_v2_oos_replicate.py --src failed_cells_nocap.json --run tmax6 --n 1e6 --replicate 2
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_v2_saa_newton as sn  # noqa: E402

de = sn.de


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", default="")
    ap.add_argument("--n", default="1e6")
    ap.add_argument("--replicate", type=int, default=2)
    a = ap.parse_args(argv)
    d = json.loads((sn.OUT / a.src).read_text())
    cells = d["runs"][a.run]["cells"] if a.run else d["cells"]
    cells = {k: c for k, c in cells.items() if c.get("status") == "converged"}
    first = next(iter(cells.values()))
    spec = de.path_spec(first["eps"], tmax=first["tmax"])
    assert all(c["eps"] == first["eps"] and c["tmax"] == first["tmax"] for c in cells.values())
    assert a.replicate >= 2, "replicates 0 and 1 are the design and first out-of-sample ensembles"
    designs = [de.Design(c["c"], c["w"], c["B"]) for c in cells.values()]
    with sn.StreamEngine(spec, int(float(a.n)), tag=sn.TAG_DEMO, replicate=a.replicate, chunk=sn.CHUNK,
                         group=sn.GROUP) as se:
        laws = se.evaluate(designs, want_grad=False)
        meta = se.meta
    out = {}
    for (k, c), law in zip(cells.items(), laws):
        m = c["m"]
        res = sn.analyse(law, np.asarray(c["bandwidths_h"]), c["achieved"]["cuts"], jackknife=True)
        y, _ = sn.f1_outputs(res, m)
        target = np.concatenate([c["targets_T"], np.asarray(c["targets_r"])[:m - 1]])
        se_o = np.concatenate([res["se"]["peaks"], res["se"]["ratios"][:m - 1]])
        se_i = np.concatenate([c["se"]["peaks"], np.asarray(c["se"]["ratios"])[:m - 1]])
        dev = y - target
        z = dev / np.sqrt(se_o ** 2 + se_i ** 2)
        out[k] = {"peaks": res["peaks"], "ratios": res["ratios"], "yield": res["yield"], "deviation": dev.tolist(),
                  "se_oos": se_o.tolist(), "se_in_sample": se_i.tolist(), "z": z.tolist(),
                  "max_abs_z": float(np.max(np.abs(z))), "max_abs_peak_dev": float(np.max(np.abs(dev[:m]))),
                  "mode_check": sn.mode_check(law, np.asarray(c["bandwidths_h"]), res)}
        print(f"[oos-r{a.replicate}] {k}: dev {np.round(dev, 5).tolist()} z {np.round(z, 2).tolist()}", flush=True)
    stem = Path(a.src).stem + (f"_{a.run}" if a.run else "")
    payload = {"item": "uplift2 item 1 NU-C (second independent FK out-of-sample check)",
               "driver": "code/fb_v2_oos_replicate.py", "source": f"V2_design/{a.src}" + (f"#runs.{a.run}" if a.run else ""),
               "engine": meta, "cells": out}
    de.write_json(sn.OUT / f"oos_r{a.replicate}_{stem}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
