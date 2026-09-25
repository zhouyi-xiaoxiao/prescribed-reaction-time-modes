#!/usr/bin/env python3
"""Total-Error Protocol for a number the main text keeps: the direct-kill check of the ten designed
(max-min) cells of N1 (main text Sec. 4.3: "Direct simulation (10^6 killed walkers) of ten designed
cells reproduces the count in all ten and the basin masses within 1.7 standard errors";
artifacts/data/exact_m_fixed_budget/N1/n1_allocation_design.json#directkill, tag 81, dt = 1e-3 only).

Here each cell is rerun with the coupled time-step ladder of fb_v2_tep_ladder.py (direct kill at dt, dt/2,
dt/4 on common random numbers; 1e6 walkers; tag 217, disjoint from N1's tag 81 and from the design
ensembles) and evaluated with the SAME functionals as N1:
  * window basin masses on the N1 window cuts (production 0.02 edges, cut index nearest to each cut time);
  * the covariance-aware protocol-P mode count (reclassify_covariance_aware.classify_both, bandwidth 0.04)
    of the 0.02-bin window histogram, at every level.
Richardson / total error as in fb_v2_tep_ladder.py (paired group bootstrap, B = 400).  Reported: count at
every level; DK - FK z at dt = 1e-3 (the original claim; the time-step bias cancels, both estimate the
dt = 1e-3 law); masses - p* at dt and in the continuum with the total error.

Subcommands:  designs | run [--walkers 1e6 --workers 3] | analyze | all
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import fb_v2_tep_ladder as lad  # noqa: E402

tep, de, fk = lad.tep, lad.de, lad.fk
N1_JSON = de.FB_DATA / "N1" / "n1_allocation_design.json"
DESIGNS = tep.TEP_ROOT / "_designs" / "n1"
TAG_N1 = 217
LEVELS = lad.LEVELS


def cell_key(c: dict) -> str:
    return f"n1_m{c['m']}_eps{c['eps']:g}_B{c['B']:g}_{c['allocation']}"


def cmd_designs(args=None) -> list:
    N1 = json.loads(N1_JSON.read_text())
    DESIGNS.mkdir(parents=True, exist_ok=True)
    out = []
    for dk in N1["directkill"]:
        c = dk["cell"]
        rows = [r for r in N1["cells"] if r["m"] == c["m"] and r["eps"] == c["eps"] and r["B"] == c["B"]
                and r["allocation"] == c["allocation"]]
        assert len(rows) == 1, c
        r = rows[0]
        import fb_n1_allocation_design as n1d
        spec = n1d.spec_for(int(c["m"]), float(c["eps"]))           # m = 5: z0 = 8, explicit centres
        d = {"label": cell_key(c), "eps": c["eps"], "B": c["B"], "c": spec.centres().tolist(),
             "z0": float(spec.z0), "w": list(dk["w"]), "tmax": 4.0, "dt": 0.001, "h": 0.02,
             "n1": {"cut_times": r["basins_window"]["cut_times"], "fk_mass": r["basins_window"]["mass"],
                    "fk_se_iid": r["basins_window"]["se_iid"], "target_p_star": r["target"],
                    "fk_protocol_mode_count_1e6": r["protocol_mode_count_1e6"],
                    "dk_tag81": {"masses": dk["basin_masses_dk"], "z_fk_vs_dk": dk["basin_z_fk_vs_dk"],
                                 "mode_count": dk["dk_mode_count_covariance_aware"]},
                    "source": f"N1/n1_allocation_design.json#directkill[cell={c}] and #cells[key={r['key']}, "
                              f"m={c['m']}, eps={c['eps']}]"}}
        p = DESIGNS / f"{d['label']}.json"
        p.write_text(json.dumps(d, indent=1))
        out.append(p)
    print(f"[designs] {len(out)} -> {DESIGNS}")
    return out


def cmd_run(args) -> None:
    for p in sorted(DESIGNS.glob("n1_*.json")):
        run = tep.TEP_ROOT / (p.stem + "_ladder")
        ns = argparse.Namespace(design=str(p), walkers=args.walkers, chunk=args.chunk, workers=args.workers,
                                groups="100", dt="1e-3", tag=TAG_N1, replicate=0, out=str(run),
                                no_wait=args.no_wait, force=False)
        t0 = time.time()
        lad.cmd_run(ns)
        print(f"[run] {p.stem} {time.time() - t0:.0f}s", flush=True)


def _hist_groups(C: np.ndarray, dt: float, edges: np.ndarray) -> np.ndarray:
    """(G, steps) kill counts per step -> (G, nbins) window histogram, np.histogram semantics on the
    production kill times (step + 1) * dt."""
    kt = (np.arange(C.shape[1]) + 1) * dt
    idx = np.searchsorted(edges, kt, side="right") - 1
    idx[kt == edges[-1]] = edges.size - 2                       # last edge inclusive
    ok = (idx >= 0) & (idx < edges.size - 1)
    HT = np.zeros((edges.size - 1, C.shape[0]))
    np.add.at(HT, idx[ok], C[:, ok].T)
    return HT.T


def analyze_cell(run: Path, B: int = 400, seed: int = 0) -> dict:
    import reclassify_covariance_aware as R
    d = json.loads((run / "design.json").read_text())
    n1 = d["n1"]
    lv = lad.load_levels(run, tep.load_design(run / "design.json"))
    edges = fk.production_window_edges()
    ci = [int(np.argmin(np.abs(edges - c))) for c in n1["cut_times"]]
    H = {L: _hist_groups(lv[L]["C"], lv[L]["meta"]["dt"], edges) for L in LEVELS}
    N = {L: lv[L]["sizes"] for L in LEVELS}

    def masses(h, n):
        cum = np.concatenate([[0.0], np.cumsum(h)])
        return np.array([(cum[ci[j + 1]] - cum[ci[j]]) / n for j in range(len(ci) - 1)])

    full = {L: masses(H[L].sum(0), N[L].sum()) for L in LEVELS}
    rng = np.random.default_rng(np.random.SeedSequence([de.BASE_SEED, 218, int(seed)]))
    G = H[1].shape[0]
    reps = {L: [] for L in LEVELS}
    for _ in range(B):
        idx = rng.integers(0, G, G)
        for L in LEVELS:
            reps[L].append(masses(H[L][idx].sum(0), N[L][idx].sum()))
    bs = {"full": {L: {"masses": full[L]} for L in LEVELS},
          "reps": {L: {"masses": np.array(reps[L])} for L in LEVELS}}
    Rm = lad.richardson_boot(bs, "masses")
    counts = {}
    for L in LEVELS:
        cls = R.classify_both(H[L].sum(0).astype(np.int64), edges, int(N[L].sum()),
                              bandwidth=fk.base.DEFAULT_BANDWIDTH)
        counts[L] = {"mode_count_covariance_aware": int(cls["mode_count_covariance_aware"]),
                     "significant_times": cls.get("significant_times_covariance_aware")}
    fkM, fkse = np.asarray(n1["fk_mass"]), np.asarray(n1["fk_se_iid"])
    tgt = np.asarray(n1["target_p_star"])
    lv1 = Rm["levels"][1]
    z_fk = (lv1["value"] - fkM) / np.sqrt(lv1["se"] ** 2 + fkse ** 2)
    te1, tec = lv1["total_error"], Rm["total_error_continuum"]
    out = {"item": "V2 TEP of main-text N1 direct-kill claim", "driver": HERE.name, "label": d["label"],
           "cell": {k: d[k] for k in ("eps", "B", "w", "c")}, "m": len(d["w"]),
           "walkers_per_level": int(N[1].sum()), "tag": lv[1]["meta"]["tag"],
           "seed_entropy": lv[1]["meta"]["seed_entropy"],
           "bootstrap": {"B": B, "seed_entropy": [de.BASE_SEED, 218, int(seed)], "paired": True},
           "mode_count_by_level": counts,
           "count_equals_m_all_levels": all(counts[L]["mode_count_covariance_aware"] == len(d["w"]) for L in LEVELS),
           "fk_protocol_mode_count_1e6": n1["fk_protocol_mode_count_1e6"],
           "masses": Rm,
           "dk_vs_fk_at_dt": {"z": z_fk, "max_abs_z": float(np.max(np.abs(z_fk)))},
           "vs_p_star": {"design_dt": {"deviation": lv1["value"] - tgt, "total_error": te1,
                                       "max_abs_dev": float(np.max(np.abs(lv1["value"] - tgt)))},
                         "continuum": {"deviation": Rm["R1"] - tgt, "total_error": tec,
                                       "max_abs_dev": float(np.max(np.abs(Rm["R1"] - tgt)))}},
           "max_abs_mass_bias_dt": float(np.max(np.abs(lv1["bias"]))),
           "max_mass_se_dt": float(np.max(lv1["se"])),
           "original_tag81": n1["dk_tag81"], "source": n1["source"]}
    de.write_json(run / f"tep_n1_{d['label']}.json", out)
    return out


def cmd_analyze(args) -> dict:
    rows = {}
    for p in sorted(DESIGNS.glob("n1_*.json")):
        run = tep.TEP_ROOT / (p.stem + "_ladder")
        if not (run / "level4.json").exists():
            continue
        r = analyze_cell(run)
        rows[p.stem] = {"count_equals_m_all_levels": r["count_equals_m_all_levels"],
                        "counts": [r["mode_count_by_level"][L]["mode_count_covariance_aware"] for L in LEVELS],
                        "m": r["m"], "max_abs_z_dk_vs_fk_dt": r["dk_vs_fk_at_dt"]["max_abs_z"],
                        "max_abs_mass_bias_dt": r["max_abs_mass_bias_dt"], "max_mass_se_dt": r["max_mass_se_dt"],
                        "max_abs_dev_p_star_dt": r["vs_p_star"]["design_dt"]["max_abs_dev"],
                        "max_abs_dev_p_star_continuum": r["vs_p_star"]["continuum"]["max_abs_dev"],
                        "max_total_error_continuum": float(np.max(r["vs_p_star"]["continuum"]["total_error"]))}
    summ = {"item": "V2 TEP of the main-text N1 direct-kill claim (ten designed cells)", "driver": HERE.name,
            "n_cells": len(rows), "cells": rows,
            "all_counts_equal_m_at_all_levels": all(v["count_equals_m_all_levels"] for v in rows.values()),
            "max_abs_z_dk_vs_fk_dt": max((v["max_abs_z_dk_vs_fk_dt"] for v in rows.values()), default=None),
            "max_abs_mass_bias_dt": max((v["max_abs_mass_bias_dt"] for v in rows.values()), default=None),
            "max_abs_dev_p_star_continuum": max((v["max_abs_dev_p_star_continuum"] for v in rows.values()),
                                                default=None)}
    de.write_json(tep.TEP_ROOT / "n1_tep_summary.json", summ)
    print(json.dumps(summ, indent=1, default=float))
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("designs")
    for nm in ("run", "all"):
        s = sub.add_parser(nm)
        s.add_argument("--walkers", default="1e6")
        s.add_argument("--chunk", default="1e4")
        s.add_argument("--workers", default="3")
        s.add_argument("--no-wait", action="store_true")
    sub.add_parser("analyze")
    a = ap.parse_args(argv)
    if a.cmd == "designs":
        cmd_designs()
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)
    elif a.cmd == "all":
        cmd_designs()
        cmd_run(a)
        cmd_analyze(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
