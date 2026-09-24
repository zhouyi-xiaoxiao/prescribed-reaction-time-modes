"""Referee-response numerical checks for the CNSNS submission (final fix pass, 2026-09-23).

Three checks requested by the referee adjudication (cnsns_final/adjudication.md, A04, A26):

``ess``
    Kish effective sample size ESS = (sum_i y_i)^2 / sum_i y_i^2 of the Feynman--Kac
    path weights y_i, recomputed from the stored ensembles (no new simulation):

    * N1 allocation design (stored per-path exposures ``dX_full``): every whole-axis
      and window-restricted basin of every scanned (m, eps, allocation, B) and of
      every grid cell; y_i = e^{-B X_i(s_{j-1})} - e^{-B X_i(s_j)};
    * N3 contact factorial (stored per-bin sums P, P2 of the per-path bin weights):
      per-0.02-bin ESS in the last whole-axis basin (t > s_{m-1}, up to t = 4) for
      every kernel variant and budget;
    * N9a census (stored per-step sums f_sum, f_sq): per-step ESS after the last
      window valley of G up to T = 3.5, gated and contact-one kernels.

    -> artifacts/data/exact_m_fixed_budget/R1_checks/ess_diagnostics.json

``dk``
    Direct-kill checks at large budget on the N3 anchor (m = 2, eps = 0.1, equal
    weights, field at the pair centre), 1e6 walkers each, with the frozen
    production direct-kill kernel used by fb_n3_contact_factorial.run_dk:
    (B = 100, a = 0.2) and (B = 1e4, a = 0.4).
    -> artifacts/data/exact_m_fixed_budget/N3/dk_<cell>.json

``tangent-dt``
    The boundary-tangent frozen-gate run at eps = 0.0125 (fb_n3_contact_factorial
    tangent arm, 2e5 paths) repeated at dt = 5e-4 (independent paths; the seed
    entropy contains dt).
    -> fk_ensembles/n3_tangent_m2_eps0.0125_dt0.0005.json

``analyze``
    Compares the dk and tangent-dt runs with the stored exact laws.
    -> artifacts/data/exact_m_fixed_budget/R1_checks/referee_checks.json

Usage (from code/, Xcode python, at most 3 worker processes):
    python3 fb_referee_checks.py ess
    python3 fb_referee_checks.py dk --workers 2
    python3 fb_referee_checks.py tangent-dt --workers 2
    python3 fb_referee_checks.py analyze
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402

OUT = fk.FB_DATA / "R1_checks"
N1_JSON = fk.FB_DATA / "N1" / "n1_allocation_design.json"
N3_DIR = fk.FB_DATA / "N3"
N9A_JSON = fk.FB_DATA / "N9a" / "n9a_census.json"

DK_EXTRA = {
    # name: (m, eps, B, contact_a, particle)   particle 0 = pair midpoint
    "a0.2_B100": (2, 0.1, 100.0, 0.2, 0),
    "a0.4_B1e4": (2, 0.1, 10000.0, 0.4, 0),
}
TANGENT_DT = (0.0125, 5e-4)


def _plain(o):
    """JSON-safe copy (numpy scalars/arrays to Python; NaN kept as null)."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _ess(s, q):
    s = np.asarray(s, float)
    q = np.asarray(q, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(q > 0, s * s / q, np.nan)


def _stats(e: np.ndarray) -> dict:
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "median": float(np.median(e)), "min": float(e.min()),
            "p10": float(np.percentile(e, 10)), "max": float(e.max())}


# ============================================================================
# ESS: N1 allocation design
# ============================================================================


def ess_n1() -> dict:
    d = json.loads(N1_JSON.read_text())
    out = {}
    for key, ens in d["ensembles"].items():
        m = int(key.split("_")[0][1:])
        eps = float(key.split("eps")[1])
        ext = [int(k) for k in ens["cuts"]["extended"]]
        win = [int(k) for k in ens["cuts"]["window"]]
        idx = sorted(set(ext) | set(win))
        pos = {k: i for i, k in enumerate(idx)}
        designs = []
        seen = set()
        for src, rows in (("scan", d["scan"]), ("cells", d["cells"])):
            for r in rows:
                if int(r["m"]) != m or abs(float(r["eps"]) - eps) > 1e-12:
                    continue
                w = np.asarray(r["w"], float)
                keyd = (r["allocation"], float(r["B"]), tuple(np.round(w, 12)))
                if keyd in seen:
                    continue
                seen.add(keyd)
                ref_ext = r.get("basins_extended", {}).get("mass") if src == "cells" else None
                ref_win = (r.get("basins_window", {}).get("mass") if src == "cells"
                           else r.get("basin_mass_window"))
                designs.append({"source": src, "allocation": r["allocation"], "B": float(r["B"]),
                                "w": w, "ref_ext": ref_ext, "ref_win": ref_win})
        nd = len(designs)
        S_e = np.zeros((nd, m)); Q_e = np.zeros((nd, m))
        S_w = np.zeros((nd, m)); Q_w = np.zeros((nd, m))
        root = fk.ENSEMBLE_ROOT / ens["ensemble"]
        files = sorted(root.glob("chunk*.npz"))
        N = 0
        t0 = time.time()
        for f in files:
            with np.load(f) as z:
                dX = z["dX_full"]
            Xc = np.cumsum(dX.astype(np.float64), axis=2)[:, :, idx]   # X(c_k), k in idx
            del dX
            N += Xc.shape[0]
            for i, ds in enumerate(designs):
                X = np.einsum("pmk,m->pk", Xc, ds["w"])
                B = ds["B"]
                for cuts, S, Q in ((ext, S_e, Q_e), (win, S_w, Q_w)):
                    for j in range(m):
                        xa = X[:, pos[cuts[j]]]
                        xb = X[:, pos[cuts[j + 1]]]
                        y = np.exp(-B * xa) * (-np.expm1(-B * (xb - xa)))
                        S[i, j] += y.sum()
                        Q[i, j] += (y * y).sum()
            del Xc
        rows = []
        max_dev = 0.0
        for i, ds in enumerate(designs):
            me = S_e[i] / N
            mw = S_w[i] / N
            for ref, mine in ((ds["ref_ext"], me), (ds["ref_win"], mw)):
                if ref is not None:
                    max_dev = max(max_dev, float(np.max(np.abs(np.asarray(ref) - mine))))
            rows.append({"source": ds["source"], "allocation": ds["allocation"], "B": ds["B"],
                         "w": ds["w"].tolist(),
                         "mass_whole_axis": me.tolist(), "ess_whole_axis": _ess(S_e[i], Q_e[i]).tolist(),
                         "mass_window": mw.tolist(), "ess_window": _ess(S_w[i], Q_w[i]).tolist()})
        ess_all = np.array([r["ess_whole_axis"] for r in rows])
        out[key] = {"ensemble": ens["ensemble"], "m": m, "eps": eps, "n_paths": N,
                    "cuts_whole_axis_index": ext, "cuts_window_index": win,
                    "n_designs": nd, "max_abs_mass_diff_vs_json": max_dev,
                    "min_ess_whole_axis": float(np.nanmin(ess_all)),
                    "min_ess_whole_axis_by_allocation": {
                        a: float(np.nanmin([min(r["ess_whole_axis"]) for r in rows
                                            if r["allocation"] == a]))
                        for a in sorted({r["allocation"] for r in rows})},
                    "min_ess_window": float(np.nanmin([r["ess_window"] for r in rows])),
                    "seconds": time.time() - t0, "rows": rows}
        print(f"[ess n1] {key}: {nd} designs, N={N}, min ESS (whole axis) "
              f"{out[key]['min_ess_whole_axis']:.0f}, mass check {max_dev:.2e}", flush=True)
    return out


# ============================================================================
# ESS: N3 contact factorial (per-bin, last basin)
# ============================================================================


def ess_n3() -> dict:
    out = {}
    last_cut = {2: 1.50, 3: 2.02}
    for m in (2, 3):
        name = f"n3_anchor_m{m}_eps0.1"
        meta = json.loads((N3_DIR / "ensembles" / f"{name}.json").read_text())
        edges = np.asarray(meta["edges"], float)
        P = P2 = None
        for c in sorted(meta["chunks"], key=lambda c: c["chunk"]):
            with np.load(c["file"]) as z:
                P = z["P"].copy() if P is None else P + z["P"]
                P2 = z["P2"].copy() if P2 is None else P2 + z["P2"]
        N = int(sum(c["size"] for c in meta["chunks"]))
        ci = int(np.argmin(np.abs(edges - last_cut[m])))
        late = np.arange(ci + 1, edges.size)          # bins (c_{k-1}, c_k], t > s_{m-1}
        late_win = late[edges[late] <= 3.5 + 1e-9]
        variants = [v["name"] for v in meta["variants"]]
        rows = []
        for iv, var in enumerate(variants):
            for ib, B in enumerate(meta["budgets"]):
                y = P[iv, ib]
                q = P2[iv, ib]
                e = _ess(y, q)
                mass_late = float(y[late].sum() / N)
                ymax = float(y[late].max()) if y[late].size else 0.0
                big = late[y[late] >= 0.01 * ymax] if ymax > 0 else late[:0]
                rows.append({"variant": var, "B": float(B), "late_mass_whole_axis": mass_late,
                             "per_bin_ess_last_basin": _stats(e[late]),
                             "per_bin_ess_last_basin_to_T": _stats(e[late_win]),
                             "per_bin_ess_bins_above_1pct_of_max": _stats(e[big])})
        out[f"m{m}"] = {"ensemble": name, "N_paths": N, "last_basin_from": float(edges[ci]),
                        "bin_convention": "bin k = (c_{k-1}, c_k], 0.02 grid on [0,4]",
                        "rows": rows}
        print(f"[ess n3] m={m}: {len(rows)} (variant, B) rows", flush=True)
    return out


# ============================================================================
# ESS: N9a census (per-step, after the last valley of G)
# ============================================================================


def ess_n9a() -> dict:
    cen = json.loads(N9A_JSON.read_text())
    out = {}
    for key, rec in cen["fk"].items():
        idx = json.loads(fk.index_path(rec["ensemble"]).read_text())
        labels = [dd.get("label") for dd in idx["declared"]]
        S = Q = None
        for c in sorted(idx["chunks"], key=lambda c: c["chunk"]):
            with np.load(c["file"]) as z:
                S = z["f_sum"].copy() if S is None else S + z["f_sum"]
                Q = z["f_sq"].copy() if Q is None else Q + z["f_sq"]
        N = int(sum(c["size"] for c in idx["chunks"]))
        dt = float(idx["spec"]["dt"])
        t = (np.arange(S.shape[1]) + 1) * dt
        lastv = float(rec["g_valleys"][-1])
        msk = (t >= lastv) & (t <= fk.WINDOW[1] + 1e-12)
        rows = []
        for i, lab in enumerate(labels):
            e = _ess(S[i, msk], Q[i, msk])
            rows.append({"label": lab, "mass_after_last_G_valley_to_T": float(S[i, msk].sum() / N),
                         "per_step_ess_after_last_G_valley": _stats(e)})
        out[key] = {"ensemble": rec["ensemble"], "N_paths": N, "dt": dt,
                    "last_G_valley": lastv, "rows": rows}
        print(f"[ess n9a] {key}: done", flush=True)
    return out


def ess_headline(E: dict) -> dict:
    h = {}
    n1 = E["N1"]
    for lab, keys in (("eps_le_0.1", [k for k in n1 if n1[k]["eps"] <= 0.1 + 1e-12]),
                      ("eps_0.15", [k for k in n1 if abs(n1[k]["eps"] - 0.15) < 1e-12])):
        h[f"N1_min_ess_whole_axis_{lab}"] = min(n1[k]["min_ess_whole_axis"] for k in keys)
        h[f"N1_min_ess_whole_axis_{lab}_by_ensemble"] = {k: n1[k]["min_ess_whole_axis"] for k in keys}
    mm = []
    for k in n1:
        for r in n1[k]["rows"]:
            if r["allocation"] in ("maxmin", "maxmin_mf", "maxmin_gate"):
                mm.append(min(r["ess_whole_axis"]))
    h["N1_min_ess_whole_axis_maxmin_designs_all_eps"] = float(min(mm))
    n3 = E["N3"]
    for m in (2, 3):
        rows = n3[f"m{m}"]["rows"]
        for a in ("0.4", "0.2", "0.15"):
            sel = [r for r in rows if r["variant"] == f"gcontact_a{a}_fmid"]
            h[f"N3_m{m}_gcontact_a{a}_median_per_bin_ess_last_basin_by_B"] = {
                f"{r['B']:g}": r["per_bin_ess_last_basin"].get("median") for r in sel}
            h[f"N3_m{m}_gcontact_a{a}_min_per_bin_ess_last_basin_to_T_by_B"] = {
                f"{r['B']:g}": r["per_bin_ess_last_basin_to_T"].get("min") for r in sel}
    n9 = E["N9a"]
    for key in n9:
        for r in n9[key]["rows"]:
            if r["label"].startswith("full_B") and float(r["label"][6:]) >= 20:
                h[f"N9a_{key}_{r['label']}_median_per_step_ess"] = \
                    r["per_step_ess_after_last_G_valley"].get("median")
    return h


def run_ess() -> dict:
    t0 = time.time()
    E = {"item": "R1_ess", "driver": HERE.name,
         "definition": "Kish ESS = (sum_i y_i)^2 / sum_i y_i^2 of the per-path Feynman-Kac "
                       "weights y_i of one estimated quantity (basin mass, bin mass, per-step mass)",
         "N1": ess_n1(), "N3": ess_n3(), "N9a": ess_n9a()}
    E["headline"] = ess_headline(E)
    E["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    core.write_json(OUT / "ess_diagnostics.json", _plain(E))
    return E


# ============================================================================
# Direct kill at large budget, tangent-dt repeat
# ============================================================================


def _n3():
    import fb_n3_contact_factorial as n3
    return n3


def run_dk(workers: int) -> dict:
    n3 = _n3()
    res = {}
    for cell, spec in DK_EXTRA.items():
        n3.DK_CELLS[cell] = spec
        f = N3_DIR / f"dk_{cell}.json"
        if f.exists():
            print(f"[dk] {cell}: exists, skipped", flush=True)
            continue
        out = n3.run_dk(cell, workers=workers)
        res[cell] = {k: out[k] for k in ("kills", "survivors", "wall_seconds",
                                         "kill_probability_max")}
        print(f"[dk] {cell}: {res[cell]}", flush=True)
    return res


def tangent_dt_name() -> str:
    eps, dt = TANGENT_DT
    return f"n3_tangent_m2_eps{eps:g}_dt{dt:g}"


def run_tangent_dt(workers: int):
    n3 = _n3()
    eps, dt = TANGENT_DT
    spec = fk.EnsembleSpec(m=2, eps=eps, r_par0=0.0, r_perp0=0.4, dt=dt)
    declared = [dict(B=B, w=(0.5, 0.5), variant="full") for B in n3.TANGENT_BUDGETS]
    declared += [dict(B=B, w=(0.5, 0.5), variant="meancontact") for B in n3.TANGENT_SUB_BUDGETS]
    declared += [dict(B=B, w=(0.5, 0.5), variant="nogate") for B in n3.TANGENT_SUB_BUDGETS]
    orig = fk.mean_contact_curve
    fk.mean_contact_curve = n3._mean_contact_em
    try:
        ens = fk.simulate_ensemble(spec, n3.TANGENT_PATHS, name=tangent_dt_name(), tag=n3.TAG,
                                   mode="declared", declared=declared,
                                   variants=("full", "meancontact", "nogate"),
                                   workers=workers, wait=False,
                                   note="referee check (A26): N3 boundary-tangent arm at "
                                        "eps=0.0125 repeated at dt=5e-4 (production 1e-3)")
    finally:
        fk.mean_contact_curve = orig
    return ens


def analyze() -> dict:
    n3 = _n3()
    out = {"item": "R1_referee_checks", "driver": HERE.name}
    # --- direct kill at large budget --------------------------------------
    dks = {}
    R = n3.load_reduced(n3.anchor_name(2))
    edges = R["edges"]
    ci = int(np.argmin(np.abs(edges - 1.50)))
    nb = R["Pb"].shape[2]
    for cell, (m, eps, B, a, particle) in DK_EXTRA.items():
        f = N3_DIR / f"dk_{cell}.json"
        if not f.exists():
            continue
        dk = json.loads(f.read_text())
        var = n3.vname("contact", a, "mid")
        iv = R["names"].index(var)
        ib = R["budgets"].index(float(B))
        fk_late = float(R["P"][iv, ib, ci + 1:].sum() / R["N"])
        fk_late_b = R["Pb"][iv, ib][:, ci + 1:].sum(1) / R["batch"]
        fk_se = float(fk_late_b.std(ddof=1) / math.sqrt(nb))
        cnt = np.asarray(dk["full_grid_counts"], float)          # (cps[k], cps[k+1]] = FK bin k+1
        W = float(dk["walkers"])
        dk_late = float(cnt[ci:].sum() / W)
        dk_se = math.sqrt(dk_late * (1 - dk_late) / W)
        fk_first = float(R["P"][iv, ib, 1:ci + 1].sum() / R["N"])
        dk_first = float(cnt[:ci].sum() / W)
        valleys = fk.g_valley_times(R["spec"], R["w"])
        cmp_win = n3.compare_dk(R, var, B, dk, [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]])
        dks[cell] = {"m": m, "eps": eps, "B": B, "contact_a": a, "variant": var,
                     "fk_paths": R["N"], "dk_walkers": int(W),
                     "whole_axis_late_basin": {"cut": float(edges[ci]), "fk": fk_late,
                                               "fk_se_batch": fk_se, "dk": dk_late,
                                               "dk_se": dk_se,
                                               "z": (dk_late - fk_late) / math.hypot(fk_se, dk_se)},
                     "whole_axis_first_basin": {"fk": fk_first, "dk": dk_first},
                     "window_comparison": cmp_win,
                     "dk_kill_probability_max": dk["kill_probability_max"],
                     "dk_wall_seconds": dk["wall_seconds"]}
    out["direct_kill_large_B"] = dks
    # --- tangent run at dt = 5e-4 ----------------------------------------
    name = tangent_dt_name()
    if fk.index_path(name).exists():
        orig = n3.tangent_name
        n3.tangent_name = lambda eps: name
        try:
            new = n3.analyze_tangent(TANGENT_DT[0])
        finally:
            n3.tangent_name = orig
        summ = json.loads((N3_DIR / "n3_summary.json").read_text())
        old = summ["tangent"][f"eps{TANGENT_DT[0]:g}"]
        rows = []
        for rn in new["rows"]:
            if rn["variant"] != "full":
                continue
            ro = [r for r in old["rows"] if r["variant"] == "full" and float(r["B"]) == float(rn["B"])]
            if not ro:
                continue
            ro = ro[0]
            Mn = np.asarray(rn["fk_masses"], float); Sn = np.asarray(rn["fk_se_batch"], float)
            Mo = np.asarray(ro["fk_masses"], float); So = np.asarray(ro["fk_se_batch"], float)
            rows.append({"B": float(rn["B"]), "late_dt0.001": float(Mo[1]),
                         "late_se_dt0.001": float(So[1]), "late_dt0.0005": float(Mn[1]),
                         "late_se_dt0.0005": float(Sn[1]),
                         "late_diff_new_minus_old": float(Mn[1] - Mo[1]),
                         "late_diff_z": float((Mn[1] - Mo[1]) / math.hypot(Sn[1], So[1])),
                         "first_diff_new_minus_old": float(Mn[0] - Mo[0]),
                         "orthant_late": float(rn["orthant_law"][1]),
                         "late_minus_orthant_dt0.0005": float(Mn[1] - rn["orthant_law"][1]),
                         "late_minus_orthant_dt0.001": float(Mo[1] - ro["orthant_law"][1])})
        out["tangent_dt_check"] = {"name": name, "eps": TANGENT_DT[0], "dt_new": TANGENT_DT[1],
                                   "dt_production": 1e-3, "N_paths": new["N_paths"],
                                   "sigma_t1": new["sigma_t1"],
                                   "seed_entropy": new["seed_entropy"],
                                   "independent_paths": True, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    core.write_json(OUT / "referee_checks.json", _plain(out))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ess")
    k = sub.add_parser("dk")
    k.add_argument("--workers", type=int, default=2)
    t = sub.add_parser("tangent-dt")
    t.add_argument("--workers", type=int, default=2)
    sub.add_parser("analyze")
    args = ap.parse_args(argv)
    if args.cmd == "ess":
        E = run_ess()
        print(json.dumps(E["headline"], indent=1, default=float))
    elif args.cmd == "dk":
        print(run_dk(min(args.workers, 3)))
    elif args.cmd == "tangent-dt":
        print(run_tangent_dt(min(args.workers, 3)))
    elif args.cmd == "analyze":
        res = analyze()
        print(json.dumps({k: v for k, v in res.items() if k != "item"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
