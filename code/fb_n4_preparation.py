#!/usr/bin/env python3
"""N4 -- preparation sensitivity (GAP_CLOSURE_PLAN.md §3 N4; closes G07, tests TH-10).

Initial midpoint law Z_0 ~ N(z0, s * eps^2 D0 / (2 gamma)) with s = Var Z_0 scale in
{0 (point release), 1 (stationary; production), 4 (four times stationary)}, at the
anchors (m, eps) = (2, 0.1) and (3, 0.1), B in {1, 4}, equal weights.

Method: Feynman--Kac stored ensembles, 2e5 unkilled paths each, seed tag 83.  The
three preparations share their Brownian increments and all other initial draws
(common random numbers): the SeedSequence entropy is computed with the Var Z_0 scale
set to 1 for every preparation, so the three path sets differ ONLY through the
initial midpoint offset sqrt(s) * eps * s_Z * N_0 (same N_0).  (2, eps) and (3, eps)
share paths as well (the kernel does not enter the seed).  Paired differences to the
production preparation therefore have small errors (path-group batch means).

Reported per cell: protocol mode count (covariance-aware 5 sigma + 5 % classifier on the
expected law at 1e6 walkers), significant sign changes of f' (z > 5), basin masses
(window and extended cuts) with paired differences to production, the limit law
(independent of the preparation) and the finite-eps bias of production, TH-1 sandwich
half-widths at the cuts, and per-basin temporal mean / s.d. of the exact law compared
with the TH-10 predictions (artifacts/data/exact_m_fixed_budget/TH10/preparation_predictions.json).

Subcommands: simulate | analyze | figure
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
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402

OUT_DIR = fk.FB_DATA / "N4"
OUT_JSON = OUT_DIR / "n4_preparation.json"
PRED_JSON = fk.FB_DATA / "TH10" / "preparation_predictions.json"
TAG = fk.TAGS["N4"]  # 83
PREPS = {"point_release": 0.0, "stationary": 1.0, "four_x_stationary": 4.0}
ANCHORS = ((2, 0.1), (3, 0.1))
B_LIST = (1.0, 4.0)
N_PATHS = 200_000
GROUPS = fk.DEFAULT_GROUPS
T975 = 2.093

_ORIG_ENTROPY = fk.path_entropy


def crn_entropy(spec, **kw):
    """Path entropy with the Var Z_0 scale pinned to 1 (preparations share all normals)."""
    return _ORIG_ENTROPY(replace(spec, var_z0_scale=1.0), **kw)


def ens_name(m, eps, prep):
    return f"n4_m{m}_eps{eps:g}_{prep}"


def spec_for(m, eps, prep):
    return fk.EnsembleSpec(m=m, eps=eps, var_z0_scale=PREPS[prep])


def cmd_simulate(args) -> None:
    fk.path_entropy = crn_entropy
    for m, eps in ANCHORS:
        for prep in PREPS:
            t0 = time.time()
            ens = fk.simulate_ensemble(spec_for(m, eps, prep), int(float(args.paths)),
                                       name=ens_name(m, eps, prep), tag=TAG, variants=("full",),
                                       workers=args.workers,
                                       note="N4 preparation ensemble; seed entropy computed with "
                                            "var_z0_scale = 1 for all preparations (CRN)")
            print(f"[n4] {ens.name} {ens.n_paths} paths wall {time.time() - t0:.1f}s", flush=True)
    fk.path_entropy = _ORIG_ENTROPY


def geometric_cuts(ens):
    c = ens.spec.centres()
    s = [-math.log((0.5 * (a + b) - ens.spec.z_bar) / (ens.spec.z0 - ens.spec.z_bar)) / ens.spec.gamma
         for a, b in zip(c[:-1], c[1:])]
    inner = [int(np.argmin(np.abs(ens.edges - x))) for x in s]
    wi = ens.window_index
    return {"window": [int(wi[0])] + inner + [int(wi[-1])],
            "extended": [0] + inner + [int(ens.edges.size - 1)]}


def ens_pass(ens, w, Bs, cuts, groups=GROUPS, block=25_000):
    """Bin masses on the FULL stored grid [0, tmax] per B, with group sums; X moments at cuts."""
    K = ens.cps.size
    N = ens.n_paths
    gb = fk._group_bounds(N, groups)
    w = np.asarray(w, float)
    acc = {B: {"P": np.zeros(K - 1), "P2": np.zeros(K - 1), "gP": np.zeros((groups, K - 1))}
           for B in Bs}
    xm = {"X1": np.zeros(K), "X2": np.zeros(K)}
    for start, dX, _ in ens.iter_chunks("full"):
        for b0 in range(0, dX.shape[0], block):
            blk = dX[b0:b0 + block].astype(np.float64)
            gid = fk._group_ids(gb, start + b0, blk.shape[0])
            dXw = np.einsum("imk,m->ik", blk, w)
            X = np.cumsum(dXw, axis=1)
            xm["X1"] += X.sum(0); xm["X2"] += (X * X).sum(0)
            for B in Bs:
                y = np.exp(-B * X[:, :-1]) * (-np.expm1(-B * dXw[:, 1:]))
                a = acc[B]
                a["P"] += y.sum(0); a["P2"] += (y * y).sum(0)
                a["gP"] += fk._group_sums(y, gid, groups)
    return {"acc": acc, "xm": xm, "N": N, "sizes": np.diff(gb).astype(float)}


def basin_stats(P, gP, sizes, edges, ci):
    """Masses, conditional mean and s.d. of time within each basin (bin-centre rule with
    the uniform-within-bin correction), with batch-means SEs over path groups."""
    t = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)
    out = []
    for j in range(len(ci) - 1):
        sl = slice(ci[j], ci[j + 1])

        def stats(p):
            M = p[sl].sum()
            mu = (t[sl] * p[sl]).sum() / M
            var = ((t[sl] - mu) ** 2 * p[sl]).sum() / M + (bw[sl] ** 2 / 12 * p[sl]).sum() / M
            return M, mu, math.sqrt(max(var, 0.0))

        M, mu, sd = stats(P)
        g = np.array([stats(gP[i]) for i in range(gP.shape[0])])
        se = fk._batch_se(g, sizes)
        out.append({"mass": float(M), "mass_se_batch": float(se[0]), "mean_time": float(mu),
                    "mean_time_se": float(se[1]), "sd_time": float(sd), "sd_time_se": float(se[2]),
                    "_g": g})
    return out


def cmd_analyze(args) -> None:
    pred = json.loads(PRED_JSON.read_text())["predictions"]
    rows = []
    ens_meta = {}
    for m, eps in ANCHORS:
        w = [1.0 / m] * m
        res = {}
        for prep in PREPS:
            ens = fk.load_ensemble(ens_name(m, eps, prep))
            cuts = geometric_cuts(ens)
            res[prep] = (ens, cuts, ens_pass(ens, w, B_LIST, cuts))
            ens_meta[ens.name] = {"n_paths": ens.n_paths, "seed": ens.index["seed"],
                                  "tag": ens.index["tag"], "seed_entropy": ens.index["seed_entropy"],
                                  "var_z0_scale": ens.spec.var_z0_scale,
                                  "process_seconds_total": ens.index.get("process_seconds_total")}
        for B in B_LIST:
            base_stats = None
            for prep in ("stationary", "point_release", "four_x_stationary"):
                ens, cuts, ps = res[prep]
                N = ps["N"]
                sizes = ps["sizes"]
                a = ps["acc"][B]
                P = a["P"] / N
                gP = a["gP"] / sizes[:, None]
                edges = ens.edges
                wi = ens.window_index
                Pw = P[wi[0]:wi[-1]]
                gPw = gP[:, wi[0]:wi[-1]]
                ew = edges[wi]
                cls = fk.classify_expected(Pw, ew, walkers=1e6)
                tw = 0.5 * (ew[:-1] + ew[1:])
                ds0 = fk.derivative_signs(tw, Pw / np.diff(ew), gPw / np.diff(ew)[None, :], bandwidth=0.0)
                ds4 = fk.derivative_signs(tw, Pw / np.diff(ew), gPw / np.diff(ew)[None, :], bandwidth=0.04)
                bs_w = basin_stats(P, gP, sizes, edges, cuts["window"])
                bs_e = basin_stats(P, gP, sizes, edges, cuts["extended"])
                lam, lim = fk.limit_masses(B, w, ens.spec)
                Lam = ps["xm"]["X1"] / N
                Var = np.maximum(ps["xm"]["X2"] / N - Lam * Lam, 0.0)
                from fb_n2_meanfield_residual import h_fun
                delta_cuts = (B * B * h_fun(B * Lam[cuts["window"]]) * Var[cuts["window"]]).tolist()
                mf_w = (np.exp(-B * Lam[cuts["window"][:-1]]) - np.exp(-B * Lam[cuts["window"][1:]])).tolist()
                pk = pred.get(f"m{m}_B{B:g}_{prep}")
                row = {"m": m, "eps": eps, "B": B, "preparation": prep,
                       "var_z0_scale": PREPS[prep], "ensemble": ens.name,
                       "protocol_mode_count_1e6": int(cls["mode_count"]),
                       "protocol_significant_times": [r["time"] for r in cls["rows"] if r["significant"]],
                       "derivative_maxima_bw0": ds0["n_maxima"], "derivative_maxima_bw0.04": ds4["n_maxima"],
                       "derivative_minima_bw0.04": ds4["n_minima"],
                       "limit_masses": lim.tolist(), "lambda": lam.tolist(),
                       "mean_field_window_masses_sampled_Lambda": mf_w,
                       "sandwich_halfwidth_delta_at_window_cuts": delta_cuts,
                       "basins_window": [{k: v for k, v in b.items() if k != "_g"} for b in bs_w],
                       "basins_extended": [{k: v for k, v in b.items() if k != "_g"} for b in bs_e]}
                if pk:
                    row["th10_prediction"] = {
                        "peak_sd_time_eps0.1": [p_["peak_sd_time_eps0.1"] for p_ in pk["passages"]],
                        "peak_shift_eps0.1": [p_["peak_shift_eps0.1"] for p_ in pk["passages"]],
                        "s2_tj": [p_["s2_tj"] for p_ in pk["passages"]]}
                if prep == "stationary":
                    base_stats = (bs_w, bs_e, row)
                else:
                    bw0, be0, r0 = base_stats
                    diffs = {}
                    for cn, bsp, bs0 in (("window", bs_w, bw0), ("extended", bs_e, be0)):
                        dm, dmse, rsd, rsdse = [], [], [], []
                        for bp, b0 in zip(bsp, bs0):
                            gd = bp["_g"][:, 0] - b0["_g"][:, 0]
                            dm.append(bp["mass"] - b0["mass"])
                            dmse.append(float(fk._batch_se(gd, sizes)))
                            gr = bp["_g"][:, 2] / b0["_g"][:, 2]
                            rsd.append(bp["sd_time"] / b0["sd_time"])
                            rsdse.append(float(fk._batch_se(gr, sizes)))
                        diffs[cn] = {"mass_minus_production": dm, "paired_se_batch": dmse,
                                     "sd_ratio_to_production": rsd, "sd_ratio_se_batch": rsdse}
                    fin_bias = [abs(b0["mass"] - l_) for b0, l_ in zip(bw0, r0["limit_masses"])]
                    diffs["finite_eps_bias_of_production_window"] = fin_bias
                    diffs["abs_mass_shift_below_finite_eps_bias"] = [abs(x) < y for x, y in zip(
                        diffs["window"]["mass_minus_production"], fin_bias)]
                    diffs["abs_mass_shift_below_sandwich_halfwidth"] = [
                        abs(x) < max(d0, d1) for x, d0, d1 in zip(
                            diffs["window"]["mass_minus_production"], r0["sandwich_halfwidth_delta_at_window_cuts"][:-1],
                            r0["sandwich_halfwidth_delta_at_window_cuts"][1:])]
                    if pk and r0.get("th10_prediction"):
                        p0 = r0["th10_prediction"]["peak_sd_time_eps0.1"]
                        p1 = row["th10_prediction"]["peak_sd_time_eps0.1"]
                        diffs["predicted_sd_ratio"] = [x / y for x, y in zip(p1, p0)]
                    row["vs_production"] = diffs
                rows.append(row)
    payload = {"item": "N4", "driver": HERE.name, "seed": fk.BASE_SEED, "tag": TAG,
               "crn": "seed entropy computed with var_z0_scale = 1 for all preparations; all "
                      "normals shared, the preparations differ only through the initial "
                      "midpoint offset", "model": core.model_payload(),
               "ensembles": ens_meta, "rows": rows,
               "prediction_source": str(PRED_JSON.relative_to(fk.REPORT)),
               "acceptance": acceptance(rows)}
    core.write_json(OUT_JSON, fk._jsonable(payload))
    print(f"[n4] wrote {OUT_JSON}")


def acceptance(rows):
    out = {"mode_count_m_every_cell": all(r["protocol_mode_count_1e6"] == r["m"] for r in rows),
           "derivative_maxima_m_every_cell_bw0.04": all(r["derivative_maxima_bw0.04"] == r["m"] for r in rows),
           "mode_counts": [(r["m"], r["B"], r["preparation"], r["protocol_mode_count_1e6"],
                            r["derivative_maxima_bw0.04"]) for r in rows]}
    sh = [r for r in rows if "vs_production" in r]
    out["max_abs_mass_shift_window"] = max(max(abs(x) for x in r["vs_production"]["window"]["mass_minus_production"])
                                           for r in sh)
    out["all_shifts_below_finite_eps_bias"] = all(all(r["vs_production"]["abs_mass_shift_below_finite_eps_bias"])
                                                  for r in sh)
    out["all_shifts_below_sandwich_halfwidth"] = all(all(r["vs_production"]["abs_mass_shift_below_sandwich_halfwidth"])
                                                     for r in sh)
    return out


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    d = json.loads(OUT_JSON.read_text())
    col = {"point_release": core.OI_VERMILLION, "stationary": "k", "four_x_stationary": core.OI_BLUE}
    lab = {"point_release": "point release ($\\mathrm{Var}\\,Z_0=0$)",
           "stationary": "stationary (production)", "four_x_stationary": "$4\\times$ stationary"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), layout="constrained", sharex=True)
    for i, (m, eps) in enumerate(ANCHORS):
        for j, B in enumerate(B_LIST):
            ax = axes[i, j]
            for prep in PREPS:
                ens = fk.load_ensemble(ens_name(m, eps, prep))
                law = fk.exact_law(ens, B, [1.0 / m] * m)
                ax.plot(law["t"], law["density"], color=col[prep], lw=1.0 if prep != "stationary" else 1.4,
                        ls="-" if prep != "stationary" else "--", label=lab[prep])
            for tj in ens.spec.times():
                ax.axvline(tj, color="0.6", lw=0.6, ls=":")
            row = [r for r in d["rows"] if r["m"] == m and r["B"] == B]
            cnt = ", ".join(str(r["protocol_mode_count_1e6"]) for r in row)
            ax.set_title(f"$m$ = {m}, $\\varepsilon$ = {eps:g}, $B$ = {B:g}  (mode counts: {cnt})")
            ax.set_xlim(0.5, 3.5)
            ax.set_ylim(bottom=0)
            if i == 1:
                ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
            if j == 0:
                ax.set_ylabel("density $f(t)$")
    axes[0, 0].legend(fontsize=6.3, frameon=False)
    core.stream_tag(fig, "fb N4")
    written = core.save_figure(fig, fk.FIGURES / "fb_n4_preparation")
    plt.close(fig)
    print("\n".join(written))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--paths", default=str(N_PATHS))
    s.add_argument("--workers", type=int, default=3)
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze, "figure": cmd_figure}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
