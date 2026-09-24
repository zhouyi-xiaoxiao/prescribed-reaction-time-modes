#!/usr/bin/env python3
"""N8: robustness battery on the exact law (fixed-budget campaign).

Item N8 of notes/gap_diagnosis_20260923/GAP_CLOSURE_PLAN.md (closes G12, G13,
G27).  All laws are exact laws of the discretised production process
(Euler--Maruyama dt = 1e-3, end-of-step Doi killing) from the N0
Feynman--Kac / Rao--Blackwell estimator ``exact_m_prr_fk_exact_law`` in its
"declared" mode (per-step exact kill probabilities with 5000-path batch sums,
plus exposure moments for the mean field).  2e5 unkilled paths per ensemble,
seed base 20260923, stream tag 86 (fk.TAGS["N8"]).  Ensembles that differ
only in the kernel (slab width rho, slab shape, rate cap, t_max prefix) share
paths by construction (common random numbers), so the contrasts with the
production kernel are paired.

Arms
----
(a) rho-scan, decoupling noise from slab width: m = 3, eps = 0.1,
    rho in {0.5, 1, 2} (slab width eps*rho), B in {1, 4}.
(b) top-hat slabs matched in mass and second moment (half-width sqrt(3) eps rho)
    at both anchors (2, 0.1), (3, 0.1); B = 1 (plan) and B = 4 (extension).
(c) L-infinity rate cap K_max in {1.0, 0.7} at both anchors, B in {1, 4}
    (declared items with cap; the uncapped production law on the same paths
    is the baseline ensemble "n8_base_m{m}").
(d) window accounting to t_max = 20 at (2, 0.1, 1) and (3, 0.1, 1).  The
    t_max = 20 paths are extensions of the baseline paths (t_max is not part
    of the path entropy), so the t <= 4 part must coincide with the baseline.
(e) d = 3 (two wrapped transverse directions, prefactor B/W^2), (2, 0.1),
    3 independent replicates, B in {0.5, 2} (plan) and B = 1 (the W5 anchor,
    for the late-peak overshoot of the mean field seen in SM Fig. S3).

Outputs
-------
artifacts/data/exact_m_fixed_budget/fk_ensembles/n8_*.json   (ensemble indices, seeds)
artifacts/data/exact_m_fixed_budget/N8/n8_summary.json        (SM table data)
artifacts/figures/fb_n8_robustness.{pdf,png}

Usage (from code/)
------------------
    python3 fb_n8_robustness.py simulate --arm base --m 2    # also: a, b, d, e
    python3 fb_n8_robustness.py analyze
    python3 fb_n8_robustness.py figure
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
import validate_exact_m_offlattice as base  # noqa: E402

OUT = fk.FB_DATA / "N8"
TAG = fk.TAGS["N8"]
PATHS = 200_000
MAX_WORKERS = 3
RHOS = (0.5, 1.0, 2.0)
CAPS = (1.0, 0.7)
D3_BUDGETS = (0.5, 1.0, 2.0)
D3_REPLICATES = (0, 1, 2)


def equal_w(m: int) -> tuple:
    return tuple([1.0 / m] * m)


def names_for(arm: str, m: int | None = None) -> list[str]:
    if arm == "a":
        return [f"n8a_rho{r:g}_m3_eps0.1" for r in RHOS]
    if arm == "base":
        return [f"n8_base_m{m}_eps0.1"]
    if arm == "b":
        return [f"n8b_tophat_m{m}_eps0.1"]
    if arm == "d":
        return [f"n8d_tmax20_m{m}_eps0.1"]
    if arm == "e":
        return [f"n8e_d3_m2_eps0.1_rep{r}" for r in D3_REPLICATES]
    raise ValueError(arm)


def simulate(arm: str, m: int | None = None, workers: int = MAX_WORKERS) -> list:
    out = []
    if arm == "a":
        for rho, name in zip(RHOS, names_for("a")):
            spec = fk.EnsembleSpec(m=3, eps=0.1, rho=rho)
            decl = [dict(B=B, w=equal_w(3), variant="full") for B in (1.0, 4.0)]
            out.append(fk.simulate_ensemble(spec, PATHS, name=name, tag=TAG, mode="declared",
                                            declared=decl, variants=("full",), workers=workers,
                                            note=f"N8(a) rho-scan: slab width eps*rho, rho={rho:g}"))
    elif arm == "base":
        spec = fk.EnsembleSpec(m=m, eps=0.1)
        decl = [dict(B=B, w=equal_w(m), variant="full") for B in (1.0, 4.0)]
        decl += [dict(B=B, w=equal_w(m), variant="full", cap=c) for c in CAPS for B in (1.0, 4.0)]
        out.append(fk.simulate_ensemble(spec, PATHS, name=names_for("base", m)[0], tag=TAG,
                                        mode="declared", declared=decl, variants=("full",),
                                        workers=workers,
                                        note="N8 baseline (production kernel) + (c) L-inf caps"))
    elif arm == "b":
        spec = fk.EnsembleSpec(m=m, eps=0.1, slab_shape="tophat")
        decl = [dict(B=B, w=equal_w(m), variant="full") for B in (1.0, 4.0)]
        out.append(fk.simulate_ensemble(spec, PATHS, name=names_for("b", m)[0], tag=TAG,
                                        mode="declared", declared=decl, variants=("full",),
                                        workers=workers,
                                        note="N8(b) top-hat slabs matched in mass and 2nd moment"))
    elif arm == "d":
        spec = fk.EnsembleSpec(m=m, eps=0.1, tmax=20.0)
        decl = [dict(B=1.0, w=equal_w(m), variant="full")]
        out.append(fk.simulate_ensemble(spec, PATHS, name=names_for("d", m)[0], tag=TAG,
                                        mode="declared", declared=decl, variants=("full",),
                                        workers=workers,
                                        note="N8(d) window accounting to t_max = 20"))
    elif arm == "e":
        for rep, name in zip(D3_REPLICATES, names_for("e")):
            spec = fk.EnsembleSpec(m=2, eps=0.1, n_perp=2)
            decl = [dict(B=B, w=(0.5, 0.5), variant=v) for v in ("full", "meancontact")
                    for B in D3_BUDGETS]
            out.append(fk.simulate_ensemble(spec, PATHS, name=name, tag=TAG, replicate=rep,
                                            mode="declared", declared=decl,
                                            variants=("full", "meancontact"), workers=workers,
                                            note=f"N8(e) d=3 replicate {rep}"))
    else:
        raise ValueError(arm)
    return out


# ============================================================================
# Declared-mode analysis helpers
# ============================================================================


def decl_item(res: dict, label: str) -> int:
    for i, d in enumerate(res["declared"]):
        if d["label"] == label:
            return i
    raise KeyError(label)


def binned(res: dict, i: int, edges=None) -> dict:
    """Production window bins (numpy.histogram semantics of the direct-kill protocol)."""
    edges = fk.production_window_edges() if edges is None else edges
    t = res["t"]
    p = res["p_step"][i]
    pb = res["batch_p_step"][i]
    mass = np.histogram(t, bins=edges, weights=p)[0]
    bmass = np.array([np.histogram(t, bins=edges, weights=row)[0] for row in pb])
    bw = np.diff(edges)
    return {"edges": edges, "t": 0.5 * (edges[:-1] + edges[1:]), "bin_mass": mass,
            "batch_mass": bmass, "density": mass / bw, "batch_density": bmass / bw[None, :],
            "cov": np.cov(bmass, rowvar=False) / bmass.shape[0]}


def meanfield_step(res: dict, i: int) -> np.ndarray:
    """Mean-field per-step kill probability exp(-B EX_{n-1}) - exp(-B EX_n)."""
    d = res["declared"][i]
    B = float(d["B"])
    mx = res["mean_X"][int(d["pair"])]
    prev = np.concatenate([[0.0], mx[:-1]])
    return np.exp(-B * prev) * (-np.expm1(-B * (mx - prev)))


def classify(law: dict) -> dict:
    cls = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000)
    clsc = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000,
                                cov=law["cov"])
    maxima = []
    for r, rc in zip(cls["rows"], clsc["rows"]):
        maxima.append({"time": r["time"], "relative_prominence": r["relative_prominence"],
                       "z_protocol_1e6": r["z"], "significant_protocol_1e6": r["significant"],
                       "z_fk_estimate": rc["z"], "significant_fk_estimate": rc["significant"]})
    return {"mode_count_protocol_1e6": int(cls["mode_count"]),
            "mode_count_fk_estimate": int(clsc["mode_count"]), "maxima": maxima}


def basins(law: dict, cuts) -> dict:
    edges = law["edges"]
    ci = [int(np.argmin(np.abs(edges - c))) for c in cuts]
    M = np.array([law["bin_mass"][ci[a]:ci[a + 1]].sum() for a in range(len(ci) - 1)])
    Mb = np.array([law["batch_mass"][:, ci[a]:ci[a + 1]].sum(1) for a in range(len(ci) - 1)]).T
    se = Mb.std(axis=0, ddof=1) / math.sqrt(Mb.shape[0])
    return {"cuts_edges": [float(edges[k]) for k in ci], "masses": M, "se_batch": se,
            "batch_masses": Mb}


def peak_times(law: dict) -> list:
    cls = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000,
                               cov=law["cov"])
    return [r["time"] for r in cls["rows"] if r["significant"]]


def _json(o):
    return fk._jsonable(o)


def item_summary(res: dict, i: int, spec: fk.EnsembleSpec, w, *, cuts=None) -> dict:
    d = res["declared"][i]
    law = binned(res, i)
    cls = classify(law)
    der = fk.derivative_signs(law["t"], law["density"], law["batch_density"], zthr=5.0)
    if cuts is None:
        cuts = [fk.WINDOW[0]] + fk.g_valley_times(spec, w) + [fk.WINDOW[1]]
    bm = basins(law, cuts)
    out = {"label": d["label"], "B": float(d["B"]), "cap": d.get("cap"),
           "mode_count_protocol_1e6": cls["mode_count_protocol_1e6"],
           "mode_count_fk_estimate": cls["mode_count_fk_estimate"],
           "derivative_census_z5": {"n_maxima": der["n_maxima"], "n_minima": der["n_minima"],
                                    "maxima_t": der["maxima_t"]},
           "maxima": cls["maxima"], "basin_cuts": bm["cuts_edges"],
           "basin_masses": bm["masses"], "basin_se_batch": bm["se_batch"],
           "window_mass": float(law["bin_mass"].sum())}
    if d.get("cap") is None:
        mfs = meanfield_step(res, i)
        mlaw = {"bin_mass": np.histogram(res["t"], bins=law["edges"], weights=mfs)[0],
                "batch_mass": np.zeros((2, law["bin_mass"].size)), "edges": law["edges"]}
        mb = basins(mlaw, cuts)
        out["mean_field_masses"] = mb["masses"]
        out["abs_err_mean_field_sum"] = float(np.abs(bm["masses"] - mb["masses"]).sum())
        lam, prod = fk.limit_masses(float(d["B"]), w, spec)
        out["product_limit_masses"] = prod
        out["lambda_j"] = lam
    return out, law, bm


def paired(bm1: dict, bm0: dict) -> dict:
    d = bm1["batch_masses"] - bm0["batch_masses"]
    diff = bm1["masses"] - bm0["masses"]
    se = d.std(axis=0, ddof=1) / math.sqrt(d.shape[0])
    return {"diff": diff, "se_batch": se}


def compare_direct_kill(law: dict, counts, walkers: int, cuts, min_expected=10.0) -> dict:
    p = law["bin_mass"]
    counts = np.asarray(counts, float)
    groups, cur, acc = [], [], 0.0
    for i in range(p.size):
        cur.append(i)
        acc += p[i] * walkers
        if acc >= min_expected:
            groups.append(cur)
            cur, acc = [], 0.0
    if cur:
        groups[-1].extend(cur)
    Mg = np.zeros((len(groups), p.size))
    for g, idx in enumerate(groups):
        Mg[g, idx] = 1.0
    pg = np.einsum("gi,i->g", Mg, p)
    cg = np.einsum("gi,i->g", Mg, counts)
    C = np.einsum("gi,ij,hj->gh", Mg, law["cov"], Mg) + (np.diag(pg) - np.outer(pg, pg)) / walkers
    r = cg / walkers - pg
    z = r / np.sqrt(np.diag(C))
    chi2 = float(r @ np.linalg.solve(C, r))
    dof = len(groups)
    x = (chi2 / dof) ** (1.0 / 3.0)
    zz = (x - (1 - 2.0 / (9 * dof))) / math.sqrt(2.0 / (9 * dof))
    edges = law["edges"]
    ci = [int(np.argmin(np.abs(edges - c))) for c in cuts]
    bz = []
    for a in range(len(ci) - 1):
        sl = slice(ci[a], ci[a + 1])
        mfk = p[sl].sum()
        sfk = law["batch_mass"][:, sl].sum(1).std(ddof=1) / math.sqrt(law["batch_mass"].shape[0])
        mdk = counts[sl].sum() / walkers
        bz.append({"fk": float(mfk), "dk": float(mdk),
                   "z": float((mdk - mfk) / math.hypot(sfk, math.sqrt(mfk * (1 - mfk) / walkers)))})
    return {"n_merged_bins": dof, "max_abs_z_merged": float(np.max(np.abs(z))),
            "chi2_cov": chi2, "chi2_dof": dof,
            "chi2_p_wilson_hilferty": 0.5 * math.erfc(zz / math.sqrt(2.0)), "basins": bz}


def smoothed_density(bin_mass, edges, bandwidth=base.DEFAULT_BANDWIDTH):
    bw = float(np.median(np.diff(edges)))
    kernel, norm, _ = fk._smoothing_operator(len(bin_mass), round(bw, 14), bandwidth)
    return np.convolve(np.asarray(bin_mass) / bw, kernel, mode="same") / norm


def analyze() -> dict:
    t0 = time.time()
    S = {"item": "N8", "driver": HERE.name, "tag": TAG, "seed": fk.BASE_SEED,
         "paths_per_ensemble": PATHS,
         "estimator": "N0 FK exact law, declared mode (exact_m_prr_fk_exact_law)"}
    # ---------------- (a) rho scan
    A = {"m": 3, "eps": 0.1, "rows": []}
    ref = {}
    for rho, name in zip(RHOS, names_for("a")):
        if not fk.index_path(name).exists():
            continue
        ens = fk.load_ensemble(name)
        res = ens.declared()
        A.setdefault("seed_entropy", ens.index["seed_entropy"])
        for i, d in enumerate(res["declared"]):
            summ, law, bm = item_summary(res, i, ens.spec, equal_w(3))
            summ.update({"rho": rho, "slab_width_eps_rho": 0.1 * rho, "name": name})
            gd = core.g_window_maxima_count(m=3, eps=0.1, weights=equal_w(3), p=ens.spec.model(),
                                            n_perp=1, target_times=ens.spec.times(),
                                            tmax=ens.spec.tmax)
            summ["free_clock_window_maxima"] = int(gd["n_window_maxima"])
            summ["free_clock_peak_times"] = [float(x["time"]) for x in gd["peaks"]]
            if rho == 1.0:
                ref[float(d["B"])] = bm
            summ["_bm"] = bm
            A["rows"].append(summ)
    for r in A["rows"]:
        bm = r.pop("_bm")
        if r["B"] in ref and r["rho"] != 1.0:
            r["paired_diff_vs_rho1"] = paired(bm, ref[r["B"]])
    S["a_rho_scan"] = A
    # ---------------- (b) top-hat, (c) caps, baseline
    Bsec, Csec = {}, {}
    base_laws = {}
    for m in (2, 3):
        nb_ = names_for("base", m)[0]
        if not fk.index_path(nb_).exists():
            continue
        ens = fk.load_ensemble(nb_)
        res = ens.declared()
        w = equal_w(m)
        cuts = [fk.WINDOW[0]] + fk.g_valley_times(ens.spec, w) + [fk.WINDOW[1]]
        rows, bms = [], {}
        for i, d in enumerate(res["declared"]):
            summ, law, bm = item_summary(res, i, ens.spec, w, cuts=cuts)
            bms[(float(d["B"]), d.get("cap"))] = bm
            base_laws[(m, float(d["B"]), d.get("cap"))] = law
            rows.append(summ)
        for r in rows:
            if r["cap"] is not None:
                r["paired_diff_vs_uncapped"] = paired(bms[(r["B"], r["cap"])], bms[(r["B"], None)])
        peak_rate = [float(B) * max(w) / (ens.spec.torus_w * math.sqrt(2 * math.pi) * ens.spec.sd())
                     for B in (1.0, 4.0)]
        Csec[f"m{m}"] = {"name": nb_, "seed_entropy": ens.index["seed_entropy"], "rows": rows,
                         "cuts": cuts,
                         "uncapped_peak_on_contact_rate_B1_B4": peak_rate}
        nt = names_for("b", m)[0]
        if fk.index_path(nt).exists():
            enst = fk.load_ensemble(nt)
            rest = enst.declared()
            trows = []
            for i, d in enumerate(rest["declared"]):
                summ, law, bm = item_summary(rest, i, enst.spec, w, cuts=cuts)
                summ["paired_diff_vs_gauss"] = paired(bm, bms[(float(d["B"]), None)])
                base_laws[(m, float(d["B"]), "tophat")] = law
                trows.append(summ)
            Bsec[f"m{m}"] = {"name": nt, "seed_entropy": enst.index["seed_entropy"],
                             "rows": trows, "cuts": cuts,
                             "tophat_half_width": math.sqrt(3.0) * enst.spec.sd()}
    S["b_tophat"] = Bsec
    S["c_cap"] = Csec
    # ---------------- (d) t_max = 20 window accounting
    D = {}
    for m in (2, 3):
        nd = names_for("d", m)[0]
        if not fk.index_path(nd).exists():
            continue
        ens = fk.load_ensemble(nd)
        res = ens.declared()
        spec = ens.spec
        t = res["t"]
        p = res["p_step"][0]
        pb = res["batch_p_step"][0]
        nb = pb.shape[0]
        B = float(res["declared"][0]["B"])

        def acc(mask):
            return float(p[mask].sum()), float(pb[:, mask].sum(1).std(ddof=1) / math.sqrt(nb))
        pre = acc(t < fk.WINDOW[0])
        win = acc((t >= fk.WINDOW[0]) & (t <= fk.WINDOW[1]))
        post4 = acc((t > fk.WINDOW[1]) & (t <= 4.0 + 1e-12))
        post20 = acc(t > fk.WINDOW[1])
        surv4 = 1.0 - float(p[t <= 4.0 + 1e-12].sum())
        surv20 = 1.0 - float(p.sum())
        mx = res["mean_X"][0]
        G = res["G_sampled"][0]
        i35 = int(np.searchsorted(t, fk.WINDOW[1], side="right")) - 1
        i4 = int(np.searchsorted(t, 4.0 + 1e-12, side="right")) - 1
        bound_c20 = B * (mx[-1] - mx[i35])
        bound_c4 = B * (mx[i4] - mx[i35])
        dens = p / spec.dt
        after = t >= fk.WINDOW[1]
        ratio_BG = float(np.max(dens[after] / np.maximum(B * G[after], 1e-300)))
        vinf = 1.0 / (spec.torus_w ** spec.n_perp * math.sqrt(2 * math.pi) * spec.sd())
        S_prev = 1.0 - np.concatenate([[0.0], np.cumsum(p)[:-1]])
        ratio_V = float(np.max(dens[after] / (B * vinf * S_prev[after])))
        der_post = fk.derivative_signs(t, dens, res["batch_p_step"][0] / spec.dt,
                                       bandwidth=0.05, zthr=5.0, window=(fk.WINDOW[1], spec.tmax))
        der_pre = fk.derivative_signs(t, dens, res["batch_p_step"][0] / spec.dt,
                                      bandwidth=0.02, zthr=5.0, window=(0.0, fk.WINDOW[0]))
        # coarse-bin monotonicity census after T (0.25-wide bins, batch SE)
        cedges = np.arange(fk.WINDOW[1], spec.tmax + 1e-9, 0.25)
        cm = np.histogram(t, bins=cedges, weights=p)[0]
        cb = np.array([np.histogram(t, bins=cedges, weights=row)[0] for row in pb])
        dcm = np.diff(cm)
        dcb = np.diff(cb, axis=1)
        zd = dcm / (dcb.std(axis=0, ddof=1) / math.sqrt(nb))
        coarse = {"bin_width": 0.25, "n_differences": int(dcm.size),
                  "n_negative_z_lt_-5": int(np.sum(zd < -5)), "n_positive_z_gt_2": int(np.sum(zd > 2)),
                  "n_positive_any": int(np.sum(dcm > 0)), "max_z": float(zd.max()),
                  "min_abs_z": float(np.abs(zd).min()),
                  "first_positive_difference_t": (float(cedges[1:-1][np.argmax(dcm > 0)])
                                                  if np.any(dcm > 0) else None)}
        # prefix consistency with the t_max = 4 baseline on common paths
        nbase = names_for("base", m)[0]
        prefix = None
        if fk.index_path(nbase).exists():
            rb = fk.load_ensemble(nbase).declared()
            ib = decl_item(rb, rb["declared"][0]["label"])
            k = rb["p_step"].shape[1]
            prefix = float(np.max(np.abs(rb["p_step"][ib] - p[:k])))
        D[f"m{m}"] = {"name": nd, "seed_entropy": ens.index["seed_entropy"], "B": B,
                      "tmax": spec.tmax, "pre_window": pre, "window": win,
                      "post_window_to_t4": post4, "post_window_to_t20": post20,
                      "survivors_t4": surv4, "survivors_t20": surv20,
                      "accounting_sum_t20": pre[0] + win[0] + post20[0] + surv20,
                      "bound_c_post_window_to_t20": bound_c20, "bound_c_post_window_to_t4": bound_c4,
                      "max_f_over_BG_after_T": ratio_BG, "max_f_over_BVinfS_after_T": ratio_V,
                      "V_inf_bound": vinf,
                      "stationary_points_after_T_z5": {"n_maxima": der_post["n_maxima"],
                                                       "n_minima": der_post["n_minima"],
                                                       "maxima_t": der_post["maxima_t"],
                                                       "minima_t": der_post["minima_t"],
                                                       "n_undecided": der_post["n_undecided_points"],
                                                       "bandwidth": 0.05},
                      "stationary_points_before_tau_z5": {"n_maxima": der_pre["n_maxima"],
                                                          "n_minima": der_pre["n_minima"],
                                                          "n_undecided": der_pre["n_undecided_points"],
                                                          "bandwidth": 0.02},
                      "coarse_monotonicity_after_T": coarse,
                      "density_at": {f"{x:g}": float(dens[int(round(x / spec.dt)) - 1])
                                     for x in (3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0)},
                      "max_abs_diff_vs_tmax4_prefix": prefix}
    S["d_window_accounting"] = D
    # ---------------- (e) d = 3
    E = {"replicates": []}
    pooled = {}
    w5 = json.loads((core.UPGRADE_DATA / "w5_d3_spotcheck" / "d3_spotcheck.json").read_text())
    for rep, name in zip(D3_REPLICATES, names_for("e")):
        if not fk.index_path(name).exists():
            continue
        ens = fk.load_ensemble(name)
        res = ens.declared()
        spec = ens.spec
        cuts = [fk.WINDOW[0]] + fk.g_valley_times(spec, (0.5, 0.5)) + [fk.WINDOW[1]]
        rows = []
        for i, d in enumerate(res["declared"]):
            summ, law, bm = item_summary(res, i, spec, (0.5, 0.5), cuts=cuts)
            key = (d["variant"], float(d["B"]))
            pooled.setdefault(key, []).append((law, bm))
            # late-peak overshoot of the mean field (production smoothing, bandwidth 0.04)
            late = law["t"] > cuts[1]
            sm_exact = smoothed_density(law["bin_mass"], law["edges"])
            mfs = meanfield_step(res, i)
            mf_bins = np.histogram(res["t"], bins=law["edges"], weights=mfs)[0]
            sm_mf = smoothed_density(mf_bins, law["edges"])
            summ["late_peak_exact_smoothed"] = float(sm_exact[late].max())
            summ["late_peak_mean_field_smoothed"] = float(sm_mf[late].max())
            summ["late_peak_overshoot_mean_field"] = float(sm_mf[late].max() / sm_exact[late].max() - 1)
            summ["variant"] = d["variant"]
            rows.append(summ)
        E["replicates"].append({"replicate": rep, "name": name,
                                "seed_entropy": ens.index["seed_entropy"], "rows": rows})
        E["cuts"] = cuts
    # replicate spread and pooled vs W5 direct kill
    spread = []
    for (var, B), lst in pooled.items():
        M = np.array([bm["masses"] for _, bm in lst])
        entry = {"variant": var, "B": B, "n_replicates": len(lst),
                 "masses_mean": M.mean(0), "masses_sd_across_replicates": M.std(0, ddof=1)
                 if len(lst) > 1 else None,
                 "masses_se_batch_each": [bm["se_batch"] for _, bm in lst]}
        if var == "full" and B == 1.0 and len(lst) == 3:
            law = {"bin_mass": np.mean([l_["bin_mass"] for l_, _ in lst], axis=0),
                   "batch_mass": np.concatenate([l_["batch_mass"] for l_, _ in lst], axis=0),
                   "edges": lst[0][0]["edges"]}
            law["cov"] = np.cov(law["batch_mass"], rowvar=False) / law["batch_mass"].shape[0]
            cl = w5["results"]["classifier"]
            entry["pooled_vs_w5_direct_kill_5e6"] = compare_direct_kill(
                law, cl["counts"], int(w5["parameters"]["config"]["walkers"]), E["cuts"])
            sm_dk = smoothed_density(np.asarray(cl["counts"], float)
                                     / w5["parameters"]["config"]["walkers"], law["edges"])
            late = law["t"] if "t" in law else None
            tt = 0.5 * (law["edges"][:-1] + law["edges"][1:])
            lm = tt > E["cuts"][1]
            entry["w5_dk_late_peak_smoothed"] = float(sm_dk[lm].max())
            entry["pooled_exact_late_peak_smoothed"] = float(
                smoothed_density(law["bin_mass"], law["edges"])[lm].max())
        spread.append(entry)
    E["pooled"] = spread
    S["e_d3"] = E
    S["analysis_seconds"] = time.time() - t0
    core.write_json(OUT / "n8_summary.json", _json(S))
    return S


def make_figure() -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")
    # (a) rho scan: m = 3, eps = 0.1, B = 1
    ax = axs[0, 0]
    cols = {0.5: core.OI_SKY, 1.0: "k", 2.0: core.OI_ORANGE}
    for rho, name in zip(RHOS, names_for("a")):
        res = fk.load_ensemble(name).declared()
        i = [k for k, d in enumerate(res["declared"]) if float(d["B"]) == 1.0][0]
        law = binned(res, i)
        ax.plot(law["t"], law["density"], color=cols[rho], lw=1.1,
                label=f"slab width $\\varepsilon\\rho={0.1 * rho:g}$")
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    ax.set_ylabel("exact density $f(t)$")
    ax.set_title("(a) $m=3$, noise $\\varepsilon=0.1$, $B=1$: slab width", loc="left")
    ax.legend(fontsize=6.5)
    # (b) slab shape and rate cap: m = 3, B = 1
    ax = axs[0, 1]
    rb = fk.load_ensemble(names_for("base", 3)[0]).declared()
    for k, d in enumerate(rb["declared"]):
        if float(d["B"]) != 1.0:
            continue
        law = binned(rb, k)
        cap = d.get("cap")
        if cap is None:
            ax.plot(law["t"], law["density"], color="k", lw=1.2, label="Gaussian slabs (production)")
        else:
            ax.plot(law["t"], law["density"], lw=1.0,
                    color=core.OI_BLUE if cap == 1.0 else core.OI_GREEN,
                    label=f"rate cap $K_{{\\max}}={cap:g}$")
    rt = fk.load_ensemble(names_for("b", 3)[0]).declared()
    k = [k for k, d in enumerate(rt["declared"]) if float(d["B"]) == 1.0][0]
    law = binned(rt, k)
    ax.plot(law["t"], law["density"], color=core.OI_VERMILLION, lw=1.0, ls="--",
            label="top-hat slabs (matched)")
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    ax.set_ylabel("exact density $f(t)$")
    ax.set_title("(b) $m=3$, $\\varepsilon=0.1$, $B=1$: shape and cap", loc="left")
    ax.legend(fontsize=6.5)
    # (c) t_max = 20
    ax = axs[1, 0]
    for m, cc in ((2, core.OI_BLUE), (3, core.OI_ORANGE)):
        ens = fk.load_ensemble(names_for("d", m)[0])
        res = ens.declared()
        t = res["t"]
        dens = res["p_step"][0] / ens.spec.dt
        edges = np.arange(0.0, 20.0 + 1e-9, 0.05)
        mass = np.histogram(t, bins=edges, weights=res["p_step"][0])[0]
        ax.semilogy(0.5 * (edges[:-1] + edges[1:]), np.maximum(mass / 0.05, 1e-12), color=cc,
                    lw=1.1, label=f"$m={m}$, exact law")
        G = res["G_sampled"][0]
        ax.semilogy(t[::50], np.maximum(G[::50], 1e-12), color=cc, lw=0.8, ls=":",
                    label=f"$m={m}$, bound $BG(t)$")
    ax.axvspan(fk.WINDOW[0], fk.WINDOW[1], color="0.9", zorder=0)
    ax.set_xlim(0, 20)
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    ax.set_ylabel("exact density $f(t)$ ($0.05$ bins)")
    ax.set_title("(c) $(m,0.1,1)$ to $t_{\\max}=20$ (window shaded)", loc="left")
    ax.legend(fontsize=6.2, loc="upper right")
    # (d) d = 3 at B = 1
    ax = axs[1, 1]
    w5 = json.loads((core.UPGRADE_DATA / "w5_d3_spotcheck" / "d3_spotcheck.json").read_text())
    cl = w5["results"]["classifier"]
    e = np.asarray(cl["edges"], float)
    c = np.asarray(cl["counts"], float) / w5["parameters"]["config"]["walkers"] / np.diff(e)
    tt = 0.5 * (e[:-1] + e[1:])
    ax.plot(tt[::2], c[::2], "o", ms=2.2, color="0.55", label="direct kill, $5\\times10^6$ (W5)")
    first = True
    for name in names_for("e"):
        res = fk.load_ensemble(name).declared()
        for k, d in enumerate(res["declared"]):
            if float(d["B"]) != 1.0:
                continue
            law = binned(res, k)
            if d["variant"] == "full":
                ax.plot(law["t"], law["density"], color=core.OI_BLUE, lw=1.0,
                        label="exact law (3 replicates)" if first else None)
                if first:
                    mfs = meanfield_step(res, k)
                    mf = np.histogram(res["t"], bins=law["edges"], weights=mfs)[0] / np.diff(law["edges"])
                    ax.plot(law["t"], mf, color=core.OI_VERMILLION, lw=1.0, ls="--",
                            label="mean field $BG\\,e^{-B\\Lambda}$")
            elif d["variant"] == "meancontact" and first:
                ax.plot(law["t"], law["density"], color=core.OI_GREEN, lw=1.0, ls="-.",
                        label="exact law, mean contact $c(t)$")
        first = False
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    ax.set_ylabel("density $f(t)$")
    ax.set_title("(d) $d=3$, $(2,0.1,1)$", loc="left")
    ax.legend(fontsize=6.2)
    return core.save_figure(fig, core.FIGURES / "fb_n8_robustness")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--arm", choices=("a", "base", "b", "d", "e"), required=True)
    s.add_argument("--m", type=int, default=None)
    s.add_argument("--workers", type=int, default=MAX_WORKERS)
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    if args.cmd == "simulate":
        for e in simulate(args.arm, args.m, workers=args.workers):
            print(e, e.index.get("last_invocation"))
    elif args.cmd == "analyze":
        S = analyze()
        print("analysis seconds", S["analysis_seconds"])
    elif args.cmd == "figure":
        make_figure()


if __name__ == "__main__":
    main()
