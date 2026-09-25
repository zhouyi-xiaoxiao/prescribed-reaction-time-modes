#!/usr/bin/env python3
"""Timing horizon of the fixed-budget design map by continuation (uplift2 item 1, NU-D).

Question.  At fixed (eps, B), m = 3, with the first two peak times held at the anchor targets
(0.8, 1.6) and the three conditional mass ratios held equal (formulation F1 minus the last peak-time
equation), how late can the LAST peak be placed?  The design map is continued along the late-stripe
position: for each late-stripe centre c_3 (nominal passage time t'_3 = mu^{-1}(c_3)) the remaining
four equations are solved for (t'_1, t'_2, u_1, u_2) by SAA Newton on the exact law, and the late
peak time T_3(c_3) is recorded together with

    g = dT_3/dt'_3 along the solution curve = J_33 - J_3R J_RR^{-1} J_R3   (Schur complement),

so that det J = det J_RR * g (J = full 5x5 F1 Jacobian in (t', u) coordinates).  det J = 0 <=> g = 0.

Finding encoded by this driver (see V2_design/horizon_*.json): g > 0 all along the continued branch,
and g ~ e^{-gamma t'_3} -> 0 only as the late stripe approaches the relaxation centre z_bar
(t'_3 -> infinity, since dc/dt' = -gamma (c - z_bar)).  Within (H4) (stripes centred at points
mu(t) crossed by the mean path, c_3 > z_bar) the latest reachable late-peak time is therefore the
limit t_max(eps, B) = T_3(c_3 -> z_bar+), evaluated directly at c_3 = z_bar (the law is continuous
in c_3).  It is compared with gamma^{-1} ln(1/eps) + c (a numerically continued scaling law, not a
theorem).  One point beyond z_bar (c_3 = -sigma_Z, outside (H4)) checks that the branch is regular
there, i.e. that the horizon is the boundary of the passage regime rather than a fold.

Ensembles: one per eps, tag 205 (NU_D); N set by --n/--n1 (production run: 1.5e5 paths, stage 1 7.5e4;
recorded in each JSON), dt = 1e-3, tmax(eps) below.
Mean-field (noiseless EM) continuation of the same system on a fine grid is stored alongside.

CLI:  python3 fb_v2_timing_horizon.py mf                 # mean-field continuation (all eps, B)
      python3 fb_v2_timing_horizon.py exact --eps 0.05    # exact-law continuation points + t_max
      python3 fb_v2_timing_horizon.py fit                 # scaling fit over the stored results
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

import fb_v2_saa_newton as sn  # noqa: E402

de = sn.de
fk = de.fk
OUT = sn.OUT
TAG_HORIZON = de.TAGS["NU_D"]          # 205
EPS_LIST = (0.1, 0.07, 0.05, 0.035, 0.025, 0.01)
B_LIST = (4.0, 8.0)
T_FIXED = (0.8, 1.6)
T3_ANCHOR = 2.8
CAP = 3.5                  # N11 window cap on the nominal stripe times
TMAX = {0.1: 6.0, 0.07: 6.0, 0.05: 6.0, 0.035: 6.5, 0.025: 7.0, 0.01: 7.5}
M = 3
ROWS = [0, 1, 3, 4]        # equations kept: peaks 1,2 and ratios 1,2 (F1 rows)
COLS = [0, 1, 3, 4]        # unknowns: t'_1, t'_2, u_1, u_2 (late stripe fixed)


def target_ratios():
    return np.ones(M) / M


def bandwidths(spec):
    # per-passage bandwidths from the anchor target times (the late one at the anchor 2.8)
    return sn.bandwidths((T_FIXED[0], T_FIXED[1], T3_ANCHOR), spec)


def eval_structure(res):
    if not all(res["peaks_ok"]) or not all(res["valleys_ok"]) or not all(res["peaks_interior"]):
        raise sn.DesignMapError("structure lost")


def jac_c_u(Jraw, m=M):
    """Raw (c, w, B) columns -> (c_1..c_m, u_1..u_{m-1})."""
    return np.hstack([Jraw[:, :m], Jraw[:, m:2 * m - 1] - Jraw[:, 2 * m - 1:2 * m]])


def reduced_pieces(Jraw, c, spec):
    """Jacobian in (t'_1, t'_2, c_3, u_1, u_2): stripes 1,2 in time, the late one in its centre."""
    Jcu = jac_c_u(Jraw)
    tt = de.t_of_c(c[:2], spec)
    J = Jcu.copy()
    J[:, :2] = Jcu[:, :2] * de.dc_dt(tt, spec)[None, :]
    JRR = J[np.ix_(ROWS, COLS)]
    g_c = J[2, 2] - J[2, COLS] @ np.linalg.solve(JRR, J[ROWS, 2])      # dT_3/dc_3 along the branch
    dc3 = float(de.dc_dt(de.t_of_c(c[2], spec), spec)) if c[2] > spec.z_bar else float("nan")
    return J, JRR, float(g_c), dc3


def point_record(res, c, w, spec, Jraw, extra=None):
    J, JRR, g_c, dc3 = reduced_pieces(Jraw, c, spec)
    t3 = float(de.t_of_c(c[2], spec)) if c[2] > spec.z_bar else float("inf")
    g_t = g_c * dc3 if math.isfinite(dc3) else 0.0
    rec = {"c3": float(c[2]), "t3_nominal": t3, "T3": float(res["peaks"][2]), "peaks": res["peaks"],
           "ratios": res["ratios"], "yield": res["yield"], "valleys": res["valleys"],
           "valley_to_peak": [fv / min(a, b) for fv, a, b in zip(res["f_valley"], res["f_peak"][:-1],
                                                                 res["f_peak"][1:])],
           "c": list(map(float, c)), "w": list(map(float, w)), "stripe_times_12": de.t_of_c(c[:2], spec).tolist(),
           "g_dT3_dc3": g_c, "g_dT3_dt3": g_t, "det_JRR": float(np.linalg.det(JRR)),
           "det_J_times_coords": float(np.linalg.det(JRR) * g_t)}
    if extra:
        rec.update(extra)
    return rec


# ----------------------------------------------------------------------------
# Mean-field continuation
# ----------------------------------------------------------------------------


def mf_reduced(c, w, B, spec, hs, c3, cuts, it=40, tol=1e-12):
    c = np.array(c, float)
    c[2] = c3
    w = np.array(w, float)
    rt = target_ratios()
    for _ in range(it):
        law = de.mean_field_law(de.Design(c, w, B), spec)
        res = sn.functionals(law.p, law.dp, law.t, hs, cuts)
        eval_structure(res)
        y, Jraw = sn.f1_outputs(res, M)
        F = np.concatenate([y[:2] - np.array(T_FIXED), y[3:] - rt[:2]])
        if np.max(np.abs(F)) < tol:
            break
        J, JRR, _, _ = reduced_pieces(Jraw, c, spec)
        dx = np.linalg.solve(JRR, -F)
        a = 1.0
        for _ in range(30):
            tn = de.t_of_c(c[:2], spec) + a * dx[:2]
            cn = c.copy()
            cn[:2] = de.mu(tn, spec)
            wn = w.copy()
            wn[:2] += a * dx[2:]
            wn[2] -= a * dx[2:].sum()
            try:
                if np.all(wn > 0):
                    lawn = de.mean_field_law(de.Design(cn, wn, B), spec, want_grad=False)
                    rn = sn.functionals(lawn.p, None, lawn.t, hs, sn.cuts_from(res, spec.tmax))
                    eval_structure(rn)
                    yn, _ = sn.f1_outputs(rn, M)
                    Fn = np.concatenate([yn[:2] - np.array(T_FIXED), yn[3:] - rt[:2]])
                    if np.linalg.norm(Fn) < np.linalg.norm(F):
                        break
            except sn.DesignMapError:
                pass
            a *= 0.5
        c, w = cn, wn
        cuts = sn.cuts_from(res, spec.tmax)
    law = de.mean_field_law(de.Design(c, w, B), spec)
    res = sn.functionals(law.p, law.dp, law.t, hs, cuts)
    eval_structure(res)
    y, Jraw = sn.f1_outputs(res, M)
    F = np.concatenate([y[:2] - np.array(T_FIXED), y[3:] - rt[:2]])
    return c, w, res, Jraw, sn.cuts_from(res, spec.tmax), float(np.max(np.abs(F)))


def mf_branch(eps, B, *, tmax=None, n_grid: int = 41, beyond: bool = True):
    spec = de.path_spec(eps, tmax=tmax or TMAX[eps])
    hs = bandwidths(spec)
    T = (T_FIXED[0], T_FIXED[1], T3_ANCHOR)
    ws = sn.mf_warm_start(T, target_ratios(), B, spec, hs)
    c = de.mu(ws["tp"], spec)
    w, cuts = ws["w"], ws["cuts"]
    c_anchor = float(c[2])
    sz = sn.sigma_z(spec)
    down = list(np.linspace(c_anchor, 0.0, n_grid)) + ([-0.25 * sz, -0.5 * sz, -sz] if beyond else [])
    c_cap = float(de.mu(CAP, spec))
    up = list(np.linspace(c_anchor, c_cap, 11))[1:] if c_cap > c_anchor else []
    pts = []
    for leg in (down, up):
        c, w, cuts = de.mu(ws["tp"], spec), ws["w"], ws["cuts"]
        for c3 in leg:
            try:
                c, w, res, Jraw, cuts, rn = mf_reduced(c, w, B, spec, hs, c3, cuts)
            except (sn.DesignMapError, np.linalg.LinAlgError) as exc:
                pts.append({"c3": float(c3), "error": str(exc)})
                break
            pts.append(point_record(res, c, w, spec, Jraw, {"reduced_residual_inf": rn}))
    pts.sort(key=lambda p: -p["c3"])
    return {"eps": eps, "B": B, "tmax": spec.tmax, "h": hs.tolist(), "anchor_solution": {
        "stripe_times": ws["tp"].tolist(), "w": ws["w"].tolist(), "c3": c_anchor}, "points": pts}


def cmd_mf(args) -> dict:
    out = {"item": "uplift2 item 1 NU-D (mean-field continuation)", "driver": "code/fb_v2_timing_horizon.py mf",
           "fixed_targets": {"T12": T_FIXED, "ratios": "equal"}, "branches": {}}
    for eps in EPS_LIST:
        for B in B_LIST:
            br = mf_branch(eps, B)
            zb = [p for p in br["points"] if "T3" in p and p["c3"] == 0.0]
            out["branches"][f"eps{eps:g}_B{B:g}"] = br
            print(f"[mf] eps={eps} B={B}: T3(c3=zbar)={zb[0]['T3'] if zb else float('nan'):.4f} "
                  f"g_t min/max over branch {min(p['g_dT3_dt3'] for p in br['points'] if 'T3' in p):.3g}",
                  flush=True)
    # tmax sensitivity of the horizon point (late-basin truncation)
    sens = {}
    for eps in EPS_LIST:
        for B in B_LIST:
            row = {}
            for tm in (TMAX[eps], TMAX[eps] + 2.5):
                br = mf_branch(eps, B, tmax=tm, beyond=False)
                zb = [p for p in br["points"] if "T3" in p and p["c3"] == 0.0]
                row[f"tmax{tm:g}"] = zb[0]["T3"] if zb else None
            vals = [v for v in row.values() if v is not None]
            row["abs_diff"] = abs(vals[0] - vals[1]) if len(vals) == 2 else None
            sens[f"eps{eps:g}_B{B:g}"] = row
    out["tmax_sensitivity_T3_at_zbar"] = sens
    print("[mf] tmax sensitivity max |diff| =",
          max(r["abs_diff"] for r in sens.values() if r["abs_diff"] is not None), flush=True)
    de.write_json(OUT / "horizon_mean_field.json", out)
    return out



# ----------------------------------------------------------------------------
# Exact-law (SAA) continuation points
# ----------------------------------------------------------------------------


class RCell:
    """Reduced F1 problem at a fixed late-stripe centre c_3 (unknowns t'_1, t'_2, u_1, u_2)."""

    def __init__(self, eps, B, c3, spec, c, w, cuts, label, hs):
        self.eps, self.B, self.spec = eps, float(B), spec
        self.c3 = float(c3)
        self.c = np.array(c, float)
        self.c[2] = self.c3
        self.w = np.array(w, float)
        self.cuts = list(cuts)
        self.hs = np.asarray(hs, float)    # fixed per point (late bandwidth from the mean-field T_3 there)
        self.key = label
        self.status = "active"
        self.res = None
        self.F = None
        self.cand = None
        self.a = 1.0
        self.hist = []
        self.chunk_set = None

    def design(self, c=None, w=None):
        return de.Design(self.c if c is None else c, self.w if w is None else w, self.B, self.key)

    def outputs(self, law):
        res = sn.analyse(law, self.hs, self.cuts, jackknife=False)
        eval_structure(res)
        y, Jraw = sn.f1_outputs(res, M)
        F = np.concatenate([y[:2] - np.array(T_FIXED), y[3:] - target_ratios()[:2]])
        return res, F, Jraw

    def step(self, dz, a):
        cn = self.c.copy()
        cn[:2] = de.mu(de.t_of_c(self.c[:2], self.spec) + a * dz[:2], self.spec)
        wn = self.w.copy()
        wn[:2] += a * dz[2:]
        wn[2] -= a * float(np.sum(dz[2:]))
        return cn, wn

    @staticmethod
    def admissible(cn, wn):
        return bool(np.all(wn > 1e-7) and cn[0] > cn[1] > cn[2])


def reduced_newton(engine, cells, *, stage, log, tol=1e-10, it_max=12, max_halvings=6):
    for c in cells:
        c.cand, c.a = None, 1.0
    passes = 0
    while True:
        todo = [c for c in cells if c.status == "active"]
        if not todo or passes > it_max * (max_halvings + 1):
            for c in todo:
                c.status = "failed_maxiter"
            break
        designs = [c.design() if c.res is None else c.design(*c.cand) for c in todo]
        laws = engine.evaluate(designs, want_grad=True, chunk_sets=[c.chunk_set for c in todo])
        passes += 1
        for c, law in zip(todo, laws):
            try:
                res, F, Jraw = c.outputs(law)
                ok = True
            except (sn.DesignMapError, np.linalg.LinAlgError) as exc:
                ok, err = False, str(exc)
            if c.res is None:
                if not ok:
                    c.status = "failed_start"
                    c.hist.append({"stage": stage, "error": err})
                    continue
                c.res, c.F, c.Jraw, c.law = res, F, Jraw, law
                c.hist.append({"stage": stage, "res_inf": sn.residual_norm(F), "a": None})
            else:
                n0 = float(np.linalg.norm(c.F))
                n1 = float(np.linalg.norm(F)) if ok else np.inf
                if ok and n1 <= (1 - 1e-4 * c.a) * n0:
                    c.c, c.w = c.cand
                    c.res, c.F, c.Jraw, c.law = res, F, Jraw, law
                    c.cuts = sn.cuts_from(res, c.spec.tmax)
                    c.hist.append({"stage": stage, "res_inf": sn.residual_norm(F), "a": c.a})
                    c.cand = None
                else:
                    if sn.residual_norm(c.F) < 1e-8 and c.a == 1.0:
                        c.status = "converged"
                        continue
                    c.a *= 0.5
                    if c.a < 0.5 ** max_halvings:
                        c.status = "failed_linesearch"
                        continue
                    c.cand = c.step(c.dz, c.a)
                    continue
            if sn.residual_norm(c.F) < tol:
                c.status = "converged"
                continue
            if len([h for h in c.hist if h.get("stage") == stage]) > it_max:
                c.status = "failed_maxiter"
                continue
            _, JRR, _, _ = reduced_pieces(c.Jraw, c.c, c.spec)
            c.dz = np.linalg.solve(JRR, -c.F)
            c.a = 1.0
            c.cand = c.step(c.dz, c.a)
            while not RCell.admissible(*c.cand) and c.a > 0.5 ** max_halvings:
                c.a *= 0.5
                c.cand = c.step(c.dz, c.a)
        log(f"[{stage} pass {passes} {engine.pass_seconds[-1]:.0f}s] " + " | ".join(
            f"{c.key}:{c.status[:4]} {sn.residual_norm(c.F) if c.F is not None else float('nan'):.1e}"
            for c in cells))
    return cells


def continuation_c3(mfbr: dict, spec) -> list:
    """Exact-law continuation points: the anchor (T_3 ~ 2.8 in mean field), t'_anchor + 0.5, + 1, + 2,
    the relaxation centre z_bar (the supremum within (H4)) and one point beyond it (-sigma_Z)."""
    c_a = mfbr["anchor_solution"]["c3"]
    t_a = float(de.t_of_c(c_a, spec))
    cs = [c_a] + [float(de.mu(t_a + d, spec)) for d in (0.5, 1.0, 2.0)] + [spec.z_bar, spec.z_bar - sn.sigma_z(spec)]
    # the N11 window cap t'_3 = 3.5: capped horizon t_max(eps; h_c = 3.5) = T_3(t'_3 = 3.5) (T_3 increases in t'_3)
    cs.append(float(de.mu(CAP, spec)))
    return cs


def point_bandwidths(spec, T3):
    """Per-point bandwidths: passages 1, 2 at their targets, the late one at the given late-peak time."""
    return sn.bandwidths((T_FIXED[0], T_FIXED[1], float(T3)), spec)


def mf_start_at(mfbr, c3, spec, B):
    """Mean-field solution of the reduced system at exactly c3 (continued from the nearest branch point),
    with the point's bandwidths (late one from the nearest branch point's T_3)."""
    pts = [p for p in mfbr["points"] if "T3" in p]
    near = min(pts, key=lambda p: abs(p["c3"] - c3))
    hs = point_bandwidths(spec, near["T3"])
    law = de.mean_field_law(de.Design(near["c"], near["w"], B), spec, want_grad=False)
    r0 = sn.functionals(law.p, None, law.t, hs, [0.0] + near["valleys"] + [spec.tmax])
    c, w, res, Jraw, cuts, rn = mf_reduced(near["c"], near["w"], B, spec, hs, c3, sn.cuts_from(r0, spec.tmax))
    return c, w, cuts, hs, point_record(res, c, w, spec, Jraw, {"reduced_residual_inf": rn, "h": hs.tolist()})


def cmd_exact(args) -> dict:
    eps = float(args.eps)
    log = sn.Logger(OUT / "logs" / f"horizon_eps{eps:g}.log")
    spec = de.path_spec(eps, tmax=TMAX[eps])
    n_full, n_1 = int(float(args.n)), int(float(args.n1))
    cells, mf_pts, branches = [], {}, {}
    for B in B_LIST:
        br = mf_branch(eps, B)
        branches[f"B{B:g}"] = br
        for k, c3 in enumerate(continuation_c3(br, spec)):
            try:
                c, w, cuts, hs, rec = mf_start_at(br, c3, spec, B)
            except (sn.DesignMapError, np.linalg.LinAlgError) as exc:
                log(f"[mf-start] B={B} c3={c3:.4g} failed: {exc}")
                continue
            label = f"eps{eps:g}_B{B:g}_p{k}"
            mf_pts[label] = rec
            cells.append(RCell(eps, B, c3, spec, c, w, cuts, label, hs))
    t0 = time.time()
    with sn.StreamEngine(spec, n_full, tag=TAG_HORIZON, chunk=sn.CHUNK, group=sn.GROUP) as se:
        for c in cells:
            c.chunk_set = list(range(max(1, n_1 // sn.CHUNK)))
        reduced_newton(se, cells, stage=f"N{n_1}", log=log)
        for c in cells:
            c.chunk_set = None
            if c.status == "converged":
                c.status, c.res = "active", None
        reduced_newton(se, cells, stage=f"N{n_full}", log=log)
        meta = se.meta
    points = {}
    for c in cells:
        if c.status != "converged":
            points[c.key] = {"status": c.status, "c3": c.c3, "B": c.B, "hist": c.hist}
            continue
        res = sn.analyse(c.law, c.hs, c.cuts, jackknife=True)
        rec = point_record(res, c.c, c.w, spec, c.Jraw, {"status": c.status, "B": c.B, "hist": c.hist,
                                                         "reduced_residual_inf": sn.residual_norm(c.F),
                                                         "se_peaks": res["se"]["peaks"].tolist(),
                                                         "se_ratios": res["se"]["ratios"].tolist(),
                                                         "n_paths": c.law.n_paths, "h": c.hs.tolist(),
                                                         "mode_check": sn.mode_check(c.law, c.hs, res),
                                                         "mean_field_T3_same_c3": mf_pts[c.key]["T3"]})
        points[c.key] = rec
        log(f"[point] {c.key} c3={c.c3:.4g} t3'={rec['t3_nominal']:.3f} T3={rec['T3']:.4f}+-{rec['se_peaks'][2]:.4f} "
            f"g_t={rec['g_dT3_dt3']:.3g} g_c={rec['g_dT3_dc3']:.3g} (MF T3 {rec['mean_field_T3_same_c3']:.4f})")
    horizon = {}
    for B in B_LIST:
        zb = [p for p in points.values() if p.get("B") == B and p.get("c3") == spec.z_bar and "T3" in p]
        branch = sorted([p for p in points.values() if p.get("B") == B and "T3" in p], key=lambda p: -p["c3"])
        inside = [p for p in branch if p["c3"] >= spec.z_bar]
        horizon[f"B{B:g}"] = {
            "t_max": zb[0]["T3"] if zb else None, "t_max_se": zb[0]["se_peaks"][2] if zb else None,
            "T3_monotone_in_t3_inside_H4": bool(all(b["T3"] > a["T3"] for a, b in zip(inside[:-1], inside[1:]))),
            "g_t_positive_inside_H4": bool(all(p["g_dT3_dt3"] > 0 for p in inside if p["c3"] > spec.z_bar)),
            "g_c_negative_through_zbar": bool(all(p["g_dT3_dc3"] < 0 for p in branch)),
            "anchor_2p8_reachable": bool(zb and zb[0]["T3"] > T3_ANCHOR),
            "capped_t_max_h3p5": next((p["T3"] for p in branch if abs(p["t3_nominal"] - CAP) < 1e-9), None),
            "capped_t_max_h3p5_se": next((p["se_peaks"][2] for p in branch if abs(p["t3_nominal"] - CAP) < 1e-9), None),
            "mean_field_t_max": next((p["T3"] for p in branches[f"B{B:g}"]["points"]
                                      if "T3" in p and p["c3"] == 0.0), None)}
        log(f"[horizon] eps={eps} B={B}: {horizon[f'B{B:g}']}")
    payload = {"item": "uplift2 item 1 NU-D (timing horizon, exact law)", "driver": "code/fb_v2_timing_horizon.py exact",
               "eps": eps, "tmax": spec.tmax, "n_paths": n_full, "stage1_paths": n_1, "engine": meta,
               "fixed_targets": {"T12": T_FIXED, "ratios": "equal (1/3)"}, "bandwidths_h": bandwidths(spec).tolist(),
               "points": points, "horizon": horizon, "mean_field_branches": branches,
               "seconds": time.time() - t0}
    de.write_json(OUT / f"horizon_exact_eps{eps:g}.json", payload)
    return payload


def _fits(eps, t, se=None):
    x = np.log(1.0 / np.asarray(eps, float))
    t = np.asarray(t, float)
    wts = None if se is None else 1.0 / np.maximum(np.asarray(se, float), 1e-4) ** 2
    out = {}
    # (a) slope fixed at 1/gamma = 1
    c_a = float(np.average(t - x, weights=wts))
    out["fixed_slope"] = {"slope": 1.0, "c": c_a, "residuals": (t - x - c_a).tolist(),
                          "max_abs_residual": float(np.max(np.abs(t - x - c_a)))}
    # (b) free slope
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.eye(len(x)) if wts is None else np.diag(wts)
    coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ t)
    out["free_slope"] = {"slope": float(coef[0]), "c": float(coef[1]),
                         "residuals": (t - A @ coef).tolist(), "max_abs_residual": float(np.max(np.abs(t - A @ coef)))}
    # (c) slope 1 with the stripe-at-centre correction -1/2 ln ln(1/eps)
    xc = x - 0.5 * np.log(x)
    c_c = float(np.average(t - xc, weights=wts))
    out["fixed_slope_loglog"] = {"c": c_c, "residuals": (t - xc - c_c).tolist(),
                                 "max_abs_residual": float(np.max(np.abs(t - xc - c_c)))}
    return out


def stripe_at_centre_time(eps: float, B: float, w3: float, T_guess: float) -> dict:
    """Heuristic closed form for the late-peak time with the late stripe at z_bar (mean-field peak
    balance d/dt ln phi_sigma(mu(t)) = kill rate): with x = |mu(t) - z_bar| / sigma,
        gamma x^2 = B w_3 cbar e^{-x^2/2} / (sqrt(2 pi) sigma W^{d-1}),   sigma = eps s~,
    i.e. x^2 e^{x^2/2} = Lambda / eps with Lambda = B w_3 cbar / (sqrt(2 pi) gamma s~ W^{d-1}), and
        T = gamma^{-1} [ ln(|z0 - z_bar| / (eps s~)) - ln x ].
    cbar = EM contact probability at T (two fixed-point passes).  Survival of the late basin neglected."""
    spec = de.path_spec(eps, tmax=TMAX.get(eps, 8.0))
    st = sn.sigma_z(spec) / eps
    gate = fk.contact_probability_em(spec)
    tg = fk.step_times(spec.dt, spec.steps())
    T = float(T_guess)
    for _ in range(3):
        cbar = float(np.interp(T, tg, gate))
        Lam = B * w3 * cbar / (math.sqrt(2 * math.pi) * spec.gamma * st * spec.torus_w ** spec.n_perp)
        q = Lam / eps
        x2 = max(2 * math.log(q), 1e-6)
        for _ in range(100):                   # Newton on x2 + 2 ln x2 = 2 ln q
            f = x2 + 2 * math.log(x2) - 2 * math.log(q)
            x2 -= f / (1 + 2 / x2)
            x2 = max(x2, 1e-9)
        T = (math.log(abs(spec.z0 - spec.z_bar) / (eps * st)) - 0.5 * math.log(x2)) / spec.gamma
    return {"T": T, "x": math.sqrt(x2), "Lambda": Lam, "cbar": cbar, "s_tilde": st}


def capped_first_order(eps: float, B: float) -> dict:
    """First-order capped horizon of the timing-horizon corollary (theory/V2_theorem4.tex, gd:cor-horizon (i)):
    t_cap = h_c + eps kappa + o(eps), kappa = rho y_*(lambda_3) / v(h_c), lambda_3 = B w_3 / (A v(h_c)), with the
    limit weights w = G_1(T^c, B, r*) of Prop. 1 at T^c = (T_1, T_2, h_c) and equal ratios (h_c = CAP)."""
    import fb_n11_shift_compensation as n11
    spec = de.path_spec(eps, tmax=TMAX.get(eps, 8.0))
    v = float(spec.gamma * abs(spec.z0 - spec.z_bar) * math.exp(-spec.gamma * CAP))
    w3 = float(sn.prop1_weights((T_FIXED[0], T_FIXED[1], CAP), target_ratios(), B)["w"][2])
    lam = B * w3 / (spec.torus_w ** spec.n_perp * v)
    y = float(n11.ystar(lam))
    kappa = spec.rho * y / v
    return {"T": CAP + eps * kappa, "kappa": kappa, "lambda3": lam, "w3_limit": w3, "y_star": y, "v_cap": v,
            "eps_rho_over_v": eps * spec.rho / v}


def cmd_fit(args) -> dict:
    rows = {}
    for eps in EPS_LIST:
        p = OUT / f"horizon_exact_eps{eps:g}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for B in B_LIST:
            h = d["horizon"][f"B{B:g}"]
            zb = [q for q in d["points"].values() if q.get("B") == B and q.get("c3") == 0.0 and "T3" in q]
            formula = stripe_at_centre_time(eps, B, zb[0]["w"][2], zb[0]["T3"]) if zb else None
            cap_fo = capped_first_order(eps, B)
            rows.setdefault(f"B{B:g}", []).append({"eps": eps, "t_max": h["t_max"], "se": h["t_max_se"],
                                                  "stripe_at_centre_formula": formula,
                                                  "mean_field_t_max": h["mean_field_t_max"],
                                                  "capped_t_max_h3p5": h["capped_t_max_h3p5"],
                                                  "capped_t_max_h3p5_se": h["capped_t_max_h3p5_se"],
                                                  "capped_first_order": cap_fo,
                                                  "capped_ratio_to_first_order": (
                                                      (h["capped_t_max_h3p5"] - CAP) / (cap_fo["T"] - CAP)
                                                      if h["capped_t_max_h3p5"] is not None else None),
                                                  "monotone": h["T3_monotone_in_t3_inside_H4"],
                                                  "g_t_positive": h["g_t_positive_inside_H4"]})
    fits = {}
    for key, r in rows.items():
        r = [x for x in r if x["t_max"] is not None]
        e = [x["eps"] for x in r]
        fits[key] = {"exact": _fits(e, [x["t_max"] for x in r], [x["se"] for x in r]),
                     "mean_field": _fits(e, [x["mean_field_t_max"] for x in r]), "rows": r,
                     "formula_minus_exact_max_abs": float(np.max([abs(x["stripe_at_centre_formula"]["T"] - x["t_max"])
                                                                  for x in r if x["stripe_at_centre_formula"]]))}
        print(f"[fit] {key}: exact slope-1 c={fits[key]['exact']['fixed_slope']['c']:.3f} "
              f"max|res|={fits[key]['exact']['fixed_slope']['max_abs_residual']:.3f}; free slope "
              f"{fits[key]['exact']['free_slope']['slope']:.3f}; loglog max|res| "
              f"{fits[key]['exact']['fixed_slope_loglog']['max_abs_residual']:.3f}; capped ratio "
              f"{[round(x['capped_ratio_to_first_order'], 3) for x in r if x['capped_ratio_to_first_order'] is not None]}",
              flush=True)
    payload = {"item": "uplift2 item 1 NU-D (timing-horizon scaling fit)", "driver": "code/fb_v2_timing_horizon.py fit",
               "model": "t_max = gamma^-1 ln(1/eps) + c (gamma = 1); free slope; slope 1 with -1/2 ln ln(1/eps)",
               "fits": fits}
    de.write_json(OUT / "horizon_fit.json", payload)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mf")
    s = sub.add_parser("exact")
    s.add_argument("--eps", required=True)
    s.add_argument("--n", default="2.5e5")
    s.add_argument("--n1", default="5e4")
    sub.add_parser("fit")
    args = ap.parse_args(argv)
    {"mf": cmd_mf, "exact": cmd_exact, "fit": cmd_fit}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
