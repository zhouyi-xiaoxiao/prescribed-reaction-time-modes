#!/usr/bin/env python3
"""TH-8b numerical check: saddle-node (fold) normal form of the EXACT law near
the finite-width fold (slab width 0.3, m = 2, equal weights).

Theorem p4:thm-fold (theory/TH8b_fold_certified.tex) predicts, for small eps,
a generic saddle-node of the critical points of f_eps(.;B) at
(t_{f,eps}, B_fold(eps)) = (t_f, B_top^det) + O(eps^2):
  * below the fold the late maximum t^+ and the valley t^- are separated by
        t^+ - t^- = 2 sqrt(2 (B_fold - B)/kappa_eps) (1 + O(sqrt(B_fold - B))),
    kappa_eps = -r0''(t_f) + O(eps^2) = 482.4 + O(eps^2) (certified value);
  * the fold statistic D_h(B) (largest log-slope after the first passage)
    crosses zero transversally, with slope -> -G0(t_f) = -0.26477.
This script re-analyses the stored N9b ensembles (no new simulation; seeds of
N9b: base 20260923, tag 87) on a fine budget grid below the fold, smooths the
exact bin masses with a Gaussian of width h (as in N9b), locates the local
minimum t^- and maximum t^+ of f_h on the late flank, and fits
    (t^+ - t^-)^2 = c (B_0 - B)     (least squares over B in a window),
    log(t^+ - t^-) = p log(B_0 - B) + const,
reporting kappa_eff = 8/c, B_0 and the exponent p; the same statistic is
applied to the deterministic law f_det binned on the same grid (like-for-like
at each h) and, unsmoothed, to f_det itself (t^+ - t^- from r0 = B).

Run from code/:  python3 fb_th8b_fold_normal_form.py [post]
("post" re-runs only the cheap post-processing on the stored JSON.)
Output: ../artifacts/data/exact_m_fixed_budget/TH8_certificate/fold_normal_form.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import fb_n9b_fold as n9b  # noqa: E402

OUT = fk.FB_DATA / "TH8_certificate" / "fold_normal_form.json"
EPS_LIST = (0.06, 0.03, 0.02)
H_LIST = (0.02, 0.03)
B_FINE = np.round(np.arange(6.0, 8.30 + 1e-9, 0.01), 10)
TQ = np.arange(1.40, 2.30 + 1e-12, 0.0005)
FLANK = (1.45, 2.30)          # late rising flank region searched for t^-, t^+
FIT_GAP = (0.08, 1.2)         # fit window: B_0 - B in this range (B_0 from a first pass)
CERT = fk.FB_DATA / "TH8_certificate" / "certificate.json"


def smooth(P, tc, h, tq):
    d = tq[:, None] - tc[None, :]
    K = np.exp(-0.5 * (d / h) ** 2)
    return P @ K.T                      # (nB, nq), normalisation irrelevant


def late_pair(fh_row, tq):
    """(t^-, t^+) = last local min followed by a local max on the late flank, or None."""
    sel = (tq >= FLANK[0]) & (tq <= FLANK[1])
    y, t = fh_row[sel], tq[sel]
    dy = np.diff(y)
    s = np.sign(dy)
    mins = [i for i in range(1, s.size) if s[i - 1] < 0 and s[i] > 0]
    maxs = [i for i in range(1, s.size) if s[i - 1] > 0 and s[i] < 0]
    if not mins or not maxs:
        return None
    i_min = mins[0]
    later = [j for j in maxs if j > i_min]
    if not later:
        return None
    j = later[0]

    def vertex(k):   # parabolic refinement on the fine grid
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(t[k] + off * (t[1] - t[0]))
    return vertex(i_min), vertex(j)


def fit_pairs(Bs, pairs):
    B = np.array([b for b, p in zip(Bs, pairs) if p is not None])
    dt = np.array([p[1] - p[0] for p in pairs if p is not None])
    if B.size < 5:
        return None
    # first pass: linear fit of dt^2 on B over the last 30 points
    A = np.vstack([B, np.ones_like(B)]).T
    k0 = max(0, B.size - 30)
    c1, c0 = np.linalg.lstsq(A[k0:], dt[k0:] ** 2, rcond=None)[0]
    B0 = -c0 / c1
    sel = (B0 - B >= FIT_GAP[0]) & (B0 - B <= FIT_GAP[1])
    if sel.sum() < 5:
        return None
    for _ in range(3):           # iterate the window with the refitted B0
        c1, c0 = np.linalg.lstsq(A[sel], dt[sel] ** 2, rcond=None)[0]
        B0 = -c0 / c1
        sel = (B0 - B >= FIT_GAP[0]) & (B0 - B <= FIT_GAP[1])
    res = dt[sel] ** 2 - (c1 * B[sel] + c0)
    x, yv = np.log(B0 - B[sel]), np.log(dt[sel])
    p, q = np.polyfit(x, yv, 1)
    return {"B0": float(B0), "kappa_eff=8/c": float(8.0 / (-c1)), "c": float(-c1),
            "loglog_exponent_p": float(p), "n_fit": int(sel.sum()),
            "fit_window_B": [float(B[sel].min()), float(B[sel].max())],
            "rms_residual_dt2": float(np.sqrt(np.mean(res ** 2))),
            "last_B_with_pair": float(B.max())}


def main():
    t0 = time.time()
    cert = json.loads(CERT.read_text())["cases"]["m2_rho0.3"]
    out = {"script": "code/fb_th8b_fold_normal_form.py",
           "item": "TH-8b numerical check of the saddle-node normal form (exact law)",
           "reuses_ensembles": {f"{e:g}": n9b.ens_name(e) for e in EPS_LIST},
           "seeds": {"base_seed": 20260923, "tag": 87, "note": "N9b ensembles, no new randomness"},
           "B_grid": [float(B_FINE[0]), float(B_FINE[-1]), 0.01], "tq_step": 0.0005,
           "flank_region": list(FLANK), "fit_gap_window": list(FIT_GAP),
           "certified_reference": {"B_top_det": cert["B_top_det"],
                                   "kappa_det=-r0pp(t_f)": cert["fold"]["a_f=-r0pp(t_f)"],
                                   "t_f": cert["D3"]["flanks"][0]["t_fold"]},
           "eps": {}, "deterministic": {}}
    # deterministic law: unsmoothed exact pair from r0 = B, and binned+smoothed
    edges = None
    for eps in EPS_LIST:
        ens = fk.load_ensemble(n9b.ens_name(eps))
        ens.cache_in_memory = True
        edges = ens.edges
        tc = 0.5 * (edges[:-1] + edges[1:])
        tab = fk.survival_table(ens, B_FINE, n9b.W, index=np.arange(edges.size), groups=n9b.GROUPS)
        P = tab["P_group_sums"].sum(0) / tab["N"]
        rec = {"n_paths": int(tab["N"]), "by_h": {}}
        for h in H_LIST:
            fh = smooth(P, tc, h, TQ)
            pairs = [late_pair(fh[i], TQ) for i in range(B_FINE.size)]
            rec["by_h"][f"{h:g}"] = {"fit": fit_pairs(B_FINE, pairs),
                                    "pairs_sample": {f"{B_FINE[i]:.2f}": pairs[i]
                                                     for i in range(0, B_FINE.size, 25)}}
        out["eps"][f"{eps:g}"] = rec
        ens.clear_cache()
        print(f"eps={eps}: done {time.time() - t0:.1f}s", flush=True)
    tc = 0.5 * (edges[:-1] + edges[1:])
    Pd = n9b.det_bin_masses(B_FINE, edges)
    for h in H_LIST:
        fh = smooth(Pd, tc, h, TQ)
        pairs = [late_pair(fh[i], TQ) for i in range(B_FINE.size)]
        out["deterministic"][f"binned_h{h:g}"] = {"fit": fit_pairs(B_FINE, pairs)}
    # unsmoothed f_det: t^+/- solve r0(t) = B on the flank (float64, fine grid)
    t = np.linspace(1.45, 2.3, 170001)
    mu = 4.0 * np.exp(-t)
    cs = [4.0 * math.exp(-1.0), 4.0 * math.exp(-2.5)]
    s = 0.3
    G = sum(0.5 * np.exp(-0.5 * ((mu - c) / s) ** 2) for c in cs) / (math.sqrt(2 * math.pi) * s)
    Gp = sum(0.5 * np.exp(-0.5 * ((mu - c) / s) ** 2) * (-(mu - c) / s ** 2) * (-mu) for c in cs) / (
        math.sqrt(2 * math.pi) * s)
    r0 = Gp / G ** 2
    pairs = []
    for B in B_FINE:
        sg = np.sign(r0 - B)
        idx = np.where(np.diff(sg) != 0)[0]
        pairs.append((float(t[idx[0]]), float(t[idx[1]])) if idx.size >= 2 else None)
    out["deterministic"]["unsmoothed_r0_level_set"] = {"fit": fit_pairs(B_FINE, pairs)}
    out["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


def post():
    """Cheap post-processing (no ensemble access): slope of the stored N9b fold
    statistic D_h(B) at its zero, and like-for-like eps-convergence of the
    fitted square-root-law constants.  Appends keys Dslope, convergence."""
    out = json.loads(OUT.read_text())
    n9 = json.loads((fk.FB_DATA / "N9b" / "n9b_fold.json").read_text())
    ds = {}
    for e in ("0.06", "0.03", "0.02"):
        E = n9["eps"][e]
        B = np.array(E["B_grid"])
        for h in ("0.02", "0.03", "0.05"):
            D = np.array(E["by_h"][h]["D"])
            bs = E["by_h"][h]["B_star"]
            i = int(np.argmin(abs(B - bs)))
            sl = slice(max(i - 3, 0), i + 4)
            pc = np.polyfit(B[sl], D[sl], 2)
            ds[f"eps{e}_h{h}"] = {"B_star": bs, "dD_dB_at_B_star": float(np.polyval(np.polyder(pc), bs)),
                                  "method": "quadratic fit over 7 grid points around B_star"}
    cert = json.loads(CERT.read_text())["cases"]["m2_rho0.3"]
    ds["prediction_eps_h_to_0"] = {"-G0(t_f)": cert["D3"]["flanks"][0]["G0_at_fold"],
                                   "note": "envelope theorem: d/dB max_t G0 (r0 - B) = -G0(t_f) at the fold"}
    out["Dslope"] = ds
    conv = {}
    for h in ("0.02", "0.03"):
        det = out["deterministic"][f"binned_h{h}"]["fit"]
        rows = {}
        for e in ("0.06", "0.03", "0.02"):
            f = out["eps"][e]["by_h"][h]["fit"]
            rows[e] = {"kappa_eff_minus_det": f["kappa_eff=8/c"] - det["kappa_eff=8/c"],
                       "B0_minus_det": f["B0"] - det["B0"]}
        q = {}
        for key in ("kappa_eff_minus_det", "B0_minus_det"):
            a, b, c = rows["0.06"][key], rows["0.03"][key], rows["0.02"][key]
            q[key] = {"q_0.06_0.03": float(np.log(a / b) / np.log(2.0)),
                      "q_0.03_0.02": float(np.log(b / c) / np.log(1.5))}
        conv[f"h{h}"] = {"rows": rows, "local_exponents": q}
    out["convergence"] = conv
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"Dslope": ds, "convergence": conv}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "post":
        post()
    else:
        main()
        post()
