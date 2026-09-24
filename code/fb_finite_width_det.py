#!/usr/bin/env python3
"""TH-8 deterministic check: finite-width stripes, deterministic threshold B_top^det.

Fixed physical stripe width rho_s (independent of the noise eps).  In the
small-noise limit the reaction-time density converges (TH-8, part (i)) to

    f_det(t; B) = B G0(t) exp(-B Lambda0(t)),   Lambda0(t) = int_0^t G0,
    G0(t) = W^{-(d-1)} sum_j w_j phi_{rho_s}(mu(t) - mu(t_j)),

and f_det'(t) = B e^{-B Lambda0} (G0' - B G0^2), so the stationary points of
f_det are the solutions of r0(t) := G0'(t)/G0(t)^2 = B.  This script computes,
for the paper's anchor geometry (gamma = D0 = 1, z0 = 4, zbar = 0, W = 1,
d = 2, window I = [0.5, 3.5]):

  * the stationary signature of G0 on I (hypothesis D1),
  * monotonicity of r0 on the first rising flank [tau, p_1] (D2),
  * the number of critical points of r0 on every interior rising flank and
    the nondegeneracy of its maximum (D3, fold transversality),
  * B_top^det = min(r0(tau), max_{flank_j} r0), the binding flank, the fold
    time, the deterministic basin masses at the fold (nominal cut at the
    free-clock valleys v_j, and the cut at the fold time t_f, which is the
    limit of the f_det valley as B increases to the fold),
  * the stationary points of f_det at selected budgets, and their location
    relative to the G0 maxima p_j and to the target times t_j,
  * the effective-width identity B_top^mf(eps) = B_top^det(eps * S_*) for the
    contact-one mean-field law (compared with the stored D1 values, which
    include the contact factor).

The D1-D3 flags come from a float64 grid scan (spacing 2e-6) with bisection
refinement: they support the hypotheses at the sampled widths but are not an
interval-certified proof.  D3 requires exactly one sign change of r0' on the
flank AND a transversal fold (r0'' < 0 there, central difference).

Pure numpy, deterministic, no random numbers.  Run from code/:
    python3 fb_finite_width_det.py
Output: ../artifacts/data/exact_m_fixed_budget/TH8/finite_width_det.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

GAMMA, D0, Z0, ZBAR, W, DIM = 1.0, 1.0, 4.0, 0.0, 1.0, 2
TAU, TEND = 0.5, 3.5
TARGETS = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}
S_STAR = math.sqrt(D0 / (2.0 * GAMMA) + 1.0)  # S_* with rho = 1 (paper anchor)
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "TH8" / "finite_width_det.json"
D1_JSON = HERE.parent / "artifacts" / "data" / "exact_m_prr_upgrade" / "mean_field_topology.json"


def mu(t):
    return ZBAR + (Z0 - ZBAR) * np.exp(-GAMMA * t)


def g0_derivs(t, m, rho_s, w=None):
    """G0, G0', G0'' (exact differentiation of the Gaussian stripe mixture along mu)."""
    w = np.full(m, 1.0 / m) if w is None else np.asarray(w, float)
    c = np.array([mu(tj) for tj in TARGETS[m]])
    mt = mu(t)
    mup = -GAMMA * (mt - ZBAR)
    mupp = GAMMA**2 * (mt - ZBAR)
    pref = 1.0 / (W ** (DIM - 1) * math.sqrt(2.0 * math.pi) * rho_s)
    s2 = rho_s**2
    f0 = np.zeros_like(t)
    f1 = np.zeros_like(t)
    f2 = np.zeros_like(t)
    for wj, cj in zip(w, c):
        dj = mt - cj
        e = wj * np.exp(-dj**2 / (2.0 * s2))
        f0 += e
        f1 += e * (-dj * mup / s2)
        f2 += e * ((dj * mup / s2) ** 2 - (mup**2 + dj * mupp) / s2)
    return pref * f0, pref * f1, pref * f2


def sign_changes(y):
    s = np.sign(y)
    return np.flatnonzero(s[:-1] * s[1:] < 0)


def refine_root(fun, a, b, it=80):
    fa = fun(np.array([a]))[0]
    for _ in range(it):
        mid = 0.5 * (a + b)
        fm = fun(np.array([mid]))[0]
        if fa * fm <= 0:
            b = mid
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def analyse(m, rho_s, w=None, n=1_500_001, budgets=()):
    t = np.linspace(TAU, TEND, n)
    G, G1, G2 = g0_derivs(t, m, rho_s, w)
    # --- signature of G0 on I (D1)
    idx = sign_changes(G1)
    crit = []
    for i in idx:
        tc = refine_root(lambda x: g0_derivs(x, m, rho_s, w)[1], t[i], t[i + 1])
        g, g1, g2 = (float(v[0]) for v in g0_derivs(np.array([tc]), m, rho_s, w))
        crit.append(dict(t=tc, G0=g, G0pp=g2, type="max" if g2 < 0 else "min"))
    types = [c["type"] for c in crit]
    n_max = types.count("max")
    n_min = types.count("min")
    alternating = all(types[i] != types[i + 1] for i in range(len(types) - 1)) and (not types or types[0] == "max")
    sig_ok = (n_max == m and n_min == m - 1 and alternating and G1[0] > 0 and G1[-1] < 0)
    out = dict(m=m, rho_s=rho_s, weights=(list(np.full(m, 1.0 / m)) if w is None else list(w)),
               G0_critical_points=crit, G0_n_max=n_max, G0_n_min=n_min,
               G0_prime_tau=float(G1[0]), G0_prime_T=float(G1[-1]), D1_signature_ok=bool(sig_ok))
    if not sig_ok:
        out["note"] = "G0 does not have the complete m-signature on I; B_top^det undefined"
        return out
    r0 = G1 / G**2
    r0p = (G2 * G - 2.0 * G1**2) / G**3  # derivative of G0'/G0^2
    maxima = [c["t"] for c in crit if c["type"] == "max"]
    minima = [c["t"] for c in crit if c["type"] == "min"]
    # --- first flank [tau, p1] (D2): r0 strictly decreasing?
    mask1 = t <= maxima[0]
    d2_ok = bool(np.all(r0p[mask1] < 0))
    r0_tau = float(r0[0])
    flanks = [dict(kind="first", a=TAU, b=maxima[0], r0_tau=r0_tau, r0_strictly_decreasing=d2_ok,
                   n_r0_critical=int(len(sign_changes(r0p[mask1]))))]
    cand = [("left_endpoint", r0_tau, TAU)]
    d3_ok = True
    for j in range(1, m):
        a, b = minima[j - 1], maxima[j]
        msk = (t > a) & (t < b)
        sc = sign_changes(r0p[msk])
        tt = t[msk]
        ncrit = int(len(sc))
        rec = dict(kind="interior", j=j + 1, a=a, b=b, n_r0_critical=ncrit)
        if ncrit >= 1:
            i = sc[0]

            def r0p_fun(x):
                g, g1, g2 = g0_derivs(x, m, rho_s, w)
                return (g2 * g - 2.0 * g1**2) / g**3

            tf = refine_root(r0p_fun, tt[i], tt[i + 1])
            g, g1, g2 = (float(v[0]) for v in g0_derivs(np.array([tf]), m, rho_s, w))
            bj = g1 / g**2
            h = 1e-5
            r0pp = float((r0p_fun(np.array([tf + h]))[0] - r0p_fun(np.array([tf - h]))[0]) / (2 * h))
            rec.update(t_fold=tf, b_j=bj, r0pp_at_fold=r0pp, transversal=bool(r0pp < 0))
            cand.append((f"flank_{j + 1}", bj, tf))
        d3_ok &= (ncrit == 1) and bool(rec.get("transversal", False))
        flanks.append(rec)
    binding = min(cand, key=lambda c: c[1])
    out.update(flanks=flanks, D2_first_flank_monotone=d2_ok, D3_flank_unimodal=bool(d3_ok),
               D3_includes_transversality=True,
               B_top_det=float(binding[1]), binding=binding[0], fold_time=float(binding[2]),
               candidates={c[0]: float(c[1]) for c in cand})
    # --- Lambda0 on [0, T] and basin masses at the fold and at selected budgets
    tt = np.linspace(0.0, TEND, 2_000_001)
    g_all = g0_derivs(tt, m, rho_s, w)[0]
    lam = np.concatenate([[0.0], np.cumsum(0.5 * (g_all[1:] + g_all[:-1]) * np.diff(tt))])
    bounds = [TAU] + minima + [TEND]
    lam_b = np.interp(bounds, tt, lam)
    out["Lambda0_at_basin_bounds"] = dict(zip([f"{x:.6f}" for x in bounds], [float(x) for x in lam_b]))

    def basin_masses(B):
        S = np.exp(-B * lam_b)
        return [float(S[k] - S[k + 1]) for k in range(len(S) - 1)]

    out["basin_masses_at_B_top_det"] = basin_masses(out["B_top_det"])
    out["basin_masses_cut_note"] = "basin_masses*: nominal fixed cuts at the G0 minima v_j (free-clock valleys)"

    def f_det_stationary(B):
        fp = G1 - B * G**2  # sign of f_det'
        sc = sign_changes(fp)
        pts = []
        for i in sc:
            tc = refine_root(lambda x: g0_derivs(x, m, rho_s, w)[1] - B * g0_derivs(x, m, rho_s, w)[0] ** 2,
                             t[i], t[i + 1])
            pts.append(dict(t=tc, type="max" if fp[i] > 0 else "min"))
        return fp, pts

    # fold-time cut: as B increases to an interior-flank fold, the f_det valley of that flank tends to t_f,
    # so the limiting mass of the disappearing mode's own basin uses the cut at t_f instead of v_{j-1}.
    if out["binding"].startswith("flank_"):
        jb = int(out["binding"].split("_")[1])  # binding flank index (2..m)
        bounds_f = [TAU] + minima[: jb - 2] + [out["fold_time"]] + minima[jb - 1:] + [TEND]
        lam_f = np.interp(bounds_f, tt, lam)
        Sf = np.exp(-out["B_top_det"] * lam_f)
        out["basin_masses_at_B_top_det_fold_cut"] = dict(
            cuts=[float(x) for x in bounds_f], masses=[float(Sf[k] - Sf[k + 1]) for k in range(len(Sf) - 1)],
            note="cut at the fold time t_f of the binding flank (limit of the f_det valley as B -> B_top_det-)")
        near = []
        for dB in (1e-2, 1e-3, 1e-4):
            Bn = out["B_top_det"] - dB
            fpn, ptsn = f_det_stationary(Bn)
            mins_n = [p["t"] for p in ptsn if p["type"] == "min"]
            if len(mins_n) >= jb - 1:
                tv = mins_n[jb - 2]
                lv = np.interp([tv, TEND], tt, lam)
                near.append(dict(B=Bn, f_det_valley=float(tv),
                                 late_basin_mass_f_det_valley_cut=float(np.exp(-Bn * lv[0]) - np.exp(-Bn * lv[1]))))
        out["late_basin_mass_near_fold_f_det_valley_cut"] = near
    rows = []
    for B in budgets:
        fp, pts = f_det_stationary(B)
        rows.append(dict(B=B, f_det_prime_tau_sign=int(np.sign(fp[0])), f_det_prime_T_sign=int(np.sign(fp[-1])),
                         stationary_points=pts, n_max=sum(p["type"] == "max" for p in pts),
                         basin_masses=basin_masses(B)))
    out["f_det_at_budgets"] = rows
    if budgets:
        # location of the f_det maxima relative to the G0 maxima p_j and the target times t_j (TH-8 drift remark)
        loc = []
        for B in (0.001, 0.01, 0.1) + tuple(budgets):
            _, pts = f_det_stationary(B)
            recs = []
            for x in (p["t"] for p in pts if p["type"] == "max"):
                # a maximum sits on the rising flank of G0 that ends at the next G0 maximum p_j (index j)
                j = next((k for k, p in enumerate(maxima) if p >= x), None)
                recs.append(dict(t=float(x), flank_index=None if j is None else j + 1,
                                 G0_max=None if j is None else float(maxima[j]),
                                 target=None if j is None else float(TARGETS[m][j]),
                                 before_G0_max=bool(j is not None and x < maxima[j]),
                                 before_target=bool(j is not None and x < TARGETS[m][j])))
            loc.append(dict(B=B, maxima=recs))
        out["peak_location_check"] = dict(G0_maxima=maxima, targets=list(TARGETS[m]), rows=loc,
                                          note="each f_det maximum lies before the next G0 maximum p_j; it need not lie "
                                               "before the target t_j (first peak at small B lies after t_1 = 1)")
    return out


def main():
    t0 = time.time()
    res = dict(script="code/fb_finite_width_det.py", item="TH-8", deterministic=True, seeds=None,
               geometry=dict(gamma=GAMMA, D0=D0, z0=Z0, zbar=ZBAR, W=W, d=DIM, window=[TAU, TEND],
                             targets={str(k): v for k, v in TARGETS.items()}, contact="1 (eps->0 limit)"))
    scan = {}
    for m in (2, 3):
        for rho_s in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
            budgets = (1.0, 4.0, 7.8, 8.0, 8.2, 9.0) if (m == 2 and rho_s == 0.3) else ()
            scan[f"m{m}_rho{rho_s}"] = analyse(m, rho_s, budgets=budgets)
    res["scan"] = scan
    # effective-width identity: contact-one mean-field threshold = B_top^det at rho_s = eps * S_*
    eff = {}
    try:
        d1 = json.loads(D1_JSON.read_text())["headline"]["B_top_mf_3sf_by_label"]
    except Exception:  # pragma: no cover
        d1 = {}
    for m in (2, 3):
        for eps in (0.05, 0.075, 0.1):
            a = analyse(m, eps * S_STAR, n=1_500_001)
            eff[f"m{m}_eps{eps}"] = dict(rho_eff=eps * S_STAR, B_top_det_at_rho_eff=a.get("B_top_det"),
                                         binding=a.get("binding"),
                                         D1_B_top_mf_with_contact=d1.get(f"m{m}_eps{eps}"))
    res["effective_width_identity"] = eff
    # small-width asymptotics of B_top^det (m = 2, equal weights, flank 2):
    # ln B_top^det = Delta^2/(8 rho^2) + ln(1/rho) + C_rho + o(1),
    # C_rho = ln[ |mu'(t_s)| Delta W^{d-1} sqrt(2 pi) / (8 sqrt(w1 w2)) ], mu(t_s) = (mu1 + mu2)/2.
    c1, c2 = mu(1.0), mu(2.5)
    delta = abs(c1 - c2)
    ts = math.log((Z0 - ZBAR) / (0.5 * (c1 + c2) - ZBAR)) / GAMMA
    c_rho = math.log(GAMMA * 0.5 * (c1 + c2) * delta * W ** (DIM - 1) * math.sqrt(2 * math.pi) / (8 * 0.5))
    asym = dict(Delta=delta, t_s=ts, C_rho_predicted=c_rho, rows=[])
    rhos = [(k, v["rho_s"], v.get("B_top_det")) for k, v in scan.items() if k.startswith("m2_")]
    rhos += [(k, v["rho_eff"], v["B_top_det_at_rho_eff"]) for k, v in eff.items() if k.startswith("m2_")]
    for key, r, bt in sorted(rhos, key=lambda x: x[1]):
        if bt:
            resid = math.log(bt) - delta**2 / (8 * r**2) - math.log(1.0 / r)
            asym["rows"].append(dict(source=key, rho=r, B_top_det=bt, residual=resid))
    res["small_width_asymptotics_m2"] = asym
    res["runtime_seconds"] = time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
    for k, v in scan.items():
        print(k, "sig_ok", v.get("D1_signature_ok"), "B_top_det", v.get("B_top_det"), "binding", v.get("binding"),
              "D2", v.get("D2_first_flank_monotone"), "D3", v.get("D3_flank_unimodal"))
    for k, v in eff.items():
        print(k, v)


if __name__ == "__main__":
    main()
