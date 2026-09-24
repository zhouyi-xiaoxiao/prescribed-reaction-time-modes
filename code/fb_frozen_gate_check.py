#!/usr/bin/env python3
"""TH-7 numerical check: frozen-gate contact law (no new simulation).

Collects, in one citable JSON, the evidence for the frozen-gate proposition of
TH-7 from ensembles that already exist:

  (1) the analytic orthant law for the boundary-tangent relative geometry
      (r_par0 = 0, r_perp0 = a): chi_j = 1{eta(t_j) < 0}, Var eta(t) =
      sigma_perp0^2 + 4 D0 t, and the frozen-gate masses
      M_j = E[exp(-sum_{i<j} lam_i chi_i)(1 - exp(-lam_j chi_j))] with their
      large-B limits (gate-avoidance probabilities);
  (2) the exact-law Feynman-Kac (FK) basin masses of the full contact-gated
      model in that geometry (theory-scout ensembles, 2e5 paths, dt = 1e-3,
      seeds 31 [eps = 0.05] and 32 [eps = 0.025]) read from
      notes/gap_diagnosis_20260923/theory_scout/analysis_h3.json, and the
      mode counts of the FK density recomputed from the cached ensembles
      (~/.local-build/prr_gap_fk_cache/fkh3_m2_e*.npz) with the scout's
      count_modes (z > 5 on 4 batches);
  (3) the finite-eps frozen-gate surrogate in the production geometry
      (empirical chi at the target times) against the product (contact = 1)
      limit law, read from theory_scout/analysis_m{2,3}.json (seeds 11, 12);
  (4) the large-B contact re-entry check (4e4 Euler-Maruyama walkers,
      seed 12345) read from theory_checks/contact_escape_check.json.

Deterministic; runs in seconds.  Run from code/:
    python3 fb_frozen_gate_check.py
Output: ../artifacts/data/exact_m_fixed_budget/TH7/frozen_gate_check.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HUB = HERE.parent
NOTES = HUB / "notes" / "gap_diagnosis_20260923"
SCOUT = NOTES / "theory_scout"
CACHE = Path.home() / ".local-build" / "prr_gap_fk_cache"
OUT = HUB / "artifacts" / "data" / "exact_m_fixed_budget" / "TH7" / "frozen_gate_check.json"

GAMMA, D0, Z0, W = 1.0, 1.0, 4.0, 1.0
SIGP0 = 0.3  # transverse initial s.d. coefficient (Sigma_perp0 = 0.09)
TARGETS = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}


def vel(t):
    return GAMMA * Z0 * math.exp(-GAMMA * t)


def lambdas(m, B, w=None):
    w = [1.0 / m] * m if w is None else w
    return [B * w[j] / (W * vel(tj)) for j, tj in enumerate(TARGETS[m])]


def frozen_gate_masses(lam, law):
    """law: dict {tuple(chi): prob}.  Returns M_j = E[exp(-sum_{i<j} lam_i chi_i)(1-exp(-lam_j chi_j))]."""
    m = len(lam)
    out = np.zeros(m)
    for chi, p in law.items():
        cum = 0.0
        for j in range(m):
            e = lam[j] * chi[j]
            out[j] += p * math.exp(-cum) * (1.0 - math.exp(-e))
            cum += e
    return out.tolist()


def avoidance_limits(law, m):
    """B -> infinity limits: pi_j = P(chi_1 = ... = chi_{j-1} = 0, chi_j = 1)."""
    pis = []
    for j in range(m):
        pis.append(sum(p for chi, p in law.items() if chi[j] == 1 and all(c == 0 for c in chi[:j])))
    return pis


def orthant_law_m2():
    v = [SIGP0**2 + 4 * D0 * t for t in TARGETS[2]]
    rho = math.sqrt(v[0] / v[1])
    p11 = 0.25 + math.asin(rho) / (2 * math.pi)
    p01 = 0.5 - p11
    p10 = 0.5 - p11
    p00 = 1.0 - p11 - p01 - p10
    return dict(var_eta=v, rho12=rho, law={(1, 1): p11, (0, 1): p01, (1, 0): p10, (0, 0): p00})


def main():
    res = dict(script="code/fb_frozen_gate_check.py", item="TH-7", new_simulation=False,
               source_seeds=dict(h3_eps0_05=31, h3_eps0_025=32, prod_m2_eps0_1=11, prod_m3_eps0_1=12,
                                 em_contact_escape=12345))
    # ---------------- (1) orthant law
    ol = orthant_law_m2()
    budgets = [0.5, 1, 2, 4, 8, 20, 50, 200]
    res["orthant_law_m2"] = dict(
        geometry="boundary-tangent: r_par0 = 0, r_perp0 = a = 0.4, W = 1, d = 2, Sigma_perp0 = 0.09",
        var_eta=ol["var_eta"], rho12=ol["rho12"],
        P11=ol["law"][(1, 1)], P01=ol["law"][(0, 1)], P10=ol["law"][(1, 0)], P00=ol["law"][(0, 0)],
        masses={str(B): frozen_gate_masses(lambdas(2, B), ol["law"]) for B in budgets},
        large_B_limit=avoidance_limits(ol["law"], 2))
    # ---------------- (2) FK exact law in the tangent geometry
    h3 = json.loads((SCOUT / "analysis_h3.json").read_text())
    tang = {}
    sys.path.insert(0, str(SCOUT))
    try:
        from analyze_fk import count_modes  # scout's classifier-free significance counter
    except Exception:  # pragma: no cover
        count_modes = None
    for eps in ("0.05", "0.025"):
        rows = []
        for r in h3[eps]["rows"]:
            ex, orth, fe, mf = (np.array(r[k]) for k in ("exact", "orthant", "frozen_emp", "mf"))
            rows.append(dict(B=r["B"], exact=ex.tolist(), orthant=orth.tolist(), frozen_emp=fe.tolist(),
                             mean_field=mf.tolist(), exact_minus_orthant=(ex - orth).tolist(),
                             exact_minus_frozen_emp=(ex - fe).tolist()))
        modes = None
        npz = CACHE / f"fkh3_m2_e{eps}.npz"
        if count_modes is not None and npz.exists():
            d = np.load(npz)
            t = d["tgrid"]
            dt = float(d["dt"])
            FK = d["FK_full"] / dt
            FKb = d["FKb_full"] / dt
            h_steps = max(1.0, 0.15 * float(eps) * 1.2247 / 1.4715 / dt)
            modes = {}
            for bi, b in enumerate(d["budgets"]):
                cm = count_modes(t, FK[bi], FKb[:, bi, :], h_steps=h_steps)
                sig = [dict(t=x["t"], rel_prom=x["rel_prom"], z=x["z"]) for x in cm if x["z"] > 5]
                modes[str(float(b))] = dict(n_significant=len(sig), significant_maxima=sig, n_local_maxima=len(cm))
            modes["_meta"] = dict(npz=str(npz), seed=int(d["seed"]), N=int(d["N"]), dt=dt, h_steps=h_steps,
                                  criterion="z > 5 on 4 batch means (scout count_modes)")
        tang[eps] = dict(gate_stats=h3[eps]["gate"], rows=rows, modes=modes)
    res["tangent_geometry_fk"] = tang
    # ---------------- (3) production geometry surrogate at eps = 0.1
    prod = {}
    for m in (2, 3):
        d = json.loads((SCOUT / f"analysis_m{m}.json").read_text())
        e = [x for x in d if float(x["eps"]) == 0.1][0]
        rows = []
        for r in e["basin_masses"]:
            ex = np.array(r["exact_full"])
            lc = np.array(r["limit_c1"])
            fz = np.array(r["limit_frozen_contact"])
            mf = np.array(r["mf_full"])
            rows.append(dict(B=r["B"], exact=ex.tolist(), product_limit=lc.tolist(), frozen_surrogate=fz.tolist(),
                             mean_field=mf.tolist(),
                             abs_err_product=np.abs(ex - lc).tolist(), abs_err_surrogate=np.abs(ex - fz).tolist(),
                             sum_abs_err_product=float(np.abs(ex - lc).sum()),
                             sum_abs_err_surrogate=float(np.abs(ex - fz).sum()),
                             sum_abs_err_mean_field=float(np.abs(ex - mf).sum())))
        prod[f"m{m}"] = dict(eps=0.1, contact_prob_at_tj=e["contact_prob_at_tj"], rows=rows)
    res["production_geometry_surrogate"] = prod
    # ---------------- (4) large-B re-entry
    ce = json.loads((NOTES / "theory_checks" / "contact_escape_check.json").read_text())
    edges = np.linspace(0.5, 3.5, 61)
    cen = 0.5 * (edges[1:] + edges[:-1])
    big = {}
    for k, v in ce.items():
        h = np.array(v["hist"])
        msk = cen >= 1.5
        late = h * msk
        big[k] = dict(M1_window=v["M1_window"], M2_window=v["M2_window"], survivors=v["survivors"],
                      late_bump_argmax_t=(float(cen[int(np.argmax(late))]) if late.sum() > 0 else None),
                      late_counts=int(late.sum()))
    res["large_B_reentry"] = dict(source="notes/gap_diagnosis_20260923/theory_checks/contact_escape_check.json",
                                  walkers=40000, seed=12345, dt="2e-4 (B<=100), 5e-5 (B>100)",
                                  hist="60 bins on [0.5, 3.5]; late bump = argmax over t >= 1.5",
                                  cells=big)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
    o = res["orthant_law_m2"]
    print("orthant rho12=%.5f P11=%.5f P01=%.5f limit=%s" % (o["rho12"], o["P11"], o["P01"], o["large_B_limit"]))
    for eps, v in tang.items():
        if v["modes"]:
            print(eps, {b: (x["n_significant"], [(round(s["t"], 3), "%.2e" % s["rel_prom"], round(s["z"], 1))
                                                   for s in x["significant_maxima"]])
                        for b, x in v["modes"].items() if b != "_meta"})
    for m, v in prod.items():
        print(m, [(r["B"], round(r["sum_abs_err_product"], 4), round(r["sum_abs_err_surrogate"], 4),
                   round(r["sum_abs_err_mean_field"], 4)) for r in v["rows"]])


if __name__ == "__main__":
    main()
