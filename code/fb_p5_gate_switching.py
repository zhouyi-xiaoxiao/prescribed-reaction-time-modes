#!/usr/bin/env python3
"""P5 (theory closer): frozen-gate C^2 profile check and gate-switching correction.

Numerical companion of manuscript/cnsns_submission/theory/TH7b_frozen_gate_c2.tex.

Sub-commands (run from code/):
  jfun      Monte Carlo of the universal Brownian function
               J(lam) = int dnu E[exp(-lam Theta(nu)) - exp(-lam 1{nu<0})],
               Theta(nu) = int phi(x) 1{nu + B_x < 0} dx,  B two-sided standard BM, B_0 = 0,
            computed per path EXACTLY in nu (Theta is a step function of nu on a grid path),
            on several grid steps h (discretization check); small-lam check against
            -Gamma(3/4)/(2 pi) lam^2.   Seeds: SeedSequence([20260923, 90, h-index, round(1e6 h)]).spawn(n_chunks); stored data used --chunk 1000.
  predict   first-order corrected masses  M_j = Mgate_j + sqrt(eps) m_j  (Proposition p5:switch)
            and switching probabilities (Proposition p5:switchprob, continuous + discrete
            monitoring by Spitzer's identity) against the stored N3 tangent data
            (artifacts/data/exact_m_fixed_budget/N3/n3_summary.json) -- no new simulation.
  oracle    per-path decomposition on the cached tangent FK ensembles
            (~/.local-build/prr_gap_fk_cache/fkh3_m2_e{0.025,0.05}.npz; seeds 32, 31):
            FK - E G(B chi_tj Q_free) (switching part) and E G(B chi_tj Q_free) - E G(lam chi_tj)
            (midpoint part), with paired per-path standard errors.
  profile   Part A check: FK density near each passage vs the frozen-gate limit profile
            alpha_j (v_j/rho) F_lam(v_j s/rho) / eps (fkh3 ensembles).
  all       predict + oracle + profile (needs jfun output), writes p5_summary.json.

Outputs: ../artifacts/data/exact_m_fixed_budget/P5_gate_switching/*.json
CPU: single process, chunked; waits while >= 9 busy python processes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HUB = HERE.parent
OUTDIR = HUB / "artifacts" / "data" / "exact_m_fixed_budget" / "P5_gate_switching"
N3 = HUB / "artifacts" / "data" / "exact_m_fixed_budget" / "N3" / "n3_summary.json"
CACHE = Path.home() / ".local-build" / "prr_gap_fk_cache"
SEED, TAG = 20260923, 90

# tangent anchor (fk.EnsembleSpec(m=2, eps, r_par0=0, r_perp0=0.4)); verified in P5 notes
GAMMA, D0, Z0, ZBAR, W, A, RHO = 1.0, 1.0, 4.0, 0.0, 1.0, 0.4, 1.0
U0, SIGP0 = 0.3, 0.3
TJ = (1.0, 2.5)
WEIGHTS = (0.5, 0.5)
SZ2 = D0 / (2 * GAMMA)                      # stationary midpoint variance coefficient
KAPPA2 = math.gamma(0.75) / (2 * math.pi)    # J(lam) = -KAPPA2 lam^2 + O(lam^3)


def busy_python() -> int:
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True).stdout
    except Exception:
        return 0
    n = 0
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "python" in parts[1].lower():
            try:
                if float(parts[0]) > 20.0:
                    n += 1
            except ValueError:
                pass
    return n


def wait_for_cpu(limit: int = 9, max_wait: float = 1200.0) -> None:
    t0 = time.time()
    while busy_python() >= limit and time.time() - t0 < max_wait:
        time.sleep(30)


def vel(t: float) -> float:
    return GAMMA * abs(Z0 - ZBAR) * math.exp(-GAMMA * t)


def lambdas(B: float) -> list[float]:
    return [B * WEIGHTS[j] / (W * vel(t)) for j, t in enumerate(TJ)]


# ---------------------------------------------------------------------------
# J(lam) by Monte Carlo
# ---------------------------------------------------------------------------

def j_chunk(rng, n_paths: int, h: float, X: float, lams: np.ndarray):
    """Per-path J values (n_paths x n_lam) and per-path sum_nu |Theta - 1{nu<0}| (L1 check)."""
    n_side = int(round(X / h))
    x = h * np.arange(-n_side, n_side + 1)
    w = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) * h
    w[0] *= 0.5
    w[-1] *= 0.5
    w /= w.sum()
    inc_r = rng.standard_normal((n_paths, n_side)) * math.sqrt(h)
    inc_l = rng.standard_normal((n_paths, n_side)) * math.sqrt(h)
    Br = np.cumsum(inc_r, axis=1)
    Bl = np.cumsum(inc_l, axis=1)[:, ::-1]
    B = np.concatenate([Bl, np.zeros((n_paths, 1)), Br], axis=1)
    y = -B                                   # nu < y_k  <=>  B_k < -nu
    order = np.argsort(-y, axis=1)           # descending y
    ys = np.take_along_axis(y, order, axis=1)
    ws = w[order]
    Wc = np.cumsum(ws, axis=1)               # Theta on [y_(n+1), y_(n))
    gaps = ys[:, :-1] - ys[:, 1:]            # >= 0
    ymax, ymin = ys[:, 0], ys[:, -1]
    out = np.empty((n_paths, lams.size))
    for li, lam in enumerate(lams):
        integ = (gaps * np.exp(-lam * Wc[:, :-1])).sum(axis=1)
        out[:, li] = integ - ymax - (-ymin) * math.exp(-lam)
    # L1 functional int dnu |Theta(nu) - 1{nu<0}| (analytic mean sqrt(2/pi) E|N|^{1/2} = 0.65600)
    # Theta = theta on the nu-interval (lo, hi) = (ys[k+1], ys[k]); the reference 1{nu<0} is 1 on
    # (lo, min(hi,0)) and 0 on (max(lo,0), hi) -- the one interval straddling nu=0 is split there
    # (fix of audit P5-N-1, 2026-09-23; J itself treats the reference analytically and is unchanged).
    # NB: x=0 is a grid node with B_0=0, so y=0 is always a breakpoint and no interval actually straddles 0;
    # the split is therefore numerically inert (verified bit-for-bit on run 0, 2026-09-23).
    theta = Wc[:, :-1]
    lo, hi = ys[:, 1:], ys[:, :-1]
    len_neg = np.clip(np.minimum(hi, 0.0) - lo, 0.0, None)
    len_pos = np.clip(hi - np.maximum(lo, 0.0), 0.0, None)
    l1 = (len_neg * np.abs(theta - 1.0) + len_pos * np.abs(theta)).sum(axis=1)
    return out, l1


def cmd_jfun(args):
    lams = np.array(args.lams, float)
    res = {"lams": lams.tolist(), "X": args.X, "runs": [], "kappa2": KAPPA2,
           "seed": [SEED, TAG], "note": "per-path exact nu-integration on grid paths"}
    for hi, h in enumerate(args.hs):
        wait_for_cpu()
        t0 = time.time()
        vals, l1s = [], []
        n_chunks = int(math.ceil(args.paths / args.chunk))
        ss = np.random.SeedSequence([SEED, TAG, hi, int(round(1e6 * h))])
        for ci, child in enumerate(ss.spawn(n_chunks)):
            rng = np.random.Generator(np.random.Philox(child))
            v, l1 = j_chunk(rng, args.chunk, h, args.X, lams)
            vals.append(v)
            l1s.append(l1)
        V = np.concatenate(vals)
        L1 = np.concatenate(l1s)
        mean = V.mean(0)
        se = V.std(0, ddof=1) / math.sqrt(V.shape[0])
        res["runs"].append({"h": h, "paths": int(V.shape[0]), "J": mean.tolist(), "se": se.tolist(),
                            "L1_mean": float(L1.mean()), "L1_se": float(L1.std(ddof=1) / math.sqrt(L1.size)),
                            "seconds": time.time() - t0,
                            "seed_entropy": [SEED, TAG, hi, int(round(1e6 * h))]})
        print(f"h={h}: paths={V.shape[0]} time={time.time()-t0:.1f}s", flush=True)
        for lam, mu, s in zip(lams, mean, se):
            print(f"   lam={lam:8.3f}  J={mu:+.5f} +- {s:.5f}   -k2 lam^2={-KAPPA2*lam*lam:+.5f}", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / args.out
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)



# ---------------------------------------------------------------------------
# first-order switching correction (Proposition p5:switch) and predictions
# ---------------------------------------------------------------------------

def Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def g_eta(t: float, b: float = 0.0) -> float:
    V = SIGP0**2 + 4 * D0 * t
    return math.exp(-b * b / (2 * V)) / math.sqrt(2 * math.pi * V)


def ell(t: float) -> float:
    return 2.0 * math.sqrt(D0 * RHO / vel(t))


def load_J(path: Path, h: float | None = None):
    d = json.loads(path.read_text())
    runs = d["runs"]
    run = min(runs, key=lambda r: r["h"]) if h is None else min(runs, key=lambda r: abs(r["h"] - h))
    lams = np.array(d["lams"])
    J = np.array(run["J"])
    se = np.array(run["se"])

    def f(lam):
        i = int(np.argmin(np.abs(lams - lam)))
        if abs(lams[i] - lam) > 1e-5 * max(1.0, lam):
            raise KeyError(f"lambda {lam} not tabulated")
        return float(J[i]), float(se[i])
    return f, run["h"]


def first_order_m2(lam, Jf, eps, far=False):
    """sqrt(eps) * (m_1, m_2) for m = 2, tangent geometry; far=True adds the wrap boundary
    b_+ = (W - 2a)/eps (contact again for eta > b_+), orthant gates with wrap."""
    l1, l2 = lam
    J1, s1 = Jf(l1)
    J2, s2 = Jf(l2)
    t1, t2 = TJ
    V1, V2 = SIGP0**2 + 4 * D0 * t1, SIGP0**2 + 4 * D0 * t2
    bounds = [0.0] + ([(W - 2 * A) / eps] if far else [])
    bp = (W - 2 * A) / eps if far else math.inf
    m1 = m2 = 0.0
    v1 = v2 = 0.0
    for b in bounds:
        # passage 1 switching at level b
        a1 = g_eta(t1, b) * ell(t1)
        m1 += -a1 * J1
        v1 += (a1 * s1) ** 2
        s12 = math.sqrt(4 * D0 * (t2 - t1))
        p2 = Phi(-b / s12) + (Phi((b - bp) / s12) if far else 0.0)
        m2 += a1 * J1 * p2 * (1 - math.exp(-l2))
        # passage 2 switching at level b
        a2 = g_eta(t2, b) * ell(t2)
        mc, sc = b * V1 / V2, math.sqrt(V1 * (1 - V1 / V2))
        p1 = Phi(-mc / sc) + ((1 - Phi((bp - mc) / sc)) if far else 0.0)
        m2 += -a2 * J2 * (p1 * math.exp(-l1) + (1 - p1))
        v2 += (a1 * s1 * p2) ** 2 + (a2 * s2) ** 2
    r = math.sqrt(eps)
    return np.array([r * m1, r * m2]), np.array([r * math.sqrt(v1), r * math.sqrt(v2)])


def spitzer_range(n: int, sd_step: float) -> float:
    """E[max - min] of a Gaussian random walk with n steps (Spitzer: E max = sum_k E S_k^+ / k)."""
    k = np.arange(1, n + 1)
    return 2.0 * sd_step / math.sqrt(2 * math.pi) * float(np.sum(1.0 / np.sqrt(k)))


def frozen_exact_patterns(eps: float, n: int = 40_000_000, chunk: int = 4_000_000) -> np.ndarray:
    """Exact finite-eps gate law at the target times (P00, P10, P01, P11) by sampling the Gaussian
    relative fluctuation (Y_par(t1), eta(t1), Y_par(t2), eta(t2)); gate |R|_mi < a with
    R_par = eps Y_par, R_perp = a + eps eta (mod W).  Seed SeedSequence([20260923, 91, round(1e4 eps)])."""
    t1, t2 = TJ
    vp = lambda t: U0**2 * math.exp(-2 * GAMMA * t) + (2 * D0 / GAMMA) * (1 - math.exp(-2 * GAMMA * t))
    v1, v2 = vp(t1), vp(t2)
    c12 = math.exp(-GAMMA * (t2 - t1)) * v1
    V1, V2 = SIGP0**2 + 4 * D0 * t1, SIGP0**2 + 4 * D0 * t2
    ss = np.random.SeedSequence([SEED, TAG + 1, int(round(1e4 * eps))])
    counts = np.zeros(4)
    done = 0
    for child in ss.spawn(int(math.ceil(n / chunk))):
        rng = np.random.Generator(np.random.Philox(child))
        m = min(chunk, n - done)
        z = rng.standard_normal((m, 4))
        yp1 = math.sqrt(v1) * z[:, 0]
        yp2 = c12 / math.sqrt(v1) * z[:, 0] + math.sqrt(v2 - c12**2 / v1) * z[:, 1]
        e1 = math.sqrt(V1) * z[:, 2]
        e2 = e1 + math.sqrt(V2 - V1) * z[:, 3]
        g = []
        for yp, e in ((yp1, e1), (yp2, e2)):
            rp = A + eps * e
            dperp = np.abs(rp - W * np.round(rp / W))
            g.append((eps * yp) ** 2 + dperp**2 < A * A)
        code = g[0].astype(int) + 2 * g[1].astype(int)
        counts += np.bincount(code, minlength=4)
        done += m
    return counts / n


def cmd_predict(args):
    Jf, hJ = load_J(OUTDIR / args.jfile)
    S = json.loads(N3.read_text())
    out = {"J_h": hJ, "J_file": args.jfile, "source": str(N3.relative_to(HUB)), "eps": {},
           "frozen_exact_patterns_P00_P10_P01_P11": {}}
    pats = {}
    for key, T in S["tangent"].items():
        eps = float(T["eps"])
        wait_for_cpu()
        pats[eps] = frozen_exact_patterns(eps)
        out["frozen_exact_patterns_P00_P10_P01_P11"][key] = pats[eps].tolist()
        print(key, "exact gate patterns", np.round(pats[eps], 5), "N3 empirical", T["contact_pattern_probs_tj"], flush=True)
    for key, T in S["tangent"].items():
        eps = float(T["eps"])
        rows = []
        for r in T["rows"]:
            if r["variant"] != "full":
                continue
            lam = r["lambda"]
            fk = np.array(r["fk_masses"])
            se = np.array(r["fk_se_batch"])
            orth = np.array(r["orthant_law"])
            femp = np.array(r["frozen_empirical_chi"])
            d1, dse = first_order_m2(lam, Jf, eps)
            d1f, _ = first_order_m2(lam, Jf, eps, far=True)
            row = {"B": r["B"], "lambda": lam, "fk": fk.tolist(), "fk_se": se.tolist(),
                   "orthant": orth.tolist(), "frozen_emp": femp.tolist(),
                   "sqrt_eps_m": d1.tolist(), "sqrt_eps_m_mc_se": dse.tolist(),
                   "sqrt_eps_m_with_wrap": d1f.tolist()}
            for name, base_ in (("orthant", orth), ("frozen_emp", femp)):
                row[f"z_fk_minus_{name}"] = ((fk - base_) / se).tolist()
                pred = base_ + d1
                row[f"corr_{name}"] = pred.tolist()
                row[f"res_corr_{name}"] = (fk - pred).tolist()
                row[f"z_corr_{name}"] = ((fk - pred) / np.sqrt(se**2 + dse**2)).tolist()
            predw = femp + d1f
            row["res_corr_frozen_emp_wrap"] = (fk - predw).tolist()
            row["z_corr_frozen_emp_wrap"] = ((fk - predw) / np.sqrt(se**2 + dse**2)).tolist()
            pe = pats[eps]
            fex = np.array([(pe[1] + pe[3]) * (1 - math.exp(-lam[0])),
                            (pe[3] * math.exp(-lam[0]) + pe[2]) * (1 - math.exp(-lam[1]))])
            row["frozen_exact"] = fex.tolist()
            for nm, dd_ in (("", d1), ("_wrap", d1f)):
                pr = fex + dd_
                row[f"corr_frozen_exact{nm}"] = pr.tolist()
                row[f"res_corr_frozen_exact{nm}"] = (fk - pr).tolist()
                row[f"z_corr_frozen_exact{nm}"] = ((fk - pr) / np.sqrt(se**2 + dse**2)).tolist()
            row["z_fk_minus_frozen_exact"] = ((fk - fex) / se).tolist()
            rows.append(row)
        # switching probabilities
        sw = S["gate_switching_FG1"][f"eps{eps:g}"]
        swrows = {}
        for wname, (lo, hi) in sw["windows"].items():
            j = 0 if wname.endswith("j1") else 1
            tj = TJ[j]
            Sloc = 0.5 * (hi - lo) / eps
            cont = math.sqrt(eps) * g_eta(tj) * 2 * math.sqrt(D0) * 4 * math.sqrt(Sloc / math.pi)
            n = int(round((hi - lo) / 1e-3))
            sd_nu = 2 * math.sqrt(D0) * math.sqrt(1e-3 / eps)
            disc = math.sqrt(eps) * g_eta(tj) * spitzer_range(n, sd_nu)
            bp = (W - 2 * A) / eps
            disc_wrap = disc * (1 + g_eta(tj, bp) / g_eta(tj))
            swrows[wname] = {"window": [lo, hi], "S_local": Sloc, "data": sw["p_gate_switches_in_window"][wname],
                             "data_se": sw["se"][wname], "pred_continuous": cont,
                             "pred_discrete_dt1e-3": disc, "pred_discrete_with_wrap": disc_wrap,
                             "n_steps": n}
        out["eps"][key] = {"eps": eps, "rows": rows, "switching": swrows}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "p5_predict.json").write_text(json.dumps(out, indent=1))
    for key, E in out["eps"].items():
        print(key)
        for r in E["rows"]:
            print("  B=%6g exact: z(fk-fex)=%s corr z=%s corr+wrap res=%s z=%s" % (r["B"], np.round(r["z_fk_minus_frozen_exact"], 1),
                  np.round(r["z_corr_frozen_exact"], 1), np.round(r["res_corr_frozen_exact_wrap"], 4), np.round(r["z_corr_frozen_exact_wrap"], 1)))
            print("  B=%6g fk-orth=%s z=%s | corr_orth res=%s z=%s | corr_emp res=%s z=%s | wrap z=%s" % (
                r["B"], np.round(np.array(r["fk"]) - np.array(r["orthant"]), 4), np.round(r["z_fk_minus_orthant"], 1),
                np.round(r["res_corr_orthant"], 4), np.round(r["z_corr_orthant"], 1),
                np.round(r["res_corr_frozen_emp"], 4), np.round(r["z_corr_frozen_emp"], 1),
                np.round(r["z_corr_frozen_emp_wrap"], 1)))
        for w, v in E["switching"].items():
            print("  %-22s data=%.4f pred_cont=%.4f pred_disc=%.4f wrap=%.4f" % (
                w, v["data"], v["pred_continuous"], v["pred_discrete_dt1e-3"], v["pred_discrete_with_wrap"]))


# ---------------------------------------------------------------------------
# per-path oracle decomposition on cached tangent FK ensembles
# ---------------------------------------------------------------------------

def cmd_oracle(args):
    Jf, hJ = load_J(OUTDIR / args.jfile)
    out = {"J_h": hJ, "ensembles": {}}
    for eps_s in ("0.025", "0.05"):
        f = CACHE / f"fkh3_m2_e{eps_s}.npz"
        d = np.load(f)
        eps = float(d["eps"])
        Lf = d["lamck_full"].astype(np.float64)
        Lq = d["lamck_free"].astype(np.float64)
        chi = d["chi_tj"].astype(np.float64)
        budgets = d["budgets"]
        N = Lf.shape[0]
        rows = []
        for B in budgets:
            lam = lambdas(float(B))
            e0 = np.exp(-B * Lf[:, 0])
            M1 = e0 * (-np.expm1(-B * (Lf[:, 1] - Lf[:, 0])))
            M2 = np.exp(-B * Lf[:, 1]) * (-np.expm1(-B * (Lf[:, 2] - Lf[:, 1])))
            q1 = chi[:, 0] * (Lq[:, 1] - Lq[:, 0])
            q2 = chi[:, 1] * (Lq[:, 2] - Lq[:, 1])
            O1 = -np.expm1(-B * q1)
            O2 = np.exp(-B * q1) * (-np.expm1(-B * q2))
            E1 = chi[:, 0] * (1 - math.exp(-lam[0]))
            E2 = np.exp(-lam[0] * chi[:, 0]) * chi[:, 1] * (1 - math.exp(-lam[1]))
            sw = np.stack([M1 - O1, M2 - O2], 1)
            mid = np.stack([O1 - E1, O2 - E2], 1)
            d1, dse = first_order_m2(lam, Jf, eps)
            rows.append({"B": float(B), "fk": [float(M1.mean()), float(M2.mean())],
                         "fk_se": [float(M1.std() / math.sqrt(N)), float(M2.std() / math.sqrt(N))],
                         "switch_part": sw.mean(0).tolist(), "switch_part_se": (sw.std(0) / math.sqrt(N)).tolist(),
                         "midpoint_part": mid.mean(0).tolist(), "midpoint_part_se": (mid.std(0) / math.sqrt(N)).tolist(),
                         "frozen_emp_same_paths": [float(E1.mean()), float(E2.mean())],
                         "sqrt_eps_m": d1.tolist(), "sqrt_eps_m_mc_se": dse.tolist(),
                         "switch_minus_pred": (sw.mean(0) - d1).tolist()})
        out["ensembles"][eps_s] = {"file": str(f), "seed": int(d["seed"]), "N": int(N), "dt": float(d["dt"]),
                                   "rows": rows}
        print("eps", eps_s)
        for r in rows:
            print("  B=%6g switch=%s +- %s pred=%s | mid=%s +- %s" % (
                r["B"], np.round(r["switch_part"], 4), np.round(r["switch_part_se"], 4), np.round(r["sqrt_eps_m"], 4),
                np.round(r["midpoint_part"], 4), np.round(r["midpoint_part_se"], 4)))
    (OUTDIR / "p5_oracle.json").write_text(json.dumps(out, indent=1))



# ---------------------------------------------------------------------------
# Part A check: FK density near the passages vs the frozen-gate limit profile
# ---------------------------------------------------------------------------

_UG = np.linspace(-12.0, 12.0, 24001)
_PH = np.exp(-0.5 * _UG * _UG) / math.sqrt(2 * math.pi)
_CPH = 0.5 * np.array([math.erfc(-x / math.sqrt(2)) for x in _UG])


def F_prof(lam: float, y: np.ndarray) -> np.ndarray:
    """F_lam(y) = int lam phi(u) exp(-lam Phi(u)) phi_theta(y-u) du, theta^2 = D0/(2 gamma rho^2)."""
    th = math.sqrt(SZ2) / RHO
    u = _UG
    du = u[1] - u[0]
    p = lam * _PH * np.exp(-lam * _CPH)
    y = np.atleast_1d(y)
    out = np.empty(y.size)
    for k, yy in enumerate(y):
        ker = np.exp(-0.5 * ((yy - u) / th) ** 2) / (math.sqrt(2 * math.pi) * th)
        out[k] = float(np.sum(p * ker) * du)
    return out


def y_star(lam: float) -> float:
    ys = np.linspace(-4, 1, 501)
    Fv = F_prof(lam, ys)
    k = int(np.argmax(Fv))
    lo, hi = ys[max(k - 1, 0)], ys[min(k + 1, ys.size - 1)]
    for _ in range(40):  # golden-section refinement
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if F_prof(lam, np.array([m1]))[0] > F_prof(lam, np.array([m2]))[0]:
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def smooth(x: np.ndarray, bw_steps: float) -> np.ndarray:
    r = int(math.ceil(4 * bw_steps))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / bw_steps) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def cmd_profile(args):
    out = {"ensembles": {}}
    for eps_s in ("0.025", "0.05"):
        d = np.load(CACHE / f"fkh3_m2_e{eps_s}.npz")
        eps, dt = float(d["eps"]), float(d["dt"])
        t = d["tgrid"]
        chi = d["chi_tj"].astype(int)
        code = chi[:, 0] + 2 * chi[:, 1]
        pat = np.bincount(code, minlength=4) / code.size   # P00, P10, P01, P11
        V1, V2 = SIGP0**2 + 4 * D0 * TJ[0], SIGP0**2 + 4 * D0 * TJ[1]
        r12 = math.sqrt(V1 / V2)
        P11o = 0.25 + math.asin(r12) / (2 * math.pi)
        P01o = 0.25 - math.asin(r12) / (2 * math.pi)
        rows = []
        for bi, B in enumerate(d["budgets"]):
            lam = lambdas(float(B))
            f = d["FK_full"][bi] / dt
            fb = d["FKb_full"][:, bi, :] / dt
            for j in (0, 1):
                v = vel(TJ[j])
                sig_t = eps * math.sqrt(SZ2 + RHO**2) / v
                alpha_orth = 0.5 if j == 0 else P11o * math.exp(-lam[0]) + P01o
                alpha_emp = (pat[1] + pat[3]) if j == 0 else pat[3] * math.exp(-lam[0]) + pat[2]
                win = np.abs(t - TJ[j]) <= 3 * sig_t
                tw = t[win]
                y = v * (tw - TJ[j]) / (eps * RHO)
                Fl = F_prof(lam[j], y)
                flim = alpha_orth / eps * (v / RHO) * Fl
                flim_emp = alpha_emp / eps * (v / RHO) * Fl
                fs = smooth(f, 0.15 * sig_t / dt)[win]
                ys_ = y_star(lam[j])
                t_pred = TJ[j] + eps * RHO * ys_ / v
                kmax = int(np.argmax(fs))
                # local quadratic fit around the smoothed max
                sl = slice(max(kmax - 5, 0), min(kmax + 6, fs.size))
                cf = np.polyfit(tw[sl] - tw[kmax], fs[sl], 2)
                t_peak = tw[kmax] - cf[1] / (2 * cf[0])
                # count local maxima of the smoothed density in the passage window t_pred +- 4 sigma_t
                wl = np.abs(t - t_pred) <= 4 * sig_t
                fsl = smooth(f, 0.15 * sig_t / dt)[wl]
                dfs = np.diff(fsl)
                nmax = int(np.sum((dfs[:-1] > 0) & (dfs[1:] <= 0)))
                # second derivative at peak (fit) vs limit
                f2_fk = 2 * cf[0]
                ypk = np.array([ys_ - 1e-3, ys_, ys_ + 1e-3])
                Fp = F_prof(lam[j], ypk)
                f2_lim = alpha_orth / eps * (v / RHO) * (Fp[0] - 2 * Fp[1] + Fp[2]) / 1e-6 * (v / (eps * RHO)) ** 2
                rows.append({"B": float(B), "j": j + 1, "lambda": lam[j], "alpha_orthant": alpha_orth,
                             "alpha_empirical": alpha_emp,
                             "rel_sup_err_orthant": float(np.max(np.abs(fs - flim)) / np.max(flim)),
                             "rel_sup_err_empirical_alpha": float(np.max(np.abs(fs - flim_emp)) / np.max(flim_emp)),
                             "t_peak_fk": float(t_peak), "t_peak_limit": float(t_pred),
                             "peak_shift_over_sigma_t": float((t_peak - t_pred) / sig_t),
                             "height_ratio_fk_over_limit": float(np.max(fs) / np.max(flim)),
                             "f2_ratio_fk_over_limit": float(f2_fk / f2_lim),
                             "n_local_max_smoothed_in_4sigma_window": nmax})
        out["ensembles"][eps_s] = {"seed": int(d["seed"]), "N": int(d["N"]), "dt": dt,
                                   "contact_patterns_P00_P10_P01_P11": pat.tolist(), "rows": rows}
        print("eps", eps_s)
        for r in rows:
            print("  B=%6g j=%d lam=%.3f supErr=%.3f/%.3f dpeak/sig=%+.3f hratio=%.3f f2ratio=%.3f nmax=%d" % (
                r["B"], r["j"], r["lambda"], r["rel_sup_err_orthant"], r["rel_sup_err_empirical_alpha"],
                r["peak_shift_over_sigma_t"], r["height_ratio_fk_over_limit"], r["f2_ratio_fk_over_limit"],
                r["n_local_max_smoothed_in_4sigma_window"]))
    # N3 census maxima (declared-mode exact law, 2e5 paths) vs predicted peak times t_j + eps rho y*(lam_j)/v_j
    S = json.loads(N3.read_text())
    cen = {}
    ycache = {}
    for key, T in S["tangent"].items():
        eps = float(T["eps"])
        rr = []
        for r in T["rows"]:
            if r["variant"] != "full":
                continue
            c = r["derivative_census_bw0.15sig"]
            pred = []
            for j in (0, 1):
                lam = r["lambda"][j]
                if lam not in ycache:
                    ycache[lam] = y_star(lam)
                pred.append(TJ[j] + eps * RHO * ycache[lam] / vel(TJ[j]))
            mt = c["maxima_t"]
            rr.append({"B": r["B"], "n_maxima": c["n_maxima"], "n_minima": c["n_minima"],
                       "maxima_t": mt, "pred_t": pred,
                       "shift_over_eps": [(mt[k] - pred[k]) / eps for k in range(min(2, len(mt)))]})
        cen[key] = rr
    out["n3_census_vs_prediction"] = cen
    out["y_star"] = {str(k): v for k, v in ycache.items()}
    for key, rr in cen.items():
        print(key, [(r["B"], r["n_maxima"], [round(x, 2) for x in r["shift_over_eps"]]) for r in rr])
    (OUTDIR / "p5_profile.json").write_text(json.dumps(out, indent=1))



# ---------------------------------------------------------------------------
# summary JSON + figure
# ---------------------------------------------------------------------------

def cmd_all(args):
    J = json.loads((OUTDIR / "p5_jfun.json").read_text())
    P = json.loads((OUTDIR / "p5_predict.json").read_text())
    O = json.loads((OUTDIR / "p5_oracle.json").read_text())
    F = json.loads((OUTDIR / "p5_profile.json").read_text())
    fine = min(J["runs"], key=lambda r: r["h"])
    lam = np.array(J["lams"])
    head = {"J_finest_h": fine["h"],
            "J_at": {f"{l:g}": [fine["J"][i], fine["se"][i]] for i, l in enumerate(lam)},
            "kappa2": KAPPA2, "L1_mean_vs_cJ": [fine["L1_mean"], fine["L1_se"], 0.6560038973],
            "J_h_spread_max_lam_le_13": float(max(max(r["J"][i] for r in J["runs"]) - min(r["J"][i] for r in J["runs"])
                                                for i, l in enumerate(lam) if l <= 13))}
    zs = {}
    for e, E in P["eps"].items():
        zs[e] = {}
        for nm in ("z_fk_minus_orthant", "z_fk_minus_frozen_exact", "z_corr_orthant", "z_corr_frozen_exact",
                   "z_corr_frozen_exact_wrap"):
            Z = np.array([r[nm] for r in E["rows"]])
            Bs = np.array([r["B"] for r in E["rows"]])
            zs[e][nm] = {"max_abs_all_B": float(np.abs(Z).max()), "max_abs_B_le_8": float(np.abs(Z[Bs <= 8]).max())}
        zs[e]["switching"] = {w: {"data": v["data"], "pred": v["pred_discrete_with_wrap"],
                                  "rel_err": v["pred_discrete_with_wrap"] / v["data"] - 1}
                              for w, v in E["switching"].items()}
    b8 = [r for r in P["eps"]["eps0.025"]["rows"] if r["B"] == 8][0]
    head["eps0.025_B8"] = {k: b8[k] for k in ("fk", "fk_se", "orthant", "frozen_exact", "sqrt_eps_m",
                                              "sqrt_eps_m_with_wrap", "corr_frozen_exact", "corr_frozen_exact_wrap",
                                              "res_corr_frozen_exact", "res_corr_frozen_exact_wrap",
                                              "z_corr_frozen_exact", "z_corr_frozen_exact_wrap")}
    ob8 = [r for r in O["ensembles"]["0.025"]["rows"] if r["B"] == 8][0]
    head["oracle_eps0.025_B8"] = ob8
    summ = {"item": "P5", "driver": "code/fb_p5_gate_switching.py", "seed": [SEED, TAG, TAG + 1],
            "headline": head, "zscores": zs,
            "files": ["p5_jfun.json", "p5_predict.json", "p5_oracle.json", "p5_profile.json"]}
    (OUTDIR / "p5_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(head["eps0.025_B8"], indent=0)[:800])
    print(json.dumps(zs, indent=0)[:2500])
    # ------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    INK, MUTED = "#222222", "#8a8a85"
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                         "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))
    # (a) J(lambda)
    a = ax[0, 0]
    Jv, Js = np.array(fine["J"]), np.array(fine["se"])
    a.fill_between(lam, Jv - 2 * Js, Jv + 2 * Js, color=C1, alpha=0.25, lw=0)
    a.plot(lam, Jv, color=C1, lw=1.6, marker="o", ms=3, label=r"Monte Carlo, $h=10^{-3}$")
    ll = lam[lam <= 3]
    a.plot(ll, -KAPPA2 * ll**2, color=MUTED, lw=1.2, ls="--", label=r"$-\kappa_2\lambda^2$")
    a.set_xscale("log")
    a.set_xlabel(r"passage exposure $\lambda$")
    a.set_ylabel(r"$\mathcal{J}(\lambda)$")
    a.set_title("(a) switching function", loc="left", fontsize=8)
    a.legend(frameon=False, fontsize=7)
    # (b) switching probability vs sqrt(eps)
    b = ax[0, 1]
    for wname, col, lab in (("passage_2sigma_t_j1", C1, r"$t_1\pm2\sigma$"), ("passage_2sigma_t_j2", C2, r"$t_2\pm2\sigma$")):
        xs, yd, ye, yp = [], [], [], []
        for e, E in P["eps"].items():
            v = E["switching"][wname]
            xs.append(math.sqrt(E["eps"]))
            yd.append(v["data"]); ye.append(v["data_se"]); yp.append(v["pred_discrete_with_wrap"])
        o = np.argsort(xs)
        xs, yd, ye, yp = (np.array(z)[o] for z in (xs, yd, ye, yp))
        b.errorbar(xs, yd, yerr=2 * ye, fmt="o", color=col, ms=5, mfc="white", mew=1.2, lw=0.8, label=lab + " exact law")
        # R3 fix (B4): the prediction is pred_discrete_with_wrap (grid monitoring + both contact boundaries)
        b.plot(xs, yp, color=col, lw=1.4, label=lab + r" prediction (grid, both boundaries)")
    b.set_xlabel(r"$\sqrt{\varepsilon}$")
    b.set_ylabel("P(gate switches in window)")
    b.set_xlim(0, 0.25); b.set_ylim(0, 0.55)
    b.set_title(r"(b) the $\sqrt{\varepsilon}$ switching law", loc="left", fontsize=8)
    b.legend(frameon=False, fontsize=6.5)
    # (c) late basin vs B at eps=0.025
    c = ax[1, 0]
    R = P["eps"]["eps0.025"]["rows"]
    Bs = np.array([r["B"] for r in R])
    g = lambda k, j=1: np.array([r[k][j] for r in R])
    c.plot(Bs, g("orthant"), color=MUTED, lw=1.2, ls=":", label=r"limit law ($\varepsilon\to0$)")
    c.plot(Bs, g("frozen_exact"), color=C2, lw=1.4, ls="--", label=r"frozen gate, exact gates at $t_j$")
    # R3 fix (B4): the solid curve is corr_frozen_exact_wrap (switching correction evaluated with the wrap-around boundary)
    c.plot(Bs, g("corr_frozen_exact_wrap"), color=C1, lw=1.6,
           label="+ switching correction " + r"$\sqrt{\varepsilon}\,\mathfrak{m}_2$" + "\n(incl. wrap-around)")
    c.errorbar(Bs, g("fk"), yerr=2 * g("fk_se"), fmt="o", color=INK, ms=3.5, mfc="white", mew=1.0, lw=0.8,
               label=r"exact law ($\pm2$ SE)")
    c.set_xscale("log")
    c.set_xlabel("budget $B$")
    c.set_ylabel(r"late-basin mass, $\varepsilon=0.025$")
    c.set_title("(c) corrected mass law", loc="left", fontsize=8)
    c.legend(frameon=False, fontsize=6.5, loc="upper right")  # R3 fix (B4): fixed location (the longer label moved "best" onto the curves)
    # (d) z of residuals
    d_ = ax[1, 1]
    d_.axhspan(-2, 2, color="#e9e9e4", lw=0)
    for e, col in (("eps0.0125", C1), ("eps0.025", C2)):
        R = P["eps"][e]["rows"]
        Bs = np.array([r["B"] for r in R])
        for j, mk in ((0, "s"), (1, "o")):
            z = np.array([r["z_corr_frozen_exact_wrap"][j] for r in R])
            d_.plot(Bs, z, color=col, marker=mk, ms=4, lw=1.0,
                    label=rf"$\varepsilon={P['eps'][e]['eps']:g}$, " + ("early" if j == 0 else "late") + " basin")
    d_.set_xscale("log")
    d_.set_ylim(-5, 5)
    d_.set_xlabel("budget $B$")
    d_.set_ylabel("(exact - corrected law)/SE")
    d_.set_title("(d) residuals; limit law: up to 43.1 SE", loc="left", fontsize=8)  # R3 RND: 43.047 rounded outward
    d_.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    for axx in ax.ravel():
        axx.grid(True, color="#ececea", lw=0.5)
        axx.set_axisbelow(True)
    fig.tight_layout()
    FIG = HUB / "artifacts" / "figures" / "fb_p5_gate_switching.pdf"
    fig.savefig(FIG)
    fig.savefig(OUTDIR / "fb_p5_gate_switching_preview.png", dpi=160)
    print("wrote", FIG)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jfun")
    j.add_argument("--paths", type=int, default=40000)
    j.add_argument("--chunk", type=int, default=500)
    j.add_argument("--X", type=float, default=6.0)
    j.add_argument("--hs", type=float, nargs="+", default=[4e-3, 1e-3])
    j.add_argument("--lams", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.718, 4.0, 8.0, 12.18, 16.0, 30.0, 68.0, 100.0, 300.0])
    j.add_argument("--out", default="p5_jfun.json")
    sub.add_parser("profile")
    sub.add_parser("all")
    for name in ("predict", "oracle"):
        q = sub.add_parser(name)
        q.add_argument("--jfile", default="p5_jfun.json")
    args = ap.parse_args()
    if args.cmd == "jfun":
        cmd_jfun(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "oracle":
        cmd_oracle(args)
    elif args.cmd == "profile":
        cmd_profile(args)
    elif args.cmd == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
