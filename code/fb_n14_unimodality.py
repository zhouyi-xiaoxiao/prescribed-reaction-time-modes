#!/usr/bin/env python3
"""N14a -- universality campaign: single-peak property of the passage profile for
general stripe profiles psi (tests Lemma th13:unimodal and Remark th13:counterex of
manuscript/cnsns_submission/theory/TH13_universality.tex).

For a stripe profile psi (probability density on R, cdf Psi) and a passage exposure
beta > 0 the frozen-offset reaction profile is

    p_beta(u) = beta psi(u) exp(-beta Psi(u)),     F_{beta,theta} = p_beta * phi_theta,

(phi_theta = centred Gaussian density of standard deviation theta; theta = 0 means F = p).
Lemma th13:unimodal: if psi is log-concave then g = (log psi)' - beta psi changes sign
exactly once (+ to -), so p_beta' has one sign change and F has exactly one critical
point, a nondegenerate maximum, for every theta > 0.

What is computed (deterministic quadrature only, no random numbers):
  * for each family: number of sign changes of g on the support (analytic (log psi)'),
    boundary atoms of the distributional derivative included;
  * number of strict local maxima of F on a fine grid (FFT convolution), counted where
    F > 1e-7 max F; the second difference at each maximum;
  * cross-check of the generalized curvature identity (A8) at the maximum for smooth psi;
  * for the non-log-concave families: the smallest beta on the grid at which p (theta=0)
    and F (theta>0) have two or more maxima; detailed maxima/dip ratios for chosen cases
    (converged under grid refinement h -> h/2).  The symmetric smooth unimodal 'core+halo'
    profile (1-q) phi_{s1} + q phi_{s2} gives a robust two-peak passage profile.

Output: artifacts/data/exact_m_fixed_budget/N14_universality/n14_unimodality.json
Run:    python3 code/fb_n14_unimodality.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "data", "exact_m_fixed_budget", "N14_universality")
OUT = os.path.join(OUT_DIR, "n14_unimodality.json")

SQ2PI = np.sqrt(2.0 * np.pi)


def phi(u):
    return np.exp(-0.5 * u * u) / SQ2PI


def Phi(u):
    return 0.5 * special.erfc(-u / np.sqrt(2.0))


# ---------------------------------------------------------------- families
# each family: dict(pdf, cdf, dlog, support=(a,b), lc=bool, U=grid half width, h=grid step)

def fam_gauss():
    return dict(pdf=phi, cdf=Phi, dlog=lambda u: -u, support=(-np.inf, np.inf), lc=True, U=30.0, h=0.002)


def fam_logistic():
    s = np.sqrt(3.0) / np.pi  # unit variance
    pdf = lambda u: 1.0 / (4.0 * s) / np.cosh(u / (2.0 * s)) ** 2
    cdf = lambda u: 0.5 * (1.0 + np.tanh(u / (2.0 * s)))
    dlog = lambda u: -np.tanh(u / (2.0 * s)) / s
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=True, U=40.0, h=0.002)


def fam_tophat():
    c = np.sqrt(3.0)
    pdf = lambda u: np.where(np.abs(u) < c, 1.0 / (2 * c), 0.0)
    cdf = lambda u: np.clip((u + c) / (2 * c), 0.0, 1.0)
    dlog = lambda u: np.zeros_like(u)
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-c, c), lc=True, U=30.0, h=0.002)


def fam_raised_cosine():
    c = 2.0
    pdf = lambda u: np.where(np.abs(u) < c, (1.0 + np.cos(np.pi * u / c)) / (2 * c), 0.0)
    cdf = lambda u: np.where(u <= -c, 0.0, np.where(u >= c, 1.0,
                             (u + c) / (2 * c) + np.sin(np.pi * u / c) / (2 * np.pi)))
    dlog = lambda u: -(np.pi / c) * np.tan(np.pi * u / (2 * c))
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-c, c), lc=True, U=30.0, h=0.002)


def fam_smooth_compact():
    c = 3.0
    # psi = K (1-u^2/c^2)^5 on (-c,c); K = 1/(c * int_{-1}^1 (1-x^2)^5 dx) ; int = 512/693
    K = 693.0 / (512.0 * c)

    def pdf(u):
        x = np.clip(1.0 - (u / c) ** 2, 0.0, None)
        return K * x ** 5

    def cdf(u):
        # antiderivative of (1-x^2)^5 from -1 to x (x=u/c), polynomial
        x = np.clip(u / c, -1.0, 1.0)
        P = x - 5 * x ** 3 / 3 + 2 * x ** 5 - 10 * x ** 7 / 7 + 5 * x ** 9 / 9 - x ** 11 / 11
        P0 = -1 + 5 / 3 - 2 + 10 / 7 - 5 / 9 + 1 / 11
        return K * c * (P - P0)

    dlog = lambda u: 5.0 * (-2.0 * u / c ** 2) / (1.0 - (u / c) ** 2)
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-c, c), lc=True, U=30.0, h=0.002)


def fam_laplace():
    b = 1.0 / np.sqrt(2.0)
    pdf = lambda u: np.exp(-np.abs(u) / b) / (2 * b)
    cdf = lambda u: np.where(u < 0, 0.5 * np.exp(u / b), 1.0 - 0.5 * np.exp(-u / b))
    dlog = lambda u: -np.sign(u) / b
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=True, U=40.0, h=0.002)


def fam_gengauss(alpha, U=60.0, h=0.002):
    K = alpha / (2.0 * special.gamma(1.0 / alpha))
    pdf = lambda u: K * np.exp(-np.abs(u) ** alpha)

    def cdf(u):
        q = 0.5 * special.gammaincc(1.0 / alpha, np.abs(u) ** alpha)
        return np.where(u < 0, q, 1.0 - q)

    def dlog(u):
        au = np.abs(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = -alpha * np.sign(u) * np.where(au > 0, au ** (alpha - 1.0), 0.0)
        return d

    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=(alpha >= 1.0), U=U, h=h)


def fam_cauchy():
    pdf = lambda u: 1.0 / (np.pi * (1.0 + u * u))
    cdf = lambda u: 0.5 + np.arctan(u) / np.pi
    dlog = lambda u: -2.0 * u / (1.0 + u * u)
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=False, U=400.0, h=0.004)


def fam_student(nu):
    from scipy import stats
    pdf = lambda u: stats.t.pdf(u, nu)
    cdf = lambda u: stats.t.cdf(u, nu)
    dlog = lambda u: -(nu + 1.0) * u / (nu + u * u)
    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=False, U=300.0, h=0.004)


def fam_mixture(q, s, d):
    """(1-q) phi(u) + q phi_s(u+d): a leading 'shoulder' (d>0 puts it on the side crossed first)."""
    pdf = lambda u: (1 - q) * phi(u) + q * phi((u + d) / s) / s
    cdf = lambda u: (1 - q) * Phi(u) + q * Phi((u + d) / s)

    def dlog(u):
        num = (1 - q) * (-u) * phi(u) + q * (-(u + d) / s ** 2) * phi((u + d) / s) / s
        return num / pdf(u)

    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=False, U=30.0, h=0.002)


def fam_corehalo(q, s1, s2):
    """Symmetric 'core + halo' stripe: (1-q) phi_{s1} + q phi_{s2}, s1 < s2 (unimodal, smooth, not log-concave)."""
    pdf = lambda u: (1 - q) * phi(u / s1) / s1 + q * phi(u / s2) / s2
    cdf = lambda u: (1 - q) * Phi(u / s1) + q * Phi(u / s2)

    def dlog(u):
        num = -(1 - q) * (u / s1 ** 2) * phi(u / s1) / s1 - q * (u / s2 ** 2) * phi(u / s2) / s2
        return num / pdf(u)

    return dict(pdf=pdf, cdf=cdf, dlog=dlog, support=(-np.inf, np.inf), lc=False, U=40.0, h=0.002)


FAMILIES = {
    "gauss": fam_gauss,
    "logistic": fam_logistic,
    "tophat": fam_tophat,
    "raised_cosine": fam_raised_cosine,
    "smooth_compact_p5": fam_smooth_compact,
    "laplace": fam_laplace,
    "gengauss_a1.5": lambda: fam_gengauss(1.5),
    "gengauss_a0.75": lambda: fam_gengauss(0.75, U=200.0, h=0.004),
    "gengauss_a0.5": lambda: fam_gengauss(0.5, U=400.0, h=0.004),
    "cauchy": fam_cauchy,
    "student_t3": lambda: fam_student(3.0),
    "shoulder_q0.15_s0.5_d1.5": lambda: fam_mixture(0.15, 0.5, 1.5),
    "shoulder_q0.1_s0.3_d1.2": lambda: fam_mixture(0.1, 0.3, 1.2),
    "bump_q0.3_s0.4_d2": lambda: fam_mixture(0.3, 0.4, 2.0),
    "two_stripes_q0.5_s0.3_d3": lambda: fam_mixture(0.5, 0.3, 3.0),
    "corehalo_q0.5_s0.3_S3": lambda: fam_corehalo(0.5, 0.3, 3.0),
    "corehalo_q0.7_s0.3_S3": lambda: fam_corehalo(0.7, 0.3, 3.0),
}

BETAS = np.logspace(-1.5, 2.5, 81)
THETAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def sign_changes(v, tol=0.0):
    s = np.sign(np.where(np.abs(v) <= tol, 0.0, v))
    s = s[s != 0]
    if s.size == 0:
        return 0, []
    idx = np.nonzero(np.diff(s))[0]
    return int(idx.size), s


def count_maxima(F, rel=1e-7):
    Fm = F.max()
    i = np.arange(1, F.size - 1)
    mask = (F[i] > F[i - 1]) & (F[i] >= F[i + 1]) & (F[i] > rel * Fm)
    return i[mask]


def g_sign_pattern(fam, u, beta):
    a, b = fam["support"]
    inside = (u > a) & (u < b)
    ui = u[inside]
    psi = fam["pdf"](ui)
    ok = psi > 1e-300
    g = fam["dlog"](ui[ok]) - beta * psi[ok]
    seq = []
    if np.isfinite(a):
        seq.append(1.0)  # atom +p(a+) at the left end
    seq.extend(list(np.sign(g[np.abs(g) > 1e-12])))
    if np.isfinite(b):
        seq.append(-1.0)  # atom -p(b-) at the right end
    seq = np.array(seq)
    seq = seq[seq != 0]
    nch = int(np.count_nonzero(np.diff(seq))) if seq.size > 1 else 0
    return nch, (int(seq[0]) if seq.size else 0), (int(seq[-1]) if seq.size else 0)


def gauss_kernel(theta, h, n):
    x = (np.arange(n) - n // 2) * h
    k = np.exp(-0.5 * (x / theta) ** 2)
    return k / (k.sum() * h), x


def fft_convolve(p, theta, h):
    n = p.size
    m = 1 << int(np.ceil(np.log2(2 * n)))
    half = int(np.ceil(10 * theta / h)) + 1
    xk = np.arange(-half, half + 1) * h
    k = np.exp(-0.5 * (xk / theta) ** 2)
    k /= k.sum() * h
    P = np.fft.rfft(p, m)
    Kf = np.fft.rfft(k, m)
    full = np.fft.irfft(P * Kf, m)[: n + k.size - 1] * h
    return full[half: half + n]


def analyse_family(name):
    fam = FAMILIES[name]()
    U, h = fam["U"], fam["h"]
    u = np.arange(-U, U + h / 2, h)
    psi = fam["pdf"](u)
    Psi = fam["cdf"](u)
    # unimodality of psi itself (on the grid)
    psi_max = count_maxima(psi, rel=1e-9)
    rec = {"log_concave": bool(fam["lc"]), "grid": {"U": U, "h": h},
           "psi_num_local_maxima": int(psi_max.size), "cells": []}
    min_beta_bimodal_p = None
    min_beta_bimodal_F = {str(t): None for t in THETAS if t > 0}
    max_signchanges = 0
    max_Fmax = 0
    worst_rel_curv = np.inf
    a8_checks = []
    for beta in BETAS:
        nch, first, last = g_sign_pattern(fam, u, beta)
        max_signchanges = max(max_signchanges, nch)
        p = beta * psi * np.exp(-beta * Psi)
        pm = count_maxima(np.concatenate([[0.0], p, [0.0]]))  # pad: edge jumps count
        if pm.size >= 2 and min_beta_bimodal_p is None:
            min_beta_bimodal_p = float(beta)
        cell = {"beta": float(beta), "g_sign_changes": nch, "g_first_sign": first, "g_last_sign": last,
                "p_num_maxima": int(pm.size), "F": {}}
        for th in THETAS:
            if th == 0:
                continue
            F = fft_convolve(p, th, h)
            im = count_maxima(F)
            nmax = int(im.size)
            max_Fmax = max(max_Fmax, nmax)
            if nmax >= 2 and min_beta_bimodal_F[str(th)] is None:
                min_beta_bimodal_F[str(th)] = float(beta)
            curv = []
            for i in im:
                d2 = (F[i + 1] - 2 * F[i] + F[i - 1]) / h ** 2
                curv.append(d2 / F[i])
            if nmax == 1:
                worst_rel_curv = min(worst_rel_curv, -curv[0])
            cell["F"][str(th)] = {"num_maxima": nmax, "argmax": [float(u[i]) for i in im][:4],
                                  "rel_curvature_at_max": [float(c) for c in curv][:4]}
            # generalized A8 identity check (smooth full-support psi only, one maximum)
            if (fam["lc"] and np.isinf(fam["support"][0]) and nmax == 1 and name in ("gauss", "logistic")
                    and th in (0.2, 1.0) and abs(np.log10(beta) - round(np.log10(beta))) < 1e-9):
                i = im[0]
                ystar = u[i]
                g = fam["dlog"](u) - beta * psi
                dp = p * g
                # u0 = zero of g (sign change + to -)
                j = np.nonzero((g[:-1] > 0) & (g[1:] <= 0))[0][0]
                u0 = u[j] - g[j] * h / (g[j + 1] - g[j])
                # refine the maximum by a parabola through the three grid points
                y0 = ystar + 0.5 * h * (F[i - 1] - F[i + 1]) / (F[i - 1] - 2 * F[i] + F[i + 1])
                kern = np.exp(-0.5 * ((y0 - u) / th) ** 2) / (SQ2PI * th)
                rhs = np.sum((u - u0) * dp * kern) * h / th ** 2
                d2 = (F[i + 1] - 2 * F[i] + F[i - 1]) / h ** 2
                a8_checks.append({"beta": float(beta), "theta": th, "y_star": float(y0), "u0": float(u0),
                                  "F2_fd": float(d2), "F2_identity": float(rhs),
                                  "rel_diff": float(abs(d2 - rhs) / abs(rhs))})
        rec["cells"].append(cell)
    rec["max_g_sign_changes"] = max_signchanges
    rec["max_num_maxima_F_theta_pos"] = max_Fmax
    rec["min_beta_p_bimodal_theta0"] = min_beta_bimodal_p
    rec["min_beta_F_bimodal_by_theta"] = min_beta_bimodal_F
    rec["min_rel_curvature_single_max"] = (None if not np.isfinite(worst_rel_curv) else float(worst_rel_curv))
    rec["a8_identity_checks"] = a8_checks
    return rec


DETAIL_CASES = [
    ("gengauss_a0.5", 8.0, 0.1), ("gengauss_a0.5", 8.0, 0.2), ("gengauss_a0.5", 20.0, 0.1),
    ("gengauss_a0.75", 8.0, 0.0), ("gengauss_a0.75", 8.0, 0.05),
    ("shoulder_q0.15_s0.5_d1.5", 2.0, 0.0), ("shoulder_q0.15_s0.5_d1.5", 2.0, 0.05),
    ("shoulder_q0.15_s0.5_d1.5", 5.0, 0.05), ("cauchy", 50.0, 0.0), ("gauss", 50.0, 0.0),
    ("gengauss_a0.5", 5.623413251903491, 0.1), ("shoulder_q0.15_s0.5_d1.5", 0.7943282347242817, 0.05),
    ("corehalo_q0.5_s0.3_S3", 8.0, 0.0), ("corehalo_q0.5_s0.3_S3", 8.0, 0.1),
    ("corehalo_q0.5_s0.3_S3", 8.0, 0.3), ("corehalo_q0.5_s0.3_S3", 8.0, 0.5),
    ("corehalo_q0.7_s0.3_S3", 4.0, 0.5), ("corehalo_q0.5_s0.3_S3", 2.0, 0.1),
]


def detail(name, beta, th, refine=1):
    fam = FAMILIES[name]()
    U, h = fam["U"], fam["h"] / refine
    u = np.arange(-U, U + h / 2, h)
    p = beta * fam["pdf"](u) * np.exp(-beta * fam["cdf"](u))
    F = p if th == 0 else fft_convolve(p, th, h)
    Fp = np.concatenate([[0.0], F, [0.0]])
    im = count_maxima(Fp) - 1
    rec = {"family": name, "beta": beta, "theta": th, "h": h, "num_maxima": int(im.size),
           "maxima": [{"u": float(u[i]), "F": float(F[i])} for i in im[:4]]}
    if im.size >= 2:
        i1, i2 = im[0], im[1]
        jmin = i1 + int(np.argmin(F[i1:i2 + 1]))
        rec["min_between"] = {"u": float(u[jmin]), "F": float(F[jmin])}
        rec["dip_ratio"] = float(F[jmin] / min(F[i1], F[i2]))
    return rec


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"script": "code/fb_n14_unimodality.py", "date": time.strftime("%Y-%m-%d"),
           "random_numbers": "none (deterministic quadrature)",
           "betas": [float(b) for b in BETAS], "thetas": THETAS,
           "count_rule": "strict local maxima on the grid with F > 1e-7 max F; theta=0 uses p padded by zeros",
           "families": {}}
    for name in FAMILIES:
        t1 = time.time()
        out["families"][name] = analyse_family(name)
        r = out["families"][name]
        print(f"{name:28s} LC={r['log_concave']!s:5s} psi_max={r['psi_num_local_maxima']} "
              f"maxSC={r['max_g_sign_changes']} maxF={r['max_num_maxima_F_theta_pos']} "
              f"beta_p2={r['min_beta_p_bimodal_theta0']} betaF2={r['min_beta_F_bimodal_by_theta']} "
              f"({time.time()-t1:.1f}s)", flush=True)
    out["details"] = []
    for (nm, be, th) in DETAIL_CASES:
        d1 = detail(nm, be, th)
        d2 = detail(nm, be, th, refine=2)
        d1["num_maxima_refined_grid"] = d2["num_maxima"]
        d1["maxima_refined_grid"] = d2["maxima"]
        out["details"].append(d1)
        print("detail", nm, be, th, d1["num_maxima"], d2["num_maxima"], d1.get("dip_ratio"), flush=True)
    lc = [k for k, v in out["families"].items() if v["log_concave"]]
    out["summary"] = {
        "log_concave_families": lc,
        "all_lc_single_sign_change": all(out["families"][k]["max_g_sign_changes"] == 1 for k in lc),
        "all_lc_single_max_F": all(out["families"][k]["max_num_maxima_F_theta_pos"] == 1 for k in lc),
        "runtime_s": time.time() - t0,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT, out["summary"])


if __name__ == "__main__":
    main()
