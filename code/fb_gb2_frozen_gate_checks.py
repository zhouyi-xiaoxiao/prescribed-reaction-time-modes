#!/usr/bin/env python3
"""Numerical checks of the frozen-gate exact count under a uniform contact floor.

Checked text (labels gb2:fg-*):
  manuscript/cnsns_submission/sm/th/GB2_frozen_gate_count.tex
  (Supplementary Material, subsection "Random frozen gates: the exact count
  under a uniform contact floor"; Theorem gb2:fg-count, Corollary
  gb2:fg-tangent, Remarks gb2:fg-rem-free, gb2:fg-rem-tangent,
  gb2:fg-rem-scope, gb2:fg-checks).

These checks test the algebra and the constants of the proofs; they are not a
substitute for the proofs.

Parts
  (1) 40-digit algebra (mpmath, 300 random parameter sets; gamma, D0 in
      [0.3, 2], rho in [0.3, 1.5], Q in [1, 5], s0^2 in {0, s_Z^2} or uniform
      in [0, 4 s_Z^2]): the bridge ODE and the form h = alpha y + D/y
      (gb2:fg-bridgeode), the two forms of the terminal slope (gb2:fg-beta),
      (log H)' = beta/eps^2 - S'/S (gb2:fg-logH), the score formula
      (gb2:fg-TY) against numerical differentiation of log p^Y, the score
      identity (gb2:fg-score-eq), the exact tilted means E T^Y = H'/H and
      E[(T^Y)^2 + d_theta T^Y] = H''/H (bridge covariance of gb1:tilt), the
      discriminant bound 4 alpha D <= theta c^2 (gb1:discriminant) and the
      bounds Lambda_h, Lambda_t of Lemma gb2:fg-free(b); and the Gaussian L^r
      identity (gb2:fg-Lp) against quadrature (40 further random cases).
  (2) Free interval of Lemma gb2:fg-free(c) at the anchors of the main text
      (gamma = D0 = rho = 1, Q = 4, tau = 0.5, T = 3.5; m = 2 targets 1.0, 2.5
      and m = 3 targets 0.8, 1.6, 2.8; s0^2 in {0, 1/2, 2}): the smallest, over
      a 61-point time grid and all l, of max_{u in [tau/8, 3tau/8]}
      dist(h_{l,t}(u), C) / r_g.
  (3) Boundary-tangent gate, d = 2 (Corollary gb2:fg-tangent, Remark
      gb2:fg-rem-tangent): |P(chi_t = 1) - 1/2| by exact one-dimensional
      quadrature at a = 0.4, gamma = D0 = 1, u0^2 = sigma0^2 = 0.09,
      t in {0.5, 2.5, 3.5}, eps in {0.1, 0.05, 0.025, 0.0125}.
  (4) Boundary-tangent gate with a transverse direction e_1 that need not be a
      coordinate axis (proof of Corollary gb2:fg-tangent; W = 1, a = 0.4):
      (a) the lift a e_1 + v with |v_i| < W/2 - a is the minimum-image
      representative (torus dimensions 2 and 3, 8000 random unit e_1 with v
      pushed to the edge of G_t); (b) d = 3, Monte Carlo (4e6 samples per
      case) of P(chi_t = 1) for an axis and a diagonal e_1; (c) the orthant
      switching probability P(chi_1 != chi_2) = 1/2 - arcsin(varrho_12)/pi at
      the m = 2 anchor (targets 1.0, 2.5; Remark gb2:fg-rem-tangent).
  (5) Illustration of the parenthesis in Remark gb2:fg-rem-scope(1): a
      deterministic longitudinal separation a e^{gamma (tau - t)} that reaches
      the contact sphere exactly at tau gives P(chi_tau = 1) -> 1/2 (exact
      quadrature, d = 2, anchor constants as in (3), tau = 0.5).

Seeds: Python random.seed(20260924) for (1); numpy default_rng(20260924) for
(4a)-(4b). One process; parts (2), (3), (4c) and (5) are deterministic.
Python: any Python 3 with mpmath and numpy (record written with CPython 3.14.6,
mpmath 1.4.1, numpy 2.5.3; about 45 s on one core).
Output: artifacts/data/exact_m_fixed_budget/GB2_frozen_gate/checks.json
Provenance: parts (1)-(3) are the writer's checks of 2026-09-24 (check_b2.py),
parts (4a)-(4c) the fixer's checks of the same day (fix_check.py); their
outputs are reproduced exactly (same seeds and call order).  Part (5) is new.
"""
from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
OUT_DIR = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "GB2_frozen_gate"

mp.mp.dps = 40


def R(a, b):
    return mp.mpf(random.uniform(a, b))


def data():
    g = R(0.3, 2.0); D0 = R(0.3, 2.0); rho = R(0.3, 1.5); Q = R(1.0, 5.0)
    sZ2 = D0 / (2 * g)
    s02 = random.choice([mp.mpf(0), sZ2, R(0.0, 4.0) * sZ2])
    return g, D0, rho, Q, sZ2, s02


def fns(g, D0, rho, Q, sZ2, s02):
    e0 = s02 - sZ2
    q = lambda t: Q * mp.e ** (-g * t)
    s2 = lambda t: sZ2 + e0 * mp.e ** (-2 * g * t)
    S2 = lambda t: s2(t) + rho ** 2
    C = lambda u, v: sZ2 * mp.e ** (-g * abs(u - v)) + e0 * mp.e ** (-g * (u + v))
    return e0, q, s2, S2, C


# --------------------------------------------------------------------------
# Part 1: 40-digit algebra
# --------------------------------------------------------------------------
def part1_algebra() -> dict:
    random.seed(20260924)
    err = {k: mp.mpf(0) for k in ['bridge_ode', 'bridge_disc', 'bridge_form', 'score_identity',
                                  'beta_forms', 'logH_prime', 'E_T', 'E_K2', 'score_formula',
                                  'Lp_identity']}
    viol = {'disc_bound': 0, 'lambda_h': 0, 'lambda_t': 0}
    N = 300
    for _ in range(N):
        g, D0, rho, Q, sZ2, s02 = data()
        e0, q, s2, S2, C = fns(g, D0, rho, Q, sZ2, s02)
        tl = R(0.2, 3.0); cl = q(tl)
        t = R(0.3, 3.5); x = q(t); dl = x - cl
        h = lambda u: q(u) - C(u, t) * dl / S2(t)
        # gb2:fg-bridgeode: h'' = g^2 h ; (h')^2 = g^2 (h^2 - 4 alpha D) ; h = alpha y + D/y
        u0 = R(0, 1) * t
        err['bridge_ode'] = max(err['bridge_ode'], abs(mp.diff(h, u0, 2) - g ** 2 * h(u0)) / (1 + abs(h(u0))))
        SS2 = sZ2 + rho ** 2
        alpha = (SS2 + e0 * x * cl / Q ** 2) / S2(t); D = sZ2 * x * (cl - x) / S2(t)
        y = q(u0)
        err['bridge_form'] = max(err['bridge_form'], abs(h(u0) - (alpha * y + D / y)))
        err['bridge_disc'] = max(err['bridge_disc'], abs(mp.diff(h, u0) ** 2 - g ** 2 * (h(u0) ** 2 - 4 * alpha * D)))
        theta0 = sZ2 / SS2
        if 4 * alpha * D > theta0 * cl ** 2 * (1 + mp.mpf(10) ** -30):
            viol['disc_bound'] += 1
        # gb2:fg-beta: three forms of the terminal slope
        v = s2
        vp = lambda s: mp.diff(v, s)
        beta1 = g * x * dl / S2(t) + vp(t) * dl ** 2 / (2 * S2(t) ** 2)
        beta2 = g * x * dl / S2(t) + (D0 / 2 - g * s2(t)) * dl ** 2 / S2(t) ** 2
        beta3 = g * x * dl * (SS2 + e0 * mp.e ** (-g * (t + tl))) / S2(t) ** 2
        err['beta_forms'] = max(err['beta_forms'], abs(beta1 - beta2) + abs(beta1 - beta3))
        # gb2:fg-logH: H'/H = beta/eps^2 - S'/S
        eps = R(0.05, 0.5)
        logH = lambda s: -mp.log(mp.sqrt(2 * mp.pi) * eps * mp.sqrt(S2(s))) - (q(s) - cl) ** 2 / (2 * eps ** 2 * S2(s))
        Sp_over_S = mp.diff(lambda s: mp.log(mp.sqrt(S2(s))), t)
        err['logH_prime'] = max(err['logH_prime'], abs(mp.diff(logH, t) - (beta1 / eps ** 2 - Sp_over_S)) * eps ** 2)
        # gb2:fg-score-eq: score identity at the tilted bridge means
        a = R(0, 0.5) * t; b = a + R(0.05, 0.45) * t
        th = b - a

        def sig2(th_):
            return sZ2 * (1 - mp.e ** (-2 * g * th_))

        def P2(th_, ya, yb):
            E = mp.e ** (-g * th_); Dl = yb - E * ya; md = -g * E * ya
            s2t = sig2(th_); s2d = mp.diff(sig2, th_)
            return Dl * md / s2t + Dl ** 2 * s2d / (2 * s2t ** 2)

        def TY(th_, ya, yb):
            s2t = sig2(th_); s2d = mp.diff(sig2, th_)
            return -s2d / (2 * s2t) + P2(th_, ya, yb) / eps ** 2

        ma, mb = h(a), h(b)
        err['score_identity'] = max(err['score_identity'], abs(P2(th, ma, mb) - beta1))
        # gb2:fg-TY: score formula = d/dtheta log p^Y
        ya, yb = ma + R(-1, 1) * eps, mb + R(-1, 1) * eps
        logp = lambda th_: -mp.log(mp.sqrt(2 * mp.pi * eps ** 2 * sig2(th_))) - (yb - ya * mp.e ** (-g * th_)) ** 2 / (2 * eps ** 2 * sig2(th_))
        err['score_formula'] = max(err['score_formula'], abs(mp.diff(logp, th) - TY(th, ya, yb)) * eps ** 2)
        # exact tilted expectations: E_{l,t} T^Y = H'/H ; E_{l,t}[(T^Y)^2 + d_theta T^Y] = H''/H
        Sig = mp.matrix([[C(a, a) - C(a, t) ** 2 / S2(t), C(a, b) - C(a, t) * C(b, t) / S2(t)],
                         [C(a, b) - C(a, t) * C(b, t) / S2(t), C(b, b) - C(b, t) ** 2 / S2(t)]]) * eps ** 2

        def quad_coeffs(F):
            f0 = F(ma, mb)
            gr = [mp.diff(lambda z: F(z, mb), ma), mp.diff(lambda z: F(ma, z), mb)]
            H11 = mp.diff(lambda z: F(z, mb), ma, 2); H22 = mp.diff(lambda z: F(ma, z), mb, 2)
            H12 = mp.diff(lambda z1, z2: F(z1, z2), (ma, mb), (1, 1))
            return f0, gr, mp.matrix([[H11, H12], [H12, H22]])

        f0, gr, Hs = quad_coeffs(lambda ya_, yb_: TY(th, ya_, yb_))
        A = Hs / 2
        ET = f0 + (A * Sig)[0, 0] + (A * Sig)[1, 1]
        Hp = mp.diff(logH, t)
        err['E_T'] = max(err['E_T'], abs(ET - Hp) * eps ** 2)
        gv = mp.matrix(gr)
        AS = A * Sig
        varT = (gv.T * Sig * gv)[0, 0] + 2 * ((AS * AS)[0, 0] + (AS * AS)[1, 1])
        ET2 = ET ** 2 + varT
        dT = lambda ya_, yb_: mp.diff(lambda th_: TY(th_, ya_, yb_), th)
        f0d, grd, Hsd = quad_coeffs(dT)
        EdT = f0d + ((Hsd / 2) * Sig)[0, 0] + ((Hsd / 2) * Sig)[1, 1]
        H2overH = mp.diff(logH, t, 2) + Hp ** 2
        err['E_K2'] = max(err['E_K2'], abs(ET2 + EdT - H2overH) * eps ** 4)
        # Lemma gb2:fg-free(b): |d_u h| <= Lambda_h, |d_t h| <= Lambda_t on u in [0, t]
        smax2 = max(s02, sZ2)
        Lh = g * Q * (1 + 2 * smax2 / rho ** 2)
        Lt = 2 * g * Q * (smax2 / rho ** 2) * (1 + smax2 / rho ** 2)
        for uu in [R(0, 1) * t for _ in range(3)]:
            if abs(mp.diff(h, uu)) > Lh:
                viol['lambda_h'] += 1
            ht = lambda tt: q(uu) - C(uu, tt) * (q(tt) - cl) / S2(tt)
            if abs(mp.diff(ht, t)) > Lt:
                viol['lambda_t'] += 1

    # gb2:fg-Lp: Gaussian L^r identity against quadrature
    for _ in range(40):
        eps = R(0.2, 1.0); beta_ = R(0.3, 2.0); p = R(2.0, 6.0); alpha_ = R(0.05, 0.95) * p * beta_
        bb = R(-1.5, 1.5); r = p / (p - 1)
        Nq = lambda z, mu, var: mp.e ** (-(z - mu) ** 2 / (2 * var)) / mp.sqrt(2 * mp.pi * var)
        integrand = lambda z: Nq(z, bb, eps ** 2 * alpha_) ** r * Nq(z, 0, eps ** 2 * beta_) ** (1 - r)
        lhs = mp.quad(integrand, [-mp.inf, bb - 10 * eps, bb, bb + 10 * eps, mp.inf])
        Dd = r * beta_ - (r - 1) * alpha_
        rhs = alpha_ ** ((1 - r) / 2) * beta_ ** (r / 2) * Dd ** (-mp.mpf(1) / 2) * mp.e ** (r * (r - 1) * bb ** 2 / (2 * eps ** 2 * Dd))
        err['Lp_identity'] = max(err['Lp_identity'], abs(lhs / rhs - 1))
        assert abs(Dd - (r - 1) * (p * beta_ - alpha_)) < mp.mpf(10) ** -30

    return {
        "n_random_sets": N,
        "n_Lp_cases": 40,
        "max_errors": {k: mp.nstr(v, 3) for k, v in err.items()},
        "max_error_overall": mp.nstr(max(err.values()), 3),
        "violations": viol,
    }


# --------------------------------------------------------------------------
# Part 2: free interval of Lemma gb2:fg-free(c)
# --------------------------------------------------------------------------
def free_interval_check(g, D0, rho, Q, s02, tj, tau, T, ngrid=300):
    sZ2 = D0 / (2 * g)
    e0, q, s2, S2, C = fns(g, D0, rho, Q, sZ2, s02)
    cs = [q(tt) for tt in tj]
    cm = cs[-1]
    Dstar = min([cm] + [cs[i] - cs[i + 1] for i in range(len(cs) - 1)])
    rg = min(Dstar / 4, g ** 2 * cm * tau ** 2 / 512)
    worst = mp.inf
    for k in range(ngrid + 1):
        t = tau + (T - tau) * k / ngrid
        for cl in cs:
            dl = q(t) - cl
            h = lambda u: q(u) - C(u, t) * dl / S2(t)
            best = max(min(abs(h(tau / 8 + (tau / 4) * i / 200) - c) for c in cs) for i in range(201))
            worst = min(worst, best / rg)
    return float(worst)


def part2_free_interval() -> dict:
    one = mp.mpf(1)
    fi = {}
    for name, tj, s02 in [('m2_stat', [1.0, 2.5], 0.5), ('m2_point', [1.0, 2.5], 0.0), ('m2_hot', [1.0, 2.5], 2.0),
                          ('m3_stat', [0.8, 1.6, 2.8], 0.5), ('m3_point', [0.8, 1.6, 2.8], 0.0),
                          ('m3_hot', [0.8, 1.6, 2.8], 2.0)]:
        fi[name] = free_interval_check(one, one, one, mp.mpf(4), mp.mpf(s02), [mp.mpf(x) for x in tj],
                                       mp.mpf(0.5), mp.mpf(3.5), ngrid=60)
    return {
        "anchors": "gamma = D0 = rho = 1, Q = 4, tau = 0.5, T = 3.5; s0^2: stat 0.5, point 0, hot 2",
        "time_grid_points": 61,
        "u_grid_points": 201,
        "min_ratio_maxdist_over_rg": fi,
        "min_over_anchors": min(fi.values()),
    }


# --------------------------------------------------------------------------
# Part 3: boundary-tangent gate, d = 2, exact quadrature
# --------------------------------------------------------------------------
def p_contact(eps, a, vpar, veta):
    """P(eps*eta^2 + 2 a eta + eps*X^2 < 0), eta ~ N(0, veta), X ~ N(0, vpar) independent."""
    sp, se = mp.sqrt(vpar), mp.sqrt(veta)

    def inner(xv):
        disc = a ** 2 - eps ** 2 * xv ** 2
        if disc <= 0:
            return mp.mpf(0)
        em = (-a - mp.sqrt(disc)) / eps; ep = (-a + mp.sqrt(disc)) / eps
        return mp.ncdf(ep / se) - mp.ncdf(em / se)

    return mp.quad(lambda xv: mp.npdf(xv, 0, sp) * inner(xv), [-a / eps, -5 * sp, 0, 5 * sp, a / eps])


def part3_tangent_d2() -> dict:
    one = mp.mpf(1)
    a = mp.mpf(0.4); D0 = one; g = one; u02 = mp.mpf(0.09); s02perp = mp.mpf(0.09)
    tang = {}
    ratios = []
    for tt in [mp.mpf(0.5), mp.mpf(2.5), mp.mpf(3.5)]:
        vpar = u02 * mp.e ** (-2 * g * tt) + (2 * D0 / g) * (1 - mp.e ** (-2 * g * tt))
        veta = s02perp + 4 * D0 * tt
        row = {}
        for eps_s in ["0.1", "0.05", "0.025", "0.0125"]:
            eps = mp.mpf(float(eps_s))
            dev = abs(p_contact(eps, a, vpar, veta) - mp.mpf(0.5))
            row["eps=" + eps_s] = {"dev": mp.nstr(dev, 5), "dev_over_sqrt_eps": mp.nstr(dev / mp.sqrt(eps), 5),
                                   "dev_over_eps": mp.nstr(dev / eps, 5)}
            ratios.append(float(dev / eps))
        tang["t=" + mp.nstr(tt, 3)] = row
    return {
        "setting": "d = 2, a = 0.4, gamma = D0 = 1, u0^2 = sigma0^2 = 0.09; dev = |P(chi_t = 1) - 1/2|",
        "rows": tang,
        "dev_over_eps_range": [round(min(ratios), 4), round(max(ratios), 4)],
    }


# --------------------------------------------------------------------------
# Part 4: non-axis transverse direction, d = 3; orthant switching probability
# --------------------------------------------------------------------------
def part4_nonaxis_and_orthant() -> dict:
    rng = np.random.default_rng(20260924)
    out = {}
    W, a = 1.0, 0.4
    # (a) minimum image on G_t for arbitrary unit e_1 in T_W^{d-1}, d-1 = 2 and 3
    viol = 0; tested = 0; worst_gap = 0.0
    for dm1 in (2, 3):
        shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * dm1)).reshape(dm1, -1).T * W
        for _ in range(4000):
            e1 = rng.normal(size=dm1); e1 /= np.linalg.norm(e1)
            # v with |v_i| < W/2 - a, pushed toward the edge
            u = rng.uniform(-1, 1, size=dm1)
            v = (W / 2 - a) * (1 - 1e-9) * np.sign(u) * np.abs(u) ** (1 / 5)
            L = a * e1 + v
            mi = np.min(np.linalg.norm(L[None, :] + shifts, axis=1))
            tested += 1
            gap = mi - np.linalg.norm(L)
            assert np.isfinite(gap)
            worst_gap = min(worst_gap, gap)
            if gap < -1e-14:
                viol += 1
    out["min_image_nonaxis_e1"] = {"tested": tested, "violations": viol, "min(mi-|lift|)": worst_gap,
                                   "note": "gap = |minimum image| - |lift| over the 3^(d-1) neighbouring images; 0 = the lift is the minimum image"}
    # (b) |P(chi_t=1)-1/2| for axis vs diagonal e_1, d = 3 (torus dimension 2), isotropic Sigma_perp0
    g, D0, u02, s02 = 1.0, 1.0, 0.09, 0.09
    aR = 2 * D0 / g
    N = 4_000_000
    res = {}
    max_z = 0.0
    for a in (0.4, 0.25):
        for t in (0.5, 2.5):
            vpar = math.exp(-2 * g * t) * u02 + aR * (1 - math.exp(-2 * g * t))
            vperp = s02 + 4 * D0 * t
            for eps in ((0.01, 0.005, 0.0025) if a == 0.4 else (0.02, 0.01, 0.005)):
                row = {}
                for name, e1 in (("axis", np.array([1.0, 0.0])), ("diag", np.array([1.0, 1.0]) / math.sqrt(2))):
                    xp = rng.normal(0, math.sqrt(vpar), N)
                    xq = rng.normal(0, math.sqrt(vperp), (N, 2))
                    Rperp = a * e1[None, :] + eps * xq
                    Rperp = (Rperp + W / 2) % W - W / 2          # wrap each coordinate: minimum image
                    r2 = (eps * xp) ** 2 + np.sum(Rperp ** 2, axis=1)
                    p = float(np.mean(r2 < a * a))
                    row[name] = {"P": round(p, 5), "dev_over_eps": round(abs(p - 0.5) / eps, 3)}
                row["mc_se"] = round(0.5 / math.sqrt(N), 6)
                z = abs(row["axis"]["P"] - row["diag"]["P"]) / (math.sqrt(2) * row["mc_se"])
                row["z_axis_minus_diag"] = round(z, 2)
                row["P(G^c) bound"] = float(2 * 2 * math.exp(-(W / 2 - a) ** 2 / (2 * eps ** 2 * vperp)))
                if row["P(G^c) bound"] < 1e-6:
                    max_z = max(max_z, z)
                res[f"a={a},t={t},eps={eps}"] = row
    out["tangent_UP_d3"] = {
        "setting": "d = 3 (torus dimension 2), W = 1, gamma = D0 = 1, u0^2 = 0.09, Sigma_perp0 = 0.09 I; N = 4e6 per direction",
        "cases": res,
        "max_z_axis_minus_diag_where_P(G^c)_bound<1e-6": round(max_z, 2),
    }
    # (c) orthant switching probability at the anchor (Remark gb2:fg-rem-tangent)
    t1, t2 = 1.0, 2.5
    r12 = math.sqrt((s02 + 4 * D0 * t1) / (s02 + 4 * D0 * t2))
    out["orthant"] = {"varrho_12": round(r12, 5), "P(chi1!=chi2)": round(0.5 - math.asin(r12) / math.pi, 5),
                      "P(chi1=0,chi2=1)": round(0.25 - math.asin(r12) / (2 * math.pi), 5)}
    return out


# --------------------------------------------------------------------------
# Part 5: atom of the entry time at tau (Remark gb2:fg-rem-scope(1))
# --------------------------------------------------------------------------
def part5_atom_example() -> dict:
    one = mp.mpf(1)
    a = mp.mpf(0.4); D0 = one; g = one; u02 = mp.mpf(0.09); s02perp = mp.mpf(0.09); tau = mp.mpf(0.5)
    # R_par(t) = a e^{g(tau - t)} + eps Xi_par(t) (OU fluctuation), R_perp(t) = eps Xi_perp(t) (Brownian);
    # at t = tau the separation direction is longitudinal: eta = Xi_par(tau), the other coordinate Xi_perp(tau).
    var_par = u02 * mp.e ** (-2 * g * tau) + (2 * D0 / g) * (1 - mp.e ** (-2 * g * tau))
    var_perp = s02perp + 4 * D0 * tau
    rows = {}
    for eps_s in ["0.02", "0.01", "0.005"]:
        eps = mp.mpf(float(eps_s))
        pc = p_contact(eps, a, var_perp, var_par)
        rows["eps=" + eps_s] = {"P(chi_tau=1)": mp.nstr(pc, 6), "dev_over_eps": mp.nstr(abs(pc - mp.mpf(0.5)) / eps, 4)}
    return {"setting": "d = 2, a = 0.4, gamma = D0 = 1, u0^2 = sigma0^2 = 0.09, tau = 0.5; noise-free separation reaches the contact sphere exactly at tau",
            "rows": rows}


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {
        "script": "code/fb_gb2_frozen_gate_checks.py",
        "checked_text": "manuscript/cnsns_submission/sm/th/GB2_frozen_gate_count.tex (labels gb2:fg-*)",
        "python": sys.version.split()[0],
        "mpmath": mp.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "mp_dps": mp.mp.dps,
    }
    t = time.time(); res["part1_algebra"] = part1_algebra(); res["part1_seconds"] = round(time.time() - t, 1)
    t = time.time(); res["part2_free_interval"] = part2_free_interval(); res["part2_seconds"] = round(time.time() - t, 1)
    t = time.time(); res["part3_tangent_d2_quadrature"] = part3_tangent_d2(); res["part3_seconds"] = round(time.time() - t, 1)
    t = time.time(); res["part4_nonaxis_e1_and_orthant"] = part4_nonaxis_and_orthant(); res["part4_seconds"] = round(time.time() - t, 1)
    t = time.time(); res["part5_atom_example"] = part5_atom_example(); res["part5_seconds"] = round(time.time() - t, 1)
    res["total_seconds"] = round(time.time() - t_start, 1)
    out = OUT_DIR / "checks.json"
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k.endswith("seconds")}))
    print(json.dumps(res["part1_algebra"]["max_errors"]))


if __name__ == "__main__":
    main()
