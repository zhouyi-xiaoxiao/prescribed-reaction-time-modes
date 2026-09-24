#!/usr/bin/env python3
"""TH-11 check: nominal floating-point evaluation of the sufficient budget B_cert,
including tightened windows tau in {0.6, 0.7, 0.8, 0.9}.

Independent float64 re-implementation of the published evaluation
(manuscript/prr_assets/b0_dyson_numerics.py, which uses mpmath at 40-60 digits
and is not runnable with the project Python because mpmath is absent):

  G(t)   = c(t) * gz(t),  c(t) = P(|R_t|_mi < a) (d = 2, quadrature in
           theta with x = a sin(theta), wrapped transverse normal, 13 images),
           gz(t) = W^{-(d-1)} sum_j w_j N(mu(t) - mu(t_j); eps^2 S_*^2),
  c', c''  from a 64-node Chebyshev interpolant on [tau-0.02, T+0.02] (as published),
  roots of G' on (tau, T) by sign change on a grid + bisection,
  root tubes = maximal intervals where sgn*G'' >= |G''(root)|/2 (outward march, step
  0.01, then bisection; as published), mu2 = min_roots |G''|/2,
  mu1 = min |G'| over the tube complement (grid) and the two endpoints,
  constants r0, t_theta, khat, Sigma_Z, Sigma_P, x_V, x_Z0, x_par0, Pi, v_inf,
  delta_v, v_eff, C_pre, Mhat and
      B_cert = log(1 + Mhat/C_pre) / (v_eff (T + r0)),  Mhat = min(r0 mu1, r0^2 mu2/2).

These are NOMINAL floating-point evaluations of the sufficient formula proved in
the archived SM (Sec. S3.5), not interval-certified bounds: mu1 is a grid minimum
and v_inf a grid maximum (neither is an enclosure).  delta_v is evaluated only at
x = x_V here, but delta_v = 0 holds analytically for every x in [0, 1): the
substitution y - zbar = (y' - zbar)/(1 - x) gives S_w(x) <= sqrt(1 - x) S_w(0)
(TH-11 remark).  Deterministic; runs in seconds.  Run from code/:
    python3 fb_bcert_window_check.py
Output: ../artifacts/data/exact_m_fixed_budget/TH11/bcert_nominal_windows.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "TH11" / "bcert_nominal_windows.json"

_erf = np.vectorize(math.erf, otypes=[float])


def Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / math.sqrt(2.0)))


class Case:
    def __init__(self, name, eps, targets, weights, tau=0.5, T=3.5):
        self.name, self.eps = name, float(eps)
        self.targets, self.w = list(targets), list(weights)
        self.gamma = self.D0 = 1.0
        self.z0, self.zbar, self.W, self.a, self.rho, self.ell0 = 4.0, 0.0, 1.0, 0.4, 1.0, 1.0
        self.tau, self.T, self.d = float(tau), float(T), 2
        self.u02, self.rpar0, self.Sperp0 = 0.09, 0.1, 0.09
        self.Sstar2 = self.D0 / (2 * self.gamma) + self.rho**2
        self.v0 = self.D0 / (2 * self.gamma)

    def mu(self, t):
        return self.zbar + (self.z0 - self.zbar) * np.exp(-self.gamma * np.asarray(t, float))

    def centres(self):
        return [float(self.mu(t)) for t in self.targets]


TH_NODES, TH_WTS = np.polynomial.legendre.leggauss(200)
THETA = 0.5 * math.pi * TH_NODES  # x = a sin(theta), theta in (-pi/2, pi/2)
THW = 0.5 * math.pi * TH_WTS


def contact(case, t):
    """c(t) = P(R_par^2 + R_perp,mi^2 < a^2), d = 2."""
    eps, g, D0, a = case.eps, case.gamma, case.D0, case.a
    vpar = case.u02 * math.exp(-2 * g * t) + (2 * D0 / g) * (1 - math.exp(-2 * g * t))
    spar = eps * math.sqrt(vpar)
    mpar = case.rpar0 * math.exp(-g * t)
    sperp = eps * math.sqrt(case.Sperp0 + 4 * D0 * t)
    x = a * np.sin(THETA)
    cth = a * np.cos(THETA)  # sqrt(a^2 - x^2) and dx/dtheta
    dens = np.exp(-(x - mpar) ** 2 / (2 * spar**2)) / (math.sqrt(2 * math.pi) * spar)
    pp = np.zeros_like(x)
    for k in range(-6, 7):
        pp += Phi((k * case.W + cth) / sperp) - Phi((k * case.W - cth) / sperp)
    return float(np.sum(THW * dens * pp * cth))


def build_contact_cheb(case):
    pad = 0.02
    f = np.vectorize(lambda tt: contact(case, tt))
    ch = Chebyshev.interpolate(f, 63, domain=[case.tau - pad, case.T + pad])
    return ch, ch.deriv(1), ch.deriv(2)


def gz_derivs(case, t):
    eps, g = case.eps, case.gamma
    s2 = eps**2 * case.Sstar2
    pref = 1.0 / (case.W ** (case.d - 1) * math.sqrt(2 * math.pi * s2))
    t = np.asarray(t, float)
    mt = case.mu(t)
    mup = -g * (mt - case.zbar)
    mupp = g**2 * (mt - case.zbar)
    f0 = np.zeros_like(t)
    f1 = np.zeros_like(t)
    f2 = np.zeros_like(t)
    for wj, cj in zip(case.w, case.centres()):
        dj = mt - cj
        e = wj * np.exp(-dj**2 / (2 * s2))
        f0 += e
        f1 += e * (-dj * mup / s2)
        f2 += e * ((dj * mup / s2) ** 2 - (mup**2 + dj * mupp) / s2)
    return pref * f0, pref * f1, pref * f2


def make_dG(case):
    c0, c1, c2 = build_contact_cheb(case)

    def dG(t, n):
        z0, z1, z2 = gz_derivs(case, t)
        a0, a1, a2 = c0(t), c1(t), c2(t)
        if n == 0:
            return a0 * z0
        if n == 1:
            return a0 * z1 + a1 * z0
        return a0 * z2 + 2 * a1 * z1 + a2 * z0
    return dG, (c0, c1)


def bisect(f, a, b, it=200):
    fa = f(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0:
            return m
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def margin_inventory(case, dG, npts=30000):
    ts = np.linspace(case.tau, case.T, npts + 1)
    d1 = dG(ts, 1)
    roots = []
    for i in np.flatnonzero(d1[:-1] * d1[1:] < 0):
        roots.append(bisect(lambda u: float(dG(np.array([u]), 1)[0]), ts[i], ts[i + 1]))
    curv = [float(dG(np.array([r]), 2)[0]) for r in roots]
    tubes = []
    for r, cv in zip(roots, curv):
        sgn = 1.0 if cv > 0 else -1.0
        half = abs(cv) / 2

        def h(u):
            return sgn * float(dG(np.array([u]), 2)[0]) - half
        step = 0.01
        lo = r
        while lo - step > case.tau and h(lo - step) > 0:
            lo -= step
        left = bisect(h, lo, lo - step) if lo - step > case.tau else case.tau
        hi = r
        while hi + step < case.T and h(hi + step) > 0:
            hi += step
        right = bisect(h, hi, hi + step) if hi + step < case.T else case.T
        tubes.append((left, right))
    mu2 = min(abs(c) for c in curv) / 2
    inside = np.zeros_like(ts, dtype=bool)
    for (L, R) in tubes:
        inside |= (ts > L) & (ts < R)
    comp = np.abs(d1[~inside])
    tcomp = ts[~inside]
    k = int(np.argmin(comp))
    mu1, argmin = float(comp[k]), float(tcomp[k])
    for tt in (case.tau, case.T):
        v = abs(float(dG(np.array([tt]), 1)[0]))
        if v <= mu1:
            mu1, argmin = v, tt
    return roots, curv, tubes, mu1, argmin, mu2


def mixture_sup(case, width2=None, shift=1.0):
    if width2 is None:
        width2 = case.eps**2 * case.rho**2
    cs = [shift * c for c in case.centres()]
    sd = math.sqrt(width2)
    z = np.linspace(min(cs) - 6 * sd, max(cs) + 6 * sd, 400001)
    f = sum(wj * np.exp(-(z - cj) ** 2 / (2 * width2)) for wj, cj in zip(case.w, cs)) / math.sqrt(2 * math.pi * width2)
    return float(f.max())


def assemble(case, mu1, mu2, lam=0.1):
    eps, g, D0, tau, T = case.eps, case.gamma, case.D0, case.tau, case.T
    cs = case.centres()
    yhat = max(abs(case.z0 - case.zbar), max(abs(c - case.zbar) for c in cs))
    r0 = (eps / (g * yhat)) * math.sqrt(D0 * tau / 2)
    R1 = r0 <= tau / 2
    t_theta = r0 / (tau - r0)
    khat = (1 - lam) ** (-1) * (1 + t_theta**2) ** ((case.d + 1) / 4.0)
    sigZ = (1 + t_theta**2 / lam) * g**2 * r0**2 / (2 * eps**2 * D0 * (tau - r0))
    sigP = sigZ / 4
    x_V = 2 * eps**2 * case.rho**2 * sigZ
    x_Z0 = 2 * eps**2 * case.v0 * sigZ
    x_par0 = 2 * eps**2 * case.u02 * sigP
    x_max = max(x_V, x_Z0, x_par0)
    R2 = x_max < 0.5
    rhat = max(case.a, abs(case.rpar0))
    Pi = (sigZ * yhat**2 + sigP * rhat**2) / (1 - x_max) + 0.5 * (x_V + x_Z0 + x_par0) / (1 - x_max)
    vsup_plain = mixture_sup(case)
    vsup_infl = mixture_sup(case, width2=eps**2 * case.rho**2 / (1 - x_V), shift=1 / (1 - x_V))
    v_inf = vsup_plain / case.W ** (case.d - 1)
    delta_v = max(0.0, vsup_infl / vsup_plain - 1)
    v_eff = khat * (1 + delta_v) * v_inf
    C_pre = khat * v_inf * math.exp(Pi)
    M_end, M_val = r0 * mu1, r0**2 * mu2 / 2
    Mhat = min(M_end, M_val)
    Bcert = math.log1p(Mhat / C_pre) / (v_eff * (T + r0))
    return dict(yhat=yhat, r0=r0, R1=R1, R2=R2, t_theta=t_theta, khat=khat, SigmaZ=sigZ, SigmaP=sigP, x_V=x_V,
                x_Z0=x_Z0, x_par0=x_par0, x_max=x_max, Pi=Pi, exp_Pi=math.exp(Pi), v_inf=v_inf, delta_v=delta_v,
                v_eff=v_eff, C_pre=C_pre, r0_mu1=M_end, r0sq_mu2_half=M_val, Mhat=Mhat,
                binding=("endpoint/complement slope (r=1)" if M_end <= M_val else "valley curvature (r=2)"),
                B_cert=Bcert, log10_B_cert=math.log10(Bcert))


def run_case(name, eps, targets, weights, tau):
    case = Case(name, eps, targets, weights, tau=tau)
    dG, (c0, c1) = make_dG(case)
    roots, curv, tubes, mu1, argmin, mu2 = margin_inventory(case, dG)
    res = dict(name=name, eps=eps, m=len(targets), targets=targets, weights=weights, tau=tau, T=case.T,
               c_tau=float(c0(tau)), b_tau=float(c1(tau) / c0(tau)),
               G_prime_tau=float(dG(np.array([tau]), 1)[0]), G_prime_T=float(dG(np.array([case.T]), 1)[0]),
               roots=roots, curvatures=curv, G_at_roots=[float(dG(np.array([r]), 0)[0]) for r in roots],
               tubes=tubes, mu1=mu1, mu1_at=argmin, mu2=mu2, n_roots=len(roots))
    res.update(assemble(case, mu1, mu2))
    DL = float(case.mu(tau) - case.mu(targets[0]))
    res["D_L"] = DL
    res["endpoint_exponent"] = DL**2 / (2 * case.Sstar2 * eps**2)
    return res


def main():
    out = dict(script="code/fb_bcert_window_check.py", item="TH-11", deterministic=True, seeds=None,
               status="nominal floating-point evaluation (float64), not interval-certified", rows=[])
    specs = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        specs.append(("m2_eps0.1", 0.1, [1.0, 2.5], [0.5, 0.5], tau))
    for tau in (0.5, 0.7, 0.9):
        specs.append(("m2_eps0.05", 0.05, [1.0, 2.5], [0.5, 0.5], tau))
    for tau in (0.5, 0.6, 0.7):
        specs.append(("m3_eps0.1", 0.1, [0.8, 1.6, 2.8], [1 / 3, 1 / 3, 1 / 3], tau))
    for s in specs:
        r = run_case(*s)
        out["rows"].append(r)
        print(f"{r['name']} tau={r['tau']}: roots={len(r['roots'])} mu1={r['mu1']:.5e}@{r['mu1_at']:.4f} "
              f"mu2={r['mu2']:.5e} r0={r['r0']:.5f} binding={r['binding']} B_cert={r['B_cert']:.4e} "
              f"(log10 {r['log10_B_cert']:.4f}) tubes={[(round(a, 5), round(b, 5)) for a, b in r['tubes']]}")
    # diagnostic exponents (leading-order, fixed geometry)
    S2 = 1.5
    mu = lambda t: 4 * math.exp(-t)
    out["diagnostic_exponents"] = dict(
        m2_endpoint_cB=(mu(0.5) - mu(1.0)) ** 2 / (2 * S2),
        m3_endpoint_cB=(mu(0.5) - mu(0.8)) ** 2 / (2 * S2),
        m2_valley_cB=(mu(1.0) - mu(2.5)) ** 2 / (8 * S2),
        m3_valley_cB=max((mu(0.8) - mu(1.6)) ** 2, (mu(1.6) - mu(2.8)) ** 2) / (8 * S2),
        note="endpoint: D_L^2/(2 S*^2); valley: Delta_max^2/(8 S*^2) (largest adjacent gap = shallowest valley)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)
    print(out["diagnostic_exponents"])


if __name__ == "__main__":
    main()
