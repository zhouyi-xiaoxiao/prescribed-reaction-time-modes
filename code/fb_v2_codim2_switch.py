#!/usr/bin/env python3
"""V2 item 3 (CV-2): interval certificates of the fold curves of the fixed-width limit law
along ALLOCATION families, the codimension-2 fold switch b_2 = b_3 at m = 3, and the
max-min allocation path at m = 2.  Deterministic, no random numbers, no seeds.
mpmath.iv, outward rounding, 100 bits (precision set by fb_th8_interval_certificate).

Model (anchor geometry of Prop. 4: gamma = D0 = 1, z0 = 4, zbar = 0, W = 1, d = 2,
I = [tau, T] = [0.5, 3.5]):  mu(t) = 4 e^{-t}, c_j = mu(t_j),
    E_j(t) = exp(-(mu(t) - c_j)^2 / (2 rho^2)),   G0 = (2 pi)^{-1/2} rho^{-1} sum_j w_j E_j,
    r0 = G0'/G0^2,  N = G0'' G0 - 2 G0'^2  (sign r0' = sign N; r0'' = N'/G0^3 where N = 0).
Every allocation family used here is AFFINE in one parameter p:
    G0(t; p) = pref * (sum_j alpha_j E_j(t) + p * sum_j beta_j E_j(t)),
so p occurs once in G0 and dG0/dp = pref * sum_j beta_j E_j exactly.
  * F2W  (m = 2, targets (1, 2.5), rho_s = 0.3):  w = (1 - p, p), p = w_2 (late-stripe weight).
  * F3S  (m = 3, targets (0.8, 1.6, 2.8), rho_s = 0.2):  w = (1 - eta, eta s, eta (1 - s)),
          eta = 1e-3, s = q/(1+q), q = w_2/w_3  (the codimension-2 switch family).
  * F3A  same with eta = 2/3, i.e. w_1 = 1/3 (the equal-weight share; the natural family).
Parameter boxes are given by exact decimal end points of the user parameter (w_2 or q);
consecutive boxes share end points, so the certified set is a union of closed intervals.

Per parameter box P (every statement holds for EVERY parameter value in P):
  * (D1) parametric zero counting of G0' on I (runs of strictly monotone cells, merged;
    definite end signs), signature max,min,...,max, G0'(tau) > 0 > G0'(T).
  * (D2) N < 0 on [tau, hi(hat t_1)].
  * (D3) for j >= 2: on [lo(check t_{j-1}), hi(hat t_j)] N has exactly one zero, a decreasing
    run (N' < 0, so r0'' < 0), N > 0 on encl(check t_{j-1}), N < 0 on encl(hat t_j).
  * fold of flank j: parametric 2-D Krawczyk on H(t, B; p) = (G0' - B G0^2, G0'' - 2 B G0 G0')
    with X_t inside the (D3) search interval (so its unique zero is (t_{f,j}(p), b_j(p)));
    also the centred-form enclosure b_j in r0(t_m) + (N/G0^3)(T_f)(T_f - t_m) (T_f = the
    (D3) zero enclosure); the two are intersected.
  * a_{f,j} = -r0''(t_{f,j}) = -N'/G0^3 > 0 enclosed on T_f.
  * r0(tau) enclosed; the binding candidate (argmin of {r0(tau), b_2, ..}) is certified when its
    enclosure lies strictly below all others ((D4) on the box).
  * d b_j / d p = d_p r0 (t_{f,j}(p); p) (envelope identity, d_t r0 = 0 at the fold) enclosed on
    T_f x P; used for the monotonicity of b_2 - b_3 near the switch.
Codimension-2 switch (F3S): verified bisection in q with thin certificates (sign of b_2 - b_3),
monotonicity of b_2 - b_3 on the final bracket (envelope derivative), enclosure of
B* = b_2 = b_3 by the thick certificate on the final bracket; independent 4-D Krawczyk on
(t_2, t_3, B, s) for the extended system {H(t_2), H(t_3)} = 0 as a cross-check.
Max-min path (F2W): the design law of Prop. 3(iii) (A = 1, v_j = 4 e^{-t_j}) is the explicit
monotone curve p -> (w_2*(p), B(p)); with x = 1 - 2p = e^{-y}:
    lambda_1 = ln 2 - ln(1 + x),  lambda_2 = ln((1 + x)/2) + y,  B = v1 lambda_1 + v2 lambda_2,
    w_2* = v2 lambda_2 / B.  B and w_2* are strictly increasing in y (lambda_2/lambda_1 = h(a)/a - 1,
    h(a) = -ln(2e^{-a} - 1) convex with h(0) = 0, a = lambda_1).  The budget B_path(w) at which
    w_2* = w is enclosed by interval bisection in y; the path lies below the fold curve on the
    w-box W_k if hi B_path(hi W_k) < lo min(r0(tau), b_2)(W_k).

Certified primitives (imported from fb_th8_interval_certificate.py, lines ~150-200; all in mpmath.iv):
    jet c[k] encloses F^{(k)}(t)/k! (Taylor coefficients with interval entries);
    jadd/jsub: termwise interval sum/difference; jmul: Cauchy product sum_{i<=k} a_i b_{k-i};
    jscale(a, s) = [s * a_k]: s is an iv interval (I(-1), I(4), inv2s2, pref, P) or an exact Python int;
    jexp(a): z_0 = iv.exp(a_0), z_k = (1/k) sum_{i=1}^k i a_i z_{k-i}  (from z' = a' z), interval ops only;
    jdiv: q_k = (a_k - sum_{i=1}^k b_i q_{k-i}) / b_0;  jderiv: jet of F^{(r)} (factor (k+1)).
    taylor_encl(jm, jJ, r, J, m) = F^{(r)}(m) + F^{(r+1)}(m)(J-m) + F^{(r+2)}(J)(J-m)^2/2 (order-2 Taylor form).
    Outward serialisation: _outward moves an mp endpoint by 1e-17 relative, prints 18 significant digits
    (nearest; relative error <= 5e-18) and asserts the printed value is outward; exact() prints a dyadic exactly.
Contracts (A3 audit):
  * certified (per box) = (D1)-(D3) and the fold enclosures hold for every parameter in the box.  (D4) and the
    binding candidate are SEPARATE fields ("D4_on_box", "binding"); a theorem-level claim that needs (D4) or a
    binding is checked by code/fb_v2_cert_validator.py with exact rational comparisons.
  * The fold value is certified by (D3) (unique nondegenerate zero of N on the search interval) plus the
    centred-form / mean-value enclosures; the thick-P Krawczyk operator is an OPTIONAL independent cross-check
    (it is intersected in when it succeeds, but certification does not depend on it).
  * Krawczyk: K = x~ - Y H(x~) + (Id - Y DH(X))(X - x~) with Y a real approximate inverse used as a fixed
    preconditioner (it need not enclose the exact inverse); K in int X proves a unique zero in X for every
    parameter in the box; X_t inside the (D3) search interval identifies it with the fold.
  * mp (nearest) arithmetic is used only for choices (grid points, Newton starts, Krawczyk centres,
    preconditioners, trial boxes); bounds are iv expressions or exact Fraction/Decimal comparisons.

Run (from code/):
    python fb_v2_codim2_switch.py [--workers 3] [--quick]
Output: ../artifacts/data/exact_m_fixed_budget/V2_bifurcation/codim2_switch.json (+ _boxes.json)
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
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fb_th8_interval_certificate as th8  # noqa: E402  (sets iv.prec = mp.prec = 100)
from fb_th8_interval_certificate import (I, lo, hi, pos, neg, to_list, down, up,  # noqa: E402
                                         jmul, jadd, jsub, jscale, jexp, jderiv)
import fb_v2_fold_curve_certificate as cv1  # noqa: E402  (PFun: parametric zero counting)
from fb_v2_fold_curve_certificate import PFun, roots_in  # noqa: E402
from mpmath import iv, mp  # noqa: E402

PREC = th8.PREC
DATA = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "V2_bifurcation"
OUT = DATA / "codim2_switch.json"
OUT_BOXES = DATA / "codim2_switch_boxes.json"
TAU, TEND = mp.mpf("0.5"), mp.mpf("3.5")
MAX_DEPTH = 4                     # adaptive bisection depth of a failed parameter box


# ----------------------------------------------------------------------------
# allocation families (affine in one parameter)
class Family:
    def __init__(self, key, m, targets, rho, alpha, beta, pname, note):
        self.key, self.m, self.targets, self.rho_str = key, m, targets, rho
        self.pname, self.note = pname, note
        self.c = [4 * iv.exp(-I(tj)) for tj in targets]
        self.rho = I(rho)
        self.pref = I(1) / (iv.sqrt(2 * iv.pi) * self.rho)
        self.inv2s2 = I(1) / (2 * self.rho * self.rho)
        self.alpha, self.beta = alpha, beta

    def parts(self, t, order):
        """jets (length order+1) of pref*sum alpha_j E_j and pref*sum beta_j E_j at the interval t."""
        tj = [t, I(1)] + [I(0)] * (order - 1)
        mu = jscale(jexp(jscale(tj, I(-1))), I(4))
        base = [I(0)] * len(mu)
        slope = [I(0)] * len(mu)
        for cj, a, b in zip(self.c, self.alpha, self.beta):
            d = list(mu)
            d[0] = d[0] - cj
            e = jexp(jscale(jmul(d, d), -self.inv2s2))
            if a is not None:
                base = jadd(base, jscale(e, a))
            if b is not None:
                slope = jadd(slope, jscale(e, b))
        return jscale(base, self.pref), jscale(slope, self.pref)

    def g0(self, t, P, order):
        base, slope = self.parts(t, order)
        return jadd(base, jscale(slope, P))

    def param_box(self, u_lo: str, u_hi: str):
        """interval P of the affine parameter for the user-parameter box [u_lo, u_hi] (decimals)."""
        U = iv.mpf([u_lo, u_hi])
        if self.pname == "w2":
            return U
        return 1 - 1 / (1 + U)          # s = q/(1+q), monotone, single occurrence

    def param_point(self, u):
        """thin-ish interval of the affine parameter at an mp number u of the user parameter."""
        U = I(u)
        if self.pname == "w2":
            return U
        return 1 - 1 / (1 + U)


def fam_F2W():
    return Family("F2W", 2, ("1.0", "2.5"), "0.3", [I(1), None], [I(-1), I(1)], "w2",
                  "m=2, rho_s=0.3, w=(1-w2, w2)")


ETA_SWITCH = "0.001"


def fam_F3(eta_str, key):
    eta = I(eta_str) if "/" not in eta_str else I(int(eta_str.split("/")[0])) / int(eta_str.split("/")[1])
    return Family(key, 3, ("0.8", "1.6", "2.8"), "0.2", [1 - eta, None, eta], [None, eta, -eta], "q",
                  f"m=3, rho_s=0.2, w=(1-eta, eta q/(1+q), eta/(1+q)), eta={eta_str}")


def fam_F3S():
    return fam_F3(ETA_SWITCH, "F3S")


def fam_F3A():
    return fam_F3("2/3", "F3A")


FAMILIES = {"F2W": fam_F2W, "F3S": fam_F3S, "F3A": fam_F3A}


# ----------------------------------------------------------------------------
# helpers
def mid(x):
    return (lo(x) + hi(x)) / 2


def r0_of(g):
    return g[1] / (g[0] * g[0])


def N_jet(g):
    g1, g2 = jderiv(g, 1), jderiv(g, 2)
    return jsub(jmul(g2, g), jscale(jmul(g1, g1), I(2)))


def jet_N(fam):
    return lambda U, A, o: N_jet(fam.g0(U, A, o + 2))


def jet_G0p(fam):
    return lambda U, A, o: jderiv(fam.g0(U, A, o + 1), 1)


def inside(x, y):
    """x strictly inside y."""
    return lo(x) > lo(y) and hi(x) < hi(y)


def hull2(x, y):
    return iv.mpf([min(lo(x), lo(y)), max(hi(x), hi(y))])


def sym(c, r):
    return iv.mpf([c - r, c + r])


# ----------------------------------------------------------------------------
# small dense interval linear algebra for Krawczyk (Y = any real matrix; mp inverse of mid)
def krawczyk(Hfun, Jfun, X, max_infl=30, bounds=None):
    """parametric Krawczyk: Hfun(list of intervals) -> list, Jfun -> matrix (list of lists).
    Returns (ok, X_final, n_inflations).  ok means K(X) in int(X): for every parameter value
    in the (thick) parameter box used inside Hfun/Jfun, H has exactly one zero in X."""
    n = len(X)
    for it in range(max_infl):
        xt = [I(mid(x)) for x in X]
        Hm = Hfun(xt)
        Jm = Jfun(xt)
        A = mp.matrix(n, n)
        for i in range(n):
            for k in range(n):
                A[i, k] = mid(Jm[i][k])
        try:
            Yinv = A ** -1
        except ZeroDivisionError:
            return False, X, it
        Y = [[I(Yinv[i, k]) for k in range(n)] for i in range(n)]
        JX = Jfun(X)
        K = []
        for i in range(n):
            yh = I(0)
            for k in range(n):
                yh += Y[i][k] * Hm[k]
            row = I(0)
            for k in range(n):
                s = I(0)
                for l_ in range(n):
                    s += Y[i][l_] * JX[l_][k]
                row += ((I(1) if i == k else I(0)) - s) * (X[k] - xt[k])
            K.append(xt[i] - yh + row)
        if all(inside(K[i], X[i]) for i in range(n)):
            # tighten: X <- K(X) cap X (each iterate still contains the unique zero)
            Xc = [K[i] for i in range(n)]
            for _ in range(6):
                xt = [I(mid(x)) for x in Xc]
                Hm = Hfun(xt)
                Jm = Jfun(xt)
                JX = Jfun(Xc)
                K2 = []
                for i in range(n):
                    yh = I(0)
                    for k in range(n):
                        yh += Y[i][k] * Hm[k]
                    row = I(0)
                    for k in range(n):
                        s = I(0)
                        for l_ in range(n):
                            s += Y[i][l_] * JX[l_][k]
                        row += ((I(1) if i == k else I(0)) - s) * (Xc[k] - xt[k])
                    K2.append(xt[i] - yh + row)
                Xn = []
                for i in range(n):
                    a_, b_ = max(lo(K2[i]), lo(Xc[i])), min(hi(K2[i]), hi(Xc[i]))
                    if a_ > b_:
                        raise RuntimeError("Krawczyk tightening: empty intersection (internal error)")
                    Xn.append(iv.mpf([a_, b_]))
                Xc = Xn
            return True, Xc, it
        # epsilon-inflation around the hull of X and K
        Xn = []
        for i in range(n):
            h = hull2(X[i], K[i])
            r = (hi(h) - lo(h)) / 2
            Xn.append(sym(mid(h), r * mp.mpf("1.1") + mp.mpf("1e-28") * (1 + abs(mid(h)))))
        if bounds is not None and any(lo(Xn[i]) < bounds[i][0] or hi(Xn[i]) > bounds[i][1] for i in range(n)):
            return False, Xn, it + 1          # inflation left the admissible region: give up
        X = Xn
    return False, X, max_infl


def fold_system(fam, P):
    """H(t, B; p) = (G0' - B G0^2, G0'' - 2 B G0 G0') and its (t, B)-Jacobian.  H at a point x is
    evaluated in the mean-value form in p: H(x; p_m) + d_p H(x; P) (P - p_m) (p occurs once in G0,
    d_p G0^{(k)} = slope^{(k)} exactly), which removes the dependency blow-up of the thick P."""
    pm = I(mid(P))
    dP = P - pm

    def H(x):
        t, B = x
        base, sl = fam.parts(t, 2)
        gm = jadd(base, jscale(sl, pm))
        gP = jadd(base, jscale(sl, P))
        g0, g1, g2 = gm[0], gm[1], 2 * gm[2]
        G0, G1 = gP[0], gP[1]
        s0, s1, s2 = sl[0], sl[1], 2 * sl[2]
        d1 = s1 - 2 * B * G0 * s0
        d2 = s2 - 2 * B * (s0 * G1 + G0 * s1)
        return [g1 - B * g0 * g0 + d1 * dP, g2 - 2 * B * g0 * g1 + d2 * dP]

    def J(x):
        t, B = x
        g = fam.g0(t, P, 3)
        g0, g1, g2, g3 = g[0], g[1], 2 * g[2], 6 * g[3]
        return [[g2 - 2 * B * g0 * g1, -g0 * g0],
                [g3 - 2 * B * (g1 * g1 + g0 * g2), -2 * g0 * g1]]
    return H, J


def r0_mv(fam, t, P):
    """r0(t; P) in the mean-value form in p (t an interval)."""
    pm = I(mid(P))
    base, sl = fam.parts(t, 1)
    gm = jadd(base, jscale(sl, pm))
    gP = jadd(base, jscale(sl, P))
    dr = sl[1] / gP[0] ** 2 - 2 * gP[1] * sl[0] / gP[0] ** 3
    naive = r0_of(gP)
    v = r0_of(gm) + dr * (P - pm)
    return iv.mpf([max(lo(v), lo(naive)), min(hi(v), hi(naive))])


def dr0_dp(fam, t, P):
    base, sl = fam.parts(t, 1)
    g = jadd(base, jscale(sl, P))
    return sl[1] / g[0] ** 2 - 2 * g[1] * sl[0] / g[0] ** 3


def newton_fold(fam, P, t0):
    """100-bit Newton (not a proof; only a centre for the Krawczyk test) for N = 0 at thin P."""
    t = mp.mpf(t0)
    for _ in range(60):
        Nj = N_jet(fam.g0(I(t), P, 4))
        f, fp = mid(Nj[0]), mid(Nj[1])
        if fp == 0:
            break
        dt = f / fp
        t -= dt
        if abs(dt) < mp.mpf("1e-28"):
            break
    return t, mid(r0_of(fam.g0(I(t), P, 1)))


def fold_derivs(fam, T, P):
    """enclosures over T x P of dt_f/dp = -(d_p N)/N' and db/dp = d_p r0; valid for the fold
    (t_f(p), p) whenever t_f(p) lies in T for every p in P (implicit function theorem, N' < 0 there;
    envelope identity d_t r0 = 0 at the fold)."""
    base, sl = fam.parts(T, 3)
    g = jadd(base, jscale(sl, P))
    g0, g1, g2, g3 = g[0], g[1], 2 * g[2], 6 * g[3]
    s0, s1, s2 = sl[0], sl[1], 2 * sl[2]
    Np = g3 * g0 - 3 * g1 * g2                      # N' = G0''' G0 - 3 G0' G0''
    dpN = s2 * g0 + g2 * s0 - 4 * g1 * s1           # d_p N
    return -dpN / Np, s1 / g0 ** 2 - 2 * g1 * s0 / g0 ** 3


# ----------------------------------------------------------------------------
# certificate on one parameter box
def certify_P(fam, P, label):
    """(D1)-(D3), folds, r0(tau), binding and envelope derivatives, for every parameter in P."""
    m = fam.m
    rec = {"box": label, "certified": False}
    t0 = time.time()
    try:
        # ---- (D1)
        Gp = PFun(jet_G0p(fam), P, "G0'")
        segs, err = Gp.segments(TAU, TEND)
        if segs is None:
            rec["fail"] = "D1: " + err
            return rec
        roots = roots_in(segs)
        types = ["max" if r["kind"] == "dec" else "min" for r in roots]
        expect = ["max", "min"] * (m - 1) + ["max"]
        g1tau, g1T = Gp.point(TAU), Gp.point(TEND)
        if not (types == expect and pos(g1tau) and neg(g1T)):
            rec["fail"] = f"D1: signature {types}, G0'(tau)>0 {pos(g1tau)}, G0'(T)<0 {neg(g1T)}"
            rec["D1_signature"] = types
            return rec
        crit = []
        for r in roots:
            L, R = Gp.shrink_root(r)
            crit.append(iv.mpf([L, R]))
        hats = crit[0::2]
        checks = crit[1::2]
        rec["D1"] = {"signature": types, "crit_t": [to_list(c) for c in crit]}
        # ---- (D2)
        Nf = PFun(jet_N(fam), P, "N")
        segs2, err = Nf.segments(TAU, hi(hats[0]))
        if segs2 is None:
            rec["fail"] = "D2: " + err
            return rec
        if not all(sg["sl"] < 0 and sg["sr"] < 0 for sg in segs2):
            rec["fail"] = "D2: N not certified negative on the first flank"
            return rec
        # ---- (D3) + folds
        flanks = []
        for j in range(2, m + 1):
            a_, b_ = lo(checks[j - 2]), hi(hats[j - 1])
            segs3, err = Nf.segments(a_, b_)
            if segs3 is None:
                rec["fail"] = f"D3 flank {j}: " + err
                return rec
            rts = roots_in(segs3)
            vc, _ = Nf.cell(lo(checks[j - 2]), hi(checks[j - 2]))
            vh, _ = Nf.cell(lo(hats[j - 1]), hi(hats[j - 1]))
            if not (len(rts) == 1 and rts[0]["kind"] == "dec" and pos(vc) and neg(vh)):
                rec["fail"] = (f"D3 flank {j}: {len(rts)} zero runs, kinds {[x['kind'] for x in rts]}, "
                               f"N>0 at check {pos(vc)}, N<0 at hat {neg(vh)}")
                return rec
            L, R = Nf.shrink_root(rts[0])
            Tf = iv.mpf([L, R])            # contains t_{f,j}(p) for every p in P (zero counting)
            bnds = [(a_, b_), (mp.mpf(0), mp.mpf(10) ** 12)]
            pm = I(mid(P))
            # (A) thin midpoint p_m: 100-bit Newton guess, then Krawczyk with epsilon-inflation from a tiny
            #     box.  Any zero of H with t in the (D3) search interval has N = 0 there, hence IS the fold
            #     (t_f(p_m), b_j(p_m)); X_t inside [L, R] (the zero-count enclosure) identifies it.
            tg, bg = newton_fold(fam, pm, mid(Tf))
            Hm, Jm = fold_system(fam, pm)
            okm, Xm, _ = krawczyk(Hm, Jm, [sym(tg, mp.mpf("1e-24")), sym(bg, abs(bg) * mp.mpf("1e-24"))],
                                  bounds=bnds)
            okm = okm and lo(Xm[0]) >= L and hi(Xm[0]) <= R
            # (B) mean-value tubes in p (derivative enclosures on the current t-enclosure x P):
            #     t_f(p) in t_f(p_m) + [dt_f/dp](P - p_m),  b_j(p) in b_j(p_m) + [db_j/dp](P - p_m)
            Tt, Bt = Tf, None
            dt_dp, db_dp = fold_derivs(fam, Tt, P)
            if okm:
                for _ in range(2):
                    Tn = Xm[0] + dt_dp * (P - pm)
                    Tt = iv.mpf([max(lo(Tn), L), min(hi(Tn), R)])
                    Bt = Xm[1] + db_dp * (P - pm)
                    dt_dp, db_dp = fold_derivs(fam, Tt, P)
                Bt = iv.mpf([max(lo(Bt), lo(Xm[1] + db_dp * (P - pm))),
                             min(hi(Bt), hi(Xm[1] + db_dp * (P - pm)))])
            # (C) centred form in t (on Tt) + mean-value form in p
            tm = I(mid(Tt))
            G0T = fam.g0(Tt, P, 0)[0]
            vN, dN = Nf.cell(lo(Tt), hi(Tt))      # Taylor-form enclosures of N and N' on Tt x P
            b_cf = r0_mv(fam, tm, P) + (vN / G0T ** 3) * (Tt - tm)
            # a_f = -r0'' = -N'/G0^3 on Tt x P
            a_f = -dN / G0T ** 3
            if not pos(a_f):
                rec["fail"] = f"flank {j}: a_f not certified positive"
                return rec
            # (D) parametric Krawczyk with thick P started from the tube box (independent enclosure)
            B0 = Bt if Bt is not None else b_cf
            X0 = [sym(mid(Tt), (hi(Tt) - lo(Tt)) * mp.mpf("0.55") + mp.mpf("1e-26")),
                  sym(mid(B0), (hi(B0) - lo(B0)) * mp.mpf("0.55") + abs(mid(B0)) * mp.mpf("1e-26"))]
            H, J = fold_system(fam, P)
            ok, X, nit = krawczyk(H, J, X0, bounds=bnds)
            k_ok = ok and lo(X[0]) > a_ and hi(X[0]) < b_
            encl = [b_cf] + ([Bt] if Bt is not None else []) + ([X[1]] if k_ok else [])
            bl, bh = max(lo(x) for x in encl), min(hi(x) for x in encl)
            if bl > bh:
                raise RuntimeError(f"flank {j}: fold enclosures disjoint (internal error)")
            bj = iv.mpf([bl, bh])
            tl, th = lo(Tt), hi(Tt)
            if k_ok:
                tl, th = max(tl, lo(X[0])), min(th, hi(X[0]))
            if tl > th:
                raise RuntimeError(f"flank {j}: fold time enclosures disjoint (internal error)")
            tf = iv.mpf([tl, th])
            flanks.append({"j": j, "search": [a_, b_], "t_f": tf, "b": bj, "a_f": a_f, "krawczyk": k_ok,
                           "krawczyk_mid": okm, "krawczyk_inflations": nit, "db_dp": db_dp,
                           "b_mid": Xm[1] if okm else None})
        # ---- r0(tau) and binding / (D4)
        r0tau = r0_mv(fam, I(TAU), P)
        cands = {"left_endpoint": r0tau}
        for f in flanks:
            cands[f"flank_{f['j']}"] = f["b"]
        binding = None
        for k, v in cands.items():
            if all(hi(v) < lo(w) for kk, w in cands.items() if kk != k):
                binding = k
        rec.update({
            "certified": True,
            "D2": True, "D3": True,
            "flanks": [{"j": f["j"], "t_f": to_list(f["t_f"]), "b": to_list(f["b"]),
                        "a_f_lower": down(lo(f["a_f"])), "krawczyk_ok": f["krawczyk"],
                        "krawczyk_mid_ok": f["krawczyk_mid"], "krawczyk_inflations": f["krawczyk_inflations"],
                        "db_dp": to_list(f["db_dp"])} for f in flanks],
            "r0_tau": to_list(r0tau),
            "binding": binding,
            "D4_on_box": binding is not None and binding != "left_endpoint",
        })
        rec["_iv"] = {"b": {f["j"]: f["b"] for f in flanks}, "db": {f["j"]: f["db_dp"] for f in flanks},
                      "bmid": {f["j"]: f["b_mid"] for f in flanks},
                      "tf": {f["j"]: f["t_f"] for f in flanks}, "r0tau": r0tau,
                      "search": {f["j"]: f["search"] for f in flanks},
                      "crit": crit}
        return rec
    except (RuntimeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        rec["fail"] = f"exception: {exc}"
        return rec
    finally:
        rec["runtime_s"] = round(time.time() - t0, 2)


# ----------------------------------------------------------------------------
# scans over parameter boxes (decimal end points; adaptive bisection of failed boxes)
def dec_mid(a: str, b: str) -> str:
    """exact decimal midpoint of two decimal strings."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    return format((Decimal(a) + Decimal(b)) / 2, "f").rstrip("0").rstrip(".")


def run_box(args):
    fk, a, b, depth = args
    fam = FAMILIES[fk]()
    rec = certify_P(fam, fam.param_box(a, b), [a, b])
    rec["depth"] = depth
    iv_part = rec.pop("_iv", None)
    if iv_part is not None:     # serialise the interval parts needed later (outward decimal strings)
        rec["_b_mid"] = {str(j): (to_list(v) if v is not None else None) for j, v in iv_part["bmid"].items()}
    return rec


def scan(fk, edges, workers):
    """certify every box [edges[k], edges[k+1]]; failed boxes are bisected up to MAX_DEPTH."""
    todo = [(fk, edges[k], edges[k + 1], 0) for k in range(len(edges) - 1)]
    leaves = []
    while todo:
        if workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                res = pool.map(run_box, todo, chunksize=1)
        else:
            res = [run_box(x) for x in todo]
        todo = []
        for r in res:
            a, b = r["box"]
            if r["certified"] or r["depth"] >= MAX_DEPTH:
                leaves.append(r)
            else:
                c = dec_mid(a, b)
                todo += [(fk, a, c, r["depth"] + 1), (fk, c, b, r["depth"] + 1)]
    from decimal import Decimal
    leaves.sort(key=lambda r: Decimal(r["box"][0]))
    return leaves


def covered_intervals(leaves):
    """maximal unions of consecutive certified boxes (exact decimal end points)."""
    out = []
    for r in leaves:
        if not r["certified"]:
            continue
        a, b = r["box"]
        if out and out[-1][1] == a:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def log_edges(a: float, b: float, n: int, sig: int = 6):
    """n log-uniform boxes on [a, b]; end points rounded to `sig` significant decimal digits."""
    out = []
    for k in range(n + 1):
        x = a * (b / a) ** (k / n)
        s = f"{x:.{sig}g}"
        if out and s == out[-1]:
            continue
        out.append(s)
    out[0], out[-1] = f"{a:.{sig}g}", f"{b:.{sig}g}"
    return out


def lin_edges(a: str, b: str, step: str):
    from decimal import Decimal
    A, Bd, S = Decimal(a), Decimal(b), Decimal(step)
    out = []
    x = A
    while x < Bd:
        out.append(format(x, "f"))
        x += S
    out.append(format(Bd, "f"))
    return out


# ----------------------------------------------------------------------------
# codimension-2 switch b_2 = b_3 (family F3S)
def switch_sign(fam, q):
    """sign of b_2 - b_3 at the point q (mp number; thin certificate).  None if not certified."""
    rec = certify_P(fam, fam.param_point(q), [mp.nstr(q, 30)] * 2)
    if not rec["certified"]:
        return None, rec
    b2, b3 = rec["_iv"]["b"][2], rec["_iv"]["b"][3]
    if hi(b2) < lo(b3):
        return -1, rec
    if lo(b2) > hi(b3):
        return 1, rec
    return 0, rec


def krawczyk_codim2(fam, T2, T3, Bx, S):
    """4-D Krawczyk for {G0'(t_i) - B G0(t_i)^2 = 0, G0''(t_i) - 2 B G0(t_i) G0'(t_i) = 0, i = 2, 3}
    in the unknowns (t_2, t_3, B, s) (s = w_2/eta is a variable here, not a parameter)."""
    def pieces(t, s):
        base, sl = fam.parts(t, 3)
        g = jadd(base, jscale(sl, s))
        return ([g[0], g[1], 2 * g[2], 6 * g[3]], [sl[0], sl[1], 2 * sl[2]])

    def H(x):
        t2, t3, B, s = x
        out = []
        for t in (t2, t3):
            g, _ = pieces(t, s)
            out += [g[1] - B * g[0] * g[0], g[2] - 2 * B * g[0] * g[1]]
        return out

    def J(x):
        t2, t3, B, s = x
        rows = []
        for idx, t in ((0, t2), (1, t3)):
            g, sl = pieces(t, s)
            dt1 = g[2] - 2 * B * g[0] * g[1]
            dt2 = g[3] - 2 * B * (g[1] * g[1] + g[0] * g[2])
            ds1 = sl[1] - 2 * B * g[0] * sl[0]
            ds2 = sl[2] - 2 * B * (sl[0] * g[1] + g[0] * sl[1])
            r1 = [I(0), I(0), -g[0] * g[0], ds1]
            r2 = [I(0), I(0), -2 * g[0] * g[1], ds2]
            r1[idx], r2[idx] = dt1, dt2
            rows += [r1, r2]
        return rows

    X0 = [T2, T3, Bx, S]
    X0 = [sym(mid(x), (hi(x) - lo(x)) * mp.mpf("0.75") + mp.mpf("1e-26") * (1 + abs(mid(x)))) for x in X0]
    ok, X, nit = krawczyk(H, J, X0)
    return ok, X, nit


def enclose_switch(fam, q_lo: str, q_hi: str, n_bisect=44):
    """exactly one q* in (q_lo, q_hi) with b_2(q*) = b_3(q*): sign change + monotonicity."""
    out = {"bracket_initial": [q_lo, q_hi]}
    # monotonicity of D = b_2 - b_3 on the whole initial bracket (envelope derivatives, thick box)
    thick = certify_P(fam, fam.param_box(q_lo, q_hi), [q_lo, q_hi])
    if not thick["certified"]:
        out["fail"] = "thick certificate on the initial bracket failed: " + thick.get("fail", "")
        return out
    dD = thick["_iv"]["db"][2] - thick["_iv"]["db"][3]           # d(b_2 - b_3)/ds, ds/dq > 0
    out["dD_ds_on_bracket"] = to_list(dD)
    out["D_strictly_monotone_on_bracket"] = bool(pos(dD) or neg(dD))
    out["left_endpoint_certified_on_bracket"] = bool(hi(thick["_iv"]["b"][2]) < lo(thick["_iv"]["r0tau"])
                                                    and hi(thick["_iv"]["b"][3]) < lo(thick["_iv"]["r0tau"]))
    a, b = mp.mpf(q_lo), mp.mpf(q_hi)
    sa, ra = switch_sign(fam, a)
    sb, rb = switch_sign(fam, b)
    out["sign_D_at_ends"] = [sa, sb]
    # re-audit (minor): evidence for the end signs, replayable with exact rational comparisons: the exact
    # (dyadic) end points used, the exact q-box of the thick certificate, and the b_2, b_3 enclosures there
    Uq = iv.mpf([q_lo, q_hi])
    out["bracket_q_box_exact"] = [_exact(lo(Uq)), _exact(hi(Uq))]
    out["sign_evidence_at_ends"] = {
        name: ({"q_exact": _exact(x), "certified": True, "b2": to_list(r["_iv"]["b"][2]),
                "b3": to_list(r["_iv"]["b"][3])} if r.get("certified") else
               {"q_exact": _exact(x), "certified": False, "fail": r.get("fail")})
        for name, x, r in (("left", a, ra), ("right", b, rb))}
    if not (sa is not None and sb is not None and sa * sb < 0):
        out["fail"] = "no certified sign change of b_2 - b_3 on the initial bracket"
        return out
    steps = 0
    for _ in range(n_bisect):
        c = (a + b) / 2
        sc, _ = switch_sign(fam, c)
        if sc is None or sc == 0:
            break
        if sc == sa:
            a = c
        else:
            b = c
        steps += 1
    out["bisection_steps"] = steps
    fin = certify_P(fam, fam.param_box(_exact(a), _exact(b)), [_exact(a), _exact(b)])
    if not fin["certified"]:
        out["fail"] = "thick certificate on the final bracket failed"
        return out
    b2, b3 = fin["_iv"]["b"][2], fin["_iv"]["b"][3]
    Bs = iv.mpf([max(lo(b2), lo(b3)), min(hi(b2), hi(b3))])
    if lo(Bs) > hi(Bs):
        out["fail"] = "b_2 and b_3 enclosures disjoint on the final bracket (internal error)"
        return out
    S = fam.param_box(_exact(a), _exact(b))
    qs = to_list(iv.mpf([a, b]))
    from decimal import Decimal
    out.update({
        "q_star": qs,
        # A3 audit (minor): the bisection bracket [a, b] (dyadic, exact) is the isolating bracket; the printed
        # outward pair is wider, and its width is the exact decimal difference of the two printed strings
        "internal_isolating_bracket": [_exact(a), _exact(b)],
        "internal_isolating_bracket_width_upper": up(b - a),
        "printed_q_star_width": format(Decimal(qs[1]) - Decimal(qs[0]), "f"),
        "w_star": {"w1": to_list(1 - I(ETA_SWITCH)), "w2": to_list(I(ETA_SWITCH) * S),
                   "w3": to_list(I(ETA_SWITCH) * (1 - S))},
        "B_star": to_list(Bs),
        "t_f2_star": fin["flanks"][0]["t_f"], "t_f3_star": fin["flanks"][1]["t_f"],
        "a_f2_lower": fin["flanks"][0]["a_f_lower"], "a_f3_lower": fin["flanks"][1]["a_f_lower"],
        "r0_tau": fin["r0_tau"],
        "unique_in_bracket": bool(out["D_strictly_monotone_on_bracket"]),
    })
    # independent cross-check: 4-D Krawczyk on (t_2, t_3, B, s)
    ok, X, nit = krawczyk_codim2(fam, fin["_iv"]["tf"][2], fin["_iv"]["tf"][3], Bs, S)
    q_of_s = (lambda Sx: Sx / (1 - Sx))
    # A3 audit (minor): identification of the 4-D zero with the switch.  If X_s lies inside S (the final
    # bracket, where the thick certificate proves (D3) for every s) and X_t2, X_t3 lie inside the (D3) search
    # intervals of flanks 2 and 3 on that bracket, then any zero (t2, t3, B, s) in X has N(t_i; s) = 0 there,
    # so t_i = t_{f,i}(s) and B = b_2(s) = b_3(s), i.e. s = s* (unique by monotonicity): the 4-D zero IS the switch.
    sr2, sr3 = fin["_iv"]["search"][2], fin["_iv"]["search"][3]
    ident = {"X_s_inside_S": bool(ok and lo(X[3]) >= lo(S) and hi(X[3]) <= hi(S)),
             "X_t2_inside_D3_search_flank2": bool(ok and lo(X[0]) >= sr2[0] and hi(X[0]) <= sr2[1]),
             "X_t3_inside_D3_search_flank3": bool(ok and lo(X[1]) >= sr3[0] and hi(X[1]) <= sr3[1]),
             "D3_search_flank2": [down(sr2[0]), up(sr2[1])], "D3_search_flank3": [down(sr3[0]), up(sr3[1])]}
    ident["identified_with_switch"] = bool(ok and all(v for k, v in ident.items() if k.startswith("X_")))
    # re-audit (minor): exact dyadic domains, so that the containments above can be replayed independently
    ident["exact_domains"] = {
        "X": {nm: [_exact(lo(x)), _exact(hi(x))] for nm, x in zip(("t2", "t3", "B", "s"), X)},
        "S": [_exact(lo(S)), _exact(hi(S))],
        "D3_search_flank2": [_exact(sr2[0]), _exact(sr2[1])],
        "D3_search_flank3": [_exact(sr3[0]), _exact(sr3[1])]}
    out["krawczyk_4d"] = {"ok": bool(ok), "inflations": nit,
                          "t2": to_list(X[0]), "t3": to_list(X[1]), "B": to_list(X[2]), "s": to_list(X[3]),
                          "q": to_list(q_of_s(X[3])), "identification": ident,
                          "consistent_with_bisection": bool(ident["identified_with_switch"]
                                                            and not (hi(X[2]) < lo(Bs) or lo(X[2]) > hi(Bs))
                                                            and not (hi(X[3]) < lo(S) or lo(X[3]) > hi(S))
                                                            and not (hi(X[0]) < lo(fin["_iv"]["tf"][2])
                                                                     or lo(X[0]) > hi(fin["_iv"]["tf"][2]))
                                                            and not (hi(X[1]) < lo(fin["_iv"]["tf"][3])
                                                                     or lo(X[1]) > hi(fin["_iv"]["tf"][3])))}
    return out


def _exact(x):
    """exact decimal string of an mp (binary) number."""
    return th8.exact(x)


# ----------------------------------------------------------------------------
# max-min path of Prop. 3(iii) at m = 2 (A = 1, v_j = 4 e^{-t_j}); y = -ln(1 - 2p)
V1 = 4 * iv.exp(I("-1.0"))
V2 = 4 * iv.exp(I("-2.5"))
LN2 = iv.log(I(2))


def path_at(Y):
    """(w_2*, B) enclosures on the max-min path at y in Y (interval)."""
    X = iv.exp(-Y)
    l1 = LN2 - iv.log(1 + X)
    l2 = iv.log((1 + X) / 2) + Y
    B = V1 * l1 + V2 * l2
    return V2 * l2 / B, B


W0_PATH = V2 / (V1 + V2)          # w_2* as B -> 0


def B_path(w: str, y_hi=mp.mpf(4000), iters=400):
    """enclosure [lower, upper] of the budget at which w_2* = w (w a decimal string); w_2* and B
    are strictly increasing in y, so a verified bracket [ya, yb] of the crossing gives
    B_path(w) in [lo B(ya), hi B(yb)].  Returns None if w <= w_2*(0+) (no path point below w)."""
    W = I(w)
    if hi(W) <= lo(W0_PATH):
        return None
    ya, yb = mp.mpf(0), y_hi
    if not lo(path_at(I(yb))[0]) > hi(W):
        raise RuntimeError("B_path: y_hi too small")
    for _ in range(iters):
        yc = (ya + yb) / 2
        wc, _ = path_at(I(yc))
        if hi(wc) < lo(W):
            ya = yc
        elif lo(wc) > hi(W):
            yb = yc
        else:
            break
    Ba = path_at(I(ya))[1] if ya > 0 else I(0)
    Bb = path_at(I(yb))[1]
    return [lo(Ba), hi(Bb)]


def path_check(leaves):
    """walk the certified F2W boxes upward from the box containing w_2*(0+); on each box the path
    must lie strictly below min(r0(tau), b_2).  Returns the certified reach."""
    # A3 audit (major/minor): the start box must CONTAIN the whole enclosure of w_2*(0+) (exact comparison of
    # the dyadic endpoints with the decimal box ends); every comparison below is exact (Fraction)
    w0_lo, w0_hi = Fraction(_exact(lo(W0_PATH))), Fraction(_exact(hi(W0_PATH)))
    rows, reach, started = [], None, False
    prev_hi = None
    for r in leaves:
        a, b = r["box"]
        if not started:
            if not (Fraction(a) <= w0_lo and w0_hi < Fraction(b)):
                if Fraction(a) <= w0_hi and w0_lo < Fraction(b):
                    raise RuntimeError("path_check: w_2*(0+) enclosure straddles a box end (refine the grid)")
                continue
            started = True
        if not r["certified"] or (prev_hi is not None and a != prev_hi):
            break
        bp = B_path(b)
        # lower bound of B_top^det on the box: the smaller of two certified lower-bound strings (exact)
        top = min(r["r0_tau"][0], r["flanks"][0]["b"][0], key=Fraction)
        ok = bp is not None and Fraction(_exact(bp[1])) < Fraction(top)
        rows.append({"w_box": [a, b], "B_path_at_hi_w_upper": up(bp[1]) if bp else None,
                     "B_top_det_lower": top, "below_fold": bool(ok)})
        if not ok:
            break
        reach = (b, bp)
        prev_hi = b
    return rows, reach


# ----------------------------------------------------------------------------
# located (not certified) G0 saddle-nodes = ends of the (D1) region; float diagnostics
def locate_edge(fam, t0, p0):
    """mp Newton (100 bits, not an interval certificate) on G0' = G0'' = 0 in (t, p)."""
    t, p = mp.mpf(t0), mp.mpf(p0)
    for _ in range(80):
        base, sl = fam.parts(I(t), 3)
        g = jadd(base, jscale(sl, I(p)))
        f1, f2 = mid(g[1]), mid(2 * g[2])
        j11, j12, j21, j22 = mid(2 * g[2]), mid(sl[1]), mid(6 * g[3]), mid(2 * sl[2])
        det = j11 * j22 - j12 * j21
        dt, dp = (f1 * j22 - f2 * j12) / det, (j11 * f2 - j21 * f1) / det
        t, p = t - dt, p - dp
        if abs(dt) + abs(dp) < mp.mpf("1e-26"):
            break
    user = p if fam.pname == "w2" else p / (1 - p)
    return {"t": float(t), "p_affine": float(p), fam.pname: mp.nstr(user, 15),
            "residual": [float(f1), float(f2)]}


EDGE_GUESS = {"F2W": [("late peak merges with the valley before it", 2.27, 0.0251),
                      ("first peak merges with the valley after it", 1.06, 0.9749)],
              "F3S": [("middle peak merges with the valley after it", 1.695, 0.3365 / 1.3365),
                      ("last peak merges with the valley before it", 2.506, 2.8643 / 3.8643)],
              "F3A": [("middle peak merges with the valley after it", 1.709, 0.3491 / 1.3491),
                      ("last peak merges with the valley before it", 2.506, 2.8643 / 3.8643)]}


def float_crosscheck(fk, leaves, n=4):
    """fb_finite_width_det.analyse (float64 grid) at a few box midpoints vs the certified enclosures."""
    import fb_finite_width_det as fw
    fam = FAMILIES[fk]()
    cert = [r for r in leaves if r["certified"]]
    rows = []
    for r in [cert[int(k * (len(cert) - 1) / (n - 1))] for k in range(n)]:
        a, b = (float(x) for x in r["box"])
        u = 0.5 * (a + b)
        if fam.pname == "w2":
            w = [1 - u, u]
        else:
            eta = 1e-3 if fk == "F3S" else 2.0 / 3.0
            s = u / (1 + u)
            w = [1 - eta, eta * s, eta * (1 - s)]
        fa = fw.analyse(fam.m, float(fam.rho_str), w=w, n=600001)
        ok = []
        for f in r["flanks"]:
            v = fa["candidates"].get(f"flank_{f['j']}")
            ok.append(v is not None and float(f["b"][0]) <= v <= float(f["b"][1]))
        rows.append({"param": u, "float_candidates": fa["candidates"], "float_binding": fa["binding"],
                     "inside_certified_box_enclosures": all(ok), "certified_binding": r["binding"]})
    return rows


def summarise(fk, leaves):
    from decimal import Decimal
    fam = FAMILIES[fk]()
    cov = covered_intervals(leaves)
    lo_all, hi_all = leaves[0]["box"][0], leaves[-1]["box"][1]
    if fam.pname == "w2":
        meas = lambda a, b: float(Decimal(b) - Decimal(a))          # noqa: E731
    else:
        meas = lambda a, b: math.log(float(b)) - math.log(float(a))   # noqa: E731
    cert = [r for r in leaves if r["certified"]]
    excl = []
    for r in leaves:
        if not r["certified"]:
            a, b = r["box"]
            if excl and excl[-1][1] == a:
                excl[-1][1] = b
            else:
                excl.append([a, b])
    runs = []
    for r in cert:
        k = r["binding"] or "undetermined"
        if runs and runs[-1]["binding"] == k and runs[-1]["range"][1] == r["box"][0]:
            runs[-1]["range"][1] = r["box"][1]
            runs[-1]["n_boxes"] += 1
        else:
            runs.append({"binding": k, "range": list(r["box"]), "n_boxes": 1})
    bh = {}
    for f in (2, 3)[:fam.m - 1]:
        vals = [x for r in cert for x in r["flanks"][f - 2]["b"]]
        bh[f"b{f}_hull"] = [min(vals, key=lambda s: mp.mpf(s)), max(vals, key=lambda s: mp.mpf(s))]
    gaps = []
    for r in cert:
        # A3 audit (minor): ratio of certified one-sided bounds evaluated in iv (the decimal strings are
        # entered as outward enclosures), lower end kept
        bmax = max((f["b"][1] for f in r["flanks"]), key=lambda s: Fraction(s))
        gaps.append(lo(iv.mpf(r["r0_tau"][0]) / iv.mpf(bmax)))
    return {
        "family": fam.note, "parameter": fam.pname, "range": [lo_all, hi_all],
        "n_leaf_boxes": len(leaves), "n_certified": len(cert),
        "coverage_fraction": sum(meas(a, b) for a, b in cov) / meas(lo_all, hi_all),
        "coverage_measure": "linear in w2" if fam.pname == "w2" else "logarithmic in q",
        "certified_intervals": cov, "excluded_intervals": excl,
        "failures": [{"box": r["box"], "fail": r.get("fail")} for r in leaves if not r["certified"]],
        "binding_runs": runs,
        "all_krawczyk_ok": all(f["krawczyk_ok"] and f["krawczyk_mid_ok"] for r in cert for f in r["flanks"]),
        "a_f_lower_bound_min": down(min(mp.mpf(f["a_f_lower"]) for r in cert for f in r["flanks"])),
        "r0_tau_over_max_b_lower_bound_min": down(min(gaps)),
        **bh,
        "max_box_runtime_s": max(r["runtime_s"] for r in leaves),
    }


def compact_table(leaves):
    rows = []
    for r in leaves:
        row = {"box": r["box"], "certified": r["certified"]}
        if r["certified"]:
            row.update({"binding": r["binding"], "r0_tau": r["r0_tau"],
                        "b": {str(f["j"]): f["b"] for f in r["flanks"]},
                        "t_f": {str(f["j"]): f["t_f"] for f in r["flanks"]},
                        "b_mid": r.get("_b_mid")})
        rows.append(row)
    return rows


def falsification(fk, leaves, fracs=("0.25", "0.5", "0.75")):
    """100-bit point solves (NOT interval; Newton on N = 0 from the box's t_f enclosure) at fixed
    fractions of every certified box: the fold value must lie in the box's b_j enclosure and the fold
    time in its t_f enclosure."""
    from decimal import Decimal
    fam = FAMILIES[fk]()
    n, bad = 0, []
    for r in leaves:
        if not r["certified"]:
            continue
        a, b = Decimal(r["box"][0]), Decimal(r["box"][1])
        for fr in fracs:
            u = a + (b - a) * Decimal(fr)
            P = fam.param_box(str(u), str(u))
            pm = I(mid(P))
            for f in r["flanks"]:
                t0 = (mp.mpf(f["t_f"][0]) + mp.mpf(f["t_f"][1])) / 2
                t, bv = newton_fold(fam, pm, t0)
                n += 1
                if not (mp.mpf(f["b"][0]) <= bv <= mp.mpf(f["b"][1])
                        and mp.mpf(f["t_f"][0]) <= t <= mp.mpf(f["t_f"][1])):
                    bad.append({"box": r["box"], "u": str(u), "j": f["j"], "t": float(t), "b": float(bv)})
    return {"n_point_solves": n, "n_violations": len(bad), "violations": bad[:20]}


NOT_A_CUSP = (
    "The critical set of f_det in the (t, B) plane is {r0(t) = B}; its folds are the critical points of r0 "
    "at positive value, and a cusp would need r0' = r0'' = 0 at r0 = B > 0.  ON THE CERTIFIED PARAMETER SETS "
    "(A3 audit: the statement is restricted to them): (D2) excludes a critical point of r0 on the first rising "
    "flank, (D3) gives on each later rising flank exactly one critical point of r0, a nondegenerate maximum "
    "(a_f = -r0'' > 0 certified), and r0 < 0 on the falling flanks; so no cusp at B > 0 there.  At the "
    "codimension-2 point the two folds are distinct (t_f2 < t_f3, disjoint enclosures in different flanks) and "
    "each is nondegenerate, and b_2 - b_3 changes sign transversally (d(b_2 - b_3)/dq > 0 certified): a "
    "transversal crossing of two fold curves (a corner of the bifurcation set B_top^det = min(b_2, b_3)), not a "
    "cusp.  Outside the certified sets (where (D1)-(D3) may fail, e.g. new critical points of r0 appear) "
    "nothing is claimed for unequal weights; for equal weights the analytic argument of CV-1 "
    "(fold_curve_m2.json#not_a_cusp) excludes a positive-level cusp for every width.")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="coarse grids (smoke test)")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.quick:
        grids = {"F2W": lin_edges("0.025", "0.975", "0.05"), "F3S": log_edges(0.3366, 2.864, 20),
                 "F3A": log_edges(0.3492, 2.864, 20)}
    else:
        grids = {"F2W": lin_edges("0.025", "0.975", "0.0025"), "F3S": log_edges(0.3366, 2.864, 300),
                 "F3A": log_edges(0.3492, 2.864, 300)}
    leaves, summ = {}, {}
    for fk, edges in grids.items():
        ts = time.time()
        leaves[fk] = scan(fk, edges, args.workers)
        summ[fk] = summarise(fk, leaves[fk])
        summ[fk]["scan_runtime_s"] = round(time.time() - ts, 1)
        print(fk, "certified", summ[fk]["n_certified"], "/", summ[fk]["n_leaf_boxes"],
              "coverage", round(summ[fk]["coverage_fraction"], 4), "runs",
              [(x["binding"], x["range"]) for x in summ[fk]["binding_runs"]], flush=True)
    # ---- codimension-2 switch (F3S)
    fam = fam_F3S()
    sw = None
    for qa, qb in (("1.52", "1.57"), ("1.535", "1.555"), ("1.54", "1.55")):
        sw = enclose_switch(fam, qa, qb)
        if "fail" not in sw and sw["unique_in_bracket"]:
            break
    from decimal import Decimal
    und = [r["box"] for r in leaves["F3S"] if r["certified"] and r["binding"] is None]
    br = sw["bracket_initial"]
    sw["undetermined_scan_boxes"] = und
    sw["undetermined_boxes_inside_bracket"] = all(Decimal(br[0]) <= Decimal(a) and Decimal(b) <= Decimal(br[1])
                                                  for a, b in und)
    left = [r["binding"] for r in leaves["F3S"] if r["certified"] and Decimal(r["box"][1]) <= Decimal(br[0])]
    right = [r["binding"] for r in leaves["F3S"] if r["certified"] and Decimal(r["box"][0]) >= Decimal(br[1])]
    sw["binding_left_of_bracket"] = sorted(set(map(str, left)))
    sw["binding_right_of_bracket"] = sorted(set(map(str, right)))
    sw["exactly_one_switch_in_certified_set"] = bool(
        "fail" not in sw and sw["unique_in_bracket"] and sw["undetermined_boxes_inside_bracket"]
        and set(left) == {"flank_2"} and set(right) == {"flank_3"})
    print("switch", {k: sw.get(k) for k in ("q_star", "B_star", "unique_in_bracket",
                                            "exactly_one_switch_in_certified_set")}, flush=True)
    # ---- F3A: natural family, no switch
    a_runs = summ["F3A"]["binding_runs"]
    no_switch = {"all_certified_boxes_last_flank_binds": all(x["binding"] == "flank_3" for x in a_runs),
                 # A3 audit (minor): ratio of certified one-sided bounds in iv, lower end kept
                 "min_b2_over_b3_lower_bound": down(min(lo(iv.mpf(r["flanks"][0]["b"][0])
                                                           / iv.mpf(r["flanks"][1]["b"][1]))
                                                        for r in leaves["F3A"] if r["certified"]))}
    # ---- max-min path (F2W)
    rows, reach = path_check(leaves["F2W"])
    # equal-weight anchor: every certified F2W box with w2 = 1/2 in it must contain the TH8 value
    # (TH8_certificate/certificate.json#cases.m2_rho0.3.B_top_det, equal weights, same geometry)
    th8_B = json.loads((HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
                      / "TH8_certificate" / "certificate.json").read_text())["cases"]["m2_rho0.3"]["B_top_det"]
    eqw = []
    for r in leaves["F2W"]:
        # exact decimal comparisons (A3 audit, minor)
        if r["certified"] and Fraction(r["box"][0]) <= Fraction(1, 2) <= Fraction(r["box"][1]):
            e = r["flanks"][0]["b"]
            eqw.append({"box": r["box"], "b2": e, "binding": r["binding"],
                        "contains_TH8_B_top_det": bool(Fraction(e[0]) <= Fraction(th8_B[0])
                                                       and Fraction(th8_B[1]) <= Fraction(e[1]))})
    path = {
        "design_law": "Prop. 3(iii), A = W^(d-1) = 1, v_j = 4 exp(-t_j); w_2*(B) strictly increasing from "
                      "v2/(v1+v2) (B -> 0) to 1 (B -> infinity)",
        "w2_star_B_to_0": to_list(W0_PATH),
        "rows": rows,
        "certified_reach": ({"w2_upper": reach[0], "B_path_lower": down(reach[1][0]), "B_path_upper": up(reach[1][1])}
                            if reach else None),
        "B_path_at_w2_0.5": [down(x) for x in B_path("0.5")][:1] + [up(B_path("0.5")[1])],
        "N1_check_w_star_B8": None,
    }
    # N1 check: w*(B = 8) = (0.1275, 0.8725) (n1_allocation_design.json#cells[key=B8_maxmin, m=2].design.w)
    y8a, y8b = mp.mpf(0), mp.mpf(100)
    for _ in range(200):
        yc = (y8a + y8b) / 2
        if hi(path_at(I(yc))[1]) < 8:
            y8a = yc
        elif lo(path_at(I(yc))[1]) > 8:
            y8b = yc
        else:
            break
    path["N1_check_w_star_B8"] = to_list(iv.mpf([lo(path_at(I(y8a))[0]), hi(path_at(I(y8b))[0])]))
    # max-min fold at the budget where equal weights lose the late peak (B = B_top^det(1/2), certified in
    # TH8_certificate/certificate.json#cases.m2_rho0.3.B_top_det): w_2*(B) enclosed by bisection in y, then the
    # certified F2W box(es) containing it give b_2 there.
    Beq = iv.mpf(["8.22466474559830162", "8.22466474559830179"])
    ya, yb = mp.mpf(0), mp.mpf(100)
    for _ in range(300):
        yc = (ya + yb) / 2
        Bc = path_at(I(yc))[1]
        if hi(Bc) < lo(Beq):
            ya = yc
        elif lo(Bc) > hi(Beq):
            yb = yc
        else:
            break
    Wq = iv.mpf([lo(path_at(I(ya))[0]), hi(path_at(I(yb))[0])])
    # A3 audit (major): select the certified boxes meeting Wq with EXACT comparisons and CHECK that their union
    # covers Wq (contiguous boxes, first starts at or below lo Wq, last ends at or above hi Wq)
    wq_lo, wq_hi = Fraction(_exact(lo(Wq))), Fraction(_exact(hi(Wq)))
    cover = [r for r in leaves["F2W"] if r["certified"] and Fraction(r["box"][0]) <= wq_hi
             and Fraction(r["box"][1]) >= wq_lo]
    covered = bool(cover) and Fraction(cover[0]["box"][0]) <= wq_lo and Fraction(cover[-1]["box"][1]) >= wq_hi \
        and all(cover[i]["box"][1] == cover[i + 1]["box"][0] for i in range(len(cover) - 1))
    if not covered:
        raise RuntimeError("at_equal_weight_fold_budget: Wq not covered by the union of certified boxes")
    b2lo = min((r["flanks"][0]["b"][0] for r in cover), key=Fraction)
    b2hi = max((r["flanks"][0]["b"][1] for r in cover), key=Fraction)
    path["at_equal_weight_fold_budget"] = {
        "B": to_list(Beq), "w2_star": to_list(Wq), "covering_boxes": [r["box"] for r in cover],
        "Wq_covered_by_union_of_certified_boxes": covered,
        "b2_at_w2_star": [b2lo, b2hi],
        "ratio_b2_over_B_lower": down(lo(iv.mpf(b2lo) / iv.mpf(to_list(Beq)[1])))}
    # ---- located (D1) edges
    edges = {}
    for fk, lst in EDGE_GUESS.items():
        fam_ = FAMILIES[fk]()
        edges[fk] = [dict(mechanism=txt, **locate_edge(fam_, tg, pg)) for txt, tg, pg in lst]
    # B_path at the upper F2W edge (where the path leaves the (D1) region; located, not certified)
    wexit = edges["F2W"][1]["w2"]
    wexit_s = format(Decimal(wexit).quantize(Decimal("1e-12")), "f")
    bp = B_path(wexit_s)
    path["path_leaves_D1_at"] = {"w2_located": wexit, "B_path": [float(bp[0]), float(bp[1])],
                                 "note": "G0 loses its first maximum (w1* too small); located by 100-bit "
                                         "Newton, not certified"}
    # coverage relative to the located (D1) interval of each family
    for fk in summ:
        e_lo, e_hi = (mp.mpf(edges[fk][0][FAMILIES[fk]().pname]), mp.mpf(edges[fk][1][FAMILIES[fk]().pname]))
        f = (lambda x: x) if fk == "F2W" else mp.log
        cov = sum(f(min(mp.mpf(b), e_hi)) - f(max(mp.mpf(a), e_lo))
                  for a, b in summ[fk]["certified_intervals"] if mp.mpf(b) > e_lo and mp.mpf(a) < e_hi)
        summ[fk]["located_D1_interval"] = [edges[fk][0][FAMILIES[fk]().pname], edges[fk][1][FAMILIES[fk]().pname]]
        summ[fk]["coverage_fraction_of_located_D1_interval"] = float(cov / (f(e_hi) - f(e_lo)))
    # ---- falsification: 100-bit point solves inside every certified box
    falsify = {fk: falsification(fk, leaves[fk]) for fk in leaves}
    print("falsification", {k: (v["n_point_solves"], v["n_violations"]) for k, v in falsify.items()}, flush=True)
    # ---- float cross-checks
    fc = {fk: float_crosscheck(fk, leaves[fk]) for fk in leaves}
    res = {
        "script": "code/fb_v2_codim2_switch.py", "item": "V2 item 3, CV-2 (m = 3 fold switch, (w, B) panel)",
        "deterministic": True, "seeds": None,
        "arithmetic": f"mpmath {mp.__class__.__module__.split('.')[0]} iv, outward rounding, {PREC}-bit mantissa",
        "serialization": "intervals as [lower, upper] decimal strings rounded outward (18 significant digits)",
        "certified_flag": "certified = (D1)-(D3) and the fold enclosures hold for EVERY parameter in the box; "
                          "(D4) and the binding flank are separate fields (D4_on_box, binding); theorem-level "
                          "claims are checked by code/fb_v2_cert_validator.py (exact rational comparisons)",
        "geometry": {"gamma": 1, "D0": 1, "z0": 4, "zbar": 0, "W": 1, "d": 2, "window": [0.5, 3.5]},
        "families": {fk: FAMILIES[fk]().note for fk in FAMILIES},
        "method": __doc__.split("Per parameter box")[1].split("Run (from")[0].strip(),
        "summaries": summ,
        "codim2_switch_F3S": sw,
        "no_switch_F3A": no_switch,
        "maxmin_path_F2W": path,
        "equal_weight_anchor_F2W": {"TH8_B_top_det_m2_rho0.3": th8_B, "boxes_containing_w2=1/2": eqw,
                                    "all_consistent": bool(eqw) and all(x["contains_TH8_B_top_det"] for x in eqw)},
        "D1_edges_located": edges,
        "not_a_cusp": NOT_A_CUSP,
        "float_crosscheck_fb_finite_width_det": fc,
        "falsification_test_point_solves": falsify,
        "tables": {fk: compact_table(v) for fk, v in leaves.items()},
        "box_records_file": "artifacts/data/exact_m_fixed_budget/V2_bifurcation/codim2_switch_boxes.json",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    import hashlib
    import platform
    import mpmath
    res["provenance"] = {
        "source_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                          for f in ("fb_v2_codim2_switch.py", "fb_v2_fold_curve_certificate.py",
                                    "fb_th8_interval_certificate.py")},
        "mpmath_version": mpmath.__version__, "python_version": platform.python_version(),
        "acceptance": "code/fb_v2_cert_validator.py --only cv2 (exact rational checks of every claim of "
                      "Prop. v2:prop-alloc)"}
    res["codim2_switch_F3S"]["bracket_is_primary"] = bool(sw.get("bracket_initial") == ["1.52", "1.57"])
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    OUT_BOXES.write_text(json.dumps({"script": res["script"], **{fk: v for fk, v in leaves.items()}}, indent=0))
    print("wrote", OUT, "runtime", res["runtime_seconds"])


if __name__ == "__main__":
    main()
