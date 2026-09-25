#!/usr/bin/env python3
"""V2 item 3 (CV-1): interval certificate of the m = 2 fold curve B_top^det(rho_s)
of the fixed-width limit law, with the stripe width rho_s an INTERVAL PARAMETER on a
subdivision of [0.15, 0.40], together with (D1)-(D4) on every rho_s-box.

Extends code/fb_th8_interval_certificate.py (point widths, t-coordinates; its jet
algebra and outward serialisation are imported) to a parametric certificate.
Deterministic, no random numbers, no seeds.  mpmath.iv, outward rounding, 100 bits.

Model (anchor geometry of Prop. 4: gamma = D0 = 1, z0 = 4, zbar = 0, W = 1, d = 2,
I = [tau, T] = [0.5, 3.5], targets t_1 = 1, t_2 = 2.5, equal weights):
    mu(t) = 4 e^{-t},  c_j = mu(t_j),  G0 = sum_j (1/2) phi_rho(mu - c_j),
    r0 = G0'/G0^2,  B_top^det = min{r0(tau), b_2}.
Exact change of coordinates (no approximation), cbar = (c1+c2)/2, Delta = c1 - c2:
    a = Delta/(2 rho),  k = cbar/rho = kappa a,  kappa = (c1+c2)/(c1-c2),
    u = (mu(t) - cbar)/rho   (strictly decreasing in t, du/dt = -(k+u), k+u = mu/rho > 0),
    G0 = exp(phi),  phi = -log(sqrt(2 pi) rho) - (u^2+a^2)/2 + log cosh(a u),
    s(u;a) = -u + a tanh(a u),   s_u = -1 + a^2 sech^2(a u),
    phi' = -s (k+u),  r0 = phi' e^{-phi},  r0' = (k+u) Phi e^{-phi},
    Phi(u;a) = (k+u)(s_u - s^2) + s.
Hence: critical points of G0 <-> zeros of s (max <-> s_u < 0, min <-> s_u > 0,
G0'' = G0 (k+u)^2 s_u there); critical points of r0 <-> zeros of Phi, and
r0''(t_f) = -(k+u)^2 Phi_u e^{-phi} (nondegenerate maximum <-> Phi_u > 0);
    log r0 = log(-s) + log(k+u) + log(sqrt(2 pi)) + log(Delta/(2a)) + (u^2+a^2)/2 - log cosh(a u).
The rho-box [rho_lo, rho_hi] maps exactly onto the a-box [Delta/(2 rho_hi), Delta/(2 rho_lo)];
every enclosure below holds for every (u, a) in the stated boxes, so for every rho_s in the box.

Method (per rho-box; A = a-box):
  * Taylor jets in u with interval coefficients (tanh by its Riccati recurrence), valid
    for all a in A; enclosure of F^{(r)} on a cell J by the order-2 Taylor form in u.
  * Parametric zero counting: cells are bisected until the value enclosure (over J x A)
    excludes 0 or the u-derivative enclosure excludes 0; consecutive strictly monotone
    cells of the same direction are merged into runs; on a run F(., a) is strictly
    monotone for every a in A, so it has exactly one zero iff the (definite) signs at the
    run ends differ.  Root enclosures are then shrunk by bisection on definite signs.
  * (D1) zeros of s on a superset of the u-image of I, three runs (dec, inc, dec in u),
    all inside the image of I for every a; s(u_tau) < 0 < s(u_T) (G0'(tau) > 0 > G0'(T)).
  * (D2) Phi < 0 on [u*(a), u_tau(a)] (the first rising flank [tau, hat t_1]).
  * (D3) exactly one zero of Phi on the flank superset, an increasing run (Phi_u > 0,
    r0'' < 0), with Phi < 0 on the enclosure of -u* (hat t_2) and Phi > 0 on that of 0
    (check t_1): r0 has exactly one critical point in (check t_1, hat t_2), a
    nondegenerate maximum.
  * Fold: parametric 2-D Krawczyk operator on H(u, beta; a) = (log r0 - beta, Phi) = 0,
    beta = log B, i.e. the augmented fold system {r0 = B, r0' = 0} in the coordinates
    (u, log B); K(X, A) = x~ - Y H(x~, A) + (I - Y J(X, A))(X - x~) in int X proves, for
    every a in A, a unique zero in X.  Cross-check: 1-D interval Newton on Phi(.; A).
  * Thin tube: beta(a) in beta(a_m) + d_a log r0(U_f, A) (a - a_m), because
    d_u log r0 = 0 on the fold (envelope identity), beta(a_m) certified with thin a_m.
  * (D4) r0(tau) > b_2 on the box (the rising flank binds).

Run (from code/):
    python fb_v2_fold_curve_certificate.py [--workers 3]
Output: ../artifacts/data/exact_m_fixed_budget/V2_bifurcation/fold_curve_m2.json
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
                                         jmul, jadd, jsub, jderiv, intersect, taylor_encl)
from mpmath import iv, mp  # noqa: E402

PREC = th8.PREC
OUT = (HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "V2_bifurcation"
       / "fold_curve_m2.json")

RHO_RANGE = (Fraction(15, 100), Fraction(40, 100))   # required range
N_BASE = 250                                          # base boxes (width 0.001)
MAX_DEPTH = 4                                         # adaptive bisection depth of a failed box
EXT_RANGE = (Fraction(40, 100), Fraction(5715, 10000))  # extension towards rho_c (not required)
N_EXT = 343                                           # width 0.0005
ORDER = 4                                             # jet order in u
INIT_CELLS = 64
MIN_WIDTH = mp.mpf("1e-12")
MAX_CELLS = 6000                                      # cell budget per scan (box fails beyond)

# ----------------------------------------------------------------------------
# constants (interval enclosures)
C1 = 4 * iv.exp(I(-1))
C2 = 4 * iv.exp(I("-2.5"))
CB = (C1 + C2) / 2
DEL = C1 - C2
KAP = (C1 + C2) / (C1 - C2)
SQ2PI = iv.sqrt(2 * iv.pi)
LOG_SQ2PI = iv.log(SQ2PI)
MU_TAU = 4 * iv.exp(I("-0.5"))
MU_T = 4 * iv.exp(I("-3.5"))
LOG4 = iv.log(I(4))
LOG2 = iv.log(I(2))
RHO_C = DEL / 2                                       # terminal width (a = 1)


def frac_str(q: Fraction) -> str:
    """exact decimal string of a Fraction with a terminating expansion."""
    num, den = q.numerator, q.denominator
    k = 0
    while (10 ** k) % den:
        k += 1
        if k > 40:
            raise ValueError("non-terminating decimal")
    ip = num * 10 ** k // den
    s = str(ip).rjust(k + 1, "0")
    return (s[:-k] + "." + s[-k:]) if k else s


def a_box(r_lo: str, r_hi: str):
    """a-box enclosing {Delta/(2 rho): rho in [r_lo, r_hi]} (decimal endpoints, outward)."""
    P = iv.mpf([r_lo, r_hi])
    return DEL / (2 * P), P


# ----------------------------------------------------------------------------
# jets in u (coefficient c[k] encloses F^{(k)}(u)/k! for all u in U, a in A)
def jet_tanh(U, A, order):
    """jet of y = tanh(a u) in u: y0 = 1 - 2/(1 + e^{2 a u}), k y_k = a z_{k-1}, z = 1 - y^2."""
    y = [1 - 2 / (1 + iv.exp(2 * A * U))]
    for kk in range(1, order + 1):
        zk1 = (I(1) if kk == 1 else I(0)) - sum((y[i] * y[kk - 1 - i] for i in range(kk)), I(0))
        y.append(A * zk1 / kk)
    return y


def jet_s(U, A, order):
    y = jet_tanh(U, A, order)
    s = [A * c for c in y]
    s[0] = s[0] - U
    if order >= 1:
        s[1] = s[1] - 1
    return s


def jet_phi_fun(U, A, order):
    """jet of Phi = (k+u)(s_u - s^2) + s, k = kappa a (order `order`)."""
    s = jet_s(U, A, order + 1)
    su = jderiv(s, 1)
    s = s[:order + 1]
    ku = [KAP * A + U, I(1)] + [I(0)] * (order - 1)
    return jadd(jmul(ku, jsub(su, jmul(s, s))), s)


def logcosh_neg(X):
    """log cosh(x) for an interval X of NEGATIVE numbers (or any sign-definite X):
    f(y) = y + log(1 + e^{-2y}) - log 2 with y = |x| is increasing in y >= 0, so its
    range is [f(lo y), f(hi y)] (thin evaluations, outward)."""
    if hi(X) < 0:
        ya, yb = -hi(X), -lo(X)
    elif lo(X) > 0:
        ya, yb = lo(X), hi(X)
    else:
        raise ValueError("logcosh_neg: X must be sign-definite")

    def f(y):
        yI = I(y)
        return yI + iv.log(1 + iv.exp(-2 * yI)) - LOG2
    return iv.mpf([lo(f(ya)), hi(f(yb))])


def tanh_iv(X):
    return 1 - 2 / (1 + iv.exp(2 * X))


def s_encl(U, A):
    """s(u; a) on U x A: naive form intersected with the centred form in u
    s(u~; A) + s_u(U; A)(U - u~) (removes the -u + a tanh(a u) cancellation near u = 0)."""
    th = tanh_iv(A * U)
    naive = -U + A * th
    if hi(U) == lo(U):
        return naive
    m = I(mid(U))
    s_m = -m + A * tanh_iv(A * m)
    su = -1 + A * A * (1 - th * th)
    return intersect(naive, s_m + su * (U - m))


def log_r0(U, A):
    """log r0 at (u, a) in U x A, valid where s < 0 (rising flank)."""
    s = s_encl(U, A)
    if not neg(s):
        raise ValueError("log_r0: s not certified negative")
    return (iv.log(-s) + iv.log(KAP * A + U) + LOG_SQ2PI + iv.log(DEL / (2 * A))
            + (U * U + A * A) / 2 - logcosh_neg(A * U))


def dlogr0_du(U, A):
    """d/du log r0 = s_u/s + 1/(k+u) + u - a tanh(a u)."""
    th = tanh_iv(A * U)
    s = s_encl(U, A)
    su = -1 + A * A * (1 - th * th)
    return su / s + 1 / (KAP * A + U) + U - A * th


def dlogr0_da(U, A):
    """partial_a log r0 at fixed u: s_a/s + kappa/(k+u) - 1/a + a - u tanh(a u),
    s_a = tanh(a u) + a u sech^2(a u)."""
    th = tanh_iv(A * U)
    s = s_encl(U, A)
    sa = th + A * U * (1 - th * th)
    return sa / s + KAP / (KAP * A + U) - 1 / A + A - U * th


# ----------------------------------------------------------------------------
# parametric zero counting in u (all a in A at once)
class PFun:
    """F(u; a) given by a jet map (U, A, order) -> jet; A fixed (an interval)."""

    def __init__(self, jetmap, A, name):
        self.jetmap, self.A, self.name = jetmap, A, name
        self.n_jets = 0
        self._pt = {}

    def jet(self, U, order=ORDER):
        self.n_jets += 1
        return self.jetmap(U, self.A, order)

    def point(self, x):
        """enclosure of F(x; A) at the exactly representable point x."""
        key = mp.nstr(x, 45)
        if key not in self._pt:
            self._pt[key] = self.jet(I(x), 0)[0]
        return self._pt[key]

    def cell(self, x, y):
        """(value enclosure, u-derivative enclosure) of F on [x, y] x A."""
        J, m = I(x, y), (x + y) / 2
        jm, jJ = self.jet(I(m), 3), self.jet(J, 3)
        v = intersect(taylor_encl(jm, jJ, 0, J, I(m)), jJ[0])
        d = intersect(taylor_encl(jm, jJ, 1, J, I(m)), jJ[1])
        return v, d

    def classify(self, a, b, init_cells=INIT_CELLS):
        """verified partition of [a, b] into cells, each 'pos'/'neg' (value sign) or
        'inc'/'dec' (strict monotonicity in u for every a in A)."""
        a, b = mp.mpf(a), mp.mpf(b)
        h = (b - a) / init_cells
        edges = [a + k * h for k in range(init_cells)] + [b]
        stack = [(edges[k], edges[k + 1]) for k in range(init_cells - 1, -1, -1)]
        leaves = []
        n_eval = 0
        while stack:
            x, y = stack.pop()
            n_eval += 1
            if n_eval > MAX_CELLS:
                return None, f"{self.name}: more than {MAX_CELLS} cells (parameter box too wide here)"
            v, d = self.cell(x, y)
            if pos(v) or neg(v):
                leaves.append([x, y, "pos" if pos(v) else "neg", v])
            elif pos(d) or neg(d):
                leaves.append([x, y, "inc" if pos(d) else "dec", d])
            else:
                if y - x < MIN_WIDTH:
                    return None, f"{self.name}: unresolved cell [{mp.nstr(x, 12)}, {mp.nstr(y, 12)}]"
                mid = (x + y) / 2
                stack.append((mid, y))
                stack.append((x, mid))
        return leaves, None

    def segments(self, a, b):
        """segments with definite end signs: value cells and maximal monotone runs.
        Returns (segments, error); each segment = dict(a, b, kind, sl, sr) with sl/sr = +-1."""
        leaves, err = self.classify(a, b)
        if leaves is None:
            return None, err
        segs = []
        for x, y, kind, _ in leaves:
            if segs and kind in ("inc", "dec") and segs[-1]["kind"] == kind:
                segs[-1]["b"] = y
                segs[-1]["n_cells"] += 1
            else:
                segs.append({"a": x, "b": y, "kind": kind, "n_cells": 1})
        for i, sg in enumerate(segs):
            if sg["kind"] in ("pos", "neg"):
                sg["sl"] = sg["sr"] = 1 if sg["kind"] == "pos" else -1
                continue
            for end, nb in (("a", i - 1), ("b", i + 1)):
                v = self.point(sg[end])
                sgn = 1 if pos(v) else (-1 if neg(v) else 0)
                if sgn == 0 and 0 <= nb < len(segs) and segs[nb]["kind"] in ("pos", "neg"):
                    sgn = 1 if segs[nb]["kind"] == "pos" else -1   # end point lies in that value cell
                if sgn == 0:
                    return None, f"{self.name}: indefinite sign at run end {mp.nstr(sg[end], 12)}"
                sg["sl" if end == "a" else "sr"] = sgn
            # consistency: an increasing run cannot go from + to -, a decreasing one from - to +
            if (sg["kind"] == "inc" and sg["sl"] > sg["sr"]) or (sg["kind"] == "dec" and sg["sl"] < sg["sr"]):
                return None, f"{self.name}: inconsistent run end signs (internal error)"
        return segs, None

    def shrink_root(self, sg, iters=90):
        """shrink the enclosure of the unique zero (for each a) of a monotone run with a sign change."""
        L, R, sl = sg["a"], sg["b"], sg["sl"]
        mids = []
        for _ in range(iters):
            m = (L + R) / 2
            v = self.point(m)
            if (pos(v) and sl > 0) or (neg(v) and sl < 0):
                L = m
            elif pos(v) or neg(v):
                R = m
            else:
                mids.append(m)
                break
        if mids:                                        # band: shrink both sides towards it
            m0 = mids[0]
            l2, r2 = L, m0
            for _ in range(iters):
                m = (l2 + r2) / 2
                v = self.point(m)
                if (pos(v) and sl > 0) or (neg(v) and sl < 0):
                    l2 = m
                else:
                    r2 = m
                if r2 - l2 < MIN_WIDTH:
                    break
            l3, r3 = m0, R
            for _ in range(iters):
                m = (l3 + r3) / 2
                v = self.point(m)
                if (pos(v) and sl < 0) or (neg(v) and sl > 0):
                    r3 = m
                else:
                    l3 = m
                if r3 - l3 < MIN_WIDTH:
                    break
            L, R = l2, r3
        return L, R


def roots_in(segs):
    return [sg for sg in segs if sg["kind"] in ("inc", "dec") and sg["sl"] != sg["sr"]]


# ----------------------------------------------------------------------------
# helpers (all bounds below are interval evaluations; mid() only picks centres)
def mid(x):
    return (lo(x) + hi(x)) / 2


def u_to_t(U, A):
    """t = log 4 - log(cbar + Delta u/(2 a)) (enclosure over U x A)."""
    return LOG4 - iv.log(CB + DEL * U / (2 * A))


def phi_val(U, A):
    return jet_phi_fun(U, A, 0)[0]


def phi_u(U, A):
    return jet_phi_fun(U, A, 1)[1]


def box_sym(c, r):
    """outward interval [c - r, c + r] for mp numbers c, r >= 0 (contains c)."""
    return iv.mpf([lo(I(c) - I(r)), hi(I(c) + I(r))])


def krawczyk_fold_param(A, Xu, Xb):
    """K(X, A) for H(u, beta; a) = (log r0 - beta, Phi); X = Xu x Xb, centre = midpoint."""
    if not all(mp.isfinite(v) for v in (lo(Xu), hi(Xu), lo(Xb), hi(Xb))):
        raise ValueError("non-finite box")
    uc, bc = mid(Xu), mid(Xb)
    if not (lo(Xu) <= uc <= hi(Xu) and lo(Xb) <= bc <= hi(Xb)):
        raise ValueError("centre not in box")
    am = mid(A)
    Hm = [log_r0(I(uc), A) - I(bc), phi_val(I(uc), A)]
    # preconditioner: inverse of the (float-rounded) Jacobian at (uc, a_m) -- any real matrix is admissible
    j11m, j21m = dlogr0_du(I(uc), I(am)), phi_u(I(uc), I(am))
    a11, a12, a21, a22 = mid(j11m), mp.mpf(-1), mid(j21m), mp.mpf(0)
    det = a11 * a22 - a12 * a21
    Y = [[I(a22 / det), I(-a12 / det)], [I(-a21 / det), I(a11 / det)]]
    JX = [[dlogr0_du(Xu, A), I(-1)], [phi_u(Xu, A), I(0)]]
    dX = [Xu - I(uc), Xb - I(bc)]
    K = []
    for i in range(2):
        yh = Y[i][0] * Hm[0] + Y[i][1] * Hm[1]
        row = I(0)
        for k in range(2):
            ik = (I(1) if i == k else I(0)) - (Y[i][0] * JX[0][k] + Y[i][1] * JX[1][k])
            row += ik * dX[k]
        K.append([I(uc), I(bc)][i] - yh + row)
    inside = (lo(K[0]) > lo(Xu) and hi(K[0]) < hi(Xu) and lo(K[1]) > lo(Xb) and hi(K[1]) < hi(Xb))
    return K, bool(inside)


def fold_enclosure(A, Xu0, max_iter=25):
    """epsilon-inflation Krawczyk iteration; returns (Ku, Kb, Xu, Xb, n_iter) or None."""
    Xu = Xu0
    try:
        Xb = log_r0(Xu, A)
    except ValueError:
        return None
    for it in range(max_iter):
        wu, wb = hi(Xu) - lo(Xu), hi(Xb) - lo(Xb)
        Xu_i = iv.mpf([lo(Xu) - wu / 10 - mp.mpf("1e-25"), hi(Xu) + wu / 10 + mp.mpf("1e-25")])
        Xb_i = iv.mpf([lo(Xb) - wb / 10 - mp.mpf("1e-25"), hi(Xb) + wb / 10 + mp.mpf("1e-25")])
        try:
            K, inside = krawczyk_fold_param(A, Xu_i, Xb_i)
        except (ValueError, ZeroDivisionError):
            return None
        if inside:
            # the unique zero lies in K cap X; iterate X <- K(X) cap X to tighten (still an enclosure)
            Ku, Kb = K
            for _ in range(8):
                try:
                    K2, _ins = krawczyk_fold_param(A, Ku, Kb)
                except (ValueError, ZeroDivisionError):
                    break
                nu, nb = intersect(K2[0], Ku), intersect(K2[1], Kb)
                if (hi(nu) - lo(nu)) > mp.mpf("0.9") * (hi(Ku) - lo(Ku)) and \
                        (hi(nb) - lo(nb)) > mp.mpf("0.9") * (hi(Kb) - lo(Kb)):
                    Ku, Kb = nu, nb
                    break
                Ku, Kb = nu, nb
            return Ku, Kb, Xu_i, Xb_i, it + 1
        Xu = intersect_or(K[0], Xu_i)
        Xb = K[1]
    return None


def intersect_or(x, y):
    a, b = max(lo(x), lo(y)), min(hi(x), hi(y))
    return iv.mpf([a, b]) if a <= b else x


def newton_1d(A, Xu):
    """1-D interval Newton on Phi(.; a): N = u~ - Phi(u~; A)/Phi_u(Xu; A) in int Xu."""
    uc = mid(Xu)
    d = phi_u(Xu, A)
    if not (pos(d) or neg(d)):
        return None, False
    N = I(uc) - phi_val(I(uc), A) / d
    return N, bool(lo(N) > lo(Xu) and hi(N) < hi(Xu))


# ----------------------------------------------------------------------------
def certify_box(r_lo: str, r_hi: str):
    """certificate of (D1)-(D4) and of the fold curve for every rho_s in [r_lo, r_hi]."""
    t0 = time.time()
    A, P = a_box(r_lo, r_hi)
    rec = {"rho_box": [r_lo, r_hi], "a_box": to_list(A), "certified": False}
    fail = []
    # ---------------- (D1): zeros of s on a superset of the u-image of I ----------
    U_tau = (MU_TAU - CB) * 2 * A / DEL
    U_T = (MU_T - CB) * 2 * A / DEL
    S = PFun(jet_s, A, "s")
    segs, err = S.segments(lo(U_T), hi(U_tau))
    if segs is None:
        rec.update(error=err, runtime_s=round(time.time() - t0, 2))
        return rec
    rts = roots_in(segs)
    kinds = [r["kind"] for r in rts]
    encl = [S.shrink_root(r) for r in rts]
    s_tau, s_T = jet_s(U_tau, A, 0)[0], jet_s(U_T, A, 0)[0]
    d1 = (kinds == ["dec", "inc", "dec"] and neg(s_tau) and pos(s_T)
          and all(x > hi(U_T) and y < lo(U_tau) for x, y in encl))
    rec["D1"] = {"holds": bool(d1), "zero_runs_of_s_in_u_order": kinds,
                 "n_segments": len(segs), "n_jets": S.n_jets,
                 "s_at_u_tau": to_list(s_tau), "s_at_u_T": to_list(s_T),
                 "u_window_superset": [down(lo(U_T)), up(hi(U_tau))]}
    if not d1:
        fail.append("D1")
        rec.update(fail=fail, runtime_s=round(time.time() - t0, 2))
        return rec
    (m_a, m_b), (z_a, z_b), (p_a, p_b) = encl        # -u* (hat t_2), 0 (check t_1), u* (hat t_1)
    Rm, R0, Rp = I(m_a, m_b), I(z_a, z_b), I(p_a, p_b)
    rec["D1"]["critical_points_t"] = {"hat_t1": to_list(u_to_t(Rp, A)), "check_t1": to_list(u_to_t(R0, A)),
                                      "hat_t2": to_list(u_to_t(Rm, A))}
    rec["D1"]["critical_points_u"] = {"u_star": to_list(Rp), "zero": to_list(R0), "minus_u_star": to_list(Rm)}
    # ---------------- (D2): Phi < 0 on [u*, u_tau] -----------------------------
    PH = PFun(jet_phi_fun, A, "Phi")
    segs2, err2 = PH.segments(p_a, hi(U_tau))
    d2 = segs2 is not None and all(sg["sl"] < 0 and sg["sr"] < 0 for sg in segs2)
    rec["D2"] = {"holds": bool(d2), "n_segments": None if segs2 is None else len(segs2),
                 "error": err2, "interval_u": [down(p_a), up(hi(U_tau))]}
    # ---------------- (D3): one nondegenerate max of r0 on the flank --------------
    segs3, err3 = PH.segments(m_a, z_b)
    d3 = False
    rec["D3"] = {"error": err3}
    if segs3 is not None:
        r3 = roots_in(segs3)
        ph_m = PH.cell(m_a, m_b)[0]
        ph_0 = PH.cell(z_a, z_b)[0]
        d3 = len(r3) == 1 and r3[0]["kind"] == "inc" and neg(ph_m) and pos(ph_0)
        rec["D3"] = {"holds": bool(d3), "n_zero_runs_of_Phi": len(r3), "n_segments": len(segs3),
                     "zero_run_direction": r3[0]["kind"] if r3 else None,
                     "Phi_on_encl_hat_t2": to_list(ph_m), "Phi_on_encl_check_t1": to_list(ph_0)}
        if d3:
            f_a, f_b = PH.shrink_root(r3[0])
            rec["D3"]["u_fold_counting_encl"] = to_list(I(f_a, f_b))
    rec["D3"]["holds"] = bool(d3)
    if not (d2 and d3):
        fail += [k for k, ok in (("D2", d2), ("D3", d3)) if not ok]
        rec.update(fail=fail, runtime_s=round(time.time() - t0, 2))
        return rec
    # ---------------- fold: parametric Krawczyk on (u, log B) ---------------------
    Xu0 = I(f_a, f_b)
    kr = fold_enclosure(A, Xu0)
    if kr is None:
        fail.append("krawczyk")
        rec.update(fail=fail, runtime_s=round(time.time() - t0, 2))
        return rec
    Ku, Kb, Xu, Xb, nit = kr
    # the Krawczyk zero is (u_f(a), log b_2(a)) because X_u lies in the (D3) region, where
    # u_f(a) is the only zero of Phi(.; a)
    in_region = bool(lo(Xu) >= m_a and hi(Xu) <= z_b)
    Nn, n_ok = newton_1d(A, Xu)
    # thin-parameter value at the a-midpoint and the envelope slope over the box
    am = mid(A)
    Am = I(am)
    krm = fold_enclosure(Am, box_sym(mid(Ku), (hi(Ku) - lo(Ku)) / 2 + mp.mpf("1e-20")))
    tube = None
    if krm is not None and lo(krm[2]) >= m_a and hi(krm[2]) <= z_b:
        Kum, Kbm = krm[0], krm[1]
        slope = dlogr0_da(Ku, A)                          # d beta / d a on the box (envelope identity)
        beta_tube = Kbm + slope * (A - Am)
        beta_best = intersect(beta_tube, Kb)
        # at the box ends the tube is narrow: beta(a_end) in Kbm + slope * (a_end - a_m)
        ends = {}
        for name, r_end in (("at_rho_hi", r_hi), ("at_rho_lo", r_lo)):
            be = Kbm + slope * (DEL / (2 * I(r_end)) - Am)      # a(rho_end) enclosed, inside A
            ends[name] = {"logB": to_list(be), "B": to_list(iv.exp(be)),
                          "rel_halfwidth_B_upper": up(hi((iv.exp(I(hi(be)) - I(lo(be))) - 1) / 2))}
        # re-audit (minor): a_m is a dyadic number and is also stored EXACTLY ("a_mid_exact"), so the tube
        # K_m + S (a - a_m) can be rebuilt from the stored coefficients without widening (validator C5)
        tube = {"a_mid": to_list(Am), "a_mid_exact": th8.exact(am),
                "logB_at_a_mid": to_list(Kbm), "B_at_a_mid": to_list(iv.exp(Kbm)),
                "u_fold_at_a_mid": to_list(Kum), "t_fold_at_a_mid": to_list(u_to_t(Kum, Am)),
                "rho_at_a_mid": to_list(DEL / (2 * Am)),
                "dlogB_da_on_box": to_list(slope),
                "dB_drho_on_box": to_list(-iv.exp(Kb) * slope * A * A * 2 / DEL),
                "logB_tube_hull": to_list(beta_best), "ends": ends}
    # nondegeneracy constant a_f = -r0''(t_f) = (k+u) Phi_u r0 / (-s) on the fold box
    s_f = s_encl(Ku, A)
    a_f = (KAP * A + Ku) * phi_u(Ku, A) * iv.exp(Kb) / (-s_f)
    # ---------------- (D4): r0(tau) > b_2 ----------------------------------------
    lr_tau = log_r0(U_tau, A)
    d4 = bool(lo(lr_tau) > hi(Kb))
    rec["fold"] = {"krawczyk_ok": True, "krawczyk_iterations": nit, "krawczyk_box_in_D3_region": in_region,
                   "X_u": to_list(Xu), "X_logB": to_list(Xb),
                   "u_fold": to_list(Ku), "logB_fold": to_list(Kb), "B_top_det": to_list(iv.exp(Kb)),
                   "t_fold": to_list(u_to_t(Ku, A)),
                   "newton_1d_ok": n_ok, "newton_1d_N": None if Nn is None else to_list(Nn),
                   "a_f=-r0pp(t_f)": to_list(a_f), "a_f_positive": bool(pos(a_f)),
                   "tube": tube}
    rec["D4"] = {"holds": d4, "log_r0_tau_lower_bound": down(lo(lr_tau)),
                 "log_b2_upper_bound": up(hi(Kb)),
                 # A3 audit (minor): the gap is an interval difference, not a nearest-rounded mp subtraction
                 "log_gap_lower_bound": down(lo(lr_tau - Kb))}
    # A3 audit (major): strict decrease of B_top^det in rho is part of the certified claim, so the sign of
    # the envelope slope d log b_2 / d a (> 0; a = Delta/(2 rho) is strictly decreasing in rho) is tested here
    mono = tube is not None and pos(slope)
    if tube is not None:
        tube["dlogB_da_positive"] = bool(pos(slope))
        tube["dB_drho_negative"] = bool(neg(-iv.exp(Kb) * slope * A * A * 2 / DEL))
    ok = d1 and d2 and d3 and d4 and in_region and n_ok and pos(a_f) and tube is not None and mono
    if not ok:
        fail += [k for k, v in (("D4", d4), ("krawczyk_region", in_region), ("newton", n_ok),
                                ("a_f", pos(a_f)), ("tube", tube is not None), ("monotone", mono)) if not v]
    rec["certified"] = bool(ok)
    rec["fail"] = fail
    rec["runtime_s"] = round(time.time() - t0, 2)
    return rec


# ----------------------------------------------------------------------------
# adaptive subdivision of a rho-range
def run_base(args):
    """certify [q_lo, q_hi] (Fractions); bisect a failed box up to MAX_DEPTH."""
    q_lo, q_hi, depth = args
    rec = certify_box(frac_str(q_lo), frac_str(q_hi))
    rec["depth"] = depth
    if rec["certified"] or depth >= MAX_DEPTH:
        return [rec]
    qm = (q_lo + q_hi) / 2
    return run_base((q_lo, qm, depth + 1)) + run_base((qm, q_hi, depth + 1))


def run_range(q_lo, q_hi, n, workers):
    h = (q_hi - q_lo) / n
    jobs = [(q_lo + i * h, q_lo + (i + 1) * h, 0) for i in range(n)]
    if workers > 1:
        import multiprocessing as mpc
        with mpc.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(run_base, jobs, chunksize=4)
    else:
        parts = [run_base(j) for j in jobs]
    return [r for p in parts for r in p]


def summarise(leaves, q_lo, q_hi, n):
    tot = q_hi - q_lo
    cert = [r for r in leaves if r["certified"]]
    wid = lambda r: Fraction(r["rho_box"][1]) - Fraction(r["rho_box"][0])      # noqa: E731
    cov = sum((wid(r) for r in cert), Fraction(0)) / tot
    excl = []
    for r in leaves:
        if not r["certified"]:
            a, b = Fraction(r["rho_box"][0]), Fraction(r["rho_box"][1])
            if excl and excl[-1][1] == a:
                excl[-1][1] = b
            else:
                excl.append([a, b])
    out = {"range": [frac_str(q_lo), frac_str(q_hi)], "n_base_boxes": n,
           "base_box_width": frac_str(tot / n), "max_bisection_depth": MAX_DEPTH,
           "n_leaf_boxes": len(leaves), "n_certified": len(cert),
           "coverage_fraction": float(cov), "coverage_fraction_exact": str(cov),
           "excluded_intervals": [[frac_str(a), frac_str(b)] for a, b in excl],
           "failures": sorted({f for r in leaves for f in r.get("fail", [])} |
                              {r["error"] for r in leaves if r.get("error")})}
    if cert:
        fo = [r["fold"] for r in cert]
        out.update({
            "all_certified_boxes_D1_D2_D3_D4": all(r["D1"]["holds"] and r["D2"]["holds"] and r["D3"]["holds"]
                                                   and r["D4"]["holds"] for r in cert),
            "all_krawczyk_and_newton_ok": all(f["krawczyk_ok"] and f["newton_1d_ok"] for f in fo),
            "max_krawczyk_inflation_iterations": max(f["krawczyk_iterations"] for f in fo),
            "B_top_det_hull_over_range": [min((f["B_top_det"][0] for f in fo), key=mp.mpf),
                                          max((f["B_top_det"][1] for f in fo), key=mp.mpf)],
            "t_fold_hull_over_range": [min((f["t_fold"][0] for f in fo), key=mp.mpf),
                                       max((f["t_fold"][1] for f in fo), key=mp.mpf)],
            "a_f_lower_bound_min": min((f["a_f=-r0pp(t_f)"][0] for f in fo), key=mp.mpf),
            "D4_log_gap_lower_bound_min": min((r["D4"]["log_gap_lower_bound"] for r in cert), key=mp.mpf),
            "tube_rel_halfwidth_B_max_upper": max((e["rel_halfwidth_B_upper"] for f in fo
                                                   for e in f["tube"]["ends"].values()), key=mp.mpf),
            "max_box_runtime_s": max(r["runtime_s"] for r in leaves)})
    return out


def table(leaves):
    """compact per-box rows for plotting (certified bounds as strings)."""
    rows = []
    for r in leaves:
        row = {"rho": r["rho_box"], "certified": r["certified"]}
        if r["certified"]:
            f = r["fold"]
            row.update(B_hull=f["B_top_det"], t_fold_hull=f["t_fold"],
                       B_at_rho_lo=f["tube"]["ends"]["at_rho_lo"]["B"],
                       B_at_rho_hi=f["tube"]["ends"]["at_rho_hi"]["B"],
                       dB_drho=f["tube"]["dB_drho_on_box"])
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# terminal width (not certified; located): rho_c = Delta/2, B_top^det -> 0 like (rho_c - rho)^{3/2}
def mp_fold(a):
    """100-bit bisection for (u*, u_f, B_top^det, t_f) at a > 1 (diagnostic, not a bound)."""
    a = mp.mpf(a)
    c1, c2 = 4 * mp.e ** -1, 4 * mp.exp(mp.mpf("-2.5"))
    cb, D = (c1 + c2) / 2, c1 - c2
    kap = (c1 + c2) / D
    s = lambda u: -u + a * mp.tanh(a * u)                                      # noqa: E731
    su = lambda u: -1 + a * a / mp.cosh(a * u) ** 2                            # noqa: E731
    Phi = lambda u: (kap * a + u) * (su(u) - s(u) ** 2) + s(u)                 # noqa: E731
    l_, h_ = mp.sqrt(a * a - 1) / (4 * a * a), a                               # s > 0 at l_ (a > 1)
    for _ in range(330):
        m_ = (l_ + h_) / 2
        l_, h_ = (m_, h_) if s(m_) > 0 else (l_, m_)
    us = l_
    x0, x1 = -us * (1 - mp.mpf("1e-25")), -us * mp.mpf("1e-25")
    for _ in range(330):
        xm = (x0 + x1) / 2
        x0, x1 = (xm, x1) if Phi(xm) < 0 else (x0, xm)
    uf = (x0 + x1) / 2
    rho = D / (2 * a)
    B = -s(uf) * (kap * a + uf) * mp.sqrt(2 * mp.pi) * rho * mp.exp((uf ** 2 + a * a) / 2) / mp.cosh(a * uf)
    tf = mp.log(4 / (cb + rho * uf))
    return us, uf, B, tf, cb, D


def terminal_width():
    rc = RHO_C
    c1, c2 = 4 * mp.e ** -1, 4 * mp.exp(mp.mpf("-2.5"))
    cb, D = (c1 + c2) / 2, c1 - c2
    rcm = D / 2
    C_pred = 4 * mp.sqrt(2) / 3 * mp.sqrt(2 * mp.pi) * mp.exp(mp.mpf("0.5")) * cb
    rows = []
    for k in range(4, 33):
        eps = mp.mpf(10) ** (-mp.mpf(k) / 4)
        rho = rcm * (1 - eps)
        us, uf, B, tf, _, _ = mp_fold(rcm / rho)
        rows.append({"eps=1-rho/rho_c": float(eps), "rho": float(rho), "B_top_det": float(B), "t_fold": float(tf),
                     "u_f_over_u_star": float(uf / us), "B_over_C_eps^1.5": float(B / (C_pred * eps ** 1.5))})
    sel = [r for r in rows if r["eps=1-rho/rho_c"] <= 1e-5]
    xs = [math.log(r["eps=1-rho/rho_c"]) for r in sel]
    ys = [math.log(r["B_top_det"]) for r in sel]
    n = len(xs)
    xm_, ym_ = sum(xs) / n, sum(ys) / n
    slope = sum((x - xm_) * (y - ym_) for x, y in zip(xs, ys)) / sum((x - xm_) ** 2 for x in xs)
    return {"rho_c=Delta/2=2(e^-1-e^-5/2)": to_list(rc), "a_at_rho_c": 1,
            "t_valley_check_t1=log(4/cbar) (rho-independent)": to_list(LOG4 - iv.log(CB)),
            "status": "located analytically (equal-weight two-Gaussian bimodality threshold a = 1); "
                      "not interval-certified as a curve endpoint",
            "mechanism": "at a = 1 the three critical points of G0 (hat t1, check t1, hat t2) merge at check t1 "
                         "(symmetric pitchfork in mu; equal weights); on the flank s = O(eps^{3/2}) so b_2 -> 0",
            "asymptote": "B_top^det ~ C (1 - rho/rho_c)^{3/2}, C = (4 sqrt2/3) sqrt(2 pi) e^{1/2} cbar",
            "C_pred": float(C_pred), "fitted_exponent_eps_le_1e-5": slope,
            "u_f/u* limit": -1 / math.sqrt(3), "rows": rows}


def float_curve():
    """dense (rho, B_top^det, t_f) samples for plotting (100-bit bisection, diagnostic)."""
    c1, c2 = 4 * mp.e ** -1, 4 * mp.exp(mp.mpf("-2.5"))
    D = c1 - c2
    out = []
    for i in range(0, 211):
        rho = mp.mpf("0.15") + mp.mpf(i) * mp.mpf("0.002")
        if rho >= D / 2:
            break
        _, uf, B, tf, _, _ = mp_fold(D / (2 * rho))
        out.append([float(rho), float(B), float(tf)])
    return {"columns": ["rho_s", "B_top_det", "t_fold"], "note": "100-bit bisection; diagnostic, not a bound",
            "rows": out}


def anchors():
    cert = json.loads((HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "TH8_certificate"
                       / "certificate.json").read_text())["cases"]
    out = {}
    for rho in ("0.15", "0.2", "0.25", "0.3", "0.35", "0.4"):
        r = certify_box(rho, rho)
        old = cert[f"m2_rho{rho}"]
        new_B = r["fold"]["B_top_det"] if r["certified"] else None
        ov = None
        if new_B:
            ov = bool(max(mp.mpf(new_B[0]), mp.mpf(old["B_top_det"][0])) <= min(mp.mpf(new_B[1]),
                                                                                 mp.mpf(old["B_top_det"][1])))
        out[rho] = {"certified_thin": r["certified"], "B_top_det_uA": new_B, "B_top_det_th8": old["B_top_det"],
                    "t_fold_uA": r["fold"]["t_fold"] if r["certified"] else None,
                    "t_fold_th8": old["D3"]["flanks"][0]["t_fold"], "overlap": ov,
                    "a_f_uA": r["fold"]["a_f=-r0pp(t_f)"] if r["certified"] else None,
                    "a_f_th8": old.get("fold", {}).get("a_f=-r0pp(t_f)")}
        # A3 audit (major): "coincide to all printed digits" is a string identity of the serialised
        # enclosures, which overlap alone does not establish
        out[rho]["printed_digits_identical"] = {
            "B_top_det": new_B == old["B_top_det"],
            "t_fold": out[rho]["t_fold_uA"] == out[rho]["t_fold_th8"],
            "a_f": (out[rho]["a_f_uA"] == out[rho]["a_f_th8"]) if out[rho]["a_f_th8"] is not None else None}
    return out


def falsification_test(leaves):
    """independent 100-bit point solves (mp_fold) at 3 interior widths of every certified box:
    B_top^det(rho) must lie in the box hull and in the thin tube, t_f in the t hull (diagnostic)."""
    n = bad_hull = bad_tube = bad_t = 0
    worst = 0.0
    for r in leaves:
        if not r["certified"]:
            continue
        f = r["fold"]
        tb = f["tube"]
        Kbm, sl, am = iv.mpf(tb["logB_at_a_mid"]), iv.mpf(tb["dlogB_da_on_box"]), iv.mpf(tb["a_mid"])
        r_lo, r_hi = Fraction(r["rho_box"][0]), Fraction(r["rho_box"][1])
        for k in (1, 2, 3):
            q = r_lo + (r_hi - r_lo) * k / 4
            rho = mp.mpf(q.numerator) / q.denominator
            c1, c2 = 4 * mp.e ** -1, 4 * mp.exp(mp.mpf("-2.5"))
            _, _, B, tf, _, _ = mp_fold((c1 - c2) / (2 * rho))
            n += 1
            bh = [mp.mpf(x) for x in f["B_top_det"]]
            th = [mp.mpf(x) for x in f["t_fold"]]
            bad_hull += not (bh[0] <= B <= bh[1])
            bad_t += not (th[0] <= tf <= th[1])
            be = iv.exp(Kbm + sl * (DEL / (2 * I(frac_str(q))) - am))
            bad_tube += not (lo(be) <= B <= hi(be))
            worst = max(worst, float((hi(be) - lo(be)) / B))
    return {"n_points": n, "violations_B_hull": bad_hull, "violations_B_tube": bad_tube,
            "violations_t_hull": bad_t, "max_tube_rel_width_at_test_points": worst,
            "note": "points at 1/4, 1/2, 3/4 of each certified rho-box; 100-bit bisection (not a bound)"}


def float_crosscheck(leaves):
    """float64 grid scan of fb_finite_width_det at non-anchor widths vs the certified tube."""
    import fb_finite_width_det as fwd
    out = {}
    for rho in ("0.1737", "0.2891", "0.3813"):
        q = Fraction(rho)
        box = next(r for r in leaves if Fraction(r["rho_box"][0]) <= q <= Fraction(r["rho_box"][1]))
        fdet = fwd.analyse(2, float(rho), n=600_001)
        tb = box["fold"]["tube"]
        Kbm = iv.mpf(tb["logB_at_a_mid"])
        sl = iv.mpf(tb["dlogB_da_on_box"])
        am = iv.mpf(tb["a_mid"])
        a_r = DEL / (2 * I(rho))
        B_t = iv.exp(Kbm + sl * (a_r - am))
        out[rho] = {"float_B_top_det": fdet["B_top_det"], "float_binding": fdet["binding"],
                    "float_D1_D2_D3": [fdet["D1_signature_ok"], fdet["D2_first_flank_monotone"],
                                       fdet["D3_flank_unimodal"]],
                    "certified_tube_B": to_list(B_t),
                    "rel_dev_float_vs_tube_mid": float(abs(fdet["B_top_det"] / mid(B_t) - 1))}
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-base", type=int, default=N_BASE)
    ap.add_argument("--n-ext", type=int, default=N_EXT)
    args = ap.parse_args(argv)
    t0 = time.time()
    leaves = run_range(RHO_RANGE[0], RHO_RANGE[1], args.n_base, args.workers)
    print(f"required range: {len(leaves)} leaves, {sum(r['certified'] for r in leaves)} certified "
          f"({time.time() - t0:.1f} s)", flush=True)
    ext = run_range(EXT_RANGE[0], EXT_RANGE[1], args.n_ext, args.workers)
    print(f"extension: {len(ext)} leaves, {sum(r['certified'] for r in ext)} certified "
          f"({time.time() - t0:.1f} s)", flush=True)
    summ, summ_ext = summarise(leaves, *RHO_RANGE, args.n_base), summarise(ext, *EXT_RANGE, args.n_ext)
    anc = anchors()
    xchk = float_crosscheck(leaves)
    fals = {"required_range": falsification_test(leaves), "extension": falsification_test(ext)}
    print(json.dumps(fals), flush=True)
    term = terminal_width()
    curve = float_curve()
    data = {
        "script": "code/fb_v2_fold_curve_certificate.py",
        "item": "V2 item 3 / CV-1: parametric interval certificate of the m=2 fold curve B_top^det(rho_s) "
                "and of (D1)-(D4) on every rho_s-box of [0.15, 0.40]",
        "deterministic": True, "seeds": None,
        "arithmetic": f"mpmath {__import__('mpmath').__version__} iv, outward rounding, {PREC}-bit mantissa; "
                      "JSON decimal bounds rounded outward (18 significant digits); rho-box endpoints are exact "
                      "decimals entered as outward enclosures, consecutive boxes share endpoints",
        "serialization": "every [lo, hi] string pair is a certified enclosure; *_lower_bound / *_upper_bound are "
                         "certified one-sided bounds; float fields (coverage_fraction, runtime, *float*, rows of "
                         "terminal_width and float_curve, fitted exponents) are diagnostics, not bounds",
        "geometry": {"gamma": 1, "D0": 1, "z0": 4, "zbar": 0, "W": 1, "d": 2, "window": [0.5, 3.5],
                     "targets": [1.0, 2.5], "weights": "equal", "m": 2},
        "reduction": {"a": "Delta/(2 rho)", "u": "(mu(t)-cbar)/rho", "k": "cbar/rho = kappa a",
                      "s": "-u + a tanh(a u)", "Phi": "(k+u)(s_u - s^2) + s",
                      "r0'": "(k+u) Phi e^{-phi}", "r0''(t_f)": "-(k+u)^2 Phi_u e^{-phi}",
                      "log r0": "log(-s)+log(k+u)+log(sqrt(2 pi) rho)+(u^2+a^2)/2-log cosh(a u)",
                      "kappa": to_list(KAP), "cbar": to_list(CB), "Delta": to_list(DEL)},
        "certified_flag": "certified = (D1) & (D2) & (D3) & (D4) & parametric Krawczyk K(X,A) in int X & "
                          "1-D interval Newton N in int X_u & a_f = -r0''(t_f) > 0 & thin-midpoint tube certified & "
                          "d log b_2/d a > 0 on the box (strict decrease in rho; A3 audit)",
        "contracts": {
            "zero_counting": "cells are closed and share end points; a value-sign cell excludes roots; merged runs of "
                             "cells with the same definite derivative sign are strictly monotone for every parameter "
                             "in the box, so a run has exactly one zero iff its definite end signs differ; an end "
                             "sign may be taken from an adjacent closed sign-certified cell",
            "krawczyk": "K(X, A) = x~ - Y H(x~, A) + (Id - Y DH(X, A))(X - x~); Y is a real approximate inverse used "
                        "as a FIXED preconditioner (it need not enclose the exact inverse); K in int X proves, for "
                        "every a in A, a unique zero of H(., a) in X; the zero is identified with the fold because "
                        "X_u lies in the (D3) region, where the fold is the only zero of Phi(., a); after success "
                        "X <- K cap X keeps the zero",
            "newton_1d": "N(X_u) = m - Phi(m; A)/Phi_u(X_u; A) with 0 excluded from Phi_u(X_u; A) and N strictly "
                         "inside X_u (cross-check)",
            "approximate_arithmetic": "mp (nearest) arithmetic is used only for CHOICES (subdivision points, Newton "
                                      "starts, Krawczyk centres, preconditioners, trial boxes, decimal grid "
                                      "points); every bound is an iv expression or an exact Fraction/Decimal "
                                      "comparison, and endpoints are serialised outward last",
            "tube_accuracy": "pointwise tube beta(a) in K_m + S (a - a_m) (K_m = thin-midpoint Krawczyk enclosure of "
                             "log b_2(a_m), S = [d_a log r0](U_f, A)); its width w(K_m) + w(S)|a - a_m| is convex in "
                             "a - a_m, so its maximum over a(rho), rho in the box, is attained at a box end, where "
                             "it is evaluated with the enclosure of a(rho_end) = Delta/(2 rho_end); with w = that "
                             "width, the midpoint M of [e^{beta_lo}, e^{beta_hi}] satisfies "
                             "|B - M| <= B (e^w - 1)/2 (rel_halfwidth_B_upper); this is NOT the whole-box hull"},
        "method": __doc__.split("Method (per rho-box; A = a-box):")[1].split("Run (from code/):")[0].strip(),
        "required_range": summ, "extension_towards_rho_c": summ_ext,
        "anchor_consistency_vs_TH8_certificate": anc,
        "float_crosscheck_fb_finite_width_det": xchk,
        "falsification_test_point_solves": fals,
        "terminal_width": term,
        "not_a_cusp": ("a cusp of the f_det critical set needs r0' = r0'' = 0 at r0 = B > 0, i.e. Phi = Phi_u = 0 "
                       "with s < 0. Equal weights, every rho < rho_c (analytic, A3 audit): on the first rising flank "
                       "(u > u*) s < 0 and s_u < 0 give Phi < 0 (no critical point of r0); on the second rising flank "
                       "(-u* < u < 0) any zero of Phi has s_u - s^2 = -s/(k+u) > 0 (so s_u > 0) and "
                       "s_uu = -2 a^3 sech^2(a u) tanh(a u) > 0, hence Phi_u = 2 s_u - s^2 + (k+u)(s_uu - 2 s s_u) > 0: "
                       "every fold there is a nondegenerate maximum of r0 (so there is exactly one); on the falling "
                       "flanks s > 0 and r0 < 0. For rho >= rho_c (a <= 1) s_u < 0 for u != 0, Phi < 0 on u > 0, and "
                       "r0 decreases strictly on [tau, t(u = 0)]: one interior peak of f_det iff 0 < B < r0(tau). "
                       "Hence no cusp at B > 0 for any width; the certified boxes re-verify (D3) numerically"),
        "table_required_range": table(leaves), "table_extension": table(ext),
        "float_curve": curve,
        "box_records_file": "fold_curve_m2_boxes.json",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    # re-audit (major): provenance of the run (source hashes, library versions); the generator's exit status only
    # reports complete certified coverage, theorem acceptance is `fb_v2_cert_validator.py --only cv1`
    import hashlib
    import platform
    import mpmath
    data["provenance"] = {
        "source_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                          for f in ("fb_v2_fold_curve_certificate.py", "fb_th8_interval_certificate.py")},
        "mpmath_version": mpmath.__version__, "python_version": platform.python_version(),
        "acceptance": "code/fb_v2_cert_validator.py --only cv1 (exact rational checks of every claim of "
                      "Prop. v2:prop-foldcurve); the exit status of this generator is NOT theorem acceptance"}
    complete = bool(summ["n_certified"] == summ["n_leaf_boxes"] and summ["coverage_fraction_exact"] == "1"
                    and summ_ext["n_certified"] == summ_ext["n_leaf_boxes"]
                    and summ_ext["coverage_fraction_exact"] == "1")
    data["complete_certified_coverage"] = complete
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    (OUT.parent / "fold_curve_m2_boxes.json").write_text(json.dumps(
        {"script": data["script"], "required_range": leaves, "extension_towards_rho_c": ext}, indent=0))
    print("wrote", OUT)
    print(json.dumps({"required": summ, "extension": summ_ext}, indent=1))
    print(json.dumps(anc, indent=1))
    print(json.dumps(xchk, indent=1))
    print({k: v for k, v in term.items() if k != "rows"})
    if not complete:
        print("INCOMPLETE certified coverage: some leaf boxes are not certified", flush=True)
        return 1
    print("complete certified coverage; run fb_v2_cert_validator.py --only cv1 for theorem acceptance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
