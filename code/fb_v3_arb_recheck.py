#!/usr/bin/env python3
"""V3: independent recheck of the Proposition 4 interval certificates with a DIFFERENT interval
library: FLINT/Arb midpoint-radius ball arithmetic (python-flint), 256-bit midpoints.

Nothing is imported from the mpmath certificates (fb_th8_interval_certificate.py,
fb_v2_fold_curve_certificate.py, fb_v2_codim2_switch.py); only their JSON outputs are read and
compared.  The limit-law quantities are re-implemented from the definitions (anchor geometry of
Prop. 4: gamma = D0 = 1, z0 = 4, zbar = 0, W = 1, d = 2, I = [tau, T] = [0.5, 3.5]):

    mu(t) = 4 e^{-t},  c_j = 4 e^{-t_j},
    G0(t) = (2 pi)^{-1/2} rho^{-1} sum_j w_j exp(-(mu(t) - c_j)^2 / (2 rho^2)),
    r0 = G0'/G0^2,  N = G0'' G0 - 2 G0'^2 (sign r0' = sign N),  Lambda0(t) = int_0^t G0,
    fold of flank j: N(t_f) = 0 on (check t_{j-1}, hat t_j),  b_j = r0(t_f),
    B_top^det = min{r0(tau), b_2, ..., b_m}.

Method (deliberately different from the mpmath certificates where a choice exists):
  * Taylor jets in t from Arb power series (arb_series: exp, products, quotients, derivative);
    a jet evaluated at a ball T encloses F^{(k)}(t)/k! for every t in T.
  * Parameters (rho for the fold curve, s = w_2/eta for the F3S allocation family) enter as balls
    in t-coordinates (NOT the (u, a) reduction of fb_v2_fold_curve_certificate.py); every jet
    coefficient is enclosed by the mean-value form in the parameter,
        c_k(T, P) in c_k(T, p_m) + [d_p c_k](T, P) (P - p_m),
    with d_p G0 written out explicitly (d_rho [rho^{-1} E_j] = rho^{-1} E_j ((mu-c_j)^2/rho^3 - 1/rho),
    d_s G0 = (2 pi)^{-1/2} rho^{-1} eta (E_2 - E_3)), intersected with the naive ball evaluation.
  * Verified zero counting on dyadic cells (128 initial cells): a cell is resolved when the
    Taylor-form value enclosure excludes 0, or the t-derivative enclosure excludes 0; monotone
    cells of one direction are merged into runs, a run has exactly one zero (for every parameter
    value) iff its definite end signs differ.
  * Zeros are enclosed by the interval Newton operator X <- (m - F(m)/F'(X)) cap X (not by
    bisection); with a thick parameter box the limit is an enclosure of the zero's range.
  * B_top^det over a rho-box: t_f(rho) enclosed by parametric Newton, then the envelope identity
    dB/drho = d_rho r0 (t_f(rho), rho) (d_t r0 = 0 at the fold) is enclosed on T_f x R; if it is
    < 0, B(R) = [B(rho_hi), B(rho_lo)] exactly, and the two end values are thin certificates.
  * Lambda0 by rigorous Arb integration (acb.integral, Petras algorithm) of the entire integrand.
  * Codimension-2 switch: interval Newton in s on D(s) = b_2(s) - b_3(s) with D'(S) from the
    envelope identity on the thick box (not bisection + 4-D Krawczyk).

Outputs are compared with the stored enclosures by EXACT rational comparison (decimal strings of
the stored JSON parsed as Fractions, Arb ball endpoints converted exactly from binary).  For each
compared quantity the record states whether the Arb enclosure lies inside the stored one, the
converse, and whether they overlap (two valid enclosures of the same number MUST overlap; a
disjoint pair is a disagreement).

Run (from code/; python-flint >= 0.7 in the interpreter):
    <venv>/bin/python fb_v3_arb_recheck.py [--workers 3] [--sample [--n-ext 44] [--n-req 20]] [--only PART]
Output: ../artifacts/data/exact_m_fixed_budget/V3_arb_recheck/recheck.json
Deterministic: the box sample uses random.Random(SEED) on the sorted box lists.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from fractions import Fraction as Q  # noqa: E402
from pathlib import Path  # noqa: E402

import flint  # noqa: E402
from flint import acb, arb, arb_series, ctx, fmpq  # noqa: E402

PREC = 256
ctx.prec = PREC
SEED = 20260926

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
TH8_JSON = DATA / "TH8_certificate" / "certificate.json"
FOLD_JSON = DATA / "V2_bifurcation" / "fold_curve_m2.json"
BOXES_JSON = DATA / "V2_bifurcation" / "fold_curve_m2_boxes.json"
CODIM2_JSON = DATA / "V2_bifurcation" / "codim2_switch.json"
OUT = DATA / "V3_arb_recheck" / "recheck.json"

TAU, TEND = Q(1, 2), Q(7, 2)
TARGETS = {2: ("1.0", "2.5"), 3: ("0.8", "1.6", "2.8")}
N0 = 128                      # initial cells of a scan
MIN_W = Q(1, 2 ** 70)         # smallest cell width before a scan gives up
MAX_CELLS = 40000
SNAP = 96                     # dyadic snapping (bits) of scan end points (outward)


# ----------------------------------------------------------------------------
# exact conversions
def qa(x: Q) -> arb:
    """ball enclosing the rational x (exact if x is dyadic with a short mantissa)."""
    return arb(fmpq(x.numerator, x.denominator))


def ball(a: Q, b: Q) -> arb:
    """ball enclosing [a, b]."""
    return qa(a).union(qa(b))


def _exact_q(x: arb) -> Q:
    if not x.is_finite():
        raise ValueError("non-finite ball")
    man, e = x.man_exp()
    man, e = int(man), int(e)
    return Q(man * 2 ** e) if e >= 0 else Q(man, 2 ** (-e))


def lo(x: arb) -> Q:
    """exact rational lower bound of the ball x (outward)."""
    return _exact_q(x.lower())


def hi(x: arb) -> Q:
    return _exact_q(x.upper())


def mid_q(x: arb) -> Q:
    return _exact_q(x.mid())


def rad_q(x: arb) -> Q:
    return _exact_q(x.rad())


def snap_down(x: Q, bits=SNAP) -> Q:
    s = 2 ** bits
    return Q((x.numerator * s) // x.denominator, s)


def snap_up(x: Q, bits=SNAP) -> Q:
    s = 2 ** bits
    return Q(-((-x.numerator * s) // x.denominator), s)


def _dec(x: Q, sig: int, up: bool) -> str:
    """decimal string with `sig` significant digits, rounded down (up=False) or up."""
    if x == 0:
        return "0"
    neg = x < 0
    a = -x if neg else x
    e = len(str(a.numerator)) - len(str(a.denominator))
    while Q(10) ** e > a:
        e -= 1
    while Q(10) ** (e + 1) <= a:
        e += 1
    k = sig - 1 - e
    v = a * Q(10) ** k
    mag_up = (up and not neg) or ((not up) and neg)
    n = -((-v.numerator) // v.denominator) if mag_up else v.numerator // v.denominator
    s = str(n)
    if k > 0:
        s = s.rjust(k + 1, "0")
        s = s[:-k] + "." + s[-k:]
        s = s.rstrip("0").rstrip(".")
    elif k < 0:
        s = s + "0" * (-k)
    return ("-" if neg else "") + s


def down(x: Q, sig=20) -> str:
    return _dec(x, sig, False)


def up(x: Q, sig=20) -> str:
    return _dec(x, sig, True)


def to_list(x: arb, sig=20):
    return [down(lo(x), sig), up(hi(x), sig)]


def isect(x: arb, y: arb) -> arb:
    try:
        return x.intersection(y)
    except ValueError:
        raise RuntimeError(f"disjoint valid enclosures (internal error): {x} vs {y}")


def sgn(v: arb) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def compare(mine: arb, theirs, name=""):
    """containment / overlap of the Arb enclosure `mine` and a stored [lo, hi] string pair."""
    a, b = Q(theirs[0]), Q(theirs[1])
    x, y = lo(mine), hi(mine)
    return {"arb": [down(x), up(y)], "stored": list(theirs),
            "arb_in_stored": bool(a <= x and y <= b), "stored_in_arb": bool(x <= a and b <= y),
            "overlap": bool(max(a, x) <= min(b, y)),
            "width_arb": float(y - x), "width_stored": float(b - a)}


def compare_iv(x: Q, y: Q, theirs):
    """same as compare for an enclosure given by exact rationals [x, y]."""
    a, b = Q(theirs[0]), Q(theirs[1])
    return {"arb": [down(x), up(y)], "stored": list(theirs),
            "arb_in_stored": bool(a <= x and y <= b), "stored_in_arb": bool(x <= a and b <= y),
            "overlap": bool(max(a, x) <= min(b, y)),
            "width_arb": float(y - x), "width_stored": float(b - a)}


# ----------------------------------------------------------------------------
# the limit law
SQ2PI = (2 * arb.pi()).sqrt()


class Law:
    """G0 with a parameter p.  kind 'rho': p = rho, equal weights.  kind 's': rho fixed,
    w = (1 - eta, eta s, eta (1 - s)) (the allocation family F3S/F3A), p = s."""

    def __init__(self, targets, kind, rho=None, eta=None):
        self.targets = tuple(targets)
        self.m = len(targets)
        self.cs = [4 * (-arb(t)).exp() for t in targets]
        self.kind = kind
        self.rho = None if rho is None else (rho if isinstance(rho, arb) else arb(rho))
        self.eta = None if eta is None else (eta if isinstance(eta, arb) else arb(eta))

    def rho_of(self, p):
        return p if self.kind == "rho" else self.rho

    def weights(self, p):
        if self.kind == "rho":
            return [arb(1) / self.m] * self.m
        e = self.eta
        return [1 - e, e * p, e * (1 - p)]

    def g0(self, T, p, L, deriv=False):
        """power series of G0(T + x) (length L); with deriv also d_p G0 (T + x)."""
        rho = self.rho_of(p)
        mu = 4 * arb_series([-T, arb(-1)], prec=L).exp()
        k = -1 / (2 * rho * rho)
        pref = 1 / (SQ2PI * rho)
        Es, d2s = [], []
        for c in self.cs:
            d = mu - c
            d2 = d * d
            Es.append((d2 * k).exp())
            d2s.append(d2)
        w = self.weights(p)
        G = Es[0] * w[0]
        for j in range(1, self.m):
            G = G + Es[j] * w[j]
        G = G * pref
        if not deriv:
            return G
        if self.kind == "rho":
            r3 = rho * rho * rho
            D = Es[0] * w[0] * (d2s[0] / r3 - 1 / rho)
            for j in range(1, self.m):
                D = D + Es[j] * w[j] * (d2s[j] / r3 - 1 / rho)
            D = D * pref
        else:
            D = (Es[1] - Es[2]) * (self.eta * pref)
        return G, D

    def g0_acb(self, p):
        """G0 as an acb function (entire in t) for thin p, for acb.integral."""
        rho = self.rho_of(p)
        k = -1 / (2 * rho * rho)
        pref = 1 / (SQ2PI * rho)
        w = self.weights(p)
        cs = self.cs

        def f(z, analytic):
            mu = 4 * (-z).exp()
            s = acb(0)
            for c, wj in zip(cs, w):
                d = mu - c
                s = s + (d * d * k).exp() * wj
            return s * pref
        return f


EXTRA = {"G0": 0, "G1": 1, "N": 2, "r0": 1}


def qty_series(q, G):
    if q == "G0":
        return G
    if q == "G1":
        return G.derivative()
    if q == "N":
        G1 = G.derivative()
        G2 = G1.derivative()
        return G2 * G - (G1 * G1) * 2
    if q == "r0":
        return G.derivative() / (G * G)
    raise KeyError(q)


def qty_dseries(q, Z, D):
    """d_p of the quantity, from G0 = Z and d_p G0 = D (series)."""
    if q == "G0":
        return D
    if q == "G1":
        return D.derivative()
    if q == "N":
        Z1, D1 = Z.derivative(), D.derivative()
        Z2, D2 = Z1.derivative(), D1.derivative()
        return D2 * Z + Z2 * D - (Z1 * D1) * 4
    if q == "r0":
        Z1, D1 = Z.derivative(), D.derivative()
        ZZ = Z * Z
        return D1 / ZZ - (Z1 * D) * 2 / (ZZ * Z)
    raise KeyError(q)


def _co(s, n):
    if s.prec < n:
        raise RuntimeError("series too short")
    return [s[k] for k in range(n)]


THICK = arb(2) ** -200


def is_thick(P):
    return P.rad() > THICK


def encl(law, q, T, P, n):
    """enclosures of Q^{(k)}(t)/k!, k < n, for all t in T and all parameters p in P."""
    L = n + EXTRA[q]
    if not is_thick(P):
        return _co(qty_series(q, law.g0(T, P, L)), n)
    pm = P.mid()
    V = _co(qty_series(q, law.g0(T, pm, L)), n)
    Z, D = law.g0(T, P, L, deriv=True)
    naive = _co(qty_series(q, Z), n)
    dq = _co(qty_dseries(q, Z, D), n)
    dP = P - pm
    return [isect(V[k] + dq[k] * dP, naive[k]) for k in range(n)]


def dparam(law, q, T, P):
    """enclosure of d_p Q on T x P (naive ball evaluation)."""
    Z, D = law.g0(T, P, 1 + EXTRA[q], deriv=True)
    return qty_dseries(q, Z, D)[0]


def point(law, q, x: Q, P):
    return encl(law, q, qa(x), P, 1)[0]


def cell(law, q, a: Q, b: Q, P):
    """(value, t-derivative) enclosures of Q on [a, b] x P (order-2 Taylor forms)."""
    J = ball(a, b)
    m = qa((a + b) / 2)
    cm = encl(law, q, m, P, 3)
    cJ = encl(law, q, J, P, 4)
    h = J - m
    val = isect(cm[0] + cm[1] * h + cJ[2] * h * h, cJ[0])
    der = isect(cm[1] + 2 * cm[2] * h + 3 * cJ[3] * h * h, cJ[1])
    return val, der


# ----------------------------------------------------------------------------
# verified zero counting
class ScanError(RuntimeError):
    pass


def scan(law, q, a: Q, b: Q, P, n0=N0):
    a, b = snap_down(a), snap_up(b)
    h = (b - a) / n0
    stack = [(a + k * h, a + (k + 1) * h) for k in range(n0 - 1, -1, -1)]
    leaves = []
    n = 0
    while stack:
        x, y = stack.pop()
        n += 1
        if n > MAX_CELLS:
            raise ScanError(f"{q}: more than {MAX_CELLS} cells")
        val, der = cell(law, q, x, y, P)
        if val > 0 or val < 0:
            leaves.append({"a": x, "b": y, "kind": "pos" if val > 0 else "neg", "val": val, "der": der})
        elif der > 0 or der < 0:
            leaves.append({"a": x, "b": y, "kind": "inc" if der > 0 else "dec", "val": val, "der": der})
        else:
            if y - x < MIN_W:
                raise ScanError(f"{q}: unresolved cell [{float(x)}, {float(y)}]")
            mdl = (x + y) / 2
            stack.append((mdl, y))
            stack.append((x, mdl))
    return leaves


def runs(law, q, leaves, P):
    segs = []
    for lf in leaves:
        if segs and lf["kind"] in ("inc", "dec") and segs[-1]["kind"] == lf["kind"]:
            segs[-1]["b"] = lf["b"]
            segs[-1]["dh"] = segs[-1]["dh"].union(lf["der"])
            segs[-1]["n"] += 1
        else:
            segs.append({"a": lf["a"], "b": lf["b"], "kind": lf["kind"], "dh": lf["der"],
                         "vmax": lf["val"], "n": 1})
    for i, sg in enumerate(segs):
        if sg["kind"] in ("pos", "neg"):
            sg["sl"] = sg["sr"] = 1 if sg["kind"] == "pos" else -1
            continue
        for end, nb in (("a", i - 1), ("b", i + 1)):
            s = sgn(point(law, q, sg[end], P))
            if s == 0 and 0 <= nb < len(segs) and segs[nb]["kind"] in ("pos", "neg"):
                s = 1 if segs[nb]["kind"] == "pos" else -1
            if s == 0:
                raise ScanError(f"{q}: indefinite sign at run end {float(sg[end])}")
            sg["sl" if end == "a" else "sr"] = s
        if (sg["kind"] == "inc" and sg["sl"] > sg["sr"]) or (sg["kind"] == "dec" and sg["sl"] < sg["sr"]):
            raise ScanError(f"{q}: inconsistent run (internal error)")
    return segs


def zero_runs(segs):
    return [s for s in segs if s["kind"] in ("inc", "dec") and s["sl"] != s["sr"]]


def newton(law, q, sg, P, max_it=200):
    """interval Newton enclosure of the zero of a sign-changing monotone run (for all p in P)."""
    a, b = sg["a"], sg["b"]
    X = ball(a, b)
    Dh = sg["dh"]          # encloses F' on the run [a, b] (every cell of the run is monotone)
    stall = 0
    it = 0
    for it in range(max_it):
        mq = mid_q(X)
        if a <= mq <= b:
            # the zero z and m lie in X and in [a, b], so F' on [m, z] is in encl(X) cap Dh
            m = X.mid()
            d = isect(encl(law, q, X, P, 2)[1], Dh)
        else:
            # (ball radii are rounded up, so X may overhang [a, b] by a sliver): use a run end point
            m = qa(min(max(mq, a), b))
            d = Dh
        fm = encl(law, q, m, P, 1)[0]
        if not (d > 0 or d < 0):
            break
        Xn = isect(m - fm / d, X)
        if rad_q(Xn) > Q(9, 10) * rad_q(X):
            stall += 1
            X = Xn
            if stall >= 3:
                break
        else:
            stall = 0
            X = Xn
    return X, it + 1


def sign_encl(law, q, X, P, want, max_depth=16):
    """enclosure of Q on X x P: the naive ball value, or, if that does not have the sign `want`,
    the hull of order-2 Taylor-form cell enclosures on an adaptive partition of [lo X, hi X]."""
    v = encl(law, q, X, P, 1)[0]
    if sgn(v) == want:
        return v
    a, b = snap_down(lo(X)), snap_up(hi(X))
    stack, hull = [(a, b, 0)], None
    while stack:
        x, y, d = stack.pop()
        val, _ = cell(law, q, x, y, P)
        if sgn(val) != want and d < max_depth:
            mdl = (x + y) / 2
            stack += [(mdl, y, d + 1), (x, mdl, d + 1)]
            continue
        hull = val if hull is None else hull.union(val)
    return hull


# ----------------------------------------------------------------------------
# certificate of (D1)-(D4) and the folds for a parameter ball P (thin or thick)
def certify(law, P, want_fold_extras=False):
    t0 = time.time()
    rec = {"ok": False}
    m = law.m
    # (D1)
    lv = scan(law, "G1", TAU, TEND, P)
    sg = runs(law, "G1", lv, P)
    zr = zero_runs(sg)
    types = ["max" if s["kind"] == "dec" else "min" for s in zr]
    g1_tau, g1_T = point(law, "G1", TAU, P), point(law, "G1", TEND, P)
    crit = []
    for s in zr:
        X, nit = newton(law, "G1", s, P)
        c = encl(law, "G0", X, P, 3)
        # X lies in the run, on which the G0'' (= d/dt G0') enclosures s["dh"] are sign-definite
        g2 = isect(2 * c[2], s["dh"])
        if not ((g2 < 0) if s["kind"] == "dec" else (g2 > 0)):
            raise RuntimeError("G0'' sign inconsistent with the run direction")
        crit.append({"type": "max" if s["kind"] == "dec" else "min", "X": X, "G0": c[0], "G0pp": g2,
                     "newton_iterations": nit})
    expect = ["max", "min"] * (m - 1) + ["max"]
    d1 = types == expect and g1_tau > 0 and g1_T < 0
    rec["D1"] = {"holds": bool(d1), "signature": types, "n_cells": len(lv),
                 "G0p_tau": g1_tau, "G0p_T": g1_T, "crit": crit}
    if not d1:
        rec["runtime_s"] = round(time.time() - t0, 3)
        return rec
    maxima = [c for c in crit if c["type"] == "max"]
    minima = [c for c in crit if c["type"] == "min"]
    # (D2): N < 0 on [tau, hi(hat t_1)]
    lv2 = scan(law, "N", TAU, hi(maxima[0]["X"]), P)
    sg2 = runs(law, "N", lv2, P)
    d2 = all(s["sl"] < 0 and s["sr"] < 0 for s in sg2)
    supN = max(hi(lf["val"]) for lf in lv2)
    rec["D2"] = {"holds": bool(d2), "n_cells": len(lv2),
                 "n_cells_value_negative": sum(1 for lf in lv2 if lf["kind"] == "neg"),
                 "sup_N_upper": supN, "interval_hi": hi(maxima[0]["X"])}
    # (D3)
    flanks = []
    d3 = True
    for j in range(2, m + 1):
        mn, mx = minima[j - 2], maxima[j - 1]
        lv3 = scan(law, "N", lo(mn["X"]), hi(mx["X"]), P)
        sg3 = runs(law, "N", lv3, P)
        zr3 = zero_runs(sg3)
        N_min = sign_encl(law, "N", mn["X"], P, 1)
        N_max = sign_encl(law, "N", mx["X"], P, -1)
        ok = len(zr3) == 1 and zr3[0]["kind"] == "dec" and N_min > 0 and N_max < 0
        fr = {"j": j, "n_cells": len(lv3), "n_zeros_of_N": len(zr3), "N_on_check_t": N_min, "N_on_hat_t": N_max}
        if ok:
            Tf, nit = newton(law, "N", zr3[0], P)
            c = encl(law, "r0", Tf, P, 3)
            bj, r0pp = c[0], 2 * c[2]
            ok = bool(r0pp < 0)
            fr.update({"t_fold": Tf, "b_j": bj, "r0pp": r0pp, "newton_iterations": nit,
                       "G0_at_fold": encl(law, "G0", Tf, P, 1)[0]})
            fr["db_dp_envelope"] = dparam(law, "r0", Tf, P)
        fr["holds"] = bool(ok)
        d3 = d3 and ok
        flanks.append(fr)
    rec["D3"] = {"holds": bool(d3), "flanks": flanks}
    # (D4)
    r0tau = point(law, "r0", TAU, P)
    cands = {"left_endpoint": r0tau}
    for f in flanks:
        if "b_j" in f:
            cands[f"flank_{f['j']}"] = f["b_j"]
    rec["candidates"] = cands
    rec["r0_tau"] = r0tau
    if d1 and d2 and d3:
        names = sorted(cands, key=lambda k: lo(cands[k]))
        best = names[0]
        unique = all(hi(cands[best]) < lo(cands[k]) for k in names[1:])
        rec["binding"] = best if unique else None
        rec["binding_candidate"] = best
        rec["binding_unique"] = bool(unique)
        rec["D4"] = bool(unique and best.startswith("flank_"))
        rec["B_top_det"] = cands[best]
        rec["next_candidate"] = names[1]
    rec["ok"] = bool(d1 and d2 and d3 and rec.get("D4", False))
    rec["runtime_s"] = round(time.time() - t0, 3)
    return rec


# ----------------------------------------------------------------------------
# (i) anchor widths of TH8_certificate
def lam0(law, p, X):
    """enclosure of Lambda0 on the ball X (thin p): acb.integral to mid(X) + G0(X)(X - mid X)."""
    f = law.g0_acb(p)
    xm = X.mid()
    val = acb.integral(f, 0, xm).real
    g = encl(law, "G0", X, p, 1)[0]
    return val + g * (X - xm)


def lam0_point(law, p, x: Q):
    return acb.integral(law.g0_acb(p), 0, qa(x)).real


def window_range(law, P, wa: Q, wb: Q, k: int, ok, n0=512, max_depth=24):
    """hull of enclosures of r0^{(k)} on [wa, wb] (order-2 Taylor form per cell); a cell whose
    enclosure fails the predicate `ok` is bisected (up to max_depth), so that a stored bound that is
    true is confirmed.  Also returns a lower bound of min G0 on [wa, wb]."""
    fk = [1, 1, 2, 6, 24, 120, 720, 5040]
    h = (wb - wa) / n0
    stack = [(wa + i * h, wa + (i + 1) * h, 0) for i in range(n0 - 1, -1, -1)]
    hull, gmin, n = None, None, 0
    while stack:
        x, y, dpt = stack.pop()
        J, mm = ball(x, y), qa((x + y) / 2)
        cm = encl(law, "r0", mm, P, k + 2)
        cJ = encl(law, "r0", J, P, k + 3)
        hJ = J - mm
        v = isect(fk[k] * cm[k] + fk[k + 1] * cm[k + 1] * hJ + (fk[k + 2] // 2) * cJ[k + 2] * hJ * hJ,
                  fk[k] * cJ[k])
        if not ok(v) and dpt < max_depth:
            mdl = (x + y) / 2
            stack.append((mdl, y, dpt + 1))
            stack.append((x, mdl, dpt + 1))
            continue
        n += 1
        hull = v if hull is None else hull.union(v)
        g = lo(encl(law, "G0", J, P, 1)[0])
        gmin = g if gmin is None else min(gmin, g)
    return {"hull": hull, "n_cells": n, "G0_min_lower": gmin}


def check_anchor(key, old):
    m = old["m"]
    law = Law(TARGETS[m], "rho")
    P = arb(old["rho_s"])
    rec = certify(law, P)
    out = {"case": key, "m": m, "rho_s": old["rho_s"], "runtime_s": rec["runtime_s"]}
    d1o = old["D1"]
    out["D1"] = {"arb_holds": rec["D1"]["holds"], "stored_holds": d1o["holds"],
                 "arb_signature": rec["D1"]["signature"], "stored_signature": d1o["signature"],
                 "G0p_tau": compare(rec["D1"]["G0p_tau"], d1o["G0p_tau"]),
                 "G0p_T": compare(rec["D1"]["G0p_T"], d1o["G0p_T"]),
                 "critical_points": []}
    for c_arb, c_old in zip(rec["D1"]["crit"], d1o["critical_points"]):
        out["D1"]["critical_points"].append({
            "type_arb": c_arb["type"], "type_stored": c_old["type"],
            "t": compare(c_arb["X"], c_old["t"]), "G0": compare(c_arb["G0"], c_old["G0"]),
            "G0pp": compare(c_arb["G0pp"], c_old["G0pp"]), "newton_iterations": c_arb["newton_iterations"]})
    out["agree_D1"] = bool(rec["D1"]["holds"] == d1o["holds"] and rec["D1"]["signature"] == d1o["signature"])
    if rec["D1"]["holds"] != d1o["holds"] or not all(k in rec for k in ("D2", "D3")) and d1o["holds"]:
        out["certified_arb"], out["certified_stored"] = rec["ok"], old["certified"]
        out["agree"] = False
        out["n_compared_enclosures"] = out["n_overlap"] = out["n_arb_in_stored"] = 0
        return out
    if not d1o["holds"]:
        out["certified_arb"], out["certified_stored"] = rec["ok"], old["certified"]
        out["agree"] = bool(out["agree_D1"] and not rec["ok"] and not old["certified"]
                            and all(c["t"]["overlap"] and c["G0"]["overlap"] and c["G0pp"]["overlap"]
                                    for c in out["D1"]["critical_points"]))
        cmpd = [c[k] for c in out["D1"]["critical_points"] for k in ("t", "G0", "G0pp")]
        cmpd += [out["D1"]["G0p_tau"], out["D1"]["G0p_T"]]
        out["n_compared_enclosures"] = len(cmpd)
        out["n_overlap"] = sum(c["overlap"] for c in cmpd)
        out["n_arb_in_stored"] = sum(c["arb_in_stored"] for c in cmpd)
        return out
    # (D2)
    d2o = old["D2"]
    supN = rec["D2"]["sup_N_upper"]
    out["D2"] = {"arb_holds": rec["D2"]["holds"], "stored_holds": d2o["holds"],
                 "arb_n_cells": rec["D2"]["n_cells"], "arb_all_cells_value_negative":
                 rec["D2"]["n_cells_value_negative"] == rec["D2"]["n_cells"],
                 "arb_sup_N_upper": up(supN), "stored_sup_N_upper": d2o["sup_N_upper_bound"],
                 "stored_bound_confirmed": bool(supN <= Q(d2o["sup_N_upper_bound"]))}
    # (D3)
    out["D3"] = {"arb_holds": rec["D3"]["holds"], "stored_holds": old["D3"]["holds"], "flanks": []}
    for fa, fo in zip(rec["D3"]["flanks"], old["D3"]["flanks"]):
        row = {"j": fa["j"], "n_zeros_arb": fa["n_zeros_of_N"], "n_zeros_stored": fo["n_zeros_of_N"],
               "N_on_check_t": compare(fa["N_on_check_t"], fo["N_on_check_t"]),
               "N_on_hat_t": compare(fa["N_on_hat_t"], fo["N_on_hat_t"])}
        if "t_fold" in fa and "t_fold" in fo:
            row.update({"t_fold": compare(fa["t_fold"], fo["t_fold"]), "b_j": compare(fa["b_j"], fo["b_j"]),
                        "r0pp_at_fold": compare(fa["r0pp"], fo["r0pp_at_fold"]),
                        "G0_at_fold": compare(fa["G0_at_fold"], fo["G0_at_fold"]),
                        "newton_iterations": fa["newton_iterations"]})
        out["D3"]["flanks"].append(row)
    # candidates, (D4), B_top^det
    out["candidates"] = {k: compare(v, old["candidates"][k]) for k, v in rec["candidates"].items()
                         if k in old["candidates"]}
    out["binding_arb"], out["binding_stored"] = rec.get("binding"), old.get("binding")
    out["D4_arb"], out["D4_stored"] = rec.get("D4"), old.get("certified_D4")
    out["B_top_det"] = compare(rec["B_top_det"], old["B_top_det"])
    nxt = rec["candidates"][rec["next_candidate"]]
    gap = nxt - rec["B_top_det"]
    out["gap_to_next_candidate"] = {"arb_lower": down(lo(gap)), "stored_lower": old["gap_to_next_candidate_lower_bound"],
                                    "stored_bound_confirmed": bool(Q(old["gap_to_next_candidate_lower_bound"]) <= lo(gap))}
    out["certified_arb"], out["certified_stored"] = rec["ok"], old["certified"]
    # fold constants (binding flank)
    fo = old.get("fold")
    if fo and rec["ok"]:
        k = int(rec["binding"].split("_")[1])
        fa = [f for f in rec["D3"]["flanks"] if f["j"] == k][0]
        Tf, b = fa["t_fold"], fa["b_j"]
        fx = {}
        fx["a_f"] = compare(-fa["r0pp"], fo["a_f=-r0pp(t_f)"])
        kr = fo["krawczyk"]
        fx["krawczyk_K_t_contains_arb_t_fold"] = compare(Tf, kr["K_t"])["arb_in_stored"]
        fx["krawczyk_K_B_contains_arb_b"] = compare(b, kr["K_B"])["arb_in_stored"]
        lamf = lam0(law, P, Tf)
        fx["Lambda0_at_t_f"] = compare(lamf, fo["Lambda0_at_t_f"])
        G0f = fa["G0_at_fold"]
        gf = (-b * lamf).exp() * G0f * G0f
        fx["g_at_fold"] = compare(gf, fo["g_at_fold=exp(-b Lambda0) G0^2"])
        lam_tau, lam_T = lam0_point(law, P, TAU), lam0_point(law, P, TEND)
        fx["Lambda0_tau"] = compare(lam_tau, fo["Lambda0_tau_T"][0])
        fx["Lambda0_T"] = compare(lam_T, fo["Lambda0_tau_T"][1])

        def S(lam):
            return (-b * lam).exp()
        if m == 2:
            mn = [c for c in rec["D1"]["crit"] if c["type"] == "min"][0]
            lam_min = lam0(law, P, mn["X"])
            mf = fo["masses_fold_cut_[tau,t_f],[t_f,T]"]
            mnom = fo["masses_nominal_cut_[tau,check_t],[check_t,T]"]
            fx["mass_fold_cut_early"] = compare(S(lam_tau) - S(lamf), mf[0])
            fx["mass_fold_cut_late"] = compare(S(lamf) - S(lam_T), mf[1])
            fx["mass_nominal_cut_early"] = compare(S(lam_tau) - S(lam_min), mnom[0])
            fx["mass_nominal_cut_late"] = compare(S(lam_min) - S(lam_T), mnom[1])
        else:
            fx["late_mass_fold_cut"] = compare(S(lamf) - S(lam_T), fo["late_mass_fold_cut_[t_f,T]"])
        # concavity window: -r0'' in [A1, A2] and |r0'''| <= bound on the stored window (exact decimals)
        cw = fo["concavity_window"]
        wa, wb = Q(cw["window_exact"][0]), Q(cw["window_exact"][1])
        A1, A2, R3 = Q(cw["A1_lower_bound_of_-r0pp"]), Q(cw["A2_upper_bound_of_-r0pp"]), Q(cw["r0ppp_bound"])
        h2 = window_range(law, P, wa, wb, 2, lambda v: A1 <= -hi(v) and -lo(v) <= A2)
        h3 = window_range(law, P, wa, wb, 3, lambda v: max(abs(lo(v)), abs(hi(v))) <= R3)
        mn2, mx2 = -hi(h2["hull"]), -lo(h2["hull"])
        mx3 = max(abs(lo(h3["hull"])), abs(hi(h3["hull"])))
        G0min = h2["G0_min_lower"]
        ncell = [h2["n_cells"], h3["n_cells"]]
        fx["concavity_window"] = {
            "window": [cw["window_exact"][0][:22], cw["window_exact"][1][:22]], "n_cells_arb": ncell,
            "arb_range_-r0pp": [down(mn2), up(mx2)], "stored_A1_A2": [cw["A1_lower_bound_of_-r0pp"],
                                                                   cw["A2_upper_bound_of_-r0pp"]],
            "stored_A1_A2_confirmed": bool(A1 <= mn2 and mx2 <= A2),
            "arb_sup_abs_r0ppp": up(mx3), "stored_r0ppp_bound": cw["r0ppp_bound"],
            "stored_r0ppp_bound_confirmed": bool(mx3 <= R3)}
        dI = arb(cw["delta"])
        s_I = qa(A2) * dI * dI
        # s is recomputed from the PRINTED (outward-rounded) A2, which exceeds the internal binary A2 by
        # <= 1e-17 relative, so it is not an enclosure of the stored s; only closeness and s < b are checked
        s_st = fo["beta_box_halfwidth_s=A2*delta^2"]
        fx["s=A2*delta^2_from_printed_A2"] = {
            "arb": to_list(s_I), "stored": s_st,
            "rel_deviation_upper": float(abs(hi(s_I) - Q(s_st[1])) / Q(s_st[1])),
            "stored_s_close": bool(abs(hi(s_I) - Q(s_st[1])) <= Q(1, 10 ** 15) * Q(s_st[1]))}
        fx["s_upper_lt_b_lower"] = bool(hi(s_I) < lo(b))
        # lower bound of g = exp(-beta Lambda0) G0^2 on window x [b - s, b + s] (Lambda0 increasing)
        lam_wb = lam0_point(law, P, wb)
        gmin = (-(qa(hi(b)) + qa(hi(s_I))) * qa(hi(lam_wb))).exp() * qa(G0min) * qa(G0min)
        fx["g_min_on_fold_box"] = {"arb_lower": down(lo(gmin)), "stored_lower": fo["g_min_on_fold_box_lower_bound"],
                                   "stored_bound_confirmed": bool(Q(fo["g_min_on_fold_box_lower_bound"]) <= lo(gmin))}
        out["fold"] = fx
    # agreement flags
    cmp_entries = []

    def walk(o):
        if isinstance(o, dict):
            if "overlap" in o and "arb_in_stored" in o:
                cmp_entries.append(o)
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(out)
    confirmed = []

    def walk2(o):
        if isinstance(o, dict):
            for kk, v in o.items():
                if kk.startswith("stored_") and kk.endswith("_confirmed") or kk == "stored_bound_confirmed":
                    confirmed.append(bool(v))
                elif kk.startswith("krawczyk_K_") or kk == "s_upper_lt_b_lower":
                    confirmed.append(bool(v))
                else:
                    walk2(v)
        elif isinstance(o, list):
            for v in o:
                walk2(v)
    walk2(out)
    out["n_compared_enclosures"] = len(cmp_entries)
    out["n_arb_in_stored"] = sum(e["arb_in_stored"] for e in cmp_entries)
    out["n_overlap"] = sum(e["overlap"] for e in cmp_entries)
    out["n_stored_bounds_confirmed"] = sum(confirmed)
    out["n_stored_bounds_checked"] = len(confirmed)
    out["agree"] = bool(out["n_overlap"] == len(cmp_entries) and all(confirmed)
                        and rec["ok"] == old["certified"] and rec.get("binding") == old.get("binding")
                        and rec["D1"]["signature"] == d1o["signature"])
    return out


# ----------------------------------------------------------------------------
# (ii) fold-curve rho-boxes
DELTA = 4 * ((-arb(1)).exp() - (-arb("2.5")).exp())


def thin_fold(rho_ball):
    """thin certificate at rho (a small ball): B, t_f, critical points, r0(tau)."""
    law = Law(TARGETS[2], "rho")
    rec = certify(law, rho_ball)
    if not rec["ok"] or rec["binding"] != "flank_2":
        return None, rec
    f = rec["D3"]["flanks"][0]
    return {"B": f["b_j"], "t_f": f["t_fold"], "r0_tau": rec["r0_tau"], "a_f": -f["r0pp"],
            "crit": [c["X"] for c in rec["D1"]["crit"]]}, rec


def certify_rho_box(r_lo: Q, r_hi: Q, depth=0, max_depth=6):
    """parametric certificate on [r_lo, r_hi] (bisected on failure); returns list of leaves."""
    law = Law(TARGETS[2], "rho")
    R = ball(r_lo, r_hi)
    try:
        rec = certify(law, R)
        err = None
    except (ScanError, RuntimeError) as e:
        rec, err = None, str(e)
    ok = rec is not None and rec["ok"] and rec.get("binding") == "flank_2"
    slope = None
    if ok:
        f = rec["D3"]["flanks"][0]
        slope = f["db_dp_envelope"]
        ok = bool(slope < 0)
    if ok or depth >= max_depth:
        return [{"r": [r_lo, r_hi], "ok": bool(ok), "rec": rec, "slope": slope, "error": err, "depth": depth}]
    rm = (r_lo + r_hi) / 2
    return certify_rho_box(r_lo, rm, depth + 1, max_depth) + certify_rho_box(rm, r_hi, depth + 1, max_depth)


def tube_logB(box, a_ball):
    tb = box["fold"]["tube"]
    Kbm = ball(Q(tb["logB_at_a_mid"][0]), Q(tb["logB_at_a_mid"][1]))
    sl = ball(Q(tb["dlogB_da_on_box"][0]), Q(tb["dlogB_da_on_box"][1]))
    am = ball(Q(tb["a_mid"][0]), Q(tb["a_mid"][1]))
    return Kbm + sl * (a_ball - am)


# thin-point or exact-range enclosures (expected INSIDE the stored ones); the *_param enclosures are thick
# parametric enclosures over the whole box (expected only to OVERLAP the stored ones)
TIGHT_KEYS = ("B_range_over_box", "B_at_rho_lo", "B_at_rho_hi", "B_at_a_mid", "t_fold_at_a_mid", "logB_at_a_mid")


def check_box(args):
    """recheck one stored leaf box of fold_curve_m2_boxes.json."""
    ctx.prec = PREC
    label, box = args
    t0 = time.time()
    r_lo, r_hi = Q(box["rho_box"][0]), Q(box["rho_box"][1])
    out = {"range": label, "rho_box": box["rho_box"], "depth_stored": box["depth"]}
    leaves = certify_rho_box(r_lo, r_hi)
    param_ok = all(lf["ok"] for lf in leaves)
    out["arb_parametric_certified"] = bool(param_ok)
    out["arb_n_subboxes"] = len(leaves)
    out["arb_max_subbox_depth"] = max(lf["depth"] for lf in leaves)
    if not param_ok:
        out["arb_fail"] = [lf["error"] or "check failed" for lf in leaves if not lf["ok"]][:3]
    fo = box["fold"]
    # thin end points and interior points
    ends = {}
    for name, r in (("rho_lo", r_lo), ("rho_hi", r_hi)):
        th, rc = thin_fold(qa(r))
        ends[name] = th
    if ends["rho_lo"] is None or ends["rho_hi"] is None:
        out["thin_end_certificate_failed"] = True
        out["agree"] = False
        return out
    Bl, Bh = ends["rho_lo"]["B"], ends["rho_hi"]["B"]
    # B over the box = [B(rho_hi), B(rho_lo)] by the certified strict decrease (envelope identity)
    out["B_range_over_box"] = compare_iv(lo(Bh), hi(Bl), fo["B_top_det"])
    out["B_at_rho_lo"] = compare(Bl, fo["tube"]["ends"]["at_rho_lo"]["B"])
    out["B_at_rho_hi"] = compare(Bh, fo["tube"]["ends"]["at_rho_hi"]["B"])
    tfs = [ends["rho_lo"]["t_f"], ends["rho_hi"]["t_f"]]
    # parametric (thick) enclosures, hull over the Arb sub-boxes
    if param_ok:
        Tf = Bth = sl = af = r0t = None
        crit = [None, None, None]
        for lf in leaves:
            f = lf["rec"]["D3"]["flanks"][0]
            Tf = f["t_fold"] if Tf is None else Tf.union(f["t_fold"])
            Bth = f["b_j"] if Bth is None else Bth.union(f["b_j"])
            sl = lf["slope"] if sl is None else sl.union(lf["slope"])
            af = -f["r0pp"] if af is None else af.union(-f["r0pp"])
            r0t = lf["rec"]["r0_tau"] if r0t is None else r0t.union(lf["rec"]["r0_tau"])
            for i, c in enumerate(lf["rec"]["D1"]["crit"]):
                crit[i] = c["X"] if crit[i] is None else crit[i].union(c["X"])
        out["t_fold_param"] = compare(Tf, fo["t_fold"])
        out["B_param_meanvalue"] = compare(Bth, fo["B_top_det"])
        out["dB_drho_param"] = compare(sl, fo["tube"]["dB_drho_on_box"])
        out["dB_drho_negative_arb"] = bool(sl < 0)
        out["a_f_param"] = compare(af, fo["a_f=-r0pp(t_f)"])
        out["a_f_positive_arb"] = bool(af > 0)
        cpt = box["D1"]["critical_points_t"]
        out["hat_t1_param"] = compare(crit[0], cpt["hat_t1"])
        out["check_t1_param"] = compare(crit[1], cpt["check_t1"])
        out["hat_t2_param"] = compare(crit[2], cpt["hat_t2"])
        # log r0(tau) - log b_2 on the box, with b_2 <= B(rho_lo) (certified decrease)
        lgap = r0t.log() - qa(hi(Bl)).log()
        out["D4_log_gap"] = {"arb_lower": down(lo(lgap)), "stored_lower": box["D4"]["log_gap_lower_bound"],
                             "arb_positive": bool(lgap > 0)}
    # thin values vs stored hulls
    thin_rows = []
    pts = [("rho_lo", r_lo), ("q1", r_lo + (r_hi - r_lo) / 4), ("q2", (r_lo + r_hi) / 2),
           ("q3", r_lo + 3 * (r_hi - r_lo) / 4), ("rho_hi", r_hi)]
    bh = [Q(x) for x in fo["B_top_det"]]
    th_ = [Q(x) for x in fo["t_fold"]]
    cpt = box["D1"]["critical_points_t"]
    for name, r in pts:
        th = ends[name] if name in ends else thin_fold(qa(r))[0]
        if th is None:
            thin_rows.append({"point": name, "rho": str(r), "thin_certificate": False})
            continue
        a_r = DELTA / (2 * qa(r))
        tube = tube_logB(box, a_r).exp()
        B = th["B"]
        thin_rows.append({
            "point": name, "rho": down(r, 12), "B_arb": to_list(B, 12),
            "B_in_stored_hull": bool(bh[0] <= lo(B) and hi(B) <= bh[1]),
            "B_in_stored_tube": bool(lo(tube) <= lo(B) and hi(B) <= hi(tube)),
            "t_f_in_stored_hull": bool(th_[0] <= lo(th["t_f"]) and hi(th["t_f"]) <= th_[1]),
            "crit_in_stored_hulls": all(Q(cpt[k][0]) <= lo(x) and hi(x) <= Q(cpt[k][1])
                                        for k, x in zip(("hat_t1", "check_t1", "hat_t2"), th["crit"])),
            "D4_log_gap_ge_stored_lower": bool(lo(th["r0_tau"].log() - B.log()) >= Q(box["D4"]["log_gap_lower_bound"]))})
    out["thin_points"] = thin_rows
    # stored thin-midpoint tube anchor: B and t_f at a_mid (exact dyadic), rho = Delta/(2 a_mid)
    tb = fo["tube"]
    rho_mid = DELTA / (2 * qa(Q(tb["a_mid_exact"])))
    thm, _ = thin_fold(rho_mid)
    if thm is not None:
        out["B_at_a_mid"] = compare(thm["B"], tb["B_at_a_mid"])
        out["t_fold_at_a_mid"] = compare(thm["t_f"], tb["t_fold_at_a_mid"])
        out["logB_at_a_mid"] = compare(thm["B"].log(), tb["logB_at_a_mid"])
    cmpk = [k for k, v in out.items() if isinstance(v, dict) and "overlap" in v]
    out["n_compared_enclosures"] = len(cmpk)
    out["n_overlap"] = sum(out[k]["overlap"] for k in cmpk)
    out["n_arb_in_stored"] = sum(out[k]["arb_in_stored"] for k in cmpk)
    tight = [k for k in cmpk if k in TIGHT_KEYS]
    out["n_tight_compared"] = len(tight)
    out["n_tight_arb_in_stored"] = sum(out[k]["arb_in_stored"] for k in tight)
    for k in cmpk:                                   # compact record (stored values: see the source file)
        v = out[k]
        out[k] = {"arb": v["arb"], "arb_in_stored": v["arb_in_stored"], "overlap": v["overlap"],
                  "width_ratio_arb_over_stored": (v["width_arb"] / v["width_stored"]) if v["width_stored"] else None}
    thin_ok = all(r.get("B_in_stored_hull") and r.get("B_in_stored_tube") and r.get("t_f_in_stored_hull")
                  and r.get("crit_in_stored_hulls") and r.get("D4_log_gap_ge_stored_lower") for r in thin_rows)
    out["thin_points_all_inside_stored"] = bool(thin_ok)
    out["agree"] = bool(param_ok and out["n_overlap"] == len(cmpk) and thin_ok
                        and out["B_range_over_box"]["arb_in_stored"]
                        and out["B_at_rho_lo"]["arb_in_stored"] and out["B_at_rho_hi"]["arb_in_stored"])
    out["runtime_s"] = round(time.time() - t0, 2)
    return out


def sample_boxes(boxes, n_ext, n_req, all_boxes):
    ext, req = boxes["extension_towards_rho_c"], boxes["required_range"]
    if all_boxes:
        return [("required", i) for i in range(len(req))] + [("extension", i) for i in range(len(ext))]
    rng = random.Random(SEED)
    wid = [Q(b["rho_box"][1]) - Q(b["rho_box"][0]) for b in ext]
    wmin = min(wid)
    must_ext = sorted({0, len(ext) - 1} | {i for i, b in enumerate(ext) if b["depth"] > 0}
                      | {i for i, w in enumerate(wid) if w == wmin})
    rest = [i for i in range(len(ext)) if i not in must_ext]
    pick_ext = sorted(set(must_ext) | set(rng.sample(rest, max(0, n_ext - len(must_ext)))))
    must_req = [0, len(req) - 1]
    rest_r = [i for i in range(len(req)) if i not in must_req]
    pick_req = sorted(set(must_req) | set(rng.sample(rest_r, max(0, n_req - len(must_req)))))
    return [("required", i) for i in pick_req] + [("extension", i) for i in pick_ext]


# ----------------------------------------------------------------------------
# (iii) codimension-2 switch of the F3S family
def s_of_q(q):
    return q / (1 + q)


def f3s_law():
    return Law(TARGETS[3], "s", rho="0.2", eta="0.001")


def b23(law, S):
    rec = certify(law, S)
    if not (rec["D1"]["holds"] and rec["D2"]["holds"] and rec["D3"]["holds"]):
        raise RuntimeError("F3S (D1)-(D3) not certified")
    fl = {f["j"]: f for f in rec["D3"]["flanks"]}
    return rec, fl[2], fl[3]


def check_switch(sw_old):
    t0 = time.time()
    law = f3s_law()
    out = {}
    # thick certificate on the initial bracket (split if needed); dD/ds > 0 on every piece
    q_lo, q_hi = Q(sw_old["bracket_initial"][0]), Q(sw_old["bracket_initial"][1])
    pieces = [(q_lo, q_hi)]
    done = []
    for _ in range(8):
        nxt = []
        for a, b in pieces:
            S = s_of_q(qa(a)).union(s_of_q(qa(b)))
            try:
                rec, f2, f3 = b23(law, S)
                dD = f2["db_dp_envelope"] - f3["db_dp_envelope"]
                mx = f2["b_j"].union(f3["b_j"])
                if dD > 0 and rec["r0_tau"] > mx:
                    done.append((a, b, dD, rec))
                    continue
            except (ScanError, RuntimeError):
                pass
            mdl = (a + b) / 2
            nxt += [(a, mdl), (mdl, b)]
        pieces = nxt
        if not pieces:
            break
    out["bracket"] = sw_old["bracket_initial"]
    out["bracket_thick_certified"] = not pieces
    out["bracket_n_pieces"] = len(done)
    dDh = None
    for a, b, dD, rec in done:
        dDh = dD if dDh is None else dDh.union(dD)
    if dDh is not None:
        out["dD_ds_on_bracket"] = compare(dDh, sw_old["dD_ds_on_bracket"])
        out["D_strictly_increasing_on_bracket_arb"] = bool(dDh > 0 and not pieces)
    # signs at the stored exact end points
    ev = {}
    for name in ("left", "right"):
        so = sw_old["sign_evidence_at_ends"][name]
        S = s_of_q(qa(Q(so["q_exact"])))
        rec, f2, f3 = b23(law, S)
        D = f2["b_j"] - f3["b_j"]
        ev[name] = {"q_exact": so["q_exact"][:24], "b2": compare(f2["b_j"], so["b2"]), "b3": compare(f3["b_j"], so["b3"]),
                    "sign_D_arb": sgn(D)}
    out["sign_evidence_at_ends"] = ev
    out["sign_change_arb"] = bool(ev["left"]["sign_D_arb"] == -1 and ev["right"]["sign_D_arb"] == 1)
    # interval Newton in s on D(s) = b2(s) - b3(s)
    S = s_of_q(qa(q_lo)).union(s_of_q(qa(q_hi)))
    hist = []
    for it in range(60):
        sm = S.mid()
        _, f2m, f3m = b23(law, sm)
        Dm = f2m["b_j"] - f3m["b_j"]
        try:
            _, f2, f3 = b23(law, S)
            dD = f2["db_dp_envelope"] - f3["db_dp_envelope"]
        except (ScanError, RuntimeError):
            dD = dDh
        if dDh is not None:
            dD = isect(dD, dDh)
        if not dD > 0:
            break
        Sn = isect(sm - Dm / dD, S)
        hist.append(float(rad_q(Sn)))
        if rad_q(Sn) > Q(1, 2) * rad_q(S):
            S = Sn
            if len(hist) > 3 and hist[-1] >= hist[-2]:
                break
        S = Sn
        if rad_q(S) < Q(1, 10 ** 60):
            break
    out["newton_iterations"] = len(hist)
    out["newton_radius_history"] = hist
    rec, f2, f3 = b23(law, S)
    # b_j over S by the mean-value form in s about s_m
    sm = S.mid()
    _, f2m, f3m = b23(law, sm)
    b2 = isect(f2m["b_j"] + f2["db_dp_envelope"] * (S - sm), f2["b_j"])
    b3 = isect(f3m["b_j"] + f3["db_dp_envelope"] * (S - sm), f3["b_j"])
    Bs = isect(b2, b3)
    qS = S / (1 - S)
    eta = arb("0.001")
    out["s_star_arb"] = to_list(S, 30)
    out["q_star"] = compare(qS, sw_old["q_star"])
    out["q_star_vs_internal_isolating_bracket"] = compare(qS, sw_old["internal_isolating_bracket"])
    out["B_star"] = compare(Bs, sw_old["B_star"])
    out["t_f2_star"] = compare(f2["t_fold"], sw_old["t_f2_star"])
    out["t_f3_star"] = compare(f3["t_fold"], sw_old["t_f3_star"])
    out["r0_tau"] = compare(rec["r0_tau"], sw_old["r0_tau"])
    out["w_star"] = {"w1": compare(1 - eta + 0 * S, sw_old["w_star"]["w1"]),
                     "w2": compare(eta * S, sw_old["w_star"]["w2"]),
                     "w3": compare(eta * (1 - S), sw_old["w_star"]["w3"])}
    af2, af3 = -f2["r0pp"], -f3["r0pp"]
    out["a_f2"] = {"arb": to_list(af2), "stored_lower": sw_old["a_f2_lower"],
                   "stored_bound_confirmed": bool(Q(sw_old["a_f2_lower"]) <= lo(af2))}
    out["a_f3"] = {"arb": to_list(af3), "stored_lower": sw_old["a_f3_lower"],
                   "stored_bound_confirmed": bool(Q(sw_old["a_f3_lower"]) <= lo(af3))}
    out["left_endpoint_not_binding_at_switch"] = bool(rec["r0_tau"] > Bs)
    k4 = sw_old["krawczyk_4d"]
    out["krawczyk_4d_box_contains_arb"] = {
        "t2": compare(f2["t_fold"], k4["t2"]), "t3": compare(f3["t_fold"], k4["t3"]),
        "B": compare(Bs, k4["B"]), "s": compare(S, k4["s"]), "q": compare(qS, k4["q"])}
    # binding on either side of the bracket (thin spot checks inside the certified F3S range)
    side = []
    for qv in ("0.3366", "0.6", "1.0", "1.3", "1.5", "1.52", "1.57", "1.6", "2.0", "2.5", "2.864"):
        rc = certify(law, s_of_q(arb(qv)))
        side.append({"q": qv, "certified_D1_D4": rc["ok"], "binding": rc.get("binding")})
    out["binding_spot_checks"] = side
    out["binding_left_flank2_right_flank3"] = bool(
        all(r["binding"] == "flank_2" for r in side if Q(r["q"]) <= q_lo)
        and all(r["binding"] == "flank_3" for r in side if Q(r["q"]) >= q_hi))
    cmp_entries = []

    def walk(o):
        if isinstance(o, dict):
            if "overlap" in o and "arb_in_stored" in o:
                cmp_entries.append(o)
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(out)
    out["n_compared_enclosures"] = len(cmp_entries)
    out["n_overlap"] = sum(e["overlap"] for e in cmp_entries)
    out["n_arb_in_stored"] = sum(e["arb_in_stored"] for e in cmp_entries)
    out["agree"] = bool(out["n_overlap"] == len(cmp_entries) and out["sign_change_arb"]
                        and out.get("D_strictly_increasing_on_bracket_arb", False)
                        and out["a_f2"]["stored_bound_confirmed"] and out["a_f3"]["stored_bound_confirmed"]
                        and out["left_endpoint_not_binding_at_switch"] and out["binding_left_flank2_right_flank3"]
                        and out["q_star"]["arb_in_stored"] and out["B_star"]["arb_in_stored"])
    out["runtime_s"] = round(time.time() - t0, 2)
    return out


# ----------------------------------------------------------------------------
def dumps_hybrid(res):
    """indent=1 JSON, except that each fold-curve box record is written on one line (file size)."""
    rows = res.get("fold_curve_boxes", {}).get("boxes")
    if not rows:
        return json.dumps(res, indent=1)
    tag = "__BOX_ROWS__"
    res2 = dict(res)
    res2["fold_curve_boxes"] = dict(res["fold_curve_boxes"], boxes=tag)
    txt = json.dumps(res2, indent=1)
    body = "[\n" + ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in rows) + "\n  ]"
    return txt.replace(json.dumps(tag), body, 1)


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-ext", type=int, default=44)
    ap.add_argument("--n-req", type=int, default=20)
    ap.add_argument("--sample", action="store_true",
                    help="check a seeded sample (--n-ext of the 347 extension leaves incl. end and bisected boxes, "
                         "--n-req of the 250 required leaves) instead of all 597 leaf boxes")
    ap.add_argument("--only", choices=["anchors", "boxes", "switch"], default=None)
    args = ap.parse_args(argv)
    workers = max(1, min(3, args.workers))
    t0 = time.time()
    res = {"script": "code/fb_v3_arb_recheck.py",
           "item": "V3: independent Arb (python-flint) recheck of the Prop. 4 interval certificates",
           "library": {"python_flint": flint.__version__, "FLINT": getattr(flint, "__FLINT_VERSION__", None),
                       "arithmetic": f"Arb (FLINT arb/acb modules) midpoint-radius balls, {PREC}-bit "
                       "midpoints, rigorous (outward) error propagation; acb.integral for Lambda0"},
           "deterministic": True, "sample_seed": SEED,
           "comparison_semantics": (
               "every compared quantity has an Arb enclosure and a stored mpmath.iv enclosure (decimal strings "
               "rounded outward); both contain the same true value(s), so they MUST overlap; arb_in_stored means "
               "the Arb ball lies inside the stored interval (exact rational comparison). *_confirmed flags: a "
               "stored one-sided certified bound is implied by the Arb enclosure. Arb decimals are rounded "
               "outward to 20 significant digits.")}
    parts = {"anchors", "boxes", "switch"} if args.only is None else {args.only}
    if OUT.exists():
        try:
            old_res = json.loads(OUT.read_text())
            for k in ("anchors", "fold_curve_boxes", "codim2_switch"):
                if k in old_res:
                    res[k] = old_res[k]
        except Exception:
            pass
    if "anchors" in parts:
        th8 = json.loads(TH8_JSON.read_text())["cases"]
        rows = {}
        for key, old in th8.items():
            rows[key] = check_anchor(key, old)
            print(key, "agree", rows[key]["agree"], rows[key].get("B_top_det", {}).get("arb"),
                  rows[key]["runtime_s"], flush=True)
        n_cert = sum(1 for v in th8.values() if v["certified"])
        res["anchors"] = {
            "source": "artifacts/data/exact_m_fixed_budget/TH8_certificate/certificate.json",
            "n_cases": len(rows), "n_certified_stored": n_cert,
            "n_certified_arb": sum(1 for v in rows.values() if v["certified_arb"]),
            "all_agree": all(v["agree"] for v in rows.values()),
            "all_B_top_det_arb_in_stored": all(v["B_top_det"]["arb_in_stored"] for v in rows.values()
                                               if "B_top_det" in v),
            "n_enclosures_compared": sum(v["n_compared_enclosures"] for v in rows.values()),
            "n_enclosures_overlap": sum(v["n_overlap"] for v in rows.values()),
            "n_enclosures_arb_in_stored": sum(v["n_arb_in_stored"] for v in rows.values()),
            "cases": rows}
    if "boxes" in parts:
        boxes = json.loads(BOXES_JSON.read_text())
        pick = sample_boxes(boxes, args.n_ext, args.n_req, not args.sample)
        key = {"required": "required_range", "extension": "extension_towards_rho_c"}
        jobs = [(f"{lab}[{i}]", boxes[key[lab]][i]) for lab, i in pick]
        if workers > 1:
            import multiprocessing as mpc
            with mpc.get_context("spawn").Pool(workers) as pool:
                rows = pool.map(check_box, jobs, chunksize=1)
        else:
            rows = [check_box(j) for j in jobs]
        for r in rows:
            print(r["range"], r["rho_box"], "agree", r["agree"], r.get("runtime_s"), flush=True)
        ext_n = sum(1 for lab, _ in pick if lab == "extension")
        req_n = sum(1 for lab, _ in pick if lab == "required")
        res["fold_curve_boxes"] = {
            "source": "artifacts/data/exact_m_fixed_budget/V2_bifurcation/fold_curve_m2_boxes.json",
            "selection": ("all 597 leaf boxes (250 of [0.15,0.40] and 347 of [0.40,0.5715])" if not args.sample else
                          f"extension [0.40,0.5715] (347 leaves): both end boxes + every bisected (depth > 0) box "
                          f"incl. the narrowest (width 1/16000) + seeded random, {ext_n} boxes; required "
                          f"[0.15,0.40] (250 leaves): both end boxes + seeded random, {req_n} boxes"),
            "n_boxes_checked": len(rows), "n_extension": ext_n, "n_required": req_n,
            "n_arb_parametric_certified": sum(r["arb_parametric_certified"] for r in rows),
            "n_agree": sum(r["agree"] for r in rows),
            "all_agree": all(r["agree"] for r in rows),
            "all_B_range_arb_in_stored_hull": all(r.get("B_range_over_box", {}).get("arb_in_stored", False)
                                                  for r in rows),
            "all_tube_ends_arb_in_stored": all(r.get("B_at_rho_lo", {}).get("arb_in_stored", False)
                                               and r.get("B_at_rho_hi", {}).get("arb_in_stored", False) for r in rows),
            "all_thin_points_inside_stored": all(r.get("thin_points_all_inside_stored", False) for r in rows),
            "n_enclosures_compared": sum(r.get("n_compared_enclosures", 0) for r in rows),
            "n_enclosures_overlap": sum(r.get("n_overlap", 0) for r in rows),
            "n_enclosures_arb_in_stored": sum(r.get("n_arb_in_stored", 0) for r in rows),
            "n_tight_enclosures_compared": sum(r.get("n_tight_compared", 0) for r in rows),
            "n_tight_enclosures_arb_in_stored": sum(r.get("n_tight_arb_in_stored", 0) for r in rows),
            "n_thin_point_certificates": sum(len(r.get("thin_points", [])) for r in rows),
            "B_range_width_ratio_arb_over_stored_hull": [
                min(r["B_range_over_box"]["width_ratio_arb_over_stored"] for r in rows),
                max(r["B_range_over_box"]["width_ratio_arb_over_stored"] for r in rows)],
            "note_counts": ("per box 13 enclosures are compared: 6 tight (exact B range over the box from the thin "
                            "end certificates and the certified decrease; thin B at rho_lo, rho_hi; B, log B, t_f at "
                            "the stored dyadic a_mid) which must lie INSIDE the stored ones, and 7 thick parametric "
                            "Arb enclosures over the whole box (t_f, b_2, dB/drho, a_f, hat t_1, check t_1, hat t_2; "
                            "t-coordinates, mean-value form in rho) which need only OVERLAP the stored ones; "
                            "5 thin certificates per box (rho_lo, 1/4, 1/2, 3/4, rho_hi) are tested against the stored "
                            "B hull, the stored thin tube, the t_f hull, the critical-point hulls and the D4 gap bound"),
            "max_rel_width_B_tube_end_stored": max(float((Q(b["fold"]["tube"]["ends"]["at_rho_hi"]["B"][1])
                                                          - Q(b["fold"]["tube"]["ends"]["at_rho_hi"]["B"][0]))
                                                         / Q(b["fold"]["tube"]["ends"]["at_rho_hi"]["B"][0]))
                                                   for _, b in jobs),
            "boxes": rows}
        if not args.sample:
            for nm in ("required_range", "extension_towards_rho_c"):
                bx = boxes[nm]
                res["fold_curve_boxes"][f"{nm}_contiguous_cover"] = bool(
                    all(bx[i]["rho_box"][1] == bx[i + 1]["rho_box"][0] for i in range(len(bx) - 1)))
            res["fold_curve_boxes"]["cover_exact"] = [boxes["required_range"][0]["rho_box"][0],
                                                      boxes["required_range"][-1]["rho_box"][1],
                                                      boxes["extension_towards_rho_c"][-1]["rho_box"][1]]
    if "switch" in parts:
        sw_old = json.loads(CODIM2_JSON.read_text())["codim2_switch_F3S"]
        sw = check_switch(sw_old)
        print("switch agree", sw["agree"], sw["q_star"]["arb"], sw["B_star"]["arb"], flush=True)
        res["codim2_switch"] = {"source": "artifacts/data/exact_m_fixed_budget/V2_bifurcation/codim2_switch.json"
                                          "#codim2_switch_F3S", **sw}
    if args.only is None:
        # negative controls: stored enclosures shifted just off the true value must be flagged
        import copy
        th8 = json.loads(TH8_JSON.read_text())["cases"]
        a = copy.deepcopy(th8["m2_rho0.3"])
        a["B_top_det"] = ["8.2246647455983020", "8.2246647455983030"]     # true value 8.22466474559830170...
        boxes = json.loads(BOXES_JSON.read_text())
        bx = copy.deepcopy(boxes["required_range"][100])
        hb = [Q(x) for x in bx["fold"]["B_top_det"]]
        bx["fold"]["B_top_det"] = [down((hb[0] + hb[1]) / 2), str(hb[1])]    # cuts off the true minimum
        sw = copy.deepcopy(json.loads(CODIM2_JSON.read_text())["codim2_switch_F3S"])
        sw["q_star"] = ["1.54500627518087440", "1.54500627518087698"]           # excludes q* = 1.545006275180874300...
        res["negative_controls"] = {
            "anchor_m2_rho0.3_B_shifted_flagged": not check_anchor("m2_rho0.3", a)["agree"],
            "box_required[100]_hull_cut_flagged": not check_box(("required[100]", bx))["agree"],
            "switch_q_star_shifted_flagged": not check_switch(sw)["agree"]}
        print("negative controls", res["negative_controls"], flush=True)
    findings = []
    if "negative_controls" in res and not all(res["negative_controls"].values()):
        findings.append("a negative control was NOT flagged (comparison logic defect)")
    if "anchors" in res:
        for k, v in res["anchors"]["cases"].items():
            if not v["agree"]:
                findings.append(f"anchor {k}: disagreement (see anchors.cases.{k})")
    if "fold_curve_boxes" in res:
        for r in res["fold_curve_boxes"]["boxes"]:
            if not r["agree"]:
                findings.append(f"fold-curve box {r['range']} {r['rho_box']}: not confirmed "
                                f"(param={r['arb_parametric_certified']}, fail={r.get('arb_fail')})")
    if "codim2_switch" in res and not res["codim2_switch"]["agree"]:
        findings.append("codim-2 switch: not confirmed (see codim2_switch)")
    res["findings"] = findings
    res["verdict"] = ("AGREE: every checked enclosure overlaps the stored one; see per-part counts"
                      if not findings else "DISAGREEMENTS/UNCONFIRMED ITEMS: see findings")
    res["provenance"] = {
        "source_sha256": {"code/fb_v3_arb_recheck.py": sha(Path(__file__).resolve())},
        "input_sha256": {str(p.relative_to(HERE.parent)): sha(p) for p in (TH8_JSON, BOXES_JSON, FOLD_JSON, CODIM2_JSON)},
        "python_version": platform.python_version(), "python_flint_version": flint.__version__,
        "flint_version": getattr(flint, "__FLINT_VERSION__", None),
        "precision_bits": PREC}
    res["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dumps_hybrid(res))
    print("wrote", OUT, "findings:", len(findings))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
