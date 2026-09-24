#!/usr/bin/env python3
"""TH-8b: interval-arithmetic certificate of (D1)-(D3), B_top^det and the fold
constants for the finite-width stripes of Proposition 4 (SM th8:*).

Replaces the float64 grid scan of code/fb_finite_width_det.py by a verified
computation with mpmath.iv (outward-rounded interval arithmetic, 100-bit
mantissa).  Deterministic, no random numbers, no seeds.

Objects (anchor geometry gamma = D0 = 1, z0 = 4, zbar = 0, W = 1, d = 2,
I = [tau, T] = [0.5, 3.5], equal weights w_j = 1/m):
    mu(t) = 4 exp(-t),  c_j = mu(t_j),
    G0(t) = sum_j w_j exp(-(mu(t)-c_j)^2/(2 rho^2)) / (sqrt(2 pi) rho),
    r0 = G0'/G0^2,   N := G0'' G0 - 2 G0'^2  (sign r0' = sign N since G0 > 0),
    N' = G0''' G0 - 3 G0' G0''   (r0'' = N'/G0^3 where N = 0),
    Lambda0(t) = int_0^t G0.

Method.
  * Taylor-mode automatic differentiation with interval coefficients: the jet
    of G0 evaluated at an interval J encloses G0^{(k)}(t)/k! for all t in J.
  * Enclosure of a derivative F^{(r)} on a cell J (centre m):
    F^{(r)}(m) + F^{(r+1)}(m)(J-m) + F^{(r+2)}(J)(J-m)^2/2  (Taylor form).
  * Verified zero counting (branch and bound, bisection): a cell is resolved if
    the enclosure of F excludes 0, or if the enclosure of F' excludes 0 (F
    strictly monotone) and the thin endpoint values have definite signs (then
    F has exactly one zero in the open cell iff the signs differ).  Cells are
    bisected otherwise; a cell narrower than MIN_WIDTH aborts the certificate.
  * (D1): zeros of G0' on I with the sign of G0'' at each (all nondegenerate).
  * (D2): N < 0 on [tau, hi(hat t_1)].
  * (D3): on [lo(check t_{j-1}), hi(hat t_j)], N has exactly one zero, with
    N' < 0 on its cell; N > 0 on the enclosure of check t_{j-1}, N < 0 on the
    enclosure of hat t_j, so that zero lies in the open flank.
  * Fold points refined by verified bisection on the sign of N (thin
    evaluations); b_j, r0''(t_f) enclosed by the Taylor form on the final cell.
  * Independent cross-check: 2-D Krawczyk operator for the fold system
    H(t,B) = (G0' - B G0^2, G0'' - 2 B G0 G0') = 0 on a box around (t_f, b_j).
  * Lambda0 by the composite midpoint rule with verified remainder
    int_{c-h/2}^{c+h/2} G = h G(c) + h^3 G''(c)/24 + h^5 G''''(xi)/1920,
    G''''(xi) enclosed over the cell.
  * Concavity window of r0 at the binding fold, lower bound of
    g(t,B) = exp(-B Lambda0) G0^2 on the fold box, basin masses at the fold.

Run:  python fb_th8_interval_certificate.py
Output: ../artifacts/data/exact_m_fixed_budget/TH8_certificate/certificate.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mpmath import iv, mp

PREC = 100
iv.prec = PREC
mp.prec = PREC

HERE = Path(__file__).resolve().parent
OUT = (HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
       / "TH8_certificate" / "certificate.json")

# ----------------------------------------------------------------------------
# geometry (strings -> exact decimal enclosures)
TAU = iv.mpf("0.5")
TEND = iv.mpf("3.5")
Z0 = iv.mpf("4")
TARGETS = {2: ("1.0", "2.5"), 3: ("0.8", "1.6", "2.8")}
RHOS = {2: ("0.15", "0.2", "0.25", "0.3", "0.35", "0.4"),
        3: ("0.15", "0.2", "0.25", "0.3", "0.35", "0.4")}
ORDER = 7            # jet order of G0 (G0^{(k)} for k <= 7)
MIN_WIDTH = mp.mpf("1e-14")
INIT_CELLS = 96      # initial uniform cells on a search interval


def lo(x):
    return mp.mpf(x.a)


def hi(x):
    return mp.mpf(x.b)


def pos(x):
    return lo(x) > 0


def neg(x):
    return hi(x) < 0


def I(a, b=None):
    if b is None:
        return iv.mpf(a)
    return iv.mpf([a, b])


def _outward(v, down):
    """Decimal string (18 significant digits) that is <= v (down) or >= v."""
    eps = abs(v) * mp.mpf("1e-17") if v != 0 else mp.mpf("1e-300")
    w = v - eps if down else v + eps
    s = mp.nstr(w, 18, min_fixed=-12, max_fixed=12)
    # nstr rounds to nearest: relative error <= 5e-18 < 1e-17, so s is outward
    assert (mp.mpf(s) <= v) if down else (mp.mpf(s) >= v)
    return s


def to_list(x):
    """[lo, hi] as outward-rounded decimal strings for JSON."""
    return [_outward(lo(x), True), _outward(hi(x), False)]


def down(v):
    """decimal string <= the real number v (v an mp number): a certified LOWER bound."""
    return _outward(mp.mpf(v), True)


def up(v):
    """decimal string >= the real number v (v an mp number): a certified UPPER bound."""
    return _outward(mp.mpf(v), False)


def exact(v):
    """exact decimal expansion of the binary (dyadic) mp number v (no rounding)."""
    sign, man, ex, _ = mp.mpf(v)._mpf_
    if man == 0:
        return "0"
    s = "-" if sign else ""
    if ex >= 0:
        return s + str(man << ex)
    num = str(man * 5 ** (-ex)).rjust(-ex + 1, "0")
    ip, fp = num[:len(num) + ex], num[len(num) + ex:].rstrip("0")
    return s + ip + ("." + fp if fp else "")


def fl(x):
    return float((lo(x) + hi(x)) / 2)


def hull(xs):
    a = min(lo(x) for x in xs)
    b = max(hi(x) for x in xs)
    return iv.mpf([a, b])


# ----------------------------------------------------------------------------
# Taylor jets with interval coefficients: c[k] encloses F^{(k)}(t)/k!
def jmul(a, b):
    n = min(len(a), len(b))
    return [sum((a[i] * b[k - i] for i in range(k + 1)), I(0)) for k in range(n)]


def jadd(a, b):
    n = min(len(a), len(b))
    return [a[k] + b[k] for k in range(n)]


def jsub(a, b):
    n = min(len(a), len(b))
    return [a[k] - b[k] for k in range(n)]


def jscale(a, s):
    return [s * x for x in a]


def jexp(a):
    n = len(a)
    y = [iv.exp(a[0])]
    for k in range(1, n):
        acc = I(0)
        for i in range(1, k + 1):
            acc += i * a[i] * y[k - i]
        y.append(acc / k)
    return y


def jdiv(a, b):
    n = min(len(a), len(b))
    q = []
    for k in range(n):
        acc = a[k]
        for i in range(1, k + 1):
            acc -= b[i] * q[k - i]
        q.append(acc / b[0])
    return q


def jderiv(a, r=1):
    """jet of F^{(r)} from jet of F (length decreases by r)."""
    out = list(a)
    for _ in range(r):
        out = [(k + 1) * out[k + 1] for k in range(len(out) - 1)]
    return out


def fact(k):
    f = 1
    for i in range(2, k + 1):
        f *= i
    return f


class Geometry:
    def __init__(self, m, rho):
        self.m = m
        self.rho_str = rho
        self.rho = I(rho)
        self.w = I(1) / m
        self.c = [Z0 * iv.exp(-I(tj)) for tj in TARGETS[m]]
        self.pref = I(1) / (iv.sqrt(2 * iv.pi) * self.rho)   # W = 1, d = 2
        self.inv2s2 = I(1) / (2 * self.rho * self.rho)

    def jet(self, t, order=ORDER):
        """jet of G0 at the (possibly thick) interval t."""
        tj = [t, I(1)] + [I(0)] * (order - 1)
        mu = jscale(jexp(jscale(tj, I(-1))), Z0)
        acc = [I(0)] * (order + 1)
        for cj in self.c:
            d = list(mu)
            d[0] = d[0] - cj
            e = jexp(jscale(jmul(d, d), -self.inv2s2))
            acc = jadd(acc, e)
        return jscale(acc, self.w * self.pref)

    # derived jets --------------------------------------------------------
    @staticmethod
    def jN(g):
        g1, g2 = jderiv(g, 1), jderiv(g, 2)
        return jsub(jmul(g2, g), jscale(jmul(g1, g1), I(2)))

    @staticmethod
    def jr0(g):
        g1 = jderiv(g, 1)
        return jdiv(g1, jmul(g, g))


def taylor_encl(jm, jJ, r, J, m):
    """Enclosure of F^{(r)} on J from jets of F at thin m and at J (order-2 form)."""
    d0 = jm[r] * fact(r)
    d1 = jm[r + 1] * fact(r + 1)
    d2 = jJ[r + 2] * fact(r + 2)
    h = J - m
    return d0 + d1 * h + d2 * h * h / 2


def thin_val(jm, r):
    return jm[r] * fact(r)


def intersect(x, y):
    a = max(lo(x), lo(y))
    b = min(hi(x), hi(y))
    if a > b:
        raise RuntimeError("empty intersection of two valid enclosures")
    return iv.mpf([a, b])


class Fun:
    """F given as a jet map t -> jet(F); counts zeros of F^{(r)}."""

    def __init__(self, jetmap, r, name):
        self.jetmap, self.r, self.name = jetmap, r, name
        self.cache = {}
        self.n_evals = 0

    def jet(self, t):
        self.n_evals += 1
        return self.jetmap(t)

    def point(self, x):
        """thin evaluation of F^{(r)} at the exactly representable point x."""
        key = mp.nstr(x, 40)
        if key not in self.cache:
            self.cache[key] = thin_val(self.jet(I(x)), self.r)
        return self.cache[key]

    def enclose(self, a, b, r_extra=0):
        """enclosure of F^{(r + r_extra)} on [a, b]."""
        J = I(a, b)
        m = (a + b) / 2
        jm, jJ = self.jet(I(m)), self.jet(J)
        rr = self.r + r_extra
        v = taylor_encl(jm, jJ, rr, J, I(m))
        return intersect(v, jJ[rr] * fact(rr))

    def resolve(self, a, b):
        J = I(a, b)
        m = (a + b) / 2
        jm, jJ = self.jet(I(m)), self.jet(J)
        r = self.r
        v = intersect(taylor_encl(jm, jJ, r, J, I(m)), jJ[r] * fact(r))
        if pos(v):
            return ("pos", v, None)
        if neg(v):
            return ("neg", v, None)
        d = intersect(taylor_encl(jm, jJ, r + 1, J, I(m)), jJ[r + 1] * fact(r + 1))
        if pos(d) or neg(d):
            va, vb = self.point(a), self.point(b)
            if (pos(va) or neg(va)) and (pos(vb) or neg(vb)):
                if pos(va) != pos(vb):
                    return ("root", v, d)
                return ("pos" if pos(va) else "neg", v, d)
        return (None, v, d)

    def scan(self, a, b, init_cells=INIT_CELLS):
        """verified partition of [a, b] into resolved cells."""
        a, b = mp.mpf(a), mp.mpf(b)
        stack = []
        h = (b - a) / init_cells
        edges = [a + k * h for k in range(init_cells)] + [b]
        for k in range(init_cells - 1, -1, -1):
            stack.append((edges[k], edges[k + 1]))
        leaves = []
        while stack:
            x, y = stack.pop()
            st, v, d = self.resolve(x, y)
            if st is None:
                if y - x < MIN_WIDTH:
                    raise RuntimeError(f"{self.name}: unresolved cell [{x}, {y}]")
                mid = (x + y) / 2
                stack.append((mid, y))
                stack.append((x, mid))
                continue
            leaves.append({"a": x, "b": y, "status": st, "val": v, "der": d})
        return leaves

    def refine(self, a, b, width=mp.mpf("1e-26")):
        """verified bisection of a simple sign change of F^{(r)} on [a, b]."""
        va, vb = self.point(a), self.point(b)
        sa = pos(va)
        assert (pos(va) or neg(va)) and (pos(vb) or neg(vb)) and sa != pos(vb)
        it = 0
        while b - a > width:
            m = (a + b) / 2
            vm = thin_val(self.jet(I(m)), self.r)
            if not (pos(vm) or neg(vm)):
                break  # value enclosure contains 0: stop, keep [a, b]
            if pos(vm) == sa:
                a = m
            else:
                b = m
            it += 1
        return a, b, it


def roots_of(leaves):
    out = []
    for lf in leaves:
        if lf["status"] == "root":
            out.append({"a": lf["a"], "b": lf["b"],
                        "slope_sign": 1 if pos(lf["der"]) else -1,
                        "slope_encl": lf["der"]})
    return out


# ----------------------------------------------------------------------------
# Lambda0 = int_0^t G0 by the composite midpoint rule with verified remainder
class Lambda0:
    H = mp.mpf(1) / 256

    def __init__(self, geo, tmax=mp.mpf("3.5")):
        self.geo = geo
        n = int(mp.ceil(tmax / self.H))
        self.prefix = [I(0)]
        for k in range(n):
            x = k * self.H
            self.prefix.append(self.prefix[-1] + self.cell(I(x), I(x + self.H), x, x + self.H))

    def cell(self, xa, xb, xa_lo, xb_hi):
        """encloses int_{xa}^{xb} G0 (xa, xb thin or thin-ish intervals)."""
        h = xb - xa
        c = (xa + xb) / 2
        jc = self.geo.jet(c, order=2)
        jJ = self.geo.jet(I(xa_lo, xb_hi), order=4)
        g2 = jc[2] * 2
        g4 = jJ[4] * 24
        return h * jc[0] + h ** 3 * g2 / 24 + h ** 5 * g4 / 1920

    def at_point(self, x):
        """Lambda0(x) for an exactly representable mp number x."""
        k = int(mp.floor(x / self.H))
        xk = k * self.H
        if xk == x:
            return self.prefix[k]
        return self.prefix[k] + self.cell(I(xk), I(x), xk, x)

    def at(self, a, b=None):
        """enclosure of Lambda0 on the point interval [a, b] (monotone)."""
        if b is None:
            return self.at_point(a)
        la, lb = self.at_point(a), self.at_point(b)
        return iv.mpf([lo(la), hi(lb)])


def krawczyk_fold(geo, t_c, b_c, rt, rb):
    """Krawczyk test for H(t,B) = (G'-B G^2, G''-2B G G') on [t_c+-rt] x [b_c+-rb]."""
    def derivs(t):
        j = geo.jet(t, order=4)
        return [j[k] * fact(k) for k in range(4)]

    def H(t, B):
        g0, g1, g2, _ = derivs(t)
        return [g1 - B * g0 * g0, g2 - 2 * B * g0 * g1]

    def Jac(t, B):
        g0, g1, g2, g3 = derivs(t)
        return [[g2 - 2 * B * g0 * g1, -g0 * g0],
                [g3 - 2 * B * (g1 * g1 + g0 * g2), -2 * g0 * g1]]

    T = I(t_c - rt, t_c + rt)
    Bx = I(b_c - rb, b_c + rb)
    tm, bm = I(t_c), I(b_c)
    Jm = Jac(tm, bm)
    # approximate inverse of the midpoint Jacobian (any real matrix is admissible)
    a11, a12, a21, a22 = [mp.mpf((lo(x) + hi(x)) / 2) for x in (Jm[0][0], Jm[0][1], Jm[1][0], Jm[1][1])]
    det = a11 * a22 - a12 * a21
    Y = [[I(a22 / det), I(-a12 / det)], [I(-a21 / det), I(a11 / det)]]
    Hm = H(tm, bm)
    JX = Jac(T, Bx)
    dX = [T - tm, Bx - bm]
    K = []
    for i in range(2):
        yh = Y[i][0] * Hm[0] + Y[i][1] * Hm[1]
        row = I(0)
        for k in range(2):
            ik = (I(1) if i == k else I(0)) - (Y[i][0] * JX[0][k] + Y[i][1] * JX[1][k])
            row += ik * dX[k]
        K.append([tm, bm][i] - yh + row)
    inside = (lo(K[0]) > lo(T) and hi(K[0]) < hi(T) and lo(K[1]) > lo(Bx) and hi(K[1]) < hi(Bx))
    return {"box_t": to_list(T), "box_B": to_list(Bx), "K_t": to_list(K[0]), "K_B": to_list(K[1]),
            "K_in_interior_of_box": bool(inside),
            "jacobian_det_at_centre": float(det)}


# ----------------------------------------------------------------------------
def certify(m, rho, full=True):
    geo = Geometry(m, rho)
    t0 = time.time()
    tau, T = mp.mpf("0.5"), mp.mpf("3.5")
    rec = {"m": m, "rho_s": rho, "weights": f"1/{m}", "targets": list(TARGETS[m])}
    # ---------------- (D1): zeros of G0' on I --------------------------------
    Gp = Fun(lambda t: geo.jet(t), 1, "G0'")
    leaves = Gp.scan(tau, T)
    roots = roots_of(leaves)
    g1_tau, g1_T = Gp.point(tau), Gp.point(T)
    types = ["max" if r["slope_sign"] < 0 else "min" for r in roots]
    expect = ["max", "min"] * (m - 1) + ["max"]
    d1 = (types == expect) and pos(g1_tau) and neg(g1_T)
    crit = []
    for r in roots:
        a, b, it = Gp.refine(r["a"], r["b"])
        g2 = Gp.enclose(a, b, r_extra=1)          # G0'' on the refined cell
        g0 = geo.jet(I(a, b), order=0)[0]
        assert (pos(g2) or neg(g2)) and (neg(g2) == (r["slope_sign"] < 0))
        crit.append({"type": "max" if r["slope_sign"] < 0 else "min",
                     "t": to_list(I(a, b)), "G0": to_list(g0), "G0pp": to_list(g2),
                     "search_cell": [float(r["a"]), float(r["b"])],
                     "G0pp_on_search_cell": to_list(r["slope_encl"]),
                     "bisection_steps": it,
                     "refined_width_upper_bound": up(hi(I(b) - I(a))),
                     "refined_width_le_1e-26": bool(hi(I(b) - I(a)) <= lo(I("1e-26"))),
                     "_a": a, "_b": b})
    rec["D1"] = {"holds": bool(d1), "signature": types,
                 "n_cells": len(leaves), "G0p_tau": to_list(g1_tau), "G0p_T": to_list(g1_T),
                 "critical_points": [{k: v for k, v in c.items() if not k.startswith("_")} for c in crit]}
    if not d1:
        rec["D1"]["note"] = ("(D1) certified to FAIL: the verified signature of G0 on I has fewer "
                             "than m maxima (all critical points simple, enclosed above)")
        rec["certified"] = False
        rec["runtime_s"] = time.time() - t0
        return rec
    maxima = [c for c in crit if c["type"] == "max"]
    minima = [c for c in crit if c["type"] == "min"]
    # ---------------- (D2): N < 0 on [tau, hi(hat t_1)] ------------------------
    Nf = Fun(lambda t: Geometry.jN(geo.jet(t)), 0, "N")
    lv2 = Nf.scan(tau, maxima[0]["_b"])
    d2 = all(lf["status"] == "neg" for lf in lv2)
    Nmax_first = max(hi(lf["val"]) for lf in lv2)
    n_type_a = sum(1 for lf in lv2 if neg(lf["val"]))   # cells whose N-enclosure itself is < 0
    rec["D2"] = {"holds": bool(d2), "interval_exact": ["0.5", exact(maxima[0]["_b"])],
                 "n_cells": len(lv2), "n_cells_type_a": n_type_a,
                 "all_cells_type_a": bool(n_type_a == len(lv2)),
                 "sup_N_upper_bound": up(Nmax_first)}
    # ---------------- (D3): one nondegenerate max of r0 per interior flank -----
    flanks = []
    d3 = True
    for j in range(2, m + 1):
        mn, mx = minima[j - 2], maxima[j - 1]
        a, b = mn["_a"], mx["_b"]
        lv = Nf.scan(a, b)
        rts = roots_of(lv)
        N_at_min = Nf.enclose(mn["_a"], mn["_b"])
        N_at_max = Nf.enclose(mx["_a"], mx["_b"])
        ok = (len(rts) == 1 and rts[0]["slope_sign"] < 0 and pos(N_at_min) and neg(N_at_max))
        fl_rec = {"j": j, "search_interval": [float(a), float(b)], "n_cells": len(lv),
                  "n_zeros_of_N": len(rts), "N_on_check_t": to_list(N_at_min),
                  "N_on_hat_t": to_list(N_at_max)}
        if ok:
            r = rts[0]
            fa, fb, it = Nf.refine(r["a"], r["b"])
            Jf = I(fa, fb)
            jr = Geometry.jr0(geo.jet(Jf))
            bj = jr[0]
            r0pp = jr[2] * 2
            jg = geo.jet(Jf, order=3)
            ok = ok and neg(r0pp)
            fl_rec.update({"t_fold": to_list(Jf), "b_j": to_list(bj), "r0pp_at_fold": to_list(r0pp),
                           "Np_on_zero_cell": to_list(r["slope_encl"]),
                           "zero_cell": [float(r["a"]), float(r["b"])],
                           "G0_at_fold": to_list(jg[0]), "bisection_steps": it,
                           "refined_width_upper_bound": up(hi(I(fb) - I(fa))),
                           "refined_width_le_1e-26": bool(hi(I(fb) - I(fa)) <= lo(I("1e-26"))),
                           "_fa": fa, "_fb": fb, "_bj": bj, "_r0pp": r0pp})
        fl_rec["holds"] = bool(ok)
        d3 = d3 and ok
        flanks.append(fl_rec)
    rec["D3"] = {"holds": bool(d3), "flanks": [{k: v for k, v in f.items() if not k.startswith("_")} for f in flanks]}
    # ---------------- B_top^det --------------------------------------------------
    r0tau = Geometry.jr0(geo.jet(I(tau)))[0]
    cands = {"left_endpoint": r0tau}
    for f in flanks:
        if "_bj" in f:
            cands[f"flank_{f['j']}"] = f["_bj"]
    rec["candidates"] = {k: to_list(v) for k, v in cands.items()}
    certified = d1 and d2 and d3
    d4 = False
    if certified:
        names = sorted(cands, key=lambda k: lo(cands[k]))
        best = names[0]
        unique = all(hi(cands[best]) < lo(cands[k]) for k in names[1:])
        second = cands[names[1]]
        rec["B_top_det"] = to_list(cands[best])
        rec["binding"] = best
        rec["binding_unique"] = bool(unique)
        rec["binding_is_interior_flank"] = best.startswith("flank_")
        rec["next_candidate"] = names[1]
        rec["next_candidate_lower_bound"] = down(lo(second))
        rec["gap_to_next_candidate_lower_bound"] = down(lo(I(lo(second)) - I(hi(cands[best]))))
        d4 = bool(unique and best.startswith("flank_"))
    rec["certified_D1_D3"] = bool(certified)
    rec["certified_D4"] = bool(d4)
    if certified and d4 and full:
        rec["fold"] = fold_constants(geo, flanks, rec, minima, maxima)
    rec["runtime_s"] = round(time.time() - t0, 2)
    # aggregate flag: (D1)-(D4) and all fold constants (Krawczyk, concavity window,
    # s = A2 delta^2 < b, g > 0 on the fold box)
    rec["certified"] = bool(certified and d4 and (not full or rec.get("fold", {}).get("ok", False)))
    return rec


def fold_constants(geo, flanks, rec, minima, maxima):
    k = int(rec["binding"].split("_")[1])
    f = [x for x in flanks if x["j"] == k][0]
    fa, fb, bj, r0pp = f["_fa"], f["_fb"], f["_bj"], f["_r0pp"]
    out = {"flank": k}
    a_f = -r0pp
    out["a_f=-r0pp(t_f)"] = to_list(a_f)
    # Krawczyk cross-check of the fold system on a small box
    tc, bc = (fa + fb) / 2, mp.mpf((lo(bj) + hi(bj)) / 2)
    kr = None
    for rt, rbrel in (("1e-8", "1e-7"), ("1e-7", "1e-6"), ("1e-9", "1e-8"), ("1e-6", "1e-5")):
        rb = mp.mpf(rbrel) * max(1, bc)
        kr = krawczyk_fold(geo, tc, bc, mp.mpf(rt), rb)
        kr["halfwidth_t"] = rt
        kr["halfwidth_B"] = f"{rbrel}*max(1,B_centre) = {mp.nstr(rb, 6)}"
        if kr["K_in_interior_of_box"]:
            break
    out["krawczyk"] = kr
    # concavity window of r0 around t_f.  delta is the exact decimal (enclosed by
    # I(delta)); [wa, wb] is rounded OUTWARD so that [t_f - delta, t_f + delta]
    # (t_f in [fa, fb]) is contained in it; the 64 cells cover [wa, wb] exactly.
    lo_flank, hi_flank = minima[k - 2]["_b"], maxima[k - 1]["_a"]
    window = None
    for delta in ("0.02", "0.01", "0.005", "0.002", "0.001", "0.0005", "0.0002", "0.0001"):
        dI = I(delta)
        wa, wb = lo(I(fa) - dI), hi(I(fb) + dI)
        if not (wa > lo_flank and wb < hi_flank):
            continue
        ncell = 64
        edges = [wa] + [wa + (wb - wa) * i / ncell for i in range(1, ncell)] + [wb]
        assert all(edges[i] < edges[i + 1] for i in range(ncell))
        encl2, encl3 = [], []
        for i in range(ncell):
            x, y = edges[i], edges[i + 1]
            J, mm = I(x, y), (x + y) / 2
            jm, jJ = Geometry.jr0(geo.jet(I(mm))), Geometry.jr0(geo.jet(J))
            encl2.append(intersect(taylor_encl(jm, jJ, 2, J, I(mm)), jJ[2] * 2))
            encl3.append(intersect(taylor_encl(jm, jJ, 3, J, I(mm)), jJ[3] * 6))
        h2, h3 = hull(encl2), hull(encl3)
        A1, A2 = -hi(h2), -lo(h2)          # exact binary numbers: A1 <= -r0'' <= A2 on [wa, wb]
        if A1 > 0 and A1 >= hi(a_f) / 2:   # -r0'' >= a_f/2 for every a_f in its enclosure
            window = {"delta": delta, "window_exact": [exact(wa), exact(wb)],
                      "N_delta_subset_window": True,          # by outward construction
                      "window_in_open_flank": True,           # tested above: hi(check t) < wa, wb < lo(hat t)
                      "A1_lower_bound_of_-r0pp": down(A1), "A2_upper_bound_of_-r0pp": up(A2),
                      "A1_ge_hi(a_f)/2": True,
                      "r0ppp_bound": up(max(abs(lo(h3)), abs(hi(h3)))),
                      "_wa": wa, "_wb": wb, "_A1": A1, "_A2": A2, "_dI": dI}
            break
    if window is None:
        out["ok"] = False
        return out
    out["concavity_window"] = {kk: v for kk, v in window.items() if not kk.startswith("_")}
    # Lambda0, g, dPhi/dbeta margins, masses
    L = Lambda0(geo)
    tau, T = mp.mpf("0.5"), mp.mpf("3.5")
    lam_f = L.at(fa, fb)
    lam_wb = L.at_point(window["_wb"])
    G0_f = geo.jet(I(fa, fb), order=0)[0]
    g_f = iv.exp(-bj * lam_f) * G0_f * G0_f
    G0_win = geo.jet(I(window["_wa"], window["_wb"]), order=0)[0]
    A2, dI = window["_A2"], window["_dI"]
    # s := A2 delta^2 (theorem), enclosed; beta-box [b - s, b + s] contains
    # r0(N_delta) in [b - A2 delta^2/2, b].  Use the UPPER end s_up >= s.
    s_I = I(A2) * dI * dI
    s_up = hi(s_I)
    s_lt_b = bool(s_up < lo(bj))
    g_min_box = iv.exp(-(I(hi(bj)) + I(s_up)) * lam_wb) * I(lo(G0_win)) ** 2
    out["Lambda0_at_t_f"] = to_list(lam_f)
    out["g_at_fold=exp(-b Lambda0) G0^2"] = to_list(g_f)
    out["log10_g_at_fold"] = to_list(iv.log(g_f) / iv.log(10))
    out["G0_on_window"] = to_list(G0_win)
    out["beta_box_halfwidth_s=A2*delta^2"] = to_list(s_I)
    out["s_upper_lt_b_lower"] = s_lt_b
    out["s_over_b_upper_bound"] = up(hi(I(s_up) / I(lo(bj))))
    out["g_min_on_fold_box_lower_bound"] = down(lo(g_min_box))
    out["log10_g_min_on_fold_box_lower_bound"] = down(lo(iv.log(I(lo(g_min_box))) / iv.log(10)))
    lam_tau, lam_T = L.at_point(tau), L.at_point(T)
    mn = minima[k - 2]
    lam_min = L.at(mn["_a"], mn["_b"])
    S = lambda lam: iv.exp(-bj * lam)
    if geo.m == 2:
        out["masses_fold_cut_[tau,t_f],[t_f,T]"] = [to_list(S(lam_tau) - S(lam_f)), to_list(S(lam_f) - S(lam_T))]
        out["masses_nominal_cut_[tau,check_t],[check_t,T]"] = [to_list(S(lam_tau) - S(lam_min)),
                                                             to_list(S(lam_min) - S(lam_T))]
    else:
        out["late_mass_fold_cut_[t_f,T]"] = to_list(S(lam_f) - S(lam_T))
    out["Lambda0_tau_T"] = [to_list(lam_tau), to_list(lam_T)]
    out["ok"] = bool(kr["K_in_interior_of_box"] and pos(a_f) and pos(g_min_box)
                     and window["_A1"] > 0 and s_lt_b)
    return out


def main():
    t0 = time.time()
    cases = []
    for m in (2, 3):
        for rho in RHOS[m]:
            rec = certify(m, rho)
            cases.append(rec)
            print(f"m={m} rho={rho}: certified={rec['certified']} "
                  f"B_top_det={rec.get('B_top_det')} binding={rec.get('binding')} "
                  f"({rec['runtime_s']} s)", flush=True)
    data = {
        "script": "code/fb_th8_interval_certificate.py",
        "item": "TH-8b (P4): interval certificate of (D1)-(D4), B_top^det and fold constants",
        "deterministic": True, "seeds": None,
        "arithmetic": f"mpmath {__import__('mpmath').__version__} iv, outward rounding, {PREC}-bit mantissa; "
                      "JSON decimal bounds rounded outward (18 significant digits)",
        "serialization": ("Every interval ([lo, hi] string pair) and every field named *lower_bound* / "
                          "*upper_bound* / A1_* / A2_* / r0ppp_bound / s_over_b_upper_bound is a decimal "
                          "string rounded in the certified direction (lower bounds down, upper bounds up). "
                          "*_exact fields are exact decimal expansions of binary numbers. Remaining float "
                          "fields (search_cell, search_interval, zero_cell, jacobian_det_at_centre) are "
                          "round-to-nearest DIAGNOSTICS, not certified bounds."),
        "certified_flag": ("certified = (D1) & (D2) & (D3) & (D4: unique binding candidate, an interior "
                           "flank) & fold constants ok (Krawczyk box contraction, concavity window with "
                           "A1 >= hi(a_f)/2 > 0, s = A2 delta^2 < b, g > 0 on the fold box)"),
        "revision": "2026-09-23 fix pass (fix_P4): outward window/s/serialization, D4 in aggregate flag, refined zero widths recorded",
        "geometry": {"gamma": 1, "D0": 1, "z0": 4, "zbar": 0, "W": 1, "d": 2, "window": [0.5, 3.5],
                     "targets": {str(k): list(v) for k, v in TARGETS.items()}, "weights": "equal"},
        "method": __doc__.split("Method.")[1].split("Run:")[0].strip(),
        "cases": {f"m{r['m']}_rho{r['rho_s']}": r for r in cases},
        "summary": {f"m{r['m']}_rho{r['rho_s']}": {"certified": r["certified"],
                                                  "D1": r["D1"]["holds"],
                                                  "B_top_det": r.get("B_top_det"),
                                                  "binding": r.get("binding"),
                                                  "binding_unique": r.get("binding_unique"),
                                                  "certified_D4": r.get("certified_D4")}
                    for r in cases},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
