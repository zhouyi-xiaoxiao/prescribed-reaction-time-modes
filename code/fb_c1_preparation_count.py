#!/usr/bin/env python3
"""C1 (theory closer): checks for theory/TH10b_preparation_count.tex.

Subcommands
-----------
interval  (needs mpmath; run with ~/.local-build/venvs/mp/bin/python)
    Directed-rounding interval certificates (mpmath.iv):
    * the hypotheses (N2)-(N3) of Theorem c1:thm-moving for the standardized
      offsets n_j(t) = Z*(Y_j - Y)/S(t) of Lemma c1:lem-std (anchor data, three
      preparations): enclosures of min_I n_j', min_I (n_j - n_{j+1}), the
      closed-form nu and d_*, and min over t>=0 of A_S + e0 Y Y_j;
    * the exact stationary-point census of the free clock with the contact
      factor set to 1, F(t) = S(t)^{-1} sum_j w_j exp(-n_j^2/(2 eps^2)), on I,
      by interval bisection on D = (log F)' and D' (formulas (c1:D));
    * an out-of-sketch case (s0^2 = 100 s_Z^2) where the sector ratio omega
      of the old Remark th10:rem-thm1 exceeds 2/3 (certified lower bound), and
      its certified census.
fk        (numpy; any python with the FK module importable)
    Small-budget check of the transfer on the stored N4 preparation ensembles
    (FK exact-law estimator, tag 83 paths; no new simulation): f/B at small B
    against the discrete free rate G, and critical-point census.
floatG    (numpy) float64 census of the full free rate G including the exact
    contact factor c(t) (mean_contact_curve), for the anchor preparations.

No random numbers are drawn here (the FK check reuses stored ensembles), so no
new seed tag is consumed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
OUT = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "C1_preparation_count"

ANCHOR = dict(gamma=1.0, D0=1.0, z0=4.0, zbar=0.0, rho=1.0, tau=0.5, T=3.5)
TARGETS = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}
PREPS = {"point_release": 0.0, "stationary": 1.0, "four_x_stationary": 4.0}  # s0^2 / sZ^2

# Exact (decimal-string / rational) versions of every input of the interval part
# (fix pass 2026-09-23, audit F9): non-dyadic decimals must never pass through a
# binary64 float before entering mpmath.iv, since iv.mpf(0.1) is the point interval
# at the binary64 number 0.1000000000000000055..., not an enclosure of 1/10.
ANCHOR_EXACT = dict(gamma="1", D0="1", z0="4", zbar="0", rho="1", tau="0.5", T="3.5")
TARGETS_EXACT = {2: ("1.0", "2.5"), 3: ("0.8", "1.6", "2.8")}
PREPS_EXACT = {"point_release": 0, "stationary": 1, "four_x_stationary": 4}  # integer s0^2/sZ^2


def _X(iv, s):
    """Interval enclosure of an exact decimal string 'd.ddd' or rational 'p/q'."""
    s = str(s)
    if "/" in s:
        p, q = s.split("/")
        return iv.mpf(p.strip()) / iv.mpf(q.strip())
    return iv.mpf(s)


def _mpf_to_fraction(v, end=0):
    """Exact Fraction of a binary mpf, or of endpoint `end` (0 = lower, 1 = upper) of an
    mpmath.iv interval (the endpoints x.a, x.b of an iv interval are themselves point
    ivmpf objects carrying `_mpi_`, not `_mpf_`)."""
    from fractions import Fraction
    raw = v._mpi_[end] if hasattr(v, "_mpi_") else v._mpf_
    sign, man, exp, _bc = raw
    if man == 0:
        return Fraction(0)
    f = Fraction(man) * (Fraction(2) ** exp)
    return -f if sign else f


def _dec(v, rounding, digits=20):
    """Directed decimal string of the (binary, exact) number v: rounding in {'floor','ceiling'}.
    For an iv interval v, 'floor' uses its lower and 'ceiling' its upper endpoint."""
    import decimal
    fr = _mpf_to_fraction(v, 0 if rounding == "floor" else 1)
    ctx = decimal.Context(prec=digits,
                          rounding=decimal.ROUND_FLOOR if rounding == "floor" else decimal.ROUND_CEILING)
    return str(ctx.divide(decimal.Decimal(fr.numerator), decimal.Decimal(fr.denominator)))


def _out(x, digits=20):
    """Outward-rounded decimal enclosure [lo, hi] (strings) of an mpmath.iv interval or mpf."""
    lo = getattr(x, "a", x)
    hi = getattr(x, "b", x)
    return [_dec(lo, "floor", digits), _dec(hi, "ceiling", digits)]


# ----------------------------------------------------------------------------
# interval part
# ----------------------------------------------------------------------------
def _iv_model(iv, *, gamma, D0, z0, zbar, rho, s0sq_ratio, targets):
    """All inputs are exact strings (see _X); s0sq_ratio = s0^2/sZ^2 is an integer."""
    sZ2 = _X(iv, D0) / (2 * _X(iv, gamma))
    A = sZ2 + _X(iv, rho) ** 2
    e0 = (iv.mpf(s0sq_ratio) - 1) * sZ2
    Z = abs(_X(iv, z0) - _X(iv, zbar))
    g = _X(iv, gamma)
    Yj = [iv.exp(-g * _X(iv, tj)) for tj in targets]

    def core(t):
        Y = iv.exp(-g * t)
        S2 = A + e0 * Y * Y
        S = iv.sqrt(S2)
        n, n1, n2 = [], [], []
        for yj in Yj:
            n.append(Z * (yj - Y) / S)
            n1.append(g * Z * Y * (A + e0 * Y * yj) / (S2 * S))
            n2.append(-g * g * Z * Y * ((A + 2 * e0 * yj * Y) / (S2 * S)
                                        - 3 * e0 * Y * Y * (A + e0 * Y * yj) / (S2 * S2 * S)))
        b = g * e0 * Y * Y / S2                     # -(log S)'
        b1 = -2 * g * g * A * e0 * Y * Y / (S2 * S2)  # -(log S)''
        return dict(Y=Y, S=S, n=n, n1=n1, n2=n2, b=b, b1=b1)

    return dict(core=core, A=A, e0=e0, Z=Z, Yj=Yj, sZ2=sZ2, g=g)


def _iv_D(iv, model, t, eps, w):
    """eps, w: interval enclosures (built with _X from exact strings)."""
    c = model["core"](t)
    m = len(w)
    E2 = 2 * eps ** 2
    ell = [iv.log(w[k]) - c["n"][k] ** 2 / E2 for k in range(m)]
    pi = []
    for k in range(m):
        s = iv.mpf(1)
        for i in range(m):
            if i != k:
                s = s + iv.exp(ell[i] - ell[k])
        pi.append(1 / s)
    gk = [c["n"][k] * c["n1"][k] for k in range(m)]
    Psi = sum((pi[k] * gk[k] for k in range(m)), iv.mpf(0))
    Epp = sum((pi[k] * (c["n1"][k] ** 2 + c["n"][k] * c["n2"][k]) for k in range(m)), iv.mpf(0))
    Var = iv.mpf(0)
    for k in range(m):
        for l in range(k + 1, m):
            Var = Var + pi[k] * pi[l] * (gk[k] - gk[l]) ** 2
    e2 = eps ** 2
    D = c["b"] - Psi / e2
    D1 = c["b1"] - Epp / e2 + Var / (e2 * e2)
    return D, D1


def _sign(x):
    if x.a > 0:
        return 1
    if x.b < 0:
        return -1
    return 0


def _cover(iv, tau, T):
    """Binary endpoints (plain mp.mpf points) of a covering domain [lo(tau), hi(T)] of the
    exact window [tau, T].  Grid and bisection points derived from them are plain mp.mpf
    numbers (rounded to nearest, which is harmless: they only have to be monotone and
    shared between neighbouring leaves), so every leaf box iv.mpf([lo, hi]) is exact."""
    from mpmath import mp
    return mp.mpf(_X(iv, tau).a), mp.mpf(_X(iv, T).b)


def iv_census(iv, model, eps, w, tau, T, max_depth=40, init_pieces=64):
    """Certified census of zeros of D=(log F)' on the exact window [tau,T].

    tau, T are exact strings; the bisection runs on the covering domain
    [lo(tau), hi(T)] (binary endpoints shared between neighbouring leaves), and the
    endpoint signs are evaluated on the interval enclosures of tau and T, so a strict
    sign there also excludes zeros in the (tiny) excess of the covering domain.
    A zero is certified in a leaf when D' has a strict sign on the leaf and D has
    strict opposite signs at the two leaf endpoints (point evaluations); a leaf
    with strict D sign has no zero.  ok=False if a leaf is unresolved at max_depth.
    All exported numbers are outward-rounded decimal strings (_out).
    """
    tau_, T_ = _cover(iv, tau, T)
    grid = [tau_] + [tau_ + (T_ - tau_) * k / init_pieces for k in range(1, init_pieces)] + [T_]
    stack = [(grid[k], grid[k + 1], 0) for k in range(init_pieces)]
    zeros, n_leaves, unresolved = [], 0, []
    while stack:
        lo, hi, depth = stack.pop()
        box = iv.mpf([lo, hi])
        D, D1 = _iv_D(iv, model, box, eps, w)
        sD = _sign(D)
        if sD != 0:
            n_leaves += 1
            continue
        sD1 = _sign(D1)
        if sD1 != 0:
            Dlo, _ = _iv_D(iv, model, iv.mpf(lo), eps, w)
            Dhi, _ = _iv_D(iv, model, iv.mpf(hi), eps, w)
            slo, shi = _sign(Dlo), _sign(Dhi)
            if slo != 0 and shi != 0:
                n_leaves += 1
                if slo != shi:
                    kind = "max" if sD1 < 0 else "min"
                    zeros.append(dict(leaf=_out(box), kind=kind, Dprime=_out(D1), _key=lo))
                continue
        if depth >= max_depth:
            unresolved.append(_out(box))
            continue
        mid = (lo + hi) / 2
        stack.append((mid, hi, depth + 1))
        stack.append((lo, mid, depth + 1))
    zeros.sort(key=lambda z: z["_key"])
    for z in zeros:
        del z["_key"]
    Dtau, _ = _iv_D(iv, model, _X(iv, tau), eps, w)
    DT, _ = _iv_D(iv, model, _X(iv, T), eps, w)
    kinds = [z["kind"] for z in zeros]
    m = len(w)
    expected = []
    for j in range(m):
        expected.append("max")
        if j < m - 1:
            expected.append("min")
    ok = (not unresolved) and kinds == expected and _sign(Dtau) == 1 and _sign(DT) == -1
    return dict(ok=bool(ok), n_max=kinds.count("max"), n_min=kinds.count("min"), kinds=kinds,
                zeros=zeros, leaves=n_leaves, unresolved=unresolved,
                D_tau=_out(Dtau), D_T=_out(DT))


def iv_min_over(iv, f, tau, T, pieces=4096):
    """[certified lower bound, certified upper bound] of min_{[tau,T]} f (outward strings).

    Lower bound: min over the boxes of a covering domain [lo(tau), hi(T)].  Upper
    bound: min of the upper ends of f at interior grid points (which lie in the exact
    window) and at the enclosures of the exact endpoints tau, T.
    """
    lo_bound = None
    up_bound = None
    tau_, T_ = _cover(iv, tau, T)
    grid = [tau_] + [tau_ + (T_ - tau_) * k / pieces for k in range(1, pieces)] + [T_]
    for k in range(pieces):
        val = f(iv.mpf([grid[k], grid[k + 1]]))
        lo_bound = val.a if lo_bound is None else min(lo_bound, val.a)
        if k >= 1:
            pv = f(iv.mpf(grid[k]))
            up_bound = pv.b if up_bound is None else min(up_bound, pv.b)
    for ends in (tau, T):
        pv = f(_X(iv, ends))
        up_bound = min(up_bound, pv.b)
    return [_dec(lo_bound, "floor"), _dec(up_bound, "ceiling")]


def cmd_interval(args):
    import mpmath
    from mpmath import iv, mp
    mp.prec = args.prec
    iv.prec = args.prec
    t0 = time.time()
    out = dict(item="C1_preparation_count", part="interval",
               note=("mpmath.iv directed rounding, prec=%d bits; contact factor set to 1 in the "
                     "census (F = S^{-1} sum_j w_j exp(-n_j^2/(2 eps^2))). All inputs are exact decimal "
                     "strings or rationals (ANCHOR_EXACT, TARGETS_EXACT, eps and weight strings) enclosed "
                     "by mpmath.iv; bisection runs on a covering domain [lo(tau), hi(T)] of the exact window. "
                     "Every exported number is an outward-rounded decimal string [floor(lo), ceil(hi)] "
                     "(20 significant digits); 'leaf' = certified box containing exactly one simple zero "
                     "of D=(log F)'. Fix pass 2026-09-23 (audit F9, F10)." % args.prec),
               mpmath_version=mpmath.__version__, anchor=ANCHOR_EXACT, cases=[])
    a = ANCHOR_EXACT
    cases = []
    for m in (2, 3):
        for prep, ratio in PREPS_EXACT.items():
            cases.append(dict(name="m%d_%s" % (m, prep), m=m, targets=TARGETS_EXACT[m],
                              s0sq_ratio=ratio, tau=a["tau"], T=a["T"], eps_list=["0.1", "0.05"]))
    # out-of-sketch case: s0^2 = 100 sZ^2, widely separated targets (old omega > 2/3)
    cases.append(dict(name="m2_s0sq_100sZ2_wide", m=2, targets=("0.3", "3.3"),
                      s0sq_ratio=100, tau="0.1", T="4.0", eps_list=["0.1", "0.05", "0.025"]))
    weights = {2: [("1/2", "1/2"), ("0.3", "0.7"), ("0.7", "0.3")],
               3: [("1/3", "1/3", "1/3"), ("0.2", "0.3", "0.5"), ("0.5", "0.3", "0.2")]}
    for cs in cases:
        model = _iv_model(iv, gamma=a["gamma"], D0=a["D0"], z0=a["z0"], zbar=a["zbar"],
                          rho=a["rho"], s0sq_ratio=cs["s0sq_ratio"], targets=cs["targets"])
        m = cs["m"]
        rec = dict(name=cs["name"], m=m, targets=list(cs["targets"]), s0sq_over_sZ2=cs["s0sq_ratio"],
                   s0sq=_out(cs["s0sq_ratio"] * model["sZ2"]), window=[cs["tau"], cs["T"]])
        # closed-form nu, d_* (Lemma c1:lem-std) as intervals
        sZ2 = model["sZ2"]
        s0sq = cs["s0sq_ratio"] * sZ2
        rho2 = _X(iv, a["rho"]) ** 2
        small = cs["s0sq_ratio"] <= 1          # s0^2 <= sZ^2 exactly (integer ratio)
        Sm2 = rho2 + (s0sq if small else sZ2)
        Sp2 = rho2 + (sZ2 if small else s0sq)
        g = model["g"]
        nu_cf = g * model["Z"] * iv.exp(-g * _X(iv, cs["T"])) * Sm2 / (Sp2 * iv.sqrt(Sp2))
        rec["nu_closed_form"] = _out(nu_cf)
        if m >= 2:
            diffs = [model["Yj"][j] - model["Yj"][j + 1] for j in range(m - 1)]
            dmin = diffs[0]
            for dj in diffs[1:]:
                dmin = iv.mpf([min(dmin.a, dj.a), min(dmin.b, dj.b)])   # enclosure of the min
            d_cf = model["Z"] * dmin / iv.sqrt(Sp2)
            rec["dstar_closed_form"] = _out(d_cf)
        # A_S + e0 Y Y_j >= S_-^2 over t >= 0  (affine in u = Y Y_j in (0,1]): endpoint values
        rec["affine_endpoint_values"] = [_out(model["A"]), _out(model["A"] + model["e0"])]
        # certified [lower, upper] bounds of min over I of n_j' and of n_j - n_{j+1}
        core = model["core"]
        rec["min_nprime"] = [iv_min_over(iv, (lambda t, j=j: core(t)["n1"][j]), cs["tau"], cs["T"], args.pieces)
                             for j in range(m)]
        if m >= 2:
            rec["min_gap"] = [iv_min_over(iv, (lambda t, j=j: core(t)["n"][j] - core(t)["n"][j + 1]),
                                          cs["tau"], cs["T"], args.pieces) for j in range(m - 1)]
        # old sector ratio omega (Remark th10:rem-thm1): certified enclosure of the value at the
        # exact decimal point t = t_1 + (t_2 - t_1) k/400 maximizing the lower end (a lower bound of sup)
        if cs["name"].endswith("wide"):
            from fractions import Fraction
            t1f, t2f = Fraction(cs["targets"][0]), Fraction(cs["targets"][1])
            best = None
            Delta = model["Z"] * (model["Yj"][0] - model["Yj"][1])
            for k in range(1, 400):
                ttf = t1f + (t2f - t1f) * k / 400
                t_iv = iv.mpf(ttf.numerator) / iv.mpf(ttf.denominator)
                Y = iv.exp(-g * t_iv)
                S2 = model["A"] + model["e0"] * Y * Y
                dlogS = abs(g * model["e0"] * Y * Y / S2)
                xprime = g * model["Z"] * Y           # l0 = 1
                om = Delta * dlogS / (2 * xprime)
                if best is None or om.a > best[1].a:
                    best = (ttf, om)
            rec["old_omega_certified_lower_bound"] = dict(
                t="%d/%d" % (best[0].numerator, best[0].denominator), t_decimal=str(float(best[0])),
                omega=_out(best[1]), exceeds_two_thirds=bool((best[1] - iv.mpf(2) / 3).a > 0),
                exceeds_one=bool((best[1] - 1).a > 0))
        rec["census"] = []
        for eps_s in cs["eps_list"]:
            eps = _X(iv, eps_s)
            for w_s in weights[m]:
                w = [_X(iv, x) for x in w_s]
                res = iv_census(iv, model, eps, w, cs["tau"], cs["T"], max_depth=args.max_depth)
                res.update(eps=eps_s, w=list(w_s))
                rec["census"].append(res)
                print(cs["name"], eps_s, w_s, "ok" if res["ok"] else "FAIL", res["kinds"], res["leaves"],
                      "%.1fs" % (time.time() - t0), flush=True)
        out["cases"].append(rec)
    out["all_ok"] = all(c["ok"] for r in out["cases"] for c in r["census"])
    out["n_census"] = sum(len(r["census"]) for r in out["cases"])
    out["n_census_ok"] = sum(c["ok"] for r in out["cases"] for c in r["census"])
    out["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "c1_interval_certificates.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path, "all_ok", out["all_ok"], out["n_census_ok"], "/", out["n_census"])


# ----------------------------------------------------------------------------
# float64 census of the full G (with the exact contact factor)
# ----------------------------------------------------------------------------
def _fk():
    sys.path.insert(0, str(HERE))
    import exact_m_prr_fk_exact_law as fk  # noqa: E402
    return fk


def _G_continuum(t, *, m, s0sq, eps, w, c=None):
    import numpy as np
    a = ANCHOR
    sZ2 = a["D0"] / (2 * a["gamma"])
    Y = np.exp(-a["gamma"] * t)
    S2 = a["rho"] ** 2 + sZ2 + (s0sq - sZ2) * Y * Y
    Z = abs(a["z0"] - a["zbar"])
    H = np.zeros_like(t)
    for wj, tj in zip(w, TARGETS[m]):
        n = Z * (np.exp(-a["gamma"] * tj) - Y) / np.sqrt(S2)
        H += wj * np.exp(-n * n / (2 * eps * eps))
    G = H / (np.sqrt(2 * np.pi) * eps * np.sqrt(S2))   # W = 1
    return G if c is None else c * G


def _census_float(t, y):
    import numpy as np
    d = np.diff(y)
    s = np.sign(d)
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    kinds = ["max" if s[i] > 0 else "min" for i in idx]
    return kinds, [float(t[i + 1]) for i in idx]


def cmd_floatG(args):
    import numpy as np
    fk = _fk()
    a = ANCHOR
    t = np.linspace(a["tau"], a["T"], 300001)
    out = dict(item="C1_preparation_count", part="floatG",
               note="float64 census of the continuum free rate G = c(t) S^{-1} sum_j w_j exp(-n_j^2/(2eps^2)) "
                    "(W=1) on a 300001-point grid of I, c(t) from exact_m_prr_fk_exact_law.mean_contact_curve",
               rows=[])
    for m in (2, 3):
        for eps in (0.1, 0.05):
            spec = fk.EnsembleSpec(m=m, eps=eps)
            c = fk.mean_contact_curve(spec, spec.contact_a, t)
            for prep, ratio in PREPS.items():
                s0sq = ratio * a["D0"] / (2 * a["gamma"])
                w = tuple([1.0 / m] * m)
                G = _G_continuum(t, m=m, s0sq=s0sq, eps=eps, w=w, c=c)
                kinds, where = _census_float(t, G)
                out["rows"].append(dict(m=m, eps=eps, prep=prep, s0sq=s0sq, w=list(w), kinds=kinds,
                                        crit_t=where, c_tau=float(c[0]), c_T=float(c[-1]),
                                        ok=kinds == (["max", "min"] * (m - 1) + ["max"])))
                print(m, eps, prep, kinds, [round(x, 4) for x in where], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1_floatG_census.json").write_text(json.dumps(out, indent=1))
    print("wrote floatG")


# ----------------------------------------------------------------------------
# FK small-budget transfer check on stored N4 ensembles
# ----------------------------------------------------------------------------
def cmd_fk(args):
    import numpy as np
    fk = _fk()
    out = dict(item="C1_preparation_count", part="fk_small_B",
               note=("FK exact-law estimator on the stored N4 ensembles (seed 20260923, tag 83, 2e5 paths "
                     "each; no new simulation). f/B on the production window bins vs the bin average of the "
                     "discrete free rate G_n (free_exposure_discrete, gate='contact'); census by "
                     "derivative_signs (zthr=5, batch SEs from 20 path groups)."),
               budgets=args.budgets, rows=[])
    for m in (2, 3):
        for prep in PREPS:
            name = "n4_m%d_eps0.1_%s" % (m, prep)
            ens = fk.load_ensemble(name)
            ens.cache_in_memory = True
            spec = ens.spec if hasattr(ens, "spec") else fk.EnsembleSpec.from_dict(ens.index["spec"])
            w = tuple([1.0 / m] * m)
            fr = fk.free_exposure_discrete(spec, w, gate="contact")
            tn, Gn = np.asarray(fr["t"]), np.asarray(fr["G"])
            cumG = np.concatenate([[0.0], np.cumsum(Gn * spec.dt)])
            tk = np.concatenate([[0.0], tn])   # kill time of step n is t_n = (n+1) dt
            for B in args.budgets:
                law = fk.exact_law(ens, B, w)
                edges = np.asarray(law["edges"])
                Gbin = np.diff(np.interp(edges, tk, cumG)) / np.diff(edges)
                fB = np.asarray(law["density"]) / B
                gb = np.asarray(law["group_density"]) / B
                ds = fk.derivative_signs(law["t"], fB, gb)
                dsG = fk.derivative_signs(law["t"], Gbin, np.tile(Gbin, (4, 1)) +
                                          1e-12 * np.arange(4)[:, None])
                rel = float(np.max(np.abs(fB - Gbin)) / np.max(Gbin))
                row = dict(m=m, prep=prep, var_z0_scale=PREPS[prep], B=B, n_paths=int(law["N"]),
                           f_over_B_maxima=ds["n_maxima"], f_over_B_minima=ds["n_minima"],
                           f_over_B_maxima_t=ds["maxima_t"], f_over_B_minima_t=ds["minima_t"],
                           G_maxima=dsG["n_maxima"], G_minima=dsG["n_minima"],
                           sup_rel_dev_fB_vs_G=rel,
                           ok=(ds["n_maxima"] == m and ds["n_minima"] == m - 1))
                out["rows"].append(row)
                print(name, B, row["f_over_B_maxima"], row["f_over_B_minima"], "rel dev %.4f" % rel, flush=True)
            ens.clear_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1_fk_small_budget.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote fk")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("interval")
    p.add_argument("--prec", type=int, default=96)
    p.add_argument("--pieces", type=int, default=2048)
    p.add_argument("--max-depth", type=int, default=40)
    p.set_defaults(func=cmd_interval)
    p = sub.add_parser("floatG")
    p.set_defaults(func=cmd_floatG)
    p = sub.add_parser("fk")
    p.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.1])
    p.set_defaults(func=cmd_fk)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
