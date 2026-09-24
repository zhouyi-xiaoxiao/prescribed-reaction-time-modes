#!/usr/bin/env python3
"""Independent checks of the task-A general-transport counterexamples.

Sources that are checked (read-only):
  W  = notes/gap_diagnosis_20260923/webpro_A_general_exact_count.md
       (GPT-6 Pro web session, 2026-09-24): counterexample CE1 (one stripe,
       variance-driven action well), CE2 (two stripes, extra maximum inside a
       genuine gap), and the stated interval enclosures.
  CX = manuscript/cnsns_submission/theory/GX_A_general_exact_count.tex
       (Codex GPT-6 Astra): Proposition gx:spiral-counterexample and
       Lemma gx:spiral-endpoint (elliptic affine spiral drift).

Model of W, CE1 (all covariances below are in units of eps^2):
  dX = (1+Y) dt + 1e-2 eps dW1,  dY = -Y dt + eps dW2,
  (X0, Y0) = eps xi,  xi ~ N(0, 1e-6 I2),
  V = B phi_{eps rho}(x - c) phi(y),  c = rho = 1e-2,  window [1/200, 8].
  S(t) = rho^2 + C11(t),  J(t) = (t - c)^2 / (2 S(t)).
CE2: same diffusion, stripes c1 = 0.01, c2 = 10, rho = 0.01,
  J_i(t) = (t - c_i)^2 / (2 S(t)).
Spiral (CX): dZ = A(Z - (1,0)) dt + eps sqrt(2/5) dW,
  A = [[-1/5, -1], [1, -1/5]],  Z0 = (-1,0) + eps xi,  xi ~ N(0, I2),
  V = (B/eps) phi(x/eps) phi(y).

Parts
  (1) CE1 covariance: closed form (variation of constants, derived in the
      comments below) checked against (a) the Lyapunov ODE residual with
      60-digit numerical differentiation and (b) the Van Loan block matrix
      exponential (mp.expm, 60 digits); identity with W's C11(t).
  (2) mpmath.iv (60 digits, rational inputs) enclosures of J'(t) at
      t = 1/200, 1/20, 1/5, 8 and containment in W's intervals.
  (3) CE2: enclosures of J1 at t = 0.2, 2, 4 (plus J2 and the action gap).
  (4) CX spiral: mean/covariance, the unique transversal crossing, the return
      of the normal mean to h ~ 0.43 near 2 pi, the action comparisons, the
      explicit thresholds and the endpoint-lemma constants.
  (5) Illustration (not a proof): the killed density of CE1 at eps = 0.3,
      B = 1, computed as f(t) = B E[g(Z_t)] * alpha(t), where E[g(Z_t)] is the
      exact Gaussian integral and alpha(t) = E~[exp(-int_0^t V)] is estimated
      with bridges drawn EXACTLY (for the time-discretised chain) from the
      terminal-tilted Gaussian path law (backward sampling). Validated against
      plain forward Monte Carlo at eps = 1. Also exact free-envelope
      (B -> 0 limit) counts on dense grids for CE1, CE2 and the spiral.

Seeds: numpy SeedSequence([20260924, 61, tag]). One process.
Python: python (mpmath 1.4.1, numpy).
Output: artifacts/data/exact_m_fixed_budget/GX_counterexamples/checks.json
"""
from __future__ import annotations

import decimal
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from mpmath import iv, mp

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
OUT_DIR = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "GX_counterexamples"
SRC_W = REPORT / "notes" / "gap_diagnosis_20260923" / "webpro_A_general_exact_count.md"
SRC_CX = REPORT / "manuscript" / "cnsns_submission" / "theory" / "GX_A_general_exact_count.tex"

SEED_BASE = 20260924
STREAM_TAG = 61

mp.dps = 60
iv.dps = 60


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256(p: Path) -> str:
    # The two checked sources are development notes; they are not part of the
    # public reproduction archive, where the hash is recorded as absent.
    if not p.exists():
        return "absent (source not archived)"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mpf_tuple_to_fraction(t) -> Fraction:
    sign, man, exp, _bc = t
    v = Fraction(int(man)) * (Fraction(2) ** int(exp))
    return -v if sign else v


def iv_bounds(x):
    """Exact rational endpoints of an mpmath iv interval."""
    lo_t, hi_t = x._mpi_
    return mpf_tuple_to_fraction(lo_t), mpf_tuple_to_fraction(hi_t)


def dec_round(fr: Fraction, n: int, rounding: str) -> str:
    """Decimal string of fr with n significant digits, rounded in the given
    direction (decimal.ROUND_FLOOR or decimal.ROUND_CEILING); the division is
    exact up to that single directed rounding."""
    ctx = decimal.Context(prec=n, rounding=rounding)
    q = ctx.divide(decimal.Decimal(fr.numerator), decimal.Decimal(fr.denominator))
    return format(q, "g") if q != 0 else "0"


def iv_str(x, n=12):
    """Outward-rounded decimal enclosure [lo, hi] of an mpmath iv interval:
    the lower endpoint is rounded down and the upper endpoint up to n
    significant digits, so the printed interval contains the exact one.
    (Release v1.1.0 printed both endpoints with round-to-nearest, which could
    exclude the true value in the last digit; the certified comparisons were
    and are made on the exact rational endpoints, not on these strings.)"""
    lo, hi = iv_bounds(x)
    return [dec_round(lo, n, decimal.ROUND_FLOOR), dec_round(hi, n, decimal.ROUND_CEILING)]


def strictly_inside(x, lo: Fraction, hi: Fraction) -> bool:
    a, b = iv_bounds(x)
    return a > lo and b < hi


def definitely_pos(x) -> bool:
    return iv_bounds(x)[0] > 0


def definitely_neg(x) -> bool:
    return iv_bounds(x)[1] < 0


def ivq(fr: Fraction):
    return iv.mpf(fr.numerator) / fr.denominator


def mpq(fr: Fraction):
    return mp.mpf(fr.numerator) / fr.denominator


# ---------------------------------------------------------------------------
# CE1/CE2 transport (W).  Derivation of the covariance (units eps^2):
#   Y_t = Y0 e^{-t} + int_0^t e^{-(t-u)} dW2(u)
#   X_t = X0 + t + Y0 (1 - e^{-t}) + int_0^t (1 - e^{-(t-u)}) dW2(u) + 1e-2 W1(t)
# with Var X0 = Var Y0 = 1e-6 and all sources independent, hence
#   S11 = 1e-6 + 1e-6 (1-e^{-t})^2 + 1e-4 t + int_0^t (1-e^{-v})^2 dv,
#         int_0^t (1-e^{-v})^2 dv = t - 2(1-e^{-t}) + (1-e^{-2t})/2,
#   S12 = 1e-6 e^{-t}(1-e^{-t}) + int_0^t (1-e^{-v}) e^{-v} dv
#       = 1e-6 e^{-t}(1-e^{-t}) + (1-e^{-t})^2/2,
#   S22 = 1e-6 e^{-2t} + (1-e^{-2t})/2.
# Mean: E X_t = t, E Y_t = 0.
# ---------------------------------------------------------------------------
def ce_consts(ctx):
    return ctx.mpf(1) / 10**4, ctx.mpf(1) / 10**6  # Q1=(1e-2)^2, S0=1e-6


def ce_sigma(t, ctx):
    Q1, S0 = ce_consts(ctx)
    e = ctx.exp(-t)
    e2 = e * e
    s11 = S0 * (1 + (1 - e) ** 2) + Q1 * t + t - 2 * (1 - e) + (1 - e2) / 2
    s12 = S0 * e * (1 - e) + (1 - e) ** 2 / 2
    s22 = S0 * e2 + (1 - e2) / 2
    return s11, s12, s22


def ce_C11_web(t, ctx):
    """W's formula, typed verbatim."""
    Q1, S0 = ce_consts(ctx)
    return S0 * (1 + (1 - ctx.exp(-t)) ** 2) + Q1 * t + t - 2 * (1 - ctx.exp(-t)) + (1 - ctx.exp(-2 * t)) / 2


def ce_S_Sp(t, ctx, rho):
    Q1, S0 = ce_consts(ctx)
    e = ctx.exp(-t)
    S = rho**2 + ce_sigma(t, ctx)[0]
    Sp = 2 * S0 * (1 - e) * e + Q1 + (1 - e) ** 2  # = 2 S12 + Q1 (Lyapunov)
    return S, Sp


def ce_J(t, ctx, c, rho):
    S, _ = ce_S_Sp(t, ctx, rho)
    return (t - c) ** 2 / (2 * S)


def ce_Jp(t, ctx, c, rho):
    S, Sp = ce_S_Sp(t, ctx, rho)
    return (t - c) / S - (t - c) ** 2 * Sp / (2 * S**2)


def part1_covariance():
    out = {}
    A = mp.matrix([[0, 1], [0, -1]])
    Q1, S0 = ce_consts(mp)
    Q = mp.matrix([[Q1, 0], [0, 1]])
    Sig0 = mp.matrix([[S0, 0], [0, S0]])
    ts = [mp.mpf(1) / 200, mp.mpf(1) / 100, mp.mpf(1) / 20, mp.mpf(1) / 5, mp.mpf(1), mp.mpf(2),
          mp.mpf(4), mp.mpf(8), mp.mpf(11)]
    max_ode = mp.mpf(0)
    max_vl = mp.mpf(0)
    max_web = mp.mpf(0)
    max_sp = mp.mpf(0)
    rows = []
    for t in ts:
        s11, s12, s22 = ce_sigma(t, mp)
        Sig = mp.matrix([[s11, s12], [s12, s22]])
        d11 = mp.diff(lambda u: ce_sigma(u, mp)[0], t)
        d12 = mp.diff(lambda u: ce_sigma(u, mp)[1], t)
        d22 = mp.diff(lambda u: ce_sigma(u, mp)[2], t)
        rhs = A * Sig + Sig * A.T + Q
        res = max(abs(d11 - rhs[0, 0]), abs(d12 - rhs[0, 1]), abs(d22 - rhs[1, 1]))
        max_ode = max(max_ode, res)
        # Van Loan: expm([[-A, Q],[0, A^T]] t) = [[F1, G1],[0, F2]], int = F2^T G1
        M = mp.zeros(4, 4)
        for i in range(2):
            for j in range(2):
                M[i, j] = -A[i, j] * t
                M[i, j + 2] = Q[i, j] * t
                M[i + 2, j + 2] = A[j, i] * t
        E = mp.expm(M)
        F2 = mp.matrix([[E[2, 2], E[2, 3]], [E[3, 2], E[3, 3]]])
        G1 = mp.matrix([[E[0, 2], E[0, 3]], [E[1, 2], E[1, 3]]])
        eAt = F2.T
        SigVL = eAt * Sig0 * eAt.T + F2.T * G1
        vl = max(abs(SigVL[i, j] - Sig[i, j]) for i in range(2) for j in range(2))
        max_vl = max(max_vl, vl)
        web = abs(ce_C11_web(t, mp) - s11)
        max_web = max(max_web, web)
        rho = mp.mpf(1) / 100
        Sp_num = mp.diff(lambda u: ce_S_Sp(u, mp, rho)[0], t)
        sp = abs(Sp_num - ce_S_Sp(t, mp, rho)[1])
        max_sp = max(max_sp, sp)
        rows.append({"t": mp.nstr(t, 8), "C11": mp.nstr(s11, 15), "C12": mp.nstr(s12, 15),
                     "C22": mp.nstr(s22, 15)})
    out["closed_form_eps2_units"] = {
        "C11": "1e-6[1+(1-e^{-t})^2] + 1e-4 t + t - 2(1-e^{-t}) + (1-e^{-2t})/2",
        "C12": "1e-6 e^{-t}(1-e^{-t}) + (1-e^{-t})^2/2",
        "C22": "1e-6 e^{-2t} + (1-e^{-2t})/2",
        "mean": "(t, 0)",
        "Sprime": "2e-6 (1-e^{-t}) e^{-t} + 1e-4 + (1-e^{-t})^2  (= 2 C12 + 1e-4)",
    }
    out["values"] = rows
    out["max_lyapunov_ode_residual"] = mp.nstr(max_ode, 5)
    out["max_vanloan_expm_minus_closed_form"] = mp.nstr(max_vl, 5)
    out["max_web_C11_minus_closed_form"] = mp.nstr(max_web, 5)
    out["max_Sprime_formula_minus_numdiff"] = mp.nstr(max_sp, 5)
    tol = mp.mpf(10) ** -40
    out["agree_web_C11"] = bool(max_ode < tol and max_vl < tol and max_web < tol and max_sp < tol)
    # mean ODE: m' = A m + a with m = (t,0), a = (1,0): (0*t + 1*0 + 1, 0*t - 0) = (1, 0) = m'
    out["mean_check"] = "m(t)=(t,0): A m + a = (1,0) = m'(t); deterministic path crosses x=c once, at t=c, speed 1"
    # exact budget: int phi_{eps rho}(x-c) dx * int phi(y) dy = 1
    eps_t = mp.mpf(3) / 10
    rho = mp.mpf(1) / 100
    I1 = mp.quad(lambda x: mp.npdf(x, rho, eps_t * rho), [-mp.inf, rho - 1, rho, rho + 1, mp.inf])
    I2 = mp.quad(lambda y: mp.npdf(y), [-mp.inf, 0, mp.inf])
    out["budget_integral_eps0.3"] = mp.nstr(I1 * I2, 20)
    # exact free density exponent including the eps^2 correlation correction
    out["note_exact_free_density"] = (
        "E g(Z_t) = (eps rho)^{-1}(2 pi)^{-1} det(I+P M)^{-1/2} exp(-(t-c)^2/(2 eps^2 S_eps)), "
        "P = eps^2 Sigma(t), M = diag(1/(eps rho)^2, 1), S_eps = S - eps^2 C12^2/(1+eps^2 C22); "
        "hence -eps^2 log E g = J + O(eps^2 log(1/eps)) with J=(t-c)^2/(2S) the leading action used by W")
    return out


def part2_Jprime():
    c = iv.mpf(1) / 100
    rho = iv.mpf(1) / 100
    claims = [
        (Fraction(1, 200), Fraction(-50), Fraction(-49)),
        (Fraction(1, 20), Fraction(180), Fraction(182)),
        (Fraction(1, 5), Fraction(-24), Fraction(-22)),
        (Fraction(8), Fraction(47, 100), Fraction(48, 100)),
    ]
    rows = []
    all_ok = True
    for t, lo, hi in claims:
        ti = ivq(t)
        Jp = ce_Jp(ti, iv, c, rho)
        # second, algebraically rearranged form: (t-c)/S * (1 - (t-c) S'/(2S))
        S, Sp = ce_S_Sp(ti, iv, rho)
        Jp2 = (ti - c) / S * (1 - (ti - c) * Sp / (2 * S))
        J = ce_J(ti, iv, c, rho)
        # independent float cross-check by 60-digit numerical differentiation
        Jp_num = mp.diff(lambda u: ce_J(u, mp, mp.mpf(1) / 100, mp.mpf(1) / 100), mpq(t))
        ok = strictly_inside(Jp, lo, hi) and strictly_inside(Jp2, lo, hi)
        a, b = iv_bounds(Jp)
        num_in = (mpq(a) - mp.mpf(10) ** -40 <= Jp_num <= mpq(b) + mp.mpf(10) ** -40)
        all_ok &= ok and bool(num_in)
        rows.append({
            "t": str(t), "claimed_open_interval": [str(lo), str(hi)],
            "Jprime_enclosure": iv_str(Jp, 15), "Jprime_enclosure_width": dec_round(b - a, 3, decimal.ROUND_CEILING),
            "Jprime_numdiff": mp.nstr(Jp_num, 15), "J_enclosure": iv_str(J, 12),
            "S_enclosure": iv_str(S, 12), "Sprime_enclosure": iv_str(Sp, 12),
            "inside_claimed": bool(ok),
            "implied_sign_f_prime": "+" if b < 0 else ("-" if a > 0 else "?"),
        })
    # critical points of J on [1/200, 8] (float, 60 digits; not interval-certified)
    Jp_mp = lambda u: ce_Jp(u, mp, mp.mpf(1) / 100, mp.mpf(1) / 100)
    J_mp = lambda u: ce_J(u, mp, mp.mpf(1) / 100, mp.mpf(1) / 100)
    t_max = mp.findroot(Jp_mp, (mp.mpf("0.05"), mp.mpf("0.2")), solver="illinois")
    t_min = mp.findroot(Jp_mp, (mp.mpf("0.2"), mp.mpf("8")), solver="illinois")
    # coarse scan for the number of sign changes of J' on [1/200, 8]
    grid = [mp.mpf(1) / 200 + (mp.mpf(8) - mp.mpf(1) / 200) * k / 20000 for k in range(20001)]
    sgn = [mp.sign(Jp_mp(u)) for u in grid]
    changes = sum(1 for k in range(len(sgn) - 1) if sgn[k] != sgn[k + 1])
    # mechanism identity: for t>c, J'<0 iff S'/S > 2/(t-c)  (J' = (t-c)/S [1 - (t-c)S'/(2S)])
    mech_ok = True
    for u in [mp.mpf("0.02"), mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("1"), mp.mpf("3"), mp.mpf("8")]:
        S, Sp = ce_S_Sp(u, mp, mp.mpf(1) / 100)
        mech_ok &= ((Jp_mp(u) < 0) == (Sp / S > 2 / (u - mp.mpf(1) / 100)))
    return {
        "rows": rows,
        "all_inside_claimed": bool(all_ok),
        "J_local_max_after_crossing": {"t": mp.nstr(t_max, 10), "J": mp.nstr(J_mp(t_max), 10)},
        "J_secondary_min": {"t": mp.nstr(t_min, 10), "J": mp.nstr(J_mp(t_min), 10)},
        "J_at_8": mp.nstr(J_mp(mp.mpf(8)), 10),
        "sign_changes_of_Jprime_on_grid_[1/200,8]_20001pts": changes,
        "note_sign_changes": "expected 3: at the crossing t=c (J=0), at the J-maximum, at the secondary J-minimum",
        "mechanism_identity_ok": bool(mech_ok),
        "logic": ("J'(1/200)<0, J'(1/20)>0, J'(1/5)<0, J'(8)>0 give, through W's relative estimate "
                  "|eps^2 f'/f + J'| <= C sqrt(eps), f'(1/200)>0>f'(1/20) and f'(1/5)>0>f'(8): at least two "
                  "interior maxima. The relative estimate itself is NOT re-proved here."),
    }


def part3_two_stripe():
    rho = iv.mpf(1) / 100
    c1 = iv.mpf(1) / 100
    c2 = iv.mpf(10)
    claims = [(Fraction(1, 5), Fraction(745, 100), Fraction(746, 100)),
              (Fraction(2), Fraction(259, 100), Fraction(261, 100)),
              (Fraction(4), Fraction(313, 100), Fraction(314, 100))]
    rows = []
    ok_all = True
    Jmin_lo = {}
    for t, lo, hi in claims:
        ti = ivq(t)
        J1 = ce_J(ti, iv, c1, rho)
        J2 = ce_J(ti, iv, c2, rho)
        ok = strictly_inside(J1, lo, hi)
        ok_all &= ok
        a1, b1 = iv_bounds(J1)
        a2, b2 = iv_bounds(J2)
        Jmin_lo[str(t)] = min(a1, a2)
        rows.append({"t": str(t), "claimed_open_interval_J1": [str(lo), str(hi)],
                     "J1_enclosure": iv_str(J1, 12), "J2_enclosure": iv_str(J2, 12),
                     "J1_inside_claimed": bool(ok), "stripe1_dominates": bool(b1 < a2)})
    J1_2_hi = iv_bounds(ce_J(iv.mpf(2), iv, c1, rho))[1]
    gap = min(Jmin_lo["1/5"], Jmin_lo["4"]) - J1_2_hi
    # stripe-2 crossing sanity: J2(10)=0 and J2 large in the gap
    return {"rows": rows, "all_inside_claimed": bool(ok_all),
            # rounded DOWN (a lower bound must stay a lower bound when printed)
            "action_gap_lower_bound": dec_round(gap, 8, decimal.ROUND_FLOOR),
            "gap_positive": bool(gap > 0),
            "note": ("min over stripes of the action at t=0.2 and t=4 minus J1(2) (upper bound); a positive gap "
                     "plus the survival sandwich B e^{-D/eps} H <= f <= B H gives an interior maximum of f in "
                     "(0.2,4) for small eps (three-time comparison), in addition to the two crossing maxima")}


# ---------------------------------------------------------------------------
# Spiral (CX)
# ---------------------------------------------------------------------------
def sp_mu(t, ctx):
    e = ctx.exp(-t / 5)
    return 1 - 2 * e * ctx.cos(t), -2 * e * ctx.sin(t)


def sp_mux_prime(t, ctx):
    return 2 * ctx.exp(-t / 5) * (ctx.cos(t) / 5 + ctx.sin(t))


def part4_spiral():
    out = {}
    pi = iv.pi
    # (a) mean solves m' = A(m - c), m(0) = (-1, 0); covariance eps^2 I is stationary
    res = mp.mpf(0)
    for t in [mp.mpf(k) / 7 for k in range(0, 60)]:
        mx, my = sp_mu(t, mp)
        dmx = mp.diff(lambda u: sp_mu(u, mp)[0], t)
        dmy = mp.diff(lambda u: sp_mu(u, mp)[1], t)
        res = max(res, abs(dmx - (-(mx - 1) / 5 - my)), abs(dmy - ((mx - 1) - my / 5)))
    out["mean_ode_residual"] = mp.nstr(res, 5)
    out["mean_initial"] = [mp.nstr(v, 5) for v in sp_mu(mp.mpf(0), mp)]
    out["A_plus_AT_plus_2over5_I_is_zero"] = True  # [[-2/5,0],[0,-2/5]] + 2/5 I = 0, exact rationals
    # (b) unique transversal crossing
    tau = mp.findroot(lambda u: sp_mu(u, mp)[0], mp.mpf("0.925"))
    delta = mp.mpf(10) ** -30
    left = sp_mu(iv.mpf(mp.nstr(tau - delta, 50)), iv)[0]
    right = sp_mu(iv.mpf(mp.nstr(tau + delta, 50)), iv)[0]
    out["tau"] = mp.nstr(tau, 20)
    out["tau_bracket_certified"] = bool(definitely_neg(left) and definitely_pos(right))
    # monotone on [0, 1.5708] (a superset of [0, pi/2]): mu_x' > 0, certified on dyadic segments
    # [k/4096, (k+1)/4096] (exact binary endpoints, so the segments cover the range without gaps)
    H_ = 4096
    mono_ok = True
    n_mono = 0
    for k in range(0, int(math.ceil(1.5708 * H_))):
        seg = iv.mpf([k / H_, (k + 1) / H_])
        mono_ok &= definitely_pos(sp_mux_prime(seg, iv))
        n_mono += 1
    out["mux_prime_positive_on_[0,1.5708]_certified"] = bool(mono_ok)
    out["mux_prime_segments"] = n_mono
    mux_pi6 = sp_mu(pi / 6, iv)[0]
    d = iv.sqrt(3) * iv.exp(-pi / 30) - 1
    out["mux(pi/6)_enclosure"] = iv_str(mux_pi6)
    out["d_enclosure"] = iv_str(d)
    out["mux(pi/6)=-d_consistent"] = bool(iv_bounds(mux_pi6 + d)[0] <= 0 <= iv_bounds(mux_pi6 + d)[1])
    out["mux(pi/2)_enclosure"] = iv_str(sp_mu(pi / 2, iv)[0])
    out["tau_in_(pi/6,pi/2)"] = bool(math.pi / 6 < float(tau) < math.pi / 2)
    # [1.5707, 4.7125] (superset of [pi/2, 3pi/2]): mu_x > 0 on dyadic segments;
    # t >= 3pi/2: mu_x >= 1 - 2 e^{-3pi/10} > 0
    pos_ok = True
    n_pos = 0
    for k in range(int(math.floor(1.5707 * H_)), int(math.ceil(4.7125 * H_))):
        seg = iv.mpf([k / H_, (k + 1) / H_])
        pos_ok &= definitely_pos(sp_mu(seg, iv)[0])
        n_pos += 1
    out["mux_positive_on_[1.5707,4.7125]_certified"] = bool(pos_ok)
    out["mux_positive_segments"] = n_pos
    tail = 1 - 2 * iv.exp(-3 * pi / 10)
    out["tail_bound_1-2e^{-3pi/10}"] = iv_str(tail)
    out["no_zero_for_t>=3pi/2_certified"] = bool(definitely_pos(tail))
    out["mux_prime_at_tau"] = mp.nstr(sp_mux_prime(tau, mp), 12)
    out["unique_transversal_crossing_on_[0,inf)"] = bool(out["tau_bracket_certified"] and mono_ok and pos_ok
                                                         and definitely_pos(tail))
    # (c) return near 2 pi
    h = 1 - 2 * iv.exp(-2 * pi / 5)
    out["h=mux(2pi)_enclosure"] = iv_str(sp_mu(2 * pi, iv)[0])
    out["h_formula_enclosure"] = iv_str(h)
    out["muy(2pi)_enclosure"] = iv_str(sp_mu(2 * pi, iv)[1], 5)
    tstar = 2 * mp.pi - mp.atan(mp.mpf(1) / 5)
    out["argmin_mux_on_[3pi/2,5pi/2]"] = mp.nstr(tstar, 12)
    out["min_mux_on_[3pi/2,5pi/2]"] = mp.nstr(sp_mu(tstar, mp)[0], 12)
    out["mux_prime_at_argmin"] = mp.nstr(sp_mux_prime(tstar, mp), 5)
    out["mux(3pi/2), mux(5pi/2)"] = [iv_str(sp_mu(3 * pi / 2, iv)[0], 10), iv_str(sp_mu(5 * pi / 2, iv)[0], 10)]
    # (d) actions J = mu_x^2 / 4 and the comparisons in Prop gx:spiral-counterexample
    a0 = d**2 / 4
    Delta = (1 - h**2) / 4
    J = lambda t: sp_mu(t, iv)[0] ** 2 / 4
    out["J(pi/6)=a0"] = iv_str(J(pi / 6))
    out["a0"] = iv_str(a0)
    out["J(pi/2)"] = iv_str(J(pi / 2))
    out["J(3pi/2), J(5pi/2)"] = [iv_str(J(3 * pi / 2)), iv_str(J(5 * pi / 2))]
    out["J(2pi)=h^2/4"] = iv_str(J(2 * pi))
    out["Delta=(1-h^2)/4"] = iv_str(Delta)
    muy_tau = sp_mu(tau, mp)[1]
    out["muy(tau)"] = mp.nstr(muy_tau, 12)
    out["|muy(tau)|<=2"] = bool(abs(muy_tau) <= 2)
    out["comparisons_ok"] = bool(definitely_pos(a0) and definitely_pos(Delta)
                                 and definitely_pos(J(pi / 2) - a0) and iv_bounds(d)[1] < 1)
    # free density formula: E phi_eps(X) with X~N(mu_x, eps^2) and E phi(Y) with Y~N(mu_y, eps^2)
    fd = []
    for eps_s, t_s in [("0.3", "1"), ("0.5", "6.283185307179586")]:
        eps_v = mp.mpf(eps_s)
        t_v = mp.mpf(t_s)
        mx, my = sp_mu(t_v, mp)
        qx = mp.quad(lambda x: mp.npdf(x, 0, eps_v) * mp.npdf(x, mx, eps_v), [-mp.inf, mx - 1, mx, mx + 1, mp.inf])
        qy = mp.quad(lambda y: mp.npdf(y) * mp.npdf(y, my, eps_v), [-mp.inf, my - 1, my, my + 1, mp.inf])
        closed = (mp.exp(-mx**2 / (4 * eps_v**2) - my**2 / (2 * (1 + eps_v**2)))
                  / (2 * mp.pi * eps_v * mp.sqrt(2 * (1 + eps_v**2))))
        fd.append({"eps": eps_s, "t": t_s, "rel_err": mp.nstr(abs(qx * qy / closed - 1), 5)})
    out["free_density_formula_quadrature"] = fd
    # (e) explicit thresholds for B_max = 1
    Bmax = iv.mpf(1)
    L = 5 * Bmax / 4
    eps0 = min(1.0, float(iv_bounds(a0 / (4 * (1 + L)))[0]),
               float(iv_bounds(iv.sqrt(a0 / 8))[0]), float(iv_bounds(Delta / (2 * (1 + L)))[0]))
    e0 = iv.mpf(eps0)
    out["L_check_(1/2pi)*(5pi/2)"] = iv_str(iv.mpf(1) / (2 * pi) * 5 * pi / 2)
    out["eps0_for_Bmax=1"] = eps0
    out["eps0_threshold_implications_ok"] = bool(
        iv_bounds(L * e0 + 2 * e0**2 - a0 / 2)[1] <= 0 and iv_bounds(L * e0 - Delta / 2)[1] <= 0)
    # (f) endpoint lemma gx:spiral-endpoint
    T0 = 3 * pi / 2
    mx0 = sp_mu(T0, iv)[0]
    Jp_T0 = mx0 * sp_mux_prime(T0, iv) / 2
    out["J'(3pi/2)_enclosure"] = iv_str(Jp_T0, 15)
    out["-e^{-3pi/10}"] = iv_str(-iv.exp(-3 * pi / 10), 15)
    out["J'(3pi/2)=-e^{-3pi/10}"] = bool(abs(float(iv_bounds(Jp_T0 + iv.exp(-3 * pi / 10))[1])) < 1e-40
                                         and abs(float(iv_bounds(Jp_T0 + iv.exp(-3 * pi / 10))[0])) < 1e-40)
    # generator identity L g / g = eps^{-2} P + R at random points (60-digit numerical derivatives)
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, STREAM_TAG, 1]))
    gen_res = mp.mpf(0)
    for _ in range(12):
        x0, y0 = (mp.mpf(float(v)) for v in rng.uniform(-2, 2, 2))
        eps_v = mp.mpf(float(rng.uniform(0.05, 1.0)))
        g = lambda x, y: mp.npdf(x, 0, eps_v) * mp.npdf(y)
        gx = mp.diff(g, (x0, y0), (1, 0))
        gy = mp.diff(g, (x0, y0), (0, 1))
        gxx = mp.diff(g, (x0, y0), (2, 0))
        gyy = mp.diff(g, (x0, y0), (0, 2))
        bx = -(x0 - 1) / 5 - y0
        by = (x0 - 1) - y0 / 5
        Lg = bx * gx + by * gy + eps_v**2 / 5 * (gxx + gyy)
        P = mp.mpf(2) / 5 * x0**2 - x0 / 5 + x0 * y0
        R = -by * y0 - mp.mpf(1) / 5 + eps_v**2 / 5 * (y0**2 - 1)
        gen_res = max(gen_res, abs(Lg / g(x0, y0) - (P / eps_v**2 + R)) / (1 + abs(P / eps_v**2 + R)))
    out["generator_identity_max_rel_residual"] = mp.nstr(gen_res, 5)
    # cancellation P(z0) = -J'(t), z0 = (mu_x/2, mu_y)
    canc = mp.mpf(0)
    for t in [mp.mpf(k) / 3 for k in range(0, 25)]:
        mx, my = sp_mu(t, mp)
        P = mp.mpf(2) / 5 * (mx / 2) ** 2 - (mx / 2) / 5 + (mx / 2) * my
        Jp = mx * sp_mux_prime(t, mp) / 2
        canc = max(canc, abs(P + Jp))
    out["P(z0)+J'_max_abs"] = mp.nstr(canc, 5)
    # Hessian of P: [[4/5,1],[1,0]] -> spectral norm 2/5 + sqrt(29)/5
    hn = mp.mpf(2) / 5 + mp.sqrt(mp.mpf(29)) / 5
    out["hessP_norm"] = mp.nstr(hn, 12)
    out["hessP_norm<3/2"] = bool(hn < mp.mpf(3) / 2)
    mgf = mp.sqrt(mp.mpf(4) / 3) * mp.sqrt(2)  # sup over eps in (0,1] of (3/4)^{-1/2}(1-1/(2(1+eps^2)))^{-1/2}
    out["tilted_mgf_sup"] = mp.nstr(mgf, 12)
    out["tilted_mgf_sup<2"] = bool(mgf < 2)
    # grad P on |z|<=3 ball is used with bound 5: max over the actual z_eps range
    T = mp.mpf(3) * mp.pi / 2
    Lb = 1 * T / (2 * mp.pi)
    Cr = 4 * (Lb + mp.log(2))
    Cs = 5 * mp.sqrt(Cr) + mp.mpf(19) / 4 * Cr + 40 + 1 / (2 * mp.pi)
    eps_star = min(mp.mpf(1), (mp.exp(-3 * mp.pi / 10) / (2 * Cs)) ** 2)
    out["endpoint_constants_Bmax=1_T=3pi/2"] = {"L": mp.nstr(Lb, 8), "C_r": mp.nstr(Cr, 8),
                                                 "C_*": mp.nstr(Cs, 8), "eps_*": mp.nstr(eps_star, 6)}
    out["all_ok"] = bool(out["unique_transversal_crossing_on_[0,inf)"] and out["comparisons_ok"]
                         and out["J'(3pi/2)=-e^{-3pi/10}"] and out["hessP_norm<3/2"]
                         and out["tilted_mgf_sup<2"] and out["eps0_threshold_implications_ok"]
                         and float(res) < 1e-40 and float(gen_res) < 1e-20 and float(canc) < 1e-40)
    return out


# ---------------------------------------------------------------------------
# Part 5: numerics (float64)
# ---------------------------------------------------------------------------
Q1F = 1e-4
S0F = 1e-6


def ce_sigma_np(t):
    t = np.asarray(t, float)
    om = -np.expm1(-t)          # 1 - e^{-t}
    om2 = -np.expm1(-2 * t)     # 1 - e^{-2t}
    e = np.exp(-t)
    s11 = S0F * (1 + om**2) + Q1F * t + (t - 2 * om + om2 / 2)
    s12 = S0F * e * om + om**2 / 2
    s22 = S0F * e * e + om2 / 2
    return s11, s12, s22


def ce_log_Eg(t, eps, c, rho):
    """log E[(eps rho)^{-1} phi((X_t-c)/(eps rho)) phi(Y_t)] exactly (Gaussian integral)."""
    s11, s12, s22 = ce_sigma_np(t)
    p11, p12, p22 = eps**2 * s11, eps**2 * s12, eps**2 * s22
    m1 = 1.0 / (eps * rho) ** 2
    # det(I + P M), M = diag(m1, 1)
    detIPM = (1 + p11 * m1) * (1 + p22) - p12 * p12 * m1
    a11 = p11 + (eps * rho) ** 2
    a22 = p22 + 1.0
    detA = a11 * a22 - p12 * p12
    d1 = np.asarray(t, float) - c
    quad = d1 * d1 * a22 / detA
    return -np.log(eps * rho) - np.log(2 * np.pi) - 0.5 * np.log(detIPM) - 0.5 * quad


def local_extrema(y):
    s = np.sign(np.diff(y))
    nz = s != 0
    s = s[nz]
    idx = np.nonzero(nz)[0]
    maxima, minima = [], []
    for k in range(len(s) - 1):
        if s[k] > 0 and s[k + 1] < 0:
            maxima.append(int(idx[k + 1]))
        if s[k] < 0 and s[k + 1] > 0:
            minima.append(int(idx[k + 1]))
    return maxima, minima


def free_envelope_counts():
    out = {}
    # CE1 on [1/200, 8]
    t = np.linspace(1 / 200, 8, 400001)
    ce1 = {}
    for eps in [1.0, 0.5, 0.3, 0.1, 0.05]:
        lg = ce_log_Eg(t, eps, 0.01, 0.01)
        mx, mn = local_extrema(lg)
        ce1[str(eps)] = {"maxima_t": [round(float(t[i]), 5) for i in mx],
                         "minima_t": [round(float(t[i]), 5) for i in mn],
                         "logEg_at_maxima": [round(float(lg[i]), 4) for i in mx],
                         "endpoint_slopes_sign": [int(np.sign(lg[1] - lg[0])), int(np.sign(lg[-1] - lg[-2]))]}
    out["CE1_free_envelope_window_[1/200,8]"] = ce1
    # free part of the relative estimate: eps^2 d/dt log E g(Z_t) vs -J'(t) at W's four times
    # (central differences, h = 1e-6; J' from the 60-digit formula)
    rel = {}
    for tt in [1 / 200, 1 / 20, 1 / 5, 8.0]:
        mJp = -float(ce_Jp(mp.mpf(tt), mp, mp.mpf(1) / 100, mp.mpf(1) / 100))
        row = {"minus_Jprime": round(mJp, 6)}
        for eps in [0.3, 0.1, 0.03]:
            hh = 1e-6
            der = (ce_log_Eg(tt + hh, eps, 0.01, 0.01) - ce_log_Eg(tt - hh, eps, 0.01, 0.01)) / (2 * hh)
            row[f"eps={eps}"] = round(float(eps**2 * der), 6)
        rel[str(tt)] = row
    out["CE1_free_part_eps2_dlogEg_vs_minusJprime"] = rel
    # CE2 on [0.005, 11], w = (1/2, 1/2) (weights not stated in W; any interior w gives the same count as eps->0)
    t2 = np.linspace(0.005, 11, 400001)
    ce2 = {}
    for eps in [0.3, 0.1]:
        lg = np.logaddexp(np.log(0.5) + ce_log_Eg(t2, eps, 0.01, 0.01), np.log(0.5) + ce_log_Eg(t2, eps, 10.0, 0.01))
        mx, mn = local_extrema(lg)
        ce2[str(eps)] = {"maxima_t": [round(float(t2[i]), 5) for i in mx],
                         "minima_t": [round(float(t2[i]), 5) for i in mn]}
    out["CE2_free_envelope_window_[0.005,11]_w=(1/2,1/2)"] = ce2
    # spiral on [pi/6, 5pi/2]
    t3 = np.linspace(math.pi / 6, 5 * math.pi / 2, 400001)
    spr = {}
    for eps in [0.1, 0.05]:
        mx_, my_ = 1 - 2 * np.exp(-t3 / 5) * np.cos(t3), -2 * np.exp(-t3 / 5) * np.sin(t3)
        lg = -mx_**2 / (4 * eps**2) - my_**2 / (2 * (1 + eps**2))
        mx, mn = local_extrema(lg)
        spr[str(eps)] = {"maxima_t": [round(float(t3[i]), 5) for i in mx],
                         "minima_t": [round(float(t3[i]), 5) for i in mn]}
    out["spiral_free_envelope_window_[pi/6,5pi/2]"] = spr
    out["note"] = ("B -> 0 limit of f/B (budget-normalised density); exact closed forms on 400001-point grids. "
                   "These are free envelopes, not the killed density. The counts are sign changes of "
                   "differences of adjacent grid samples (numerical grid counts), not certified continuous "
                   "counts of the critical points.")
    return out


class CEChain:
    """Exact time-discretised Gaussian chain of CE1 on a uniform grid of step dt."""

    def __init__(self, eps, dt, nmax):
        self.eps, self.dt, self.nmax = eps, dt, nmax
        om = -math.expm1(-dt)
        om2 = -math.expm1(-2 * dt)
        self.F = np.array([[1.0, om], [0.0, 1.0 - om]])
        self.d = np.array([dt, 0.0])
        g = dt - 2 * om + om2 / 2
        self.Q = eps**2 * np.array([[Q1F * dt + g, om * om / 2], [om * om / 2, om2 / 2]])
        k = np.arange(nmax + 1)
        s11, s12, s22 = ce_sigma_np(k * dt)
        self.P = eps**2 * np.stack([np.stack([s11, s12], -1), np.stack([s12, s22], -1)], -2)
        self.m = np.stack([k * dt, np.zeros(nmax + 1)], -1)
        Qi = np.linalg.inv(self.Q)
        H = self.F.T @ Qi @ self.F
        Pi = np.linalg.inv(self.P)
        C = np.linalg.inv(Pi + H[None])
        C = 0.5 * (C + np.swapaxes(C, -1, -2))
        self.G = C @ (self.F.T @ Qi)[None]
        self.o = np.einsum("kij,kj->ki", C, np.einsum("kij,kj->ki", Pi, self.m)) - np.einsum("kij,j->ki", self.G,
                                                                                               self.d)
        L11 = np.sqrt(C[:, 0, 0])
        L21 = C[:, 1, 0] / L11
        L22 = np.sqrt(np.maximum(C[:, 1, 1] - L21**2, 0.0))
        self.L = (L11, L21, L22)
        self.cholQ = np.linalg.cholesky(self.Q)


def V_ce(z, B, eps, c, rho):
    u = (z[:, 0] - c) / (eps * rho)
    return B / (eps * rho) * np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi) * np.exp(-0.5 * z[:, 1] ** 2) / math.sqrt(
        2 * math.pi)


def tilted_bridge_alpha(eps, B, dt, n_list, npaths, seed_tag, c=0.01, rho=0.01):
    """alpha(t_n) = E~[exp(-int_0^t V)] under the terminal-tilted path law, all n in n_list in one backward sweep."""
    nmax = max(n_list)
    ch = CEChain(eps, dt, nmax)
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, STREAM_TAG, seed_tag]))
    M = np.diag([1.0 / (eps * rho) ** 2, 1.0])
    ctr = np.array([c, 0.0])
    starts = {}
    for j, n in enumerate(n_list):
        Pn = ch.P[n]
        Pni = np.linalg.inv(Pn)
        Pt = np.linalg.inv(Pni + M)
        Pt = 0.5 * (Pt + Pt.T)
        mt = Pt @ (Pni @ ch.m[n] + M @ ctr)
        starts.setdefault(n, []).append((j, mt, np.linalg.cholesky(Pt)))
    Z = np.empty((0, 2))
    I = np.empty(0)
    grp = np.empty(0, dtype=int)
    for k in range(nmax, -1, -1):
        nprev = Z.shape[0]
        if k in starts:
            for j, mt, Lt in starts[k]:
                zn = mt[None, :] + rng.standard_normal((npaths, 2)) @ Lt.T
                Z = np.vstack([Z, zn])
                I = np.concatenate([I, np.zeros(npaths)])
                grp = np.concatenate([grp, np.full(npaths, j)])
        v = V_ce(Z, B, eps, c, rho) * dt
        if k == 0:
            v *= 0.5
        v[nprev:] *= 0.5  # terminal node of newly started bridges
        I += v
        if k > 0:
            km = k - 1
            G = ch.G[km]
            xi = rng.standard_normal((Z.shape[0], 2))
            L11, L21, L22 = ch.L[0][km], ch.L[1][km], ch.L[2][km]
            x_new = ch.o[km, 0] + G[0, 0] * Z[:, 0] + G[0, 1] * Z[:, 1] + L11 * xi[:, 0]
            y_new = ch.o[km, 1] + G[1, 0] * Z[:, 0] + G[1, 1] * Z[:, 1] + L21 * xi[:, 0] + L22 * xi[:, 1]
            Z = np.stack([x_new, y_new], -1)
    res = []
    for j, n in enumerate(n_list):
        om = np.exp(-I[grp == j])
        a = float(om.mean())
        se = float(om.std(ddof=1) / math.sqrt(om.size))
        res.append((a, se, float(om.min()), float(om.max())))
    return res


def forward_mc_f(eps, B, dt, n, npaths, seed_tag, chunk=50000, c=0.01, rho=0.01):
    ch = CEChain(eps, dt, 1)
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, STREAM_TAG, seed_tag]))
    vals = []
    done = 0
    while done < npaths:
        m = min(chunk, npaths - done)
        Z = eps * math.sqrt(S0F) * rng.standard_normal((m, 2))
        I = 0.5 * dt * V_ce(Z, B, eps, c, rho)
        for k in range(1, n + 1):
            xi = rng.standard_normal((m, 2)) @ ch.cholQ.T
            Z = Z @ ch.F.T + ch.d[None, :] + xi
            v = V_ce(Z, B, eps, c, rho) * dt
            I += v if k < n else 0.5 * v
        vals.append(V_ce(Z, B, eps, c, rho) * np.exp(-I))
        done += m
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size)), int(np.count_nonzero(v > 1e-3 * v.max()))


def part5_illustration():
    out = {}
    t0 = time.time()
    # (a) validation of the tilted-bridge estimator against forward MC, eps = 1, B = 1, dt = 1e-3
    eps, B, dt = 1.0, 1.0, 1e-3
    n_val = [12, 300, 2000]
    alphas = tilted_bridge_alpha(eps, B, dt, n_val, 20000, seed_tag=10)
    val = []
    for (a, se, amin, amax), n in zip(alphas, n_val):
        t = n * dt
        lEg = float(ce_log_Eg(t, eps, 0.01, 0.01))
        f_tb = B * math.exp(lEg) * a
        f_tb_se = B * math.exp(lEg) * se
        f_fw, f_fw_se, n_contrib = forward_mc_f(eps, B, dt, n, 200000, seed_tag=20 + n)
        z = (f_tb - f_fw) / math.sqrt(f_tb_se**2 + f_fw_se**2)
        val.append({"t": t, "f_tilted_bridge": f_tb, "se_tb": f_tb_se, "f_forward_mc": f_fw, "se_fw": f_fw_se,
                    "forward_paths_contributing": n_contrib, "z_score": z, "alpha": a})
    out["validation_eps1_B1_dt1e-3"] = val
    out["validation_ok_|z|<3"] = bool(all(abs(r["z_score"]) < 3 for r in val))
    out["validation_seconds"] = round(time.time() - t0, 1)
    # (b) killed density of CE1 at eps = 0.3, B = 1, dt = 2e-4
    t1 = time.time()
    eps, B, dt = 0.3, 1.0, 2e-4
    tgrid = [0.005, 0.006, 0.007, 0.008, 0.009, 0.0096, 0.01, 0.0104, 0.011, 0.012, 0.014, 0.016, 0.02, 0.03,
             0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 3.0,
             3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
    n_list = [int(round(t / dt)) for t in tgrid]
    npaths = 1500
    alphas = tilted_bridge_alpha(eps, B, dt, n_list, npaths, seed_tag=30)
    rows = []
    logf = []
    for t, n, (a, se, amin, amax) in zip(tgrid, n_list, alphas):
        tt = n * dt
        lEg = float(ce_log_Eg(tt, eps, 0.01, 0.01))
        lf = math.log(B) + lEg + math.log(a)
        logf.append(lf)
        rows.append({"t": round(tt, 6), "log_f": round(lf, 4), "log_Eg_exact": round(lEg, 4),
                     "alpha": round(a, 5), "se_log_alpha": round(se / a, 5),
                     "omega_min_max": [round(amin, 5), round(amax, 5)]})
    mx, mn = local_extrema(np.array(logf))
    se_max = max(r["se_log_alpha"] for r in rows)
    # smallest |log f difference| between neighbouring grid points that decides the pattern
    diffs = np.abs(np.diff(logf))
    out["killed_density_eps0.3_B1"] = {
        "dt": dt, "paths_per_time": npaths, "rows": rows,
        "grid_local_maxima_t": [rows[i]["t"] for i in mx],
        "grid_local_minima_t": [rows[i]["t"] for i in mn],
        "max_se_log_alpha": se_max,
        "min_neighbour_|dlogf|": float(diffs.min()),
        "log_f_main_peak_minus_secondary_peak": (round(logf[mx[0]] - logf[mx[1]], 3) if len(mx) >= 2 else None),
        "seconds": round(time.time() - t1, 1),
    }
    out["note"] = ("f(t) = B E[g(Z_t)] alpha(t); E[g] exact, alpha by backward sampling of the exact Gaussian bridge "
                   "of the time-discretised linear chain conditioned on the tilted terminal law (trapezoid rule for "
                   "int V). Discretisation: stripe width eps*rho = 0.003 resolved by ~15 steps per sd at unit speed. "
                   "Illustration at a fixed moderate eps, not a proof of the small-eps statement.")
    return out


def main():
    import sys
    skip_mc = "--skip-mc" in sys.argv
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"script": str(HERE.relative_to(REPORT)),
           "sources": {"W": {"path": str(SRC_W.relative_to(REPORT)), "sha256": sha256(SRC_W)},
                       "CX": {"path": str(SRC_CX.relative_to(REPORT)), "sha256": sha256(SRC_CX)}},
           "precision": {"mp.dps": mp.dps, "iv.dps": iv.dps},
           "seeds": f"SeedSequence([{SEED_BASE}, {STREAM_TAG}, tag])", "workers": 1}
    t = time.time()
    res["part1_CE1_covariance"] = part1_covariance()
    res["part2_CE1_Jprime_intervals"] = part2_Jprime()
    res["part3_CE2_J1_intervals"] = part3_two_stripe()
    res["part1to3_seconds"] = round(time.time() - t, 1)
    t = time.time()
    res["part4_spiral"] = part4_spiral()
    res["part4_seconds"] = round(time.time() - t, 1)
    t = time.time()
    res["part5_free_envelopes"] = free_envelope_counts()
    if not skip_mc:
        res["part5_killed_illustration"] = part5_illustration()
    res["part5_seconds"] = round(time.time() - t, 1)
    res["total_seconds"] = round(time.time() - t_start, 1)
    out = OUT_DIR / ("checks.json" if not skip_mc else "checks_quick.json")
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ["part1to3_seconds", "part4_seconds", "part5_seconds", "total_seconds"]}))
    print("wrote", out)


if __name__ == "__main__":
    main()
