"""P5 fixer check (2026-09-23): numbers quoted in TH7b Prop p5:switch, Step 6, and the audit counterexample (GPT-6 F5).

(a) F5 counterexample: with the old truncated profile Theta_nu = int_{-L}^inf p(u) 1{...} du (mass a_L < 1) the bracket
    psi(lam Theta_nu) - psi(lam 1{...}) tends to psi(lam a_L) - psi(lam) != 0 as nu -> -inf (psi = e^{-y}).
(b) Tail of the two-sided comparison in Step 6:  sigma * E_xi int_{|u|>L} p(u) sqrt(2|u|/pi) du, with
    p(u) = (v/rho) phi((v u + xi)/rho), xi ~ N(0, s2), i.e. u ~ N(0, (rho^2 + s2)/v^2).  Deterministic quadrature
    (mpmath, 30 digits); no randomness.  Output: artifacts/data/exact_m_fixed_budget/P5_gate_switching/p5_step6_tail.json
Usage: python fb_p5_step6_tail.py
"""
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 30
OUT = Path(__file__).resolve().parents[1] / "artifacts/data/exact_m_fixed_budget/P5_gate_switching/p5_step6_tail.json"


def bracket_limit(v, rho, lam, xi, L):
    aL = mp.ncdf((v * L - xi) / rho)
    return aL, mp.e ** (-lam * aL) - mp.e ** (-lam)


def tail(L, v=1, rho=1, s2=1, sigma=1):
    sd = mp.sqrt(rho ** 2 + s2) / v
    f = lambda u: mp.npdf(u, 0, sd) * mp.sqrt(2 * abs(u) / mp.pi)
    return sigma * 2 * mp.quad(f, [L, L + 20 * sd, mp.inf])


def main():
    res = {"note": __doc__.strip().splitlines()[0], "F5_counterexample": [], "step6_tail": []}
    for v, rho, lam, xi, L in [(1, 1, 1, 0, 2), (0.3283, 1, 1, 0, 2)]:
        aL, lim = bracket_limit(v, rho, lam, xi, L)
        res["F5_counterexample"].append({"v": v, "rho": rho, "lam": lam, "xi": xi, "L": L,
                                         "a_L": mp.nstr(aL, 15), "bracket_limit": mp.nstr(lim, 15)})
    for L in [2, 3, 4, 6, 8]:
        t = tail(L)
        res["step6_tail"].append({"L": L, "v": 1, "rho": 1, "s2": 1, "sigma": 1, "tail": mp.nstr(t, 6),
                                  "ratio_to_exp_-L2_over_8": mp.nstr(t / mp.e ** (-mp.mpf(L) ** 2 / 8), 4)})
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
