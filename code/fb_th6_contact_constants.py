#!/usr/bin/env python3
"""Constants of Proposition th6:prop-contact (TH6_exact_count.tex) at the paper's anchors.

Deterministic, no random numbers.  Evaluates, for the m=2 and m=3 anchors
(gamma = D0 = rho = W = 1, d = 2, Q = |z0 - zbar| = 4, a = 0.4, r_par0 = 0.1,
u0^2 = (Sigma_perp0)_ii = 0.09, window I = [0.5, 3.5]):

  eta_*  = a - sup_I |r_*(t)|,  eta = eta_*/2 (margin kept on [t0, T])
  s0^2   = max{u0^2, max_i (Sigma_perp0)_ii}
  q_T    = (exp(2 gamma T) - 1)/(2 gamma)
  c_R    = min{eta^2/(8 d s0^2), eta^2/(32 d D0 q_T)}           (Step 1)
  C_X    = Q^2/(2 sZ^2) + r_par0^2/(2(2 a_R - u0^2)), a_R = 2 D0/gamma (Step 3)
  J_*    = max_{l, t in I} delta_l(t)^2/(2 S_*^2)                (Step 5)
  n      = max{3, ceil(16 (J_* + C_X + c_R)/c_R)}                (Step 5)

These reproduce the Fable audit's independent evaluation (finding F-4) and
pin the numbers quoted in Remark th6:rem-contact-existence to a repo artifact.

Output: artifacts/data/exact_m_fixed_budget/TH6/th6_contact_constants_anchor.json
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "TH6" / "th6_contact_constants_anchor.json"


def anchor_constants(targets):
    gamma = D0 = rho = 1.0
    d = 2
    Q = 4.0
    a = 0.4
    r_par0 = 0.1
    u0_sq = 0.09
    sigma_perp_ii = 0.09
    tau, T = 0.5, 3.5
    sZ_sq = D0 / (2 * gamma)
    S_sq = sZ_sq + rho ** 2
    a_R = 2 * D0 / gamma
    # contact margin on the window: |r_*(t)| = r_par0 e^{-gamma t} is largest at t = tau
    eta_star = a - r_par0 * math.exp(-gamma * tau)
    eta = eta_star / 2
    s0_sq = max(u0_sq, sigma_perp_ii)
    q_T = (math.exp(2 * gamma * T) - 1) / (2 * gamma)
    c_R_a = eta ** 2 / (8 * d * s0_sq)
    c_R_b = eta ** 2 / (32 * d * D0 * q_T)
    c_R = min(c_R_a, c_R_b)
    C_X = Q ** 2 / (2 * sZ_sq) + r_par0 ** 2 / (2 * (2 * a_R - u0_sq))
    centres = [Q * math.exp(-gamma * t) for t in targets]
    # delta_l(t) = q(t) - c_l is monotone in t, so the max over I is at an endpoint
    q_ends = [Q * math.exp(-gamma * tau), Q * math.exp(-gamma * T)]
    J_star = max((qe - c) ** 2 / (2 * S_sq) for qe in q_ends for c in centres)
    n = max(3, math.ceil(16 * (J_star + C_X + c_R) / c_R))
    return {
        "target_times": targets,
        "centres": centres,
        "parameters": {"gamma": gamma, "D0": D0, "rho": rho, "d": d, "Q": Q, "a": a,
                        "r_par0": r_par0, "u0_sq": u0_sq, "Sigma_perp0_ii": sigma_perp_ii,
                        "tau": tau, "T": T},
        "eta_star": eta_star,
        "eta": eta,
        "s0_sq": s0_sq,
        "q_T": q_T,
        "c_R_initial_term": c_R_a,
        "c_R_brownian_term": c_R_b,
        "c_R": c_R,
        "C_X": C_X,
        "J_star": J_star,
        "n_interpolation_order": n,
        "log10_n": math.log10(n),
    }


def main():
    out = {
        "script": "code/fb_th6_contact_constants.py",
        "purpose": "constants of Proposition th6:prop-contact at the anchors (Remark th6:rem-contact-existence)",
        "random_numbers": "none (deterministic)",
        "anchors": {
            "m2": anchor_constants([1.0, 2.5]),
            "m3": anchor_constants([0.8, 1.6, 2.8]),
        },
        "comparison": {
            "TH3_leading_order_gate_rate_c0": "local_profile_check.json#gate_transfer_anchor.c0_leading_order (4.086e-3; gives n > 7.8e3)",
            "note": "The 1e4 order quoted before the gate edit used TH-3's leading-order rate; the proposition's own c_R is ~5000x smaller.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out["anchors"].items():
        print(k, "c_R=%.4e n=%d J*=%.4f C_X=%.5f eta*=%.5f" % (v["c_R"], v["n_interpolation_order"], v["J_star"], v["C_X"], v["eta_star"]))


if __name__ == "__main__":
    main()
