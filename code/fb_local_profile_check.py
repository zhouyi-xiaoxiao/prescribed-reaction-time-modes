#!/usr/bin/env python3
"""TH-3 (Theorem 2(b)) deterministic checks of the local passage profile.

The fixed-budget local passage profile is

    F_lam(y) = (p_lam * phi_theta)(y),  p_lam(u) = lam phi(u) exp(-lam Phi(u)),
    theta^2 = D0 / (2 gamma rho^2),

and the normalized profile is hatF_lam = F_lam / lam (hatF_0 = phi_sqrt(1+theta^2)).
This script checks, by deterministic quadrature only (no random numbers, no seeds):

 (1) profile lemma: exactly one sign change of hatF', every critical point has
     hatF'' < 0, the curvature identity A8
        hatF''(y*) = theta^-2 int (u-u0) hatp'(u) phi_theta(y*-u) du
     (two independent evaluations), sign of the A8 integrand, mass identity,
     hatF'(0) < 0 (so y* < 0 for lam > 0), the log-concavity threshold of hatp;
 (2) large-lam asymptotics y*(lam) + a_lam -> 0, a_lam = -Phi^{-1}(1/lam);
 (3) limit peak shifts eps rho y*(lam_j)/v_j (exact profile) and the mean-field
     shifts eps S_* u0(lam_j)/v_j at every configuration stored in the Feynman-Kac
     scout analyses theory_scout/analysis_m{2,3}.json, against the stored
     exact-law (FK) peak shifts and heights;
 (4) significant-mode counts (z > 5) read from the same stored analyses.

Post-audit additions (theory fixer, 2026-09-23; audits GPT-6 TH3-F1/F9, Fable F-1/F-4):
 (5) smoothing-matched height/shift comparison: the scout's FK peak heights and
     times are those of the FK density smoothed with a Gaussian of sd
     h_steps*dt, h_steps = max(1, 0.15 eps S_*/(v_max dt)) (analyze_fk.py);
     the matching limit is hatp_lam * phi_theta_eff, theta_eff^2 = theta^2 +
     (v h_steps dt/(eps rho))^2.  Monte Carlo standard errors of the smoothed
     peak heights are recomputed from the stored FK ensembles
     (~/.local-build/prr_gap_fk_cache/fk_m{m}_e{eps}.npz, seeds 11-16 of the
     scout run; no new simulation) when those files are present;
 (6) gate constants at the anchor: sup over [0,T] of ||Cov Y_t|| (the event
     A_eps uses M_Y = sup_[0,T]|Y|), the leading logarithmic rate
     eta_*^2/(2 sup_[0,T]||Cov Y_t||), and the constant c0 = c_Y eta_*^2 of
     Lemma th3:coupling actually used in Lemma th3:gate;
 (7) provenance copies of two externally stored numbers quoted in the TeX
     (GPT-6 profile root at beta=1; the (m,eps,B)=(2,0.05,50) mode list).

Output: artifacts/data/exact_m_fixed_budget/TH3/local_profile_check.json
Runtime: a few seconds, one process.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import validate_exact_m_offlattice as base  # noqa: E402  (read-only: parameters)

REPORT = HERE.parents[1]
OUT_DIR = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "TH3"
SCOUT = REPORT / "notes" / "gap_diagnosis_20260923" / "theory_scout"
FK_CACHE = Path.home() / ".local-build" / "prr_gap_fk_cache"          # stored scout FK ensembles (read-only)
GPT6_PROFILE = Path.home() / ".local-build" / "codex_theory_prr_20260923" / "theory_checks" / "profile_results.json"

MODEL = base.MODEL
TARGET_TIMES = base.TARGET_TIMES
GAMMA, D0, RHO, W = MODEL.gamma, MODEL.d0, MODEL.rho, MODEL.torus_w
Z0, ZBAR, DIM = MODEL.z0, MODEL.z_bar, MODEL.dim
S_Z = math.sqrt(D0 / (2.0 * GAMMA))          # stationary midpoint sd / eps
THETA = S_Z / RHO                             # kernel width in slab units
S_STAR = math.sqrt(S_Z ** 2 + RHO ** 2)       # free-clock width / eps

SQ2PI = math.sqrt(2.0 * math.pi)
_erfc = np.frompyfunc(math.erfc, 1, 1)


def phi(u):
    return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2) / SQ2PI


def Phi(u):
    return 0.5 * np.asarray(_erfc(-np.asarray(u, dtype=float) / math.sqrt(2.0)), dtype=float)


def Phi_scalar(u: float) -> float:
    return 0.5 * math.erfc(-u / math.sqrt(2.0))


def Phi_inv(p: float) -> float:
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi_scalar(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def v_of_t(t: float) -> float:
    return GAMMA * abs(Z0 - ZBAR) * math.exp(-GAMMA * t)   # |mu'(t)|


# ---------------------------------------------------------------- quadrature
DU = 0.002
U = np.arange(-16.0, 16.0 + DU / 2, DU)
PHI_U = phi(U)
PHI_CDF_U = Phi(U)
DY = 0.01
Y = np.arange(-10.0, 7.0 + DY / 2, DY)


def kern_w(x, order, theta):
    """phi_theta^{(order)}(x) for the normalized Gaussian of sd theta."""
    g = np.exp(-0.5 * (x / theta) ** 2) / (SQ2PI * theta)
    if order == 0:
        return g
    if order == 1:
        return -x / theta ** 2 * g
    if order == 2:
        return (x * x / theta ** 4 - 1.0 / theta ** 2) * g
    raise ValueError(order)


def kern(x, order):
    """phi_theta^{(order)}(x) for the normalized Gaussian of sd THETA."""
    return kern_w(x, order, THETA)


def smoothed_peak(lam: float, theta_eff: float, y_guess: float) -> tuple[float, float]:
    """Unique critical point and maximum of hatp_lam * phi_theta_eff (Lemma th3:profile
    applies verbatim with theta replaced by theta_eff > 0)."""
    p = hatp(lam)

    def d1(y):
        return float(np.sum(p * kern_w(y - U, 1, theta_eff)) * DU)

    lo, hi = y_guess - 2.0, y_guess + 2.0
    assert d1(lo) > 0 > d1(hi), (lam, theta_eff)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if d1(mid) > 0:
            lo = mid
        else:
            hi = mid
    ys = 0.5 * (lo + hi)
    return ys, float(np.sum(p * kern_w(ys - U, 0, theta_eff)) * DU)


def smooth_like_scout(y, h_steps):
    """Verbatim copy of theory_scout/analyze_fk.py::smooth (discrete Gaussian, sd h_steps)."""
    if h_steps < 1:
        return y.copy()
    k = np.arange(-int(5 * h_steps), int(5 * h_steps) + 1)
    ker = np.exp(-0.5 * (k / h_steps) ** 2)
    ker /= ker.sum()
    return np.convolve(y, ker, mode="same")


def hatp(lam):
    return PHI_U * np.exp(-lam * PHI_CDF_U)


def hatp_prime(lam):
    return hatp(lam) * (-U - lam * PHI_U)


def conv_at(pvals, y, order):
    return float(np.sum(pvals * kern(y - U, order)) * DU)


def conv_grid(pvals, order):
    out = np.empty_like(Y)
    for lo in range(0, Y.size, 200):
        yy = Y[lo:lo + 200, None]
        out[lo:lo + 200] = np.einsum("ij,j->i", kern(yy - U[None, :], order), pvals) * DU
    return out


def u0_of(lam: float) -> float:
    """Unique zero of h(u) = u + lam phi(u) on (-inf, 0]."""
    if lam == 0.0:
        return 0.0
    lo, hi = -60.0, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid + lam * math.exp(-0.5 * mid * mid) / SQ2PI < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def profile_row(lam: float) -> dict:
    p = hatp(lam)
    F = conv_grid(p, 0)
    Fp = conv_grid(p, 1)
    Fmax = float(F.max())
    live = F >= 1e-12 * Fmax
    idx = np.flatnonzero(live)
    seg = Fp[idx[0]: idx[-1] + 1]
    sign_changes = int(np.count_nonzero(np.sign(seg[:-1]) * np.sign(seg[1:]) < 0))
    # bracket and bisect the root of hatF'
    k = int(np.argmax(F))
    lo, hi = Y[max(k - 5, 0)], Y[min(k + 5, Y.size - 1)]
    flo, fhi = conv_at(p, lo, 1), conv_at(p, hi, 1)
    assert flo > 0 > fhi, (lam, flo, fhi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if conv_at(p, mid, 1) > 0:
            lo = mid
        else:
            hi = mid
    ystar = 0.5 * (lo + hi)
    u0 = u0_of(lam)
    pp = hatp_prime(lam)
    a8_integrand = (U - u0) * pp
    curv_direct = conv_at(p, ystar, 2)
    curv_a8 = float(np.sum(a8_integrand * kern(ystar - U, 0)) * DU) / THETA ** 2
    peak = conv_at(p, ystar, 0)
    mass_hat = float(np.sum(p) * DU)
    mass_exact_hat = (-math.expm1(-lam) / lam) if lam > 0 else 1.0
    return dict(
        lam=lam,
        sign_changes_of_hatF_prime=sign_changes,
        y_star=ystar,
        u0=u0,
        hatF_at_ystar=peak,
        hatF2_at_ystar_direct=curv_direct,
        hatF2_at_ystar_A8=curv_a8,
        A8_relative_residual=abs(curv_direct - curv_a8) / abs(curv_direct),
        A8_integrand_max=float(a8_integrand.max()),
        curvature_over_peak=curv_direct / peak,
        hatF_prime_at_0=conv_at(p, 0.0, 1),
        mass_hatp=mass_hat,
        mass_hatp_exact=mass_exact_hat,
        mass_rel_error=abs(mass_hat - mass_exact_hat) / mass_exact_hat,
        hatp_log_concave=bool(lam * float(np.max(U * PHI_U)) <= 1.0),
    )


def main() -> None:
    t0 = time.monotonic()
    out: dict = dict(
        item="TH-3 local passage profile (Theorem 2(b))",
        script="code/fb_local_profile_check.py",
        deterministic=True,
        seeds=("no new random numbers: deterministic quadrature, plus read-only post-processing of the stored "
               "scout FK ensembles (their seeds 11-16 are recorded in fk_npz_used)"),
        anchor=dict(gamma=GAMMA, D0=D0, rho=RHO, W=W, z0=Z0, zbar=ZBAR, dim=DIM,
                    s_Z=S_Z, theta=THETA, theta_sq=THETA ** 2, S_star=S_STAR),
        quadrature=dict(du=DU, u_range=[float(U[0]), float(U[-1])], dy=DY,
                        y_range=[float(Y[0]), float(Y[-1])],
                        sign_change_count_region="hatF >= 1e-12 * max hatF"),
    )
    # (1) profile lemma on a log grid (plus lam = 0)
    lams = [0.0] + list(np.logspace(-3, 4, 29))
    rows = [profile_row(float(l)) for l in lams]
    lam_lc = 1.0 / float(phi(1.0))
    out["profile"] = dict(
        rows=rows,
        n_lambda=len(rows),
        lambda_range=[0.0, float(lams[-1])],
        all_single_sign_change=all(r["sign_changes_of_hatF_prime"] == 1 for r in rows),
        all_curvature_negative=all(r["hatF2_at_ystar_direct"] < 0 for r in rows),
        max_A8_relative_residual=max(r["A8_relative_residual"] for r in rows),
        max_A8_integrand=max(r["A8_integrand_max"] for r in rows),
        all_hatF_prime_at_0_negative_for_lam_pos=all(r["hatF_prime_at_0"] < 0 for r in rows if r["lam"] > 0),
        max_mass_rel_error=max(r["mass_rel_error"] for r in rows),
        curvature_over_peak_lam0=rows[0]["curvature_over_peak"],
        curvature_over_peak_lam0_exact=-1.0 / (1.0 + THETA ** 2),
        log_concavity_threshold_lambda=lam_lc,
        log_concavity_threshold_formula="1/phi(1) = sqrt(2 pi e)",
        n_log_concave=sum(r["hatp_log_concave"] for r in rows),
    )
    ratios = [S_STAR * r["u0"] / (RHO * r["y_star"]) for r in rows if r["lam"] > 0]
    out["profile"]["meanfield_to_exact_shift_ratio"] = dict(
        definition="S_* u0(lam) / (rho y*(lam)) over the lam>0 grid",
        min=min(ratios), max=max(ratios),
        small_lam_limit_exact=math.sqrt(1.0 + 2.0 * THETA ** 2),
        large_lam_limit_exact=math.sqrt(1.0 + THETA ** 2),
    )
    # (2) large-lambda asymptotics
    big = []
    for lam in (1e2, 1e3, 1e4, 1e6, 1e8):
        r = profile_row(lam)
        a = -Phi_inv(1.0 / lam)
        big.append(dict(lam=lam, y_star=r["y_star"], a_lam=a, y_star_plus_a=r["y_star"] + a,
                        sqrt_2_log_lam=math.sqrt(2 * math.log(lam))))
    out["large_lambda"] = big
    # (3) FK configurations
    fk_rows = []
    cache: dict = {}
    npz_used = []
    for m in (2, 3):
        path = SCOUT / f"analysis_m{m}.json"
        data = json.loads(path.read_text())
        times = TARGET_TIMES[m]
        for entry in data:
            eps = float(entry["eps"])
            h_steps = float(entry["h_steps"])
            dt_fk = float(entry["dt"])
            h_check = max(1.0, 0.15 * (eps * S_STAR / max(v_of_t(tj) for tj in times)) / dt_fk)
            assert abs(h_check - h_steps) < 1e-9, (m, eps, h_check, h_steps)
            sig_t = h_steps * dt_fk if h_steps >= 1 else 0.0     # smoothing sd in time units
            npz_path = FK_CACHE / f"fk_m{m}_e{eps:g}.npz"
            fkd = np.load(npz_path) if npz_path.exists() else None
            if fkd is not None:
                npz_used.append(dict(file=str(npz_path).replace(str(Path.home()), "~"), m=m, eps=eps,
                                     seed=int(fkd["seed"]), N=int(fkd["N"]), dt=float(fkd["dt"])))
                tg, bud, cks = fkd["tgrid"], fkd["budgets"], fkd["checkpoints"]
                FKf, FKbf = fkd["FK_free"], fkd["FKb_free"]
            for ps in entry["peak_shifts_free"]:
                B = float(ps["B"])
                lam_all = [B * (1.0 / m) / (W ** (DIM - 1) * v_of_t(tj)) for tj in times]
                if fkd is not None:
                    bi = int(np.flatnonzero(np.isclose(bud, B))[0])
                    f_s = smooth_like_scout(FKf[bi] / dt_fk, h_steps)
                    fb_s = np.array([smooth_like_scout(FKbf[b, bi] / dt_fk, h_steps) for b in range(FKbf.shape[0])])
                for pk in ps["peaks"]:
                    j = int(pk["j"])
                    lam = lam_all[j]
                    key = round(lam, 12)
                    if key not in cache:
                        cache[key] = profile_row(lam)
                    r = cache[key]
                    v = v_of_t(times[j])
                    pij = math.exp(-sum(lam_all[:j]))
                    shift_prof = eps * RHO * r["y_star"] / v
                    shift_mf = eps * S_STAR * r["u0"] / v
                    height_prof = pij * (v / RHO) * lam * r["hatF_at_ystar"] / eps
                    u0 = r["u0"]
                    height_mf = pij * (v / S_STAR) * lam * float(phi(u0)) * math.exp(-lam * Phi_scalar(u0)) / eps
                    # (5) smoothing-matched limit: same Gaussian smoothing as the scout's FK analysis
                    theta_eff = math.sqrt(THETA ** 2 + (v * sig_t / (eps * RHO)) ** 2)
                    ys_eff, F_eff = smoothed_peak(lam, theta_eff, r["y_star"])
                    shift_prof_s = eps * RHO * ys_eff / v
                    height_prof_s = pij * (v / RHO) * lam * F_eff / eps
                    mc = {}
                    if fkd is not None:
                        lo_c, hi_c = cks[j], cks[j + 1]
                        idx = np.flatnonzero((tg > lo_c) & (tg < hi_c))
                        kk = idx[int(np.argmax(f_s[idx]))]
                        se = float(np.std(fb_s[:, kk], ddof=1) / math.sqrt(fb_s.shape[0]))
                        mc = dict(height_FK_recomputed=float(f_s[kk]), t_peak_FK_recomputed=float(tg[kk]),
                                  height_FK_se=se, height_FK_rel_se=se / float(f_s[kk]),
                                  n_batches=int(fb_s.shape[0]))
                    fk_rows.append(dict(
                        smoothing_sd_time=sig_t, theta_eff=theta_eff,
                        y_star_smoothed=ys_eff,
                        shift_limit_profile_smoothed=shift_prof_s,
                        err_profile_smoothed=float(pk["shift_exact"]) - shift_prof_s,
                        height_limit_profile_smoothed=height_prof_s,
                        height_rel_dev_FK_vs_smoothed_profile=float(pk["peak_height_exact"]) / height_prof_s - 1.0,
                        **mc,
                        source=f"notes/gap_diagnosis_20260923/theory_scout/analysis_m{m}.json#[eps={eps}].peak_shifts_free[B={B}].peaks[j={j}]",
                        kernel="free (contact gate removed)",
                        m=m, eps=eps, B=B, j=j, t_target=times[j],
                        lam=lam, lam_stored=float(pk["lam"]), v=v, Pi=pij,
                        y_star=r["y_star"], u0=u0,
                        shift_FK_exact_law=float(pk["shift_exact"]),
                        shift_limit_stored_grid=float(pk["shift_limit"]),
                        shift_limit_profile=shift_prof,
                        shift_limit_meanfield=shift_mf,
                        err_profile=float(pk["shift_exact"]) - shift_prof,
                        err_meanfield=float(pk["shift_exact"]) - shift_mf,
                        height_FK_exact_law=float(pk["peak_height_exact"]),
                        height_limit_stored=float(pk["peak_height_limit"]),
                        height_limit_profile=height_prof,
                        height_limit_meanfield=height_mf,
                        height_rel_dev_FK_vs_profile=float(pk["peak_height_exact"]) / height_prof - 1.0,
                    ))
    out["fk_peaks"] = fk_rows
    out["fk_npz_used"] = npz_used
    # (5) height comparison with and without the scout's smoothing (Fable audit F-1)
    first = [r for r in fk_rows if r["j"] == 0]
    late = [r for r in fk_rows if r["j"] == r["m"] - 1]

    def hsum(sel):
        d = dict(n=len(sel),
                 max_abs_rel_dev_vs_profile=max(abs(r["height_rel_dev_FK_vs_profile"]) for r in sel),
                 max_abs_rel_dev_vs_smoothed_profile=max(abs(r["height_rel_dev_FK_vs_smoothed_profile"]) for r in sel),
                 min_rel_dev_vs_smoothed_profile=min(r["height_rel_dev_FK_vs_smoothed_profile"] for r in sel),
                 max_rel_dev_vs_smoothed_profile=max(r["height_rel_dev_FK_vs_smoothed_profile"] for r in sel),
                 rel_dev_vs_profile=[r["height_rel_dev_FK_vs_profile"] for r in sel],
                 rel_dev_vs_smoothed_profile=[r["height_rel_dev_FK_vs_smoothed_profile"] for r in sel],
                 cells=[(r["m"], r["eps"], r["B"], r["j"]) for r in sel])
        if all("height_FK_rel_se" in r for r in sel):
            d["max_rel_se"] = max(r["height_FK_rel_se"] for r in sel)
            d["max_abs_z_vs_smoothed_profile"] = max(
                abs(r["height_FK_exact_law"] - r["height_limit_profile_smoothed"]) / r["height_FK_se"] for r in sel)
            d["max_abs_recomputed_minus_stored_height"] = max(
                abs(r["height_FK_recomputed"] - r["height_FK_exact_law"]) for r in sel)
        return d
    out["fk_height_summary"] = dict(
        smoothing_rule=("scout analyze_fk.py: FK density smoothed by a discrete Gaussian of sd h_steps*dt, "
                        "h_steps = max(1, 0.15 eps S_*/(v_max dt)) (0.15 x narrowest bump sd); matching limit = "
                        "hatp_lam * phi_theta_eff with theta_eff^2 = theta^2 + (v h_steps dt/(eps rho))^2"),
        gaussian_bump_bias_fastest_peak=(1.0 + 0.15 ** 2) ** -0.5 - 1.0,
        first_peaks_eps_0p035=hsum([r for r in first if abs(r["eps"] - 0.035) < 1e-12]),
        first_peaks_all_eps=hsum(first),
        last_peaks_eps_0p035=hsum([r for r in late if abs(r["eps"] - 0.035) < 1e-12]),
    )
    series = []
    for key in sorted({(r["m"], r["j"], r["B"]) for r in fk_rows}):
        sel = sorted([r for r in fk_rows if (r["m"], r["j"], r["B"]) == key], key=lambda r: -r["eps"])
        dev = [r["height_rel_dev_FK_vs_smoothed_profile"] for r in sel]
        series.append(dict(m=key[0], j=key[1], B=key[2], lam=sel[0]["lam"], v=sel[0]["v"],
                           eps=[r["eps"] for r in sel], rel_dev_vs_smoothed_profile=dev,
                           rel_se=[r.get("height_FK_rel_se") for r in sel],
                           positive=all(x > 0 for x in dev),
                           decreasing_as_eps_decreases=all(a > b for a, b in zip(dev, dev[1:]))))
    out["fk_height_convergence"] = dict(
        definition="FK smoothed peak height / smoothing-matched limit height - 1, per (m, j, B) at eps = 0.1, 0.05, 0.035",
        series=series,
        n_series=len(series),
        all_positive=all(x["positive"] for x in series),
        all_decreasing=all(x["decreasing_as_eps_decreases"] for x in series),
        max_rel_se=max(max(v for v in x["rel_se"] if v is not None) for x in series) if all(
            all(v is not None for v in x["rel_se"]) for x in series) else None,
    )

    def summ(rows_, eps_max):
        sel = [r for r in rows_ if r["eps"] <= eps_max + 1e-12]
        return dict(
            eps_max=eps_max, n=len(sel),
            max_abs_err_profile=max(abs(r["err_profile"]) for r in sel),
            max_abs_err_profile_smoothed=max(abs(r["err_profile_smoothed"]) for r in sel),
            max_abs_err_meanfield=max(abs(r["err_meanfield"]) for r in sel),
            mean_abs_err_profile=float(np.mean([abs(r["err_profile"]) for r in sel])),
            mean_abs_err_profile_smoothed=float(np.mean([abs(r["err_profile_smoothed"]) for r in sel])),
            max_abs_smoothing_shift_of_limit=max(abs(r["shift_limit_profile_smoothed"] - r["shift_limit_profile"]) for r in sel),
            mean_abs_err_meanfield=float(np.mean([abs(r["err_meanfield"]) for r in sel])),
            n_profile_closer=int(sum(abs(r["err_profile"]) < abs(r["err_meanfield"]) for r in sel)),
            max_abs_lam_mismatch=max(abs(r["lam"] - r["lam_stored"]) for r in sel),
            max_abs_profile_vs_stored_grid=max(abs(r["shift_limit_profile"] - r["shift_limit_stored_grid"]) for r in sel),
        )
    out["fk_peaks_summary"] = dict(all_eps=summ(fk_rows, 1.0), eps_le_0p05=summ(fk_rows, 0.05),
                                   eps_le_0p035=summ(fk_rows, 0.035))
    # stored scout limits that sat on the lower end (tau = -6) of the scout's tau grid
    out["fk_peaks_stored_limit_clipped"] = [
        dict(m=r["m"], eps=r["eps"], B=r["B"], j=r["j"], shift_limit_stored_grid=r["shift_limit_stored_grid"],
             shift_limit_profile=r["shift_limit_profile"])
        for r in fk_rows if abs(r["shift_limit_stored_grid"] / r["eps"] + 6.0) < 1e-9]
    # convergence of the slow late passages: error in units of eps (o(eps) claim) and height deviation
    conv = []
    for (m_, j_, B_) in ((2, 1, 1.0), (2, 1, 8.0), (3, 2, 1.0), (3, 2, 8.0)):
        sel = sorted([r for r in fk_rows if r["m"] == m_ and r["j"] == j_ and r["B"] == B_], key=lambda r: -r["eps"])
        conv.append(dict(m=m_, j=j_, B=B_, lam=sel[0]["lam"], v=sel[0]["v"],
                         eps=[r["eps"] for r in sel],
                         err_profile_over_eps=[r["err_profile"] / r["eps"] for r in sel],
                         err_meanfield_over_eps=[r["err_meanfield"] / r["eps"] for r in sel],
                         height_rel_dev=[r["height_rel_dev_FK_vs_profile"] for r in sel],
                         height_rel_dev_vs_smoothed=[r["height_rel_dev_FK_vs_smoothed_profile"] for r in sel],
                         height_rel_se=[r.get("height_FK_rel_se") for r in sel],
                         err_profile_smoothed_over_eps=[r["err_profile_smoothed"] / r["eps"] for r in sel]))
    out["fk_slow_passage_convergence"] = conv
    # (4) significant-mode counts from the stored FK analyses
    counts = []
    for m in (2, 3):
        data = json.loads((SCOUT / f"analysis_m{m}.json").read_text())
        for entry in data:
            for kern_name in ("free", "full"):
                for Bs, lst in entry["modes"][kern_name].items():
                    zs = [md["z"] for md in lst]
                    counts.append(dict(m=m, eps=float(entry["eps"]), kernel=kern_name, B=float(Bs),
                                       n_local_max=len(lst), n_significant_z_gt_5=int(sum(z > 5 for z in zs)),
                                       significant_times=[md["t"] for md in lst if md["z"] > 5]))
    out["fk_mode_counts"] = counts
    out["fk_mode_counts_summary"] = dict(
        rule="count of local maxima of the smoothed FK density with z > 5 (scout analyze_fk.py count_modes)",
        cells_with_exactly_m=[(c["m"], c["eps"], c["kernel"], c["B"]) for c in counts if c["n_significant_z_gt_5"] == c["m"]],
        cells_not_m=[(c["m"], c["eps"], c["kernel"], c["B"], c["n_significant_z_gt_5"]) for c in counts
                     if c["n_significant_z_gt_5"] != c["m"]],
        n_cells_B_le_8=sum(1 for c in counts if c["B"] <= 8.0),
        n_cells_B_le_8_exactly_m=sum(1 for c in counts if c["B"] <= 8.0 and c["n_significant_z_gt_5"] == c["m"]),
        n_cells_B_le_8_at_least_m=sum(1 for c in counts if c["B"] <= 8.0 and c["n_significant_z_gt_5"] >= c["m"]),
        budgets_B_le_8=sorted({c["B"] for c in counts if c["B"] <= 8.0}),
    )
    # contact / exposure diagnostics copied (with source) from the stored analyses
    diag = []
    for m in (2, 3):
        data = json.loads((SCOUT / f"analysis_m{m}.json").read_text())
        for entry in data:
            diag.append(dict(m=m, eps=float(entry["eps"]),
                             contact_prob_at_tj=entry["contact_prob_at_tj"],
                             exposure_mean_ratio_free=entry["exposure_ratio_to_limit"]["free"]["mean_ratio"],
                             exposure_std_ratio_free=entry["exposure_ratio_to_limit"]["free"]["std_ratio"],
                             pred_rel_std_free=entry["pred_rel_std_free"]))
    out["fk_contact_exposure_diagnostics"] = diag
    # gate transfer diagnostics: significant peak times, gated ('full') vs gate-free ('free') FK kernels
    gate_rows = []
    for m in (2, 3):
        data = json.loads((SCOUT / f"analysis_m{m}.json").read_text())
        for entry in data:
            for Bs in ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0"):
                tf = [md["t"] for md in entry["modes"]["free"][Bs] if md["z"] > 5]
                tg = [md["t"] for md in entry["modes"]["full"][Bs] if md["z"] > 5]
                same_n = len(tf) == len(tg)
                gate_rows.append(dict(m=m, eps=float(entry["eps"]), B=float(Bs), t_free=tf, t_full=tg,
                                      max_abs_dt=(max(abs(a - b) for a, b in zip(tf, tg)) if same_n and tf else None)))
        for entry in data:
            for bm in entry["basin_masses"]:
                if bm["B"] in (1, 8):
                    gate_rows.append(dict(m=m, eps=float(entry["eps"]), B=float(bm["B"]), masses_free=bm["exact_free"],
                                          masses_full=bm["exact_full"], limit=bm["limit_c1"]))
    out["fk_gate_diagnostics"] = gate_rows
    out["fk_gate_diagnostics_summary"] = dict(
        max_abs_dt_by_eps={str(e): max([r["max_abs_dt"] for r in gate_rows if r.get("max_abs_dt") is not None
                                        and r["eps"] == e] or [float("nan")]) for e in (0.1, 0.05, 0.035)},
        n_cells_B_le_8_same_count=sum(1 for r in gate_rows if "t_free" in r and len(r["t_free"]) == len(r["t_full"])),
        n_cells_B_le_8=sum(1 for r in gate_rows if "t_free" in r),
        max_abs_basin_mass_diff_full_vs_free_by_eps={
            str(e): max(abs(a - b) for r in gate_rows if "masses_free" in r and r["eps"] == e
                        for a, b in zip(r["masses_free"], r["masses_full"])) for e in (0.1, 0.05, 0.035)},
        basin_mass_diff_note="B in {1, 8}, m in {2, 3}; exact basin masses of the stored FK ensembles, gated ('full') vs gate-free ('free')",
    )
    # gate-transfer constants at the anchor (remark-level)
    r_par0, u0v, sp0 = MODEL.r_par0, MODEL.u0, MODEL.sigma_perp0
    tau_w, T_w = base.WINDOW
    rstar_max = max(abs(r_par0) * math.exp(-GAMMA * tau_w), abs(r_par0) * math.exp(-GAMMA * T_w))
    eta_star = MODEL.contact_a - math.hypot(rstar_max, MODEL.r_perp0)
    # ||Cov Y_t|| = max(Var Y_par(t), Var Y_perp,i(t)) (independent coordinates, Sigma_perp0 = sp0^2 I);
    # both variances are monotone in t, so the suprema are attained at the ends of the interval.
    def var_par(t):
        return u0v ** 2 * math.exp(-2 * GAMMA * t) + (2 * D0 / GAMMA) * (1 - math.exp(-2 * GAMMA * t))

    def var_perp(t):
        return sp0 ** 2 + 4.0 * D0 * t

    def sup_cov(a_, b_):
        return max(var_par(a_), var_par(b_), var_perp(a_), var_perp(b_))
    sup_0T, sup_I = sup_cov(0.0, T_w), sup_cov(tau_w, T_w)
    C_X = GAMMA * (Z0 - ZBAR) ** 2 / D0 + r_par0 ** 2 / (2.0 * (4.0 * D0 / GAMMA - u0v ** 2))
    c0_lead = eta_star ** 2 / (2.0 * sup_0T)
    sigma0_sq = max(u0v ** 2, sp0 ** 2)
    c_Y = min(1.0 / (8.0 * DIM * sigma0_sq), 1.0 / (32.0 * DIM * D0 * T_w * math.exp(2 * GAMMA * T_w)))
    c0_lemma = c_Y * eta_star ** 2
    out["gate_transfer_anchor"] = dict(
        window=[tau_w, T_w], eta_star=eta_star,
        max_var_Y_on_window=sup_I, sup_cov_Y_on_0T=sup_0T, sup_cov_Y_on_I=sup_I,
        cov_note="||Cov Y_t|| = largest eigenvalue; A_eps uses M_Y = sup_[0,T]|Y|, so the relevant supremum is over [0,T] "
                 "(equal to the window value at the anchor because both variances increase)",
        C_X=C_X, C_X_formula="gamma (z0-zbar)^2/D0 + r_par0^2/(2(4D0/gamma-u0^2))",
        c0_leading_order=c0_lead,
        c0_formula="eta_*^2/(2 sup_[0,T]||Cov Y_t||): leading logarithmic rate of P(A_eps^c) (Borell-TIS upper bound, "
                   "one-point Gaussian lower bound); any strictly smaller coefficient gives a usable bound",
        interpolation_order_lower_bound=2.0 * (c0_lead + C_X) / c0_lead,
        interpolation_order_lower_bound_note="n > 2(c0+C_X)/c0 is necessary for the k=2 exponent to be negative",
        lemma_constants=dict(C_Y=4 * DIM, sigma0_sq=sigma0_sq, c_Y=c_Y, c0_lemma=c0_lemma,
                             formula="c_Y = min{(8 d sigma0^2)^-1, (32 d D0 T e^{2 gamma T})^-1}, c0 = c_Y eta_*^2",
                             interpolation_order_used_in_proof=math.ceil(4.0 * (c0_lemma + C_X) / c0_lemma),
                             interpolation_order_used_in_proof_formula="n = ceil(4(c0+C_X)/c0) (Lemma th3:gate, Step 4)"),
    )
    # (7) provenance copies of numbers quoted in the TeX from files outside this JSON
    prov = {}
    if GPT6_PROFILE.exists():
        g6 = json.loads(GPT6_PROFILE.read_text())
        rowb = [r for r in g6["results"] if abs(r["beta"] - 1.0) < 1e-12][0]
        prov["gpt6_profile_root_beta1"] = dict(
            source=str(GPT6_PROFILE).replace(str(Path.home()), "~") + "#results[beta=1]",
            beta=rowb["beta"], root=rowb["root"], derivative_sign_changes=rowb["derivative_sign_changes"],
            curvature_over_peak=rowb["curvature_over_peak"], lambda_noise=g6["lambda_noise"],
            beta_range=[g6["results"][0]["beta"], g6["results"][-1]["beta"]], beta_count=g6["beta_count"],
            y_grid=g6["y_grid"], du=g6["du"])
    d2 = [e for e in json.loads((SCOUT / "analysis_m2.json").read_text()) if abs(float(e["eps"]) - 0.05) < 1e-12][0]
    prov["fk_modes_m2_eps0p05_B50"] = dict(
        source="notes/gap_diagnosis_20260923/theory_scout/analysis_m2.json#[eps=0.05].modes.{free,full}['50.0']",
        free_significant=[dict(t=md["t"], rel_prom=md["rel_prom"], z=md["z"]) for md in d2["modes"]["free"]["50.0"] if md["z"] > 5],
        full_significant=[dict(t=md["t"], rel_prom=md["rel_prom"], z=md["z"]) for md in d2["modes"]["full"]["50.0"] if md["z"] > 5],
        full_max_z_below_5=max(md["z"] for md in d2["modes"]["full"]["50.0"] if md["z"] <= 5),
    )
    out["provenance_copies"] = prov
    out["elapsed_seconds"] = time.monotonic() - t0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "local_profile_check.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    P = out["profile"]
    print("profile: n_lambda", P["n_lambda"], "single_sign_change", P["all_single_sign_change"],
          "curv_neg", P["all_curvature_negative"], "max_A8_rel_res %.3e" % P["max_A8_relative_residual"],
          "max_A8_integrand %.3e" % P["max_A8_integrand"], "Fp0_neg", P["all_hatF_prime_at_0_negative_for_lam_pos"],
          "max_mass_rel_err %.3e" % P["max_mass_rel_error"])
    for r in rows[::4]:
        print("  lam=%.4g y*=%.6f u0=%.6f curv/peak=%.6f" % (r["lam"], r["y_star"], r["u0"], r["curvature_over_peak"]))
    for b in big:
        print("  big lam=%.0e y*=%.4f a=%.4f y*+a=%.4f" % (b["lam"], b["y_star"], b["a_lam"], b["y_star_plus_a"]))
    for k, s in out["fk_peaks_summary"].items():
        print(k, json.dumps(s))
    print("gate", json.dumps(out["gate_transfer_anchor"]))
    print("modes not m:", out["fk_mode_counts_summary"]["cells_not_m"])
    s = out["fk_mode_counts_summary"]
    print("B<=8 cells:", s["n_cells_B_le_8"], "exactly m:", s["n_cells_B_le_8_exactly_m"], "budgets", s["budgets_B_le_8"])
    print("clipped stored limits:", out["fk_peaks_stored_limit_clipped"])
    for c in out["fk_slow_passage_convergence"]:
        print("slow", json.dumps(c))
    hs = out["fk_height_summary"]
    for k in ("first_peaks_eps_0p035", "first_peaks_all_eps", "last_peaks_eps_0p035"):
        print("heights", k, json.dumps({kk: vv for kk, vv in hs[k].items() if not isinstance(vv, list)}))
    print("npz used:", len(out["fk_npz_used"]))
    hc = out["fk_height_convergence"]
    print("height convergence: n", hc["n_series"], "all_positive", hc["all_positive"], "all_decreasing", hc["all_decreasing"],
          "max_rel_se", hc["max_rel_se"])
    print("provenance:", json.dumps(out["provenance_copies"])[:600])
    print("elapsed %.2f s -> %s" % (out["elapsed_seconds"], path))


if __name__ == "__main__":
    main()
