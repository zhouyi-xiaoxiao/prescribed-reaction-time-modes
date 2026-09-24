#!/usr/bin/env python3
"""Classifier-free mean-field topology threshold B_top^mf(eps).

For the parameter-free mean-field hazard--survival law built from the free
exposure clock G of the campaign,

    f_1(t; B) = B G(t) exp(-B Lambda(t)),   Lambda(t) = int_0^t G(s) ds,

this script counts the INTERIOR nondegenerate maxima of f_1 on the declared
window I = [tau, T] as a function of the budget B and defines

    B_top^mf(eps) := sup{ B > 0 : f_1(.; B) has exactly m interior maxima on I }.

No classifier, bandwidth, walker count, or prominence floor enters; the only
inputs are the model constants, the slab design (m, centres, weights), eps,
and the window.  It is a threshold of the mean-field law, not of the exact
Doi process, and is reported as such (no theorem is claimed).

Exact derivative factorization.  Because B exp(-B Lambda) > 0,

    sign f_1'(t; B) = sign psi_B(t),   psi_B(t) = G'(t) - B G(t)^2,

so an interior maximum of f_1 is a + -> - sign change of psi_B inside (tau, T)
and the count is evaluated from psi_B directly (no underflow of exp(-B Lambda)
can affect it).  Equivalently, where G > 0, a maximum is a downward crossing of
the level B by the exposure-hazard ratio r(t) = G'(t) / G(t)^2; the local
maxima of r on the rising flanks of G are therefore the exact budget levels at
which f_1 loses a maximum.  This level-set characterization is computed
independently and compared with the grid-and-bisection value on every
configuration (cross-check, stored).

Numerical conventions (all stored under ``conventions`` in the output):

* time grid t_k = k dt on [0, tmax] with dt = 2e-5 (canonical; window edges
  and the origin lie on the grid), so Lambda carries its lower limit t = 0;
  grid checks repeat every threshold at dt = 4e-5 and 1e-5;
* G(t) = core.free_exposure_general (contact factor by the campaign's folded
  wrapped-normal quadrature: 400 transverse points and 10 images for d = 2,
  160 x 160 points for d = 3);
* G'(t) is semi-analytic: G = P c(t) H(x(t)) with H the Gaussian mixture, so
  G' = P [c' H + c H'(x) x'] with H' and x' in closed form and c' by
  second-order central differences of the quadrature (c varies on the O(1)
  time scale); a fully finite-difference G' is run as a further check;
* sign changes use the scale-free ratio psi~ = psi_B / (|G'| + B G^2) in
  [-1, 1]: grid points with |psi~| <= delta are treated as undetermined
  (hysteresis), a maximum is a determined + followed by a determined - inside
  the window; delta = 1e-9 canonical, checks at 1e-12, 1e-6, 1e-3;
* budget scan: geometric grid on [1e-3, 1e60] with 25 points per decade; every
  change of the count N(B) along the scan is recorded; B_top^mf is the last
  scanned budget with N = m bracketed against the next scan point and bisected
  in log B to a relative half-width of 1e-5 (well inside three significant
  digits); a retained set that is not an interval is reported explicitly.

Configurations: the W1 phase-diagram geometry (m = 2, 3 at equal weights on
the eight W1 eps values, which contain the four W2 chains per m), the W4
m = 5 stretched design (z0 = 8, explicit centres) and the W5 d = 3 spot check.

Output: artifacts/data/exact_m_fixed_budget/I4_mean_field_topology/mean_field_topology.json
(split run: --labels/--dt-checks/--dt-check-only with --part-out, then --merge).
The 2026-09-09 single-process output in exact_m_prr_upgrade/ is superseded.
Run:    python code/exact_m_prr_mean_field_topology.py   (deterministic, ~15 min)
        --quick skips the grid and tolerance checks (development only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import validate_exact_m_offlattice as base  # noqa: E402

REPORT = HERE.parents[1]
OUT_JSON_LEGACY = core.UPGRADE_DATA / "mean_field_topology.json"  # 2026-09-09 run (probe defect)
FIXED_BUDGET_DATA = core.REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
OUT_DIR = FIXED_BUDGET_DATA / "I4_mean_field_topology"
OUT_JSON = OUT_DIR / "mean_field_topology.json"
W2_JSON = core.UPGRADE_DATA / "w2_b0_empirical" / "B0_empirical.json"
RECLASS_JSON = core.UPGRADE_DATA / "covariance_aware_reclassification.json"
MFB_JSON = core.UPGRADE_DATA / "mean_field_boundary.json"
BCERT_ASSET = (
    REPORT / "manuscript" / "prr_submission" / "prr_assets"
    / "b0_quantitative_bound.tex"
)

WINDOW = core.WINDOW
T_MAX = base.DEFAULT_TMAX
DT_CANONICAL = 2.0e-5
DT_CHECKS = (4.0e-5, 1.0e-5)
TOL_CANONICAL = 1.0e-9
TOL_CHECKS = (1.0e-12, 1.0e-6, 1.0e-3)
B_SCAN = (1.0e-3, 1.0e60)
SCAN_PER_DECADE = 25
BISECT_REL_HALFWIDTH = 1.0e-5
G_CHUNK = 2000

W1_EPS_GRID = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25)
W2_EPS_GRID = (0.05, 0.1, 0.15, 0.2)
M_WEIGHTS = {2: (0.5, 0.5), 3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)}
W4_DESIGN = {
    "m": 5,
    "z0": 8.0,
    "centres_z": (2.8, 2.2, 1.6, 1.0, 0.4),
    "weights": (0.2, 0.2, 0.2, 0.2, 0.2),
    "eps": 0.10,
}
W5_DESIGN = {"m": 2, "eps": 0.10, "n_perp": 2}

# Nominal certificate values quoted from the Supplemental Material asset
# (prr_assets/b0_quantitative_bound.tex); they are NOT recomputed here.
BCERT_QUOTED = {
    "m2_eps0.1": {
        "value": 8.8974e-15,
        "source": "b0_quantitative_bound.tex eq. exactmfull-b0-anchor-value",
    },
    "m3_eps0.1": {
        "value": 2.5819e-7,
        "source": "b0_quantitative_bound.tex, m=3 geometry paragraph",
    },
}


# ----------------------------------------------------------------------------
# Mean-field clock: G, G', Lambda on a fine grid from t = 0.
# ----------------------------------------------------------------------------


class MeanFieldClock:
    """G(t), G'(t) (semi-analytic and finite-difference), Lambda(t)."""

    def __init__(
        self,
        *,
        label: str,
        m: int,
        eps: float,
        weights: tuple[float, ...],
        centres_z: np.ndarray,
        p,
        n_perp: int,
        dt: float,
    ):
        steps = int(round(T_MAX / dt))
        if abs(steps * dt - T_MAX) > 1e-12:
            raise ValueError("tmax must be an integer multiple of dt")
        self.label, self.m, self.eps, self.dt = label, m, eps, dt
        self.weights = tuple(float(w) for w in weights)
        self.centres_z = np.asarray(centres_z, dtype=float)
        self.p, self.n_perp = p, n_perp
        self.t = np.arange(steps + 1) * dt
        t = self.t

        g = np.empty_like(t)
        contact = np.empty_like(t)
        for start in range(0, t.size, G_CHUNK):
            block = core.free_exposure_general(
                t[start : start + G_CHUNK],
                centres_z=self.centres_z,
                eps=eps,
                weights=self.weights,
                p=p,
                n_perp=n_perp,
            )
            g[start : start + G_CHUNK] = block["g"]
            contact[start : start + G_CHUNK] = block["contact_factor"]
        sigma = base.mixture_sigma(eps, p)
        sgn = base.sign_mu_prime(p)
        centres_x = sgn * self.centres_z / p.ell0
        x = np.asarray(base.x_of_t(t, p), dtype=float)
        dx = sgn * (-p.gamma * (p.z0 - p.z_bar) * np.exp(-p.gamma * t)) / p.ell0
        h = base.mixture_h(x, centres_x, np.asarray(self.weights), sigma)
        dh = np.zeros_like(x)
        for w, c in zip(self.weights, centres_x):
            dh += w * np.exp(-((x - c) ** 2) / (2.0 * sigma**2)) * (-(x - c) / sigma**2)
        prefactor = 1.0 / (
            p.torus_w**n_perp
            * math.sqrt(2.0 * math.pi)
            * eps
            * math.sqrt(p.d0 / (2.0 * p.gamma) + p.rho**2)
        )
        g_rebuilt = prefactor * contact * h
        if not np.allclose(g_rebuilt, g, rtol=1e-12, atol=0.0):
            raise AssertionError("semi-analytic G disagrees with free_exposure_general")
        dcontact = np.gradient(contact, dt, edge_order=2)
        self.g = g
        self.contact = contact
        self.dg = prefactor * (dcontact * h + contact * dh * dx)
        self.dg_fd = np.gradient(g, dt, edge_order=2)
        self.sigma_x = float(sigma)
        self.centres_x = centres_x
        lam = np.zeros_like(g)
        lam[1:] = np.cumsum(0.5 * dt * (g[1:] + g[:-1]))
        self.lam = lam

        lo, hi = WINDOW
        i_lo, i_hi = int(round(lo / dt)), int(round(hi / dt))
        if abs(i_lo * dt - lo) > 1e-12 or abs(i_hi * dt - hi) > 1e-12:
            raise ValueError("window edges are not on the time grid")
        self.i_lo, self.i_hi = i_lo, i_hi
        self.window = slice(i_lo, i_hi + 1)

    # -- free-clock stationary points (B -> 0 limit) ------------------------

    def g_stationary_points(self) -> dict:
        t, g = self.t[self.window], self.g[self.window]
        peaks = [
            i for i in range(1, t.size - 1)
            if g[i] > g[i - 1] and g[i] >= g[i + 1]
        ]
        valleys = [
            i for i in range(1, t.size - 1)
            if g[i] < g[i - 1] and g[i] <= g[i + 1]
        ]
        return {
            "n_window_maxima": len(peaks),
            "peak_times": [float(t[i]) for i in peaks],
            "peak_values": [float(g[i]) for i in peaks],
            "valley_times": [float(t[i]) for i in valleys],
            "valley_values": [float(g[i]) for i in valleys],
            "valley_to_adjacent_peak_ratio": [
                float(g[v] / min(max([g[i] for i in peaks if i < v] or [np.inf]),
                                 max([g[i] for i in peaks if i > v] or [np.inf])))
                for v in valleys
            ],
            "lambda_at_valleys": [float(self.lam[self.i_lo + i]) for i in valleys],
            "lambda_at_window_end": float(self.lam[self.i_hi]),
            "lambda_total_to_tmax": float(self.lam[-1]),
        }

    # -- sign-change count of psi_B = G' - B G^2 on the window -------------

    def psi_normalized(self, budget: float, *, use_fd: bool = False) -> np.ndarray:
        dg = (self.dg_fd if use_fd else self.dg)[self.window]
        g2 = self.g[self.window] ** 2
        num = dg - budget * g2
        den = np.abs(dg) + budget * g2
        safe = den > 0.0
        out = np.zeros_like(num)
        out[safe] = num[safe] / den[safe]
        return out

    def count_interior_maxima(
        self, budget: float, *, tol: float = TOL_CANONICAL, use_fd: bool = False
    ) -> dict:
        psi = self.psi_normalized(budget, use_fd=use_fd)
        det = np.flatnonzero(np.abs(psi) > tol)
        if det.size < 2:
            return {"count": 0, "max_times": [], "min_times": []}
        s = np.sign(psi[det])
        down = np.flatnonzero((s[:-1] > 0.0) & (s[1:] < 0.0))
        up = np.flatnonzero((s[:-1] < 0.0) & (s[1:] > 0.0))
        t = self.t[self.window]
        max_times = [float(0.5 * (t[det[k]] + t[det[k + 1]])) for k in down]
        min_times = [float(0.5 * (t[det[k]] + t[det[k + 1]])) for k in up]
        return {"count": len(max_times), "max_times": max_times, "min_times": min_times}

    # -- level-set characterization: local maxima of r = G'/G^2 -------------

    def hazard_ratio_levels(self) -> dict:
        t = self.t[self.window]
        g = self.g[self.window]
        dg = self.dg[self.window]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.where(g > 0.0, dg / np.where(g > 0.0, g, 1.0) ** 2, np.nan)
        stat = self.g_stationary_points()
        peak_times = stat["peak_times"]
        valley_times = stat["valley_times"]
        rows = []
        for i in range(1, t.size - 1):
            if not (np.isfinite(r[i - 1]) and np.isfinite(r[i]) and np.isfinite(r[i + 1])):
                continue
            if r[i] > 0.0 and r[i] > r[i - 1] and r[i] >= r[i + 1]:
                denom = r[i + 1] - 2.0 * r[i] + r[i - 1]
                refined = (
                    r[i] - (r[i + 1] - r[i - 1]) ** 2 / (8.0 * denom)
                    if denom < 0.0
                    else r[i]
                )
                flank = 1 + sum(1 for tp in peak_times if tp < t[i])
                rows.append(
                    {
                        "time": float(t[i]),
                        "r_grid": float(r[i]),
                        "r_parabolic": float(refined),
                        "rising_flank_index": int(flank),
                        "preceding_valley_time": (
                            max([tv for tv in valley_times if tv < t[i]] or [None])
                        ),
                        "g_at_r_max": float(g[i]),
                    }
                )
        first_flank_sup = None
        if np.isfinite(r[0]) and (not peak_times or t[0] < peak_times[0]):
            first_flank_sup = float(r[0])
        levels_after_first_peak = [
            row["r_parabolic"] for row in rows if row["rising_flank_index"] >= 2
        ]
        return {
            "definition": (
                "interior maxima of f_1(.;B) are downward crossings of the level B "
                "by r(t) = G'(t)/G(t)^2; a rising flank of G contributes one "
                "maximum iff B < max_flank r"
            ),
            "r_at_window_start": float(r[0]) if np.isfinite(r[0]) else None,
            "first_flank_sup_if_monotone": first_flank_sup,
            "local_maxima_of_r": rows,
            "n_local_maxima_of_r_after_first_peak": len(levels_after_first_peak),
            "predicted_B_top_mf": (
                float(min(levels_after_first_peak)) if levels_after_first_peak else None
            ),
            "predicted_first_loss_flank": (
                int(min(
                    (row for row in rows if row["rising_flank_index"] >= 2),
                    key=lambda row: row["r_parabolic"],
                )["rising_flank_index"])
                if levels_after_first_peak
                else None
            ),
        }

    # -- exact (unsmoothed) f_1 diagnostics at a given budget ---------------

    def f1(self, budget: float) -> np.ndarray:
        return budget * self.g * np.exp(-budget * self.lam)

    def endpoint_psi_signs(self, budget: float, *, tol: float = TOL_CANONICAL) -> dict:
        """Signs of psi~_B at the window edges tau and T (0 = undetermined).

        +1 at tau and -1 at T means f_1 is increasing into the window and
        decreasing out of it, i.e. neither endpoint is stationary and the
        window maximum is interior (complete stationary signature)."""
        psi = self.psi_normalized(budget)

        def _sgn(v: float) -> int:
            return 0 if abs(v) <= tol else (1 if v > 0.0 else -1)

        return {"psi_sign_at_tau": _sgn(float(psi[0])), "psi_sign_at_T": _sgn(float(psi[-1]))}

    def last_mode_prominence(self, budget: float) -> dict | None:
        """Contour-base relative prominence of the last interior maximum of f_1.

        Log-domain cumulative-sum evaluation (I4 fix, 2026-09-23).  The earlier
        version formed log f_1 = log B + log G - B Lambda pointwise; at
        (m, eps) = (2, 0.05) near B_top^mf one has B Lambda ~ 1e20, whose ulp
        (~3e4) swamps the local variation of log f_1, so the 0.5/0.9/0.99 x
        B_top^mf probes miscounted maxima.  Here the per-step increments

            d_k = log(G_{k+1}/G_k) - B dLambda_k,
            dLambda_k = dt (G_k + G_{k+1}) / 2   (trapezoid, formed directly,
                                                    never as a difference of
                                                    the cumulative Lambda),

        are exact to relative rounding; maxima are + -> - sign changes of d_k,
        and all heights/bases are cumulative sums of d_k anchored AT the last
        maximum, so the local contour (base, height) is resolved to rounding
        of O(local variation) however large B Lambda is.  The relative
        prominence is exp(log h - log max) [1 - exp(log base - log h)], with
        its log10 evaluated entirely in log space."""
        g = self.g[self.window]
        if np.any(g <= 0.0):
            return None
        t = self.t[self.window]
        dlam = 0.5 * self.dt * (g[1:] + g[:-1])
        inc = np.log(g[1:] / g[:-1]) - budget * dlam  # inc[k] = lf[k+1] - lf[k]
        # interior maxima: lf[i] > lf[i-1] and lf[i] >= lf[i+1]
        peaks = [i for i in range(1, t.size - 1) if inc[i - 1] > 0.0 and inc[i] <= 0.0]
        minima = [i for i in range(1, t.size - 1) if inc[i - 1] < 0.0 and inc[i] >= 0.0]
        psi_count = self.count_interior_maxima(budget)
        signs = self.endpoint_psi_signs(budget)
        if not peaks:
            return {
                "budget": float(budget),
                "n_interior_maxima": 0,
                "n_interior_maxima_psi": psi_count["count"],
                **signs,
            }
        i = peaks[-1]
        # rel[j] = lf[j] - lf[i], anchored at the last maximum
        rel = np.zeros(t.size)
        rel[i + 1 :] = np.cumsum(inc[i:])
        rel[:i] = -np.cumsum(inc[:i][::-1])[::-1]
        left_higher = np.flatnonzero(rel[:i] > 0.0)
        j = int(left_higher[-1]) if left_higher.size else -1
        base_left = float(rel[j + 1 : i + 1].min())
        right_higher = np.flatnonzero(rel[i + 1 :] > 0.0)
        k = i + 1 + int(right_higher[0]) if right_higher.size else t.size
        base_right = float(rel[i:k].min())
        log_base = max(base_left, base_right)  # <= 0, relative to the peak
        log_max = float(rel.max())  # >= 0, window max relative to the peak
        depth = -math.expm1(log_base)  # 1 - base/height in [0, 1]
        rel_prom = math.exp(-log_max) * depth
        log10_rel = (-log_max) / math.log(10.0) + math.log10(depth) if depth > 0.0 else None
        # absolute log height of the last peak (for reference only; its
        # absolute accuracy is limited by B Lambda, the relative quantities
        # above are not)
        h_abs = math.log(budget) + math.log(g[i]) - budget * float(self.lam[self.i_lo + i])
        return {
            "budget": float(budget),
            "method": "log-domain cumulative sum of per-step increments anchored at the last maximum",
            "n_interior_maxima": len(peaks),
            "n_interior_minima": len(minima),
            "n_interior_maxima_psi": psi_count["count"],
            "maxima_count_agrees_with_psi": bool(len(peaks) == psi_count["count"]),
            "maxima_times": [float(t[q]) for q in peaks],
            "minima_times": [float(t[q]) for q in minima],
            **signs,
            "last_max_time": float(t[i]),
            "log10_last_max_height_abs_approx": h_abs / math.log(10.0),
            "log10_window_max_over_last_max": log_max / math.log(10.0),
            "log10_base_over_last_max": log_base / math.log(10.0),
            "relative_prominence": float(rel_prom),
            "log10_relative_prominence": log10_rel,
        }


# ----------------------------------------------------------------------------
# Scan and bisection.
# ----------------------------------------------------------------------------


def scan_budget_grid() -> np.ndarray:
    lo, hi = B_SCAN
    n = int(round(SCAN_PER_DECADE * math.log10(hi / lo))) + 1
    return np.geomspace(lo, hi, n)


def scan_and_bisect(
    clock: MeanFieldClock,
    *,
    tol: float = TOL_CANONICAL,
    use_fd: bool = False,
    keep_scan: bool = False,
) -> dict:
    m = clock.m
    grid = scan_budget_grid()
    counts = np.array(
        [clock.count_interior_maxima(float(b), tol=tol, use_fd=use_fd)["count"] for b in grid],
        dtype=int,
    )
    transitions = [
        {
            "from_budget": float(grid[i]),
            "to_budget": float(grid[i + 1]),
            "from_count": int(counts[i]),
            "to_count": int(counts[i + 1]),
        }
        for i in range(grid.size - 1)
        if counts[i] != counts[i + 1]
    ]
    retained = np.flatnonzero(counts == m)
    out = {
        "tolerance": tol,
        "derivative": "finite_difference" if use_fd else "semi_analytic",
        "count_at_scan_minimum": int(counts[0]),
        "count_at_scan_maximum": int(counts[-1]),
        "max_count_in_scan": int(counts.max()),
        "n_scan_points": int(grid.size),
        "n_retained_scan_points": int(retained.size),
        "transitions": transitions,
    }
    if keep_scan:
        out["scan"] = [
            {"budget": float(b), "count": int(c)} for b, c in zip(grid, counts)
        ]
    if retained.size == 0:
        out.update(
            status="no_scanned_budget_retains_m_maxima",
            B_top_mf=None,
            bracket=None,
            retained_set_is_interval=None,
        )
        return out
    last = int(retained[-1])
    contiguous = bool(np.all(counts[: last + 1] == m))
    out["retained_set_is_interval"] = contiguous
    out["retained_scan_range"] = [float(grid[retained[0]]), float(grid[last])]
    if not contiguous:
        out["retained_set_gaps"] = [
            float(grid[i]) for i in range(last + 1) if counts[i] != m
        ]
    if last == grid.size - 1:
        out.update(status="right_censored_at_scan_maximum", B_top_mf=None, bracket=None)
        return out
    b_lo, b_hi = float(grid[last]), float(grid[last + 1])
    iterations = 0
    while math.sqrt(b_hi / b_lo) - 1.0 > BISECT_REL_HALFWIDTH:
        mid = math.sqrt(b_lo * b_hi)
        if clock.count_interior_maxima(mid, tol=tol, use_fd=use_fd)["count"] == m:
            b_lo = mid
        else:
            b_hi = mid
        iterations += 1
    below = clock.count_interior_maxima(b_lo, tol=tol, use_fd=use_fd)
    above = clock.count_interior_maxima(b_hi, tol=tol, use_fd=use_fd)
    signs_below = clock.endpoint_psi_signs(b_lo, tol=tol)
    signs_above = clock.endpoint_psi_signs(b_hi, tol=tol)
    lost = sorted(set(np.round(below["max_times"], 4)) - set(np.round(above["max_times"], 4)))
    out.update(
        status="bisected",
        B_top_mf=math.sqrt(b_lo * b_hi),
        bracket=[b_lo, b_hi],
        relative_halfwidth=math.sqrt(b_hi / b_lo) - 1.0,
        bisection_iterations=iterations,
        count_just_below=int(below["count"]),
        count_just_above=int(above["count"]),
        maxima_times_just_below=below["max_times"],
        maxima_times_just_above=above["max_times"],
        minima_times_just_below=below["min_times"],
        minima_times_just_above=above["min_times"],
        endpoint_psi_signs_just_below=signs_below,
        endpoint_psi_signs_just_above=signs_above,
        lost_maximum_time=[float(v) for v in lost],
    )
    return out


def three_sig(x: float | None) -> str | None:
    if x is None:
        return None
    return f"{x:.3g}"


# ----------------------------------------------------------------------------
# Configurations.
# ----------------------------------------------------------------------------


def build_clock(cfg: dict, dt: float) -> MeanFieldClock:
    p = core.make_model(**cfg.get("model_overrides", {}))
    if cfg.get("centres_z") is not None:
        centres = np.asarray(cfg["centres_z"], dtype=float)
    else:
        centres = core.centres_z_for(cfg["m"], p=p)
    return MeanFieldClock(
        label=cfg["label"],
        m=cfg["m"],
        eps=cfg["eps"],
        weights=tuple(cfg["weights"]),
        centres_z=centres,
        p=p,
        n_perp=cfg.get("n_perp", 1),
        dt=dt,
    )


def configurations(pilot: bool = False) -> list[dict]:
    cfgs = []
    for m in (2, 3):
        for eps in W1_EPS_GRID:
            cfgs.append(
                {
                    "label": f"m{m}_eps{eps:g}",
                    "family": "w1_w2_standard_design",
                    "m": m,
                    "eps": eps,
                    "weights": M_WEIGHTS[m],
                    "centres_z": None,
                    "n_perp": 1,
                    "on_w2_grid": eps in W2_EPS_GRID,
                    "target_times": list(base.TARGET_TIMES[m]),
                }
            )
    cfgs.append(
        {
            "label": "w4_m5_z0_8_eps0.1",
            "family": "w4_m5_demo",
            "m": W4_DESIGN["m"],
            "eps": W4_DESIGN["eps"],
            "weights": W4_DESIGN["weights"],
            "centres_z": list(W4_DESIGN["centres_z"]),
            "model_overrides": {"z0": W4_DESIGN["z0"]},
            "n_perp": 1,
            "on_w2_grid": False,
            "target_times": [
                math.log(W4_DESIGN["z0"] / mu) for mu in W4_DESIGN["centres_z"]
            ],
        }
    )
    cfgs.append(
        {
            "label": "w5_d3_m2_eps0.1",
            "family": "w5_d3_spotcheck",
            "m": W5_DESIGN["m"],
            "eps": W5_DESIGN["eps"],
            "weights": M_WEIGHTS[2],
            "centres_z": None,
            "n_perp": W5_DESIGN["n_perp"],
            "on_w2_grid": False,
            "target_times": list(base.TARGET_TIMES[2]),
        }
    )
    if pilot:
        cfgs = [c for c in cfgs if c["label"] in ("m2_eps0.1", "m3_eps0.2")]
    return cfgs


# ----------------------------------------------------------------------------
# Stored comparison values.
# ----------------------------------------------------------------------------


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stored_comparators() -> dict:
    """B_op (covariance-aware, stored) and B_op^mf (mean_field_boundary.json)."""
    recl = _read(RECLASS_JSON)
    mfb = _read(MFB_JSON)
    out = {}
    for c in recl["w2_chains"]:
        key = f"m{int(c['m'])}_eps{float(c['eps']):g}"
        out.setdefault(key, {})
        out[key]["B_op_status"] = c["new_status"]
        out[key]["B_op"] = c.get("new_b0")
        out[key]["B_op_bracket"] = c.get("new_bracket")
    for ch in mfb["chains"]:
        key = f"m{int(ch['m'])}_eps{float(ch['eps']):g}"
        out.setdefault(key, {})
        out[key]["B_op_mf_status"] = ch["headline_status"]
        out[key]["B_op_mf"] = ch["headline_upper_crossing"]
    return out


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------


def run_dt_check(cfg: dict, dt: float) -> dict:
    """One grid check (separate process for the expensive d = 3 builds)."""
    started = time.perf_counter()
    clock_dt = build_clock(cfg, dt)
    res = scan_and_bisect(clock_dt)
    lv = clock_dt.hazard_ratio_levels()
    return {
        "label": cfg["label"],
        "dt": dt,
        "status": res["status"],
        "B_top_mf": res["B_top_mf"],
        "level_set_predicted_B_top_mf": lv["predicted_B_top_mf"],
        "runtime_seconds": time.perf_counter() - started,
    }


def run_configuration(
    cfg: dict, *, quick: bool, comparators: dict, dt_checks: tuple = DT_CHECKS
) -> dict:
    started = time.perf_counter()
    clock = build_clock(cfg, DT_CANONICAL)
    stationary = clock.g_stationary_points()
    canonical = scan_and_bisect(clock, keep_scan=False)
    levels = clock.hazard_ratio_levels()
    b_top = canonical["B_top_mf"]

    checks = {"tolerance": [], "grid": [], "finite_difference_derivative": None}
    if not quick:
        for tol in TOL_CHECKS:
            res = scan_and_bisect(clock, tol=tol)
            checks["tolerance"].append(
                {
                    "tolerance": tol,
                    "status": res["status"],
                    "B_top_mf": res["B_top_mf"],
                    "relative_difference": (
                        res["B_top_mf"] / b_top - 1.0
                        if (res["B_top_mf"] is not None and b_top)
                        else None
                    ),
                }
            )
        res = scan_and_bisect(clock, use_fd=True)
        checks["finite_difference_derivative"] = {
            "status": res["status"],
            "B_top_mf": res["B_top_mf"],
            "relative_difference": (
                res["B_top_mf"] / b_top - 1.0
                if (res["B_top_mf"] is not None and b_top)
                else None
            ),
        }
        for dt in dt_checks:
            clock_dt = build_clock(cfg, dt)
            res = scan_and_bisect(clock_dt)
            lv = clock_dt.hazard_ratio_levels()
            checks["grid"].append(
                {
                    "dt": dt,
                    "status": res["status"],
                    "B_top_mf": res["B_top_mf"],
                    "relative_difference": (
                        res["B_top_mf"] / b_top - 1.0
                        if (res["B_top_mf"] is not None and b_top)
                        else None
                    ),
                    "level_set_predicted_B_top_mf": lv["predicted_B_top_mf"],
                }
            )
    rel_devs = [
        abs(row["relative_difference"])
        for row in checks["tolerance"] + checks["grid"]
        if row["relative_difference"] is not None
    ]
    fd = checks["finite_difference_derivative"]
    if fd and fd["relative_difference"] is not None:
        rel_devs.append(abs(fd["relative_difference"]))

    # Diagnostics at and below the threshold (exact f_1, unsmoothed).
    diagnostics = {}
    if b_top is not None:
        valleys = stationary["valley_times"]
        lam_v = stationary["lambda_at_valleys"][-1] if valleys else None
        lam_T = stationary["lambda_at_window_end"]
        diagnostics = {
            "fold_time_of_lost_maximum": canonical["lost_maximum_time"],
            "last_valley_time_free_clock": valleys[-1] if valleys else None,
            "lambda_at_last_valley": lam_v,
            "log10_survival_at_last_valley_at_B_top_mf": (
                -b_top * lam_v / math.log(10.0) if lam_v is not None else None
            ),
            "log10_last_basin_mass_fraction_at_B_top_mf": (
                (
                    math.log10(math.exp(-b_top * lam_v) - math.exp(-b_top * lam_T))
                    if b_top * lam_v < 700.0
                    else -b_top * lam_v / math.log(10.0)
                )
                if lam_v is not None
                else None
            ),
            "last_mode_relative_prominence": {},
        }
        probes = {
            "free_clock_limit_B_to_0": 1e-6,
            "0.5_B_top_mf": 0.5 * b_top,
            "0.9_B_top_mf": 0.9 * b_top,
            "0.99_B_top_mf": 0.99 * b_top,
        }
        comp = comparators.get(cfg["label"], {})
        if comp.get("B_op") is not None:
            probes["B_op_stored"] = comp["B_op"]
        if comp.get("B_op_mf") is not None:
            probes["B_op_mf_stored"] = comp["B_op_mf"]
        for name, b in probes.items():
            diagnostics["last_mode_relative_prominence"][name] = clock.last_mode_prominence(b)

    level_vs_bisect = None
    if b_top is not None and levels["predicted_B_top_mf"] is not None:
        level_vs_bisect = levels["predicted_B_top_mf"] / b_top - 1.0

    comp = comparators.get(cfg["label"], {})
    ordering = {}
    if b_top is not None:
        if comp.get("B_op") is not None:
            ordering["B_top_mf_over_B_op"] = b_top / comp["B_op"]
            ordering["log10_B_top_mf_over_B_op"] = math.log10(b_top / comp["B_op"])
        if comp.get("B_op_mf") is not None:
            ordering["B_top_mf_over_B_op_mf"] = b_top / comp["B_op_mf"]
            ordering["log10_B_top_mf_over_B_op_mf"] = math.log10(b_top / comp["B_op_mf"])
        if comp.get("B_op_status") == "right_censored":
            ordering["B_top_mf_over_B_op"] = None
            ordering["note"] = "B_op right-censored at 8; B_top_mf compared with the censoring limit"
            ordering["B_top_mf_over_8"] = b_top / 8.0

    return {
        "label": cfg["label"],
        "family": cfg["family"],
        "m": cfg["m"],
        "eps": cfg["eps"],
        "n_perp": cfg.get("n_perp", 1),
        "dimension": cfg.get("n_perp", 1) + 1,
        "weights": list(cfg["weights"]),
        "centres_z": [float(c) for c in clock.centres_z],
        "centres_x": [float(c) for c in clock.centres_x],
        "target_times": cfg["target_times"],
        "z0": clock.p.z0,
        "sigma_x_space": clock.sigma_x,
        "on_w2_grid": cfg["on_w2_grid"],
        "free_clock": stationary,
        "free_clock_has_m_window_maxima": bool(stationary["n_window_maxima"] == cfg["m"]),
        "status": canonical["status"],
        "B_top_mf": b_top,
        "B_top_mf_3sf": three_sig(b_top),
        "log10_B_top_mf": math.log10(b_top) if b_top else None,
        "bracket": canonical["bracket"],
        "relative_halfwidth": canonical.get("relative_halfwidth"),
        "bisection_iterations": canonical.get("bisection_iterations"),
        "retained_set_is_interval": canonical.get("retained_set_is_interval"),
        "retained_scan_range": canonical.get("retained_scan_range"),
        "retained_set_gaps": canonical.get("retained_set_gaps"),
        "max_count_in_scan": canonical["max_count_in_scan"],
        "count_at_scan_minimum": canonical["count_at_scan_minimum"],
        "count_at_scan_maximum": canonical["count_at_scan_maximum"],
        "transitions": canonical["transitions"],
        "first_loss": {
            "count_just_below": canonical.get("count_just_below"),
            "count_just_above": canonical.get("count_just_above"),
            "maxima_times_just_below": canonical.get("maxima_times_just_below"),
            "maxima_times_just_above": canonical.get("maxima_times_just_above"),
            "minima_times_just_below": canonical.get("minima_times_just_below"),
            "minima_times_just_above": canonical.get("minima_times_just_above"),
            "n_minima_just_below": (
                len(canonical["minima_times_just_below"])
                if canonical.get("minima_times_just_below") is not None else None
            ),
            "n_minima_just_above": (
                len(canonical["minima_times_just_above"])
                if canonical.get("minima_times_just_above") is not None else None
            ),
            "endpoint_psi_signs_just_below": canonical.get("endpoint_psi_signs_just_below"),
            "endpoint_psi_signs_just_above": canonical.get("endpoint_psi_signs_just_above"),
            "complete_signature_just_below": (
                bool(
                    canonical["count_just_below"] == cfg["m"]
                    and len(canonical["minima_times_just_below"]) == cfg["m"] - 1
                    and canonical["endpoint_psi_signs_just_below"]["psi_sign_at_tau"] == 1
                    and canonical["endpoint_psi_signs_just_below"]["psi_sign_at_T"] == -1
                )
                if canonical.get("count_just_below") is not None else None
            ),
            "lost_maximum_time": canonical.get("lost_maximum_time"),
        },
        "level_set_cross_check": {
            **levels,
            "relative_difference_level_set_vs_bisection": level_vs_bisect,
        },
        "sensitivity": {
            **checks,
            "max_abs_relative_deviation_from_canonical": (
                max(rel_devs) if rel_devs else None
            ),
        },
        "diagnostics_at_threshold": diagnostics,
        "stored_comparators": comp,
        "ordering": ordering,
        "runtime_seconds": time.perf_counter() - started,
    }


def _run_part(args) -> None:
    """Run a subset of labels (and/or isolated dt checks) into a part file."""
    started = time.perf_counter()
    comparators = stored_comparators()
    cfgs = {c["label"]: c for c in configurations(pilot=args.pilot)}
    labels = [x for x in (args.labels.split(",") if args.labels else cfgs) if x]
    unknown = [x for x in labels if x not in cfgs]
    if unknown:
        raise SystemExit(f"unknown labels {unknown}")
    dt_checks = DT_CHECKS if args.dt_checks is None else tuple(
        float(v) for v in args.dt_checks.split(",") if v.strip() and v.strip() != "none"
    )
    results, dt_only = [], []
    if args.dt_check_only is not None:
        for lab in labels:
            row = run_dt_check(cfgs[lab], args.dt_check_only)
            dt_only.append(row)
            print(f"[{lab}] dt={row['dt']} B_top_mf={three_sig(row['B_top_mf'])} "
                  f"[{row['runtime_seconds']:.1f}s]", flush=True)
    else:
        for lab in labels:
            res = run_configuration(
                cfgs[lab], quick=args.quick, comparators=comparators, dt_checks=dt_checks
            )
            res["dt_checks_run_in_this_part"] = list(dt_checks)
            results.append(res)
            print(
                f"[{res['label']}] status={res['status']} B_top_mf={res['B_top_mf_3sf']} "
                f"level-set={three_sig(res['level_set_cross_check']['predicted_B_top_mf'])} "
                f"interval={res['retained_set_is_interval']} "
                f"maxdev={res['sensitivity']['max_abs_relative_deviation_from_canonical']} "
                f"sig={res['first_loss'].get('complete_signature_just_below')} "
                f"[{res['runtime_seconds']:.1f}s]",
                flush=True,
            )
    core.write_json(
        args.part_out,
        {
            "part": True,
            "labels": labels,
            "dt_checks_run": list(dt_checks) if args.dt_check_only is None else [],
            "dt_check_only": args.dt_check_only,
            "configurations": results,
            "dt_check_rows": dt_only,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[part] wrote {args.part_out} in {time.perf_counter() - started:.1f} s", flush=True)


def _merge_parts(paths: list[Path]) -> tuple[list[dict], list[dict], float, list[float]]:
    parts = [_read(p) for p in paths]
    results = [r for part in parts for r in part["configurations"]]
    dt_rows = [r for part in parts for r in part["dt_check_rows"]]
    by_label = {r["label"]: r for r in results}
    for row in dt_rows:
        res = by_label[row["label"]]
        b_top = res["B_top_mf"]
        res["sensitivity"]["grid"].append(
            {
                "dt": row["dt"],
                "status": row["status"],
                "B_top_mf": row["B_top_mf"],
                "relative_difference": (
                    row["B_top_mf"] / b_top - 1.0
                    if (row["B_top_mf"] is not None and b_top) else None
                ),
                "level_set_predicted_B_top_mf": row["level_set_predicted_B_top_mf"],
                "run_in_separate_process": True,
                **({"carried_over_from": row["carried_over_from"]} if row.get("carried_over_from") else {}),
            }
        )
        res["runtime_seconds"] += row["runtime_seconds"]
    for res in results:
        sens = res["sensitivity"]
        devs = [
            abs(r["relative_difference"])
            for r in sens["tolerance"] + sens["grid"]
            if r["relative_difference"] is not None
        ]
        fd = sens["finite_difference_derivative"]
        if fd and fd["relative_difference"] is not None:
            devs.append(abs(fd["relative_difference"]))
        sens["max_abs_relative_deviation_from_canonical"] = max(devs) if devs else None
        sens["grid"].sort(key=lambda r: -r["dt"])
        got = sorted(r["dt"] for r in sens["grid"])
        if got != sorted(DT_CHECKS):
            raise SystemExit(f"{res['label']}: dt checks {got} incomplete")
    order = [c["label"] for c in configurations()]
    missing = [x for x in order if x not in by_label]
    if missing:
        raise SystemExit(f"merge incomplete, missing {missing}")
    results.sort(key=lambda r: order.index(r["label"]))
    part_runtimes = [float(p["runtime_seconds"]) for p in parts]
    return results, dt_rows, sum(part_runtimes), part_runtimes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="skip grid/tolerance checks")
    parser.add_argument("--pilot", action="store_true", help="two configurations only")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--labels", default=None, help="comma-separated subset (part run)")
    parser.add_argument("--dt-checks", default=None,
                        help="comma-separated dt grid checks for this part ('none' = skip)")
    parser.add_argument("--dt-check-only", type=float, default=None,
                        help="run only this dt grid check for --labels (separate process)")
    parser.add_argument("--part-out", type=Path, default=None,
                        help="write a part file instead of the merged payload")
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="merge part files into --out")
    args = parser.parse_args()
    started = time.perf_counter()

    if args.part_out is not None:
        _run_part(args)
        return
    part_runtimes = None
    if args.merge:
        results, _, total_process_seconds, part_runtimes = _merge_parts(args.merge)
    else:
        comparators = stored_comparators()
        results = [
            run_configuration(cfg, quick=args.quick, comparators=comparators)
            for cfg in configurations(pilot=args.pilot)
        ]
        total_process_seconds = time.perf_counter() - started

    table = []
    for res in results:
        if not (res["family"] == "w1_w2_standard_design" and res["on_w2_grid"]):
            continue
        comp = res["stored_comparators"]
        key = res["label"]
        table.append(
            {
                "m": res["m"],
                "eps": res["eps"],
                "B_cert_quoted": BCERT_QUOTED.get(key, {}).get("value"),
                "B_cert_source": BCERT_QUOTED.get(key, {}).get("source"),
                "B_op_covariance_aware": comp.get("B_op"),
                "B_op_status": comp.get("B_op_status"),
                "B_op_mf": comp.get("B_op_mf"),
                "B_op_mf_status": comp.get("B_op_mf_status"),
                "B_top_mf": res["B_top_mf"],
                "B_top_mf_3sf": res["B_top_mf_3sf"],
                "log10_B_top_mf": res["log10_B_top_mf"],
                "status": res["status"],
                "log10_survival_at_last_valley_at_B_top_mf": res["diagnostics_at_threshold"].get(
                    "log10_survival_at_last_valley_at_B_top_mf"
                ),
                "ordering": res["ordering"],
            }
        )

    def _all(key):
        return [r[key] for r in results if r[key] is not None]

    payload = {
        "schema_version": 1,
        "analysis": (
            "classifier-free topology threshold B_top^mf(eps) of the mean-field "
            "hazard-survival law f_1 = B G exp(-B Lambda): sup of B such that f_1 "
            "has exactly m interior nondegenerate maxima on the window"
        ),
        "not_a_theorem": (
            "B_top^mf is a threshold of the parameter-free mean-field law, not of the "
            "exact Doi process; it is neither B_top^unif (theorem) nor B_cert (certificate) "
            "nor B_op (classifier crossing)"
        ),
        "conventions": {
            "definition": (
                "B_top^mf := sup{B > 0 : f_1(.;B) has exactly m interior maxima on "
                "I = [tau, T]}; an interior maximum is a determined + followed by a "
                "determined - of psi~_B = (G' - B G^2)/(|G'| + B G^2) inside the window"
            ),
            "derivative_factorization": "sign f_1' = sign(G' - B G^2) because B exp(-B Lambda) > 0",
            "window": list(WINDOW),
            "time_grid": {
                "t0": 0.0,
                "tmax": T_MAX,
                "dt_canonical": DT_CANONICAL,
                "dt_checks": list(DT_CHECKS),
                "lambda_lower_limit": "t = 0 (composite trapezoid rule on the fine grid)",
            },
            "free_clock": (
                "exact_m_prr_upgrade_core.free_exposure_general: G = c(t) H(x(t)) / "
                "(W^(d-1) sqrt(2 pi) eps sqrt(D0/2gamma + rho^2)); contact quadrature "
                f"{base.THEORY_Y_POINTS} transverse points x {base.THEORY_WRAP_IMAGES} "
                f"images (d=2), {core.D3_Y_POINTS}^2 points (d=3)"
            ),
            "g_derivative": (
                "semi-analytic: G' = P [c' H + c H'(x) x'] with H', x' in closed form "
                "and c' by second-order central differences of the contact quadrature; "
                "the fully finite-difference G' is a stored check"
            ),
            "sign_change_tolerance": {
                "canonical": TOL_CANONICAL,
                "checks": list(TOL_CHECKS),
                "rule": "|psi~| <= tolerance is undetermined; count + -> - transitions of the determined sequence",
            },
            "budget_scan": {
                "interval": list(B_SCAN),
                "points_per_decade": SCAN_PER_DECADE,
                "n_points": int(scan_budget_grid().size),
                "bisection": "log-B bisection of the last N = m scan point against the next point",
                "bisection_relative_halfwidth": BISECT_REL_HALFWIDTH,
            },
            "level_set_cross_check": (
                "local maxima of r = G'/G^2 on the rising flanks of G after the first "
                "free-clock maximum, parabolic-refined on the grid; the smallest such "
                "level is the predicted B_top^mf"
            ),
            "stored_comparators": {
                "B_op": "covariance_aware_reclassification.json w2_chains new_b0 (formal classifier)",
                "B_op_mf": "mean_field_boundary.json chains headline_upper_crossing",
                "B_cert": "quoted from the SM asset b0_quantitative_bound.tex (two anchors)",
            },
        },
        "headline": {
            "w2_comparison_table": table,
            "B_top_mf_3sf_by_label": {r["label"]: r["B_top_mf_3sf"] for r in results},
            "log10_B_top_mf_by_label": {r["label"]: r["log10_B_top_mf"] for r in results},
            "status_by_label": {r["label"]: r["status"] for r in results},
            "retained_set_is_interval_all": all(
                r["retained_set_is_interval"] for r in results if r["retained_set_is_interval"] is not None
            ),
            "max_count_in_scan_exceeds_m_any": any(r["max_count_in_scan"] > r["m"] for r in results),
            "max_abs_relative_deviation_level_set_vs_bisection": max(
                abs(r["level_set_cross_check"]["relative_difference_level_set_vs_bisection"])
                for r in results
                if r["level_set_cross_check"]["relative_difference_level_set_vs_bisection"] is not None
            ),
            "max_abs_relative_deviation_sensitivity": max(
                _all("B_top_mf") and [
                    r["sensitivity"]["max_abs_relative_deviation_from_canonical"]
                    for r in results
                    if r["sensitivity"]["max_abs_relative_deviation_from_canonical"] is not None
                ] or [0.0]
            ),
        },
        "configurations": results,
        "runtime_seconds": total_process_seconds,
        "runtime_note": (
            "sum of process CPU-wall seconds over the parts"
            if part_runtimes is not None else "single-process wall seconds"
        ),
        "part_runtimes_seconds": part_runtimes,
        "merged_from_parts": [str(p) for p in args.merge] if args.merge else None,
        "revision": (
            "2026-09-23 I4: log-domain cumulative-sum last_mode_prominence (fixes the "
            "invalid 0.5/0.9/0.99 x B_top probes at (2, 0.05)); first_loss stores "
            "minima times and endpoint psi signs; split-by-label execution"
        ),
    }
    core.write_json(args.out, payload)
    print(f"[done] wrote {args.out} in {payload['runtime_seconds']:.1f} s", flush=True)
    print(json.dumps(payload["headline"]["B_top_mf_3sf_by_label"], indent=1))


if __name__ == "__main__":
    main()
