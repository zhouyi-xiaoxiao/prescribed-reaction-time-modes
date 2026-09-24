#!/usr/bin/env python3
"""Post-selection (null) calibration of the covariance-aware five-sigma classifier.

Question.  The article's operational classifier
(``reclassify_covariance_aware.classify_both``: bin width 0.02 on the window
I = [0.5, 3.5], Gaussian smoothing bandwidth 0.04, a local maximum counts as
a mode iff its topographic prominence exceeds 5 sigma_prom AND 5% of the
maximum smoothed height, sigma_prom^2 = sum_j (A_pj - A_bj)^2 C_j / (N delta)^2)
selects the peak bin and the contour-base bin from the data.  The plug-in
sigma_prom does not account for that selection (look-elsewhere effect across
~150 bins), so the "five sigma" gate is not automatically a calibrated
post-selection significance level.  This driver measures the actual null
behaviour: how often does the full classifier report a SPURIOUS significant
maximum on a histogram drawn from a smooth UNIMODAL reference density at the
production walker numbers, and how large does the largest spurious z get?

Null references (all unimodal on I at bin level; asserted in-run):
  A  ``mf_single_slab``          -- mean-field law f1 = B G e^{-B Lambda} for a
       single slab (m = 1) centred at a production target time t_j
       (mu(t_j) via core.centres_z_for), eps = 0.1, B = 1.
  B  ``mf_dominant_component``   -- for a production / phase-grid configuration
       (m, eps, B, w): the mean-field law is linear in the slab components
       given Lambda, f1 = sum_j f1^(j), f1^(j) = B w_j G_j e^{-B Lambda}, with
       Lambda = int_0^t G from the FULL mixture G = sum_j w_j G_j.  Only the
       dominant component (largest window maximum of f1^(j)) is retained.
  C  ``empirical_reweighted``    -- the stored production smoothed density
       s_i (bandwidth 0.04, N = 5e6) multiplied by the mean-field share of the
       dominant component at the bin centre, w_d G_d(t_i) / G(t_i) (the
       survival factor cancels in the ratio), so the other modes are removed
       "via the mean-field law" while the empirical peak shape is kept.
Expected bin counts lambda_i = N int_bin f dt (composite trapezoid on the
mean-field fine grid, dt_mf = 2.5e-4 = 80 steps per window bin, exactly as in
``exact_m_prr_mean_field_boundary.py``).  Replicas draw C_i ~ Poisson(lambda_i)
independently (the classifier's own variance model; conservative relative to
the multinomial truth), plus one multinomial cross-check.  The formal
classifier is applied verbatim.  In every replica the local maximum of largest
smoothed height is the true mode; every other local maximum is spurious.

Reported per reference (Wilson 95% intervals): gate-only false-positive rate
P(max spurious z >= 5), full-rule false-positive rate P(a spurious maximum
passes z >= 5 AND the 5% floor), floor-only rate, P(mode count != 1), and the
distribution of the largest spurious z (quantiles, maximum, fixed-bin
histogram).  The 48-cell phase-diagram multiplicity is emulated directly:
each (eps, B) cell of the m = 2 grid (and of the m = 3 grid) gets its own
dominant-component null at the W1 grid walker count, F synthetic phase
diagrams are drawn, and the family-wise rates over 48 cells (and over the
96-cell union) are counted.

Seeds: SeedSequence([CAMPAIGN_SEED, TAG_D2_NULL, reference index, chunk
index]) -> Philox; every chunk is an independent, reproducible substream.

Outputs (2026-09-23 I5 revision; the 2026-09-09 pilot in
exact_m_prr_upgrade/robustness/classifier_null_calibration/ is reproduced
bit-for-bit by --pilot, written to pilot_reproduction.json):
  artifacts/data/exact_m_fixed_budget/I5_classifier_null/summary.json
  artifacts/data/exact_m_fixed_budget/I5_classifier_null/power_curves.json (--power)
  artifacts/figures/fb_classifier_null_calibration.{png,pdf}
  artifacts/figures/fb_classifier_power.{png,pdf}
Additional blocks: stress_flat_plateau (look-elsewhere on a flat window),
shoulder_family (full mean-field f1 in the grid cells where the late mode is a
shoulder; seed tag 72), per-cell Wilson bounds, power curves (seed base
20260923, tag 87).  Wording: operational gate, not a calibrated
post-selection level.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import exact_m_prr_mean_field_boundary as mfb  # noqa: E402
import exact_m_prr_upgrade_w1 as w1  # noqa: E402  (grid constants only)
import reclassify_covariance_aware as recl  # noqa: E402
import validate_exact_m_offlattice as base  # noqa: E402

LEGACY_PILOT_JSON = (
    core.UPGRADE_DATA / "robustness" / "classifier_null_calibration" / "summary.json"
)  # 2026-09-09 pilot (2000 replicas, 100 families); reproduced by --pilot
OUT_DIR = core.REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "I5_classifier_null"
OUT_JSON = OUT_DIR / "summary.json"
PILOT_JSON = OUT_DIR / "pilot_reproduction.json"
POWER_JSON = OUT_DIR / "power_curves.json"
FIG_STEM = core.FIGURES / "fb_classifier_null_calibration"
POWER_FIG_STEM = core.FIGURES / "fb_classifier_power"
PRODUCTION_DIR = core.REPORT / "artifacts" / "data" / "exact_m_offlattice_production"

TAG_D2_NULL = 71  # disjoint from 1-3 (base), 11-53 (W1-W5), 61-65 (robustness)
TAG_SHOULDER = 72  # full mean-field shoulder family (plan I5), campaign seed 20260813
POWER_SEED = 20260923  # fixed-budget campaign base seed
TAG_POWER = 87  # power curves (plan tags 80-89; 81-86 claimed by N-items)
MATCH_RADIUS_BINS = 10  # 0.2 time units: a true maximum is the tallest smoothed max within +-10 bins
POWER_REL_PROMINENCES = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
POWER_WALKERS = (200_000, 1_000_000, 5_000_000)
POWER_DESIGNS = ((2, 0.1), (3, 0.1))
POWER_REPLICAS = 2_000
SEED = core.CAMPAIGN_SEED
DT_MF = mfb.DT_MF
T_MAX = base.DEFAULT_TMAX
G_CHUNK = mfb.G_CHUNK
BANDWIDTH = base.DEFAULT_BANDWIDTH
SIGMA_FACTOR = base.PROMINENCE_SIGMA_FACTOR
RELATIVE_FLOOR = base.PROMINENCE_RELATIVE_FLOOR
WINDOW = base.WINDOW
BIN_WIDTH = base.WINDOW_BIN

PRODUCTION_WALKERS = 5_000_000
GRID_WALKERS = w1.GRID_WALKERS  # 1e6
REFINE_WALKERS = w1.REFINE_WALKERS  # 3e6
JITTER_WALKERS = 200_000  # W3 replica size (exact_m_prr_upgrade_w3.REPLICA_WALKERS)

Z_HIST_EDGES = np.round(np.arange(0.0, 7.0 + 1e-9, 0.1), 10)
BULK_FRACTION = 0.01  # "bulk" bins: expected count >= 1% of the largest expected count
# Composite-trapezoid bin integrals versus the exact survival difference of the
# full f1 (O(dt_mf^2); observed <= 2.2e-6 at eps = 0.05, B = 4 -- four orders
# below the Poisson noise of any bin).  The per-reference value is recorded.
QUADRATURE_TOL = 1e-5

# Phase-grid family: identical to the W1 grid.
EPS_GRID = tuple(w1.EPS_GRID)
B_GRID = tuple(w1.B_GRID)
M_WEIGHTS = {2: (0.5, 0.5), 3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)}


# ----------------------------------------------------------------------------
# Mean-field component clock on the aligned fine grid.
# ----------------------------------------------------------------------------


class ComponentClock:
    """G_j(t), G(t) and Lambda(t) for explicit slab centres on the fine grid.

    G is evaluated by ``core.free_exposure_general`` (the campaign's d = 2 free
    clock: exact contact quadrature, matched-law mixture); the per-slab
    components G_j = prefactor * c(t) * exp(-(x(t) - x_j)^2 / 2 sigma^2) are
    reconstructed from the same contact factor and checked against G.
    """

    def __init__(self, centres_z, eps: float, weights, dt: float = DT_MF, p=core.MODEL):
        steps = int(round(T_MAX / dt))
        if abs(steps * dt - T_MAX) > 1e-12:
            raise ValueError("tmax must be an integer multiple of dt")
        self.centres_z = np.asarray(centres_z, dtype=float)
        self.eps = float(eps)
        self.weights = np.asarray(weights, dtype=float)
        self.dt = dt
        self.p = p
        self.t = np.arange(steps + 1) * dt
        m = self.centres_z.size
        g = np.empty_like(self.t)
        contact = np.empty_like(self.t)
        for start in range(0, self.t.size, G_CHUNK):
            block = core.free_exposure_general(
                self.t[start : start + G_CHUNK],
                centres_z=self.centres_z,
                eps=self.eps,
                weights=tuple(self.weights),
                p=p,
                n_perp=1,
            )
            g[start : start + G_CHUNK] = block["g"]
            contact[start : start + G_CHUNK] = block["contact_factor"]
            sigma = float(block["sigma"])
            centres_x = np.asarray(block["centres_x"], dtype=float)
        self.sigma = sigma
        self.centres_x = centres_x
        prefactor = 1.0 / (
            p.torus_w * math.sqrt(2.0 * math.pi) * self.eps
            * math.sqrt(p.d0 / (2.0 * p.gamma) + p.rho**2)
        )
        x_vals = np.asarray(base.x_of_t(self.t, p), dtype=float)
        self.g_components = np.empty((m, self.t.size))
        for j in range(m):
            self.g_components[j] = prefactor * contact * np.exp(
                -((x_vals - centres_x[j]) ** 2) / (2.0 * sigma**2)
            )
        # einsum (not BLAS matmul): Accelerate raises spurious FP flags on the
        # subnormal Gaussian tails.
        recon = np.einsum("j,jt->t", self.weights, self.g_components)
        scale = float(np.max(g))
        self.component_reconstruction_max_abs_error = float(
            np.max(np.abs(recon - g)) / scale
        )
        if not (self.component_reconstruction_max_abs_error <= 1e-12):
            raise AssertionError("component reconstruction of G failed")
        self.g = g
        self.contact = contact
        lam = np.zeros_like(g)
        lam[1:] = np.cumsum(0.5 * dt * (g[1:] + g[:-1]))
        self.lam = lam
        self.window_edges = np.arange(
            WINDOW[0], WINDOW[1] + 0.5 * BIN_WIDTH, BIN_WIDTH
        )
        idx = np.rint(self.window_edges / dt).astype(int)
        if np.max(np.abs(idx * dt - self.window_edges)) > 1e-9:
            raise ValueError("window bin edges are not aligned with the time grid")
        self.edge_index = idx
        centres_t = 0.5 * (self.window_edges[:-1] + self.window_edges[1:])
        cidx = np.rint(centres_t / dt).astype(int)
        if np.max(np.abs(cidx * dt - centres_t)) > 1e-9:
            raise ValueError("window bin centres are not aligned with the time grid")
        self.centre_index = cidx

    # -- densities ---------------------------------------------------------
    def component_density(self, j: int, budget: float) -> np.ndarray:
        return budget * self.weights[j] * self.g_components[j] * np.exp(-budget * self.lam)

    def full_density(self, budget: float) -> np.ndarray:
        return budget * self.g * np.exp(-budget * self.lam)

    def _bin_integrals(self, f: np.ndarray, walkers: float) -> np.ndarray:
        cum = np.zeros_like(f)
        cum[1:] = np.cumsum(0.5 * self.dt * (f[1:] + f[:-1]))
        c = cum[self.edge_index]
        return walkers * (c[1:] - c[:-1])

    def expected_counts_component(self, j: int, budget: float, walkers: float) -> np.ndarray:
        return self._bin_integrals(self.component_density(j, budget), walkers)

    def expected_counts_full_quadrature(self, budget: float, walkers: float) -> np.ndarray:
        return self._bin_integrals(self.full_density(budget), walkers)

    def expected_counts_full_exact(self, budget: float, walkers: float) -> np.ndarray:
        survival = np.exp(-budget * self.lam[self.edge_index])
        return walkers * (survival[:-1] - survival[1:])

    def dominant_component(self, budget: float) -> int:
        lo, hi = self.edge_index[0], self.edge_index[-1]
        peaks = [
            float(np.max(self.component_density(j, budget)[lo : hi + 1]))
            for j in range(self.centres_z.size)
        ]
        return int(np.argmax(peaks))

    def dominant_share_at_centres(self, j: int) -> np.ndarray:
        """w_j G_j / G at the window bin centres (survival factor cancels)."""
        idx = self.centre_index
        return self.weights[j] * self.g_components[j][idx] / self.g[idx]


_CLOCKS: dict[tuple, ComponentClock] = {}


def clock_for(centres_z, eps: float, weights) -> ComponentClock:
    key = (
        tuple(round(float(c), 12) for c in centres_z),
        round(float(eps), 12),
        tuple(round(float(w), 12) for w in weights),
    )
    if key not in _CLOCKS:
        _CLOCKS[key] = ComponentClock(centres_z, eps, weights)
    return _CLOCKS[key]


# ----------------------------------------------------------------------------
# Reference definitions.
# ----------------------------------------------------------------------------


def _interior_strict_maxima(values: np.ndarray) -> list[int]:
    v = np.asarray(values, dtype=float)
    return [
        i for i in range(1, v.size - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]
    ]


def _noise_free_classifier(lam: np.ndarray, edges: np.ndarray, walkers: float) -> dict:
    res = mfb.classify_real(lam, edges, walkers)
    rows = res["rows"]
    top = max(rows, key=lambda r: r["smoothed_height"]) if rows else None
    return {
        "n_smoothed_local_maxima": len(rows),
        "mode_count_covariance_aware": int(res["mode_count_covariance_aware"]),
        "true_mode_time": None if top is None else float(top["time"]),
        "true_mode_z_covariance_aware": (
            None if top is None else float(top["z_covariance_aware"])
        ),
    }


def _finish_reference(ref: dict, lam: np.ndarray, edges: np.ndarray) -> dict:
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0.0):
        raise AssertionError(f"negative expected count in {ref['name']}")
    maxima = _interior_strict_maxima(lam)
    argmax = int(np.argmax(lam))
    true_modes = int(ref.get("true_modes", 1))
    if true_modes == 1:
        unimodal = len(maxima) == 1 and maxima[0] == argmax
        if not unimodal:
            raise AssertionError(
                f"reference {ref['name']} is not unimodal at bin level: maxima bins {maxima}"
            )
    elif true_modes == 0:
        if maxima:
            raise AssertionError(f"flat reference {ref['name']} has interior maxima {maxima}")
        unimodal = False
    else:  # multi-mode (shoulder) reference: the true maxima are the bin-level maxima
        if [int(i) for i in maxima] != [int(i) for i in ref["true_mode_bins"]]:
            raise AssertionError(f"shoulder reference {ref['name']}: maxima {maxima} != declared")
        unimodal = False
    centres = 0.5 * (edges[:-1] + edges[1:])
    bulk = lam >= BULK_FRACTION * lam.max()
    ref.update(
        {
            "true_modes": true_modes,
            "expected_counts": [float(v) for v in lam],
            "expected_window_events": float(lam.sum()),
            "expected_window_event_fraction": float(lam.sum() / ref["walkers"]),
            "true_mode_bin": argmax if true_modes == 1 else None,
            "true_mode_time": float(centres[argmax]) if true_modes == 1 else None,
            "true_mode_bins": (
                [int(i) for i in ref["true_mode_bins"]] if true_modes >= 2
                else ([argmax] if true_modes == 1 else [])
            ),
            "bin_level_interior_maxima": [int(i) for i in maxima],
            "unimodal_at_bin_level": bool(unimodal),
            "bulk_bins": [int(i) for i in np.flatnonzero(bulk)],
            "bulk_time_range": [float(centres[bulk][0]), float(centres[bulk][-1])],
            "noise_free_classifier": _noise_free_classifier(lam, edges, ref["walkers"]),
        }
    )
    return ref


def reference_flat_plateau(lambda_per_bin: float, walkers: int) -> dict:
    """Adversarial look-elsewhere stress: constant expected count on every bin.

    Not a model density; every smoothed local maximum is spurious (true_modes
    = 0), so the largest z per histogram is the post-selected maximum of the
    prominence statistic over the entire 150-bin window at the given count
    level.  The relative floor is measured against the (noise) global maximum.
    """

    edges = np.arange(WINDOW[0], WINDOW[1] + 0.5 * BIN_WIDTH, BIN_WIDTH)
    lam = np.full(edges.size - 1, float(lambda_per_bin))
    ref = {
        "name": f"D_flat_plateau_lambda{lambda_per_bin:g}_N{walkers:.0e}",
        "family": "flat_plateau_stress",
        "stress": True,
        "true_modes": 0,
        "construction": (
            "constant expected count lambda per bin on all 150 window bins (no true mode); "
            "the walker count only sets the density scale, z is invariant to it"
        ),
        "lambda_per_bin": float(lambda_per_bin),
        "walkers": int(walkers),
        "sampling": "poisson",
    }
    return _finish_reference(ref, lam, edges)


def reference_single_slab(target_time: float, eps: float, budget: float, walkers: int) -> dict:
    centres_z = core.centres_z_for(1, target_times=(target_time,))
    clock = clock_for(centres_z, eps, (1.0,))
    lam_q = clock.expected_counts_component(0, budget, walkers)
    lam_e = clock.expected_counts_full_exact(budget, walkers)
    quad_err = float(np.max(np.abs(lam_q - lam_e)) / np.max(lam_e))
    if not (quad_err <= QUADRATURE_TOL):
        raise AssertionError("single-slab quadrature disagrees with exact survival difference")
    ref = {
        "name": f"A_single_slab_t{target_time:g}_eps{eps:g}_B{budget:g}_N{walkers:.0e}",
        "family": "mf_single_slab",
        "construction": (
            "mean-field law f1 = B G exp(-B Lambda) for one slab (m = 1) centred at "
            "mu(t_j) = core.centres_z_for(1, target_times=(t_j,)); G by "
            "core.free_exposure_general (d = 2), Lambda = int_0^t G (trapezoid, dt_mf)"
        ),
        "m": 1,
        "target_time": float(target_time),
        "centres_z": [float(c) for c in centres_z],
        "eps": float(eps),
        "budget": float(budget),
        "weights": [1.0],
        "dominant_component": 0,
        "walkers": int(walkers),
        "sampling": "poisson",
        "quadrature_vs_exact_survival_max_rel_error": quad_err,
        "component_reconstruction_max_rel_error": clock.component_reconstruction_max_abs_error,
    }
    return _finish_reference(ref, lam_q, clock.window_edges)


def reference_dominant_component(
    m: int, eps: float, budget: float, weights, walkers: int, *, sampling: str = "poisson",
    name_prefix: str = "B_dominant",
) -> dict:
    centres_z = core.centres_z_for(m)
    clock = clock_for(centres_z, eps, tuple(weights))
    j = clock.dominant_component(budget)
    lam_dom = clock.expected_counts_component(j, budget, walkers)
    lam_sum = sum(
        clock.expected_counts_component(k, budget, walkers) for k in range(m)
    )
    lam_exact = clock.expected_counts_full_exact(budget, walkers)
    quad_err = float(np.max(np.abs(lam_sum - lam_exact)) / np.max(lam_exact))
    if not (quad_err <= QUADRATURE_TOL):
        raise AssertionError("component quadrature sum disagrees with exact survival difference")
    wtag = "-".join(f"{100 * w:.0f}" for w in weights)
    ref = {
        "name": f"{name_prefix}_m{m}_eps{eps:g}_B{budget:g}_w{wtag}_N{walkers:.0e}"
        + ("" if sampling == "poisson" else f"_{sampling}"),
        "family": "mf_dominant_component",
        "construction": (
            "mean-field law is linear in the slab components given Lambda: "
            "f1 = sum_j f1^(j), f1^(j) = B w_j G_j exp(-B Lambda), Lambda from the full "
            "mixture G; only the dominant component (largest window maximum) is kept; "
            "expected counts by composite trapezoid on the dt_mf grid"
        ),
        "m": int(m),
        "target_times": [float(t) for t in base.TARGET_TIMES[m]],
        "centres_z": [float(c) for c in centres_z],
        "eps": float(eps),
        "budget": float(budget),
        "weights": [float(w) for w in weights],
        "dominant_component": int(j),
        "dominant_target_time": float(base.TARGET_TIMES[m][j]),
        "removed_components_expected_events": float(lam_sum.sum() - lam_dom.sum()),
        "walkers": int(walkers),
        "sampling": sampling,
        "quadrature_vs_exact_survival_max_rel_error": quad_err,
        "component_reconstruction_max_rel_error": clock.component_reconstruction_max_abs_error,
    }
    return _finish_reference(ref, lam_dom, clock.window_edges)


def reference_empirical_reweighted(production_file: str) -> dict:
    payload = json.loads((PRODUCTION_DIR / production_file).read_text(encoding="utf-8"))
    cfg = payload["parameters"]["config"]
    m, eps, budget = int(cfg["m"]), float(cfg["eps"]), float(cfg["budget"])
    weights = tuple(float(w) for w in cfg["weights"])
    walkers = int(cfg["walkers"])
    classifier = payload["results"]["classifier"]
    edges = np.asarray(classifier["edges"], dtype=float)
    smoothed = np.asarray(classifier["smoothed_density"], dtype=float)
    if abs(float(classifier["bandwidth"]) - BANDWIDTH) > 1e-12:
        raise AssertionError("production record bandwidth differs from the calibration bandwidth")
    centres_z = core.centres_z_for(m)
    clock = clock_for(centres_z, eps, weights)
    if np.max(np.abs(clock.window_edges - edges)) > 1e-9:
        raise AssertionError("production edges do not match the window grid")
    j = clock.dominant_component(budget)
    share = clock.dominant_share_at_centres(j)
    lam = walkers * smoothed * share * BIN_WIDTH
    wtag = "-".join(f"{100 * w:.0f}" for w in weights)
    ref = {
        "name": f"C_empirical_m{m}_eps{eps:g}_B{budget:g}_w{wtag}_N{walkers:.0e}",
        "family": "empirical_reweighted",
        "construction": (
            "stored production smoothed density s_i (bandwidth 0.04) multiplied by the "
            "mean-field share of the dominant component at the bin centre, "
            "w_d G_d(t_i) / G(t_i); lambda_i = N s_i share_i delta"
        ),
        "production_file": production_file,
        "m": m,
        "target_times": [float(t) for t in base.TARGET_TIMES[m]],
        "eps": eps,
        "budget": budget,
        "weights": list(weights),
        "dominant_component": int(j),
        "dominant_target_time": float(base.TARGET_TIMES[m][j]),
        "share_min": float(share.min()),
        "share_max": float(share.max()),
        "stored_window_events": int(sum(classifier["counts"])),
        "walkers": walkers,
        "sampling": "poisson",
        "component_reconstruction_max_rel_error": clock.component_reconstruction_max_abs_error,
    }
    return _finish_reference(ref, lam, edges)


def reference_full_shoulder(m: int, eps: float, budget: float, walkers: int) -> dict | None:
    """FULL mean-field f1 (all slabs) in a grid cell where the expected bin
    counts carry fewer than m interior maxima (the lost late mode is a
    shoulder / has merged).  The null is 'no more than the true k modes';
    a false positive is an over-count or a significant unmatched maximum.
    Returns None when the cell still has m bin-level maxima."""
    centres_z = core.centres_z_for(m)
    clock = clock_for(centres_z, eps, M_WEIGHTS[m])
    lam = clock.expected_counts_full_exact(budget, walkers)
    maxima = _interior_strict_maxima(lam)
    if len(maxima) >= m:
        return None
    lam_q = clock.expected_counts_full_quadrature(budget, walkers)
    quad_err = float(np.max(np.abs(lam_q - lam)) / np.max(lam))
    ref = {
        "name": f"S_shoulder_m{m}_eps{eps:g}_B{budget:g}_N{walkers:.0e}",
        "family": "mf_full_shoulder",
        "construction": (
            "full mean-field law f1 = B G exp(-B Lambda) (all slabs), expected counts by the "
            "exact survival difference; the cell has fewer than m bin-level maxima (lost late "
            "mode = shoulder); true modes = the bin-level maxima"
        ),
        "m": int(m),
        "target_times": [float(t) for t in base.TARGET_TIMES[m]],
        "eps": float(eps),
        "budget": float(budget),
        "weights": list(M_WEIGHTS[m]),
        "walkers": int(walkers),
        "sampling": "poisson",
        "true_modes": len(maxima),
        "true_mode_bins": [int(i) for i in maxima],
        "lost_modes": int(m - len(maxima)),
        "quadrature_vs_exact_survival_max_rel_error": quad_err,
        "dominant_target_time": None,
    }
    return _finish_reference(ref, lam, clock.window_edges)


def build_references(pilot: bool) -> tuple[list[dict], dict[str, list[dict]]]:
    """Per-histogram references and the two 48-cell phase-grid families.

    Returns (references, stress, families, shoulder).  The per-histogram
    references and the family cells keep their 2026-09-09 order so that the
    stored pilot seeds (ref_index) are reproduced bit-for-bit; the flat-plateau
    stress references are indexed after the family cells, and the shoulder
    family is seeded on its own tag (72)."""

    single = []
    for tj in base.TARGET_TIMES[2]:
        for walkers in (JITTER_WALKERS, GRID_WALKERS, PRODUCTION_WALKERS):
            if tj == base.TARGET_TIMES[2][1] and walkers == JITTER_WALKERS:
                continue
            single.append(reference_single_slab(tj, 0.1, 1.0, walkers))
    dominant = [
        reference_dominant_component(2, 0.1, 1.0, M_WEIGHTS[2], JITTER_WALKERS),
        reference_dominant_component(2, 0.1, 1.0, M_WEIGHTS[2], GRID_WALKERS),
        reference_dominant_component(2, 0.1, 1.0, M_WEIGHTS[2], REFINE_WALKERS),
        reference_dominant_component(2, 0.1, 1.0, M_WEIGHTS[2], PRODUCTION_WALKERS),
        reference_dominant_component(3, 0.1, 1.0, M_WEIGHTS[3], GRID_WALKERS),
        reference_dominant_component(3, 0.1, 1.0, M_WEIGHTS[3], PRODUCTION_WALKERS),
        reference_dominant_component(3, 0.2, 1.0, M_WEIGHTS[3], GRID_WALKERS),
        reference_dominant_component(3, 0.2, 0.5, M_WEIGHTS[3], REFINE_WALKERS),
        reference_dominant_component(
            2, 0.1, 1.0, M_WEIGHTS[2], PRODUCTION_WALKERS, sampling="multinomial"
        ),
    ]
    empirical = [
        reference_empirical_reweighted("m2_eps0.1_B1_w50-50.json"),
        reference_empirical_reweighted("m3_eps0.1_B1_w33-33-33.json"),
    ]
    stress = [
        reference_flat_plateau(5000.0, GRID_WALKERS),  # bulk-level counts at N = 1e6
        reference_flat_plateau(50.0, GRID_WALKERS),  # tail-level counts
        reference_flat_plateau(5.0, GRID_WALKERS),  # sparse counts
    ]
    references = single + dominant + empirical

    families: dict[str, list[dict]] = {}
    for m in (2, 3):
        cells = []
        for eps in EPS_GRID:
            for budget in B_GRID:
                cells.append(
                    reference_dominant_component(
                        m, eps, budget, M_WEIGHTS[m], GRID_WALKERS,
                        name_prefix="cell_dominant",
                    )
                )
        families[f"m{m}_grid_48"] = cells
    shoulder = []
    for m in (2, 3):
        for eps in EPS_GRID:
            for budget in B_GRID:
                ref = reference_full_shoulder(m, eps, budget, GRID_WALKERS)
                if ref is not None:
                    shoulder.append(ref)
    return references, stress, families, shoulder


# ----------------------------------------------------------------------------
# Replica sampling and classification (worker).
# ----------------------------------------------------------------------------

FIELDS = (
    "n_true_matched",
    "n_local_maxima",
    "mode_count",
    "true_mode_z",
    "true_mode_rel",
    "max_spurious_z_cov",
    "max_spurious_z_peak_only",
    "max_spurious_rel",
    "max_spurious_z_cov_floor_ok",
    "t_max_spurious",
    "n_spurious_floor_ok",
    "gate_fp",
    "full_fp",
)


def _match_true_rows(rows: list[dict], true_bins) -> list[int]:
    """Row indices matched to the true maxima: for each true bin (in the given
    order) the tallest not-yet-matched smoothed maximum within
    MATCH_RADIUS_BINS; unmatched true maxima are skipped."""
    assigned: list[int] = []
    for b in true_bins:
        cands = [
            q for q, r in enumerate(rows)
            if q not in assigned and abs(int(r["bin"]) - int(b)) <= MATCH_RADIUS_BINS
        ]
        if cands:
            assigned.append(max(cands, key=lambda q: rows[q]["smoothed_height"]))
    return assigned


def _classify_replica(
    counts: np.ndarray, edges: np.ndarray, walkers: int, true_modes: int = 1,
    true_bins=None,
) -> tuple:
    res = recl.classify_both(
        counts, edges, walkers,
        bandwidth=BANDWIDTH, sigma_factor=SIGMA_FACTOR, relative_floor=RELATIVE_FLOOR,
    )
    rows = res["rows"]
    if not rows:
        return (0, 0, 0, math.nan, math.nan, -1.0, -1.0, -1.0, -1.0, math.nan, 0, False, False)
    if true_modes == 1:
        top = max(range(len(rows)), key=lambda k: rows[k]["smoothed_height"])
        true_row = rows[top]
        spurious = [r for k, r in enumerate(rows) if k != top]
        n_matched = 1
    elif true_modes == 0:  # flat stress reference: every local maximum is spurious
        true_row = None
        spurious = list(rows)
        n_matched = 0
    else:  # shoulder reference: match each true maximum, the rest are spurious
        matched = _match_true_rows(rows, true_bins)
        n_matched = len(matched)
        true_row = min((rows[q] for q in matched), key=lambda r: r["z_covariance_aware"]) if matched else None
        spurious = [r for q, r in enumerate(rows) if q not in matched]
    if spurious:
        best = max(spurious, key=lambda r: r["z_covariance_aware"])
        mz = float(best["z_covariance_aware"])
        mz_peak = float(max(r["z_peak_only"] for r in spurious))
        mrel = float(max(r["relative_prominence"] for r in spurious))
        floor_rows = [r for r in spurious if r["prominence"] >= RELATIVE_FLOOR * res["global_max"]]
        mz_floor = float(max(r["z_covariance_aware"] for r in floor_rows)) if floor_rows else -1.0
        t_best = float(best["time"])
        n_floor = len(floor_rows)
        full_fp = any(r["significant_covariance_aware"] for r in spurious)
    else:
        mz, mz_peak, mrel, mz_floor, t_best, n_floor, full_fp = -1.0, -1.0, -1.0, -1.0, math.nan, 0, False
    return (
        n_matched,
        len(rows),
        int(res["mode_count_covariance_aware"]),
        math.nan if true_row is None else float(true_row["z_covariance_aware"]),
        math.nan if true_row is None else float(true_row["relative_prominence"]),
        mz,
        mz_peak,
        mrel,
        mz_floor,
        t_best,
        n_floor,
        bool(mz >= SIGMA_FACTOR),
        bool(full_fp),
    )


def run_chunk(task: dict) -> dict:
    """Sample and classify one chunk of replicas of one reference."""

    lam = np.asarray(task["expected_counts"], dtype=float)
    edges = np.asarray(task["edges"], dtype=float)
    walkers = int(task["walkers"])
    n_rep = int(task["n_replicas"])
    rng = np.random.Generator(
        np.random.Philox(
            np.random.SeedSequence([
                SEED, int(task.get("tag", TAG_D2_NULL)),
                int(task.get("seed_index", task["ref_index"])), int(task["chunk_index"]),
            ])
        )
    )
    if task["sampling"] == "multinomial":
        p = lam / walkers
        p_full = np.append(p, max(0.0, 1.0 - p.sum()))
    out = {f: [] for f in FIELDS}
    t0 = time.perf_counter()
    for _ in range(n_rep):
        if task["sampling"] == "poisson":
            counts = rng.poisson(lam)
        elif task["sampling"] == "multinomial":
            counts = rng.multinomial(walkers, p_full)[:-1]
        else:
            raise ValueError(f"unknown sampling {task['sampling']}")
        values = _classify_replica(
            counts, edges, walkers, int(task.get("true_modes", 1)), task.get("true_bins")
        )
        for f, v in zip(FIELDS, values):
            out[f].append(v)
    return {
        "ref_index": task["ref_index"],
        "chunk_index": task["chunk_index"],
        "start": task["start"],
        "n_replicas": n_rep,
        "seconds": time.perf_counter() - t0,
        "arrays": {f: np.asarray(v) for f, v in out.items()},
    }


# ----------------------------------------------------------------------------
# Aggregation.
# ----------------------------------------------------------------------------


def _rate(k: int, n: int) -> dict:
    lo, hi = core.wilson_ci(int(k), int(n))
    return {"count": int(k), "trials": int(n), "rate": (k / n if n else None),
            "wilson95": [float(lo), float(hi)]}


def _z_distribution(z: np.ndarray) -> dict:
    present = z[z >= 0.0]
    if present.size == 0:
        return {"n_with_spurious_maximum": 0}
    hist, _ = np.histogram(present, bins=Z_HIST_EDGES)
    q = np.quantile(present, [0.5, 0.9, 0.99, 0.999])
    return {
        "n_with_spurious_maximum": int(present.size),
        "max": float(present.max()),
        "mean": float(present.mean()),
        "quantiles": {"q50": float(q[0]), "q90": float(q[1]), "q99": float(q[2]), "q999": float(q[3])},
        "n_ge_3": int(np.sum(present >= 3.0)),
        "n_ge_4": int(np.sum(present >= 4.0)),
        "n_ge_4p5": int(np.sum(present >= 4.5)),
        "n_ge_5": int(np.sum(present >= SIGMA_FACTOR)),
        "histogram_edges": [float(e) for e in Z_HIST_EDGES],
        "histogram_counts": [int(c) for c in hist],
        "histogram_overflow_ge_7": int(np.sum(present >= Z_HIST_EDGES[-1])),
    }


def aggregate_reference(ref: dict, arrays: dict) -> dict:
    n = int(arrays["mode_count"].size)
    true_modes = int(ref.get("true_modes", 1))
    z = arrays["max_spurious_z_cov"]
    zp = arrays["max_spurious_z_peak_only"]
    tz = arrays["t_max_spurious"]
    bulk_lo, bulk_hi = ref["bulk_time_range"]
    present = z >= 0.0
    in_bulk = present & (tz >= bulk_lo - 1e-9) & (tz <= bulk_hi + 1e-9)
    imax = int(np.argmax(z)) if present.any() else None
    true_z = arrays["true_mode_z"]
    finite_true = true_z[np.isfinite(true_z)]
    return {
        "name": ref["name"],
        "family": ref.get("family"),
        "stress": bool(ref.get("stress", False)),
        "walkers": int(ref["walkers"]),
        "replicas": n,
        "gate_only_false_positive": _rate(int(arrays["gate_fp"].sum()), n),
        "full_rule_false_positive": _rate(int(arrays["full_fp"].sum()), n),
        "floor_only_spurious": _rate(int(np.sum(arrays["n_spurious_floor_ok"] > 0)), n),
        "true_modes": true_modes,
        "mode_count_mismatch": _rate(int(np.sum(arrays["mode_count"] != true_modes)), n),
        "over_count": _rate(int(np.sum(arrays["mode_count"] > true_modes)), n),
        "under_count": _rate(int(np.sum(arrays["mode_count"] < true_modes)), n),
        "all_true_maxima_matched": _rate(int(np.sum(arrays["n_true_matched"] == true_modes)), n),
        "unmatched_significant_without_over_count": _rate(
            int(np.sum(arrays["full_fp"] & (arrays["mode_count"] <= true_modes))), n
        ),
        "true_mode_counted": (
            _rate(int(np.sum(arrays["mode_count"] >= 1)), n) if true_modes == 1 else None
        ),
        "mode_count_values": {
            str(int(v)): int(c) for v, c in zip(*np.unique(arrays["mode_count"], return_counts=True))
        },
        "mean_local_maxima": float(arrays["n_local_maxima"].mean()),
        "max_local_maxima": int(arrays["n_local_maxima"].max()),
        "fraction_with_spurious_maximum": float(present.mean()),
        "fraction_argmax_spurious_in_bulk": float(in_bulk.sum() / max(1, present.sum())),
        "true_mode_z": {
            "min": float(finite_true.min()) if finite_true.size else None,
            "median": float(np.median(finite_true)) if finite_true.size else None,
        },
        "true_mode_relative_prominence_min": (
            float(np.nanmin(arrays["true_mode_rel"])) if finite_true.size else None
        ),
        "max_spurious_z_covariance_aware": _z_distribution(z),
        "max_spurious_z_peak_only": _z_distribution(zp),
        "max_spurious_relative_prominence": float(arrays["max_spurious_rel"].max()),
        "max_spurious_z_among_floor_passing": float(arrays["max_spurious_z_cov_floor_ok"].max()),
        "extreme_replica": None if imax is None else {
            "max_spurious_z_cov": float(z[imax]),
            "max_spurious_z_peak_only_same_replica": float(zp[imax]),
            "time": float(tz[imax]),
            "mode_count": int(arrays["mode_count"][imax]),
        },
    }


def aggregate_family(name: str, cells: list[dict], cell_arrays: list[dict], n_families: int) -> dict:
    """Family-wise rates: family f draws histogram f in every cell."""
    z_cells = np.stack([a["max_spurious_z_cov"][:n_families] for a in cell_arrays])
    gate = np.stack([a["gate_fp"][:n_families] for a in cell_arrays]).any(axis=0)
    full = np.stack([a["full_fp"][:n_families] for a in cell_arrays]).any(axis=0)
    mismatch = np.stack([
        a["mode_count"][:n_families] != int(c.get("true_modes", 1)) for c, a in zip(cells, cell_arrays)
    ]).any(axis=0)
    over = np.stack([
        a["mode_count"][:n_families] > int(c.get("true_modes", 1)) for c, a in zip(cells, cell_arrays)
    ]).any(axis=0)
    fam_max = z_cells.max(axis=0)
    per_cell = []
    for c, a in zip(cells, cell_arrays):
        k_true = int(c.get("true_modes", 1))
        mc = a["mode_count"][:n_families]
        per_cell.append(
            {
                "cell": c["name"],
                "m": c["m"],
                "eps": c["eps"],
                "budget": c["budget"],
                "true_modes": k_true,
                "dominant_target_time": c.get("dominant_target_time"),
                "max_spurious_z_cov": float(a["max_spurious_z_cov"][:n_families].max()),
                "gate_fp_count": int(a["gate_fp"][:n_families].sum()),
                "full_fp_count": int(a["full_fp"][:n_families].sum()),
                "gate_only_false_positive": _rate(int(a["gate_fp"][:n_families].sum()), n_families),
                "full_rule_false_positive": _rate(int(a["full_fp"][:n_families].sum()), n_families),
                "over_count": _rate(int(np.sum(mc > k_true)), n_families),
                "under_count": _rate(int(np.sum(mc < k_true)), n_families),
                "min_true_mode_z": (
                    float(np.nanmin(a["true_mode_z"][:n_families]))
                    if np.isfinite(a["true_mode_z"][:n_families]).any() else None
                ),
            }
        )
    wilson_uppers = [p["gate_only_false_positive"]["wilson95"][1] for p in per_cell]
    return {
        "name": name,
        "n_cells": len(cells),
        "families": int(n_families),
        "histograms_total": int(n_families * len(cells)),
        "family_wise_gate_only_false_positive": _rate(int(gate.sum()), n_families),
        "family_wise_full_rule_false_positive": _rate(int(full.sum()), n_families),
        "family_wise_mode_count_mismatch": _rate(int(mismatch.sum()), n_families),
        "family_wise_over_count": _rate(int(over.sum()), n_families),
        "family_max_spurious_z_cov": _z_distribution(fam_max),
        "pooled_cell_max_spurious_z_cov": _z_distribution(z_cells.ravel()),
        "pooled_cell_histograms_gate_only_false_positive": _rate(
            int(sum(p["gate_fp_count"] for p in per_cell)), n_families * len(cells)
        ),
        "pooled_cell_histograms_full_rule_false_positive": _rate(
            int(sum(p["full_fp_count"] for p in per_cell)), n_families * len(cells)
        ),
        "per_cell_wilson95_upper_gate_only": {
            "max_over_cells": float(max(wilson_uppers)),
            "union_bound_sum_over_cells": float(min(1.0, sum(wilson_uppers))),
            "note": "per-cell Wilson 95% upper limits (n = families per cell); the sum is a "
                    "Bonferroni-style family-wise upper bound that needs no independence",
        },
        "per_cell": per_cell,
        "_family_flags": {"gate": gate, "full": full, "not_one": mismatch, "over": over, "max": fam_max},
    }


# ----------------------------------------------------------------------------
# Figure (drawn from summary.json only).
# ----------------------------------------------------------------------------


def _nlabel(n: int) -> str:
    e = int(math.floor(math.log10(n)))
    mant = n / 10**e
    return rf"$N={mant:g}\times10^{{{e}}}$" if mant != 1 else rf"$N=10^{{{e}}}$"


def make_figure(summary: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt

    edges = np.asarray(Z_HIST_EDGES)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), constrained_layout=True, sharey=False)
    ax_a, ax_b, ax_c = axes

    def _headroom(ax):
        lo, hi = ax.get_ylim()
        ax.set_ylim(0.7, hi * 60.0)

    def _gate(ax):
        ax.axvline(SIGMA_FACTOR, color="0.2", lw=0.9, ls="-.")
        ax.text(SIGMA_FACTOR + 0.1, 0.03, r"$5\sigma$ gate", transform=ax.get_xaxis_transform(),
                fontsize=6.4, ha="left", va="bottom", color="0.2", rotation=90)

    # (a) per-histogram tier: unimodal references only (no stress, no family cells).
    uni = [r for r in summary["reference_results"] if r["true_modes"] == 1 and not r["stress"]]
    by_n: dict[int, np.ndarray] = {}
    n_by_n: dict[int, int] = {}
    for r in uni:
        if r["name"].endswith("_multinomial"):
            continue
        d = r["max_spurious_z_covariance_aware"]
        if "histogram_counts" not in d:
            continue
        by_n.setdefault(r["walkers"], np.zeros(edges.size - 1))
        by_n[r["walkers"]] += np.asarray(d["histogram_counts"], dtype=float)
        n_by_n[r["walkers"]] = n_by_n.get(r["walkers"], 0) + r["replicas"]
    colours = {JITTER_WALKERS: core.OI_ORANGE, GRID_WALKERS: core.OI_BLUE,
               REFINE_WALKERS: core.OI_PURPLE, PRODUCTION_WALKERS: core.OI_VERMILLION}
    for n_w in sorted(by_n):
        ax_a.stairs(by_n[n_w], edges, color=colours.get(n_w, "0.4"), lw=1.1,
                    label=_nlabel(n_w) + f" ({n_by_n[n_w]:,})")
    hl = summary["headline"]["per_histogram_references_only"]
    _gate(ax_a)
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.0, 7.0)
    ax_a.set_ylim(0.7, None)
    _headroom(ax_a)
    ax_a.set_xlabel(r"largest spurious $z$ per histogram")
    ax_a.set_ylabel("null histograms")
    ax_a.legend(loc="upper right", fontsize=6.0, frameon=True, framealpha=1.0, edgecolor="none", handlelength=1.4,
                title=f"{hl['histograms']:,} hist.; max {hl['max_spurious_z_cov']:.2f}", title_fontsize=6.0)
    ax_a.set_title("(a) unimodal references", loc="left", fontsize=8.0)

    # (b) family-wise over the 48-cell grids and the shoulder family.
    fam_styles = {
        "m2_grid_48": (core.OI_BLUE, "-", r"$m=2$ grid (48)"),
        "m3_grid_48": (core.OI_VERMILLION, "--", r"$m=3$ grid (48)"),
        "union_96": (core.OI_GREEN, ":", "union (96)"),
        "shoulder_full_mf": (core.OI_PURPLE, "-", "shoulder cells"),
    }
    fam_list = list(summary["families"])  # shoulder: over-count metric, reported in the JSON, not a z histogram
    for fam in fam_list:
        colour, ls, label = fam_styles[fam["name"]]
        if fam["name"] == "shoulder_full_mf":
            label = f"shoulder cells ({fam['n_cells']})"
        d = fam["family_max_spurious_z_cov"]
        if "histogram_counts" not in d:
            continue
        ax_b.stairs(np.asarray(d["histogram_counts"], dtype=float), edges, color=colour, ls=ls, lw=1.1, label=label)
    _gate(ax_b)
    fam_n = summary["families"][0]["families"]
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, 7.0)
    ax_b.set_ylim(0.7, None)
    _headroom(ax_b)
    ax_b.set_xlabel(r"largest spurious $z$ over one phase diagram")
    ax_b.set_ylabel("synthetic phase diagrams")
    ax_b.legend(loc="upper right", fontsize=6.0, frameon=True, framealpha=1.0, edgecolor="none", handlelength=1.8,
                title=f"{fam_n:,} diagrams per family", title_fontsize=6.0)
    ax_b.set_title("(b) family-wise", loc="left", fontsize=8.0)

    # (c) flat-plateau stress (look-elsewhere): largest z over the whole window.
    st_cols = [core.OI_BLUE, core.OI_VERMILLION, core.OI_GREEN]
    for r, col in zip(summary["stress_flat_plateau"]["references"], st_cols):
        d = r["max_spurious_z_covariance_aware"]
        if "histogram_counts" not in d:
            continue
        lam = r["lambda_per_bin"]
        ax_c.stairs(np.asarray(d["histogram_counts"], dtype=float), edges, color=col, lw=1.1,
                    label=rf"$\lambda={lam:g}$/bin; max {d['max']:.2f}")
    _gate(ax_c)
    ax_c.set_yscale("log")
    ax_c.set_xlim(0.0, 7.0)
    ax_c.set_ylim(0.7, None)
    _headroom(ax_c)
    ax_c.set_xlabel(r"largest $z$ on a flat window")
    ax_c.set_ylabel("stress histograms")
    n_st = summary["stress_flat_plateau"]["references"][0]["replicas"]
    ax_c.legend(loc="upper right", fontsize=6.0, frameon=True, framealpha=1.0, edgecolor="none", handlelength=1.4,
                title=f"{n_st:,} per level", title_fontsize=6.0)
    ax_c.set_title("(c) flat-plateau stress", loc="left", fontsize=8.0)

    written = core.save_figure(fig, FIG_STEM)
    plt.close(fig)
    return written


def refs_result(summary: dict, name: str) -> dict:
    for r in summary["reference_results"]:
        if r["name"] == name:
            return r
    raise KeyError(name)


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------


def _chunks(n_total: int, chunk: int) -> list[tuple[int, int]]:
    out = []
    start = 0
    k = 0
    while start < n_total:
        n = min(chunk, n_total - start)
        out.append((k, start, n))
        start += n
        k += 1
    return [(k, s) for k, s, _ in out], [n for _, _, n in out]


def _alloc(n: int) -> dict:
    return {
        "n_true_matched": np.zeros(n, dtype=np.int32),
        "n_local_maxima": np.zeros(n, dtype=np.int32),
        "mode_count": np.zeros(n, dtype=np.int32),
        "true_mode_z": np.full(n, np.nan),
        "true_mode_rel": np.full(n, np.nan),
        "max_spurious_z_cov": np.full(n, -1.0),
        "max_spurious_z_peak_only": np.full(n, -1.0),
        "max_spurious_rel": np.full(n, -1.0),
        "max_spurious_z_cov_floor_ok": np.full(n, -1.0),
        "t_max_spurious": np.full(n, np.nan),
        "n_spurious_floor_ok": np.zeros(n, dtype=np.int32),
        "gate_fp": np.zeros(n, dtype=bool),
        "full_fp": np.zeros(n, dtype=bool),
    }


def _pool_run(tasks: list[dict], arrays: list[dict], workers: int, chunk: int) -> tuple[float, float]:
    done = 0
    cpu_seconds = 0.0
    total_hist = sum(t["n_replicas"] for t in tasks)
    t_run = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_chunk, task) for task in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            arr = arrays[res["ref_index"]]
            s, n = res["start"], res["n_replicas"]
            for f in FIELDS:
                arr[f][s : s + n] = res["arrays"][f]
            done += n
            cpu_seconds += res["seconds"]
            if done % (20 * chunk) < n or done == total_hist:
                print(f"  {done:,}/{total_hist:,} histograms, "
                      f"{time.perf_counter() - t_run:.0f}s wall", flush=True)
    return time.perf_counter() - t_run, cpu_seconds


def _stress_block(ref: dict, agg: dict, arr: dict) -> dict:
    z = arr["max_spurious_z_cov"]
    n = int(z.size)
    return {
        "name": ref["name"],
        "lambda_per_bin": ref["lambda_per_bin"],
        "replicas": n,
        "gate_only_z_ge_5": _rate(int(np.sum(z >= SIGMA_FACTOR)), n),
        "z_ge_4": _rate(int(np.sum(z >= 4.0)), n),
        "z_ge_4p5": _rate(int(np.sum(z >= 4.5)), n),
        "full_rule_any_mode_counted": _rate(int(np.sum(arr["mode_count"] > 0)), n),
        "max_spurious_z_covariance_aware": agg["max_spurious_z_covariance_aware"],
        "max_spurious_relative_prominence": agg["max_spurious_relative_prominence"],
        "max_spurious_z_among_floor_passing": agg["max_spurious_z_among_floor_passing"],
        "relative_prominence_of_replicas_with_z_ge_4": [
            float(v) for v in arr["max_spurious_rel"][z >= 4.0]
        ][:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicas", type=lambda v: int(float(v)), default=20_000,
                        help="null histograms per per-histogram / stress reference")
    parser.add_argument("--families", type=lambda v: int(float(v)), default=2_000,
                        help="synthetic phase diagrams per grid (= histograms per cell)")
    parser.add_argument("--chunk", type=int, default=2_500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pilot", action="store_true",
                        help="replicas=2000, families=100: reproduce the 2026-09-09 pilot and compare")
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figure(s) from the stored JSON")
    parser.add_argument("--power", action="store_true",
                        help="power curves only (detection probability vs relative prominence)")
    parser.add_argument("--power-replicas", type=lambda v: int(float(v)), default=POWER_REPLICAS)
    args = parser.parse_args()
    if args.workers <= 0 or args.workers > 3:
        raise SystemExit("workers must be in 1..3 (machine rule, 2026-09-23 campaign)")
    if args.figure_only:
        if OUT_JSON.exists():
            for path in make_figure(json.loads(OUT_JSON.read_text(encoding="utf-8"))):
                print("wrote", path)
        if POWER_JSON.exists():
            for path in make_power_figure(json.loads(POWER_JSON.read_text(encoding="utf-8"))):
                print("wrote", path)
        return
    if args.power:
        run_power(args)
        return
    if args.pilot:
        args.replicas, args.families = 2_000, 100

    t_start = time.perf_counter()
    references, stress, families, shoulder = build_references(args.pilot)
    print(f"[references] {len(references)} per-histogram references, {len(stress)} stress, "
          f"{sum(len(v) for v in families.values())} family cells, {len(shoulder)} shoulder cells "
          f"built in {time.perf_counter() - t_start:.1f}s", flush=True)
    for c in shoulder:
        print(f"  shoulder {c['name']}: true maxima {c['true_modes']} (m={c['m']}), "
              f"noise-free classifier {c['noise_free_classifier']['mode_count_covariance_aware']}", flush=True)

    # ref_index: references (0..15) and family cells (16..111) keep the pilot
    # order; stress next; shoulder last on its own tag with seed_index = cell index.
    all_refs: list[tuple[str, dict, int, int, int]] = []  # (tier, ref, n, tag, seed_index)
    for ref in references:
        all_refs.append(("reference", ref, args.replicas, TAG_D2_NULL, -1))
    for fname, cells in families.items():
        for ref in cells:
            all_refs.append((fname, ref, args.families, TAG_D2_NULL, -1))
    for ref in stress:
        all_refs.append(("stress", ref, args.replicas, TAG_D2_NULL, -1))
    for k, ref in enumerate(shoulder):
        all_refs.append(("shoulder", ref, args.families, TAG_SHOULDER, k))
    edges = np.arange(WINDOW[0], WINDOW[1] + 0.5 * BIN_WIDTH, BIN_WIDTH)
    tasks = []
    arrays = []
    for ref_index, (_tier, ref, n_rep, tag, seed_index) in enumerate(all_refs):
        ref["ref_index"] = ref_index
        ref["seed_tag"] = tag
        ref["seed_index"] = ref_index if seed_index < 0 else seed_index
        arrays.append(_alloc(n_rep))
        (keys, sizes) = _chunks(n_rep, args.chunk)
        for (chunk_index, start), n in zip(keys, sizes):
            tasks.append({
                "ref_index": ref_index,
                "seed_index": ref["seed_index"],
                "tag": tag,
                "chunk_index": chunk_index,
                "start": start,
                "n_replicas": n,
                "expected_counts": ref["expected_counts"],
                "edges": [float(e) for e in edges],
                "walkers": ref["walkers"],
                "sampling": ref["sampling"],
                "true_modes": int(ref.get("true_modes", 1)),
                "true_bins": ref.get("true_mode_bins"),
            })
    total_hist = sum(t["n_replicas"] for t in tasks)
    print(f"[tasks] {len(tasks)} chunks, {total_hist:,} histograms, {args.workers} workers", flush=True)
    wall, cpu_seconds = _pool_run(tasks, arrays, args.workers, args.chunk)
    print(f"[run] {total_hist:,} histograms in {wall:.0f}s wall, "
          f"{cpu_seconds:.0f}s worker CPU, {1e3 * cpu_seconds / total_hist:.2f} ms/histogram",
          flush=True)

    # Aggregation.
    ref_results = [aggregate_reference(ref, arrays[ref["ref_index"]]) for ref in references]
    stress_results = [aggregate_reference(ref, arrays[ref["ref_index"]]) for ref in stress]
    fam_results = []
    fam_flag_store = {}
    for fname, cells in families.items():
        agg = aggregate_family(fname, cells, [arrays[c["ref_index"]] for c in cells], args.families)
        fam_flag_store[fname] = agg.pop("_family_flags")
        fam_results.append(agg)
    flags = list(fam_flag_store.values())
    gate_u = np.logical_or.reduce([f["gate"] for f in flags])
    full_u = np.logical_or.reduce([f["full"] for f in flags])
    not_one_u = np.logical_or.reduce([f["not_one"] for f in flags])
    max_u = np.max(np.stack([f["max"] for f in flags]), axis=0)
    fam_results.append({
        "name": "union_96",
        "n_cells": sum(len(c) for c in families.values()),
        "families": int(args.families),
        "histograms_total": int(args.families * sum(len(c) for c in families.values())),
        "family_wise_gate_only_false_positive": _rate(int(gate_u.sum()), args.families),
        "family_wise_full_rule_false_positive": _rate(int(full_u.sum()), args.families),
        "family_wise_mode_count_mismatch": _rate(int(not_one_u.sum()), args.families),
        "family_max_spurious_z_cov": _z_distribution(max_u),
    })
    shoulder_block = None
    if shoulder:
        sh = aggregate_family("shoulder_full_mf", shoulder, [arrays[c["ref_index"]] for c in shoulder], args.families)
        sh.pop("_family_flags")
        sh_results = [aggregate_reference(c, arrays[c["ref_index"]]) for c in shoulder]
        n_sh = int(sum(r["replicas"] for r in sh_results))
        shoulder_block = {
            "definition": (
                "full mean-field f1 (all slabs) in every W1 grid cell whose expected bin counts have "
                "fewer than m interior maxima (the lost late mode is a shoulder); true modes = the "
                "bin-level maxima; a true maximum is matched to the tallest smoothed maximum within "
                f"+-{MATCH_RADIUS_BINS} bins; over-count (mode count > true) is the primary false-positive metric"
            ),
            "seed": {"campaign_seed": SEED, "tag": TAG_SHOULDER, "entropy": "[seed, 72, cell index, chunk index]"},
            "n_cells": len(shoulder),
            "cells": [c["name"] for c in shoulder],
            "pooled_over_count": _rate(int(sum(r["over_count"]["count"] for r in sh_results)), n_sh),
            "pooled_under_count": _rate(int(sum(r["under_count"]["count"] for r in sh_results)), n_sh),
            "pooled_full_rule_false_positive_unmatched": _rate(
                int(sum(r["full_rule_false_positive"]["count"] for r in sh_results)), n_sh),
            "pooled_gate_only_false_positive_unmatched": _rate(
                int(sum(r["gate_only_false_positive"]["count"] for r in sh_results)), n_sh),
            "max_unmatched_spurious_z_cov": float(max(
                r["max_spurious_z_covariance_aware"].get("max", -1.0) for r in sh_results)),
            "pooled_unmatched_significant_without_over_count": _rate(
                int(sum(r["unmatched_significant_without_over_count"]["count"] for r in sh_results)), n_sh),
            "unmatched_metric_caveat": (
                "an 'unmatched' significant maximum with mode count <= true count is the true broad "
                "maximum displaced by more than the match radius (flat-topped merged maximum, e.g. "
                "eps = 0.25 at B <= 0.5 where lambda stays >= 89% of its peak over +-15 bins), not a "
                "false mode; over-count is the valid false-positive metric"
            ),
            "noise_free_mode_count_equals_true_all": bool(all(
                c["noise_free_classifier"]["mode_count_covariance_aware"] == c["true_modes"] for c in shoulder)),
            "family": sh,
            "cell_results": sh_results,
        }

    # Pooled per-histogram statistics: unimodal Poisson references + family cells
    # (stress and shoulder references are EXCLUDED; they have their own blocks).
    uni_refs = [r for r in references if r["sampling"] == "poisson" and int(r.get("true_modes", 1)) == 1]
    pooled_idx = [ref["ref_index"] for ref in uni_refs]
    pooled_idx += [c["ref_index"] for cells in families.values() for c in cells]
    z_all = np.concatenate([arrays[i]["max_spurious_z_cov"] for i in pooled_idx])
    gate_all = np.concatenate([arrays[i]["gate_fp"] for i in pooled_idx])
    full_all = np.concatenate([arrays[i]["full_fp"] for i in pooled_idx])
    not_one_all = np.concatenate([arrays[i]["mode_count"] != 1 for i in pooled_idx])
    z_refs_only = np.concatenate([arrays[ref["ref_index"]]["max_spurious_z_cov"] for ref in uni_refs])
    n_all = int(z_all.size)
    gate_rate = _rate(int(gate_all.sum()), n_all)
    p_upper = gate_rate["wilson95"][1]
    ref_true = [r for r in ref_results if r["true_mode_counted"] is not None]
    headline = {
        "per_histogram_pooled": {
            "histograms": n_all,
            "note": "unimodal Poisson references plus every 48-cell family histogram (stress/shoulder excluded)",
            "gate_only_false_positive": gate_rate,
            "full_rule_false_positive": _rate(int(full_all.sum()), n_all),
            "mode_count_not_one": _rate(int(not_one_all.sum()), n_all),
            "max_spurious_z_cov": float(z_all.max()),
            "n_max_spurious_z_ge_4": int(np.sum(z_all >= 4.0)),
            "n_max_spurious_z_ge_4p5": int(np.sum(z_all >= 4.5)),
            "quantiles": {k: float(v) for k, v in zip(("q50", "q90", "q99", "q999"), np.quantile(z_all[z_all >= 0], [0.5, 0.9, 0.99, 0.999]))},
            "analytic_48_cell_family_wise_upper_bound_from_pooled": float(1.0 - (1.0 - p_upper) ** 48),
            "analytic_bound_note": "uses the POOLED per-histogram Wilson upper limit across heterogeneous references; the per-cell bounds are in families[*].per_cell_wilson95_upper_gate_only",
        },
        "per_histogram_references_only": {
            "histograms": int(z_refs_only.size),
            "max_spurious_z_cov": float(z_refs_only.max()),
            "gate_only_false_positive": _rate(int(sum(arrays[ref["ref_index"]]["gate_fp"].sum() for ref in uni_refs)), int(z_refs_only.size)),
            "full_rule_false_positive": _rate(int(sum(arrays[ref["ref_index"]]["full_fp"].sum() for ref in uni_refs)), int(z_refs_only.size)),
        },
        "family_wise": {
            f["name"]: {
                "families": f["families"],
                "gate_only": f["family_wise_gate_only_false_positive"],
                "full_rule": f["family_wise_full_rule_false_positive"],
                "max_spurious_z_cov": f["family_max_spurious_z_cov"].get("max"),
                "q999_family_max_z": f["family_max_spurious_z_cov"].get("quantiles", {}).get("q999"),
                "per_cell_wilson95_upper_gate_only": f.get("per_cell_wilson95_upper_gate_only"),
            }
            for f in fam_results
        },
        "true_mode_always_counted": bool(all(r["true_mode_counted"]["count"] == r["replicas"] for r in ref_true)),
        "true_mode_min_z_over_references": float(min(r["true_mode_z"]["min"] for r in ref_true if r["true_mode_z"]["min"] is not None)),
        "floor_only_spurious_total": int(sum(r["floor_only_spurious"]["count"] for r in ref_results)),
        "max_spurious_relative_prominence_references": float(max(r["max_spurious_relative_prominence"] for r in ref_results)),
        "stress_flat_plateau_max_z": float(max(r["max_spurious_z_covariance_aware"].get("max", -1.0) for r in stress_results)) if stress_results else None,
        "stress_flat_plateau_n_z_ge_5": int(sum(r["max_spurious_z_covariance_aware"].get("n_ge_5", 0) for r in stress_results)),
        "shoulder_pooled_over_count": shoulder_block["pooled_over_count"] if shoulder_block else None,
        "wording": "operational gate, not a calibrated post-selection level",
    }
    stress_block = {
        "definition": (
            "constant expected count lambda on all 150 window bins, no true mode; every smoothed local "
            "maximum is spurious; z is the post-selected maximum over the whole window (look-elsewhere)"
        ),
        "references": [
            {**_stress_block(ref, agg, arrays[ref["ref_index"]]), "replicas": agg["replicas"]}
            for ref, agg in zip(stress, stress_results)
        ],
    }
    pilot_repro = None
    if args.pilot and LEGACY_PILOT_JSON.exists():
        legacy = json.loads(LEGACY_PILOT_JSON.read_text(encoding="utf-8"))
        leg_refs = {r["name"]: r for r in legacy["reference_results"]}
        mism = []
        for r in ref_results:
            lr = leg_refs.get(r["name"])
            if lr is None:
                mism.append((r["name"], "missing"))
                continue
            a = r["max_spurious_z_covariance_aware"].get("max")
            b = lr["max_spurious_z_covariance_aware"].get("max")
            if a != b or r["fraction_with_spurious_maximum"] != lr["fraction_with_spurious_maximum"]:
                mism.append((r["name"], a, b))
        leg_cells = {c["cell"]: c for f in legacy["families"] if "per_cell" in f for c in f["per_cell"]}
        n_cells_cmp = 0
        for f in fam_results:
            for c in f.get("per_cell", []):
                lc = leg_cells.get(c["cell"])
                n_cells_cmp += 1
                if lc is None or lc["max_spurious_z_cov"] != c["max_spurious_z_cov"]:
                    mism.append((c["cell"], c["max_spurious_z_cov"], None if lc is None else lc["max_spurious_z_cov"]))
        pilot_repro = {
            "legacy_file": str(LEGACY_PILOT_JSON),
            "references_compared": len(ref_results),
            "family_cells_compared": n_cells_cmp,
            "mismatches": mism,
            "bit_for_bit": not mism,
            "legacy_pooled_max_z": legacy["headline"]["per_histogram_pooled"]["max_spurious_z_cov"],
            "new_pooled_max_z": headline["per_histogram_pooled"]["max_spurious_z_cov"],
        }
        print(f"[pilot] reproduction mismatches: {len(mism)}", flush=True)

    summary = {
        "stream": "I5_D2_null_calibration",
        "revision": (
            "2026-09-23 I5: (a) stress references filtered out of the headline (rerun crash fixed); "
            "(b) separate stress_flat_plateau block; (c) full mean-field shoulder family (tag 72); "
            "(d) per-cell Wilson bounds; (e) figure fix. Pilot seeds preserved (ref_index order)."
        ),
        "wording": "operational gate, not a calibrated post-selection level",
        "analysis": (
            "post-selection null behaviour of the covariance-aware five-sigma classifier: "
            "Poisson-resampled histograms of smooth unimodal reference densities at the production "
            "walker numbers, flat-plateau look-elsewhere stress, and full mean-field shoulder cells, "
            "classified verbatim by reclassify_covariance_aware.classify_both"
        ),
        "classifier": {
            "function": "reclassify_covariance_aware.classify_both",
            "bandwidth": BANDWIDTH,
            "bin_width": BIN_WIDTH,
            "window": list(WINDOW),
            "prominence_sigma_factor": SIGMA_FACTOR,
            "prominence_relative_floor": RELATIVE_FLOOR,
            "sigma_convention": "covariance-aware Poisson plug-in (article's formal classifier); peak-only z recorded for comparison",
        },
        "null_model": {
            "sampling": "independent Poisson bin counts C_i ~ Poisson(lambda_i) (multinomial cross-check on one reference)",
            "spurious_definition": "unimodal: every smoothed local maximum other than the tallest; flat: every maximum; shoulder: every maximum not matched to a true bin-level maximum",
            "gate_only_false_positive": "largest spurious z_cov >= 5 (relative floor ignored)",
            "full_rule_false_positive": "a spurious maximum with z_cov >= 5 AND prominence >= 0.05 * max smoothed height (the article's rule)",
            "floor_only_spurious": "a spurious maximum with prominence >= 0.05 * max smoothed height (z ignored)",
            "mean_field_grid": {"dt_mf": DT_MF, "tmax": T_MAX, "steps_per_window_bin": int(round(BIN_WIDTH / DT_MF))},
            "bulk_definition": f"bins with expected count >= {BULK_FRACTION:g} x the largest expected count",
            "caveat": "unimodal references place spurious maxima only in sparse tails (see fraction_argmax_spurious_in_bulk); the flat-plateau and shoulder blocks are the look-elsewhere stress tests",
        },
        "seeds": {
            "campaign_seed": SEED,
            "tag_references_families_stress": TAG_D2_NULL,
            "tag_shoulder": TAG_SHOULDER,
            "rng": "numpy Philox, SeedSequence([seed, tag, seed_index, chunk_index]) per chunk; seed_index = ref_index except shoulder (cell index)",
            "chunk": args.chunk,
        },
        "sizes": {
            "replicas_per_reference": args.replicas,
            "families_per_grid": args.families,
            "n_references": len(references),
            "n_stress_references": len(stress),
            "n_family_cells": sum(len(c) for c in families.values()),
            "n_shoulder_cells": len(shoulder),
            "histograms_total": total_hist,
            "pilot": bool(args.pilot),
        },
        "runtime": {"wall_seconds": wall, "worker_cpu_seconds": cpu_seconds, "ms_per_histogram": 1e3 * cpu_seconds / total_hist, "workers": args.workers},
        "model_parameters": core.model_dict(core.MODEL),
        "phase_grid": {"eps": list(EPS_GRID), "budget": list(B_GRID), "walkers": GRID_WALKERS, "weights": {str(k): list(v) for k, v in M_WEIGHTS.items()}},
        "headline": headline,
        "stress_flat_plateau": stress_block,
        "shoulder_family": shoulder_block,
        "pilot_reproduction": pilot_repro,
        "references": references + stress,
        "reference_results": ref_results + stress_results,
        "family_cells": {fname: cells for fname, cells in families.items()},
        "shoulder_cells": shoulder,
        "families": fam_results,
    }
    out = PILOT_JSON if args.pilot else OUT_JSON
    core.write_json(out, _py(summary))
    print("wrote", out, flush=True)
    if not args.pilot:
        for path in make_figure(json.loads(OUT_JSON.read_text(encoding="utf-8"))):
            print("wrote", path, flush=True)
    print("DONE", flush=True)


# ----------------------------------------------------------------------------
# Power curves (O-E5): detection probability of the last mode vs its relative
# prominence, full mean-field f1, B tuned so that the noise-free smoothed
# relative prominence of the last mode equals p.
# ----------------------------------------------------------------------------


def _last_mode_row(lam: np.ndarray, edges: np.ndarray, walkers: float) -> dict | None:
    res = mfb.classify_real(lam, edges, walkers)
    rows = res["rows"]
    return max(rows, key=lambda r: r["time"]) if rows else None


def _power_budget(clock: ComponentClock, m: int, target: float, walkers: float) -> dict:
    """Smallest B (above the prominence maximum) with noise-free smoothed
    relative prominence of the last mode == target, by log-B bisection."""
    grid = np.geomspace(0.25, 400.0, 321)
    prom = []
    for b in grid:
        lam = clock.expected_counts_full_exact(float(b), walkers)
        n_max = len(_interior_strict_maxima(lam))
        res = mfb.classify_real(lam, clock.window_edges, walkers)
        rows = res["rows"]
        ok = n_max == m and len(rows) == m
        prom.append(max(rows, key=lambda r: r["time"])["relative_prominence"] if ok else -1.0)
    prom = np.asarray(prom)
    k0 = int(np.argmax(prom))
    idx = [k for k in range(k0, grid.size - 1) if prom[k] >= target > prom[k + 1]]
    if not idx:
        return {"status": "not_bracketed", "target": target}
    k = idx[0]
    lo, hi = float(grid[k]), float(grid[k + 1])

    def rp(b):
        lam = clock.expected_counts_full_exact(b, walkers)
        row = _last_mode_row(lam, clock.window_edges, walkers)
        return (row["relative_prominence"] if row is not None else -1.0), lam, row

    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if rp(mid)[0] >= target:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-10:
            break
    b = math.sqrt(lo * hi)
    val, lam, row = rp(b)
    return {
        "status": "bisected",
        "target": target,
        "budget": b,
        "achieved_relative_prominence": float(val),
        "last_mode_time": float(row["time"]),
        "last_mode_bin": int(row["bin"]),
        "n_bin_level_maxima": len(_interior_strict_maxima(lam)),
        "monotone_after_peak_scan": bool(np.all(np.diff(prom[k0:][prom[k0:] >= 0]) <= 1e-12)),
    }


def power_chunk(task: dict) -> dict:
    lam = np.asarray(task["expected_counts"], dtype=float)
    edges = np.asarray(task["edges"], dtype=float)
    walkers = int(task["walkers"])
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(task["entropy"])))
    b_last = int(task["last_bin"])
    m = int(task["m"])
    det_gate = det_full = count_m = 0
    z_last = []
    t0 = time.perf_counter()
    for _ in range(int(task["n_replicas"])):
        counts = rng.poisson(lam)
        res = recl.classify_both(counts, edges, walkers, bandwidth=BANDWIDTH,
                                 sigma_factor=SIGMA_FACTOR, relative_floor=RELATIVE_FLOOR)
        rows = res["rows"]
        matched = _match_true_rows(rows, [b_last])
        if matched:
            r = rows[matched[0]]
            z_last.append(float(r["z_covariance_aware"]))
            det_gate += int(r["z_covariance_aware"] >= SIGMA_FACTOR)
            det_full += int(bool(r["significant_covariance_aware"]))
        else:
            z_last.append(-1.0)
        count_m += int(res["mode_count_covariance_aware"] == m)
    return {"key": task["key"], "n": int(task["n_replicas"]), "det_gate": det_gate, "det_full": det_full,
            "count_m": count_m, "z_last": z_last, "seconds": time.perf_counter() - t0}


def run_power(args) -> None:
    t0 = time.perf_counter()
    edges = np.arange(WINDOW[0], WINDOW[1] + 0.5 * BIN_WIDTH, BIN_WIDTH)
    cells, tasks = [], []
    for di, (m, eps) in enumerate(POWER_DESIGNS):
        clock = clock_for(core.centres_z_for(m), eps, M_WEIGHTS[m])
        for pi, p in enumerate(POWER_REL_PROMINENCES):
            tune = _power_budget(clock, m, p, 1.0e6)
            print(f"[power] m={m} eps={eps} p={p}: {tune}", flush=True)
            if tune["status"] != "bisected":
                cells.append({"m": m, "eps": eps, "p": p, "tuning": tune})
                continue
            for ni, n_w in enumerate(POWER_WALKERS):
                lam = clock.expected_counts_full_exact(tune["budget"], n_w)
                nf = mfb.classify_real(lam, edges, n_w)
                last = max(nf["rows"], key=lambda r: r["time"])
                key = f"m{m}_eps{eps:g}_p{p:g}_N{n_w:.0e}"
                entropy = [POWER_SEED, TAG_POWER, di, pi, ni]
                cells.append({
                    "key": key, "m": m, "eps": eps, "p": p, "walkers": n_w, "tuning": tune,
                    "noise_free_last_mode_z_cov": float(last["z_covariance_aware"]),
                    "noise_free_last_mode_relative_prominence": float(last["relative_prominence"]),
                    "noise_free_mode_count": int(nf["mode_count_covariance_aware"]),
                    "expected_window_events": float(lam.sum()),
                    "seed_entropy": entropy,
                })
                half = args.power_replicas // 2
                for ci, n_rep in enumerate((half, args.power_replicas - half)):
                    tasks.append({"key": key, "expected_counts": [float(v) for v in lam],
                                  "edges": [float(e) for e in edges], "walkers": n_w, "m": m,
                                  "last_bin": tune["last_mode_bin"], "n_replicas": n_rep,
                                  "entropy": entropy + [ci]})
    print(f"[power] {len(tasks)} tasks built in {time.perf_counter() - t0:.1f}s", flush=True)
    acc: dict[str, dict] = {}
    cpu = 0.0
    t_run = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(power_chunk, tasks):
            a = acc.setdefault(res["key"], {"n": 0, "det_gate": 0, "det_full": 0, "count_m": 0, "z_last": []})
            for k in ("n", "det_gate", "det_full", "count_m"):
                a[k] += res[k]
            a["z_last"] += res["z_last"]
            cpu += res["seconds"]
    for c in cells:
        if "key" not in c:
            continue
        a = acc[c["key"]]
        z = np.asarray(a["z_last"])
        zf = z[z >= 0]
        c.update({
            "replicas": a["n"],
            "detection_gate_only": _rate(a["det_gate"], a["n"]),
            "detection_full_rule": _rate(a["det_full"], a["n"]),
            "mode_count_equals_m": _rate(a["count_m"], a["n"]),
            "last_mode_matched_fraction": float(zf.size / z.size),
            "last_mode_z_quantiles": (
                {k: float(v) for k, v in zip(("q05", "q50", "q95"), np.quantile(zf, [0.05, 0.5, 0.95]))}
                if zf.size else None
            ),
        })
    payload = {
        "stream": "I5_power_curves",
        "wording": "operational gate, not a calibrated post-selection level",
        "definition": (
            "full mean-field f1 for (m, eps) at the budget B_p where the noise-free smoothed "
            "(bandwidth 0.04) relative prominence of the LAST mode equals p (log-B bisection above the "
            "prominence maximum, expected counts at N = 1e6; relative prominence is N-invariant); "
            "Poisson histograms at N walkers; detection = the smoothed maximum matched to the "
            f"noise-free last-mode bin (tallest within +-{MATCH_RADIUS_BINS} bins) has z_cov >= 5 "
            "(gate only) or z_cov >= 5 and prominence >= 5% of the global max (full rule)"
        ),
        "classifier": {"bandwidth": BANDWIDTH, "bin_width": BIN_WIDTH, "window": list(WINDOW),
                       "sigma_factor": SIGMA_FACTOR, "relative_floor": RELATIVE_FLOOR},
        "designs": [list(d) for d in POWER_DESIGNS],
        "relative_prominences": list(POWER_REL_PROMINENCES),
        "walkers": list(POWER_WALKERS),
        "replicas_per_cell": args.power_replicas,
        "seeds": {"base": POWER_SEED, "tag": TAG_POWER,
                  "entropy": "[20260923, 87, design index, p index, N index, half index]"},
        "cells": cells,
        "runtime": {"wall_seconds": time.perf_counter() - t_run, "worker_cpu_seconds": cpu, "workers": args.workers},
    }
    core.write_json(POWER_JSON, _py(payload))
    print("wrote", POWER_JSON, flush=True)
    for path in make_power_figure(json.loads(POWER_JSON.read_text(encoding="utf-8"))):
        print("wrote", path, flush=True)


def make_power_figure(payload: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt

    designs = [tuple(d) for d in payload["designs"]]
    fig, axes = plt.subplots(1, len(designs), figsize=(6.6, 2.6), constrained_layout=True, sharey=True)
    colours = {200_000: core.OI_ORANGE, 1_000_000: core.OI_BLUE, 5_000_000: core.OI_VERMILLION}
    for ax, (m, eps) in zip(np.atleast_1d(axes), designs):
        for n_w in payload["walkers"]:
            rows = sorted([c for c in payload["cells"] if c.get("key") and c["m"] == m and abs(c["eps"] - eps) < 1e-12
                           and c["walkers"] == n_w], key=lambda c: c["p"])
            if not rows:
                continue
            p = np.array([100 * c["p"] for c in rows])
            for kind, ls, mk in (("detection_gate_only", "--", "o"), ("detection_full_rule", "-", "s")):
                y = np.array([c[kind]["rate"] for c in rows])
                lo = np.array([c[kind]["wilson95"][0] for c in rows])
                hi = np.array([c[kind]["wilson95"][1] for c in rows])
                ax.errorbar(p, y, yerr=[y - lo, hi - y], color=colours.get(n_w, "0.4"), ls=ls, marker=mk,
                            ms=3.0, lw=1.0, capsize=1.5,
                            label=(_nlabel(n_w) if kind == "detection_full_rule" else None))
        ax.axvline(100 * RELATIVE_FLOOR, color="0.5", lw=0.8, ls=":")
        ax.text(100 * RELATIVE_FLOOR + 0.15, 0.03, "5% floor", fontsize=6.4, color="0.35", rotation=90,
                ha="left", va="bottom")
        zmin = min((c["noise_free_last_mode_z_cov"] for c in payload["cells"]
                    if c.get("key") and c["m"] == m and abs(c["eps"] - eps) < 1e-12), default=None)
        if zmin is not None:
            ax.text(0.97, 0.08, f"noise-free last-mode $z\\geq{math.floor(10 * zmin) / 10:.1f}$\nin every cell", transform=ax.transAxes,
                    fontsize=6.2, ha="right", va="bottom", color="0.25")
        ax.set_xlabel("relative prominence of the last mode (%)")
        ax.set_title(rf"$(m,\varepsilon)=({m},{eps:g})$", loc="left", fontsize=8.0)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.03, 1.03)
    np.atleast_1d(axes)[0].set_ylabel("detection probability")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="0.3", ls="-", marker="s", ms=3), Line2D([], [], color="0.3", ls="--", marker="o", ms=3)]
    l += [r"full rule ($5\sigma$ & 5%)", r"$5\sigma$ gate only"]
    np.atleast_1d(axes)[-1].legend(h, l, loc="center right", fontsize=6.2, frameon=False)
    written = core.save_figure(fig, POWER_FIG_STEM)
    plt.close(fig)
    return written


def _py(obj):
    """Recursively convert numpy scalars/arrays for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


if __name__ == "__main__":
    main()
