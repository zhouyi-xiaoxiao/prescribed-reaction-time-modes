#!/usr/bin/env python3
"""Release-time randomisation robustness of the prescribed modes (stream S3).

Deterministic post-processing of RETAINED histogram counts; no walkers are
simulated.  The prescribed modes assume the pair is released at t = 0 from
z0 != z_bar.  If the release time t0 is random with law rho and independent
of the subsequent dynamics, and reaction times are measured from the nominal
clock, the observed density is the convolution

    f_obs(t) = int f(t - t0) rho(t0) dt0 ,

which smears the modes.  This script convolves the stored reaction-time
histograms of the headline configurations with zero-mean release-time laws

    uniform  : rho = U(-sqrt(3) sigma_t, +sqrt(3) sigma_t)   (sd sigma_t)
    gaussian : rho = N(0, sigma_t^2)                          (sd sigma_t)

for sigma_t / dt_min in {0.1, 0.2, 0.3, 0.5, 0.75, 1.0} (dt_min = smallest
adjacent spacing of the prescribed target times t_j), and re-runs the
covariance-aware classifier on the smeared window histograms.  A nonzero
mean of rho is a rigid translation t_j -> t_j + E[t0] and is not a smearing;
the laws are therefore centred.

Smearing operator (exact, bin to bin).  With Phi_2 the second antiderivative
of rho (Phi_2'' = rho), the probability that an event uniformly distributed in
a source bin [a, b] lands, after adding an independent draw from rho, in the
target bin [c, d] is

    K([c,d] <- [a,b]) = [Phi_2(d-a) - Phi_2(d-b) - Phi_2(c-a) + Phi_2(c-b)] / (b-a).

The smeared expected window counts are C~_i = sum_s K_is C_s over all retained
source bins s: the 150 window bins (width 0.02 on I = [0.5, 3.5]) plus the
retained mass outside I (see "tail treatment").  Events after t_max = 4
(survivors) are unrecorded in every stream and contribute nothing.

Statistic.  The classifier is the article's covariance-aware prominence rule
(``reclassify_covariance_aware.classify_both``; bandwidth 0.04, five-sigma
gate, 5 % relative floor) evaluated on the real-valued smeared counts through
the regression-tested adapter ``exact_m_prr_mean_field_boundary.classify_real``.
The Poisson plug-in variance is taken at the smeared expected counts C~, i.e.
z is evaluated as for an N-walker experiment with jittered release whose
observed counts are Poisson with mean C~ (the counting noise the experimenter
would face).  Since 2026-09-23 the full Monte Carlo estimation covariance of
C~, K diag(C) K^T, propagated through the prominence functional, is also
added in a separate variant (``first_loss_with_estimation_variance``).

Boundary (2026-09-23 audit).  The survival boundary is the FIRST LOSS: the
count is scanned on a 0.01 grid of sigma_t/dt_min in [0, 1] and bisected 12
times between the last passing and the first failing fine point; re-entrance
above the first loss (non-monotone survival, seen for the uniform law) is
reported, and the gate that fails at the loss (relative floor or 5 sigma) is
recorded.  The earlier coarse-grid bisection is kept as
``refined_boundary_legacy_coarse_bisection`` (it is not the first loss when
the count re-enters: m3_eps0.1_B1_w50-30-20 uniform, legacy 0.4323 versus
first loss 0.3312).  Two tail-extension variants continue the density beyond
t_max = 4 to t = 10 (exponential and flat) to bound the truncation effect.

Tail treatment.  The contour base of the last counted peak is the window's
right edge in every headline cell, so the recorded mass in (3.5, 4] matters
for the last peak's prominence once it is smeared into the window:
  ``retained``          production records retain a full-range 0.04-bin
                        histogram on [0, 4]; its bins outside I are used
                        exactly (straddling bins split against the window
                        counts).                                [18 cells]
  ``exponential_fill``  the W4 (m=5) and W5 (d=3) records retain only the
                        total count outside I (kills - kills_in_window,
                        all in (3.5, 4]; the first window bins are empty).
                        The tail is filled with A exp(-lambda (t - 3.5)),
                        A from a log-linear fit of the last five window
                        bins, lambda from the retained total, discretised
                        exactly into 0.02 bins on (3.5, 4].  Validated
                        against ``retained`` on the 18 production cells.
  ``window_only``       control: mass outside I set to zero.
The primary verdict of a cell uses ``retained`` when available, otherwise
``exponential_fill``; every variant is stored.

Outputs
  artifacts/data/exact_m_prr_upgrade/robustness/release_time_smearing/
      summary.json, cell_<name>.json
  artifacts/figures/exact_m_release_time_smearing_prr.{png,pdf}

Reproduce (about one minute, deterministic):
  python3 code/exact_m_prr_release_time_smearing.py
  python3 code/exact_m_prr_release_time_smearing.py --replot   # figure only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import exact_m_prr_mean_field_boundary as mf  # noqa: E402  (classify_real)
import reclassify_covariance_aware as recl  # noqa: E402  (classify_both)
import validate_exact_m_offlattice as base  # noqa: E402

OUT_DIR = core.UPGRADE_DATA / "robustness" / "release_time_smearing"
SUMMARY_OUT = OUT_DIR / "summary.json"
FIGURE_STEM = core.FIGURES / "exact_m_release_time_smearing_prr"
PRODUCTION_DIR = core.REPORT / "artifacts" / "data" / "exact_m_offlattice_production"
W4_PATH = core.UPGRADE_DATA / "w4_m5_demo" / "m5_demo.json"
W5_PATH = core.UPGRADE_DATA / "w5_d3_spotcheck" / "d3_spotcheck.json"

RATIOS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
FINE_STEP = 0.01
FINE_RATIOS = tuple(float(v) for v in np.round(np.arange(0.0, 1.0 + 1e-9, FINE_STEP), 2))
LAWS = ("uniform", "gaussian")
TAIL_TREATMENTS = ("retained", "exponential_fill", "window_only",
                   "tail_extension_exp", "tail_extension_flat")
T_EXTENSION = 10.0  # sources continued beyond t_max = 4 up to this time (variants)
BISECT_ITERATIONS = 12
EDGE_FIT_BINS = 5
T_MAX = base.DEFAULT_TMAX
WINDOW = base.WINDOW
BANDWIDTH = base.DEFAULT_BANDWIDTH
SIGMA_FACTOR = base.PROMINENCE_SIGMA_FACTOR
RELATIVE_FLOOR = base.PROMINENCE_RELATIVE_FLOOR

SELF_TEST_SEED = 20260909
SELF_TEST_SAMPLES = 2_000_000

# (name, path, short label); the first four are the headline cells (figure).
HEADLINE_CELLS = (
    ("m2_eps0.1_B1_w50-50", PRODUCTION_DIR / "m2_eps0.1_B1_w50-50.json",
     r"$(m,\varepsilon,B)=(2,0.1,1)$"),
    ("m3_eps0.1_B1_w33-33-33", PRODUCTION_DIR / "m3_eps0.1_B1_w33-33-33.json",
     r"$(m,\varepsilon,B)=(3,0.1,1)$"),
    ("m5_demo", W4_PATH, r"$m=5$ demo $(\varepsilon,B)=(0.1,1)$"),
    ("d3_spotcheck", W5_PATH, r"$d=3$ spot check $(2,0.1,1)$"),
)

_erf_scalar = np.vectorize(math.erf)


def _erf(x) -> np.ndarray:
    """math.erf elementwise, evaluated once per distinct argument (the
    bin-difference arguments repeat heavily on uniform grids); bit-identical
    to the plain vectorised evaluation."""
    x = np.asarray(x, dtype=float)
    uniq, inv = np.unique(x, return_inverse=True)
    return _erf_scalar(uniq)[inv].reshape(x.shape)


# ----------------------------------------------------------------------------
# Release-time laws: CDF and second antiderivative (both zero-mean, sd sigma).
# ----------------------------------------------------------------------------


def law_cdf(x, law: str, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sigma == 0.0:
        return (x >= 0.0).astype(float)
    if law == "gaussian":
        return 0.5 * (1.0 + _erf(x / (sigma * math.sqrt(2.0))))
    if law == "uniform":
        a = math.sqrt(3.0) * sigma
        return np.clip((x + a) / (2.0 * a), 0.0, 1.0)
    raise ValueError(f"unknown law {law!r}")


def law_second_antiderivative(x, law: str, sigma: float) -> np.ndarray:
    """Phi_2(x) = int_{-inf}^x F_rho(u) du (Phi_2'' = rho)."""
    x = np.asarray(x, dtype=float)
    if sigma == 0.0:
        return np.maximum(x, 0.0)
    if law == "gaussian":
        u = x / sigma
        cdf = 0.5 * (1.0 + _erf(u / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
        return x * cdf + sigma * pdf
    if law == "uniform":
        a = math.sqrt(3.0) * sigma
        return np.where(
            x <= -a, 0.0, np.where(x >= a, x, (x + a) ** 2 / (4.0 * a))
        )
    raise ValueError(f"unknown law {law!r}")


def transfer_matrix(
    target_edges: np.ndarray,
    source_lo: np.ndarray,
    source_hi: np.ndarray,
    law: str,
    sigma: float,
) -> np.ndarray:
    """K[i, s] = P(uniform draw in source bin s + rho draw lands in target bin i)."""
    c = np.asarray(target_edges[:-1], float)[:, None]
    d = np.asarray(target_edges[1:], float)[:, None]
    a = np.asarray(source_lo, float)[None, :]
    b = np.asarray(source_hi, float)[None, :]
    p2 = lambda v: law_second_antiderivative(v, law, sigma)  # noqa: E731
    kmat = (p2(d - a) - p2(d - b) - p2(c - a) + p2(c - b)) / (b - a)
    if kmat.min() < -1e-10 or kmat.max() > 1.0 + 1e-10:
        raise AssertionError("transfer matrix left [0, 1] beyond round-off")
    return np.clip(kmat, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Source bins of a stored record (window counts + retained outside mass).
# ----------------------------------------------------------------------------


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exponential_tail_fill(
    counts: np.ndarray,
    edges: np.ndarray,
    walkers: int,
    outside_total: int,
    *,
    n_fit: int = EDGE_FIT_BINS,
) -> dict:
    """Fill (hi, T_MAX] with A exp(-lambda (t - hi)): A from a log-linear fit of
    the last ``n_fit`` window bins, lambda from the retained outside total."""
    hi = float(edges[-1])
    bin_width = round(float((edges[-1] - edges[0]) / counts.size), 14)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dens = counts[-n_fit:] / (walkers * bin_width)
    if np.any(dens <= 0.0):
        raise ValueError("edge bins must be populated for the exponential fill")
    slope, intercept = np.polyfit(centres[-n_fit:] - hi, np.log(dens), 1)
    level = float(math.exp(intercept))  # density at t = hi
    lam_fit = float(-slope)
    span = T_MAX - hi
    target = outside_total / walkers  # mass to place in (hi, T_MAX]
    if target >= level * span:
        lam_mass = 0.0
        flat = True
        level_used = target / span
    else:
        flat = False
        level_used = level

        def mass(lam: float) -> float:
            return level * (1.0 - math.exp(-lam * span)) / lam

        lo, up = 1e-9, 1e3
        for _ in range(200):
            mid = 0.5 * (lo + up)
            if mass(mid) > target:
                lo = mid
            else:
                up = mid
        lam_mass = 0.5 * (lo + up)
    n_bins = int(round(span / bin_width))
    tail_edges = hi + bin_width * np.arange(n_bins + 1)
    if flat:
        tail_counts = np.full(n_bins, outside_total / n_bins, dtype=float)
    else:
        rel = tail_edges - hi
        cum = walkers * level_used * (1.0 - np.exp(-lam_mass * rel)) / lam_mass
        tail_counts = np.diff(cum)
    if abs(tail_counts.sum() - outside_total) > 1e-6 * max(1.0, outside_total):
        raise AssertionError("exponential fill does not reproduce the outside total")
    return {
        "edges": tail_edges,
        "counts": tail_counts,
        "level_density_at_edge": level,
        "lambda_fit_from_edge_bins": lam_fit,
        "lambda_from_outside_total": lam_mass,
        "flat_fallback": flat,
        "outside_total": int(outside_total),
        "n_fit_bins": n_fit,
    }


def load_cell(name: str, path: Path, label: str) -> dict:
    payload = _read(path)
    params = payload["parameters"]
    cfg = params["config"]
    res = payload["results"]
    cls = res["classifier"]
    walkers = int(cfg.get("walkers", params.get("walkers")))
    edges = np.asarray(cls["edges"], dtype=float)
    counts = np.asarray(cls["counts"], dtype=np.int64)
    lo, hi = float(edges[0]), float(edges[-1])
    kills = int(res["kills"])
    in_window = int(res["kills_in_window"])
    if int(counts.sum()) != in_window:
        raise AssertionError(f"{path.name}: window counts do not sum to kills_in_window")
    target_times = sorted(float(t) for t in cfg["target_times"])
    dt_min = float(min(np.diff(target_times)))
    outside_total = kills - in_window

    window_sources = [(float(edges[j]), float(edges[j + 1]), float(counts[j]))
                      for j in range(counts.size)]
    variants: dict[str, dict] = {}

    full = res.get("histograms", {}).get("linear_full_range")
    head_note = None
    if full is not None:
        h_edges = np.asarray(full["edges"], dtype=float)
        h_counts = np.asarray(full["counts"], dtype=np.int64)
        if int(h_counts.sum()) != kills:
            raise AssertionError(f"{path.name}: full-range histogram does not sum to kills")
        # Reaction times lie on the 0.001 grid, so events exactly on a bin edge
        # shared by the 0.04 and 0.02 grids can be assigned differently by the
        # two float edge arrays; the interior sums therefore differ by a few
        # counts.  Fully-outside 0.04 bins are used as stored; the two
        # straddling bins contribute max(0, F_k - C_window) to their outside
        # piece; the residual bookkeeping discrepancy is recorded.
        outside = []
        head_count = 0
        tail_count = 0
        interior_full = 0
        for k in range(h_counts.size):
            ha, hb = float(h_edges[k]), float(h_edges[k + 1])
            if hb <= lo + 1e-12:
                outside.append((ha, hb, float(h_counts[k])))
                head_count += int(h_counts[k])
                continue
            if ha >= hi - 1e-12:
                outside.append((ha, hb, float(h_counts[k])))
                tail_count += int(h_counts[k])
                continue
            straddles_lo = ha < lo - 1e-12 < hb
            straddles_hi = ha < hi - 1e-12 and hb > hi + 1e-12
            if not (straddles_lo or straddles_hi):
                interior_full += int(h_counts[k])
                continue
            inside_mask = (edges[:-1] >= ha - 1e-12) & (edges[1:] <= hb + 1e-12)
            piece = max(0, int(h_counts[k]) - int(counts[inside_mask].sum()))
            if straddles_lo:
                if piece:
                    outside.append((ha, lo, float(piece)))
                head_count += piece
            else:
                if piece:
                    outside.append((hi, hb, float(piece)))
                tail_count += piece
        discrepancy = outside_total - (head_count + tail_count)
        # Edge-assignment artefact between two histograms of the same events:
        # the relevant scale is the total kill count (observed: 1e-4 of kills).
        if abs(discrepancy) > 1e-3 * kills:
            raise AssertionError(f"{path.name}: outside mass bookkeeping off by {discrepancy}")
        variants["retained"] = {
            "sources": window_sources + outside,
            "head_count": head_count,
            "tail_count": tail_count,
            "edge_assignment_discrepancy_counts": int(discrepancy),
            "interior_full_minus_window_counts": int(interior_full - int(counts[1:-1].sum())),
            "note": ("full-range 0.04-bin histogram bins outside I used exactly; "
                     "straddling bins split against the window counts"),
        }
    else:
        if int(counts[:EDGE_FIT_BINS].sum()) != 0:
            raise AssertionError(f"{path.name}: first window bins populated; head not empty")
        head_note = ("no full-range histogram retained; the first window bins are "
                     "empty so the outside total is placed in (3.5, 4]")

    fill = exponential_tail_fill(counts, edges, walkers, outside_total)
    variants["exponential_fill"] = {
        "sources": window_sources + [
            (float(fill["edges"][k]), float(fill["edges"][k + 1]), float(fill["counts"][k]))
            for k in range(fill["counts"].size)
        ],
        "head_count": 0,
        "tail_count": int(outside_total),
        "fill": {k: v for k, v in fill.items() if k not in ("edges", "counts")},
        "fill_counts": [float(v) for v in fill["counts"]],
        "note": "tail (3.5, 4] reconstructed as A exp(-lambda (t-3.5)); head assumed empty",
    }
    variants["window_only"] = {
        "sources": window_sources,
        "head_count": 0,
        "tail_count": 0,
        "note": "control: mass outside I set to zero",
    }
    primary = "retained" if "retained" in variants else "exponential_fill"

    # Tail-extension variants (audit fix 4): the primary sources plus a
    # continuation of the reaction-time density beyond t_max = 4 up to
    # T_EXTENSION, (a) exponential, (b) flat at the t = 4 level.  Events after
    # t_max are unrecorded, so this bounds the effect of the truncation.
    survivors = 1.0 - kills / walkers
    if full is not None:
        f_edges = np.asarray(full["edges"], dtype=float)
        f_counts = np.asarray(full["counts"], dtype=float)
        f_bw = round(float(f_edges[1] - f_edges[0]), 14)
        f_cen = 0.5 * (f_edges[:-1] + f_edges[1:])
        dens_edge = f_counts[-EDGE_FIT_BINS:] / (walkers * f_bw)
        if np.any(dens_edge <= 0.0):
            raise AssertionError(f"{path.name}: empty full-range edge bins, no tail fit")
        slope, icpt = np.polyfit(f_cen[-EDGE_FIT_BINS:] - T_MAX, np.log(dens_edge), 1)
        ext_level = float(math.exp(icpt))
        ext_lam = float(max(-slope, 0.0))
        ext_bw = f_bw
        ext_source = ("log-linear fit of the last five 0.04 bins of the full-range "
                      "histogram (density at t_max and decay rate)")
    else:
        fd = variants["exponential_fill"]["fill"]
        if fd["flat_fallback"]:
            ext_lam = 0.0
            ext_level = outside_total / walkers / (T_MAX - hi)
        else:
            ext_lam = float(fd["lambda_from_outside_total"])
            ext_level = float(fd["level_density_at_edge"] * math.exp(-ext_lam * (T_MAX - hi)))
        ext_bw = round(float((edges[-1] - edges[0]) / counts.size), 14)
        ext_source = "exponential fill of (3.5, 4] continued (level at t_max, lambda from the retained outside total)"
    n_ext = int(round((T_EXTENSION - T_MAX) / ext_bw))
    ext_edges = T_MAX + ext_bw * np.arange(n_ext + 1)
    for kind in ("exp", "flat"):
        if kind == "flat" or ext_lam <= 0.0:
            ext_dens = np.full(n_ext, ext_level)
        else:
            ext_dens = ext_level * (np.exp(-ext_lam * (ext_edges[:-1] - T_MAX))
                                    - np.exp(-ext_lam * (ext_edges[1:] - T_MAX))) / (ext_lam * ext_bw)
        ext_counts = ext_dens * ext_bw * walkers
        added = float(ext_counts.sum() / walkers)
        variants[f"tail_extension_{kind}"] = {
            "sources": variants[primary]["sources"] + [
                (float(ext_edges[k]), float(ext_edges[k + 1]), float(ext_counts[k]))
                for k in range(n_ext)
            ],
            "head_count": variants[primary]["head_count"],
            "tail_count": variants[primary]["tail_count"],
            "extension": {
                "kind": kind,
                "base_variant": primary,
                "t_range": [T_MAX, T_EXTENSION],
                "bin_width": ext_bw,
                "density_at_t_max": ext_level,
                "decay_rate": ext_lam if kind == "exp" else 0.0,
                "added_mass_per_walker": added,
                "survivor_fraction_at_t_max": survivors,
                "added_mass_exceeds_survivors": bool(added > survivors),
                "source": ext_source,
            },
            "note": (f"{primary} sources plus a {kind} continuation of the density on "
                     f"({T_MAX:g}, {T_EXTENSION:g}] (truncation sensitivity)"),
        }

    return {
        "name": name,
        "label": label,
        "file": str(path.relative_to(core.REPORT)),
        "stream": params.get("stream", "production"),
        "m": int(cfg["m"]),
        "eps": float(cfg["eps"]),
        "budget": float(cfg["budget"]),
        "weights": [float(w) for w in cfg["weights"]],
        "dim": int(cfg.get("dim", 2)),
        "walkers": walkers,
        "dt": float(cfg.get("dt", params.get("dt"))),
        "seed": int(cfg.get("seed", params.get("seed"))),
        "target_times": target_times,
        "dt_min": dt_min,
        "kills": kills,
        "kills_in_window": in_window,
        "outside_total": outside_total,
        "stored_mode_count_peak_only": int(cls["mode_count"]),
        "edges": edges,
        "counts": counts,
        "variants": variants,
        "primary_tail_treatment": primary,
        "head_note": head_note,
    }


# ----------------------------------------------------------------------------
# Smearing + classification.
# ----------------------------------------------------------------------------


def smoothing_matrix(n: int, bin_width: float, bandwidth: float = BANDWIDTH) -> np.ndarray:
    """The classifier's normalised Gaussian smoothing matrix A (identical
    construction to ``exact_m_prr_mean_field_boundary.classify_real``)."""
    radius = max(1, int(math.ceil(4.0 * bandwidth / bin_width)))
    offsets = np.arange(-radius, radius + 1) * bin_width
    kernel = np.exp(-(offsets**2) / (2.0 * bandwidth**2))
    kernel /= kernel.sum()
    norm = np.convolve(np.ones(n), kernel, mode="same")
    amat = np.zeros((n, n))
    for i in range(n):
        for off, kv in zip(range(-radius, radius + 1), kernel):
            j = i - off
            if 0 <= j < n:
                amat[i, j] = kv / norm[i]
    return amat


def _variant_arrays(cell: dict, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = cell.setdefault("_variant_arrays", {})
    if variant not in cache:
        sources = cell["variants"][variant]["sources"]
        cache[variant] = (
            np.array([s[0] for s in sources]),
            np.array([s[1] for s in sources]),
            np.array([s[2] for s in sources]),
        )
    return cache[variant]


def smear_and_classify(cell: dict, variant: str, law: str, sigma: float) -> dict:
    """Smear the variant's sources, classify, and evaluate the prominence
    statistic both with the Poisson plug-in variance at the smeared counts
    (the article's convention) and with the Monte Carlo estimation variance of
    the smeared counts added (audit fix 2):

        var_P = sum_i C~_i c_i^2,   var_E = sum_s C_s (K^T c)_s^2,
        c = (A[peak] - A[base]) / (N dt),

    i.e. the full estimation covariance K diag(C) K^T of C~ propagated through
    the prominence functional.  At sigma_t = 0, K = I and var_E = var_P by
    construction (the smeared counts ARE the observed histogram), so the
    estimation fraction is informative only for sigma_t > 0."""
    src_lo, src_hi, src_c = _variant_arrays(cell, variant)
    edges = cell["edges"]
    walkers = cell["walkers"]
    kmat = transfer_matrix(edges, src_lo, src_hi, law, sigma)
    # einsum instead of BLAS matmul (Accelerate emits spurious FP warnings).
    smeared = np.einsum("is,s->i", kmat, src_c)
    result = mf.classify_real(
        smeared, edges, walkers,
        bandwidth=BANDWIDTH, sigma_factor=SIGMA_FACTOR, relative_floor=RELATIVE_FLOOR,
    )
    n = smeared.size
    bin_width = result["bin_width"]
    if "_A" not in cell:
        cell["_A"] = smoothing_matrix(n, bin_width)
    amat = cell["_A"]
    rows = []
    for r in result["rows"]:
        coeff = (amat[r["bin"]] - amat[r["base_bin"]]) / (walkers * bin_width)
        var_p = float(np.einsum("i,i->", smeared, coeff * coeff))
        g = np.einsum("is,i->s", kmat, coeff)
        var_e = float(np.einsum("s,s->", src_c, g * g))
        z_tot = r["prominence"] / math.sqrt(var_p + var_e) if var_p + var_e > 0 else math.inf
        floor_ok = bool(r["passes_relative_floor"])
        sig_ok = bool(r["z_covariance_aware"] >= SIGMA_FACTOR)
        rows.append({
            "time": r["time"],
            "base_time": r["base_time"],
            "relative_prominence": r["relative_prominence"],
            "z_covariance_aware": r["z_covariance_aware"],
            "z_with_estimation_variance": z_tot,
            "estimation_to_poisson_variance": var_e / var_p if var_p > 0 else None,
            "passes_relative_floor": floor_ok,
            "passes_sigma_gate": sig_ok,
            "significant_covariance_aware": bool(r["significant_covariance_aware"]),
            "significant_with_estimation_variance": bool(floor_ok and z_tot >= SIGMA_FACTOR),
        })
    counted = [r for r in rows if r["significant_covariance_aware"]]
    frac = [r["estimation_to_poisson_variance"] for r in counted
            if r["estimation_to_poisson_variance"] is not None]
    return {
        "sigma_t": float(sigma),
        "mode_count_covariance_aware": int(result["mode_count_covariance_aware"]),
        "mode_count_with_estimation_variance": int(
            sum(r["significant_with_estimation_variance"] for r in rows)),
        "mode_count_peak_only": int(result["mode_count_peak_only"]),
        "n_local_maxima": len(rows),
        "counted_times": [r["time"] for r in counted],
        "counted_z_covariance_aware": [r["z_covariance_aware"] for r in counted],
        "counted_relative_prominence": [r["relative_prominence"] for r in counted],
        "min_counted_z_covariance_aware": (
            min(r["z_covariance_aware"] for r in counted) if counted else None
        ),
        "min_counted_relative_prominence": (
            min(r["relative_prominence"] for r in counted) if counted else None
        ),
        "all_local_maxima": rows,
        "smeared_window_mass_per_walker": float(smeared.sum() / walkers),
        "estimation_to_poisson_prominence_variance_max_counted": (
            max(frac) if (frac and sigma > 0.0) else None
        ),
        "smeared_counts": smeared,
    }


def _intervals(ratios: list[float], flags: list[bool]) -> list[list[float]]:
    """Group consecutive True flags on the fine grid into [first, last] intervals."""
    out, start, prev = [], None, None
    for r, f in zip(ratios, flags):
        if f and start is None:
            start = r
        if not f and start is not None:
            out.append([start, prev])
            start = None
        prev = r
    if start is not None:
        out.append([start, prev])
    return out


def loss_reason(lo_row: dict, hi_row: dict, key: str) -> dict:
    """Why the mode count drops between the passing (lo) and failing (hi) end
    of the first-loss bracket: match every counted maximum at lo to the
    nearest local maximum at hi (within 0.1) and report which gate fails."""
    sig = f"significant_{key}"
    out = []
    his = hi_row["all_local_maxima"]
    for r in lo_row["all_local_maxima"]:
        if not r[sig]:
            continue
        near = min(his, key=lambda h: abs(h["time"] - r["time"])) if his else None
        if near is None or abs(near["time"] - r["time"]) > 0.1:
            out.append({"time_lo": r["time"], "reason": "local_maximum_disappears (merged)"})
            continue
        if near[sig]:
            continue
        floor_fail = not near["passes_relative_floor"]
        z = near["z_covariance_aware"] if key == "covariance_aware" else near["z_with_estimation_variance"]
        sig_fail = z < SIGMA_FACTOR
        reason = ("relative_floor_and_sigma_gate" if (floor_fail and sig_fail)
                  else "relative_floor" if floor_fail else "sigma_gate")
        out.append({
            "time_lo": r["time"], "time_hi": near["time"], "reason": reason,
            "relative_prominence_lo_hi": [r["relative_prominence"], near["relative_prominence"]],
            "z_lo_hi": [r["z_covariance_aware"] if key == "covariance_aware" else r["z_with_estimation_variance"], z],
            "estimation_to_poisson_variance_hi": near["estimation_to_poisson_variance"],
        })
    return {
        "lost_modes": out,
        "n_local_maxima_lo_hi": [lo_row["n_local_maxima"], hi_row["n_local_maxima"]],
    }


def _first_loss(cell: dict, variant: str, law: str, target: int,
                fine_rows: list[dict], key: str) -> dict:
    count_key = ("mode_count_covariance_aware" if key == "covariance_aware"
                 else "mode_count_with_estimation_variance")
    ratios = list(FINE_RATIOS)
    counts = [r[count_key] for r in fine_rows]
    ok = [c == target for c in counts]
    dt_min = cell["dt_min"]
    if not ok[0]:
        return {"status": "unsmeared_verdict_below_target", "fine_counts": counts}
    if all(ok):
        return {"status": "right_censored", "lower_bound": ratios[-1], "fine_counts": counts}
    k = ok.index(False)
    r_lo, r_hi = ratios[k - 1], ratios[k]
    lo_row, hi_row = fine_rows[k - 1], fine_rows[k]
    for _ in range(BISECT_ITERATIONS):
        mid = 0.5 * (r_lo + r_hi)
        out = smear_and_classify(cell, variant, law, mid * dt_min)
        if out[count_key] == target:
            r_lo, lo_row = mid, out
        else:
            r_hi, hi_row = mid, out
    reentry = [r for r, o in zip(ratios, ok) if o and r > ratios[k]]
    return {
        "status": "bisected",
        "definition": ("first loss: smallest sigma_t/dt_min at which the count drops "
                       "below m, located on the 0.01 fine grid and bisected "
                       f"{BISECT_ITERATIONS} times between the last passing and first "
                       "failing fine points"),
        "bracket": [r_lo, r_hi],
        "midpoint": 0.5 * (r_lo + r_hi),
        "sigma_t_midpoint": 0.5 * (r_lo + r_hi) * dt_min,
        "iterations": BISECT_ITERATIONS,
        "fine_counts": counts,
        "monotone_on_fine_grid": not reentry,
        "reentrance_fine_ratios": reentry,
        "reentrance_intervals": _intervals(
            [r for r in ratios if r > ratios[k]], [o for r, o in zip(ratios, ok) if r > ratios[k]]
        ),
        "largest_passing_fine_ratio": max(r for r, o in zip(ratios, ok) if o),
        "loss_reason": loss_reason(lo_row, hi_row, key),
    }


def survival_scan(cell: dict, variant: str, law: str, target: int) -> dict:
    """Coarse grid (reported), 0.01 fine pre-scan with first loss and
    re-entrance (primary boundary; audit fix 1), the same under the
    estimation-variance-augmented statistic (audit fix 2), and the legacy
    coarse-grid bisection kept for comparison."""
    dt_min = cell["dt_min"]
    grid = (0.0,) + RATIOS
    rows = []
    for ratio in grid:
        out = smear_and_classify(cell, variant, law, ratio * dt_min)
        out["ratio"] = float(ratio)
        rows.append(out)
    fine_rows = [smear_and_classify(cell, variant, law, r * dt_min) for r in FINE_RATIOS]
    for r, fr in zip(FINE_RATIOS, fine_rows):
        fr["ratio"] = float(r)
    passes = [r["mode_count_covariance_aware"] == target for r in rows]
    pass_ratios = [r["ratio"] for r, ok in zip(rows, passes) if ok]
    monotone = passes == sorted(passes, reverse=True)
    if not passes[0]:
        boundary = None
        status = "unsmeared_verdict_below_target"
        legacy = None
    elif all(passes):
        boundary = float(grid[-1])
        status = "right_censored"
        legacy = {"status": "right_censored", "lower_bound": float(grid[-1])}
    else:
        boundary = float(max(pass_ratios))
        status = "bracketed" if monotone else "bracketed_nonmonotone"
        r_lo = boundary
        r_hi = float(min(r["ratio"] for r in rows if r["ratio"] > r_lo
                         and r["mode_count_covariance_aware"] != target))
        for _ in range(BISECT_ITERATIONS):
            mid = 0.5 * (r_lo + r_hi)
            out = smear_and_classify(cell, variant, law, mid * dt_min)
            if out["mode_count_covariance_aware"] == target:
                r_lo = mid
            else:
                r_hi = mid
        legacy = {
            "status": "bisected",
            "bracket": [r_lo, r_hi],
            "midpoint": 0.5 * (r_lo + r_hi),
            "sigma_t_midpoint": 0.5 * (r_lo + r_hi) * dt_min,
            "iterations": BISECT_ITERATIONS,
            "note": ("LEGACY (pre-2026-09-23): bisection between the largest passing "
                     "coarse-grid ratio and the next failing one; misses the first "
                     "loss when the count re-enters (use first_loss)"),
        }
    first = _first_loss(cell, variant, law, target, fine_rows, "covariance_aware")
    first_tot = _first_loss(cell, variant, law, target, fine_rows, "with_estimation_variance")
    fracs = [r["estimation_to_poisson_prominence_variance_max_counted"] for r in rows + fine_rows
             if r["estimation_to_poisson_prominence_variance_max_counted"] is not None]
    return {
        "law": law,
        "tail_treatment": variant,
        "target_mode_count": target,
        "grid_ratios": [float(g) for g in grid],
        "grid_mode_counts": [r["mode_count_covariance_aware"] for r in rows],
        "grid_mode_counts_peak_only": [r["mode_count_peak_only"] for r in rows],
        "grid_pass": passes,
        "monotone": monotone,
        "survival_boundary_grid": boundary,
        "survival_status": status,
        "fine_ratios": list(FINE_RATIOS),
        "fine_mode_counts": [r["mode_count_covariance_aware"] for r in fine_rows],
        "fine_mode_counts_with_estimation_variance": [
            r["mode_count_with_estimation_variance"] for r in fine_rows],
        "fine_n_local_maxima": [r["n_local_maxima"] for r in fine_rows],
        "first_loss": first,
        "first_loss_with_estimation_variance": first_tot,
        "estimation_to_poisson_prominence_variance_max_counted_sigma_positive": (
            max(fracs) if fracs else None),
        "refined_boundary": first if first.get("status") == "bisected" else (
            legacy if legacy else first),
        "refined_boundary_legacy_coarse_bisection": legacy,
        "rows": rows,
    }


# ----------------------------------------------------------------------------
# Self-tests (asserted before any result is written).
# ----------------------------------------------------------------------------


def self_tests(cells: list[dict]) -> dict:
    report: dict = {}
    # (1) Phi_2' == CDF for both laws (central differences).
    xs = np.linspace(-2.0, 2.0, 41)
    hstep = 1e-5
    worst = 0.0
    for law in LAWS:
        for sigma in (0.05, 0.3, 1.0):
            num = (law_second_antiderivative(xs + hstep, law, sigma)
                   - law_second_antiderivative(xs - hstep, law, sigma)) / (2 * hstep)
            worst = max(worst, float(np.max(np.abs(num - law_cdf(xs, law, sigma)))))
    if worst > 1e-6:
        raise AssertionError(f"second antiderivative inconsistent with CDF ({worst:.2e})")
    report["antiderivative_vs_cdf_max_abs"] = worst

    # (2) Mass conservation: every source column sums to one on a wide grid.
    wide = np.arange(-20.0, 24.0 + 1e-9, 0.02)
    src_lo = np.array([1.0, 3.5, 0.48])
    src_hi = np.array([1.02, 3.52, 0.50])
    worst = 0.0
    for law in LAWS:
        for sigma in (0.0, 0.024, 0.3, 1.5):
            kmat = transfer_matrix(wide, src_lo, src_hi, law, sigma)
            worst = max(worst, float(np.max(np.abs(kmat.sum(axis=0) - 1.0))))
    if worst > 1e-9:
        raise AssertionError(f"mass conservation violated ({worst:.2e})")
    report["column_sum_deviation_max"] = worst

    # (3) Monte Carlo check of one transfer column per law (deterministic seed).
    rng = np.random.default_rng(SELF_TEST_SEED)
    edges = cells[0]["edges"]
    mc = {}
    for law in LAWS:
        sigma = 0.3
        a, b = 1.0, 1.02
        t0 = rng.uniform(a, b, SELF_TEST_SAMPLES)
        if law == "gaussian":
            s = rng.normal(0.0, sigma, SELF_TEST_SAMPLES)
        else:
            w = math.sqrt(3.0) * sigma
            s = rng.uniform(-w, w, SELF_TEST_SAMPLES)
        hist, _ = np.histogram(t0 + s, bins=edges)
        phat = hist / SELF_TEST_SAMPLES
        kcol = transfer_matrix(edges, np.array([a]), np.array([b]), law, sigma)[:, 0]
        sd = np.sqrt(np.maximum(kcol * (1.0 - kcol), 1e-12) / SELF_TEST_SAMPLES)
        zmax = float(np.max(np.abs(phat - kcol) / sd))
        if zmax > 5.5:
            raise AssertionError(f"Monte Carlo check of K failed for {law} (z={zmax:.2f})")
        mc[law] = {"sigma_t": sigma, "samples": SELF_TEST_SAMPLES, "max_abs_z": zmax}
    report["monte_carlo_column_check"] = mc

    # (4) sigma_t = 0 reproduces the stored covariance-aware classification.
    worst_z = 0.0
    for cell in cells:
        ref = recl.classify_both(
            cell["counts"], cell["edges"], cell["walkers"],
            bandwidth=BANDWIDTH, sigma_factor=SIGMA_FACTOR, relative_floor=RELATIVE_FLOOR,
        )
        for variant in cell["variants"]:
            out = smear_and_classify(cell, variant, "gaussian", 0.0)
            if not np.allclose(out["smeared_counts"], cell["counts"], rtol=0, atol=1e-9):
                raise AssertionError(f"{cell['name']}/{variant}: sigma_t=0 is not the identity")
            if out["mode_count_covariance_aware"] != ref["mode_count_covariance_aware"]:
                raise AssertionError(f"{cell['name']}/{variant}: sigma_t=0 verdict mismatch")
            z_ref = [r["z_covariance_aware"] for r in ref["rows"]]
            z_new = [r["z_covariance_aware"] for r in out["all_local_maxima"]]
            if len(z_ref) != len(z_new):
                raise AssertionError(f"{cell['name']}/{variant}: local-maximum count mismatch")
            for zr, zn in zip(z_ref, z_new):
                if math.isfinite(zr):
                    worst_z = max(worst_z, abs(zr - zn) / max(zr, 1e-300))
        if ref["mode_count_peak_only"] != cell["stored_mode_count_peak_only"]:
            raise AssertionError(f"{cell['name']}: stored peak-only mode count not reproduced")
    if worst_z > 1e-9:
        raise AssertionError(f"sigma_t=0 z mismatch ({worst_z:.2e})")
    report["sigma_zero_identity_max_rel_z_diff"] = worst_z
    report["cells_checked"] = len(cells)
    return report


# ----------------------------------------------------------------------------
# Figure.
# ----------------------------------------------------------------------------


def make_figure(cell_payloads: list[dict]) -> list[str]:
    """Counted modes against sigma_t/dt_min as STEP functions on the 0.01 fine
    grid (audit fix 6; the old figure joined coarse-grid counts with slanted
    lines), coarse-grid points as markers, first loss as a dotted line."""
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt

    n = len(cell_payloads)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.25), constrained_layout=True,
                             sharex=True)
    style = {
        "uniform": {"color": core.OI_BLUE, "ls": "-", "marker": "o", "label": "uniform",
                    "dy": 0.06},
        "gaussian": {"color": core.OI_VERMILLION, "ls": "--", "marker": "s",
                     "label": "Gaussian", "dy": -0.06},
    }
    for ax, cp in zip(np.atleast_1d(axes), cell_payloads):
        m = cp["m"]
        primary = cp["primary_tail_treatment"]
        for law in LAWS:
            scan = cp["scans"][primary][law]
            st = style[law]
            fr = np.asarray(scan["fine_ratios"])
            fc = np.asarray(scan["fine_mode_counts"], dtype=float) + st["dy"]
            ax.step(fr, fc, where="post", color=st["color"], ls=st["ls"], lw=1.1,
                    label=st["label"])
            ax.plot(scan["grid_ratios"], np.asarray(scan["grid_mode_counts"]) + st["dy"],
                    ls="none", marker=st["marker"], ms=3.2, color=st["color"],
                    markerfacecolor="white" if law == "gaussian" else st["color"],
                    markeredgewidth=0.9)
            fl = scan.get("first_loss") or {}
            if fl.get("status") == "bisected":
                ax.axvline(fl["midpoint"], color=st["color"], ls=":", lw=0.8, alpha=0.9)
        ax.axhline(m, color="0.5", lw=0.6, ls=(0, (1, 2)))
        ax.set_ylim(-0.4, m + 0.6)
        ax.set_yticks(range(0, m + 1))
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
        ax.set_xlabel(r"$\sigma_t/\Delta t_{\min}$")
        ax.set_title(f"{cp['label']}\n" + rf"$\Delta t_{{\min}}={cp['dt_min']:.2f}$",
                     fontsize=7.5)
        ax.grid(True, lw=0.4, alpha=0.4)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("counted modes")
    axes[0].legend(loc="lower left", fontsize=7.0, handlelength=2.2)
    written = core.save_figure(fig, FIGURE_STEM)
    plt.close(fig)
    return written


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------


def _cell_list(only_headline: bool) -> list[tuple[str, Path, str]]:
    cells = list(HEADLINE_CELLS)
    if only_headline:
        return cells
    known = {c[0] for c in cells}
    for path in sorted(PRODUCTION_DIR.glob("*.json")):
        if path.stem not in known:
            cells.append((path.stem, path, path.stem))
    return cells


def _strip_arrays(scan: dict) -> dict:
    out = json.loads(json.dumps({k: v for k, v in scan.items() if k != "rows"}))
    out["rows"] = []
    for r in scan["rows"]:
        row = {k: v for k, v in r.items() if k != "smeared_counts"}
        out["rows"].append(row)
    return out


def _deviation(scans: dict, primary: str, law: str) -> dict:
    """Largest relative change of the weakest counted z and verdict changes
    between the primary tail treatment and the other variants."""
    ref = scans[primary][law]
    out = {}
    for variant, per_law in scans.items():
        if variant == primary:
            continue
        alt = per_law[law]
        verdict_diffs = sum(
            1 for a, b in zip(ref["grid_mode_counts"], alt["grid_mode_counts"]) if a != b
        )
        rel = []
        for ra, rb in zip(ref["rows"], alt["rows"]):
            za, zb = ra["min_counted_z_covariance_aware"], rb["min_counted_z_covariance_aware"]
            if za and zb:
                rel.append(abs(zb - za) / za)
        rb_ref = ref["refined_boundary"]
        rb_alt = alt["refined_boundary"]
        bdiff = None
        if (rb_ref and rb_alt and rb_ref.get("status") == "bisected"
                and rb_alt.get("status") == "bisected"):
            bdiff = rb_alt["midpoint"] - rb_ref["midpoint"]
        out[variant] = {
            "grid_verdict_differences": verdict_diffs,
            "max_rel_change_min_counted_z": max(rel) if rel else None,
            "survival_boundary_grid_alt": alt["survival_boundary_grid"],
            "refined_boundary_difference": bdiff,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--headline-only", action="store_true",
                        help="process only the four headline cells")
    parser.add_argument("--replot", action="store_true",
                        help="regenerate the figure from the stored cell JSONs")
    args = parser.parse_args()
    started = time.perf_counter()

    if args.replot:
        payloads = [_read(OUT_DIR / f"cell_{name}.json") for name, _, _ in HEADLINE_CELLS]
        for path in make_figure(payloads):
            print(f"figure -> {path}", flush=True)
        return

    specs = _cell_list(args.headline_only)
    cells = [load_cell(name, path, label) for name, path, label in specs]
    print(f"loaded {len(cells)} cells", flush=True)
    tests = self_tests(cells)
    print(f"self-tests passed: {json.dumps(tests)}", flush=True)

    cell_payloads = []
    for cell in cells:
        t0 = time.perf_counter()
        scans = {}
        for variant in cell["variants"]:
            scans[variant] = {law: survival_scan(cell, variant, law, cell["m"])
                              for law in LAWS}
        primary = cell["primary_tail_treatment"]
        unsmeared = scans[primary]["gaussian"]["rows"][0]["mode_count_covariance_aware"]
        # Secondary boundary: retention of the unsmeared covariance-aware count
        # (informative for cells whose unsmeared verdict is already below m).
        secondary = {}
        if unsmeared != cell["m"]:
            secondary = {law: survival_scan(cell, primary, law, unsmeared) for law in LAWS}
        payload = {
            "name": cell["name"],
            "label": cell["label"],
            "file": cell["file"],
            "stream": cell["stream"],
            "m": cell["m"],
            "eps": cell["eps"],
            "budget": cell["budget"],
            "weights": cell["weights"],
            "dim": cell["dim"],
            "walkers": cell["walkers"],
            "dt": cell["dt"],
            "seed": cell["seed"],
            "target_times": cell["target_times"],
            "dt_min": cell["dt_min"],
            "kills": cell["kills"],
            "kills_in_window": cell["kills_in_window"],
            "outside_total": cell["outside_total"],
            "head_note": cell["head_note"],
            "stored_mode_count_peak_only": cell["stored_mode_count_peak_only"],
            "unsmeared_mode_count_covariance_aware": unsmeared,
            "primary_tail_treatment": primary,
            "tail_treatments": {
                v: {k: val for k, val in d.items() if k != "sources"}
                for v, d in cell["variants"].items()
            },
            "scans": {v: {law: _strip_arrays(s) for law, s in per.items()}
                      for v, per in scans.items()},
            "secondary_scans_unsmeared_count": {
                law: _strip_arrays(s) for law, s in secondary.items()
            },
            "primary_smeared_counts": {
                law: {str(r["ratio"]): [float(v) for v in r["smeared_counts"]]
                      for r in scans[primary][law]["rows"]}
                for law in LAWS
            },
            "tail_treatment_deviation_from_primary": {
                law: _deviation(scans, primary, law) for law in LAWS
            },
            "seconds": time.perf_counter() - t0,
        }
        core.write_json(OUT_DIR / f"cell_{cell['name']}.json", payload)
        cell_payloads.append(payload)
        summary_line = ", ".join(
            f"{law}: grid={scans[primary][law]['survival_boundary_grid']} "
            f"refined={(scans[primary][law]['refined_boundary'] or {}).get('midpoint')}"
            for law in LAWS
        )
        print(f"{cell['name']}: m={cell['m']} dt_min={cell['dt_min']:.4f} "
              f"[{primary}] {summary_line} ({payload['seconds']:.1f}s)", flush=True)

    # Validation of the exponential fill against the exact retained tails.
    fill_validation = []
    for cp in cell_payloads:
        if "retained" not in cp["scans"]:
            continue
        for law in LAWS:
            dev = cp["tail_treatment_deviation_from_primary"][law]["exponential_fill"]
            fill_validation.append({"cell": cp["name"], "law": law, **dev})
    window_only_validation = []
    for cp in cell_payloads:
        for law in LAWS:
            dev = cp["tail_treatment_deviation_from_primary"][law]["window_only"]
            window_only_validation.append({"cell": cp["name"], "law": law, **dev})

    def _agg(rows: list[dict]) -> dict:
        zs = [r["max_rel_change_min_counted_z"] for r in rows
              if r["max_rel_change_min_counted_z"] is not None]
        bd = [abs(r["refined_boundary_difference"]) for r in rows
              if r["refined_boundary_difference"] is not None]
        return {
            "n_comparisons": len(rows),
            "grid_verdict_differences_total": sum(r["grid_verdict_differences"] for r in rows),
            "max_rel_change_min_counted_z": max(zs) if zs else None,
            "max_abs_refined_boundary_difference": max(bd) if bd else None,
        }

    table = []
    for cp in cell_payloads:
        primary = cp["primary_tail_treatment"]
        row = {
            "name": cp["name"],
            "m": cp["m"],
            "eps": cp["eps"],
            "budget": cp["budget"],
            "weights": cp["weights"],
            "dim": cp["dim"],
            "walkers": cp["walkers"],
            "dt_min": cp["dt_min"],
            "unsmeared_mode_count_covariance_aware": cp["unsmeared_mode_count_covariance_aware"],
            "primary_tail_treatment": primary,
        }
        for law in LAWS:
            scan = cp["scans"][primary][law]
            ref = scan["refined_boundary"] or {}
            row[law] = {
                "grid_mode_counts": scan["grid_mode_counts"],
                "survival_boundary_grid": scan["survival_boundary_grid"],
                "survival_status": scan["survival_status"],
                "monotone": scan["monotone"],
                "refined_boundary_midpoint": ref.get("midpoint"),
                "refined_boundary_bracket": ref.get("bracket"),
                "refined_sigma_t": ref.get("sigma_t_midpoint"),
                "min_counted_z_by_ratio": [
                    r["min_counted_z_covariance_aware"] for r in scan["rows"]
                ],
                "min_counted_relative_prominence_by_ratio": [
                    r["min_counted_relative_prominence"] for r in scan["rows"]
                ],
                "counted_times_by_ratio": [r["counted_times"] for r in scan["rows"]],
                "smeared_window_mass_by_ratio": [
                    r["smeared_window_mass_per_walker"] for r in scan["rows"]
                ],
                "estimation_to_poisson_prominence_variance_max_counted_sigma_positive": scan[
                    "estimation_to_poisson_prominence_variance_max_counted_sigma_positive"],
                "first_loss": {k: v for k, v in scan["first_loss"].items() if k != "fine_counts"},
                "first_loss_with_estimation_variance": {
                    k: v for k, v in scan["first_loss_with_estimation_variance"].items()
                    if k not in ("fine_counts", "loss_reason")},
                "fine_mode_counts": scan["fine_mode_counts"],
                "refined_boundary_legacy_coarse_bisection": (
                    (scan["refined_boundary_legacy_coarse_bisection"] or {}).get("midpoint")),
                "estimation_to_poisson_counted_at_ratio_0.3": [
                    r["estimation_to_poisson_variance"]
                    for r in next(x for x in scan["rows"] if abs(x["ratio"] - 0.3) < 1e-12)["all_local_maxima"]
                    if r["significant_covariance_aware"]],
                "tail_extension_first_loss": {
                    v: (cp["scans"][v][law]["first_loss"].get("midpoint"))
                    for v in ("tail_extension_exp", "tail_extension_flat")
                },
            }
            if cp["secondary_scans_unsmeared_count"]:
                sec = cp["secondary_scans_unsmeared_count"][law]
                row[law]["unsmeared_count_survival_boundary_grid"] = sec["survival_boundary_grid"]
                row[law]["unsmeared_count_refined_midpoint"] = (
                    (sec["refined_boundary"] or {}).get("midpoint")
                )
        table.append(row)

    headline_names = [c[0] for c in HEADLINE_CELLS]
    headline = {}
    for row in table:
        if row["name"] in headline_names:
            headline[row["name"]] = {
                law: {
                    "survival_boundary_grid": row[law]["survival_boundary_grid"],
                    "refined_boundary_midpoint": row[law]["refined_boundary_midpoint"],
                    "refined_sigma_t": row[law]["refined_sigma_t"],
                    "grid_mode_counts": row[law]["grid_mode_counts"],
                    "first_loss_bracket": row[law]["first_loss"].get("bracket"),
                    "reentrance_intervals": row[law]["first_loss"].get("reentrance_intervals"),
                }
                for law in LAWS
            }
    # Audit aggregates (2026-09-23 S3 audit, fixes 1-5) over the cells whose
    # unsmeared covariance-aware count equals m.
    passing = [row for row in table if row["unsmeared_mode_count_covariance_aware"] == row["m"]]
    audit = {"n_passing_cells": len(passing),
             "excluded_cells_unsmeared_below_m": [
                 row["name"] for row in table
                 if row["unsmeared_mode_count_covariance_aware"] != row["m"]]}
    for law in LAWS:
        fl = [(row["name"], row[law]["first_loss"].get("midpoint")) for row in passing]
        fl_ok = [v for _, v in fl if v is not None]
        tot = [row[law]["first_loss_with_estimation_variance"].get("midpoint") for row in passing]
        d_tot = [abs(a - b) for (_, a), b in zip(fl, tot) if a is not None and b is not None]
        d_ext = []
        for row in passing:
            base_v = row[law]["first_loss"].get("midpoint")
            for v in row[law]["tail_extension_first_loss"].values():
                if base_v is not None and v is not None:
                    d_ext.append(abs(v - base_v))
        legacy_diff = [
            {"cell": row["name"], "first_loss": row[law]["first_loss"].get("midpoint"),
             "legacy_coarse_bisection": row[law]["refined_boundary_legacy_coarse_bisection"]}
            for row in passing
            if row[law]["refined_boundary_legacy_coarse_bisection"] is not None
            and row[law]["first_loss"].get("midpoint") is not None
            and abs(row[law]["refined_boundary_legacy_coarse_bisection"]
                    - row[law]["first_loss"]["midpoint"]) > 1e-3
        ]
        reasons: dict[str, int] = {}
        z_at_loss, relp_at_loss, est_at_loss = [], [], []
        for row in passing:
            lr = row[law]["first_loss"].get("loss_reason") or {}
            for lost in lr.get("lost_modes", []):
                reasons[lost["reason"]] = reasons.get(lost["reason"], 0) + 1
                if "z_lo_hi" in lost:
                    z_at_loss.append(lost["z_lo_hi"][1])
                    relp_at_loss.append(lost["relative_prominence_lo_hi"][1])
                    if lost.get("estimation_to_poisson_variance_hi") is not None:
                        est_at_loss.append(lost["estimation_to_poisson_variance_hi"])
        fracs = [row[law]["estimation_to_poisson_prominence_variance_max_counted_sigma_positive"]
                 for row in passing
                 if row[law]["estimation_to_poisson_prominence_variance_max_counted_sigma_positive"] is not None]
        audit[law] = {
            "first_loss_by_cell": dict(fl),
            "first_loss_range": [min(fl_ok), max(fl_ok)] if fl_ok else None,
            "all_pass_at_ratio": {
                str(r): all(row[law]["fine_mode_counts"][FINE_RATIOS.index(r)] == row["m"]
                            for row in passing) for r in (0.1, 0.2, 0.3)},
            "nonmonotone_cells": {
                row["name"]: row[law]["first_loss"].get("reentrance_intervals")
                for row in passing if row[law]["first_loss"].get("reentrance_fine_ratios")},
            "legacy_coarse_bisection_differs_from_first_loss": legacy_diff,
            "max_abs_first_loss_change_with_estimation_variance": max(d_tot) if d_tot else None,
            "max_abs_first_loss_change_tail_extension_to_t10": max(d_ext) if d_ext else None,
            "loss_reason_counts": reasons,
            "z_covariance_aware_of_lost_mode_at_first_loss_range": (
                [min(z_at_loss), max(z_at_loss)] if z_at_loss else None),
            "relative_prominence_of_lost_mode_at_first_loss_range": (
                [min(relp_at_loss), max(relp_at_loss)] if relp_at_loss else None),
            "estimation_to_poisson_prominence_variance_max_counted_range": (
                [min(fracs), max(fracs)] if fracs else None),
            "estimation_to_poisson_variance_of_lost_mode_at_first_loss_range": (
                [min(est_at_loss), max(est_at_loss)] if est_at_loss else None),
            "estimation_to_poisson_variance_counted_at_ratio_0.3_range": (
                lambda v: [min(v), max(v)] if v else None)(
                [x for row in passing for x in row[law]["estimation_to_poisson_counted_at_ratio_0.3"]]),
        }
    figure_paths = make_figure([cp for cp in cell_payloads if cp["name"] in headline_names])

    summary = {
        "analysis": (
            "release-time randomisation robustness: stored reaction-time histograms "
            "convolved with zero-mean uniform and Gaussian release-time laws of sd "
            "sigma_t, classified by the covariance-aware prominence rule on the smeared "
            "expected window counts; deterministic post-processing, no new walkers"
        ),
        "definitions": {
            "observed_density": "f_obs(t) = int f(t - t0) rho(t0) dt0, release time t0 ~ rho independent of the dynamics, reaction time measured from the nominal clock",
            "laws": {
                "uniform": "U(-sqrt(3) sigma_t, +sqrt(3) sigma_t), zero mean, sd sigma_t",
                "gaussian": "N(0, sigma_t^2)",
                "centring": "a nonzero mean of rho is a rigid translation t_j -> t_j + E[t0], not a smearing; laws are centred",
            },
            "sigma_grid": "sigma_t = ratio * dt_min, ratio in grid_ratios; dt_min = smallest adjacent spacing of the prescribed target times of the cell",
            "smearing_operator": "C~_i = sum_s K_is C_s, K([c,d]<-[a,b]) = [Phi_2(d-a) - Phi_2(d-b) - Phi_2(c-a) + Phi_2(c-b)]/(b-a), Phi_2'' = rho; exact bin-to-bin transfer for an event uniformly distributed inside its source bin; targets are the 150 window bins of width 0.02 on I=[0.5,3.5]; sources are all retained bins (window plus outside mass per tail treatment); events after t_max=4 are unrecorded and contribute nothing",
            "classifier": "reclassify_covariance_aware.classify_both algorithm on real-valued counts (exact_m_prr_mean_field_boundary.classify_real): bandwidth 0.04, z_prom >= 5 with covariance-aware sigma_prom^2 = sum_j (A_peak,j - A_base,j)^2 C~_j/(N dt)^2, prominence >= 5% of the window maximum",
            "statistic": "z evaluated as for an N-walker experiment with jittered release whose observed counts are Poisson with mean C~ (Poisson plug-in variance at the smeared expected counts); the variant with the Monte Carlo estimation covariance of C~ added is reported separately (first_loss_with_estimation_variance)",
            "tail_treatments": {
                "retained": "production records: full-range 0.04-bin histogram bins outside I used exactly (straddling bins split against the window counts)",
                "exponential_fill": "records without a full-range histogram (W4 m=5, W5 d=3): tail on (3.5,4] filled with A exp(-lambda (t-3.5)), A from a log-linear fit of the last five window bins, lambda from the retained outside total kills - kills_in_window; head assumed empty (first window bins are empty); validated against 'retained' on the 18 production cells",
                "window_only": "control: outside mass set to zero",
            },
            "primary_tail_treatment": "retained where available, otherwise exponential_fill",
            "survival_boundary_grid": "largest grid ratio sigma_t/dt_min at which the covariance-aware mode count equals the prescribed m (the unsmeared ratio 0 is part of the grid)",
            "refined_boundary": "FIRST LOSS (since 2026-09-23): smallest ratio at which the count drops below m, located on a 0.01 fine grid of ratio in [0,1] and bisected 12 times between the last passing and the first failing fine point; midpoint quoted.  The pre-2026-09-23 value (bisection between the largest passing and next failing COARSE grid point) is kept as refined_boundary_legacy_coarse_bisection; it is not the first loss when the count re-enters (e.g. m3_eps0.1_B1_w50-30-20 uniform: legacy 0.4323, first loss 0.3312)",
            "reentrance": "fine-grid ratios above the first loss at which the count is again m (non-monotone survival); reported, not used for the boundary",
            "loss_reason": "at the first-loss bracket, each counted maximum at the passing end is matched to the nearest local maximum at the failing end; reason = relative_floor / sigma_gate / both / local maximum disappears",
            "estimation_variance_variant": "first_loss_with_estimation_variance: z = prominence / sqrt(var_P + var_E), var_E = sum_s C_s (K^T c)_s^2 = c^T K diag(C) K^T c, the full Monte Carlo estimation covariance of the smeared counts propagated through the prominence functional; the ratio var_E/var_P of counted maxima is reported for sigma_t > 0 only (at sigma_t = 0, K = I and var_E = var_P by construction)",
            "tail_extension_variants": "tail_extension_exp / tail_extension_flat: primary sources plus a continuation of the density on (t_max, 10] (exponential from a log-linear fit of the last five 0.04 bins of the full-range histogram, or of the (3.5,4] fill; flat at the t_max level); bounds the effect of the unrecorded events after t_max",
            "pass_rule": "mode_count_covariance_aware == m",
        },
        "grid_ratios": [0.0] + list(RATIOS),
        "bandwidth": BANDWIDTH,
        "prominence_sigma_factor": SIGMA_FACTOR,
        "prominence_relative_floor": RELATIVE_FLOOR,
        "self_tests": tests,
        "audit_2026_09_23": audit,
        "fine_ratios": list(FINE_RATIOS),
        "n_cells": len(table),
        "headline_cells": headline_names,
        "headline": headline,
        "cells": table,
        "tail_treatment_validation": {
            "exponential_fill_vs_retained_on_production_cells": _agg(fill_validation),
            "window_only_vs_primary_all_cells": _agg(window_only_validation),
            "exponential_fill_rows": fill_validation,
            "window_only_rows": window_only_validation,
        },
        "figure": figure_paths,
        "cell_files": [str((OUT_DIR / f"cell_{cp['name']}.json").relative_to(core.REPORT))
                       for cp in cell_payloads],
        "reproduction_command": "python3 code/exact_m_prr_release_time_smearing.py",
        "wall_seconds": time.perf_counter() - started,
    }
    core.write_json(SUMMARY_OUT, summary)
    print(f"summary -> {SUMMARY_OUT}", flush=True)
    for path in figure_paths:
        print(f"figure -> {path}", flush=True)
    print(json.dumps(headline, indent=1))
    print(f"DONE in {summary['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
