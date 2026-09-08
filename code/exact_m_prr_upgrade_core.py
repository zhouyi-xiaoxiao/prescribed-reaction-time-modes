#!/usr/bin/env python3
"""Shared core for the exact-m PRR upgrade numerics campaign (W1..W5).

Reuses the production-proven off-lattice Euler-Maruyama machinery from
``validate_exact_m_offlattice.py`` (imported, never modified: that module is
hash-frozen) and generalizes it along three axes needed by the campaign:

  * explicit slab centres in Z space (W3 centre jitter, W4 stretched design),
  * arbitrary z0 via dataclasses.replace on ModelParameters (W4),
  * a second wrapped transverse coordinate, n_perp = 2, i.e. d = 3 (W5),
    with killing prefactor B / W^(d-1) = B / W^2 per the spine.

The standard d=2 paths (W1 phase diagram, W2 bisection probes) go through the
validated ``base.run_config`` unchanged.

Validation gates for every new config type (asserted in-run):
  (1) B = 0 kills nothing from a contact-sampling start (gate_zero_general);
  (2) per-step kill probability in [0, 1];
  (3) mass balance kills + survivors == walkers.
The d=3 code path additionally gets one dt-halving gate (W5 driver).

Deterministic seeds: one SeedSequence entropy list per (seed, tag, config)
including the explicit centres, weights, z0, n_perp, dt, so every stream is
reproducible and disjoint from the production streams (campaign seed differs).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import validate_exact_m_offlattice as base  # noqa: E402  (validated, frozen)

REPORT = HERE.parents[1]
UPGRADE_DATA = REPORT / "artifacts" / "data" / "exact_m_prr_upgrade"
FIGURES = REPORT / "artifacts" / "figures"

CAMPAIGN_SEED = 20260813

# Stream tags (SeedSequence entropy component; disjoint from base TAG_* 1..3).
TAG_W1_GRID = 11
TAG_W1_REFINE = 12
TAG_W2_PROBE = 21
TAG_W3_CENTRE = 31
TAG_W3_WEIGHT = 32
TAG_W3_GATE = 33
TAG_W3_DRAWS = 34
TAG_W4_MAIN = 41
TAG_W4_GATE = 42
TAG_W5_MAIN = 51
TAG_W5_GATE = 52
TAG_W5_DTCHECK = 53

MODEL = base.MODEL
WINDOW = base.WINDOW
TARGET_TIMES = base.TARGET_TIMES

GATE_ZERO_WALKERS = 20_000
GATE_ZERO_STEPS = 500


def centres_z_for(m: int, p=MODEL, target_times=None) -> np.ndarray:
    """Physical slab centres mu(t_j) for the target times."""
    times = TARGET_TIMES[m] if target_times is None else tuple(target_times)
    return np.array([float(base.mu_of_t(tj, p)) for tj in times])


# ----------------------------------------------------------------------------
# Generalized simulation core (explicit centres, n_perp transverse coords).
# ----------------------------------------------------------------------------


def simulate_chunk_general(
    rng: np.random.Generator,
    count: int,
    *,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    centres_z: np.ndarray,
    dt: float,
    step_count: int,
    p=MODEL,
    n_perp: int = 1,
) -> dict:
    """EM evolution of one walker chunk; mirrors base.simulate_chunk exactly
    for n_perp=1 and standard centres, generalized to explicit centres_z and
    n_perp wrapped transverse coordinates (d = n_perp + 1)."""

    mu_j = np.asarray(centres_z, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    slab_norm = 1.0 / (math.sqrt(2.0 * math.pi) * eps * p.rho)
    inv_two_var = 1.0 / (2.0 * (eps * p.rho) ** 2)
    rate_prefactor = budget / p.torus_w**n_perp  # B / W^(d-1)

    decay = 1.0 - p.gamma * dt
    pull = p.gamma * p.z_bar * dt
    noise_z = eps * math.sqrt(p.d0 * dt)
    noise_r = 2.0 * eps * math.sqrt(p.d0 * dt)
    contact_sq = p.contact_a**2
    torus_w = p.torus_w

    z = p.z0 + math.sqrt(eps**2 * p.d0 / (2.0 * p.gamma)) * rng.standard_normal(count)
    r_par = p.r_par0 + eps * p.u0 * rng.standard_normal(count)
    r_perp = np.mod(
        p.r_perp0 + eps * p.sigma_perp0 * rng.standard_normal((n_perp, count)),
        torus_w,
    )

    kill_time_blocks: list[np.ndarray] = []
    contact_steps = 0
    kill_probability_max = 0.0
    walker_steps = 0

    for step in range(step_count):
        alive = z.size
        if alive == 0:
            break
        walker_steps += alive
        noise = rng.standard_normal((2 + n_perp, alive))
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        r_perp += noise_r * noise[2:]
        np.mod(r_perp, torus_w, out=r_perp)

        perp_mi = np.minimum(r_perp, torus_w - r_perp)
        dist_sq = r_par * r_par + np.sum(perp_mi * perp_mi, axis=0)
        contact_index = np.flatnonzero(dist_sq < contact_sq)
        if contact_index.size == 0:
            continue
        contact_steps += int(contact_index.size)
        if rate_prefactor == 0.0:
            continue
        z_contact = z[contact_index]
        kappa = np.zeros(contact_index.size)
        for weight, centre in zip(w_arr, mu_j):
            diff = z_contact - centre
            kappa += weight * np.exp(-diff * diff * inv_two_var)
        kappa *= rate_prefactor * slab_norm
        kill_probability = -np.expm1(-kappa * dt)
        step_max = float(kill_probability.max())
        if step_max > kill_probability_max:
            kill_probability_max = step_max
        uniforms = rng.random(contact_index.size)
        killed_local = uniforms < kill_probability
        if not killed_local.any():
            continue
        killed_index = contact_index[killed_local]
        kill_time_blocks.append(
            np.full(killed_index.size, (step + 1) * dt, dtype=float)
        )
        keep = np.ones(alive, dtype=bool)
        keep[killed_index] = False
        z = z[keep]
        r_par = r_par[keep]
        r_perp = r_perp[:, keep]

    return {
        "kill_times": (
            np.concatenate(kill_time_blocks)
            if kill_time_blocks
            else np.empty(0, dtype=float)
        ),
        "survivors": int(z.size),
        "contact_steps": int(contact_steps),
        "kill_probability_max": float(kill_probability_max),
        "walker_steps": int(walker_steps),
    }


def _general_entropy(
    seed: int,
    tag: int,
    *,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    centres_z: np.ndarray,
    dt: float,
    z0: float,
    n_perp: int,
    extra: tuple[int, ...] = (),
) -> list[int]:
    # SeedSequence entropy must be non-negative; jittered centres can be
    # negative, so encode every component modulo 2**63 (identity for the
    # non-negative values used by all fixed-design streams).
    mod = 1 << 63
    entropy = [
        int(seed) % mod,
        int(tag) % mod,
        int(n_perp) % mod,
        int(round(eps * 1e9)) % mod,
        int(round(budget * 1e9)) % mod,
        int(round(dt * 1e15)) % mod,
        int(round(z0 * 1e9)) % mod,
    ]
    entropy.extend(int(round(w * 1e9)) % mod for w in weights)
    entropy.extend(
        int(round(c * 1e9)) % mod for c in np.asarray(centres_z, float)
    )
    entropy.extend(int(v) % mod for v in extra)
    return entropy


def run_config_general(
    *,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    centres_z: np.ndarray,
    walkers: int,
    chunk: int,
    dt: float,
    tmax: float,
    seed: int,
    tag: int,
    p=MODEL,
    n_perp: int = 1,
    extra_entropy: tuple[int, ...] = (),
    verbose: bool = False,
) -> dict:
    """Chunked generalized run; asserts gates (2) and (3)."""

    step_count = int(round(tmax / dt))
    if abs(step_count * dt - tmax) > 1e-9 * max(1.0, tmax):
        raise ValueError("tmax must be an integer multiple of dt")
    chunk_sizes = [chunk] * (walkers // chunk)
    if walkers % chunk:
        chunk_sizes.append(walkers % chunk)
    root = np.random.SeedSequence(
        _general_entropy(
            seed,
            tag,
            eps=eps,
            budget=budget,
            weights=weights,
            centres_z=centres_z,
            dt=dt,
            z0=p.z0,
            n_perp=n_perp,
            extra=extra_entropy,
        )
    )
    children = root.spawn(len(chunk_sizes))
    kill_times: list[np.ndarray] = []
    survivors = 0
    contact_steps = 0
    kill_probability_max = 0.0
    walker_steps = 0
    started = time.perf_counter()
    for index, (size, child) in enumerate(zip(chunk_sizes, children)):
        rng = np.random.Generator(np.random.Philox(child))
        outcome = simulate_chunk_general(
            rng,
            size,
            eps=eps,
            budget=budget,
            weights=weights,
            centres_z=centres_z,
            dt=dt,
            step_count=step_count,
            p=p,
            n_perp=n_perp,
        )
        kill_times.append(outcome["kill_times"])
        survivors += outcome["survivors"]
        contact_steps += outcome["contact_steps"]
        kill_probability_max = max(
            kill_probability_max, outcome["kill_probability_max"]
        )
        walker_steps += outcome["walker_steps"]
        if verbose:
            print(
                f"  chunk {index + 1}/{len(chunk_sizes)}: "
                f"killed={outcome['kill_times'].size}/{size} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    times = np.concatenate(kill_times)
    # Gate (2): per-step kill probability in [0, 1].
    assert 0.0 <= kill_probability_max <= 1.0, (
        f"per-step kill probability left [0, 1]: {kill_probability_max!r}"
    )
    # Gate (3): mass balance.
    assert times.size + survivors == walkers, (
        f"mass balance violated: kills={times.size} survivors={survivors} "
        f"walkers={walkers}"
    )
    return {
        "eps": eps,
        "budget": budget,
        "weights": list(weights),
        "centres_z": [float(c) for c in np.asarray(centres_z, float)],
        "walkers": int(walkers),
        "kill_times": times,
        "survivors": int(survivors),
        "contact_steps": int(contact_steps),
        "kill_probability_max": float(kill_probability_max),
        "walker_steps": int(walker_steps),
        "runtime_seconds": float(elapsed),
        "n_perp": n_perp,
        "z0": p.z0,
    }


def gate_zero_general(
    seed: int,
    *,
    dt: float,
    p=MODEL,
    n_perp: int = 1,
    tag: int,
    centres_z: np.ndarray | None = None,
    weights: tuple[float, ...] | None = None,
) -> dict:
    """Gate (1) for a new code path: B = 0 kills nothing, contact sampled."""

    if centres_z is None:
        centres_z = centres_z_for(2, p=p)
    if weights is None:
        weights = tuple(1.0 / len(centres_z) for _ in centres_z)
    outcome = run_config_general(
        eps=0.10,
        budget=0.0,
        weights=weights,
        centres_z=centres_z,
        walkers=GATE_ZERO_WALKERS,
        chunk=GATE_ZERO_WALKERS,
        dt=dt,
        tmax=GATE_ZERO_STEPS * dt,
        seed=seed,
        tag=tag,
        p=p,
        n_perp=n_perp,
    )
    assert outcome["kill_times"].size == 0, "B=0 run produced kills"
    assert outcome["survivors"] == GATE_ZERO_WALKERS, "B=0 run lost walkers"
    assert outcome["contact_steps"] > 0, (
        "B=0 gate never sampled the contact set; the gate is vacuous"
    )
    assert outcome["kill_probability_max"] == 0.0
    return {
        "walkers": GATE_ZERO_WALKERS,
        "steps": GATE_ZERO_STEPS,
        "dt": dt,
        "n_perp": n_perp,
        "z0": p.z0,
        "contact_steps": outcome["contact_steps"],
        "kills": 0,
        "passed": True,
    }


# ----------------------------------------------------------------------------
# Free-exposure law G(t) for general centres and d = 3.
# ----------------------------------------------------------------------------

D3_T_POINTS = 400
D3_Y_POINTS = 160


def contact_probability_d3(ts: np.ndarray, eps: float, p=MODEL) -> np.ndarray:
    """c(t) = P(|R_t|_mi < a) with two independent wrapped transverse coords.

    2D quadrature over the folded wrapped-normal minimum-image transverse
    distances (y1, y2), times the longitudinal Gaussian chord probability
    |R_par| < sqrt(a^2 - y1^2 - y2^2).  Assumes r_perp0 = 0 and isotropic
    Sigma_perp0 = sigma_perp0^2 I."""

    if p.r_perp0 != 0.0:
        raise NotImplementedError("contact_probability_d3 assumes r_perp0 = 0")
    ts = np.asarray(ts, dtype=float)
    mean_par = p.r_par0 * np.exp(-p.gamma * ts)
    var_par = eps**2 * (
        p.u0**2 * np.exp(-2.0 * p.gamma * ts)
        + (2.0 * p.d0 / p.gamma) * (1.0 - np.exp(-2.0 * p.gamma * ts))
    )
    var_perp = eps**2 * p.sigma_perp0**2 + 4.0 * eps**2 * p.d0 * ts

    y_max = min(p.contact_a, 0.5 * p.torus_w)
    y = np.linspace(0.0, y_max, D3_Y_POINTS)
    images = np.arange(
        -base.THEORY_WRAP_IMAGES, base.THEORY_WRAP_IMAGES + 1
    ) * p.torus_w

    out = np.empty(ts.size)
    erf = base._erf
    for it in range(ts.size):
        vp = var_perp[it]
        # Folded density of one minimum-image transverse coordinate on [0, W/2].
        offs = y[:, None] + images[None, :]
        dens = np.exp(-(offs**2) / (2.0 * vp)) / math.sqrt(2.0 * math.pi * vp)
        folded = 2.0 * dens.sum(axis=1)  # shape (Y,)
        # 2D grid over (y1, y2).
        rr = y[:, None] ** 2 + y[None, :] ** 2
        chord = np.sqrt(np.maximum(p.contact_a**2 - rr, 0.0))
        scale = math.sqrt(var_par[it])
        upper = (chord - mean_par[it]) / scale
        lower = (-chord - mean_par[it]) / scale
        prob_par = 0.5 * (
            erf(upper / math.sqrt(2.0)) - erf(lower / math.sqrt(2.0))
        )
        prob_par[rr >= p.contact_a**2] = 0.0
        integrand = folded[:, None] * folded[None, :] * prob_par
        inner = np.trapezoid(integrand, y, axis=1)
        out[it] = np.trapezoid(inner, y)
    return np.clip(out, 0.0, 1.0)


def free_exposure_general(
    ts: np.ndarray,
    *,
    centres_z: np.ndarray,
    eps: float,
    weights: tuple[float, ...],
    p=MODEL,
    n_perp: int = 1,
) -> dict:
    """G(t) for explicit physical slab centres (unit budget).

    G(t) = c(t) / W^(d-1) * sum_j w_j N(mu(t); zeta_j, eps^2 (D0/2gamma + rho^2))
    which in x-space is the standard mixture H_{sigma,w}(x(t)) with
    c_j = sgn(mu') zeta_j / ell0."""

    ts = np.asarray(ts, dtype=float)
    sgn = base.sign_mu_prime(p)
    centres_x = sgn * np.asarray(centres_z, float) / p.ell0
    sigma = base.mixture_sigma(eps, p)
    x_vals = np.asarray(base.x_of_t(ts, p), dtype=float)
    h_vals = base.mixture_h(x_vals, centres_x, np.asarray(weights, float), sigma)
    if n_perp == 1:
        c_vals = base.contact_probability(ts, eps, p)
    elif n_perp == 2:
        c_vals = contact_probability_d3(ts, eps, p)
    else:
        raise ValueError("n_perp must be 1 or 2")
    prefactor = 1.0 / (
        p.torus_w**n_perp
        * math.sqrt(2.0 * math.pi)
        * eps
        * math.sqrt(p.d0 / (2.0 * p.gamma) + p.rho**2)
    )
    g_vals = prefactor * c_vals * h_vals
    return {
        "t": ts,
        "g": g_vals,
        "contact_factor": c_vals,
        "sigma": sigma,
        "centres_x": centres_x,
    }


def g_stationary_diagnostics(
    ts: np.ndarray, g: np.ndarray, window: tuple[float, float] = WINDOW
) -> dict:
    """Peaks/valleys of G in the window with curvature and depth diagnostics.

    Curvatures are finite-difference second derivatives on the ts grid.
    Cumulative exposure integrals int_0^t G ds (from ts[0]) are stored at every
    stationary point so first-order depletion exp(-B Lambda(t)) can be
    reconstructed later without a rerun."""

    lo, hi = window
    d2 = np.gradient(np.gradient(g, ts), ts)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(ts)))
    )
    peaks, valleys = [], []
    for i in range(1, ts.size - 1):
        if not (lo <= ts[i] <= hi):
            continue
        if g[i] > g[i - 1] and g[i] >= g[i + 1]:
            peaks.append(i)
        elif g[i] < g[i - 1] and g[i] <= g[i + 1]:
            valleys.append(i)

    def _row(i: int) -> dict:
        return {
            "time": float(ts[i]),
            "g": float(g[i]),
            "g_second_derivative": float(d2[i]),
            "cumulative_exposure_Lambda": float(cumulative[i]),
        }

    peak_rows = [_row(i) for i in peaks]
    valley_rows = [_row(i) for i in valleys]
    valley_depths = []
    for vi, vrow in zip(valleys, valley_rows):
        left = [g[i] for i in peaks if ts[i] < ts[vi]]
        right = [g[i] for i in peaks if ts[i] > ts[vi]]
        if left and right:
            ref = min(max(left), max(right))
            valley_depths.append(
                {
                    "valley_time": vrow["time"],
                    "valley_g": vrow["g"],
                    "adjacent_peak_g_min": float(ref),
                    "valley_to_peak_ratio": float(vrow["g"] / ref)
                    if ref > 0
                    else None,
                }
            )
    return {
        "window": [lo, hi],
        "n_window_maxima": len(peak_rows),
        "n_window_minima": len(valley_rows),
        "peaks": peak_rows,
        "valleys": valley_rows,
        "valley_depths": valley_depths,
        "min_abs_curvature_at_stationary": (
            float(min(abs(r["g_second_derivative"]) for r in peak_rows + valley_rows))
            if (peak_rows or valley_rows)
            else None
        ),
        "g_window_max": float(np.max(g[(ts >= lo) & (ts <= hi)])),
        "total_exposure_Lambda_to_tmax": float(cumulative[-1]),
    }


def g_window_maxima_count(
    *,
    m: int,
    eps: float,
    weights: tuple[float, ...],
    p=MODEL,
    n_perp: int = 1,
    target_times=None,
    tmax: float = base.DEFAULT_TMAX,
    n_points: int = 1600,
) -> dict:
    """Semi-analytic verdict: does G have exactly m maxima in the window?"""

    ts = np.linspace(base.THEORY_T_MIN, tmax, n_points)
    theory = free_exposure_general(
        ts,
        centres_z=centres_z_for(m, p=p, target_times=target_times),
        eps=eps,
        weights=weights,
        p=p,
        n_perp=n_perp,
    )
    diag = g_stationary_diagnostics(ts, theory["g"])
    diag["sigma_x_space"] = float(theory["sigma"])
    diag["centres_x"] = [float(v) for v in theory["centres_x"]]
    diag["g_has_m_window_maxima"] = bool(diag["n_window_maxima"] == m)
    return diag


# ----------------------------------------------------------------------------
# Publication figure style (PRR submission).
# ----------------------------------------------------------------------------

# TrueType (Type 42) fonts only -- APS production rejects Type 3 -- and all
# text >= 7 pt at the final printed size (figures are designed at final size).
PRR_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "savefig.dpi": 300,
}

# Okabe-Ito colourblind-safe palette.
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_GREY = "#BBBBBB"


def apply_prr_style() -> None:
    import matplotlib

    matplotlib.rcParams.update(PRR_RC)


def stream_tag(fig, label: str) -> None:
    """Tiny corner tag identifying the campaign stream (parameters -> SM)."""
    engine = fig.get_layout_engine()
    if engine is not None:
        # Reserve a thin top strip so the tag never touches the axes frame.
        engine.set(rect=(0.0, 0.0, 1.0, 0.97))
    fig.text(
        0.995, 0.995, label, ha="right", va="top", fontsize=7.0, color="0.45"
    )


def save_figure(fig, stem: Path) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
        written.append(str(stem.with_suffix(suffix)))
    return written


# ----------------------------------------------------------------------------
# Summaries and small utilities.
# ----------------------------------------------------------------------------


def summarize_general(
    result: dict,
    *,
    bandwidth: float = base.DEFAULT_BANDWIDTH,
    target_times: tuple[float, ...],
    m: int,
) -> dict:
    """Light summary for generalized runs (classifier + basic counts)."""

    times = result["kill_times"]
    walkers = result["walkers"]
    window_mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    classifier = base.classify_modes(times, walkers, bandwidth=bandwidth)
    significant = classifier["significant_maxima"]
    return {
        "m_target": m,
        "kills": int(times.size),
        "kill_fraction": times.size / walkers,
        "survivors": result["survivors"],
        "kills_in_window": int(window_mask.sum()),
        "event_fraction_in_window": float(window_mask.sum() / walkers),
        "mode_count": classifier["mode_count"],
        "peak_times": [row["time"] for row in significant],
        "prominences": [row["prominence"] for row in significant],
        "prominence_over_sigma": [
            row["prominence_over_sigma"] for row in significant
        ],
        "passes_m_mode_criterion": classifier["mode_count"] == m,
        "target_times": list(target_times),
        "classifier": classifier,
        "contact_steps": result["contact_steps"],
        "kill_probability_max": result["kill_probability_max"],
        "walker_steps": result["walker_steps"],
        "runtime_seconds": result["runtime_seconds"],
    }


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    half = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
        / denom
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(path, payload)


def model_payload(p=MODEL, extra: str = "") -> str:
    return (
        f"gamma={p.gamma:g}, D0={p.d0:g}, z_bar={p.z_bar:g}, z0={p.z0:g}, "
        f"W={p.torus_w:g}, a={p.contact_a:g}, rho={p.rho:g}, u0={p.u0:g}, "
        f"sigma_perp0={p.sigma_perp0:g}, r_par0={p.r_par0:g}, "
        f"r_perp0={p.r_perp0:g}; window I={list(WINDOW)}"
        + (f"; {extra}" if extra else "")
    )


def make_model(**overrides):
    return replace(MODEL, **overrides)


def model_dict(p) -> dict:
    return asdict(p)


def deterministic_contact_check(
    p, window: tuple[float, float] = WINDOW
) -> dict:
    """Spine contact-interior condition: sup_{t in I} |r_*(t)|_mi <= a - eta.

    r_*(t) = (r_par0 e^{-gamma t}, r_perp0); with r_perp0 = 0 the supremum on
    I = [tau, T] is at t = tau."""

    tau, t_end = window
    sup_norm = math.sqrt(
        (p.r_par0 * math.exp(-p.gamma * tau)) ** 2 + p.r_perp0**2
    )
    eta = p.contact_a - sup_norm
    return {
        "window": [tau, t_end],
        "sup_deterministic_relative_norm": sup_norm,
        "contact_a": p.contact_a,
        "margin_eta": eta,
        "passed": bool(eta > 0.0),
    }
