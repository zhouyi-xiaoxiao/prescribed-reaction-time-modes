#!/usr/bin/env python3
"""Off-lattice Euler-Maruyama validation of the exact prescribed-modality slab design.

Model (spine Eqs. exact-m-process / exact-m-initial / exact-m-killing,
``manuscript/exact_m_theorem_spine.tex``), specialized to d = 2 (one
transverse torus coordinate):

    dZ        = -gamma (Z - z_bar) dt + eps sqrt(D0) dW            (midpoint OU)
    dR_par    = -gamma R_par dt      + 2 eps sqrt(D0) dW           (longitudinal)
    dR_perp   =                        2 eps sqrt(D0) dW  (mod W)  (transverse torus)

    Z_0      ~ N(z0, eps^2 D0 / (2 gamma))
    R_par_0  ~ N(r_par0, eps^2 u0^2)
    R_perp_0 ~ WrappedNormal(r_perp0, eps^2 sigma_perp0^2) on the torus of width W

Contact is ``|R|_mi < a`` with the minimum-image transverse distance.  While
in contact the pair is killed at the conserved-budget slab rate

    kappa(Z) = (B / W^(d-1)) sum_j w_j phi_j(Z),
    phi_j    = Gaussian pdf, mean mu(t_j), std eps * rho,
    mu(t)    = z_bar + (z0 - z_bar) exp(-gamma t),

applied per Euler-Maruyama step with probability ``1 - exp(-kappa dt)``
(via expm1), exactly as in
``encounter_heterogeneous_catalytic/code/validate_2d_offlattice_fold.py``
(same Philox/SeedSequence chunking, same histogram conventions).

Free-exposure reference curve.  The spine factorization gives

    G(t) = c(t) H_{sigma,w}(x(t)) / (W^(d-1) sqrt(2 pi) eps sqrt(D0/(2 gamma) + rho^2)),
    sigma = (eps / ell0) sqrt(D0 / (2 gamma) + rho^2),
    x(t)  = sgn(mu') mu(t) / ell0,   c_j = x(t_j),

with ``c(t) = P(|R_t|_mi < a)`` computed here by exact 1D quadrature over the
minimum-image transverse distance (wrapped-normal image sum) times the
longitudinal Gaussian chord probability.  ``B * G(t)`` is overlaid on the
simulated reaction-time density; the theorem asserts f/B -> G as B -> 0, so
finite-B curves may sit below B*G (survival depletion) while keeping the same
stationary signature for B below the protocol-defined operational threshold
B_op(eps).

Mode classification.  Gaussian-smoothed window histogram (bandwidth
reported); interior local maxima are ``significant`` when their prominence
exceeds BOTH 5 smoothed-Poisson sigmas AND 5 percent of the smoothed maximum.
This is a smoothed-histogram diagnostic in the spirit of the repository
classifier conventions; no curve is labelled with a mode count the
prominence rule does not support.

Validation gates (asserted unless stated):
  (1) B = 0 kills nothing from a contact-forcing start;
  (2) per-step kill probability stays in [0, 1];
  (3) mass balance: kills + survivors == walkers (every run);
  (4) dt versus dt/2 window histogram agreement in Monte Carlo sigmas
      (reported, not asserted).

Run modes:
  --smoke        full smoke grid (m=2: 2 eps x 3 B x 2 w; m=3: 2 eps x 1 B x 2 w)
                 + gates + dt-halving on the anchor config + one figure per m;
  default        one config from --m/--eps/--budget/--weights (+ cheap gates).

Pure stdlib + numpy (+ matplotlib for figures).  Deterministic under --seed.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
DEFAULT_DATA_SUBDIR = "exact_m_offlattice_smoke"
DATA = REPORT / "artifacts" / "data" / DEFAULT_DATA_SUBDIR
FIGURES = REPORT / "artifacts" / "figures"

# ----------------------------------------------------------------------------
# Model constants (documented deviations from the task sheet: none).
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelParameters:
    """Physical parameters of the d=2 slab-design encounter process."""

    gamma: float = 1.0
    d0: float = 1.0
    z_bar: float = 0.0
    z0: float = 4.0
    torus_w: float = 1.0
    contact_a: float = 0.4
    rho: float = 1.0
    u0: float = 0.3
    sigma_perp0: float = 0.3
    r_par0: float = 0.1
    r_perp0: float = 0.0
    ell0: float = 1.0
    dim: int = 2


MODEL = ModelParameters()

TARGET_TIMES = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}
WEIGHT_PRESETS = {
    2: ((0.5, 0.5), (0.7, 0.3)),
    3: ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (0.5, 0.3, 0.2)),
}
EPS_GRID = (0.10, 0.20)
BUDGET_GRID = (0.5, 1.0, 2.0)
M3_BUDGETS = (1.0,)

WINDOW = (0.5, 3.5)
WINDOW_BIN = 0.02
FULL_BIN = 0.04

DEFAULT_DT = 1e-3
DEFAULT_TMAX = 4.0
DEFAULT_WALKERS = 200_000
DEFAULT_CHUNK = 100_000
DEFAULT_SEED = 20260807
DEFAULT_BANDWIDTH = 0.04

PROMINENCE_SIGMA_FACTOR = 5.0
PROMINENCE_RELATIVE_FLOOR = 0.05

GATE_ZERO_WALKERS = 20_000
GATE_ZERO_STEPS = 500
DT_CHECK_WALKERS = 100_000
DT_CHECK_BIN = 0.1
DT_CHECK_MIN_POOLED = 10

# Anchor config for the dt-halving gate in --smoke mode.
ANCHOR = (2, 0.10, 1.0, (0.5, 0.5))

# Entropy tags keep independent RNG streams disjoint under one --seed.
TAG_MAIN = 1
TAG_GATE_ZERO = 2
TAG_DT_CHECK = 3

THEORY_T_MIN = 0.02
THEORY_T_POINTS = 800
THEORY_Y_POINTS = 400
THEORY_WRAP_IMAGES = 10

_erf = np.vectorize(math.erf)


def mu_of_t(t: np.ndarray | float, p: ModelParameters = MODEL) -> np.ndarray | float:
    return p.z_bar + (p.z0 - p.z_bar) * np.exp(-p.gamma * np.asarray(t, dtype=float))


def sign_mu_prime(p: ModelParameters = MODEL) -> float:
    # mu'(t) = -gamma (z0 - z_bar) exp(-gamma t): constant sign.
    return -math.copysign(1.0, p.z0 - p.z_bar)


def x_of_t(t: np.ndarray | float, p: ModelParameters = MODEL) -> np.ndarray | float:
    return sign_mu_prime(p) * mu_of_t(t, p) / p.ell0


def mixture_sigma(eps: float, p: ModelParameters = MODEL) -> float:
    return (eps / p.ell0) * math.sqrt(p.d0 / (2.0 * p.gamma) + p.rho**2)


# ----------------------------------------------------------------------------
# Simulation core (vectorized numpy EM + per-step Doi killing).
# ----------------------------------------------------------------------------


def simulate_chunk(
    rng: np.random.Generator,
    count: int,
    *,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    target_times: tuple[float, ...],
    dt: float,
    step_count: int,
    p: ModelParameters = MODEL,
) -> dict:
    """Euler-Maruyama evolution of one walker chunk with per-step Doi killing.

    Killed walkers are removed at the step of death (cost proportional to the
    number of surviving walker-steps).
    """

    mu_j = np.array([float(mu_of_t(tj, p)) for tj in target_times])
    w_arr = np.asarray(weights, dtype=float)
    slab_norm = 1.0 / (math.sqrt(2.0 * math.pi) * eps * p.rho)
    inv_two_var = 1.0 / (2.0 * (eps * p.rho) ** 2)
    rate_prefactor = budget / p.torus_w ** (p.dim - 1)

    decay = 1.0 - p.gamma * dt
    pull = p.gamma * p.z_bar * dt
    noise_z = eps * math.sqrt(p.d0 * dt)
    noise_r = 2.0 * eps * math.sqrt(p.d0 * dt)
    contact_sq = p.contact_a**2
    torus_w = p.torus_w

    z = p.z0 + math.sqrt(eps**2 * p.d0 / (2.0 * p.gamma)) * rng.standard_normal(count)
    r_par = p.r_par0 + eps * p.u0 * rng.standard_normal(count)
    r_perp = np.mod(
        p.r_perp0 + eps * p.sigma_perp0 * rng.standard_normal(count), torus_w
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
        noise = rng.standard_normal((3, alive))
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        r_perp += noise_r * noise[2]
        np.mod(r_perp, torus_w, out=r_perp)

        perp_mi = np.minimum(r_perp, torus_w - r_perp)
        contact_index = np.flatnonzero(r_par * r_par + perp_mi * perp_mi < contact_sq)
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
        r_perp = r_perp[keep]

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


def _config_entropy(
    seed: int,
    tag: int,
    m: int,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    dt: float,
) -> list[int]:
    entropy = [
        int(seed),
        int(tag),
        int(m),
        int(round(eps * 1e9)),
        int(round(budget * 1e9)),
        int(round(dt * 1e15)),
    ]
    entropy.extend(int(round(w * 1e9)) for w in weights)
    return entropy


def run_config(
    *,
    m: int,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    walkers: int,
    chunk: int,
    dt: float,
    tmax: float,
    seed: int,
    tag: int = TAG_MAIN,
    verbose: bool = True,
) -> dict:
    """Simulate one configuration over independent Philox substream chunks."""

    target_times = TARGET_TIMES[m]
    step_count = int(round(tmax / dt))
    if abs(step_count * dt - tmax) > 1e-9 * max(1.0, tmax):
        raise ValueError("tmax must be an integer multiple of dt")
    chunk_sizes = [chunk] * (walkers // chunk)
    if walkers % chunk:
        chunk_sizes.append(walkers % chunk)
    root = np.random.SeedSequence(
        _config_entropy(seed, tag, m, eps, budget, weights, dt)
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
        outcome = simulate_chunk(
            rng,
            size,
            eps=eps,
            budget=budget,
            weights=weights,
            target_times=target_times,
            dt=dt,
            step_count=step_count,
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
                f"  m={m} eps={eps:g} B={budget:g} chunk {index + 1}/"
                f"{len(chunk_sizes)}: killed={outcome['kill_times'].size}/{size} "
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
        "m": m,
        "eps": eps,
        "budget": budget,
        "weights": list(weights),
        "target_times": list(target_times),
        "walkers": int(walkers),
        "kill_times": times,
        "survivors": int(survivors),
        "contact_steps": int(contact_steps),
        "kill_probability_max": float(kill_probability_max),
        "walker_steps": int(walker_steps),
        "runtime_seconds": float(elapsed),
    }


# ----------------------------------------------------------------------------
# Free-exposure theory curve G(t) (spine factorization, exact quadrature).
# ----------------------------------------------------------------------------


def contact_probability(
    ts: np.ndarray, eps: float, p: ModelParameters = MODEL
) -> np.ndarray:
    """c(t) = P(|R_t|_mi < a) for the unkilled relative process (d = 2).

    Exact 1D quadrature: the minimum-image transverse distance density is the
    folded wrapped normal (image sum), and the longitudinal factor is the
    Gaussian probability of the chord ``|R_par| < sqrt(a^2 - y^2)``.
    Assumes r_perp0 = 0 (transverse law symmetric about 0).
    """

    if p.r_perp0 != 0.0:
        raise NotImplementedError("contact_probability assumes r_perp0 = 0")
    ts = np.asarray(ts, dtype=float)
    mean_par = p.r_par0 * np.exp(-p.gamma * ts)
    var_par = eps**2 * (
        p.u0**2 * np.exp(-2.0 * p.gamma * ts)
        + (2.0 * p.d0 / p.gamma) * (1.0 - np.exp(-2.0 * p.gamma * ts))
    )
    var_perp = eps**2 * p.sigma_perp0**2 + 4.0 * eps**2 * p.d0 * ts

    y_max = min(p.contact_a, 0.5 * p.torus_w)
    y = np.linspace(0.0, y_max, THEORY_Y_POINTS)
    images = np.arange(-THEORY_WRAP_IMAGES, THEORY_WRAP_IMAGES + 1) * p.torus_w
    # Folded wrapped-normal density of the minimum-image distance, shape (T, Y).
    offsets = y[None, :, None] + images[None, None, :]
    dens = np.exp(
        -(offsets**2) / (2.0 * var_perp[:, None, None])
    ) / np.sqrt(2.0 * math.pi * var_perp)[:, None, None]
    folded = 2.0 * dens.sum(axis=2)

    chord = np.sqrt(np.maximum(p.contact_a**2 - y**2, 0.0))
    scale = np.sqrt(var_par)[:, None]
    upper = (chord[None, :] - mean_par[:, None]) / scale
    lower = (-chord[None, :] - mean_par[:, None]) / scale
    prob_par = 0.5 * (_erf(upper / math.sqrt(2.0)) - _erf(lower / math.sqrt(2.0)))
    return np.clip(np.trapezoid(folded * prob_par, y, axis=1), 0.0, 1.0)


def mixture_h(
    x: np.ndarray, centres: np.ndarray, weights: np.ndarray, sigma: float
) -> np.ndarray:
    out = np.zeros_like(np.asarray(x, dtype=float))
    for weight, centre in zip(weights, centres):
        out += weight * np.exp(-((x - centre) ** 2) / (2.0 * sigma**2))
    return out


def free_exposure_clock(
    ts: np.ndarray,
    *,
    m: int,
    eps: float,
    weights: tuple[float, ...],
    p: ModelParameters = MODEL,
) -> dict:
    """G(t) = c(t) H_{sigma,w}(x(t)) / (W^(d-1) sqrt(2 pi) eps sqrt(D0/2g + rho^2))."""

    target_times = np.asarray(TARGET_TIMES[m], dtype=float)
    centres = np.asarray(x_of_t(target_times, p), dtype=float)
    sigma = mixture_sigma(eps, p)
    x_vals = np.asarray(x_of_t(ts, p), dtype=float)
    h_vals = mixture_h(x_vals, centres, np.asarray(weights), sigma)
    c_vals = contact_probability(ts, eps, p)
    prefactor = 1.0 / (
        p.torus_w ** (p.dim - 1)
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
        "centres": centres,
        "target_times": target_times,
    }


def _grid_local_maxima(ts: np.ndarray, values: np.ndarray, lo: float, hi: float) -> list[float]:
    mask = (ts >= lo) & (ts <= hi)
    idx = np.flatnonzero(mask)
    peaks = []
    for i in idx:
        if 0 < i < values.size - 1 and values[i] > values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(float(ts[i]))
    return peaks


# ----------------------------------------------------------------------------
# Mode classification (Gaussian-smoothed window histogram + prominence rule).
# ----------------------------------------------------------------------------


def classify_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    walkers: int,
    *,
    bandwidth: float = DEFAULT_BANDWIDTH,
    prominence_sigma_factor: float = PROMINENCE_SIGMA_FACTOR,
    prominence_relative_floor: float = PROMINENCE_RELATIVE_FLOOR,
) -> dict:
    """Classify modes from stored, uniformly binned sufficient statistics.

    This is the authoritative implementation used both by fresh simulations
    and by post-hoc classifier-sensitivity analyses.  Density units are
    per-walker per-time.  The Poisson sigma uses the kernel-squared convolution
    of the raw counts (exact for an edge-unnormalized kernel and mildly
    conservative at the window edges).
    """

    counts = np.asarray(counts, dtype=np.int64)
    edges = np.asarray(edges, dtype=float)
    if walkers <= 0:
        raise ValueError("walkers must be positive")
    if counts.ndim != 1 or edges.ndim != 1 or edges.size != counts.size + 1:
        raise ValueError("edges must have exactly one more entry than counts")
    if np.any(counts < 0):
        raise ValueError("histogram counts must be nonnegative")
    widths = np.diff(edges)
    if widths.size == 0 or np.any(widths <= 0.0):
        raise ValueError("histogram edges must be strictly increasing")
    # Round the span/count quotient at a precision far below the simulation
    # grid.  This recovers the intended decimal bin width (for example 0.02)
    # from serialized edges and exactly reproduces fresh-run arithmetic.
    bin_width = round(float((edges[-1] - edges[0]) / counts.size), 14)
    if not np.allclose(widths, bin_width, rtol=1e-10, atol=1e-12):
        raise ValueError("classify_histogram requires uniformly spaced bins")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if prominence_sigma_factor < 0.0 or prominence_relative_floor < 0.0:
        raise ValueError("prominence thresholds must be nonnegative")

    centres = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (walkers * bin_width)

    radius = max(1, int(math.ceil(4.0 * bandwidth / bin_width)))
    offsets = np.arange(-radius, radius + 1) * bin_width
    kernel = np.exp(-(offsets**2) / (2.0 * bandwidth**2))
    kernel /= kernel.sum()
    ones = np.ones_like(density)
    norm = np.convolve(ones, kernel, mode="same")
    smoothed = np.convolve(density, kernel, mode="same") / norm
    variance = (
        np.convolve(counts.astype(float), kernel**2, mode="same")
        / (norm**2)
        / (walkers * bin_width) ** 2
    )
    sigma = np.sqrt(variance)

    global_max = float(smoothed.max()) if smoothed.size else 0.0
    maxima: list[dict] = []
    for index in range(1, smoothed.size - 1):
        if not (
            smoothed[index] > smoothed[index - 1]
            and smoothed[index] >= smoothed[index + 1]
        ):
            continue
        height = float(smoothed[index])
        saddles: list[float] = []
        for direction in (-1, 1):
            probe = index + direction
            lowest = height
            while 0 <= probe < smoothed.size and smoothed[probe] <= height:
                lowest = min(lowest, float(smoothed[probe]))
                probe += direction
            saddles.append(lowest)
        prominence = height - max(saddles)
        local_sigma = float(sigma[index])
        significant = bool(
            prominence >= prominence_sigma_factor * local_sigma
            and prominence >= prominence_relative_floor * global_max
        )
        maxima.append(
            {
                "time": float(centres[index]),
                "smoothed_height": height,
                "prominence": float(prominence),
                "poisson_sigma": local_sigma,
                "prominence_over_sigma": (
                    float(prominence / local_sigma) if local_sigma > 0.0 else None
                ),
                "significant": significant,
            }
        )
    significant_rows = [row for row in maxima if row["significant"]]
    return {
        "window": [float(edges[0]), float(edges[-1])],
        "bin_width": bin_width,
        "bandwidth": bandwidth,
        "prominence_sigma_factor": prominence_sigma_factor,
        "prominence_relative_floor": prominence_relative_floor,
        "kernel_radius_bins": radius,
        "prominence_rule": (
            f"significant iff prominence >= {prominence_sigma_factor:g} * "
            f"smoothed-Poisson sigma AND >= "
            f"{prominence_relative_floor:g} * max smoothed height"
        ),
        "edges": [float(v) for v in edges],
        "counts": [int(v) for v in counts],
        "smoothed_density": [float(v) for v in smoothed],
        "local_maxima": maxima,
        "significant_maxima": significant_rows,
        "mode_count": len(significant_rows),
        "note": (
            "smoothed-histogram prominence diagnostic; density units are "
            "per-walker per-time so the curve is comparable with B*G(t)"
        ),
    }


def classify_modes(
    kill_times: np.ndarray,
    walkers: int,
    *,
    bandwidth: float = DEFAULT_BANDWIDTH,
    window: tuple[float, float] = WINDOW,
    bin_width: float = WINDOW_BIN,
    prominence_sigma_factor: float = PROMINENCE_SIGMA_FACTOR,
    prominence_relative_floor: float = PROMINENCE_RELATIVE_FLOOR,
) -> dict:
    """Histogram reaction times and classify significant local maxima."""

    lo, hi = window
    edges = np.arange(lo, hi + 0.5 * bin_width, bin_width)
    counts, _ = np.histogram(kill_times, bins=edges)
    return classify_histogram(
        counts,
        edges,
        walkers,
        bandwidth=bandwidth,
        prominence_sigma_factor=prominence_sigma_factor,
        prominence_relative_floor=prominence_relative_floor,
    )


# ----------------------------------------------------------------------------
# Validation gates.
# ----------------------------------------------------------------------------


def gate_zero_budget(seed: int, *, dt: float) -> dict:
    """Gate (1): with B = 0 nothing dies, from a contact-forcing start."""

    outcome = run_config(
        m=2,
        eps=0.10,
        budget=0.0,
        weights=(0.5, 0.5),
        walkers=GATE_ZERO_WALKERS,
        chunk=GATE_ZERO_WALKERS,
        dt=dt,
        tmax=GATE_ZERO_STEPS * dt,
        seed=seed,
        tag=TAG_GATE_ZERO,
        verbose=False,
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
        "contact_steps": outcome["contact_steps"],
        "kills": 0,
        "passed": True,
    }


def dt_halving_check(
    *,
    m: int,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    walkers: int,
    chunk: int,
    dt: float,
    tmax: float,
    seed: int,
) -> dict:
    """Gate (4): dt versus dt/2 window histogram in Monte Carlo sigmas.

    Reported, not asserted.  Bins with pooled counts below
    ``DT_CHECK_MIN_POOLED`` are excluded from the z statistics.
    """

    edges = np.arange(WINDOW[0], WINDOW[1] + 0.5 * DT_CHECK_BIN, DT_CHECK_BIN)
    runs = {}
    for label, step in (("dt", dt), ("dt_half", 0.5 * dt)):
        outcome = run_config(
            m=m,
            eps=eps,
            budget=budget,
            weights=weights,
            walkers=walkers,
            chunk=chunk,
            dt=step,
            tmax=tmax,
            seed=seed,
            tag=TAG_DT_CHECK,
            verbose=False,
        )
        counts, _ = np.histogram(outcome["kill_times"], bins=edges)
        runs[label] = {
            "dt": step,
            "counts": counts,
            "kill_fraction": outcome["kill_times"].size / walkers,
            "runtime_seconds": outcome["runtime_seconds"],
        }
    counts_a = runs["dt"]["counts"]
    counts_b = runs["dt_half"]["counts"]
    fraction_a = counts_a / walkers
    fraction_b = counts_b / walkers
    pooled = (counts_a + counts_b) / (2.0 * walkers)
    variance = pooled * (1.0 - pooled) * (2.0 / walkers)
    usable = (counts_a + counts_b) >= DT_CHECK_MIN_POOLED
    z_scores = np.zeros(edges.size - 1)
    z_scores[usable] = (fraction_a[usable] - fraction_b[usable]) / np.sqrt(
        variance[usable]
    )
    z_used = z_scores[usable]
    return {
        "config": {"m": m, "eps": eps, "budget": budget, "weights": list(weights)},
        "walkers_per_run": int(walkers),
        "window": list(WINDOW),
        "bin_width": DT_CHECK_BIN,
        "counts_dt": [int(v) for v in counts_a],
        "counts_dt_half": [int(v) for v in counts_b],
        "kill_fraction_dt": runs["dt"]["kill_fraction"],
        "kill_fraction_dt_half": runs["dt_half"]["kill_fraction"],
        "usable_bins": int(usable.sum()),
        "excluded_bins": int((~usable).sum()),
        "max_abs_z": float(np.max(np.abs(z_used))) if z_used.size else None,
        "mean_abs_z": float(np.mean(np.abs(z_used))) if z_used.size else None,
        "fraction_abs_z_above_2": (
            float(np.mean(np.abs(z_used) > 2.0)) if z_used.size else None
        ),
        "runtime_seconds_dt": runs["dt"]["runtime_seconds"],
        "runtime_seconds_dt_half": runs["dt_half"]["runtime_seconds"],
        "note": (
            "under the no-bias null the per-bin z are approximately standard "
            "normal; with ~30 usable bins the expected max|z| is about 2.5"
        ),
    }


# ----------------------------------------------------------------------------
# Summaries, JSON, figures.
# ----------------------------------------------------------------------------


def summarize_config(result: dict, *, bandwidth: float, tmax: float) -> dict:
    times = result["kill_times"]
    walkers = result["walkers"]
    kills = int(times.size)
    window_mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    window_kills = int(window_mask.sum())
    classifier = classify_modes(times, walkers, bandwidth=bandwidth)

    theory_ts = np.linspace(THEORY_T_MIN, tmax, THEORY_T_POINTS)
    theory = free_exposure_clock(
        theory_ts,
        m=result["m"],
        eps=result["eps"],
        weights=tuple(result["weights"]),
    )
    g_peaks = _grid_local_maxima(theory_ts, theory["g"], *WINDOW)
    window_theory_mask = (theory_ts >= WINDOW[0]) & (theory_ts <= WINDOW[1])
    g_window_integral = float(
        np.trapezoid(theory["g"][window_theory_mask], theory_ts[window_theory_mask])
    )
    g_total_integral = float(np.trapezoid(theory["g"], theory_ts))

    significant = classifier["significant_maxima"]
    peak_times = [row["time"] for row in significant]
    peak_vs_g = []
    for peak in peak_times:
        if g_peaks:
            nearest = min(g_peaks, key=lambda gp: abs(gp - peak))
            peak_vs_g.append({"classifier": peak, "g_peak": nearest, "delta": peak - nearest})
        else:
            peak_vs_g.append({"classifier": peak, "g_peak": None, "delta": None})
    target_times = result["target_times"]
    peak_vs_target = [
        {
            "classifier": peak,
            "t_j": min(target_times, key=lambda tj: abs(tj - peak)),
            "delta": peak - min(target_times, key=lambda tj: abs(tj - peak)),
        }
        for peak in peak_times
    ]

    full_edges = np.arange(0.0, tmax + 0.5 * FULL_BIN, FULL_BIN)
    full_counts, _ = np.histogram(times, bins=full_edges)

    return {
        "kills": kills,
        "kill_fraction": kills / walkers,
        "survivors": result["survivors"],
        "survivor_fraction": result["survivors"] / walkers,
        "kills_in_window": window_kills,
        "event_fraction_in_window": window_kills / walkers,
        "classifier": classifier,
        "mode_count": classifier["mode_count"],
        "peak_times": peak_times,
        "prominences": [row["prominence"] for row in significant],
        "passes_m_mode_criterion": classifier["mode_count"] == result["m"],
        "theory": {
            "sigma_x_space": theory["sigma"],
            "centres_x_space": [float(v) for v in theory["centres"]],
            "target_times": list(target_times),
            "g_peak_times_in_window": g_peaks,
            "g_window_integral": g_window_integral,
            "g_total_integral_to_tmax": g_total_integral,
            "small_b_window_mass_estimate": result["budget"] * g_window_integral,
        },
        "peak_times_vs_g_peaks": peak_vs_g,
        "peak_times_vs_target_times": peak_vs_target,
        "histograms": {
            "linear_full_range": {
                "edges": [float(v) for v in full_edges],
                "counts": [int(v) for v in full_counts],
            }
        },
        "contact_steps": result["contact_steps"],
        "kill_probability_max": result["kill_probability_max"],
        "walker_steps": result["walker_steps"],
        "runtime_seconds": result["runtime_seconds"],
        "walker_steps_per_second": (
            result["walker_steps"] / result["runtime_seconds"]
            if result["runtime_seconds"] > 0.0
            else None
        ),
    }


def _weight_tag(weights: tuple[float, ...]) -> str:
    return "-".join(f"{100.0 * w:.0f}" for w in weights)


def _config_stem(m: int, eps: float, budget: float, weights: tuple[float, ...]) -> str:
    return f"m{m}_eps{eps:g}_B{budget:g}_w{_weight_tag(weights)}"


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _base_parameters(args: argparse.Namespace) -> dict:
    return {
        "model": (
            "d=2 slab-design encounter process: midpoint OU + longitudinal OU "
            "+ transverse torus BM, Doi killing kappa(Z) = (B/W^(d-1)) "
            "sum_j w_j phi_j(Z) on the contact set |R|_mi < a"
        ),
        "spine": "manuscript/exact_m_theorem_spine.tex Eqs. exact-m-process/-initial/-killing",
        "model_parameters": asdict(MODEL),
        "target_times": {str(k): list(v) for k, v in TARGET_TIMES.items()},
        "window": list(WINDOW),
        "dt": args.dt,
        "tmax": args.tmax,
        "walkers": args.walkers,
        "chunk": args.chunk,
        "seed": args.seed,
        "bandwidth": args.bandwidth,
        "rng": "numpy Philox, one SeedSequence-spawned substream per chunk",
        "killing": "per-step probability 1 - exp(-kappa dt) via expm1",
        "boundary": (
            "transverse coordinate periodic on [0, W) with minimum-image "
            "contact distance; midpoint and longitudinal coordinates free"
        ),
    }


def _panel_label(summary: dict) -> str:
    peaks = ", ".join(f"{t:.2f}" for t in summary["peak_times"])
    return (
        f"modes(classifier)={summary['mode_count']}"
        f" @ t=[{peaks}]\n"
        f"window events={summary['kills_in_window']}"
        f" ({summary['event_fraction_in_window']:.3f}/walker)"
    )


def make_figure(
    m: int,
    entries: list[dict],
    args: argparse.Namespace,
    path_stem: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_values = sorted({entry["config"]["eps"] for entry in entries})
    weight_sets = []
    for entry in entries:
        wt = tuple(entry["config"]["weights"])
        if wt not in weight_sets:
            weight_sets.append(wt)
    budgets = sorted({entry["config"]["budget"] for entry in entries})
    rows = [(eps, wt) for eps in eps_values for wt in weight_sets]
    n_rows, n_cols = len(rows), len(budgets)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.1 * n_rows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    lookup = {
        (
            entry["config"]["eps"],
            tuple(entry["config"]["weights"]),
            entry["config"]["budget"],
        ): entry
        for entry in entries
    }
    theory_ts = np.linspace(THEORY_T_MIN, args.tmax, THEORY_T_POINTS)
    for row_index, (eps, wt) in enumerate(rows):
        theory = free_exposure_clock(theory_ts, m=m, eps=eps, weights=wt)
        for col_index, budget in enumerate(budgets):
            ax = axes[row_index][col_index]
            entry = lookup.get((eps, wt, budget))
            if entry is None:
                ax.set_visible(False)
                continue
            summary = entry["summary"]
            classifier = summary["classifier"]
            edges = np.asarray(classifier["edges"])
            centres = 0.5 * (edges[:-1] + edges[1:])
            counts = np.asarray(classifier["counts"], dtype=float)
            density = counts / (entry["config"]["walkers"] * classifier["bin_width"])
            smoothed = np.asarray(classifier["smoothed_density"])
            ax.stairs(
                density,
                edges,
                fill=True,
                alpha=0.30,
                color="#4878b0",
                label="EM histogram",
            )
            ax.plot(
                centres,
                smoothed,
                color="#20415f",
                lw=1.6,
                label=f"smoothed (h={classifier['bandwidth']:g})",
            )
            ax.plot(
                theory_ts,
                budget * theory["g"],
                color="#c44e52",
                lw=1.4,
                ls="--",
                label="B*G(t) free exposure",
            )
            for tj in TARGET_TIMES[m]:
                ax.axvline(tj, color="0.45", lw=0.9, ls=":")
            for row in classifier["significant_maxima"]:
                ax.plot(
                    row["time"],
                    row["smoothed_height"],
                    marker="v",
                    color="#2ca02c",
                    ms=8,
                    ls="none",
                )
            ax.set_xlim(0.0, args.tmax)
            ax.set_title(
                f"eps={eps:g}, B={budget:g}, w=({', '.join(f'{w:g}' for w in wt)})",
                fontsize=10,
            )
            ax.text(
                0.02,
                0.97,
                _panel_label(summary),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            if row_index == n_rows - 1:
                ax.set_xlabel("reaction time t")
            if col_index == 0:
                ax.set_ylabel("density per walker [1/t]")
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8, loc="upper right")
    payload = (
        f"exact-m slab design, d=2: gamma={MODEL.gamma:g}, D0={MODEL.d0:g}, "
        f"z_bar={MODEL.z_bar:g}, z0={MODEL.z0:g}, W={MODEL.torus_w:g}, "
        f"a={MODEL.contact_a:g}, rho={MODEL.rho:g}, u0={MODEL.u0:g}, "
        f"sigma_perp0={MODEL.sigma_perp0:g}, r_par0={MODEL.r_par0:g}; "
        f"t_j={TARGET_TIMES[m]}; dt={args.dt:g}, tmax={args.tmax:g}, "
        f"walkers={args.walkers}, seed={args.seed}; window I={list(WINDOW)}; "
        f"transverse torus periodic (minimum image), killing 1-exp(-kappa dt); "
        f"markers = significant maxima "
        f"(prominence >= {PROMINENCE_SIGMA_FACTOR:g} sigma and >= "
        f"{100 * PROMINENCE_RELATIVE_FLOOR:g}% of max); dotted lines = target t_j"
    )
    wrapped = "\n".join(textwrap.wrap(payload, width=max(70, 34 * n_cols)))
    tag = getattr(args, "figure_tag", "smoke")
    fig.suptitle(
        f"Off-lattice EM validation of exact m={m} modality ({tag})\n{wrapped}",
        fontsize=9,
    )
    written = []
    for suffix in (".png", ".pdf"):
        out = path_stem.with_suffix(suffix)
        fig.savefig(out, dpi=150)
        written.append(str(out))
    plt.close(fig)
    return written


# ----------------------------------------------------------------------------
# Drivers.
# ----------------------------------------------------------------------------


def run_and_record(
    *,
    m: int,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    args: argparse.Namespace,
    gate_zero: dict,
) -> dict:
    result = run_config(
        m=m,
        eps=eps,
        budget=budget,
        weights=weights,
        walkers=args.walkers,
        chunk=args.chunk,
        dt=args.dt,
        tmax=args.tmax,
        seed=args.seed,
        verbose=args.verbose,
    )
    summary = summarize_config(result, bandwidth=args.bandwidth, tmax=args.tmax)
    config = {
        "m": m,
        "eps": eps,
        "budget": budget,
        "weights": list(weights),
        "target_times": list(TARGET_TIMES[m]),
        "walkers": args.walkers,
    }
    payload = {
        "parameters": {**_base_parameters(args), "config": config},
        "validation_gates": {
            "gate1_zero_budget_no_kills": gate_zero,
            "gate2_kill_probability_max": summary["kill_probability_max"],
            "gate2_kill_probability_in_unit_interval": bool(
                0.0 <= summary["kill_probability_max"] <= 1.0
            ),
            "gate3_mass_balance": {
                "kills": summary["kills"],
                "survivors": summary["survivors"],
                "walkers": args.walkers,
                "passed": bool(
                    summary["kills"] + summary["survivors"] == args.walkers
                ),
            },
        },
        "results": summary,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    stem_suffix = getattr(args, "stem_suffix", "") or ""
    path = DATA / f"{_config_stem(m, eps, budget, weights)}{stem_suffix}.json"
    _write_json(path, payload)
    peaks = ", ".join(f"{t:.2f}" for t in summary["peak_times"])
    print(
        f"{_config_stem(m, eps, budget, weights)}: "
        f"kill_fraction={summary['kill_fraction']:.4f} "
        f"window_fraction={summary['event_fraction_in_window']:.4f} "
        f"modes={summary['mode_count']} (target m={m}, "
        f"{'PASS' if summary['passes_m_mode_criterion'] else 'FAIL'}) "
        f"peaks=[{peaks}] "
        f"runtime={summary['runtime_seconds']:.1f}s -> {path.name}",
        flush=True,
    )
    return {"config": {**config, "walkers": args.walkers}, "summary": summary}


def smoke(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    gate_zero = gate_zero_budget(args.seed, dt=args.dt)
    print(
        f"gate B=0 PASS: contact_steps={gate_zero['contact_steps']} kills=0",
        flush=True,
    )

    grids = {
        2: [
            (eps, budget, weights)
            for eps in EPS_GRID
            for budget in BUDGET_GRID
            for weights in WEIGHT_PRESETS[2]
        ],
        3: [
            (eps, budget, weights)
            for eps in EPS_GRID
            for budget in M3_BUDGETS
            for weights in WEIGHT_PRESETS[3]
        ],
    }
    entries_by_m: dict[int, list[dict]] = {2: [], 3: []}
    for m, grid in grids.items():
        for eps, budget, weights in grid:
            entries_by_m[m].append(
                run_and_record(
                    m=m,
                    eps=eps,
                    budget=budget,
                    weights=weights,
                    args=args,
                    gate_zero=gate_zero,
                )
            )

    anchor_m, anchor_eps, anchor_budget, anchor_weights = ANCHOR
    print("dt-halving check on anchor config ...", flush=True)
    dt_check = dt_halving_check(
        m=anchor_m,
        eps=anchor_eps,
        budget=anchor_budget,
        weights=anchor_weights,
        walkers=DT_CHECK_WALKERS,
        chunk=args.chunk,
        dt=args.dt,
        tmax=args.tmax,
        seed=args.seed,
    )
    print(
        f"dt-halving: max|z|={dt_check['max_abs_z']:.2f} "
        f"mean|z|={dt_check['mean_abs_z']:.2f} "
        f"frac|z|>2={dt_check['fraction_abs_z_above_2']:.3f} "
        f"({dt_check['usable_bins']} usable bins)",
        flush=True,
    )

    figures = {}
    for m, entries in entries_by_m.items():
        if entries:
            figures[m] = make_figure(
                m,
                entries,
                args,
                FIGURES / f"exact_m_offlattice_{args.figure_tag}_m{m}",
            )
            print(f"figure m={m}: {figures[m]}", flush=True)

    overview_rows = []
    for m, entries in entries_by_m.items():
        for entry in entries:
            summary = entry["summary"]
            overview_rows.append(
                {
                    **entry["config"],
                    "kill_fraction": summary["kill_fraction"],
                    "event_fraction_in_window": summary["event_fraction_in_window"],
                    "kills_in_window": summary["kills_in_window"],
                    "mode_count": summary["mode_count"],
                    "passes_m_mode_criterion": summary["passes_m_mode_criterion"],
                    "peak_times": summary["peak_times"],
                    "prominences": summary["prominences"],
                    "g_peak_times": summary["theory"]["g_peak_times_in_window"],
                    "kill_probability_max": summary["kill_probability_max"],
                    "runtime_seconds": summary["runtime_seconds"],
                    "walker_steps_per_second": summary["walker_steps_per_second"],
                }
            )
    overview = {
        "parameters": _base_parameters(args),
        "gate1_zero_budget": gate_zero,
        "gate4_dt_halving": dt_check,
        "configs": overview_rows,
        "figures": {str(m): paths for m, paths in figures.items()},
        "total_runtime_seconds": time.perf_counter() - started,
    }
    summary_path = DATA / f"exact_m_offlattice_{args.figure_tag}_summary.json"
    _write_json(summary_path, overview)
    print(
        f"smoke complete in {overview['total_runtime_seconds']:.1f}s -> "
        f"{summary_path}",
        flush=True,
    )


def replot(args: argparse.Namespace) -> None:
    """Rebuild the per-m overview figures from the stored per-config JSONs."""

    entries_by_m: dict[int, list[dict]] = {}
    for path in sorted(DATA.glob("m*_eps*_B*_w*.json")):
        if "_dt" in path.name:
            # Skip dt-halving reports and alternate-dt bias-check runs; the
            # overview panels show one dt per (m, eps, B, w) cell.
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["parameters"]["config"]
        entries_by_m.setdefault(config["m"], []).append(
            {"config": config, "summary": payload["results"]}
        )
    if not entries_by_m:
        raise SystemExit(f"no config JSONs found under {DATA}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for m, entries in sorted(entries_by_m.items()):
        paths = make_figure(
            m, entries, args, FIGURES / f"exact_m_offlattice_{args.figure_tag}_m{m}"
        )
        print(f"figure m={m}: {paths}", flush=True)


def _weights_key(weights: list[float]) -> tuple[float, ...]:
    return tuple(round(float(w), 6) for w in weights)


def summarize_from_data(args: argparse.Namespace) -> None:
    """Aggregate stored per-config JSONs into one summary (same fields as smoke).

    Adds: production seed, measured wall time (--wall-seconds), an alternate-dt
    bias check built from any ``*_dt*.json`` companion runs (window histograms
    rebinned to 0.1 and compared in Monte Carlo sigmas), and a mode-count
    comparison against the smoke summary when it exists.
    """

    smoke_modes: dict[tuple, int] = {}
    smoke_summary_path = (
        REPORT
        / "artifacts"
        / "data"
        / DEFAULT_DATA_SUBDIR
        / "exact_m_offlattice_smoke_summary.json"
    )
    if smoke_summary_path.exists():
        smoke_payload = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
        for row in smoke_payload["configs"]:
            key = (
                row["m"],
                round(row["eps"], 9),
                round(row["budget"], 9),
                _weights_key(row["weights"]),
            )
            smoke_modes[key] = row["mode_count"]

    rows = []
    alternate_dt_payloads = []
    gate1 = None
    for path in sorted(DATA.glob("m*_eps*_B*_w*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "_dt" in path.name:
            if "dt_halving" not in payload:
                alternate_dt_payloads.append((path.name, payload))
            continue
        config = payload["parameters"]["config"]
        summary = payload["results"]
        if gate1 is None:
            gate1 = payload["validation_gates"]["gate1_zero_budget_no_kills"]
        key = (
            config["m"],
            round(config["eps"], 9),
            round(config["budget"], 9),
            _weights_key(config["weights"]),
        )
        rows.append(
            {
                **config,
                "dt": payload["parameters"]["dt"],
                "source_file": path.name,
                "kill_fraction": summary["kill_fraction"],
                "event_fraction_in_window": summary["event_fraction_in_window"],
                "kills_in_window": summary["kills_in_window"],
                "mode_count": summary["mode_count"],
                "passes_m_mode_criterion": summary["passes_m_mode_criterion"],
                "peak_times": summary["peak_times"],
                "prominences": summary["prominences"],
                "g_peak_times": summary["theory"]["g_peak_times_in_window"],
                "kill_probability_max": summary["kill_probability_max"],
                "runtime_seconds": summary["runtime_seconds"],
                "walker_steps_per_second": summary["walker_steps_per_second"],
                "mode_count_smoke": smoke_modes.get(key),
                "mode_count_changed_vs_smoke": (
                    smoke_modes[key] != summary["mode_count"]
                    if key in smoke_modes
                    else None
                ),
            }
        )
    if not rows:
        raise SystemExit(f"no config JSONs found under {DATA}")

    dt_bias_checks = []
    lookup = {
        (
            row["m"],
            round(row["eps"], 9),
            round(row["budget"], 9),
            _weights_key(row["weights"]),
        ): row
        for row in rows
    }
    for name, payload in alternate_dt_payloads:
        config = payload["parameters"]["config"]
        key = (
            config["m"],
            round(config["eps"], 9),
            round(config["budget"], 9),
            _weights_key(config["weights"]),
        )
        base_row = lookup.get(key)
        if base_row is None:
            continue
        base_payload = json.loads(
            (DATA / base_row["source_file"]).read_text(encoding="utf-8")
        )
        rebin = 5  # 0.02 window bins -> 0.1 comparison bins
        counts_a = np.asarray(
            base_payload["results"]["classifier"]["counts"], dtype=float
        )
        counts_b = np.asarray(
            payload["results"]["classifier"]["counts"], dtype=float
        )
        usable_len = (min(counts_a.size, counts_b.size) // rebin) * rebin
        counts_a = counts_a[:usable_len].reshape(-1, rebin).sum(axis=1)
        counts_b = counts_b[:usable_len].reshape(-1, rebin).sum(axis=1)
        walkers_a = base_payload["parameters"]["config"]["walkers"]
        walkers_b = payload["parameters"]["config"]["walkers"]
        fraction_a = counts_a / walkers_a
        fraction_b = counts_b / walkers_b
        pooled = (counts_a + counts_b) / (walkers_a + walkers_b)
        variance = pooled * (1.0 - pooled) * (1.0 / walkers_a + 1.0 / walkers_b)
        usable = (counts_a + counts_b) >= DT_CHECK_MIN_POOLED
        z = np.zeros(counts_a.size)
        z[usable] = (fraction_a[usable] - fraction_b[usable]) / np.sqrt(
            variance[usable]
        )
        z_used = z[usable]
        dt_bias_checks.append(
            {
                "config": config,
                "dt_base": base_payload["parameters"]["dt"],
                "dt_alternate": payload["parameters"]["dt"],
                "source_files": [base_row["source_file"], name],
                "comparison_bin_width": rebin * WINDOW_BIN,
                "usable_bins": int(usable.sum()),
                "excluded_bins": int((~usable).sum()),
                "max_abs_z": float(np.max(np.abs(z_used))) if z_used.size else None,
                "mean_abs_z": float(np.mean(np.abs(z_used))) if z_used.size else None,
                "fraction_abs_z_above_2": (
                    float(np.mean(np.abs(z_used) > 2.0)) if z_used.size else None
                ),
                "kill_fraction_base": base_payload["results"]["kill_fraction"],
                "kill_fraction_alternate": payload["results"]["kill_fraction"],
                "mode_count_base": base_payload["results"]["mode_count"],
                "mode_count_alternate": payload["results"]["mode_count"],
                "peak_times_alternate": payload["results"]["peak_times"],
            }
        )

    overview = {
        "parameters": _base_parameters(args),
        "seed": args.seed,
        "wall_seconds": args.wall_seconds,
        "total_runtime_seconds_core": float(
            sum(row["runtime_seconds"] for row in rows)
        ),
        "gate1_zero_budget": gate1,
        "gate4_alternate_dt_bias_checks": dt_bias_checks,
        "configs": rows,
        "figures": {
            "expected": [
                str(FIGURES / f"exact_m_offlattice_{args.figure_tag}_m{m}{ext}")
                for m in sorted({row["m"] for row in rows})
                for ext in (".png", ".pdf")
            ]
        },
    }
    out = DATA / f"exact_m_offlattice_{args.figure_tag}_summary.json"
    _write_json(out, overview)
    print(f"summary ({len(rows)} configs, {len(dt_bias_checks)} dt checks) -> {out}")


def parse_weights(text: str | None, m: int) -> tuple[float, ...]:
    if text is None:
        return WEIGHT_PRESETS[m][0]
    values = tuple(float(part) for part in text.split(","))
    if len(values) != m:
        raise SystemExit(f"--weights must supply exactly m={m} values")
    if any(v <= 0.0 for v in values):
        raise SystemExit("--weights must be strictly positive")
    total = sum(values)
    return tuple(v / total for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Off-lattice Euler-Maruyama validation of the exact prescribed-"
            "modality slab design (spine: exact_m_theorem_spine.tex)."
        )
    )
    parser.add_argument("--m", type=int, choices=sorted(TARGET_TIMES), default=2)
    parser.add_argument("--eps", type=float, default=0.10)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="comma-separated positive weights (normalized); default uniform preset",
    )
    parser.add_argument(
        "--walkers",
        type=lambda value: int(float(value)),
        default=DEFAULT_WALKERS,
        help="walkers per config (accepts scientific notation)",
    )
    parser.add_argument(
        "--chunk", type=lambda value: int(float(value)), default=DEFAULT_CHUNK
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT_BANDWIDTH,
        help="Gaussian smoothing bandwidth for the mode classifier (time units)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the full smoke grid + gates + dt-halving + figures",
    )
    parser.add_argument(
        "--dt-check",
        action="store_true",
        help="single-config mode: also run the dt versus dt/2 comparison",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="rebuild the overview figures from stored config JSONs (no simulation)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="aggregate stored config JSONs into one summary JSON (no simulation)",
    )
    parser.add_argument(
        "--data-subdir",
        type=str,
        default=DEFAULT_DATA_SUBDIR,
        help="artifacts/data/<subdir> for config JSONs (default: smoke dir)",
    )
    parser.add_argument(
        "--figure-tag",
        type=str,
        default="smoke",
        help="tag used in figure/summary filenames and titles (e.g. production)",
    )
    parser.add_argument(
        "--stem-suffix",
        type=str,
        default="",
        help="suffix appended to the per-config JSON stem (e.g. _dt5e-4)",
    )
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=None,
        help="measured wall time recorded by --summarize",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.dt <= 0.0 or args.tmax <= 0.0:
        raise SystemExit("dt and tmax must be positive")
    if args.walkers <= 0 or args.chunk <= 0:
        raise SystemExit("walkers and chunk must be positive")
    if args.budget < 0.0:
        raise SystemExit("budget must be nonnegative")
    if args.bandwidth <= 0.0:
        raise SystemExit("bandwidth must be positive")

    global DATA
    DATA = REPORT / "artifacts" / "data" / args.data_subdir

    if args.replot:
        replot(args)
        return
    if args.summarize:
        summarize_from_data(args)
        return
    if args.smoke:
        smoke(args)
        return

    weights = parse_weights(args.weights, args.m)
    gate_zero = gate_zero_budget(args.seed, dt=args.dt)
    print(
        f"gate B=0 PASS: contact_steps={gate_zero['contact_steps']} kills=0",
        flush=True,
    )
    run_and_record(
        m=args.m,
        eps=args.eps,
        budget=args.budget,
        weights=weights,
        args=args,
        gate_zero=gate_zero,
    )
    if args.dt_check:
        report = dt_halving_check(
            m=args.m,
            eps=args.eps,
            budget=args.budget,
            weights=weights,
            walkers=DT_CHECK_WALKERS,
            chunk=args.chunk,
            dt=args.dt,
            tmax=args.tmax,
            seed=args.seed,
        )
        path = DATA / f"{_config_stem(args.m, args.eps, args.budget, weights)}_dtcheck.json"
        _write_json(path, {"parameters": _base_parameters(args), "dt_halving": report})
        print(
            f"dt-halving: max|z|={report['max_abs_z']:.2f} "
            f"mean|z|={report['mean_abs_z']:.2f} "
            f"frac|z|>2={report['fraction_abs_z_above_2']:.3f} -> {path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
