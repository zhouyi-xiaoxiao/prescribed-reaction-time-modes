#!/usr/bin/env python3
"""W6: single-particle reactivity-field control (the test the Discussion names).

Production (W1--W5) evaluates the conserved-budget slab field at the PAIR
CENTRE Z_t,

    K(Z, R) = 1{|R|_mi < a} (B / W^(d-1)) sum_j w_j phi_j(Z).

This stream keeps every other ingredient identical (slab centres mu(t_j),
widths eps*rho, weights, budget B, initial law, window, Doi contact kernel,
Euler--Maruyama step, per-step thinning 1 - exp(-kappa dt), histogram
binning, classifier, campaign seed) and evaluates the same static field at
the longitudinal LAB-FRAME position of ONE particle,

    x_1 = Z + R_par / 2        (R = x_1 - x_2,  Z = (x_1 + x_2) / 2),

    K_1(Z, R) = 1{|R|_mi < a} (B / W^(d-1)) sum_j w_j phi_j(Z + R_par / 2).

The slabs are longitudinal (translation invariant in the transverse
directions), so the transverse position of the particle does not enter and
the variant lives in the same quotient (Z, R_par, R_perp) as production.

Which particle.  With r_par0 = 0.1 > 0, particle 1 starts on the z0 side of
the midpoint and trails the midpoint mean along the drift (mu decreases from
z0 = 4 towards z_bar = 0).  Along the deterministic relative trajectory the
field on particle 1 is reached later by

    delta_t = r_par0 / (2 gamma |z0 - z_bar|) = 0.0125,

a constant because r_*(t)/2 and |mu'(t)| both decay as e^{-gamma t}; the
other particle (x_2 = Z - R_par/2, ``--particle 2``) is reached earlier by
the same amount.  Both shifts are below one histogram bin (0.02).  The
stochastic effect is the convolution of every clock with the relative law:
the ungated mixture variance grows from eps^2 (D0/(2 gamma) + rho^2) to
eps^2 (D0/(2 gamma) + rho^2 + Var R_par / 4), i.e. sigma grows by up to
sqrt((D0/(2 gamma) + rho^2 + D0/(2 gamma)) / (D0/(2 gamma) + rho^2))
= 1.155 at the stationary relative variance (main-text Discussion).

Semi-analytic overlay.  The single-particle free exposure clock is

    G_1(t) = E_0[K_1(Z_t, R_t)] / B
           = (1/W^(d-1)) int dr p_par(r; t) P_perp(|R_perp|_mi < sqrt(a^2 - r^2); t)
                                  sum_j w_j N(mu(t) + r/2; mu(t_j), eps^2 (D0/(2 gamma) + rho^2)),

a 1D quadrature over the longitudinal relative coordinate with the exact
image-sum wrapped-normal transverse CDF; with the shift factor set to zero
it reduces to the production clock G (checked in-run against
``base.free_exposure_clock``).  The mean-field law f_1 = B G_1 exp(-B Lambda_1),
Lambda_1 = int_0^t G_1, is overlaid as in W5.

Gates on the new code path (asserted unless stated):
  (0) shift factor 0 reproduces ``core.simulate_chunk_general`` bit-for-bit
      on the same Philox stream (regression gate for the copied kernel);
  (1) B = 0 kills nothing from a contact-sampling start;
  (2) per-step kill probability in [0, 1];
  (3) mass balance kills + survivors == walkers;
  (4) dt versus dt/2 window histograms in Monte Carlo sigmas on the cheap
      cell (reported, not asserted, as in W5).

Classifier.  The stored run-time record is ``base.classify_modes`` (peak-only
sigma, retained for audit exactly as in every production record); the formal
verdict is the covariance-aware statistic
``reclassify_covariance_aware.classify_both`` applied to the stored counts
(z >= 5 sigma_prom AND prominence >= 5% of the window maximum).

Cells (all at the campaign anchor budget B = 1):
    (m, eps) = (2, 0.05)  comparator W1 cell_m2_eps0.05_B1.json (1e6 walkers)
    (m, eps) = (3, 0.10)  comparator production m3_eps0.1_B1_w33-33-33.json
    (m, eps) = (2, 0.10)  comparator production m2_eps0.1_B1_w50-50.json

Outputs: artifacts/data/exact_m_prr_upgrade/w6_single_particle/<cell>.json,
         .../w6_single_particle/summary.json,
         artifacts/figures/exact_m_single_particle_control_prr.{png,pdf}.

Reproduction (production size, 4 worker processes):
    python3 exact_m_prr_upgrade_w6_single_particle.py --walkers 5e6 --workers 4
    python3 exact_m_prr_upgrade_w6_single_particle.py --particle 2 --walkers 5e6 --workers 4
    python3 exact_m_prr_upgrade_w6_single_particle.py --summarize --figure
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import reclassify_covariance_aware as cov  # noqa: E402
import validate_exact_m_offlattice as base  # noqa: E402

W6_DIR = core.UPGRADE_DATA / "w6_single_particle"
PRODUCTION_DIR = core.REPORT / "artifacts" / "data" / "exact_m_offlattice_production"
W1_DIR = core.UPGRADE_DATA / "w1_phase_diagram"
COV_JSON = core.UPGRADE_DATA / "covariance_aware_reclassification.json"

# Stream tags (SeedSequence entropy component; disjoint from base 1..3 and
# the campaign tags 11..53 of exact_m_prr_upgrade_core).
TAG_W6_MAIN = 61
TAG_W6_GATE = 62
TAG_W6_DTCHECK = 63

BUDGET = 1.0
CELLS = (
    # (m, eps, budget, comparator relative to artifacts/data)
    (2, 0.05, BUDGET, "exact_m_prr_upgrade/w1_phase_diagram/cell_m2_eps0.05_B1.json"),
    (3, 0.10, BUDGET, "exact_m_offlattice_production/m3_eps0.1_B1_w33-33-33.json"),
    (2, 0.10, BUDGET, "exact_m_offlattice_production/m2_eps0.1_B1_w50-50.json"),
)
CHEAP_CELL = (2, 0.10)  # dt-halving gate runs on this cell only
M_WEIGHTS = {2: (0.5, 0.5), 3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)}
WALKERS = 5_000_000
CHUNK = 100_000
DT_CHECK_WALKERS = 200_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH
SEED = core.CAMPAIGN_SEED
N_PERP = 1
PARTICLE_SIGN = {1: 1.0, 2: -1.0, 0: 0.0}  # 0 = pair-centre limit (gates only)
THEORY_R_POINTS = 801
THEORY_T_POINTS = 800

MODEL = core.MODEL


# ----------------------------------------------------------------------------
# Simulation kernel: identical to core.simulate_chunk_general except for the
# field argument z + shift * r_par on the contact set.
# ----------------------------------------------------------------------------


def simulate_chunk_single_particle(
    rng: np.random.Generator,
    count: int,
    *,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    centres_z: np.ndarray,
    dt: float,
    step_count: int,
    particle_sign: float,
    p=MODEL,
    n_perp: int = N_PERP,
) -> dict:
    mu_j = np.asarray(centres_z, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    slab_norm = 1.0 / (math.sqrt(2.0 * math.pi) * eps * p.rho)
    inv_two_var = 1.0 / (2.0 * (eps * p.rho) ** 2)
    rate_prefactor = budget / p.torus_w**n_perp
    shift = 0.5 * float(particle_sign)

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
        # Field argument: lab-frame longitudinal position of the chosen
        # particle, x = Z + shift * R_par (shift = +1/2, -1/2, or 0).
        x_eval = z[contact_index] + shift * r_par[contact_index]
        kappa = np.zeros(contact_index.size)
        for weight, centre in zip(w_arr, mu_j):
            diff = x_eval - centre
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


def _chunk_task(task: dict) -> dict:
    rng = np.random.Generator(np.random.Philox(task["child"]))
    out = simulate_chunk_single_particle(
        rng,
        task["size"],
        eps=task["eps"],
        budget=task["budget"],
        weights=task["weights"],
        centres_z=np.asarray(task["centres_z"], dtype=float),
        dt=task["dt"],
        step_count=task["step_count"],
        particle_sign=task["particle_sign"],
        p=MODEL,
        n_perp=N_PERP,
    )
    return out


def run_config_single_particle(
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
    particle: int,
    workers: int = 1,
    verbose: bool = False,
) -> dict:
    """Chunked run over independent Philox substreams; chunk -> substream
    mapping is fixed, so the result is independent of the worker count."""

    step_count = int(round(tmax / dt))
    if abs(step_count * dt - tmax) > 1e-9 * max(1.0, tmax):
        raise ValueError("tmax must be an integer multiple of dt")
    chunk_sizes = [chunk] * (walkers // chunk)
    if walkers % chunk:
        chunk_sizes.append(walkers % chunk)
    root = np.random.SeedSequence(
        core._general_entropy(
            seed,
            tag,
            eps=eps,
            budget=budget,
            weights=weights,
            centres_z=centres_z,
            dt=dt,
            z0=MODEL.z0,
            n_perp=N_PERP,
            extra=(int(particle),),
        )
    )
    children = root.spawn(len(chunk_sizes))
    tasks = [
        {
            "child": child,
            "size": size,
            "eps": eps,
            "budget": budget,
            "weights": tuple(weights),
            "centres_z": [float(c) for c in np.asarray(centres_z, float)],
            "dt": dt,
            "step_count": step_count,
            "particle_sign": PARTICLE_SIGN[int(particle)],
        }
        for size, child in zip(chunk_sizes, children)
    ]
    started = time.perf_counter()
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_chunk_task, tasks))
    else:
        outcomes = []
        for index, task in enumerate(tasks):
            outcomes.append(_chunk_task(task))
            if verbose:
                print(
                    f"  chunk {index + 1}/{len(tasks)}: "
                    f"killed={outcomes[-1]['kill_times'].size}/{task['size']} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    times = np.concatenate([o["kill_times"] for o in outcomes])
    survivors = sum(o["survivors"] for o in outcomes)
    contact_steps = sum(o["contact_steps"] for o in outcomes)
    kill_probability_max = max(o["kill_probability_max"] for o in outcomes)
    walker_steps = sum(o["walker_steps"] for o in outcomes)
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
        "particle": int(particle),
        "particle_sign": PARTICLE_SIGN[int(particle)],
        "workers": int(workers),
    }


# ----------------------------------------------------------------------------
# Gates.
# ----------------------------------------------------------------------------


def gate_pair_centre_limit(centres_z: np.ndarray, eps: float, budget: float,
                           weights: tuple[float, ...]) -> dict:
    """Gate (0): shift factor 0 reproduces core.simulate_chunk_general
    bit-for-bit on the same Philox stream."""

    count = 20_000
    steps = 1_400  # reaches t = 1.4, past the first slab centre mu(t_1 = 1)
    seq = np.random.SeedSequence([SEED, TAG_W6_GATE, 0])
    rng_a = np.random.Generator(np.random.Philox(seq))
    rng_b = np.random.Generator(np.random.Philox(seq))
    a = simulate_chunk_single_particle(
        rng_a, count, eps=eps, budget=budget, weights=weights,
        centres_z=centres_z, dt=DT, step_count=steps, particle_sign=0.0,
    )
    b = core.simulate_chunk_general(
        rng_b, count, eps=eps, budget=budget, weights=weights,
        centres_z=centres_z, dt=DT, step_count=steps, p=MODEL, n_perp=N_PERP,
    )
    same = (
        a["kill_times"].size == b["kill_times"].size
        and np.array_equal(a["kill_times"], b["kill_times"])
        and a["survivors"] == b["survivors"]
        and a["contact_steps"] == b["contact_steps"]
        and a["kill_probability_max"] == b["kill_probability_max"]
    )
    assert same, "shift-factor-0 kernel does not reproduce core.simulate_chunk_general"
    assert a["kill_times"].size > 0, "pair-centre regression gate produced no kills"
    return {
        "walkers": count,
        "steps": steps,
        "kills": int(a["kill_times"].size),
        "contact_steps": int(a["contact_steps"]),
        "identical_to_core_simulate_chunk_general": bool(same),
        "passed": True,
    }


def gate_zero_single_particle(centres_z: np.ndarray, weights, particle: int) -> dict:
    """Gate (1): B = 0 kills nothing on the single-particle path."""

    outcome = run_config_single_particle(
        eps=0.10,
        budget=0.0,
        weights=weights,
        centres_z=centres_z,
        walkers=core.GATE_ZERO_WALKERS,
        chunk=core.GATE_ZERO_WALKERS,
        dt=DT,
        tmax=core.GATE_ZERO_STEPS * DT,
        seed=SEED,
        tag=TAG_W6_GATE,
        particle=particle,
    )
    assert outcome["kill_times"].size == 0, "B=0 run produced kills"
    assert outcome["survivors"] == core.GATE_ZERO_WALKERS, "B=0 run lost walkers"
    assert outcome["contact_steps"] > 0, "B=0 gate never sampled the contact set"
    assert outcome["kill_probability_max"] == 0.0
    return {
        "walkers": core.GATE_ZERO_WALKERS,
        "steps": core.GATE_ZERO_STEPS,
        "dt": DT,
        "particle": particle,
        "contact_steps": outcome["contact_steps"],
        "kills": 0,
        "passed": True,
    }


def dt_halving_check(m: int, eps: float, budget: float, weights, centres_z,
                     particle: int, workers: int) -> dict:
    """Gate (4): dt versus dt/2 window histogram in Monte Carlo sigmas."""

    edges = np.arange(
        core.WINDOW[0], core.WINDOW[1] + 0.5 * base.DT_CHECK_BIN, base.DT_CHECK_BIN
    )
    runs = {}
    for label, step in (("dt", DT), ("dt_half", 0.5 * DT)):
        outcome = run_config_single_particle(
            eps=eps, budget=budget, weights=weights, centres_z=centres_z,
            walkers=DT_CHECK_WALKERS, chunk=CHUNK, dt=step, tmax=TMAX,
            seed=SEED, tag=TAG_W6_DTCHECK, particle=particle, workers=workers,
        )
        counts, _ = np.histogram(outcome["kill_times"], bins=edges)
        runs[label] = {
            "dt": step,
            "counts": counts,
            "kill_fraction": outcome["kill_times"].size / DT_CHECK_WALKERS,
            "runtime_seconds": outcome["runtime_seconds"],
        }
    counts_a = runs["dt"]["counts"]
    counts_b = runs["dt_half"]["counts"]
    walkers = DT_CHECK_WALKERS
    fraction_a = counts_a / walkers
    fraction_b = counts_b / walkers
    pooled = (counts_a + counts_b) / (2.0 * walkers)
    variance = pooled * (1.0 - pooled) * (2.0 / walkers)
    usable = (counts_a + counts_b) >= base.DT_CHECK_MIN_POOLED
    z_scores = np.zeros(edges.size - 1)
    z_scores[usable] = (fraction_a[usable] - fraction_b[usable]) / np.sqrt(
        variance[usable]
    )
    z_used = z_scores[usable]
    return {
        "config": {"m": m, "eps": eps, "budget": budget, "weights": list(weights),
                   "particle": particle},
        "walkers_per_run": walkers,
        "window": list(core.WINDOW),
        "bin_width": base.DT_CHECK_BIN,
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
# Semi-analytic single-particle free exposure clock G_1(t).
# ----------------------------------------------------------------------------


def transverse_contact_cdf(b: np.ndarray, var_perp: float, p=MODEL) -> np.ndarray:
    """P(|R_perp|_mi < b) for one wrapped-normal transverse coordinate with
    variance var_perp (mean 0): image sum of Gaussian interval probabilities
    P(R in U_k (kW - b, kW + b)), b clipped to W/2."""

    b = np.clip(np.asarray(b, dtype=float), 0.0, 0.5 * p.torus_w)
    s = math.sqrt(var_perp)
    images = np.arange(-base.THEORY_WRAP_IMAGES, base.THEORY_WRAP_IMAGES + 1) * p.torus_w
    upper = (b[:, None] + images[None, :]) / (s * math.sqrt(2.0))
    lower = (-b[:, None] + images[None, :]) / (s * math.sqrt(2.0))
    return np.clip(0.5 * (base._erf(upper) - base._erf(lower)).sum(axis=1), 0.0, 1.0)


def free_exposure_single_particle(
    ts: np.ndarray,
    *,
    centres_z: np.ndarray,
    eps: float,
    weights: tuple[float, ...],
    particle_sign: float,
    p=MODEL,
    r_points: int = THEORY_R_POINTS,
) -> dict:
    """G_1(t) = (1/W) int dr p_par(r;t) P_perp(|R_perp|_mi < sqrt(a^2 - r^2); t)
    sum_j w_j N(mu(t) + shift*r; mu(t_j), eps^2 (D0/(2 gamma) + rho^2)).

    shift = particle_sign / 2; shift = 0 reproduces the production clock
    c(t) H(x(t)) / (W sqrt(2 pi) eps sqrt(D0/(2 gamma) + rho^2))."""

    ts = np.asarray(ts, dtype=float)
    shift = 0.5 * float(particle_sign)
    mu_t = np.asarray(base.mu_of_t(ts, p), dtype=float)
    mean_par = p.r_par0 * np.exp(-p.gamma * ts)
    var_par = eps**2 * (
        p.u0**2 * np.exp(-2.0 * p.gamma * ts)
        + (2.0 * p.d0 / p.gamma) * (1.0 - np.exp(-2.0 * p.gamma * ts))
    )
    var_perp = eps**2 * p.sigma_perp0**2 + 4.0 * eps**2 * p.d0 * ts
    var_slab = eps**2 * (p.d0 / (2.0 * p.gamma) + p.rho**2)
    r = np.linspace(-p.contact_a, p.contact_a, r_points)
    chord = np.sqrt(np.maximum(p.contact_a**2 - r**2, 0.0))
    w_arr = np.asarray(weights, dtype=float)
    mu_j = np.asarray(centres_z, dtype=float)
    g = np.empty(ts.size)
    contact = np.empty(ts.size)
    for it in range(ts.size):
        dens_par = np.exp(-((r - mean_par[it]) ** 2) / (2.0 * var_par[it])) / math.sqrt(
            2.0 * math.pi * var_par[it]
        )
        cdf_perp = transverse_contact_cdf(chord, var_perp[it], p)
        gate = dens_par * cdf_perp  # contact-gated longitudinal density
        contact[it] = float(np.trapezoid(gate, r))
        arg = mu_t[it] + shift * r
        mix = np.zeros_like(r)
        for weight, centre in zip(w_arr, mu_j):
            mix += weight * np.exp(-((arg - centre) ** 2) / (2.0 * var_slab))
        mix /= math.sqrt(2.0 * math.pi * var_slab)
        g[it] = float(np.trapezoid(gate * mix, r)) / p.torus_w ** N_PERP
    return {"t": ts, "g": g, "contact_factor": contact,
            "sigma_pair_centre_x_space": base.mixture_sigma(eps, p)}


def theory_bundle(m: int, eps: float, budget: float, weights, centres_z,
                  particle_sign: float) -> dict:
    """G_1, its pair-centre limit check, stationary diagnostics, mean-field
    peaks, and the ungated sigma-enlargement estimate."""

    ts = np.linspace(0.0, TMAX, THEORY_T_POINTS + 1)
    ts_pos = ts[1:]
    single = free_exposure_single_particle(
        ts_pos, centres_z=centres_z, eps=eps, weights=weights,
        particle_sign=particle_sign,
    )
    limit = free_exposure_single_particle(
        ts_pos, centres_z=centres_z, eps=eps, weights=weights, particle_sign=0.0,
    )
    pair = base.free_exposure_clock(ts_pos, m=m, eps=eps, weights=tuple(weights))
    mask = ts_pos >= 0.2
    rel = np.abs(limit["g"][mask] - pair["g"][mask]) / np.maximum(pair["g"][mask], 1e-300)
    limit_check = {
        "max_relative_difference_shift0_vs_base_G_for_t_ge_0.2": float(rel.max()),
        "passed": bool(rel.max() < 1e-3),
    }
    assert limit_check["passed"], "shift-0 quadrature does not reproduce base G"

    diag_single = core.g_stationary_diagnostics(ts_pos, single["g"])
    diag_pair = core.g_stationary_diagnostics(ts_pos, pair["g"])

    def mean_field(g_vals):
        g_full = np.concatenate(([0.0], g_vals))
        lam = np.concatenate(
            ([0.0], np.cumsum(0.5 * (g_full[1:] + g_full[:-1]) * np.diff(ts)))
        )
        f1 = budget * g_full * np.exp(-budget * lam)
        return ts, f1, lam

    _, f1_single, lam_single = mean_field(single["g"])
    _, f1_pair, _ = mean_field(pair["g"])
    mf_single = core.g_stationary_diagnostics(ts, f1_single)
    mf_pair = core.g_stationary_diagnostics(ts, f1_pair)

    var_rel_stationary = 2.0 * eps**2 * MODEL.d0 / MODEL.gamma
    sigma_ratio_ungated = math.sqrt(
        (MODEL.d0 / (2.0 * MODEL.gamma) + MODEL.rho**2 + var_rel_stationary / (4.0 * eps**2))
        / (MODEL.d0 / (2.0 * MODEL.gamma) + MODEL.rho**2)
    )
    deterministic_time_shift = (
        particle_sign * MODEL.r_par0 / (2.0 * MODEL.gamma * abs(MODEL.z0 - MODEL.z_bar))
    )
    return {
        "grid": {"t_min": 0.0, "t_max": TMAX, "n_points": THEORY_T_POINTS + 1,
                 "r_points": THEORY_R_POINTS},
        "pair_centre_limit_check": limit_check,
        "single_particle_G1": {
            "n_window_maxima": diag_single["n_window_maxima"],
            "g_has_m_window_maxima": bool(diag_single["n_window_maxima"] == m),
            "peaks": diag_single["peaks"],
            "valleys": diag_single["valleys"],
            "valley_depths": diag_single["valley_depths"],
            "window_integral_Lambda": float(
                np.trapezoid(single["g"][(ts_pos >= core.WINDOW[0]) & (ts_pos <= core.WINDOW[1])],
                             ts_pos[(ts_pos >= core.WINDOW[0]) & (ts_pos <= core.WINDOW[1])])
            ),
            "total_exposure_Lambda_to_tmax": float(lam_single[-1]),
        },
        "pair_centre_G": {
            "n_window_maxima": diag_pair["n_window_maxima"],
            "peaks": diag_pair["peaks"],
            "valleys": diag_pair["valleys"],
            "total_exposure_Lambda_to_tmax": float(np.trapezoid(pair["g"], ts_pos)),
        },
        "mean_field_single_particle_f1": {
            "n_window_maxima": mf_single["n_window_maxima"],
            "peak_times": [row["time"] for row in mf_single["peaks"]],
            "peak_heights": [row["g"] for row in mf_single["peaks"]],
        },
        "mean_field_pair_centre_f1": {
            "n_window_maxima": mf_pair["n_window_maxima"],
            "peak_times": [row["time"] for row in mf_pair["peaks"]],
        },
        "ungated_estimates": {
            "deterministic_time_shift_of_field_particle": deterministic_time_shift,
            "note_time_shift": (
                "r_par0/(2 gamma |z0 - z_bar|) with sign of the particle: "
                "constant in t because r_*(t)/2 and |mu'(t)| decay alike"
            ),
            "sigma_ratio_single_over_pair_stationary": sigma_ratio_ungated,
            "note_sigma": (
                "sqrt((D0/2g + rho^2 + D0/2g)/(D0/2g + rho^2)) using the "
                "stationary relative variance Var R_par = 2 eps^2 D0/gamma"
            ),
        },
        "curves": {
            "t": [float(v) for v in ts],
            "B_G1": [float(v) for v in budget * np.concatenate(([0.0], single["g"]))],
            "f1_single": [float(v) for v in f1_single],
            "B_G_pair": [float(v) for v in budget * np.concatenate(([0.0], pair["g"]))],
        },
    }


# ----------------------------------------------------------------------------
# Summary / comparator / record.
# ----------------------------------------------------------------------------


def _cell_stem(m: int, eps: float, budget: float, particle: int) -> str:
    return f"m{m}_eps{eps:g}_B{budget:g}_p{particle}"


def covariance_aware_verdict(classifier: dict, walkers: int) -> dict:
    res = cov.classify_both(
        classifier["counts"], classifier["edges"], walkers,
        bandwidth=float(classifier["bandwidth"]),
        # Older production records omit the two threshold keys; fall back to
        # the classifier defaults exactly as reclassify_covariance_aware does.
        sigma_factor=float(classifier.get("prominence_sigma_factor", cov.SIGMA_FACTOR)),
        relative_floor=float(classifier.get("prominence_relative_floor", cov.RELATIVE_FLOOR)),
    )
    stored_sm = np.asarray(classifier["smoothed_density"], dtype=float)
    assert np.allclose(res["smoothed"], stored_sm, rtol=0, atol=1e-12)
    assert res["mode_count_peak_only"] == int(classifier["mode_count"])
    rows = res["rows"]
    return {
        "rule": (
            "significant iff prominence >= 5 sigma_prom AND >= 5% of max smoothed "
            "height; sigma_prom^2 = sum_j (A_peak,j - A_base,j)^2 C_j/(N delta)^2"
        ),
        "mode_count": int(res["mode_count_covariance_aware"]),
        "mode_count_peak_only": int(res["mode_count_peak_only"]),
        "local_maxima": [
            {
                "time": r["time"],
                "base_time": r["base_time"],
                "smoothed_height": r["smoothed_height"],
                "prominence": r["prominence"],
                "relative_prominence": r["relative_prominence"],
                "sigma_covariance_aware": r["sigma_covariance_aware"],
                "z_covariance_aware": r["z_covariance_aware"],
                "z_peak_only": r["z_peak_only"],
                "significant": r["significant_covariance_aware"],
            }
            for r in rows
        ],
        "significant_maxima": [
            {
                "time": r["time"],
                "smoothed_height": r["smoothed_height"],
                "prominence": r["prominence"],
                "relative_prominence": r["relative_prominence"],
                "z_covariance_aware": r["z_covariance_aware"],
            }
            for r in rows if r["significant_covariance_aware"]
        ],
    }


def load_comparator(rel: str) -> dict:
    path = core.REPORT / "artifacts" / "data" / rel
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["parameters"]["config"]
    res = payload["results"]
    walkers = int(cfg["walkers"])
    verdict = covariance_aware_verdict(res["classifier"], walkers)
    # Cross-check against the campaign-wide reclassification record.
    cov_payload = json.loads(COV_JSON.read_text(encoding="utf-8"))
    rec = None
    for row in cov_payload["records"]:
        if row["file"].endswith(rel.split("/")[-1]) and row["classifier_path"] == "/results/classifier":
            rec = row
            break
    assert rec is not None, f"no reclassification record for {rel}"
    assert rec["mode_count_new"] == verdict["mode_count"]
    for a, b in zip(rec["maxima"], verdict["local_maxima"]):
        assert math.isclose(a["z_new"], b["z_covariance_aware"], rel_tol=1e-9)
    return {
        "file": rel,
        "walkers": walkers,
        "seed": payload["parameters"].get("seed", cfg.get("seed")),
        "budget": float(cfg["budget"]),
        "mode_count": verdict["mode_count"],
        "peak_times": [r["time"] for r in verdict["significant_maxima"]],
        "z_covariance_aware": [r["z_covariance_aware"] for r in verdict["significant_maxima"]],
        "relative_prominence": [r["relative_prominence"] for r in verdict["significant_maxima"]],
        "event_fraction_in_window": float(res["event_fraction_in_window"]),
        "kill_fraction": float(res["kill_fraction"]),
        "classifier_edges": res["classifier"]["edges"],
        "classifier_smoothed_density": res["classifier"]["smoothed_density"],
        "classifier_counts": res["classifier"]["counts"],
        "cross_checked_against": str(COV_JSON.relative_to(core.REPORT)),
    }


def summarize(result: dict, *, m: int, target_times, theory: dict, comparator: dict) -> dict:
    times = result["kill_times"]
    walkers = result["walkers"]
    window_mask = (times >= core.WINDOW[0]) & (times <= core.WINDOW[1])
    classifier = base.classify_modes(times, walkers, bandwidth=BANDWIDTH)
    verdict = covariance_aware_verdict(classifier, walkers)
    sig = verdict["significant_maxima"]
    peak_times = [r["time"] for r in sig]
    full_edges = np.arange(0.0, TMAX + 0.5 * base.FULL_BIN, base.FULL_BIN)
    full_counts, _ = np.histogram(times, bins=full_edges)
    g1_peaks = [row["time"] for row in theory["single_particle_G1"]["peaks"]]
    mf_peaks = theory["mean_field_single_particle_f1"]["peak_times"]

    def nearest(values, t):
        return min(values, key=lambda v: abs(v - t)) if values else None

    comp_peaks = comparator["peak_times"]
    return {
        "m_target": m,
        "target_times": list(target_times),
        "kills": int(times.size),
        "kill_fraction": times.size / walkers,
        "survivors": result["survivors"],
        "kills_in_window": int(window_mask.sum()),
        "event_fraction_in_window": float(window_mask.sum() / walkers),
        "mode_count": verdict["mode_count"],
        "mode_count_peak_only_runtime_gate": verdict["mode_count_peak_only"],
        "passes_m_mode_criterion": verdict["mode_count"] == m,
        "peak_times": peak_times,
        "prominences": [r["prominence"] for r in sig],
        "relative_prominences": [r["relative_prominence"] for r in sig],
        "prominence_over_sigma_prom": [r["z_covariance_aware"] for r in sig],
        "min_counted_z_covariance_aware": min((r["z_covariance_aware"] for r in sig), default=None),
        "peak_times_vs_target_times": [
            {"classifier": t, "t_j": nearest(list(target_times), t),
             "delta": t - nearest(list(target_times), t)} for t in peak_times
        ],
        "peak_times_vs_G1_peaks": [
            {"classifier": t, "G1_peak": nearest(g1_peaks, t),
             "delta": (t - nearest(g1_peaks, t)) if g1_peaks else None} for t in peak_times
        ],
        "peak_times_vs_mean_field_f1_peaks": [
            {"classifier": t, "f1_peak": nearest(mf_peaks, t),
             "delta": (t - nearest(mf_peaks, t)) if mf_peaks else None} for t in peak_times
        ],
        "peak_times_vs_pair_centre_comparator": [
            {"single_particle": t, "pair_centre": nearest(comp_peaks, t),
             "delta": (t - nearest(comp_peaks, t)) if comp_peaks else None} for t in peak_times
        ],
        "mode_count_vs_pair_centre_comparator": {
            "single_particle": verdict["mode_count"],
            "pair_centre": comparator["mode_count"],
            "retained_equal": verdict["mode_count"] == comparator["mode_count"],
        },
        "event_fraction_in_window_vs_pair_centre_comparator": {
            "single_particle": float(window_mask.sum() / walkers),
            "pair_centre": comparator["event_fraction_in_window"],
        },
        "classifier": classifier,
        "classifier_covariance_aware": verdict,
        "histograms": {
            "linear_full_range": {
                "edges": [float(v) for v in full_edges],
                "counts": [int(v) for v in full_counts],
            }
        },
        "contact_steps": result["contact_steps"],
        "contact_fraction_per_walker_step": (
            result["contact_steps"] / result["walker_steps"] if result["walker_steps"] else None
        ),
        "kill_probability_max": result["kill_probability_max"],
        "walker_steps": result["walker_steps"],
        "runtime_seconds": result["runtime_seconds"],
        "walker_steps_per_second": (
            result["walker_steps"] / result["runtime_seconds"] if result["runtime_seconds"] > 0 else None
        ),
        "workers": result["workers"],
    }


def run_cell(m: int, eps: float, budget: float, comparator_rel: str, *,
             particle: int, walkers: int, workers: int, out_dir: Path,
             gate0: dict, gate1: dict, with_dt_check: bool) -> dict:
    weights = M_WEIGHTS[m]
    target_times = base.TARGET_TIMES[m]
    centres = core.centres_z_for(m)
    sign = PARTICLE_SIGN[particle]
    comparator = load_comparator(comparator_rel)
    theory = theory_bundle(m, eps, budget, weights, centres, sign)
    print(
        f"[{_cell_stem(m, eps, budget, particle)}] theory: G1 window maxima="
        f"{theory['single_particle_G1']['n_window_maxima']} "
        f"(pair G: {theory['pair_centre_G']['n_window_maxima']}), "
        f"f1 peaks={[round(t, 3) for t in theory['mean_field_single_particle_f1']['peak_times']]}, "
        f"limit-check max rel diff="
        f"{theory['pair_centre_limit_check']['max_relative_difference_shift0_vs_base_G_for_t_ge_0.2']:.2e}",
        flush=True,
    )
    dt_check = None
    if with_dt_check:
        print(f"[{_cell_stem(m, eps, budget, particle)}] dt-halving gate ...", flush=True)
        dt_check = dt_halving_check(m, eps, budget, weights, centres, particle, workers)
        print(
            f"  dt-halving: max|z|={dt_check['max_abs_z']:.2f} "
            f"mean|z|={dt_check['mean_abs_z']:.2f} "
            f"frac|z|>2={dt_check['fraction_abs_z_above_2']:.3f} "
            f"({dt_check['usable_bins']} usable bins)",
            flush=True,
        )
    result = run_config_single_particle(
        eps=eps, budget=budget, weights=weights, centres_z=centres,
        walkers=walkers, chunk=CHUNK, dt=DT, tmax=TMAX, seed=SEED,
        tag=TAG_W6_MAIN, particle=particle, workers=workers,
    )
    summary = summarize(result, m=m, target_times=target_times, theory=theory,
                        comparator=comparator)
    payload = {
        "parameters": {
            "stream": "w6_single_particle",
            "variant": (
                "static conserved-budget slab field evaluated at the longitudinal "
                "lab-frame position of ONE particle, x = Z + (particle_sign/2) R_par "
                "(particle 1: +1/2 = Z + R_par/2 with R = x_1 - x_2; particle 2: -1/2), "
                "on the contact set |R|_mi < a; everything else identical to production"
            ),
            "particle": particle,
            "particle_sign": sign,
            "particle_choice_note": (
                "particle 1 is the primary choice: with r_par0 = 0.1 > 0 it starts on "
                "the z0 side of the midpoint and trails the midpoint mean along the "
                "drift, so its clocks are delayed by r_par0/(2 gamma |z0 - z_bar|) = "
                "0.0125 (< one 0.02 bin); particle 2 is the documented alternative "
                "(advanced by the same amount)"
            ),
            "model_parameters": core.model_dict(MODEL),
            "config": {
                "m": m,
                "eps": eps,
                "budget": budget,
                "weights": list(weights),
                "target_times": list(target_times),
                "centres_z": [float(c) for c in centres],
                "dim": 2,
                "n_perp": N_PERP,
                "walkers": walkers,
                "chunk": CHUNK,
                "dt": DT,
                "tmax": TMAX,
                "seed": SEED,
                "tag": TAG_W6_MAIN,
                "bandwidth": BANDWIDTH,
                "window": list(core.WINDOW),
                "workers": workers,
            },
            "rng": (
                "numpy Philox, SeedSequence-spawned substream per chunk; entropy = "
                "core._general_entropy(seed, tag, ..., extra=(particle,))"
            ),
            "killing": "per-step probability 1 - exp(-kappa dt) via expm1",
            "comparator": comparator_rel,
        },
        "validation_gates": {
            "gate0_pair_centre_limit_regression": gate0,
            "gate1_zero_budget_no_kills": gate1,
            "gate2_kill_probability_max": summary["kill_probability_max"],
            "gate2_kill_probability_in_unit_interval": bool(
                0.0 <= summary["kill_probability_max"] <= 1.0
            ),
            "gate3_mass_balance": {
                "kills": summary["kills"],
                "survivors": summary["survivors"],
                "walkers": walkers,
                "passed": bool(summary["kills"] + summary["survivors"] == walkers),
            },
            "gate4_dt_halving": dt_check,
        },
        "theory": theory,
        "comparator_pair_centre": {
            k: v for k, v in comparator.items()
            if k not in ("classifier_edges", "classifier_smoothed_density", "classifier_counts")
        },
        "comparator_curve": {
            "edges": comparator["classifier_edges"],
            "smoothed_density": comparator["classifier_smoothed_density"],
        },
        "results": summary,
    }
    out = out_dir / f"{_cell_stem(m, eps, budget, particle)}.json"
    core.write_json(out, payload)
    peaks = ", ".join(f"{t:.2f}" for t in summary["peak_times"])
    zs = ", ".join(f"{z:.0f}" for z in summary["prominence_over_sigma_prom"])
    print(
        f"[{_cell_stem(m, eps, budget, particle)}] modes={summary['mode_count']} "
        f"(target m={m}, {'PASS' if summary['passes_m_mode_criterion'] else 'FAIL'}; "
        f"pair-centre comparator {comparator['mode_count']}) peaks=[{peaks}] "
        f"z_prom=[{zs}] winfrac={summary['event_fraction_in_window']:.4f} "
        f"killfrac={summary['kill_fraction']:.4f} runtime={summary['runtime_seconds']:.0f}s "
        f"-> {out}",
        flush=True,
    )
    return payload


# ----------------------------------------------------------------------------
# Summary and figure.
# ----------------------------------------------------------------------------


def load_records(out_dir: Path) -> list[dict]:
    records = []
    for path in sorted(out_dir.glob("m*_eps*_B*_p*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def write_summary(out_dir: Path, records: list[dict], wall_seconds: float | None) -> Path:
    rows = []
    for rec in records:
        cfg = rec["parameters"]["config"]
        res = rec["results"]
        comp = rec["comparator_pair_centre"]
        rows.append(
            {
                "m": cfg["m"],
                "eps": cfg["eps"],
                "budget": cfg["budget"],
                "particle": rec["parameters"]["particle"],
                "walkers": cfg["walkers"],
                "mode_count": res["mode_count"],
                "passes_m_mode_criterion": res["passes_m_mode_criterion"],
                "peak_times": res["peak_times"],
                "prominence_over_sigma_prom": res["prominence_over_sigma_prom"],
                "relative_prominences": res["relative_prominences"],
                "peak_deltas_vs_target": [r["delta"] for r in res["peak_times_vs_target_times"]],
                "peak_deltas_vs_pair_centre": [
                    r["delta"] for r in res["peak_times_vs_pair_centre_comparator"]
                ],
                "event_fraction_in_window": res["event_fraction_in_window"],
                "kill_fraction": res["kill_fraction"],
                "kill_probability_max": res["kill_probability_max"],
                "G1_window_maxima": rec["theory"]["single_particle_G1"]["n_window_maxima"],
                "G1_peak_times": [r["time"] for r in rec["theory"]["single_particle_G1"]["peaks"]],
                "mean_field_f1_peak_times": rec["theory"]["mean_field_single_particle_f1"]["peak_times"],
                "comparator": {
                    "file": comp["file"],
                    "walkers": comp["walkers"],
                    "mode_count": comp["mode_count"],
                    "peak_times": comp["peak_times"],
                    "z_covariance_aware": comp["z_covariance_aware"],
                    "event_fraction_in_window": comp["event_fraction_in_window"],
                },
                "gates_passed": {
                    "gate0": rec["validation_gates"]["gate0_pair_centre_limit_regression"]["passed"],
                    "gate1": rec["validation_gates"]["gate1_zero_budget_no_kills"]["passed"],
                    "gate2": rec["validation_gates"]["gate2_kill_probability_in_unit_interval"],
                    "gate3": rec["validation_gates"]["gate3_mass_balance"]["passed"],
                    "theory_limit_check": rec["theory"]["pair_centre_limit_check"]["passed"],
                },
                "dt_halving_max_abs_z": (
                    rec["validation_gates"]["gate4_dt_halving"]["max_abs_z"]
                    if rec["validation_gates"]["gate4_dt_halving"] else None
                ),
                "runtime_seconds": res["runtime_seconds"],
                "file": f"{_cell_stem(cfg['m'], cfg['eps'], cfg['budget'], rec['parameters']['particle'])}.json",
            }
        )
    primary = [r for r in rows if r["particle"] == 1]
    alternative = [r for r in rows if r["particle"] == 2]
    summary = {
        "stream": "w6_single_particle",
        "variant": records[0]["parameters"]["variant"] if records else None,
        "seed": SEED,
        "tags": {"main": TAG_W6_MAIN, "gate": TAG_W6_GATE, "dt_check": TAG_W6_DTCHECK},
        "budget": BUDGET,
        "cells": rows,
        "headline": {
            "all_primary_cells_retain_m_modes": (
                all(r["passes_m_mode_criterion"] for r in primary) if primary else None
            ),
            "all_alternative_cells_retain_m_modes": (
                all(r["passes_m_mode_criterion"] for r in alternative) if alternative else None
            ),
            "n_primary_cells": len(primary),
            "n_alternative_cells": len(alternative),
            "max_abs_peak_delta_vs_pair_centre_primary": max(
                (abs(d) for r in primary for d in r["peak_deltas_vs_pair_centre"]), default=None
            ),
            "max_abs_peak_delta_vs_pair_centre_alternative": max(
                (abs(d) for r in alternative for d in r["peak_deltas_vs_pair_centre"]), default=None
            ),
            "min_counted_z_primary": min(
                (z for r in primary for z in r["prominence_over_sigma_prom"]), default=None
            ),
            "min_counted_z_alternative": min(
                (z for r in alternative for z in r["prominence_over_sigma_prom"]), default=None
            ),
            "kill_probability_max_over_stream": max((r["kill_probability_max"] for r in rows), default=None),
            "all_gates_passed": all(all(r["gates_passed"].values()) for r in rows),
        },
        "ungated_estimates": records[0]["theory"]["ungated_estimates"] if records else None,
        "wall_seconds": wall_seconds,
    }
    out = out_dir / "summary.json"
    core.write_json(out, summary)
    return out


def make_figure(records: list[dict], *, particle: int = 1,
                stem_name: str = "exact_m_single_particle_control_prr") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt

    chosen = [r for r in records if r["parameters"]["particle"] == particle]
    order = {(2, 0.05): 0, (3, 0.10): 1, (2, 0.10): 2}
    chosen.sort(key=lambda r: order.get((r["parameters"]["config"]["m"],
                                          round(r["parameters"]["config"]["eps"], 6)), 9))
    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 0.2, 2.75), constrained_layout=True,
                             squeeze=False)
    letters = "abcdefgh"
    for k, rec in enumerate(chosen):
        ax = axes[0][k]
        cfg = rec["parameters"]["config"]
        res = rec["results"]
        classifier = res["classifier"]
        verdict = res["classifier_covariance_aware"]
        walkers = cfg["walkers"]
        edges = np.asarray(classifier["edges"])
        centres_t = 0.5 * (edges[:-1] + edges[1:])
        counts = np.asarray(classifier["counts"], dtype=float)
        density = counts / (walkers * classifier["bin_width"])
        smoothed = np.asarray(classifier["smoothed_density"])
        comp = rec["comparator_curve"]
        comp_edges = np.asarray(comp["edges"])
        comp_centres = 0.5 * (comp_edges[:-1] + comp_edges[1:])
        comp_smoothed = np.asarray(comp["smoothed_density"])
        curves = rec["theory"]["curves"]
        ts = np.asarray(curves["t"])
        b_g1 = np.asarray(curves["B_G1"])
        f1 = np.asarray(curves["f1_single"])

        ax.stairs(density, edges, fill=True, alpha=0.30, color=core.OI_BLUE,
                  label="EM histogram (single-particle field)")
        ax.plot(comp_centres, comp_smoothed, color="0.55", lw=1.6, ls="-",
                alpha=0.9, label="pair-centre field (production/W1), smoothed")
        ax.plot(centres_t, smoothed, color="#20415f", lw=1.3,
                label=f"smoothed ($h={classifier['bandwidth']:g}$)")
        ax.plot(ts, b_g1, color=core.OI_VERMILLION, lw=1.1, ls="--",
                label=r"$B\,G_1(t)$ single-particle free clock")
        ax.plot(ts, f1, color=core.OI_GREEN, lw=1.1, ls="-.",
                label=r"$f_1=B\,G_1\,\mathrm{e}^{-B\Lambda_1}$ mean field")
        for tj in cfg["target_times"]:
            ax.axvline(tj, color="0.45", lw=0.7, ls=":")
        for i, row in enumerate(verdict["significant_maxima"]):
            ax.plot(row["time"], row["smoothed_height"], marker="v", color="#2ca02c",
                    ms=5.5, ls="none",
                    label="classifier-significant maximum" if i == 0 else None)
            ax.annotate(
                f"{row['time']:.2f}", (row["time"], row["smoothed_height"]),
                textcoords="offset points", xytext=(0, 7), ha="center",
                fontsize=7.0, color="#207020",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.3},
            )
        comp_modes = rec["comparator_pair_centre"]["mode_count"]
        note = (
            f"modes={verdict['mode_count']} (target $m={cfg['m']}$; pair-centre {comp_modes})\n"
            f"window events {res['event_fraction_in_window']:.3f}/walker"
        )
        ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.0, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(
            rf"({letters[k]}) $m={cfg['m']}$, $\varepsilon={cfg['eps']:g}$, $B={cfg['budget']:g}$",
            fontsize=8.0,
        )
        ax.set_xlim(0.0, TMAX)
        ymax = max(float(density.max()), float(comp_smoothed.max()), float(b_g1.max()))
        ax.set_ylim(0.0, 1.30 * ymax)
        ax.set_xlabel("reaction time $t$")
        if k == 0:
            ax.set_ylabel("density per walker [$1/t$]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=7.0,
               frameon=False, handlelength=2.2, columnspacing=1.2)
    stem = core.FIGURES / stem_name
    written = core.save_figure(fig, stem)
    plt.close(fig)
    return written


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--walkers", type=lambda v: int(float(v)), default=WALKERS)
    parser.add_argument("--workers", type=int, default=4,
                        help="worker processes for chunk-level parallelism (<= 10)")
    parser.add_argument("--particle", type=int, choices=(1, 2), default=1,
                        help="which particle carries the field: 1 = Z + R_par/2 (primary), 2 = Z - R_par/2")
    parser.add_argument("--cells", type=str, default="all",
                        help="comma-separated cell indices 0..2 (see CELLS) or 'all'")
    parser.add_argument("--out-subdir", type=str, default="w6_single_particle",
                        help="artifacts/data/exact_m_prr_upgrade/<subdir> (use a pilot subdir for pilots)")
    parser.add_argument("--no-dt-check", action="store_true",
                        help="skip the dt-halving gate on the cheap cell")
    parser.add_argument("--summarize", action="store_true",
                        help="rebuild summary.json from stored cell JSONs (no simulation)")
    parser.add_argument("--figure", action="store_true",
                        help="regenerate the figure from stored cell JSONs (no simulation)")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 10:
        raise SystemExit("--workers must be in 1..10")
    out_dir = core.UPGRADE_DATA / args.out_subdir
    started = time.perf_counter()

    if args.summarize or args.figure:
        records = load_records(out_dir)
        if not records:
            raise SystemExit(f"no cell JSONs under {out_dir}")
        if args.summarize:
            print(f"summary -> {write_summary(out_dir, records, None)}", flush=True)
        if args.figure:
            for path in make_figure(records, particle=1):
                print(f"figure -> {path}", flush=True)
            if any(r["parameters"]["particle"] == 2 for r in records):
                for path in make_figure(records, particle=2,
                                        stem_name="exact_m_single_particle_control_p2_prr"):
                    print(f"figure (alternative particle) -> {path}", flush=True)
        return

    indices = (
        list(range(len(CELLS))) if args.cells == "all"
        else [int(v) for v in args.cells.split(",")]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gates on the new code path (shared across cells of this invocation).
    centres_gate = core.centres_z_for(2)
    gate0 = gate_pair_centre_limit(centres_gate, 0.10, 1.0, M_WEIGHTS[2])
    print(f"gate 0 (shift-0 kernel == core.simulate_chunk_general) PASS: "
          f"kills={gate0['kills']} contact_steps={gate0['contact_steps']}", flush=True)
    gate1 = gate_zero_single_particle(centres_gate, M_WEIGHTS[2], args.particle)
    print(f"gate 1 (B=0, particle {args.particle}) PASS: contact_steps={gate1['contact_steps']} kills=0",
          flush=True)

    for idx in indices:
        m, eps, budget, comparator_rel = CELLS[idx]
        with_dt = (not args.no_dt_check) and (m, eps) == CHEAP_CELL
        run_cell(m, eps, budget, comparator_rel, particle=args.particle,
                 walkers=args.walkers, workers=args.workers, out_dir=out_dir,
                 gate0=gate0, gate1=gate1, with_dt_check=with_dt)

    records = load_records(out_dir)
    print(f"summary -> {write_summary(out_dir, records, time.perf_counter() - started)}",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
