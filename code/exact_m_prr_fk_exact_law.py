#!/usr/bin/env python3
"""Exact-law Feynman--Kac / Rao--Blackwell estimator for the exact-m Doi slab model (item N0).

What "exact law" means
----------------------
The production simulator (``validate_exact_m_offlattice.simulate_chunk``) evolves
the pair by Euler--Maruyama with step ``dt`` and, at the END of every step n,
kills a surviving walker with probability ``1 - exp(-kappa(X_n) dt)``.  For that
discrete-time process the reaction-time law is known exactly in terms of the
UNKILLED path (discrete Feynman--Kac identity):

    P(T_R = t_n) = E[ (1 - exp(-B e_n)) exp(-B X_{n-1}) ],
    P(T_R > t_n) = E[ exp(-B X_n) ],

with the unit-budget step exposure ``e_n = gate_n * sum_j w_j phi_j(F_n) dt / W^(d-1)``
and ``X_n = e_1 + ... + e_n``.  Averaging these weights over independent unkilled
paths integrates the kill coin out conditionally on the path (Rao--Blackwell), so
the estimator is unbiased for the exact discrete-time law, never has larger
variance than the direct-kill histogram at equal path count, and a SINGLE path
ensemble serves every budget B and (because X is linear in w) every allocation
w.  The only approximations are the Euler--Maruyama time step (identical to the
production simulator) and Monte Carlo sampling (quantified by iid per-path
standard errors and batch means / jackknife over path groups).

Storage modes
-------------
* ``stored``: per path, per slab j and per variant, the unit-budget exposure
  INCREMENTS dX^{(j)}_k = X^{(j)}(c_k) - X^{(j)}(c_{k-1}) (w_j = 1, B = 1) over
  the checkpoint intervals, computed in float64 and stored as float32 chunk
  files (``.npz``, compressed) under ``~/.local-build/prr_fk_ensembles/<name>/``.
  Storing increments (not cumulative X) keeps full float32 relative precision
  for every bin, down to bin masses ~1e-28 (self-test).  Any (B, w) is
  evaluated after the fact: X = B * sum_j w_j cumsum(dX^{(j)}) (float64) and
  bin masses are accumulated directly as exp(-B X(c_a)) * (-expm1(-B dX)), so
  there is no S_k - S_{k+1} cancellation.  The default ``production``
  checkpoint grid reproduces the production histogram binning *bit for bit*:
  a checkpoint for bin edge e is the number of steps with kill time
  (n+1)*dt < e (numpy.histogram left-closed bins; the window's upper edge 3.5
  and t_max are inclusive).  NB: np.arange(0.5, 3.51, 0.02) edges and the
  step times (n+1)*dt differ in the last ulp at 147 of 151 window edges, so a
  nominal grid would shift one step of mass per edge (~5 % on steep flanks at
  eps = 0.05).  The squared minimum-image pair distance at the target-time
  steps is stored too (``dist2_tj``), so frozen-gate laws for any contact
  radius can be evaluated offline.  Disk: ~0.1-0.2 GB per variant per 1e5
  paths for m = 2-3 (compressed); keep the variant list to what is needed.
* ``declared``: O(1)-memory accumulation, at every time step, of the exact
  per-step kill probability f_n for a declared list of (variant, B, w[, cap])
  items plus exposure moments (E X, E X^2, E V, E V X) for every distinct
  (variant, w); batch sums give SEs.  Needed for the L-infinity cap (nonlinear in
  B) and for step-resolution derivative/critical-point censuses (N9).

Statistical errors
------------------
Paths are iid, so every estimate is a path average and gets an iid per-path
SE; path groups (default 20 contiguous groups) give batch-means SEs and
delete-one-group jackknife SEs for nonlinear functionals (B_p).  Bin errors
of one FK ensemble are positively correlated across neighbouring bins (a path
feeds several bins): use ``exact_law(..., with_cov=True)`` and a
covariance-aware chi-square, not a diagonal one, when comparing curves.

Kernel variants (per stored array, all on common paths)
-------------------------------------------------------
``full`` (contact gate |R|_mi < a, field at the pair midpoint Z; the paper's
model), ``nogate`` (gate removed), ``meancontact`` (gate replaced by the
deterministic contact probability c(t) = P(|R_t|_mi < a), exact quadrature),
``p1`` / ``p2`` (field evaluated at particle 1 / 2, x = Z +/- R_par/2),
``a0.2`` etc. (contact radius a), and generic ``gate=contact|none|mean,a=..,field=mid|p1|p2``.
Configuration-level options (separate path laws or kernels): ``var_z0_scale``
(Var Z_0 multiplier; 0 = point release), ``slab_shape`` (gauss or tophat matched
in mass and second moment), ``slab_sd`` (fixed physical slab width, independent
of eps; TH-8), geometry overrides (``r_par0``, ``r_perp0``, ``u0``,
``sigma_perp0``; boundary-tangent = r_par0 0, r_perp0 a; TH-7), ``n_perp`` = 2
(d = 3, prefactor B/W^2, as in exact_m_prr_upgrade_w5), ``z0`` and explicit
``centres_z`` (W4 stretched geometry), and ``cap`` (L-infinity rate cap K_max,
declared mode only).

Seeds
-----
Base seed 20260923 with the stream tags of GAP_CLOSURE_PLAN.md: 80 N0, 81 N1,
82 N3, 83 N4, 84 N5, 85 N6, 86 N8, 87 N9 (88, 89 spare).  The SeedSequence
entropy contains only path-law parameters (plus seed, tag, replicate and chunk
size), so ensembles that differ only in the kernel (m, centres, weights, slab
shape/width) share paths by design (common random numbers).  Chunk i uses child
i of ``SeedSequence(entropy).spawn(n_chunks)`` with a Philox generator, so the
result is independent of worker scheduling.  Every seed/entropy is recorded in
the index JSON.

Python API (import this module; ``sys.path`` must contain ``code/``)
-------------------------------------------------------------------
    import exact_m_prr_fk_exact_law as fk
    spec = fk.EnsembleSpec(m=2, eps=0.1)                    # production model
    ens  = fk.simulate_ensemble(spec, 200_000, name="n1_m2_eps0.1", tag=fk.TAGS["N1"],
                                variants=("full", "nogate", "meancontact"), workers=3)
    ens  = fk.load_ensemble("n1_m2_eps0.1")                  # later / other process
    ens.cache_in_memory = True                               # optional: keep chunks in RAM
    law  = fk.exact_law(ens, B=1.0, w=(0.5, 0.5))            # window bins, iid SE, groups
    law  = fk.exact_law(ens, 1.0, (0.5, 0.5), variant="nogate", with_cov=True)
    for start, dX, dist2 in ens.iter_chunks("full"): ...    # raw per-path increments
    bm   = fk.basin_masses(ens, B=1.0, w=(0.5, 0.5))         # + mean-field & limit law
    cps  = fk.derivative_signs(law["t"], law["density"], law["group_density"])
    mom  = fk.exposure_moments(ens, w=(0.5, 0.5))            # E X_t, Var X_t, sampled G, Cov
    cls  = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000)
    bp   = fk.prominence_floor_crossing(ens, w=(0.5, 0.5), p=0.05, B_lo=5, B_hi=9)
    lam, M = fk.limit_masses(B=1.0, w=(0.5, 0.5), spec=spec) # stick-breaking law
    dec  = fk.simulate_ensemble(spec, 200_000, name="n9_decl", tag=fk.TAGS["N9"],
                                mode="declared",
                                declared=[dict(B=4.0, w=(0.5, 0.5)), dict(B=4.0, w=(0.5, 0.5), cap=1.0)])
    res  = fk.load_ensemble("n9_decl").declared()            # per-step f, batches, moments

CLI
---
    python3 exact_m_prr_fk_exact_law.py simulate --m 2 --eps 0.1 --paths 2e5 --name n0_m2_eps0.1 --tag 80
    python3 exact_m_prr_fk_exact_law.py selftest
    python3 exact_m_prr_fk_exact_law.py validate-n0
    python3 exact_m_prr_fk_exact_law.py info --name n0_m2_eps0.1

Machine etiquette: at most 3 worker processes; ``simulate`` waits (60 s polls, up
to 30 min) while 9 or more python3 processes are running.  Runs are chunked (5e4
paths per chunk, ~1 min of one core per chunk at m = 2, eps = 0.1).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import validate_exact_m_offlattice as base  # noqa: E402  (hash-frozen; imported only)
import exact_m_prr_upgrade_core as core  # noqa: E402

REPORT = HERE.parents[1]
FB_DATA = REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
INDEX_DIR = FB_DATA / "fk_ensembles"
ENSEMBLE_ROOT = Path(
    os.environ.get("PRR_FK_ENSEMBLE_ROOT",
                   str(Path.home() / ".local-build" / "prr_fk_ensembles"))
)
SCOUT_CACHE = Path.home() / ".local-build" / "prr_gap_fk_cache"
FIGURES = core.FIGURES

BASE_SEED = 20260923
TAGS = {"N0": 80, "N1": 81, "N3": 82, "N4": 83, "N5": 84, "N6": 85, "N8": 86,
        "N9": 87, "spare88": 88, "spare89": 89}
USED_TAGS_ELSEWHERE = (1, 2, 3, 11, 12, 21, 31, 32, 33, 34, 41, 42, 51, 52, 53,
                       61, 62, 63, 66, 72, 9071, 9072)

DEFAULT_CHUNK = 50_000
DEFAULT_BATCH = 5_000          # declared-mode batch size (paths)
DEFAULT_GROUPS = 20            # stored-mode path groups (batch means / jackknife)
MAX_WORKERS = 3
GAUSS_CUTOFF_SD = 12.0         # default kernel truncation: exp(-72) ~ 5e-32 relative
PY_PROCESS_LIMIT = 9

MODEL = base.MODEL
WINDOW = base.WINDOW
WINDOW_BIN = base.WINDOW_BIN
TARGET_TIMES = base.TARGET_TIMES

VARIANT_SHORTHANDS = {
    "full": dict(gate="contact", a=None, field="mid"),
    "nogate": dict(gate="none", a=None, field="mid"),
    "free": dict(gate="none", a=None, field="mid"),
    "meancontact": dict(gate="mean", a=None, field="mid"),
    "p1": dict(gate="contact", a=None, field="p1"),
    "p2": dict(gate="contact", a=None, field="p2"),
}
FIELD_SHIFT = {"mid": 0.0, "p1": 0.5, "p2": -0.5}


# ============================================================================
# Specification
# ============================================================================


@dataclass(frozen=True)
class EnsembleSpec:
    """Path law + kernel of one ensemble.

    Path law (enters the seed): eps, dt, z0, var_z0_scale, r_par0, r_perp0, u0,
    sigma_perp0, n_perp, gamma, d0, z_bar, torus_w.  Kernel (does not enter the
    seed): m / target_times / centres_z, slab_shape, slab_sd, rho, contact_a
    (default gate radius; variants may override), kernel_cutoff_sd (Gaussian
    slabs are evaluated only within this many sd of a centre; 12 sd changes
    the rate by < 6e-32 of its peak), tmax (paths are prefixes).
    Defaults are the production model (validate_exact_m_offlattice.MODEL).
    """

    m: int
    eps: float
    dt: float = base.DEFAULT_DT
    tmax: float = base.DEFAULT_TMAX
    z0: float = MODEL.z0
    var_z0_scale: float = 1.0
    r_par0: float = MODEL.r_par0
    r_perp0: float = MODEL.r_perp0
    u0: float = MODEL.u0
    sigma_perp0: float = MODEL.sigma_perp0
    n_perp: int = 1
    gamma: float = MODEL.gamma
    d0: float = MODEL.d0
    z_bar: float = MODEL.z_bar
    torus_w: float = MODEL.torus_w
    contact_a: float = MODEL.contact_a
    rho: float = MODEL.rho
    slab_shape: str = "gauss"
    slab_sd: float | None = None
    target_times: tuple | None = None
    centres_z: tuple | None = None
    kernel_cutoff_sd: float = GAUSS_CUTOFF_SD

    # -- derived -----------------------------------------------------------
    def model(self):
        return replace(MODEL, gamma=self.gamma, d0=self.d0, z_bar=self.z_bar,
                       z0=self.z0, torus_w=self.torus_w, contact_a=self.contact_a,
                       rho=self.rho, u0=self.u0, sigma_perp0=self.sigma_perp0,
                       r_par0=self.r_par0, r_perp0=self.r_perp0, dim=self.n_perp + 1)

    def times(self) -> tuple:
        if self.target_times is not None:
            return tuple(float(t) for t in self.target_times)
        if self.centres_z is not None:
            return tuple(float(-math.log((c - self.z_bar) / (self.z0 - self.z_bar)) / self.gamma)
                         for c in self.centres_z)
        return tuple(TARGET_TIMES[self.m])

    def centres(self) -> np.ndarray:
        if self.centres_z is not None:
            return np.asarray(self.centres_z, dtype=float)
        return np.array([self.z_bar + (self.z0 - self.z_bar) * math.exp(-self.gamma * t)
                         for t in self.times()])

    def sd(self) -> float:
        return float(self.slab_sd) if self.slab_sd is not None else self.eps * self.rho

    def steps(self) -> int:
        n = int(round(self.tmax / self.dt))
        if abs(n * self.dt - self.tmax) > 1e-9 * max(1.0, self.tmax):
            raise ValueError("tmax must be an integer multiple of dt")
        return n

    def mu_prime_abs(self, t) -> np.ndarray:
        return self.gamma * abs(self.z0 - self.z_bar) * np.exp(-self.gamma * np.asarray(t, float))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_times_resolved"] = list(self.times())
        d["centres_z_resolved"] = [float(c) for c in self.centres()]
        d["slab_sd_resolved"] = self.sd()
        return d

    @staticmethod
    def from_dict(d: dict) -> "EnsembleSpec":
        keys = EnsembleSpec.__dataclass_fields__.keys()
        kw = {k: d[k] for k in keys if k in d}
        for k in ("target_times", "centres_z"):
            if kw.get(k) is not None:
                kw[k] = tuple(kw[k])
        return EnsembleSpec(**kw)


def path_entropy(spec: EnsembleSpec, *, seed: int, tag: int, replicate: int,
                 chunk: int) -> list[int]:
    """SeedSequence entropy from path-law parameters only (see module doc)."""
    mod = 1 << 63

    def q(x, s=1e9):
        return int(round(float(x) * s)) % mod

    return [int(seed) % mod, int(tag) % mod, int(replicate) % mod, int(spec.n_perp),
            q(spec.eps), q(spec.dt, 1e15), q(spec.z0), q(spec.var_z0_scale),
            q(spec.r_par0), q(spec.r_perp0), q(spec.u0), q(spec.sigma_perp0),
            q(spec.gamma), q(spec.d0), q(spec.z_bar), q(spec.torus_w), int(chunk)]


def parse_variant(text: str, spec: EnsembleSpec) -> dict:
    """'full' | 'nogate' | 'meancontact' | 'p1' | 'p2' | 'a0.2' | 'gate=..,a=..,field=..'."""
    text = text.strip()
    if text in VARIANT_SHORTHANDS:
        v = dict(VARIANT_SHORTHANDS[text])
    elif text.startswith("a") and text[1:].replace(".", "", 1).isdigit():
        v = dict(gate="contact", a=float(text[1:]), field="mid")
    else:
        v = dict(gate="contact", a=None, field="mid")
        for part in text.split(","):
            k, val = part.split("=")
            k = k.strip()
            if k == "a":
                v["a"] = float(val)
            elif k in ("gate", "field"):
                v[k] = val.strip()
            else:
                raise ValueError(f"unknown variant key {k!r}")
    if v["gate"] not in ("contact", "none", "mean"):
        raise ValueError(f"bad gate {v['gate']!r}")
    if v["field"] not in FIELD_SHIFT:
        raise ValueError(f"bad field {v['field']!r}")
    if v["a"] is None:
        v["a"] = spec.contact_a
    v["name"] = text if "=" not in text else (
        f"g{v['gate']}_a{v['a']:g}_f{v['field']}")
    return v


# ============================================================================
# Checkpoint grids
# ============================================================================


def step_times(dt: float, steps: int) -> np.ndarray:
    """Kill times of the production simulator: (step + 1) * dt, bitwise."""
    return (np.arange(steps) + 1) * dt


def production_window_edges() -> np.ndarray:
    """Identical to validate_exact_m_offlattice.classify_modes edges."""
    lo, hi = WINDOW
    return np.arange(lo, hi + 0.5 * WINDOW_BIN, WINDOW_BIN)


def build_grid(spec: EnsembleSpec, grid: str = "production") -> dict:
    """Checkpoint step counts for the stored mode.

    'production': 0.02-bin grid on [0, tmax] whose window part [0.5, 3.5]
        uses the exact production classifier edges and numpy.histogram
        semantics (the window's upper edge and tmax inclusive).
    'nominal:<bin>[:<lo>:<hi>]': uniform nominal edges, histogram semantics.
    """
    steps = spec.steps()
    T = step_times(spec.dt, steps)
    if grid == "production":
        win = production_window_edges()
        lo_edges = np.arange(0.0, WINDOW[0] - 0.5 * WINDOW_BIN, WINDOW_BIN)
        nhi = int(round((spec.tmax - WINDOW[1]) / WINDOW_BIN))
        hi_edges = WINDOW[1] + WINDOW_BIN * np.arange(1, nhi + 1)
        edges = np.concatenate([lo_edges, win, hi_edges])
        cps = np.searchsorted(T, edges, side="left")
        iw_hi = lo_edges.size + win.size - 1
        cps[iw_hi] = np.searchsorted(T, win[-1], side="right")
        window_index = np.arange(lo_edges.size, lo_edges.size + win.size)
    elif grid.startswith("nominal:"):
        parts = grid.split(":")
        bw = float(parts[1])
        lo = float(parts[2]) if len(parts) > 2 else 0.0
        hi = float(parts[3]) if len(parts) > 3 else spec.tmax
        n = int(round((hi - lo) / bw))
        edges = lo + bw * np.arange(n + 1)
        cps = np.searchsorted(T, edges, side="left")
        wmask = (edges >= WINDOW[0] - 1e-12) & (edges <= WINDOW[1] + 1e-12)
        window_index = np.flatnonzero(wmask)
    else:
        raise ValueError(f"unknown grid {grid!r}")
    if edges[-1] >= spec.tmax - 1e-12:
        cps[-1] = steps
    if np.any(np.diff(cps) <= 0):
        raise ValueError("checkpoint step counts must be strictly increasing")
    return {"grid": grid, "edges": edges, "cps": cps.astype(np.int64),
            "window_index": window_index.astype(np.int64)}


# ============================================================================
# Kernel helpers
# ============================================================================


def _slab_components(F: np.ndarray, centres: np.ndarray, spec: EnsembleSpec) -> tuple:
    """Active indices and per-slab unit-mass densities * dt / W^n_perp."""
    sd = spec.sd()
    dmin = np.min(np.abs(F[:, None] - centres[None, :]), axis=1)
    if spec.slab_shape == "gauss":
        act = np.flatnonzero(dmin < spec.kernel_cutoff_sd * sd)
        if act.size == 0:
            return act, None
        diff = F[act, None] - centres[None, :]
        pref = spec.dt / (math.sqrt(2.0 * math.pi) * sd * spec.torus_w ** spec.n_perp)
        comps = np.exp(-(diff * diff) * (1.0 / (2.0 * sd * sd))) * pref
    elif spec.slab_shape == "tophat":
        half = math.sqrt(3.0) * sd
        act = np.flatnonzero(dmin < half)
        if act.size == 0:
            return act, None
        diff = F[act, None] - centres[None, :]
        pref = spec.dt / (2.0 * half * spec.torus_w ** spec.n_perp)
        comps = (np.abs(diff) < half).astype(float) * pref
    else:
        raise ValueError(f"unknown slab_shape {spec.slab_shape!r}")
    return act, comps


def mean_contact_curve(spec: EnsembleSpec, a: float, T: np.ndarray) -> np.ndarray:
    """c(t) = P(|R_t|_mi < a) for the unkilled relative process (exact quadrature)."""
    p = replace(spec.model(), contact_a=a)
    out = np.empty(T.size)
    block = 250
    for s in range(0, T.size, block):
        tt = T[s:s + block]
        if spec.n_perp == 1:
            out[s:s + block] = base.contact_probability(tt, spec.eps, p)
        elif spec.n_perp == 2:
            out[s:s + block] = core.contact_probability_d3(tt, spec.eps, p)
        else:
            raise ValueError("n_perp must be 1 or 2")
    return out


def em_moments(spec: EnsembleSpec) -> dict:
    """Exact Gaussian moments of the Euler--Maruyama chain after n = 1..steps steps.

    The EM recursions are linear with Gaussian noise, so Z_n, R_par,n, R_perp,n
    are exactly Gaussian with these means/variances (R_perp before wrapping).
    """
    steps = spec.steps()
    dt = spec.dt
    a = 1.0 - spec.gamma * dt
    n = np.arange(1, steps + 1)
    decay = a ** n
    mz = spec.z_bar + (spec.z0 - spec.z_bar) * decay
    vz = np.empty(steps)
    vr = np.empty(steps)
    v = spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)
    u = (spec.eps * spec.u0) ** 2
    for i in range(steps):
        v = a * a * v + spec.eps**2 * spec.d0 * dt
        u = a * a * u + 4.0 * spec.eps**2 * spec.d0 * dt
        vz[i] = v
        vr[i] = u
    return {"t": n * dt, "mean_z": mz, "var_z": vz, "mean_par": spec.r_par0 * decay,
            "var_par": vr,
            "var_perp": spec.eps**2 * spec.sigma_perp0**2 + 4.0 * spec.eps**2 * spec.d0 * n * dt}


def _folded_min_image_density(y: np.ndarray, mean: float, var: np.ndarray, W: float,
                              images: int = 10) -> np.ndarray:
    """Density of the minimum-image distance |x|_mi on [0, W/2] for x ~ N(mean, var) mod W."""
    k = np.arange(-images, images + 1) * W
    out = np.zeros((var.size, y.size))
    for sgn in (1.0, -1.0):
        off = sgn * y[None, :, None] - mean - k[None, None, :]
        out += (np.exp(-off**2 / (2.0 * var[:, None, None]))
                / np.sqrt(2.0 * math.pi * var)[:, None, None]).sum(axis=2)
    return out


def contact_probability_em(spec: EnsembleSpec, a: float | None = None, n_theta: int = 256,
                           block: int = 200) -> np.ndarray:
    """P(|R_n|_mi < a) for the EM relative chain at every step (arbitrary r_perp0, d = 2, 3).

    Quadrature in the substituted variable y = a sin(theta) (d = 2) or polar
    radius r = a sin(theta) (d = 3), which removes the chord's square-root
    endpoint singularity; requires a <= W/2.
    """
    a = spec.contact_a if a is None else a
    W = spec.torus_w
    if a > 0.5 * W:
        raise NotImplementedError("contact_probability_em assumes a <= W/2")
    mom = em_moments(spec)
    th, wth = np.polynomial.legendre.leggauss(n_theta)
    th = 0.25 * math.pi * (th + 1.0)
    wth = 0.25 * math.pi * wth
    out = np.empty(mom["t"].size)
    erf = np.vectorize(math.erf)
    for s0 in range(0, out.size, block):
        sl = slice(s0, s0 + block)
        mp = mom["mean_par"][sl][:, None]
        sp = np.sqrt(mom["var_par"][sl])[:, None]
        vp = mom["var_perp"][sl]
        if spec.n_perp == 1:
            y = a * np.sin(th)
            chord = a * np.cos(th)
            jac = a * np.cos(th) * wth
            fy = _folded_min_image_density(y, spec.r_perp0 % W, vp, W)
            ppar = 0.5 * (erf((chord[None, :] - mp) / (math.sqrt(2) * sp))
                          - erf((-chord[None, :] - mp) / (math.sqrt(2) * sp)))
            out[sl] = np.sum(fy * ppar * jac[None, :], axis=1)
        elif spec.n_perp == 2:
            if spec.r_perp0 != 0.0:
                raise NotImplementedError("d = 3 EM contact assumes r_perp0 = 0")
            r = a * np.sin(th)                      # radius in the (y1, y2) quarter plane
            chord = a * np.cos(th)
            jr = a * np.cos(th) * wth * r           # dr * r
            ph, wph = np.polynomial.legendre.leggauss(n_theta)
            ph = 0.25 * math.pi * (ph + 1.0)
            wph = 0.25 * math.pi * wph
            y1 = (r[:, None] * np.cos(ph)[None, :]).ravel()
            y2 = (r[:, None] * np.sin(ph)[None, :]).ravel()
            f1 = _folded_min_image_density(y1, 0.0, vp, W)
            f2 = _folded_min_image_density(y2, 0.0, vp, W)
            wts = (jr[:, None] * wph[None, :]).ravel()
            ch = np.repeat(chord, ph.size)
            ppar = 0.5 * (erf((ch[None, :] - mp) / (math.sqrt(2) * sp))
                          - erf((-ch[None, :] - mp) / (math.sqrt(2) * sp)))
            out[sl] = np.sum(f1 * f2 * ppar * wts[None, :], axis=1)
        else:
            raise ValueError("n_perp must be 1 or 2")
    return np.clip(out, 0.0, 1.0)


def free_exposure_discrete(spec: EnsembleSpec, w, *, gate: str = "contact",
                           a: float | None = None) -> dict:
    """Exact unit-budget free-exposure clock G_n = E[V_n] of the EM chain (field at Z).

    G_n = g_n * sum_j w_j E[phi_j(Z_n)] / W^(d-1), with Z_n ~ N(mean_z, var_z)
    exactly (EM is a Gaussian linear recursion) and g_n = P(contact) under EM
    (gate 'contact'), 1 ('none') or the continuous-time c(t) used by the
    'meancontact' variant ('mean').  Z is independent of R, so the product is
    exact.  Gaussian or top-hat slabs.  Differences from the continuum G(t)
    (validate_exact_m_offlattice / core.free_exposure_general) are the O(dt)
    Euler--Maruyama bias.
    """
    mom = em_moments(spec)
    w = np.asarray(w, float)
    sd = spec.sd()
    h = np.zeros(mom["t"].size)
    for wj, c in zip(w, spec.centres()):
        if spec.slab_shape == "gauss":
            v = mom["var_z"] + sd * sd
            h += wj * np.exp(-(mom["mean_z"] - c) ** 2 / (2 * v)) / np.sqrt(2 * math.pi * v)
        else:
            hw = math.sqrt(3.0) * sd
            s_ = np.sqrt(2.0 * mom["var_z"])
            erf = np.vectorize(math.erf)
            h += wj * 0.5 * (erf((c + hw - mom["mean_z"]) / s_)
                             - erf((c - hw - mom["mean_z"]) / s_)) / (2 * hw)
    h /= spec.torus_w ** spec.n_perp
    if gate == "contact":
        g = contact_probability_em(spec, a)
    elif gate == "none":
        g = np.ones_like(h)
    elif gate == "mean":
        g = mean_contact_curve(spec, spec.contact_a if a is None else a, mom["t"])
    else:
        raise ValueError(gate)
    return {"t": mom["t"], "G": g * h, "gate_factor": g, "field_factor": h}


# ============================================================================
# Chunk worker
# ============================================================================


def _run_chunk(task: dict) -> dict:
    spec = EnsembleSpec.from_dict(task["spec"])
    n = int(task["size"])
    steps = spec.steps()
    dt = spec.dt
    mode = task["mode"]
    variants = task["variants"]
    centres = spec.centres()
    m = centres.size
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()

    decay = 1.0 - spec.gamma * dt
    pull = spec.gamma * spec.z_bar * dt
    noise_z = spec.eps * math.sqrt(spec.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    W = spec.torus_w

    # Initial law, identical draw order to the production/core simulators.
    z = spec.z0 + math.sqrt(spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)) \
        * rng.standard_normal(n)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    r_perp = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0
                    * rng.standard_normal((spec.n_perp, n)), W)

    fields = sorted({v["field"] for v in variants})
    mean_c = task.get("mean_contact") or {}

    tj_steps = [int(round(tj / dt)) for tj in spec.times()]
    dist2_tj = np.zeros((n, m), np.float32)

    if mode == "stored":
        cps = np.asarray(task["cps"], np.int64)
        K = cps.size
        X = {v["name"]: np.zeros((n, m)) for v in variants}
        Xlast = {v["name"]: np.zeros((n, m)) for v in variants}
        buf = {v["name"]: np.zeros((n, m, K), np.float32) for v in variants}
        kidx = 0
        while kidx < K and cps[kidx] == 0:
            kidx += 1
    else:
        decl = task["declared"]
        nd = len(decl)
        batch = int(task["batch"])
        nb_local = -(-n // batch)
        bid = np.arange(n) // batch
        # distinct (variant, w) exposure channels
        pairs = task["pairs"]
        npair = len(pairs)
        Xp = np.zeros((n, npair))
        H = np.zeros((n, nd))
        f_sum = np.zeros((nd, steps))
        f_sq = np.zeros((nd, steps))
        f_b = np.zeros((nd, nb_local, steps))
        mx = np.zeros((npair, steps)); mxx = np.zeros((npair, steps))
        mv = np.zeros((npair, steps)); mvx = np.zeros((npair, steps))
        mx_b = np.zeros((npair, nb_local, steps)); mxx_b = np.zeros((npair, nb_local, steps))
        mv_b = np.zeros((npair, nb_local, steps)); mvx_b = np.zeros((npair, nb_local, steps))
        pair_w = [np.asarray(p["w"], float) for p in pairs]
        pair_var = [p["variant"] for p in pairs]
        vmap = {v["name"]: v for v in variants}
        decl_pair = [int(d["pair"]) for d in decl]
        decl_B = np.array([float(d["B"]) for d in decl])
        decl_cap = [d.get("cap") for d in decl]

    contact_steps = 0
    max_step_exposure = 0.0
    for s in range(steps):
        noise = rng.standard_normal((2 + spec.n_perp, n))
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        r_perp += noise_r * noise[2:]
        np.mod(r_perp, W, out=r_perp)
        perp_mi = np.minimum(r_perp, W - r_perp)
        dist2 = r_par * r_par + np.sum(perp_mi * perp_mi, axis=0)
        step_no = s + 1
        for jj, sj in enumerate(tj_steps):
            if step_no == sj:
                dist2_tj[:, jj] = dist2
        contact_steps += int(np.count_nonzero(dist2 < spec.contact_a**2))

        comp_by_field = {}
        for fld in fields:
            F = z if fld == "mid" else z + FIELD_SHIFT[fld] * r_par
            comp_by_field[fld] = _slab_components(F, centres, spec)

        def gated(v):
            act, comps = comp_by_field[v["field"]]
            if act.size == 0:
                return act, None
            if v["gate"] == "contact":
                g = (dist2[act] < v["a"] * v["a"]).astype(float)
            elif v["gate"] == "none":
                g = None
            else:
                g = np.full(act.size, mean_c[v["name"]][s])
            return act, (comps if g is None else comps * g[:, None])

        if mode == "stored":
            for v in variants:
                act, c = gated(v)
                if act.size:
                    X[v["name"]][act] += c
            if kidx < K and step_no == cps[kidx]:
                for v in variants:
                    nm = v["name"]
                    # float64 increment over (c_{k-1}, c_k], stored as float32:
                    # full relative precision even for 1e-30 bin exposures.
                    buf[nm][:, :, kidx] = X[nm] - Xlast[nm]
                    Xlast[nm][:] = X[nm]
                kidx += 1
        else:
            gv = {v["name"]: gated(v) for v in variants}
            for ip in range(npair):
                act, c = gv[pair_var[ip]]
                if act.size:
                    e = _wsum(c, pair_w[ip])
                    mx_e = float(e.max())
                    if mx_e > max_step_exposure:
                        max_step_exposure = mx_e
                    # declared items on this channel (use H before update)
                    for idd in range(nd):
                        if decl_pair[idd] != ip:
                            continue
                        Bv = decl_B[idd]
                        if decl_cap[idd] is None:
                            h = Bv * e
                        else:
                            h = np.minimum(Bv * e / dt, float(decl_cap[idd])) * dt
                        y = -np.expm1(-h) * np.exp(-H[act, idd])
                        f_sum[idd, s] += y.sum()
                        f_sq[idd, s] += (y * y).sum()
                        f_b[idd, :, s] += np.bincount(bid[act], weights=y, minlength=nb_local)
                        H[act, idd] += h
                    Xp[act, ip] += e
                    v_rate = e / dt
                    mv[ip, s] += v_rate.sum()
                    mvx[ip, s] += (v_rate * Xp[act, ip]).sum()
                    mv_b[ip, :, s] += np.bincount(bid[act], weights=v_rate, minlength=nb_local)
                    mvx_b[ip, :, s] += np.bincount(bid[act], weights=v_rate * Xp[act, ip],
                                                   minlength=nb_local)
                xcol = Xp[:, ip]
                mx[ip, s] += xcol.sum()
                mxx[ip, s] += (xcol * xcol).sum()
                mx_b[ip, :, s] += np.bincount(bid, weights=xcol, minlength=nb_local)
                mxx_b[ip, :, s] += np.bincount(bid, weights=xcol * xcol, minlength=nb_local)

    runtime = time.perf_counter() - t0
    out = {"chunk": int(task["chunk"]), "size": n, "runtime_seconds": runtime,
           "contact_steps": contact_steps}
    fpath = Path(task["file"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_suffix(".tmp.npz")
    if mode == "stored":
        for v in variants:
            arr = buf[v["name"]]
            if not np.all(np.isfinite(arr)) or arr.min() < 0.0:
                raise AssertionError("non-finite or negative exposure increment")
            tot = arr.astype(np.float64).sum(axis=2)
            if not np.allclose(tot, X[v["name"]], rtol=1e-6, atol=1e-30):
                raise AssertionError("increments do not add up to the total exposure")
        payload = {f"dX_{v['name']}": buf[v["name"]] for v in variants}
        payload["dist2_tj"] = dist2_tj
        np.savez_compressed(tmp, **payload)
    else:
        np.savez_compressed(
            tmp, f_sum=f_sum, f_sq=f_sq, f_b=f_b, mx=mx, mxx=mxx, mv=mv, mvx=mvx,
            mx_b=mx_b, mxx_b=mxx_b, mv_b=mv_b, mvx_b=mvx_b, dist2_tj=dist2_tj)
        out["max_step_unit_exposure"] = max_step_exposure
    os.replace(tmp, fpath)
    out["file"] = str(fpath)
    out["sha256"] = _sha256(fpath)
    out["bytes"] = fpath.stat().st_size
    return out


def _wsum(c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sum_j c[:, j] w[j] without BLAS (Accelerate matmul raises spurious FP warnings)."""
    e = c[:, 0] * w[0]
    for j in range(1, w.size):
        e += c[:, j] * w[j]
    return e


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================================
# Simulation driver
# ============================================================================


def python_process_count() -> int:
    try:
        out = subprocess.run(["pgrep", "-f", "python3"], capture_output=True, text=True,
                             timeout=20).stdout
        return len([ln for ln in out.split() if ln.strip()])
    except Exception:
        return 0


def wait_for_cpu_slot(limit: int = PY_PROCESS_LIMIT, poll: float = 60.0,
                      max_wait: float = 1800.0, verbose: bool = True) -> dict:
    started = time.time()
    while True:
        n = python_process_count()
        if n < limit + 1:  # this process counts itself
            return {"python_processes": n, "waited_seconds": time.time() - started}
        if time.time() - started > max_wait:
            return {"python_processes": n, "waited_seconds": time.time() - started,
                    "gave_up_waiting": True}
        if verbose:
            print(f"[fk] {n} python3 processes running; waiting {poll:.0f}s", flush=True)
        time.sleep(poll)


SELFTEST_INDEX_DIR = ENSEMBLE_ROOT / "_selftest_index"


def index_path(name: str, index_dir: Path | None = None) -> Path:
    if index_dir is None and name.startswith("selftest_"):
        index_dir = SELFTEST_INDEX_DIR   # keep throw-away self-test indices out of the repo
    return (INDEX_DIR if index_dir is None else Path(index_dir)) / f"{name}.json"


def simulate_ensemble(spec: EnsembleSpec, n_paths: int, *, name: str, tag: int,
                      seed: int = BASE_SEED, replicate: int = 0, mode: str = "stored",
                      variants=("full",), declared=None, grid: str = "production",
                      workers: int = MAX_WORKERS, chunk: int = DEFAULT_CHUNK,
                      batch: int = DEFAULT_BATCH, chunks_subset=None,
                      overwrite: bool = False, wait: bool = True,
                      verbose: bool = True, note: str = "") -> "Ensemble":
    """Simulate (or resume) an ensemble; write chunk files + the index JSON.

    ``chunks_subset`` (list of chunk indices) lets a long ensemble be built in
    several invocations; already-present chunks with matching size are skipped
    unless ``overwrite``.  ``declared`` (mode 'declared') is a list of dicts
    {B, w, variant='full', cap=None, label}.
    """
    if workers > MAX_WORKERS:
        raise ValueError(f"at most {MAX_WORKERS} workers (machine etiquette)")
    if tag in USED_TAGS_ELSEWHERE:
        raise ValueError(f"tag {tag} already used by another stream")
    variants = [parse_variant(v, spec) if isinstance(v, str) else v for v in variants]
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("duplicate variant names")
    steps = spec.steps()
    T = step_times(spec.dt, steps)
    sizes = [chunk] * (n_paths // chunk) + ([n_paths % chunk] if n_paths % chunk else [])
    entropy = path_entropy(spec, seed=seed, tag=tag, replicate=replicate, chunk=chunk)
    children = np.random.SeedSequence(entropy).spawn(len(sizes))
    ens_dir = ENSEMBLE_ROOT / name
    ens_dir.mkdir(parents=True, exist_ok=True)

    mean_contact = {}
    for v in variants:
        if v["gate"] == "mean":
            mean_contact[v["name"]] = mean_contact_curve(spec, v["a"], T)

    idx = {"name": name, "schema": "fk_exact_law_ensemble_v1", "mode": mode,
           "spec": spec.to_dict(), "variants": variants, "n_paths": int(n_paths),
           "chunk": int(chunk), "chunk_sizes": sizes, "seed": int(seed), "tag": int(tag),
           "replicate": int(replicate), "seed_entropy": entropy,
           "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
           "ensemble_dir": str(ens_dir), "note": note,
           "driver": str(HERE.name),
           "exactness": ("exact discrete-time Feynman-Kac law of the production "
                         "Euler-Maruyama + end-of-step Doi killing process; sampling "
                         "error only (iid per-path SE, path-group batch means)")}
    if mode == "stored":
        g = build_grid(spec, grid)
        idx.update({"grid": g["grid"], "edges": [float(e) for e in g["edges"]],
                    "cps": [int(c) for c in g["cps"]],
                    "window_index": [int(i) for i in g["window_index"]],
                    "storage": ("float32 unit-budget exposure INCREMENTS dX[path, slab, k] = "
                                "X(c_k) - X(c_{k-1}) (c_{-1} = 0 steps); X = cumsum in float64")})
    elif mode == "declared":
        if not declared:
            raise ValueError("declared mode needs a declared list")
        vnames = {v["name"] for v in variants}
        pairs, decl = [], []
        for d in declared:
            var = d.get("variant", "full")
            if var not in vnames:
                v = parse_variant(var, spec)
                variants.append(v)
                vnames.add(v["name"])
                if v["gate"] == "mean":
                    mean_contact[v["name"]] = mean_contact_curve(spec, v["a"], T)
                var = v["name"]
            w = [float(x) for x in d["w"]]
            key = (var, tuple(w))
            ip = next((i for i, p in enumerate(pairs) if (p["variant"], tuple(p["w"])) == key), None)
            if ip is None:
                pairs.append({"variant": var, "w": w})
                ip = len(pairs) - 1
            decl.append({"B": float(d["B"]), "w": w, "variant": var, "cap": d.get("cap"),
                         "pair": ip, "label": d.get("label", f"{var}_B{float(d['B']):g}"
                                                    + (f"_cap{d['cap']:g}" if d.get("cap") else ""))})
        idx.update({"declared": decl, "pairs": pairs, "batch": int(batch),
                    "variants": variants,
                    "storage": "per-step sums of exact kill probability and exposure moments"})
        if chunk % batch:
            raise ValueError("chunk must be a multiple of batch in declared mode")
    else:
        raise ValueError("mode must be 'stored' or 'declared'")

    existing = {}
    ip_ = index_path(name)
    if ip_.exists() and not overwrite:
        old = json.loads(ip_.read_text())
        for key in ("seed_entropy", "mode", "chunk_sizes", "variants", "cps", "declared"):
            if json.loads(json.dumps(old.get(key), default=_json_default)) != \
                    json.loads(json.dumps(idx.get(key), default=_json_default)):
                raise ValueError(f"existing index {ip_} differs in {key}; use overwrite")
        for c in old.get("chunks", []):
            if Path(c["file"]).exists():
                existing[int(c["chunk"])] = c

    todo = list(range(len(sizes))) if chunks_subset is None else list(chunks_subset)
    todo = [i for i in todo if overwrite or i not in existing]
    tasks = []
    for i in todo:
        t = {"spec": spec.to_dict(), "size": sizes[i], "seedseq": children[i], "mode": mode,
             "variants": variants, "chunk": i, "file": str(ens_dir / f"chunk{i:04d}.npz"),
             "mean_contact": mean_contact}
        if mode == "stored":
            t["cps"] = idx["cps"]
        else:
            t.update({"declared": idx["declared"], "pairs": idx["pairs"], "batch": batch})
        tasks.append(t)

    wait_info = wait_for_cpu_slot(verbose=verbose) if (wait and tasks) else {}
    started = time.time()
    results = []
    if tasks:
        if workers <= 1 or len(tasks) == 1:
            for t in tasks:
                results.append(_run_chunk(t))
                if verbose:
                    print(f"[fk] {name} chunk {results[-1]['chunk']} done "
                          f"{results[-1]['runtime_seconds']:.1f}s", flush=True)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(workers, len(tasks))) as pool:
                for r in pool.imap_unordered(_run_chunk, tasks):
                    results.append(r)
                    if verbose:
                        print(f"[fk] {name} chunk {r['chunk']} done "
                              f"{r['runtime_seconds']:.1f}s", flush=True)
    for r in results:
        existing[int(r["chunk"])] = r
    idx["chunks"] = [existing[i] for i in sorted(existing)]
    idx["complete"] = len(idx["chunks"]) == len(sizes)
    idx["last_invocation"] = {"wall_seconds": time.time() - started, "workers": workers,
                              "chunks_run": [int(r["chunk"]) for r in results],
                              "cpu_wait": wait_info,
                              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    idx["process_seconds_total"] = float(sum(c["runtime_seconds"] for c in idx["chunks"]))
    ip_.parent.mkdir(parents=True, exist_ok=True)
    tmp = ip_.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, indent=1, default=_json_default))
    os.replace(tmp, ip_)
    return Ensemble(idx)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.random.SeedSequence):
        return {"entropy": o.entropy, "spawn_key": list(o.spawn_key)}
    raise TypeError(type(o))


# ============================================================================
# Ensemble access
# ============================================================================


class Ensemble:
    """Lazy view of an ensemble written by ``simulate_ensemble``."""

    def __init__(self, index: dict):
        self.index = index
        self.name = index["name"]
        self.mode = index["mode"]
        self.spec = EnsembleSpec.from_dict(index["spec"])
        self.variants = {v["name"]: v for v in index["variants"]}
        if self.mode == "stored":
            self.edges = np.asarray(index["edges"], float)
            self.cps = np.asarray(index["cps"], np.int64)
            self.window_index = np.asarray(index["window_index"], np.int64)
        self.chunks = index.get("chunks", [])
        self.n_paths = int(sum(c["size"] for c in self.chunks))
        self.cache_in_memory = False
        self._cache: dict = {}

    def __repr__(self):
        return (f"Ensemble({self.name!r}, mode={self.mode}, m={self.spec.m}, "
                f"eps={self.spec.eps}, paths={self.n_paths}, variants={list(self.variants)})")

    def iter_chunks(self, variant: str = "full"):
        """Yield (global_start, dX float32 (n, m, K), dist2_tj) per chunk (stored mode).

        dX[:, j, k] is the unit-budget (B = 1, w_j = 1) exposure of slab j over
        checkpoint interval (cps[k-1], cps[k]] (first interval from step 0);
        the cumulative exposure is ``np.cumsum(dX, axis=2, dtype=np.float64)``.

        Set ``ens.cache_in_memory = True`` to keep decompressed chunks in RAM
        across calls (m=3, 2e5 paths: ~0.5 GB per variant); ``ens.clear_cache()``.
        """
        if self.mode != "stored":
            raise ValueError("iter_chunks needs a stored ensemble")
        start = 0
        for c in self.chunks:
            key = (c["file"], variant)
            if key in self._cache:
                X, d2 = self._cache[key]
            else:
                with np.load(c["file"]) as d:
                    X = d[f"dX_{variant}"]
                    d2 = d["dist2_tj"]
                if self.cache_in_memory:
                    self._cache[key] = (X, d2)
            yield start, X, d2
            start += int(c["size"])

    def clear_cache(self):
        self._cache.clear()

    def declared(self) -> dict:
        """Merge declared-mode accumulators over chunks; return estimates and SEs."""
        if self.mode != "declared":
            raise ValueError("not a declared ensemble")
        acc = None
        for c in self.chunks:
            with np.load(c["file"]) as d:
                part = {k: d[k] for k in d.files if k != "dist2_tj"}
            if acc is None:
                acc = {k: v.copy() for k, v in part.items()}
            else:
                for k in ("f_sum", "f_sq", "mx", "mxx", "mv", "mvx"):
                    acc[k] += part[k]
                for k in ("f_b", "mx_b", "mxx_b", "mv_b", "mvx_b"):
                    acc[k] = np.concatenate([acc[k], part[k]], axis=1)
        N = self.n_paths
        dt = self.spec.dt
        steps = self.spec.steps()
        t = step_times(dt, steps)
        bsz = int(self.index["batch"])
        f = acc["f_sum"] / N
        var_iid = np.maximum(acc["f_sq"] / N - f * f, 0.0)
        out = {"t": t, "dt": dt, "N": N, "declared": self.index["declared"],
               "pairs": self.index["pairs"],
               "p_step": f, "p_step_se_iid": np.sqrt(var_iid / N),
               "density": f / dt, "density_se_iid": np.sqrt(var_iid / N) / dt,
               "batch_p_step": acc["f_b"] / bsz,
               "survival": 1.0 - np.cumsum(f, axis=1)}
        mx = acc["mx"] / N
        vx = np.maximum(acc["mxx"] / N - mx * mx, 0.0)
        mv = acc["mv"] / N
        cov = acc["mvx"] / N - mv * mx
        bm = acc["mx_b"] / bsz
        bv = acc["mxx_b"] / bsz - bm * bm
        nb = bm.shape[1]
        out.update({"mean_X": mx, "var_X": vx * N / max(N - 1, 1),
                    "var_X_se_batch": bv.std(axis=1, ddof=1) / math.sqrt(nb),
                    "G_sampled": mv, "cov_V_X": cov,
                    "batch_mean_X": bm, "batch_G": acc["mv_b"] / bsz})
        return out


def load_ensemble(name_or_path) -> Ensemble:
    p = Path(name_or_path)
    if not (p.suffix == ".json" and p.is_file()):
        p = index_path(str(name_or_path))
    return Ensemble(json.loads(p.read_text()))


# ============================================================================
# Stored-mode analysis
# ============================================================================


def _weights(ens: Ensemble, w) -> np.ndarray:
    m = ens.spec.centres().size
    w = np.full(m, 1.0 / m) if w is None else np.asarray(w, float)
    if w.size != m:
        raise ValueError(f"weights must have length {m}")
    return w


def _group_bounds(N: int, G: int) -> np.ndarray:
    return (np.arange(G + 1) * N) // G


def _group_ids(gb: np.ndarray, start: int, n: int) -> np.ndarray:
    return np.searchsorted(gb, start + np.arange(n), side="right") - 1


def _group_sums(Y: np.ndarray, gid: np.ndarray, G: int) -> np.ndarray:
    """Sum rows of Y by (sorted, contiguous) group id."""
    out = np.zeros((G,) + Y.shape[1:])
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    out[gid[starts]] += np.add.reduceat(Y, starts, axis=0)
    return out


def _selected_exposure(dX: np.ndarray, w: np.ndarray, index: np.ndarray) -> tuple:
    """Cumulative exposure at selected checkpoints and the increments between them.

    dX (n, m, K) float32 per-interval increments; returns (Xsel (n, Ks) float64,
    seg (n, Ks-1) float64) with seg[:, a] = X(c_{index[a+1]}) - X(c_{index[a]})
    computed as a sum of stored increments (no cancellation).
    """
    dXw = np.einsum("imk,m->ik", dX[:, :, : index[-1] + 1].astype(np.float64), w)
    Xsel = np.cumsum(dXw, axis=1)[:, index]
    seg = np.add.reduceat(dXw, index[:-1] + 1, axis=1) if index.size > 1 else \
        np.zeros((dXw.shape[0], 0))
    return Xsel, seg


def survival_table(ens: Ensemble, budgets, w=None, *, variant: str = "full",
                   index=None, groups: int = DEFAULT_GROUPS, second_moments: bool = False,
                   block: int = 25_000) -> dict:
    """One pass over the ensemble for several budgets.

    For selected checkpoints (``index``; default all, e.g. ``ens.window_index``)
    accumulates survival S = E exp(-B X) and bin masses
    P = E[exp(-B X(c_a)) (1 - exp(-B (X(c_{a+1}) - X(c_a))))] (expm1 form, full
    relative precision in deep tails), their path-group sums and optionally the
    per-path second moments of the bin masses (iid SEs).
    """
    w = _weights(ens, w)
    Bs = np.atleast_1d(np.asarray(budgets, float))
    kidx = np.arange(ens.cps.size) if index is None else np.asarray(index)
    K = kidx.size
    N = ens.n_paths
    gb = _group_bounds(N, groups)
    S_g = np.zeros((groups, Bs.size, K))
    P_g = np.zeros((groups, Bs.size, K - 1))
    sq = np.zeros((Bs.size, K - 1)) if second_moments else None
    for start, dX, _ in ens.iter_chunks(variant):
        n = dX.shape[0]
        for b0 in range(0, n, block):
            Xs, seg = _selected_exposure(dX[b0:b0 + block], w, kidx)
            gid = _group_ids(gb, start + b0, Xs.shape[0])
            for ib, B in enumerate(Bs):
                E = np.exp(-B * Xs)
                y = E[:, :-1] * (-np.expm1(-B * seg))
                S_g[:, ib, :] += _group_sums(E, gid, groups)
                P_g[:, ib, :] += _group_sums(y, gid, groups)
                if second_moments:
                    sq[ib] += (y * y).sum(0)
    sizes = np.diff(gb).astype(float)
    out = {"budgets": Bs, "w": w, "index": kidx, "S": S_g.sum(0) / N, "S_group_sums": S_g,
           "P": P_g.sum(0) / N, "P_group_sums": P_g, "group_sizes": sizes, "N": N}
    if second_moments:
        out["bin_sq"] = sq
    return out


def _batch_se(group_means: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Batch-means SE (weighted by group size) along axis 0."""
    Nt = sizes.sum()
    mean = (group_means * sizes.reshape((-1,) + (1,) * (group_means.ndim - 1))).sum(0) / Nt
    G = sizes.size
    dev = group_means - mean
    return np.sqrt((dev * dev).sum(0) / (G * (G - 1)))


def exact_law(ens: Ensemble, B: float, w=None, *, variant: str = "full",
              region: str = "window", groups: int = DEFAULT_GROUPS,
              with_cov: bool = False) -> dict:
    """Exact-law bin masses f_k = E[S(e_k) - S(e_{k+1})] on the stored grid.

    region: 'window' (production classifier bins on [0.5, 3.5]) or 'all'.
    Returns edges, t (bin centres), bin_mass, se_iid (per-path), se_batch
    (path groups), density, group_bin_mass (G, K-1) / group_density, survival
    at the edges, and with ``with_cov`` the (K-1)x(K-1) covariance matrix of the
    bin-mass estimator (FK bin errors are positively correlated across bins).
    """
    index = ens.window_index if region == "window" else np.arange(ens.cps.size)
    tab = survival_table(ens, [B], w, variant=variant, index=index, groups=groups,
                         second_moments=True)
    N = tab["N"]
    p = tab["P"][0]
    sq = tab["bin_sq"][0]
    se_iid = np.sqrt(np.maximum(sq / N - p * p, 0.0) / N)
    sizes = tab["group_sizes"]
    pg = tab["P_group_sums"][:, 0, :] / sizes[:, None]
    se_b = _batch_se(pg, sizes)
    edges = ens.edges[tab["index"]]
    bw = np.diff(edges)
    cov = None
    if with_cov:
        wv = tab["w"]
        C = np.zeros((p.size, p.size))
        for _, dX, _ in ens.iter_chunks(variant):
            Xs, seg = _selected_exposure(dX, wv, np.asarray(tab["index"]))
            y = np.exp(-B * Xs[:, :-1]) * (-np.expm1(-B * seg))
            C += np.einsum("ik,il->kl", y, y)
        cov = (C / N - np.outer(p, p)) / N
    return {"cov": cov, "B": float(B), "w": tab["w"].tolist(), "variant": variant, "N": N,
            "edges": edges, "t": 0.5 * (edges[:-1] + edges[1:]), "bin_mass": p,
            "se_iid": se_iid, "se_batch": se_b, "density": p / bw,
            "density_se_iid": se_iid / bw, "group_bin_mass": pg,
            "group_density": pg / bw[None, :], "group_sizes": sizes,
            "survival_at_edges": tab["S"][0], "cps": ens.cps[tab["index"]]}


def g_valley_times(spec: EnsembleSpec, w) -> list[float]:
    """Window valleys of the semi-analytic free-exposure clock G (as in W1/W2)."""
    w = tuple(float(x) for x in w)
    try:
        diag = core.g_window_maxima_count(m=spec.centres().size, eps=spec.eps, weights=w,
                                          p=spec.model(), n_perp=spec.n_perp,
                                          target_times=spec.times(), tmax=spec.tmax)
        if spec.slab_shape != "gauss" or spec.slab_sd is not None:
            raise NotImplementedError
        return [float(v["time"]) for v in diag["valleys"]]
    except NotImplementedError:
        # contact-factor-free mixture valleys (tangent geometry, non-default slabs)
        t = np.arange(0.3, spec.tmax - 0.2, 1e-4)
        mu = spec.z_bar + (spec.z0 - spec.z_bar) * np.exp(-spec.gamma * t)
        s2 = spec.eps**2 * spec.d0 / (2 * spec.gamma) + spec.sd() ** 2
        H = sum(wj * np.exp(-(mu - c) ** 2 / (2 * s2)) for wj, c in zip(w, spec.centres()))
        d = np.diff(H)
        ii = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
        return [float(t[i]) for i in ii if WINDOW[0] <= t[i] <= WINDOW[1]]


def lambdas(B: float, w, spec: EnsembleSpec) -> np.ndarray:
    """Passage exposures lambda_j = B w_j / (W^(d-1) |mu'(t_j)|)."""
    w = np.asarray(w, float)
    return B * w / (spec.torus_w ** spec.n_perp * spec.mu_prime_abs(np.asarray(spec.times())))


def limit_masses(B: float, w, spec: EnsembleSpec, chi=None) -> tuple[np.ndarray, np.ndarray]:
    """Stick-breaking limit masses M_j = exp(-sum_{i<j} lam_i)(1 - exp(-lam_j)).

    With ``chi`` (N x m contact indicators at t_j) returns the frozen-gate
    average E[exp(-sum_{i<j} chi_i lam_i)(1 - exp(-chi_j lam_j))].
    """
    lam = lambdas(B, w, spec)
    if chi is None:
        cum = np.concatenate([[0.0], np.cumsum(lam)])
        return lam, np.exp(-cum[:-1]) * (1.0 - np.exp(-lam))
    ex = np.asarray(chi, float) * lam[None, :]
    cum = np.concatenate([np.zeros((ex.shape[0], 1)), np.cumsum(ex, axis=1)], axis=1)
    return lam, np.mean(np.exp(-cum[:, :-1]) * (1.0 - np.exp(-ex)), axis=0)


def basin_masses(ens: Ensemble, B: float, w=None, *, variant: str = "full", cuts=None,
                 groups: int = DEFAULT_GROUPS) -> dict:
    """Unconditional basin masses between cut times (snapped to stored edges).

    Default cuts: [0.5] + window valleys of G + [3.5] (window basins, as in the
    referee basin-mass check), each snapped to the nearest stored edge.
    Also returns the mean-field masses exp(-B Lambda(a)) - exp(-B Lambda(b))
    with Lambda = E X_w from the same ensemble, and the stick-breaking limit.
    """
    w = _weights(ens, w)
    if cuts is None:
        cuts = [WINDOW[0]] + g_valley_times(ens.spec, w) + [WINDOW[1]]
    ci = []
    for c in cuts:
        k = int(np.argmin(np.abs(ens.edges - c)))
        if abs(ens.edges[k] - c) > 0.5 * WINDOW_BIN + 1e-9:
            raise ValueError(f"cut {c} not representable on the stored grid")
        ci.append(k)
    # production window edges for the window endpoints
    wi = ens.window_index
    if abs(cuts[0] - WINDOW[0]) < 1e-12:
        ci[0] = int(wi[0])
    if abs(cuts[-1] - WINDOW[1]) < 1e-12:
        ci[-1] = int(wi[-1])
    N = ens.n_paths
    gb = _group_bounds(N, groups)
    sizes = np.diff(gb).astype(float)
    nb = len(ci) - 1
    s1 = np.zeros(nb)
    s2 = np.zeros(nb)
    sg = np.zeros((groups, nb))
    lam_sum = np.zeros(len(ci))
    ci_arr = np.asarray(ci)
    if np.any(np.diff(ci_arr) <= 0):
        raise ValueError("cuts must be strictly increasing on the stored grid")
    for start, dX, _ in ens.iter_chunks(variant):
        Xs, seg = _selected_exposure(dX, w, ci_arr)
        y = np.exp(-B * Xs[:, :-1]) * (-np.expm1(-B * seg))
        s1 += y.sum(0)
        s2 += (y * y).sum(0)
        sg += _group_sums(y, _group_ids(gb, start, y.shape[0]), groups)
        lam_sum += Xs.sum(0)
    M = s1 / N
    se = np.sqrt(np.maximum(s2 / N - M * M, 0.0) / N)
    se_b = _batch_se(sg / sizes[:, None], sizes)
    Lam = B * lam_sum / N
    mf = np.exp(-Lam[:-1]) - np.exp(-Lam[1:])
    lam, lim = limit_masses(B, w, ens.spec)
    return {"B": float(B), "w": w.tolist(), "variant": variant,
            "cuts_requested": [float(c) for c in cuts],
            "cuts_edges": [float(ens.edges[k]) for k in ci],
            "cuts_steps": [int(ens.cps[k]) for k in ci], "masses": M, "se_iid": se,
            "se_batch": se_b, "mean_field": mf, "B_Lambda_at_cuts": Lam,
            "lambda_j": lam, "limit_stick_breaking": lim, "N": N}


def exposure_moments(ens: Ensemble, w=None, *, variant: str = "full",
                     groups: int = DEFAULT_GROUPS) -> dict:
    """E X_t, Var X_t (unit budget) at stored checkpoints, sampled G, Cov(V, X).

    V on the stored grid is the bin-averaged rate (X(e_{k+1}) - X(e_k)) / dt_k,
    and Cov(V, X) uses X at the bin end.  For step-resolution moments use a
    declared ensemble (Ensemble.declared()).
    """
    w = _weights(ens, w)
    N = ens.n_paths
    gb = _group_bounds(N, groups)
    sizes = np.diff(gb).astype(float)
    K = ens.cps.size
    sx = np.zeros(K); sxx = np.zeros(K); sv = np.zeros(K - 1); svx = np.zeros(K - 1)
    svv = np.zeros(K - 1)
    gx = np.zeros((groups, K)); gxx = np.zeros((groups, K))
    tb = np.diff(ens.cps) * ens.spec.dt
    for start, dX, _ in ens.iter_chunks(variant):
        dXw = np.einsum("imk,m->ik", dX.astype(np.float64), w)
        Xw = np.cumsum(dXw, axis=1)
        V = dXw[:, 1:] / tb[None, :]
        sx += Xw.sum(0); sxx += (Xw * Xw).sum(0)
        sv += V.sum(0); svx += (V * Xw[:, 1:]).sum(0); svv += (V * V).sum(0)
        gid = _group_ids(gb, start, Xw.shape[0])
        gx += _group_sums(Xw, gid, groups)
        gxx += _group_sums(Xw * Xw, gid, groups)
    mx = sx / N
    vx = (sxx / N - mx * mx) * N / (N - 1)
    gm = gx / sizes[:, None]
    gv = gxx / sizes[:, None] - gm * gm
    return {"w": w.tolist(), "variant": variant, "t_edges": ens.edges,
            "cps": ens.cps, "mean_X": mx, "var_X": vx,
            "var_X_se_batch": _batch_se(gv, sizes), "mean_X_se_batch": _batch_se(gm, sizes),
            "G_sampled_bins": sv / N,
            "G_sampled_bins_se": np.sqrt(np.maximum(svv / N - (sv / N) ** 2, 0.0) / N),
            "bin_step_ranges": [[int(a) + 1, int(b)] for a, b in zip(ens.cps[:-1], ens.cps[1:])],
            "cov_V_X_bins": svx / N - (sv / N) * mx[1:], "N": N}


# ============================================================================
# Derivative / critical-point census (either mode)
# ============================================================================


def _gauss_smooth(Y: np.ndarray, h_pts: float) -> np.ndarray:
    """Edge-normalised Gaussian smoothing along the last axis (h in grid points)."""
    Y = np.asarray(Y, float)
    if h_pts <= 0:
        return Y.copy()
    r = int(math.ceil(5 * h_pts))
    k = np.arange(-r, r + 1)
    ker = np.exp(-0.5 * (k / h_pts) ** 2)
    ker /= ker.sum()
    n = Y.shape[-1]
    norm = np.convolve(np.ones(n), ker, mode="same")
    flat = Y.reshape(-1, n)
    out = np.array([np.convolve(row, ker, mode="same") for row in flat]) / norm
    return out.reshape(Y.shape)


def derivative_signs(t, density, batch_density, *, bandwidth: float = 0.0,
                     zthr: float = 5.0, window=WINDOW) -> dict:
    """Significant sign changes of d/dt of the (optionally smoothed) density.

    ``batch_density`` (nb, n) are independent equal-size batch estimates whose
    mean is ``density``; SE of the derivative is the batch-means SE of the same
    linear operation.  A grid point is +1/-1 when |f'|/SE > zthr, else 0; maxima
    are +1 -> -1 transitions of the nonzero sequence, minima -1 -> +1.
    """
    t = np.asarray(t, float)
    f = np.asarray(density, float)
    fb = np.asarray(batch_density, float)
    msk = (t >= window[0]) & (t <= window[1])
    tt, ff, bb = t[msk], f[msk], fb[:, msk]
    h = float(np.median(np.diff(tt)))
    hp = bandwidth / h if bandwidth > 0 else 0.0
    fs = _gauss_smooth(ff, hp)
    bs = _gauss_smooth(bb, hp)
    d = np.gradient(fs, tt)
    db = np.gradient(bs, tt, axis=1)
    nb = bb.shape[0]
    se = db.std(axis=0, ddof=1) / math.sqrt(nb)
    z = np.where(se > 0, d / np.where(se > 0, se, 1.0), 0.0)
    sgn = np.where(z > zthr, 1, np.where(z < -zthr, -1, 0))
    nz = np.flatnonzero(sgn)
    maxima, minima = [], []
    for a, b in zip(nz[:-1], nz[1:]):
        if sgn[a] == 1 and sgn[b] == -1:
            maxima.append(float(0.5 * (tt[a] + tt[b])))
        elif sgn[a] == -1 and sgn[b] == 1:
            minima.append(float(0.5 * (tt[a] + tt[b])))
    return {"n_maxima": len(maxima), "n_minima": len(minima), "maxima_t": maxima,
            "minima_t": minima, "zthr": zthr, "bandwidth": bandwidth,
            "n_undecided_points": int((sgn == 0).sum()), "t": tt, "dfdt": d, "se": se}


# ============================================================================
# Classifier on the expected law (covariance-aware; float counts)
# ============================================================================

_A_CACHE: dict = {}


def _smoothing_operator(n: int, bin_width: float, bandwidth: float):
    key = (n, round(bin_width, 14), bandwidth)
    if key in _A_CACHE:
        return _A_CACHE[key]
    radius = max(1, int(math.ceil(4.0 * bandwidth / bin_width)))
    offsets = np.arange(-radius, radius + 1) * bin_width
    kernel = np.exp(-(offsets**2) / (2.0 * bandwidth**2))
    kernel /= kernel.sum()
    norm = np.convolve(np.ones(n), kernel, mode="same")
    A = np.zeros((n, n))
    for i in range(n):
        for off, kv in zip(range(-radius, radius + 1), kernel):
            j = i - off
            if 0 <= j < n:
                A[i, j] = kv / norm[i]
    _A_CACHE[key] = (kernel, norm, A)
    return _A_CACHE[key]


def classify_expected(bin_mass, edges, *, walkers: float = 1_000_000,
                      bandwidth: float = base.DEFAULT_BANDWIDTH, sigma_factor: float = 5.0,
                      relative_floor: float = 0.05, cov=None) -> dict:
    """Covariance-aware 5 sigma + 5% classifier applied to an EXPECTED histogram.

    Mirrors reclassify_covariance_aware.classify_both (same smoothing, maxima
    scan, contour base and prominence sigma) with expected counts
    walkers * bin_mass in place of integer counts, i.e. the verdict the protocol
    would return at ``walkers`` walkers if the histogram equalled its
    expectation.  With ``cov`` (bin-mass covariance of an estimate) the sigma is
    instead the actual sampling SE of the prominence statistic.
    """
    p = np.asarray(bin_mass, float)
    edges = np.asarray(edges, float)
    n = p.size
    bw = round(float((edges[-1] - edges[0]) / n), 14)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel, norm, A = _smoothing_operator(n, bw, bandwidth)
    density = p / bw
    smoothed = np.convolve(density, kernel, mode="same") / norm
    counts = p * walkers
    gmax = float(smoothed.max()) if n else 0.0
    rows = []
    for i in range(1, n - 1):
        if not (smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]):
            continue
        height = float(smoothed[i])
        sides = []
        for direction in (-1, 1):
            probe = i + direction
            lowest, low_bin = height, i
            while 0 <= probe < n and smoothed[probe] <= height:
                if float(smoothed[probe]) < lowest:
                    lowest, low_bin = float(smoothed[probe]), probe
                probe += direction
            sides.append((lowest, low_bin))
        base_value, base_bin = max(sides, key=lambda pr: pr[0])
        prom = height - base_value
        coeff = (A[i] - A[base_bin]) / bw
        if cov is None:
            s = math.sqrt(max(float(np.sum(counts * (coeff / walkers) ** 2)), 0.0))
        else:
            s = math.sqrt(max(float(np.einsum("i,ij,j->", coeff, np.asarray(cov), coeff)), 0.0))
        z = prom / s if s > 0 else math.inf
        rel = prom / gmax if gmax > 0 else 0.0
        rows.append({"time": float(centres[i]), "bin": i, "base_bin": int(base_bin),
                     "prominence": float(prom), "relative_prominence": float(rel),
                     "sigma": s, "z": float(z),
                     "significant": bool(z >= sigma_factor and prom >= relative_floor * gmax)})
    return {"rows": rows, "mode_count": sum(r["significant"] for r in rows),
            "smoothed": smoothed, "global_max": gmax, "walkers": walkers,
            "bandwidth": bandwidth}


def last_mode_prominence(cls: dict, after_time: float, sigma_factor: float = 5.0) -> float:
    """Largest relative prominence among maxima after ``after_time`` passing 5 sigma."""
    vals = [r["relative_prominence"] for r in cls["rows"]
            if r["time"] > after_time and r["z"] >= sigma_factor]
    return max(vals) if vals else 0.0


def prominence_curve(ens: Ensemble, w, budgets, *, after_time: float, variant: str = "full",
                     walkers: float = 1_000_000, groups: int = DEFAULT_GROUPS,
                     bandwidth: float = base.DEFAULT_BANDWIDTH) -> dict:
    """Last-mode relative prominence r(B) of the exact law and its G jackknife replicates."""
    tab = survival_table(ens, budgets, w, variant=variant, index=ens.window_index,
                         groups=groups)
    edges = ens.edges[ens.window_index]
    N = tab["N"]
    sizes = tab["group_sizes"]
    Pg = tab["P_group_sums"]
    tot = Pg.sum(0)
    Bs = tab["budgets"]
    r_full = np.zeros(Bs.size)
    r_jack = np.zeros((groups, Bs.size))
    n_modes = np.zeros(Bs.size, int)
    for ib in range(Bs.size):
        cls = classify_expected(tot[ib] / N, edges, walkers=walkers, bandwidth=bandwidth)
        r_full[ib] = last_mode_prominence(cls, after_time)
        n_modes[ib] = cls["mode_count"]
        for g in range(groups):
            cj = classify_expected((tot[ib] - Pg[g, ib]) / (N - sizes[g]), edges,
                                   walkers=walkers, bandwidth=bandwidth)
            r_jack[g, ib] = last_mode_prominence(cj, after_time)
    return {"budgets": Bs, "r": r_full, "r_jack": r_jack, "mode_count": n_modes,
            "after_time": after_time, "walkers": walkers, "groups": groups}


def _crossing(Bs: np.ndarray, r: np.ndarray, p: float):
    above = r >= p
    for i in range(Bs.size - 1):
        if above[i] and not above[i + 1]:
            x0, x1 = math.log(Bs[i]), math.log(Bs[i + 1])
            y0, y1 = r[i] - p, r[i + 1] - p
            return float(math.exp(x0 + (x1 - x0) * y0 / (y0 - y1))), i
    return None, None


def prominence_floor_crossing(ens: Ensemble, w=None, *, p: float = 0.05, B_lo: float,
                              B_hi: float, after_time=None, variant: str = "full",
                              walkers: float = 1_000_000, groups: int = DEFAULT_GROUPS,
                              n_coarse: int = 13, n_fine: int = 17) -> dict:
    """B_p: smallest B where the last-mode relative prominence falls to p.

    Two passes (geometric coarse grid on [B_lo, B_hi], then a fine grid around
    the first down-crossing), log-linear interpolation, delete-one-group
    jackknife SE.  ``after_time`` defaults to the last G valley (W2 pass rule).
    """
    w = _weights(ens, w)
    if after_time is None:
        vals = g_valley_times(ens.spec, w)
        after_time = vals[-1]
    coarse = np.geomspace(B_lo, B_hi, n_coarse)
    pc = prominence_curve(ens, w, coarse, after_time=after_time, variant=variant,
                          walkers=walkers, groups=groups)
    Bx, i = _crossing(coarse, pc["r"], p)
    if Bx is None:
        return {"B_p": None, "status": "no_crossing_in_range", "coarse": _jsonable(pc),
                "p": p}
    lo = coarse[max(i - 1, 0)]
    hi = coarse[min(i + 2, coarse.size - 1)]
    fine = np.geomspace(lo, hi, n_fine)
    pf = prominence_curve(ens, w, fine, after_time=after_time, variant=variant,
                          walkers=walkers, groups=groups)
    Bf, _ = _crossing(fine, pf["r"], p)
    reps = []
    for g in range(groups):
        bj, _ = _crossing(fine, pf["r_jack"][g], p)
        if bj is None:
            bj, _ = _crossing(coarse, pc["r_jack"][g], p)
        reps.append(bj)
    reps_ok = np.array([x for x in reps if x is not None])
    G = reps_ok.size
    se = float(math.sqrt((G - 1) / G * np.sum((reps_ok - reps_ok.mean()) ** 2))) if G > 1 else None
    return {"B_p": Bf, "p": p, "jackknife_se": se,
            "ci95": [Bf - 1.96 * se, Bf + 1.96 * se] if se is not None else None,
            "jackknife_replicates": reps, "after_time": after_time,
            "walkers_nominal": walkers, "groups": groups, "status": "interpolated",
            "coarse": _jsonable(pc), "fine": _jsonable(pf)}


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


# ============================================================================
# Scout-cache reader (independent PCG64 ensembles of 2026-09-23 scouting)
# ============================================================================


def load_scout_cache(path) -> dict:
    """Per-step FK law from a theory-scout cache file, binned on production window edges.

    Scout files (fk_paths.py): FK_full/FK_free (nB, steps) per-step kill
    probabilities for the equal-weight production model, 4 path batches.
    """
    d = np.load(path)
    dt = float(d["dt"])
    steps = int(d["tgrid"].size)
    T = step_times(dt, steps)
    edges = production_window_edges()
    out = {"m": int(d["m"]), "eps": float(d["eps"]), "N": int(d["N"]), "dt": dt,
           "budgets": d["budgets"].tolist(), "edges": edges, "seed": int(d["seed"])}
    for key in ("FK_full", "FK_free"):
        if key not in d.files:
            continue
        FK = d[key]
        FKb = d[key.replace("FK_", "FKb_")]
        out[key] = np.array([np.histogram(T, bins=edges, weights=FK[i])[0]
                             for i in range(FK.shape[0])])
        out[key + "_batches"] = np.array([[np.histogram(T, bins=edges, weights=FKb[b, i])[0]
                                           for i in range(FK.shape[0])]
                                          for b in range(FKb.shape[0])])
    return out


# ============================================================================
# Self tests
# ============================================================================


def selftest(out_json: Path | None = None) -> dict:
    """Stored vs declared on common paths; B=0; small direct-kill comparison."""
    res = {}
    spec = EnsembleSpec(m=2, eps=0.1)
    n = 10_000
    st = simulate_ensemble(spec, n, name="selftest_stored", tag=TAGS["spare89"],
                           variants=("full", "nogate", "meancontact", "p1"), workers=1,
                           chunk=5_000, overwrite=True, wait=False, verbose=False)
    dc = simulate_ensemble(spec, n, name="selftest_declared", tag=TAGS["spare89"],
                           mode="declared", workers=1, chunk=5_000, batch=1_000,
                           declared=[dict(B=1.0, w=(0.5, 0.5)), dict(B=8.0, w=(0.7, 0.3)),
                                     dict(B=1.0, w=(0.5, 0.5), variant="nogate"),
                                     dict(B=1.0, w=(0.5, 0.5), cap=1e9)],
                           overwrite=True, wait=False, verbose=False)
    D = dc.declared()
    T = D["t"]
    edges = st.edges
    worst = 0.0
    worst_tail = 0.0
    for idd, (B, w, var) in enumerate([(1.0, (0.5, 0.5), "full"), (8.0, (0.7, 0.3), "full"),
                                       (1.0, (0.5, 0.5), "nogate"), (1.0, (0.5, 0.5), "full")]):
        law = exact_law(st, B, w, variant=var, region="all")
        binned = np.histogram(T, bins=edges, weights=D["p_step"][idd])[0]
        # per-bin sums of the declared per-step law with production semantics (cps)
        from_steps = np.add.reduceat(D["p_step"][idd], st.cps[:-1])
        err = np.abs(from_steps - law["bin_mass"]).max()
        rel = err / max(law["bin_mass"].max(), 1e-300)
        worst = max(worst, rel)
        nz = from_steps > 1e-300
        tail_rel = float(np.max(np.abs(from_steps[nz] - law["bin_mass"][nz]) / from_steps[nz]))
        res[f"stored_vs_declared_{var}_B{B:g}_w{w[0]:g}_max_rel_err"] = float(rel)
        res[f"stored_vs_declared_{var}_B{B:g}_w{w[0]:g}_max_binwise_rel_err"] = tail_rel
        res[f"stored_vs_declared_{var}_B{B:g}_w{w[0]:g}_smallest_bin_mass"] = float(
            from_steps[nz].min())
        worst_tail = max(worst_tail, tail_rel)
        res[f"hist_vs_cps_binning_diff_{var}_B{B:g}"] = float(np.abs(binned - from_steps).max())
    res["stored_vs_declared_worst_rel_err"] = float(worst)
    res["stored_vs_declared_worst_binwise_rel_err"] = float(worst_tail)
    # cap = 1e9 must equal uncapped
    res["cap_huge_equals_uncapped_max_abs"] = float(np.abs(D["p_step"][3] - D["p_step"][0]).max())
    # B = 0: no kills
    law0 = exact_law(st, 0.0, (0.5, 0.5), region="all")
    res["B0_total_kill_mass"] = float(law0["bin_mass"].sum())
    # moments: declared vs stored at checkpoints
    mom = exposure_moments(st, (0.5, 0.5))
    k = st.cps[1:] - 1
    res["moments_meanX_max_rel_err"] = float(np.max(np.abs(mom["mean_X"][1:] - D["mean_X"][0][k])
                                                    / np.maximum(mom["mean_X"][1:], 1e-12)))
    # direct-kill production simulator vs FK at small N (independent streams)
    dk = base.run_config(m=2, eps=0.1, budget=1.0, weights=(0.5, 0.5), walkers=100_000,
                         chunk=50_000, dt=spec.dt, tmax=spec.tmax, seed=BASE_SEED,
                         tag=TAGS["spare89"], verbose=False)
    counts, _ = np.histogram(dk["kill_times"], bins=production_window_edges())
    lawf = exact_law(st, 1.0, (0.5, 0.5))
    pdk = counts / 1e5
    z = (lawf["bin_mass"] - pdk) / np.sqrt(lawf["se_iid"] ** 2
                                           + lawf["bin_mass"] * (1 - lawf["bin_mass"]) / 1e5)
    res["small_dk_vs_fk_max_abs_z"] = float(np.abs(z).max())
    res["small_dk_vs_fk_chi2_over_dof"] = float(np.sum(z * z) / z.size)
    res["variant_smoke"] = variant_smoke()
    res["passed"] = bool(res["variant_smoke"]["all_ok"] and worst < 1e-5 and worst_tail < 1e-4 and res["B0_total_kill_mass"] == 0.0
                         and res["cap_huge_equals_uncapped_max_abs"] < 1e-15
                         and res["small_dk_vs_fk_max_abs_z"] < 5.0)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(res, indent=1))
    return res


def variant_smoke(n: int = 2_000) -> dict:
    """Tiny runs of every configuration/kernel option: finiteness, mass balance, B = 0."""
    cases = {
        "tophat": (EnsembleSpec(m=2, eps=0.1, slab_shape="tophat"), ("full",)),
        "fixed_slab_sd_0.3": (EnsembleSpec(m=2, eps=0.05, slab_sd=0.3), ("full",)),
        "point_release_varz0_0": (EnsembleSpec(m=3, eps=0.1, var_z0_scale=0.0), ("full",)),
        "varz0_4x": (EnsembleSpec(m=3, eps=0.1, var_z0_scale=4.0), ("full",)),
        "boundary_tangent": (EnsembleSpec(m=2, eps=0.05, r_par0=0.0, r_perp0=0.4),
                             ("full", "nogate", "a0.2")),
        "d3_meancontact": (EnsembleSpec(m=2, eps=0.1, n_perp=2), ("full", "meancontact")),
        "m5_z0_8": (EnsembleSpec(m=5, eps=0.1, z0=8.0, centres_z=(2.8, 2.2, 1.6, 1.0, 0.4)),
                    ("full", "p1", "p2", "gate=contact,a=0.15,field=p1")),
    }
    out = {}
    ok_all = True
    for label, (spec, variants) in cases.items():
        row = {}
        try:
            ens = simulate_ensemble(spec, n, name=f"selftest_smoke_{label}", tag=TAGS["spare89"],
                                    variants=variants, workers=1, chunk=1_000,
                                    overwrite=True, wait=False, verbose=False)
            for v in ens.variants:
                law = exact_law(ens, 2.0, None, variant=v, region="all", groups=4)
                zero = exact_law(ens, 0.0, None, variant=v, region="all", groups=4)
                tot = float(law["bin_mass"].sum())
                bal = abs(tot + law["survival_at_edges"][-1] - 1.0)
                ok = bool(np.all(np.isfinite(law["bin_mass"])) and np.all(law["bin_mass"] >= 0)
                          and 0 < tot < 1 and bal < 1e-9 and zero["bin_mass"].sum() == 0.0)
                row[v] = {"kill_mass_B2": tot, "mass_balance_err": bal, "ok": ok}
                ok_all &= ok
        except Exception as exc:  # pragma: no cover - reported, not raised
            row["error"] = repr(exc)
            ok_all = False
        out[label] = row
    # mean-contact must refuse the tangent geometry (quadrature assumes r_perp0 = 0)
    try:
        simulate_ensemble(EnsembleSpec(m=2, eps=0.05, r_perp0=0.4), 1_000,
                          name="selftest_smoke_refuse", tag=TAGS["spare89"],
                          variants=("meancontact",), workers=1, chunk=1_000,
                          overwrite=True, wait=False, verbose=False)
        out["meancontact_tangent_refused"] = False
        ok_all = False
    except NotImplementedError:
        out["meancontact_tangent_refused"] = True
    # declared mode with a cap, p1 field and the derivative census
    dec = simulate_ensemble(EnsembleSpec(m=2, eps=0.1), 4_000, name="selftest_smoke_declared",
                            tag=TAGS["spare89"], mode="declared", workers=1, chunk=2_000,
                            batch=500, overwrite=True, wait=False, verbose=False,
                            declared=[dict(B=4.0, w=(0.5, 0.5)),
                                      dict(B=4.0, w=(0.5, 0.5), cap=1.0),
                                      dict(B=4.0, w=(0.5, 0.5), variant="p1")])
    D = dec.declared()
    ds = derivative_signs(D["t"], D["density"][0], D["batch_p_step"][0] / D["dt"],
                          bandwidth=0.04, zthr=5.0)
    out["declared_cap_reduces_mass"] = bool(D["p_step"][1].sum() < D["p_step"][0].sum())
    out["declared_derivative_census_B4"] = {"n_maxima": ds["n_maxima"], "maxima_t": ds["maxima_t"],
                                            "n_minima": ds["n_minima"]}
    ok_all &= out["declared_cap_reduces_mass"]
    out["all_ok"] = bool(ok_all)
    return out


# ============================================================================
# CLI
# ============================================================================


def _parse_weights(text, m):
    if text is None:
        return None
    return tuple(float(x) for x in text.split(","))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="simulate/resume an ensemble")
    s.add_argument("--m", type=int, required=True)
    s.add_argument("--eps", type=float, required=True)
    s.add_argument("--paths", type=lambda v: int(float(v)), required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--tag", type=int, required=True)
    s.add_argument("--seed", type=int, default=BASE_SEED)
    s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--mode", choices=("stored", "declared"), default="stored")
    s.add_argument("--variants", default="full")
    s.add_argument("--declared", default=None,
                   help="JSON list of {B, w, variant, cap, label} (declared mode)")
    s.add_argument("--grid", default="production")
    s.add_argument("--workers", type=int, default=MAX_WORKERS)
    s.add_argument("--chunk", type=lambda v: int(float(v)), default=DEFAULT_CHUNK)
    s.add_argument("--batch", type=lambda v: int(float(v)), default=DEFAULT_BATCH)
    s.add_argument("--chunks", default=None, help="comma list of chunk indices to run")
    s.add_argument("--dt", type=float, default=base.DEFAULT_DT)
    s.add_argument("--tmax", type=float, default=base.DEFAULT_TMAX)
    s.add_argument("--z0", type=float, default=MODEL.z0)
    s.add_argument("--var-z0-scale", type=float, default=1.0)
    s.add_argument("--r-par0", type=float, default=MODEL.r_par0)
    s.add_argument("--r-perp0", type=float, default=MODEL.r_perp0)
    s.add_argument("--u0", type=float, default=MODEL.u0)
    s.add_argument("--sigma-perp0", type=float, default=MODEL.sigma_perp0)
    s.add_argument("--n-perp", type=int, default=1)
    s.add_argument("--contact-a", type=float, default=MODEL.contact_a)
    s.add_argument("--rho", type=float, default=MODEL.rho)
    s.add_argument("--slab-shape", choices=("gauss", "tophat"), default="gauss")
    s.add_argument("--slab-sd", type=float, default=None)
    s.add_argument("--target-times", default=None)
    s.add_argument("--centres-z", default=None)
    s.add_argument("--kernel-cutoff-sd", type=float, default=GAUSS_CUTOFF_SD)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--note", default="")
    sub.add_parser("selftest", help="stored-vs-declared, B=0, small direct-kill check")
    v = sub.add_parser("validate-n0", help="N0 acceptance vs production histograms")
    v.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    i = sub.add_parser("info")
    i.add_argument("--name", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        tt = tuple(float(x) for x in args.target_times.split(",")) if args.target_times else None
        cz = tuple(float(x) for x in args.centres_z.split(",")) if args.centres_z else None
        spec = EnsembleSpec(m=args.m, eps=args.eps, dt=args.dt, tmax=args.tmax, z0=args.z0,
                            var_z0_scale=args.var_z0_scale, r_par0=args.r_par0,
                            r_perp0=args.r_perp0, u0=args.u0, sigma_perp0=args.sigma_perp0,
                            n_perp=args.n_perp, contact_a=args.contact_a, rho=args.rho,
                            slab_shape=args.slab_shape, slab_sd=args.slab_sd,
                            target_times=tt, centres_z=cz,
                            kernel_cutoff_sd=args.kernel_cutoff_sd)
        decl = json.loads(args.declared) if args.declared else None
        chunks = [int(x) for x in args.chunks.split(",")] if args.chunks else None
        ens = simulate_ensemble(spec, args.paths, name=args.name, tag=args.tag, seed=args.seed,
                                replicate=args.replicate, mode=args.mode,
                                variants=tuple(args.variants.split(";")), declared=decl,
                                grid=args.grid, workers=args.workers, chunk=args.chunk,
                                batch=args.batch, chunks_subset=chunks,
                                overwrite=args.overwrite, wait=not args.no_wait,
                                note=args.note)
        print(json.dumps({"name": ens.name, "complete": ens.index["complete"],
                          "chunks": len(ens.chunks), "paths_done": ens.n_paths,
                          "index": str(index_path(ens.name))}))
    elif args.cmd == "selftest":
        res = selftest(FB_DATA / "n0_selftest.json")
        print(json.dumps(res, indent=1))
    elif args.cmd == "validate-n0":
        import exact_m_prr_fk_n0_validation as val  # noqa: WPS433
        val.run(groups=args.groups)
    elif args.cmd == "info":
        ens = load_ensemble(args.name)
        print(ens)
        print(json.dumps({k: ens.index[k] for k in ("seed", "tag", "replicate", "seed_entropy",
                                                    "complete", "process_seconds_total")},
                         indent=1))


if __name__ == "__main__":
    main()
