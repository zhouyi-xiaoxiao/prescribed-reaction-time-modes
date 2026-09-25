#!/usr/bin/env python3
"""Differentiable exact-law design estimator (uplift2 item 1, NU-A).

Purpose
-------
Evaluate, for ANY stripe design theta = (c_1..c_m, w_1..w_m, B) on ONE fixed ensemble of
unkilled paths, the exact discrete-time reaction-time law of the production model
(Euler--Maruyama pair + end-of-step Doi kill, ``validate_exact_m_offlattice``) together with
its exact pathwise gradients, and from them the peak times, valley cuts, basin masses and
conditional mass ratios with their exact (sample-level) Jacobians.  This is the engine of the
sample-average-approximation (SAA) Newton design of NU-B.

Why this is exact
-----------------
The unkilled pair (Z, R) does not depend on the stripes.  For a path with unit-budget step
exposures e_n(theta) = g_n sum_j w_j phi(Z_n - c_j) dt / W^(d-1) (g_n = contact gate, phi the
Gaussian stripe of sd s = eps*rho) and X_n = e_1 + ... + e_n, the discrete Feynman--Kac identity
gives (see exact_m_prr_fk_exact_law)

    p_n(theta) = P(T_R = t_n) = E[ exp(-B X_{n-1}) (1 - exp(-B e_n)) ],   t_n = n dt.

The integrand is a smooth function of theta for every path, so the pathwise derivative is
unbiased for d p_n / d theta and, on a FIXED sample, is the exact derivative of the SAA
estimator (checked against central finite differences to ~1e-7 relative, selftest):

    d y_n = B A_n [ (1 - Q_n) d e_n - Q_n d X_{n-1} ],   A_n = e^{-B X_{n-1}}, Q_n = 1 - e^{-B e_n},
    d e_n / d w_j = phi_{jn},   d e_n / d c_j = w_j phi_{jn} (Z_n - c_j) / s^2,
    d y_n / d B   = A_n [ e_n (1 - Q_n) ] - y_n X_{n-1}.

Storage ("kernel evaluation on stored paths")
--------------------------------------------
Per path and step we keep one float32: zeta_n = Z_n - m_n (m_n = exact EM mean of Z_n) when the
contact gate is on, and a sentinel 1e6 when it is off (the Gaussian stripe then underflows to
exactly 0).  4 bytes per path-step: 1e5 paths x 4000 steps = 1.6 GB in RAM, split over the
workers.  float32 zeta has absolute precision ~1e-9 (|zeta| < 1), i.e. a stripe-argument error
< 1e-6 relative at eps >= 0.01 -- far below Monte Carlo error.  Paths are generated with the
SAME draw order and SeedSequence layout as exact_m_prr_fk_exact_law._run_chunk, so an engine
with (tag, replicate, chunk) of an existing FK ensemble reproduces its paths (crosscheck).
A dense grid of candidate centres was rejected: interpolation error, and no analytic d/dc.
Stripe-by-stripe step windows (|m_n - c_j| <= 12 s + max|zeta| of the block) skip blocks where
a stripe is exactly negligible (< exp(-72) relative), as in the FK production kernel.

Peak times, valleys and masses (smooth functionals, SAA-differentiable)
---------------------------------------------------------------------
f_h(t) = sum_n p_n K_h(t - t_n) (Gaussian kernel, bandwidth h in TIME units; the same
functional is applied to direct-kill histograms in fb_v2_tep, at every dt).  Peak time t*_j =
the maximiser of f_h in basin j, found on the step grid and refined by Newton on f_h'(t) = 0;
implicit differentiation gives dt*/dtheta = - d_theta f_h'(t*) / f_h''(t*).  Valley cuts s_j
likewise (minimisers of f_h between consecutive peaks).  Basin masses use the smoothed CDF
F_h(s) = sum_n p_n Phi((s - t_n)/h) at the valley cuts (its bias vanishes to O(h^4) at a
valley because f'(s) = 0) and the exact end points F(0) = 0, F(tmax) = sum_n p_n:
    M_j = F_h(s_j) - F_h(s_{j-1}),  dM_j = dF_h(s_j) + f_h(s_j) ds_j - (same at s_{j-1}).
Ratios r_j = M_j / sum_i M_i (basins inside [0, tmax]).  The initial basins used to LOCATE
the peaks are the valleys of the design's exact EM free clock G (deterministic, as in N11) or
user cuts; the reported masses use the f_h-valley cuts (mode "valley") or fixed cuts.
Smoothing bias of t* is O(h^2 f'''/f''); ``smoothing_bias`` measures it on the noiseless
mean-field curve of the same design, and the TEP applies the same h to direct kill.

Parameters and Jacobian conventions
-----------------------------------
Raw parameter order: [c_1..c_m, w_1..w_m, B].  ``to_design_coords`` converts a raw gradient
to (a) stripe times t_j (c_j = mu(t_j), dc/dt = -gamma (z0 - z_bar) e^{-gamma t}), (b) the
paper's shifts delta_j with t_j = T_j - eps delta_j, (c) simplex tangent coordinates
u_k (w = w0 + sum_k u_k (e_k - e_m), d/du_k = d/dw_k - d/dw_m) and (d) the projected simplex
gradient g - mean(g).

Statistical errors
------------------
Paths are split into groups of ``group`` paths (default 5000) in chunk order; every functional
is recomputed on delete-one-group samples (window/basin selection held fixed) and the
jackknife SE is reported, for values and for gradients.

Python API (sys.path must contain code/)
----------------------------------------
    import fb_v2_design_estimator as de
    spec = de.path_spec(eps=0.05)                        # production path law (fk.EnsembleSpec)
    eng  = de.Engine(spec, 100_000, tag=de.TAGS["NU_A"], workers=3)   # generates paths once
    th   = de.Design.from_times(times=(0.8, 1.6, 2.8), w=(0.2, 0.3, 0.5), B=8.0, spec=spec)
    law  = eng.evaluate(th)                                # Law: p_n, dp_n/dtheta, group sums
    res  = de.analyse(law, h=0.02)                         # peaks, valleys, masses, ratios + Jacobians + SEs
    J    = de.to_design_coords(res["jac"]["peaks"], th, spec, coords="times_tangent")
    eng.close()
    # also: Engine(..., workers=0) (in-process), eng.evaluate([th1, th2, ...]) (batched),
    #       de.mean_field_law(th, spec) (noiseless EM mean-field law in the same Law format),
    #       de.smoothing_bias(th, spec, h), de.design_outputs(res, "F1") (peak times + m-1 ratios),
    #       de.design_map(law, "F1", targets=...) (y, J in (t_j, u_k), residual), de.count_modes(law, h).

CLI
---
    python3 fb_v2_design_estimator.py selftest            # FD gradients, invariances (small, local)
    python3 fb_v2_design_estimator.py crosscheck-fk       # bitwise path reproduction vs FK ensemble n0_m2_eps0.1
    python3 fb_v2_design_estimator.py validate-dk         # values vs existing N11 direct-kill data
    python3 fb_v2_tep.py dkfd --design D.json --param t3   # gradients vs CRN direct-kill finite differences
Outputs: artifacts/data/exact_m_fixed_budget/V2_NU_A/*.json.

Machine etiquette: <= 3 local workers; waits while >= 9 python processes use > 20 % CPU
(idle MCP/helper interpreters are not counted).  Multiprocessing uses the spawn context, so
driver scripts MUST guard their entry point with ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402

REPORT = HERE.parents[1]
FB_DATA = REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
OUT_DIR = FB_DATA / "V2_NU_A"
FIGURES = REPORT / "artifacts" / "figures"
PATH_CACHE_ROOT = Path(os.environ.get("PRR_FB_V2_PATH_ROOT",
                                      str(Path.home() / ".local-build" / "prr_fb_v2_paths")))

BASE_SEED = fk.BASE_SEED                      # 20260923
# uplift2 seed tags (all previous items use tags <= 98 or 999): disjoint streams.
TAGS = {"NU_A": 201, "NU_A_FD": 202, "NU_B": 203, "NU_C": 204, "NU_D": 205, "HERO": 206,
        "TEP_DK": 211, "NU_E_DK": 212, "HERO_DK": 213, "DKFD": 214}
SENTINEL = np.float32(1.0e6)                  # zeta for gate-off steps: stripe underflows to 0
CUTOFF_SD = 12.0                              # stripe negligible beyond 12 s (exp(-72))
DEFAULT_CHUNK = 25_000
DEFAULT_GROUP = 5_000
STEP_BLOCK = 200                              # steps per evaluation block
PATH_BLOCK = 5_000                            # paths per evaluation block (= group granularity)
MAX_LOCAL_WORKERS = int(os.environ.get("PRR_FB_V2_MAX_WORKERS", "3"))   # set 144 on Isambard
BUSY_LIMIT = 9

MODEL = fk.MODEL


# ============================================================================
# Path law and designs
# ============================================================================


def path_spec(eps: float, *, dt: float = fk.base.DEFAULT_DT, tmax: float = fk.base.DEFAULT_TMAX,
              **kw) -> fk.EnsembleSpec:
    """Production path law (fk.EnsembleSpec; m/centres are irrelevant for the paths)."""
    return fk.EnsembleSpec(m=2, eps=float(eps), dt=float(dt), tmax=float(tmax), **kw)


def mu(t, spec: fk.EnsembleSpec):
    """Continuum mean passage position mu(t) = z_bar + (z0 - z_bar) e^{-gamma t}."""
    return spec.z_bar + (spec.z0 - spec.z_bar) * np.exp(-spec.gamma * np.asarray(t, float))


def t_of_c(c, spec: fk.EnsembleSpec):
    return -np.log((np.asarray(c, float) - spec.z_bar) / (spec.z0 - spec.z_bar)) / spec.gamma


def dc_dt(t, spec: fk.EnsembleSpec):
    return -spec.gamma * (spec.z0 - spec.z_bar) * np.exp(-spec.gamma * np.asarray(t, float))


def speed(t, spec: fk.EnsembleSpec):
    return np.abs(dc_dt(t, spec))


@dataclass
class Design:
    """Stripe design: centres c_j (Z coordinates), allocation w (simplex), budget B."""

    c: np.ndarray
    w: np.ndarray
    B: float
    label: str = ""

    def __post_init__(self):
        self.c = np.asarray(self.c, float).ravel()
        self.w = np.asarray(self.w, float).ravel()
        self.B = float(self.B)
        if self.c.size != self.w.size:
            raise ValueError("c and w must have the same length")
        if np.any(self.w < 0):
            raise ValueError("weights must be non-negative")

    @property
    def m(self) -> int:
        return int(self.c.size)

    @property
    def K(self) -> int:
        return 2 * self.m + 1

    @staticmethod
    def from_times(times, w, B, spec, label: str = "") -> "Design":
        return Design(mu(np.asarray(times, float), spec), w, B, label)

    def times(self, spec) -> np.ndarray:
        return t_of_c(self.c, spec)

    def to_dict(self, spec=None) -> dict:
        d = {"c": self.c.tolist(), "w": self.w.tolist(), "B": self.B, "label": self.label}
        if spec is not None:
            d["stripe_times"] = self.times(spec).tolist()
        return d

    @staticmethod
    def from_dict(d: dict, spec=None) -> "Design":
        if "c" in d:
            return Design(d["c"], d["w"], d["B"], d.get("label", ""))
        if "centres_z" in d:
            return Design(d["centres_z"], d["w"], d["B"], d.get("label", ""))
        return Design.from_times(d["stripe_times"], d["w"], d["B"], spec, d.get("label", ""))

    def param_names(self) -> list[str]:
        m = self.m
        return [f"c{j + 1}" for j in range(m)] + [f"w{j + 1}" for j in range(m)] + ["B"]


def em_mean_z(spec: fk.EnsembleSpec) -> np.ndarray:
    """Exact EM mean m_n of Z_n, n = 1..steps (float64)."""
    return fk.em_moments(spec)["mean_z"]


# ============================================================================
# Path generation (mirrors exact_m_prr_fk_exact_law._run_chunk draw order exactly)
# ============================================================================


def generate_chunk(spec: fk.EnsembleSpec, n: int, seedseq, *, gate: str = "contact") -> dict:
    """Unkilled pair paths -> float32 zeta[n, steps] (sentinel where the gate is off).

    Draw order: z0 (n), r_par0 (n), r_perp0 (n_perp, n), then per step one
    standard_normal((2 + n_perp, n)) block -- identical to fk._run_chunk and
    core.simulate_chunk_general, so equal seeds give equal paths.
    """
    steps = spec.steps()
    dt = spec.dt
    rng = np.random.Generator(np.random.Philox(seedseq))
    decay = 1.0 - spec.gamma * dt
    pull = spec.gamma * spec.z_bar * dt
    noise_z = spec.eps * math.sqrt(spec.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    W = spec.torus_w
    a2 = spec.contact_a ** 2
    z = spec.z0 + math.sqrt(spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)) \
        * rng.standard_normal(n)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    r_perp = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0
                    * rng.standard_normal((spec.n_perp, n)), W)
    mz = em_mean_z(spec)
    zeta = np.empty((n, steps), np.float32)
    contact = 0
    zmax = 0.0
    for s in range(steps):
        noise = rng.standard_normal((2 + spec.n_perp, n))
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        r_perp += noise_r * noise[2:]
        np.mod(r_perp, W, out=r_perp)
        dev = z - mz[s]
        zmax = max(zmax, float(np.max(np.abs(dev))))
        if gate == "contact":
            perp_mi = np.minimum(r_perp, W - r_perp)
            on = (r_par * r_par + np.sum(perp_mi * perp_mi, axis=0)) < a2
            contact += int(np.count_nonzero(on))
            zeta[:, s] = np.where(on, dev, SENTINEL)
        elif gate == "none":
            contact += n
            zeta[:, s] = dev
        else:
            raise ValueError(f"unknown gate {gate!r}")
    return {"zeta": zeta, "contact_steps": contact, "max_abs_zeta": zmax, "n": n}


def block_max_abs_zeta(zeta: np.ndarray, L: int = STEP_BLOCK) -> np.ndarray:
    """max |zeta| over gated entries of each step block (sentinel entries ignored)."""
    S = zeta.shape[1]
    nb = -(-S // L)
    out = np.zeros(nb)
    for b in range(nb):
        blk = zeta[:, b * L:(b + 1) * L]
        a = np.abs(blk)
        a[blk == SENTINEL] = 0.0
        out[b] = float(a.max()) if a.size else 0.0
    return out


# ============================================================================
# Exact law + pathwise gradients on one chunk
# ============================================================================


def stripe_activity(spec: fk.EnsembleSpec, design: Design, blockmax: np.ndarray,
                    mz: np.ndarray, L: int = STEP_BLOCK) -> np.ndarray:
    """active[j, b]: stripe j can exceed exp(-CUTOFF^2/2) of its peak in step block b."""
    s = spec.sd()
    nb = blockmax.size
    act = np.zeros((design.m, nb), bool)
    for b in range(nb):
        seg = mz[b * L:(b + 1) * L]
        for j in range(design.m):
            dmin = float(np.min(np.abs(seg - design.c[j])))
            act[j, b] = dmin <= CUTOFF_SD * s + blockmax[b] + 1e-12
    return act


def evaluate_chunk(zeta: np.ndarray, blockmax: np.ndarray, spec: fk.EnsembleSpec,
                   design: Design, *, want_grad: bool = True, group: int = DEFAULT_GROUP,
                   mz: np.ndarray | None = None) -> dict:
    """Group sums of y_n and d y_n / d theta over the paths of one chunk.

    Returns {"Y": (G, S), "dY": (G, K, S) or None, "sizes": (G,)}, where
    y_n = e^{-B X_{n-1}} (1 - e^{-B e_n}) per path and theta = (c_1..c_m, w_1..w_m, B).
    """
    n, S = zeta.shape
    mz = em_mean_z(spec) if mz is None else mz
    m, K = design.m, design.K
    s = spec.sd()
    inv2 = 1.0 / (2.0 * s * s)
    inv_s2 = 1.0 / (s * s)
    pref = spec.dt / (math.sqrt(2.0 * math.pi) * s * spec.torus_w ** spec.n_perp)
    B, w, c = design.B, design.w, design.c
    L = STEP_BLOCK
    nb = -(-S // L)
    active = stripe_activity(spec, design, blockmax, mz, L)
    G = -(-n // group)
    Y = np.zeros((G, S))
    dY = np.zeros((G, K, S)) if want_grad else None
    sizes = np.zeros(G, np.int64)
    for g in range(G):
        r0, r1 = g * group, min(n, (g + 1) * group)
        nr = r1 - r0
        sizes[g] = nr
        carryX = np.zeros(nr)
        carry_d = np.zeros((2 * m, nr)) if want_grad else None
        for b in range(nb):
            act = np.flatnonzero(active[:, b])
            if act.size == 0:
                continue                      # e = 0 on the block: y = 0, dy = 0, carries fixed
            c0, c1 = b * L, min(S, (b + 1) * L)
            Zb = zeta[r0:r1, c0:c1].astype(np.float64)
            Zb += mz[None, c0:c1]
            e = np.zeros_like(Zb)
            phis = {}
            for j in act:
                d = Zb - c[j]
                ph = np.exp(-(d * d) * inv2)
                ph *= pref
                phis[j] = (ph, d)
                e += w[j] * ph
            cs = np.cumsum(e, axis=1)
            Xprev = cs - e
            Xprev += carryX[:, None]
            A = np.exp(-B * Xprev)
            E1 = np.exp(-B * e)               # 1 - Q
            Q = -np.expm1(-B * e)
            y = A * Q
            Y[g, c0:c1] += y.sum(axis=0)
            if want_grad:
                BAE = B * A * E1
                By = B * y
                for k in range(2 * m):
                    j = k % m
                    if j in phis:
                        ph, d = phis[j]
                        de = ph if k >= m else (w[j] * inv_s2) * ph * d
                        dcs = np.cumsum(de, axis=1)
                        dX = dcs - de
                        dX += carry_d[k][:, None]
                        dY[g, k, c0:c1] += (BAE * de - By * dX).sum(axis=0)
                        carry_d[k] += dcs[:, -1]
                    elif np.any(carry_d[k] != 0.0):
                        dY[g, k, c0:c1] -= B * np.einsum("i,ij->j", carry_d[k], y)
                dY[g, 2 * m, c0:c1] += (A * E1 * e - y * Xprev).sum(axis=0)
            carryX += cs[:, -1]
    return {"Y": Y, "dY": dY, "sizes": sizes}


# ============================================================================
# Engine: persistent workers holding their chunks' paths in RAM
# ============================================================================


def busy_python_processes(threshold: float = 20.0) -> int:
    """Python processes using > threshold % CPU (idle MCP/helper interpreters excluded)."""
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for ln in out.splitlines()[1:]:
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and "ython" in parts[1]:
            try:
                n += float(parts[0]) > threshold
            except ValueError:
                pass
    return n


def wait_for_cpu(limit: int = BUSY_LIMIT, poll: float = 60.0, max_wait: float = 1800.0) -> dict:
    t0 = time.time()
    k = busy_python_processes()
    while k >= limit and time.time() - t0 < max_wait:
        print(f"[wait] {k} busy python processes >= {limit}; sleeping {poll:.0f}s", flush=True)
        time.sleep(poll)
        k = busy_python_processes()
    return {"busy_python_processes": k, "waited_seconds": time.time() - t0}


def chunk_plan(spec: fk.EnsembleSpec, n_paths: int, *, tag: int, replicate: int = 0,
               chunk: int = DEFAULT_CHUNK, seed: int = BASE_SEED) -> dict:
    """Chunk sizes and SeedSequence children -- same layout as fk.simulate_ensemble."""
    sizes = [chunk] * (n_paths // chunk) + ([n_paths % chunk] if n_paths % chunk else [])
    entropy = fk.path_entropy(spec, seed=seed, tag=tag, replicate=replicate, chunk=chunk)
    children = np.random.SeedSequence(entropy).spawn(len(sizes))
    return {"sizes": sizes, "entropy": entropy, "children": children, "chunk": chunk,
            "seed": seed, "tag": tag, "replicate": replicate}


def _load_or_generate(spec: fk.EnsembleSpec, size: int, seedseq, cache: str | None,
                      gate: str) -> dict:
    t0 = time.perf_counter()
    if cache:
        p = Path(cache)
        if p.exists():
            zeta = np.load(p, mmap_mode=None)
            if zeta.shape == (size, spec.steps()) and zeta.dtype == np.float32:
                bm = block_max_abs_zeta(zeta)
                return {"zeta": zeta, "blockmax": bm, "source": "cache",
                        "seconds": time.perf_counter() - t0, "contact_steps": None,
                        "max_abs_zeta": float(bm.max())}
    out = generate_chunk(spec, size, seedseq, gate=gate)
    if cache:
        p = Path(cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, out["zeta"])
        os.replace(tmp, p)
    return {"zeta": out["zeta"], "blockmax": block_max_abs_zeta(out["zeta"]), "source": "generated",
            "seconds": time.perf_counter() - t0, "contact_steps": out["contact_steps"],
            "max_abs_zeta": out["max_abs_zeta"]}


def _eval_designs(stores: dict, spec, designs: list, want_grad: bool, group: int, mz) -> list:
    res = []
    for dd in designs:
        d = Design.from_dict(dd)
        res.append({ci: evaluate_chunk(st["zeta"], st["blockmax"], spec, d, want_grad=want_grad,
                                       group=group, mz=mz) for ci, st in stores.items()})
    return res


def _worker_main(conn, spec_dict: dict, tasks: list, gate: str, group: int) -> None:
    spec = fk.EnsembleSpec.from_dict(spec_dict)
    mz = em_mean_z(spec)
    stores = {}
    info = []
    for t in tasks:
        st = _load_or_generate(spec, t["size"], t["seedseq"], t.get("cache"), gate)
        stores[t["chunk"]] = st
        info.append({"chunk": t["chunk"], "size": t["size"], "source": st["source"],
                     "seconds": st["seconds"], "contact_steps": st["contact_steps"],
                     "max_abs_zeta": st["max_abs_zeta"]})
    conn.send(("ready", info))
    while True:
        msg = conn.recv()
        if msg[0] == "eval":
            try:
                conn.send(("ok", _eval_designs(stores, spec, msg[1], msg[2], group, mz)))
            except Exception as exc:  # noqa: BLE001
                conn.send(("error", repr(exc)))
        elif msg[0] == "stop":
            conn.close()
            return


@dataclass
class Law:
    """Exact discrete-time law of one design on one ensemble, with pathwise gradients."""

    spec: fk.EnsembleSpec
    design: Design
    t: np.ndarray                 # step (kill) times t_n = n dt, n = 1..S
    p: np.ndarray                 # P(T = t_n)
    dp: np.ndarray | None         # (K, S): d p_n / d theta, theta = (c, w, B)
    gY: np.ndarray                # (G, S) group sums of y_n
    gdY: np.ndarray | None        # (G, K, S)
    sizes: np.ndarray             # (G,) paths per group
    meta: dict = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return int(self.sizes.sum())

    def loo(self, g: int) -> tuple:
        """Delete-one-group (p, dp); group totals are cached on first use."""
        if "_tot" not in self.meta:
            self.meta["_tot"] = self.gY.sum(0)
            self.meta["_dtot"] = None if self.gdY is None else self.gdY.sum(0)
        N = self.n_paths - self.sizes[g]
        p = (self.meta["_tot"] - self.gY[g]) / N
        dp = None if self.gdY is None else (self.meta["_dtot"] - self.gdY[g]) / N
        return p, dp


def _assemble(spec, design, per_chunk: dict, meta: dict) -> Law:
    keys = sorted(per_chunk)
    gY = np.concatenate([per_chunk[k]["Y"] for k in keys])
    sizes = np.concatenate([per_chunk[k]["sizes"] for k in keys])
    grad = per_chunk[keys[0]]["dY"] is not None
    gdY = np.concatenate([per_chunk[k]["dY"] for k in keys]) if grad else None
    N = float(sizes.sum())
    t = fk.step_times(spec.dt, spec.steps())
    return Law(spec, design, t, gY.sum(0) / N, None if not grad else gdY.sum(0) / N,
               gY, gdY, sizes, dict(meta))


class Engine:
    """One fixed ensemble of unkilled paths; evaluates any number of designs on it."""

    def __init__(self, spec: fk.EnsembleSpec, n_paths: int, *, tag: int, replicate: int = 0,
                 chunk: int = DEFAULT_CHUNK, group: int = DEFAULT_GROUP, workers: int = 3,
                 seed: int = BASE_SEED, cache_name: str | None = None, chunks: list | None = None,
                 gate: str = "contact", max_workers: int = MAX_LOCAL_WORKERS, wait: bool = True):
        if chunk % group:
            raise ValueError("chunk must be a multiple of group")
        self.spec = spec
        self.group = group
        self.gate = gate
        self.plan = chunk_plan(spec, n_paths, tag=tag, replicate=replicate, chunk=chunk, seed=seed)
        idx = list(range(len(self.plan["sizes"]))) if chunks is None else list(chunks)
        self.chunk_ids = idx
        cache_dir = PATH_CACHE_ROOT / cache_name if cache_name else None
        tasks = [{"chunk": i, "size": self.plan["sizes"][i], "seedseq": self.plan["children"][i],
                  "cache": (str(cache_dir / f"chunk{i:04d}_zeta.npy") if cache_dir else None)}
                 for i in idx]
        self.n_paths = int(sum(t["size"] for t in tasks))
        self.workers = max(0, min(int(workers), int(max_workers), len(tasks)))
        self.meta = {"n_paths": self.n_paths, "chunk": chunk, "group": group, "seed": seed,
                     "tag": tag, "replicate": replicate, "entropy": self.plan["entropy"],
                     "chunks": idx, "gate": gate, "workers": self.workers,
                     "rng": "numpy Philox; chunk i uses SeedSequence(entropy).spawn(n_chunks)[i] "
                            "(entropy = fk.path_entropy(spec, seed, tag, replicate, chunk))",
                     "spec": spec.to_dict()}
        t0 = time.time()
        if wait and self.workers > 0:
            self.meta["cpu_wait"] = wait_for_cpu()
        self._mz = em_mean_z(spec)
        if self.workers == 0:
            self._stores = {}
            info = []
            for t in tasks:
                st = _load_or_generate(spec, t["size"], t["seedseq"], t["cache"], gate)
                self._stores[t["chunk"]] = st
                info.append({k: st[k] for k in ("source", "seconds", "contact_steps", "max_abs_zeta")}
                            | {"chunk": t["chunk"], "size": t["size"]})
            self.chunk_info = info
            self._procs = []
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self._conns, self._procs = [], []
            for k in range(self.workers):
                mine = tasks[k::self.workers]
                a, b = ctx.Pipe()
                pr = ctx.Process(target=_worker_main, args=(b, spec.to_dict(), mine, gate, group),
                                 daemon=True)
                pr.start()
                self._conns.append(a)
                self._procs.append(pr)
            info = []
            for cn in self._conns:
                tag_, payload = cn.recv()
                if tag_ != "ready":
                    raise RuntimeError(payload)
                info.extend(payload)
            self.chunk_info = sorted(info, key=lambda r: r["chunk"])
        self.meta["setup_seconds"] = time.time() - t0
        self.meta["chunk_info"] = self.chunk_info
        self.meta["max_abs_zeta"] = max(r["max_abs_zeta"] for r in self.chunk_info)

    def evaluate(self, designs, want_grad: bool = True):
        single = isinstance(designs, Design)
        ds = [designs] if single else list(designs)
        dicts = [d.to_dict() for d in ds]
        t0 = time.time()
        if self.workers == 0:
            parts = [_eval_designs(self._stores, self.spec, dicts, want_grad, self.group, self._mz)]
        else:
            for cn in self._conns:
                cn.send(("eval", dicts, want_grad))
            parts = []
            for cn in self._conns:
                tag_, payload = cn.recv()
                if tag_ != "ok":
                    raise RuntimeError(payload)
                parts.append(payload)
        wall = time.time() - t0
        laws = []
        for i, d in enumerate(ds):
            per_chunk = {}
            for part in parts:
                per_chunk.update(part[i])
            laws.append(_assemble(self.spec, d, per_chunk,
                                  {"eval_seconds": wall / len(ds), "engine": self.meta_brief()}))
        return laws[0] if single else laws

    def meta_brief(self) -> dict:
        return {k: self.meta[k] for k in ("n_paths", "chunk", "group", "seed", "tag", "replicate",
                                          "entropy", "chunks", "gate")}

    def close(self):
        for cn in getattr(self, "_conns", []):
            try:
                cn.send(("stop",))
            except Exception:  # noqa: BLE001
                pass
        for pr in self._procs:
            pr.join(timeout=30)
        self._procs = []
        self._conns = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================================
# Smooth functionals: f_h, peaks, valleys, masses, ratios (+ implicit derivatives)
# ============================================================================

_SQ2PI = math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf)


def _kernel_rows(u: np.ndarray, h: float, order):
    """Gaussian kernel K_h(u) (order 0), K_h'(u) (1), K_h''(u) (2) or CDF Phi(u/h) ('cdf')."""
    z = u / h
    if order == "cdf":
        return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    k = np.exp(-0.5 * z * z) / (_SQ2PI * h)
    if order == 0:
        return k
    if order == 1:
        return -(u / (h * h)) * k
    if order == 2:
        return ((u * u) / h**4 - 1.0 / (h * h)) * k
    raise ValueError(order)


def smooth_grid(p: np.ndarray, dt: float, h: float, order: int = 0) -> np.ndarray:
    """sum_n p_n K_h^{(order)}(t_i - t_n) at every step time t_i (last axis; reflect-free, zero pad)."""
    K = int(math.ceil(8.0 * h / dt))
    u = np.arange(-K, K + 1) * dt
    ker = _kernel_rows(u, h, order)
    p2 = np.atleast_2d(p)
    out = np.array([np.convolve(r, ker, mode="same") for r in p2])
    return out[0] if np.ndim(p) == 1 else out


def smooth_at(t0: float, p: np.ndarray, tgrid: np.ndarray, h: float, order) -> np.ndarray:
    """sum_n p_n K_h^{(order)}(t0 - t_n); p may be (S,) or (K, S)."""
    if order == "cdf":
        lo = np.searchsorted(tgrid, t0 - 9.0 * h)
        hi = np.searchsorted(tgrid, t0 + 9.0 * h)
        u = t0 - tgrid[lo:hi]
        full = np.sum(p[..., :lo], axis=-1)
        return full + np.sum(p[..., lo:hi] * _kernel_rows(u, h, "cdf"), axis=-1)
    lo = np.searchsorted(tgrid, t0 - 9.0 * h)
    hi = np.searchsorted(tgrid, t0 + 9.0 * h)
    u = t0 - tgrid[lo:hi]
    return np.sum(p[..., lo:hi] * _kernel_rows(u, h, order), axis=-1)


def refine_stationary(p: np.ndarray, tgrid: np.ndarray, h: float, t_start: float,
                      kind: str, max_iter: int = 60) -> tuple[float, bool]:
    """Newton on f_h'(t) = 0 from t_start (kind 'max' or 'min'); safeguarded by bisection."""
    t = float(t_start)
    dt = float(tgrid[1] - tgrid[0])
    lo, hi = t - 3 * dt, t + 3 * dt
    f1lo, f1hi = smooth_at(lo, p, tgrid, h, 1), smooth_at(hi, p, tgrid, h, 1)
    bracket = (f1lo > 0 > f1hi) if kind == "max" else (f1lo < 0 < f1hi)
    ok = True
    for _ in range(max_iter):
        g1 = smooth_at(t, p, tgrid, h, 1)
        g2 = smooth_at(t, p, tgrid, h, 2)
        step = -g1 / g2 if g2 != 0 else 0.0
        tn = t + step
        if bracket and not (lo < tn < hi):
            # bisection fallback inside the bracket
            if (g1 > 0) == (kind == "max"):
                lo = t
            else:
                hi = t
            tn = 0.5 * (lo + hi)
        if abs(tn - t) < 1e-13:
            t = tn
            break
        t = tn
    g2 = smooth_at(t, p, tgrid, h, 2)
    ok = (g2 < 0) if kind == "max" else (g2 > 0)
    return float(t), bool(ok)


def g_clock_cuts(design: Design, spec: fk.EnsembleSpec) -> list[float]:
    """[0, valleys of the design's exact EM free clock G between stripe times, tmax] (as N11)."""
    mf = _mf_cache(spec)
    G = np.zeros(mf["t"].size)
    for wj, cj in zip(design.w, design.c):
        G += wj * mf["gate"] * np.exp(-(mf["mz"] - cj) ** 2 / (2 * mf["v"])) / np.sqrt(2 * math.pi * mf["v"])
    tp = np.sort(design.times(spec))
    t = mf["t"]
    out = [0.0]
    for a, b in zip(tp[:-1], tp[1:]):
        sel = np.flatnonzero((t > a) & (t < b))
        i = sel[int(np.argmin(G[sel]))]
        y0, y1, y2 = G[i - 1], G[i], G[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        out.append(float(t[i] + off * spec.dt))
    out.append(float(spec.tmax))
    return out


def _functionals(p, dp, tgrid, h, cuts0, anchors=None, mass_cuts="valley"):
    """Peaks/valleys/masses/ratios and their raw-parameter gradients for one (p, dp).

    anchors (from the full sample) fix which stationary points are followed (jackknife)."""
    S = p.size
    tmax = float(tgrid[-1])
    m = len(cuts0) - 1
    peaks, pk_ok, pk_interior = [], [], []
    if anchors is None:
        fh = smooth_grid(p, tgrid[1] - tgrid[0], h, 0)
        for j in range(m):
            sel = np.flatnonzero((tgrid > cuts0[j]) & (tgrid <= cuts0[j + 1]))
            k = int(np.argmax(fh[sel]))
            pk_interior.append(bool(2 <= k <= sel.size - 3))
            t_, ok = refine_stationary(p, tgrid, h, tgrid[sel[k]], "max")
            peaks.append(t_)
            pk_ok.append(ok)
        valleys, vl_ok = [], []
        for j in range(m - 1):
            sel = np.flatnonzero((tgrid > peaks[j]) & (tgrid < peaks[j + 1]))
            if sel.size < 3:
                valleys.append(float("nan"))
                vl_ok.append(False)
                continue
            k = int(np.argmin(fh[sel]))
            t_, ok = refine_stationary(p, tgrid, h, tgrid[sel[k]], "min")
            valleys.append(t_)
            vl_ok.append(ok and 1 <= k <= sel.size - 2)
    else:
        peaks, pk_ok = zip(*[refine_stationary(p, tgrid, h, a, "max") for a in anchors["peaks"]])
        peaks, pk_ok = list(peaks), list(pk_ok)
        pk_interior = list(anchors["peaks_interior"])
        vv = [refine_stationary(p, tgrid, h, a, "min") if np.isfinite(a) else (float("nan"), False)
              for a in anchors["valleys"]]
        valleys, vl_ok = [v[0] for v in vv], [v[1] for v in vv]
    out = {"peaks": peaks, "peaks_ok": pk_ok, "peaks_interior": pk_interior,
           "valleys": valleys, "valleys_ok": vl_ok}
    out["f_peak"] = [float(smooth_at(t, p, tgrid, h, 0)) for t in peaks]
    out["f_valley"] = [float(smooth_at(t, p, tgrid, h, 0)) if np.isfinite(t) else float("nan")
                       for t in valleys]
    grad = dp is not None
    if grad:
        dpk = np.array([-smooth_at(t, dp, tgrid, h, 1) / smooth_at(t, p, tgrid, h, 2) for t in peaks])
        dvl = np.array([-smooth_at(t, dp, tgrid, h, 1) / smooth_at(t, p, tgrid, h, 2)
                        if np.isfinite(t) else np.full(dp.shape[0], np.nan) for t in valleys])
        dvl = dvl.reshape(len(valleys), dp.shape[0])
    # mass cuts
    if mass_cuts == "valley":
        cuts = [0.0] + list(valleys) + [tmax]
        dcuts = ([np.zeros(dp.shape[0])] + list(dvl) + [np.zeros(dp.shape[0])]) if grad else None
    else:
        cuts = [float(x) for x in mass_cuts]
        dcuts = [np.zeros(dp.shape[0])] * len(cuts) if grad else None
    total = float(p.sum())
    Fc, dFc = [], []
    for i, s in enumerate(cuts):
        if i == 0 and s <= 0.0:
            Fc.append(0.0)
            if grad:
                dFc.append(np.zeros(dp.shape[0]))
        elif i == len(cuts) - 1 and s >= tmax - 1e-12:
            Fc.append(total)
            if grad:
                dFc.append(dp.sum(axis=1))
        else:
            Fc.append(float(smooth_at(s, p, tgrid, h, "cdf")))
            if grad:
                dFc.append(smooth_at(s, dp, tgrid, h, "cdf")
                           + smooth_at(s, p, tgrid, h, 0) * dcuts[i])
    M = np.diff(Fc)
    out.update({"cuts": cuts, "masses": M.tolist(), "total_mass": total,
                "ratios": (M / M.sum()).tolist()})
    if grad:
        dF = np.array(dFc)
        dM = np.diff(dF, axis=0)
        dtot = dM.sum(axis=0)
        dr = (dM - (M / M.sum())[:, None] * dtot[None, :]) / M.sum()
        out["jac"] = {"peaks": dpk, "valleys": dvl, "masses": dM, "ratios": dr,
                      "total_mass": dp.sum(axis=1)}
    return out


def analyse(law: Law, h: float = 0.02, *, cuts0=None, mass_cuts="valley",
            jackknife: bool = True, jackknife_jac: bool = True) -> dict:
    """Peaks, valleys, masses, ratios with exact SAA Jacobians and delete-one-group SEs."""
    spec, d = law.spec, law.design
    cuts0 = g_clock_cuts(d, spec) if cuts0 is None else [float(x) for x in cuts0]
    full = _functionals(law.p, law.dp, law.t, h, cuts0, None, mass_cuts)
    res = {"h": h, "cuts0": cuts0, "mass_cuts_mode": mass_cuts if isinstance(mass_cuts, str) else "fixed",
           "param_names": d.param_names(), "n_paths": law.n_paths, "groups": int(law.sizes.size),
           "design": d.to_dict(spec)}
    res.update({k: v for k, v in full.items() if k != "jac"})
    if law.dp is not None:
        res["jac"] = full["jac"]
    if jackknife and law.sizes.size >= 2:
        G = law.sizes.size
        keys = ("peaks", "valleys", "masses", "ratios", "total_mass")
        reps = {k: [] for k in keys}
        jreps = {k: [] for k in keys}
        for g in range(G):
            p_g, dp_g = law.loo(g)
            o = _functionals(p_g, dp_g if jackknife_jac else None, law.t, h, cuts0,
                             anchors=full, mass_cuts=mass_cuts)
            for k in keys:
                reps[k].append(np.asarray(o[k], float))
                if jackknife_jac and "jac" in o:
                    jreps[k].append(np.asarray(o["jac"][k], float))

        def jse(a):
            a = np.asarray(a, float)
            return np.sqrt((G - 1) / G * np.sum((a - a.mean(0)) ** 2, axis=0))

        res["se"] = {k: jse(reps[k]) for k in keys}
        if jackknife_jac and jreps["peaks"]:
            res["se_jac"] = {k: jse(jreps[k]) for k in keys}
    return res


def design_outputs(res: dict, formulation: str = "F1") -> tuple[np.ndarray, np.ndarray]:
    """Output vector and raw Jacobian: F1 = (peaks, ratios[:m-1]); F2 = (peaks, masses)."""
    m = len(res["peaks"])
    if formulation == "F1":
        y = np.concatenate([res["peaks"], res["ratios"][:m - 1]])
        J = np.vstack([res["jac"]["peaks"], res["jac"]["ratios"][:m - 1]])
    elif formulation == "F2":
        y = np.concatenate([res["peaks"], res["masses"]])
        J = np.vstack([res["jac"]["peaks"], res["jac"]["masses"]])
    else:
        raise ValueError(formulation)
    return y, J


def design_map(law: Law, formulation: str = "F1", *, h: float = 0.02, cuts0=None, targets=None,
               coords: str | None = None, jackknife: bool = False) -> dict:
    """Outputs y (F1: m peak times + m-1 ratios; F2: m peak times + m masses), their exact SAA
    Jacobian in design coordinates (default F1: (t_1..t_m, u_1..u_{m-1}); F2: + B) and, with
    targets, the residual y - targets.  This is the map NU-B's SAA Newton solves."""
    res = analyse(law, h, cuts0=cuts0, jackknife=jackknife)
    y, Jraw = design_outputs(res, formulation)
    coords = coords or ("times_tangent" if formulation == "F1" else "times_tangent_B")
    J = to_design_coords(Jraw, law.design, law.spec, coords)
    r = None if targets is None else y - np.asarray(targets, float)
    return {"y": y, "J": J, "J_raw": Jraw, "residual": r, "coords": coords, "analysis": res}


def count_modes(law: Law, h: float = 0.02, *, t_lo: float = 0.2, rel_prominence: float = 1e-3) -> dict:
    """Local maxima of f_h on the step grid in (t_lo, tmax - 4h) whose prominence (height above
    the higher of the two adjacent minima) exceeds rel_prominence * max f_h (diagnostic count)."""
    dt = law.spec.dt
    f = smooth_grid(law.p, dt, h, 0)
    t = law.t
    sel = np.flatnonzero((t > t_lo) & (t < t[-1] - 4 * h))
    fs = f[sel]
    d = np.sign(np.diff(fs))
    mx = [i + 1 for i in range(d.size - 1) if d[i] > 0 and d[i + 1] <= 0]
    mn = [i + 1 for i in range(d.size - 1) if d[i] < 0 and d[i + 1] >= 0]
    top = float(f.max())
    peaks = []
    for i in mx:
        left = [fs[j] for j in mn if j < i]
        right = [fs[j] for j in mn if j > i]
        base = max(left[-1] if left else fs[0], right[0] if right else fs[-1])
        prom = fs[i] - base
        if prom > rel_prominence * top:
            peaks.append({"t": float(t[sel[i]]), "f": float(fs[i]), "prominence": float(prom)})
    return {"h": h, "n_modes": len(peaks), "modes": peaks, "rel_prominence": rel_prominence}


def to_design_coords(J: np.ndarray, design: Design, spec: fk.EnsembleSpec,
                     coords: str = "times_tangent", eps_delta: float | None = None) -> np.ndarray:
    """Convert raw columns (c_1..c_m, w_1..w_m, B) to design coordinates.

    'raw' | 'times' (t_j, w_j, B) | 'times_tangent' (t_j, u_k) | 'times_tangent_B' (t_j, u_k, B)
    | 'delta_tangent' (delta_j with t_j = T_j - eps delta_j, u_k) | 'projected' (c_j, w_j - mean, B)
    with u_k the simplex tangent coordinates w = w0 + sum_k u_k (e_k - e_m), k = 1..m-1.
    """
    J = np.atleast_2d(np.asarray(J, float))
    m = design.m
    Jc, Jw, JB = J[:, :m], J[:, m:2 * m], J[:, 2 * m:2 * m + 1]
    Jt = Jc * dc_dt(design.times(spec), spec)[None, :]
    Ju = Jw[:, :m - 1] - Jw[:, m - 1:m]
    if coords == "raw":
        return J
    if coords == "times":
        return np.hstack([Jt, Jw, JB])
    if coords == "times_tangent":
        return np.hstack([Jt, Ju])
    if coords == "times_tangent_B":
        return np.hstack([Jt, Ju, JB])
    if coords == "delta_tangent":
        e = spec.eps if eps_delta is None else eps_delta
        return np.hstack([-e * Jt, Ju])
    if coords == "projected":
        return np.hstack([Jc, Jw - Jw.mean(axis=1, keepdims=True), JB])
    raise ValueError(coords)


# ============================================================================
# Noiseless EM mean-field law (warm starts, smoothing-bias diagnostics)
# ============================================================================

_MF: dict = {}


def _spec_key(spec: fk.EnsembleSpec) -> tuple:
    return tuple(sorted((k, v) for k, v in spec.to_dict().items()
                        if k not in ("target_times_resolved", "centres_z_resolved",
                                     "target_times", "centres_z", "m")
                        and not isinstance(v, (list, dict))))


def _mf_cache(spec: fk.EnsembleSpec) -> dict:
    key = _spec_key(spec)
    if key not in _MF:
        mom = fk.em_moments(spec)
        _MF[key] = {"t": mom["t"], "mz": mom["mean_z"], "v": mom["var_z"] + spec.sd() ** 2,
                    "gate": fk.contact_probability_em(spec)}
    return _MF[key]


def mean_field_law(design: Design, spec: fk.EnsembleSpec, want_grad: bool = True) -> Law:
    """p_n = e^{-B Lam_{n-1}} (1 - e^{-B G_n dt}), G the exact EM free clock (field at Z,
    EM contact probability); the same formulas as the path estimator with one 'mean path'."""
    mf = _mf_cache(spec)
    m = design.m
    area = spec.torus_w ** spec.n_perp
    phis, dphis = [], []
    for cj in design.c:
        d = mf["mz"] - cj
        ph = mf["gate"] * np.exp(-d * d / (2 * mf["v"])) / np.sqrt(2 * math.pi * mf["v"]) * spec.dt / area
        phis.append(ph)
        dphis.append(ph * d / mf["v"])
    phis = np.array(phis)
    e = np.sum(design.w[:, None] * phis, axis=0)
    B = design.B
    X = np.concatenate([[0.0], np.cumsum(e)[:-1]])
    A = np.exp(-B * X)
    E1 = np.exp(-B * e)
    y = A * (-np.expm1(-B * e))
    dy = None
    if want_grad:
        de = np.vstack([design.w[:, None] * np.array(dphis), phis])       # (2m, S)
        dX = np.hstack([np.zeros((2 * m, 1)), np.cumsum(de, axis=1)[:, :-1]])
        dy = np.vstack([B * A * E1 * de - B * y * dX, (A * E1 * e - y * X)[None, :]])
    return Law(spec, design, mf["t"], y, dy, y[None, :], None if dy is None else dy[None],
               np.array([1]), {"kind": "mean_field_em"})


def smoothing_bias(design: Design, spec: fk.EnsembleSpec, h: float, cuts0=None) -> dict:
    """Peak-time offset of the h-smoothed functional on the noiseless mean-field law:
    t*_h - t*_0, with t*_0 the 3-point-parabola argmax of the raw per-step density."""
    law = mean_field_law(design, spec, want_grad=False)
    r = analyse(law, h, cuts0=cuts0, jackknife=False)
    dens = law.p / spec.dt
    t = law.t
    raw = []
    cuts = r["cuts0"]
    for j in range(len(cuts) - 1):
        sel = np.flatnonzero((t > cuts[j]) & (t <= cuts[j + 1]))
        i = sel[int(np.argmax(dens[sel]))]
        y0, y1, y2 = dens[i - 1], dens[i], dens[i + 1]
        den = y0 - 2 * y1 + y2
        raw.append(float(t[i] + (0.5 * (y0 - y2) / den if den < 0 else 0.0) * spec.dt))
    return {"h": h, "peaks_h": r["peaks"], "peaks_raw": raw,
            "bias": [a - b for a, b in zip(r["peaks"], raw)]}


# ============================================================================
# Validation commands
# ============================================================================


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.random.SeedSequence):
        return {"entropy": o.entropy, "spawn_key": list(o.spawn_key)}
    return o


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(_jsonable(payload), indent=1))
    os.replace(tmp, path)
    print(f"[write] {path}", flush=True)


def _perturbed(d: Design, k: int, step: float) -> Design:
    c, w, B = d.c.copy(), d.w.copy(), d.B
    if k < d.m:
        c[k] += step
    elif k < 2 * d.m:
        w[k - d.m] += step
    else:
        B += step
    return Design(c, w, B)


def cmd_selftest(args) -> dict:
    """Small local checks: worker invariance, FD gradients (law and functionals), Euler identity,
    mean-field gradient, smoothing bias."""
    spec = path_spec(0.1)
    T = (0.8, 1.6, 2.8)
    th = Design.from_times(T, (0.2, 0.3, 0.5), 4.0, spec, "selftest")
    out = {"item": "V2_NU_A selftest", "driver": HERE.name, "spec": spec.to_dict(),
           "design": th.to_dict(spec), "h": 0.02}
    N = int(float(args.paths))
    e0 = Engine(spec, N, tag=TAGS["NU_A_FD"], chunk=N // 2, group=N // 10, workers=0)
    law0 = e0.evaluate(th)
    with Engine(spec, N, tag=TAGS["NU_A_FD"], chunk=N // 2, group=N // 10, workers=2,
                wait=False) as e2:
        law2 = e2.evaluate(th)
    out["worker_invariance_max_abs_diff"] = {
        "p": float(np.max(np.abs(law0.p - law2.p))), "dp": float(np.max(np.abs(law0.dp - law2.dp)))}
    fdl, fdf = {}, {}
    r0 = analyse(law0, 0.02, jackknife=False)
    step = 1e-5
    for k, nm in enumerate(th.param_names()):
        lp = e0.evaluate(_perturbed(th, k, step))
        lm = e0.evaluate(_perturbed(th, k, -step))
        fd = (lp.p - lm.p) / (2 * step)
        fdl[nm] = {"max_rel_err": float(np.max(np.abs(fd - law0.dp[k])) / np.max(np.abs(law0.dp[k]))),
                   "scale": float(np.max(np.abs(law0.dp[k])))}
        rp = analyse(lp, 0.02, cuts0=r0["cuts0"], jackknife=False)
        rm = analyse(lm, 0.02, cuts0=r0["cuts0"], jackknife=False)
        fdf[nm] = {}
        for key in ("peaks", "valleys", "masses", "ratios"):
            fdv = (np.asarray(rp[key]) - np.asarray(rm[key])) / (2 * step)
            an = np.asarray(r0["jac"][key])[:, k]
            fdf[nm][key] = {"max_abs_err": float(np.max(np.abs(fdv - an))),
                            "scale": float(np.max(np.abs(an)))}
    out["fd_law"] = fdl
    out["fd_functionals"] = fdf
    lhs = np.sum(th.w[:, None] * law0.dp[th.m:2 * th.m], axis=0)
    rhs = th.B * law0.dp[2 * th.m]
    out["euler_identity_rel"] = float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)))
    mfl = mean_field_law(th, spec)
    out["fd_mean_field"] = {}
    for k, nm in enumerate(th.param_names()):
        fd = (mean_field_law(_perturbed(th, k, 1e-6), spec, False).p
              - mean_field_law(_perturbed(th, k, -1e-6), spec, False).p) / 2e-6
        out["fd_mean_field"][nm] = float(np.max(np.abs(fd - mfl.dp[k])) / np.max(np.abs(mfl.dp[k])))
    out["smoothing_bias_mf"] = {str(h): smoothing_bias(th, spec, h) for h in (0.01, 0.02, 0.04)}
    out["analysis_example"] = {k: r0[k] for k in ("peaks", "valleys", "masses", "ratios",
                                                   "total_mass", "cuts0", "cuts")}
    rj = analyse(law0, 0.02)
    out["analysis_example"]["se"] = rj["se"]
    out["n_paths"] = N
    out["timing"] = {"setup_seconds": e0.meta["setup_seconds"],
                     "eval_seconds_with_grad": law0.meta["eval_seconds"]}
    worst_law = max(v["max_rel_err"] for v in fdl.values())
    worst_fun = max(v2["max_abs_err"] / max(v2["scale"], 1e-12) for v in fdf.values() for v2 in v.values())
    out["pass"] = bool(worst_law < 1e-6 and worst_fun < 1e-5 and out["euler_identity_rel"] < 1e-12
                       and out["worker_invariance_max_abs_diff"]["p"] == 0.0)
    out["worst"] = {"fd_law_rel": worst_law, "fd_functionals_rel": worst_fun}
    write_json(OUT_DIR / "selftest.json", out)
    return out


def cmd_crosscheck_fk(args) -> dict:
    """Same seeds as FK ensemble n0_m2_eps0.1 (tag 80, chunk 5e4): chunk-0 bin masses must agree."""
    ens = fk.load_ensemble(args.ensemble)
    idx = ens.index
    spec = ens.spec
    pspec = path_spec(spec.eps, dt=spec.dt, tmax=spec.tmax)
    B, w = float(args.B), np.full(spec.m, 1.0 / spec.m)
    th = Design(spec.centres(), w, B)
    eng = Engine(pspec, int(idx["n_paths"]), tag=int(idx["tag"]), replicate=int(idx["replicate"]),
                 chunk=int(idx["chunk"]), group=int(idx["chunk"]) // 10, workers=0, chunks=[0],
                 seed=int(idx["seed"]))
    if eng.plan["entropy"] != idx["seed_entropy"]:
        raise AssertionError("path entropy differs from the FK ensemble's")
    law = eng.evaluate(th, want_grad=False)
    _, dX, _ = next(ens.iter_chunks("full"))
    cps = ens.cps
    kidx = np.arange(cps.size)
    Xs, seg = fk._selected_exposure(dX, w, kidx)
    y = np.exp(-B * Xs[:, :-1]) * (-np.expm1(-B * seg))
    fk_bins = y.mean(0)
    cp = np.concatenate([[0.0], np.cumsum(law.p)])
    mine = cp[cps[1:]] - cp[cps[:-1]]
    first = cp[cps[0]]
    sig = fk_bins > 1e-10
    rel = np.abs(mine[sig] - fk_bins[sig]) / fk_bins[sig]
    out = {"item": "V2_NU_A crosscheck vs FK stored ensemble", "ensemble": args.ensemble,
           "chunk": 0, "paths": int(dX.shape[0]), "B": B, "w": w, "centres": th.c,
           "entropy": idx["seed_entropy"], "bins_compared": int(sig.sum()),
           "max_rel_diff": float(rel.max()), "median_rel_diff": float(np.median(rel)),
           "mass_before_first_checkpoint": float(first),
           "total_mass": {"v2": float(law.p.sum()), "fk": float(fk_bins.sum())},
           "pass": bool(rel.max() < 1e-4)}
    write_json(OUT_DIR / "crosscheck_fk.json", out)
    return out


N11_DK_DIR = FB_DATA / "N11_shift_compensation" / "directkill"


def dk_law_from_counts(C: np.ndarray, walkers: int, spec: fk.EnsembleSpec) -> Law:
    """Direct-kill histogram (counts per step per chunk) in the Law format (groups = chunks)."""
    C = np.asarray(C, float)
    sizes = np.full(C.shape[0], walkers / C.shape[0])
    return Law(spec, None, fk.step_times(spec.dt, spec.steps()), C.sum(0) / walkers, None, C, None,
               sizes, {"kind": "direct_kill"})


def functionals_with_se(law: Law, h: float, cuts0, mass_cuts="valley") -> dict:
    """_functionals on the full sample + delete-one-group jackknife SEs (no gradients)."""
    full = _functionals(law.p, None, law.t, h, cuts0, None, mass_cuts)
    G = law.sizes.size
    keys = ("peaks", "valleys", "masses", "ratios", "total_mass")
    reps = {k: [] for k in keys}
    for g in range(G):
        pg, _ = law.loo(g)
        o = _functionals(pg, None, law.t, h, cuts0, anchors=full, mass_cuts=mass_cuts)
        for k in keys:
            reps[k].append(np.asarray(o[k], float))
    full["se"] = {k: np.sqrt((G - 1) / G * np.sum((np.asarray(v) - np.mean(v, 0)) ** 2, axis=0))
                  for k, v in reps.items()}
    return full


def binned_chi2(law_a: Law, law_b: Law, spec, lo: float = 0.5, hi: float = 3.5,
                width: float = 0.02, min_count: float = 20.0) -> dict:
    """Diagonal chi-square between two independent laws on common bins (group-based variances)."""
    e = np.rint(np.arange(lo, hi + 0.5 * width, width) / spec.dt).astype(int)

    def bins(L):
        cg = np.concatenate([np.zeros((L.gY.shape[0], 1)), np.cumsum(L.gY, axis=1)], axis=1)
        gb = (cg[:, e[1:]] - cg[:, e[:-1]]) / L.sizes[:, None]
        mean = (gb * L.sizes[:, None]).sum(0) / L.sizes.sum()
        if L.meta.get("kind") == "direct_kill":
            # multinomial counts: exact variance p (1 - p) / N (no noisy group estimate)
            return mean, mean * (1.0 - mean) / L.sizes.sum()
        G = L.sizes.size
        var = ((gb - mean) ** 2).sum(0) / (G * (G - 1))
        return mean, var

    a, va = bins(law_a)
    b, vb = bins(law_b)
    Nb = law_b.n_paths
    sel = b * Nb >= min_count
    z = (a[sel] - b[sel]) / np.sqrt(va[sel] + vb[sel])
    from math import erfc, sqrt
    chi2 = float(np.sum(z * z))
    dof = int(sel.sum())
    # Wilson--Hilferty p-value
    x = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof))
    return {"chi2": chi2, "dof": dof, "chi2_per_dof": chi2 / dof, "p_value_wh": 0.5 * erfc(x / sqrt(2)),
            "max_abs_z": float(np.max(np.abs(z))), "bin_width": width, "range": [lo, hi]}


def cmd_validate_dk(args) -> dict:
    """Estimator values vs the existing N11 direct-kill data (1e6 walkers, tag 91)."""
    h = float(args.h)
    files = sorted(N11_DK_DIR.glob("dk_*.json"))
    by_eps: dict = {}
    for f in files:
        d = json.loads(f.read_text())
        by_eps.setdefault(float(d["cell"]["eps"]), []).append((f, d))
    out = {"item": "V2_NU_A validation vs existing direct-kill data", "driver": HERE.name, "h": h,
           "fk_paths": int(float(args.paths)), "fk_tag": TAGS["NU_A"],
           "definitions": {
               "peak": "maximiser of the Gaussian-kernel-smoothed density f_h (h in time units) in the "
                       "basin, Newton on f_h'=0; same functional for FK and DK",
               "basins": "located on the design's EM free-clock G valleys; masses on the FK f_h-valley "
                         "cuts (applied as fixed cuts to DK)",
               "se": "delete-one-group jackknife (FK: groups of --group paths; DK: 20 chunks of 5e4); "
                     "binned chi2: FK bin variance from groups, DK multinomial p(1-p)/N",
               "z": "(DK - FK) / sqrt(se_DK^2 + se_FK^2)"},
           "cells": {}}
    for eps, items in sorted(by_eps.items()):
        spec = path_spec(eps)
        with Engine(spec, int(float(args.paths)), tag=TAGS["NU_A"], workers=int(args.workers),
                    group=int(float(args.group))) as eng:
            designs = [Design(d["centres_z"], d["w"], d["cell"]["B"], d["design"]) for _, d in items]
            laws = eng.evaluate(designs)
            eng_meta = eng.meta_brief() | {"setup_seconds": eng.meta["setup_seconds"]}
        for (f, d), th, law in zip(items, designs, laws):
            ck = f"m{d['cell']['m']}_eps{eps:g}_B{d['cell']['B']:g}"
            cuts0 = g_clock_cuts(th, spec)
            rf = analyse(law, h, cuts0=cuts0)
            dkl = dk_law_from_counts(d["counts_per_step_per_chunk"], int(d["walkers"]), spec)
            rd = functionals_with_se(dkl, h, cuts0, mass_cuts=rf["cuts"])
            rd_own = functionals_with_se(dkl, h, cuts0, mass_cuts="valley")

            def z(a, sa, b, sb):
                return ((np.asarray(a) - np.asarray(b)) / np.sqrt(np.asarray(sa) ** 2
                                                                  + np.asarray(sb) ** 2)).tolist()

            row = {"dk_file": str(f.relative_to(REPORT)), "design": th.to_dict(spec),
                   "dk_walkers": int(d["walkers"]), "dk_seed_entropy": d["seed_entropy"],
                   "fk": {k: rf[k] for k in ("peaks", "valleys", "masses", "ratios", "total_mass", "cuts")}
                   | {"se": rf["se"], "jac": rf["jac"], "eval_seconds": law.meta["eval_seconds"]},
                   "dk": {k: rd[k] for k in ("peaks", "valleys", "masses", "ratios", "total_mass")}
                   | {"se": rd["se"], "masses_own_valley_cuts": rd_own["masses"],
                      "cuts_own": rd_own["cuts"]},
                   "z": {"peaks": z(rd["peaks"], rd["se"]["peaks"], rf["peaks"], rf["se"]["peaks"]),
                         "valleys": z(rd["valleys"], rd["se"]["valleys"], rf["valleys"], rf["se"]["valleys"]),
                         "masses": z(rd["masses"], rd["se"]["masses"], rf["masses"], rf["se"]["masses"]),
                         "ratios": z(rd["ratios"], rd["se"]["ratios"], rf["ratios"], rf["se"]["ratios"]),
                         "total_mass": z(rd["total_mass"], rd["se"]["total_mass"], rf["total_mass"],
                                         rf["se"]["total_mass"])},
                   "binned_chi2": binned_chi2(law, dkl, spec),
                   "smoothing_bias_mf": smoothing_bias(th, spec, h, cuts0)}
            out["cells"].setdefault(ck, {"engine": eng_meta, "designs": {}})["designs"][d["design"]] = row
        # directional (gradient) consistency between the two designs of each cell
        for ck, cell in out["cells"].items():
            if not ck.endswith(f"eps{eps:g}_B{items[0][1]['cell']['B']:g}") and f"eps{eps:g}" not in ck:
                continue
            ds = cell["designs"]
            if "compFK2" in ds and "nwt2" in ds and "directional" not in cell:
                a, b = ds["compFK2"], ds["nwt2"]
                da = np.concatenate([np.subtract(b["design"]["c"], a["design"]["c"]),
                                     np.subtract(b["design"]["w"], a["design"]["w"]), [0.0]])
                dirc = {}
                for key in ("peaks", "masses"):
                    lin = 0.5 * (np.asarray(a["fk"]["jac"][key]) + np.asarray(b["fk"]["jac"][key])) @ da
                    dfk = np.subtract(b["fk"][key], a["fk"][key])
                    ddk = np.subtract(b["dk"][key], a["dk"][key])
                    sdk = np.sqrt(np.square(a["dk"]["se"][key]) + np.square(b["dk"]["se"][key]))
                    dirc[key] = {"fk_crn_difference": dfk.tolist(), "fk_linearised": lin.tolist(),
                                 "dk_difference": ddk.tolist(), "dk_se": sdk.tolist(),
                                 "z_dk_vs_linearised": ((ddk - lin) / sdk).tolist(),
                                 "rel_lin_vs_crn": (np.abs(lin - dfk) / np.maximum(np.abs(dfk), 1e-15)).tolist()}
                cell["directional"] = {"from": "compFK2", "to": "nwt2", "dtheta_raw": da.tolist(), **dirc}
    zs = [abs(v) for c in out["cells"].values() for r in c["designs"].values()
          for k in ("peaks", "masses", "ratios") for v in r["z"][k]]
    out["summary"] = {"max_abs_z": max(zs), "n_z": len(zs),
                      "chi2_per_dof": {f"{ck}/{dn}": r["binned_chi2"]["chi2_per_dof"]
                                       for ck, c in out["cells"].items() for dn, r in c["designs"].items()}}
    write_json(OUT_DIR / "validate_dk.json", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--paths", default="10000")
    s = sub.add_parser("crosscheck-fk")
    s.add_argument("--ensemble", default="n0_m2_eps0.1")
    s.add_argument("--B", default="1.0")
    s = sub.add_parser("validate-dk")
    s.add_argument("--paths", default="2e5")
    s.add_argument("--h", default="0.02")
    s.add_argument("--workers", default="3")
    s.add_argument("--group", default="1000")
    args = ap.parse_args(argv)
    fn = {"selftest": cmd_selftest, "crosscheck-fk": cmd_crosscheck_fk,
          "validate-dk": cmd_validate_dk}[args.cmd]
    res = fn(args)
    brief = {k: res[k] for k in ("pass", "worst", "max_rel_diff", "summary") if k in res}
    print(json.dumps(_jsonable(brief), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
