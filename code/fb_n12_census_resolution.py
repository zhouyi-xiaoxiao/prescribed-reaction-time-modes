#!/usr/bin/env python3
"""N12 -- census resolution of the three N9a cells that missed the last maximum.

N9a (fb_n9a_matching_census.py) searched the per-step exact law of the gated
model for critical points.  In three cells, (m, eps, B) = (2, 0.05, 50),
(3, 0.05, 20), (3, 0.05, 50) (equal weights, contact radius a = 0.4, dt = 1e-3),
the last maximum was not resolved at some or all resolutions; the late basin
holds 2.5e-5, 3.2e-5, 1.5e-6 of the mass and is carried by a median of 2-22
effective FK paths per step.

Why the plain FK estimator fails there.  At eps = 0.05 the pair stays in
contact during a passage unless |R|_mi >= a, a large deviation of the
transverse separation (P ~ 1e-4 at t ~ 1).  With B >= 20 the passage exposures
lambda_j = B w_j / (W |mu'(t_j)|) are 17 (m=2, B=50) and 3.7 + 8.3 / 9.3 + 20.6
(m=3, B=20 / 50), so the mean-field late mass exp(-sum_{i<m} lambda_i) is
4e-8, 6e-6, 1e-13 and the late basin is carried by GATE-ESCAPE paths (out of
contact during the earlier crossings).  Unkilled FK paths almost never escape.

Estimator.  Importance-sampled (tilted) FK for the SAME discrete chain.  The
transverse separation R_perp (a Brownian motion on the torus, increments
2 eps sqrt(d0 dt) xi) is simulated under a defensive mixture proposal
q = pi_0 p + sum_k pi_k q_k, where q_k adds a state-dependent drift
b_k(t, x) dt to the R_perp increment: during 'pull' phases towards the
nearer of {l, W - l} (|x|_mi -> l, out of contact for l > a), during 'back'
phases towards the nearer image of 0.  All other coordinates are untouched.
With Delta the realised R_perp increment and s^2 = 4 eps^2 d0 dt,
    log(q_k / p) += (2 b_k Delta - b_k^2) / (2 s^2)     (every step, every k),
and the path weight is LR_n = 1 / (pi_0 + sum_k pi_k exp(L_k,n)) <= 1 / pi_0.
Because the mixture's marginal on the first n steps is the mixture of the
marginals, E_q[LR_n h(path_{0:n})] = E_p[h] for every n (unbiased; the kernel
of the chain and the FK weights are exactly those of exact_m_prr_fk_exact_law):
    f_n = E_q[ LR_n (1 - exp(-B e_n)) exp(-B X_{n-1}) ].
Checks: E_q[LR] = 1 at every phase end; early basins against the N9a plain FK.

Census: fb_n9a_matching_census.census_one on the IS per-step law (same
resolutions h/eps in {0.05, 0.1, 0.25, 0.5, 1}, 50 batches, |z| > 5).
Prominence: protocol P (fk.classify_expected, 0.02 production bins, bandwidth
0.04, 1e6 walkers) on the IS expected histogram; jackknife over batches.

Seeds: base 20260923; tag 91 IS ensembles (dt = 1e-3), tag 93 IS dt = 5e-4
check, tag 92 direct kill (large-N late-basin check), tag 94 plain-FK diagnostic
of the passage-1 survival (m = 2), tag 999 pilots (throwaway).  Entropy recorded
per run.

Validation lesson (2026-09-23).  The first two m = 2 proposals (M, M2: h-transform
confining the tilted pairs inside the out-of-contact band for the whole passage)
were 17 % low in the late basin (z = -4.75 against 4e7 direct-kill walkers) with
self-consistent error bars; the plain 1e7-path diagnostic (diag-plain) located the
deficit in brief-contact paths (0.1 <= B X_cut < 2).  M4 adds 'free_zone'
components and agrees with direct kill (proposal_validation_m2 in the summary).
Production: m = 2 -> M4 (1e6 paths), m = 3 -> M3 (2e6 paths).

Usage
-----
    python3 fb_n12_census_resolution.py pilot --cell 0 --paths 2e4 --proposal A
    python3 fb_n12_census_resolution.py simulate --cell 0 --paths 1e6 [--dt 5e-4]
    python3 fb_n12_census_resolution.py dk --cell 0 --walkers 3e7
    python3 fb_n12_census_resolution.py diag-plain --paths 1e7
    python3 fb_n12_census_resolution.py analyze
    python3 fb_n12_census_resolution.py figure
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402

OUT = fk.FB_DATA / "N12_census_resolution"
STORE = fk.ENSEMBLE_ROOT / "n12_is"
SEED = fk.BASE_SEED
TAG_IS = 91
TAG_DK = 92
TAG_IS_DT = 93
MAX_WORKERS = 3
CHUNK = 50_000

CELLS = [
    {"m": 2, "eps": 0.05, "B": (50.0,)},
    {"m": 3, "eps": 0.05, "B": (20.0, 50.0)},
]
# the three N9a cells: (cell index, B)
TARGET_CASES = [(0, 50.0), (1, 20.0), (1, 50.0)]

# Z-aware proposal components (component 0 = no drift is implicit).  Each component
# lists the stripes j (0-based) whose passage the pair should spend OUT of contact
# ('escape'), and optionally a stripe for which it should be back IN contact ('back').
# The drift on the transverse separation x = R_perp mod W depends on (x, Z):
#   before the exposure zone of the next escape stripe (Z > c_j + H):
#       softened Brownian bridge to |x|_mi = 'lvl' timed on the deterministic
#       zone-entry time tau_e(Z) = ln((Z - zbar)/(c_j + H - zbar))/gamma:
#       b = (tgt - x) dt / (tau_e + tau0);
#   from the first to the last escape zone: Doob h-transform of BM conditioned to stay
#       in the out-of-contact band (a, W - a), h = sin(pi (x - a)/(W - 2a));
#   after the last escape zone, if 'back' = k: bridge to |x|_mi = 'lvl_back' timed on
#       the entry of Z into c_k + Hb; otherwise no drift.
# H = H_sd * eps * rho is where the passage exposure is non-negligible (lambda Phi(-3)
# <= 0.03 for the exposures of these cells).
def _mix(specs, pi0):
    """specs: (escape stripes, back stripe or None, levels, zone half-widths in sd)."""
    comps = []
    for esc, back, lvls, hs in specs:
        for lv in lvls:
            for h in hs:
                comps.append({"escape": esc, "back": back, "lvl": lv, "H_sd": h})
    k = len(comps)
    return {"pi": [pi0] + [(1.0 - pi0) / k] * k, "comps": comps}


def _mix_soft(specs, pi0):
    """As _mix, but every tilted component is 'soft_zone' (no in-zone forcing while the
    pair is in contact), so partial exposures and late band entries are generated."""
    base = _mix(specs, pi0)
    base["comps"] = [dict(c, soft_zone=True) for c in base["comps"]]
    return base


PROPOSALS = {
    # M3 (m = 2): soft-zone analogue of the m = 3 production proposal, added after the
    # 4e7-walker direct-kill check showed the hard-zone M2 run 17 % low in the late basin
    # (z = -4.75; partial-exposure paths reached only through the untilted component).
    # M4 (m = 2): M3 plus 'free_zone' components (pull to the band edge before the zone,
    # no drift inside it).  A 1e7-path plain FK diagnostic (tag 94) located the M2 deficit
    # in the partial-exposure classes 0.1 <= B X_cut < 2 (-40 % in [0.1, 0.5)).
    2: {"M4": {"pi": [0.20] + [0.80 / 13] * 13,
               "comps": ([dict(c, soft_zone=True) for c in
                          _mix([([0], None, (0.41, 0.45), (3.0, 2.0, 1.25)),
                                ([0], 1, (0.43,), (3.0,))], 0.2)["comps"]]
                         + [{"escape": [0], "back": None, "lvl": lv, "H_sd": h, "free_zone": True}
                            for lv in (0.40, 0.42, 0.44) for h in (3.0, 2.0)])},
        "M3": _mix_soft([([0], None, (0.41, 0.45), (3.0, 2.0, 1.25)),
                         ([0], 1, (0.43,), (3.0,))], 0.20),
        "M2": _mix([([0], None, (0.40, 0.43, 0.46), (3.0, 2.0, 1.25)),
                    ([0], 1, (0.43,), (3.0,))], 0.15),
        "M": _mix([([0], None, (0.40, 0.43, 0.46), (3.0, 2.0)),
                   ([0], 1, (0.43,), (3.0,))], 0.15)},
    3: {"M3": _mix_soft([([0, 1], None, (0.41, 0.45), (3.0, 2.0)),
                         ([1], None, (0.41, 0.45), (3.0, 2.0)),
                         ([0, 1], 2, (0.43,), (3.0,)),
                         ([1], 2, (0.43,), (3.0,))], 0.20),
        "M2": _mix([([0, 1], None, (0.41, 0.45), (3.0, 2.0, 1.25)),
                    ([1], None, (0.41, 0.45), (3.0, 2.0, 1.25)),
                    ([0, 1], 2, (0.43,), (3.0,)),
                    ([1], 2, (0.43,), (3.0,)),
                    ([0], 1, (0.43,), (2.0,))], 0.25),
        "M": _mix([([0, 1], None, (0.40, 0.43, 0.46), (3.0, 2.0)),
                   ([1], None, (0.40, 0.43, 0.46), (3.0, 2.0)),
                   ([0, 1], 2, (0.43,), (3.0,)),
                   ([1], 2, (0.43,), (3.0,)),
                   ([0], 1, (0.43,), (3.0,))], 0.30)},
}
PROPOSAL_PARAMS = {"lvl": 0.433, "tau0": 0.06, "H_sd": 3.0, "dmin": 0.005,
                   "lvl_back": 0.28, "tau0_back": 0.05, "Hb_sd": 3.5}


def spec_for(m: int, eps: float, dt: float = 1e-3) -> fk.EnsembleSpec:
    return fk.EnsembleSpec(m=m, eps=eps, dt=dt)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(fk._jsonable(obj), indent=1, default=fk._json_default))
    os.replace(tmp, path)


def _drift(cp: dict, x: np.ndarray, z: np.ndarray, W: float, dt: float, s2rate: float,
           geo: dict, pp: dict = None) -> np.ndarray:
    """Drift increment b dt of one Z-aware component (see PROPOSALS).

    The escape stripes of a component are consecutive, j0..j1.  Before the exposure
    zone of j0 (Z > c_j0 + H): softened bridge to |x|_mi = lvl timed on the zone entry;
    while c_j1 - H <= Z <= c_j0 + H: h-transform of BM conditioned to stay in the
    out-of-contact band (a, W - a), h(x) = sin(pi (x - a)/(W - 2a)) (drift
    s2 h'/h, x clamped to [a + dmin, W - a - dmin]); afterwards optional bridge back
    to |x|_mi = lvl_back timed on the entry of Z into c_back + Hb.
    """
    pp = PROPOSAL_PARAMS if pp is None else pp
    c = geo["centres"]
    zbar, gam, a = geo["z_bar"], geo["gamma"], geo["a"]
    H = cp.get("H_sd", pp["H_sd"]) * geo["sd"]
    b = np.zeros_like(x)
    j0, j1 = cp["escape"][0], cp["escape"][-1]
    lowside = x < 0.5 * W
    before = z > c[j0] + H
    inz = (~before) & (z >= c[j1] - H)
    after = z < c[j1] - H
    if before.any():
        tau = np.log((z[before] - zbar) / (c[j0] + H - zbar)) / gam
        lv = cp.get("lvl", pp["lvl"])
        tgt = np.where(lowside[before], lv, W - lv)
        b[before] = (tgt - x[before]) * dt / (tau + pp["tau0"])
    if inz.any():
        Lb = W - 2.0 * a
        xc = np.clip(x[inz], a + pp["dmin"], W - a - pp["dmin"])
        k = math.pi / Lb
        bz = (s2rate * dt) * k / np.tan(k * (xc - a))
        if cp.get("free_zone", False):
            # no in-zone drift at all: the pair hovers near the band edge after the
            # pre-zone pull, so in-out switching during the passage (partial exposure of
            # either order) occurs at its natural rate
            bz = np.zeros_like(bz)
        elif cp.get("soft_zone", False):
            # no forcing while in contact inside the zone: late entries / re-entries into
            # the band are then generated at their natural rate (covers partial exposure)
            bz = np.where((x[inz] > a) & (x[inz] < W - a), bz, 0.0)
        b[inz] = bz
    if cp.get("back") is not None and after.any():
        jb = cp["back"]
        Hb = pp["Hb_sd"] * geo["sd"]
        sel = after & (z > c[jb] + Hb)
        if sel.any():
            tau = np.log((z[sel] - zbar) / (c[jb] + Hb - zbar)) / gam
            tb = np.where(lowside[sel], pp["lvl_back"], W - pp["lvl_back"])
            b[sel] = (tb - x[sel]) * dt / (tau + pp["tau0_back"])
    return b


def geometric_cut_times(spec: fk.EnsembleSpec) -> list:
    c = spec.centres()
    mids = 0.5 * (c[:-1] + c[1:])
    return [float(-math.log((x - spec.z_bar) / (spec.z0 - spec.z_bar)) / spec.gamma) for x in mids]


# ---------------------------------------------------------------------------
# Importance-sampled FK chunk
# ---------------------------------------------------------------------------


def _is_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    if spec.n_perp != 1 or spec.slab_shape != "gauss":
        raise NotImplementedError("d = 2 Gaussian stripes only")
    n = int(task["size"])
    steps = spec.steps()
    dt = spec.dt
    prop = task["proposal"]
    pis = np.asarray(prop["pi"], float)
    comps = prop["comps"]
    Bs = np.asarray(task["budgets"], float)
    w = np.asarray(task["w"], float)
    batch = int(task["batch"])
    nb = -(-n // batch)
    bid = np.arange(n) // batch
    cut_steps = np.asarray(task["cut_steps"], np.int64)      # basin j: steps s in [cut_j, cut_{j+1})
    basin_of_step = np.searchsorted(cut_steps, np.arange(steps), side="right") - 1
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()

    decay = 1.0 - spec.gamma * dt
    pull = spec.gamma * spec.z_bar * dt
    noise_z = spec.eps * math.sqrt(spec.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    s2 = noise_r * noise_r
    W = spec.torus_w
    a2 = spec.contact_a ** 2
    centres = spec.centres()
    sd = spec.sd()
    cutoff = spec.kernel_cutoff_sd * sd
    pref = dt / (math.sqrt(2.0 * math.pi) * sd * W ** spec.n_perp)
    inv2v = 1.0 / (2.0 * sd * sd)
    geo = {"centres": centres, "z_bar": spec.z_bar, "gamma": spec.gamma, "a": spec.contact_a,
           "sd": sd}

    z = spec.z0 + math.sqrt(spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)) \
        * rng.standard_normal(n)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    x = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0 * rng.standard_normal(n), W)
    comp = rng.choice(pis.size, size=n, p=pis)
    Kc = len(comps)
    L = np.zeros((Kc, n))
    lr = np.ones(n)
    X = np.zeros(n)
    nI = Bs.size
    nbas = cut_steps.size - 1
    f_sum = np.zeros((nI, steps))
    f_sq = np.zeros((nI, steps))
    f_b = np.zeros((nI, nb, steps))
    basin = np.zeros((n, nI, nbas))
    lr_at = {}
    check_steps = set(int(round(tc / dt)) for tc in task.get("lr_check_times", ()))
    gate_on = np.zeros(steps)
    for s in range(steps):
        t = s * dt
        noise = rng.standard_normal((3, n))
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        dlt = noise_r * noise[2]
        bks = [_drift(cp, x, z, W, dt, s2 / dt, geo) for cp in comps]
        used = np.zeros(n)
        for k, b in enumerate(bks):
            sel = comp == k + 1
            used[sel] = b[sel]
        dlt = dlt + used
        for k, b in enumerate(bks):
            L[k] += (2.0 * b * dlt - b * b) / (2.0 * s2)
        if True:
            with np.errstate(over="ignore"):
                den = pis[0] + np.sum(pis[1:, None] * np.exp(L), axis=0)
            lr = 1.0 / den
        if s == int(task["cut_steps"][-2]) - 1:
            X_cut = X.copy()
        if s in check_steps:
            lr_at[f"{t:.4f}"] = [float(lr.sum()), float((lr * lr).sum())]
        x += dlt
        np.mod(x, W, out=x)
        perp = np.minimum(x, W - x)
        gate = (r_par * r_par + perp * perp) < a2
        gate_on[s] += float(np.sum(lr * gate))
        dmin = np.min(np.abs(z[:, None] - centres[None, :]), axis=1)
        act = np.flatnonzero(gate & (dmin < cutoff))
        if act.size == 0:
            continue
        diff = z[act, None] - centres[None, :]
        ph_ = np.exp(-(diff * diff) * inv2v) * pref
        e = ph_[:, 0] * w[0]
        for jj in range(1, w.size):
            e = e + ph_[:, jj] * w[jj]
        la = lr[act]
        Xa = X[act]
        j = basin_of_step[s]
        for i, B in enumerate(Bs):
            y = la * (-np.expm1(-B * e)) * np.exp(-B * Xa)
            f_sum[i, s] += y.sum()
            f_sq[i, s] += (y * y).sum()
            f_b[i, :, s] += np.bincount(bid[act], weights=y, minlength=nb)
            if j >= 0:
                basin[act, i, j] += y
        X[act] += e
    comp_counts = np.bincount(comp, minlength=pis.size)
    return {"path_comp": comp.astype(np.int8), "path_lr": lr, "path_X": X, "path_X_cut": X_cut,
            "chunk": int(task["chunk"]), "size": n, "f_sum": f_sum, "f_sq": f_sq, "f_b": f_b,
            "basin": basin, "lr_end": [float(lr.sum()), float((lr * lr).sum())],
            "lr_at": lr_at, "gate_on": gate_on, "comp_counts": comp_counts,
            "runtime_seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Tasks, pilot, simulation
# ---------------------------------------------------------------------------


def run_name(ci: int, dt: float, proposal: str, tag: int) -> str:
    c = CELLS[ci]
    return f"n12_m{c['m']}_eps{c['eps']:g}_dt{dt:g}_prop{proposal}_tag{tag}"


def make_tasks(ci: int, n_paths: int, *, dt: float, proposal: str, tag: int,
               chunk: int = CHUNK, batch: int | None = None) -> tuple:
    c = CELLS[ci]
    m, eps = c["m"], c["eps"]
    spec = spec_for(m, eps, dt)
    cuts = geometric_cut_times(spec)
    T = fk.step_times(dt, spec.steps())
    # basin j = kill steps s with kill time (s+1) dt in [cut_{j-1}, cut_j); first basin from 0
    cut_steps = [0] + [int(np.searchsorted(T, x, side="left")) for x in cuts] + [spec.steps()]
    if batch is None:                      # production: 50 chunks = 50 batches
        batch = max(1, n_paths // 50)
        chunk = batch
    nchunks = -(-n_paths // chunk)
    if chunk % batch:
        raise ValueError("chunk must be a multiple of the batch size")
    q = lambda x, s=1e9: int(round(float(x) * s)) % (1 << 63)  # noqa: E731
    prop = PROPOSALS[m][proposal]
    ent = [SEED, tag, m, q(eps), q(dt, 1e15), sum(ord(ch) for ch in proposal), int(n_paths), int(chunk)]
    children = np.random.SeedSequence(ent).spawn(nchunks)
    tasks = []
    for i in range(nchunks):
        size = min(chunk, n_paths - i * chunk)
        tasks.append({"spec": spec.to_dict(), "size": size, "seedseq": children[i], "chunk": i,
                      "proposal": prop, "budgets": list(c["B"]), "w": [1.0 / m] * m,
                      "batch": batch, "cut_steps": cut_steps,
                      "lr_check_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]})
    meta = {"cell": c, "m": m, "eps": eps, "dt": dt, "proposal_name": proposal, "proposal": prop,
            "proposal_params": PROPOSAL_PARAMS,
            "tag": tag, "seed": SEED, "seed_entropy": ent, "n_paths": n_paths, "chunk": chunk,
            "batch": batch, "geometric_cuts": cuts, "cut_steps": cut_steps,
            "budgets": list(c["B"]), "w": [1.0 / m] * m, "spec": spec.to_dict(),
            "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]"}
    return tasks, meta


def merge_chunks(parts: list, meta: dict) -> dict:
    """Merge chunk results into per-step estimates (p_step, iid SE, batch means)."""
    parts = sorted(parts, key=lambda r: r["chunk"])
    N = sum(r["size"] for r in parts)
    f_sum = sum(r["f_sum"] for r in parts)
    f_sq = sum(r["f_sq"] for r in parts)
    f_b = np.concatenate([r["f_b"] for r in parts], axis=1)
    bsz = meta["batch"]
    p = f_sum / N
    var = np.maximum(f_sq / N - p * p, 0.0)
    basin = np.concatenate([r["basin"] for r in parts], axis=0)          # (N, nI, nbas)
    lr_end = np.sum([r["lr_end"] for r in parts], axis=0)
    lr_at = {}
    for r in parts:
        for k, v in r["lr_at"].items():
            lr_at.setdefault(k, np.zeros(2))
            lr_at[k] += np.asarray(v)
    return {"N": N, "p_step": p, "se_iid": np.sqrt(var / N), "ess_step": np.divide(
        f_sum ** 2, f_sq, out=np.zeros_like(f_sum), where=f_sq > 0), "batch_p_step": f_b / bsz,
        "basin_paths": basin, "lr_end_mean": lr_end[0] / N,
        "lr_end_se": math.sqrt(max(lr_end[1] / N - (lr_end[0] / N) ** 2, 0.0) / N),
        "lr_at": {k: {"mean": v[0] / N, "se": math.sqrt(max(v[1] / N - (v[0] / N) ** 2, 0) / N)}
                  for k, v in lr_at.items()},
        "comp_counts": np.sum([r["comp_counts"] for r in parts], axis=0),
        "gate_on_weighted": sum(r["gate_on"] for r in parts) / N,
        "runtime_seconds_total": float(sum(r["runtime_seconds"] for r in parts))}


def basin_summary(res: dict, meta: dict) -> list:
    out = []
    V = res["basin_paths"]
    N = V.shape[0]
    for i, B in enumerate(meta["budgets"]):
        rows = []
        for j in range(V.shape[2]):
            col = V[:, i, j]
            tot = col.sum()
            kk = max(1, int(round(0.01 * N)))
            rows.append({"basin": j + 1, "mass": tot / N, "se_iid": float(col.std(ddof=1) / math.sqrt(N)),
                         "kish_ess": float(tot ** 2 / (col * col).sum()) if tot > 0 else 0.0,
                         "top1pct_share": float(np.partition(col, N - kk)[-kk:].sum() / tot) if tot > 0 else None,
                         "max_path_share": float(col.max() / tot) if tot > 0 else None})
        out.append({"B": B, "basins": rows})
    return out


def cmd_pilot(args) -> None:
    tasks, meta = make_tasks(args.cell, int(float(args.paths)), dt=args.dt, proposal=args.proposal,
                             tag=999, chunk=int(float(args.paths)), batch=int(float(args.paths)) // 10)
    t0 = time.time()
    r = _is_chunk(tasks[0])
    res = merge_chunks([r], meta)
    bs = basin_summary(res, meta)
    T = fk.step_times(meta["dt"], len(res["p_step"][0]))
    late = T >= meta["geometric_cuts"][-1]
    print(json.dumps({"wall": time.time() - t0, "lr_end": [res["lr_end_mean"], res["lr_end_se"]],
                      "lr_at": res["lr_at"], "comp_counts": res["comp_counts"].tolist(),
                      "basins": bs,
                      "late_step_ess_median": [float(np.median(res["ess_step"][i][late & (res["p_step"][i] > 1e-3 * res["p_step"][i][late].max())]))
                                               for i in range(len(meta["budgets"]))]},
                     default=fk._json_default, indent=0))


def _is_worker(task):
    return _is_chunk(task)


def cmd_simulate(args) -> None:
    import multiprocessing as mp
    n_paths = int(float(args.paths))
    tag = TAG_IS if abs(args.dt - 1e-3) < 1e-15 else TAG_IS_DT
    tasks, meta = make_tasks(args.cell, n_paths, dt=args.dt, proposal=args.proposal, tag=tag)
    name = run_name(args.cell, args.dt, args.proposal, tag)
    d = STORE / name
    d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (d / f"chunk{t['chunk']:04d}.npz").exists()]
    print(f"[n12] {name}: {len(todo)}/{len(tasks)} chunks to run", flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(processes=min(args.workers, MAX_WORKERS)) as pool:
        for r in pool.imap_unordered(_is_worker, todo):
            np.savez_compressed(d / f"chunk{r['chunk']:04d}.npz",
                                **{k: (np.asarray(v) if not isinstance(v, dict) else np.asarray(json.dumps(v)))
                                   for k, v in r.items()})
            print(f"[n12] chunk {r['chunk']} {r['runtime_seconds']:.0f}s", flush=True)
    meta["store"] = str(d)
    meta["wall_seconds_last_invocation"] = time.time() - t0
    write_json(OUT / "runs" / f"{name}.json", meta)


def load_run(name: str) -> tuple:
    meta = json.loads((OUT / "runs" / f"{name}.json").read_text())
    d = Path(meta["store"])
    parts = []
    for i in range(-(-meta["n_paths"] // meta["chunk"])):
        with np.load(d / f"chunk{i:04d}.npz", allow_pickle=False) as z:
            r = {k: z[k] for k in z.files}
        r["chunk"] = int(r["chunk"])
        r["size"] = int(r["size"])
        r["lr_at"] = json.loads(str(r["lr_at"]))
        r["runtime_seconds"] = float(r["runtime_seconds"])
        parts.append(r)
    return merge_chunks(parts, meta), meta


# ---------------------------------------------------------------------------
# Direct kill (independent check of the late basin at large N)
# ---------------------------------------------------------------------------


def _dk_worker(task):
    import exact_m_prr_upgrade_core as core
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    out = core.simulate_chunk_general(
        rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]), weights=tuple(task["w"]),
        centres_z=spec.centres(), dt=spec.dt, step_count=spec.steps(), p=spec.model(),
        n_perp=spec.n_perp)
    steps = np.rint(out["kill_times"] / spec.dt).astype(np.int64) - 1
    return {"chunk": task["chunk"], "hist_steps": np.bincount(steps, minlength=spec.steps())[: spec.steps()],
            "survivors": out["survivors"], "walker_steps": out["walker_steps"]}


def cmd_dk(args) -> None:
    import multiprocessing as mp
    ci, B = TARGET_CASES[args.case]
    c = CELLS[ci]
    m, eps = c["m"], c["eps"]
    spec = spec_for(m, eps, 1e-3)
    walkers = int(float(args.walkers))
    chunk = 250_000
    q = lambda x: int(round(float(x) * 1e9)) % (1 << 63)  # noqa: E731
    ent = [SEED, TAG_DK, m, q(eps), q(B), walkers, chunk]
    n = -(-walkers // chunk)
    kids = np.random.SeedSequence(ent).spawn(n)
    tasks = [{"spec": spec.to_dict(), "size": min(chunk, walkers - i * chunk), "seedseq": kids[i],
              "chunk": i, "B": B, "w": [1.0 / m] * m} for i in range(n)]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    hist = np.zeros(spec.steps(), np.int64)
    batches = []
    surv = 0
    wsteps = 0
    with ctx.Pool(processes=min(args.workers, MAX_WORKERS)) as pool:
        for r in pool.imap_unordered(_dk_worker, tasks):
            hist += r["hist_steps"]
            batches.append((r["chunk"], r["hist_steps"]))
            surv += r["survivors"]
            wsteps += r["walker_steps"]
    batches.sort(key=lambda x: x[0])
    # 20 batch histograms (contiguous chunk groups) on the step grid, sparse beyond the first cut
    G = 20
    grp = np.array_split(np.arange(n), G)
    bh = np.stack([np.sum([batches[i][1] for i in g], axis=0) for g in grp])
    payload = {"case": {"m": m, "eps": eps, "B": B}, "walkers": walkers, "chunk": chunk,
               "seed": SEED, "tag": TAG_DK, "seed_entropy": ent, "dt": spec.dt,
               "simulator": "exact_m_prr_upgrade_core.simulate_chunk_general (EM + end-of-step Doi kill)",
               "hist_steps": hist, "batch_hist_steps": bh,
               "batch_walkers": [int(sum(tasks[i]["size"] for i in g)) for g in grp],
               "survivors_at_tmax": surv, "walker_steps": wsteps, "wall_seconds": time.time() - t0}
    write_json(OUT / "directkill" / f"dk_m{m}_eps{eps:g}_B{B:g}_N{walkers:.0e}.json", payload)
    print(f"[n12] dk m{m} B{B:g}: {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def window_bins(p_step: np.ndarray, spec: fk.EnsembleSpec) -> tuple:
    g = fk.build_grid(spec, "production")
    cps = g["cps"]
    wi = g["window_index"]
    cs = np.concatenate([[0.0], np.cumsum(p_step)])
    pw = cs[cps[wi[1:]]] - cs[cps[wi[:-1]]]
    return pw, g["edges"][wi]


def protocol_last(p_step, spec, after, walkers=1_000_000):
    pw, ew = window_bins(p_step, spec)
    cls = fk.classify_expected(pw, ew, walkers=walkers, bandwidth=fk.base.DEFAULT_BANDWIDTH)
    rows = [r for r in cls["rows"] if r["time"] > after]
    best = max(rows, key=lambda r: r["relative_prominence"]) if rows else None
    return {"mode_count_1e6": int(cls["mode_count"]),
            "significant_times": [r["time"] for r in cls["rows"] if r["significant"]],
            "all_maxima": [{"time": r["time"], "rel_prom": r["relative_prominence"], "z": r["z"]}
                           for r in cls["rows"]],
            "last": ({"time": best["time"], "relative_prominence": best["relative_prominence"],
                      "z_at_1e6": best["z"]} if best else None)}


def late_peak_raw(t, p_step, lo, hi, k):
    """Maximum of the k-step binned density on (lo, hi) (no smoothing)."""
    nbin = p_step.size // k
    F = p_step[: nbin * k].reshape(nbin, k).sum(1) / (k * (t[1] - t[0]))
    tc = t[: nbin * k].reshape(nbin, k).mean(1)
    sel = (tc > lo) & (tc < hi)
    i = int(np.argmax(np.where(sel, F, -np.inf)))
    return float(tc[i]), float(F[i]), float(F.max())


CHOSEN = {2: "M4", 3: "M3"}  # pilot-selected, DK-validated (M, M2 for m = 2 were low: see OUT runs + notes)
PAPER_RESOLUTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)


def _jack(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    G = v.size
    return float(math.sqrt((G - 1) / G * np.sum((v - v.mean()) ** 2))) if G > 1 else None


def analyze_case(ci: int, B: float, dt: float = 1e-3, proposal: str | None = None) -> dict:
    import fb_n9a_matching_census as n9a
    c = CELLS[ci]
    m, eps = c["m"], c["eps"]
    proposal = proposal or CHOSEN[m]
    tag = TAG_IS if abs(dt - 1e-3) < 1e-15 else TAG_IS_DT
    name = run_name(ci, dt, proposal, tag)
    res, meta = load_run(name)
    i = [abs(b - B) < 1e-12 for b in meta["budgets"]].index(True)
    spec = spec_for(m, eps, dt)
    T = fk.step_times(dt, spec.steps())
    p = res["p_step"][i]
    pb = res["batch_p_step"][i]
    targets = spec.times()
    valleys = fk.g_valley_times(fk.EnsembleSpec(m=m, eps=eps), [1.0 / m] * m)
    cuts = meta["geometric_cuts"]
    census = {}
    for r in PAPER_RESOLUTIONS:
        k = max(1, int(math.floor(r * eps / dt + 1e-9)))
        cr = n9a.census_one(T, p, pb, dt, k, m, targets, eps, valleys)
        census[f"{r:g}"] = {kk: cr[kk] for kk in ("h_g", "k_steps", "n_max", "n_min", "changes",
                                                  "maxima_per_G_basin", "one_max_per_basin",
                                                  "first_sign", "last_sign", "alternating",
                                                  "exactly_m_pattern", "EXTRA_PAIR", "n_undecided",
                                                  "undecided_far_from_critical_points")}
    # coarse extension beyond the N9a resolution set (h = 2 eps, 4 eps): a slow decline
    # after the late maximum has no single first difference above |z| = 5 at h <= eps
    census_coarse = {}
    for r in (2.0, 4.0):
        k = max(1, int(math.floor(r * eps / dt + 1e-9)))
        cr = n9a.census_one(T, p, pb, dt, k, m, targets, eps, valleys)
        census_coarse[f"{r:g}"] = {kk: cr[kk] for kk in ("h_g", "k_steps", "n_max", "n_min", "changes",
                                                         "exactly_m_pattern", "EXTRA_PAIR", "n_undecided")}
    # protocol P and last-mode prominence with batch jackknife
    after = cuts[-1]
    prot = protocol_last(p, spec, after)
    N = res["N"]
    bsz = meta["batch"]
    reps = []
    tl = []
    for b in range(pb.shape[0]):
        pj = (N * p - bsz * pb[b]) / (N - bsz)
        r_ = protocol_last(pj, spec, after)["last"]
        reps.append(r_["relative_prominence"] if r_ else 0.0)
        tl.append(r_["time"] if r_ else None)
    last = prot["last"]
    zs = [abs(v["z"]) for v in prot["all_maxima"]]
    out = {"case": {"m": m, "eps": eps, "B": B, "dt": dt, "w": [1.0 / m] * m}, "run": name,
           "N_paths": N, "proposal": meta["proposal"], "seed_entropy": meta["seed_entropy"],
           "tag": meta["tag"], "lr_end": [res["lr_end_mean"], res["lr_end_se"]],
           "lr_at": res["lr_at"], "comp_counts": res["comp_counts"],
           "census": census,
           "census_exact_all_resolutions": all(v["exactly_m_pattern"] for v in census.values()),
           "census_extra_pair_any": any(v["EXTRA_PAIR"] for v in census.values()),
           "census_counts_by_resolution": {k: [v["n_max"], v["n_min"]] for k, v in census.items()},
           "census_coarse_extension": census_coarse,
           "census_coarse_counts": {k: [v["n_max"], v["n_min"]] for k, v in census_coarse.items()},
           "protocol": prot,
           "last_mode_rel_prom": last["relative_prominence"] if last else None,
           "last_mode_rel_prom_jack_se": _jack(reps),
           "last_mode_time_protocol": last["time"] if last else None,
           "last_mode_time_jack_se": _jack(tl),
           "last_mode_z_protocol_1e6": last["z_at_1e6"] if last else None}
    if last and last["z_at_1e6"] > 0:
        out["walkers_for_z5"] = 1e6 * (5.0 / last["z_at_1e6"]) ** 2
    # basins (geometric cuts), ESS
    bs = basin_summary(res, meta)[i]["basins"]
    out["basins_geometric"] = {"cuts": cuts, "rows": bs}
    late = T >= cuts[-1]
    pl = p[late]
    sel = pl > 1e-3 * pl.max()
    out["late_step_ess"] = {"median": float(np.median(res["ess_step"][i][late][sel])),
                            "min": float(np.min(res["ess_step"][i][late][sel])),
                            "n_steps": int(sel.sum())}
    out["mass_after_last_G_valley"] = float(p[(T > valleys[-1]) & (T <= 3.5)].sum())
    out["mass_after_last_G_valley_se_batch"] = float(
        np.std([(pb[b][(T > valleys[-1]) & (T <= 3.5)]).sum() for b in range(pb.shape[0])], ddof=1)
        / math.sqrt(pb.shape[0]))
    # bracketing certificate for an interior late maximum: at bin width h = r eps,
    # F(peak bin) - F(valley bin) and F(peak bin) - F(last window bin), paired batch SEs
    brk = {}
    for r in (0.5, 1.0):
        k = max(1, int(math.floor(r * eps / dt + 1e-9)))
        nbin = T.size // k
        tcb = T[: nbin * k].reshape(nbin, k).mean(1)
        Fk = p[: nbin * k].reshape(nbin, k).sum(1) / (k * dt)
        Fbk = pb[:, : nbin * k].reshape(pb.shape[0], nbin, k).sum(2) / (k * dt)
        late_sel = np.flatnonzero((tcb > cuts[-1]) & (tcb < 3.5))
        ipk = int(late_sel[np.argmax(Fk[late_sel])])
        prev = np.flatnonzero((tcb > targets[-2]) & (tcb < tcb[ipk]))
        ival = int(prev[np.argmin(Fk[prev])])
        iend = int(np.flatnonzero(tcb < 3.5)[-1])
        def zdiff(i1, i2):
            dd = Fbk[:, i1] - Fbk[:, i2]
            se_ = dd.std(ddof=1) / math.sqrt(dd.size)
            return {"diff": float(Fk[i1] - Fk[i2]), "se": float(se_),
                    "z": float((Fk[i1] - Fk[i2]) / se_) if se_ > 0 else None}
        brk[f"{r:g}"] = {"h": k * dt, "t_peak": float(tcb[ipk]), "F_peak": float(Fk[ipk]),
                         "t_valley": float(tcb[ival]), "F_valley": float(Fk[ival]),
                         "t_end": float(tcb[iend]), "F_end": float(Fk[iend]),
                         "peak_minus_valley": zdiff(ipk, ival), "peak_minus_end": zdiff(ipk, iend)}
    out["late_max_bracketing"] = brk
    # raw late peak at h = 0.25 eps
    k = max(1, int(math.floor(0.25 * eps / dt + 1e-9)))
    tpk, fpk, fmax = late_peak_raw(T, p, cuts[-1], 3.5, k)
    out["late_peak_raw_h0.25eps"] = {"t": tpk, "density": fpk, "global_max_density": fmax,
                                     "relative_height": fpk / fmax}
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exact_m_prr_upgrade_core as core
    core.apply_prr_style()
    S = json.loads((OUT / "n12_census_resolution.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), constrained_layout=True)
    for col, rec in enumerate(S["cases"]):
        c = rec["case"]
        m, eps, B = c["m"], c["eps"], c["B"]
        ci = [i for i, cc in enumerate(CELLS) if cc["m"] == m][0]
        res, meta = load_run(rec["run"])
        i = [abs(b - B) < 1e-12 for b in meta["budgets"]].index(True)
        spec = spec_for(m, eps, 1e-3)
        T = fk.step_times(1e-3, spec.steps())
        k = max(1, int(math.floor(0.25 * eps / 1e-3 + 1e-9)))
        nbin = T.size // k
        tc = T[: nbin * k].reshape(nbin, k).mean(1)
        F = res["p_step"][i][: nbin * k].reshape(nbin, k).sum(1) / (k * 1e-3)
        Fb = res["batch_p_step"][i][:, : nbin * k].reshape(-1, nbin, k).sum(2) / (k * 1e-3)
        se = Fb.std(0, ddof=1) / math.sqrt(Fb.shape[0])
        ax = axes[0, col]
        ok = F > 0
        ax.plot(tc[ok], F[ok], color="C0", lw=0.8, label="IS exact law")
        ref = None
        name = f"n9a_m{m}_eps{eps:g}_decl"
        if fk.index_path(name).exists():
            Dd = fk.load_ensemble(name).declared()
            idd = [d_["label"] for d_ in Dd["declared"]].index(f"full_B{B:g}")
            Fr = Dd["p_step"][idd][: nbin * k].reshape(nbin, k).sum(1) / (k * 1e-3)
            okr = Fr > 0
            ax.plot(tc[okr], Fr[okr], color="C1", lw=0.6, alpha=0.8, label="plain FK (N9a)")
        for dk in rec.get("direct_kill", []):
            dkj = json.loads((OUT / "directkill" / dk["file"]).read_text())
            h = np.asarray(dkj["hist_steps"], float)
            kk = 20
            nb2 = h.size // kk
            Hd = h[: nb2 * kk].reshape(nb2, kk).sum(1) / (dkj["walkers"] * kk * 1e-3)
            td = T[: nb2 * kk].reshape(nb2, kk).mean(1)
            okd = Hd > 0
            ax.plot(td[okd], Hd[okd], ".", ms=2.0, color="k", label=f"direct kill {dkj['walkers']:.0e}")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(1e-9, 50)
        ax.set_title(f"({'abc'[col]}) $m={m}$, $\\varepsilon={eps:g}$, $B={B:g}$", loc="left", fontsize=7)
        brk = rec["late_max_bracketing"]["1"]
        jb = int(np.argmin(np.abs(tc - brk["t_peak"])))
        ax.plot(brk["t_peak"], F[jb], marker="o", mfc="none", mec="C3", ms=5, lw=0,
                label="bracketed late max ($h=\\varepsilon$)")
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        cz = rec["census"]["0.25"]["changes"]
        for ch in cz:
            j = int(np.argmin(np.abs(tc - ch["t"])))
            if F[j] > 0:
                ax.plot(ch["t"], F[j], marker="v" if ch["kind"] == "max" else "^", ms=3.5,
                        color="C3" if ch["kind"] == "max" else "C2", lw=0)
        ax = axes[1, col]
        lo = rec["basins_geometric"]["cuts"][-1]
        sel = (tc > lo) & (tc < 3.5)
        ax.fill_between(tc[sel], (F - 2 * se)[sel], (F + 2 * se)[sel], color="C0", alpha=0.3, lw=0)
        ax.plot(tc[sel], F[sel], color="C0", lw=0.8, label="IS ($h=\\varepsilon/4$, $\\pm2$ SE)")
        for dk in rec.get("direct_kill", []):
            dkj = json.loads((OUT / "directkill" / dk["file"]).read_text())
            h = np.asarray(dkj["hist_steps"], float)
            kk = 50 if B < 40 or m == 2 else 100
            nb2 = h.size // kk
            cnt = h[: nb2 * kk].reshape(nb2, kk).sum(1)
            td = T[: nb2 * kk].reshape(nb2, kk).mean(1)
            sd_ = (td > lo) & (td < 3.5)
            nrm = dkj["walkers"] * kk * 1e-3
            ax.errorbar(td[sd_], cnt[sd_] / nrm, yerr=2 * np.sqrt(cnt[sd_]) / nrm, fmt="o", ms=1.8,
                        color="k", elinewidth=0.5, capsize=0,
                        label=f"direct kill ($h={kk * 1e-3:g}$, $\\pm2$ SE)")
        if col == 0:
            ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        for ch in cz:
            if ch["t"] > lo:
                ax.axvline(ch["t"], color="C3" if ch["kind"] == "max" else "C2", lw=0.6, ls="--")
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel("late-basin density (IS, $\\pm$2 SE)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ext in ("pdf", "png"):
        fig.savefig(fk.FIGURES / f"fb_n12_census_resolution.{ext}", dpi=200)
    print("[n12] figure written", flush=True)


def plain_fk_reference(m: int, eps: float, B: float, cuts) -> dict | None:
    """N9a plain-FK per-step law of the same case (tag 87, 5e5 paths): basin masses."""
    name = f"n9a_m{m}_eps{eps:g}_decl"
    if not fk.index_path(name).exists():
        return None
    Dd = fk.load_ensemble(name).declared()
    lab = f"full_B{B:g}"
    idd = [d["label"] for d in Dd["declared"]].index(lab)
    T = Dd["t"]
    p = Dd["p_step"][idd]
    pb = Dd["batch_p_step"][idd]
    edges = [0.0] + list(cuts) + [T[-1] + 1e-9]
    rows = []
    for j in range(len(edges) - 1):
        sel = (T >= edges[j]) & (T < edges[j + 1])
        mb = pb[:, sel].sum(1)
        rows.append({"basin": j + 1, "mass": float(p[sel].sum()),
                     "se_batch": float(mb.std(ddof=1) / math.sqrt(mb.size))})
    # per-step Kish ESS of the plain FK weights, N p^2 / (N se^2 + p^2), in the late basin
    N = Dd["N"]
    se = Dd["p_step_se_iid"][idd]
    ess = np.divide(N * p * p, N * se * se + p * p, out=np.zeros_like(p), where=(p > 0))
    late = T >= cuts[-1]
    pl = p[late]
    sel = pl > 1e-3 * pl.max()
    return {"ensemble": name, "N": N, "rows": rows,
            "late_step_ess": {"median": float(np.median(ess[late][sel])),
                              "min": float(np.min(ess[late][sel])), "n_steps": int(sel.sum())}}


def dk_reference(m, eps, B, cuts, spec) -> list:
    outs = []
    d = OUT / "directkill"
    if not d.exists():
        return outs
    for f in sorted(d.glob(f"dk_m{m}_eps{eps:g}_B{B:g}_N*.json")):
        dk = json.loads(f.read_text())
        N = dk["walkers"]
        h = np.asarray(dk["hist_steps"], float)
        bh = np.asarray(dk["batch_hist_steps"], float)
        bw = np.asarray(dk["batch_walkers"], float)
        T = fk.step_times(spec.dt, spec.steps())
        edges = [0.0] + list(cuts) + [T[-1] + 1e-9]
        rows = []
        for j in range(len(edges) - 1):
            sel = (T >= edges[j]) & (T < edges[j + 1])
            M = h[sel].sum() / N
            rows.append({"basin": j + 1, "count": int(h[sel].sum()), "mass": M,
                         "se": math.sqrt(M * (1 - M) / N)})
        # late-basin peak time of the protocol-smoothed histogram, batch jackknife
        pw, ew = window_bins(h / N, spec)
        after = cuts[-1]
        def lt(pw_):
            s = fk.classify_expected(pw_, ew, walkers=N)["smoothed"]
            tc = 0.5 * (ew[:-1] + ew[1:])
            sel = tc > after
            return float(tc[sel][np.argmax(s[sel])])
        tl = lt(pw)
        reps = [lt(window_bins((h - bh[g]) / (N - bw[g]), spec)[0]) for g in range(bh.shape[0])]
        outs.append({"file": f.name, "walkers": N, "tag": dk["tag"], "seed_entropy": dk["seed_entropy"],
                     "basins": rows, "late_smoothed_peak_time": tl, "late_peak_time_jack_se": _jack(reps),
                     "survivors_at_tmax": dk["survivors_at_tmax"], "wall_seconds": dk["wall_seconds"]})
    return outs


def dk_extras(rec: dict, dk_rec: dict, spec: fk.EnsembleSpec) -> dict:
    """IS-independent evidence for the late maximum from the direct-kill histogram.

    Bracketing: at bin width h the late maximum of the binned law is bracketed when
    F(peak bin) > F(valley bin) and F(peak bin) > F(end bin).  To avoid selection bias
    the peak and valley bins are the ones chosen by the IS law (an independent
    ensemble); the end bin is the last bin before t = 3.5.  Counts are multinomial,
    so Var(n1 - n2) ~= n1 + n2 and z = (n1 - n2)/sqrt(n1 + n2).
    Protocol P on the DK law (h/N), last-mode relative prominence, jackknife over the
    20 walker batches.
    """
    dk = json.loads((OUT / "directkill" / dk_rec["file"]).read_text())
    N = dk["walkers"]
    h = np.asarray(dk["hist_steps"], float)
    bh = np.asarray(dk["batch_hist_steps"], float)
    bw = np.asarray(dk["batch_walkers"], float)
    dt = spec.dt
    T = fk.step_times(dt, spec.steps())
    eps = spec.eps
    out = {"bracketing": {}}
    for r in (0.5, 1.0, 2.0, 4.0):
        k = max(1, int(math.floor(r * eps / dt + 1e-9)))
        nbin = T.size // k
        tcb = T[: nbin * k].reshape(nbin, k).mean(1)
        cnt = h[: nbin * k].reshape(nbin, k).sum(1)
        ref = rec["late_max_bracketing"].get(f"{r:g}") or rec["late_max_bracketing"]["1"]
        ipk = int(np.argmin(np.abs(tcb - ref["t_peak"])))
        ival = int(np.argmin(np.abs(tcb - ref["t_valley"])))
        iend = int(np.flatnonzero(tcb < 3.5)[-1])
        def zc(i1, i2):
            n1, n2 = cnt[i1], cnt[i2]
            return {"n1": int(n1), "n2": int(n2),
                    "z": float((n1 - n2) / math.sqrt(n1 + n2)) if n1 + n2 > 0 else None}
        out["bracketing"][f"{r:g}"] = {
            "h": k * dt, "t_peak_bin": float(tcb[ipk]), "t_valley_bin": float(tcb[ival]),
            "t_end_bin": float(tcb[iend]), "density_peak": float(cnt[ipk] / (N * k * dt)),
            "peak_minus_valley": zc(ipk, ival), "peak_minus_end": zc(ipk, iend)}
    # census of the DK law itself (20 walker batches), IS-independent
    import fb_n9a_matching_census as n9a
    m = rec["case"]["m"]
    valleys = fk.g_valley_times(fk.EnsembleSpec(m=m, eps=eps), [1.0 / m] * m)
    pbd = bh / bw[:, None]
    out["dk_census"] = {}
    for r in (0.5, 1.0, 2.0, 4.0):
        k = max(1, int(math.floor(r * eps / dt + 1e-9)))
        cr = n9a.census_one(T, h / N, pbd, dt, k, m, spec.times(), eps, valleys)
        out["dk_census"][f"{r:g}"] = {"counts": [cr["n_max"], cr["n_min"]],
                                      "exactly_m_pattern": cr["exactly_m_pattern"],
                                      "EXTRA_PAIR": cr["EXTRA_PAIR"],
                                      "changes": [(c["kind"], c["t"]) for c in cr["changes"]]}
    after = rec["basins_geometric"]["cuts"][-1]
    prot = protocol_last(h / N, spec, after)
    reps = []
    for g in range(bh.shape[0]):
        rr = protocol_last((h - bh[g]) / (N - bw[g]), spec, after)["last"]
        reps.append(rr["relative_prominence"] if rr else 0.0)
    last = prot["last"]
    out["protocol_on_dk_law"] = {
        "mode_count_1e6": prot["mode_count_1e6"],
        "last_time": last["time"] if last else None,
        "last_rel_prom": last["relative_prominence"] if last else None,
        "last_rel_prom_jack_se": _jack(reps)}
    return out


def regime(rec: dict) -> dict:
    """Where the census cells sit relative to the eps -> 0 asymptotics (Theorems 2-3).

    lambda_j, stick-breaking (Theorem 2 limit) masses, the frozen-gate law of the
    stored n1 ensemble (contact indicators chi_j = 1{|R(t_j)| < a}, 5e5 paths, tag 81),
    and a leading-order large-deviation estimate of the crossover eps_x(B) below which
    the stick-breaking late mass exceeds the gate-escape mass: escaping the passages
    S (out of contact from t_min(S) on) costs ~ a_eff^2 / (2 sigma^2(t_min)) with
    sigma^2(t) = eps^2 (sigma_perp0^2 + 4 d0 t), a_eff^2 = a^2 - (r_par0 e^{-gamma t})^2,
    and gains lambda_S = sum_{i in S} lambda_i; eps_x = min_S a_eff/sqrt(2 lambda_S s^2).
    Estimate only (no prefactors, no in-passage gate switching).
    """
    c = rec["case"]
    m, eps, B = c["m"], c["eps"], c["B"]
    spec = fk.EnsembleSpec(m=m, eps=eps)
    w = [1.0 / m] * m
    lam, sb = fk.limit_masses(B, w, spec)
    out = {"lambda_j": lam, "stick_breaking_masses": sb}
    name = f"n1_m{m}_eps{eps:g}"
    try:
        ens = fk.load_ensemble(name)
        d2 = np.concatenate([d for _, _, d in ens.iter_chunks("full")], axis=0)
        chi = (d2 < spec.contact_a ** 2).astype(float)
        _, fg = fk.limit_masses(B, w, spec, chi=chi)
        pats = {}
        ex = chi * lam[None, :]
        cum = np.concatenate([np.zeros((ex.shape[0], 1)), np.cumsum(ex, axis=1)], axis=1)
        late = np.exp(-cum[:, m - 1]) * (1.0 - np.exp(-ex[:, m - 1]))
        for row in np.unique(chi[:, : m - 1], axis=0):
            sel = np.all(chi[:, : m - 1] == row, axis=1)
            pats["".join(str(int(v)) for v in row)] = {"n_paths": int(sel.sum()),
                                                       "late_mass_share": float(late[sel].sum() / late.sum())}
        out.update({"frozen_gate_masses": fg, "frozen_gate_ensemble": name, "N": int(chi.shape[0]),
                    "P_chi_zero": (1.0 - chi.mean(0)).tolist(),
                    "late_mass_by_early_contact_pattern": pats})
    except Exception as exc:  # pragma: no cover
        out["frozen_gate_error"] = repr(exc)
    tj = np.asarray(spec.times())
    s2 = spec.sigma_perp0 ** 2 + 4.0 * spec.d0 * tj
    aeff = np.sqrt(spec.contact_a ** 2 - (spec.r_par0 * np.exp(-spec.gamma * tj)) ** 2)
    best = None
    rows = []
    import itertools
    for size in range(1, m):
        for S in itertools.combinations(range(m - 1), size):
            i0 = min(S)
            lamS = float(sum(lam[i] for i in S))
            ex_ = float(aeff[i0] / math.sqrt(2.0 * lamS * s2[i0]))
            rows.append({"S": [i + 1 for i in S], "lambda_S": lamS, "eps_x": ex_})
            best = ex_ if best is None else min(best, ex_)
    out["crossover_estimate"] = {"rows": rows, "eps_x": best, "eps": eps,
                                 "eps_above_crossover": bool(eps > best)}
    return out



# ---------------------------------------------------------------------------
# Proposal validation for m = 2 (why M and M2 were superseded by M4)
# ---------------------------------------------------------------------------

TAG_DIAG = 94
BX_EDGES = [0.0, 1e-3, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]


def _plain_chunk(task: dict) -> dict:
    """Untilted m = 2, eps = 0.05 chain up to the geometric cut (same kernel and draw order
    as _is_chunk with no tilt); features of paths with passage-1 survival e^{-B X} > 1e-9."""
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    n = int(task["n"])
    dt = spec.dt
    B = float(task["B"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    decay = 1.0 - spec.gamma * dt
    pull = spec.gamma * spec.z_bar * dt
    noise_z = spec.eps * math.sqrt(spec.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    W = spec.torus_w
    a2 = spec.contact_a ** 2
    c = spec.centres()
    sd = spec.sd()
    pref = dt / (math.sqrt(2.0 * math.pi) * sd * W)
    inv2v = 1.0 / (2.0 * sd * sd)
    w = np.asarray(task["w"], float)
    z = spec.z0 + math.sqrt(spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)) \
        * rng.standard_normal(n)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    x = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0 * rng.standard_normal(n), W)
    X = np.zeros(n)
    perp_t1 = np.zeros(n)
    rpar_t1 = np.zeros(n)
    t1_step = int(round(spec.times()[0] / dt))
    for s_ in range(int(task["cut_step"])):
        noise = rng.standard_normal((3, n))
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        x += noise_r * noise[2]
        np.mod(x, W, out=x)
        perp = np.minimum(x, W - x)
        gate = (r_par * r_par + perp * perp) < a2
        act = np.flatnonzero(np.abs(z - c[0]) < 6.0 * sd)
        if act.size:
            diff = z[act, None] - c[None, :]
            ph_ = np.exp(-(diff * diff) * inv2v) * pref
            e = ph_[:, 0] * w[0] + ph_[:, 1] * w[1]
            X[act] += e * gate[act]
        if s_ + 1 == t1_step:
            perp_t1[:] = perp
            rpar_t1[:] = r_par
    sv = np.exp(-B * X)
    keep = sv > 1e-9
    return {"chunk": int(task["chunk"]), "n": n, "sum_sv": float(sv.sum()),
            "sum_sv2": float((sv * sv).sum()), "X": X[keep], "perp_t1": perp_t1[keep],
            "rpar_t1": rpar_t1[keep]}


def cmd_diag_plain(args) -> None:
    """Plain-FK passage-1 survival of (2, 0.05, 50) with 1e7 paths (tag 94), by exposure class."""
    import multiprocessing as mp
    ci, B = TARGET_CASES[0]
    spec = spec_for(2, 0.05, 1e-3)
    N = int(float(args.paths))
    CH = 250_000
    T = fk.step_times(1e-3, spec.steps())
    cut_step = int(np.searchsorted(T, geometric_cut_times(spec)[-1], side="left"))
    ent = [SEED, TAG_DIAG, 2, 50000000, N, CH]
    kids = np.random.SeedSequence(ent).spawn(-(-N // CH))
    tasks = [{"spec": spec.to_dict(), "n": min(CH, N - i * CH), "seedseq": kids[i], "chunk": i,
              "cut_step": cut_step, "B": B, "w": [0.5, 0.5]} for i in range(len(kids))]
    d = STORE / f"n12_plain_m2_eps0.05_tag{TAG_DIAG}_N{N:.0e}"
    d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (d / f"c{t['chunk']:04d}.npz").exists()]
    if todo:
        with mp.get_context("spawn").Pool(min(args.workers, MAX_WORKERS)) as pool:
            for r in pool.imap_unordered(_plain_chunk, todo):
                np.savez_compressed(d / f"c{r['chunk']:04d}.npz", **{k: np.asarray(v) for k, v in r.items()})
    fs = [np.load(d / f"c{t['chunk']:04d}.npz") for t in tasks]
    S = sum(float(f["sum_sv"]) for f in fs)
    S2 = sum(float(f["sum_sv2"]) for f in fs)
    X = np.concatenate([f["X"] for f in fs])
    p1 = np.concatenate([f["perp_t1"] for f in fs])
    r1 = np.concatenate([f["rpar_t1"] for f in fs])
    sv = np.exp(-B * X)
    k = np.digitize(B * X, BX_EDGES) - 1
    classes = [{"BX": [BX_EDGES[i], BX_EDGES[i + 1]], "mass": float(sv[k == i].sum() / N),
                "se": float(math.sqrt((sv[k == i] ** 2).sum()) / N), "paths": int((k == i).sum())}
               for i in range(len(BX_EDGES) - 1)]
    a = spec.contact_a
    out = {"case": {"m": 2, "eps": 0.05, "B": B}, "N": N, "seed": SEED, "tag": TAG_DIAG,
           "seed_entropy": ent, "cut_step": cut_step,
           "passage1_survival": [S / N, math.sqrt(max(S2 / N - (S / N) ** 2, 0.0) / N)],
           "kish_ess": S * S / S2, "classes_by_BX_cut": classes,
           "share_perp_t1_below_a": float(sv[p1 < a].sum() / S),
           "share_perp_t1_below_a_and_abs_rpar_gt_0.1": float(sv[(p1 < a) & (np.abs(r1) > 0.1)].sum() / S),
           "store": str(d)}
    write_json(OUT / "diagnostics" / f"plain_passage1_m2_tag{TAG_DIAG}.json", out)
    print(f"[n12] plain passage-1 survival {out['passage1_survival']}", flush=True)


def proposal_validation_m2() -> dict:
    """Late basin and passage-1 survival E_q[LR e^{-B X_cut}] of every m = 2 IS run vs DK/plain."""
    ci, B = TARGET_CASES[0]
    spec = spec_for(2, 0.05, 1e-3)
    dkf = sorted((OUT / "directkill").glob("dk_m2_eps0.05_B50_N*.json"))
    dk = json.loads(dkf[0].read_text()) if dkf else None
    out = {"direct_kill": None, "runs": {}}
    if dk:
        N = dk["walkers"]
        h = np.asarray(dk["hist_steps"], float)
        T = fk.step_times(1e-3, spec.steps())
        late = h[T >= geometric_cut_times(spec)[-1]].sum()
        s1 = late + dk["survivors_at_tmax"]
        out["direct_kill"] = {"file": dkf[0].name, "walkers": N, "late_count": int(late),
                              "late_mass": [late / N, math.sqrt(late) / N],
                              "passage1_survival": [s1 / N, math.sqrt(s1) / N]}
    pf = OUT / "diagnostics" / f"plain_passage1_m2_tag{TAG_DIAG}.json"
    if pf.exists():
        out["plain"] = json.loads(pf.read_text())
    for prop in ("M", "M2", "M4"):
        name = run_name(ci, 1e-3, prop, TAG_IS)
        if not (OUT / "runs" / f"{name}.json").exists():
            continue
        res, meta = load_run(name)
        d = Path(meta["store"])
        ys = []
        for i in range(-(-meta["n_paths"] // meta["chunk"])):
            with np.load(d / f"chunk{i:04d}.npz") as z_:
                ys.append((z_["path_lr"] * np.exp(-B * z_["path_X_cut"]), z_["path_X_cut"]))
        y = np.concatenate([a for a, _ in ys])
        Xc = np.concatenate([b for _, b in ys])
        Np = y.size
        k = np.digitize(B * Xc, BX_EDGES) - 1
        late = basin_summary(res, meta)[0]["basins"][-1]
        rec = {"N": Np, "late_mass": [late["mass"], late["se_iid"]], "late_kish": late["kish_ess"],
               "late_max_path_share": late["max_path_share"],
               "passage1_survival": [float(y.mean()), float(y.std() / math.sqrt(Np))],
               "passage1_kish": float(y.sum() ** 2 / (y * y).sum()),
               "classes_by_BX_cut": [{"BX": [BX_EDGES[i], BX_EDGES[i + 1]],
                                      "mass": float(y[k == i].sum() / Np),
                                      "se": float(math.sqrt((y[k == i] ** 2).sum()) / Np)}
                                     for i in range(len(BX_EDGES) - 1)]}
        if out["direct_kill"]:
            a_, b_ = rec["late_mass"], out["direct_kill"]["late_mass"]
            rec["z_late_vs_DK"] = (a_[0] - b_[0]) / math.sqrt(a_[1] ** 2 + b_[1] ** 2)
            a_, b_ = rec["passage1_survival"], out["direct_kill"]["passage1_survival"]
            rec["z_passage1_vs_DK"] = (a_[0] - b_[0]) / math.sqrt(a_[1] ** 2 + b_[1] ** 2)
        out["runs"][prop] = rec
    return out


def cmd_analyze(args) -> None:
    summary = {"item": "N12 census resolution of the N9a cells (2,0.05,50), (3,0.05,20), (3,0.05,50)",
               "driver": "code/fb_n12_census_resolution.py", "seed": SEED,
               "tags": {"IS_dt1e-3": TAG_IS, "DK": TAG_DK, "IS_dt5e-4": TAG_IS_DT},
               "resolutions_h_over_eps": list(PAPER_RESOLUTIONS), "zthr": 5.0, "cases": []}
    for (ci, B) in TARGET_CASES:
        c = CELLS[ci]
        m, eps = c["m"], c["eps"]
        rec = analyze_case(ci, B)
        spec = spec_for(m, eps, 1e-3)
        cuts = rec["basins_geometric"]["cuts"]
        ref = plain_fk_reference(m, eps, B, cuts)
        if ref:
            zz = []
            for a, b in zip(rec["basins_geometric"]["rows"], ref["rows"]):
                s = math.sqrt(a["se_iid"] ** 2 + b["se_batch"] ** 2)
                zz.append((a["mass"] - b["mass"]) / s if s > 0 else None)
            ref["z_IS_minus_plainFK"] = zz
        rec["plain_fk_reference"] = ref
        rec["direct_kill"] = dk_reference(m, eps, B, cuts, spec)
        for dk in rec["direct_kill"]:
            dk.update(dk_extras(rec, dk, spec))
            dk["z_IS_minus_DK"] = [((a["mass"] - b["mass"]) / math.sqrt(a["se_iid"] ** 2 + b["se"] ** 2)
                                    if (a["se_iid"] ** 2 + b["se"] ** 2) > 0 else None)
                                   for a, b in zip(rec["basins_geometric"]["rows"], dk["basins"])]
        # dt = 5e-4 check if present
        name_dt = run_name(ci, 5e-4, CHOSEN[m], TAG_IS_DT)
        if (OUT / "runs" / f"{name_dt}.json").exists():
            r2 = analyze_case(ci, B, dt=5e-4)
            rec["dt_half_check"] = {k: r2[k] for k in (
                "N_paths", "census_counts_by_resolution", "census_exact_all_resolutions",
                "census_extra_pair_any", "last_mode_rel_prom", "last_mode_rel_prom_jack_se",
                "last_mode_time_protocol", "last_mode_time_jack_se", "basins_geometric",
                "late_peak_raw_h0.25eps", "lr_end")}
        rec["regime"] = regime(rec)
        summary["cases"].append(rec)
        print(f"[n12] analyzed m{m} B{B:g}: census {rec['census_counts_by_resolution']}", flush=True)
    summary["proposal_validation_m2"] = proposal_validation_m2()
    summary["chosen_proposals"] = CHOSEN
    write_json(OUT / "n12_census_resolution.json", summary)


def main(argv=None) -> int:
    cmds = {"pilot": cmd_pilot, "simulate": cmd_simulate, "dk": cmd_dk, "analyze": cmd_analyze,
            "diag-plain": cmd_diag_plain}
    if "cmd_figure" in globals():
        cmds["figure"] = globals()["cmd_figure"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=sorted(cmds))
    ap.add_argument("--cell", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--paths", default="1e6")
    ap.add_argument("--walkers", default="3e7")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--proposal", default=None)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args(argv)
    if args.proposal is None:
        args.proposal = CHOSEN[CELLS[args.cell]["m"]]
    cmds[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
