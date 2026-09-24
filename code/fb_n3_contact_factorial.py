#!/usr/bin/env python3
"""N3: contact and field-location factorial on the exact law (fixed-budget campaign).

Item N3 of notes/gap_diagnosis_20260923/GAP_CLOSURE_PLAN.md (closes G02/G03,
tests TH-7).  Every law reported here is the exact reaction-time law of the
discretised production process (Euler--Maruyama, dt = 1e-3, end-of-step Doi
killing), computed with the N0 Feynman--Kac / Rao--Blackwell estimator
``exact_m_prr_fk_exact_law`` (imported as ``fk``):

    P(T_R in (c_{k-1}, c_k]) = E[ exp(-B X(c_{k-1})) (1 - exp(-B (X(c_k) - X(c_{k-1})))) ],

with X the unit-budget exposure of the UNKILLED path, so one path set serves
every kernel variant and every budget (common random numbers: every contrast
between variants is paired).

Arms
----
* anchors (m, eps) in {(2, 0.1), (3, 0.1)}: 14 kernel variants on one path set,
  field argument {pair midpoint Z; particle 1, x1 = Z + R_par/2} x gate
  {contact |R|_mi < a for a in {0.4 (production), 0.2, 0.15}; deterministic
  mean contact c(t) = P(|R_t|_mi < a) for the same radii; no gate}, budgets
  B in {1, 4, 8, 20} plus the large-B remark arm B in {1e2, 1e3, 1e4};
  equal weights; 1e6 paths per anchor (20 chunks of 5e4, built in 5
  invocations of 4 chunks).
* rpar arm: Fable's initial separation r_par0 = 0.35 at a = 0.4 (4 variants).
* tangent arm (frozen gate, TH-7): boundary-tangent relative start
  r_par0 = 0, r_perp0 = a = 0.4, m = 2, eps in {0.05, 0.025} (+ 0.0125 as an
  extension), B in {0.5, 1, 2, 4, 8, 20, 50, 200}; N0 declared mode (per-step
  law, step-resolution derivative census).  The mean-contact variant uses the
  exact contact probability of the Euler--Maruyama relative chain
  (fk.contact_probability_em), because the continuous-time quadrature of
  fk.mean_contact_curve assumes r_perp0 = 0.
* large-B dt check: production (2, 0.1) at dt in {5e-4, 2.5e-4}, 2e5 paths.
* direct-kill confirmations (1e6 walkers, the frozen production simulator
  kernel of exact_m_prr_upgrade_w6_single_particle.simulate_chunk_single_particle):
  (2, 0.1, B=1) midpoint field at a = 0.2, and (3, 0.1, B=4) particle-1 field
  at a = 0.4.

Streaming reduction (why not fk stored/declared mode for the anchors)
--------------------------------------------------------------------
Storing 14 variants x 1e6 paths in fk "stored" mode needs ~10-20 GB (more than
the free disk), and fk "declared" mode evaluates 98 (variant, B) items at every
one of the 4000 steps.  ``simulate_reduced`` therefore runs the SAME path
simulation as fk._run_chunk (same SeedSequence entropy fk.path_entropy, same
Philox chunk streams, same draw order, same kernel helpers fk._slab_components,
fk.parse_variant, fk.mean_contact_curve, same production checkpoint grid
fk.build_grid) and reduces each chunk in memory at the checkpoints: per
(variant, B) the checkpoint-interval masses in the expm1 form above, their
per-path squares and 5000-path batch sums, E X and E X^2 at the checkpoints
(mean field), and the joint contact pattern (chi_1..chi_m) at the target
times for every radius (frozen-gate surrogate).  ``selftest`` checks the
reduction against fk stored mode on common paths (max relative difference of
bin masses, float32 storage limited).

Seeds: base 20260923, tag 82 (fk.TAGS["N3"]); direct kill uses the entropy
[20260923, 82, 7001, m, eps, B, a, particle].  Every entropy is recorded.

Outputs
-------
artifacts/data/exact_m_fixed_budget/N3/ensembles/<name>.json   (index, seeds)
artifacts/data/exact_m_fixed_budget/N3/dk_<cell>.json          (direct kill)
artifacts/data/exact_m_fixed_budget/N3/n3_summary.json         (analysis)
artifacts/figures/fb_n3_contact_factorial.{pdf,png}
Large chunk reductions: ~/.local-build/prr_fk_ensembles/n3_reduced/<name>/

Usage (from code/)
------------------
    python3 fb_n3_contact_factorial.py selftest
    python3 fb_n3_contact_factorial.py anchor --m 2 --part 0      # parts 0..4
    python3 fb_n3_contact_factorial.py rpar --m 2 --part 0
    python3 fb_n3_contact_factorial.py tangent --eps 0.05
    python3 fb_n3_contact_factorial.py dtcheck --dt 2.5e-4
    python3 fb_n3_contact_factorial.py dk --cell a0.2
    python3 fb_n3_contact_factorial.py analyze
    python3 fb_n3_contact_factorial.py figure
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402
import validate_exact_m_offlattice as base  # noqa: E402

OUT = fk.FB_DATA / "N3"
ENS_INDEX = OUT / "ensembles"
ENS_ROOT = fk.ENSEMBLE_ROOT / "n3_reduced"
TAG = fk.TAGS["N3"]
SEED = fk.BASE_SEED
CHUNK = 50_000
BATCH = 5_000
MAX_WORKERS = 3

ANCHOR_BUDGETS = (1.0, 4.0, 8.0, 20.0)
LARGE_BUDGETS = (100.0, 1000.0, 10000.0)
RADII = (0.4, 0.2, 0.15)
ANCHOR_PATHS = 1_000_000
ANCHOR_PARTS = 5
TANGENT_BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 50.0, 200.0)
TANGENT_SUB_BUDGETS = (1.0, 8.0, 20.0, 50.0, 200.0)   # mean-contact / no-gate items
TANGENT_PATHS = 200_000
DT_CHECK_PATHS = 200_000
DK_WALKERS = 1_000_000
DK_CHUNK = 100_000

DK_CELLS = {
    # name: (m, eps, B, contact_a, particle)   particle 0 = pair midpoint
    "a0.2": (2, 0.1, 1.0, 0.2, 0),
    "p1_m3_B4": (3, 0.1, 4.0, 0.4, 1),
}


def factorial_variants() -> list[str]:
    out = []
    for field in ("mid", "p1"):
        for a in RADII:
            out.append(f"gate=contact,a={a:g},field={field}")
        for a in RADII:
            out.append(f"gate=mean,a={a:g},field={field}")
        out.append(f"gate=none,field={field}")
    return out


def vname(gate: str, a: float = 0.4, field: str = "mid") -> str:
    """Variant name as produced by fk.parse_variant for generic strings."""
    return f"g{gate}_a{a:g}_f{field}"


def equal_w(m: int) -> tuple:
    return tuple([1.0 / m] * m)


# ============================================================================
# Streaming reduction (same paths as fk._run_chunk)
# ============================================================================


def _reduce_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    n = int(task["size"])
    steps = spec.steps()
    dt = spec.dt
    variants = task["variants"]
    nv = len(variants)
    Bs = np.asarray(task["budgets"], float)
    nB = Bs.size
    w = np.asarray(task["w"], float)
    cps = np.asarray(task["cps"], np.int64)
    K = cps.size
    batch = int(task["batch"])
    if n % batch:
        raise ValueError("chunk size must be a multiple of the batch size")
    nb = n // batch
    bid = np.arange(n) // batch
    centres = spec.centres()
    m = centres.size
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()

    decay = 1.0 - spec.gamma * dt
    pull = spec.gamma * spec.z_bar * dt
    noise_z = spec.eps * math.sqrt(spec.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    W = spec.torus_w

    # Initial law and draw order identical to fk._run_chunk (common paths).
    z = spec.z0 + math.sqrt(spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)) \
        * rng.standard_normal(n)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    r_perp = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0
                    * rng.standard_normal((spec.n_perp, n)), W)

    fields = sorted({v["field"] for v in variants})
    mean_c = task.get("mean_contact") or {}
    chi_radii = [float(r) for r in task["chi_radii"]]
    tj_steps = [int(round(tj / dt)) for tj in spec.times()]
    chi_code = np.zeros((len(chi_radii), n), np.int64)

    Xlast = np.zeros((nv, n))
    D = np.zeros((nv, n))
    P = np.zeros((nv, nB, K))
    P2 = np.zeros((nv, nB, K))
    Pb = np.zeros((nv, nB, nb, K))
    EX = np.zeros((nv, K))
    EXX = np.zeros((nv, K))
    EXb = np.zeros((nv, nb, K))
    kidx = 0
    while kidx < K and cps[kidx] == 0:
        kidx += 1
    contact_steps = 0
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
        contact_steps += int(np.count_nonzero(dist2 < spec.contact_a**2))
        for jj, sj in enumerate(tj_steps):
            if step_no == sj:
                for ir, r in enumerate(chi_radii):
                    chi_code[ir] += (dist2 < r * r).astype(np.int64) << jj
        comp = {}
        for fld in fields:
            F = z if fld == "mid" else z + fk.FIELD_SHIFT[fld] * r_par
            act, comps = fk._slab_components(F, centres, spec)
            comp[fld] = (act, fk._wsum(comps, w) if act.size else None)
        for iv, v in enumerate(variants):
            act, e = comp[v["field"]]
            if not act.size:
                continue
            if v["gate"] == "contact":
                e = e * (dist2[act] < v["a"] * v["a"])
            elif v["gate"] == "mean":
                e = e * mean_c[v["name"]][s]
            D[iv, act] += e
        if kidx < K and step_no == cps[kidx]:
            for iv in range(nv):
                Xn = Xlast[iv] + D[iv]
                for ib in range(nB):
                    B = Bs[ib]
                    y = np.exp(-B * Xlast[iv]) * (-np.expm1(-B * D[iv]))
                    P[iv, ib, kidx] += y.sum()
                    P2[iv, ib, kidx] += (y * y).sum()
                    Pb[iv, ib, :, kidx] += np.bincount(bid, weights=y, minlength=nb)
                EX[iv, kidx] += Xn.sum()
                EXX[iv, kidx] += (Xn * Xn).sum()
                EXb[iv, :, kidx] += np.bincount(bid, weights=Xn, minlength=nb)
                Xlast[iv] = Xn
                D[iv] = 0.0
            kidx += 1
    if kidx != K:
        raise AssertionError("not all checkpoints were reached")
    npat = 1 << m
    chi_b = np.stack([np.bincount(bid * npat + chi_code[ir], minlength=nb * npat).reshape(nb, npat)
                      for ir in range(len(chi_radii))])
    runtime = time.perf_counter() - t0
    fpath = Path(task["file"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, P=P, P2=P2, Pb=Pb, EX=EX, EXX=EXX, EXb=EXb, chi_b=chi_b)
    os.replace(tmp, fpath)
    return {"chunk": int(task["chunk"]), "size": n, "runtime_seconds": runtime,
            "contact_steps": contact_steps, "file": str(fpath),
            "sha256": fk._sha256(fpath), "bytes": fpath.stat().st_size}


def simulate_reduced(spec: fk.EnsembleSpec, n_paths: int, *, name: str, variants,
                     budgets, w=None, tag: int = TAG, seed: int = SEED, replicate: int = 0,
                     chunk: int = CHUNK, batch: int = BATCH, grid: str = "production",
                     chi_radii=RADII, chunks_subset=None, workers: int = MAX_WORKERS,
                     index_dir: Path = ENS_INDEX, ens_root: Path = ENS_ROOT,
                     overwrite: bool = False, note: str = "") -> dict:
    if workers > MAX_WORKERS:
        raise ValueError("at most 3 workers")
    if tag in fk.USED_TAGS_ELSEWHERE:
        raise ValueError("tag in use elsewhere")
    m = spec.centres().size
    w = equal_w(m) if w is None else tuple(float(x) for x in w)
    variants = [fk.parse_variant(v, spec) if isinstance(v, str) else v for v in variants]
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("duplicate variant names")
    steps = spec.steps()
    T = fk.step_times(spec.dt, steps)
    sizes = [chunk] * (n_paths // chunk) + ([n_paths % chunk] if n_paths % chunk else [])
    entropy = fk.path_entropy(spec, seed=seed, tag=tag, replicate=replicate, chunk=chunk)
    children = np.random.SeedSequence(entropy).spawn(len(sizes))
    ens_dir = ens_root / name
    g = fk.build_grid(spec, grid)
    mean_contact = {}
    cache = {}
    for v in variants:
        if v["gate"] == "mean":
            if v["a"] not in cache:
                cache[v["a"]] = fk.mean_contact_curve(spec, v["a"], T)
            mean_contact[v["name"]] = cache[v["a"]]
    idx = {"name": name, "schema": "fb_n3_reduced_v1", "driver": HERE.name,
           "estimator": "N0 Feynman-Kac exact law (exact_m_prr_fk_exact_law), streaming "
                        "checkpoint reduction; same paths as fk stored/declared mode",
           "spec": spec.to_dict(), "variants": variants, "budgets": [float(b) for b in budgets],
           "w": list(w), "n_paths": int(n_paths), "chunk": int(chunk), "batch": int(batch),
           "chunk_sizes": sizes, "seed": int(seed), "tag": int(tag), "replicate": int(replicate),
           "seed_entropy": entropy,
           "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
           "grid": g["grid"], "edges": [float(e) for e in g["edges"]],
           "cps": [int(c) for c in g["cps"]],
           "window_index": [int(i) for i in g["window_index"]],
           "chi_radii": [float(r) for r in chi_radii], "ensemble_dir": str(ens_dir),
           "note": note}
    ipath = index_dir / f"{name}.json"
    existing = {}
    if ipath.exists() and not overwrite:
        old = json.loads(ipath.read_text())
        for key in ("seed_entropy", "chunk_sizes", "variants", "cps", "budgets", "w", "chi_radii"):
            if json.loads(json.dumps(old.get(key), default=fk._json_default)) != \
                    json.loads(json.dumps(idx.get(key), default=fk._json_default)):
                raise ValueError(f"existing index {ipath} differs in {key}")
        for c in old.get("chunks", []):
            if Path(c["file"]).exists():
                existing[int(c["chunk"])] = c
    todo = list(range(len(sizes))) if chunks_subset is None else list(chunks_subset)
    todo = [i for i in todo if overwrite or i not in existing]
    tasks = [{"spec": spec.to_dict(), "size": sizes[i], "seedseq": children[i],
              "variants": variants, "budgets": [float(b) for b in budgets], "w": list(w),
              "cps": idx["cps"], "batch": batch, "chunk": i,
              "file": str(ens_dir / f"chunk{i:04d}.npz"), "mean_contact": mean_contact,
              "chi_radii": idx["chi_radii"]} for i in todo]
    wait_info = wait_for_cpu() if tasks else {}
    started = time.time()
    results = []
    if tasks:
        if workers <= 1 or len(tasks) == 1:
            for t in tasks:
                results.append(_reduce_chunk(t))
                print(f"[n3] {name} chunk {results[-1]['chunk']} "
                      f"{results[-1]['runtime_seconds']:.1f}s", flush=True)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(workers, len(tasks))) as pool:
                for r in pool.imap_unordered(_reduce_chunk, tasks):
                    results.append(r)
                    print(f"[n3] {name} chunk {r['chunk']} {r['runtime_seconds']:.1f}s",
                          flush=True)
    for r in results:
        existing[int(r["chunk"])] = r
    idx["chunks"] = [existing[i] for i in sorted(existing)]
    idx["complete"] = len(idx["chunks"]) == len(sizes)
    idx["n_paths_done"] = int(sum(c["size"] for c in idx["chunks"]))
    idx["process_seconds_total"] = float(sum(c["runtime_seconds"] for c in idx["chunks"]))
    idx["last_invocation"] = {"wall_seconds": time.time() - started, "workers": workers,
                              "chunks_run": [int(r["chunk"]) for r in results],
                              "cpu_wait": wait_info,
                              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ipath.parent.mkdir(parents=True, exist_ok=True)
    tmp = ipath.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, indent=1, default=fk._json_default))
    os.replace(tmp, ipath)
    return idx


def load_reduced(name: str, index_dir: Path = ENS_INDEX) -> dict:
    idx = json.loads((index_dir / f"{name}.json").read_text())
    acc = None
    for c in sorted(idx["chunks"], key=lambda c: c["chunk"]):
        with np.load(c["file"]) as d:
            part = {k: d[k] for k in d.files}
        if acc is None:
            acc = {k: v.copy() for k, v in part.items()}
        else:
            for k in ("P", "P2", "EX", "EXX"):
                acc[k] += part[k]
            acc["Pb"] = np.concatenate([acc["Pb"], part["Pb"]], axis=2)
            acc["EXb"] = np.concatenate([acc["EXb"], part["EXb"]], axis=1)
            acc["chi_b"] = np.concatenate([acc["chi_b"], part["chi_b"]], axis=1)
    acc["N"] = int(sum(c["size"] for c in idx["chunks"]))
    acc["idx"] = idx
    acc["names"] = [v["name"] for v in idx["variants"]]
    acc["budgets"] = [float(b) for b in idx["budgets"]]
    acc["edges"] = np.asarray(idx["edges"], float)
    acc["window_index"] = np.asarray(idx["window_index"], np.int64)
    acc["spec"] = fk.EnsembleSpec.from_dict(idx["spec"])
    acc["w"] = np.asarray(idx["w"], float)
    acc["batch"] = int(idx["batch"])
    return acc


# ============================================================================
# CPU etiquette
# ============================================================================


def heavy_python_count() -> int:
    """Python processes using > 50% CPU (Xcode python runs as '.../Python')."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True,
                             timeout=20).stdout.splitlines()[1:]
    except Exception:
        return 0
    n = 0
    for ln in out:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pc = float(parts[0])
        except ValueError:
            continue
        if pc > 50.0 and ("python" in parts[1].lower()):
            n += 1
    return n


def wait_for_cpu(limit: int = 9, poll: float = 60.0, max_wait: float = 1800.0) -> dict:
    started = time.time()
    while True:
        n1 = fk.python_process_count()
        n2 = heavy_python_count()
        if n1 < limit + 1 and n2 < limit:
            return {"pgrep_python3": n1, "heavy_python": n2,
                    "waited_seconds": time.time() - started}
        if time.time() - started > max_wait:
            return {"pgrep_python3": n1, "heavy_python": n2,
                    "waited_seconds": time.time() - started, "gave_up_waiting": True}
        print(f"[n3] python processes pgrep={n1} heavy={n2}; waiting", flush=True)
        time.sleep(poll)


# ============================================================================
# Arms
# ============================================================================


def anchor_name(m: int, eps: float = 0.1) -> str:
    return f"n3_anchor_m{m}_eps{eps:g}"


def run_anchor(m: int, part: int, workers: int = MAX_WORKERS, eps: float = 0.1):
    spec = fk.EnsembleSpec(m=m, eps=eps)
    per = (ANCHOR_PATHS // CHUNK) // ANCHOR_PARTS
    subset = list(range(part * per, (part + 1) * per))
    return simulate_reduced(spec, ANCHOR_PATHS, name=anchor_name(m, eps),
                            variants=factorial_variants(),
                            budgets=ANCHOR_BUDGETS + LARGE_BUDGETS, chunks_subset=subset,
                            workers=workers,
                            note="N3 anchor factorial: field {mid,p1} x gate {contact,mean} x "
                                 "a {0.4,0.2,0.15} + no gate; equal weights")


def rpar_name(m: int) -> str:
    return f"n3_rpar0.35_m{m}_eps0.1"


def run_rpar(m: int, part: int, workers: int = MAX_WORKERS):
    spec = fk.EnsembleSpec(m=m, eps=0.1, r_par0=0.35)
    per = (ANCHOR_PATHS // CHUNK) // ANCHOR_PARTS
    subset = list(range(part * per, (part + 1) * per))
    variants = ["gate=contact,a=0.4,field=mid", "gate=mean,a=0.4,field=mid",
                "gate=none,field=mid", "gate=contact,a=0.4,field=p1"]
    return simulate_reduced(spec, ANCHOR_PATHS, name=rpar_name(m), variants=variants,
                            budgets=ANCHOR_BUDGETS + LARGE_BUDGETS, chunks_subset=subset,
                            workers=workers, chi_radii=(0.4,),
                            note="N3 Fable arm: r_par0 = 0.35 at a = 0.4")


def dtcheck_name(dt: float) -> str:
    return f"n3_dtcheck_m2_eps0.1_dt{dt:g}"


def run_dtcheck(dt: float, workers: int = MAX_WORKERS):
    spec = fk.EnsembleSpec(m=2, eps=0.1, dt=dt)
    variants = ["gate=contact,a=0.4,field=mid", "gate=mean,a=0.4,field=mid",
                "gate=contact,a=0.2,field=mid"]
    return simulate_reduced(spec, DT_CHECK_PATHS, name=dtcheck_name(dt), variants=variants,
                            budgets=(1.0, 20.0) + LARGE_BUDGETS, workers=workers,
                            chi_radii=(0.4, 0.2),
                            note="N3 large-B arm: time-step check at production geometry")


def tangent_name(eps: float) -> str:
    return f"n3_tangent_m2_eps{eps:g}"


def _mean_contact_em(spec, a, T):
    """EM-chain contact probability (valid for r_perp0 != 0); replaces fk.mean_contact_curve."""
    c = fk.contact_probability_em(spec, a)
    if c.size != T.size:
        raise ValueError("EM contact curve length mismatch")
    return c


def run_tangent(eps: float, workers: int = MAX_WORKERS, n_paths: int = TANGENT_PATHS):
    spec = fk.EnsembleSpec(m=2, eps=eps, r_par0=0.0, r_perp0=0.4)
    declared = [dict(B=B, w=(0.5, 0.5), variant="full") for B in TANGENT_BUDGETS]
    declared += [dict(B=B, w=(0.5, 0.5), variant="meancontact") for B in TANGENT_SUB_BUDGETS]
    declared += [dict(B=B, w=(0.5, 0.5), variant="nogate") for B in TANGENT_SUB_BUDGETS]
    wait_for_cpu()
    orig = fk.mean_contact_curve
    fk.mean_contact_curve = _mean_contact_em
    try:
        ens = fk.simulate_ensemble(spec, n_paths, name=tangent_name(eps), tag=TAG,
                                   mode="declared", declared=declared,
                                   variants=("full", "meancontact", "nogate"),
                                   workers=workers, wait=False,
                                   note="N3 frozen-gate arm (TH-7): boundary-tangent start "
                                        "r_par0=0, r_perp0=a=0.4; meancontact = EM-chain "
                                        "contact probability (fk.contact_probability_em)")
    finally:
        fk.mean_contact_curve = orig
    return ens


# ============================================================================
# Direct-kill confirmations
# ============================================================================


def _dk_task(task: dict) -> dict:
    import exact_m_prr_upgrade_w6_single_particle as w6
    rng = np.random.Generator(np.random.Philox(task["child"]))
    p = replace(base.MODEL, contact_a=task["a"])
    out = w6.simulate_chunk_single_particle(
        rng, task["size"], eps=task["eps"], budget=task["B"], weights=tuple(task["w"]),
        centres_z=np.asarray(task["centres"], float), dt=base.DEFAULT_DT,
        step_count=int(round(base.DEFAULT_TMAX / base.DEFAULT_DT)),
        particle_sign=float(task["particle_sign"]), p=p, n_perp=1)
    return out


def run_dk(cell: str, workers: int = MAX_WORKERS, walkers: int = DK_WALKERS) -> dict:
    m, eps, B, a, particle = DK_CELLS[cell]
    spec = fk.EnsembleSpec(m=m, eps=eps)
    w = equal_w(m)
    entropy = [SEED, TAG, 7001, m, int(round(eps * 1e9)), int(round(B * 1e9)),
               int(round(a * 1e9)), particle]
    sizes = [DK_CHUNK] * (walkers // DK_CHUNK)
    children = np.random.SeedSequence(entropy).spawn(len(sizes))
    sign = {0: 0.0, 1: 1.0, 2: -1.0}[particle]
    tasks = [{"child": ch, "size": sz, "eps": eps, "B": B, "w": list(w), "a": a,
              "centres": [float(c) for c in spec.centres()], "particle_sign": sign}
             for sz, ch in zip(sizes, children)]
    wait_info = wait_for_cpu()
    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        res = pool.map(_dk_task, tasks)
    times = np.concatenate([r["kill_times"] for r in res])
    survivors = int(sum(r["survivors"] for r in res))
    assert times.size + survivors == walkers
    kmax = max(r["kill_probability_max"] for r in res)
    g = fk.build_grid(spec, "production")
    edges = g["edges"]
    wedges = fk.production_window_edges()
    counts_window = np.histogram(times, bins=wedges)[0]
    # full-grid counts with the same checkpoint semantics as the FK grid
    T = fk.step_times(spec.dt, spec.steps())
    step_idx = np.rint(times / spec.dt).astype(np.int64)      # kill at step n -> time n*dt
    cps = g["cps"]
    full_counts = np.array([np.count_nonzero((step_idx > cps[k]) & (step_idx <= cps[k + 1]))
                            for k in range(cps.size - 1)])
    out = {"cell": cell, "m": m, "eps": eps, "B": B, "contact_a": a, "particle": particle,
           "field": {0: "pair midpoint Z", 1: "particle 1 (Z + R_par/2)"}[particle],
           "weights": list(w), "walkers": walkers, "chunk": DK_CHUNK, "seed_entropy": entropy,
           "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
           "kernel": "exact_m_prr_upgrade_w6_single_particle.simulate_chunk_single_particle "
                     "with p = replace(MODEL, contact_a=a)",
           "dt": spec.dt, "tmax": spec.tmax, "survivors": survivors,
           "kills": int(times.size), "kill_probability_max": kmax,
           "window_edges": wedges.tolist(), "window_counts": counts_window.tolist(),
           "full_grid_edges": edges.tolist(), "full_grid_counts": full_counts.tolist(),
           "full_grid_check_total": int(full_counts.sum()),
           "wall_seconds": time.time() - t0, "cpu_wait": wait_info,
           "step_time_check": bool(np.allclose(T[step_idx - 1], times, rtol=0, atol=0))}
    core.write_json(OUT / f"dk_{cell}.json", out)
    return out


# ============================================================================
# Self-test: streaming reduction == fk stored mode on common paths
# ============================================================================


def selftest() -> dict:
    spec = fk.EnsembleSpec(m=2, eps=0.1)
    n, chunk = 10_000, 5_000
    tag = fk.TAGS["spare89"]
    variants = ["full", "meancontact", "p1", "a0.2", "nogate"]
    st = fk.simulate_ensemble(spec, n, name="selftest_n3_stored", tag=tag, variants=variants,
                              workers=1, chunk=chunk, overwrite=True, wait=False, verbose=False)
    budgets = (1.0, 20.0, 1e4)
    tmpdir = fk.ENSEMBLE_ROOT / "_selftest_index"
    red = simulate_reduced(spec, n, name="selftest_n3_reduced", variants=variants,
                           budgets=budgets, tag=tag, chunk=chunk, batch=5_000, workers=1,
                           index_dir=tmpdir, ens_root=fk.ENSEMBLE_ROOT / "_selftest_n3",
                           overwrite=True)
    R = load_reduced("selftest_n3_reduced", index_dir=tmpdir)
    rows = []
    worst = 0.0
    for v in variants:
        iv = R["names"].index(v)
        for ib, B in enumerate(budgets):
            law = fk.exact_law(st, B, (0.5, 0.5), variant=v, region="all")
            mine = R["P"][iv, ib, 1:] / R["N"]
            ref = law["bin_mass"]
            msk = ref > 1e-300
            rel = np.abs(mine[msk] - ref[msk]) / ref[msk]
            # compare only bins above 1e-12 of the total for the headline number
            big = ref > 1e-12
            relb = float(np.max(np.abs(mine[big] - ref[big]) / ref[big])) if big.any() else 0.0
            worst = max(worst, relb)
            rows.append({"variant": v, "B": B, "max_rel_diff_bins_gt_1e-12": relb,
                         "max_rel_diff_all_bins": float(rel.max()),
                         "total_window_mass_stored": float(ref.sum()),
                         "total_mass_reduced": float(mine.sum())})
    # mean exposure vs fk.exposure_moments
    mom = fk.exposure_moments(st, (0.5, 0.5), variant="full")
    iv = R["names"].index("full")
    ex_rel = float(np.max(np.abs(R["EX"][iv] / R["N"] - mom["mean_X"])
                          / np.maximum(mom["mean_X"], 1e-300)))
    # contact indicators at t_j vs stored dist2_tj
    chi_ok = True
    chi_counts_stored = np.zeros(4)
    for _, _, d2 in st.iter_chunks("full"):
        code = (d2[:, 0] < 0.16).astype(int) + 2 * (d2[:, 1] < 0.16).astype(int)
        chi_counts_stored += np.bincount(code, minlength=4)
    chi_red = R["chi_b"][0].sum(0)
    chi_ok = bool(np.array_equal(chi_counts_stored.astype(int), chi_red.astype(int)))
    res = {"rows": rows, "worst_rel_diff_bins_gt_1e-12": worst,
           "mean_X_max_rel_diff": ex_rel, "chi_patterns_equal": chi_ok,
           "chi_counts_stored": chi_counts_stored.tolist(), "chi_counts_reduced": chi_red.tolist(),
           "passed": bool(worst < 1e-5 and ex_rel < 1e-5 and chi_ok),
           "note": "fk stored mode keeps float32 per-interval increments; the reduction "
                   "works in float64, so agreement is limited by float32 rounding (~1e-7 "
                   "relative per increment)"}
    core.write_json(OUT / "n3_selftest.json", res)
    return res


# ============================================================================
# Analysis helpers
# ============================================================================


def red_law(R: dict, var: str, B: float) -> dict:
    """Window-bin exact law, batch estimates and mean field for (variant, B)."""
    iv = R["names"].index(var)
    ib = R["budgets"].index(float(B))
    N = R["N"]
    bs = R["batch"]
    wi = R["window_index"]
    bins = wi[:-1] + 1                     # interval (cps[k-1], cps[k]] index for bin k-1
    mass = R["P"][iv, ib, bins] / N
    bmass = R["Pb"][iv, ib][:, bins] / bs
    edges = R["edges"][wi]
    bw = np.diff(edges)
    EXm = R["EX"][iv] / N
    Swin = np.exp(-B * EXm[wi])
    mf = Swin[:-1] * (-np.expm1(-B * np.diff(EXm[wi])))
    return {"t": 0.5 * (edges[:-1] + edges[1:]), "edges": edges, "bin_mass": mass,
            "batch_mass": bmass, "density": mass / bw, "batch_density": bmass / bw[None, :],
            "mf_mass": mf, "mf_density": mf / bw,
            "cov": np.cov(bmass, rowvar=False) / bmass.shape[0]}


def red_basins(R: dict, var: str, B: float, cuts) -> dict:
    iv = R["names"].index(var)
    ib = R["budgets"].index(float(B))
    edges = R["edges"]
    wi = R["window_index"]
    ci = []
    for c in cuts:
        k = int(np.argmin(np.abs(edges - c)))
        ci.append(k)
    if abs(cuts[0] - fk.WINDOW[0]) < 1e-12:
        ci[0] = int(wi[0])
    if abs(cuts[-1] - fk.WINDOW[1]) < 1e-12:
        ci[-1] = int(wi[-1])
    N, bs = R["N"], R["batch"]
    P = R["P"][iv, ib]
    Pb = R["Pb"][iv, ib]
    M = np.array([P[ci[a] + 1: ci[a + 1] + 1].sum() / N for a in range(len(ci) - 1)])
    Mb = np.array([Pb[:, ci[a] + 1: ci[a + 1] + 1].sum(1) / bs for a in range(len(ci) - 1)]).T
    se = Mb.std(axis=0, ddof=1) / math.sqrt(Mb.shape[0])
    EXm = R["EX"][iv] / N
    L = B * EXm[ci]
    mf = np.exp(-L[:-1]) - np.exp(-L[1:])
    return {"cuts_edges": [float(edges[k]) for k in ci], "masses": M, "se_batch": se,
            "batch_masses": Mb, "mean_field": mf, "total_window": float(M.sum())}


def frozen_surrogate(R: dict, radius: float, B: float) -> dict:
    """E[exp(-sum_{i<j} lam_i chi_i)(1 - exp(-lam_j chi_j))] from the chi patterns."""
    spec = R["spec"]
    m = spec.centres().size
    ir = [float(r) for r in R["idx"]["chi_radii"]].index(float(radius))
    cb = R["chi_b"][ir].astype(float)             # (nb, 2^m)
    lam = fk.lambdas(B, R["w"], spec)
    pats = np.arange(1 << m)
    chi = ((pats[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    ex = chi * lam[None, :]
    cum = np.concatenate([np.zeros((pats.size, 1)), np.cumsum(ex, axis=1)], axis=1)
    Mpat = np.exp(-cum[:, :-1]) * (1.0 - np.exp(-ex))   # (2^m, m)
    prob_b = cb / cb.sum(1, keepdims=True)
    Mb = np.einsum("bp,pj->bj", prob_b, Mpat)
    tot = cb.sum(0) / cb.sum()
    M = np.einsum("p,pj->j", tot, Mpat)
    se = Mb.std(axis=0, ddof=1) / math.sqrt(Mb.shape[0])
    return {"lambda": lam.tolist(), "pattern_prob": tot.tolist(), "masses": M,
            "se_batch": se, "contact_prob_tj": (tot @ chi).tolist()}


def logparabola(t, y, centre, hw):
    """sigma and vertex of a least-squares parabola in log y over |t - centre| < hw."""
    msk = (np.abs(t - centre) < hw) & (y > 0)
    if msk.sum() < 4:
        return None, None
    x = t[msk] - centre
    a, b, _ = np.polyfit(x, np.log(y[msk]), 2)
    if a >= 0:
        return None, None
    return math.sqrt(-1.0 / (2.0 * a)), float(centre - b / (2.0 * a))


def _classify(law: dict) -> dict:
    cls = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000)
    clsc = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000,
                                cov=law["cov"])
    rows = [{"time": r["time"], "relative_prominence": r["relative_prominence"],
             "z_protocol_1e6": r["z"], "significant_protocol_1e6": r["significant"]}
            for r in cls["rows"]]
    for r, rc in zip(rows, clsc["rows"]):
        r["z_fk_estimate"] = rc["z"]
        r["significant_fk_estimate"] = rc["significant"]
    return {"mode_count_protocol_1e6": cls["mode_count"],
            "mode_count_fk_estimate": clsc["mode_count"], "maxima": rows,
            "global_max_smoothed": cls["global_max"]}


def _json(o):
    return fk._jsonable(o)


# ============================================================================
# Frozen-gate hypothesis (FG1) diagnostic on the tangent-arm paths
# ============================================================================


def _passage_windows(spec) -> dict:
    """Two window families around each t_j: the TH-7 window t_j +- eps*log(1/eps) and the
    effective passage t_j +- 2 sigma_t(j), sigma_t = eps*sqrt(D0/(2 gamma) + rho^2)/|mu'(t_j)|."""
    L = math.log(1.0 / spec.eps)
    tj = spec.times()
    sig = [spec.eps * math.sqrt(spec.d0 / (2 * spec.gamma) + spec.rho**2)
           / float(spec.mu_prime_abs(t)) for t in tj]
    return {"J_theory_eps_log": [(t - spec.eps * L, t + spec.eps * L) for t in tj],
            "passage_2sigma_t": [(t - 2 * s_, t + 2 * s_) for t, s_ in zip(tj, sig)]}


def _switch_chunk(task: dict) -> dict:
    """Relative process only, same draws as fk._run_chunk: does chi switch on each window?"""
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    n = int(task["size"])
    dt = spec.dt
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    decay = 1.0 - spec.gamma * dt
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    W = spec.torus_w
    rng.standard_normal(n)                     # z_0 draw (unused, keeps the stream aligned)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    r_perp = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0
                    * rng.standard_normal((spec.n_perp, n)), W)
    fams = _passage_windows(spec)
    wins = []
    for fam, lst in fams.items():
        for j, (a_, b_) in enumerate(lst):
            wins.append((fam, j, max(1, int(math.ceil(a_ / dt))), int(math.floor(b_ / dt))))
    first = np.full((len(wins), n), -1, np.int8)
    switched = np.zeros((len(wins), n), bool)
    a2 = spec.contact_a**2
    last = max(w_[3] for w_ in wins)
    for s in range(last):
        noise = rng.standard_normal((2 + spec.n_perp, n))
        r_par *= decay
        r_par += noise_r * noise[1]
        r_perp += noise_r * noise[2:]
        np.mod(r_perp, W, out=r_perp)
        step_no = s + 1
        chi = None
        for k, (_, _, lo, hi) in enumerate(wins):
            if lo <= step_no <= hi:
                if chi is None:
                    perp_mi = np.minimum(r_perp, W - r_perp)
                    chi = (r_par * r_par + np.sum(perp_mi * perp_mi, axis=0) < a2).astype(np.int8)
                if step_no == lo:
                    first[k] = chi
                else:
                    switched[k] |= chi != first[k]
    return {"chunk": int(task["chunk"]), "n": n,
            "switched_counts": {f"{f}_j{j + 1}": int(switched[k].sum())
                                for k, (f, j, _, _) in enumerate(wins)},
            "windows": {f"{f}_j{j + 1}": [lo * dt, hi * dt] for (f, j, lo, hi) in wins}}


def gate_switching(eps: float, n_paths: int = TANGENT_PATHS, workers: int = MAX_WORKERS) -> dict:
    spec = fk.EnsembleSpec(m=2, eps=eps, r_par0=0.0, r_perp0=0.4)
    chunk = fk.DEFAULT_CHUNK
    sizes = [chunk] * (n_paths // chunk)
    entropy = fk.path_entropy(spec, seed=SEED, tag=TAG, replicate=0, chunk=chunk)
    children = np.random.SeedSequence(entropy).spawn(len(sizes))
    tasks = [{"spec": spec.to_dict(), "size": sz, "seedseq": ch, "chunk": i}
             for i, (sz, ch) in enumerate(zip(sizes, children))]
    wait_for_cpu()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        res = pool.map(_switch_chunk, tasks)
    keys = list(res[0]["switched_counts"])
    p = {k: sum(r["switched_counts"][k] for r in res) / n_paths for k in keys}
    return {"eps": eps, "N_paths": n_paths, "L_eps": math.log(1 / eps),
            "windows": res[0]["windows"], "p_gate_switches_in_window": p,
            "se": {k: math.sqrt(v * (1 - v) / n_paths) for k, v in p.items()},
            "seed_entropy": entropy,
            "note": "same relative paths as the tangent FK ensemble (same entropy and draw "
                    "order); switch = contact indicator not constant on the window"}


# ============================================================================
# Analysis
# ============================================================================

S1_DIR = core.UPGRADE_DATA / "w6_single_particle"
S1_CELLS = {2: "m2_eps0.1_B1_p1.json", 3: "m3_eps0.1_B1_p1.json"}


def _vertex(t, density, tm, bw):
    s, v = logparabola(t, density, tm, 3.5 * bw)
    return v


def _selfc(t, mass, tp, hw):
    """Iterated self-centred log-parabola (exact_m_prr_w6_width_diagnostic.fit/selfc)."""
    c = mass * 1e6
    for _ in range(6):
        msk = (t > tp - hw) & (t < tp + hw) & (c > 0)
        x = t[msk] - tp
        a, b, _ = np.polyfit(x, np.log(c[msk]), 2, w=np.sqrt(c[msk]))
        s, tp = math.sqrt(-1.0 / (2.0 * a)), tp - b / (2.0 * a)
    return s, tp


def summarize_variant(R: dict, var: str, B: float, cuts, *, with_surrogate=True) -> dict:
    law = red_law(R, var, B)
    cls = _classify(law)
    der = fk.derivative_signs(law["t"], law["density"], law["batch_density"], zthr=5.0)
    bm = red_basins(R, var, B, cuts)
    spec = R["spec"]
    lam, prod = fk.limit_masses(B, R["w"], spec)
    v = next(x for x in R["idx"]["variants"] if x["name"] == var)
    bw = float(np.median(np.diff(law["edges"])))
    peaks = []
    for r in cls["maxima"]:
        if r["significant_fk_estimate"]:
            peaks.append({"time_bin": r["time"],
                          "vertex": _vertex(law["t"], law["density"], r["time"], bw),
                          "relative_prominence": r["relative_prominence"],
                          "z_fk_estimate": r["z_fk_estimate"],
                          "significant_protocol_1e6": r["significant_protocol_1e6"]})
    out = {"variant": var, "gate": v["gate"], "a": v["a"], "field": v["field"], "B": B,
           "mode_count_protocol_1e6": cls["mode_count_protocol_1e6"],
           "mode_count_fk_estimate": cls["mode_count_fk_estimate"],
           "derivative_census_z5": {"n_maxima": der["n_maxima"], "n_minima": der["n_minima"],
                                    "maxima_t": der["maxima_t"], "minima_t": der["minima_t"]},
           "significant_peaks": peaks,
           "n_local_maxima_smoothed": len(cls["maxima"]),
           "nonsignificant_maxima": [{"time": r["time"], "relative_prominence": r["relative_prominence"],
                                      "z_fk_estimate": r["z_fk_estimate"],
                                      "z_protocol_1e6": r["z_protocol_1e6"]}
                                     for r in cls["maxima"] if not r["significant_protocol_1e6"]][:6],
           "basin_cuts": bm["cuts_edges"], "basin_masses": bm["masses"],
           "basin_se_batch": bm["se_batch"], "mean_field_masses": bm["mean_field"],
           "window_mass": bm["total_window"],
           "abs_err_mean_field_sum": float(np.abs(bm["masses"] - bm["mean_field"]).sum()),
           "product_limit_masses": prod, "lambda_j": lam,
           "abs_err_product_sum": float(np.abs(bm["masses"] - prod).sum())}
    if with_surrogate and v["gate"] == "contact" and float(v["a"]) in \
            [float(r) for r in R["idx"]["chi_radii"]]:
        fs = frozen_surrogate(R, float(v["a"]), B)
        out["frozen_surrogate_masses"] = fs["masses"]
        out["frozen_surrogate_se"] = fs["se_batch"]
        out["contact_prob_tj"] = fs["contact_prob_tj"]
        out["abs_err_frozen_surrogate_sum"] = float(np.abs(bm["masses"] - fs["masses"]).sum())
    return out


def paired_contrast(R: dict, var1: str, var2: str, B: float, cuts) -> dict:
    b1 = red_basins(R, var1, B, cuts)
    b2 = red_basins(R, var2, B, cuts)
    d = b1["batch_masses"] - b2["batch_masses"]
    diff = b1["masses"] - b2["masses"]
    se = d.std(axis=0, ddof=1) / math.sqrt(d.shape[0])
    l1 = red_law(R, var1, B)
    l2 = red_law(R, var2, B)
    dl = np.abs(l1["bin_mass"] - l2["bin_mass"]).sum()
    dlb = np.abs(l1["batch_mass"] - l2["batch_mass"]).sum(1)
    return {"minus": [var1, var2], "B": B, "basin_mass_diff": diff, "se_batch": se,
            "ci95": [(diff - 1.96 * se).tolist(), (diff + 1.96 * se).tolist()],
            "window_L1_distance": float(dl),
            "window_L1_distance_batch_sd": float(dlb.std(ddof=1))}


def late_mode(R: dict, var: str, B: float, after: float) -> dict:
    law = red_law(R, var, B)
    cls = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000,
                               cov=law["cov"])
    sm = cls["smoothed"]
    t = law["t"]
    late = t > after
    i = int(np.flatnonzero(late)[np.argmax(sm[late])])
    rows = [r for r in cls["rows"] if r["time"] > after]
    best = max(rows, key=lambda r: r["prominence"]) if rows else None
    return {"variant": var, "B": B, "late_after": after, "late_argmax_smoothed_t": float(t[i]),
            "late_mass_after": float(law["bin_mass"][late].sum()),
            "late_mass_after_se": float(law["batch_mass"][:, late].sum(1).std(ddof=1)
                                        / math.sqrt(law["batch_mass"].shape[0])),
            "window_mass": float(law["bin_mass"].sum()),
            "late_max": ({"time": best["time"], "relative_prominence": best["relative_prominence"],
                          "z_fk_estimate": best["z"], "significant": best["significant"]}
                         if best else None),
            "global_max_time": float(t[int(np.argmax(sm))])}


def compare_dk(R: dict, var: str, B: float, dk: dict, cuts, min_expected: float = 10.0) -> dict:
    """FK expected law vs a direct-kill histogram: merged-bin z, covariance chi2, basins."""
    law = red_law(R, var, B)
    counts = np.asarray(dk["window_counts"], float)
    Nw = float(dk["walkers"])
    p = law["bin_mass"]
    # merge adjacent bins until expected DK count >= min_expected (left to right)
    groups, cur, acc = [], [], 0.0
    for i in range(p.size):
        cur.append(i)
        acc += p[i] * Nw
        if acc >= min_expected:
            groups.append(cur)
            cur, acc = [], 0.0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    Mg = np.zeros((len(groups), p.size))
    for g, idx in enumerate(groups):
        Mg[g, idx] = 1.0
    pg = np.einsum("gi,i->g", Mg, p)
    cg = np.einsum("gi,i->g", Mg, counts)
    covfk = np.einsum("gi,ij,hj->gh", Mg, law["cov"], Mg)
    covdk = (np.diag(pg) - np.outer(pg, pg)) / Nw
    z = (cg / Nw - pg) / np.sqrt(np.diag(covfk) + np.diag(covdk))
    C = covfk + covdk
    r = cg / Nw - pg
    chi2 = float(r @ np.linalg.solve(C, r))
    dof = len(groups)
    # Wilson-Hilferty p-value
    x = (chi2 / dof) ** (1.0 / 3.0)
    zz = (x - (1 - 2.0 / (9 * dof))) / math.sqrt(2.0 / (9 * dof))
    pval = 0.5 * math.erfc(zz / math.sqrt(2.0))
    # basins
    edges = law["edges"]
    ci = [int(np.argmin(np.abs(edges - c))) for c in cuts]
    bz = []
    for a_ in range(len(ci) - 1):
        sl = slice(ci[a_], ci[a_ + 1])
        mfk = p[sl].sum()
        sfk = law["batch_mass"][:, sl].sum(1).std(ddof=1) / math.sqrt(law["batch_mass"].shape[0])
        mdk = counts[sl].sum() / Nw
        sdk = math.sqrt(max(mfk * (1 - mfk), 0.0) / Nw)
        bz.append({"fk": float(mfk), "dk": float(mdk), "z": float((mdk - mfk) / math.hypot(sfk, sdk))})
    import reclassify_covariance_aware as covmod
    dkc = covmod.classify_both(counts.astype(np.int64), np.asarray(dk["window_edges"]),
                               int(Nw), bandwidth=base.DEFAULT_BANDWIDTH)
    fkc = _classify(law)
    return {"variant": var, "B": B, "dk_cell": dk["cell"], "dk_walkers": int(Nw),
            "n_merged_bins": dof, "max_abs_z_merged": float(np.max(np.abs(z))),
            "n_abs_z_ge_3": int(np.sum(np.abs(z) >= 3)), "chi2_cov": chi2, "chi2_dof": dof,
            "chi2_p_wilson_hilferty": pval, "basins": bz,
            "dk_mode_count_cov_aware": int(dkc["mode_count_covariance_aware"]),
            "dk_significant_times": [r["time"] for r in dkc["rows"]
                                     if r["significant_covariance_aware"]],
            "fk_mode_count_protocol_1e6": fkc["mode_count_protocol_1e6"],
            "fk_significant_times": [r["time"] for r in fkc["maxima"]
                                     if r["significant_protocol_1e6"]],
            "dk_seed_entropy": dk["seed_entropy"]}


def compare_s1(R: dict, m: int) -> dict:
    """Particle-1 FK law at B = 1 vs the S1 5e6 direct-kill record and its widths."""
    rec = json.loads((S1_DIR / S1_CELLS[m]).read_text())
    cl = rec["results"]["classifier"]
    dk = {"cell": S1_CELLS[m], "walkers": int(rec["parameters"]["config"]["walkers"]),
          "window_counts": cl["counts"], "window_edges": cl["edges"],
          "seed_entropy": "S1 record (seed 20260813, tag 61, particle 1)"}
    valleys = fk.g_valley_times(R["spec"], R["w"])
    cuts = [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]]
    cmp_ = compare_dk(R, vname("contact", 0.4, "p1"), 1.0, dk, cuts)
    wd = json.loads((S1_DIR / "width_diagnostic_selfcentred.json").read_text())["cells"][S1_CELLS[m]]
    lp = red_law(R, vname("contact", 0.4, "p1"), 1.0)
    lm = red_law(R, vname("contact", 0.4, "mid"), 1.0)
    rows = []
    for row in wd:
        tp0, hw = row["t_G1"], row["hw"]
        s1, v1 = _selfc(lp["t"], lp["bin_mass"], tp0, hw)
        s0, v0 = _selfc(lm["t"], lm["bin_mass"], tp0, hw)
        # batch SE of ratio / shift
        rb, sb = [], []
        nb = lp["batch_mass"].shape[0]
        G = 20
        grp = np.array_split(np.arange(nb), G)
        for g in grp:
            a1, b1 = _selfc(lp["t"], lp["batch_mass"][g].mean(0), tp0, hw)
            a0, b0 = _selfc(lm["t"], lm["batch_mass"][g].mean(0), tp0, hw)
            rb.append(a1 / a0)
            sb.append(b1 - b0)
        rows.append({"t_G1": tp0, "hw": hw, "fk_sigma_p1": s1, "fk_sigma_mid": s0,
                     "fk_ratio": s1 / s0, "fk_ratio_se": float(np.std(rb, ddof=1) / math.sqrt(G)),
                     "fk_vertex_shift": v1 - v0,
                     "fk_vertex_shift_se": float(np.std(sb, ddof=1) / math.sqrt(G)),
                     "s1_mc_ratio": row["ratio"], "s1_mc_ratio_se": row["se_ratio"],
                     "s1_mc_vertex_shift": row["vertex_shift"], "s1_mc_shift_se": row["se_shift"]})
    return {"s1_record": str((S1_DIR / S1_CELLS[m]).relative_to(core.REPORT)),
            "comparison": cmp_, "widths_selfcentred": rows,
            "note": "self-centred log-parabola fits exactly as exact_m_prr_w6_width_diagnostic.py "
                    "(same t_G1 starts and half-windows), applied to the FK bin masses; SE from "
                    "20 path groups"}


def analyze_reduced(name: str, *, pairs=True, large=True) -> dict:
    R = load_reduced(name)
    spec = R["spec"]
    m = spec.centres().size
    valleys = fk.g_valley_times(spec, R["w"])
    cuts = [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]]
    out = {"name": name, "N_paths": R["N"], "batches": int(R["Pb"].shape[2]),
           "batch_size": R["batch"], "m": m, "eps": spec.eps, "r_par0": spec.r_par0,
           "cuts": cuts, "g_valleys": valleys, "seed_entropy": R["idx"]["seed_entropy"],
           "tag": R["idx"]["tag"], "variants": R["names"], "budgets": R["budgets"],
           "cells": []}
    for B in R["budgets"]:
        for var in R["names"]:
            out["cells"].append(summarize_variant(R, var, B, cuts))
    if pairs:
        con = []
        for B in R["budgets"]:
            for field in ("mid", "p1"):
                for a in RADII:
                    c, mn = vname("contact", a, field), vname("mean", a, field)
                    if c in R["names"] and mn in R["names"]:
                        con.append(dict(kind="contact_contribution(full-meancontact)",
                                        **paired_contrast(R, c, mn, B, cuts)))
                    nog = vname("none", 0.4, field)
                    if c in R["names"] and nog in R["names"]:
                        con.append(dict(kind="gate_effect(contact-nogate)",
                                        **paired_contrast(R, c, nog, B, cuts)))
                for a in RADII:
                    c1, c0 = vname("contact", a, "p1"), vname("contact", a, "mid")
                    if field == "mid" and c1 in R["names"] and c0 in R["names"]:
                        con.append(dict(kind="field_effect(p1-mid)",
                                        **paired_contrast(R, c1, c0, B, cuts)))
        out["contrasts"] = con
    if large:
        lm = []
        for B in R["budgets"]:
            for var in R["names"]:
                lm.append(late_mode(R, var, B, valleys[0]))
        out["late_modes"] = lm
    return out


def analyze_tangent(eps: float) -> dict:
    ens = fk.load_ensemble(tangent_name(eps))
    res = ens.declared()
    spec = ens.spec
    w = (0.5, 0.5)
    valleys = fk.g_valley_times(spec, w)
    cuts = [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]]
    t = res["t"]
    dt = spec.dt
    # contact patterns at t_j from the stored dist2_tj
    codes = []
    for c in ens.chunks:
        with np.load(c["file"]) as d:
            d2 = d["dist2_tj"]
        codes.append((d2[:, 0] < spec.contact_a**2).astype(int)
                     + 2 * (d2[:, 1] < spec.contact_a**2).astype(int))
    codes = np.concatenate(codes)
    pat = np.bincount(codes, minlength=4) / codes.size
    # orthant law (TH-7 lemma th7:lem-tangent), sigma0^2 = sigma_perp0^2
    s0 = spec.sigma_perp0**2
    v1, v2 = s0 + 4 * spec.d0 * spec.times()[0], s0 + 4 * spec.d0 * spec.times()[1]
    rho12 = math.sqrt(v1 / v2)
    P11 = 0.25 + math.asin(rho12) / (2 * math.pi)
    P01 = 0.25 - math.asin(rho12) / (2 * math.pi)
    sig_t1 = spec.eps * math.sqrt(spec.d0 / (2 * spec.gamma) + spec.rho**2) / \
        float(spec.mu_prime_abs(spec.times()[0]))
    nb = res["batch_p_step"].shape[1]
    rows = []
    for i, d in enumerate(res["declared"]):
        B = float(d["B"])
        lam = fk.lambdas(B, w, spec)
        orth = [0.5 * (1 - math.exp(-lam[0])),
                (P11 * math.exp(-lam[0]) + P01) * (1 - math.exp(-lam[1]))]
        femp = [pat[1] * (1 - math.exp(-lam[0])) + pat[3] * (1 - math.exp(-lam[0])),
                (pat[3] * math.exp(-lam[0]) + pat[2]) * (1 - math.exp(-lam[1]))]
        p = res["p_step"][i]
        pb = res["batch_p_step"][i]
        ci = [0] + [int(np.searchsorted(t, c, side="left")) for c in cuts[1:-1]] + [None]
        # basins on step times: [0.5, valley), [valley, 3.5] (histogram semantics)
        lo = t >= cuts[0]
        hi = t <= cuts[-1]
        masks = []
        for a_ in range(len(cuts) - 1):
            mk = (t >= cuts[a_]) & ((t < cuts[a_ + 1]) if a_ < len(cuts) - 2 else hi)
            masks.append(mk)
        M = np.array([p[mk].sum() for mk in masks])
        Mb = np.array([pb[:, mk].sum(1) for mk in masks]).T
        se = Mb.std(axis=0, ddof=1) / math.sqrt(nb)
        mf_step = None
        mx = res["mean_X"][int(d["pair"])]
        prev = np.concatenate([[0.0], mx[:-1]])
        mfs = np.exp(-B * prev) * (-np.expm1(-B * (mx - prev)))
        Mmf = np.array([mfs[mk].sum() for mk in masks])
        dens = p / dt
        bd = pb / dt
        der = fk.derivative_signs(t, dens, bd, bandwidth=0.15 * sig_t1, zthr=5.0)
        der2 = fk.derivative_signs(t, dens, bd, bandwidth=0.5 * sig_t1, zthr=5.0)
        wedges = fk.production_window_edges()
        bmass_ = np.histogram(t, bins=wedges, weights=p)[0]
        bb_ = np.array([np.histogram(t, bins=wedges, weights=row)[0] for row in pb])
        law = {"bin_mass": bmass_, "edges": wedges,
               "cov": np.cov(bb_, rowvar=False) / bb_.shape[0]}
        cls = fk.classify_expected(law["bin_mass"], law["edges"], walkers=1_000_000,
                                   cov=law["cov"])
        rows.append({"variant": d["variant"], "B": B, "lambda": lam.tolist(),
                     "fk_masses": M, "fk_se_batch": se, "orthant_law": orth,
                     "frozen_empirical_chi": femp, "mean_field": Mmf,
                     "z_fk_minus_orthant": ((M - np.array(orth)) / se),
                     "fk_minus_orthant": (M - np.array(orth)),
                     "fk_minus_frozen_empirical": (M - np.array(femp)),
                     "window_mass": float(M.sum()),
                     "pre_window_mass": float(p[t < cuts[0]].sum()),
                     "post_window_mass_to_tmax": float(p[t > cuts[-1]].sum()),
                     "derivative_census_bw0.15sig": {"n_maxima": der["n_maxima"],
                                                     "n_minima": der["n_minima"],
                                                     "maxima_t": der["maxima_t"],
                                                     "minima_t": der["minima_t"],
                                                     "bandwidth": der["bandwidth"]},
                     "derivative_census_bw0.5sig": {"n_maxima": der2["n_maxima"],
                                                    "n_minima": der2["n_minima"],
                                                    "maxima_t": der2["maxima_t"],
                                                    "minima_t": der2["minima_t"],
                                                    "bandwidth": der2["bandwidth"]},
                     "production_bins_fk_mode_count": int(cls["mode_count"]),
                     "production_bins_maxima": [{"time": r["time"], "rel": r["relative_prominence"],
                                                 "z": r["z"], "sig": r["significant"]}
                                                for r in cls["rows"]]})
    return {"name": tangent_name(eps), "eps": eps, "N_paths": res["N"], "batches": nb,
            "seed_entropy": ens.index["seed_entropy"], "cuts": cuts,
            "contact_pattern_probs_tj": {"P00": pat[0], "P10(chi1=1,chi2=0)": pat[1],
                                         "P01(chi1=0,chi2=1)": pat[2], "P11": pat[3]},
            "orthant": {"rho12": rho12, "P11": P11, "P01": P01, "P10": P01, "P00": P11,
                        "saturation_late_mass": P01},
            "sigma_t1": sig_t1, "rows": rows}


def analyze_dtcheck() -> dict:
    out = {}
    for dt in (5e-4, 2.5e-4):
        name = dtcheck_name(dt)
        if not (ENS_INDEX / f"{name}.json").exists():
            continue
        R = load_reduced(name)
        valleys = fk.g_valley_times(R["spec"], R["w"])
        cuts = [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]]
        rows = []
        for B in R["budgets"]:
            for var in R["names"]:
                bm = red_basins(R, var, B, cuts)
                lm = late_mode(R, var, B, valleys[0])
                rows.append({"variant": var, "B": B, "basin_masses": bm["masses"],
                             "basin_se_batch": bm["se_batch"], **{k: lm[k] for k in (
                                 "late_argmax_smoothed_t", "late_mass_after",
                                 "late_mass_after_se", "late_max", "global_max_time")}})
        out[f"dt{dt:g}"] = {"name": name, "N_paths": R["N"], "dt": dt,
                            "seed_entropy": R["idx"]["seed_entropy"], "rows": rows}
    return out


def acceptance_checks(S: dict) -> dict:
    """Programmatic TH-7 / N3 acceptance numbers (see GAP_CLOSURE_PLAN.md N3, TH-7)."""
    out = {}
    tg = S.get("tangent", {})
    if "eps0.025" in tg:
        rows = [r for r in tg["eps0.025"]["rows"] if r["variant"] == "full" and r["B"] <= 8]
        zs = [abs(z) for r in rows for z in r["z_fk_minus_orthant"]]
        out["orthant_within_3se_eps0.025_B_le_8"] = bool(max(zs) < 3.0)
        out["max_abs_z_fk_minus_orthant_eps0.025_B_le_8"] = max(zs)
        out["max_abs_diff_fk_minus_orthant_eps0.025_B_le_8"] = max(
            abs(d) for r in rows for d in r["fk_minus_orthant"])
    gap = {}
    for key, T in tg.items():
        for r in T["rows"]:
            if r["variant"] == "full":
                gap.setdefault(f"B{r['B']:g}", {})[key] = r["fk_minus_orthant"]
    out["fk_minus_orthant_by_B_and_eps"] = gap
    sat = {}
    for key, T in tg.items():
        rows = [r for r in T["rows"] if r["variant"] == "full" and r["B"] >= 20]
        sat[key] = {"late_mass_B20_50_200": [r["fk_masses"][1] for r in sorted(rows, key=lambda r: r["B"])],
                    "min_late_mass_B_ge_20": min(r["fk_masses"][1] for r in rows) if rows else None,
                    "late_modes_census": [r["derivative_census_bw0.15sig"]["n_maxima"]
                                          for r in sorted(rows, key=lambda r: r["B"])]}
        sat[key]["saturation_ge_0.15"] = bool(sat[key]["min_late_mass_B_ge_20"] is not None
                                              and sat[key]["min_late_mass_B_ge_20"] >= 0.15)
    out["saturation"] = sat
    prod = {}
    for mkey, A in S.get("anchors", {}).items():
        rows = []
        for c in A["cells"]:
            if c["variant"] in (vname("contact", a, "mid") for a in RADII) and \
                    c["B"] in ANCHOR_BUDGETS:
                rows.append({"variant": c["variant"], "B": c["B"],
                             "err_product": c["abs_err_product_sum"],
                             "err_frozen_surrogate": c.get("abs_err_frozen_surrogate_sum"),
                             "err_mean_field": c["abs_err_mean_field_sum"],
                             "surrogate_reduces_error": bool(c.get("abs_err_frozen_surrogate_sum", 9)
                                                             < c["abs_err_product_sum"])})
        prod[mkey] = rows
    out["production_frozen_surrogate_vs_product"] = prod
    s1 = {}
    for mkey, A in S.get("anchors", {}).items():
        w = A.get("s1_reproduction_B1", {}).get("widths_selfcentred", [])
        s1[mkey] = {"fk_ratios": [r["fk_ratio"] for r in w],
                    "s1_mc_ratios": [r["s1_mc_ratio"] for r in w],
                    "fk_vertex_shifts": [r["fk_vertex_shift"] for r in w],
                    "max_abs_shift": max([abs(r["fk_vertex_shift"]) for r in w], default=None),
                    "ratio_minus_s1_over_se": [(r["fk_ratio"] - r["s1_mc_ratio"])
                                               / math.hypot(r["fk_ratio_se"], r["s1_mc_ratio_se"])
                                               for r in w]}
    out["s1_widths_shifts"] = s1
    return out


def analyze() -> dict:
    t0 = time.time()
    summary = {"item": "N3", "driver": HERE.name, "tag": TAG, "seed": SEED,
               "estimator": "N0 FK exact law (exact_m_prr_fk_exact_law); anchors via the "
                            "streaming checkpoint reduction (selftest: n3_selftest.json)",
               "selftest": json.loads((OUT / "n3_selftest.json").read_text())
               if (OUT / "n3_selftest.json").exists() else None}
    anchors = {}
    for m in (2, 3):
        name = anchor_name(m)
        if (ENS_INDEX / f"{name}.json").exists():
            a = analyze_reduced(name)
            R = load_reduced(name)
            a["s1_reproduction_B1"] = compare_s1(R, m)
            anchors[f"m{m}"] = a
    summary["anchors"] = anchors
    rp = {}
    for m in (2, 3):
        name = rpar_name(m)
        if (ENS_INDEX / f"{name}.json").exists():
            rp[f"m{m}"] = analyze_reduced(name)
    summary["rpar0.35"] = rp
    tg = {}
    for eps in (0.05, 0.025, 0.0125):
        if fk.index_path(tangent_name(eps)).exists():
            tg[f"eps{eps:g}"] = analyze_tangent(eps)
    summary["tangent"] = tg
    summary["dtcheck"] = analyze_dtcheck()
    if (OUT / "n3_gate_switching.json").exists():
        summary["gate_switching_FG1"] = json.loads((OUT / "n3_gate_switching.json").read_text())
    dks = {}
    for cell, (m, eps, B, a, particle) in DK_CELLS.items():
        f = OUT / f"dk_{cell}.json"
        if not f.exists() or f"m{m}" not in anchors:
            continue
        dk = json.loads(f.read_text())
        R = load_reduced(anchor_name(m))
        var = vname("contact", a, "p1" if particle == 1 else "mid")
        valleys = fk.g_valley_times(R["spec"], R["w"])
        dks[cell] = compare_dk(R, var, B, dk, [fk.WINDOW[0]] + valleys + [fk.WINDOW[1]])
    summary["direct_kill_confirmations"] = dks
    summary["acceptance_checks"] = acceptance_checks(summary)
    summary["analysis_seconds"] = time.time() - t0
    core.write_json(OUT / "n3_summary.json", _json(summary))
    return summary


# ============================================================================
# Figure
# ============================================================================


def make_figure() -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    S = json.loads((OUT / "n3_summary.json").read_text())
    R = load_reduced(anchor_name(2))
    cols = {0.4: core.OI_BLUE, 0.2: core.OI_ORANGE, 0.15: core.OI_VERMILLION}
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")
    # (a) exact densities at (2, 0.1), B = 20: contact gate vs deterministic mean contact
    ax = axs[0, 0]
    Bp = 20.0
    for a in RADII:
        lc = red_law(R, vname("contact", a, "mid"), Bp)
        lm = red_law(R, vname("mean", a, "mid"), Bp)
        ax.semilogy(lc["t"], np.maximum(lc["density"], 1e-6), color=cols[a], lw=1.2,
                    label=f"contact gate, $a={a:g}$")
        ax.semilogy(lm["t"], np.maximum(lm["density"], 1e-6), color=cols[a], lw=1.0, ls="--",
                    label=f"mean contact $c_a(t)$, $a={a:g}$")
    ln = red_law(R, vname("none", 0.4, "mid"), Bp)
    ax.semilogy(ln["t"], np.maximum(ln["density"], 1e-6), color="0.35", lw=0.9, ls=":",
                label="no gate")
    ax.set_ylim(1e-4, 3e3)
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    ax.set_ylabel("exact density $f(t)$")
    ax.set_title("(a) $m=2$, $\\varepsilon=0.1$, $B=20$", loc="left")
    ax.legend(fontsize=5.8, ncol=2, loc="upper center", columnspacing=0.8, handlelength=1.8)
    # (b) late basin mass vs B: contact vs mean contact (m = 2)
    ax = axs[0, 1]
    cells = S["anchors"]["m2"]["cells"]
    for a in RADII:
        for gate, mk, ls in (("contact", "o", "-"), ("mean", "s", "--")):
            rows = [c for c in cells if c["variant"] == vname(gate, a, "mid")]
            rows.sort(key=lambda c: c["B"])
            B = np.array([c["B"] for c in rows])
            M2 = np.array([c["basin_masses"][-1] for c in rows])
            se = np.array([c["basin_se_batch"][-1] for c in rows])
            ax.errorbar(B, np.maximum(M2, 1e-7), yerr=2 * se, color=cols[a], marker=mk,
                        ms=3.2, lw=1.0, ls=ls, mfc=cols[a] if gate == "contact" else "white",
                        label=(f"$a={a:g}$, contact" if gate == "contact" else f"$a={a:g}$, mean contact"))
        rows = sorted([c for c in cells if c["variant"] == vname("contact", a, "mid")],
                      key=lambda c: c["B"])
        ax.plot([c["B"] for c in rows], [c["frozen_surrogate_masses"][-1] for c in rows],
                color=cols[a], lw=0.8, ls=":")
    ax.plot([], [], color="0.3", lw=0.8, ls=":", label="frozen-gate law, measured $\\chi$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("late basin mass $M_2$")
    ax.set_title("(b) $m=2$, $\\varepsilon=0.1$: late mode vs budget", loc="left")
    ax.legend(fontsize=5.8, ncol=2, loc="lower left")
    # (c) frozen-gate arm
    ax = axs[1, 0]
    tcols = {"eps0.05": core.OI_SKY, "eps0.025": core.OI_BLUE, "eps0.0125": core.OI_PURPLE}
    orth_B, orth_M = None, None
    for key, T in S["tangent"].items():
        rows = sorted([r for r in T["rows"] if r["variant"] == "full"], key=lambda r: r["B"])
        B = np.array([r["B"] for r in rows])
        M2 = np.array([r["fk_masses"][1] for r in rows])
        se = np.array([r["fk_se_batch"][1] for r in rows])
        ax.errorbar(B, M2, yerr=2 * se, color=tcols.get(key, "k"), marker="o", ms=3.2, lw=1.0,
                    label=f"exact law, $\\varepsilon={T['eps']:g}$")
        orth_B, orth_M = B, np.array([r["orthant_law"][1] for r in rows])
        mf = np.array([r["mean_field"][1] for r in rows])
        if key == "eps0.025":
            ax.plot(B, mf, color="0.4", ls="--", lw=1.0, label="mean field, $\\varepsilon=0.025$")
    if orth_B is not None:
        ax.plot(orth_B, orth_M, color="k", lw=1.2, label="frozen-gate (orthant) law")
        sat = list(S["tangent"].values())[0]["orthant"]["saturation_late_mass"]
        ax.axhline(sat, color="k", ls=":", lw=0.8)
        ax.text(1.1 * orth_B[0], sat * 1.06, f"$\\pi_2={sat:.3f}$", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("late basin mass $M_2$")
    ax.set_title("(c) boundary-tangent start ($r_{\\parallel,0}=0$, $r_{\\perp,0}=a$)", loc="left")
    ax.legend(fontsize=6.0, loc="upper right")
    # (d) large-B: late-mode time vs B (production geometry)
    ax = axs[1, 1]
    lms = [x for x in S["anchors"]["m2"]["late_modes"] if x["variant"] == vname("contact", 0.4, "mid")]
    lms.sort(key=lambda x: x["B"])
    ax.plot([x["B"] for x in lms], [x["late_argmax_smoothed_t"] for x in lms], "o-",
            color=core.OI_BLUE, ms=3.5, label="$\\Delta t=10^{-3}$ ($10^6$ paths)")
    dmk = {"dt0.0005": ("s", core.OI_GREEN, "$\\Delta t=5\\times10^{-4}$"),
           "dt0.00025": ("^", core.OI_VERMILLION, "$\\Delta t=2.5\\times10^{-4}$")}
    for key, blk in S.get("dtcheck", {}).items():
        rows = sorted([r for r in blk["rows"] if r["variant"] == vname("contact", 0.4, "mid")],
                      key=lambda r: r["B"])
        mk, cc, lab = dmk.get(key, ("x", "k", key))
        ax.plot([r["B"] for r in rows], [r["late_argmax_smoothed_t"] for r in rows], mk,
                color=cc, ms=4, mfc="none", label=lab + " ($2\\times10^5$)")
    ax.axhline(2.5, color="0.5", ls=":", lw=0.8)
    ax.text(300, 2.485, "second passage $t_2=2.5$", fontsize=6.5, color="0.35", ha="center", va="top")
    ax.set_ylim(1.7, 2.56)
    ax.set_xscale("log")
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("late-mode time (units of $1/\\gamma$)")
    ax.set_title("(d) $m=2$, $\\varepsilon=0.1$, $a=0.4$: large-$B$ displacement", loc="left")
    ax.legend(fontsize=6.0, loc="lower left")
    return core.save_figure(fig, core.FIGURES / "fb_n3_contact_factorial")


# ============================================================================
# CLI
# ============================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    a = sub.add_parser("anchor")
    a.add_argument("--m", type=int, required=True)
    a.add_argument("--part", type=int, required=True)
    a.add_argument("--workers", type=int, default=MAX_WORKERS)
    r = sub.add_parser("rpar")
    r.add_argument("--m", type=int, required=True)
    r.add_argument("--part", type=int, required=True)
    r.add_argument("--workers", type=int, default=MAX_WORKERS)
    t = sub.add_parser("tangent")
    t.add_argument("--eps", type=float, required=True)
    t.add_argument("--workers", type=int, default=MAX_WORKERS)
    d = sub.add_parser("dtcheck")
    d.add_argument("--dt", type=float, required=True)
    d.add_argument("--workers", type=int, default=MAX_WORKERS)
    k = sub.add_parser("dk")
    k.add_argument("--cell", choices=sorted(DK_CELLS), required=True)
    k.add_argument("--workers", type=int, default=MAX_WORKERS)
    b = sub.add_parser("bench")
    b.add_argument("--m", type=int, default=2)
    b.add_argument("--n", type=int, default=5000)
    sub.add_parser("switching")
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        res = selftest()
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    elif args.cmd == "anchor":
        idx = run_anchor(args.m, args.part, workers=args.workers)
        print(idx["n_paths_done"], idx["last_invocation"])
    elif args.cmd == "rpar":
        idx = run_rpar(args.m, args.part, workers=args.workers)
        print(idx["n_paths_done"], idx["last_invocation"])
    elif args.cmd == "tangent":
        ens = run_tangent(args.eps, workers=args.workers)
        print(ens)
    elif args.cmd == "dtcheck":
        idx = run_dtcheck(args.dt, workers=args.workers)
        print(idx["n_paths_done"], idx["last_invocation"])
    elif args.cmd == "dk":
        out = run_dk(args.cell, workers=args.workers)
        print({k: out[k] for k in ("cell", "kills", "survivors", "wall_seconds")})
    elif args.cmd == "bench":
        spec = fk.EnsembleSpec(m=args.m, eps=0.1)
        t0 = time.time()
        simulate_reduced(spec, args.n, name=f"bench_n3_m{args.m}", variants=factorial_variants(),
                         budgets=ANCHOR_BUDGETS + LARGE_BUDGETS, chunk=args.n, workers=1,
                         index_dir=fk.ENSEMBLE_ROOT / "_selftest_index",
                         ens_root=fk.ENSEMBLE_ROOT / "_selftest_n3", overwrite=True)
        print("bench seconds", time.time() - t0, "paths", args.n)
    elif args.cmd == "switching":
        out = {f"eps{e:g}": gate_switching(e) for e in (0.05, 0.025, 0.0125)}
        core.write_json(OUT / "n3_gate_switching.json", _json(out))
        print(json.dumps({k: v["p_gate_switches_in_window"] for k, v in out.items()}))
    elif args.cmd == "analyze":
        analyze()
    elif args.cmd == "figure":
        make_figure()


if __name__ == "__main__":
    main()
