#!/usr/bin/env python3
"""Total-Error Protocol (TEP) harness (uplift2 EXECUTION_SPEC section 3).

For one stripe design (JSON, see below) this runs DIRECT-KILL simulations of the production
model (exact_m_prr_upgrade_core.simulate_chunk_general: Euler--Maruyama pair + end-of-step Doi
kill, walker compaction; identical to validate_exact_m_offlattice.simulate_chunk for d = 2) at
three time steps dt, dt/2, dt/4 with seeds that are disjoint across levels and from every design
ensemble, and evaluates the SAME smooth functionals as the design estimator
(fb_v2_design_estimator: peak times = maximisers of the Gaussian-kernel-smoothed density f_h,
valley cuts, basin masses on the f_h-valley cuts, ratios), with delete-one-chunk jackknife SEs.

Richardson extrapolation (weak order 1 in dt; Q(dt) = Q0 + a dt + b dt^2 + ...):
    R1  = 2 Q(dt/4) - Q(dt/2)                    first order, two finest levels (reported value)
    R1c = 2 Q(dt/2) - Q(dt)                      first order, two coarsest levels (consistency)
    R2  = (8 Q(dt/4) - 6 Q(dt/2) + Q(dt)) / 3    second order, all three levels
    order ratio rho = (Q(dt) - Q(dt/2)) / (Q(dt/2) - Q(dt/4))   (-> 2 for weak order 1)
Total error (the TEP number):
    level L (quoted value Q(dt_L)):  bias_L = Q(dt_L) - R1,   TE_L = |bias_L| + 2 SE(Q(dt_L))
    continuum (quoted value R1):     bias_c = R2 - R1,        TE_c = |bias_c| + 2 SE(R1)
with SE(R1) = sqrt(4 SE_4^2 + SE_2^2) (levels independent).  Target checks (if the design JSON
carries targets): peak times pass if |Q - T| <= max(0.01, TE); masses / ratios pass if
|Q - target| <= TE (i.e. |z| <= 2 with the bias included).  Both the design-dt convention (level 1,
the dt of the FK design estimator) and the continuum convention are reported.

Design JSON
-----------
    {"label": "...", "m": 3, "eps": 0.05, "B": 8.0,
     "c": [...]  or  "stripe_times": [...],   "w": [...],
     "h": 0.02,                       (optional; smoothing bandwidth in time units)
     "cuts0": [...],                  (optional; basins used to LOCATE peaks; default: EM G valleys)
     "mass_cuts": "valley" | [...],   (optional; default "valley")
     "targets": {"peaks": [...], "ratios": [...], "masses": [...]},        (optional)
     "fk_prediction": {"dt": 0.001, "peaks": [...], "masses": [...], "ratios": [...],
                       "se": {"peaks": [...], ...}}}                          (optional)

Subcommands
-----------
    run      --design D.json [--walkers 2e5 --chunk 2.5e4 --workers 3 --dt 1e-3 --levels 1,2,4
              --tag 211 --replicate 0 --out DIR --no-wait]
    analyze  --run DIR                      -> DIR/tep_<label>.json
    sbatch   --design D.json [--walkers 1e7 --chunk 5e3 --hours 2 --job NAME]
              writes a SLURM script for Isambard 3 (grace, 144 cores, <= 4 h)
    push     --design D.json  (rsync code + design to Isambard, md5-verified)   [needs SSH]
    submit   --design D.json  (push + sbatch on the remote)                      [needs SSH]
    fetch    --label L        (rsync the remote run dir back, md5-verified)      [needs SSH]
    dkfd     --design D.json [--param c3 --step 0.01 --walkers 2e5]
              gradient check: common-random-number direct-kill central differences
              (non-compacting CRN simulator, same law) vs the estimator's CRN difference
              and pathwise gradient.
    smoke    tiny local end-to-end run (run + analyze) on an N11 design

Seeds: base 20260923; tag 211 (TEP), 212 (NU-E validation), 213 (hero), 214 (DK-FD); entropy
= [seed, tag, replicate, level, dt, eps, B, w..., c..., walkers, chunk] -> disjoint from all
design ensembles (fk.path_entropy layout) and across levels.
Remote: Isambard 3, host alias isambard3, repo ~/valley-k-small (.venv311); code is pushed
to .../encounter_multimodal_prr/code_v2sync/ (never overwrites remote repo files).
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
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_upgrade_core as core  # noqa: E402
import fb_v2_design_estimator as de  # noqa: E402

fk = de.fk
REPORT = de.REPORT
TEP_ROOT = de.FB_DATA / "V2_TEP"
TAG_TEP = de.TAGS["TEP_DK"]
LEVELS = (1, 2, 4)
REMOTE_HOST = "isambard3"
REMOTE_REPO = "valley-k-small"                  # relative to the remote $HOME (ssh, rsync)
REMOTE_REPORT = f"{REMOTE_REPO}/research/reports/encounter_multimodal_prr"
REMOTE_CODE = f"{REMOTE_REPORT}/code_v2sync"
REMOTE_TEP = f"{REMOTE_REPORT}/artifacts/data/exact_m_fixed_budget/V2_TEP"
SYNC_FILES = ("fb_v2_tep.py", "fb_v2_design_estimator.py", "exact_m_prr_fk_exact_law.py",
              "exact_m_prr_upgrade_core.py", "validate_exact_m_offlattice.py", "fb_v2_saa_newton.py",
              "fb_allocation_law.py", "fb_v2_tep_ladder.py", "reclassify_covariance_aware.py")


# ============================================================================
# Design IO and seeds
# ============================================================================


def load_design(path) -> dict:
    d = json.loads(Path(path).read_text())
    spec = de.path_spec(float(d["eps"]), tmax=float(d.get("tmax", 4.0)))
    th = de.Design.from_dict(d, spec)
    d["_design"] = th
    d.setdefault("label", Path(path).stem)
    d.setdefault("h", 0.02)
    d.setdefault("mass_cuts", "valley")
    return d


def _q(x, s=1e9) -> int:
    return int(round(float(x) * s)) % (1 << 63)


def level_entropy(d: dict, *, tag: int, replicate: int, level: int, dt: float, walkers: int,
                  chunk: int) -> list[int]:
    th = d["_design"]
    return ([de.BASE_SEED, int(tag), int(replicate), int(level), _q(dt, 1e15), _q(d["eps"]),
             _q(th.B)] + [_q(x) for x in th.w] + [_q(x) for x in th.c] + [int(walkers), int(chunk)])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================================
# Direct-kill chunk workers
# ============================================================================


def _dk_chunk(task: dict) -> dict:
    """Production direct kill (walker compaction); counts per step."""
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()
    out = core.simulate_chunk_general(
        rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]), weights=tuple(task["w"]),
        centres_z=np.asarray(task["c"], float), dt=spec.dt, step_count=spec.steps(),
        p=spec.model(), n_perp=spec.n_perp)
    steps = spec.steps()
    k = np.rint(out["kill_times"] / spec.dt).astype(np.int64)
    counts = np.bincount(k - 1, minlength=steps)[:steps]
    return {"level": task["level"], "chunk": task["chunk"], "counts": counts,
            "survivors": int(out["survivors"]), "kill_probability_max": out["kill_probability_max"],
            "seconds": time.perf_counter() - t0}


def simulate_chunk_crn(rng, count: int, *, spec: fk.EnsembleSpec, designs: list) -> list:
    """Direct kill WITHOUT compaction, for several designs on common random numbers.

    Every walker keeps its own noise column and its own kill uniform at every step, so the
    walkers of different designs share paths and coins (dead walkers are masked).  Each walker's
    law is that of the production simulator (per-walker increments are iid N(0, 1) and the kill
    coin is an independent U(0, 1) compared with 1 - exp(-kappa dt)); only the RNG consumption
    differs.  Returns per-design counts per step and survivors.
    """
    p = spec.model()
    dt = spec.dt
    steps = spec.steps()
    decay = 1.0 - p.gamma * dt
    pull = p.gamma * p.z_bar * dt
    noise_z = spec.eps * math.sqrt(p.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(p.d0 * dt)
    W = p.torus_w
    a2 = p.contact_a ** 2
    s = spec.sd()
    inv2 = 1.0 / (2.0 * s * s)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * s)
    z = p.z0 + math.sqrt(spec.eps**2 * p.d0 / (2.0 * p.gamma)) * rng.standard_normal(count)
    r_par = p.r_par0 + spec.eps * p.u0 * rng.standard_normal(count)
    r_perp = np.mod(p.r_perp0 + spec.eps * p.sigma_perp0 * rng.standard_normal((spec.n_perp, count)), W)
    alive = [np.ones(count, bool) for _ in designs]
    counts = [np.zeros(steps, np.int64) for _ in designs]
    for st in range(steps):
        noise = rng.standard_normal((2 + spec.n_perp, count))
        u = rng.random(count)
        z *= decay
        z += pull
        z += noise_z * noise[0]
        r_par *= decay
        r_par += noise_r * noise[1]
        r_perp += noise_r * noise[2:]
        np.mod(r_perp, W, out=r_perp)
        perp_mi = np.minimum(r_perp, W - r_perp)
        on = (r_par * r_par + np.sum(perp_mi * perp_mi, axis=0)) < a2
        for i, th in enumerate(designs):
            idx = np.flatnonzero(on & alive[i])
            if idx.size == 0:
                continue
            zc = z[idx]
            kap = np.zeros(idx.size)
            for wj, cj in zip(th.w, th.c):
                dd = zc - cj
                kap += wj * np.exp(-dd * dd * inv2)
            kap *= th.B * norm / W ** spec.n_perp
            killed = idx[u[idx] < -np.expm1(-kap * dt)]
            if killed.size:
                alive[i][killed] = False
                counts[i][st] += killed.size
    return [{"counts": c, "survivors": int(a.sum())} for c, a in zip(counts, alive)]


def _crn_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    designs = [de.Design.from_dict(x) for x in task["designs"]]
    res = simulate_chunk_crn(rng, int(task["size"]), spec=spec, designs=designs)
    return {"chunk": task["chunk"], "res": res}


# ============================================================================
# run: direct kill at dt, dt/2, dt/4
# ============================================================================


def _design_payload(d: dict) -> dict:
    out = {k: v for k, v in d.items() if not k.startswith("_")}
    out["c"] = d["_design"].c.tolist()
    return out


def cmd_run(args) -> Path:
    d = load_design(args.design)
    th = d["_design"]
    out = Path(args.out) if args.out else TEP_ROOT / d["label"]
    out.mkdir(parents=True, exist_ok=True)
    de.write_json(out / "design.json", _design_payload(d))
    walkers = int(float(args.walkers))
    chunk = int(float(args.chunk))
    dt0 = float(args.dt)
    levels = [int(x) for x in str(args.levels).split(",")]
    tasks, plan = [], {}
    for L in levels:
        if (out / f"level{L}.json").exists() and not args.force:
            print(f"[run] level {L} exists, skip", flush=True)
            continue
        spec = de.path_spec(d["eps"], dt=dt0 / L, tmax=float(d.get("tmax", 4.0)))
        sizes = [chunk] * (walkers // chunk) + ([walkers % chunk] if walkers % chunk else [])
        ent = level_entropy(d, tag=args.tag, replicate=args.replicate, level=L, dt=spec.dt,
                            walkers=walkers, chunk=chunk)
        kids = np.random.SeedSequence(ent).spawn(len(sizes))
        plan[L] = {"spec": spec, "sizes": sizes, "entropy": ent}
        for i, (sz, kid) in enumerate(zip(sizes, kids)):
            tasks.append({"spec": spec.to_dict(), "size": sz, "seedseq": kid, "B": th.B,
                          "w": th.w.tolist(), "c": th.c.tolist(), "level": L, "chunk": i})
    if not tasks:
        return out
    tasks.sort(key=lambda t: -t["level"] * t["size"])
    wait = {"skipped": True} if args.no_wait else de.wait_for_cpu()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.time()
    res: dict = {L: {} for L in plan}
    with ctx.Pool(processes=int(args.workers)) as pool:
        for r in pool.imap_unordered(_dk_chunk, tasks):
            res[r["level"]][r["chunk"]] = r
    wall = time.time() - t0
    for L, pl in plan.items():
        rows = [res[L][i] for i in range(len(pl["sizes"]))]
        C = np.array([r["counts"] for r in rows], np.int64)
        surv = int(sum(r["survivors"] for r in rows))
        assert int(C.sum()) + surv == walkers, "walker bookkeeping"
        # jackknife groups = blocks of consecutive chunks (at most --groups groups)
        G = min(int(args.groups), C.shape[0])
        blocks = np.array_split(np.arange(C.shape[0]), G)
        Cg = np.array([C[b].sum(0) for b in blocks], np.int64)
        sg = np.array([sum(pl["sizes"][i] for i in b) for b in blocks], np.int64)
        npz = out / f"level{L}.npz"
        np.savez_compressed(npz, counts=Cg, sizes=sg)
        meta = {"level": L, "dt": pl["spec"].dt, "steps": pl["spec"].steps(), "walkers": walkers,
                "chunk": chunk, "sizes": pl["sizes"], "jackknife_groups": int(G),
                "group_sizes": sg.tolist(), "tag": int(args.tag),
                "replicate": int(args.replicate), "seed_entropy": pl["entropy"],
                "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
                "simulator": "exact_m_prr_upgrade_core.simulate_chunk_general (EM + end-of-step Doi "
                             "kill, walker compaction)",
                "survivors_at_tmax": surv,
                "kill_probability_max": max(r["kill_probability_max"] for r in rows),
                "cpu_seconds": sum(r["seconds"] for r in rows), "wall_seconds_all_levels": wall,
                "workers": int(args.workers), "cpu_wait": wait, "npz": npz.name,
                "npz_sha256": _sha256(npz), "design": _design_payload(d)}
        de.write_json(out / f"level{L}.json", meta)
    return out


# ============================================================================
# analyze: functionals per level, Richardson, total error
# ============================================================================

QUANTITIES = ("peaks", "valleys", "masses", "ratios", "total_mass")


def _functionals_se(law, h, cuts0, mass_cuts="valley") -> dict:
    """Scalar h: estimator functionals (de.functionals_with_se).  Vector h (one bandwidth per passage, as in
    the SAA designs of fb_v2_saa_newton / V2_design/designs/*.json): the same per-passage functionals the
    design was solved with (fb_v2_saa_newton.analyse, valley cuts), mapped to this module's keys."""
    if np.ndim(h) == 0:
        return de.functionals_with_se(law, float(h), cuts0, mass_cuts=mass_cuts)
    import fb_v2_saa_newton as sn
    r = sn.analyse(law, np.asarray(h, float), cuts0, jackknife=True)
    r["se"]["total_mass"] = r["se"]["yield"]
    pk, cu = np.asarray(r["peaks"], float), np.asarray(r["cuts"], float)
    r["peaks_interior"] = bool(np.all((pk > cu[:-1]) & (pk < cu[1:])))
    r["peaks_ok"] = True                     # sn.functionals raises DesignMapError otherwise
    return r


def level_functionals(run: Path, L: int, d: dict) -> dict:
    meta = json.loads((run / f"level{L}.json").read_text())
    with np.load(run / meta["npz"]) as z:
        C = z["counts"].astype(float)
        sizes = z["sizes"].astype(float)
    spec = de.path_spec(d["eps"], dt=meta["dt"], tmax=float(d.get("tmax", 4.0)))
    law = de.Law(spec, d["_design"], fk.step_times(spec.dt, spec.steps()), C.sum(0) / sizes.sum(),
                 None, C, None, sizes, {"kind": "direct_kill"})
    cuts0 = d.get("cuts0") or de.g_clock_cuts(d["_design"], spec)
    r = _functionals_se(law, d["h"], cuts0, d.get("mass_cuts", "valley"))
    return {"dt": meta["dt"], "walkers": meta["walkers"], "chunks": len(meta["sizes"]),
            "jackknife_groups": int(sizes.size),
            "cuts0": cuts0, "cuts": r["cuts"], "peaks_interior": r["peaks_interior"],
            "peaks_ok": r["peaks_ok"], **{q: r[q] for q in QUANTITIES},
            "se": {q: r["se"][q] for q in QUANTITIES}, "survivors_at_tmax": meta["survivors_at_tmax"],
            "npz_sha256": meta["npz_sha256"], "seed_entropy": meta["seed_entropy"]}


def richardson(Q: dict, SE: dict) -> dict:
    """Q, SE: {level: array}; levels 1, 2, 4 (dt, dt/2, dt/4)."""
    q1, q2, q4 = (np.asarray(Q[L], float) for L in (1, 2, 4))
    s1, s2, s4 = (np.asarray(SE[L], float) for L in (1, 2, 4))
    R1 = 2 * q4 - q2
    R1c = 2 * q2 - q1
    R2 = (8 * q4 - 6 * q2 + q1) / 3.0
    se_R1 = np.sqrt(4 * s4**2 + s2**2)
    se_R2 = np.sqrt(64 * s4**2 + 36 * s2**2 + s1**2) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (q1 - q2) / (q2 - q4)
    out = {"R1": R1, "R1_se": se_R1, "R1_coarse": R1c, "R2": R2, "R2_se": se_R2,
           "order_ratio": rho, "bias_continuum": R2 - R1,
           "total_error_continuum": np.abs(R2 - R1) + 2 * se_R1, "levels": {}}
    for L, q, s in ((1, q1, s1), (2, q2, s2), (4, q4, s4)):
        b = q - R1
        out["levels"][L] = {"value": q, "se": s, "bias": b, "total_error": np.abs(b) + 2 * s}
    return out


def _target_checks(values, te, target, tol_floor: float) -> dict:
    v, t, e = (np.asarray(x, float) for x in (values, target, te))
    dev = v - t
    lim = np.maximum(tol_floor, e)
    return {"deviation": dev, "allowed": lim, "pass": (np.abs(dev) <= lim).tolist(),
            "all_pass": bool(np.all(np.abs(dev) <= lim))}


def cmd_analyze(args) -> dict:
    run = Path(args.run)
    d = load_design(run / "design.json")
    levels = sorted(int(p.stem[5:]) for p in run.glob("level*.json"))
    per = {L: level_functionals(run, L, d) for L in levels}
    out = {"item": "V2 TEP", "driver": HERE.name, "label": d["label"], "design": _design_payload(d),
           "h": d["h"], "levels": per,
           "definitions": {
               "functionals": "fb_v2_design_estimator smooth functionals (f_h maximisers/minimisers, "
                              "masses on f_h-valley cuts), same h at every dt",
               "se": "delete-one-chunk jackknife",
               "R1": "2 Q(dt/4) - Q(dt/2)", "R2": "(8 Q(dt/4) - 6 Q(dt/2) + Q(dt))/3",
               "total_error_level": "|Q(dt_L) - R1| + 2 SE(Q(dt_L))",
               "total_error_continuum": "|R2 - R1| + 2 SE(R1)",
               "target_rule": "peaks: |Q - T| <= max(0.01, TE); masses/ratios: |Q - target| <= TE"}}
    if all(L in per for L in (1, 2, 4)):
        out["richardson"] = {q: richardson({L: per[L][q] for L in (1, 2, 4)},
                                           {L: per[L]["se"][q] for L in (1, 2, 4)}) for q in QUANTITIES}
        tg = d.get("targets") or {}
        checks = {}
        for q, floor in (("peaks", 0.01), ("ratios", 0.0), ("masses", 0.0)):
            if q in tg and tg[q] is not None:
                tv = np.asarray(tg[q], float)
                n = tv.size
                R = out["richardson"][q]
                checks[q] = {
                    "target": tv,
                    "design_dt": _target_checks(R["levels"][1]["value"][:n], R["levels"][1]["total_error"][:n],
                                                tv, floor),
                    "continuum": _target_checks(R["R1"][:n], R["total_error_continuum"][:n], tv, floor)}
        if checks:
            out["target_checks"] = checks
        fp = d.get("fk_prediction")
        if fp:
            comp = {}
            for q in ("peaks", "masses", "ratios"):
                if q in fp:
                    v = np.asarray(per[1][q], float)[:len(fp[q])]
                    s = np.asarray(per[1]["se"][q], float)[:len(fp[q])]
                    sf = np.asarray((fp.get("se") or {}).get(q, np.zeros(len(fp[q]))), float)
                    comp[q] = {"dk_level1": v, "fk": fp[q],
                               "z": ((v - np.asarray(fp[q])) / np.sqrt(s**2 + sf**2)).tolist()}
            out["fk_vs_dk_at_design_dt"] = comp
    de.write_json(run / f"tep_{d['label']}.json", out)
    return out


# ============================================================================
# Isambard 3: sbatch script, md5-verified push / fetch, submit
# ============================================================================


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ssh(cmd: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", REMOTE_HOST, cmd],
                          capture_output=True, text=True, timeout=timeout)


def _rsync(src: list, dst: str, timeout: float = 600.0) -> None:
    r = subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20", *src, dst],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()[-400:]}")


def sbatch_text(label: str, *, walkers: int, chunk: int, workers: int, dt: float, levels: str,
                tag: int, replicate: int, hours: float, job: str | None = None) -> str:
    hours = min(float(hours), 4.0)                      # house rule: each node-job <= 4 h
    hh = int(hours)
    mm = int(round((hours - hh) * 60))
    run = f"$REPORT/artifacts/data/exact_m_fixed_budget/V2_TEP/{label}"
    return f"""#!/bin/bash
#SBATCH --job-name={job or 'tep_' + label[:40]}
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={workers}
#SBATCH --time={hh:02d}:{mm:02d}:00
#SBATCH --output=%x-%j.out
# fb_v2_tep.py generated script (uplift2 TEP).  One node-job, <= 4 h.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO=$HOME/valley-k-small
REPORT=$REPO/research/reports/encounter_multimodal_prr
CODE=$REPORT/code_v2sync
PY=$REPO/.venv311/bin/python
cd "$CODE"
md5sum -c MD5SUMS                       # synced inputs must match the local copies
mkdir -p "{run}"
"$PY" fb_v2_tep.py run --design "designs/{label}.json" --walkers {walkers} --chunk {chunk} \\
    --workers {workers} --dt {dt!r} --levels {levels} --tag {tag} --replicate {replicate} \\
    --out "{run}" --no-wait
"$PY" fb_v2_tep.py analyze --run "{run}"
cd "{run}" && md5sum design.json level*.json level*.npz tep_*.json > MD5SUMS.out
"""


def cmd_sbatch(args) -> Path:
    d = load_design(args.design)
    txt = sbatch_text(d["label"], walkers=int(float(args.walkers)), chunk=int(float(args.chunk)),
                      workers=int(args.workers), dt=float(args.dt), levels=args.levels,
                      tag=int(args.tag), replicate=int(args.replicate), hours=float(args.hours),
                      job=args.job)
    out = TEP_ROOT / "_sbatch" / f"tep_{d['label']}.sbatch"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    r = subprocess.run(["bash", "-n", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch script syntax: {r.stderr}")
    print(f"[sbatch] {out}", flush=True)
    return out


def cmd_push(args) -> dict:
    """rsync the code files and the design JSON to REMOTE_CODE and md5-verify them there."""
    d = load_design(args.design)
    stage = TEP_ROOT / "_sbatch" / "stage"
    (stage / "designs").mkdir(parents=True, exist_ok=True)
    dj = stage / "designs" / f"{d['label']}.json"
    dj.write_text(json.dumps(de._jsonable(_design_payload(d)), indent=1))
    files = [HERE.parent / f for f in SYNC_FILES]
    lines = [f"{_md5(p)}  {p.name}" for p in files] + [f"{_md5(dj)}  designs/{dj.name}"]
    (stage / "MD5SUMS").write_text("\n".join(lines) + "\n")
    r = _ssh(f"mkdir -p {REMOTE_CODE}/designs")
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed: {r.stderr.strip()[-400:]}")
    _rsync([str(p) for p in files] + [str(stage / "MD5SUMS")], f"{REMOTE_HOST}:{REMOTE_CODE}/")
    _rsync([str(dj)], f"{REMOTE_HOST}:{REMOTE_CODE}/designs/")
    r = _ssh(f"cd {REMOTE_CODE} && md5sum -c MD5SUMS")
    ok = r.returncode == 0 and r.stdout.count(": OK") == len(lines)
    print(r.stdout.strip(), flush=True)
    if not ok:
        raise RuntimeError(f"remote md5 verification failed: {r.stdout} {r.stderr}")
    return {"pushed": [p.name for p in files] + [f"designs/{dj.name}"], "md5_verified": ok}


def cmd_submit(args) -> dict:
    info = cmd_push(args)
    sb = cmd_sbatch(args)
    _rsync([str(sb)], f"{REMOTE_HOST}:{REMOTE_CODE}/")
    r = _ssh(f"cd {REMOTE_CODE} && sbatch {sb.name}")
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr.strip()[-400:]}")
    info["sbatch"] = r.stdout.strip()
    print(info["sbatch"], flush=True)
    return info


def cmd_fetch(args) -> dict:
    local = TEP_ROOT / args.label
    local.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_TEP}/{args.label}/"
    _rsync([f"{REMOTE_HOST}:{remote}"], str(local) + "/")
    sums = (local / "MD5SUMS.out").read_text().split("\n")
    bad = []
    for ln in sums:
        if not ln.strip():
            continue
        h, name = ln.split(None, 1)
        if _md5(local / name.strip()) != h:
            bad.append(name)
    if bad:
        raise RuntimeError(f"md5 mismatch after fetch: {bad}")
    print(f"[fetch] {local}: {len(sums) - 1} files md5-verified", flush=True)
    return {"fetched": str(local), "md5_verified": True}


# ============================================================================
# dkfd: gradient check against common-random-number direct-kill differences
# ============================================================================


def _direction(th: de.Design, spec, param: str) -> np.ndarray:
    """Raw-parameter tangent vector of a design coordinate at th."""
    m = th.m
    v = np.zeros(th.K)
    kind, j = param[0], (int(param[1:]) - 1 if len(param) > 1 else None)
    if param == "B":
        v[2 * m] = 1.0
    elif kind == "c":
        v[j] = 1.0
    elif kind == "t":
        v[j] = float(de.dc_dt(th.times(spec)[j], spec))
    elif kind == "w":
        v[m + j] = 1.0
    elif kind == "u":
        v[m + j] = 1.0
        v[2 * m - 1] = -1.0
    else:
        raise ValueError(param)
    return v


def _moved(th: de.Design, spec, param: str, step: float) -> de.Design:
    c, w, B = th.c.copy(), th.w.copy(), th.B
    m = th.m
    kind, j = param[0], (int(param[1:]) - 1 if len(param) > 1 else None)
    if param == "B":
        B += step
    elif kind == "c":
        c[j] += step
    elif kind == "t":
        c[j] = float(de.mu(th.times(spec)[j] + step, spec))
    elif kind == "w":
        w[j] += step
    elif kind == "u":
        w[j] += step
        w[m - 1] -= step
    return de.Design(c, w, B)


def _paired_diff(lp: de.Law, lm: de.Law, h: float, cuts0, step: float) -> dict:
    """Central-difference quotient of the functionals with a paired delete-one-group jackknife."""
    keys = ("peaks", "masses", "ratios")
    fp = de._functionals(lp.p, None, lp.t, h, cuts0)
    fm = de._functionals(lm.p, None, lm.t, h, cuts0)
    full = {k: (np.asarray(fp[k]) - np.asarray(fm[k])) / (2 * step) for k in keys}
    G = lp.sizes.size
    reps = {k: [] for k in keys}
    for g in range(G):
        pp, _ = lp.loo(g)
        pm, _ = lm.loo(g)
        a = de._functionals(pp, None, lp.t, h, cuts0, anchors=fp)
        b = de._functionals(pm, None, lm.t, h, cuts0, anchors=fm)
        for k in keys:
            reps[k].append((np.asarray(a[k]) - np.asarray(b[k])) / (2 * step))
    se = {k: np.sqrt((G - 1) / G * np.sum((np.asarray(v) - np.mean(v, 0)) ** 2, axis=0))
          for k, v in reps.items()}
    return {"value": full, "se": se}


def cmd_dkfd(args) -> dict:
    d = load_design(args.design)
    th = d["_design"]
    spec = de.path_spec(d["eps"])
    h = float(d["h"])
    step = float(args.step)
    cuts0 = d.get("cuts0") or de.g_clock_cuts(th, spec)
    thp, thm = _moved(th, spec, args.param, step), _moved(th, spec, args.param, -step)
    walkers, chunk = int(float(args.walkers)), int(float(args.chunk))
    sizes = [chunk] * (walkers // chunk) + ([walkers % chunk] if walkers % chunk else [])
    ent = level_entropy(d, tag=de.TAGS["DKFD"], replicate=int(args.replicate), level=1, dt=spec.dt,
                        walkers=walkers, chunk=chunk) + [_q(step), sum(map(ord, args.param))]
    kids = np.random.SeedSequence(ent).spawn(len(sizes))
    tasks = [{"spec": spec.to_dict(), "size": sz, "seedseq": k, "chunk": i,
              "designs": [thm.to_dict(), th.to_dict(), thp.to_dict()]} for i, (sz, k) in enumerate(zip(sizes, kids))]
    wait = de.wait_for_cpu()
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=min(int(args.workers), de.MAX_LOCAL_WORKERS)) as pool:
        res = sorted(pool.map(_crn_chunk, tasks), key=lambda r: r["chunk"])
    dk_wall = time.time() - t0
    laws = []
    G = min(int(args.groups), len(sizes))
    blocks = np.array_split(np.arange(len(sizes)), G)          # jackknife groups of chunks
    for i in range(3):
        Cc = np.array([r["res"][i]["counts"] for r in res], float)
        C = np.array([Cc[b].sum(0) for b in blocks])
        sz = np.array([sum(sizes[j] for j in b) for b in blocks], float)
        laws.append(de.Law(spec, None, fk.step_times(spec.dt, spec.steps()), C.sum(0) / sz.sum(), None,
                           C, None, sz, {"kind": "direct_kill_crn"}))
    dk = _paired_diff(laws[2], laws[0], h, cuts0, step)
    with de.Engine(spec, int(float(args.fk_paths)), tag=de.TAGS["NU_A"], workers=int(args.workers),
                   group=int(float(args.fk_group)), wait=False) as eng:
        lm, l0, lp = eng.evaluate([thm, th, thp])
    fkd = _paired_diff(lp, lm, h, cuts0, step)
    r0 = de.analyse(l0, h, cuts0=cuts0)
    v = _direction(th, spec, args.param)
    grad = {k: np.asarray(r0["jac"][k]) @ v for k in ("peaks", "masses", "ratios")}
    grad_se = {k: np.asarray(r0["se_jac"][k]) @ np.abs(v) for k in ("peaks", "masses", "ratios")}
    z = {}
    for k in dk["value"]:
        a, b = np.asarray(dk["value"][k]), np.asarray(fkd["value"][k])
        sa, sb = np.asarray(dk["se"][k]), np.asarray(fkd["se"][k])
        resolved = np.maximum(np.abs(a), np.abs(b)) > 1e-6          # else both zero at resolution
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = (a - b) / np.sqrt(sa**2 + sb**2)
        z[k] = [float(v) if r else None for v, r in zip(zz, resolved)]
    out = {"item": "V2_NU_A gradient check vs CRN direct kill", "driver": HERE.name,
           "label": d["label"], "param": args.param, "step": step, "h": h, "cuts0": cuts0,
           "design": _design_payload(d), "dk_walkers": walkers, "dk_chunk": chunk, "dk_seed_entropy": ent,
           "dk_jackknife_groups": G,
           "dk_simulator": "fb_v2_tep.simulate_chunk_crn (no compaction; common paths and kill coins "
                           "for theta-, theta, theta+)",
           "fk_paths": int(float(args.fk_paths)), "fk_engine": eng.meta_brief(),
           "dk_central_difference": dk, "fk_central_difference": fkd,
           "fk_pathwise_gradient": grad, "fk_pathwise_gradient_se_bound": grad_se,
           "z_dk_minus_fk_difference": z,
           "z_note": "null = both difference quotients below 1e-6 (unaffected coordinate)",
           "nonlinearity_fk_diff_minus_grad": {k: (fkd["value"][k] - grad[k]).tolist() for k in grad},
           "dk_wall_seconds": dk_wall, "cpu_wait": wait}
    de.write_json(de.OUT_DIR / f"dkfd_{d['label']}_{args.param}.json", out)
    return out


# ============================================================================
# smoke + CLI
# ============================================================================


def n11_design_json(cell: str = "m2_eps0.1_B8", name: str = "nwt2") -> Path:
    """Write an N11 design as a TEP design JSON (targets = N11 targets, equal ratios)."""
    D = json.loads((de.FB_DATA / "N11_shift_compensation" / "n11_designs.json").read_text())
    c = D["cells"][cell]
    dd = c["designs"][name]
    m = int(c["cell"]["m"])
    out = {"label": f"n11_{cell}_{name}", "m": m, "eps": float(c["cell"]["eps"]),
           "B": float(c["cell"]["B"]), "c": dd["centres_z"], "w": dd["w"], "h": 0.02,
           "targets": {"peaks": list(c["targets"]), "ratios": [1.0 / m] * m},
           "source": f"N11_shift_compensation/n11_designs.json cells.{cell}.designs.{name}"}
    p = TEP_ROOT / "_designs" / f"{out['label']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return p


def cmd_smoke(args) -> dict:
    p = n11_design_json(args.cell, args.name)
    ns = argparse.Namespace(design=str(p), walkers=args.walkers, chunk=args.chunk, workers=args.workers,
                            dt=args.dt, levels="1,2,4", tag=TAG_TEP, replicate=int(args.replicate),
                            out=str(TEP_ROOT / f"smoke_{json.loads(p.read_text())['label']}"),
                            no_wait=False, force=args.force, groups="100")
    run = cmd_run(ns)
    res = cmd_analyze(argparse.Namespace(run=str(run)))
    ns2 = argparse.Namespace(design=str(p), walkers="1e7", chunk="5e3", workers="144", dt=args.dt,
                             levels="1,2,4", tag=TAG_TEP, replicate=0, hours="1", job=None)
    res["sbatch_script"] = str(cmd_sbatch(ns2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_run(s, walkers="2e5", chunk="2.5e4", workers="3"):
        s.add_argument("--design", required=True)
        s.add_argument("--walkers", default=walkers)
        s.add_argument("--chunk", default=chunk)
        s.add_argument("--workers", default=workers)
        s.add_argument("--dt", default=str(fk.base.DEFAULT_DT))
        s.add_argument("--levels", default="1,2,4")
        s.add_argument("--tag", type=int, default=TAG_TEP)
        s.add_argument("--replicate", type=int, default=0)

    s = sub.add_parser("run")
    common_run(s)
    s.add_argument("--groups", default="100")
    s.add_argument("--out", default=None)
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("analyze")
    s.add_argument("--run", required=True)
    for nm in ("sbatch", "push", "submit"):
        s = sub.add_parser(nm)
        common_run(s, walkers="1e7", chunk="5e3", workers="144")
        s.add_argument("--hours", default="2")
        s.add_argument("--job", default=None)
    s = sub.add_parser("fetch")
    s.add_argument("--label", required=True)
    s = sub.add_parser("dkfd")
    s.add_argument("--design", required=True)
    s.add_argument("--param", default="t2")
    s.add_argument("--step", default="0.02")
    s.add_argument("--walkers", default="1e6")
    s.add_argument("--chunk", default="5e3")
    s.add_argument("--groups", default="50")
    s.add_argument("--workers", default="3")
    s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--fk-paths", default="2e5")
    s.add_argument("--fk-group", default="1000")
    s = sub.add_parser("smoke")
    s.add_argument("--cell", default="m2_eps0.1_B8")
    s.add_argument("--name", default="nwt2")
    s.add_argument("--walkers", default="3e4")
    s.add_argument("--chunk", default="5e3")
    s.add_argument("--workers", default="3")
    s.add_argument("--dt", default=str(fk.base.DEFAULT_DT))
    s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    fn = {"run": cmd_run, "analyze": cmd_analyze, "sbatch": cmd_sbatch, "push": cmd_push,
          "submit": cmd_submit, "fetch": cmd_fetch, "dkfd": cmd_dkfd, "smoke": cmd_smoke}[args.cmd]
    res = fn(args)
    if isinstance(res, dict):
        brief = {k: res[k] for k in ("target_checks", "z_dk_minus_fk_difference", "sbatch_script",
                                     "md5_verified", "sbatch") if k in res}
        print(json.dumps(de._jsonable(brief), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
