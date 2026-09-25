#!/usr/bin/env python3
"""Coupled time-step ladder for the Total-Error Protocol (uplift2 EXECUTION_SPEC section 3; NU-E).

fb_v2_tep.py runs the three direct-kill levels dt, dt/2, dt/4 with INDEPENDENT seeds, so its
Richardson bias estimates Q(dt) - R1 are differences of independent noisy estimates and, at 1e6
walkers, are dominated by Monte Carlo noise (order ratios of either sign).  This driver runs the
SAME three direct-kill laws on common random numbers:

  * one set of initial conditions per walker, shared by the three levels;
  * Brownian coupling: Gaussian increments are drawn on the finest grid dt/4; level dt/2 uses
    (xi_a + xi_b)/sqrt(2), level dt uses (xi_a + xi_b + xi_c + xi_d)/2 (each a standard normal);
  * kill coupling: one Exp(1) threshold E per walker; at level L the walker is killed at the end of
    the first step n with sum_{k<=n} kappa(Z_k) dt_L >= E (hazard accumulated only in contact).

Per level this is exactly the production law of exact_m_prr_upgrade_core.simulate_chunk_general
(Euler--Maruyama pair, end-of-step Doi kill with probability 1 - exp(-kappa dt), since
P(E > H_n | E > H_{n-1}) = exp(-(H_n - H_{n-1}))), only the random-number consumption differs.
Groups (jackknife blocks) contain the same walkers at every level, so the delete-one-group
jackknife of any linear combination of levels (R1, R2, bias) is PAIRED and resolves the time-step
bias far below the single-level noise.

Definitions (as fb_v2_tep.py; Q(dt) = Q0 + a dt + ...):
    R1 = 2 Q(dt/4) - Q(dt/2),  R2 = (8 Q(dt/4) - 6 Q(dt/2) + Q(dt))/3,  rho = (Q1 - Q2)/(Q2 - Q4)
    level L:    bias_L = Q_L - R1,    TE_L = |bias_L| + 2 SE(Q_L)
    continuum:  bias_c = R2 - R1,     TE_c = |bias_c| + 2 SE(R1)
    SEs: paired delete-one-group jackknife over walker groups (common to all levels).
Target rule (spec acceptance (c)): peaks |Q - T| <= max(0.01, TE); ratios/masses |Q - target| <= TE.

Subcommands
    run      --design D.json [--walkers 1e6 --chunk 1e4 --workers 3 --groups 100 --tag 215 --replicate 0
              --out DIR --no-wait]                        -> DIR/level{1,2,4}.npz/.json (paired groups)
    analyze  --run DIR [--B 400 --seed 0]                 -> DIR/tep_ladder_<label>.json (paired bootstrap)
    analyze-indep --run DIR (an fb_v2_tep.py run dir)    -> DIR/tep_boot_<label>.json (unpaired bootstrap)
    selftest                                              small consistency checks (see cmd_selftest)
Seeds: base 20260923, tag 215 (NU-E coupled ladder), entropy = fb_v2_tep.level_entropy(..., level=0),
disjoint from every design ensemble (tags 201-206, fk.path_entropy layout) and from fb_v2_tep runs
(tags 211-214, level >= 1).
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import fb_v2_tep as tep  # noqa: E402

de, fk, core = tep.de, tep.fk, tep.core
TAG_LADDER = 215
LEVELS = (1, 2, 4)
QUANT = ("peaks", "valleys", "masses", "ratios", "yield")


# ============================================================================
# coupled simulator
# ============================================================================


def simulate_chunk_ladder(rng, count: int, *, eps: float, budget: float, weights, centres_z, dt: float,
                          steps: int, p, n_perp: int = 1) -> dict:
    """Direct kill at dt, dt/2, dt/4 on common random numbers.  steps = number of dt-steps (level 1).
    Returns per-level kill counts per step (level L has L*steps steps) and survivors."""
    mu_j = np.asarray(centres_z, float)
    w_arr = np.asarray(weights, float)
    slab_norm = 1.0 / (math.sqrt(2.0 * math.pi) * eps * p.rho)
    inv_two_var = 1.0 / (2.0 * (eps * p.rho) ** 2)
    rate_prefactor = budget / p.torus_w ** n_perp
    contact_sq = p.contact_a ** 2
    W = p.torus_w
    par = {}
    for L in LEVELS:
        h = dt / L
        par[L] = {"dt": h, "decay": 1.0 - p.gamma * h, "pull": p.gamma * p.z_bar * h,
                  "nz": eps * math.sqrt(p.d0 * h), "nr": 2.0 * eps * math.sqrt(p.d0 * h),
                  "norm": 1.0 / math.sqrt(4 // L)}
    # common initial conditions (same law as the production simulator)
    z_init = p.z0 + math.sqrt(eps ** 2 * p.d0 / (2.0 * p.gamma)) * rng.standard_normal(count)
    rpar_init = p.r_par0 + eps * p.u0 * rng.standard_normal(count)
    rperp_init = np.mod(p.r_perp0 + eps * p.sigma_perp0 * rng.standard_normal((n_perp, count)), W)
    E = rng.standard_exponential(count)
    st = {L: {"z": z_init.copy(), "rpar": rpar_init.copy(), "rperp": rperp_init.copy(),
              "H": np.zeros(count), "alive": np.ones(count, bool)} for L in LEVELS}
    acc = {L: np.zeros((2 + n_perp, count)) for L in (1, 2)}
    counts = {L: np.zeros(L * steps, np.int64) for L in LEVELS}
    kpmax = 0.0
    n_all = count
    for k in range(4 * steps):
        if n_all == 0:
            break
        xi = rng.standard_normal((2 + n_perp, n_all))
        acc[1] += xi
        acc[2] += xi
        todo = [4] + ([2] if (k + 1) % 2 == 0 else []) + ([1] if (k + 1) % 4 == 0 else [])
        for L in todo:
            q, s = par[L], st[L]
            if L == 4:
                g = xi
            else:
                g = acc[L] * q["norm"]
            z, rpar, rperp = s["z"], s["rpar"], s["rperp"]
            z *= q["decay"]
            z += q["pull"]
            z += q["nz"] * g[0]
            rpar *= q["decay"]
            rpar += q["nr"] * g[1]
            rperp += q["nr"] * g[2:]
            np.mod(rperp, W, out=rperp)
            if L != 4:
                acc[L][:] = 0.0
            if rate_prefactor == 0.0:
                continue
            pm = np.minimum(rperp, W - rperp)
            d2 = rpar * rpar + np.sum(pm * pm, axis=0)
            ci = np.flatnonzero((d2 < contact_sq) & s["alive"])
            if ci.size == 0:
                continue
            zc = z[ci]
            kap = np.zeros(ci.size)
            for wj, cj in zip(w_arr, mu_j):
                dz = zc - cj
                kap += wj * np.exp(-dz * dz * inv_two_var)
            kap *= rate_prefactor * slab_norm
            kpmax = max(kpmax, float(-np.expm1(-kap.max() * q["dt"])))
            s["H"][ci] += kap * q["dt"]
            dead = ci[s["H"][ci] >= E[ci]]
            if dead.size:
                n_step = (k + 1) * L // 4          # 1-based step index of level L
                counts[L][n_step - 1] += dead.size
                s["alive"][dead] = False
        # compaction: drop walkers dead at every level
        if (k + 1) % 16 == 0:
            keep = st[1]["alive"] | st[2]["alive"] | st[4]["alive"]
            nk = int(keep.sum())
            if nk < 0.85 * n_all:
                for L in LEVELS:
                    s = st[L]
                    s["z"], s["rpar"], s["rperp"] = s["z"][keep], s["rpar"][keep], s["rperp"][:, keep]
                    s["H"], s["alive"] = s["H"][keep], s["alive"][keep]
                for L in (1, 2):
                    acc[L] = acc[L][:, keep]
                E = E[keep]
                n_all = nk
    return {"counts": counts, "survivors": {L: int(st[L]["alive"].sum()) for L in LEVELS},
            "kill_probability_max": kpmax}


def _ladder_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()
    out = simulate_chunk_ladder(rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]),
                                weights=tuple(task["w"]), centres_z=np.asarray(task["c"], float),
                                dt=spec.dt, steps=spec.steps(), p=spec.model(), n_perp=spec.n_perp)
    out.update(chunk=task["chunk"], seconds=time.perf_counter() - t0)
    return out


# ============================================================================
# run
# ============================================================================


def cmd_run(args) -> Path:
    d = tep.load_design(args.design)
    th = d["_design"]
    out = Path(args.out) if args.out else tep.TEP_ROOT / (d["label"] + "_ladder")
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f"level{L}.json").exists() for L in LEVELS) and not args.force:
        print(f"[run] {out} complete, skip", flush=True)
        return out
    de.write_json(out / "design.json", tep._design_payload(d))
    walkers, chunk = int(float(args.walkers)), int(float(args.chunk))
    dt0 = float(args.dt)
    # optional path-law override (N1's m = 5 cell starts from z0 = 8); default = production z0 = 4
    spec = de.path_spec(d["eps"], dt=dt0, tmax=float(d.get("tmax", 4.0)),
                        **({"z0": float(d["z0"])} if "z0" in d else {}))
    sizes = [chunk] * (walkers // chunk) + ([walkers % chunk] if walkers % chunk else [])
    ent = tep.level_entropy(d, tag=args.tag, replicate=args.replicate, level=0, dt=spec.dt,
                            walkers=walkers, chunk=chunk)
    kids = np.random.SeedSequence(ent).spawn(len(sizes))
    tasks = [{"spec": spec.to_dict(), "size": sz, "seedseq": kid, "B": th.B, "w": th.w.tolist(),
              "c": th.c.tolist(), "chunk": i} for i, (sz, kid) in enumerate(zip(sizes, kids))]
    wait = {"skipped": True} if args.no_wait else de.wait_for_cpu()
    import multiprocessing as mp
    t0 = time.time()
    res = {}
    if int(args.workers) <= 1:
        for tk in tasks:
            r = _ladder_chunk(tk)
            res[r["chunk"]] = r
    else:
        with mp.get_context("spawn").Pool(processes=int(args.workers)) as pool:
            for r in pool.imap_unordered(_ladder_chunk, tasks):
                res[r["chunk"]] = r
    wall = time.time() - t0
    rows = [res[i] for i in range(len(sizes))]
    G = min(int(args.groups), len(rows))
    blocks = np.array_split(np.arange(len(rows)), G)
    sg = np.array([sum(sizes[i] for i in b) for b in blocks], np.int64)
    for L in LEVELS:
        C = np.array([r["counts"][L] for r in rows], np.int64)
        surv = int(sum(r["survivors"][L] for r in rows))
        assert int(C.sum()) + surv == walkers, "walker bookkeeping"
        Cg = np.array([C[b].sum(0) for b in blocks], np.int64)
        npz = out / f"level{L}.npz"
        np.savez_compressed(npz, counts=Cg, sizes=sg)
        meta = {"level": L, "dt": spec.dt / L, "steps": spec.steps() * L, "walkers": walkers, "chunk": chunk,
                "sizes": sizes, "jackknife_groups": int(G), "group_sizes": sg.tolist(),
                "paired_groups": "group g holds the same walkers at every level (coupled ladder)",
                "tag": int(args.tag), "replicate": int(args.replicate), "seed_entropy": ent,
                "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
                "simulator": "fb_v2_tep_ladder.simulate_chunk_ladder (EM + end-of-step Doi kill; Brownian and "
                             "exponential-clock coupling of dt, dt/2, dt/4; per-level law = production)",
                "survivors_at_tmax": surv,
                "kill_probability_max": max(r["kill_probability_max"] for r in rows),
                "cpu_seconds_all_levels": sum(r["seconds"] for r in rows), "wall_seconds_all_levels": wall,
                "workers": int(args.workers), "cpu_wait": wait, "npz": npz.name,
                "npz_sha256": tep._sha256(npz), "design": tep._design_payload(d)}
        de.write_json(out / f"level{L}.json", meta)
    return out


# ============================================================================
# analyze: group bootstrap (paired across levels for the ladder)
# ============================================================================
#
# SEs.  The peak functional (maximiser of f_h in a basin) is not smooth in the data when a late
# peak is flat-topped: a delete-one-group jackknife that FOLLOWS the full-sample stationary point
# can jump to a neighbouring stationary point and then reports an absurd SE (seen in the 1e5 smoke
# run: 0.28 for a peak whose level values agree to 1e-3).  All SEs here are therefore group-bootstrap
# SEs (B resamples of the walker groups with replacement, fixed seed; every replicate recomputes the
# functionals with the SAME rule as the full sample: fresh maximiser of f_{h_j} in basin j of cuts0,
# valleys between the peaks, masses on the valley cuts).  For the coupled ladder the resampled group
# indices are the same at every level (paired), so bias estimates are resolved; for independent
# levels (fb_v2_tep.py runs) each level is resampled independently.


def load_levels(run: Path, d: dict) -> dict:
    out = {}
    for L in LEVELS:
        meta = json.loads((run / f"level{L}.json").read_text())
        with np.load(run / meta["npz"]) as z:
            C = z["counts"].astype(float)
            sizes = z["sizes"].astype(float)
        spec = de.path_spec(d["eps"], dt=meta["dt"], tmax=float(d.get("tmax", 4.0)))
        out[L] = {"C": C, "sizes": sizes, "t": fk.step_times(spec.dt, spec.steps()), "meta": meta,
                  "spec": spec}
    return out


def _hs(d: dict) -> np.ndarray:
    m = len(d["_design"].w)
    return np.asarray(d["h"], float) if np.ndim(d["h"]) else np.full(m, float(d["h"]))


def functional_vec(p: np.ndarray, t: np.ndarray, hs, cuts0) -> dict:
    import fb_v2_saa_newton as sn
    m = len(hs)
    try:
        r = sn.functionals(p, None, t, hs, cuts0)
        return {q: np.atleast_1d(np.asarray(r[q], float)) for q in QUANT} | {"_ok": True,
                                                                           "_cuts": r["cuts"]}
    except Exception:                                          # replicate left the chart
        nan = {"peaks": m, "valleys": m - 1, "masses": m, "ratios": m, "yield": 1}
        return {q: np.full(n, np.nan) for q, n in nan.items()} | {"_ok": False, "_cuts": None}


def bootstrap_levels(lv: dict, d: dict, *, paired: bool, B: int, seed: int) -> dict:
    hs = _hs(d)
    cuts0 = d.get("cuts0")
    full = {L: functional_vec(lv[L]["C"].sum(0) / lv[L]["sizes"].sum(), lv[L]["t"], hs, cuts0) for L in LEVELS}
    rng = np.random.default_rng(np.random.SeedSequence([de.BASE_SEED, 216, int(seed)]))
    G = {L: lv[L]["C"].shape[0] for L in LEVELS}
    reps = {L: {q: [] for q in QUANT} for L in LEVELS}
    n_fail = {L: 0 for L in LEVELS}
    for _ in range(B):
        if paired:
            idx0 = rng.integers(0, G[1], G[1])
        for L in LEVELS:
            idx = idx0 if paired else rng.integers(0, G[L], G[L])
            p = lv[L]["C"][idx].sum(0) / lv[L]["sizes"][idx].sum()
            r = functional_vec(p, lv[L]["t"], hs, cuts0)
            n_fail[L] += int(not r["_ok"])
            for q in QUANT:
                reps[L][q].append(r[q])
    return {"full": full, "reps": {L: {q: np.array(v) for q, v in reps[L].items()} for L in LEVELS},
            "n_fail": n_fail, "B": B, "paired": paired}


COMB = {"R1": {4: 2.0, 2: -1.0}, "R1_coarse": {2: 2.0, 1: -1.0}, "R2": {4: 8 / 3, 2: -2.0, 1: 1 / 3},
        "bias_c": {4: 2 / 3, 2: -1.0, 1: 1 / 3}, "d12": {1: 1.0, 2: -1.0}, "d24": {2: 1.0, 4: -1.0},
        "bias_1": {1: 1.0, 2: 1.0, 4: -2.0}, "bias_2": {2: 2.0, 4: -2.0}, "bias_4": {2: 1.0, 4: -1.0}}


def richardson_boot(bs: dict, q: str) -> dict:
    V = {L: bs["full"][L][q] for L in LEVELS}
    Rp = {L: bs["reps"][L][q] for L in LEVELS}
    o = {}
    for name, cf in COMB.items():
        val = sum(c * V[L] for L, c in cf.items())
        rep = sum(c * Rp[L] for L, c in cf.items())
        o[name] = (val, np.nanstd(rep, axis=0, ddof=1))
    se = {L: np.nanstd(Rp[L], axis=0, ddof=1) for L in LEVELS}
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = o["d12"][0] / o["d24"][0]
    res = {"R1": o["R1"][0], "R1_se": o["R1"][1], "R1_coarse": o["R1_coarse"][0], "R2": o["R2"][0],
           "R2_se": o["R2"][1], "bias_continuum": o["bias_c"][0], "bias_continuum_se": o["bias_c"][1],
           "total_error_continuum": np.abs(o["bias_c"][0]) + 2 * o["R1"][1],
           "d12": o["d12"][0], "d12_se": o["d12"][1], "d24": o["d24"][0], "d24_se": o["d24"][1],
           "order_ratio": rho, "levels": {}}
    for L in LEVELS:
        b, bse = o[f"bias_{L}"]
        res["levels"][L] = {"value": V[L], "se": se[L], "bias": b, "bias_se": bse,
                            "total_error": np.abs(b) + 2 * se[L],
                            "frac_reps_off_by_more_than_0.01": np.nanmean(np.abs(Rp[L] - V[L]) > 0.01, axis=0)}
    return res


def target_block(R: dict, tg: dict) -> dict:
    checks = {}
    for q, floor in (("peaks", 0.01), ("ratios", 0.0), ("masses", 0.0)):
        if q in tg and tg[q] is not None:
            tv = np.asarray(tg[q], float)
            n = tv.size
            lv1 = R[q]["levels"][1]
            checks[q] = {
                "target": tv,
                "design_dt": tep._target_checks(lv1["value"][:n], lv1["total_error"][:n], tv, floor),
                "continuum": tep._target_checks(R[q]["R1"][:n], R[q]["total_error_continuum"][:n], tv, floor),
                "z_design_dt_no_bias": ((lv1["value"][:n] - tv) / lv1["se"][:n]).tolist(),
                "z_continuum_bias_included": (np.abs(R[q]["R1"][:n] - tv)
                                              / np.maximum(R[q]["total_error_continuum"][:n] / 2, 1e-300)).tolist()}
    return checks


def protocol_p_count(C: np.ndarray, dt: float, walkers: int) -> dict:
    """covariance-aware protocol-P mode count (reclassify_covariance_aware.classify_both, bandwidth 0.04) of
    the 0.02-bin histogram of the kill times on the production window I = [0.5, 3.5] (same classifier as the
    paper's direct-kill counts); peaks outside I are not seen by it."""
    import reclassify_covariance_aware as Rc
    edges = fk.production_window_edges()
    kt = (np.arange(C.shape[1]) + 1) * dt
    idx = np.searchsorted(edges, kt, side="right") - 1
    idx[kt == edges[-1]] = edges.size - 2
    ok = (idx >= 0) & (idx < edges.size - 1)
    h = np.zeros(edges.size - 1)
    np.add.at(h, idx[ok], C.sum(0)[ok])
    cls = Rc.classify_both(h.astype(np.int64), edges, int(walkers), bandwidth=fk.base.DEFAULT_BANDWIDTH)
    return {"mode_count_covariance_aware": int(cls["mode_count_covariance_aware"]),
            "significant_times": cls.get("significant_times_covariance_aware")}


def analyze_run(run: Path, *, paired: bool, B: int, seed: int, tag_out: str) -> dict:
    d = tep.load_design(run / "design.json")
    lv = load_levels(run, d)
    t0 = time.time()
    bs = bootstrap_levels(lv, d, paired=paired, B=B, seed=seed)
    R = {q: richardson_boot(bs, q) for q in QUANT}
    import fb_v2_saa_newton as sn
    modes = {}
    for L in LEVELS:
        spec = lv[L]["spec"]
        law = de.Law(spec, d["_design"], lv[L]["t"], lv[L]["C"].sum(0) / lv[L]["sizes"].sum(), None,
                     lv[L]["C"], None, lv[L]["sizes"], {"kind": "direct_kill"})
        try:
            r0 = sn.functionals(law.p, None, law.t, _hs(d), d.get("cuts0"))
            modes[L] = sn.mode_check(law, _hs(d), r0)
        except Exception as exc:
            modes[L] = {"error": str(exc)}
        try:
            modes[L]["protocol_P"] = protocol_p_count(lv[L]["C"], lv[L]["meta"]["dt"], int(lv[L]["sizes"].sum()))
        except Exception as exc:
            modes[L]["protocol_P"] = {"error": str(exc)}
    out = {"item": "V2 NU-E TEP", "driver": HERE.name, "label": d["label"], "run_dir": run.name,
           "sample": "coupled ladder (paired levels)" if paired else "independent levels (fb_v2_tep.py run)",
           "design": tep._design_payload(d), "h": d["h"],
           "levels": {L: {"dt": lv[L]["meta"]["dt"], "walkers": lv[L]["meta"]["walkers"],
                          "groups": int(lv[L]["C"].shape[0]), "tag": lv[L]["meta"].get("tag"),
                          "seed_entropy": lv[L]["meta"].get("seed_entropy"),
                          "survivors_at_tmax": lv[L]["meta"]["survivors_at_tmax"],
                          "npz_sha256": lv[L]["meta"]["npz_sha256"], "cuts": bs["full"][L]["_cuts"],
                          "mode_check": modes[L], "bootstrap_failures": bs["n_fail"][L]} for L in LEVELS},
           "bootstrap": {"B": B, "seed_entropy": [de.BASE_SEED, 216, int(seed)], "paired": paired,
                         "rule": "resample walker groups with replacement; functionals recomputed with the "
                                 "full-sample rule (fresh f_h maximisers in the cuts0 basins)"},
           "definitions": {"R1": "2 Q(dt/4) - Q(dt/2)", "R2": "(8 Q(dt/4) - 6 Q(dt/2) + Q(dt))/3",
                           "total_error_level": "|Q(dt_L) - R1| + 2 SE(Q(dt_L))",
                           "total_error_continuum": "|R2 - R1| + 2 SE(R1)",
                           "target_rule": "peaks: |Q - T| <= max(0.01, TE); masses/ratios: |Q - target| <= TE"},
           "richardson": R}
    tg = d.get("targets") or {}
    out["target_checks"] = target_block(R, tg)
    out["all_pass_design_dt"] = bool(all(c["design_dt"]["all_pass"] for c in out["target_checks"].values()))
    out["all_pass_continuum"] = bool(all(c["continuum"]["all_pass"] for c in out["target_checks"].values()))
    out["exactly_m_modes_all_levels"] = bool(all(modes[L].get("exactly_m") for L in LEVELS))
    m = len(d["_design"].w)
    in_window = int(np.sum((np.asarray((d.get("targets") or {}).get("peaks", []), float) > 0.5)
                           & (np.asarray((d.get("targets") or {}).get("peaks", []), float) < 3.5)))
    out["protocol_P_count_by_level"] = {L: modes[L].get("protocol_P", {}).get("mode_count_covariance_aware")
                                        for L in LEVELS}
    out["protocol_P_expected_in_window"] = in_window
    out["protocol_P_count_equals_expected_all_levels"] = bool(all(out["protocol_P_count_by_level"][L] == in_window
                                                                  for L in LEVELS))
    fp = d.get("fk_prediction")
    if fp:
        comp = {}
        for q in ("peaks", "ratios"):
            if q in fp:
                n = len(fp[q])
                v, s = R[q]["levels"][1]["value"][:n], R[q]["levels"][1]["se"][:n]
                sf = np.asarray((fp.get("se") or {}).get(q, np.zeros(n)), float)
                comp[q] = {"dk_level1": v, "fk": fp[q],
                           "z": ((v - np.asarray(fp[q])) / np.sqrt(s ** 2 + sf ** 2)).tolist()}
        out["fk_vs_dk_at_design_dt"] = comp
        out["fk_prediction_source"] = fp.get("source")
    out["analysis_seconds"] = time.time() - t0
    de.write_json(run / f"{tag_out}_{d['label']}.json", out)
    return out


def cmd_analyze(args) -> dict:
    return analyze_run(Path(args.run), paired=True, B=int(args.B), seed=int(args.seed), tag_out="tep_ladder")


def cmd_analyze_indep(args) -> dict:
    return analyze_run(Path(args.run), paired=False, B=int(args.B), seed=int(args.seed), tag_out="tep_boot")

# ============================================================================
# selftest
# ============================================================================


def cmd_selftest(args) -> dict:
    """(1) bookkeeping; (2) per-level law vs the production simulator (same dt, independent seeds):
    binned kill-time histograms (0.05 bins) chi^2 and total killed; (3) coupling strength."""
    d = tep.load_design(args.design)
    th = d["_design"]
    spec = de.path_spec(d["eps"], dt=float(args.dt), tmax=float(d.get("tmax", 4.0)))
    n = int(float(args.walkers))
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([de.BASE_SEED, 299, 1, int(args.seed)])))
    t0 = time.time()
    lad = simulate_chunk_ladder(rng, n, eps=spec.eps, budget=th.B, weights=tuple(th.w), centres_z=th.c,
                                dt=spec.dt, steps=spec.steps(), p=spec.model(), n_perp=spec.n_perp)
    t_lad = time.time() - t0
    rep = {"walkers": n, "seconds_ladder": t_lad, "levels": {}}
    binw = 0.05
    for L in LEVELS:
        dtL = spec.dt / L
        rng2 = np.random.Generator(np.random.Philox(np.random.SeedSequence([de.BASE_SEED, 299, 2, L, int(args.seed)])))
        t1 = time.time()
        pr = core.simulate_chunk_general(rng2, n, eps=spec.eps, budget=th.B, weights=tuple(th.w),
                                         centres_z=th.c, dt=dtL, step_count=spec.steps() * L,
                                         p=spec.model(), n_perp=spec.n_perp)
        t_pr = time.time() - t1
        cl = lad["counts"][L]
        tl = (np.arange(cl.size) + 1) * dtL
        nb = int(round(spec.tmax / binw))
        hl = np.bincount(np.minimum((tl / binw - 1e-9).astype(int), nb - 1), weights=cl, minlength=nb)
        hp = np.bincount(np.minimum((pr["kill_times"] / binw - 1e-9).astype(int), nb - 1), minlength=nb)
        sel = (hl + hp) > 20
        chi2 = float(np.sum((hl[sel] - hp[sel]) ** 2 / (hl[sel] + hp[sel])))
        pk = (cl.sum() + pr["kill_times"].size) / (2.0 * n)
        rep["levels"][L] = {"killed_ladder": int(cl.sum()), "killed_production": int(pr["kill_times"].size),
                            "z_killed": float((cl.sum() - pr["kill_times"].size)
                                              / math.sqrt(2 * n * pk * (1 - pk) + 1e-12)),
                            "chi2": chi2, "dof": int(sel.sum()), "seconds_production": t_pr,
                            "survivors_ladder": lad["survivors"][L], "survivors_production": pr["survivors"]}
    print(json.dumps(rep, indent=1, default=float))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--design", required=True)
    s.add_argument("--walkers", default="1e6")
    s.add_argument("--chunk", default="1e4")
    s.add_argument("--workers", default="3")
    s.add_argument("--groups", default="100")
    s.add_argument("--dt", default="1e-3")
    s.add_argument("--tag", type=int, default=TAG_LADDER)
    s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--out", default=None)
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--force", action="store_true")
    for nm in ("analyze", "analyze-indep"):
        s = sub.add_parser(nm)
        s.add_argument("--run", required=True)
        s.add_argument("--B", default="400")
        s.add_argument("--seed", default="0")
    s = sub.add_parser("selftest")
    s.add_argument("--design", required=True)
    s.add_argument("--walkers", default="2e4")
    s.add_argument("--seed", default="0")
    s.add_argument("--dt", default="1e-3")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "analyze-indep":
        cmd_analyze_indep(args)
    elif args.cmd == "selftest":
        cmd_selftest(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
