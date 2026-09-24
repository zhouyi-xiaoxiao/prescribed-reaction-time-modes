#!/usr/bin/env python3
"""HPC item D (HPC_census): extreme-scale FK census of the three gated N9a cells.

N9a (fb_n9a_matching_census.py, 5e5 paths, tag 87) found the TH-6 pattern
(exactly m maxima, m-1 minima) at every resolution in 69/72 cases; the three
gated cells (m, eps, B) = (2, 0.05, 50), (3, 0.05, 20), (3, 0.05, 50) missed
the LAST maximum at some or all resolutions: there the late passage carries
mass 2.5e-5 / 3.2e-5 / 1.5e-6 and the FK weights are degenerate (few paths
avoid contact at the earlier passages).  This driver repeats the census with
>= 1e8 unkilled paths per ensemble, declared-grid accumulation (per-step exact
kill probability, O(1) memory), same model and time step as N9a
(Euler--Maruyama dt = 1e-3 = 0.02 eps, end-of-step Doi kill, production
parameters, equal weights, 'full' = gated kernel), paths run to t = 3.52
(the census window is [0.5, 3.5]).

Chunks of 5e4 paths are computed by exact_m_prr_fk_exact_law._run_chunk
(declared mode, unchanged numerics) and folded, as they arrive, into G = 400
round-robin path groups (chunk i -> group i mod G) held by the parent; the
per-chunk files are deleted after folding.  A checkpoint of the accumulators
(with the list of folded chunks) is written every 200 chunks, so a job can be
resumed / extended (more chunks = more paths, same streams).

Census: the N9a statistic (fb_n9a_matching_census.census_one): f' on grid
spacing h_g = k dt from first differences of the binned per-step law, batch
SE over the G groups, a point is significant when |f'|/SE > 5; every sign
change of the significant sign sequence is a critical point.  Resolutions
h_g / eps in {0.05, 0.1, 0.25, 0.5, 1, 2}.  Late mode: the last-basin
(after the last valley of G) maximum of the bandwidth-0.04 smoothed and of
the unsmoothed density on 0.02 bins, relative prominence (fb_n6 late_prominence)
with delete-one-group jackknife CI.  Decision per cell: 'resolvable_present'
(a significant + -> - sign change of f' in the last basin at some
resolution), 'resolvable_absent' (f' significantly negative on the whole last
basin at some resolution), else 'unresolved'.

Seeds: base 20260923, tag 98, replicate 0; entropy fk.path_entropy(spec, seed,
tag=98, replicate=0, chunk=50000) (spec: dt = 1e-3, eps = 0.05, production
path law); chunk i = SeedSequence(entropy).spawn(n_chunks_max = 40000)[i].

Usage (from code/):
    python3 fb_hpc_census.py simulate --cell m2_eps0.05 --paths 5e8 --workers 140 --deadline-min 300
    python3 fb_hpc_census.py analyze
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

import fb_hpc_common as hc  # noqa: E402
import exact_m_prr_fk_exact_law as fk  # noqa: E402
import fb_n9a_matching_census as n9a  # noqa: E402

ITEM = "census"
TAG = hc.HPC_TAGS[ITEM]
DT = 1e-3
TMAX = 3.52
CHUNK = 50_000
N_CHUNKS_MAX = 40_000            # stream length (2e9 paths); a run uses a prefix
GROUPS = 400
RES_LIST = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
CELLS = {
    "m2_eps0.05": dict(m=2, eps=0.05, budgets=(8.0, 20.0, 50.0), gated=(50.0,)),
    "m3_eps0.05": dict(m=3, eps=0.05, budgets=(8.0, 20.0, 50.0), gated=(20.0, 50.0)),
}
CHECKPOINT_EVERY = 200


def spec_for(cell: str) -> fk.EnsembleSpec:
    c = CELLS[cell]
    return fk.EnsembleSpec(m=c["m"], eps=c["eps"], dt=DT, tmax=TMAX)


def declared_setup(cell: str):
    spec = spec_for(cell)
    m = spec.centres().size
    w = [1.0 / m] * m
    variants = [fk.parse_variant("full", spec)]
    pairs = [{"variant": "full", "w": w}]
    decl = [{"B": float(B), "w": w, "variant": "full", "cap": None, "pair": 0,
             "label": f"full_B{B:g}"} for B in CELLS[cell]["budgets"]]
    return spec, variants, pairs, decl


def entropy(cell: str) -> list[int]:
    return fk.path_entropy(spec_for(cell), seed=hc.BASE_SEED, tag=TAG, replicate=0, chunk=CHUNK)


def acc_path(cell: str) -> Path:
    return hc.work_dir(ITEM) / f"{cell}_acc.npz"


def load_acc(cell: str) -> dict | None:
    p = acc_path(cell)
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k].copy() for k in d.files}


def save_acc(cell: str, acc: dict) -> None:
    p = acc_path(cell)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **acc)
    os.replace(tmp, p)


def new_acc(nd: int, steps: int) -> dict:
    return {"f_sum": np.zeros((nd, steps)), "f_sq": np.zeros((nd, steps)),
            "f_g": np.zeros((nd, GROUPS, steps)), "g_paths": np.zeros(GROUPS, np.int64),
            "mx": np.zeros((1, steps)), "mxx": np.zeros((1, steps)),
            "mv": np.zeros((1, steps)), "mvx": np.zeros((1, steps)),
            "done": np.zeros(0, np.int64), "runtime_sum": np.zeros(1),
            "max_step_unit_exposure": np.zeros(1)}


def cmd_simulate(args):
    cell = args.cell
    spec, variants, pairs, decl = declared_setup(cell)
    steps = spec.steps()
    n_target = int(float(args.paths))
    n_chunks = -(-n_target // CHUNK)
    if n_chunks > N_CHUNKS_MAX:
        raise ValueError("too many chunks for the declared stream length")
    ent = entropy(cell)
    kids = np.random.SeedSequence(ent).spawn(N_CHUNKS_MAX)
    acc = load_acc(cell) or new_acc(len(decl), steps)
    done = set(int(x) for x in acc["done"])
    tmpdir = hc.work_dir(ITEM) / f"tmp_{cell}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(n_chunks):
        if i in done:
            continue
        tasks.append({"spec": spec.to_dict(), "size": CHUNK, "seedseq": kids[i], "mode": "declared",
                      "variants": variants, "chunk": i, "file": str(tmpdir / f"chunk{i:05d}.npz"),
                      "mean_contact": {}, "declared": decl, "pairs": pairs, "batch": CHUNK})
    print(f"[census] {cell}: target {n_target:.3g} paths = {n_chunks} chunks; "
          f"{len(done)} already folded; {len(tasks)} to run; env {hc.env_info()}", flush=True)
    deadline = time.time() + 60 * args.deadline_min if args.deadline_min else None
    state = {"since": 0}
    t0 = time.time()

    def fold(r):
        f = Path(r["file"])
        g = int(r["chunk"]) % GROUPS
        with np.load(f) as d:
            acc["f_sum"] += d["f_sum"]
            acc["f_sq"] += d["f_sq"]
            acc["f_g"][:, g, :] += d["f_b"][:, 0, :]
            for k in ("mx", "mxx", "mv", "mvx"):
                acc[k] += d[k]
        acc["g_paths"][g] += int(r["size"])
        acc["done"] = np.append(acc["done"], int(r["chunk"]))
        acc["runtime_sum"][0] += float(r["runtime_seconds"])
        acc["max_step_unit_exposure"][0] = max(float(acc["max_step_unit_exposure"][0]),
                                               float(r.get("max_step_unit_exposure", 0.0)))
        f.unlink()
        state["since"] += 1
        if state["since"] >= CHECKPOINT_EVERY:
            save_acc(cell, acc)
            state["since"] = 0
            print(f"[census] {cell}: checkpoint, {acc['done'].size} chunks folded "
                  f"({acc['done'].size * CHUNK:.3g} paths), wall {time.time() - t0:.0f}s", flush=True)

    hc.run_pool(fk._run_chunk, tasks, args.workers, fold, label=f"census:{cell}", deadline=deadline)
    save_acc(cell, acc)
    meta = {"cell": cell, "entropy": ent, "tag": TAG, "base_seed": hc.BASE_SEED,
            "chunks_folded": int(acc["done"].size), "paths": int(acc["done"].size * CHUNK),
            "last_invocation_wall_s": time.time() - t0, "env": hc.env_info(),
            "declared": decl}
    (hc.work_dir(ITEM) / f"{cell}_meta.json").write_text(json.dumps(meta, indent=1, default=fk._json_default))
    print(f"[census] {cell}: done, {acc['done'].size} chunks = {acc['done'].size * CHUNK:.3g} paths, "
          f"wall {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def last_basin_decision(by_res: dict, m: int, t_lo: float, t_hi: float = fk.WINDOW[1]) -> dict:
    present, absent = [], []
    for res, c in by_res.items():
        late_max = [x for x in c["changes"] if x["kind"] == "max" and x["t"] > t_lo]
        if late_max:
            present.append({"res": res, "t": [x["t"] for x in late_max],
                            "z": [(x["z_before"], x["z_after"]) for x in late_max]})
        if c.get("_late_all_negative"):
            absent.append(res)
    decision = ("resolvable_present" if present and not absent else
                "resolvable_absent" if absent and not present else
                "contradictory" if present and absent else "unresolved")
    return {"decision": decision, "present_at": present, "monotone_decrease_resolved_at": absent}


def late_sign_profile(p_step, batch_p, t, dt, k, t_lo, t_hi=fk.WINDOW[1]):
    """Significance profile of f' on the last basin at spacing k dt (for the 'absent' test)."""
    steps = p_step.size
    nb = steps // k
    F = p_step[: nb * k].reshape(nb, k).sum(1) / (k * dt)
    Fb = batch_p[:, : nb * k].reshape(batch_p.shape[0], nb, k).sum(2) / (k * dt)
    tb = t[: nb * k].reshape(nb, k)
    tc = 0.5 * (tb[:, 0] - dt + tb[:, -1])
    d = np.diff(F) / (k * dt)
    db = np.diff(Fb, axis=1) / (k * dt)
    td = 0.5 * (tc[1:] + tc[:-1])
    se = db.std(axis=0, ddof=1) / math.sqrt(Fb.shape[0])
    msk = (td > t_lo) & (td <= t_hi)
    z = np.divide(d[msk], se[msk], out=np.zeros(int(msk.sum())), where=se[msk] > 0)
    return {"t": td[msk], "z": z, "all_negative": bool(z.size and np.all(z < -n9a.ZTHR)),
            "n_pos": int(np.sum(z > n9a.ZTHR)), "n_neg": int(np.sum(z < -n9a.ZTHR)),
            "n_undecided": int(np.sum(np.abs(z) <= n9a.ZTHR))}


def late_prominence_ci(p_step, f_g, g_paths, t, dt, after_time, bin_k=20, bandwidth=0.04):
    """Relative prominence of the last-basin maximum on 0.02 bins (k = 20 steps), smoothed
    (bandwidth 0.04, protocol smoothing) and unsmoothed; jackknife over groups."""
    import fb_n6_visibility_thresholds as n6
    steps = p_step.size
    nb = steps // bin_k
    edges_t = t[: nb * bin_k].reshape(nb, bin_k)
    tc = 0.5 * (edges_t[:, 0] - dt + edges_t[:, -1])
    win = (tc >= fk.WINDOW[0]) & (tc <= fk.WINDOW[1])
    bw = bin_k * dt

    def rp(p):
        mass = p[: nb * bin_k].reshape(nb, bin_k).sum(1)[win]
        dens = mass / bw
        sm = fk._gauss_smooth(dens, bandwidth / bw)
        rs, ts, ns = n6.late_prominence(sm, tc[win], after_time)
        ru, tu, nu = n6.late_prominence(dens, tc[win], after_time)
        return rs, (ts if ts is not None else np.nan), ns, ru, (tu if tu is not None else np.nan), nu

    full = rp(p_step)
    N = g_paths.sum()
    tot = p_step * N
    use = np.flatnonzero(g_paths > 0)
    jack = np.array([rp((tot - f_g[g]) / (N - g_paths[g])) for g in use])
    se = hc.jackknife_se(jack)
    return {"smoothed_bw0.04": {"rel_prominence": full[0], "time": full[1], "n_late_maxima": full[2],
                                "jackknife_se": float(se[0]),
                                "ci95": [full[0] - 1.96 * float(se[0]), full[0] + 1.96 * float(se[0])],
                                "time_se": float(se[1])},
            "unsmoothed_0.02": {"rel_prominence": full[3], "time": full[4], "n_late_maxima": full[5],
                                "jackknife_se": float(se[3]),
                                "ci95": [full[3] - 1.96 * float(se[3]), full[3] + 1.96 * float(se[3])],
                                "time_se": float(se[4])},
            "after_time": after_time, "groups_used": int(use.size)}


def analyze_cell(cell: str) -> dict | None:
    acc = load_acc(cell)
    if acc is None:
        return None
    spec, _, _, decl = declared_setup(cell)
    m = spec.centres().size
    eps = spec.eps
    steps = spec.steps()
    t = fk.step_times(DT, steps)
    N = int(acc["g_paths"].sum())
    use = acc["g_paths"] > 0
    targets = spec.times()
    valleys = fk.g_valley_times(fk.EnsembleSpec(m=m, eps=eps), [1.0 / m] * m)
    out = {"cell": cell, "m": m, "eps": eps, "dt": DT, "tmax": TMAX, "N_paths": N,
           "chunks_folded": int(acc["done"].size), "groups": int(use.sum()),
           "group_paths_min_max": [int(acc["g_paths"][use].min()), int(acc["g_paths"][use].max())],
           "core_hours": float(acc["runtime_sum"][0] / 3600.0),
           "seed": {"base_seed": hc.BASE_SEED, "tag": TAG, "replicate": 0, "entropy": entropy(cell),
                    "chunk": CHUNK, "stream": f"SeedSequence(entropy).spawn({N_CHUNKS_MAX})[i], i < n_chunks"},
           "targets": list(targets), "g_valleys": valleys, "items": {}}
    for idd, it in enumerate(decl):
        B = it["B"]
        p = acc["f_sum"][idd] / N
        batch_p = acc["f_g"][idd][use] / acc["g_paths"][use][:, None]
        ess = np.divide(acc["f_sum"][idd] ** 2, acc["f_sq"][idd], out=np.zeros(steps),
                        where=acc["f_sq"][idd] > 0)
        late = (t > valleys[-1]) & (t <= fk.WINDOW[1])
        by_res = {}
        for rres in RES_LIST:
            k = max(1, int(math.floor(rres * eps / DT + 1e-9)))
            c = n9a.census_one(t, p, batch_p, DT, k, m, targets, eps, valleys)
            prof = late_sign_profile(p, batch_p, t, DT, k, valleys[-1])
            c["_late_all_negative"] = prof["all_negative"]
            c["late_basin_sign_profile"] = {"n_pos": prof["n_pos"], "n_neg": prof["n_neg"],
                                            "n_undecided": prof["n_undecided"],
                                            "all_negative": prof["all_negative"]}
            by_res[f"{rres:g}"] = c
        dec = last_basin_decision(by_res, m, valleys[-1])
        for c in by_res.values():
            c.pop("_late_all_negative", None)
        late_se = float(np.std(batch_p[:, late].sum(1), ddof=1) / math.sqrt(batch_p.shape[0]))
        out["items"][it["label"]] = {
            "B": B, "gated_n9a_cell": B in CELLS[cell]["gated"],
            "counts_by_resolution": {r: [v["n_max"], v["n_min"]] for r, v in by_res.items()},
            "exactly_m_by_resolution": {r: v["exactly_m_pattern"] for r, v in by_res.items()},
            "any_EXTRA_PAIR": any(v["EXTRA_PAIR"] for v in by_res.values()),
            "decision_last_mode": dec,
            "mass_after_last_G_valley": float(p[late].sum()), "mass_after_last_G_valley_se": late_se,
            "window_mass": float(p[(t >= 0.5) & (t <= 3.5)].sum()),
            "per_step_ess_last_basin": {"median": float(np.median(ess[late])),
                                        "min": float(ess[late].min()), "max": float(ess[late].max())},
            "late_prominence": late_prominence_ci(p, acc["f_g"][idd][use],
                                                  acc["g_paths"][use], t, DT, valleys[-1]),
            "by_resolution": by_res}
        print(f"[census] {cell} B={B:g}: {out['items'][it['label']]['counts_by_resolution']} "
              f"decision {dec['decision']}", flush=True)
    return out


def cmd_analyze(args):
    res = {"item": "HPC_census (extreme-scale FK census of the gated N9a cells)", "driver": HERE.name,
           "env": hc.env_info(), "zthr": n9a.ZTHR, "resolutions_h_over_eps": list(RES_LIST),
           "model": "production EM dt=1e-3, end-of-step Doi kill, gated kernel 'full', equal weights",
           "n9a_reference": "artifacts/data/exact_m_fixed_budget/N9a/n9a_census.json (5e5 paths, tag 87)",
           "cells": {}}
    for cell in CELLS:
        r = analyze_cell(cell)
        if r is not None:
            res["cells"][cell] = r
    ref = fk.FB_DATA / "N9a" / "n9a_census.json"
    if ref.exists():
        d = json.loads(ref.read_text())
        cmp_ = {}
        for cell, r in res["cells"].items():
            for lab, it in r["items"].items():
                loc = d["fk"].get(cell, {}).get("items", {}).get(lab)
                if loc:
                    cmp_[f"{cell}/{lab}"] = {
                        "n9a_counts_by_resolution": {k: [v["n_max"], v["n_min"]]
                                                     for k, v in loc["by_resolution"].items()},
                        "n9a_status": loc["status"], "n9a_N": d["fk"][cell]["N"],
                        "hpc_counts_by_resolution": it["counts_by_resolution"],
                        "hpc_decision": it["decision_last_mode"]["decision"]}
        res["vs_n9a"] = cmp_
    hc.write_json(hc.out_dir(ITEM) / "hpc_census.json", res)
    print("[census] wrote", hc.out_dir(ITEM) / "hpc_census.json", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--cell", required=True, choices=list(CELLS))
    s.add_argument("--paths", default="1e8")
    s.add_argument("--workers", type=int, default=140)
    s.add_argument("--deadline-min", type=float, default=None)
    sub.add_parser("analyze")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
