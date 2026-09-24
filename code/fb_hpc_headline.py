#!/usr/bin/env python3
"""HPC item C (HPC_headline): 1e7-walker direct-kill confirmation of the headline cells.

The main text (sections/04_numerics.tex, "Designed allocations") quotes the
exact-law (FK, 5e5 paths) basin masses and protocol mode counts of equal
weights vs the max--min design w* at (m, eps, B) = (2, 0.1, 8) and (3, 0.1, 4),
and a 1e6-walker direct-kill confirmation of ten designed cells
(N1 DK_CELLS: max--min at m in {2,3} x eps in {0.05, 0.1} x B in {2, 8}, the two
demonstration cells and m = 5 at B = 2).  This driver repeats all of them with
the production direct-kill simulator (exact_m_prr_upgrade_core.simulate_chunk_general:
Euler--Maruyama dt = 1e-3, end-of-step Doi kill, identical to the N1 direct kill)
at 1e7 walkers x 2 independent seeds, for the designed AND the equal-weight
allocation of every cell (20 cells).  Designs from fb_n1_allocation_design.design.

Per cell and seed the kill times are reduced on the fly to 0.02-bin histograms
(production window edges on [0.5, 3.5] and the full [0, 4] grid) per 1e5-walker
chunk; nothing else is kept.

Reported (hpc_headline.json): mode counts (covariance-aware 5 sigma + 5 %
classifier on the 1e7 and 2e7 histograms, and its distribution over the 20
disjoint 1e6-walker sub-histograms = protocol size), significant sign changes
of f' (batch SE from the sub-histograms, z > 5), basin masses (extended
[0, s_j, 4] and window [0.5, s_j, 3.5]) with binomial and batch 95% CIs, peak
times per basin (smoothed vertex, jackknife SE), and z-scores against the N1
exact-law values and the limit-law targets.

Seeds: base 20260923, tag 97; seed index s in {0, 1}: entropy
[20260923, 97, s, m, q(eps), q(B), q(w_1..w_m), q(z0), walkers] (q = round(1e9 x));
chunk i = SeedSequence(entropy).spawn(100)[i], Philox.

Usage (from code/):
    python3 fb_hpc_headline.py simulate --workers 140
    python3 fb_hpc_headline.py analyze
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
import exact_m_prr_upgrade_core as core  # noqa: E402
import fb_n1_allocation_design as n1  # noqa: E402

ITEM = "headline"
TAG = hc.HPC_TAGS[ITEM]
WALKERS = 10_000_000
CHUNK = 100_000
SEEDS = (0, 1)
SUB = 10                       # chunks per protocol-size (1e6-walker) sub-histogram

PRIORITY = [(2, 0.1, 8.0, "equal"), (2, 0.1, 8.0, "maxmin"),
            (3, 0.1, 4.0, "equal"), (3, 0.1, 4.0, "maxmin")]
DESIGNED = [(2, 0.05, 2.0), (2, 0.05, 8.0), (2, 0.1, 2.0),
            (3, 0.05, 2.0), (3, 0.05, 8.0), (3, 0.1, 2.0), (3, 0.1, 8.0), (5, 0.1, 2.0)]
CELLS = PRIORITY + [c + ("maxmin",) for c in DESIGNED] + [c + ("equal",) for c in DESIGNED]


def cell_name(c) -> str:
    m, eps, B, a = c
    return f"m{m}_eps{eps:g}_B{B:g}_{a}"


def q(x) -> int:
    return int(round(float(x) * 1e9)) % (1 << 63)


def entropy(c, design: dict, s: int, walkers: int = WALKERS) -> list[int]:
    m, eps, B, _ = c
    spec = n1.spec_for(m, eps)
    return [hc.BASE_SEED, TAG, int(s), int(m), q(eps), q(B)] + [q(x) for x in design["w"]] + \
        [q(spec.z0), int(walkers)]


FULL_EDGES = np.round(np.arange(0.0, 4.0 + 1e-9, 0.02), 10)


def _chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()
    out = core.simulate_chunk_general(
        rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]),
        weights=tuple(task["w"]), centres_z=spec.centres(), dt=spec.dt,
        step_count=spec.steps(), p=spec.model(), n_perp=spec.n_perp)
    kt = out["kill_times"]
    edges = fk.production_window_edges()
    win, _ = np.histogram(kt[(kt >= edges[0]) & (kt <= edges[-1])], bins=edges)
    full, _ = np.histogram(kt, bins=FULL_EDGES)
    rt = time.perf_counter() - t0
    f = Path(task["file"])
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, win=win.astype(np.int64), full=full.astype(np.int64),
                        survivors=out["survivors"], kills=kt.size,
                        kill_probability_max=out["kill_probability_max"],
                        walker_steps=out["walker_steps"], runtime=rt)
    os.replace(tmp, f)
    return {"cell": task["cell"], "seed": task["seed"], "chunk": task["chunk"], "runtime_seconds": rt}


def build_tasks(cells=CELLS, seeds=SEEDS, walkers=WALKERS, chunk=CHUNK, force=False, root=None):
    root = hc.work_dir(ITEM) if root is None else Path(root)
    tasks = []
    n_chunks = walkers // chunk
    for c in cells:
        m, eps, B, a = c
        spec = n1.spec_for(m, eps)
        d = n1.design(spec, B, a)
        for s in seeds:
            ent = entropy(c, d, s, walkers)
            kids = np.random.SeedSequence(ent).spawn(n_chunks)
            for i in range(n_chunks):
                f = root / cell_name(c) / f"seed{s}_chunk{i:03d}.npz"
                if f.exists() and not force:
                    continue
                tasks.append({"cell": cell_name(c), "seed": s, "chunk": i, "size": chunk,
                              "seedseq": kids[i], "spec": spec.to_dict(), "B": B, "w": d["w"],
                              "file": str(f)})
    return tasks


def cmd_simulate(args):
    cells = CELLS if not args.only else [CELLS[int(i)] for i in args.only.split(",")]
    tasks = build_tasks(cells, force=args.force)
    print(f"[headline] {len(tasks)} chunk tasks ({len(cells)} cells x {len(SEEDS)} seeds), "
          f"env {hc.env_info()}", flush=True)
    deadline = time.time() + 60 * args.deadline_min if args.deadline_min else None
    t0 = time.time()
    hc.run_pool(_chunk, tasks, args.workers, None, label="headline", deadline=deadline)
    print(f"[headline] simulate wall {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def load(c, s):
    root = hc.work_dir(ITEM) / cell_name(c)
    n_chunks = WALKERS // CHUNK
    files = [root / f"seed{s}_chunk{i:03d}.npz" for i in range(n_chunks)]
    if not all(f.exists() for f in files):
        return None
    win, full, surv, kp, ws, rt = [], [], 0, 0.0, 0, 0.0
    for f in files:
        with np.load(f) as d:
            win.append(d["win"])
            full.append(d["full"])
            surv += int(d["survivors"])
            kp = max(kp, float(d["kill_probability_max"]))
            ws += int(d["walker_steps"])
            rt += float(d["runtime"])
    return {"win": np.array(win), "full": np.array(full), "survivors": surv,
            "kill_probability_max": kp, "walker_steps": ws, "core_seconds": rt}


def sub_hists(arr: np.ndarray) -> np.ndarray:
    """(chunks, bins) -> (chunks // SUB, bins) disjoint 1e6-walker sub-histograms."""
    k = arr.shape[0] // SUB
    return arr[: k * SUB].reshape(k, SUB, -1).sum(1)


def peak_times(counts_sub: np.ndarray, edges: np.ndarray, cut_times, bandwidth=0.04) -> dict:
    """Smoothed-density vertex per basin (parabolic refinement); jackknife over sub-histograms."""
    t = 0.5 * (edges[:-1] + edges[1:])
    bw = float(np.median(np.diff(edges)))

    def vertices(cnt):
        dens = cnt / bw
        sm = fk._gauss_smooth(dens, bandwidth / bw)
        out = []
        for a, b in zip(cut_times[:-1], cut_times[1:]):
            msk = np.flatnonzero((t > a) & (t < b))
            i = int(msk[np.argmax(sm[msk])])
            if 0 < i < sm.size - 1:
                y0, y1, y2 = sm[i - 1], sm[i], sm[i + 1]
                den = y0 - 2 * y1 + y2
                off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
                off = max(-0.5, min(0.5, off))
            else:
                off = 0.0
            out.append(float(t[i] + off * bw))
        return np.array(out)

    tot = counts_sub.sum(0)
    full = vertices(tot.astype(float))
    G = counts_sub.shape[0]
    jack = np.array([vertices((tot - counts_sub[g]).astype(float)) for g in range(G)])
    return {"peak_times": full, "jackknife_se": hc.jackknife_se(jack)}


def basin_block(full_counts_sub: np.ndarray, N: int, cut_times, edges) -> dict:
    ci = [int(np.argmin(np.abs(edges - c))) for c in cut_times]
    tot = full_counts_sub.sum(0)
    M = np.array([tot[a:b].sum() / N for a, b in zip(ci[:-1], ci[1:])])
    se_bin = np.sqrt(M * (1 - M) / N)
    nsub = full_counts_sub.shape[0]
    Nsub = N / nsub
    Mb = np.array([[full_counts_sub[g, a:b].sum() / Nsub for a, b in zip(ci[:-1], ci[1:])]
                   for g in range(nsub)])
    se_b = Mb.std(axis=0, ddof=1) / math.sqrt(nsub)
    return {"cuts": [float(edges[k]) for k in ci], "mass": M, "se_binomial": se_bin,
            "se_batch": se_b, "ci95_binomial": [[float(x - 1.96 * s), float(x + 1.96 * s)]
                                                 for x, s in zip(M, se_bin)]}


def census(win_sub: np.ndarray, N: int, edges: np.ndarray, bandwidth: float) -> dict:
    t = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)
    nsub = win_sub.shape[0]
    dens_b = win_sub / ((N / nsub) * bw[None, :])
    d = fk.derivative_signs(t, dens_b.mean(0), dens_b, bandwidth=bandwidth, zthr=5.0)
    return {"n_maxima": d["n_maxima"], "n_minima": d["n_minima"], "maxima_t": d["maxima_t"],
            "minima_t": d["minima_t"], "bandwidth": bandwidth, "n_undecided_points": d["n_undecided_points"]}


def classify(counts, N, edges):
    import reclassify_covariance_aware as R
    c = R.classify_both(np.asarray(counts, np.int64), edges, int(N), bandwidth=fk.base.DEFAULT_BANDWIDTH)
    return {"mode_count": int(c["mode_count_covariance_aware"]),
            "significant_times": c["significant_times_covariance_aware"],
            "maxima": [{"time": r["time"], "relative_prominence": r["relative_prominence"],
                        "z_covariance_aware": r["z_covariance_aware"],
                        "significant": r["significant_covariance_aware"]} for r in c["rows"]]}


def n1_row(c) -> dict | None:
    p = fk.FB_DATA / "N1" / "n1_allocation_design.json"
    if not p.exists():
        return None
    m, eps, B, a = c
    for x in json.loads(p.read_text())["cells"]:
        if x["m"] == m and abs(x["eps"] - eps) < 1e-12 and abs(x["B"] - B) < 1e-9 and x["allocation"] == a:
            return x
    return None


def analyze_cell(c) -> dict | None:
    m, eps, B, a = c
    spec = n1.spec_for(m, eps)
    d = n1.design(spec, B, a)
    edges_w = fk.production_window_edges()
    geo = n1.geometric_cuts(spec)
    cuts_ext = [0.0] + geo + [4.0]
    cuts_win = [0.5] + geo + [3.5]
    seeds = {}
    for s in SEEDS:
        L = load(c, s)
        if L is not None:
            seeds[s] = L
    if not seeds:
        return None
    out = {"cell": {"m": m, "eps": eps, "B": B, "allocation": a}, "design": d,
           "walkers_per_seed": WALKERS, "chunk": CHUNK,
           "seed_entropies": {str(s): entropy(c, d, s) for s in seeds},
           "simulator": "exact_m_prr_upgrade_core.simulate_chunk_general (EM dt=1e-3, end-of-step Doi kill)",
           "per_seed": {}, "pooled": None}

    def block(win, full, N):
        ws = sub_hists(win)
        fs = sub_hists(full)
        cls = classify(win.sum(0), N, edges_w)
        prot = [classify(ws[g], N / ws.shape[0], edges_w)["mode_count"] for g in range(ws.shape[0])]
        return {"walkers": N, "window_mass": float(win.sum() / N),
                "classifier_full_histogram": cls,
                "protocol_1e6_subhistograms": {"n": len(prot), "mode_counts": prot,
                                               "n_eq_m": int(sum(x == m for x in prot)),
                                               "histogram": {str(u): int(sum(x == u for x in prot))
                                                             for u in sorted(set(prot))}},
                "derivative_census_bw0.04": census(ws, N, edges_w, 0.04),
                "derivative_census_bw0": census(ws, N, edges_w, 0.0),
                "basins_extended": basin_block(fs, N, cuts_ext, FULL_EDGES),
                "basins_window": basin_block(ws, N, cuts_win, edges_w),
                "peaks_extended_basins": peak_times(fs, FULL_EDGES, cuts_ext)}

    for s, L in seeds.items():
        out["per_seed"][str(s)] = block(L["win"], L["full"], WALKERS)
        out["per_seed"][str(s)].update({"survivors_at_tmax": L["survivors"],
                                        "kill_probability_max": L["kill_probability_max"],
                                        "core_seconds": L["core_seconds"]})
    if len(seeds) == 2:
        win = np.concatenate([seeds[s]["win"] for s in SEEDS])
        full = np.concatenate([seeds[s]["full"] for s in SEEDS])
        out["pooled"] = block(win, full, WALKERS * 2)
        b0 = out["per_seed"]["0"]["basins_extended"]
        b1 = out["per_seed"]["1"]["basins_extended"]
        out["seed_consistency_z_extended"] = ((b0["mass"] - b1["mass"])
                                              / np.hypot(b0["se_binomial"], b1["se_binomial"]))
    # versus the exact law (N1 FK, 5e5 paths) and the limit-law target
    row = n1_row(c)
    ref = out["pooled"] or out["per_seed"][str(min(seeds))]
    if row is not None:
        fk_ext = np.asarray(row["basins_extended"]["mass"])
        fk_se = np.asarray(row["basins_extended"].get("se_iid", row["basins_extended"].get("se_batch")))
        dk = ref["basins_extended"]
        z = (dk["mass"] - fk_ext) / np.hypot(dk["se_binomial"], fk_se)
        out["vs_N1_exact_law"] = {"n1_cuts": row["basins_extended"]["cut_times"],
                                  "fk_mass_extended": fk_ext, "fk_se_iid": fk_se, "z_dk_minus_fk": z,
                                  "max_abs_z": float(np.max(np.abs(z))),
                                  "fk_protocol_mode_count_1e6": row["protocol_mode_count_1e6"],
                                  "fk_protocol_significant_times": row.get("protocol_significant_times"),
                                  "target_limit_law": row["target"],
                                  "dk_minus_target": (dk["mass"] - np.asarray(row["target"])),
                                  "w_n1": row["w"]}
    return out


def cmd_analyze(args):
    res = {"item": "HPC_headline (1e7-walker direct-kill confirmation, 2 seeds)", "driver": HERE.name,
           "env": hc.env_info(), "tag": TAG, "base_seed": hc.BASE_SEED,
           "classifier": "reclassify_covariance_aware.classify_both, bandwidth 0.04, 5 sigma + 5% floor",
           "cells": {}}
    for c in CELLS:
        r = analyze_cell(c)
        if r is not None:
            res["cells"][cell_name(c)] = r
            print(f"[headline] analyzed {cell_name(c)}", flush=True)
    # headline table
    rows = []
    for k, r in res["cells"].items():
        ref = r["pooled"] or next(iter(r["per_seed"].values()))
        rows.append({"cell": k, "walkers": ref["walkers"],
                     "mode_count_full": ref["classifier_full_histogram"]["mode_count"],
                     "protocol_1e6_n_eq_m": ref["protocol_1e6_subhistograms"]["n_eq_m"],
                     "protocol_1e6_n": ref["protocol_1e6_subhistograms"]["n"],
                     "protocol_1e6_histogram": ref["protocol_1e6_subhistograms"]["histogram"],
                     "deriv_maxima_bw0.04": ref["derivative_census_bw0.04"]["n_maxima"],
                     "basins_extended": ref["basins_extended"]["mass"],
                     "basins_extended_se": ref["basins_extended"]["se_binomial"],
                     "peak_times": ref["peaks_extended_basins"]["peak_times"],
                     "fk_protocol_mode_count_1e6": r.get("vs_N1_exact_law", {}).get("fk_protocol_mode_count_1e6"),
                     "max_abs_z_vs_fk": r.get("vs_N1_exact_law", {}).get("max_abs_z"),
                     "seed_consistency_max_abs_z": (float(np.max(np.abs(r["seed_consistency_z_extended"])))
                                                    if "seed_consistency_z_extended" in r else None)})
    res["headline_table"] = rows
    hc.write_json(hc.out_dir(ITEM) / "hpc_headline.json", res)
    print("[headline] wrote", hc.out_dir(ITEM) / "hpc_headline.json", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--workers", type=int, default=140)
    s.add_argument("--only", default=None, help="comma list of CELLS indices")
    s.add_argument("--deadline-min", type=float, default=None)
    s.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
