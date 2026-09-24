#!/usr/bin/env python3
"""N10 -- large-B validation of the FK exact-law estimator against direct kill.

Answers referee finding Opus F4 ("FK reliability at large B is not documented;
direct-kill validation stops at B <= 8").

Main arm.  (m, eps) in {(2, 0.05), (2, 0.1), (3, 0.05), (3, 0.1)}, B in {16, 32, 64},
allocations 'equal' and 'maxmin' (the equal-mass design p*(B, m) of TH-4,
fb_n1_allocation_design.design).  Per cell: 1e6 direct-kill walkers
(exact_m_prr_upgrade_core.simulate_chunk_general, EM + end-of-step Doi kill, the
production simulator of N1), seed tag 90, against the FK exact law from the
existing tag-81 ensembles n1_m{m}_eps{eps} (5e5 unkilled paths, stored mode,
production grid; secondary FK set: tag-80 n0_m{m}_eps{eps}, 2e5 paths).
Compared: basin masses (window cuts [0.5, s_1.., 3.5] and extended cuts
[0, s_1.., 4]; s_j = geometric midpoints as in N1) with z-scores, covariance-aware
chi-square on merged window bins, protocol-P mode counts, per-basin peak times of
the protocol-smoothed density (jackknife CIs: 10 DK chunks, 20 FK path groups),
last-mode relative prominence, and FK Kish effective sample sizes
ESS = (sum_i y_i)^2 / sum_i y_i^2 (per basin and per 0.02 bin) plus the weight share
of the top 1% of paths.

Remark arm.  Production geometry, eps = 0.1, m in {2, 3}, equal weights, contact
radius a = 0.4, B in {1e2, 1e3}: per-basin masses and the time of the late maximum
(DK 1e6 walkers vs FK).

Direct-kill histograms are binned by KILL STEP on the FK checkpoint grid
(bin k = steps s with cps[k] <= s < cps[k+1], kill time (s+1) dt), so DK and FK
use identical bins by construction (the window part reproduces the production
histogram bit for bit, exact_m_prr_fk_exact_law.build_grid).

Seeds: base 20260923, tag 90 (new).  DK entropy
[20260923, 90, arm, m, eps*1e9, B*1e9, round(w_j*1e9)..., z0*1e9, walkers], arm 0 = main,
1 = remark; chunk i uses SeedSequence(entropy).spawn(n_chunks)[i] with Philox.

Usage
-----
    python3 fb_n10_largeB_validation.py dk [--only i,j] [--arm main|remark|all]
    python3 fb_n10_largeB_validation.py fk          # FK passes (one per (m, eps))
    python3 fb_n10_largeB_validation.py compare     # -> N10_largeB_validation/n10_summary.json
    python3 fb_n10_largeB_validation.py figure
CPU etiquette: <= 3 worker processes; waits while >= 6 other python processes use
> 20% CPU (the pgrep count on this machine is inflated by ~40 idle MCP servers).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402
import fb_n1_allocation_design as n1  # noqa: E402

OUT = fk.FB_DATA / "N10_largeB_validation"
DK_DIR = OUT / "directkill"
FK_DIR = OUT / "fk"
SEED = fk.BASE_SEED
TAG = 90
WALKERS = 1_000_000
CHUNK = 100_000
MAX_WORKERS = 3
GROUPS = 20
BANDWIDTH = fk.base.DEFAULT_BANDWIDTH       # 0.04 (protocol P)

MAIN_MEPS = ((2, 0.05), (2, 0.1), (3, 0.05), (3, 0.1))
MAIN_B = (16.0, 32.0, 64.0)
ALLOCS = ("equal", "maxmin")
REMARK_M = (2, 3)
REMARK_EPS = 0.1
REMARK_B = (100.0, 1000.0)
FK_PRIMARY = "n1_m{m}_eps{eps:g}"          # tag 81, 5e5 paths
FK_SECONDARY = "n0_m{m}_eps{eps:g}"        # tag 80, 2e5 paths
PROM_SCAN = tuple(float(x) for x in np.geomspace(16.0, 64.0, 9))


def main_cells():
    return [(m, e, B, a, "main") for (m, e) in MAIN_MEPS for B in MAIN_B for a in ALLOCS]


def remark_cells():
    return [(m, REMARK_EPS, B, "equal", "remark") for m in REMARK_M for B in REMARK_B]


def all_cells():
    return main_cells() + remark_cells()


def cell_name(m, eps, B, a, arm):
    return f"dk_{arm}_m{m}_eps{eps:g}_B{B:g}_{a}"


def busy_python() -> dict:
    """Python processes using > 20% CPU (compute load), and the raw pgrep -if count."""
    try:
        ps = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True,
                            timeout=20).stdout.splitlines()[1:]
        busy = 0
        for ln in ps:
            parts = ln.strip().split(None, 1)
            if len(parts) == 2 and "python" in parts[1].lower() and float(parts[0]) > 20.0:
                busy += 1
        pg = subprocess.run(["pgrep", "-if", "python"], capture_output=True, text=True,
                            timeout=20).stdout.split()
        return {"busy_python_gt20pct": busy, "pgrep_if_python": len(pg)}
    except Exception as exc:  # pragma: no cover
        return {"busy_python_gt20pct": 0, "pgrep_if_python": -1, "error": str(exc)}


def wait_cpu(limit: int = 6, poll: float = 60.0, max_wait: float = 1500.0) -> dict:
    t0 = time.time()
    while True:
        b = busy_python()
        if b["busy_python_gt20pct"] < limit or time.time() - t0 > max_wait:
            b["waited_seconds"] = time.time() - t0
            return b
        print(f"[n10] {b} busy; waiting", flush=True)
        time.sleep(poll)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(fk._jsonable(obj), indent=1, default=fk._json_default))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Direct kill
# ---------------------------------------------------------------------------


def dk_entropy(arm, m, eps, B, w, z0, walkers):
    q = lambda x: int(round(float(x) * 1e9)) % (1 << 63)  # noqa: E731
    return ([SEED, TAG, 0 if arm == "main" else 1, int(m), q(eps), q(B)]
            + [q(x) for x in w] + [q(z0), int(walkers)])


def _dk_chunk(task: dict) -> dict:
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    p = spec.model()
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    out = core.simulate_chunk_general(
        rng, int(task["size"]), eps=spec.eps, budget=float(task["B"]),
        weights=tuple(task["w"]), centres_z=spec.centres(), dt=spec.dt,
        step_count=spec.steps(), p=p, n_perp=spec.n_perp)
    steps = np.rint(out["kill_times"] / spec.dt).astype(np.int64) - 1   # 0-based kill step
    cps = np.asarray(task["cps"], np.int64)
    k = np.searchsorted(cps, steps, side="right") - 1
    ok = (k >= 0) & (k < cps.size - 1)
    counts = np.bincount(k[ok], minlength=cps.size - 1)[: cps.size - 1]
    return {"chunk": task["chunk"], "counts": counts.astype(np.int64),
            "n_killed": int(steps.size), "n_outside_grid": int((~ok).sum()),
            "survivors": out["survivors"], "kill_probability_max": out["kill_probability_max"],
            "walker_steps": out["walker_steps"]}


def cmd_dk(args) -> None:
    import multiprocessing as mp
    cells = {"main": main_cells(), "remark": remark_cells(), "all": all_cells()}[args.arm]
    if args.only:
        cells = [cells[int(i)] for i in args.only.split(",")]
    walkers = int(float(args.walkers))
    ctx = mp.get_context("spawn")
    for (m, eps, B, a, arm) in cells:
        name = cell_name(m, eps, B, a, arm)
        outp = DK_DIR / f"{name}.json"
        if outp.exists() and not args.overwrite:
            print(f"[n10] {name} exists, skip", flush=True)
            continue
        cpu = wait_cpu()
        spec = n1.spec_for(m, eps)
        d = n1.design(spec, B, a)
        grid = fk.build_grid(spec, "production")
        ent = dk_entropy(arm, m, eps, B, d["w"], spec.z0, walkers)
        sizes = [CHUNK] * (walkers // CHUNK) + ([walkers % CHUNK] if walkers % CHUNK else [])
        children = np.random.SeedSequence(ent).spawn(len(sizes))
        tasks = [{"spec": spec.to_dict(), "size": s, "seedseq": c, "B": B, "w": d["w"],
                  "chunk": i, "cps": grid["cps"].tolist()} for i, (s, c) in enumerate(zip(sizes, children))]
        t0 = time.time()
        with ctx.Pool(processes=min(args.workers, MAX_WORKERS)) as pool:
            res = sorted(pool.map(_dk_chunk, tasks), key=lambda r: r["chunk"])
        counts = np.sum([r["counts"] for r in res], axis=0)
        surv = int(sum(r["survivors"] for r in res))
        nk = int(sum(r["n_killed"] for r in res))
        assert nk + surv == walkers, "walker bookkeeping"
        assert sum(r["n_outside_grid"] for r in res) == 0
        payload = {"cell": {"m": m, "eps": eps, "B": B, "allocation": a, "arm": arm},
                   "design": d, "walkers": walkers, "chunk": CHUNK, "seed": SEED, "tag": TAG,
                   "seed_entropy": ent,
                   "rng": "numpy Philox; chunk i uses SeedSequence(seed_entropy).spawn(n_chunks)[i]",
                   "simulator": "exact_m_prr_upgrade_core.simulate_chunk_general (EM + end-of-step Doi kill)",
                   "spec": spec.to_dict(), "grid_edges": grid["edges"].tolist(),
                   "grid_cps": grid["cps"].tolist(), "window_index": grid["window_index"].tolist(),
                   "counts": counts.tolist(),
                   "chunk_counts": [r["counts"].tolist() for r in res],
                   "survivors_at_tmax": surv,
                   "kill_probability_max": max(r["kill_probability_max"] for r in res),
                   "walker_steps": int(sum(r["walker_steps"] for r in res)),
                   "wall_seconds": time.time() - t0, "workers": min(args.workers, MAX_WORKERS),
                   "cpu_gate": cpu}
        write_json(outp, payload)
        print(f"[n10] {name}: wall {time.time() - t0:.1f}s survivors {surv}", flush=True)




# ---------------------------------------------------------------------------
# FK pass: one read of an ensemble serves every (B, allocation) item
# ---------------------------------------------------------------------------


def fk_items(m: int, eps: float) -> list:
    spec = n1.spec_for(m, eps)
    items = []
    for a in ALLOCS:
        for B in PROM_SCAN:
            d = n1.design(spec, B, a)
            full = any(abs(B - b) < 1e-9 for b in MAIN_B)
            items.append({"key": f"B{B:.6g}_{a}", "B": B, "alloc": a, "w": d["w"],
                          "target": d["target"], "arm": "main", "full": full})
    if abs(eps - REMARK_EPS) < 1e-12:
        for B in REMARK_B:
            d = n1.design(spec, B, "equal")
            items.append({"key": f"B{B:g}_equal_remark", "B": B, "alloc": "equal", "w": d["w"],
                          "target": d["target"], "arm": "remark", "full": True})
    return items


def fk_pass(m: int, eps: float, ens_name: str, groups: int = GROUPS) -> dict:
    ens = fk.load_ensemble(ens_name)
    spec = ens.spec
    N = ens.n_paths
    K = ens.cps.size
    wi = ens.window_index
    w0, w1 = int(wi[0]), int(wi[-1])
    cuts = n1.cut_indices(ens)
    cut_sets = {"window": cuts["window"], "extended": cuts["extended"]}
    items = fk_items(m, eps)
    gb = fk._group_bounds(N, groups)
    acc = []
    for it in items:
        a = {"s1": np.zeros(K - 1), "s2": np.zeros(K - 1), "g": np.zeros((groups, K - 1)),
             "surv4": 0.0}
        if it["full"]:
            a["cw"] = np.zeros((w1 - w0, w1 - w0))
            a["basin_paths"] = {c: [] for c in cut_sets}
        acc.append(a)
    wv = {}
    t0 = time.time()
    for start, dX, _ in ens.iter_chunks("full"):
        dX = dX.astype(np.float64)
        n = dX.shape[0]
        gid = fk._group_ids(gb, start, n)
        cache = {}
        for it, a in zip(items, acc):
            wkey = tuple(round(x, 15) for x in it["w"])
            if wkey not in cache:
                w = np.asarray(it["w"], float)
                dXw = np.einsum("imk,m->ik", dX, w)
                cache = {wkey: (dXw, np.cumsum(dXw, axis=1))}   # keep one w in memory
            dXw, Xw = cache[wkey]
            B = it["B"]
            E = np.exp(-B * Xw)
            y = E[:, :-1] * (-np.expm1(-B * dXw[:, 1:]))
            a["s1"] += y.sum(0)
            a["s2"] += (y * y).sum(0)
            a["g"] += fk._group_sums(y, gid, groups)
            a["surv4"] += float(E[:, -1].sum())
            if it["full"]:
                yw = y[:, w0:w1]
                with np.errstate(all="ignore"):
                    a["cw"] += yw.T @ yw
                cs = np.concatenate([np.zeros((n, 1)), np.cumsum(y, axis=1)], axis=1)
                for c, ci in cut_sets.items():
                    a["basin_paths"][c].append(np.stack([cs[:, ci[j + 1]] - cs[:, ci[j]]
                                                         for j in range(len(ci) - 1)], axis=1))
        del dX
    sizes = np.diff(gb).astype(float)
    out = {"ensemble": ens_name, "N": N, "tag": int(ens.index["tag"]),
           "seed_entropy": ens.index.get("seed_entropy"), "m": m, "eps": eps,
           "edges": ens.edges, "cps": ens.cps, "window_index": wi, "cuts": cuts,
           "group_sizes": sizes, "runtime_seconds": time.time() - t0, "items": {}}
    arrays = {}
    for it, a in zip(items, acc):
        p = a["s1"] / N
        se = np.sqrt(np.maximum(a["s2"] / N - p * p, 0.0) / N)
        ess_bin = np.divide(a["s1"] ** 2, a["s2"], out=np.zeros_like(a["s1"]), where=a["s2"] > 0)
        rec = {k: it[k] for k in ("key", "B", "alloc", "w", "target", "arm", "full")}
        rec["survival_t4"] = a["surv4"] / N
        arrays[f"{it['key']}__p"] = p
        arrays[f"{it['key']}__se"] = se
        arrays[f"{it['key']}__ess_bin"] = ess_bin
        arrays[f"{it['key']}__g"] = a["g"] / sizes[:, None]
        if it["full"]:
            pw = p[w0:w1]
            cov = (a["cw"] / N - np.outer(pw, pw)) / N
            arrays[f"{it['key']}__covw"] = cov
            bas = {}
            for c, ci in cut_sets.items():
                V = np.concatenate(a["basin_paths"][c], axis=0)       # (N, nb)
                M = V.mean(0)
                s2 = (V * V).sum(0)
                ess = np.divide(V.sum(0) ** 2, s2, out=np.zeros(V.shape[1]), where=s2 > 0)
                top = []
                for j in range(V.shape[1]):
                    col = V[:, j]
                    tot = col.sum()
                    kk = max(1, int(round(0.01 * col.size)))
                    top.append(float(np.partition(col, col.size - kk)[-kk:].sum() / tot) if tot > 0 else None)
                gm = np.stack([V[int(gb[g]):int(gb[g + 1])].mean(0) for g in range(groups)])
                bas[c] = {"cut_indices": ci, "cut_times": [float(ens.edges[k]) for k in ci],
                          "mass": M, "se_iid": V.std(0, ddof=1) / math.sqrt(V.shape[0]),
                          "se_batch": fk._batch_se(gm, sizes), "group_mass": gm,
                          "kish_ess": ess, "top1pct_weight_share": top,
                          "max_path_weight_share": (V.max(0) / np.maximum(V.sum(0), 1e-300))}
                # per-bin ESS inside each basin (bins holding >= 1% of the basin's largest bin)
                eb = []
                for j in range(len(ci) - 1):
                    seg = slice(ci[j], ci[j + 1])
                    pj = p[seg]
                    if pj.size == 0 or pj.max() <= 0:
                        eb.append(None)
                        continue
                    sel = pj >= 0.01 * pj.max()
                    e = ess_bin[seg][sel]
                    eb.append({"median": float(np.median(e)), "min": float(e.min()),
                               "n_bins": int(sel.sum())})
                bas[c]["per_bin_ess"] = eb
            rec["basins"] = bas
        out["items"][it["key"]] = rec
    return out, arrays


def cmd_fk(args) -> None:
    FK_DIR.mkdir(parents=True, exist_ok=True)
    sets = [(FK_PRIMARY, "primary")] + ([(FK_SECONDARY, "secondary")] if args.secondary else [])
    for (m, eps) in MAIN_MEPS:
        for pat, role in sets:
            name = pat.format(m=m, eps=eps)
            outp = FK_DIR / f"fk_{name}.json"
            if outp.exists() and not args.overwrite:
                print(f"[n10] {name} exists, skip", flush=True)
                continue
            cpu = wait_cpu()
            t0 = time.time()
            rec, arrays = fk_pass(m, eps, name)
            rec["role"] = role
            rec["cpu_gate"] = cpu
            np.savez_compressed(FK_DIR / f"fk_{name}.npz", **arrays)
            write_json(outp, rec)
            print(f"[n10] fk {name}: {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _smooth(p: np.ndarray, bw: float = 0.02) -> np.ndarray:
    _, _, A = fk._smoothing_operator(p.size, bw, BANDWIDTH)
    return A @ (p / bw)


def _peak_in(p: np.ndarray, tc: np.ndarray, lo: int, hi: int) -> dict:
    """Maximum of the protocol-smoothed density on bins [lo, hi) (parabolic refinement)."""
    s = _smooth(p)
    seg = s[lo:hi]
    i = int(np.argmax(seg)) + lo
    interior = lo < i < hi - 1
    t = float(tc[i])
    if interior:
        den = s[i - 1] - 2 * s[i] + s[i + 1]
        if den < 0:
            t += 0.02 * 0.5 * (s[i - 1] - s[i + 1]) / den
    return {"t": t, "interior": bool(interior), "height": float(s[i])}


def _jack_se(vals) -> float | None:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    G = v.size
    if G < 2:
        return None
    return float(math.sqrt((G - 1) / G * np.sum((v - v.mean()) ** 2)))


def _last_prom(p_window: np.ndarray, edges_w: np.ndarray, walkers: float, after: float) -> dict:
    cls = fk.classify_expected(p_window, edges_w, walkers=walkers, bandwidth=BANDWIDTH)
    rows = [r for r in cls["rows"] if r["time"] > after]
    best = max(rows, key=lambda r: r["relative_prominence"]) if rows else None
    return {"mode_count": int(cls["mode_count"]),
            "significant_times": [r["time"] for r in cls["rows"] if r["significant"]],
            "last_basin_max": ({"time": best["time"], "relative_prominence": best["relative_prominence"],
                                "z_poisson_at_walkers": best["z"]} if best else None)}


def _replicates_dk(counts: np.ndarray, chunk_counts: np.ndarray, N: int, nsz: np.ndarray):
    for i in range(chunk_counts.shape[0]):
        yield (counts - chunk_counts[i]) / (N - nsz[i])


def _replicates_fk(p: np.ndarray, g: np.ndarray, N: int, sizes: np.ndarray):
    for i in range(g.shape[0]):
        yield (N * p - sizes[i] * g[i]) / (N - sizes[i])


def compare_cell(dk: dict, fkrec: dict, fkarr, key: str) -> dict:
    import reclassify_covariance_aware as RC
    from exact_m_prr_fk_n0_validation import _chi2_sf, _cov_chi2, merge_groups
    N = int(dk["walkers"])
    edges = np.asarray(fkrec["edges"], float)
    if not np.allclose(edges, np.asarray(dk["grid_edges"], float), rtol=0, atol=0):
        raise ValueError("DK and FK grids differ")
    wi = fkrec["window_index"]
    w0, w1 = int(wi[0]), int(wi[-1])
    tc = 0.5 * (edges[:-1] + edges[1:])
    cnt = np.asarray(dk["counts"], float)
    ch = np.asarray(dk["chunk_counts"], float)
    nsz = np.full(ch.shape[0], float(dk["chunk"]))
    pdk = cnt / N
    it = fkrec["items"][key]
    p = fkarr[f"{key}__p"]
    g = fkarr[f"{key}__g"]
    C = fkarr[f"{key}__covw"]
    sizes = np.asarray(fkrec["group_sizes"], float)
    Nf = int(fkrec["N"])
    # (1) window bins: merged covariance chi-square (as N1 compare_directkill)
    pw, pdw = p[w0:w1], pdk[w0:w1]
    grp = merge_groups(N * pw, 10.0)
    A = np.zeros((len(grp), pw.size))
    for gi, gg in enumerate(grp):
        A[gi, gg] = 1.0
    pm, pdm = A @ pw, A @ pdw
    Cm = A @ C @ A.T
    zm = (pm - pdm) / np.sqrt(np.diag(Cm) + pm * (1 - pm) / N)
    chi2, dof = _cov_chi2(pm - pdm, Cm + (np.diag(pm) - np.outer(pm, pm)) / N)
    # (2) basin masses, both conventions
    bas = {}
    for conv, b in it["basins"].items():
        ci = b["cut_indices"]
        cum = np.concatenate([[0.0], np.cumsum(cnt)])
        Mdk = np.array([(cum[ci[j + 1]] - cum[ci[j]]) / N for j in range(len(ci) - 1)])
        sedk = np.sqrt(Mdk * (1 - Mdk) / N)
        Mfk = np.asarray(b["mass"])
        sef = np.maximum(np.asarray(b["se_iid"]), np.asarray(b["se_batch"]))
        z = (Mfk - Mdk) / np.sqrt(sef ** 2 + sedk ** 2)
        bas[conv] = {"cut_times": b["cut_times"], "fk": Mfk, "fk_se_iid": b["se_iid"],
                     "fk_se_batch": b["se_batch"], "dk": Mdk, "dk_se": sedk,
                     "dk_counts": [int(round(x * N)) for x in Mdk], "z": z,
                     "fk_kish_ess": b["kish_ess"], "fk_top1pct_weight_share": b["top1pct_weight_share"],
                     "fk_max_path_weight_share": b["max_path_weight_share"],
                     "fk_per_bin_ess": b["per_bin_ess"], "limit_target": it["target"]}
    # (3) protocol P
    ew = edges[w0:w1 + 1]
    cls_dk = RC.classify_both(cnt[w0:w1].astype(np.int64), fk.production_window_edges(), N,
                              bandwidth=BANDWIDTH)
    after = it["basins"]["window"]["cut_times"][-2]
    lp_dk = _last_prom(pdw, ew, N, after)
    lp_fk = _last_prom(pw, ew, 1_000_000, after)
    lp_fk_N = _last_prom(pw, ew, N, after)
    # jackknife of the last-basin relative prominence
    def rp(x):
        r = _last_prom(x[w0:w1], ew, N, after)["last_basin_max"]
        return r["relative_prominence"] if r else 0.0
    se_r_dk = _jack_se([rp(x) for x in _replicates_dk(cnt, ch, N, nsz)])
    se_r_fk = _jack_se([rp(x) for x in _replicates_fk(p, g, Nf, sizes)])
    # (4) peak times per extended basin (full axis, protocol smoothing)
    ce = it["basins"]["extended"]["cut_indices"]
    peaks = []
    for j in range(len(ce) - 1):
        lo, hi = ce[j], ce[j + 1]
        pk_dk, pk_fk = _peak_in(pdk, tc, lo, hi), _peak_in(p, tc, lo, hi)
        sdk = _jack_se([_peak_in(x, tc, lo, hi)["t"] for x in _replicates_dk(cnt, ch, N, nsz)])
        sfk = _jack_se([_peak_in(x, tc, lo, hi)["t"] for x in _replicates_fk(p, g, Nf, sizes)])
        zz = ((pk_fk["t"] - pk_dk["t"]) / math.sqrt(sdk ** 2 + sfk ** 2)
              if sdk and sfk and (sdk ** 2 + sfk ** 2) > 0 else None)
        peaks.append({"basin": j + 1, "dk": pk_dk, "fk": pk_fk, "dk_jack_se": sdk,
                      "fk_jack_se": sfk, "z": zz})
    rdk = lp_dk["last_basin_max"]["relative_prominence"] if lp_dk["last_basin_max"] else 0.0
    rfk = lp_fk_N["last_basin_max"]["relative_prominence"] if lp_fk_N["last_basin_max"] else 0.0
    zr = ((rfk - rdk) / math.sqrt(se_r_dk ** 2 + se_r_fk ** 2)
          if se_r_dk is not None and se_r_fk is not None and (se_r_dk ** 2 + se_r_fk ** 2) > 0 else None)
    return {"key": key, "walkers": N, "fk_paths": Nf, "w": it["w"],
            "window_bins": {"n_merged": len(grp), "max_abs_z": float(np.max(np.abs(zm))),
                            "n_abs_z_ge_3": int(np.sum(np.abs(zm) >= 3)),
                            "chi2_cov": chi2, "dof": dof, "p_wilson_hilferty": _chi2_sf(chi2, dof)},
            "basins": bas,
            "protocol": {"dk_mode_count_cov_aware": int(cls_dk["mode_count_covariance_aware"]),
                         "dk_significant_times": cls_dk["significant_times_covariance_aware"],
                         "fk_mode_count_1e6": lp_fk["mode_count"],
                         "fk_significant_times": lp_fk["significant_times"]},
            "last_mode": {"after_time": after, "dk": lp_dk["last_basin_max"],
                          "fk_at_dk_walkers": lp_fk_N["last_basin_max"],
                          "dk_rel_prom_jack_se": se_r_dk, "fk_rel_prom_jack_se": se_r_fk,
                          "z_rel_prom_fk_minus_dk": zr},
            "peaks_extended_basins": peaks,
            "survival_t4": {"dk": dk["survivors_at_tmax"] / N, "fk": it["survival_t4"]},
            "dk_wall_seconds": dk["wall_seconds"], "dk_walker_steps": dk["walker_steps"],
            "dk_kill_probability_max": dk["kill_probability_max"]}


def prominence_scan(fkrec: dict, fkarr, alloc: str) -> dict:
    wi = fkrec["window_index"]
    w0, w1 = int(wi[0]), int(wi[-1])
    edges = np.asarray(fkrec["edges"], float)
    ew = edges[w0:w1 + 1]
    rows = []
    for key, it in fkrec["items"].items():
        if it["alloc"] != alloc or it["arm"] != "main":
            continue
        after = fkrec["cuts"]["cuts_snapped"][-1]
        lp = _last_prom(fkarr[f"{key}__p"][w0:w1], ew, 1_000_000, after)
        r = lp["last_basin_max"]
        rows.append({"B": it["B"], "mode_count_1e6": lp["mode_count"],
                     "last_rel_prom": r["relative_prominence"] if r else 0.0,
                     "last_z_1e6": r["z_poisson_at_walkers"] if r else 0.0})
    rows.sort(key=lambda r: r["B"])
    Bs = np.array([r["B"] for r in rows])
    rr = np.array([r["last_rel_prom"] for r in rows])
    cross, _ = fk._crossing(Bs, rr, 0.05)
    return {"alloc": alloc, "rows": rows, "B5pct_crossing_in_16_64": cross,
            "status": "interpolated" if cross else ("above_floor_throughout" if rr.min() >= 0.05
                                                    else "below_floor_throughout" if rr.max() < 0.05
                                                    else "no_down_crossing")}


def cmd_compare(args) -> None:
    summary = {"item": "N10 large-B validation of the FK exact-law estimator (Opus F4)",
               "driver": "code/fb_n10_largeB_validation.py", "seed": SEED, "dk_tag": TAG,
               "cells": [], "prominence_scans": {}, "secondary_fk": []}
    for (m, eps) in MAIN_MEPS:
        name = FK_PRIMARY.format(m=m, eps=eps)
        fkrec = json.loads((FK_DIR / f"fk_{name}.json").read_text())
        fkarr = np.load(FK_DIR / f"fk_{name}.npz")
        summary["prominence_scans"][f"m{m}_eps{eps:g}"] = {
            a: prominence_scan(fkrec, fkarr, a) for a in ALLOCS}
        sec = None
        sname = FK_SECONDARY.format(m=m, eps=eps)
        if (FK_DIR / f"fk_{sname}.json").exists():
            sec = json.loads((FK_DIR / f"fk_{sname}.json").read_text())
        for (mm, e, B, a, arm) in all_cells():
            if mm != m or abs(e - eps) > 1e-12:
                continue
            dkp = DK_DIR / f"{cell_name(mm, e, B, a, arm)}.json"
            if not dkp.exists():
                continue
            dk = json.loads(dkp.read_text())
            key = f"B{B:.6g}_{a}" if arm == "main" else f"B{B:g}_equal_remark"
            row = compare_cell(dk, fkrec, fkarr, key)
            row["cell"] = {"m": m, "eps": eps, "B": B, "allocation": a, "arm": arm}
            row["dk_seed_entropy"] = dk["seed_entropy"]
            row["fk_ensemble"] = name
            if sec is not None and key in sec["items"]:
                s_it = sec["items"][key]
                row["secondary_fk"] = {c: {"mass": s_it["basins"][c]["mass"],
                                           "se_iid": s_it["basins"][c]["se_iid"],
                                           "kish_ess": s_it["basins"][c]["kish_ess"]}
                                       for c in s_it["basins"]}
            summary["cells"].append(row)
    write_json(OUT / "n10_summary.json", summary)
    print(f"[n10] compare: {len(summary['cells'])} cells", flush=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    S = json.loads((OUT / "n10_summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
    mk = {("main", "equal"): ("o", "C0"), ("main", "maxmin"): ("s", "C1"), ("remark", "equal"): ("^", "C3")}
    ax = axes[0]
    for r in S["cells"]:
        c = r["cell"]
        b = r["basins"]["extended"]
        m_, col = mk[(c["arm"], c["allocation"])]
        ax.scatter(np.maximum(b["fk_kish_ess"], 1.0), b["z"], s=9, marker=m_, color=col,
                   edgecolors="none", alpha=0.85)
    ax.axhspan(-2, 2, color="0.9", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("FK Kish ESS of the basin")
    ax.set_ylabel(r"basin mass $z$ (FK $-$ DK)")
    ax.set_ylim(-4, 4)
    ax.set_title("(a)", loc="left")
    ax = axes[1]
    for r in S["cells"]:
        c = r["cell"]
        b = r["basins"]["extended"]
        m_, col = mk[(c["arm"], c["allocation"])]
        dk = np.asarray(b["dk"])
        fkm = np.asarray(b["fk"])
        ok = dk > 0
        ax.errorbar(dk[ok], fkm[ok], xerr=np.asarray(b["dk_se"])[ok],
                    yerr=np.maximum(np.asarray(b["fk_se_iid"]), np.asarray(b["fk_se_batch"]))[ok],
                    fmt=m_, ms=3, color=col, lw=0.6, alpha=0.85)
    lim = [5e-7, 2]
    ax.plot(lim, lim, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("direct kill (1e6 walkers)")
    ax.set_ylabel("FK exact law")
    ax.set_title("(b) basin masses", loc="left")
    for (a_, (m_, col)), lab in zip(mk.items(), ("equal", "max-min", r"$B=10^2,10^3$")):
        ax.plot([], [], m_, color=col, ms=3, label=lab)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax = axes[2]
    fkrec = json.loads((FK_DIR / "fk_n1_m2_eps0.1.json").read_text())
    fkarr = np.load(FK_DIR / "fk_n1_m2_eps0.1.npz")
    edges = np.asarray(fkrec["edges"])
    tc = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)
    for B, col in ((100.0, "C3"), (1000.0, "C4")):
        dk = json.loads((DK_DIR / f"{cell_name(2, 0.1, B, 'equal', 'remark')}.json").read_text())
        cnt = np.asarray(dk["counts"], float)
        p = fkarr[f"B{B:g}_equal_remark__p"]
        ax.step(tc, np.where(cnt > 0, cnt / dk["walkers"] / bw, np.nan), where="mid", color=col,
                lw=0.6, alpha=0.6)
        ax.plot(tc, p / bw, color=col, lw=0.9, label=f"B={B:g}")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 4.0)
    ax.set_ylim(1e-4, 30)
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.set_title(r"(c) $m=2$, $\varepsilon=0.1$", loc="left")
    ax.plot([], [], color="0.4", lw=0.6, alpha=0.6, label="direct kill")
    ax.legend(fontsize=6, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(fk.FIGURES / f"fb_n10_largeB_validation.{ext}", dpi=200)
    print("[n10] figure written", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    cmds = {"dk": cmd_dk, "fk": cmd_fk, "compare": cmd_compare}
    if "cmd_figure" in globals():
        cmds["figure"] = globals()["cmd_figure"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=sorted(cmds))
    ap.add_argument("--arm", default="all", choices=("main", "remark", "all"))
    ap.add_argument("--only", default="")
    ap.add_argument("--walkers", default=str(WALKERS))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--secondary", action="store_true", help="fk: also the tag-80 ensembles")
    args = ap.parse_args(argv)
    cmds[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
