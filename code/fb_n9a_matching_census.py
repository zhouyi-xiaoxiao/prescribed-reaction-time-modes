#!/usr/bin/env python3
"""N9a: TH-6 matching-zone census of critical points of the exact law (GAP_CLOSURE_PLAN.md section 3, N9a).

TH-6 (exact m-count at fixed B): for fixed B and small eps the exact density
has exactly m nondegenerate maxima and m-1 minima on the window I, alternating,
with non-stationary endpoints.  The open estimate (MATCH) concerns the zones
1 << |t - t_j|/eps << 1/eps; GPT-6 also flagged a subdominant m=3 tilted-bridge
tangency (observation time 2.4819).  This driver searches for EXTRA critical
points numerically.

Part 1 (FK, exact law of the production process).  Declared-mode FK ensembles
(exact_m_prr_fk_exact_law, per-step kill probabilities with batch sums), m in
{2, 3} x eps in {0.05, 0.035, 0.025}, 5e5 unkilled paths each, time step
dt = 1e-3 (eps/0.05)^2 rounded to {1e-3, 5e-4, 2.5e-4} (plan: dt ~ eps^2),
budgets B in {0.5, 1, 4, 8, 20, 50}, kernels 'full' (the model) and 'nogate'
(contact = 1, for the PDE cross-check).  For a grid spacing h_g (a multiple of
dt) the density on bins of width h_g gives f'(t) at bin boundaries by first
differences; its SE is the batch-means SE over 50 batches of 1e4 paths.  A
point is significant when |f'|/SE > 5; every sign change of the significant
sign sequence is a critical point (max: + -> -, min: - -> +).  Resolutions
h_g/eps in {0.05, 0.1, 0.25, 0.5, 1} (all <= 0.05 eps is the plan's
requirement for the finest one).  EVERY significant sign change is reported with
its zone label (inner |t-t_j| <= 3 eps; matching 3 eps < |t-t_j| <= 0.3; gap
otherwise) and flagged if it is beyond the expected m maxima / m-1 minima
("EXTRA_PAIR: potential counterexample to TH-6").  Undecided stretches (no
significant sign) are listed: they are the census' blind spots.

Part 2 (PDE, deterministic).  GPT-6's contact = 1 killed-OU PDE (theory_checks/
reduced_ou_pde.py; finite volumes in x = (Z - mu(t))/eps, Crank--Nicolson,
Strang-split exact killing, log-domain survival), re-implemented here, at eps in
{0.025, 0.0125}, m in {2, 3}, same budgets; extrema from sign changes of the
semi-discrete d log f / dt.

Usage
-----
    python3 fb_n9a_matching_census.py simulate --m 2 --eps 0.05 [--chunks 0-4]
    python3 fb_n9a_matching_census.py pde --m 2 --eps 0.025
    python3 fb_n9a_matching_census.py analyze
    python3 fb_n9a_matching_census.py figure
Seeds: base 20260923, tag 87 (N9), replicate 0 (declared-mode entropy includes eps, dt).
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
import exact_m_prr_upgrade_core as core  # noqa: E402

OUT = fk.FB_DATA / "N9a"
TAG = fk.TAGS["N9"]
M_LIST = (2, 3)
EPS_LIST = (0.05, 0.035, 0.025)
DT = {0.05: 1e-3, 0.035: 5e-4, 0.025: 2.5e-4}
BUDGETS = (0.5, 1.0, 4.0, 8.0, 20.0, 50.0)
VARIANTS = ("full", "nogate")
N_PATHS = 500_000
CHUNK = 50_000
BATCH = 10_000
RES_LIST = (0.05, 0.1, 0.25, 0.5, 1.0)       # h_g / eps
ZTHR = 5.0
INNER_L = 3.0                                 # inner zone |t - t_j| <= INNER_L eps
MATCH_DELTA = 0.3                             # matching zone upper edge |t - t_j| <= 0.3
TANGENCY_REGION = (2.3, 2.7)                  # m = 3 tilted-bridge tangency (GPT-6: t = 2.4819)
PDE_EPS = (0.025, 0.0125)


def ens_name(m: int, eps: float) -> str:
    return f"n9a_m{m}_eps{eps:g}_decl"


def spec_for(m: int, eps: float) -> fk.EnsembleSpec:
    return fk.EnsembleSpec(m=m, eps=eps, dt=DT[eps])


def declared_items(m: int):
    w = tuple([1.0 / m] * m)
    return [dict(B=B, w=w, variant=v, label=f"{v}_B{B:g}") for v in VARIANTS for B in BUDGETS]


def cmd_simulate(args):
    m, eps = args.m, args.eps
    spec = spec_for(m, eps)
    subset = None
    if args.chunks:
        a, b = (int(x) for x in args.chunks.split("-"))
        subset = list(range(a, b + 1))
    t0 = time.time()
    fk.simulate_ensemble(spec, N_PATHS, name=ens_name(m, eps), tag=TAG, mode="declared",
                         variants=("full",), declared=declared_items(m), workers=args.workers,
                         chunk=CHUNK, batch=BATCH, chunks_subset=subset,
                         note="N9a TH-6 matching-zone census (fb_n9a_matching_census.py)")
    print(f"[n9a] {ens_name(m, eps)} chunks {subset} done {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# PDE (contact = 1), port of GPT-6 reduced_ou_pde.py
# ----------------------------------------------------------------------------


def _bern(z):
    return np.divide(z, np.expm1(z), out=np.ones_like(z), where=np.abs(z) > 1e-14)


def pde_solve(m: int, eps: float, budgets, dx: float, dt: float, L: float, tmax: float = 4.5,
              record_every: float = 5e-4) -> dict:
    """Contact = 1 killed-OU PDE in x = (Z - mu(t))/eps (see module doc)."""
    from scipy.linalg import solve_banded   # scipy 1.13.1 present in the Xcode python
    p = fk.MODEL
    spec = fk.EnsembleSpec(m=m, eps=eps)
    B = np.asarray(budgets, float)
    nb = B.size
    w = np.full(m, 1.0 / m)
    centres = spec.centres()
    n = int(round(2 * L / dx)) + 1
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    N = int(round(tmax / dt))
    dt = tmax / N
    var = p.d0 / (2 * p.gamma)
    dV = (x[1:] ** 2 - x[:-1] ** 2) / (2 * var)
    right = p.d0 / (2 * dx * dx) * _bern(dV)
    left = p.d0 / (2 * dx * dx) * _bern(-dV)
    diag = -np.r_[right, 0] - np.r_[0, left]

    def generator(q):
        z = diag[:, None] * q
        z[1:] += right[:, None] * q[:-1]
        z[:-1] += left[:, None] * q[1:]
        return z

    ab = np.zeros((3, n))
    ab[0, 1:] = -dt / 2 * left
    ab[1] = 1 - dt / 2 * diag
    ab[2, :-1] = -dt / 2 * right
    q0 = np.exp(-x * x / (2 * var))
    q0 /= q0.sum()
    q = np.tile(q0[:, None], (1, nb))
    logS = np.zeros(nb)
    norm = 1 / (math.sqrt(2 * math.pi) * eps * p.rho * p.torus_w ** (p.dim - 1))
    stride = max(1, int(round(record_every / dt)))
    rec_t, rec_lf, rec_sl = [], [], []

    def rates(t):
        mu = p.z_bar + (p.z0 - p.z_bar) * np.exp(-p.gamma * t)
        diff = mu + eps * x[:, None] - centres
        by = norm * w * np.exp(-diff * diff / (2 * (eps * p.rho) ** 2))
        return by, diff, -p.gamma * (mu - p.z_bar)

    def record(t):
        by, diff, mup = rates(t)
        K = by.sum(axis=1)[:, None] * B
        Kt = (-diff * mup / (eps * p.rho) ** 2 * by).sum(axis=1)[:, None] * B
        F = (K * q).sum(axis=0)
        Fp = ((Kt - K * K) * q + K * generator(q)).sum(axis=0)
        rec_t.append(t)
        rec_lf.append(logS + np.log(np.maximum(F, np.finfo(float).tiny)))
        rec_sl.append(np.divide(Fp, F, out=np.zeros_like(F), where=F > 0))

    t0 = time.time()
    record(0.0)
    min_q = 1.0
    for step in range(N):
        t = step * dt
        by, _, _ = rates(t + dt / 2)
        K = by.sum(axis=1)[:, None] * B
        q = q * np.exp(-K * dt / 2)
        rhs = q + dt / 2 * generator(q)
        q = solve_banded((1, 1), ab, rhs, overwrite_b=True, check_finite=False)
        mn = float(q.min())
        min_q = min(min_q, mn)
        if mn < -1e-16:
            raise RuntimeError(f"positivity loss {mn}")
        q = np.maximum(q, 0)
        q = q * np.exp(-K * dt / 2)
        nrm = q.sum(axis=0)
        logS += np.log(nrm)
        q /= nrm
        if (step + 1) % stride == 0 or step + 1 == N:
            record((step + 1) * dt)
    return {"t": np.array(rec_t), "logf": np.array(rec_lf), "dlogf": np.array(rec_sl),
            "budgets": B, "m": m, "eps": eps, "dx": dx, "dt": dt, "L": L, "tmax": tmax,
            "min_q": min_q, "wall_seconds": time.time() - t0,
            "record_every": stride * dt}


def pde_extrema(t, slope, logf, window=fk.WINDOW):
    ii = np.flatnonzero((t >= window[0]) & (t <= window[1]))
    ex = []
    for a, c in zip(ii[:-1], ii[1:]):
        if slope[a] * slope[c] < 0:
            u = abs(slope[a]) / (abs(slope[a]) + abs(slope[c]))
            ex.append({"kind": "max" if slope[a] > 0 else "min",
                       "t": float(t[a] * (1 - u) + t[c] * u),
                       "logf": float(logf[a] * (1 - u) + logf[c] * u)})
    return ex, float(slope[ii[0]]), float(slope[ii[-1]])


def cmd_pde(args):
    OUT.mkdir(parents=True, exist_ok=True)
    m, eps = args.m, args.eps
    dt = args.dt if args.dt else (1e-4 if eps >= 0.025 else 5e-5)
    r = pde_solve(m, eps, BUDGETS, dx=args.dx, dt=dt, L=args.L)
    rows = []
    for j, B in enumerate(BUDGETS):
        ex, s0, s1 = pde_extrema(r["t"], r["dlogf"][:, j], r["logf"][:, j])
        rows.append({"B": B, "extrema": ex, "n_max": sum(e["kind"] == "max" for e in ex),
                     "n_min": sum(e["kind"] == "min" for e in ex),
                     "endpoint_log_slopes": [s0, s1]})
    out = {"m": m, "eps": eps, "model": "contact = 1 killed midpoint OU (continuum), port of "
           "GPT-6 theory_checks/reduced_ou_pde.py", "dx": r["dx"], "dt": r["dt"], "L": r["L"],
           "tmax": r["tmax"], "record_every": r["record_every"], "min_q": r["min_q"],
           "wall_seconds": r["wall_seconds"], "rows": rows}
    stem = OUT / (f"pde_m{m}_eps{eps:g}" + (f"_{args.suffix}" if args.suffix else ""))
    core.write_json(Path(str(stem) + ".json"), fk._jsonable(out))   # NB eps has a '.' -> no with_suffix
    np.savez_compressed(str(stem) + ".npz", t=r["t"], logf=r["logf"], dlogf=r["dlogf"],
                        budgets=r["budgets"])
    print(json.dumps({"m": m, "eps": eps, "wall": r["wall_seconds"],
                      "counts": [(x["B"], x["n_max"], x["n_min"]) for x in rows]}), flush=True)


# ----------------------------------------------------------------------------
# FK census
# ----------------------------------------------------------------------------


def zone_label(t: float, targets, eps: float) -> dict:
    """Nearest passage j, scaled offset s = (t - t_j)/eps and a coarse zone label:
    inner |s| <= 3; matching 3 < |s| and |t - t_j| <= min(0.3, gap_j/4); gap otherwise."""
    j = int(np.argmin([abs(t - tj) for tj in targets]))
    d = abs(t - targets[j])
    gaps = [abs(targets[j] - targets[i]) for i in range(len(targets)) if i != j]
    lim = min(MATCH_DELTA, 0.25 * min(gaps)) if gaps else MATCH_DELTA
    zone = "inner" if d <= INNER_L * eps else ("matching" if d <= lim else "gap_or_exterior")
    return {"nearest_passage": j + 1, "s_scaled": (t - targets[j]) / eps, "zone": zone}


def census_one(t_step, p_step, batch_p, dt, k, m, targets, eps, valleys, window=fk.WINDOW):
    """Census for one declared item at grid spacing k*dt."""
    steps = p_step.size
    nb_full = steps // k
    F = p_step[: nb_full * k].reshape(nb_full, k).sum(1) / (k * dt)
    Fb = batch_p[:, : nb_full * k].reshape(batch_p.shape[0], nb_full, k).sum(2) / (k * dt)
    tb = t_step[: nb_full * k].reshape(nb_full, k)
    tc = 0.5 * (tb[:, 0] - dt + tb[:, -1])            # bin centres (kill times ((i)k dt, (i+1)k dt])
    hg = k * dt
    d = np.diff(F) / hg
    db = np.diff(Fb, axis=1) / hg
    td = 0.5 * (tc[1:] + tc[:-1])
    nbt = Fb.shape[0]
    se = db.std(axis=0, ddof=1) / math.sqrt(nbt)
    msk = (td >= window[0]) & (td <= window[1])
    td, d, se = td[msk], d[msk], se[msk]
    z = np.divide(d, se, out=np.zeros_like(d), where=se > 0)   # se == 0 only where f == 0 exactly
    sgn = np.where(z > ZTHR, 1, np.where(z < -ZTHR, -1, 0))
    nz = np.flatnonzero(sgn)
    changes = []
    for a, b in zip(nz[:-1], nz[1:]):
        if sgn[a] != sgn[b]:
            tt = 0.5 * (td[a] + td[b])
            zl = zone_label(tt, targets, eps)
            changes.append({"kind": "max" if sgn[a] == 1 else "min", "t": float(tt),
                            "t_bracket": [float(td[a]), float(td[b])],
                            "z_before": float(z[a]), "z_after": float(z[b]),
                            "zone": zl["zone"], "nearest_passage": zl["nearest_passage"],
                            "s_scaled": zl["s_scaled"],
                            "basin": int(np.searchsorted(valleys, tt)) + 1,
                            "in_tangency_region": bool(TANGENCY_REGION[0] <= tt <= TANGENCY_REGION[1])})
    n_max = sum(c["kind"] == "max" for c in changes)
    n_min = sum(c["kind"] == "min" for c in changes)
    # undecided stretches (blind spots)
    und = []
    i = 0
    while i < sgn.size:
        if sgn[i] == 0:
            j = i
            while j + 1 < sgn.size and sgn[j + 1] == 0:
                j += 1
            und.append([float(td[i]), float(td[j])])
            i = j + 1
        else:
            i += 1
    und_len = [b - a + hg for a, b in und]
    first = int(sgn[nz[0]]) if nz.size else 0
    last = int(sgn[nz[-1]]) if nz.size else 0
    expected_ok = (n_max == m and n_min == m - 1 and first == 1 and last == -1)
    extra = n_max > m or n_min > m - 1
    # order check: alternating starting with max
    kinds = [c["kind"] for c in changes]
    alternating = all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))
    # blind spots away from detected critical points
    crit_t = [c["t"] for c in changes]
    far_und = [(a, b) for (a, b), L in zip(und, und_len)
               if not any(a - 3 * hg <= ct <= b + 3 * hg for ct in crit_t)]
    basin_max = [sum(1 for c in changes if c["kind"] == "max" and c["basin"] == b) for b in range(1, m + 1)]
    return {"h_g": hg, "k_steps": k, "n_max": n_max, "n_min": n_min, "changes": changes,
            "maxima_per_G_basin": basin_max, "one_max_per_basin": basin_max == [1] * m,
            "first_sign": first, "last_sign": last, "alternating": alternating,
            "exactly_m_pattern": bool(expected_ok), "EXTRA_PAIR": bool(extra),
            "n_points": int(sgn.size), "n_undecided": int((sgn == 0).sum()),
            "undecided_intervals_longest": sorted(zip(und_len, und), reverse=True)[:6],
            "undecided_far_from_critical_points": far_und[:10],
            "n_undecided_far": len(far_und)}


def analyze_fk() -> dict:
    out = {}
    for m in M_LIST:
        for eps in EPS_LIST:
            name = ens_name(m, eps)
            ip = fk.index_path(name)
            if not ip.exists():
                continue
            ens = fk.load_ensemble(name)
            if not ens.index.get("complete"):
                print(f"[n9a] {name} incomplete", flush=True)
                continue
            D = ens.declared()
            dt = D["dt"]
            targets = ens.spec.times()
            valleys = fk.g_valley_times(fk.EnsembleSpec(m=m, eps=eps), [1.0 / m] * m)
            key = f"m{m}_eps{eps:g}"
            rec = {"m": m, "eps": eps, "dt": dt, "N": D["N"], "ensemble": name, "g_valleys": valleys,
                   "tag": int(ens.index["tag"]), "seed_entropy": ens.index["seed_entropy"],
                   "batches": int(D["batch_p_step"].shape[1]), "targets": list(targets),
                   "items": {}}
            for idd, it in enumerate(D["declared"]):
                lab = it["label"]
                rows = {}
                for rres in RES_LIST:
                    k = max(1, int(math.floor(rres * eps / dt + 1e-9)))
                    rows[f"{rres:g}"] = census_one(D["t"], D["p_step"][idd], D["batch_p_step"][idd],
                                                   dt, k, m, targets, eps, valleys)
                pm = D["p_step"][idd]
                late_mass = float(pm[(D["t"] > valleys[-1]) & (D["t"] <= 3.5)].sum())
                pat = [(v["n_max"], v["n_min"]) for v in rows.values()]
                status = ("exactly_m_all_resolutions" if all(v["exactly_m_pattern"] for v in rows.values())
                          else "EXTRA" if any(v["EXTRA_PAIR"] for v in rows.values())
                          else "fewer_than_m_resolved_at_some_resolution")
                rec["items"][lab] = {"B": it["B"], "variant": it["variant"], "by_resolution": rows,
                                     "mass_after_last_G_valley": late_mass, "status": status,
                                     "window_mass": float(D["p_step"][idd][(D["t"] >= 0.5) & (D["t"] <= 3.5)].sum()),
                                     "any_EXTRA_PAIR": any(v["EXTRA_PAIR"] for v in rows.values()),
                                     "exactly_m_at_all_resolutions": all(v["exactly_m_pattern"] for v in rows.values())}
            out[key] = rec
            print(f"[n9a] analyzed {key}", flush=True)
    return out


def cmd_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    fkres = analyze_fk()
    pde = {}
    for m in M_LIST:
        for eps in PDE_EPS:
            p = OUT / f"pde_m{m}_eps{eps:g}.json"
            if p.exists():
                pde[f"m{m}_eps{eps:g}"] = json.loads(p.read_text())
    # headline
    flags = []
    summary_rows = []
    for key, rec in fkres.items():
        for lab, it in rec["items"].items():
            fine = it["by_resolution"][f"{RES_LIST[0]:g}"]
            row = {"cell": key, "item": lab, "B": it["B"], "variant": it["variant"],
                   "counts_by_resolution": {r: [v["n_max"], v["n_min"]] for r, v in it["by_resolution"].items()},
                   "exactly_m_all_res": it["exactly_m_at_all_resolutions"],
                   "any_extra_pair": it["any_EXTRA_PAIR"],
                   "undecided_far_by_resolution": {r: v["n_undecided_far"] for r, v in it["by_resolution"].items()},
                   "tangency_region_changes": [c for v in it["by_resolution"].values() for c in v["changes"]
                                               if c["in_tangency_region"]],
                   "finest_changes": [(c["kind"], round(c["t"], 4), c["zone"]) for c in fine["changes"]]}
            summary_rows.append(row)
            if it["any_EXTRA_PAIR"]:
                flags.append({"cell": key, "item": lab,
                              "detail": {r: v["changes"] for r, v in it["by_resolution"].items() if v["EXTRA_PAIR"]}})
    pde_rows = []
    for key, r in pde.items():
        m = r["m"]
        for row in r["rows"]:
            interior = row["n_max"] == m and row["n_min"] == m - 1
            ok = interior and row["endpoint_log_slopes"][0] > 0 and row["endpoint_log_slopes"][1] < 0
            pde_rows.append({"cell": key, "B": row["B"], "n_max": row["n_max"], "n_min": row["n_min"],
                             "exactly_m_pattern": bool(ok), "interior_exactly_m": bool(interior),
                             "endpoint_log_slopes": row["endpoint_log_slopes"],
                             "left_endpoint_density_underflow": bool(row["endpoint_log_slopes"][0] == 0.0),
                             "extrema": [(e["kind"], round(e["t"], 4)) for e in row["extrema"]]})
            if row["n_max"] > m or row["n_min"] > m - 1:
                flags.append({"cell": key + "_PDE", "B": row["B"], "detail": row["extrema"]})
    # FK 'nogate' (contact = 1, EM at dt) vs PDE (continuum) critical-point times at eps = 0.025
    fk_vs_pde = {}
    for m in M_LIST:
        key = f"m{m}_eps0.025"
        if key in fkres and key in pde:
            rows = []
            for row in pde[key]["rows"]:
                it = fkres[key]["items"].get(f"nogate_B{row['B']:g}")
                if it is None:
                    continue
                ch = it["by_resolution"]["0.25"]["changes"]
                pe = row["extrema"]
                same = [c["kind"] for c in ch] == [e["kind"] for e in pe]
                inside = [c["t_bracket"][0] - 1e-9 <= e["t"] <= c["t_bracket"][1] + 1e-9
                          for c, e in zip(ch, pe)] if same else None
                rows.append({"B": row["B"], "same_pattern": same, "pde_extremum_inside_fk_bracket": inside,
                             "fk_brackets": [c["t_bracket"] for c in ch],
                             "fk_times": [c["t"] for c in ch], "pde_times": [e["t"] for e in pe],
                             "max_abs_time_diff": (max(abs(c["t"] - e["t"]) for c, e in zip(ch, pe))
                                                   if same else None)})
            fk_vs_pde[key] = rows
    refine = {}
    for m in M_LIST:
        for eps in PDE_EPS:
            base_p = OUT / f"pde_m{m}_eps{eps:g}.json"
            ref_p = OUT / f"pde_m{m}_eps{eps:g}_refined.json"
            if base_p.exists() and ref_p.exists():
                b0, r0 = json.loads(base_p.read_text()), json.loads(ref_p.read_text())
                rows = []
                for rb, rr in zip(b0["rows"], r0["rows"]):
                    same = [e["kind"] for e in rb["extrema"]] == [e["kind"] for e in rr["extrema"]]
                    rows.append({"B": rb["B"], "same_pattern": same,
                                 "max_abs_time_diff": max(abs(a["t"] - b["t"]) for a, b in
                                                          zip(rb["extrema"], rr["extrema"])) if same else None,
                                 "max_abs_logf_diff": max(abs(a["logf"] - b["logf"]) for a, b in
                                                          zip(rb["extrema"], rr["extrema"])) if same else None})
                refine[f"m{m}_eps{eps:g}"] = {"base": [b0["dx"], b0["dt"]], "refined": [r0["dx"], r0["dt"]],
                                              "rows": rows}
    out = {"analysis": "N9a TH-6 matching-zone census (FK exact law + contact=1 PDE)",
           "pde_refinement_check": refine,
           "fk_nogate_vs_pde_eps0.025": fk_vs_pde,
           "zthr": ZTHR, "resolutions_h_over_eps": list(RES_LIST), "budgets": list(BUDGETS),
           "zones": {"inner": f"|t-t_j| <= {INNER_L} eps", "matching": f"{INNER_L} eps < |t-t_j| <= {MATCH_DELTA}",
                     "tangency_region_m3": list(TANGENCY_REGION)},
           "POTENTIAL_COUNTEREXAMPLE_FLAGS": flags,
           "n_flags": len(flags), "summary_rows": summary_rows, "pde_rows": pde_rows,
           "fk": fkres, "seeds": {"base_seed": fk.BASE_SEED, "tag": TAG, "replicate": 0}}
    core.write_json(OUT / "n9a_census.json", fk._jsonable(out))
    print(f"[n9a] flags: {len(flags)}")
    for r in summary_rows:
        print(r["cell"], r["item"], r["counts_by_resolution"], "extra" if r["any_extra_pair"] else "")


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    R = json.loads((OUT / "n9a_census.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), constrained_layout=True, sharex=True)
    cmap = plt.get_cmap("viridis")
    for row, m in enumerate(M_LIST):
        for col, eps in enumerate(EPS_LIST):
            ax = axes[row, col]
            name = ens_name(m, eps)
            if not fk.index_path(name).exists():
                ax.set_visible(False)
                continue
            ens = fk.load_ensemble(name)
            D = ens.declared()
            dt = D["dt"]
            k = max(1, int(math.floor(0.25 * eps / dt + 1e-9)))
            for idd, it in enumerate(D["declared"]):
                if it["variant"] != "full":
                    continue
                ib = BUDGETS.index(it["B"])
                p = D["p_step"][idd]
                nbf = p.size // k
                F = p[: nbf * k].reshape(nbf, k).sum(1) / (k * dt)
                tc = D["t"][: nbf * k].reshape(nbf, k).mean(1)
                msk = (tc >= 0.5) & (tc <= 3.5)
                ccol = cmap(ib / (len(BUDGETS) - 1))
                lf = np.full(F.shape, np.nan)
                lf[F > 0] = np.log10(F[F > 0])
                ax.plot(tc[msk], lf[msk], color=ccol, lw=0.8, label=f"B={it['B']:g}")
                cen = R["fk"][f"m{m}_eps{eps:g}"]["items"][it["label"]]["by_resolution"]["0.25"]["changes"]
                for c in cen:
                    if c["kind"] == "max":
                        j = int(np.argmin(np.abs(tc - c["t"])))
                        if F[j] > 0:
                            ax.plot(c["t"], math.log10(F[j]), marker="v", ms=3.0, color="k", lw=0)
                    else:
                        ax.axvline(c["t"], ymin=0.0, ymax=0.07, color=ccol, lw=1.4)
            for tj in ens.spec.times():
                ax.axvline(tj, color="0.7", lw=0.5, ls=":")
            if m == 3:
                ax.axvspan(*TANGENCY_REGION, color=core.OI_YELLOW, alpha=0.25, lw=0)
            ax.set_title(f"$m={m}$, $\\varepsilon={eps:g}$ (dt={dt:g})", fontsize=7.5)
            if row == 1:
                ax.set_xlabel("reaction time $t$ (model time units)")
            if col == 0:
                ax.set_ylabel(r"$\log_{10}$ exact density $f(t)$")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=5.8, loc="lower left", ncol=2)
    written = core.save_figure(fig, core.FIGURES / "fb_n9a_census")
    plt.close(fig)
    # PDE (contact = 1) at eps = 0.025 and 0.0125
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), layout="constrained", sharex=True)
    for row, m in enumerate(M_LIST):
        for col, eps in enumerate(PDE_EPS):
            ax = axes[row, col]
            stem = OUT / f"pde_m{m}_eps{eps:g}"
            if not Path(str(stem) + ".json").exists():
                ax.set_visible(False)
                continue
            d = np.load(str(stem) + ".npz")
            meta = json.loads(Path(str(stem) + ".json").read_text())
            t = d["t"]
            msk = (t >= 0.5) & (t <= 3.5)
            for j, B in enumerate(d["budgets"]):
                ib = BUDGETS.index(float(B))
                ax.plot(t[msk], d["logf"][msk, j] / math.log(10), color=cmap(ib / (len(BUDGETS) - 1)),
                        lw=0.8, label=f"B={B:g}")
                for e in meta["rows"][j]["extrema"]:
                    ax.plot(e["t"], e["logf"] / math.log(10), marker="v" if e["kind"] == "max" else "o",
                            ms=3.0, color="k" if e["kind"] == "max" else core.OI_VERMILLION,
                            mfc="k" if e["kind"] == "max" else "none", lw=0)
            for tj in fk.EnsembleSpec(m=m, eps=eps).times():
                ax.axvline(tj, color="0.7", lw=0.5, ls=":")
            if m == 3:
                ax.axvspan(*TANGENCY_REGION, color=core.OI_YELLOW, alpha=0.25, lw=0)
            ax.set_title(f"PDE, contact = 1: $m={m}$, $\\varepsilon={eps:g}$", fontsize=7.5)
            if row == 1:
                ax.set_xlabel("reaction time $t$ (model time units)")
            if col == 0:
                ax.set_ylabel(r"$\log_{10} f(t)$")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=5.8, loc="lower left", ncol=2)
    written += core.save_figure(fig, core.FIGURES / "fb_n9a_pde")
    plt.close(fig)
    print(written)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--m", type=int, required=True)
    s.add_argument("--eps", type=float, required=True)
    s.add_argument("--chunks", default=None, help="range a-b of chunk indices")
    s.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("pde")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--dx", type=float, default=0.025)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--L", type=float, default=24.0)
    p.add_argument("--suffix", default="", help="output stem suffix (refinement checks)")
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "pde": cmd_pde, "analyze": cmd_analyze, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
