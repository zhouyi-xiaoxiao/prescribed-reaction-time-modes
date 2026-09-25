#!/usr/bin/env python3
"""SAA Newton design of peak times and conditional mass ratios (uplift2 item 1, NU-B / NU-C).

Problem (formulation F1 of the execution spec, fixed budget B)
--------------------------------------------------------------
Unknowns x = (t'_1..t'_m, u_1..u_{m-1}): nominal stripe times t'_j (centres c_j = mu(t'_j)) and
simplex tangent coordinates of the allocation w (w = w0 + sum_k u_k (e_k - e_m)); B is fixed.
Outputs y(x) = (t*_1..t*_m, r_1..r_{m-1}): peak times of the exact discrete-time density and
conditional mass ratios r_j = M_j / sum_i M_i (basins cut at the density's valleys, basins inside
[0, tmax]).  Targets: T_1 < ... < T_m and a ratio profile r (rising ~ j, falling ~ m+1-j, equal).
The yield sum_i M_i is reported, not prescribed (F1 never prescribes m times and m absolute masses).

SAA
---
The exact law of a design is p_n(theta) = E[e^{-B X_{n-1}}(1 - e^{-B e_n})] over unkilled pair paths
(fb_v2_design_estimator).  On ONE fixed ensemble of N paths the sample-average map y_N(x) is smooth in x
and its Jacobian is exact (pathwise derivatives + implicit differentiation), so Newton on y_N(x) = target
converges quadratically to the SAA root x_N; the sample error x_N - x_inf is O(N^{-1/2}) (delta method:
Cov x_N ~ J^{-1} Cov(y_N) J^{-T}, estimated here with the delete-one-group jackknife and checked on
disjoint sub-ensembles).

Peak and valley functionals (per-passage bandwidths)
----------------------------------------------------
The estimator's functionals use one Gaussian bandwidth h.  Passage j has time width
sigma_j = sigma_Z / v(T_j) (sigma_Z = eps sqrt(D0/(2 gamma) + rho^2), v = |mu'|), which grows like
e^{gamma T_j}: at eps = 0.025 the first peak has sigma ~ 0.017 and the last ~ 0.13.  Here peak j is the
maximiser of f_{h_j} = sum_n p_n K_{h_j}(t - t_n), with h_j = clip(kappa sigma_j, h_lo, h_hi) FIXED per cell
(from the targets, so the functional does not move with the design and stays smooth); the valley between
peaks j and j+1 is the minimiser of f_{h_v}, h_v = min(h_j, h_{j+1}); masses use the smoothed CDF with h_v
at each valley cut (bias O(h^4) there) and the exact end points 0 and tmax.  The smoothing offset
(peak of f_h minus the raw argmax, measured on the noiseless mean-field curve) is reported per cell.
With a scalar h (all h_j equal) the functionals coincide with fb_v2_design_estimator._functionals.

Warm start
----------
Prop. 1 (limit allocation of the ratio profile at budget B, fb_allocation_law.max_common_scale) with
stripes at the targets, then a homotopy in B (B0 -> B) of Newton on the noiseless EM mean-field law
(fb_v2_design_estimator.mean_field_law; analytic gradient), then SAA Newton on a small ensemble, then
on the full ensemble.

Evaluation engine
-----------------
StreamEngine: persistent spawn workers; each owns a set of chunk ids of the SAME SeedSequence layout as
fb_v2_design_estimator.Engine (same paths), regenerates its chunks on every pass (no 16 GB store for
1e6 paths) unless the chunk fits into the per-worker keep budget, and evaluates all queued designs on
them.  Designs of all cells at one eps share one ensemble (paths do not depend on the design).

CLI (outputs under artifacts/data/exact_m_fixed_budget/V2_design/)
    python3 fb_v2_saa_newton.py mf          # warm starts (Prop. 1 + mean-field homotopy), all cells
    python3 fb_v2_saa_newton.py demo --eps 0.05 --n 5e5
    python3 fb_v2_saa_newton.py showcase    # residual table + sample-size study
    python3 fb_v2_saa_newton.py failed      # (3,0.1,8), (3,0.1,4) without the 3.5 cap
Multiprocessing uses the spawn context: entry point guarded by ``if __name__ == "__main__"``.
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

import fb_v2_design_estimator as de  # noqa: E402
import fb_allocation_law as alloc  # noqa: E402

fk = de.fk
OUT = de.FB_DATA / "V2_design"

# ----------------------------------------------------------------------------
# Targets and cells
# ----------------------------------------------------------------------------

# Anchor target times (fk.TARGET_TIMES) and their nested extension: m = 4 inserts the midpoint of
# the widest anchor gap (1.6, 2.8) -> 2.2; m = 5 also inserts the midpoint of (0.8, 1.6) -> 1.2.
TARGET_TIMES = {2: tuple(fk.TARGET_TIMES[2]), 3: tuple(fk.TARGET_TIMES[3]),
                4: (0.8, 1.6, 2.2, 2.8), 5: (0.8, 1.2, 1.6, 2.2, 2.8)}
PROFILES = ("rising", "falling", "equal")
KAPPA = 0.10            # h_j = clip(KAPPA * sigma_j, H_LO, H_HI); MF smoothing offset <~ 1e-3
H_LO = 0.003
H_HI = 0.05
FLAT_VALLEY_REL = 1e-12  # valley treated as an empty gap below this fraction of the smaller adjacent peak


def profile(m: int, kind: str) -> np.ndarray:
    j = np.arange(1, m + 1, dtype=float)
    r = {"rising": j, "falling": m + 1.0 - j, "equal": np.ones(m)}[kind]
    return r / r.sum()


def cell_key(m: int, eps: float, B: float, kind: str, tag: str = "") -> str:
    return f"m{m}_eps{eps:g}_B{B:g}_{kind}" + (f"_{tag}" if tag else "")


def sigma_z(spec) -> float:
    """Stationary sd of Z (EM, small dt) combined with the stripe sd eps*rho."""
    return math.sqrt(spec.eps ** 2 * spec.d0 / (2.0 * spec.gamma) + spec.sd() ** 2)


def bandwidths(T, spec, kappa: float = KAPPA, lo: float = H_LO, hi: float = H_HI) -> np.ndarray:
    """Per-passage bandwidths h_j = clip(kappa sigma_Z / v(T_j), lo, hi) (fixed per cell)."""
    s = sigma_z(spec) / de.speed(np.asarray(T, float), spec)
    return np.clip(kappa * s, lo, hi)


# ----------------------------------------------------------------------------
# Functionals with per-passage bandwidths (+ exact SAA Jacobians)
# ----------------------------------------------------------------------------


class DesignMapError(RuntimeError):
    """A peak or valley of the prescribed structure could not be located (design left the chart)."""


def _hv(hs, j):
    return float(min(hs[j], hs[j + 1]))


def functionals(p, dp, tgrid, hs, cuts0, anchors=None):
    """Peaks (f_{h_j} maximisers), valleys (f_{h_v} minimisers), valley-cut masses and ratios, with
    raw-parameter Jacobians (IFT).  anchors (from the full sample) fix the followed stationary points."""
    hs = np.asarray(hs, float)
    m = hs.size
    dtg = float(tgrid[1] - tgrid[0])
    tmax = float(tgrid[-1])
    if anchors is None:
        if len(cuts0) != m + 1:
            raise DesignMapError(f"need {m + 1} cuts, got {len(cuts0)}")
        peaks, pk_ok, pk_int = [], [], []
        for j in range(m):
            sel = np.flatnonzero((tgrid > cuts0[j]) & (tgrid <= cuts0[j + 1]))
            if sel.size < 5:
                raise DesignMapError(f"basin {j} has {sel.size} grid points")
            fh = de.smooth_grid(p, dtg, hs[j], 0)[sel]
            k = int(np.argmax(fh))
            pk_int.append(bool(2 <= k <= sel.size - 3))
            t_, ok = de.refine_stationary(p, tgrid, hs[j], tgrid[sel[k]], "max")
            peaks.append(t_)
            pk_ok.append(ok)
        valleys, vl_ok, vl_flat = [], [], []
        for j in range(m - 1):
            hv = _hv(hs, j)
            sel = np.flatnonzero((tgrid > peaks[j]) & (tgrid < peaks[j + 1]))
            if sel.size < 3:
                raise DesignMapError(f"no room for valley {j}")
            fh = de.smooth_grid(p, dtg, hv, 0)[sel]
            fref = min(float(de.smooth_at(peaks[j], p, tgrid, hv, 0)), float(de.smooth_at(peaks[j + 1], p, tgrid, hv, 0)))
            flat = np.flatnonzero(fh <= FLAT_VALLEY_REL * fref)
            if flat.size:
                # empty gap (small eps): the smoothed density vanishes to below FLAT_VALLEY_REL of the smaller
                # adjacent peak.  Every cut inside the gap gives the same masses (to that relative level), so
                # cut at the midpoint of the longest such run, with zero cut derivative (f(cut) ~ 0).
                runs = np.split(flat, np.flatnonzero(np.diff(flat) > 1) + 1)
                r = max(runs, key=len)
                valleys.append(0.5 * float(tgrid[sel[r[0]]] + tgrid[sel[r[-1]]]))
                vl_ok.append(True)
                vl_flat.append(True)
                continue
            k = int(np.argmin(fh))
            t_, ok = de.refine_stationary(p, tgrid, hv, tgrid[sel[k]], "min")
            valleys.append(t_)
            vl_ok.append(bool(ok and 1 <= k <= sel.size - 2))
            vl_flat.append(False)
    else:
        pk = [de.refine_stationary(p, tgrid, hs[j], a, "max") for j, a in enumerate(anchors["peaks"])]
        peaks, pk_ok = [x[0] for x in pk], [x[1] for x in pk]
        pk_int = list(anchors["peaks_interior"])
        vl_flat = list(anchors.get("valleys_flat", [False] * len(anchors["valleys"])))
        vv = [(a, True) if vl_flat[j] else de.refine_stationary(p, tgrid, _hv(hs, j), a, "min")
              for j, a in enumerate(anchors["valleys"])]
        valleys, vl_ok = [x[0] for x in vv], [x[1] for x in vv]
    out = {"peaks": peaks, "peaks_ok": pk_ok, "peaks_interior": pk_int, "valleys": valleys,
           "valleys_ok": vl_ok, "valleys_flat": vl_flat, "h": hs.tolist()}
    out["f_peak"] = [float(de.smooth_at(t, p, tgrid, hs[j], 0)) for j, t in enumerate(peaks)]
    out["f_valley"] = [float(de.smooth_at(t, p, tgrid, _hv(hs, j), 0)) for j, t in enumerate(valleys)]
    grad = dp is not None
    K = dp.shape[0] if grad else 0
    if grad:
        dpk = np.array([-de.smooth_at(t, dp, tgrid, hs[j], 1) / de.smooth_at(t, p, tgrid, hs[j], 2)
                        for j, t in enumerate(peaks)])
        dvl = np.array([np.zeros(K) if vl_flat[j] else
                        -de.smooth_at(t, dp, tgrid, _hv(hs, j), 1) / de.smooth_at(t, p, tgrid, _hv(hs, j), 2)
                        for j, t in enumerate(valleys)]).reshape(len(valleys), K)
    cuts = [0.0] + list(valleys) + [tmax]
    total = float(p.sum())
    Fc, dFc = [0.0], [np.zeros(K)] if grad else None
    for j, s in enumerate(valleys):
        hv = _hv(hs, j)
        Fc.append(float(de.smooth_at(s, p, tgrid, hv, "cdf")))
        if grad:
            dFc.append(de.smooth_at(s, dp, tgrid, hv, "cdf") + de.smooth_at(s, p, tgrid, hv, 0) * dvl[j])
    Fc.append(total)
    if grad:
        dFc.append(dp.sum(axis=1))
    M = np.diff(Fc)
    out.update({"cuts": cuts, "masses": M.tolist(), "total_mass": total, "yield": float(M.sum()),
                "ratios": (M / M.sum()).tolist()})
    if grad:
        dF = np.array(dFc)
        dM = np.diff(dF, axis=0)
        dtot = dM.sum(axis=0)
        dr = (dM - (M / M.sum())[:, None] * dtot[None, :]) / M.sum()
        out["jac"] = {"peaks": dpk, "valleys": dvl, "masses": dM, "ratios": dr, "yield": dtot}
    return out


def analyse(law, hs, cuts0, *, jackknife: bool = True, jackknife_jac: bool = False) -> dict:
    """functionals() on the full sample plus delete-one-group jackknife SEs (stationary points
    followed from the full-sample ones)."""
    full = functionals(law.p, law.dp, law.t, hs, cuts0)
    res = {k: v for k, v in full.items() if k != "jac"}
    if "jac" in full:
        res["jac"] = full["jac"]
    res["n_paths"] = law.n_paths
    res["groups"] = int(law.sizes.size)
    if jackknife and law.sizes.size >= 2:
        G = law.sizes.size
        keys = ("peaks", "valleys", "masses", "ratios", "yield")
        reps = {k: [] for k in keys}
        jreps = []
        for g in range(G):
            p_g, dp_g = law.loo(g)
            o = functionals(p_g, dp_g if jackknife_jac else None, law.t, hs, cuts0, anchors=full)
            for k in keys:
                reps[k].append(np.atleast_1d(np.asarray(o[k], float)))
            if jackknife_jac:
                jreps.append(o["jac"])
        res["jk_reps"] = {k: np.array(v) for k, v in reps.items()}
        res["se"] = {k: np.sqrt((G - 1) / G * np.sum((a - a.mean(0)) ** 2, axis=0))
                     for k, a in res["jk_reps"].items()}
        if jackknife_jac:
            res["jk_jac"] = jreps
    return res


def f1_outputs(res: dict, m: int):
    """F1 outputs y = (peaks, ratios[:m-1]) and raw Jacobian rows."""
    y = np.concatenate([res["peaks"], res["ratios"][:m - 1]])
    J = None
    if "jac" in res:
        J = np.vstack([res["jac"]["peaks"], res["jac"]["ratios"][:m - 1]])
    return y, J


def mode_check(law, hs, res, rel_prominence: float = 1e-3) -> dict:
    """Exactly-m-peaks diagnostic under the per-basin bandwidths: in basin j (between the valley cuts)
    f_{h_j} must have exactly one local maximum with prominence > rel_prominence * max f."""
    t, p = law.t, law.p
    dtg = float(t[1] - t[0])
    cuts = res["cuts"]
    counts = []
    for j, h in enumerate(res["h"]):
        f = de.smooth_grid(p, dtg, h, 0)
        lo = cuts[j] if j > 0 else 0.2
        hi = cuts[j + 1] if j + 1 < len(cuts) - 1 else t[-1] - 4 * h
        sel = np.flatnonzero((t > lo) & (t < hi))
        fs = f[sel]
        top = float(f.max())
        n = 0
        for i in range(1, fs.size - 1):
            if fs[i] > fs[i - 1] and fs[i] >= fs[i + 1]:
                left = fs[:i].min() if i > 0 else fs[i]
                right = fs[i + 1:].min()
                if fs[i] - max(left, right) > rel_prominence * top:
                    n += 1
        counts.append(n)
    # tail beyond the last peak: no further significant maximum (checked in the last basin above)
    return {"per_basin_counts": counts, "exactly_m": bool(all(c == 1 for c in counts)),
            "valleys_strict": bool(all(res["valleys_ok"])), "rel_prominence": rel_prominence}


def cuts_from(res: dict, tmax: float) -> list:
    return [0.0] + [float(v) for v in res["valleys"]] + [float(tmax)]


# ----------------------------------------------------------------------------
# Design coordinates, Newton step, warm start
# ----------------------------------------------------------------------------


def design_from_x(tp, w, B, spec, label: str = "") -> de.Design:
    return de.Design.from_times(np.asarray(tp, float), np.asarray(w, float), float(B), spec, label)


def tangent_jac(Jraw, design: de.Design, spec) -> np.ndarray:
    """Raw columns (c, w, B) -> (t'_1..t'_m, u_1..u_{m-1})."""
    return de.to_design_coords(Jraw, design, spec, "times_tangent")


def apply_step(tp, w, dx, a):
    m = len(tp)
    tn = np.asarray(tp, float) + a * dx[:m]
    wn = np.asarray(w, float).copy()
    wn[:m - 1] += a * dx[m:]
    wn[m - 1] -= a * float(np.sum(dx[m:]))
    return tn, wn


def admissible(tn, wn, spec, *, t_cap: float | None, sep: float = 0.02) -> bool:
    if np.any(wn <= 1e-7) or np.any(np.diff(tn) < sep) or tn[0] <= 0.05:
        return False
    if t_cap is not None and tn[-1] > t_cap + 1e-12:
        return False
    return True


def prop1_weights(T, r, B) -> dict:
    """Prop. 1: limit allocation realising the ratio profile r at budget B (stripes at T)."""
    v = alloc.speeds(list(T))
    out = alloc.max_common_scale(float(B), list(map(float, r)), v, 1.0)
    return {"w": np.array(out["weights"]), "theta": out["theta"], "masses": out["masses"],
            "exposures": out["exposures"]}


def residual_norm(res_vec) -> float:
    return float(np.max(np.abs(res_vec)))


def mf_eval(tp, w, B, spec, hs, cuts0):
    d = design_from_x(tp, w, B, spec)
    law = de.mean_field_law(d, spec)
    r = functionals(law.p, law.dp, law.t, hs, cuts0)
    if not all(r["peaks_ok"]) or not all(r["valleys_ok"]) or not all(r["peaks_interior"]):
        raise DesignMapError("mean-field structure lost")
    return d, law, r


def mf_newton(T, r_t, B, spec, hs, tp0, w0, *, t_cap=None, tol=1e-12, it_max=40, cuts0=None):
    """Newton on the noiseless mean-field F1 map (analytic Jacobian) from (tp0, w0)."""
    m = len(T)
    tgt = np.concatenate([T, r_t[:m - 1]])
    tp, w = np.asarray(tp0, float), np.asarray(w0, float)
    cuts = cuts0 if cuts0 is not None else de.g_clock_cuts(design_from_x(tp, w, B, spec), spec)
    d, law, r = mf_eval(tp, w, B, spec, hs, cuts)
    hist = []
    for _ in range(it_max):
        y, Jraw = f1_outputs(r, m)
        F = y - tgt
        nr = residual_norm(F)
        hist.append(nr)
        if nr < tol:
            break
        dx = np.linalg.solve(tangent_jac(Jraw, d, spec), -F)
        a, accepted = 1.0, False
        for _ in range(25):
            tn, wn = apply_step(tp, w, dx, a)
            if admissible(tn, wn, spec, t_cap=t_cap):
                try:
                    dn, lawn, rn = mf_eval(tn, wn, B, spec, hs, cuts_from(r, spec.tmax))
                    yn, _ = f1_outputs(rn, m)
                    if residual_norm(yn - tgt) < (1 - 1e-4 * a) * nr or nr < 1e-9:
                        accepted = True
                        break
                except DesignMapError:
                    pass
            a *= 0.5
        if not accepted:
            break
        tp, w, d, law, r = tn, wn, dn, lawn, rn
    return {"tp": tp, "w": w, "res": r, "hist": hist, "converged": bool(hist and hist[-1] < tol * 10),
            "cuts": cuts_from(r, spec.tmax)}


def mf_warm_start(T, r_t, B, spec, hs, *, t_cap=None, B0: float = 0.5, grow: float = 1.25) -> dict:
    """Prop. 1 weights at stripes = targets, then mean-field Newton along a B-homotopy B0 -> B."""
    T = np.asarray(T, float)
    path = []
    Bk = min(B0, B)
    w = prop1_weights(T, r_t, Bk)["w"]
    tp = T.copy()
    step = grow
    last = None
    while True:
        try:
            out = mf_newton(T, r_t, Bk, spec, hs, tp, w, t_cap=t_cap)
            ok = out["converged"]
        except DesignMapError:
            ok = False
        if ok:
            tp, w, last = out["tp"], out["w"], out
            path.append({"B": Bk, "tp": tp.tolist(), "w": w.tolist(), "iters": len(out["hist"])})
            if Bk >= B:
                break
            Bk = min(B, Bk * step)
            # Prop. 1 re-scaling of the weights to the new budget as predictor
            w = np.clip(w, 1e-6, None)
            w = w / w.sum()
        else:
            if last is None or step < 1.005:
                raise DesignMapError(f"mean-field homotopy failed at B={Bk:.4g}")
            Bprev = path[-1]["B"]
            step = math.sqrt(step)
            Bk = min(B, Bprev * step)
            tp, w = np.array(path[-1]["tp"]), np.array(path[-1]["w"])
    last["homotopy"] = path
    return last


# ----------------------------------------------------------------------------
# Streaming evaluation engine (same paths as fb_v2_design_estimator.Engine)
# ----------------------------------------------------------------------------


def _stream_worker(conn, spec_dict: dict, tasks: list, group: int, keep: int) -> None:
    spec = fk.EnsembleSpec.from_dict(spec_dict)
    mz = de.em_mean_z(spec)
    kept = {}
    info = []
    for t in tasks[:keep]:
        g = de.generate_chunk(spec, t["size"], t["seedseq"])
        kept[t["chunk"]] = (g["zeta"], de.block_max_abs_zeta(g["zeta"]))
        info.append({"chunk": t["chunk"], "max_abs_zeta": g["max_abs_zeta"], "contact_steps": g["contact_steps"]})
    conn.send(("ready", info))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            conn.close()
            return
        if msg[0] != "eval":
            continue
        _, designs, want_grad, chunk_sets = msg
        try:
            out = [dict() for _ in designs]
            for t in tasks:
                ci = t["chunk"]
                todo = [i for i, cs in enumerate(chunk_sets) if cs is None or ci in cs]
                if not todo:
                    continue
                if ci in kept:
                    zeta, bm = kept[ci]
                else:
                    g = de.generate_chunk(spec, t["size"], t["seedseq"])
                    zeta, bm = g["zeta"], de.block_max_abs_zeta(g["zeta"])
                for i in todo:
                    d = de.Design.from_dict(designs[i])
                    r = de.evaluate_chunk(zeta, bm, spec, d, want_grad=want_grad, group=group, mz=mz)
                    # jackknife needs group sums of y only; the Jacobian is kept as a chunk total
                    out[i][ci] = {"Y": r["Y"], "sizes": r["sizes"],
                                  "dY": None if r["dY"] is None else r["dY"].sum(axis=0)}
                del zeta
            conn.send(("ok", out))
        except Exception as exc:  # noqa: BLE001
            import traceback
            conn.send(("error", traceback.format_exc() + repr(exc)))


class StreamEngine:
    """Evaluate designs on a fixed ensemble whose chunks are regenerated on every pass.

    Chunk i of an ensemble (spec, tag, replicate, chunk size) uses
    SeedSequence(fk.path_entropy(...)).spawn(n)[i] exactly as fb_v2_design_estimator.Engine, so the
    paths coincide with the estimator's for the same arguments; sub-ensembles are sets of chunk ids.
    """

    def __init__(self, spec, n_paths: int, *, tag: int, replicate: int = 0, chunk: int = 25_000,
                 group: int = 5_000, workers: int = 3, keep_per_worker: int = 0, wait: bool = True):
        self.spec = spec
        self.group = group
        self.plan = de.chunk_plan(spec, n_paths, tag=tag, replicate=replicate, chunk=chunk)
        self.chunk_ids = list(range(len(self.plan["sizes"])))
        tasks = [{"chunk": i, "size": self.plan["sizes"][i], "seedseq": self.plan["children"][i]}
                 for i in self.chunk_ids]
        self.n_paths = int(sum(self.plan["sizes"]))
        self.workers = max(1, min(int(workers), de.MAX_LOCAL_WORKERS, len(tasks)))
        self.meta = {"n_paths": self.n_paths, "chunk": chunk, "group": group, "seed": de.BASE_SEED,
                     "tag": tag, "replicate": replicate, "entropy": self.plan["entropy"],
                     "n_chunks": len(tasks), "workers": self.workers, "dt": spec.dt, "tmax": spec.tmax,
                     "eps": spec.eps,
                     "rng": "numpy Philox; chunk i uses SeedSequence(entropy).spawn(n_chunks)[i], entropy = "
                            "fk.path_entropy(spec, seed, tag, replicate, chunk) (as fb_v2_design_estimator.Engine)"}
        if wait:
            self.meta["cpu_wait"] = de.wait_for_cpu()
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for k in range(self.workers):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_stream_worker,
                             args=(b, spec.to_dict(), tasks[k::self.workers], group, keep_per_worker),
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
        self.meta["kept_chunks"] = sorted(r["chunk"] for r in info)
        self.passes = 0
        self.pass_seconds = []

    def evaluate(self, designs, want_grad: bool = True, chunk_sets=None):
        """Laws of designs (list) on the full ensemble or on per-design chunk subsets."""
        ds = list(designs)
        if chunk_sets is None:
            chunk_sets = [None] * len(ds)
        cs = [None if c is None else sorted(set(int(x) for x in c)) for c in chunk_sets]
        dicts = [d.to_dict() for d in ds]
        t0 = time.time()
        for cn in self._conns:
            cn.send(("eval", dicts, want_grad, cs))
        parts = []
        for cn in self._conns:
            tag_, payload = cn.recv()
            if tag_ != "ok":
                raise RuntimeError(payload)
            parts.append(payload)
        wall = time.time() - t0
        self.passes += 1
        self.pass_seconds.append(wall)
        laws = []
        tgrid = fk.step_times(self.spec.dt, self.spec.steps())
        for i, d in enumerate(ds):
            per = {}
            for part in parts:
                per.update(part[i])
            keys = sorted(per)
            gY = np.concatenate([per[k]["Y"] for k in keys])
            sizes = np.concatenate([per[k]["sizes"] for k in keys])
            N = float(sizes.sum())
            dp = (np.sum([per[k]["dY"] for k in keys], axis=0) / N) if want_grad else None
            laws.append(de.Law(self.spec, d, tgrid, gY.sum(0) / N, dp, gY, None, sizes,
                               {"chunks": keys, "n_paths": int(N), "pass_seconds": wall}))
        return laws

    def close(self):
        for cn in self._conns:
            try:
                cn.send(("stop",))
            except Exception:  # noqa: BLE001
                pass
        for pr in self._procs:
            pr.join(timeout=30)
        self._conns, self._procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------
# SAA Newton (lockstep over cells sharing one ensemble)
# ----------------------------------------------------------------------------


class Cell:
    """One F1 design problem: targets (T, r) at fixed (eps, B); state of its Newton iteration."""

    def __init__(self, m, eps, B, kind, spec, *, T=None, r=None, t_cap=None, tag="", kappa=KAPPA,
                 chunk_set=None):
        self.m, self.eps, self.B, self.kind = int(m), float(eps), float(B), kind
        self.spec = spec
        self.T = np.asarray(T if T is not None else TARGET_TIMES[m], float)
        self.r = np.asarray(r if r is not None else profile(m, kind), float)
        self.target = np.concatenate([self.T, self.r[:m - 1]])
        self.hs = bandwidths(self.T, spec, kappa=kappa)
        self.kappa = kappa
        self.t_cap = t_cap
        self.key = cell_key(m, eps, B, kind, tag)
        self.chunk_set = chunk_set
        self.tp = self.w = None
        self.cuts = None
        self.res = None          # analysis of the current accepted design
        self.F = None
        self.hist = []
        self.status = "new"
        self.cand = None
        self.a = 1.0
        self.dx = None
        self.n_backtracks = 0

    def design(self, tp=None, w=None) -> de.Design:
        return design_from_x(self.tp if tp is None else tp, self.w if w is None else w, self.B, self.spec,
                             self.key)

    def set_start(self, tp, w, cuts):
        self.tp, self.w = np.asarray(tp, float), np.asarray(w, float)
        self.cuts = list(cuts)
        self.status = "active"
        self.cand = None

    def to_dict(self) -> dict:
        d = self.design()
        return {"key": self.key, "m": self.m, "eps": self.eps, "B": self.B, "profile": self.kind,
                "targets_T": self.T.tolist(), "targets_r": self.r.tolist(), "bandwidths_h": self.hs.tolist(),
                "kappa": self.kappa, "t_cap": self.t_cap, "tmax": self.spec.tmax, "dt": self.spec.dt,
                "status": self.status, "stripe_times": self.tp.tolist(), "c": d.c.tolist(),
                "w": self.w.tolist(), "hist": self.hist}


def _analyse_cell(cell: Cell, law, jackknife=False):
    res = analyse(law, cell.hs, cell.cuts, jackknife=jackknife)
    if not all(res["peaks_ok"]) or not all(res["valleys_ok"]) or not all(res["peaks_interior"]):
        raise DesignMapError(f"structure lost: peaks_ok={res['peaks_ok']} valleys_ok={res['valleys_ok']} "
                             f"interior={res['peaks_interior']}")
    y, Jraw = f1_outputs(res, cell.m)
    return res, y, Jraw


def saa_newton(engine: StreamEngine, cells: list, *, tol: float = 1e-10, it_max: int = 12,
               max_halvings: int = 6, log=print, stage: str = "") -> list:
    """Lockstep damped Newton on the SAA F1 maps of all cells (one engine pass per iteration).

    Each pass evaluates, per active cell, either its current design (first pass) or its candidate
    x + a dx; a candidate is accepted if ||F||_2 decreases by the Armijo factor (1 - 1e-4 a), else a is
    halved.  Converged: max|F| < tol, or the full Newton step no longer decreases ||F|| while
    max|F| < 1e-8 (round-off floor)."""
    for c in cells:
        c.cand = None
        c.a = 1.0
    it = 0
    while True:
        todo = [c for c in cells if c.status == "active"]
        if not todo:
            break
        if it > it_max * (max_halvings + 1):
            for c in todo:
                c.status = "failed_maxiter"
            break
        designs, sets = [], []
        for c in todo:
            if c.res is None:
                designs.append(c.design())
            else:
                tn, wn = c.cand
                designs.append(c.design(tn, wn))
            sets.append(c.chunk_set)
        laws = engine.evaluate(designs, want_grad=True, chunk_sets=sets)
        it += 1
        for c, law in zip(todo, laws):
            try:
                res, y, Jraw = _analyse_cell(c, law)
                F = y - c.target
                ok = True
            except DesignMapError as exc:
                ok = False
                err = str(exc)
            if c.res is None:            # evaluation of the starting design
                if not ok:
                    c.status = "failed_start"
                    c.hist.append({"stage": stage, "error": err})
                    continue
                c.res, c.F, c.Jraw, c.law = res, F, Jraw, law
                c.hist.append({"stage": stage, "iter": 0, "res_inf": residual_norm(F),
                               "res_2": float(np.linalg.norm(F)), "a": None, "n_paths": law.n_paths,
                               "peaks": res["peaks"], "ratios": res["ratios"]})
            else:
                n0 = float(np.linalg.norm(c.F))
                n1 = float(np.linalg.norm(F)) if ok else np.inf
                if ok and (n1 <= (1.0 - 1e-4 * c.a) * n0):
                    c.tp, c.w = c.cand
                    c.res, c.F, c.Jraw, c.law = res, F, Jraw, law
                    c.cuts = cuts_from(res, c.spec.tmax)
                    c.hist.append({"stage": stage, "iter": len([h for h in c.hist if h.get("stage") == stage]),
                                   "res_inf": residual_norm(F), "res_2": n1, "a": c.a,
                                   "n_paths": law.n_paths, "peaks": res["peaks"], "ratios": res["ratios"]})
                    c.cand = None
                else:
                    if residual_norm(c.F) < 1e-8 and c.a == 1.0:
                        c.status = "converged"
                        c.hist.append({"stage": stage, "note": "round-off floor: full step does not decrease",
                                       "res_inf_candidate": residual_norm(F) if ok else None})
                        continue
                    c.a *= 0.5
                    c.n_backtracks += 1
                    if c.a < 0.5 ** max_halvings:
                        c.status = "failed_linesearch"
                        c.hist.append({"stage": stage, "error": "line search failed" + ("" if ok else ": " + err)})
                        continue
                    c.cand = apply_step(c.tp, c.w, c.dx, c.a)
                    if not admissible(*c.cand, c.spec, t_cap=c.t_cap):
                        c.cand = None
                        c.a *= 0.5
                        c.cand = apply_step(c.tp, c.w, c.dx, c.a)
                    continue
            # new Newton direction from the accepted state
            if residual_norm(c.F) < tol:
                c.status = "converged"
                continue
            nsteps = len([h for h in c.hist if h.get("stage") == stage and "iter" in h])
            if nsteps > it_max:
                c.status = "failed_maxiter"
                continue
            J = tangent_jac(c.Jraw, c.design(), c.spec)
            try:
                c.dx = np.linalg.solve(J, -c.F)
            except np.linalg.LinAlgError:
                c.status = "failed_singular_jacobian"
                c.hist.append({"stage": stage, "error": "singular Jacobian"})
                continue
            c.a = 1.0
            while True:
                c.cand = apply_step(c.tp, c.w, c.dx, c.a)
                if admissible(*c.cand, c.spec, t_cap=c.t_cap) or c.a < 0.5 ** max_halvings:
                    break
                c.a *= 0.5
        msg = " | ".join(f"{c.key}:{c.status[:4]} {residual_norm(c.F) if c.F is not None else float('nan'):.1e} a={c.a:g}"
                         for c in cells)
        log(f"[{stage} pass {it} {engine.pass_seconds[-1]:.0f}s] {msg}")
    return cells


def finalize_cell(cell: Cell, engine: StreamEngine | None = None, law=None) -> dict:
    """Jackknife SEs at the final design, delta-method design covariance, structure checks."""
    law = law if law is not None else getattr(cell, "law", None)
    if law is None:
        law = engine.evaluate([cell.design()], chunk_sets=[cell.chunk_set])[0]
    res = analyse(law, cell.hs, cell.cuts, jackknife=True)
    y, Jraw = f1_outputs(res, cell.m)
    J = tangent_jac(Jraw, cell.design(), cell.spec)
    reps = np.hstack([res["jk_reps"]["peaks"], res["jk_reps"]["ratios"][:, :cell.m - 1]])
    G = reps.shape[0]
    Jinv = np.linalg.inv(J)
    with np.errstate(all="ignore"):      # macOS Accelerate emits spurious FP flags in matmul
        dev = reps - reps.mean(0)
        cov_y = (G - 1) / G * (dev.T @ dev)
        cov_x = Jinv @ cov_y @ Jinv.T
    if not (np.all(np.isfinite(cov_x)) and np.all(np.isfinite(cov_y))):
        raise FloatingPointError("non-finite design covariance")
    mc = mode_check(law, cell.hs, res)
    d = cell.design()
    mf = de.mean_field_law(d, cell.spec, want_grad=False)
    try:
        mfr = functionals(mf.p, None, mf.t, cell.hs, cuts_from(res, cell.spec.tmax))
        mf_peaks = mfr["peaks"]
    except Exception:  # noqa: BLE001
        mf_peaks = None
    out = cell.to_dict()
    out.update({
        "n_paths": law.n_paths, "jackknife_groups": G,
        "achieved": {"peaks": res["peaks"], "ratios": res["ratios"], "masses": res["masses"],
                     "yield": res["yield"], "valleys": res["valleys"], "f_peak": res["f_peak"],
                     "f_valley": res["f_valley"], "cuts": res["cuts"]},
        "se": {k: np.asarray(v).tolist() for k, v in res["se"].items()},
        "residual": (y - cell.target).tolist(), "residual_inf": residual_norm(y - cell.target),
        "jacobian_times_tangent": J.tolist(), "cond_J": float(np.linalg.cond(J)),
        "det_J": float(np.linalg.det(J)),
        "design_se_delta": np.sqrt(np.diag(cov_x)).tolist(),
        "design_se_labels": [f"t'{j + 1}" for j in range(cell.m)] + [f"u{k + 1}" for k in range(cell.m - 1)],
        "mode_check": mc, "valley_to_peak": [fv / min(a, b) for fv, a, b in
                                             zip(res["f_valley"], res["f_peak"][:-1], res["f_peak"][1:])],
        "hypotheses": {"weights_positive": bool(np.all(cell.w > 0)), "min_weight": float(cell.w.min()),
                       "stripes_ordered": bool(np.all(np.diff(cell.tp) > 0)),
                       "last_stripe_time": float(cell.tp[-1]), "inside_3p5_window": bool(cell.tp[-1] <= 3.5)},
        "mean_field_peaks_at_design": mf_peaks,
        "engine": engine.meta if engine is not None else None,
    })
    return out


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------

TAG_SHOWCASE = de.TAGS["NU_B"]     # 203: showcase residual table + sample-size study
TAG_DEMO = de.TAGS["NU_C"]         # 204: demonstrations and the two rerun cells (replicate 0)
OOS_REPLICATE = 1                  # tag 204, replicate 1: independent FK ensemble (out-of-sample check)
DEMO_EPS = (0.05, 0.025)
DEMO_B = (4.0, 8.0)
DEMO_M = (4, 5)
CHUNK = 25_000
GROUP = 5_000


def raw_peaks(law, cuts) -> list:
    """3-point-parabola argmax of the raw per-step density in each basin (smoothing-offset reference)."""
    t, dens = law.t, law.p / law.spec.dt
    out = []
    for j in range(len(cuts) - 1):
        sel = np.flatnonzero((t > cuts[j]) & (t <= cuts[j + 1]))
        i = sel[int(np.argmax(dens[sel]))]
        y0, y1, y2 = dens[i - 1], dens[i], dens[i + 1]
        den = y0 - 2 * y1 + y2
        out.append(float(t[i] + (0.5 * (y0 - y2) / den if den < 0 else 0.0) * law.spec.dt))
    return out


def smoothing_offset(cell: Cell) -> dict:
    """Peak of f_{h_j} minus raw argmax, on the noiseless mean-field curve of the final design."""
    mf = de.mean_field_law(cell.design(), cell.spec, want_grad=False)
    cuts = cell.cuts
    r = functionals(mf.p, None, mf.t, cell.hs, cuts)
    raw = raw_peaks(mf, cuts_from(r, cell.spec.tmax))
    off = [a - b for a, b in zip(r["peaks"], raw)]
    return {"mf_peaks_h": r["peaks"], "mf_peaks_raw": raw, "offset": off, "max_abs": float(np.max(np.abs(off)))}


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


def warm_start_cell(cell: Cell) -> dict:
    ws = mf_warm_start(cell.T, cell.r, cell.B, cell.spec, cell.hs, t_cap=cell.t_cap)
    cell.set_start(ws["tp"], ws["w"], ws["cuts"])
    p1 = prop1_weights(cell.T, cell.r, cell.B)
    return {"prop1_w": p1["w"].tolist(), "prop1_theta": p1["theta"], "mf_tp": ws["tp"].tolist(),
            "mf_w": ws["w"].tolist(), "mf_newton_hist": ws["hist"], "homotopy_steps": len(ws["homotopy"]),
            "mf_peaks": ws["res"]["peaks"], "mf_ratios": ws["res"]["ratios"]}


def residual_table(cell: Cell, stage: str) -> list:
    rows = [h for h in cell.hist if h.get("stage") == stage and "res_inf" in h]
    out = []
    for k, h in enumerate(rows):
        q = None
        if k >= 1 and rows[k - 1]["res_inf"] > 0 and h["res_inf"] > 0:
            q = h["res_inf"] / rows[k - 1]["res_inf"] ** 2
        out.append({"iter": k, "res_inf": h["res_inf"], "res_2": h["res_2"], "step": h["a"],
                    "ratio_to_prev_squared": q})
    return out


def out_of_sample(cells: list, spec, n_paths: int, log) -> dict:
    """Evaluate final designs on an independent FK ensemble (tag 204, replicate 1): z of
    (achieved - target) with SE^2 = SE_oos^2 + SE_in-sample^2 (the latter = design sampling error)."""
    out = {}
    with StreamEngine(spec, n_paths, tag=TAG_DEMO, replicate=OOS_REPLICATE, chunk=CHUNK, group=GROUP) as se:
        laws = se.evaluate([c.design() for c in cells], want_grad=False)
        meta = se.meta
    for c, law in zip(cells, laws):
        try:
            res = analyse(law, c.hs, c.cuts, jackknife=True)
        except DesignMapError as exc:
            out[c.key] = {"error": str(exc)}
            continue
        y, _ = f1_outputs(res, c.m)
        se_oos = np.concatenate([res["se"]["peaks"], res["se"]["ratios"][:c.m - 1]])
        fin = getattr(c, "final", None) or {}
        se_in = np.concatenate([fin["se"]["peaks"], fin["se"]["ratios"][:c.m - 1]]) if fin else 0.0 * se_oos
        z = (y - c.target) / np.sqrt(se_oos ** 2 + se_in ** 2)
        k = int(round(0.01 / spec.dt))
        nb = law.p.size // k
        dens = law.p[:nb * k].reshape(nb, k).sum(axis=1) / (k * spec.dt)
        out[c.key] = {"peaks": res["peaks"], "ratios": res["ratios"], "yield": res["yield"],
                      "masses": res["masses"], "valleys": res["valleys"],
                      "density_bins": {"width": k * spec.dt, "t_left": 0.0, "density": dens.tolist()},
                      "se_oos": se_oos.tolist(), "se_in_sample": np.asarray(se_in).tolist(),
                      "deviation": (y - c.target).tolist(), "z": z.tolist(), "max_abs_z": float(np.max(np.abs(z))),
                      "max_abs_peak_dev": float(np.max(np.abs(y[:c.m] - c.T))),
                      "mode_check": mode_check(law, c.hs, res)}
    log(f"[oos] {n_paths:.0e} paths: max|z| per cell " +
        ", ".join(f"{k}:{v.get('max_abs_z', float('nan')):.2f}" for k, v in out.items()))
    return {"engine": meta, "cells": out}


def _finalize_all(cells, se, log) -> dict:
    fin = {}
    for c in cells:
        if c.status != "converged":
            fin[c.key] = c.to_dict()
            continue
        f = finalize_cell(c, se)
        f["smoothing_offset_mf"] = smoothing_offset(c)
        c.final = f
        fin[c.key] = f
        log(f"[final] {c.key}: res {f['residual_inf']:.1e} peaks {np.round(f['achieved']['peaks'], 4).tolist()} "
            f"se {np.round(f['se']['peaks'], 4).tolist()} modes {f['mode_check']['per_basin_counts']} "
            f"t'm {c.tp[-1]:.3f} yield {f['achieved']['yield']:.3f}")
    return fin


def cmd_showcase(args) -> dict:
    """m=5, eps=0.05, B=8, equal: SAA Newton from the mean-field warm start on 1e6 paths (residual table),
    then the sample-size study on disjoint sub-ensembles of the same ensemble."""
    OUT.mkdir(parents=True, exist_ok=True)
    log = Logger(OUT / "logs" / "showcase.log")
    eps, B, m, kind = 0.05, 8.0, 5, "equal"
    spec = de.path_spec(eps)
    n_full = int(float(args.n))
    cell = Cell(m, eps, B, kind, spec, tag="showcase")
    ws = warm_start_cell(cell)
    t0 = time.time()
    with StreamEngine(spec, n_full, tag=TAG_SHOWCASE, chunk=CHUNK, group=GROUP) as se:
        saa_newton(se, [cell], stage="full", log=log)
        fin = finalize_cell(cell, se)
        fin["smoothing_offset_mf"] = smoothing_offset(cell)
        table = residual_table(cell, "full")
        log(f"[showcase] residual table {[('%.1e' % r['res_inf']) for r in table]}")
        # sample-size study: disjoint sub-ensembles, warm start = full-ensemble solution
        n_chunks = len(se.chunk_ids)
        levels = [int(float(x)) for x in args.levels.split(",")]
        subs = []
        for N in levels:
            k = N // CHUNK
            for i in range(n_chunks // k):
                ids = list(range(i * k, (i + 1) * k))
                if k == n_chunks:
                    continue           # the full ensemble is the showcase solve itself
                c = Cell(m, eps, B, kind, spec, tag=f"N{N}_s{i}", chunk_set=ids)
                c.set_start(cell.tp, cell.w, cell.cuts)
                c.level = N
                subs.append(c)
        saa_newton(se, subs, stage="subset", log=log)
        sub_out = []
        for c in subs:
            row = {"key": c.key, "N": c.level, "chunks": c.chunk_set, "status": c.status,
                   "iters": len([h for h in c.hist if "res_inf" in h]) - 1}
            if c.status == "converged":
                f = finalize_cell(c, se)
                row.update({"tp": c.tp.tolist(), "w": c.w.tolist(), "design_se_delta": f["design_se_delta"],
                            "residual_inf": f["residual_inf"], "achieved_peaks": f["achieved"]["peaks"]})
            sub_out.append(row)
    # summary: spread of designs across disjoint subsets vs N
    x_full = np.concatenate([cell.tp, cell.w[:m - 1]])
    summ = []
    for N in levels:
        rows = [r for r in sub_out if r["N"] == N and r["status"] == "converged"]
        if N == n_full:
            summ.append({"N": N, "n_subsets": 1, "delta_se": fin["design_se_delta"]})
            continue
        X = np.array([np.concatenate([r["tp"], r["w"][:m - 1]]) for r in rows])
        sd = X.std(axis=0, ddof=1) if len(rows) > 1 else np.full(X.shape[1], np.nan)
        rms_to_full = np.sqrt(np.mean((X - x_full) ** 2, axis=0))
        summ.append({"N": N, "n_subsets": len(rows), "empirical_sd": sd.tolist(),
                     "rms_to_full": rms_to_full.tolist(),
                     "delta_se_mean": np.mean([r["design_se_delta"] for r in rows], axis=0).tolist()})
    # slope of log(delta-SE of t'_m) and log(empirical sd of t'_m) vs log N
    lab = [f"t'{j + 1}" for j in range(m)] + [f"w{k + 1}" for k in range(m - 1)]
    Ns = np.array([s["N"] for s in summ], float)
    dse = np.array([s["delta_se"][m - 1] if "delta_se" in s else s["delta_se_mean"][m - 1] for s in summ])
    slope_delta = float(np.polyfit(np.log(Ns), np.log(dse), 1)[0])
    emp = [(s["N"], s["empirical_sd"][m - 1]) for s in summ if "empirical_sd" in s and np.isfinite(s["empirical_sd"][m - 1])]
    slope_emp = float(np.polyfit(np.log([e[0] for e in emp]), np.log([e[1] for e in emp]), 1)[0]) if len(emp) >= 2 else None
    payload = {"item": "uplift2 item 1 NU-B (SAA Newton showcase + sample-size study)",
               "driver": "code/fb_v2_saa_newton.py showcase", "cell": cell.key, "warm_start": ws,
               "residual_table": table, "final": fin, "subsets": sub_out, "sample_size_summary": summ,
               "coords": lab, "slope_log_se_tm_vs_logN": {"delta_method": slope_delta, "empirical": slope_emp},
               "seconds": time.time() - t0}
    de.write_json(OUT / "showcase_m5_eps0.05_B8_equal.json", payload)
    log(f"[showcase] slopes delta {slope_delta:.3f} empirical {slope_emp}")
    return payload



def _samplesize_summary(rows, levels, fin, m):
    """Per-N spread of independent SAA designs vs the mean delta-method SE; log-log slopes of the spread."""
    x_full = np.concatenate([fin["stripe_times"], fin["w"][:m - 1]])
    summ = []
    for N in levels:
        R = [r for r in rows if r["N"] == N and r["status"] == "converged"]
        if len(R) < 2:
            continue
        X = np.array([np.concatenate([r["tp"], r["w"][:m - 1]]) for r in R])
        sd = X.std(axis=0, ddof=1)
        dse = np.mean([r["design_se_delta"] for r in R], axis=0)
        summ.append({"N": N, "K": len(R), "empirical_sd": sd.tolist(), "delta_se_mean": dse.tolist(),
                     "ratio_emp_to_delta": (sd / dse).tolist(), "mean": X.mean(axis=0).tolist(),
                     "mean_minus_full": (X.mean(axis=0) - x_full).tolist()})
    Ns = np.array([q["N"] for q in summ], float)
    slopes = {}
    for j, lab in enumerate([f"t'{i + 1}" for i in range(m)] + [f"w{i + 1}" for i in range(m - 1)]):
        sd = np.array([q["empirical_sd"][j] for q in summ])
        if sd.size >= 2 and np.all(sd > 0):
            slopes[lab] = float(np.polyfit(np.log(Ns), np.log(sd), 1)[0])
    pooled = [float(np.sqrt(np.mean(np.square(q["ratio_emp_to_delta"])))) for q in summ]
    return summ, slopes, pooled


def cmd_samplesize_merge(args) -> dict:
    """Merge independent-replicate runs (samplesize_*.json) with the showcase run (replicate 0, 1e6) and
    recompute the per-N summary, the log-log slopes and a pooled slope over all coordinates."""
    sc = json.loads((OUT / "showcase_m5_eps0.05_B8_equal.json").read_text())
    fin = sc["final"]
    m = fin["m"]
    rows, srcs = [], []
    for name in args.files.split(","):
        d = json.loads((OUT / name).read_text())
        rows += d["rows"]
        srcs.append(name)
    rows.append({"N": int(fin["n_paths"]), "replicate": 0, "status": fin["status"], "tp": fin["stripe_times"],
                 "w": fin["w"], "design_se_delta": fin["design_se_delta"], "residual_inf": fin["residual_inf"],
                 "source": "showcase_m5_eps0.05_B8_equal.json#final"})
    reps = [(r["N"], r["replicate"]) for r in rows]
    assert len({r for _, r in reps}) == len(reps), "replicate ids must be distinct (independent ensembles)"
    levels = sorted({r["N"] for r in rows})
    summ, slopes, pooled = _samplesize_summary(rows, levels, fin, m)
    Ns = np.array([q["N"] for q in summ], float)
    # pooled slope: regress log(sd_j / sd_j at the smallest N) on log N over all coordinates j
    Y = np.log(np.array([q["empirical_sd"] for q in summ]))
    Y = Y - Y[0][None, :]
    x = np.log(Ns / Ns[0])
    pooled_slope = float(np.sum(x[:, None] * Y) / (Y.shape[1] * np.sum(x * x)))
    D = np.log(np.array([q["delta_se_mean"] for q in summ]))
    dslope = float(np.sum(x[:, None] * (D - D[0][None, :])) / (D.shape[1] * np.sum(x * x)))
    # parametric null: sd_j(N) = s_j (N0/N)^{1/2} exactly (s_j = mean delta-method SE at the smallest N), K_N normal
    # replicates per level, coordinates independent (their correlation would only widen the null range)
    Ks = [q["K"] for q in summ]
    s0 = np.array(summ[0]["delta_se_mean"])
    rng = np.random.default_rng(20260925)
    null = np.empty(20000)
    for b in range(null.size):
        Yb = np.log(np.array([(rng.normal(size=(k, s0.size)) * s0 * np.sqrt(Ns[0] / N)).std(axis=0, ddof=1)
                              for N, k in zip(Ns, Ks)]))
        Yb = Yb - Yb[0][None, :]
        null[b] = np.sum(x[:, None] * Yb) / (Yb.shape[1] * np.sum(x * x))
    null_test = {"null": "sd proportional to N^-1/2 with the delta-method scale; K_N normal replicates; independent coords",
                 "n_sim": int(null.size), "seed": 20260925,
                 "null_pooled_slope_q025_q50_q975": np.percentile(null, [2.5, 50, 97.5]).tolist(),
                 "p_two_sided": float(2 * min((null <= pooled_slope).mean(), (null >= pooled_slope).mean()))}
    payload = {"item": "uplift2 item 1 NU-B (sample-size study, merged: N = 1.25e5 ... 1e6)",
               "driver": "code/fb_v2_saa_newton.py samplesize-merge", "cell": fin["key"], "sources": srcs,
               "levels": levels, "summary": summ, "slope_log_sd_vs_logN": slopes,
               "pooled_slope_all_coords": pooled_slope, "pooled_slope_delta_se": dslope, "null_test": null_test,
               "pooled_rms_ratio_emp_to_delta": pooled, "rows": rows}
    de.write_json(OUT / "samplesize_merged_m5_eps0.05_B8_equal.json", payload)
    print(f"[samplesize-merge] levels {levels} K {[q['K'] for q in summ]} pooled slope {pooled_slope:.3f} "
          f"(delta-SE {dslope:.3f}) ratio emp/delta {np.round(pooled, 2).tolist()} null 95% "
          f"{np.round(null_test['null_pooled_slope_q025_q50_q975'], 3).tolist()} p {null_test['p_two_sided']:.3f}")
    return payload


def cmd_samplesize(args) -> dict:
    """Independent-replicate sample-size study for the showcase cell: K independent ensembles
    (tag 203, replicates 1, 2, ...) at each N, SAA Newton from the 1e6 showcase solution, spread of the
    SAA designs vs N and vs the delta-method SE (the nested disjoint-subset study of `showcase` shares
    its base sub-ensembles across levels and cannot test the N-scaling by itself)."""
    log = Logger(OUT / "logs" / "samplesize.log")
    sc = json.loads((OUT / "showcase_m5_eps0.05_B8_equal.json").read_text())
    fin = sc["final"]
    eps, B, m, kind = fin["eps"], fin["B"], fin["m"], fin["profile"]
    spec = de.path_spec(eps)
    levels = [int(float(x)) for x in args.levels.split(",")]
    reps = [int(x) for x in args.reps.split(",")]
    rows = []
    rep_id = int(getattr(args, "rep_start", 1) or 1)
    t0 = time.time()
    for N, K in zip(levels, reps):
        for k in range(K):
            c = Cell(m, eps, B, kind, spec, tag=f"indep_N{N}_r{rep_id}")
            c.set_start(fin["stripe_times"], fin["w"], fin["achieved"]["cuts"])
            with StreamEngine(spec, N, tag=TAG_SHOWCASE, replicate=rep_id, chunk=CHUNK, group=GROUP) as se:
                saa_newton(se, [c], stage="indep", log=log)
                row = {"N": N, "replicate": rep_id, "status": c.status,
                       "iters": len([h for h in c.hist if "res_inf" in h]) - 1, "entropy": se.meta["entropy"]}
                if c.status == "converged":
                    f = finalize_cell(c, se)
                    row.update({"tp": c.tp.tolist(), "w": c.w.tolist(), "design_se_delta": f["design_se_delta"],
                                "residual_inf": f["residual_inf"]})
            rows.append(row)
            rep_id += 1
    summ, slopes, pooled = _samplesize_summary(rows, levels, fin, m)
    payload = {"item": "uplift2 item 1 NU-B (independent-replicate sample-size study)",
               "driver": "code/fb_v2_saa_newton.py samplesize", "cell": fin["key"], "rows": rows,
               "summary": summ, "slope_log_sd_vs_logN": slopes, "pooled_rms_ratio_emp_to_delta": pooled,
               "reference": "showcase_m5_eps0.05_B8_equal.json#final (1e6 paths, replicate 0)",
               "seconds": time.time() - t0}
    de.write_json(OUT / (getattr(args, "out", "") or "samplesize_m5_eps0.05_B8_equal.json"), payload)
    log(f"[samplesize] slopes {slopes} pooled ratio {pooled}")
    return payload


def demo_cells(eps: float, spec) -> list:
    cells = []
    for m in DEMO_M:
        for B in DEMO_B:
            for kind in PROFILES:
                cells.append(Cell(m, eps, B, kind, spec))
    return cells


def cmd_demo(args) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = float(args.eps)
    log = Logger(OUT / "logs" / f"demo_eps{eps:g}.log")
    spec = de.path_spec(eps)
    n_full, n_1 = int(float(args.n)), int(float(args.n1))
    cells = demo_cells(eps, spec)
    if args.only:
        cells = [c for c in cells if c.key in args.only.split(",")]
    warm = {}
    for c in cells:
        try:
            warm[c.key] = warm_start_cell(c)
        except DesignMapError as exc:
            c.status = "failed_warm_start"
            c.tp, c.w = c.T.copy(), prop1_weights(c.T, c.r, c.B)["w"]
            warm[c.key] = {"error": str(exc)}
            log(f"[warm] {c.key} failed: {exc}")
    t0 = time.time()
    with StreamEngine(spec, n_full, tag=TAG_DEMO, chunk=CHUNK, group=GROUP) as se:
        stage1 = list(range(n_1 // CHUNK))
        for c in cells:
            c.chunk_set = stage1
        saa_newton(se, cells, stage=f"N{n_1}", log=log)
        for c in cells:
            c.chunk_set = None
            if c.status == "converged":
                c.status = "active"
                c.res = None
        saa_newton(se, cells, stage=f"N{n_full}", log=log)
        tables = {c.key: residual_table(c, f"N{n_full}") for c in cells}
        fin = _finalize_all(cells, se, log)
        meta = se.meta
    oos = out_of_sample([c for c in cells if c.status == "converged"], spec, int(float(args.n_oos)), log)
    payload = {"item": "uplift2 item 1 NU-C (demonstrations)", "driver": "code/fb_v2_saa_newton.py demo",
               "eps": eps, "n_paths": n_full, "stage1_paths": n_1, "engine": meta, "warm_start": warm,
               "residual_tables": tables, "cells": fin, "out_of_sample": oos, "seconds": time.time() - t0,
               "targets": {str(m): TARGET_TIMES[m] for m in DEMO_M}, "kappa": KAPPA, "h_lo": H_LO, "h_hi": H_HI}
    outdir = Path(args.out) if args.out else OUT
    de.write_json(outdir / f"demo_eps{eps:g}{'_' + args.label if args.label else ''}.json", payload)
    return payload


def cmd_failed(args) -> dict:
    """The two N11 cells that failed under the 3.5 cap, rerun with no cap on the stripe times."""
    OUT.mkdir(parents=True, exist_ok=True)
    log = Logger(OUT / "logs" / "failed_cells.log")
    eps = 0.1
    res_all = {}
    for tmax in [float(x) for x in args.tmax.split(",")]:
        spec = de.path_spec(eps, tmax=tmax)
        cells = [Cell(3, eps, B, "equal", spec, tag=f"nocap_tmax{tmax:g}") for B in (8.0, 4.0)]
        warm = {c.key: warm_start_cell(c) for c in cells}
        with StreamEngine(spec, int(float(args.n)), tag=TAG_DEMO, chunk=CHUNK, group=GROUP) as se:
            n_1 = int(float(args.n1))
            for c in cells:
                c.chunk_set = list(range(n_1 // CHUNK))
            saa_newton(se, cells, stage="stage1", log=log)
            for c in cells:
                c.chunk_set = None
                if c.status == "converged":
                    c.status, c.res = "active", None
            saa_newton(se, cells, stage="full", log=log)
            tables = {c.key: residual_table(c, "full") for c in cells}
            fin = _finalize_all(cells, se, log)
            meta = se.meta
        oos = out_of_sample([c for c in cells if c.status == "converged"], spec, int(float(args.n_oos)), log)
        res_all[f"tmax{tmax:g}"] = {"engine": meta, "warm_start": warm, "residual_tables": tables,
                                    "cells": fin, "out_of_sample": oos}
    payload = {"item": "uplift2 item 1 NU-C (rerun of the capped N11 cells)",
               "driver": "code/fb_v2_saa_newton.py failed", "n11_reference":
               "N11_shift_compensation/n11_shift_compensation.json#summary.{m3_eps0.1_B8,m3_eps0.1_B4}.nwt2 (cap 3.5)",
               "runs": res_all}
    de.write_json(OUT / "failed_cells_nocap.json", payload)
    return payload



def cmd_export(args) -> dict:
    """Design JSONs for out-of-sample direct kill (NU-E / fb_v2_tep): one file per converged cell."""
    ddir = OUT / "designs"
    ddir.mkdir(parents=True, exist_ok=True)
    written = []
    sources = sorted(OUT.glob("demo_eps*.json")) + [OUT / "failed_cells_nocap.json", OUT / "showcase_m5_eps0.05_B8_equal.json"]
    for src in sources:
        if not src.exists():
            continue
        d = json.loads(src.read_text())
        if "runs" in d:
            cells = {k: v for run in d["runs"].values() for k, v in run["cells"].items()}
        elif "final" in d:
            cells = {d["final"]["key"]: d["final"]}
        else:
            cells = d["cells"]
        for key, c in cells.items():
            if c.get("status") != "converged":
                continue
            payload = {"label": key, "eps": c["eps"], "B": c["B"], "c": c["c"], "w": c["w"],
                       "stripe_times": c["stripe_times"], "tmax": c["tmax"], "dt": c["dt"],
                       "h": c["bandwidths_h"], "mass_cuts": "valley", "cuts0": c["achieved"]["cuts"],
                       "targets": {"peaks": c["targets_T"], "ratios": c["targets_r"]},
                       "formulation": "F1 (peak times + m-1 conditional ratios at fixed B)",
                       "source": f"artifacts/data/exact_m_fixed_budget/V2_design/{src.name}#cells.{key}",
                       "note": "h is a per-passage bandwidth vector: analyse with fb_v2_saa_newton.analyse(law, h, cuts0)"}
            de.write_json(ddir / f"{key}.json", payload)
            written.append(key)
    print(f"[export] {len(written)} designs -> {ddir}")
    return {"written": written}


def cmd_figure(args) -> list:
    """fb_v2_design_programming.pdf: (a) target vs achieved densities, m = 5 (out-of-sample FK law);
    (b) target vs achieved conditional ratios; (c) timing horizon t_max(eps) for m = 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exact_m_prr_upgrade_core as core
    core.apply_prr_style()
    eps, B = float(args.eps), float(args.B)
    demo = json.loads((Path(args.demo) if args.demo else OUT / f"demo_eps{eps:g}.json").read_text())
    kinds = [k for k in PROFILES if cell_key(5, eps, B, k) in demo["out_of_sample"]["cells"]]
    col = {"rising": core.OI_BLUE, "falling": core.OI_VERMILLION, "equal": core.OI_GREEN}
    mk = {"rising": "o", "falling": "s", "equal": "D"}
    fig = plt.figure(figsize=(7.2, 2.55), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.15])
    axa, axb, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    T = TARGET_TIMES[5]
    for tj in T:
        axa.axvline(tj, color="0.55", lw=0.6, ls=":", zorder=0)
    for kind in kinds:
        key = cell_key(5, eps, B, kind)
        o = demo["out_of_sample"]["cells"][key]
        db = o["density_bins"]
        dens = np.asarray(db["density"])
        tc = db["t_left"] + db["width"] * (np.arange(dens.size) + 0.5)
        axa.plot(tc, dens, color=col[kind], lw=1.0, label=kind)
    axa.set_xlim(0.5, 3.6)
    axa.set_yscale("log")
    top = max(max(demo["out_of_sample"]["cells"][cell_key(5, eps, B, k)]["density_bins"]["density"]) for k in kinds)
    axa.set_ylim(top * 3e-3, top * 2.5)
    axa.set_xlabel(r"time $t$ ($1/\gamma$)")
    axa.set_ylabel("density $f(t)$")
    axa.text(0.985, 0.975, f"$m=5$, $B={B:g}$\n$\\varepsilon={eps:g}$", transform=axa.transAxes, va="top",
             ha="right", fontsize=7.5)
    axa.text(0.01, 0.99, "(a)", transform=axa.transAxes, va="top", ha="left", fontweight="bold")
    # (b) ratios: target = dark bar drawn across the achieved marker (they agree to ~1e-4)
    from matplotlib.lines import Line2D
    xs = np.arange(1, 6)
    off = {"rising": -0.24, "falling": 0.0, "equal": 0.24}
    for kind in kinds:
        key = cell_key(5, eps, B, kind)
        o = demo["out_of_sample"]["cells"][key]
        c = demo["cells"][key]
        r_t = np.asarray(c["targets_r"])
        r_a = np.asarray(o["ratios"])
        axb.plot(xs + off[kind], r_a, ls="none", marker=mk[kind], ms=3.4, color=col[kind], zorder=2)
        axb.hlines(r_t, xs + off[kind] - 0.13, xs + off[kind] + 0.13, color="0.1", lw=0.8, zorder=3)
    axb.set_xticks(xs)
    axb.set_xlabel("basin $j$")
    axb.set_ylabel(r"conditional ratio $r_j = M_j/\Sigma_i M_i$")
    axb.set_ylim(0, 0.47)
    handles = [Line2D([], [], ls="none", marker=mk[k], ms=3.4, color=col[k], label=k) for k in kinds]
    handles.append(Line2D([], [], color="0.1", lw=0.8, label="target"))
    axb.legend(handles=handles, loc="upper right", ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8,
               handletextpad=0.4, fontsize=7.0, borderaxespad=0.2)
    axb.text(0.02, 0.99, "(b)", transform=axb.transAxes, va="top", ha="left", fontweight="bold")
    # (c) horizon
    fit_p = OUT / "horizon_fit.json"
    if fit_p.exists():
        fit = json.loads(fit_p.read_text())["fits"]
        bcol = {"B4": core.OI_BLUE, "B8": core.OI_VERMILLION}
        bmk = {"B4": "o", "B8": "s"}
        from matplotlib.lines import Line2D
        for kB, f in sorted(fit.items()):
            rows = sorted([r for r in f["rows"] if r["t_max"] is not None], key=lambda r: r["eps"])
            e = np.array([r["eps"] for r in rows])
            t = np.array([r["t_max"] for r in rows])
            se = np.array([r["se"] for r in rows])
            ff = np.array([r["stripe_at_centre_formula"]["T"] for r in rows])
            axc.plot(e, ff, color=bcol[kB], lw=0.8, ls="--", zorder=1)
            axc.errorbar(e, t, yerr=2 * se, ls="none", marker=bmk[kB], ms=3.2, color=bcol[kB], zorder=3)
            cap = [(r["eps"], r["capped_t_max_h3p5"], r.get("capped_t_max_h3p5_se") or 0.0) for r in rows
                   if r.get("capped_t_max_h3p5") is not None]
            if cap:
                ce, ct, cs = (np.array(v) for v in zip(*cap))
                axc.errorbar(ce, ct, yerr=2 * cs, ls="none", marker=bmk[kB], ms=3.2, mfc="white",
                             color=bcol[kB], zorder=3)
        axc.axhline(2.8, color="0.35", lw=0.6, ls=":", zorder=0)
        axc.set_xscale("log")
        axc.set_xlabel(r"noise $\varepsilon$")
        axc.set_ylabel(r"latest late peak $t_{\max}$")
        hd = [Line2D([], [], ls="none", marker=bmk[k], ms=3.2, color=bcol[k], label=f"$B={k[1:]}$")
              for k in sorted(fit)]
        hd += [Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", color="0.3", label="window end 3.5"),
               Line2D([], [], color="0.3", lw=0.8, ls="--", label="closed form")]
        axc.legend(handles=hd, loc="upper right", frameon=False, handlelength=1.2, fontsize=7.0,
                   borderaxespad=0.2, handletextpad=0.4)
        axc.text(0.03, 2.8, "target 2.8", transform=axc.get_yaxis_transform(), fontsize=7.0, color="0.35",
                 va="bottom", ha="left")
    axc.text(0.02, 0.02, "(c)", transform=axc.transAxes, va="bottom", ha="left", fontweight="bold")
    stem = Path(args.stem) if args.stem else de.FIGURES / "fb_v2_design_programming"
    written = []
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
        written.append(str(stem.with_suffix(suffix)))
    print("[figure]", written)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("showcase")
    s.add_argument("--n", default="1e6")
    s.add_argument("--levels", default="1.25e5,2.5e5,5e5,1e6")
    s = sub.add_parser("demo")
    s.add_argument("--eps", required=True)
    s.add_argument("--n", default="5e5")
    s.add_argument("--n1", default="1.25e5")
    s.add_argument("--n-oos", dest="n_oos", default="2.5e5")
    s.add_argument("--only", default="")
    s.add_argument("--label", default="")
    s.add_argument("--out", default="")
    s = sub.add_parser("failed")
    s.add_argument("--n", default="1e6")
    s.add_argument("--n1", default="1.25e5")
    s.add_argument("--n-oos", dest="n_oos", default="2.5e5")
    s.add_argument("--tmax", default="4,6")
    s = sub.add_parser("samplesize")
    s.add_argument("--levels", default="1.25e5,2.5e5,5e5")
    s.add_argument("--reps", default="8,4,4")
    s.add_argument("--rep-start", dest="rep_start", type=int, default=1)
    s.add_argument("--out", default="")
    s = sub.add_parser("samplesize-merge")
    s.add_argument("--files", default="samplesize_m5_eps0.05_B8_equal.json,samplesize_1e6_m5_eps0.05_B8_equal.json")
    sub.add_parser("export")
    s = sub.add_parser("figure")
    s.add_argument("--eps", default="0.05")
    s.add_argument("--B", default="8")
    s.add_argument("--demo", default="")
    s.add_argument("--stem", default="")
    args = ap.parse_args(argv)
    {"showcase": cmd_showcase, "demo": cmd_demo, "failed": cmd_failed, "export": cmd_export,
     "figure": cmd_figure, "samplesize": cmd_samplesize,
     "samplesize-merge": cmd_samplesize_merge}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
