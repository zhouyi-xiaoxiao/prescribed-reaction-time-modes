#!/usr/bin/env python3
"""N13 -- third-order mean-field residual (answers TH-1 acceptance criterion 4 / item N2).

Theory: theory/TH1b_third_order.tex.  With xi_t = X_t - Lambda(t) (unit-budget exposure
minus its mean), the exact identity

  f - f_1 = -B^2 e^{-B Lam} kappa_11 + (B^3/2) e^{-B Lam} (kappa_12 + G kappa_02) + rho_3,
  |rho_3| <= B^4 h3(B Lam) E[V |xi|^3],   h3(u) = int_0^1 (r^2/2) e^{-r u} dr,

holds at every t (kappa_11 = Cov(V,X), kappa_12 = kappa(V,X,X) = E[(V-G) xi^2],
kappa_02 = Var X).  Undamped form: -B^2 k11 + B^3 [Lam k11 + (k12 + G k02)/2] + O(B^4).
Cumulant resummations: f^(2) = B (G - B k11) exp(-B Lam + B^2 k02/2)   (N2's closure),
                       f^(3) = B (G - B k11 + B^2 k12/2) exp(-B Lam + B^2 k02/2 - B^3 k03/6).

Data: the N2 Feynman--Kac ensembles (n0_m2_eps0.05, n0_m2_eps0.1, n0_m3_eps0.1: tag 80;
n2_m3_eps0.15: tag 81; 2e5 paths each) store exposure increments only on the 0.02
checkpoint grid, so V_t is not available.  This driver RE-GENERATES THE SAME PATHS bit for
bit (same SeedSequence children, same Philox draw order and arithmetic as
exact_m_prr_fk_exact_law._run_chunk, whose _slab_components it calls) and accumulates at
every time step, for the model kernel ('full': contact gate, field at the midpoint) and
equal weights: sums of (X-c)^k (k=1..3, all paths), V (X-c)^k (k=0..4), V e^{-B X}
(continuous-time density on the path) and the discrete kill probability
(1 - e^{-B e_n}) e^{-B X_{n-1}}, per path group (20 groups, as N2).  c_n is the
deterministic EM-exact mean exposure (a centring constant only; all identities use the
sample mean).  Common paths are verified against the stored increments (chunk 0, all paths,
every checkpoint).  No new random numbers.

Outputs: artifacts/data/exact_m_fixed_budget/N13_third_order/n13_third_order.json
         (+ per-cell step-level accumulators under ~/.local-build/prr_fk_ensembles/n13_*/)
Subcommands: simulate [--only CELL] | analyze | figure(optional)
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

import exact_m_prr_fk_exact_law as fk  # noqa: E402

OUT_DIR = fk.FB_DATA / "N13_third_order"
OUT_JSON = OUT_DIR / "n13_third_order.json"
ACC_ROOT = fk.ENSEMBLE_ROOT
CELLS = {
    "m2_eps0.05": "n0_m2_eps0.05",
    "m2_eps0.1": "n0_m2_eps0.1",
    "m3_eps0.1": "n0_m3_eps0.1",
    "m3_eps0.15": "n2_m3_eps0.15",
}
B_LIST = (0.5, 1.0, 2.0, 4.0)
B_MAIN = (0.5, 1.0, 2.0)
GROUPS = 20
KX = 3          # all-path moments (X-c)^k, k = 1..KX
KV = 4          # active-path moments V (X-c)^k, k = 0..KV


def em_centre(spec: fk.EnsembleSpec, w) -> np.ndarray:
    """c_n = EM-exact E X_n (n = 1..steps), unit budget; X_n includes step n."""
    fe = fk.free_exposure_discrete(spec, w, gate="contact")
    return np.cumsum(fe["G"]) * spec.dt


def _chunk(task: dict) -> dict:
    """Regenerate one chunk of an N2 ensemble (identical to fk._run_chunk's draws and
    arithmetic for the 'full' variant) and accumulate per-step moments per path group."""
    spec = fk.EnsembleSpec.from_dict(task["spec"])
    n = int(task["size"])
    steps = spec.steps()
    dt = spec.dt
    centres = spec.centres()
    m = centres.size
    rng = np.random.Generator(np.random.Philox(task["seedseq"]))
    t0 = time.perf_counter()
    decay = 1.0 - spec.gamma * dt
    pull = spec.gamma * spec.z_bar * dt
    noise_z = spec.eps * math.sqrt(spec.d0 * dt)
    noise_r = 2.0 * spec.eps * math.sqrt(spec.d0 * dt)
    W = spec.torus_w
    z = spec.z0 + math.sqrt(spec.var_z0_scale * spec.eps**2 * spec.d0 / (2.0 * spec.gamma)) \
        * rng.standard_normal(n)
    r_par = spec.r_par0 + spec.eps * spec.u0 * rng.standard_normal(n)
    r_perp = np.mod(spec.r_perp0 + spec.eps * spec.sigma_perp0
                    * rng.standard_normal((spec.n_perp, n)), W)
    a = spec.contact_a
    w = np.asarray(task["w"], float)
    cen = np.asarray(task["centre"], float)          # c_n, n = 1..steps
    gid = np.asarray(task["gid"], np.int64)          # local group index per path
    G = int(task["n_groups_local"])
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    Bs = np.asarray(task["budgets"], float)
    nb = Bs.size
    AX = np.zeros((KX, G, steps))
    AV = np.zeros((KV + 1, G, steps))
    FB = np.zeros((nb, G, steps))
    PD = np.zeros((nb, G, steps))
    X = np.zeros(n)
    verify = task.get("verify")
    if verify:
        with np.load(verify["file"]) as d:
            dXs = d["dX_full"].astype(np.float64)
        cps = np.asarray(verify["cps"], np.int64)
        Xs = np.zeros((n, m))
        ref = np.zeros((n, m))
        kidx = 0
        while kidx < cps.size and cps[kidx] == 0:
            kidx += 1
        max_rel = 0.0
        max_abs = 0.0
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
        act, comps = fk._slab_components(z, centres, spec)
        if act.size:
            g = (dist2[act] < a * a).astype(float)
            c = comps * g[:, None]
            e = fk._wsum(c, w)
            xprev = X[act]
            X[act] = xprev + e
            if verify:
                Xs[act] += c
            V = e / dt
            xa = X[act]
            ga = gid[act]
            dv = xa - cen[s]
            pw = V.copy()
            for k in range(KV + 1):
                AV[k, :, s] = np.bincount(ga, weights=pw, minlength=G)
                pw = pw * dv
            for ib, Bv in enumerate(Bs):
                FB[ib, :, s] = np.bincount(ga, weights=V * np.exp(-Bv * xa), minlength=G)
                PD[ib, :, s] = np.bincount(ga, weights=-np.expm1(-Bv * e) * np.exp(-Bv * xprev),
                                           minlength=G)
        d1 = X - cen[s]
        pw = d1.copy()
        for k in range(KX):
            AX[k, :, s] = np.add.reduceat(pw, starts)
            pw = pw * d1
        if verify and kidx < cps.size and s + 1 == cps[kidx]:
            ref += dXs[:, :, kidx]
            diff = np.abs(Xs - ref)
            max_abs = max(max_abs, float(diff.max()))
            mask = ref > 1e-12
            if mask.any():
                max_rel = max(max_rel, float((diff[mask] / ref[mask]).max()))
            kidx += 1
    out = {"chunk": int(task["chunk"]), "AX": AX, "AV": AV, "FB": FB, "PD": PD,
           "runtime": time.perf_counter() - t0}
    if verify:
        out["verify"] = {"checkpoints_compared": int(kidx), "max_abs_diff": max_abs,
                         "max_rel_diff_where_ref_gt_1e-12": max_rel,
                         "paths": n, "note": "cumulative per-slab exposure vs stored float32 increments"}
    return out


def cell_tasks(label: str) -> tuple[dict, list]:
    idx = json.loads(fk.index_path(CELLS[label]).read_text())
    spec = fk.EnsembleSpec.from_dict(idx["spec"])
    m = spec.centres().size
    w = [1.0 / m] * m
    sizes = [int(x) for x in idx["chunk_sizes"]]
    children = np.random.SeedSequence(idx["seed_entropy"]).spawn(len(sizes))
    N = int(idx["n_paths"])
    gb = fk._group_bounds(N, GROUPS)
    cen = em_centre(spec, w)
    tasks = []
    start = 0
    for i, (sz, ch) in enumerate(zip(sizes, children)):
        gg = fk._group_ids(gb, start, sz)
        t = {"spec": spec.to_dict(), "size": sz, "seedseq": ch, "chunk": i, "w": w,
             "centre": cen, "gid": gg - gg.min(), "n_groups_local": int(gg.max() - gg.min() + 1),
             "group_offset": int(gg.min()), "budgets": list(B_LIST), "label": label}
        if i == 0:
            t["verify"] = {"file": idx["chunks"][0]["file"], "cps": idx["cps"]}
        tasks.append(t)
        start += sz
    meta = {"ensemble": CELLS[label], "seed": idx["seed"], "tag": idx["tag"],
            "seed_entropy": idx["seed_entropy"], "n_paths": N, "chunk_sizes": sizes,
            "spec": idx["spec"], "w": w, "groups": GROUPS}
    return meta, tasks


def _run(task):
    r = _chunk(task)
    d = ACC_ROOT / f"n13_{task['label']}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"chunk{task['chunk']:04d}.npz"
    np.savez_compressed(f, AX=r["AX"], AV=r["AV"], FB=r["FB"], PD=r["PD"],
                        group_offset=task["group_offset"])
    return {"label": task["label"], "chunk": r["chunk"], "runtime": r["runtime"], "file": str(f),
            "verify": r.get("verify")}


def cmd_simulate(args) -> None:
    import multiprocessing as mp
    labels = list(CELLS) if not args.only else args.only.split(",")
    tasks, metas = [], {}
    for lab in labels:
        meta, tl = cell_tasks(lab)
        metas[lab] = meta
        tasks += tl
    ctx = mp.get_context("spawn")
    t0 = time.time()
    res = []
    with ctx.Pool(processes=min(3, args.workers)) as pool:
        for r in pool.imap_unordered(_run, tasks):
            res.append(r)
            print(f"[n13] {r['label']} chunk {r['chunk']} {r['runtime']:.1f}s verify={r['verify']}",
                  flush=True)
    for lab in labels:
        metas[lab]["chunks"] = sorted([r for r in res if r["label"] == lab], key=lambda r: r["chunk"])
        p = ACC_ROOT / f"n13_{lab}" / "meta.json"
        p.write_text(json.dumps(metas[lab], indent=1, default=fk._json_default))
    print(f"[n13] wall {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def h2(u):
    """h(u) = int_0^1 r e^{-r u} dr (TH-1), stable for small u."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    sm = u < 1e-3
    us = u[sm]
    out[sm] = 0.5 - us / 3 + us * us / 8
    ul = u[~sm]
    out[~sm] = (1.0 - (1.0 + ul) * np.exp(-ul)) / ul**2
    return out


def h3(u):
    """h3(u) = int_0^1 (r^2/2) e^{-r u} dr = [2 - (2 + 2u + u^2) e^{-u}] / (2 u^3), h3(0) = 1/6."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    sm = u < 1e-2
    us = u[sm]
    out[sm] = 1.0 / 6 - us / 8 + us * us / 20 - us**3 / 72
    ul = u[~sm]
    out[~sm] = (2.0 - (2.0 + 2.0 * ul + ul * ul) * np.exp(-ul)) / (2.0 * ul**3)
    return out


def load_sums(label: str):
    meta = json.loads((ACC_ROOT / f"n13_{label}" / "meta.json").read_text())
    acc = None
    for c in meta["chunks"]:
        with np.load(c["file"]) as d:
            off = int(d["group_offset"])
            part = {k: d[k] for k in ("AX", "AV", "FB", "PD")}
        if acc is None:
            acc = {k: np.zeros((v.shape[0], GROUPS, v.shape[2])) for k, v in part.items()}
        for k, v in part.items():
            acc[k][:, off:off + v.shape[1], :] += v
    return meta, acc


def estimates(S: dict, N: float, cen: np.ndarray, Bs) -> dict:
    """Per-step estimates from total sums S (arrays with the group axis summed out)."""
    Ed = S["AX"][0] / N
    Ed2 = S["AX"][1] / N
    Ed3 = S["AX"][2] / N
    Lam = cen + Ed
    k02 = Ed2 - Ed * Ed
    k03 = Ed3 - 3 * Ed * Ed2 + 2 * Ed**3
    EVd = S["AV"] / N                       # E[V d^k], k = 0..4
    G = EVd[0]
    dl = Ed                                 # Lambda - c
    EVx = []
    for k in range(KV + 1):
        EVx.append(sum(math.comb(k, i) * EVd[i] * (-dl) ** (k - i) for i in range(k + 1)))
    k11 = EVx[1]
    EVx2 = EVx[2]
    EVx4 = EVx[4]
    k12 = EVx2 - G * k02
    out = {"Lam": Lam, "G": G, "k02": k02, "k03": k03, "k11": k11, "k12": k12,
           "EVx2": EVx2, "EVx4": EVx4, "B": {}}
    for ib, B in enumerate(Bs):
        f = B * S["FB"][ib] / N
        E = np.exp(-B * Lam)
        f1 = B * G * E
        D = f - f1
        T2 = -B * B * E * k11
        T3 = 0.5 * B**3 * E * EVx2
        rho3 = D - T2 - T3
        U3 = B**4 * h3(B * Lam) * np.sqrt(np.maximum(EVx2, 0) * np.maximum(EVx4, 0))
        rhoB = D - T2
        UB = B**3 * h2(B * Lam) * EVx2
        T2u = -B * B * k11
        T3u = B**3 * (Lam * k11 + 0.5 * EVx2)
        fc2 = B * (G - B * k11) * np.exp(-B * Lam + 0.5 * B * B * k02)
        fc3 = B * (G - B * k11 + 0.5 * B * B * k12) * np.exp(-B * Lam + 0.5 * B * B * k02 - B**3 * k03 / 6)
        out["B"][B] = {"f": f, "f1": f1, "D": D, "T2": T2, "T3": T3, "rho3": rho3, "U3": U3,
                       "rhoB": rhoB, "UB": UB, "T2u": T2u, "T3u": T3u, "fc2": fc2, "fc3": fc3,
                       "p_disc": S["PD"][ib] / N}
    return out


APPROX = {
    "B2cov_damped": lambda r: r["T2"],
    "two_term_damped": lambda r: r["T2"] + r["T3"],
    "B2cov_undamped": lambda r: r["T2u"],
    "two_term_undamped": lambda r: r["T2u"] + r["T3u"],
    "cumulant2_closure": lambda r: r["fc2"] - r["f1"],
    "cumulant3_closure": lambda r: r["fc3"] - r["f1"],
}


def fractions(r: dict, sel: np.ndarray, binw: int | None = None) -> dict:
    """Share of D = f - f1 explained on the window steps, at step resolution (binw None) or
    after averaging over bins of binw steps.  L1: 1-||D-A||_1/||D||_1; L2sq: 1-||D-A||_2^2/||D||_2^2
    (one minus the SQUARED relative L2 error; audit fix F17, R1 finding B1: key renamed from "L2")."""
    D = r["D"][sel]

    def rb(x):
        if binw is None:
            return x
        nb = x.size // binw
        return x[: nb * binw].reshape(nb, binw).mean(axis=1)

    Db = rb(D)
    out = {}
    for name, fn in APPROX.items():
        A = rb(fn(r)[sel])
        out[name] = {"L1": 1.0 - float(np.sum(np.abs(Db - A)) / np.sum(np.abs(Db))),
                     "L2sq": 1.0 - float(np.sum((Db - A) ** 2) / np.sum(Db**2))}
    return out


def analyze_cell(label: str) -> dict:
    meta, acc = load_sums(label)
    spec = fk.EnsembleSpec.from_dict(meta["spec"])
    cen = em_centre(spec, meta["w"])
    N = float(meta["n_paths"])
    gb = fk._group_bounds(int(N), GROUPS)
    Ng = np.diff(gb).astype(float)
    tot = {k: v.sum(axis=1) for k, v in acc.items()}
    est = estimates(tot, N, cen, B_LIST)
    t = (np.arange(spec.steps()) + 1) * spec.dt
    sel = (t > 0.5 + 1e-9) & (t <= 3.5 + 1e-9)
    binw = int(round(0.02 / spec.dt))
    out = {"label": label, "ensemble": meta["ensemble"], "tag": meta["tag"],
           "seed_entropy": meta["seed_entropy"], "n_paths": int(N), "w": meta["w"],
           "common_path_verification_chunk0": meta["chunks"][0]["verify"], "budgets": {}}
    for B in B_LIST:
        r = est["B"][B]
        rec = {"step": fractions(r, sel), "bin0.02": fractions(r, sel, binw)}
        # jackknife over the 20 path groups
        jk = {"step": [], "bin0.02": []}
        for g in range(GROUPS):
            Sg = {k: tot[k] - acc[k][:, g, :] for k in acc}
            eg = estimates(Sg, N - Ng[g], cen, B_LIST)["B"][B]
            jk["step"].append(fractions(eg, sel))
            jk["bin0.02"].append(fractions(eg, sel, binw))
        for res in ("step", "bin0.02"):
            for name in APPROX:
                for nrm in ("L1", "L2sq"):
                    vals = np.array([j[name][nrm] for j in jk[res]])
                    rec[res][name][f"{nrm}_se_jk"] = float(np.sqrt((GROUPS - 1) / GROUPS
                                                                   * np.sum((vals - vals.mean()) ** 2)))
        D = r["D"][sel]
        scale = float(np.max(np.abs(D)))
        rec["checks"] = {
            "L1_f_minus_f1_window": float(np.sum(np.abs(D)) * spec.dt),
            "max_abs_D": scale,
            "third_order_remainder_bound_violations": int(np.sum(np.abs(r["rho3"][sel]) > r["U3"][sel] + 1e-9 * scale)),
            "max_ratio_abs_rho3_over_bound_where_bound_gt_1e-8_maxabsD": float(np.max(
                (np.abs(r["rho3"][sel]) / np.maximum(r["U3"][sel], 1e-300))[r["U3"][sel] > 1e-8 * scale])),
            "n_steps_bound_gt_1e-8_maxabsD": int(np.sum(r["U3"][sel] > 1e-8 * scale)),
            "max_abs_rho3_over_maxabsD_where_bound_le_1e-8_maxabsD": float(np.max(
                np.abs(r["rho3"][sel])[r["U3"][sel] <= 1e-8 * scale], initial=0.0) / scale),
            "th1_remainder_negative_steps": int(np.sum(r["rhoB"][sel] < -1e-9 * scale)),
            "th1_remainder_above_bound_steps": int(np.sum(r["rhoB"][sel] > r["UB"][sel] + 1e-9 * scale)),
            "L1_rho3_over_L1_D": float(np.sum(np.abs(r["rho3"][sel])) / np.sum(np.abs(D))),
            "window_mass_continuous_density": float(np.sum(r["f"][sel]) * spec.dt),
            "window_mass_discrete_law": float(np.sum(r["p_disc"][sel])),
            "max_B_times_sd_X_window": float(B * np.sqrt(np.max(est["k02"][sel]))),
            "max_lambda_passage_limit": float(B * max(meta["w"]) / min(spec.mu_prime_abs(np.asarray(spec.times()))))}
        out["budgets"][f"{B:g}"] = rec
    out["_est"] = est
    out["_t"] = t
    return out


def cmd_analyze(args) -> None:
    res = {"item": "N13", "driver": HERE.name, "theory": "manuscript/cnsns_submission/theory/TH1b_third_order.tex",
           "definition": "fraction explained over window steps t in (0.5, 3.5]: L1: 1-||D-A||_1/||D||_1; "
                         "L2sq: 1-||D-A||_2^2/||D||_2^2 "
                         "(step resolution dt = 1e-3, and after averaging over 0.02 bins), "
                         "D = f - f_1 with f = B E[V_t e^{-B X_t}] and f_1 = B G e^{-B Lambda} on the sampled "
                         "Euler--Maruyama paths (exact identities for the empirical law); SE: delete-one-group "
                         "jackknife over 20 path groups",
           "cells": {}}
    parts = {}
    for lab in CELLS:
        t0 = time.time()
        parts[lab] = analyze_cell(lab)
        res["cells"][lab] = {k: v for k, v in parts[lab].items() if not k.startswith("_")}
        print(f"[n13] analyzed {lab} {time.time() - t0:.1f}s", flush=True)
    summ = {}
    for B in B_MAIN:
        row = {}
        for lab in CELLS:
            fr = res["cells"][lab]["budgets"][f"{B:g}"]["bin0.02"]
            row[lab] = {n: round(fr[n]["L1"], 4) for n in APPROX}
        summ[f"B{B:g}"] = row
    res["summary_L1_bin0.02"] = summ
    res["summary_ranges_L1_bin0.02"] = {
        f"B{B:g}": {n: [min(summ[f"B{B:g}"][lab][n] for lab in CELLS),
                        max(summ[f"B{B:g}"][lab][n] for lab in CELLS)] for n in APPROX}
        for B in B_MAIN}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=fk._json_default))
    os.replace(tmp, OUT_JSON)
    np.savez_compressed(OUT_DIR / "n13_step_series.npz",
                        **{f"{lab}_{q}_B{B:g}": parts[lab]["_est"]["B"][B][q]
                           for lab in CELLS for B in B_MAIN for q in ("D", "T2", "T3", "fc3")},
                        t=parts[next(iter(CELLS))]["_t"])
    print(f"wrote {OUT_JSON}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("simulate")
    a.add_argument("--only", default="")
    a.add_argument("--workers", type=int, default=3)
    sub.add_parser("analyze")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
