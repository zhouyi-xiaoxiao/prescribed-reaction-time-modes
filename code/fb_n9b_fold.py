#!/usr/bin/env python3
"""N9b: TH-8 finite-width fold, exact law at slab width 0.3 (GAP_CLOSURE_PLAN.md section 3, N9b).

TH-8 claims: at FIXED slab width rho_s (independent of eps), fixed B and w, the
exact reaction-time density converges (in C^2 on the window) to the
deterministic law f_det = B G0 exp(-B Lambda0) as eps -> 0, so the topology
threshold (the fold where the late local maximum and the valley annihilate)
converges to B_top^det = min over rising flanks of max G0'/G0^2, with an O(eps^2)
correction.  For m = 2, rho_s = 0.3: B_top^det = 8.2247 at fold time 1.642
(TH8/finite_width_det.json#scan.m2_rho0.3.B_top_det).

This driver locates the fold of the EXACT law of the production process
(Euler--Maruyama, dt = 1e-3, contact gate, N0 FK estimator) at slab width
0.3, m = 2, equal weights, eps in {0.06, 0.03, 0.02}, on a dense budget grid.

Fold statistic.  For a Gaussian smoothing bandwidth h, let f_h = f * K_h (the
exact bin masses on the production 0.02 grid smoothed analytically:
f_h(t) = sum_i p_i K_h(t - t_i)) and
    D_h(B) = max_{t in [1.0, 2.6]} d/dt log f_h(t; B),
the largest log-slope after the first passage.  D_h > 0 iff f_h still rises
again after the valley (late maximum present).  B*_h(eps) = the zero of D_h(B)
(linear interpolation on the dense grid; jackknife over 20 path groups).
Smoothing shifts the fold by O(h^2); the SAME statistic is applied to the
deterministic law f_det (binned on the same grid) and to the finite-eps
mean-field law B G_eps exp(-B Lambda_eps) (G_eps = exact EM free-exposure clock),
so B*_h(eps) -> B^det_h is tested like-for-like at each h, and an h -> 0
extrapolation (quadratic in h) is reported for reference.

Usage
-----
    python3 fb_n9b_fold.py simulate [--eps 0.06,0.03,0.02]
    python3 fb_n9b_fold.py analyze
    python3 fb_n9b_fold.py figure
Seeds: base 20260923, tag 87 (N9), replicate 0; 2e5 paths per eps.
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

OUT = fk.FB_DATA / "N9b"
TAG = fk.TAGS["N9"]
SLAB_SD = 0.3
EPS_LIST = (0.06, 0.03, 0.02)
N_PATHS = 200_000
B_GRID = np.round(np.arange(5.5, 9.0 + 1e-9, 0.05), 10)   # contains the plan's [7, 8.5] step 0.05
H_LIST = (0.02, 0.025, 0.03, 0.04, 0.05)
T_REGION = (1.0, 2.6)
GROUPS = 20
W = (0.5, 0.5)
DET_JSON = fk.FB_DATA / "TH8" / "finite_width_det.json"


def ens_name(eps: float) -> str:
    return f"n9b_m2_slab0.3_eps{eps:g}"


def cmd_simulate(args):
    for eps in [float(x) for x in args.eps.split(",")]:
        name = ens_name(eps)
        ip = fk.index_path(name)
        if ip.exists() and json.loads(ip.read_text()).get("complete") and not args.force:
            print(f"[n9b] {name} complete; skip", flush=True)
            continue
        t0 = time.time()
        spec = fk.EnsembleSpec(m=2, eps=eps, slab_sd=SLAB_SD)
        fk.simulate_ensemble(spec, N_PATHS, name=name, tag=TAG, variants=("full",),
                             workers=args.workers, overwrite=args.force,
                             note="N9b TH-8 fold, fixed slab width 0.3 (fb_n9b_fold.py)")
        print(f"[n9b] {name} done {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Fold statistic
# ----------------------------------------------------------------------------


def smoothed_logslope_max(P: np.ndarray, tc: np.ndarray, h: float, tq: np.ndarray) -> np.ndarray:
    """P (nB, K) bin masses at bin centres tc -> max_{tq} d/dt log(sum_i P_i K_h(t - t_i))."""
    d = tq[:, None] - tc[None, :]
    K = np.exp(-0.5 * (d / h) ** 2)                 # (nq, K), normalisation cancels in log-slope
    Kp = -(d / (h * h)) * K
    fh = np.einsum("bk,qk->bq", P, K)
    fp = np.einsum("bk,qk->bq", P, Kp)
    ls = fp / fh
    return ls.max(axis=1), tq[np.argmax(ls, axis=1)]


def zero_crossing(Bs: np.ndarray, D: np.ndarray):
    """Largest B where D changes sign + -> - (last down-crossing on the grid)."""
    pos = D > 0
    idx = [i for i in range(Bs.size - 1) if pos[i] and not pos[i + 1]]
    if not idx:
        return None, ("positive_throughout" if pos.all() else
                      "negative_throughout" if not pos.any() else "no_down_crossing")
    i = idx[-1]
    x = Bs[i] + (Bs[i + 1] - Bs[i]) * D[i] / (D[i] - D[i + 1])
    return float(x), ("unique" if len(idx) == 1 else f"{len(idx)}_down_crossings_last_taken")


def det_bin_masses(Bs, edges, eps_mf=None, dt_mf=1e-3, n_sub=40):
    """Deterministic (eps = 0) or finite-eps mean-field law binned on `edges`.

    eps_mf None: G0(t) = sum_j w_j phi_{0.3}(mu(t) - c_j) (contact 1, W = 1);
    else: G_eps = exact EM free-exposure clock of the production chain at eps_mf
    (fk.free_exposure_discrete with the contact gate).  f = B G exp(-B Lambda).
    """
    spec0 = fk.EnsembleSpec(m=2, eps=0.05 if eps_mf is None else eps_mf, slab_sd=SLAB_SD)
    if eps_mf is None:
        t = np.linspace(0.0, edges[-1], int(round(edges[-1] / 1e-3)) * n_sub + 1)
        mu = spec0.z_bar + (spec0.z0 - spec0.z_bar) * np.exp(-spec0.gamma * t)
        G = sum(wj * np.exp(-0.5 * ((mu - c) / SLAB_SD) ** 2) / (math.sqrt(2 * math.pi) * SLAB_SD)
                for wj, c in zip(W, spec0.centres())) / spec0.torus_w
        Lam = np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1]) * np.diff(t))])
        out = []
        for B in Bs:
            S = np.exp(-B * np.interp(edges, t, Lam))
            out.append(S[:-1] - S[1:])
        return np.array(out)
    fe = fk.free_exposure_discrete(spec0, W, gate="contact")
    T = fe["t"]
    Lam_steps = np.cumsum(fe["G"] * spec0.dt)             # exposure after step n (kill at end of step)
    cps = np.searchsorted(T, edges, side="left")
    Lam_e = np.where(cps > 0, Lam_steps[np.maximum(cps - 1, 0)], 0.0)
    out = []
    for B in Bs:
        S = np.exp(-B * Lam_e)
        out.append(S[:-1] - S[1:])
    return np.array(out)


def det_fold_exact() -> dict:
    d = json.loads(DET_JSON.read_text())
    s = d["scan"]["m2_rho0.3"]
    return {"B_top_det": s["B_top_det"], "fold_time": s["fold_time"],
            "source": "artifacts/data/exact_m_fixed_budget/TH8/finite_width_det.json#scan.m2_rho0.3"}


def analyze_eps(eps: float) -> dict:
    t0 = time.time()
    ens = fk.load_ensemble(ens_name(eps))
    ens.cache_in_memory = True
    edges = ens.edges
    tc = 0.5 * (edges[:-1] + edges[1:])
    tq = np.arange(T_REGION[0], T_REGION[1] + 1e-9, 0.002)
    tab = fk.survival_table(ens, B_GRID, W, index=np.arange(edges.size), groups=GROUPS)
    N = tab["N"]
    Pg = tab["P_group_sums"]            # (G, nB, K)
    sizes = tab["group_sizes"]
    P = Pg.sum(0) / N
    res = {"eps": eps, "ensemble": ens.name, "n_paths": N, "tag": int(ens.index["tag"]),
           "seed": int(ens.index["seed"]), "seed_entropy": ens.index["seed_entropy"],
           "slab_sd": SLAB_SD, "B_grid": B_GRID.tolist(), "by_h": {}}
    mf = det_bin_masses(B_GRID, edges, eps_mf=eps)
    for h in H_LIST:
        D, tstar = smoothed_logslope_max(P, tc, h, tq)
        Bz, st = zero_crossing(B_GRID, D)
        jack = []
        for g in range(GROUPS):
            Pj = (Pg.sum(0) - Pg[g]) / (N - sizes[g])
            Dj, _ = smoothed_logslope_max(Pj, tc, h, tq)
            jack.append(zero_crossing(B_GRID, Dj)[0])
        jr = np.array([x for x in jack if x is not None])
        se = float(math.sqrt((jr.size - 1) / jr.size * np.sum((jr - jr.mean()) ** 2))) if jr.size > 1 else None
        Dm, _ = smoothed_logslope_max(mf, tc, h, tq)
        Bmf, stmf = zero_crossing(B_GRID, Dm)
        i_near = int(np.argmin(np.abs(B_GRID - (Bz if Bz else 8.0))))
        res["by_h"][f"{h:g}"] = {"h": h, "B_star": Bz, "status": st, "jackknife_se": se,
                                 "n_jack_valid": int(jr.size), "D": D.tolist(),
                                 "t_argmax": tstar.tolist(),
                                 "fold_time_near_B_star": float(tstar[i_near]),
                                 "B_star_meanfield_eps": Bmf, "meanfield_status": stmf}
    # late basin mass at the fold (cut at the G0 valley 1.4917 -> nearest edge)
    cut = int(np.argmin(np.abs(edges - 1.4917)))
    res["late_mass_by_B"] = P[:, cut:].sum(1).tolist()
    res["late_cut_edge"] = float(edges[cut])
    # raw (unsmoothed) significant maxima census on 0.02 bins, per B
    gm = Pg / sizes[:, None, None]
    dens = P / np.diff(edges)[None, :]
    gd = gm / np.diff(edges)[None, None, :]
    census = []
    for ib, B in enumerate(B_GRID):
        ds = fk.derivative_signs(tc, dens[ib], gd[:, ib, :], bandwidth=0.0, zthr=5.0)
        census.append({"B": float(B), "n_maxima": ds["n_maxima"], "maxima_t": ds["maxima_t"],
                       "n_minima": ds["n_minima"], "minima_t": ds["minima_t"]})
    res["unsmoothed_significant_extrema_z5"] = census
    res["wall_seconds"] = time.time() - t0
    ens.clear_cache()
    return res


def cmd_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    det = det_fold_exact()
    edges = fk.build_grid(fk.EnsembleSpec(m=2, eps=0.03, slab_sd=SLAB_SD))["edges"]
    tc = 0.5 * (edges[:-1] + edges[1:])
    tq = np.arange(T_REGION[0], T_REGION[1] + 1e-9, 0.002)
    Bfine = np.round(np.arange(5.0, 12.0 + 1e-9, 0.005), 10)
    Pdet = det_bin_masses(Bfine, edges, None)
    det_h = {}
    for h in H_LIST:
        D, _ = smoothed_logslope_max(Pdet, tc, h, tq)
        det_h[f"{h:g}"] = zero_crossing(Bfine, D)[0]
    out = {"analysis": "N9b: TH-8 fold of the exact law at slab width 0.3, m=2, equal weights",
           "deterministic_limit": det, "deterministic_same_statistic_by_h": det_h,
           "definition": ("B*_h = zero of D_h(B) = max_{t in [1.0,2.6]} d/dt log (f*K_h)(t;B), "
                          "f = exact bin masses on the production 0.02 grid, K_h Gaussian of sd h"),
           "h_list": list(H_LIST), "t_region": list(T_REGION), "groups": GROUPS,
           "seeds": {"base_seed": fk.BASE_SEED, "tag": TAG, "replicate": 0}, "eps": {}}
    for eps in EPS_LIST:
        ip = fk.index_path(ens_name(eps))
        if not ip.exists():
            continue
        r = analyze_eps(eps)
        out["eps"][f"{eps:g}"] = r
        print(f"[n9b] eps {eps}: " + ", ".join(
            f"h={k}: {v['B_star']} ± {v['jackknife_se']}" for k, v in r["by_h"].items()), flush=True)
    # convergence fits per h: B*(eps) = B0 + c eps^2 (weighted LS), compare B0 with det_h
    fits = {}
    for h in H_LIST:
        k = f"{h:g}"
        xs, ys, ss = [], [], []
        for eps in EPS_LIST:
            r = out["eps"].get(f"{eps:g}")
            if r and r["by_h"][k]["B_star"] is not None and r["by_h"][k]["jackknife_se"]:
                xs.append(eps ** 2)
                ys.append(r["by_h"][k]["B_star"])
                ss.append(r["by_h"][k]["jackknife_se"])
        if len(xs) < 2:
            continue
        xs, ys, ss = map(np.asarray, (xs, ys, ss))
        wts = 1.0 / ss ** 2
        X = np.vstack([np.ones_like(xs), xs]).T
        cov = np.linalg.inv(X.T @ (X * wts[:, None]))
        beta = cov @ (X.T @ (wts * ys))
        resid = ys - X @ beta
        chi2 = float(np.sum(wts * resid ** 2))
        # fit with B0 fixed to the deterministic value: B* - Bdet = c eps^q
        if det_h[k] is None:
            continue
        dy = ys - det_h[k]
        q_pairs = []
        for i in range(len(xs) - 1):
            if dy[i] < 0 and dy[i + 1] < 0:
                q_pairs.append(float(math.log(dy[i] / dy[i + 1]) /
                                     math.log(math.sqrt(xs[i]) / math.sqrt(xs[i + 1]))))
        fits[k] = {"B0": float(beta[0]), "B0_se": float(math.sqrt(cov[0, 0])), "c": float(beta[1]),
                   "c_se": float(math.sqrt(cov[1, 1])), "chi2": chi2, "dof": int(len(xs) - 2),
                   "B0_minus_det_h": float(beta[0] - det_h[k]),
                   "B0_minus_det_h_z": float((beta[0] - det_h[k]) / math.sqrt(cov[0, 0])),
                   "local_exponents_q_from_det_anchor": q_pairs,
                   "B_star_minus_det_h": dy.tolist()}
    out["convergence_fits_eps2"] = fits
    # h -> 0 extrapolation per eps (quadratic in h)
    hx = {}
    for eps in EPS_LIST:
        r = out["eps"].get(f"{eps:g}")
        if not r:
            continue
        hs = np.array([h for h in H_LIST if r["by_h"][f"{h:g}"]["B_star"] is not None])
        bs = np.array([r["by_h"][f"{h:g}"]["B_star"] for h in hs])
        if hs.size >= 3:
            A = np.vstack([np.ones_like(hs), hs ** 2]).T
            coef, *_ = np.linalg.lstsq(A, bs, rcond=None)
            hx[f"{eps:g}"] = {"B_star_h0": float(coef[0]), "slope_h2": float(coef[1])}
    hd = [h for h in H_LIST if det_h[f"{h:g}"] is not None]
    Adet = np.array([[1.0, h * h] for h in hd])
    cdet, *_ = np.linalg.lstsq(Adet, np.array([det_h[f"{h:g}"] for h in hd]), rcond=None)
    hx["deterministic"] = {"B_star_h0": float(cdet[0]), "slope_h2": float(cdet[1]),
                           "exact_B_top_det": det["B_top_det"], "h_used": hd}
    out["h_to_0_extrapolation"] = hx
    core.write_json(OUT / "n9b_fold.json", fk._jsonable(out))
    print(json.dumps({"det_h": det_h, "fits": fits, "hx": hx}, indent=1))


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    R = json.loads((OUT / "n9b_fold.json").read_text())
    hsel = "0.03"
    cols = {"0.06": core.OI_BLUE, "0.03": core.OI_ORANGE, "0.02": core.OI_GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    ax = axes[0]
    for e, r in R["eps"].items():
        Bs = np.array(r["B_grid"])
        D = np.array(r["by_h"][hsel]["D"])
        ax.plot(Bs, D, color=cols.get(e, "k"), lw=1.1, label=f"exact law, $\\varepsilon={e}$")
        bz = r["by_h"][hsel]["B_star"]
        if bz:
            ax.axvline(bz, color=cols.get(e, "k"), lw=0.6, ls=":")
    ax.axvline(R["deterministic_same_statistic_by_h"][hsel], color="k", lw=0.9, ls="--",
               label=r"deterministic limit $f_{\rm det}$ (same statistic)")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel(r"budget $B$ (dimensionless)")
    ax.set_ylabel(r"$D_h(B)=\max_t\,\partial_t\log f_h$ (1/time)")
    ax.set_title(f"slab width 0.3, $m=2$; smoothing $h={hsel}$", fontsize=7.5)
    ax.legend(fontsize=6.2, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    hmark = {"0.02": "o", "0.025": "s", "0.03": "^", "0.04": "D", "0.05": "v"}
    hcol = {"0.02": core.OI_BLUE, "0.025": core.OI_ORANGE, "0.03": core.OI_GREEN,
            "0.04": core.OI_VERMILLION, "0.05": core.OI_PURPLE}
    for k, mk in hmark.items():
        xs, ys, es = [], [], []
        for e, r in R["eps"].items():
            v = r["by_h"].get(k)
            if v and v["B_star"] is not None and R["deterministic_same_statistic_by_h"].get(k) is not None:
                xs.append(float(e) ** 2 * 1e3)
                ys.append(v["B_star"] - R["deterministic_same_statistic_by_h"][k])
                es.append(1.96 * (v["jackknife_se"] or 0.0))
        ax.errorbar(xs, ys, yerr=es, marker=mk, ms=3.2, lw=0, elinewidth=0.8, capsize=1.5,
                    color=hcol[k], label=f"$h={k}$")
        f = R["convergence_fits_eps2"].get(k)
        if f:
            xx = np.linspace(0, 3.7, 50)
            ax.plot(xx, f["B0"] - R["deterministic_same_statistic_by_h"][k] + f["c"] * xx * 1e-3,
                    lw=0.7, ls=":", color=hcol[k])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$\varepsilon^2$ ($10^{-3}$)")
    ax.set_ylabel(r"$B^*_h(\varepsilon)-B^{\rm det}_h$ (budget)")
    ax.set_title("fold offset vs $\\varepsilon^2$ (dotted: fits $B_0+c\\varepsilon^2$)", fontsize=7.5)
    ax.legend(fontsize=6.0, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    written = core.save_figure(fig, core.FIGURES / "fb_n9b_fold")
    plt.close(fig)
    print(written)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--eps", default=",".join(str(e) for e in EPS_LIST))
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args(argv)
    {"simulate": cmd_simulate, "analyze": cmd_analyze, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
