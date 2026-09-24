#!/usr/bin/env python3
"""N0 acceptance run for the exact-law Feynman--Kac estimator (GAP_CLOSURE_PLAN.md section 3, N0).

Compares the FK/Rao--Blackwell exact law of ``exact_m_prr_fk_exact_law`` (stored
N0 ensembles, base seed 20260923, tag 80) with every stored direct-kill
histogram of the production model at the same (m, eps):

* primary cells (acceptance): (2, 0.1, 1) and (3, 0.1, 1) at 5e6 walkers
  (production), (2, 0.05, 1) at 1e6 walkers (W1; no 5e6 cell exists);
* secondary cells: all W1 phase-diagram cells, the 70/30 and 50/30/20 production
  allocations (tests the linear-in-w reweighting) and the W2 bisection probes.

Per cell: bin z-scores on the 150 production window bins (z uses the FK iid SE
and the multinomial SE of the direct-kill histogram evaluated at the FK
probability), chi-square, window basin masses (cuts at the snapped G valleys),
kill fractions, and the expected-count classifier verdict.  Then the
prominence-floor crossing (B at which the last-mode relative prominence of the
exact expected law falls to 5 %, protocol walkers 1e6, W2 last-basin rule) with
a delete-one-group jackknife SE, compared with the stored B_op brackets and the
S2 floor-interpolated crossings; and an FK-vs-FK cross-check against the
independent theory-scout ensembles (PCG64 streams, 2026-09-23 cache).

Writes artifacts/data/exact_m_fixed_budget/n0_validation.json and figures
artifacts/figures/fb_n0_fk_validation.{png,pdf}, fb_n0_prominence_floor.{png,pdf}.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402

DATA = fk.REPORT / "artifacts" / "data"
UPG = DATA / "exact_m_prr_upgrade"
PROD = DATA / "exact_m_offlattice_production"
OUT = fk.FB_DATA / "n0_validation.json"

ENSEMBLES = {(2, 0.1): "n0_m2_eps0.1", (3, 0.1): "n0_m3_eps0.1",
             (2, 0.05): "n0_m2_eps0.05", (3, 0.05): "n0_m3_eps0.05"}
PRIMARY = [
    ("m2_eps0.1_B1", PROD / "m2_eps0.1_B1_w50-50.json"),
    ("m3_eps0.1_B1", PROD / "m3_eps0.1_B1_w33-33-33.json"),
    ("m2_eps0.05_B1", UPG / "w1_phase_diagram" / "cell_m2_eps0.05_B1.json"),
]
BOP_CHAINS = [(2, 0.05, 5.0, 9.0), (3, 0.05, 3.0, 5.5), (2, 0.1, 6.0, 9.0), (3, 0.1, 2.8, 4.6)]
SCOUT_FILES = {(2, 0.1): "fk_m2_e0.1.npz", (3, 0.1): "fk_m3_e0.1.npz",
               (2, 0.05): "fk_m2_e0.05.npz"}
SCOUT_BOP_FILES = {(2, 0.05): "fkb_m2_e0.05.npz", (3, 0.05): "fkb_m3_e0.05.npz"}
# kernel/geometry variants vs their own direct-kill production data (5e6 walkers each)
VARIANT_CHECKS = [
    ("w6_p1_m2_eps0.05", UPG / "w6_single_particle" / "m2_eps0.05_B1_p1.json", "n0_m2_eps0.05", "p1"),
    ("w6_p2_m2_eps0.05", UPG / "w6_single_particle" / "m2_eps0.05_B1_p2.json", "n0_m2_eps0.05", "p2"),
    ("w6_p1_m2_eps0.1", UPG / "w6_single_particle" / "m2_eps0.1_B1_p1.json", "n0_m2_eps0.1", "p1"),
    ("w6_p2_m2_eps0.1", UPG / "w6_single_particle" / "m2_eps0.1_B1_p2.json", "n0_m2_eps0.1", "p2"),
    ("w6_p1_m3_eps0.1", UPG / "w6_single_particle" / "m3_eps0.1_B1_p1.json", "n0_m3_eps0.1", "p1"),
    ("w6_p2_m3_eps0.1", UPG / "w6_single_particle" / "m3_eps0.1_B1_p2.json", "n0_m3_eps0.1", "p2"),
    ("w5_d3_m2_eps0.1", UPG / "w5_d3_spotcheck" / "d3_spotcheck.json", "n0_d3_m2_eps0.1", "full"),
    ("w4_m5_z0_8", UPG / "w4_m5_demo" / "m5_demo.json", "n0_m5_z08_eps0.1", "full"),
]


def _chi2_sf(x: float, k: int) -> float:
    """Wilson--Hilferty upper tail of chi-square(k)."""
    if k <= 0:
        return float("nan")
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def read_reference(path: Path, model=None) -> dict:
    model = fk.MODEL if model is None else model
    d = json.loads(path.read_text())
    par = d["parameters"]
    cfg = par.get("config", par)
    res = d["results"]
    cl = res["classifier"]
    walkers = int(cfg.get("walkers") or par.get("walkers"))
    mp = par.get("model_parameters")
    if mp is not None:
        for k, v in mp.items():
            if k in ("dim",):
                continue
            if abs(float(getattr(model, k)) - float(v)) > 1e-12:
                raise ValueError(f"{path.name}: model parameter {k} differs")
    dt = float(cfg.get("dt", par.get("dt", 1e-3)))
    tmax = float(cfg.get("tmax", par.get("tmax", 4.0)))
    return {"file": str(path.relative_to(fk.REPORT)), "m": int(cfg["m"]), "eps": float(cfg["eps"]),
            "B": float(cfg["budget"]), "w": tuple(float(x) for x in cfg["weights"]),
            "walkers": walkers, "dt": dt, "tmax": tmax,
            "seed": cfg.get("seed", par.get("seed")), "tag": cfg.get("tag"),
            "counts": np.asarray(cl["counts"], float), "edges": np.asarray(cl["edges"], float),
            "kills": int(res["kills"]), "kills_in_window": int(res["kills_in_window"]),
            "stored_mode_count": int(res.get("mode_count", cl.get("mode_count", -1))),
            "stored_peak_times": res.get("peak_times")}


def secondary_references() -> list:
    refs = []
    for p in sorted((UPG / "w1_phase_diagram").glob("cell_m*_eps*_B*.json")):
        refs.append(("w1", p))
    for p in sorted(PROD.glob("m*_eps0.1_B*_w*.json")):
        if "dt5e-4" in p.name:
            continue
        refs.append(("production", p))
    for p in sorted((UPG / "w2_b0_empirical").glob("probe_m*_eps*_B*.json")):
        refs.append(("w2_probe", p))
    return refs


def merge_groups(expected_counts: np.ndarray, min_count: float = 10.0) -> list:
    """Contiguous bin groups with expected count >= min_count (greedy; remainder merged left)."""
    groups, cur, acc = [], [], 0.0
    for i, c in enumerate(expected_counts):
        cur.append(i)
        acc += c
        if acc >= min_count:
            groups.append(cur)
            cur, acc = [], 0.0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def _cov_chi2(d: np.ndarray, S: np.ndarray, rel_cut: float = 1e-10) -> tuple:
    """d^T S^+ d with an eigen-pseudo-inverse (eigenvalues below rel_cut*max dropped)."""
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    keep = vals > rel_cut * vals.max()
    proj = np.einsum("ij,i->j", vecs[:, keep], d)
    return float(np.sum(proj * proj / vals[keep])), int(keep.sum())


def compare_cell(ens: fk.Ensemble, ref: dict, groups: int, variant: str = "full") -> dict:
    if not np.array_equal(ref["edges"], fk.production_window_edges()):
        raise ValueError("reference edges differ from production window edges")
    law = fk.exact_law(ens, ref["B"], ref["w"], groups=groups, with_cov=True, variant=variant)
    p = law["bin_mass"]
    C = law["cov"]
    N = ref["walkers"]
    cnt = ref["counts"]
    pdk = cnt / N
    # (a) raw per-bin z (Gaussian; invalid for tiny expected counts, kept as specified)
    var = law["se_iid"] ** 2 + p * (1.0 - p) / N
    z = np.where(var > 0, (p - pdk) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vo = law["se_iid"] ** 2 + pdk * (1 - pdk) / N
        z_obs = np.where(vo > 0, (p - pdk) / np.sqrt(np.where(vo > 0, vo, 1.0)), 0.0)
    # (b) merged bins, expected direct-kill count >= 10 (Cochran-type rule)
    grp = merge_groups(N * p, 10.0)
    A = np.zeros((len(grp), p.size))
    for gi, g in enumerate(grp):
        A[gi, g] = 1.0
    pm = np.einsum("gk,k->g", A, p)
    pdm = np.einsum("gk,k->g", A, pdk)
    Cm = np.einsum("gk,kl,hl->gh", A, C, A)
    var_m = np.diag(Cm) + pm * (1 - pm) / N
    zm = (pm - pdm) / np.sqrt(var_m)
    Sig = Cm + (np.diag(pm) - np.outer(pm, pm)) / N
    chi2_cov, dof_cov = _cov_chi2(pm - pdm, Sig)
    # basins, window and total kill fractions (per-path SEs)
    bm = fk.basin_masses(ens, ref["B"], ref["w"], groups=groups, variant=variant)
    ci = [int(np.argmin(np.abs(law["edges"] - c))) for c in bm["cuts_edges"]]
    cum = np.concatenate([[0.0], np.cumsum(cnt)])
    dk_basin = np.array([(cum[ci[j + 1]] - cum[ci[j]]) / N for j in range(len(ci) - 1)])
    se_dk = np.sqrt(bm["masses"] * (1 - bm["masses"]) / N)
    zb = (bm["masses"] - dk_basin) / np.sqrt(bm["se_iid"] ** 2 + se_dk**2)
    win = fk.basin_masses(ens, ref["B"], ref["w"], cuts=[0.5, 3.5], groups=groups,
                          variant=variant)
    tot = fk.basin_masses(ens, ref["B"], ref["w"], cuts=[0.0, ens.spec.tmax], groups=groups,
                          variant=variant)
    wf, wse = float(win["masses"][0]), float(win["se_iid"][0])
    kf, kse = float(tot["masses"][0]), float(tot["se_iid"][0])
    wdk, kdk = ref["kills_in_window"] / N, ref["kills"] / N
    cls = fk.classify_expected(p, law["edges"], walkers=N)
    sig = [r for r in cls["rows"] if r["significant"]]
    pos = p > 1e-6
    return {
        "reference": ref["file"], "m": ref["m"], "eps": ref["eps"], "B": ref["B"],
        "w": list(ref["w"]), "walkers_direct_kill": N, "reference_seed": ref["seed"],
        "reference_tag": ref["tag"], "fk_paths": ens.n_paths, "fk_ensemble": ens.name,
        "fk_variant": variant, "n_bins": int(p.size), "max_abs_z": float(np.max(np.abs(z))),
        "argmax_abs_z_time": float(law["t"][int(np.argmax(np.abs(z)))]),
        "expected_dk_count_at_argmax": float(N * p[int(np.argmax(np.abs(z)))]),
        "n_bins_abs_z_ge_4": int(np.sum(np.abs(z) >= 4.0)),
        "n_bins_abs_z_ge_3": int(np.sum(np.abs(z) >= 3.0)),
        "max_abs_z_observed_variance": float(np.max(np.abs(z_obs))),
        "chi2_diag": float(np.sum(z * z)), "chi2_diag_over_dof": float(np.sum(z * z) / p.size),
        "merged_min_expected_count": 10.0, "n_merged_bins": len(grp),
        "merged_max_abs_z": float(np.max(np.abs(zm))),
        "merged_n_abs_z_ge_4": int(np.sum(np.abs(zm) >= 4.0)),
        "merged_n_abs_z_ge_3": int(np.sum(np.abs(zm) >= 3.0)),
        "chi2_cov_merged": chi2_cov, "chi2_cov_dof": dof_cov,
        "chi2_cov_over_dof": chi2_cov / max(dof_cov, 1),
        "chi2_cov_p_value_wilson_hilferty": _chi2_sf(chi2_cov, dof_cov),
        "se_ratio_fk_to_dk_median": float(np.median(law["se_iid"][pos]
                                                    / np.sqrt(p[pos] * (1 - p[pos]) / N))),
        "se_batch_over_se_iid_median": float(np.median(law["se_batch"][pos] / law["se_iid"][pos])),
        "basin_cuts": bm["cuts_edges"], "basin_masses_fk": bm["masses"].tolist(),
        "basin_se_fk": bm["se_iid"].tolist(), "basin_masses_dk": dk_basin.tolist(),
        "basin_se_dk": se_dk.tolist(), "basin_z": zb.tolist(),
        "basin_within_2se": bool(np.all(np.abs(zb) <= 2.0)),
        "basin_mean_field": bm["mean_field"].tolist(),
        "basin_limit_stick_breaking": bm["limit_stick_breaking"].tolist(),
        "window_fraction_fk": wf, "window_fraction_fk_se": wse, "window_fraction_dk": wdk,
        "window_fraction_z": (wf - wdk) / math.sqrt(wse**2 + wf * (1 - wf) / N),
        "kill_fraction_fk": kf, "kill_fraction_fk_se": kse, "kill_fraction_dk": kdk,
        "kill_fraction_z": (kf - kdk) / math.sqrt(kse**2 + kf * (1 - kf) / N),
        "fk_expected_classifier_mode_count_at_dk_walkers": int(cls["mode_count"]),
        "fk_expected_significant_times": [r["time"] for r in sig],
        "stored_mode_count": ref["stored_mode_count"],
        "stored_peak_times": ref["stored_peak_times"],
        "_plot": {"t": law["t"].tolist(), "fk_density": law["density"].tolist(),
                  "fk_density_se": law["density_se_iid"].tolist(),
                  "dk_density": (pdk / np.diff(law["edges"])).tolist(), "z": z.tolist()},
    }


def bop_reference(m: int, eps: float) -> dict:
    b0 = json.loads((UPG / "w2_b0_empirical" / "B0_empirical.json").read_text())
    chain = next(c for c in b0["chains"] if c["m"] == m and abs(c["eps"] - eps) < 1e-12)
    out = {"stored_b0": chain.get("b0"), "stored_bracket": chain.get("b0_bracket"),
           "stored_status": chain.get("status"),
           "basin_edge_last_g_valley_time": chain["basin_edge_last_g_valley_time"],
           "source": "exact_m_prr_upgrade/w2_b0_empirical/B0_empirical.json#chains"}
    s2 = json.loads((UPG / "robustness" / "w2_seed_repeat_summary.json").read_text())
    cell = next((c for c in s2["cells"] if c["cell"] == f"m{m}_eps{eps:g}"), None)
    if cell is not None:
        sd = cell["sampling_scale_diagnostic"]
        out["s2_four_chain_envelope"] = cell["four_chain_including_campaign_seed"]["envelope"]
        out["s2_floor_interpolated_crossings"] = sd["floor_interpolated_crossings"]
        out["s2_floor_interpolated_mean"] = sd["floor_interpolated_mean"]
        out["s2_floor_interpolated_sd"] = sd["floor_interpolated_sd"]
        out["s2_pooled_regression_crossing"] = sd["pooled_regression"]["crossing"]
        out["s2_pooled_regression_se"] = sd["pooled_regression"]["crossing_se_1sigma"]
        out["s2_source"] = "exact_m_prr_upgrade/robustness/w2_seed_repeat_summary.json#cells"
    return out


def scout_crossing(m: int, eps: float, after_time: float) -> dict | None:
    f = fk.SCOUT_CACHE / SCOUT_BOP_FILES.get((m, eps), "")
    if not f.is_file():
        return None
    sc = fk.load_scout_cache(f)
    edges = sc["edges"]
    Bs = np.asarray(sc["budgets"], float)
    r = np.array([fk.last_mode_prominence(fk.classify_expected(sc["FK_full"][i], edges,
                                                               walkers=1_000_000), after_time)
                  for i in range(Bs.size)])
    Bx, _ = fk._crossing(Bs, r, 0.05)
    return {"file": str(f), "paths": sc["N"], "scout_seed": sc["seed"], "budgets": Bs.tolist(),
            "last_mode_rel_prominence": r.tolist(), "crossing_loglinear": Bx}


def scout_crosscheck(ens: fk.Ensemble, m: int, eps: float, groups: int) -> dict | None:
    """FK(N0, Philox) vs FK(scout, PCG64) at B = 1: two independent estimates of one law."""
    f = fk.SCOUT_CACHE / SCOUT_FILES.get((m, eps), "")
    if not f.is_file():
        return None
    sc = fk.load_scout_cache(f)
    bi = int(np.flatnonzero(np.isclose(sc["budgets"], 1.0))[0])
    ps = sc["FK_full"][bi]
    pb = sc["FK_full_batches"][:, bi, :]
    law = fk.exact_law(ens, 1.0, None, groups=groups, with_cov=True)
    p = law["bin_mass"]
    ratio = ens.n_paths / sc["N"]
    se_s = law["se_iid"] * math.sqrt(ratio)
    var = law["se_iid"] ** 2 + se_s**2
    z = np.where(var > 0, (p - ps) / np.sqrt(np.where(var > 0, var, 1)), 0.0)
    sel = p >= 1e-6
    Sig = law["cov"][np.ix_(sel, sel)] * (1.0 + ratio)
    chi2, dof = _cov_chi2((p - ps)[sel], Sig)
    se_sb = pb.std(axis=0, ddof=1) / math.sqrt(pb.shape[0])
    wn = float(p.sum()); ws = float(ps.sum())
    return {"scout_file": str(f), "scout_paths": sc["N"], "scout_seed": sc["seed"],
            "scout_rng": "PCG64 (fk_paths.py, float64 per-step), independent of the Philox N0 streams",
            "B": 1.0, "max_abs_z_all_bins": float(np.abs(z).max()),
            "argmax_time": float(law["t"][int(np.argmax(np.abs(z)))]),
            "max_abs_z_bins_p_ge_1e-6": float(np.abs(z[sel]).max()),
            "chi2_diag_over_dof_p_ge_1e-6": float(np.sum(z[sel] ** 2) / sel.sum()),
            "chi2_cov_p_ge_1e-6": chi2, "chi2_cov_dof": dof, "chi2_cov_over_dof": chi2 / dof,
            "chi2_cov_p_value_wilson_hilferty": _chi2_sf(chi2, dof),
            "window_mass_n0": wn, "window_mass_scout": ws,
            "se_model": ("scout covariance taken as the N0 covariance x N0/N_scout (same estimator); "
                         "chi2_cov uses the full bin covariance (FK bin errors are correlated)"),
            "median_ratio_scout_4batch_se_to_assumed": float(np.median(se_sb[sel] / se_s[sel]))}


def make_figures(primary: list, bops: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core.apply_prr_style()
    written = []
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]}, layout="constrained")
    for col, row in enumerate(primary):
        pl = row["_plot"]
        t = np.asarray(pl["t"]); f = np.asarray(pl["fk_density"]); se = np.asarray(pl["fk_density_se"])
        ax = axes[0, col]
        ax.fill_between(t, f - 2 * se, f + 2 * se, color=core.OI_BLUE, alpha=0.3, lw=0,
                        label="FK exact law $\\pm2$SE")
        ax.plot(t, f, color=core.OI_BLUE, lw=1.0)
        ax.plot(t, pl["dk_density"], ".", ms=2.0, color=core.OI_VERMILLION,
                label=f"direct kill ($N$={row['walkers_direct_kill']:.0e})")
        ax.set_title(f"$m$={row['m']}, $\\varepsilon$={row['eps']:g}, $B$={row['B']:g}", fontsize=8)
        if col == 0:
            ax.set_ylabel("density $f(t)$")
            ax.legend(fontsize=6.5, frameon=False, loc="upper right")
        axz = axes[1, col]
        axz.plot(t, pl["z"], "o", ms=1.6, color="0.25")
        for yv in (-4, 4):
            axz.axhline(yv, color=core.OI_VERMILLION, lw=0.7, ls="--")
        axz.axhline(0, color="0.6", lw=0.5)
        axz.set_ylim(-5, 5)
        axz.set_xlabel("$t$")
        if col == 0:
            axz.set_ylabel("$z$ per bin")
        axz.text(0.02, 0.05, f"max$|z|$={row['max_abs_z']:.2f}, $\\chi^2_{{cov}}$/dof={row['chi2_cov_over_dof']:.2f}",
                 transform=axz.transAxes, fontsize=6.5)
    core.stream_tag(fig, "fb N0")
    written += core.save_figure(fig, fk.FIGURES / "fb_n0_fk_validation")
    plt.close(fig)

    keys = [k for k in bops if bops[k].get("fk") and bops[k]["fk"].get("B_p")]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(1.75 * len(keys) + 0.4, 2.3),
                                 layout="constrained", squeeze=False)
        for ax, k in zip(axes[0], keys):
            b = bops[k]
            fine = b["fk"]["fine"]
            Bs = np.asarray(fine["budgets"]); r = np.asarray(fine["r"])
            rj = np.asarray(fine["r_jack"])
            G = rj.shape[0]
            se = np.sqrt((G - 1) / G * np.sum((rj - rj.mean(0)) ** 2, axis=0))
            ax.fill_between(Bs, r - 2 * se, r + 2 * se, color=core.OI_BLUE, alpha=0.3, lw=0)
            ax.plot(Bs, r, color=core.OI_BLUE, lw=1.0, label="FK exact law")
            ax.axhline(0.05, color="0.4", lw=0.7, ls=":")
            br = b["reference"].get("stored_bracket")
            if br:
                ax.axvspan(br[0], br[1], color=core.OI_ORANGE, alpha=0.35, lw=0,
                           label="stored W2 bracket")
            ax.axvline(b["fk"]["B_p"], color=core.OI_BLUE, lw=0.8, ls="--")
            mm, ee = k.split("_eps")
            ax.set_title(f"$m$={mm[1:]}, $\\varepsilon$={ee}", fontsize=7.5)
            ax.set_xlabel("$B$")
            ax.tick_params(labelsize=6.5)
        axes[0, 0].set_ylabel("last-mode rel. prominence")
        axes[0, 0].legend(fontsize=6, frameon=False)
        core.stream_tag(fig, "fb N0")
        written += core.save_figure(fig, fk.FIGURES / "fb_n0_prominence_floor")
        plt.close(fig)
    return written


def run(groups: int = fk.DEFAULT_GROUPS) -> dict:
    t0 = time.time()
    ens = {k: fk.load_ensemble(v) for k, v in ENSEMBLES.items() if fk.index_path(v).exists()}
    out = {"schema": "n0_validation_v1", "item": "N0",
           "plan": "notes/gap_diagnosis_20260923/GAP_CLOSURE_PLAN.md section 3 N0",
           "driver": "code/exact_m_prr_fk_exact_law.py", "validation_script": "code/" + HERE.name,
           "estimator": ("P(T_R in bin k) = E[exp(-B X(e_k)) - exp(-B X(e_{k+1}))] over unkilled "
                         "Euler-Maruyama paths; exact discrete-time law; iid per-path SE"),
           "z_definition": ("z = (p_FK - p_DK) / sqrt(SE_FK^2 + p_FK (1 - p_FK) / N_DK); "
                            "basin z analogous; acceptance |z_bin| < 4 all bins, |z_basin| <= 2"),
           "groups_for_batch_means_and_jackknife": groups, "ensembles": {}}
    for (m, eps), e in ens.items():
        out["ensembles"][f"m{m}_eps{eps:g}"] = {
            "name": e.name, "index": str(fk.index_path(e.name).relative_to(fk.REPORT)),
            "paths": e.n_paths, "seed": e.index["seed"], "tag": e.index["tag"],
            "replicate": e.index["replicate"], "seed_entropy": e.index["seed_entropy"],
            "variants": list(e.variants), "complete": e.index["complete"],
            "process_seconds_total": e.index["process_seconds_total"],
            "chunk_runtimes_s": [c["runtime_seconds"] for c in e.chunks],
            "chunk_bytes": [c["bytes"] for c in e.chunks]}
    # G valley reproduction
    vcheck = {}
    b0 = json.loads((UPG / "w2_b0_empirical" / "B0_empirical.json").read_text())
    for c in b0["chains"]:
        spec = fk.EnsembleSpec(m=c["m"], eps=c["eps"])
        v = fk.g_valley_times(spec, [1.0 / c["m"]] * c["m"])
        vcheck[f"m{c['m']}_eps{c['eps']:g}"] = {"stored": c["basin_edge_last_g_valley_time"],
                                                "recomputed_last": v[-1] if v else None}
    out["g_valley_reproduction"] = vcheck
    # primary
    prim = []
    for label, path in PRIMARY:
        ref = read_reference(path)
        e = ens[(ref["m"], ref["eps"])]
        row = compare_cell(e, ref, groups)
        row["label"] = label
        prim.append(row)
        print(f"[N0] {label}: max|z|={row['max_abs_z']:.2f} merged max|z|={row['merged_max_abs_z']:.2f} "
              f"chi2cov/dof={row['chi2_cov_over_dof']:.3f} "
              f"basin z={np.round(row['basin_z'], 2).tolist()}", flush=True)
    out["primary"] = [{k: v for k, v in r.items() if k != "_plot"} for r in prim]
    # secondary (grouped by ensemble so decompressed chunks can stay cached)
    sec = []
    refs_sorted = []
    for kind, path in secondary_references():
        try:
            rr = read_reference(path)
            refs_sorted.append(((rr["m"], rr["eps"]), kind, path))
        except (KeyError, ValueError):
            refs_sorted.append(((0, 0.0), kind, path))
    refs_sorted.sort(key=lambda x: (x[0][0], x[0][1], str(x[2])))
    current = None
    for key0, kind, path in refs_sorted:
        if key0 in ens and current != key0:
            if current is not None and current in ens:
                ens[current].cache_in_memory = False
                ens[current].clear_cache()
            ens[key0].cache_in_memory = True
            current = key0
        try:
            ref = read_reference(path)
        except (KeyError, ValueError) as exc:
            sec.append({"reference": str(path.relative_to(fk.REPORT)), "skipped": str(exc)})
            continue
        key = (ref["m"], ref["eps"])
        if key not in ens or abs(ref["dt"] - 1e-3) > 1e-15 or abs(ref["tmax"] - 4.0) > 1e-12:
            continue
        if any(str(path) == str(p) for _, p in PRIMARY):
            continue
        row = compare_cell(ens[key], ref, groups)
        row["kind"] = kind
        row.pop("_plot")
        sec.append(row)
        print(f"[N0] {kind} {path.name}: max|z|={row['max_abs_z']:.2f} "
              f"merged={row['merged_max_abs_z']:.2f} chi2cov/dof={row['chi2_cov_over_dof']:.3f} "
              f"basin within 2SE={row['basin_within_2se']}",
              flush=True)
    for e in ens.values():
        e.cache_in_memory = False
        e.clear_cache()
    out["secondary"] = sec
    ok_sec = [r for r in sec if "max_abs_z" in r]
    out["secondary_summary"] = {
        "n_cells": len(ok_sec),
        "n_cells_all_bins_abs_z_lt_4": int(sum(r["max_abs_z"] < 4.0 for r in ok_sec)),
        "n_bins_total": int(sum(r["n_bins"] for r in ok_sec)),
        "n_bins_abs_z_ge_4_total": int(sum(r["n_bins_abs_z_ge_4"] for r in ok_sec)),
        "n_bins_abs_z_ge_3_total": int(sum(r["n_bins_abs_z_ge_3"] for r in ok_sec)),
        "n_cells_merged_bins_abs_z_lt_4": int(sum(r["merged_max_abs_z"] < 4.0 for r in ok_sec)),
        "n_merged_bins_total": int(sum(r["n_merged_bins"] for r in ok_sec)),
        "n_merged_bins_abs_z_ge_4_total": int(sum(r["merged_n_abs_z_ge_4"] for r in ok_sec)),
        "n_merged_bins_abs_z_ge_3_total": int(sum(r["merged_n_abs_z_ge_3"] for r in ok_sec)),
        "pooled_chi2": float(sum(r["chi2_cov_merged"] for r in ok_sec)),
        "pooled_dof": int(sum(r["chi2_cov_dof"] for r in ok_sec)),
        "max_abs_window_fraction_z": float(max(abs(r["window_fraction_z"]) for r in ok_sec)),
        "max_abs_kill_fraction_z": float(max(abs(r["kill_fraction_z"]) for r in ok_sec)),
        "n_cells_basin_within_2se": int(sum(r["basin_within_2se"] for r in ok_sec)),
        "n_basin_masses": int(sum(len(r["basin_z"]) for r in ok_sec)),
        "n_basin_abs_z_gt_2": int(sum(sum(abs(z) > 2 for z in r["basin_z"]) for r in ok_sec)),
        "max_abs_basin_z": float(max(max(abs(z) for z in r["basin_z"]) for r in ok_sec)),
        "n_mode_count_agree": int(sum(r["fk_expected_classifier_mode_count_at_dk_walkers"]
                                      == r["stored_mode_count"] for r in ok_sec)),
        "note": ("secondary cells are not independent of each other (common FK paths per (m, eps)); "
                 "note in particular that the bins of one direct-kill histogram are compared with "
                 "one FK ensemble reweighted to many (B, w)"),
    }
    out["secondary_summary"]["pooled_chi2_p_value_wilson_hilferty"] = _chi2_sf(
        out["secondary_summary"]["pooled_chi2"], out["secondary_summary"]["pooled_dof"])
    # kernel/geometry variants against their own direct-kill data
    vchk = []
    for label, path, ename, variant in VARIANT_CHECKS:
        if not fk.index_path(ename).exists() or not path.is_file():
            vchk.append({"label": label, "skipped": "ensemble or reference missing"})
            continue
        e = fk.load_ensemble(ename)
        e.cache_in_memory = True
        ref = read_reference(path, model=e.spec.model())
        row = compare_cell(e, ref, groups, variant=variant)
        row["label"] = label
        row.pop("_plot")
        vchk.append(row)
        e.clear_cache()
        print(f"[N0] variant {label}: max|z|={row['max_abs_z']:.2f} merged={row['merged_max_abs_z']:.2f} "
              f"chi2cov/dof={row['chi2_cov_over_dof']:.3f} basin z={np.round(row['basin_z'], 2).tolist()}",
              flush=True)
    out["variant_checks"] = vchk
    # kernel normalisation: sampled free-exposure clock vs the exact EM-chain G_n
    gchk = {}
    for (m, eps), e in ens.items():
        spec = e.spec
        T = fk.step_times(spec.dt, spec.steps())
        w = [1.0 / m] * m
        qc = core.free_exposure_general(T, centres_z=spec.centres(), eps=eps, weights=tuple(w),
                                        p=spec.model(), n_perp=spec.n_perp)
        for v in [x for x in ("full", "meancontact", "nogate") if x in e.variants]:
            gate = {"full": "contact", "meancontact": "mean", "nogate": "none"}[v]
            qd = fk.free_exposure_discrete(spec, w, gate=gate)
            mom = fk.exposure_moments(e, w, variant=v, groups=groups)
            gc_step = qc["g"] if v != "nogate" else qc["g"] / np.maximum(qc["contact_factor"], 1e-300)
            rng_ = mom["bin_step_ranges"]
            gd = np.array([qd["G"][a - 1:b].mean() for a, b in rng_])
            gc = np.array([gc_step[a - 1:b].mean() for a, b in rng_])
            gs, se = mom["G_sampled_bins"], mom["G_sampled_bins_se"]
            sel = (gd > 1e-3 * gd.max()) & (se > 0)
            z = (gs[sel] - gd[sel]) / se[sel]
            zc = (gs[sel] - gc[sel]) / se[sel]
            key = f"m{m}_eps{eps:g}_{v}"
            gchk[key] = {
                "n_bins": int(sel.sum()), "max_abs_z_vs_em_exact": float(np.abs(z).max()),
                "mean_z_vs_em_exact": float(z.mean()),
                "chi2_diag_over_dof_vs_em_exact": float(np.sum(z * z) / z.size),
                "max_abs_z_vs_continuum_G": float(np.abs(zc).max()),
                "chi2_diag_over_dof_vs_continuum_G": float(np.sum(zc * zc) / zc.size),
                "max_rel_diff_em_vs_continuum_G_selected_bins": float(
                    np.max(np.abs(gd[sel] - gc[sel]) / gc[sel])),
                "Lambda_tmax_sampled": float(np.sum(gs * np.diff(e.edges))),
                "Lambda_tmax_em_exact": float(np.sum(gd * np.diff(e.edges))),
                "Lambda_tmax_continuum": float(np.sum(gc * np.diff(e.edges))),
                "note": ("G_sampled = bin-averaged E[V] (unit budget) from the FK paths; em_exact = "
                         "free_exposure_discrete (exact Gaussian law of the Euler-Maruyama chain); "
                         "continuum = core.free_exposure_general (continuous-time OU); their gap is "
                         "the O(dt) EM bias, not an estimator error")}
            print(f"[N0] G check {key}: max|z| vs EM-exact={gchk[key]['max_abs_z_vs_em_exact']:.2f}, "
                  f"vs continuum={gchk[key]['max_abs_z_vs_continuum_G']:.2f}", flush=True)
    out["G_sampled_vs_quadrature"] = gchk
    out["G_check_note"] = ("bins of one FK ensemble are positively correlated (a path feeds several "
                           "bins), so the diagonal chi2 of the G check is not calibrated; use max|z| "
                           "and the per-slab totals in replicate_checks")
    # independent replicates (replicate index 1, 2 of tag 80)
    rep = {}
    slab_rows = []
    for name in ["n0_m2_eps0.1", "n0_m3_eps0.1", "n0_m2_eps0.05", "n0chk_m2_eps0.1_rep1"]:
        if not fk.index_path(name).exists():
            continue
        e = fk.load_ensemble(name)
        if "nogate" not in e.variants:
            continue
        m = e.spec.centres().size
        s1 = np.zeros(m); s2 = np.zeros(m); s3 = np.zeros(m)
        for _, dX, _ in e.iter_chunks("nogate"):
            Xj = dX.astype(np.float64).sum(axis=2)
            s1 += Xj.sum(0); s2 += (Xj * Xj).sum(0); s3 += (Xj**3).sum(0)
        N = e.n_paths
        mean = s1 / N
        var = s2 / N - mean**2
        skew = (s3 / N - 3 * mean * var - mean**3) / var**1.5
        ex = np.array([fk.free_exposure_discrete(e.spec, np.eye(m)[j], gate="none")["G"].sum()
                       * e.spec.dt for j in range(m)])
        slab_rows.append({"ensemble": name, "paths": N, "replicate": e.index["replicate"],
                          "slab_exposure_sampled": mean.tolist(), "slab_exposure_em_exact": ex.tolist(),
                          "z": ((mean - ex) / np.sqrt(var / N)).tolist(),
                          "relative_difference": ((mean - ex) / ex).tolist(),
                          "per_path_skewness": skew.tolist()})
    rep["slab_exposure_vs_em_exact"] = slab_rows
    rep["slab_exposure_note"] = ("unit-weight passage exposures X_j(t_max) (gate removed) vs the exact "
                                 "Gaussian EM-chain expectation; per-path X_j is right-skewed, so "
                                 "moderate-N sample means scatter asymmetrically (more often low)")
    if fk.index_path("n0chk_m2_eps0.1_rep2_p1").exists():
        e2 = fk.load_ensemble("n0chk_m2_eps0.1_rep2_p1")
        e1 = ens[(2, 0.1)]
        ref = read_reference(UPG / "w6_single_particle" / "m2_eps0.1_B1_p1.json")
        r2 = compare_cell(e2, ref, groups, variant="p1")
        r2.pop("_plot")
        a = fk.exact_law(e1, 1.0, (0.5, 0.5), variant="p1", with_cov=True)
        b = fk.exact_law(e2, 1.0, (0.5, 0.5), variant="p1", with_cov=True)
        sel = a["bin_mass"] >= 1e-6
        c2, d2 = _cov_chi2((a["bin_mass"] - b["bin_mass"])[sel], (a["cov"] + b["cov"])[np.ix_(sel, sel)])
        rep["w6_p1_m2_eps0.1_replicate2"] = {
            "ensemble": e2.name, "paths": e2.n_paths,
            "vs_w6_direct_kill": {k: r2[k] for k in ("max_abs_z", "merged_max_abs_z", "chi2_cov_merged",
                                                     "chi2_cov_dof", "chi2_cov_p_value_wilson_hilferty",
                                                     "basin_z", "window_fraction_z", "kill_fraction_z")},
            "fk_n0_vs_fk_replicate2_chi2_cov": c2, "dof": d2,
            "p_value_wilson_hilferty": _chi2_sf(c2, d2),
            "note": ("two independent FK path sets agree with each other; both show a mild excess "
                     "chi2 against the single W6 p1 direct-kill histogram they share, spread over the "
                     "whole window without a localized residual")}
    out["replicate_checks"] = rep
    # B_op / prominence-floor crossings
    bops = {}
    for m, eps, lo, hi in BOP_CHAINS:
        if (m, eps) not in ens:
            continue
        ref = bop_reference(m, eps)
        e = ens[(m, eps)]
        res = fk.prominence_floor_crossing(e, None, p=0.05, B_lo=lo, B_hi=hi,
                                           after_time=ref["basin_edge_last_g_valley_time"],
                                           walkers=1_000_000, groups=groups)
        row = {"reference": ref, "fk": res}
        if res.get("B_p") is not None and ref.get("stored_bracket"):
            br = ref["stored_bracket"]
            row["fk_inside_stored_bracket"] = bool(br[0] <= res["B_p"] <= br[1])
            row["fk_minus_stored_b0"] = res["B_p"] - ref["stored_b0"]
            row["fk_minus_stored_b0_over_bracket_halfwidth"] = (
                (res["B_p"] - ref["stored_b0"]) / (0.5 * (br[1] - br[0])))
            if "s2_pooled_regression_crossing" in ref:
                d = res["B_p"] - ref["s2_pooled_regression_crossing"]
                row["fk_minus_s2_pooled"] = d
                row["fk_minus_s2_pooled_z"] = d / math.sqrt(res["jackknife_se"] ** 2
                                                            + ref["s2_pooled_regression_se"] ** 2)
                row["fk_minus_s2_floor_mean"] = res["B_p"] - ref["s2_floor_interpolated_mean"]
        sc = scout_crossing(m, eps, ref["basin_edge_last_g_valley_time"])
        if sc:
            pc = fk.prominence_curve(e, None, sc["budgets"],
                                     after_time=ref["basin_edge_last_g_valley_time"],
                                     walkers=1_000_000, groups=groups)
            sc["n0_on_scout_budget_grid_crossing_loglinear"] = fk._crossing(
                np.asarray(sc["budgets"]), pc["r"], 0.05)[0]
            sc["note"] = ("like-for-like: both crossings log-linearly interpolated on the scout's "
                          "coarse budget grid; the difference is between independent ensembles")
            row["scout_cache_crossing"] = sc
        bops[f"m{m}_eps{eps:g}"] = row
        print(f"[N0] B_p m={m} eps={eps}: FK {res.get('B_p')} +- {res.get('jackknife_se')} ; "
              f"stored {ref.get('stored_b0')} {ref.get('stored_bracket')}", flush=True)
    out["prominence_floor_crossings"] = bops
    # scout cross-check
    xs = {}
    for (m, eps), e in ens.items():
        r = scout_crosscheck(e, m, eps, groups)
        if r:
            xs[f"m{m}_eps{eps:g}"] = r
    out["scout_cache_crosscheck_B1"] = xs
    # self-test record
    st = fk.FB_DATA / "n0_selftest.json"
    if st.exists():
        out["selftest"] = json.loads(st.read_text())
        out["selftest_source"] = str(st.relative_to(fk.REPORT))
    # acceptance
    acc = {
        "all_primary_bins_abs_z_lt_4": bool(all(r["max_abs_z"] < 4.0 for r in prim)),
        "all_primary_basins_within_2se": bool(all(r["basin_within_2se"] for r in prim)),
    }
    for k in ("m2_eps0.05", "m3_eps0.05"):
        if k in bops and bops[k]["fk"].get("B_p") is not None:
            b = bops[k]
            se = b["fk"]["jackknife_se"]
            br = b["reference"]["stored_bracket"]
            acc[f"bop_{k}_reproduced_within_resolution"] = bool(
                br[0] - 2 * se <= b["fk"]["B_p"] <= br[1] + 2 * se)
    acc["passed"] = bool(all(acc.values()))
    out["acceptance"] = acc
    out["wall_seconds"] = time.time() - t0
    out["figures"] = make_figures(prim, bops)
    fk.FB_DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fk._jsonable(out), indent=1))
    print(json.dumps(acc, indent=1))
    return out


if __name__ == "__main__":
    run()
