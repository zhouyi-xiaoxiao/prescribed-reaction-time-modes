#!/usr/bin/env python3
"""Regenerate the main-text data figures (Figs. 2-7) of the CNSNS paper from recorded data.

Figures are drawn at their final printed size (elsarticle preprint 12pt text width,
390 pt = 5.40 in) with every text element at 8 pt, Okabe-Ito colours, panel letters
and no parameter titles or stream tags.  Nothing is simulated here: every curve and
marker is read from the recorded JSON summaries in
``artifacts/data/exact_m_fixed_budget/`` (and, for Fig. 5, from the stored
reduced Feynman-Kac accumulators that those summaries index).

Basin convention (paper decision 2): basin masses are shown on WHOLE-AXIS basins,
cuts [0, s_1, ..., s_{m-1}, t_max = 4] with interior cuts where the mean path is
midway between adjacent stripe centres, snapped to the 0.02 grid.  Fig. 4 reads
``basins_extended`` from the N1 JSON; Fig. 5(b,c) recompute the whole-axis masses from
the stored accumulators (the N3 JSON records window-restricted basins only) and check
the recomputation against the recorded window values first.

Outputs (PNG + PDF, via core.save_figure) in artifacts/figures/:
  fb_paper_fig2_allocation_law     Fig. 2  allocation law (limit law only)
  fb_paper_fig3_design_densities   Fig. 3  equal weights vs max-min design
  fb_paper_fig4_design_masses      Fig. 4  design masses, whole-axis basins
  fb_paper_fig5_contact            Fig. 5  contact / frozen-gate factorial (a = 0.4, 0.2)
  fb_paper_fig6_visibility         Fig. 6  visibility thresholds B_p (smoothed only)
  fb_paper_fig7_fold               Fig. 7  finite-width fold; (c) bifurcation diagram of the limit
                                           law f_det (added 2026-09-23, referee finding A36)
SM figures (--sm), drawn for width=0.8\textwidth of the SM (seed repeat: 0.40\textwidth
minipage), no campaign tags, no internal wording:
  fb_paper_sm_fig5_contact_radii          Fig. 5(a,b) with a = 0.4, 0.2, 0.15
  fb_paper_sm_fig6_visibility_unsmoothed  Fig. 6 plus unsmoothed B_p and the seed bracket
  fb_paper_sm_n0_fk_validation            exact law vs direct kill (stored N0 ensembles)
  fb_paper_sm_n2_residual_decomposition   f - f_1 decomposition (stored N0 ensembles)
  fb_paper_sm_single_particle_control{,_p2}  single-particle field (W6 cell JSONs)
  fb_paper_sm_w2_seed_repeat              seed repeats (w2_seed_repeat_summary.json)
  fb_paper_sm_fb_n1_pstar, fb_paper_sm_fb_n4_preparation  untagged re-runs (see sm_untagged)
--copy-to copies every PDF byte-identically; the SM redraws of figures the SM already
includes are copied under the included name (SM_COPY_AS), e.g.
fb_paper_sm_n0_fk_validation.pdf -> fb_n0_fk_validation.pdf.
A provenance/check record is written to
artifacts/figures/fb_paper_figures_manifest.json.

Usage:  python3 fb_make_paper_figures.py [--only 2,5] [--sm [--sm-figs n0,n2]] [--copy-to DIR]
(the manifest is written only for a full run: all figures plus all of --sm)
Deterministic: fixed SOURCE_DATE_EPOCH (PDF dates) and no random numbers.
"""
from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1758585600")  # fixed PDF metadata date

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exact_m_prr_upgrade_core as core  # noqa: E402
import exact_m_prr_fk_exact_law as fk  # noqa: E402
import fb_allocation_law as al  # noqa: E402

DATA = core.REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
ROBUST = core.REPORT / "artifacts" / "data" / "exact_m_prr_upgrade" / "robustness"
MANIFEST = core.FIGURES / "fb_paper_figures_manifest.json"

TEXT_WIDTH_IN = 390.0 / 72.27          # elsarticle preprint 12pt \textwidth
SM_TEXT_WIDTH_PT = 483.69687           # SM: article 10pt, A4, 2 cm margins (supplement_cnsns.log)
SM_WIDTH_IN = 0.8 * SM_TEXT_WIDTH_PT / 72.27   # SM figures are drawn for width=0.8\textwidth
SM_MINIPAGE_IN = 0.40 * SM_TEXT_WIDTH_PT / 72.27  # seed-repeat figure sits in a 0.40\textwidth minipage
FS = 8.0                               # every text element >= 8 pt at print size

PAPER_RC = dict(core.PRR_RC)
PAPER_RC.update({
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "legend.title_fontsize": FS, "figure.titlesize": FS,
    "mathtext.fontset": "dejavusans", "font.family": "DejaVu Sans",
    "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25, "legend.columnspacing": 0.9,
    "axes.titlelocation": "left", "axes.titlepad": 3.0,
    "savefig.dpi": 300, "pdf.compression": 9,
})

C_EQUAL, C_MAXMIN, C_SHAPE = core.OI_ORANGE, core.OI_BLUE, core.OI_GREEN
CHECKS: dict = {}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update(PAPER_RC)
    return plt


def _load(rel: str) -> dict:
    return json.loads((DATA / rel).read_text())


def _save(fig, stem: str) -> list[str]:
    import matplotlib.pyplot as plt
    out = core.save_figure(fig, core.FIGURES / stem)
    plt.close(fig)
    return out


def _check(name: str, value, expected, tol: float) -> None:
    ok = abs(float(value) - float(expected)) <= tol
    CHECKS[name] = {"value": float(value), "expected": float(expected), "tol": tol, "pass": ok}
    if not ok:
        raise SystemExit(f"check failed: {name}: {value} vs {expected} (tol {tol})")


# ============================================================================
# Fig. 2 -- allocation law (limit law only; no basin convention involved)
# src: artifacts/data/exact_m_fixed_budget/allocation_law.json
# ============================================================================


def fig2_allocation() -> list[str]:
    plt = _plt()
    P = _load("allocation_law.json")
    curve = P["p_star_curve"]
    b = curve["budgets"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.05), layout="constrained")
    ax = axes[0]
    for key, col, m, lab in (("m2", core.OI_BLUE, 2, "$m=2$"), ("m3", core.OI_VERMILLION, 3, "$m=3$"),
                             ("m5_z0_8", core.OI_GREEN, 5, "$m=5$")):
        ax.plot(b, [m * y for y in curve[key]], color=col, label=lab)
    ax.plot(b, [3 * y for y in curve["m3_equal_weight_min_mass"]], color=core.OI_VERMILLION, ls="--",
            lw=1.0)
    beq = np.asarray(b); yeq = 3 * np.asarray(curve["m3_equal_weight_min_mass"])
    i4 = int(np.argmin(np.abs(beq - 4.0)))
    ax.annotate("equal\nweights", xy=(beq[i4], yeq[i4]), xytext=(18.0, 0.55), color=core.OI_VERMILLION,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color=core.OI_VERMILLION, lw=0.6))
    ax.set_xscale("log")
    ax.set_xticks([1e-2, 1, 1e2])
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("$m\\,p^*(B,m)$")
    ax.set_ylim(0, 1.03)
    ax.legend(loc="upper left", frameon=False, handlelength=1.0)
    ax.set_title("(a)")
    ax = axes[1]
    wopt = curve["m3_w_opt"]
    for j, col in enumerate((core.OI_BLUE, core.OI_ORANGE, core.OI_VERMILLION)):
        ax.plot(b, [w[j] for w in wopt], color=col, label=f"$w_{j + 1}^*$")
    ax.axhline(1 / 3, color=core.OI_GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([1e-2, 1, 1e2])
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("max–min weights ($m=3$)")
    ax.set_ylim(0, 1)
    # R2 fix editor (A13, 2026-09-23): the stacked upper-left legend ran into the w1* curve
    # (about 0.63 at small B). The region B in [10, 100], weight in [0.35, 0.65] is free of
    # curves, but the default-width legend box is wider than it (an opaque framed box there
    # hid the rising w3* curve, and a one-row legend along the top ran into w3*), so the
    # legend is placed at centre right, raised clear of the dotted 1/3 line, with short
    # handles and no frame.
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), frameon=False, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("(b)")
    ax = axes[2]
    geo = P["geometries"]["m2"]
    v, area = geo["speeds_v"], geo["area_factor_W_pow_d_minus_1"]
    # B in {0.5, 1, 8}: at B = 4 the curve and the max-min square coincide with those of
    # B = 8 on this scale, and at B = 2 the equal-weight point and both squares overlap
    # (final fix pass, 2026-09-23; referee finding A23)
    rows = {float(r["budget"]): r for r in P["reachable_hypersurface"]["m2"]}
    for row, col in zip([rows[0.5], rows[1.0], rows[8.0]],
                        (core.OI_BLUE, core.OI_ORANGE, core.OI_VERMILLION)):
        m1 = [r["M1"] for r in row["samples"]]
        m2 = [r["M2"] for r in row["samples"]]
        ax.plot(m1, m2, color=col, label=f"$B={row['budget']:g}$")
        eq = al.masses(row["budget"], [0.5, 0.5], v, area)
        ax.plot([eq[0]], [eq[1]], marker="o", ms=3.5, color=col, ls="none")
        ps = al.p_star(row["budget"], v, area)["p_star"]
        ax.plot([ps], [ps], marker="s", ms=3.5, color=col, mfc="white", ls="none")
    ax.plot([0, 0.5], [0, 0.5], color=core.OI_GREY, lw=0.7, ls=":")
    ax.set_xlabel("$M_1$")
    ax.set_ylabel("$M_2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    # markers (explained in the caption): filled circle = equal weights, open square = max-min optimum
    ax.legend(loc="upper right", frameon=False, handlelength=1.2)
    ax.set_title("(c)")
    # consistency with the recorded optimum (m = 3, B = 1)
    chk = P["checks"]["gpt6_m3_B1"]
    i1 = int(np.argmin(np.abs(np.asarray(b) - 1.0)))
    if abs(b[i1] - 1.0) < 1e-12:
        _check("fig2_m3_pstar_B1", curve["m3"][i1], chk["p_star"], 1e-9)
    return _save(fig, "fb_paper_fig2_allocation_law")


# ============================================================================
# Fig. 3 -- equal weights vs max-min design (densities; convention-free)
# src: N1/n1_curves.json#curves, #directkill; N1/n1_allocation_design.json#cells
# ============================================================================


def fig3_densities() -> list[str]:
    plt = _plt()
    d = _load("N1/n1_allocation_design.json")
    pl = _load("N1/n1_curves.json")
    spec_times = {2: None, 3: None}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.25), layout="constrained")
    col = {"equal": C_EQUAL, "maxmin": C_MAXMIN}
    lab = {"equal": "equal weights", "maxmin": "max–min design"}
    for ax, (m, eps, B), letter in zip(axes, [(2, 0.1, 8.0), (3, 0.1, 4.0)], "ab"):
        for a in ("equal", "maxmin"):
            c = pl["curves"][f"m{m}_eps{eps:g}__B{B:g}_{a}"]
            t = np.asarray(c["t"]); f = np.asarray(c["density"]); s = np.asarray(c["density_se_batch"])
            row = next(r for r in d["cells"] if r["m"] == m and r["eps"] == eps and r["B"] == B
                       and r["allocation"] == a)
            n = row["protocol_mode_count_1e6"]
            ax.fill_between(t, f - 2 * s, f + 2 * s, color=col[a], alpha=0.3, lw=0)
            ax.plot(t, f, color=col[a], lw=1.2, label=f"{lab[a]} ({n} {'mode' if n == 1 else 'modes'})")
            CHECKS[f"fig3_m{m}_{a}_w"] = row["w"]
            CHECKS[f"fig3_m{m}_{a}_modes"] = n
        dkp = pl["directkill"][f"m{m}_eps{eps:g}_B{B:g}"]
        ax.plot(dkp["t"], dkp["dk_density"], ".", ms=2.2, color="k", label="direct simulation")
        times = _target_times(m)
        spec_times[m] = times
        for tt in times:
            ax.axvline(tt, color="0.55", lw=0.7, ls=":")
        ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=True, framealpha=1.0, edgecolor="none", loc="upper right")
        ax.set_title(f"({letter}) $m={m}$, $B={B:g}$")
    axes[0].set_ylabel("reaction-time density $f(t)$")
    CHECKS["fig3_target_times"] = spec_times
    return _save(fig, "fb_paper_fig3_design_densities")


def _target_times(m: int) -> list[float]:
    spec = fk.load_ensemble(f"n1_m{m}_eps0.1").spec
    return [float(x) for x in spec.times()]


# ============================================================================
# Fig. 4 -- design masses on whole-axis basins (no maxmin_mf markers; R1)
# src: N1/n1_allocation_design.json#cells[*].basins_extended.{mass,target_limit_law,ci95_batch}
# ============================================================================


def fig4_masses() -> list[str]:
    plt = _plt()
    from matplotlib.lines import Line2D
    d = _load("N1/n1_allocation_design.json")
    col = {"equal": C_EQUAL, "maxmin": C_MAXMIN, "shape": C_SHAPE}
    lab = {"equal": "equal weights", "maxmin": "max–min design", "shape": "target-shape design"}
    mk = {2: "o", 3: "s", 5: "^"}
    eps_list = (0.05, 0.1, 0.15)
    band = {0.05: 0.02, 0.1: 0.07}
    # acceptance values (whole-axis basins), m = 2, 3 families equal/maxmin/shape
    dev = {e: [] for e in eps_list}
    ci_half = []
    for c in d["cells"]:
        if c["allocation"] in col and c["m"] in (2, 3):
            dev[c["eps"]].append(c["basins_extended"]["max_abs_deviation"])
        if c["allocation"] in col:
            ci = np.asarray(c["basins_extended"]["ci95_batch"])
            ci_half.append(float(np.max(0.5 * (ci[:, 1] - ci[:, 0]))))
    acc = d["acceptance"]
    _check("fig4_eps0.05_max_abs_dev", max(dev[0.05]), acc["eps0.05_max_abs_dev_extended"]["value"], 1e-12)
    _check("fig4_eps0.1_max_abs_dev", max(dev[0.1]), acc["eps0.1_extended"]["worst"][1], 1e-12)
    CHECKS["fig4_eps0.05_n_within_0.02"] = int(sum(x <= 0.02 for x in dev[0.05]))
    CHECKS["fig4_eps0.1_n_within_0.07"] = int(sum(x <= 0.07 for x in dev[0.1]))
    CHECKS["fig4_eps0.15_max_abs_dev"] = max(dev[0.15])
    CHECKS["fig4_max_ci95_halfwidth"] = max(ci_half)
    if CHECKS["fig4_eps0.1_n_within_0.07"] != 30 or CHECKS["fig4_eps0.05_n_within_0.02"] != 30:
        raise SystemExit("fig4 acceptance counts differ from 30/30")

    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, 3.95),
                             gridspec_kw={"height_ratios": [1.45, 1.0]}, layout="constrained")
    ylim_r = {}
    for e in eps_list:
        devs = [abs(x) for c in d["cells"] if c["eps"] == e and c["allocation"] in col
                for x in c["basins_extended"]["deviation_from_target"]]
        ylim_r[e] = max(band.get(e, 0.0), max(devs)) * 1.18
    CHECKS["fig4_ylim_residual"] = ylim_r
    for ci, eps in enumerate(eps_list):
        ax, axr = axes[0, ci], axes[1, ci]
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.7)
        for c in d["cells"]:
            if c["eps"] != eps or c["allocation"] not in col:
                continue
            bx = c["basins_extended"]
            T = np.asarray(bx["target_limit_law"]); M = np.asarray(bx["mass"])
            kw = dict(ls="none", marker=mk[c["m"]], ms=3.4, mfc=col[c["allocation"]], mec="k",
                      mew=0.3, alpha=0.9)
            ax.plot(T, M, **kw)
            axr.plot(T, M - T, **kw)
        axr.axhline(0, color="0.5", lw=0.7)
        tol = band.get(eps)
        if tol:
            axr.axhspan(-tol, tol, color="0.87", lw=0, zorder=0)
        ax.set_title(f"({'abc'[ci]}) $\\varepsilon={eps:g}$")
        axr.set_title(f"({'def'[ci]})")
        for a_ in (ax, axr):
            a_.set_xlim(0, 1)
            a_.set_xticks([0, 0.5, 1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        axr.set_ylim(-ylim_r[eps], ylim_r[eps])
        ax.set_xticklabels([])
        if ci == 0:
            ax.set_ylabel("exact mass $M_j$")
            axr.set_ylabel("$M_j-M_j^{\\mathrm{lim}}$")
    axes[1, 1].set_xlabel("limit-law (target) basin mass $M_j^{\\mathrm{lim}}$")
    handles = []
    for a, m in zip(("equal", "maxmin", "shape"), (2, 3, 5)):   # ncol=3 fills column-wise
        handles.append(Line2D([], [], ls="", marker="o", ms=4, mfc=col[a], mec="k", mew=0.3, label=lab[a]))
        handles.append(Line2D([], [], ls="", marker=mk[m], ms=4, mfc="w", mec="k", label=f"$m={m}$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return _save(fig, "fb_paper_fig4_design_masses")


# ============================================================================
# Fig. 5 -- contact rule and frozen gate (N3)
# src: N3/n3_summary.json; N3/ensembles/n3_anchor_m2_eps0.1.json (stored accumulators);
#      fk_ensembles/n3_tangent_m2_eps{0.05,0.025,0.0125}.json
# ============================================================================


def _geometric_cut_snapped(spec) -> float:
    c = spec.centres()
    mid = 0.5 * (c[0] + c[1])
    s = -math.log((mid - spec.z_bar) / (spec.z0 - spec.z_bar)) / spec.gamma
    return round(s / 0.02) * 0.02


def _tangent_whole_axis(eps: float, rows_json: list, cut_window: list) -> dict:
    """Whole-axis late basin mass [s, t_max] from the stored declared-mode ensemble."""
    ens = fk.load_ensemble(f"n3_tangent_m2_eps{eps:g}")
    res = ens.declared()
    t = res["t"]
    s = _geometric_cut_snapped(ens.spec)
    nb = res["batch_p_step"].shape[1]
    out = {"cut": s, "B": [], "M2": [], "M2_se": [], "orthant": [], "mean_field": []}
    hi_w = cut_window[-1]
    for i, dcl in enumerate(res["declared"]):
        if dcl["variant"] != "full":
            continue
        B = float(dcl["B"])
        p = res["p_step"][i]
        pb = res["batch_p_step"][i]
        # validate against the recorded window-restricted late mass first
        wmask = (t >= cut_window[1]) & (t <= hi_w)
        rec = next(r for r in rows_json if r["variant"] == "full" and float(r["B"]) == B)
        _check(f"fig5c_eps{eps:g}_B{B:g}_window_M2_reproduced", p[wmask].sum(), rec["fk_masses"][1], 1e-12)
        mask = t >= s
        out["B"].append(B)
        out["M2"].append(float(p[mask].sum()))
        out["M2_se"].append(float(pb[:, mask].sum(1).std(ddof=1) / math.sqrt(nb)))
        out["orthant"].append(float(rec["orthant_law"][1]))
        mx = res["mean_X"][int(dcl["pair"])]
        prev = np.concatenate([[0.0], mx[:-1]])
        mfs = np.exp(-B * prev) * (-np.expm1(-B * (mx - prev)))
        out["mean_field"].append(float(mfs[mask].sum()))
    order = np.argsort(out["B"])
    for k in ("B", "M2", "M2_se", "orthant", "mean_field"):
        out[k] = [out[k][j] for j in order]
    return out


FIG5_COLS = {0.4: core.OI_BLUE, 0.2: core.OI_ORANGE, 0.15: core.OI_VERMILLION}
FIG5_MAIN_RADII = (0.4, 0.2)          # a = 0.15 is shown in the SM version only


def _fig5_density_panel(ax, n3, R, radii, legend_loc="upper right") -> None:
    """Exact densities at (m, eps, B) = (2, 0.1, 20): contact (solid) vs c_a(t) (dashed)."""
    from matplotlib.lines import Line2D
    Bp = 20.0
    for a in radii:
        lc = n3.red_law(R, n3.vname("contact", a, "mid"), Bp)
        lm = n3.red_law(R, n3.vname("mean", a, "mid"), Bp)
        ax.semilogy(lc["t"], np.maximum(lc["density"], 1e-6), color=FIG5_COLS[a], lw=1.2)
        ax.semilogy(lm["t"], np.maximum(lm["density"], 1e-6), color=FIG5_COLS[a], lw=1.0, ls="--")
    ln = n3.red_law(R, n3.vname("none", 0.4, "mid"), Bp)
    ax.semilogy(ln["t"], np.maximum(ln["density"], 1e-6), color="0.35", lw=1.0, ls=":")
    ax.set_ylim(1e-4, 3e2)
    ax.set_xlim(0.5, 3.5)
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e2])
    ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    ax.set_ylabel("reaction-time density $f(t)$")
    handles = [Line2D([], [], color=FIG5_COLS[a], lw=1.4, label=f"$a={a:g}$") for a in radii]
    handles.append(Line2D([], [], color="0.35", lw=1.0, ls=":", label="no contact rule"))
    ax.legend(handles=handles, loc=legend_loc, frameon=False, ncol=1)


def _fig5_mass_panel(ax, n3, S, R, radii, legend_loc="lower left") -> None:
    """Whole-axis late basin mass vs B, recomputed from the stored accumulators (checked)."""
    from matplotlib.lines import Line2D
    cells = S["anchors"]["m2"]["cells"]
    wcuts = S["anchors"]["m2"]["cuts"]
    s_cut = _geometric_cut_snapped(R["spec"])
    CHECKS["fig5b_whole_axis_cuts"] = [0.0, s_cut, float(R["edges"][-1])]
    for a in radii:
        for gate, mkr, ls in (("contact", "o", "-"), ("mean", "s", "--")):
            var = n3.vname(gate, a, "mid")
            rows = sorted([c for c in cells if c["variant"] == var], key=lambda c: c["B"])
            Bs, M2, se = [], [], []
            for c in rows:
                bw = n3.red_basins(R, var, c["B"], wcuts)
                _check(f"fig5b_{var}_B{c['B']:g}_window_M2_reproduced", bw["masses"][-1],
                       c["basin_masses"][-1], 1e-12)
                bx = n3.red_basins(R, var, c["B"], [float(R["edges"][0]), s_cut, float(R["edges"][-1])])
                Bs.append(c["B"]); M2.append(bx["masses"][-1]); se.append(bx["se_batch"][-1])
            M2 = np.asarray(M2); se = np.asarray(se)
            ax.errorbar(Bs, np.maximum(M2, 1e-7), yerr=2 * se, color=FIG5_COLS[a], marker=mkr, ms=3.2,
                        lw=1.0, ls=ls, mfc=FIG5_COLS[a] if gate == "contact" else "white")
            CHECKS[f"fig5b_{var}_whole_axis_M2"] = dict(zip([f"{b:g}" for b in Bs], M2.tolist()))
        rows = sorted([c for c in cells if c["variant"] == n3.vname("contact", a, "mid")],
                      key=lambda c: c["B"])
        ax.plot([c["B"] for c in rows], [c["frozen_surrogate_masses"][-1] for c in rows],
                color=FIG5_COLS[a], lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.5)
    ax.set_xticks([1, 1e2, 1e4])
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("late basin mass $M_2$")
    ax.legend(handles=[Line2D([], [], color="k", lw=1.0, ls=":", label="frozen-gate law")],
              loc=legend_loc, frameon=False)


def fig5_contact() -> list[str]:
    """Main Fig. 5: a = 0.4, 0.2 in (a, b); line-style code (solid = contact rule, dashed = c_a(t))
    is stated in the caption only; every panel carries its own legend inside the axes."""
    plt = _plt()
    import fb_n3_contact_factorial as n3
    S = _load("N3/n3_summary.json")
    R = n3.load_reduced(n3.anchor_name(2))
    fig, axs = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.15), layout="constrained")

    _fig5_density_panel(axs[0, 0], n3, R, FIG5_MAIN_RADII)
    axs[0, 0].set_title("(a)")
    _fig5_mass_panel(axs[0, 1], n3, S, R, FIG5_MAIN_RADII)
    axs[0, 1].set_title("(b)")

    # (c) boundary-tangent start: convergence in eps to the frozen-gate (orthant) law
    ax = axs[1, 0]
    tcols = {0.05: core.OI_SKY, 0.025: core.OI_BLUE, 0.0125: core.OI_PURPLE}
    tangent = {}
    for key, T in S["tangent"].items():
        eps = float(T["eps"])
        tangent[eps] = _tangent_whole_axis(eps, T["rows"], T["cuts"])
    sat = list(S["tangent"].values())[0]["orthant"]["saturation_late_mass"]
    for eps in sorted(tangent, reverse=True):
        tw = tangent[eps]
        ax.errorbar(tw["B"], tw["M2"], yerr=2 * np.asarray(tw["M2_se"]), color=tcols[eps], marker="o",
                    ms=3.0, lw=1.0, label=f"$\\varepsilon={eps:g}$")
        CHECKS[f"fig5c_eps{eps:g}_whole_axis"] = tw
    tw = tangent[0.025]
    ax.plot(tw["B"], tw["mean_field"], color="0.4", ls="--", lw=1.0, label="mean field")
    ax.plot(tw["B"], tw["orthant"], color="k", lw=1.3, label="frozen-gate law")
    ax.axhline(sat, color="k", ls=":", lw=0.8)
    ax.text(0.62, sat - 0.012, f"$\\pi_2={sat:.3f}$", va="top", ha="left")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("late basin mass $M_2$")
    ax.set_title("(c)")
    ax.legend(loc="upper right", frameon=False)

    # (d) late-mode time vs B (a = 0.4), three time steps
    ax = axs[1, 1]
    var04 = n3.vname("contact", 0.4, "mid")
    lms = sorted([x for x in S["anchors"]["m2"]["late_modes"] if x["variant"] == var04], key=lambda x: x["B"])
    ax.plot([x["B"] for x in lms], [x["late_argmax_smoothed_t"] for x in lms], "o-", color=core.OI_BLUE,
            ms=3.2, label="$\\Delta t=10^{-3}$")
    dmk = {"dt0.0005": ("s", core.OI_GREEN, "$\\Delta t=5\\times10^{-4}$"),
           "dt0.00025": ("^", core.OI_VERMILLION, "$\\Delta t=2.5\\times10^{-4}$")}
    for key in ("dt0.0005", "dt0.00025"):
        blk = S["dtcheck"][key]
        rows = sorted([r for r in blk["rows"] if r["variant"] == var04], key=lambda r: r["B"])
        mkr, cc, lab = dmk[key]
        ax.plot([r["B"] for r in rows], [r["late_argmax_smoothed_t"] for r in rows], mkr, color=cc, ms=4,
                mfc="none", label=lab)
    ax.axhline(2.5, color="0.5", ls=":", lw=0.8)
    ax.text(1.0, 2.49, "second passage $t_2$", va="top", ha="left", color="0.3")
    ax.set_ylim(1.7, 2.56)
    ax.set_xscale("log")
    ax.set_xticks([1, 1e2, 1e4])
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("late-peak time $t$ (units of $1/\\gamma$)")
    ax.set_title("(d)")
    ax.legend(loc="lower left", frameon=False)
    return _save(fig, "fb_paper_fig5_contact")


def sm_fig5_radii() -> list[str]:
    """SM version of Fig. 5(a,b) with all three contact radii a = 0.4, 0.2, 0.15."""
    plt = _plt()
    import fb_n3_contact_factorial as n3
    S = _load("N3/n3_summary.json")
    R = n3.load_reduced(n3.anchor_name(2))
    fig, axs = plt.subplots(1, 2, figsize=(SM_WIDTH_IN, 2.35), layout="constrained")
    _fig5_density_panel(axs[0], n3, R, n3.RADII)
    axs[0].set_title("(a)")
    _fig5_mass_panel(axs[1], n3, S, R, n3.RADII)
    axs[1].set_title("(b)")
    return _save(fig, "fb_paper_sm_fig5_contact_radii")


# ============================================================================
# Fig. 6 -- visibility thresholds (N6; tag-80 ensemble values)
# src: N6/n6_cell_m{2,3}_eps*.json#B_p; N6/n6_visibility_thresholds.json#eps_to_zero_limits;
#      exact_m_prr_upgrade/robustness/w2_seed_repeat_summary.json#cells[*].three_seed.envelope
# ============================================================================


def _fig6_draw(*, full: bool, stem: str, width: float) -> list[str]:
    """Visibility thresholds B_p.  full=False (main Fig. 6): smoothed thresholds with 95% CIs,
    censoring and eps->0 markers only.  full=True (SM): also the unsmoothed thresholds (open
    symbols, dotted) and the three-seed bracket at (m, eps) = (2, 0.05)."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    S = _load("N6/n6_visibility_thresholds.json")
    lim = S["eps_to_zero_limits"]
    seeds = json.loads((ROBUST / "w2_seed_repeat_summary.json").read_text())
    envelopes = {c["cell"]: c["three_seed"]["envelope"] for c in seeds["cells"]}
    eps_list = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25)
    p_levels = (0.01, 0.05, 0.10)
    b_hi = 64.0
    colors = {0.01: core.OI_BLUE, 0.05: core.OI_VERMILLION, 0.10: core.OI_GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.45 if not full else 2.75), layout="constrained",
                             sharey=True)
    for ax, m, letter in zip(axes, (2, 3), "ab"):
        for p in p_levels:
            pk = int(round(100 * p))
            xs, ys, lo, hi, yu, xc = [], [], [], [], [], []
            for eps in eps_list:
                path = DATA / "N6" / f"n6_cell_m{m}_eps{eps:g}.json"
                c = json.loads(path.read_text())
                v = c["B_p"].get(f"r_s_p{pk}", {})
                u = c["B_p"].get(f"r_u_p{pk}", {})
                xs.append(eps)
                if v.get("B_p") is not None:
                    ys.append(v["B_p"])
                    ci = v.get("bootstrap_ci95") or [v["B_p"], v["B_p"]]
                    lo.append(v["B_p"] - ci[0]); hi.append(ci[1] - v["B_p"])
                else:
                    ys.append(np.nan); lo.append(0.0); hi.append(0.0)
                    if v.get("status") == "censored":
                        xc.append(eps)
                yu.append(u["B_p"] if u.get("B_p") is not None else np.nan)
                if m == 2 and eps == 0.1 and pk == 5:
                    _check("fig6_m2_eps0.1_B5", v["B_p"], 7.569105981175878, 1e-9)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colors[p], marker="o", ms=3.2, lw=1.1, capsize=1.5)
            if full:
                ax.plot(xs, yu, color=colors[p], marker="o", mfc="white", ms=3.0, lw=0.8, ls=":")
            if xc:
                ax.plot(xc, [b_hi] * len(xc), ls="none", marker="^", ms=5, color=colors[p])
            lv = lim.get(f"m{m}_smoothed_p{pk}")
            if lv:
                ax.plot([0.0], [lv], marker="<", color=colors[p], ms=5.5, clip_on=False, ls="none")
        env = envelopes.get(f"m{m}_eps0.05") if m == 2 else None
        if env and full:
            ax.plot([0.05 - 0.009] * 2, env, color="k", lw=3.5, alpha=0.5, solid_capstyle="butt")
            CHECKS["fig6sm_seed_bracket_m2_eps0.05"] = env
        ax.axhline(b_hi, color="0.5", lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_xlim(-0.012, 0.262)
        ax.set_ylim(0.2, 100)
        ax.set_xticks([0, 0.1, 0.2])
        ax.set_xlabel("noise amplitude $\\varepsilon$")
        ax.set_title(f"({letter}) $m={m}$")
        ax.grid(alpha=0.3, which="major")
        ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30, 100]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.3", "1", "3", "10", "30", "100"]))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("visibility threshold $B_p$")
    h_p = [Line2D([], [], color=colors[p], marker="o", ms=3.2, lw=1.1, label=f"$p={100 * p:g}\\%$")
           for p in p_levels]
    h_mk = [Line2D([], [], color="k", marker="<", ms=5, ls="none", label="$\\varepsilon\\to0$ limit"),
            Line2D([], [], color="k", marker="^", ms=5, ls="none", label="no crossing up to $B=64$")]
    leg_kw = dict(frameon=True, framealpha=1.0, edgecolor="none", borderaxespad=0.4)
    if full:
        h_mk = [Line2D([], [], color="k", marker="o", ms=3.2, lw=1.1, label="smoothed (95% CI)"),
                Line2D([], [], color="k", marker="o", mfc="white", ms=3.0, lw=0.8, ls=":",
                       label="unsmoothed")] + h_mk + \
               [Line2D([], [], color="k", lw=3.5, alpha=0.5, label="seed-repeat bracket")]
        fig.legend(handles=h_p + h_mk, loc="outside lower center", ncol=4, frameon=False)
    else:
        # per-panel legends inside the axes, in regions free of data
        axes[1].legend(handles=h_p, loc="upper right", bbox_to_anchor=(1.0, 0.86), **leg_kw)
        axes[0].legend(handles=h_mk, loc="lower right", **leg_kw)
    return _save(fig, stem)


def fig6_visibility() -> list[str]:
    return _fig6_draw(full=False, stem="fb_paper_fig6_visibility", width=TEXT_WIDTH_IN)


def sm_fig6_unsmoothed() -> list[str]:
    return _fig6_draw(full=True, stem="fb_paper_sm_fig6_visibility_unsmoothed", width=SM_WIDTH_IN)


# ============================================================================
# Fig. 7 -- finite-width fold (N9b)
# src: N9b/n9b_fold.json
# ============================================================================


def fig7_fold() -> list[str]:
    plt = _plt()
    R = _load("N9b/n9b_fold.json")
    hsel = "0.03"
    cols = {"0.06": core.OI_BLUE, "0.03": core.OI_ORANGE, "0.02": core.OI_GREEN}
    # panels (a), (b) as before on the top row; the bifurcation diagram (c) of the limit law
    # spans the bottom row (final fix pass, 2026-09-23, referee finding A36)
    fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "c"]], figsize=(TEXT_WIDTH_IN, 4.15),
                                  layout="constrained", height_ratios=[1.0, 0.72])
    axes = [axd["a"], axd["b"], axd["c"]]
    ax = axes[0]
    for e in ("0.02", "0.03", "0.06"):
        r = R["eps"][e]
        Bs = np.array(r["B_grid"])
        D = np.array(r["by_h"][hsel]["D"])
        ax.plot(Bs, D, color=cols[e], lw=1.2, label=f"$\\varepsilon={e}$")
        bz = r["by_h"][hsel]["B_star"]
        if bz:
            ax.axvline(bz, color=cols[e], lw=0.7, ls=":")
    ax.axvline(R["deterministic_same_statistic_by_h"][hsel], color="k", lw=1.0, ls="--",
               label="deterministic limit")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("fold statistic $D_h$ (units of $\\gamma$)")
    ax.set_title("(a)")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    hmark = {"0.02": "o", "0.025": "s", "0.03": "^", "0.04": "D", "0.05": "v"}
    hcol = {"0.02": core.OI_BLUE, "0.025": core.OI_ORANGE, "0.03": core.OI_GREEN,
            "0.04": core.OI_VERMILLION, "0.05": core.OI_PURPLE}
    for k, mkr in hmark.items():
        xs, ys, es = [], [], []
        det = R["deterministic_same_statistic_by_h"][k]
        for e, r in R["eps"].items():
            v = r["by_h"].get(k)
            if v and v["B_star"] is not None:
                xs.append(float(e) ** 2 * 1e3)
                ys.append(v["B_star"] - det)
                es.append(1.96 * (v["jackknife_se"] or 0.0))
        ax.errorbar(xs, ys, yerr=es, marker=mkr, ms=3.2, lw=0, elinewidth=0.8, capsize=1.5,
                    color=hcol[k], label=f"$h={k}$")
        f = R["convergence_fits_eps2"][k]
        xx = np.linspace(0, 3.7, 50)
        ax.plot(xx, f["B0"] - det + f["c"] * xx * 1e-3, lw=0.8, ls=":", color=hcol[k])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("$\\varepsilon^2$ ($\\times 10^{-3}$)")
    ax.set_ylabel("fold offset $B^*_h-B^{\\mathrm{det}}_h$")
    ax.set_title("(b)")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    _check("fig7_Bstar_eps0.03_h0.03", R["eps"]["0.03"]["by_h"]["0.03"]["B_star"], 7.870510019733099, 1e-9)
    # (c) bifurcation diagram of the limit law f_det = B G0 exp(-B Lambda0) (final fix pass,
    # 2026-09-23; referee finding A36): its stationary points solve r0(t) = G0'/G0^2 = B
    # (Proposition 4); on the rising flank of the second bump of G0 the valley (r0' > 0) and
    # the late peak (r0' < 0) merge at the fold (b_2, t_f2).  Closed-form G0 of
    # code/fb_finite_width_det.py; src: artifacts/data/exact_m_fixed_budget/TH8/finite_width_det.json
    import fb_finite_width_det as fwd
    ax = axes[2]
    Dd = _load("TH8/finite_width_det.json")["scan"]["m2_rho0.3"]
    crit = [c["t"] for c in Dd["G0_critical_points"]]          # max, min, max of G0
    tf, bf = Dd["fold_time"], Dd["B_top_det"]
    tg = np.linspace(fwd.TAU, fwd.TEND, 600001)
    G0, G1, _ = fwd.g0_derivs(tg, 2, 0.3)
    r0 = G1 / G0**2
    bmax = 10.0
    m_first = (tg <= crit[0]) & (r0 <= bmax)
    m_valley = (tg > crit[1]) & (tg <= tf)
    m_late = (tg >= tf) & (tg < crit[2])
    ax.plot(r0[m_first], tg[m_first], color=core.OI_BLUE, lw=1.2, label="peaks")
    ax.plot(r0[m_late], tg[m_late], color=core.OI_BLUE, lw=1.2)
    ax.plot(r0[m_valley], tg[m_valley], color=core.OI_VERMILLION, lw=1.2, ls="--", label="valley")
    ax.plot([bf], [tf], marker="D", ms=3.5, color="k", ls="none", label="fold")
    worst = 0.0
    for row in Dd["f_det_at_budgets"]:
        for sp in row["stationary_points"]:
            g, g1, _ = fwd.g0_derivs(np.array([sp["t"]]), 2, 0.3)
            worst = max(worst, abs(float(g1[0] / g[0] ** 2) / row["B"] - 1.0))
            if row["B"] <= bmax:
                ax.plot([row["B"]], [sp["t"]], marker="o", ms=2.4, mfc="white", mec="0.25",
                        ls="none")
    ax.axvline(bf, color="0.5", lw=0.6, ls=":")
    ax.set_xlim(0, bmax)
    ax.set_ylim(0.8, 2.6)
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("critical time $t$")
    ax.set_title("(c)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, ncol=3)
    ax.grid(alpha=0.3)
    _check("fig7c_fold_B_equals_max_r0_on_flank_2", float(np.max(r0[m_valley | m_late])), bf, 1e-6)
    CHECKS["fig7c_recorded_fdet_stationary_points_on_r0_equals_B_max_rel_dev"] = worst
    if worst > 1e-6:
        raise RuntimeError(f"fig7c: recorded f_det stationary points off r0 = B by {worst:g}")
    return _save(fig, "fb_paper_fig7_fold")


# ============================================================================
# SM figures redrawn at SM print size without campaign tags or internal wording.
# Each is written as fb_paper_sm_<stem> in artifacts/figures/ (the campaign originals
# are not touched) and copied by --copy-to under the file name the SM includes
# (SM_COPY_AS), byte-identically.
# ============================================================================

SM_COPY_AS = {
    "fb_paper_sm_n0_fk_validation.pdf": "fb_n0_fk_validation.pdf",
    "fb_paper_sm_n2_residual_decomposition.pdf": "fb_n2_residual_decomposition.pdf",
    "fb_paper_sm_single_particle_control.pdf": "exact_m_single_particle_control_prr.pdf",
    "fb_paper_sm_single_particle_control_p2.pdf": "exact_m_single_particle_control_p2_prr.pdf",
    "fb_paper_sm_w2_seed_repeat.pdf": "exact_m_w2_seed_repeat_prr.pdf",
}


def sm_n0_validation() -> list[str]:
    """Exact-law vs direct-kill validation (N0 primary cells), recomputed from the stored
    N0 ensembles with the unchanged compare_cell of exact_m_prr_fk_n0_validation.py; the
    per-cell max|z| and chi2_cov/dof are checked against n0_validation.json#primary first.
    src: artifacts/data/exact_m_fixed_budget/n0_validation.json#primary; stored ensembles
    n0_m{2,3}_eps{0.1,0.05} (2e5 paths); direct-kill references PRIMARY of that script."""
    plt = _plt()
    import exact_m_prr_fk_n0_validation as n0
    rec = {r["label"]: r for r in _load("n0_validation.json")["primary"]}
    groups = int(_load("n0_validation.json")["groups_for_batch_means_and_jackknife"])
    rows = []
    for label, path in n0.PRIMARY:
        ref = n0.read_reference(path)
        ens = fk.load_ensemble(n0.ENSEMBLES[(ref["m"], ref["eps"])])
        row = n0.compare_cell(ens, ref, groups)
        _check(f"smN0_{label}_max_abs_z", row["max_abs_z"], rec[label]["max_abs_z"], 1e-9)
        _check(f"smN0_{label}_chi2_cov_over_dof", row["chi2_cov_over_dof"],
               rec[label]["chi2_cov_over_dof"], 1e-9)
        rows.append(row)
    fig, axes = plt.subplots(2, 3, figsize=(SM_WIDTH_IN, 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]}, layout="constrained")
    for col, (row, letter) in enumerate(zip(rows, "abc")):
        pl = row["_plot"]
        t = np.asarray(pl["t"]); f = np.asarray(pl["fk_density"]); se = np.asarray(pl["fk_density_se"])
        ax = axes[0, col]
        ax.fill_between(t, f - 2 * se, f + 2 * se, color=core.OI_BLUE, alpha=0.3, lw=0)
        ax.plot(t, f, color=core.OI_BLUE, lw=1.0)
        ax.plot(t, pl["dk_density"], ".", ms=2.0, color=core.OI_VERMILLION)
        ax.set_title(f"({letter}) $m={row['m']}$, $\\varepsilon={row['eps']:g}$")
        ax.set_ylim(bottom=0)
        axz = axes[1, col]
        axz.plot(t, pl["z"], "o", ms=1.4, color="0.25")
        for yv in (-4, 4):
            axz.axhline(yv, color=core.OI_VERMILLION, lw=0.7, ls="--")
        axz.axhline(0, color="0.6", lw=0.5)
        axz.set_ylim(-5, 8.5)
        axz.set_yticks([-4, 0, 4])
        axz.set_xlim(0.5, 3.5)
        axz.set_xticks([1, 2, 3])
        axz.text(0.03, 0.96, f"max$|z|$ {row['max_abs_z']:.2f}, $\\chi^2$/dof {row['chi2_cov_over_dof']:.2f}",
                 transform=axz.transAxes, va="top", ha="left", fontsize=7.0)
        CHECKS[f"smN0_{row['m']}_{row['eps']:g}_walkers_direct_kill"] = row["walkers_direct_kill"]
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(color=core.OI_BLUE, alpha=0.45, label="exact law"),
                               Line2D([], [], ls="none", marker="o", ms=2.5, color=core.OI_VERMILLION,
                                      label="direct kill")],
                      loc="upper right", frameon=False, handlelength=1.2)
    axes[0, 0].set_ylabel("density $f(t)$")
    axes[1, 0].set_ylabel("$z$ per bin")
    axes[1, 1].set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    return _save(fig, "fb_paper_sm_n0_fk_validation")


def sm_n2_residual() -> list[str]:
    """Decomposition of f - f_1 (N2), recomputed from the stored N0 ensembles with the same
    estimators as fb_n2_meanfield_residual.cmd_figure; the L1 norms of the total and of the
    contact-gating part are checked against N2/n2_meanfield_residual.json#cells.*.decomposition.
    src: artifacts/data/exact_m_fixed_budget/N2/n2_meanfield_residual.json"""
    plt = _plt()
    import exact_m_prr_mean_field_boundary as mfb
    import fb_n2_meanfield_residual as n2
    d = _load("N2/n2_meanfield_residual.json")
    panels = [("m2_eps0.05", 1.0), ("m2_eps0.05", 4.0), ("m3_eps0.1", 1.0), ("m3_eps0.1", 4.0)]
    fig, axes = plt.subplots(2, 2, figsize=(SM_WIDTH_IN, 3.75), layout="constrained", sharex=True)
    cols = {"total": "k", "gating_full_minus_mc": core.OI_VERMILLION,
            "exposure_mc_minus_f1mc": core.OI_BLUE, "dt_f1full_minus_f1cont": core.OI_GREEN}
    labs = {"total": "total: $f - f_1$",
            "gating_full_minus_mc": "contact gating: $f - f_{\\bar c}$",
            "exposure_mc_minus_f1mc": "exposure fluctuation: $f_{\\bar c} - f_1^{\\rm EM}$",
            "dt_f1full_minus_f1cont": "time step: $f_1^{\\rm EM} - f_1$"}
    for ax, (lab, B), letter in zip(axes.ravel(), panels, "abcd"):
        m, eps, name = n2.CELLS[lab]
        ens = fk.load_ensemble(name)
        w = [1.0 / m] * m
        wi = np.asarray(ens.window_index)
        ew = ens.edges[wi]
        t = 0.5 * (ew[:-1] + ew[1:])
        bw = np.diff(ew)
        law = {v: fk.exact_law(ens, B, w, variant=v)["bin_mass"] for v in ("full", "meancontact")}
        fe = fk.free_exposure_discrete(ens.spec, w, gate="contact")
        lam_f = np.concatenate([[0.0], np.cumsum(fe["G"]) * ens.spec.dt])[ens.cps][wi]
        fm = fk.free_exposure_discrete(ens.spec, w, gate="mean")
        lam_m = np.concatenate([[0.0], np.cumsum(fm["G"]) * ens.spec.dt])[ens.cps][wi]
        clock = mfb.clock_for(m, eps, tuple(w))
        lam_c = clock.lam[clock.edge_index]
        comp = {"total": law["full"] - n2.mf_bins(lam_c, B),
                "gating_full_minus_mc": law["full"] - law["meancontact"],
                "exposure_mc_minus_f1mc": law["meancontact"] - n2.mf_bins(lam_m, B),
                "dt_f1full_minus_f1cont": n2.mf_bins(lam_f, B) - n2.mf_bins(lam_c, B)}
        rec = d["cells"][lab]["decomposition"][f"{B:g}"]
        _check(f"smN2_{lab}_B{B:g}_L1_total", np.abs(comp["total"]).sum(), rec["L1_total"], 1e-9)
        _check(f"smN2_{lab}_B{B:g}_L1_gating", np.abs(comp["gating_full_minus_mc"]).sum(),
               rec["L1"]["gating_full_minus_mc"], 1e-9)
        for k in ("exposure_mc_minus_f1mc", "dt_f1full_minus_f1cont"):
            CHECKS[f"smN2_{lab}_B{B:g}_L1_{k}"] = {"figure": float(np.abs(comp[k]).sum()),
                                                  "json": rec["L1"][k]}
        for k, c in comp.items():
            ax.plot(t, c / bw, color=cols[k], lw=1.3 if k == "total" else 1.0,
                    ls="-" if k != "dt_f1full_minus_f1cont" else "--", label=labs[k])
        ax.axhline(0, color="0.6", lw=0.5)
        for tj in ens.spec.times():
            ax.axvline(tj, color="0.7", lw=0.6, ls=":")
        ax.set_title(f"({letter}) $m={m}$, $\\varepsilon={eps:g}$, $B={B:g}$")
        ax.set_xlim(0.5, 3.5)
    for ax in axes[1]:
        ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density difference (units of $\\gamma$)")
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="outside lower center", ncol=2, frameon=False)
    return _save(fig, "fb_paper_sm_n2_residual_decomposition")


def _sm_single_particle_one(records: list, particle: int, letters: str, legend: bool,
                            stem: str) -> list[str]:
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    chosen = [r for r in records if r["parameters"]["particle"] == particle]
    order = {(2, 0.05): 0, (3, 0.10): 1, (2, 0.10): 2}
    chosen.sort(key=lambda r: order.get((r["parameters"]["config"]["m"],
                                          round(r["parameters"]["config"]["eps"], 6)), 9))
    if len(chosen) != 3:
        raise SystemExit(f"single-particle control: expected 3 cells for particle {particle}")
    fig, axes = plt.subplots(1, 3, figsize=(SM_WIDTH_IN, 2.55 if legend else 2.05),
                             layout="constrained")
    c_smooth = "#20415f"
    for k, (ax, rec) in enumerate(zip(axes, chosen)):
        cfg = rec["parameters"]["config"]
        res = rec["results"]
        classifier = res["classifier"]
        verdict = res["classifier_covariance_aware"]
        walkers = cfg["walkers"]
        edges = np.asarray(classifier["edges"])
        centres_t = 0.5 * (edges[:-1] + edges[1:])
        density = np.asarray(classifier["counts"], float) / (walkers * classifier["bin_width"])
        smoothed = np.asarray(classifier["smoothed_density"])
        comp = rec["comparator_curve"]
        ce = np.asarray(comp["edges"])
        curves = rec["theory"]["curves"]
        ts = np.asarray(curves["t"])
        ax.stairs(density, edges, fill=True, alpha=0.30, color=core.OI_BLUE)
        ax.plot(0.5 * (ce[:-1] + ce[1:]), comp["smoothed_density"], color="0.55", lw=1.6, alpha=0.9)
        ax.plot(centres_t, smoothed, color=c_smooth, lw=1.1)
        ax.plot(ts, curves["B_G1"], color=core.OI_VERMILLION, lw=1.0, ls="--")
        ax.plot(ts, curves["f1_single"], color=core.OI_GREEN, lw=1.0, ls="-.")
        for tj in cfg["target_times"]:
            ax.axvline(tj, color="0.45", lw=0.6, ls=":")
        for row in verdict["significant_maxima"]:
            ax.plot(row["time"], row["smoothed_height"], marker="v", color=core.OI_GREEN, mec="k",
                    mew=0.3, ms=4.5, ls="none")
        comp_modes = rec["comparator_pair_centre"]["mode_count"]
        CHECKS[f"smI1_p{particle}_m{cfg['m']}_eps{cfg['eps']:g}_modes"] = [verdict["mode_count"], comp_modes]
        ax.text(0.97, 0.96, f"{verdict['mode_count']} modes\n(pair-centre field: {comp_modes})",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.0,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))
        ax.set_title(f"({letters[k]}) $m={cfg['m']}$, $\\varepsilon={cfg['eps']:g}$")
        ax.set_xlim(0.0, 4.0)
        ymax = max(float(density.max()), float(np.max(comp["smoothed_density"])),
                   float(np.max(curves["B_G1"])))
        ax.set_ylim(0.0, 1.32 * ymax)
        ax.set_xlabel("reaction time $t$ (units of $1/\\gamma$)" if k == 1 else "")
        if k == 0:
            ax.set_ylabel("density $f(t)$")
    if legend:
        handles = [Patch(color=core.OI_BLUE, alpha=0.35, label="EM histogram"),
                   Line2D([], [], color=c_smooth, lw=1.1, label="smoothed ($h=0.04$)"),
                   Line2D([], [], color="0.55", lw=1.6, label="pair-centre field, smoothed"),
                   Line2D([], [], color=core.OI_VERMILLION, lw=1.0, ls="--",
                          label="free exposure rate $BG_1(t)$"),
                   Line2D([], [], color=core.OI_GREEN, lw=1.0, ls="-.", label="mean field $f_1$"),
                   Line2D([], [], color=core.OI_GREEN, marker="v", mec="k", mew=0.3, ms=4.5, ls="none",
                          label="significant maximum")]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    return _save(fig, stem)


def sm_single_particle() -> list[str]:
    """Single-particle field control (W6), from the stored cell JSONs (no simulation).
    src: artifacts/data/exact_m_prr_upgrade/w6_single_particle/m*_eps*_B1_p{1,2}.json"""
    import exact_m_prr_upgrade_w6_single_particle as w6
    records = w6.load_records(core.UPGRADE_DATA / "w6_single_particle")
    out = _sm_single_particle_one(records, 1, "abc", True, "fb_paper_sm_single_particle_control")
    out += _sm_single_particle_one(records, 2, "def", False, "fb_paper_sm_single_particle_control_p2")
    return out


def sm_seed_repeat() -> list[str]:
    """Seed repeats of the direct-kill prominence-floor crossing (bisection brackets =
    resolution, not CI), drawn for the 0.40\\textwidth SM minipage.
    src: artifacts/data/exact_m_prr_upgrade/robustness/w2_seed_repeat_summary.json#cells"""
    plt = _plt()
    import matplotlib
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    fs = 7.0
    rc = {k: fs for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                          "ytick.labelsize", "legend.fontsize")}
    summ = json.loads((ROBUST / "w2_seed_repeat_summary.json").read_text())
    orig_seed = str(summ["campaign_seed_of_stored_chain"])
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(SM_MINIPAGE_IN, 2.45), layout="constrained")
        for k, (ax, cell) in enumerate(zip(axes, summ["cells"])):
            chains = [c for c in cell["chains"] if c.get("status") == "bisected"]
            stored = cell["stored_w2_chain"]
            interp = cell["sampling_scale_diagnostic"]["floor_interpolated_crossings"]
            labels, mids, lo, hi, colours, fi = [], [], [], [], [], []
            col = core.OI_BLUE if cell["m"] == 2 else core.OI_VERMILLION
            entries = [(orig_seed, stored["b0"], stored["b0_bracket"], "0.55")]
            entries += [(str(c["seed"]), c["b0"], c["b0_bracket"], col) for c in chains]
            for sd, b0, br, cc in entries:
                labels.append(sd[-2:]); mids.append(b0)
                lo.append(b0 - br[0]); hi.append(br[1] - b0)
                colours.append(cc); fi.append(interp.get(sd))
            xs = np.arange(len(labels))
            for x, y, l, h, cc, f in zip(xs, mids, lo, hi, colours, fi):
                ax.errorbar([x], [y], yerr=[[l], [h]], marker="_", ms=5, lw=1.1, capsize=2.5, color=cc)
                if f is not None:
                    ax.plot([x + 0.2], [f], marker="o", ms=3.0, mfc="none", mec=cc, mew=0.9, ls="none")
            mf = cell["mean_field"]["upper_crossing_unrounded"]
            ax.axhline(mf, ls="--", lw=1.0, color=core.OI_ORANGE)
            y_lo = min(mv - l for mv, l in zip(mids, lo))
            y_hi = max(max(mv + h for mv, h in zip(mids, hi)), mf)
            span = y_hi - y_lo
            ax.set_ylim(y_lo - 1.25 * span, y_hi + 0.10 * span)
            ax.set_xticks(xs, labels)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.set_title(f"({'ab'[k]}) $m={cell['m']}$, $\\varepsilon={cell['eps']:g}$")
            ax.grid(True, alpha=0.3)
            CHECKS[f"smW2_{cell['cell']}_mean_field_crossing"] = mf
        axes[0].set_ylabel("floor crossing $B$")
        fig.supxlabel("chain seed (last two digits)", fontsize=fs)
        handles = [Line2D([], [], ls="--", color=core.OI_ORANGE, lw=1.0, label="mean field"),
                   Line2D([], [], marker="_", ms=5, ls="-", lw=1.1, color="0.3", label="bracket"),
                   Line2D([], [], marker="o", ms=3.0, mfc="none", mec="0.3", ls="none",
                          label="interpolated")]
        axes[0].legend(handles=handles, loc="lower left", frameon=False, handlelength=1.4,
                       borderaxespad=0.2)
        return _save(fig, "fb_paper_sm_w2_seed_repeat")


SM_FIGS = {"fig5_radii": sm_fig5_radii, "fig6_unsmoothed": sm_fig6_unsmoothed,
           "n0": sm_n0_validation, "n2": sm_n2_residual, "single_particle": sm_single_particle,
           "seed_repeat": sm_seed_repeat}


FIGS = {2: fig2_allocation, 3: fig3_densities, 4: fig4_masses, 5: fig5_contact,
        6: fig6_visibility, 7: fig7_fold}


# ============================================================================
# SM copies without the campaign corner tag ("fb N1", "fb N4").  The original
# plotting functions are re-run unchanged from their recorded data, with
# core.stream_tag disabled and the output stem renamed fb_paper_sm_<stem>;
# the campaign originals in artifacts/figures/ are not touched.
#   fb_n1_pstar       <- fb_n1_allocation_design.cmd_figure (JSON only; window-restricted
#                        basins, as the recorded scan holds only those)
#   fb_n4_preparation <- fb_n4_preparation.cmd_figure (stored ensembles, exact law)
# ============================================================================

SM_KEEP = {"fb_n1_pstar", "fb_n4_preparation"}


def sm_untagged() -> list[str]:
    import matplotlib.pyplot as plt
    written: list[str] = []
    orig_save, orig_tag = core.save_figure, core.stream_tag

    def save(fig, stem):
        stem = Path(stem)
        if stem.name not in SM_KEEP:
            return []
        out = orig_save(fig, core.FIGURES / f"fb_paper_sm_{stem.name}")
        written.extend(out)
        return out

    core.save_figure = save
    core.stream_tag = lambda fig, label: None
    try:
        import fb_n1_allocation_design as n1
        n1.cmd_figure(None)
        plt.close("all")
        import fb_n4_preparation as n4
        n4.cmd_figure(None)
        plt.close("all")
    finally:
        core.save_figure, core.stream_tag = orig_save, orig_tag
    return written


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated figure numbers (default: all; 0 = none)")
    ap.add_argument("--copy-to", default="", help="copy the final PDFs byte-identically to this directory")
    ap.add_argument("--sm", action="store_true", help="also write the SM figures (fb_paper_sm_*)")
    ap.add_argument("--sm-figs", default="",
                    help=f"comma-separated subset of SM figures ({','.join(SM_FIGS)},untagged); default all")
    args = ap.parse_args(argv)
    which = [int(x) for x in args.only.split(",") if x.strip()] or sorted(FIGS)
    which = [k for k in which if k != 0]
    written = {}
    for k in which:
        written[k] = FIGS[k]()
        print(f"Fig. {k}:", *written[k])
    sm_sel = [x.strip() for x in args.sm_figs.split(",") if x.strip()] or (list(SM_FIGS) + ["untagged"])
    if args.sm:
        for key in sm_sel:
            written[f"sm_{key}"] = sm_untagged() if key == "untagged" else SM_FIGS[key]()
            print(f"SM {key}:", *written[f"sm_{key}"])
    sha = {}
    for k, files in written.items():
        for f in files:
            sha[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    copied_as = {}
    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        for files in written.values():
            for f in files:
                if f.endswith(".pdf"):
                    name = SM_COPY_AS.get(Path(f).name, Path(f).name)
                    shutil.copyfile(f, dest / name)
                    assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == sha[Path(f).name]
                    copied_as[Path(f).name] = name
    if sorted(which) == sorted(FIGS) and args.sm and set(sm_sel) == set(SM_FIGS) | {"untagged"}:
        MANIFEST.write_text(json.dumps({"script": "code/fb_make_paper_figures.py",
                                        "text_width_in": TEXT_WIDTH_IN, "sm_width_in": SM_WIDTH_IN,
                                        "sm_minipage_in": SM_MINIPAGE_IN, "font_pt": FS,
                                        "copied_as": copied_as or {k: v for k, v in SM_COPY_AS.items()},
                                        "checks": CHECKS, "sha256": sha}, indent=1, sort_keys=True,
                                       default=_json_default) + "\n")
        print("manifest:", MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
