#!/usr/bin/env python3
"""V2 item 3 (CV-1 + CV-2): bifurcation set of the fixed-width limit law f_det (replaces Fig. 7(c)).

Reads only certified JSON:
  ../artifacts/data/exact_m_fixed_budget/V2_bifurcation/fold_curve_m2.json   (CV-1, rho_s-boxes, m = 2)
  ../artifacts/data/exact_m_fixed_budget/V2_bifurcation/codim2_switch.json   (CV-2, allocation boxes)
  ../artifacts/data/exact_m_fixed_budget/N9b/n9b_fold.json                   (Sec. 4.6 exact-law fold statistic)
The max-min path of Prop. 3(iii) is drawn from its closed form (A = 1, v_j = 4 e^{-t_j}) and checked
against the interval values stored by CV-2 (B_path(0.5), w*(B = 8)).

(a) (rho_s, B), m = 2, equal weights: certified boxes of B_top^det (rho_s-box x B-hull), terminal width
    rho_c; inset: exact-law fold statistic B*_h (h = 0.03) at rho_s = 0.3 against eps^2 (+-2 SE), the
    same statistic of the limit law (dashed) and the certified B_top^det(0.3) (solid).
(b) (w_2, B), m = 2, rho_s = 0.3: certified fold boxes b_2(w_2), r0(tau)(w_2), (D1) edges (located), the
    equal-weight fold and the max-min path w_2*(B).
(c) (q = w_2/w_3, B), m = 3, rho_s = 0.2, w_1 = 0.999: certified b_2, b_3 boxes, r0(tau), and the
    codimension-2 point b_2 = b_3 (enclosed), where the first peak lost switches from the last to the middle.

Run from code/:  python3 fb_v2_bifurcation_figure.py
Output: ../artifacts/figures/fb_v2_bifurcation_set.pdf (+ .png)
"""
from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1758585600")

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
CV1 = DATA / "V2_bifurcation" / "fold_curve_m2.json"
CV2 = DATA / "V2_bifurcation" / "codim2_switch.json"
N9B = DATA / "N9b" / "n9b_fold.json"
OUT = HERE.parent / "artifacts" / "figures" / "fb_v2_bifurcation_set.pdf"

FS = 8.0
TEXT_WIDTH_IN = 390.0 / 72.27
BLUE, ORANGE, GREEN, VERM, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "0.45"
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                     "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "mathtext.fontset": "dejavusans",
                     "font.family": "DejaVu Sans", "axes.titlesize": FS, "axes.titlelocation": "left",
                     "savefig.dpi": 300, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "lines.linewidth": 0.9})
CHECKS = {}


def check(name, value, expected, tol):
    ok = abs(value - expected) <= tol
    CHECKS[name] = (value, expected, ok)
    if not ok:
        raise SystemExit(f"figure check failed: {name}: {value} vs {expected}")


def boxes(ax, rows, key, color, alpha=1.0, lw=0.6):
    for r in rows:
        if not r["certified"]:
            continue
        x0, x1 = (float(x) for x in r["box"])
        b0, b1 = (float(x) for x in (r["b"][key] if key != "r0" else r["r0_tau"]))
        ax.add_patch(Rectangle((x0, b0), x1 - x0, b1 - b0, lw=lw, ec=color, fc=color, alpha=alpha))


def mids(rows, key):
    xs, ys = [], []
    for r in rows:
        if not r["certified"]:
            continue
        x0, x1 = (float(x) for x in r["box"])
        v = r["b_mid"][key] if key != "r0" else r["r0_tau"]
        xs.append(0.5 * (x0 + x1))
        ys.append(0.5 * (float(v[0]) + float(v[1])))
    return np.array(xs), np.array(ys)


def maxmin_path(ys):
    """closed form of Prop. 3(iii) at m = 2; y = -ln(1 - 2p)."""
    v1, v2 = 4 * math.exp(-1.0), 4 * math.exp(-2.5)
    x = np.exp(-ys)
    l1 = math.log(2) - np.log1p(x)
    l2 = np.log((1 + x) / 2) + ys
    B = v1 * l1 + v2 * l2
    return v2 * l2 / B, B


def panel_a(ax, d1):
    for key, col in (("table_required_range", BLUE), ("table_extension", SKY)):
        for row in d1[key]:
            if not row["certified"]:
                continue
            r0, r1 = (float(x) for x in row["rho"])
            b0, b1 = (float(x) for x in row["B_hull"])
            ax.add_patch(Rectangle((r0, b0), r1 - r0, b1 - b0, lw=0.8, ec=col, fc=col))
    rc = float(d1["terminal_width"]["rho_c=Delta/2=2(e^-1-e^-5/2)"][0])
    ax.axvline(rc, color=GREY, ls="--", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlim(0.14, 0.60)
    ax.set_ylim(3e-3, 3e4)
    ax.set_xlabel(r"stripe width $\rho_s$")
    ax.set_ylabel(r"budget $B$")
    ax.text(0.405, 6.5e-3, "2 peaks", ha="left", va="center")
    ax.text(0.36, 2e2, "1 peak", ha="left", va="center")
    ax.text(rc - 0.008, 3e1, r"$\rho_c$", ha="right", va="center", color=GREY)
    ax.plot([0.3], [8.2247], "o", ms=3.2, mfc="white", mec=ORANGE, mew=0.9, zorder=5)
    ax.set_title(r"(a) $m=2$, $w_1=w_2$")
    # inset: exact-law fold statistic at rho_s = 0.3 (Sec. 4.6)
    n9 = json.loads(N9B.read_text())
    ins = ax.inset_axes([0.075, 0.085, 0.36, 0.33])
    eps = ["0.06", "0.03", "0.02"]
    e2 = [float(e) ** 2 for e in eps]
    bs = [n9["eps"][e]["by_h"]["0.03"]["B_star"] for e in eps]
    se = [n9["eps"][e]["by_h"]["0.03"]["jackknife_se"] for e in eps]
    bdet_h = n9["deterministic_same_statistic_by_h"]["0.03"]
    bt = float(d1["anchor_consistency_vs_TH8_certificate"]["0.3"]["B_top_det_uA"][0])
    check("inset_B_star_eps0.03", bs[1], 7.870510019733099, 1e-9)
    check("inset_Bdet_h0.03", bdet_h, 8.005553803954516, 1e-9)
    check("inset_Btopdet_0.3", bt, 8.22466474559830, 1e-12)
    ins.errorbar(e2, bs, yerr=[2 * s for s in se], fmt="o", ms=2.4, color=ORANGE, elinewidth=0.7, capsize=1.2)
    ins.axhline(bdet_h, color=ORANGE, ls="--", lw=0.7)
    ins.axhline(bt, color=BLUE, lw=0.9)
    ins.set_xlim(-3e-4, 4.2e-3)
    ins.set_ylim(7.3, 8.4)
    ins.set_xticks([0, 3e-3])
    ins.set_xticklabels(["0", ".003"])
    ins.set_yticks([7.5, 8.0])
    ins.tick_params(labelsize=7, length=2, pad=1)
    ins.yaxis.tick_right()
    ins.text(0.45, 0.07, r"$\varepsilon^2$", transform=ins.transAxes, ha="center", va="bottom", fontsize=7)
    ins.patch.set_alpha(0.9)


def panel_b(ax, d2):
    rows = d2["tables"]["F2W"]
    boxes(ax, rows, "2", BLUE)
    boxes(ax, rows, "r0", GREY)
    xs, ys = mids(rows, "2")
    xr, yr = mids(rows, "r0")
    ax.fill_between(xs, 1e-3, ys, color=BLUE, alpha=0.10, lw=0)
    ax.fill_between(xs, ys, np.interp(xs, xr, yr), color=GREY, alpha=0.08, lw=0)
    e = d2["D1_edges_located"]["F2W"]
    wl, wh = float(e[0]["w2"]), float(e[1]["w2"])
    for w in (wl, wh):
        ax.axvline(w, color=GREY, ls="--", lw=0.7)
    ax.axvspan(0, wl, color="0.85", lw=0)
    ax.axvspan(wh, 1, color="0.85", lw=0)
    # max-min path
    pth = d2["maxmin_path_F2W"]
    reach_w = float(pth["certified_reach"]["w2_upper"])
    exit_w = float(pth["path_leaves_D1_at"]["w2_located"])
    y = np.concatenate([np.geomspace(1e-3, 5, 400), np.geomspace(5.01, 400, 2000)])
    w, B = maxmin_path(y)
    check("path_B_at_w0.5", float(np.interp(0.5, w, B)), float(pth["B_path_at_w2_0.5"][0]), 1e-3)
    check("path_w_at_B8", float(np.interp(8.0, B, w)), float(pth["N1_check_w_star_B8"][0]), 1e-4)
    m1 = w <= reach_w
    m2 = (w >= reach_w) & (w <= exit_w)
    ax.plot(w[m1], B[m1], color=ORANGE, lw=1.3)
    ax.text(0.68, 0.78, r"max–min" + "\n" + r"$w^*(B)$", color=ORANGE, ha="center", va="center", fontsize=FS - 0.5)
    ax.plot(w[m2], B[m2], color=ORANGE, lw=1.3, ls=":")
    ax.plot([0.5], [8.2247], "o", ms=3.2, mfc="white", mec=BLUE, mew=0.9, zorder=5)
    ax.annotate("equal\nweights", xy=(0.5, 8.2247), xytext=(0.30, 60), fontsize=FS - 1, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 3e5)
    ax.set_xlabel(r"late-stripe weight $w_2$")
    ax.text(0.30, 3.0, "2 peaks", ha="center", va="center")
    ax.text(0.40, 800, "1 peak", ha="center", va="center")
    ax.text(0.50, 1.5e5, r"$r_0(\tau)$", ha="center", va="center", color=GREY, fontsize=FS - 1)
    ax.set_title(r"(b) $m=2$, $\rho_s=0.3$")


def panel_c(ax, d2):
    rows = d2["tables"]["F3S"]
    boxes(ax, rows, "2", BLUE)
    boxes(ax, rows, "3", GREEN)
    x2, y2 = mids(rows, "2")
    x3, y3 = mids(rows, "3")
    lo_ = np.minimum(y2, y3)
    hi_ = np.maximum(y2, y3)
    r0t = float(d2["codim2_switch_F3S"]["r0_tau"][0])
    ax.fill_between(x2, 1, lo_, color=BLUE, alpha=0.10, lw=0)
    ax.fill_between(x2, lo_, hi_, color=GREEN, alpha=0.08, lw=0)
    ax.axhline(r0t, color=GREY, lw=0.8)
    e = d2["D1_edges_located"]["F3S"]
    ql, qh = float(e[0]["q"]), float(e[1]["q"])
    for q in (ql, qh):
        ax.axvline(q, color=GREY, ls="--", lw=0.7)
    ax.axvspan(0.2, ql, color="0.85", lw=0)
    ax.axvspan(qh, 5, color="0.85", lw=0)
    sw = d2["codim2_switch_F3S"]
    qs, bs = float(sw["q_star"][0]), float(sw["B_star"][0])
    ax.plot([qs], [bs], "*", ms=7, color=VERM, mec="k", mew=0.4, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.30, 3.2)
    ax.set_ylim(8, 6000)
    ax.set_xticks([0.4, 1, 2])
    ax.set_xticklabels(["0.4", "1", "2"])
    ax.set_xlabel(r"$q=w_2/w_3$  ($w_1=0.999$)")
    ax.text(1.0, 60, "3 peaks", ha="center", va="center")
    ax.text(0.52, 780, "2", ha="center", va="center")
    ax.text(2.05, 330, "2", ha="center", va="center")
    ax.annotate(r"$b_2=b_3$", xy=(qs, bs), xytext=(0.62, 230), fontsize=FS - 0.5, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    ax.text(1.0, 1700, "1 peak", ha="center", va="center")
    ax.text(0.37, 380, r"$b_2$", color=BLUE, ha="center", va="center")
    ax.text(2.45, 26, r"$b_3$", color=GREEN, ha="center", va="center")
    ax.text(2.0, r0t * 1.35, r"$r_0(\tau)$", color=GREY, ha="center", va="center", fontsize=FS - 1)
    ax.set_title(r"(c) $m=3$, $\rho_s=0.2$")


def main():
    d1 = json.loads(CV1.read_text())
    d2 = json.loads(CV2.read_text())
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.35), layout="constrained",
                            gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]})
    panel_a(axs[0], d1)
    panel_b(axs[1], d2)
    panel_c(axs[2], d2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT.with_suffix(".png"))
    print("wrote", OUT)
    for k, v in CHECKS.items():
        print("check", k, v)


if __name__ == "__main__":
    main()
