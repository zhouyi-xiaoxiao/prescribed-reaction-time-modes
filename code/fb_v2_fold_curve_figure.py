#!/usr/bin/env python3
"""V2 item 3 (CV-1): draft panel of the certified m = 2 fold curve B_top^det(rho_s).

Reads only ../artifacts/data/exact_m_fixed_budget/V2_bifurcation/fold_curve_m2.json
(written by code/fb_v2_fold_curve_certificate.py) and, for the exact-law fold points of
Sec. 4.6 (rho_s = 0.3), ../artifacts/data/exact_m_fixed_budget/N9b/n9b_fold.json.
Draft for the item-3 figure (to be merged with the m = 3 / (w, B) panels); not in the paper yet.

(a) (rho_s, B) plane: certified boxes (rho-box x B-hull) on [0.15, 0.40] and the extension to
    0.5715, the 100-bit curve beyond it (dotted), the six TH8 anchor widths, the terminal width
    rho_c = Delta/2 and the exact-law fold statistic B*_h (h = 0.03) at eps = 0.06, 0.03, 0.02
    as short bars at rho_s = 0.3 (the smoothing lowers that statistic by O(h^2); SM N9b).
(b) approach to the terminal width: B_top^det against 1 - rho_s/rho_c with the 3/2 law.

Run from code/:  python3 fb_v2_fold_curve_figure.py
Output: ../artifacts/figures/fb_v2_fold_curve_m2.pdf (+ .png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
SRC = DATA / "V2_bifurcation" / "fold_curve_m2.json"
N9B = DATA / "N9b" / "n9b_fold.json"
OUT = HERE.parent / "artifacts" / "figures" / "fb_v2_fold_curve_m2.pdf"

FS = 8.0
TEXT_WIDTH_IN = 390.0 / 72.27
OI_BLUE, OI_ORANGE, OI_GREEN, OI_VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "0.45"
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "mathtext.fontset": "dejavusans", "font.family": "DejaVu Sans",
                     "axes.titlesize": FS, "axes.titlelocation": "left", "savefig.dpi": 300})


def main():
    d = json.loads(SRC.read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.5), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})
    # ---- (a) certified boxes (rho-box x B-hull); the 100-bit curve only beyond the certified set
    for key, col in (("table_required_range", OI_BLUE), ("table_extension", OI_GREEN)):
        for row in d[key]:
            if not row["certified"]:
                continue
            r0, r1 = (float(x) for x in row["rho"])
            b0, b1 = (float(x) for x in row["B_hull"])
            ax.add_patch(Rectangle((r0, b0), r1 - r0, b1 - b0, lw=0.9, ec=col, fc=col))
    rows = d["float_curve"]["rows"]
    last = float(d["extension_towards_rho_c"]["range"][1])
    tail = [r for r in rows if r[0] >= last] + [[r["rho"], r["B_top_det"]] for r in d["terminal_width"]["rows"]
                                                 if r["rho"] >= last]
    tail.sort()
    ax.plot([r[0] for r in tail], [r[1] for r in tail], color="k", lw=0.8, ls=":")
    rc = float(d["terminal_width"]["rho_c=Delta/2=2(e^-1-e^-5/2)"][0])
    ax.axvline(rc, color=GREY, ls="--", lw=0.8)
    anc = d["anchor_consistency_vs_TH8_certificate"]
    ax.plot([float(k) for k in anc], [float(v["B_top_det_uA"][0]) for v in anc.values()], "D", ms=3.5,
            mfc="white", mec=OI_VERM, mew=0.9, ls="none", label=r"widths of Prop. 4")
    n9 = json.loads(N9B.read_text())
    bs = [n9["eps"][e]["by_h"]["0.03"]["B_star"] for e in ("0.06", "0.03", "0.02")]
    ax.plot([0.3] * 3, bs, "_", ms=7, mew=1.2, color=OI_ORANGE, ls="none",
            label=r"exact law $B^*_{0.03}$, $\varepsilon=0.06,0.03,0.02$")
    ax.set_yscale("log")
    ax.set_xlim(0.14, 0.585)
    ax.set_ylim(1e-9, 3e4)
    ax.set_xlabel(r"stripe width $\rho_s$")
    ax.set_ylabel(r"budget $B$")
    ax.text(0.30, 2e-1, "$m$ peaks", ha="left", va="center")
    ax.text(0.40, 1e3, "$m-1$ peaks", ha="left", va="center")
    ax.text(rc - 0.005, 1e1, r"$\rho_c=\Delta/2$", rotation=90, ha="right", va="center", color=GREY)
    ax.plot([], [], "s", color=OI_BLUE, label=r"certified boxes, $[0.15,0.40]$")
    ax.plot([], [], "s", color=OI_GREEN, label=r"certified boxes, $[0.40,0.5715]$")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title(r"(a) fold curve $B_{\rm top}^{\rm det}(\rho_s)$, $m=2$")
    # ---- (b) approach to rho_c
    tr = d["terminal_width"]["rows"]
    x = [r["eps=1-rho/rho_c"] for r in tr]
    y = [r["B_top_det"] for r in tr]
    C = d["terminal_width"]["C_pred"]
    bx.loglog(x, y, "o", ms=2.5, color=OI_BLUE, label=r"$B_{\rm top}^{\rm det}$ (100-bit)")
    bx.loglog(x, [C * v ** 1.5 for v in x], color="k", lw=0.7, label=r"$C\,(1-\rho_s/\rho_c)^{3/2}$")
    bx.set_xlabel(r"$1-\rho_s/\rho_c$")
    bx.set_ylabel(r"$B_{\rm top}^{\rm det}$")
    bx.legend(loc="lower right", frameon=False)
    bx.set_title(r"(b) valley loss at $\rho_c$")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
