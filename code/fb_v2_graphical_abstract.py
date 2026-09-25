#!/usr/bin/env python3
"""Graphical abstract of the CNSNS submission, version 2 (optional Elsevier item).

Built from the programmed plug-flow channel of Fig. 1(b) of the article
(code/fb_v2_hero_channel.py): one fixed total amount of catalyst, split three ways
among five stripes on the floor of an idealised electro-osmotic channel, gives five
reaction pulses at the programmed times 10, 13, 18, 22 and 28 s with rising, falling
or equal weights.

Left: the channel floor, stripes at x = U t_j' (mm) with bar height = catalyst share
w_j of each design (the stored designs, V2_hero/designs/*eps0.03*).  Right: the
exact reaction-time densities of the three designs (the quadrature law of
fb_v2_hero_channel.quad_law, eps = 0.03, Peclet number 1111), with the target times
dotted.  Deterministic: no simulation, no random numbers.

Sources: artifacts/data/exact_m_fixed_budget/V2_hero/design_eps0.03.json#phys
(U = 100 um/s, D = 9e-11 m^2/s, k_tot = 800 um/s, time unit 10 s) and the design
files read by fb_v2_hero_channel.load_designs(0.03).

Output: manuscript/cnsns_submission/graphical_abstract.{pdf,png}.
Size: 13 cm x 5.2 cm; PNG at 300 dpi = 1535 x 614 px (Elsevier minimum
1328 x 531 px, readable at 13 x 5 cm); smallest text 7 pt at the printed size.
Colours: Okabe-Ito blue / vermillion / green, as in the article; each curve is
also labelled by name.

Usage (from code/):  python3 fb_v2_graphical_abstract.py   (NumPy, Matplotlib)
"""
from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1758758400")  # fixed PDF metadata date

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import fb_v2_hero_channel as hero  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
OUT_STEM = REPORT / "manuscript" / "cnsns_submission" / "graphical_abstract"
EPS = 0.03
CM = 1.0 / 2.54


def main() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    import exact_m_prr_upgrade_core as core

    phys = json.loads((hero.OUT / f"design_eps{EPS:g}.json").read_text(encoding="utf-8"))["phys"]
    tu = hero.PHYS["L_m"] / hero.PHYS["U_m_per_s"]          # seconds per model time unit (10 s)
    mm_per_unit = hero.PHYS["L_m"] * 1e3                       # stripe position x = U t, in mm
    designs = hero.load_designs(EPS)
    col = {"rising": core.OI_BLUE, "falling": core.OI_VERMILLION, "equal": core.OI_GREEN}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "xtick.major.size": 2.5, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig = plt.figure(figsize=(13 * CM, 5.2 * CM))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.78, 1.55], left=0.035, right=0.985,
                          bottom=0.25, top=0.80, wspace=0.10, hspace=0.12)
    fig.text(0.5, 0.985, "Splitting one fixed catalyst budget programs the reaction pulses",
             ha="center", va="top", fontsize=8.5, weight="bold")
    xs_lo, xs_hi = 0.75 * mm_per_unit, 3.1 * mm_per_unit
    t_lo, t_hi = 0.75 * tu, 3.35 * tu
    for i, kind in enumerate(hero.PROFILES):
        c = np.asarray(designs[kind]["c"], dtype=float)
        w = np.asarray(designs[kind]["w"], dtype=float)
        # left: channel floor with the stripes (position mm, bar height = share)
        axl = fig.add_subplot(gs[i, 0])
        axl.add_patch(Rectangle((xs_lo, -0.06), xs_hi - xs_lo, 0.06, color="0.85", lw=0))
        for cj, wj in zip(c, w):
            axl.add_patch(Rectangle((cj * mm_per_unit - 0.045, 0.0), 0.09, wj, color=col[kind], lw=0))
        axl.set_xlim(xs_lo, xs_hi)
        axl.set_ylim(-0.08, 1.0)
        axl.set_yticks([])
        axl.spines[["left", "right", "top"]].set_visible(False)
        axl.text(0.02, 0.93, kind, transform=axl.transAxes, ha="left", va="top", color=col[kind],
                 fontsize=8, weight="bold")
        if i == 0:
            axl.set_title("catalyst share of each stripe", fontsize=8, pad=3)
            arr = FancyArrowPatch((0.40, 0.52), (0.74, 0.52), transform=axl.transAxes,
                                  arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.3")
            axl.add_patch(arr)
            axl.text(0.57, 0.57, "flow", transform=axl.transAxes, ha="center", va="bottom",
                     fontsize=7.5, color="0.3")
        axl.set_xticks([1, 2, 3])
        if i < 2:
            axl.set_xticklabels([])
        else:
            axl.set_xlabel("position along channel (mm)", labelpad=1)
        # right: exact density (per second) with the target times dotted
        ql = hero.quad_law(c, w, hero.B_BUDGET, EPS, grad=False)
        f = ql.p / ql.meta["dt"] / tu
        axr = fig.add_subplot(gs[i, 1])
        if i == 0:
            axr.set_title("exact reaction-time density", fontsize=8, pad=3)
        for tj in hero.T_TARGET:
            axr.axvline(tj * tu, color="0.45", lw=0.6, ls=":", zorder=0)
        axr.fill_between(ql.t * tu, f, color=col[kind], alpha=0.30, lw=0)
        axr.plot(ql.t * tu, f, color=col[kind], lw=0.9)
        axr.set_xlim(t_lo, t_hi)
        axr.set_ylim(0, 1.12 * float(f.max()))
        axr.set_yticks([])
        axr.spines[["left", "right", "top"]].set_visible(False)
        axr.set_xticks([10, 13, 18, 22, 28])
        if i < 2:
            axr.set_xticklabels([])
        else:
            axr.set_xlabel("reaction time (s)", labelpad=1)
    fig.text(0.5, 0.012,
             f"same total catalyst ($k_{{\\rm tot}}={phys['k_tot_um_per_s']:.0f}\\,\\mu$m/s) in all three;"
             " idealised model channel",
             ha="center", va="bottom", fontsize=7, color="0.3", style="italic")

    outs = []
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=300)
        outs.append(OUT_STEM.with_suffix(suffix).name)
    plt.close(fig)
    assert math.isclose(phys["k_tot_um_per_s"], 800.0)
    print("[graphical abstract]", outs, flush=True)
    return outs


if __name__ == "__main__":
    main()
