#!/usr/bin/env python3
"""Graphical abstract of the CNSNS paper (optional Elsevier item).

One schematic in three steps, read left to right:
  (1) catalyst stripes along the relaxing path of the pair centre,
  (2) a fixed budget B split among the stripes (the design variable),
  (3) the programmed reaction-time peaks with their weights M_j.

Everything is computed from the paper's formulas; nothing is simulated and no random
numbers are used.
  * mean path mu(t) = zbar + (z0 - zbar) exp(-gamma t) with the anchor data
    gamma = D0 = rho = 1, z0 = 4, zbar = 0, m = 3 target times (0.8, 1.6, 2.8);
  * stripe centres z_j = mu(t_j), crossing speeds v_j = gamma (z_j - zbar);
  * chosen weights M = (0.30, 0.20, 0.35) -> passage exposures by the closed-form
    inverse of the product law, lambda_j = -ln(1 - M_j / Pi_j), Pi_j = 1 - sum_{i<j} M_i
    (main text Eq. eq:inverse), and the budget split by the budget identity
    B w_j = A v_j lambda_j with A = W^{d-1} = 1 (Eq. eq:budget-identity);
  * the density is the weak-noise limit of Theorem 2(b) (thm:fixedB):
      f(t) ~ sum_j Pi_j (v_j/(eps rho)) F_{lambda_j}(v_j (t - t_j)/(eps rho)),
      F_lambda = p_lambda * phi_theta, p_lambda(u) = lambda phi(u) exp(-lambda Phi(u)),
      theta = s_Z/rho, s_Z^2 = D0/(2 gamma),
    drawn at eps = 0.08 so that the three peaks are visible at the printed size.
    Each peak's area is M_j (checked below), and each peak comes slightly before t_j,
    as Theorem 2(c) states.

Output: manuscript/cnsns_submission/graphical_abstract.{pdf,png}.
Size: 13 cm x 5 cm (Elsevier: readable at 5 x 13 cm); PNG at 300 dpi = 1535 x 591 px
(minimum 1328 x 531 px).  Smallest text 8 pt at the printed size.
Colours: Okabe-Ito blue / vermillion / green (as in the paper figures), validated for
colour-vision deficiency; every stripe and peak is also labelled by its index.

Usage:  python3 fb_graphical_abstract.py
"""
from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1758585600")  # fixed PDF metadata date

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # research/reports/encounter_multimodal_prr
OUT_DIR = ROOT / "manuscript" / "cnsns_submission"
STEM = "graphical_abstract"

# ---------------------------------------------------------------- model data (anchor)
GAMMA, D0, RHO, Z0, ZBAR = 1.0, 1.0, 1.0, 4.0, 0.0
T_TARGET = np.array([0.8, 1.6, 2.8])            # m = 3 anchor target times
M_TARGET = np.array([0.30, 0.20, 0.35])         # chosen peak weights (schematic choice)
EPS_DRAW = 0.08                                 # noise amplitude used for the drawing
A_AREA = 1.0                                    # W^{d-1} with W = 1
S_Z = math.sqrt(D0 / (2.0 * GAMMA))
THETA = S_Z / RHO

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73"}
COLS = [OI["blue"], OI["vermillion"], OI["green"]]
INK, MUTED = "#222222", "#6b6b6b"
FS = 8.0          # smallest text (pt at the printed size)
FS_TITLE = 9.0

CM = 1.0 / 2.54
FIG_W, FIG_H = 13.0 * CM, 5.0 * CM
DPI = 300


def mu(t):
    return ZBAR + (Z0 - ZBAR) * np.exp(-GAMMA * t)


def std_phi(u):
    return np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)


def std_Phi(u):
    return 0.5 * (1.0 + np.vectorize(math.erf)(u / math.sqrt(2.0)))


def design():
    """Closed-form inverse of the product law and the budget identity."""
    pi = 1.0 - np.concatenate([[0.0], np.cumsum(M_TARGET)[:-1]])
    lam = -np.log(1.0 - M_TARGET / pi)
    z = mu(T_TARGET)
    v = GAMMA * (z - ZBAR)
    bw = A_AREA * v * lam
    budget = float(bw.sum())
    return dict(pi=pi, lam=lam, z=z, v=v, bw=bw, B=budget, w=bw / budget)


def passage_profile(lam, y):
    """F_lambda(y) = (p_lambda * phi_theta)(y) on a uniform grid y (numerical convolution)."""
    du = y[1] - y[0]
    p = lam * std_phi(y) * np.exp(-lam * std_Phi(y))
    k = std_phi(y / THETA) / THETA
    F = np.convolve(p, k, mode="same") * du
    return F


def limit_density(t, d):
    f = np.zeros_like(t)
    parts = []
    y = np.linspace(-14.0, 14.0, 5601)
    for j in range(3):
        F = passage_profile(d["lam"][j], y)
        s = d["v"][j] * (t - T_TARGET[j]) / (EPS_DRAW * RHO)
        fj = d["pi"][j] * (d["v"][j] / (EPS_DRAW * RHO)) * np.interp(s, y, F, left=0.0, right=0.0)
        parts.append(fj)
        f += fj
    return f, parts


def main():
    d = design()
    t = np.linspace(0.0, 3.8, 7601)
    f, parts = limit_density(t, d)
    dt = t[1] - t[0]
    trap = getattr(np, "trapezoid", None) or np.trapz
    areas = [float(trap(p, dx=dt)) for p in parts]
    peak_t = [float(t[np.argmax(p)]) for p in parts]
    # checks: areas equal the chosen weights; peaks come before the crossings
    for j in range(3):
        assert abs(areas[j] - M_TARGET[j]) < 2e-3, (j, areas[j])
        assert peak_t[j] < T_TARGET[j], (j, peak_t[j])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    # panel boxes in figure fractions: [left, bottom, width, height]
    ax1 = fig.add_axes([0.055, 0.285, 0.275, 0.53])
    ax2 = fig.add_axes([0.415, 0.285, 0.165, 0.53])
    ax3 = fig.add_axes([0.665, 0.285, 0.320, 0.53])

    # ---------------- (1) stripes along the path
    band = 2.0 * EPS_DRAW * S_Z
    ax1.fill_between(t, mu(t) - band, mu(t) + band, color="#d9d9d9", lw=0, zorder=1)
    for j in range(3):
        zc = d["z"][j]
        zz = np.linspace(zc - 3 * EPS_DRAW * RHO, zc + 3 * EPS_DRAW * RHO, 60)
        prof = std_phi((zz - zc) / (EPS_DRAW * RHO))
        alpha = 0.25 + 0.6 * d["w"][j] / d["w"].max()
        for k in range(len(zz) - 1):
            ax1.axhspan(zz[k], zz[k + 1], color=COLS[j], alpha=alpha * prof[k] / prof.max(),
                        lw=0, zorder=0)
        ax1.plot([T_TARGET[j]], [zc], "o", ms=3.2, color=COLS[j], mec="white", mew=0.6, zorder=4)
        ax1.text(3.78, zc + 0.10, f"{j + 1}", color=INK, fontsize=FS, ha="right", va="bottom")
    ax1.plot(t, mu(t), color=INK, lw=1.2, zorder=3)
    ax1.text(0.30, 3.60, "pair centre", fontsize=FS, color=INK, ha="left", va="center")
    ax1.set_xlim(0.0, 3.8)
    ax1.set_ylim(-0.1, 4.2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("time", labelpad=2)
    ax1.set_ylabel("position", labelpad=2)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.set_title("Catalyst stripes\nalong the path", fontsize=FS_TITLE, pad=4, color=INK,
                  linespacing=1.0)

    # ---------------- (2) split a fixed budget
    x = np.arange(3)
    ax2.bar(x, d["w"], width=0.68, color=COLS, edgecolor="white", linewidth=1.0)
    for j in range(3):
        ax2.text(x[j], d["w"][j] + 0.02, f"{100 * d['w'][j]:.0f}%", ha="center", va="bottom",
                 fontsize=FS, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["1", "2", "3"])
    ax2.tick_params(axis="x", length=0, pad=2)
    ax2.set_yticks([])
    ax2.set_ylim(0, 0.72)
    ax2.set_xlim(-0.55, 2.55)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.set_xlabel("stripe", labelpad=2)
    ax2.set_title("Split a fixed\nbudget $B$", fontsize=FS_TITLE, pad=4, color=INK, linespacing=1.0)

    # ---------------- (3) programmed peaks
    for j in range(3):
        ax3.fill_between(t, 0, parts[j], color=COLS[j], alpha=0.85, lw=0, zorder=2)
    ax3.plot(t, f, color=INK, lw=0.8, zorder=3)
    fmax = float(f.max())
    label_xy = [(peak_t[0] + 0.12, 0.80 * fmax, "left"), (peak_t[1], 0.39 * fmax, "center"),
                (peak_t[2] + 0.12, 0.22 * fmax, "center")]
    for j in range(3):
        x_, y_, ha_ = label_xy[j]
        ax3.text(x_, y_, f"$M_{j + 1}={M_TARGET[j]:.2f}$", fontsize=FS, color=INK,
                 ha=ha_, va="bottom")
    ax3.set_xlim(0.0, 3.8)
    ax3.set_ylim(0, 1.08 * fmax)
    ax3.set_xticks(T_TARGET)
    ax3.set_xticklabels(["$t_1$", "$t_2$", "$t_3$"])
    ax3.tick_params(axis="x", length=2.5, pad=1.5)
    ax3.set_yticks([])
    ax3.set_xlabel("reaction time", labelpad=1)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.set_title("Programmed peaks\nwith weights $M_j$", fontsize=FS_TITLE, pad=4, color=INK,
                  linespacing=1.0)

    # ---------------- arrows between the steps and the design law
    for x0, x1 in ((0.342, 0.402), (0.592, 0.652)):
        fig.patches.append(FancyArrowPatch((x0, 0.55), (x1, 0.55), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=10, lw=1.2, color=INK))
    fig.text(0.5, 0.025,
             r"$M_j=\mathrm{e}^{-(\lambda_1+\cdots+\lambda_{j-1})}\,(1-\mathrm{e}^{-\lambda_j})$,"
             r"   $\lambda_j=B\,w_j/(A\,v_j)$   ($v_j$: crossing speed; $A$: transverse area)",  # R3 fix (O36-3): A defined
             ha="center", va="bottom", fontsize=FS, color=INK)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    png = OUT_DIR / f"{STEM}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=DPI)
    plt.close(fig)

    record = {
        "script": "code/fb_graphical_abstract.py",
        "outputs": [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))],
        "size_cm": [13.0, 5.0], "dpi": DPI, "min_font_pt": FS,
        "target_times": T_TARGET.tolist(), "chosen_weights": M_TARGET.tolist(),
        "lambda": d["lam"].tolist(), "crossing_speeds": d["v"].tolist(),
        "budget_B": d["B"], "weights_w": d["w"].tolist(),
        "eps_draw": EPS_DRAW, "peak_areas_check": areas, "peak_times": peak_t,
    }
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
