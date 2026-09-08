#!/usr/bin/env python3
"""W4: m=5 demonstration in a stretched window (z0 = 8, d = 2).

Design (fixed by the semi-analytic preflight, candidate A):
  z0 = 8 (stretches the drift range so five separated slabs fit in I),
  physical slab centres mu_j = {2.8, 2.2, 1.6, 1.0, 0.4}
    -> target times t_j = ln(z0/mu_j) = {1.050, 1.291, 1.609, 2.079, 2.996},
  eps = 0.1, B = 1, uniform weights, adjacent centre spacing 0.6 = 4.9 sigma
  (sigma = eps sqrt(D0/(2 gamma) + rho^2) = 0.1225).
  Preflight: G has 5 window maxima; first-order depleted density
  B G e^{-B Lambda} keeps the last peak's prominence at 43% of max.

Gates on this new config type (z0 != 4, explicit centres):
  (1) B=0 zero kills (contact-sampling start), (2) per-step prob in [0,1],
  (3) mass balance, plus the spine contact-interior condition
  sup_{t in I} |r_*(t)|_mi <= a - eta checked deterministically.

Outputs: artifacts/data/exact_m_prr_upgrade/w4_m5_demo/m5_demo.json,
         artifacts/figures/exact_m_m5_demo_prr.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import validate_exact_m_offlattice as base

W4_DIR = core.UPGRADE_DATA / "w4_m5_demo"
Z0 = 8.0
MU_CENTRES = (2.8, 2.2, 1.6, 1.0, 0.4)
M = 5
EPS = 0.10
BUDGET = 1.0
WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)
WALKERS = 5_000_000
CHUNK = 100_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH
SEED = core.CAMPAIGN_SEED


def theory_curve() -> tuple[np.ndarray, np.ndarray]:
    """Semi-analytic free-exposure overlay B*G(t) (no simulation).

    The grid starts at t=0 so that the cumulative exposure Lambda(t) used by
    the mean-field overlay implements the lower limit of Eq. (mean-field).
    """
    p4 = core.make_model(z0=Z0)
    ts = np.linspace(0.0, TMAX, 1600)
    theory = core.free_exposure_general(
        ts,
        centres_z=np.array(MU_CENTRES, dtype=float),
        eps=EPS,
        weights=WEIGHTS,
        p=p4,
        n_perp=1,
    )
    return ts, theory["g"]


def make_figure(payload: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt

    summary = payload["results"]
    classifier = summary["classifier"]
    walkers = payload["parameters"]["config"]["walkers"]
    target_times = payload["parameters"]["config"]["target_times"]
    ts, g_vals = theory_curve()

    edges = np.asarray(classifier["edges"])
    centres_t = 0.5 * (edges[:-1] + edges[1:])
    counts = np.asarray(classifier["counts"], dtype=float)
    density = counts / (walkers * classifier["bin_width"])
    smoothed = np.asarray(classifier["smoothed_density"])

    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.stairs(density, edges, fill=True, alpha=0.30, color="#4878b0",
              label="EM histogram (window)")
    ax.plot(centres_t, smoothed, color="#20415f", lw=1.4,
            label=f"smoothed ($h={classifier['bandwidth']:g}$)")
    ax.plot(ts, BUDGET * g_vals, color=core.OI_VERMILLION, lw=1.2, ls="--",
            label=r"$B\,G(t)$ free exposure")
    # Mean-field hazard--survival law f1 = B G exp(-B Lambda), Lambda = int_0^t G
    # (deterministic; trapezoid on the theory grid; no simulation input).
    B = float(payload["parameters"]["config"]["budget"])
    lam = np.concatenate(
        ([0.0], np.cumsum(0.5 * (g_vals[1:] + g_vals[:-1]) * np.diff(ts)))
    )
    f1 = B * g_vals * np.exp(-B * lam)
    ax.plot(ts, f1, linestyle="-.", color="#009E73", linewidth=1.2,
            label=r"$f_1=B\,G\,\mathrm{e}^{-B\Lambda}$ mean field")
    for tj in target_times:
        ax.axvline(tj, color="0.45", lw=0.7, ls=":")
    # Peak labels: time only (prominences are quoted in the caption);
    # staggered vertical offsets keep the close early peaks collision-free.
    label_offsets = (8, 17, 8, 8, 8)
    for i, row in enumerate(classifier["significant_maxima"]):
        ax.plot(row["time"], row["smoothed_height"], marker="v",
                color="#2ca02c", ms=6, ls="none",
                label="classifier-significant maximum" if i == 0 else None)
        ax.annotate(
            f"{row['time']:.2f}",
            (row["time"], row["smoothed_height"]),
            textcoords="offset points",
            xytext=(0, label_offsets[i % len(label_offsets)]),
            ha="center", fontsize=7.5, color="#207020",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none",
                  "pad": 0.4},
        )
    ax.set_xlim(0.0, TMAX)
    ax.set_xlabel("reaction time $t$")
    ax.set_ylabel("density per walker [$1/t$]")
    ax.legend(fontsize=7.5, loc="upper right")
    note = (
        f"modes(classifier)={summary['mode_count']} (target $m=5$)\n"
        f"window events={summary['kills_in_window']:,} "
        f"({summary['event_fraction_in_window']:.3f}/walker)"
    )
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, bbox={"facecolor": "white", "alpha": 0.8,
                                "edgecolor": "none"})
    stem = core.FIGURES / "exact_m_m5_demo_prr"
    written = core.save_figure(fig, stem)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkers", type=lambda v: int(float(v)), default=WALKERS)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="regenerate the figure from the stored JSON (no simulation)",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    if args.replot:
        payload = json.loads(
            (W4_DIR / "m5_demo.json").read_text(encoding="utf-8")
        )
        for path in make_figure(payload):
            print(f"figure -> {path}", flush=True)
        return

    p4 = core.make_model(z0=Z0)
    target_times = tuple(float(math.log(Z0 / mu)) for mu in MU_CENTRES)
    centres = np.array(MU_CENTRES, dtype=float)

    # Spine contact-interior condition (deterministic relative trajectory).
    contact_check = core.deterministic_contact_check(p4)
    assert contact_check["passed"], contact_check
    print(f"contact-interior check: {contact_check}", flush=True)

    # Preflight theory: G must have exactly 5 window maxima.
    ts = np.linspace(base.THEORY_T_MIN, TMAX, 1600)
    theory = core.free_exposure_general(
        ts, centres_z=centres, eps=EPS, weights=WEIGHTS, p=p4, n_perp=1
    )
    g_diag = core.g_stationary_diagnostics(ts, theory["g"])
    assert g_diag["n_window_maxima"] == M, g_diag["n_window_maxima"]
    print(f"G theory: {g_diag['n_window_maxima']} window maxima", flush=True)

    # Gate (1) on this code path.
    gate_zero = core.gate_zero_general(
        SEED,
        dt=DT,
        p=p4,
        n_perp=1,
        tag=core.TAG_W4_GATE,
        centres_z=centres,
        weights=WEIGHTS,
    )
    print(f"gate B=0 PASS: contact_steps={gate_zero['contact_steps']}", flush=True)

    result = core.run_config_general(
        eps=EPS,
        budget=BUDGET,
        weights=WEIGHTS,
        centres_z=centres,
        walkers=args.walkers,
        chunk=CHUNK,
        dt=DT,
        tmax=TMAX,
        seed=SEED,
        tag=core.TAG_W4_MAIN,
        p=p4,
        n_perp=1,
        verbose=True,
    )
    summary = core.summarize_general(
        result, bandwidth=BANDWIDTH, target_times=target_times, m=M
    )

    payload = {
        "parameters": {
            "stream": "w4_m5_demo",
            "model_parameters": core.model_dict(p4),
            "design_note": (
                "stretched window via z0=4->8; physical centres mu_j chosen "
                "equally spaced (0.6 = 4.9 sigma) in the drift range so five "
                "slabs fit inside I with the last centre at mu=0.4; "
                "t_j = ln(z0/mu_j); preflight (preflight_theory.json, "
                "candidate A) confirms 5 G window maxima and 43% last-peak "
                "prominence after first-order depletion at B=1"
            ),
            "config": {
                "m": M,
                "eps": EPS,
                "budget": BUDGET,
                "weights": list(WEIGHTS),
                "mu_centres": list(MU_CENTRES),
                "target_times": list(target_times),
                "walkers": args.walkers,
                "chunk": CHUNK,
                "dt": DT,
                "tmax": TMAX,
                "seed": SEED,
                "tag": core.TAG_W4_MAIN,
                "bandwidth": BANDWIDTH,
                "window": list(core.WINDOW),
            },
            "rng": "numpy Philox, SeedSequence-spawned substream per chunk",
            "killing": "per-step probability 1 - exp(-kappa dt) via expm1",
        },
        "validation_gates": {
            "gate1_zero_budget_no_kills": gate_zero,
            "gate2_kill_probability_max": summary["kill_probability_max"],
            "gate2_kill_probability_in_unit_interval": bool(
                0.0 <= summary["kill_probability_max"] <= 1.0
            ),
            "gate3_mass_balance": {
                "kills": summary["kills"],
                "survivors": summary["survivors"],
                "walkers": args.walkers,
                "passed": bool(
                    summary["kills"] + summary["survivors"] == args.walkers
                ),
            },
            "contact_interior_condition": contact_check,
        },
        "g_theory_diagnostics": g_diag,
        "results": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    out = W4_DIR / "m5_demo.json"
    core.write_json(out, payload)
    peaks = ", ".join(f"{t:.2f}" for t in summary["peak_times"])
    print(
        f"W4 m=5: modes={summary['mode_count']} "
        f"({'PASS' if summary['mode_count'] == M else 'FAIL'}) peaks=[{peaks}] "
        f"killfrac={summary['kill_fraction']:.4f} "
        f"winfrac={summary['event_fraction_in_window']:.4f} -> {out}",
        flush=True,
    )

    for path in make_figure(payload):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
