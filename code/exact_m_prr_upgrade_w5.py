#!/usr/bin/env python3
"""W5: d = 3 spot check (two wrapped transverse coordinates).

Config: m=2, eps=0.1, B=1, weights 50/50, standard centres (z0=4), with the
relative transverse coordinate R_perp in T_W^2 and killing prefactor
B / W^(d-1) = B / W^2 per the spine.  Sigma_perp0 = sigma_perp0^2 I (isotropic).

Event-rate decision: the semi-analytic preflight (preflight_theory.json) gives
total exposure Lambda_d3 = 1.20 (vs 1.68 in d=2) and a first-order kill
fraction ~0.70 at B=1, so the contact/event rate stays usable WITHOUT
adjusting a or B; both are left at their d=2 values (a=0.4, B=1) and the
measured contact fraction is recorded below.

Gates on this new code path: (1) B=0 zero kills, (2) per-step prob in [0,1],
(3) mass balance, plus the dt-halving gate (dt vs dt/2 window histograms in
Monte Carlo sigmas, reported not asserted) run once for the d=3 path.

Outputs: artifacts/data/exact_m_prr_upgrade/w5_d3_spotcheck/d3_spotcheck.json,
         artifacts/figures/exact_m_d3_spotcheck_prr.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import validate_exact_m_offlattice as base

W5_DIR = core.UPGRADE_DATA / "w5_d3_spotcheck"
M = 2
EPS = 0.10
BUDGET = 1.0
WEIGHTS = (0.5, 0.5)
WALKERS = 5_000_000
DT_CHECK_WALKERS = 200_000
CHUNK = 100_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH
SEED = core.CAMPAIGN_SEED
N_PERP = 2


def dt_halving_check_d3(centres: np.ndarray, p) -> dict:
    edges = np.arange(
        core.WINDOW[0], core.WINDOW[1] + 0.5 * base.DT_CHECK_BIN, base.DT_CHECK_BIN
    )
    runs = {}
    for label, step in (("dt", DT), ("dt_half", 0.5 * DT)):
        outcome = core.run_config_general(
            eps=EPS,
            budget=BUDGET,
            weights=WEIGHTS,
            centres_z=centres,
            walkers=DT_CHECK_WALKERS,
            chunk=CHUNK,
            dt=step,
            tmax=TMAX,
            seed=SEED,
            tag=core.TAG_W5_DTCHECK,
            p=p,
            n_perp=N_PERP,
        )
        counts, _ = np.histogram(outcome["kill_times"], bins=edges)
        runs[label] = {
            "dt": step,
            "counts": counts,
            "kill_fraction": outcome["kill_times"].size / DT_CHECK_WALKERS,
            "runtime_seconds": outcome["runtime_seconds"],
        }
    counts_a = runs["dt"]["counts"]
    counts_b = runs["dt_half"]["counts"]
    walkers = DT_CHECK_WALKERS
    fraction_a = counts_a / walkers
    fraction_b = counts_b / walkers
    pooled = (counts_a + counts_b) / (2.0 * walkers)
    variance = pooled * (1.0 - pooled) * (2.0 / walkers)
    usable = (counts_a + counts_b) >= base.DT_CHECK_MIN_POOLED
    z_scores = np.zeros(edges.size - 1)
    z_scores[usable] = (fraction_a[usable] - fraction_b[usable]) / np.sqrt(
        variance[usable]
    )
    z_used = z_scores[usable]
    return {
        "config": {"m": M, "eps": EPS, "budget": BUDGET, "weights": list(WEIGHTS),
                   "n_perp": N_PERP},
        "walkers_per_run": walkers,
        "window": list(core.WINDOW),
        "bin_width": base.DT_CHECK_BIN,
        "counts_dt": [int(v) for v in counts_a],
        "counts_dt_half": [int(v) for v in counts_b],
        "kill_fraction_dt": runs["dt"]["kill_fraction"],
        "kill_fraction_dt_half": runs["dt_half"]["kill_fraction"],
        "usable_bins": int(usable.sum()),
        "excluded_bins": int((~usable).sum()),
        "max_abs_z": float(np.max(np.abs(z_used))) if z_used.size else None,
        "mean_abs_z": float(np.mean(np.abs(z_used))) if z_used.size else None,
        "fraction_abs_z_above_2": (
            float(np.mean(np.abs(z_used) > 2.0)) if z_used.size else None
        ),
        "note": (
            "under the no-bias null the per-bin z are approximately standard "
            "normal; with ~30 usable bins the expected max|z| is about 2.5"
        ),
    }


def theory_curve() -> tuple[np.ndarray, np.ndarray]:
    """Semi-analytic d=3 free-exposure overlay B*G(t) (quadrature only).

    The grid starts at t=0 so that the cumulative exposure Lambda(t) used by
    the mean-field overlay implements the lower limit of Eq. (mean-field).
    """
    p5 = core.make_model(dim=3)
    centres = core.centres_z_for(M, p=p5)
    ts = np.linspace(0.0, TMAX, core.D3_T_POINTS)
    theory = core.free_exposure_general(
        ts, centres_z=centres, eps=EPS, weights=WEIGHTS, p=p5, n_perp=N_PERP
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
    contact_fraction_per_step = summary["contact_fraction_per_walker_step"]
    ts, g_vals = theory_curve()

    edges = np.asarray(classifier["edges"])
    centres_t = 0.5 * (edges[:-1] + edges[1:])
    counts = np.asarray(classifier["counts"], dtype=float)
    density = counts / (walkers * classifier["bin_width"])
    smoothed = np.asarray(classifier["smoothed_density"])

    fig, ax = plt.subplots(figsize=(5.3, 3.3), constrained_layout=True)
    ax.stairs(density, edges, fill=True, alpha=0.30, color="#4878b0",
              label="EM histogram (window)")
    ax.plot(centres_t, smoothed, color="#20415f", lw=1.4,
            label=f"smoothed ($h={classifier['bandwidth']:g}$)")
    ax.plot(ts, BUDGET * g_vals, color=core.OI_VERMILLION, lw=1.2, ls="--",
            label=r"$B\,G(t)$, $d=3$ contact factor")
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
    # Info note drawn before the peak labels so the labels stay on top of
    # its white backing where the first peak reaches into the note box.
    note = (
        f"modes(classifier)={summary['mode_count']} (target $m=2$), $d=3$\n"
        f"window events={summary['kills_in_window']:,} "
        f"({summary['event_fraction_in_window']:.3f}/walker)\n"
        f"contact fraction/step={contact_fraction_per_step:.4f}"
    )
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, bbox={"facecolor": "white", "alpha": 0.8,
                                "edgecolor": "none"})
    for i, row in enumerate(classifier["significant_maxima"]):
        ax.plot(row["time"], row["smoothed_height"], marker="v",
                color="#2ca02c", ms=6, ls="none",
                label="classifier-significant maximum" if i == 0 else None)
        ax.annotate(
            f"{row['time']:.2f}",
            (row["time"], row["smoothed_height"]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color="#207020",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none",
                  "pad": 0.4},
        )
    ax.set_xlim(0.0, TMAX)
    ax.set_ylim(0.0, 1.32)
    ax.set_xlabel("reaction time $t$")
    ax.set_ylabel("density per walker [$1/t$]")
    ax.legend(fontsize=7.5, loc="upper right")
    stem = core.FIGURES / "exact_m_d3_spotcheck_prr"
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
            (W5_DIR / "d3_spotcheck.json").read_text(encoding="utf-8")
        )
        for path in make_figure(payload):
            print(f"figure -> {path}", flush=True)
        return

    p5 = core.make_model(dim=3)
    centres = core.centres_z_for(M, p=p5)
    target_times = base.TARGET_TIMES[M]

    gate_zero = core.gate_zero_general(
        SEED,
        dt=DT,
        p=p5,
        n_perp=N_PERP,
        tag=core.TAG_W5_GATE,
        centres_z=centres,
        weights=WEIGHTS,
    )
    print(f"gate B=0 (d=3) PASS: contact_steps={gate_zero['contact_steps']}",
          flush=True)

    print("dt-halving gate on the d=3 code path ...", flush=True)
    dt_check = dt_halving_check_d3(centres, p5)
    print(
        f"dt-halving (d=3): max|z|={dt_check['max_abs_z']:.2f} "
        f"mean|z|={dt_check['mean_abs_z']:.2f} "
        f"frac|z|>2={dt_check['fraction_abs_z_above_2']:.3f} "
        f"({dt_check['usable_bins']} usable bins)",
        flush=True,
    )

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
        tag=core.TAG_W5_MAIN,
        p=p5,
        n_perp=N_PERP,
        verbose=True,
    )
    summary = core.summarize_general(
        result, bandwidth=BANDWIDTH, target_times=target_times, m=M
    )

    # Theory overlay (d=3 contact factor via 2D quadrature).
    ts = np.linspace(base.THEORY_T_MIN, TMAX, core.D3_T_POINTS)
    theory = core.free_exposure_general(
        ts, centres_z=centres, eps=EPS, weights=WEIGHTS, p=p5, n_perp=N_PERP
    )
    g_diag = core.g_stationary_diagnostics(ts, theory["g"])

    contact_fraction_per_step = (
        summary["contact_steps"] / summary["walker_steps"]
        if summary["walker_steps"]
        else None
    )
    payload = {
        "parameters": {
            "stream": "w5_d3_spotcheck",
            "model_parameters": core.model_dict(p5),
            "design_note": (
                "d=3: relative transverse coordinate on T_W^2 (two wrapped "
                "coordinates, isotropic initial wrapped normal), killing "
                "prefactor B/W^(d-1) = B/W^2; a=0.4 and B=1 kept at their d=2 "
                "values because the preflight gives Lambda_d3=1.20 and a "
                "first-order kill fraction ~0.70 at B=1, so no event-rate "
                "adjustment is needed; measured contact/event rates recorded"
            ),
            "config": {
                "m": M,
                "eps": EPS,
                "budget": BUDGET,
                "weights": list(WEIGHTS),
                "target_times": list(target_times),
                "n_perp": N_PERP,
                "dim": 3,
                "walkers": args.walkers,
                "chunk": CHUNK,
                "dt": DT,
                "tmax": TMAX,
                "seed": SEED,
                "tag": core.TAG_W5_MAIN,
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
            "gate4_dt_halving_d3": dt_check,
        },
        "g_theory_diagnostics": g_diag,
        "results": {
            **summary,
            "contact_fraction_per_walker_step": contact_fraction_per_step,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    out = W5_DIR / "d3_spotcheck.json"
    core.write_json(out, payload)
    peaks = ", ".join(f"{t:.2f}" for t in summary["peak_times"])
    print(
        f"W5 d=3: modes={summary['mode_count']} "
        f"({'PASS' if summary['mode_count'] == M else 'FAIL'}) peaks=[{peaks}] "
        f"killfrac={summary['kill_fraction']:.4f} "
        f"winfrac={summary['event_fraction_in_window']:.4f} -> {out}",
        flush=True,
    )

    for path in make_figure(payload):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
