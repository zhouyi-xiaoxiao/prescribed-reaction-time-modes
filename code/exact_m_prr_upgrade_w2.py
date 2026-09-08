#!/usr/bin/env python3
"""W2: protocol-defined operational budget threshold B_op(eps) by bisection.

For each (m, eps) chain -- m=2 on eps in {0.05,0.10,0.15,0.20}, m=3 on the
subset where the semi-analytic G retains 3 window maxima -- bisect B for the
threshold where the LAST mode stops satisfying the classifier significance
rule (prominence >= 5 smoothed-Poisson sigma AND >= 5% of max smoothed
height) at 1e6 walkers/probe.

Pass rule: a classifier-significant maximum exists in the last mode's basin,
i.e. at a time later than the last valley of the semi-analytic G (stored per
chain).  Bracketing starts from the W1 grid verdicts at the same walker count
(largest passing B, first failing B above it; B_hi = 8 if the whole W1 column
passes), followed by BISECT_ITERS geometric bisections (~2% final resolution
for a 4x bracket).

Per-chain theory diagnostics stored for later comparison with a theoretical
lower bound (no rerun needed): G valley depth at the relevant (last)
crossover, curvature margins at all window stationary points, cumulative
exposure integrals Lambda(t) at the stationary points, sigma, centre spacings.

Outputs: artifacts/data/exact_m_prr_upgrade/w2_b0_empirical/probe_*.json,
         B0_empirical.json, artifacts/figures/exact_m_b0_empirical_prr.{png,pdf}.

The ``b0`` directory/file/field spellings are frozen legacy schema names.  In
this script they denote only B_op under the declared finite-walker classifier
protocol; they do not denote the theorem threshold B_top or the sufficient
certificate B_cert.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import validate_exact_m_offlattice as base

W1_DIR = core.UPGRADE_DATA / "w1_phase_diagram"
W2_DIR = core.UPGRADE_DATA / "w2_b0_empirical"
EPS_SET = (0.05, 0.10, 0.15, 0.20)
M_WEIGHTS = {2: (0.5, 0.5), 3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)}
PROBE_WALKERS = 1_000_000
CHUNK = 100_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH
SEED = core.CAMPAIGN_SEED
B_MAX = 8.0
BISECT_ITERS = 6


def chain_theory(m: int, eps: float) -> dict:
    diag = core.g_window_maxima_count(m=m, eps=eps, weights=M_WEIGHTS[m])
    valleys = diag["valleys"]
    diag["last_valley_time"] = valleys[-1]["time"] if valleys else None
    spacings = np.abs(np.diff(core.centres_z_for(m)))
    diag["centre_spacings_z"] = [float(v) for v in spacings]
    return diag


def probe(m: int, eps: float, budget: float, basin_edge: float) -> dict:
    """One 1e6-walker probe; verdict = last mode classifier-significant."""
    weights = M_WEIGHTS[m]
    result = base.run_config(
        m=m,
        eps=eps,
        budget=budget,
        weights=weights,
        walkers=PROBE_WALKERS,
        chunk=CHUNK,
        dt=DT,
        tmax=TMAX,
        seed=SEED,
        tag=core.TAG_W2_PROBE,
        verbose=False,
    )
    summary = base.summarize_config(result, bandwidth=BANDWIDTH, tmax=TMAX)
    last_mode_rows = [
        row
        for row in summary["classifier"]["significant_maxima"]
        if row["time"] > basin_edge
    ]
    verdict = bool(last_mode_rows)
    payload = {
        "parameters": {
            "stream": "w2_b0_empirical",
            "config": {
                "m": m,
                "eps": eps,
                "budget": budget,
                "weights": list(weights),
                "walkers": PROBE_WALKERS,
                "dt": DT,
                "tmax": TMAX,
                "seed": SEED,
                "tag": core.TAG_W2_PROBE,
                "bandwidth": BANDWIDTH,
            },
            "pass_rule": (
                "significant maximum with time > last G valley time "
                f"({basin_edge:.4f})"
            ),
        },
        "validation_gates": {
            "gate2_kill_probability_max": summary["kill_probability_max"],
            "gate3_mass_balance_passed": bool(
                summary["kills"] + summary["survivors"] == PROBE_WALKERS
            ),
        },
        "results": {
            **{
                k: summary[k]
                for k in (
                    "kills",
                    "kill_fraction",
                    "kills_in_window",
                    "event_fraction_in_window",
                    "mode_count",
                    "peak_times",
                    "prominences",
                    "runtime_seconds",
                )
            },
            "classifier_significant_maxima": summary["classifier"][
                "significant_maxima"
            ],
            # Retain the complete binned sufficient statistics so every
            # threshold probe can be reclassified under declared alternative
            # smoothing/prominence choices without another Monte Carlo run.
            "classifier": summary["classifier"],
            "histograms": summary["histograms"],
            "last_mode_present": verdict,
            "last_mode_rows": last_mode_rows,
        },
    }
    path = W2_DIR / f"probe_m{m}_eps{eps:g}_B{budget:.6g}.json"
    core.write_json(path, payload)
    return {
        "budget": budget,
        "verdict": verdict,
        "mode_count": summary["mode_count"],
        "peak_times": summary["peak_times"],
        "runtime_seconds": summary["runtime_seconds"],
        "file": path.name,
    }


def w1_bracket(m: int, eps: float) -> dict:
    """Initial bracket from the W1 grid verdicts (same walker count)."""
    passes = {}
    for path in W1_DIR.glob(f"cell_m{m}_eps{eps:g}_B*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["parameters"]["config"]
        if cfg["refined"]:
            continue
        passes[cfg["budget"]] = payload["results"]["passes_m_mode_criterion"]
    passing = sorted(b for b, ok in passes.items() if ok)
    failing = sorted(b for b, ok in passes.items() if not ok)
    b_lo = max(passing) if passing else None
    b_hi = min([b for b in failing if b_lo is not None and b > b_lo] or [None])
    return {"w1_verdicts": passes, "b_lo": b_lo, "b_hi": b_hi}


def run_chain(task: tuple) -> dict:
    m, eps = task
    started = time.perf_counter()
    theory = chain_theory(m, eps)
    basin_edge = theory["last_valley_time"]
    history: list[dict] = []
    chain: dict = {
        "m": m,
        "eps": eps,
        "weights": list(M_WEIGHTS[m]),
        "probe_walkers": PROBE_WALKERS,
        "basin_edge_last_g_valley_time": basin_edge,
        "theory_diagnostics": theory,
    }
    if not theory["g_has_m_window_maxima"]:
        chain["status"] = "excluded_g_lacks_m_maxima"
        chain["b0"] = None
        return chain

    bracket = w1_bracket(m, eps)
    chain["w1_bracket"] = bracket
    b_lo = bracket["b_lo"]
    if b_lo is None:
        # No passing budget on the W1 column: probe the smallest grid budget
        # once to confirm, then report not-detectable.
        row = probe(m, eps, 0.125, basin_edge)
        history.append(row)
        chain["probes"] = history
        if row["verdict"]:
            b_lo = 0.125
        else:
            chain["status"] = "no_passing_budget_at_1e6_walkers"
            chain["b0"] = None
            chain["wall_seconds"] = time.perf_counter() - started
            return chain
    b_hi = bracket["b_hi"]
    if b_hi is None:
        row = probe(m, eps, B_MAX, basin_edge)
        history.append(row)
        if row["verdict"]:
            chain["status"] = "right_censored"
            chain["b0"] = None
            chain["b0_lower_bound"] = B_MAX
            chain["probes"] = history
            chain["wall_seconds"] = time.perf_counter() - started
            return chain
        b_hi = B_MAX

    for _ in range(BISECT_ITERS):
        mid = math.sqrt(b_lo * b_hi)
        row = probe(m, eps, mid, basin_edge)
        history.append(row)
        if row["verdict"]:
            b_lo = mid
        else:
            b_hi = mid
    chain["status"] = "bisected"
    chain["probes"] = history
    chain["b0_bracket"] = [b_lo, b_hi]
    chain["b0"] = math.sqrt(b_lo * b_hi)
    chain["b0_relative_halfwidth"] = math.sqrt(b_hi / b_lo) - 1.0
    chain["wall_seconds"] = time.perf_counter() - started
    return chain


def make_figure(chains: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(3.40, 2.75), constrained_layout=True)
    colours = {2: core.OI_BLUE, 3: core.OI_VERMILLION}
    censored = False
    for m in sorted(M_WEIGHTS):
        rows = [c for c in chains if c["m"] == m and c.get("b0") is not None]
        if rows:
            xs = [c["eps"] for c in rows]
            ys = [c["b0"] for c in rows]
            ylo = [c["b0"] - c["b0_bracket"][0] for c in rows]
            yhi = [c["b0_bracket"][1] - c["b0"] for c in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=[ylo, yhi],
                marker="o",
                ms=4.5,
                lw=1.2,
                capsize=2.5,
                color=colours[m],
                label=f"$m={m}$ (bisection bracket)",
            )
        # Right-censored chains: open symbols at the scan limit (legend entry
        # below); no in-plot text.
        cens = [
            c
            for c in chains
            if c["m"] == m
            and c.get("b0") is None
            and c.get("status") == "right_censored"
        ]
        if cens:
            censored = True
            ax.plot(
                [c["eps"] for c in cens],
                [c["b0_lower_bound"] for c in cens],
                marker="^",
                ms=6,
                markerfacecolor="none",
                markeredgecolor=colours[m],
                markeredgewidth=1.1,
                ls="none",
            )
    ax.set_yscale("log")
    ax.set_ylim(0.22, 11.5)
    ax.set_yticks(
        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        ["0.25", "0.5", "1", "2", "4", "8"],
    )
    ax.yaxis.set_minor_formatter("")
    ax.set_xlabel(r"slab width parameter $\varepsilon$")
    ax.set_ylabel(r"operational budget threshold $B_{\rm op}(\varepsilon)$")
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if censored:
        handles.append(
            Line2D(
                [], [], marker="^", ms=6, markerfacecolor="none",
                markeredgecolor="0.35", markeredgewidth=1.1, ls="none",
            )
        )
        labels.append(r"open: $B_{\rm op} > 8$ (right-censored)")
    ax.legend(handles, labels, loc="lower left", fontsize=7)
    stem = core.FIGURES / "exact_m_b0_empirical_prr"
    written = core.save_figure(fig, stem)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    out_path = W2_DIR / "B0_empirical.json"
    if args.figure_only:
        chains = json.loads(out_path.read_text(encoding="utf-8"))["chains"]
        for path in make_figure(chains):
            print(f"figure -> {path}", flush=True)
        return

    tasks = [(m, eps) for m in sorted(M_WEIGHTS) for eps in EPS_SET]
    chains = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_chain, task): task for task in tasks}
        for future in as_completed(futures):
            chain = future.result()
            chains.append(chain)
            print(
                f"chain m={chain['m']} eps={chain['eps']:g}: "
                f"status={chain.get('status')} B_op={chain.get('b0')} "
                f"bracket={chain.get('b0_bracket')} "
                f"probes={len(chain.get('probes', []))}",
                flush=True,
            )
    chains.sort(key=lambda c: (c["m"], c["eps"]))
    summary = {
        "legacy_schema_note": (
            "serialized b0 fields and B0_empirical filenames denote the "
            "protocol-defined operational threshold B_op only, not B_top "
            "or B_cert"
        ),
        "stream": "w2_b0_empirical",
        "eps_set": list(EPS_SET),
        "m_weights": {str(m): list(w) for m, w in M_WEIGHTS.items()},
        "probe_walkers": PROBE_WALKERS,
        "bisect_iterations": BISECT_ITERS,
        "b_max": B_MAX,
        "pass_rule": (
            "significant maximum (5 sigma + 5% rule) at time > last "
            "semi-analytic G valley; bracket seeded from W1 grid verdicts "
            "at the same walker count"
        ),
        "model_parameters": core.model_dict(core.MODEL),
        "dt": DT,
        "tmax": TMAX,
        "seed": SEED,
        "chains": chains,
        "wall_seconds": time.perf_counter() - started,
    }
    core.write_json(out_path, summary)
    print(f"summary -> {out_path}", flush=True)
    for path in make_figure(chains):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
