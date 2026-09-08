#!/usr/bin/env python3
"""W3: jitter robustness (deliberately beyond the theorem hypotheses).

Anchors: (m=2, eps=0.1, B=1, w=(0.5,0.5)) and (m=3, eps=0.1, B=1, uniform),
standard model (z0=4, d=2), target times from the validated grid.

Centre jitter: slab centres zeta_j = mu(t_j) + delta_j with delta_j iid
N(0, (eta s)^2), s = min adjacent centre spacing |mu(t_j) - mu(t_j+1)|,
eta in {0.05, 0.1, 0.15, 0.2, 0.3, 0.4}; REPLICAS replicas x 2e5 walkers per
point (window events per replica ~1.5e5 at the anchors, comfortably enough
for the 5 sigma + 5% classifier).

Weight jitter: one sweep with w ~ Dirichlet(conc * w_anchor), conc = 50
(moderate concentration; component sd ~ w_j sqrt((1-w_j)/ (w_j (conc+1)))),
centres fixed at the anchor, REPLICAS replicas per anchor.

Success = classifier mode count == m (exactly-m modality).  Survival
probabilities carry Wilson 95% binomial CIs.  Each replica also stores the
semi-analytic G verdict for its perturbed configuration (does G retain m
window maxima), so classifier-vs-theory agreement can be assessed offline.

Gates: gate (1) B=0 zero kills on the explicit-centres code path once;
gates (2)-(3) asserted inside every run.

Outputs: artifacts/data/exact_m_prr_upgrade/w3_jitter/point_*.json,
         jitter_robustness.json,
         artifacts/figures/exact_m_jitter_robustness_prr.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import validate_exact_m_offlattice as base

W3_DIR = core.UPGRADE_DATA / "w3_jitter"
ANCHORS = {
    2: {"eps": 0.10, "budget": 1.0, "weights": (0.5, 0.5)},
    3: {"eps": 0.10, "budget": 1.0, "weights": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)},
}
ETA_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
REPLICAS = 50
REPLICA_WALKERS = 200_000
DIRICHLET_CONC = 50.0
CHUNK = 100_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH
SEED = core.CAMPAIGN_SEED
THEORY_POINTS = 800


def min_adjacent_spacing(m: int) -> float:
    return float(np.min(np.abs(np.diff(core.centres_z_for(m)))))


def replica_task(task: tuple) -> dict:
    """One replica: draw perturbation, simulate, classify, theory verdict."""
    kind, m, eta_index, replica = task
    anchor = ANCHORS[m]
    eps, budget = anchor["eps"], anchor["budget"]
    centres0 = core.centres_z_for(m)
    weights = np.asarray(anchor["weights"], dtype=float)

    if kind == "centre":
        eta = ETA_GRID[eta_index]
        s = min_adjacent_spacing(m)
        draw_rng = np.random.Generator(
            np.random.Philox(
                np.random.SeedSequence(
                    [SEED, core.TAG_W3_DRAWS, 1, m, eta_index, replica]
                )
            )
        )
        centres = centres0 + eta * s * draw_rng.standard_normal(centres0.size)
        tag = core.TAG_W3_CENTRE
        extra = (1, eta_index, replica)
    else:  # weight jitter
        eta = None
        draw_rng = np.random.Generator(
            np.random.Philox(
                np.random.SeedSequence(
                    [SEED, core.TAG_W3_DRAWS, 2, m, 0, replica]
                )
            )
        )
        centres = centres0.copy()
        weights = draw_rng.dirichlet(DIRICHLET_CONC * weights)
        tag = core.TAG_W3_WEIGHT
        extra = (2, 0, replica)

    result = core.run_config_general(
        eps=eps,
        budget=budget,
        weights=tuple(float(w) for w in weights),
        centres_z=centres,
        walkers=REPLICA_WALKERS,
        chunk=CHUNK,
        dt=DT,
        tmax=TMAX,
        seed=SEED,
        tag=tag,
        n_perp=1,
        extra_entropy=extra,
    )
    times = result["kill_times"]
    classifier = base.classify_modes(times, REPLICA_WALKERS, bandwidth=BANDWIDTH)
    window_mask = (times >= core.WINDOW[0]) & (times <= core.WINDOW[1])

    ts = np.linspace(base.THEORY_T_MIN, TMAX, THEORY_POINTS)
    theory = core.free_exposure_general(
        ts, centres_z=centres, eps=eps, weights=tuple(weights), n_perp=1
    )
    g_diag = core.g_stationary_diagnostics(ts, theory["g"])

    return {
        "kind": kind,
        "m": m,
        "eta_index": eta_index,
        "eta": eta,
        "replica": replica,
        "centres_z": [float(c) for c in centres],
        "weights": [float(w) for w in weights],
        "mode_count": classifier["mode_count"],
        "success": bool(classifier["mode_count"] == m),
        "peak_times": [r["time"] for r in classifier["significant_maxima"]],
        "kills_in_window": int(window_mask.sum()),
        "kill_fraction": float(times.size / REPLICA_WALKERS),
        "kill_probability_max": result["kill_probability_max"],
        "g_n_window_maxima": g_diag["n_window_maxima"],
        "g_has_m_window_maxima": bool(g_diag["n_window_maxima"] == m),
        "runtime_seconds": result["runtime_seconds"],
    }


def aggregate_point(kind: str, m: int, eta_index: int, rows: list[dict]) -> dict:
    successes = sum(r["success"] for r in rows)
    trials = len(rows)
    lo, hi = core.wilson_ci(successes, trials)
    g_agree = sum(r["success"] == r["g_has_m_window_maxima"] for r in rows)
    point = {
        "kind": kind,
        "m": m,
        "anchor": {**ANCHORS[m], "weights": list(ANCHORS[m]["weights"])},
        "eta": ETA_GRID[eta_index] if kind == "centre" else None,
        "dirichlet_concentration": DIRICHLET_CONC if kind == "weight" else None,
        "min_adjacent_spacing_s": min_adjacent_spacing(m),
        "replicas": trials,
        "replica_walkers": REPLICA_WALKERS,
        "successes": successes,
        "survival_probability": successes / trials if trials else None,
        "wilson_95_ci": [lo, hi],
        "g_theory_successes": sum(r["g_has_m_window_maxima"] for r in rows),
        "classifier_theory_agreement": g_agree / trials if trials else None,
        "median_window_events": float(
            np.median([r["kills_in_window"] for r in rows])
        ),
        "mode_count_histogram": {
            str(k): int(sum(1 for r in rows if r["mode_count"] == k))
            for k in sorted({r["mode_count"] for r in rows})
        },
        "replica_rows": sorted(rows, key=lambda r: r["replica"]),
    }
    tag = (
        f"point_centre_m{m}_eta{ETA_GRID[eta_index]:g}"
        if kind == "centre"
        else f"point_weight_m{m}_conc{DIRICHLET_CONC:g}"
    )
    core.write_json(W3_DIR / f"{tag}.json", point)
    return point


def make_figure(points: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.35, 2.75),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": [2.1, 1.0]},
    )
    colours = {2: core.OI_BLUE, 3: core.OI_VERMILLION}
    ax = axes[0]
    for m in sorted(ANCHORS):
        rows = sorted(
            [p for p in points if p["kind"] == "centre" and p["m"] == m],
            key=lambda p: p["eta"],
        )
        xs = [p["eta"] for p in rows]
        ys = [p["survival_probability"] for p in rows]
        ylo = [p["survival_probability"] - p["wilson_95_ci"][0] for p in rows]
        yhi = [p["wilson_95_ci"][1] - p["survival_probability"] for p in rows]
        ax.errorbar(
            xs, ys, yerr=[ylo, yhi], marker="o", ms=4, lw=1.2, capsize=2.5,
            color=colours[m],
            label=f"$m={m}$ classifier ($s={min_adjacent_spacing(m):.2f}$)",
        )
        gtheory = [
            p["g_theory_successes"] / p["replicas"] for p in rows
        ]
        ax.plot(
            xs, gtheory, marker="s", ms=3, lw=1.0, ls=":", color=colours[m],
            alpha=0.75, label=f"$m={m}$ $G$ retains $m$ maxima",
        )
    ax.set_xlabel(
        r"centre jitter scale $\eta$"
        "\n"
        r"$\delta_j \sim \mathcal{N}(0,\,(\eta s)^2)$, Wilson 95% CIs"
    )
    ax.set_ylabel(r"$P$(exactly-$m$ modality)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("centre jitter", fontsize=8)

    ax = axes[1]
    xpos = {2: 0, 3: 1}
    for m in sorted(ANCHORS):
        rows = [p for p in points if p["kind"] == "weight" and p["m"] == m]
        if not rows:
            continue
        p = rows[0]
        y = p["survival_probability"]
        ax.errorbar(
            [xpos[m]], [y],
            yerr=[[y - p["wilson_95_ci"][0]], [p["wilson_95_ci"][1] - y]],
            marker="D", ms=5.5, capsize=3, color=colours[m],
            label=f"$m={m}$: {p['successes']}/{p['replicas']}",
        )
    ax.set_xticks(list(xpos.values()), [f"$m={m}$" for m in xpos])
    ax.set_xlim(-0.7, 1.7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(
        rf"weight jitter, $w \sim \mathrm{{Dir}}({DIRICHLET_CONC:g}\,"
        rf"\bar{{w}})$",
        fontsize=8,
    )
    stem = core.FIGURES / "exact_m_jitter_robustness_prr"
    written = core.save_figure(fig, stem)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    out_path = W3_DIR / "jitter_robustness.json"

    if args.figure_only:
        points = json.loads(out_path.read_text(encoding="utf-8"))["points"]
        for path in make_figure(points):
            print(f"figure -> {path}", flush=True)
        return

    gate_zero = core.gate_zero_general(
        SEED, dt=DT, n_perp=1, tag=core.TAG_W3_GATE
    )
    print(
        f"gate B=0 (explicit centres) PASS: "
        f"contact_steps={gate_zero['contact_steps']}",
        flush=True,
    )

    tasks = [
        ("centre", m, ei, rep)
        for m in sorted(ANCHORS)
        for ei in range(len(ETA_GRID))
        for rep in range(args.replicas)
    ] + [
        ("weight", m, 0, rep)
        for m in sorted(ANCHORS)
        for rep in range(args.replicas)
    ]
    buckets: dict[tuple, list[dict]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replica_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            done += 1
            buckets.setdefault(
                (row["kind"], row["m"], row["eta_index"]), []
            ).append(row)
            if done % 25 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] replicas complete", flush=True)

    points = []
    for (kind, m, eta_index), rows in sorted(buckets.items()):
        point = aggregate_point(kind, m, eta_index, rows)
        points.append(point)
        label = f"eta={point['eta']}" if kind == "centre" else "weight"
        print(
            f"point {kind} m={m} {label}: "
            f"survival={point['survival_probability']:.2f} "
            f"CI=[{point['wilson_95_ci'][0]:.2f},{point['wilson_95_ci'][1]:.2f}] "
            f"G-agree={point['classifier_theory_agreement']:.2f}",
            flush=True,
        )

    summary = {
        "stream": "w3_jitter",
        "anchors": {
            str(m): {**a, "weights": list(a["weights"]),
                     "target_times": list(base.TARGET_TIMES[m]),
                     "centres_z": [float(c) for c in core.centres_z_for(m)],
                     "min_adjacent_spacing_s": min_adjacent_spacing(m)}
            for m, a in ANCHORS.items()
        },
        "eta_grid": list(ETA_GRID),
        "replicas": args.replicas,
        "replica_walkers": REPLICA_WALKERS,
        "dirichlet_concentration": DIRICHLET_CONC,
        "success_rule": "classifier mode count == m (5 sigma + 5% rule)",
        "gate1_zero_budget": gate_zero,
        "model_parameters": core.model_dict(core.MODEL),
        "dt": DT,
        "tmax": TMAX,
        "seed": SEED,
        "bandwidth": BANDWIDTH,
        "points": [
            {k: v for k, v in p.items() if k != "replica_rows"} for p in points
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    core.write_json(out_path, summary)
    print(f"summary -> {out_path}", flush=True)
    for path in make_figure(points):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
