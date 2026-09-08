#!/usr/bin/env python3
"""Independent-seed repeats at the halved time step for the boundary-adjacent cell.

The stored dt-halving comparison ``robustness/dt_halving/m3_phase_boundary.json``
(seed 20260819, 5 x 10^5 walkers per resolution) found that the deliberately
boundary-adjacent cell (m, eps, B, w) = (3, 0.175, 0.5, (1/3, 1/3, 1/3)) retains
three classified modes at dt = 10^-3 but two at dt = 5 x 10^-4.  Because the
step size enters the SeedSequence entropy, that pair uses independent streams
and cannot separate a time-step effect from sampling variation.

This script adds a separately identified ``dt_half_seed_repeat`` study: the
same cell at dt = 5 x 10^-4 with the three seeds 20260814, 20260815, 20260816
and 10^6 walkers per run, matching the walker count of the three existing
dt = 10^-3 seed repeats (``robustness/seed_repeats/m3_boundary_pass_seed*.json``).
The completed dt = 10^-3 records are not touched, and the streams at the two
resolutions are NOT paired common random numbers.

Every run stores the raw 0.02-wide window counts and edges, and is classified
explicitly through ``reclassify_covariance_aware.classify_both`` (both sigma
conventions, with the individual five-sigma and 5 % flags of every local
maximum).  The per-run records deliberately use the ``classify_both`` output
schema rather than the legacy ``classify_histogram`` schema, so they are not
double-counted among the 201 retained-campaign records re-judged by
``reclassify_covariance_aware.py``.  The summary compares the continuous
late-candidate diagnostics (relative prominence and covariance-aware z) across
the two resolutions as well as the binary verdicts.

Outputs:
    artifacts/data/exact_m_prr_upgrade/robustness/dt_half_seed_repeats/
        m3_phase_boundary_dthalf_seed<seed>.json
    artifacts/data/exact_m_prr_upgrade/robustness/dt_half_seed_repeat_summary.json

Usage: ``python code/exact_m_prr_dt_half_seed_repeat.py --workers 3``
(``--force`` reruns completed seeds; ``--walkers`` defaults to 10^6).
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

import validate_exact_m_offlattice as base  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402
import exact_m_prr_robustness as rob  # noqa: E402
import reclassify_covariance_aware as recl  # noqa: E402

OUT_DIR = rob.ROBUST_DIR / "dt_half_seed_repeats"
SUMMARY_OUT = rob.ROBUST_DIR / "dt_half_seed_repeat_summary.json"
NAME = "m3_phase_boundary"
CELL = {
    "m": 3,
    "eps": 0.175,
    "budget": 0.5,
    "weights": tuple(rob.M3_WEIGHTS),
}
SEEDS = rob.SEEDS  # (20260814, 20260815, 20260816)
WALKERS = rob.SEED_WALKERS  # 1_000_000
DT_HALF = 0.5 * rob.DT  # 5e-4
TAG_ROBUST_BASE_DT_HALF_SEED = 65  # disjoint from tags 61--64 of the robustness driver
BANDWIDTH = rob.BANDWIDTH
COARSE_RECORDS = tuple(
    rob.SEED_DIR / f"m3_boundary_pass_seed{seed}.json" for seed in SEEDS
)
DT_PAIR_RECORD = rob.DT_DIR / f"{NAME}.json"


def _path(seed: int) -> Path:
    return OUT_DIR / f"{NAME}_dthalf_seed{seed}.json"


def _classify_record(counts, edges, walkers: int, *, edge_time: float) -> dict:
    result = recl.classify_both(counts, edges, walkers, bandwidth=BANDWIDTH)
    rows = [
        {k: v for k, v in row.items()}
        for row in result["rows"]
    ]
    for row in rows:
        row["passes_sigma_gate_covariance_aware"] = bool(
            row["z_covariance_aware"] >= recl.SIGMA_FACTOR
        )
        row["passes_relative_floor"] = bool(
            row["relative_prominence"] >= recl.RELATIVE_FLOOR
        )
    late = [r for r in rows if r["time"] > edge_time]
    late_candidate = max(late, key=lambda r: r["z_covariance_aware"]) if late else None
    return {
        "bandwidth": BANDWIDTH,
        "prominence_sigma_factor": recl.SIGMA_FACTOR,
        "prominence_relative_floor": recl.RELATIVE_FLOOR,
        "global_max_smoothed_height": result["global_max"],
        "local_maxima_both_conventions": rows,
        "mode_count_peak_only": int(result["mode_count_peak_only"]),
        "mode_count_covariance_aware": int(result["mode_count_covariance_aware"]),
        "counted_times_covariance_aware": result["significant_times_covariance_aware"],
        "late_basin_edge_time": edge_time,
        "late_candidate": late_candidate,
        "late_mode_retained_covariance_aware": bool(
            late_candidate is not None
            and late_candidate["significant_covariance_aware"]
        ),
    }


def _late_edge_time() -> float:
    """Last valley of the free clock G in the window (the W2 last-basin edge)."""

    ts = np.linspace(base.WINDOW[0], base.WINDOW[1], 3001)
    g = base.free_exposure_clock(ts, m=CELL["m"], eps=CELL["eps"], weights=CELL["weights"])["g"]
    valleys = [
        float(ts[i]) for i in range(1, ts.size - 1) if g[i] < g[i - 1] and g[i] <= g[i + 1]
    ]
    if not valleys:
        raise RuntimeError("free clock has no window valley for this cell")
    return valleys[-1]


def _run_seed(task: tuple[int, int, bool, float]) -> dict:
    seed, walkers, force, edge_time = task
    path = _path(seed)
    if path.exists() and not force:
        existing = rob._read(path)
        cfg = existing["parameters"]["config"]
        if int(cfg["walkers"]) != walkers or int(cfg["seed"]) != seed:
            raise RuntimeError(
                f"{path.name} exists with a different walker count or seed; "
                "pass --force to replace it"
            )
        return existing

    outcome = base.run_config(
        m=CELL["m"],
        eps=CELL["eps"],
        budget=CELL["budget"],
        weights=CELL["weights"],
        walkers=walkers,
        chunk=min(rob.CHUNK, walkers),
        dt=DT_HALF,
        tmax=rob.TMAX,
        seed=seed,
        tag=TAG_ROBUST_BASE_DT_HALF_SEED,
        verbose=False,
    )
    kill_times = outcome["kill_times"]
    edges = np.arange(
        base.WINDOW[0], base.WINDOW[1] + 0.5 * base.WINDOW_BIN, base.WINDOW_BIN
    )
    counts, _ = np.histogram(kill_times, bins=edges)
    window_kills = int(np.sum((kill_times >= base.WINDOW[0]) & (kill_times <= base.WINDOW[1])))
    classifier = _classify_record(counts, edges, walkers, edge_time=edge_time)
    payload = {
        "schema_version": 1,
        "analysis": "dt_half_seed_repeat: independent-seed repeat at the halved time step",
        "parameters": {
            "model_parameters": core.model_dict(core.MODEL),
            "config": {
                "name": NAME,
                "m": CELL["m"],
                "eps": CELL["eps"],
                "budget": CELL["budget"],
                "weights": list(CELL["weights"]),
                "target_times": list(base.TARGET_TIMES[CELL["m"]]),
                "walkers": walkers,
                "chunk": min(rob.CHUNK, walkers),
                "dt": DT_HALF,
                "tmax": rob.TMAX,
                "seed": seed,
                "tag": TAG_ROBUST_BASE_DT_HALF_SEED,
                "classifier_bandwidth": BANDWIDTH,
                "classifier_prominence_sigma_factor": recl.SIGMA_FACTOR,
                "classifier_prominence_relative_floor": recl.RELATIVE_FLOOR,
            },
            "rng_note": (
                "deterministic Philox stream; seed, tag, and dt are part of the "
                "SeedSequence entropy, so this stream is independent of the "
                "dt = 1e-3 seed repeats with the same seed and of the dt-halving pair"
            ),
            "classifier_note": (
                "classified through reclassify_covariance_aware.classify_both on the "
                "stored counts; stored in the classify_both schema, not the legacy "
                "classify_histogram schema, so this record is outside the 201 "
                "retained-campaign records"
            ),
        },
        "results": {
            "histogram": {
                "window": list(base.WINDOW),
                "bin_width": base.WINDOW_BIN,
                "counts": [int(v) for v in counts],
                "edges": [float(v) for v in edges],
            },
            "classifier_both_conventions": classifier,
            "kills": int(kill_times.size),
            "survivors": int(outcome["survivors"]),
            "kill_fraction": float(kill_times.size / walkers),
            "kills_in_window": window_kills,
            "event_fraction_in_window": float(window_kills / walkers),
            "kill_probability_max": float(outcome["kill_probability_max"]),
            "mass_balance_passed": bool(kill_times.size + outcome["survivors"] == walkers),
            "runtime_seconds": float(outcome["runtime_seconds"]),
        },
    }
    core.write_json(path, payload)
    return payload


def _coarse_reference(edge_time: float) -> list[dict]:
    """Re-judge the matched-N dt = 1e-3 seed repeats and the dt-halving pair."""

    rows = []
    for path in COARSE_RECORDS:
        payload = rob._read(path)
        cfg = payload["parameters"]["config"]
        stored = payload["results"]["classifier"]
        cls = _classify_record(
            stored["counts"], stored["edges"], int(cfg["walkers"]), edge_time=edge_time
        )
        rows.append(
            {
                "source": f"seed_repeats/{path.name}",
                "dt": float(cfg["dt"]),
                "seed": int(cfg["seed"]),
                "walkers": int(cfg["walkers"]),
                **_diag(cls),
            }
        )
    pair = rob._read(DT_PAIR_RECORD)
    walkers = int(pair["parameters"]["config"]["walkers_per_run"])
    for label in ("dt", "dt_half"):
        stored = pair["runs"][label]["classifier"]
        cls = _classify_record(stored["counts"], stored["edges"], walkers, edge_time=edge_time)
        rows.append(
            {
                "source": f"dt_halving/{DT_PAIR_RECORD.name}#{label}",
                "dt": float(pair["runs"][label]["dt"]),
                "seed": int(pair["parameters"]["config"]["seed"]),
                "walkers": walkers,
                **_diag(cls),
            }
        )
    return rows


def _diag(cls: dict) -> dict:
    late = cls["late_candidate"]
    return {
        "mode_count_covariance_aware": cls["mode_count_covariance_aware"],
        "mode_count_peak_only": cls["mode_count_peak_only"],
        "counted_times_covariance_aware": cls["counted_times_covariance_aware"],
        "late_candidate_time": late["time"] if late else None,
        "late_relative_prominence": late["relative_prominence"] if late else None,
        "late_z_covariance_aware": late["z_covariance_aware"] if late else None,
        "late_passes_sigma_gate": late["passes_sigma_gate_covariance_aware"] if late else None,
        "late_passes_relative_floor": late["passes_relative_floor"] if late else None,
        "late_mode_retained": cls["late_mode_retained_covariance_aware"],
    }


def _range(values):
    vals = [v for v in values if v is not None]
    return [min(vals), max(vals)] if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--walkers", type=lambda v: int(float(v)), default=WALKERS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.walkers <= 0:
        raise SystemExit("worker and walker counts must be positive")

    started = time.perf_counter()
    edge_time = _late_edge_time()
    tasks = [(seed, args.walkers, args.force, edge_time) for seed in SEEDS]
    payloads = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(_run_seed, task): task for task in tasks}
        for future in as_completed(futures):
            payload = future.result()
            cfg = payload["parameters"]["config"]
            cls = payload["results"]["classifier_both_conventions"]
            print(
                f"[dt/2 seed {cfg['seed']}] modes(cov)={cls['mode_count_covariance_aware']} "
                f"late z={cls['late_candidate']['z_covariance_aware'] if cls['late_candidate'] else None} "
                f"({payload['results']['runtime_seconds']:.0f} s)"
            )
            payloads.append(payload)
    payloads.sort(key=lambda p: p["parameters"]["config"]["seed"])

    fine_rows = []
    for payload in payloads:
        cfg = payload["parameters"]["config"]
        cls = payload["results"]["classifier_both_conventions"]
        fine_rows.append(
            {
                "source": f"dt_half_seed_repeats/{_path(cfg['seed']).name}",
                "dt": float(cfg["dt"]),
                "seed": int(cfg["seed"]),
                "walkers": int(cfg["walkers"]),
                "kill_fraction": payload["results"]["kill_fraction"],
                "event_fraction_in_window": payload["results"]["event_fraction_in_window"],
                "runtime_seconds": payload["results"]["runtime_seconds"],
                **_diag(cls),
            }
        )
    coarse_rows = _coarse_reference(edge_time)
    coarse_matched = [r for r in coarse_rows if r["source"].startswith("seed_repeats/")]

    n_retain = sum(1 for r in fine_rows if r["mode_count_covariance_aware"] == CELL["m"])
    summary = {
        "schema_version": 1,
        "analysis": (
            "dt_half_seed_repeat: three independent-seed runs of the boundary-adjacent "
            "cell at dt = 5e-4, compared with the matched-N dt = 1e-3 seed repeats "
            "and the stored dt-halving pair"
        ),
        "cell": {**CELL, "weights": list(CELL["weights"])},
        "late_basin_edge_time": edge_time,
        "classifier": {
            "definition": "reclassify_covariance_aware.classify_both (covariance-aware verdicts are the article's)",
            "bandwidth": BANDWIDTH,
            "prominence_sigma_factor": recl.SIGMA_FACTOR,
            "prominence_relative_floor": recl.RELATIVE_FLOOR,
            "late_candidate_rule": "local maximum later than the last window valley of G with the largest covariance-aware z",
        },
        "dt_half_runs": fine_rows,
        "dt_half_walkers": args.walkers,
        "dt_half_seeds": list(SEEDS),
        "dt_half_three_mode_retention": [n_retain, len(fine_rows)],
        "dt_half_late_relative_prominence_range": _range(
            r["late_relative_prominence"] for r in fine_rows
        ),
        "dt_half_late_z_range": _range(r["late_z_covariance_aware"] for r in fine_rows),
        "dt_half_late_time_range": _range(r["late_candidate_time"] for r in fine_rows),
        "dt_half_kill_fraction_range": _range(r["kill_fraction"] for r in fine_rows),
        "coarse_reference_runs": coarse_rows,
        "coarse_matched_n_three_mode_retention": [
            sum(1 for r in coarse_matched if r["mode_count_covariance_aware"] == CELL["m"]),
            len(coarse_matched),
        ],
        "coarse_matched_n_late_relative_prominence_range": _range(
            r["late_relative_prominence"] for r in coarse_matched
        ),
        "coarse_matched_n_late_z_range": _range(
            r["late_z_covariance_aware"] for r in coarse_matched
        ),
        "interpretation_note": (
            "These additional independent-stream runs characterize sampling variability "
            "at the finer step, but three seeds do not exclude a systematic time-step "
            "contribution near this classifier boundary.  The streams at the two "
            "resolutions are not paired common random numbers.  This is not a "
            "convergence test and does not establish a theorem threshold."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    core.write_json(SUMMARY_OUT, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("dt_half_runs", "coarse_reference_runs")}, indent=1))
    for row in fine_rows + coarse_rows:
        print(
            f"  {row['source']}: dt={row['dt']} N={row['walkers']} modes={row['mode_count_covariance_aware']} "
            f"late t={row['late_candidate_time']} rel={row['late_relative_prominence']} z={row['late_z_covariance_aware']}"
        )


if __name__ == "__main__":
    main()
