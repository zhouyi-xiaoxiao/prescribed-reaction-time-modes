#!/usr/bin/env python3
"""Reproducible robustness checks for the finite-parameter PRR numerics.

The script has three independent phases:

``sensitivity``
    Reclassify stored histogram sufficient statistics over declared bandwidth
    and relative-prominence grids.  No Monte Carlo simulation is performed.
``seeds``
    Repeat the m=3 anchor, two empirical-threshold sides, two phase-boundary
    sides, and the m=5 anchor on three independent deterministic seeds.
``dt``
    Compare dt=1e-3 with dt/2 at the m=3 empirical threshold, an m=3 phase
    boundary point, and the m=5 anchor.

All outputs are JSON files below
``artifacts/data/exact_m_prr_upgrade/robustness``.  Existing completed result
files are reused unless ``--force`` is supplied, so interrupted campaigns can
be resumed without repeating finished simulations.
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


ROBUST_DIR = core.UPGRADE_DATA / "robustness"
SENSITIVITY_OUT = ROBUST_DIR / "classifier_sensitivity.json"
SEED_DIR = ROBUST_DIR / "seed_repeats"
DT_DIR = ROBUST_DIR / "dt_halving"

PRODUCTION_DIR = core.REPORT / "artifacts" / "data" / "exact_m_offlattice_production"
W1_DIR = core.UPGRADE_DATA / "w1_phase_diagram"
W2_DIR = core.UPGRADE_DATA / "w2_b0_empirical"
W4_PATH = core.UPGRADE_DATA / "w4_m5_demo" / "m5_demo.json"
W5_PATH = core.UPGRADE_DATA / "w5_d3_spotcheck" / "d3_spotcheck.json"

PRIMARY_BANDWIDTHS = (0.03, 0.04, 0.05)
PRIMARY_RELATIVE_FLOORS = (0.03, 0.05, 0.07)
WIDE_BANDWIDTHS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
WIDE_RELATIVE_FLOORS = (0.0, 0.01, 0.03, 0.05, 0.07, 0.10)

SEEDS = (20260814, 20260815, 20260816)
SEED_WALKERS = 1_000_000
DT_SEED = 20260819
DT_WALKERS = 500_000
CHUNK = 100_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH

# Campaign-specific tags keep these streams disjoint from W1--W5.
TAG_ROBUST_BASE_SEED = 61
TAG_ROBUST_M5_SEED = 62
TAG_ROBUST_BASE_DT = 63
TAG_ROBUST_M5_DT = 64

M3_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
M5_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)
M5_CENTRES = np.array((2.8, 2.2, 1.6, 1.0, 0.4), dtype=float)
M5_Z0 = 8.0
M5_TARGET_TIMES = tuple(float(math.log(M5_Z0 / mu)) for mu in M5_CENTRES)

SEED_CONFIGS = (
    {
        "name": "m3_anchor",
        "kind": "base",
        "m": 3,
        "eps": 0.10,
        "budget": 1.0,
        "weights": M3_WEIGHTS,
        "expected_mode_count": 3,
        "interpretation": "principal m=3 finite-parameter anchor",
    },
    {
        "name": "m3_threshold_low",
        "kind": "base",
        "m": 3,
        "eps": 0.10,
        "budget": 3.50,
        "weights": M3_WEIGHTS,
        "expected_mode_count": 3,
        "interpretation": "below the baseline operational B_op estimate 3.57",
    },
    {
        "name": "m3_threshold_high",
        "kind": "base",
        "m": 3,
        "eps": 0.10,
        "budget": 3.64,
        "weights": M3_WEIGHTS,
        "expected_mode_count": 2,
        "interpretation": "above the baseline operational B_op estimate 3.57",
    },
    {
        "name": "m3_boundary_pass",
        "kind": "base",
        "m": 3,
        "eps": 0.175,
        "budget": 0.50,
        "weights": M3_WEIGHTS,
        "expected_mode_count": 3,
        "interpretation": "passing side of a phase-diagram boundary",
    },
    {
        "name": "m3_boundary_fail",
        "kind": "base",
        "m": 3,
        "eps": 0.175,
        "budget": 1.0,
        "weights": M3_WEIGHTS,
        "expected_mode_count": 2,
        "interpretation": "failing side of a phase-diagram boundary",
    },
    {
        "name": "m5_anchor",
        "kind": "m5",
        "m": 5,
        "eps": 0.10,
        "budget": 1.0,
        "weights": M5_WEIGHTS,
        "expected_mode_count": 5,
        "interpretation": "five-mode stretched-geometry anchor",
    },
)

DT_CONFIGS = (
    {
        "name": "m3_anchor",
        "kind": "base",
        "m": 3,
        "eps": 0.10,
        "budget": 1.0,
        "weights": M3_WEIGHTS,
        "role": "interior_control",
        "interpretation": "principal m=3 interior anchor",
    },
    {
        "name": "m3_threshold",
        "kind": "base",
        "m": 3,
        "eps": 0.10,
        "budget": 3.57,
        "weights": M3_WEIGHTS,
        "role": "operational_threshold",
        "interpretation": "baseline operational empirical threshold",
    },
    {
        "name": "m3_phase_boundary",
        "kind": "base",
        "m": 3,
        "eps": 0.175,
        "budget": 0.50,
        "weights": M3_WEIGHTS,
        "role": "phase_boundary",
        "interpretation": "passing cell adjacent to the phase boundary",
    },
    {
        "name": "m5_anchor",
        "kind": "m5",
        "m": 5,
        "eps": 0.10,
        "budget": 1.0,
        "weights": M5_WEIGHTS,
        "role": "interior_control",
        "interpretation": "five-mode stretched-geometry anchor",
    },
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _walkers(payload: dict) -> int:
    parameters = payload["parameters"]
    config = parameters.get("config", {})
    value = config.get("walkers", parameters.get("walkers"))
    if value is None:
        raise KeyError("walker count is absent from payload")
    return int(value)


def _reclassify(
    classifier: dict,
    walkers: int,
    *,
    bandwidth: float,
    relative_floor: float,
) -> dict:
    result = base.classify_histogram(
        np.asarray(classifier["counts"], dtype=np.int64),
        np.asarray(classifier["edges"], dtype=float),
        walkers,
        bandwidth=bandwidth,
        prominence_sigma_factor=base.PROMINENCE_SIGMA_FACTOR,
        prominence_relative_floor=relative_floor,
    )
    return {
        "bandwidth": bandwidth,
        "prominence_sigma_factor": base.PROMINENCE_SIGMA_FACTOR,
        "prominence_relative_floor": relative_floor,
        "mode_count": result["mode_count"],
        "peak_times": [row["time"] for row in result["significant_maxima"]],
        "prominence_over_sigma": [
            row["prominence_over_sigma"] for row in result["significant_maxima"]
        ],
    }


def _variant_grid(
    classifier: dict,
    walkers: int,
    bandwidths: tuple[float, ...],
    floors: tuple[float, ...],
) -> list[dict]:
    return [
        _reclassify(
            classifier,
            walkers,
            bandwidth=bandwidth,
            relative_floor=floor,
        )
        for bandwidth in bandwidths
        for floor in floors
    ]


def _final_w1_payloads() -> list[tuple[Path, dict]]:
    best: dict[tuple[int, float, float], tuple[Path, dict]] = {}
    for path in sorted(W1_DIR.glob("cell_m*_eps*_B*.json")):
        payload = _read(path)
        cfg = payload["parameters"]["config"]
        key = (int(cfg["m"]), float(cfg["eps"]), float(cfg["budget"]))
        if key not in best or bool(cfg["refined"]):
            best[key] = (path, payload)
    return [best[key] for key in sorted(best)]


def _w2_transition_summaries(w2_rows: list[dict]) -> list[dict]:
    """Summarize the transition supported by the already-computed probes.

    The W2 probe locations were selected for the baseline classifier, so a
    changed classifier can be left- or right-censored by the stored set.  We
    report that honestly rather than extrapolating a new threshold.
    """

    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in w2_rows:
        grouped.setdefault((int(row["m"]), float(row["eps"])), []).append(row)

    summaries = []
    for (m, eps), rows in sorted(grouped.items()):
        settings = []
        for variant_index, reference in enumerate(rows[0]["variants"]):
            observations = sorted(
                (
                    float(row["budget"]),
                    bool(row["variants"][variant_index]["last_mode_present"]),
                )
                for row in rows
            )
            passing = [budget for budget, verdict in observations if verdict]
            failing = [budget for budget, verdict in observations if not verdict]
            monotonic_violations = sum(
                1
                for left_index, (_, left_verdict) in enumerate(observations)
                for _, right_verdict in observations[left_index + 1 :]
                if not left_verdict and right_verdict
            )
            if not failing:
                status = "right_censored_above_max_stored_probe"
                bracket = [max(passing), None]
                estimate = None
            elif not passing:
                status = "left_censored_below_min_stored_probe"
                bracket = [None, min(failing)]
                estimate = None
            else:
                lower = max(passing)
                upper_candidates = [budget for budget in failing if budget > lower]
                if not upper_candidates:
                    status = "nonmonotone_or_right_censored"
                    bracket = [lower, None]
                    estimate = None
                else:
                    upper = min(upper_candidates)
                    status = "bracketed_on_stored_probes"
                    bracket = [lower, upper]
                    estimate = math.sqrt(lower * upper)
            settings.append(
                {
                    "bandwidth": float(reference["bandwidth"]),
                    "prominence_relative_floor": float(
                        reference["prominence_relative_floor"]
                    ),
                    "status": status,
                    "stored_probe_bracket": bracket,
                    "geometric_midpoint_if_bracketed": estimate,
                    "monotonicity_violations": monotonic_violations,
                }
            )
        estimates = [
            float(row["geometric_midpoint_if_bracketed"])
            for row in settings
            if row["geometric_midpoint_if_bracketed"] is not None
        ]
        summaries.append(
            {
                "m": m,
                "eps": eps,
                "bracketed_midpoint_range": (
                    [min(estimates), max(estimates)] if estimates else None
                ),
                "settings": settings,
            }
        )
    return summaries


def run_sensitivity() -> dict:
    w1_rows = []
    for path, payload in _final_w1_payloads():
        cfg = payload["parameters"]["config"]
        classifier = payload["results"]["classifier"]
        variants = _variant_grid(
            classifier,
            int(cfg["walkers"]),
            PRIMARY_BANDWIDTHS,
            PRIMARY_RELATIVE_FLOORS,
        )
        counts = sorted({int(row["mode_count"]) for row in variants})
        w1_rows.append(
            {
                "m": int(cfg["m"]),
                "eps": float(cfg["eps"]),
                "budget": float(cfg["budget"]),
                "source_file": path.name,
                "refined": bool(cfg["refined"]),
                "walkers": int(cfg["walkers"]),
                "baseline_mode_count": int(payload["results"]["mode_count"]),
                "mode_counts_over_grid": counts,
                "sensitive": len(counts) > 1,
                "variants": variants,
            }
        )

    anchor_specs = (
        (
            "m3_anchor",
            PRODUCTION_DIR / "m3_eps0.1_B1_w33-33-33.json",
        ),
        ("m5_anchor", W4_PATH),
        ("d3_anchor", W5_PATH),
    )
    anchors = []
    for name, path in anchor_specs:
        payload = _read(path)
        classifier = payload["results"]["classifier"]
        variants = _variant_grid(
            classifier,
            _walkers(payload),
            WIDE_BANDWIDTHS,
            WIDE_RELATIVE_FLOORS,
        )
        counts = sorted({int(row["mode_count"]) for row in variants})
        anchors.append(
            {
                "name": name,
                "source_file": str(path.relative_to(core.REPORT)),
                "walkers": _walkers(payload),
                "baseline_mode_count": int(payload["results"]["mode_count"]),
                "mode_counts_over_wide_grid": counts,
                "stable": len(counts) == 1,
                "variants": variants,
            }
        )

    w2_summary_path = W2_DIR / "B0_empirical.json"
    w2_summary = _read(w2_summary_path)
    basin_edges = {
        (int(chain["m"]), float(chain["eps"])): float(
            chain["basin_edge_last_g_valley_time"]
        )
        for chain in w2_summary["chains"]
        if chain.get("basin_edge_last_g_valley_time") is not None
    }
    w2_rows = []
    for path in sorted(W2_DIR.glob("probe_*.json")):
        payload = _read(path)
        results = payload["results"]
        if "classifier" not in results:
            raise RuntimeError(
                f"{path.name} lacks classifier counts; rerun W2 with the current code"
            )
        cfg = payload["parameters"]["config"]
        edge = basin_edges[(int(cfg["m"]), float(cfg["eps"]))]
        variants = _variant_grid(
            results["classifier"],
            int(cfg["walkers"]),
            PRIMARY_BANDWIDTHS,
            PRIMARY_RELATIVE_FLOORS,
        )
        for row in variants:
            row["last_mode_present"] = bool(
                any(float(t) > edge for t in row["peak_times"])
            )
        verdicts = sorted({bool(row["last_mode_present"]) for row in variants})
        w2_rows.append(
            {
                "m": int(cfg["m"]),
                "eps": float(cfg["eps"]),
                "budget": float(cfg["budget"]),
                "source_file": path.name,
                "walkers": int(cfg["walkers"]),
                "basin_edge_last_g_valley_time": edge,
                "baseline_last_mode_present": bool(results["last_mode_present"]),
                "last_mode_verdicts_over_grid": verdicts,
                "sensitive": len(verdicts) > 1,
                "variants": variants,
            }
        )

    sensitive_w1 = [row for row in w1_rows if row["sensitive"]]
    sensitive_w2 = [row for row in w2_rows if row["sensitive"]]
    w2_transitions = _w2_transition_summaries(w2_rows)
    payload = {
        "schema_version": 1,
        "analysis": "stored-count classifier sensitivity; no Monte Carlo rerun",
        "classifier": {
            "bin_width": base.WINDOW_BIN,
            "prominence_sigma_factor": base.PROMINENCE_SIGMA_FACTOR,
            "primary_bandwidths": list(PRIMARY_BANDWIDTHS),
            "primary_relative_prominence_floors": list(PRIMARY_RELATIVE_FLOORS),
            "wide_anchor_bandwidths": list(WIDE_BANDWIDTHS),
            "wide_anchor_relative_prominence_floors": list(WIDE_RELATIVE_FLOORS),
        },
        "summary": {
            "w1_final_cells": len(w1_rows),
            "w1_sensitive_cells": len(sensitive_w1),
            "w1_sensitive_coordinates": [
                {k: row[k] for k in ("m", "eps", "budget")}
                for row in sensitive_w1
            ],
            "w2_probes": len(w2_rows),
            "w2_sensitive_probes": len(sensitive_w2),
            "w2_note": (
                "W2 probes are deliberately concentrated near each baseline "
                "decision boundary; probe-level sensitivity is therefore not "
                "a phase-diagram-wide instability rate."
            ),
            "wide_anchor_stability": {
                row["name"]: row["stable"] for row in anchors
            },
        },
        "w1_cells": w1_rows,
        "w2_probes": w2_rows,
        "w2_stored_probe_transition_summaries": w2_transitions,
        "wide_anchor_checks": anchors,
    }
    core.write_json(SENSITIVITY_OUT, payload)
    print(
        f"sensitivity: W1 {len(sensitive_w1)}/{len(w1_rows)} sensitive; "
        f"W2 {len(sensitive_w2)}/{len(w2_rows)} sensitive -> {SENSITIVITY_OUT}",
        flush=True,
    )
    return payload


def _seed_path(name: str, seed: int) -> Path:
    return SEED_DIR / f"{name}_seed{seed}.json"


def _seed_row(payload: dict, path: Path) -> dict:
    cfg = payload["parameters"]["config"]
    results = payload["results"]
    expected = next(
        int(spec["expected_mode_count"])
        for spec in SEED_CONFIGS
        if spec["name"] == cfg["name"]
    )
    return {
        "name": cfg["name"],
        "seed": int(cfg["seed"]),
        "m": int(cfg["m"]),
        "eps": float(cfg["eps"]),
        "budget": float(cfg["budget"]),
        "walkers": int(cfg["walkers"]),
        "mode_count": int(results["mode_count"]),
        "expected_mode_count": expected,
        "expected_mode_count_observed": bool(results["mode_count"] == expected),
        "target_mode_count_present": bool(results["mode_count"] == cfg["m"]),
        "peak_times": list(results["peak_times"]),
        "kill_fraction": float(results["kill_fraction"]),
        "event_fraction_in_window": float(results["event_fraction_in_window"]),
        "runtime_seconds": float(results["runtime_seconds"]),
        "file": path.name,
    }


def _run_seed_task(task: tuple[dict, int, int, bool]) -> dict:
    spec, seed, walkers, force = task
    path = _seed_path(spec["name"], seed)
    if path.exists() and not force:
        existing = _read(path)
        cfg = existing["parameters"]["config"]
        if int(cfg["walkers"]) != walkers or int(cfg["seed"]) != seed:
            raise RuntimeError(
                f"{path.name} exists with a different walker count or seed; "
                "pass --force to replace it"
            )
        return _seed_row(existing, path)

    weights = tuple(float(v) for v in spec["weights"])
    if spec["kind"] == "base":
        result = base.run_config(
            m=spec["m"],
            eps=spec["eps"],
            budget=spec["budget"],
            weights=weights,
            walkers=walkers,
            chunk=min(CHUNK, walkers),
            dt=DT,
            tmax=TMAX,
            seed=seed,
            tag=TAG_ROBUST_BASE_SEED,
            verbose=False,
        )
        summary = base.summarize_config(result, bandwidth=BANDWIDTH, tmax=TMAX)
        model = core.model_dict(core.MODEL)
        target_times = tuple(base.TARGET_TIMES[spec["m"]])
        tag = TAG_ROBUST_BASE_SEED
    else:
        p5 = core.make_model(z0=M5_Z0)
        result = core.run_config_general(
            eps=spec["eps"],
            budget=spec["budget"],
            weights=weights,
            centres_z=M5_CENTRES,
            walkers=walkers,
            chunk=min(CHUNK, walkers),
            dt=DT,
            tmax=TMAX,
            seed=seed,
            tag=TAG_ROBUST_M5_SEED,
            p=p5,
            n_perp=1,
            verbose=False,
        )
        summary = core.summarize_general(
            result,
            bandwidth=BANDWIDTH,
            target_times=M5_TARGET_TIMES,
            m=5,
        )
        model = core.model_dict(p5)
        target_times = M5_TARGET_TIMES
        tag = TAG_ROBUST_M5_SEED

    payload = {
        "schema_version": 1,
        "parameters": {
            "stream": "independent_seed_repeat",
            "model_parameters": model,
            "config": {
                "name": spec["name"],
                "interpretation": spec["interpretation"],
                "m": spec["m"],
                "eps": spec["eps"],
                "budget": spec["budget"],
                "expected_mode_count": spec["expected_mode_count"],
                "weights": list(weights),
                "target_times": list(target_times),
                "walkers": walkers,
                "chunk": min(CHUNK, walkers),
                "dt": DT,
                "tmax": TMAX,
                "seed": seed,
                "tag": tag,
                "bandwidth": BANDWIDTH,
                "prominence_sigma_factor": base.PROMINENCE_SIGMA_FACTOR,
                "prominence_relative_floor": base.PROMINENCE_RELATIVE_FLOOR,
            },
            "rng": "numpy Philox, SeedSequence-spawned substream per chunk",
        },
        "validation_gates": {
            "kill_probability_max": summary["kill_probability_max"],
            "kill_probability_in_unit_interval": bool(
                0.0 <= summary["kill_probability_max"] <= 1.0
            ),
            "mass_balance": {
                "kills": summary["kills"],
                "survivors": summary["survivors"],
                "walkers": walkers,
                "passed": bool(summary["kills"] + summary["survivors"] == walkers),
            },
        },
        "results": summary,
    }
    core.write_json(path, payload)
    return _seed_row(payload, path)


def run_seed_repeats(*, workers: int, walkers: int, force: bool) -> dict:
    started = time.perf_counter()
    tasks = [
        (dict(spec), seed, walkers, force)
        for spec in SEED_CONFIGS
        for seed in SEEDS
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_seed_task, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[seed {index}/{len(tasks)}] {row['name']} seed={row['seed']}: "
                f"modes={row['mode_count']} t={row['runtime_seconds']:.1f}s",
                flush=True,
            )
    rows.sort(key=lambda row: (row["name"], row["seed"]))

    configs = []
    for spec in SEED_CONFIGS:
        selected = [row for row in rows if row["name"] == spec["name"]]
        successes = sum(row["expected_mode_count_observed"] for row in selected)
        ci = core.wilson_ci(successes, len(selected))
        common_peak_count = min(len(row["peak_times"]) for row in selected)
        peak_ranges = [
            [
                min(row["peak_times"][index] for row in selected),
                max(row["peak_times"][index] for row in selected),
            ]
            for index in range(common_peak_count)
        ]
        configs.append(
            {
                "name": spec["name"],
                "interpretation": spec["interpretation"],
                "m": spec["m"],
                "eps": spec["eps"],
                "budget": spec["budget"],
                "expected_mode_count": spec["expected_mode_count"],
                "expected_mode_count_successes": successes,
                "all_repeats_match_expected": successes == len(selected),
                "repeats": len(selected),
                "wilson_95_ci": list(ci),
                "mode_counts": [row["mode_count"] for row in selected],
                "mode_count_consistent": len(
                    {row["mode_count"] for row in selected}
                )
                == 1,
                "common_index_peak_time_ranges": peak_ranges,
                "maximum_common_peak_span": max(
                    (upper - lower for lower, upper in peak_ranges),
                    default=None,
                ),
                "kill_fraction_range": [
                    min(row["kill_fraction"] for row in selected),
                    max(row["kill_fraction"] for row in selected),
                ],
                "event_fraction_in_window_range": [
                    min(row["event_fraction_in_window"] for row in selected),
                    max(row["event_fraction_in_window"] for row in selected),
                ],
                "rows": selected,
            }
        )

    payload = {
        "schema_version": 1,
        "analysis": "independent deterministic-seed repeats",
        "seeds": list(SEEDS),
        "walkers_per_repeat": walkers,
        "dt": DT,
        "classifier": {
            "bin_width": base.WINDOW_BIN,
            "bandwidth": BANDWIDTH,
            "prominence_sigma_factor": base.PROMINENCE_SIGMA_FACTOR,
            "prominence_relative_floor": base.PROMINENCE_RELATIVE_FLOOR,
        },
        "configs": configs,
        "wall_seconds": time.perf_counter() - started,
    }
    out = ROBUST_DIR / "seed_repeat_summary.json"
    core.write_json(out, payload)
    print(f"seed-repeat summary -> {out}", flush=True)
    return payload


def _dt_path(name: str) -> Path:
    return DT_DIR / f"{name}.json"


def _run_dt_task(task: tuple[dict, int, int, bool]) -> dict:
    spec, seed, walkers, force = task
    path = _dt_path(spec["name"])
    if path.exists() and not force:
        existing = _read(path)
        cfg = existing["parameters"]["config"]
        if int(cfg["walkers_per_run"]) != walkers or int(cfg["seed"]) != seed:
            raise RuntimeError(
                f"{path.name} exists with a different walker count or seed; "
                "pass --force to replace it"
            )
        return existing

    weights = tuple(float(v) for v in spec["weights"])
    comparison_edges = np.arange(
        core.WINDOW[0],
        core.WINDOW[1] + 0.5 * base.DT_CHECK_BIN,
        base.DT_CHECK_BIN,
    )
    runs = {}
    for label, step in (("dt", DT), ("dt_half", 0.5 * DT)):
        if spec["kind"] == "base":
            outcome = base.run_config(
                m=spec["m"],
                eps=spec["eps"],
                budget=spec["budget"],
                weights=weights,
                walkers=walkers,
                chunk=min(CHUNK, walkers),
                dt=step,
                tmax=TMAX,
                seed=seed,
                tag=TAG_ROBUST_BASE_DT,
                verbose=False,
            )
            model = core.model_dict(core.MODEL)
            tag = TAG_ROBUST_BASE_DT
        else:
            p5 = core.make_model(z0=M5_Z0)
            outcome = core.run_config_general(
                eps=spec["eps"],
                budget=spec["budget"],
                weights=weights,
                centres_z=M5_CENTRES,
                walkers=walkers,
                chunk=min(CHUNK, walkers),
                dt=step,
                tmax=TMAX,
                seed=seed,
                tag=TAG_ROBUST_M5_DT,
                p=p5,
                n_perp=1,
                verbose=False,
            )
            model = core.model_dict(p5)
            tag = TAG_ROBUST_M5_DT
        kill_times = outcome["kill_times"]
        classifier = base.classify_modes(
            kill_times,
            walkers,
            bandwidth=BANDWIDTH,
        )
        coarse_counts, _ = np.histogram(kill_times, bins=comparison_edges)
        window_kills = int(
            np.sum(
                (kill_times >= core.WINDOW[0])
                & (kill_times <= core.WINDOW[1])
            )
        )
        runs[label] = {
            "dt": step,
            "kills": int(kill_times.size),
            "survivors": int(outcome["survivors"]),
            "kill_fraction": float(kill_times.size / walkers),
            "kills_in_window": window_kills,
            "event_fraction_in_window": float(window_kills / walkers),
            "kill_probability_max": float(outcome["kill_probability_max"]),
            "comparison_bin_counts": [int(v) for v in coarse_counts],
            "classifier": classifier,
            "mode_count": int(classifier["mode_count"]),
            "peak_times": [
                row["time"] for row in classifier["significant_maxima"]
            ],
            "prominences": [
                row["prominence"] for row in classifier["significant_maxima"]
            ],
            "runtime_seconds": float(outcome["runtime_seconds"]),
            "mass_balance_passed": bool(
                kill_times.size + outcome["survivors"] == walkers
            ),
        }

    counts_a = np.asarray(runs["dt"]["comparison_bin_counts"], dtype=float)
    counts_b = np.asarray(runs["dt_half"]["comparison_bin_counts"], dtype=float)
    frac_a = counts_a / walkers
    frac_b = counts_b / walkers
    pooled = (counts_a + counts_b) / (2.0 * walkers)
    variance = pooled * (1.0 - pooled) * (2.0 / walkers)
    usable = (counts_a + counts_b) >= base.DT_CHECK_MIN_POOLED
    z_scores = np.zeros(counts_a.size)
    z_scores[usable] = (frac_a[usable] - frac_b[usable]) / np.sqrt(
        variance[usable]
    )
    used = z_scores[usable]
    peaks_a = runs["dt"]["peak_times"]
    peaks_b = runs["dt_half"]["peak_times"]
    peak_deltas = (
        [float(b - a) for a, b in zip(peaks_a, peaks_b)]
        if len(peaks_a) == len(peaks_b)
        else None
    )

    payload = {
        "schema_version": 1,
        "analysis": "Euler-Maruyama dt-halving consistency check",
        "parameters": {
            "model_parameters": model,
            "config": {
                "name": spec["name"],
                "interpretation": spec["interpretation"],
                "role": spec["role"],
                "m": spec["m"],
                "eps": spec["eps"],
                "budget": spec["budget"],
                "weights": list(weights),
                "walkers_per_run": walkers,
                "chunk": min(CHUNK, walkers),
                "dt": DT,
                "dt_half": 0.5 * DT,
                "tmax": TMAX,
                "seed": seed,
                "tag": tag,
                "classifier_bandwidth": BANDWIDTH,
                "classifier_prominence_sigma_factor": base.PROMINENCE_SIGMA_FACTOR,
                "classifier_prominence_relative_floor": base.PROMINENCE_RELATIVE_FLOOR,
            },
            "rng_note": (
                "Both runs use deterministic Philox streams; dt is part of the "
                "SeedSequence entropy, so the two samples are independent."
            ),
        },
        "runs": runs,
        "comparison": {
            "window": list(core.WINDOW),
            "bin_width": base.DT_CHECK_BIN,
            "usable_bins": int(np.sum(usable)),
            "excluded_bins": int(np.sum(~usable)),
            "z_scores": [float(v) if use else None for v, use in zip(z_scores, usable)],
            "max_abs_z": float(np.max(np.abs(used))) if used.size else None,
            "mean_abs_z": float(np.mean(np.abs(used))) if used.size else None,
            "fraction_abs_z_above_2": (
                float(np.mean(np.abs(used) > 2.0)) if used.size else None
            ),
            "mode_counts_match": bool(
                runs["dt"]["mode_count"] == runs["dt_half"]["mode_count"]
            ),
            "peak_time_deltas_dt_half_minus_dt": peak_deltas,
            "kill_fraction_delta_dt_half_minus_dt": float(
                runs["dt_half"]["kill_fraction"] - runs["dt"]["kill_fraction"]
            ),
            "note": (
                "This is a finite-resolution consistency diagnostic, not a "
                "proof of temporal-discretization convergence."
            ),
        },
    }
    core.write_json(path, payload)
    return payload


def run_dt_halving(*, workers: int, walkers: int, force: bool) -> dict:
    started = time.perf_counter()
    tasks = [(dict(spec), DT_SEED, walkers, force) for spec in DT_CONFIGS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_dt_task, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            rows.append(payload)
            cfg = payload["parameters"]["config"]
            comparison = payload["comparison"]
            print(
                f"[dt {index}/{len(tasks)}] {cfg['name']}: "
                f"modes={payload['runs']['dt']['mode_count']}/"
                f"{payload['runs']['dt_half']['mode_count']} "
                f"max|z|={comparison['max_abs_z']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda row: row["parameters"]["config"]["name"])
    summary_rows = []
    for payload in rows:
        cfg = payload["parameters"]["config"]
        comparison = payload["comparison"]
        spec = next(spec for spec in DT_CONFIGS if spec["name"] == cfg["name"])
        summary_rows.append(
            {
                "name": cfg["name"],
                "role": spec["role"],
                "m": cfg["m"],
                "eps": cfg["eps"],
                "budget": cfg["budget"],
                "walkers_per_run": cfg["walkers_per_run"],
                "mode_count_dt": payload["runs"]["dt"]["mode_count"],
                "mode_count_dt_half": payload["runs"]["dt_half"]["mode_count"],
                "mode_counts_match": comparison["mode_counts_match"],
                "peak_times_dt": payload["runs"]["dt"]["peak_times"],
                "peak_times_dt_half": payload["runs"]["dt_half"]["peak_times"],
                "max_abs_z": comparison["max_abs_z"],
                "mean_abs_z": comparison["mean_abs_z"],
                "fraction_abs_z_above_2": comparison["fraction_abs_z_above_2"],
                "usable_comparison_bins": comparison["usable_bins"],
                "peak_time_deltas_dt_half_minus_dt": comparison[
                    "peak_time_deltas_dt_half_minus_dt"
                ],
                "max_abs_peak_time_delta": (
                    max(
                        abs(value)
                        for value in comparison[
                            "peak_time_deltas_dt_half_minus_dt"
                        ]
                    )
                    if comparison["peak_time_deltas_dt_half_minus_dt"]
                    is not None
                    else None
                ),
                "kill_fraction_dt": payload["runs"]["dt"]["kill_fraction"],
                "kill_fraction_dt_half": payload["runs"]["dt_half"][
                    "kill_fraction"
                ],
                "kill_fraction_delta_dt_half_minus_dt": comparison[
                    "kill_fraction_delta_dt_half_minus_dt"
                ],
                "event_fraction_in_window_dt": payload["runs"]["dt"][
                    "event_fraction_in_window"
                ],
                "event_fraction_in_window_dt_half": payload["runs"]["dt_half"][
                    "event_fraction_in_window"
                ],
                "file": _dt_path(cfg["name"]).name,
            }
        )
    payload = {
        "schema_version": 1,
        "analysis": "dt-halving checks at reviewer-sensitive configurations",
        "seed": DT_SEED,
        "walkers_per_run": walkers,
        "stable_mode_count_controls": [
            row["name"]
            for row in summary_rows
            if row["role"] == "interior_control" and row["mode_counts_match"]
        ],
        "mode_count_sensitive_cases": [
            row["name"] for row in summary_rows if not row["mode_counts_match"]
        ],
        "interpretation_note": (
            "Interior controls and deliberately boundary-adjacent cases are "
            "reported separately; no global dt-stability claim is made."
        ),
        "rows": summary_rows,
        "wall_seconds": time.perf_counter() - started,
    }
    out = ROBUST_DIR / "dt_halving_summary.json"
    core.write_json(out, payload)
    print(f"dt-halving summary -> {out}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("sensitivity", "seeds", "dt", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--seed-walkers", type=lambda value: int(float(value)), default=SEED_WALKERS
    )
    parser.add_argument(
        "--dt-walkers", type=lambda value: int(float(value)), default=DT_WALKERS
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun completed seed/dt JSON files instead of reusing them",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.seed_walkers <= 0 or args.dt_walkers <= 0:
        raise SystemExit("worker and walker counts must be positive")

    if args.phase in ("sensitivity", "all"):
        run_sensitivity()
    if args.phase in ("seeds", "all"):
        run_seed_repeats(
            workers=args.workers,
            walkers=args.seed_walkers,
            force=args.force,
        )
    if args.phase in ("dt", "all"):
        run_dt_halving(
            workers=min(args.workers, len(DT_CONFIGS)),
            walkers=args.dt_walkers,
            force=args.force,
        )


if __name__ == "__main__":
    main()
