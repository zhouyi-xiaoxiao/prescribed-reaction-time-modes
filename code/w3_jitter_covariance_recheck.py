#!/usr/bin/env python3
"""Deterministic W3 jitter recheck under the covariance-aware prominence rule.

The W3 stream stores per-replica mode counts but not the binned counts, so the
covariance-aware statistic (Codex cross-audit finding C1) cannot be applied to
the stored JSONs directly.  Every W3 replica is, however, fully deterministic:
the perturbation draw and the simulation stream are both keyed by
(SEED, tag, m, eta_index, replica).  This driver re-simulates every replica
bit-for-bit, verifies the reproduced peak-only mode count against the stored
row (700/700 required), and reclassifies each replica under the
covariance-aware rule.

Nothing under w3_jitter/ is modified; the result is written to
    artifacts/data/exact_m_prr_upgrade/w3_jitter/covariance_aware_recheck.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_upgrade_core as core
import exact_m_prr_upgrade_w3 as w3
import validate_exact_m_offlattice as base
from reclassify_covariance_aware import classify_both

W3_DIR = core.UPGRADE_DATA / "w3_jitter"
OUT_PATH = W3_DIR / "covariance_aware_recheck.json"


def replica_both(task: tuple) -> dict:
    """Re-simulate one W3 replica; classify under both sigma conventions."""
    kind, m, eta_index, replica = task
    anchor = w3.ANCHORS[m]
    eps, budget = anchor["eps"], anchor["budget"]
    centres0 = core.centres_z_for(m)
    weights = np.asarray(anchor["weights"], dtype=float)

    if kind == "centre":
        s = w3.min_adjacent_spacing(m)
        eta = w3.ETA_GRID[eta_index]
        draw_rng = np.random.Generator(
            np.random.Philox(
                np.random.SeedSequence(
                    [w3.SEED, core.TAG_W3_DRAWS, 1, m, eta_index, replica]
                )
            )
        )
        centres = centres0 + eta * s * draw_rng.standard_normal(centres0.size)
        tag = core.TAG_W3_CENTRE
        extra = (1, eta_index, replica)
    else:
        eta = None
        draw_rng = np.random.Generator(
            np.random.Philox(
                np.random.SeedSequence(
                    [w3.SEED, core.TAG_W3_DRAWS, 2, m, 0, replica]
                )
            )
        )
        centres = centres0.copy()
        weights = draw_rng.dirichlet(w3.DIRICHLET_CONC * weights)
        tag = core.TAG_W3_WEIGHT
        extra = (2, 0, replica)

    result = core.run_config_general(
        eps=eps,
        budget=budget,
        weights=tuple(float(w) for w in weights),
        centres_z=centres,
        walkers=w3.REPLICA_WALKERS,
        chunk=w3.CHUNK,
        dt=w3.DT,
        tmax=w3.TMAX,
        seed=w3.SEED,
        tag=tag,
        n_perp=1,
        extra_entropy=extra,
    )
    classifier = base.classify_modes(
        result["kill_times"], w3.REPLICA_WALKERS, bandwidth=w3.BANDWIDTH
    )
    both = classify_both(
        classifier["counts"],
        classifier["edges"],
        w3.REPLICA_WALKERS,
        bandwidth=w3.BANDWIDTH,
    )
    if both["mode_count_peak_only"] != classifier["mode_count"]:
        raise AssertionError("internal peak-only reclassification mismatch")
    return {
        "kind": kind,
        "m": m,
        "eta_index": eta_index,
        "eta": eta,
        "replica": replica,
        "mode_count_old": int(classifier["mode_count"]),
        "mode_count_new": int(both["mode_count_covariance_aware"]),
        "success_old": bool(classifier["mode_count"] == m),
        "success_new": bool(both["mode_count_covariance_aware"] == m),
        "min_counted_z_new": min(
            (
                r["z_covariance_aware"]
                for r in both["rows"]
                if r["significant_covariance_aware"]
            ),
            default=None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N tasks (smoke test)")
    args = parser.parse_args()
    started = time.perf_counter()

    # Stored per-replica verdicts for the determinism check.
    stored: dict[tuple, dict] = {}
    stored_points: dict[tuple, dict] = {}
    for path in sorted(W3_DIR.glob("point_*.json")):
        point = json.loads(path.read_text(encoding="utf-8"))
        kind = point["kind"]
        m = point["m"]
        eta = point.get("eta")
        eta_index = (
            w3.ETA_GRID.index(eta) if kind == "centre" else 0
        )
        stored_points[(kind, m, eta_index)] = point
        for row in point["replica_rows"]:
            stored[(kind, m, eta_index, row["replica"])] = row

    tasks = [
        ("centre", m, ei, rep)
        for m in sorted(w3.ANCHORS)
        for ei in range(len(w3.ETA_GRID))
        for rep in range(w3.REPLICAS)
    ] + [
        ("weight", m, 0, rep)
        for m in sorted(w3.ANCHORS)
        for rep in range(w3.REPLICAS)
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    rows = []
    mismatches = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replica_both, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            done += 1
            key = (row["kind"], row["m"], row["eta_index"], row["replica"])
            srow = stored.get(key)
            row["stored_mode_count"] = srow["mode_count"] if srow else None
            row["reproduced"] = bool(
                srow is not None and srow["mode_count"] == row["mode_count_old"]
            )
            row["g_has_m_window_maxima"] = (
                bool(srow["g_has_m_window_maxima"]) if srow else None
            )
            if not row["reproduced"]:
                mismatches.append(key)
            rows.append(row)
            if done % 50 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] replicas done", flush=True)

    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["kind"], row["m"], row["eta_index"]), []).append(row)
    points = []
    for (kind, m, eta_index), bucket in sorted(buckets.items()):
        n = len(bucket)
        succ_old = sum(r["success_old"] for r in bucket)
        succ_new = sum(r["success_new"] for r in bucket)
        lo, hi = core.wilson_ci(succ_new, n)
        agree_new = sum(
            r["success_new"] == r["g_has_m_window_maxima"] for r in bucket
        )
        flips = sorted(
            r["replica"] for r in bucket if r["success_old"] != r["success_new"]
        )
        spoint = stored_points.get((kind, m, eta_index), {})
        points.append(
            {
                "kind": kind,
                "m": m,
                "eta": w3.ETA_GRID[eta_index] if kind == "centre" else None,
                "replicas": n,
                "successes_old_stored": spoint.get("successes"),
                "successes_old_reproduced": succ_old,
                "successes_new": succ_new,
                "survival_probability_new": succ_new / n,
                "wilson_95_ci_new": [lo, hi],
                "classifier_theory_agreement_new": agree_new / n,
                "replica_flips_old_vs_new": flips,
                "mode_count_histogram_new": {
                    str(k): int(sum(1 for r in bucket if r["mode_count_new"] == k))
                    for k in sorted({r["mode_count_new"] for r in bucket})
                },
            }
        )
        print(
            f"point {kind} m={m} "
            f"eta={points[-1]['eta']}: old {succ_old}/{n} -> new {succ_new}/{n} "
            f"flips={flips}",
            flush=True,
        )

    payload = {
        "analysis": (
            "deterministic bit-for-bit W3 replica recheck; peak-only verdicts "
            "must reproduce the stored rows; covariance-aware rule applied to "
            "the regenerated binned counts"
        ),
        "replicas_rerun": len(rows),
        "reproduction_mismatches": [list(k) for k in mismatches],
        "all_reproduced": not mismatches,
        "points": points,
        "wall_seconds": time.perf_counter() - started,
    }
    core.write_json(OUT_PATH, payload)
    print(f"summary -> {OUT_PATH}", flush=True)
    print(
        f"all_reproduced={payload['all_reproduced']} "
        f"wall={payload['wall_seconds']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
