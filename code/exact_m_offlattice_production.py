#!/usr/bin/env python3
"""Deterministic driver for the 18-row off-lattice production table.

The matrix is the one reported in Supplemental Table S-I: seventeen primary
configurations at dt=1e-3 plus the m=2, eps=0.1, B=1, uniform-allocation
dt/2 twin.  Full runs use five million walkers, chunk size 250,000, seed
20260808, tag 1, tmax 4, and the declared h=0.04 classifier.

Phases:
  audit  Validate exact coverage and metadata of an existing 18-file tree.
  smoke  Exercise the complete matrix with 5,000 walkers per row in a separate
         non-production directory.
  full   Run the complete production matrix, resumably, into a chosen output
         subdirectory.  Use a new subdirectory to preserve archived results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import validate_exact_m_offlattice as base


REPORT = HERE.parents[1]
DEFAULT_SUBDIR = "exact_m_offlattice_production"
SMOKE_SUBDIR = "exact_m_offlattice_production_driver_smoke"
SEED = 20260808
FULL_WALKERS = 5_000_000
FULL_CHUNK = 250_000
SMOKE_WALKERS = 5_000
SMOKE_CHUNK = 5_000
DT = 1.0e-3
DT_HALF = 5.0e-4
TMAX = 4.0
BANDWIDTH = 0.04

W2_UNIFORM = (0.5, 0.5)
W2_ASYMMETRIC = (0.7, 0.3)
W3_UNIFORM = (1.0 / 3.0,) * 3
W3_ASYMMETRIC = (0.5, 0.3, 0.2)


def _spec(
    m: int,
    eps: float,
    budget: float,
    weights: tuple[float, ...],
    *,
    dt: float = DT,
    suffix: str = "",
) -> dict:
    stem = base._config_stem(m, eps, budget, weights) + suffix
    return {
        "m": m,
        "eps": eps,
        "budget": budget,
        "weights": weights,
        "dt": dt,
        "suffix": suffix,
        "filename": f"{stem}.json",
    }


PRODUCTION_MATRIX = tuple(
    [
        _spec(m=2, eps=eps, budget=budget, weights=weights)
        for eps in (0.1, 0.2)
        for budget in (0.5, 1.0, 2.0)
        for weights in (W2_UNIFORM, W2_ASYMMETRIC)
    ]
    + [
        _spec(3, 0.1, 1.0, W3_UNIFORM),
        _spec(3, 0.1, 1.0, W3_ASYMMETRIC),
        _spec(3, 0.2, 1.0, W3_UNIFORM),
        _spec(3, 0.2, 1.0, W3_ASYMMETRIC),
        _spec(3, 0.2, 0.25, W3_UNIFORM),
        _spec(
            2,
            0.1,
            1.0,
            W2_UNIFORM,
            dt=DT_HALF,
            suffix="_dt5e-4",
        ),
    ]
)


def _parameters(
    spec: dict,
    *,
    walkers: int,
    chunk: int,
) -> tuple[dict, dict]:
    args = SimpleNamespace(
        dt=spec["dt"],
        tmax=TMAX,
        walkers=walkers,
        chunk=chunk,
        seed=SEED,
        bandwidth=BANDWIDTH,
    )
    config = {
        "m": spec["m"],
        "eps": spec["eps"],
        "budget": spec["budget"],
        "weights": list(spec["weights"]),
        "target_times": list(base.TARGET_TIMES[spec["m"]]),
        "walkers": walkers,
    }
    # The archived production records predate serialization of this field,
    # but the deterministic driver fixes the production stream tag to 1.
    # New reruns are therefore fully self-describing without modifying the
    # immutable archived measurements.
    return {
        **base._base_parameters(args),
        "tag": base.TAG_MAIN,
        "config": config,
    }, config


def _validate_payload(
    path: Path,
    spec: dict,
    *,
    walkers: int,
    chunk: int,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    config = parameters["config"]
    expected = {
        "m": spec["m"],
        "eps": spec["eps"],
        "budget": spec["budget"],
        "weights": list(spec["weights"]),
        "walkers": walkers,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(
                f"{path.name}: config {key}={config[key]!r}, expected {value!r}"
            )
    checks = {
        "dt": spec["dt"],
        "tmax": TMAX,
        "walkers": walkers,
        "chunk": chunk,
        "seed": SEED,
        "bandwidth": BANDWIDTH,
    }
    for key, value in checks.items():
        if parameters[key] != value:
            raise ValueError(
                f"{path.name}: parameters.{key}={parameters[key]!r}, "
                f"expected {value!r}"
            )
    results = payload["results"]
    classifier = results["classifier"]
    if len(classifier["counts"]) != 150 or len(classifier["edges"]) != 151:
        raise ValueError(f"{path.name}: incomplete classifier histogram")
    if not payload["validation_gates"]["gate3_mass_balance"]["passed"]:
        raise ValueError(f"{path.name}: failed mass-balance gate")
    return payload


def audit(directory: Path, *, walkers: int, chunk: int) -> dict:
    expected = {spec["filename"] for spec in PRODUCTION_MATRIX}
    observed = {path.name for path in directory.glob("*.json")}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            f"production coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    rows = []
    for spec in PRODUCTION_MATRIX:
        payload = _validate_payload(
            directory / spec["filename"],
            spec,
            walkers=walkers,
            chunk=chunk,
        )
        rows.append(
            {
                "file": spec["filename"],
                "dt": spec["dt"],
                "mode_count": payload["results"]["mode_count"],
                "mass_balance": True,
            }
        )
    result = {
        "status": "PASS",
        "directory": directory.name,
        "expected_files": 18,
        "validated_files": len(rows),
        "primary_dt_rows": sum(row["dt"] == DT for row in rows),
        "dt_half_rows": sum(row["dt"] == DT_HALF for row in rows),
        "walkers_per_row": walkers,
        "seed": SEED,
        "stream_tag": base.TAG_MAIN,
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _run_one(task: tuple[dict, int, int, str, dict, bool]) -> dict:
    spec, walkers, chunk, output_text, gate_zero, force = task
    output = Path(output_text)
    path = output / spec["filename"]
    if path.exists() and not force:
        payload = _validate_payload(
            path,
            spec,
            walkers=walkers,
            chunk=chunk,
        )
        return {
            "file": path.name,
            "mode_count": payload["results"]["mode_count"],
            "runtime_seconds": payload["results"]["runtime_seconds"],
            "reused": True,
        }

    result = base.run_config(
        m=spec["m"],
        eps=spec["eps"],
        budget=spec["budget"],
        weights=spec["weights"],
        walkers=walkers,
        chunk=chunk,
        dt=spec["dt"],
        tmax=TMAX,
        seed=SEED,
        tag=base.TAG_MAIN,
        verbose=False,
    )
    summary = base.summarize_config(result, bandwidth=BANDWIDTH, tmax=TMAX)
    parameters, _ = _parameters(spec, walkers=walkers, chunk=chunk)
    payload = {
        "parameters": parameters,
        "validation_gates": {
            "gate1_zero_budget_no_kills": gate_zero,
            "gate2_kill_probability_max": summary["kill_probability_max"],
            "gate2_kill_probability_in_unit_interval": bool(
                0.0 <= summary["kill_probability_max"] <= 1.0
            ),
            "gate3_mass_balance": {
                "kills": summary["kills"],
                "survivors": summary["survivors"],
                "walkers": walkers,
                "passed": bool(summary["kills"] + summary["survivors"] == walkers),
            },
        },
        "results": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    base._write_json(path, payload)
    return {
        "file": path.name,
        "mode_count": summary["mode_count"],
        "runtime_seconds": summary["runtime_seconds"],
        "reused": False,
    }


def run_matrix(
    directory: Path,
    *,
    walkers: int,
    chunk: int,
    workers: int,
    force: bool,
) -> None:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    gates = {
        dt: base.gate_zero_budget(SEED, dt=dt) for dt in (DT, DT_HALF)
    }
    tasks = [
        (spec, walkers, chunk, str(directory), gates[spec["dt"]], force)
        for spec in PRODUCTION_MATRIX
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/18] {row['file']}: modes={row['mode_count']} "
                f"{'reused' if row['reused'] else 'ran'}",
                flush=True,
            )
    result = audit(directory, walkers=walkers, chunk=chunk)
    result["wall_seconds_this_invocation"] = time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "smoke", "full"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--walkers", type=lambda v: int(float(v)), default=None)
    parser.add_argument("--chunk", type=lambda v: int(float(v)), default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    if args.phase == "audit":
        subdir = args.output_subdir or DEFAULT_SUBDIR
        audit(
            REPORT / "artifacts" / "data" / subdir,
            walkers=args.walkers or FULL_WALKERS,
            chunk=args.chunk or FULL_CHUNK,
        )
        return

    if args.phase == "smoke":
        subdir = args.output_subdir or SMOKE_SUBDIR
        walkers = args.walkers or SMOKE_WALKERS
        chunk = args.chunk or min(SMOKE_CHUNK, walkers)
    else:
        subdir = args.output_subdir or f"{DEFAULT_SUBDIR}_rerun"
        walkers = args.walkers or FULL_WALKERS
        chunk = args.chunk or FULL_CHUNK
    if walkers <= 0 or chunk <= 0:
        raise SystemExit("walker and chunk counts must be positive")
    run_matrix(
        REPORT / "artifacts" / "data" / subdir,
        walkers=walkers,
        chunk=chunk,
        workers=min(args.workers, len(PRODUCTION_MATRIX)),
        force=args.force,
    )


if __name__ == "__main__":
    main()
