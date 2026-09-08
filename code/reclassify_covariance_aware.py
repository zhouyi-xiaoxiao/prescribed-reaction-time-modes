#!/usr/bin/env python3
"""Covariance-aware prominence reclassification of every stored classifier record.

Motivation (Codex cross-audit finding C1, 2026-08-24): the published classifier
``validate_exact_m_offlattice.classify_histogram`` divides the topographic
prominence (peak minus contour base) by the PEAK-ONLY smoothed-Poisson sigma

    sigma_peak^2 = sum_j A_{i,j}^2 C_j / (N dt)^2 ,

omitting the base-height variance and the peak/base covariance.  For the actual
prominence statistic with the selected (fixed) contour-base bin b, the
Poisson variance of the difference of the two linear statistics is

    sigma_prom^2 = sum_j (A_{i,j} - A_{b,j})^2 C_j / (N dt)^2 ,

where A_{i,j} = k_{i-j} / norm_i are the edge-normalised Gaussian kernel
weights, C_j the raw bin counts, N the walker count and dt the bin width.
This module re-evaluates every stored classifier record (raw binned counts are
retained in all production/W1/W2/robustness/W4/W5 JSONs) under the
covariance-aware statistic, with the unchanged acceptance rule

    significant  iff  prominence >= 5 * sigma_prom
                 and  prominence >= 0.05 * max smoothed height ,

and reports (old z, new z, old verdict, new verdict) for every local maximum,
re-derives the W2 operational-crossing brackets from the stored probes, and
recomputes the classifier-sensitivity sweeps under the corrected statistic.

No Monte Carlo rerun is performed here; everything is recomputed from stored
sufficient statistics.  (The W3 jitter stream stores no per-replica counts;
its deterministic recheck lives in ``w3_jitter_covariance_recheck.py``.)

Outputs:
    artifacts/data/exact_m_prr_upgrade/covariance_aware_reclassification.json
    artifacts/data/exact_m_prr_upgrade/covariance_aware_reclassification_summary.txt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
DATA = REPORT / "artifacts" / "data"
PRODUCTION_DIR = DATA / "exact_m_offlattice_production"
UPGRADE = DATA / "exact_m_prr_upgrade"
W1_DIR = UPGRADE / "w1_phase_diagram"
W2_DIR = UPGRADE / "w2_b0_empirical"
OUT_JSON = UPGRADE / "covariance_aware_reclassification.json"
OUT_TXT = UPGRADE / "covariance_aware_reclassification_summary.txt"

SIGMA_FACTOR = 5.0
RELATIVE_FLOOR = 0.05
PRIMARY_BANDWIDTHS = (0.03, 0.04, 0.05)
PRIMARY_RELATIVE_FLOORS = (0.03, 0.05, 0.07)
WIDE_BANDWIDTHS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
WIDE_RELATIVE_FLOORS = (0.0, 0.01, 0.03, 0.05, 0.07, 0.10)


# ----------------------------------------------------------------------------
# Classifier re-evaluation (both sigma conventions) from stored counts.
# ----------------------------------------------------------------------------


def classify_both(
    counts,
    edges,
    walkers: int,
    *,
    bandwidth: float,
    sigma_factor: float = SIGMA_FACTOR,
    relative_floor: float = RELATIVE_FLOOR,
) -> dict:
    """Re-run the published smoothing/peak logic; report both sigma conventions.

    The smoothing pipeline, local-maximum scan and contour-base construction
    reproduce ``validate_exact_m_offlattice.classify_histogram`` bit-for-bit;
    additionally the contour-base BIN of each maximum is tracked so the
    covariance-aware prominence sigma can be evaluated.
    """

    counts = np.asarray(counts, dtype=np.int64)
    edges = np.asarray(edges, dtype=float)
    n = counts.size
    bin_width = round(float((edges[-1] - edges[0]) / n), 14)
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (walkers * bin_width)

    radius = max(1, int(math.ceil(4.0 * bandwidth / bin_width)))
    offsets = np.arange(-radius, radius + 1) * bin_width
    kernel = np.exp(-(offsets**2) / (2.0 * bandwidth**2))
    kernel /= kernel.sum()
    norm = np.convolve(np.ones(n), kernel, mode="same")
    smoothed = np.convolve(density, kernel, mode="same") / norm
    variance_peak = (
        np.convolve(counts.astype(float), kernel**2, mode="same")
        / (norm**2)
        / (walkers * bin_width) ** 2
    )
    sigma_peak = np.sqrt(variance_peak)

    # Dense linear-coefficient matrix: smoothed[i] = sum_j A[i,j]*C_j/(N*dt).
    A = np.zeros((n, n))
    for i in range(n):
        for off, kv in zip(range(-radius, radius + 1), kernel):
            j = i - off
            if 0 <= j < n:
                A[i, j] = kv / norm[i]

    global_max = float(smoothed.max()) if smoothed.size else 0.0
    rows = []
    for i in range(1, n - 1):
        if not (smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]):
            continue
        height = float(smoothed[i])
        sides = []  # (lowest value, argmin bin) per direction
        for direction in (-1, 1):
            probe = i + direction
            lowest = height
            low_bin = i
            while 0 <= probe < n and smoothed[probe] <= height:
                if float(smoothed[probe]) < lowest:
                    lowest = float(smoothed[probe])
                    low_bin = probe
                probe += direction
            sides.append((lowest, low_bin))
        base_value, base_bin = max(sides, key=lambda pair: pair[0])
        prominence = height - base_value
        s_old = float(sigma_peak[i])
        coeff = (A[i] - A[base_bin]) / (walkers * bin_width)
        s_new = float(math.sqrt(float(np.sum(counts * coeff * coeff))))
        z_old = prominence / s_old if s_old > 0.0 else math.inf
        z_new = prominence / s_new if s_new > 0.0 else math.inf
        rel = prominence / global_max if global_max > 0.0 else 0.0
        floor_ok = prominence >= relative_floor * global_max
        rows.append(
            {
                "time": float(centres[i]),
                "bin": i,
                "base_bin": int(base_bin),
                "base_time": float(centres[base_bin]),
                "smoothed_height": height,
                "prominence": float(prominence),
                "relative_prominence": float(rel),
                "sigma_peak_only": s_old,
                "sigma_covariance_aware": s_new,
                "z_peak_only": float(z_old),
                "z_covariance_aware": float(z_new),
                "significant_peak_only": bool(z_old >= sigma_factor and floor_ok),
                "significant_covariance_aware": bool(
                    z_new >= sigma_factor and floor_ok
                ),
            }
        )
    return {
        "bin_width": bin_width,
        "bandwidth": bandwidth,
        "global_max": global_max,
        "smoothed": smoothed,
        "rows": rows,
        "mode_count_peak_only": sum(r["significant_peak_only"] for r in rows),
        "mode_count_covariance_aware": sum(
            r["significant_covariance_aware"] for r in rows
        ),
        "significant_times_covariance_aware": [
            r["time"] for r in rows if r["significant_covariance_aware"]
        ],
    }


# ----------------------------------------------------------------------------
# Record discovery.
# ----------------------------------------------------------------------------


def find_classifiers(node, path=""):
    if isinstance(node, dict):
        keys = ("counts", "edges", "local_maxima", "significant_maxima", "mode_count")
        if all(k in node for k in keys):
            yield path, node
        for key, value in node.items():
            yield from find_classifiers(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from find_classifiers(value, f"{path}/{index}")


def walkers_of(payload: dict, path: Path) -> int:
    parameters = payload.get("parameters", {})
    config = parameters.get("config", {})
    for candidate in (
        config.get("walkers"),
        config.get("walkers_per_run"),
        parameters.get("walkers"),
    ):
        if candidate:
            return int(candidate)
    raise KeyError(f"walker count not found in {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_record_files():
    yield from sorted(PRODUCTION_DIR.glob("*.json"))
    yield from sorted(UPGRADE.rglob("*.json"))


# ----------------------------------------------------------------------------
# Main analysis.
# ----------------------------------------------------------------------------


def main() -> None:
    skip_names = {
        OUT_JSON.name,
        "campaign_summary.json",
        "preflight_theory.json",
        "classifier_sensitivity.json",
        "dt_halving_summary.json",
        "seed_repeat_summary.json",
        "jitter_robustness.json",
        "B0_empirical.json",
        "covariance_aware_recheck.json",
    }
    records = []
    flips = []
    checked_files = 0
    for path in iter_record_files():
        if path.name in skip_names or "w3_jitter" in path.parts:
            continue
        payload = read_json(path)
        found = list(find_classifiers(payload))
        if not found:
            continue
        walkers = walkers_of(payload, path)
        checked_files += 1
        for loc, stored in found:
            factor = float(stored.get("prominence_sigma_factor", SIGMA_FACTOR))
            floor = float(stored.get("prominence_relative_floor", RELATIVE_FLOOR))
            result = classify_both(
                stored["counts"],
                stored["edges"],
                walkers,
                bandwidth=float(stored["bandwidth"]),
                sigma_factor=factor,
                relative_floor=floor,
            )
            # Cross-checks against the stored record (alignment + arithmetic).
            stored_sm = np.asarray(stored["smoothed_density"], dtype=float)
            if not np.allclose(result["smoothed"], stored_sm, rtol=0, atol=1e-12):
                raise AssertionError(f"smoothed mismatch in {path}:{loc}")
            stored_rows = stored["local_maxima"]
            if len(stored_rows) != len(result["rows"]):
                raise AssertionError(f"row-count mismatch in {path}:{loc}")
            for srow, row in zip(stored_rows, result["rows"]):
                if abs(srow["time"] - row["time"]) > 1e-12:
                    raise AssertionError(f"row time mismatch in {path}:{loc}")
                if not math.isclose(
                    srow["prominence"], row["prominence"], rel_tol=1e-10, abs_tol=1e-15
                ):
                    raise AssertionError(f"prominence mismatch in {path}:{loc}")
                if not math.isclose(
                    srow["poisson_sigma"],
                    row["sigma_peak_only"],
                    rel_tol=1e-10,
                    abs_tol=1e-18,
                ):
                    raise AssertionError(f"peak-only sigma mismatch in {path}:{loc}")
                if bool(srow["significant"]) != row["significant_peak_only"]:
                    raise AssertionError(f"stored verdict mismatch in {path}:{loc}")
            if int(stored["mode_count"]) != result["mode_count_peak_only"]:
                raise AssertionError(f"mode-count mismatch in {path}:{loc}")

            rel_path = str(path.relative_to(REPORT))
            record = {
                "file": rel_path,
                "classifier_path": loc,
                "walkers": walkers,
                "prominence_sigma_factor": factor,
                "prominence_relative_floor": floor,
                "mode_count_old": int(stored["mode_count"]),
                "mode_count_new": result["mode_count_covariance_aware"],
                "maxima": [
                    {
                        "time": row["time"],
                        "prominence": row["prominence"],
                        "relative_prominence": row["relative_prominence"],
                        "z_old": row["z_peak_only"],
                        "z_new": row["z_covariance_aware"],
                        "verdict_old": row["significant_peak_only"],
                        "verdict_new": row["significant_covariance_aware"],
                    }
                    for row in result["rows"]
                ],
            }
            records.append(record)
            for row in record["maxima"]:
                if row["verdict_old"] != row["verdict_new"]:
                    flips.append(
                        {
                            "file": rel_path,
                            "classifier_path": loc,
                            "time": row["time"],
                            "z_old": row["z_old"],
                            "z_new": row["z_new"],
                            "relative_prominence": row["relative_prominence"],
                            "verdict_old": row["verdict_old"],
                            "verdict_new": row["verdict_new"],
                        }
                    )

    all_rows = [row for rec in records for row in rec["maxima"]]
    ratios = [
        row["z_new"] / row["z_old"]
        for row in all_rows
        if row["z_old"] > 0 and math.isfinite(row["z_old"])
    ]
    direction = {
        "n_local_maxima": len(all_rows),
        "n_records": len(records),
        "max_ratio_new_over_old": max(ratios),
        "min_ratio_new_over_old": min(ratios),
        "n_new_z_above_old_z": sum(1 for r in ratios if r > 1.0),
        "conservative_everywhere": max(ratios) <= 1.0,
        "max_inflation_factor_old_over_new": 1.0 / min(ratios),
        "min_inflation_factor_old_over_new": 1.0 / max(ratios),
    }

    # ------------------------------------------------------------------
    # W1 final phase-diagram map under the covariance-aware rule.
    # ------------------------------------------------------------------
    w1_final = {}
    for rec in records:
        if "w1_phase_diagram" not in rec["file"]:
            continue
        payload = read_json(REPORT / rec["file"])
        cfg = payload["parameters"]["config"]
        key = (int(cfg["m"]), float(cfg["eps"]), float(cfg["budget"]))
        entry = w1_final.get(key)
        if entry is None or bool(cfg["refined"]):
            w1_final[key] = {
                "refined": bool(cfg["refined"]),
                "mode_count_old": rec["mode_count_old"],
                "mode_count_new": rec["mode_count_new"],
                "file": rec["file"],
            }
    w1_map_changes = [
        {"m": k[0], "eps": k[1], "budget": k[2], **v}
        for k, v in sorted(w1_final.items())
        if v["mode_count_old"] != v["mode_count_new"]
    ]
    m2_final_counts = sorted(
        {v["mode_count_new"] for k, v in w1_final.items() if k[0] == 2}
    )
    staircase = {}
    for (m, eps, budget), v in sorted(w1_final.items()):
        if m == 3:
            staircase.setdefault(eps, {})[budget] = v["mode_count_new"]

    # ------------------------------------------------------------------
    # W2 crossings re-derived from stored probes under the new rule.
    # ------------------------------------------------------------------
    b0_summary = read_json(W2_DIR / "B0_empirical.json")
    by_probe_file = {
        rec["file"].split("/")[-1]: rec
        for rec in records
        if "w2_b0_empirical" in rec["file"]
    }
    w2_chains = []
    for chain in b0_summary["chains"]:
        m, eps = int(chain["m"]), float(chain["eps"])
        edge = chain.get("basin_edge_last_g_valley_time")
        probes = []
        for probe in chain.get("probes", []):
            rec = by_probe_file[probe["file"]]
            payload_rows = rec["maxima"]
            old_verdict = bool(probe["verdict"])
            new_rows = [
                row
                for row in payload_rows
                if row["verdict_new"] and row["time"] > edge
            ]
            old_rows = [
                row
                for row in payload_rows
                if row["verdict_old"] and row["time"] > edge
            ]
            if bool(old_rows) != old_verdict:
                raise AssertionError(
                    f"stored W2 verdict mismatch for {probe['file']}"
                )
            probes.append(
                {
                    "budget": float(probe["budget"]),
                    "file": probe["file"],
                    "verdict_old": old_verdict,
                    "verdict_new": bool(new_rows),
                    "last_mode_z_old": max((r["z_old"] for r in old_rows), default=None),
                    "last_mode_z_new": max((r["z_new"] for r in new_rows), default=None)
                    if new_rows
                    else max(
                        (
                            r["z_new"]
                            for r in payload_rows
                            if r["time"] > edge
                        ),
                        default=None,
                    ),
                }
            )
        probes.sort(key=lambda p: p["budget"])
        passing = [p["budget"] for p in probes if p["verdict_new"]]
        failing = [p["budget"] for p in probes if not p["verdict_new"]]
        monotone_violations = sum(
            1
            for i, p in enumerate(probes)
            for q in probes[i + 1 :]
            if (not p["verdict_new"]) and q["verdict_new"]
        )
        entry = {
            "m": m,
            "eps": eps,
            "basin_edge_last_g_valley_time": edge,
            "old_status": chain.get("status"),
            "old_b0": chain.get("b0"),
            "old_bracket": chain.get("b0_bracket"),
            "old_lower_bound": chain.get("b0_lower_bound"),
            "probes": probes,
            "monotonicity_violations_new_rule": monotone_violations,
        }
        if chain.get("status") == "right_censored":
            # Single stored probe at the scan limit.
            top = probes[-1]
            if top["verdict_new"]:
                entry["new_status"] = "right_censored"
                entry["new_b0"] = None
                entry["new_lower_bound"] = 8.0
            else:
                entry["new_status"] = "scan_limit_probe_no_longer_passes"
                entry["new_b0"] = None
        elif not passing:
            entry["new_status"] = "no_certified_crossing_at_1e6_walkers"
            entry["new_b0"] = None
            entry["new_bracket"] = None
        else:
            lower = max(passing)
            upper_candidates = [b for b in failing if b > lower]
            if not upper_candidates:
                entry["new_status"] = "right_censored_on_stored_probes"
                entry["new_b0"] = None
                entry["new_lower_bound"] = lower
            else:
                upper = min(upper_candidates)
                entry["new_status"] = "bisected"
                entry["new_bracket"] = [lower, upper]
                entry["new_b0"] = math.sqrt(lower * upper)
                entry["bracket_unchanged"] = (
                    chain.get("b0_bracket") is not None
                    and math.isclose(lower, chain["b0_bracket"][0])
                    and math.isclose(upper, chain["b0_bracket"][1])
                )
        w2_chains.append(entry)

    # W1 column coverage for chains without a passing probe (supporting the
    # all-fail statement over the full scanned budget range).
    for entry in w2_chains:
        if entry.get("new_status") != "no_certified_crossing_at_1e6_walkers":
            continue
        m, eps, edge = entry["m"], entry["eps"], entry["basin_edge_last_g_valley_time"]
        coverage = []
        for rec in records:
            if "w1_phase_diagram" not in rec["file"]:
                continue
            payload = read_json(REPORT / rec["file"])
            cfg = payload["parameters"]["config"]
            if int(cfg["m"]) != m or float(cfg["eps"]) != eps or cfg["refined"]:
                continue
            last_new = [
                row
                for row in rec["maxima"]
                if row["verdict_new"] and row["time"] > edge
            ]
            coverage.append(
                {
                    "budget": float(cfg["budget"]),
                    "w1_last_mode_present_new": bool(last_new),
                }
            )
        coverage.sort(key=lambda r: r["budget"])
        entry["w1_column_last_mode_coverage_new_rule"] = coverage

    # ------------------------------------------------------------------
    # Sensitivity sweeps under the covariance-aware rule.
    # ------------------------------------------------------------------
    def sweep(stored, walkers, bandwidths, floors):
        variants = []
        for bandwidth in bandwidths:
            for floor in floors:
                res = classify_both(
                    stored["counts"],
                    stored["edges"],
                    walkers,
                    bandwidth=bandwidth,
                    relative_floor=floor,
                )
                variants.append(
                    {
                        "bandwidth": bandwidth,
                        "prominence_relative_floor": floor,
                        "mode_count": res["mode_count_covariance_aware"],
                        "peak_times": res["significant_times_covariance_aware"],
                    }
                )
        return variants

    # Final W1 cells (refined replaces grid).
    final_paths = {}
    for path in sorted(W1_DIR.glob("cell_m*_eps*_B*.json")):
        payload = read_json(path)
        cfg = payload["parameters"]["config"]
        key = (int(cfg["m"]), float(cfg["eps"]), float(cfg["budget"]))
        if key not in final_paths or bool(cfg["refined"]):
            final_paths[key] = (path, payload)
    w1_sensitivity = []
    for key in sorted(final_paths):
        path, payload = final_paths[key]
        cfg = payload["parameters"]["config"]
        variants = sweep(
            payload["results"]["classifier"],
            int(cfg["walkers"]),
            PRIMARY_BANDWIDTHS,
            PRIMARY_RELATIVE_FLOORS,
        )
        counts = sorted({v["mode_count"] for v in variants})
        w1_sensitivity.append(
            {
                "m": key[0],
                "eps": key[1],
                "budget": key[2],
                "source_file": path.name,
                "mode_counts_over_grid": counts,
                "sensitive": len(counts) > 1,
            }
        )
    w1_sensitive_cells = [row for row in w1_sensitivity if row["sensitive"]]

    # Wide anchor checks (7x6 grid), as in exact_m_prr_robustness.py, plus
    # the m=2 production anchor named alongside them in the supplement.
    anchor_specs = (
        ("m2_anchor", PRODUCTION_DIR / "m2_eps0.1_B1_w50-50.json"),
        ("m3_anchor", PRODUCTION_DIR / "m3_eps0.1_B1_w33-33-33.json"),
        ("m5_anchor", UPGRADE / "w4_m5_demo" / "m5_demo.json"),
        ("d3_anchor", UPGRADE / "w5_d3_spotcheck" / "d3_spotcheck.json"),
    )
    wide_anchor_checks = []
    for name, path in anchor_specs:
        payload = read_json(path)
        variants = sweep(
            payload["results"]["classifier"],
            walkers_of(payload, path),
            WIDE_BANDWIDTHS,
            WIDE_RELATIVE_FLOORS,
        )
        counts = sorted({v["mode_count"] for v in variants})
        wide_anchor_checks.append(
            {
                "name": name,
                "source_file": str(path.relative_to(REPORT)),
                "baseline_mode_count": int(payload["results"]["mode_count"]),
                "mode_counts_over_wide_grid": counts,
                "stable": len(counts) == 1,
            }
        )

    # W2 probe sensitivity + transition summaries under the new rule.
    basin_edges = {
        (int(c["m"]), float(c["eps"])): float(c["basin_edge_last_g_valley_time"])
        for c in b0_summary["chains"]
        if c.get("basin_edge_last_g_valley_time") is not None
    }
    w2_rows = []
    for path in sorted(W2_DIR.glob("probe_*.json")):
        payload = read_json(path)
        cfg = payload["parameters"]["config"]
        edge = basin_edges[(int(cfg["m"]), float(cfg["eps"]))]
        variants = sweep(
            payload["results"]["classifier"],
            int(cfg["walkers"]),
            PRIMARY_BANDWIDTHS,
            PRIMARY_RELATIVE_FLOORS,
        )
        for v in variants:
            v["last_mode_present"] = bool(any(t > edge for t in v["peak_times"]))
        verdicts = sorted({v["last_mode_present"] for v in variants})
        w2_rows.append(
            {
                "m": int(cfg["m"]),
                "eps": float(cfg["eps"]),
                "budget": float(cfg["budget"]),
                "source_file": path.name,
                "sensitive": len(verdicts) > 1,
                "variants": variants,
            }
        )
    w2_sensitive = [row for row in w2_rows if row["sensitive"]]

    grouped = {}
    for row in w2_rows:
        grouped.setdefault((row["m"], row["eps"]), []).append(row)
    w2_transitions = []
    for (m, eps), rows in sorted(grouped.items()):
        settings = []
        for vi, reference in enumerate(rows[0]["variants"]):
            observations = sorted(
                (row["budget"], row["variants"][vi]["last_mode_present"])
                for row in rows
            )
            passing = [b for b, ok in observations if ok]
            failing = [b for b, ok in observations if not ok]
            violations = sum(
                1
                for i, (_, lv) in enumerate(observations)
                for _, rv in observations[i + 1 :]
                if (not lv) and rv
            )
            if not failing:
                status, bracket, estimate = (
                    "right_censored_above_max_stored_probe",
                    [max(passing), None],
                    None,
                )
            elif not passing:
                status, bracket, estimate = (
                    "no_passing_stored_probe",
                    [None, min(failing)],
                    None,
                )
            else:
                lower = max(passing)
                uppers = [b for b in failing if b > lower]
                if not uppers:
                    status, bracket, estimate = (
                        "nonmonotone_or_right_censored",
                        [lower, None],
                        None,
                    )
                else:
                    upper = min(uppers)
                    status, bracket, estimate = (
                        "bracketed_on_stored_probes",
                        [lower, upper],
                        math.sqrt(lower * upper),
                    )
            settings.append(
                {
                    "bandwidth": reference["bandwidth"],
                    "prominence_relative_floor": reference[
                        "prominence_relative_floor"
                    ],
                    "status": status,
                    "stored_probe_bracket": bracket,
                    "geometric_midpoint_if_bracketed": estimate,
                    "monotonicity_violations": violations,
                }
            )
        estimates = [
            s["geometric_midpoint_if_bracketed"]
            for s in settings
            if s["geometric_midpoint_if_bracketed"] is not None
        ]
        n_no_pass = sum(1 for s in settings if s["status"] == "no_passing_stored_probe")
        n_right = sum(
            1
            for s in settings
            if s["status"]
            in ("right_censored_above_max_stored_probe", "nonmonotone_or_right_censored")
        )
        w2_transitions.append(
            {
                "m": m,
                "eps": eps,
                "bracketed_midpoint_range": (
                    [min(estimates), max(estimates)] if estimates else None
                ),
                "n_settings_no_passing_probe": n_no_pass,
                "n_settings_right_censored": n_right,
                "n_settings_bracketed": len(estimates),
                "settings": settings,
            }
        )

    # ------------------------------------------------------------------
    # Manuscript-facing headline numbers.
    # ------------------------------------------------------------------
    def rec_for(rel: str, loc: str = "/results/classifier"):
        for rec in records:
            if rec["file"].endswith(rel) and rec["classifier_path"] == loc:
                return rec
        raise KeyError(rel)

    production = []
    for path in sorted(PRODUCTION_DIR.glob("*.json")):
        rec = rec_for(f"exact_m_offlattice_production/{path.name}")
        counted = [row for row in rec["maxima"] if row["verdict_new"]]
        counted_old = [row for row in rec["maxima"] if row["verdict_old"]]
        production.append(
            {
                "file": path.name,
                "mode_count_old": rec["mode_count_old"],
                "mode_count_new": rec["mode_count_new"],
                "min_counted_z_old": min((r["z_old"] for r in counted_old), default=None),
                "min_counted_z_new": min((r["z_new"] for r in counted), default=None),
                "subthreshold_maxima": [
                    {
                        "time": r["time"],
                        "z_old": r["z_old"],
                        "z_new": r["z_new"],
                        "relative_prominence": r["relative_prominence"],
                    }
                    for r in rec["maxima"]
                    if not r["verdict_old"]
                ],
            }
        )
    m2_production_min_z_new = min(
        row["min_counted_z_new"]
        for row in production
        if row["file"].startswith("m2_") and "dt5e-4" not in row["file"]
    )
    m2_production_min_z_old = min(
        row["min_counted_z_old"]
        for row in production
        if row["file"].startswith("m2_") and "dt5e-4" not in row["file"]
    )

    m5 = rec_for("w4_m5_demo/m5_demo.json")
    d3 = rec_for("w5_d3_spotcheck/d3_spotcheck.json")
    headline = {
        "production_table": production,
        "m2_production_min_counted_z_old": m2_production_min_z_old,
        "m2_production_min_counted_z_new": m2_production_min_z_new,
        "m5_counted_z_old": [
            r["z_old"] for r in m5["maxima"] if r["verdict_old"]
        ],
        "m5_counted_z_new": [
            r["z_new"] for r in m5["maxima"] if r["verdict_new"]
        ],
        "d3_counted_z_old": [r["z_old"] for r in d3["maxima"] if r["verdict_old"]],
        "d3_counted_z_new": [r["z_new"] for r in d3["maxima"] if r["verdict_new"]],
    }
    # B=0.25 partial-recovery subthreshold maximum (production).
    b025 = rec_for("exact_m_offlattice_production/m3_eps0.2_B0.25_w33-33-33.json")
    for row in b025["maxima"]:
        if not row["verdict_old"] and row["time"] > 2.0:
            headline["b025_partial_recovery"] = row
    # dt-halving boundary case: late candidate of (3, 0.175, 0.5).
    for rec in records:
        if rec["file"].endswith("dt_halving/m3_phase_boundary.json"):
            payload_key = rec["classifier_path"]
            late = [r for r in rec["maxima"] if r["time"] > 2.0]
            headline.setdefault("dt_halving_m3_phase_boundary", {})[payload_key] = late

    output = {
        "analysis": (
            "covariance-aware prominence reclassification from stored counts; "
            "sigma_prom^2 = sum_j (A_peak,j - A_base,j)^2 C_j / (N dt)^2; "
            "acceptance rule unchanged (z >= 5 AND prominence >= 5% of max)"
        ),
        "n_files": checked_files,
        "n_classifier_records": len(records),
        "n_local_maxima": len(all_rows),
        "flips": flips,
        "n_flips": len(flips),
        "direction_check": direction,
        "w1_final_map_changes": w1_map_changes,
        "w1_m2_final_mode_counts": m2_final_counts,
        "w1_m3_staircase_new_rule": {
            str(eps): {str(b): c for b, c in sorted(row.items())}
            for eps, row in sorted(staircase.items())
        },
        "w2_chains": w2_chains,
        "sensitivity_new_rule": {
            "w1_final_cells": len(w1_sensitivity),
            "w1_sensitive_cells": len(w1_sensitive_cells),
            "w1_sensitive_coordinates": [
                {"m": r["m"], "eps": r["eps"], "budget": r["budget"]}
                for r in w1_sensitive_cells
            ],
            "w2_probes": len(w2_rows),
            "w2_sensitive_probes": len(w2_sensitive),
            "w2_transition_summaries": w2_transitions,
            "wide_anchor_checks": wide_anchor_checks,
        },
        "headline_numbers": headline,
        "records": records,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------
    # Human summary.
    # ------------------------------------------------------------------
    lines = []
    lines.append("Covariance-aware prominence reclassification (Codex C1 fix)")
    lines.append(
        f"records={len(records)} local_maxima={len(all_rows)} "
        f"files={checked_files} flips={len(flips)}"
    )
    lines.append("")
    lines.append("FLIPS (old verdict != new verdict):")
    for f in flips:
        lines.append(
            f"  {f['file']} {f['classifier_path']} t={f['time']:.2f} "
            f"z_old={f['z_old']:.3f} z_new={f['z_new']:.3f} "
            f"rel={f['relative_prominence']:.3f} "
            f"{f['verdict_old']} -> {f['verdict_new']}"
        )
    lines.append("")
    lines.append(
        "direction: max(z_new/z_old)="
        f"{direction['max_ratio_new_over_old']:.6f} "
        f"min={direction['min_ratio_new_over_old']:.6f} "
        f"conservative_everywhere={direction['conservative_everywhere']}"
    )
    lines.append(
        f"W1 final-map changes: {len(w1_map_changes)}; "
        f"m=2 final mode counts {m2_final_counts}"
    )
    lines.append("")
    lines.append("W2 crossings (old vs covariance-aware):")
    for c in w2_chains:
        old = (
            f"{c['old_b0']:.4f} {c['old_bracket']}"
            if c.get("old_b0")
            else f"status={c['old_status']} lb={c.get('old_lower_bound')}"
        )
        new = (
            f"{c['new_b0']:.4f} {c.get('new_bracket')}"
            if c.get("new_b0")
            else f"status={c['new_status']} lb={c.get('new_lower_bound')}"
        )
        lines.append(f"  m={c['m']} eps={c['eps']:g}: OLD {old} | NEW {new}")
    lines.append("")
    lines.append(
        "sensitivity (new rule): "
        f"W1 sensitive {len(w1_sensitive_cells)}/{len(w1_sensitivity)} "
        f"{[ (r['m'],r['eps'],r['budget']) for r in w1_sensitive_cells ]}; "
        f"W2 sensitive {len(w2_sensitive)}/{len(w2_rows)}; "
        f"anchors stable {[a['stable'] for a in wide_anchor_checks]}"
    )
    for t in w2_transitions:
        lines.append(
            f"  transition m={t['m']} eps={t['eps']:g}: "
            f"midpoint_range={t['bracketed_midpoint_range']} "
            f"no_pass={t['n_settings_no_passing_probe']} "
            f"right_censored={t['n_settings_right_censored']}"
        )
    lines.append("")
    lines.append("headline z updates:")
    lines.append(
        f"  m=2 production weakest counted peak: "
        f"{m2_production_min_z_old:.1f} -> {m2_production_min_z_new:.1f}"
    )
    lines.append(
        f"  m5 counted z: old {sorted(headline['m5_counted_z_old'])} "
        f"new {sorted(headline['m5_counted_z_new'])}"
    )
    lines.append(
        f"  d3 counted z: old {headline['d3_counted_z_old']} "
        f"new {headline['d3_counted_z_new']}"
    )
    if "b025_partial_recovery" in headline:
        b = headline["b025_partial_recovery"]
        lines.append(
            f"  B=0.25 partial recovery: t={b['time']:.2f} "
            f"z {b['z_old']:.3f} -> {b['z_new']:.3f} rel={b['relative_prominence']:.3f}"
        )
    lines.append("")
    lines.append("production table (modes old->new, min counted z old->new):")
    for row in production:
        lines.append(
            f"  {row['file']}: {row['mode_count_old']}->{row['mode_count_new']} "
            f"minZ {row['min_counted_z_old'] and round(row['min_counted_z_old'],1)}"
            f"->{row['min_counted_z_new'] and round(row['min_counted_z_new'],1)}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
