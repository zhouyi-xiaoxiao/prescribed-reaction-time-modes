#!/usr/bin/env python3
"""Mean-field hazard--survival evaluation of the operational retention boundary.

The main text quotes, for the parameter-free mean-field law

    f_1(t) = B G(t) exp(-B Lambda(t)),   Lambda(t) = int_0^t G(s) ds,

built from the free exposure clock G of ``validate_exact_m_offlattice``:

* the mean-field operational crossings B_op^mf of the eight W2 chains
  (m = 2, 3; eps = 0.05, 0.10, 0.15, 0.20), obtained by applying the fixed
  covariance-aware classifier and the W2 last-basin pass rule to the expected
  window histogram of f_1 at a nominal walker count;
* the comparison of the mean-field classifier verdict with the covariance-aware
  final W1 phase-diagram map (96 cells);
* the comparison of the mean-field counted peak times with the covariance-aware
  counted maxima of the seventeen production configurations;
* the contact factors c(tau; eps) at the window opening tau.

This script computes all of those quantities from the stored model constants
and writes them, together with every numerical convention, to
``artifacts/data/exact_m_prr_upgrade/mean_field_boundary.json``.  No Monte
Carlo walkers are simulated; the stored classifier records are read only for
the comparisons and for the regression test of the real-valued classifier.

Numerical conventions (all stored in the output under ``conventions``):

* time grid: t_k = k * dt_mf on [0, tmax] with dt_mf = 2.5e-4 (canonical);
  the grid is aligned with the 0.02-wide window bins (80 steps per bin);
  grid checks repeat the headline crossings at dt_mf = 5e-4 and 1.25e-4;
* G(t) is ``validate_exact_m_offlattice.free_exposure_clock`` (contact
  factor by the 400-point folded-wrapped-normal quadrature with 10 images,
  d = 2, matched-law weights);
* Lambda(t) is the composite trapezoid rule on the fine grid from t = 0;
* expected bin counts are N * [exp(-B Lambda(e_j)) - exp(-B Lambda(e_{j+1}))]
  (the exact integral of f_1 over the bin given Lambda), so no separate
  quadrature of f_1 is needed;
* the classifier is the covariance-aware prominence rule of
  ``reclassify_covariance_aware.classify_both`` (bandwidth 0.04, five-sigma
  gate, 5 % relative floor, edge-normalised Gaussian kernel, Poisson plug-in
  variance) evaluated on REAL-VALUED expected counts.  ``classify_both``
  casts counts to ``int64``; the adapter below reproduces its algorithm
  without that cast and is regression-tested against ``classify_both`` on
  every stored integer histogram of the production grid, W1 map, and W2
  probes before any mean-field number is produced;
* the pass rule of a W2 chain is the stored one: a covariance-aware
  significant maximum at a time later than the last valley of G in the
  window (``basin_edge_last_g_valley_time`` of ``B0_empirical.json``);
* the crossing search first scans the declared budget interval
  [0.125, 8] on a geometric grid (97 points, ratio 1.0443), records every
  pass/fail transition, and then bisects the UPPER retention boundary (the
  last pass-to-fail transition) to an absolute tolerance of 1e-9 in B;
  a chain whose largest scanned budget passes is reported as right-censored
  at B = 8, and a chain with no passing scan point as having no certified
  crossing in the scanned range;
* nominal walker counts N = 1e5, 1e6, 1e7 are reported as the sensitivity of
  the upper mean-field crossing to N (through the five-sigma gate only);
  the headline value uses N = 1e6, the W2 probe walker count.

Run from anywhere: ``python code/exact_m_prr_mean_field_boundary.py``
(deterministic; rewrites the JSON in place).  ``--quick`` skips the grid
checks and the N sensitivity (development only; not used for the archive).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import validate_exact_m_offlattice as base  # noqa: E402
import reclassify_covariance_aware as recl  # noqa: E402

REPORT = HERE.parents[1]
DATA = REPORT / "artifacts" / "data"
PRODUCTION_DIR = DATA / "exact_m_offlattice_production"
UPGRADE = DATA / "exact_m_prr_upgrade"
W1_DIR = UPGRADE / "w1_phase_diagram"
W2_DIR = UPGRADE / "w2_b0_empirical"
RECLASS_JSON = UPGRADE / "covariance_aware_reclassification.json"
OUT_JSON = UPGRADE / "mean_field_boundary.json"

DT_MF = 2.5e-4
DT_CHECKS = (5.0e-4, 1.25e-4)
T_MAX = base.DEFAULT_TMAX
BUDGET_INTERVAL = (0.125, 8.0)
SCAN_POINTS = 97
BISECT_TOL = 1e-9
NOMINAL_WALKERS = (100_000, 1_000_000, 10_000_000)
HEADLINE_WALKERS = 1_000_000
BANDWIDTH = base.DEFAULT_BANDWIDTH
SIGMA_FACTOR = base.PROMINENCE_SIGMA_FACTOR
RELATIVE_FLOOR = base.PROMINENCE_RELATIVE_FLOOR
W1_EPS_GRID = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25)
W2_EPS_GRID = (0.05, 0.1, 0.15, 0.2)
G_CHUNK = 500


# ----------------------------------------------------------------------------
# Real-valued covariance-aware classifier (algorithm of classify_both).
# ----------------------------------------------------------------------------


def classify_real(
    counts,
    edges,
    walkers: float,
    *,
    bandwidth: float = BANDWIDTH,
    sigma_factor: float = SIGMA_FACTOR,
    relative_floor: float = RELATIVE_FLOOR,
) -> dict:
    """``reclassify_covariance_aware.classify_both`` on real-valued counts.

    Identical smoothing, local-maximum scan, contour-base construction, and
    both sigma conventions; the only difference is that ``counts`` are kept as
    floating-point expected counts instead of being cast to ``int64``.  The
    regression test ``regression_test_against_classify_both`` verifies
    bit-level agreement on integer histograms.
    """

    counts = np.asarray(counts, dtype=float)
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
        np.convolve(counts, kernel**2, mode="same")
        / (norm**2)
        / (walkers * bin_width) ** 2
    )
    sigma_peak = np.sqrt(variance_peak)

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
        sides = []
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
                "passes_sigma_gate": bool(z_new >= sigma_factor),
                "passes_relative_floor": bool(floor_ok),
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


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def regression_test_against_classify_both() -> dict:
    """Check classify_real == classify_both on the 173 stored integer histograms of the production grid, W1 map, and W2 probes."""

    files = sorted(PRODUCTION_DIR.glob("*.json"))
    files += sorted(W1_DIR.glob("cell_*.json"))
    files += sorted(W2_DIR.glob("probe_*.json"))
    n_records = 0
    n_rows = 0
    max_abs_diff = 0.0
    for path in files:
        payload = _read(path)
        walkers = recl.walkers_of(payload, path)
        for _loc, stored in recl.find_classifiers(payload):
            n_records += 1
            kwargs = dict(
                bandwidth=float(stored["bandwidth"]),
                sigma_factor=float(
                    stored.get("prominence_sigma_factor", SIGMA_FACTOR)
                ),
                relative_floor=float(
                    stored.get("prominence_relative_floor", RELATIVE_FLOOR)
                ),
            )
            ref = recl.classify_both(stored["counts"], stored["edges"], walkers, **kwargs)
            got = classify_real(stored["counts"], stored["edges"], walkers, **kwargs)
            if not np.array_equal(ref["smoothed"], got["smoothed"]):
                raise AssertionError(f"smoothed mismatch in {path.name}")
            if len(ref["rows"]) != len(got["rows"]):
                raise AssertionError(f"row-count mismatch in {path.name}")
            for a, b in zip(ref["rows"], got["rows"]):
                n_rows += 1
                for key in (
                    "time",
                    "prominence",
                    "relative_prominence",
                    "sigma_peak_only",
                    "sigma_covariance_aware",
                    "z_peak_only",
                    "z_covariance_aware",
                ):
                    diff = abs(float(a[key]) - float(b[key]))
                    max_abs_diff = max(max_abs_diff, diff)
                    if diff > 1e-12 * max(1.0, abs(float(a[key]))):
                        raise AssertionError(f"{key} mismatch in {path.name}")
                for key in ("significant_peak_only", "significant_covariance_aware"):
                    if bool(a[key]) != bool(b[key]):
                        raise AssertionError(f"{key} verdict mismatch in {path.name}")
            if ref["mode_count_covariance_aware"] != got["mode_count_covariance_aware"]:
                raise AssertionError(f"mode-count mismatch in {path.name}")
    return {
        "files_checked": len(files),
        "classifier_records_checked": n_records,
        "local_maxima_checked": n_rows,
        "max_abs_difference": max_abs_diff,
        "passed": True,
    }


# ----------------------------------------------------------------------------
# Free clock, exposure, and expected window histogram.
# ----------------------------------------------------------------------------


class FreeClock:
    """G(t) and Lambda(t) on an aligned fine grid for one (m, eps, weights)."""

    def __init__(self, m: int, eps: float, weights: tuple[float, ...], dt: float):
        steps = int(round(T_MAX / dt))
        if abs(steps * dt - T_MAX) > 1e-12:
            raise ValueError("tmax must be an integer multiple of dt")
        self.m, self.eps, self.weights, self.dt = m, eps, tuple(weights), dt
        self.t = np.arange(steps + 1) * dt
        g = np.empty_like(self.t)
        contact = np.empty_like(self.t)
        for start in range(0, self.t.size, G_CHUNK):
            block = base.free_exposure_clock(
                self.t[start : start + G_CHUNK], m=m, eps=eps, weights=self.weights
            )
            g[start : start + G_CHUNK] = block["g"]
            contact[start : start + G_CHUNK] = block["contact_factor"]
        self.g = g
        self.contact = contact
        lam = np.zeros_like(g)
        lam[1:] = np.cumsum(0.5 * dt * (g[1:] + g[:-1]))
        self.lam = lam
        self.window_edges = np.arange(
            base.WINDOW[0], base.WINDOW[1] + 0.5 * base.WINDOW_BIN, base.WINDOW_BIN
        )
        idx = np.rint(self.window_edges / dt).astype(int)
        if np.max(np.abs(idx * dt - self.window_edges)) > 1e-9:
            raise ValueError("window bin edges are not aligned with the time grid")
        self.edge_index = idx

    def expected_counts(self, budget: float, walkers: float) -> np.ndarray:
        survival = np.exp(-budget * self.lam[self.edge_index])
        return walkers * (survival[:-1] - survival[1:])

    def classify(self, budget: float, walkers: float) -> dict:
        return classify_real(
            self.expected_counts(budget, walkers), self.window_edges, walkers
        )

    def last_window_valley_time(self) -> float | None:
        lo, hi = base.WINDOW
        mask = (self.t >= lo) & (self.t <= hi)
        idx = np.flatnonzero(mask)
        g = self.g
        valleys = [
            float(self.t[i])
            for i in idx[1:-1]
            if g[i] < g[i - 1] and g[i] <= g[i + 1]
        ]
        return valleys[-1] if valleys else None

    def window_maxima_count(self) -> int:
        lo, hi = base.WINDOW
        idx = np.flatnonzero((self.t >= lo) & (self.t <= hi))
        g = self.g
        return int(
            sum(1 for i in idx[1:-1] if g[i] > g[i - 1] and g[i] >= g[i + 1])
        )


_CLOCKS: dict[tuple, FreeClock] = {}


def clock_for(m: int, eps: float, weights: tuple[float, ...], dt: float = DT_MF) -> FreeClock:
    key = (m, round(eps, 12), tuple(round(w, 12) for w in weights), dt)
    if key not in _CLOCKS:
        _CLOCKS[key] = FreeClock(m, eps, weights, dt)
    return _CLOCKS[key]


# ----------------------------------------------------------------------------
# Crossing search.
# ----------------------------------------------------------------------------


def last_basin_pass(result: dict, edge: float) -> tuple[bool, dict | None]:
    late = [
        r
        for r in result["rows"]
        if r["significant_covariance_aware"] and r["time"] > edge
    ]
    if late:
        return True, late[-1]
    later_rows = [r for r in result["rows"] if r["time"] > edge]
    return False, (later_rows[-1] if later_rows else None)


def _row_summary(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "time": row["time"],
        "relative_prominence": row["relative_prominence"],
        "z_covariance_aware": row["z_covariance_aware"],
        "passes_sigma_gate": row["passes_sigma_gate"],
        "passes_relative_floor": row["passes_relative_floor"],
    }


def find_upper_crossing(clock: FreeClock, edge: float, walkers: float) -> dict:
    lo, hi = BUDGET_INTERVAL
    scan = np.geomspace(lo, hi, SCAN_POINTS)
    table = []
    for b in scan:
        res = clock.classify(float(b), walkers)
        ok, row = last_basin_pass(res, edge)
        table.append(
            {
                "budget": float(b),
                "passes": bool(ok),
                "mode_count": int(res["mode_count_covariance_aware"]),
                "last_basin_candidate": _row_summary(row),
            }
        )
    flags = [t["passes"] for t in table]
    transitions = [
        {
            "from_budget": table[i]["budget"],
            "to_budget": table[i + 1]["budget"],
            "kind": "pass_to_fail" if flags[i] else "fail_to_pass",
        }
        for i in range(len(flags) - 1)
        if flags[i] != flags[i + 1]
    ]
    passing = [t["budget"] for t in table if t["passes"]]
    out = {
        "walkers": walkers,
        "scan": table,
        "transitions": transitions,
        "n_passing_scan_points": len(passing),
        "passing_budget_range_in_scan": (
            [min(passing), max(passing)] if passing else None
        ),
        "monotone_pass_then_fail": len(transitions) <= 1
        and all(t["kind"] == "pass_to_fail" for t in transitions),
    }
    if not passing:
        out.update(
            status="no_certified_crossing_in_scanned_range",
            upper_crossing=None,
            upper_bracket=None,
        )
        return out
    if flags[-1]:
        out.update(
            status="right_censored_at_scan_maximum",
            upper_crossing=None,
            upper_bracket=[max(passing), None],
            lower_bound=hi,
        )
        return out
    last_pass_index = max(i for i, f in enumerate(flags) if f)
    b_lo = table[last_pass_index]["budget"]
    b_hi = table[last_pass_index + 1]["budget"]
    iterations = 0
    while b_hi - b_lo > BISECT_TOL:
        mid = 0.5 * (b_lo + b_hi)
        ok, _ = last_basin_pass(clock.classify(mid, walkers), edge)
        if ok:
            b_lo = mid
        else:
            b_hi = mid
        iterations += 1
    fail_res = clock.classify(b_hi, walkers)
    _, fail_row = last_basin_pass(fail_res, edge)
    out.update(
        status="bisected",
        upper_crossing=0.5 * (b_lo + b_hi),
        upper_bracket=[b_lo, b_hi],
        bisection_iterations=iterations,
        first_failing_candidate=_row_summary(fail_row),
    )
    return out


# ----------------------------------------------------------------------------
# Comparisons with the stored covariance-aware records.
# ----------------------------------------------------------------------------


def w1_final_map(records: list[dict]) -> dict:
    final = {}
    for rec in records:
        if "w1_phase_diagram" not in rec["file"]:
            continue
        cfg = _read(REPORT / rec["file"])["parameters"]["config"]
        key = (int(cfg["m"]), float(cfg["eps"]), float(cfg["budget"]))
        entry = final.get(key)
        if entry is None or bool(cfg["refined"]):
            final[key] = {
                "refined": bool(cfg["refined"]),
                "walkers": int(cfg["walkers"]),
                "weights": tuple(float(w) for w in cfg["weights"]),
                "mode_count_covariance_aware": int(rec["mode_count_new"]),
                "file": rec["file"],
            }
    return final


def compare_w1(records: list[dict]) -> dict:
    final = w1_final_map(records)
    rows = []
    for (m, eps, budget), v in sorted(final.items()):
        clock = clock_for(m, eps, v["weights"])
        res_nominal = clock.classify(budget, HEADLINE_WALKERS)
        res_cell = clock.classify(budget, v["walkers"])
        rows.append(
            {
                "m": m,
                "eps": eps,
                "budget": budget,
                "stored_walkers": v["walkers"],
                "stored_mode_count_covariance_aware": v["mode_count_covariance_aware"],
                "mf_mode_count_at_1e6": int(res_nominal["mode_count_covariance_aware"]),
                "mf_mode_count_at_stored_walkers": int(
                    res_cell["mode_count_covariance_aware"]
                ),
                "mf_counted_times_at_1e6": res_nominal[
                    "significant_times_covariance_aware"
                ],
                "mf_maxima_at_1e6": [_row_summary(r) for r in res_nominal["rows"]],
                "agrees_at_1e6": bool(
                    res_nominal["mode_count_covariance_aware"]
                    == v["mode_count_covariance_aware"]
                ),
                "agrees_at_stored_walkers": bool(
                    res_cell["mode_count_covariance_aware"]
                    == v["mode_count_covariance_aware"]
                ),
                "file": v["file"],
            }
        )
    mism = [r for r in rows if not r["agrees_at_1e6"]]
    mism_cell = [r for r in rows if not r["agrees_at_stored_walkers"]]
    return {
        "n_cells": len(rows),
        "n_agree_at_1e6": len(rows) - len(mism),
        "mismatches_at_1e6": [
            {k: r[k] for k in ("m", "eps", "budget", "stored_mode_count_covariance_aware", "mf_mode_count_at_1e6", "mf_maxima_at_1e6", "file")}
            for r in mism
        ],
        "n_agree_at_stored_walkers": len(rows) - len(mism_cell),
        "mismatches_at_stored_walkers": [
            {k: r[k] for k in ("m", "eps", "budget", "stored_mode_count_covariance_aware", "mf_mode_count_at_stored_walkers", "file")}
            for r in mism_cell
        ],
        "cells": rows,
    }


def compare_production(records: list[dict]) -> dict:
    rows = []
    for path in sorted(PRODUCTION_DIR.glob("*.json")):
        if "dt5e-4" in path.name:
            continue
        payload = _read(path)
        cfg = payload["parameters"]["config"]
        rec = next(
            r
            for r in records
            if r["file"].endswith(f"exact_m_offlattice_production/{path.name}")
            and r["classifier_path"] == "/results/classifier"
        )
        stored_times = [r["time"] for r in rec["maxima"] if r["verdict_new"]]
        m, eps, budget = int(cfg["m"]), float(cfg["eps"]), float(cfg["budget"])
        weights = tuple(float(w) for w in cfg["weights"])
        walkers = int(cfg["walkers"])
        res = clock_for(m, eps, weights).classify(budget, walkers)
        mf_times = res["significant_times_covariance_aware"]
        same_count = len(mf_times) == len(stored_times)
        deltas = (
            [float(b - a) for a, b in zip(stored_times, mf_times)] if same_count else None
        )
        rows.append(
            {
                "file": path.name,
                "m": m,
                "eps": eps,
                "budget": budget,
                "weights": list(weights),
                "walkers": walkers,
                "stored_mode_count_covariance_aware": int(rec["mode_count_new"]),
                "stored_counted_times": stored_times,
                "mf_mode_count": int(res["mode_count_covariance_aware"]),
                "mf_counted_times": mf_times,
                "mode_counts_agree": bool(same_count),
                "peak_time_deltas_mf_minus_stored": deltas,
                "max_abs_peak_time_delta": (
                    max(abs(d) for d in deltas) if deltas else None
                ),
                "mf_maxima": [_row_summary(r) for r in res["rows"]],
            }
        )
    max_delta = max(
        (r["max_abs_peak_time_delta"] for r in rows if r["max_abs_peak_time_delta"] is not None),
        default=None,
    )
    return {
        "n_configurations": len(rows),
        "n_mode_counts_agree": sum(r["mode_counts_agree"] for r in rows),
        "max_abs_peak_time_delta": max_delta,
        "max_abs_peak_time_delta_in_bins": (
            max_delta / base.WINDOW_BIN if max_delta is not None else None
        ),
        "rows": rows,
    }


def contact_factor_table() -> dict:
    tau = base.WINDOW[0]
    eps_values = sorted(set(W1_EPS_GRID) | set(W2_EPS_GRID))
    rows = []
    for eps in eps_values:
        c = float(base.contact_probability(np.array([tau]), eps)[0])
        rows.append(
            {
                "eps": eps,
                "on_w1_grid": eps in W1_EPS_GRID,
                "on_w2_grid": eps in W2_EPS_GRID,
                "contact_factor_at_tau": c,
            }
        )
    return {"tau": tau, "rows": rows}


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="skip grid and N checks")
    args = parser.parse_args()
    started = time.perf_counter()

    regression = regression_test_against_classify_both()
    print(
        f"[regression] classify_real == classify_both on "
        f"{regression['classifier_records_checked']} records / "
        f"{regression['local_maxima_checked']} maxima "
        f"(max |diff| {regression['max_abs_difference']:.3e})"
    )

    reclass = _read(RECLASS_JSON)
    records = reclass["records"]
    b0_summary = _read(W2_DIR / "B0_empirical.json")
    stored_chains = {
        (int(c["m"]), float(c["eps"])): c for c in reclass["w2_chains"]
    }
    stored_edges = {
        (int(c["m"]), float(c["eps"])): float(c["basin_edge_last_g_valley_time"])
        for c in b0_summary["chains"]
    }
    weights_for = {
        int(k): tuple(float(w) for w in v) for k, v in b0_summary["m_weights"].items()
    }

    walker_list = NOMINAL_WALKERS if not args.quick else (HEADLINE_WALKERS,)
    chains = []
    for m in (2, 3):
        for eps in W2_EPS_GRID:
            clock = clock_for(m, eps, weights_for[m])
            edge = stored_edges[(m, eps)]
            own_edge = clock.last_window_valley_time()
            per_n = {}
            for n in walker_list:
                per_n[str(n)] = find_upper_crossing(clock, edge, float(n))
            head = per_n[str(HEADLINE_WALKERS)]
            stored = stored_chains[(m, eps)]
            stored_mid = stored.get("new_b0")
            deviation = (
                (head["upper_crossing"] - stored_mid) / stored_mid
                if head["upper_crossing"] is not None and stored_mid
                else None
            )
            crossings = [
                per_n[str(n)]["upper_crossing"] for n in walker_list
            ]
            chains.append(
                {
                    "m": m,
                    "eps": eps,
                    "weights": list(weights_for[m]),
                    "basin_edge_last_g_valley_time_stored": edge,
                    "basin_edge_last_g_valley_time_fine_grid": own_edge,
                    "g_window_maxima_count": clock.window_maxima_count(),
                    "headline_walkers": HEADLINE_WALKERS,
                    "headline_status": head["status"],
                    "headline_upper_crossing": head["upper_crossing"],
                    "headline_upper_bracket": head["upper_bracket"],
                    "stored_covariance_aware_status": stored.get("new_status"),
                    "stored_covariance_aware_b0": stored_mid,
                    "stored_covariance_aware_bracket": stored.get("new_bracket"),
                    "relative_deviation_mf_vs_stored_midpoint": deviation,
                    "nominal_walker_sensitivity": {
                        "walkers": list(walker_list),
                        "upper_crossings": crossings,
                        "max_abs_spread": (
                            max(crossings) - min(crossings)
                            if all(c is not None for c in crossings)
                            else None
                        ),
                    },
                    "by_walkers": per_n,
                }
            )
            print(
                f"[chain] m={m} eps={eps}: {head['status']} "
                f"upper={head['upper_crossing']} stored={stored_mid}"
            )

    grid_checks = []
    if not args.quick:
        for dt in DT_CHECKS:
            for ch in chains:
                if ch["headline_status"] != "bisected":
                    continue
                m, eps = ch["m"], ch["eps"]
                clock = clock_for(m, eps, weights_for[m], dt)
                res = find_upper_crossing(
                    clock, stored_edges[(m, eps)], float(HEADLINE_WALKERS)
                )
                grid_checks.append(
                    {
                        "m": m,
                        "eps": eps,
                        "dt_mf": dt,
                        "status": res["status"],
                        "upper_crossing": res["upper_crossing"],
                        "difference_from_canonical": (
                            res["upper_crossing"] - ch["headline_upper_crossing"]
                            if res["upper_crossing"] is not None
                            else None
                        ),
                    }
                )
                print(
                    f"[grid dt={dt}] m={m} eps={eps}: {res['upper_crossing']} "
                    f"(canonical {ch['headline_upper_crossing']})"
                )

    w1 = compare_w1(records)
    print(f"[w1] {w1['n_agree_at_1e6']}/{w1['n_cells']} cells agree at N=1e6")
    prod = compare_production(records)
    print(
        f"[production] {prod['n_mode_counts_agree']}/{prod['n_configurations']} "
        f"mode counts agree; max |dt_peak| = {prod['max_abs_peak_time_delta']}"
    )
    contact = contact_factor_table()

    def rounded_headline(ch):
        x = ch["headline_upper_crossing"]
        if ch["headline_status"] == "right_censored_at_scan_maximum":
            return f">{BUDGET_INTERVAL[1]:g}"
        if x is None:
            return "none"
        return f"{x:.2f}"

    payload = {
        "schema_version": 1,
        "analysis": (
            "mean-field hazard-survival law f_1 = B G exp(-B Lambda) applied to "
            "the fixed covariance-aware classifier and the W2 last-basin pass rule"
        ),
        "conventions": {
            "time_grid": {
                "t0": 0.0,
                "tmax": T_MAX,
                "dt_mf_canonical": DT_MF,
                "dt_mf_grid_checks": list(DT_CHECKS),
                "steps_per_window_bin_canonical": int(round(base.WINDOW_BIN / DT_MF)),
            },
            "free_clock": (
                "validate_exact_m_offlattice.free_exposure_clock: G(t) = c(t) "
                "H_{sigma,w}(x(t)) / (W^(d-1) sqrt(2 pi) eps sqrt(D0/2gamma + rho^2)); "
                "matched-law weights; d = 2"
            ),
            "contact_quadrature": {
                "transverse_points": base.THEORY_Y_POINTS,
                "wrap_images": base.THEORY_WRAP_IMAGES,
                "rule": "trapezoid over the folded wrapped-normal minimum-image density "
                "times the longitudinal Gaussian chord probability",
            },
            "exposure_integration": "composite trapezoid rule for Lambda(t) on the fine grid from t = 0",
            "expected_bin_counts": (
                "N [exp(-B Lambda(e_j)) - exp(-B Lambda(e_{j+1}))] on the stored "
                "window bins; this is the exact integral of f_1 over each bin given Lambda"
            ),
            "window": list(base.WINDOW),
            "window_bin_width": base.WINDOW_BIN,
            "window_bin_count": int(round((base.WINDOW[1] - base.WINDOW[0]) / base.WINDOW_BIN)),
            "classifier": {
                "definition": "reclassify_covariance_aware.classify_both algorithm on real-valued expected counts (classify_real)",
                "bandwidth": BANDWIDTH,
                "kernel": "Gaussian, truncated at 4 bandwidths, edge-normalised (mode='same' with ones-normalisation)",
                "prominence_sigma_factor": SIGMA_FACTOR,
                "prominence_relative_floor": RELATIVE_FLOOR,
                "sigma_convention": "covariance-aware Poisson plug-in: sigma_prom^2 = sum_j (A_pj - A_bj)^2 C_j / (N delta)^2 with C_j the expected counts",
                "count_convention": "floating-point expected counts; no int64 cast; the sigma uses the expected counts themselves",
                "regression_test": regression,
            },
            "pass_rule": (
                "covariance-aware significant maximum at a time later than the "
                "stored last valley of G in the window (B0_empirical.json "
                "basin_edge_last_g_valley_time)"
            ),
            "crossing_search": {
                "budget_interval": list(BUDGET_INTERVAL),
                "scan": f"{SCAN_POINTS}-point geometric grid",
                "bisection_target": "upper retention boundary (last pass-to-fail transition of the scan)",
                "bisection_abs_tolerance": BISECT_TOL,
                "right_censoring": f"largest scanned budget passing -> reported as > {BUDGET_INTERVAL[1]:g}",
                "no_pass": "no passing scan point -> no certified crossing in the scanned range",
            },
            "nominal_walker_counts": list(NOMINAL_WALKERS),
            "headline_walkers": HEADLINE_WALKERS,
            "walker_count_note": (
                "the walker count enters only through the five-sigma gate of the "
                "classifier; the values are reported as the sensitivity of the upper "
                "mean-field crossing to the nominal N, not as N-insensitivity of every verdict"
            ),
        },
        "headline": {
            "walkers": HEADLINE_WALKERS,
            "upper_crossings_two_decimals": {
                f"m{ch['m']}_eps{ch['eps']:g}": rounded_headline(ch) for ch in chains
            },
            "upper_crossings_unrounded": {
                f"m{ch['m']}_eps{ch['eps']:g}": ch["headline_upper_crossing"] for ch in chains
            },
            "relative_deviation_from_stored_midpoints": {
                f"m{ch['m']}_eps{ch['eps']:g}": ch["relative_deviation_mf_vs_stored_midpoint"]
                for ch in chains
            },
            "max_abs_relative_deviation": max(
                abs(ch["relative_deviation_mf_vs_stored_midpoint"])
                for ch in chains
                if ch["relative_deviation_mf_vs_stored_midpoint"] is not None
            ),
            "w1_cells_agreeing_at_1e6": [w1["n_agree_at_1e6"], w1["n_cells"]],
            "w1_mismatches_at_1e6": [
                (r["m"], r["eps"], r["budget"]) for r in w1["mismatches_at_1e6"]
            ],
            "production_mode_counts_agreeing": [
                prod["n_mode_counts_agree"],
                prod["n_configurations"],
            ],
            "production_max_abs_peak_time_delta": prod["max_abs_peak_time_delta"],
            "contact_factor_at_tau": {
                f"eps{r['eps']:g}": r["contact_factor_at_tau"] for r in contact["rows"]
            },
        },
        "chains": chains,
        "grid_checks": grid_checks,
        "w1_comparison": w1,
        "production_comparison": prod,
        "contact_factors": contact,
        "runtime_seconds": time.perf_counter() - started,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {OUT_JSON.relative_to(REPORT)} in {payload['runtime_seconds']:.1f} s")
    print(json.dumps(payload["headline"], indent=1))


if __name__ == "__main__":
    main()
