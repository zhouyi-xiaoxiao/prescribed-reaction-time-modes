#!/usr/bin/env python3
"""Independent-seed repeats of the W2 operational-crossing bisection chains.

The stored W2 chains (``w2_b0_empirical/B0_empirical.json``) determine the
operational budget crossing B_op(eps; m, w_eq, P) by one deterministic
bisection per (m, eps) cell at seed 20260813.  Their brackets are the
bisection resolution, not sampling intervals.  The mean-field hazard-survival
evaluation (``mean_field_boundary.json``) reproduces the eight chains within
about 5 %, with the largest relative deviation, +5.3 %, at (m, eps) = (2, 0.05)
(mean-field 7.358 against the stored bracket [6.949, 7.025]).

This script re-runs the complete bisection chain of a cell with the three
robustness seeds 20260814, 20260815, 20260816 under the W2 protocol, with one
deviation: it re-probes the inherited endpoints with the repeat seed's own
stream (the stored W2 chain trusted the W1 grid verdicts, which came from the
W1 stream).  The bisection budget grid is therefore identical to W2's:

  * initial bracket = the W1-derived bracket stored in the W2 chain
    (b_lo = largest passing W1 budget, b_hi = first failing W1 budget above it
    or the scan limit B_MAX = 8), here re-verified with the repeat seed's own
    probes (expanded geometrically if a verdict differs);
  * six geometric bisections at 10^6 walkers per probe (relative half-width
    0.54 % for a 2x bracket);
  * pass rule = a classifier-significant maximum later than the last valley
    of the semi-analytic free clock G (basin edge stored in the W2 chain).

Verdicts use the covariance-aware prominence statistic
(``reclassify_covariance_aware.classify_both``), which is the article's formal
classifier; the superseded peak-only verdict is stored alongside for
reference.  Every probe is saved with its raw 0.02-wide window counts so it
can be re-judged under alternative classifier settings without a rerun.

Deliverables (per cell): the per-seed crossing brackets, their spread, the
seed envelope (union of the brackets), and the position of the mean-field
crossing relative to that envelope in units of one bisection bracket.

Labelling rule (2026-09-23 audit, stream S2).  The bisection runs on a fixed
geometric grid, so per-seed brackets either coincide or differ by whole grid
steps: a bracket is the RESOLUTION of one chain, not a confidence interval,
and envelope widths in "bracket widths" are grid artefacts.  The sampling
scale is estimated separately (``sampling_scale_diagnostic``) from the
floor-interpolated crossings of all four chains (three repeat seeds plus the
campaign chain, whose stored W2 probes are re-judged here from their stored
window counts with the same covariance-aware classifier) and from a pooled
log-linear regression of the last-basin relative prominence on ln B near the
floor.  The quantity is the prominence-floor crossing under protocol P
(formerly B_op); none of this speaks to a theorem threshold.

Outputs:
    artifacts/data/exact_m_prr_upgrade/robustness/w2_seed_repeat_m2_eps0.05/
        probe_m2_eps0.05_seed<seed>_B<budget>.json   (every probe)
        chains_m2_eps0.05.json                        (per-cell summary)
    artifacts/data/exact_m_prr_upgrade/robustness/w2_seed_repeat_m3_eps0.1/ (same layout)
    artifacts/data/exact_m_prr_upgrade/robustness/w2_seed_repeat_summary.json
    artifacts/figures/exact_m_w2_seed_repeat_prr.{png,pdf}

Usage:
    python code/exact_m_prr_w2_seed_repeat.py --pilot            # 1e5-walker timing probe
    python code/exact_m_prr_w2_seed_repeat.py --workers 6        # production, both cells
    python code/exact_m_prr_w2_seed_repeat.py --cells m2_eps0.05 --workers 3
    python code/exact_m_prr_w2_seed_repeat.py --summary-only     # re-summarize stored probes

Completed probes are reused on restart unless ``--force`` is given.
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

import validate_exact_m_offlattice as base  # noqa: E402
import exact_m_prr_upgrade_core as core  # noqa: E402
import exact_m_prr_robustness as rob  # noqa: E402
import reclassify_covariance_aware as recl  # noqa: E402

W2_SUMMARY = rob.W2_DIR / "B0_empirical.json"
MEAN_FIELD = core.UPGRADE_DATA / "mean_field_boundary.json"
SUMMARY_OUT = rob.ROBUST_DIR / "w2_seed_repeat_summary.json"
FIGURE_STEM = core.FIGURES / "exact_m_w2_seed_repeat_prr"

SEEDS = rob.SEEDS  # (20260814, 20260815, 20260816)
CAMPAIGN_SEED = core.CAMPAIGN_SEED  # 20260813 (stored W2 chain)
PROBE_WALKERS = 1_000_000
CHUNK = rob.CHUNK
DT = rob.DT
TMAX = rob.TMAX
BANDWIDTH = rob.BANDWIDTH
B_MAX = 8.0
B_MIN = 0.125
BISECT_ITERS = 6
TAG_W2_SEED_REPEAT = 66  # disjoint from campaign tags 11--53 and robustness 61--65

CELLS = {
    "m2_eps0.05": {"m": 2, "eps": 0.05, "weights": (0.5, 0.5)},
    "m3_eps0.1": {"m": 3, "eps": 0.10, "weights": tuple(rob.M3_WEIGHTS)},
}


def cell_dir(name: str) -> Path:
    return rob.ROBUST_DIR / f"w2_seed_repeat_{name}"


def probe_path(name: str, seed: int, budget: float, pilot: bool = False) -> Path:
    stem = f"probe_{name}_seed{seed}_B{budget:.6g}.json"
    return cell_dir(name) / ("pilot" if pilot else "") / stem


# ----------------------------------------------------------------------------
# Stored references.
# ----------------------------------------------------------------------------


def stored_chain(m: int, eps: float) -> dict:
    payload = rob._read(W2_SUMMARY)
    for chain in payload["chains"]:
        if chain["m"] == m and abs(chain["eps"] - eps) < 1e-12:
            return chain
    raise KeyError(f"no stored W2 chain for m={m}, eps={eps}")


def mean_field_crossing(name: str) -> dict:
    payload = rob._read(MEAN_FIELD)
    head = payload["headline"]
    return {
        "upper_crossing_unrounded": head["upper_crossings_unrounded"][name],
        "upper_crossing_two_decimals": head["upper_crossings_two_decimals"][name],
        "relative_deviation_from_stored_midpoint": head[
            "relative_deviation_from_stored_midpoints"
        ][name],
        "source": f"{MEAN_FIELD.name}#headline/upper_crossings_unrounded/{name}",
    }


def initial_bracket(chain: dict) -> tuple[float, float]:
    """W2 protocol bracket: W1 verdicts, hi capped at B_MAX (chain-stored)."""

    bracket = chain["w1_bracket"]
    b_lo = bracket["b_lo"]
    b_hi = bracket["b_hi"] if bracket["b_hi"] is not None else B_MAX
    if b_lo is None:
        raise RuntimeError("stored chain has no passing W1 budget; not supported here")
    return float(b_lo), float(b_hi)


# ----------------------------------------------------------------------------
# One probe.
# ----------------------------------------------------------------------------


def classify_record(counts, edges, walkers: int, *, basin_edge: float) -> dict:
    result = recl.classify_both(counts, edges, walkers, bandwidth=BANDWIDTH)
    rows = [dict(row) for row in result["rows"]]
    for row in rows:
        row["passes_sigma_gate_covariance_aware"] = bool(
            row["z_covariance_aware"] >= recl.SIGMA_FACTOR
        )
        row["passes_relative_floor"] = bool(
            row["relative_prominence"] >= recl.RELATIVE_FLOOR
        )
    late = [r for r in rows if r["time"] > basin_edge]
    late_cov = [r for r in late if r["significant_covariance_aware"]]
    late_peak = [r for r in late if r["significant_peak_only"]]
    candidate = (
        max(late, key=lambda r: r["z_covariance_aware"]) if late else None
    )
    return {
        "bandwidth": BANDWIDTH,
        "prominence_sigma_factor": recl.SIGMA_FACTOR,
        "prominence_relative_floor": recl.RELATIVE_FLOOR,
        "global_max_smoothed_height": result["global_max"],
        "local_maxima_both_conventions": rows,
        "mode_count_peak_only": int(result["mode_count_peak_only"]),
        "mode_count_covariance_aware": int(result["mode_count_covariance_aware"]),
        "counted_times_covariance_aware": result["significant_times_covariance_aware"],
        "basin_edge_last_g_valley_time": basin_edge,
        "last_basin_candidate": candidate,
        "last_mode_present_covariance_aware": bool(late_cov),
        "last_mode_present_peak_only": bool(late_peak),
    }


def run_probe(task: dict) -> dict:
    """Simulate one probe (or reuse a stored one) and return its verdict row."""

    name = task["name"]
    cell = CELLS[name]
    seed, budget, walkers = task["seed"], task["budget"], task["walkers"]
    path = probe_path(name, seed, budget, pilot=task.get("pilot", False))
    if path.exists() and not task.get("force", False):
        existing = rob._read(path)
        cfg = existing["parameters"]["config"]
        if (
            int(cfg["walkers"]) == walkers
            and int(cfg["seed"]) == seed
            and abs(float(cfg["budget"]) - budget) < 1e-12
        ):
            return probe_row(existing, path, reused=True)
        raise RuntimeError(
            f"{path.name} exists with a different configuration; pass --force"
        )

    outcome = base.run_config(
        m=cell["m"],
        eps=cell["eps"],
        budget=budget,
        weights=cell["weights"],
        walkers=walkers,
        chunk=min(CHUNK, walkers),
        dt=DT,
        tmax=TMAX,
        seed=seed,
        tag=TAG_W2_SEED_REPEAT,
        verbose=False,
    )
    kill_times = outcome["kill_times"]
    edges = np.arange(
        base.WINDOW[0], base.WINDOW[1] + 0.5 * base.WINDOW_BIN, base.WINDOW_BIN
    )
    counts, _ = np.histogram(kill_times, bins=edges)
    window_kills = int(
        np.sum((kill_times >= base.WINDOW[0]) & (kill_times <= base.WINDOW[1]))
    )
    classifier = classify_record(
        counts, edges, walkers, basin_edge=task["basin_edge"]
    )
    payload = {
        "schema_version": 1,
        "analysis": (
            "w2_seed_repeat: independent-seed repeat of the W2 operational-"
            "crossing bisection chain (one probe)"
        ),
        "parameters": {
            "model_parameters": core.model_dict(core.MODEL),
            "config": {
                "cell": name,
                "m": cell["m"],
                "eps": cell["eps"],
                "budget": budget,
                "weights": list(cell["weights"]),
                "target_times": list(base.TARGET_TIMES[cell["m"]]),
                "walkers": walkers,
                "chunk": min(CHUNK, walkers),
                "dt": DT,
                "tmax": TMAX,
                "seed": seed,
                "tag": TAG_W2_SEED_REPEAT,
                "classifier_bandwidth": BANDWIDTH,
                "classifier_prominence_sigma_factor": recl.SIGMA_FACTOR,
                "classifier_prominence_relative_floor": recl.RELATIVE_FLOOR,
            },
            "pass_rule": (
                "covariance-aware classifier-significant maximum (z >= 5 and "
                "prominence >= 5 % of max smoothed height) with time > last "
                f"semi-analytic G valley ({task['basin_edge']:.4f})"
            ),
            "rng_note": (
                "deterministic Philox stream; seed, tag, m, eps, budget, weights "
                "and dt enter the SeedSequence entropy, so every probe is an "
                "independent stream, disjoint from the campaign (seed 20260813, "
                "tag 21) and robustness (tags 61--65) streams"
            ),
            "classifier_note": (
                "classified through reclassify_covariance_aware.classify_both on "
                "the stored counts (classify_both schema, outside the 201 "
                "retained-campaign records re-judged by reclassify_covariance_aware.py)"
            ),
        },
        "validation_gates": {
            "gate2_kill_probability_max": float(outcome["kill_probability_max"]),
            "gate3_mass_balance_passed": bool(
                kill_times.size + outcome["survivors"] == walkers
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
            "runtime_seconds": float(outcome["runtime_seconds"]),
        },
    }
    core.write_json(path, payload)
    return probe_row(payload, path, reused=False)


def probe_row(payload: dict, path: Path, *, reused: bool) -> dict:
    cfg = payload["parameters"]["config"]
    cls = payload["results"]["classifier_both_conventions"]
    cand = cls["last_basin_candidate"]
    return {
        "budget": float(cfg["budget"]),
        "seed": int(cfg["seed"]),
        "walkers": int(cfg["walkers"]),
        "verdict": bool(cls["last_mode_present_covariance_aware"]),
        "verdict_peak_only": bool(cls["last_mode_present_peak_only"]),
        "mode_count_covariance_aware": cls["mode_count_covariance_aware"],
        "counted_times_covariance_aware": cls["counted_times_covariance_aware"],
        "last_candidate_time": cand["time"] if cand else None,
        "last_candidate_relative_prominence": (
            cand["relative_prominence"] if cand else None
        ),
        "last_candidate_z_covariance_aware": (
            cand["z_covariance_aware"] if cand else None
        ),
        "last_candidate_passes_sigma_gate": (
            cand["passes_sigma_gate_covariance_aware"] if cand else None
        ),
        "last_candidate_passes_relative_floor": (
            cand["passes_relative_floor"] if cand else None
        ),
        "runtime_seconds": payload["results"]["runtime_seconds"],
        "file": path.name,
        "reused": reused,
    }


# ----------------------------------------------------------------------------
# One chain (sequential bisection for one seed).
# ----------------------------------------------------------------------------


def run_chain(task: dict) -> dict:
    name, seed = task["name"], task["seed"]
    cell = CELLS[name]
    started = time.perf_counter()
    b_lo, b_hi = task["initial_bracket"]
    basin_edge = task["basin_edge"]
    common = {
        "name": name,
        "seed": seed,
        "walkers": task["walkers"],
        "basin_edge": basin_edge,
        "force": task["force"],
    }
    history: list[dict] = []

    def probe(budget: float, role: str) -> dict:
        row = run_probe({**common, "budget": budget})
        row["role"] = role
        history.append(row)
        return row

    chain = {
        "cell": name,
        "m": cell["m"],
        "eps": cell["eps"],
        "weights": list(cell["weights"]),
        "seed": seed,
        "probe_walkers": task["walkers"],
        "initial_bracket_from_stored_w2_chain": [b_lo, b_hi],
        "basin_edge_last_g_valley_time": basin_edge,
        "bisect_iterations": BISECT_ITERS,
    }

    # Re-verify the inherited bracket with this seed's own probes.
    lo_row = probe(b_lo, "bracket_lo_verification")
    expansions = 0
    while not lo_row["verdict"]:
        expansions += 1
        if b_lo <= B_MIN:
            chain["status"] = "no_passing_budget_at_probe_walkers"
            chain["probes"] = history
            chain["wall_seconds"] = time.perf_counter() - started
            return chain
        b_hi = b_lo
        b_lo = max(B_MIN, b_lo / 2.0)
        lo_row = probe(b_lo, "bracket_lo_expansion")
    hi_row = probe(b_hi, "bracket_hi_verification")
    while hi_row["verdict"]:
        expansions += 1
        if b_hi >= B_MAX:
            chain["status"] = "right_censored"
            chain["b0_lower_bound"] = B_MAX
            chain["probes"] = history
            chain["wall_seconds"] = time.perf_counter() - started
            return chain
        b_lo = b_hi
        b_hi = min(B_MAX, b_hi * 2.0)
        hi_row = probe(b_hi, "bracket_hi_expansion")
    chain["bracket_expansions"] = expansions
    chain["verified_initial_bracket"] = [b_lo, b_hi]

    for _ in range(BISECT_ITERS):
        mid = math.sqrt(b_lo * b_hi)
        row = probe(mid, "bisection")
        if row["verdict"]:
            b_lo = mid
        else:
            b_hi = mid
    chain["status"] = "bisected"
    chain["probes"] = history
    chain["b0_bracket"] = [b_lo, b_hi]
    chain["b0"] = math.sqrt(b_lo * b_hi)
    chain["b0_bracket_width"] = b_hi - b_lo
    chain["b0_relative_halfwidth"] = math.sqrt(b_hi / b_lo) - 1.0
    chain["monotonicity_violations"] = monotonicity_violations(history)
    chain["floor_interpolated_crossing"] = floor_interpolated_crossing(history)
    chain["wall_seconds"] = time.perf_counter() - started
    return chain


def monotonicity_violations(history: list[dict]) -> int:
    """Count passes at a budget above some failing budget (should be 0)."""

    passing = [r["budget"] for r in history if r["verdict"]]
    failing = [r["budget"] for r in history if not r["verdict"]]
    return sum(1 for p in passing for f in failing if p > f)


def floor_interpolated_crossing(history: list[dict]) -> dict | None:
    """Diagnostic only: log-B linear interpolation of the last-basin
    relative prominence through the 5 % floor between the two probes that
    define the final bracket.  Not the protocol quantity."""

    rows = [
        r for r in history if r["last_candidate_relative_prominence"] is not None
    ]
    if len(rows) < 2:
        return None
    floor = recl.RELATIVE_FLOOR
    lo = [r for r in rows if r["verdict"]]
    hi = [r for r in rows if not r["verdict"]]
    if not lo or not hi:
        return None
    a = max(lo, key=lambda r: r["budget"])
    b = min(hi, key=lambda r: r["budget"])
    ra, rb = a["last_candidate_relative_prominence"], b["last_candidate_relative_prominence"]
    if ra == rb:
        return None
    frac = (ra - floor) / (ra - rb)
    log_b = math.log(a["budget"]) + frac * (math.log(b["budget"]) - math.log(a["budget"]))
    return {
        "budget": float(math.exp(log_b)),
        "between": [a["budget"], b["budget"]],
        "relative_prominence_at_bracket": [ra, rb],
        "note": "diagnostic interpolation of the 5 % relative-prominence floor; not the protocol crossing",
    }


# ----------------------------------------------------------------------------
# Sampling-scale diagnostics (brackets are resolution, not CIs).
# ----------------------------------------------------------------------------

POOLED_WINDOWS = {"m2_eps0.05": (6.5, 7.5), "m3_eps0.1": (3.3, 3.7)}


def campaign_chain_rows(stored: dict) -> list[dict]:
    """Re-judge every stored W2 probe of this cell (seed 20260813, tag 21)
    from its stored 0.02-wide window counts with the covariance-aware
    classifier and the same basin edge; returns probe-row-like dicts."""

    rows = []
    edge = stored["basin_edge_last_g_valley_time"]
    for p in stored["probes"]:
        payload = rob._read(rob.W2_DIR / p["file"])
        cfg = payload["parameters"]["config"]
        cls0 = payload["results"]["classifier"]
        cls = classify_record(
            np.asarray(cls0["counts"], dtype=float),
            np.asarray(cls0["edges"], dtype=float),
            int(cfg["walkers"]),
            basin_edge=edge,
        )
        cand = cls["last_basin_candidate"]
        rows.append(
            {
                "budget": float(cfg["budget"]),
                "seed": int(cfg["seed"]),
                "tag": int(cfg["tag"]),
                "walkers": int(cfg["walkers"]),
                "verdict": bool(cls["last_mode_present_covariance_aware"]),
                "stored_verdict_peak_only_run_time": bool(p["verdict"]),
                "last_candidate_time": cand["time"] if cand else None,
                "last_candidate_relative_prominence": (
                    cand["relative_prominence"] if cand else None
                ),
                "last_candidate_z_covariance_aware": (
                    cand["z_covariance_aware"] if cand else None
                ),
                "file": p["file"],
            }
        )
    return rows


def pooled_floor_regression(rows: list[dict], window: tuple[float, float]) -> dict | None:
    """OLS of last-basin relative prominence on ln B over all probes (all
    chains) with B in ``window``; crossing of the 5 % floor with a
    delta-method 1-sigma error (residual-variance based)."""

    pts = [
        (math.log(r["budget"]), r["last_candidate_relative_prominence"])
        for r in rows
        if r["last_candidate_relative_prominence"] is not None
        and window[0] <= r["budget"] <= window[1]
    ]
    if len(pts) < 4:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(x) - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    a, b = float(beta[0]), float(beta[1])
    floor = recl.RELATIVE_FLOOR
    lx = (floor - a) / b
    grad = np.array([-1.0 / b, -(floor - a) / b**2])
    se_lx = float(math.sqrt(grad @ cov @ grad))
    bstar = math.exp(lx)
    return {
        "window_B": list(window),
        "n_probes": int(len(x)),
        "intercept": a,
        "slope_d_relprom_d_lnB": b,
        "residual_sd": math.sqrt(s2),
        "crossing": bstar,
        "crossing_se_1sigma": bstar * se_lx,
        "note": (
            "pooled over all four chains; delta-method 1-sigma from the OLS "
            "covariance (residual variance), a sampling-scale diagnostic, not "
            "the protocol quantity"
        ),
    }


def sampling_scale_diagnostic(name: str, chains: list[dict], stored: dict, mf_val: float) -> dict:
    campaign_rows = campaign_chain_rows(stored)
    campaign_chain = {
        "seed": CAMPAIGN_SEED,
        "probes": campaign_rows,
        "floor_interpolated_crossing": floor_interpolated_crossing(campaign_rows),
        "monotonicity_violations": monotonicity_violations(campaign_rows),
    }
    lo = max((r["budget"] for r in campaign_rows if r["verdict"]), default=None)
    hi = min((r["budget"] for r in campaign_rows if not r["verdict"]), default=None)
    campaign_chain["covariance_aware_bracket_from_stored_probes"] = [lo, hi]
    campaign_chain["covariance_aware_bracket_equals_stored_b0_bracket"] = bool(
        stored.get("b0_bracket") is not None
        and lo is not None and hi is not None
        and abs(lo - stored["b0_bracket"][0]) < 1e-9
        and abs(hi - stored["b0_bracket"][1]) < 1e-9
    )
    interp = {}
    for c in chains:
        fic = c.get("floor_interpolated_crossing")
        if fic:
            interp[str(c["seed"])] = fic["budget"]
    if campaign_chain["floor_interpolated_crossing"]:
        interp[str(CAMPAIGN_SEED)] = campaign_chain["floor_interpolated_crossing"]["budget"]
    vals = np.array(list(interp.values()), dtype=float)
    all_rows = [r for c in chains for r in c.get("probes", [])] + campaign_rows
    pooled = pooled_floor_regression(all_rows, POOLED_WINDOWS[name])
    out = {
        "campaign_chain_rejudged": campaign_chain,
        "floor_interpolated_crossings": interp,
        "n_chains": int(vals.size),
    }
    if vals.size >= 2:
        mean, sd = float(vals.mean()), float(vals.std(ddof=1))
        out.update(
            {
                "floor_interpolated_mean": mean,
                "floor_interpolated_sd": sd,
                "mean_field_minus_floor_interpolated_mean": mf_val - mean,
                "mean_field_gap_in_floor_interpolated_sd": (mf_val - mean) / sd
                if sd > 0 else None,
            }
        )
    out["pooled_regression"] = pooled
    if pooled:
        out["mean_field_gap_in_pooled_se"] = (mf_val - pooled["crossing"]) / pooled[
            "crossing_se_1sigma"
        ]
    fixed_b = {}
    by_budget: dict[float, list[float]] = {}
    for r in all_rows:
        if r["last_candidate_relative_prominence"] is None:
            continue
        by_budget.setdefault(round(r["budget"], 9), []).append(
            r["last_candidate_relative_prominence"]
        )
    for b, v in sorted(by_budget.items()):
        if len(v) >= 3:
            fixed_b[f"{b:.6g}"] = {
                "n": len(v),
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)),
            }
    out["relative_prominence_across_chains_at_fixed_B"] = fixed_b
    out["note"] = (
        "Sampling scale of the prominence-floor crossing: floor-interpolated "
        "crossings (log-B linear between the two final-bracket probes of each "
        "chain) and a pooled regression; bisection brackets are resolution, "
        "not confidence intervals."
    )
    return out


# ----------------------------------------------------------------------------
# Summaries.
# ----------------------------------------------------------------------------


def summarize_cell(name: str, chains: list[dict], stored: dict) -> dict:
    chains = sorted(chains, key=lambda c: c["seed"])
    mf = mean_field_crossing(name)
    mf_val = mf["upper_crossing_unrounded"]
    bis = [c for c in chains if c.get("status") == "bisected"]
    out = {
        "cell": name,
        "m": CELLS[name]["m"],
        "eps": CELLS[name]["eps"],
        "weights": list(CELLS[name]["weights"]),
        "seeds": [c["seed"] for c in chains],
        "probe_walkers": chains[0]["probe_walkers"] if chains else None,
        "chains": chains,
        "n_bisected": len(bis),
        "statuses": {str(c["seed"]): c.get("status") for c in chains},
        "stored_w2_chain": {
            "seed": CAMPAIGN_SEED,
            "status": stored.get("status"),
            "b0": stored.get("b0"),
            "b0_bracket": stored.get("b0_bracket"),
            "source": f"{W2_SUMMARY.name}#chains[m={stored['m']},eps={stored['eps']}]",
        },
        "mean_field": mf,
    }
    if not bis:
        return out
    brackets = [c["b0_bracket"] for c in bis]
    mids = [c["b0"] for c in bis]
    widths = [c["b0_bracket_width"] for c in bis]
    env_lo, env_hi = min(b[0] for b in brackets), max(b[1] for b in brackets)
    mean_width = float(np.mean(widths))
    mid_spread = max(mids) - min(mids)
    per_seed = {
        str(c["seed"]): {
            "bracket": c["b0_bracket"],
            "midpoint": c["b0"],
            "width": c["b0_bracket_width"],
            "relative_halfwidth": c["b0_relative_halfwidth"],
            "n_probes": len(c["probes"]),
            "monotonicity_violations": c["monotonicity_violations"],
            "floor_interpolated_crossing": (
                c["floor_interpolated_crossing"]["budget"]
                if c.get("floor_interpolated_crossing")
                else None
            ),
        }
        for c in bis
    }
    # Pairwise bracket overlap between seeds.
    overlaps = []
    for i in range(len(bis)):
        for j in range(i + 1, len(bis)):
            a, b = bis[i]["b0_bracket"], bis[j]["b0_bracket"]
            overlaps.append(
                {
                    "seeds": [bis[i]["seed"], bis[j]["seed"]],
                    "overlap": bool(a[0] < b[1] and b[0] < a[1]),
                }
            )
    stored_bracket = stored.get("b0_bracket")
    all_brackets = brackets + ([stored_bracket] if stored_bracket else [])
    all_mids = mids + ([stored["b0"]] if stored.get("b0") is not None else [])
    env4_lo, env4_hi = min(b[0] for b in all_brackets), max(b[1] for b in all_brackets)
    gap_env = mf_val - env_hi
    out.update(
        {
            "per_seed": per_seed,
            "pairwise_bracket_overlap": overlaps,
            "three_seed": {
                "brackets": brackets,
                "midpoints": mids,
                "midpoint_min": min(mids),
                "midpoint_max": max(mids),
                "midpoint_mean": float(np.mean(mids)),
                "midpoint_spread": mid_spread,
                "midpoint_spread_relative": mid_spread / float(np.mean(mids)),
                "midpoint_spread_in_bracket_widths": mid_spread / mean_width,
                "mean_bracket_width": mean_width,
                "envelope": [env_lo, env_hi],
                "envelope_width": env_hi - env_lo,
                "envelope_width_in_bracket_widths": (env_hi - env_lo) / mean_width,
                "all_brackets_mutually_overlap": all(o["overlap"] for o in overlaps),
            },
            "four_chain_including_campaign_seed": {
                "seeds": [c["seed"] for c in bis] + [CAMPAIGN_SEED],
                "midpoints": all_mids,
                "envelope": [env4_lo, env4_hi],
                "envelope_width": env4_hi - env4_lo,
                "midpoint_spread": max(all_mids) - min(all_mids),
                "stored_bracket_overlaps_three_seed_envelope": bool(
                    stored_bracket is not None
                    and stored_bracket[0] < env_hi
                    and env_lo < stored_bracket[1]
                ),
            },
            "mean_field_comparison": {
                "mean_field_crossing": mf_val,
                "three_seed_envelope": [env_lo, env_hi],
                "mean_field_inside_three_seed_envelope": bool(env_lo <= mf_val <= env_hi),
                "gap_above_envelope": gap_env,
                "gap_above_envelope_in_bracket_widths": gap_env / mean_width,
                "gap_above_envelope_in_midpoint_spreads": (
                    gap_env / mid_spread if mid_spread > 0 else None
                ),
                "gap_above_four_chain_envelope": mf_val - env4_hi,
                "gap_above_four_chain_envelope_in_bracket_widths": (mf_val - env4_hi)
                / mean_width,
                "relative_deviation_from_three_seed_mean_midpoint": mf_val
                / float(np.mean(mids))
                - 1.0,
                "relative_deviation_range_over_seeds": [
                    mf_val / max(mids) - 1.0,
                    mf_val / min(mids) - 1.0,
                ],
                "relative_deviation_from_stored_midpoint": mf[
                    "relative_deviation_from_stored_midpoint"
                ],
            },
            "bracket_labelling": (
                "bisection bracket (resolution, not CI): the geometric grid is "
                "shared by all chains, so brackets coincide or differ by whole "
                "grid steps; spreads in bracket widths are grid-quantised"
            ),
            "sampling_scale_diagnostic": sampling_scale_diagnostic(
                name, bis, stored, mf_val
            ),
        }
    )
    return out


def make_figure(cells: list[dict]) -> list[str]:
    """Per-chain prominence-floor crossing brackets (resolution, not CI).

    Fixes of the 2026-09-23 S2 audit: one shared x-label, legend at lower
    left, y headroom so the mean-field line is always visible, the error bars
    labelled as bisection brackets (resolution, not CI), and the
    floor-interpolated crossing of every chain drawn as an open marker (the
    sampling-scale diagnostic)."""

    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(cells)
    fig, axes = plt.subplots(
        1, n, figsize=(3.40, 2.55), constrained_layout=True, squeeze=False
    )
    for k, (ax, cell) in enumerate(zip(axes[0], cells)):
        chains = [c for c in cell["chains"] if c.get("status") == "bisected"]
        stored = cell["stored_w2_chain"]
        interp = cell.get("sampling_scale_diagnostic", {}).get(
            "floor_interpolated_crossings", {}
        )
        labels, mids, lo, hi, colours, fi = [], [], [], [], [], []
        if stored["b0_bracket"]:
            labels.append("13")
            mids.append(stored["b0"])
            lo.append(stored["b0"] - stored["b0_bracket"][0])
            hi.append(stored["b0_bracket"][1] - stored["b0"])
            colours.append(core.OI_GREY)
            fi.append(interp.get(str(CAMPAIGN_SEED)))
        for c in chains:
            labels.append(str(c["seed"])[-2:])
            mids.append(c["b0"])
            lo.append(c["b0"] - c["b0_bracket"][0])
            hi.append(c["b0_bracket"][1] - c["b0"])
            colours.append(core.OI_BLUE if cell["m"] == 2 else core.OI_VERMILLION)
            fi.append(interp.get(str(c["seed"])))
        xs = np.arange(len(labels))
        for x, y, l, h, col, f in zip(xs, mids, lo, hi, colours, fi):
            ax.errorbar(
                [x], [y], yerr=[[l], [h]], marker="_", ms=6, lw=1.2,
                capsize=3.0, color=col,
            )
            if f is not None:
                ax.plot([x + 0.18], [f], marker="o", ms=3.2, mfc="none",
                        mec=col, mew=0.9, ls="none")
        mf = cell["mean_field"]["upper_crossing_unrounded"]
        y_lo = min(m - l for m, l in zip(mids, lo))
        y_hi = max(m + h for m, h in zip(mids, hi))
        if mf is not None:
            ax.axhline(mf, ls="--", lw=1.0, color=core.OI_ORANGE)
            y_hi = max(y_hi, mf)
        span = y_hi - y_lo
        ax.set_ylim(y_lo - 1.05 * span, y_hi + 0.12 * span)
        ax.set_xticks(xs, labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        if k == 0:
            ax.set_ylabel(r"prominence-floor crossing $B$")
        ax.set_title(
            rf"$(m,\varepsilon)=({cell['m']},{cell['eps']:g})$", fontsize=8
        )
        ax.grid(True, alpha=0.3)
    fig.supxlabel("chain seed 202608NN (13 = campaign W2 chain)", fontsize=7)
    handles = [
        Line2D([], [], ls="--", color=core.OI_ORANGE, lw=1.0),
        Line2D([], [], marker="_", ms=6, ls="-", lw=1.2, color="0.3"),
        Line2D([], [], marker="o", ms=3.2, mfc="none", mec="0.3", ls="none"),
    ]
    axes[0][0].legend(
        handles,
        ["mean-field crossing", "bisection bracket\n(resolution, not CI)",
         "floor-interpolated\ncrossing"],
        loc="lower left",
        fontsize=5.0,
        framealpha=0.9,
        handlelength=1.6,
        borderaxespad=0.3,
    )
    written = core.save_figure(fig, FIGURE_STEM)
    plt.close(fig)
    return written


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------


def pilot(names: list[str], walkers: int) -> None:
    """One 1e5-walker probe per cell at the stored midpoint: timing only."""

    for name in names:
        cell = CELLS[name]
        stored = stored_chain(cell["m"], cell["eps"])
        task = {
            "name": name,
            "seed": SEEDS[0],
            "budget": float(stored["b0"]),
            "walkers": walkers,
            "basin_edge": stored["basin_edge_last_g_valley_time"],
            "force": True,
            "pilot": True,
        }
        row = run_probe(task)
        b_lo, b_hi = initial_bracket(stored)
        n_probes = 2 + BISECT_ITERS
        per_probe_1e6 = row["runtime_seconds"] * (PROBE_WALKERS / walkers)
        print(
            f"[pilot {name}] walkers={walkers} B={task['budget']:.4f} "
            f"runtime={row['runtime_seconds']:.1f} s verdict={row['verdict']} "
            f"-> projected {per_probe_1e6:.0f} s per 1e6-walker probe, "
            f"{n_probes} probes/chain -> {n_probes * per_probe_1e6 / 60:.1f} min per "
            f"chain (seeds run in parallel); bracket [{b_lo:g}, {b_hi:g}]",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--cells", nargs="+", default=list(CELLS), choices=list(CELLS)
    )
    parser.add_argument(
        "--walkers", type=lambda v: int(float(v)), default=PROBE_WALKERS
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-walkers", type=lambda v: int(float(v)), default=100_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.walkers <= 0:
        raise SystemExit("worker and walker counts must be positive")

    if args.pilot:
        pilot(args.cells, args.pilot_walkers)
        return

    started = time.perf_counter()
    tasks = []
    stored_by_cell = {}
    for name in args.cells:
        cell = CELLS[name]
        stored = stored_chain(cell["m"], cell["eps"])
        stored_by_cell[name] = stored
        # Consistency check of the stored basin edge against the semi-analytic G.
        diag = core.g_window_maxima_count(
            m=cell["m"], eps=cell["eps"], weights=cell["weights"]
        )
        edge = diag["valleys"][-1]["time"]
        assert abs(edge - stored["basin_edge_last_g_valley_time"]) < 1e-9, (
            name, edge, stored["basin_edge_last_g_valley_time"]
        )
        for seed in SEEDS:
            tasks.append(
                {
                    "name": name,
                    "seed": seed,
                    "walkers": args.walkers,
                    "initial_bracket": initial_bracket(stored),
                    "basin_edge": stored["basin_edge_last_g_valley_time"],
                    "force": args.force and not args.summary_only,
                }
            )

    chains_by_cell: dict[str, list[dict]] = {name: [] for name in args.cells}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_chain, task): task for task in tasks}
        for future in as_completed(futures):
            chain = future.result()
            chains_by_cell[chain["cell"]].append(chain)
            print(
                f"chain {chain['cell']} seed={chain['seed']}: status={chain.get('status')} "
                f"B_op={chain.get('b0')} bracket={chain.get('b0_bracket')} "
                f"probes={len(chain.get('probes', []))} "
                f"wall={chain.get('wall_seconds', 0):.0f} s",
                flush=True,
            )

    cells = []
    for name in args.cells:
        summary = summarize_cell(name, chains_by_cell[name], stored_by_cell[name])
        core.write_json(cell_dir(name) / f"chains_{name}.json", summary)
        cells.append(summary)
        mfc = summary.get("mean_field_comparison")
        if mfc:
            print(
                f"[{name}] per-seed brackets "
                f"{[[round(b[0], 4), round(b[1], 4)] for b in summary['three_seed']['brackets']]} "
                f"envelope {[round(v, 4) for v in summary['three_seed']['envelope']]} "
                f"midpoint spread {summary['three_seed']['midpoint_spread']:.4f} "
                f"({summary['three_seed']['midpoint_spread_in_bracket_widths']:.2f} bracket widths); "
                f"mean-field {mfc['mean_field_crossing']:.4f} lies "
                f"{mfc['gap_above_envelope']:+.4f} above the envelope = "
                f"{mfc['gap_above_envelope_in_bracket_widths']:.1f} bracket widths",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "analysis": (
            "w2_seed_repeat: independent-seed repeats of the W2 operational-"
            "crossing bisection chains (protocol-identical: inherited W1 bracket "
            "re-verified, six geometric bisections, 1e6 walkers per probe, "
            "last-basin pass rule under the covariance-aware classifier)"
        ),
        "seeds": list(SEEDS),
        "campaign_seed_of_stored_chain": CAMPAIGN_SEED,
        "tag": TAG_W2_SEED_REPEAT,
        "probe_walkers": args.walkers,
        "bisect_iterations": BISECT_ITERS,
        "b_max": B_MAX,
        "dt": DT,
        "tmax": TMAX,
        "classifier": {
            "definition": (
                "reclassify_covariance_aware.classify_both; covariance-aware "
                "verdicts are the article's formal classifier"
            ),
            "bandwidth": BANDWIDTH,
            "prominence_sigma_factor": recl.SIGMA_FACTOR,
            "prominence_relative_floor": recl.RELATIVE_FLOOR,
            "pass_rule": (
                "covariance-aware significant maximum later than the last "
                "semi-analytic G valley (basin edge stored in the W2 chain)"
            ),
        },
        "model_parameters": core.model_dict(core.MODEL),
        "cells": cells,
        "interpretation_note": (
            "Per-seed brackets are the bisection resolution of one independent "
            "stream on a shared geometric grid (bisection bracket = resolution, "
            "not CI); the seed envelope (union of brackets) is a protocol "
            "statement and its width in bracket widths is grid-quantised.  The "
            "sampling scale of the prominence-floor crossing under protocol P "
            "at 1e6 walkers is given by cells[*].sampling_scale_diagnostic "
            "(floor-interpolated crossings of four chains and a pooled "
            "regression).  The repeat chains re-probe the inherited endpoints "
            "with their own streams (the only deviation from W2).  Four chains "
            "bound neither the classifier-setting sensitivity (reported "
            "separately) nor a theorem threshold; the mean-field law is a "
            "zero-parameter prediction and is not fitted."
        ),
        "labelling_rules_2026_09_23": [
            "bisection bracket (resolution, not CI)",
            "quantity = prominence-floor crossing under protocol P (formerly B_op)",
            "repeat chains re-probe inherited endpoints",
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    core.write_json(SUMMARY_OUT, summary)
    print(f"summary -> {SUMMARY_OUT}", flush=True)
    if not args.no_figure:
        for path in make_figure(cells):
            print(f"figure -> {path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
