#!/usr/bin/env python3
"""Aggregate the five exact-m PRR upgrade streams into one campaign summary.

Reads the per-stream summary JSONs under artifacts/data/exact_m_prr_upgrade/
and writes campaign_summary.json with the key numbers, gate results, and
artifact paths for all of W1..W5.  No simulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core

D = core.UPGRADE_DATA


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def figure_path(filename: str) -> str:
    """Return a portable path relative to the report/archive root."""

    return (core.FIGURES / filename).relative_to(core.REPORT).as_posix()


def main() -> None:
    out: dict = {"campaign_seed": core.CAMPAIGN_SEED, "streams": {}}

    # W1
    s = load(D / "w1_phase_diagram" / "w1_phase_diagram_summary.json")
    cells = s["cells"]
    final = {}
    for c in cells:
        key = (c["m"], c["eps"], c["budget"])
        if key not in final or c["refined"]:
            final[key] = c
    out["streams"]["w1_phase_diagram"] = {
        "grid": s["grid"],
        "n_grid_cells": sum(1 for c in cells if not c["refined"]),
        "n_refined_cells": sum(1 for c in cells if c["refined"]),
        "final_mode_count_map": {
            f"m{m}_eps{eps:g}_B{b:g}": final[(m, eps, b)]["mode_count"]
            for (m, eps, b) in sorted(final)
        },
        "m2_all_cells_exact": all(
            v["mode_count"] == 2 for (m, _, _), v in final.items() if m == 2
        ),
        "theory_boundary": s["theory_boundary"],
        "figures": [
            figure_path(f"exact_m_phase_diagram_m{m}_prr{ext}")
            for m in (2, 3)
            for ext in (".png", ".pdf")
        ],
    }

    # W2
    s = load(D / "w2_b0_empirical" / "B0_empirical.json")
    out["streams"]["w2_b0_empirical"] = {
        "legacy_schema_note": s.get(
            "legacy_schema_note",
            (
                "serialized b0 fields denote the protocol-defined operational "
                "threshold B_op only, not B_top or B_cert"
            ),
        ),
        # Mirror the 2026-08-24 provenance note verbatim: the m=3, eps=0.2
        # chain fields below were written under the superseded peak-only
        # convention (see covariance_aware_reclassification.json, w2_chains).
        "covariance_aware_note": s.get("covariance_aware_note"),
        "pass_rule": s["pass_rule"],
        "probe_walkers": s["probe_walkers"],
        "chains": [
            {
                k: c.get(k)
                for k in (
                    "m",
                    "eps",
                    "status",
                    "b0",
                    "b0_bracket",
                    "b0_lower_bound",
                )
            }
            for c in s["chains"]
        ],
        "figures": [
            figure_path(f"exact_m_b0_empirical_prr{ext}")
            for ext in (".png", ".pdf")
        ],
    }

    # W3
    s = load(D / "w3_jitter" / "jitter_robustness.json")
    out["streams"]["w3_jitter"] = {
        "eta_grid": s["eta_grid"],
        "replicas": s["replicas"],
        "replica_walkers": s["replica_walkers"],
        "gate1_zero_budget": s["gate1_zero_budget"]["passed"],
        "points": [
            {
                k: p.get(k)
                for k in (
                    "kind",
                    "m",
                    "eta",
                    "successes",
                    "replicas",
                    "survival_probability",
                    "wilson_95_ci",
                    "classifier_theory_agreement",
                )
            }
            for p in s["points"]
        ],
        "figures": [
            figure_path(f"exact_m_jitter_robustness_prr{ext}")
            for ext in (".png", ".pdf")
        ],
    }

    # W4
    s = load(D / "w4_m5_demo" / "m5_demo.json")
    out["streams"]["w4_m5_demo"] = {
        "config": s["parameters"]["config"],
        "mode_count": s["results"]["mode_count"],
        "peak_times": s["results"]["peak_times"],
        "target_times": s["results"]["target_times"],
        "prominence_over_sigma": s["results"]["prominence_over_sigma"],
        "kills_in_window": s["results"]["kills_in_window"],
        "gates_passed": {
            "gate1": s["validation_gates"]["gate1_zero_budget_no_kills"]["passed"],
            "gate2": s["validation_gates"][
                "gate2_kill_probability_in_unit_interval"
            ],
            "gate3": s["validation_gates"]["gate3_mass_balance"]["passed"],
            "contact_interior": s["validation_gates"][
                "contact_interior_condition"
            ]["passed"],
        },
        "figures": [
            figure_path(f"exact_m_m5_demo_prr{ext}")
            for ext in (".png", ".pdf")
        ],
    }

    # W5
    s = load(D / "w5_d3_spotcheck" / "d3_spotcheck.json")
    out["streams"]["w5_d3_spotcheck"] = {
        "config": s["parameters"]["config"],
        "mode_count": s["results"]["mode_count"],
        "peak_times": s["results"]["peak_times"],
        "kill_fraction": s["results"]["kill_fraction"],
        "event_fraction_in_window": s["results"]["event_fraction_in_window"],
        "contact_fraction_per_walker_step": s["results"][
            "contact_fraction_per_walker_step"
        ],
        "dt_halving_max_abs_z": s["validation_gates"]["gate4_dt_halving_d3"][
            "max_abs_z"
        ],
        "gates_passed": {
            "gate1": s["validation_gates"]["gate1_zero_budget_no_kills"]["passed"],
            "gate2": s["validation_gates"][
                "gate2_kill_probability_in_unit_interval"
            ],
            "gate3": s["validation_gates"]["gate3_mass_balance"]["passed"],
        },
        "figures": [
            figure_path(f"exact_m_d3_spotcheck_prr{ext}")
            for ext in (".png", ".pdf")
        ],
    }

    out_path = D / "campaign_summary.json"
    core.write_json(out_path, out)
    print(f"campaign summary -> {out_path}")


if __name__ == "__main__":
    main()
