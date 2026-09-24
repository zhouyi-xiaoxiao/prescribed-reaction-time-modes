#!/usr/bin/env python3
"""Regenerate the B_op(eps) figure under the covariance-aware classifier.

Codex cross-audit finding C1: under the covariance-aware prominence statistic
the three formerly passing (m=3, eps=0.2) probes fail the five-sigma rule, so
that chain has no certified crossing anywhere in the scanned budget range
[0.125, 8] and the previously plotted B_op = 0.28 point is withdrawn.  All
other chains' probe verdicts are unchanged (see
covariance_aware_reclassification.json), so their plotted brackets are
identical.

This driver reuses exact_m_prr_upgrade_w2.make_figure on the stored chains
with the (3, 0.2) chain reclassified to ``no_certified_crossing`` (plotted as
absent; the caption carries the statement).  No Monte Carlo rerun.

Stream D1 (2026-09-09; default restored 2026-09-23): the default reproduces
the single-panel figure as submitted (HEAD PDF).  ``--topology`` (opt-in)
overlays the classifier-free mean-field topology threshold B_top^mf(eps) from
``mean_field_topology.json`` (written by ``exact_m_prr_mean_field_topology.py``)
as a second panel and, where it falls inside the B_op axis range, as open
diamonds on the B_op panel.  ``--no-topology`` is accepted for backward
compatibility and is the default behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import exact_m_prr_upgrade_w2 as w2


TOPOLOGY_JSON = core.UPGRADE_DATA / "mean_field_topology.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--topology",
        action="store_true",
        help="opt-in: overlay B_top^mf (requires mean_field_topology.json)",
    )
    parser.add_argument(
        "--no-topology",
        action="store_true",
        help="default; single-panel figure as submitted (kept for compatibility)",
    )
    args = parser.parse_args()
    out_path = w2.W2_DIR / "B0_empirical.json"
    chains = json.loads(out_path.read_text(encoding="utf-8"))["chains"]
    recl = json.loads(
        (core.UPGRADE_DATA / "covariance_aware_reclassification.json").read_text(
            encoding="utf-8"
        )
    )
    new_by_key = {(c["m"], c["eps"]): c for c in recl["w2_chains"]}
    for chain in chains:
        new = new_by_key[(chain["m"], chain["eps"])]
        if new["new_status"] == "no_certified_crossing_at_1e6_walkers":
            chain["status"] = "no_certified_crossing"
            chain["b0"] = None
            chain["b0_bracket"] = None
        elif new["new_status"] == "bisected":
            assert new.get("bracket_unchanged"), (chain["m"], chain["eps"])
        elif new["new_status"] == "right_censored":
            assert chain["status"] == "right_censored"
        else:
            raise AssertionError(new["new_status"])
    topology = None
    if args.topology and not args.no_topology:
        if not TOPOLOGY_JSON.exists():
            raise SystemExit(
                f"{TOPOLOGY_JSON} missing: run exact_m_prr_mean_field_topology.py "
                "first or pass --no-topology"
            )
        topology = json.loads(TOPOLOGY_JSON.read_text(encoding="utf-8"))
        print(
            "B_top^mf overlay: "
            + ", ".join(
                f"{c['label']}={c['B_top_mf_3sf']}"
                for c in topology["configurations"]
                if c.get("family") == "w1_w2_standard_design"
            ),
            flush=True,
        )
    for path in w2.make_figure(chains, topology=topology):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
