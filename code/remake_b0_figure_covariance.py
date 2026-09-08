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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import exact_m_prr_upgrade_w2 as w2


def main() -> None:
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
    for path in w2.make_figure(chains):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
