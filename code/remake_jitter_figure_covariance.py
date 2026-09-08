#!/usr/bin/env python3
"""Regenerate the W3 jitter figure under the covariance-aware classifier.

Reads the deterministic recheck (w3_jitter/covariance_aware_recheck.json,
produced by w3_jitter_covariance_recheck.py, which reproduced every stored
peak-only replica verdict bit-for-bit before reclassifying) and redraws
exact_m_jitter_robustness_prr.{png,pdf} with the covariance-aware survival
fractions.  The dotted semi-analytic G curves are unchanged (they do not
depend on the classifier statistic).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import exact_m_prr_upgrade_w3 as w3


def main() -> None:
    recheck = json.loads(
        (w3.W3_DIR / "covariance_aware_recheck.json").read_text(encoding="utf-8")
    )
    if not recheck["all_reproduced"]:
        raise SystemExit("stored verdicts not reproduced; refusing to redraw")
    stored = {
        (p["kind"], p["m"], p.get("eta")): p
        for path in sorted(w3.W3_DIR.glob("point_*.json"))
        for p in [json.loads(path.read_text(encoding="utf-8"))]
    }
    points = []
    for p in recheck["points"]:
        sp = stored[(p["kind"], p["m"], p.get("eta"))]
        points.append(
            {
                "kind": p["kind"],
                "m": p["m"],
                "eta": p.get("eta"),
                "replicas": p["replicas"],
                "successes": p["successes_new"],
                "survival_probability": p["survival_probability_new"],
                "wilson_95_ci": p["wilson_95_ci_new"],
                "g_theory_successes": sp["g_theory_successes"],
            }
        )
    for path in w3.make_figure(points):
        print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
