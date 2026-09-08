#!/usr/bin/env python3
"""W1 phase diagram: observed mode count over (eps, B) for m=2 and m=3.

Grid: eps in {0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25} x
      B in {0.125,0.25,0.5,1,2,4}; m=2 weights 50/50, m=3 uniform;
      1e6 walkers/cell through the validated base.run_config path
      (dt=1e-3, tmax=4, window [0.5,3.5], 5sigma+5% classifier).
Refinement: cells 4-adjacent to a verdict change (mode_count == m flips)
      rerun at 3e6 walkers on an independent stream.
Theory overlay: per (m, eps), semi-analytic count of G window maxima
      (free-exposure factorization; B-independent).

Outputs: artifacts/data/exact_m_prr_upgrade/w1_phase_diagram/cell_*.json,
         w1_phase_diagram_summary.json,
         artifacts/figures/exact_m_phase_diagram_m{2,3}_prr.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exact_m_prr_upgrade_core as core
import validate_exact_m_offlattice as base

W1_DIR = core.UPGRADE_DATA / "w1_phase_diagram"
EPS_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25)
B_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
M_WEIGHTS = {2: (0.5, 0.5), 3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)}
GRID_WALKERS = 1_000_000
REFINE_WALKERS = 3_000_000
CHUNK = 100_000
DT = base.DEFAULT_DT
TMAX = base.DEFAULT_TMAX
BANDWIDTH = base.DEFAULT_BANDWIDTH
SEED = core.CAMPAIGN_SEED


def _cell_stem(m: int, eps: float, budget: float, refined: bool) -> str:
    return f"cell_m{m}_eps{eps:g}_B{budget:g}" + ("_refined" if refined else "")


def run_cell(task: tuple) -> dict:
    m, eps, budget, refined, gate_zero = task
    weights = M_WEIGHTS[m]
    walkers = REFINE_WALKERS if refined else GRID_WALKERS
    tag = core.TAG_W1_REFINE if refined else core.TAG_W1_GRID
    result = base.run_config(
        m=m,
        eps=eps,
        budget=budget,
        weights=weights,
        walkers=walkers,
        chunk=CHUNK,
        dt=DT,
        tmax=TMAX,
        seed=SEED,
        tag=tag,
        verbose=False,
    )
    summary = base.summarize_config(result, bandwidth=BANDWIDTH, tmax=TMAX)
    theory = core.g_window_maxima_count(m=m, eps=eps, weights=weights)
    payload = {
        "parameters": {
            "stream": "w1_phase_diagram",
            "model_parameters": core.model_dict(core.MODEL),
            "config": {
                "m": m,
                "eps": eps,
                "budget": budget,
                "weights": list(weights),
                "target_times": list(base.TARGET_TIMES[m]),
                "walkers": walkers,
                "chunk": CHUNK,
                "dt": DT,
                "tmax": TMAX,
                "seed": SEED,
                "tag": tag,
                "refined": refined,
                "bandwidth": BANDWIDTH,
            },
            "rng": "numpy Philox, SeedSequence-spawned substream per chunk",
            "killing": "per-step probability 1 - exp(-kappa dt) via expm1",
        },
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
        "g_theory_diagnostics": theory,
    }
    path = W1_DIR / f"{_cell_stem(m, eps, budget, refined)}.json"
    core.write_json(path, payload)
    return {
        "m": m,
        "eps": eps,
        "budget": budget,
        "refined": refined,
        "walkers": walkers,
        "mode_count": summary["mode_count"],
        "passes": summary["passes_m_mode_criterion"],
        "peak_times": summary["peak_times"],
        "kill_fraction": summary["kill_fraction"],
        "event_fraction_in_window": summary["event_fraction_in_window"],
        "g_n_window_maxima": theory["n_window_maxima"],
        "runtime_seconds": summary["runtime_seconds"],
        "file": path.name,
    }


def _neighbours(ie: int, ib: int) -> list[tuple[int, int]]:
    out = []
    for de, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        je, jb = ie + de, ib + db
        if 0 <= je < len(EPS_GRID) and 0 <= jb < len(B_GRID):
            out.append((je, jb))
    return out


def boundary_cells(rows: list[dict]) -> list[tuple[int, float, float]]:
    """Cells 4-adjacent to a verdict flip (mode_count == m), per m."""
    marked = []
    for m in sorted(M_WEIGHTS):
        verdict = {}
        for row in rows:
            if row["m"] == m and not row["refined"]:
                verdict[(row["eps"], row["budget"])] = row["passes"]
        for ie, eps in enumerate(EPS_GRID):
            for ib, budget in enumerate(B_GRID):
                mine = verdict.get((eps, budget))
                if mine is None:
                    continue
                for je, jb in _neighbours(ie, ib):
                    other = verdict.get((EPS_GRID[je], B_GRID[jb]))
                    if other is not None and other != mine:
                        marked.append((m, eps, budget))
                        break
    return marked


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(W1_DIR.glob("cell_m*_eps*_B*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["parameters"]["config"]
        summary = payload["results"]
        rows.append(
            {
                "m": cfg["m"],
                "eps": cfg["eps"],
                "budget": cfg["budget"],
                "refined": cfg["refined"],
                "walkers": cfg["walkers"],
                "mode_count": summary["mode_count"],
                "passes": summary["passes_m_mode_criterion"],
                "peak_times": summary["peak_times"],
                "kill_fraction": summary["kill_fraction"],
                "event_fraction_in_window": summary["event_fraction_in_window"],
                "g_n_window_maxima": payload["g_theory_diagnostics"][
                    "n_window_maxima"
                ],
                "runtime_seconds": summary["runtime_seconds"],
                "file": path.name,
            }
        )
    return rows


def write_summary(rows: list[dict], *, wall_seconds: float | None) -> Path:
    theory_boundary = {}
    for m, weights in M_WEIGHTS.items():
        cols = {}
        for row in rows:
            if row["m"] == m and not row["refined"]:
                cols[row["eps"]] = row["g_n_window_maxima"]
        retained = [eps for eps in EPS_GRID if cols.get(eps) == m]
        lost = [eps for eps in EPS_GRID if cols.get(eps, m) != m]
        theory_boundary[str(m)] = {
            "g_maxima_by_eps": {f"{eps:g}": cols.get(eps) for eps in EPS_GRID},
            "eps_retaining_m_maxima": retained,
            "eps_losing_m_maxima": lost,
            "boundary_eps_interval": (
                [max(retained), min(lost)] if retained and lost else None
            ),
        }
    summary = {
        "stream": "w1_phase_diagram",
        "grid": {
            "eps": list(EPS_GRID),
            "budget": list(B_GRID),
            "m_weights": {str(m): list(w) for m, w in M_WEIGHTS.items()},
            "grid_walkers": GRID_WALKERS,
            "refine_walkers": REFINE_WALKERS,
            "dt": DT,
            "tmax": TMAX,
            "seed": SEED,
            "bandwidth": BANDWIDTH,
            "window": list(core.WINDOW),
        },
        "verdict_rule": "cell verdict = classifier mode count (5 sigma + 5% rule) == m",
        "theory_boundary": theory_boundary,
        "cells": rows,
        "wall_seconds": wall_seconds,
    }
    out = W1_DIR / "w1_phase_diagram_summary.json"
    core.write_json(out, summary)
    return out


def make_figures(rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    core.apply_prr_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

    def level_colour(k: int, m: int) -> str:
        """Okabe-Ito colourblind-safe mapping for the mode count."""
        if k == m:
            return core.OI_BLUE
        if k == m - 1:
            return core.OI_ORANGE
        if k == m + 1:
            return core.OI_GREEN
        if k == m - 2:
            return core.OI_YELLOW
        return core.OI_GREY

    def text_colour(k: int, m: int) -> str:
        r, g, b = to_rgb(level_colour(k, m))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "white" if luminance < 0.5 else "black"

    written = []
    for m in sorted(M_WEIGHTS):
        best = {}
        for row in rows:
            if row["m"] != m:
                continue
            key = (row["eps"], row["budget"])
            if key not in best or row["refined"]:
                best[key] = row
        counts = np.full((len(B_GRID), len(EPS_GRID)), np.nan)
        refined_mask = np.zeros_like(counts, dtype=bool)
        for ie, eps in enumerate(EPS_GRID):
            for ib, budget in enumerate(B_GRID):
                row = best.get((eps, budget))
                if row is not None:
                    counts[ib, ie] = row["mode_count"]
                    refined_mask[ib, ie] = row["refined"]

        max_count = int(np.nanmax(counts)) if np.isfinite(counts).any() else m
        n_levels = max(max_count, m) + 1
        levels = np.arange(-0.5, n_levels + 0.5, 1.0)
        cmap = ListedColormap([level_colour(k, m) for k in range(n_levels)])
        norm = BoundaryNorm(levels, cmap.N)

        fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
        mesh = ax.imshow(
            counts,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(-0.5, len(EPS_GRID) - 0.5, -0.5, len(B_GRID) - 0.5),
        )
        for ie in range(len(EPS_GRID)):
            for ib in range(len(B_GRID)):
                if np.isnan(counts[ib, ie]):
                    continue
                cell = int(counts[ib, ie])
                mark = "*" if refined_mask[ib, ie] else ""
                ax.text(
                    ie,
                    ib,
                    f"{cell}{mark}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_colour(cell, m),
                    fontweight="bold" if cell == m else "normal",
                )
        ax.set_xticks(range(len(EPS_GRID)), [f"{v:g}" for v in EPS_GRID])
        ax.set_yticks(range(len(B_GRID)), [f"{v:g}" for v in B_GRID])
        ax.set_xlabel(r"slab width parameter $\varepsilon$")
        ax.set_ylabel(r"installed budget $B$ (log-spaced grid)")
        cbar = fig.colorbar(mesh, ax=ax, ticks=range(0, n_levels))
        cbar.set_label("classifier mode count in window")
        cbar.ax.tick_params(length=0)

        # Theory overlay: per-column semi-analytic G window-maxima counts on a
        # top secondary axis, and the eps interval where G loses the m-th
        # window maximum as a dashed boundary line (annotation lives in the
        # top axis label, clear of every cell label).
        theory_by_eps = {}
        for row in rows:
            if row["m"] == m and not row["refined"]:
                theory_by_eps[row["eps"]] = row["g_n_window_maxima"]
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(m, f"{m}th")
        retained = [eps for eps in EPS_GRID if theory_by_eps.get(eps) == m]
        lost = [eps for eps in EPS_GRID if theory_by_eps.get(eps, m) != m]
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(
            range(len(EPS_GRID)),
            [f"{theory_by_eps.get(eps, '?')}" for eps in EPS_GRID],
        )
        sec.tick_params(axis="x", colors=core.OI_VERMILLION, length=0, pad=2)
        if retained and lost:
            xline = (
                EPS_GRID.index(max(retained)) + EPS_GRID.index(min(lost))
            ) / 2.0
            # Darker than the orange field so the dashed boundary stays
            # visible where it crosses count-2 cells.
            ax.axvline(xline, color="#7F2704", lw=1.6, ls="--")
            sec.set_xlabel(
                rf"semi-analytic $G$ window maxima per column; dashed: $G$ "
                rf"loses the {ordinal} maximum for "
                rf"$\varepsilon\in({max(retained):g},\,{min(lost):g}]$",
                color=core.OI_VERMILLION,
                fontsize=7.5,
            )
        else:
            sec.set_xlabel(
                rf"semi-analytic $G$ window maxima per column: {m} retained "
                rf"for all $\varepsilon\leq{max(EPS_GRID):g}$",
                color=core.OI_VERMILLION,
                fontsize=7.5,
            )
        stem = core.FIGURES / f"exact_m_phase_diagram_m{m}_prr"
        written.extend(core.save_figure(fig, stem))
        plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("grid", "refine", "figure", "all"), default="all"
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()

    if args.phase in ("grid", "all"):
        gate_zero = base.gate_zero_budget(SEED, dt=DT)
        print(
            f"gate B=0 PASS: contact_steps={gate_zero['contact_steps']} kills=0",
            flush=True,
        )
        tasks = [
            (m, eps, budget, False, gate_zero)
            for m in sorted(M_WEIGHTS)
            for eps in EPS_GRID
            for budget in B_GRID
        ]
        # Slowest (small B) first for tail load-balance.
        tasks.sort(key=lambda t: t[2])
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                done += 1
                print(
                    f"[{done}/{len(tasks)}] m={row['m']} eps={row['eps']:g} "
                    f"B={row['budget']:g}: modes={row['mode_count']} "
                    f"({'PASS' if row['passes'] else 'FAIL'}) "
                    f"killfrac={row['kill_fraction']:.3f} "
                    f"G_max={row['g_n_window_maxima']} "
                    f"t={row['runtime_seconds']:.0f}s",
                    flush=True,
                )

    if args.phase in ("refine", "all"):
        rows = load_rows()
        base_rows = [r for r in rows if not r["refined"]]
        marked = boundary_cells(base_rows)
        existing = {
            (r["m"], r["eps"], r["budget"]) for r in rows if r["refined"]
        }
        marked = [c for c in marked if c not in existing]
        print(f"refining {len(marked)} verdict-boundary-adjacent cells "
              f"at {REFINE_WALKERS:,} walkers: {marked}", flush=True)
        gate_zero = base.gate_zero_budget(SEED, dt=DT)
        tasks = [(m, eps, budget, True, gate_zero) for (m, eps, budget) in marked]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                done += 1
                print(
                    f"[refine {done}/{len(tasks)}] m={row['m']} "
                    f"eps={row['eps']:g} B={row['budget']:g}: "
                    f"modes={row['mode_count']} "
                    f"({'PASS' if row['passes'] else 'FAIL'}) "
                    f"t={row['runtime_seconds']:.0f}s",
                    flush=True,
                )

    rows = load_rows()
    out = write_summary(rows, wall_seconds=time.perf_counter() - started)
    print(f"summary -> {out}", flush=True)
    if args.phase in ("figure", "all"):
        for path in make_figures(rows):
            print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
