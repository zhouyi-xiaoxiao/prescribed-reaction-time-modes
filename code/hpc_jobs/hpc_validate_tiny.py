#!/usr/bin/env python3
"""Tiny end-to-end validation of the four fb_hpc_* drivers (simulate + analyze).

Run on a compute node before the production jobs, with PRR_HPC_WORK and
PRR_HPC_OUT pointing at a throw-away test directory (NOT the repository data):
    PRR_HPC_WORK=$HOME/prr_hpc_test/work PRR_HPC_OUT=$HOME/prr_hpc_test/out \
        python3 hpc_jobs/hpc_validate_tiny.py [n5 n3 headline census] --workers 16
Sizes are tiny (statistics meaningless); it checks that every code path runs,
reports per-chunk timings (for sizing the production runs) and a few invariants.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CODE = next(p for p in (HERE.parents[1], HERE.parents[1] / "code") if (p / "fb_hpc_common.py").exists())
sys.path.insert(0, str(CODE))

assert os.environ.get("PRR_HPC_OUT") and os.environ.get("PRR_HPC_WORK"), "set PRR_HPC_OUT/PRR_HPC_WORK"

import numpy as np  # noqa: E402
import fb_hpc_common as hc  # noqa: E402

# NOTE (2026-09-23, resumed operator): the body must sit under a __main__ guard, because the
# 'spawn' worker pool re-imports the main module in every child (job 5768121 failed with the
# multiprocessing bootstrapping RuntimeError).  Module-level size overrides below (CHUNK, ...)
# are NOT propagated to spawn children; they only affect tasks built in the parent.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", default=["census", "headline", "n5", "n3"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    report = {"env": hc.env_info(), "items": {}}


    def timed(label, fn):
        t0 = time.time()
        fn()
        report["items"].setdefault(label, {})["wall_s"] = time.time() - t0
        print(f"[validate] {label} ok, wall {time.time() - t0:.1f}s", flush=True)


    if "census" in args.items:
        import fb_hpc_census as C
        C.CHUNK = 5000
        C.GROUPS = 8
        C.CHECKPOINT_EVERY = 4

        def run_census():
            for cell in C.CELLS:
                C.cmd_simulate(argparse.Namespace(cell=cell, paths="40000", workers=args.workers,
                                                  deadline_min=None))
            C.cmd_analyze(None)
            acc = C.load_acc("m3_eps0.05")
            report["items"]["census"] = {"mean_chunk_s_5e3_paths": float(acc["runtime_sum"][0] / acc["done"].size),
                                         "paths": int(acc["g_paths"].sum())}
        timed("census", run_census)

    if "headline" in args.items:
        import fb_hpc_headline as Hd
        Hd.WALKERS = 40_000
        Hd.CHUNK = 2_000

        def run_headline():
            cells = [Hd.CELLS[i] for i in (0, 1, 3, 11)]
            tasks = Hd.build_tasks(cells, walkers=Hd.WALKERS, chunk=Hd.CHUNK)
            res = hc.run_pool(Hd._chunk, tasks, args.workers, None, label="headline-test")
            report["items"]["headline"] = {"mean_chunk_s_2e3_walkers": float(np.mean([r["runtime_seconds"] for r in res]))}
            Hd.CELLS = cells
            Hd.cmd_analyze(None)
        timed("headline", run_headline)

    if "n5" in args.items:
        import fb_hpc_n5full as N5
        N5.CHUNK = 1000
        N5.BATCH = 100

        def run_n5():
            tasks = N5.build_tasks(list(N5.CELLS), N5.REPS, n_per_rep=2000, chunk=1000)
            res = hc.run_pool(N5.run_chunk, tasks, args.workers, None, label="n5-test")
            report["items"]["n5"] = {"mean_chunk_s_1e3_paths": float(np.mean([r["runtime_seconds"] for r in res]))}
            # bridge-free rerun of one chunk must reproduce em/ex exactly (separate stream)
            t = dict(tasks[0])
            t["bridges"] = False
            t["file"] = str(Path(t["file"]).with_name("nobridge_test.npz"))
            N5.run_chunk(t)
            with np.load(tasks[0]["file"]) as a, np.load(t["file"]) as b:
                same = bool(np.array_equal(a["bm"][:6], b["bm"][:6]))
            os.remove(t["file"])
            report["items"]["n5_em_ex_independent_of_bridges"] = same
            assert same, "bridge stream changed the em/ex schemes"
            N5.cmd_analyze(argparse.Namespace(replicas=50))
        timed("n5", run_n5)

    if "n3" in args.items:
        import fb_hpc_n3full as N3
        N3.BATCH = 50

        def run_n3():
            tasks = N3.build_tasks(list(N3.ANCHORS), N3.REPS, n_per_rep=1000, chunk=500)
            res = hc.run_pool(N3.run_chunk, tasks, args.workers, None, label="n3-test")
            report["items"]["n3"] = {"mean_chunk_s_500_paths": float(np.mean([r["runtime_seconds"] for r in res]))}
            N3.cmd_analyze(None)
        timed("n3", run_n3)

    out = Path(os.environ["PRR_HPC_OUT"]) / "validate_tiny_report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
