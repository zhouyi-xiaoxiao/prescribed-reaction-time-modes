#!/usr/bin/env python3
"""Shared helpers for the Isambard-3 (HPC) runs of the fixed-budget campaign.

Drivers: fb_hpc_n5full.py (A), fb_hpc_n3full.py (B), fb_hpc_headline.py (C),
fb_hpc_census.py (D).  Seeds: base 20260923 with the HPC stream tags 95-99
(95 N5full, 96 N3full, 97 headline direct kill, 98 census, 99 smoke); these are
disjoint from the local campaign tags 80-89 and from fk.USED_TAGS_ELSEWHERE.

Large per-chunk accumulators go to ``$PRR_HPC_WORK`` (default ~/prr_hpc_work,
outside the repository); only reduced summary JSON is written under
artifacts/data/exact_m_fixed_budget/HPC_<item>/.

Worker pools: multiprocessing 'spawn' pools with up to ``--workers`` processes
(140 on a 144-core Grace node).  The local 3-worker etiquette of the laptop
drivers does not apply on a dedicated compute node; the drivers only call the
kernels of the local drivers (identical numerics).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import exact_m_prr_fk_exact_law as fk  # noqa: E402

BASE_SEED = 20260923
HPC_TAGS = {"N5full": 95, "N3full": 96, "headline": 97, "census": 98, "smoke": 99}
for _t in HPC_TAGS.values():
    if _t in fk.USED_TAGS_ELSEWHERE or _t in fk.TAGS.values():
        raise RuntimeError(f"HPC tag {_t} collides with an existing stream")

WORK_ROOT = Path(os.environ.get("PRR_HPC_WORK", str(Path.home() / "prr_hpc_work")))


def out_dir(item: str) -> Path:
    root = Path(os.environ["PRR_HPC_OUT"]) if os.environ.get("PRR_HPC_OUT") else fk.FB_DATA
    d = root / f"HPC_{item}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir(item: str) -> Path:
    d = WORK_ROOT / item
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_info() -> dict:
    return {"host": socket.gethostname(), "machine": platform.machine(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cpu_count": os.cpu_count(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_pool(fn, tasks: list, workers: int, on_result=None, label: str = "hpc",
             deadline: float | None = None) -> list:
    """Run ``fn`` over ``tasks`` in a spawn pool; call ``on_result`` as results arrive.

    ``deadline`` (epoch seconds): a task whose worker starts after the deadline
    is skipped (returns without computing), so the job finishes cleanly with a
    partial, resumable result instead of being killed at the SLURM time limit.
    """
    results = []
    t0 = time.time()
    n = len(tasks)
    if n == 0:
        return results
    skipped = 0
    if workers <= 1:
        for t in tasks:
            r = _guarded_call((fn, t, deadline))
            if r.get("_skipped"):
                skipped += 1
                continue
            results.append(r)
            if on_result:
                on_result(r)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, n)) as pool:
            for r in pool.imap_unordered(_guarded_call, [(fn, t, deadline) for t in tasks]):
                if r.get("_skipped"):
                    skipped += 1
                    continue
                results.append(r)
                if on_result:
                    on_result(r)
                k = len(results)
                if k == 1 or k % max(1, n // 20) == 0 or k == n:
                    print(f"[{label}] {k}/{n} tasks done, wall {time.time() - t0:.0f}s", flush=True)
    if skipped:
        print(f"[{label}] deadline: {skipped}/{n} tasks skipped", flush=True)
    return results


def _guarded_call(args):
    fn, task, deadline = args
    if deadline is not None and time.time() > deadline:
        return {"_skipped": True}
    return fn(task)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fk._jsonable(payload), indent=1, default=fk._json_default))
    os.replace(tmp, path)


def jackknife_se(x_jack: np.ndarray) -> np.ndarray:
    """Delete-one-group jackknife SE along axis 0."""
    x_jack = np.asarray(x_jack, float)
    G = x_jack.shape[0]
    return np.sqrt((G - 1) / G * np.sum((x_jack - x_jack.mean(0)) ** 2, axis=0))
