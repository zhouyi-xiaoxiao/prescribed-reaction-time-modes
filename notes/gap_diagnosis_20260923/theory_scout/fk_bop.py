#!/usr/bin/env python3
"""Feynman-Kac path-weight estimator for the exact-m slab Doi model (scratch, theory scout).

Simulates UNKILLED Euler-Maruyama paths of the same model as
code/validate_exact_m_offlattice.py (same parameters, same discretization:
positions updated first, killing evaluated at end-of-step position), and uses
the exact discrete Feynman-Kac identity

   P(first kill at step n) = E[(1 - exp(-B e_n)) exp(-B Lambda_{n-1})],
   e_n = k(X_n) dt,  Lambda_n = sum_{k<=n} e_k   (unit-budget exposure)

to obtain, from ONE path ensemble, the exact-law density for every budget B,
for two kernels:
   'full' : K = 1{|R|_mi < a} (B/W) sum_j w_j phi_j(Z)      (paper's model, d=2)
   'free' : K =               (B/W) sum_j w_j phi_j(Z)      (contact gate removed)
It also stores per-walker cumulative exposure at checkpoint times (-> exact
basin masses for any B offline), per-walker per-slab exposures, and contact
indicators at the target times.

Usage: fk_paths.py m eps N seed out.npz [dt]
"""
import math
import sys
import time

import numpy as np

GAMMA, D0, ZBAR, Z0, W, A, RHO, U0, SIGP0, RPAR0, RPERP0 = (
    1.0, 1.0, 0.0, 4.0, 1.0, 0.4, 1.0, 0.3, 0.3, 0.1, 0.0)
TARGETS = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}
TMAX = 4.0
import os
BUDGETS = np.array([float(x) for x in os.environ.get("FK_BUDGETS","4,8").split(",")])


def mu(t):
    return ZBAR + (Z0 - ZBAR) * np.exp(-GAMMA * t)


def mixture_valleys(m, eps):
    t = np.arange(0.3, 3.8, 1e-5)
    s2 = eps**2 * (D0 / (2 * GAMMA) + RHO**2)
    cz = np.array([mu(tj) for tj in TARGETS[m]])
    H = sum(np.exp(-(mu(t) - c) ** 2 / (2 * s2)) for c in cz) / m
    d = np.diff(H)
    idx = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return [float(t[i]) for i in idx]


def run(m, eps, N, seed, dt=1e-3, chunk=50_000):
    w = np.full(m, 1.0 / m)
    cz = np.array([mu(tj) for tj in TARGETS[m]])
    steps = int(round(TMAX / dt))
    tgrid = (np.arange(steps) + 1) * dt
    valleys = mixture_valleys(m, eps)
    checkpoints = [0.5] + valleys + [3.5, TMAX]
    ck_steps = [int(round(c / dt)) for c in checkpoints]  # Lambda after step index c/dt
    tj_steps = [int(round(tj / dt)) for tj in TARGETS[m]]
    slab_norm = 1.0 / (math.sqrt(2 * math.pi) * eps * RHO) / W
    inv2v = 1.0 / (2 * (eps * RHO) ** 2)
    decay = 1.0 - GAMMA * dt
    pull = GAMMA * ZBAR * dt
    nz = eps * math.sqrt(D0 * dt)
    nr = 2 * eps * math.sqrt(D0 * dt)
    nB = BUDGETS.size
    # accumulators (sums over walkers; divide by N at the end)
    G = {"full": np.zeros(steps), "free": np.zeros(steps)}
    FK = {"full": np.zeros((nB, steps)), "free": np.zeros((nB, steps))}
    sizes0 = [chunk] * (N // chunk) + ([N % chunk] if N % chunk else [])
    nbatch = len(sizes0)
    FKb = {"full": np.zeros((nbatch, nB, steps)), "free": np.zeros((nbatch, nB, steps))}
    lam_ck = {"full": [], "free": []}
    slab_exp = {"full": [], "free": []}
    chi_tj = []
    ss = np.random.SeedSequence([int(seed), m, int(round(eps * 1e6)), int(round(dt * 1e9))])
    sizes = [chunk] * (N // chunk) + ([N % chunk] if N % chunk else [])
    kids = ss.spawn(len(sizes))
    t0 = time.time()
    gidx = 0
    for size, kid in zip(sizes, kids):
        rng = np.random.Generator(np.random.PCG64(kid))
        z = Z0 + math.sqrt(eps**2 * D0 / (2 * GAMMA)) * rng.standard_normal(size)
        rp = RPAR0 + eps * U0 * rng.standard_normal(size)
        rq = np.mod(RPERP0 + eps * SIGP0 * rng.standard_normal(size), W)
        Lam = {"full": np.zeros(size), "free": np.zeros(size)}
        ck = {"full": np.zeros((size, len(ck_steps)), np.float32),
              "free": np.zeros((size, len(ck_steps)), np.float32)}
        sx = {"full": np.zeros((size, m)), "free": np.zeros((size, m))}
        chi = np.zeros((size, m), np.int8)
        cidx = gidx
        gidx += 1
        for n in range(steps):
            noise = rng.standard_normal((3, size))
            z *= decay; z += pull; z += nz * noise[0]
            rp *= decay; rp += nr * noise[1]
            rq += nr * noise[2]; np.mod(rq, W, out=rq)
            qmi = np.minimum(rq, W - rq)
            contact = (rp * rp + qmi * qmi) < A * A
            step_no = n + 1
            for jj, s_ in enumerate(tj_steps):
                if step_no == s_:
                    chi[:, jj] = contact
            # active walkers: within 9 slab-widths of some centre
            dmin = np.min(np.abs(z[:, None] - cz[None, :]), axis=1)
            act = np.flatnonzero(dmin < 9.0 * eps * RHO)
            if act.size:
                za = z[act]
                comps = np.exp(-((za[:, None] - cz[None, :]) ** 2) * inv2v) * (w[None, :] * slab_norm * dt)
                e_free = comps.sum(axis=1)
                ca = contact[act]
                e_full = e_free * ca
                for key, e, cmp_ in ((("full", e_full, comps * ca[:, None]),) if os.environ.get("FK_FULL_ONLY") else (("free", e_free, comps), ("full", e_full, comps * ca[:, None]))):
                    Lprev = Lam[key][act]
                    G[key][n] += e.sum() / dt
                    # FK density for every budget: (1-exp(-B e)) exp(-B Lprev)
                    wts = -np.expm1(-BUDGETS[:, None] * e[None, :]) * np.exp(-BUDGETS[:, None] * Lprev[None, :])
                    sw = wts.sum(axis=1)
                    FK[key][:, n] += sw
                    FKb[key][cidx, :, n] += sw
                    Lam[key][act] = Lprev + e
                    sx[key][act] += cmp_
            for ci, cs in enumerate(ck_steps):
                if step_no == cs:
                    ck["full"][:, ci] = Lam["full"]
                    ck["free"][:, ci] = Lam["free"]
        for key in ("full", "free"):
            lam_ck[key].append(ck[key])
            slab_exp[key].append(sx[key].astype(np.float32))
        chi_tj.append(chi)
        print(f"chunk done size={size} elapsed={time.time()-t0:.1f}s", flush=True)
    out = dict(
        m=m, eps=eps, N=N, seed=seed, dt=dt, tgrid=tgrid, budgets=BUDGETS,
        checkpoints=np.array(checkpoints), valleys=np.array(valleys),
        centres=cz, weights=w,
        G_full=G["full"] / N, G_free=G["free"] / N,
        FK_full=FK["full"] / N, FK_free=FK["free"] / N,
        FKb_full=FKb["full"] / np.array(sizes0)[:, None, None], FKb_free=FKb["free"] / np.array(sizes0)[:, None, None],
        lamck_full=np.concatenate(lam_ck["full"]), lamck_free=np.concatenate(lam_ck["free"]),
        slabexp_full=np.concatenate(slab_exp["full"]), slabexp_free=np.concatenate(slab_exp["free"]),
        chi_tj=np.concatenate(chi_tj), runtime=time.time() - t0,
    )
    return out


if __name__ == "__main__":
    m = int(sys.argv[1]); eps = float(sys.argv[2]); N = int(sys.argv[3]); seed = int(sys.argv[4])
    outp = sys.argv[5]
    dt = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3
    res = run(m, eps, N, seed, dt=dt)
    np.savez_compressed(outp, **res)
    print("saved", outp, "runtime", res["runtime"])
