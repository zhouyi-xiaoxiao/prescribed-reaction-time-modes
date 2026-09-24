#!/usr/bin/env python3
"""TH-6 numerical checks: the exact tilted-bridge factorization (midpoint-only
killing, contact = 1) and the key estimates of the TH-6 proof.

Theory (manuscript/cnsns_submission/theory/TH6_exact_count*.tex):
  f_mid(t) = B sum_k h_k Gamma_k(t) A_k(t),  Gamma_k = phi_{eps S*}(mu(t)-mu_k),
  A_k(t)   = E exp(-B int_0^t V(M_k(t,r) + eps Y_r) dr),
  M_k(t,r) = mu(t-r) + T_k(t) e^{-gamma r},  T_k = (sZ^2/S*^2)(mu_k - mu(t)),
  Y_r      = c_Y eta e^{-gamma r} + N_r  (N = OU from 0; law independent of t, k),
  A_k'(t)  = -B E[(V(M_k(t,t)+eps Y_t) + int_0^t V'(P_r) d_t M_k dr) e^{-E}].
Because Y does not depend on t, ONE ensemble of Y paths serves every t
(common random numbers), and A_k = O(1) for the dominant components, so
log f and (log f)' are obtained with RELATIVE accuracy even where
f ~ e^{-100} (no rare-event sampling needed).

Parts:
  (a) factorization check against GPT-6's killed-OU PDE (contact = 1) at the
      PDE grid times, including the Gaussian-small valleys;
  (b) key-estimate diagnostics at eps in {0.05, 0.025, 0.0125}:
      matching zones  eps*b_j vs the Gaussian slope -v_j^2 s / S*^2,
      gap sectors     eps^2 * max|b_k| (retained k) and the sign of D;
  (c) summary of the PDE extrema (counts of maxima/minima, endpoint slopes)
      read from GPT-6's earlier outputs (provenance hashed);
  (d) convexity off the passage windows on the same PDE outputs:
      f''/f = D' + D^2 > 0 wherever min_j |mu(t)-mu_j| >= L eps.

Seeds: numpy SeedSequence([20260923, 88, case_index]) (base 20260923, TH-6
stream tag 88).  Workers: 1 process.  Output:
  artifacts/data/exact_m_fixed_budget/TH6/th6_tilted_bridge_checks.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
OUT_DIR = REPORT / "artifacts" / "data" / "exact_m_fixed_budget" / "TH6"
PDE_DIR = Path("<local-ensemble-store>/codex_theory_prr_20260923/theory_checks")

SEED_BASE = 20260923
STREAM_TAG = 88

# Paper anchor (validate_exact_m_offlattice.ModelParameters defaults).
GAMMA = 1.0
D0 = 1.0
ZBAR = 0.0
Z0 = 4.0
RHO = 1.0
W = 1.0
DIM = 2
SZ2 = D0 / (2 * GAMMA)
S2 = SZ2 + RHO**2
CY = math.sqrt(SZ2 * RHO**2 / S2)
TARGET_TIMES = {2: (1.0, 2.5), 3: (0.8, 1.6, 2.8)}
WINDOW = (0.5, 3.5)


def mu(t):
    return ZBAR + (Z0 - ZBAR) * np.exp(-GAMMA * np.asarray(t, dtype=float))


def mup(t):
    return -GAMMA * (Z0 - ZBAR) * np.exp(-GAMMA * np.asarray(t, dtype=float))


class Case:
    def __init__(self, m, eps, weights=None, n_paths=2000, case_index=0, rmax=3.6):
        self.m = m
        self.eps = eps
        self.w = np.full(m, 1.0 / m) if weights is None else np.asarray(weights, float)
        self.h = self.w / W ** (DIM - 1)
        self.tj = np.array(TARGET_TIMES[m])
        self.muk = mu(self.tj)
        self.dr = eps / 50.0
        self.n_paths = n_paths
        ss = np.random.SeedSequence([SEED_BASE, STREAM_TAG, case_index])
        self.seed_entropy = [SEED_BASE, STREAM_TAG, case_index]
        rng = np.random.default_rng(ss)
        nr = int(math.ceil(rmax / self.dr))
        a = math.exp(-GAMMA * self.dr)
        sd = math.sqrt(SZ2 * (1 - a * a))
        Y = np.empty((n_paths, nr + 1))
        Y[:, 0] = CY * rng.standard_normal(n_paths)
        for i in range(nr):
            Y[:, i + 1] = a * Y[:, i] + sd * rng.standard_normal(n_paths)
        self.Y = Y
        self.halfwidth = 12.0 * eps * (RHO + math.sqrt(SZ2))

    def comp(self, t_index, k, B):
        """A_k(t), A_k'(t) with batch standard errors; t = t_index * dr."""
        eps, dr = self.eps, self.dr
        t = t_index * dr
        r = np.arange(t_index + 1) * dr
        Tk = (SZ2 / S2) * (self.muk[k] - mu(t))
        Tkp = -(SZ2 / S2) * mup(t)
        M = mu(t - r) + Tk * np.exp(-GAMMA * r)
        dM = mup(t - r) + Tkp * np.exp(-GAMMA * r)
        near = np.min(np.abs(M[:, None] - self.muk[None, :]), axis=1) < self.halfwidth
        idx = np.nonzero(near)[0]
        wts = np.full(t_index + 1, dr)
        wts[0] *= 0.5
        wts[-1] *= 0.5
        if idx.size == 0:
            ones = np.ones(self.n_paths)
            return 1.0, 0.0, 0.0, 0.0, ones, np.zeros(self.n_paths)
        P = M[idx][None, :] + eps * self.Y[:, idx]
        V = np.zeros_like(P)
        Vp = np.zeros_like(P)
        for i in range(self.m):
            y = (P - self.muk[i]) / (eps * RHO)
            ph = self.h[i] * np.exp(-0.5 * y * y) / (math.sqrt(2 * math.pi) * eps * RHO)
            V += ph
            Vp += -y / (eps * RHO) * ph
        E = B * (V * wts[idx]).sum(axis=1)
        J = (Vp * (dM[idx] * wts[idx])).sum(axis=1)
        V0 = V[:, -1] if idx[-1] == t_index else np.zeros(self.n_paths)
        e = np.exp(-E)
        samples_A = e
        samples_dA = -B * (V0 + J) * e
        nb = 20
        bA = samples_A.reshape(nb, -1).mean(axis=1)
        bdA = samples_dA.reshape(nb, -1).mean(axis=1)
        return (float(bA.mean()), float(bdA.mean()), float(bA.std(ddof=1) / math.sqrt(nb)),
                float(bdA.std(ddof=1) / math.sqrt(nb)), samples_A, samples_dA)

    def density(self, t_index, B):
        eps = self.eps
        t = t_index * self.dr
        Gam = np.exp(-(mu(t) - self.muk) ** 2 / (2 * eps**2 * S2)) / (math.sqrt(2 * math.pi) * eps * math.sqrt(S2))
        ell = -mup(t) * (mu(t) - self.muk) / (eps**2 * S2)
        A = np.zeros(self.m)
        dA = np.zeros(self.m)
        seA = np.zeros(self.m)
        sedA = np.zeros(self.m)
        sa = []
        sd = []
        # log-scale the Gaussian factors to avoid underflow
        logGam = -(mu(t) - self.muk) ** 2 / (2 * eps**2 * S2) - math.log(math.sqrt(2 * math.pi) * eps * math.sqrt(S2))
        for k in range(self.m):
            A[k], dA[k], seA[k], sedA[k], s1, s2 = self.comp(t_index, k, B)
            sa.append(s1)
            sd.append(s2)
        lmax = np.max(logGam + np.log(self.h) + np.log(np.maximum(A, 1e-300)))
        wk = np.exp(logGam + np.log(self.h) - lmax)
        num = wk * A
        dnum = wk * (ell * A + dA)
        ftil = num.sum()
        logf = math.log(B) + lmax + math.log(ftil)
        D = dnum.sum() / ftil
        # batch SE of D (common paths across k)
        nb = 20
        Ab = np.array([s.reshape(nb, -1).mean(axis=1) for s in sa])
        dAb = np.array([s.reshape(nb, -1).mean(axis=1) for s in sd])
        Db = ((wk[:, None] * (ell[:, None] * Ab + dAb)).sum(axis=0)) / ((wk[:, None] * Ab).sum(axis=0))
        seD = float(Db.std(ddof=1) / math.sqrt(nb))
        pi = num / ftil
        return dict(t=float(t), logf=float(logf), D=float(D), seD=seD, A=A.tolist(), seA=seA.tolist(),
                    b=(dA / np.maximum(A, 1e-300)).tolist(), ell=ell.tolist(), pi=pi.tolist())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def part_a(results):
    """Factorization check vs PDE (contact = 1)."""
    specs = [(2, 0.05, "reduced_m2_e005_fine"), (2, 0.025, "reduced_m2_e0025_fine"),
             (3, 0.05, "reduced_m3_e.05_fine"), (3, 0.025, "reduced_m3_e.025_fine")]
    out = []
    for ci, (m, eps, stem) in enumerate(specs):
        npz = PDE_DIR / f"{stem}.npz"
        js = PDE_DIR / f"{stem}.json"
        if not npz.exists():
            out.append(dict(m=m, eps=eps, missing=str(npz)))
            continue
        pde = np.load(npz)
        meta = json.loads(js.read_text())
        case = Case(m, eps, n_paths=4000, case_index=100 + ci)
        tg = pde["t"]
        rows = []
        for bi, B in enumerate(pde["budgets"]):
            B = float(B)
            if B not in (0.5, 1.0, 4.0):
                continue
            summ = [s for s in meta["summaries"] if abs(s["B"] - B) < 1e-12][0]
            times = [e["t"] for e in summ["extrema_I"]]
            times += [0.6, 3.3] + [0.5 * (a + b) for a, b in zip(TARGET_TIMES[m][:-1], TARGET_TIMES[m][1:])]
            for tt in times:
                gi = int(np.argmin(np.abs(tg - tt)))
                tgrid = float(tg[gi])
                t_index = int(round(tgrid / case.dr))
                rep = case.density(t_index, B)
                rows.append(dict(B=B, t=tgrid, rep_logf=rep["logf"], pde_logf=float(pde["logf"][gi, bi]),
                                 rep_D=rep["D"], rep_seD=rep["seD"], pde_D=float(pde["log_derivative"][gi, bi]),
                                 A=rep["A"], b=rep["b"]))
        dl = [abs(r["rep_logf"] - r["pde_logf"]) for r in rows]
        out.append(dict(m=m, eps=eps, pde_file=str(npz), pde_sha256=sha256(npz), n_paths=case.n_paths,
                        dr=case.dr, seed_entropy=case.seed_entropy, rows=rows,
                        max_abs_dlogf=max(dl), min_pde_logf=min(r["pde_logf"] for r in rows)))
        print(f"[a] m={m} eps={eps}: max|dlogf|={max(dl):.4f} over {len(rows)} points "
              f"(min log f={min(r['pde_logf'] for r in rows):.1f})", flush=True)
    results["part_a_factorization_vs_pde"] = out


def profile_peak(lam: float) -> float:
    """y_lambda: unique critical point of F_lambda = p_lambda * phi_theta (TH-3 profile),
    via F_lambda(y) = lam phi_{sqrt(1+th^2)}(y) E exp(-lam Phi(y/(1+th^2) + nu eta))."""
    th2 = SZ2 / RHO**2
    nu = math.sqrt(th2 / (1 + th2))
    xg, wg = np.polynomial.hermite_e.hermegauss(80)
    wg = wg / wg.sum()
    from math import erf
    Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / math.sqrt(2))))
    ys = np.linspace(-8, 3, 4401)
    vals = []
    for y in ys:
        vals.append(-y * y / (2 * (1 + th2)) + math.log(np.dot(wg, np.exp(-lam * Phi(y / (1 + th2) + nu * xg)))))
    vals = np.array(vals)
    return float(ys[int(np.argmax(vals))])


def part_b(results):
    """Key-estimate diagnostics (matching zones relative to the limiting peak, gap sectors)."""
    out = []
    ci = 200
    for m in (2, 3):
        for eps in (0.05, 0.025, 0.0125):
            case = Case(m, eps, n_paths=2000, case_index=ci)
            ci += 1
            vj = np.abs(mup(case.tj))
            for B in (1.0, 8.0, 50.0):
                lam = B * case.h / vj
                sstar = np.array([RHO * profile_peak(float(l)) / v for l, v in zip(lam, vj)])
                match = []
                for j in range(m):
                    for ds in (-16, -8, -4, -2, 2, 4, 8, 16):
                        s = sstar[j] + ds
                        t = case.tj[j] + eps * s
                        if not (WINDOW[0] <= t <= WINDOW[1]) or abs(eps * s) > 0.25:
                            continue
                        rep = case.density(int(round(t / case.dr)), B)
                        slope = -vj[j] ** 2 * s / S2
                        match.append(dict(j=j + 1, s_star=float(sstar[j]), ds=ds, s=float(s),
                                          eps_D=eps * rep["D"], eps_seD=eps * rep["seD"],
                                          eps_b_j=eps * rep["b"][j], gauss_slope=float(slope)))
                gaps = []
                for j in range(m - 1):
                    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                        t = case.tj[j] + frac * (case.tj[j + 1] - case.tj[j])
                        rep = case.density(int(round(t / case.dr)), B)
                        gaps.append(dict(gap=j + 1, t=rep["t"], D=rep["D"], seD=rep["seD"],
                                         eps2_max_abs_b_retained=float(eps**2 * max(abs(rep["b"][j]), abs(rep["b"][j + 1]))),
                                         eps2_abs_ell=[float(eps**2 * abs(x)) for x in rep["ell"][j:j + 2]],
                                         A_retained=rep["A"][j:j + 2]))
                sign_ok = all(((r["eps_D"] < 0) == (r["ds"] > 0)) and abs(r["eps_D"]) > 5 * r["eps_seD"] for r in match)
                far = [abs(r["eps_b_j"]) for r in match if abs(r["ds"]) >= 8 and abs(r["s"]) >= 4]
                out.append(dict(m=m, eps=eps, B=B, lambda_j=lam.tolist(), s_star=sstar.tolist(),
                                n_paths=case.n_paths, dr=case.dr, seed_entropy=case.seed_entropy,
                                matching=match, gaps=gaps, matching_signs_ok=sign_ok,
                                max_abs_eps_b_j_far=(max(far) if far else None),
                                max_eps2_abs_b_gaps=(max(r["eps2_max_abs_b_retained"] for r in gaps) if gaps else None)))
                print(f"[b] m={m} eps={eps} B={B}: s*={np.round(sstar, 2)}  matching signs ok (5 SE)={sign_ok}  "
                      f"max|eps b_j| (|ds|>=8)={out[-1]['max_abs_eps_b_j_far']}  "
                      f"max eps^2|b| gaps={out[-1]['max_eps2_abs_b_gaps']}", flush=True)
    results["part_b_key_estimates"] = out


def part_c(results):
    """Extrema summary of GPT-6's earlier PDE runs (contact = 1)."""
    out = []
    for js in sorted(PDE_DIR.glob("reduced_m*_fine.json")) + sorted(PDE_DIR.glob("reduced_m*_stress*.json")):
        meta = json.loads(js.read_text())
        for s in meta["summaries"]:
            ext = s["extrema_I"]
            nmax = sum(e["kind"] == "max" for e in ext)
            nmin = sum(e["kind"] == "min" for e in ext)
            kinds = "".join("M" if e["kind"] == "max" else "m" for e in ext)
            out.append(dict(file=js.name, sha256=sha256(js), m=meta["m"], eps=meta["eps"], B=s["B"],
                            n_max=nmax, n_min=nmin, order=kinds,
                            endpoint_log_slopes=s.get("endpoint_log_slopes"),
                            dx=meta["dx"], dt=meta["dt"]))
    results["part_c_pde_extrema"] = out
    for r in out:
        print(f"[c] {r['file']} B={r['B']}: {r['order']} endpoints={r['endpoint_log_slopes']}", flush=True)


def part_d(results):
    """Convexity check on the PDE (contact = 1): f''/f = D' + D^2 > 0 wherever
    d(t) = min_j |mu(t) - mu_j| >= L eps (off the passage windows)."""
    out = []
    for js in sorted(PDE_DIR.glob("reduced_m*_fine.json")):
        meta = json.loads(js.read_text())
        npz = js.with_suffix(".npz")
        arr = np.load(npz)
        t = arr["t"]
        eps = meta["eps"]
        m = meta["m"]
        muk = mu(np.array(TARGET_TIMES[m]))
        inwin = (t >= WINDOW[0]) & (t <= WINDOW[1])
        d = np.min(np.abs(mu(t)[:, None] - muk[None, :]), axis=1)
        for bi, B in enumerate(arr["budgets"]):
            B = float(B)
            D = arr["log_derivative"][:, bi]
            Dp = np.gradient(D, t)
            fpp = Dp + D * D
            rows = {}
            for L in (1.0, 2.0, 3.0, 4.0, 6.0):
                sel = inwin & (d >= L * eps)
                sel[:2] = False
                sel[-2:] = False
                n = int(sel.sum())
                neg = int((fpp[sel] <= 0).sum())
                rows[str(L)] = dict(n_points=n, n_nonpositive=neg)
            Lmin = None
            for L in np.arange(0.5, 12.01, 0.25):
                sel = inwin & (d >= L * eps)
                sel[:2] = False
                sel[-2:] = False
                if sel.sum() and (fpp[sel] > 0).all():
                    Lmin = float(L)
                    break
            out.append(dict(file=js.name, sha256=sha256(js), npz_sha256=sha256(npz), m=m, eps=eps, B=B,
                            by_L=rows, min_L_all_convex=Lmin))
            print(f"[d] {js.name} B={B}: min L with f''>0 off windows = {Lmin}; "
                  f"L=3: {rows['3.0']}", flush=True)
    results["part_d_pde_convexity"] = out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = dict(
        script=str(HERE.relative_to(REPORT)),
        description=__doc__.strip().splitlines()[0],
        seed_base=SEED_BASE, stream_tag=STREAM_TAG,
        model=dict(gamma=GAMMA, D0=D0, zbar=ZBAR, z0=Z0, rho=RHO, W=W, d=DIM, target_times=TARGET_TIMES,
                   window=WINDOW, contact="identically 1 (midpoint-only killing)"),
    )
    parts = sys.argv[1:] or ["a", "b", "c", "d"]
    if "c" in parts:
        part_c(results)
    if "a" in parts:
        part_a(results)
    if "b" in parts:
        part_b(results)
    if "d" in parts:
        part_d(results)
    results["wall_seconds"] = time.time() - t0
    name = "th6_tilted_bridge_checks" + ("" if parts == ["a", "b", "c", "d"] else "_" + "".join(parts)) + ".json"
    (OUT_DIR / name).write_text(json.dumps(results, indent=1))
    print("wrote", OUT_DIR / name, f"{results['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
