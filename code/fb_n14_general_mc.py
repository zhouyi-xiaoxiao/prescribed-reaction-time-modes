#!/usr/bin/env python3
"""N14b -- universality campaign: Feynman--Kac Monte Carlo for GENERAL small-noise
transport crossing m thin stripes (tests Theorem th13:mass, Proposition th13:rate,
Theorem th13:profile, Corollary th13:peaks, Proposition th13:examples of
manuscript/cnsns_submission/theory/TH13_universality.tex).

Models (all gate-free; budget B, allocation w; eps = noise amplitude = stripe width scale):
  A  plug flow in a channel: dX = U dt + eps sqrt(2D) dW, X0 = eps xi0, bands at x_j,
     kappa_j = (B w_j / A)(eps rho)^-1 psi((x-x_j)/(eps rho));  psi in {gauss, logistic, tophat}
  B  anharmonic gradient relaxation: dZ = -(z + z^3) dt + eps sig(Z) dW, sig(z) = 1 + 0.3 sin(3z)
     (state-dependent noise), stripes at z_j in (1.2, 0.8, 0.5), z0 = 2;  psi in {gauss, logistic};
     allocations: equal and the max-min design of Corollary th13:design (costs c_j = A V'(z_j))
  C  planar spiral b(z) = -z + omega J z, curved stripes = circles |z| = r_j with angular weight
     eta_j = (1 + alpha cos(arg z - theta_j))/(2 pi r_j);  psi = gauss
  H  counterexample: plug flow, ONE band with the non-log-concave 'core+halo' profile
     (1-q) phi_{0.3} + q phi_{3}, q = 0.5, lambda = 8, small arrival spread (theta ~ 0.09):
     two peaks from one crossing (Remark th13:rem-counterex)

Estimators (no killing is sampled; every path carries its exposure):
  masses  M_j = mean(exp(-A(s_{j-1})) - exp(-A(s_j)))           (Feynman--Kac)
  density f(t_k) = mean(kappa(Z_k) exp(-A_k))                   (on the time grid)
  arrival offsets: mean/var of n_j.(Z_{t_j} - phi(t_j))/eps  vs  tau_j^2 (Lyapunov)
  passage exposures: mean/var of I_j = int_{window j} kappa_j(Z) dt  vs lambda_j
Euler--Maruyama, dt = eps*rho/(n_per_width * v_max); trapezoidal exposure.

Seeds: base 20260923, tag 97; numpy Philox with SeedSequence([20260923, 97, model_code, psi_code,
round(eps*1e6), chunk]).  All recorded in the output.

Subcommands:
  run --model A|B|C|H [--paths N]   -> artifacts/data/exact_m_fixed_budget/N14_universality/mc_<model>.json
  analyze                           -> .../N14_universality/n14_general_mc.json
  figure                            -> artifacts/figures/fb_n14_mc_rates.pdf, fb_n14_mc_profiles.pdf
(The sibling driver code/fb_n14_universality.py of the numerics pass covers plug flow with
Gaussian/top-hat bands, Poiseuille flow and a constant-noise anharmonic demo; this driver adds
state-dependent noise, curved stripes with an angular weight, the Lyapunov arrival-spread check
and the non-log-concave counterexample.)
Python: python3 (numpy, scipy, matplotlib)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "data", "exact_m_fixed_budget", "N14_universality")
FIG_DIR = os.path.join(ROOT, "artifacts", "figures")
SEED, TAG = 20260923, 97
SQ2PI = np.sqrt(2 * np.pi)


# ------------------------------------------------------------------ profiles
def _phi(u):
    return np.exp(-0.5 * u * u) / SQ2PI


def _Phi(u):
    return 0.5 * special.erfc(-u / np.sqrt(2.0))


_LS = np.sqrt(3.0) / np.pi
_TH = np.sqrt(3.0)
PROFILES = {
    "gauss": (_phi, _Phi),
    "logistic": (lambda u: 1.0 / (4 * _LS) / np.cosh(u / (2 * _LS)) ** 2,
                 lambda u: 0.5 * (1 + np.tanh(u / (2 * _LS)))),
    "tophat": (lambda u: np.where(np.abs(u) < _TH, 1 / (2 * _TH), 0.0),
               lambda u: np.clip((u + _TH) / (2 * _TH), 0.0, 1.0)),
    "corehalo": (lambda u: 0.5 * _phi(u / 0.3) / 0.3 + 0.5 * _phi(u / 3.0) / 3.0,
                 lambda u: 0.5 * _Phi(u / 0.3) + 0.5 * _Phi(u / 3.0)),
}
PSI_CODE = {"gauss": 1, "logistic": 2, "tophat": 3, "corehalo": 4}


# ------------------------------------------------------------------ models
def model_spec(name):
    if name == "A":
        return dict(name="A", n=1, U=1.0, D=0.5, s0=0.5, Aarea=1.0, xs=[1.0, 2.0, 3.0], rho=1.0, T=4.0,
                    B=3.0, profiles=["gauss", "logistic", "tophat"], eps=[0.2, 0.1, 0.05, 0.025, 0.0125],
                    code=1)
    if name == "B":
        return dict(name="B", n=1, z0=2.0, zs=[1.2, 0.8, 0.5], rho=1.0, T=1.2, s0=0.5, Aarea=1.0,
                    B=3.0, profiles=["gauss", "logistic"], eps=[0.2, 0.1, 0.05, 0.025, 0.0125], code=2)
    if name == "C":
        return dict(name="C", n=2, omega=2.0, r0=2.0, rs=[1.5, 1.0, 0.6], alpha=0.6, thetas=[0.0, 0.0, 0.0],
                    sig0=1.0, s0=1.0 / np.sqrt(2.0), rho=1.0, T=1.6, B=20.0, profiles=["gauss"],
                    eps=[0.2, 0.1, 0.05, 0.025, 0.0125], code=3)
    if name == "H":
        return dict(name="H", n=1, U=1.0, D=0.0005, s0=0.05, Aarea=1.0, xs=[5.0], rho=1.0, T=6.0,
                    B=8.0, profiles=["corehalo"], eps=[0.1, 0.05, 0.025], code=4)
    raise ValueError(name)


def geometry(spec):
    """Deterministic crossing data: t_j, p_j, v_j, sigma_j (orientation), eta_j(p_j), tau_j^2, cost c_j."""
    nm = spec["name"]
    if nm in ("A", "H"):
        U, D, s0, A = spec["U"], spec["D"], spec["s0"], spec["Aarea"]
        xs = np.array(spec["xs"])
        t = xs / U
        v = np.full_like(xs, U)
        orient = np.ones_like(xs)
        eta = np.full_like(xs, 1.0 / A)
        tau2 = s0 ** 2 + 2 * D * xs / U
    elif nm == "B":
        zs = np.array(spec["zs"])
        z0 = spec["z0"]
        Vp = lambda z: z + z ** 3
        sig = lambda z: 1.0 + 0.3 * np.sin(3 * z)
        from scipy import integrate
        t = np.array([integrate.quad(lambda z: 1 / Vp(z), zj, z0, epsabs=1e-13, epsrel=1e-13)[0] for zj in zs])
        v = Vp(zs)
        orient = -np.ones_like(zs)
        eta = np.full_like(zs, 1.0 / spec["Aarea"])
        tau2 = np.array([Vp(zj) ** 2 * (spec["s0"] ** 2 / Vp(z0) ** 2 +
                                        integrate.quad(lambda z: sig(z) ** 2 / Vp(z) ** 3, zj, z0,
                                                       epsabs=1e-13, epsrel=1e-13)[0]) for zj in zs])
    elif nm == "C":
        rs = np.array(spec["rs"])
        t = np.log(spec["r0"] / rs)
        v = rs.copy()
        orient = -np.ones_like(rs)
        ang = spec["omega"] * t
        eta = (1 + spec["alpha"] * np.cos(ang - np.array(spec["thetas"]))) / (2 * np.pi * rs)
        tau2 = np.exp(-2 * t) * spec["s0"] ** 2 + spec["sig0"] ** 2 * (1 - np.exp(-2 * t)) / 2
    cost = v / eta
    return dict(t=t, v=v, orient=orient, eta=eta, tau2=tau2, cost=cost)


def maxmin_design(B, cost):
    m = len(cost)
    g = lambda p: sum(cost[j] * np.log((1 - j * p) / (1 - (j + 1) * p)) for j in range(m)) - B
    from scipy import optimize
    pstar = optimize.brentq(g, 1e-14, 1.0 / m - 1e-14, xtol=1e-15)
    lam = np.array([np.log((1 - j * pstar) / (1 - (j + 1) * pstar)) for j in range(m)])
    w = cost * lam / B
    return float(pstar), w / w.sum()


def stick_breaking(lam):
    lam = np.asarray(lam, float)
    Pi = np.exp(-np.concatenate([[0.0], np.cumsum(lam)[:-1]]))
    return Pi * (1 - np.exp(-lam)), Pi


def limit_profile(psi_name, lam, theta, orient, s_grid, v, rho, Pi):
    """P_j(s) = Pi (v/rho) F(v s/rho), F = p^{psi^orient}_lam * phi_theta."""
    pdf, cdf = PROFILES[psi_name]
    h = 0.002
    u = np.arange(-60, 60 + h / 2, h)
    if orient > 0:
        ps, Ps = pdf(u), cdf(u)
    else:
        ps, Ps = pdf(-u), 1.0 - cdf(-u)
    p = lam * ps * np.exp(-lam * Ps)
    if theta > 0:
        half = int(np.ceil(10 * theta / h)) + 1
        xk = np.arange(-half, half + 1) * h
        k = np.exp(-0.5 * (xk / theta) ** 2)
        k /= k.sum()
        F = np.convolve(p, k, mode="same")
    else:
        F = p
    y = v * np.asarray(s_grid) / rho
    return Pi * (v / rho) * np.interp(y, u, F)


# ------------------------------------------------------------------ simulation
def simulate(spec, psi_name, eps, N, n_per_width=30, chunk=50000):
    geo = geometry(spec)
    nm = spec["name"]
    m = len(geo["t"])
    rho = spec["rho"]
    pdf, _ = PROFILES[psi_name]
    T = spec["T"]
    # resolve each stripe crossing: dt << eps*rho / (normal speed at the crossing)
    dt = eps * rho / (n_per_width * geo["v"].max())
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps
    tgrid = np.arange(nsteps + 1) * dt
    tj = geo["t"]
    cuts = np.concatenate([[0.0], 0.5 * (tj[:-1] + tj[1:]), [T]])
    cut_idx = np.clip(np.round(cuts / dt).astype(int), 0, nsteps)
    tj_idx = np.round(tj / dt).astype(int)
    # allocations
    allocs = {"equal": np.full(m, 1.0 / m)}
    if nm == "B":
        ps, wstar = maxmin_design(spec["B"], geo["cost"])
        allocs["maxmin"] = wstar
    if nm == "H":
        allocs = {"single": np.array([1.0])}
    Bv = spec["B"]
    # deterministic path phi on the grid
    if nm in ("A", "H"):
        phi_path = lambda tt: np.array([spec["U"] * tt])
    elif nm == "B":
        # RK4 on the grid
        zz = np.empty(nsteps + 1)
        zz[0] = spec["z0"]
        f = lambda z: -(z + z ** 3)
        for k in range(nsteps):
            z = zz[k]
            k1 = f(z); k2 = f(z + 0.5 * dt * k1); k3 = f(z + 0.5 * dt * k2); k4 = f(z + dt * k3)
            zz[k + 1] = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        phi_path = None
    elif nm == "C":
        om, r0 = spec["omega"], spec["r0"]
        phi_path = lambda tt: np.array([r0 * np.exp(-tt) * np.cos(om * tt), r0 * np.exp(-tt) * np.sin(om * tt)])

    def unit_rates(Z):
        """Unit-budget stripe rates X_j'(t) = kappa_j / (B w_j), shape (m, n_paths)."""
        out = np.empty((m, Z.shape[-1]))
        if nm in ("A", "H"):
            for j, xj in enumerate(spec["xs"]):
                out[j] = pdf((Z[0] - xj) / (eps * rho)) / (eps * rho) / spec["Aarea"]
        elif nm == "B":
            for j, zj in enumerate(spec["zs"]):
                out[j] = pdf((Z[0] - zj) / (eps * rho)) / (eps * rho) / spec["Aarea"]
        elif nm == "C":
            r = np.hypot(Z[0], Z[1])
            for j, rj in enumerate(spec["rs"]):
                ell = (r * r - rj * rj) / (2 * rj)
                cosang = (Z[0] * np.cos(spec["thetas"][j]) + Z[1] * np.sin(spec["thetas"][j])) / np.maximum(r, 1e-300)
                eta = (1 + spec["alpha"] * cosang) / (2 * np.pi * rj)
                out[j] = pdf(ell / (eps * rho)) / (eps * rho) * eta
        return out

    def drift(Z):
        if nm in ("A", "H"):
            return np.full_like(Z, spec["U"])
        if nm == "B":
            return -(Z + Z ** 3)
        om = spec["omega"]
        return np.stack([-Z[0] - om * Z[1], -Z[1] + om * Z[0]])

    def noise(Z, dW):
        if nm in ("A", "H"):
            return np.sqrt(2 * spec["D"]) * dW
        if nm == "B":
            return (1.0 + 0.3 * np.sin(3 * Z)) * dW
        return spec["sig0"] * dW

    n = spec["n"]
    acc = {k: dict(dens=np.zeros(nsteps + 1), dens2=np.zeros(nsteps + 1)) for k in allocs}
    Xcut_all = []  # per path unit exposures at cuts: (paths, m, m+1)
    off_sum = np.zeros(m); off_sq = np.zeros(m)
    nchunks = int(np.ceil(N / chunk))
    seeds = []
    for c in range(nchunks):
        size = min(chunk, N - c * chunk)
        ent = [SEED, TAG, spec["code"], PSI_CODE[psi_name], int(round(eps * 1e6)), c]
        seeds.append(ent)
        rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(ent)))
        if nm in ("A", "H"):
            Z = np.zeros((1, size)) + eps * spec["s0"] * rng.standard_normal((1, size))
        elif nm == "B":
            Z = np.full((1, size), spec["z0"]) + eps * spec["s0"] * rng.standard_normal((1, size))
        else:
            Z = np.array([[spec["r0"]], [0.0]]) + eps * spec["s0"] * rng.standard_normal((2, size))
        X = np.zeros((m, size))
        rate_prev = unit_rates(Z)
        Xcut = np.zeros((size, m, m + 1))
        ci = 1  # cut 0 at t=0 is zero exposure
        Aw = {k: np.zeros(size) for k in allocs}
        for k_, w in allocs.items():
            kap = Bv * (w[:, None] * rate_prev).sum(0)
            acc[k_]["dens"][0] += kap.sum()
            acc[k_]["dens2"][0] += (kap ** 2).sum()
        sq = np.sqrt(dt)
        for k in range(1, nsteps + 1):
            dW = sq * rng.standard_normal((n, size))
            Z = Z + drift(Z) * dt + eps * noise(Z, dW)
            rate = unit_rates(Z)
            X += 0.5 * dt * (rate + rate_prev)
            rate_prev = rate
            for k_, w in allocs.items():
                A_ = Bv * (w[:, None] * X).sum(0)
                kap = Bv * (w[:, None] * rate).sum(0)
                val = kap * np.exp(-A_)
                acc[k_]["dens"][k] += val.sum()
                acc[k_]["dens2"][k] += (val ** 2).sum()
            while ci <= m and cut_idx[ci] == k:
                Xcut[:, :, ci] = X.T
                ci += 1
            for j in np.nonzero(tj_idx == k)[0]:
                if nm == "B":
                    ph = np.array([zz[k]])
                    nvec = np.array([1.0])
                else:
                    ph = phi_path(tgrid[k])
                    nvec = ph / np.linalg.norm(ph) if nm == "C" else np.array([1.0])
                off = (nvec[:, None] * (Z - ph[:, None])).sum(0) / eps
                off_sum[j] += off.sum(); off_sq[j] += (off ** 2).sum()
        Xcut_all.append(Xcut)
    Xcut = np.concatenate(Xcut_all, 0)
    res = dict(model=nm, psi=psi_name, eps=eps, N=N, dt=dt, nsteps=nsteps, seeds=seeds,
               cuts=cuts.tolist(), t=tj.tolist(), v=geo["v"].tolist(), eta=geo["eta"].tolist(),
               tau2_lyapunov=geo["tau2"].tolist(), cost=geo["cost"].tolist(),
               offset_mean=(off_sum / N).tolist(), offset_var=(off_sq / N - (off_sum / N) ** 2).tolist(),
               allocs={})
    for k_, w in allocs.items():
        lam = Bv * w / geo["cost"]
        Mlim, Pi = stick_breaking(lam)
        A_cut = Bv * np.einsum("j,pjc->pc", w, Xcut)
        E = np.exp(-A_cut)
        Mi = E[:, :-1] - E[:, 1:]
        M = Mi.mean(0)
        Mse = Mi.std(0, ddof=1) / np.sqrt(N)
        # passage exposures I_j = B w_j (X_j(s_j) - X_j(s_{j-1}))
        Ij = np.stack([Bv * w[j] * (Xcut[:, j, j + 1] - Xcut[:, j, j]) for j in range(m)], 1)
        dens = acc[k_]["dens"] / N
        dens_se = np.sqrt(np.maximum(acc[k_]["dens2"] / N - dens ** 2, 0) / N)
        # local profiles on |s| <= L
        L = 6.0 if nm != "H" else 14.0
        if m > 1:  # keep the window inside the passage's own basin (no overlap with neighbours)
            L = min(L, 0.45 * float(np.min(np.diff(tj))) / eps)
        prof = []
        for j in range(m):
            sel = np.nonzero(np.abs(tgrid - tj[j]) <= eps * L)[0]
            s = (tgrid[sel] - tj[j]) / eps
            theta = np.sqrt(geo["tau2"][j]) / spec["rho"]
            P = limit_profile(psi_name, lam[j], theta, geo["orient"][j], s, geo["v"][j], spec["rho"], Pi[j])
            ef = eps * dens[sel]
            efse = eps * dens_se[sel]
            prof.append(dict(L=float(L), sup_err=float(np.max(np.abs(ef - P))), peak_limit=float(P.max()),
                             peak_mc=float(ef.max()), s_peak_limit=float(s[np.argmax(P)]),
                             s_peak_mc=float(s[np.argmax(ef)]), max_se=float(efse.max()),
                             s=s[:: max(1, len(s) // 400)].tolist(), ef=ef[:: max(1, len(s) // 400)].tolist(),
                             P=P[:: max(1, len(s) // 400)].tolist()))
        # peak count of the MC density (smoothed over eps/10 in time; prominence > 5 SE)
        dens_s = smooth(dens, max(1, int(round(0.1 * eps / dt))))
        peaks = count_peaks(tgrid, dens_s, dens_se, eps)
        res["allocs"][k_] = dict(w=w.tolist(), lam=lam.tolist(), M_lim=Mlim.tolist(), M_mc=M.tolist(),
                                 M_se=Mse.tolist(), M_err=(M - Mlim).tolist(),
                                 total_mc=float(1 - E[:, -1].mean()), total_lim=float(1 - np.exp(-lam.sum())),
                                 I_mean_minus_lam=(Ij.mean(0) - lam).tolist(), I_var=Ij.var(0, ddof=1).tolist(),
                                 I_se=(Ij.std(0, ddof=1) / np.sqrt(N)).tolist(),
                                 profiles=prof, peaks=peaks)
    return res


def smooth(y, k):
    if k <= 1:
        return y.copy()
    ker = np.ones(k) / k
    return np.convolve(y, ker, mode="same")


def count_peaks(t, y, se, eps):
    """Local maxima of the smoothed density with prominence > 5 * local SE (and > 1e-3 max)."""
    idx = np.nonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    out = []
    ymax = y.max()
    for i in idx:
        # prominence against the minima within 3 eps (time) on either side
        lmin = y[max(0, i - int(3 / (t[1] - t[0]) * eps)):i].min() if i > 0 else y[i]
        rmin = y[i + 1:i + 1 + int(3 / (t[1] - t[0]) * eps)].min() if i + 1 < len(y) else y[i]
        prom = y[i] - max(lmin, rmin)
        if prom > 5 * max(se[i], 1e-12) and y[i] > 1e-3 * ymax:
            out.append(dict(t=float(t[i]), height=float(y[i]), prominence=float(prom), se=float(se[i])))
    return dict(num=len(out), list=out[:10])


def cmd_run(args):
    spec = model_spec(args.model)
    if args.eps:
        spec["eps"] = [float(x) for x in args.eps.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = dict(script="code/fb_n14_general_mc.py", model=args.model, spec=spec, seed=SEED, tag=TAG,
               rng="numpy Philox; SeedSequence([20260923, 97, model_code, psi_code, round(eps*1e6), chunk])",
               runs=[])
    t0 = time.time()
    for psi in spec["profiles"]:
        if args.psi and psi != args.psi:
            continue
        for eps in spec["eps"]:
            N = args.paths if eps >= 0.025 else args.paths_small
            t1 = time.time()
            r = simulate(spec, psi, eps, N)
            r["runtime_s"] = time.time() - t1
            out["runs"].append(r)
            a = next(iter(r["allocs"].values()))
            print(f"{args.model} {psi} eps={eps} N={N} steps={r['nsteps']} "
                  f"Merr={np.round(a['M_err'], 5)} se={np.round(a['M_se'], 5)} "
                  f"prof_sup={[round(p['sup_err'], 4) for p in a['profiles']]} peaks={a['peaks']['num']} "
                  f"({r['runtime_s']:.0f}s)", flush=True)
            with open(os.path.join(OUT_DIR, f"mc_{args.model}_{psi}.json"), "w") as fh:
                json.dump(out, fh)
    out["runtime_s"] = time.time() - t0  # per-(model, psi) files were written incrementally above


def psi_l2sq(name):
    pdf, _ = PROFILES[name]
    u = np.linspace(-60, 60, 600001)
    return float(np.trapezoid(pdf(u) ** 2, u))


def noise_normal_var(spec, j):
    """a_j = n_j^T a(p_j) n_j."""
    nm = spec["name"]
    if nm in ("A", "H"):
        return 2 * spec["D"]
    if nm == "B":
        return (1.0 + 0.3 * np.sin(3 * spec["zs"][j])) ** 2
    return spec["sig0"] ** 2


def jensen_var_pred(spec, psi, eps, lam, v, rho):
    """Prop th13:jensen(i): V_j = eps beta_j^2 a_j ||psi||^2 / (rho v_j^3), beta_j = lam_j v_j."""
    l2 = psi_l2sq(psi)
    return [float(eps * (lam[j] * v[j]) ** 2 * noise_normal_var(spec, j) * l2 / (rho * v[j] ** 3))
            for j in range(len(lam))]


def fit_slope(eps, err):
    eps = np.asarray(eps); err = np.abs(np.asarray(err))
    ok = err > 0
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(eps[ok]), np.log(err[ok]), 1)[0])


def cmd_analyze(args):
    summary = dict(script="code/fb_n14_general_mc.py analyze", models={})
    import glob
    for mdl in ["A", "B", "C", "H"]:
        files = sorted(f for f in glob.glob(os.path.join(OUT_DIR, f"mc_{mdl}_*.json")) if not f.endswith("_all.json"))
        if not files:
            continue
        runs, seen = [], set()
        for p in files:
            for r_ in json.load(open(p))["runs"]:
                if (r_["psi"], r_["eps"]) not in seen:  # a multi-profile run file also holds earlier profiles
                    seen.add((r_["psi"], r_["eps"]))
                    runs.append(r_)
        summary.setdefault("files", []).extend([os.path.relpath(f, ROOT) for f in files])
        ms = {}
        for r in runs:
            for an, a in r["allocs"].items():
                key = f"{r['psi']}/{an}"
                e = ms.setdefault(key, dict(eps=[], N=[], M_lim=a["M_lim"], lam=a["lam"], w=a["w"],
                                            max_abs_M_err=[], max_M_se=[], total_err=[],
                                            prof_sup_err=[], prof_max_se=[], peaks=[],
                                            I_mean_minus_lam=[], I_var=[], offset_var=[],
                                            tau2=r["tau2_lyapunov"], s_peak_limit=[], s_peak_mc=[]))
                e["eps"].append(r["eps"]); e["N"].append(r["N"])
                e["max_abs_M_err"].append(float(np.max(np.abs(a["M_err"]))))
                e["max_M_se"].append(float(np.max(a["M_se"])))
                e["total_err"].append(a["total_mc"] - a["total_lim"])
                e["prof_sup_err"].append([p_["sup_err"] for p_ in a["profiles"]])
                e["prof_max_se"].append([p_["max_se"] for p_ in a["profiles"]])
                e["s_peak_limit"].append([p_["s_peak_limit"] for p_ in a["profiles"]])
                e["s_peak_mc"].append([p_["s_peak_mc"] for p_ in a["profiles"]])
                e["peaks"].append(a["peaks"]["num"])
                e["I_mean_minus_lam"].append(a["I_mean_minus_lam"])
                e["I_var"].append(a["I_var"])
                spec_ = model_spec(mdl)
                vp = jensen_var_pred(spec_, r["psi"], r["eps"], a["lam"], r["v"], spec_["rho"])
                e.setdefault("I_var_pred", []).append(vp)
                e.setdefault("I_var_ratio", []).append([float(x / y) if y > 0 else None
                                                        for x, y in zip(a["I_var"], vp)])
                # first-order (Jensen) law with the measured passage-exposure means and variances
                lam_ = np.array(a["lam"]); mu_ = np.array(a["I_mean_minus_lam"]); V_ = np.array(a["I_var"])
                Lc = np.concatenate([[0.0], np.cumsum(lam_)])
                Sc = np.exp(-Lc) * (1 - np.concatenate([[0.0], np.cumsum(mu_)]) + 0.5 * np.concatenate([[0.0], np.cumsum(V_)]))
                M1 = Sc[:-1] - Sc[1:]
                e.setdefault("max_abs_M_err_first_order", []).append(float(np.max(np.abs(np.array(a["M_mc"]) - M1))))
                # fully predicted Jensen law: V_j from Prop th13:jensen(i), mu_j = 0
                if an == "maxmin":
                    ps_, _ = maxmin_design(spec_["B"], np.array(r["cost"]))
                    e["design_pstar"] = ps_
                    e.setdefault("max_abs_M_minus_pstar", []).append(float(np.max(np.abs(np.array(a["M_mc"]) - ps_))))
                if mdl == "H":
                    e.setdefault("peaks_s_height", []).append(
                        [[(p_["t"] - r["t"][0]) / r["eps"], p_["height"] * r["eps"], p_["prominence"] / p_["se"]]
                         for p_ in a["peaks"]["list"]])
                    P_ = np.array(a["profiles"][0]["P"]); s_ = np.array(a["profiles"][0]["s"])
                    e.setdefault("limit_maxima_s_height", []).append(
                        [[float(s_[i]), float(P_[i])] for i in range(1, len(P_) - 1) if P_[i] > P_[i - 1] and P_[i] >= P_[i + 1]])
                Sp = np.exp(-Lc) * (1 + 0.5 * np.concatenate([[0.0], np.cumsum(vp)]))
                Mp = Sp[:-1] - Sp[1:]
                e.setdefault("max_abs_M_err_jensen_pred", []).append(float(np.max(np.abs(np.array(a["M_mc"]) - Mp))))
                e["offset_var"].append(r["offset_var"])
        for key, e in ms.items():
            e["slope_max_abs_M_err"] = fit_slope(e["eps"], e["max_abs_M_err"])
            e["slope_prof_sup_err"] = fit_slope(e["eps"], [max(x) for x in e["prof_sup_err"]])
            e["slope_I_var"] = fit_slope(e["eps"], [max(x) for x in e["I_var"]])
            e["max_rel_offset_var_err_smallest_eps"] = float(np.max(np.abs(np.array(e["offset_var"][-1]) /
                                                                           np.array(e["tau2"]) - 1)))
        summary["models"][mdl] = ms
    with open(os.path.join(OUT_DIR, "n14_general_mc.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for mdl, ms in summary["models"].items():
        for key, e in ms.items():
            print(mdl, key, "jensen_pred_resid", np.round(e["max_abs_M_err_jensen_pred"], 6))
            print(mdl, key, "eps", e["eps"], "maxMerr", np.round(e["max_abs_M_err"], 5), "se",
                  np.round(e["max_M_se"], 5), "slope", e["slope_max_abs_M_err"], "prof", e["slope_prof_sup_err"],
                  "Ivar", e["slope_I_var"], "peaks", e["peaks"], "offvar", round(e["max_rel_offset_var_err_smallest_eps"], 4))


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    S = json.load(open(os.path.join(OUT_DIR, "n14_general_mc.json")))
    # (1) mass error vs eps
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    mk = dict(A="o", B="s", C="^")
    for mdl, ms in S["models"].items():
        if mdl == "H":
            continue
        for key, e in ms.items():
            ln, = ax[0].loglog(e["eps"], e["max_abs_M_err"], marker=mk[mdl], lw=1, label=f"{mdl}: {key}")
            ax[0].loglog(e["eps"], e["max_abs_M_err_jensen_pred"], marker=mk[mdl], lw=0.8, ls=":", mfc="none",
                         color=ln.get_color())
            ax[1].loglog(e["eps"], [max(x) for x in e["prof_sup_err"]], marker=mk[mdl], lw=1, label=f"{mdl}: {key}")
    xx = np.array([0.0125, 0.2])
    for a_ in ax:
        a_.loglog(xx, 0.05 * xx / 0.2, "k--", lw=0.8, label=r"slope 1")
        a_.set_xlabel(r"noise $\varepsilon$")
    ax[0].set_ylabel(r"$\max_j|M_j-M_j^{\rm lim}|$ (MC)")
    ax[1].set_ylabel(r"$\max_j\sup_{|s|\leq 6}|\varepsilon f(t_j+\varepsilon s)-\mathcal{P}_j(s)|$")
    ax[0].legend(fontsize=6, ncol=1)
    ax[0].set_title("mass law (filled) and Jensen-corrected law (open, dotted)", fontsize=9)
    ax[1].set_title("local profile, general models", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fb_n14_mc_rates.pdf"))
    plt.close(fig)
    # (2) profiles: model B gauss equal, model C gauss equal, model H corehalo at the smallest eps
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a_, (mdl, key, title) in zip(ax, [("B", "gauss", "B: anharmonic, state-dependent noise"),
                                          ("C", "gauss", "C: planar spiral, curved stripes"),
                                          ("H", "corehalo", "H: one core+halo stripe")]):
        p = os.path.join(OUT_DIR, f"mc_{mdl}_{key}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        runs = [r for r in d["runs"] if r["psi"] == key]
        r = runs[-1]
        a = next(iter(r["allocs"].values()))
        for j, pr in enumerate(a["profiles"]):
            a_.plot(pr["s"], pr["ef"], lw=1.0, label=f"MC, passage {j+1}")
            a_.plot(pr["s"], pr["P"], "k--", lw=0.8, label="limit $\\mathcal{P}_j$" if j == 0 else None)
        a_.set_xlabel(r"$s=(t-t_j)/\varepsilon$")
        a_.set_ylabel(r"$\varepsilon f(t_j+\varepsilon s)$")
        a_.set_title(f"{title}, $\\varepsilon$={r['eps']}", fontsize=8)
        a_.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fb_n14_mc_profiles.pdf"))
    plt.close(fig)
    print("figures written")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--paths", type=int, default=200000)
    r.add_argument("--paths-small", type=int, default=100000)
    r.add_argument("--psi", default=None)
    r.add_argument("--eps", default=None, help="comma-separated override of the eps ladder")
    sub.add_parser("analyze")
    sub.add_parser("figure")
    args = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
