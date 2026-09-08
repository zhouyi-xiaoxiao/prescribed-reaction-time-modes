#!/usr/bin/env python3
"""Numerical companion to b0_dyson_sharpening_draft.md.

Verifies Lemmas 1-3 numerically, evaluates the matched-initial-law margin
inventory, and evaluates the fixed-allocation sufficient budget B_cert at the
anchor and at two secondary points.  The reported extrema are high-precision
floating-point evaluations, not directed-rounding interval enclosures.

Run:  python3 b0_dyson_numerics.py
"""
import math
import random
from mpmath import mp, mpf, mpc, sqrt, exp, log, log10, erf, quad, diff, findroot, pi as MPPI

mp.dps = 40

# ---------------------------------------------------------------- model data
class Case:
    def __init__(self, name, eps, targets, weights):
        self.name = name
        self.eps = mpf(eps)
        self.targets = [mpf(t) for t in targets]
        self.w = [mpf(w) for w in weights]
        # shared anchor geometry
        self.gamma = mpf(1); self.D0 = mpf(1)
        self.z0 = mpf(4); self.zbar = mpf(0)
        self.W = mpf(1); self.a = mpf('0.4'); self.rho = mpf(1)
        self.ell0 = mpf(1)
        self.tau = mpf('0.5'); self.T = mpf('3.5')
        self.d = 2
        # Match the off-lattice campaign's relative-coordinate initial law:
        # u0 = 0.3, r_parallel,0 = 0.1, and transverse s.d. coefficient 0.3.
        self.u02 = mpf('0.09')
        self.rpar0 = mpf('0.1')
        self.Sperp0 = mpf('0.09')
        self.Sstar2 = self.D0/(2*self.gamma) + self.rho**2   # = 3/2
        self.v0 = self.D0/(2*self.gamma)                     # midpoint variance param

    def mu(self, t):
        return self.zbar + (self.z0 - self.zbar)*exp(-self.gamma*t)

    def centres(self):
        return [self.mu(tj) for tj in self.targets]

# ------------------------------------------------------- free-clock G(t)
def Phi(x):
    return (1 + erf(x/sqrt(2)))/2

def P_perp(c, s, W, kmax=6):
    """P(|wrapped N(0,s^2)|_mi < c), torus width W, 0<c<W/2."""
    tot = mpf(0)
    for k in range(-kmax, kmax+1):
        tot += Phi((k*W + c)/s) - Phi((k*W - c)/s)
    return tot

def contact(case, t):
    """c_{d,eps}(t) = P(R_par^2 + R_perp,mi^2 < a^2), d=2."""
    eps, g, D0 = case.eps, case.gamma, case.D0
    vpar = case.u02*exp(-2*g*t) + (2*D0/g)*(1 - exp(-2*g*t))   # relative long. variance/eps^2
    spar = eps*sqrt(vpar)
    mpar = case.rpar0*exp(-g*t)
    sperp = eps*sqrt(case.Sperp0 + 4*D0*t)
    a = case.a
    def integrand(x):
        return exp(-(x - mpar)**2/(2*spar**2))/(sqrt(2*MPPI)*spar) * P_perp(sqrt(a**2 - x**2), sperp, case.W)
    return quad(integrand, [-a, 0, a])

def gz(case, t):
    """Longitudinal (midpoint x slab) factor of G: E[phi_w(Z_t)] / W^{d-1}."""
    eps = case.eps
    pref = 1/(case.W**(case.d-1) * sqrt(2*MPPI) * eps * sqrt(case.Sstar2))
    mt = case.mu(t)
    s = mpf(0)
    for wj, cj in zip(case.w, case.centres()):
        s += wj*exp(-(mt - cj)**2/(2*eps**2*case.Sstar2))
    return pref*s

def G(case, t):
    return contact(case, t)*gz(case, t)

# --------------------------------------------- fast evaluation machinery
class Cheb:
    """Chebyshev interpolant (Gauss nodes) with derivative series."""
    def __init__(self, f, a, b, N=64):
        self.a, self.b, self.N = mpf(a), mpf(b), N
        xk = [mp.cos(MPPI*(mpf(k) + mpf('0.5'))/N) for k in range(N)]
        fk = [f((self.a + self.b)/2 + (self.b - self.a)/2*x) for x in xk]
        self.c = []
        for j in range(N):
            s = sum(fk[k]*mp.cos(j*MPPI*(mpf(k) + mpf('0.5'))/N) for k in range(N))
            self.c.append(s*2/N)
        self.c[0] /= 2
        self.dc = self._derive(self.c)
        self.ddc = self._derive(self.dc)

    def _derive(self, c):
        N = len(c)
        d = [mpf(0)]*(N + 2)
        for k in range(N - 1, 0, -1):
            d[k - 1] = d[k + 1] + 2*k*c[k]
        d[0] /= 2
        scale = 2/(self.b - self.a)
        return [x*scale for x in d[:N]]

    def _clenshaw(self, c, t):
        x = (2*t - self.a - self.b)/(self.b - self.a)
        b1 = b2 = mpf(0)
        for ck in reversed(c[1:]):
            b1, b2 = 2*x*b1 - b2 + ck, b1
        return x*b1 - b2 + c[0]

    def val(self, t):  return self._clenshaw(self.c, t)
    def d1(self, t):   return self._clenshaw(self.dc, t)
    def d2(self, t):   return self._clenshaw(self.ddc, t)

_cheb_cache = {}
def contact_cheb(case, N=64):
    key = (case.name, N, mp.dps)
    if key not in _cheb_cache:
        pad = mpf('0.02')
        _cheb_cache[key] = Cheb(lambda t: contact(case, t), case.tau - pad, case.T + pad, N)
    return _cheb_cache[key]

def gz_derivs(case, t):
    """(gz, gz', gz'') by exact differentiation of the Gaussian mixture."""
    eps = case.eps
    g = case.gamma
    s2 = eps**2*case.Sstar2
    pref = 1/(case.W**(case.d-1) * sqrt(2*MPPI*s2))
    mt = case.mu(t)
    mup = -g*(mt - case.zbar)
    mupp = g**2*(mt - case.zbar)
    f0 = f1 = f2 = mpf(0)
    for wj, cj in zip(case.w, case.centres()):
        dj = mt - cj
        e = wj*exp(-dj**2/(2*s2))
        f0 += e
        f1 += e*(-dj*mup/s2)
        f2 += e*((dj*mup/s2)**2 - (mup**2 + dj*mupp)/s2)
    return pref*f0, pref*f1, pref*f2

def G_fast(case, t):
    ch = contact_cheb(case)
    return ch.val(t)*gz_derivs(case, t)[0]

def dG(case, t, n=1):
    """G^{(n)} via product rule: Chebyshev contact x analytic Gaussian part."""
    ch = contact_cheb(case)
    c0, c1, c2 = ch.val(t), ch.d1(t), ch.d2(t)
    z0, z1, z2 = gz_derivs(case, t)
    if n == 0: return c0*z0
    if n == 1: return c0*z1 + c1*z0
    if n == 2: return c0*z2 + 2*c1*z1 + c2*z0
    raise ValueError(n)

def dG_direct(case, t, n=1):
    return diff(lambda u: G(case, u), t, n)

# ------------------------------------------------------- lemma verifications
def verify_lemma1(ntrials=10000, seed=1):
    """|N_C(x;m,s2)| <= kappa * g_sig2(x-mR) * exp(P_lam mI^2)."""
    random.seed(seed)
    worst = mpf(0)
    for _ in range(ntrials):
        th = mpf(random.uniform(-1.4, 1.4))       # arg s2, well inside (-pi/2,pi/2)
        r2 = mpf(10)**random.uniform(-6, 2)       # |s2|
        s2 = r2*exp(mpc(0, 1)*th)
        lam = mpf(random.uniform(0.02, 0.9))
        mR = mpf(random.uniform(-5, 5)); mI = mpf(random.uniform(-5, 5))
        x = mpf(random.uniform(-20, 20))
        w = 1/(2*s2); aa = w.real; bb = -w.imag if False else w.imag
        # N_C
        NC = exp(-(x - mpc(mR, mI))**2*w)/sqrt(2*MPPI*s2)
        lhs = abs(NC)
        sig2 = 1/(2*aa*(1 - lam))
        gval = exp(-(x - mR)**2/(2*sig2))/sqrt(2*MPPI*sig2)
        kappa = 1/sqrt((1 - lam)*mp.cos(th))
        P = aa + bb**2/(lam*aa)
        rhs = kappa*gval*exp(P*mI**2)
        if rhs > 0:
            worst = max(worst, lhs/rhs)
    return worst

def verify_lemma2(ntrials=4000, seed=2):
    """Z-block: P_lam*(Im E)^2 <= (1+tan^2 th/lam) * g^2 h b^2/(2 eps^2 D0 u); also arg bound."""
    random.seed(seed)
    g = mpf(1); D0 = mpf(1); eps = mpf('0.1'); lam = mpf('0.1')
    worst_pen, worst_arg = mpf(0), mpf(0)
    for _ in range(ntrials):
        u = mpf(random.uniform(0.25, 4.0)); beta = mpf(random.uniform(-0.05, 0.05))
        h = mpf(10)**random.uniform(-6, 0)
        z = mpc(u, beta); zeta = z*h
        E = exp(-g*zeta)
        v = (D0/(2*g))*(1 - exp(-2*g*zeta))
        assert v.real > 0
        th = mp.atan2(v.imag, v.real)
        worst_arg = max(worst_arg, abs(mp.tan(th))/(abs(beta)/u)) if beta != 0 else worst_arg
        s2 = eps**2*v; w = 1/(2*s2); aa = w.real; bb = w.imag
        P = aa + bb**2/(lam*aa)
        pen = P*(E.imag)**2
        tt = abs(beta)/u
        bound = (1 + tt**2/lam)*g**2*h*beta**2/(2*eps**2*D0*u)
        if bound > 0:
            worst_pen = max(worst_pen, pen/bound)
    return worst_pen, worst_arg

def verify_lemma3(seed=3):
    """Complete-square identities, numeric spot checks."""
    random.seed(seed)
    out = []
    for _ in range(200):
        eps = mpf('0.1'); rho = mpf(random.uniform(0.5, 2))
        sig = mpf(random.uniform(0, 20)); c = mpf(random.uniform(-4, 4)); zb = mpf(0)
        x = 2*eps**2*rho**2*sig
        if x >= 1: continue
        y = mpf(random.uniform(-6, 6))
        lhs = exp(-(y - c)**2/(2*eps**2*rho**2))/sqrt(2*MPPI*eps**2*rho**2)*exp(sig*(y - zb)**2)
        csig = (c - x*zb)/(1 - x)
        rhs = (1 - x)**mpf('-0.5')*exp(sig*(c - zb)**2/(1 - x)) \
              * exp(-(y - csig)**2/(2*eps**2*rho**2/(1 - x)))/sqrt(2*MPPI*eps**2*rho**2/(1 - x))
        out.append(abs(lhs/rhs - 1))
    # initial-law integral identity
    eps = mpf('0.1'); v0 = mpf('0.5'); sig = mpf(3); z0 = mpf(4); zb = mpf(0)
    lhs = quad(lambda y: exp(-(y - z0)**2/(2*eps**2*v0))/sqrt(2*MPPI*eps**2*v0)*exp(sig*(y - zb)**2),
               [z0 - 2, z0 + 2])
    rhs = (1 - 2*eps**2*v0*sig)**mpf('-0.5')*exp(sig*(z0 - zb)**2/(1 - 2*eps**2*v0*sig))
    out.append(abs(lhs/rhs - 1))
    return max(out)

# ------------------------------------------------------- mixture sup / v_inf
def mixture_sup(case, width2=None, shift=1):
    """sup_z sum_j w_j N(z - shift*chat_j; width2), grid + refinement."""
    eps = case.eps
    if width2 is None:
        width2 = eps**2*case.rho**2
    cs = [shift*c for c in case.centres()]
    lo = min(cs) - 6*sqrt(width2); hi = max(cs) + 6*sqrt(width2)
    def f(zv):
        return sum(wj*exp(-(zv - cj)**2/(2*width2)) for wj, cj in zip(case.w, cs))/sqrt(2*MPPI*width2)
    n = 4000
    best, zbest = mpf(0), lo
    for i in range(n + 1):
        zv = lo + (hi - lo)*i/n
        val = f(zv)
        if val > best: best, zbest = val, zv
    # local refine (golden-ish trisection)
    aL, bR = zbest - (hi - lo)/n, zbest + (hi - lo)/n
    for _ in range(80):
        m1 = aL + (bR - aL)/3; m2 = bR - (bR - aL)/3
        if f(m1) > f(m2): bR = m2
        else: aL = m1
    return f((aL + bR)/2)

# ------------------------------------------------------- margin inventory
def bisect(f, a, b, iters=200):
    fa, fb = f(a), f(b)
    assert fa*fb < 0
    for _ in range(iters):
        m = (a + b)/2
        fm = f(m)
        if fm == 0: return m
        if fa*fm < 0: b, fb = m, fm
        else: a, fa = m, fm
    return (a + b)/2

def find_roots(case, npts=700):
    """Roots of G' on (tau,T) by sign-change bisection on a grid."""
    tau, T = case.tau, case.T
    ts = [tau + (T - tau)*i/npts for i in range(npts + 1)]
    d1 = [dG(case, t, 1) for t in ts]
    roots = []
    for i in range(npts):
        if d1[i] == 0: continue
        if d1[i]*d1[i+1] < 0:
            roots.append(bisect(lambda u: dG(case, u, 1), ts[i], ts[i+1]))
    return roots, ts, d1

def tube_edges(case, root, half_curv):
    """Find interval around root where |G''| >= half_curv, same sign as at root."""
    sgn = 1 if dG(case, root, 2) > 0 else -1
    def h(u):
        return sgn*dG(case, u, 2) - half_curv
    # march outward
    step = mpf('0.01')
    lo = root
    while lo - step > case.tau and h(lo - step) > 0:
        lo -= step
    left = bisect(h, lo, lo - step) if lo - step > case.tau else case.tau
    hi = root
    while hi + step < case.T and h(hi + step) > 0:
        hi += step
    right = bisect(h, hi, hi + step) if hi + step < case.T else case.T
    return left, right

def margin_inventory(case, npts=700, verbose=True):
    roots, ts, d1 = find_roots(case, npts)
    curv = [dG(case, r, 2) for r in roots]
    if verbose:
        for r, c in zip(roots, curv):
            print(f"  root t*={mp.nstr(r,8)}  G={mp.nstr(G(case,r),6)}  G''={mp.nstr(c,6)}")
    half = min(abs(c) for c in curv)/2
    mu2 = half
    tubes = [tube_edges(case, r, abs(c)/2) for r, c in zip(roots, curv)]
    # complement scan for mu1 using the precomputed grid derivative values
    mu1, argmin = mpf('inf'), None
    for t, v in zip(ts, d1):
        inside = any(L < t < R for (L, R) in tubes)
        if not inside and abs(v) < mu1:
            mu1, argmin = abs(v), t
    # refine at endpoints explicitly
    for t in (case.tau, case.T):
        v = abs(dG(case, t, 1))
        if v < mu1: mu1, argmin = v, t
    if verbose:
        print("  tubes:", [(mp.nstr(L,6), mp.nstr(R,6)) for (L, R) in tubes])
        print(f"  mu2 = {mp.nstr(mu2,6)}   mu1 = {mp.nstr(mu1,6)} at t = {mp.nstr(argmin,6)}")
    return roots, curv, tubes, mu1, mu2

# ------------------------------------------------------- new-threshold assembly
def assemble(case, mu1, mu2, lam=mpf('0.1'), verbose=True):
    eps, g, D0, tau, T = case.eps, case.gamma, case.D0, case.tau, case.T
    cs = case.centres()
    yhat = max(abs(case.z0 - case.zbar), max(abs(c - case.zbar) for c in cs))
    r0 = (eps/(g*yhat))*sqrt(D0*tau/2)
    assert r0 <= tau/2, "R1 fails"
    t_theta = r0/(tau - r0)
    khat = (1 - lam)**(-1) * (1 + t_theta**2)**mpf(str((case.d + 1)/4.0))
    sigZ = (1 + t_theta**2/lam)*g**2*r0**2/(2*eps**2*D0*(tau - r0))
    sigP = sigZ/4
    x_V = 2*eps**2*case.rho**2*sigZ
    x_Z0 = 2*eps**2*case.v0*sigZ
    x_par0 = 2*eps**2*case.u02*sigP
    x_max = max(x_V, x_Z0, x_par0)
    assert x_max < mpf('0.5'), "R2 fails"
    rhat = max(case.a, abs(case.rpar0))
    Pi = (sigZ*yhat**2 + sigP*rhat**2)/(1 - x_max) \
         + mpf('0.5')*(x_V + x_Z0 + x_par0)/(1 - x_max)
    # v_inf and width-inflation slack delta_v
    vsup_plain = mixture_sup(case)
    vsup_infl = mixture_sup(case, width2=eps**2*case.rho**2/(1 - x_V), shift=1/(1 - x_V))
    v_inf = vsup_plain/case.W**(case.d - 1)
    delta_v = max(mpf(0), vsup_infl/vsup_plain - 1)
    v_eff = khat*(1 + delta_v)*v_inf
    C_pre = khat*v_inf*exp(Pi)
    Mhat = min(r0*mu1, r0**2*mu2/2)
    Bcert = log(1 + Mhat/C_pre)/(v_eff*(T + r0))
    if verbose:
        print(f"  yhat={mp.nstr(yhat,6)}  r0={mp.nstr(r0,6)}  t_theta={mp.nstr(t_theta,6)}")
        print(f"  khat={mp.nstr(khat,8)}  sigZ={mp.nstr(sigZ,6)}  sigP={mp.nstr(sigP,6)}")
        print(f"  x_V={mp.nstr(x_V,4)}  x_Z0={mp.nstr(x_Z0,4)}  x_par0={mp.nstr(x_par0,4)}  x_max={mp.nstr(x_max,4)}")
        print(f"  Pi={mp.nstr(Pi,8)}  e^Pi={mp.nstr(exp(Pi),8)}")
        print(f"  v_inf={mp.nstr(v_inf,8)}  delta_v={mp.nstr(delta_v,4)}  v_eff={mp.nstr(v_eff,8)}")
        print(f"  C_pre={mp.nstr(C_pre,8)}")
        print(f"  r0*mu1={mp.nstr(r0*mu1,6)}  r0^2*mu2/2={mp.nstr(r0**2*mu2/2,6)}  Mhat={mp.nstr(Mhat,6)}")
        print(f"  B_cert={mp.nstr(Bcert,8)}   log10={mp.nstr(log10(Bcert),8)}")
    return dict(yhat=yhat, r0=r0, khat=khat, Pi=Pi, v_inf=v_inf, delta_v=delta_v,
                v_eff=v_eff, C_pre=C_pre, Mhat=Mhat, Bcert=Bcert)

# ------------------------------------------------------------------- main
if __name__ == '__main__':
    print("=== Lemma verifications ===")
    w1 = verify_lemma1()
    print(f"Lemma 1 worst |LHS/RHS| over 1e4 samples: {mp.nstr(w1, 8)}  (must be <= 1)")
    wp, wa = verify_lemma2()
    print(f"Lemma 2 worst penalty/bound: {mp.nstr(wp,8)}  worst tan(arg)/(beta/u): {mp.nstr(wa,8)}  (both <= 1)")
    w3 = verify_lemma3()
    print(f"Lemma 3 worst identity mismatch: {mp.nstr(w3, 4)}  (must be ~1e-30)")

    print("\n=== Matched-law anchor (eps=0.1, m=2) ===")
    A = Case('anchor', '0.1', ['1.0', '2.5'], ['0.5', '0.5'])
    ctau = contact(A, A.tau)
    print(f"c(tau) = {mp.nstr(ctau, 8)}")
    cT = contact(A, A.T)
    print(f"c(T)   = {mp.nstr(cT, 8)}")
    btau = diff(lambda u: log(contact(A, u)), A.tau)
    print(f"b(tau) = dlog c/dt at tau = {mp.nstr(btau, 8)}")
    g1 = dG(A, A.tau, 1)
    print(f"G'(tau) = {mp.nstr(g1, 8)}")
    g1T = dG(A, A.T, 1)
    print(f"|G'(T)| = {mp.nstr(abs(g1T), 8)}")
    print("matched-law floating-point margin inventory:")
    roots, curv, tubes, mu1_own, mu2_own = margin_inventory(A, npts=700)
    resA = assemble(A, mu1_own, mu2_own)

    print("\n=== Matched-law secondary A (eps=0.05, m=2) ===")
    mp.dps = 60
    B = Case('eps005', '0.05', ['1.0', '2.5'], ['0.5', '0.5'])
    print(f"c(tau) = {mp.nstr(contact(B, B.tau), 8)}")
    fastB, dirB = dG(B, B.tau, 1), dG_direct(B, B.tau, 1)
    print(f"G'(tau) = {mp.nstr(fastB, 8)}  (direct check rel err {mp.nstr(abs(fastB-dirB)/abs(dirB),3)})")
    rootsB, curvB, tubesB, mu1B, mu2B = margin_inventory(B, npts=700)
    resB = assemble(B, mu1B, mu2B)

    print("\n=== Matched-law secondary B (eps=0.1, m=3, targets 0.8/1.6/2.8) ===")
    mp.dps = 40
    C = Case('m3', '0.1', ['0.8', '1.6', '2.8'], [mpf(1)/3, mpf(1)/3, mpf(1)/3])
    print(f"centres: {[mp.nstr(c,6) for c in C.centres()]}")
    fastC, dirC = dG(C, C.tau, 1), dG_direct(C, C.tau, 1)
    print(f"G'(tau) = {mp.nstr(fastC, 8)}  (direct check rel err {mp.nstr(abs(fastC-dirC)/abs(dirC),3)})")
    rootsC, curvC, tubesC, mu1C, mu2C = margin_inventory(C, npts=900)
    resC = assemble(C, mu1C, mu2C)

    print("\n=== Exponent constants ===")
    for case, res in ((A, resA), (B, resB), (C, resC)):
        DL = case.mu(case.tau) - case.mu(case.targets[0])
        cB = DL**2/(2*case.Sstar2)
        print(f"{case.name}: D_L={mp.nstr(DL,6)}  c_B(cert)=D_L^2/(2 S*^2)={mp.nstr(cB,6)}")
