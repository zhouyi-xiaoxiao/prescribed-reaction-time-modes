#!/usr/bin/env python3
"""Independent spot-check of Proposition 1 for n = 1 and n = 2 (Z-block).

Computes the exact flat chain integral for the Z block (the block carrying the
whole e^{1600} danger) by vectorized quadrature at complex time, and compares
against the assembled per-block bound

    |C_n^Z| <= kappa_Z^{n+1} * (sup phi_w)^{n+1-1} * ...

i.e. for the Z block alone the Proposition-1 bookkeeping gives

    |C_n^Z| <= kappa_Z^{n+1} * (sup phi_w)^n * (sup phi_w at obs node) * e^{Pi_Z}
             = kappa_Z^{n+1} * v_phi^{n+1} * e^{Pi_Z},    v_phi := sup_z phi_w(z).

Run:  python3 b0_dyson_chaincheck.py
"""
import numpy as np

# anchor data
eps, g, D0, z0, zbar = 0.1, 1.0, 1.0, 4.0, 0.0
rho, tau, T = 1.0, 0.5, 3.5
targets = [1.0, 2.5]
wts = [0.5, 0.5]
chat = [zbar + (z0 - zbar)*np.exp(-g*t) for t in targets]
v0 = D0/(2*g)
lam = 0.1

def phiw(y):
    s = 0
    for wj, cj in zip(wts, chat):
        s = s + wj*np.exp(-(y - cj)**2/(2*eps**2*rho**2))/np.sqrt(2*np.pi*eps**2*rho**2)
    return s

def KZ(x, y, zeta):
    """Complex-time Z-block Mehler kernel; x column, y row (broadcast)."""
    E = np.exp(-g*zeta)
    v = (D0/(2*g))*(1 - np.exp(-2*g*zeta))
    m = zbar + (y - zbar)*E
    s2 = eps**2*v
    return np.exp(-(x - m)**2/(2*s2))/np.sqrt(2*np.pi*s2)

def q0z(y):
    return np.exp(-(y - z0)**2/(2*eps**2*v0))/np.sqrt(2*np.pi*eps**2*v0)

def grid_union(centres, half, n):
    gs, ws = [], []
    for c in centres:
        x = np.linspace(c - half, c + half, n)
        gs.append(x); ws.append(np.full(n, x[1] - x[0]))
    return np.concatenate(gs), np.concatenate(ws)

def chain(z, fracs, npts=400, half=8*eps):
    """|C_n^Z| for leg fractions fracs (len n+1, sum 1): q0 -> nodes -> obs."""
    n = len(fracs) - 1
    x0, w0 = grid_union([z0], half, npts)
    vec = q0z(x0)*w0                       # weights on x0
    cur_pts, cur_vec = x0, vec
    for j in range(n + 1):
        zeta = z*fracs[-(j + 1)]           # legs applied right-to-left: h_n first
        xn, wn = grid_union(chat, half, npts)
        Kmat = KZ(xn[:, None], cur_pts[None, :], zeta)
        newvec = Kmat @ cur_vec
        if j < n:
            cur_vec = phiw(xn)*wn*newvec   # interior node
        else:
            cur_vec = phiw(xn)*newvec      # observable node (no dx weight: final pairing value)
            return np.sum(cur_vec*wn)
        cur_pts = xn

def assembled_bound(n, r0):
    t_theta = r0/(tau - r0)
    sigZ = (1 + t_theta**2/lam)*g**2*r0**2/(2*eps**2*D0*(tau - r0))
    xmax = 2*eps**2*rho**2*sigZ
    x_v0 = 2*eps**2*v0*sigZ
    yhat = 4.0
    PiZ = sigZ*yhat**2/(1 - xmax) + 0.5*xmax/(1 - xmax) + 0.5*x_v0/(1 - x_v0)
    kZ = ((1 - lam)*np.cos(np.arctan(t_theta)))**-0.5
    v_phi = phiw(np.array(chat)).max()
    return kZ**(n + 1)*v_phi**(n + 1)*np.exp(PiZ)

if __name__ == '__main__':
    r0 = (eps/(g*4.0))*np.sqrt(D0*tau/2)
    worst = 0.0
    for n, fracs_list in ((1, [[0.5, 0.5], [0.01, 0.99], [0.99, 0.01], [0.2, 0.8]]),
                          (2, [[1/3, 1/3, 1/3], [0.01, 0.01, 0.98], [0.8, 0.1, 0.1]])):
        bound = assembled_bound(n, r0)
        print(f"n={n}: assembled Z-block bound = {bound:.6g}")
        for t in (tau, 2.0, T):
            for ang in (0.5, 1.0, 1.5):
                z = t + r0*np.exp(1j*np.pi*ang)
                for fr in fracs_list:
                    val = abs(chain(z, fr))
                    ratio = val/bound
                    worst = max(worst, ratio)
                    print(f"  t={t:.2f} ang={ang:.1f}pi fr={fr}: |C{n}|={val:.6g} ratio={ratio:.4g}")
    print(f"WORST ratio = {worst:.6g}  (must be <= 1)")
