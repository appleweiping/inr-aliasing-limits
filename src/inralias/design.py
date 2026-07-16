r"""AliasGuard: constructive, data-independent sampling design against structured aliasing.

Given a fixed model frequency set :math:`\Lambda` (size :math:`m`), a finite candidate
out-of-band set :math:`\Omega_{\rm out}` (size :math:`K`) the practitioner wants to keep
*visible*, and a sample budget :math:`N`, AliasGuard chooses the sample locations (a
subset of an allowed super-grid, or continuous times in :math:`[0,1)^d`) to jointly

* keep the in-band synthesis matrix :math:`\Phi_\Lambda` well conditioned (stable
  recovery / small noise gain, Lemma~1), and
* drive the :math:`\Lambda\leftrightarrow\Omega_{\rm out}` cross-coherence toward zero,
  which simultaneously *raises* the worst-case visibility :math:`\min_\nu v_T(\nu)` and
  *lowers* the worst-case aliasability :math:`\max_\nu a_T(\nu)` (T1 objects).

Frequencies and times may be scalar (1-D) or ``d``-dimensional vectors (e.g. images,
``d=2``); everything below is written dimension-agnostically.

Why this is not standard experimental design
--------------------------------------------
D-/E-optimal and coherence-minimizing sensing designs optimize a **single, symmetric**
Gram criterion (minimize the coherence of, or maximize :math:`\lambda_{\min}` of, one
dictionary's Gram).  AliasGuard optimizes an **asymmetric block target** on the Gram of
the *augmented* dictionary :math:`[\Phi_\Lambda\ \Psi_{\Omega_{\rm out}}]`:

.. math::  \tfrac1N\Phi_\Lambda^{*}\Phi_\Lambda \to I_m
   \quad(\text{conditioning}),\qquad
   \tfrac1N\Phi_\Lambda^{*}\Psi_{\Omega_{\rm out}} \to 0
   \quad(\text{visibility / anti-aliasing}),

and is *indifferent* to the :math:`\Omega_{\rm out}` self-block.  Protecting the
:math:`\Lambda` conditioning while suppressing coupling to a **specified** candidate set
is the increment: it is exactly the quantity the T3 concentration bound controls in
expectation, made a design target rather than a property of random draws.

Objective and surrogate
------------------------
The exact multi-objective
:math:`(\max\min_\nu v_T(\nu),\ \min\max_\nu a_T(\nu),\ \min\kappa(\Phi_\Lambda))` is
non-smooth and (for :math:`a_T`, via :math:`\Phi_\Lambda^{\dagger}`) not submodular, so we
claim **no** greedy approximation ratio.  We minimize the smooth surrogate

.. math::  J(T)=\underbrace{\tfrac1{N^2}\!\sum_{k,\nu}|(\Phi_\Lambda^{*}\Psi)_{k\nu}|^2}
   _{E_{\Lambda\Omega}}
   +\beta\underbrace{\tfrac1{N^2}\!\sum_{k\ne l}|(\Phi_\Lambda^{*}\Phi_\Lambda)_{kl}|^2}
   _{E_{\Lambda\Lambda}},

both functions of the difference-frequency sums :math:`s(\delta)=\sum_j e^{i2\pi\langle
\delta,t_j\rangle}`.  :math:`J` upper-bounds the cross-coherence
:math:`\mu(T)=\max_{k,\nu}|(\Phi_\Lambda^{*}\Psi)_{k\nu}|/N` and hence, via the T3
perturbation step (supplement Prop.~A), the worst-case aliasability.  Minimizing
:math:`J` attains a :math:`\mu` no larger than the draw it is seeded from, and empirically
far smaller (see ``experiments/run_aliasguard.py``).

Deterministic given initialization; uses only :math:`\Lambda`, :math:`\Omega_{\rm out}`,
N -- never the signal, noise, or the true out-of-band frequencies at test time.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "design_metrics",
    "cross_coherence",
    "surrogate_objective",
    "visibility_of",
    "aliasability_of",
    "aliasguard_greedy",
    "aliasguard_continuous",
    "aliasguard_continuous_nd",
    "condition_only_design",
    "coherence_only_design",
    "e_optimal_design",
    "fixed_jitter",
    "aliasability_certificate",
    "visibility_certificate",
]


# --------------------------------------------------------------------------------------
# dimension-agnostic primitives
# --------------------------------------------------------------------------------------
def _as2d(x):
    """Return (array, is_scalar) with array shape (n, d); scalar input -> (n, 1)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        return x.reshape(-1, 1), True
    return x, False


def _synth(freqs, t):
    r""":math:`\Phi_{jk}=e^{i2\pi\langle\omega_k,t_j\rangle}`, freqs (m,d), t (N,d)."""
    F, _ = _as2d(freqs)
    T, _ = _as2d(t)
    return np.exp(1j * 2 * np.pi * (T @ F.T))


def _diff_freqs(freqs, out_freqs, beta):
    """Frequency gaps in J with weights (cross gaps weight 1, in-band off-diag weight beta).

    Returns (deltas (D,d), weights (D,))."""
    F, _ = _as2d(freqs)
    O, _ = _as2d(out_freqs)
    m = F.shape[0]
    cross = (O[None, :, :] - F[:, None, :]).reshape(-1, F.shape[1])       # nu - omega_k
    D = F[None, :, :] - F[:, None, :]                                     # omega_l - omega_k
    iu = ~np.eye(m, dtype=bool)
    inband = D[iu]
    deltas = np.concatenate([cross, inband], axis=0)
    weights = np.concatenate([np.ones(cross.shape[0]), beta * np.ones(inband.shape[0])])
    return deltas, weights


def visibility_of(freqs, t, nu) -> float:
    r"""Visibility :math:`v_T(\nu)=\lVert(I-P_\Lambda)\phi_\nu\rVert/\sqrt N\in[0,1]`."""
    T, _ = _as2d(t)
    Phi = _synth(freqs, t)
    phi = _synth(np.atleast_2d(nu), t)[:, 0]
    U, s, _ = np.linalg.svd(Phi, full_matrices=False)
    tol = max(Phi.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > max(tol, 1e-10 * (s[0] if s.size else 1.0))))
    Ur = U[:, :r]
    resid = phi - Ur @ (Ur.conj().T @ phi)
    return float(np.linalg.norm(resid) / np.sqrt(T.shape[0]))


def aliasability_of(freqs, t, nu) -> float:
    r"""Aliasability :math:`a_T(\nu)=\lVert\Phi_\Lambda^{\dagger}\phi_\nu\rVert_2`."""
    Phi = _synth(freqs, t)
    phi = _synth(np.atleast_2d(nu), t)[:, 0]
    c, *_ = np.linalg.lstsq(Phi, phi, rcond=None)
    return float(np.linalg.norm(c))


def condition_number_of(freqs, t) -> float:
    s = np.linalg.svd(_synth(freqs, t), compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else np.inf


def cross_coherence(freqs, out_freqs, t) -> float:
    r""":math:`\mu(T)=\max_{k,\nu}|(\Phi_\Lambda^{*}\Psi)_{k\nu}|/N`."""
    T, _ = _as2d(t)
    C = _synth(freqs, t).conj().T @ _synth(out_freqs, t) / T.shape[0]
    return float(np.max(np.abs(C))) if C.size else 0.0


def surrogate_objective(freqs, out_freqs, t, beta: float = 1.0) -> float:
    T, _ = _as2d(t)
    deltas, w = _diff_freqs(freqs, out_freqs, beta)
    S = np.exp(1j * 2 * np.pi * (deltas @ T.T)).sum(axis=1)               # (D,)
    return float(np.sum(w * np.abs(S) ** 2) / T.shape[0] ** 2)


def design_metrics(freqs, out_freqs, t) -> dict:
    r"""min_visibility (higher better), max_aliasability (lower better), cross_coherence,
    condition_number of :math:`\Phi_\Lambda`, surrogate J."""
    O, _ = _as2d(out_freqs)
    vs = [visibility_of(freqs, t, O[i]) for i in range(O.shape[0])]
    az = [aliasability_of(freqs, t, O[i]) for i in range(O.shape[0])]
    T, _ = _as2d(t)
    return {
        "min_visibility": float(np.min(vs)) if vs else 1.0,
        "mean_visibility": float(np.mean(vs)) if vs else 1.0,
        "max_aliasability": float(np.max(az)) if az else 0.0,
        "mean_aliasability": float(np.mean(az)) if az else 0.0,
        "cross_coherence": cross_coherence(freqs, out_freqs, t),
        "condition_number": condition_number_of(freqs, t),
        "surrogate_J": surrogate_objective(freqs, out_freqs, t),
        "N": int(T.shape[0]),
    }


# --------------------------------------------------------------------------------------
# AliasGuard designs
# --------------------------------------------------------------------------------------
def aliasguard_greedy(freqs, out_freqs, N, super_grid=None, Q=512, beta=1.0,
                      n_swaps=40, seed=0, init="farthest"):
    r"""Discrete AliasGuard on an allowed super-grid (1-D): greedy min-J insertion + local
    swaps via rank-one updates of the difference-frequency sums.  Deterministic given
    ``(super_grid, seed, init)``.  Grid subset -> inherits T1: cannot break exact grid folds."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    if super_grid is None:
        super_grid = np.arange(Q) / Q
    super_grid = np.asarray(super_grid, float)
    P = super_grid.size
    deltas, w = _diff_freqs(freqs.reshape(-1, 1), out_freqs.reshape(-1, 1), beta)
    E = np.exp(1j * 2 * np.pi * np.outer(super_grid, deltas[:, 0]))       # (P, D)

    rng = np.random.default_rng(seed)
    if init == "random":
        chosen = list(rng.choice(P, size=min(2, N), replace=False))
    else:
        chosen = [0]
        if N >= 2:
            d = np.abs(((super_grid - super_grid[0] + 0.5) % 1.0) - 0.5)
            chosen.append(int(np.argmax(d)))
    chosen = list(dict.fromkeys(chosen))
    S = E[chosen].sum(axis=0)
    while len(chosen) < N:
        n1 = len(chosen) + 1
        rem = np.setdiff1d(np.arange(P), chosen)
        base = float(np.sum(w * np.abs(S) ** 2))
        cross = 2.0 * np.real(E[rem] @ (w * np.conj(S)))
        Js = (base + cross + float(np.sum(w))) / n1**2
        best = int(rem[int(np.argmin(Js))])
        chosen.append(best)
        S = S + E[best]
    N2 = N**2
    curJ = float(np.sum(w * np.abs(S) ** 2) / N2)
    for _ in range(n_swaps):
        improved = False
        for ci in range(len(chosen)):
            B = S - E[chosen[ci]]
            rem = np.setdiff1d(np.arange(P), chosen[:ci] + chosen[ci + 1:])
            base = float(np.sum(w * np.abs(B) ** 2))
            Js = (base + 2.0 * np.real(E[rem] @ (w * np.conj(B))) + float(np.sum(w))) / N2
            j = int(np.argmin(Js))
            if Js[j] < curJ - 1e-13:
                S = B + E[int(rem[j])]
                chosen[ci] = int(rem[j])
                curJ = float(Js[j])
                improved = True
        if not improved:
            break
    t = np.sort(super_grid[chosen])
    return t, design_metrics(freqs, out_freqs, t)


def _coord_descent(freqs, out_freqs, N, objective, d=1, n_sweeps=15, grid_res=720,
                   seed=0, t0=None):
    """Generic coordinate descent over continuous times in [0,1)^d minimizing ``objective``
    (a callable of the current sample set).  1-D uses a fast surrogate scan; d>1 uses a
    per-axis scan.  Deterministic given (seed, t0)."""
    rng = np.random.default_rng(seed)
    if t0 is None:
        t = rng.uniform(0, 1, (N, d))
    else:
        t = np.array(t0, float).reshape(N, d).copy()
    scan = np.arange(grid_res) / grid_res
    for _ in range(n_sweeps):
        moved = 0.0
        for j in range(N):
            for ax in range(d):
                best_val, best_o = t[j, ax], np.inf
                for gi in range(grid_res):
                    cand = t.copy()
                    cand[j, ax] = scan[gi]
                    o = objective(cand)
                    if o < best_o:
                        best_o, best_val = o, scan[gi]
                moved += abs(best_val - t[j, ax])
                t[j, ax] = best_val
        if moved < 1e-6:
            break
    return t


def aliasguard_continuous(freqs, out_freqs, N, beta=1.0, n_sweeps=15, grid_res=720,
                          seed=0, t0=None):
    r"""Continuous AliasGuard (1-D): coordinate descent of :math:`J` over times in
    :math:`[0,1)`; each coordinate set to its 1-D minimizer on a ``grid_res`` scan.
    Fast rank-one surrogate updates.  Deterministic given ``(seed, t0)``."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    deltas, w = _diff_freqs(freqs.reshape(-1, 1), out_freqs.reshape(-1, 1), beta)
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, N)) if t0 is None else np.asarray(t0, float).reshape(-1).copy()
    scan = np.arange(grid_res) / grid_res
    Escan = np.exp(1j * 2 * np.pi * np.outer(scan, deltas[:, 0]))         # (G, D)
    S = np.exp(1j * 2 * np.pi * np.outer(deltas[:, 0], t)).sum(axis=1)
    N2 = N**2
    for _ in range(n_sweeps):
        moved = 0.0
        for j in range(N):
            ej = np.exp(1j * 2 * np.pi * deltas[:, 0] * t[j])
            B = S - ej
            base = float(np.sum(w * np.abs(B) ** 2))
            Js = (base + 2.0 * np.real(Escan @ (w * np.conj(B))) + float(np.sum(w))) / N2
            gi = int(np.argmin(Js))
            moved += abs(scan[gi] - t[j])
            S = B + Escan[gi]
            t[j] = scan[gi]
        if moved < 1e-6:
            break
    t = np.sort(t)
    return t, design_metrics(freqs, out_freqs, t)


def aliasguard_continuous_nd(freqs, out_freqs, N, d, beta=1.0, n_sweeps=12, grid_res=64,
                             seed=0, t0=None):
    r"""Continuous AliasGuard in ``d`` dimensions (e.g. 2-D frequency vectors for images).
    Per-axis coordinate descent on the exact surrogate.  Returns ``(t (N,d), metrics)``."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    obj = lambda cand: surrogate_objective(freqs, out_freqs, cand, beta)
    t = _coord_descent(freqs, out_freqs, N, obj, d=d, n_sweeps=n_sweeps,
                       grid_res=grid_res, seed=seed, t0=t0)
    return t, design_metrics(freqs, out_freqs, t)


# --------------------------------------------------------------------------------------
# baselines (all data-independent: use only Lambda / Omega / N)
# --------------------------------------------------------------------------------------
def fixed_jitter(N, d=1):
    r"""Deterministic low-discrepancy design: van der Corput (base 2) in 1-D, Halton
    (bases 2,3) in 2-D.  Data-independent and reproducible (no seed)."""
    def vdc(n, base):
        out = np.zeros(n)
        for i in range(n):
            f, b, x = 1.0, base, 0.0
            k = i + 1
            while k > 0:
                f /= b
                x += f * (k % b)
                k //= b
            out[i] = x
        return out
    if d == 1:
        return np.sort(vdc(N, 2))
    cols = [vdc(N, b) for b in (2, 3, 5, 7)[:d]]
    return np.stack(cols, axis=1)


def condition_only_design(freqs, N, n_sweeps=12, grid_res=512, seed=0):
    r"""Coordinate descent minimizing only :math:`\kappa(\Phi_\Lambda)` (a
    conditioning-only / E-optimal-style baseline that ignores :math:`\Omega_{\rm out}`)."""
    freqs = np.asarray(freqs, float)
    obj = lambda cand: condition_number_of(freqs, cand)
    t = _coord_descent(freqs, freqs[:1], N, obj, d=1, n_sweeps=n_sweeps,
                       grid_res=grid_res, seed=seed)
    return np.sort(t.reshape(-1))


def coherence_only_design(freqs, out_freqs, N, n_sweeps=12, grid_res=512, seed=0):
    r"""Coordinate descent minimizing only the cross-coherence (AliasGuard *without* the
    conditioning term; the ablation isolating the joint objective's value)."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    obj = lambda cand: cross_coherence(freqs, out_freqs, cand)
    t = _coord_descent(freqs, out_freqs, N, obj, d=1, n_sweeps=n_sweeps,
                       grid_res=grid_res, seed=seed)
    return np.sort(t.reshape(-1))


# --------------------------------------------------------------------------------------
# continuum certificate (1-D): worst-case over an entire band, not a finite candidate set
# --------------------------------------------------------------------------------------
def _trig_lipschitz_sup(M, dt, band, n_grid):
    r"""Certified supremum of :math:`g(\nu)=\sum_{j,l}M_{jl}e^{i2\pi\nu(t_l-t_j)}` (real) over
    a union of intervals ``band=[(a1,b1),...]``.

    ``g`` is :math:`L`-Lipschitz with :math:`L=2\pi\sum_{j,l}|M_{jl}||t_l-t_j|`, so
    :math:`\sup_{[a,b]}g \le \max_{\text{grid}}g + L\,h/2` for grid spacing ``h``.  Returns
    ``(certified_sup, grid_max, L)``.  Deterministic and sample-only (``M`` depends only on
    the sample times)."""
    L = float(2 * np.pi * np.sum(np.abs(M) * np.abs(dt)))
    dtf = dt.ravel()
    Mf = M.ravel()
    grid_max, slack = -np.inf, 0.0
    for (a, b) in band:
        n = max(2, int(np.ceil((b - a) * n_grid)))
        nu = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
        # g(nu) = Re sum_{j,l} M_{jl} exp(i2pi nu (t_l - t_j)); dt = t_l - t_j
        for lo in range(0, n, 4096):                              # chunk rows to bound memory
            blk = nu[lo:lo + 4096]
            g = np.real(np.exp(1j * 2 * np.pi * np.outer(blk, dtf)) @ Mf)
            grid_max = max(grid_max, float(np.max(g)))
        slack = max(slack, L * h / 2.0)
    return grid_max + slack, grid_max, L


def aliasability_certificate(freqs, t, band, n_grid=2000):
    r"""Certified upper bound on :math:`\max_{\nu\in\text{band}} a_T(\nu)` over a *continuous*
    out-of-band region (a union of intervals), not a finite candidate set.

    ``n_grid`` is the evaluation grid density in points \emph{per unit frequency}; the
    Lipschitz slack is :math:`\approx L/(2\,n_{\rm grid})` (typically :math:`L\sim3`--$5$,
    so a few $10^3$/unit gives slack :math:`<10^{-3}`).  The bound is sound at any density.

    Uses :math:`a_T(\nu)^2=\phi_\nu^{*}M\phi_\nu`, :math:`M=\Phi(\Phi^{*}\Phi)^{-2}\Phi^{*}`,
    a real trigonometric polynomial in :math:`\nu` with frequencies :math:`t_l-t_j`;
    a Bernstein/Lipschitz grid bound certifies the supremum (sample-only, deterministic).
    Returns ``{"certified_max_aliasability", "grid_max_aliasability", "lipschitz"}``.
    """
    t = np.asarray(t, float).reshape(-1)
    Phi = _synth(np.asarray(freqs, float), t)
    G = Phi.conj().T @ Phi
    Gi2 = np.linalg.matrix_power(np.linalg.inv(G), 2)
    M = Phi @ Gi2 @ Phi.conj().T                                  # (N,N) Hermitian PSD
    dt = t[None, :] - t[:, None]                                  # t_l - t_j
    sup_g, grid_g, L = _trig_lipschitz_sup(M, dt, band, n_grid)
    return {"certified_max_aliasability": float(np.sqrt(max(sup_g, 0.0))),
            "grid_max_aliasability": float(np.sqrt(max(grid_g, 0.0))),
            "lipschitz": L}


def visibility_certificate(freqs, t, band, n_grid=4000):
    r"""Certified *lower* bound on :math:`\min_{\nu\in\text{band}} v_T(\nu)` over a continuous
    band.  Uses :math:`v_T(\nu)^2=1-\phi_\nu^{*}P\phi_\nu/N`,
    :math:`P=\Phi(\Phi^{*}\Phi)^{-1}\Phi^{*}`; certifies :math:`\max_\nu\phi_\nu^{*}P\phi_\nu`
    by the same Lipschitz argument.  Returns
    ``{"certified_min_visibility", "grid_min_visibility"}``.
    """
    t = np.asarray(t, float).reshape(-1)
    N = t.size
    Phi = _synth(np.asarray(freqs, float), t)
    P = Phi @ np.linalg.inv(Phi.conj().T @ Phi) @ Phi.conj().T
    dt = t[None, :] - t[:, None]
    sup_h, grid_h, _L = _trig_lipschitz_sup(P, dt, band, n_grid)
    return {"certified_min_visibility": float(np.sqrt(max(1.0 - sup_h / N, 0.0))),
            "grid_min_visibility": float(np.sqrt(max(1.0 - grid_h / N, 0.0)))}


def e_optimal_design(freqs, N, n_restarts=200, seed=0):
    r"""E-optimal random-search baseline: among ``n_restarts`` i.i.d. uniform designs,
    return the one maximizing :math:`\lambda_{\min}(\Phi_\Lambda^{*}\Phi_\Lambda/N)`
    (classical optimal-design criterion; uses only :math:`\Lambda`, N)."""
    freqs = np.asarray(freqs, float)
    rng = np.random.default_rng(seed)
    best_t, best_lam = None, -np.inf
    for _ in range(n_restarts):
        t = np.sort(rng.uniform(0, 1, N))
        Phi = _synth(freqs, t)
        lam = float(np.linalg.eigvalsh(Phi.conj().T @ Phi / N)[0])
        if lam > best_lam:
            best_lam, best_t = lam, t
    return best_t
