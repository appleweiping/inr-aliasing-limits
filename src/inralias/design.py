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
    "surrogate_hessian",
    "aliasguard_frankwolfe",
    "design_measure_round",
    "rounding_gap_bound",
    "visibility_of",
    "aliasability_of",
    "aliasability_L2_of",
    "aliasguard_greedy",
    "aliasguard_continuous",
    "aliasguard_continuous_nd",
    "condition_only_design",
    "coherence_only_design",
    "e_optimal_design",
    "fixed_jitter",
    "aliasability_certificate",
    "aliasability_certificate_nd",
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
    r"""Design/limit metrics.  ``max_aliasability`` is the coefficient-norm worst case
    :math:`\max_\nu\at(\nu)`; ``max_aliasability_L2`` is the **function-space** worst case
    :math:`\max_\nu a_{L^2,T}(\nu)` (equal only for orthonormal dictionaries)."""
    O, _ = _as2d(out_freqs)
    vs = [visibility_of(freqs, t, O[i]) for i in range(O.shape[0])]
    az = [aliasability_of(freqs, t, O[i]) for i in range(O.shape[0])]
    azl2 = [aliasability_L2_of(freqs, t, O[i]) for i in range(O.shape[0])]
    T, _ = _as2d(t)
    return {
        "min_visibility": float(np.min(vs)) if vs else 1.0,
        "mean_visibility": float(np.mean(vs)) if vs else 1.0,
        "max_aliasability": float(np.max(az)) if az else 0.0,
        "mean_aliasability": float(np.mean(az)) if az else 0.0,
        "max_aliasability_L2": float(np.max(azl2)) if azl2 else 0.0,
        "mean_aliasability_L2": float(np.mean(azl2)) if azl2 else 0.0,
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
    r"""Continuous AliasGuard in ``d`` dimensions (e.g. 2-D/3-D frequency vectors for images).

    Fast rank-one **per-axis** update (the n-D generalization of :func:`aliasguard_continuous`):
    holding the other axes of sample $j$ fixed, the surrogate as a function of $t_{j,\rm ax}$ is
    a single moving-sum scan, using the factorization $e^{i2\pi\langle\delta,t_j\rangle}=
    e^{i2\pi\delta_{\rm ax}t_{j,\rm ax}}\,c_{j,\delta}$ with the cross-axis factor
    $c_{j,\delta}=e^{i2\pi\langle\delta_{-\rm ax},t_{j,-\rm ax}\rangle}$.  ~$N\times$ faster than
    recomputing the full surrogate per grid point, so ``grid_res`` can be raised for statistical
    separation in the n-D design study.  Returns ``(t (N,d), metrics)``."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    F, _ = _as2d(freqs)
    O, _ = _as2d(out_freqs)
    deltas, wts = _diff_freqs(F, O, beta)                       # (D,d), (D,)
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, (N, d)) if t0 is None else np.asarray(t0, float).reshape(N, d).copy()
    scan = np.arange(grid_res) / grid_res
    Escan = [np.exp(1j * 2 * np.pi * np.outer(scan, deltas[:, ax])) for ax in range(d)]  # ax:(G,D)
    N2, sw = N * N, float(np.sum(wts))
    S = np.exp(1j * 2 * np.pi * (deltas @ t.T)).sum(axis=1)     # (D,)
    for _ in range(n_sweeps):
        moved = 0.0
        for j in range(N):
            for ax in range(d):
                other = [a for a in range(d) if a != ax]
                cj = (np.exp(1j * 2 * np.pi * (deltas[:, other] @ t[j, other]))
                      if other else np.ones(deltas.shape[0], complex))
                B = S - np.exp(1j * 2 * np.pi * deltas[:, ax] * t[j, ax]) * cj
                base = float(np.sum(wts * np.abs(B) ** 2))
                Js = (base + 2.0 * np.real(Escan[ax] @ (wts * np.conj(B) * cj)) + sw) / N2
                gi = int(np.argmin(Js))
                moved += abs(scan[gi] - t[j, ax])
                S = B + np.exp(1j * 2 * np.pi * deltas[:, ax] * scan[gi]) * cj
                t[j, ax] = scan[gi]
        if moved < 1e-6:
            break
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
# U2 -- certified-design ALGORITHM: convex relaxation over the design measure + Frank-Wolfe
# with a computable duality-gap certificate (converts "no greedy guarantee" -> certified
# epsilon-optimal for the surrogate / D_s objective).
# --------------------------------------------------------------------------------------
def surrogate_hessian(freqs, out_freqs, super_grid, beta: float = 1.0) -> np.ndarray:
    r"""Hessian of the anti-aliasing surrogate over the design MEASURE.

    Replacing the empirical measure :math:`\tfrac1N\sum_j\delta_{t_j}` by a weighted measure
    :math:`\sum_p w_p\delta_{\tau_p}` on a super-grid :math:`\mathcal T=\{\tau_p\}` turns the
    surrogate into a convex quadratic :math:`J(w)=w^{\top}Q\,w` with

    .. math:: Q[p,q]=\sum_\delta w_\delta\cos\!\big(2\pi\langle\delta,\tau_p-\tau_q\rangle\big)
              \ \succeq\ 0 ,

    the sum over the difference frequencies :math:`\delta` (cross gaps weight 1, in-band gaps
    weight :math:`\beta`) from :func:`_diff_freqs`.  PSD since each term is
    :math:`w_\delta\,\mathrm{Re}(\bm e_\delta\bm e_\delta^{*})` with
    :math:`\bm e_\delta=(e^{i2\pi\langle\delta,\tau_p\rangle})_p`.
    """
    F, _ = _as2d(freqs)
    O, _ = _as2d(out_freqs)
    T, _ = _as2d(super_grid)
    deltas, wts = _diff_freqs(F, O, beta)
    E = np.exp(1j * 2 * np.pi * (deltas @ T.T))          # (D, P), E[d,p]=e^{i2pi<delta_d,tau_p>}
    Q = np.real((wts[:, None] * E).T @ E.conj())         # Q[p,q]=sum_d w_d E[d,p] conj(E[d,q])
    return 0.5 * (Q + Q.T)


def aliasguard_frankwolfe(freqs, out_freqs, super_grid, beta: float = 1.0,
                          n_iter: int = 400, w0=None):
    r"""Frank--Wolfe on :math:`\min_{w\in\Delta(\mathcal T)} w^{\top}Q\,w` (convex).

    Iterates :math:`w_{s+1}=(1-\gamma_s)w_s+\gamma_s\bm e_{p_s}`,
    :math:`p_s=\arg\min_p(\nabla J)_p`, :math:`\gamma_s=2/(s+2)`.  Returns ``(w, info)`` with
    ``info = {J, gap, support}``: ``J`` monotone-decreasing, and the **Frank--Wolfe duality
    gap** :math:`g_s=\langle\nabla J(w_s),w_s-\bm e_{p_s}\rangle\ge J(w_s)-J^{*}` is a
    *computable certificate* of :math:`\varepsilon`-optimality (``min gap`` over iterates).
    Each LMO vertex :math:`\bm e_{p_s}` is a single admissible sample time, so the design is a
    sparse Fedorov-type vertex set.  Convergence :math:`J(w_s)-J^{*}\le 2C_J/(s+2)`.
    """
    Q = surrogate_hessian(freqs, out_freqs, super_grid, beta)
    P = Q.shape[0]
    w = np.full(P, 1.0 / P) if w0 is None else np.asarray(w0, float) / np.sum(w0)
    Qw = Q @ w
    Js, gaps = [], []
    for s in range(n_iter):
        grad = 2.0 * Qw
        p = int(np.argmin(grad))
        gap = float(grad @ w - grad[p])                  # <grad, w - e_p> >= J(w)-J*
        Js.append(float(w @ Qw)); gaps.append(max(gap, 0.0))
        if gap <= 1e-12:
            break
        # exact line search for the quadratic: d = e_p - w, gamma* = (gap/2)/(d^T Q d)
        d = -w.copy(); d[p] += 1.0
        Qd = Q @ d
        dQd = float(d @ Qd)
        gamma = 1.0 if dQd <= 0 else min(1.0, max(0.0, (gap / 2.0) / dQd))
        w = w + gamma * d
        Qw = Qw + gamma * Qd
    support = np.where(w > 1e-6)[0]
    return w, {"J": Js, "gap": gaps, "min_gap": float(min(gaps)) if gaps else 0.0,
               "support_size": int(support.size), "support": support.tolist()}


def design_measure_round(w, super_grid, N: int, seed: int = 0) -> np.ndarray:
    r"""Round a design measure ``w`` to an equal-weight :math:`N`-point design by sampling
    :math:`\mathcal T` with probabilities :math:`w` (systematic residual sampling for low
    variance).  The realized-vs-target coherence gap is bounded by the matrix-Bernstein bound
    of Theorem 1' (see :func:`rounding_gap_bound`)."""
    T, _ = _as2d(super_grid)
    w = np.asarray(w, float); w = w / w.sum()
    rng = np.random.default_rng(seed)
    # systematic sampling of N points proportional to w (with replacement on the grid)
    cum = np.cumsum(w)
    u = (rng.random() + np.arange(N)) / N
    idx = np.searchsorted(cum, u)
    idx = np.clip(idx, 0, T.shape[0] - 1)
    t = T[idx]
    return np.sort(t.reshape(-1)) if t.shape[1] == 1 else t


def rounding_gap_bound(freqs, out_freqs, w, super_grid, N, beta: float = 1.0,
                       delta: float = 0.05, seed: int = 0) -> dict:
    r"""Empirical measure->N rounding gap and its high-probability bound.

    Compares the surrogate at the continuous optimum ``J(w)`` with the realized ``J(t)`` for a
    rounded :math:`N`-point design, and reports the matrix-Bernstein cross-coherence radius
    (Theorem 1', :func:`~inralias.identifiability.cross_coherence_bernstein_eps`) that bounds
    the per-coherence deviation from rounding."""
    from .identifiability import cross_coherence_bernstein_eps
    F = np.asarray(freqs, float); O = np.asarray(out_freqs, float)
    t = design_measure_round(w, super_grid, N, seed=seed)
    J_meas = float(w @ (surrogate_hessian(F, O, super_grid, beta) @ w))
    J_round = surrogate_objective(F.reshape(-1, 1) if F.ndim == 1 else F,
                                  O.reshape(-1, 1) if O.ndim == 1 else O, t, beta)
    m, K = F.reshape(-1, F.shape[-1] if F.ndim > 1 else 1).shape[0], np.atleast_1d(O).shape[0]
    eps_c = cross_coherence_bernstein_eps(m, max(1, K), N, delta)
    return {"J_measure": J_meas, "J_rounded": float(J_round),
            "rounding_gap": float(J_round - J_meas), "coherence_radius": float(eps_c),
            "N": int(N)}


# --------------------------------------------------------------------------------------
# function-space aliasability and the explicit Lipschitz grid certificate
# --------------------------------------------------------------------------------------
def _continuous_gram(freqs):
    r"""Continuous :math:`L^2([0,1)^d)` Gram of the atoms:
    :math:`G[k,\ell]=\prod_d e^{i\pi\delta_d}\operatorname{sinc}(\delta_d)`,
    :math:`\delta=\omega_\ell-\omega_k`.  Integer-spaced frequencies give :math:`G=I`."""
    F, _ = _as2d(freqs)
    D = F[None, :, :] - F[:, None, :]                            # (m,m,d): omega_l - omega_k
    G = np.prod(np.exp(1j * np.pi * D) * np.sinc(D), axis=2)
    np.fill_diagonal(G, 1.0)
    return G


def _gram_factors(Phi, rcond=1e-10):
    r"""Hermitian eigendecomposition of :math:`G=\Phi^{*}\Phi` with a rank/conditioning
    report.  Returns ``(G, evals, evecs, sigma_min, cond, tol, full_rank)``; never forms a
    bare inverse.  ``sigma_min`` and ``cond`` are of :math:`\Phi` (square roots of the
    eigenvalues of ``G``)."""
    G = Phi.conj().T @ Phi
    evals, evecs = np.linalg.eigh(G)                              # ascending, real >= 0
    evals = np.clip(evals, 0.0, None)
    lam_max = float(evals[-1]) if evals.size else 0.0
    tol = max(Phi.shape) * np.finfo(float).eps * lam_max
    lam_min = float(evals[0]) if evals.size else 0.0
    full_rank = lam_min > max(tol, rcond * lam_max)
    sig_min = float(np.sqrt(lam_min))
    cond = float(np.sqrt(lam_max / lam_min)) if lam_min > 0 else np.inf
    return G, evals, evecs, sig_min, cond, tol, full_rank


def _apply_Ginv(evals, evecs, X, power=1):
    r"""Return :math:`(G^{-1})^{power} X` from the eigendecomposition (no bare inverse)."""
    inv = evals.copy()
    inv[inv > 0] = 1.0 / inv[inv > 0]
    return evecs @ (np.diag(inv ** power) @ (evecs.conj().T @ X))


def aliasability_L2_of(freqs, t, nu) -> float:
    r"""**Function-space** aliasability
    :math:`a_{L^2,T}(\nu)=\sqrt{\Delta c^{*}G_{L^2}\Delta c}`, :math:`\Delta c=\Phi^{\dagger}
    \phi_\nu`: the :math:`L^2([0,1)^d)` energy the aliased tone injects into the fitted
    model.  Equals the coefficient norm :func:`aliasability_of` iff the dictionary is
    :math:`L^2`-orthonormal (e.g. integer frequencies); differs otherwise."""
    freqs = np.asarray(freqs, float)
    Phi = _synth(freqs, t)
    phi = _synth(np.atleast_2d(nu), t)[:, 0]
    dc, *_ = np.linalg.lstsq(Phi, phi, rcond=None)
    G_L2 = _continuous_gram(freqs)
    return float(np.sqrt(max(np.real(dc.conj() @ G_L2 @ dc), 0.0)))


def _lipschitz_sup(M, dt, band, n_grid):
    r"""Explicit Lipschitz grid certificate for the sup of the real finite exponential
    polynomial :math:`g(\nu)=\sum_{j,l}M_{jl}e^{i2\pi\nu(t_l-t_j)}` over a union of intervals.

    The frequencies :math:`t_l-t_j` are generally non-integer, so :math:`g` is a
    *generalized* trigonometric polynomial.  Its derivative obeys the elementary bound
    :math:`|g'|\le L:=2\pi\sum_{j,l}|M_{jl}||t_l-t_j|` (triangle inequality; each exponential
    is unit-modulus), hence :math:`\sup_{[a,b]}g\le\max_{\text{grid}}g+Lh/2` for grid
    spacing ``h``.  Returns ``(certified_sup, grid_max, L, slack)``.  Sample-only."""
    L = float(2 * np.pi * np.sum(np.abs(M) * np.abs(dt)))
    dtf = dt.ravel()
    Mf = M.ravel()
    grid_max, slack = -np.inf, 0.0
    for (a, b) in band:
        n = max(2, int(np.ceil((b - a) * n_grid)))
        nu = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
        for lo in range(0, n, 4096):
            blk = nu[lo:lo + 4096]
            g = np.real(np.exp(1j * 2 * np.pi * np.outer(blk, dtf)) @ Mf)
            grid_max = max(grid_max, float(np.max(g)))
        slack = max(slack, L * h / 2.0)
    return grid_max + slack, grid_max, L, slack


def aliasability_certificate(freqs, t, band, n_grid=4000, metric="l2", rcond=1e-10):
    r"""Deterministic post-hoc certificate: an upper bound on
    :math:`\max_{\nu\in\text{band}}a(\nu)` over a *continuous* band (union of intervals),
    for **any** realized design ``t`` (random, E-optimal, or optimized alike).

    ``metric='l2'`` certifies the function-space aliasability
    :math:`a_{L^2,T}` (:math:`M=\Phi(\Phi^{*}\Phi)^{-1}G_{L^2}(\Phi^{*}\Phi)^{-1}\Phi^{*}`);
    ``metric='coeff'`` certifies the coefficient norm
    (:math:`M=\Phi(\Phi^{*}\Phi)^{-2}\Phi^{*}`, i.e. :math:`G_{L^2}=I`).  Numerics use a
    Hermitian eigendecomposition of :math:`\Phi^{*}\Phi` (no bare inverse); if
    :math:`\Phi` is numerically rank deficient the certificate is **vacuous**
    (``inf``/``0``), reported via ``full_rank=False``.

    Returns ``certified_max_aliasability`` (upper bound over the band),
    ``grid_max_aliasability``, ``lipschitz``, ``lipschitz_slack``, ``sigma_min``,
    ``condition_number``, ``rank_tol``, ``full_rank``, ``metric``.
    """
    freqs = np.asarray(freqs, float)
    t = np.asarray(t, float).reshape(-1)
    Phi = _synth(freqs, t)
    G, evals, evecs, sig_min, cond, tol, full_rank = _gram_factors(Phi, rcond)
    base = {"sigma_min": sig_min, "condition_number": cond, "rank_tol": tol,
            "full_rank": bool(full_rank), "metric": metric}
    if not full_rank:
        return {"certified_max_aliasability": float("inf"), "grid_max_aliasability": float("inf"),
                "lipschitz": float("inf"), "lipschitz_slack": float("inf"), **base}
    PhiGi = _apply_Ginv(evals, evecs, Phi.conj().T, power=1)      # (Φ*Φ)^{-1} Φ*  (m,N)
    if metric == "l2":
        Gl2 = _continuous_gram(freqs)
        M = PhiGi.conj().T @ Gl2 @ PhiGi                          # Φ(Φ*Φ)^{-1}G(Φ*Φ)^{-1}Φ*
    else:
        M = PhiGi.conj().T @ PhiGi                               # Φ(Φ*Φ)^{-2}Φ*
    dt = t[None, :] - t[:, None]
    sup_g, grid_g, L, slack = _lipschitz_sup(M, dt, band, n_grid)
    return {"certified_max_aliasability": float(np.sqrt(max(sup_g, 0.0))),
            "grid_max_aliasability": float(np.sqrt(max(grid_g, 0.0))),
            "lipschitz": L, "lipschitz_slack": float(np.sqrt(max(slack, 0.0))), **base}


def _lipschitz_sup_nd(M, dt, box, n_per_axis):
    r"""n-D explicit Lipschitz grid certificate for the sup of
    :math:`g(\nu)=\sum_{j,l}M_{jl}e^{i2\pi\langle\nu,t_l-t_j\rangle}` over a box.

    Per-axis bound :math:`|\partial g/\partial\nu_{\rm ax}|\le L_{\rm ax}:=2\pi\sum_{j,l}
    |M_{jl}||t_{l,\rm ax}-t_{j,\rm ax}|`, so over a tensor-product grid with spacing
    :math:`h_{\rm ax}` the multivariate slack is :math:`\sum_{\rm ax}L_{\rm ax}h_{\rm ax}/2`."""
    d = dt.shape[-1]
    Mf = M.ravel()
    dtf = dt.reshape(-1, d)
    L_ax = [float(2 * np.pi * np.sum(np.abs(M) * np.abs(dt[:, :, ax]))) for ax in range(d)]
    axes = [np.linspace(lo, hi, n_per_axis) for (lo, hi) in box]
    grids = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)
    grid_max = -np.inf
    for lo in range(0, pts.shape[0], 2048):
        blk = pts[lo:lo + 2048]
        g = np.real(np.exp(1j * 2 * np.pi * (blk @ dtf.T)) @ Mf)
        grid_max = max(grid_max, float(np.max(g)))
    slack = float(sum(L_ax[ax] * (box[ax][1] - box[ax][0]) / (n_per_axis - 1) / 2.0
                      for ax in range(d)))
    return grid_max + slack, grid_max, L_ax, slack


def aliasability_certificate_nd(freqs, t, box, n_per_axis=48, metric="l2", rcond=1e-10):
    r"""Deterministic post-hoc certificate on :math:`\max_{\nu\in\text{box}}a(\nu)` over a
    *continuous n-D box* of frequency vectors, for **any** realized design (the n-D analogue of
    :func:`aliasability_certificate`).  Reuses the Hermitian-eigendecomposition numerics
    (vacuous if :math:`\Phi` is rank-deficient) and the tensor-product Lipschitz slack.  This
    is a certified anti-aliasing guarantee for n-D acquisition (e.g. k-space / image design)."""
    freqs = np.asarray(freqs, float)
    F, _ = _as2d(freqs)
    T, _ = _as2d(t)
    Phi = _synth(freqs, t)
    G, evals, evecs, sig_min, cond, tol, full_rank = _gram_factors(Phi, rcond)
    base = {"sigma_min": sig_min, "condition_number": cond, "rank_tol": tol,
            "full_rank": bool(full_rank), "metric": metric, "d": int(F.shape[1])}
    if not full_rank:
        return {"certified_max_aliasability": float("inf"),
                "grid_max_aliasability": float("inf"), "lipschitz_slack": float("inf"), **base}
    PhiGi = _apply_Ginv(evals, evecs, Phi.conj().T, power=1)
    if metric == "l2":
        M = PhiGi.conj().T @ _continuous_gram(freqs) @ PhiGi
    else:
        M = PhiGi.conj().T @ PhiGi
    dt = T[None, :, :] - T[:, None, :]                           # (N,N,d)
    sup_g, grid_g, _L, slack = _lipschitz_sup_nd(M, dt, box, n_per_axis)
    return {"certified_max_aliasability": float(np.sqrt(max(sup_g, 0.0))),
            "grid_max_aliasability": float(np.sqrt(max(grid_g, 0.0))),
            "lipschitz_slack": float(np.sqrt(max(slack, 0.0))), **base}


def visibility_certificate(freqs, t, band, n_grid=4000, rcond=1e-10):
    r"""Deterministic post-hoc certified *lower* bound on
    :math:`\min_{\nu\in\text{band}}v_T(\nu)` over a continuous band, for any design.
    Uses :math:`v_T(\nu)^2=1-\phi_\nu^{*}P\phi_\nu/N`, :math:`P=\Phi(\Phi^{*}\Phi)^{-1}
    \Phi^{*}`; certifies :math:`\max_\nu\phi_\nu^{*}P\phi_\nu` by the same explicit
    Lipschitz argument (eigendecomposition; vacuous if rank deficient)."""
    freqs = np.asarray(freqs, float)
    t = np.asarray(t, float).reshape(-1)
    N = t.size
    Phi = _synth(freqs, t)
    _G, evals, evecs, sig_min, cond, tol, full_rank = _gram_factors(Phi, rcond)
    if not full_rank:
        return {"certified_min_visibility": 0.0, "grid_min_visibility": 0.0,
                "sigma_min": sig_min, "condition_number": cond, "full_rank": False}
    P = Phi @ _apply_Ginv(evals, evecs, Phi.conj().T, power=1)
    dt = t[None, :] - t[:, None]
    sup_h, grid_h, _L, _s = _lipschitz_sup(P, dt, band, n_grid)
    return {"certified_min_visibility": float(np.sqrt(max(1.0 - sup_h / N, 0.0))),
            "grid_min_visibility": float(np.sqrt(max(1.0 - grid_h / N, 0.0))),
            "sigma_min": sig_min, "condition_number": cond, "full_rank": True}


def e_optimal_design(freqs, N, n_restarts=200, seed=0, d=1):
    r"""E-optimal random-search baseline: among ``n_restarts`` i.i.d. uniform designs (in
    :math:`[0,1)^d`), return the one maximizing :math:`\lambda_{\min}(\Phi_\Lambda^{*}
    \Phi_\Lambda/N)` (classical optimal-design criterion; uses only :math:`\Lambda`, N)."""
    freqs = np.asarray(freqs, float)
    rng = np.random.default_rng(seed)
    best_t, best_lam = None, -np.inf
    for _ in range(n_restarts):
        t = rng.uniform(0, 1, (N, d)) if d > 1 else np.sort(rng.uniform(0, 1, N))
        Phi = _synth(freqs, t)
        lam = float(np.linalg.eigvalsh(Phi.conj().T @ Phi / N)[0])
        if lam > best_lam:
            best_lam, best_t = lam, t
    return best_t


def ds_optimal_criterion(freqs, out_freqs, t, ridge=1e-8):
    r"""Exact :math:`D_s`-optimality objective for treating :math:`\Omega` as the nuisance
    block: :math:`\log\det S`, :math:`S=M_{\Lambda\Lambda}-M_{\Lambda\Omega}
    M_{\Omega\Omega}^{-1}M_{\Omega\Lambda}` the Schur complement of the normalized augmented
    Gram :math:`M=\tfrac1N[\Phi\ \Psi]^{*}[\Phi\ \Psi]`.  A small ``ridge`` regularizes the
    (possibly rank-deficient) nuisance block.  Higher is better."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    t = np.asarray(t, float)
    Phi = _synth(freqs, t)
    Psi = _synth(out_freqs, t)
    N = _as2d(t)[0].shape[0]
    Mll = Phi.conj().T @ Phi / N
    Mlo = Phi.conj().T @ Psi / N
    Moo = Psi.conj().T @ Psi / N + ridge * np.eye(out_freqs.shape[0] if out_freqs.ndim > 1
                                                  else out_freqs.size)
    S = Mll - Mlo @ np.linalg.solve(Moo, Mlo.conj().T)
    sign, logdet = np.linalg.slogdet(S)
    return float(logdet) if sign.real > 0 else -np.inf


def ds_optimal_design(freqs, out_freqs, N, n_sweeps=10, grid_res=384, seed=0, t0=None, d=1):
    r"""Exact-:math:`D_s` sample design: coordinate descent maximizing
    :func:`ds_optimal_criterion` (Schur-complement log-det) over continuous times in
    :math:`[0,1)^d`.  This is the *standard* nuisance-parameter optimal design against which
    the AliasGuard surrogate is compared (they are related but not identical objectives).
    Deterministic given ``(seed, t0)``."""
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    neg = lambda cand: -ds_optimal_criterion(freqs, out_freqs, cand)
    t = _coord_descent(freqs, out_freqs, N, neg, d=d, n_sweeps=n_sweeps,
                       grid_res=grid_res, seed=seed, t0=t0)
    return np.sort(t.reshape(-1)) if d == 1 else t


def annealing_design(freqs, out_freqs, N, maxiter=200, seed=0):
    r"""Global-optimizer baseline (SciPy ``dual_annealing``) minimizing the TRUE worst-case
    function-space aliasability :math:`\max_\nu a_{L^2,T}(\nu)` directly (not the surrogate),
    at a comparable evaluation budget.  Shows the AliasGuard coordinate descent is not merely
    exploiting a local optimum of a weak objective."""
    from scipy.optimize import dual_annealing

    freqs = np.asarray(freqs, float)
    O, _ = _as2d(out_freqs)

    def obj(t):
        ts = np.asarray(t, float)
        return max(aliasability_L2_of(freqs, ts, O[i]) for i in range(O.shape[0]))

    res = dual_annealing(obj, bounds=[(0.0, 1.0)] * N, maxiter=maxiter, seed=seed,
                         no_local_search=True)
    return np.sort(res.x)
