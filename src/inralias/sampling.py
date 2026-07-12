r"""Sampling / frame machinery on a finite frequency dictionary.

An implicit neural representation with a fixed Fourier feature set is, in its output
layer, a linear model over complex exponential atoms

.. math::  e_\omega(x) = e^{i 2\pi \omega x},\qquad \omega \in \Lambda,

where :math:`\Lambda=\{\omega_1,\dots,\omega_m\}` is the network's *representable
frequency set*.  Given sample locations :math:`t_1,\dots,t_N\in[0,1)` the **synthesis
matrix** is :math:`\Phi\in\mathbb C^{N\times m}`, :math:`\Phi_{jk}=e^{i2\pi\omega_k t_j}`.
Fitting the INR to samples is least squares with :math:`\Phi`.  All achievability and
converse limits reduce to the spectral geometry of :math:`\Phi`, which this module
computes exactly (no training, no approximation) so that the theorems are directly
Monte-Carlo checkable.

Conventions
-----------
* Frequencies ``freqs`` and sample times ``t`` are real 1-D arrays.
* Everything is complex-exponential based; real cos/sin dictionaries are handled by
  passing the signed frequency set ``{+/- omega}`` and taking real parts downstream.
* "Frame bounds" are the squared singular values of :math:`\Phi` (unnormalized): the
  lower/upper frame bounds :math:`A=\sigma_{\min}(\Phi)^2`, :math:`B=\sigma_{\max}(\Phi)^2`.
  The dictionary is a frame for its sampled span iff :math:`A>0`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "synthesis_matrix",
    "gram",
    "frame_bounds",
    "condition_number",
    "noise_gain",
    "alias_projector",
    "pinv_apply",
    "bandwidth",
]


def synthesis_matrix(freqs: np.ndarray, t: np.ndarray) -> np.ndarray:
    r"""Return :math:`\Phi\in\mathbb C^{N\times m}`, :math:`\Phi_{jk}=e^{i2\pi\omega_k t_j}`.

    Parameters
    ----------
    freqs : (m,) real array of dictionary frequencies :math:`\Lambda`.
    t : (N,) real array of sample locations in ``[0, 1)``.
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    # outer product t_j * omega_k  ->  (N, m)
    phase = 2.0 * np.pi * np.outer(t, freqs)
    return np.exp(1j * phase)


def gram(Phi: np.ndarray, normalize: bool = False) -> np.ndarray:
    r"""Gram matrix :math:`G=\Phi^\ast\Phi` (optionally divided by ``N``)."""
    G = Phi.conj().T @ Phi
    if normalize:
        G = G / Phi.shape[0]
    return G


def frame_bounds(Phi: np.ndarray) -> tuple[float, float]:
    r"""Lower/upper frame bounds :math:`(A,B)=(\sigma_{\min}^2,\sigma_{\max}^2)` of ``Phi``.

    ``A > 0`` certifies that the sampled dictionary is a frame for its span, i.e. the
    in-band signal is stably recoverable from the samples (Theorem 1 hypothesis).
    """
    s = np.linalg.svd(Phi, compute_uv=False)
    A = float(s[-1] ** 2)
    B = float(s[0] ** 2)
    return A, B


def condition_number(Phi: np.ndarray) -> float:
    r"""Frame condition number :math:`\sqrt{B/A}=\sigma_{\max}/\sigma_{\min}`."""
    s = np.linalg.svd(Phi, compute_uv=False)
    if s[-1] <= 0:
        return np.inf
    return float(s[0] / s[-1])


def noise_gain(Phi: np.ndarray) -> float:
    r"""Least-squares **noise gain** :math:`\kappa=\lVert(\Phi^\ast\Phi)^{-1}\Phi^\ast\rVert_2
    =1/\sigma_{\min}(\Phi)`.

    This is the worst-case amplification of measurement noise into the recovered
    coefficients (Theorem 1): :math:`\lVert\hat c-c^\star\rVert\le\kappa\,\lVert\varepsilon\rVert`.
    Requires full column rank (:math:`m\le N`); returns ``inf`` otherwise, since
    :math:`(\Phi^\ast\Phi)^{-1}` then does not exist.
    """
    N, m = Phi.shape
    if m > N:
        return np.inf
    s = np.linalg.svd(Phi, compute_uv=False)
    if s[-1] <= 0:
        return np.inf
    return float(1.0 / s[-1])


def alias_projector(Phi: np.ndarray) -> np.ndarray:
    r"""Orthogonal projector :math:`P=\Phi(\Phi^\ast\Phi)^{-1}\Phi^\ast` onto the sampled
    dictionary subspace :math:`\mathrm{range}(\Phi)\subseteq\mathbb C^N`.

    Out-of-band sample vectors :math:`g` decompose as :math:`g=Pg+(I-P)g`; the component
    :math:`Pg` is *aliased* (indistinguishable from an in-band signal at the sample
    points) while :math:`(I-P)g` is the visible residual (Theorem 2).
    """
    # Use a stable pseudo-inverse based projector.
    U, s, _ = np.linalg.svd(Phi, full_matrices=False)
    tol = max(Phi.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    Ur = U[:, :r]
    return Ur @ Ur.conj().T


def pinv_apply(Phi: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    r"""Least-squares / ridge coefficients, numerically safe at any rank.

    * ``ridge == 0``: the SVD **pseudoinverse** solution :math:`\hat c=\Phi^\dagger y`
      (via ``lstsq``).  In the overdetermined full-rank case this is ordinary least
      squares; in the underdetermined or rank-deficient case it is the
      **minimum-norm** interpolant -- these are different estimators and are treated
      separately in the paper.
    * ``ridge > 0``: the ridge estimator
      :math:`\hat c=(\Phi^\ast\Phi+\lambda I)^{-1}\Phi^\ast y` (always well posed, but
      biased; the exact theory statements in :mod:`inralias.limits` are for the
      unridged estimators).
    """
    if ridge > 0:
        m = Phi.shape[1]
        G = Phi.conj().T @ Phi + ridge * np.eye(m)
        return np.linalg.solve(G, Phi.conj().T @ y)
    c, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return c


def bandwidth(freqs: np.ndarray) -> float:
    """Half-extent (max |omega|) of a frequency set -- its two-sided bandwidth reach."""
    freqs = np.asarray(freqs, dtype=float)
    return float(np.max(np.abs(freqs))) if freqs.size else 0.0
