r"""Closed-form achievability and converse limits.

We observe ``N`` samples of ``f = f_in + f_out`` where ``f_in`` has spectrum inside the
INR's representable set :math:`\Lambda` (frequencies ``freqs``) and ``f_out`` has spectrum
disjoint from :math:`\Lambda`.  With synthesis matrix :math:`\Phi` (see
:mod:`inralias.sampling`) the least-squares INR fit is
:math:`\hat c=(\Phi^\ast\Phi)^{-1}\Phi^\ast y`, :math:`y=\Phi c^\star+g+\varepsilon`,
:math:`g_j=f_{\text{out}}(t_j)`.

* **Theorem 1 (achievability).**  If :math:`f_{\text{out}}=0` and :math:`\sigma_{\min}(\Phi)>0`,
  :math:`\lVert\hat c-c^\star\rVert\le\kappa\lVert\varepsilon\rVert` with noise gain
  :math:`\kappa=1/\sigma_{\min}(\Phi)`, and :math:`\mathbb E\lVert\hat c-c^\star\rVert^2
  =\sigma^2\,\mathrm{tr}((\Phi^\ast\Phi)^{-1})`.  (:func:`inralias.sampling.noise_gain`,
  :func:`mmse_recoverable`.)

* **Theorem 2 (converse / aliasing).**  With :math:`f_{\text{out}}\neq0`, the coefficient
  estimate is biased by :math:`\Delta c=(\Phi^\ast\Phi)^{-1}\Phi^\ast g` (:func:`aliasing_bias`).
  No INR in the class fits the samples better than :math:`\lVert(I-P)g\rVert`
  (:func:`aliasing_floor`), yet the *spectral* corruption :math:`\lVert\Delta c\rVert`
  can be large even when that residual is tiny -- **silent aliasing**
  (:func:`silent_aliasing_ratio`).  An out-of-band tone folds onto the :math:`\Lambda`
  frequency of :func:`folded_frequency`.

* **Theorem 3 (statistical).**  For random out-of-band content + Gaussian noise the MSE of
  the least-squares estimator splits exactly into an estimation-variance term
  :math:`\sigma^2\,\mathrm{tr}((\Phi^\ast\Phi)^{-1})` (:func:`mmse_recoverable`) and an
  aliasing-variance term :math:`\mathrm{tr}(M\,\Sigma_g\,M^\ast)`,
  :math:`M=(\Phi^\ast\Phi)^{-1}\Phi^\ast` (:func:`mmse_aliasing_floor`).  The aliasing term
  is a fixed-``N`` quantity: it stays at full power along *coherent* folds (an out-of-band
  atom with :math:`\lVert M\phi_\nu\rVert` bounded away from zero, e.g.
  :math:`\nu\equiv\omega_k\ (\mathrm{mod}\ Q)` for samples on a rate-``Q`` grid), while for
  a fixed tone it converges, as density grows, to the tone's window-leakage energy onto
  :math:`\mathrm{span}\,\mathcal M_\Lambda` -- exactly zero for integer-disjoint tones
  under uniform sampling, :math:`O(m/N)` in expectation under i.i.d. sampling, and a
  positive constant :math:`\sum_k \mathrm{sinc}^2(\nu-\omega_k)` for non-integer tones.
  Each regime is pinned in ``tests/test_theory_vs_sim.py``.
"""
from __future__ import annotations

import numpy as np

from inralias.sampling import synthesis_matrix

__all__ = [
    "aliasing_bias",
    "aliasing_floor",
    "reconstruction_error",
    "folded_frequency",
    "silent_aliasing_ratio",
    "mmse_recoverable",
    "mmse_aliasing_floor",
]


def _ls_operator(Phi: np.ndarray) -> np.ndarray:
    r"""The least-squares operator :math:`M=(\Phi^\ast\Phi)^{-1}\Phi^\ast` (m x N)."""
    G = Phi.conj().T @ Phi
    return np.linalg.solve(G, Phi.conj().T)


def aliasing_bias(
    freqs: np.ndarray,
    t: np.ndarray,
    out_freqs: np.ndarray,
    out_coeffs: np.ndarray,
) -> np.ndarray:
    r"""Coefficient bias :math:`\Delta c=(\Phi^\ast\Phi)^{-1}\Phi^\ast g` induced by the
    out-of-band content :math:`f_{\text{out}}=\sum_l a_l e_{\nu_l}` (Theorem 2).

    Returns the length-``m`` complex vector of *aliased* coefficients on ``freqs``.
    """
    Phi = synthesis_matrix(freqs, t)
    Phi_out = synthesis_matrix(np.asarray(out_freqs, float), t)
    g = Phi_out @ np.asarray(out_coeffs, complex).reshape(-1)
    M = _ls_operator(Phi)
    return M @ g


def aliasing_floor(
    freqs: np.ndarray,
    t: np.ndarray,
    out_freqs: np.ndarray,
    out_coeffs: np.ndarray,
) -> float:
    r"""Irreducible sample-domain fit error :math:`\lVert(I-P)g\rVert` -- the converse
    floor: no INR restricted to ``freqs`` can fit the noiseless samples better."""
    Phi = synthesis_matrix(freqs, t)
    Phi_out = synthesis_matrix(np.asarray(out_freqs, float), t)
    g = Phi_out @ np.asarray(out_coeffs, complex).reshape(-1)
    # residual of projecting g onto range(Phi)
    c, *_ = np.linalg.lstsq(Phi, g, rcond=None)
    resid = g - Phi @ c
    return float(np.linalg.norm(resid))


def reconstruction_error(
    freqs: np.ndarray,
    t_fit: np.ndarray,
    out_freqs: np.ndarray,
    out_coeffs: np.ndarray,
    t_eval: np.ndarray,
    in_freqs: np.ndarray | None = None,
    in_coeffs: np.ndarray | None = None,
) -> dict:
    r"""Full noiseless picture of Theorem 2 for a planted signal.

    Fits the INR (frequencies ``freqs``) by least squares to samples of ``f_in+f_out`` at
    ``t_fit``, then evaluates on a dense grid ``t_eval``.  Returns a dict with the
    sample-domain residual (the floor), the global reconstruction MSE, and the spectral
    bias ``||Delta c||`` -- exposing "silent aliasing" when global MSE is small but the
    spectrum is corrupted.
    """
    freqs = np.asarray(freqs, float)
    Phi_fit = synthesis_matrix(freqs, t_fit)
    Phi_out_fit = synthesis_matrix(np.asarray(out_freqs, float), t_fit)
    g = Phi_out_fit @ np.asarray(out_coeffs, complex).reshape(-1)

    c_star = np.zeros(freqs.shape[0], complex)
    if in_freqs is not None:
        # place the in-band coeffs onto the matching dictionary atoms
        in_freqs = np.asarray(in_freqs, float)
        in_coeffs = np.asarray(in_coeffs, complex).reshape(-1)
        for nu, a in zip(in_freqs, in_coeffs):
            k = int(np.argmin(np.abs(freqs - nu)))
            c_star[k] += a
    y = Phi_fit @ c_star + g

    M = _ls_operator(Phi_fit)
    c_hat = M @ y
    dc = c_hat - c_star

    # dense reconstruction vs truth
    Phi_eval = synthesis_matrix(freqs, t_eval)
    f_hat = Phi_eval @ c_hat
    f_true = Phi_eval @ c_star + synthesis_matrix(np.asarray(out_freqs, float), t_eval) @ np.asarray(out_coeffs, complex).reshape(-1)

    resid = y - Phi_fit @ c_hat
    return {
        "sample_residual": float(np.linalg.norm(resid)),
        "spectral_bias": float(np.linalg.norm(dc)),
        "global_mse": float(np.mean(np.abs(f_hat - f_true) ** 2)),
        "in_band_true_energy": float(np.linalg.norm(c_star) ** 2),
        "c_hat": c_hat,
        "c_star": c_star,
    }


def folded_frequency(
    omega_out: float,
    freqs: np.ndarray,
    t: np.ndarray,
) -> dict:
    r"""Which :math:`\Lambda` frequency an out-of-band tone :math:`e_{\omega_{\text{out}}}`
    aliases onto, and with what magnitude.

    Returns ``{"fold_freq", "fold_index, "bias" (Delta c), "leakage"}``.  For uniform
    sampling ``t_j=j/N`` and integer frequencies this reproduces classical folding
    :math:`\omega_{\text{out}}\bmod N` (verified in the test suite), while for structured
    /nonuniform sampling it gives the general learned-aliasing target.
    """
    freqs = np.asarray(freqs, float)
    dc = aliasing_bias(freqs, t, np.array([float(omega_out)]), np.array([1.0 + 0j]))
    k = int(np.argmax(np.abs(dc)))
    return {
        "fold_freq": float(freqs[k]),
        "fold_index": k,
        "bias": dc,
        "leakage": float(np.abs(dc[k])),
    }


def silent_aliasing_ratio(
    freqs: np.ndarray,
    t: np.ndarray,
    out_freqs: np.ndarray,
    out_coeffs: np.ndarray,
) -> float:
    r"""Silent-aliasing index :math:`\lVert\Delta c\rVert / (\lVert(I-P)g\rVert+\epsilon)`.

    Large values mean the samples are fit well (small visible residual) while the spectrum
    is badly corrupted -- the practically dangerous regime a global MSE would miss.
    """
    dc = aliasing_bias(freqs, t, out_freqs, out_coeffs)
    floor = aliasing_floor(freqs, t, out_freqs, out_coeffs)
    return float(np.linalg.norm(dc) / (floor + 1e-12))


def mmse_recoverable(
    Phi: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray | float | None = None,
) -> float:
    r"""Recoverable-regime MMSE (Theorem 3 variance term).

    * If ``prior_var`` is ``None`` (unbiased LS): :math:`\sigma^2\,\mathrm{tr}((\Phi^\ast\Phi)^{-1})`.
    * If a Gaussian prior with per-coefficient variance ``prior_var`` is given, returns the
      Bayes posterior-covariance trace
      :math:`\mathrm{tr}\big((\Sigma_0^{-1}+\Phi^\ast\Phi/\sigma^2)^{-1}\big)`.
    """
    m = Phi.shape[1]
    G = Phi.conj().T @ Phi
    if prior_var is None:
        return float(np.real(np.trace(np.linalg.inv(G))) * sigma2)
    if np.isscalar(prior_var):
        Sig0_inv = np.eye(m) / float(prior_var)
    else:
        Sig0_inv = np.diag(1.0 / np.asarray(prior_var, float))
    post = np.linalg.inv(Sig0_inv + G / sigma2)
    return float(np.real(np.trace(post)))


def mmse_aliasing_floor(
    freqs: np.ndarray,
    t: np.ndarray,
    out_freqs: np.ndarray,
    out_power: np.ndarray | float,
) -> float:
    r"""Non-vanishing aliasing-bias floor (Theorem 3):
    :math:`\mathrm{tr}(M\,\Sigma_g\,M^\ast)` where :math:`M=(\Phi^\ast\Phi)^{-1}\Phi^\ast`
    and :math:`\Sigma_g=\Phi_{\text{out}}\,\mathrm{diag}(\text{out\_power})\,\Phi_{\text{out}}^\ast`
    is the covariance of the out-of-band sample vector for random, uncorrelated out-of-band
    coefficients.  Unlike the variance term this does **not** vanish as ``N`` grows when the
    out-of-band content is *coherent* with :math:`\Lambda` under the sampling, i.e. when
    :math:`\lVert M\phi_{\nu_l}\rVert` stays bounded away from zero (for samples on a
    rate-``Q`` grid: :math:`\nu_l\equiv\omega_k\ (\mathrm{mod}\ Q)`); for a fixed
    incoherent tone it converges to the tone's window-leakage energy (see module
    docstring and the regime tests).
    """
    freqs = np.asarray(freqs, float)
    out_freqs = np.asarray(out_freqs, float)
    Phi = synthesis_matrix(freqs, t)
    Phi_out = synthesis_matrix(out_freqs, t)
    if np.isscalar(out_power):
        p = np.full(out_freqs.shape[0], float(out_power))
    else:
        p = np.asarray(out_power, float)
    M = _ls_operator(Phi)
    # E||Delta c||^2 = tr(M Sigma_g M^*) = sum_l p_l ||M Phi_out[:,l]||^2
    MPhi = M @ Phi_out
    col_energy = np.sum(np.abs(MPhi) ** 2, axis=0)
    return float(np.sum(p * col_energy))
