r"""Least-squares error identities for the fixed Fourier-feature coordinate model.

We observe ``N`` samples of ``f = f_in + f_out`` where ``f_in`` has spectrum inside the
model's frequency set :math:`\Lambda` (``freqs``) and ``f_out`` is a finite exponential
sum on frequencies outside :math:`\Lambda`.  With synthesis matrix :math:`\Phi` (see
:mod:`inralias.sampling`) the ordinary-LS fit is
:math:`\hat c=(\Phi^\ast\Phi)^{-1}\Phi^\ast y`, :math:`y=\Phi c^\star+g+\varepsilon`,
:math:`g_j=f_{\text{out}}(t_j)`.

These are *background* identities (standard LS perturbation / omitted-variable analysis,
instantiated on the exponential dictionary); the paper's main results live in
:mod:`inralias.identifiability` (visibility, aliasability, exact indistinguishability,
random-sampling concentration, detection).

* **LS stability (background).**  If :math:`f_{\text{out}}=0` and
  :math:`\sigma_{\min}(\Phi)>0`,
  :math:`\lVert\hat c-c^\star\rVert\le\kappa\lVert\varepsilon\rVert` with
  :math:`\kappa=1/\sigma_{\min}(\Phi)` (:func:`inralias.sampling.noise_gain`), and
  :math:`\mathbb E\lVert\hat c-c^\star\rVert^2=\sigma^2\,\mathrm{tr}((\Phi^\ast\Phi)^{-1})`
  (:func:`ls_coefficient_mse`).  Coefficient-space quantities; convert to function space
  with :func:`inralias.identifiability.riesz_bounds`.

* **Aliasing bias (background + T1 objects).**  With :math:`f_{\text{out}}\neq0` the LS
  estimate is biased by :math:`\Delta c=(\Phi^\ast\Phi)^{-1}\Phi^\ast g`
  (:func:`aliasing_bias`); no model in the class fits the noiseless samples better than
  :math:`\lVert(I-P)g\rVert` (:func:`aliasing_floor` = :math:`\sqrt N\times` the
  *visibility* of :mod:`inralias.identifiability`).  In the grid-coherent regime the
  residual is exactly zero while :math:`\lVert\Delta c\rVert` carries the full tone
  energy -- silent aliasing (:func:`silent_aliasing_ratio`, and T1/T4 for the
  indistinguishability statements).

* **Average-case decomposition (background).**  For zero-mean uncorrelated out-of-band
  coefficients independent of white noise, the LS coefficient MSE splits exactly into
  :math:`\sigma^2\,\mathrm{tr}((\Phi^\ast\Phi)^{-1})` (:func:`ls_coefficient_mse`) plus
  :math:`\mathrm{tr}(M\,\Sigma_g\,M^\ast)` (:func:`aliasing_variance_term`).  The second
  term stays at full power along coherent folds (:math:`\nu\equiv\omega_k\pmod Q` on a
  rate-``Q`` grid) and, for a fixed tone under growing density, converges to the tone's
  window-leakage energy -- exactly zero for integer-disjoint tones under uniform
  sampling, :math:`O(m/N)` in expectation under i.i.d. sampling, and
  :math:`\sum_k\mathrm{sinc}^2(\nu-\omega_k)` for non-integer tones.  Each regime is
  pinned in ``tests/test_theory_vs_sim.py``.
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
    "ls_coefficient_mse",
    "bayes_coefficient_mmse",
    "aliasing_variance_term",
]

_COND_WARN = 1e8


def _ls_operator(Phi: np.ndarray) -> np.ndarray:
    r"""The least-squares operator :math:`M=(\Phi^\ast\Phi)^{-1}\Phi^\ast` (m x N).

    Requires full column rank; raises ``LinAlgError`` if the synthesis matrix is
    (numerically) rank deficient and warns when its condition number exceeds
    ``1e8`` (theory statements assume well-posed ordinary least squares).
    """
    import warnings

    s = np.linalg.svd(Phi, compute_uv=False)
    tol = max(Phi.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    if s.size == 0 or s[-1] <= tol:
        raise np.linalg.LinAlgError(
            "synthesis matrix is rank deficient; the ordinary-LS operator does not "
            "exist (use pinv_apply for the minimum-norm estimator)"
        )
    if s[0] / s[-1] > _COND_WARN:
        warnings.warn(
            f"synthesis matrix condition number {s[0]/s[-1]:.2e} > {_COND_WARN:.0e}; "
            "least-squares quantities may be numerically fragile",
            stacklevel=2,
        )
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
    r"""Full noiseless picture of the converse for a planted signal.

    Fits the model (frequencies ``freqs``) by least squares to samples of ``f_in+f_out``
    at ``t_fit``, then evaluates on a dense grid ``t_eval``.  Returns a dict with the
    sample-domain residual, the global reconstruction MSE, and the coefficient bias
    ``||Delta c||``.  In the *silent aliasing* regime (grid-coherent out-of-band content)
    the sample residual is at the noise level while both the coefficient bias and the
    global reconstruction error are large -- the fit looks good only where it was
    evaluated.
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


def ls_coefficient_mse(Phi: np.ndarray, sigma2: float) -> float:
    r"""Coefficient-space MSE of ordinary least squares under white noise:
    :math:`\mathbb E\lVert\hat c-c^\star\rVert_2^2=\sigma^2\,\mathrm{tr}((\Phi^\ast\Phi)^{-1})`.

    This is a **coefficient** quantity for the **LS estimator**; it is not a Bayes MMSE
    and, for non-orthonormal dictionaries, not a function-space error either -- convert
    with :func:`inralias.identifiability.riesz_bounds` /
    :func:`inralias.identifiability.function_error_decomposition`.
    """
    G = Phi.conj().T @ Phi
    return float(np.real(np.trace(np.linalg.inv(G))) * sigma2)


def bayes_coefficient_mmse(
    Phi: np.ndarray, sigma2: float, prior_var: np.ndarray | float
) -> float:
    r"""Bayes MMSE of the coefficients in the conjugate linear-Gaussian model.

    For :math:`c\sim\mathcal{CN}(0,\Sigma_0)` (diagonal ``prior_var``) and white Gaussian
    noise, the posterior is Gaussian and the MMSE equals the posterior-covariance trace
    :math:`\mathrm{tr}\big((\Sigma_0^{-1}+\Phi^\ast\Phi/\sigma^2)^{-1}\big)` (standard
    conjugacy).  Kept separate from :func:`ls_coefficient_mse`, which assumes no prior.
    """
    m = Phi.shape[1]
    G = Phi.conj().T @ Phi
    if np.isscalar(prior_var):
        Sig0_inv = np.eye(m) / float(prior_var)
    else:
        Sig0_inv = np.diag(1.0 / np.asarray(prior_var, float))
    post = np.linalg.inv(Sig0_inv + G / sigma2)
    return float(np.real(np.trace(post)))


def aliasing_variance_term(
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
