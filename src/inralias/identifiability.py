r"""Identifiability, visibility, and aliasability of out-of-band tones (Theorems T1--T4).

This module is the theory core of the paper *Silent Aliasing in Fixed Fourier-Feature
Coordinate Models*.  The objects it computes are exactly the objects the theorems are
stated about; every closed form here is pinned against Monte Carlo in the test suite.

Setting
-------
A fixed Fourier-feature coordinate model with frequency set
:math:`\Lambda=\{\omega_1,\dots,\omega_m\}` and linear coefficients (equivalently: linear
least squares on the dictionary of complex exponentials).  Samples
:math:`T=\{t_1,\dots,t_N\}\subset[0,1)` define the *sampling operator*
:math:`S_T f=(f(t_j))_j` and the synthesis matrix
:math:`\Phi_\Lambda\in\mathbb C^{N\times m}`, :math:`(\Phi_\Lambda)_{jk}=e^{i2\pi\omega_k t_j}`.

Key definitions (T1)
--------------------
* **sample atom**    :math:`\phi_\nu=(e^{i2\pi\nu t_j})_j\in\mathbb C^N`;
* **visibility**     :math:`v_T(\nu)=\lVert(I-P_\Lambda)\phi_\nu\rVert_2/\sqrt N\in[0,1]`,
  with :math:`P_\Lambda` the orthogonal projector onto :math:`\mathrm{range}(\Phi_\Lambda)`;
* **aliasability**   :math:`a_T(\nu)=\lVert\Phi_\Lambda^\dagger\phi_\nu\rVert_2`
  (pseudoinverse; equals the coefficient bias magnitude a unit tone at :math:`\nu`
  induces in the least-squares fit);
* **exact equivalence**: :math:`\nu\sim_T\omega` iff :math:`\phi_\nu=\phi_\omega`, i.e. the
  two tones are *indistinguishable on the samples*.  :math:`v_T(\nu)=0` iff
  :math:`\phi_\nu\in\mathrm{range}(\Phi_\Lambda)` iff the tone is exactly reproducible by
  the model class on the samples.

Grid classification (T1): if :math:`T` is a subset of the rate-:math:`Q` grid
:math:`\{0,\dots,Q-1\}/Q` then :math:`\phi_\nu` depends on :math:`\nu\in\mathbb Z` only
through :math:`\nu\bmod Q`, so integer tones fall into equivalence classes indexed by
:math:`\mathbb Z_Q`; :math:`\nu\equiv\omega_k\pmod Q` forces :math:`v_T(\nu)=0` for *every*
subset of the grid.

Function-space objects (T2)
---------------------------
* :func:`continuous_gram` -- :math:`G_{L^2}[k,\ell]=\langle e_{\omega_\ell},
  e_{\omega_k}\rangle_{L^2[0,1)}` in closed form;
* :func:`riesz_bounds` -- extreme eigenvalues of :math:`G_{L^2}` (Riesz constants of the
  finite dictionary), converting coefficient-space errors into function-space errors;
* :func:`function_error_decomposition` -- exact split of
  :math:`\lVert\hat f-f\rVert_{L^2}^2` into modeled-component error, truncation energy,
  and their cross term.

Random / jittered sampling (T3)
-------------------------------
* :func:`expected_jitter_coherence` -- for grid-coherent :math:`\nu=\omega+kQ` and jitter
  :math:`t_j=g_j/Q+\eta_j` the expected sample coherence with the fold atom is the
  characteristic function :math:`\chi_\eta(2\pi kQ)` (Gaussian:
  :math:`e^{-2\pi^2k^2Q^2\sigma_t^2}`; uniform width :math:`w`:
  :math:`\mathrm{sinc}(kQw)`), giving the small-jitter visibility law
  :math:`v_T(\nu)\approx\sqrt{1-|\chi|^2}\approx 2\pi|k|Q\sigma_t`;
* :func:`coherence_epsilon` / :func:`aliasability_concentration_bound` -- finite-candidate
  Hoeffding + union + perturbation bound: for i.i.d. samples and a size-:math:`K`
  out-of-band candidate set, with probability :math:`\ge1-\delta`,
  :math:`\max_{\nu\in\Omega}a_T(\nu)\le\sqrt m\,\varepsilon/(\lambda_{\min}-m\varepsilon)`
  with :math:`\varepsilon=2\sqrt{\log(4(mK+m^2)/\delta)/N}` (valid while
  :math:`m\varepsilon<\lambda_{\min}`, the infinite-:math:`N` Riesz constant of the
  sampling density).

Detection (T4)
--------------
* If two signals induce identical sample vectors (exact equivalence) they induce identical
  observation distributions under any exchangeable noise, so **no sample-based detector
  can separate them**: :func:`exactly_indistinguishable`.
* For visible tones the residual (goodness-of-fit) test has exact noncentral-:math:`\chi^2`
  power :func:`residual_test_power`, with noncentrality
  :math:`\lambda=\lVert(I-P)\bm s\rVert^2/\sigma^2` driven by amplitude *times visibility*.
"""
from __future__ import annotations

import numpy as np

from inralias.sampling import synthesis_matrix

__all__ = [
    "sample_atom",
    "visibility",
    "aliasability",
    "alias_coefficients",
    "grid_equivalence_class",
    "exactly_indistinguishable",
    "continuous_gram",
    "riesz_bounds",
    "function_error_decomposition",
    "expected_jitter_coherence",
    "coherence_epsilon",
    "aliasability_concentration_bound",
    "residual_test_power",
]

_RANK_TOL = 1e-10


def sample_atom(nu: float, t: np.ndarray) -> np.ndarray:
    r"""Sample vector :math:`\phi_\nu=(e^{i2\pi\nu t_j})_j` of a unit complex tone."""
    t = np.asarray(t, float).reshape(-1)
    return np.exp(1j * 2 * np.pi * float(nu) * t)


def _projector_basis(Phi: np.ndarray) -> tuple[np.ndarray, int]:
    """Orthonormal basis U_r of range(Phi) via SVD, plus the numerical rank."""
    U, s, _ = np.linalg.svd(Phi, full_matrices=False)
    tol = max(Phi.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > max(tol, _RANK_TOL * (s[0] if s.size else 1.0))))
    return U[:, :r], r


def visibility(freqs: np.ndarray, t: np.ndarray, nu: float) -> float:
    r"""Visibility :math:`v_T(\nu)=\lVert(I-P_\Lambda)\phi_\nu\rVert/\sqrt N\in[0,1]`.

    ``0`` means the tone is exactly reproducible by the model class on the samples
    (indistinguishable; T1); ``1`` means the tone is orthogonal to the model space on the
    samples.  Computed with an SVD projector, valid at any rank.
    """
    t = np.asarray(t, float).reshape(-1)
    Phi = synthesis_matrix(np.asarray(freqs, float), t)
    phi = sample_atom(nu, t)
    U, _ = _projector_basis(Phi)
    resid = phi - U @ (U.conj().T @ phi)
    return float(np.linalg.norm(resid) / np.sqrt(t.size))


def alias_coefficients(freqs: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
    r"""Pseudoinverse alias pattern :math:`\Phi_\Lambda^\dagger\phi_\nu\in\mathbb C^m`.

    This is the coefficient bias a unit tone at ``nu`` induces in the (minimum-norm)
    least-squares fit.  Uses ``lstsq``/SVD, well defined at any rank (T1).
    """
    Phi = synthesis_matrix(np.asarray(freqs, float), np.asarray(t, float))
    c, *_ = np.linalg.lstsq(Phi, sample_atom(nu, t), rcond=None)
    return c


def aliasability(freqs: np.ndarray, t: np.ndarray, nu: float) -> float:
    r"""Aliasability :math:`a_T(\nu)=\lVert\Phi_\Lambda^\dagger\phi_\nu\rVert_2` (T1)."""
    return float(np.linalg.norm(alias_coefficients(freqs, t, nu)))


def grid_equivalence_class(nu: float, Q: int) -> int:
    r"""Residue class of an integer frequency on a rate-``Q`` grid.

    For samples contained in :math:`\{0,\dots,Q-1\}/Q` the sample atom of an integer tone
    depends only on :math:`\nu\bmod Q`; two integer tones are exactly indistinguishable on
    every grid subset iff they share this class (T1).
    """
    if abs(nu - round(nu)) > 1e-12:
        raise ValueError("grid equivalence classes are defined for integer frequencies")
    return int(round(nu)) % int(Q)


def exactly_indistinguishable(freqs_a, coeffs_a, freqs_b, coeffs_b, t, tol=1e-10) -> bool:
    r"""True iff two finite exponential sums induce identical sample vectors on ``t``.

    Identical sample vectors + identical noise law => identical observation distributions
    => no sample-based detector separates the two hypotheses better than chance (T4a);
    and no estimator can have small worst-case error against both (Le Cam two-point, T1).
    """
    t = np.asarray(t, float).reshape(-1)
    ya = synthesis_matrix(np.asarray(freqs_a, float), t) @ np.asarray(coeffs_a, complex).reshape(-1)
    yb = synthesis_matrix(np.asarray(freqs_b, float), t) @ np.asarray(coeffs_b, complex).reshape(-1)
    return bool(np.max(np.abs(ya - yb)) <= tol * max(1.0, float(np.max(np.abs(ya)))))


# --------------------------------------------------------------------------------------
# T2: function space
# --------------------------------------------------------------------------------------
def continuous_gram(freqs: np.ndarray) -> np.ndarray:
    r"""Continuous-domain Gram matrix :math:`G_{L^2}[k,\ell]=\int_0^1
    e^{i2\pi(\omega_\ell-\omega_k)t}\,dt` in closed form.

    Off-diagonal entries are :math:`e^{i\pi\delta}\,\mathrm{sinc}(\delta)`,
    :math:`\delta=\omega_\ell-\omega_k`, :math:`\mathrm{sinc}(x)=\sin(\pi x)/(\pi x)`;
    integer-spaced frequencies are exactly orthonormal.
    """
    f = np.asarray(freqs, float).reshape(-1)
    D = f[None, :] - f[:, None]          # delta[k, l] = omega_l - omega_k
    G = np.exp(1j * np.pi * D) * np.sinc(D)
    np.fill_diagonal(G, 1.0)
    return G


def riesz_bounds(freqs: np.ndarray) -> tuple[float, float]:
    r"""Extreme eigenvalues :math:`(\lambda_{\min},\lambda_{\max})` of :math:`G_{L^2}`.

    These are the Riesz constants of the finite dictionary on :math:`L^2[0,1)`:
    :math:`\lambda_{\min}\lVert c\rVert^2\le\lVert\sum_k c_k e_{\omega_k}\rVert_{L^2}^2
    \le\lambda_{\max}\lVert c\rVert^2`, the bridge from coefficient error to function
    error (T2).  :math:`\lambda_{\min}>0` iff the frequencies are distinct.
    """
    w = np.linalg.eigvalsh(continuous_gram(freqs))
    return float(w[0]), float(w[-1])


def function_error_decomposition(
    freqs: np.ndarray,
    c_hat: np.ndarray,
    c_star: np.ndarray,
    out_freqs: np.ndarray,
    out_coeffs: np.ndarray,
) -> dict:
    r"""Exact :math:`L^2[0,1)` decomposition of the reconstruction error (T2).

    With :math:`\hat f-f=\sum_k\Delta c_k e_{\omega_k}-\sum_l a_l e_{\nu_l}`:

    .. math:: \lVert\hat f-f\rVert_{L^2}^2
        =\underbrace{\Delta c^{*}G_{\Lambda\Lambda}\Delta c}_{\text{modeled-component}}
        -\ 2\,\mathrm{Re}\big(\Delta c^{*}G_{\Lambda,\mathrm{out}}\,a\big)
        +\underbrace{a^{*}G_{\mathrm{out}}\,a}_{\text{truncation}} .

    The cross term vanishes when the out-of-band atoms are :math:`L^2`-orthogonal to the
    dictionary (e.g. all-integer frequencies).  Note the modeled-component term is
    :math:`\lVert\Delta c\rVert_2^2` *only* for orthonormal dictionaries; in general it is
    the :math:`G_{L^2}` quadratic form (Riesz-equivalent, :func:`riesz_bounds`).
    """
    freqs = np.asarray(freqs, float).reshape(-1)
    out_freqs = np.asarray(out_freqs, float).reshape(-1)
    dc = (np.asarray(c_hat, complex) - np.asarray(c_star, complex)).reshape(-1)
    a = np.asarray(out_coeffs, complex).reshape(-1)

    allf = np.concatenate([freqs, out_freqs])
    G = continuous_gram(allf)
    m = freqs.size
    G_LL = G[:m, :m]
    G_LO = G[:m, m:]
    G_OO = G[m:, m:]

    modeled = float(np.real(dc.conj() @ G_LL @ dc))
    truncation = float(np.real(a.conj() @ G_OO @ a))
    cross = float(-2.0 * np.real(dc.conj() @ G_LO @ a))
    total = modeled + truncation + cross
    return {
        "total_sq": total,
        "modeled_component_sq": modeled,
        "truncation_sq": truncation,
        "cross": cross,
        "total_rmse": float(np.sqrt(max(total, 0.0))),
    }


# --------------------------------------------------------------------------------------
# T3: random / jittered sampling
# --------------------------------------------------------------------------------------
def expected_jitter_coherence(k: int, Q: int, scale: float, dist: str = "gaussian") -> float:
    r"""Expected sample coherence of a grid-coherent tone with its fold atom under jitter.

    Samples :math:`t_j=g_j/Q+\eta_j` with i.i.d. mean-zero jitter :math:`\eta_j`; the tone
    :math:`\nu=\omega+kQ` satisfies
    :math:`\tfrac1N\big|\sum_j e^{i2\pi(\nu-\omega)t_j}\big|
    \to|\chi_\eta(2\pi kQ)|` where :math:`\chi_\eta` is the characteristic function:

    * ``gaussian`` (std ``scale`` seconds): :math:`e^{-2\pi^2k^2Q^2\,\mathrm{scale}^2}`;
    * ``uniform`` (width ``scale`` seconds): :math:`\mathrm{sinc}(kQ\,\mathrm{scale})`.

    Small-jitter law (T3b): visibility :math:`\approx\sqrt{1-\chi^2}\approx
    2\pi|k|Q\sigma_t` -- jitter *continuously breaks* the exact fold.
    """
    x = float(k) * float(Q) * float(scale)
    if dist == "gaussian":
        return float(np.exp(-2.0 * (np.pi * x) ** 2))
    if dist == "uniform":
        return float(np.sinc(x))
    raise ValueError(f"unknown jitter dist {dist!r}")


def coherence_epsilon(N: int, delta: float, n_events: int = 1) -> float:
    r"""Hoeffding radius for empirical coherences (T3a).

    For i.i.d. samples and unit-modulus terms, each empirical coherence
    :math:`\tfrac1N\sum_j e^{i2\pi\delta t_j}` deviates from its mean by more than
    :math:`\varepsilon` with probability at most :math:`4e^{-N\varepsilon^2/4}` (real and
    imaginary Hoeffding).  With a union bound over ``n_events`` coherences the uniform
    radius at confidence :math:`1-\delta` is
    :math:`\varepsilon=2\sqrt{\log(4\,n_{\rm events}/\delta)/N}`.
    """
    return float(2.0 * np.sqrt(np.log(4.0 * max(1, n_events) / float(delta)) / float(N)))


def aliasability_concentration_bound(
    m: int, n_candidates: int, N: int, delta: float, lam_min: float = 1.0
) -> float:
    r"""High-probability uniform aliasability bound over a finite candidate set (T3a).

    Assume i.i.d. samples whose infinite-:math:`N` normalized Gram
    :math:`G_\infty=\lim\Phi^{*}\Phi/N` satisfies :math:`G_\infty\succeq\lambda_{\min}I`
    (for the uniform density and distinct integer frequencies, :math:`G_\infty=I`,
    :math:`\lambda_{\min}=1`), and that every limiting model--candidate coherence is zero
    (integer-disjoint candidates under the uniform density).  With
    :math:`\varepsilon=` :func:`coherence_epsilon` over the :math:`mK+m^2` relevant
    coherences, with probability :math:`\ge1-\delta`:

    .. math:: \max_{\nu\in\Omega_{\rm out}} a_T(\nu)\ \le\
        \frac{\sqrt m\,\varepsilon}{\lambda_{\min}-m\varepsilon},
        \qquad\text{provided } m\varepsilon<\lambda_{\min}.

    Returns ``inf`` when the proviso fails (bound vacuous at that :math:`N`).
    Proof: perturbation of the normalized Gram plus Cauchy--Schwarz on
    :math:`a_T(\nu)=\lVert(\Phi^{*}\Phi/N)^{-1}(\Phi^{*}\phi_\nu/N)\rVert`.
    """
    eps = coherence_epsilon(N, delta, n_events=m * n_candidates + m * m)
    if m * eps >= lam_min:
        return float("inf")
    return float(np.sqrt(m) * eps / (lam_min - m * eps))


# --------------------------------------------------------------------------------------
# T4: detection
# --------------------------------------------------------------------------------------
def residual_test_power(
    design: np.ndarray,
    tone_sample_vector: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
) -> dict:
    r"""Exact power of the residual (goodness-of-fit) test against an additive component.

    Model: real observations :math:`y=D\beta+s+\varepsilon`, :math:`D` the real design
    (``N x p``), :math:`s` the sample vector of the visible component,
    :math:`\varepsilon\sim\mathcal N(0,\sigma^2 I)`.  The statistic
    :math:`\lVert(I-P_D)y\rVert^2/\sigma^2` is :math:`\chi^2_{N-p}` under H0 and
    noncentral :math:`\chi^2_{N-p}(\lambda)`, :math:`\lambda=\lVert(I-P_D)s\rVert^2/
    \sigma^2`, under H1 -- i.e. detection power is driven by *amplitude times
    visibility*; an exactly coherent component has :math:`\lambda=0` and the test is
    blind, matching T4a (T4b).
    """
    from scipy.stats import chi2, ncx2

    D = np.asarray(design, float)
    s = np.asarray(tone_sample_vector, float).reshape(-1)
    N, p = D.shape
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    r = int(np.sum(sv > max(N, p) * np.finfo(float).eps * (sv[0] if sv.size else 0)))
    Ur = U[:, :r]
    resid = s - Ur @ (Ur.T @ s)
    lam = float(np.dot(resid, resid) / sigma**2)
    dof = N - r
    thresh = chi2.ppf(1.0 - alpha, dof)
    power = float(ncx2.sf(thresh, dof, lam)) if lam > 0 else float(alpha)
    return {"noncentrality": lam, "dof": dof, "threshold": float(thresh),
            "power": power, "alpha": float(alpha)}
