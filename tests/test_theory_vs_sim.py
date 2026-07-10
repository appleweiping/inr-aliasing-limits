r"""Theory-vs-simulation validation.

Every closed-form limit used in the paper is asserted against a Monte-Carlo experiment
here.  A theorem is not allowed into the manuscript until its test passes.
"""
import numpy as np
import pytest

from inralias.sampling import synthesis_matrix, noise_gain
from inralias.limits import (
    aliasing_bias,
    aliasing_floor,
    folded_frequency,
    silent_aliasing_ratio,
    mmse_recoverable,
    mmse_aliasing_floor,
)


def _complex_noise(rng, n, sigma):
    """Circularly-symmetric complex Gaussian with E|eps|^2 = sigma^2."""
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (sigma / np.sqrt(2))


# --------------------------------------------------------------------------------------
# Theorem 1 -- achievability: noise gain and recoverable MMSE
# --------------------------------------------------------------------------------------
def test_thm1_noise_gain_upper_bounds_every_trial():
    rng = np.random.default_rng(2)
    t = np.sort(rng.random(60))
    freqs = np.linspace(-5, 5, 7)
    Phi = synthesis_matrix(freqs, t)
    kappa = noise_gain(Phi)
    c_star = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    G = Phi.conj().T @ Phi
    Minv = np.linalg.inv(G) @ Phi.conj().T
    sigma = 0.1
    for _ in range(200):
        eps = _complex_noise(rng, t.size, sigma)
        c_hat = c_star + Minv @ eps
        # ||c_hat - c*|| <= kappa * ||eps||  (deterministic bound)
        assert np.linalg.norm(c_hat - c_star) <= kappa * np.linalg.norm(eps) + 1e-9


def test_thm1_recoverable_mmse_matches_montecarlo():
    rng = np.random.default_rng(3)
    t = np.sort(rng.random(80))
    freqs = np.linspace(-6, 6, 9)
    Phi = synthesis_matrix(freqs, t)
    G = Phi.conj().T @ Phi
    Minv = np.linalg.inv(G) @ Phi.conj().T
    sigma = 0.3
    predicted = mmse_recoverable(Phi, sigma**2)  # sigma^2 tr((Phi*Phi)^-1)
    trials = 8000
    acc = 0.0
    c_star = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    for _ in range(trials):
        eps = _complex_noise(rng, t.size, sigma)
        c_hat = c_star + Minv @ eps
        acc += np.linalg.norm(c_hat - c_star) ** 2
    empirical = acc / trials
    assert empirical == pytest.approx(predicted, rel=0.06)


# --------------------------------------------------------------------------------------
# Theorem 2 -- converse / aliasing
# --------------------------------------------------------------------------------------
def test_thm2_classical_folding_special_case():
    # Uniform sampling t_j = j/N, lowpass dictionary {0..K-1}. A tone at k0 + N is
    # out of band and MUST fold exactly to k0 (classical aliasing).
    N, K = 24, 6
    t = np.arange(N) / N
    freqs = np.arange(K).astype(float)
    for k0 in range(K):
        omega_out = k0 + N
        info = folded_frequency(omega_out, freqs, t)
        assert info["fold_freq"] == pytest.approx(float(k0))
        # and the fold is essentially loss-free onto that atom
        assert info["leakage"] == pytest.approx(1.0, rel=1e-6)


def test_thm2_bias_equals_least_squares_fit():
    # aliasing_bias (closed form) must equal the empirical LS coefficient of a noiseless
    # fit to the out-of-band samples.
    rng = np.random.default_rng(4)
    t = np.sort(rng.random(50))
    freqs = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    out_freqs = np.array([6.5, -5.2])
    out_coeffs = np.array([0.8 + 0.2j, -0.5 + 0.9j])
    dc = aliasing_bias(freqs, t, out_freqs, out_coeffs)
    Phi = synthesis_matrix(freqs, t)
    g = synthesis_matrix(out_freqs, t) @ out_coeffs
    c_ls, *_ = np.linalg.lstsq(Phi, g, rcond=None)
    assert np.allclose(dc, c_ls, atol=1e-8)


def test_thm2_floor_is_minimal_over_the_class():
    # No coefficient vector fits the noiseless out-of-band samples better than the LS floor.
    rng = np.random.default_rng(5)
    t = np.sort(rng.random(45))
    freqs = np.array([-2.0, 0.0, 1.0, 3.0])
    out_freqs = np.array([5.5])
    out_coeffs = np.array([1.0 + 0j])
    Phi = synthesis_matrix(freqs, t)
    g = synthesis_matrix(out_freqs, t) @ out_coeffs
    floor = aliasing_floor(freqs, t, out_freqs, out_coeffs)
    c_ls, *_ = np.linalg.lstsq(Phi, g, rcond=None)
    for _ in range(500):
        c_try = c_ls + 0.1 * (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size))
        err = np.linalg.norm(g - Phi @ c_try)
        assert err >= floor - 1e-9


def test_thm2_silent_aliasing_regime_exists():
    # A structured, densely-sampled setup where the visible residual is tiny but the
    # spectral bias is large: silent aliasing.
    rng = np.random.default_rng(6)
    N = 200
    t = np.sort(rng.random(N))
    freqs = np.array([-2.0, 0.0, 1.0, 3.0])
    # out-of-band tone close (in sample correlation) to atom 3.0
    out_freqs = np.array([3.0 + 1.0 / N])  # just outside the dictionary
    out_coeffs = np.array([1.0 + 0j])
    ratio = silent_aliasing_ratio(freqs, t, out_freqs, out_coeffs)
    floor = aliasing_floor(freqs, t, out_freqs, out_coeffs)
    dc = aliasing_bias(freqs, t, out_freqs, out_coeffs)
    assert floor < 0.2 * np.linalg.norm([1.0])  # samples fit well (small residual)
    assert np.linalg.norm(dc) > 0.5             # but spectrum badly biased
    assert ratio > 3.0


# --------------------------------------------------------------------------------------
# Theorem 3 -- statistical aliasing floor
# --------------------------------------------------------------------------------------
def test_thm3_aliasing_floor_matches_montecarlo():
    rng = np.random.default_rng(7)
    t = np.sort(rng.random(120))
    freqs = np.linspace(-4, 4, 5)
    out_freqs = np.array([6.3, -7.1, 9.0])
    out_power = np.array([0.5, 0.3, 0.8])
    predicted = mmse_aliasing_floor(freqs, t, out_freqs, out_power)
    Phi = synthesis_matrix(freqs, t)
    Phi_out = synthesis_matrix(out_freqs, t)
    G = Phi.conj().T @ Phi
    M = np.linalg.inv(G) @ Phi.conj().T
    trials = 6000
    acc = 0.0
    for _ in range(trials):
        a = (rng.standard_normal(out_freqs.size) + 1j * rng.standard_normal(out_freqs.size))
        a *= np.sqrt(out_power / 2)  # E|a_l|^2 = out_power_l
        g = Phi_out @ a
        dc = M @ g
        acc += np.linalg.norm(dc) ** 2
    empirical = acc / trials
    assert empirical == pytest.approx(predicted, rel=0.07)


def test_thm3_aliasing_floor_does_not_vanish_with_N():
    # Coherent out-of-band content: the bias floor stays bounded away from 0 as N grows,
    # unlike the variance term which decays like 1/N.  Under uniform rate-N sampling the
    # alias of in-band atom 0 sits at frequency N, so the out-of-band tone must TRACK N to
    # stay coherently aliased (a fixed frequency would eventually be resolved -- exactly
    # the classical fact that raising the rate removes aliasing).
    freqs = np.arange(4).astype(float)  # lowpass {0,1,2,3}
    floors = []
    variances = []
    for N in [32, 64, 128, 256]:
        t = np.arange(N) / N
        out_freqs = np.array([float(N)])  # first alias of atom 0 at sampling rate N
        Phi = synthesis_matrix(freqs, t)
        floors.append(mmse_aliasing_floor(freqs, t, out_freqs, 1.0))
        variances.append(mmse_recoverable(Phi, sigma2=1.0))
    floors = np.array(floors)
    variances = np.array(variances)
    # variance (estimation) term decays ~ m/N
    assert variances[-1] < 0.5 * variances[0]
    # aliasing-bias floor stays ~constant (unit fold onto atom 0), not decaying with N
    assert floors == pytest.approx(1.0, rel=1e-6)
