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
    ls_coefficient_mse,
    aliasing_variance_term,
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
    predicted = ls_coefficient_mse(Phi, sigma**2)  # sigma^2 tr((Phi*Phi)^-1)
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


def test_thm2_classical_folding_symmetric_band_and_higher_aliases():
    # Remark 1 in congruence-class form: for the two-sided band {-B..B} the fold target is
    # the unique Lambda atom CONGRUENT to nu (mod N) -- not the arithmetic residue in
    # [0, N).  Covers negative folds and second-order aliases.
    N, B = 24, 3
    t = np.arange(N) / N
    freqs = np.arange(-B, B + 1).astype(float)
    cases = [
        (21.0, -3.0),   # 21 = -3 (mod 24): folds to the NEGATIVE atom, not 21 mod 24 = 21
        (26.0, 2.0),    # 26 = 2 (mod 24)
        (45.0, -3.0),   # second alias: 45 = -3 (mod 48 -> mod 24 twice)
        (-27.0, -3.0),  # negative out-of-band frequency
    ]
    for omega_out, target in cases:
        info = folded_frequency(omega_out, freqs, t)
        assert info["fold_freq"] == pytest.approx(target)
        assert info["leakage"] == pytest.approx(1.0, rel=1e-6)
    # and when the residue class falls in the gap (B < |residue|), nothing folds at all
    info = folded_frequency(10.0, freqs, t)  # 10 mod 24 = 10, not congruent to any atom
    assert np.linalg.norm(info["bias"]) < 1e-10


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


def test_thm2_exact_grid_fold_silent_aliasing():
    # Theorem 2(iii), headline form: if the samples lie on a rate-Q grid and the
    # out-of-band tone satisfies nu = omega_k (mod Q), then phi_nu = phi_{omega_k}
    # EXACTLY on the samples.  The residual is exactly zero, the spectral bias equals
    # the full out-of-band coefficient energy, and the L2[0,1) reconstruction error is
    # bounded below by ||f_out|| -- silent aliasing with nu arbitrarily far from Lambda,
    # at any N >> m, under nonuniform (grid-subset) sampling.
    from inralias.limits import reconstruction_error

    rng = np.random.default_rng(11)
    Q = 64
    freqs = np.array([0.0, 5.0, -5.0, 11.0, -11.0, 17.0, -17.0, 23.0, -23.0])
    m = freqs.size
    N = 48                                     # N >> m, but N < Q: nonuniform subset
    t = np.sort(rng.choice(Q, size=N, replace=False)) / Q
    a = 0.7
    phase = 0.9
    # real tone a*cos(2*pi*47*t + phase): pair +-47, and 47 = -17 (mod 64)
    out_freqs = np.array([47.0, -47.0])
    out_coeffs = np.array([a / 2 * np.exp(1j * phase), a / 2 * np.exp(-1j * phase)])

    # residual exactly zero (up to float): the fold is invisible in-sample
    floor = aliasing_floor(freqs, t, out_freqs, out_coeffs)
    assert floor < 1e-9
    # spectral bias carries the full out-of-band energy onto the +-17 atoms
    dc = aliasing_bias(freqs, t, out_freqs, out_coeffs)
    assert np.linalg.norm(dc) == pytest.approx(a / np.sqrt(2), rel=1e-6)
    k_neg17 = int(np.where(freqs == -17.0)[0][0])
    k_pos17 = int(np.where(freqs == 17.0)[0][0])
    assert abs(dc[k_neg17]) == pytest.approx(a / 2, rel=1e-6)
    assert abs(dc[k_pos17]) == pytest.approx(a / 2, rel=1e-6)
    # single-tone fold law identifies the grid alias
    info = folded_frequency(47.0, freqs, t)
    assert info["fold_freq"] == pytest.approx(-17.0)
    assert info["leakage"] == pytest.approx(1.0, rel=1e-6)
    # L2 reconstruction error >= ||f_out||: mean-square error at least a^2/2, because the
    # fit lives in M_Lambda and e_{+-47} is orthogonal to every Lambda atom on [0,1)
    t_eval = np.arange(4096) / 4096
    rec = reconstruction_error(freqs, t, out_freqs, out_coeffs, t_eval)
    assert rec["sample_residual"] < 1e-9
    assert rec["global_mse"] >= 0.99 * a**2 / 2


def test_thm2_near_collision_limit_basis_mismatch():
    # The nu -> omega_k limit (basis-mismatch regime): residual -> 0 ~linearly in
    # eps = nu - omega_k while ||Delta c|| -> |a|.  (In this limit the reconstruction
    # error also vanishes -- the exact-grid-fold test above is the harmful regime.)
    rng = np.random.default_rng(6)
    t = np.sort(rng.random(200))
    freqs = np.array([-2.0, 0.0, 1.0, 3.0])
    a = 0.7 + 0j
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    floors, biases = [], []
    for eps in eps_list:
        out_freqs = np.array([3.0 + eps])
        floors.append(aliasing_floor(freqs, t, out_freqs, np.array([a])))
        biases.append(np.linalg.norm(aliasing_bias(freqs, t, out_freqs, np.array([a]))))
    floors = np.array(floors)
    # residual -> 0, monotonically, ~linear in eps (each decade drops the floor ~10x)
    assert np.all(np.diff(floors) < 0)
    assert np.all(floors[1:] / floors[:-1] == pytest.approx(0.1, rel=0.15))
    assert floors[-1] < 1e-3
    # ||Delta c|| -> |a|
    assert biases[-1] == pytest.approx(abs(a), rel=1e-4)


# --------------------------------------------------------------------------------------
# Theorem 3 -- statistical aliasing floor
# --------------------------------------------------------------------------------------
def test_thm3_aliasing_floor_matches_montecarlo():
    rng = np.random.default_rng(7)
    t = np.sort(rng.random(120))
    freqs = np.linspace(-4, 4, 5)
    out_freqs = np.array([6.3, -7.1, 9.0])
    out_power = np.array([0.5, 0.3, 0.8])
    predicted = aliasing_variance_term(freqs, t, out_freqs, out_power)
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
    # WORST-CASE (per-N) persistence: at every sampling density there is out-of-band
    # content that folds coherently and keeps the aliasing-variance term at full power.
    # Under uniform rate-N sampling the alias of in-band atom 0 sits at frequency N, so
    # the worst-case tone TRACKS N (a fixed frequency is eventually resolved -- exactly
    # the classical fact that raising the rate removes aliasing; the per-N worst case is
    # what no density removes).
    freqs = np.arange(4).astype(float)  # lowpass {0,1,2,3}
    floors = []
    variances = []
    for N in [32, 64, 128, 256]:
        t = np.arange(N) / N
        out_freqs = np.array([float(N)])  # first alias of atom 0 at sampling rate N
        Phi = synthesis_matrix(freqs, t)
        floors.append(aliasing_variance_term(freqs, t, out_freqs, 1.0))
        variances.append(ls_coefficient_mse(Phi, sigma2=1.0))
    floors = np.array(floors)
    variances = np.array(variances)
    # variance (estimation) term decays ~ m/N
    assert variances[-1] < 0.5 * variances[0]
    # aliasing-bias floor stays ~constant (unit fold onto atom 0), not decaying with N
    assert floors == pytest.approx(1.0, rel=1e-6)


def test_thm3_fixed_tone_regimes():
    # For a FIXED out-of-band tone the aliasing-variance term converges to the tone's
    # window-leakage energy onto span(M_Lambda):
    #   (i)  integer-disjoint tone, uniform sampling: leakage 0, term EXACTLY zero;
    #   (ii) integer-disjoint tone, iid random sampling: decays like m/N (slope ~ -1);
    #   (iii) non-integer tone, uniform sampling: converges to the POSITIVE constant
    #         sum_k sinc^2(nu - omega_k) -- it never vanishes.
    freqs = np.arange(-4, 5).astype(float)
    m = freqs.size
    # (i) exactly zero
    for N in [64, 256, 1024]:
        t = np.arange(N) / N
        assert aliasing_variance_term(freqs, t, np.array([7.0]), 1.0) < 1e-20
    # (ii) m/N law under iid sampling
    rng = np.random.default_rng(1)
    Ns = np.array([128, 512, 2048])
    means = []
    for N in Ns:
        vals = [
            aliasing_variance_term(freqs, np.sort(rng.uniform(0, 1, N)), np.array([7.0]), 1.0)
            for _ in range(30)
        ]
        means.append(float(np.mean(vals)))
    means = np.array(means)
    slope = np.polyfit(np.log(Ns), np.log(means), 1)[0]
    assert -1.35 < slope < -0.75
    assert 0.5 * m / Ns[-1] < means[-1] < 2.0 * m / Ns[-1]
    # (iii) window-leakage limit for a non-integer tone under uniform sampling
    nu = 7.3
    leak = float(np.sum(np.sinc(nu - freqs) ** 2))  # np.sinc(x) = sin(pi x)/(pi x)
    t = np.arange(4096) / 4096
    val = aliasing_variance_term(freqs, t, np.array([nu]), 1.0)
    assert val == pytest.approx(leak, rel=1e-2)


def test_thm3_joint_decomposition_matches_montecarlo():
    # Eq. (1) as a JOINT identity: with independent noise and random out-of-band
    # coefficients drawn simultaneously, the total MSE equals the sum of the
    # estimation-variance and aliasing-variance terms (cross terms vanish).
    rng = np.random.default_rng(8)
    t = np.sort(rng.random(120))
    freqs = np.linspace(-4, 4, 5)
    out_freqs = np.array([6.3, -7.1, 9.0])
    out_power = np.array([0.5, 0.3, 0.8])
    sigma = 0.3
    Phi = synthesis_matrix(freqs, t)
    Phi_out = synthesis_matrix(out_freqs, t)
    G = Phi.conj().T @ Phi
    M = np.linalg.inv(G) @ Phi.conj().T
    predicted = ls_coefficient_mse(Phi, sigma**2) + aliasing_variance_term(
        freqs, t, out_freqs, out_power
    )
    trials = 6000
    acc = 0.0
    for _ in range(trials):
        a = rng.standard_normal(out_freqs.size) + 1j * rng.standard_normal(out_freqs.size)
        a *= np.sqrt(out_power / 2)  # E|a_l|^2 = out_power_l
        eps = _complex_noise(rng, t.size, sigma)
        dc = M @ (Phi_out @ a + eps)
        acc += np.linalg.norm(dc) ** 2
    empirical = acc / trials
    assert empirical == pytest.approx(predicted, rel=0.07)
