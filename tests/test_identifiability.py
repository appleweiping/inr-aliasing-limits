r"""Monte-Carlo validation of the identifiability theory core (Theorems T1--T4).

Each stated closed form / bound in :mod:`inralias.identifiability` is checked against
direct simulation here before it is used in the manuscript.
"""
import numpy as np
import pytest

from inralias.identifiability import (
    sample_atom,
    visibility,
    aliasability,
    alias_coefficients,
    grid_equivalence_class,
    exactly_indistinguishable,
    continuous_gram,
    riesz_bounds,
    function_error_decomposition,
    expected_jitter_coherence,
    aliasability_concentration_bound,
    residual_test_power,
)
from inralias.sampling import synthesis_matrix, pinv_apply
from inralias.inr import real_design

LAMBDA = np.array([0.0, 5.0, -5.0, 11.0, -11.0, 17.0, -17.0, 23.0, -23.0])


# --------------------------------------------------------------------------------------
# T1: visibility, aliasability, exact equivalence on grids
# --------------------------------------------------------------------------------------
def test_t1_grid_coherent_tone_has_zero_visibility_on_every_subset():
    Q = 64
    rng = np.random.default_rng(0)
    for N in (16, 40, 60):
        t = np.sort(rng.choice(Q, size=N, replace=False)) / Q
        # 47 = -17 (mod 64): exact equivalence class of an in-band atom
        assert grid_equivalence_class(47, Q) == grid_equivalence_class(-17, Q)
        assert visibility(LAMBDA, t, 47.0) < 1e-8
        # and the alias pattern is exactly the unit vector on the -17 atom
        c = alias_coefficients(LAMBDA, t, 47.0)
        k = int(np.where(LAMBDA == -17.0)[0][0])
        e = np.zeros(LAMBDA.size, complex); e[k] = 1.0
        assert np.allclose(c, e, atol=1e-8)
        assert aliasability(LAMBDA, t, 47.0) == pytest.approx(1.0, abs=1e-8)


def test_t1_noncongruent_tone_is_visible_on_full_rank_subset():
    Q = 64
    rng = np.random.default_rng(1)
    t = np.sort(rng.choice(Q, size=48, replace=False)) / Q
    # 40 is congruent to none of LAMBDA mod 64 -> visible (extended matrix full rank)
    ext = np.column_stack([synthesis_matrix(LAMBDA, t), sample_atom(40.0, t)])
    assert np.linalg.matrix_rank(ext) == LAMBDA.size + 1
    assert visibility(LAMBDA, t, 40.0) > 0.05


def test_t1_visibility_zero_iff_atom_in_range():
    # v_T(nu) = 0 <=> phi_nu in range(Phi): both directions on explicit examples
    rng = np.random.default_rng(2)
    t = np.sort(rng.random(50))
    freqs = np.array([-2.0, 0.0, 3.0])
    # an atom OF the dictionary is trivially in range
    assert visibility(freqs, t, 3.0) < 1e-10
    # a generic off-dictionary tone on iid samples is not
    assert visibility(freqs, t, 7.3) > 0.1


def test_t1_lecam_two_point_lower_bound():
    # Exactly indistinguishable pair => ANY sample-based estimator returns the same
    # function for both, so its worst-case L2 error >= ||f1 - f2||/2.
    Q, N = 64, 48
    rng = np.random.default_rng(3)
    t = np.sort(rng.choice(Q, size=N, replace=False)) / Q
    a = 0.7
    f1 = (np.array([47.0, -47.0]), np.array([a / 2, a / 2]).astype(complex))
    f2 = (np.array([-17.0, 17.0]), np.array([a / 2, a / 2]).astype(complex))
    assert exactly_indistinguishable(*f1, *f2, t)
    # distance between the two hypotheses in L2 (both Hermitian pairs, integer freqs)
    allf = np.concatenate([f1[0], f2[0]])
    z = np.concatenate([f1[1], -f2[1]])
    dist_sq = float(np.real(z.conj() @ continuous_gram(allf) @ z))
    # the LS estimator (as one instance of "any estimator") outputs identical fits
    y1 = synthesis_matrix(f1[0], t) @ f1[1]
    y2 = synthesis_matrix(f2[0], t) @ f2[1]
    c1 = pinv_apply(synthesis_matrix(LAMBDA, t), y1)
    c2 = pinv_apply(synthesis_matrix(LAMBDA, t), y2)
    assert np.allclose(c1, c2, atol=1e-8)
    # so max over the two of its L2 error is >= half distance
    e1 = function_error_decomposition(LAMBDA, c1, np.zeros(LAMBDA.size), *f1)["total_rmse"]
    e2 = function_error_decomposition(LAMBDA, c2, np.zeros(LAMBDA.size), *f2)["total_rmse"]
    assert max(e1, e2) >= 0.5 * np.sqrt(dist_sq) - 1e-9


# --------------------------------------------------------------------------------------
# T2: continuous Gram, Riesz bounds, function-space decomposition
# --------------------------------------------------------------------------------------
def test_t2_continuous_gram_matches_numerical_integration():
    freqs = np.array([-2.5, 0.0, 1.0, 3.3])
    G = continuous_gram(freqs)
    tt = np.arange(200000) / 200000
    E = np.exp(1j * 2 * np.pi * np.outer(tt, freqs))
    G_num = E.conj().T @ E / tt.size
    assert np.allclose(G, G_num, atol=2e-4)


def test_t2_riesz_bounds_sandwich_function_norm():
    rng = np.random.default_rng(4)
    freqs = np.array([-4.4, -1.0, 0.0, 2.2, 3.0])
    lo, hi = riesz_bounds(freqs)
    assert 0 < lo <= hi
    tt = np.arange(100000) / 100000
    E = np.exp(1j * 2 * np.pi * np.outer(tt, freqs))
    for _ in range(20):
        c = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
        f_sq = float(np.mean(np.abs(E @ c) ** 2))
        n_sq = float(np.linalg.norm(c) ** 2)
        assert lo * n_sq - 1e-6 <= f_sq <= hi * n_sq + 1e-6


def test_t2_function_error_decomposition_matches_dense_grid():
    rng = np.random.default_rng(5)
    freqs = np.array([-3.0, 0.0, 2.0, 5.5])       # deliberately non-orthogonal (5.5)
    out_freqs = np.array([8.3, -7.0])
    c_star = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    c_hat = c_star + 0.3 * (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size))
    a = 0.5 * (rng.standard_normal(2) + 1j * rng.standard_normal(2))
    dec = function_error_decomposition(freqs, c_hat, c_star, out_freqs, a)
    tt = np.arange(400000) / 400000
    f_hat = np.exp(1j * 2 * np.pi * np.outer(tt, freqs)) @ c_hat
    f_true = (np.exp(1j * 2 * np.pi * np.outer(tt, freqs)) @ c_star
              + np.exp(1j * 2 * np.pi * np.outer(tt, out_freqs)) @ a)
    mse_num = float(np.mean(np.abs(f_hat - f_true) ** 2))
    assert dec["total_sq"] == pytest.approx(mse_num, rel=1e-3)
    # components sum exactly
    assert dec["total_sq"] == pytest.approx(
        dec["modeled_component_sq"] + dec["truncation_sq"] + dec["cross"], rel=1e-12
    )


# --------------------------------------------------------------------------------------
# T3: jitter damping and iid concentration
# --------------------------------------------------------------------------------------
def test_t3_jitter_coherence_matches_characteristic_function():
    rng = np.random.default_rng(6)
    Q, N, k = 64, 4000, 1                       # nu = omega + kQ
    g = rng.integers(0, Q, size=N)
    for dist, scale in (("gaussian", 0.3 / Q), ("uniform", 0.8 / Q)):
        if dist == "gaussian":
            eta = rng.normal(0, scale, N)
        else:
            eta = rng.uniform(-scale / 2, scale / 2, N)
        t = (g / Q + eta)
        emp = abs(np.mean(np.exp(1j * 2 * np.pi * (k * Q) * t)))
        pred = abs(expected_jitter_coherence(k, Q, scale, dist))
        assert emp == pytest.approx(pred, abs=0.03)


def test_t3_small_jitter_visibility_law():
    # v ~ sqrt(1 - chi^2) ~ 2 pi |k| Q sigma_t for small Gaussian jitter
    rng = np.random.default_rng(7)
    Q, N, k = 64, 2000, 1
    g = rng.integers(0, Q, size=N)
    freqs = LAMBDA
    nu = -17.0 + k * Q                          # grid-coherent tone
    for sigma_t in (2e-4, 5e-4, 1e-3):
        t = g / Q + rng.normal(0, sigma_t, N)
        v = visibility(freqs, t, nu)
        pred = 2 * np.pi * abs(k) * Q * sigma_t
        assert v == pytest.approx(pred, rel=0.25)


def test_t3_iid_concentration_bound_holds_and_shrinks():
    rng = np.random.default_rng(8)
    freqs = np.arange(-4, 5).astype(float)
    m = freqs.size
    cands = np.arange(10, 41).astype(float)     # K = 31 integer out-of-band candidates
    K = cands.size
    prev_emp = None
    for N in (6400, 25600):
        bound = aliasability_concentration_bound(m, K, N, delta=0.05, lam_min=1.0)
        assert np.isfinite(bound)
        emp = max(
            aliasability(freqs, np.sort(rng.uniform(0, 1, N)), nu) for nu in cands
        )
        assert emp <= bound
        if prev_emp is not None:
            # empirical worst-case aliasability shrinks roughly like 1/sqrt(N)
            assert emp < prev_emp
        prev_emp = emp
    # bound is vacuous (inf) at tiny N -- by design, never silently wrong
    assert np.isinf(aliasability_concentration_bound(m, K, 50, delta=0.05, lam_min=1.0))


# --------------------------------------------------------------------------------------
# T4: detection power and impossibility
# --------------------------------------------------------------------------------------
def test_t4_residual_test_power_matches_montecarlo():
    rng = np.random.default_rng(9)
    N = 64
    t = np.sort(rng.uniform(0, 1, N))
    freqs = np.arange(-2, 3).astype(float)
    D, _ = real_design(freqs, t)
    amp, nu, sigma, alpha = 0.35, 7.5, 0.15, 0.05
    s = amp * np.cos(2 * np.pi * nu * t + 0.7)
    res = residual_test_power(D, s, sigma, alpha=alpha)
    # Monte-Carlo rejection rate
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    r = int(np.sum(sv > 1e-10)); Ur = U[:, :r]
    from scipy.stats import chi2
    thresh = chi2.ppf(1 - alpha, N - r)
    rej = 0
    trials = 4000
    for _ in range(trials):
        y = s + rng.normal(0, sigma, N)
        resid = y - Ur @ (Ur.T @ y)
        if np.dot(resid, resid) / sigma**2 > thresh:
            rej += 1
    assert rej / trials == pytest.approx(res["power"], abs=0.03)


def test_t4_exactly_coherent_component_is_undetectable():
    # A grid-coherent component contributes zero noncentrality: the residual test's power
    # equals its size (alpha) -- the impossibility statement made quantitative.
    rng = np.random.default_rng(10)
    Q, N = 64, 48
    t = np.sort(rng.choice(Q, size=N, replace=False)) / Q
    D, _ = real_design(LAMBDA, t)
    s = 0.8 * np.cos(2 * np.pi * 47.0 * t + 0.2)   # 47 = -17 (mod 64)
    res = residual_test_power(D, s, sigma=0.05, alpha=0.05)
    assert res["noncentrality"] < 1e-16
    assert res["power"] == pytest.approx(0.05, abs=1e-12)
