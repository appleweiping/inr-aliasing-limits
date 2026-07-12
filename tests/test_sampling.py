"""Unit tests for the sampling / frame machinery (inralias.sampling)."""
import numpy as np
import pytest

from inralias.sampling import (
    synthesis_matrix,
    gram,
    frame_bounds,
    noise_gain,
    alias_projector,
    condition_number,
    pinv_apply,
)


def test_dc_atom_is_ones():
    t = np.linspace(0, 1, 17, endpoint=False)
    Phi = synthesis_matrix(np.array([0.0]), t)
    assert np.allclose(Phi[:, 0], 1.0)


def test_uniform_integer_freqs_are_orthogonal():
    # N uniform samples, integer frequencies 0..K-1 (K<=N) -> columns orthogonal, norm sqrt(N)
    N, K = 32, 8
    t = np.arange(N) / N
    freqs = np.arange(K).astype(float)
    Phi = synthesis_matrix(freqs, t)
    G = gram(Phi)
    assert np.allclose(G, N * np.eye(K), atol=1e-9)
    A, B = frame_bounds(Phi)
    assert A == pytest.approx(N, rel=1e-9)
    assert B == pytest.approx(N, rel=1e-9)
    assert noise_gain(Phi) == pytest.approx(1 / np.sqrt(N), rel=1e-9)
    assert condition_number(Phi) == pytest.approx(1.0, rel=1e-9)


def test_alias_projector_is_orthogonal_projector():
    rng = np.random.default_rng(0)
    t = np.sort(rng.random(40))
    freqs = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    Phi = synthesis_matrix(freqs, t)
    P = alias_projector(Phi)
    assert np.allclose(P, P.conj().T, atol=1e-9)          # Hermitian
    assert np.allclose(P @ P, P, atol=1e-8)               # idempotent
    assert np.linalg.matrix_rank(P) == freqs.size         # rank = m (full column rank)
    # P fixes columns of Phi
    assert np.allclose(P @ Phi, Phi, atol=1e-8)


def test_noise_gain_matches_min_singular_value():
    rng = np.random.default_rng(1)
    t = np.sort(rng.random(50))
    freqs = np.linspace(-6, 6, 9)
    Phi = synthesis_matrix(freqs, t)
    s = np.linalg.svd(Phi, compute_uv=False)
    assert noise_gain(Phi) == pytest.approx(1 / s[-1], rel=1e-9)


def test_pinv_apply_rank_deficient_gives_min_norm():
    # ridge=0 must give the SVD pseudoinverse (minimum-norm) solution even when the
    # Gram matrix is singular -- never a solve() on a singular Gram.
    rng = np.random.default_rng(2)
    t = np.sort(rng.random(4))                    # N=4 < m=7: underdetermined
    freqs = np.linspace(-3, 3, 7)
    Phi = synthesis_matrix(freqs, t)
    y = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    c = pinv_apply(Phi, y, ridge=0.0)
    c_ref = np.linalg.pinv(Phi) @ y
    assert np.allclose(c, c_ref, atol=1e-9)
    # interpolates the samples and has minimum norm among interpolants
    assert np.allclose(Phi @ c, y, atol=1e-8)
