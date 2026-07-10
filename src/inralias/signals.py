r"""Signal generators and sampling patterns.

Utilities to build the synthetic in-band / out-of-band signals used in the phase-transition
experiments, the nonuniform sampling patterns, and helpers to characterise a real signal's
spectrum relative to an INR's representable set :math:`\Lambda`.

A signal is represented spectrally as a set of (frequency, complex-amplitude) atoms; the
time-domain waveform is the real part of the complex-exponential sum, so everything is
consistent with :func:`inralias.sampling.synthesis_matrix`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "lowpass_dictionary",
    "structured_dictionary",
    "nonuniform_times",
    "evaluate",
    "random_inband",
    "out_of_band_atoms",
    "spectrum_energy_split",
]


def lowpass_dictionary(bandwidth: int) -> np.ndarray:
    """Integer lowpass dictionary ``{-B,...,-1,0,1,...,B}`` (size ``2B+1``)."""
    return np.arange(-bandwidth, bandwidth + 1).astype(float)


def structured_dictionary(base: float, m: int, rng=None, jitter: float = 0.0) -> np.ndarray:
    r"""A *structured* (non-interval) dictionary: ``m`` harmonics of a base frequency with
    optional jitter, modelling the integer-combination frequency atoms of a Fourier-feature
    INR (Yüce et al. 2021).  Symmetrised to be real-signal compatible."""
    k = np.arange(1, m + 1)
    f = base * k
    if jitter and rng is not None:
        f = f + rng.normal(0, jitter, size=f.shape)
    return np.concatenate([-f[::-1], [0.0], f])


def nonuniform_times(N: int, rng, kind: str = "jitter", span: float = 1.0) -> np.ndarray:
    """Sample locations in ``[0, span)``.

    ``kind`` = ``uniform`` | ``jitter`` (uniform grid + Gaussian jitter) | ``random``
    (i.i.d. uniform) | ``poisson`` (normalised exponential gaps).
    """
    if kind == "uniform":
        t = np.arange(N) / N * span
    elif kind == "jitter":
        base = np.arange(N) / N * span
        t = base + rng.normal(0, 0.25 * span / N, size=N)
        t = np.mod(t, span)
    elif kind == "random":
        t = rng.random(N) * span
    elif kind == "poisson":
        gaps = rng.exponential(1.0, size=N)
        t = np.cumsum(gaps)
        t = (t - t.min()) / (t.max() - t.min()) * span * (1 - 1e-6)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return np.sort(t)


def evaluate(freqs: np.ndarray, coeffs: np.ndarray, t: np.ndarray, real: bool = False) -> np.ndarray:
    """Evaluate the spectral signal at times ``t`` (complex, or real part if ``real``)."""
    from inralias.sampling import synthesis_matrix

    v = synthesis_matrix(np.asarray(freqs, float), t) @ np.asarray(coeffs, complex).reshape(-1)
    return v.real if real else v


def random_inband(Lambda: np.ndarray, rng, power: float = 1.0, hermitian: bool = True):
    r"""Random in-band coefficients on ``Lambda`` with total power ``power``.

    If ``hermitian`` the coefficients satisfy :math:`c_{-\omega}=\overline{c_\omega}` so the
    time-domain signal is real.
    """
    Lambda = np.asarray(Lambda, float)
    m = Lambda.size
    c = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2)
    if hermitian:
        order = np.argsort(Lambda)
        inv = {}
        for idx in order:
            w = Lambda[idx]
            j = int(np.argmin(np.abs(Lambda + w)))  # index of -w
            if w > 0:
                inv[idx] = j
        for i, j in inv.items():
            c[j] = np.conj(c[i])
        zero = np.isclose(Lambda, 0.0)
        c[zero] = c[zero].real
    c = c / np.linalg.norm(c) * np.sqrt(power)
    return c


def out_of_band_atoms(Lambda: np.ndarray, ratio: float, n_atoms: int, rng,
                      power: float = 1.0):
    r"""Out-of-band atoms at frequencies just beyond ``Lambda`` scaled by ``ratio``.

    ``ratio = B_signal / B_INR``: ``ratio<=1`` returns empty (fully in band); ``ratio>1``
    places ``n_atoms`` frequencies in ``(B_INR, ratio*B_INR]`` (Hermitian-symmetrised).
    Returns ``(freqs, coeffs)``.
    """
    B = float(np.max(np.abs(Lambda)))
    if ratio <= 1.0:
        return np.zeros(0), np.zeros(0, complex)
    hi = ratio * B
    pos = rng.uniform(B + 1e-6, hi, size=n_atoms)
    freqs = np.concatenate([pos, -pos])
    a = (rng.standard_normal(n_atoms) + 1j * rng.standard_normal(n_atoms)) / np.sqrt(2)
    coeffs = np.concatenate([a, np.conj(a)])
    coeffs = coeffs / np.linalg.norm(coeffs) * np.sqrt(power)
    return freqs, coeffs


def spectrum_energy_split(freqs: np.ndarray, coeffs: np.ndarray, B_inr: float):
    """Split spectral energy into in-band (|f|<=B_inr) and out-of-band fractions."""
    freqs = np.asarray(freqs, float)
    e = np.abs(np.asarray(coeffs, complex)) ** 2
    inb = e[np.abs(freqs) <= B_inr + 1e-9].sum()
    out = e[np.abs(freqs) > B_inr + 1e-9].sum()
    tot = inb + out + 1e-30
    return {"in_band": float(inb / tot), "out_of_band": float(out / tot)}
