r"""Ground-truth-free aliasing diagnostics.

Given only the noisy samples ``(t, y)`` and the INR's representable set :math:`\Lambda`
(``freqs``), decide whether the fit is in the safe (achievability) regime or the aliasing
(converse) regime -- **without** access to the true signal.  Two complementary tests:

* :func:`extended_dictionary_test` -- fit :math:`\Lambda` versus an enlarged dictionary
  :math:`\Lambda^+=\Lambda\cup\Lambda_{\text{ring}}` of candidate frequencies just outside
  :math:`\Lambda`.  Significant recovered energy on the ring reveals out-of-band content
  (hence aliasing risk).  Catches even the *coherent* fold that is invisible to resampling.

* :func:`crossfit_aliasing_energy` -- fit on one random half of the samples, predict the
  other; prediction error beyond the noise floor signals that the data are not consistent
  with band-limitation to :math:`\Lambda`.  Sensitive under nonuniform sampling.

Honesty note: under *perfectly coherent* uniform-rate aliasing (an out-of-band tone whose
samples coincide exactly with an in-band atom) the fold is, provably, undetectable from the
samples alone -- resampling sees a consistent signal.  This is the diagnostic's own limit
and matches the fundamental (worst-case) nature of the converse; the extended-dictionary
test still flags it when a ring atom sits near the true out-of-band frequency.
"""
from __future__ import annotations

import numpy as np

from inralias.sampling import synthesis_matrix, pinv_apply

__all__ = [
    "residual_energy",
    "extended_dictionary_test",
    "crossfit_aliasing_energy",
]


def residual_energy(freqs: np.ndarray, t: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    r"""In-sample visible misfit :math:`\lVert(I-P)y\rVert^2/N` of the best :math:`\Lambda` fit."""
    Phi = synthesis_matrix(np.asarray(freqs, float), t)
    c = pinv_apply(Phi, y, ridge=ridge)
    r = y - Phi @ c
    return float(np.mean(np.abs(r) ** 2))


def extended_dictionary_test(
    freqs: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    ring: np.ndarray,
    ridge: float = 1e-6,
) -> dict:
    r"""Fit an enlarged dictionary :math:`\Lambda\cup\Lambda_{\text{ring}}` and report how much
    recovered energy lands on the ring (out-of-band) atoms.

    Returns ``{"out_of_band_frac", "ring_energy", "inband_energy", "flag",
    "underdetermined"}`` where ``out_of_band_frac`` is the fraction of recovered
    coefficient energy on ring atoms and ``flag`` is True when it exceeds a threshold
    (0.1; validated against a null-calibrated false-positive rate in
    ``experiments/run_diagnostic_roc.py``).

    **Validity requirement**: the test is only meaningful in the overdetermined regime
    ``N > |Lambda| + |ring|``.  Underdetermined ridge fits spread energy across all atoms
    and flag essentially every signal (null out-of-band fraction ~0.7); the returned
    ``underdetermined`` key marks this case and the ``flag`` should then be ignored.
    """
    freqs = np.asarray(freqs, float)
    ring = np.asarray(ring, float)
    ext = np.concatenate([freqs, ring])
    Phi = synthesis_matrix(ext, t)
    c = pinv_apply(Phi, y, ridge=ridge)
    m = freqs.size
    inb = float(np.sum(np.abs(c[:m]) ** 2))
    out = float(np.sum(np.abs(c[m:]) ** 2))
    frac = out / (inb + out + 1e-30)
    underdetermined = np.asarray(t).size <= ext.size
    return {
        "out_of_band_frac": frac,
        "ring_energy": out,
        "inband_energy": inb,
        "flag": bool(frac > 0.10),
        "underdetermined": bool(underdetermined),
    }


def crossfit_aliasing_energy(
    freqs: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    sigma2: float | None = None,
    n_splits: int = 20,
    seed: int = 0,
    ridge: float = 1e-6,
) -> dict:
    r"""Cross-fit prediction error beyond the noise floor.

    Fit :math:`\Lambda` on a random half, predict the held-out half; average the excess
    (over ``sigma2``) mean-squared prediction error across ``n_splits`` random splits.  A
    positive value indicates the samples are not consistent with band-limitation to
    :math:`\Lambda` (aliasing / misspecification).  If ``sigma2`` is None it is estimated
    from the in-sample residual of an over-complete fit.
    """
    freqs = np.asarray(freqs, float)
    t = np.asarray(t, float)
    y = np.asarray(y)
    N = t.size
    rng = np.random.default_rng(seed)

    if sigma2 is None:
        # crude noise estimate: residual of an over-complete fit (dictionary padded by a ring)
        pad = np.arange(1, 6) * (np.max(np.abs(freqs)) + 1)
        ring = np.concatenate([pad, -pad])
        sigma2 = residual_energy(np.concatenate([freqs, ring]), t, y, ridge=ridge)

    excess = []
    for _ in range(n_splits):
        idx = rng.permutation(N)
        half = N // 2
        a, b = idx[:half], idx[half:]
        Phi_a = synthesis_matrix(freqs, t[a])
        c = pinv_apply(Phi_a, y[a], ridge=ridge)
        Phi_b = synthesis_matrix(freqs, t[b])
        pred_err = np.mean(np.abs(y[b] - Phi_b @ c) ** 2)
        excess.append(pred_err - sigma2)
    excess = np.array(excess)
    val = float(max(0.0, np.mean(excess)))
    return {
        "aliasing_energy": val,
        "sigma2": float(sigma2),
        "mean_excess": float(np.mean(excess)),
        "flag": bool(val > 2.0 * sigma2 / max(1, N)),
    }
