"""Tests for the fixed-feature INR, bandwidth matching, and the aliasing diagnostics."""
import numpy as np
import pytest

from inralias.signals import lowpass_dictionary, nonuniform_times, evaluate, random_inband, out_of_band_atoms
from inralias.inr import FixedFeatureINR, bandwidth_matched_freqs
from inralias.diagnostics import extended_dictionary_test, crossfit_aliasing_energy


def test_fixed_feature_inr_recovers_in_band_signal():
    rng = np.random.default_rng(0)
    Lam = lowpass_dictionary(6)
    t = nonuniform_times(200, rng, "jitter")
    c = random_inband(Lam, rng, power=1.0)
    y = evaluate(Lam, c, t, real=True) + rng.normal(0, 0.02, t.size)
    inr = FixedFeatureINR(Lam, real=True).fit(t, y, ridge=1e-9)
    tt = np.linspace(0, 1, 2000, endpoint=False)
    rmse = np.sqrt(np.mean((inr.predict(tt) - evaluate(Lam, c, tt, real=True)) ** 2))
    assert rmse < 0.05  # recovered to ~noise level


def test_bandwidth_matched_covers_true_band():
    rng = np.random.default_rng(1)
    Lam = lowpass_dictionary(5)
    t = nonuniform_times(300, rng, "jitter")
    c = random_inband(Lam, rng, power=1.0)
    y = evaluate(Lam, c, t, real=True) + rng.normal(0, 0.01, t.size)
    bm = bandwidth_matched_freqs(t, y, max_bandwidth=25, energy_keep=0.999)
    assert bm.max() >= 5     # covers the true band
    assert bm.max() <= 15    # but not absurdly wide


def test_extended_dictionary_test_flags_out_of_band():
    rng = np.random.default_rng(2)
    Lam = lowpass_dictionary(6)
    t = nonuniform_times(200, rng, "jitter")
    c = random_inband(Lam, rng, power=1.0)
    ring = np.concatenate([np.arange(7, 18), -np.arange(7, 18)]).astype(float)

    y_clean = evaluate(Lam, c, t, real=True) + rng.normal(0, 0.02, t.size)
    of, oc = out_of_band_atoms(Lam, ratio=2.0, n_atoms=4, rng=rng, power=0.7)
    y_alias = y_clean + evaluate(of, oc, t, real=True)

    d_clean = extended_dictionary_test(Lam, t, y_clean, ring)
    d_alias = extended_dictionary_test(Lam, t, y_alias, ring)
    assert not d_clean["flag"]
    assert d_alias["flag"]
    assert d_alias["out_of_band_frac"] > d_clean["out_of_band_frac"]


def test_crossfit_flags_out_of_band_under_nonuniform_sampling():
    rng = np.random.default_rng(3)
    Lam = lowpass_dictionary(6)
    t = nonuniform_times(240, rng, "random")  # nonuniform -> crossfit is sensitive
    c = random_inband(Lam, rng, power=1.0)
    y_clean = evaluate(Lam, c, t, real=True) + rng.normal(0, 0.02, t.size)
    of, oc = out_of_band_atoms(Lam, ratio=2.5, n_atoms=5, rng=rng, power=1.0)
    y_alias = y_clean + evaluate(of, oc, t, real=True)

    e_clean = crossfit_aliasing_energy(Lam, t, y_clean, sigma2=0.02**2, seed=1)
    e_alias = crossfit_aliasing_energy(Lam, t, y_alias, sigma2=0.02**2, seed=1)
    assert e_alias["aliasing_energy"] > 5 * (e_clean["aliasing_energy"] + 1e-9)
