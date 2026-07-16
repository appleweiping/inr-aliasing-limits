r"""Tests for the AliasGuard sampling-design module (Ds-optimal / null-steering sample
design + the continuum aliasability certificate)."""
import numpy as np
import pytest

from inralias.design import (
    design_metrics, visibility_of, aliasability_of, condition_number_of,
    aliasguard_continuous, aliasguard_greedy, aliasguard_continuous_nd,
    coherence_only_design, condition_only_design, e_optimal_design, fixed_jitter,
    aliasability_certificate, visibility_certificate,
)

LAMBDA = np.array([0.0, 5.0, -5.0, 11.0, -11.0, 17.0, -17.0, 23.0, -23.0])


def _focused_sets():
    concern = np.array([26.5, 33.0, 41.5, -26.5, -33.0, -41.5])
    Odes = np.sort(np.concatenate([concern + d for d in (-0.5, 0.0, 0.5)]))
    Odes = Odes[np.min(np.abs(Odes[:, None] - LAMBDA[None, :]), axis=1) > 0.3]
    Otest = np.sort(np.concatenate([concern + 0.27, concern - 0.31]))
    Otest = Otest[np.min(np.abs(Otest[:, None] - LAMBDA[None, :]), axis=1) > 0.2]
    return Odes, Otest


def test_aliasguard_beats_random_and_generalizes():
    Odes, Otest = _focused_sets()
    N = 40
    t, _ = aliasguard_continuous(LAMBDA, Odes, N, seed=0)
    ag = max(aliasability_of(LAMBDA, t, float(nu)) for nu in Otest)      # held-out
    # best of many random jitter designs (agnostic baseline)
    rng = np.random.default_rng(0)
    best_rand = min(
        max(aliasability_of(LAMBDA, np.sort((np.arange(N) + 0.5 * rng.uniform(-1, 1, N)) / N % 1.0),
                            float(nu)) for nu in Otest)
        for _ in range(20))
    assert ag < 0.6 * best_rand          # substantial held-out advantage
    assert condition_number_of(LAMBDA, t) < 1.6   # conditioning preserved


def test_joint_objective_necessary():
    # coherence-only suppresses aliasing but wrecks conditioning; condition-only keeps
    # conditioning but does not suppress aliasing; the joint AliasGuard gets both.
    Odes, Otest = _focused_sets()
    N = 28
    t_ag, _ = aliasguard_continuous(LAMBDA, Odes, N, seed=0)
    t_coh = coherence_only_design(LAMBDA, Odes, N, seed=0, n_sweeps=6, grid_res=256)
    t_cond = condition_only_design(LAMBDA, N, seed=0, n_sweeps=6, grid_res=256)
    a_ag = max(aliasability_of(LAMBDA, t_ag, float(nu)) for nu in Otest)
    a_cond = max(aliasability_of(LAMBDA, t_cond, float(nu)) for nu in Otest)
    k_ag = condition_number_of(LAMBDA, t_ag)
    k_coh = condition_number_of(LAMBDA, t_coh)
    assert a_ag < a_cond                 # AG suppresses aliasing better than cond-only
    assert k_ag < k_coh                  # AG conditions better than coherence-only


def test_greedy_grid_inherits_t1_on_coherent_folds():
    # a grid-restricted design cannot break an exact grid-coherent fold (T1): the fold
    # frequency stays exactly invisible whatever grid subset is chosen.
    Q = 256
    Ofold = np.array([w + Q for w in LAMBDA if w != 0] + [w - Q for w in LAMBDA if w != 0])
    t, _ = aliasguard_greedy(LAMBDA, Ofold, 40, Q=Q, seed=0)
    assert min(visibility_of(LAMBDA, t, float(nu)) for nu in Ofold) < 1e-8
    # but the continuous (off-grid) design breaks them
    tc, _ = aliasguard_continuous(LAMBDA, Ofold, 40, seed=0)
    assert min(visibility_of(LAMBDA, tc, float(nu)) for nu in Ofold) > 0.5


def test_certificate_is_sound():
    # certified max-aliasability >= the true max over a fine reference grid; certified
    # min-visibility <= the true min. Small band for test speed.
    N = 28
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1, N))
    band = [(25.0, 30.0)]
    ca = aliasability_certificate(LAMBDA, t, band, n_grid=4000)
    cv = visibility_certificate(LAMBDA, t, band, n_grid=4000)
    ref = np.arange(25.0, 30.0, 0.0025)
    true_a = max(aliasability_of(LAMBDA, t, float(x)) for x in ref)
    true_v = min(visibility_of(LAMBDA, t, float(x)) for x in ref)
    assert ca["certified_max_aliasability"] >= true_a - 1e-4       # sound upper bound
    assert cv["certified_min_visibility"] <= true_v + 1e-4         # sound lower bound
    # non-vacuous: certified bound within a small factor of the grid max
    assert ca["certified_max_aliasability"] <= ca["grid_max_aliasability"] + 0.15


def test_2d_design_beats_random():
    LAM2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [2, 1], [-2, -1]], float)
    O2 = np.array([[8, 0], [0, 8], [8, 8], [-8, 0], [6, 3], [-6, -3]], float)
    N = 28
    tr = np.random.default_rng(0).uniform(0, 1, (N, 2))
    mr = design_metrics(LAM2, O2, tr)
    _t, m2 = aliasguard_continuous_nd(LAM2, O2, N, d=2, n_sweeps=4, grid_res=28, seed=0)
    assert m2["max_aliasability"] < 0.7 * mr["max_aliasability"]
    assert m2["min_visibility"] > mr["min_visibility"]
