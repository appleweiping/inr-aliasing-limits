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


def test_certificate_is_sound_both_metrics():
    # certified max-aliasability >= the true max over a fine reference grid, for BOTH the
    # coefficient-norm and the function-space (L2) metric; certified min-visibility <= true.
    from inralias.design import aliasability_L2_of
    N = 28
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1, N))
    band = [(25.0, 30.0)]
    ref = np.arange(25.0, 30.0, 0.0025)
    for metric, truth in (("coeff", aliasability_of), ("l2", aliasability_L2_of)):
        ca = aliasability_certificate(LAMBDA, t, band, n_grid=4000, metric=metric)
        true_a = max(truth(LAMBDA, t, float(x)) for x in ref)
        assert ca["full_rank"]
        assert ca["certified_max_aliasability"] >= true_a - 1e-4          # sound upper bound
        assert ca["certified_max_aliasability"] <= ca["grid_max_aliasability"] + 0.15  # tight
        assert ca["sigma_min"] > 0 and np.isfinite(ca["condition_number"])
    cv = visibility_certificate(LAMBDA, t, band, n_grid=4000)
    true_v = min(visibility_of(LAMBDA, t, float(x)) for x in ref)
    assert cv["certified_min_visibility"] <= true_v + 1e-4


def test_certificate_applies_to_any_design():
    # the certificate is a deterministic post-hoc guarantee for ANY realized design --
    # random, low-discrepancy, or optimized -- not only AliasGuard.
    from inralias.design import fixed_jitter
    band = [(25.0, 32.0)]
    for t in (np.sort(np.random.default_rng(3).uniform(0, 1, 30)), fixed_jitter(30)):
        c = aliasability_certificate(LAMBDA, t, band, n_grid=3000)
        assert c["full_rank"] and np.isfinite(c["certified_max_aliasability"])


def test_certificate_vacuous_when_rank_deficient():
    # N < m (underdetermined) and duplicated samples -> rank deficient -> vacuous certificate
    band = [(25.0, 30.0)]
    t_small = np.sort(np.random.default_rng(2).uniform(0, 1, 5))       # N=5 < m=9
    c = aliasability_certificate(LAMBDA, t_small, band, metric="l2")
    assert not c["full_rank"]
    assert c["certified_max_aliasability"] == float("inf")
    cv = visibility_certificate(LAMBDA, t_small, band)
    assert not cv["full_rank"] and cv["certified_min_visibility"] == 0.0
    # duplicated samples (exactly singular Gram) also vacuous
    t_dup = np.concatenate([t_small, t_small, t_small[:2]])            # N=12 but rank<=5
    c2 = aliasability_certificate(LAMBDA, t_dup, band)
    assert not c2["full_rank"]


def test_certificate_scale_and_near_singular():
    # near-singular design (clustered samples) -> large but finite condition number, and the
    # certificate is still sound where it is non-vacuous; a wide band stays sound.
    t = np.sort(np.concatenate([np.random.default_rng(5).uniform(0.0, 0.25, 34)]))
    band = [(25.0, 45.0)]                                              # wide band (scale)
    c = aliasability_certificate(LAMBDA, t, band, n_grid=2500, metric="l2")
    if c["full_rank"]:
        ref = np.arange(25.0, 45.0, 0.01)
        from inralias.design import aliasability_L2_of
        true_a = max(aliasability_L2_of(LAMBDA, t, float(x)) for x in ref)
        assert c["certified_max_aliasability"] >= true_a - 1e-3
    assert c["condition_number"] >= 1.0


def test_l2_equals_coeff_for_integer_dictionary():
    from inralias.design import aliasability_L2_of
    t = np.sort(np.random.default_rng(7).uniform(0, 1, 30))
    for nu in (30.4, 41.2, -27.3):
        assert aliasability_L2_of(LAMBDA, t, nu) == pytest.approx(
            aliasability_of(LAMBDA, t, nu), abs=1e-9)   # integer LAMBDA is orthonormal
    # non-integer dictionary: the two differ
    lam_nonint = np.array([0.0, 4.6, -4.6, 9.3, -9.3, 15.1, -15.1])
    d = abs(aliasability_L2_of(lam_nonint, t, 30.4) - aliasability_of(lam_nonint, t, 30.4))
    assert d > 1e-4


def test_2d_design_beats_random():
    LAM2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [2, 1], [-2, -1]], float)
    O2 = np.array([[8, 0], [0, 8], [8, 8], [-8, 0], [6, 3], [-6, -3]], float)
    N = 28
    tr = np.random.default_rng(0).uniform(0, 1, (N, 2))
    mr = design_metrics(LAM2, O2, tr)
    _t, m2 = aliasguard_continuous_nd(LAM2, O2, N, d=2, n_sweeps=4, grid_res=28, seed=0)
    assert m2["max_aliasability"] < 0.7 * mr["max_aliasability"]
    assert m2["min_visibility"] > mr["min_visibility"]


def test_u2_frankwolfe_convex_certificate():
    """Frank-Wolfe on the convex design-measure surrogate: PSD Hessian, monotone decrease,
    and a duality gap that -> 0 (certified eps-optimality). Its optimum lower-bounds the
    coordinate-descent design's surrogate value (global vs local)."""
    import numpy as np
    from inralias.design import (aliasguard_frankwolfe, surrogate_hessian,
                                 aliasguard_continuous, surrogate_objective)
    Lam = np.array([0, 3, -3, 7, -7], float)
    Om = np.array([28.5, -28.5, 33.5, -33.5, 40.5, -40.5], float)
    grid = np.arange(400) / 400.0
    Q = surrogate_hessian(Lam, Om, grid)
    assert float(np.linalg.eigvalsh(Q)[0]) >= -1e-8                 # PSD
    w, info = aliasguard_frankwolfe(Lam, Om, grid, beta=1.0, n_iter=2000)
    J = info["J"]
    assert all(J[i] >= J[i + 1] - 1e-10 for i in range(len(J) - 1))  # monotone (line search)
    assert info["min_gap"] <= 1e-6                                   # certified near-global optimum
    # the convex-relaxation optimum is <= any realized (coordinate-descent) design's surrogate
    t_cd, _ = aliasguard_continuous(Lam, Om, 40, n_sweeps=15, grid_res=480, seed=0)
    assert J[-1] <= surrogate_objective(Lam, Om, t_cd, 1.0) + 1e-9
