# Novelty matrix (theorem-by-theorem prior-art audit)

Internal working document (P0.1 of the major revision).  For every result in the paper:
what is already known, what we state, and what — precisely — is the new rigorous part.
The related-work paragraph of the paper is the condensed version of this table.

| Prior art | What it establishes | What we state | The strictly new part |
|---|---|---|---|
| **Standard LS / pseudoinverse stability** (textbook) | `‖ĉ−c*‖ ≤ ‖ε‖/σ_min`, `E‖ĉ−c*‖² = σ²tr((Φ*Φ)⁻¹)` | Same, instantiated on the exponential dictionary | **Nothing** — demoted to background Lemma 1; `ls_coefficient_mse` is labelled a coefficient-space LS quantity, not an MMSE |
| **Generalized sampling** (Adcock–Hansen; Unser) | Stable recovery of a model subspace from nonideal samples; stable sampling rates | We use the same recast (LS on a finite dictionary) as the *mechanism* | Nothing in the recast itself; our contribution sits on top (equivalence classes, sampling design, detection) |
| **eGAD generalized aliasing decomposition** (Transtrum et al. 2024) | Exact decomposition of LS/random-feature prediction error incl. a label-free generalized aliasing operator, for *given* samples | Our aliasability `a_T(ν) = ‖Φ⁺φ_ν‖` is a column norm of their operator, specialized | **Probabilistic sampling design**: high-probability uniform bound on `max_ν a_T(ν)` over finite candidate sets under i.i.d. sampling (Hoeffding + union + perturbation), and the jitter characteristic-function law — eGAD computes the operator for fixed T, we control it over random T |
| **Basis mismatch** (Chi et al. 2011) | Sensitivity of LS/CS to off-grid tones *near* dictionary atoms | Cited as the near-collision regime | Our silent-aliasing statement is the complementary **far** regime: exact equivalence classes on grid subsets, zero residual with full-energy failure, at any budget `N` satisfying the rank condition |
| **Spectral window / Lomb–Scargle** (VanderPlas 2018) | Where out-of-band power appears in a nonuniform periodogram | Fold target = dictionary-projection analogue | The **estimator-independent** indistinguishability statement (Le Cam two-point), the visibility/aliasability formalization, and the detection theory (T4) |
| **Structured-dictionary view of INRs** (Yüce et al. 2022) | Fourier-feature/periodic networks span structured frequency dictionaries | Motivates the model class; scope stated as the fixed-feature linear core | The sampling/identifiability theory itself; Yüce et al. give expressivity, not sampling |
| **Classical modulo-grid aliasing** (Shannon; textbook) | `ν ≡ ω (mod N)` folding for full uniform grids | Special case `T = full grid` of T1 | T1 covers **arbitrary subsets** of a rate-Q grid (nonuniform designs) and gives the equivalence-class classification + function-space converse there |
| **INR sampling theory** (Najaf & Ongie 2024/25; Saratchandran et al. 2024) | Achievability: exact recovery of realizable signals; activation design for reconstruction | Cited as the achievability side | The converse/identifiability side: what happens for non-realizable signals, when it is invisible, and how sampling design changes it |
| **NTK missing-frequency floor** (Ma et al. 2025, arXiv:2502.05482) | NTK lower bound on error from frequencies absent from the embedding | Cited | We characterize **where** folded energy lands (equivalence classes/fold vectors incl. 2-D), its exact invisibility, and detection — not just an error floor |
| **Noncentral-χ² GOF power** (textbook) | Power of residual tests under Gaussian noise | Same machinery | The *instantiation*: power is driven by amplitude × **visibility**, connecting the identifiability quantity to detection; plus the exact-fold impossibility statement (identical likelihoods) |

## What the paper claims as new (and nothing more)

1. **T1** Exact-equivalence classification of out-of-band tones on arbitrary subsets of
   rate-Q grids (visibility = 0 ⟺ sample-space membership; congruence classes), with an
   estimator-independent two-point (Le Cam) function-space lower bound.
2. **T3** Quantitative *sampling design* results: (a) jitter visibility law equal to the
   jitter characteristic function (small-jitter expansion `v ≈ 2π|k|Qσ_t`); (b)
   high-probability uniform aliasability bound `√m·ε/(λ_min−mε)` over finite candidate
   sets under i.i.d. sampling. Together: exact aliasing is a *grid* phenomenon that
   randomization provably destroys at explicit rates.
3. **T4** Detection dichotomy: exact folds are undetectable by any sample-only test
   (identical observation laws), while visible tones admit exact noncentral-χ² power —
   with a calibrated, baseline-compared empirical operating characteristic.
4. **T2** is a bookkeeping theorem (continuous Gram/Riesz conversion + exact error
   decomposition); claimed as *necessary hygiene*, not as deep novelty.

## Demotions performed

- Old "Theorem 1" (κ = 1/σ_min, LS MSE) → background **Lemma 1** (standard).
- Old "Theorem 3" (average-case variance/aliasing decomposition) → supplement
  (Prop. S1); it is omitted-variable bias bookkeeping and is not load-bearing for the
  main line.
- "Learned Nyquist" / "fundamental limits of INRs" framing → dropped from title,
  abstract, and README; theory object named exactly (fixed Fourier-feature coordinate
  model = the linear core of Fourier-feature INRs).
