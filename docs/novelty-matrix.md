# Novelty matrix (theorem-by-theorem prior-art audit)

For every result in the paper: what is already known, what we state, and what — precisely —
is the new rigorous part. The related-work paragraph of the paper is the condensed version.

## Honest verdict (fresh independent audit, 2026-07-17)

An independent literature audit reached a blunt, honest verdict that the paper now reflects
in its framing — **the paper's value is a corrective, a finite-N quantification, a certified
constructive design, and empirical honesty, not theorem depth**:

- **T1** — *defensible.* The one non-textbook fact: arbitrary nonuniform **grid subsets
  inherit the grid's congruence equivalence classes** (irregular-on-a-grid does not break
  exact aliasing). Nearest prior: Shannon congruence (full grid) + Le Cam two-point
  (converse) + Chi et al. (near regime). The corrective and the far/exact regime are ours.
- **T2** — *not novel.* sinc/prolate exponential Gram + Riesz bookkeeping (Young;
  Christensen) + elementary error split. Kept as background (Prop. S2), not claimed.
- **T3** — *weak but with a legitimate delta.* Balakrishnan jitter attenuation + alias-free
  random sampling (Shapiro–Silverman, Beutler) + Hoeffding/union/Neumann concentration +
  the eGAD column norm. The delta: **finite-N, dictionary-referenced, uniform-over-candidates**
  control (an explicit rate and the visibility conversion), not asymptotic replica magnitude.
- **T4** — *weak.* (a) is a corollary of T1(ii); (b) is the textbook noncentral-χ²
  linear-model detection statistic (Scheffé). Own piece: the instantiation noncentrality =
  amplitude² × visibility², and a calibrated (train/test-separated) detector suite. Supplement.
- **AliasGuard (the design)** — *incremental; principle credited, not claimed.* The
  asymmetric protect-block / null-cross-block / ignore-nuisance-block criterion is
  **Ds-optimal (nuisance-parameter) experimental design** (Silvey 1980; Atkinson–Donev–Tobias)
  — essentially isomorphic via the Schur complement (interest-block information equals the
  isolated block iff the cross-block J_IN = Φ*Ψ = 0) — and **LCMV/MVDR null-steering**
  (Frost 1972; Van Trees). The paper credits both explicitly. Our additions: (i) the
  identifiability reading (which cross-terms; and Thm 1 says grid designs cannot realize it
  for coherent folds); (ii) the **continuum certificate** — a sample-only worst-case
  guarantee over an entire band via the trig-polynomial/Bernstein structure — which the
  finite-candidate bound and random designs do not provide.

**Conclusion.** No result is presented as deeper than it is. The defensible contributions
are: the T1 grid-inheritance corrective, the T3 finite-N bound, and the certified design.
The continuum certificate is a genuinely useful tool but, as the dedicated audit below
records, its certification *machinery* is standard; the increment is its *application*.
Everything else is credited synthesis.

## Certificate — dedicated prior-art audit (P0-C)

The continuum certificate proves `max_{ν∈[a,b]} a_{L2,T}(ν) ≤ (max_grid g + L·h/2)^{1/2}`
where `g(ν)=φ_ν* M φ_ν` is a real **finite/generalized exponential polynomial** (exponents
`t_ℓ−t_j`, generally non-integer) and `L` is an explicit Lipschitz constant. Honest reading
against the relevant bodies of work:

| Prior art | What it already provides | Bearing on our certificate |
|---|---|---|
| **Semi-infinite programming / optimal design** (Hettich–Kortanek; Kelley; Reemtsen) | Optimizing/bounding a functional under a *continuum* of constraints (e.g. frequencies), including outer approximation and discretization with error control | Certifying a sup over `[a,b]` **is** a semi-infinite problem; our grid+slack bound is a discretization-with-error-control certificate — standard SIP machinery |
| **Lipschitz / grid global optimization** (Shubert–Piyavskii; interval methods) | Certified bounds on `max f` over an interval via samples + a modulus-of-continuity (Lipschitz) slack | This is exactly our `max_grid + L·h/2` construction; not a new technique |
| **Trig-/exponential-polynomial extremal bounds** (Bernstein; Markov; DeVore–Lorentz) | Derivative bounds for (generalized) trigonometric polynomials → grid-to-sup control | Supplies our explicit `L`; we use it, do not extend it |
| **Robust / minimax experimental design** (Wiens; Fedorov–Hackl) | Designs robust over a *region* of the response/parameter space | Same "protect over a continuum" spirit as certifying a band; our object is aliasability, not response variance |
| **Frequency-selective / null-broadening array design** (adaptive beamforming) | Nulls/robustness over a *band* of angles, not a point | The band-wise analogue of our concern set; motivates, and is credited by, the design |
| **Exchange / cutting-plane design** (Fedorov; Cook–Nachtsheim) | Iteratively add the worst continuum point to a finite support | An alternative to our certificate for the *same* semi-infinite goal |

**Calibrated claim.** The certification *method* (gridding + a Bernstein/Lipschitz slack to
turn a finite scan into a continuum guarantee) is **standard** and we do not claim it. The
increment is (i) recognizing `a_{L2,T}(ν)²` as a certifiable finite exponential polynomial,
(ii) delivering a **sample-only, deterministic, post-hoc worst-case aliasability guarantee
for any realized design** in the identifiability/anti-aliasing setting, and (iii) the robust
numerics (Hermitian eigendecomposition, rank/condition detection, vacuous-on-rank-deficient
output). This is an application-and-framing contribution, not a new certification theory.

## Detailed table

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
