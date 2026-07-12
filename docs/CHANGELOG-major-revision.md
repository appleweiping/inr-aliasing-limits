# Change log: major revision (2026-07-12/13)

One-page summary of the method-level revision from the previous version
(`3b406a4`, "Learned Nyquist") to the current one ("Silent Aliasing in Fixed
Fourier-Feature Coordinate Models").

## Strategy

* **Retitled and rescoped.** Theory object named exactly: fixed Fourier-feature
  coordinate model (linear coefficients on a finite frequency set) — the linear core of
  Fourier-feature INRs. "Learned Nyquist", "fundamental limits of INRs", and all-INR
  claims removed. Trained nets = empirical extrapolation; adaptive frequencies = out of
  scope (stated in abstract, intro, conclusion, README).
* **Prior-art audit** (`docs/novelty-matrix.md`): old Theorem 1 demoted to background
  Lemma 1 (textbook LS stability); old Theorem 3 demoted to supplement Prop. S1
  (standard omitted-variable bookkeeping).

## New theory (T1–T4, full proofs in `paper/supplement.pdf`)

* **T1** Visibility/aliasability framework; exact equivalence classes on arbitrary
  rate-Q grid subsets (explicit rank conditions); estimator-independent two-point
  (Le Cam) function-space lower bound |a|/√2.
* **T2** Closed-form continuous Gram + Riesz conversion; exact L² decomposition
  separating aliasing-induced modeled-component error / cross term / truncation error.
* **T3** Random sampling provably breaks exact aliasing: jitter visibility =
  characteristic-function law (Gaussian: v ≈ 2π|r|Qσ_t); i.i.d. finite-candidate
  concentration bound √m·ε/(λ_min−mε), ε = 2√(log(4(mK+m²)/δ)/N). "No density removes
  aliasing" claims replaced by the correct trichotomy (grid persistence / fixed-tone
  decay / per-N adversarial).
* **T4** Detection dichotomy: exact folds undetectable by any sample-only test
  (identical observation laws); noncentral-χ² power (amplitude × visibility) otherwise.

## Code corrections (P0.3)

`mmse_recoverable`→`ls_coefficient_mse` (+ separate `bayes_coefficient_mmse`);
`mmse_aliasing_floor`→`aliasing_variance_term`; `landau_min_samples` deleted;
`pinv_apply(ridge=0)` = SVD pseudoinverse (min-norm case documented separately); rank/
conditioning guards in the LS operator; `real=True` now a true cosine/sine design
(Hermitian symmetry enforced in the parameterization); real Lomb–Scargle implemented
(scipy) and the correlation periodogram renamed honestly; diagnostics take
null-calibrated thresholds only (`null_calibrated_threshold` API); silent-aliasing and
coherent-fold docstrings corrected. Tests 22→34, all green, CI added.

## Experiments (all rebuilt)

* **E1** unified synthetic matrix: jitter law vs theory, concentration bound overlay,
  grid persistence vs i.i.d. fixed/adversarial tones, bias-vs-truncation decomposition
  (mean ± 95% CI everywhere).
* **E2** detection: calibration/validation/test strictly separated; 5 sample-only
  detectors vs the exact T4 power curve; bootstrap CIs, PR-AUC, TPR@1%/5%; ring-width /
  off-grid / outside-ring / sampling-family sweeps; coherent fold reported separately as
  the impossibility demo (AUC 0.21). The old "validated fixed 0.1 threshold" claim is
  gone.
* **E3** real signals: true oracle (best-in-grid on actual error; the old
  reference-spectrum heuristic was not an oracle and exploded on speech); all deployable
  selectors sample-only; anti-aliased `resample_poly` (no bare `x[::step]`); 9 speech
  segments from 3 recordings with sha256 provenance; full records (no max-energy window
  selection); paired 95% CIs. Headlines: CO₂ sample-only LS-periodogram 0.167 vs oracle
  0.171; broadband speech floor ≈ 1.0 for every selector (truncation-dominated).
* **E4** trained nets: ≥20 sampling seeds × 3 tones × 3 architectures; predictions from
  each network's own NTK linearization (no arbitrary Λ=20); ablation attribution;
  label-permutation/amplitude/weight-seed controls. FF-MLP: exact fold 77%, top-3 98%,
  corr median 0.96. Band-limited head: corr 0.98. **SIREN fails (20%, corr 0.49) —
  reported as a negative result**; old "fold-for-fold persistence" claim removed.
* **E5** 2-D: exact frequency-vector fold on the linear model (predicted = measured,
  residual 1e-15); trained nets on lattice (coherent) vs random masks: predicted DFT
  replicas at 40±32× control energy vs 0.99±0.09; ridge/early stopping do NOT remove
  replicas (89/25) — not an overfitting artifact; 66 dB observed-pixel vs 12 dB
  held-out-pixel PSNR; clean/noisy/held-out/full-field PSNR separated. The old
  "matched-is-best" framing dropped (it wasn't).

## Repository (P1)

`requirements.lock`; GitHub Actions CI; `_meta` (commit/env/versions) in every result
JSON; honest two-tier `run_all.py` (CPU vs `--full`); data licenses + sha256 provenance;
AI-assistance disclosure in README (git history unmodified); superseded
scripts/figures/JSONs deleted.

## Final adversarial pass (4 personas) and fixes

A four-persona review (sampling-theory / INR / statistics / reproducibility) of the
revised draft confirmed 11 findings; all were fixed:

* **(reject-level, root-caused)** The detection "impossibility demo" used a no-tone null,
  which does **not** instantiate T4(a) — the released JSON itself showed a Lomb detector
  at AUC 0.81 there. Fixed by instantiating the theorem's two-point pair exactly (H0 =
  same-amplitude tone at the in-band twin −17; H1 = tone at 111): rerun, all five
  detectors now sit at chance (AUC 0.48–0.51, every CI covering 0.5), as the theorem
  requires. The paper now distinguishes *attribution* impossibility from "something
  changed" detection.
* T3(a) statement–proof match: random-grid-draw and Λ-distinct-mod-Q hypotheses added to
  the theorem statement (and the supplement).
* Classical randomized-sampling literature added and positioned (Shapiro–Silverman 1960;
  Balakrishnan 1962 — the jitter characteristic-function attenuation is classical;
  Beutler 1970); our delta stated as finite-N, high-probability, dictionary-uniform
  control. Anti-aliasing INR architectures (BACON/mip-NeRF/BANF) cited.
* Oracle dominance restored by construction (minimizes true error over the selector grid
  ∪ per-draw selector choices); speech CIs switched to segment-level t-intervals with
  per-(segment,seed) decorrelated RNG streams.
* Modeled-component term relabelled (contains estimation noise, not only aliasing bias);
  T3a/T3b label swap between paper and code fixed; ±s.d. qualifiers; matched-scale 2-D
  negative stated; controls-scope caveat; license wording unified; Prop. S1 numbering;
  GIT_COMMIT provenance for server-produced JSONs; stale docstrings.

## Author to-dos before submission (cannot be done by this repository)

* Verify the UMN affiliation statement complies with the institution's policy for this
  submission.
* Complete the venue's AI-assisted writing/coding disclosure form per the submission
  year's policy.
