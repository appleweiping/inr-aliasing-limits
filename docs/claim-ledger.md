# Claim ledger

Every headline claim in the paper, traced to its proof, code, raw result, figure, whether
it is reproduced from a fresh run, and its scope/known failures. Built from an independent
read of `paper/main.tex`, `paper/supplement.tex`, the committed `results/*.json`, and a
fresh-clone reproduction. **No unverified claim remains in the paper.**

Status legend: ✅ reproduced from a fresh run and paper matches JSON; 🔬 proved
(theorem, verified numerically by a test); ⚠️ scoped/negative result stated honestly.

| # | Claim | Proof | Code entry | Raw result | Figure | Status | Scope / failure |
|---|-------|-------|-----------|-----------|--------|--------|-----------------|
| 1 | T1: grid subsets inherit congruence equivalence classes; ν≡ω (mod Q) ⇒ v_T=0 on every subset | supp. §1, Thm 1(i) | `identifiability.visibility`, `grid_equivalence_class` | `synthetic_matrix.json` panel_a v_mean[0]≈1.9e-14 | Fig 1a,c | 🔬✅ | requires N≥m, Λ distinct mod Q |
| 2 | T1(ii): estimator-independent two-point converse, error ≥ |a|/√2 | supp. §1, Thm 1(ii) | `exactly_indistinguishable`, `function_error_decomposition` | `test_identifiability::test_t1_lecam...` | — | 🔬 | realization-wise Le Cam bound |
| 3 | T3a: jitter visibility = √(1−χ_η²); Gaussian v≈2π|r|Qσ_t | supp. §3.1 | `expected_jitter_coherence`, `visibility` | `synthetic_matrix.json` panel_a | Fig 1a | 🔬✅ | i.i.d. grid draws + Λ distinct mod Q for the v_T conversion |
| 4 | T3b: max_ν a_T(ν) ≤ √m ε/(λmin−mε), ε=2√(log(4(mK+m²)/δ)/N) | supp. §3.2 | `aliasability_concentration_bound` | `synthetic_matrix.json` panel_b | Fig 1b | 🔬✅ | finite candidate set K; union-bound constants loose |
| 5 | Error decomposition: modeled-component / cross / truncation, exact | supp. Prop S2 | `function_error_decomposition` | `test_identifiability::test_t2_...` | Fig 1d | 🔬 | truncation removed by no design |
| 6 | **Design: ablation — joint objective needed** (coh-only κ≈2.1; cond-only max a_T≈0.53; joint gets both) | supp. §"Sampling design", Prop A | `coherence_only_design`, `condition_only_design`, `aliasguard_continuous` | `aliasguard.json` A_regimes.near_band | Fig 2a | ✅ | — |
| 7 | **Design: held-out max-aliasability ≈0.19±0.02 vs 0.63 random / 0.59 E-opt** | — (empirical) | `run_aliasguard.section_C_heldout` | `aliasguard.json` C_heldout | Fig 2b | ✅⚠️ | needs focused set K=O(N); misspec ≈0.20 (still good) |
| 8 | **Continuum certificate: sound & tight; AG certified ≈0.29 band-wide vs 0.57 random** | supp. Prop S(cert) | `aliasability_certificate`, `visibility_certificate` | `aliasguard.json` G_certificate; `test_design::test_certificate_is_sound` | (numbers) | 🔬✅ | Bernstein/Lipschitz; certified ≥ true |
| 9 | Design generalizes to 2-D (max a_T ≈0.18 vs 0.68) | — (empirical) | `aliasguard_continuous_nd` | `aliasguard.json` F_2d | (numbers) | ✅ | — |
| 10 | Greedy grid design inherits T1 (cannot break exact folds); continuous can | Thm 1 | `aliasguard_greedy` vs `aliasguard_continuous` | `test_design::test_greedy_grid_inherits_t1...` | — | 🔬 | motivates off-grid design |
| 11 | T4a: coherent fold indistinguishable from in-band twin — all 5 detectors at chance | supp. §4 | `run_diagnostic_roc` coherent_demo | `diagnostic_roc.json` (AUC 0.506, CI covers 0.5) | supp. fig | 🔬✅ | tested against the *twin* null (theorem-matched) |
| 12 | T4b: noncentral-χ² power = amplitude × visibility | supp. §4 | `residual_test_power` | `diagnostic_roc.json` power sweep | supp. fig | 🔬✅ | calibration/test separated |
| 13 | Real CO₂: sample-only LS 0.167 vs oracle 0.161 | — (empirical) | `run_real_signal` | `real_co2.json` | supp. fig | ✅ | oracle dominates by construction |
| 14 | Real speech: no selector beats RMSE≈1.0 (truncation-dominated) | Prop S2 | `run_real_signal` | `real_speech.json` | supp. fig | ✅⚠️ | broadband; segment-level t-CIs |
| 15 | Trained FF-MLP fold match 77%/corr 0.96 (≥20 seeds) | — (empirical extension) | `run_nonlinear` | `nonlinear.json` | supp. fig | ✅ | NTK-lazy regime only |
| 16 | **Trained SIREN FAILS (match 20%, corr 0.49)** — negative result | — | `run_nonlinear` | `nonlinear.json` summary.siren | supp. fig | ✅⚠️ | SIRENs leave init NTK; theory makes no claim |
| 17 | 2-D lattice masks show predicted replicas (ratio 40±32) vs random 0.99±0.09 | Thm 1 (2-D) | `run_image2d` | `image2d_aliasing.json` | supp. fig | ✅ | over-parameterized scale; ridge/early-stop don't remove |

## Fresh-clone reproduction

- `git clone` of `main@4ecc0f8` in a clean venv from `requirements.lock`: **39/39 tests
  pass**; `run_all.py` regenerates E1–E3 and E5-A; the released JSONs match (E2
  coherent_demo AUC 0.506 reproduced). The working branch adds E6 (design) + `test_design`.
- Torch experiments (E4/E5-B) reproduce statistically (not bitwise across CUDA versions),
  as stated in the README.

## Independent novelty audit (see `docs/novelty-matrix.md`)

- The design principle is **Ds-optimal / LCMV null-steering** — classical, credited, not
  claimed as new. T2/T4 are standard (supplement). The **defensible, non-derivable** pieces:
  the T1 grid-inheritance corrective, the T3 finite-N dictionary-referenced bound, and the
  **continuum certificate** with its identifiability-driven use. The paper states this
  explicitly; no result is presented as deeper than it is.
