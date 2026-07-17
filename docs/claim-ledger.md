# Claim ledger

Every **number** below is auto-extracted from the committed `results/*.json` by `paper/gen_claim_ledger.py` (never hand-typed). `paper/check_consistency.py` fails CI if this file is stale versus a fresh regeneration, so the ledger cannot silently drift from the results. Claims are stated with their scope and known failure modes; negative and off-target results are listed alongside the positive ones.

## Provenance of each result file

| File | Present | Commit | Tree state | Source-tree sha256 | Produced (UTC) |
|------|---------|--------|-----------|--------------------|----------------|
| `aliasguard.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:01:22 |
| `nonlinear.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T15:57:34 |
| `image2d_aliasing.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:27:39 |
| `diagnostic_roc.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:02:28 |
| `synthetic_matrix.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T15:51:11 |
| `real_speech.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:04:45 |
| `real_co2.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:04:52 |
| `real_sunspots.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:05:21 |
| `real_sunspots_smooth.json` | present | `a7db7ea9` | clean | `1f28d5f0416a1197` | 2026-07-17T16:05:41 |

## Claims

| # | Claim (numbers auto-filled from JSON) | Proof | Code | Result | Status | Scope / failure |
|---|------|-------|------|--------|--------|-----------------|
| 1 | T1 (grid inheritance): ν≡ω (mod Q) ⇒ v_T=0 on every admissible grid subset satisfying the rank assumptions | supp. Lemma S1 / main Thm 1 | `identifiability.visibility` | `synthetic_matrix.json` | proved+numeric | needs N≥m, Λ distinct mod Q |
| 2 | T1 converse: realization-wise error ≥ |a|/√2 (estimator-independent minimax lower bound, separate from the LS exact-fold reconstruction error) | supp. Lemma S1 | `exactly_indistinguishable` | `test suite` | proved | Le Cam two-point |
| 3 | Jitter (Thm 2): fold-coherence mean = χ_η(2πrQ); squared visibility ≈ 1−|χ_η|²; small-jitter linear law under the stated joint condition | supp. Thm 2 | `expected_jitter_coherence` | `synthetic_matrix.json` | proved+numeric | i.i.d. grid draws; |χ|² form |
| 4 | Finite-N aliasability bound max_ν a_T(ν) ≤ √m ε/(λmin−mε) | supp. Prop S2 | `aliasability_concentration_bound` | `synthetic_matrix.json` | proved+numeric | finite candidate set; loose constants |
| 5 | Design (function-space held-out, on-target 'near' band): AliasGuard max a_L2 = 0.153 vs random 0.576 vs exact-Ds 0.270 | empirical (Ds/LCMV-motivated) | `design.ds_optimal_design`, AliasGuard | `aliasguard.json` | empirical | improves the tested Pareto trade-off; not a necessity proof |
| 6 | Design degrades OFF-target (honest): broadband-OOD held-out a_L2 = 0.498 (vs 0.153 on-target) | empirical | `run_aliasguard` held-out-by-type | `aliasguard.json` | empirical (negative) | targeted design is not uniformly better |
| 7 | Continuum certificate (post-hoc, ANY design): certified band-wide a_L2 — AliasGuard 0.52 vs random 0.72 vs exact-Ds 0.66 | supp. Prop S4 | `design.aliasability_certificate` | `aliasguard.json` | proved+numeric | explicit Lipschitz grid bound; vacuous if rank-deficient |
| 8 | Design extends to 2-D held-out vectors: AliasGuard 0.539 vs random 0.583 | empirical | `aliasability_L2_of` (n-D) | `aliasguard.json` | empirical | — |
| 9 | Coherent fold vs in-band twin (T1 two-point): all detectors at chance (AUC≈0.5) | supp. Thm 1 | `run_diagnostic_roc` coherent_demo | `diagnostic_roc.json` | proved+numeric | twin null (theorem-matched) |
| 10 | Detector power: ONLY the residual detector has a closed-form curve (exact noncentral-χ²); ring/lomb/heldout/crossfit are empirical baselines, NOT claimed to follow it | supp. Thm 2 (detection) | `residual_test_power` | `diagnostic_roc.json` | proved+numeric | calibration/test separated |
| 11 | Real CO₂ (units 1/yr): sample-only LS RMSE 0.511 vs oracle 0.441 (oracle dominates every selector per draw by construction) | empirical | `run_real_signal` | `real_co2.json` | empirical | block-clustered CIs; 50% missing + 30 dB AWGN |
| 12 | Trained FF-MLP fold match 0.75 (init-NTK rel. drift median 6.92) | empirical extension | `run_nonlinear` | `nonlinear.json` | empirical | attribution valid in small-amplitude limit |
| 13 | Trained SIREN responses inconsistent with the init-NTK prediction (match 0.30); NOT an NTK-drift artifact (SIREN kernel drift 0.64 < FF-MLP 6.92, yet FF-MLP prediction matches) — the fixed-feature/init-NTK description does not extrapolate to SIRENs | empirical (negative) | `run_nonlinear` | `nonlinear.json` | empirical | explicitly NOT attributed to SIRENs 'leaving' a fixed kernel |
| 14 | 2-D exact fold verified over held-out tone set (fold_exact_rate 1.00); trained-net EXCESS replica energy lattice−random 1.957 (lattice>random in 1.00 of ≥20 paired seeds) | main Thm 1 (2-D) | `run_image2d` | `image2d_aliasing.json` | proved+empirical | anti-aliased resize; error-spectrum metric; no-alias control |

## Reproduction & novelty

- Fresh-clone reproduction log: `docs/repro-log.md` (regenerated by `experiments/run_all.py` in a clean venv from `requirements.lock`).
- The sampling-design principle is **Ds-optimal / LCMV null-steering** — classical, credited (see `docs/novelty-matrix.md`), not claimed as new. The defensible contributions are the grid-inheritance corrective, the finite-N dictionary-referenced bound, and the **continuum certificate**.
- Torch experiments (nonlinear / 2-D Part B) reproduce statistically, not bitwise across CUDA versions.
