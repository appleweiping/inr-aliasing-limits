# Change log: submission hardening + method elevation (2026-07-17)

From `main@4ecc0f8` ("Silent Aliasing in Fixed Fourier-Feature Coordinate Models") to the
ICASSP-2027 submission version. Driven by an independent fresh-clone reproduction, a
claim ledger, a prior-art novelty audit, and the addition of a constructive method.

## Independent verification (treated all prior "passed" claims as unverified leads)

* **Fresh clone + CPU reproduction** of `4ecc0f8`: 39/39 tests pass (34 original + 5 new
  design tests); `run_all.py` regenerates E1–E3/E5-A; released JSONs match (E2
  coherent_demo AUC 0.506 reproduced). Git history scanned — **no secrets committed**.
* **Claim ledger** (`docs/claim-ledger.md`): every quoted number traced to its JSON;
  **0 mismatches** (rounding notes only).
* **Novelty audit** (`docs/novelty-matrix.md`, updated): honest verdict — T2/T4 are
  standard; T1's genuine piece is the grid-inheritance corrective; T3's is the finite-N
  dictionary-referenced bound; the design principle is **Ds-optimal / LCMV null-steering**
  (classical). The paper now credits these explicitly and does not overclaim.

## Method elevation: certified anti-aliasing sampling design

* New module `src/inralias/design.py`: visibility/aliasability-driven **AliasGuard** sample
  design (a Ds-optimal / null-steering criterion on sample times), a fast difference-frequency
  surrogate with an **ablation proving the joint objective is necessary** (coherence-only
  wrecks conditioning; condition-only leaves aliasing), and baselines (fixed jitter,
  condition-only, coherence-only, E-optimal, random search).
* **Continuum certificate** (the genuinely new tool): `a_T(ν)²` is a real trig polynomial
  in ν, so a Bernstein/Lipschitz bound **certifies worst-case aliasability over an entire
  band from the samples alone** — lifting the finite-candidate limitation of T3(b) and of
  random designs. Proved (supplement) and tested (`test_design::test_certificate_is_sound`).
* Experiment `experiments/run_aliasguard.py` (E6): ablation, held-out generalization,
  candidate-set misspecification, budget sweep, signal-domain payoff, the continuum
  certificate, and 1-D + 2-D — all with CIs; auto JSON + figure; wired into `run_all.py`.
* Result: held-out worst-case aliasability ≈0.19 (AliasGuard) vs ≈0.63 (random) / ≈0.59
  (E-optimal); certified ≈0.29 band-wide vs ≈0.57 random; 2-D ≈0.18 vs ≈0.68. **Honest
  limitation stated**: needs a focused concern set (K=O(N)); degrades to ~random for a
  broad band K≫N; grid-restricted greedy inherits T1 and cannot break exact folds.

## Paper restructured around one line

* New main line: **structured indistinguishability (T1) → randomization breaks it, at a
  rate (T3) → certified constructive design (Sec. 5)**. T2 → one-line lemma/Prop S2; T4 →
  a paragraph + supplement. Two full-width figures (theory validation; the design).
* Explicit prior-art credit (Ds-optimal design: Silvey; Atkinson–Donev–Tobias; LCMV/MVDR:
  Frost; Van Trees; Bernstein). Contribution list rewritten so each item is a
  non-derivable result. Abstract/intro within proven scope; SIREN failure, CV failure,
  matched-scale negative all kept.
* `paper/sync_macros.py`: AliasGuard numbers generated from `results/aliasguard.json`
  (no hand-entered numbers). Compliance-with-Ethical-Standards + AI disclosure added.

## Submission package + reproducibility

* `submission/` package (single-anonymous, no fake anonymization): manuscript + supplement
  PDFs, LaTeX source, figures, metadata, submission-form draft, compliance checklist,
  reproducibility, dependency lock, environment record, checksummed results manifest,
  claim ledger, PDF-compliance + assembly scripts, and `AUTHOR-TODO.md`.
* ICASSP 2027 requirements confirmed from official sources (single-anonymous review; 4+1
  pages; spconf template; Sep 16 2026 deadline; Toronto May 2027).

## Not claimed

The paper is **not** presented as deep new theory. Its value is the grid-inheritance
corrective, the finite-N bound, the certified design + continuum certificate, and a
rigorously honest empirical program — with prior art credited throughout.
