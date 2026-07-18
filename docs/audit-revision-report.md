# Independent-audit revision — honest report

**Baseline:** `main@6f8cb6f`. **This revision:** branch `revision-audit-fixes`, final commit
below. Experiments were run on an independent GPU server (RTX 4080 SUPER, 128-core CPU) at
the final commit; all nine `results/*.json` carry `git_dirty=false` and a source-tree
fingerprint the committed code reproduces (verified by `paper/check_consistency.py`).

> **This paper is NOT declared "ready to submit."** Four independent reviewers landed at
> Weak Accept / Weak Reject; their blocking issue (design-comparison budget fairness) was
> fixed and the result survives, but the honest ceiling here is a **borderline** theory-track
> paper (see §6). No "all passed" claim is made on the basis of reviewer scores.

## 1. New commit SHA
Results-producing commit (stamped in every result `_meta`, `git_dirty=false`,
source-tree `03a901c5bc5e384f`): **`3e4df18`**. The tip of `revision-audit-fixes` adds the
regenerated macros/ledger, the rebuilt PDFs, and this report. History was NOT rewritten; AI
assistance (Anthropic Claude, under the author's direction) and failed/negative experiments
are disclosed, not hidden. A full-history secret scan (regex + detect-secrets) is clean; the
SSH helper reads all connection details from the environment (no committed endpoint/secret).

## 2. Before → after claim diff (headline)
| Claim | Before (baseline) | After (this revision) |
|---|---|---|
| Jitter theorem | "visibility equal to the jitter characteristic function" | fold-coherence **mean** = χ_η(2πrQ); squared visibility ≈ 1−|χ_η|²; small-jitter linear law under a stated joint condition; \|χ\|² throughout |
| "N^{-1/2} rate is exact" | asserted | removed — labelled an upper-bound rate, no matching lower bound |
| Certificate scope | "random designs have no analogue"; "1-D and 2-D" | deterministic post-hoc bound for **any** design (random/E-opt/D_s alike); **1-D only** (no 2-D certificate exists); non-vacuous in **20/24** scenarios (rest reported vacuous) |
| Design principle | implied novel | **credited** as D_s-optimal / LCMV null-steering; exact-D_s baseline compared **at matched budget**; certification machinery credited as standard gridding+Bernstein |
| Design headline | coefficient-norm, single fixed held-out band | **function-space** a_{L2}; ≥20 paired scenarios; 5 held-out band types incl. **honest off-target degradation**; paired-difference CI |
| "joint objective necessary" | asserted | "improves the tested trade-off at matched budget" |
| Fig-2d label | noise-containing term mislabelled `aliasing_bias_rmse`; "equal noise gain κ" | bias/noise/truncation **separated**; κ (condition number) ≠ noise gain, both reported |
| Real signals | full-reference standardization (leakage); bandwidth-only oracle | **sample-only** detrend/normalization; oracle over the actual (B,ridge) family, **dominance asserted per draw**; **recording/block-clustered** CIs; correct units (1/yr, 1/month) |
| SIREN | "SIRENs leave their init NTK" | "trained SIREN responses inconsistent with the init-NTK prediction"; **NOT** a drift artifact (SIREN drift < FF-MLP drift, yet FF-MLP matches) — measured, not asserted |
| 2-D | "generalizes" | fold-exact over held-out tones; trained-net **excess** vs no-alias control (error spectrum, anti-aliased resize); design-vs-random margin **overlapping/not separated**, stated as preliminary |
| Provenance / claims | dirty-tree JSON (commit lacking the code); manual ledger with "No unverified claim remains" | full SHA + dirty flag + source-tree hash + host/GPU; **auto-generated** ledger (numbers from JSON); CI gate; absolutes forbidden |

## 3. Theorem / proof change log
- **Thm 1 (grid inheritance).** Statement unchanged in substance; wording tightened —
  invisibility (i)–(ii) need **no** rank condition (hold on any grid subset); the full-rank
  claims (iii) require N≥m. "≥ full out-of-band energy" now attributed to the **LS**
  estimator (√2|a|); the estimator-independent minimax floor is |a|/√2 (Le Cam), stated
  separately.
- **Thm 2 (jitter/randomization).** Proof corrected: the squared-visibility expansion
  v² = 1−|χ_η|²+O_P(mε) does **not** yield v=√A+O(mε) as A→0; the supplement now gives the
  squared bound, a uniform O(√(mε)) bound, a refined O(mε/√A) bound for A bounded below, and
  the joint asymptotic condition under which the small-jitter linear law v≈2π|r|Qσ_t holds.
  |χ_η|² used throughout. The Fig-1a match is flagged as **empirical**, holding beyond the
  regime where the finite-N union bound is provably tight.
- **Prop 1 / Prop S4 (certificate).** `aliasability_certificate` rewritten: Hermitian
  eigendecomposition (no bare inverse), rank/condition detection, **vacuous (∞)** on
  rank-deficiency, reports σ_min/cond/tolerance; a real **finite/generalized exponential
  polynomial** (non-integer exponents) with an **explicit Lipschitz grid bound**. Function-
  space metric a_{L2,T}. Prior-art audit added (semi-infinite programming, Lipschitz global
  optimization, Bernstein/Markov, robust design, null-broadening, exchange methods); the
  certification method is credited as **standard**, the application as the increment.
- **Prop S1/S2/S3** renumbered and cross-referenced via a shared `xref.tex` (main↔supplement
  numbering can no longer desync; enforced in CI).

## 4. Fresh-clone reproduction
See `docs/repro-log.md`. Fresh `git clone` + `43/43 tests pass`; the deterministic
`synthetic_matrix` experiment reproduces the released visibilities **exactly**
(0/0/0.6434/0.9463/0.9468). Torch experiments reproduce statistically, not bit-for-bit
(stated). `check_consistency.py` (strict): **ALL PASS**, including that the committed source
tree reproduces every result's fingerprint.

## 5. Full result manifest (all `git_dirty=false`, one commit, one source-tree hash)
`aliasguard.json`, `synthetic_matrix.json`, `diagnostic_roc.json`, `nonlinear.json`,
`image2d_aliasing.json`, `real_{speech,co2,sunspots,sunspots_smooth}.json`. Headline numbers
are auto-synced into the paper (`paper/macros_ag.tex`) and the ledger (`docs/claim-ledger.md`).

## 6. Honest overall assessment & strongest reject reason
The four reviewers (sampling-theory, optimal-design, statistics, reproducibility) returned
**Weak Accept, Weak Accept, Weak Reject, Weak Accept**. Every concrete concern was addressed
(the blocking one — the exact-D_s baseline being optimized at 1/7.5 the budget — is fixed:
all coordinate-descent methods now share n_sweeps/grid, and AliasGuard still beats exact-D_s
with a **significant paired difference**). Reviewer scores are **not** used to declare
acceptance.

**Strongest remaining reject reason (theory track):** the *proven* content is thin. Thm 1 is
a one-line congruence fact; Thm 2 is Balakrishnan attenuation plus textbook
Hoeffding/union/Neumann whose sufficient condition does not bite at the validated N; the
design principle and the certificate machinery are both, honestly, standard and credited. The
paper's value is a **corrective + a finite-N quantification + a certified, honestly-scoped
constructive design + empirical honesty (including a genuine SIREN negative result and an
off-target/2-D non-result)** — not theorem depth. A theory-methods committee may find that
insufficiently novel. That is the honest ceiling; the paper does not claim more.

## 7. Unresolved risks / limitations (stated, not hidden)
- 2-D design advantage is small and **not statistically separated**; only an i.i.d.-uniform
  baseline. Presented as preliminary, not a headline.
- Speech clusters on **3 recordings** (df=2 t-intervals are wide); CO2 on 3 blocks.
- Nonlinear seeds are independently drawn but **paired per scenario** (not a full factorial
  cross); weight variance is isolated only by the stability control.
- Concentration constants are loose (union bound); the design needs a focused concern set
  (K=O(N)) and loses its edge on broadband/badly-misspecified targets.
- Anonymization: author name and repo are shown (single-anonymous assumption, **pending
  author-kit verification**); the SSH helper is now endpoint- and secret-free, but result
  `_meta` records the run host (required provenance) — scrub if double-anonymous is required.
