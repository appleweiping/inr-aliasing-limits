# Multi-target paper build + AutoResearchClaw usage (record)

## Target strategy (user-approved plan)
One provenance-clean source → three PDFs, all sharing the auto-generated `paper/macros_ag.tex`
(no per-target hand-entered numbers):
- **`mlconf-{neurips|iclr|icml}`** (primary): top-tier ML conference, ~9pp + appendix, reframed
  around **coordinate-network / INR acquisition** (choosing where/when to query). Uses the
  ARC-vendored styles in `paper/targets/styles/`.
- **`tsp`** (fallback): IEEE T-SP journal, SP framing, full proofs inline.
- **`icassp`** (kept): the current 4-page `paper/main.tex`, SP framing.

Build: `python paper/build.py <target|all>` (regenerates macros+ledger, then latexmk). The
`tsp`/`mlconf` wrappers and the `paper/common/*.tex` shared fragments are assembled in **Phase 3**
once the new theory (U1/U3a/U2) and experiments (Programs 1–3) are stable — front-loading the
deep `main.tex`→fragments split would cause churn as the content changes. Phase 0 established the
infra: `paper/build.py`, `paper/targets/styles/` (vendored), `paper/common/` and
`paper/exports/` dirs, and the citation gate.

## AutoResearchClaw à-la-carte usage
ARC is a generative topic→paper pipeline with no "import my project" entry; used READ-ONLY,
advisory, and NEVER given jurisdiction over our `.tex`/`.bib`/`results/*.json` (its
number-sanitizer would strip real numbers). Clone lives OUTSIDE the repo in scratchpad
(`ARC_HOME`); creds/run-dirs are gitignored.

| ARC subsystem | Status | Backend |
|---|---|---|
| Citation integrity (`verify_citations`) | **LIVE** → `paper/check_citations.py` + `paper/citation_allowlist.json` (offline gate in `check_consistency.py`) | none (stdlib) |
| Top-venue LaTeX styles (`templates.conference`) | **DONE** → vendored into `paper/targets/styles/` | none |
| Adversarial peer review (Stage 18 personas) | Phase 3 | see backend note |
| Quality-assessor + venue-recommender | Phase 3 | heuristic (no key) or LLM |
| Hypothesis-debate (Stage 8) | Phase 3 | see backend note |
| FigureAgent (schematics only) | Phase 3, optional | LLM |

**LLM-backend decision (probed 2026-07-18):** no `acpx` on PATH and no `ANTHROPIC_API_KEY` in
this environment, so ARC's ACP-claude and native-anthropic paths are unavailable headless.
→ **Fallback B:** for the LLM-driven stages (peer review, hypothesis debate, quality scoring) we
replicate ARC's reviewer-persona / debate-role / rubric PROMPTS (from `researchclaw/prompts/ml.py`
and `researchclaw/assessor/`) using our own subagents, feeding a hand-authored **prose brief**
(`paper/exports/review_brief.md`, NOT our `.tex`). Citation-verify and template-export need no
backend and are already integrated.
