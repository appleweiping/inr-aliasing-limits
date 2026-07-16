# ICASSP 2027 submission package

Everything needed to submit and to reproduce. Regenerate with `python submission/assemble.py`
(from the repo root), which rebuilds the PDFs, runs compliance checks, and refreshes the
manifests.

## Contents

| File | What |
|---|---|
| `manuscript.pdf` | the paper (4 pages incl. references; single-anonymous, author shown) |
| `supplement.pdf` | full proofs + the certificate proposition (technical appendix) |
| `latex/` | complete LaTeX source (`main.tex`, `supplement.tex`, `refs.bib`, `spconf.sty`, `IEEEbib.bst`, `macros_ag.tex`, figures) |
| `figures/` | final figures (PNG, print typography) |
| `metadata.md` | title, abstract, keywords, topic area, COI, funding, AI disclosure |
| `submission-form-draft.md` | field-by-field values for the submission system |
| `compliance-checklist.md` | ICASSP 2027 format/policy compliance (with sources) |
| `reproducibility.md` | how to reproduce every result from the public repo |
| `requirements.lock` | pinned CPU dependency versions |
| `environment.txt` | Python/NumPy/SciPy/torch/CUDA/GPU record + git commit |
| `results-manifest.txt` | SHA-256 of every released result JSON/figure |
| `claim-ledger.md` | every claim → proof → code → JSON → figure → reproduced |
| `novelty-matrix.md` | independent prior-art audit (what is / isn't novel) |
| `check_pdf.py`, `assemble.py` | compliance-check and package-assembly scripts |
| `AUTHOR-TODO.md` | actions only the author can perform (identity, accounts, final submit) |

## Review model

ICASSP 2027 is **single-anonymous** (reviewers see authors). No manuscript anonymization is
required; the public repository `github.com/appleweiping/inr-aliasing-limits` and the author
name on the PDF are compliant, and preprint posting is permitted. There is therefore **no
separate anonymized snapshot** — the author-version PDF is the submission PDF. This is a
deliberate, policy-based choice, not an omission.

## Status

The package assembles, all PDF compliance checks pass, and every headline number in the
manuscript is regenerated from `results/*.json`. Remaining steps are author-only actions in
`AUTHOR-TODO.md` (identity/affiliation confirmation, account, and the final click).
