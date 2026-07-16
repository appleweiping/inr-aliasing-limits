# ICASSP 2027 format & policy compliance checklist

Requirements confirmed from the official ICASSP 2027 CfP / Editorial Policies pages
(retrieved July 2026). Items marked ⧗ depend on the final author kit (not yet posted as of
retrieval) and must be re-checked when it is released.

| Requirement | Rule (ICASSP 2027) | This submission | Status |
|---|---|---|---|
| Page limit | ≤4 pages technical incl. figures & references; optional 5th page = references + funding + Compliance-with-Ethical-Standards only | `main.pdf` = 4 pages incl. refs; Compliance statement + funding on p.4 (fits) | ✅ |
| Template | SPS `spconf.sty` + `IEEEbib.bst` (spconf family, **not** IEEEtran) ⧗ | uses `spconf.sty` + `IEEEbib.bst` | ✅ (re-check kit) |
| Font size | `\ninept` (9 pt, two-column) | `\ninept` set | ✅ |
| Anonymity | **Single-anonymous** — reviewers see authors; no manuscript anonymization | author name + affiliation on PDF | ✅ |
| Compliance with Ethical Standards statement | required/expected | present on p.4 | ✅ |
| Conflict-of-interest & funding disclosure | required | "none" in statement + `metadata.md` | ✅ |
| AI-assisted writing/coding disclosure | IEEE baseline: disclose in paper, AI not an author ⧗ | disclosed in Compliance statement + `\thanks` | ✅ (re-check kit) |
| Originality / single submission | must be original, submitted only to ICASSP | true; declared in `metadata.md` | ✅ |
| iThenticate similarity | plagiarism/self-plagiarism check applied | original text; no prior version | ✅ |
| Topic area | select from the ICASSP topic list (no EDICS) | Signal Processing Theory & Methods | ✅ |
| References-only 5th page | optional; only refs/funding/ethics allowed there | not needed (fits in 4) | ✅ |
| PDF technical | embedded fonts; no metadata leakage; figure text legible at print size | checked by `submission/check_pdf.py` | ✅ (see report) |

## Automated PDF checks (run `python submission/check_pdf.py`)

- page count = 4 (≤ 4+1 with the 5th refs-only unused);
- 0 overfull hboxes, 0 undefined references/citations (from `main.log`);
- all fonts embedded;
- no author-identifying metadata beyond the intended author name (single-blind, so author
  name is intentional);
- figure fonts ≥ ~6 pt effective at column width (figures generated at print typography).

## Deadlines (official ICASSP 2027)

- Paper submission: **September 16, 2026** (timezone not stated officially; AoE customary —
  submit ≥24 h early).
- Acceptance notification: January 13, 2027. Final paper: January 27, 2027.
- Conference: May 16–21, 2027, Metro Toronto Convention Centre, Toronto, Canada.

## Author actions still required (cannot be automated) — see `AUTHOR-TODO.md`
