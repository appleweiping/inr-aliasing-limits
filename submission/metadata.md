# Submission metadata (ICASSP 2027)

**Suggested title:** Silent Aliasing and Certified Sampling Design for Fixed
Fourier-Feature Models

**Author:** Weiping Yan — University of Minnesota, Twin Cities, Minneapolis, MN, USA.
(Single author. Verify the affiliation statement and email against institutional policy
before submitting — see `submission/AUTHOR-TODO.md`.)

**Topic area (ICASSP uses topic areas, not EDICS):** Signal Processing Theory & Methods
(primary). Secondary candidate: Sensor Array & Multichannel Signal Processing (the design
connects to null-steering) — select per the submission form's allowance.

**Abstract (≤ the form's limit; matches the manuscript):**
Fixed Fourier-feature coordinate models — the linear core of Fourier-feature implicit
neural representations — fit a signal by least squares on an exponential dictionary Λ. We
analyze, and then design against, the aliasing of out-of-band content. (i) A
visibility/aliasability calculus shows that arbitrary nonuniform subsets of a sampling
grid inherit the grid's aliasing equivalence classes: an integer tone congruent mod Q to a
model frequency is exactly indistinguishable from it — zero training residual, full-energy
failure, and no estimator can avoid it; irregular-on-a-grid sampling does not break exact
aliasing. (ii) Off-grid jitter destroys the fold with visibility equal to the jitter
characteristic function, and i.i.d. sampling gives a finite-N, dictionary-referenced
concentration bound on worst-case aliasability. (iii) We show the anti-aliasing objective
is a Ds-optimal / null-steering criterion on sample times, give a fast surrogate whose
joint objective an ablation shows is necessary, and — the new tool — a computable continuum
certificate that guarantees worst-case aliasability over an entire band from the samples
alone. All closed forms are Monte-Carlo validated; trained networks are an explicitly
labeled empirical extension with failures reported.

**Keywords:** sampling theory; aliasing; identifiability; optimal experimental design;
Fourier features.

**Conflicts of interest:** none declared.

**Funding:** none.

**Prior/related submissions:** none. This is original work, submitted only to ICASSP 2027.
A public code repository exists (`github.com/appleweiping/inr-aliasing-limits`); ICASSP 2027
is single-anonymous, so this does not violate anonymity and preprint posting is permitted.

**AI-assistance disclosure:** code and text were produced with AI assistance (Anthropic
Claude) under the author's direction; AI is not an author; all results were produced by the
released reproducible code. Disclosed in the manuscript's Compliance-with-Ethical-Standards
statement (per IEEE policy).
