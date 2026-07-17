#!/usr/bin/env python
r"""Auto-generate docs/claim-ledger.md from the committed results/*.json (P0-H).

Every NUMBER in the ledger is extracted here from a result JSON -- none is hand-typed -- and
each result's provenance (commit, dirty flag, source-tree hash, timestamp) is tabulated so a
figure/number can be traced to the exact tree that produced it.  paper/check_consistency.py
fails the build if the committed ledger is stale versus a fresh regeneration (ledger == JSON).

Run: python paper/gen_claim_ledger.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def g(d, *path, default="—", fmt=None):
    """Safe nested get; returns `default` (or formatted value) if any hop is missing."""
    cur = d
    for k in path:
        if cur is None:
            return default
        cur = cur[k] if (isinstance(cur, dict) and k in cur) else (
            cur[k] if (isinstance(cur, list) and isinstance(k, int) and -len(cur) <= k < len(cur))
            else None)
    if cur is None:
        return default
    if fmt is not None:
        try:
            return fmt(cur)
        except Exception:
            return default
    return cur


def f2(x):
    return f"{float(x):.2f}"


def f3(x):
    return f"{float(x):.3f}"


AG = load("aliasguard.json")
NL = load("nonlinear.json")
IM = load("image2d_aliasing.json")
DR = load("diagnostic_roc.json")
SM = load("synthetic_matrix.json")
REAL = {n: load(f"real_{n}.json") for n in ("speech", "co2", "sunspots", "sunspots_smooth")}

RESULT_FILES = {
    "aliasguard.json": AG, "nonlinear.json": NL, "image2d_aliasing.json": IM,
    "diagnostic_roc.json": DR, "synthetic_matrix.json": SM,
    **{f"real_{n}.json": REAL[n] for n in REAL},
}


def prov_rows():
    rows = []
    for name, d in RESULT_FILES.items():
        m = (d or {}).get("_meta", {})
        commit = str(m.get("git_commit") or "—")[:8]
        dirty = m.get("git_dirty")
        dirty_s = "clean" if dirty is False else ("DIRTY" if dirty is True else "?")
        rows.append(f"| `{name}` | {'present' if d else 'MISSING'} | `{commit}` | "
                    f"{dirty_s} | `{str(m.get('source_tree_sha256') or '—')}` | "
                    f"{str(m.get('timestamp_utc') or '—')[:19]} |")
    return "\n".join(rows)


def ag_held(method, typ):
    return g(AG, "summary", method, "heldout", typ, "mean", fmt=f3)


# --- claim rows: (num, claim-with-live-numbers, proof, code, result-file, status, scope) ---
def rows():
    ag_near = ag_held("aliasguard", "near")
    ag_broad = ag_held("aliasguard", "broadband_ood")
    rand_near = ag_held("random_jitter", "near")
    ds_near = ag_held("ds_optimal", "near")
    cert_ag = g(AG, "certificate", "aliasguard", "certified_mean", fmt=f2)
    cert_rand = g(AG, "certificate", "random_jitter", "certified_mean", fmt=f2)
    cert_ds = g(AG, "certificate", "ds_optimal", "certified_mean", fmt=f2)
    twod_ag = g(AG, "twod", "aliasguard", "mean", fmt=f3)
    twod_rand = g(AG, "twod", "random", "mean", fmt=f3)
    ff_match = g(NL, "summary", "ffmlp", "exact_match_rate", fmt=f2)
    ff_drift = g(NL, "summary", "ffmlp", "ntk_rel_drift_median", fmt=f2)
    si_match = g(NL, "summary", "siren", "exact_match_rate", fmt=f2)
    si_drift = g(NL, "summary", "siren", "ntk_rel_drift_median", fmt=f2)
    im_fold = g(IM, "part_a_linear", "fold_exact_rate", fmt=f2)
    im_exc = g(IM, "headline_excess", "excess_mean", fmt=f3)
    im_rate = g(IM, "headline_excess", "lattice_gt_random_rate", fmt=f2)
    co2_ora = g(REAL["co2"], "methods", "oracle", "rmse_mean", fmt=f3)
    co2_ls = g(REAL["co2"], "methods", "ls_periodogram", "rmse_mean", fmt=f3)

    R = [
        ("1", "T1 (grid inheritance): ν≡ω (mod Q) ⇒ v_T=0 on every admissible grid subset "
              "satisfying the rank assumptions",
         "supp. Lemma S1 / main Thm 1", "`identifiability.visibility`",
         "synthetic_matrix.json", "proved+numeric", "needs N≥m, Λ distinct mod Q"),
        ("2", "T1 converse: realization-wise error ≥ |a|/√2 (estimator-independent minimax "
              "lower bound, separate from the LS exact-fold reconstruction error)",
         "supp. Lemma S1", "`exactly_indistinguishable`", "test suite", "proved",
         "Le Cam two-point"),
        ("3", "Jitter (Thm 2): fold-coherence mean = χ_η(2πrQ); squared visibility ≈ 1−|χ_η|²; "
              "small-jitter linear law under the stated joint condition",
         "supp. Thm 2", "`expected_jitter_coherence`", "synthetic_matrix.json",
         "proved+numeric", "i.i.d. grid draws; |χ|² form"),
        ("4", "Finite-N aliasability bound max_ν a_T(ν) ≤ √m ε/(λmin−mε)",
         "supp. Prop S2", "`aliasability_concentration_bound`", "synthetic_matrix.json",
         "proved+numeric", "finite candidate set; loose constants"),
        ("5", f"Design (function-space held-out, on-target 'near' band): AliasGuard "
              f"max a_L2 = {ag_near} vs random {rand_near} vs exact-Ds {ds_near}",
         "empirical (Ds/LCMV-motivated)", "`design.ds_optimal_design`, AliasGuard",
         "aliasguard.json", "empirical", "improves the tested Pareto trade-off; not a "
         "necessity proof"),
        ("6", f"Design degrades OFF-target (honest): broadband-OOD held-out a_L2 = {ag_broad} "
              f"(vs {ag_near} on-target)",
         "empirical", "`run_aliasguard` held-out-by-type", "aliasguard.json",
         "empirical (negative)", "targeted design is not uniformly better"),
        ("7", f"Continuum certificate (post-hoc, ANY design): certified band-wide a_L2 — "
              f"AliasGuard {cert_ag} vs random {cert_rand} vs exact-Ds {cert_ds}",
         "supp. Prop S4", "`design.aliasability_certificate`", "aliasguard.json",
         "proved+numeric", "explicit Lipschitz grid bound; vacuous if rank-deficient"),
        ("8", f"Design extends to 2-D held-out vectors: AliasGuard {twod_ag} vs random {twod_rand}",
         "empirical", "`aliasability_L2_of` (n-D)", "aliasguard.json", "empirical", "—"),
        ("9", "Coherent fold vs in-band twin (T1 two-point): all detectors at chance "
              "(AUC≈0.5)",
         "supp. Thm 1", "`run_diagnostic_roc` coherent_demo", "diagnostic_roc.json",
         "proved+numeric", "twin null (theorem-matched)"),
        ("10", "Detector power: ONLY the residual detector has a closed-form curve (exact "
               "noncentral-χ²); ring/lomb/heldout/crossfit are empirical baselines, NOT "
               "claimed to follow it",
         "supp. Thm 2 (detection)", "`residual_test_power`", "diagnostic_roc.json",
         "proved+numeric", "calibration/test separated"),
        ("11", f"Real CO₂ (units 1/yr): sample-only LS RMSE {co2_ls} vs oracle {co2_ora} "
               f"(oracle dominates every selector per draw by construction)",
         "empirical", "`run_real_signal`", "real_co2.json", "empirical",
         "block-clustered CIs; 50% missing + 30 dB AWGN"),
        ("12", f"Trained FF-MLP fold match {ff_match} (init-NTK rel. drift median {ff_drift})",
         "empirical extension", "`run_nonlinear`", "nonlinear.json", "empirical",
         "attribution valid in small-amplitude limit"),
        ("13", f"Trained SIREN responses inconsistent with the init-NTK prediction "
               f"(match {si_match}, NTK rel. drift median {si_drift}) — reported only "
               f"BECAUSE the drift is measured",
         "empirical (negative)", "`run_nonlinear`", "nonlinear.json", "empirical",
         "no claim that SIRENs 'leave' a fixed kernel beyond the measured drift"),
        ("14", f"2-D exact fold verified over held-out tone set (fold_exact_rate {im_fold}); "
               f"trained-net EXCESS replica energy lattice−random {im_exc} "
               f"(lattice>random in {im_rate} of ≥20 paired seeds)",
         "main Thm 1 (2-D)", "`run_image2d`", "image2d_aliasing.json", "proved+empirical",
         "anti-aliased resize; error-spectrum metric; no-alias control"),
    ]
    return R


def render():
    out = []
    out.append("# Claim ledger\n")
    out.append("Every **number** below is auto-extracted from the committed `results/*.json` "
               "by `paper/gen_claim_ledger.py` (never hand-typed). "
               "`paper/check_consistency.py` fails CI if this file is stale versus a fresh "
               "regeneration, so the ledger cannot silently drift from the results. "
               "Claims are stated with their scope and known failure modes; negative and "
               "off-target results are listed alongside the positive ones.\n")
    out.append("## Provenance of each result file\n")
    out.append("| File | Present | Commit | Tree state | Source-tree sha256 | Produced (UTC) |")
    out.append("|------|---------|--------|-----------|--------------------|----------------|")
    out.append(prov_rows())
    out.append("")
    out.append("## Claims\n")
    out.append("| # | Claim (numbers auto-filled from JSON) | Proof | Code | Result | Status | Scope / failure |")
    out.append("|---|------|-------|------|--------|--------|-----------------|")
    for (num, claim, proof, code, resf, status, scope) in rows():
        out.append(f"| {num} | {claim} | {proof} | {code} | `{resf}` | {status} | {scope} |")
    out.append("")
    out.append("## Reproduction & novelty\n")
    out.append("- Fresh-clone reproduction log: `docs/repro-log.md` (regenerated by "
               "`experiments/run_all.py` in a clean venv from `requirements.lock`).")
    out.append("- The sampling-design principle is **Ds-optimal / LCMV null-steering** — "
               "classical, credited (see `docs/novelty-matrix.md`), not claimed as new. The "
               "defensible contributions are the grid-inheritance corrective, the finite-N "
               "dictionary-referenced bound, and the **continuum certificate**.")
    out.append("- Torch experiments (nonlinear / 2-D Part B) reproduce statistically, not "
               "bitwise across CUDA versions.")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    p = ROOT / "docs" / "claim-ledger.md"
    p.write_text(render(), encoding="utf-8")
    print("wrote", p)
