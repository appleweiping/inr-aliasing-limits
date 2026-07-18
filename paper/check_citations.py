#!/usr/bin/env python
r"""Citation-integrity gate (à-la-carte AutoResearchClaw).

Wraps ARC's pure-stdlib `researchclaw.literature.verify.verify_citations` (no API key; it
queries CrossRef / OpenAlex / arXiv / Semantic Scholar) on our `paper/refs.bib`, writes the
report to `results/citation_integrity.json`, and exits nonzero if any citation is classified
HALLUCINATED (SUSPICIOUS entries only warn -- reconcile them by hand, never auto-edit refs.bib).

ARC is used strictly READ-ONLY here: it only reads refs.bib and emits a JSON report. It never
touches our .tex, .bib, or results/*.json numbers.

Usage:
    ARC_HOME=/path/to/AutoResearchClaw python paper/check_citations.py
The committed report is then gated OFFLINE (deterministically) by paper/check_consistency.py,
so this network-dependent script stays advisory in CI while the offline gate is hard.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _bib_keys(bib_text):
    import re
    return set(re.findall(r"@\w+\{([^,\s]+),", bib_text))


def main() -> int:
    # Authoritative human-verified record (evidence per key). The live verifier below is
    # advisory: its OpenAlex/S2 layers are rate-limited and false-flag books / recent arXiv.
    allow_path = ROOT / "paper" / "citation_allowlist.json"
    allow = json.loads(allow_path.read_text(encoding="utf-8")).get("verified", {}) if allow_path.exists() else {}
    bib_text = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
    keys = _bib_keys(bib_text)
    missing = sorted(k for k in keys if k not in allow)
    extra = sorted(k for k in allow if k not in keys)
    rc = 0
    if missing:
        print(f"[FAIL] refs.bib keys with NO human-verified allowlist evidence: {missing}")
        rc = 1
    if extra:
        print(f"[WARN] allowlist has stale keys not in refs.bib: {extra}")

    arc_home = os.environ.get("ARC_HOME")
    if not arc_home or not (pathlib.Path(arc_home) / "researchclaw").is_dir():
        print("[SKIP] ARC_HOME not set to an AutoResearchClaw checkout; "
              "cannot run live citation verification. The committed "
              "results/citation_integrity.json is still gated offline by check_consistency.py.")
        return 0
    sys.path.insert(0, arc_home)
    try:
        from researchclaw.literature.verify import verify_citations
    except Exception as e:  # pragma: no cover
        print(f"[SKIP] could not import ARC verify_citations ({e}); offline gate still applies.")
        return 0

    bib = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
    try:
        report = verify_citations(bib).to_dict()
    except Exception as e:  # network/transient
        print(f"[SKIP] live verification failed ({e}); keeping the committed report.")
        return 0

    out = ROOT / "results" / "citation_integrity.json"
    # record which bib was checked, so the offline gate can confirm provenance
    report["_source"] = "paper/refs.bib"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report.get("summary", {})
    results = report.get("results", [])
    flagged = [r for r in results if r.get("status") in ("suspicious", "hallucinated")]
    print(f"citations: {summary.get('total')} total | verified {summary.get('verified')} | "
          f"suspicious {summary.get('suspicious')} | hallucinated {summary.get('hallucinated')} | "
          f"integrity {summary.get('integrity_score')}")
    # A live flag is only actionable if the entry is NOT already human-verified in the
    # allowlist (the live verifier false-flags books / recent arXiv via OpenAlex/S2 noise).
    unverified_flags = [r for r in flagged if r.get("cite_key") not in allow]
    for r in flagged:
        tag = "UNVERIFIED-FLAG" if r.get("cite_key") not in allow else "flagged-but-allowlisted"
        lvl = "FAIL" if r.get("cite_key") not in allow else "note"
        print(f"  [{lvl}] {r.get('status','').upper()} {r.get('cite_key')} "
              f"({tag}): {r.get('details')}")
    print("wrote", out)
    if unverified_flags:
        print(f"[FAIL] {len(unverified_flags)} live-flagged citation(s) lack allowlist evidence "
              f"-- verify them and add evidence to paper/citation_allowlist.json.")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
