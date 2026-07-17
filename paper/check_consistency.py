#!/usr/bin/env python
r"""Automatic consistency gate (P0-A / P0-H).  Fails (exit 1) if the paper, README, macros,
and result JSONs disagree, if cross-document numbering desyncs, or if a headline result was
produced from a dirty/unknown source tree.  Run in CI and before any 'ready' claim.

Checks:
  1. paper/macros_ag.tex is byte-identical to a fresh regeneration from results/aliasguard.json
     (i.e. every AliasGuard number in the paper == the JSON).
  2. main.log / supplement.log have no undefined references or citations.
  3. main.tex and supplement.tex both use \SharedTitle; the OLD title string is absent from
     both compiled PDFs.
  4. the supplement PDF actually prints Lemma S1 and Proposition S1..S4 (cross-doc numbering).
  5. results/aliasguard.json provenance: git_dirty is False and git_commit is set
     (headline results must come from a committed tree).  A --allow-dirty flag downgrades
     this to a warning for local iteration.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
OLD_TITLE = "Silent Aliasing in Fixed Fourier-Feature Coordinate Models"
ok = True
warn = []


def fail(msg):
    global ok
    ok = False
    print("[FAIL]", msg)


def ppass(msg):
    print("[PASS]", msg)


# 1. macros == JSON
macros = PAPER / "macros_ag.tex"
before = macros.read_text() if macros.exists() else ""
subprocess.run([sys.executable, str(PAPER / "sync_macros.py")], capture_output=True, cwd=ROOT)
after = macros.read_text()
if before == after:
    ppass("paper macros == results/aliasguard.json (no drift)")
else:
    fail("macros_ag.tex is stale vs results/aliasguard.json -- run paper/sync_macros.py")

# 1b. claim ledger == JSON (numbers auto-extracted; fail if stale)
ledger = ROOT / "docs" / "claim-ledger.md"
lbefore = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
subprocess.run([sys.executable, str(PAPER / "gen_claim_ledger.py")], capture_output=True,
               cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"})
lafter = ledger.read_text(encoding="utf-8")
if lbefore == lafter:
    ppass("docs/claim-ledger.md == results/*.json (no drift)")
else:
    fail("claim-ledger.md is stale vs results/*.json -- run paper/gen_claim_ledger.py")

# 1c. forbidden absolutes must not appear in the paper/ledger (audit: no overclaiming)
FORBIDDEN = ["No unverified claim remains", "ready to submit", "all claims verified",
             "guarantees optimal", "no analogue"]
for rel in ("docs/claim-ledger.md", "paper/main.tex", "paper/supplement.tex", "README.md"):
    fp = ROOT / rel
    if not fp.exists():
        continue
    txt = fp.read_text(encoding="utf-8", errors="ignore").lower()
    hit = [ph for ph in FORBIDDEN if ph.lower() in txt]
    if hit:
        fail(f"{rel} contains forbidden absolute/overclaim phrasing: {hit}")
    else:
        ppass(f"{rel}: no forbidden absolutes")

# 2. undefined refs
for name in ("main", "supplement"):
    log = PAPER / f"{name}.log"
    if not log.exists():
        warn.append(f"{name}.log missing (build first)")
        continue
    txt = log.read_text(errors="ignore").lower()
    if "undefined reference" in txt or "undefined citation" in txt or "there were undefined references" in txt:
        fail(f"{name}.log reports undefined references/citations")
    else:
        ppass(f"{name}.log: no undefined references/citations")

# 3/4. PDF text checks (need pypdf)
try:
    import pypdf

    def text(pdf):
        r = pypdf.PdfReader(str(PAPER / pdf))
        return " ".join(" ".join((p.extract_text() or "").split()) for p in r.pages)

    mt, st = text("main.pdf"), text("supplement.pdf")
    if OLD_TITLE.replace(" ", "") in mt.replace(" ", "") or OLD_TITLE.replace(" ", "") in st.replace(" ", ""):
        fail("OLD title string still present in a compiled PDF")
    else:
        ppass("old title absent from both PDFs")
    need = ["Lemma S1", "Proposition S1", "Proposition S2", "Proposition S3", "Proposition S4"]
    missing = [n for n in need if n.replace(" ", "") not in st.replace(" ", "")]
    if missing:
        fail(f"supplement PDF missing cross-doc items: {missing}")
    else:
        ppass("supplement prints Lemma S1 + Proposition S1..S4")
except ModuleNotFoundError:
    warn.append("pypdf not installed; skipped PDF-text checks")

# 5. provenance of headline results
allow_dirty = "--allow-dirty" in sys.argv
for jf in ("aliasguard.json", "synthetic_matrix.json", "diagnostic_roc.json"):
    p = ROOT / "results" / jf
    if not p.exists():
        warn.append(f"results/{jf} missing")
        continue
    meta = json.loads(p.read_text()).get("_meta", {})
    dirty, commit = meta.get("git_dirty"), meta.get("git_commit")
    if dirty is True or commit is None:
        msg = f"results/{jf} produced from dirty/unknown tree (git_dirty={dirty}, commit={str(commit)[:8]})"
        (warn.append(msg) if allow_dirty else fail(msg + " -- commit code, then regenerate"))
    else:
        ppass(f"results/{jf}: committed provenance ({str(commit)[:8]}, clean)")

for w in warn:
    print("[WARN]", w)
print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
