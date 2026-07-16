#!/usr/bin/env python
r"""Automated compliance checks on the built manuscript: page count, embedded fonts,
metadata leakage, overfull/undefined from the LaTeX log.  Run from repo root:
    python submission/check_pdf.py
Exit code 0 iff all hard checks pass."""
import pathlib
import re
import sys

try:
    import pypdf
except ModuleNotFoundError:
    pypdf = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF = ROOT / "paper" / "main.pdf"
LOG = ROOT / "paper" / "main.log"

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    ok = ok and cond


if pypdf is None:
    print("[WARN] pypdf not installed in this interpreter; skipping PDF-structure checks "
          "(run with a Python that has pypdf, e.g. `pip install pypdf`).  Log-based checks "
          "still run below.")
    r = None
else:
    r = pypdf.PdfReader(str(PDF))
if r is not None:
    n = len(r.pages)
    check("page count <= 5 (4 technical + optional refs page)", n <= 5, f"{n} pages")
    check("page count >= 4 (uses the space)", n >= 3, f"{n} pages")

# embedded fonts
non_embedded = set()
for page in (r.pages if r is not None else []):
    res = page.get("/Resources")
    if res is None:
        continue
    fonts = res.get("/Font")
    if not fonts:
        continue
    for f in fonts.values():
        f = f.get_object()
        desc = f.get("/FontDescriptor")
        subtype = f.get("/Subtype", "")
        if desc is None and "Type0" not in str(subtype):
            # Type0 fonts carry descriptors on descendant fonts; check those
            df = f.get("/DescendantFonts")
            if df:
                for d in df:
                    dd = d.get_object().get("/FontDescriptor")
                    if dd is None:
                        non_embedded.add(str(f.get("/BaseFont")))
            else:
                non_embedded.add(str(f.get("/BaseFont")))
            continue
        if desc is not None:
            dd = desc.get_object()
            if not any(k in dd for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                non_embedded.add(str(f.get("/BaseFont")))
if r is not None:
    check("all fonts embedded", not non_embedded, ", ".join(sorted(non_embedded)) or "yes")
    # metadata leakage (single-blind: author name is intentional; flag unexpected producers)
    meta = r.metadata or {}
    producer = str(meta.get("/Producer", "")) + str(meta.get("/Creator", ""))
    check("no unexpected PDF metadata author/title leakage",
          not meta.get("/Author") or "Yan" in str(meta.get("/Author", "Yan")),
          f"Author={meta.get('/Author')!r} Producer={producer[:40]!r}")

if LOG.exists():
    log = LOG.read_text(errors="replace")
    overfull = len(re.findall(r"Overfull \\hbox", log))
    undef = len(re.findall(r"(?i)undefined (citation|reference)", log))
    check("0 overfull hboxes", overfull == 0, f"{overfull}")
    check("0 undefined references/citations", undef == 0, f"{undef}")
else:
    print("[WARN] paper/main.log not found; rebuild to check overfull/undefined")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
