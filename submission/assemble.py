#!/usr/bin/env python
r"""Assemble the ICASSP 2027 submission package: rebuild PDFs, run compliance checks, copy
the LaTeX source + figures + PDFs, and generate the environment record and a checksummed
results manifest.  Run from repo root:  python submission/assemble.py
Does not perform any network or submission action."""
import hashlib
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"
PAPER = ROOT / "paper"


def sha256(p):
    h = hashlib.sha256()
    h.update(pathlib.Path(p).read_bytes())
    return h.hexdigest()


def build_paper():
    subprocess.run([sys.executable, str(PAPER / "sync_macros.py")], check=True, cwd=ROOT)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main"], cwd=PAPER,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["bibtex", "main"], cwd=PAPER, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main"], cwd=PAPER,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplement"], cwd=PAPER,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def environment_txt():
    lines = [f"python: {platform.python_version()}", f"platform: {platform.platform()}"]
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}: {m.__version__}")
        except Exception:
            lines.append(f"{mod}: (not importable)")
    try:
        import torch
        lines.append(f"torch: {torch.__version__}; cuda: {torch.version.cuda}")
    except Exception:
        lines.append("torch: not installed in this (CPU) environment")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        lines.append(f"git_commit: {commit}")
    except Exception:
        pass
    (SUB / "environment.txt").write_text("\n".join(lines) + "\n")


def results_manifest():
    lines = ["# SHA-256  path  (raw results released with the paper)"]
    for p in sorted((ROOT / "results").glob("*.json")):
        lines.append(f"{sha256(p)}  results/{p.name}")
    for p in sorted((ROOT / "results" / "figures").glob("*.png")):
        lines.append(f"{sha256(p)}  results/figures/{p.name}")
    (SUB / "results-manifest.txt").write_text("\n".join(lines) + "\n")


def copy_artifacts():
    (SUB / "manuscript.pdf").write_bytes((PAPER / "main.pdf").read_bytes())
    (SUB / "supplement.pdf").write_bytes((PAPER / "supplement.pdf").read_bytes())
    latex = SUB / "latex"
    latex.mkdir(exist_ok=True)
    for f in ("main.tex", "supplement.tex", "refs.bib", "spconf.sty", "IEEEbib.bst",
              "macros_ag.tex", "sync_macros.py"):
        src = PAPER / f
        if src.exists():
            shutil.copy2(src, latex / f)
    figdst = SUB / "figures"
    figdst.mkdir(exist_ok=True)
    (latex / "figures").mkdir(exist_ok=True)
    for png in (PAPER / "figures").glob("*.png"):
        shutil.copy2(png, figdst / png.name)
        shutil.copy2(png, latex / "figures" / png.name)
    for doc in ("claim-ledger.md", "novelty-matrix.md", "CHANGELOG-major-revision.md"):
        src = ROOT / "docs" / doc
        if src.exists():
            shutil.copy2(src, SUB / doc)
    shutil.copy2(ROOT / "requirements.lock", SUB / "requirements.lock")


def main():
    build_paper()
    copy_artifacts()
    environment_txt()
    results_manifest()
    print("assembled submission/ ; running PDF compliance checks ...")
    rc = subprocess.run([sys.executable, str(SUB / "check_pdf.py")], cwd=ROOT).returncode
    print("submission/ ready." if rc == 0 else "submission/ assembled but PDF checks FAILED.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
