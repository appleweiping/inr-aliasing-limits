#!/usr/bin/env python
r"""Multi-target paper build driver.

One provenance-clean content source compiles to three targets, all sharing the SAME
auto-generated numbers (paper/macros_ag.tex) so no per-target number is ever hand-entered:

  icassp        -> paper/main.tex        (spconf, 4pp, SP framing; today's paper)     [live]
  tsp           -> paper/targets/tsp.tex (IEEEtran journal, SP framing, full proofs)  [Phase 3]
  mlconf-<v>    -> paper/targets/mlconf.tex with \def\mlvenuestyle{neurips|iclr|icml}  [Phase 3]
                  (coordinate-network-acquisition reframing; 9pp + appendix)

Every build first regenerates macros + ledger from results/*.json, then latexmk-s the target,
so the numbers can never drift. `check_consistency.py` gates the result.

Usage: python paper/build.py <target|all>
The tsp/mlconf wrappers + the paper/common/*.tex fragments are assembled in Phase 3 (once the
new theory/experiment content is stable); until then only `icassp` is live.
"""
import subprocess
import sys
import pathlib

PAPER = pathlib.Path(__file__).resolve().parent
ROOT = PAPER.parent

TARGETS = {
    "icassp": PAPER / "main.tex",
    "tsp": PAPER / "targets" / "tsp.tex",
    "mlconf-neurips": PAPER / "targets" / "mlconf.tex",
    "mlconf-iclr": PAPER / "targets" / "mlconf.tex",
    "mlconf-icml": PAPER / "targets" / "mlconf.tex",
}
ML_STYLE = {"mlconf-neurips": "neurips", "mlconf-iclr": "iclr", "mlconf-icml": "icml"}


def _run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def regen():
    _run([sys.executable, str(PAPER / "sync_macros.py")], cwd=ROOT, capture_output=True)
    _run([sys.executable, str(PAPER / "gen_claim_ledger.py")], cwd=ROOT,
         capture_output=True, env={"PYTHONUTF8": "1", **_env()})


def _env():
    import os
    return dict(os.environ)


def build(target):
    tex = TARGETS.get(target)
    if tex is None:
        sys.exit(f"unknown target {target!r}; choose from {list(TARGETS)} or 'all'")
    if not tex.exists():
        print(f"[skip] {target}: {tex.name} not present yet (added in Phase 3)")
        return False
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode"]
    if target in ML_STYLE:
        cmd += [f"-usepretex=\\def\\mlvenuestyle{{{ML_STYLE[target]}}}"]
    cmd += [tex.name]
    r = _run(cmd, cwd=tex.parent)
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {target}")
    return ok


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    regen()
    targets = list(TARGETS) if sys.argv[1] == "all" else [sys.argv[1]]
    results = {t: build(t) for t in targets}
    if any(v is False and TARGETS[t].exists() for t, v in results.items()):
        sys.exit(1)


if __name__ == "__main__":
    main()
