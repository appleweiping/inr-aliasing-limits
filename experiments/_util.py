"""Shared experiment utilities: src path bootstrap, results IO, headless plotting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# make `import inralias` work when running `python experiments/run_*.py` directly
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless / no window (respects window-management rules)
# Print-size typography: figures are scaled to ~0.5x in the two-column paper, so fonts are
# set large here to remain >=6-7pt effective at ICASSP column width.
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 1.6,
})
import matplotlib.pyplot as plt  # noqa: E402,F401

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"


def ensure_dirs() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)


def _source_tree_hash() -> str:
    """SHA-256 fingerprint of the CODE that produces a result: all Python under src/ and
    experiments/.  This is environment-portable (byte-identical on the local checkout and the
    server, both LF), so check_consistency.py can recompute it from the committed tree and
    fail if that tree does not reproduce the stamp a headline result carries.  The dependency
    environment (torch/numpy/scipy versions, host, GPU) is stamped separately in ``_meta``;
    it is deliberately NOT folded in here, since the pinned lock describes the CI/local
    environment, not necessarily the server conda environment that produced a given result."""
    import hashlib

    h = hashlib.sha256()
    root = RESULTS.parent
    files = sorted(list((root / "src").rglob("*.py")) + list((root / "experiments").rglob("*.py")))
    for f in files:
        try:
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes())
        except Exception:
            pass
    return h.hexdigest()[:16]


def run_metadata(config: dict | None = None) -> dict:
    """Full provenance stamp attached to every result JSON: exact source + environment +
    git state (with a DIRTY flag) + command + timestamp + host/CPU/GPU.  Dirty-tree or
    unknown-provenance results must not be used as paper headline (checked by the CI
    consistency test)."""
    import platform
    import subprocess
    import datetime
    import hashlib
    import os

    import numpy
    import scipy

    def _git(args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=RESULTS.parent, timeout=10).stdout.strip() or None
        except Exception:
            return None

    full = _git(["rev-parse", "HEAD"])
    porcelain = _git(["status", "--porcelain"])
    dirty = None if porcelain is None else (porcelain != "")
    if full is None:
        # server tarball sync ships GIT_COMMIT as "<full-sha>\n<clean|dirty>"
        stamp = RESULTS.parent / "GIT_COMMIT"
        if stamp.exists():
            parts = stamp.read_text().split()
            full = parts[0] if parts else None
            if dirty is None and len(parts) > 1:
                dirty = (parts[1].strip().lower() == "dirty")
    meta = {
        "git_commit": full,
        "git_short": (full[:8] if full else None),
        "git_dirty": dirty,
        "source_tree_sha256": _source_tree_hash(),
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    if config is not None:
        meta["config_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    try:
        import torch  # noqa

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        meta["torch"] = None
    return meta


def save_json(name: str, obj: dict) -> Path:
    ensure_dirs()
    if isinstance(obj, dict) and "_meta" not in obj:
        obj = {**obj, "_meta": run_metadata(obj.get("config"))}
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def savefig(fig, name: str) -> Path:
    ensure_dirs()
    for sub in (FIGDIR, RESULTS.parent / "paper" / "figures"):
        sub.mkdir(parents=True, exist_ok=True)
        fig.savefig(sub / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return FIGDIR / name
