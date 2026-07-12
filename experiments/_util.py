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


def run_metadata() -> dict:
    """Environment + provenance stamp attached to every result JSON."""
    import platform
    import subprocess

    import numpy
    import scipy

    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import torch  # noqa

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        meta["torch"] = None
    try:
        meta["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=RESULTS.parent, timeout=10,
        ).stdout.strip() or None
    except Exception:
        meta["git_commit"] = None
    return meta


def save_json(name: str, obj: dict) -> Path:
    ensure_dirs()
    if isinstance(obj, dict) and "_meta" not in obj:
        obj = {**obj, "_meta": run_metadata()}
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
