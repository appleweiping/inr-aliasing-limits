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
import matplotlib.pyplot as plt  # noqa: E402,F401

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"


def ensure_dirs() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)


def save_json(name: str, obj: dict) -> Path:
    ensure_dirs()
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
