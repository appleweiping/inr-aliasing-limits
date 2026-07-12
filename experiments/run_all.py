r"""Reproduction entry points.

CPU-only reproduction (regenerates E1--E3 results and figures):
    python experiments/run_all.py

Full reproduction (additionally regenerates the trained-network studies E4--E5 part B;
REQUIRES torch + a GPU for reasonable runtime):
    python experiments/run_all.py --full

Order:
  0. fetch real data (data/fetch_real.py) if the .npz files are missing;
  1. E1 synthetic theorem-validation matrix (CPU);
  2. E2 detection operating characteristics (CPU);
  3. E3 real-signal sample-only bandwidth selection (CPU);
  4. E5 part A: linear 2-D exact fold (CPU; part B skipped without --full/torch);
  5. [--full] E4 trained nonlinear study + E5 part B (torch).

If torch is absent, this script reproduces ONLY the CPU results and says so explicitly --
it does NOT claim to have regenerated the trained-network results.  Fixed seeds
throughout; GPU training uses deterministic algorithms where available but bitwise
reproducibility across CUDA versions is not guaranteed (see README).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
DATA = ROOT / "data"


def _ensure_data():
    needed = ["co2", "sunspots_smooth", "speech", "sunspots"]
    def _have(n):
        if n == "speech":
            return (DATA / "speech_rec0.npz").exists()
        return (DATA / f"{n}.npz").exists()
    missing = [n for n in needed if not _have(n)]
    if missing:
        print(f"[run_all] fetching missing real data: {missing}", flush=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("fetch_real", DATA / "fetch_real.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main(missing)


def main():
    full = "--full" in sys.argv
    _ensure_data()

    import run_synthetic_matrix
    print("\n=== E1 synthetic theorem-validation matrix ==="); run_synthetic_matrix.main()

    import run_diagnostic_roc
    print("\n=== E2 detection operating characteristics ===")
    sys.argv = ["run_diagnostic_roc"]
    run_diagnostic_roc.main()

    import run_real_signal
    print("\n=== E3 real-signal sample-only selection ===")
    sys.argv = ["run_real_signal", "speech", "co2", "sunspots", "sunspots_smooth"]
    run_real_signal.main()

    from inralias.inr import torch_available
    if full and torch_available():
        import run_nonlinear
        print("\n=== E4 trained nonlinear study (torch) ===")
        sys.argv = ["run_nonlinear"]
        run_nonlinear.main()
        import run_image2d
        print("\n=== E5 2-D study (torch) ===")
        sys.argv = ["run_image2d"]
        run_image2d.main()
        print("\n[run_all] done: ALL results (CPU + trained-network) regenerated.",
              flush=True)
    else:
        import run_image2d
        print("\n=== E5 part A: linear 2-D exact fold (CPU) ===")
        import numpy as _np
        res, *_ = run_image2d.part_a_linear(_np.random.default_rng(3))
        print(f"[2d-linear] fold predicted={res['predicted_fold']} "
              f"measured={res['measured_fold']} exact={res['exact']}", flush=True)
        why = "--full not given" if torch_available() else "torch unavailable"
        print(f"\n[run_all] done: CPU results regenerated. Trained-network results "
              f"(E4, E5 part B) were NOT regenerated ({why}); the released JSONs for "
              f"those were produced on the GPU server (see _meta stamps).", flush=True)


if __name__ == "__main__":
    main()
