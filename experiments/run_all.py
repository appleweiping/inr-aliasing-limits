r"""Regenerate every result and figure.

Order:
  0. fetch real data (data/fetch_real.py) if the .npz files are missing;
  1. E1 phase transition (CPU);
  2. E2 silent aliasing (CPU);
  3. real-signal learned-Nyquist sweeps (CPU): co2, sunspots_smooth, speech, sunspots;
  4. trained nonlinear INR persistence + 2-D image demo (GPU/torch, skipped if torch absent).

Every experiment uses a fixed seed, so results are deterministic. The GPU steps are the only
ones needing torch; they are skipped with a message if torch is unavailable (run them on the
server -- see README).
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
    missing = [n for n in needed if not (DATA / f"{n}.npz").exists()]
    if missing:
        print(f"[run_all] fetching missing real data: {missing}", flush=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("fetch_real", DATA / "fetch_real.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main(missing)


def main():
    _ensure_data()

    import run_phase_transition
    print("\n=== E1 phase transition ==="); run_phase_transition.main()

    import run_silent_aliasing
    print("\n=== E2 silent aliasing ==="); run_silent_aliasing.main()

    import run_real_signal
    print("\n=== real-data learned-Nyquist sweeps ===")
    sys.argv = ["run_real_signal", "co2", "sunspots_smooth", "speech", "sunspots"]
    run_real_signal.main()

    from inralias.inr import torch_available
    if torch_available():
        import run_nonlinear
        print("\n=== nonlinear INR persistence (torch) ==="); run_nonlinear.main()
        import run_image2d
        print("\n=== 2-D image demo (torch) ==="); run_image2d.run()
    else:
        print("\n[run_all] torch unavailable -- skipping nonlinear + 2-D demos "
              "(run experiments/run_nonlinear.py and run_image2d.py on the GPU server)", flush=True)

    print("\n[run_all] done. Results in results/ and results/figures/.", flush=True)


if __name__ == "__main__":
    main()
