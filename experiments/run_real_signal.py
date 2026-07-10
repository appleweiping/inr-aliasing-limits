r"""E3/E4/E5 -- the learned-Nyquist limit on real 1-D signals.

For a real signal we cannot choose the spectrum, so we demonstrate the limit by sweeping the
INR's representable bandwidth ``B_INR``.  A too-narrow INR cannot represent the signal's band
and **aliases** it (folded spectrum, high error); a bandwidth-matched INR **recovers** it.
The reconstruction-error-vs-``B_INR`` curve has a knee at the signal's essential bandwidth
``B_sig`` -- the empirical learned-Nyquist rate -- and the recovered spectrum at a narrow
``B_INR`` shows the predicted folding, flagged by the ground-truth-free diagnostic.

Usage:  python experiments/run_real_signal.py <name>   (name in data/<name>.npz)

"Both sides of the limit" across signals: a smooth/low-bandwidth signal (small ``B_sig``,
recovered at modest width) vs a broadband signal (large ``B_sig``, aliased unless wide).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _util import save_json, savefig, RESULTS
from inralias.signals import lowpass_dictionary, nonuniform_times, evaluate
from inralias.inr import FixedFeatureINR, nonuniform_periodogram, torch_available
from inralias.diagnostics import extended_dictionary_test

import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "data"


def load_signal(name: str, M: int = 2000) -> np.ndarray:
    """Load data/<name>.npz, take a mono segment, decimate to <=M samples, normalise."""
    d = np.load(DATA / f"{name}.npz")
    x = np.asarray(d["x"], float).reshape(-1)
    # pick the most energetic contiguous window of length ~M*k, then decimate to M
    if x.size > M:
        # choose window with max energy
        w = min(x.size, 8 * M)
        if x.size > w:
            energy = np.convolve(x**2, np.ones(w), "valid")
            s = int(np.argmax(energy))
            x = x[s:s + w]
        step = max(1, x.size // M)
        x = x[::step][:M]
    # linear detrend: a Fourier-feature INR represents periodic content; a monotonic trend is
    # non-periodic (its periodic extension has a jump -> Gibbs), so we remove the best-fit line
    # as standard preprocessing before studying the bandwidth limit.
    n = np.arange(x.size)
    x = x - np.polyval(np.polyfit(n, x, 1), n)
    x = (x - x.mean()) / (np.std(x) + 1e-9)
    return x


def essential_bandwidth(x: np.ndarray, keep: float = 0.95, bmax: int | None = None) -> int:
    """Essential two-sided bandwidth (in cycles/record) capturing ``keep`` spectral energy."""
    M = x.size
    X = np.abs(np.fft.rfft(x)) ** 2
    X[0] = 0  # ignore DC for bandwidth
    csum = np.cumsum(X) / (X.sum() + 1e-30)
    B = int(np.searchsorted(csum, keep))
    if bmax is None:
        bmax = M // 4
    return max(2, min(B, bmax))


def _fit_rmse(B_inr: int, t_samp, y_samp, t_ref, x_ref, ridge=1e-6) -> tuple[float, dict]:
    Lam = lowpass_dictionary(B_inr)
    inr = FixedFeatureINR(Lam, real=True).fit(t_samp, y_samp, ridge=ridge)
    f_hat = inr.predict(t_ref)
    rmse = float(np.sqrt(np.mean((f_hat - x_ref) ** 2)))
    ring_hi = min(B_inr + max(4, B_inr // 3), len(x_ref) // 2 - 1)
    ring = np.concatenate([np.arange(B_inr + 1, ring_hi + 1),
                           -np.arange(B_inr + 1, ring_hi + 1)]).astype(float)
    diag = extended_dictionary_test(Lam, t_samp, y_samp, ring, ridge=ridge)
    return rmse, diag


def run(name: str, density: float = 3.0, snr_db: float = 30.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_ref = load_signal(name)
    M = x_ref.size
    t_ref = np.arange(M) / M
    B_sig = essential_bandwidth(x_ref)

    # nonuniform noisy samples at density * Nyquist(B_sig)
    N = int(min(M, max(2 * B_sig + 5, round(density * 2 * B_sig))))
    idx = np.sort(rng.choice(M, size=N, replace=False))
    t_samp = t_ref[idx]
    sig_p = np.mean(x_ref**2)
    noise_std = np.sqrt(sig_p / (10 ** (snr_db / 10)))
    y_samp = x_ref[idx] + rng.normal(0, noise_std, N)

    # sweep INR bandwidth
    Bmax = min(int(1.8 * B_sig) + 4, N // 2 - 1, M // 2 - 1)
    grid = np.unique(np.linspace(max(2, B_sig // 6), Bmax, 26).astype(int))
    rmse, oob = [], []
    for B in grid:
        r, d = _fit_rmse(int(B), t_samp, y_samp, t_ref, x_ref)
        rmse.append(r); oob.append(d["out_of_band_frac"])
    rmse = np.array(rmse); oob = np.array(oob)

    # knee: smallest B within 1.3x of the min RMSE
    floor = rmse.min()
    knee_idx = int(np.argmax(rmse <= 1.3 * floor))
    B_knee = int(grid[knee_idx])

    # narrow vs matched spectra
    B_narrow = int(max(2, B_sig // 3))
    B_match = int(min(Bmax, int(1.3 * B_sig)))
    specs = {}
    for tag, B in (("narrow", B_narrow), ("matched", B_match)):
        Lam = lowpass_dictionary(B)
        inr = FixedFeatureINR(Lam, real=True).fit(t_samp, y_samp, ridge=1e-6)
        specs[tag] = {"B": B, "freqs": Lam.tolist(),
                      "coef_abs": np.abs(inr.coeffs_).tolist(),
                      "recon": inr.predict(t_ref)}

    # ---- figures ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax1.plot(grid, rmse, "o-", ms=3, color="C3")
    ax1.axvline(B_sig, color="C0", ls="--", lw=1.2, label=f"$B_{{sig}}$={B_sig}")
    ax1.axvline(B_knee, color="k", ls=":", lw=1.0, label=f"knee={B_knee}")
    ax1.set_xlabel("INR bandwidth $B_{INR}$"); ax1.set_ylabel("reconstruction RMSE")
    ax1.set_yscale("log"); ax1.set_title(f"{name}: learned-Nyquist knee"); ax1.legend(fontsize=7)

    Xref = np.abs(np.fft.rfft(x_ref)); fr = np.arange(Xref.size)
    ax2.plot(fr, Xref / Xref.max(), color="0.6", lw=1.0, label="true spectrum")
    ax2.axvline(B_narrow, color="C1", ls="--", lw=1.0, label=f"narrow $B$={B_narrow}")
    ax2.axvline(B_match, color="C2", ls="--", lw=1.0, label=f"matched $B$={B_match}")
    ax2.set_xlim(0, min(Bmax * 1.5, Xref.size))
    ax2.set_xlabel("frequency (cycles/record)"); ax2.set_ylabel("|X| (norm)")
    ax2.set_title("spectrum & INR bands"); ax2.legend(fontsize=7)
    savefig(fig, f"real_{name}.png")

    out = {
        "name": name, "M": M, "B_sig": B_sig, "N": N, "snr_db": snr_db, "density": density,
        "grid": grid.tolist(), "rmse": rmse.tolist(), "out_of_band_frac": oob.tolist(),
        "B_knee": B_knee, "rmse_floor": float(floor),
        "narrow": {"B": B_narrow, "rmse": float(_fit_rmse(B_narrow, t_samp, y_samp, t_ref, x_ref)[0])},
        "matched": {"B": B_match, "rmse": float(_fit_rmse(B_match, t_samp, y_samp, t_ref, x_ref)[0])},
        "note": "RMSE-vs-B_INR knee at essential bandwidth; narrow INR aliases (folds), matched recovers",
    }
    save_json(f"real_{name}.json", out)
    print(f"[real:{name}] M={M} B_sig={B_sig} N={N} knee={B_knee} "
          f"narrow_rmse={out['narrow']['rmse']:.3f} matched_rmse={out['matched']['rmse']:.3f}",
          flush=True)
    return out


def main():
    names = sys.argv[1:] or ["speech", "sunspots"]
    for n in names:
        if (DATA / f"{n}.npz").exists():
            run(n)
        else:
            print(f"[real:{n}] data/{n}.npz missing -- run data/fetch_real.py first", flush=True)


if __name__ == "__main__":
    main()
