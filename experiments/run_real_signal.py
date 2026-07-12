r"""E3 -- deployable, sample-only bandwidth selection on real signals.

What changed relative to the earlier revision (and why):

* **No oracle leakage.**  The full reference signal is used ONLY for evaluation.  Every
  deployable method selects its bandwidth (and any regularization) from the training
  samples alone.  The oracle selector is reported separately, as an upper bound.
* **Anti-aliased preprocessing.**  Decimation uses ``scipy.signal.resample_poly`` (with
  its built-in low-pass filter) -- never bare ``x[::step]``.  Physical sample rates and
  record durations are carried through and reported.
* **No cherry-picking.**  Full series are used (fixed deterministic windows); speech uses
  9 disjoint 2-second segments from 3 different Open Speech Repository recordings, and
  results are reported as paired statistics (mean +- 95% CI over segments x seeds).
* **Smoothed sunspots** is explicitly an intentionally low-pass PROXY, not a natural
  signal, and is labelled as such everywhere.

Methods compared (per segment, per sampling seed):
  oracle          band from the full reference spectrum (evaluation upper bound ONLY);
  ls_periodogram  Lomb--Scargle band estimate from the training samples;
  corr_periodogram  correlation-periodogram band estimate (baseline);
  cv              bandwidth minimizing 5-fold cross-validated prediction error;
  fixed_narrow    constant B=8;
  fixed_wide      constant B=N//4;
  ridge_wide      B=N//3 with ridge strength chosen by cross-validation.

Usage:  python experiments/run_real_signal.py [names...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from _util import save_json, savefig
from inralias.signals import lowpass_dictionary
from inralias.inr import FixedFeatureINR, bandwidth_matched_freqs

import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "data"
M_TARGET = 2000
SNR_DB = 30.0
N_SEEDS = 10
SAMPLE_FRAC = 0.5
CI = 1.96

METHODS = ("oracle", "ls_periodogram", "corr_periodogram", "cv",
           "fixed_narrow", "fixed_wide", "ridge_wide")


def _standardize(x):
    x = np.asarray(x, float)
    n = np.arange(x.size)
    x = x - np.polyval(np.polyfit(n, x, 1), n)   # linear detrend (documented)
    return (x - x.mean()) / (np.std(x) + 1e-9)


def load_segments(name: str):
    """Return a list of dicts {x, fs_eff, duration_s, label} -- anti-aliased, full-record."""
    segs = []
    if name == "speech":
        for rec in range(3):
            d = np.load(DATA / f"speech_rec{rec}.npz")
            x, fs = np.asarray(d["x"], float), float(d["fs"])
            seg_len = int(2.0 * fs)                      # 2-second segments
            # 3 disjoint segments per recording, deterministic offsets (skip lead-in)
            for si, start in enumerate([int(1.0 * fs), int(12.0 * fs), int(24.0 * fs)]):
                seg = x[start:start + seg_len]
                if seg.size < seg_len:
                    continue
                down = seg.size // M_TARGET              # 16000 -> 2000: down=8
                xs = resample_poly(seg, up=1, down=down)[:M_TARGET]
                segs.append({"x": _standardize(xs), "fs_eff": fs / down,
                             "duration_s": seg.size / fs,
                             "label": f"rec{rec}_seg{si}"})
    else:
        d = np.load(DATA / f"{name}.npz")
        x, fs = np.asarray(d["x"], float), float(d.get("fs", 1.0))
        if x.size > M_TARGET:
            down = int(np.ceil(x.size / M_TARGET))
            xs = resample_poly(x, up=1, down=down)[:M_TARGET]
            fs_eff = fs / down
        else:
            xs, fs_eff = x, fs
        segs.append({"x": _standardize(xs), "fs_eff": fs_eff,
                     "duration_s": xs.size / fs_eff, "label": name})
    return segs


def _band_from_reference(x_ref, keep=0.95):
    X = np.abs(np.fft.rfft(x_ref)) ** 2
    X[0] = 0
    csum = np.cumsum(X) / (X.sum() + 1e-30)
    return max(2, min(int(np.searchsorted(csum, keep)), x_ref.size // 4))


def _fit_eval(B, t, y, t_ref, x_ref, ridge=0.0):
    model = FixedFeatureINR(lowpass_dictionary(int(B)), real=True).fit(t, y, ridge=ridge)
    return float(np.sqrt(np.mean((model.predict(t_ref) - x_ref) ** 2)))


def _cv_error(B, t, y, ridge=0.0, k=5, rng=None):
    idx = np.random.default_rng(rng).permutation(t.size)
    folds = np.array_split(idx, k)
    errs = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        model = FixedFeatureINR(lowpass_dictionary(int(B)), real=True).fit(
            t[tr], y[tr], ridge=ridge)
        errs.append(np.mean((model.predict(t[f]) - y[f]) ** 2))
    return float(np.mean(errs))


def select_bandwidths(t, y, N, x_ref, seed):
    """All selectors; only `oracle` sees x_ref."""
    out = {}
    Bmax_est = N // 4
    out["oracle"] = (_band_from_reference(x_ref), 0.0)
    out["ls_periodogram"] = (
        int(np.max(bandwidth_matched_freqs(t, y, Bmax_est, 0.95, method="lombscargle"))), 0.0)
    out["corr_periodogram"] = (
        int(np.max(bandwidth_matched_freqs(t, y, Bmax_est, 0.95, method="correlation"))), 0.0)
    Bgrid = np.unique(np.geomspace(4, Bmax_est, 16).astype(int))
    cv_errs = [_cv_error(B, t, y, rng=seed) for B in Bgrid]
    out["cv"] = (int(Bgrid[int(np.argmin(cv_errs))]), 0.0)
    out["fixed_narrow"] = (8, 0.0)
    out["fixed_wide"] = (N // 4, 0.0)
    lam_grid = [1e-6, 1e-4, 1e-2, 1.0]
    lam_errs = [_cv_error(N // 3, t, y, ridge=lam, rng=seed) for lam in lam_grid]
    out["ridge_wide"] = (N // 3, float(lam_grid[int(np.argmin(lam_errs))]))
    return out


def run(name: str):
    segs = load_segments(name)
    rows = {m: [] for m in METHODS}          # RMSE per (segment, seed)
    bands = {m: [] for m in METHODS}
    for seg in segs:
        x_ref = seg["x"]
        M = x_ref.size
        t_ref = np.arange(M) / M
        N = int(SAMPLE_FRAC * M)
        noise_std = np.sqrt(np.mean(x_ref**2) / 10 ** (SNR_DB / 10))
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + seed)
            idx = np.sort(rng.choice(M, size=N, replace=False))
            t, y = t_ref[idx], x_ref[idx] + rng.normal(0, noise_std, N)
            sel = select_bandwidths(t, y, N, x_ref, seed)
            for meth, (B, lam) in sel.items():
                rows[meth].append(_fit_eval(B, t, y, t_ref, x_ref, ridge=lam))
                bands[meth].append(int(B))
    summary = {}
    oracle = np.asarray(rows["oracle"])
    for m in METHODS:
        v = np.asarray(rows[m])
        diff = v - oracle
        summary[m] = {
            "rmse_mean": float(v.mean()),
            "rmse_ci95": float(CI * v.std(ddof=1) / np.sqrt(v.size)),
            "paired_diff_vs_oracle_mean": float(diff.mean()),
            "paired_diff_ci95": float(CI * diff.std(ddof=1) / np.sqrt(diff.size))
            if m != "oracle" else 0.0,
            "band_mean": float(np.mean(bands[m])),
            "n": int(v.size),
        }
    out = {
        "name": name,
        "n_segments": len(segs), "n_seeds": N_SEEDS, "snr_db": SNR_DB,
        "sample_frac": SAMPLE_FRAC, "M": int(segs[0]["x"].size),
        "fs_eff_hz": segs[0]["fs_eff"], "duration_s": segs[0]["duration_s"],
        "segments": [s["label"] for s in segs],
        "methods": summary,
        "lowpass_proxy": bool(name == "sunspots_smooth"),
        "note": "oracle sees the full reference (upper bound only); all other methods "
                "are sample-only; anti-aliased resample_poly decimation; full records, "
                "no energy-based window selection",
    }
    save_json(f"real_{name}.json", out)
    line = "  ".join(f"{m}={summary[m]['rmse_mean']:.3f}±{summary[m]['rmse_ci95']:.3f}"
                     for m in METHODS)
    print(f"[real:{name}] segs={len(segs)} " + line, flush=True)
    return out


def figure(results):
    """Grouped bar chart: RMSE mean +- CI per method, per signal."""
    names = [r["name"] for r in results]
    fig, ax = plt.subplots(figsize=(7.4, 3.0), layout="constrained")
    width = 0.11
    xs = np.arange(len(names))
    for i, m in enumerate(METHODS):
        mu = [r["methods"][m]["rmse_mean"] for r in results]
        ci = [r["methods"][m]["rmse_ci95"] for r in results]
        ax.bar(xs + (i - len(METHODS) / 2) * width, mu, width, yerr=ci,
               label=m.replace("_", " "), capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([n + ("\n(low-pass proxy)" if n == "sunspots_smooth" else "")
                        for n in names], fontsize=11)
    ax.set_ylabel("reconstruction RMSE")
    ax.set_yscale("log")
    ax.set_title("sample-only bandwidth selection vs oracle (mean ± 95% CI)")
    ax.legend(fontsize=8.5, ncols=2)
    savefig(fig, "real_signals.png")


def main():
    names = sys.argv[1:] or ["speech", "co2", "sunspots", "sunspots_smooth"]
    results = []
    for n in names:
        try:
            results.append(run(n))
        except FileNotFoundError as e:
            print(f"[real:{n}] missing data ({e}); run data/fetch_real.py", flush=True)
    if results:
        figure(results)


if __name__ == "__main__":
    main()
