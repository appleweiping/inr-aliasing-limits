r"""E3 -- deployable, sample-only bandwidth selection on real signals.

Design decisions (and the audit points they answer):

* **No preprocessing leakage (P0-I).**  The full reference signal is used ONLY for
  evaluation.  Detrending and standardization are estimated *from the training samples of
  each draw alone* (``_sample_transform``); the reference is mapped into that same
  sample-estimated frame for scoring.  Nothing a method sees depends on statistics of the
  held-out samples.  (An earlier version standardized the whole record up front -- that
  leaked the full-record mean/variance/trend into every training set.)
* **Honest oracle (P0-I).**  The oracle minimizes the TRUE reconstruction error over the
  *entire method family* -- the full ``bandwidth x ridge`` grid actually available to the
  deployable selectors, not just bandwidth at ridge 0.  Because every deployable selector's
  chosen ``(B, ridge)`` is inside the oracle's search set, the oracle dominates every
  selector on every draw BY CONSTRUCTION; we ``assert`` this each draw.
* **Blocks, not masks, are the replication unit (P0-I).**  Short single records (CO2,
  sunspots) are split into disjoint contiguous time BLOCKS.  Confidence intervals cluster
  on blocks (the honest unit), never on mask/noise seeds within one record.  Speech uses 9
  disjoint 2 s segments from 3 recordings.
* **Anti-aliased preprocessing.**  Speech decimation uses ``resample_poly`` (built-in
  low-pass); non-speech series are kept at native cadence (no bare ``x[::step]``).
* **Correct physical units (P0-I).**  CO2 and sunspots carry ``1/yr`` / ``1/month`` rates
  and year/month time axes -- never Hz or seconds (those are speech only).
* **Stated synthetic degradation.**  Every real trace is degraded by 50% uniform-random
  missing samples + additive white Gaussian noise at 30 dB SNR (reported in the JSON).
* **Smoothed sunspots** is an intentionally low-pass PROXY, labelled as such everywhere.

Methods (per block, per sampling seed):
  oracle          best (B, ridge) in the family by TRUE error (evaluation upper bound ONLY);
  ls_periodogram  Lomb--Scargle band estimate from the training samples;
  corr_periodogram  correlation-periodogram band estimate (baseline);
  cv              (B) minimizing 5-fold CV prediction error;
  fixed_narrow    constant B=8;
  fixed_wide      constant B=N//4;
  ridge_wide      B=N//3 with ridge strength chosen by 5-fold CV.

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
M_TARGET = 2000          # speech decimation target
SNR_DB = 30.0            # additive white Gaussian noise level (stated degradation)
N_SEEDS = 6              # mask/noise seeds per block (NOT the CI replication unit -- the
                         # disjoint time BLOCKS are; seeds only average within a block)
SAMPLE_FRAC = 0.5        # 50% uniform-random missingness (stated degradation)
LAM_GRID = (0.0, 1e-4, 1e-2, 1.0)
B_GRID_CAP = 128         # cap the bandwidth GRID (oracle/CV) to bound fit size; the
                         # fixed_wide/ridge_wide baselines may still exceed it

METHODS = ("oracle", "ls_periodogram", "corr_periodogram", "cv",
           "fixed_narrow", "fixed_wide", "ridge_wide")

# Physical units of each archived series -- so CO2/sunspots are never mislabelled as Hz/s.
SIGNAL_META = {
    "speech":          {"rate_unit": "Hz",     "time_unit": "s"},      # 8 kHz audio
    "co2":             {"rate_unit": "1/yr",   "time_unit": "yr"},     # Mauna Loa monthly (12/yr)
    "sunspots":        {"rate_unit": "1/month", "time_unit": "month"}, # SILSO monthly index
    "sunspots_smooth": {"rate_unit": "1/yr",   "time_unit": "yr"},     # 13-month smoothed monthly
}


def _sample_transform(t, y):
    """Linear detrend + standardization estimated from the TRAINING SAMPLES ONLY.

    Returns (poly, mu, sd) so the identical map can be applied to the reference for scoring.
    No held-out sample influences these coefficients -- this is what removes the leakage."""
    p = np.polyfit(t, y, 1)
    r = y - np.polyval(p, t)
    return p, float(r.mean()), float(r.std() + 1e-9)


def _apply(p, mu, sd, tt, xx):
    return (xx - np.polyval(p, tt) - mu) / sd


def load_segments(name: str):
    """Return blocks {x_raw, rate, rate_unit, time_unit, duration, label}.

    Speech: 9 disjoint anti-aliased 2 s segments.  Other series: disjoint contiguous time
    BLOCKS at native cadence (the CI replication unit)."""
    meta = SIGNAL_META[name]
    segs = []
    if name == "speech":
        for rec in range(3):
            d = np.load(DATA / f"speech_rec{rec}.npz")
            x, fs = np.asarray(d["x"], float), float(d["fs"])
            seg_len = int(2.0 * fs)                              # 2 s segments
            for si, start in enumerate([int(1.0 * fs), int(12.0 * fs), int(24.0 * fs)]):
                seg = x[start:start + seg_len]
                if seg.size < seg_len:
                    continue
                down = max(1, seg.size // M_TARGET)              # 16000 -> 2000: down=8
                xs = resample_poly(seg, up=1, down=down)[:M_TARGET]
                # cluster = RECORDING (segments from one recording share a speaker and are
                # NOT independent replicates), so the honest CI replication unit is the record.
                segs.append({"x_raw": xs, "rate": fs / down, "cluster": rec,
                             "rate_unit": meta["rate_unit"], "time_unit": meta["time_unit"],
                             "duration": seg.size / fs, "label": f"rec{rec}_seg{si}"})
    else:
        d = np.load(DATA / f"{name}.npz")
        x, fs = np.asarray(d["x"], float), float(d.get("fs", 1.0))
        # >=3 disjoint contiguous blocks, each >= ~200 samples so a bandwidth family fits.
        nblk = int(np.clip(x.size // 250, 3, 6))
        blen = x.size // nblk
        for bi in range(nblk):
            seg = x[bi * blen:(bi + 1) * blen]
            segs.append({"x_raw": seg, "rate": fs, "cluster": bi,
                         "rate_unit": meta["rate_unit"], "time_unit": meta["time_unit"],
                         "duration": seg.size / fs, "label": f"{name}_blk{bi}"})
    return segs


def _fit_eval(B, t, y, t_ref, x_ref_t, ridge=0.0):
    model = FixedFeatureINR(lowpass_dictionary(int(B)), real=True).fit(t, y, ridge=ridge)
    return float(np.sqrt(np.mean((model.predict(t_ref) - x_ref_t) ** 2)))


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


def select_bandwidths(t, y, N, t_ref, x_ref_t, seed):
    """Deployable selectors (sample-only) + an oracle over the whole (B, ridge) family.

    All fitting/scoring happens in the sample-estimated standardized frame (``y`` and
    ``x_ref_t`` are already transformed).  The oracle searches the full Bgrid x LAM_GRID
    product plus every selector's own pick, so oracle error <= every selector's error on
    this draw by construction (asserted by the caller)."""
    out = {}
    Bmax = max(8, N // 4)
    # grid capped to bound the oracle/CV fit size; fixed_wide/ridge_wide (below) may exceed
    Bgrid = np.unique(np.geomspace(4, min(Bmax, B_GRID_CAP), 10).astype(int))
    out["ls_periodogram"] = (
        int(np.max(bandwidth_matched_freqs(t, y, Bmax, 0.95, method="lombscargle"))), 0.0)
    out["corr_periodogram"] = (
        int(np.max(bandwidth_matched_freqs(t, y, Bmax, 0.95, method="correlation"))), 0.0)
    cv_errs = [_cv_error(B, t, y, rng=seed) for B in Bgrid]
    out["cv"] = (int(Bgrid[int(np.argmin(cv_errs))]), 0.0)
    out["fixed_narrow"] = (8, 0.0)
    out["fixed_wide"] = (N // 4, 0.0)
    lam_errs = [_cv_error(N // 3, t, y, ridge=lam, rng=seed) for lam in LAM_GRID]
    out["ridge_wide"] = (N // 3, float(LAM_GRID[int(np.argmin(lam_errs))]))
    # Oracle: whole method family (bandwidth x ridge) + every selector's realized pick.
    cands = [(int(B), float(lam)) for B in Bgrid for lam in LAM_GRID]
    cands += [(int(b), float(lam)) for (b, lam) in out.values()]
    cands = sorted(set(cands))
    o_err = [_fit_eval(B, t, y, t_ref, x_ref_t, ridge=lam) for (B, lam) in cands]
    out["oracle"] = cands[int(np.argmin(o_err))]
    return out


def run(name: str):
    segs = load_segments(name)
    rows = {m: [] for m in METHODS}          # RMSE per (block, seed)
    seg_of = []                              # block index per row (for clustered CIs)
    bands = {m: [] for m in METHODS}
    for gi, seg in enumerate(segs):
        x_raw = np.asarray(seg["x_raw"], float)
        M = x_raw.size
        t_ref = np.arange(M) / M
        N = int(SAMPLE_FRAC * M)
        # 30 dB AWGN defined w.r.t. this block's clean RMS (a simulation parameter, not
        # something any selector reads).
        noise_std = np.sqrt(np.mean(x_raw ** 2) / 10 ** (SNR_DB / 10))
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 * (gi + 1) + seed)   # decorrelated per (block, seed)
            idx = np.sort(rng.choice(M, size=N, replace=False))
            t = t_ref[idx]
            y_raw = x_raw[idx] + rng.normal(0, noise_std, N)
            # sample-only preprocessing; map the reference into the SAME frame for scoring
            p, mu, sd = _sample_transform(t, y_raw)
            y = _apply(p, mu, sd, t, y_raw)
            x_ref_t = _apply(p, mu, sd, t_ref, x_raw)
            sel = select_bandwidths(t, y, N, t_ref, x_ref_t, seed)
            draw = {}
            for meth, (B, lam) in sel.items():
                draw[meth] = _fit_eval(B, t, y, t_ref, x_ref_t, ridge=lam)
                rows[meth].append(draw[meth])
                bands[meth].append(int(B))
            # honest-oracle guarantee: upper bound must dominate every deployable selector
            for meth in METHODS:
                if meth != "oracle":
                    assert draw["oracle"] <= draw[meth] + 1e-9, (name, gi, seed, meth)
            seg_of.append(seg["cluster"])          # cluster = recording (speech) or block
    seg_of = np.asarray(seg_of)
    clusters = sorted(set(seg_of.tolist()))        # honest replication unit
    n_clusters = len(clusters)

    def _ci(values):
        """95% CI half-width, clustered on the honest replication unit (RECORDING for
        speech -- same-recording segments share a speaker and are not independent -- or
        time BLOCK otherwise): t-interval on cluster-level means.  Mask/noise seeds are NOT
        treated as independent replicates."""
        from scipy.stats import t as tdist
        v = np.asarray(values)
        means = np.array([v[seg_of == c].mean() for c in clusters])
        if n_clusters < 2:
            return float("nan")
        return float(tdist.ppf(0.975, n_clusters - 1) * means.std(ddof=1) / np.sqrt(n_clusters))

    summary = {}
    oracle = np.asarray(rows["oracle"])
    for m in METHODS:
        v = np.asarray(rows[m])
        diff = v - oracle
        summary[m] = {
            "rmse_mean": float(v.mean()),
            "rmse_ci95": _ci(v),
            "paired_diff_vs_oracle_mean": float(diff.mean()),
            "paired_diff_ci95": _ci(diff) if m != "oracle" else 0.0,
            "band_mean": float(np.mean(bands[m])),
            "n": int(v.size),
            "n_clusters": n_clusters,
        }
    s0 = segs[0]
    out = {
        "name": name,
        "n_blocks": len(segs), "n_seeds": N_SEEDS, "snr_db": SNR_DB,
        "sample_frac": SAMPLE_FRAC,
        "block_len": int(s0["x_raw"].size),
        "rate": s0["rate"], "rate_unit": s0["rate_unit"],
        "duration": s0["duration"], "time_unit": s0["time_unit"],
        "blocks": [s["label"] for s in segs],
        "methods": summary,
        "lowpass_proxy": bool(name == "sunspots_smooth"),
        "degradation": f"50% uniform-random missing samples + AWGN at {SNR_DB:.0f} dB SNR",
        "note": "oracle = best (B,ridge) in the family by TRUE error (evaluation upper "
                "bound only, dominates every selector per draw); all other methods are "
                "sample-only with detrend/standardization estimated from the training "
                "samples of each draw (no full-record leakage); CIs cluster on disjoint "
                "time blocks, not mask seeds; anti-aliased decimation (speech only).",
    }
    save_json(f"real_{name}.json", out)
    line = "  ".join(f"{m}={summary[m]['rmse_mean']:.3f}±{summary[m]['rmse_ci95']:.3f}"
                     for m in METHODS)
    print(f"[real:{name}] blocks={len(segs)} rate={s0['rate']:g} {s0['rate_unit']} " + line,
          flush=True)
    return out


def figure(results):
    """Grouped bar chart: RMSE mean +- block-clustered CI per method, per signal."""
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
    ax.set_title("sample-only bandwidth selection vs oracle (mean ± 95% CI, block-clustered)")
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
