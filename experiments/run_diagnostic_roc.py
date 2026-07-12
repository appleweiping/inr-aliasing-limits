r"""E2 -- operating characteristic of sample-only aliasing detectors.

Protocol (strictly honest):

* **calibration / test separation**: for every condition, decision thresholds are set on
  an independent calibration set of H0 draws (95th percentile) and *never* touched again;
  the reported FPR/TPR come from a disjoint test set.
* **detectors** (all sample-only):
  - ``ring``      recovered out-of-band energy fraction of the extended-dictionary fit;
  - ``residual``  in-sample residual energy of the model-band fit;
  - ``lomb``      max Lomb--Scargle power outside the model band;
  - ``heldout``   held-out prediction MSE (single random split);
  - ``crossfit``  multi-split cross-fit excess prediction error.
* **metrics**: ROC-AUC with 95% bootstrap CI, PR-AUC (average precision), TPR at
  test-H0-quantile FPR 1% and 5%, and FPR/TPR at the calibrated threshold.
* **sweeps**: tone amplitude, noise, N, ring width, off-grid distance, sampling family
  (jittered / i.i.d. / grid subset), and the ring-miss case (true frequency outside the
  ring).  The exactly grid-coherent condition is reported *separately* as the
  indistinguishability demonstration (T4a) -- by construction no detector can beat
  chance there, and that is the point, not a headline operating characteristic.

CPU-only.  Usage: python experiments/run_diagnostic_roc.py
"""
from __future__ import annotations

import sys

import numpy as np

from _util import save_json, savefig
from inralias.signals import nonuniform_times, evaluate, random_inband
from inralias.diagnostics import (
    extended_dictionary_test,
    residual_energy,
    crossfit_aliasing_energy,
)
from inralias.inr import FixedFeatureINR, lombscargle_periodogram

import matplotlib.pyplot as plt

LAMBDA = np.array([0.0, 5.0, -5.0, 11.0, -11.0, 17.0, -17.0, 23.0, -23.0])
Q = 128
NU_COH = 111.0
N_CAL = 300
N_TEST = 500
N_BOOT = 1000
DETECTORS = ("ring", "residual", "lomb", "heldout", "crossfit")


def _sample_times(rng, N, sampling):
    if sampling == "jitter":
        return nonuniform_times(N, rng, "jitter")
    if sampling == "iid":
        return np.sort(rng.uniform(0, 1, N))
    if sampling == "grid":
        return np.sort(rng.choice(Q, size=N, replace=False)) / Q
    raise ValueError(sampling)


def _draw(rng, N, sampling, sigma, amp, nu_mode, ring_hi, offgrid):
    """One draw; returns (t, y).  H0 when amp == 0."""
    t = _sample_times(rng, N, sampling)
    c = random_inband(LAMBDA, rng, power=1.0)
    y = evaluate(LAMBDA, c, t, real=True) + rng.normal(0, sigma, N)
    if amp > 0:
        if nu_mode == "coherent":
            nu = NU_COH
        elif nu_mode == "outside_ring":
            nu = float(rng.uniform(ring_hi * 2 + 2, ring_hi * 2 + 15))
        else:  # inside ring reach, off-grid by `offgrid`
            base = int(rng.integers(24, min(45, ring_hi - 1)))
            while base in np.abs(LAMBDA).astype(int):
                base = int(rng.integers(24, min(45, ring_hi - 1)))
            nu = base + offgrid
        ph = rng.uniform(0, 2 * np.pi)
        # real tone with RMS = amp x (unit in-band RMS)
        y = y + amp * np.sqrt(2) * np.cos(2 * np.pi * nu * t + ph)
    return t, y


def _scores(t, y, ring, sigma):
    """All detector scores for one draw (larger = more suspicious)."""
    out = {}
    d = extended_dictionary_test(LAMBDA, t, y, ring)
    out["ring"] = d["out_of_band_frac"]
    out["residual"] = residual_energy(LAMBDA, t, y)
    grid = np.arange(int(np.max(np.abs(LAMBDA))) + 1, int(np.max(ring)) + 1).astype(float)
    P = lombscargle_periodogram(t, y, grid)
    out["lomb"] = float(np.max(P)) if P.size else 0.0
    # held-out prediction error, single split
    n = t.size
    idx = np.random.default_rng(int(1e6 * t[0]) % (2**31)).permutation(n)
    a_idx, b_idx = idx[: n // 2], idx[n // 2:]
    m = FixedFeatureINR(LAMBDA, real=True).fit(t[a_idx], y[a_idx], ridge=1e-8)
    out["heldout"] = float(np.mean((m.predict(t[b_idx]) - y[b_idx]) ** 2))
    cf = crossfit_aliasing_energy(LAMBDA, t, y, sigma2=sigma**2, n_splits=8, seed=0)
    out["crossfit"] = cf["aliasing_energy"]
    return out


def _auc(h0, h1):
    return float((h1[:, None] > h0[None, :]).mean() + 0.5 * (h1[:, None] == h0[None, :]).mean())


def _auc_ci(h0, h1, rng, n_boot=N_BOOT):
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i0 = rng.integers(0, h0.size, h0.size)
        i1 = rng.integers(0, h1.size, h1.size)
        vals[b] = _auc(h0[i0], h1[i1])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _average_precision(h0, h1):
    scores = np.concatenate([h0, h1])
    labels = np.concatenate([np.zeros(h0.size), np.ones(h1.size)])
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    prec = tp / np.arange(1, labels.size + 1)
    return float(np.sum(prec * labels) / max(1, int(labels.sum())))


def run_condition(label, N=150, sampling="jitter", sigma=0.03, amp=0.5,
                  nu_mode="ring", ring_hi=30, offgrid=0.5, seed_base=0):
    import zlib

    rng = np.random.default_rng(zlib.crc32(label.encode()))
    ring = np.setdiff1d(
        np.concatenate([np.arange(-ring_hi, ring_hi + 1).astype(float)]), LAMBDA
    )
    n_atoms = LAMBDA.size + ring.size
    underdetermined = N <= n_atoms

    def batch(n, with_tone):
        rows = {k: [] for k in DETECTORS}
        for _ in range(n):
            t, y = _draw(rng, N, sampling, sigma, amp if with_tone else 0.0,
                         nu_mode, ring_hi, offgrid)
            s = _scores(t, y, ring, sigma)
            for k in DETECTORS:
                rows[k].append(s[k])
        return {k: np.asarray(v) for k, v in rows.items()}

    cal = batch(N_CAL, with_tone=False)          # calibration H0 only
    test0 = batch(N_TEST, with_tone=False)       # test H0
    test1 = batch(N_TEST, with_tone=True)        # test H1

    out = {"label": label, "N": N, "sampling": sampling, "sigma": sigma, "amp": amp,
           "nu_mode": nu_mode, "ring_hi": ring_hi, "offgrid": offgrid,
           "underdetermined": bool(underdetermined),
           "n_cal": N_CAL, "n_test": N_TEST, "detectors": {}}
    boot_rng = np.random.default_rng(zlib.crc32((label + "boot").encode()))
    for k in DETECTORS:
        thr = float(np.quantile(cal[k], 0.95))            # calibrated on H0 cal set only
        h0, h1 = test0[k], test1[k]
        lo, hi = _auc_ci(h0, h1, boot_rng)
        out["detectors"][k] = {
            "auc": _auc(h0, h1), "auc_ci95": [lo, hi],
            "pr_auc": _average_precision(h0, h1),
            "tpr_at_fpr01": float((h1 >= np.quantile(h0, 0.99)).mean()),
            "tpr_at_fpr05": float((h1 >= np.quantile(h0, 0.95)).mean()),
            "cal_threshold": thr,
            "test_fpr_at_cal": float((h0 >= thr).mean()),
            "test_tpr_at_cal": float((h1 >= thr).mean()),
        }
    return out


def make_figure(conditions):
    """Detection-power figure from computed conditions (also --figure-only from JSON)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    power_amps = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
    styles = {"ring": ("C0", "o-"), "residual": ("C1", "s--"), "lomb": ("C2", "^-."),
              "heldout": ("C4", "v:"), "crossfit": ("C5", "d-")}
    by_label = {c["label"]: c for c in conditions}
    for k in DETECTORS:
        ys = [by_label[f"power_amp{a}"]["detectors"][k]["tpr_at_fpr05"]
              for a in power_amps]
        col, fmt = styles[k]
        ax.plot(power_amps, ys, fmt, color=col, ms=4, lw=1.4, label=k)
    from inralias.identifiability import residual_test_power
    from inralias.inr import real_design
    import zlib

    prng = np.random.default_rng(zlib.crc32(b"theory-power"))
    theo = []
    for a in power_amps:
        vals = []
        for _ in range(60):
            t = _sample_times(prng, 150, "jitter")
            D, _ = real_design(LAMBDA, t)
            nu = int(prng.integers(24, 29)) + 0.5
            s = a * np.sqrt(2) * np.cos(2 * np.pi * nu * t + prng.uniform(0, 2 * np.pi))
            vals.append(residual_test_power(D, s, sigma=0.1, alpha=0.05)["power"])
        theo.append(float(np.mean(vals)))
    ax.plot(power_amps, theo, "k--", lw=1.4, label="T4b residual-test power (theory)")
    coh = by_label["coherent_demo"]["detectors"]["ring"]["tpr_at_fpr05"]
    ax.axhline(0.05, color="0.75", lw=0.8, ls=":")
    ax.plot([power_amps[0], power_amps[-1]], [coh, coh], color="k", lw=1.0, alpha=0.5,
            ls="-", label=f"coherent fold, any amp (TPR {coh:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(power_amps)
    ax.set_xticklabels([str(a) for a in power_amps])
    ax.minorticks_off()
    ax.set_xlabel("out-of-band tone amplitude (x in-band RMS), $\\sigma{=}0.1$")
    ax.set_ylabel("TPR at 5% FPR")
    ax.set_title("detection power: theory vs sample-only detectors")
    ax.legend(fontsize=8.5)
    savefig(fig, "diagnostic_roc.png")


def main():
    if "--figure-only" in sys.argv:
        import json
        from _util import RESULTS

        conditions = json.loads((RESULTS / "diagnostic_roc.json").read_text())["conditions"]
        make_figure(conditions)
        print("[roc] figure regenerated from JSON", flush=True)
        return

    conditions = []
    base = dict(N=150, sampling="jitter", sigma=0.03, amp=0.5, nu_mode="ring",
                ring_hi=30, offgrid=0.5)

    def add(label, **kw):
        cfg = dict(base); cfg.update(kw)
        print(f"[roc] {label} ...", flush=True)
        conditions.append(run_condition(label, **cfg))

    add("base")
    for a in (0.125, 0.25, 1.0):
        add(f"amp{a}", amp=a)
    # power sweep at low SNR: the regime where detectors actually differ
    for a in (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4):
        add(f"power_amp{a}", amp=a, sigma=0.1)
    for s in (0.01, 0.1):
        add(f"sigma{s}", sigma=s)
    for n in (100, 300):
        add(f"N{n}", N=n)
    for rh in (45, 60):
        add(f"ring{rh}", ring_hi=rh)
    for og in (0.0, 0.25):
        add(f"offgrid{og}", offgrid=og)
    add("iid", sampling="iid")
    add("gridsubset", sampling="grid", N=96)
    add("outside_ring", nu_mode="outside_ring")
    # indistinguishability demonstration (T4a) -- separate, NOT a headline ROC
    add("coherent_demo", sampling="grid", N=96, nu_mode="coherent", amp=1.0)

    make_figure(conditions)

    out = {"Lambda": LAMBDA.tolist(), "Q": Q, "nu_coherent": NU_COH,
           "protocol": {"n_cal": N_CAL, "n_test": N_TEST, "n_boot": N_BOOT,
                        "threshold_rule": "95th percentile of calibration H0 scores",
                        "separation": "calibration, test-H0, test-H1 all disjoint draws"},
           "conditions": conditions,
           "note": "coherent_demo is the T4a indistinguishability demonstration, not an "
                   "operating characteristic; underdetermined ring fits are flagged"}
    save_json("diagnostic_roc.json", out)
    for c in conditions:
        r = c["detectors"]["ring"]
        print(f"[roc] {c['label']}: ring AUC={r['auc']:.3f} CI[{r['auc_ci95'][0]:.3f},"
              f"{r['auc_ci95'][1]:.3f}] TPR@5%={r['tpr_at_fpr05']:.3f} "
              f"calFPR={r['test_fpr_at_cal']:.3f}", flush=True)


if __name__ == "__main__":
    main()
