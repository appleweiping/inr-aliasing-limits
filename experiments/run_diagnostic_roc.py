r"""E7 -- operating characteristic (ROC) of the ground-truth-free ring diagnostic.

The ring diagnostic (Sec. 4 of the paper; :func:`inralias.diagnostics.extended_dictionary_test`)
claims to detect out-of-band content from the samples alone.  This script measures that claim
instead of asserting it: over ``K`` random draws per condition we compute the recovered
out-of-band energy fraction under

* **H0** (null): a random Hermitian in-band signal on the structured ``Lambda`` plus noise;
* **H1**: H0 plus one out-of-band tone of relative amplitude ``a`` at a frequency ``nu`` that
  is either an integer on the ring grid ("on-grid") or a uniform real frequency ("off-grid");
* **H1-coherent** (worst case): samples restricted to a rate-``Q`` grid with
  ``nu = -17 (mod Q)`` -- the exact fold of E2, which is *provably invisible* from the
  samples; the diagnostic must degrade to chance here, matching the converse.

and report, per condition, the ROC AUC, the TPR/FPR at the fixed 0.1 threshold used in the
package, and the null-calibrated threshold (H0 95th percentile).  Everything is
overdetermined (``N > |Lambda| + |ring|``) except an explicitly labelled underdetermined
demo condition documenting WHY the guard in ``extended_dictionary_test`` exists.

CPU-only, ~1 minute.  Usage: python experiments/run_diagnostic_roc.py
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import nonuniform_times, evaluate, random_inband
from inralias.diagnostics import extended_dictionary_test

import matplotlib.pyplot as plt

LAMBDA = np.array([0.0, 5.0, 11.0, 17.0, 23.0])
LAMBDA = np.concatenate([-LAMBDA[::-1][:-1], LAMBDA])          # {0,+-5,+-11,+-17,+-23}, m=9
RING = np.setdiff1d(np.arange(-30, 31).astype(float), LAMBDA)  # 52 ring atoms, ext size 61
SIGMA = 0.03
K = 500
Q = 128          # grid rate for the coherent (exact-fold) worst case
NU_COH = 111.0   # 111 = -17 (mod 128)


def _draw_frac(rng, N, amp, nu_mode, sampling):
    """One draw: return the diagnostic's out-of-band fraction (and validity flag)."""
    c = random_inband(LAMBDA, rng, power=1.0)
    freqs, coeffs = LAMBDA.copy(), np.asarray(c, complex)
    if nu_mode is not None:
        if nu_mode == "on":
            cand = np.setdiff1d(np.arange(24, 46).astype(float), LAMBDA)
            nu = float(rng.choice(cand))
        elif nu_mode == "off":
            nu = float(rng.uniform(24.0, 45.0))
        elif nu_mode == "coherent":
            nu = NU_COH
        phase = rng.uniform(0, 2 * np.pi)
        a = amp / np.sqrt(2)  # tone RMS = amp x in-band RMS (unit power)
        freqs = np.concatenate([freqs, [nu, -nu]])
        coeffs = np.concatenate([coeffs, [a * np.exp(1j * phase), a * np.exp(-1j * phase)]])
    if sampling == "grid":
        t = np.sort(rng.choice(Q, size=N, replace=False)) / Q
    else:
        t = nonuniform_times(N, rng, "jitter")
    y = evaluate(freqs, coeffs, t, real=True) + rng.normal(0, SIGMA, N)
    d = extended_dictionary_test(LAMBDA, t, y, RING)
    return d["out_of_band_frac"], d["underdetermined"]


def _roc(h0, h1, thresholds):
    tpr = [(h1 >= th).mean() for th in thresholds]
    fpr = [(h0 >= th).mean() for th in thresholds]
    # AUC by rank statistic (Mann-Whitney)
    auc = float((h1[:, None] > h0[None, :]).mean() + 0.5 * (h1[:, None] == h0[None, :]).mean())
    return np.array(tpr), np.array(fpr), auc


def main():
    thresholds = np.linspace(0, 1, 201)
    conditions = []
    roc_curves = {}
    # H0 per (N, sampling) -- shared across the H1 conditions with matching sampling
    plans = [
        # (label, N, sampling, list of (amp, nu_mode))
        ("N80_jitter", 80, "jitter", [(0.25, "on"), (0.5, "on"), (1.0, "on"),
                                      (0.25, "off"), (0.5, "off"), (1.0, "off")]),
        ("N150_jitter", 150, "jitter", [(0.5, "on"), (0.5, "off")]),
        ("N96_grid_coherent", 96, "grid", [(1.0, "coherent")]),
        # underdetermined demo: documents the guard, EXCLUDED from headline numbers
        ("N36_jitter_underdetermined", 36, "jitter", [(1.0, "on")]),
    ]
    import zlib

    for label, N, sampling, h1_specs in plans:
        rng = np.random.default_rng(zlib.crc32(label.encode()))  # deterministic per label
        h0 = np.array([_draw_frac(rng, N, 0.0, None, sampling)[0] for _ in range(K)])
        under = _draw_frac(rng, N, 0.0, None, sampling)[1]
        for amp, mode in h1_specs:
            h1 = np.array([_draw_frac(rng, N, amp, mode, sampling)[0] for _ in range(K)])
            tpr, fpr, auc = _roc(h0, h1, thresholds)
            i01 = int(np.argmin(np.abs(thresholds - 0.10)))
            cond = {
                "label": label, "N": N, "sampling": sampling, "amp": amp, "nu_mode": mode,
                "underdetermined": bool(under), "n_draws": K,
                "auc": auc,
                "tpr_at_0.1": float(tpr[i01]), "fpr_at_0.1": float(fpr[i01]),
                "h0_p95": float(np.quantile(h0, 0.95)),
                "tpr_at_h0_p95": float((h1 >= np.quantile(h0, 0.95)).mean()),
            }
            conditions.append(cond)
            roc_curves[f"{label}_a{amp}_{mode}"] = {"tpr": tpr.tolist(), "fpr": fpr.tolist()}
            print(f"[roc] {label} a={amp} nu={mode}: AUC={auc:.3f} "
                  f"TPR@0.1={cond['tpr_at_0.1']:.3f} FPR@0.1={cond['fpr_at_0.1']:.3f} "
                  f"underdet={under}", flush=True)

    # figure: ROC curves for the headline N=80 conditions + the coherent worst case
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for key, style in [("N80_jitter_a0.25_on", "-"), ("N80_jitter_a0.5_on", "-"),
                       ("N80_jitter_a1.0_on", "-"), ("N80_jitter_a0.5_off", "--"),
                       ("N96_grid_coherent_a1.0_coherent", ":")]:
        if key in roc_curves:
            c = roc_curves[key]
            ax.plot(c["fpr"], c["tpr"], style, lw=1.4, label=key.replace("_jitter", ""))
    ax.plot([0, 1], [0, 1], color="0.8", lw=0.8)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title(f"ring diagnostic ROC ({K} draws/condition)")
    ax.legend(fontsize=6)
    savefig(fig, "diagnostic_roc.png")

    out = {"Lambda": LAMBDA.tolist(), "ring_lo": -30, "ring_hi": 30, "sigma": SIGMA,
           "K": K, "Q": Q, "nu_coherent": NU_COH,
           "conditions": conditions,
           "note": "ring-diagnostic operating characteristic; the coherent grid-fold "
                   "condition is the provably-undetectable worst case (AUC ~ 0.5 expected); "
                   "the underdetermined condition documents the validity guard"}
    save_json("diagnostic_roc.json", out)
    print("[roc] wrote diagnostic_roc.json", flush=True)


if __name__ == "__main__":
    main()
