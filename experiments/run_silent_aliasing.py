r"""E2 -- silent aliasing.

The headline phenomenon of the converse (Theorem 2): an INR fits the noisy samples well
(small sample-domain residual) yet its recovered **spectrum** is corrupted by an out-of-band
tone folded onto an in-band atom.  A practitioner watching only the global error would not
notice.  We plant a sparse in-band spectrum plus one out-of-band tone, fit the fixed-feature
INR, and show:

* time domain -- INR reconstruction matches the signal at the samples;
* spectrum -- a spurious peak appears at the frequency predicted by :func:`folded_frequency`,
  absent from the true in-band spectrum;
* the ground-truth-free diagnostic flags the out-of-band content.
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import lowpass_dictionary, nonuniform_times, evaluate
from inralias.inr import FixedFeatureINR
from inralias.limits import folded_frequency
from inralias.diagnostics import extended_dictionary_test, residual_energy

import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
B_INR = 10
LAMBDA = lowpass_dictionary(B_INR)
SIGMA = 0.03


def main():
    # sparse in-band spectrum: tones at +/-3 and +/-6
    in_tones = np.array([3.0, 6.0])
    in_f = np.concatenate([in_tones, -in_tones])
    ia = np.array([1.0, 0.6]) * np.exp(1j * np.array([0.4, 1.1]))
    in_c = np.concatenate([ia, np.conj(ia)])

    # one out-of-band tone that folds EXACTLY onto an in-band atom under rate-N uniform
    # sampling: this is the worst case where the fold is coherent, so the samples are fit to
    # the noise floor (silent) while the in-band spectrum is corrupted by a spurious peak.
    N = 2 * B_INR + 3  # 23 uniform samples; atoms {-10..10} stay orthogonal (distinct mod N)
    k0 = 4             # in-band atom that carries NO true signal energy
    omega_out = float(N + k0)  # aliases to atom k0 (and -k0)
    out_f = np.array([omega_out, -omega_out])
    oa = np.array([0.7 * np.exp(1j * 0.2)])
    out_c = np.concatenate([oa, np.conj(oa)])

    allf = np.concatenate([in_f, out_f])
    allc = np.concatenate([in_c, out_c])

    t = nonuniform_times(N, RNG, "uniform")
    y = evaluate(allf, allc, t, real=True) + RNG.normal(0, SIGMA, N)

    inr = FixedFeatureINR(LAMBDA, real=True).fit(t, y, ridge=1e-8)
    c_hat = inr.coeffs_

    # true in-band coefficients placed on LAMBDA
    c_star = np.zeros(LAMBDA.size, complex)
    for w, a in zip(in_f, in_c):
        c_star[int(np.argmin(np.abs(LAMBDA - w)))] += a

    fold = folded_frequency(omega_out, LAMBDA, t)
    resid = float(np.sqrt(residual_energy(LAMBDA, t, y)))
    spectral_err = float(np.linalg.norm(c_hat - c_star))
    ring = np.concatenate([np.arange(B_INR + 1, B_INR + 10), -np.arange(B_INR + 1, B_INR + 10)]).astype(float)
    diag = extended_dictionary_test(LAMBDA, t, y, ring)

    # ---- figure ----
    t_dense = np.linspace(0, 1, 2000, endpoint=False)
    f_true = evaluate(allf, allc, t_dense, real=True)
    f_hat = inr.predict(t_dense)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax1.plot(t_dense, f_true, color="0.6", lw=1.2, label="true signal")
    ax1.plot(t_dense, f_hat, color="C3", lw=1.0, ls="--", label="INR fit")
    ax1.plot(t, y, "k.", ms=4, label="samples")
    ax1.set_xlabel("t"); ax1.set_ylabel("amplitude")
    ax1.set_title(f"time domain (sample RMSE={resid:.3f})")
    ax1.legend(fontsize=7)

    order = np.argsort(LAMBDA)
    ax2.stem(LAMBDA[order], np.abs(c_star)[order], linefmt="0.7", markerfmt="o",
             basefmt=" ", label="true in-band")
    ax2.stem(LAMBDA[order], np.abs(c_hat)[order], linefmt="C3-", markerfmt="C3x",
             basefmt=" ", label="INR recovered")
    ax2.axvline(fold["fold_freq"], color="C0", ls=":", lw=1.5)
    ax2.annotate(f"folded from $\\omega_{{out}}={omega_out:.0f}$",
                 xy=(fold["fold_freq"], np.abs(c_hat)[int(np.argmin(np.abs(LAMBDA-fold['fold_freq'])))]),
                 xytext=(fold["fold_freq"] + 1, 0.5), fontsize=7, color="C0",
                 arrowprops=dict(arrowstyle="->", color="C0"))
    ax2.set_xlabel("frequency"); ax2.set_ylabel("|coefficient|")
    ax2.set_title(f"spectrum (spectral err={spectral_err:.2f})")
    ax2.legend(fontsize=7)
    savefig(fig, "silent_aliasing.png")

    out = {
        "N": N, "omega_out": omega_out, "fold_freq": fold["fold_freq"],
        "sample_rmse": resid, "spectral_error": spectral_err,
        "diagnostic_out_of_band_frac": diag["out_of_band_frac"], "diagnostic_flag": diag["flag"],
        "note": "small sample RMSE but large spectral error at the predicted fold frequency",
    }
    save_json("silent_aliasing.json", out)
    print(f"[E2] sample RMSE={resid:.3f}  spectral err={spectral_err:.2f}  "
          f"fold->{fold['fold_freq']:.0f}  diag_flag={diag['flag']}", flush=True)


if __name__ == "__main__":
    main()
