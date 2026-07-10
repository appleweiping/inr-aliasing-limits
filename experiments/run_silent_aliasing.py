r"""E2 -- silent aliasing onto a STRUCTURED representable set.

The headline phenomenon of the converse (Theorem 2): a fixed-feature INR fits the noisy samples
well (small residual) yet its recovered spectrum is corrupted by an out-of-band tone folded onto
an in-band atom. We show the genuinely non-classical case -- a *structured* (non-interval)
representable set ``Lambda`` and *nonuniform* samples -- where the fold target is NOT the
classical ``nu mod f_s`` and the sample residual is small but nonzero. The classical
uniform-lowpass fold is included as a sanity check (Remark 1 in the paper).
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import nonuniform_times, evaluate
from inralias.inr import FixedFeatureINR
from inralias.limits import folded_frequency
from inralias.diagnostics import extended_dictionary_test, residual_energy

import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
# structured (non-interval) representable set: an INR whose feature frequencies are irregular
LAMBDA = np.array([0.0, 5.0, 11.0, 17.0, 23.0])
LAMBDA = np.concatenate([-LAMBDA[::-1][:-1], LAMBDA])  # hermitian-symmetric: {+-5,+-11,+-17,+-23,0}
SIGMA = 0.03


def _hermitian(freqs, amps_phases):
    f = np.array([w for w in freqs])
    a = np.array([m * np.exp(1j * p) for (m, p) in amps_phases])
    return np.concatenate([f, -f]), np.concatenate([a, np.conj(a)])


def main():
    # in-band tones on two structured atoms (11, 23); out-of-band tone at nu=34 (not in Lambda,
    # not a classical alias of any atom for these nonuniform samples)
    in_f, in_c = _hermitian([11.0, 23.0], [(1.0, 0.4), (0.6, 1.1)])
    nu = 34.0
    out_f, out_c = _hermitian([nu], [(0.8, 0.2)])
    allf = np.concatenate([in_f, out_f])
    allc = np.concatenate([in_c, out_c])

    # nonuniform (jittered) samples, N just above |Lambda| so the out-of-band tone is absorbed
    # into the fit (small residual) while corrupting the recovered coefficients (silent aliasing)
    N = LAMBDA.size + 2
    t = nonuniform_times(N, RNG, "jitter")
    y = evaluate(allf, allc, t, real=True) + RNG.normal(0, SIGMA, N)

    inr = FixedFeatureINR(LAMBDA, real=True).fit(t, y, ridge=1e-8)
    c_hat = inr.coeffs_
    c_star = np.zeros(LAMBDA.size, complex)
    for w, a in zip(in_f, in_c):
        c_star[int(np.argmin(np.abs(LAMBDA - w)))] += a

    fold = folded_frequency(nu, LAMBDA, t)
    resid = float(np.sqrt(residual_energy(LAMBDA, t, y)))
    spectral_err = float(np.linalg.norm(c_hat - c_star))
    ring = np.setdiff1d(np.arange(-30, 31).astype(float), LAMBDA)
    diag = extended_dictionary_test(LAMBDA, t, y, ring)

    t_dense = np.linspace(0, 1, 2000, endpoint=False)
    f_true = evaluate(allf, allc, t_dense, real=True)
    f_hat = inr.predict(t_dense)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax1.plot(t_dense, f_true, color="0.6", lw=1.2, label="true signal")
    ax1.plot(t_dense, f_hat, color="C3", lw=1.0, ls="--", label="INR fit")
    ax1.plot(t, y, "k.", ms=4, label="samples")
    ax1.set_xlabel("t"); ax1.set_ylabel("amplitude")
    ax1.set_title(f"time domain (sample RMSE={resid:.3f})"); ax1.legend(fontsize=7)

    order = np.argsort(LAMBDA)
    ax2.stem(LAMBDA[order], np.abs(c_star)[order], linefmt="0.7", markerfmt="o",
             basefmt=" ", label="true (structured $\\Lambda$)")
    ax2.stem(LAMBDA[order], np.abs(c_hat)[order], linefmt="C3-", markerfmt="C3x",
             basefmt=" ", label="INR recovered")
    ax2.axvline(fold["fold_freq"], color="C0", ls=":", lw=1.5)
    ax2.annotate(f"$\\nu={nu:.0f}$ folds to {fold['fold_freq']:.0f}",
                 xy=(fold["fold_freq"], np.abs(c_hat)[int(np.argmin(np.abs(LAMBDA-fold['fold_freq'])))]),
                 xytext=(fold["fold_freq"] - 10, 0.55), fontsize=7, color="C0",
                 arrowprops=dict(arrowstyle="->", color="C0"))
    ax2.set_xlabel("frequency ($\\Lambda$ is non-uniform)"); ax2.set_ylabel("|coefficient|")
    ax2.set_title(f"spectrum (spectral err={spectral_err:.2f})"); ax2.legend(fontsize=7)
    savefig(fig, "silent_aliasing.png")

    out = {"Lambda": LAMBDA.tolist(), "N": N, "nu": nu, "fold_freq": fold["fold_freq"],
           "sample_rmse": resid, "spectral_error": spectral_err,
           "diagnostic_out_of_band_frac": diag["out_of_band_frac"], "diagnostic_flag": diag["flag"],
           "note": "structured Lambda + nonuniform samples: small sample residual, large spectral "
                   "error, non-classical fold target"}
    save_json("silent_aliasing.json", out)
    print(f"[E2] structured/nonuniform: sample RMSE={resid:.3f} spectral err={spectral_err:.2f} "
          f"nu={nu:.0f}->fold {fold['fold_freq']:.0f} diag_flag={diag['flag']}", flush=True)


if __name__ == "__main__":
    main()
