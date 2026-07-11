r"""E2 -- silent aliasing onto a STRUCTURED representable set (exact grid fold).

The headline phenomenon of the converse (Theorem 2(ii)): when the sample locations lie on a
rate-``Q`` grid and an out-of-band tone satisfies ``nu = omega_k (mod Q)``, the tone's sample
vector coincides EXACTLY with an in-band atom's.  The INR then fits the samples down to the
noise floor (zero structural residual) while its recovered spectrum carries the full
out-of-band energy on the fold atom, and the L2 reconstruction error is bounded below by
``||f_out||`` -- silent aliasing that survives ``N >> m`` and is provably undetectable from
the samples alone.

The design is deliberately nonuniform: the ``N`` samples are a random SUBSET of the ``Q``
grid points, so the fold target is set by the underlying grid, not by the sample count or
the average density -- with a structured ``Lambda`` the energy lands on a structured atom
(here ``nu = 111 -> -17``), unlike the classical lowpass fold.  The classical uniform-lowpass
fold is the special case samples = full grid (Remark 1 in the paper).

Statistics: the fold is exact for EVERY grid subset (asserted over all seeds); sample RMSE /
spectral error / reconstruction RMSE are reported as mean +- s.d. over ``N_SEEDS`` independent
subset+noise draws, with seed ``FIG_SEED`` used for the illustrative figure.
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import evaluate
from inralias.inr import FixedFeatureINR
from inralias.limits import folded_frequency
from inralias.diagnostics import extended_dictionary_test, residual_energy

import matplotlib.pyplot as plt

# structured (non-interval) representable set: an INR whose feature frequencies are irregular
LAMBDA = np.array([0.0, 5.0, 11.0, 17.0, 23.0])
LAMBDA = np.concatenate([-LAMBDA[::-1][:-1], LAMBDA])  # hermitian-symmetric: {+-5,+-11,+-17,+-23,0}
SIGMA = 0.03
Q = 128            # underlying sampling grid rate
N = 96             # nonuniform: a random N-subset of the Q grid points, N >> m = |Lambda| = 9
NU = 111.0         # out-of-band tone: 111 = -17 (mod 128) -> exact fold onto the atom -17
AMP, PHASE = 0.8, 0.2
N_SEEDS = 20
FIG_SEED = 7


def _hermitian(freqs, amps_phases):
    f = np.array([w for w in freqs])
    a = np.array([m * np.exp(1j * p) for (m, p) in amps_phases])
    return np.concatenate([f, -f]), np.concatenate([a, np.conj(a)])


def _one_draw(seed: int):
    rng = np.random.default_rng(seed)
    in_f, in_c = _hermitian([11.0, 23.0], [(1.0, 0.4), (0.6, 1.1)])
    out_f, out_c = _hermitian([NU], [(AMP, PHASE)])
    allf = np.concatenate([in_f, out_f])
    allc = np.concatenate([in_c, out_c])

    t = np.sort(rng.choice(Q, size=N, replace=False)) / Q
    y = evaluate(allf, allc, t, real=True) + rng.normal(0, SIGMA, N)

    inr = FixedFeatureINR(LAMBDA, real=True).fit(t, y, ridge=1e-8)
    c_hat = inr.coeffs_
    c_star = np.zeros(LAMBDA.size, complex)
    for w, a in zip(in_f, in_c):
        c_star[int(np.argmin(np.abs(LAMBDA - w)))] += a

    fold = folded_frequency(NU, LAMBDA, t)
    resid = float(np.sqrt(residual_energy(LAMBDA, t, y)))
    spectral_err = float(np.linalg.norm(c_hat - c_star))

    t_dense = np.linspace(0, 1, 4096, endpoint=False)
    f_true = evaluate(allf, allc, t_dense, real=True)
    f_hat = inr.predict(t_dense)
    recon_rmse = float(np.sqrt(np.mean((f_hat - f_true) ** 2)))

    return {
        "t": t, "y": y, "c_hat": c_hat, "c_star": c_star, "fold": fold,
        "sample_rmse": resid, "spectral_err": spectral_err, "recon_rmse": recon_rmse,
        "t_dense": t_dense, "f_true": f_true, "f_hat": f_hat,
    }


def main():
    draws = [_one_draw(s) for s in range(N_SEEDS)]
    # the grid fold is exact: the target must be -17 for every subset draw
    assert all(d["fold"]["fold_freq"] == -17.0 for d in draws)
    s_rmse = np.array([d["sample_rmse"] for d in draws])
    s_spec = np.array([d["spectral_err"] for d in draws])
    s_recon = np.array([d["recon_rmse"] for d in draws])

    d = _one_draw(FIG_SEED)
    resid, spectral_err = d["sample_rmse"], d["spectral_err"]
    fold, c_hat, c_star = d["fold"], d["c_hat"], d["c_star"]

    # ground-truth-free ring diagnostic on the figure draw: the exact grid fold is provably
    # invisible (e_111 == e_-17 on every grid point), so the ring must NOT fire -- the
    # diagnostic's honest worst case.  N=96 > |Lambda|+|ring| = 61, so the test is valid
    # (overdetermined); see run_diagnostic_roc.py for its operating characteristic on
    # generic (off-grid) sampling, where it does detect folds.
    ring = np.setdiff1d(np.arange(-30, 31).astype(float), LAMBDA)
    diag = extended_dictionary_test(LAMBDA, d["t"], d["y"], ring)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax1.plot(d["t_dense"], d["f_true"], color="0.6", lw=1.2, label="true signal")
    ax1.plot(d["t_dense"], d["f_hat"], color="C3", lw=1.0, ls="--", label="INR fit")
    ax1.plot(d["t"], d["y"], "k.", ms=4, label="samples")
    ax1.set_xlabel("t"); ax1.set_ylabel("amplitude")
    ax1.set_title(f"time domain (sample RMSE={resid:.3f}, recon RMSE={d['recon_rmse']:.2f})")
    ax1.legend(fontsize=7)

    order = np.argsort(LAMBDA)
    ax2.stem(LAMBDA[order], np.abs(c_star)[order], linefmt="0.7", markerfmt="o",
             basefmt=" ", label="true (structured $\\Lambda$)")
    ax2.stem(LAMBDA[order], np.abs(c_hat)[order], linefmt="C3-", markerfmt="C3x",
             basefmt=" ", label="INR recovered")
    ax2.axvline(fold["fold_freq"], color="C0", ls=":", lw=1.5)
    ax2.annotate(f"$\\nu={NU:.0f}$ folds\nexactly to ${fold['fold_freq']:.0f}$",
                 xy=(fold["fold_freq"], np.abs(c_hat)[int(np.argmin(np.abs(LAMBDA - fold['fold_freq'])))]),
                 xytext=(fold["fold_freq"] - 9.5, 0.72), fontsize=11, color="k",
                 arrowprops=dict(arrowstyle="->", color="k"))
    ax2.set_xlabel("frequency ($\\Lambda$ is non-uniform)"); ax2.set_ylabel("|coefficient|")
    ax2.set_title(f"spectrum (spectral err={spectral_err:.2f})"); ax2.legend(fontsize=7)
    savefig(fig, "silent_aliasing.png")

    out = {"Lambda": LAMBDA.tolist(), "m": int(LAMBDA.size), "Q": Q, "N": N, "nu": NU,
           "sigma": SIGMA, "out_amp": AMP, "fold_freq": fold["fold_freq"],
           "n_seeds": N_SEEDS,
           "sample_rmse_mean": float(s_rmse.mean()), "sample_rmse_std": float(s_rmse.std()),
           "spectral_error_mean": float(s_spec.mean()), "spectral_error_std": float(s_spec.std()),
           "recon_rmse_mean": float(s_recon.mean()), "recon_rmse_std": float(s_recon.std()),
           "fig_seed": FIG_SEED, "fig_sample_rmse": resid, "fig_spectral_error": spectral_err,
           "fig_recon_rmse": d["recon_rmse"],
           "diagnostic_out_of_band_frac": diag["out_of_band_frac"],
           "diagnostic_flag": diag["flag"],
           "diagnostic_underdetermined": diag["underdetermined"],
           "note": "exact grid fold: samples on an N-of-Q grid subset (nonuniform), "
                   "nu = -17 (mod Q) -> zero structural residual, full-energy fold onto the "
                   "structured atom -17, recon RMSE >= ||f_out||; provably invisible to any "
                   "sample-based diagnostic (ring correctly does not fire)"}
    save_json("silent_aliasing.json", out)
    print(f"[E2] exact grid fold: sample RMSE={s_rmse.mean():.3f}+-{s_rmse.std():.3f} "
          f"spectral err={s_spec.mean():.3f}+-{s_spec.std():.3f} "
          f"recon RMSE={s_recon.mean():.3f}+-{s_recon.std():.3f} "
          f"nu={NU:.0f}->fold {fold['fold_freq']:.0f} (exact, all {N_SEEDS} seeds) "
          f"diag_flag={diag['flag']} (expected False: fold is provably invisible)", flush=True)


if __name__ == "__main__":
    main()
