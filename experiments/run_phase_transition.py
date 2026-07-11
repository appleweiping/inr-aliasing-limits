r"""E1 -- the aliasing phase transition.

Two views of the limit, from the fixed-feature (exact-theory) INR:

* **Fig A (achievability vs converse).**  Fix the INR representable band ``B_INR`` and sweep
  the sampling density ``rho = N / (2 B_INR)``.  For an in-band signal (bandwidth ratio
  ``r = B_signal/B_INR = 1``) the reconstruction error falls to the noise floor once
  ``rho >= 1`` (Theorem 1: the samples form a frame, error ~ sigma * kappa).  For
  out-of-band signals (``r > 1``) the error *plateaus* at a positive **aliasing floor** that
  no sampling density removes (Theorem 2).  The theory noise-floor and aliasing-floor curves
  are overlaid.

* **Fig B (phase diagram).**  Recovered spectral error over the ``(r, rho)`` plane, with the
  predicted boundaries ``r = 1`` (signal leaves the representable band) and ``rho = 1``
  (samples drop below the stable rate).
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import lowpass_dictionary, nonuniform_times, evaluate
from inralias.inr import FixedFeatureINR
from inralias.sampling import synthesis_matrix, noise_gain
from inralias.limits import mmse_aliasing_floor

import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260711)
B_INR = 8                      # INR representable band: Lambda = {-8..8}
LAMBDA = lowpass_dictionary(B_INR)
SIGMA = 0.05                   # measurement noise std
N_DENSE = 4000                 # dense grid for reconstruction error
N_TRIALS = 40                  # Monte-Carlo signal/noise draws per grid point


def _make_signal(ratio: float, rng):
    """In-band coeffs on LAMBDA + out-of-band atoms filling (B_INR, ratio*B_INR].

    Total signal power normalised to 1; in/out split by band. Returns (freqs, coeffs).
    """
    # in-band: integer atoms in [-B_INR, B_INR]
    in_f = LAMBDA.copy()
    a = (rng.standard_normal(in_f.size) + 1j * rng.standard_normal(in_f.size)) / np.sqrt(2)
    # out-of-band: integer atoms in (B_INR, floor(ratio*B_INR)]
    hi = int(np.floor(ratio * B_INR))
    if hi > B_INR:
        pos = np.arange(B_INR + 1, hi + 1).astype(float)
        out_f = np.concatenate([pos, -pos])
        b = (rng.standard_normal(pos.size) + 1j * rng.standard_normal(pos.size)) / np.sqrt(2)
        out_c = np.concatenate([b, np.conj(b)])
    else:
        out_f = np.zeros(0)
        out_c = np.zeros(0, complex)
    # hermitian-symmetrise in-band for a real signal
    order = np.argsort(in_f)
    for i in order:
        w = in_f[i]
        if w > 0:
            j = int(np.argmin(np.abs(in_f + w)))
            a[j] = np.conj(a[i])
    a[np.isclose(in_f, 0)] = a[np.isclose(in_f, 0)].real
    freqs = np.concatenate([in_f, out_f])
    coeffs = np.concatenate([a, out_c])
    coeffs = coeffs / np.linalg.norm(coeffs)
    n = in_f.size
    return in_f, out_f, coeffs[:n], coeffs[n:]


def _recon_error(ratio: float, N: int, rng) -> tuple[float, float]:
    """Mean and s.d. of the fixed-feature INR reconstruction RMSE over N_TRIALS draws."""
    t_dense = np.linspace(0, 1, N_DENSE, endpoint=False)
    errs = []
    for _ in range(N_TRIALS):
        in_f, out_f, a, out_c = _make_signal(ratio, rng)
        allf = np.concatenate([in_f, out_f])
        coeffs = np.concatenate([a, out_c])
        t = nonuniform_times(N, rng, "jitter")
        y = evaluate(allf, coeffs, t, real=True) + rng.normal(0, SIGMA, N)
        m = FixedFeatureINR(LAMBDA, real=True).fit(t, y, ridge=1e-8)
        f_hat = m.predict(t_dense)
        f_true = evaluate(allf, coeffs, t_dense, real=True)
        errs.append(np.sqrt(np.mean((f_hat - f_true) ** 2)))
    errs = np.array(errs)
    return float(errs.mean()), float(errs.std())


def _theory_noise_floor(N: int, rng, n_draws: int = 10) -> float:
    """Exact Thm-1 dense-grid noise-floor RMSE for an in-band signal:
    sigma * ||Re(Phi_eval M)||_F / sqrt(N_dense), averaged over jitter draws."""
    t_dense = np.linspace(0, 1, N_DENSE, endpoint=False)
    Phi_eval = synthesis_matrix(LAMBDA, t_dense)
    vals = []
    for _ in range(n_draws):
        t = nonuniform_times(N, rng, "jitter")
        Phi = synthesis_matrix(LAMBDA, t)
        G = Phi.conj().T @ Phi + 1e-8 * np.eye(LAMBDA.size)
        M = np.linalg.solve(G, Phi.conj().T)
        A = np.real(Phi_eval @ M)
        vals.append(SIGMA * np.linalg.norm(A) / np.sqrt(N_DENSE))
    return float(np.mean(vals))


def _plateau_rmse(ratio: float) -> float:
    """Expected aliasing plateau ||f_out||: with unit total power and iid coefficient draws
    the mean out-of-band energy fraction is n_out / (n_in + n_out)."""
    hi = int(np.floor(ratio * B_INR))
    n_out = 2 * max(0, hi - B_INR)
    n_in = LAMBDA.size
    return float(np.sqrt(n_out / (n_in + n_out))) if n_out else 0.0


def figure_A():
    densities = np.linspace(0.5, 3.0, 22)
    ratios = [1.0, 1.25, 1.5]
    curves, stds = {}, {}
    for r in ratios:
        row, srow = [], []
        for rho in densities:
            # honest density: N really varies with rho, including the sub-critical N < m
            # regime (the ridge pseudo-inverse gives the minimum-norm interpolant there)
            N = max(2, int(round(rho * 2 * B_INR)))
            mu, sd = _recon_error(r, N, RNG)
            row.append(mu); srow.append(sd)
        curves[f"r={r}"] = row
        stds[f"r={r}"] = srow

    # exact theory overlays: Thm-1 noise floor for r=1, ||f_out|| plateaus for r>1
    noise_floor_curve = [
        _theory_noise_floor(max(2, int(round(rho * 2 * B_INR))), RNG) for rho in densities
    ]
    plateaus = {f"r={r}": _plateau_rmse(r) for r in ratios if r > 1.0}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    se = 1.0 / np.sqrt(N_TRIALS)
    for r in ratios:
        mu = np.array(curves[f"r={r}"]); sd = np.array(stds[f"r={r}"])
        (line,) = ax.plot(densities, mu, marker="o", ms=3, label=f"$B_{{sig}}/B_{{INR}}={r}$")
        ax.fill_between(densities, mu - se * sd, mu + se * sd, color=line.get_color(), alpha=0.2)
    ax.plot(densities, noise_floor_curve, color="0.4", ls="--", lw=1.0,
            label="Thm 1 noise floor ($r{=}1$)")
    for key, val in plateaus.items():
        ax.axhline(val, color="0.4", ls=":", lw=0.9)
    ax.axvline(1.0, color="k", ls=":", lw=1, label="stable rate $\\rho=1$")
    ax.set_xlabel("sampling density $\\rho = N/(2 B_{INR})$")
    ax.set_ylabel("reconstruction RMSE")
    ax.set_yscale("log")
    ax.set_title("Achievability ($r{=}1$) vs aliasing floor ($r{>}1$)")
    ax.legend(fontsize=6.5)
    savefig(fig, "phase_transition_A.png")
    return {"densities": densities.tolist(), "curves": curves, "stds": stds,
            "n_trials": N_TRIALS,
            "theory_noise_floor": noise_floor_curve, "theory_plateaus": plateaus}


def figure_B():
    ratios = np.linspace(0.6, 2.0, 22)
    densities = np.linspace(0.6, 2.4, 22)
    Z = np.zeros((len(ratios), len(densities)))
    for i, r in enumerate(ratios):
        for j, rho in enumerate(densities):
            N = max(2, int(round(rho * 2 * B_INR)))
            Z[i, j] = _recon_error(r, N, RNG)[0]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.pcolormesh(densities, ratios, np.log10(Z + 1e-6), shading="auto", cmap="magma")
    ax.axhline(1.0, color="cyan", ls="--", lw=1.2, label="$r=1$ (leaves band)")
    ax.axvline(1.0, color="w", ls=":", lw=1.2, label="$\\rho=1$ (stable rate)")
    ax.set_xlabel("sampling density $\\rho$")
    ax.set_ylabel("bandwidth ratio $r=B_{sig}/B_{INR}$")
    ax.set_title("Phase diagram: $\\log_{10}$ recon RMSE")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.legend(fontsize=7, loc="upper right")
    savefig(fig, "phase_transition_B.png")
    return {"ratios": ratios.tolist(), "densities": densities.tolist(), "logZ": np.log10(Z + 1e-6).tolist()}


def main():
    print("[E1] Fig A: achievability vs converse ...", flush=True)
    A = figure_A()
    print("[E1] Fig B: phase diagram ...", flush=True)
    B = figure_B()
    out = {
        "B_INR": B_INR, "sigma": SIGMA, "n_trials": N_TRIALS,
        "figA": A, "figB": B,
        "description": "E1 aliasing phase transition: sampling-density sweep + (r,rho) phase diagram",
    }
    p = save_json("phase_transition.json", out)
    print(f"[E1] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
