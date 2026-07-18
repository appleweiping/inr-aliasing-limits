r"""E1 -- the synthetic theorem-validation matrix (T1--T3).

One experiment, four panels, closing the theory--experiment loop:

(a) **Jitter breaks the exact fold (T3a).**  Grid-coherent tone (``nu = -17 + Q`` on a
    rate-``Q`` grid subset): visibility vs jitter std, empirical mean +- 95% CI over
    draws, against the characteristic-function law ``v = sqrt(1 - chi^2)``.

(b) **i.i.d. concentration (T3b).**  Worst-case aliasability over a finite out-of-band
    candidate set vs ``N`` under i.i.d. uniform sampling, with the proved
    Hoeffding/union bound and the ``N^{-1/2}`` reference slope.

(c) **Fixed vs adversarial tones (T1 vs T3).**  Visibility vs ``N`` for: a grid-coherent
    tone on grid subsets (exactly zero at *every* ``N`` -- the persistence statement),
    the same fixed tone under i.i.d. sampling (visible immediately), and the per-``N``
    adversarial (minimum-visibility) candidate under i.i.d. sampling (still becomes
    visible: random sampling defeats even the adversary restricted to a finite set).

(d) **Function-space error decomposition (T2).**  Total L2 reconstruction error of the
    noisy LS fit across the jitter sweep, decomposed into the modeled-component
    (aliasing-bias) error and the truncation (unrepresentable-energy) floor: jitter
    removes the aliasing bias, never the truncation error -- the two are distinct
    quantities and only the first is "aliasing".

All quantities: sample residual, coefficient bias, modeled-component function error,
total reconstruction error, truncation error, visibility, aliasability; recorded per
condition in the JSON together with conditioning and frequency-separation metadata.
CPU-only.  Usage: python experiments/run_synthetic_matrix.py
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import evaluate
from inralias.inr import FixedFeatureINR
from inralias.identifiability import (
    visibility,
    aliasability,
    expected_jitter_coherence,
    aliasability_concentration_bound,
    aliasability_matrix_bound,
    function_error_decomposition,
    continuous_gram,
)
from inralias.sampling import synthesis_matrix, condition_number

import matplotlib.pyplot as plt

LAMBDA = np.array([0.0, 5.0, -5.0, 11.0, -11.0, 17.0, -17.0, 23.0, -23.0])
M = LAMBDA.size
Q = 128
NU = 111.0          # -17 + Q: grid-coherent (k = 1)
K_FOLD = 1
N_GRID = 96
SIGMA = 0.03
AMP, PHASE = 0.8, 0.2
N_DRAWS = 20
CI = 1.96


def _tone(freqs_nu, amp, phase):
    return np.array([freqs_nu, -freqs_nu]), np.array(
        [amp / 2 * np.exp(1j * phase), amp / 2 * np.exp(-1j * phase)]
    )


def _mean_ci(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(CI * v.std(ddof=1) / np.sqrt(v.size))


def panel_a_jitter(rng):
    """Visibility of the grid-coherent tone vs Gaussian jitter std (T3a)."""
    sig_ts = np.concatenate([[0.0], np.geomspace(2e-5, 4e-3, 12)])
    emp_mean, emp_ci, pred = [], [], []
    for st in sig_ts:
        vs = []
        for _ in range(N_DRAWS):
            g = rng.choice(Q, size=N_GRID, replace=False)
            t = g / Q + (rng.normal(0, st, N_GRID) if st > 0 else 0.0)
            vs.append(visibility(LAMBDA, t, NU))
        mu, ci = _mean_ci(vs)
        emp_mean.append(mu); emp_ci.append(ci)
        chi = expected_jitter_coherence(K_FOLD, Q, st, "gaussian") if st > 0 else 1.0
        pred.append(float(np.sqrt(max(0.0, 1.0 - chi**2))))
    return {"sigma_t": sig_ts.tolist(), "v_mean": emp_mean, "v_ci": emp_ci, "v_pred": pred}


def panel_b_concentration(rng):
    """Worst-case aliasability over a candidate set vs N, iid uniform sampling (T3b).

    Compares the loose union-bound (``aliasability_concentration_bound``) with the
    matrix-Bernstein/Chernoff bound (Theorem 1'): both the a-priori bound (finite ~30x
    sooner) and the data-dependent bound (uses the realized lambda_min -> finite even at
    design-scale N).  Emits the crossover N of each for the paper macros.
    """
    from inralias.sampling import synthesis_matrix
    cands = np.setdiff1d(np.arange(24, 55).astype(float), LAMBDA)
    K = cands.size
    delta = 0.05
    # design-scale grid (40..) through the crossover into the asymptotic regime
    Ns = [40, 64, 96, 128, 256, 512, 1024, 3200, 6400, 12800, 25600]
    emp_mean, emp_ci, bound, mbound_ap, mbound_dd, lam_min_med = [], [], [], [], [], []
    for N in Ns:
        vals, lmins = [], []
        for _ in range(20):
            t = np.sort(rng.uniform(0, 1, N))
            vals.append(max(aliasability(LAMBDA, t, nu) for nu in cands))
            Phi = synthesis_matrix(LAMBDA, t)
            lmins.append(float(np.linalg.eigvalsh(Phi.conj().T @ Phi / N)[0]))
        mu, ci = _mean_ci(vals)
        lmed = float(np.median(lmins))
        emp_mean.append(mu); emp_ci.append(ci); lam_min_med.append(lmed)
        bound.append(aliasability_concentration_bound(M, K, N, delta=delta, lam_min=1.0))
        mb = aliasability_matrix_bound(M, K, N, delta=delta, data_lam_min=lmed)
        mbound_ap.append(mb["apriori"]); mbound_dd.append(mb.get("data_dependent", float("inf")))
    old_x = float(4 * M * M * np.log(4 * (M * K + M * M) / delta))     # union-bound crossover
    new_x = float(aliasability_matrix_bound(M, K, 100, delta=delta)["crossover_N"])
    return {"N": Ns, "K": int(K), "amax_mean": emp_mean, "amax_ci": emp_ci,
            "bound_delta05": bound, "matrix_bound_apriori": mbound_ap,
            "matrix_bound_data_dependent": mbound_dd, "lam_min_median": lam_min_med,
            "crossover_N_old": old_x, "crossover_N_new": new_x}


def panel_c_fixed_vs_adversarial(rng):
    """Visibility vs N: grid persistence (T1) vs iid killing fixed AND adversarial tones."""
    Q_big = 512
    nu_big = -17.0 + Q_big                     # grid-coherent on the rate-Q_big grid
    cands = np.setdiff1d(np.arange(470, 521).astype(float), np.abs(LAMBDA))
    Ns = [32, 64, 128, 256, 384]
    rows = {"grid_fixed": [], "iid_fixed": [], "iid_adversarial": []}
    cis = {k: [] for k in rows}
    for N in Ns:
        gvals, ivals, avals = [], [], []
        for _ in range(N_DRAWS):
            tg = np.sort(rng.choice(Q_big, size=N, replace=False)) / Q_big
            gvals.append(visibility(LAMBDA, tg, nu_big))
            ti = np.sort(rng.uniform(0, 1, N))
            ivals.append(visibility(LAMBDA, ti, nu_big))
            avals.append(min(visibility(LAMBDA, ti, nu) for nu in cands))
        for key, vals in (("grid_fixed", gvals), ("iid_fixed", ivals),
                          ("iid_adversarial", avals)):
            mu, ci = _mean_ci(vals)
            rows[key].append(mu); cis[key].append(ci)
    return {"N": Ns, "Q": Q_big, "nu": nu_big, "curves": rows, "cis": cis,
            "n_candidates": int(cands.size)}


def panel_d_decomposition(rng):
    """L2 error decomposition across the jitter sweep (T2): aliasing bias vs truncation."""
    sig_ts = np.concatenate([[0.0], np.geomspace(2e-5, 4e-3, 12)])
    out_f, out_c = _tone(NU, AMP, PHASE)
    in_f, in_c = _tone(11.0, 1.0, 0.4)
    in_f2, in_c2 = _tone(23.0, 0.6, 1.1)
    sig_f = np.concatenate([in_f, in_f2]); sig_c = np.concatenate([in_c, in_c2])
    allf = np.concatenate([sig_f, out_f]); allc = np.concatenate([sig_c, out_c])
    c_star = np.zeros(M, complex)
    for w, a in zip(sig_f, sig_c):
        c_star[int(np.argmin(np.abs(LAMBDA - w)))] += a
    trunc_sq = float(np.real(out_c.conj() @ continuous_gram(out_f) @ out_c))

    rows = {"total": [], "modeled": [], "residual": []}
    cis = {k: [] for k in rows}
    for st in sig_ts:
        tot, mod, res = [], [], []
        for _ in range(N_DRAWS):
            g = rng.choice(Q, size=N_GRID, replace=False)
            t = np.sort(g / Q + (rng.normal(0, st, N_GRID) if st > 0 else 0.0))
            y = evaluate(allf, allc, t, real=True) + rng.normal(0, SIGMA, N_GRID)
            model = FixedFeatureINR(LAMBDA, real=True).fit(t, y)
            dec = function_error_decomposition(LAMBDA, model.coeffs_, c_star, out_f, out_c)
            tot.append(dec["total_rmse"])
            mod.append(np.sqrt(max(dec["modeled_component_sq"], 0.0)))
            res.append(np.sqrt(np.mean((model.predict(t) - y) ** 2)))
        for key, vals in (("total", tot), ("modeled", mod), ("residual", res)):
            mu, ci = _mean_ci(vals)
            rows[key].append(mu); cis[key].append(ci)
    return {"sigma_t": sig_ts.tolist(), "curves": rows, "cis": cis,
            "truncation_rmse": float(np.sqrt(trunc_sq))}


def sampling_type_table(rng):
    """Summary across sampling families at fixed N (recorded in JSON, quoted in text)."""
    N = N_GRID
    out = {}
    samplers = {
        "full_grid": lambda: np.arange(Q)[:N] / Q,                       # first N of grid
        "grid_subset": lambda: np.sort(rng.choice(Q, N, replace=False)) / Q,
        "jittered_grid": lambda: np.sort(
            rng.choice(Q, N, replace=False) / Q + rng.normal(0, 1e-3, N)),
        "iid_uniform": lambda: np.sort(rng.uniform(0, 1, N)),
        "clustered": lambda: np.sort(rng.uniform(0, 0.45, N)),           # adversarial
    }
    cands = np.setdiff1d(np.arange(24, 55).astype(float), LAMBDA)
    for name, fn in samplers.items():
        vs, am, cond = [], [], []
        for _ in range(10):
            t = fn()
            vs.append(visibility(LAMBDA, t, NU))
            am.append(max(aliasability(LAMBDA, t, nu) for nu in cands))
            cond.append(condition_number(synthesis_matrix(LAMBDA, t)))
        out[name] = {"visibility_nu111_mean": float(np.mean(vs)),
                     "max_aliasability_mean": float(np.mean(am)),
                     "condition_number_mean": float(np.mean(cond))}
    return out


def main():
    rng = np.random.default_rng(20260712)
    A = panel_a_jitter(rng)
    B = panel_b_concentration(rng)
    C = panel_c_fixed_vs_adversarial(rng)
    D = panel_d_decomposition(rng)
    table = sampling_type_table(rng)

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.1), layout="constrained")

    ax = axes[0]
    st = np.array(A["sigma_t"]); mu = np.array(A["v_mean"]); ci = np.array(A["v_ci"])
    ax.errorbar(st[1:], mu[1:], yerr=ci[1:], fmt="o", ms=4, color="C0", label="empirical")
    ax.plot(st[1:], A["v_pred"][1:], "k--", lw=1.2, label=r"$\sqrt{1-\chi_\eta^2}$")
    ax.plot([st[1] * 0.5], [mu[0]], marker="s", color="C3", ls="none",
            label=r"$\sigma_t=0$ (exact fold)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jitter std $\sigma_t$"); ax.set_ylabel(r"visibility $v_T(\nu)$")
    ax.set_title("(a) jitter breaks the fold")
    ax.legend(fontsize=9)

    ax = axes[1]
    Ns = np.array(B["N"], float)
    mu = np.array(B["amax_mean"]); ci = np.array(B["amax_ci"])
    ax.errorbar(Ns, mu, yerr=ci, fmt="o-", ms=4, color="C0", label="empirical max")
    bnd = np.array(B["bound_delta05"]); ok = np.isfinite(bnd)
    ax.plot(Ns[ok], bnd[ok], "k:", lw=1.1, label="union bound (loose)")
    mb = np.array(B.get("matrix_bound_apriori", [np.inf] * len(Ns))); okm = np.isfinite(mb)
    if okm.any():
        ax.plot(Ns[okm], mb[okm], "C1--", lw=1.3, label="matrix-Bernstein (a-priori)")
    dd = np.array(B.get("matrix_bound_data_dependent", [np.inf] * len(Ns))); okd = np.isfinite(dd)
    if okd.any():
        ax.plot(Ns[okd], dd[okd], "C2-", lw=1.3, label="matrix-Bernstein (data-dep.)")
    ax.plot(Ns, mu[0] * np.sqrt(Ns[0] / Ns), color="0.6", ls=":", lw=1.0,
            label=r"$N^{-1/2}$ slope")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$N$ (i.i.d. samples)")
    ax.set_ylabel(r"$\max_{\nu\in\Omega} a_T(\nu)$")
    ax.set_title("(b) i.i.d. concentration")
    ax.legend(fontsize=9)

    ax = axes[2]
    Ns = np.array(C["N"], float)
    styles = {"grid_fixed": ("C3", "s", "grid subset, coherent tone"),
              "iid_fixed": ("C0", "o", "i.i.d., same tone"),
              "iid_adversarial": ("C2", "^", "i.i.d., adversarial tone")}
    for key, (col, mk, lbl) in styles.items():
        mu = np.array(C["curves"][key]); ci = np.array(C["cis"][key])
        ax.errorbar(Ns, np.maximum(mu, 1e-12), yerr=ci, fmt=f"{mk}-", ms=4,
                    color=col, label=lbl)
    ax.set_yscale("log"); ax.set_ylim(1e-12, 2)
    ax.set_xlabel("$N$"); ax.set_ylabel(r"visibility $v_T$")
    ax.set_title("(c) persistence vs random sampling")
    ax.legend(fontsize=9)

    ax = axes[3]
    st = np.array(D["sigma_t"])
    for key, col, lbl in (("total", "C0", "total $L^2$ error"),
                          ("modeled", "C3", "modeled-component error"),
                          ("residual", "0.5", "sample residual")):
        mu = np.array(D["curves"][key]); ci = np.array(D["cis"][key])
        ax.errorbar(st[1:], mu[1:], yerr=ci[1:], fmt="o-", ms=3.5, color=col, label=lbl)
        ax.plot([st[1] * 0.5], [mu[0]], marker="s", color=col, ls="none")
    ax.axhline(D["truncation_rmse"], color="k", ls="--", lw=1.2,
               label="truncation floor $\\|f_{out}\\|$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jitter std $\sigma_t$"); ax.set_ylabel("RMSE")
    ax.set_title("(d) aliasing bias vs truncation")
    ax.legend(fontsize=8.5)

    savefig(fig, "synthetic_matrix.png")

    lo_sep = float(np.min(np.diff(np.sort(LAMBDA))))
    out = {"Lambda": LAMBDA.tolist(), "m": int(M), "Q": Q, "nu": NU, "N_grid": N_GRID,
           "sigma": SIGMA, "amp": AMP, "n_draws": N_DRAWS,
           "min_freq_separation": lo_sep,
           "panel_a_jitter": A, "panel_b_concentration": B,
           "panel_c_fixed_vs_adversarial": C, "panel_d_decomposition": D,
           "sampling_type_table": table,
           "note": "theorem-validation matrix: (a) T3a characteristic-function law; "
                   "(b) T3b concentration + proved bound; (c) T1 grid persistence vs "
                   "iid visibility of fixed AND adversarial tones; (d) T2 decomposition "
                   "separating aliasing bias from truncation error"}
    save_json("synthetic_matrix.json", out)
    print("[E1-matrix] table:", {k: {kk: round(vv, 4) for kk, vv in v.items()}
                                 for k, v in table.items()}, flush=True)


if __name__ == "__main__":
    main()
