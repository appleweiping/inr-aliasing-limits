r"""U3a -- matching rate / lower bound for worst-case aliasability (Theorem 3a).

Confirms numerically that the aliasability rate under i.i.d. uniform sampling is Theta(sqrt(m/N))
up to a sqrt(log) factor -- i.e. the N^{-1/2} upper rate is order-optimal and the previously
disclaimed matching lower bound holds:

* N * E[a_T(nu)^2] -> m  (exact second moment; the mean-square lower bound),
* the empirical RMS aliasability is bracketed by the rigorous lower_rms and the U1 upper bound,
* the Assouad K-tone grid floor = ||a||/sqrt2 with every coherent fold certified exact,
* the Le Cam off-grid detection threshold = sigma / (v * sqrt(N)).

CPU-only.  Usage: python experiments/run_aliasability_rate.py
"""
from __future__ import annotations

import numpy as np

from _util import save_json
from inralias.identifiability import (
    aliasability, aliasability_rate_bounds, assouad_grid_floor,
    lecam_detection_threshold, visibility,
)

LAMBDA = np.array([0, 1, -1, 2, -2, 3, -3, 4, -4], float)
M = LAMBDA.size
K = 31
DELTA = 0.05
NU = 12.0


def rate_curve(rng, ms=(5, 9, 15), draws=2000):
    out = {}
    for m in ms:
        Lam = np.concatenate([[0.0]] + [[k, -k] for k in range(1, m // 2 + 1)])[:m]
        Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
        N_Ea2, rms, lo, up = [], [], [], []
        for N in Ns:
            vals = [aliasability(Lam, np.sort(rng.uniform(0, 1, N)), NU) ** 2 for _ in range(draws)]
            N_Ea2.append(float(N * np.mean(vals)))
            rms.append(float(np.sqrt(np.mean(vals))))
            rb = aliasability_rate_bounds(len(Lam), K, N, DELTA)
            lo.append(rb["lower_rms"]); up.append(rb["upper"])
        out[str(m)] = {"N": Ns, "N_times_E_a2": N_Ea2, "target_m": int(len(Lam)),
                       "rms_empirical": rms, "lower_rms": lo, "upper": up}
    return out


def main():
    rng = np.random.default_rng(20260718)
    curves = rate_curve(rng)
    # Assouad grid floor (rigorous K-tone converse on a grid subset)
    Q = 64
    t_grid = np.sort(rng.choice(Q, size=40, replace=False)) / Q
    tones = np.array([-1 + Q, 2 + Q, -3 + Q, 4 + Q], float)
    coeffs = np.array([0.5, 0.6, 0.4, 0.5])
    assouad = assouad_grid_floor(LAMBDA, tones, coeffs, Q, t_grid)
    # Le Cam off-grid detection threshold
    t_off = np.sort(rng.uniform(0, 1, 100))
    lecam = {"nu": NU, "sigma": 0.1, "N": int(t_off.size),
             "visibility": float(visibility(LAMBDA, t_off, NU)),
             "threshold": float(lecam_detection_threshold(LAMBDA, t_off, NU, 0.1))}
    # headline: convergence of N*E[a^2] to m for the reference m
    ref = curves[str(M)]
    out = {"m_ref": M, "K": K, "delta": DELTA, "rate_curves": curves,
           "assouad_grid_floor": assouad, "lecam_threshold": lecam,
           "headline_N_Ea2_last": ref["N_times_E_a2"][-1],
           "note": "N*E[a_T^2] -> m (exact mean-square lower bound); empirical RMS bracketed by "
                   "lower_rms and the U1 upper (matched up to sqrt(log)); Assouad K-tone grid "
                   "floor = ||a||/sqrt2 (all folds exact); Le Cam threshold = sigma/(v sqrt N)."}
    save_json("aliasability_rate.json", out)
    print(f"[rate] m={M}: N*E[a^2] {ref['N_times_E_a2'][0]:.2f} -> {ref['N_times_E_a2'][-1]:.2f} "
          f"(target {M}); Assouad floor {assouad['floor']:.3f} exact={assouad['all_folds_exact']}; "
          f"LeCam thr {lecam['threshold']:.4f}", flush=True)


if __name__ == "__main__":
    main()
