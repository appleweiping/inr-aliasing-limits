r"""U2 -- certified anti-aliasing design via a convex relaxation + Frank-Wolfe (Theorem 2).

The position-space AliasGuard surrogate is non-convex ("no greedy guarantee").  Lifting to the
design MEASURE over a super-grid makes it a convex quadratic J(w)=w^T Q w (Q PSD); Frank-Wolfe
with exact line search converges monotonically to the GLOBAL optimum with a computable
duality-gap CERTIFICATE g_s -> 0 that upper-bounds J(w_s)-J*.  This converts the design from a
heuristic into a certified epsilon-optimal procedure.  We report, over several scenarios: the
FW gap decay, the certified optimum vs the coordinate-descent (local) surrogate value, and the
measure->N rounding gap (bounded by Theorem 1').

CPU-only.  Usage: python experiments/run_certified_design.py
"""
from __future__ import annotations

import numpy as np

from _util import save_json
from inralias.design import (
    aliasguard_frankwolfe, aliasguard_continuous, surrogate_objective,
    surrogate_hessian, rounding_gap_bound,
)


def _scenario(s):
    rng = np.random.default_rng(50_000 + s)
    m_half = 2
    pos = np.sort(rng.uniform(2, 9, m_half))
    Lam = np.concatenate([[0.0], pos, -pos])
    centers = rng.uniform(26, 44, size=3)
    Om = np.sort(np.concatenate([centers + 0.5, -centers - 0.5, centers - 0.5, -centers + 0.5]))
    return Lam, Om


def main():
    P = 500
    grid = np.arange(P) / P
    N = 40
    recs = []
    for s in range(8):
        Lam, Om = _scenario(s)
        Q = surrogate_hessian(Lam, Om, grid)
        min_eig = float(np.linalg.eigvalsh(Q)[0])
        w, info = aliasguard_frankwolfe(Lam, Om, grid, beta=1.0, n_iter=2500)
        t_cd, _ = aliasguard_continuous(Lam, Om, N, n_sweeps=15, grid_res=480, seed=s)
        J_cd = surrogate_objective(Lam, Om, t_cd, 1.0)
        rg = rounding_gap_bound(Lam, Om, w, grid, N, beta=1.0, delta=0.05, seed=s)
        recs.append({
            "s": s, "hessian_min_eig": min_eig, "iters": len(info["J"]),
            "J_star": float(info["J"][-1]), "min_gap": info["min_gap"],
            "monotone": bool(all(info["J"][i] >= info["J"][i + 1] - 1e-10
                                 for i in range(len(info["J"]) - 1))),
            "J_coord_descent": float(J_cd), "support_size": info["support_size"],
            "rounding_gap": rg["rounding_gap"], "coherence_radius": rg["coherence_radius"],
        })
    summary = {
        "n_scenarios": len(recs), "N": N, "grid_size": P,
        "hessian_psd_all": bool(all(r["hessian_min_eig"] >= -1e-7 for r in recs)),
        "monotone_all": bool(all(r["monotone"] for r in recs)),
        "max_certified_gap": float(max(r["min_gap"] for r in recs)),
        "fw_le_coord_descent_all": bool(all(r["J_star"] <= r["J_coord_descent"] + 1e-9
                                            for r in recs)),
        "mean_rounding_gap": float(np.mean([r["rounding_gap"] for r in recs])),
    }
    out = {"summary": summary, "scenarios": recs,
           "note": "Convex design-measure relaxation + Frank-Wolfe with exact line search: PSD "
                   "Hessian, monotone decrease, duality gap -> 0 (certified global optimum of "
                   "the surrogate), FW optimum <= coordinate-descent value; measure->N rounding "
                   "gap bounded by the Theorem 1' coherence radius. D_s-optimal measures admit "
                   "the analogous Kiefer-Wolfowitz equivalence certificate (classical)."}
    save_json("certified_design.json", out)
    print(f"[cert-design] scenarios={len(recs)} PSD={summary['hessian_psd_all']} "
          f"monotone={summary['monotone_all']} max_cert_gap={summary['max_certified_gap']:.2e} "
          f"FW<=CD={summary['fw_le_coord_descent_all']} mean_round_gap={summary['mean_rounding_gap']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
