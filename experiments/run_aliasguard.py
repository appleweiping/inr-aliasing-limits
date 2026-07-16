r"""E6 -- AliasGuard: constructive sampling design against structured aliasing.

Pre-registered comparison (config, metrics, and seed list fixed at the top of this file;
no test-set tuning).  Every design uses ONLY the model dictionary ``LAMBDA``, the
candidate out-of-band set it is given, and the budget ``N`` -- never the signal, the
noise, or the held-out test frequencies.

Sections
--------
A. Design-metric comparison across methods (min visibility, max aliasability, condition
   number) in three regimes -- exact grid-coherent folds, off-grid near-band separated
   tones, and a broad band -- reported as mean +/- 95% CI.
B. Ablation: AliasGuard (joint objective) vs coherence-only and condition-only designs,
   showing the joint objective is necessary (neither single criterion controls both
   aliasing and conditioning).
C. Generalization: design on a focused, thickened concern set; evaluate on HELD-OUT
   frequencies (perturbed, unseen) and under candidate-set MISSPECIFICATION.
D. Budget sweep: worst-case aliasability vs N/m.
E. Downstream: LS reconstruction under each design (aliasing-bias part of the error and
   the in-band noise gain), so the design metric translates to signal-domain gain.
F. Two-dimensional frequency vectors (images): AliasGuard generalizes.

Honest scope (reported): AliasGuard needs a FOCUSED concern set (K = O(N)); for a broad
band densely covered with K >> N no design beats random much (over-constrained), and a
badly misspecified concern set loses the edge.  The grid-restricted greedy variant
inherits T1 and cannot break exact grid-coherent folds -- only the continuous (off-grid)
design can.

CPU-only.  Usage: python experiments/run_aliasguard.py [--quick]
"""
from __future__ import annotations

import sys

import numpy as np

from _util import save_json, savefig
from inralias.signals import evaluate
from inralias.inr import FixedFeatureINR
from inralias.identifiability import function_error_decomposition
from inralias.design import (
    design_metrics, visibility_of, aliasability_of, condition_number_of,
    aliasguard_continuous, aliasguard_greedy, aliasguard_continuous_nd,
    condition_only_design, coherence_only_design, e_optimal_design, fixed_jitter,
    aliasability_certificate, visibility_certificate,
)

import matplotlib.pyplot as plt

# ---- pre-registered config ----
LAMBDA = np.array([0.0, 5.0, -5.0, 11.0, -11.0, 17.0, -17.0, 23.0, -23.0])
M = LAMBDA.size
Q_GRID = 256
RANDOM_SEEDS = list(range(12))        # for random baselines and AG restarts
AG_SEEDS = list(range(3))             # AliasGuard init seeds (near-deterministic)
CI = 1.96
CONCERN = np.array([26.5, 33.0, 41.5, -26.5, -33.0, -41.5, 23.0 + 256, -(23.0 + 256)])


def _mean_ci(vals):
    v = np.asarray(vals, float)
    if v.size < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(CI * v.std(ddof=1) / np.sqrt(v.size))


def _worst(t, out_freqs):
    vs = [visibility_of(LAMBDA, t, float(n)) for n in out_freqs]
    az = [aliasability_of(LAMBDA, t, float(n)) for n in out_freqs]
    return min(vs), max(az), condition_number_of(LAMBDA, t)


# ---- data-independent design methods (all seeded / deterministic) ----
def _baselines(out_freqs, N, seed):
    rng = np.random.default_rng(seed)
    return {
        "random_grid_subset": np.sort(rng.choice(Q_GRID, N, replace=False)) / Q_GRID,
        "iid_uniform": np.sort(rng.uniform(0, 1, N)),
        "random_jitter": np.sort((np.arange(N) + 0.5 * rng.uniform(-1, 1, N)) / N % 1.0),
    }


def _random_search(out_freqs, N, n=200, seed=0):
    """Best-of-n i.i.d. design by the (sample-only) surrogate/max-aliasability objective."""
    rng = np.random.default_rng(seed)
    best_t, best = None, np.inf
    for _ in range(n):
        t = np.sort(rng.uniform(0, 1, N))
        a = max(aliasability_of(LAMBDA, t, float(nu)) for nu in out_freqs)
        if a < best:
            best, best_t = a, t
    return best_t


def _design_bank(out_freqs, N, quick=False):
    """All methods that produce a single deterministic design (given seeds)."""
    ns = 60 if quick else 200
    cd_kw = dict(n_sweeps=8, grid_res=320)
    bank = {
        "full_grid": np.arange(N) / max(N, Q_GRID) if N <= Q_GRID else np.arange(N) / N,
        "fixed_jitter": fixed_jitter(N),
        "condition_only": condition_only_design(LAMBDA, N, seed=0, **cd_kw),
        "coherence_only": coherence_only_design(LAMBDA, out_freqs, N, seed=0, **cd_kw),
        "e_optimal": e_optimal_design(LAMBDA, N, n_restarts=ns, seed=0),
        "random_search": _random_search(out_freqs, N, n=ns, seed=0),
        "aliasguard_greedy": aliasguard_greedy(LAMBDA, out_freqs, N, Q=Q_GRID, seed=0)[0],
        "aliasguard": aliasguard_continuous(LAMBDA, out_freqs, N, seed=0)[0],
    }
    return bank


# ======================================================================================
def section_A_regimes(quick):
    """Design metrics per method across three regimes, with CI over seeds (randoms) or
    over AG init seeds; evaluated on the design's OWN candidate set."""
    regimes = {
        "coherent_folds": np.array([w + Q_GRID for w in LAMBDA if w != 0]
                                   + [w - Q_GRID for w in LAMBDA if w != 0]),
        "near_band": _near_band_set(),
        "broad_band": np.sort(np.concatenate([np.linspace(24, 52, 15), -np.linspace(24, 52, 15)])),
    }
    N = 40
    out = {}
    for rname, OUT in regimes.items():
        methods = {}
        # randomized baselines: CI over seeds
        for label in ("random_grid_subset", "iid_uniform", "random_jitter"):
            rows = [_worst(_baselines(OUT, N, s)[label], OUT) for s in RANDOM_SEEDS]
            methods[label] = _summarize(rows)
        # deterministic / optimized designs
        bank = _design_bank(OUT, N, quick)
        for label, t in bank.items():
            methods[label] = _summarize([_worst(t, OUT)])
        # AliasGuard CI over init seeds
        ag_rows = [_worst(aliasguard_continuous(LAMBDA, OUT, N, seed=s)[0], OUT) for s in AG_SEEDS]
        methods["aliasguard"] = _summarize(ag_rows)
        out[rname] = {"N": N, "K": int(OUT.size), "methods": methods}
    return out


def _near_band_set():
    pos = LAMBDA[LAMBDA > 0]
    O = np.sort(np.concatenate([pos + 1.3, pos - 1.3, -(pos + 1.3), -(pos - 1.3)]))
    return O[np.min(np.abs(O[:, None] - LAMBDA[None, :]), axis=1) > 0.5]


def _summarize(rows):
    mv = _mean_ci([r[0] for r in rows]); ma = _mean_ci([r[1] for r in rows])
    cn = _mean_ci([r[2] for r in rows])
    return {"min_visibility": mv[0], "min_visibility_ci": mv[1],
            "max_aliasability": ma[0], "max_aliasability_ci": ma[1],
            "condition_number": cn[0], "condition_number_ci": cn[1]}


def section_C_heldout(quick):
    """Design on a focused, thickened concern set; test on held-out perturbed frequencies
    and under misspecification.  This is the deployable claim."""
    N = 40
    Odes = np.sort(np.concatenate([CONCERN + d for d in (-0.5, 0.0, 0.5)]))
    Odes = Odes[np.min(np.abs(Odes[:, None] - LAMBDA[None, :]), axis=1) > 0.3]
    Otest = np.sort(np.concatenate([CONCERN + 0.27, CONCERN - 0.31]))
    Otest = Otest[np.min(np.abs(Otest[:, None] - LAMBDA[None, :]), axis=1) > 0.2]
    Owrong = Odes + 0.5     # misspecified design set (shift), test on true Otest

    res = {"K_design": int(Odes.size), "K_test": int(Otest.size), "N": N, "methods": {}}
    # baselines: agnostic, CI over seeds on the held-out test set
    for label in ("random_jitter", "iid_uniform"):
        rows = [_worst(_baselines(Otest, N, s)[label], Otest) for s in RANDOM_SEEDS]
        res["methods"][label] = _summarize(rows)
    res["methods"]["random_search"] = _summarize([_worst(_random_search(Odes, N, 200 if not quick else 60), Otest)])
    res["methods"]["e_optimal"] = _summarize([_worst(e_optimal_design(LAMBDA, N, 200 if not quick else 60), Otest)])
    # AliasGuard designed on Odes, evaluated on held-out Otest; CI over init seeds
    ag = [_worst(aliasguard_continuous(LAMBDA, Odes, N, seed=s)[0], Otest) for s in AG_SEEDS]
    res["methods"]["aliasguard_heldout"] = _summarize(ag)
    # misspecified design
    agm = [_worst(aliasguard_continuous(LAMBDA, Owrong, N, seed=s)[0], Otest) for s in AG_SEEDS]
    res["methods"]["aliasguard_misspecified"] = _summarize(agm)
    return res


def section_D_budget(quick):
    """Worst-case aliasability vs N (held-out concern)."""
    Odes = np.sort(np.concatenate([CONCERN + d for d in (-0.5, 0.0, 0.5)]))
    Odes = Odes[np.min(np.abs(Odes[:, None] - LAMBDA[None, :]), axis=1) > 0.3]
    Otest = np.sort(np.concatenate([CONCERN + 0.27, CONCERN - 0.31]))
    Otest = Otest[np.min(np.abs(Otest[:, None] - LAMBDA[None, :]), axis=1) > 0.2]
    Ns = [16, 24, 32, 48, 64] if not quick else [24, 40]
    curves = {"aliasguard": [], "aliasguard_ci": [], "random_jitter": [],
              "random_jitter_ci": [], "e_optimal": [], "N": Ns}
    for N in Ns:
        ag = [_worst(aliasguard_continuous(LAMBDA, Odes, N, seed=s)[0], Otest)[1] for s in AG_SEEDS]
        rj = [_worst(_baselines(Otest, N, s)["random_jitter"], Otest)[1] for s in RANDOM_SEEDS]
        eo = _worst(e_optimal_design(LAMBDA, N, 120 if quick else 200), Otest)[1]
        m, c = _mean_ci(ag); curves["aliasguard"].append(m); curves["aliasguard_ci"].append(c)
        m, c = _mean_ci(rj); curves["random_jitter"].append(m); curves["random_jitter_ci"].append(c)
        curves["e_optimal"].append(eo)
    return curves


def section_E_downstream(quick):
    """Signal-domain payoff: LS reconstruction under each design.  Plant f_in + a e_nu for
    nu in the held-out concern set, fit LS on LAMBDA with noise, and split the error into
    the aliasing-bias (modeled-component) part and report the in-band noise gain."""
    N, sigma = 40, 0.03
    Odes = np.sort(np.concatenate([CONCERN + d for d in (-0.5, 0.0, 0.5)]))
    Odes = Odes[np.min(np.abs(Odes[:, None] - LAMBDA[None, :]), axis=1) > 0.3]
    Otest = np.sort(np.concatenate([CONCERN + 0.27, CONCERN - 0.31]))
    Otest = Otest[np.min(np.abs(Otest[:, None] - LAMBDA[None, :]), axis=1) > 0.2]
    amps = [0.25, 0.5, 1.0] if not quick else [0.5]
    designs = {
        "random_jitter": _baselines(Otest, N, 0)["random_jitter"],
        "e_optimal": e_optimal_design(LAMBDA, N, 120 if quick else 200),
        "aliasguard": aliasguard_continuous(LAMBDA, Odes, N, seed=0)[0],
    }
    rng = np.random.default_rng(0)
    # random Hermitian in-band signal
    def in_signal():
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        return c / np.linalg.norm(c)
    out = {"amps": amps, "sigma": sigma, "N": N, "methods": {}}
    for label, t in designs.items():
        cn = condition_number_of(LAMBDA, t)
        by_amp = {}
        for a in amps:
            biases, totals = [], []
            for nu in Otest:
                for _ in range(6):
                    c_star = in_signal()
                    of = np.array([float(nu), -float(nu)])
                    oc = np.array([a / 2, a / 2], complex)
                    y = (evaluate(LAMBDA, c_star, t, real=False)
                         + evaluate(of, oc, t, real=False)
                         + (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)) * sigma / np.sqrt(2))
                    model = FixedFeatureINR(LAMBDA, real=False).fit(t, y)
                    dec = function_error_decomposition(LAMBDA, model.coeffs_, c_star, of, oc)
                    biases.append(np.sqrt(max(dec["modeled_component_sq"], 0.0)))
                    totals.append(dec["total_rmse"])
            mb, cb = _mean_ci(biases); mt, ct = _mean_ci(totals)
            by_amp[str(a)] = {"aliasing_bias_rmse": mb, "aliasing_bias_ci": cb,
                              "total_rmse": mt, "total_ci": ct}
        out["methods"][label] = {"condition_number": cn, "by_amp": by_amp}
    return out


def section_G_certificate(quick):
    """Continuum certificate: design against a fine grid over an out-of-band BAND, then
    CERTIFY worst-case aliasability over the whole (continuous) band -- a guarantee the
    finite-candidate concentration bound and random designs do not provide."""
    N = 48
    band = [(25.0, 60.0), (-60.0, -25.0)]
    band_grid = np.sort(np.concatenate([np.arange(25, 60.001, 1.0), -np.arange(25, 60.001, 1.0)]))
    designs = {
        "random_jitter": _baselines(band_grid, N, 0)["random_jitter"],
        "e_optimal": e_optimal_design(LAMBDA, N, 120 if quick else 200),
        "aliasguard": aliasguard_continuous(LAMBDA, band_grid, N, n_sweeps=18, seed=0)[0],
    }
    ng = 600 if quick else 900   # points per unit frequency; Lipschitz slack ~L/(2*ng)
    out = {"band": band, "N": N, "methods": {}}
    for label, t in designs.items():
        ca = aliasability_certificate(LAMBDA, t, band, n_grid=ng)
        cv = visibility_certificate(LAMBDA, t, band, n_grid=ng)
        out["methods"][label] = {
            "certified_max_aliasability": ca["certified_max_aliasability"],
            "grid_max_aliasability": ca["grid_max_aliasability"],
            "certified_min_visibility": cv["certified_min_visibility"],
            "condition_number": condition_number_of(LAMBDA, t)}
    return out


def section_F_2d(quick):
    """AliasGuard in 2-D frequency vectors (images)."""
    LAM2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [2, 1], [-2, -1],
                     [1, -2], [-1, 2]], float)
    O2 = np.array([[8, 0], [0, 8], [8, 8], [-8, 0], [0, -8], [6, 3], [-6, -3], [4, 7]], float)
    N2 = 48
    rows = {"random": [], "aliasguard": []}
    for s in AG_SEEDS:
        tr = np.random.default_rng(s).uniform(0, 1, (N2, 2))
        mr = design_metrics(LAM2, O2, tr)
        rows["random"].append((mr["min_visibility"], mr["max_aliasability"], mr["condition_number"]))
        _t, m2 = aliasguard_continuous_nd(LAM2, O2, N2, d=2, n_sweeps=6 if quick else 10,
                                          grid_res=40 if quick else 56, seed=s)
        rows["aliasguard"].append((m2["min_visibility"], m2["max_aliasability"], m2["condition_number"]))
    return {"m": int(LAM2.shape[0]), "K": int(O2.shape[0]), "N": N2,
            "random": _summarize(rows["random"]), "aliasguard": _summarize(rows["aliasguard"])}


# ======================================================================================
def make_figure(A, C, D, E, F):
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.2), layout="constrained")

    # (a) ablation on the near-band regime: the joint objective is necessary
    ax = axes[0]
    reg = A["near_band"]["methods"]
    order = ["random_jitter", "e_optimal", "condition_only", "coherence_only", "aliasguard"]
    labs = ["rand\njitter", "E-opt", "cond\nonly", "coh\nonly", "Alias\nGuard"]
    ma = [reg[k]["max_aliasability"] for k in order]
    cn = [reg[k]["condition_number"] for k in order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, ma, 0.4, label="max aliasability", color="C3")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, cn, 0.4, label="cond. number", color="C0", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("max aliasability", color="C3"); ax2.set_ylabel("cond. number", color="C0")
    ax.set_title("(a) joint objective needed\n(near-band, N=40)", fontsize=9)
    ax.axhline(0, color="k", lw=0.5)

    # (b) held-out generalization
    ax = axes[1]
    hm = C["methods"]
    order = ["random_jitter", "e_optimal", "random_search", "aliasguard_misspecified", "aliasguard_heldout"]
    labs = ["rand\njitter", "E-opt", "rand\nsearch", "AG\n(misspec)", "AG\n(held-out)"]
    ma = [hm[k]["max_aliasability"] for k in order]
    ci = [hm[k]["max_aliasability_ci"] for k in order]
    cols = ["0.6", "0.6", "0.6", "C1", "C2"]
    ax.bar(np.arange(len(order)), ma, yerr=ci, color=cols, capsize=3)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("max aliasability (held-out)")
    ax.set_title("(b) generalization to\nunseen frequencies", fontsize=9)

    # (c) budget sweep
    ax = axes[2]
    Ns = np.array(D["N"], float)
    for key, col, mk in (("aliasguard", "C2", "o"), ("random_jitter", "0.5", "s"), ("e_optimal", "C0", "^")):
        y = np.array(D[key]); ax.plot(Ns, y, mk + "-", color=col, ms=4, label=key.replace("_", " "))
        if key + "_ci" in D:
            c = np.array(D[key + "_ci"]); ax.fill_between(Ns, y - c, y + c, color=col, alpha=0.2)
    ax.set_xlabel("budget N"); ax.set_ylabel("max aliasability (held-out)")
    ax.set_title("(c) vs sample budget", fontsize=9); ax.legend(fontsize=8)

    # (d) downstream aliasing-bias reduction
    ax = axes[3]
    amps = np.array(E["amps"], float)
    for key, col, mk in (("aliasguard", "C2", "o"), ("random_jitter", "0.5", "s"), ("e_optimal", "C0", "^")):
        y = [E["methods"][key]["by_amp"][str(a)]["aliasing_bias_rmse"] for a in amps]
        c = [E["methods"][key]["by_amp"][str(a)]["aliasing_bias_ci"] for a in amps]
        ax.errorbar(amps, y, yerr=c, fmt=mk + "-", color=col, ms=4,
                    label=f"{key.replace('_',' ')} (κ={E['methods'][key]['condition_number']:.2f})")
    ax.set_xlabel("out-of-band amplitude"); ax.set_ylabel("aliasing-bias RMSE")
    ax.set_title("(d) signal-domain payoff", fontsize=9); ax.legend(fontsize=7.5)
    savefig(fig, "aliasguard.png")


def main():
    quick = "--quick" in sys.argv
    print("[AG] section A: regimes ...", flush=True); A = section_A_regimes(quick)
    print("[AG] section C: held-out ...", flush=True); C = section_C_heldout(quick)
    print("[AG] section D: budget sweep ...", flush=True); D = section_D_budget(quick)
    print("[AG] section E: downstream ...", flush=True); E = section_E_downstream(quick)
    print("[AG] section G: continuum certificate ...", flush=True); G = section_G_certificate(quick)
    print("[AG] section F: 2-D ...", flush=True); F = section_F_2d(quick)
    make_figure(A, C, D, E, F)
    out = {"config": {"LAMBDA": LAMBDA.tolist(), "m": M, "Q_grid": Q_GRID,
                      "random_seeds": RANDOM_SEEDS, "ag_seeds": AG_SEEDS,
                      "concern": CONCERN.tolist()},
           "A_regimes": A, "C_heldout": C, "D_budget": D, "E_downstream": E,
           "G_certificate": G, "F_2d": F,
           "note": "AliasGuard constructive sampling design; all designs use only "
                   "LAMBDA/Omega/N; held-out and misspecification reported; ablation shows "
                   "the joint objective is necessary; grid-greedy inherits T1 (cannot "
                   "break exact grid folds), continuous design can."}
    save_json("aliasguard.json", out)
    # headline print
    hm = C["methods"]
    print(f"[AG] HELD-OUT maxAliasability: AliasGuard {hm['aliasguard_heldout']['max_aliasability']:.3f}"
          f"+-{hm['aliasguard_heldout']['max_aliasability_ci']:.3f} vs "
          f"random_jitter {hm['random_jitter']['max_aliasability']:.3f}, "
          f"E-opt {hm['e_optimal']['max_aliasability']:.3f}; "
          f"misspec {hm['aliasguard_misspecified']['max_aliasability']:.3f}", flush=True)
    print(f"[AG] 2-D: AliasGuard maxAlias {F['aliasguard']['max_aliasability']:.3f} vs "
          f"random {F['random']['max_aliasability']:.3f}", flush=True)
    gm = G["methods"]
    print(f"[AG] CONTINUUM certificate (band-wide certified max-aliasability): "
          f"AliasGuard {gm['aliasguard']['certified_max_aliasability']:.3f} vs "
          f"random_jitter {gm['random_jitter']['certified_max_aliasability']:.3f}, "
          f"E-opt {gm['e_optimal']['certified_max_aliasability']:.3f}", flush=True)


if __name__ == "__main__":
    main()
