r"""E6 -- constructive anti-aliasing sample design, evaluated honestly.

Design principle: choose sample times so the model Gram is well conditioned AND its
coupling to a specified out-of-band concern set is small.  This is *related to / motivated
by* Ds-optimal (nuisance-parameter) experimental design and LCMV null-steering (classical;
credited, not claimed).  The AliasGuard surrogate is NOT identical to the exact Ds
objective -- ``ds_optimal_design`` is the exact Ds baseline, included for comparison.

Statistical protocol (pre-registered; fixed here; NO test-set tuning):
* ``N_SCENARIOS`` independent OUTER scenarios, each fixing its own dictionary ``Lambda``
  (alternating orthonormal/integer and non-orthonormal/non-integer), concern set, noise,
  and optimizer initialization.  ALL methods run on the SAME scenarios (paired).
* the reported quantity is the FUNCTION-SPACE worst-case aliasability
  ``max_nu a_{L2,T}(nu)`` on HELD-OUT frequency sets, of several distinct types.
* CIs are paired bootstrap over the outer scenarios (the replication unit); every method,
  even a deterministic one, has a genuine scenario distribution -- no CI=0 artifact.
* wall-clock per method is recorded (matched-budget context).

Each design is computed ONCE per scenario and reused across all sections.  Scenarios are
processed in parallel (ProcessPoolExecutor) when available.

CPU-only.  Usage: python experiments/run_aliasguard.py [--quick] [--scenarios N] [--jobs J]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from _util import save_json, savefig
from inralias.identifiability import continuous_gram, function_error_decomposition
from inralias.sampling import synthesis_matrix
from inralias.design import (
    aliasability_L2_of, visibility_of, condition_number_of,
    aliasguard_continuous, ds_optimal_design, e_optimal_design,
    coherence_only_design, condition_only_design, fixed_jitter, annealing_design,
    aliasability_certificate,
)

import matplotlib.pyplot as plt

N_DEFAULT = 40
N_SCENARIOS = 24
CERT_BAND = [(25.0, 60.0), (-60.0, -25.0)]
HELDOUT_TYPES = ["near", "omitted_band", "severe_shift", "scale_error", "broadband_ood"]
METHODS = ["aliasguard", "ds_optimal", "e_optimal", "random_jitter", "iid_uniform",
           "low_discrepancy", "random_search", "coherence_only", "condition_only", "annealing"]
DECOMP_DESIGNS = ["aliasguard", "ds_optimal", "random_jitter"]
CERT_DESIGNS = ["aliasguard", "ds_optimal", "e_optimal", "random_jitter"]


def make_scenario(s: int):
    rng = np.random.default_rng(10_000 + s)
    m_half = 4
    if s % 2 == 0:
        pos = np.sort(rng.choice(np.arange(3, 26), size=m_half, replace=False)).astype(float)
    else:
        pos = np.sort(rng.uniform(3, 26, size=m_half))
    LAM = np.concatenate([[0.0], pos, -pos])
    n_centers = int(rng.integers(3, 5))
    centers = rng.uniform(28, 52, size=n_centers)
    Odes = np.sort(np.concatenate([np.concatenate([centers, -centers]) + d for d in (-0.5, 0.0, 0.5)]))
    Odes = _clean(Odes, LAM, 0.4)
    held = {}
    held["near"] = _clean(np.concatenate([centers + 0.27, -centers + 0.27,
                                          centers - 0.31, -centers - 0.31]), LAM)
    omit = rng.uniform(28, 52)
    held["omitted_band"] = _clean(np.concatenate(
        [omit + np.linspace(-0.4, 0.4, 5), -(omit + np.linspace(-0.4, 0.4, 5))]), LAM)
    held["severe_shift"] = _clean(np.concatenate([centers + 3.0, -centers - 3.0]), LAM)
    held["scale_error"] = _clean(np.concatenate([centers * 1.15, -centers * 1.15]), LAM)
    held["broadband_ood"] = _clean(np.concatenate([np.linspace(27, 55, 20), -np.linspace(27, 55, 20)]), LAM)
    sigma = float(rng.choice([0.02, 0.05, 0.1]))
    return {"s": s, "LAMBDA": LAM.tolist(), "Odes": Odes.tolist(),
            "held": {k: v.tolist() for k, v in held.items()}, "sigma": sigma,
            "orthonormal": bool(s % 2 == 0)}


def _clean(freqs, LAM, guard=0.25):
    f = np.unique(np.round(np.asarray(freqs, float), 6))
    LAM = np.asarray(LAM, float)
    return f[np.min(np.abs(f[:, None] - LAM[None, :]), axis=1) > guard]


def build_design(method, LAM, Odes, N, seed, quick):
    ns = 60 if quick else 200
    t0 = time.perf_counter()
    if method == "aliasguard":
        t, _ = aliasguard_continuous(LAM, Odes, N, n_sweeps=15, grid_res=480, seed=seed)
    elif method == "ds_optimal":
        t = ds_optimal_design(LAM, Odes, N, n_sweeps=6, grid_res=160, seed=seed)
    elif method == "e_optimal":
        t = e_optimal_design(LAM, N, n_restarts=ns, seed=seed)
    elif method == "random_jitter":
        r = np.random.default_rng(seed)
        t = np.sort((np.arange(N) + 0.5 * r.uniform(-1, 1, N)) / N % 1.0)
    elif method == "iid_uniform":
        t = np.sort(np.random.default_rng(seed).uniform(0, 1, N))
    elif method == "low_discrepancy":
        t = fixed_jitter(N)
    elif method == "random_search":
        t = _random_search(LAM, Odes, N, n=ns, seed=seed)
    elif method == "coherence_only":
        t = coherence_only_design(LAM, Odes, N, seed=seed, n_sweeps=6, grid_res=160)
    elif method == "condition_only":
        t = condition_only_design(LAM, N, seed=seed, n_sweeps=6, grid_res=160)
    elif method == "annealing":
        t = annealing_design(LAM, Odes, N, maxiter=60 if not quick else 40, seed=seed)
    else:
        raise ValueError(method)
    return np.asarray(t, float), time.perf_counter() - t0


def _random_search(LAM, Odes, N, n, seed):
    rng = np.random.default_rng(seed)
    O = np.atleast_1d(np.asarray(Odes, float))
    best_t, best = None, np.inf
    for _ in range(n):
        t = np.sort(rng.uniform(0, 1, N))
        a = max(aliasability_L2_of(LAM, t, float(nu)) for nu in O)
        if a < best:
            best, best_t = a, t
    return best_t


def _worst_L2(LAM, t, freqs):
    freqs = np.asarray(freqs, float)
    return max((aliasability_L2_of(LAM, t, float(nu)) for nu in freqs), default=0.0)


def _min_vis(LAM, t, freqs):
    freqs = np.asarray(freqs, float)
    return min((visibility_of(LAM, t, float(nu)) for nu in freqs), default=1.0)


# --------------------------------------------------------------------------------------
# one scenario -> all metrics (design computed once per method)
# --------------------------------------------------------------------------------------
def process_scenario(args):
    sc, N, quick, methods, amps, cert_ng = args
    LAM = np.asarray(sc["LAMBDA"], float)
    Odes = np.asarray(sc["Odes"], float)
    held = {k: np.asarray(v, float) for k, v in sc["held"].items()}
    sigma = sc["sigma"]
    designs = {m: build_design(m, LAM, Odes, N, seed=sc["s"], quick=quick) for m in methods}
    rec = {"s": sc["s"], "methods": {}}
    for m, (t, dt) in designs.items():
        rec["methods"][m] = {
            "design_set": _worst_L2(LAM, t, Odes),
            "cond": condition_number_of(LAM, t),
            "min_vis_near": _min_vis(LAM, t, held["near"]),
            "time": dt,
            "heldout": {ht: _worst_L2(LAM, t, held[ht]) for ht in HELDOUT_TYPES},
        }
    # signal-domain decomposition (P0-F): paired bias/noise/truncation/total
    rec["decomp"] = {}
    G_L2 = continuous_gram(LAM)
    for d in DECOMP_DESIGNS:
        t = designs[d][0]
        Phi = synthesis_matrix(LAM, t)
        G = Phi.conj().T @ Phi
        s = np.linalg.svd(Phi, compute_uv=False)
        entry = {"tr_ginv": float(np.real(np.trace(np.linalg.inv(G)))),
                 "sigma_min_inv2": float(1.0 / s[-1] ** 2), "cond": float(s[0] / s[-1]),
                 "by_amp": {}}
        rng = np.random.default_rng(20_000 + sc["s"])
        m = LAM.size
        for a in amps:
            bias, noise, trunc, total = [], [], [], []
            for nu in held["near"][:4]:
                c_star = rng.standard_normal(m) + 1j * rng.standard_normal(m)
                c_star = c_star / np.linalg.norm(c_star)
                of = np.array([float(nu), -float(nu)]); oc = np.array([a / 2, a / 2], complex)
                y_in = Phi @ c_star
                y_tone = synthesis_matrix(of, t) @ oc
                eps = (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)) * sigma / np.sqrt(2)
                ch_bias = np.linalg.lstsq(Phi, y_in + y_tone, rcond=None)[0]
                ch_noise = np.linalg.lstsq(Phi, y_in + eps, rcond=None)[0]
                ch_total = np.linalg.lstsq(Phi, y_in + y_tone + eps, rcond=None)[0]
                bias.append(np.sqrt(max(function_error_decomposition(LAM, ch_bias, c_star, of, oc)["modeled_component_sq"], 0.0)))
                dcn = ch_noise - c_star
                noise.append(float(np.sqrt(max(np.real(dcn.conj() @ G_L2 @ dcn), 0.0))))
                trunc.append(float(np.sqrt(max(np.real(oc.conj() @ continuous_gram(of) @ oc), 0.0))))
                total.append(function_error_decomposition(LAM, ch_total, c_star, of, oc)["total_rmse"])
            entry["by_amp"][str(a)] = {"bias": float(np.mean(bias)), "noise": float(np.mean(noise)),
                                       "trunc": float(np.mean(trunc)), "total": float(np.mean(total))}
        rec["decomp"][d] = entry
    # continuum certificate (function-space) for the cert designs
    rec["cert"] = {}
    for d in CERT_DESIGNS:
        if d == "aliasguard":
            band_grid = np.sort(np.concatenate([np.arange(25, 60.01, 0.5), -np.arange(25, 60.01, 0.5)]))
            t, _ = aliasguard_continuous(LAM, band_grid, N, n_sweeps=12, grid_res=480, seed=sc["s"])
        else:
            t = designs[d][0]
        c = aliasability_certificate(LAM, t, CERT_BAND, n_grid=cert_ng, metric="l2")
        rec["cert"][d] = {"certified": c["certified_max_aliasability"],
                          "cond": c["condition_number"], "full_rank": bool(c["full_rank"])}
    return rec


def _paired_bootstrap_ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, float)
    if v.size < 2:
        return float(v.mean()) if v.size else 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(records, methods, amps):
    summ = {}
    for m in methods:
        heldout = {}
        for ht in HELDOUT_TYPES:
            vals = [r["methods"][m]["heldout"][ht] for r in records]
            mean, lo, hi = _paired_bootstrap_ci(vals)
            heldout[ht] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                           "values": [round(x, 4) for x in vals]}
        dm, dlo, dhi = _paired_bootstrap_ci([r["methods"][m]["design_set"] for r in records])
        cm, clo, chi = _paired_bootstrap_ci([r["methods"][m]["cond"] for r in records])
        times = [r["methods"][m]["time"] for r in records]
        summ[m] = {"heldout": heldout, "design_set": {"mean": dm, "ci_lo": dlo, "ci_hi": dhi},
                   "condition_number": {"mean": cm, "ci_lo": clo, "ci_hi": chi},
                   "min_visibility_near": float(np.mean([r["methods"][m]["min_vis_near"] for r in records])),
                   "wall_clock_s": {"mean": float(np.mean(times)), "total": float(np.sum(times))}}
    # decomposition
    decomp = {"amps": amps, "designs": DECOMP_DESIGNS, "agg": {}}
    for d in DECOMP_DESIGNS:
        by_amp = {}
        for a in amps:
            for k in ("bias", "noise", "trunc", "total"):
                vals = [r["decomp"][d]["by_amp"][str(a)][k] for r in records]
                by_amp.setdefault(str(a), {})[k] = {"mean": float(np.mean(vals)),
                    "ci": float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)}
        decomp["agg"][d] = {"by_amp": by_amp,
                            "tr_ginv": float(np.mean([r["decomp"][d]["tr_ginv"] for r in records])),
                            "sigma_min_inv2": float(np.mean([r["decomp"][d]["sigma_min_inv2"] for r in records])),
                            "cond": float(np.mean([r["decomp"][d]["cond"] for r in records]))}
    # certificate
    cert = {}
    for d in CERT_DESIGNS:
        cvals = [r["cert"][d]["certified"] for r in records if np.isfinite(r["cert"][d]["certified"])]
        cert[d] = {"certified_mean": float(np.mean(cvals)) if cvals else float("inf"),
                   "certified_values": [round(r["cert"][d]["certified"], 4)
                                        if np.isfinite(r["cert"][d]["certified"]) else None for r in records],
                   "cond_mean": float(np.mean([r["cert"][d]["cond"] for r in records])),
                   "all_full_rank": bool(all(r["cert"][d]["full_rank"] for r in records))}
    return summ, decomp, cert


# --- module-level task workers so the serial sections can run on the shared pool ----------
def _pareto_ag_task(args):
    b, sc, N = args
    LAM = np.asarray(sc["LAMBDA"], float)
    t, _ = aliasguard_continuous(LAM, np.asarray(sc["Odes"], float), N, beta=b,
                                 n_sweeps=12, grid_res=480, seed=sc["s"])
    return float(_worst_L2(LAM, t, np.asarray(sc["held"]["near"], float))), \
        float(condition_number_of(LAM, t))


def _pareto_ref_task(args):
    ref, sc, N, quick = args
    LAM = np.asarray(sc["LAMBDA"], float)
    if ref == "ds_optimal":
        t = ds_optimal_design(LAM, np.asarray(sc["Odes"], float), N, n_sweeps=5,
                              grid_res=200, seed=sc["s"])
    else:
        t = e_optimal_design(LAM, N, n_restarts=120 if quick else 200, seed=sc["s"])
    return ref, float(_worst_L2(LAM, t, np.asarray(sc["held"]["near"], float))), \
        float(condition_number_of(LAM, t))


def _budget_task(args):
    mth, N, sc, quick = args
    LAM = np.asarray(sc["LAMBDA"], float)
    t, _ = build_design(mth, LAM, np.asarray(sc["Odes"], float), N, seed=sc["s"], quick=quick)
    return mth, N, float(_worst_L2(LAM, t, np.asarray(sc["held"]["near"], float)))


def _twod_task(args):
    s, quick = args
    from inralias.design import aliasguard_continuous_nd
    N2 = 48
    rng = np.random.default_rng(30_000 + s)
    LAM2 = np.unique(rng.integers(-3, 4, size=(14, 2)), axis=0).astype(float)[:9]
    des = rng.integers(-9, 10, size=(8, 2)).astype(float)
    des = des[np.min(np.linalg.norm(des[:, None] - LAM2[None], axis=2), axis=1) > 1.0]
    test = rng.integers(-9, 10, size=(8, 2)).astype(float)
    test = test[np.min(np.linalg.norm(test[:, None] - LAM2[None], axis=2), axis=1) > 1.0]
    tr = rng.uniform(0, 1, (N2, 2))
    rd = max((aliasability_L2_of(LAM2, tr, test[i]) for i in range(len(test))), default=0.0)
    t2, _ = aliasguard_continuous_nd(LAM2, des, N2, d=2, n_sweeps=5 if quick else 8,
                                     grid_res=32 if quick else 44, seed=s)
    ag = max((aliasability_L2_of(LAM2, t2, test[i]) for i in range(len(test))), default=0.0)
    return float(ag), float(rd)


def _pmap(ex, fn, tasks):
    """Map via the shared executor if present, else serially."""
    return list(ex.map(fn, tasks)) if ex is not None else [fn(t) for t in tasks]


def section_pareto(scenarios, N, quick, ex=None):
    betas = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0] if not quick else [0.2, 1.0, 5.0]
    pts = {"beta": betas, "aliasguard": [], "aliasguard_cond": []}
    sub = scenarios[: (4 if quick else 8)]
    ag_res = _pmap(ex, _pareto_ag_task, [(b, sc, N) for b in betas for sc in sub])
    ag_res = [ag_res[i * len(sub):(i + 1) * len(sub)] for i in range(len(betas))]
    for block in ag_res:
        pts["aliasguard"].append(float(np.mean([a for a, _ in block])))
        pts["aliasguard_cond"].append(float(np.mean([c for _, c in block])))
    ref_res = _pmap(ex, _pareto_ref_task,
                    [(ref, sc, N, quick) for ref in ("ds_optimal", "e_optimal") for sc in sub])
    for ref in ("ds_optimal", "e_optimal"):
        block = [r for r in ref_res if r[0] == ref]
        pts[ref] = [float(np.mean([al for _, al, _ in block])),
                    float(np.mean([cn for _, _, cn in block]))]
    return pts


def section_budget(scenarios, quick, ex=None):
    Ns = [16, 24, 32, 48, 64] if not quick else [24, 40]
    sub = scenarios[: (6 if quick else 12)]
    curves = {"N": Ns}
    methods = ("aliasguard", "ds_optimal", "random_jitter")
    tasks = [(mth, N, sc, quick) for mth in methods for N in Ns for sc in sub]
    res = _pmap(ex, _budget_task, tasks)
    for mth in methods:
        means, cis = [], []
        for N in Ns:
            vals = [v for (m, n, v) in res if m == mth and n == N]
            m_, lo, hi = _paired_bootstrap_ci(vals)
            means.append(m_); cis.append((hi - lo) / 2)
        curves[mth] = means; curves[mth + "_ci"] = cis
    return curves


def section_2d(quick, ex=None):
    res = _pmap(ex, _twod_task, [(s, quick) for s in range(6 if quick else 12)])
    ag = [a for a, _ in res]; rd = [r for _, r in res]
    am, alo, ahi = _paired_bootstrap_ci(ag); rm, rlo, rhi = _paired_bootstrap_ci(rd)
    return {"aliasguard": {"mean": am, "ci_lo": alo, "ci_hi": ahi},
            "random": {"mean": rm, "ci_lo": rlo, "ci_hi": rhi},
            "note": "held-out 2-D frequency VECTORS distinct from the design set"}


def make_figure(summ, pareto, decomp, cert):
    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.3), layout="constrained")
    ax = axes[0]
    ax.plot(pareto["aliasguard_cond"], pareto["aliasguard"], "o-", color="C2", ms=4, label="AliasGuard ($\\beta$)")
    ax.plot(pareto["ds_optimal"][1], pareto["ds_optimal"][0], "s", color="C0", ms=8, label="exact $D_s$")
    ax.plot(pareto["e_optimal"][1], pareto["e_optimal"][0], "^", color="C3", ms=8, label="E-optimal")
    ax.set_xlabel("condition number"); ax.set_ylabel("held-out max $a_{L^2}$")
    ax.set_title("(a) aliasability--conditioning\nPareto", fontsize=9); ax.legend(fontsize=7.5)
    ax = axes[1]
    x = np.arange(len(HELDOUT_TYPES))
    for i, mth in enumerate(["aliasguard", "ds_optimal", "random_jitter"]):
        y = [summ[mth]["heldout"][ht]["mean"] for ht in HELDOUT_TYPES]
        lo = [max(0, summ[mth]["heldout"][ht]["mean"] - summ[mth]["heldout"][ht]["ci_lo"]) for ht in HELDOUT_TYPES]
        hi = [max(0, summ[mth]["heldout"][ht]["ci_hi"] - summ[mth]["heldout"][ht]["mean"]) for ht in HELDOUT_TYPES]
        ax.errorbar(x + (i - 1) * 0.22, y, yerr=[lo, hi], fmt="o", ms=4, capsize=2, label=mth.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in HELDOUT_TYPES], fontsize=7)
    ax.set_ylabel("held-out max $a_{L^2}$"); ax.set_title("(b) held-out by type\n(paired bootstrap CI)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax = axes[2]
    ds = CERT_DESIGNS
    ax.bar(range(len(ds)), [cert[d]["certified_mean"] for d in ds], color=["C2", "C0", "C3", "0.6"])
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([d.replace("_", "\n") for d in ds], fontsize=7.5)
    ax.set_ylabel("certified band-wide $a_{L^2}$"); ax.set_title("(c) continuum certificate\n(any design)", fontsize=9)
    ax = axes[3]
    amps = np.array(decomp["amps"], float)
    for d, col in (("aliasguard", "C2"), ("random_jitter", "0.5")):
        ba = decomp["agg"][d]["by_amp"]
        ax.plot(amps, [ba[str(a)]["bias"]["mean"] for a in amps], "o-", color=col, label=f"{d.replace('_',' ')}: alias bias")
    d0 = decomp["agg"]["aliasguard"]["by_amp"]
    ax.axhline(d0[str(amps[-1])]["trunc"]["mean"], color="k", ls="--", lw=1, label="truncation")
    ax.axhline(d0[str(amps[-1])]["noise"]["mean"], color="C1", ls=":", lw=1, label="noise")
    ax.set_xlabel("out-of-band amplitude"); ax.set_ylabel("$L^2$ error component")
    ax.set_title("(d) bias/noise/truncation\nseparated", fontsize=9); ax.legend(fontsize=7)
    savefig(fig, "aliasguard.png")


def main():
    quick = "--quick" in sys.argv
    nsc = N_SCENARIOS
    if "--scenarios" in sys.argv:
        nsc = int(sys.argv[sys.argv.index("--scenarios") + 1])
    if quick:
        nsc = min(nsc, 6)
    jobs = 1
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
    N = N_DEFAULT
    amps = [0.5, 1.0] if quick else [0.25, 0.5, 1.0]
    cert_ng = 2500 if quick else 3000
    methods = [m for m in METHODS if not (quick and m == "annealing")]
    scenarios = [make_scenario(s) for s in range(nsc)]

    print(f"[AG] scenarios={nsc} methods={len(methods)} jobs={jobs} ...", flush=True)
    tasks = [(sc, N, quick, methods, amps, cert_ng) for sc in scenarios]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            records = list(ex.map(process_scenario, tasks))
            records.sort(key=lambda r: r["s"])
            summ, decomp, cert = aggregate(records, methods, amps)
            # serial sections reuse the SAME pool (they were the wall-clock bottleneck)
            print("[AG] pareto ...", flush=True); pareto = section_pareto(scenarios, N, quick, ex)
            print("[AG] budget ...", flush=True); budget = section_budget(scenarios, quick, ex)
            print("[AG] 2-D ...", flush=True); twod = section_2d(quick, ex)
    else:
        records = [process_scenario(t) for t in tasks]
        records.sort(key=lambda r: r["s"])
        summ, decomp, cert = aggregate(records, methods, amps)
        print("[AG] pareto ...", flush=True); pareto = section_pareto(scenarios, N, quick)
        print("[AG] budget ...", flush=True); budget = section_budget(scenarios, quick)
        print("[AG] 2-D ...", flush=True); twod = section_2d(quick)
    make_figure(summ, pareto, decomp, cert)

    out = {"config": {"N": N, "n_scenarios": nsc, "methods": methods,
                      "heldout_types": HELDOUT_TYPES, "cert_band": CERT_BAND,
                      "metric": "function-space a_{L2,T} worst-case on held-out"},
           "summary": summ, "pareto": pareto, "decomposition": decomp,
           "certificate": cert, "budget": budget, "twod": twod,
           "scenario_records": records,
           "note": "AliasGuard = Ds-optimal/LCMV-motivated design (credited, not claimed); "
                   "exact Ds baseline included; function-space metric; paired outer "
                   "scenarios; diverse held-out incl. out-of-target degradation; the "
                   "certificate is a post-hoc guarantee for ANY design."}
    save_json("aliasguard.json", out)
    ag, rj, ds = summ["aliasguard"]["heldout"], summ["random_jitter"]["heldout"], summ["ds_optimal"]["heldout"]
    print(f"[AG] HELD-OUT 'near' max a_L2: AliasGuard {ag['near']['mean']:.3f}"
          f"[{ag['near']['ci_lo']:.3f},{ag['near']['ci_hi']:.3f}] vs random {rj['near']['mean']:.3f} "
          f"vs exact-Ds {ds['near']['mean']:.3f}", flush=True)
    print("[AG] by type (AliasGuard):", {ht: round(ag[ht]['mean'], 3) for ht in HELDOUT_TYPES}, flush=True)
    print("[AG] by type (random):", {ht: round(rj[ht]['mean'], 3) for ht in HELDOUT_TYPES}, flush=True)
    print(f"[AG] certificate band-wide: AliasGuard {cert['aliasguard']['certified_mean']:.3f} "
          f"vs random {cert['random_jitter']['certified_mean']:.3f} vs exact-Ds {cert['ds_optimal']['certified_mean']:.3f}", flush=True)
    print(f"[AG] 2-D held-out: AliasGuard {twod['aliasguard']['mean']:.3f} vs random {twod['random']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
