r"""E4 -- do trained nonlinear networks reproduce the linear fold?  (empirical only)

The theorems cover the fixed-feature linear-coefficient model; this experiment measures
how far they extrapolate to *trained* networks.  Protocol:

* **Ablation attribution.**  For each run, two networks with identical initialization are
  trained on samples with and without the out-of-band tone; the spectrum of the
  difference of their reconstructions isolates where the trained net put the tone.
* **Principled predictions.**  (i) For the ``bandlimited`` architecture (trainable linear
  head over fixed integer cosine/sine features) the theory dictionary IS the model's
  frequency set -- exact correspondence.  (ii) For FF-MLP / SIREN, the prediction comes
  from the network's own **Jacobian (NTK) linearization at initialization**: the tone is
  pushed through kernel regression with the empirical NTK and its reconstruction
  spectrum is the predicted pattern.  No arbitrary reference band is used.
* **Seed hygiene.**  Sampling seed, feature-initialization seed, and weight/optimizer
  seed are separated and varied independently (20 sampling seeds; weight-seed stability
  control on a subset).
* **Controls.**  Label permutation (attribution must die), amplitude sweep, and
  weight-seed stability.
* **No hidden failures.**  Every run's (predicted, measured, correlation) triple is
  recorded; the summary reports the full distribution (exact-match rate, top-3 rate,
  correlation quantiles, bootstrap CIs) including failures.

Requires torch (GPU server).  Usage: python experiments/run_nonlinear.py [--quick]
"""
from __future__ import annotations

import sys

import numpy as np

from _util import save_json, savefig
from inralias.signals import nonuniform_times, evaluate
from inralias.inr import FourierFeatureMLP, SIREN, train_inr, torch_available

import matplotlib.pyplot as plt

N = 46
SIGMA = 0.02
EPOCHS = 6000
LR = 2e-4
T_REF = np.linspace(0, 1, 2000, endpoint=False)
N_SEEDS = 20
F_OUTS = (35.0, 50.0, 70.0)
IN_F = np.array([3.0, 7.0, -3.0, -7.0])
_ic = np.array([1.0, 0.7]) * np.exp(1j * np.array([0.3, 1.0]))
IN_C = np.concatenate([_ic, np.conj(_ic)])
FMAX_SPEC = 30            # attribution spectra inspected below this frequency


def _spec(x):
    X = np.abs(np.fft.rfft(x))
    return X / (X.max() + 1e-12)


def _make_model(arch, feature_seed, weight_seed):
    import torch

    torch.manual_seed(weight_seed)
    if arch == "ffmlp":
        return FourierFeatureMLP(n_features=96, scale=6.0, hidden=128, layers=2,
                                 seed=feature_seed, in_dim=1)
    if arch == "siren":
        return SIREN(hidden=128, layers=2, w0=12.0)
    if arch == "bandlimited":
        import torch.nn as nn

        B = 20

        class _BandLimited(nn.Module):
            """Trainable linear head over FIXED integer cos/sin features up to B."""

            def __init__(self):
                super().__init__()
                self.head = nn.Linear(2 * B + 1, 1, bias=False)

            def forward(self, x):
                ks = torch.arange(1, B + 1, dtype=x.dtype, device=x.device)
                ph = 2 * np.pi * x * ks
                feats = torch.cat([torch.ones_like(x), torch.cos(ph), torch.sin(ph)], -1)
                return self.head(feats)

        return _BandLimited()
    raise ValueError(arch)


def _train_predict(model, t, y, dev):
    import torch

    model, _ = train_inr(model, t, y, epochs=EPOCHS, lr=LR, device=dev,
                         weight_decay=1e-6)
    with torch.no_grad():
        return model(torch.tensor(T_REF.reshape(-1, 1), dtype=torch.float32,
                                  device=dev)).cpu().numpy().ravel(), model


def _ntk_prediction(model, t, tone_samples, dev, lam=1e-6):
    """Predicted tone reconstruction via the empirical NTK at initialization.

    Kernel regression of the tone samples with K = J J^T (J = param Jacobian at init):
    f_pred(x) = k(x, t) K^{-1} tone_samples.  Returns the dense reconstruction.
    """
    import torch

    tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=dev)
    te = torch.tensor(T_REF.reshape(-1, 1), dtype=torch.float32, device=dev)
    params = [p for p in model.parameters() if p.requires_grad]

    def jac(inputs):
        rows = []
        for i in range(inputs.shape[0]):
            model.zero_grad(set_to_none=True)
            out = model(inputs[i:i + 1]).squeeze()
            grads = torch.autograd.grad(out, params, retain_graph=False)
            rows.append(torch.cat([g.reshape(-1) for g in grads]))
        return torch.stack(rows)

    J_t = jac(tt)                                   # (N, P)
    K = J_t @ J_t.T
    K = K + lam * torch.eye(K.shape[0], device=dev) * K.diagonal().mean()
    alpha = torch.linalg.solve(K, torch.tensor(tone_samples, dtype=torch.float32,
                                               device=dev))
    # evaluate in batches to bound memory
    preds = []
    bs = 200
    for i in range(0, te.shape[0], bs):
        J_e = jac(te[i:i + bs])
        preds.append((J_e @ J_t.T @ alpha).detach().cpu())
    import torch as _t

    return _t.cat(preds).numpy().ravel()


def _ntk_gram(model, t, dev):
    """Empirical NTK Gram K = J J^T on the sample points (J = param Jacobian).

    Used to MEASURE how far the tangent kernel moves from initialization to after training
    -- the evidence needed before claiming a trained network departs from its init-NTK."""
    import torch

    tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=dev)
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(tt.shape[0]):
        model.zero_grad(set_to_none=True)
        out = model(tt[i:i + 1]).squeeze()
        grads = torch.autograd.grad(out, params, retain_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    J = torch.stack(rows)
    return (J @ J.T).detach().cpu().numpy()


def _ntk_drift(K0, K1):
    """Init-vs-trained NTK drift: relative Frobenius change + top-eigenvector alignment.

    Small drift => the linearization (init NTK) is a good model of training => an init-NTK
    prediction is expected to match.  Large drift => the network left its init tangent
    space and a mismatch is expected (this is what distinguishes SIREN from FF-MLP here)."""
    f0 = float(np.linalg.norm(K0))
    rel = float(np.linalg.norm(K1 - K0) / (f0 + 1e-12))
    w0, V0 = np.linalg.eigh(K0)
    w1, V1 = np.linalg.eigh(K1)
    cos = abs(float(V0[:, -1] @ V1[:, -1]))          # top-eigvec alignment (1 = unchanged)
    return {"ntk_rel_drift": rel, "ntk_top_evec_cos": cos}


def _fold_metrics(diff_recon, pred_recon):
    """Measured vs predicted attribution spectra below FMAX_SPEC.

    The DC bin is excluded from fold extraction: network bias paths give both the
    measured and predicted reconstructions a common offset component that is not a
    frequency fold (the full spectra, DC included, enter the pattern correlation).
    """
    d = _spec(diff_recon)[: FMAX_SPEC + 1]
    p = _spec(pred_recon)[: FMAX_SPEC + 1]
    meas = 1 + int(np.argmax(d[1:]))
    pred = 1 + int(np.argmax(p[1:]))
    corr = float(np.corrcoef(d, p)[0, 1])
    top3 = bool(meas in (1 + np.argsort(p[1:])[::-1][:3]))
    return {"measured_fold": meas, "predicted_fold": pred,
            "exact_match": bool(meas == pred), "top3_match": top3,
            "pattern_corr": corr}


def one_run(arch, f_out, samp_seed, weight_seed, feature_seed=None, amp=0.8,
            permute=False, measure_drift=False, dev="cuda"):
    """One ablation-attribution run.  ``samp_seed`` (design t + noise), ``feature_seed``
    (Fourier-feature draw) and ``weight_seed`` (weight init / optimizer) are INDEPENDENT
    inputs -- the caller crosses them instead of binding all three to one counter."""
    if feature_seed is None:
        feature_seed = 100 + samp_seed
    rng = np.random.default_rng(samp_seed)
    t = nonuniform_times(N, rng, "jitter")
    noise = rng.normal(0, SIGMA, N)
    y_in = evaluate(IN_F, IN_C, t, real=True) + noise
    oc = amp * np.exp(1j * 0.2)
    tone = evaluate(np.array([f_out, -f_out]), np.array([oc, np.conj(oc)]), t, real=True)
    y_all = y_in + tone
    if permute:
        y_all = rng.permutation(y_all)

    m1 = _make_model(arch, feature_seed, weight_seed)
    f_with, _ = _train_predict(m1, t, y_all, dev)
    m0 = _make_model(arch, feature_seed, weight_seed)
    f_wo, m0_trained = _train_predict(m0, t, y_in, dev)
    diff = f_with - f_wo

    m_ref = _make_model(arch, feature_seed, weight_seed)
    m_ref = m_ref.to(dev) if hasattr(m_ref, "to") else m_ref
    drift = {}
    if measure_drift:
        # NTK Gram at init (m_ref) vs after training (m0_trained) -- same init, so this is a
        # clean before/after comparison of the tangent kernel on the same points.
        K_init = _ntk_gram(m_ref, t, dev)
        m0_trained = m0_trained.to(dev) if hasattr(m0_trained, "to") else m0_trained
        K_trained = _ntk_gram(m0_trained, t, dev)
        drift = _ntk_drift(K_init, K_trained)
    pred_recon = _ntk_prediction(m_ref, t, tone, dev)

    res = _fold_metrics(diff, pred_recon)
    res.update({"arch": arch, "f_out": f_out, "samp_seed": samp_seed,
                "weight_seed": weight_seed, "feature_seed": int(feature_seed),
                "amp": amp, "permute": permute,
                "diff_energy": float(np.mean(diff**2)),
                "linear_response": float(np.mean(diff**2)) / (amp**2 + 1e-12), **drift})
    return res, diff, pred_recon, t


def _boot_ci(bools, rng, n=2000):
    v = np.asarray(bools, float)
    stats = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)]
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def _cluster_boot_ci(vals, clusters, rng, n=2000):
    """Cluster bootstrap: resample SAMPLING SCENARIOS (each shares one design t), not
    individual runs -- runs from the same sampling seed are correlated, so the honest
    replication unit is the scenario."""
    vals = np.asarray(vals, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    stats = []
    for _ in range(n):
        pick = rng.choice(uniq, uniq.size, replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        stats.append(vals[sel].mean())
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def make_figure(runs, controls, n_seeds, dev):
    """One representative run (spectra) + correlation distributions per architecture."""
    perm_corr = [r["pattern_corr"] for r in controls
                 if r.get("control") == "label_permutation"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
    r, diff, pred, _t = one_run("ffmlp", 50.0, 0, 200, dev=dev)
    fr = np.arange(FMAX_SPEC + 1)
    ax1.plot(fr, _spec(diff)[: FMAX_SPEC + 1], "C3-", lw=1.5,
             label="measured (ablation)")
    ax1.plot(fr, _spec(pred)[: FMAX_SPEC + 1], "C0--", lw=1.4,
             label=f"NTK prediction (corr {r['pattern_corr']:.2f})")
    ax1.set_xlabel("frequency"); ax1.set_ylabel("|X| (norm)")
    ax1.set_title("FF-MLP: measured vs NTK")
    ax1.legend(fontsize=9)
    data = [[q["pattern_corr"] for q in runs if q["arch"] == a]
            for a in ("ffmlp", "siren", "bandlimited")]
    ax2.boxplot(data, tick_labels=["FF-MLP", "SIREN", "band-limited"], whis=(5, 95))
    if perm_corr:
        ax2.axhline(float(np.mean(perm_corr)), color="0.5", ls=":",
                    label="label-permutation control")
        ax2.legend(fontsize=9)
    ax2.set_ylabel("pattern correlation")
    ax2.set_title(f"correlation over {n_seeds} seeds")
    savefig(fig, "nonlinear_folds.png")


def main():
    if not torch_available():
        print("[nonlinear] torch unavailable -- run on the GPU server", flush=True)
        return
    if "--figure-only" in sys.argv:
        import json
        from _util import RESULTS

        d = json.loads((RESULTS / "nonlinear.json").read_text())
        import torch

        make_figure(d["runs"], d["controls"], d["n_seeds"],
                    "cuda" if torch.cuda.is_available() else "cpu")
        print("[nl] figure regenerated", flush=True)
        return
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    quick = "--quick" in sys.argv
    n_seeds = 4 if quick else N_SEEDS

    # Independent seed pools: sampling, feature-init, and weight/optimizer seeds are drawn
    # from separate streams so they are NOT one-to-one bound to a single counter (P1-A).
    seedgen = np.random.default_rng(20260717)
    feat_pool = seedgen.integers(0, 1_000_000, n_seeds).tolist()
    wt_pool = seedgen.integers(0, 1_000_000, n_seeds).tolist()

    runs = []
    # main grid: ffmlp x f_out x sampling scenario (each scenario an independent triple)
    for f_out in F_OUTS:
        for s in range(n_seeds):
            r, *_ = one_run("ffmlp", f_out, s, wt_pool[s], feature_seed=feat_pool[s],
                            measure_drift=True, dev=dev)
            runs.append(r)
            print(f"[nl] ffmlp f={f_out} s={s}: meas={r['measured_fold']} "
                  f"pred={r['predicted_fold']} corr={r['pattern_corr']:.2f} "
                  f"drift={r.get('ntk_rel_drift', float('nan')):.2f}", flush=True)
    # siren + bandlimited at f_out = 50
    for arch in ("siren", "bandlimited"):
        for s in range(n_seeds):
            r, *_ = one_run(arch, 50.0, s, wt_pool[s], feature_seed=feat_pool[s],
                            measure_drift=True, dev=dev)
            runs.append(r)
            print(f"[nl] {arch} s={s}: meas={r['measured_fold']} pred={r['predicted_fold']} "
                  f"corr={r['pattern_corr']:.2f} drift={r.get('ntk_rel_drift', float('nan')):.2f}",
                  flush=True)

    # controls
    controls = []
    step = max(1, n_seeds // 4)
    for s in range(0, n_seeds, step):
        r, *_ = one_run("ffmlp", 50.0, s, wt_pool[s], feature_seed=feat_pool[s],
                        permute=True, dev=dev)
        r["control"] = "label_permutation"; controls.append(r)
    # optimizer/weight-seed stability at FIXED sampling scenario (isolates weight variance,
    # decoupled from sampling): vary only the weight seed with s and features held fixed.
    for j in range(min(5, n_seeds)):
        r, *_ = one_run("ffmlp", 50.0, 0, 900 + j, feature_seed=feat_pool[0], dev=dev)
        r["control"] = "weight_seed_stability"; controls.append(r)

    # Amplitude sweep -> 0 (infinitesimal-influence test): the two-net difference is a clean
    # linear attribution only in the small-amplitude limit.  Report pattern_corr and the
    # linear-response energy (diff_energy / amp^2, which should stabilize as amp -> 0).
    amp_sweep = []
    AMPS = (0.05, 0.1, 0.2, 0.4, 0.8)
    for s in range(0, n_seeds, step):
        for amp in AMPS:
            r, *_ = one_run("ffmlp", 50.0, s, wt_pool[s], feature_seed=feat_pool[s],
                            amp=amp, dev=dev)
            amp_sweep.append({"amp": amp, "samp_seed": s,
                              "pattern_corr": r["pattern_corr"],
                              "exact_match": r["exact_match"],
                              "linear_response": r["linear_response"]})
            print(f"[nl] amp-sweep s={s} amp={amp}: corr={r['pattern_corr']:.2f} "
                  f"lin_resp={r['linear_response']:.3f}", flush=True)

    rng = np.random.default_rng(0)
    summary = {}
    for arch in ("ffmlp", "siren", "bandlimited"):
        sub = [r for r in runs if r["arch"] == arch]
        if not sub:
            continue
        ex = [r["exact_match"] for r in sub]
        t3 = [r["top3_match"] for r in sub]
        co = [r["pattern_corr"] for r in sub]
        clu = [r["samp_seed"] for r in sub]           # cluster = sampling scenario
        drifts = [r["ntk_rel_drift"] for r in sub if "ntk_rel_drift" in r]
        summary[arch] = {
            "n_runs": len(sub),
            "exact_match_rate": float(np.mean(ex)),
            "exact_match_ci95_cluster": _cluster_boot_ci(ex, clu, rng),
            "top3_match_rate": float(np.mean(t3)),
            "top3_match_ci95_cluster": _cluster_boot_ci(t3, clu, rng),
            "pattern_corr_median": float(np.median(co)),
            "pattern_corr_q10_q90": [float(np.quantile(co, 0.1)),
                                     float(np.quantile(co, 0.9))],
            "ntk_rel_drift_median": float(np.median(drifts)) if drifts else None,
            "ntk_rel_drift_q10_q90": ([float(np.quantile(drifts, 0.1)),
                                       float(np.quantile(drifts, 0.9))] if drifts else None),
        }
    perm_corr = [r["pattern_corr"] for r in controls
                 if r.get("control") == "label_permutation"]

    make_figure(runs, controls, n_seeds, dev)

    # Honest framing of the SIREN result, conditioned on the MEASURED drift.
    ff_d = summary.get("ffmlp", {}).get("ntk_rel_drift_median")
    si_d = summary.get("siren", {}).get("ntk_rel_drift_median")
    siren_note = ("insufficient data" if (ff_d is None or si_d is None) else
                  (f"trained SIREN NTK drifts {si_d:.2f} (median rel.) vs FF-MLP {ff_d:.2f}; "
                   "where drift is large the init-NTK prediction is not expected to hold, so "
                   "we report only that trained SIREN responses are inconsistent with the "
                   "initialization-NTK prediction -- not that they 'leave' a fixed kernel."))

    out = {"N": N, "sigma": SIGMA, "epochs": EPOCHS, "n_seeds": n_seeds,
           "f_outs": list(F_OUTS), "device": dev, "quick": quick,
           "runs": runs, "controls": controls, "amp_sweep": amp_sweep, "summary": summary,
           "siren_framing": siren_note,
           "note": "empirical extrapolation only; predictions from each network's own "
                   "init-NTK linearization (bandlimited: exact dictionary); NTK drift "
                   "measured init-vs-trained; seeds (sampling/feature/weight) crossed "
                   "independently; CIs cluster-bootstrapped by sampling scenario; amplitude "
                   "swept to 0 for the infinitesimal-influence limit; all runs recorded "
                   "including failures"}
    save_json("nonlinear.json", out)
    print("[nl] summary:", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                for kk, vv in v.items() if "ci" not in kk and vv is not None}
                            for k, v in summary.items()}, flush=True)
    print("[nl] siren_framing:", siren_note, flush=True)


if __name__ == "__main__":
    main()
