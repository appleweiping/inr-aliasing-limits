r"""Nonlinear-INR persistence: silent aliasing happens for a *trained* nonlinear INR too.

The linear theory (Thm 2) predicts that a signal component beyond the INR's representable
band folds onto an in-band frequency while the samples are still fit well.  We reproduce this
with a *trained* nonlinear Fourier-feature INR: a synthetic signal has two in-band tones plus
one tone far beyond the (shallow, small-scale) network's representable hull, sampled below the
high tone's Nyquist rate.  The trained INR fits the samples (small residual) yet its recovered
spectrum shows a spurious peak -- the high tone folded in -- matching what the linear
fixed-feature INR does on the same data.

We also confirm the complementary *achievability* transfer: on the real CO2 signal a trained
INR whose band covers the signal recovers it, agreeing with the linear theory.

Requires torch (GPU server).
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.signals import nonuniform_times, evaluate, lowpass_dictionary
from inralias.inr import FixedFeatureINR, FourierFeatureMLP, train_inr, torch_available
from run_real_signal import load_signal, essential_bandwidth

import matplotlib.pyplot as plt


def _spec(x):
    X = np.abs(np.fft.rfft(x)); return X / (X.max() + 1e-12)


def silent_aliasing_nonlinear(seed=0, make_fig=True):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    # in-band tones + one tone far beyond a shallow/small-scale INR's representable hull
    in_f = np.array([3.0, 7.0, -3.0, -7.0])
    ic = np.array([1.0, 0.7]) * np.exp(1j * np.array([0.3, 1.0]))
    in_c = np.concatenate([ic, np.conj(ic)])
    f_out = 50.0
    out_c = np.concatenate([[0.8 * np.exp(1j * 0.2)], [np.conj(0.8 * np.exp(1j * 0.2))]])
    allf = np.concatenate([in_f, [f_out, -f_out]])
    allc = np.concatenate([in_c, out_c])

    # undersample the high tone (N well below its Nyquist of 2*50=100)
    N = 46
    t = nonuniform_times(N, rng, "jitter")
    noise = rng.normal(0, 0.02, N)
    y_inband = evaluate(in_f, in_c, t, real=True) + noise
    y = y_inband + evaluate(np.array([f_out, -f_out]), out_c, t, real=True)
    t_ref = np.linspace(0, 1, 2000, endpoint=False)
    f_true = evaluate(allf, allc, t_ref, real=True)

    # trained nonlinear INR: shallow, small feature scale -> representable hull well below f_out
    def _train(target):
        torch.manual_seed(seed)
        model = FourierFeatureMLP(n_features=96, scale=6.0, hidden=128, layers=2, seed=seed, in_dim=1)
        model, _ = train_inr(model, t, target, epochs=8000, lr=2e-4, device=dev, weight_decay=1e-6)
        with torch.no_grad():
            f_ref = model(torch.tensor(t_ref.reshape(-1, 1), dtype=torch.float32, device=dev)).cpu().numpy().ravel()
            y_fit = model(torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=dev)).cpu().numpy().ravel()
        return f_ref, y_fit

    f_nn, y_nn = _train(y)
    sample_rmse = float(np.sqrt(np.mean((y_nn - y) ** 2)))
    # controlled ablation: identical net/seed trained WITHOUT the out-of-band tone.  The
    # difference of the two reconstructions isolates where the trained net puts the tone's
    # energy, removing harmonic-distortion artifacts common to both runs.
    f_nn0, _ = _train(y_inband)
    diff_spec = np.abs(np.fft.rfft(f_nn - f_nn0))
    diff_spec = diff_spec / (diff_spec.max() + 1e-12)

    # linear fixed-feature INR on a lowpass band that also excludes f_out
    Lam = lowpass_dictionary(20)
    lin = FixedFeatureINR(Lam, real=True).fit(t, y, ridge=1e-6)
    f_lin = lin.predict(t_ref)

    # linear-theory prediction for the REAL (Hermitian-pair) tone: the bias pattern of
    # {+f_out, -f_out} on the lowpass band under these samples; its argmax is the fold target
    from inralias.limits import aliasing_bias
    dc = aliasing_bias(Lam, t, np.array([f_out, -f_out]), out_c)
    lin_pattern = np.zeros(21)
    for w, v in zip(Lam, np.abs(dc)):
        if w >= 0:
            lin_pattern[int(w)] = max(lin_pattern[int(w)], v)
    lin_pattern = lin_pattern / (lin_pattern.max() + 1e-12)
    predicted_fold = int(np.argmax(lin_pattern))

    measured_fold = int(np.argmax(diff_spec[:31]))
    corr = float(np.corrcoef(diff_spec[:21], lin_pattern)[0, 1])

    if not make_fig:
        return {"seed": seed, "sample_rmse": sample_rmse,
                "predicted_fold": predicted_fold, "measured_fold": measured_fold,
                "pattern_corr": corr}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax1.plot(t_ref, f_true, color="0.6", lw=1.0, label="true (has $f_{out}$=50)")
    ax1.plot(t_ref, f_nn, color="C3", lw=1.0, ls="--", label=f"trained INR (sample rmse {sample_rmse:.3f})")
    ax1.plot(t, y, "k.", ms=4)
    ax1.set_xlabel("t"); ax1.set_ylabel("amp"); ax1.set_title("nonlinear INR: fits samples"); ax1.legend(fontsize=7)
    fr = np.arange(diff_spec.size)
    ax2.plot(fr, _spec(f_true), color="0.6", lw=1.0, label="true")
    ax2.plot(fr, diff_spec, color="C3", lw=1.2, label="tone-attributable (trained, ablation)")
    ax2.plot(np.arange(21), lin_pattern, color="C0", lw=1.0, ls="--",
             label=f"linear-theory bias (corr {corr:.2f})")
    ax2.axvline(predicted_fold, color="C1", ls=":", lw=1.2,
                label=f"predicted fold={predicted_fold}")
    ax2.set_xlim(0, 30); ax2.set_xlabel("frequency"); ax2.set_ylabel("|X| (norm)")
    ax2.set_title("where the trained net puts $f_{out}$=50"); ax2.legend(fontsize=7)
    savefig(fig, "nonlinear_silent_aliasing.png")

    out = {"f_out": f_out, "N": N, "sample_rmse": sample_rmse,
           "predicted_fold": predicted_fold, "measured_fold": measured_fold,
           "pattern_corr": corr, "device": dev,
           "note": "trained nonlinear INR fits samples but folds the out-of-hull tone "
                   "(silent aliasing); measured by ablation (identical net trained with vs "
                   "without the tone, difference spectrum) against the Hermitian-pair "
                   "linear-theory bias pattern"}
    print(f"[nonlinear] sample_rmse={sample_rmse:.3f} predicted_fold={predicted_fold} "
          f"measured_fold={measured_fold} pattern_corr={corr:.3f}", flush=True)
    return out


def achievability_transfer(name="co2", seed=0):
    """Trained INR whose band covers the real signal recovers it (agrees with linear theory)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    x = load_signal(name); M = x.size; t_ref = np.arange(M) / M
    B = essential_bandwidth(x)
    N = int(min(M, 0.6 * M)); idx = np.sort(rng.choice(M, N, replace=False))
    t, y = t_ref[idx], x[idx] + rng.normal(0, np.sqrt(np.mean(x**2)) / 31.6, N)
    torch.manual_seed(seed)
    model = FourierFeatureMLP(n_features=256, scale=B / 2.5, hidden=256, layers=3, seed=seed, in_dim=1)
    model, _ = train_inr(model, t, y, epochs=6000, lr=2e-4, device=dev, weight_decay=1e-5)
    with torch.no_grad():
        f_nn = model(torch.tensor(t_ref.reshape(-1, 1), dtype=torch.float32, device=dev)).cpu().numpy().ravel()
    nn_rmse = float(np.sqrt(np.mean((f_nn - x) ** 2)))
    lin = FixedFeatureINR(lowpass_dictionary(int(1.1 * B)), real=True).fit(t, y, ridge=1e-6)
    lin_rmse = float(np.sqrt(np.mean((lin.predict(t_ref) - x) ** 2)))
    print(f"[nonlinear:{name}] achievability transfer  nonlinear rmse={nn_rmse:.3f}  linear rmse={lin_rmse:.3f}", flush=True)
    return {"name": name, "nonlinear_rmse": nn_rmse, "linear_rmse": lin_rmse, "B_sig": B}


def main():
    if not torch_available():
        print("[nonlinear] torch unavailable -- run on server", flush=True)
        return
    # seed 0 draws the figure; two extra seeds check fold-target and residual stability
    main_run = silent_aliasing_nonlinear(seed=0)
    extra = [silent_aliasing_nonlinear(seed=s, make_fig=False) for s in (1, 2)]
    out = {"silent_aliasing": main_run,
           "silent_aliasing_extra_seeds": extra,
           "achievability": achievability_transfer("co2")}
    save_json("nonlinear.json", out)


if __name__ == "__main__":
    main()
