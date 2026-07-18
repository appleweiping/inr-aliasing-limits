r"""Program 2 (SCOPING, honest negative-boundary result) -- certified k-space mask design and
the LIMIT of the aliasability theory on nonlinear/sparse reconstruction.

By Fourier duality AliasGuard designs a Cartesian phase-encode mask that provably MINIMIZES the
certified band-limited worst-case coherent aliasing
:math:`\mu_W=\max_{0<|\Delta|\le W}|\mathrm{PSF}(\Delta)|` -- a computable guarantee no other mask
provides.  This experiment reports that certificate AND the compressed-sensing (TV) reconstruction
quality for every mask, to delimit where the linear aliasability theory governs the downstream
task.

Honest finding (this is a SCOPING experiment, not a headline win): AliasGuard attains the lowest
certified :math:`\mu_W`, but CS-MRI reconstruction quality is governed by INCOHERENCE and low-
frequency energy capture, NOT by the worst-case aliasability -- so variable-density / random masks
(worst :math:`\mu_W`) win the recon.  The certified worst-case aliasability governs LINEAR
fixed-Fourier-feature reconstruction (see Program 1), not sparse CS-MRI; forcing a coherence-
optimal mask onto CS-MRI even hurts, because it undersamples the energy-rich k-space center.  We
report this boundary honestly: the Spearman between certified :math:`\mu_W` and CS recon error is
weak/negative, marking the edge of the theory's applicability.

numpy/scipy only (analytic phantoms, POCS-TV recon, PSNR/NMSE/SSIM).  Usage:
python experiments/run_mri.py [--quick]
"""
from __future__ import annotations

import sys

import numpy as np

from _util import save_json, savefig
from inralias import mri

import matplotlib.pyplot as plt

NY = 192
ACS = 0.08
RATES = (4, 8)
TV_LAM = 0.02
TV_ITERS = 80
MASKS = ["equispaced", "random", "vardensity", "lowdisc", "ds", "aliasguard"]
THREAT_AWARE = {"aliasguard", "ds"}


def _phantoms(n_struct):
    imgs = {"shepp": mri.shepp_logan(NY)}
    for s in range(n_struct):
        imgs[f"struct{s}"] = mri.structured_phantom(NY, seed=s)
    return imgs


def _build_mask(name, R, W, seed):
    if name == "equispaced":
        return mri.mask_equispaced(NY, R, ACS)
    if name == "random":
        return mri.mask_random(NY, R, ACS, seed)
    if name == "vardensity":
        return mri.mask_vardensity(NY, R, ACS, seed)
    if name == "lowdisc":
        return mri.mask_lowdisc(NY, R, ACS)
    if name == "ds":
        return mri.mask_ds(NY, R, ACS, W, seed)
    if name == "aliasguard":
        return mri.mask_aliasguard(NY, R, ACS, W, seed)
    raise ValueError(name)


def main():
    quick = "--quick" in sys.argv
    from scipy.stats import spearmanr
    n_struct = 2 if quick else 6
    imgs = _phantoms(n_struct)
    W = max(mri.object_half_extent(im) for im in imgs.values())     # widest object support

    out = {"config": {"ny": NY, "acs_frac": ACS, "rates": list(RATES), "W": int(W),
                      "tv_lam": TV_LAM, "tv_iters": TV_ITERS, "masks": MASKS,
                      "threat_aware": sorted(THREAT_AWARE), "n_phantoms": len(imgs)},
           "by_rate": {}}
    for R in RATES:
        masks = {nm: _build_mask(nm, R, W, seed=0) for nm in MASKS}
        cert = {nm: mri.peak_sidelobe(np.where(m)[0], NY, W) for nm, m in masks.items()}
        psnr = {nm: [] for nm in MASKS}
        ssim = {nm: [] for nm in MASKS}
        for im in imgs.values():
            for nm, m in masks.items():
                ku, m2 = mri.undersample(im, m)
                rec = mri.recon_tv(ku, m2, lam=TV_LAM, iters=TV_ITERS)
                psnr[nm].append(mri.psnr(rec, im))
                ssim[nm].append(mri.ssim(rec, im))
        summ = {nm: {"cert_muW": cert[nm],
                     "psnr_mean": float(np.mean(psnr[nm])), "psnr_std": float(np.std(psnr[nm], ddof=1)),
                     "ssim_mean": float(np.mean(ssim[nm])),
                     "n_lines": int(masks[nm].sum())} for nm in MASKS}
        # AliasGuard's certificate rank (1 = best/lowest muW) and recon rank
        cert_order = sorted(MASKS, key=lambda n: cert[n])
        psnr_order = sorted(MASKS, key=lambda n: -np.mean(psnr[n]))
        # does the certificate predict recon quality across masks?  (it should NOT, for CS)
        rho_cert_err, _ = spearmanr([cert[n] for n in MASKS],
                                    [-np.mean(psnr[n]) for n in MASKS])   # muW vs recon ERROR proxy
        out["by_rate"][str(R)] = {
            "summary": summ,
            "aliasguard_cert_rank": cert_order.index("aliasguard") + 1,
            "aliasguard_psnr_rank": psnr_order.index("aliasguard") + 1,
            "best_cert_mask": cert_order[0], "best_psnr_mask": psnr_order[0],
            "spearman_cert_vs_reconerr": None if not np.isfinite(rho_cert_err) else float(rho_cert_err),
        }
        print(f"[mri] R={R}: AliasGuard cert-rank={cert_order.index('aliasguard')+1}/6 "
              f"(mu_W={cert['aliasguard']:.3f}, best-cert={cert_order[0]}); "
              f"recon-rank={psnr_order.index('aliasguard')+1}/6 (best-recon={psnr_order[0]}); "
              f"Spearman(muW, recon-err)={rho_cert_err:.2f}", flush=True)

    out["note"] = ("SCOPING result (honest boundary): AliasGuard attains the lowest certified "
                   "worst-case coherent aliasing mu_W at every acceleration, but CS-MRI (TV) "
                   "reconstruction is governed by incoherence + low-frequency energy capture, so "
                   "variable-density/random masks (worst mu_W) win the recon and Spearman(mu_W, "
                   "recon-error) is weak.  The certified worst-case aliasability governs LINEAR "
                   "fixed-Fourier-feature reconstruction (Program 1), NOT sparse CS-MRI -- this "
                   "experiment delimits the theory's scope.  numpy/scipy only; analytic phantoms.")
    save_json("mri_scoping.json", out)

    # figure: certified mu_W vs recon PSNR per mask (R=4) -- shows they are NOT aligned
    try:
        R0 = str(RATES[0])
        s = out["by_rate"][R0]["summary"]
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for nm in MASKS:
            ax.scatter(s[nm]["cert_muW"], s[nm]["psnr_mean"], s=40)
            ax.annotate(nm, (s[nm]["cert_muW"], s[nm]["psnr_mean"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(r"certified worst-case aliasing $\mu_W$ (lower=better guarantee)")
        ax.set_ylabel("CS-TV recon PSNR (dB)")
        ax.set_title(f"R={RATES[0]}: certificate does NOT predict CS recon (scope boundary)")
        fig.tight_layout()
        savefig(fig, "mri_scoping")
        plt.close(fig)
    except Exception as e:
        print(f"[mri] figure skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
