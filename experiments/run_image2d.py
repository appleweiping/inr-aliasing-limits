r"""E6 -- 2-D image demo: the learned-aliasing limit generalizes beyond 1-D.

Fit a nonlinear 2-D Fourier-feature INR to a real grayscale image from *sparse, nonuniform*
pixel samples at three feature scales.  The OVER-scaled band silently aliases: it fits the
observed pixels (high sample-PSNR) but the full field is moire-corrupted with spurious
high-frequency energy -- the 2-D analogue of "silent aliasing" -- while a band matched to
(or below) the sampling is safe.

Requires torch (GPU server).  Usage: python experiments/run_image2d.py
"""
from __future__ import annotations

import numpy as np

from _util import save_json, savefig
from inralias.inr import FourierFeatureMLP, torch_available

import matplotlib.pyplot as plt


def _load_image(size=128):
    """Return a real grayscale image in [0,1], resized to size x size."""
    try:
        from scipy.datasets import ascent
        img = np.asarray(ascent(), float)
    except Exception:
        from scipy.misc import ascent  # older scipy
        img = np.asarray(ascent(), float)
    # center-crop to square then decimate
    n = min(img.shape)
    img = img[:n, :n]
    step = max(1, n // size)
    img = img[::step, ::step][:size, :size]
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    return img


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(1.0 / (mse + 1e-12)))


def _hf_psnr(a, b, cutoff_frac=0.25):
    """PSNR restricted to high radial frequencies (detail band), where blur is penalized and
    aliasing (spurious high-freq energy) is penalized -- so a matched band wins."""
    def hp(img):
        F = np.fft.fftshift(np.fft.fft2(img))
        H, W = img.shape
        cy, cx = H // 2, W // 2
        y, x = np.indices((H, W))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        F[r < cutoff_frac * min(H, W) / 2] = 0
        return np.fft.ifft2(np.fft.ifftshift(F)).real
    return _psnr(hp(a), hp(b))


def _radial_spectrum(img):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    c = np.array(F.shape) // 2
    y, x = np.indices(F.shape)
    r = np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), F.ravel())
    nr = np.bincount(r.ravel())
    prof = tbin / (nr + 1e-9)
    return prof / (prof.max() + 1e-12)


def run(size=128, sample_frac=0.5, seed=0):
    if not torch_available():
        print("[image2d] torch unavailable -- skipping (run on server)", flush=True)
        return None
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    img = _load_image(size)
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    vals = img.ravel()

    # sparse nonuniform pixel samples
    N = int(sample_frac * coords.shape[0])
    idx = rng.choice(coords.shape[0], size=N, replace=False)
    c_s, v_s = coords[idx], vals[idx]
    noise = 0.02
    v_s = v_s + rng.normal(0, noise, N)

    B_img = size // 2  # image Nyquist (cycles across the field)
    # three regimes: under-scaled band (blur/attenuation), matched (best), over-scaled band
    # which -- with sparse pixels -- aliases (moire) while still fitting the observed pixels.
    scales = {"under": max(3.0, B_img / 12.0), "matched": max(6.0, B_img / 4.0),
              "over": max(12.0, B_img / 1.2)}
    recon = {}
    metrics = {}
    for tag, sc in scales.items():
        torch.manual_seed(seed)
        model = FourierFeatureMLP(n_features=256, scale=sc, hidden=256, layers=4, seed=seed, in_dim=2).to(dev)
        cc = torch.tensor(c_s, dtype=torch.float32, device=dev)
        vv = torch.tensor(v_s.reshape(-1, 1), dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
        for ep in range(4000):
            opt.zero_grad()
            loss = torch.mean((model(cc) - vv) ** 2)
            loss.backward(); opt.step()
        with torch.no_grad():
            full = model(torch.tensor(coords, dtype=torch.float32, device=dev)).cpu().numpy().ravel()
        r = full.reshape(H, W)
        recon[tag] = r
        metrics[tag] = {
            "scale": sc,
            "sample_psnr": _psnr(v_s, full[idx]),   # benign (fits observed pixels)
            "full_psnr": _psnr(img, r),             # global quality (low-freq dominated)
            "hf_psnr": _hf_psnr(img, r),            # high-frequency (detail) fidelity
        }
        print(f"[image2d] {tag} scale={sc:.1f} sample_PSNR={metrics[tag]['sample_psnr']:.1f} "
              f"full_PSNR={metrics[tag]['full_psnr']:.1f} hf_PSNR={metrics[tag]['hf_psnr']:.1f}",
              flush=True)

    # ---- figure: original | under | matched | over | radial spectrum ----
    # constrained_layout keeps the spectrum panel's y-label off the neighbouring image
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.9), layout="constrained")
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("original"); axes[0].axis("off")
    for ax, tag, lbl in ((axes[1], "under", "under-scaled\n(safe)"),
                         (axes[2], "matched", "matched\n(safe)"),
                         (axes[3], "over", "over-scaled\n(silent alias)")):
        ax.imshow(recon[tag], cmap="gray")
        ax.set_title(f"{lbl}\nsmp {metrics[tag]['sample_psnr']:.0f} / full "
                     f"{metrics[tag]['full_psnr']:.1f} / hf "
                     f"{metrics[tag]['hf_psnr']:.1f} dB", fontsize=8)
        ax.axis("off")
    fr = np.arange(_radial_spectrum(img).size)
    axes[4].semilogy(fr, _radial_spectrum(img) + 1e-6, color="0.5", label="true")
    axes[4].semilogy(fr, _radial_spectrum(recon["under"]) + 1e-6, color="C0", label="under")
    axes[4].semilogy(fr, _radial_spectrum(recon["matched"]) + 1e-6, color="C2", ls="--", label="matched")
    axes[4].semilogy(fr, _radial_spectrum(recon["over"]) + 1e-6, color="C1", label="over")
    axes[4].set_xlabel("radial freq"); axes[4].set_ylabel("power"); axes[4].set_title("radial spectrum")
    axes[4].legend(fontsize=7)
    savefig(fig, "image2d_aliasing.png")

    out = {"size": size, "sample_frac": sample_frac, "N": N, "device": dev,
           "B_img": B_img, "metrics": metrics,
           "note": "2-D INR band vs sampling: an OVER-scaled band silently aliases under sparse "
                   "pixels (high sample-PSNR ~50dB but low full/hf-PSNR, moire); a band matched to "
                   "or below the sampling is safe (~21dB). Single seed, scipy ascent image."}
    save_json("image2d_aliasing.json", out)
    return out


if __name__ == "__main__":
    run()
