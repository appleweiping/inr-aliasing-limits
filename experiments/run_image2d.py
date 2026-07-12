r"""E5 -- two-dimensional structured aliasing: exact theory + trained-network study.

Part A (CPU, exact).  The 2-D fold relation: for samples on a rate-(Q,Q) grid, the sample
atom of an integer frequency VECTOR depends only on (nu mod Q) componentwise, so an
out-of-band tone with nu = omega (mod Q) is exactly indistinguishable from the in-band
atom omega.  We plant such a tone, fit the linear 2-D Fourier model on a random grid
subset, and show the recovered coefficient map: the energy lands exactly on the predicted
frequency vector.

Part B (GPU, empirical extrapolation).  Trained 2-D Fourier-feature MLPs on real images:

* **lattice masks** (every-2nd-pixel sublattices) create a *coherent* 2-D fold: the
  reconstruction's DFT should show replicas of the image spectrum at the predicted alias
  shifts (Q/2, 0), (0, Q/2), (Q/2, Q/2);
* **random masks** of the same density have no coherent fold: no predicted replicas.

We quantify replica energy at the predicted positions (vs a control high-frequency
region), report clean-sample / noisy-target / held-out-pixel / full-field PSNR
separately, and include ridge and early-stopping baselines on the over-scaled model
(architectures are parameter-matched across scales by construction -- only the feature
scale differs).  3 images x (4 lattice offsets + 8 random masks) x 2 scales + baselines.

Usage: python experiments/run_image2d.py [--quick]   (Part B requires torch)
"""
from __future__ import annotations

import sys

import numpy as np

from _util import save_json, savefig
from inralias.inr import FourierFeatureMLP, torch_available

import matplotlib.pyplot as plt

SIZE = 128
NOISE = 0.02
EPOCHS = 4000
LR = 2e-4


# --------------------------------------------------------------------------------------
# Part A: exact 2-D fold on the linear model (CPU)
# --------------------------------------------------------------------------------------
def _synthesis_2d(freq_vecs, pts):
    """Phi[j, k] = exp(i 2 pi <nu_k, p_j>) for 2-D frequency vectors and points."""
    return np.exp(1j * 2 * np.pi * (pts @ freq_vecs.T))


def part_a_linear(rng):
    Q, B, N = 32, 6, 600
    ks = np.arange(-B, B + 1)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    Lam = np.stack([KX.ravel(), KY.ravel()], axis=1).astype(float)      # (169, 2)
    sites = np.stack(np.meshgrid(np.arange(Q), np.arange(Q), indexing="ij"),
                     axis=-1).reshape(-1, 2)
    sel = rng.choice(sites.shape[0], size=N, replace=False)
    pts = sites[sel] / Q                                                # grid subset

    nu_out = np.array([27.0, 3.0])            # 27 = -5 (mod 32): predicted fold (-5, 3)
    pred_fold = np.array([-5.0, 3.0])         # Hermitian partner folds to (5, -3)
    in_vecs = np.array([[2.0, 1.0], [-2.0, -1.0], [1.0, 4.0], [-1.0, -4.0]])
    in_c = np.array([1.0, 1.0, 0.5, 0.5]).astype(complex)
    a = 0.8
    Phi = _synthesis_2d(Lam, pts)
    y = (_synthesis_2d(in_vecs, pts) @ in_c
         + a * (_synthesis_2d(nu_out[None], pts)[:, 0]
                + _synthesis_2d(-nu_out[None], pts)[:, 0])).real
    c, *_ = np.linalg.lstsq(Phi, y.astype(complex), rcond=None)
    Cmap = np.abs(c).reshape(2 * B + 1, 2 * B + 1)
    # measured fold: largest recovered coefficient OUTSIDE the true in-band vectors
    search = Cmap * (1.0 - _mask_true(KX, KY, in_vecs))
    k_meas = np.unravel_index(int(np.argmax(search)), Cmap.shape)
    meas_fold = np.array([ks[k_meas[0]], ks[k_meas[1]]], float)
    exact = bool(np.allclose(meas_fold, pred_fold) or np.allclose(meas_fold, -pred_fold))
    resid = float(np.linalg.norm(y - (Phi @ c).real) / np.sqrt(N))
    return {"Q": Q, "B": B, "N": N, "nu_out": nu_out.tolist(),
            "predicted_fold": pred_fold.tolist(), "measured_fold": meas_fold.tolist(),
            "exact": exact, "sample_residual": resid}, Cmap, ks, in_vecs, pred_fold


def _mask_true(KX, KY, in_vecs):
    m = np.zeros_like(KX, dtype=float)
    for v in in_vecs:
        m[(KX == int(v[0])) & (KY == int(v[1]))] = 1.0
    return m


# --------------------------------------------------------------------------------------
# Part B: trained 2-D networks, lattice vs random masks (GPU)
# --------------------------------------------------------------------------------------
def _load_images(size=SIZE):
    from scipy.datasets import ascent, face

    imgs = {}
    a = np.asarray(ascent(), float)
    imgs["ascent"] = a[::a.shape[0] // size, ::a.shape[1] // size][:size, :size]
    f = np.asarray(face(gray=True), float)
    k = min(f.shape) // size
    f = f[:k * size:k, :k * size:k][:size, :size]
    imgs["face"] = f
    # deterministic synthetic multi-scale texture (no dataset dependency)
    yy, xx = np.mgrid[0:size, 0:size] / size
    imgs["synthetic"] = (np.sin(2 * np.pi * (6 * xx + 2 * yy))
                         + 0.7 * np.sin(2 * np.pi * (1 * xx - 9 * yy))
                         + 0.5 * np.sin(2 * np.pi * (15 * xx + 12 * yy))
                         + 0.3 * (xx - 0.5) ** 2 * 8)
    for k_ in imgs:
        v = imgs[k_]
        imgs[k_] = (v - v.min()) / (np.ptp(v) + 1e-9)
    return imgs


def _psnr(a, b):
    mse = np.mean((np.asarray(a) - np.asarray(b)) ** 2)
    return float(10 * np.log10(1.0 / (mse + 1e-12)))


def _hf_psnr(a, b, cutoff_frac=0.25):
    def hp(img):
        F = np.fft.fftshift(np.fft.fft2(img))
        H, W = img.shape
        y, x = np.indices((H, W))
        r = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
        F[r < cutoff_frac * min(H, W) / 2] = 0
        return np.fft.ifft2(np.fft.ifftshift(F)).real
    return _psnr(hp(a), hp(b))


def _replica_energy_ratio(recon, size=SIZE):
    """Energy near the predicted alias shifts (Q/2,0),(0,Q/2),(Q/2,Q/2) of the low-pass
    content, relative to a control high-frequency annulus away from those shifts."""
    F = np.abs(np.fft.fft2(recon))
    q = size // 2
    lo = 12                                        # low-pass core half-width
    def block(cy, cx):
        ys = (np.arange(-lo, lo + 1) + cy) % size
        xs = (np.arange(-lo, lo + 1) + cx) % size
        return float(np.mean(F[np.ix_(ys, xs)] ** 2))
    replicas = np.mean([block(q, 0), block(0, q), block(q, q)])
    # control: same-size blocks at odd positions away from replicas and DC
    controls = np.mean([block(q // 2, q // 4), block(q // 4, 3 * q // 2),
                        block(3 * q // 2, q // 2)])
    return float(replicas / (controls + 1e-12))


def _masks(rng, size=SIZE):
    out = []
    for oy in (0, 1):
        for ox in (0, 1):
            m = np.zeros((size, size), bool)
            m[oy::2, ox::2] = True
            out.append((f"lattice2_o{oy}{ox}", m))
    for i in range(8):
        m = np.zeros(size * size, bool)
        m[rng.choice(size * size, size * size // 4, replace=False)] = True
        out.append((f"random_{i}", m.reshape(size, size)))
    return out


def _train_2d(img, mask, scale, dev, weight_decay=1e-5, early_stop=False, seed=0,
              noise=NOISE, rng=None):
    import torch

    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    vals_clean = img.ravel()
    rng = rng or np.random.default_rng(seed)
    noise_vec = rng.normal(0, noise, vals_clean.size)
    vals_noisy = vals_clean + noise_vec
    train_idx = np.flatnonzero(mask.ravel())
    if early_stop:
        v = rng.permutation(train_idx)
        val_idx, train_idx = v[: v.size // 10], v[v.size // 10:]
    torch.manual_seed(seed)
    model = FourierFeatureMLP(n_features=256, scale=scale, hidden=256, layers=4,
                              seed=seed, in_dim=2).to(dev)
    cc = torch.tensor(coords[train_idx], dtype=torch.float32, device=dev)
    vv = torch.tensor(vals_noisy[train_idx, None], dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    best, best_state, patience = np.inf, None, 0
    if early_stop:
        cv = torch.tensor(coords[val_idx], dtype=torch.float32, device=dev)
        vv_val = torch.tensor(vals_noisy[val_idx, None], dtype=torch.float32, device=dev)
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = torch.mean((model(cc) - vv) ** 2)
        loss.backward(); opt.step()
        if early_stop and ep % 100 == 0:
            with torch.no_grad():
                vl = float(torch.mean((model(cv) - vv_val) ** 2))
            if vl < best - 1e-7:
                best, best_state, patience = vl, {k: v.clone() for k, v in
                                                  model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 5:
                    break
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        full = model(torch.tensor(coords, dtype=torch.float32, device=dev)
                     ).cpu().numpy().ravel()
    recon = full.reshape(H, W)
    held = np.setdiff1d(np.arange(vals_clean.size), np.flatnonzero(mask.ravel()))
    return recon, {
        "sample_psnr_clean": _psnr(vals_clean[train_idx], full[train_idx]),
        "sample_psnr_noisy": _psnr(vals_noisy[train_idx], full[train_idx]),
        "heldout_psnr": _psnr(vals_clean[held], full[held]),
        "full_psnr": _psnr(img, recon),
        "hf_psnr": _hf_psnr(img, recon),
        "replica_energy_ratio": _replica_energy_ratio(recon),
    }


def part_b_networks(quick=False):
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = _load_images()
    rng = np.random.default_rng(7)
    masks = _masks(rng)
    if quick:
        imgs = {"ascent": imgs["ascent"]}
        masks = masks[:2] + masks[4:6]
    B_img = SIZE // 2
    scales = {"matched": B_img / 4.0, "over": B_img / 1.2}
    runs = []
    for img_name, img in imgs.items():
        for mask_name, mask in masks:
            for scale_name, sc in scales.items():
                _, met = _train_2d(img, mask, sc, dev, rng=np.random.default_rng(11))
                met.update({"image": img_name, "mask": mask_name, "scale": scale_name,
                            "mask_kind": mask_name.split("_")[0], "variant": "plain"})
                runs.append(met)
                print(f"[2d] {img_name}/{mask_name}/{scale_name}: "
                      f"held={met['heldout_psnr']:.1f} full={met['full_psnr']:.1f} "
                      f"replica={met['replica_energy_ratio']:.2f}", flush=True)
        # regularization baselines on over-scale, lattice masks
        for mask_name, mask in [m for m in masks if m[0].startswith("lattice")][:4]:
            for variant, kw in (("ridge", {"weight_decay": 1e-2}),
                                ("early_stop", {"early_stop": True})):
                _, met = _train_2d(img, mask, scales["over"], dev,
                                   rng=np.random.default_rng(11), **kw)
                met.update({"image": img_name, "mask": mask_name, "scale": "over",
                            "mask_kind": "lattice2", "variant": variant})
                runs.append(met)
                print(f"[2d] {img_name}/{mask_name}/over/{variant}: "
                      f"full={met['full_psnr']:.1f} "
                      f"replica={met['replica_energy_ratio']:.2f}", flush=True)
    return runs, imgs, masks, scales, dev


def figure(partA_pack, runs, imgs, dev):
    Ares, Cmap, ks, in_vecs, pred_fold = partA_pack
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), layout="constrained")

    ax = axes[0]
    im = ax.imshow(np.log10(Cmap + 1e-6), origin="lower",
                   extent=[ks[0], ks[-1], ks[0], ks[-1]], cmap="magma")
    ax.plot(pred_fold[1], pred_fold[0], "o", ms=12, mfc="none", mec="cyan", mew=2)
    for v in in_vecs:
        ax.plot(v[1], v[0], "s", ms=9, mfc="none", mec="w", mew=1.2)
    ax.set_xlabel("$k_y$"); ax.set_ylabel("$k_x$")
    ax.set_title("(a) linear 2-D model: exact fold\n(circle = predicted alias vector)")
    fig.colorbar(im, ax=ax, shrink=0.85)

    # panel (b)/(c): DFT of over-scale reconstructions, lattice vs random masks, on the
    # synthetic image (sharp spectral lines -> replicas are visually unambiguous)
    if torch_available():
        img = imgs["synthetic"]
        rngm = np.random.default_rng(7)
        masks = _masks(rngm)
        lat = [m for n, m in masks if n == "lattice2_o00"][0]
        rnd = [m for n, m in masks if n == "random_0"][0]
        for ax, mask, name in ((axes[1], lat, "lattice mask"),
                               (axes[2], rnd, "random mask")):
            recon, _ = _train_2d(img, mask, SIZE / 2 / 1.2, dev,
                                 rng=np.random.default_rng(11))
            Fm = np.abs(np.fft.fftshift(np.fft.fft2(recon)))
            F = np.log10(Fm / Fm.max() + 1e-6)
            ax.imshow(F, cmap="magma", vmin=-4.5, vmax=0)
            q = SIZE // 2
            for (dy, dx) in ((q, 0), (0, q), (q, q)):
                cy, cx = (SIZE // 2 + dy) % SIZE, (SIZE // 2 + dx) % SIZE
                circ = plt.Circle((cx, cy), 13, fill=False, color="cyan", lw=1.6)
                ax.add_patch(circ)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"({'b' if name.startswith('lat') else 'c'}) over-scale DFT, "
                         f"{name}\n(circles = predicted replicas)")
    savefig(fig, "image2d_aliasing.png")


def main():
    if "--figure-only" in sys.argv:
        rng = np.random.default_rng(3)
        partA = part_a_linear(rng)
        print(f"[2d-linear] exact={partA[0]['exact']}", flush=True)
        imgs = _load_images()
        dev = "cpu"
        if torch_available():
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        figure(partA, [], imgs, dev)
        print("[2d] figure regenerated", flush=True)
        return

    quick = "--quick" in sys.argv
    rng = np.random.default_rng(3)
    partA = part_a_linear(rng)
    Ares = partA[0]
    print(f"[2d-linear] fold predicted={Ares['predicted_fold']} "
          f"measured={Ares['measured_fold']} exact={Ares['exact']}", flush=True)

    runs = []
    imgs = {}
    dev = "cpu"
    if torch_available():
        runs, imgs, *_rest, dev = part_b_networks(quick)
    else:
        print("[2d] torch unavailable -- Part B skipped (run on GPU server)", flush=True)
        imgs = _load_images()

    # aggregate replica-energy contrast (the headline 2-D quantity)
    agg = {}
    for kind in ("lattice2", "random"):
        for scale in ("matched", "over"):
            vals = [r["replica_energy_ratio"] for r in runs
                    if r["mask_kind"] == kind and r["scale"] == scale
                    and r["variant"] == "plain"]
            if vals:
                agg[f"{kind}_{scale}"] = {
                    "replica_ratio_mean": float(np.mean(vals)),
                    "replica_ratio_std": float(np.std(vals)),
                    "n": len(vals)}
    figure(partA, runs, imgs, dev)
    out = {"part_a_linear": Ares, "runs": runs, "aggregate_replica": agg,
           "size": SIZE, "epochs": EPOCHS, "noise": NOISE, "quick": quick,
           "note": "Part A: exact 2-D frequency-vector fold on the linear model. "
                   "Part B: trained networks, lattice (coherent) vs random masks; "
                   "parameter-matched scales; ridge/early-stop baselines; PSNR "
                   "reported separately vs clean samples / noisy targets / held-out "
                   "pixels / full field."}
    save_json("image2d_aliasing.json", out)
    print("[2d] aggregate:", {k: round(v["replica_ratio_mean"], 2)
                              for k, v in agg.items()}, flush=True)


if __name__ == "__main__":
    main()
