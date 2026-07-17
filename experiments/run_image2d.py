r"""E5 -- two-dimensional structured aliasing: exact theory + trained-network study.

Part A (CPU, exact).  The 2-D fold relation: on a rate-(Q,Q) grid the sample atom of an
integer frequency VECTOR depends only on ``nu mod Q`` componentwise, so an out-of-band tone
with ``nu = omega (mod Q)`` is exactly indistinguishable from the in-band atom ``omega``.
We verify this over a HELD-OUT test set of out-of-band frequency vectors (disjoint from the
model's design dictionary), planting each in turn and checking the recovered coefficient map
folds onto the predicted in-band vector.

Part B (GPU, empirical extrapolation).  Trained 2-D Fourier-feature MLPs on real images:

* **lattice masks** (every-2nd-pixel sublattices) create a *coherent* 2-D fold;
* **random masks** of matched density are the NO-ALIAS control.

The headline quantity is the *excess* coherent-replica energy of lattice over random masks,
measured on the reconstruction-ERROR spectrum (so true image content does not inflate it),
over >= 20 independent (offset, mask, weight, noise) seeds with a paired bootstrap CI.  We
also report clean-sample / noisy-target / held-out-pixel / full-field PSNR separately, a
noise-level sweep, and ridge / early-stopping baselines on the over-scaled model
(architectures are parameter-matched across scales -- only the feature scale differs).

Anti-aliased resize throughout (``scipy.ndimage``), never bare striding.

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
HEADLINE_SEEDS = 20
TABLE_SEEDS = 3
NOISE_LEVELS = (0.01, 0.05)


# --------------------------------------------------------------------------------------
# Part A: exact 2-D fold on the linear model (CPU) -- tested over held-out tone vectors
# --------------------------------------------------------------------------------------
def _synthesis_2d(freq_vecs, pts):
    """Phi[j, k] = exp(i 2 pi <nu_k, p_j>) for 2-D frequency vectors and points."""
    return np.exp(1j * 2 * np.pi * (pts @ freq_vecs.T))


def _principal_mod(nu, Q):
    """Componentwise fold of a frequency vector into the principal cell (-Q/2, Q/2]."""
    r = np.mod(nu, Q)
    r = np.where(r > Q / 2, r - Q, r)
    return r


def _mask_true(KX, KY, in_vecs):
    m = np.zeros_like(KX, dtype=float)
    for v in in_vecs:
        m[(KX == int(v[0])) & (KY == int(v[1]))] = 1.0
    return m


def _fit_fold(Lam, pts, in_vecs, in_c, nu_out, KX, KY, ks, B, a=0.8):
    """Plant +-nu_out on the in-band background, fit the 2-D model, return the recovered
    out-of-band fold vector."""
    Phi = _synthesis_2d(Lam, pts)
    y = (_synthesis_2d(in_vecs, pts) @ in_c
         + a * (_synthesis_2d(nu_out[None], pts)[:, 0]
                + _synthesis_2d(-nu_out[None], pts)[:, 0])).real
    c, *_ = np.linalg.lstsq(Phi, y.astype(complex), rcond=None)
    Cmap = np.abs(c).reshape(2 * B + 1, 2 * B + 1)
    search = Cmap * (1.0 - _mask_true(KX, KY, in_vecs))
    k_meas = np.unravel_index(int(np.argmax(search)), Cmap.shape)
    meas_fold = np.array([ks[k_meas[0]], ks[k_meas[1]]], float)
    resid = float(np.linalg.norm(y - (Phi @ c).real) / np.sqrt(pts.shape[0]))
    return meas_fold, Cmap, resid


def part_a_linear(rng):
    Q, B, N = 32, 6, 600
    ks = np.arange(-B, B + 1)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    Lam = np.stack([KX.ravel(), KY.ravel()], axis=1).astype(float)      # design dictionary
    sites = np.stack(np.meshgrid(np.arange(Q), np.arange(Q), indexing="ij"),
                     axis=-1).reshape(-1, 2)
    sel = rng.choice(sites.shape[0], size=N, replace=False)
    pts = sites[sel] / Q
    in_vecs = np.array([[2.0, 1.0], [-2.0, -1.0], [1.0, 4.0], [-1.0, -4.0]])
    in_c = np.array([1.0, 1.0, 0.5, 0.5]).astype(complex)

    # HELD-OUT test set: out-of-band tone vectors NOT in the design dictionary, each with a
    # known in-band fold.  First is fixed for a reproducible figure; rest are random.
    tests = [(np.array([27.0, 3.0]), np.array([-5.0, 3.0]))]              # 27 = -5 (mod 32)
    for _ in range(7):
        tgt = rng.integers(-B, B + 1, 2).astype(float)
        shift = Q * rng.integers(1, 3, 2) * rng.choice([-1, 1], 2)
        nu = tgt + shift
        if np.any(np.abs(nu) <= B):                                       # keep it out-of-band
            nu = nu + Q * np.sign(nu + 1e-9)
        tests.append((nu.astype(float), _principal_mod(nu, Q)))

    results = []
    rep_pack = None
    for i, (nu_out, pred_fold) in enumerate(tests):
        meas_fold, Cmap, resid = _fit_fold(Lam, pts, in_vecs, in_c, nu_out, KX, KY, ks, B)
        exact = bool(np.allclose(meas_fold, pred_fold) or np.allclose(meas_fold, -pred_fold))
        results.append({"nu_out": nu_out.tolist(), "predicted_fold": pred_fold.tolist(),
                        "measured_fold": meas_fold.tolist(), "exact": exact,
                        "sample_residual": resid})
        if i == 0:
            rep_pack = (Cmap, ks, in_vecs, pred_fold)
    exact_rate = float(np.mean([r["exact"] for r in results]))
    summary = {"Q": Q, "B": B, "N": N, "n_test_tones": len(tests),
               "fold_exact_rate": exact_rate, "tests": results,
               # kept for backward-compatible figure/caption references
               "predicted_fold": results[0]["predicted_fold"],
               "measured_fold": results[0]["measured_fold"],
               "exact": bool(exact_rate == 1.0), "sample_residual": results[0]["sample_residual"]}
    Cmap, ks, in_vecs, pred_fold = rep_pack
    return summary, Cmap, ks, in_vecs, pred_fold


# --------------------------------------------------------------------------------------
# Part B: trained 2-D networks, lattice vs random masks (GPU)
# --------------------------------------------------------------------------------------
def _aa_resize(img, size):
    """Anti-aliased downsample: Gaussian low-pass at the decimation scale, then interpolate
    to exactly (size, size).  Never bare striding (which aliases)."""
    from scipy.ndimage import gaussian_filter, zoom

    img = np.asarray(img, float)
    fy, fx = img.shape[0] / size, img.shape[1] / size
    lp = gaussian_filter(img, sigma=(max(fy, 1.0) / 2.0, max(fx, 1.0) / 2.0))
    out = zoom(lp, (size / img.shape[0], size / img.shape[1]), order=1)
    if out.shape[0] < size or out.shape[1] < size:
        out = np.pad(out, ((0, max(0, size - out.shape[0])),
                           (0, max(0, size - out.shape[1]))), mode="edge")
    return out[:size, :size]


def _load_images(size=SIZE):
    from scipy.datasets import ascent, face

    imgs = {}
    imgs["ascent"] = _aa_resize(np.asarray(ascent(), float), size)
    imgs["face"] = _aa_resize(np.asarray(face(gray=True), float), size)
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


def _replica_error_ratio(recon, img, size=SIZE):
    """Coherent-fold energy on the reconstruction-ERROR spectrum |FFT(recon - truth)|^2 at
    the predicted alias shifts (Q/2,0),(0,Q/2),(Q/2,Q/2), relative to total error energy.

    Using the ERROR (not the reconstruction) removes the true image spectrum, so this
    isolates aliasing.  The lattice-minus-random version (excess) is the headline."""
    E = np.abs(np.fft.fft2(np.asarray(recon) - np.asarray(img))) ** 2
    q, lo = size // 2, 12

    def block(cy, cx):
        ys = (np.arange(-lo, lo + 1) + cy) % size
        xs = (np.arange(-lo, lo + 1) + cx) % size
        return float(np.mean(E[np.ix_(ys, xs)]))

    replica = np.mean([block(q, 0), block(0, q), block(q, q)])
    return float(replica / (float(np.mean(E)) + 1e-12))


def _lattice_mask(size, oy, ox):
    m = np.zeros((size, size), bool)
    m[oy::2, ox::2] = True
    return m


def _random_mask(size, k, rng):
    m = np.zeros(size * size, bool)
    m[rng.choice(size * size, int(k), replace=False)] = True
    return m.reshape(size, size)


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
        "replica_error_ratio": _replica_error_ratio(recon, img),
    }


def _paired_boot(diff, rng, n=2000):
    diff = np.asarray(diff, float)
    bs = [diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(n)]
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def part_b_headline(dev, img, sc, n_seeds, noise=NOISE, tag=""):
    """>= 20 independent seeds; per seed a PAIRED lattice vs matched-density random mask
    (same weight init + noise within the pair).  Excess = lattice - random replica energy."""
    lat_vals, rnd_vals = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        oy, ox = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        latm = _lattice_mask(SIZE, oy, ox)
        rndm = _random_mask(SIZE, latm.sum(), rng)             # matched density (no-alias control)
        _, ml = _train_2d(img, latm, sc, dev, seed=2000 + s,
                          noise=noise, rng=np.random.default_rng(3000 + s))
        _, mr = _train_2d(img, rndm, sc, dev, seed=2000 + s,
                          noise=noise, rng=np.random.default_rng(3000 + s))
        lat_vals.append(ml["replica_error_ratio"])
        rnd_vals.append(mr["replica_error_ratio"])
        print(f"[2d-headline{tag}] s={s} lattice={lat_vals[-1]:.3f} random={rnd_vals[-1]:.3f}",
              flush=True)
    lat, rnd = np.array(lat_vals), np.array(rnd_vals)
    diff = lat - rnd
    return {"n_seeds": n_seeds, "noise": noise,
            "lattice_replica_mean": float(lat.mean()),
            "random_replica_mean": float(rnd.mean()),
            "excess_mean": float(diff.mean()),
            "excess_ci95_paired": _paired_boot(diff, np.random.default_rng(0)),
            "lattice_gt_random_rate": float(np.mean(diff > 0))}


def part_b_table(dev, quick=False):
    """Per-config PSNR table over TABLE_SEEDS independent (weight, noise, mask) seeds."""
    imgs = _load_images()
    if quick:
        imgs = {"synthetic": imgs["synthetic"]}
    B_img = SIZE // 2
    scales = {"matched": B_img / 4.0, "over": B_img / 1.2}
    nseeds = 1 if quick else TABLE_SEEDS
    runs = []
    for img_name, img in imgs.items():
        for scale_name, sc in scales.items():
            for s in range(nseeds):
                rng = np.random.default_rng(500 + s)
                configs = [("lattice2", _lattice_mask(SIZE, s % 2, (s // 2) % 2))]
                configs.append(("random", _random_mask(SIZE, SIZE * SIZE // 4, rng)))
                for kind, mask in configs:
                    _, met = _train_2d(img, mask, sc, dev, seed=2000 + s,
                                      rng=np.random.default_rng(3000 + s))
                    met.update({"image": img_name, "mask_kind": kind, "scale": scale_name,
                                "seed": s, "variant": "plain"})
                    runs.append(met)
        # ridge / early-stop baselines on the over-scale lattice model (one seed)
        for variant, kw in (("ridge", {"weight_decay": 1e-2}),
                            ("early_stop", {"early_stop": True})):
            _, met = _train_2d(img, _lattice_mask(SIZE, 0, 0), scales["over"], dev,
                              seed=2000, rng=np.random.default_rng(3000), **kw)
            met.update({"image": img_name, "mask_kind": "lattice2", "scale": "over",
                        "seed": 0, "variant": variant})
            runs.append(met)
            print(f"[2d-table] {img_name}/over/{variant}: full={met['full_psnr']:.1f} "
                  f"replica={met['replica_error_ratio']:.3f}", flush=True)
    return runs


def figure(partA_pack, imgs, dev):
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

    if torch_available():
        img = imgs["synthetic"]
        lat = _lattice_mask(SIZE, 0, 0)
        rnd = _random_mask(SIZE, lat.sum(), np.random.default_rng(7))
        for ax, mask, name in ((axes[1], lat, "lattice mask"),
                               (axes[2], rnd, "random mask")):
            recon, _ = _train_2d(img, mask, SIZE / 2 / 1.2, dev,
                                 seed=2000, rng=np.random.default_rng(3000))
            Fm = np.abs(np.fft.fftshift(np.fft.fft2(recon - img)))
            F = np.log10(Fm / (Fm.max() + 1e-12) + 1e-6)
            ax.imshow(F, cmap="magma", vmin=-4.5, vmax=0)
            for (dy, dx) in ((SIZE // 2, 0), (0, SIZE // 2), (SIZE // 2, SIZE // 2)):
                cy, cx = (SIZE // 2 + dy) % SIZE, (SIZE // 2 + dx) % SIZE
                ax.add_patch(plt.Circle((cx, cy), 13, fill=False, color="cyan", lw=1.6))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"({'b' if name.startswith('lat') else 'c'}) error-spectrum, "
                         f"{name}\n(circles = predicted replicas)")
    savefig(fig, "image2d_aliasing.png")


def main():
    quick = "--quick" in sys.argv
    figure_only = "--figure-only" in sys.argv
    rng = np.random.default_rng(3)
    partA = part_a_linear(rng)
    Ares = partA[0]
    print(f"[2d-linear] fold_exact_rate={Ares['fold_exact_rate']:.2f} "
          f"over {Ares['n_test_tones']} held-out tones", flush=True)

    dev = "cpu"
    if torch_available():
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = _load_images()

    if figure_only:
        figure(partA, imgs, dev)
        print("[2d] figure regenerated", flush=True)
        return

    headline, headline_noise, table = None, [], []
    if torch_available():
        sc = SIZE / 2 / 1.2                                   # over-scale
        ns = 4 if quick else HEADLINE_SEEDS
        headline = part_b_headline(dev, imgs["synthetic"], sc, ns)
        # noise-level robustness of the excess (fewer seeds each)
        for nz in ((NOISE_LEVELS[0],) if quick else NOISE_LEVELS):
            headline_noise.append(part_b_headline(dev, imgs["synthetic"], sc,
                                                  max(3, ns // 4), noise=nz, tag=f"-nz{nz}"))
        table = part_b_table(dev, quick)
    else:
        print("[2d] torch unavailable -- Part B skipped (run on GPU server)", flush=True)

    figure(partA, imgs, dev)
    out = {"part_a_linear": Ares, "headline_excess": headline,
           "headline_noise_sweep": headline_noise, "table_runs": table,
           "size": SIZE, "epochs": EPOCHS, "noise": NOISE, "quick": quick,
           "note": "Part A: exact 2-D frequency-vector fold verified over a held-out tone "
                   "set (disjoint from the design dictionary). Part B: headline is the "
                   "EXCESS coherent-replica energy of lattice over matched-density random "
                   "(no-alias control) masks on the reconstruction-ERROR spectrum, >=20 "
                   "paired seeds with bootstrap CI, plus a noise sweep; anti-aliased "
                   "resize; PSNR reported separately vs clean / noisy / held-out / full."}
    save_json("image2d_aliasing.json", out)
    if headline:
        print(f"[2d] headline excess={headline['excess_mean']:.3f} "
              f"CI{headline['excess_ci95_paired']} "
              f"lattice>random {headline['lattice_gt_random_rate']*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
