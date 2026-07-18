r"""Certified Cartesian MRI undersampling-mask design (Program 2).

Fourier duality maps the AliasGuard sampling principle onto k-space mask design.  For Cartesian
acquisition the readout axis is fully sampled and we CHOOSE a subset of phase-encode lines
:math:`k_y\in\{0,\dots,n_y-1\}` (a *mask*) at acceleration :math:`R` with a fixed central
autocalibration (ACS) block.  Undersampling folds the image along the phase-encode axis; the
fold structure is the mask's point spread function
:math:`\mathrm{PSF}(\Delta)=\tfrac1N\sum_{i\in\text{mask}}e^{i2\pi\Delta i/n_y}`
(:math:`\mathrm{PSF}(0)=1`).

Key fact (Parseval): the TOTAL sidelobe energy :math:`\sum_{\Delta\neq0}|\hat S(\Delta)|^2=n_yN-N^2`
is the SAME for every N-line mask -- what separates a coherent (equispaced) mask from an
incoherent (variable-density) one is how that energy is DISTRIBUTED.  A compact object of
half-extent :math:`W` (rows) is corrupted only by sidelobes at shifts :math:`0<|\Delta|\le W`, so
the mask-dependent, certifiable quantity is the **band-limited worst-case coherent aliasing**
:math:`\mu_W(\text{mask})=\max_{0<|\Delta|\le W}|\mathrm{PSF}(\Delta)|` (an exact max over a finite
set -- no Lipschitz slack needed).  AliasGuard designs the mask to minimize it; it is the only
mask with such a certified worst-case guarantee.

Everything here is numpy/scipy only (no external MRI/data deps): analytic phantoms, a TV-regularized
FISTA reconstruction applied IDENTICALLY to every mask, and PSNR / NMSE / SSIM metrics.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# centered orthonormal 2-D FFT (image <-> k-space)
# --------------------------------------------------------------------------------------
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def ifft2c(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X), norm="ortho"))


# --------------------------------------------------------------------------------------
# analytic phantoms (compactly supported objects -> the band-limited aliasing model applies)
# --------------------------------------------------------------------------------------
_SHEPP = [  # (intensity, a, b, x0, y0, phi_deg) -- standard Shepp-Logan ellipses
    (1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0),
    (-.2, .11, .31, .22, 0, -18), (-.2, .16, .41, -.22, 0, 18),
    (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0),
    (.1, .046, .046, 0, -.1, 0), (.1, .046, .023, -.08, -.605, 0),
    (.1, .023, .023, 0, -.606, 0), (.1, .023, .046, .06, -.605, 0),
]


def shepp_logan(n=192):
    """Analytic Shepp-Logan phantom on an ``n x n`` grid, intensities in [0, 1]."""
    ax = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(ax, ax)
    img = np.zeros((n, n))
    for inten, a, b, x0, y0, phi in _SHEPP:
        t = np.deg2rad(phi)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += inten
    return np.clip(img, 0, None) / img.max()


def structured_phantom(n=192, seed=0):
    """Random compact object: a few overlapping ellipses with smooth intensity (brain-like
    energy concentrated centrally, compact support so aliasing folds the object onto itself)."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(ax, ax)
    img = np.zeros((n, n))
    for _ in range(rng.integers(5, 9)):
        a, b = rng.uniform(0.15, 0.5, 2)
        x0, y0 = rng.uniform(-0.35, 0.35, 2)
        t = rng.uniform(0, np.pi)
        inten = rng.uniform(0.3, 1.0)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += inten
    img = np.clip(img, 0, None)
    return img / (img.max() + 1e-9)


def object_half_extent(img, axis=0, thresh=0.02):
    """Half-extent W (in rows) of the object's support along ``axis`` (phase-encode).  The
    band-limited certificate uses shifts up to W; a compact object has W < n/2."""
    proj = np.abs(img).sum(axis=1 - axis)
    on = np.where(proj > thresh * proj.max())[0]
    if on.size == 0:
        return img.shape[axis] // 2
    c = img.shape[axis] / 2.0
    return int(np.ceil(max(abs(on[0] - c), abs(on[-1] - c))))


# --------------------------------------------------------------------------------------
# Cartesian phase-encode masks (line selection); all include the ACS block, budget ~ ny/R
# --------------------------------------------------------------------------------------
def _budget(ny, R):
    return max(1, int(round(ny / R)))


def acs_indices(ny, acs_frac):
    n_acs = max(2, int(round(acs_frac * ny)))
    c = ny // 2
    return list(range(c - n_acs // 2, c - n_acs // 2 + n_acs))


def _psf_abs(idx, ny, W):
    """|PSF(Delta)| for 1<=Delta<=W from a set of line indices (mainlobe removed)."""
    idx = np.asarray(sorted(set(int(i) % ny for i in idx)))
    N = idx.size
    deltas = np.arange(1, W + 1)
    S = np.exp(1j * 2 * np.pi * np.outer(deltas, idx) / ny).sum(axis=1)
    return np.abs(S) / N


def peak_sidelobe(idx, ny, W):
    """Certified band-limited worst-case coherent aliasing mu_W = max_{0<|Delta|<=W}|PSF|."""
    return float(np.max(_psf_abs(idx, ny, W)))


def sidelobe_energy(idx, ny, W):
    """Band-limited sidelobe energy sum_{0<Delta<=W}|PSF|^2 (the smooth design surrogate)."""
    return float(np.sum(_psf_abs(idx, ny, W) ** 2))


def mask_equispaced(ny, R, acs_frac):
    acs = set(acs_indices(ny, acs_frac))
    lines = set(range(0, ny, R)) | acs
    return _finalize(lines, ny, R, acs)


def mask_random(ny, R, acs_frac, seed=0):
    rng = np.random.default_rng(seed)
    acs = set(acs_indices(ny, acs_frac))
    pool = [i for i in range(ny) if i not in acs]
    need = _budget(ny, R) - len(acs)
    extra = rng.choice(pool, size=max(0, min(need, len(pool))), replace=False) if need > 0 else []
    return _finalize(acs | set(int(i) for i in extra), ny, R, acs)


def mask_vardensity(ny, R, acs_frac, seed=0, poly=6):
    """Polynomial variable-density: sample non-ACS lines with prob ~ (1-|k|/kmax)^poly."""
    rng = np.random.default_rng(seed)
    acs = set(acs_indices(ny, acs_frac))
    c = ny // 2
    k = np.abs(np.arange(ny) - c) / (ny / 2)
    w = (1 - k) ** poly
    for i in acs:
        w[i] = 0
    w = w / w.sum()
    need = _budget(ny, R) - len(acs)
    chosen = set(acs)
    if need > 0:
        pick = rng.choice(ny, size=min(need, int((w > 0).sum())), replace=False, p=w)
        chosen |= set(int(i) for i in pick)
    return _finalize(chosen, ny, R, acs)


def mask_lowdisc(ny, R, acs_frac):
    """Deterministic low-discrepancy (van der Corput base 2) line positions + ACS."""
    acs = set(acs_indices(ny, acs_frac))
    need = _budget(ny, R) - len(acs)
    chosen = set(acs)
    i, cnt = 1, 0
    while cnt < max(0, need) and i < 100 * ny:
        f, b, x = 1.0, 2, 0.0
        k = i
        while k > 0:
            f /= b
            x += f * (k % b)
            k //= b
        pos = int(round(x * ny)) % ny
        if pos not in chosen:
            chosen.add(pos)
            cnt += 1
        i += 1
    return _finalize(chosen, ny, R, acs)


def mask_aliasguard(ny, R, acs_frac, W, seed=0, restarts=4):
    r"""AliasGuard mask: greedily choose phase-encode lines (from the ACS block) to MINIMIZE the
    band-limited worst-case coherent aliasing (the L-infinity certificate)
    :math:`\mu_W=\max_{0<\Delta\le W}|\mathrm{PSF}(\Delta)|`.  Total sidelobe ENERGY is
    Parseval-invariant, so the discriminating objective is the PEAK: at each step add the line
    minimizing the resulting peak sidelobe.  Multi-restart (deterministic given seed) with
    tie-break jitter; keeps the lowest-peak mask found."""
    acs = set(acs_indices(ny, acs_frac))
    budget = _budget(ny, R)
    deltas = np.arange(1, W + 1)
    rng = np.random.default_rng(seed)
    best_set, best_peak = None, np.inf
    for r in range(restarts):
        chosen = sorted(acs)
        S = np.exp(1j * 2 * np.pi * np.outer(deltas, chosen) / ny).sum(axis=1)  # (W,)
        cand = [i for i in range(ny) if i not in acs]
        if r > 0:
            rng.shuffle(cand)                                   # diversify greedy tie-breaks
        cand = np.array(cand)
        alive = np.ones(cand.size, bool)
        while len(chosen) < budget and alive.any():
            E = np.exp(1j * 2 * np.pi * np.outer(deltas, cand) / ny)   # (W, |cand|)
            newS = S[:, None] + E                               # (W, |cand|)
            peaks = np.max(np.abs(newS), axis=0)                # resulting peak per candidate
            peaks[~alive] = np.inf
            j = int(np.argmin(peaks))
            S = newS[:, j]
            chosen.append(int(cand[j]))
            alive[j] = False
        peak = float(np.max(np.abs(S)) / len(chosen))
        if peak < best_peak:
            best_peak, best_set = peak, set(chosen)
    return _finalize(best_set, ny, R, acs)


def mask_ds(ny, R, acs_frac, W, seed=0):
    r"""D_s-style greedy: maximize log-det of the in-band Fourier Gram (rows = object support
    band of half-extent W) over selected lines -- the classical optimal-design baseline against
    which AliasGuard's coherence objective is compared.  Adding line :math:`i` is a rank-one
    update :math:`G\!+\!\varphi_i\varphi_i^{*}`, so the log-det gain is
    :math:`\log(1+\varphi_i^{*}G^{-1}\varphi_i)` and all candidates are scored at once
    (Sherman-Morrison), then :math:`G^{-1}` is updated -- no per-candidate factorization."""
    acs = set(acs_indices(ny, acs_frac))
    budget = _budget(ny, R)
    rows = np.arange(-W, W + 1)                                 # object-support rows (band); m=2W+1
    ridge = 1e-6
    chosen = sorted(acs)
    Phi0 = np.exp(1j * 2 * np.pi * np.outer(chosen, rows) / ny)
    Ginv = np.linalg.inv(Phi0.conj().T @ Phi0 + ridge * np.eye(rows.size))
    chosen = set(chosen)
    cand = np.array([i for i in range(ny) if i not in chosen])
    Ecand = np.exp(1j * 2 * np.pi * np.outer(cand, rows) / ny)  # (|cand|, m)
    alive = np.ones(cand.size, bool)
    while len(chosen) < budget and alive.any():
        V = Ecand @ Ginv                                       # (|cand|, m)
        q = np.real(np.sum(np.conj(Ecand) * V, axis=1))        # phi^* Ginv phi per candidate
        q[~alive] = -np.inf
        j = int(np.argmax(q))                                  # max log(1+q) == max q
        phi = Ecand[j]
        u = Ginv @ phi
        Ginv = Ginv - np.outer(u, np.conj(u)) / (1.0 + q[j])   # Sherman-Morrison
        chosen.add(int(cand[j]))
        alive[j] = False
    return _finalize(chosen, ny, R, acs)


def _finalize(lines, ny, R, acs):
    """Trim/pad a line set to the exact budget ny/R (never drop ACS); return a sorted mask array."""
    lines = set(int(i) % ny for i in lines)
    budget = _budget(ny, R)
    if len(lines) > budget:                                    # drop farthest-from-center non-ACS
        c = ny // 2
        extra = sorted((i for i in lines if i not in acs),
                       key=lambda i: -abs(i - c))
        for i in extra[:len(lines) - budget]:
            lines.discard(i)
    m = np.zeros(ny, bool)
    m[sorted(lines)] = True
    return m


def mask_2d(mask_lines, nx):
    """Broadcast a phase-encode line mask (ny,) to a full 2-D k-space mask (ny, nx)."""
    return np.repeat(mask_lines[:, None], nx, axis=1)


# --------------------------------------------------------------------------------------
# reconstruction: zero-filled and TV-regularized FISTA (identical for every mask)
# --------------------------------------------------------------------------------------
def undersample(img, mask_lines):
    """Return masked k-space (zeros on unacquired lines) of a real image."""
    ksp = fft2c(img)
    m2 = mask_2d(mask_lines, img.shape[1])
    return ksp * m2, m2


def recon_zerofilled(ksp_us):
    return np.abs(ifft2c(ksp_us))


def _tv_denoise(b, weight, iters=12, dt=0.2, eps=1e-4):
    r"""ROF total-variation denoise: gradient flow of
    :math:`\tfrac12\|u-b\|^2+\text{weight}\cdot\mathrm{TV}(u)`,
    :math:`\partial_t u=\text{weight}\,\nabla\!\cdot(\nabla u/|\nabla u|)-(u-b)`.
    Stable for ``dt<=0.25`` (the fidelity coefficient is 1; weight scales only the TV term)."""
    u = b.copy()
    for _ in range(iters):
        ux = np.diff(u, axis=0, append=u[-1:, :])
        uy = np.diff(u, axis=1, append=u[:, -1:])
        mag = np.sqrt(ux ** 2 + uy ** 2 + eps)
        nx, ny_ = ux / mag, uy / mag
        div = (nx - np.roll(nx, 1, axis=0)) + (ny_ - np.roll(ny_, 1, axis=1))
        u = u + dt * (weight * div - (u - b))
    return u


def recon_tv(ksp_us, m2, lam=0.03, iters=60, tv_inner=10):
    r"""Data-consistent TV reconstruction by POCS: alternate ROF TV-denoising with k-space data
    consistency (measured lines restored each iteration).  Robust and standard for CS-MRI;
    identical solver/params for every mask so comparisons reflect the MASK, not the recon.
    Real, nonnegative output."""
    x = np.abs(ifft2c(ksp_us))
    for _ in range(iters):
        x = _tv_denoise(x, weight=lam, iters=tv_inner)
        X = fft2c(x)
        X = np.where(m2, ksp_us, X)                            # data consistency on measured lines
        x = np.clip(np.real(ifft2c(X)), 0, None)
    return x


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def psnr(x, ref):
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    mse = float(np.mean((x - ref) ** 2))
    return float(10 * np.log10((np.ptp(ref) ** 2) / (mse + 1e-12)))


def nmse(x, ref):
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    return float(np.sum((x - ref) ** 2) / (np.sum(ref ** 2) + 1e-12))


def ssim(x, ref, win=7, sigma=1.5):
    """Compact Gaussian-windowed SSIM (numpy only), global data range from the reference."""
    from scipy.ndimage import gaussian_filter
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    L = float(np.ptp(ref)) + 1e-12
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_x = gaussian_filter(x, sigma)
    mu_r = gaussian_filter(ref, sigma)
    mx2, mr2, mxr = mu_x ** 2, mu_r ** 2, mu_x * mu_r
    vx = gaussian_filter(x * x, sigma) - mx2
    vr = gaussian_filter(ref * ref, sigma) - mr2
    vxr = gaussian_filter(x * ref, sigma) - mxr
    s = ((2 * mxr + c1) * (2 * vxr + c2)) / ((mx2 + mr2 + c1) * (vx + vr + c2))
    return float(np.mean(s))
