r"""Implicit neural representations: the analyzable fixed-feature core and trained nonlinear
INRs.

* :class:`FixedFeatureINR` -- a Fourier-feature network with *fixed* feature frequencies and
  a linear output layer.  Its representable set is exactly :math:`\mathrm{span}\{e_\omega:
  \omega\in\Lambda\}`, so fitting is least squares and every closed-form limit of
  :mod:`inralias.limits` applies exactly.  This is the object the theorems are about.

* :func:`bandwidth_matched_freqs` -- pick :math:`\Lambda` to cover a signal's essential band
  estimated *from the samples* (nonuniform periodogram), realising the Theorem-1
  (achievability) regime; the matched estimator of the paper.

* :class:`SIREN`, :class:`FourierFeatureMLP`, :func:`train_inr` -- trained *nonlinear* INRs
  (require torch) used to show the aliasing phase transition persists beyond the linear core,
  on synthetic and real signals.  Torch is imported lazily; the theory path needs only numpy.
"""
from __future__ import annotations

import numpy as np

from inralias.sampling import synthesis_matrix, pinv_apply, noise_gain

__all__ = [
    "FixedFeatureINR",
    "real_design",
    "bandwidth_matched_freqs",
    "correlation_periodogram",
    "lombscargle_periodogram",
    "SIREN",
    "FourierFeatureMLP",
    "train_inr",
    "torch_available",
]


def real_design(freqs: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Real cosine/sine design matrix for a Hermitian-symmetric frequency set.

    Returns ``(D, pos)`` where ``pos`` is the sorted array of distinct nonnegative
    frequencies in ``freqs`` and ``D`` has columns ``[1?] + [cos(2 pi w t), sin(2 pi w t)
    for w in pos, w > 0]`` (the constant column present iff ``0 in pos``).  Fitting a real
    signal by least squares on ``D`` is exactly equivalent to complex LS on the
    Hermitian-symmetric exponential dictionary with the constraint
    :math:`c_{-\omega}=\overline{c_\omega}` *enforced*, not merely applied post hoc.
    """
    t = np.asarray(t, float).reshape(-1)
    pos = np.unique(np.abs(np.asarray(freqs, float)))
    cols = []
    if pos.size and pos[0] == 0.0:
        cols.append(np.ones_like(t))
    for w in pos[pos > 0]:
        ph = 2 * np.pi * w * t
        cols.append(np.cos(ph))
        cols.append(np.sin(ph))
    return np.stack(cols, axis=1), pos


class FixedFeatureINR:
    r"""Fixed Fourier-feature coordinate model (linear coefficients). The theory-core model.

    Parameters
    ----------
    freqs : the model frequency set :math:`\Lambda`.
    real  : if True, fit a **real cosine/sine design** (Hermitian symmetry enforced in the
        parameterization); ``freqs`` must then be Hermitian-symmetric.  ``coeffs_`` still
        exposes the equivalent complex coefficients aligned with ``self.freqs`` for
        spectrum inspection.
    """

    def __init__(self, freqs: np.ndarray, real: bool = False):
        self.freqs = np.asarray(freqs, float)
        self.real = real
        self.coeffs_: np.ndarray | None = None
        self._beta: np.ndarray | None = None

    def _real_fit(self, t, y, ridge):
        D, pos = real_design(self.freqs, t)
        yv = np.asarray(y, float).reshape(-1)
        if ridge > 0:
            p = D.shape[1]
            D_aug = np.vstack([D, np.sqrt(ridge) * np.eye(p)])
            y_aug = np.concatenate([yv, np.zeros(p)])
            beta, *_ = np.linalg.lstsq(D_aug, y_aug, rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(D, yv, rcond=None)
        self._beta = beta
        # equivalent complex Hermitian coefficients aligned with self.freqs
        c = np.zeros(self.freqs.size, complex)
        idx = 0
        has_dc = pos.size and pos[0] == 0.0
        if has_dc:
            c[np.isclose(self.freqs, 0.0)] = beta[0]
            idx = 1
        for w in pos[pos > 0]:
            a, b = beta[idx], beta[idx + 1]
            idx += 2
            c[np.isclose(self.freqs, w)] = (a - 1j * b) / 2
            c[np.isclose(self.freqs, -w)] = (a + 1j * b) / 2
        self.coeffs_ = c

    def fit(self, t: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> "FixedFeatureINR":
        if self.real:
            self._real_fit(t, y, ridge)
        else:
            Phi = synthesis_matrix(self.freqs, t)
            self.coeffs_ = pinv_apply(Phi, np.asarray(y, complex), ridge=ridge)
        return self

    def predict(self, t: np.ndarray) -> np.ndarray:
        assert self.coeffs_ is not None, "call fit() first"
        if self.real:
            D, _ = real_design(self.freqs, t)
            return D @ self._beta
        return synthesis_matrix(self.freqs, t) @ self.coeffs_

    def noise_gain(self, t: np.ndarray) -> float:
        if self.real:
            D, _ = real_design(self.freqs, t)
            s = np.linalg.svd(D, compute_uv=False)
            return float(1.0 / s[-1]) if s.size and s[-1] > 0 else np.inf
        return noise_gain(synthesis_matrix(self.freqs, t))


def correlation_periodogram(t: np.ndarray, y: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
    r"""Correlation (Schuster-type) periodogram of nonuniform samples on ``freq_grid``.

    This is the squared magnitude of the plain per-frequency correlation
    :math:`|\tfrac1N\sum_j y_j e^{-i2\pi f t_j}|^2`.  It is **not** the Lomb--Scargle
    periodogram (no per-frequency least-squares normalization, no time-offset
    invariance); use :func:`lombscargle_periodogram` for that.
    """
    t = np.asarray(t, float)
    y = np.asarray(y)
    freq_grid = np.asarray(freq_grid, float)
    E = np.exp(-1j * 2 * np.pi * np.outer(freq_grid, t))  # (F, N)
    corr = (E @ y) / t.size
    return np.abs(corr) ** 2


def lombscargle_periodogram(t: np.ndarray, y: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
    r"""Classical Lomb--Scargle periodogram (per-frequency least squares with the
    time-offset that decorrelates the cosine and sine terms), via
    :func:`scipy.signal.lombscargle`.  ``y`` is centered first; zero frequencies are
    assigned the squared mean so the DC term remains comparable."""
    from scipy.signal import lombscargle

    t = np.asarray(t, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    f = np.asarray(freq_grid, float).reshape(-1)
    out = np.zeros(f.size)
    nz = f > 0
    if np.any(nz):
        out[nz] = lombscargle(t, y - y.mean(), 2 * np.pi * f[nz], precenter=False)
    out[~nz] = y.mean() ** 2 * t.size / 4.0
    return out


def bandwidth_matched_freqs(
    t: np.ndarray,
    y: np.ndarray,
    max_bandwidth: int,
    energy_keep: float = 0.99,
    real: bool = True,
    method: str = "lombscargle",
) -> np.ndarray:
    r"""Estimate the essential band **from the samples only** and return an integer
    dictionary covering it.

    Computes a periodogram (``method``: ``"lombscargle"`` or ``"correlation"``) on the
    integer grid up to ``max_bandwidth``, keeps the smallest band capturing
    ``energy_keep`` of the periodogram energy, and returns the integer lowpass dictionary
    of that bandwidth (Hermitian-symmetric for real signals).  This is the *deployable*
    estimator; it never sees the full reference signal.
    """
    grid = np.arange(0, max_bandwidth + 1).astype(float)
    P = (lombscargle_periodogram if method == "lombscargle" else correlation_periodogram)(
        t, y, grid
    )
    csum = np.cumsum(P) / (np.sum(P) + 1e-30)
    B = int(np.searchsorted(csum, energy_keep))
    B = max(1, min(B, max_bandwidth))
    return np.arange(-B, B + 1).astype(float) if real else np.arange(0, B + 1).astype(float)


# --------------------------------------------------------------------------------------
# Trained nonlinear INRs (torch, optional)
# --------------------------------------------------------------------------------------
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "This nonlinear-INR feature needs torch. Install with "
            "`uv pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple` "
            "(or run on the GPU server)."
        ) from e
    return __import__("torch")


def SIREN(hidden: int = 256, layers: int = 3, w0: float = 30.0):  # noqa: N802
    """Construct a SIREN (sine-activation coordinate MLP). Returns an ``nn.Module``."""
    torch = _require_torch()
    import torch.nn as nn

    class _Sine(nn.Module):
        def __init__(self, w0):
            super().__init__()
            self.w0 = w0

        def forward(self, x):
            return torch.sin(self.w0 * x)

    mods: list = []
    in_dim = 1
    for i in range(layers):
        lin = nn.Linear(in_dim, hidden)
        with torch.no_grad():
            if i == 0:
                lin.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                b = np.sqrt(6 / in_dim) / w0
                lin.weight.uniform_(-b, b)
        mods += [lin, _Sine(w0)]
        in_dim = hidden
    mods += [nn.Linear(in_dim, 1)]
    return nn.Sequential(*mods)


def FourierFeatureMLP(n_features: int = 128, scale: float = 10.0, hidden: int = 256,
                      layers: int = 3, seed: int = 0, in_dim: int = 1):  # noqa: N802
    """Random-Fourier-feature input mapping + ReLU MLP (Tancik et al.). Returns ``nn.Module``.

    ``scale`` sets the standard deviation of the feature frequencies and hence the INR's
    representable band -- the direct nonlinear analogue of ``Lambda``'s bandwidth.
    ``in_dim`` = 1 for signals, 2 for images.
    """
    torch = _require_torch()
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)
    B = torch.randn(n_features, in_dim, generator=g) * scale  # fixed feature frequencies

    class _FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("B", B)
            mlp: list = []
            in_dim = 2 * n_features
            for _ in range(layers):
                mlp += [nn.Linear(in_dim, hidden), nn.ReLU()]
                in_dim = hidden
            mlp += [nn.Linear(in_dim, 1)]
            self.mlp = nn.Sequential(*mlp)

        def forward(self, x):
            proj = 2 * np.pi * x @ self.B.T
            feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            return self.mlp(feats)

    return _FFN()


def train_inr(model, t: np.ndarray, y: np.ndarray, epochs: int = 2000, lr: float = 1e-4,
              device: str = "cpu", weight_decay: float = 0.0, verbose: bool = False):
    """Fit a torch INR to real samples ``(t, y)`` by full-batch Adam on MSE.

    Returns ``(model, history)`` with the per-epoch loss. ``t`` should be in ``[0,1)``.
    """
    torch = _require_torch()
    model = model.to(device)
    ta = np.asarray(t, float)
    if ta.ndim == 1:
        ta = ta.reshape(-1, 1)
    tt = torch.tensor(ta, dtype=torch.float32, device=device)
    yy = torch.tensor(np.asarray(y, float).reshape(-1, 1), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(tt)
        loss = torch.mean((pred - yy) ** 2)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"epoch {ep:5d}  loss {loss.item():.3e}", flush=True)
    return model, hist
