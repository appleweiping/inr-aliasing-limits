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
    "bandwidth_matched_freqs",
    "nonuniform_periodogram",
    "SIREN",
    "FourierFeatureMLP",
    "train_inr",
    "torch_available",
]


class FixedFeatureINR:
    r"""Fixed Fourier-feature INR (linear output layer). Exactly the theory-core model.

    Parameters
    ----------
    freqs : the representable frequency set :math:`\Lambda`.
    real  : if True, model a real signal (coefficients paired so the waveform is real).
    """

    def __init__(self, freqs: np.ndarray, real: bool = False):
        self.freqs = np.asarray(freqs, float)
        self.real = real
        self.coeffs_: np.ndarray | None = None

    def fit(self, t: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> "FixedFeatureINR":
        Phi = synthesis_matrix(self.freqs, t)
        yv = np.asarray(y)
        if self.real and not np.iscomplexobj(yv):
            yv = yv.astype(complex)
        self.coeffs_ = pinv_apply(Phi, yv, ridge=ridge)
        return self

    def predict(self, t: np.ndarray) -> np.ndarray:
        assert self.coeffs_ is not None, "call fit() first"
        v = synthesis_matrix(self.freqs, t) @ self.coeffs_
        return v.real if self.real else v

    def noise_gain(self, t: np.ndarray) -> float:
        return noise_gain(synthesis_matrix(self.freqs, t))


def nonuniform_periodogram(t: np.ndarray, y: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
    r"""Least-squares (Lomb-Scargle-style) periodogram of nonuniform samples on ``freq_grid``.

    Returns the squared magnitude of the per-frequency least-squares correlation, usable to
    estimate a signal's essential band from irregular samples.
    """
    t = np.asarray(t, float)
    y = np.asarray(y)
    freq_grid = np.asarray(freq_grid, float)
    E = np.exp(-1j * 2 * np.pi * np.outer(freq_grid, t))  # (F, N)
    corr = (E @ y) / t.size
    return np.abs(corr) ** 2


def bandwidth_matched_freqs(
    t: np.ndarray,
    y: np.ndarray,
    max_bandwidth: int,
    energy_keep: float = 0.99,
    real: bool = True,
) -> np.ndarray:
    r"""Estimate the essential band from samples and return an integer dictionary covering it.

    Computes a nonuniform periodogram on the integer grid up to ``max_bandwidth``, keeps the
    smallest symmetric band capturing ``energy_keep`` of the periodogram energy, and returns
    the integer lowpass dictionary of that bandwidth (Hermitian-symmetric for real signals).
    """
    grid = np.arange(0, max_bandwidth + 1).astype(float)
    P = nonuniform_periodogram(t, y, grid)
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
                      layers: int = 3, seed: int = 0):  # noqa: N802
    """Random-Fourier-feature input mapping + ReLU MLP (Tancik et al.). Returns ``nn.Module``."""
    torch = _require_torch()
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)
    B = torch.randn(n_features, 1, generator=g) * scale  # fixed feature frequencies

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
    tt = torch.tensor(np.asarray(t, float).reshape(-1, 1), dtype=torch.float32, device=device)
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
