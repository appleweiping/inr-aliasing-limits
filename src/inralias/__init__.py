"""inr-aliasing-limits (inralias)

Fundamental aliasing limits of implicit neural representations (INRs).

An INR (Fourier-feature / SIREN coordinate network) can only represent signals whose
spectrum lies in a *structured* representable frequency set ``Lambda`` induced by its
feature frequencies and width/depth. This package provides:

* the sampling/frame machinery on a finite frequency dictionary (``sampling``),
* the closed-form achievability (noise gain) and converse (aliasing floor, folded
  frequency, "silent aliasing" bias) limits (``limits``),
* the linear fixed-feature INR (exact least-squares theory core) and trained nonlinear
  INRs used in experiments (``inr``),
* a ground-truth-free aliasing diagnostic (``diagnostics``),
* synthetic + real signal generators/loaders (``signals``).

Every theorem in the paper is asserted against Monte-Carlo simulation in the test suite
before it is used in the manuscript.
"""

from inralias.sampling import (
    synthesis_matrix,
    gram,
    frame_bounds,
    noise_gain,
    alias_projector,
    landau_min_samples,
)
from inralias.limits import (
    aliasing_bias,
    aliasing_floor,
    folded_frequency,
    mmse_recoverable,
    mmse_aliasing_floor,
    silent_aliasing_ratio,
)

__all__ = [
    "synthesis_matrix",
    "gram",
    "frame_bounds",
    "noise_gain",
    "alias_projector",
    "landau_min_samples",
    "aliasing_bias",
    "aliasing_floor",
    "folded_frequency",
    "mmse_recoverable",
    "mmse_aliasing_floor",
    "silent_aliasing_ratio",
]

__version__ = "0.1.0"
