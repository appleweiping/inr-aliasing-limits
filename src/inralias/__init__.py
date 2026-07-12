"""inr-aliasing-limits (inralias)

Silent aliasing in fixed Fourier-feature coordinate models.

A coordinate model with a *fixed* finite Fourier frequency set ``Lambda`` and linear
coefficients can only represent signals whose spectrum lies in ``Lambda``.  This package
provides:

* the sampling machinery on a finite frequency dictionary (``sampling``),
* the identifiability theory core -- visibility, aliasability, exact equivalence
  classes, function-space (Riesz) conversion, random/jitter sampling concentration, and
  detection power (``identifiability``; Theorems T1--T4 of the paper),
* background least-squares error identities on the dictionary (``limits``),
* the linear fixed-feature coordinate model (exact theory core) and trained nonlinear
  networks used for empirical extrapolation only (``inr``),
* a ground-truth-free aliasing diagnostic with an explicit null-calibration API
  (``diagnostics``),
* synthetic + real signal generators/loaders (``signals``).

Scope: the theorems cover the fixed-feature linear-coefficient model.  Trained nonlinear
SIREN / Fourier-feature MLPs appear only as empirical extrapolation; adaptive/learned
frequency sets are outside the theory.  Every stated closed form is asserted against
Monte Carlo simulation in the test suite.
"""

from inralias.sampling import (
    synthesis_matrix,
    gram,
    frame_bounds,
    noise_gain,
    alias_projector,
)
from inralias.limits import (
    aliasing_bias,
    aliasing_floor,
    folded_frequency,
    ls_coefficient_mse,
    bayes_coefficient_mmse,
    aliasing_variance_term,
    silent_aliasing_ratio,
)
from inralias.identifiability import (
    sample_atom,
    visibility,
    aliasability,
    alias_coefficients,
    grid_equivalence_class,
    exactly_indistinguishable,
    continuous_gram,
    riesz_bounds,
    function_error_decomposition,
    expected_jitter_coherence,
    coherence_epsilon,
    aliasability_concentration_bound,
    residual_test_power,
)

__all__ = [
    "synthesis_matrix",
    "gram",
    "frame_bounds",
    "noise_gain",
    "alias_projector",
    "aliasing_bias",
    "aliasing_floor",
    "folded_frequency",
    "ls_coefficient_mse",
    "bayes_coefficient_mmse",
    "aliasing_variance_term",
    "silent_aliasing_ratio",
    "sample_atom",
    "visibility",
    "aliasability",
    "alias_coefficients",
    "grid_equivalence_class",
    "exactly_indistinguishable",
    "continuous_gram",
    "riesz_bounds",
    "function_error_decomposition",
    "expected_jitter_coherence",
    "coherence_epsilon",
    "aliasability_concentration_bound",
    "residual_test_power",
]

__version__ = "0.2.0"
