# Main-network implementation notes

This package extracts only the optimized 32/64/128-channel six-branch main
MWAF-Net and the modules required to run it. Comparator networks, historical
commented models, component variants, feature-plot methods, and experiment
analysis code are absent. Output classes are the only architectural task choice.

## Preserved archived behavior

The branch, fusion, attention, residual-block, classifier, and kernel-generator
calculations follow the archived main implementation. The model-specific global
initializer is retained: it applies Kaiming initialization to `Conv1d` weights,
including those initially generated from wavelet templates. Consequently those
generated coefficients are overwritten during construction. Fixing that behavior
would be a distinct training change and is not part of this release.

The stored scale/shift parameters are used during initial kernel generation only;
forward passes use the discrete convolution weights. This public code does not
claim differentiable scale/shift-based kernel regeneration. The archived
coordinate-attention and DropBlock calculations are retained as implemented.

The focal-style loss uses `pt = exp(-smoothed_cross_entropy)`, with the archived
gamma and smoothing defaults. It is not identical to a hard-label focal loss
when smoothing is nonzero. The original optimizer is Adam. The public defaults
use inverse-frequency sampling rather than additionally injecting class weights
into the focal objective.

## Interface and runtime changes

- `MWAFNet(num_classes=...)` has a conventional tensor-in/tensor-out `forward`.
  It can be moved between CPU and CUDA with `.to(device)`.
- Loss, optimization, and data loading are separate from network construction.
- The standalone trainer selects checkpoints by validation UAR and always saves
  the first finite model, even if validation UAR is zero.
- The final partial gradient-accumulation window is updated. The historical
  loader-driven training method could discard that window; it is not copied.
- A singleton training remainder is dropped to avoid BatchNorm errors. This
  decision is recorded in the locally generated effective configuration.
- Checkpoints carry class order, task, and preprocessing settings. Test evaluation
  and WAV prediction reuse those settings rather than guessing them.
- Local outputs are protected against accidental overwrite.

The public configurations are practical main-model defaults. They are not the
complete historical paper experiment suite or a promise of identical research
scores. Task-specific ensembles/calibration and comparisons remain unpublished
in this package. No paper result is stored in code, configuration, or documentation.
