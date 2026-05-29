# Reproducibility and Review-Stage Release Plan

## Current Public Scope

Because the manuscript is still under review, this repository is designed as a
review-stage public package. It documents the parts needed to understand and
recreate the experimental protocol without prematurely exposing the proposed
method.

Public now:

- data preparation script;
- dataset structure and task definitions;
- file-level split protocol;
- preprocessing parameters;
- environment dependencies;
- high-level training and evaluation settings.

Withheld until acceptance:

- proposed model implementation;
- ablation variants that reveal the core innovation;
- full batch-running scripts for the proposed method;
- unpublished result tables and figure source files;
- trained checkpoints.

## Recommended Options

### Option A: Public Skeleton + Private Reviewer Archive

Use this repository as the public page. Keep the full implementation in a
private repository or private branch. If reviewers request code, provide a
fixed private archive with a short access window.

This is usually the best balance for an article under review.

### Option B: Public Baselines Only

Release data preparation plus standard baseline models only. Add the proposed
model after acceptance. This can make the repository more runnable before
acceptance, but requires careful cleanup to avoid leaking ablation details.

### Option C: Full Anonymous Release

If the venue requires full reproducibility during review, prepare an anonymized
private repository or archived package. Remove author names, local paths,
unpublished notes, and any nonessential history.

## After Acceptance Checklist

1. Add the proposed model implementation.
2. Add the exact training entry points used in the paper.
3. Add repeated-run seeds and final hyperparameter files.
4. Add scripts to reproduce tables and figures.
5. Add expected result ranges for sanity checks.
6. Tag the paper release, for example `v1.0-paper`.
7. Archive the release with a DOI if required.

## Files to Avoid Publishing Accidentally

- raw dataset files and labels without permission;
- `.npy` processed datasets;
- checkpoints: `.pt`, `.pth`, `.ckpt`;
- local logs and result folders;
- notebooks containing unpublished figures or notes;
- `__pycache__` and temporary files;
- private reviewer comments or manuscript drafts.

