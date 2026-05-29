# MWAF-Net: Multi-Wavelet Attention Fusion Network for Robust Heart Sound Classificatio

This repository is a review-stage reproducibility package for heart sound
classification experiments on the ZCHSound dataset.

The full model implementation and paper-specific experimental code are
temporarily withheld because the manuscript is still under peer review. This
public package provides the dataset preparation protocol, environment
requirements, split strategy, experiment settings, and release plan. The full
training code will be added after acceptance or can be shared with reviewers
through a private/restricted channel if required by the journal or conference.

## What Is Included

- `scripts/prepare_dataset.py`: build reproducible train/validation/test `.npy`
  files from raw ZCHSound wav files and label CSV files.
- `configs/experiment_settings.yaml`: documented experiment settings used for
  clean binary, noisy binary, and clean multi-class tasks.
- `requirements.txt`: Python dependencies for data preparation, training, and
  analysis.
- `docs/DATASET.md`: dataset source, expected local structure, labels, and
  split protocol.
- `docs/REPRODUCIBILITY.md`: recommended release strategy while the paper is
  under review.

## What Is Temporarily Withheld

The following items are intentionally not included in this review-stage public
package:

- the proposed model source code;
- ablation-study implementation details;
- trained checkpoints and logs;
- paper figure source data generated from unpublished experiments;
- scripts that reveal the main innovation before peer review is complete.

This keeps the project useful for reproducibility review while protecting the
novel contribution until the manuscript decision is finalized.

## Dataset Source

The experiments use the ZCHSound heart sound dataset. The raw data are expected
to be obtained from the dataset owner or the release channel specified in the
paper. This repository does not redistribute raw audio files or clinical labels.

Expected local structure:

```text
project_root/
  raw_data/
    clean Heartsound Data/
    Noise Heartsound Data Details/
    Clean_label_used.csv
    Noise_label_used.csv
    Clean_label_multi_class.csv
```

See `docs/DATASET.md` for details.

## Environment

Recommended environment:

- Python 3.10 or 3.11
- PyTorch 2.x
- CUDA-enabled GPU for full training, CPU is sufficient for dataset preparation
- Windows or Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from
the official PyTorch installation page before installing the remaining
dependencies.

## Data Preparation

Run from the project root:

```bash
python scripts/prepare_dataset.py --task clean_binary --project_root .
python scripts/prepare_dataset.py --task noise_binary --project_root .
python scripts/prepare_dataset.py --task clean_multiclass --project_root .
```

Default outputs:

```text
dataset/
  clean_binary_segment/
  noise_binary_segment/
  clean_multiclass_segment/
```

Each output directory contains:

- `train_data.npy`, `train_labels.npy`
- `val_data.npy`, `val_labels.npy`
- `test_data.npy`, `test_labels.npy`
- `dataset_metadata.json`

## Dataset Split

The split is performed at file level, not segment level, to avoid leakage
between train/validation/test sets.

- train: 80%
- validation: 10%
- test: 10%
- random seed: 42 by default
- stratified by class label
- target sampling rate: 1000 Hz
- segment length: 1.5 seconds
- normalization: train-set mean/std applied to all splits

## Experiment Settings

The documented settings are in `configs/experiment_settings.yaml`.

Main settings:

- optimizer: Adam
- learning rate: 0.001
- weight decay: 1e-6
- epochs: 200
- batch size: 16
- early stopping patience: 50
- gradient clipping: 0.5
- mixup alpha: 0.2
- primary metrics: accuracy, UAR, weighted F1, confusion matrix

## Suggested Review-Stage Workflow

1. Keep this repository public with data preparation and settings.
2. Keep the proposed model code in a private repository or private branch.
3. Create a fixed private archive for reviewers if required.
4. After acceptance, merge the withheld implementation into this repository and
   tag the release used by the paper.
5. Optionally archive the final code and metadata on Zenodo or another DOI
   service.

## Citation

Citation information will be added after the manuscript is accepted.

