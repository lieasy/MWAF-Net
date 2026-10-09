# MWAF-Net: Multi-Wavelet Attention Fusion Network for Pediatric Heart Sound Classification

This public source package includes local-data preprocessing, reproducible data
partitioning, and the optimized six-branch main MWAF-Net with training, evaluation,
and WAV inference entry points. It supports the three ZCHSound tasks and the
CirCor Outcome and Murmur tasks.

The repository contains **software, configuration, and usage documentation only**.
Download the datasets separately. Raw audio, clinical labels, prepared arrays,
recording manifests, trained weights, logs, predictions, unpublished scores,
comparison models, component variants, and experiment/figure-generation packages
are not distributed here.

## Package contents

```text
MWAF-Net-main/
  mwaf/             main network, loss, preprocessing, training utilities
  scripts/          prepare_dataset.py, train.py, evaluate.py, predict.py
  configs/          five task configurations
  docs/             dataset and implementation notes
  tests/            source-only tests using temporary synthetic data
  requirements.txt
  .gitignore
```

## Environment

Python 3.10-3.12 is supported. Create an isolated environment, then install
PyTorch for your hardware using the [official PyTorch version guide](https://docs.pytorch.org/get-started/previous-versions/).
For example, the archived PyTorch 2.3.0 CPU build is installed with:

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.1, use the `https://download.pytorch.org/whl/cu121` index instead.
The validation environment used Python 3.12 and PyTorch 2.3.0 CPU. NumPy is
limited to version 1.x for compatibility with that archived PyTorch build.
No torchvision, plotting library, or statistical-analysis package is required.

## Download data

- ZCHSound: [dataset project website](http://zchsound.ncrcch.org.cn/) and
  [dataset repository/download entry](https://github.com/WeiJieOvO/ZCHSound-Dataset).
- CirCor: [PhysioNet, version 1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/).

See [dataset notes](docs/DATASET.md) for directory structure and label formats.
Keep the downloaded data and all generated files outside Git or under the ignored
local directories. This export contains no dataset files or example patient records.

## Prepare and split data

Run from this repository's root. The original ZCHSound task names remain available:

```bash
python scripts/prepare_dataset.py --task clean_binary --project_root . --seed 42
python scripts/prepare_dataset.py --task noise_binary --project_root . --seed 42
python scripts/prepare_dataset.py --task clean_multiclass --project_root . --seed 42
python scripts/prepare_dataset.py --task circor_outcome --project_root . --seed 42
python scripts/prepare_dataset.py --task circor_murmur --project_root . --seed 42
```

Use `--audio_dir`, `--label_file`, and `--output_dir` to select your own local
paths. Quote paths containing spaces. Outputs default to
`dataset/<task>_segment/`; use a separate output directory for each split seed.
Outputs are created locally and are not part of this repository.

Partitioning precedes segmentation. ZCHSound acquisition files represent distinct
participants in this study; an optional CSV `subject_id` column can group related
recordings explicitly. CirCor groups recording sites and visits using `Patient ID`
and `Additional ID` from its official CSV. Byte-identical WAV files are also kept
in one partition. Group overlap is rejected.

The usual partition is a two-stage, class-stratified 80/10/10 group split.
For very small classes, the script uses a deterministic per-class allocation with
at least one group in each partition and reports that the proportions are approximate.
Audio is resampled to 1,000 Hz with librosa and cut into complete, non-overlapping
1.5-second segments. Incomplete tails are discarded.

The archived preparation step applies training-split global mean/std; model input
then uses per-segment mean/std normalization. `--no_global_normalize` omits only
the global step. No validation/test statistics are fitted for normalization.

The local `split_manifest.csv` and `dataset_metadata.json` record assignments and
input hashes. Replay the same partition with:

```bash
python scripts/prepare_dataset.py --task clean_binary --audio_dir "path/to/wavs" --label_file "path/to/labels.csv" --split_manifest "local/split_manifest.csv" --output_dir "dataset/replayed_split"
```

## Train the main network

```bash
python scripts/train.py --config configs/clean_binary.yaml --data_dir dataset/clean_binary_segment --output_dir runs/clean_binary_seed42 --seed 42
```

For another task, replace both the configuration and prepared-data paths with the
matching task name. `--device cpu` forces CPU execution; the default selects CUDA
when available. All configuration paths and output paths are portable local paths.

The provided defaults use Adam, learning rate 0.001, batch size 16, weighted
training sampling, the archived augmentation/Mixup and smoothed focal-style loss,
warmup and cosine scheduling, gradient accumulation, and validation-UAR early
stopping. They are runnable public defaults rather than the complete set of
task-specific paper experiment configurations. No published score is promised by
this source-only release.

`best.pt`, the effective configuration, and training history are produced **only
when the user runs training**. They are ignored by Git and absent from this export.
Choose a new output directory for each run; existing outputs are not overwritten.

## Test or predict with your own checkpoint

```bash
python scripts/evaluate.py --checkpoint runs/clean_binary_seed42/best.pt --data_dir dataset/clean_binary_segment
python scripts/predict.py --checkpoint runs/clean_binary_seed42/best.pt --audio "path/to/recording.wav" --output_csv "runs/local_predictions.csv"
```

Evaluation reports the single model's test metrics to the console. Prediction
saves per-segment probabilities to the specified local CSV. The displayed mean
segment probabilities are a simple aggregation, not a clinically validated
patient diagnosis. No pretrained model or saved prediction is shipped.

## Model behavior and scope

The six branches use Morlet, db4, sym4, coif4, Haar, and Mexican Hat templates,
32/64/128 channels, learned global branch weights, feature refinement, and a
256-to-128 classifier. Output classes are configurable as 2, 3, or 5.

**Archived initialization behavior is retained intentionally**: the global
`Conv1d` initializer overwrites the coefficients generated during wavelet setup.
Scale/shift parameters are stored by the original layer but do not regenerate
kernels during forward passes. See [implementation notes](docs/IMPLEMENTATION.md).
This update does not silently repair or reinterpret the archived experiment model.

Comparison models, component-removal variants, template controls, calibration,
task-specific ensembles, noise-study suites, statistics, and paper figures remain
outside this public package. The code is the single main-network workflow.

## Software tests

```bash
python -m unittest discover -s tests -v
```

Tests create temporary synthetic recordings and arrays, exercise partition leakage
checks, and validate main-network gradients and checkpoint reload. They do not
use clinical data or reproduce manuscript experiments.
