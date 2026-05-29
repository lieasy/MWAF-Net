# Dataset Notes

## Source

The experiments are based on the ZCHSound heart sound dataset. Raw audio and
label files should be obtained from the dataset owner or from the official
access channel described in the manuscript.

This public repository does not include raw audio, label CSV files, processed
`.npy` arrays, or any clinical metadata.

## Expected Files

Place the dataset under the repository or project root:

```text
raw_data/
  clean Heartsound Data/
  Noise Heartsound Data Details/
  Clean_label_used.csv
  Noise_label_used.csv
  Clean_label_multi_class.csv
```

Task presets:

| Task | Audio directory | Label file | Output |
| --- | --- | --- | --- |
| `clean_binary` | `raw_data/clean Heartsound Data` | `raw_data/Clean_label_used.csv` | `dataset/clean_binary_segment` |
| `noise_binary` | `raw_data/Noise Heartsound Data Details` | `raw_data/Noise_label_used.csv` | `dataset/noise_binary_segment` |
| `clean_multiclass` | `raw_data/clean Heartsound Data` | `raw_data/Clean_label_multi_class.csv` | `dataset/clean_multiclass_segment` |

## Multi-Class Labels

The multi-class task uses this mapping:

| Class | ID |
| --- | --- |
| NORMAL | 0 |
| ASD | 1 |
| PDA | 2 |
| PFO | 3 |
| VSD | 4 |

## Split Protocol

The split is performed before segmentation, at the original wav-file level:

- 80% train
- 10% validation
- 10% test
- stratified by label
- seed 42 unless otherwise specified

This prevents segments from the same original recording from appearing in
different splits.

## Preprocessing

The data preparation script performs:

1. Read label CSV files.
2. Match labels to wav filenames.
3. Stratified file-level train/validation/test split.
4. Load and resample audio to 1000 Hz.
5. Cut non-overlapping 1.5-second segments.
6. Normalize all splits using train-set mean and standard deviation.
7. Save arrays and `dataset_metadata.json`.

## Commands

```bash
python scripts/prepare_dataset.py --task clean_binary --project_root .
python scripts/prepare_dataset.py --task noise_binary --project_root .
python scripts/prepare_dataset.py --task clean_multiclass --project_root .
```

