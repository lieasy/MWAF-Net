# Local dataset inputs

Download ZCHSound from its [project website](http://zchsound.ncrcch.org.cn/) or
[dataset repository](https://github.com/WeiJieOvO/ZCHSound-Dataset), and CirCor
from [PhysioNet 1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/).
Refer to each provider for dataset attribution and terms.

No recordings, clinical labels, metadata CSVs, processed arrays, or real split
manifests are included in this repository.

## Default directory layout

```text
raw_data/
  clean Heartsound Data/          ZCHSound high-quality WAVs
  Noise Heartsound Data Details/  ZCHSound naturally low-quality WAVs
  Clean_label_used.csv            high-quality binary labels
  Noise_label_used.csv            low-quality binary labels
  Clean_label_multi_class.csv     high-quality diagnostic labels
  circor/
    training_data.csv             official metadata, including Additional ID
    training_data/                patient TXT headers and WAVs
```

These are path conventions, not bundled files. Specify `--audio_dir` and
`--label_file` for different download layouts. Do not rename labels or remap IDs
independently between partitions.

## ZCHSound CSVs

A headerless two-column CSV is supported, as is a header with `filename` (or
`fileName`) and `label` (or `diagnosis`). Values can be the class names below or
zero-based integer IDs. An optional `subject_id`/`patient_id` column binds related
recordings. Otherwise the acquisition-file stem identifies the participant, as
in the study's ZCHSound cohort.

| Task | Class order / IDs |
| --- | --- |
| clean_binary | Normal=0, Abnormal=1 |
| noise_binary | Normal=0, Abnormal=1 |
| clean_multiclass | Normal=0, ASD=1, PDA=2, PFO=3, VSD=4 |

The binary tasks also accept named ASD/PDA/PFO/VSD diagnoses and map them to
Abnormal. Labels for missing WAVs, unknown classes, empty explicit subject IDs,
and duplicate recording rows raise errors rather than being silently ignored.

## CirCor inputs and grouping

The official CSV must contain `Patient ID`, `Additional ID`, and the chosen
`Outcome` or `Murmur` column. The script reads each listed patient TXT header
to find its WAVs. All auscultation sites inherit the CSV's patient-level target.
The Murmur task here uses patient-level Murmur labels for every site; it does
not redefine them as site-specific murmur-presence labels.

| Task | Class order / IDs |
| --- | --- |
| circor_outcome | Normal=0, Abnormal=1 |
| circor_murmur | Absent=0, Present=1, Unknown=2 |

`Additional ID` links repeated visits before any partitioning. Links are
transitive. If linked visits have different labels, the group majority determines
stratification (ties use the lowest class ID); each recording retains its own
target. Identical audio hashes further bind groups. All group members stay in one
partition regardless of the number of recording sites or segments.

## Local generated files

Each partition has `data.npy`, `labels.npy`, `patient_ids.npy`, and
`recording_ids.npy` with a `train_`, `val_`, or `test_` prefix. Waveform arrays
are `[segments,1500]`. The split manifest contains recording IDs, subject/group
IDs, labels, hashes, partition names, and complete-segment counts. The metadata
records preprocessing, class order, split strategy, and generated-file hashes.

The manifest and arrays are private local data products, excluded by `.gitignore`.
Replaying a manifest rejects changed WAVs, missing/extra recordings, and related
groups assigned to multiple partitions. The loader also checks data integrity,
class mapping, and subject/recording overlap.

No full 1.5-second segment is created from recordings shorter than 1.5 seconds.
Such recordings are reported with zero segments in the manifest. If a partition
then lacks any class, preparation fails instead of silently changing the split.
