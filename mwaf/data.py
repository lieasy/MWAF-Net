"""Download-independent preprocessing and leakage-checked recording splits."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np

TASKS = {
    "clean_binary": ["Normal", "Abnormal"],
    "noise_binary": ["Normal", "Abnormal"],
    "clean_multiclass": ["Normal", "ASD", "PDA", "PFO", "VSD"],
    "circor_outcome": ["Normal", "Abnormal"],
    "circor_murmur": ["Absent", "Present", "Unknown"],
}
DEFAULT_PATHS = {
    "clean_binary": ("raw_data/clean Heartsound Data", "raw_data/Clean_label_used.csv"),
    "noise_binary": (
        "raw_data/Noise Heartsound Data Details",
        "raw_data/Noise_label_used.csv",
    ),
    "clean_multiclass": (
        "raw_data/clean Heartsound Data",
        "raw_data/Clean_label_multi_class.csv",
    ),
    "circor_outcome": (
        "raw_data/circor/training_data",
        "raw_data/circor/training_data.csv",
    ),
    "circor_murmur": (
        "raw_data/circor/training_data",
        "raw_data/circor/training_data.csv",
    ),
}


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_id(value, names):
    raw = str(value).strip()
    lookup = {name.upper(): i for i, name in enumerate(names)}
    if raw.upper() in lookup:
        return lookup[raw.upper()]
    # Diagnostic labels can also supply the binary ZCHSound task.
    if names == ["Normal", "Abnormal"] and raw.upper() in {"ASD", "PDA", "PFO", "VSD"}:
        return 1
    try:
        integer = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown label {raw!r}; expected {names} or zero-based IDs"
        ) from exc
    if not 0 <= integer < len(names):
        raise ValueError(f"Label {integer} outside [0, {len(names)-1}]")
    return integer


def collect_zch(audio_dir, label_file, task):
    audio_dir, label_file = Path(audio_dir), Path(label_file)
    with label_file.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Label CSV is empty")
    header = [v.strip().lower() for v in rows[0]]
    aliases = {"filename", "file_name", "file", "recording", "recording_id"}
    if aliases.intersection(header):
        file_col = next(i for i, x in enumerate(header) if x in aliases)
        label_col = next(
            (i for i, x in enumerate(header) if x in {"label", "diagnosis"}), None
        )
        if label_col is None:
            raise ValueError("CSV header must contain label or diagnosis")
        subject_col = next(
            (i for i, x in enumerate(header) if x in {"subject_id", "patient_id"}), None
        )
        rows = rows[1:]
    else:
        file_col, label_col, subject_col = 0, 1, None
    records, seen = [], set()
    for row in rows:
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) <= max(file_col, label_col):
            raise ValueError(f"Incomplete label row: {row}")
        filename = row[file_col].strip()
        if not Path(filename).suffix:
            filename += ".wav"
        path = (audio_dir / filename).resolve()
        if not path.is_relative_to(audio_dir.resolve()):
            raise ValueError("Audio filename must stay inside audio_dir")
        if not path.is_file():
            raise FileNotFoundError(path)
        if filename in seen:
            raise ValueError(f"Duplicate label row for {filename}")
        seen.add(filename)
        subject = row[subject_col].strip() if subject_col is not None else path.stem
        if not subject:
            raise ValueError(f"Empty subject ID for {filename}")
        records.append(
            dict(
                path=path,
                recording_id=filename,
                subject_id=subject,
                label=label_id(row[label_col], TASKS[task]),
            )
        )
    if not records:
        raise ValueError("No labelled recordings found")
    return sorted(records, key=lambda x: x["recording_id"])


def collect_circor(audio_dir, metadata_csv, task):
    """Group visits using the official CSV's Patient ID and Additional ID."""
    audio_dir = Path(audio_dir)
    with Path(metadata_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "Patient ID" not in rows[0] or "Additional ID" not in rows[0]:
        raise ValueError(
            "Use the official CirCor CSV with Patient ID and Additional ID"
        )
    field = "Outcome" if task == "circor_outcome" else "Murmur"
    groups = Union()
    seen = set()
    for row in rows:
        patient = row["Patient ID"].strip()
        if not patient or patient in seen:
            raise ValueError("Missing or duplicated CirCor Patient ID")
        seen.add(patient)
        groups.find(patient)
        extra = row["Additional ID"].strip()
        if extra and extra.lower() not in {"nan", "none", "na"}:
            if extra.endswith(".0"):
                extra = extra[:-2]
            groups.join(patient, extra)
    records = []
    for row in rows:
        patient = row["Patient ID"].strip()
        lines = (
            (audio_dir / f"{patient}.txt").read_text(encoding="utf-8-sig").splitlines()
        )
        first = lines[0].split()
        if first[0] != patient:
            raise ValueError(f"Patient header mismatch for {patient}")
        count = int(first[1])
        if count <= 0 or len(lines) < count + 1:
            raise ValueError(f"Invalid recording count for {patient}")
        target = label_id(row[field], TASKS[task])
        for line in lines[1 : count + 1]:
            waves = [v for v in line.split() if v.lower().endswith(".wav")]
            if len(waves) != 1:
                raise ValueError(f"Cannot find one WAV in {patient} header: {line}")
            path = (audio_dir / waves[0]).resolve()
            if not path.is_relative_to(audio_dir.resolve()):
                raise ValueError("CirCor WAV must stay inside audio_dir")
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                dict(
                    path=path,
                    recording_id=waves[0],
                    subject_id=groups.find(patient),
                    label=target,
                )
            )
    if len({r["recording_id"] for r in records}) != len(records):
        raise ValueError("CirCor headers contain duplicated recording names")
    return sorted(records, key=lambda x: x["recording_id"])


def bind_groups(records):
    """Also prevent byte-identical recordings from crossing partitions."""
    groups, identical = Union(), {}
    for record in records:
        subject = record["subject_id"]
        groups.find(subject)
        digest = file_sha(record["path"])
        record["raw_sha256"] = digest
        if digest in identical:
            groups.join(subject, identical[digest])
        else:
            identical[digest] = subject
    for record in records:
        record["group_id"] = groups.find(record["subject_id"])
    return records


def split_records(records, seed):
    from sklearn.model_selection import train_test_split

    labels = defaultdict(list)
    for r in records:
        labels[r["group_id"]].append(r["label"])
    ids = np.array(sorted(labels))
    # Linked visits with differing labels stay together; ties use the lowest ID.
    targets = np.array(
        [
            min(Counter(labels[k]), key=lambda v: (-Counter(labels[k])[v], v))
            for k in ids
        ]
    )
    try:
        train, hold, _, hold_y = train_test_split(
            ids, targets, test_size=0.2, stratify=targets, random_state=seed
        )
        val, test = train_test_split(
            hold, test_size=0.5, stratify=hold_y, random_state=seed
        )
        assignment = {
            k: name
            for name, keys in [("train", train), ("val", val), ("test", test)]
            for k in keys
        }
        return assignment, "two-stage stratified 80/10/10 group split"
    except ValueError:
        # A rare class with seven groups cannot always survive a second stratified split.
        rng = np.random.default_rng(seed)
        assignment = {}
        for label in sorted(set(targets)):
            keys = ids[targets == label].copy()
            if len(keys) < 3:
                raise ValueError(
                    f"Class {label} needs at least three independent groups for train/val/test"
                )
            rng.shuffle(keys)
            n_val = max(1, int(round(0.1 * len(keys))))
            n_test = max(1, int(round(0.1 * len(keys))))
            for name, part in [
                ("val", keys[:n_val]),
                ("test", keys[n_val : n_val + n_test]),
                ("train", keys[n_val + n_test :]),
            ]:
                assignment.update({k: name for k in part})
        warnings.warn(
            "Rare classes use deterministic per-class group allocation; proportions are approximate",
            stacklevel=2,
        )
        return (
            assignment,
            "rare-class allocation with at least one group per class per partition",
        )


def replay_manifest(path, records):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["recording_id"]: r for r in records}
    assignment, seen = {}, set()
    for row in rows:
        rid, split = row["recording_id"], row["split"]
        if rid not in by_id or rid in seen or split not in {"train", "val", "test"}:
            raise ValueError(
                "Manifest has unknown/duplicated recordings or invalid partition names"
            )
        r = by_id[rid]
        if row.get("label") is not None and int(row["label"]) != r["label"]:
            raise ValueError(f"Label changed since the split manifest: {rid}")
        if row.get("raw_sha256") and row["raw_sha256"] != r["raw_sha256"]:
            raise ValueError(f"Audio changed since the split manifest: {rid}")
        group = r["group_id"]
        if group in assignment and assignment[group] != split:
            raise ValueError(f"Subject or duplicate-audio leakage in manifest: {group}")
        assignment[group] = split
        seen.add(rid)
    if seen != set(by_id):
        raise ValueError("Manifest must cover every input recording exactly once")
    return assignment


def audio_segments(path, target_sr=1000, segment_seconds=1.5, min_seconds=1.0):
    import librosa

    audio, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    size = int(round(target_sr * segment_seconds))
    if len(audio) < target_sr * min_seconds or len(audio) < size:
        return np.empty((0, size), dtype=np.float32)
    segments = audio[: len(audio) // size * size].reshape(-1, size).astype(np.float32)
    if not np.isfinite(segments).all():
        raise ValueError(f"Nonfinite samples in {path}")
    return segments


def segment_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def prepare(
    task,
    audio_dir,
    label_file,
    output_dir,
    seed=42,
    split_manifest=None,
    global_normalize=True,
):
    if task not in TASKS:
        raise ValueError(task)
    records = (
        collect_circor(audio_dir, label_file, task)
        if task.startswith("circor_")
        else collect_zch(audio_dir, label_file, task)
    )
    records = bind_groups(records)
    if split_manifest:
        assignment, strategy = (
            replay_manifest(split_manifest, records),
            "replayed immutable manifest",
        )
    else:
        assignment, strategy = split_records(records, seed)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Choose a new/empty output directory to preserve previous splits"
        )
    arrays, manifest, statistics = {}, [], {}
    for split in ["train", "val", "test"]:
        xs, ys, subjects, recording_ids = [], [], [], []
        selected = [r for r in records if assignment[r["group_id"]] == split]
        for r in selected:
            x = audio_segments(r["path"])
            row = {
                k: r[k]
                for k in [
                    "recording_id",
                    "subject_id",
                    "group_id",
                    "label",
                    "raw_sha256",
                ]
            }
            row.update(split=split, segments=len(x))
            manifest.append(row)
            if len(x):
                xs.append(x)
                ys.extend([r["label"]] * len(x))
                subjects.extend([r["subject_id"]] * len(x))
                recording_ids.extend([r["recording_id"]] * len(x))
        if not xs or set(ys) != set(range(len(TASKS[task]))):
            raise ValueError(
                f"{split} has no full segments for one or more classes; inspect short recordings"
            )
        arrays[split] = (
            np.concatenate(xs),
            np.asarray(ys, dtype=np.int64),
            np.asarray(subjects),
            np.asarray(recording_ids),
        )
        statistics[split] = dict(
            groups=len({r["group_id"] for r in selected}),
            recordings=len(selected),
            segments=len(ys),
            class_counts=np.bincount(ys, minlength=len(TASKS[task])).tolist(),
        )
    normalisation = None
    if global_normalize:
        mean, std = float(arrays["train"][0].mean()), float(
            arrays["train"][0].std() + 1e-8
        )
        normalisation = dict(method="train-split mean/std", mean=mean, std=std)
        for split, (x, *rest) in arrays.items():
            arrays[split] = ((x - mean) / std, *rest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in arrays.items():
        for suffix, value in zip(
            ["data", "labels", "patient_ids", "recording_ids"], values
        ):
            np.save(output_dir / f"{split}_{suffix}.npy", value, allow_pickle=False)
    with (output_dir / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    metadata = dict(
        task=task,
        seed=seed,
        class_names=TASKS[task],
        target_sr=1000,
        segment_seconds=1.5,
        min_seconds=1.0,
        samples_per_segment=1500,
        split_strategy=strategy,
        global_normalization=normalisation,
        model_input_normalization="per-segment mean/std",
        statistics=statistics,
        files={p.name: file_sha(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_prepared(path, expected_task=None):
    path = Path(path)
    metadata = json.loads((path / "dataset_metadata.json").read_text(encoding="utf-8"))
    task = metadata["task"]
    if (
        task not in TASKS
        or metadata["class_names"] != TASKS[task]
        or (expected_task and task != expected_task)
    ):
        raise ValueError(
            "Prepared task/class mapping does not match the selected configuration"
        )
    arrays, subjects, recordings = {}, {}, {}
    for split in ["train", "val", "test"]:
        x = np.load(path / f"{split}_data.npy", allow_pickle=False)
        y = np.load(path / f"{split}_labels.npy", allow_pickle=False)
        ids = np.load(path / f"{split}_patient_ids.npy", allow_pickle=False)
        rids = np.load(path / f"{split}_recording_ids.npy", allow_pickle=False)
        if (
            x.ndim != 2
            or x.shape[1] != 1500
            or y.shape != (len(x),)
            or ids.shape != y.shape
            or rids.shape != y.shape
        ):
            raise ValueError(f"Invalid prepared array shapes in {split}")
        if (
            not len(x)
            or not np.isfinite(x).all()
            or y.dtype.kind not in "iu"
            or set(y.tolist()) != set(range(len(TASKS[task])))
        ):
            raise ValueError(f"Invalid samples/labels or missing class in {split}")
        subjects[split], recordings[split] = set(ids.tolist()), set(rids.tolist())
        arrays[split] = (segment_normalize(x), y)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if subjects[a] & subjects[b] or recordings[a] & recordings[b]:
            raise ValueError(f"Data leakage between {a} and {b}")
    for name, digest in metadata.get("files", {}).items():
        target = path / name
        if target.parent.resolve() != path.resolve() or file_sha(target) != digest:
            raise ValueError(f"Prepared data changed: {name}")
    return arrays, metadata
