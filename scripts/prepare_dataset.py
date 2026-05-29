import argparse
import json
import os
from pathlib import Path


DEFAULT_TASKS = {
    "clean_binary": {
        "audio_dir": "raw_data/clean Heartsound Data",
        "label_file": "raw_data/Clean_label_used.csv",
        "output_dir": "dataset/clean_binary_segment",
        "label_mode": "numeric",
    },
    "noise_binary": {
        "audio_dir": "raw_data/Noise Heartsound Data Details",
        "label_file": "raw_data/Noise_label_used.csv",
        "output_dir": "dataset/noise_binary_segment",
        "label_mode": "numeric",
    },
    "clean_multiclass": {
        "audio_dir": "raw_data/clean Heartsound Data",
        "label_file": "raw_data/Clean_label_multi_class.csv",
        "output_dir": "dataset/clean_multiclass_segment",
        "label_mode": "multiclass",
    },
}

MULTICLASS_MAPPING = {
    "NORMAL": 0,
    "ASD": 1,
    "PDA": 2,
    "PFO": 3,
    "VSD": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test npy files for ZCHSound experiments."
    )
    parser.add_argument(
        "--task",
        choices=sorted(DEFAULT_TASKS),
        required=True,
        help="Dataset preset to build.",
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing raw_data and dataset directories.",
    )
    parser.add_argument("--audio_dir", type=Path, default=None)
    parser.add_argument("--label_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_sr", type=int, default=1000)
    parser.add_argument("--segment_seconds", type=float, default=1.5)
    parser.add_argument("--min_seconds", type=float, default=1.0)
    parser.add_argument(
        "--no_global_normalize",
        action="store_true",
        help="Skip train-set mean/std normalization before saving npy files.",
    )
    return parser.parse_args()


def import_dependencies():
    global np, pd, train_test_split
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "numpy, pandas, and scikit-learn are required to prepare datasets. "
            "Install them with: pip install numpy pandas scikit-learn"
        ) from exc


def resolve_path(project_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def read_labels(label_file):
    """Read common ZCHSound CSV formats into filename -> raw label."""
    df = pd.read_csv(label_file)
    lower_cols = {col.lower(): col for col in df.columns}

    if "filename" in lower_cols:
        filename_col = lower_cols.get("filename")
        label_col = lower_cols.get("diagnosis") or lower_cols.get("label")
    elif "filename" in df.columns:
        filename_col = "filename"
        label_col = "label" if "label" in df.columns else df.columns[1]
    elif "fileName" in df.columns:
        filename_col = "fileName"
        label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
    else:
        df = pd.read_csv(label_file, header=None, names=["filename", "label"])
        filename_col = "filename"
        label_col = "label"

    return dict(zip(df[filename_col].astype(str), df[label_col]))


def convert_label(raw_label, mode):
    if mode == "numeric":
        return int(raw_label)
    if mode == "multiclass":
        if isinstance(raw_label, str):
            key = raw_label.strip().upper()
            if key in MULTICLASS_MAPPING:
                return MULTICLASS_MAPPING[key]
        return int(raw_label)
    raise ValueError(f"Unsupported label mode: {mode}")


def collect_audio_files(audio_dir, label_file, label_mode):
    labels = read_labels(label_file)
    records = []
    for audio_path in sorted(audio_dir.glob("*.wav")):
        if audio_path.name not in labels:
            continue
        try:
            label = convert_label(labels[audio_path.name], label_mode)
        except (TypeError, ValueError):
            continue
        records.append((audio_path, label))
    if not records:
        raise RuntimeError(f"No labeled wav files found in {audio_dir}")
    return records


def split_records(records, seed):
    paths = np.array([str(path) for path, _ in records])
    labels = np.array([label for _, label in records], dtype=np.int64)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def segment_audio(paths, labels, target_sr, samples_per_segment, min_samples):
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required to build datasets from wav files. "
            "Install it with: pip install librosa"
        ) from exc

    segments = []
    segment_labels = []
    skipped = 0

    for audio_path, label in zip(paths, labels):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except Exception as exc:
            print(f"[WARN] Failed to read {audio_path}: {exc}")
            skipped += 1
            continue

        if len(y) < min_samples:
            skipped += 1
            continue

        segment_count = len(y) // samples_per_segment
        for idx in range(segment_count):
            start = idx * samples_per_segment
            stop = start + samples_per_segment
            segments.append(y[start:stop])
            segment_labels.append(label)

    data = np.asarray(segments, dtype=np.float32)
    y = np.asarray(segment_labels, dtype=np.int64)
    return data, y, skipped


def save_split(output_dir, split_name, data, labels):
    np.save(output_dir / f"{split_name}_data.npy", data)
    np.save(output_dir / f"{split_name}_labels.npy", labels)


def main():
    args = parse_args()
    import_dependencies()

    preset = DEFAULT_TASKS[args.task]
    project_root = args.project_root.resolve()

    audio_dir = resolve_path(project_root, args.audio_dir or preset["audio_dir"])
    label_file = resolve_path(project_root, args.label_file or preset["label_file"])
    output_dir = resolve_path(project_root, args.output_dir or preset["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_per_segment = int(args.target_sr * args.segment_seconds)
    min_samples = int(args.target_sr * args.min_seconds)

    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Audio dir: {audio_dir}")
    print(f"[INFO] Label file: {label_file}")
    print(f"[INFO] Output dir: {output_dir}")

    records = collect_audio_files(audio_dir, label_file, preset["label_mode"])
    labels = np.array([label for _, label in records], dtype=np.int64)
    print(f"[INFO] Files: {len(records)} | file-level labels: {np.bincount(labels)}")

    split_map = split_records(records, args.seed)
    outputs = {}
    for split_name, (paths, split_labels) in split_map.items():
        data, y, skipped = segment_audio(
            paths,
            split_labels,
            args.target_sr,
            samples_per_segment,
            min_samples,
        )
        outputs[split_name] = [data, y]
        print(
            f"[INFO] {split_name}: files={len(paths)}, segments={len(data)}, "
            f"skipped_files={skipped}, labels={np.bincount(y) if len(y) else []}"
        )

    stats = {}
    if not args.no_global_normalize:
        train_data = outputs["train"][0]
        if len(train_data) == 0:
            raise RuntimeError("Training split produced no segments.")
        mean = float(train_data.mean())
        std = float(train_data.std() + 1e-8)
        for split_name in outputs:
            outputs[split_name][0] = (outputs[split_name][0] - mean) / std
        stats = {"mean": mean, "std": std}

    for split_name, (data, y) in outputs.items():
        save_split(output_dir, split_name, data, y)

    metadata = {
        "task": args.task,
        "seed": args.seed,
        "target_sr": args.target_sr,
        "segment_seconds": args.segment_seconds,
        "min_seconds": args.min_seconds,
        "samples_per_segment": samples_per_segment,
        "audio_dir": str(audio_dir),
        "label_file": str(label_file),
        "class_mapping": MULTICLASS_MAPPING if preset["label_mode"] == "multiclass" else None,
        "normalization": None if args.no_global_normalize else stats,
    }
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[SUCCESS] Dataset saved.")


if __name__ == "__main__":
    main()
