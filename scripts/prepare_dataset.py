"""Prepare downloaded local data; no dataset is redistributed."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mwaf.data import DEFAULT_PATHS, TASKS, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--audio_dir", type=Path)
    parser.add_argument(
        "--label_file",
        type=Path,
        help="ZCHSound label CSV or official CirCor training_data.csv",
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_manifest",
        type=Path,
        help="Replay an existing local recording-to-partition CSV",
    )
    parser.add_argument("--no_global_normalize", action="store_true")
    args = parser.parse_args()

    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else args.project_root / p

    audio, label = DEFAULT_PATHS[args.task]
    meta = prepare(
        args.task,
        resolve(args.audio_dir or audio),
        resolve(args.label_file or label),
        resolve(args.output_dir or f"dataset/{args.task}_segment"),
        args.seed,
        resolve(args.split_manifest) if args.split_manifest else None,
        not args.no_global_normalize,
    )
    print(
        json.dumps(
            {
                "task": meta["task"],
                "split_strategy": meta["split_strategy"],
                "statistics": meta["statistics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
