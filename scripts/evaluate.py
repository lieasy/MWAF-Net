"""Evaluate one locally trained MWAF-Net on its prepared test partition."""

import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from mwaf.data import load_prepared
from mwaf.engine import Segments, evaluate_loader, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    model, checkpoint, device = load_checkpoint(a.checkpoint, a.device)
    arrays, meta = load_prepared(a.data_dir, checkpoint["task"])
    result = evaluate_loader(
        model, DataLoader(Segments(*arrays["test"]), batch_size=a.batch_size), device
    )
    result["class_names"] = meta["class_names"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
