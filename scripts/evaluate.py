import argparse
import json
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
    p.add_argument("--output_json", type=Path, help="Optional metrics file for run-level aggregation")
    a = p.parse_args()
    if a.batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    if a.output_json is not None and a.output_json.exists():
        raise FileExistsError(a.output_json)
    model, checkpoint, device = load_checkpoint(a.checkpoint, a.device)
    arrays, meta = load_prepared(a.data_dir, checkpoint["task"])
    expected_hashes = checkpoint.get("dataset_hashes")
    if expected_hashes and expected_hashes != meta.get("files"):
        raise ValueError("Prepared dataset differs from the checkpoint's training/validation split")
    result = evaluate_loader(model, DataLoader(Segments(*arrays["test"]), batch_size=a.batch_size), device)
    result.update(task=checkpoint["task"], class_names=meta["class_names"], split="test",
                  checkpoint=str(a.checkpoint), replicas=len(checkpoint["members"]),
                  replica_seeds=checkpoint.get("replica_seeds", [m["seed"] for m in checkpoint["members"]]),
                  class_bias=checkpoint["class_bias"], calibration_status=checkpoint["calibration"]["status"])
    text = json.dumps(result, indent=2)
    if a.output_json is not None:
        a.output_json.parent.mkdir(parents=True, exist_ok=True)
        a.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
