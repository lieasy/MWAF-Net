"""Train MWAF-Net using the selected ZCH or shared CirCor protocol."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mwaf.engine import read_config, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42, help="Training base seed; prepared split is not changed")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, help="Override maximum epochs (short runs are only smoke tests)")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--replicas", type=int, help="Override replicas; published CirCor protocol uses two")
    a = p.parse_args()
    config = read_config(a.config)
    overrides = {key: getattr(a, key) for key in ["epochs", "batch_size", "replicas"] if getattr(a, key) is not None}
    config["training"].update(overrides)
    config["command_line_overrides"] = overrides
    print(train(config, a.data_dir, a.output_dir, a.seed, a.device))


if __name__ == "__main__":
    main()
