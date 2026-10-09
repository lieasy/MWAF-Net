"""Train only the main MWAF-Net from locally prepared data."""

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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    a = p.parse_args()
    config = read_config(a.config)
    if a.epochs is not None:
        config["training"]["epochs"] = a.epochs
    if a.batch_size is not None:
        config["training"]["batch_size"] = a.batch_size
    print(train(config, a.data_dir, a.output_dir, a.seed, a.device))


if __name__ == "__main__":
    main()
