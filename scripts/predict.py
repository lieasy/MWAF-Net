"""Predict a local WAV with a locally trained MWAF-Net checkpoint."""

import argparse, csv, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mwaf.data import audio_segments, segment_normalize
from mwaf.engine import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    if a.output_csv.exists():
        raise FileExistsError(a.output_csv)
    model, checkpoint, device = load_checkpoint(a.checkpoint, a.device)
    pre = checkpoint["preprocessing"]
    x = audio_segments(
        a.audio, pre["target_sr"], pre["segment_seconds"], pre["min_seconds"]
    )
    if not len(x):
        raise ValueError("WAV has no complete 1.5-second segment")
    normalisation = pre["global_normalization"]
    if normalisation:
        x = (x - normalisation["mean"]) / normalisation["std"]
    x = segment_normalize(x)
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), a.batch_size):
            tensor = torch.from_numpy(x[i : i + a.batch_size]).unsqueeze(1).to(device)
            probs.append(model(tensor).softmax(1).cpu().numpy())
    probs = np.concatenate(probs)
    names = checkpoint["class_names"]
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["segment_index", "start_seconds", "predicted_class"]
            + ["prob_" + n for n in names]
        )
        for i, row in enumerate(probs):
            writer.writerow(
                [i, i * pre["segment_seconds"], names[int(row.argmax())], *row.tolist()]
            )
    print(
        json.dumps(
            {
                "segments": len(x),
                "class_names": names,
                "mean_segment_probabilities": probs.mean(0).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
