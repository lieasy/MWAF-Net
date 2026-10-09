"""Temporary synthetic inputs only; no real patient records are stored here."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from mwaf.data import (
    bind_groups,
    collect_circor,
    load_prepared,
    prepare,
    replay_manifest,
    split_records,
)


class DataTests(unittest.TestCase):
    def test_preprocessing_and_manifest_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio"
            audio.mkdir()
            labels = root / "labels.csv"
            rng = np.random.default_rng(12)
            with labels.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "label"])
                for c in range(2):
                    for i in range(20):
                        name = f"synthetic_{c}_{i}.wav"
                        sf.write(audio / name, rng.normal(0, 0.03, 6400), 2000)
                        writer.writerow([name, c])
            a = root / "first"
            b = root / "replay"
            meta = prepare("clean_binary", audio, labels, a, 42)
            self.assertEqual(meta["samples_per_segment"], 1500)
            arrays, _ = load_prepared(a, "clean_binary")
            for x, y in arrays.values():
                self.assertEqual(x.shape[1], 1500)
                self.assertLess(abs(float(x.mean())), 1e-5)
                self.assertEqual(set(y), {0, 1})
            prepare("clean_binary", audio, labels, b, 999, a / "split_manifest.csv")
            for name in ["train_data.npy", "val_data.npy", "test_data.npy"]:
                np.testing.assert_array_equal(np.load(a / name), np.load(b / name))
            bad = np.load(a / "test_patient_ids.npy")
            bad[0] = np.load(a / "train_patient_ids.npy")[0]
            np.save(a / "test_patient_ids.npy", bad)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_prepared(a)

    def test_linked_circor_visits_and_duplicate_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "training_data.csv"
            rng = np.random.default_rng(13)
            with metadata.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Patient ID", "Additional ID", "Outcome", "Murmur"])
                for i, additional in [(1, "2"), (2, ""), (3, "")]:
                    writer.writerow([i, additional, "Normal", "Absent"])
                    (root / f"{i}.txt").write_text(
                        f"{i} 2 2000\nAV {i}_AV.hea {i}_AV.wav\nMV {i}_MV.hea {i}_MV.wav\n",
                        encoding="utf-8",
                    )
                    for site in ["AV", "MV"]:
                        sf.write(
                            root / f"{i}_{site}.wav", rng.normal(0, 0.02, 3000), 1000
                        )
            records = collect_circor(root, metadata, "circor_outcome")
            linked = {
                r["subject_id"]
                for r in records
                if r["recording_id"].startswith(("1_", "2_"))
            }
            self.assertEqual(len(linked), 1)
            (root / "3_AV.wav").write_bytes((root / "1_AV.wav").read_bytes())
            bind_groups(records)
            self.assertEqual(len({r["group_id"] for r in records}), 1)
            manifest = root / "bad.csv"
            with manifest.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["recording_id", "split"])
                writer.writeheader()
                for i, r in enumerate(records):
                    writer.writerow(
                        dict(
                            recording_id=r["recording_id"],
                            split="train" if i == 0 else "test",
                        )
                    )
            with self.assertRaisesRegex(ValueError, "leakage"):
                replay_manifest(manifest, records)

    def test_rare_class_group_allocation(self):
        records = [
            dict(group_id=f"class_{c}_{i}", label=c)
            for c, n in [(0, 20), (1, 3)]
            for i in range(n)
        ]
        assignment, strategy = split_records(records, 8)
        self.assertIn("rare-class", strategy)
        for c in range(2):
            self.assertEqual(
                {assignment[r["group_id"]] for r in records if r["label"] == c},
                {"train", "val", "test"},
            )
        self.assertEqual(assignment, split_records(records, 8)[0])


if __name__ == "__main__":
    unittest.main()
