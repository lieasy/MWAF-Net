"""Synthetic protocol checks; these are not experiments on clinical data."""

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn

from mwaf.calibration import (apply_bias, class_balanced_nll,
                              fit_validation_bias, selection_nll)
from mwaf.data import TASKS
from mwaf.engine import (ModelEMA, ProbabilityEnsemble, WarmupCosinePlateau,
                         hierarchical_sampling_weights, load_checkpoint,
                         read_config, resolve_config, train)
from mwaf.loss import FocalLoss, SampleMeanCrossEntropy, inverse_frequency_alpha


PROJECT = Path(__file__).resolve().parents[1]


class TinyNet(nn.Module):
    def __init__(self, num_classes, width=4):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool1d(width)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.fc(self.pool(x).squeeze(1))


def synthetic_prepared(root, task="circor_outcome"):
    """Produce complete arrays/manifest with strictly separated artificial IDs."""
    root.mkdir()
    rng = np.random.default_rng(11)
    rows = []
    for split, nsegments in [("train", 3), ("val", 2), ("test", 1)]:
        xs, labels, subjects, recordings = [], [], [], []
        for c in range(len(TASKS[task])):
            for i in range(3):
                subject = f"{split}_patient_{c}_{i}"
                recording = f"{subject}_AV.wav"
                xs.extend(rng.normal(0, 1, (nsegments, 1500)).astype(np.float32))
                labels.extend([c] * nsegments)
                subjects.extend([subject] * nsegments)
                recordings.extend([recording] * nsegments)
                rows.append(dict(recording_id=recording, subject_id=subject,
                                 group_id=subject, label=c, raw_sha256="synthetic",
                                 split=split, segments=nsegments))
        for suffix, value in [("data", xs), ("labels", labels), ("patient_ids", subjects), ("recording_ids", recordings)]:
            np.save(root / f"{split}_{suffix}.npy", np.asarray(value), allow_pickle=False)
    with (root / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(task=task, seed=12, class_names=TASKS[task], target_sr=1000,
                segment_seconds=1.5, min_seconds=1.0, global_normalization=None,
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})
    (root / "dataset_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_manuscript_defaults_and_circor_protocol_parity(self):
        zch = read_config(PROJECT / "configs/clean_binary.yaml")["training"]
        self.assertEqual((zch["optimizer"], zch["weight_decay"], zch["warmup_epochs"], zch["grad_accum_steps"]),
                         ("adamw", 1e-4, 10, 4))
        self.assertEqual((zch["mixup_alpha"], zch["mixup_probability"], zch["patience"]), (0.4, 0.6, 20))
        outcome = read_config(PROJECT / "configs/circor_outcome.yaml")["training"]
        murmur = read_config(PROJECT / "configs/circor_murmur.yaml")["training"]
        self.assertEqual(outcome["selection_unit"], "segment")
        self.assertEqual(murmur["selection_unit"], "recording")
        self.assertEqual({k: v for k, v in outcome.items() if k != "selection_unit"},
                         {k: v for k, v in murmur.items() if k != "selection_unit"})
        self.assertEqual((outcome["replicas"], outcome["epochs"], outcome["min_epochs"], outcome["ema_decay"]),
                         (2, 70, 25, 0.99))

    def test_focal_equation_and_inverse_class_weights(self):
        y = torch.tensor([0, 0, 0, 1])
        alpha = inverse_frequency_alpha(y.numpy(), 2)
        np.testing.assert_allclose(alpha, [2 / 3, 2])
        logits = torch.tensor([[2.0, -0.3], [0.4, 0.2], [-1.0, 0.7], [0.1, 0.8]], requires_grad=True)
        criterion = FocalLoss(2, gamma=2.5, alpha=alpha, label_smoothing=0.2, reduction="none")
        smooth = torch.nn.functional.one_hot(y, 2) * 0.8 + 0.1
        ce = -(smooth * logits.log_softmax(1)).sum(1)
        expected = torch.tensor(alpha, dtype=torch.float32)[y] * (1 - torch.exp(-ce)).pow(2.5) * ce
        torch.testing.assert_close(criterion(logits, y), expected)
        criterion(logits, y).mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIn("alpha", dict(criterion.named_buffers()))

    def test_hierarchical_sampler_balances_subjects_and_recordings(self):
        y = np.array([0] * 11 + [1] * 5)
        subjects = np.array(["A"] * 10 + ["B"] + ["C"] * 5)
        recordings = np.array(["A1"] * 8 + ["A2"] * 2 + ["B1"] + ["C1"] * 5)
        w = hierarchical_sampling_weights(y, subjects, recordings)
        self.assertAlmostEqual(w[y == 0].sum(), 0.5)
        self.assertAlmostEqual(w[subjects == "A"].sum(), 0.25)
        self.assertAlmostEqual(w[subjects == "B"].sum(), 0.25)
        self.assertAlmostEqual(w[recordings == "A1"].sum(), 0.125)
        self.assertAlmostEqual(w[recordings == "A2"].sum(), 0.125)
        self.assertAlmostEqual(w[recordings == "C1"].sum(), 0.5)

    def test_weighted_ce_accumulation_matches_a_complete_sample_mean(self):
        logits = torch.tensor([[1.0, 0.2], [0.1, 1.2], [0.8, 0.1], [0.9, 0.4]], requires_grad=True)
        y = torch.tensor([0, 1, 0, 0])
        loss = SampleMeanCrossEntropy(2, alpha=[2 / 3, 2], label_smoothing=0.03)
        whole = loss(logits, y)
        pieces = loss(logits[:3], y[:3]) * 3 / 4 + loss(logits[3:], y[3:]) / 4
        torch.testing.assert_close(whole, pieces)
        torch.testing.assert_close(torch.autograd.grad(whole, logits, retain_graph=True)[0],
                                   torch.autograd.grad(pieces, logits)[0])

    def test_class_balanced_nll_uses_the_requested_unit(self):
        p = np.array([[0.9, 0.1], [0.9, 0.1], [0.3, 0.7], [0.1, 0.9]])
        y = np.array([0, 0, 0, 1])
        rids = np.array(["long", "long", "short", "positive"])
        segment = (-np.mean(np.log([0.9, 0.9, 0.3])) - np.log(0.9)) / 2
        recording = (-np.mean(np.log([0.9, 0.3])) - np.log(0.9)) / 2
        self.assertAlmostEqual(selection_nll(p, y, rids, "segment"), segment)
        self.assertAlmostEqual(selection_nll(p, y, rids, "recording"), recording)
        self.assertNotAlmostEqual(segment, recording)
        with self.assertRaisesRegex(ValueError, "every class"):
            class_balanced_nll(p[:2], y[:2])

    def test_ema_updates_floats_and_copies_counters(self):
        model = nn.BatchNorm1d(2)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.75)
        with torch.no_grad():
            model.weight.fill_(2)
            model.running_mean.fill_(4)
            model.num_batches_tracked.fill_(7)
        ema.update(model)
        torch.testing.assert_close(ema.model.weight, torch.full((2,), 0.5))
        torch.testing.assert_close(ema.model.running_mean, torch.ones(2))
        self.assertEqual(int(ema.model.num_batches_tracked), 7)
        self.assertFalse(ema.model.training)
        self.assertFalse(ema.model.weight.requires_grad)

    def test_plateau_reduction_survives_next_cosine_step(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        t = resolve_config(dict(task="clean_binary", training=dict(epochs=20, plateau_patience=2)))["training"]
        schedule = WarmupCosinePlateau(optimizer, t)
        self.assertAlmostEqual(schedule.start_epoch(1), 0.0001)
        schedule.observe(0.5)
        schedule.observe(0.5)
        schedule.observe(0.4)
        self.assertEqual(schedule.scale, 0.5)
        self.assertAlmostEqual(schedule.start_epoch(10), 0.0005)
        baseline = WarmupCosinePlateau(optimizer, t)
        self.assertAlmostEqual(schedule.start_epoch(11), baseline.start_epoch(11) / 2)
        self.assertAlmostEqual(schedule.start_epoch(20), 1e-6)

    def test_equal_probability_ensemble_and_frozen_bias(self):
        class Constant(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.num_classes = 2
                self.register_buffer("logits", torch.tensor(p).log())
            def forward(self, x):
                return self.logits.expand(len(x), -1)
        ensemble = ProbabilityEnsemble([Constant([0.9, 0.1]), Constant([0.2, 0.8])], [0.2, -0.2])
        actual = ensemble(torch.zeros(2, 1, 10)).softmax(1).detach().numpy()
        expected = apply_bias(np.array([[0.55, 0.45], [0.55, 0.45]]), [0.2, -0.2])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_calibration_is_grouped_bounded_and_validation_only(self):
        y = np.tile([0, 1], 9)
        groups = np.repeat([f"patient_{i}" for i in range(9)], 2)
        rids = np.array([f"recording_{i}" for i in range(len(y))])
        p = np.tile([0.9, 0.1], (len(y), 1))
        bias, report = fit_validation_bias(p, y, groups, rids, source_split="val", seed=3)
        self.assertEqual(report["status"], "selected")
        self.assertEqual(len(report["folds"]), 3)
        self.assertTrue(np.all(np.abs(bias) <= 1.5))
        self.assertLess(class_balanced_nll(apply_bias(p, bias), y), class_balanced_nll(p, y))
        for row in report["folds"]:
            self.assertEqual(row["fit_groups"] + row["heldout_groups"], 9)
        with self.assertRaisesRegex(ValueError, "val only"):
            fit_validation_bias(p, y, groups, rids, source_split="test")
        with self.assertRaisesRegex(ValueError, "val only"):
            fit_validation_bias(p, y, groups, rids, source_split="train")

    def test_calibration_prefers_no_bias_and_does_not_relax_grouping(self):
        y = np.tile([0, 1], 9)
        groups = np.repeat(np.arange(9), 2)
        rids = np.arange(len(y))
        p = np.array([[0.8, 0.2] if c == 0 else [0.2, 0.8] for c in y])
        bias, report = fit_validation_bias(p, y, groups, rids, source_split="val")
        np.testing.assert_array_equal(bias, [0, 0])
        self.assertIsNone(report["selected_penalty"])
        bias, report = fit_validation_bias(p, y, np.zeros(len(y)), rids, source_split="val")
        self.assertEqual(report["status"], "no_bias_insufficient_class_groups")
        np.testing.assert_array_equal(bias, [0, 0])

    def test_end_to_end_shared_protocol_with_custom_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "prepared"
            synthetic_prepared(data)
            config = resolve_config(dict(task="circor_outcome", model=dict(width=4),
                                         training=dict(epochs=2, min_epochs=1, batch_size=4,
                                                       segments_per_recording=2)))
            with patch("mwaf.engine.fit_validation_bias", wraps=fit_validation_bias) as calibration:
                path = train(config, data, root / "run", seed=123, device="cpu", model_factory=TinyNet)
            args, kwargs = calibration.call_args
            self.assertTrue(all(str(group).startswith("val_") for group in args[2]))
            self.assertEqual(kwargs["source_split"], "val")
            model, checkpoint, device = load_checkpoint(path, "cpu", model_factory=TinyNet)
            self.assertEqual(len(checkpoint["members"]), 2)
            self.assertEqual(checkpoint["replica_seeds"], [123, 124])
            self.assertTrue(all(member["ema"] for member in checkpoint["members"]))
            self.assertTrue(all(member["selection_metric"] == "balanced_nll" for member in checkpoint["members"]))
            self.assertFalse(torch.equal(checkpoint["members"][0]["state_dict"]["fc.weight"],
                                         checkpoint["members"][1]["state_dict"]["fc.weight"]))
            p = model(torch.randn(3, 1, 1500)).softmax(1)
            torch.testing.assert_close(p.sum(1), torch.ones(3))
            self.assertEqual(p.shape, (3, 2))
            with self.assertRaisesRegex(ValueError, "matching model_factory"):
                load_checkpoint(path, "cpu")
            with self.assertRaises(FileExistsError):
                train(config, data, root / "run", device="cpu", model_factory=TinyNet)

    def test_murmur_uses_recording_selection_and_preserves_all_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_prepared(root / "prepared", "circor_murmur")
            config = resolve_config(dict(task="circor_murmur", model=dict(width=4),
                                         training=dict(epochs=1, batch_size=4, segments_per_recording=1)))
            path = train(config, root / "prepared", root / "run", device="cpu", model_factory=TinyNet)
            _, checkpoint, _ = load_checkpoint(path, "cpu", model_factory=TinyNet)
            self.assertEqual(checkpoint["num_classes"], 3)
            self.assertEqual(checkpoint["calibration"]["unit"], "recording")
            self.assertTrue(all(m["selection_unit"] == "recording" for m in checkpoint["members"]))
            effective = json.loads((root / "run/replica_1/effective_run.json").read_text())
            self.assertEqual(effective["nominal_samples_per_epoch"], 9)
            self.assertEqual(effective["actual_samples_per_epoch"], 8)

    def test_cli_train_evaluate_predict_and_old_checkpoint_rejection(self):
        # A small-channel actual MWAF-Net verifies the complete CLI/checkpoint
        # route quickly; model tests independently cover the full-width model.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_prepared(root / "prepared", "clean_binary")
            config = dict(task="clean_binary", model=dict(branch_channels=[6, 6, 12],
                          post_channels=12, classifier_channels=6),
                          training=dict(epochs=1, batch_size=4))
            import yaml
            config_file = root / "smoke.yaml"
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            def run_cli(script, *args):
                result = subprocess.run([sys.executable, str(PROJECT / "scripts" / script), *map(str, args)],
                                        env=env, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            run_cli("train.py", "--config", config_file, "--data_dir", root / "prepared",
                    "--output_dir", root / "run", "--device", "cpu")
            checkpoint_path = root / "run/best.pt"
            text = run_cli("evaluate.py", "--checkpoint", checkpoint_path, "--data_dir", root / "prepared",
                           "--device", "cpu", "--output_json", root / "metrics.json")
            result = json.loads(text)
            self.assertEqual(result["samples"], 6)
            self.assertEqual(result["replicas"], 1)
            self.assertEqual(json.loads((root / "metrics.json").read_text()), result)
            audio = root / "synthetic.wav"
            sf.write(audio, np.random.default_rng(2).normal(0, 0.02, 3000), 1000)
            result = json.loads(run_cli("predict.py", "--checkpoint", checkpoint_path,
                                       "--audio", audio, "--output_csv", root / "predictions.csv", "--device", "cpu"))
            self.assertEqual(result["segments"], 2)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint["members"].append(copy.deepcopy(checkpoint["members"][0]))
            checkpoint["replica_seeds"] = [42, 42]
            torch.save(checkpoint, root / "synthetic_ensemble.pt")
            result = json.loads(run_cli("evaluate.py", "--checkpoint", root / "synthetic_ensemble.pt",
                                       "--data_dir", root / "prepared", "--device", "cpu"))
            self.assertEqual(result["replicas"], 2)
            torch.save(dict(state_dict={}), root / "old.pt")
            with self.assertRaisesRegex(ValueError, "archived/incompatible"):
                load_checkpoint(root / "old.pt", "cpu")


if __name__ == "__main__":
    unittest.main()
