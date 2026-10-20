"""Manuscript-aligned training, validation selection, and frozen inference.

The CirCor protocol is shared by model factories: nothing in this engine grants
MWAF-Net or component C5 an extra training or inference strategy.
"""

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .calibration import (class_balanced_nll, fit_validation_bias,
                          selection_nll)
from .data import TASKS, load_prepared
from .loss import (FocalLoss, SampleMeanCrossEntropy, inverse_frequency_alpha,
                   mixup_data, mixup_criterion)
from .model import MWAFNet, MODEL_CONFIG_VERSION


DEFAULT_TRAINING = dict(
    epochs=200, min_epochs=1, replicas=1, batch_size=16,
    optimizer="adamw", learning_rate=0.001, weight_decay=0.0001,
    betas=[0.9, 0.999], scheduler="warmup_cosine", warmup_epochs=10,
    min_learning_rate=1e-6, plateau_factor=0.5, plateau_patience=8,
    grad_accum_steps=4, grad_clip_norm=1.0,
    mixup_alpha=0.4, mixup_probability=0.6, patience=20,
    loss="smoothed_focal", gamma=2.5, label_smoothing=0.2,
    class_weighting="inverse_frequency", train_augmentation=False,
    sampling="shuffle", segments_per_recording=None,
    selection="uar", selection_unit="segment", ema_decay=None,
    num_workers=0,
)
CIRCOR_TRAINING = {
    **DEFAULT_TRAINING,
    "epochs": 70, "min_epochs": 25, "replicas": 2, "batch_size": 64,
    "learning_rate": 5e-4, "weight_decay": 1e-3,
    "scheduler": "constant", "warmup_epochs": 0, "plateau_patience": None,
    "grad_accum_steps": 1, "grad_clip_norm": None,
    "mixup_alpha": 0.15, "mixup_probability": 0.3, "patience": 18,
    "loss": "cross_entropy", "label_smoothing": 0.03,
    "class_weighting": "none", "sampling": "hierarchical",
    "segments_per_recording": 4, "selection": "balanced_nll", "ema_decay": 0.99,
}
DEFAULT_CALIBRATION = dict(enabled=False, folds=3, penalties=[1.0, 0.3, 0.1, 0.03],
                           bound=1.5)
CHECKPOINT_FORMAT = 2


def resolve_config(config):
    config = copy.deepcopy(config)
    task = config.get("task")
    if task not in TASKS:
        raise ValueError("Configuration must specify one supported task")
    circor = task.startswith("circor_")
    defaults = copy.deepcopy(CIRCOR_TRAINING if circor else DEFAULT_TRAINING)
    if task == "circor_murmur":
        defaults["selection_unit"] = "recording"
    unknown = set(config.get("training", {})) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}; use the revised configuration schema")
    unknown_calibration = set(config.get("calibration", {})) - set(DEFAULT_CALIBRATION)
    if unknown_calibration:
        raise ValueError(f"Unknown calibration settings: {sorted(unknown_calibration)}")
    config.setdefault("protocol", "circor_shared" if circor else "zch_main")
    config["training"] = {**defaults, **config.get("training", {})}
    config["calibration"] = {**DEFAULT_CALIBRATION, "enabled": circor,
                             **config.get("calibration", {})}
    config.setdefault("model", {})
    return config


def read_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return resolve_config(config)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class Segments(Dataset):
    def __init__(self, x, y, augmentation=False):
        self.x, self.y, self.augmentation = x, y, augmentation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index].copy()
        if self.augmentation:
            if random.random() < 0.5:
                start = random.randint(0, 200)
                x = np.pad(x[start:start + 1300], (100, 100), mode="reflect")
            if random.random() < 0.3:
                x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
            if random.random() < 0.4:
                x *= random.uniform(0.8, 1.2)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(
            self.y[index], dtype=torch.long
        )


def get_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu")
    return device


def metrics(y, predicted, num_classes):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    if not len(y) or y.shape != predicted.shape:
        raise ValueError("Predictions and labels must be aligned and nonempty")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, predicted), 1)
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(num_classes), where=cm.sum(1) > 0)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(num_classes), where=cm.sum(0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(num_classes), where=(precision + recall) > 0)
    result = dict(accuracy_percent=float(np.trace(cm) / cm.sum() * 100),
                  uar_percent=float(recall.mean() * 100),
                  class_recall_percent=(recall * 100).tolist(),
                  class_precision_percent=(precision * 100).tolist(),
                  class_f1_percent=(f1 * 100).tolist(),
                  confusion_counts=cm.tolist(), samples=int(cm.sum()),
                  prediction_unit="segment")
    if num_classes == 2:
        result.update(positive_class_index=1, precision_percent=float(precision[1] * 100),
                      f1_percent=float(f1[1] * 100))
    return result


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    model.eval()
    ys, probabilities = [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if logits.ndim != 2 or logits.shape != (len(y), model.num_classes):
            raise ValueError("The model must return [batch, num_classes] logits")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite model outputs")
        if criterion is not None:
            total_loss += float(criterion(logits, y)) * len(y)
        probabilities.append(logits.softmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    if not ys:
        raise ValueError("Cannot evaluate an empty loader")
    labels = np.concatenate(ys)
    return np.concatenate(probabilities), labels, total_loss / len(labels)


def evaluate_loader(model, loader, device, criterion=None):
    probabilities, labels, loss = predict_loader(model, loader, device, criterion)
    result = metrics(labels, probabilities.argmax(1), model.num_classes)
    result["segment_class_balanced_nll"] = class_balanced_nll(probabilities, labels)
    if criterion is not None:
        result["loss"] = loss
    return result


def hierarchical_sampling_weights(labels, patient_ids, recording_ids):
    """P(class) P(patient|class) P(recording|patient,class) P(segment|recording).

    Each conditional distribution is uniform. Unlike inverse segment-frequency
    weights alone, long files and subjects with many files do not dominate.
    A subject with distinct labels across visits is balanced within each class.
    """
    y = np.asarray(labels, dtype=int)
    subjects, recordings = np.asarray(patient_ids).astype(str), np.asarray(recording_ids).astype(str)
    if subjects.shape != y.shape or recordings.shape != y.shape or not len(y):
        raise ValueError("Sampling IDs must align with the training labels")
    record_info = {}
    for c, subject, recording in zip(y, subjects, recordings):
        info = (int(c), subject)
        if recording in record_info and record_info[recording] != info:
            raise ValueError("Each recording must have one class and subject")
        record_info[recording] = info
    class_subjects, subject_recordings = defaultdict(set), defaultdict(set)
    for recording, (c, subject) in record_info.items():
        class_subjects[c].add(subject)
        subject_recordings[(c, subject)].add(recording)
    segment_counts = Counter(recordings)
    weights = np.asarray([
        1.0 / (len(class_subjects) * len(class_subjects[int(c)])
               * len(subject_recordings[(int(c), subject)]) * segment_counts[recording])
        for c, subject, recording in zip(y, subjects, recordings)
    ], dtype=np.float64)
    return weights / weights.sum()


def _split_identifiers(data_dir, split):
    root = Path(data_dir)
    subjects = np.load(root / f"{split}_patient_ids.npy", allow_pickle=False).astype(str)
    recordings = np.load(root / f"{split}_recording_ids.npy", allow_pickle=False).astype(str)
    with (root / "split_manifest.csv").open(encoding="utf-8", newline="") as f:
        manifest = {row["recording_id"]: row for row in csv.DictReader(f)}
    groups = []
    for subject, recording in zip(subjects, recordings):
        row = manifest.get(recording)
        if row is None or row["split"] != split or row["subject_id"] != subject:
            raise ValueError(f"Segment identifiers do not match the {split} manifest")
        groups.append(row["group_id"])
    return dict(subjects=subjects, recordings=recordings, groups=np.asarray(groups))


class ModelEMA:
    """EMA after every optimiser update; integer counters are copied exactly."""

    def __init__(self, model, decay=0.99):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.mul_(self.decay).add_(source[key].detach(), alpha=1 - self.decay)
            else:
                value.copy_(source[key])


class WarmupCosinePlateau:
    """A persistent plateau multiplier on the scheduled learning rate.

    The next cosine step cannot undo a plateau reduction. The LR is floored
    at min_learning_rate. Plateau reductions take effect on the next epoch.
    """

    def __init__(self, optimizer, training, mode="max"):
        self.optimizer, self.t, self.mode = optimizer, training, mode
        self.scale, self.stale = 1.0, 0
        self.best = -float("inf") if mode == "max" else float("inf")

    def start_epoch(self, epoch):
        t, base = self.t, self.t["learning_rate"]
        if t["scheduler"] == "constant":
            nominal = base
        elif epoch <= t["warmup_epochs"] and t["warmup_epochs"]:
            nominal = base * epoch / t["warmup_epochs"]
        else:
            progress = (epoch - t["warmup_epochs"]) / max(1, t["epochs"] - t["warmup_epochs"])
            progress = min(max(progress, 0), 1)
            nominal = t["min_learning_rate"] + (base - t["min_learning_rate"]) * (1 + math.cos(math.pi * progress)) / 2
        lr = max(t["min_learning_rate"], nominal * self.scale)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def observe(self, value):
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best, self.stale = float(value), 0
        else:
            self.stale += 1
        patience = self.t["plateau_patience"]
        if patience is not None and self.stale >= patience:
            self.scale *= self.t["plateau_factor"]
            self.stale = 0


class ProbabilityEnsemble(nn.Module):
    """Equal replica probability mean, then one frozen segment-wise class bias.

    Forward returns log probabilities, so ordinary ``softmax`` consumers work
    for both single-model and multi-replica checkpoints without separate paths.
    """

    def __init__(self, models, bias=None):
        super().__init__()
        if not models or len({m.num_classes for m in models}) != 1:
            raise ValueError("An ensemble needs models with equal class counts")
        self.models = nn.ModuleList(models)
        self.num_classes = models[0].num_classes
        bias = torch.zeros(self.num_classes) if bias is None else torch.as_tensor(bias, dtype=torch.float32)
        if bias.shape != (self.num_classes,) or not torch.isfinite(bias).all():
            raise ValueError("Invalid frozen class bias")
        self.register_buffer("class_bias", bias.clone())

    def forward(self, x):
        probabilities = torch.stack([m(x).softmax(1) for m in self.models]).mean(0)
        return torch.log_softmax(probabilities.clamp_min(1e-12).log() + self.class_bias, dim=1)


def _validate_training(t):
    for key in ["epochs", "min_epochs", "replicas", "batch_size", "grad_accum_steps", "patience"]:
        if not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if t["batch_size"] < 2 or t["num_workers"] < 0 or t["warmup_epochs"] < 0:
        raise ValueError("Use batch_size >= 2 and nonnegative workers/warmup")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or not 0 < t["min_learning_rate"] <= t["learning_rate"]:
        raise ValueError("Invalid learning rate or weight decay")
    if not 0 <= t["mixup_probability"] <= 1 or t["mixup_alpha"] < 0 or not 0 <= t["label_smoothing"] < 1:
        raise ValueError("Invalid Mixup or smoothing setting")
    if t["optimizer"] not in {"adamw", "adam"} or t["scheduler"] not in {"warmup_cosine", "constant"}:
        raise ValueError("Unsupported optimiser or LR schedule")
    if t["sampling"] not in {"shuffle", "class", "hierarchical"}:
        raise ValueError("sampling must be shuffle, class, or hierarchical")
    if t["selection"] not in {"uar", "balanced_nll"} or t["selection_unit"] not in {"segment", "recording"}:
        raise ValueError("Unsupported validation selection metric/unit")
    if t["selection"] == "uar" and t["selection_unit"] != "segment":
        raise ValueError("UAR checkpoint selection is implemented at segment level")
    if t["grad_clip_norm"] is not None and t["grad_clip_norm"] <= 0:
        raise ValueError("Clipping must be positive or null")
    if t["plateau_patience"] is not None and (t["plateau_patience"] < 1 or not 0 < t["plateau_factor"] < 1):
        raise ValueError("Invalid plateau settings")
    if t["class_weighting"] not in {"none", "inverse_frequency"}:
        raise ValueError("Unsupported class weighting")
    if t["loss"] not in {"smoothed_focal", "cross_entropy"}:
        raise ValueError("Unsupported training loss")
    if len(t["betas"]) != 2 or any(not 0 <= v < 1 for v in t["betas"]):
        raise ValueError("Adam betas must be in [0, 1)")


def _new_model(num_classes, kwargs, factory=None):
    model = (factory or MWAFNet)(num_classes=num_classes, **kwargs)
    if not hasattr(model, "num_classes"):
        model.num_classes = num_classes
    if model.num_classes != num_classes:
        raise ValueError("Model factory returned the wrong output dimension")
    return model


def _checkpoint_meta(model, config, meta):
    return dict(
        checkpoint_format=CHECKPOINT_FORMAT,
        architecture=model.__class__.__name__,
        model_config_version=MODEL_CONFIG_VERSION if isinstance(model, MWAFNet) else "custom-factory-v1",
        model_config=model.configuration() if hasattr(model, "configuration") else config["model"],
        num_classes=len(meta["class_names"]), task=meta["task"], class_names=meta["class_names"],
        config=config, dataset_hashes=meta.get("files", {}),
        preprocessing={k: meta[k] for k in ["target_sr", "segment_seconds", "min_seconds", "global_normalization"]},
        inference=dict(replica_aggregation="equal mean of probabilities", prediction_unit="segment",
                       calibration_applied="class bias after replica probability averaging"),
    )


def _train_replica(config, arrays, ids, meta, output_dir, seed, device, model_factory):
    t = config["training"]
    seed_all(seed)
    x, y = arrays["train"]
    nclasses = len(meta["class_names"])
    generator = torch.Generator().manual_seed(seed)
    num_samples = len(y)
    if t["sampling"] == "hierarchical":
        per_recording = t["segments_per_recording"]
        if not isinstance(per_recording, int) or per_recording < 1:
            raise ValueError("Hierarchical sampling requires positive integer segments_per_recording")
        num_samples = per_recording * len(np.unique(ids["train"]["recordings"]))
        weights = hierarchical_sampling_weights(y, ids["train"]["subjects"], ids["train"]["recordings"])
    elif t["sampling"] == "class":
        weights = inverse_frequency_alpha(y, nclasses)
        weights = weights[y]
    else:
        weights = None
    sampler = None if weights is None else WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples,
        replacement=True, generator=generator
    )
    drop_singleton = num_samples % t["batch_size"] == 1
    loader = DataLoader(Segments(x, y, t["train_augmentation"]), batch_size=t["batch_size"],
                        sampler=sampler, shuffle=sampler is None, num_workers=t["num_workers"],
                        generator=generator, worker_init_fn=_seed_worker,
                        drop_last=drop_singleton, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Segments(*arrays["val"]), batch_size=t["batch_size"],
                            shuffle=False, num_workers=t["num_workers"])
    model = _new_model(nclasses, config["model"], model_factory).to(device)
    alpha = inverse_frequency_alpha(y, nclasses) if t["class_weighting"] == "inverse_frequency" else None
    if t["loss"] == "smoothed_focal":
        criterion = FocalLoss(nclasses, gamma=t["gamma"], alpha=alpha,
                              label_smoothing=t["label_smoothing"]).to(device)
    else:
        criterion = SampleMeanCrossEntropy(nclasses, alpha=alpha,
                                           label_smoothing=t["label_smoothing"]).to(device)
    optimizer_cls = torch.optim.AdamW if t["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
                               betas=tuple(t["betas"]))
    mode = "max" if t["selection"] == "uar" else "min"
    schedule = WarmupCosinePlateau(optimizer, t, mode)
    ema = None if t["ema_decay"] is None else ModelEMA(model, t["ema_decay"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta = _checkpoint_meta(model, config, meta)
    effective = dict(seed=seed, training_class_counts=np.bincount(y, minlength=nclasses).tolist(),
                     alpha=None if alpha is None else alpha.tolist(),
                     alpha_normalisation="training-sample mean is one" if alpha is not None else None,
                     nominal_samples_per_epoch=num_samples,
                     actual_samples_per_epoch=num_samples - int(drop_singleton),
                     singleton_remainder_dropped=drop_singleton,
                     selection=f"validation {t['selection_unit']} {t['selection']}",
                     ema_used_for_selection=ema is not None,
                     model_config=checkpoint_meta["model_config"])
    (output_dir / "effective_run.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
    best = -float("inf") if mode == "max" else float("inf")
    stale = 0
    for epoch in range(1, t["epochs"] + 1):
        learning_rate = schedule.start_epoch(epoch)
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if t["mixup_alpha"] > 0 and np.random.random() < t["mixup_probability"]:
                mixed, a, b, lam = mixup_data(batch_x, batch_y, t["mixup_alpha"])
                loss = mixup_criterion(criterion, model(mixed), a, b, lam)
            else:
                loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            # Weight each mini-batch by its actual sample count in this update.
            # The final partial accumulation window must not be underweighted.
            accum = t["grad_accum_steps"]
            start_batch = (i // accum) * accum
            end_batch = min(start_batch + accum, len(loader))
            samples_this_window = min(end_batch * t["batch_size"], num_samples - int(drop_singleton)) - start_batch * t["batch_size"]
            (loss * len(batch_y) / samples_this_window).backward()
            total_loss += float(loss.detach()) * len(batch_y)
            seen += len(batch_y)
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                if t["grad_clip_norm"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
        selected_model = ema.model if ema is not None else model
        probabilities, val_y, val_loss = predict_loader(selected_model, val_loader, device, criterion)
        result = metrics(val_y, probabilities.argmax(1), nclasses)
        result["loss"] = val_loss
        nll = selection_nll(probabilities, val_y, ids["val"]["recordings"], t["selection_unit"])
        result["selection_class_balanced_nll"] = nll
        score = result["uar_percent"] if mode == "max" else nll
        row = dict(epoch=epoch, train_loss=total_loss / seen, validation=result,
                   learning_rate=learning_rate, plateau_multiplier=schedule.scale)
        with (output_dir / "training_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"seed {seed} epoch {epoch}/{t['epochs']} loss={total_loss / seen:.5f} "
              f"val_UAR={result['uar_percent']:.2f}% val_{t['selection_unit']}_NLL={nll:.5f}", flush=True)
        better = score > best if mode == "max" else score < best
        if better:
            best, stale = float(score), 0
            member = dict(state_dict={k: v.detach().cpu().clone() for k, v in selected_model.state_dict().items()},
                          seed=int(seed), epoch=epoch, selection_value=float(score),
                          selection_metric=t["selection"], selection_unit=t["selection_unit"],
                          ema=ema is not None)
            torch.save({**checkpoint_meta, "members": [member], "class_bias": [0.0] * nclasses,
                        "calibration": {"status": "not_fitted_replica_checkpoint"}}, output_dir / "best.pt")
        else:
            stale += 1
        schedule.observe(score)
        if epoch >= t["min_epochs"] and stale >= t["patience"]:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["members"][0]["state_dict"], strict=True)
    probabilities, _, _ = predict_loader(model, val_loader, device)
    return checkpoint, probabilities


def train(config, data_dir, output_dir, seed=42, device="auto", model_factory=None):
    """Train one protocol with MWAF-Net or a supplied comparable model factory.

    A custom factory accepts ``num_classes`` and ``config['model']`` keyword
    arguments and returns logits. The same protocol code then handles sampling,
    EMA, checkpoints, replicas and validation-only calibration for that model.
    """
    config = resolve_config(config)
    t = config["training"]
    _validate_training(t)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new/empty training output directory")
    device = get_device(device)
    arrays, meta = load_prepared(data_dir, config["task"])
    # Loading verifies immutable dataset hashes and partition integrity, but no
    # test predictions or labels enter training, model selection or calibration.
    del arrays["test"]
    if len(arrays["train"][1]) < 2:
        raise ValueError("BatchNorm training requires at least two samples")
    ids = {split: _split_identifiers(data_dir, split) for split in ["train", "val"]}
    if set(ids["train"]["groups"]) & set(ids["val"]["groups"]):
        raise ValueError("Patient/duplicate-audio group leakage between train and val")
    seeds = [int(seed) + i for i in range(t["replicas"])]
    output_dir.mkdir(parents=True, exist_ok=True)
    effective = dict(config=config, seed=int(seed), replica_seeds=seeds,
                     replica_seed_convention="base seed + zero-based replica index",
                     class_names=meta["class_names"], split_seed=meta.get("seed"),
                     dataset_hashes=meta.get("files", {}),
                     model_config_version=MODEL_CONFIG_VERSION if model_factory is None else "custom-factory-v1",
                     torch_version=str(torch.__version__),
                     scope="public manuscript-aligned implementation; historical scores/weights not regenerated",
                     calibration_split="val" if config["calibration"]["enabled"] else None)
    (output_dir / "run_config.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
    checkpoints, val_predictions = [], []
    for i, replica_seed in enumerate(seeds):
        checkpoint, probabilities = _train_replica(config, arrays, ids, meta,
                                                   output_dir / f"replica_{i + 1}", replica_seed,
                                                   device, model_factory)
        checkpoints.append(checkpoint)
        val_predictions.append(probabilities)
    combined = np.mean(val_predictions, axis=0)
    c = config["calibration"]
    if c["enabled"]:
        bias, calibration = fit_validation_bias(
            combined, arrays["val"][1], ids["val"]["groups"], ids["val"]["recordings"],
            source_split="val", unit=t["selection_unit"], folds=c["folds"],
            penalties=c["penalties"], bound=c["bound"], seed=int(seed),
        )
    else:
        bias, calibration = np.zeros(len(meta["class_names"])), dict(status="disabled", bias=[0.0] * len(meta["class_names"]))
    final = {**checkpoints[0], "members": [ck["members"][0] for ck in checkpoints],
             "class_bias": bias.tolist(), "calibration": calibration,
             "replica_seeds": seeds}
    torch.save(final, output_dir / "best.pt")
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return output_dir / "best.pt"


def load_checkpoint(path, device="auto", model_factory=None):
    device = get_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("This is an archived/incompatible checkpoint. The updated architecture requires "
                         "retraining with the revised code; old weights cannot be silently converted.")
    if checkpoint["task"] not in TASKS or checkpoint["class_names"] != TASKS[checkpoint["task"]]:
        raise ValueError("Checkpoint task/class mapping is inconsistent")
    if checkpoint["num_classes"] != len(checkpoint["class_names"]):
        raise ValueError("Checkpoint output dimension is inconsistent")
    if model_factory is None and (checkpoint["architecture"] != "MWAFNet" or checkpoint["model_config_version"] != MODEL_CONFIG_VERSION):
        raise ValueError("Checkpoint architecture differs; supply its matching model_factory or retrain MWAF-Net")
    if not checkpoint.get("members"):
        raise ValueError("Checkpoint has no trained members")
    models = []
    for member in checkpoint["members"]:
        model = _new_model(checkpoint["num_classes"], checkpoint["model_config"], model_factory)
        try:
            model.load_state_dict(member["state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError("Checkpoint weights do not match the recorded model configuration; no partial load is allowed") from exc
        models.append(model)
    model = ProbabilityEnsemble(models, checkpoint["class_bias"]).to(device).eval()
    return model, checkpoint, device
